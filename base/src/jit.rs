use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::{Linkage, Module};


/// A function a host calls: input, its length, output, its length; a status.
pub type BaseFunction = unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i64;

/// The symbol the arena is linked under.
pub(crate) const ARENA: &str = "arena";

/// A compiled program and its named functions.
pub(crate) struct Program {
    pub(crate) module: cranelift_jit::JITModule,
    pub(crate) exports: std::collections::HashMap<String, BaseFunction>,
}

/// Every symbol a program links to, under its link name.
fn register_symbols(builder: &mut JITBuilder, symbols: &[(String, usize)]) {
    for (name, addr) in symbols {
        builder.symbol(name.clone(), *addr as *const u8);
    }
}

/// Code space reserved for any program.
const CODE_RESERVE_MIN: usize = 16 << 20;

/// And per instruction: generous against the ~4 bytes the densest program needs.
const CODE_RESERVE_PER_INST: usize = 64;

/// At most AArch64's ±128 MiB direct-call reach.
const CODE_RESERVE_MAX: usize = 128 << 20;

/// The host's ISA, optimized for speed, frames probed.
pub(crate) fn isa() -> cranelift_codegen::isa::OwnedTargetIsa {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    // A frame larger than the guard page faults on it instead of stepping past.
    flag_builder.set("enable_probestack", "true").unwrap();
    flag_builder.set("probestack_strategy", "inline").unwrap();
    let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
    isa_builder.finish(settings::Flags::new(flag_builder)).unwrap()
}

/// A JIT module with `symbols` registered and its code in one region, so calls
/// between a program's functions stay within a direct call's reach.
fn new_module(
    insts: usize,
    symbols: &[(String, usize)],
) -> Result<cranelift_jit::JITModule, String> {
    let mut builder = JITBuilder::with_isa(isa(), cranelift_module::default_libcall_names());
    register_symbols(&mut builder, symbols);
    let reserve = CODE_RESERVE_MIN
        .saturating_add(insts.saturating_mul(CODE_RESERVE_PER_INST))
        .min(CODE_RESERVE_MAX);
    let region = cranelift_jit::ArenaMemoryProvider::new_with_size(reserve)
        .map_err(|e| format!("reserving {reserve} bytes for code: {e}"))?;
    builder.memory_provider(Box::new(region));
    Ok(cranelift_jit::JITModule::new(builder))
}

/// Each named function and its position; a name given twice is refused.
pub(crate) fn exports_of(functions: &[base_types::clif::Function]) -> Result<Vec<(String, usize)>, String> {
    let mut exports: Vec<(String, usize)> = Vec::new();
    let mut named: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, f) in functions.iter().enumerate() {
        let Some(name) = &f.name else { continue };
        if let Some(prev) = named.insert(name, i) {
            return Err(format!("u0:{prev} and u0:{i} are both named {name:?}"));
        }
        exports.push((name.clone(), i));
    }
    Ok(exports)
}

/// Compiles the functions an artifact carries, its primitives declared first
/// so each `FuncId` is the one `clif_decode::decode` named it by.
pub(crate) fn compile(
    primitives: &[base_types::clif::Primitive],
    functions: &[base_types::clif::Function],
    arena: *mut u8,
) -> Result<Program, String> {
    let exports = exports_of(functions)?;
    let decoded = crate::clif_decode::decode(primitives, functions, isa().default_call_conv())?;
    let insts = functions.iter().flat_map(|f| &f.blocks).map(|b| b.insts.len()).sum();
    let mut symbols: Vec<(String, usize)> = Vec::with_capacity(primitives.len() + 1);
    for p in primitives {
        let name = crate::link::link_name(p);
        if symbols.iter().any(|(n, _)| *n == name) {
            return Err(format!("the primitive {}!{} is declared twice", p.lib, p.symbol));
        }
        symbols.push((name, crate::link::bind(p)));
    }
    symbols.push((ARENA.into(), arena as usize));
    let mut module = new_module(insts, &symbols)?;
    let declared = |what: &str, id: u32, at: usize| {
        if id as usize == at { Ok(()) } else { Err(format!("{what} was declared as {id}, not {at}")) }
    };
    let data = module.declare_data(ARENA, Linkage::Import, true, false).map_err(|e| format!("declaring {ARENA}: {e}"))?;
    declared("the arena", data.as_u32(), 0)?;
    for (k, (p, sig)) in primitives.iter().zip(&decoded.primitive_sigs).enumerate() {
        let name = crate::link::link_name(p);
        let id = module.declare_function(&name, Linkage::Import, sig).map_err(|e| format!("declaring {name}: {e}"))?;
        declared(&name, id.as_u32(), k)?;
    }
    let mut ids = Vec::with_capacity(functions.len());
    for (i, sig) in decoded.function_sigs.iter().enumerate() {
        let id = module.declare_function(&format!("fn_{i}"), Linkage::Local, sig).map_err(|e| format!("declaring fn_{i}: {e}"))?;
        declared(&format!("u0:{i}"), id.as_u32(), primitives.len() + i)?;
        ids.push(id);
    }
    for (i, func) in decoded.functions.into_iter().enumerate() {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        let named = match &functions[i].name {
            Some(name) => format!(" (named {name:?}, so a host calls it: it takes the input and output buffers and answers a status)"),
            None => String::new(),
        };
        // Debug, not Display: only it names the instructions a verifier refuses.
        module.define_function(ids[i], &mut ctx).map_err(|e| format!("compiling u0:{i}{named}: {e:?}"))?;
    }

    module.finalize_definitions().map_err(|e| format!("{e}"))?;
    let exports = exports
        .into_iter()
        // SAFETY: each was declared at `BaseFunction`'s signature and its body
        // verified against it, and is mapped for as long as the module is.
        .map(|(name, k)| {
            let addr = module.get_finalized_function(ids[k]);
            (name, unsafe { std::mem::transmute::<*const u8, BaseFunction>(addr) })
        })
        .collect();
    Ok(Program { module, exports })
}

