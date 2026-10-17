use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::Module;

use crate::clif_decode::Resolved;

/// What a host calls a function as: the input, its length, the output, its
/// length, answering the status. The arena is not an argument: a function
/// takes its address itself, from the symbol [`ARENA`] the engine links it to.
pub type FunctionPtr = unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i64;

/// The symbol the arena is linked under: what `symbol_value` of
/// [`base_types::clif::Global::Arena`] answers.
pub(crate) const ARENA: &str = "arena";

/// A compiled program: its code, and each named function as the C function a
/// host calls.
pub(crate) struct Program {
    pub(crate) module: cranelift_jit::JITModule,
    pub(crate) exports: std::collections::HashMap<String, FunctionPtr>,
}

/// Every symbol a program links to, registered under its link name.
fn register_symbols(builder: &mut JITBuilder, symbols: &[(String, usize)]) {
    for (name, addr) in symbols {
        builder.symbol(name.clone(), *addr as *const u8);
    }
}

/// Address space reserved for a program's code, however little it compiles to.
const CODE_RESERVE_MIN: usize = 16 << 20;

/// Reserved per instruction on top of that. The densest shipped program
/// (`vit_block`) compiles to about 4 bytes per instruction, so this leaves
/// room for constants, alignment and a far worse ratio.
const CODE_RESERVE_PER_INST: usize = 64;

/// The most that is ever reserved: AArch64's direct call reaches ±128 MiB, so a
/// region no larger than that keeps every call within the module in range.
const CODE_RESERVE_MAX: usize = 128 << 20;

/// The host's ISA, as every program is compiled for it: optimized for speed,
/// frames probed.
pub(crate) fn isa() -> cranelift_codegen::isa::OwnedTargetIsa {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    // A program runs on its caller's stack. A frame larger than the guard
    // page could otherwise step past it and write whatever is mapped below,
    // where probing it first faults on the guard: an overflow ends the
    // process, as the C API says, and frames under a page are not probed.
    flag_builder.set("enable_probestack", "true").unwrap();
    flag_builder.set("probestack_strategy", "inline").unwrap();
    let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
    isa_builder.finish(settings::Flags::new(flag_builder)).unwrap()
}

/// The JIT module every compilation starts from: host ISA, speed, all FFI
/// symbols registered, and code placed in one contiguous region.
///
/// The region is what makes a call between a program's own functions safe to
/// emit PC-relative. Allocated separately, two functions can land further
/// apart than a direct call reaches — ±128 MiB on AArch64, ±2 GiB on x86-64
/// and RISC-V — with the program's data, or a pinned host buffer, mapped in
/// between.
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

/// Each named function by its name, and its position.
///
/// A name given twice is refused: which function a host reached would
/// depend on the order a map was filled. That each takes what [`FunctionPtr`]
/// passes is not checked here: a named function is declared at that
/// signature (`signature_of`), and the verifier refuses a body that is not.
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

/// Compiles the functions an artifact carries.
///
/// Its primitives are declared first, each once, as the module's imports, and
/// its functions after, so a call resolves by position to either.
pub(crate) fn compile(
    primitives: &[base_types::clif::Primitive],
    functions: &[base_types::clif::Function],
    arena: *mut u8,
) -> Result<Program, String> {
    let exports = exports_of(functions)?;
    let insts = functions
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len())
        .sum();
    // The C functions the program calls, bound before the module is built so
    // each resolves like any other import. A primitive is declared once: a
    // second declaration of the same symbol is the same fact stated twice.
    let mut symbols: Vec<(String, usize)> = Vec::with_capacity(primitives.len() + 1);
    for p in primitives {
        let name = crate::libs::link_name(p);
        if symbols.iter().any(|(n, _)| *n == name) {
            return Err(format!("the primitive {}!{} is declared twice", p.lib, p.symbol));
        }
        symbols.push((name, crate::libs::bind(p)));
    }
    symbols.push((ARENA.into(), arena as usize));
    let mut module = new_module(insts, &symbols)?;
    let cc = module.isa().default_call_conv();
    let arena_data = module
        .declare_data(ARENA, cranelift_module::Linkage::Import, true, false)
        .map_err(|e| format!("declaring {ARENA}: {e}"))?;

    let mut prims = Vec::with_capacity(primitives.len());
    for p in primitives {
        let sig = crate::clif_decode::signature(&p.signature, cc)
            .map_err(|e| format!("the primitive {}!{}: {e}", p.lib, p.symbol))?;
        let name = crate::libs::link_name(p);
        let id = module
            .declare_function(&name, cranelift_module::Linkage::Import, &sig)
            .map_err(|e| format!("declaring {name}: {e}"))?;
        prims.push((id, sig));
    }
    let sigs = functions
        .iter()
        .enumerate()
        .map(|(i, f)| crate::clif_decode::signature_of(f, i, cc))
        .collect::<Result<Vec<_>, String>>()?;
    let mut func_ids = Vec::with_capacity(functions.len());
    for (i, sig) in sigs.iter().enumerate() {
        func_ids.push(
            module
                .declare_function(&format!("fn_{i}"), cranelift_module::Linkage::Local, sig)
                .map_err(|e| format!("declaring fn_{i}: {e}"))?,
        );
    }

    let mut decoded = Vec::with_capacity(functions.len());
    for (i, f) in functions.iter().enumerate() {
        let mut resolve = |callee: base_types::clif::Callee| match callee {
            base_types::clif::Callee::Function(n) => func_ids
                .get(n as usize)
                .zip(sigs.get(n as usize))
                .map(|(id, sig)| (Resolved { id: id.as_u32(), colocated: true }, sig.clone()))
                .ok_or_else(|| format!("a call to u0:{n}, which the program does not define")),
            base_types::clif::Callee::Primitive(n) => prims
                .get(n as usize)
                .map(|(id, sig)| (Resolved { id: id.as_u32(), colocated: false }, sig.clone()))
                .ok_or_else(|| format!("a call to primitive {n}, which the program does not declare")),
        };
        let mut m = crate::clif_decode::Module { primitives, arena: arena_data.as_u32(), resolve: &mut resolve };
        decoded.push(crate::clif_decode::decode_function(f, i, cc, &mut m)?);
    }

    for (i, func) in decoded.into_iter().enumerate() {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        let exported = match &functions[i].name {
            Some(name) => format!(
                " (named {name:?}, so a host calls it: it takes the input and output \
                 buffers and answers a status)"
            ),
            None => String::new(),
        };
        module
            .define_function(func_ids[i], &mut ctx)
            // Debug rather than Display: a verifier rejection names the
            // offending instructions only in the former.
            .map_err(|e| format!("compiling u0:{i}{exported}: {e:?}"))?;
    }

    module.finalize_definitions().map_err(|e| format!("{e}"))?;
    let exports = exports
        .into_iter()
        // SAFETY: each was declared at `FunctionPtr`'s signature and its body
        // verified against it, and is mapped for as long as the module is.
        .map(|(name, k)| {
            let addr = module.get_finalized_function(func_ids[k]);
            (name, unsafe { std::mem::transmute::<*const u8, FunctionPtr>(addr) })
        })
        .collect();
    Ok(Program { module, exports })
}

