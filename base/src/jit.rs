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


/// A JIT module with `symbols` registered and its code in one region, so calls
/// between a program's functions stay within a direct call's reach.
fn new_module(
    insts: usize,
    symbols: &[(String, usize)],
) -> Result<cranelift_jit::JITModule, String> {
    let mut builder = JITBuilder::with_isa(crate::clif_decode::isa(), cranelift_module::default_libcall_names());
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

/// A function's machine code and the references in it still to be linked.
struct Code {
    alignment: u64,
    bytes: Vec<u8>,
    relocs: Vec<cranelift_module::ModuleReloc>,
}

/// Function `i`, decoded and compiled on its own.
fn compile_one(
    isa: &dyn cranelift_codegen::isa::TargetIsa,
    decoded: &crate::clif_decode::Module,
    f: &base_types::clif::Function,
    i: usize,
    id: cranelift_module::FuncId,
) -> Result<Code, String> {
    let mut ctx = cranelift_codegen::Context::for_function(decoded.function(f, i)?);
    // Debug, not Display: only it names the instructions a verifier refuses.
    let compiled = ctx
        .compile(isa, &mut cranelift_codegen::control::ControlPlane::default())
        .map_err(|e| format!("{:?}", cranelift_module::ModuleError::from(e)))?;
    let alignment = compiled.buffer.alignment as u64;
    let compiled = ctx.compiled_code().expect("compiled just above");
    let relocs = compiled
        .buffer
        .relocs()
        .iter()
        .map(|r| cranelift_module::ModuleReloc::from_mach_reloc(r, &ctx.func, id))
        .collect();
    Ok(Code { alignment, bytes: compiled.code_buffer().to_vec(), relocs })
}

/// Every function's code, in order. Functions compile independently, so they
/// compile on as many threads as the process may run, largest first, the
/// calling thread among them; the threads end before this returns. Where no
/// thread can start, the calling thread compiles them all.
fn compile_all(
    decoded: &crate::clif_decode::Module,
    functions: &[base_types::clif::Function],
    ids: &[cranelift_module::FuncId],
) -> Vec<Result<Code, String>> {
    let isa = crate::clif_decode::isa();
    let mut order: Vec<usize> = (0..functions.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(functions[i].blocks.iter().map(|b| b.insts.len()).sum::<usize>()));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(functions.len()).max(1);
    let work = || {
        let mut done = Vec::new();
        loop {
            let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&i) = order.get(k) else { return done };
            done.push((i, compile_one(&*isa, decoded, &functions[i], i, ids[i])));
        }
    };
    let mut slots: Vec<Option<Result<Code, String>>> = (0..functions.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        // A thread the system will not start leaves its share to the others.
        let helpers: Vec<_> =
            (1..threads).filter_map(|_| std::thread::Builder::new().spawn_scoped(s, work).ok()).collect();
        let mine = work();
        for (i, code) in helpers.into_iter().flat_map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))).chain(mine) {
            slots[i] = Some(code);
        }
    });
    slots.into_iter().map(|c| c.expect("every function was taken")).collect()
}

/// Compiles the functions an artifact carries, its primitives declared first
/// so each `FuncId` is the one `clif_decode::decode` named it by.
pub(crate) fn compile(
    primitives: &[base_types::clif::Primitive],
    functions: &[base_types::clif::Function],
    arena: *mut u8,
) -> Result<Program, String> {
    let exports = exports_of(functions)?;
    let decoded = crate::clif_decode::decode(primitives, functions, crate::clif_decode::isa().default_call_conv())?;
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
    for (i, code) in compile_all(&decoded, functions, &ids).into_iter().enumerate() {
        let code = code.map_err(|e| {
            let named = match &functions[i].name {
                Some(name) => format!(" (named {name:?}, so a host calls it: it takes the input and output buffers and answers a status)"),
                None => String::new(),
            };
            format!("compiling u0:{i}{named}: {e}")
        })?;
        module
            .define_function_bytes(ids[i], code.alignment, &code.bytes, &code.relocs)
            .map_err(|e| format!("placing u0:{i}: {e:?}"))?;
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

