use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::Module;

use crate::clif_decode::Resolved;
use std::sync::Arc;
use tracing::info;

/// One compiled function: where it starts, how many arguments it takes, and
/// whether it answers a status.
///
/// Deliberately not an `fn` pointer: the shape is the one the function's own
/// body declared (see `clif_decode::signature_of`), so it travels with the
/// address and a caller checks it before transmuting rather than assuming.
///
/// A raw pointer is neither `Send` nor `Sync`, and the table crosses into
/// spawned threads. It is code, mapped for the life of the module and never
/// written, so sharing the address is sound; that is what these assert.
#[derive(Copy, Clone)]
pub(crate) struct Compiled {
    pub(crate) addr: *const u8,
    pub(crate) arity: usize,
    pub(crate) answers: bool,
}

unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

thread_local! {
    pub(crate) static THREAD_COMPILED_FNS: std::cell::RefCell<Option<Arc<Vec<Compiled>>>> = const { std::cell::RefCell::new(None) };
}

/// Every function a program may import, registered under its name.
fn register_symbols(builder: &mut JITBuilder) {
    for import in crate::imports::imports() {
        builder.symbol(import.name, import.addr as *const u8);
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

/// The JIT module every compilation starts from: host ISA, speed, all FFI
/// symbols registered, and code placed in one contiguous region.
///
/// The region is what makes a call between a program's own functions safe to
/// emit PC-relative. Allocated separately, two functions can land further
/// apart than a direct call reaches — ±128 MiB on AArch64, ±2 GiB on x86-64
/// and RISC-V — with the program's data, or a pinned host buffer, mapped in
/// between.
fn new_module(insts: usize) -> Result<cranelift_jit::JITModule, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_symbols(&mut builder);
    let reserve = CODE_RESERVE_MIN
        .saturating_add(insts.saturating_mul(CODE_RESERVE_PER_INST))
        .min(CODE_RESERVE_MAX);
    let region = cranelift_jit::ArenaMemoryProvider::new_with_size(reserve)
        .map_err(|e| format!("reserving {reserve} bytes for code: {e}"))?;
    builder.memory_provider(Box::new(region));
    Ok(cranelift_jit::JITModule::new(builder))
}

/// Finalizes a module whose functions have all been defined, and hands back
/// pointers to them.
fn finalize(
    mut module: cranelift_jit::JITModule,
    func_ids: Vec<cranelift_module::FuncId>,
    shapes: Vec<(usize, bool)>,
) -> Result<(cranelift_jit::JITModule, Arc<Vec<Compiled>>), String> {
    module.finalize_definitions().map_err(|e| format!("{e}"))?;
    let compiled_fns: Vec<Compiled> = func_ids
        .iter()
        .zip(shapes)
        .map(|(&id, (arity, answers))| Compiled {
            addr: module.get_finalized_function(id),
            arity,
            answers,
        })
        .collect();
    info!(count = compiled_fns.len(), "CLIF compiled successfully");
    Ok((module, Arc::new(compiled_fns)))
}

/// Compiles the functions an artifact carries.
///
/// A call carries its callee, so the prologue Cranelift wants is interned from
/// the body as it is built: there is no name to rewrite afterward and no
/// declaration order for anything to depend on.
pub(crate) fn compile(
    functions: &[base_types::clif::Function],
) -> Result<(cranelift_jit::JITModule, Arc<Vec<Compiled>>), String> {
    info!(functions = functions.len(), "compiling CLIF functions");
    let insts = functions
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len())
        .sum();
    let mut module = new_module(insts)?;
    let cc = module.isa().default_call_conv();

    // Declared before any body is built, so `u0:N` resolves to FuncId(N).
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
    let shapes = sigs.iter().map(|s| (s.params.len(), !s.returns.is_empty())).collect();

    let mut decoded = Vec::with_capacity(functions.len());
    for (i, f) in functions.iter().enumerate() {
        let mut declare = |callee: &base_types::clif::Callee| match callee {
            // Only what base provides, at the signature it provides it at. A
            // name outside the table would otherwise be looked up in whatever
            // the process has loaded, and be called with arguments in the
            // wrong registers.
            base_types::clif::Callee::Import(name) => {
                let import = crate::imports::lookup(name).ok_or_else(|| {
                    format!("u0:{i} imports {name}, which base does not provide")
                })?;
                let sig = crate::clif_decode::signature(&import.params, import.result, cc);
                let id = module
                    .declare_function(name, cranelift_module::Linkage::Import, &sig)
                    .map_err(|e| format!("declaring import {name}: {e}"))?;
                Ok((Resolved { id: id.as_u32(), colocated: false }, sig))
            }
            // A local callee's signature is the one its own body declares,
            // read once here and used for both the call site and the
            // definition, so the two cannot disagree.
            base_types::clif::Callee::Local(n) => {
                let (id, sig) = func_ids
                    .get(*n as usize)
                    .zip(sigs.get(*n as usize))
                    .ok_or_else(|| format!("call to u0:{n}, which the program does not define"))?;
                Ok((Resolved { id: id.as_u32(), colocated: true }, sig.clone()))
            }
            // Called through its address; `decode_function` never resolves it.
            base_types::clif::Callee::Native => Err("machine code has no symbol to resolve".into()),
        };
        decoded.push(crate::clif_decode::decode_function(f, i, cc, &mut declare)?);
    }

    let dump = std::env::var("BASE_DISASM").is_ok();
    // The decoded function in Cranelift's own textual form. The artifact
    // carries instruction records, not text, so this is the only way to read
    // what was handed to the backend rather than what came out of it.
    let dump_clif = std::env::var("BASE_DUMP_CLIF").is_ok();
    for (i, func) in decoded.into_iter().enumerate() {
        if dump_clif {
            eprintln!("=== clif fn {i} ===\n{}", func.display());
        }
        let mut ctx = cranelift_codegen::Context::for_function(func);
        if dump {
            ctx.set_disasm(true);
        }
        module
            .define_function(func_ids[i], &mut ctx)
            // Debug rather than Display: a verifier rejection names the
            // offending instructions only in the former.
            .map_err(|e| format!("compiling u0:{i}: {e:?}"))?;
        if dump {
            if let Some(vc) = ctx.compiled_code().and_then(|c| c.vcode.as_deref()) {
                eprintln!("=== fn {i} ===\n{vc}");
            }
        }
    }

    finalize(module, func_ids, shapes)
}

