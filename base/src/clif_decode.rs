//! Builds `cranelift_codegen::ir::Function` from the program an artifact
//! carries.
//!
//! The text path this replaces went through `cranelift-reader`, a crate meant
//! for Cranelift's own filetests. Two of its behaviours had to be worked around
//! rather than used: `%name` parses to `ExternalName::TestCase`, so every
//! callee had to be rewritten after the fact, and `u0:N` resolves by treating
//! the parsed index as a `FuncId`, so declaration order had to accidentally
//! match. Constructing the function directly makes both questions disappear —
//! callees are declared as they are built, and indices are assigned rather than
//! recovered.

use base_types::clif;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{
    self, AbiParam, ExtFuncData, ExternalName, InstBuilder, MemFlags, Signature, UserExternalName,
    UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use std::collections::HashMap;

fn ty(t: clif::ClifTy) -> ir::Type {
    match t {
        clif::ClifTy::I8 => ir::types::I8,
        clif::ClifTy::I16 => ir::types::I16,
        clif::ClifTy::I32 => ir::types::I32,
        clif::ClifTy::I64 => ir::types::I64,
        clif::ClifTy::F32 => ir::types::F32,
        clif::ClifTy::F64 => ir::types::F64,
        clif::ClifTy::F32x4 => ir::types::F32X4,
        clif::ClifTy::I8x16 => ir::types::I8X16,
    }
}

fn int_cc(c: clif::IntCC) -> ir::condcodes::IntCC {
    use ir::condcodes::IntCC as C;
    match c {
        clif::IntCC::Eq => C::Equal,
        clif::IntCC::Ne => C::NotEqual,
        clif::IntCC::Uge => C::UnsignedGreaterThanOrEqual,
        clif::IntCC::Ugt => C::UnsignedGreaterThan,
        clif::IntCC::Ule => C::UnsignedLessThanOrEqual,
        clif::IntCC::Ult => C::UnsignedLessThan,
        clif::IntCC::Slt => C::SignedLessThan,
        clif::IntCC::Sle => C::SignedLessThanOrEqual,
        clif::IntCC::Sgt => C::SignedGreaterThan,
        clif::IntCC::Sge => C::SignedGreaterThanOrEqual,
    }
}

fn float_cc(c: clif::FloatCC) -> ir::condcodes::FloatCC {
    use ir::condcodes::FloatCC as C;
    match c {
        clif::FloatCC::Eq => C::Equal,
        clif::FloatCC::Ne => C::NotEqual,
        clif::FloatCC::Lt => C::LessThan,
        clif::FloatCC::Le => C::LessThanOrEqual,
        clif::FloatCC::Gt => C::GreaterThan,
        clif::FloatCC::Ge => C::GreaterThanOrEqual,
    }
}

/// The value each `Val` names, and the error when a program names one it never
/// defined — a generator bug, and one the text path reported as a parse error.
///
/// Indexed rather than hashed: generators number values densely from zero, so
/// the id is the position.
#[derive(Default)]
struct Vals(Vec<Option<ir::Value>>);

impl Vals {
    fn get(&self, v: clif::Val) -> Result<ir::Value, String> {
        self.0
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("v{} used before it is defined", v.0))
    }
    fn get_all(&self, vs: &[clif::Val]) -> Result<Vec<ir::Value>, String> {
        vs.iter().map(|v| self.get(*v)).collect()
    }
    fn set(&mut self, v: clif::Val, val: ir::Value) {
        let i = v.0 as usize;
        if i >= self.0.len() {
            self.0.resize(i + 1, None);
        }
        self.0[i] = Some(val);
    }
}

/// Branch destinations take `BlockArg`, which is wider than a value — it also
/// spans the exception-table forms this fragment does not use.
fn block_args(vs: &[ir::Value]) -> Vec<ir::BlockArg> {
    vs.iter().map(|v| ir::BlockArg::Value(*v)).collect()
}

/// A function's signature, read off its body.
///
/// The entry block is `blocks[0]` and its parameters are exactly the values a
/// caller supplies; a `Ret` carrying a value is a function that answers an
/// `i64`. So the signature is not a second fact that has to be kept in
/// agreement with the body.
///
/// Declaring a function and defining it are separate calls into Cranelift and
/// both need this. Reading it from the same place twice is what makes them
/// agree.
///
/// `cc` is the host's C convention, never a fixed one: the artifact names no
/// convention, and the other side of every boundary — the host calling an
/// entry point, a program calling the FFI — is Rust `extern "C"` compiled for
/// the machine running it.
pub fn signature_of(f: &clif::Function, index: usize, cc: CallConv) -> Result<Signature, String> {
    let entry = f.blocks.first().ok_or_else(|| {
        format!(
            "u0:{index} defines no blocks, so there is no entry to take its signature from"
        )
    })?;
    let mut sig = Signature::new(cc);
    for (_, t) in &entry.params {
        sig.params.push(AbiParam::new(ty(*t)));
    }
    // Whether the function answers is one fact about it, so its returns have
    // to agree with each other. They are refused here rather than left to the
    // verifier, which reports the mismatch against a signature this derived
    // from them and so cannot say which half is wrong.
    let mut rets = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| match i {
            clif::Inst::Ret(v) => Some(v.is_some()),
            _ => None,
        });
    let answers = match rets.next() {
        None => false,
        Some(first) => {
            if rets.any(|a| a != first) {
                return Err(format!(
                    "u0:{index} returns a value on some paths and nothing on others, so \
                     there is no one signature to give it"
                ));
            }
            first
        }
    };
    if answers {
        sig.returns.push(AbiParam::new(ir::types::I64));
    }
    Ok(sig)
}

/// The Cranelift signature of a callee taking `params` and answering `result`.
pub(crate) fn signature(params: &[clif::ClifTy], result: Option<clif::ClifTy>, cc: CallConv) -> Signature {
    let mut sig = Signature::new(cc);
    sig.params.extend(params.iter().map(|p| AbiParam::new(ty(*p))));
    sig.returns.extend(result.map(|r| AbiParam::new(ty(r))));
    sig
}

/// What machine code a program placed itself is called as (`Callee::Native`):
/// four `i64`s in, one out, under the convention the code was written to. On
/// x86-64 that is System V on every OS, so a body is the same bytes on Windows
/// as on Linux; on AArch64 the host's C convention is the architecture's one.
pub(crate) fn native_signature(cc: CallConv) -> Signature {
    let cc = if cfg!(target_arch = "x86_64") { CallConv::SystemV } else { cc };
    signature(&[clif::ClifTy::I64; 4], Some(clif::ClifTy::I64), cc)
}

/// Where a callee resolved: the `FuncId` it is declared under, and whether it
/// is defined in the same module as its caller.
///
/// The second is the runtime's to know, not the program's. A call to code in
/// the module may be PC-relative, because the module's code is placed in one
/// region; a call to a host symbol may not, because the loader put it wherever
/// it put it. The program cannot tell which it has.
pub struct Resolved {
    pub id: u32,
    pub colocated: bool,
}

/// Builds one function.
///
/// `declare_callee` is called once per callee and must return the `FuncId` the
/// runtime will resolve it to, so the reference is correct when it is created
/// rather than patched afterward, together with the signature that callee
/// actually has — from base's import table, or from the callee's own body.
/// Nothing here compares that against a declared one, because the artifact
/// declares none.
pub fn decode_function(
    f: &clif::Function,
    index: usize,
    cc: CallConv,
    declare_callee: &mut dyn FnMut(&clif::Callee) -> Result<(Resolved, Signature), String>,
) -> Result<ir::Function, String> {
    let sig = signature_of(f, index, cc)?;

    let mut func = ir::Function::with_name_signature(UserFuncName::user(0, index as u32), sig);

    // Cranelift wants each callee declared once, up front, and a call naming
    // that declaration. The artifact carries the callee in the call instead, so
    // the prologue is built here by walking the body and interning on first
    // use. Nothing chooses anything: `FuncRef` numbering never leaves this
    // struct, and any order gives the same machine code.
    let mut fn_refs: HashMap<clif::Callee, ir::FuncRef> = HashMap::new();
    let mut callees: HashMap<clif::Callee, (String, Signature)> = HashMap::new();
    let mut native_sig = None;
    for callee in f.blocks.iter().flat_map(|b| b.insts.iter()).filter_map(|i| match i {
        // An atomic is emitted in place and declares nothing.
        clif::Inst::Call(_, clif::Callee::Atomic(_), _) => None,
        clif::Inst::Call(_, c, _) | clif::Inst::FuncAddr(_, c) => Some(c),
        _ => None,
    }) {
        if fn_refs.contains_key(callee) || callees.contains_key(callee) {
            continue;
        }
        // Machine code resolves to nothing: the call is indirect, through the
        // address it is handed, so only its signature is declared. It is
        // checked as five `i64`s, the address first.
        if *callee == clif::Callee::Native {
            let sig = native_signature(cc);
            native_sig = Some(func.import_signature(sig.clone()));
            let mut checked = sig;
            checked.params.insert(0, AbiParam::new(ir::types::I64));
            callees.insert(clif::Callee::Native, ("native code".into(), checked));
            continue;
        }
        let (resolved, sig) = declare_callee(callee)?;
        callees.insert(
            callee.clone(),
            (
                match callee {
                    clif::Callee::Import(n) => n.clone(),
                    clif::Callee::Local(n) => format!("u0:{n}"),
                    clif::Callee::Extern(e) => format!("{}!{}", e.lib, e.symbol),
                    clif::Callee::Native | clif::Callee::Atomic(_) => unreachable!("declared above"),
                },
                sig.clone(),
            ),
        );
        let sr = func.import_signature(sig);
        let user_ref = func.declare_imported_user_function(UserExternalName {
            namespace: 0,
            index: resolved.id,
        });
        fn_refs.insert(
            callee.clone(),
            func.import_function(ExtFuncData {
                name: ExternalName::user(user_ref),
                signature: sr,
                colocated: resolved.colocated,
            }),
        );
    }

    // Blocks are created before any body is emitted, so a forward branch has a
    // target to name.
    let mut blocks: HashMap<u32, ir::Block> = HashMap::new();
    let mut vals = Vals::default();
    for b in &f.blocks {
        let blk = func.dfg.make_block();
        func.layout.append_block(blk);
        blocks.insert(b.reference.0, blk);
    }
    let block_of = |r: clif::BlockRef| -> Result<ir::Block, String> {
        blocks
            .get(&r.0)
            .copied()
            .ok_or_else(|| format!("branch to undeclared block{}", r.0))
    };

    // Block parameters, all of them, before any instruction can use one.
    for b in &f.blocks {
        let blk = blocks[&b.reference.0];
        for (v, t) in &b.params {
            vals.set(*v, func.dfg.append_block_param(blk, ty(*t)));
        }
    }

    for b in &f.blocks {
        let blk = blocks[&b.reference.0];
        for inst in &b.insts {
            check_call(index, inst, &vals, &func, &callees)?;
            emit(&mut func, blk, inst, &mut vals, &fn_refs, native_sig, &block_of)?;
        }
    }

    Ok(func)
}

/// A call against the signature its callee actually has.
///
/// Cranelift's verifier refuses the same programs a step later, reporting the
/// instruction it could not type. This says which callee was called with what
/// instead, which is the fact a generator has to act on. The signature is the
/// callee's own, so this compares the body against the callee rather than
/// against a declaration the artifact could have got wrong.
fn check_call(
    index: usize,
    inst: &clif::Inst,
    vals: &Vals,
    func: &ir::Function,
    callees: &HashMap<clif::Callee, (String, Signature)>,
) -> Result<(), String> {
    let clif::Inst::Call(_, c, args) = inst else { return Ok(()) };
    let Some((name, sig)) = callees.get(c) else { return Ok(()) };
    if args.len() != sig.params.len() {
        return Err(format!(
            "u0:{index} calls {name} with {} argument{}, and {name} takes {}",
            args.len(),
            if args.len() == 1 { "" } else { "s" },
            sig.params.len(),
        ));
    }
    for (n, (a, p)) in args.iter().zip(&sig.params).enumerate() {
        let got = func.dfg.value_type(vals.get(*a)?);
        if got != p.value_type {
            return Err(format!(
                "u0:{index} calls {name} with {got} as argument {n}, and {name} takes {} there",
                p.value_type
            ));
        }
    }
    Ok(())
}

fn emit(
    func: &mut ir::Function,
    blk: ir::Block,
    inst: &clif::Inst,
    vals: &mut Vals,
    fn_refs: &HashMap<clif::Callee, ir::FuncRef>,
    native_sig: Option<ir::SigRef>,
    block_of: &dyn Fn(clif::BlockRef) -> Result<ir::Block, String>,
) -> Result<(), String> {
    use clif::Inst as I;
    let mut cur = FuncCursor::new(func).at_bottom(blk);

    /// Binds the result of a one-result instruction to its declared value.
    macro_rules! def {
        ($dst:expr, $e:expr) => {{
            let r = $e;
            vals.set($dst, r);
        }};
    }

    match inst {
        I::Iconst(d, t, n) => def!(*d, cur.ins().iconst(ty(*t), *n)),
        I::Iadd(d, a, b) => def!(*d, cur.ins().iadd(vals.get(*a)?, vals.get(*b)?)),
        I::IaddImm(d, a, k) => def!(*d, cur.ins().iadd_imm(vals.get(*a)?, *k)),
        I::Isub(d, a, b) => def!(*d, cur.ins().isub(vals.get(*a)?, vals.get(*b)?)),
        I::Imul(d, a, b) => def!(*d, cur.ins().imul(vals.get(*a)?, vals.get(*b)?)),
        I::Udiv(d, a, b) => def!(*d, cur.ins().udiv(vals.get(*a)?, vals.get(*b)?)),
        I::Ineg(d, a) => def!(*d, cur.ins().ineg(vals.get(*a)?)),
        I::Ishl(d, a, b) => def!(*d, cur.ins().ishl(vals.get(*a)?, vals.get(*b)?)),
        I::Ushr(d, a, b) => def!(*d, cur.ins().ushr(vals.get(*a)?, vals.get(*b)?)),
        I::Band(d, a, b) => def!(*d, cur.ins().band(vals.get(*a)?, vals.get(*b)?)),
        I::BandNot(d, a, b) => def!(*d, cur.ins().band_not(vals.get(*a)?, vals.get(*b)?)),
        I::Bor(d, a, b) => def!(*d, cur.ins().bor(vals.get(*a)?, vals.get(*b)?)),
        I::Bxor(d, a, b) => def!(*d, cur.ins().bxor(vals.get(*a)?, vals.get(*b)?)),
        I::Ireduce32(d, a) => def!(*d, cur.ins().ireduce(ir::types::I32, vals.get(*a)?)),
        I::Uextend64(d, a) => def!(*d, cur.ins().uextend(ir::types::I64, vals.get(*a)?)),
        I::Sextend64(d, a) => def!(*d, cur.ins().sextend(ir::types::I64, vals.get(*a)?)),

        I::Store(v, addr, off) => {
            cur.ins().store(access(false), vals.get(*v)?, vals.get(*addr)?, *off);
        }
        I::Istore8(v, addr, off) => {
            cur.ins().istore8(access(false), vals.get(*v)?, vals.get(*addr)?, *off);
        }
        I::StoreTyped(_, v, addr, off) => {
            // The type is carried by the stored value.
            let v = vals.get(*v)?;
            let vector = cur.func.dfg.value_type(v).is_vector();
            cur.ins().store(access(vector), v, vals.get(*addr)?, *off);
        }
        I::Load(d, op, addr, off) => {
            let flags = access(op.notrap_aligned && ty(op.ty).is_vector());
            let a = vals.get(*addr)?;
            let r = match op.kind {
                clif::LoadKind::Plain => cur.ins().load(ty(op.ty), flags, a, *off),
                clif::LoadKind::Uload8 => cur.ins().uload8(ty(op.ty), flags, a, *off),
                clif::LoadKind::Sload8 => cur.ins().sload8(ty(op.ty), flags, a, *off),
                clif::LoadKind::Uload32 => cur.ins().uload32(flags, a, *off),
                clif::LoadKind::Uload16 => cur.ins().uload16(ty(op.ty), flags, a, *off),
                clif::LoadKind::Sload16 => cur.ins().sload16(ty(op.ty), flags, a, *off),
                clif::LoadKind::Sload32 => cur.ins().sload32(flags, a, *off),
            };
            vals.set(*d, r);
        }

        I::Icmp(d, c, a, b) => {
            def!(*d, cur.ins().icmp(int_cc(*c), vals.get(*a)?, vals.get(*b)?))
        }
        I::Fcmp(d, c, a, b) => {
            def!(*d, cur.ins().fcmp(float_cc(*c), vals.get(*a)?, vals.get(*b)?))
        }
        I::Select(d, c, a, b) => def!(
            *d,
            cur.ins().select(vals.get(*c)?, vals.get(*a)?, vals.get(*b)?)
        ),
        I::Bitselect(d, c, a, b) => def!(
            *d,
            cur.ins()
                .bitselect(vals.get(*c)?, vals.get(*a)?, vals.get(*b)?)
        ),

        I::Call(d, clif::Callee::Atomic(k), args) => {
            let a = vals.get_all(args)?;
            let r = atomic(&mut cur, *k, &a)?;
            match (d, r) {
                (Some(dst), Some(r)) => vals.set(*dst, r),
                (None, _) => {}
                (Some(_), None) => return Err(format!("{k:?} has no result to bind")),
            }
        }
        I::Call(d, c, args) => {
            let a = vals.get_all(args)?;
            let call = if *c == clif::Callee::Native {
                let sig = native_sig.ok_or("a native call the prologue did not declare")?;
                let (addr, rest) = a.split_first().ok_or("a native call without its address")?;
                cur.ins().call_indirect(sig, *addr, rest)
            } else {
                let f = *fn_refs
                    .get(c)
                    .ok_or_else(|| format!("call to {c:?}, which the prologue does not name"))?;
                cur.ins().call(f, &a)
            };
            if let Some(dst) = d {
                let results = cur.func.dfg.inst_results(call);
                let r = *results
                    .first()
                    .ok_or_else(|| format!("{c:?} has no result to bind"))?;
                vals.set(*dst, r);
            }
        }
        I::Jump(t, args) => {
            let a = block_args(&vals.get_all(args)?);
            cur.ins().jump(block_of(*t)?, &a);
        }
        I::Brif(c, tb, ta, eb, ea) => {
            let c = vals.get(*c)?;
            let ta = block_args(&vals.get_all(ta)?);
            let ea = block_args(&vals.get_all(ea)?);
            cur.ins().brif(c, block_of(*tb)?, &ta, block_of(*eb)?, &ea);
        }
        I::Ret(v) => {
            let vs = match v {
                Some(v) => vec![vals.get(*v)?],
                None => vec![],
            };
            cur.ins().return_(&vs);
        }

        I::Fconst(d, t, bits) => {
            let r = match t {
                clif::ClifTy::F32 => cur.ins().f32const(f32::from_bits(*bits as u32)),
                clif::ClifTy::F64 => cur.ins().f64const(f64::from_bits(*bits)),
                other => return Err(format!("fconst of non-float type {other:?}")),
            };
            vals.set(*d, r);
        }
        I::Fadd(d, a, b) => def!(*d, cur.ins().fadd(vals.get(*a)?, vals.get(*b)?)),
        I::Fsub(d, a, b) => def!(*d, cur.ins().fsub(vals.get(*a)?, vals.get(*b)?)),
        I::Fmul(d, a, b) => def!(*d, cur.ins().fmul(vals.get(*a)?, vals.get(*b)?)),
        I::Fmax(d, a, b) => def!(*d, cur.ins().fmax(vals.get(*a)?, vals.get(*b)?)),
        I::Fmin(d, a, b) => def!(*d, cur.ins().fmin(vals.get(*a)?, vals.get(*b)?)),
        I::Fneg(d, a) => def!(*d, cur.ins().fneg(vals.get(*a)?)),
        I::Fpromote(d, a) => def!(*d, cur.ins().fpromote(ir::types::F64, vals.get(*a)?)),
        I::FcvtFromSint(d, t, s) => {
            def!(*d, cur.ins().fcvt_from_sint(ty(*t), vals.get(*s)?))
        }
        I::FcvtToUint(d, t, s) => {
            def!(*d, cur.ins().fcvt_to_uint_sat(ty(*t), vals.get(*s)?))
        }
        I::Splat(d, t, s) => def!(*d, cur.ins().splat(ty(*t), vals.get(*s)?)),
        I::Extractlane(d, s, lane) => def!(*d, cur.ins().extractlane(vals.get(*s)?, *lane)),
        I::Bitcast(d, t, s) => {
            def!(*d, cur.ins().bitcast(ty(*t), MemFlags::new(), vals.get(*s)?))
        }
        I::FuncAddr(d, c) => {
            let f = *fn_refs.get(c).ok_or_else(|| {
                format!("func_addr of {c:?}, which the prologue does not name")
            })?;
            def!(*d, cur.ins().func_addr(ir::types::I64, f))
        }
        I::Ctz(d, a) => def!(*d, cur.ins().ctz(vals.get(*a)?)),
        I::Popcnt(d, a) => def!(*d, cur.ins().popcnt(vals.get(*a)?)),
        I::VhighBits(d, a) => def!(*d, cur.ins().vhigh_bits(ir::types::I32, vals.get(*a)?)),
        I::Ibin(d, k, a, b) => {
            let (x, y) = (vals.get(*a)?, vals.get(*b)?);
            let r = match k {
                clif::IBin::Sdiv => cur.ins().sdiv(x, y),
                clif::IBin::Urem => cur.ins().urem(x, y),
                clif::IBin::Srem => cur.ins().srem(x, y),
                clif::IBin::Smin => cur.ins().smin(x, y),
                clif::IBin::Smax => cur.ins().smax(x, y),
                clif::IBin::Umin => cur.ins().umin(x, y),
                clif::IBin::Umax => cur.ins().umax(x, y),
                clif::IBin::Umulhi => cur.ins().umulhi(x, y),
                clif::IBin::Smulhi => cur.ins().smulhi(x, y),
            };
            vals.set(*d, r);
        }
        I::Ishift(d, k, a, b) => {
            let (x, y) = (vals.get(*a)?, vals.get(*b)?);
            let r = match k {
                clif::IShift::Sshr => cur.ins().sshr(x, y),
                clif::IShift::Rotl => cur.ins().rotl(x, y),
                clif::IShift::Rotr => cur.ins().rotr(x, y),
            };
            vals.set(*d, r);
        }
        I::Iun(d, k, a) => {
            let x = vals.get(*a)?;
            let r = match k {
                clif::IUn::Bnot => cur.ins().bnot(x),
                clif::IUn::Iabs => cur.ins().iabs(x),
                clif::IUn::Clz => cur.ins().clz(x),
                clif::IUn::Bswap => cur.ins().bswap(x),
                clif::IUn::Bitrev => cur.ins().bitrev(x),
            };
            vals.set(*d, r);
        }
        I::Fbin(d, k, a, b) => {
            let (x, y) = (vals.get(*a)?, vals.get(*b)?);
            let r = match k {
                clif::FBin::Fdiv => cur.ins().fdiv(x, y),
                clif::FBin::Fcopysign => cur.ins().fcopysign(x, y),
            };
            vals.set(*d, r);
        }
        I::Fun1(d, k, a) => {
            let x = vals.get(*a)?;
            let r = match k {
                clif::FUn::Sqrt => cur.ins().sqrt(x),
                clif::FUn::Fabs => cur.ins().fabs(x),
                clif::FUn::Ceil => cur.ins().ceil(x),
                clif::FUn::Floor => cur.ins().floor(x),
                clif::FUn::Trunc => cur.ins().trunc(x),
                clif::FUn::Nearest => cur.ins().nearest(x),
            };
            vals.set(*d, r);
        }
        I::Fconv(d, k, t, a) => {
            let x = vals.get(*a)?;
            let r = match k {
                clif::FConv::ToSint => cur.ins().fcvt_to_sint_sat(ty(*t), x),
                clif::FConv::FromUint => cur.ins().fcvt_from_uint(ty(*t), x),
                clif::FConv::Demote => cur.ins().fdemote(ty(*t), x),
            };
            vals.set(*d, r);
        }
        I::Iext(d, k, t, a) => {
            let x = vals.get(*a)?;
            let r = match k {
                clif::IExt::Reduce => cur.ins().ireduce(ty(*t), x),
                clif::IExt::Uextend => cur.ins().uextend(ty(*t), x),
                clif::IExt::Sextend => cur.ins().sextend(ty(*t), x),
            };
            vals.set(*d, r);
        }
        I::Fma(d, a, b, c) => {
            def!(*d, cur.ins().fma(vals.get(*a)?, vals.get(*b)?, vals.get(*c)?))
        }
    }
    Ok(())
}

/// An atomic instruction on its arguments, the address first; its result, if
/// it answers one.
///
/// `trusted` flags: `notrap`, as every access here carries, and `aligned`,
/// which the model requires of an atomic because AArch64 faults on one that
/// is not.
fn atomic(
    cur: &mut FuncCursor,
    k: clif::Atomic,
    a: &[ir::Value],
) -> Result<Option<ir::Value>, String> {
    use ir::AtomicRmwOp as Op;
    let f = MemFlags::trusted();
    let arity = |n: usize| {
        if a.len() == n { Ok(()) } else { Err(format!("{k:?} takes {n} arguments, given {}", a.len())) }
    };
    Ok(match k {
        clif::Atomic::Fence => {
            arity(0)?;
            cur.ins().fence();
            None
        }
        clif::Atomic::Load(t) => {
            arity(1)?;
            Some(cur.ins().atomic_load(ty(t), f, a[0]))
        }
        clif::Atomic::Store(_) => {
            arity(2)?;
            cur.ins().atomic_store(f, a[0], a[1]);
            None
        }
        clif::Atomic::Rmw(t, op) => {
            arity(2)?;
            let op = match op {
                clif::AtomicRmw::Add => Op::Add,
                clif::AtomicRmw::Sub => Op::Sub,
                clif::AtomicRmw::And => Op::And,
                clif::AtomicRmw::Nand => Op::Nand,
                clif::AtomicRmw::Or => Op::Or,
                clif::AtomicRmw::Xor => Op::Xor,
                clif::AtomicRmw::Xchg => Op::Xchg,
                clif::AtomicRmw::Umin => Op::Umin,
                clif::AtomicRmw::Umax => Op::Umax,
                clif::AtomicRmw::Smin => Op::Smin,
                clif::AtomicRmw::Smax => Op::Smax,
            };
            Some(cur.ins().atomic_rmw(ty(t), f, op, a[0], a[1]))
        }
        clif::Atomic::Cas(_) => {
            arity(3)?;
            Some(cur.ins().atomic_cas(f, a[0], a[1], a[2]))
        }
    })
}

/// The flags every load and store carries: `notrap`, and `aligned` on the
/// vector accesses that assert it.
///
/// `notrap` on every access, because base installs no trap handler: an access
/// that faults ends the process whether or not Cranelift was told it could.
/// Saying so everywhere is what keeps the load and the store of a
/// read-modify-write under identical flags, which x64 requires before it fuses
/// them into one `add [mem], x` (`store_x64_add_mem`); a counter bumped in
/// memory is otherwise three instructions.
///
/// `aligned` only on vectors, the one place it changes code: a legacy-SSE
/// instruction takes a memory operand only when it is aligned. On a scalar it
/// changes nothing on x64 except whether the flags match.
fn access(aligned_vector: bool) -> MemFlags {
    let mut f = MemFlags::new();
    f.set_notrap();
    if aligned_vector {
        f.set_aligned();
    }
    f
}
