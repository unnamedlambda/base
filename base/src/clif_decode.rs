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
pub fn signature_of(f: &clif::Function, cc: CallConv) -> Result<Signature, String> {
    let entry = f.blocks.first().ok_or_else(|| {
        format!(
            "u0:{} defines no blocks, so there is no entry to take its signature from",
            f.index
        )
    })?;
    let mut sig = Signature::new(cc);
    for (_, t) in &entry.params {
        sig.params.push(AbiParam::new(ty(*t)));
    }
    let answers = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, clif::Inst::Ret(Some(_))));
    if answers {
        sig.returns.push(AbiParam::new(ir::types::I64));
    }
    Ok(sig)
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
/// `declare_callee` is called once per callee in the prologue and must return
/// the `FuncId` the runtime will resolve it to, so the reference is correct
/// when it is created rather than patched afterward.
pub fn decode_function(
    f: &clif::Function,
    cc: CallConv,
    declare_callee: &mut dyn FnMut(&clif::Callee, &Signature) -> Result<Resolved, String>,
) -> Result<ir::Function, String> {
    let sig = signature_of(f, cc)?;

    let mut func = ir::Function::with_name_signature(UserFuncName::user(0, f.index), sig);

    // Prologue: signatures, then callees that reference them.
    let mut sig_refs: HashMap<u32, ir::SigRef> = HashMap::new();
    for s in &f.sigs {
        let mut csig = Signature::new(cc);
        for p in &s.params {
            csig.params.push(AbiParam::new(ty(*p)));
        }
        if let Some(r) = s.result {
            csig.returns.push(AbiParam::new(ty(r)));
        }
        sig_refs.insert(s.reference.0, func.import_signature(csig));
    }

    let mut fn_refs: HashMap<u32, ir::FuncRef> = HashMap::new();
    for d in &f.fns {
        let sr = *sig_refs
            .get(&d.sig.0)
            .ok_or_else(|| format!("fn{} names undeclared sig{}", d.reference.0, d.sig.0))?;
        let resolved = declare_callee(&d.callee, &func.dfg.signatures[sr].clone())?;
        let user_ref = func.declare_imported_user_function(UserExternalName {
            namespace: 0,
            index: resolved.id,
        });
        let fr = func.import_function(ExtFuncData {
            name: ExternalName::user(user_ref),
            signature: sr,
            colocated: resolved.colocated,
        });
        fn_refs.insert(d.reference.0, fr);
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
            emit(&mut func, blk, inst, &mut vals, &fn_refs, &block_of)?;
        }
    }

    Ok(func)
}

fn emit(
    func: &mut ir::Function,
    blk: ir::Block,
    inst: &clif::Inst,
    vals: &mut Vals,
    fn_refs: &HashMap<u32, ir::FuncRef>,
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
            cur.ins().store(MemFlags::new(), vals.get(*v)?, vals.get(*addr)?, *off);
        }
        I::Istore8(v, addr, off) => {
            cur.ins().istore8(MemFlags::new(), vals.get(*v)?, vals.get(*addr)?, *off);
        }
        I::StoreTyped(_, v, addr, off) => {
            // The type is carried by the stored value; `notrap aligned` is the
            // part that matters here.
            cur.ins().store(trusted(), vals.get(*v)?, vals.get(*addr)?, *off);
        }
        I::Load(d, op, addr, off) => {
            let flags = if op.notrap_aligned { trusted() } else { MemFlags::new() };
            let a = vals.get(*addr)?;
            let r = match op.kind {
                clif::LoadKind::Plain => cur.ins().load(ty(op.ty), flags, a, *off),
                clif::LoadKind::Uload8 => cur.ins().uload8(ty(op.ty), flags, a, *off),
                clif::LoadKind::Sload8 => cur.ins().sload8(ty(op.ty), flags, a, *off),
                clif::LoadKind::Uload32 => cur.ins().uload32(flags, a, *off),
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

        I::Call(d, fr, args) => {
            let f = *fn_refs
                .get(&fr.0)
                .ok_or_else(|| format!("call to undeclared fn{}", fr.0))?;
            let a = vals.get_all(args)?;
            let call = cur.ins().call(f, &a);
            if let Some(dst) = d {
                let results = cur.func.dfg.inst_results(call);
                let r = *results
                    .first()
                    .ok_or_else(|| format!("fn{} has no result to bind", fr.0))?;
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
        I::FuncAddr(d, fr) => {
            let f = *fn_refs
                .get(&fr.0)
                .ok_or_else(|| format!("func_addr of undeclared fn{}", fr.0))?;
            def!(*d, cur.ins().func_addr(ir::types::I64, f))
        }
        I::Ctz(d, a) => def!(*d, cur.ins().ctz(vals.get(*a)?)),
        I::Popcnt(d, a) => def!(*d, cur.ins().popcnt(vals.get(*a)?)),
        I::VhighBits(d, a) => def!(*d, cur.ins().vhigh_bits(ir::types::I32, vals.get(*a)?)),
    }
    Ok(())
}

/// `notrap aligned` — the flags the vector and float accessors carry.
fn trusted() -> MemFlags {
    let mut f = MemFlags::new();
    f.set_notrap();
    f.set_aligned();
    f
}
