//! Builds `cranelift_codegen::ir::Function` from the program an artifact
//! carries.
//!
//! The artifact is CLIF as data: an instruction is a Cranelift instruction
//! format, an opcode of it by CLIF's name, and the format's fields, so this is
//! one rule per format rather than one per opcode. Every name is parsed by
//! Cranelift's own parser and printed back, and one that does not come back as
//! it was written is refused, so a name has CLIF's spelling and no other.
//! Everything Cranelift could panic on rather than refuse — an opcode of
//! another format, a type the opcode is not defined at — is checked before
//! the instruction is built, so a load answers an error or a function, and
//! never stops the process.

use base_types::clif;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::instructions::InstructionFormat;
use cranelift_codegen::ir::{
    self, ArgumentExtension, ArgumentPurpose, ExtFuncData, ExternalName, InstBuilder, InstBuilderBase,
    MemFlags, Opcode, Signature, TrapCode, UserExternalName, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use std::collections::HashMap;
use std::str::FromStr;

/// The type CLIF spells `t`: a lane type, and `x` and a lane count for a
/// vector, printed back to check it was CLIF's spelling.
fn ty(t: &clif::ClifTy) -> Result<ir::Type, String> {
    use ir::types::*;
    let name = &*t.0;
    let (lane, lanes) = match name.split_once('x') {
        Some((lane, n)) => (lane, Some(n)),
        None => (name, None),
    };
    let lane = match lane {
        "i8" => I8,
        "i16" => I16,
        "i32" => I32,
        "i64" => I64,
        "i128" => I128,
        "f16" => F16,
        "f32" => F32,
        "f64" => F64,
        "f128" => F128,
        _ => return Err(format!("{name:?} is not a CLIF type")),
    };
    let t = match lanes {
        None => Some(lane),
        Some(n) => n.parse().ok().and_then(|n| lane.by(n)),
    };
    t.filter(|t| t.to_string() == name).ok_or_else(|| format!("{name:?} is not a CLIF type"))
}

/// A parameter: its type, and CLIF's words for how it is passed.
fn abi_param(p: &clif::AbiParam) -> Result<ir::AbiParam, String> {
    let mut a = ir::AbiParam::new(ty(&p.0)?);
    for w in &p.1 {
        match w.as_str() {
            "uext" => a.extension = ArgumentExtension::Uext,
            "sext" => a.extension = ArgumentExtension::Sext,
            w => {
                a.purpose =
                    ArgumentPurpose::from_str(w).map_err(|_| format!("{w:?} is not a CLIF parameter attribute"))?
            }
        }
    }
    let stated: Vec<&str> = std::iter::once(&*p.0 .0).chain(p.1.iter().map(String::as_str)).collect();
    let stated = stated.join(" ");
    if a.to_string() != stated {
        return Err(format!("{stated:?} is not a parameter as CLIF writes one; it writes {a}"));
    }
    Ok(a)
}

/// A signature, under the convention it names: `Host` is `host`, the
/// convention of the machine this runs on, and `Arch` is Cranelift's System V,
/// which is the architecture's on every OS.
pub(crate) fn signature(s: &clif::Signature, host: CallConv) -> Result<Signature, String> {
    let mut sig = Signature::new(match s.convention {
        clif::Convention::Host => host,
        clif::Convention::Arch => CallConv::SystemV,
    });
    for p in &s.params {
        sig.params.push(abi_param(p)?);
    }
    for r in &s.returns {
        sig.returns.push(abi_param(r)?);
    }
    Ok(sig)
}

/// The signature of function `index`. A named function's is `BaseFunction`,
/// since a host calls it as one; a body that does not take those arguments or
/// answer that status is ill-typed CLIF, which Cranelift's verifier refuses as
/// it would any other.
pub fn signature_of(f: &clif::Function, index: usize, host: CallConv) -> Result<Signature, String> {
    if f.name.is_some() && f.signature != clif::Signature::base_function() {
        return Err(format!("u0:{index} has a name, so a host calls it, and its signature is not BaseFunction's"));
    }
    signature(&f.signature, host)
}

/// The value each `Val` names, defined once each.
#[derive(Default)]
struct Vals(HashMap<u32, ir::Value>);

impl Vals {
    fn get(&self, v: clif::Val) -> Result<ir::Value, String> {
        self.0.get(&v.0).copied().ok_or_else(|| format!("v{} used before it is defined", v.0))
    }
    fn get_all(&self, vs: &[clif::Val]) -> Result<Vec<ir::Value>, String> {
        vs.iter().map(|v| self.get(*v)).collect()
    }
    fn set(&mut self, v: clif::Val, val: ir::Value) -> Result<(), String> {
        match self.0.insert(v.0, val) {
            Some(_) => Err(format!("v{} is defined twice", v.0)),
            None => Ok(()),
        }
    }
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

/// What the decoder knows of the module a function is in: what its callees
/// resolve to, and where the arena is declared.
pub struct Module<'a> {
    /// The artifact's primitives, which a call names by position.
    pub primitives: &'a [clif::Primitive],
    /// The `DataId` the arena is declared under.
    pub arena: u32,
    /// The `FuncId` a callee is declared under, and the signature it has —
    /// from the primitive's declaration, or the function's own.
    pub resolve: &'a mut dyn FnMut(clif::Callee) -> Result<(Resolved, Signature), String>,
}

/// Builds one function.
pub fn decode_function(
    f: &clif::Function,
    index: usize,
    host: CallConv,
    module: &mut Module,
) -> Result<ir::Function, String> {
    let sig = signature_of(f, index, host)?;
    if f.blocks.is_empty() {
        return Err(format!("u0:{index} has no blocks, so no entry block"));
    }
    let mut func = ir::Function::with_name_signature(UserFuncName::user(0, index as u32), sig);

    let slots = f
        .stack_slots
        .iter()
        .map(|s| {
            if !s.align.is_power_of_two() {
                return Err(format!("u0:{index} has a stack slot aligned to {}, not a power of two", s.align));
            }
            Ok(func.create_sized_stack_slot(ir::StackSlotData::new(
                ir::StackSlotKind::ExplicitSlot,
                s.size,
                s.align.trailing_zeros() as u8,
            )))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Cranelift wants each callee, signature and global declared once in the
    // function, and an instruction naming that declaration. The artifact
    // carries the thing itself in the instruction instead, so the declarations
    // are made here, on first use. Nothing chooses anything: the numbering
    // never leaves this function, and any order gives the same machine code.
    let mut callees: HashMap<clif::Callee, (ir::FuncRef, String, Signature)> = HashMap::new();
    let mut sigs: HashMap<&clif::Signature, (ir::SigRef, Signature)> = HashMap::new();
    let mut arena = None;
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        match inst {
            clif::Inst::Call(_, _, _, c, _) | clif::Inst::FuncAddr(_, _, _, c) if !callees.contains_key(c) => {
                let (resolved, sig) = (module.resolve)(*c)?;
                let name = match c {
                    clif::Callee::Function(n) => format!("u0:{n}"),
                    clif::Callee::Primitive(i) => module
                        .primitives
                        .get(*i as usize)
                        .map(|p| format!("{}!{}", p.lib, p.symbol))
                        .unwrap_or_else(|| format!("primitive {i}")),
                };
                let sr = func.import_signature(sig.clone());
                let user = func.declare_imported_user_function(UserExternalName { namespace: 0, index: resolved.id });
                let fr = func.import_function(ExtFuncData {
                    name: ExternalName::user(user),
                    signature: sr,
                    colocated: resolved.colocated,
                });
                callees.insert(*c, (fr, name, sig));
            }
            clif::Inst::CallIndirect(_, _, _, s, _, _) if !sigs.contains_key(s) => {
                let sig = signature(s, host)?;
                sigs.insert(s, (func.import_signature(sig.clone()), sig));
            }
            clif::Inst::UnaryGlobalValue(_, _, _, clif::Global::Arena) if arena.is_none() => {
                // Data is namespace 1 of a module's names, as functions are 0.
                let name =
                    func.declare_imported_user_function(UserExternalName { namespace: 1, index: module.arena });
                arena = Some(func.create_global_value(ir::GlobalValueData::Symbol {
                    name: ExternalName::user(name),
                    offset: ir::immediates::Imm64::new(0),
                    colocated: false,
                    tls: false,
                }));
            }
            _ => {}
        }
    }

    // Blocks are created, in the order they are laid out, before any body is
    // emitted, so a forward branch has a target to name; and their parameters
    // before any instruction can use one.
    let mut blocks: HashMap<u32, ir::Block> = HashMap::new();
    let mut order = Vec::with_capacity(f.blocks.len());
    for b in &f.blocks {
        let blk = func.dfg.make_block();
        func.layout.append_block(blk);
        if blocks.insert(b.reference.0, blk).is_some() {
            return Err(format!("u0:{index} names two blocks block{}", b.reference.0));
        }
        order.push(blk);
    }
    let mut vals = Vals::default();
    for (b, blk) in f.blocks.iter().zip(&order) {
        for (v, t) in &b.params {
            vals.set(*v, func.dfg.append_block_param(*blk, ty(t)?)).map_err(|e| format!("u0:{index}: {e}"))?;
        }
    }

    let ctx = Ctx { callees: &callees, sigs: &sigs, arena, slots: &slots, blocks: &blocks };
    for (b, blk) in f.blocks.iter().zip(&order) {
        for inst in &b.insts {
            emit(&mut func, *blk, inst, &mut vals, &ctx).map_err(|e| format!("u0:{index}: {e}"))?;
        }
    }
    Ok(func)
}

/// The declarations a function's instructions name.
struct Ctx<'a> {
    callees: &'a HashMap<clif::Callee, (ir::FuncRef, String, Signature)>,
    sigs: &'a HashMap<&'a clif::Signature, (ir::SigRef, Signature)>,
    arena: Option<ir::GlobalValue>,
    slots: &'a [ir::StackSlot],
    blocks: &'a HashMap<u32, ir::Block>,
}

/// A call against the signature its callee has.
///
/// Cranelift's verifier refuses the same programs a step later, reporting the
/// instruction it could not type. This says which callee was called with what
/// instead, which is the fact a generator has to act on.
fn check_call(name: &str, sig: &Signature, args: &[ir::Value], dfg: &ir::DataFlowGraph) -> Result<(), String> {
    if args.len() != sig.params.len() {
        return Err(format!(
            "{name} is called with {} argument{}, and takes {}",
            args.len(),
            if args.len() == 1 { "" } else { "s" },
            sig.params.len(),
        ));
    }
    for (n, (a, p)) in args.iter().zip(&sig.params).enumerate() {
        let got = dfg.value_type(*a);
        if got != p.value_type {
            return Err(format!("{name} is called with {got} as argument {n}, and takes {} there", p.value_type));
        }
    }
    Ok(())
}

/// The opcode CLIF spells `name`, which must be of `format`: the builder
/// Cranelift offers for a format checks that only in debug builds.
fn opcode(name: &str, format: InstructionFormat) -> Result<Opcode, String> {
    let op = Opcode::from_str(name).map_err(|_| format!("{name:?} is not a CLIF opcode"))?;
    if op.format() != format {
        return Err(format!("{name} is a {:?} instruction, not a {format:?} one", op.format()));
    }
    Ok(op)
}

/// The controlling type `op` is built at: the type of the operand CLIF takes
/// it from, or the one the instruction names where CLIF takes it from none.
/// Exactly one of the two, so an instruction has one spelling; and a type in
/// the set the opcode is defined at, checked before building, since building
/// at another can panic inside Cranelift rather than fail.
fn ctrl_type(
    op: Opcode,
    named: Option<&clif::ClifTy>,
    data: &ir::InstructionData,
    dfg: &ir::DataFlowGraph,
) -> Result<ir::Type, String> {
    let c = op.constraints();
    let Some(set) = c.ctrl_typeset() else {
        return match named {
            None => Ok(ir::types::INVALID),
            Some(t) => Err(format!("{op} has no type to name, and names {t}")),
        };
    };
    let t = match (c.use_typevar_operand(), named) {
        (true, None) => {
            let v = data
                .typevar_operand(&dfg.value_lists)
                .ok_or_else(|| format!("{op} has no operand to take its type from"))?;
            dfg.value_type(v)
        }
        (false, Some(t)) => ty(t)?,
        (true, Some(t)) => return Err(format!("{op} takes its type from an operand, and names {t}")),
        (false, None) => return Err(format!("{op} names the type it is at, and names none")),
    };
    if !set.contains(t) {
        return Err(format!("{op} is not defined at {t}"));
    }
    Ok(t)
}

/// The flags of a memory access: CLIF's words, in the order CLIF prints them,
/// and `notrap` on every access, which the engine sets and a program does not
/// state.
///
/// `notrap` everywhere because base installs no trap handler: an access that
/// faults ends the process whether or not Cranelift was told it could. Saying
/// so everywhere is what keeps the load and the store of a read-modify-write
/// under identical flags, which x64 requires before it fuses them into one
/// `add [mem], x` (`store_x64_add_mem`); a counter bumped in memory is
/// otherwise three instructions. An instruction of a memory format that
/// touches no memory (`bitcast`) takes only what it states.
fn flags(op: Opcode, words: &[String]) -> Result<MemFlags, String> {
    let mut m = MemFlags::new();
    for w in words {
        if w == "notrap" || TrapCode::from_str(w).is_ok() {
            return Err(format!("{op} states {w:?}, and whether an access traps is the engine's to say"));
        }
        match m.set_by_name(w) {
            Ok(true) => {}
            Ok(false) => return Err(format!("{w:?} is not a CLIF memory flag")),
            Err(e) => return Err(format!("{op} states {w:?}: {e}")),
        }
    }
    let stated: String = words.iter().map(|w| format!(" {w}")).collect();
    if m.to_string() != stated {
        return Err(format!("{op} states the flags{stated}, which CLIF writes as{m}"));
    }
    if op.can_load() || op.can_store() {
        m.set_notrap();
    }
    Ok(m)
}

/// Binds every result of `inst` to the values the program names for them,
/// which must be all of them, as CLIF text names them.
fn bind(vals: &mut Vals, names: &[clif::Val], inst: ir::Inst, dfg: &ir::DataFlowGraph) -> Result<(), String> {
    let results = dfg.inst_results(inst);
    if results.len() != names.len() {
        return Err(format!(
            "{} answers {} result{}, and the program names {}",
            dfg.insts[inst].opcode(),
            results.len(),
            if results.len() == 1 { "" } else { "s" },
            names.len()
        ));
    }
    for (n, r) in names.iter().zip(results) {
        vals.set(*n, *r)?;
    }
    Ok(())
}

fn emit(
    func: &mut ir::Function,
    blk: ir::Block,
    inst: &clif::Inst,
    vals: &mut Vals,
    ctx: &Ctx,
) -> Result<(), String> {
    use clif::Inst as I;
    use ir::InstructionData as D;
    use InstructionFormat as F;
    let mut cur = FuncCursor::new(func).at_bottom(blk);

    // Each format the same way: its opcode, checked against the format; its
    // controlling type, from the data it is made of; then the instruction,
    // built by the format's own builder where that masks an immediate to the
    // type and from the data itself otherwise; and its results bound.
    macro_rules! build {
        ($res:expr, $name:expr, $named:expr, $format:ident, |$op:ident| $data:expr, |$t:ident, $d:ident| $make:expr) => {{
            let $op = opcode($name, F::$format)?;
            let $d = $data;
            let $t = ctrl_type($op, $named.as_ref(), &$d, &cur.func.dfg)?;
            let (inst, dfg) = $make;
            bind(vals, $res, inst, dfg)?;
        }};
    }
    fn list(cur: &mut FuncCursor<'_>, vs: &[ir::Value]) -> ir::ValueList {
        ir::ValueList::from_slice(vs, &mut cur.func.dfg.value_lists)
    }
    fn block_call(cur: &mut FuncCursor<'_>, ctx: &Ctx, vals: &Vals, c: &clif::BlockCall) -> Result<ir::BlockCall, String> {
        let n = c.0 .0;
        let blk = *ctx.blocks.get(&n).ok_or_else(|| format!("a branch to block{n}, which there is not"))?;
        let args: Vec<ir::BlockArg> = vals.get_all(&c.1)?.into_iter().map(ir::BlockArg::Value).collect();
        Ok(ir::BlockCall::new(blk, args, &mut cur.func.dfg.value_lists))
    }
    let slot = |s: u32| ctx.slots.get(s as usize).copied().ok_or_else(|| format!("stack slot {s}, which there is not"));
    let cond = |c: &str| ir::condcodes::IntCC::from_str(c).map_err(|_| format!("{c:?} is not an integer condition"));
    let fcond = |c: &str| ir::condcodes::FloatCC::from_str(c).map_err(|_| format!("{c:?} is not a float condition"));

    match inst {
        I::Unary(r, n, t, a) => {
            let a = vals.get(*a)?;
            build!(r, n, t, Unary, |op| D::Unary { opcode: op, arg: a }, |ct, _d| cur.ins().Unary(op, ct, a))
        }
        I::UnaryImm(r, n, t, k) => {
            let imm: ir::immediates::Imm64 = (*k).into();
            build!(r, n, t, UnaryImm, |op| D::UnaryImm { opcode: op, imm }, |ct, _d| cur.ins().UnaryImm(op, ct, imm))
        }
        I::UnaryIeee32(r, n, t, bits) => {
            let imm = ir::immediates::Ieee32::with_bits(*bits);
            build!(r, n, t, UnaryIeee32, |op| D::UnaryIeee32 { opcode: op, imm }, |ct, _d| {
                cur.ins().UnaryIeee32(op, ct, imm)
            })
        }
        I::UnaryIeee64(r, n, t, bits) => {
            let imm = ir::immediates::Ieee64::with_bits(*bits);
            build!(r, n, t, UnaryIeee64, |op| D::UnaryIeee64 { opcode: op, imm }, |ct, _d| {
                cur.ins().UnaryIeee64(op, ct, imm)
            })
        }
        I::UnaryGlobalValue(r, n, t, clif::Global::Arena) => {
            let gv = ctx.arena.ok_or("the arena is not declared")?;
            build!(r, n, t, UnaryGlobalValue, |op| D::UnaryGlobalValue { opcode: op, global_value: gv }, |ct, d| {
                cur.ins().build(d, ct)
            })
        }
        I::Binary(r, n, t, a, b) => {
            let (a, b) = (vals.get(*a)?, vals.get(*b)?);
            build!(r, n, t, Binary, |op| D::Binary { opcode: op, args: [a, b] }, |ct, _d| cur.ins().Binary(op, ct, a, b))
        }
        I::BinaryImm8(r, n, t, a, k) => {
            let a = vals.get(*a)?;
            build!(r, n, t, BinaryImm8, |op| D::BinaryImm8 { opcode: op, arg: a, imm: *k }, |ct, _d| {
                cur.ins().BinaryImm8(op, ct, *k, a)
            })
        }
        I::BinaryImm64(r, n, t, a, k) => {
            let a = vals.get(*a)?;
            let imm: ir::immediates::Imm64 = (*k).into();
            build!(r, n, t, BinaryImm64, |op| D::BinaryImm64 { opcode: op, arg: a, imm }, |ct, _d| {
                cur.ins().BinaryImm64(op, ct, imm, a)
            })
        }
        I::Ternary(r, n, t, a, b, c) => {
            let (a, b, c) = (vals.get(*a)?, vals.get(*b)?, vals.get(*c)?);
            build!(r, n, t, Ternary, |op| D::Ternary { opcode: op, args: [a, b, c] }, |ct, _d| {
                cur.ins().Ternary(op, ct, a, b, c)
            })
        }
        I::MultiAry(r, n, t, args) => {
            let args = vals.get_all(args)?;
            let args = list(&mut cur, &args);
            build!(r, n, t, MultiAry, |op| D::MultiAry { opcode: op, args }, |ct, d| cur.ins().build(d, ct))
        }
        I::NullAry(r, n, t) => {
            build!(r, n, t, NullAry, |op| D::NullAry { opcode: op }, |ct, _d| cur.ins().NullAry(op, ct))
        }
        I::IntCompare(r, n, t, c, a, b) => {
            let (c, a, b) = (cond(c)?, vals.get(*a)?, vals.get(*b)?);
            build!(r, n, t, IntCompare, |op| D::IntCompare { opcode: op, args: [a, b], cond: c }, |ct, _d| {
                cur.ins().IntCompare(op, ct, c, a, b)
            })
        }
        I::FloatCompare(r, n, t, c, a, b) => {
            let (c, a, b) = (fcond(c)?, vals.get(*a)?, vals.get(*b)?);
            build!(r, n, t, FloatCompare, |op| D::FloatCompare { opcode: op, args: [a, b], cond: c }, |ct, _d| {
                cur.ins().FloatCompare(op, ct, c, a, b)
            })
        }
        I::Jump(r, n, t, dest) => {
            let destination = block_call(&mut cur, ctx, vals, dest)?;
            build!(r, n, t, Jump, |op| D::Jump { opcode: op, destination }, |ct, d| cur.ins().build(d, ct))
        }
        I::Brif(r, n, t, c, then, els) => {
            let arg = vals.get(*c)?;
            let blocks = [block_call(&mut cur, ctx, vals, then)?, block_call(&mut cur, ctx, vals, els)?];
            build!(r, n, t, Brif, |op| D::Brif { opcode: op, arg, blocks }, |ct, d| cur.ins().build(d, ct))
        }
        I::Call(r, n, t, c, args) => {
            let (func_ref, name, sig) = ctx.callees.get(c).ok_or_else(|| format!("a call to {c:?}, undeclared"))?;
            let args = vals.get_all(args)?;
            check_call(name, sig, &args, &cur.func.dfg)?;
            let args = list(&mut cur, &args);
            build!(r, n, t, Call, |op| D::Call { opcode: op, func_ref: *func_ref, args }, |ct, d| {
                cur.ins().build(d, ct)
            })
        }
        I::CallIndirect(r, n, t, s, callee, args) => {
            let (sig_ref, sig) = ctx.sigs.get(s).ok_or("an indirect call to an undeclared signature")?;
            let args = vals.get_all(args)?;
            check_call("an indirect callee", sig, &args, &cur.func.dfg)?;
            let args = list(&mut cur, &[&[vals.get(*callee)?][..], &args].concat());
            build!(r, n, t, CallIndirect, |op| D::CallIndirect { opcode: op, sig_ref: *sig_ref, args }, |ct, d| {
                cur.ins().build(d, ct)
            })
        }
        I::FuncAddr(r, n, t, c) => {
            let (func_ref, _, _) = ctx.callees.get(c).ok_or_else(|| format!("func_addr of {c:?}, undeclared"))?;
            build!(r, n, t, FuncAddr, |op| D::FuncAddr { opcode: op, func_ref: *func_ref }, |ct, d| {
                cur.ins().build(d, ct)
            })
        }
        I::AtomicRmw(r, n, t, f, rmw, a, x) => {
            let rmw = ir::AtomicRmwOp::from_str(rmw).map_err(|_| format!("{rmw:?} is not an atomic operation"))?;
            let (a, x) = (vals.get(*a)?, vals.get(*x)?);
            let flags = flags(opcode(n, F::AtomicRmw)?, f)?;
            build!(r, n, t, AtomicRmw, |op| D::AtomicRmw { opcode: op, args: [a, x], flags, op: rmw }, |ct, _d| {
                cur.ins().AtomicRmw(op, ct, flags, rmw, a, x)
            })
        }
        I::AtomicCas(r, n, t, f, a, e, x) => {
            let (a, e, x) = (vals.get(*a)?, vals.get(*e)?, vals.get(*x)?);
            let flags = flags(opcode(n, F::AtomicCas)?, f)?;
            build!(r, n, t, AtomicCas, |op| D::AtomicCas { opcode: op, args: [a, e, x], flags }, |ct, _d| {
                cur.ins().AtomicCas(op, ct, flags, a, e, x)
            })
        }
        I::Load(r, n, t, f, a, off) => {
            let a = vals.get(*a)?;
            let off = ir::immediates::Offset32::new(*off);
            let flags = flags(opcode(n, F::Load)?, f)?;
            build!(r, n, t, Load, |op| D::Load { opcode: op, arg: a, flags, offset: off }, |ct, _d| {
                cur.ins().Load(op, ct, flags, off, a)
            })
        }
        I::LoadNoOffset(r, n, t, f, a) => {
            let a = vals.get(*a)?;
            let flags = flags(opcode(n, F::LoadNoOffset)?, f)?;
            build!(r, n, t, LoadNoOffset, |op| D::LoadNoOffset { opcode: op, arg: a, flags }, |ct, _d| {
                cur.ins().LoadNoOffset(op, ct, flags, a)
            })
        }
        I::Store(r, n, t, f, v, a, off) => {
            let (v, a) = (vals.get(*v)?, vals.get(*a)?);
            let off = ir::immediates::Offset32::new(*off);
            let flags = flags(opcode(n, F::Store)?, f)?;
            build!(r, n, t, Store, |op| D::Store { opcode: op, args: [v, a], flags, offset: off }, |ct, _d| {
                cur.ins().Store(op, ct, flags, off, v, a)
            })
        }
        I::StoreNoOffset(r, n, t, f, v, a) => {
            let (v, a) = (vals.get(*v)?, vals.get(*a)?);
            let flags = flags(opcode(n, F::StoreNoOffset)?, f)?;
            build!(r, n, t, StoreNoOffset, |op| D::StoreNoOffset { opcode: op, args: [v, a], flags }, |ct, _d| {
                cur.ins().StoreNoOffset(op, ct, flags, v, a)
            })
        }
        I::StackLoad(r, n, t, s, off) => {
            let stack_slot = slot(*s)?;
            let offset = ir::immediates::Offset32::new(*off);
            build!(r, n, t, StackLoad, |op| D::StackLoad { opcode: op, stack_slot, offset }, |ct, _d| {
                cur.ins().StackLoad(op, ct, stack_slot, offset)
            })
        }
        I::StackStore(r, n, t, v, s, off) => {
            let v = vals.get(*v)?;
            let stack_slot = slot(*s)?;
            let offset = ir::immediates::Offset32::new(*off);
            build!(r, n, t, StackStore, |op| D::StackStore { opcode: op, arg: v, stack_slot, offset }, |ct, _d| {
                cur.ins().StackStore(op, ct, stack_slot, offset, v)
            })
        }
    }
    Ok(())
}
