//! Building artifacts for tests: one helper per instruction, spelled as CLIF.

#![allow(dead_code)]

use std::cell::RefCell;

use base::Driver;
pub use base_types::clif::*;
use base_types::{Artifact, Segment};

pub const I32: ClifTy = ClifTy::I32;
pub const I64: ClifTy = ClifTy::I64;
pub const F64: ClifTy = ClifTy::F64;
pub const F32X4: ClifTy = ClifTy::F32X4;
pub const I8X16: ClifTy = ClifTy::I8X16;

pub fn v(n: u32) -> Val {
    Val(n)
}

/// The four parameters of a function a host calls: input, its length, output,
/// its length. Numbered clear of what test bodies use.
pub fn input() -> Val {
    v(9500)
}
pub fn input_len() -> Val {
    v(9501)
}
pub fn output() -> Val {
    v(9502)
}
pub fn output_len() -> Val {
    v(9503)
}

pub struct Func(Function);

pub fn function() -> Func {
    Func(Function {
        name: None,
        signature: Signature { convention: Convention::Host, params: vec![], returns: vec![] },
        stack_slots: vec![],
        blocks: vec![],
    })
}

impl Func {
    pub fn named(mut self, name: &str) -> Self {
        self.0.name = Some(name.to_string());
        self
    }

    /// A function a host calls: block0 takes the four host parameters, and `v0`
    /// is the arena's address.
    pub fn entry(self, insts: Vec<Inst>) -> Self {
        let mut body = vec![arena(v(0))];
        body.extend(insts);
        self.block(0, &[(input(), I64), (input_len(), I64), (output(), I64), (output_len(), I64)], body)
    }

    pub fn block(mut self, n: u32, params: &[(Val, ClifTy)], insts: Vec<Inst>) -> Self {
        self.0.blocks.push(Block { reference: BlockRef(n), params: params.to_vec(), insts });
        self
    }
}

/// The functions, each at the signature its body has. A function a host calls
/// answers a status, `0` where its body returns nothing.
pub fn functions(fs: Vec<Func>) -> Vec<Function> {
    fs.into_iter()
        .map(|Func(mut f)| {
            if f.blocks.first().is_some_and(|b| b.params.first() == Some(&(input(), I64))) {
                f.signature = Signature::base_function();
                let mut fresh = 9600;
                for b in &mut f.blocks {
                    for i in std::mem::take(&mut b.insts) {
                        if i == ret() {
                            b.insts.extend([iconst64(v(fresh), 0), ret_status(v(fresh))]);
                            fresh += 1;
                        } else {
                            b.insts.push(i);
                        }
                    }
                }
            } else {
                let answers = f.blocks.iter().flat_map(|b| &b.insts).any(|i| matches!(i, Inst::MultiAry(_, op, _, a) if op == "return" && !a.is_empty()));
                f.signature.params = f.blocks.first().map(|b| b.params.iter().map(|(_, t)| t.clone().into()).collect()).unwrap_or_default();
                f.signature.returns = if answers { vec![I64.into()] } else { vec![] };
            }
            f
        })
        .collect()
}

thread_local! {
    /// The primitives the running test has named, in first-named order.
    static PRIMITIVES: RefCell<Vec<Primitive>> = const { RefCell::new(Vec::new()) };
}

pub fn primitive(p: Primitive) -> Callee {
    PRIMITIVES.with_borrow_mut(|ps| {
        let i = ps.iter().position(|q| *q == p).unwrap_or_else(|| {
            ps.push(p);
            ps.len() - 1
        });
        Callee::Primitive(i as u32)
    })
}

/// An artifact of `fs` in an arena of `size` bytes starting as `data`,
/// declaring every primitive the test has named.
pub fn artifact(fs: Vec<Function>, size: u64, data: Vec<Segment>) -> Artifact {
    Artifact { primitives: PRIMITIVES.with_borrow(|ps| ps.clone()), functions: fs, arena_size: size, data }
}

/// One function named `main`, in a 4 KiB arena.
pub fn main_of(insts: Vec<Inst>) -> Artifact {
    artifact(functions(vec![function().named("main").entry(insts)]), 4096, vec![])
}

pub fn driver(a: Artifact) -> Driver {
    unsafe { Driver::load(&a.to_bytes()).unwrap() }
}

/// Calls `name` with `input` and `output`, answering its status.
pub fn call_with(d: &Driver, name: &str, input: &[u8], output: &mut [u8]) -> i64 {
    d.function(name).unwrap()(input, output)
}

/// The `n` bytes of the arena from `offset`.
pub fn read(d: &Driver, offset: usize, n: usize) -> Vec<u8> {
    d.arena()[offset..offset + n].iter().map(|b| b.load(std::sync::atomic::Ordering::Relaxed)).collect()
}

pub fn read_i64(d: &Driver, offset: usize) -> i64 {
    i64::from_le_bytes(read(d, offset, 8).try_into().unwrap())
}

// --- instructions -------------------------------------------------------------

fn un(d: Val, op: &str, a: Val) -> Inst { Inst::Unary(vec![d], op.into(), None, a) }
fn un_at(d: Val, op: &str, ty: ClifTy, a: Val) -> Inst { Inst::Unary(vec![d], op.into(), Some(ty), a) }
fn bin(d: Val, op: &str, a: Val, b: Val) -> Inst { Inst::Binary(vec![d], op.into(), None, a, b) }
fn ter(d: Val, op: &str, a: Val, b: Val, c: Val) -> Inst { Inst::Ternary(vec![d], op.into(), None, a, b, c) }
fn aligned(yes: bool) -> Vec<String> { if yes { vec!["aligned".into()] } else { vec![] } }

pub fn arena(d: Val) -> Inst { Inst::UnaryGlobalValue(vec![d], "symbol_value".into(), Some(I64), Global::Arena) }
pub fn iconst(d: Val, ty: ClifTy, k: i64) -> Inst { Inst::UnaryImm(vec![d], "iconst".into(), Some(ty), k) }
pub fn iconst64(d: Val, k: i64) -> Inst { iconst(d, I64, k) }
pub fn iconst32(d: Val, k: i64) -> Inst { iconst(d, I32, k) }
pub fn iadd(d: Val, a: Val, b: Val) -> Inst { bin(d, "iadd", a, b) }
pub fn iadd_imm(d: Val, a: Val, k: i64) -> Inst { Inst::BinaryImm64(vec![d], "iadd_imm".into(), None, a, k) }
pub fn imul(d: Val, a: Val, b: Val) -> Inst { bin(d, "imul", a, b) }
pub fn udiv(d: Val, a: Val, b: Val) -> Inst { bin(d, "udiv", a, b) }
pub fn ineg(d: Val, a: Val) -> Inst { un(d, "ineg", a) }
pub fn band(d: Val, a: Val, b: Val) -> Inst { bin(d, "band", a, b) }
pub fn band_not(d: Val, a: Val, b: Val) -> Inst { bin(d, "band_not", a, b) }
pub fn bor(d: Val, a: Val, b: Val) -> Inst { bin(d, "bor", a, b) }
pub fn bxor(d: Val, a: Val, b: Val) -> Inst { bin(d, "bxor", a, b) }
pub fn ireduce32(d: Val, a: Val) -> Inst { un_at(d, "ireduce", I32, a) }
pub fn uextend64(d: Val, a: Val) -> Inst { un_at(d, "uextend", I64, a) }
pub fn sextend64(d: Val, a: Val) -> Inst { un_at(d, "sextend", I64, a) }
pub fn icmp(d: Val, cc: &str, a: Val, b: Val) -> Inst { Inst::IntCompare(vec![d], "icmp".into(), None, cc.into(), a, b) }
pub fn fcmp(d: Val, cc: &str, a: Val, b: Val) -> Inst { Inst::FloatCompare(vec![d], "fcmp".into(), None, cc.into(), a, b) }
pub fn select(d: Val, c: Val, a: Val, b: Val) -> Inst { ter(d, "select", c, a, b) }
pub fn bitselect(d: Val, c: Val, a: Val, b: Val) -> Inst { ter(d, "bitselect", c, a, b) }
pub fn ctz(d: Val, a: Val) -> Inst { un(d, "ctz", a) }
pub fn popcnt(d: Val, a: Val) -> Inst { un(d, "popcnt", a) }
pub fn fadd(d: Val, a: Val, b: Val) -> Inst { bin(d, "fadd", a, b) }
pub fn fsub(d: Val, a: Val, b: Val) -> Inst { bin(d, "fsub", a, b) }
pub fn fmul(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmul", a, b) }
pub fn fmax(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmax", a, b) }
pub fn fmin(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmin", a, b) }
pub fn fneg(d: Val, a: Val) -> Inst { un(d, "fneg", a) }
pub fn fpromote(d: Val, a: Val) -> Inst { un_at(d, "fpromote", F64, a) }
pub fn fcvt_from_sint(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "fcvt_from_sint", ty, s) }
pub fn fcvt_to_uint(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "fcvt_to_uint_sat", ty, s) }
pub fn splat(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "splat", ty, s) }
pub fn extractlane(d: Val, s: Val, lane: u8) -> Inst { Inst::BinaryImm8(vec![d], "extractlane".into(), None, s, lane) }
pub fn bitcast(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::LoadNoOffset(vec![d], "bitcast".into(), Some(ty), vec![], s) }
pub fn vhigh_bits(d: Val, a: Val) -> Inst { un_at(d, "vhigh_bits", I32, a) }
pub fn f32const(d: Val, x: f32) -> Inst { Inst::UnaryIeee32(vec![d], "f32const".into(), None, x.to_bits()) }
pub fn f64const(d: Val, x: f64) -> Inst { Inst::UnaryIeee64(vec![d], "f64const".into(), None, x.to_bits()) }
pub fn atomic_add(d: Val, addr: Val, x: Val) -> Inst {
    Inst::AtomicRmw(vec![d], "atomic_rmw".into(), Some(I64), vec![], "add".into(), addr, x)
}

pub fn load(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst { Inst::Load(vec![d], "load".into(), Some(ty), vec![], addr, off) }
pub fn load64(d: Val, addr: Val, off: i32) -> Inst { load(d, I64, addr, off) }
pub fn load_aligned(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst {
    Inst::Load(vec![d], "load".into(), Some(ty), aligned(true), addr, off)
}
pub fn store(val: Val, addr: Val, off: i32) -> Inst { Inst::Store(vec![], "store".into(), None, vec![], val, addr, off) }
pub fn istore8(val: Val, addr: Val, off: i32) -> Inst { Inst::Store(vec![], "istore8".into(), None, vec![], val, addr, off) }
pub fn store_aligned(val: Val, addr: Val, off: i32) -> Inst {
    Inst::Store(vec![], "store".into(), None, aligned(true), val, addr, off)
}

pub fn sig(convention: Convention, params: &[ClifTy], result: Option<ClifTy>) -> Signature {
    Signature { convention, params: params.iter().map(|t| t.clone().into()).collect(), returns: result.into_iter().map(Into::into).collect() }
}

/// One of base's own functions, at the signature given.
pub fn base_fn(symbol: &str, params: &[ClifTy], result: Option<ClifTy>) -> Callee {
    lib_fn("base", symbol, params, result)
}

/// A function of one of the engine's libraries, at the signature given.
pub fn lib_fn(lib: &str, symbol: &str, params: &[ClifTy], result: Option<ClifTy>) -> Callee {
    primitive(Primitive { lib: lib.into(), symbol: symbol.into(), signature: sig(Convention::Host, params, result) })
}

/// One of base's own functions, at the signature Lean declares it at.
pub fn base_primitive(name: &str) -> Callee {
    let (params, result): (&[ClifTy], _) = match name {
        "cl_native_load" => (&[I64; 2], Some(I64)),
        "cl_native_free" => (&[I64], None),
        "cl_native_arch" => (&[], Some(I32)),
        _ => panic!("no signature here for {name}"),
    };
    base_fn(name, params, result)
}

pub fn local(index: u32) -> Callee {
    Callee::Function(index)
}

pub fn call(d: Option<Val>, c: Callee, args: &[Val]) -> Inst {
    Inst::Call(d.into_iter().collect(), "call".into(), None, c, args.to_vec())
}
pub fn call_indirect(d: Option<Val>, s: Signature, addr: Val, args: &[Val]) -> Inst {
    Inst::CallIndirect(d.into_iter().collect(), "call_indirect".into(), None, s, addr, args.to_vec())
}
pub fn jump(t: u32, args: &[Val]) -> Inst { Inst::Jump(vec![], "jump".into(), None, BlockCall(BlockRef(t), args.to_vec())) }
pub fn brif(c: Val, t: u32, ta: &[Val], e: u32, ea: &[Val]) -> Inst {
    Inst::Brif(vec![], "brif".into(), None, c, BlockCall(BlockRef(t), ta.to_vec()), BlockCall(BlockRef(e), ea.to_vec()))
}
pub fn ret() -> Inst { Inst::MultiAry(vec![], "return".into(), None, vec![]) }
pub fn ret_status(v: Val) -> Inst { Inst::MultiAry(vec![], "return".into(), None, vec![v]) }
