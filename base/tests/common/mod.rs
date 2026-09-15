//! Constructing CLIF programs for tests.
//!
//! These tests are about FFI linkage — that the JIT resolves a symbol, calls it
//! with the arguments the program named, and gets the result back. The program
//! itself is scaffolding, so the point of these helpers is that the scaffolding
//! stays as legible as the CLIF text it replaces.

#![allow(dead_code)]

pub use base_types::clif::*;

pub const I8: ClifTy = ClifTy::I8;
pub const I16: ClifTy = ClifTy::I16;
pub const I32: ClifTy = ClifTy::I32;
pub const I64: ClifTy = ClifTy::I64;
pub const F32: ClifTy = ClifTy::F32;
pub const F64: ClifTy = ClifTy::F64;
pub const F32X4: ClifTy = ClifTy::F32x4;
pub const I8X16: ClifTy = ClifTy::I8x16;

/// `vN`
pub fn v(n: u32) -> Val {
    Val(n)
}

/// The caller's input buffer, as the entry block receives it.
pub fn data_ptr() -> Val {
    v(9500)
}

/// How many bytes of input the caller supplied.
pub fn data_len() -> Val {
    v(9501)
}

/// The caller's output buffer.
pub fn out_ptr() -> Val {
    v(9502)
}

/// How much room the caller left for the answer.
pub fn out_len() -> Val {
    v(9503)
}

/// `blockN`
pub fn b(n: u32) -> BlockRef {
    BlockRef(n)
}

/// Accumulates one function. `index` is its `u0:N`, and must match its position
/// in the program — the runtime resolves call targets by treating it as a
/// `FuncId`.
pub struct Func {
    inner: Function,
}

pub fn function(index: u32) -> Func {
    Func {
        inner: Function { index, sigs: vec![], fns: vec![], blocks: vec![] },
    }
}

impl Func {
    /// `sigN = (params) -> result system_v`
    pub fn sig(mut self, n: u32, params: &[ClifTy], result: Option<ClifTy>) -> Self {
        self.inner.sigs.push(SigDecl {
            reference: SigRef(n),
            params: params.to_vec(),
            result,
        });
        self
    }

    /// `fnN = %name sigS`
    pub fn import(mut self, n: u32, name: &str, sig: u32) -> Self {
        self.inner.fns.push(FnDecl {
            reference: FnRef(n),
            callee: Callee::Import(name.to_string()),
            sig: SigRef(sig),
        });
        self
    }

    /// `fnN = colocated u0:I sigS` — a call to another function of this program.
    pub fn local(mut self, n: u32, index: u32, sig: u32) -> Self {
        self.inner.fns.push(FnDecl {
            reference: FnRef(n),
            callee: Callee::Local(index),
            sig: SigRef(sig),
        });
        self
    }

    /// `block0(v0, v9500, v9501, v9502, v9503: i64):` — the memory base
    /// pointer, then the caller's input buffer and its length and the caller's
    /// output buffer and its length, which is what an entry point is called
    /// with.
    ///
    /// The four are numbered clear of everything a test body uses, so a body
    /// that wants one names it with [`data_ptr`] and the rest keep low ids.
    pub fn entry(self, insts: Vec<Inst>) -> Self {
        self.block(
            0,
            &[(v(0), I64), (data_ptr(), I64), (data_len(), I64), (out_ptr(), I64), (out_len(), I64)],
            insts,
        )
    }

    /// `block0(v0: i64):` — one pointer, and not the arena base: what
    /// `cl_thread_spawn` hands a worker is whatever the spawning program chose.
    /// A function reached that way is not an entry point and says so here.
    pub fn entry_spawned(self, insts: Vec<Inst>) -> Self {
        self.block(0, &[(v(0), I64)], insts)
    }

    pub fn block(mut self, n: u32, params: &[(Val, ClifTy)], insts: Vec<Inst>) -> Self {
        self.inner.blocks.push(Block {
            reference: BlockRef(n),
            params: params.to_vec(),
            insts,
        });
        self
    }
}

/// One function, the common case.
pub fn program(f: Func) -> Program {
    Program { functions: vec![f.inner] }
}

/// Several functions, in `u0:N` order.
pub fn programs(fs: Vec<Func>) -> Program {
    Program { functions: fs.into_iter().map(|f| f.inner).collect() }
}

/// `u0:0` doing nothing — the slot generated artifacts reserve so that the
/// interesting function is `u0:1`.
pub fn noop(index: u32) -> Func {
    function(index).entry(vec![Inst::Ret])
}

// --- instructions -----------------------------------------------------------
//
// One name per instruction. Each is the constructor with a shorter name, so the
// tests read as CLIF and rustc still checks every operand.

pub fn iconst(d: Val, ty: ClifTy, k: i64) -> Inst { Inst::Iconst(d, ty, k) }
pub fn iconst64(d: Val, k: i64) -> Inst { Inst::Iconst(d, I64, k) }
pub fn iconst32(d: Val, k: i64) -> Inst { Inst::Iconst(d, I32, k) }
pub fn iadd(d: Val, a: Val, b: Val) -> Inst { Inst::Iadd(d, a, b) }
pub fn iadd_imm(d: Val, a: Val, k: i64) -> Inst { Inst::IaddImm(d, a, k) }
pub fn isub(d: Val, a: Val, b: Val) -> Inst { Inst::Isub(d, a, b) }
pub fn imul(d: Val, a: Val, b: Val) -> Inst { Inst::Imul(d, a, b) }
pub fn udiv(d: Val, a: Val, b: Val) -> Inst { Inst::Udiv(d, a, b) }
pub fn ineg(d: Val, a: Val) -> Inst { Inst::Ineg(d, a) }
pub fn ishl(d: Val, a: Val, b: Val) -> Inst { Inst::Ishl(d, a, b) }
pub fn ushr(d: Val, a: Val, b: Val) -> Inst { Inst::Ushr(d, a, b) }
pub fn band(d: Val, a: Val, b: Val) -> Inst { Inst::Band(d, a, b) }
pub fn band_not(d: Val, a: Val, b: Val) -> Inst { Inst::BandNot(d, a, b) }
pub fn bor(d: Val, a: Val, b: Val) -> Inst { Inst::Bor(d, a, b) }
pub fn bxor(d: Val, a: Val, b: Val) -> Inst { Inst::Bxor(d, a, b) }
pub fn ireduce32(d: Val, a: Val) -> Inst { Inst::Ireduce32(d, a) }
pub fn uextend64(d: Val, a: Val) -> Inst { Inst::Uextend64(d, a) }
pub fn sextend64(d: Val, a: Val) -> Inst { Inst::Sextend64(d, a) }

pub fn store(val: Val, addr: Val, off: i32) -> Inst { Inst::Store(val, addr, off) }
pub fn istore8(val: Val, addr: Val, off: i32) -> Inst { Inst::Istore8(val, addr, off) }
pub fn store_typed(ty: ClifTy, val: Val, addr: Val, off: i32) -> Inst {
    Inst::StoreTyped(ty, val, addr, off)
}

pub fn icmp(d: Val, cc: IntCC, a: Val, b: Val) -> Inst { Inst::Icmp(d, cc, a, b) }
pub fn fcmp(d: Val, cc: FloatCC, a: Val, b: Val) -> Inst { Inst::Fcmp(d, cc, a, b) }
pub fn select(d: Val, c: Val, a: Val, b: Val) -> Inst { Inst::Select(d, c, a, b) }
pub fn bitselect(d: Val, c: Val, a: Val, b: Val) -> Inst { Inst::Bitselect(d, c, a, b) }

/// `dst = call fnN(args)`, or `call fnN(args)` when the callee returns nothing.
pub fn call(d: Option<Val>, f: u32, args: &[Val]) -> Inst {
    Inst::Call(d, FnRef(f), args.to_vec())
}
/// `dst = func_addr.i64 fnN`
pub fn func_addr(d: Val, f: u32) -> Inst { Inst::FuncAddr(d, FnRef(f)) }
pub fn jump(t: u32, args: &[Val]) -> Inst { Inst::Jump(BlockRef(t), args.to_vec()) }
pub fn brif(c: Val, t: u32, ta: &[Val], e: u32, ea: &[Val]) -> Inst {
    Inst::Brif(c, BlockRef(t), ta.to_vec(), BlockRef(e), ea.to_vec())
}
pub fn ret() -> Inst { Inst::Ret }

pub fn fadd(d: Val, a: Val, b: Val) -> Inst { Inst::Fadd(d, a, b) }
pub fn fsub(d: Val, a: Val, b: Val) -> Inst { Inst::Fsub(d, a, b) }
pub fn fmul(d: Val, a: Val, b: Val) -> Inst { Inst::Fmul(d, a, b) }
pub fn fmax(d: Val, a: Val, b: Val) -> Inst { Inst::Fmax(d, a, b) }
pub fn fmin(d: Val, a: Val, b: Val) -> Inst { Inst::Fmin(d, a, b) }
pub fn fneg(d: Val, a: Val) -> Inst { Inst::Fneg(d, a) }
pub fn fpromote(d: Val, a: Val) -> Inst { Inst::Fpromote(d, a) }
pub fn fcvt_from_sint(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::FcvtFromSint(d, ty, s) }
pub fn fcvt_to_uint(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::FcvtToUint(d, ty, s) }
pub fn splat(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::Splat(d, ty, s) }
pub fn extractlane(d: Val, s: Val, lane: u8) -> Inst { Inst::Extractlane(d, s, lane) }
pub fn bitcast(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::Bitcast(d, ty, s) }
pub fn ctz(d: Val, a: Val) -> Inst { Inst::Ctz(d, a) }
pub fn popcnt(d: Val, a: Val) -> Inst { Inst::Popcnt(d, a) }
pub fn vhigh_bits(d: Val, a: Val) -> Inst { Inst::VhighBits(d, a) }

// --- loads -----------------------------------------------------------------
//
// The only instruction with helpers. `LoadOp` is three independent axes and no
// default suffices: `load64` and `load_trusted` differ in whether Cranelift may
// reorder the access. These are the same eleven combinations `IR.lean` names on
// the Lean side. Every other instruction is constructed directly.

fn load_op(kind: LoadKind, ty: ClifTy, notrap_aligned: bool) -> LoadOp {
    LoadOp { kind, ty, notrap_aligned }
}

/// `dst = load.ty addr+off`
pub fn load(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst {
    Inst::Load(d, load_op(LoadKind::Plain, ty, false), addr, off)
}
pub fn load64(d: Val, addr: Val, off: i32) -> Inst {
    load(d, I64, addr, off)
}
pub fn load32(d: Val, addr: Val, off: i32) -> Inst {
    load(d, I32, addr, off)
}
/// `dst = load.ty notrap aligned addr+off`
pub fn load_trusted(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst {
    Inst::Load(d, load_op(LoadKind::Plain, ty, true), addr, off)
}
pub fn uload8(d: Val, addr: Val, off: i32) -> Inst {
    Inst::Load(d, load_op(LoadKind::Uload8, I64, false), addr, off)
}
pub fn uload32(d: Val, addr: Val, off: i32) -> Inst {
    Inst::Load(d, load_op(LoadKind::Uload32, I64, false), addr, off)
}
pub fn sload8(d: Val, addr: Val, off: i32) -> Inst {
    Inst::Load(d, load_op(LoadKind::Sload8, I64, false), addr, off)
}

/// Float constants carry bits, so a literal that cannot be spelled is a
/// compile error here rather than a JIT-time parse failure.
pub fn f32const(d: Val, x: f32) -> Inst {
    Inst::Fconst(d, F32, x.to_bits() as u64)
}
pub fn f64const(d: Val, x: f64) -> Inst {
    Inst::Fconst(d, F64, x.to_bits())
}

// --- reading a program ------------------------------------------------------

/// Prints the program as CLIF when `BASE_DUMP_CLIF` is set.
///
/// Nothing asserts on this text: it exists so a test can be read as the CLIF it
/// actually compiles. Because there is no expected value, a change in how
/// Cranelift renders IR changes what is printed and breaks nothing.
pub fn dump(prog: &Program) {
    if std::env::var_os("BASE_DUMP_CLIF").is_none() || prog.is_empty() {
        return;
    }
    let test = std::thread::current().name().unwrap_or("?").to_string();
    match base::clif_text(prog) {
        Ok(text) => println!("=== {test} ===\n{text}"),
        Err(e) => println!("=== {test} === does not decode: {e}"),
    }
}
