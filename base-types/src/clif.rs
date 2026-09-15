//! The CLIF program an artifact carries, as data rather than text.
//!
//! Lean builds this shape directly and serializes it; [`crate::Setup`] carries
//! it; the runtime decodes it into `cranelift_codegen::ir::Function`. Nothing
//! in that path formats or parses CLIF source.
//!
//! # Wire format
//!
//! Serde's default enum representation, with tuple variants whose fields are in
//! the same order as the corresponding Lean constructor:
//!
//! ```json
//! {"Iadd": [12, 10, 11]}
//! {"Call": [null, 3, [4, 5]]}
//! ```
//!
//! `Val`, `BlockRef`, `SigRef` and `FnRef` are newtypes, so they appear as bare
//! numbers. Keeping the field order aligned with Lean's constructors is what
//! lets the emitter be a one-line-per-variant mapping rather than a schema.

use serde::{Deserialize, Serialize};

/// An SSA value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Val(pub u32);

/// A basic block.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockRef(pub u32);

/// A signature declared in the function prologue.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SigRef(pub u32);

/// A callee declared in the function prologue.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FnRef(pub u32);

/// The value types the DSL can name.
///
/// Deliberately smaller than Cranelift's set: these are the ones the generators
/// use, and an artifact naming anything else is a bug rather than a feature.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClifTy {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    F32x4,
    I8x16,
}

/// Integer comparison conditions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntCC {
    Eq,
    Ne,
    Uge,
    Ugt,
    Ule,
    Ult,
    Slt,
    Sle,
    Sgt,
    Sge,
}

/// Float comparison conditions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatCC {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Which load instruction, independent of the type it yields.
///
/// `Uload8`/`Uload32`/`Sload8` narrow-then-extend in one instruction; `Plain`
/// loads the result type directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadKind {
    Plain,
    Uload8,
    Uload32,
    Sload8,
}

/// A load: what to read, as what type, under which memory flags.
///
/// The generators emit eleven distinct combinations. Modelling the axes
/// separately rather than enumerating those eleven means adding a twelfth costs
/// nothing, and means the decoder cannot silently accept a spelling that has no
/// meaning.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadOp {
    pub kind: LoadKind,
    pub ty: ClifTy,
    /// `notrap aligned`. The vector and float loads set it; the integer loads
    /// do not, and that difference is load-bearing for what Cranelift may
    /// reorder.
    pub notrap_aligned: bool,
}

/// One CLIF instruction.
///
/// Field order matches the Lean `Inst` constructors exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Inst {
    /// `dst = iconst.ty value`
    Iconst(Val, ClifTy, i64),
    Iadd(Val, Val, Val),
    /// `dst = iadd_imm a, k` — one instruction where `iconst` + `iadd` is two,
    /// and the immediate reaches the machine addressing mode.
    IaddImm(Val, Val, i64),
    Isub(Val, Val, Val),
    Imul(Val, Val, Val),
    Udiv(Val, Val, Val),
    Ineg(Val, Val),
    Ishl(Val, Val, Val),
    Ushr(Val, Val, Val),
    Band(Val, Val, Val),
    BandNot(Val, Val, Val),
    Bor(Val, Val, Val),
    Bxor(Val, Val, Val),
    Ireduce32(Val, Val),
    Uextend64(Val, Val),
    Sextend64(Val, Val),
    /// `store val, addr+offset`
    ///
    /// The offset folds into the machine addressing mode, so computing
    /// `addr + k` with a separate `iadd` costs an instruction for nothing.
    Store(Val, Val, i32),
    Istore8(Val, Val, i32),
    Load(Val, LoadOp, Val, i32),
    Icmp(Val, IntCC, Val, Val),
    Select(Val, Val, Val, Val),
    Bitselect(Val, Val, Val, Val),
    /// `dst = call fnN(args)`, or no destination when the callee returns void.
    Call(Option<Val>, FnRef, Vec<Val>),
    Jump(BlockRef, Vec<Val>),
    /// `brif cond, then(args), else(args)`
    Brif(Val, BlockRef, Vec<Val>, BlockRef, Vec<Val>),
    Ret,
    /// `dst = f32const/f64const`, carrying the **bit pattern** rather than a
    /// literal spelling. A malformed float is unrepresentable rather than a
    /// parse error discovered at JIT time.
    Fconst(Val, ClifTy, u64),
    Fadd(Val, Val, Val),
    Fsub(Val, Val, Val),
    Fmul(Val, Val, Val),
    /// IEEE `maximumNumber`, not the hardware's `maxps`.
    Fmax(Val, Val, Val),
    /// IEEE `minimumNumber`, not the hardware's `minps`. Costs a NaN-correct
    /// sequence; use the `fcmp`/`bitselect` pair where the data cannot be NaN.
    Fmin(Val, Val, Val),
    Fpromote(Val, Val),
    Splat(Val, ClifTy, Val),
    Extractlane(Val, Val, u8),
    /// `store.ty notrap aligned val, addr+offset`
    StoreTyped(ClifTy, Val, Val, i32),
    Fneg(Val, Val),
    FcvtFromSint(Val, ClifTy, Val),
    FcvtToUint(Val, ClifTy, Val),
    Fcmp(Val, FloatCC, Val, Val),
    Bitcast(Val, ClifTy, Val),
    /// `dst = func_addr.i64 fnN` — materializes a callee's address without
    /// calling it, which is what forces the JIT to resolve the symbol.
    FuncAddr(Val, FnRef),
    Ctz(Val, Val),
    Popcnt(Val, Val),
    VhighBits(Val, Val),
}

/// A basic block: its parameters, then its instructions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub reference: BlockRef,
    pub params: Vec<(Val, ClifTy)>,
    pub insts: Vec<Inst>,
}

/// A signature in the function prologue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SigDecl {
    pub reference: SigRef,
    pub params: Vec<ClifTy>,
    pub result: Option<ClifTy>,
}

/// What a `fn` declaration names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Callee {
    /// A symbol resolved through the JIT's symbol table.
    Import(String),
    /// Another function of this same program, by its `u0:N` index. Kept
    /// distinct from `Import` because the two resolve by different means, and
    /// spelling a local call as a symbol name is how the text form lost that.
    Local(u32),
}

/// A callee in the function prologue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FnDecl {
    pub reference: FnRef,
    pub callee: Callee,
    pub sig: SigRef,
    /// Intra-module call. Imports resolved through the JIT's symbol table are
    /// not colocated.
    #[serde(default)]
    pub colocated: bool,
}

/// One function. Its signature is its entry block's parameter list, under
/// `system_v`, returning nothing — so the signature is not a separate field
/// that could disagree with the body. An entry point takes the memory base
/// pointer; a function reached through `cl_thread_spawn` takes its spawn
/// argument.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Function {
    /// The `u0:N` index. Must equal the function's position in
    /// [`Program::functions`]; the runtime resolves call targets by treating it
    /// as a `FuncId`.
    pub index: u32,
    pub sigs: Vec<SigDecl>,
    pub fns: Vec<FnDecl>,
    pub blocks: Vec<Block>,
}

/// Every function in one artifact, compiled as a unit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl Program {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}
