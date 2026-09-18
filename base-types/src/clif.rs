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
//! `Val`, `BlockRef` and `FnRef` are newtypes, so they appear as bare numbers.
//! Keeping the field order aligned with Lean's constructors is what lets the
//! emitter be a one-line-per-variant mapping rather than a schema.
//!
//! A reference is an id, not a position: a compiler allocates blocks and
//! callees it goes on to drop, so what ships is a selection whose numbering has
//! gaps. A reference naming nothing is refused at load.

use serde::{Deserialize, Serialize};

/// An SSA value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Val(pub u32);

/// A basic block.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockRef(pub u32);

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
#[serde(deny_unknown_fields)]
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
    /// `return v`, or `return` when a function answers nothing. A function
    /// that answers returns an `i64` — the status its caller reads — and what
    /// it means is between that caller and whoever built the program.
    Ret(Option<Val>),
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
///
/// It carries its own reference because the ids are not positions: a compiler
/// allocates a block it goes on to drop, so what ships is a list with gaps in
/// its numbering, and a branch names the id rather than the place.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub reference: BlockRef,
    pub params: Vec<(Val, ClifTy)>,
    pub insts: Vec<Inst>,
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

/// A callee the body may call, under the reference it names it by.
///
/// No signature travels with it: an import's is the one base's table provides,
/// and a local's is read off the callee's own entry block. Both are recovered
/// at load, so a declaration cannot describe a callee in a way the callee
/// disagrees with — the rule this format already applies to a function's own
/// signature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FnDecl {
    pub reference: FnRef,
    pub callee: Callee,
}

/// One function: what it is called by, what it may call, and what it does.
///
/// Its signature is read off the body: the entry block's parameters are what it
/// takes, and whether its `Ret` carries a value is whether it answers an `i64`.
/// Under the host's C calling convention. So the signature is not a separate
/// field that could disagree with the body — and neither is a callee's, which
/// is why `callees` says only what to call and not how. An entry point takes
/// the memory base, the input buffer and its length, and the output buffer and
/// its length; a function reached through `cl_thread_spawn` takes its spawn
/// argument.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    /// The name a host calls this function by, if it is an entry point.
    ///
    /// A function's `u0:N` is its position in the artifact, which is the
    /// generator's to choose and free to change; a name is what stays put. A
    /// function without one is the program's own, reached only by its other
    /// functions. Names are unique within an artifact, and a named function
    /// has to be shaped like an entry point.
    pub export_name: Option<String>,
    pub fns: Vec<FnDecl>,
    pub blocks: Vec<Block>,
}
