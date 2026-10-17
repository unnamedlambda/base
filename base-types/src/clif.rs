//! The CLIF program an artifact carries, as data rather than text.
//!
//! Lean builds this shape directly and serializes it; [`crate::Artifact`]
//! carries it; the runtime decodes it into `cranelift_codegen::ir::Function`.
//! Nothing in that path formats or parses CLIF source.
//!
//! # What is whose
//!
//! Everything here is either Cranelift's, spelled as CLIF spells it, or the
//! artifact's own. An instruction is a Cranelift instruction format and an
//! opcode of it; types, condition codes, atomic operations, memory flags and
//! parameter attributes are CLIF's names. A name is a string rather than
//! anything enumerated here, so what Cranelift has, an artifact can use, and
//! what it gains changes nothing here; and it has one spelling, CLIF's, which
//! the engine checks by printing back what it parsed. The artifact's own are
//! the positions things are found at, the arena, and the calling conventions.
//!
//! Where the order of a list means something — blocks are laid out in theirs,
//! instructions run in theirs — what is in it is named, as CLIF names blocks
//! and values, so a block can move without a branch to it changing. Where the
//! order means nothing — functions, primitives, stack slots — the position is
//! the name.
//!
//! Where Cranelift keeps a table in a function — the functions and signatures
//! it calls, the globals it reads, a block a branch passes arguments to — an
//! instruction carries the thing the entry stands for: the callee, the
//! signature, the global, the block and its arguments. The one table is the
//! artifact's: its primitives, each declared once.
//!
//! # Wire format
//!
//! Serde's data model, in the profile [`crate::Artifact`] describes: a struct
//! is a map of its fields, an instruction an array of its fields under its
//! format's name. Written as JSON, which has the same shapes, two read:
//!
//! ```json
//! {"Binary": [[12], "iadd", null, 10, 11]}
//! {"Call": [[], "call", null, {"Primitive": 3}, [4, 5]]}
//! ```
//!
//! `Val` and `BlockRef` are newtypes, so they appear as bare numbers, and a
//! type is CLIF's name for it, `"i64"`.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// An SSA value, by the name its function gives it: numbered as the writer
/// likes, and defined once.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Val(pub u32);

/// A block, by the name its function gives it: numbered as the writer likes,
/// and given to one block.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockRef(pub u32);

/// A value type, spelled as CLIF spells it: `"i64"`, `"f32"`, `"f32x4"`.
///
/// A name that is no type, or not CLIF's spelling of one (`"i32x04"`), is
/// refused at load.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClifTy(pub Cow<'static, str>);

impl ClifTy {
    pub const I8: ClifTy = ClifTy(Cow::Borrowed("i8"));
    pub const I16: ClifTy = ClifTy(Cow::Borrowed("i16"));
    pub const I32: ClifTy = ClifTy(Cow::Borrowed("i32"));
    pub const I64: ClifTy = ClifTy(Cow::Borrowed("i64"));
    pub const F32: ClifTy = ClifTy(Cow::Borrowed("f32"));
    pub const F64: ClifTy = ClifTy(Cow::Borrowed("f64"));
    pub const F32X4: ClifTy = ClifTy(Cow::Borrowed("f32x4"));
    pub const I8X16: ClifTy = ClifTy(Cow::Borrowed("i8x16"));
}

impl std::fmt::Display for ClifTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A parameter or return of a signature: its type, and CLIF's words for how
/// it is passed (`"uext"`, `"sext"`, `"sret"`, `"sarg(24)"`), in the order
/// CLIF prints them. A C function taking a `char` is called with `uext` or
/// `sext`; one returning a structure through a pointer takes it as `sret`,
/// which AArch64 passes in its own register.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbiParam(pub ClifTy, pub Vec<String>);

impl From<ClifTy> for AbiParam {
    fn from(t: ClifTy) -> Self {
        AbiParam(t, vec![])
    }
}

/// How a function is called. The artifact's own, because Cranelift's names
/// one per platform and an artifact is the same file on every one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Convention {
    /// The host's C convention: what a C library, the engine and the host
    /// itself are compiled to.
    Host,
    /// The architecture's one convention, whatever the OS: System V on x86-64,
    /// AAPCS64 on AArch64, Apple's included. Machine code a program carries is
    /// written to it, so its bytes are the same on every OS.
    Arch,
}

/// A function's signature, as Cranelift's: its convention, what it takes, and
/// what it returns.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signature {
    pub convention: Convention,
    pub params: Vec<AbiParam>,
    pub returns: Vec<AbiParam>,
}

impl Signature {
    /// What a host calls a function as, `BaseFunction`: the input and its
    /// length, the output and its length, answering a status.
    pub fn base_function() -> Signature {
        Signature {
            convention: Convention::Host,
            params: vec![ClifTy::I64.into(); 4],
            returns: vec![ClifTy::I64.into()],
        }
    }
}

/// A C function the program reaches the rest of the board through: the
/// library it is in, its symbol, and the signature to call it at.
///
/// Which file a library is on each operating system is not here: that is how
/// the system packages it, not what the program means, and the engine knows
/// it for each library it may open. A library or symbol that is not there
/// binds to a stub answering `-1`. An empty `symbol` is the presence probe,
/// answering `1` when the library is there and `0` when it is not.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub lib: String,
    pub symbol: String,
    pub signature: Signature,
}

/// What a direct call or a `func_addr` names: a function of the artifact or
/// one of its primitives, by position.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Callee {
    Function(u32),
    Primitive(u32),
}

/// A global an instruction reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Global {
    /// The arena, whose address `symbol_value` answers. The engine places it
    /// when it loads the program.
    Arena,
}

/// A branch's destination: the block and the arguments it is passed.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockCall(pub BlockRef, pub Vec<Val>);

/// One CLIF instruction: a Cranelift instruction format, and its fields.
///
/// Every format starts the same way. `results` names every result the
/// instruction has, as CLIF text does, and none for one that has none (a
/// store, a branch). The opcode is CLIF's name and must be of the format. The
/// type is the instruction's controlling type, named only where CLIF does not
/// take it from an operand (`iconst`, `uextend`, a load); one where it does, or
/// a type where the opcode has none, is refused, so an instruction has one
/// spelling. The format's own fields follow, in the order Cranelift's format
/// definition gives them.
///
/// Memory flags are CLIF's words (`["aligned"]`), in the order CLIF prints
/// them. Every access is also `notrap`, which the engine sets and a program
/// does not state: nothing of it catches a fault, so a faulting access ends
/// the process whether or not Cranelift was told it could.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Inst {
    Unary(Vec<Val>, String, Option<ClifTy>, Val),
    UnaryImm(Vec<Val>, String, Option<ClifTy>, i64),
    /// An `f32` immediate, as its bits.
    UnaryIeee32(Vec<Val>, String, Option<ClifTy>, u32),
    /// An `f64` immediate, as its bits.
    UnaryIeee64(Vec<Val>, String, Option<ClifTy>, u64),
    UnaryGlobalValue(Vec<Val>, String, Option<ClifTy>, Global),
    Binary(Vec<Val>, String, Option<ClifTy>, Val, Val),
    BinaryImm8(Vec<Val>, String, Option<ClifTy>, Val, u8),
    BinaryImm64(Vec<Val>, String, Option<ClifTy>, Val, i64),
    Ternary(Vec<Val>, String, Option<ClifTy>, Val, Val, Val),
    MultiAry(Vec<Val>, String, Option<ClifTy>, Vec<Val>),
    NullAry(Vec<Val>, String, Option<ClifTy>),
    /// The condition, then the operands.
    IntCompare(Vec<Val>, String, Option<ClifTy>, String, Val, Val),
    FloatCompare(Vec<Val>, String, Option<ClifTy>, String, Val, Val),
    Jump(Vec<Val>, String, Option<ClifTy>, BlockCall),
    /// The condition, then where to go when it is not zero and when it is.
    Brif(Vec<Val>, String, Option<ClifTy>, Val, BlockCall, BlockCall),
    Call(Vec<Val>, String, Option<ClifTy>, Callee, Vec<Val>),
    /// The signature, then the callee's address, then the arguments. The
    /// callee is whatever is at the address: a host's callback, a C function a
    /// host handed over, the program's own function by its `func_addr`, or
    /// machine code the program placed. Nothing of the engine's runs between
    /// the caller and it.
    CallIndirect(Vec<Val>, String, Option<ClifTy>, Signature, Val, Vec<Val>),
    FuncAddr(Vec<Val>, String, Option<ClifTy>, Callee),
    /// The flags, the operation, the address, the operand.
    AtomicRmw(Vec<Val>, String, Option<ClifTy>, Vec<String>, String, Val, Val),
    /// The flags, the address, the expected value, the replacement.
    AtomicCas(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val, Val, Val),
    /// The flags, the address, the offset, which folds into the machine
    /// addressing mode.
    Load(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val, i32),
    LoadNoOffset(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val),
    /// The flags, the value, the address, the offset.
    Store(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val, Val, i32),
    StoreNoOffset(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val, Val),
    /// The stack slot, by position, and the offset into it.
    StackLoad(Vec<Val>, String, Option<ClifTy>, u32, i32),
    /// The value, the stack slot, the offset into it.
    StackStore(Vec<Val>, String, Option<ClifTy>, Val, u32, i32),
}

/// A basic block: its name, its parameters, then its instructions. The
/// function lays its blocks out in the order it lists them, the first being
/// the entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub reference: BlockRef,
    pub params: Vec<(Val, ClifTy)>,
    pub insts: Vec<Inst>,
}

/// Memory a function has on its own stack (CLIF's `explicit_slot`): `size`
/// bytes aligned to `align`, a power of two.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackSlot {
    pub size: u32,
    pub align: u32,
}

/// One function: what it is called by, and what it does.
///
/// Its signature is CLIF's, stated as CLIF's function header states it, and
/// its entry block takes what the signature does. A named function's signature
/// is [`Signature::base_function`], since a host calls it. A function started on a thread
/// (`cl_thread_start`) takes the one pointer it is handed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    /// The name a host calls this function by, if a host calls it.
    ///
    /// A function's position in the artifact is the generator's to choose and
    /// free to change; a name is what stays put. A function without one is the
    /// program's own, reached only by its other functions. Names are unique
    /// within an artifact.
    pub name: Option<String>,
    pub signature: Signature,
    pub stack_slots: Vec<StackSlot>,
    pub blocks: Vec<Block>,
}
