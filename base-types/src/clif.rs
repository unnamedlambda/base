//! The CLIF program an artifact carries, as data.
//!
//! Each part is Cranelift's, spelled as CLIF spells it (types, opcodes,
//! conditions, flags), or the artifact's own (positions, the arena, calling
//! conventions). A function, primitive or stack slot is named by its position;
//! blocks and values, whose order means something, are named.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// An SSA value, defined once in its function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Val(pub u32);

/// A block's name, unique in its function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockRef(pub u32);

/// A value type, as CLIF spells it: `"i64"`, `"f32x4"`.
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

/// A parameter or return: its type, and CLIF's attributes for how it is passed
/// (`"uext"`, `"sret"`), in CLIF's order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbiParam(pub ClifTy, pub Vec<String>);

impl From<ClifTy> for AbiParam {
    fn from(t: ClifTy) -> Self {
        AbiParam(t, vec![])
    }
}

/// How a function is called.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Convention {
    /// The host's C convention.
    Host,
    /// The architecture's convention on every OS: System V, or AAPCS64.
    Arch,
}

/// A function's signature.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signature {
    pub convention: Convention,
    pub params: Vec<AbiParam>,
    pub returns: Vec<AbiParam>,
}

impl Signature {
    /// `BaseFunction`, what a host calls: input, its length, output, its length; a status.
    pub fn base_function() -> Signature {
        Signature {
            convention: Convention::Host,
            params: vec![ClifTy::I64.into(); 4],
            returns: vec![ClifTy::I64.into()],
        }
    }
}

/// A C function the program calls: its library, its symbol, its signature.
/// A missing one answers `-1`; an empty `symbol` answers whether the library
/// is present.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub lib: String,
    pub symbol: String,
    pub signature: Signature,
}

/// A function of the artifact or one of its primitives, by position.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Callee {
    Function(u32),
    Primitive(u32),
}

/// A global an instruction reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Global {
    /// The arena; `symbol_value` answers its address.
    Arena,
}

/// A branch's destination and its arguments.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockCall(pub BlockRef, pub Vec<Val>);

/// One instruction, as a Cranelift instruction format: the values it defines
/// (all of them), the opcode, the controlling type where CLIF names one rather
/// than taking it from an operand, then the format's fields in Cranelift's
/// order. Memory flags are CLIF's words; `notrap` is the engine's to set.
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
    /// The signature, the callee's address, the arguments.
    CallIndirect(Vec<Val>, String, Option<ClifTy>, Signature, Val, Vec<Val>),
    FuncAddr(Vec<Val>, String, Option<ClifTy>, Callee),
    /// The flags, the operation, the address, the operand.
    AtomicRmw(Vec<Val>, String, Option<ClifTy>, Vec<String>, String, Val, Val),
    /// The flags, the address, the expected value, the replacement.
    AtomicCas(Vec<Val>, String, Option<ClifTy>, Vec<String>, Val, Val, Val),
    /// The flags, the address, the offset.
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

/// A basic block. Blocks are laid out in order; the first is the entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub reference: BlockRef,
    pub params: Vec<(Val, ClifTy)>,
    pub insts: Vec<Inst>,
}

/// An `explicit_slot` of `size` bytes aligned to `align`, a power of two.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackSlot {
    pub size: u32,
    pub align: u32,
}

/// One function. A named one is called by a host, at
/// [`Signature::base_function`]; names are unique in an artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub name: Option<String>,
    pub signature: Signature,
    pub stack_slots: Vec<StackSlot>,
    pub blocks: Vec<Block>,
}
