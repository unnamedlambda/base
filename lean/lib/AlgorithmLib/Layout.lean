import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace Layout

/-- Field types with statically known sizes -/
inductive FieldTy where
  | u8                          -- 1 byte
  | i32                         -- 4 bytes
  | i64                         -- 8 bytes
  | bytes (n : Nat)             -- fixed-size byte region
  deriving Repr, BEq

/-- Size in bytes of a field type -/
def FieldTy.size : FieldTy → Nat
  | .u8 => 1
  | .i32 => 4
  | .i64 => 8
  | .bytes n => n

/-- A typed field handle. The FieldTy parameter is carried at the type level,
    so you can't accidentally use an i32 field where i64 is expected.
    The offset is computed by the layout builder — no manual hex constants. -/
structure Fld (t : FieldTy) where
  offset : Nat
  deriving Repr

/-- Type-erased field handle for payload initialization -/
structure AnyFld where
  offset : Nat
  ty : FieldTy

/-- Erase the type parameter from a field handle -/
def Fld.toAny (f : Fld t) : AnyFld := { offset := f.offset, ty := t }

/-- Layout builder state -/
structure BuilderState where
  fields : List AnyFld := []    -- accumulated fields (in order)
  cursor : Nat := 0             -- next free byte offset

/-- Layout builder monad -/
abbrev LayoutBuilder := StateM BuilderState

/-- Skip to an absolute offset. Fails silently if cursor is already past it. -/
def skipTo (offset : Nat) : LayoutBuilder Unit :=
  modify fun s => { s with cursor := max s.cursor offset }

/-- Skip forward by n bytes -/
def skip (n : Nat) : LayoutBuilder Unit :=
  modify fun s => { s with cursor := s.cursor + n }

/-- Add a field at the current cursor position. Returns a typed handle. -/
def field (ty : FieldTy) : LayoutBuilder (Fld ty) :=
  modifyGet fun s =>
    let f : Fld ty := { offset := s.cursor }
    (f, { fields := s.fields ++ [f.toAny], cursor := s.cursor + ty.size })

/-- Add a field at a specific absolute offset. Returns a typed handle. -/
def fieldAt (ty : FieldTy) (offset : Nat) : LayoutBuilder (Fld ty) :=
  modifyGet fun s =>
    let f : Fld ty := { offset }
    (f, { fields := s.fields ++ [f.toAny], cursor := max s.cursor (offset + ty.size) })

-- -------------------------------------------------------------------------
-- Layout finalization
-- -------------------------------------------------------------------------

/-- A finalized memory layout with total size -/
structure LayoutMeta where
  totalSize : Nat
  deriving Repr

/-- Run a layout builder, return both the builder's result and the layout metadata -/
def build (builder : LayoutBuilder α) : α × LayoutMeta :=
  let (a, st) := builder.run {}
  (a, { totalSize := st.cursor })

-- -------------------------------------------------------------------------
-- Payload generation
-- -------------------------------------------------------------------------

/-- A payload initializer: typed field handle + byte content -/
structure FieldInit where
  offset : Nat
  bytes : List UInt8
  deriving Inhabited

/-- Create a payload initializer from a typed field handle.
    Panics at build time if `bytes` exceeds the field's declared size. -/
def Fld.init (_f : Fld t) (bytes : List UInt8) : FieldInit :=
  if bytes.length > t.size then
    panic! s!"Fld.init: {bytes.length} bytes exceeds field size {t.size}"
  else
    { offset := _f.offset, bytes }

/-- Write bytes into a list at a given offset -/
private def writeBytesAux (buf : List UInt8) (pos : Nat) (bytes : List UInt8) (limit : Nat) : List UInt8 :=
  match bytes with
  | [] => buf
  | b :: rest =>
    if pos < limit then
      writeBytesAux (buf.set pos b) (pos + 1) rest limit
    else buf

/-- Generate a payload byte list from a total size and field initializers.
    Fields without initializers are zero-filled. -/
def mkPayload (size : Nat) (inits : List FieldInit) : List UInt8 :=
  let buf : List UInt8 := List.replicate size (0 : UInt8)
  inits.foldl (init := buf) fun acc init =>
    writeBytesAux acc init.offset init.bytes size

end Layout

end AlgorithmLib
