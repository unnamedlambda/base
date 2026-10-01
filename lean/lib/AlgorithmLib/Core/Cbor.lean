module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# CBOR, as artifacts are written

The encoder for the one profile `base_types::Artifact` reads (RFC 8949):
definite lengths, the shortest head for every value, structs as maps whose
text keys are the fields in declaration order, enums externally tagged, `None`
as null, bytes as byte strings. The build decodes every artifact and refuses
one that does not re-encode to the same bytes, so a disagreement between this
file and serde is a build failure, not a different reading.

Values are written straight into a `ByteArray`; there is no value tree.
-/

namespace AlgorithmLib.Cbor

/-- A writer: appends to the output, or fails naming what it could not write. -/
abbrev W := EStateM String ByteArray

def push (b : UInt8) : W Unit := modify (·.push b)

def pushBE (n : Nat) : Nat → W Unit
  | 0 => pure ()
  | k + 1 => do push (n >>> (8 * k)).toUInt8; pushBE n k

/-- The head of a data item: major type and argument, in the fewest bytes. -/
def head (major : UInt8) (n : Nat) : W Unit := do
  let m := major <<< 5
  if n < 24 then push (m ||| n.toUInt8)
  else if n < 0x100 then do push (m ||| 24); pushBE n 1
  else if n < 0x10000 then do push (m ||| 25); pushBE n 2
  else if n < 0x100000000 then do push (m ||| 26); pushBE n 4
  else if n < 0x10000000000000000 then do push (m ||| 27); pushBE n 8
  else throw s!"{n} does not fit a CBOR head"

def nat (n : Nat) : W Unit := head 0 n

def int (i : Int) : W Unit :=
  match i with
  | .ofNat n => head 0 n
  | .negSucc n => head 1 n

/-- An `i64`: an integer the reader holds in 64 signed bits, refused here
    rather than by the reader when it does not fit. -/
def i64 (i : Int) : W Unit :=
  if i < -2 ^ 63 || i ≥ 2 ^ 63 then throw s!"{i} does not fit an i64" else int i

def text (s : String) : W Unit := do
  let b := s.toUTF8
  head 3 b.size
  modify (· ++ b)

def bytes (b : ByteArray) : W Unit := do
  head 2 b.size
  modify (· ++ b)

def null : W Unit := push 0xf6

/-- A tag (major type 6); the tagged value follows. -/
def tag (n : Nat) : W Unit := head 6 n

def bool (b : Bool) : W Unit := push (if b then 0xf5 else 0xf4)

def array {α} (xs : List α) (f : α → W Unit) : W Unit := do
  head 4 xs.length
  for x in xs do f x

def option {α} (f : α → W Unit) : Option α → W Unit
  | some x => f x
  | none => null

/-- A struct: a map from each field's name to its value, in the order given. -/
def struct (fields : List (String × W Unit)) : W Unit := do
  head 5 fields.length
  for (k, v) in fields do
    text k
    v

/-- A tuple variant: `{name: [fields]}`. -/
def variant (name : String) (fields : List (W Unit)) : W Unit := do
  head 5 1
  text name
  head 4 fields.length
  for f in fields do f

/-- A newtype variant: `{name: value}`. -/
def newtypeVariant (name : String) (value : W Unit) : W Unit := do
  head 5 1
  text name
  value

class ToCbor (α : Type) where
  cbor : α → W Unit

export ToCbor (cbor)

/-- The bytes a writer produces on its own. -/
def run (w : W Unit) : Except String ByteArray :=
  match w.run ByteArray.empty with
  | .ok _ out => .ok out
  | .error e _ => .error e

/-- Encode one value on its own. -/
def encode {α} [ToCbor α] (x : α) : Except String ByteArray :=
  run (cbor x)

end AlgorithmLib.Cbor
