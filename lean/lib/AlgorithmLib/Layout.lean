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

-- ---------------------------------------------------------------------------
-- Region maps: hand-maintained offsets, checked at build time
-- ---------------------------------------------------------------------------

/-!
  ## Why this exists

  `LayoutBuilder` allocates offsets automatically, but large applications keep
  a *hand-written* memory map: fixed hex constants, chosen once and referenced
  from CLIF code all over the file.  Qwen2's map has forty-odd of them.

  Nothing checked that map, and the failure mode is brutal — a new constant
  landing on an existing one produces a segfault inside JIT'd code with no
  usable backtrace, three seconds into a run.  That happened: `META_STAGE_OFF`
  was placed at `0x05E0`, already occupied by `TOK_BUF_PTR_OFF`.

  A `RegionMap` states the map as data and makes both properties `decide`-able:
  regions do not overlap, and they fit inside the declared memory.  A collision
  becomes a build error, which is the whole point of doing layout at build time.
-/

/-- A named half-open byte range `[off, off + size)`. -/
structure Region where
  name : String
  off  : Nat
  size : Nat
  deriving Repr, DecidableEq

/-- Two regions are disjoint when one ends at or before the other begins.
    Zero-sized regions are vacuously disjoint from everything. -/
def Region.disjointB (a b : Region) : Bool :=
  a.size == 0 || b.size == 0 || a.off + a.size ≤ b.off || b.off + b.size ≤ a.off

abbrev RegionMap := List Region

namespace RegionMap

/-- Every pair of regions is disjoint. -/
def okB : RegionMap → Bool
  | []      => true
  | r :: rs => rs.all (Region.disjointB r) && okB rs

/-- Every region fits inside `size` bytes. -/
def withinB (size : Nat) (m : RegionMap) : Bool :=
  m.all (fun r => r.off + r.size ≤ size)

/-- The colliding pairs, by name — for `#eval` when a check fails.  `okB`
    answers *whether*; this answers *which*, which is what you actually need at
    3am with a segfault. -/
def collisions : RegionMap → List (String × String)
  | []      => []
  | r :: rs =>
      (rs.filter (fun s => !Region.disjointB r s)).map (fun s => (r.name, s.name))
        ++ collisions rs

/-- Regions that run past the end of memory, by name. -/
def overflows (size : Nat) (m : RegionMap) : List String :=
  (m.filter (fun r => !(r.off + r.size ≤ size))).map Region.name

/-- **Regions laid end to end, in order, from `start`** — no gaps, no overlap,
    *and* the list order is the memory order.

    Stronger than `okB`, and it is the property a **packed payload** needs: a
    host writes fields back to back, so a gap or a reordering is a silent
    misread rather than a collision.  `okB` would not catch either. -/
def packedB (start : Nat) : RegionMap → Bool
  | []      => true
  | r :: rs => r.off == start && packedB (start + r.size) rs

/-- Total bytes the regions occupy. -/
def total (m : RegionMap) : Nat := m.foldl (fun a r => a + r.size) 0

/-- Offset of region `i`, or `0` if there is none.  Reading offsets *out* of the
    map is what keeps one source of truth: the uploader and the layout check
    cannot drift because they are the same list. -/
def offAt (m : RegionMap) (i : Nat) : Nat := ((m[i]?).map Region.off).getD 0

/-- Size of region `i`. -/
def sizeAt (m : RegionMap) (i : Nat) : Nat := ((m[i]?).map Region.size).getD 0

/-- Size of the region called `name`, or `none`. -/
def sizeOf? (m : RegionMap) (name : String) : Option Nat :=
  (m.find? (fun r => r.name == name)).map Region.size

/-- **Region `name` has room for `n` elements of `elemBytes` bytes each.**

    `okB` relates regions to *each other*; this relates a region to the loop
    that fills it.  That is a different property of the same declaration, and
    the one nothing else checks: a map whose regions are perfectly disjoint
    still corrupts its neighbour if a writer runs past a region's declared
    size.  A missing region is `false`, not vacuously true — a renamed region
    must fail rather than silently stop checking.

    Use it wherever a bound and a capacity are two separate constants:

        theorem buf_holds_input : memMap.holdsB "token_buf" MAX_RECV 4 = true := by decide
    -/
def holdsB (m : RegionMap) (name : String) (n elemBytes : Nat) : Bool :=
  match m.find? (fun r => r.name == name) with
  | some r => n * elemBytes ≤ r.size
  | none   => false

/-- How many `elemBytes`-sized elements region `name` holds — what to widen a
    bound to, once `holdsB` has told you it does not fit. -/
def capacityOf? (m : RegionMap) (name : String) (elemBytes : Nat) : Option Nat :=
  (m.sizeOf? name).map (fun s => s / max elemBytes 1)

/-- Free gaps, largest first — where a new constant can safely go.  Answers the
    question you ask immediately after a collision. -/
def gaps (size : Nat) (m : RegionMap) : List (Nat × Nat) :=
  let ends := (m.map (fun r => (r.off, r.off + r.size))).mergeSort
                (fun a b => a.1 ≤ b.1)
  let rec go (cur : Nat) : List (Nat × Nat) → List (Nat × Nat)
    | [] => if cur < size then [(cur, size - cur)] else []
    | (s, e) :: rest =>
        let here := if cur < s then [(cur, s - cur)] else []
        here ++ go (max cur e) rest
  go 0 ends

end RegionMap

end Layout

end AlgorithmLib
