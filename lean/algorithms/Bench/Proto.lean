module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Bench.Memops
meta import Bench.Memops
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace Proto

/-!
  # A protobuf codec generated from a `.proto`

  `benchmarks/protobuf` measures it beside protobuf C++ and upb on
  HyperProtoBench. The generator reads a proto2 schema (messages, nested
  messages and enums, `optional` and `repeated` fields of the scalar, string,
  bytes, enum and message types) and writes one artifact: a decoder and an
  encoder, which read a table of each message's fields, over base's own
  layout of a message.

  **The layout.** A message is a record in the arena: its encoded size (cached
  by the encoder), a presence word for every 64 fields, then eight bytes a
  field. A field keeps its wire value: a varint's 64 bits, a fixed field's 4
  or 8 bytes, a string's or bytes' arena offset and length, a message's
  record; a repeated field, its first and last element of a list in arrival
  order. So the encoder writes back exactly what the wire said, in field
  number order, as protobuf C++ does.

  Its functions:

  * `decode`: messages framed as `(type u32, length u32, bytes)`, the type an
    index into the schema's top-level messages: each decoded into the arena
    (emptied first). The status 0, or 1 where a message is not that type's
    wire format, nests records deeper than 100 (as protobuf and upb refuse),
    or the arena is full.
  * `encode`: the decoded messages, framed the same way, to the output; the
    status 0, or 1 where the output is too small.
-/

/-! ## The schema -/

/-- What a field holds on the wire. -/
inductive Kind | varint | fixed32 | fixed64 | len | msg (index : Nat)
  deriving Repr, BEq, Inhabited

structure Field where
  name : String
  number : Nat
  repeated : Bool
  kind : Kind
  deriving Repr, Inhabited

structure Message where
  name : String
  fields : List Field
  deriving Repr, Inhabited

/-- A field's type before names are resolved. -/
structure RawField where
  name : String
  number : Nat
  repeated : Bool
  ty : String

/-- A message before names are resolved: its full name and its scope's path. -/
structure RawMessage where
  path : List String
  fields : List RawField

/-- The schema's tokens: names (with dots), numbers, strings and single
    symbols; `//` comments and whitespace dropped. -/
def tokens (s : String) : List String := Id.run do
  let cs := s.toList.toArray
  let mut out : Array String := #[]
  let mut i := 0
  while i < cs.size do
    let ch := cs[i]!
    if ch.isWhitespace then i := i + 1
    else if ch == '/' && i + 1 < cs.size && cs[i + 1]! == '/' then
      while i < cs.size && cs[i]! != '\n' do i := i + 1
    else if ch.isAlphanum || ch == '_' || ch == '.' then
      let mut j := i
      while j < cs.size && (cs[j]!.isAlphanum || cs[j]! == '_' || cs[j]! == '.') do j := j + 1
      out := out.push (String.ofList (cs.extract i j).toList)
      i := j
    else if ch == '"' then
      let mut j := i + 1
      while j < cs.size && cs[j]! != '"' do j := j + 1
      out := out.push (String.ofList (cs.extract i (j + 1)).toList)
      i := j + 1
    else
      out := out.push ch.toString
      i := i + 1
  return out.toList

/-- The messages (full paths, depth first) and the enums (full paths) of a
    proto2 schema, or what about it this generator does not take. -/
partial def parse (toks : List String) : Except String (List RawMessage × List (List String)) := do
  let rec block (path : List String) (ts : List String) (msgs : List RawMessage) (enums : List (List String))
      (fields : List RawField) :
      Except String (List String × List RawMessage × List (List String) × List RawField) := do
    match ts with
    | [] => pure ([], msgs, enums, fields)
    | "}" :: rest => pure (rest, msgs, enums, fields)
    | "message" :: name :: "{" :: rest =>
        let (rest', msgs', enums', fs) ← block (path ++ [name]) rest msgs enums []
        block path rest' (msgs' ++ [⟨path ++ [name], fs⟩]) enums' fields
    | "enum" :: name :: "{" :: rest =>
        let rec skip : List String → List String
          | "}" :: r => r
          | _ :: r => skip r
          | [] => []
        block path (skip rest) msgs (enums ++ [path ++ [name]]) fields
    | "syntax" :: "=" :: v :: ";" :: rest =>
        if v != "\"proto2\"" then throw s!"syntax {v}: this generator takes proto2"
        block path rest msgs enums fields
    | "package" :: _ :: ";" :: rest => block path rest msgs enums fields
    | label :: ty :: name :: "=" :: num :: ";" :: rest =>
        let rep ← match label with
          | "optional" => pure false
          | "repeated" => pure true
          | l => throw s!"a field labelled {l}: this generator takes optional and repeated"
        let some n := num.toNat? | throw s!"field {name}'s number {num}"
        block path rest msgs enums (fields ++ [⟨name, n, rep, ty⟩])
    | t :: _ => throw s!"unexpected `{t}` in the schema"
  let (_, msgs, enums, _) ← block [] toks [] [] []
  pure (msgs, enums)

/-- `ty` named from inside `scope`: the innermost scope outward, as protobuf
    resolves it. -/
def resolve (names : List (List String)) (scope : List String) (ty : String) : Option (List String) :=
  let parts := ty.splitOn "."
  ((List.range (scope.length + 1)).reverse.map fun k => scope.take k ++ parts).find? names.contains

/-- The schema resolved: its messages (index = position) and its top-level
    messages' indices. -/
def schema (src : String) : Except String (Array Message × List Nat) := do
  let (raw, enums) ← parse (tokens src)
  let paths := raw.map (·.path)
  let idx (p : List String) : Nat := (paths.idxOf p)
  let msgs ← raw.mapM fun m => do
    let fs ← m.fields.mapM fun f => do
      let k ← match f.ty with
        | "int32" | "int64" | "uint32" | "uint64" | "sint32" | "sint64" | "bool" => pure Kind.varint
        | "fixed32" | "sfixed32" | "float" => pure Kind.fixed32
        | "fixed64" | "sfixed64" | "double" => pure Kind.fixed64
        | "string" | "bytes" => pure Kind.len
        | t =>
          match resolve paths m.path t with
          | some p => pure (Kind.msg (idx p))
          | none => match resolve enums m.path t with
            | some _ => pure Kind.varint
            | none => throw s!"{".".intercalate m.path}.{f.name}: no type {t}"
      pure ({ name := f.name, number := f.number, repeated := f.repeated, kind := k } : Field)
    pure ({ name := ".".intercalate m.path, fields := fs.mergeSort (·.number ≤ ·.number) } : Message)
  let tops := (List.range raw.length).filter fun i => (paths.getD i []).length == 1
  pure (msgs.toArray, tops)

/-! ## The arena -/

/-- State words: the bump pointer, the decoded messages' count; then the
    decoded messages' table, `(type, record)` a pair of words each. -/
def BUMP : Nat := 0x100
def NTOP : Nat := 0x108
/-- Where the varint helper leaves its value, and the helper's place among
    the program's functions. -/
def SCRATCH : Nat := 0x128
def VARINT_IX : Nat := 2
/-- Base's AVX2 copy (`MemopsAsm.copy`), for the strings 128 bytes and up:
    its slot word (`Memops.mappedAt`), `"avx2"` for the CPU test, its bytes. -/
def COPY_SLOT : Nat := 0x110
def COPY_NAME : Nat := 0x118
/-- Whether the heap has been offered huge pages yet. -/
def ADVISED : Nat := 0x120
/-- The decoder's depth of records, at most `MAX_DEPTH` (protobuf's and
    upb's limit). -/
def DEPTH : Nat := 0x130
def MAX_DEPTH : Nat := 100
def TOP : Nat := 0x200
def MAX_TOP : Nat := 4096
def COPY_CODE : Nat := TOP + 16 * MAX_TOP
/-- The decoder's tables (`tableBytes`), in at most `TABLE_BYTES`. -/
def TABLES : Nat := Memops.align64 (COPY_CODE + (Memops.bodyBytes MemopsAsm.copy).length)
def TABLE_BYTES : Nat := 1 <<< 20
/-- The encoder's frames (`walk`), 64 bytes a record of the depth. -/
def ESTACK : Nat := TABLES + TABLE_BYTES
def HEAP : Nat := ESTACK + 64 * (MAX_DEPTH + 1)
def MEM : Nat := HEAP + (1 <<< 28)

/-- A record's bytes: its cached size, its presence words, eight bytes a field. -/
def presenceWords (m : Message) : Nat := (m.fields.length + 63) / 64
def presenceOff (w : Nat) : Nat := 8 + 8 * w
def slotOff (m : Message) (i : Nat) : Nat := 8 + 8 * presenceWords m + 8 * i
def recordBytes (m : Message) : Nat := slotOff m m.fields.length
/-- A list element: its value, the next element (an arena offset, 0 for none). -/
def NODE : Nat := 16

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def c (n : Int) : Prog V L (V .i64) := iconst64 n
def b2i (x : V .i8) : Prog V L (V .i64) := do select x (← c 1) (← c 0)
def at_ (off : V .i64) : Prog V L (V .i64) := do iadd (← basePtr) off

/-- `n` bytes from the arena, not cleared, at a 64-byte boundary (so a copy
    into them stores whole lines), their offset; `0` where it is full. -/
def allocRaw (n : V .i64) : Prog V L (V .i64) := do
  let base ← basePtr
  let b ← band (← iaddImm (← load64 (← iaddImm base BUMP)) 63) (← c (-64))
  let e ← iadd b (← band (← iaddImm n 7) (← c (-8)))
  let r ← ifte (jTys := [.i64]) .ugt e (← c MEM)
    (thn := do pure %[← c 0])
    (els := do
      storeI64 e (← iaddImm base BUMP)
      pure %[b])
  pure r.head

/-- `copyN`'s loop in CLIF. -/
def copyLoop (src dst n : V .i64) : Prog V L Unit := do
  let _ ← ifte (jTys := []) .uge n (← c 16)
    (thn := do
      let last ← iaddImm n (-16)
      let big ← wloop %[← c 0] (fun cs => do pure (contIf .ule (← iaddImm cs.head 64) n, %[cs.head], ()))
        (fun cs _ => do
          let vs ← (List.range 4).mapM fun (k : Nat) => do
            load { ty := .i8x16 } (← iadd src (← iaddImm cs.head (16 * k)))
          for (v, k) in vs.zipIdx do
            storeUnaligned v (← iadd dst (← iaddImm cs.head (16 * k)))
          pure %[← iaddImm cs.head 64])
      let _ ← wloop %[big.head] (fun cs => pure (contIfULt cs.head last, %[], ()))
        (fun cs _ => do
          storeUnaligned (← load { ty := .i8x16 } (← iadd src cs.head)) (← iadd dst cs.head)
          pure %[← iaddImm cs.head 16])
      storeUnaligned (← load { ty := .i8x16 } (← iadd src last)) (← iadd dst last)
      pure %[])
    (els := do
      -- under 16: two moves of the widest power of two that fits, the second
      -- ending at the end (overlapping the first where `n` is not that width)
      let pair (w : Nat) (ld : V .i64 → Prog V L (V .i64)) (st : V .i64 → V .i64 → Prog V L Unit) :
          Prog V L Unit := do
        let a ← ld src
        let tail ← iaddImm n (-(w : Int))
        let b ← ld (← iadd src tail)
        st a dst
        st b (← iadd dst tail)
      let _ ← ifte (jTys := []) .uge n (← c 8)
        (thn := do pair 8 load64 (fun v p => storeI64 v p); pure %[])
        (els := do
          let _ ← ifte (jTys := []) .uge n (← c 4)
            (thn := do pair 4 uload32_64 (fun v p => do storeI32 (← ireduce32 v) p); pure %[])
            (els := do
              let _ ← ifte (jTys := []) .uge n (← c 2)
                (thn := do pair 2 uload16_64 (fun v p => istore16 v p); pure %[])
                (els := do
                  when .ne n (← c 0) do istore8 (← uload8_64 src) dst
                  pure %[])
              pure %[])
          pure %[])
      pure %[])
  pure ()

/-- `n` bytes from `src` to `dst`, which do not overlap: from 128, base's AVX2
    copy where it is mapped; otherwise 64 at a time, then 16, the last 16
    overlapping the trips before; under 16, two overlapping moves of 8, 4 or 2
    bytes, or the one byte. -/
def copyN (src dst n : V .i64) : Prog V L Unit := do
  let fn ← ifte (jTys := [.i64]) .uge n (← c 128)
    (thn := do pure %[← Memops.mappedAt COPY_SLOT COPY_NAME COPY_CODE MemopsAsm.copy])
    (els := do pure %[← c Memops.UNMAPPED])
  let _ ← ifte (jTys := []) .ne fn.head (← c Memops.UNMAPPED)
    (thn := do
      let _ ← nativeCall fn.head src dst n (← c 0)
      pure %[])
    (els := do
      copyLoop src dst n
      pure %[])
  pure ()

/-- `n` bytes from the arena, zeroed, their offset; `0` where it is full. -/
def alloc (n : V .i64) : Prog V L (V .i64) := do
  let base ← basePtr
  let b ← load64 (← iaddImm base BUMP)
  let e ← iadd b (← band (← iaddImm n 7) (← c (-8)))
  let r ← ifte (jTys := [.i64]) .ugt e (← c MEM)
    (thn := do pure %[← c 0])
    (els := do
      storeI64 e (← iaddImm base BUMP)
      forLoop (← ushrImm (← isub e b) 3) fun i => do
        storeI64 (← c 0) (← iadd (← iadd base b) (← ishlImm i 3))
      pure %[b])
  pure r.head

/-- `readVarint`'s general case: a byte at a time. -/
def readVarintLoop (p e : V .i64) : Prog V L (V .i64 × V .i64) := do
  let zero ← c 0
  let r ← wloopL %[p, zero, zero]
    (head := fun _ cs => do pure (contIf .eq zero zero, %[cs.head, cs.snd], ()))
    (body := fun out cs _ => do
      let q := cs.head
      let ok ← band (← b2i (← icmp .ult q e)) (← b2i (← icmp .ult cs.thd (← c 64)))
      when .eq ok zero do brk out %[zero, zero]
      let b ← uload8_64 q
      let v ← bor cs.snd (← ishl (← band b (← c 0x7F)) cs.thd)
      when .ult b (← c 0x80) do brk out %[← iaddImm q 1, v]
      pure %[← iaddImm q 1, v, ← iaddImm cs.thd 7])
  pure (r.head, r.snd)

/-- A varint at `p` before `e`: its value and where it ends, `0` for `p` where
    it runs past `e` or past ten bytes. -/
def readVarint (p e : V .i64) : Prog V L (V .i64 × V .i64) := do
  let zero ← c 0
  -- one byte, which most tags and lengths are, without the loop
  let b0 ← uload8_64 (← select (← icmp .ult p e) p (← iaddImm e (-1)))
  let one ← band (← b2i (← icmp .ult p e)) (← b2i (← icmp .ult b0 (← c 0x80)))
  let r ← ifte (jTys := [.i64, .i64]) .ne one zero (thn := do pure %[← iaddImm p 1, b0]) (els := do
    let q ← callLocal (⟨.local VARINT_IX⟩ : LocalRef [.i64, .i64, .i64, .i64] (some .i64)) %[p, e, zero, zero]
    pure %[q, ← load64 (← iaddImm (← basePtr) SCRATCH)])
  pure (r.head, r.snd)

/-- `v` as a varint at `o`; where it ends. -/
def writeVarint (o v : V .i64) : Prog V L (V .i64) := do
  let r ← wloop %[o, v] (fun cs => do pure (contIf .uge cs.snd (← c 0x80), %[cs.head, cs.snd], ()))
    (fun cs _ => do
      istore8 (← bor (← band cs.snd (← c 0x7F)) (← c 0x80)) cs.head
      pure %[← iaddImm cs.head 1, ← ushrImm cs.snd 7])
  istore8 r.snd r.head
  iaddImm r.head 1

/-- The bytes of `v` as a varint. -/
def varintSize (v : V .i64) : Prog V L (V .i64) := do
  -- 1 + floor((bits - 1) / 7), bits the width of `v` (1 for 0), which is
  -- (9 · bits + 64) / 64 for every width to 64
  let bits ← isub (← c 64) (← clz (← bor v (← c 1)))
  ushrImm (← iaddImm (← imul bits (← c 9)) 64) 6

end

/-! ## The generated functions

    Function `k` of the program: the two entries at 0 and 1; the shared
    helpers (a varint's general case, a string's copy, a list's append, a
    record's allocation, a packed run) at 2 to 6; the decoder, the sizer and
    the writer at 7, 8 and 9, each over the message tables. One function of
    each over tables, and one copy of each helper rather than one a field,
    keep the code small enough to stay in the caches, whatever the schema. -/

def COPY_IX : Nat := 3
def APPEND_IX : Nat := 4
def ALLOC_IX : Nat := 5
def PACKED_IX : Nat := 6
def DEC_IX : Nat := 7
def SIZE_IX : Nat := 8
def WRITE_IX : Nat := 9

/-- Every generated function takes four words, as an entry does, so the
    arena's base reaches it the way it reaches an entry (`basePtr`); the
    words a function does not use are 0. -/
def P4 : List ClifTy := [.i64, .i64, .i64, .i64]

def wireType : Kind → Nat
  | .varint => 0 | .fixed64 => 1 | .len => 2 | .msg _ => 2 | .fixed32 => 5

/-- A field's kind in its table entry. -/
def KIND_VARINT : Nat := 0
def KIND_FIXED32 : Nat := 1
def KIND_FIXED64 : Nat := 2
def KIND_LEN : Nat := 3
def KIND_MSG : Nat := 4
def kindCode : Kind → Nat
  | .varint => KIND_VARINT | .fixed32 => KIND_FIXED32 | .fixed64 => KIND_FIXED64 | .len => KIND_LEN | .msg _ => KIND_MSG

/-- `x`'s `n` bytes, little-endian. -/
def le (x n : Nat) : List UInt8 := (List.range n).map fun i => UInt8.ofNat (x / 256 ^ i % 256)

/-- A field's table entry's bytes, and its shift. -/
def ENTRY : Nat := 32
def ENTRY_SHIFT : Nat := 5

/-- `v` as a varint. -/
def varintBytes (v : Nat) : List UInt8 :=
  if v < 0x80 then [UInt8.ofNat v] else UInt8.ofNat (v % 128 + 128) :: varintBytes (v / 128)
termination_by v
decreasing_by omega

/-- The codec's tables, at `TABLES`: 16 bytes a message (its fields' table
    as an arena offset, their count, its record's bytes), then 32 bytes a
    field, a message's in schema order: its number; its slot's offset in the
    record; its presence word's offset and bit; its kind (`kindCode`); a
    message field's message; 1 if it repeats; the wire type it is written
    with; at 16 its tag's bytes as a varint, at 24 their count, at 25 that
    count plus a fixed field's 4 or 8 bytes. -/
def tableBytes (msgs : Array Message) : List UInt8 :=
  let ms := msgs.toList
  let starts := ms.foldl (fun (acc : List Nat × Nat) m => (acc.1 ++ [acc.2], acc.2 + m.fields.length)) ([], 0) |>.1
  let fields0 := TABLES + 16 * ms.length
  let hdrs := (ms.zip starts).flatMap fun (m, k) =>
    le (fields0 + ENTRY * k) 4 ++ le m.fields.length 4 ++ le (recordBytes m) 4 ++ le 0 4
  let ents := ms.flatMap fun m => m.fields.zipIdx.flatMap fun (f, i) =>
    let sub := match f.kind with | .msg j => j | _ => 0
    le f.number 4 ++ le (slotOff m i) 4 ++ le (presenceOff (i / 64)) 2 ++ le (i % 64) 1 ++ le (kindCode f.kind) 1 ++
      le sub 2 ++ le (if f.repeated then 1 else 0) 1 ++ le (wireType f.kind) 1 ++
      let tag := varintBytes (f.number * 8 + wireType f.kind)
      let fixed := match f.kind with | .fixed32 => 4 | .fixed64 => 8 | _ => 0
      tag ++ zeros (8 - tag.length) ++ le tag.length 1 ++ le (tag.length + fixed) 1 ++ zeros 6
  hdrs ++ ents

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- A field's value appended to repeated slot `slot`: a list element after
    the last; `0` where the arena is full. -/
def appendNode (slot v : V .i64) : Prog V L (V .i64) := do
  let node ← alloc (← c NODE)
  let r ← ifte (jTys := [.i64]) .eq node (← c 0) (thn := do pure %[← c 0]) (els := do
    storeI64 v (← at_ node)
    let head ← uload32_64 slot
    let tail ← uload32_64 (← iaddImm slot 4)
    let _ ← ifte (jTys := []) .eq head (← c 0)
      (thn := do storeI32 (← ireduce32 node) slot; pure %[])
      (els := do storeI32 (← ireduce32 node) (← iaddImm (← at_ tail) 8); pure %[])
    storeI32 (← ireduce32 node) (← iaddImm slot 4)
    pure %[← c 1])
  pure r.head

/-- A string or bytes of `n` at `q`, copied into the arena: its slot word
    (offset, length); `0` where the arena is full. -/
def copyBytes (q n : V .i64) : Prog V L (V .i64) := do
  let off ← allocRaw (← iaddImm n 1)
  let r ← ifte (jTys := [.i64]) .eq off (← c 0) (thn := do pure %[← c 0]) (els := do
    copyN q (← at_ off) n
    pure %[← bor off (← ishlImm n 32)])
  pure r.head

/-- **The decoder**: params `(p, e, rec, mi)`; message `mi`'s fields until
    `e`, into record `rec` (merged with what it holds), each found in the
    message's field table (`tableBytes`) from the field after the last one
    seen, as a writer orders them; where it stopped, `0` on an error. -/
def decodeBody : Prog V L (V .i64) := do
  let ps ← entryParams P4
  let p0 := ps.head; let e := ps.snd; let rec_ := ps.thd; let mi := ps.fth
  let zero ← c 0
  let hdr ← at_ (← iaddImm (← ishlImm mi 4) TABLES)
  let fa ← uload32_64 hdr
  let nf ← uload32_64 (← iaddImm hdr 4)
  let recA ← at_ rec_
  -- carried: the position, the field to look at first
  let r ← wloopL %[p0, zero]
    (head := fun _ cs => do pure (contIf .ult cs.head e, %[cs.head], ()))
    (body := fun out cs _ => do
      let (q, tag) ← readVarint cs.head e
      when .eq q zero do brk out %[zero]
      let num ← ushrImm tag 3
      let wt ← band tag (← c 7)
      -- the field numbered `num`: its index, `nf` for none
      let found ← wloopL %[cs.snd, nf]
        (head := fun _ fs => do pure (contIf .ne fs.snd zero, %[nf], ()))
        (body := fun inner fs _ => do
          when .eq (← uload32_64 (← at_ (← iadd fa (← ishlImm fs.head ENTRY_SHIFT)))) num do brk inner %[fs.head]
          let k ← iaddImm fs.head 1
          pure %[← select (← icmp .eq k nf) zero k, ← iaddImm fs.snd (-1)])
      let idx := found.head
      when .ne idx nf do
        let ent ← at_ (← iadd fa (← ishlImm idx ENTRY_SHIFT))
        let slot ← iadd recA (← uload32_64 (← iaddImm ent 4))
        let pw ← iadd recA (← uload16_64 (← iaddImm ent 8))
        let bit ← ishl (← c 1) (← uload8_64 (← iaddImm ent 10))
        let kind ← uload8_64 (← iaddImm ent 11)
        let sub ← uload16_64 (← iaddImm ent 12)
        let rep ← uload8_64 (← iaddImm ent 14)
        let want ← uload8_64 (← iaddImm ent 15)
        let mark : Prog V L Unit := do storeI64 (← bor (← load64 pw) bit) pw
        -- the next field looked at: this one again if it repeats, else the next
        let k ← iadd idx (← bxor rep (← c 1))
        let hint ← select (← icmp .eq k nf) zero k
        let lenThen (k : V .i64 → V .i64 → Prog V L (Vals V [.i64, .i64])) : Prog V L (Vals V [.i64, .i64]) := do
          let (q2, n) ← readVarint q e
          let ok ← band (← b2i (← icmp .ne q2 zero)) (← b2i (← icmp .ule n (← isub e q2)))
          ifte (jTys := [.i64, .i64]) .ne ok zero (thn := k q2 n) (els := do pure %[zero, zero])
        when .eq wt want do
          -- a value, as the slot keeps it, and where the text resumes
          let v ← ifte (jTys := [.i64, .i64]) .eq kind (← c KIND_VARINT)
            (thn := do
              let (q2, x) ← readVarint q e
              pure %[q2, x])
            (els := ifte .eq kind (← c KIND_FIXED32)
              (thn := do
                let ok ← icmp .ule (← iaddImm q 4) e
                ifte (jTys := [.i64, .i64]) .ne (← b2i ok) zero
                  (thn := do pure %[← iaddImm q 4, ← uload32_64 q]) (els := do pure %[zero, zero]))
              (els := ifte .eq kind (← c KIND_FIXED64)
                (thn := do
                  let ok ← icmp .ule (← iaddImm q 8) e
                  ifte (jTys := [.i64, .i64]) .ne (← b2i ok) zero
                    (thn := do pure %[← iaddImm q 8, ← load64 q]) (els := do pure %[zero, zero]))
                (els := ifte .eq kind (← c KIND_LEN)
                  (thn := lenThen fun q2 n => do
                    let w ← callLocal (⟨.local COPY_IX⟩ : LocalRef [.i64, .i64, .i64, .i64] (some .i64)) %[q2, n, zero, zero]
                    pure %[← select (← icmp .eq w zero) zero (← iadd q2 n), w])
                  (els := lenThen fun q2 n => do
                    -- an optional message seen again merges into its record
                    let again ← band (← b2i (← icmp .ne (← band (← load64 pw) bit) zero)) (← bxor rep (← c 1))
                    let fresh ← ifte (jTys := [.i64]) .ne again zero
                      (thn := do pure %[← load64 slot])
                      (els := do
                        let bytes ← uload32_64 (← at_ (← iaddImm (← ishlImm sub 4) (TABLES + 8)))
                        pure %[← callLocal (⟨.local ALLOC_IX⟩ : LocalRef [.i64, .i64, .i64, .i64] (some .i64)) %[bytes, zero, zero, zero]])
                    let sr := fresh.head
                    let dp ← at_ (← c DEPTH)
                    let d ← load64 dp
                    let ok ← band (← b2i (← icmp .ne sr zero)) (← b2i (← icmp .ult d (← c MAX_DEPTH)))
                    let res ← ifte (jTys := [.i64]) .eq ok zero (thn := do pure %[zero]) (els := do
                      let end_ ← iadd q2 n
                      storeI64 (← iaddImm d 1) dp
                      let got ← callLocal (⟨.local DEC_IX⟩ : LocalRef P4 (some .i64)) %[q2, end_, sr, sub]
                      storeI64 d dp
                      pure %[← select (← icmp .eq got end_) end_ zero])
                    pure %[res.head, sr]))))
          when .eq v.head zero do brk out %[zero]
          let _ ← ifte (jTys := []) .ne rep zero
            (thn := do
              when .eq (← callLocal (⟨.local APPEND_IX⟩ : LocalRef [.i64, .i64, .i64, .i64] (some .i64)) %[slot, v.snd, zero, zero]) zero do brk out %[zero]
              pure %[])
            (els := do storeI64 v.snd slot; pure %[])
          mark
          continueWith out %[v.head, hint]
        -- a packed run of a repeated scalar's values
        let packable ← band rep (← b2i (← icmp .ult kind (← c KIND_LEN)))
        when .ne (← band packable (← b2i (← icmp .eq wt (← c 2)))) zero do
          let (q2, n) ← readVarint q e
          let ok ← band (← b2i (← icmp .ne q2 zero)) (← b2i (← icmp .ule n (← isub e q2)))
          when .eq ok zero do brk out %[zero]
          let end_ ← iadd q2 n
          -- 0 for varints, else the value's bytes
          let width ← select (← icmp .eq kind (← c KIND_FIXED32)) (← c 4)
            (← select (← icmp .eq kind (← c KIND_FIXED64)) (← c 8) zero)
          let got ← callLocal (⟨.local PACKED_IX⟩ : LocalRef [.i64, .i64, .i64, .i64] (some .i64))
            %[q2, end_, slot, width]
          when .ne got end_ do brk out %[zero]
          mark
          continueWith out %[end_, hint]
      -- an unknown field, or a known one on a wire type it does not have: skipped
      let skip ← ifte (jTys := [.i64]) .eq wt zero
        (thn := do
          let (q2, _) ← readVarint q e
          pure %[q2])
        (els := ifte .eq wt (← c 1)
          (thn := do pure %[← select (← icmp .ule (← iaddImm q 8) e) (← iaddImm q 8) zero])
          (els := ifte .eq wt (← c 5)
            (thn := do pure %[← select (← icmp .ule (← iaddImm q 4) e) (← iaddImm q 4) zero])
            (els := ifte .eq wt (← c 2)
              (thn := do
                let (q2, n) ← readVarint q e
                let ok ← band (← b2i (← icmp .ne q2 zero)) (← b2i (← icmp .ule n (← isub e q2)))
                pure %[← select (← icmp .ne ok zero) (← iadd q2 n) zero])
              (els := do pure %[zero]))))
      when .eq skip.head zero do brk out %[zero]
      pure %[skip.head, cs.snd])
  pure r.head

/-- A field's table entry (`tableBytes`) as the encoder reads it, each
    part loaded where it is used: the entry and the record. -/
structure Entry (V : ClifTy → Type) where
  ent : V .i64
  recA : V .i64

/-- Entry `k` of the table at `fa`, for record `recA`. -/
def entryAt (fa recA k : V .i64) : Prog V L (Entry V) := do
  pure { ent := ← at_ (← iadd fa (← ishlImm k ENTRY_SHIFT)), recA }

/-- The record's slot of the field, its kind, its message, 1 if it repeats,
    its tag's bytes and their count, and the bytes of its tag and fixed
    value. -/
def Entry.slot (f : Entry V) : Prog V L (V .i64) := do iadd f.recA (← uload32_64 (← iaddImm f.ent 4))
def Entry.kind (f : Entry V) : Prog V L (V .i64) := do uload8_64 (← iaddImm f.ent 11)
def Entry.sub (f : Entry V) : Prog V L (V .i64) := do uload16_64 (← iaddImm f.ent 12)
def Entry.rep (f : Entry V) : Prog V L (V .i64) := do uload8_64 (← iaddImm f.ent 14)
def Entry.tagW (f : Entry V) : Prog V L (V .i64) := do load64 (← iaddImm f.ent 16)
def Entry.tagN (f : Entry V) : Prog V L (V .i64) := do uload8_64 (← iaddImm f.ent 24)
def Entry.sizeConst (f : Entry V) : Prog V L (V .i64) := do uload8_64 (← iaddImm f.ent 25)

/-- Message `mi`'s field table's address and count. -/
def tableOf (mi : V .i64) : Prog V L (V .i64 × V .i64) := do
  let hdr ← at_ (← iaddImm (← ishlImm mi 4) TABLES)
  pure (← uload32_64 hdr, ← uload32_64 (← iaddImm hdr 4))

/-- Message `mi`'s record at `rec`: its address, its table and field count,
    and its first presence word (0 with no fields). -/
def openRec (rec mi : V .i64) : Prog V L (V .i64 × V .i64 × V .i64 × V .i64) := do
  let recA ← at_ rec
  let (fa, nf) ← tableOf mi
  let none_ ← icmp .eq nf (← c 0)
  let w ← load64 (← select none_ recA (← iaddImm recA (presenceOff 0)))
  pure (recA, fa, nf, ← select none_ (← c 0) w)

/-- **The encoder's walk** of record `rec` of message `mi` and each record
    under it: a record's present fields in number order, a repeated field's
    values in arrival order, from `acc0`, with no calls. Each record of the
    depth has a frame at `ESTACK`: its address, table, field count and
    presence word's index, then, while a submessage's record is walked, where
    its own walk resumes (the decoder refuses records nested deeper than
    `MAX_DEPTH`). `scalar f kind v acc`, a value that is not a message; `enter f v
    acc`, a submessage's record `v`: the acc its walk starts from, and the acc
    this record keeps meanwhile; `leave recA acc`, a record's fields done;
    `resume f kept acc`, the acc after the submessage under field `f`. -/
def walk (rec mi acc0 : V .i64)
    (scalar : Entry V → V .i64 → V .i64 → V .i64 → Prog V L (V .i64))
    (enter : Entry V → V .i64 → V .i64 → Prog V L (V .i64 × V .i64))
    (leave : V .i64 → V .i64 → Prog V L Unit)
    (resume : Entry V → V .i64 → V .i64 → Prog V L (V .i64)) : Prog V L (V .i64) := do
  let zero ← c 0
  let stk ← at_ (← c ESTACK)
  -- record `r` of message `m` opened in the frame at `fp`: its first presence word
  let open_ (fp r m : V .i64) : Prog V L (V .i64) := do
    let (recA, fa, nf, w) ← openRec r m
    for (x, k) in [recA, fa, nf, zero].zipIdx do storeI64 x (← iaddImm fp (8 * k))
    pure w
  let w0 ← open_ stk rec mi
  -- carried: the frame, what is left of the presence word, the acc, the next
  -- value of the list being walked (0 for none) and its field
  let r ← wloopL %[stk, w0, acc0, zero, zero]
    (head := fun _ _ => do pure (contIf .eq zero zero, %[zero], ()))
    (body := fun out cs _ => do
      let fp := cs.head; let w := cs.snd; let acc := cs.thd; let node := cs.fth; let i := (cs.drop 4).head
      let recA ← load64 fp
      let fa ← load64 (← iaddImm fp 8)
      -- value `v` of field `i` (entry `f`), the walk going on from `w'`, `node'`
      let visit (f : Entry V) (i v w' node' : V .i64) : Prog V L Unit := do
        let kind ← f.kind
        when .eq kind (← c KIND_MSG) do
          let (inner, kept) ← enter f v acc
          for (x, k) in [w', kept, node', i].zipIdx do storeI64 x (← iaddImm fp (32 + 8 * k))
          let fp2 ← iaddImm fp 64
          continueWith out %[fp2, ← open_ fp2 v (← f.sub), inner, zero, zero]
        continueWith out %[fp, w', ← scalar f kind v acc, node', i]
      -- the list's next value
      when .ne node zero do
        let nd ← at_ node
        visit (← entryAt fa recA i) i (← load64 nd) w (← uload32_64 (← iaddImm nd 8))
      -- the next present field
      let wi ← load64 (← iaddImm fp 24)
      when .ne w zero do
        let i2 ← iadd (← ishlImm wi 6) (← ctz w)
        let w2 ← band w (← iaddImm w (-1))
        let f ← entryAt fa recA i2
        let slot ← f.slot
        when .ne (← f.rep) zero do
          let nd ← at_ (← uload32_64 slot)
          visit f i2 (← load64 nd) w2 (← uload32_64 (← iaddImm nd 8))
        visit f i2 (← load64 slot) w2 zero
      -- the next presence word
      let wi2 ← iaddImm wi 1
      when .ult (← ishlImm wi2 6) (← load64 (← iaddImm fp 16)) do
        storeI64 wi2 (← iaddImm fp 24)
        continueWith out %[fp, ← load64 (← iadd recA (← iaddImm (← ishlImm wi2 3) (presenceOff 0))), acc, zero, zero]
      -- the record done: the walk of the record it is under resumed
      leave recA acc
      when .eq fp stk do brk out %[acc]
      let pf ← iaddImm fp (-64)
      let ld (k : Nat) : Prog V L (V .i64) := do load64 (← iaddImm pf (8 * k))
      let pi ← ld 7
      let pe ← entryAt (← ld 1) (← ld 0) pi
      pure %[pf, ← ld 4, ← resume pe (← ld 5) acc, ← ld 6, pi])
  pure r.head

/-- The bytes a value `v` of entry `f`, of kind `kind`, not a message,
    takes, tag first. -/
def scalarBytes (f : Entry V) (kind v : V .i64) : Prog V L (V .i64) := do
  let k ← f.sizeConst
  let r ← ifte (jTys := [.i64]) .eq kind (← c KIND_VARINT) (thn := do pure %[← varintSize v])
    (els := ifte .eq kind (← c KIND_LEN)
      (thn := do
        let n ← ushrImm v 32
        pure %[← iadd (← varintSize n) n])
      (els := do pure %[← c 0]))
  iadd k r.head

/-- **The sizer**: params `(rec, mi)`; message `mi`'s encoded bytes, cached
    in the record (and each record's under it). -/
def sizeBody : Prog V L (V .i64) := do
  let ps ← entryParams P4
  walk ps.head ps.snd (← c 0)
    (scalar := fun f kind v t => do iadd t (← scalarBytes f kind v))
    (enter := fun _ _ t => do pure (← c 0, t))
    (leave := fun recA t => storeI64 t recA)
    (resume := fun f kept t => do iadd kept (← iadd (← f.tagN) (← iadd (← varintSize t) t)))

/-- Entry `f`'s tag at `o`, before `oe`: its bytes' word, with 8 bytes
    before `oe`, else each byte; where it ends. -/
def writeTag (f : Entry V) (o oe : V .i64) : Prog V L (V .i64) := do
  let tw ← f.tagW
  let tn ← f.tagN
  let _ ← ifte (jTys := []) .ule (← iaddImm o 8) oe
    (thn := do storeI64 tw o; pure %[])
    (els := do
      forLoop tn fun i => do istore8 (← ushr tw (← ishlImm i 3)) (← iadd o i)
      pure %[])
  iadd o tn

/-- A value `v` of entry `f`, of kind `kind`, not a message, written at `o`
    before `oe`, tag first; where it ends. -/
def writeScalar (f : Entry V) (kind o v oe : V .i64) : Prog V L (V .i64) := do
  let o1 ← writeTag f o oe
  let r ← ifte (jTys := [.i64]) .eq kind (← c KIND_VARINT) (thn := do pure %[← writeVarint o1 v])
    (els := ifte .eq kind (← c KIND_LEN)
      (thn := do
        let n ← ushrImm v 32
        let src ← at_ (← band v (← c 0xFFFFFFFF))
        let o2 ← writeVarint o1 n
        copyN src o2 n
        pure %[← iadd o2 n])
      (els := ifte .eq kind (← c KIND_FIXED32) (thn := do storeI32 (← ireduce32 v) o1; pure %[← iaddImm o1 4])
        (els := do storeI64 v o1; pure %[← iaddImm o1 8])))
  pure r.head

/-- **The writer**: params `(rec, o, mi, oe)`, the sizes cached and room
    for them before `oe`; message `mi`'s fields at `o`, in number order;
    where they end. -/
def writeBody : Prog V L (V .i64) := do
  let ps ← entryParams P4
  let oe := ps.fth
  walk ps.head ps.thd ps.snd
    (scalar := fun f kind v o => writeScalar f kind o v oe)
    (enter := fun f v o => do
      let o1 ← writeTag f o oe
      pure (← writeVarint o1 (← load64 (← at_ v)), ← c 0))
    (leave := fun _ _ => pure ())
    (resume := fun _ _ o => pure o)

/-- **`decode`**: the framed messages, each into a fresh record; the arena
    emptied first (and, on the first call, offered huge pages). -/
def decodeEntry (msgs : Array Message) (tops : List Nat) : Prog V L (V .i64) := do
  let base ← basePtr
  let zero ← c 0
  -- the heap on huge pages where the system grants them, asked once: a large
  -- message's strings and records otherwise cross more pages than the
  -- translation cache holds
  when .eq (← load64 (← iaddImm base ADVISED)) zero do
    let _ ← memAdviseHuge (← iaddImm base HEAP) (← c (MEM - HEAP))
    storeI64 (← c 1) (← iaddImm base ADVISED)
  storeI64 (← c HEAP) (← iaddImm base BUMP)
  storeI64 zero (← iaddImm base NTOP)
  storeI64 zero (← iaddImm base DEPTH)
  let p0 ← dataPtr
  let e ← iadd p0 (← dataLen)
  let r ← wloopL %[p0, zero]
    (head := fun _ cs => do pure (contIf .ult cs.head e, %[zero], ()))
    (body := fun out cs _ => do
      let p := cs.head
      let hdr ← band (← b2i (← icmp .ule (← iaddImm p 8) e)) (← b2i (← icmp .ult cs.snd (← c MAX_TOP)))
      when .eq hdr zero do brk out %[← c 1]
      let ty ← uload32_64 p
      let n ← uload32_64 (← iaddImm p 4)
      let q ← iaddImm p 8
      when .ugt n (← isub e q) do brk out %[← c 1]
      let end_ ← iadd q n
      let mut st : V .i64 := zero
      let mut done : V .i64 := zero
      for (mi, t) in tops.zipIdx do
        let res ← ifte (jTys := [.i64, .i64]) .eq ty (← c t) (thn := do
          let rec_ ← alloc (← c (recordBytes msgs[mi]!))
          let ok ← ifte (jTys := [.i64]) .eq rec_ zero (thn := do pure %[zero]) (els := do
            let got ← callLocal (⟨.local DEC_IX⟩ : LocalRef P4 (some .i64)) %[q, end_, rec_, ← c mi]
            pure %[← b2i (← icmp .eq got end_)])
          let slot ← iadd (← iaddImm base TOP) (← ishlImm cs.snd 4)
          storeI64 ty slot
          storeI64 rec_ (← iaddImm slot 8)
          pure %[← select (← icmp .ne ok.head zero) zero (← c 1), ← c 1])
          (els := do pure %[st, done])
        st := res.head
        done := res.snd
      let bad ← bor st (← bxor done (← c 1))
      when .ne bad zero do brk out %[← c 1]
      storeI64 (← iaddImm cs.snd 1) (← iaddImm base NTOP)
      pure %[end_, ← iaddImm cs.snd 1])
  pure r.head

/-- **`encode`**: every decoded message, framed, to the output. -/
def encodeEntry (tops : List Nat) : Prog V L (V .i64) := do
  let base ← basePtr
  let zero ← c 0
  let o0 ← outPtr
  let oe ← iadd o0 (← outLen)
  let r ← wloopL %[zero, o0]
    (head := fun _ cs => do pure (contIf .ult cs.head (← load64 (← iaddImm base NTOP)), %[zero], ()))
    (body := fun out cs _ => do
      let slot ← iadd (← iaddImm base TOP) (← ishlImm cs.head 4)
      let ty ← load64 slot
      let rec_ ← load64 (← iaddImm slot 8)
      let mut o := cs.snd
      for (mi, t) in tops.zipIdx do
        let res ← ifte (jTys := [.i64]) .eq ty (← c t) (thn := do
          let n ← callLocal (⟨.local SIZE_IX⟩ : LocalRef P4 (some .i64)) %[rec_, ← c mi, zero, zero]
          let fits ← icmp .ule (← iaddImm (← iadd o n) 8) oe
          ifte (jTys := [.i64]) .ne (← b2i fits) zero
            (thn := do
              storeI32 (← ireduce32 ty) o
              storeI32 (← ireduce32 n) (← iaddImm o 4)
              pure %[← callLocal (⟨.local WRITE_IX⟩ : LocalRef P4 (some .i64)) %[rec_, ← iaddImm o 8, ← c mi, oe]])
            (els := do pure %[zero]))
          (els := do pure %[o])
        o := res.head
      when .eq o zero do brk out %[← c 1]
      pure %[← iaddImm cs.head 1, o])
  pure r.head

end

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The varint helper: params `(p, e)`; where it ends (`0` for an error),
    its value at `SCRATCH`. -/
def varintHelper : Prog V L (V .i64) := do
  let ps ← entryParams P4
  let (q, v) ← readVarintLoop ps.head ps.snd
  storeI64 v (← iaddImm (← basePtr) SCRATCH)
  pure q

/-- The copy helper: params `(q, n)`; `copyBytes`. -/
def copyHelper : Prog V L (V .i64) := do
  let ps ← entryParams P4
  copyBytes ps.head ps.snd

/-- The append helper: params `(slot, v)`; `appendNode`. -/
def appendHelper : Prog V L (V .i64) := do
  let ps ← entryParams P4
  appendNode ps.head ps.snd

/-- The allocation helper: param `n`; `alloc`. -/
def allocHelper : Prog V L (V .i64) := do
  let ps ← entryParams P4
  alloc ps.head

/-- The packed-run helper: params `(q, e, slot, width)`, `width` 0 for
    varints, else 4 or 8 bytes; each value at `q` before `e` appended to the
    repeated slot; where it stopped (`e`, unless an error). -/
def packedHelper : Prog V L (V .i64) := do
  let ps ← entryParams P4
  let q0 := ps.head; let e := ps.snd; let slot := ps.thd; let width := ps.fth
  let zero ← c 0
  let r ← wloopL %[q0]
    (head := fun _ cs => do pure (contIf .ult cs.head e, %[cs.head], ()))
    (body := fun out cs _ => do
      let q := cs.head
      let x ← ifte (jTys := [.i64, .i64]) .eq width zero
        (thn := do
          let (q3, v) ← readVarint q e
          pure %[q3, v])
        (els := do
          let ok ← icmp .ule (← iadd q width) e
          let v ← ifte (jTys := [.i64]) .eq width (← c 4) (thn := do pure %[← uload32_64 q]) (els := do pure %[← load64 q])
          pure %[← select ok (← iadd q width) zero, v.head])
      when .eq x.head zero do brk out %[zero]
      when .eq (← appendNode slot x.snd) zero do brk out %[zero]
      pure %[x.head])
  pure r.head

end

/-- The program: the two entries, the helpers, the decoder, the sizer and
    the writer. -/
def clifIR (msgs : Array Message) (tops : List Nat) : Except String (List FuncData) := do
  -- a table entry keeps a message's index in 16 bits
  if msgs.size > 65536 ∨ (tableBytes msgs).length > TABLE_BYTES then
    throw s!"the schema's {msgs.size} messages are past the decoder's tables"
  Prog.program <|
    [Prog.entry "decode" (Prog.compileProgStatus 0 (decodeEntry msgs tops)),
     Prog.entry "encode" (Prog.compileProgStatus 1 (encodeEntry tops)),
     Prog.compileProgStatus VARINT_IX varintHelper P4,
     Prog.compileProgStatus COPY_IX copyHelper P4,
     Prog.compileProgStatus APPEND_IX appendHelper P4,
     Prog.compileProgStatus ALLOC_IX allocHelper P4,
     Prog.compileProgStatus PACKED_IX packedHelper P4,
     Prog.compileProgStatus DEC_IX decodeBody P4,
     Prog.compileProgStatus SIZE_IX sizeBody P4,
     Prog.compileProgStatus WRITE_IX writeBody P4]

/-- The arena's image: the copy's CPU test name and its bytes, the decoder's
    tables. -/
def initialMemory (msgs : Array Message) : List UInt8 :=
  Memops.writeAt (Memops.writeAt (Memops.writeAt (zeros HEAP) COPY_NAME ("avx2".toUTF8.toList ++ [0]))
    COPY_CODE (Memops.bodyBytes MemopsAsm.copy)) TABLES (tableBytes msgs)

def artifacts (msgs : Array Message) (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "proto" { functions := clif, required_memory := MEM, initial_memory := initialMemory msgs }]

end Proto

namespace Proto

/-- The schema the generator writes a codec for when it is given none: every
    kind of field it takes, nested, repeated and enum-typed, so the tree's
    build makes and checks one real codec. -/
def example_ : String := "
syntax = \"proto2\";
package example;
message Every {
  message Inner {
    optional int32 a = 1;
    repeated string b = 2;
    enum Mode { OFF = 0; ON = 1; }
    optional Mode mode = 3;
  }
  optional int32 i32 = 1;
  optional int64 i64 = 2;
  optional uint32 u32 = 3;
  optional uint64 u64 = 4;
  optional sint32 s32 = 5;
  optional sint64 s64 = 6;
  optional bool flag = 7;
  optional fixed32 f32 = 8;
  optional fixed64 f64 = 9;
  optional sfixed32 sf32 = 10;
  optional sfixed64 sf64 = 11;
  optional float fl = 12;
  optional double db = 13;
  optional string text = 14;
  optional bytes blob = 15;
  optional Inner inner = 16;
  repeated Inner inners = 17;
  repeated int64 nums = 18;
  repeated fixed32 words = 19;
}
"

end Proto

/-- `genprotobench [<schema.proto>] <out dir>`: the codec of that schema, or
    of `Proto.example_` given none. -/
def Bench.Proto.main (args : List String) : IO Unit := do
  let (src, outArgs) ← match args with
    | [out] => pure (Proto.example_, [out])
    | p :: rest => pure (← IO.FS.readFile p, rest)
    | [] => throw (IO.userError "usage: genprotobench [<schema.proto>] <out dir>")
  let (msgs, tops) ← match Proto.schema src with
    | .ok r => pure r
    | .error e => throw (IO.userError e)
  let outDir ← requireOutputDir outArgs
  let clif ← Prog.orDie (Proto.clifIR msgs tops)
  emitArtifacts outDir (Proto.artifacts msgs clif)

#eval ShipScan.check "Bench.Proto" `Bench.Proto.main
