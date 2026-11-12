module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Bench.HashAsm
meta import Bench.HashAsm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace BaseHash

/-!
  # rapidhash and gxhash, as base's own code

  `benchmarks/hash` measures it in SMHasher3 beside the field's hashes. It is
  rapidhash v3 exactly (the 64-bit, unprotected variant SMHasher3 registers
  as `rapidhash`): SMHasher3 checks its verification value, rapidhash's,
  before anything is timed, so a result that differs in one bit anywhere is
  not reported.

  Its function:

  * `rapidhash`: the key as the input; the output's first word holds the
    seed on entry and the hash on return (the C API has no seed argument).
  * `gxhash`: the same, the low 64 bits of gxhash's 128-bit hash (SMHasher3's
    `gxhash-64`), by base's own AES loop (`HashAsm.gx`), mapped the first
    time on a CPU with AES and AVX; status 2 on one without, which gxhash's
    x86 form needs.
-/

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def c (n : Int) : Prog V L (V .i64) := iconst64 n
/-- A constant as the word it is, written below `2^63` as the signed value. -/
def w (n : Nat) : Prog V L (V .i64) := iconst64 (if n ≥ 2 ^ 63 then (n : Int) - 2 ^ 64 else n)

def secret : List Nat :=
  [0x2d358dccaa6c78a5, 0x8bb84b93962eacc9, 0x4b33a62ed433d4a3, 0x4d5a2da51de1aa47,
   0xa0761d6478bd642f, 0xe7037ed1a0b428db, 0x90ed1765281c388c, 0xaaaaaaaaaaaaaaaa]
def s (i : Nat) : Prog V L (V .i64) := w (secret.getD i 0)

/-- The 128-bit product's low and high words. -/
def mum (a b : V .i64) : Prog V L (V .i64 × V .i64) := do pure (← imul a b, ← umulhi a b)
def mix (a b : V .i64) : Prog V L (V .i64) := do
  let (lo, hi) ← mum a b
  bxor lo hi

def r64 (p : V .i64) (off : V .i64) : Prog V L (V .i64) := do load64 (← iadd p off)

/-- `mix(read(p + o) ^ s, read(p + o + 8) ^ acc)`, the step every stage makes. -/
def step (p : V .i64) (o : Nat) (sec : Nat) (acc : V .i64) : Prog V L (V .i64) := do
  mix (← bxor (← load64 (← iaddImm p o)) (← s sec)) (← bxor (← load64 (← iaddImm p (o + 8))) acc)

/-- The last 16-byte steps: step `k` (at `16 · k`) where more than
    `16 · (k + 1)` bytes remain, each nested in the one before, as rapidhash
    writes them; `fuel` the steps left of the six. -/
def tail16 (p i : V .i64) (k : Nat) : (fuel : Nat) → V .i64 → Prog V L (V .i64)
  | 0, acc => pure acc
  | f + 1, acc => do
    let secs : List Nat := [2, 2, 1, 1, 2, 1]
    let r ← ifte (jTys := [.i64]) .ugt i (← c (16 * (k + 1)))
      (thn := do pure %[← tail16 p i (k + 1) f (← step p (16 * k) (secs.getD k 0) acc)])
      (els := do pure %[acc])
    pure r.head

def hashBody : Prog V L (V .i64) := do
  let zero ← c 0
  let r ← ifte (jTys := [.i64]) .uge (← outLen) (← c 8)
    (els := do pure %[← c 1])
    (thn := do
      let p0 ← dataPtr
      let len ← dataLen
      let out ← outPtr
      let seed0 ← load64 out
      let seed ← bxor seed0 (← mix (← bxor seed0 (← s 2)) (← s 1))
      -- (a, b, seed, i) before the final mix
      let ab ← ifte (jTys := [.i64, .i64, .i64, .i64]) .ule len (← c 16)
        (thn := do
          let r ← ifte (jTys := [.i64, .i64, .i64]) .uge len (← c 4)
            (thn := do
              let sd ← bxor seed len
              let x ← ifte (jTys := [.i64, .i64]) .uge len (← c 8)
                (thn := do pure %[← load64 p0, ← r64 p0 (← iaddImm len (-8))])
                (els := do pure %[← uload32_64 p0, ← uload32_64 (← iadd p0 (← iaddImm len (-4)))])
              pure %[x.head, x.snd, sd])
            (els := ifte .ne len zero
              (thn := do
                let a ← bor (← ishlImm (← uload8_64 p0) 45) (← uload8_64 (← iadd p0 (← iaddImm len (-1))))
                pure %[a, ← uload8_64 (← iadd p0 (← ushrImm len 1)), seed])
              (els := do pure %[zero, zero, seed]))
          pure %[r.head, r.snd, r.thd, len])
        (els := do
          -- 112 bytes a trip in seven lanes while more than 112 remain
          let big ← ifte (jTys := [.i64, .i64, .i64]) .ugt len (← c 112)
            (thn := do
              let st ← wloop (Vals.append %[p0, len] (Vals.ofFn (n := 7) fun _ => seed))
                (fun cs => do pure (contIf .ugt cs.snd (← iconst64 112), cs, ()))
                (fun cs _ => do
                  let p := cs.head
                  let v := (Vals.uniformToList (cs.drop 2 : Vals V (List.replicate 7 .i64)))
                  let mut nv : List (V .i64) := []
                  for (acc, k) in v.zipIdx do
                    nv := nv ++ [← step p (16 * k) k acc]
                  pure (Vals.append %[← iaddImm p 112, ← iaddImm cs.snd (-112)]
                    (Vals.ofFn (n := 7) fun j => nv.getD j.val zero)))
              let v := (Vals.uniformToList (st.drop 2 : Vals V (List.replicate 7 .i64)))
              let g (j : Nat) := v.getD j zero
              let sd ← bxor (← bxor (← bxor (g 0) (g 1)) (g 6)) (← bxor (← bxor (g 2) (g 3)) (← bxor (g 4) (g 5)))
              pure %[st.head, st.snd, sd])
            (els := do pure %[p0, len, seed])
          let p := big.head
          let i := big.snd
          -- the rest in 16s, at most six of them
          let sd ← tail16 p i 0 6 big.thd
          let a ← bxor (← r64 p (← iaddImm i (-16))) i
          let b ← r64 p (← iaddImm i (-8))
          pure %[a, b, sd, i])
      let a ← bxor ab.head (← s 1)
      let b ← bxor ab.snd ab.thd
      let (a2, b2) ← mum a b
      let h ← mix (← bxor a2 (← s 7)) (← bxor (← bxor b2 (← s 1)) ab.fth)
      storeI64 h out
      pure %[zero])
  pure r.head

end

/-! ## gxhash's machine code and where it sits

At `SLOT_OFF`, 0 until the code's first use, then where it was mapped or
`UNMAPPED`; at `AES_OFF` and `AVX_OFF` the features it needs, NUL-terminated,
for `base_cpu_has`; from `CODE_OFF`, the code. -/

def SLOT_OFF : Nat := 0
def AES_OFF : Nat := 0x10
def AVX_OFF : Nat := 0x20
def CODE_OFF : Nat := 0x40

def gxBytes : List UInt8 :=
  match X86.assemble HashAsm.gx with
  | .ok bs => bs
  | .error e => panic! e

#guard (X86.assemble HashAsm.gx).isOk

def MEM_SIZE : Nat := (CODE_OFF + gxBytes.length + 63) / 64 * 64

def writeAt (m : List UInt8) (off : Nat) (bs : List UInt8) : List UInt8 :=
  m.take off ++ bs ++ m.drop (off + bs.length)

def initialMemory : List UInt8 :=
  writeAt (writeAt (writeAt (zeros MEM_SIZE) AES_OFF ("aes".toUTF8.toList ++ [0]))
    AVX_OFF ("avx".toUTF8.toList ++ [0])) CODE_OFF gxBytes

/-- The slot's word when the code could not be mapped (no x86-64, no AES or
    AVX, or the map failed). -/
def UNMAPPED : Int := 1

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The code's address, mapping it the first time. -/
def gxMapped : Prog V L (V .i64) := do
  let base ← basePtr
  let at_ ← iaddImm base SLOT_OFF
  when .eq (← load64 at_) (← iconst64 0) do
    let one ← iconst32 1
    let has (off : Nat) : Prog V L (V .i32) := do
      select (← icmp .eq (← cpuHas (← iaddImm base off) (← iconst64 4)) one) one (← iconst32 0)
    let ok ← band (← select (← icmp .eq (← nativeArch) one) one (← iconst32 0)) (← band (← has AES_OFF) (← has AVX_OFF))
    let fn ← ifte (jTys := [.i64]) .ne ok (← iconst32 0)
      (thn := do pure %[← nativeLoad (← iaddImm base CODE_OFF) (← iconst64 gxBytes.length)])
      (els := do pure %[← iconst64 0])
    storeI64 (← select (← icmp .eq fn.head (← iconst64 0)) (← iconst64 UNMAPPED) fn.head) at_
  load64 at_

def gxBody : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .uge (← outLen) (← c 8)
    (els := do pure %[← c 1])
    (thn := do
      let fn ← gxMapped
      let r2 ← ifte (jTys := [.i64]) .eq fn (← iconst64 UNMAPPED)
        (thn := do pure %[← c 2])
        (els := do
          let out ← outPtr
          storeI64 (← nativeCall fn (← dataPtr) (← dataLen) (← load64 out) (← c 0)) out
          pure %[← c 0])
      pure %[r2.head])
  pure r.head

end

def clifIR : Except String (List FuncData) :=
  Prog.program [Prog.entry "rapidhash" (Prog.compileProgStatus 0 hashBody),
    Prog.entry "gxhash" (Prog.compileProgStatus 1 gxBody)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "hash" { functions := clif, required_memory := MEM_SIZE, initial_memory := initialMemory }]

end BaseHash

def Bench.Hash.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie BaseHash.clifIR
  emitArtifacts outDir (BaseHash.artifacts clif)

#eval ShipScan.check "Bench.Hash" `Bench.Hash.main
