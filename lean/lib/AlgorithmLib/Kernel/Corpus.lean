module
public import Lean
public import AlgorithmLib.Kernel.Run
meta import AlgorithmLib.Kernel.Run
public import AlgorithmLib.Kernel.PtxRun
meta import AlgorithmLib.Kernel.PtxRun
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Corpus` — kernels, their inputs, and what the model says they leave

A corpus case is a kernel, its geometry, its arguments and its buffers;
`write` prints each case's PTX and runs the model's launch (`Kernel.launch`)
on it, leaving files `tools/kernel_agree.py` launches on the GPU and compares
with, bit for bit. It also runs the PTX the case compiles to under PTX's
semantics (`runPtx`) and says whether that leaves the same buffers, up to a
NaN's payload, which the body's semantics and the GPU's differ on. An
application builds its own cases and calls `write`.
-/

open Lean
open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Kernel.Corpus

/-- A fixed pseudo-random sequence. -/
def lcg (seed n : Nat) : List Nat :=
  (List.range n).foldl (fun (s, acc) _ => let s' := (s * 6364136223846793005 + 1442695040888963407) % 2 ^ 64
    (s', (s' / 2 ^ 32) :: acc)) (seed, []) |>.2.reverse

def le32 (x : Nat) : List UInt8 := (List.range 4).map fun i => UInt8.ofNat (x / 256 ^ i)

/-- `n` floats in `[-2, 2)`, then the special values. -/
def floats (seed n : Nat) : ByteArray :=
  let xs := (lcg seed n).map fun r => ((r % 4096).toFloat / 1024.0 - 2.0).toFloat32.toBits.toNat
  let specials := [0x7f800000, 0xff800000, 0x80000000, 0x00000001, 0x7fc00000]
  ⟨((specials ++ xs).take n).flatMap le32 |>.toArray⟩

def f32s (xs : List Float) (extra : List Nat := []) : ByteArray :=
  ⟨(xs.map (·.toFloat32.toBits.toNat) ++ extra).flatMap le32 |>.toArray⟩

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

/-- One case: the kernel, its geometry, its arguments after the buffers'
    addresses, and its buffers. -/
structure Case where
  kernel : Kernel
  grid : Nat
  block : Nat
  /-- How many of the arguments are buffer addresses, given first. -/
  nbufs : Nat
  scalars : List (String × Nat)
  bufs : List ByteArray
  /-- How far, in units in the last place, the GPU may be from the model on
      an f32 word: zero, unless the kernel uses the hardware's approximations,
      which the model runs exactly rounded. -/
  ulps : Nat := 0

def scalarV : String × Nat → V
  | ("u32", n) => .sc .i32 (UInt64.ofNat n)
  | ("f32", b) => .sc .f32 (UInt64.ofNat b)
  | (_, n) => .sc .i64 (UInt64.ofNat n)

def kernelOf (name : String) (params : List ClifTy) (p : Body) (smem : Nat := 0) : Except String Kernel := do
  let code ← emitChecked p (idParams ++ params)
  pure { name, params, code, smem }

def bufArgs (c : Case) : List V :=
  (List.range c.nbufs).map fun i => V.sc .i64 (if h : i < 268435449 then addrOf (.buf ⟨i, h⟩) 0 else 0)

def run (c : Case) : Except String (List ByteArray) :=
  launch c.kernel c.grid c.block (bufArgs c ++ c.scalars.map scalarV) c.bufs

/-- The case run as the PTX it compiles to, by `PtxRun`. -/
def runPtx (c : Case) : Except String (List ByteArray) := do
  let e ← c.kernel.entry
  let out ← PtxRun.launchOn e (max 16 c.kernel.smem) c.grid c.block (bufArgs c ++ c.scalars.map scalarV)
    fun i => c.bufs.getD i .empty
  pure ((List.range c.bufs.length).map out)

/-- Whether two buffers hold the same bytes, a 4-byte word both hold as an
    f32 NaN matching whatever its payload. -/
def sameUpToNaN (a b : ByteArray) : Bool :=
  let word (x : ByteArray) (k : Nat) : Nat :=
    (List.range 4).foldl (fun acc j => acc + (x.get! (4 * k + j)).toNat * 256 ^ j) 0
  let nan (w : Nat) := w / 2 ^ 23 % 256 == 255 && w % 2 ^ 23 != 0
  a.size == b.size && (List.range (a.size / 4)).all (fun k =>
    word a k == word b k || (nan (word a k) && nan (word b k))) &&
    (List.range (a.size % 4)).all fun j => a.get! (a.size / 4 * 4 + j) == b.get! (a.size / 4 * 4 + j)

/-- Each case's PTX, inputs and the model's outputs under `dir/kernel_corpus`,
    with an index `tools/kernel_agree.py` reads. -/
def write (dir : System.FilePath) (cs : List Case) : IO Unit := do
  let mut index : Array Json := #[]
  for c in cs do
    let ptx ← IO.ofExcept c.kernel.ptx
    let outs ← IO.ofExcept (run c)
    let d := dir / "kernel_corpus" / c.kernel.name
    IO.FS.createDirAll d
    IO.FS.writeFile (d / "kernel.ptx") ptx
    for (b, i) in c.bufs.zipIdx do IO.FS.writeBinFile (d / s!"in{i}.bin") b
    for (b, i) in outs.zipIdx do IO.FS.writeBinFile (d / s!"out{i}.bin") b
    index := index.push <| Json.mkObj [("name", toJson c.kernel.name), ("grid", toJson c.grid),
      ("block", toJson c.block), ("nbufs", toJson c.nbufs),
      ("scalars", toJson (c.scalars.map fun (k, v) => [toJson k, toJson v])), ("ulps", toJson c.ulps)]
    let ptx ← IO.ofExcept (runPtx c)
    let agree := (outs.zip ptx).all fun (a, b) => sameUpToNaN a b
    IO.println s!"{c.kernel.name}: {outs.length} buffers; as PTX {if agree then "the same" else "DIFFERENT"}"
  IO.FS.writeFile (dir / "kernel_corpus" / "index.json") (Json.arr index).compress

end AlgorithmLib.Kernel.Corpus
