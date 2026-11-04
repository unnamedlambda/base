module
public import Lz4.CodeLayout
meta import Lz4.CodeLayout
public import Lz4.Kernel
meta import Lz4.Kernel
@[expose] public section

/-!
# `Lz4.CodeShip` — the compressor at the shipped geometry

The artifact's kernel (`Lz4Ship.WP.code`) at both block sizes it ships, 32 KiB
and 64 KiB, a 4096-entry table: it compiles to PTX, and a launch on device
memory big enough for the inputs and outputs round-trips every block, on the
Code and on the PTX (`shipped_roundtrips`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel LZ4WarpEvalBytes
  AlgorithmLib.LZ4Plan

namespace Lz4Code

/-- The bytes a launch at block size `2 ^ b` reads and writes in buffer 0. -/
abbrev _root_.Lz4Ship.WP.bytes (w : Lz4Ship.WP) : Nat := w.numBlk * w.inStride + copySlack + w.numBlk * w.outStride

theorem code16 : (Lz4Ship.WP.mk 16).code = compressorKernel "main" 3200 65536 69896 69888 12 := rfl
theorem code15 : (Lz4Ship.WP.mk 15).code = compressorKernel "main" 6400 32768 35080 35072 12 := rfl

theorem fine16 : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (ML.Body.body 2 (compressorS 3200 65536 69896 69888 12)))
    ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2 :=
  ⟨by native_decide, by native_decide⟩

theorem fine15 : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (ML.Body.body 2 (compressorS 6400 32768 35080 35072 12)))
    ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2 :=
  ⟨by native_decide, by native_decide⟩

theorem flags16 : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
    (flagRegs (compressorBodyWStmt 65536 69888 12)).contains r = false := by decide

theorem flags15 : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
    (flagRegs (compressorBodyWStmt 32768 35072 12)).contains r = false := by decide

theorem compiles16 : (Lz4Ship.WP.mk 16).code.entry.isOk = true := by native_decide
theorem compiles15 : (Lz4Ship.WP.mk 15).code.entry.isOk = true := by native_decide

/-- What the shipped round-trip says, at one geometry: the kernel compiles, a
    launch runs on the Code and on the PTX that ships (`Kernel.shipped`) to the
    same memory, the inputs and
    slack are as they were, buffer 0 still holds the layout, and each block's
    output holds its length and decodes to its input. -/
def RoundTrips (k : AlgorithmLib.Kernel.Kernel) (nb iS oS lO : Nat) (hw : Hw) (outA : UInt64) (bufs : Nat → ByteArray) :
    Prop :=
  ∃ e, k.shipped = .ok e ∧
  ∃ u : World, LaunchRuns k nb 32 [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw ∧
    RunsLaunch (PtxRun.runner e hw) k.smem nb 32 [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw.nan ∧
    (∀ o, o < nb * iS + copySlack → (u.mem.bufs 0).get! o = (bufs 0).get! o) ∧
    nb * iS + copySlack + nb * oS ≤ (u.mem.bufs 0).size ∧
    ∀ c, c < nb → ∃ n, n ≤ lO ∧
      u32At (u.mem.bufs 0) (nb * iS + copySlack + c * oS + lO) = n ∧
      AlgorithmLib.LZ4Imp.decompress ((List.range n).map fun i =>
          (u.mem.bufs 0).get! (nb * iS + copySlack + c * oS + i)) iS
        = some ((List.range iS).map fun j => (bufs 0).get! (c * iS + j))

theorem roundtrips16 (hw : Hw) (hhw : PtxSound.HwOk hw) (outA : UInt64) (bufs : Nat → ByteArray)
    (hsize : 3200 * 65536 + copySlack + 3200 * 69896 ≤ (bufs 0).size) :
    RoundTrips (compressorKernel "main" 3200 65536 69896 69888 12) 3200 65536 69896 69888 hw outA bufs := by
  have hc : (compressorKernel "main" 3200 65536 69896 69888 12).entry.isOk = true := by rw [← code16]; exact compiles16
  cases he : (compressorKernel "main" 3200 65536 69896 69888 12).entry with
  | error _ => rw [he] at hc; cases hc
  | ok e =>
      refine ⟨Coalesce.run e, PtxSound.shipped_ok _ e he, ?_⟩
      rw [PtxSound.runner_shipped e hw hhw]
      exact launch_roundtrips hw hhw "main" outA he bufs (by decide) (by decide) (by decide) (by decide)
        fine16 flags16 (by decide) (by decide) (by decide) (by decide) hsize (by decide)

theorem roundtrips15 (hw : Hw) (hhw : PtxSound.HwOk hw) (outA : UInt64) (bufs : Nat → ByteArray)
    (hsize : 6400 * 32768 + copySlack + 6400 * 35080 ≤ (bufs 0).size) :
    RoundTrips (compressorKernel "main" 6400 32768 35080 35072 12) 6400 32768 35080 35072 hw outA bufs := by
  have hc : (compressorKernel "main" 6400 32768 35080 35072 12).entry.isOk = true := by rw [← code15]; exact compiles15
  cases he : (compressorKernel "main" 6400 32768 35080 35072 12).entry with
  | error _ => rw [he] at hc; cases hc
  | ok e =>
      refine ⟨Coalesce.run e, PtxSound.shipped_ok _ e he, ?_⟩
      rw [PtxSound.runner_shipped e hw hhw]
      exact launch_roundtrips hw hhw "main" outA he bufs (by decide) (by decide) (by decide) (by decide)
        fine15 flags15 (by decide) (by decide) (by decide) (by decide) hsize (by decide)

/-- **The shipped compressor round-trips every block, on the Code and on the
    PTX it compiles to**, at both shipped block sizes, 32 KiB and 64 KiB: it
    compiles, and a launch on device memory whose buffer 0 holds the input
    blocks then room for their outputs runs; each block's output holds its
    encoded length at `lenOff` and begins with an encoded block that the LZ4
    decompressor turns back into its input. -/
theorem shipped_roundtrips (b : Nat) (hb : b = 15 ∨ b = 16) (hw : Hw) (hhw : PtxSound.HwOk hw) (outA : UInt64)
    (bufs : Nat → ByteArray) (hsize : (Lz4Ship.WP.mk b).bytes ≤ (bufs 0).size) :
    RoundTrips (Lz4Ship.WP.mk b).code (Lz4Ship.WP.mk b).numBlk (Lz4Ship.WP.mk b).inStride
      (Lz4Ship.WP.mk b).outStride (Lz4Ship.WP.mk b).lenOff hw outA bufs := by
  rcases hb with rfl | rfl
  · rw [code15, show (Lz4Ship.WP.mk 15).numBlk = 6400 from rfl, show (Lz4Ship.WP.mk 15).inStride = 32768 from rfl,
      show (Lz4Ship.WP.mk 15).outStride = 35080 from rfl, show (Lz4Ship.WP.mk 15).lenOff = 35072 from rfl]
    exact roundtrips15 hw hhw outA bufs hsize
  · rw [code16, show (Lz4Ship.WP.mk 16).numBlk = 3200 from rfl, show (Lz4Ship.WP.mk 16).inStride = 65536 from rfl,
      show (Lz4Ship.WP.mk 16).outStride = 69896 from rfl, show (Lz4Ship.WP.mk 16).lenOff = 69888 from rfl]
    exact roundtrips16 hw hhw outA bufs hsize

/-- `n` launches in turn on the same device memory, each on the Code and on the
    PTX entry `e` that ships. -/
def LaunchesRun (k : AlgorithmLib.Kernel.Kernel) (e : PtxIr.Entry) (nb : Nat) (params : List V) (hw : Hw) :
    Nat → (Nat → ByteArray) → (Nat → ByteArray) → Prop
  | 0, b, b' => b' = b
  | n + 1, b, b'' => ∃ b', LaunchRuns k nb 32 params b b' hw ∧
      RunsLaunch (PtxRun.runner e hw) k.smem nb 32 params b b' hw.nan ∧ LaunchesRun k e nb params hw n b' b''

/-- **The shipped compressor's repeated launches round-trip every block**, as
    the artifact runs it (`rLaunches` times, `Lz4CompSafe.warp15_calls`): after
    any positive number of launches on the same memory, each block's output
    holds its length and decodes to the block's input as it was before the
    first. -/
theorem shipped_launches (b : Nat) (hb : b = 15 ∨ b = 16) (hw : Hw) (hhw : PtxSound.HwOk hw) (outA : UInt64)
    (bufs : Nat → ByteArray) (hsize : (Lz4Ship.WP.mk b).bytes ≤ (bufs 0).size) (n : Nat) (hn : 0 < n) :
    let w := Lz4Ship.WP.mk b
    ∃ e, w.code.shipped = .ok e ∧ ∃ bN, LaunchesRun w.code e w.numBlk [.sc .i64 base0, .sc .i64 outA] hw n bufs bN ∧
      ∀ c, c < w.numBlk → ∃ m, m ≤ w.lenOff ∧
        u32At (bN 0) (w.numBlk * w.inStride + copySlack + c * w.outStride + w.lenOff) = m ∧
        AlgorithmLib.LZ4Imp.decompress ((List.range m).map fun i =>
            (bN 0).get! (w.numBlk * w.inStride + copySlack + c * w.outStride + i)) w.inStride
          = some ((List.range w.inStride).map fun j => (bufs 0).get! (c * w.inStride + j)) := by
  intro w
  obtain ⟨e, he, -⟩ := shipped_roundtrips b hb hw hhw outA bufs hsize
  refine ⟨e, he, ?_⟩
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  clear hn
  induction k generalizing bufs with
  | zero =>
      obtain ⟨e', he', u, hL, hP, -, -, hdec⟩ := shipped_roundtrips b hb hw hhw outA bufs hsize
      rw [he] at he'; cases he'
      exact ⟨u.mem.bufs, ⟨u.mem.bufs, hL, hP, rfl⟩, hdec⟩
  | succ k ih =>
      obtain ⟨e', he', u, hL, hP, hin, hsz, -⟩ := shipped_roundtrips b hb hw hhw outA bufs hsize
      rw [he] at he'; cases he'
      obtain ⟨bN, hrun, hdec⟩ := ih u.mem.bufs hsz
      refine ⟨bN, ⟨u.mem.bufs, hL, hP, hrun⟩, fun c hc => ?_⟩
      obtain ⟨m, hm, hlen, hd⟩ := hdec c hc
      have hc2 : c < (Lz4Ship.WP.mk b).numBlk := hc
      have hc' : c * (Lz4Ship.WP.mk b).inStride + (Lz4Ship.WP.mk b).inStride ≤
          (Lz4Ship.WP.mk b).numBlk * (Lz4Ship.WP.mk b).inStride := by
        have := Nat.mul_le_mul_right (Lz4Ship.WP.mk b).inStride (show c + 1 ≤ (Lz4Ship.WP.mk b).numBlk by omega)
        rw [Nat.succ_mul] at this; exact this
      have hmap : ((List.range (Lz4Ship.WP.mk b).inStride).map fun j =>
            (u.mem.bufs 0).get! (c * (Lz4Ship.WP.mk b).inStride + j)) =
          (List.range (Lz4Ship.WP.mk b).inStride).map fun j => (bufs 0).get! (c * (Lz4Ship.WP.mk b).inStride + j) :=
        List.map_congr_left fun j hj => hin _ (by have := List.mem_range.mp hj; omega)
      exact ⟨m, hm, hlen, hd.trans (congrArg some hmap)⟩

end Lz4Code
