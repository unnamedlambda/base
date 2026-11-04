module
public import Lz4.CompressorBlock
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeGrid` — the compressor's blocks, in order

Each block runs from the world the blocks before it leave (`block_run`): its
input bytes are the launch's, since every block's byte stores land in its own
output (`gstOf_O`, `blockStores_O`) and no block's output meets a later
block's input; its output may still be stored. So the first `n` blocks run,
and every byte outside shared memory is the launch's with each block's byte
stores made in block order (`grid_runs`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

open AlgorithmLib.ML.LaneWarp (NotArena Writable)

namespace Lz4Code

/-- The warp's byte stores land in `O`. -/
theorem gstOf_O {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} :
    ∀ (s : WStmt) (f : Nat) (ws : WState), okOf I O okC f s ws → ∀ x ∈ gstOf f s ws, O x.2.1 := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f ws hok x hx; simp only [okOf] at hok; simp only [gstOf, List.mem_append] at hx
      rcases hx with h | h
      · exact iha f ws hok.1 x h
      · exact ihb f _ hok.2 x h
  | uif c t e iht ihe =>
      intro f ws hok x hx; simp only [okOf] at hok
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hok; simp only [gstOf, hc, if_true] at hx; exact iht f ws hok x hx
      · simp only [hc, Bool.false_eq_true, if_false] at hok
        simp only [gstOf, hc, Bool.false_eq_true, if_false] at hx; exact ihe f ws hok x hx
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws _ x hx; simp [gstOf] at hx
      | succ f ihf =>
          intro ws hok x hx; simp only [okOf] at hok
          by_cases hc : (ws.regs c == 1) = true
          · simp only [gstOf, hc, if_true, List.mem_append] at hx
            rcases hx with h | h
            · exact ih f ws (hok hc).1 x h
            · exact ihf _ (hok hc).2 x h
          · simp [gstOf, hc] at hx
  | stgB addr val =>
      intro f ws hok x hx; simp only [okOf] at hok; simp only [gstOf, List.mem_singleton] at hx
      subst hx; exact hok.2.2
  | coopCopy dst src len =>
      intro f ws hok x hx; simp only [okOf] at hok
      simp only [gstOf, copyTr, List.mem_map, List.mem_range] at hx
      obtain ⟨i, hi, rfl⟩ := hx
      exact (hok.2.2.2.2 i hi).2.2
  | _ => intro f ws _ x hx; simp [gstOf] at hx

/-- A block's byte stores: each thread's share of the warp's, in thread order. -/
def blockStores (f : Nat) (body : WStmt) (ws : WState) : List (UInt64 × UInt64) :=
  (List.range 32).flatMap fun t => lanePart t (gstOf f body ws)

theorem blockStores_O {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} {f : Nat} {body : WStmt} {ws : WState}
    (hok : okOf I O okC f body ws) : ∀ x ∈ blockStores f body ws, O x.1 := by
  intro x hx
  simp only [blockStores, List.mem_flatMap] at hx
  obtain ⟨t, -, hx⟩ := hx
  obtain ⟨who, hm⟩ := lanePart_mem hx
  exact gstOf_O body f ws hok _ hm

set_option maxHeartbeats 2000000 in
/-- **The compressor's blocks run, in order**: the first `n` blocks of the
    grid, each from the world the blocks before it leave, run, and every byte
    outside shared memory is the launch's with each block's byte stores made
    in block order; given, for each block, that the warp's run of its body
    makes only the accesses `okOf` allows, its input `I c` holds its bytes
    `G c` at the start and no block before it stores into it, and its output
    `O c` may be stored. -/
theorem grid_runs (hw : Hw) (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) (name : String) {numBlocks inStride outStride lenOff hashLog f : Nat}
    {inP outA : UInt64} (I O : Nat → UInt64 → Prop) (G : Nat → Array UInt8) (w : World)
    (hnb : numBlocks * 32 + 32 < 2 ^ 32) (h5 : 5 ≤ hashLog) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hf : f ≤ 2 ^ 64)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog)))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIna : ∀ c a, I c a → NotArena a) (hOna : ∀ c a, O c a → NotArena a)
    (hok : ∀ c, c < numBlocks → okOf (I c) (O c) (okLZ4 (I c)) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog c inP (G c)))
    (hI : ∀ c, c < numBlocks → ∀ a, I c a → w.mem.load a 1 = some ((G c).getD a.toNat 0).toUInt64)
    (hO : ∀ c, c < numBlocks → ∀ a, O c a → Writable w.mem a 1)
    (hdis : ∀ c c', c' < c → c < numBlocks → ∀ a, O c' a → ¬ I c a) (hnan : w.mem.nan = hw.nan)
    (hlent0 : w.mem.lent = []) :
    ∀ n, n ≤ numBlocks → ∃ u, BlocksRun (compressorKernel name numBlocks inStride outStride lenOff hashLog) 32
        [.sc .i64 inP, .sc .i64 outA] hw n w u ∧
      (∀ a, NotArena a → u.mem.load a 1 = replay ((List.range n).flatMap fun c =>
        blockStores f (compressorBodyWStmt inStride lenOff hashLog)
          (startWS numBlocks inStride outStride hashLog c inP (G c))) a (w.mem.load a 1)) ∧
      (∀ a, NotArena a → Writable w.mem a 1 → Writable u.mem a 1) ∧ u.mem.nan = hw.nan ∧ u.mem.lent = [] := by
  intro n
  induction n with
  | zero => intro _; exact ⟨w, rfl, fun a _ => rfl, fun _ _ h => h, hnan, hlent0⟩
  | succ n ih =>
      intro hn
      obtain ⟨v, hv, hvm, hvw, hvn, hvl⟩ := ih (by omega)
      obtain ⟨u, hu, hum, huw, hun, hul⟩ := block_run (cta := n) (outA := outA) (I := I n) (O := O n) (G := G n) hw hw32
        name v (by omega) (by omega) (by omega) h5 hh hf hfine hflags (hIna n) (hOna n) (hok n (by omega))
        (fun a ha => by
          rw [hvm a (hIna n a ha), replay_notin]
          · exact hI n (by omega) a ha
          · intro x hx e
            simp only [List.mem_flatMap, List.mem_range] at hx
            obtain ⟨c', hc', hx⟩ := hx
            exact hdis n c' hc' (by omega) a (e ▸ blockStores_O (hok c' (by omega)) x hx) ha)
        (fun a ha => hvw a (hOna n a ha) (hO n (by omega) a ha)) hvn hvl
      refine ⟨u, ⟨v, hv, hu⟩, fun a ha => ?_, fun a ha h => huw a ha (hvw a ha h), hun, hul⟩
      rw [hum a ha, hvm a ha, show List.range (n + 1) = List.range n ++ [n] from List.range_succ, List.flatMap_append,
        replay_append]
      simp [blockStores]

end Lz4Code
