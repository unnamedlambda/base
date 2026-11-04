module
public import AlgorithmLib.ML.Ptx.BodyPhase
meta import AlgorithmLib.ML.Ptx.BodyPhase
public import AlgorithmLib.ML.Kernel.Library
meta import AlgorithmLib.ML.Kernel.Library
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Kernel.BodyMap` — a map kernel's launch computes its spec

`mapKernel_body`: `g` one-warp blocks of a map kernel's body
(`ML.Body.kernel` of `mapKernel spec inB out`), each lane on its own element
`ctaid·32 + laneid`, leave output word `j` as the spec's value at `F32` over
input words `j`, with the hardware's approximations `hw`'s. An input may be
the output buffer; it is read before it is written. The compiled expression
only moves registers (`compileW_ok`), the loads and the store touch the
lane's own element (`elemAcc`), and `launch_lockstep` does the rest.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase

namespace AlgorithmLib.ML.BodyMap

/-- The elements a map kernel's lane touches: its own, in the inputs and the
    output, in the launch's `g` blocks. -/
def elemAcc {Γ : Nat} (inB : Fin Γ → Buf) (out : Buf) (g cta : Nat) : Acc :=
  fun l b j => cta < g ∧ (b = out ∨ ∃ i, inB i = b) ∧ j = cta * 32 + l.val

theorem elemIx_eval (cta i : Nat) (l : Lane) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    elemIx.eval cta i l ir im = cta * 32 + l.val := rfl

theorem idxL_elem (rd : Buf → Nat → Option Nat) (cta i l : Nat) (h : cta * 32 + l < 2 ^ 32) :
    (idxL rd cta i l elemIx).isSome := by
  have h1 : cta < 2 ^ 32 := by omega
  have h2 : cta * 32 < 2 ^ 32 := by omega
  have h3 : l < 2 ^ 32 := by omega
  simp [idxL, elemIx, h1, h2, h3, h]

/-- Statements that only move registers: every index fits and every access
    is allowed, there being none. -/
def RegOnly (rd : Buf → Nat → Option Nat) (cta : Nat) (A : Acc) (s : EWStmt) : Prop :=
  ∀ i, fits rd cta i s = true ∧ AccOk cta A rd i s

theorem sumSeq_regOnly (rd : Buf → Nat → Option Nat) (cta : Nat) (A : Acc) (acc : Nat) :
    ∀ (L : List (EWStmt × WFExp)) (init : EWStmt), RegOnly rd cta A init → (∀ p ∈ L, RegOnly rd cta A p.1) →
      RegOnly rd cta A (L.foldl (fun s (p : EWStmt × WFExp) => .seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2)))) init) := by
  intro L
  induction L with
  | nil => intro init h _; exact h
  | cons p L ih =>
    intro init h hL
    refine ih _ (fun i => ?_) (fun q hq => hL q (List.mem_cons_of_mem _ hq))
    have hp := hL p List.mem_cons_self i
    simp only [fits, Bool.and_eq_true, AccOk]
    exact ⟨⟨(h i).1, hp.1, trivial⟩, (h i).2, hp.2, trivial⟩

/-- **The compiler only moves registers.** -/
theorem compileW_ok (rd : Buf → Nat → Option Nat) (cta : Nat) (A : Acc) : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    RegOnly rd cta A (compileW ve c e).1 := by
  intro Γ e
  induction e with
  | var _ => intro _ _ _; exact ⟨rfl, trivial⟩
  | lit _ => intro _ _ _; exact ⟨rfl, trivial⟩
  | add a b iha ihb =>
    intro ve c i
    simp only [compileW, fits, Bool.and_eq_true, AccOk]
    exact ⟨⟨(iha ve c i).1, (ihb ve _ i).1⟩, (iha ve c i).2, (ihb ve _ i).2⟩
  | mul a b iha ihb =>
    intro ve c i
    simp only [compileW, fits, Bool.and_eq_true, AccOk]
    exact ⟨⟨(iha ve c i).1, (ihb ve _ i).1⟩, (iha ve c i).2, (ihb ve _ i).2⟩
  | neg a ih => intro ve c; exact ih ve c
  | inv a ih => intro ve c; exact ih ve c
  | exp a ih => intro ve c; exact ih ve c
  | rsqrt a ih => intro ve c; exact ih ve c
  | sum n f ih =>
    intro ve c
    refine sumSeq_regOnly rd cta A c _ _ (fun _ => ⟨rfl, trivial⟩) ?_
    intro p hp
    obtain ⟨j, _, hj⟩ := List.mem_map.mp hp
    rw [← hj]
    exact ih j ve (c + 1)
  | letE a b iha ihb =>
    intro ve c i
    simp only [compileW, fits, Bool.and_eq_true, AccOk]
    exact ⟨⟨⟨(iha ve c i).1, trivial⟩, (ihb _ _ i).1⟩, ⟨(iha ve c i).2, trivial⟩, (ihb _ _ i).2⟩

theorem loadSeq_ok {Γ : Nat} (rd : Buf → Nat → Option Nat) (inB : Fin Γ → Buf) (out : Buf) (g cta : Nat) (hc : cta < g)
    (hg : g * 32 ≤ 2 ^ 32) :
    ∀ k, RegOnly rd cta (elemAcc inB out g cta) (loadSeqN inB (fun _ => elemIx) k) := by
  intro k
  induction k with
  | zero => exact fun _ => ⟨rfl, trivial⟩
  | succ k ih =>
    intro i
    simp only [loadSeqN, fits, Bool.and_eq_true, AccOk]
    refine ⟨⟨(ih i).1, ?_⟩, (ih i).2, ?_⟩
    · split
      · simp only [fits, List.all_eq_true, List.mem_range]
        intro l hl; exact idxL_elem rd cta i l (by omega)
      · rfl
    · split
      · rename_i h
        intro l
        exact ⟨hc, Or.inr ⟨⟨k, h⟩, rfl⟩, rfl⟩
      · trivial

theorem mapKernel_ew {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf) :
    (mapKernel spec inB out).ew = compileWKernel inB out (fun _ => elemIx) spec elemIx (Γ + slots spec + 1) := rfl

/-- **A map kernel's launch computes its spec**: `g` blocks of `nw` warps of
    the body of `mapKernel spec inB out` leave each output word `j < 32·g·nw`
    as `spec`'s value at `F32`, with `hw`'s approximations, over input words
    `j` as the launch found them. -/
theorem mapKernel_body {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf) (name : String)
    (nbuf nw g : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (hB : ∀ i, inB i < nbuf) (hout : out < nbuf) (hn : nbuf ≤ 268435449) (hg' : g * nw * 32 ≤ 2 ^ 32)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf (mapKernel spec inB out).ew nw))
      ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray)
    (hszI : ∀ i, 4 * (g * nw * 32) ≤ (bufs (inB i)).size) (hszO : 4 * (g * nw * 32) ≤ (bufs out).size)
    (hr : LaunchRuns (Body.kernel name nbuf 0 (mapKernel spec inB out).ew nw) g (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ j, j < g * nw * 32 → ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf out j) 4 =
      some (@denote F32 (F32.ops hw) Γ (fun i => ⟨((kmem bufs).load (bufW nbuf (inB i) j) 4).getD 0⟩) spec).bits := by
  intro j hj
  generalize hG : g * nw = G at hj hszI hszO hg'
  have hgpos : 0 < g := by rcases Nat.eq_zero_or_pos g with h | h
                           · subst h; simp at hG; omega
                           · exact h
  have hN : 32 * nw ≤ 2 ^ 32 := by
    have : nw ≤ g * nw := Nat.le_mul_of_pos_left nw hgpos
    omega
  have hl : j % 32 < 32 := Nat.mod_lt _ (by decide)
  have hjd : j / 32 * 32 + j % 32 = j := Nat.div_add_mod' j 32
  have hcg : j / 32 < G := by omega
  have hacc : ∀ cta, cta < G → AccOk cta (elemAcc inB out G cta) (fun _ _ => none) 0 (mapKernel spec inB out).ew := by
    intro cta hc
    rw [mapKernel_ew]
    simp only [compileWKernel, AccOk]
    refine ⟨⟨(loadSeq_ok _ inB out G cta hc (by omega) Γ 0).2, (compileW_ok _ cta _ spec _ _ 0).2, trivial⟩, fun l => ?_⟩
    refine ⟨⟨hc, Or.inl rfl, rfl⟩, fun l' h => Fin.ext ?_⟩
    have := h.2.2; rw [elemIx_eval] at this; omega
  have hfits : ∀ cta, cta < G → fits (fun _ _ => none) cta 0 (mapKernel spec inB out).ew = true := by
    intro cta hc
    rw [mapKernel_ew]
    simp only [compileWKernel, fits, Bool.and_eq_true, List.all_eq_true, List.mem_range]
    exact ⟨⟨(loadSeq_ok _ inB out G cta hc (by omega) Γ 0).1, (compileW_ok _ cta (elemAcc inB out G cta) spec _ _ 0).1, trivial⟩,
      fun l hl' => idxL_elem _ cta 0 l (by omega)⟩
  have h := launch_lockstep hw hw32 name nbuf 0 (mapKernel spec inB out).ew nw hnw hnw31 g (by omega) hN
    (elemAcc inB out G) (fun _ _ => none) (by rw [hG]; exact hacc)
    (fun c l b j h => by
      obtain ⟨h1, h2, h3⟩ := h
      have : l.val < 32 := l.isLt
      have hb : b < nbuf := by
        rcases h2 with h2 | ⟨i, h2⟩
        · subst h2; exact hout
        · subst h2; exact hB i
      exact ⟨hb, Nat.lt_of_lt_of_le hb hn, by omega⟩)
    (fun c c' l l' b j hc hc' hne h h' => by
      rw [hG] at hc hc'
      have h'' := stores_acc c' (elemAcc inB out G c') (fun _ _ => none) l' _ 0 b j (hacc c' hc') h'
      have hl1 : l.val < 32 := l.isLt
      have hl2 : l'.val < 32 := l'.isLt
      have e1 := h.2.2
      have e2 := h''.2.2
      exact hne (by omega))
    hfine bufs bufs' (fun _ _ _ h => by cases h) (fun _ _ _ _ _ _ => rfl) (by rw [hG]; exact hfits)
    (fun c l b j hc h => by
      obtain ⟨h1, h2, h3⟩ := h
      have : l.val < 32 := l.isLt
      have hs : 4 * (G * 32) ≤ (bufs b).size := by
        rcases h2 with h2 | ⟨i, h2⟩
        · subst h2; exact hszO
        · subst h2; exact hszI i
      omega)
    hr (j / 32) ⟨j % 32, hl⟩ out j (by rw [hG]; exact hcg) ⟨hcg, Or.inl rfl, hjd.symm⟩
  rw [h]
  congr 2
  have hs := @compileWKernel_storesAt Γ F32 (F32.ops hw) _ (litNatF32 hw) inB out (fun _ => elemIx) spec elemIx
    (Γ + slots spec + 1) (j / 32) 0 (fun _ _ => 0) (fun _ _ => 0) (ws0 nbuf (kmem bufs)) ⟨j % 32, hl⟩
    (fun l h => by
      have : l = ⟨j % 32, hl⟩ := Fin.ext (by simp [elemIx, IdxE.eval] at h; omega)
      subst this; rfl)
  have hj' : elemIx.eval (j / 32) 0 ⟨j % 32, hl⟩ (fun _ _ => 0) (fun _ _ => 0) = j := hjd
  rw [hj'] at hs
  rw [mapKernel_ew]
  exact hs.trans (by simp only [ws0, hj'])

end AlgorithmLib.ML.BodyMap
