module
public import AlgorithmLib.ML.Ptx.BodyPhase
meta import AlgorithmLib.ML.Ptx.BodyPhase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyEnds` — a warp statement's launch runs

A launch of a statement's kernel has a run (`LaunchRuns`, `launch_exists`).
Each block's threads, answered with the warp machine's answers for their lanes
(`tapeOf`), each end as their lanes' runs (`lane_ends`, `thread_ends`), at one
fuel (`pass_worlds`); `pass_phases` gives each thread's collectives as its
lane's in the warp's run, so the laws on butterfly shuffles give those answers
back (`answersOf_shfl`, `block_exists`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase

namespace AlgorithmLib.ML.BodyEnds

open BodyLane LaneWarp LaneSettle

/-! ## The laws on butterfly shuffles -/

/-- Thread `t`'s partner in a butterfly shuffle with mask `m`. -/
def partner (t m : Nat) : Nat := 32 * (t / 32) + ((t % 32) ^^^ m) % 32

theorem answersOf_shfl {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} (lawAt : Nat → Answer)
    (h : ∀ k x m, (calls t)[k]? = some (.shflBfly, [.sc .i32 x, .sc .i32 m]) →
      ∃ args, (calls (partner t m.toNat))[k]? = some (.shflBfly, args) ∧ lawAt k = .val (lane32 args))
    (hsh : ∀ e ∈ calls t, ∃ x m, e = (.shflBfly, [.sc .i32 x, .sc .i32 m])) :
    answersOf calls snaps smem0 t = .ok ((List.range (calls t).length).map lawAt) := by
  unfold answersOf
  suffices hs : ∀ (l : List (Intr × List V)) (st : Nat) (out : Array Answer),
      (∀ k e, l[k]? = some e → (calls t)[st + k]? = some e) →
      (l.zipIdx st).foldlM (answerStep calls snaps smem0 (t / 32) (t % 32)) (0, out) =
        .ok (0, out ++ ((List.range l.length).map fun k => lawAt (st + k)).toArray) by
    have := hs (calls t) 0 #[] (fun k e he => by simpa using he)
    simp only [bind, Except.bind, this, pure, Except.pure]
    simp
  intro l
  induction l with
  | nil => intro st out _; simp [pure, Except.pure]
  | cons e es ih =>
    intro st out hl
    have he := hl 0 e rfl
    rw [Nat.add_zero] at he
    obtain ⟨x, m, rfl⟩ := hsh e (List.mem_of_getElem? he)
    obtain ⟨args, hp, hlaw⟩ := h st x m he
    have hstep : answerStep calls snaps smem0 (t / 32) (t % 32) (0, out)
        ((.shflBfly, [.sc .i32 x, .sc .i32 m]), st) = .ok (0, out.push (lawAt st)) := by
      simp only [answerStep, laneArg, bind, Except.bind, pure, Except.pure]
      have hp' : (calls (32 * (t / 32) + (t % 32 ^^^ m.toNat) % 32))[st]? = some (.shflBfly, args) := hp
      rw [hp', hlaw]
      rfl
    simp only [List.zipIdx_cons, List.foldlM_cons, bind, Except.bind]
    rw [hstep]
    dsimp only
    rw [ih (st + 1) (out.push (lawAt st)) (fun k e' he' => by
      have := hl (k + 1) e' he'; rwa [show st + (k + 1) = st + 1 + k by omega] at this)]
    congr 2
    simp only [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map, Nat.add_zero]
    apply Array.ext'
    simp [Function.comp_def, Nat.add_assoc, Nat.add_comm 1]


/-! ## A block's threads end -/

/-- **A lane's run of a phased statement ends**, whatever its answers, where
    its indexes fit and every word each phase may touch lies inside its
    buffer, none a word `rd` reads. -/
theorem lane_ends (nbuf cta : Nat) (A : Nat → Acc) (rd : Buf → Nat → Option Nat) (l : Lane)
    {s : EWStmt} {ps : List EWStmt} (hsp : Split s ps)
    (hA : ∀ q b j, A q l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hRd : ∀ q b j, A q l b j → rd b j = none)
    (hacc : ∀ q, q < ps.length → AccOk cta (A q) rd 0 (ps.getD q .skip))
    (hfits : fits rd cta 0 s = true) (x : LCtx) (hx : CtxOk nbuf cta l x) (hi : x.i = 0) (li : LI)
    (hs : x.start ≤ li.w.obs.length) (hr : RdOk nbuf li.w.mem rd)
    (hsz : ∀ q b j, q < ps.length → A q l b j → 4 * j + 4 ≤ (li.w.mem.bufs b).size) :
    ∃ li', laneI x s li = some li' := by
  rw [laneI_split x hsp]
  have key : ∀ (L : List EWStmt) (k : Nat) (li : LI), (∀ i, i < L.length → L[i]? = ps[k + i]?) →
      k + L.length ≤ ps.length → x.start ≤ li.w.obs.length → RdOk nbuf li.w.mem rd →
      (∀ q b j, q < ps.length → A q l b j → 4 * j + 4 ≤ (li.w.mem.bufs b).size) →
      ∃ li', L.foldlM (m := Option) (fun li p => laneI x p li) li = some li' := by
    intro L
    induction L with
    | nil => intro k li _ _ _ _ _; exact ⟨li, rfl⟩
    | cons p ps' ih =>
      intro k li hL hk hs hr hsz
      have hp : ps[k]? = some p := by have := hL 0 (by simp); simpa using this.symm
      have hk' : k < ps.length := by simp at hk; omega
      have hpk : ps.getD k .skip = p := by simp [List.getD_eq_getElem?_getD, hp]
      obtain ⟨l1, e1, sh⟩ := laneI_shape nbuf cta (A k) rd l (hA k) (hRd k) p x li hx
        (by rw [hi]; exact fits_split rd cta 0 hsp hfits p (List.mem_of_getElem? hp)) (by rw [hi, ← hpk]; exact hacc k hk')
        hs hr (fun b j h => hsz k b j hk' h)
      obtain ⟨l2, e2⟩ := ih (k + 1) l1
        (fun i hi => by rw [show k + 1 + i = k + (i + 1) by omega, ← hL (i + 1) (by simp; omega)]; rfl)
        (by simp at hk; omega)
        (Nat.le_trans hs (by obtain ⟨L', hL'⟩ := laneI_grows p _ li l1 e1; rw [hL']; simp)) sh.rdok
        (fun q b j hq h => by
          have := sh.same.size (.buf ⟨b, (hA q b j h).2.1⟩)
          simp only [Mem.region] at this
          rw [← this]; exact hsz q b j hq h)
      exact ⟨l2, by simp only [List.foldlM_cons, bind, Option.bind, e1, e2]⟩
  exact key ps 0 li (fun i _ => by simp) (by simp) hs hr hsz

/-- **A pass of threads that end**: where each thread, from the world before
    it with shared memory reset, ends at some fuel in the next, the pass
    finishes alike at every fuel past some, its threads' collectives and
    shared stores what their runs made. -/
theorem pass_worlds {k : Kernel} {hw : Hw} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer}
    (Ws : Nat → World) : ∀ n, (∀ t, t < n → ∃ steps,
      thread k.code (ans t) (args t) { Ws t with mem := { (Ws t).mem with arena := smem0 } } hw steps = .ok (Ws (t + 1))) →
      ∃ S, ∀ S', S ≤ S' → ∃ c st, (List.range n).foldlM (passStep (k.run hw S') smem0 args ans)
        (Ws 0, fun _ => [], fun _ => []) = .ok (Ws n, c, st) ∧
        ∀ t, t < n → c t = collCalls (Ws (t + 1)) (Ws t).obs.length ∧
          st t = sharedStores (Ws (t + 1)) (Ws t).obs.length := by
  intro n
  induction n with
  | zero => intro _; exact ⟨0, fun _ _ => ⟨_, _, rfl, fun t ht => absurd ht (by omega)⟩⟩
  | succ n ih =>
    intro h
    obtain ⟨Sn, hSn⟩ := ih (fun t ht => h t (by omega))
    obtain ⟨kn, hkn⟩ := h n (by omega)
    refine ⟨max Sn kn, fun S' hS' => ?_⟩
    obtain ⟨c, st, hf, hc⟩ := hSn S' (by omega)
    refine ⟨fun t' => if t' = n then collCalls (Ws (n + 1)) (Ws n).obs.length else c t',
      fun t' => if t' = n then sharedStores (Ws (n + 1)) (Ws n).obs.length else st t', ?_, fun t ht => ?_⟩
    · rw [List.range_succ, List.foldlM_append, hf]
      simp only [bind, Except.bind, List.foldlM_cons, List.foldlM_nil, passStep, Kernel.run]
      rw [thread_mono hkn (by omega)]
      rfl
    · by_cases e : t = n
      · subst e; simp
      · simp only [e, if_false]; exact hc t (by omega)

/-- The warp machine's answers to thread `t`'s collectives: its lane's, phase
    after phase. -/
def tapeOf (hw : Hw) (nbuf cta nw smem : Nat) (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (m0 : Mem)
    (t : Nat) : List Answer :=
  (jOf hw nbuf cta nw rd ps m0 (t / 32)).map (ansOne (cta * nw + t / 32) smem ⟨t % 32, Nat.mod_lt _ (by decide)⟩)

/-- A statement's fragment's collective leaves are butterfly shuffles. -/
theorem jOf_bfly (hw : Hw) (nbuf cta nw : Nat) (rd : Buf → Nat → Option Nat) {s : EWStmt} {ps : List EWStmt}
    (hsp : Split s ps) (m0 : Mem) (wi : Nat) (hf : fits rd (cta * nw + wi) 0 s = true) :
    ∀ e ∈ jOf hw nbuf cta nw rd ps m0 wi, ∃ i d sr m ws, e = (i, .shflXor d sr m, ws) ∧ m < 2 ^ 31 := by
  intro e he
  simp only [jOf, List.mem_flatMap, List.mem_range] at he
  obtain ⟨q, hq, he⟩ := he
  have hfq : fits rd (cta * nw + wi) 0 (ps.getD q .skip) = true := by
    rw [show ps.getD q .skip = ps[q] by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]]
    exact fits_split rd _ 0 hsp hf _ (List.getElem_mem hq)
  obtain ⟨i', s', ws', hQ, hm⟩ := mem_trace hw _ (imOf rd) jLeaf BflyOnly e _ 0 _ (allAt_of_fits rd _ _ 0 hfq) he
  have he' := jLeaf_mem hm
  subst he'
  cases s' with
  | shflXor d sr m => exact ⟨_, d, sr, m, _, rfl, hQ⟩
  | ballot | shflIdx | barrier => exact hQ.elim
  | _ => simp [jLeaf] at hm

/-- **A warp statement's block runs, in phases**: a block of `nw` warps of the
    statement's kernel, as `block_phases` asks of it, has a run (`BlockRuns`):
    its threads, answered with the warp machine's answers for their lanes,
    which the laws give back, each end; it keeps memory's shape, what `rd`
    reads and the NaN rule. -/
theorem block_exists (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (cta : Nat) (hcta : cta * nw + nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ wi q, wi < nw → q < ps.length → AccOk (cta * nw + wi) (A q (cta * nw + wi)) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hone : ∀ t t' q q' b j, t < 32 * nw → t' < 32 * nw → q < ps.length → q' < ps.length →
      StoresL (cta * nw + t / 32) (imOf rd) ⟨t % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD q .skip) b j →
      StoresL (cta * nw + t' / 32) (imOf rd) ⟨t' % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD q' .skip) b j →
      t = t' ∧ q = q')
    (hlate : ∀ t o q p b j, t < 32 * nw → o < 32 * nw → q < ps.length → p < ps.length →
      A q (cta * nw + t / 32) ⟨t % 32, Nat.mod_lt _ (by decide)⟩ b j →
      StoresL (cta * nw + o / 32) (imOf rd) ⟨o % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD p .skip) b j →
      t ≠ o → p < q ∧ o < t ∧ o / 32 = t / 32)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (w : World) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hfits : ∀ wi, wi < nw → fits rd (cta * nw + wi) 0 s = true)
    (hRd : ∀ q wi l b j, wi < nw → A q (cta * nw + wi) l b j → rd b j = none)
    (hsz : ∀ q wi l b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j → 4 * j + 4 ≤ (w.mem.bufs b).size) :
    ∃ u, BlockRuns (Body.kernel name nbuf smem s nw) cta (32 * nw) (bufArgs nbuf) hw w u ∧
      MemSame w.mem u.mem ∧ RdOk nbuf u.mem rd ∧ u.mem.nan = hw.nan := by
  let N := 32 * nw
  let A0 := ByteArray.mk (Array.replicate smem 0)
  let T (t : Nat) : List Answer := tapeOf hw nbuf cta nw smem rd ps w.mem t
  let Vw (W : World) : World := { W with mem := { W.mem with arena := A0 } }
  let step (t : Nat) (W : World) : World :=
    ((laneI (laneCtx nbuf cta nw t (T t) W.obs.length hw) s (LI.init (Vw W))).map LI.w).getD W
  let Ws (t : Nat) : World := Nat.rec (motive := fun _ => World) (Vw w) (fun t W => step t W) t
  have hWsS : ∀ t, Ws (t + 1) = step t (Ws t) := fun _ => rfl
  have hvc : ∀ t, t < N → cta * nw + t / 32 < 2 ^ 32 := fun t ht => by
    have : t / 32 < nw := by omega
    omega
  have hWd : ∀ t, t < N → Words (T t) := by
    intro t ht a ha v hv
    simp only [T, tapeOf, List.mem_map] at ha
    obtain ⟨e, he, rfl⟩ := ha
    obtain ⟨i, d, sr, m, ws, rfl, _⟩ := jOf_bfly hw nbuf cta nw rd hsp w.mem (t / 32) (hfits _ (by omega)) e he
    simp only [ansOne, ansLeaf, List.headD_cons] at hv
    cases hv
    exact ⟨_, rfl⟩
  have hWs : ∀ t, t < N → MemSame (rm A0 w.mem) (rm A0 (Ws t).mem) → RdOk nbuf (Ws t).mem rd →
      (Ws t).mem.nan = hw.nan → ∃ steps, thread (Body.kernel name nbuf smem s nw).code (T t)
        (threadArgs cta (32 * nw) (bufArgs nbuf) t) (Vw (Ws t)) hw steps = .ok (Ws (t + 1)) := by
    intro t ht hs hr hn
    have hwi : t / 32 < nw := by omega
    obtain ⟨li', hl⟩ := lane_ends nbuf (cta * nw + t / 32) (fun q => A q (cta * nw + t / 32)) rd
      ⟨t % 32, Nat.mod_lt _ (by decide)⟩ hsp (fun q b j h => hA q _ _ b j h) (fun q b j h => hRd q _ _ b j hwi h)
      (fun q hq => hacc _ q hwi hq) (hfits _ hwi) (laneCtx nbuf cta nw t (T t) (Ws t).obs.length hw)
      (laneCtx_ok nbuf cta nw t _ _ hw (hvc t ht)) rfl
      (LI.init (Vw (Ws t))) (Nat.le_refl _) ((rdOk_arena A0).mpr hr)
      (fun q b j hq h => by
        have := hs.size (.buf ⟨b, (hA q _ _ b j h).2.1⟩)
        simp only [Mem.region] at this
        show 4 * j + 4 ≤ ((rm A0 (Ws t).mem).bufs b).size
        rw [← this]; exact hsz q _ _ b j hq hwi h)
    obtain ⟨steps, u, hu, he⟩ := thread_ends hw hw32 name nbuf smem s nw hnw hnw31 cta t (32 * nw) ht (by omega)
      (hvc t ht) (Vw (Ws t)) (T t) (hWd t ht) li' hfine hl
    refine ⟨steps, ?_⟩
    rw [hu, he, hWsS]
    simp only [step, hl, Option.map_some, Option.getD_some]
  have hlaw : ∀ q wi j, q < ps.length → wi < nw →
      j < (trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ps.getD q .skip)
        (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf w.mem) q)).length →
      ∀ l : Lane, (T (32 * wi + l.val))[cntOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf w.mem) q + j]? =
        (trace hw (cta * nw + wi) (imOf rd) (ansLeaf (cta * nw + wi) smem l) 0 (ps.getD q .skip)
          (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf w.mem) q))[j]? := by
    intro q wi j hq hwi hj l
    obtain ⟨e1, e2⟩ := lane_div wi l
    have hT : T (32 * wi + l.val) = (jOf hw nbuf cta nw rd ps w.mem wi).map (ansOne (cta * nw + wi) smem l) := by
      simp only [T, tapeOf, e1]
      congr 2; exact Fin.ext e2
    obtain ⟨R, hJ⟩ : ∃ R, jOf hw nbuf cta nw rd ps w.mem wi =
        ((List.range q).flatMap fun q => trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ps.getD q .skip)
          (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf w.mem) q)) ++
        (trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ps.getD q .skip)
          (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf w.mem) q) ++ R) := by
      simp only [jOf]
      rw [show ps.length = (q + 1) + (ps.length - (q + 1)) by omega, List.range_add, List.range_succ,
        List.flatMap_append, List.flatMap_append]
      rw [List.append_assoc, List.flatMap_cons, List.flatMap_nil, List.append_nil]
      exact ⟨_, rfl⟩
    rw [hT, hJ, cntOf_eq, trace_ans, List.getElem?_map, List.getElem?_map,
      List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left, List.getElem?_append_left hj]
  obtain ⟨hInv, hcl, -, -⟩ := pass_phases hw hw32 name nbuf smem s ps hsp nw hnw hnw31 cta hcta hN A rd hacc hA hone
    hlate hfine w hrd hnan hfits hRd hsz w.mem
    (fun q wi l b j hq hwi h => ⟨rfl, (bufW_ok (hA _ _ l b j h) (hsz q wi l b j hq hwi h) 0).1⟩) T hWd Ws rfl hWs
    (fun t => collCalls (Ws (t + 1)) (Ws t).obs.length) (fun _ _ => rfl)
    (fun t ht => Or.inr (by simp [T, tapeOf]))
    (fun q wi j hq hwi hj _ l => hlaw q wi j hq hwi hj l)
  obtain ⟨S, hS⟩ := pass_worlds (k := Body.kernel name nbuf smem s nw) (hw := hw) (smem0 := A0)
    (args := threadArgs cta (32 * nw) (bufArgs nbuf)) (ans := T) Ws N
    (fun t ht => hWs t ht (hInv t (by omega)).1 (hInv t (by omega)).2.1 (hInv t (by omega)).2.2)
  obtain ⟨c, st, hfold, hc⟩ := hS S (Nat.le_refl _)
  -- the laws give the warp's answers back
  have hans_t : ∀ t, t < N → answersOf c (snapshots N A0 c st) A0 t = .ok (T t) := by
    intro t ht
    have hwi : t / 32 < nw := by omega
    let l : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
    have htl : 32 * (t / 32) + l.val = t := by show 32 * (t / 32) + t % 32 = t; omega
    have hct : ∀ t', t' < N → c t' = (jOf hw nbuf cta nw rd ps w.mem (t' / 32)).map
        (callOne (cta * nw + t' / 32) ⟨t' % 32, Nat.mod_lt _ (by decide)⟩) := by
      intro t' ht'
      have h := hcl (t' / 32) (by omega) ⟨t' % 32, Nat.mod_lt _ (by decide)⟩
      rw [show 32 * (t' / 32) + t' % 32 = t' by omega] at h
      rw [(hc t' ht').1]; exact h
    have hlen : (c t).length = (T t).length := by rw [hct t ht]; simp [T, tapeOf]
    rw [answersOf_shfl (fun k => (T t).getD k (.smem .empty)) ?_ ?_]
    · rw [hlen]
      congr 1
      apply List.ext_getElem (by simp)
      intro k h1 h2
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]
    · intro k x m hk
      rw [hct t ht, List.getElem?_map] at hk
      obtain ⟨e, he, hke⟩ := Option.map_eq_some_iff.mp hk
      obtain ⟨i, d, sr, m', ws, rfl, hm'⟩ := jOf_bfly hw nbuf cta nw rd hsp w.mem (t / 32) (hfits _ hwi) e
        (List.mem_of_getElem? he)
      simp only [callOne, callLeaf, List.headD_cons, Prod.mk.injEq, List.cons.injEq, V.sc.injEq, true_and,
        and_true] at hke
      obtain ⟨-, hmm⟩ := hke
      have hmn : m.toNat = m' := by rw [← hmm]; simp only [s32, UInt32.toNat_toUInt64]; exact toNat_ofNat32 (by omega)
      let P : Lane := xorLane l m'
      have hpt : partner t m.toNat = 32 * (t / 32) + P.val := by simp only [partner, hmn]; rfl
      have hPN : 32 * (t / 32) + P.val < N := by have : P.val < 32 := P.isLt; omega
      refine ⟨[.sc .i32 (ws.regs sr P).bits, .sc .i32 (s32 (UInt32.ofNat m'))], ?_, ?_⟩
      · rw [hpt, hct _ hPN, List.getElem?_map]
        obtain ⟨e1, e2⟩ := lane_div (t / 32) P
        have hP : (⟨(32 * (t / 32) + P.val) % 32, Nat.mod_lt _ (by decide)⟩ : Lane) = P := Fin.ext e2
        rw [e1, he]
        simp only [Option.map_some, callOne, callLeaf, List.headD_cons, hP]
      · simp only [T, tapeOf, List.getD_eq_getElem?_getD, List.getElem?_map, he, Option.map_some, Option.getD_some,
          ansOne, ansLeaf, List.headD_cons, lane32, s32]
        rfl
    · intro e he
      rw [hct t ht, List.mem_map] at he
      obtain ⟨e', he', rfl⟩ := he
      obtain ⟨i, d, sr, m', ws, rfl, _⟩ := jOf_bfly hw nbuf cta nw rd hsp w.mem (t / 32) (hfits _ hwi) e' he'
      exact ⟨_, _, rfl⟩
  have hans : Kernel.answers N A0 c st = .ok fun t => ((List.range N).map T).getD t [] := by
    unfold Kernel.answers
    rw [mapM_range_ok _ T N hans_t]
    rfl
  have hIN := hInv N (Nat.le_refl _)
  refine ⟨{ Ws N with mem := { (Ws N).mem with arena := w.mem.arena } },
    ⟨S, T, Ws N, c, st, _, hfold, hans, fun t ht => ?_, rfl⟩, memSame_rm hIN.1 w.mem.arena,
    (rdOk_arena w.mem.arena).mpr hIN.2.1, hIN.2.2⟩
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getElem?_range (show t < N from ht)]
  rfl

/-- **A warp statement's launch runs, in phases** (`LaunchRuns`): `grid`
    blocks of `nw` warps of the statement's kernel, split into phases, as
    `BodyPhase.launch_phases` asks of them — each phase's accesses ones its
    lanes may make there, a stored word one lane's in one phase, a lane
    touching what another stores only later and from a higher lane, no block
    storing what another touches, every touched word inside its buffer. -/
theorem launch_exists (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ c q, c < grid * nw → q < ps.length → AccOk c (A q c) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hown : ∀ c (l l' : Lane) q q' b j, c < grid * nw → q < ps.length → q' < ps.length →
      StoresL c (imOf rd) l 0 (ps.getD q .skip) b j → StoresL c (imOf rd) l' 0 (ps.getD q' .skip) b j →
      l = l' ∧ q = q')
    (hrdr : ∀ c (t o : Lane) q p b j, c < grid * nw → q < ps.length → p < ps.length → A q c t b j →
      StoresL c (imOf rd) o 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o.val < t.val)
    (hdisj : ∀ c c' l l' q b j, c < grid * nw → c' < grid * nw → c ≠ c' → q < ps.length → A q c l b j →
      ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ q c l b j, c < grid * nw → A q c l b j → rd b j = none)
    (hfits : ∀ c, c < grid * nw → fits rd c 0 s = true)
    (hin : ∀ q c l b j, c < grid * nw → q < ps.length → A q c l b j → 4 * j + 4 ≤ (bufs b).size) :
    ∃ bufs', LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw := by

  have hblk : ∀ bi, bi < grid → bi * nw + nw ≤ grid * nw := fun bi hb => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hb
  have hsto : ∀ c l b j q, q < ps.length → StoresL c (imOf rd) l 0 (ps.getD q .skip) b j →
      StoresL c (imOf rd) l 0 s b j := fun c l b j q hq h =>
    (storesL_split c (imOf rd) l 0 hsp b j).mpr ⟨ps[q], List.getElem_mem hq, by
      have hpq : ps.getD q .skip = ps[q] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
      rw [← hpq]; exact h⟩
  have inv : ∀ c, c ≤ grid → ∃ Wc, BlocksRun (Body.kernel name nbuf smem s nw) (32 * nw) (bufArgs nbuf) hw c
      ({ mem := { arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } } : World) Wc ∧
      MemSame (kmem bufs) Wc.mem ∧ RdOk nbuf Wc.mem rd ∧ Wc.mem.nan = hw.nan := by
    intro c
    induction c with
    | zero =>
      intro _
      have hld : ∀ a n, ({ arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } : Mem).load a n =
          (kmem bufs).load a n := Mem.load_congr (fun r => by cases r <;> rfl) rfl rfl
      refine ⟨_, rfl, ⟨fun r => by cases r <;> rfl, rfl, rfl⟩, fun b j n h => ?_, rfl⟩
      obtain ⟨h1, h2⟩ := hrd b j n h
      exact ⟨h1, by simp only [rdM] at h2 ⊢; rw [hld]; exact h2⟩
    | succ c ih =>
      intro hc
      obtain ⟨Wp, hp, hs, hr, hn⟩ := ih (by omega)
      have hcg : c < grid := by omega
      have hb := hblk c hcg
      let ln (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
      have hvt : ∀ t, t < 32 * nw → c * nw + t / 32 < grid * nw := fun t ht => by
        have : t / 32 < nw := by omega
        omega
      obtain ⟨Wc, hWc, h3, h4, h5⟩ := block_exists hw hw32 name nbuf smem s ps hsp nw hnw hnw31 c (by omega) hN A rd
        (fun wi q hwi hq => hacc _ q (by omega) hq) hA
        (fun t t' q q' b j ht ht' hq hq' hs1 hs2 => by
          by_cases hw : t / 32 = t' / 32
          · rw [hw] at hs1
            obtain ⟨hl, hqq⟩ := hown _ _ _ q q' b j (hvt t' ht') hq hq' hs1 hs2
            have := congrArg Fin.val hl
            simp only at this
            exact ⟨by omega, hqq⟩
          · exfalso
            exact hdisj (c * nw + t / 32) (c * nw + t' / 32) (ln t) (ln t') q b j (hvt t ht) (hvt t' ht')
              (by omega) hq (stores_acc _ _ rd _ _ 0 b j (hacc _ q (hvt t ht) hq) hs1) (hsto _ _ b j q' hq' hs2))
        (fun t o' q p b j ht ho hq hp hAt hs' hto => by
          by_cases hw : o' / 32 = t / 32
          · rw [hw] at hs'
            have hne : ln t ≠ ln o' := fun e => hto (by
              have := congrArg Fin.val e; simp only [ln] at this; omega)
            obtain ⟨hpq, hlt⟩ := hrdr _ (ln t) (ln o') q p b j (hvt t ht) hq hp hAt hs' hne
            simp only [ln] at hlt
            exact ⟨hpq, by omega, hw⟩
          · exfalso
            exact hdisj (c * nw + t / 32) (c * nw + o' / 32) (ln t) (ln o') q b j (hvt t ht) (hvt o' ho)
              (by omega) hq hAt (hsto _ _ b j p hp hs'))
        hfine Wp hr hn (fun wi hwi => hfits _ (by omega)) (fun q wi l b j hwi h => hRd q _ l b j (by omega) h)
        (fun q wi l b j hq hwi h => by
          have := hs.size (.buf ⟨b, (hA q _ l b j h).2.1⟩)
          simp only [Mem.region] at this
          rw [← this]; exact hin q _ l b j (by omega) hq h)
      exact ⟨Wc, ⟨Wp, hp, hWc⟩, hs.trans h3, h4, h5⟩
  obtain ⟨W, hW, -, -, -⟩ := inv grid (Nat.le_refl _)
  exact ⟨W.mem.bufs, W, hW, rfl⟩

end AlgorithmLib.ML.BodyEnds
