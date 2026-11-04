module
public import AlgorithmLib.Kernel.Elementwise
meta import AlgorithmLib.Kernel.Elementwise
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
public import AlgorithmLib.Proof.EndsRules
meta import AlgorithmLib.Proof.EndsRules
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Warp` — collectives in a thread's proof

A thread's collectives are answered from the laws (`Kernel.Run`), so a
thread's proof takes its answers as given: `AnswersFrom` says what a
collective gives back — the value the list holds for it (`valAt`), or for a
barrier its shared memory — and a thread's locals answer so
(`answersFrom_thread`). `thread_of_triple_ans` reads a body's triple, proven
under that, as what the thread leaves; a thread that finishes at two fuels
leaves one world (`thread_det`). A collective's step is `wp_coll2`, a
barrier's `wp_bar`.

A block runs its threads in order with answers the laws give back from what
those runs passed (`block_settled`). `pass_spec` composes the pass's threads:
their effects on memory and the collectives each made. `answersOf_bfly` is the
warp law for a butterfly shuffle: lane `l xor m`'s word at the same
collective; the laws answer only words (`answersOf_words`). From these a
kernel's proof finds its answers: each a value its partner lane computed.

Shared memory is the arena at `smemAddr`; a barrier's answer replaces it
(`wp_bar`). The barrier law answers a thread whose collectives are all
barriers with the block's snapshots in order (`answersOf_bars`), each the
size of shared memory (`answersOf_smem_size`); stores into device buffers
ignore which shared memory is in place (`store_buf_arena`, `load_buf_arena`).
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Prog


/-- Locals that answer collectives as a thread answered from `ans` does: any
    collective but a barrier the value `valAt` gives, leaving the world as it
    is; a barrier, where it answers, the shared memory `ans` holds, put in
    place. -/
structure AnswersFrom (ans : List Answer) (start : Nat) (cfg : Cfg) : Prop where
  val : ∀ (c : Intr) (vs : List V) (w : World), c.isColl = true → c ≠ .barrier →
    cfg.locals c.slot vs w = .ok (some (valAt ans ((collCalls w start).length - 1)), w)
  smem : ∀ (vs : List V) (w w' : World) (r : Option V), cfg.locals Intr.barrier.slot vs w = .ok (r, w') →
    r = none ∧ ∃ b, w' = { w with mem := { w.mem with arena := b } } ∧
      ans[(collCalls w start).length - 1]? = some (.smem b)
  bar : ∀ (vs : List V) (w : World) (b : ByteArray), ans[(collCalls w start).length - 1]? = some (.smem b) →
    cfg.locals Intr.barrier.slot vs w = .ok (none, { w with mem := { w.mem with arena := b } })

/-- Locals that answer an approximating intrinsic as the hardware `hw` does. -/
def ApproxFrom (hw : Hw) (cfg : Cfg) : Prop :=
  ∀ (c : Intr) (ty : ClifTy) (x : UInt64) (w : World), c.isColl = false →
    cfg.locals c.slot [.sc ty x] w = .ok (some (.sc .f32 (hw.approx c x)), w)

/-- A thread's locals answer its collectives from its answers. -/
theorem answersFrom_thread (ans : List Answer) (start : Nat) (hw : Hw) :
    AnswersFrom ans start { env := [], locals := threadLocals ans start hw } := by
  constructor
  · intro c vs w1 hc hb
    simp [threadLocals, Intr.find_slot, hc, hb]
  · intro vs w1 w2 r hl
    simp only [threadLocals, Intr.find_slot, Intr.isColl, if_true] at hl
    split at hl
    · rename_i b hb
      simp only [Except.ok.injEq, Prod.mk.injEq] at hl
      obtain ⟨rfl, rfl⟩ := hl
      exact ⟨rfl, b, rfl, hb⟩
    · cases hl
  · intro vs w1 b hb
    simp [threadLocals, Intr.find_slot, Intr.isColl, hb]

/-- A thread's locals answer an approximating intrinsic as the hardware does. -/
theorem approxFrom_thread (ans : List Answer) (start : Nat) (hw : Hw) :
    ApproxFrom hw { env := [], locals := threadLocals ans start hw } := by
  intro c ty x w1 hc
  simp [threadLocals, Intr.find_slot, hc]

/-- A thread that finishes at two fuels finishes in the same world. -/
theorem thread_det {code : Code} {ans : List Answer} {args : List V} {w u1 u2 : World} {hw : Hw}
    {s1 s2 : Nat} (h1 : thread code ans args w hw s1 = .ok u1) (h2 : thread code ans args w hw s2 = .ok u2) :
    u1 = u2 := by
  unfold thread at h1 h2
  generalize ({ env := [], locals := threadLocals ans w.obs.length hw } : Cfg) = cfg at h1 h2
  cases e1 : runCode s1 cfg args.toArray w code <;> rw [e1] at h1 <;> try cases h1
  cases e2 : runCode s2 cfg args.toArray w code <;> rw [e2] at h2 <;> try cases h2
  rcases Nat.le_total s1 s2 with h | h
  · rw [runCode_mono h e1 rfl] at e2; cases e2; rfl
  · rw [runCode_mono h e2 rfl] at e1; cases e1; rfl

/-- A thread that finishes at some fuel finishes so at every larger one. -/
theorem thread_mono {code : Code} {ans : List Answer} {args : List V} {w u : World} {hw : Hw} {s1 s2 : Nat}
    (h : thread code ans args w hw s1 = .ok u) (hs : s1 ≤ s2) : thread code ans args w hw s2 = .ok u := by
  unfold thread at h ⊢
  generalize ({ env := [], locals := threadLocals ans w.obs.length hw } : Cfg) = cfg at h ⊢
  cases e1 : runCode s1 cfg args.toArray w code <;> rw [e1] at h <;> try cases h
  rw [runCode_mono hs e1 rfl]

/-- Laws that answer each thread give back answers for all of them. -/
theorem mapM_range_ok {β : Type} (g : Nat → Except String β) (F : Nat → β) :
    ∀ n, (∀ t, t < n → g t = .ok (F t)) → (List.range n).mapM g = .ok ((List.range n).map F)
  | 0, _ => rfl
  | n + 1, h => by
    rw [List.range_succ, List.mapM_append, mapM_range_ok g F n (fun t ht => h t (by omega))]
    simp [h n (by omega), List.map_append]
    rfl

/-- **A pass of threads, each ending, runs**: where each thread, from a pass
    state the invariant `Inv` allows before it, ends at some fuel in a state it
    allows after, the pass finishes, the same way at every fuel past some, in a
    state `Inv` allows after all `n`. -/
theorem pass_ends {k : Kernel} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer} {hw : Hw}
    (Inv : Nat → PassSt → Prop) :
    ∀ (n : Nat) (s0 : PassSt), Inv 0 s0 →
      (∀ t, t < n → ∀ s : PassSt, Inv t s → ∃ steps u,
        thread k.code (ans t) (args t) { s.1 with mem := { s.1.mem with arena := smem0 } } hw steps = .ok u ∧
        Inv (t + 1) (u, fun t' => if t' = t then collCalls u s.1.obs.length else s.2.1 t',
          fun t' => if t' = t then sharedStores u s.1.obs.length else s.2.2 t')) →
      ∃ S r, ∀ S', S ≤ S' → (List.range n).foldlM (passStep (k.run hw S') smem0 args ans) s0 = .ok r ∧ Inv n r
  | 0, s0, h0, _ => ⟨0, s0, fun _ _ => ⟨rfl, h0⟩⟩
  | n + 1, s0, h0, hrun => by
    obtain ⟨Sn, r, hn⟩ := pass_ends Inv n s0 h0 (fun t ht => hrun t (by omega))
    obtain ⟨-, hr⟩ := hn Sn (Nat.le_refl _)
    obtain ⟨kn, u, hu, hinv⟩ := hrun n (Nat.lt_succ_self n) r hr
    refine ⟨max Sn kn, _, fun S' hS' => ⟨?_, hinv⟩⟩
    obtain ⟨hf, -⟩ := hn S' (by omega)
    rw [List.range_succ, List.foldlM_append, hf]
    obtain ⟨w1, c1, st1⟩ := r
    simp only [bind, Except.bind, List.foldlM_cons, List.foldlM_nil, passStep, Kernel.run]
    rw [thread_mono hu (by omega)]
    rfl

/-- **A thread that finishes ends where its body's proof says**, the proof
    given only that the thread's collectives give back their answers and its
    approximations the hardware's. -/
theorem thread_of_triple_ans {code : Code} {args : List V} {w u : World} {Q : World → Prop}
    {ans : List Answer} {hw : Hw} {steps : Nat}
    (h : ∀ cfg : Cfg, AnswersFrom ans w.obs.length cfg → ApproxFrom hw cfg →
      Triple cfg (At args.toArray w) code { ok := fun _ w' => Q w', faultOk := false })
    (hr : thread code ans args w hw steps = .ok u) : Q u := by
  unfold thread at hr
  have ht := h _ (answersFrom_thread ans w.obs.length hw) (approxFrom_thread ans w.obs.length hw)
    steps args.toArray w ⟨rfl, rfl⟩
  cases hc : runCode steps { env := [], locals := threadLocals ans w.obs.length hw } args.toArray w code <;>
    rw [hc] at hr ht <;> simp_all [CodeRes.Sat]

theorem collCalls_call {w : World} {start : Nat} (hs : start ≤ w.obs.length) (c : Intr) (vs : List V) :
    collCalls (obsCall w (.intr c) vs) start = collCalls w start ++ (if c.isColl then [(c, vs)] else []) := by
  unfold collCalls obsCall
  simp only [callWords, List.append_nil, List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]
  cases c <;> simp [Intr.isColl, collOf]


theorem foldlM_zipIdx_push {α β σ : Type} (g : σ × Array β → α × Nat → Except String (σ × Array β))
    (P : Nat → α → β → Prop)
    (hg : ∀ s out x k r, g (s, out) (x, k) = .ok r → ∃ y, r.2 = out.push y ∧ P k x y) :
    ∀ (l : List α) (st : Nat) (s : σ) (out : Array β) (r : σ × Array β),
      (l.zipIdx st).foldlM g (s, out) = .ok r →
      r.2.size = out.size + l.length ∧ (∀ i, i < out.size → r.2[i]? = out[i]?) ∧
      ∀ i x, l[i]? = some x → ∃ y, r.2[out.size + i]? = some y ∧ P (st + i) x y
  | [], st, s, out, r, h => by
    simp only [List.zipIdx_nil, List.foldlM_nil] at h
    cases h; simp
  | x :: xs, st, s, out, r, h => by
    simp only [List.zipIdx_cons, List.foldlM_cons] at h
    cases hg1 : g (s, out) (x, st) with
    | error e => rw [hg1] at h; cases h
    | ok r1 =>
      rw [hg1] at h
      obtain ⟨y, hy, hP⟩ := hg s out x st r1 hg1
      obtain ⟨s1, o1⟩ := r1
      simp only at hy; subst hy
      obtain ⟨h1, h2, h3⟩ := foldlM_zipIdx_push g P hg xs (st + 1) s1 (out.push y) r h
      refine ⟨by simp [h1]; omega, fun i hi => ?_, fun i z hz => ?_⟩
      · rw [h2 i (by simp; omega)]; simp [Array.getElem?_push, show i ≠ out.size by omega]
      · cases i with
        | zero =>
          simp at hz; subst hz
          refine ⟨y, ?_, by simpa using hP⟩
          rw [Nat.add_zero, h2 out.size (by simp)]; simp
        | succ i =>
          obtain ⟨y', hy', hP'⟩ := h3 i z (by simpa using hz)
          refine ⟨y', ?_, by rw [show st + (i + 1) = st + 1 + i by omega]; exact hP'⟩
          rw [← hy']; simp; congr 1; omega

/-- **A butterfly shuffle's answer**: lane `l xor m`'s value at the same
    collective. -/
theorem answersOf_bfly {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} {as : List Answer} (h : answersOf calls snaps smem0 t = .ok as)
    {k : Nat} {a : V} {ty : ClifTy} {m : UInt64} (hk : (calls t)[k]? = some (.shflBfly, [a, .sc ty m])) :
    ∃ args, laneArg calls (t / 32) ((t % 32 ^^^ m.toNat) % 32) k .shflBfly = .ok args ∧
      as[k]? = some (.val (lane32 args)) := by
  unfold answersOf at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i r hr
  obtain ⟨_, out⟩ := r
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨_, _, h3⟩ := foldlM_zipIdx_push _
    (fun k (x : Intr × List V) y => ∀ a ty m, x = (.shflBfly, [a, .sc ty m]) →
      ∃ args, laneArg calls (t / 32) ((t % 32 ^^^ m.toNat) % 32) k .shflBfly = .ok args ∧
        y = .val (lane32 args))
    (fun s out x k r hr => by
      obtain ⟨c, args⟩ := x
      unfold answerStep at hr
      dsimp only at hr
      split at hr
      · simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun a ty m he => by cases he⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        rename_i a' ha'
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        refine ⟨_, rfl, fun a ty m he => ?_⟩
        simp only [Prod.mk.injEq, List.cons.injEq, V.sc.injEq, and_true] at he
        obtain ⟨-, -, -, rfl⟩ := he
        exact ⟨a', ha', rfl⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun a ty m he => by cases he⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun a ty m he => by cases he⟩
      · cases hr)
    (calls t) 0 0 #[] _ hr
  obtain ⟨y, hy, hP⟩ := h3 k _ hk
  obtain ⟨args, h1, rfl⟩ := hP a ty m rfl
  refine ⟨args, by simpa using h1, ?_⟩
  simpa using hy

/-- One collective's step pushes its answer, whatever was answered before,
    and counts it if it is a barrier. -/
theorem answerStep_push {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {warp lane nb : Nat} {out : Array Answer} {x : Intr × List V} {k : Nat} {r : Nat × Array Answer}
    (h : answerStep calls snaps smem0 warp lane (nb, out) (x, k) = .ok r) :
    ∃ y, r = (nb + (if x.1 == .barrier then 1 else 0), out.push y) ∧
      answerStep calls snaps smem0 warp lane (nb, #[]) (x, k) =
        .ok (nb + (if x.1 == .barrier then 1 else 0), #[y]) := by
  obtain ⟨c, args⟩ := x
  unfold answerStep at h ⊢
  dsimp only at h ⊢
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    exact ⟨_, by simp [show (Intr.barrier == Intr.barrier) = true by decide]; rfl,
      by simp [show (Intr.barrier == Intr.barrier) = true by decide]; rfl⟩
  · simp only [bind, Except.bind] at h ⊢
    split at h
    · cases h
    rename_i a ha
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    exact ⟨_, by simp [show (Intr.shflBfly == Intr.barrier) = false by decide]; rfl,
      by simp [ha, show (Intr.shflBfly == Intr.barrier) = false by decide]; rfl⟩
  · simp only [bind, Except.bind] at h ⊢
    split at h
    · cases h
    rename_i a ha
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    exact ⟨_, by simp [show (Intr.shflIdx == Intr.barrier) = false by decide]; rfl,
      by simp [ha, show (Intr.shflIdx == Intr.barrier) = false by decide]; rfl⟩
  · simp only [bind, Except.bind] at h ⊢
    split at h
    · cases h
    rename_i a ha
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    exact ⟨_, by simp [show (Intr.ballot == Intr.barrier) = false by decide]; rfl,
      by simp [ha, show (Intr.ballot == Intr.barrier) = false by decide]; rfl⟩
  · cases h

theorem foldAns {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {warp lane : Nat} :
    ∀ (l : List (Intr × List V)) (st nb : Nat) (out : Array Answer) (r : Nat × Array Answer),
      (l.zipIdx st).foldlM (answerStep calls snaps smem0 warp lane) (nb, out) = .ok r →
      ∀ i x, l[i]? = some x → ∃ y nb', r.2[out.size + i]? = some y ∧
        answerStep calls snaps smem0 warp lane (nb + (l.take i).countP (·.1 == .barrier), #[]) (x, st + i) =
          .ok (nb', #[y])
  | [], st, nb, out, r, h, i, x, hx => by simp at hx
  | x0 :: xs, st, nb, out, r, h, i, x, hx => by
    simp only [List.zipIdx_cons, List.foldlM_cons] at h
    cases hg : answerStep calls snaps smem0 warp lane (nb, out) (x0, st) with
    | error e => rw [hg] at h; cases h
    | ok r1 =>
      rw [hg] at h
      obtain ⟨y, rfl, hy⟩ := answerStep_push hg
      have hkeep : ∀ (l : List (Intr × List V)) st nb (out : Array Answer) r,
          (l.zipIdx st).foldlM (answerStep calls snaps smem0 warp lane) (nb, out) = .ok r →
          ∀ j, j < out.size → r.2[j]? = out[j]? := by
        intro l
        induction l with
        | nil => intro st nb out r h j _; simp at h; cases h; rfl
        | cons z zs ih =>
          intro st nb out r h j hj
          simp only [List.zipIdx_cons, List.foldlM_cons] at h
          cases hg' : answerStep calls snaps smem0 warp lane (nb, out) (z, st) with
          | error e => rw [hg'] at h; cases h
          | ok r2 =>
            rw [hg'] at h
            obtain ⟨y', rfl, -⟩ := answerStep_push hg'
            rw [ih _ _ _ _ h j (by simp; omega)]
            simp [Array.getElem?_push, show j ≠ out.size by omega]
      cases i with
      | zero =>
        simp at hx; subst hx
        refine ⟨y, _, ?_, by simpa using hy⟩
        rw [Nat.add_zero, hkeep xs _ _ _ r h out.size (by simp)]; simp
      | succ i =>
        obtain ⟨y', nb', hy', hs'⟩ := foldAns xs (st + 1) _ (out.push y) r h i x (by simpa using hx)
        refine ⟨y', nb', by rw [← hy']; simp; congr 1; omega, ?_⟩
        rw [show st + (i + 1) = st + 1 + i by omega]
        have e : nb + ((x0 :: xs).take (i + 1)).countP (·.1 == .barrier) =
            nb + (if x0.1 == .barrier then 1 else 0) + (xs.take i).countP (·.1 == .barrier) := by
          simp only [List.take_succ_cons, List.countP_cons]; split <;> simp_all <;> omega
        rw [e]; exact hs'

/-- **What the laws answer a thread's `k`-th collective**: its step from the
    barriers before it. -/
theorem answersOf_get {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} {as : List Answer} (h : answersOf calls snaps smem0 t = .ok as) {k : Nat} {x : Intr × List V}
    (hk : (calls t)[k]? = some x) :
    ∃ y nb', as[k]? = some y ∧ answerStep calls snaps smem0 (t / 32) (t % 32)
      (((calls t).take k).countP (·.1 == .barrier), #[]) (x, k) = .ok (nb', #[y]) := by
  unfold answersOf at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i r hr
  obtain ⟨_, out⟩ := r
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨y, nb', hy, hs⟩ := foldAns (calls t) 0 0 #[] _ hr k x hk
  exact ⟨y, nb', by simpa using hy, by simpa using hs⟩

/-- A collective's step answers the same whatever was answered before it. -/
theorem answerStep_out {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {warp lane nb : Nat} (out : Array Answer) (x : (Intr × List V) × Nat) :
    answerStep calls snaps smem0 warp lane (nb, out) x =
      (answerStep calls snaps smem0 warp lane (nb, #[]) x).map fun r => (r.1, out ++ r.2) := by
  obtain ⟨⟨c, args⟩, k⟩ := x
  unfold answerStep
  dsimp only
  split
  · simp only [pure, Except.pure, Except.map]; rw [Array.push_eq_append]; rfl
  · simp only [bind, Except.bind]; split
    · rfl
    · simp only [pure, Except.pure, Except.map]; rw [Array.push_eq_append]; rfl
  · simp only [bind, Except.bind]; split
    · rfl
    · simp only [pure, Except.pure, Except.map]; rw [Array.push_eq_append]; rfl
  · simp only [bind, Except.bind]; split
    · rfl
    · simp only [pure, Except.pure, Except.map]; rw [Array.push_eq_append]; rfl
  · rfl

theorem foldAns_of {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {warp lane : Nat} :
    ∀ (l : List (Intr × List V)) (as : List Answer) (st nb : Nat) (out : Array Answer), as.length = l.length →
      (∀ i x a, l[i]? = some x → as[i]? = some a → ∃ nb',
        answerStep calls snaps smem0 warp lane (nb + (l.take i).countP (·.1 == .barrier), #[]) (x, st + i) =
          .ok (nb', #[a])) →
      ∃ nbF, (l.zipIdx st).foldlM (answerStep calls snaps smem0 warp lane) (nb, out) = .ok (nbF, out ++ as.toArray)
  | [], [], _, nb, out, _, _ => ⟨nb, by simp; rfl⟩
  | [], _ :: _, _, _, _, h, _ => by simp at h
  | _ :: _, [], _, _, _, h, _ => by simp at h
  | x0 :: xs, a0 :: as, st, nb, out, hl, h => by
    obtain ⟨nb1, h0⟩ := h 0 x0 a0 rfl rfl
    simp only [List.take_zero, List.countP_nil, Nat.add_zero] at h0
    obtain ⟨y, hy, h0'⟩ := answerStep_push (out := #[]) h0
    simp only [Prod.mk.injEq] at hy
    obtain ⟨rfl, hy2⟩ := hy
    have hya : y = a0 := by
      have := congrArg (·[0]?) hy2; simpa using this.symm
    subst hya
    have hs : answerStep calls snaps smem0 warp lane (nb, out) (x0, st) =
        .ok (nb + (if x0.1 == .barrier then 1 else 0), out.push y) := by
      rw [answerStep_out, h0]; simp only [Except.map]; rw [Array.push_eq_append]
    obtain ⟨nbF, hF⟩ := foldAns_of xs as (st + 1) (nb + (if x0.1 == .barrier then 1 else 0)) (out.push y)
      (by simpa using hl) (fun i x a hx ha => by
        obtain ⟨nb', h'⟩ := h (i + 1) x a (by simpa using hx) (by simpa using ha)
        refine ⟨nb', ?_⟩
        have e : nb + ((x0 :: xs).take (i + 1)).countP (·.1 == .barrier) =
            nb + (if x0.1 == .barrier then 1 else 0) + (xs.take i).countP (·.1 == .barrier) := by
          simp only [List.take_succ_cons, List.countP_cons]
          by_cases hb : (x0.1 == Intr.barrier) = true <;> simp only [hb, if_true, Bool.false_eq_true, if_false] <;> omega
        rw [← e, show st + 1 + i = st + (i + 1) by omega]; exact h')
    refine ⟨nbF, ?_⟩
    simp only [List.zipIdx_cons, List.foldlM_cons, hs, bind, Except.bind]
    rw [hF, List.toArray_cons, ← Array.append_assoc, Array.push_eq_append]

/-- The laws answer thread `t`'s `k`-th collective `a`: its step, from the
    barriers before it, answers `a`. -/
def LawAt (calls : Nat → List (Intr × List V)) (snaps : Array ByteArray) (smem0 : ByteArray) (t k : Nat)
    (a : Answer) : Prop :=
  ∃ x nb', (calls t)[k]? = some x ∧ answerStep calls snaps smem0 (t / 32) (t % 32)
    (((calls t).take k).countP (·.1 == .barrier), #[]) (x, k) = .ok (nb', #[a])

/-- **The laws answer a thread with a list** where they answer each of its
    collectives with that list's entry there. -/
theorem answersOf_of_get {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} {as : List Answer} (hlen : as.length = (calls t).length)
    (h : ∀ k a, as[k]? = some a → LawAt calls snaps smem0 t k a) :
    answersOf calls snaps smem0 t = .ok as := by
  obtain ⟨nbF, hF⟩ := foldAns_of (calls := calls) (snaps := snaps) (smem0 := smem0) (warp := t / 32)
    (lane := t % 32) (calls t) as 0 0 #[] hlen (fun i x a hx ha => by
      obtain ⟨x', nb', hx', hs⟩ := h i a ha
      rw [hx] at hx'; cases hx'
      exact ⟨nb', by simpa using hs⟩)
  unfold answersOf
  simp only [bind, Except.bind, hF]
  simp [pure, Except.pure]

/-- A collective's law reads the `k`-th call of each thread, nothing else of
    the calls. -/
theorem laneArg_congr {F F' : Nat → List (Intr × List V)} {k : Nat} (hF : ∀ t, (F t)[k]? = (F' t)[k]?)
    (warp lane : Nat) (c : Intr) : laneArg F warp lane k c = laneArg F' warp lane k c := by
  unfold laneArg; rw [hF]

/-- **A law answers alike** where every thread's `k`-th call is alike and, at
    a barrier, the snapshot it reads. -/
theorem answerStep_congr {F F' : Nat → List (Intr × List V)} {snaps snaps' : Array ByteArray} {smem0 : ByteArray}
    {warp lane nb k : Nat} {out : Array Answer} {x : Intr × List V} (hF : ∀ t, (F t)[k]? = (F' t)[k]?)
    (hs : x.1 = .barrier → snaps[nb]? = snaps'[nb]?) :
    answerStep F snaps smem0 warp lane (nb, out) (x, k) = answerStep F' snaps' smem0 warp lane (nb, out) (x, k) := by
  obtain ⟨c, args⟩ := x
  have hl := laneArg_congr hF
  unfold answerStep
  dsimp only
  split
  · rw [hs rfl]
  all_goals simp only [hl]

/-- **What the laws give back stays given back** as the threads go on: where
    every thread's `k`-th call is the same, thread `t`'s calls before it are
    the same, and a barrier's snapshot is the same. -/
theorem LawAt.mono {F F' : Nat → List (Intr × List V)} {snaps snaps' : Array ByteArray} {smem0 : ByteArray}
    {t k : Nat} {a : Answer} (h : LawAt F snaps smem0 t k a) (hF : ∀ t', (F t')[k]? = (F' t')[k]?)
    (htk : (F t).take k = (F' t).take k)
    (hs : ∀ x, (F t)[k]? = some x → x.1 = .barrier →
      snaps[((F t).take k).countP (·.1 == .barrier)]? = snaps'[((F t).take k).countP (·.1 == .barrier)]?) :
    LawAt F' snaps' smem0 t k a := by
  obtain ⟨x, nb', hx, hst⟩ := h
  refine ⟨x, nb', by rw [← hF]; exact hx, ?_⟩
  rw [← htk, ← answerStep_congr hF (hs x hx), hst]

theorem collCalls_obsStore {w : World} {st : Nat} (hs : st ≤ w.obs.length) (p : UInt64) (n : Nat) (x : UInt64) :
    collCalls (obsStore w p n x) st = collCalls w st := by
  unfold collCalls obsStore
  simp only [List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]
  simp [collOf]


/-- Answers whose values are 32-bit words. -/
def Words (as : List Answer) : Prop := ∀ a ∈ as, ∀ v, a = .val v → ∃ x : UInt32, v = .sc .i32 (s32 x)

theorem lane32_word (a : List V) : ∃ x : UInt32, lane32 a = .sc .i32 (s32 x) := by
  unfold lane32; split
  · exact ⟨_, rfl⟩
  · exact ⟨0, rfl⟩

/-- **The laws answer words.** -/
theorem answersOf_words {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} {as : List Answer} (h : answersOf calls snaps smem0 t = .ok as) : Words as := by
  unfold answersOf at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i r hr
  obtain ⟨_, out⟩ := r
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨h1, _, h3⟩ := foldlM_zipIdx_push _
    (fun _ (_ : Intr × List V) y => ∀ v, y = .val v → ∃ x : UInt32, v = .sc .i32 (s32 x))
    (fun s out x k r hr => by
      obtain ⟨c, args⟩ := x
      unfold answerStep at hr
      dsimp only at hr
      split at hr
      · simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact lane32_word _⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact lane32_word _⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact ⟨_, rfl⟩⟩
      · cases hr)
    (calls t) 0 0 #[] _ hr
  intro a ha v hv
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
  simp only [Array.length_toList] at hi
  obtain ⟨y, hy, hP⟩ := h3 i ((calls t)[i]'(by simp at h1; omega)) (by simp)
  simp only [Array.size_empty, Nat.zero_add] at hy
  simp only [Array.getElem_toList] at hv
  rw [Array.getElem?_eq_getElem hi] at hy
  cases hy
  exact hP v hv

theorem mapM_ok {α β : Type} (g : α → Except String β) : ∀ (l : List α) (per : List β), l.mapM g = .ok per →
    per.length = l.length ∧ ∀ i (h : i < l.length), ∃ y, per[i]? = some y ∧ g l[i] = .ok y
  | [], per, h => by cases h; exact ⟨rfl, fun i h => by cases h⟩
  | x :: xs, per, h => by
    simp only [List.mapM_cons, bind, Except.bind] at h
    split at h
    · cases h
    rename_i y hy
    split at h
    · cases h
    rename_i ys hys
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨h1, h2⟩ := mapM_ok g xs ys hys
    refine ⟨by simp [h1], fun i hi => ?_⟩
    cases i with
    | zero => exact ⟨y, rfl, hy⟩
    | succ i =>
      obtain ⟨z, hz, hg⟩ := h2 i (by simpa using hi)
      exact ⟨z, by simpa using hz, by simpa using hg⟩

/-- What the laws answer thread `t`. -/
theorem answers_at {n : Nat} {smem0 : ByteArray} {calls : Nat → List (Intr × List V)}
    {stores : Nat → List (Nat × UInt64 × Nat × UInt64)} {f : Nat → List Answer}
    (h : answers n smem0 calls stores = .ok f) {t : Nat} (ht : t < n) :
    answersOf calls (snapshots n smem0 calls stores) smem0 t = .ok (f t) := by
  unfold answers at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i per hper
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨-, hi⟩ := mapM_ok _ _ _ hper
  obtain ⟨y, hy, hg⟩ := hi t (by simpa using ht)
  simp only [List.getElem_range] at hg
  rw [hg]
  simp [List.getD, hy]

theorem passStep_other {run : Runner} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer}
    {s s' : PassSt} {t t' : Nat} (h : passStep run smem0 args ans s t = .ok s') (hne : t' ≠ t) :
    s'.2.1 t' = s.2.1 t' ∧ s'.2.2 t' = s.2.2 t' := by
  obtain ⟨w1, c1, s1⟩ := s
  simp only [passStep, bind, Except.bind] at h
  split at h
  · cases h
  cases h
  simp [hne]

theorem pass_keep {run : Runner} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer}
    {t : Nat} (ts : List Nat) (ht : t ∉ ts) (s r : PassSt)
    (h : ts.foldlM (passStep run smem0 args ans) s = .ok r) : r.2.1 t = s.2.1 t ∧ r.2.2 t = s.2.2 t :=
  pass_inv (fun s' => s'.2.1 t = s.2.1 t ∧ s'.2.2 t = s.2.2 t) ts s r
    (fun s1 t1 s2 ht1 h1 h2 => by
      have := passStep_other (t' := t) h2 (fun e => ht (by rw [e]; exact ht1))
      exact ⟨this.1.trans h1.1, this.2.trans h1.2⟩) ⟨rfl, rfl⟩ h

/-- **A pass of threads**: if each thread, from memory where its precondition
    holds, leaves memory with its effect and passes collectives and shared
    stores `C` allows of the memory it started from, and no thread disturbs
    another's precondition, then a pass that finishes leaves the effects
    composed, every thread's collectives and shared stores as `C` allows of the
    memory the threads before it left. -/
theorem pass_spec {run : Runner} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer}
    (P : Nat → Mem → Prop) (f : Nat → Mem → Mem)
    (C : Nat → Mem → List (Intr × List V) → List (Nat × UInt64 × Nat × UInt64) → Prop)
    (hrun : ∀ t (s u : World), P t s.mem →
      run false (ans t) (args t) { s with mem := { s.mem with arena := smem0 } } = .ok u →
      u.mem = f t s.mem ∧ C t s.mem (collCalls u s.obs.length) (sharedStores u s.obs.length))
    (hkeep : ∀ t u m, t ≠ u → P u m → P u (f t m)) :
    ∀ (ts : List Nat) (s r : PassSt), ts.Nodup → (∀ t ∈ ts, P t s.1.mem) →
      ts.foldlM (passStep run smem0 args ans) s = .ok r →
      r.1.mem = ts.foldl (fun m t => f t m) s.1.mem ∧
        ∀ t ∈ ts, C t ((ts.takeWhile (· ≠ t)).foldl (fun m t => f t m) s.1.mem) (r.2.1 t) (r.2.2 t)
  | [], s, r, _, _, h => by cases h; exact ⟨rfl, fun _ h => by cases h⟩
  | t :: ts, s, r, hnd, hP, h => by
    simp only [List.foldlM_cons, bind, Except.bind] at h
    split at h
    · cases h
    rename_i s1 hs1
    have hnt : t ∉ ts := (List.nodup_cons.mp hnd).1
    obtain ⟨w1, c1, st1⟩ := s
    have hs1' := hs1
    simp only [passStep, bind, Except.bind] at hs1'
    split at hs1'
    · cases hs1'
    rename_i u hu
    cases hs1'
    obtain ⟨hm, hc⟩ := hrun t w1 u (hP t List.mem_cons_self) hu
    obtain ⟨h1, h2⟩ := pass_spec P f C hrun hkeep ts _ r (List.nodup_cons.mp hnd).2
      (fun u' hu' => by
        show P u' u.mem
        rw [hm]; exact hkeep t u' _ (fun e => hnt (e ▸ hu')) (hP u' (List.mem_cons_of_mem _ hu'))) h
    refine ⟨by rw [h1, List.foldl_cons, hm], fun t' ht' => ?_⟩
    rcases List.mem_cons.mp ht' with rfl | ht'
    · rw [(pass_keep ts hnt _ r h).1, (pass_keep ts hnt _ r h).2]
      simpa using hc
    · have hne : t' ≠ t := fun e => hnt (e ▸ ht')
      have := h2 t' ht'
      simp only at this
      rw [hm] at this
      simpa [List.takeWhile_cons, hne.symm] using this

/-- The word answered to a thread's `j`-th collective, zero where it is not
    one. -/
def wordAt (ans : List Answer) (j : Nat) : UInt32 :=
  match ans[j]? with
  | some (.val (.sc _ x)) => x.toUInt32
  | _ => 0

theorem words_at {ans : List Answer} (hw : Words ans) :
    ∀ (j : Nat) (v : V), ans[j]? = some (.val v) → v = .sc .i32 (s32 (wordAt ans j)) := by
  intro j v h
  obtain ⟨x, rfl⟩ := hw _ (List.mem_of_getElem? h) v rfl
  simp [wordAt, h]

theorem lane32_s32 (ty : ClifTy) (a : UInt32) (rest : List V) : lane32 (.sc ty (s32 a) :: rest) = .sc .i32 (s32 a) := by
  simp [lane32]

theorem wordAt_val {ans : List Answer} {j : Nat} {x : UInt32} {ty : ClifTy}
    (h : ans[j]? = some (.val (.sc ty (s32 x)))) : wordAt ans j = x := by
  simp [wordAt, h]

/-- Where the answers are words, what a collective is answered is its word. -/
theorem valAt_words {ans : List Answer} (hw : Words ans) (j : Nat) :
    valAt ans j = .sc .i32 (s32 (wordAt ans j)) := by
  unfold valAt wordAt
  cases h : ans[j]? with
  | none => rfl
  | some a =>
    cases a with
    | smem b => rfl
    | val v =>
      obtain ⟨x, rfl⟩ := hw _ (List.mem_of_getElem? h) v rfl
      simp [s32]

section Steps

/-- A two-argument collective: what it gives back is its answer. -/
theorem wp_coll2 {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    {c : Intr} (hc : c.isColl = true) (r : LocalRef [.i32, .i32] (some .i32)) (hr : r.callee = .intr c)
    {a b : Slot .i32} {x y : V} (ha : Γ[a]? = some x) (hb : Γ[b]? = some y)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k)
    {v0 : V} (hv : valAt ans k = v0)
    {Q : Slot .i32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push v0) (obsCall w (.intr c) [x, y])) (hcb : c ≠ .barrier := by decide) :
    wp cfg d J (AlgorithmLib.Prog.callLocal r %[a, b]) Q Γ w := by
  obtain ⟨callee⟩ := r
  cases hr
  have hl := hA.val c [x, y] (obsCall w (.intr c) [x, y]) hc hcb
  rw [collCalls_call hs, hc] at hl
  simp only [if_true, List.length_append, List.length_cons, List.length_nil, hk, Nat.add_sub_cancel, hv] at hl
  simp only [AlgorithmLib.Prog.callLocal, wp_callLocal]
  intro vs' hvs'
  simp [Vals.slots, ha, hb] at hvs'
  subst hvs'
  refine ⟨fun hn => by simp [callOf, hl, Except.toOption] at hn, fun x' w' hc' => ?_⟩
  simp only [callOf, hl, Except.toOption, Option.some.injEq, Prod.mk.injEq] at hc'
  obtain ⟨rfl, rfl⟩ := hc'
  simp only [Option.isSome, if_true]
  intro v hv'; cases hv'
  rw [wp_ret]
  exact h

/-- A barrier, the world after it as `obsCall` builds it: what it gives back is
    the block's shared memory its answer holds. -/
theorem wp_bar_at {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    (r : LocalRef [] none) (hr : r.callee = .intr .barrier)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k)
    {Q : Unit → Env → World → Prop}
    (h : ∀ b, ans[k]? = some (.smem b) →
      Q () Γ { obsCall w (.intr .barrier) [] with
        mem := { (obsCall w (.intr .barrier) []).mem with arena := b } }) :
    wp cfg d J (AlgorithmLib.Prog.callLocalVoid r %[]) Q Γ w := by
  obtain ⟨callee⟩ := r
  cases hr
  simp only [AlgorithmLib.Prog.callLocalVoid, AlgorithmLib.Prog.callLocal, wp_bind, wp_callLocal]
  intro vs' hvs'
  simp [Vals.slots] at hvs'
  subst hvs'
  refine ⟨fun _ => ⟨fun m hm => ?_, fun m hm => ?_⟩, fun x' w' hc' => ?_⟩
  · simp [failOf, refused] at hm
  · simp [failOf, refused] at hm
  · simp only [callOf] at hc'
    simp only [Option.isSome, Bool.false_eq_true, if_false]
    cases hl : cfg.locals Intr.barrier.slot [] (obsCall w (.intr .barrier) []) with
    | error e => simp [hl, Except.toOption] at hc'
    | ok p =>
      obtain ⟨r1, w1⟩ := p
      simp only [hl, Except.toOption, Option.some.injEq, Prod.mk.injEq] at hc'
      obtain ⟨rfl, rfl⟩ := hc'
      obtain ⟨-, b', rfl, hans⟩ := hA.smem _ _ _ _ hl
      rw [collCalls_call hs] at hans
      simp only [Intr.isColl, if_true, List.length_append, List.length_cons, List.length_nil, hk,
        Nat.add_sub_cancel] at hans
      rw [wp_ret, wp_pure]
      exact h b' hans

/-! ### Ending rules for collectives -/

section Ends
open AlgorithmLib.LibSpec
variable {cfg : Cfg}

/-- A two-argument collective ends: it gives back the value its answer holds. -/
theorem wpE_coll2 {Γ : Env} {w : World} {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    {c : Intr} (hc : c.isColl = true) (r : LocalRef [.i32, .i32] (some .i32)) (hr : r.callee = .intr c)
    {a b : Slot .i32} {x y : V} (ha : Γ[a]? = some x) (hb : Γ[b]? = some y)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k)
    {v0 : V} (hv : valAt ans k = v0) {Q : Slot .i32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push v0) (obsCall w (.intr c) [x, y])) (hcb : c ≠ .barrier := by decide) :
    wpE cfg (AlgorithmLib.Prog.callLocal r %[a, b]) Q Γ w := by
  obtain ⟨callee⟩ := r
  cases hr
  have hl := hA.val c [x, y] (obsCall w (.intr c) [x, y]) hc hcb
  rw [collCalls_call hs, hc] at hl
  simp only [if_true, List.length_append, List.length_cons, List.length_nil, hk, Nat.add_sub_cancel, hv] at hl
  exact wpE_callLocal (vs := [x, y]) (v := v0) (w' := obsCall w (.intr c) [x, y]) rfl
    (by simp [Vals.slots, ha, hb]) (by simp [callOf, hl, Except.toOption]) (wpE_ret h)

/-- A one-argument collective ends: it gives back the value its answer holds. -/
theorem wpE_coll1 {Γ : Env} {w : World} {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    {c : Intr} (hc : c.isColl = true) {ty : ClifTy} (r : LocalRef [ty] (some .i32)) (hr : r.callee = .intr c)
    {a : Slot ty} {x : V} (ha : Γ[a]? = some x)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k)
    {v0 : V} (hv : valAt ans k = v0) {Q : Slot .i32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push v0) (obsCall w (.intr c) [x])) (hcb : c ≠ .barrier := by decide) :
    wpE cfg (AlgorithmLib.Prog.callLocal r %[a]) Q Γ w := by
  obtain ⟨callee⟩ := r
  cases hr
  have hl := hA.val c [x] (obsCall w (.intr c) [x]) hc hcb
  rw [collCalls_call hs, hc] at hl
  simp only [if_true, List.length_append, List.length_cons, List.length_nil, hk, Nat.add_sub_cancel, hv] at hl
  exact wpE_callLocal (vs := [x]) (v := v0) (w' := obsCall w (.intr c) [x]) rfl
    (by simp [Vals.slots, ha]) (by simp [callOf, hl, Except.toOption]) (wpE_ret h)

/-- A barrier ends where its answer holds shared memory, which it puts in
    place. -/
theorem wpE_bar {Γ : Env} {w : World} {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    (r : LocalRef [] none) (hr : r.callee = .intr .barrier)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k) {b : ByteArray}
    (hb : ans[k]? = some (.smem b)) {Q : Unit → Env → World → Prop}
    (h : Q () Γ { obsCall w (.intr .barrier) [] with
      mem := { (obsCall w (.intr .barrier) []).mem with arena := b } }) :
    wpE cfg (AlgorithmLib.Prog.callLocalVoid r %[]) Q Γ w := by
  obtain ⟨callee⟩ := r
  cases hr
  have hl := hA.bar [] (obsCall w (.intr .barrier) []) b (by
    rw [collCalls_call hs]
    simp only [Intr.isColl, if_true, List.length_append, List.length_cons, List.length_nil, hk,
      Nat.add_sub_cancel]
    exact hb)
  unfold AlgorithmLib.Prog.callLocalVoid AlgorithmLib.Prog.callLocal
  exact wpE_bind (wpE_callLocalVoid (vs := []) (x := none) rfl (by simp [Vals.slots])
    (by simp [callOf, hl, Except.toOption]) (wpE_ret (show wpE cfg (pure ()) Q Γ _ from wpE_pure h)))

end Ends

theorem obsCall_mem (w : World) (c : Callee) (vs : List V) : (obsCall w c vs).mem = w.mem := rfl

theorem obsCall_length (w : World) (c : Callee) (vs : List V) :
    (obsCall w c vs).obs.length = w.obs.length + 1 := by simp [obsCall]

end Steps

-- ---------------------------------------------------------------------------
-- Shared memory and barriers
-- ---------------------------------------------------------------------------

/-- Shared memory's address. -/
abbrev smemAddr : UInt64 := addrOf .arena 0

/-- Word `i` of shared memory `B`. -/
def aload (B : ByteArray) (i : Nat) : Option UInt64 :=
  ({ arena := B, data := .empty, out := .empty } : Mem).load (smemAddr + off (UInt32.ofNat i)) 4

theorem decode_arena (i : UInt32) : decodeAddr (smemAddr + off i) = some (.arena, 4 * i.toNat) := by
  have := i.toNat_lt
  rw [off_eq, smemAddr, addrOf, show regionBase .arena + UInt64.ofNat 0 = regionBase .arena by simp]
  exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; omega)

theorem load_arena (m : Mem) (hL : m.lent = []) (i : UInt32) :
    m.load (smemAddr + off i) 4 = aload m.arena i.toNat := by
  unfold aload
  rw [UInt32.ofNat_toNat]
  simp only [Mem.load]
  rw [decode_arena]
  simp [Mem.region, Mem.readable, Mem.lentClear, hL]

theorem load_arena0 (m : Mem) (hL : m.lent = []) : m.load smemAddr 4 = aload m.arena 0 := by
  have := load_arena m hL 0
  simpa using this

theorem store_arena_eq {m m' : Mem} {i : UInt32} {v : UInt64} (h : m.store (smemAddr + off i) 4 v = some m') :
    m' = { m with arena := m'.arena } := by
  simp only [Mem.store] at h
  rw [decode_arena] at h
  simp only [bind, Option.bind] at h
  split at h
  · cases h
  · cases h; rfl

theorem store_arena_some (m : Mem) (i : UInt32) (v : UInt64) (h : 4 * i.toNat + 4 ≤ m.arena.size)
    (hL : m.lent = []) : (m.store (smemAddr + off i) 4 v).isSome := by
  simp only [Mem.store]
  rw [decode_arena]
  simp [Mem.region, Mem.reachable, Mem.lentClear, hL]
  omega

theorem addr_step (t : UInt32) (s : Nat) (k : Int) (hk : k = 4 * (s : Int)) (h : t.toNat + s < 2 ^ 32) :
    smemAddr + off t + c64 k = smemAddr + off (UInt32.ofNat (t.toNat + s)) := by
  subst hk
  have h64 : (1 <<< 64 : Nat) = 18446744073709551616 := rfl
  have he : (4 * (s : Int)).emod ((1 <<< 64 : Nat) : Int) = 4 * (s : Int) := by
    rw [h64]; exact Int.emod_eq_of_lt (by omega) (by omega)
  rw [off_eq, off_eq, toNat_ofNat32 h, UInt64.add_assoc]
  congr 1
  simp only [c64]
  rw [he]
  rw [← UInt64.ofNat_add]
  congr 1
  omega

theorem wp_store_arena {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World} {v : Slot .f32} {a : Slot .i64}
    {x : UInt64} {i : UInt32} {Q : Unit → Env → World → Prop}
    (hv : Γ[v]? = some (.sc .f32 x)) (ha : Γ[a]? = some (.sc .i64 (smemAddr + off i)))
    (hsz : 4 * i.toNat + 4 ≤ w.mem.arena.size) (hL : w.mem.lent = [])
    (h : ∀ w1 : World, w1.obs = .store (smemAddr + off i) 4 x :: w.obs →
      w1.mem = { w.mem with arena := w1.mem.arena } → Q () Γ w1) :
    wp cfg d J (store v a) Q Γ w := by
  obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (store_arena_some w.mem i x hsz hL)
  exact wp_storeF32 hv ha hm (h _ rfl (store_arena_eq hm))

theorem collCalls_evs (w u : World) (L : List Obs) (h : u.obs = L.reverse ++ w.obs) :
    collCalls u w.obs.length = L.filterMap collOf := by
  unfold collCalls
  rw [h, List.reverse_append, List.reverse_reverse, List.drop_append_of_le_length (by simp),
    show w.obs.length = w.obs.reverse.length by simp, List.drop_length]
  simp

theorem sharedStores_evs (w u : World) (L : List Obs) (h : u.obs = L.reverse ++ w.obs) :
    sharedStores u w.obs.length = (L.foldl shStep (0, [])).2.reverse := by
  unfold sharedStores
  rw [h, List.reverse_append, List.reverse_reverse, List.drop_append_of_le_length (by simp),
    show w.obs.length = w.obs.reverse.length by simp, List.drop_length]
  simp

theorem shStep_arena (n : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)) (t : UInt32) (v : UInt64) :
    shStep (n, acc) (.store (smemAddr + off t) 4 v) = (n, (n, smemAddr + off t, 4, v) :: acc) := by
  simp only [shStep]; rw [decode_arena]

theorem shStep_buf (n : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)) {a : UInt64} {i : Fin 268435449} {o : Nat}
    (hd : decodeAddr a = some (.buf i, o)) (v : UInt64) : shStep (n, acc) (.store a 4 v) = (n, acc) := by
  simp [shStep, hd]

/-- Word `i` of shared memory `B`, zero where there is none. -/
def aword (B : ByteArray) (i : Nat) : UInt32 := ((aload B i).getD 0).toUInt32

theorem aload_word (B : ByteArray) (i : Nat) (hi : i < 2 ^ 32) (h : 4 * i + 4 ≤ B.size) :
    aload B i = some (s32 (aword B i)) := by
  have hl : (aload B i).isSome := by
    unfold aload
    have hc := toNat_ofNat32 hi
    simp only [Mem.load]
    rw [decode_arena]
    simp [Mem.region, Mem.readable, Mem.lentClear]
    omega
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp hl
  have hlt := load4_lt hv
  simp only [aword, hv, Option.getD_some, s32]
  congr 1
  apply UInt64.toNat_inj.mp
  simp [Nat.mod_eq_of_lt hlt]

/-- Word `t` of shared memory's address. -/
abbrev aaddr (t : Nat) : UInt64 := smemAddr + off (UInt32.ofNat t)

theorem aword_of (m : Mem) (hL : m.lent = []) (i : Nat) (hi : i < 2 ^ 32) :
    aword m.arena i = ((m.load (aaddr i) 4).getD 0).toUInt32 := by
  rw [aaddr, load_arena m hL, toNat_ofNat32 hi]; rfl

theorem puts_size (val : Nat → UInt64) : ∀ (ts : List Nat) (m : Mem),
    (ts.foldl (fun m t => put aaddr val t m) m).arena.size = m.arena.size
  | [], _ => rfl
  | t :: ts, m => by
    rw [List.foldl_cons, puts_size val ts]
    unfold put
    cases h : m.store (aaddr t) 4 (val t) with
    | none => rfl
    | some m' => exact store_sizes h .arena

theorem puts_lent (val : Nat → UInt64) : ∀ (ts : List Nat) (m : Mem),
    (ts.foldl (fun m t => put aaddr val t m) m).lent = m.lent
  | [], _ => rfl
  | t :: ts, m => by
    rw [List.foldl_cons, puts_lent val ts]
    unfold put
    cases h : m.store (aaddr t) 4 (val t) with
    | none => rfl
    | some m' => exact Mem.store_lent h

theorem store_buf_arena (m : Mem) (A' : ByteArray) {a : UInt64} {i : Fin 268435449} {o : Nat}
    (hd : decodeAddr a = some (.buf i, o)) (n : Nat) (v : UInt64) :
    ({ m with arena := A' } : Mem).store a n v = (m.store a n v).map fun m' => { m' with arena := A' } := by
  simp only [Mem.store, hd, Mem.region, Mem.reachable, Mem.lentClear, Mem.setRegion]
  simp only [bind, Option.bind]
  split <;> simp_all

theorem load_buf_arena (m : Mem) (A' : ByteArray) {a : UInt64} {i : Fin 268435449} {o : Nat}
    (hd : decodeAddr a = some (.buf i, o)) (n : Nat) :
    ({ m with arena := A' } : Mem).load a n = m.load a n :=
  load_buf_eq hd rfl

theorem flatMap_congr' {α β : Type} {f g : α → List β} : ∀ (l : List α), (∀ x ∈ l, f x = g x) →
    l.flatMap f = l.flatMap g
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.flatMap_cons, h x List.mem_cons_self,
      flatMap_congr' xs (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem phase_size_any (all : List (Nat × UInt64 × Nat × UInt64)) (k : Nat) : ∀ (cur : Mem),
    (phase all k cur).arena.size = cur.arena.size := by
  unfold phase
  induction all with
  | nil => intro cur; rfl
  | cons x xs ih =>
    intro cur
    rw [List.foldl_cons, ih]
    split
    · cases h : cur.store x.2.1 x.2.2.1 x.2.2.2 with
      | none => rfl
      | some m' => exact store_sizes h .arena
    · rfl

theorem snapshots_size (n : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) :
    ∀ B ∈ snapshots n smem0 calls stores, B.size = smem0.size := by
  unfold snapshots
  generalize (List.range n).foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) 0 = nb
  generalize (List.range n).flatMap stores = all
  have key : ∀ (l : List Nat) (acc : Mem × Array ByteArray), acc.1.arena.size = smem0.size →
      (∀ B ∈ acc.2, B.size = smem0.size) →
      (l.foldl (fun (acc : Mem × Array ByteArray) b =>
        (phase all b acc.1, acc.2.push (phase all b acc.1).arena)) acc).1.arena.size = smem0.size ∧
      ∀ B ∈ (l.foldl (fun (acc : Mem × Array ByteArray) b =>
        (phase all b acc.1, acc.2.push (phase all b acc.1).arena)) acc).2, B.size = smem0.size := by
    intro l
    induction l with
    | nil => intro acc h1 h2; exact ⟨h1, h2⟩
    | cons x xs ih =>
      intro acc h1 h2
      rw [List.foldl_cons]
      apply ih
      · simp only; rw [phase_size_any]; exact h1
      · intro B hB
        simp only [Array.mem_push] at hB
        rcases hB with hB | rfl
        · exact h2 B hB
        · rw [phase_size_any]; exact h1
  exact (key _ _ rfl (by simp)).2

/-- **The laws answer shared memory of shared memory's size.** -/
theorem answersOf_smem_size {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    (hsn : ∀ B ∈ snaps, B.size = smem0.size) {t : Nat} {as : List Answer}
    (h : answersOf calls snaps smem0 t = .ok as) :
    ∀ (k : Nat) (B : ByteArray), as[k]? = some (.smem B) → B.size = smem0.size := by
  unfold answersOf at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i r hr
  obtain ⟨_, out⟩ := r
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨h1, _, h3⟩ := foldlM_zipIdx_push _
    (fun _ (_ : Intr × List V) y => ∀ B, y = .smem B → B.size = smem0.size)
    (fun s out x k r hr => by
      obtain ⟨c, args⟩ := x
      unfold answerStep at hr
      dsimp only at hr
      split at hr
      · simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        refine ⟨_, rfl, fun B hB => ?_⟩
        cases hB
        cases hs : snaps[s]? with
        | none => simp
        | some B' => simp; exact hsn B' (Array.mem_of_getElem? hs)
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun B hB => by cases hB⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun B hB => by cases hB⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun B hB => by cases hB⟩
      · cases hr)
    (calls t) 0 0 #[] _ hr
  intro k B hk
  simp only [Array.getElem?_toList] at hk
  have hkl : k < (calls t).length := by
    have := (Array.getElem?_eq_some_iff.mp hk).1; simp at h1; omega
  obtain ⟨y, hy, hP⟩ := h3 k ((calls t)[k]'hkl) (by simp)
  simp only [Array.size_empty, Nat.zero_add] at hy
  rw [hy] at hk
  cases hk
  exact hP B rfl

/-- What the laws answer a thread whose collectives are all barriers: the
    snapshots, in order. -/
theorem answersOf_bars {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t n : Nat} (hc : calls t = List.replicate n (.barrier, [])) :
    answersOf calls snaps smem0 t = .ok ((List.range n).map fun k => .smem (snaps[k]?.getD smem0)) := by
  have key : ∀ (n st m : Nat) (out : Array Answer),
      ((List.replicate n ((Intr.barrier, []) : Intr × List V)).zipIdx st).foldlM
        (answerStep calls snaps smem0 (t / 32) (t % 32)) (m, out) =
        .ok (m + n, out ++ ((List.range n).map fun k => Answer.smem (snaps[m + k]?.getD smem0)).toArray) := by
    intro n
    induction n with
    | zero => intro st m out; simp; rfl
    | succ n ih =>
      intro st m out
      rw [List.replicate_succ, List.zipIdx_cons, List.foldlM_cons]
      simp only [answerStep, pure, Except.pure, bind, Except.bind]
      rw [ih]
      congr 2
      · omega
      · simp [List.range_succ_eq_map, Nat.add_assoc, Nat.add_comm 1]; rfl
  unfold answersOf
  rw [hc]
  simp only [bind, Except.bind]
  rw [key]
  simp
  rfl

theorem foldl_arena_init {g : Nat → Mem → Mem}
    (hg : ∀ t m x, g t { m with arena := x } = { g t m with arena := x }) :
    ∀ (l : List Nat) (m : Mem) (x a : ByteArray),
      ({ l.foldl (fun m t => g t m) { m with arena := x } with arena := a } : Mem) =
        { l.foldl (fun m t => g t m) m with arena := a }
  | [], _, _, _ => rfl
  | u :: us, m, x, a => by rw [List.foldl_cons, List.foldl_cons, hg, foldl_arena_init hg us]

theorem foldl_arena_eq {f g : Nat → Mem → Mem} (Ar : Nat → ByteArray)
    (hf : ∀ t m, f t m = { g t m with arena := Ar t })
    (hg : ∀ t m x, g t { m with arena := x } = { g t m with arena := x }) :
    ∀ (l : List Nat) (m : Mem) (a : ByteArray),
      ({ l.foldl (fun m t => f t m) m with arena := a } : Mem) = { l.foldl (fun m t => g t m) m with arena := a }
  | [], _, _ => rfl
  | t :: ts, m, a => by
    rw [List.foldl_cons, List.foldl_cons, hf, foldl_arena_eq Ar hf hg ts, foldl_arena_init hg]

/-- smemAddr barrier, the world after it described by what it holds. -/
theorem wp_bar {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World} {ans : List Answer}
    {start : Nat} (hA : AnswersFrom ans start cfg) (r : LocalRef [] none) (hr : r.callee = .intr .barrier)
    (hs : start ≤ w.obs.length) {k : Nat} (hk : (collCalls w start).length = k)
    {Q : Unit → Env → World → Prop}
    (h : ∀ (w1 : World) (b : ByteArray), ans[k]? = some (.smem b) →
      w1.obs = .call (.intr .barrier) [] :: w.obs → w1.mem = { w.mem with arena := b } → Q () Γ w1) :
    wp cfg d J (AlgorithmLib.Prog.callLocalVoid r %[]) Q Γ w :=
  wp_bar_at hA r hr hs hk (fun b hb => h _ b hb (by simp [obsCall, callWords]) rfl)

end AlgorithmLib.Kernel
