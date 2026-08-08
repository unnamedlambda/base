import AlgorithmLib.ML.Fuse
import AlgorithmLib.ML.LocalBind
import AlgorithmLib.ML.Sched
import AlgorithmLib.ML.PtxPrint

/-!
# A schedule is one value

The model term says what to compute.  Everything about *how* — which
implementation a contraction gets, where row passes fuse, how a warp walks a
row, which loads may take the read-only path — is a `TenSchedule`, and
`TenProg.compile` is the single place it is applied.

The point of collecting them is `compile_den`: whatever schedule is named, the
compiled tape computes what the default one computes, on every buffer a fusion
did not remove.  A model gets that at build time with no proof text, because
every side condition was decided inside the library — `fuseNormAt` refuses a
site it cannot justify, and a retarget cannot change a denotation it does not
appear in.
-/

namespace AlgorithmLib.ML

/-- **Send every contraction to one implementation.**

    The backend appears in exactly three constructors, and in none of their
    denotations: it is a lowering choice, so moving it is a schedule and not an
    edit to the model. -/
def TOp.retarget (bk : Backend) : TOp → TOp
  | .mv    _ w x o b i ow => .mv    bk w x o b i ow
  | .mvT   _ w d o b i ow => .mvT   bk w d o b i ow
  | .outer _ d x o b i ow => .outer bk d x o b i ow
  | op => op

/-- **Retargeting computes the same thing** — by cases, for every backend and
    every operation.  This is what makes the choice a schedule: `Backend.laws`
    records what the vendor call assumes, and the mathematics is untouched. -/
theorem TOp.retarget_den (bk : Backend) (op : TOp) : (op.retarget bk).den = op.den := by
  cases op <;> rfl

/-- **A rewrite a user asserts is correct.**

    `apply` runs on the tape unchecked.  `name` exists so the assertion has an
    identity in a report: an unnamed hole is one nobody can audit. -/
structure AssertedRewrite where
  name  : String
  apply : List TOp → List TOp

/-- What asserting one costs: that it preserves what every buffer holds. -/
def AssertedRewrite.Sound (a : AssertedRewrite) : Prop :=
  ∀ (t : List TOp) (m : Buf → Nat → Float32) (b : Buf),
    ((a.apply t).foldl (fun mm o => o.den mm) m) b
      = (t.foldl (fun mm o => o.den mm) m) b

def AssertedSound (as : List AssertedRewrite) : Prop := ∀ a ∈ as, a.Sound

/-- Asserting nothing costs nothing, so the default schedule's `compile_den`
    keeps its shape. -/
theorem assertedSound_nil : AssertedSound [] := by intro a ha; exact absurd ha (by simp)

def applyAsserted : List AssertedRewrite → List TOp → List TOp
  | [],      t => t
  | a :: as, t => applyAsserted as (a.apply t)

theorem applyAsserted_den : ∀ (as : List AssertedRewrite), AssertedSound as →
    ∀ (t : List TOp) (m : Buf → Nat → Float32) (b : Buf),
      ((applyAsserted as t).foldl (fun mm o => o.den mm) m) b
        = (t.foldl (fun mm o => o.den mm) m) b := by
  intro as
  induction as with
  | nil => intro _ t m b; rfl
  | cons a as ih =>
      intro h t m b
      exact Eq.trans (ih (fun x hx => h x (List.mem_cons_of_mem _ hx)) (a.apply t) m b)
        (h a (by simp) t m b)

/-- **A schedule.**

    Every field is a choice among moves the library has already proven
    equivalent, and every field has a default — so `{}` is the schedule that
    makes none of them. -/
structure TenSchedule where
  /-- Which implementation the contractions get. -/
  impl : Backend := .proven
  /-- Where to fuse a row-pass pair, applied in the order given.  `fuseTargets`
      reports the sites a tape offers, so this is chosen from a list rather than
      counted out. -/
  fuse : List FuseSite := []
  /-- How a warp walks a row. -/
  warp : Sched := .vec4
  /-- Which loads may take the read-only path. -/
  ro : ROPolicy := .all
  /-- **Rewrites the library has not proven.**

      The escape hatch: a move outside the menu, applied to the tape as given.
      It does not leave the accounting, it moves a line in it — `compile_den`
      for a schedule carrying these takes their soundness as a *hypothesis*, the
      same way a vendor call's contract is one, and `assertedNames` is what a
      build report prints.  A schedule with an assertion cannot bill nothing. -/
  asserted : List AssertedRewrite := []

/-- Fuse at each named position in turn, collecting the intermediates removed.

    A position no fusion applies to leaves the tape alone: `fuseNormAt` decides
    the site, so naming a bad one costs a kernel that is not fused, never a
    tape that computes something else. -/
def applyFuse (tbl : List (String × Buf)) : List FuseSite → List TOp → List Buf × List TOp
  | [], t => ([], t)
  | s :: ss, t =>
      match fuseAt tbl t s with
      | some (tmp, t') =>
          match applyFuse tbl ss t' with
          | (bs, tt) => (tmp :: bs, tt)
      | none => applyFuse tbl ss t

/-- **Fusing preserves the tape's mathematics on every buffer it did not
    remove**, for any list of sites. -/
theorem applyFuse_den (tbl : List (String × Buf)) :
    ∀ (ss : List FuseSite) (t : List TOp) (m : Buf → Nat → Float32)
    (b : Buf), b ∉ (applyFuse tbl ss t).1 →
    ((applyFuse tbl ss t).2.foldl (fun mm o => o.den mm) m) b
      = (t.foldl (fun mm o => o.den mm) m) b := by
  intro ss
  induction ss with
  | nil => intro t m b _; rfl
  | cons s ss ih =>
      intro t m b hb
      cases hf : fuseAt tbl t s with
      | none =>
          rw [show applyFuse tbl (s :: ss) t = applyFuse tbl ss t by
                simp only [applyFuse, hf]] at hb ⊢
          exact ih t m b hb
      | some p =>
          obtain ⟨tmp, t'⟩ := p
          rw [show applyFuse tbl (s :: ss) t
                = (tmp :: (applyFuse tbl ss t').1, (applyFuse tbl ss t').2) by
                simp only [applyFuse, hf]] at hb ⊢
          have hne : b ≠ tmp := by
            intro e; subst e; exact hb (by simp)
          have hrest : b ∉ (applyFuse tbl ss t').1 := fun hm => hb (List.mem_cons_of_mem _ hm)
          exact Eq.trans (ih t' m b hrest) (fuseAt_den tbl t t' s tmp hf m b hne)

/-- **A retargeted tape folds to the same memory**, operation by operation. -/
theorem map_retarget_den (bk : Backend) : ∀ (t : List TOp) (m : Buf → Nat → Float32),
    (t.map (TOp.retarget bk)).foldl (fun mm o => o.den mm) m
      = t.foldl (fun mm o => o.den mm) m := by
  intro t
  induction t with
  | nil => intro m; rfl
  | cons o os ih =>
      intro m
      simp only [List.map_cons, List.foldl_cons, TOp.retarget_den]
      exact ih _

/-- The operations a program flattens to, with computed buffers allocated from
    `base` upward. -/
def TenProg.tape {r c : Nat} (base : Ref) (p : TenProg r c) : List TOp :=
  ((p RefV).flat base).2.2

/-- The names the model bound, and the buffer each one landed in. -/
def TenProg.labels {r c : Nat} (base : Ref) (p : TenProg r c) : List (String × Buf) :=
  (p RefV).labels base

/-- **The one place a schedule is applied.**

    Returns the intermediates fusion removed alongside the tape, because those
    are exactly the buffers `compile_den` cannot speak about. -/
def TenProg.compile {r c : Nat} (base : Ref) (p : TenProg r c) (s : TenSchedule := {}) :
    List Buf × List TOp :=
  match applyFuse (p.labels base) s.fuse ((p.tape base).map (TOp.retarget s.impl)) with
  | (bs, t) => (bs, applyAsserted s.asserted t)

/-- The assertions a schedule carries, for a build report. -/
def TenSchedule.assertedNames (s : TenSchedule) : List String :=
  s.asserted.map AssertedRewrite.name

/-- **Whatever the schedule, the compiled tape computes the model.**

    The build-time statement the frontend rests on: a model is written once,
    and any schedule named against it is checked to perform its mathematics —
    on every buffer except the intermediates the named fusions removed, which
    is the only thing a fusion is allowed to change. -/
theorem TenProg.compile_den {r c : Nat} (base : Ref) (p : TenProg r c) (s : TenSchedule)
    (ha : AssertedSound s.asserted)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ∉ (p.compile base s).1) :
    ((p.compile base s).2.foldl (fun mm o => o.den mm) m) b
      = ((p.tape base).foldl (fun mm o => o.den mm) m) b := by
  unfold TenProg.compile at hb ⊢
  exact Eq.trans (applyAsserted_den s.asserted ha _ m b)
    (Eq.trans (applyFuse_den (p.labels base) s.fuse _ m b hb)
      (congrFun (map_retarget_den s.impl (p.tape base) m) b))

end AlgorithmLib.ML
