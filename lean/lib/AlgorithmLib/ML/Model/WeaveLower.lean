module
public import AlgorithmLib.ML.Model.WeaveTOp
meta import AlgorithmLib.ML.Model.WeaveTOp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Compiling the algebra

  A `WOp` is an operation written the way the algebra describes one: a target
  primitive, the shape its result ranges over, and the shape it contracts away.
  It carries no grid, no lane count and no extent arithmetic — those are what
  `WOp.lower` *derives*, which is the paper's claim that an architecture's
  shapes determine its launch geometry rather than being annotations on it.

  `WOp.den` says what such an operation computes, addressing every operand as
  the row-major address of a reindexed point.  `WOp.lower_den` then shows the
  emitted `TOp` computes exactly that.  The two differ only in how an address
  is written — `flatten [rows, cols] [row, col]` against `row * cols + col`,
  and a `BCast` against the affine map `WeaveBCast` shows it to be — which is
  the content: the addressing a shipped kernel performs *is* indexing a shape
  at a reindexed point.

  `WOp.ofTOp` runs the other way, so an already-shipped tape can be *checked*
  to be an algebra program rather than rewritten into one.  That is what makes
  the ViT statement possible: 2294 operations, none of them written twice.

  The fragment is deliberately narrow.  A shape that is not rank two (rank one
  for a contracted axis) lowers to `none`, because nothing below this layer can
  address it: `IdxE` is affine in `(ctaId, loopI, laneId)` with no division, so
  a rank-three operand would need an index expression the hardware path cannot
  express.  Refusing is the honest answer, and it is a decidable one.
-/

namespace AlgorithmLib.ML.Broadcast

open AlgorithmLib.ML

/-- Row-major size of a rank-two shape. -/
theorem size_pair (r c : Nat) : Ix.size [r, c] = r * c := by
  simp [Ix.size]

/-- The address a row reduction reads at trip `t`, lane `l`, is the reindexed
    point of the operand's broadcast. -/
theorem rowdot_operand_is_reindex (mode : BCast) (n rows a t : Nat) (l : Lane) :
    mode.ix.eval a t l
      = flatten [rows, bcastCols n mode]
          ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mode]) mode).apply
            [a, t * 32 + l.val]) := by
  rw [BCast.ix_ev]
  exact (bcast_is_reindex mode n rows a (t * 32 + l.val)).symm

/-- **An operation, as the algebra describes one.**

    `dom` is the shape the result ranges over and `arg` the shape contracted
    away.  Everything the kernel needs — grid, extents, lane trip count — is a
    function of those two. -/
inductive WOp where
  /-- Contraction: `out[s][o] = Σₜ w[o][t] · x[s][t]`.  `alloc` is how many
      rows the output buffer holds, which differs from the contracted row count
      where a consumer's extent is taller (attention keys, padded). -/
  | contract   : Backend → (wb xb ob : Buf) → (dom arg : Ix) → (alloc : Nat) → WOp
  /-- The transposed walk of a contraction. -/
  | contractT  : Backend → (wb db ob : Buf) → (dom arg : Ix) → WOp
  /-- The batch-summed outer product. -/
  | outerProd  : Backend → (db xb ob : Buf) → (dom arg : Ix) → WOp
  /-- A one-operand pointwise pass. -/
  | pointwise1 : Expr 1 → (ab ob : Buf) → (dom : Ix) → WOp
  /-- A two-operand pointwise pass. -/
  | pointwise2 : Expr 2 → (ab bb ob : Buf) → (dom : Ix) → WOp
  /-- Softmax with the cross-entropy gradient folded in. -/
  | softmaxCE  : (lb bib ohb ob : Buf) → (dom : Ix) → WOp
  /-- A pointwise pass that writes its own first operand. -/
  | update2    : Expr 2 → (ab bb : Buf) → (dom : Ix) → WOp
  /-- A row pass over a window, each operand read through its own broadcast. -/
  | rowPass    : (ab bb ob : Buf) → WFExp → BCast → BCast → (dom : Ix) →
                 (off w : Nat) → WOp
  /-- The three-operand row pass. -/
  | rowPass3   : (ab bb cb ob : Buf) → WFExp → BCast → BCast → BCast →
                 (dom : Ix) → (off w : Nat) → WOp
  /-- The four-operand row pass. -/
  | rowPass4   : (ab bb cb db ob : Buf) → WFExp → BCast → BCast → BCast → BCast →
                 (dom : Ix) → (off w : Nat) → WOp
  /-- A row reduction of two broadcast operands. -/
  | rowReduce  : (ab bb ob : Buf) → BCast → BCast → (dom arg : Ix) → WOp
  /-- The row reduction with a row pass folded into its left factor. -/
  | rowReduce4 : (ab bb cb db ob : Buf) → WFExp → BCast → BCast → BCast → BCast →
                 (dom arg : Ix) → WOp
  /-- The row maximum, from a seed. -/
  | rowMax     : (ab ob : Buf) → (dom arg : Ix) → Float32 → WOp
  /-- The row sum of squares. -/
  | rowSq      : (ab ob : Buf) → (dom arg : Ix) → WOp

/-- **The launch geometry, derived from the shapes.**

    Every extent the emitted operation carries is read off `dom` and `arg`.  A
    shape outside the fragment the index language can address is refused. -/
def WOp.lower : WOp → Option TOp
  | .contract bk wb xb ob [b, outW] [inW] al => some (.mv bk wb xb ob b inW outW al)
  | .contractT bk wb db ob [b, inW] [outW]   => some (.mvT bk wb db ob b inW outW)
  | .outerProd bk db xb ob [outW, inW] [b]   => some (.outer bk db xb ob b inW outW)
  | .pointwise1 f ab ob [g, 32]              => some (.ew1 f ab ob g)
  | .pointwise2 f ab bb ob [g, 32]           => some (.ew2 f ab bb ob g)
  | .softmaxCE lb bib ohb ob [g, 32]         => some (.smce lb bib ohb ob g)
  | .update2 f ab bb [g, 32]                 => some (.upd2 f ab bb g)
  | .rowPass ab bb ob f mA mB [rows, n] off w =>
      some (.ziprow ab bb ob f mA mB n off w rows)
  | .rowPass3 ab bb cb ob f mA mB mC [rows, n] off w =>
      some (.ziprow3 ab bb cb ob f mA mB mC n off w rows)
  | .rowPass4 ab bb cb db ob f mA mB mC mD [rows, n] off w =>
      some (.ziprow4 ab bb cb db ob f mA mB mC mD n off w rows)
  | .rowReduce ab bb ob mA mB [rows] [n]     => some (.rowdot ab bb ob mA mB n rows)
  | .rowReduce4 ab bb cb db ob f mA mB mC mD [rows] [n] =>
      some (.rowdot4 ab bb cb db ob f mA mB mC mD n rows)
  | .rowMax ab ob [rows] [n] init            => some (.rowmax ab ob n rows init)
  | .rowSq ab ob [rows] [n]                  => some (.rowsq ab ob n rows)
  | _                                         => none

open Classical in
/-- **What an operation computes**, with every operand read at the row-major
    address of a reindexed point.  Outside its own domain it changes nothing,
    so composing operations is composing memory transformers. -/
noncomputable def WOp.den : WOp → (Buf → Nat → Float32) → Buf → Nat → Float32
  | .contract _ wb xb ob [b, outW] [inW] _ => fun m b' a =>
      if b' = ob ∧ a < b * outW then
        rowDot (m wb) (m xb)
          (fun t => flatten [outW, inW] [a % outW, t])
          (fun t => flatten [b, inW] [a / outW, t]) (inW / 32)
      else m b' a
  | .contractT _ wb db ob [b, inW] [outW] => fun m b' a =>
      if b' = ob ∧ a < b * inW then
        bflyFold (dotStridedLane (m wb) (m db)
          (fun i l => flatten [outW, inW] [i * 32 + l.val, a % inW])
          (fun i l => flatten [b, outW] [a / inW, i * 32 + l.val]) (outW / 32))
          ⟨0, by decide⟩
      else m b' a
  | .outerProd _ db xb ob [outW, inW] [b] => fun m b' a =>
      if b' = ob ∧ a / inW < outW ∧ a % inW < inW then
        dotStridedLane (m db) (m xb)
          (fun s _ => flatten [b, outW] [s, a / inW])
          (fun s _ => flatten [b, inW] [s, a % inW]) b (laneMod a)
      else m b' a
  | .pointwise1 f ab ob [g, 32] => fun m b' a =>
      if b' = ob ∧ a < g * 32 then denote (fun _ => m ab a) f else m b' a
  | .pointwise2 f ab bb ob [g, 32] => fun m b' a =>
      if b' = ob ∧ a < g * 32 then
        denote (fun v : Fin 2 => if v.val = 0 then m ab a else m bb a) f
      else m b' a
  | .softmaxCE lb bib ohb ob [g, 32] => fun m b' a =>
      if b' = ob ∧ a < g * 32 then
        smSpec (m lb) (m bib) (m ohb)
          (fun ln => flatten [g, 32] [a / 32, ln.val])
          (fun ln => flatten [1, 32] [0, ln.val]) (laneMod a)
      else m b' a
  | .update2 f ab bb [g, 32] => fun m b' a =>
      if b' = ab ∧ a < g * 32 then
        denote (fun v : Fin 2 => if v.val = 0 then m ab a else m bb a) f
      else m b' a
  | .rowPass ab bb ob f mA mB [rows, n] off w => fun m b' a =>
      if b' = ob ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalPair
          (m ab (flatten [rows, bcastCols w mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mA]) mA).apply
              [a / n, a % n - off])))
          (m bb (flatten [rows, bcastCols w mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mB]) mB).apply
              [a / n, a % n - off])))
      else m b' a
  | .rowPass3 ab bb cb ob f mA mB mC [rows, n] off w => fun m b' a =>
      if b' = ob ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalTriple
          (m ab (flatten [rows, bcastCols w mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mA]) mA).apply
              [a / n, a % n - off])))
          (m bb (flatten [rows, bcastCols w mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mB]) mB).apply
              [a / n, a % n - off])))
          (m cb (flatten [rows, bcastCols w mC]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mC]) mC).apply
              [a / n, a % n - off])))
      else m b' a
  | .rowPass4 ab bb cb db ob f mA mB mC mD [rows, n] off w => fun m b' a =>
      if b' = ob ∧ a / n < rows ∧ off ≤ a % n ∧ a % n < off + w then
        f.evalQuad
          (m ab (flatten [rows, bcastCols w mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mA]) mA).apply
              [a / n, a % n - off])))
          (m bb (flatten [rows, bcastCols w mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mB]) mB).apply
              [a / n, a % n - off])))
          (m cb (flatten [rows, bcastCols w mC]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mC]) mC).apply
              [a / n, a % n - off])))
          (m db (flatten [rows, bcastCols w mD]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols w mD]) mD).apply
              [a / n, a % n - off])))
      else m b' a
  | .rowReduce ab bb ob mA mB [rows] [n] => fun m b' a =>
      if b' = ob ∧ a < rows then
        bflyFold (dotStridedLane (m ab) (m bb)
          (fun t l => flatten [rows, bcastCols n mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mA]) mA).apply
              [a, t * 32 + l.val]))
          (fun t l => flatten [rows, bcastCols n mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mB]) mB).apply
              [a, t * 32 + l.val]))
          (n / 32)) ⟨0, by decide⟩
      else m b' a
  | .rowReduce4 ab bb cb db ob f mA mB mC mD [rows] [n] => fun m b' a =>
      if b' = ob ∧ a < rows then
        bflyFold (dotStridedLane4 (m ab) (m bb) (m cb) (m db)
          (fun t l => flatten [rows, bcastCols n mA]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mA]) mA).apply
              [a, t * 32 + l.val]))
          (fun t l => flatten [rows, bcastCols n mB]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mB]) mB).apply
              [a, t * 32 + l.val]))
          (fun t l => flatten [rows, bcastCols n mC]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mC]) mC).apply
              [a, t * 32 + l.val]))
          (fun t l => flatten [rows, bcastCols n mD]
            ((bcastReindex (P := [rows, n]) (Q := [rows, bcastCols n mD]) mD).apply
              [a, t * 32 + l.val]))
          (fun x y z => f.evalTriple x y z) (n / 32)) ⟨0, by decide⟩
      else m b' a
  | .rowMax ab ob [rows] [n] init => fun m b' a =>
      if b' = ob ∧ a < rows then
        bflyFoldOp (fun p q => NumOps.max p q)
          (maxStridedLane (m ab)
            (fun t l => flatten [rows, n] [a, t * 32 + l.val]) (n / 32) init)
          ⟨0, by decide⟩
      else m b' a
  | .rowSq ab ob [rows] [n] => fun m b' a =>
      if b' = ob ∧ a < rows then
        rowDot (m ab) (m ab)
          (fun t => flatten [rows, n] [a, t]) (fun t => flatten [rows, n] [a, t])
          (n / 32)
      else m b' a
  | _ => fun m => m

-- ---------------------------------------------------------------------------
-- The emitted operation computes what the algebra says
-- ---------------------------------------------------------------------------

/-- **Every operation the fragment lowers computes what the algebra says it
    does.**  One case per primitive, each closing on the addressing identities:
    `flatten [rows, cols] [row, col] = row * cols + col` for the direct walks,
    and `bcast_is_reindex` for every operand read through a broadcast. -/
theorem WOp.lower_den (w : WOp) (op : TOp) (h : WOp.lower w = some op)
    (m : Buf → Nat → Float32) : op.den m = w.den m := by
  match w, h with
  | .contract bk wb xb ob [b, outW] [inW] al, h =>
      cases h; funext b' a; simp only [TOp.den, WOp.den, flatten_pair]
  | .contractT bk wb db ob [b, inW] [outW], h =>
      cases h; funext b' a; simp only [TOp.den, WOp.den, flatten_pair]
  | .outerProd bk db xb ob [outW, inW] [b], h =>
      cases h; funext b' a; simp only [TOp.den, WOp.den, flatten_pair]
  | .pointwise1 f ab ob [g, 32], h => cases h; rfl
  | .pointwise2 f ab bb ob [g, 32], h => cases h; rfl
  | .softmaxCE lb bib ohb ob [g, 32], h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, flatten_pair, Nat.zero_mul, Nat.zero_add]
  | .update2 f ab bb [g, 32], h => cases h; rfl
  | .rowPass ab bb ob f mA mB [rows, n] off w, h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, bcast_is_reindex]
  | .rowPass3 ab bb cb ob f mA mB mC [rows, n] off w, h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, bcast_is_reindex]
  | .rowPass4 ab bb cb db ob f mA mB mC mD [rows, n] off w, h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, bcast_is_reindex]
  | .rowReduce ab bb ob mA mB [rows] [n], h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, rowdot_operand_is_reindex mA n rows a,
        rowdot_operand_is_reindex mB n rows a]
  | .rowReduce4 ab bb cb db ob f mA mB mC mD [rows] [n], h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, rowdot_operand_is_reindex mA n rows a,
        rowdot_operand_is_reindex mB n rows a, rowdot_operand_is_reindex mC n rows a,
        rowdot_operand_is_reindex mD n rows a]
  | .rowMax ab ob [rows] [n] init, h =>
      cases h; funext b' a
      simp only [TOp.den, WOp.den, flatten_pair]
      rfl
  | .rowSq ab ob [rows] [n], h =>
      cases h; funext b' a; simp only [TOp.den, WOp.den, flatten_pair]

-- ---------------------------------------------------------------------------
-- Reading a shipped tape back as an algebra program
-- ---------------------------------------------------------------------------

/-- **The abstraction**: the algebra operation a shipped one is an instance of.

    Every extent is repackaged as the shape it came from, so this is a total
    inverse of `lower` on the fragment — see `lower_ofTOp`. -/
def WOp.ofTOp : TOp → Option WOp
  | .mv bk wb xb ob b inW outW al  => some (.contract bk wb xb ob [b, outW] [inW] al)
  | .mvT bk wb db ob b inW outW    => some (.contractT bk wb db ob [b, inW] [outW])
  | .outer bk db xb ob b inW outW  => some (.outerProd bk db xb ob [outW, inW] [b])
  | .ew1 f ab ob g                 => some (.pointwise1 f ab ob [g, 32])
  | .ew2 f ab bb ob g              => some (.pointwise2 f ab bb ob [g, 32])
  | .smce lb bib ohb ob g          => some (.softmaxCE lb bib ohb ob [g, 32])
  | .upd2 f ab bb g                => some (.update2 f ab bb [g, 32])
  | .ziprow ab bb ob f mA mB n off w rows =>
      some (.rowPass ab bb ob f mA mB [rows, n] off w)
  | .ziprow3 ab bb cb ob f mA mB mC n off w rows =>
      some (.rowPass3 ab bb cb ob f mA mB mC [rows, n] off w)
  | .ziprow4 ab bb cb db ob f mA mB mC mD n off w rows =>
      some (.rowPass4 ab bb cb db ob f mA mB mC mD [rows, n] off w)
  | .rowdot ab bb ob mA mB n rows  => some (.rowReduce ab bb ob mA mB [rows] [n])
  | .rowdot4 ab bb cb db ob f mA mB mC mD n rows =>
      some (.rowReduce4 ab bb cb db ob f mA mB mC mD [rows] [n])
  | .rowmax ab ob n rows init      => some (.rowMax ab ob [rows] [n] init)
  | .rowsq ab ob n rows            => some (.rowSq ab ob [rows] [n])
  | _                              => none

/-- Abstracting an operation and lowering it again returns it unchanged. -/
theorem WOp.lower_ofTOp (op : TOp) (w : WOp) (h : WOp.ofTOp op = some w) :
    WOp.lower w = some op := by
  match op, h with
  | .mv _ _ _ _ _ _ _ _, h => cases h; rfl
  | .mvT _ _ _ _ _ _ _, h => cases h; rfl
  | .outer _ _ _ _ _ _ _, h => cases h; rfl
  | .ew1 _ _ _ _, h => cases h; rfl
  | .ew2 _ _ _ _ _, h => cases h; rfl
  | .smce _ _ _ _ _, h => cases h; rfl
  | .upd2 _ _ _ _, h => cases h; rfl
  | .ziprow _ _ _ _ _ _ _ _ _ _, h => cases h; rfl
  | .ziprow3 _ _ _ _ _ _ _ _ _ _ _ _, h => cases h; rfl
  | .ziprow4 _ _ _ _ _ _ _ _ _ _ _ _ _ _, h => cases h; rfl
  | .rowdot _ _ _ _ _ _ _, h => cases h; rfl
  | .rowdot4 _ _ _ _ _ _ _ _ _ _ _ _, h => cases h; rfl
  | .rowmax _ _ _ _ _, h => cases h; rfl
  | .rowsq _ _ _ _, h => cases h; rfl

/-- **A shipped operation computes what the algebra says it does.** -/
theorem WOp.ofTOp_den (op : TOp) (w : WOp) (h : WOp.ofTOp op = some w)
    (m : Buf → Nat → Float32) : op.den m = w.den m :=
  WOp.lower_den w op (WOp.lower_ofTOp op w h) m

-- ---------------------------------------------------------------------------
-- Whole tapes
-- ---------------------------------------------------------------------------

/-- Lower a whole program, refusing the whole of it if any operation is
    outside the fragment. -/
def WOp.lowerAll : List WOp → Option (List TOp)
  | []      => some []
  | w :: ws => match WOp.lower w, WOp.lowerAll ws with
    | some op, some ops => some (op :: ops)
    | _, _              => none

/-- Read a whole shipped tape back as an algebra program. -/
def WOp.ofTape : List TOp → Option (List WOp)
  | []        => some []
  | op :: ops => match WOp.ofTOp op, WOp.ofTape ops with
    | some w, some ws => some (w :: ws)
    | _, _            => none

open Classical in
/-- What a program computes: each operation in turn. -/
noncomputable def WOp.denAll (ws : List WOp) (m : Buf → Nat → Float32) :
    Buf → Nat → Float32 :=
  ws.foldl (fun mm w => w.den mm) m

/-- **The emitted tape computes the program's denotation.**

    This is the statement the existing pipeline plugs into: the left-hand side
    is `(tape).foldl TOp.den`, which is what `TenProg.run_den` and
    `TenProg.compile_den` are stated about, so a program written in the algebra
    inherits the whole chain down to the launch sequence. -/
theorem WOp.lowerAll_den (ws : List WOp) (ops : List TOp)
    (h : WOp.lowerAll ws = some ops) (m : Buf → Nat → Float32) :
    ops.foldl (fun mm o => o.den mm) m = WOp.denAll ws m := by
  induction ws generalizing ops m with
  | nil => cases h; rfl
  | cons w ws ih =>
      simp only [WOp.lowerAll] at h
      cases hw : WOp.lower w with
      | none => rw [hw] at h; simp at h
      | some op =>
          cases hws : WOp.lowerAll ws with
          | none => rw [hw, hws] at h; simp at h
          | some rest =>
              rw [hw, hws] at h
              simp only [Option.some.injEq] at h
              subst h
              show rest.foldl (fun mm o => o.den mm) (op.den m) = WOp.denAll (w :: ws) m
              rw [WOp.lower_den w op hw m]
              exact ih rest hws (w.den m)

/-- **Programs compose by concatenation.**

    Placing one program after another lowers to the two tapes end to end — so a
    model assembled from parts emits exactly the parts' operations, in order,
    and nothing about the parts is recomputed. -/
theorem WOp.lowerAll_append (ws₁ ws₂ : List WOp) (ops₁ ops₂ : List TOp)
    (h₁ : WOp.lowerAll ws₁ = some ops₁) (h₂ : WOp.lowerAll ws₂ = some ops₂) :
    WOp.lowerAll (ws₁ ++ ws₂) = some (ops₁ ++ ops₂) := by
  induction ws₁ generalizing ops₁ with
  | nil => cases h₁; simpa using h₂
  | cons w ws ih =>
      simp only [WOp.lowerAll] at h₁
      cases hw : WOp.lower w with
      | none => rw [hw] at h₁; simp at h₁
      | some op =>
          cases hws : WOp.lowerAll ws with
          | none => rw [hw, hws] at h₁; simp at h₁
          | some rest =>
              rw [hw, hws] at h₁
              simp only [Option.some.injEq] at h₁
              subst h₁
              show (match WOp.lower w, WOp.lowerAll (ws ++ ws₂) with
                    | some o, some os => some (o :: os)
                    | _, _ => none) = some (op :: (rest ++ ops₂))
              rw [hw, ih rest hws]

/-- **And their denotations compose.**

    `denAll` carries concatenation of programs to composition of the memory
    transformers they denote — the algebra's sequential composition, at the
    representation that compiles.  Together with `lowerAll_append` this is what
    makes a model assembled from blocks mean the composite of the blocks. -/
theorem WOp.denAll_append (ws₁ ws₂ : List WOp) (m : Buf → Nat → Float32) :
    WOp.denAll (ws₁ ++ ws₂) m = WOp.denAll ws₂ (WOp.denAll ws₁ m) := by
  simp only [WOp.denAll, List.foldl_append]

/-- The empty program is the identity, so composition has a unit. -/
theorem WOp.denAll_nil (m : Buf → Nat → Float32) : WOp.denAll [] m = m := rfl

/-- Abstracting a whole tape and lowering it again returns it unchanged. -/
theorem WOp.lowerAll_ofTape (ops : List TOp) (ws : List WOp)
    (h : WOp.ofTape ops = some ws) : WOp.lowerAll ws = some ops := by
  induction ops generalizing ws with
  | nil => cases h; rfl
  | cons op ops ih =>
      simp only [WOp.ofTape] at h
      cases ho : WOp.ofTOp op with
      | none => rw [ho] at h; simp at h
      | some w =>
          cases hos : WOp.ofTape ops with
          | none => rw [ho, hos] at h; simp at h
          | some rest =>
              rw [ho, hos] at h
              simp only [Option.some.injEq] at h
              subst h
              show (match WOp.lower w, WOp.lowerAll rest with
                    | some o, some os => some (o :: os)
                    | _, _ => none) = some (op :: ops)
              rw [WOp.lower_ofTOp op w ho, ih rest hos]

/-- **A shipped tape computes the denotation of the algebra program it is.**

    Stated over the tape, so it applies to a model that was never written in
    the algebra — which is what lets it be checked on ViT's 2294 operations
    without rewriting any of them. -/
theorem WOp.ofTape_den (ops : List TOp) (ws : List WOp) (h : WOp.ofTape ops = some ws)
    (m : Buf → Nat → Float32) :
    ops.foldl (fun mm o => o.den mm) m = WOp.denAll ws m :=
  WOp.lowerAll_den ws ops (WOp.lowerAll_ofTape ops ws h) m

end AlgorithmLib.ML.Broadcast
