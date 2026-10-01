module
public import AlgorithmLib.ML.Tensor.Ten
meta import AlgorithmLib.ML.Tensor.Ten
public import AlgorithmLib.ML.Math.Transformer
meta import AlgorithmLib.ML.Math.Transformer
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- The surface
-- ---------------------------------------------------------------------------

/-!
  `w * x` is a contraction: the shared width lives in both operand types, so
  the operator is total on the terms it accepts and a mismatch is a
  unification failure.  The elementwise functions keep their `Transformer`
  definitions rather than re-spelling them, so a model written here and the
  shipped kernels cannot drift.

  `tlet x := e; body` is the binder.  It elaborates to `letT`, so a value used
  twice is one buffer, and it is a macro rather than a monad — there is no
  bind to reduce through and no `←`.
-/

instance {v : Nat → Nat → Type} {b i o : Nat} :
    HMul (Ten v o i) (Ten v b i) (Ten v b o) := ⟨Ten.mv Backend.proven⟩

/-- The residual add and the gated feed-forward, as scalar specs. -/
def addW2 : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def gatedW2 : Expr 2 := .mul (Transformer.silu (.var ⟨0, by decide⟩)) (.var ⟨1, by decide⟩)

/-- `silu(x) = x·σ(x)` — `Transformer.silu`, the spec the shipped kernel is
    proven against. -/
def Ten.silu {v : Nat → Nat → Type} {r c : Nat} (t : Ten v r c) : Ten v r c :=
  .ew1 (Transformer.silu (.var ⟨0, by decide⟩)) t

/-- **RMSNorm** — `out[s][i] = γ[i]·x[s][i]·rsqrt(Σⱼ x[s][j]²/n + ε)`, as three
    row passes: the sum of squares, the scale, the gain.

    `x` is mentioned twice, so bind it with `tlet` first.  A bound variable
    flattens to no operation at all, which is what keeps the statistic and the
    scaled row reading one buffer rather than two computations of it. -/
def rmsScaleW (c : Nat) (eps : Float32) : WFExp :=
  .mul (.reg 1)
    (.rsqrt (.add (.mul (.reg 2) (.lit (NumOps.inv (NumOps.ofNat c)))) (.lit eps)))

def Ten.rmsNorm {v : Nat → Nat → Type} {r c : Nat} (eps : Float32)
    (gamma : Ten v 1 c) (x : Ten v r c) (nm : String := "norm") : Ten v r c :=
  .zipB mulW (.named (nm ++ ".scale") (.zipS (rmsScaleW c eps) x x.rowsq)) gamma

/-- **The same norm as two passes instead of three** — the scale and the gain
    in one kernel.  `fuse_ziprow_den` is what says the number is the same one,
    bit for bit: `WFExp.fuseA` keeps every operation in its original order, so
    this is a schedule that needs no law. -/
def Ten.rmsNormFused {v : Nat → Nat → Type} {r c : Nat} (eps : Float32)
    (gamma : Ten v 1 c) (x : Ten v r c) (nm : String := "norm") : Ten v r c :=
  .named (nm ++ ".fused") (.zip3 (WFExp.fuseA mulW (rmsScaleW c eps)) x x.rowsq gamma)

/-- A seed for a row maximum: below any logit a model produces. -/
def softmaxFloor : Float32 := NumOps.neg (NumOps.ofNat 1000000000)

/-- **Row-wise softmax** — the row maximum, the shift, the exponential, the row
    sum against a shared vector of ones, and the scale.  Five passes, all
    proven stages.

    Subtracting the maximum is what keeps the exponential in range, and it is
    the same shift `Transformer.softmaxAt` takes as a parameter — the kernel
    now computes it rather than being handed one. -/
def Ten.softmaxRow {v : Nat → Nat → Type} {r c : Nat}
    (ones : Ten v 1 c) (t : Ten v r c) : Ten v r c :=
  .letT t (fun tb =>
    .letT (Ten.rowMax softmaxFloor (.var tb)) (fun mx =>
      .letT (.zipS (.add (.reg 1) (.neg (.reg 2))) (.var tb) (.var mx)) (fun sh =>
        .letT (.ew1 (.exp (.var ⟨0, by decide⟩)) (.var sh)) (fun e =>
          .zipS (.mul (.reg 1) (.inv (.reg 2))) (.var e)
            (Ten.rowB (.var e) ones)))))

/-- **One attention head at a decode step** — scores, softmax, weighted sum.

    Heads are rows.  At a single position the whole head is two matrix-vector
    products against the cache and a row-wise softmax between them, so it needs
    no index beyond `row × column`: `q · Kᵀ` is `mv`, `p · V` is `mvT`. -/
def Ten.attnHead {v : Nat → Nat → Type} {hd sq nh : Nat} (impl : Backend)
    (ones : Ten v 1 sq) (kc vc : Ten v sq hd) (q : Ten v nh hd) : Ten v nh hd :=
  .letT (.ew1 (.mul (.var ⟨0, by decide⟩) (.rsqrt (.lit hd))) (.mv impl kc q)) (fun sc =>
    .letT (Ten.softmaxRow ones (.var sc)) (fun p =>
      .mvT impl vc (.var p)))

/-- One expert: a gated feed-forward. -/
def Ten.expert {v : Nat → Nat → Type} {dm dff : Nat} (impl : Backend)
    (w1 w3 : Ten v dff dm) (w2 : Ten v dm dff) (x : Ten v 1 dm) : Ten v 1 dm :=
  .mv impl w2 (.ew2 gatedW2 (.mv impl w1 x) (.mv impl w3 x))

/-- Experts `0…k`, each scaled by its own gate and added. -/
def Ten.moeAcc {v : Nat → Nat → Type} {dm dff nE : Nat} (impl : Backend)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff) (g : Ten v 1 nE)
    (x : Ten v 1 dm) : Nat → Ten v 1 dm
  | 0     => .zipC mulW 0 (Ten.expert impl (w 0).1 (w 0).2.1 (w 0).2.2 x) g
  | k + 1 => .ew2 addW2 (Ten.moeAcc impl w g x k)
               (.zipC mulW (k + 1)
                 (Ten.expert impl (w (k+1)).1 (w (k+1)).2.1 (w (k+1)).2.2 x) g)

/-- **The expert slots a launch sequence has, with the gates the host chose.**

    `moe` below evaluates every expert because one graph is one launch
    sequence.  This is the other half of the same vocabulary: `nUsed` slots,
    each reading the weights bound to it and scaled by element `j` of a gate
    row the host packed.  Which expert a slot *is* appears nowhere in the term
    — it is which buffers the bind array points slot `j` at, so choosing
    experts costs a rebinding and no new kernel.

    The gate row is an operand rather than the router's own output for the same
    reason: slot `j` reads gate `j`, and the router scored expert `e`, so
    something has to compact `nE` scores into `nUsed`.  The host does it with
    the read it already makes to rank them. -/
def Ten.moeSlots {v : Nat → Nat → Type} {dm dff nUsed : Nat} (impl : Backend)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff)
    (g : Ten v 1 nUsed) (x : Ten v 1 dm) : Ten v 1 dm :=
  .letT x (fun xb => Ten.moeAcc impl w g (.var xb) (nUsed - 1))

/-- **The router on its own** — a projection and a row-wise softmax.

    Separate from the experts because the host runs it first, reads the row it
    writes, and only then knows which weights to bind. -/
def Ten.router {v : Nat → Nat → Type} {dm nE : Nat} (impl : Backend)
    (ones : Ten v 1 nE) (wr : Ten v nE dm) (x : Ten v 1 dm) : Ten v 1 nE :=
  Ten.softmaxRow ones (.mv impl wr x)

/-- **A mixture-of-experts feed-forward.**  The router is a projection and a
    row-wise softmax; each expert is a gated feed-forward; the gate for expert
    `e` is one element of the router row, read at a constant address.

    Every expert is evaluated.  That is the *dense* reading of the mixture — it
    computes the right answer and none of the saving, because choosing which
    experts to run is a host decision about which kernels to launch, and this
    graph describes one launch sequence. Sparsity is a rebinding, not a term. -/
def Ten.moe {v : Nat → Nat → Type} {dm dff nE : Nat} (impl : Backend)
    (ones : Ten v 1 nE) (wr : Ten v nE dm)
    (w : Nat → Ten v dff dm × Ten v dff dm × Ten v dm dff) (nUsed : Nat)
    (x : Ten v 1 dm) : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (Ten.softmaxRow ones (.mv impl wr (.var xb))) (fun g =>
      Ten.moeAcc impl w (.var g) (.var xb) nUsed))

/-- **A pre-norm transformer block**, at one decode position.

    Norm, projections, rotation, attention against the cache, the output
    projection and the residual; then norm, a gated feed-forward and the second
    residual.  Every step is a `Ten` constructor that lowers to a proven stage,
    so a block costs no new proof text — only the two `view`s that say the
    residual stream and the head layout are the same bytes.

    The cache is read, not appended to: an append is a host-side write this
    model does not describe. -/
def Ten.block {v : Nat → Nat → Type} {dm nh hd sq dff : Nat} (impl : Backend)
    (hv : 1 * dm = nh * hd) (hv' : nh * hd = 1 * dm) (eps : Float32)
    (g1 g2 : Ten v 1 dm) (wq wo : Ten v dm dm) (w1 w3 : Ten v dff dm)
    (w2 : Ten v dm dff) (cosT sinT : Ten v 1 hd) (ones : Ten v 1 sq)
    (kc vc : Ten v sq hd) (x : Ten v 1 dm) (nm : String := "block") : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (.named (nm ++ ".n1") (Ten.rmsNorm eps g1 (.var xb) (nm ++ ".attn"))) (fun n1 =>
      .letT (.named (nm ++ ".q") (.rope (.view hv (.mv impl wq (.var n1))) cosT sinT)) (fun q =>
        .letT (.named (nm ++ ".att") (.mv impl wo
                (.view hv' (Ten.attnHead impl ones kc vc (.var q))))) (fun att =>
          .letT (.named (nm ++ ".xr") (.ew2 addW2 (.var xb) (.var att))) (fun xr =>
            .letT (.named (nm ++ ".n2") (Ten.rmsNorm eps g2 (.var xr) (nm ++ ".ffn"))) (fun n2 =>
              .letT (.named (nm ++ ".ff")
                      (.ew2 gatedW2 (.mv impl w1 (.var n2)) (.mv impl w3 (.var n2))))
                (fun ff => .ew2 addW2 (.var xr) (.mv impl w2 (.var ff)))))))))

/-- **A block's shape, with the tiling condition carried rather than written.**

    A rotation needs the residual stream and the head layout to be the same
    bytes, which is an arithmetic fact about the widths — `1 * dm = nh * hd`.
    Making it a field with a tactic default means a model states its widths and
    nothing else: the condition is discharged where the widths are given, and a
    geometry that does not tile fails there with the false proposition printed,
    rather than at the use site. -/
structure BlockGeom where
  dm : Nat
  nh : Nat
  hd : Nat
  sq : Nat
  dff : Nat
  tiles : 1 * dm = nh * hd := by decide

/-- **A block at a geometry** — the same term as `Ten.block`, with the two
    proofs taken from the geometry and the implementation defaulted.

    This is the surface a model is written against: widths, weights, and the
    residual stream.  `impl` is trailing and defaulted because it is a schedule
    choice, not part of what the block computes; a model that never mentions it
    gets the proven kernels. -/
def Ten.blockAt {v : Nat → Nat → Type} (G : BlockGeom) (eps : Float32)
    (g1 g2 : Ten v 1 G.dm) (wq wo : Ten v G.dm G.dm) (w1 w3 : Ten v G.dff G.dm)
    (w2 : Ten v G.dm G.dff) (cosT sinT : Ten v 1 G.hd) (ones : Ten v 1 G.sq)
    (kc vc : Ten v G.sq G.hd) (x : Ten v 1 G.dm)
    (impl : Backend := .proven) (nm : String := "block") : Ten v 1 G.dm :=
  Ten.block impl G.tiles G.tiles.symm eps g1 g2 wq wo w1 w3 w2 cosT sinT ones kc vc x nm

/-- **The same block, scheduled with both norms fused.**

    Identical structure, two kernels shorter.  `fuse_ziprow_den` is what says it
    is the same block rather than a different one. -/
def Ten.blockFused {v : Nat → Nat → Type} {dm nh hd sq dff : Nat} (impl : Backend)
    (hv : 1 * dm = nh * hd) (hv' : nh * hd = 1 * dm) (eps : Float32)
    (g1 g2 : Ten v 1 dm) (wq wo : Ten v dm dm) (w1 w3 : Ten v dff dm)
    (w2 : Ten v dm dff) (cosT sinT : Ten v 1 hd) (ones : Ten v 1 sq)
    (kc vc : Ten v sq hd) (x : Ten v 1 dm) (nm : String := "block") : Ten v 1 dm :=
  .letT x (fun xb =>
    .letT (Ten.rmsNormFused eps g1 (.var xb) (nm ++ ".attn")) (fun n1 =>
      .letT (.rope (.view hv (.mv impl wq (.var n1))) cosT sinT) (fun q =>
        .letT (.mv impl wo
                (.view hv' (Ten.attnHead impl ones kc vc (.var q)))) (fun att =>
          .letT (.ew2 addW2 (.var xb) (.var att)) (fun xr =>
            .letT (Ten.rmsNormFused eps g2 (.var xr) (nm ++ ".ffn")) (fun n2 =>
              .letT (.ew2 gatedW2 (.mv impl w1 (.var n2)) (.mv impl w3 (.var n2)))
                (fun ff => .ew2 addW2 (.var xr) (.mv impl w2 (.var ff)))))))))

/-- An elementwise combination of two tensors of the same extent. -/
def Ten.zipWith {v : Nat → Nat → Type} {r c : Nat} (f : Expr 2)
    (a b : Ten v r c) : Ten v r c := .ew2 f a b

/-- Send one operation to a vendor implementation, leaving what it computes
    alone.  The tag a schedule adds to an otherwise unchanged equation. -/
def Ten.on {v : Nat → Nat → Type} {r c : Nat} (impl : Backend) :
    Ten v r c → Ten v r c
  | .mv _ w x    => .mv impl w x
  | .mvAt bA _ w x => .mvAt bA impl w x
  | .mvT _ w d   => .mvT impl w d
  | .outer _ d x => .outer impl d x
  | t            => t

/-- The forward spec of a unary elementwise pass, from an ordinary Lean
    function on expressions. -/
def ewFwd (f : Expr 1 → Expr 1) : Expr 1 := f (.var ⟨0, by decide⟩)

/-- **The backward spec of a unary elementwise pass, derived from the same
    Lean function.**  `∂L/∂in = ∂L/∂out · f'(in)`, with `f'` taken
    symbolically by `sderiv` — input 0 is the primal, input 1 the incoming
    cotangent.

    Writing the activation once and reading both passes off it is the property
    that makes training and inference one source rather than two hand-written
    graphs. -/
def ewBack (f : Expr 2 → Expr 2) : Expr 2 :=
  .mul (.var ⟨1, by decide⟩) (sderiv (f (.var ⟨0, by decide⟩)) ⟨0, by decide⟩)

/-- Weaken a unary spec into the binary context a backward pass works in. -/
def wk1 (e : Expr 1) : Expr 2 := rename (Fin.castLE (by decide)) e

/-- The backward spec of a unary elementwise op, taken from its *forward*
    spec — the term-level counterpart of `ewBack`. -/
def ewBackOf (f : Expr 1) : Expr 2 :=
  .mul (.var ⟨1, by decide⟩) (sderiv (wk1 f) ⟨0, by decide⟩)

/-- Weaken a binary spec into the ternary context its adjoints work in. -/
def wk2 (e : Expr 2) : Expr 3 := rename (Fin.castLE (by decide)) e

/-- The adjoint of a *binary* elementwise op with respect to input `i`:
    `∂L/∂inᵢ = ∂L/∂out · ∂f/∂inᵢ`.  Variables 0 and 1 are the primal inputs,
    variable 2 the incoming cotangent. -/
def ewBack2 (f : Expr 2) (i : Fin 2) : Expr 3 :=
  .mul (.var ⟨2, by decide⟩) (sderiv (wk2 f) i.castSucc)

/-- Weaken a ternary spec into the quaternary context its adjoints work in. -/
def wk3 (e : Expr 3) : Expr 4 := rename (Fin.castLE (by decide)) e

/-- The adjoint of a *ternary* elementwise op — a gated feed-forward is the
    standard case — with respect to input `i`.  Variables 0–2 are the primal
    inputs, variable 3 the incoming cotangent. -/
def ewBack3 (f : Expr 3) (i : Fin 3) : Expr 4 :=
  .mul (.var ⟨3, by decide⟩) (sderiv (wk3 f) i.castSucc)

/-- Fold the constants out of a lane expression.

    `WFExp.deriv` is syntactic: the derivative of `a + b` with respect to `a` is
    `1 + 0`, not `1`.  This is what makes those literals visible. -/
def WFExp.fold : WFExp → WFExp
  | .add a b =>
      match a.fold, b.fold with
      | .lit x, .lit y => .lit (x + y)
      | .lit x, e      => if x == 0 then e else .add (.lit x) e
      | e, .lit y      => if y == 0 then e else .add e (.lit y)
      | x, y           => .add x y
  | .mul a b =>
      match a.fold, b.fold with
      | .lit x, .lit y => .lit (x * y)
      | .lit x, e      => if x == 1 then e else if x == 0 then .lit 0 else .mul (.lit x) e
      | e, .lit y      => if y == 1 then e else if y == 0 then .lit 0 else .mul e (.lit y)
      | x, y           => .mul x y
  | .neg a => match a.fold with | .lit x => .lit (-x) | e => .neg e
  | e => e

/-- **The adjoint is the incoming gradient itself.**

    An addition's derivative is one, so its adjoint pass computes `d · 1` — a
    kernel that copies a tensor.  Where this holds the contribution *is* `d`,
    and the cotangent map can be given `d` directly. -/
def WFExp.isReg (f : WFExp) (r : Nat) : Bool :=
  match f.fold with
  | .reg k => k == r
  | _      => false

/-- Fold the constants out of a scalar spec, for the same reason `WFExp.fold`
    exists: `sderiv` of `a + b` is `1 + 0`, not `1`. -/
def Expr.fold : {Γ : Nat} → Expr Γ → Expr Γ
  | _, .add a b =>
      match a.fold, b.fold with
      | .lit x, .lit y => .lit (x + y)
      | .lit 0, e      => e
      | e, .lit 0      => e
      | x, y           => .add x y
  | _, .mul a b =>
      match a.fold, b.fold with
      | .lit x, .lit y => .lit (x * y)
      | .lit 1, e      => e
      | e, .lit 1      => e
      | .lit 0, _      => .lit 0
      | _, .lit 0      => .lit 0
      | x, y           => .mul x y
  | _, e => e

/-- **The adjoint is the incoming cotangent itself**, as `WFExp.isReg` is for a
    lane expression: an addition's adjoint is `d · 1`, a kernel that copies. -/
def Expr.isVar {Γ : Nat} (e : Expr Γ) (i : Fin Γ) : Bool :=
  match e.fold with
  | .var j => j == i
  | _      => false

/-- `a + b`, the cotangent accumulator. -/
def addSpec : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

/-- Which buffer holds `∂L/∂b`, per forward buffer `b`. -/
abbrev CoT := List (Ref × Ref)

def CoT.get (m : CoT) (r : Ref) : Option Ref :=
  match m.find? (fun p => p.1 == r) with
  | some p => some p.2
  | none   => none

/-- **Record a cotangent contribution.**  A value used more than once — a
    residual stream is the standard case — receives one contribution per use,
    and reverse mode is only correct if they are *summed*.  The first
    contribution binds; every later one emits an add. -/
def CoT.accum (m : CoT) (r d : Ref) (fresh grid : Nat) : List TOp × CoT × Nat :=
  match m.get r with
  | none      => ([], (r, d) :: m, fresh)
  | some prev => ([.ew2 addSpec prev d fresh grid], (r, fresh) :: m, fresh + 1)

/-- **The backward operations of one forward operation.**

    A contraction contributes the weight gradient always and the input
    gradient only where one is wanted, so a model does not compute a gradient
    with respect to its own input.  `smce` emits nothing: it *is* the seed —
    the buffer it already writes is the cotangent of its logits. -/
def TOp.grad (elideIdent : Bool) (needs : Ref → Bool) (ones : Ref)
    (batch fresh : Nat) (ct : CoT) : TOp → Option (List TOp × CoT × Nat)
  | .mv bk w x out bb inW outW _ =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addW, ct1, f1) := CoT.accum ct w fresh (fresh + 1) (outW * inW / 32)
          if needs x then
            let (addX, ct2, f2) := CoT.accum ct1 x f1 (f1 + 1) (batch * inW / 32)
            some ([.outer bk d x fresh bb inW outW] ++ addW
                    ++ [.mvT bk w d f1 bb inW outW] ++ addX, ct2, f2)
          else
            some ([.outer bk d x fresh bb inW outW] ++ addW, ct1, f1)
  | .ew1 f a out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) grid
          some ([.ew2 (ewBackOf f) a d fresh grid] ++ addA, ct1, f1)
  | .ew2 f a b out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          -- An input the operation merely adds gets `d` itself, not a copy of
          -- it: the cotangent is in variable 2 of the adjoint spec.
          let fA := ewBack2 f ⟨0, by decide⟩
          let fB := ewBack2 f ⟨1, by decide⟩
          let idA := elideIdent && fA.isVar ⟨2, by decide⟩
          let idB := elideIdent && fB.isVar ⟨2, by decide⟩
          let opsA := if idA then [] else [TOp.ew3 fA a b d fresh grid]
          let (addA, ct1, f1) :=
            if idA then CoT.accum ct a d fresh grid
            else CoT.accum ct a fresh (fresh + 1) grid
          let opsB := if idB then [] else [TOp.ew3 fB a b d f1 grid]
          let (addB, ct2, f2) :=
            if idB then CoT.accum ct1 b d f1 grid
            else CoT.accum ct1 b f1 (f1 + 1) grid
          some (opsA ++ addA ++ opsB ++ addB, ct2, f2)
  | .ew3 f a b c out grid =>
      match ct.get out with
      | none   => none
      | some d =>
          let (addA, ct1, f1) := CoT.accum ct a fresh (fresh + 1) grid
          let (addB, ct2, f2) := CoT.accum ct1 b f1 (f1 + 1) grid
          let (addC, ct3, f3) := CoT.accum ct2 c f2 (f2 + 1) grid
          some ([.ew4 (ewBack3 f ⟨0, by decide⟩) a b c d fresh grid] ++ addA
                  ++ [.ew4 (ewBack3 f ⟨1, by decide⟩) a b c d f1 grid] ++ addB
                  ++ [.ew4 (ewBack3 f ⟨2, by decide⟩) a b c d f2 grid] ++ addC,
                ct3, f3)
  | .smce l _ _ out _ => some ([], (l, out) :: ct, fresh)
  | .rowsq x out n rows =>
      match ct.get out with
      | none => none
      | some ds =>
          let (addX, ct1, f1) := CoT.accum ct x fresh (fresh + 1) (rows * n / 32)
          some ([.ziprow x ds fresh (.mul (.add (.reg 1) (.reg 1)) (.reg 2))
                  (.rowOf n 0) .scalar n 0 n rows] ++ addX, ct1, f1)
  | .mvT bk w d out b inW outW =>
      match ct.get out with
      | none => none
      | some g =>
          -- `dW[o][i] = Σₛ d[s][o]·g[s][i]`, `dd[s][o] = Σᵢ W[o][i]·g[s][i]`
          let (addW, ct1, f1) := CoT.accum ct w fresh (fresh + 1) (outW * inW / 32)
          if needs d then
            let (addD, ct2, f2) := CoT.accum ct1 d f1 (f1 + 1) (b * outW / 32)
            some ([.outer bk d g fresh b inW outW] ++ addW
                    ++ [.mv bk w g f1 b inW outW b] ++ addD, ct2, f2)
          else
            some ([.outer bk d g fresh b inW outW] ++ addW, ct1, f1)
  | .rowdot i j out mA mB n rows =>
      match ct.get out with
      | none => none
      | some ds =>
          match mA with
          | .rowOf nA kA =>
              let opA := TOp.ziprow j ds fresh mulW mB .scalar nA kA n rows
              let (addA, ct1, f1) := CoT.accum ct i fresh (fresh + 1) (rows * nA / 32)
              if needs j then
                match mB with
                | .rowOf nB kB =>
                    let (addB, ct2, f2) := CoT.accum ct1 j f1 (f1 + 1) (rows * nB / 32)
                    some ([opA] ++ addA
                            ++ [TOp.ziprow i ds f1 mulW mA .scalar nB kB n rows]
                            ++ addB, ct2, f2)
                | _ => none
              else some ([opA] ++ addA, ct1, f1)
          | _ => none
  | .rowmax x out n rows _ =>
      match ct.get out with
      | none => none
      | some ds =>
          -- the gradient goes where the maximum was attained; `geF` is that
          -- indicator as a value, which is why no predicate is needed
          let sel : WFExp := .mul (.reg 3) (.geF (.reg 1) (.reg 2))
          let (addX, ct1, f1) := CoT.accum ct x fresh (fresh + 1) (rows * n / 32)
          some ([.ziprow3 x out ds fresh sel (.rowOf n 0) .scalar .scalar n 0 n rows]
                  ++ addX, ct1, f1)
  | .ziprow a b out f mA mB n off w rows =>
      match ct.get out with
      | none => none
      | some d =>
          match mA, adjW f 1 with
          | .rowOf nA kA, some fa =>
              -- When the adjoint is the incoming gradient and lands exactly
              -- where `d` already sits, the pass would copy a tensor: give the
              -- cotangent map `d` and emit nothing.
              let identA := elideIdent && fa.isReg 3 && nA == n && kA == off
              let opsA := if identA then [] else
                [TOp.ziprow3 a b d fresh fa mA mB (.rowOf n off) nA kA w rows]
              let (addA, ct1, f1) :=
                if identA then CoT.accum ct a d fresh (rows * nA / 32)
                else CoT.accum ct a fresh (fresh + 1) (rows * nA / 32)
              if needs b then
                match adjW f 2 with
                | none => none
                | some fb =>
                    match mB with
                    | .rowOf nB kB =>
                        let (addB, ct2, f2) := CoT.accum ct1 b f1 (f1 + 1) (rows * nB / 32)
                        some (opsA ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) nB kB w rows]
                                ++ addB, ct2, f2)
                    | .scalar =>
                        -- one value per row: the products, then a row sum
                        let (addB, ct2, f2) :=
                          CoT.accum ct1 b (f1 + 1) (f1 + 2) ((rows + 31) / 32)
                        some (opsA ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) w 0 w rows,
                                    TOp.rowdot f1 ones (f1 + 1)
                                      (.rowOf w 0) (.sharedAt 0) w rows]
                                ++ addB, ct2, f2)
                    | .sharedAt 0 =>
                        -- one vector shared by every row: the products, then a
                        -- column sum, which is an outer product against ones
                        let (addB, ct2, f2) :=
                          CoT.accum ct1 b (f1 + 1) (f1 + 2) (w / 32)
                        some (opsA ++ addA
                                ++ [TOp.ziprow3 a b d f1 fb mA mB
                                      (.rowOf n off) w 0 w rows,
                                    TOp.outer Backend.proven ones f1 (f1 + 1) rows w 1]
                                ++ addB, ct2, f2)
                    | _ => none
              else some (opsA ++ addA, ct1, f1)
          | _, _ => none
  | _ => none

def gradRev (elideIdent : Bool) (needs : Ref → Bool) (ones : Ref) (batch : Nat) :
    List TOp → Nat → CoT → Option (List TOp × CoT × Nat)
  | [],         fresh, ct => some ([], ct, fresh)
  | op :: rest, fresh, ct =>
      match op.grad elideIdent needs ones batch fresh ct with
      | none => none
      | some (bs, ct', f1) =>
          match gradRev elideIdent needs ones batch rest f1 ct' with
          | none => none
          | some (rs, ct'', f2) => some (bs ++ rs, ct'', f2)

/-- **The backward pass of a forward program, derived.**

    Reverse-mode: the forward tape walked backwards, each operation
    contributing its own adjoint.  Every buffer it allocates is fresh, and the
    activation adjoints come from `sderiv` — so the backward is a function of
    the forward, not a second program written alongside it.

    An operation the reverse pass has no rule for, or one whose output has no
    cotangent, yields `none` rather than an empty contribution — a missing
    gradient is a build failure, not a silently untrained parameter. -/
def Ten.backward (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (ops : List TOp) (elideIdent : Bool := false) : Option (List TOp) :=
  (gradRev elideIdent needs ones batch ops.reverse fresh []).map Prod.fst

/-- The same, from a cotangent already known — a slice of a tape whose loss is
    computed elsewhere, or a block differentiated on its own. -/
def Ten.backwardFrom (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (seed : CoT) (ops : List TOp) (elideIdent : Bool := false) : Option (List TOp) :=
  (gradRev elideIdent needs ones batch ops.reverse fresh seed).map Prod.fst

/-- **Where each gradient lands.**  The reverse pass allocates as it goes, so
    which buffer holds `∂L/∂w` is a fact about the derivation, not a convention
    a host can assume. -/
def Ten.backwardCoT (needs : Ref → Bool) (ones : Ref) (batch fresh : Nat)
    (seed : CoT) (ops : List TOp) (elideIdent : Bool := false) : Option CoT :=
  (gradRev elideIdent needs ones batch ops.reverse fresh seed).map (fun r => r.2.1)

/-- `tlet x := e; body` — the tensor binder. -/
syntax "tlet " ident " := " term "; " ppLine term : term

macro_rules
  | `(tlet $x := $e; $b) =>
      `(Ten.letT (Ten.named $(Lean.quote x.getId.toString) $e)
          fun w => let $x := Ten.var w; $b)

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
