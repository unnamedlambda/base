import AlgorithmLib
open AlgorithmLib AlgorithmLib.ML

-- The twelve-block lowering is one `rfl` over the whole tape, which nests
-- deeper than the default allows.
set_option maxRecDepth 4000

/-! DeiT-Tiny's twelve blocks as one `Ten` term, at the padded geometry. -/
namespace Vit

def SQ  : Nat := 224     -- 1 class token + 196 patches + 27 padding, 32-aligned
def DM  : Nat := 192
def NH  : Nat := 3
def HD  : Nat := 64
def DFF : Nat := 768
def NC  : Nat := 128     -- 102 classes, padded to a multiple of 32
def EPS : Float32 := 0.000001

/-- Per-layer parameter block: 30 buffers, laid out in this order. -/
def pBase (i : Nat) : Nat := 20 + 30 * i

def addW : WFExp := .add (.reg 1) (.reg 2)
def mulW : WFExp := .mul (.reg 1) (.reg 2)
def add2 : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

/-- `(x - mean)·rsqrt(var + eps)·gamma + beta`, from row passes only. -/
def layerNorm {v : Nat → Nat → Type} {r c : Nat}
    (gamma beta onesN : Ten v 1 c) (x : Ten v r c) : Ten v r c :=
  -- `x` is read twice below, so it must be bound: a `Ten` is a tree, and an
  -- unbound repeat re-emits the whole subterm rather than reusing its buffer.
  .letT x (fun xv =>
  .letT (.zipS (.add (.reg 1) (.neg (.reg 2))) (.var xv) (.rowB (.var xv) onesN)) (fun ctr =>
    .letT (.ew1 (.mul (.var ⟨0, by decide⟩) (.var ⟨0, by decide⟩)) (.var ctr)) (fun sq =>
      let nrm : Ten v r c :=
        .zipS (.mul (.reg 1) (.rsqrt (.add (.reg 2) (.lit EPS))))
          (.var ctr) (.rowB (.var sq) onesN)
      .zipB addW (.zipB mulW nrm gamma) beta)))

/-- tanh-GELU, `0.5x(1+tanh(c(x+0.044715x³)))`, as a row pass: `Expr` literals
    are `Nat` and these constants are not.  A declared approximation to the
    exact erf form, measured at 3.4e-3 relative on DeiT-Tiny's logits. -/
def geluW : WFExp :=
  let x := WFExp.reg 1
  let inner := WFExp.mul (.lit (0.7978845608028654 : Float32))
                 (.add x (.mul (.lit (0.044715 : Float32)) (.mul x (.mul x x))))
  let e2z := WFExp.exp (.add inner inner)
  let th := WFExp.mul (.add e2z (.neg (.lit (1 : Float32))))
              (.inv (.add e2z (.lit (1 : Float32))))
  .mul (.mul (.lit (0.5 : Float32)) x) (.add (.lit (1 : Float32)) th)

/-- One head, with the padded keys masked off before the softmax. -/
def head {v : Nat → Nat → Type} (onesSQ mask : Ten v 1 SQ)
    (wq wk wv : Ten v HD DM) (bq bk bv : Ten v 1 HD)
    (n1 : Ten v SQ DM) : Ten v SQ HD :=
  let proj := fun (w : Ten v HD DM) (b : Ten v 1 HD) =>
    Ten.zipB addW (.mv Backend.proven w n1) b
  .letT (.zipB addW
          (.ew1 (.mul (.var ⟨0, by decide⟩) (.rsqrt (.lit HD)))
            (.mv Backend.proven (proj wk bk) (proj wq bq))) mask) (fun sc =>
    .mvT Backend.proven (proj wv bv) (Ten.softmaxRow onesSQ (.var sc)))

def block {v : Nat → Nat → Type} (i : Nat)
    (onesN : Ten v 1 DM) (onesF : Ten v 1 DFF) (onesSQ mask : Ten v 1 SQ)
    (x : Ten v SQ DM) : Ten v SQ DM :=
  let p := pBase i
  .letT x (fun xb =>
    .letT (layerNorm (.inp p) (.inp (p+1)) onesN (.var xb)) (fun n1 =>
      let hd := fun h : Nat =>
        head onesSQ mask (.inp (p+2+h)) (.inp (p+8+h)) (.inp (p+14+h))
          (.inp (p+5+h)) (.inp (p+11+h)) (.inp (p+17+h)) (.var n1)
      -- heads 1.. are summed onto head 0; `List.range NH` here would count
      -- head 0 twice, and a `Ten` is a tree, so the duplicate re-emits.
      let att := ((List.range NH).drop 1).foldl
        (fun acc h => .ew2 add2 acc (.mv Backend.proven (.inp (p+20+h)) (hd h)))
        (.mv Backend.proven (.inp (p+20)) (hd 0))
      .letT (.ew2 add2 (.var xb) (.zipB addW att (.inp (p+23)))) (fun xr =>
        .letT (layerNorm (.inp (p+24)) (.inp (p+25)) onesN (.var xr)) (fun n2 =>
          .ew2 add2 (.var xr)
            (.zipB addW
              (.mv Backend.proven (.inp (p+28))
                (.zipB geluW (.zipB addW (.mv Backend.proven (.inp (p+26)) (.var n2))
                                (.inp (p+27))) onesF))
              (.inp (p+29)))))))

/-- Twelve blocks, a final LayerNorm, and the classifier on the class token. -/
def model (n : Nat) : TenProg SQ NC := fun v =>
  let onesN : Ten v 1 DM := .inp 0
  let onesF : Ten v 1 DFF := .inp 1
  let onesSQ : Ten v 1 SQ := .inp 2
  let mask : Ten v 1 SQ := .inp 3
  let toks : Ten v SQ DM := .inp 4
  let x := (List.range n).foldl (fun a i => block i onesN onesF onesSQ mask a) toks
  .zipB addW (.mv Backend.proven (.inp 6) (layerNorm (.inp 5) (.inp 7) onesN x)) (.inp 8)

end Vit

#eval do
  IO.println s!"[vit] ops, 1 block  : {(Vit.model 1).graph 1000 |>.length}"
  IO.println s!"[vit] ops, 12 blocks: {(Vit.model 12).graph 1000 |>.length}"

theorem vit1_lowers  : ((Vit.model 1).stages Vit.SQ 1000).isSome = true := by rfl

theorem vit12_lowers : ((Vit.model 12).stages Vit.SQ 1000).isSome = true := by rfl

#eval do
  for n in [1, 2] do
    let ops : List AlgorithmLib.ML.TOp := (Vit.model n).tape 1000
    let ptx := ops.map (fun (op : AlgorithmLib.ML.TOp) =>
      AlgorithmLib.ML.emitProvenKernelN "main" (op.bufs Vit.SQ).length 0
        (op.localStmt Vit.SQ))
    let mut seen : Std.HashSet String := {}
    for t in ptx do seen := seen.insert t
    IO.println s!"[vit] {n} block(s): {ops.length} ops, {seen.size} distinct kernels"

/-! Where the operations go, piece by piece. -/
def lnOnly : TenProg Vit.SQ Vit.DM := fun _ =>
  Vit.layerNorm (.inp 0) (.inp 1) (.inp 2) (.inp 3)
def headOnly : TenProg Vit.SQ Vit.HD := fun _ =>
  Vit.head (.inp 0) (.inp 1) (.inp 2) (.inp 3) (.inp 4) (.inp 5) (.inp 6) (.inp 7) (.inp 8)
def mlpOnly : TenProg Vit.SQ Vit.DM := fun v =>
  let h1 : Ten v Vit.SQ Vit.DFF :=
    .zipB Vit.geluW
      (.zipB Vit.addW (.mv Backend.proven
        (.inp 1 : Ten v Vit.DFF Vit.DM) (.inp 2 : Ten v Vit.SQ Vit.DM)) (.inp 3)) (.inp 4)
  .zipB Vit.addW (.mv Backend.proven (.inp 0) h1) (.inp 5)

#eval do
  for (nm, k) in [("layerNorm", (lnOnly.tape 500).length),
                  ("one head ", (headOnly.tape 500).length),
                  ("mlp      ", (mlpOnly.tape 500).length)] do
    IO.println s!"[vit] {nm}: {k} ops"
