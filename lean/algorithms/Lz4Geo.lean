import Lz4Interleave
import AlgorithmLib.LZ4Confine
import AlgorithmLib.LZ4OpBound

set_option maxRecDepth 8192

/-!
  # The extend loop's read addresses

  Two of the twelve load sites sit inside the match-extend loop: `110 ldgo bP,
  [aP]` and `111 ldgo bC, [aC]`, with `aP = inBase + peD` and `aC = inBase + caD`.

  `peD` is clamped by the kernel (`min pe ec1`), exactly like the two search
  clamps.  `caD` is not: it is `cand0 + (peC - p0)`, and nothing in the
  instruction stream bounds it.  What bounds it is the *select* that produced
  `cand0` — `vote → brev → clz → shfl` picks the earliest lane whose `pHit` is
  set, and that lane's guards say `cand < posP < searchLim`.  So the candidate
  the warp extends from is behind the position it extends from, and the extend
  can only walk forward to `ec1`.

  Getting that requires one fact the machine does not state locally: the loop
  cursor `searchPos` is the same in every lane.  Branches read lane 0, so the
  loop guard `searchPos < searchLim` constrains lane 0 and nothing else, while
  the bound is needed at the selected lane.  `uni_at` supplies it.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

/-- **Every program-shape check the window needs, in one `Bool`, over an
    arbitrary program and stride.**

    Parameterised over `p` and `S` rather than fixed to the shipped 32 KiB kernel:
    the pcs are the same at every geometry (the kernel has the same 274
    instructions whatever `blkLog` is — only immediates move), so only the clamp
    constant `S - 4` varies.  One `decide` per geometry establishes this; every
    lemma below is a projection out of it and evaluates no array at all. -/

def winShapeB (p : Array SInstr) (S : Nat) : Bool :=
  ((List.range 51).map (· + 42)).all
      (fun q => p[q]?.map fallthroughOnlyB == some true)
  && ((List.range 47).map (· + 45)).all
      (fun q => p[q]?.map (fun i => destOf i != some "cap4") == some true)
  && ((List.range 46).map (· + 46)).all
      (fun q => p[q]?.map (fun i => destOf i != some "rp") == some true)
  && ((List.range 27).map (· + 65)).all
      (fun q => p[q]?.map (fun i => destOf i != some "rc") == some true)
  && (p[44]? == some (SInstr.mov "cap4" (.imm (S - 4))))
  && (p[45]? == some (SInstr.binr .min "rp" "posP" "cap4"))
  && (p[64]? == some (SInstr.binr .min "rc" "cand" "cap4"))

-- ── Program-shape predicates, over an arbitrary geometry `(p, S)` ──────────

/-- Everything from the loop label to the back edge: pcs 38–207. -/
def loopS : List Nat := (List.range 170).map (· + 38)

/-- Where the guard's `searchPos < searchLim` is still in force.  Not the whole
    body: `200 litAnchor := p0 + ml` and `201 searchPos := litAnchor` move the
    cursor forward past the limit on the last iteration, and `204` is the
    no-match advance, which happens before that. -/
def GuardLive (q : Nat) : Prop := (42 ≤ q ∧ q ≤ 200) ∨ (203 ≤ q ∧ q ≤ 204)
instance : DecidablePred GuardLive := fun q => by unfold GuardLive; infer_instance

/-- The match extend: `94 ecR := 32763` through `120 bra Lh4`.  The only edge in
    from outside is the guard at 93 falling through. -/
def extS : List Nat := (List.range 27).map (· + 94)

/-- `true` on instructions that always fall through to the next pc. -/
def isStraightB : SInstr → Bool
  | .braif _ _ => false
  | .braifnot _ _ => false
  | .bra _ => false
  | .ret => false
  | _ => true

/-- The match path: pcs 94–200.  Stops at 200 because that is where `litAnchor`
    is moved on to the next iteration. -/
def matchS : List Nat := (List.range 107).map (· + 94)

/-- The token emit, from just below `123 litLen := p0 - litAnchor` to just below
    the match-length field: pcs 124–199. -/
def litS : List Nat := (List.range 76).map (· + 124)

/-- Everything from the extend loop's exit to the last iteration's anchor
    update: pcs 121–200.  The only edge in from outside is `99 → 121`. -/
def tailS : List Nat := (List.range 80).map (· + 121)

/-- Everything from the loop label to just below the `OOB` exit: pcs 38–271.
    Stops at 271 because the guard at pc 11 branches to `OOB` at 272, which
    would make 272 a second entry point. -/
def laS : List Nat := (List.range 234).map (· + 38)

/-- From just below `210 fLen := inStride - litAnchor` to the `OOB` exit. -/
def ftS : List Nat := (List.range 61).map (· + 211)

/-- Where each register the select argument reads is last written, as a window
    `[lo, 93]` with `lo + n = 94`.  Batched into one `decide`: the widest reaches
    back to pc 42, so per-use `decide`s would cost minutes. -/
def selFrames : List (String × Nat × Nat) :=
  [("posP", 43, 51), ("pValid", 44, 50), ("pCO", 78, 16), ("pNE", 77, 17),
   ("pEq", 79, 15), ("pH1", 80, 14), ("pH2", 81, 13), ("pHit", 82, 12),
   ("bal", 83, 11), ("rev", 84, 10), ("fl", 85, 9), ("p0", 91, 3),
   ("cand0", 92, 2), ("found", 93, 1), ("cand", 64, 30), ("searchPos", 42, 52), ("lane", 42, 52)]

/-- The prologue shape checks, batched into one `decide` so the emitted array
    reduces once (see `winShapeOK`). -/
def preShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (9 ≤ q') || decide (9 ≤ q ∨ q = 8)))
  && (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (3 ≤ q') || decide (3 ≤ q ∨ q = 2)))
  && (p[2]? == some (SInstr.mov "tid" (.reg "%tid.x")))
  && (p[8]? == some (SInstr.bin .band "lane" "tid" (.imm 31)))
  && noDest p "%tid.x"
  && ((List.range 271).map (· + 3)).all
      (fun q => p[q]?.map (fun i => destOf i != some "tid") == some true)
  && ((List.range 265).map (· + 9)).all
      (fun q => p[q]?.map (fun i => destOf i != some "lane") == some true)

/-- The guard checks, batched. -/
def guardShapeB (p : Array SInstr) (S : Nat) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' = 40) || decide (q = 39 ∨ q = 207)))
  && (p[39]? == some (SInstr.setp .lt "loopC" "searchPos" (.imm (S - 12))))
  && (p[206]? == some (SInstr.setp .lt "loopC" "searchPos" (.imm (S - 12))))
  && (p[40]? == some (SInstr.lbl "Lh0"))
  && (p[41]? == some (SInstr.braifnot "loopC" "Lx1"))
  && (p[207]? == some (SInstr.bra "Lh0"))
  && ((List.range 51).map (· + 42)).all
      (fun q => p[q]?.map (fun i => destOf i != some "searchPos") == some true)

def loopShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (GuardLive q') || decide (GuardLive q ∨ q = 41)))
  && loopS.all (fun q => !decide (q ≠ 200) ||
      (p[q]?.map (fun i => destOf i != some "litAnchor") == some true))
  && loopS.all (fun q => !decide (q ≠ 201 ∧ q ≠ 204) ||
      (p[q]?.map (fun i => destOf i != some "searchPos") == some true))
  && (p[200]? == some (SInstr.bin .add "litAnchor" "p0" (.reg "ml")))
  && (p[201]? == some (SInstr.mov "searchPos" (.reg "litAnchor")))
  && (p[204]? == some (SInstr.bin .add "searchPos" "searchPos" (.imm 32)))
  && (p[41]? == some (SInstr.braifnot "loopC" "Lx1"))
  && loopS.all (fun q => (succsOf p q).all (fun q' => decide (q' ∈ loopS) || decide (q' = 208)))
  && (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' = 38) || decide (q = 37)))
  && (p[35]? == some (SInstr.mov "litAnchor" (.imm 0)))
  && (p[36]? == some (SInstr.mov "searchPos" (.imm 0)))
  && ((List.range 3).map (· + 35)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && (p[36]?.map (fun i => destOf i != some "litAnchor") == some true)
  && (p[37]?.map (fun i => destOf i != some "litAnchor") == some true)
  && (p[37]?.map (fun i => destOf i != some "searchPos") == some true)


-- ── Regions as intervals ───────────────────────────────────────────────────
-- Every region here is a contiguous pc range, so `q ∈ R` should be arithmetic,
-- not a linear scan of a 232-element list.  Measured on the body region: the
-- entry check 9.35s → 0.42s (22x), the exit check 13.8s → 1.36s (10x).

/-- `q` lies in the pc window `[lo, lo+n)`. -/
def inIv (lo n q : Nat) : Bool := decide (lo ≤ q) && decide (q < lo + n)

/-- The list spelling of a region and the interval test agree. -/
theorem mem_ivList (lo n q : Nat) :
    q ∈ (List.range n).map (· + lo) ↔ lo ≤ q ∧ q < lo + n := by
  simp only [List.mem_map, List.mem_range]
  constructor
  · rintro ⟨a, ha, rfl⟩; omega
  · intro h; exact ⟨q - lo, by omega, by omega⟩

theorem inIv_iff (lo n q : Nat) : inIv lo n q = true ↔ lo ≤ q ∧ q < lo + n := by
  simp only [inIv, Bool.and_eq_true, decide_eq_true_eq]

/-- Nothing outside the window jumps into it except at `e`. -/
def ivEntryB (p : Array SInstr) (lo n e : Nat) : Bool :=
  cfgAll p (fun q q' => inIv lo n q || !inIv lo n q' || decide (q' = e))

theorem ivEntry_at (p : Array SInstr) (lo n e : Nat) (hsz : p.size = 274)
    (h : ivEntryB p lo n e = true) :
    ∀ q, q < 274 → q ∉ (List.range n).map (· + lo) →
      ∀ q' ∈ succsOf p q, q' ∈ (List.range n).map (· + lo) → q' = e := by
  intro q hq hnm q' hq' hmem
  have hs : q < p.size := by rw [hsz]; exact hq
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at this
  rcases this with (h1 | h1) | h1
  · exact absurd ((mem_ivList lo n q).mpr ((inIv_iff lo n q).mp h1)) hnm
  · exact absurd ((inIv_iff lo n q').mpr ((mem_ivList lo n q').mp hmem)) (by simp [h1])
  · exact h1

/-- Every successor of a pc in the window is in the window or at/after `b`. -/
def ivExitB (p : Array SInstr) (lo n b : Nat) : Bool :=
  cfgAll p (fun q q' => !inIv lo n q || inIv lo n q' || decide (b ≤ q'))

theorem ivExit_at (p : Array SInstr) (lo n b : Nat) (hsz : p.size = 274)
    (hlo : lo + n ≤ 274) (h : ivExitB p lo n b = true) :
    ∀ q ∈ (List.range n).map (· + lo),
      ∀ q' ∈ succsOf p q, q' ∈ (List.range n).map (· + lo) ∨ b ≤ q' := by
  intro q hq q' hq'
  have hqi := (mem_ivList lo n q).mp hq
  have hs : q < p.size := by rw [hsz]; omega
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at this
  rcases this with (h1 | h1) | h1
  · exact absurd ((inIv_iff lo n q).mpr hqi) (by simp [h1])
  · exact Or.inl ((mem_ivList lo n q').mpr ((inIv_iff lo n q').mp h1))
  · exact Or.inr h1

/-- Nothing outside `R` jumps into `R` except at `e`.  Stated through `cfgAll`
    so it is one traversal, not 274 indexings. -/
def regionEntryB (p : Array SInstr) (R : List Nat) (e : Nat) : Bool :=
  cfgAll p (fun q q' => decide (q ∈ R) || !decide (q' ∈ R) || decide (q' = e))

/-- The regions' entry facts.  Kept out of `loopShapeB` so that bundle's conjunct
    order — and every `obtain` pattern that destructures it — is untouched. -/
def entryShapeB (p : Array SInstr) : Bool :=
  ivEntryB p 38 170 38
  && ivEntryB p 94 27 94
  && ivEntryB p 94 107 94
  && ivEntryB p 124 76 124

/-- The select-and-extend-entry shape checks, batched. -/
def selShapeB (p : Array SInstr) (S : Nat) : Bool :=
  (p[42]? == some (SInstr.binr .add "posP" "searchPos" "lane"))
  && (p[43]? == some (SInstr.setp .lt "pValid" "posP" (.imm (S - 12))))
  && (p[77]? == some (SInstr.setp .lt "pCO" "cand" (.reg "posP")))
  && (p[79]? == some (SInstr.andp "pH1" "pValid" "pNE"))
  && (p[80]? == some (SInstr.andp "pH2" "pH1" "pCO"))
  && (p[81]? == some (SInstr.andp "pHit" "pH2" "pEq"))
  && (p[82]? == some (SInstr.vote "bal" "pHit"))
  && (p[83]? == some (SInstr.brev "rev" "bal"))
  && (p[84]? == some (SInstr.clz "fl" "rev"))
  && (p[90]? == some (SInstr.binr .add "p0" "searchPos" "fl"))
  && (p[91]? == some (SInstr.shfl "cand0" "cand" "fl"))
  && (p[92]? == some (SInstr.setp .ne "found" "bal" (.imm 0)))
  && (p[93]? == some (SInstr.braifnot "found" "Le2"))

def selFrameB (p : Array SInstr) : Bool :=
  ((List.range 53).map (· + 42)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && selFrames.all (fun t => ((List.range t.2.2).map (· + t.2.1)).all
      (fun q => p[q]?.map (fun i => destOf i != some t.1) == some true))
  && (sfindLabel p "Le2" == 203)

/-- The extend-region shape checks, batched. -/
def extShapeB (p : Array SInstr) (S : Nat) : Bool :=
  (p[94]? == some (SInstr.mov "ecR" (.imm (S - 5))))
  && (p[95]? == some (SInstr.mov "ec1" (.imm (S - 6))))
  && extS.all (fun q => p[q]?.map (fun i => destOf i != some "p0") == some true)
  && extS.all (fun q => p[q]?.map (fun i => destOf i != some "cand0") == some true)
  && ((List.range 25).map (· + 96)).all
      (fun q => p[q]?.map (fun i => destOf i != some "ec1") == some true)
  && ((List.range 26).map (· + 95)).all
      (fun q => p[q]?.map (fun i => destOf i != some "ecR") == some true)
  && extS.all (fun q => (succsOf p q).all (fun q' => decide (q' ∈ extS) || decide (q' = 121)))

def extLoadShapeB (p : Array SInstr) : Bool :=
  ((List.range 11).map (· + 100)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && ((List.range 3).map (· + 107)).all
      (fun q => p[q]?.map (fun i => destOf i != some "peD") == some true)
  && ((List.range 6).map (· + 104)).all
      (fun q => p[q]?.map (fun i => destOf i != some "peC") == some true)
  && (p[103]? == some (SInstr.binr .min "peC" "pe" "ec1"))
  && (p[106]? == some (SInstr.mov "peD" (.reg "peC")))

/-- The frames the `adv = 32` argument needs, batched.  Windows are `[118-n, 117]`
    for each register, chosen to stop just below its own assignment. -/
def advShapeB (p : Array SInstr) : Bool :=
  ((List.range 19).map (· + 100)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && [("revM", 117, 1), ("mis", 116, 2), ("balOk", 115, 3), ("pOk", 114, 4),
      ("pIn", 103, 15), ("pe", 102, 16), ("ecR", 102, 16), ("p0", 101, 17),
      ("idx", 101, 17), ("ml", 100, 18), ("lane", 100, 18)].all
      (fun t => ((List.range t.2.2).map (· + t.2.1)).all
        (fun q => p[q]?.map (fun i => destOf i != some t.1) == some true))
  && (p[100]? == some (SInstr.binr .add "idx" "ml" "lane"))
  && (p[101]? == some (SInstr.binr .add "pe" "p0" "idx"))
  && (p[102]? == some (SInstr.setp .lt "pIn" "pe" (.reg "ecR")))
  && (p[113]? == some (SInstr.andp "pOk" "pIn" "pEqB"))
  && (p[114]? == some (SInstr.vote "balOk" "pOk"))
  && (p[115]? == some (SInstr.bnot "mis" "balOk"))
  && (p[116]? == some (SInstr.brev "revM" "mis"))
  && (p[117]? == some (SInstr.clz "adv" "revM"))

def mlShapeB (p : Array SInstr) : Bool :=
  (p[96]? == some (SInstr.mov "ml" (.imm 4)))
  && (p[99]? == some (SInstr.braifnot "extC" "Lx5"))
  && (p[118]? == some (SInstr.bin .add "ml" "ml" (.reg "adv")))
  && (p[119]? == some (SInstr.setp .eq "extC" "adv" (.imm 32)))
  && (p[120]? == some (SInstr.bra "Lh4"))
  && (p[98]? == some (SInstr.lbl "Lh4"))
  && (sfindLabel p "Lx5" == 121)
  && extS.all (fun q => p[q]?.map (fun i => destOf i != some "p0") == some true)
  && (((List.range 21).map (· + 97)) ++ [119, 120]).all
      (fun q => p[q]?.map (fun i => destOf i != some "ml") == some true)
  && ([98, 120] : List Nat).all
      (fun q => p[q]?.map (fun i => destOf i != some "extC") == some true)
  && ((List.range 3).map (· + 118)).all
      (fun q => p[q]?.map (fun i => destOf i != some "adv") == some true)
  && ((((List.range 5).map (· + 94)) ++ ((List.range 20).map (· + 100))).all
      (fun q => p[q]?.map isStraightB == some true))

def caShapeB (p : Array SInstr) : Bool :=
  ((List.range 12).map (· + 100)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && [("caD", 109, 2), ("caC", 106, 5), ("dfe", 105, 6), ("peC", 104, 7),
      ("pe", 102, 9), ("idx", 101, 10), ("p0", 100, 11), ("cand0", 100, 11),
      ("ec1", 100, 11), ("ml", 100, 11), ("lane", 100, 11)].all
      (fun t => ((List.range t.2.2).map (· + t.2.1)).all
        (fun q => p[q]?.map (fun i => destOf i != some t.1) == some true))
  && (p[100]? == some (SInstr.binr .add "idx" "ml" "lane"))
  && (p[101]? == some (SInstr.binr .add "pe" "p0" "idx"))
  && (p[103]? == some (SInstr.binr .min "peC" "pe" "ec1"))
  && (p[104]? == some (SInstr.binr .sub "dfe" "peC" "p0"))
  && (p[105]? == some (SInstr.binr .add "caC" "cand0" "dfe"))
  && (p[108]? == some (SInstr.mov "caD" (.reg "caC")))

def matchShapeB (p : Array SInstr) : Bool :=
  matchS.all (fun q => p[q]?.map (fun i => destOf i != some "p0") == some true)
  && matchS.all (fun q => p[q]?.map (fun i => destOf i != some "searchPos") == some true)

def litShapeB (p : Array SInstr) : Bool :=
  litS.all (fun q => p[q]?.map (fun i => destOf i != some "litAnchor") == some true)
  && litS.all (fun q => p[q]?.map (fun i => destOf i != some "litLen") == some true)
  && (p[123]? == some (SInstr.bin .sub "litLen" "p0" (.reg "litAnchor")))
  && (p[124]?.map fallthroughOnlyB == some true)

def cpShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' = 156) || decide (q = 155 ∨ q = 168)))
  && (p[155]? == some (SInstr.setp .lt "cpCont" "cpI" (.reg "litLen")))
  && (p[167]? == some (SInstr.setp .lt "cpCont" "cpI" (.reg "litLen")))
  && (p[156]? == some (SInstr.lbl "Ch10"))
  && (p[157]? == some (SInstr.braifnot "cpCont" "Cx11"))
  && (p[168]? == some (SInstr.bra "Ch10"))
  && (sfindLabel p "Cx11" == 169)
  && ((List.range 15).map (· + 154)).all
      (fun q => p[q]?.map (fun i => destOf i != some "cpSrc") == some true)
  && ((List.range 15).map (· + 154)).all
      (fun q => p[q]?.map (fun i => destOf i != some "inBase") == some true)
  && ((List.range 10).map (· + 155)).all
      (fun q => p[q]?.map (fun i => destOf i != some "cpI") == some true)
  && ((List.range 11).map (· + 154)).all
      (fun q => p[q]?.map (fun i => destOf i != some "litLen") == some true)
  && (p[153]? == some (SInstr.bin .add "cpSrc" "inBase" (.reg "litAnchor")))
  && ((List.range 12).map (· + 157)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && (p[154]?.map fallthroughOnlyB == some true)
  && (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ ((List.range 15).map (· + 154))) || decide (q' = 154)
        || decide (q ∈ ((List.range 15).map (· + 154)))))

def tailShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ tailS) || decide (q' = 121) || decide (q ∈ tailS)))
  && tailS.all (fun q => p[q]?.map (fun i => destOf i != some "ml") == some true)
  && tailS.all (fun q => p[q]?.map (fun i => destOf i != some "p0") == some true)
  && (p[99]?.map (fun i => destOf i != some "ml") == some true)
  && (p[99]?.map (fun i => destOf i != some "p0") == some true)
  && (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' = 121) || decide (q = 99)))

def laShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ laS) || decide (q' = 38) || decide (q ∈ laS)))
  && laS.all (fun q => !decide (q ≠ 200) ||
      (p[q]?.map (fun i => destOf i != some "litAnchor") == some true))
  && (p[200]? == some (SInstr.bin .add "litAnchor" "p0" (.reg "ml")))

def ftShapeB (p : Array SInstr) (S : Nat) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ ftS) || decide (q' = 211) || decide (q ∈ ftS)))
  && ftS.all (fun q => p[q]?.map (fun i => destOf i != some "litAnchor") == some true)
  && ftS.all (fun q => p[q]?.map (fun i => destOf i != some "fLen") == some true)
  && (p[209]? == some (SInstr.mov "fLen" (.imm S)))
  && (p[210]? == some (SInstr.bin .sub "fLen" "fLen" (.reg "litAnchor")))
  && (p[210]?.map (fun i => destOf i != some "litAnchor") == some true)
  && ((List.range 2).map (· + 210)).all (fun q => p[q]?.map fallthroughOnlyB == some true)

def cp2ShapeB (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' = 242) || decide (q = 241 ∨ q = 254)))
  && (p[241]? == some (SInstr.setp .lt "cpCont" "cpI" (.reg "fLen")))
  && (p[253]? == some (SInstr.setp .lt "cpCont" "cpI" (.reg "fLen")))
  && (p[242]? == some (SInstr.lbl "Ch20"))
  && (p[243]? == some (SInstr.braifnot "cpCont" "Cx21"))
  && (p[254]? == some (SInstr.bra "Ch20"))
  && (sfindLabel p "Cx21" == 255)
  && ((List.range 15).map (· + 240)).all
      (fun q => p[q]?.map (fun i => destOf i != some "cpSrcF") == some true)
  && ((List.range 15).map (· + 240)).all
      (fun q => p[q]?.map (fun i => destOf i != some "inBase") == some true)
  && ((List.range 15).map (· + 240)).all
      (fun q => p[q]?.map (fun i => destOf i != some "litAnchor") == some true)
  && ((List.range 9).map (· + 241)).all
      (fun q => p[q]?.map (fun i => destOf i != some "cpI") == some true)
  && ((List.range 10).map (· + 240)).all
      (fun q => p[q]?.map (fun i => destOf i != some "fLen") == some true)
  && (p[239]? == some (SInstr.bin .add "cpSrcF" "inBase" (.reg "litAnchor")))
  && ((List.range 12).map (· + 243)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && (p[240]?.map fallthroughOnlyB == some true)
  && (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ ((List.range 15).map (· + 240))) || decide (q' = 240)
        || decide (q ∈ ((List.range 15).map (· + 240)))))

def ibShapeB (p : Array SInstr) (S : Nat) : Bool :=
  ((List.range 20).map (· + 0)).all (fun q => p[q]?.map fallthroughOnlyB == some true)
  && (p[0]? == some (SInstr.mov "inP" (.reg "in_ptr")))
  && (p[2]? == some (SInstr.mov "tid" (.reg "%tid.x")))
  && (p[7]? == some (SInstr.bin .shr "gwarp" "gtid" (.imm 5)))
  && (p[5]? == some (SInstr.binr .mul "gtid" "ctab" "ntid"))
  && (p[6]? == some (SInstr.binr .add "gtid" "gtid" "tid"))
  && (p[3]? == some (SInstr.mov "ctab" (.reg "%ctaid.x")))
  && (p[4]? == some (SInstr.mov "ntid" (.reg "%ntid.x")))
  && (p[13]? == some (SInstr.mov "gwD" (.reg "gwarp")))
  && (p[14]? == some (SInstr.mov "inOff" (.imm S)))
  && (p[15]? == some (SInstr.binr .mul "inOff" "gwD" "inOff"))
  && (p[18]? == some (SInstr.binr .add "inBase" "inP" "inOff"))
  && noDest p "in_ptr" && noDest p "%ctaid.x" && noDest p "%tid.x" && noDest p "%ntid.x"
  && ((List.range 18).map (· + 1)).all
      (fun q => p[q]?.map (fun i => destOf i != some "inP") == some true)
  && ((List.range 11).map (· + 8)).all
      (fun q => p[q]?.map (fun i => destOf i != some "gwarp") == some true)
  && ((List.range 5).map (· + 14)).all
      (fun q => p[q]?.map (fun i => destOf i != some "gwD") == some true)
  && ((List.range 3).map (· + 16)).all
      (fun q => p[q]?.map (fun i => destOf i != some "inOff") == some true)
  && ((List.range 12).map (· + 7)).all
      (fun q => p[q]?.map (fun i => destOf i != some "gtid") == some true)
  && ((List.range 15).map (· + 4)).all
      (fun q => p[q]?.map (fun i => destOf i != some "ctab") == some true)
  && ((List.range 14).map (· + 5)).all
      (fun q => p[q]?.map (fun i => destOf i != some "ntid") == some true)
  && ((List.range 16).map (· + 3)).all
      (fun q => p[q]?.map (fun i => destOf i != some "tid") == some true)

/-- **The lane-uniform closure.**  Every write to a member has only members (or
    immediates, or a ballot) as sources: `searchPos ← litAnchor ← p0 + ml`,
    `p0 ← searchPos + fl`, `ml ← ml + adv`, and `fl`/`adv` are `clz ∘ brev` of a
    `vote`, which is warp-wide by construction. -/
def uniR : List String :=
  ["searchPos", "litAnchor", "p0", "ml", "fl", "bal", "rev", "adv", "balOk", "mis", "revM", "extC", "cpI", "litLen", "cpCont", "fLen"]

def ibS : List Nat := (List.range 253).map (· + 19)

def ibRegOK (p : Array SInstr) : Bool :=
  (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ ibS) || decide (q' = 19) || decide (q ∈ ibS)))
  && ibS.all (fun q => p[q]?.map (fun i => destOf i != some "inBase") == some true)

/-- **Every program-shape premise of this file, bundled.**  An instance is the
    claim that the emitted array really has the shape the proofs below assume.

    The geometry-independent half of the program shape: everything the
    confinement proof assumes that does not mention the block stride. -/
class Shape (p : Array SInstr) : Prop where
  size : p.size = 274
  preShape : preShapeB p = true
  entryShape : entryShapeB p = true
  uniShape : p.toList.all (unifOK uniR) = true
  lx1 : sfindLabel p "Lx1" = 208
  lh4 : sfindLabel p "Lh4" = 98

  ibReg : ibRegOK p = true
  loopShape : loopShapeB p = true
  selFrame : selFrameB p = true
  extLoadShape : extLoadShapeB p = true
  advShape : advShapeB p = true
  mlShape : mlShapeB p = true
  caShape : caShapeB p = true
  matchShape : matchShapeB p = true
  litShape : litShapeB p = true
  cpShape : cpShapeB p = true
  tailShape : tailShapeB p = true
  laShape : laShapeB p = true
  cp2Shape : cp2ShapeB p = true

/-- A geometry: a program together with its block stride `S`.  Carried as a
    class so no theorem has to thread eighteen hypotheses. -/
class Geo (p : Array SInstr) (S : Nat) : Prop extends Shape p where
  sBound : 32 ≤ S ∧ S < 2 ^ 32
  winShape : winShapeB p S = true
  guardShape : guardShapeB p S = true
  selShape : selShapeB p S = true
  extShape : extShapeB p S = true
  ftShape : ftShapeB p S = true
  ibShape : ibShapeB p S = true


/-- **Nothing outside a region jumps into it except at its entry pc.** -/
theorem region_entry_lt {p : Array SInstr} [Shape p] {R : List Nat} {e : Nat}
    (h : regionEntryB p R e = true) :
    ∀ q, q < 274 → q ∉ R → ∀ q' ∈ succsOf p q, q' ∈ R → q' = e := by
  intro q hq hnm q' hq' hmem
  have hs : q < p.size := by rw [Shape.size (p := p)]; exact hq
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true',
    decide_eq_false_iff_not] at this
  rcases this with (h1 | h1) | h1
  · exact absurd h1 hnm
  · exact absurd hmem h1
  · exact h1

/-- Everything reachable from pc ≥ `b` stays at pc ≥ `b`.  One traversal. -/
def upClosedB (p : Array SInstr) (b : Nat) : Bool :=
  cfgAll p (fun q q' => !decide (b ≤ q) || decide (b ≤ q'))

theorem upClosed_at (p : Array SInstr) (b : Nat) (hsz : p.size = 274)
    (h : upClosedB p b = true) :
    ∀ q, q < 274 → b ≤ q → ∀ q' ∈ succsOf p q, b ≤ q' := by
  intro q hq hb q' hq'
  have hs : q < p.size := by rw [hsz]; exact hq
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at this
  rcases this with h1 | h1
  · exact absurd hb h1
  · exact h1

/-- `region_entry_lt` with the size as a hypothesis rather than a `Shape`
    instance, so it is usable upstream of the geometry instances. -/
theorem regionEntry_at (p : Array SInstr) (R : List Nat) (e : Nat) (hsz : p.size = 274)
    (h : regionEntryB p R e = true) :
    ∀ q, q < 274 → q ∉ R → ∀ q' ∈ succsOf p q, q' ∈ R → q' = e := by
  intro q hq hnm q' hq' hmem
  have hs : q < p.size := by rw [hsz]; exact hq
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true',
    decide_eq_false_iff_not] at this
  rcases this with (h1 | h1) | h1
  · exact absurd h1 hnm
  · exact absurd hmem h1
  · exact h1

/-- Every successor of a pc inside `R` is inside `R` or at/after `b`. -/
def regionExitB (p : Array SInstr) (R : List Nat) (b : Nat) : Bool :=
  cfgAll p (fun q q' => !decide (q ∈ R) || decide (q' ∈ R) || decide (b ≤ q'))

theorem regionExit_at (p : Array SInstr) (R : List Nat) (b : Nat) (hsz : p.size = 274)
    (hR : ∀ q ∈ R, q < 274) (h : regionExitB p R b = true) :
    ∀ q ∈ R, ∀ q' ∈ succsOf p q, q' ∈ R ∨ b ≤ q' := by
  intro q hq q' hq'
  have hs : q < p.size := by rw [hsz]; exact hR q hq
  have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true',
    decide_eq_false_iff_not] at this
  rcases this with (h1 | h1) | h1
  · exact absurd hq h1
  · exact Or.inl h1
  · exact Or.inr h1

/-- Two `cfgAll` scans with pointwise-equal predicates agree. -/
theorem cfgAll_congr (p : Array SInstr) (f g : Nat → Nat → Bool)
    (h : ∀ q q', f q q' = g q q') : cfgAll p f = cfgAll p g := by
  unfold cfgAll
  have he : (fun x : SInstr × Nat => (succsOfI p x.2 x.1).all (f x.2))
          = (fun x : SInstr × Nat => (succsOfI p x.2 x.1).all (g x.2)) := by
    funext x; congr 1; funext q'; exact h x.2 q'
  rw [he]

/-- **The region-entry scan, stated with list membership, equals the interval
    form.**  Statement-preserving: the shape bundles keep their spelling and every
    consumer projection is untouched, while `decide` stops doing a linear search
    of a 234-element list for each successor of each pc. -/
theorem cfgRegion_eq (p : Array SInstr) (hsz : p.size = 274) (lo n e : Nat) :
    (List.range 274).all (fun q => (succsOf p q).all
      (fun q' => !decide (q' ∈ (List.range n).map (· + lo)) || decide (q' = e)
        || decide (q ∈ (List.range n).map (· + lo))))
      = ivEntryB p lo n e := by
  rw [range_succs_eq_cfgAll p 274 hsz.symm]
  refine cfgAll_congr p _ _ (fun q q' => ?_)
  have h1 : decide (q' ∈ (List.range n).map (· + lo)) = inIv lo n q' := by
    simp [mem_ivList, inIv]
  have h2 : decide (q ∈ (List.range n).map (· + lo)) = inIv lo n q := by
    simp [mem_ivList, inIv]
  rw [h1, h2]
  cases inIv lo n q <;> cases inIv lo n q' <;> cases (decide (q' = e)) <;> rfl

/-- The same, for a scan already in `cfgAll` form (no size hypothesis needed). -/
theorem cfgAllRegion_eq (p : Array SInstr) (lo n e : Nat) :
    cfgAll p (fun q q' => !decide (q' ∈ (List.range n).map (· + lo)) || decide (q' = e)
      || decide (q ∈ (List.range n).map (· + lo)))
      = ivEntryB p lo n e := by
  refine cfgAll_congr p _ _ (fun q q' => ?_)
  have h1 : decide (q' ∈ (List.range n).map (· + lo)) = inIv lo n q' := by
    simp [mem_ivList, inIv]
  have h2 : decide (q ∈ (List.range n).map (· + lo)) = inIv lo n q := by
    simp [mem_ivList, inIv]
  rw [h1, h2]
  cases inIv lo n q <;> cases inIv lo n q' <;> cases (decide (q' = e)) <;> rfl

/-- `PcClosed` for an interval region, by one traversal. -/
def ivClosedB (p : Array SInstr) (lo n : Nat) (exits : List Nat) : Bool :=
  cfgAll p (fun q q' => !inIv lo n q || decide (q ∈ exits) || inIv lo n q')

theorem ivClosed_at (p : Array SInstr) (lo n : Nat) (exits : List Nat)
    (hsz : p.size = 274) (hlo : lo + n ≤ 274) (h : ivClosedB p lo n exits = true) :
    PcClosed p ((List.range n).map (· + lo)) exits := by
  intro q hq
  by_cases he : q ∈ exits
  · exact Or.inl he
  · refine Or.inr (fun q' hq' => ?_)
    have hqi := (mem_ivList lo n q).mp hq
    have hs : q < p.size := by rw [hsz]; omega
    have := List.all_eq_true.mp (cfgAll_at p _ h q hs) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (h1 | h1) | h1
    · exact absurd ((inIv_iff lo n q).mpr hqi) (by simp [h1])
    · exact absurd h1 he
    · exact (mem_ivList lo n q').mpr ((inIv_iff lo n q').mp h1)

-- ── Build-time only: the same scans, stated identically, decided linearly ────
-- `(List.range 274).all (fun q => … p[q]? …)` re-indexes the array at every step
-- and is O(n²).  These rewrite a shape predicate's clauses into the traversal
-- form *before* `decide` runs, so the fact stated is unchanged and only the
-- evaluation gets cheaper.  Measured on the 274-instruction kernel: the CFG
-- scan 8.4s → 1.6s, a window scan 7.1s → 0.7s.

theorem cfg_eqG (p : Array SInstr) (hp : p.size = 274) (f : Nat → Nat → Bool) :
    (List.range 274).all (fun q => (succsOf p q).all (f q)) = cfgAll p f :=
  range_succs_eq_cfgAll p 274 hp.symm f

theorem winG_eqG (p : Array SInstr) (hp : p.size = 274) (lo n : Nat) (P : Nat → Bool)
    (f : SInstr → Bool) (hsz : lo + n ≤ 274) :
    ((List.range n).map (· + lo)).all (fun q => !P q || (p[q]?.map f == some true))
      = winAllG p lo n P f :=
  range_winG_eq p lo n (by rw [hp]; exact hsz) P f

theorem win_eqG (p : Array SInstr) (hp : p.size = 274) (b k : Nat) (f : SInstr → Bool)
    (hsz : b + k ≤ 274) :
    ((List.range k).map (· + b)).all (fun q => p[q]?.map f == some true)
      = winAll p b k f :=
  range_win_eq p b k (by rw [hp]; exact hsz) f



variable {p : Array SInstr} [Shape p]

theorem uni_init (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    Unif uniR (initSt w inPtr outPtr gm smemB) := by
  intro r hr l l'
  simp only [uniR, List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl

/-- **Lane-uniformity, at every step of every trace.** -/
theorem uni_at (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) (k : Nat) :
    Unif uniR (siter p k (initSt w inPtr outPtr gm smemB)) :=
  siter_unif p uniR
    (fun _ i hi => by
      have := List.all_eq_true.mp (Shape.uniShape (p := p)) i
        (List.mem_of_getElem? (by simpa using hi))
      simpa using this)
    _ (uni_init w inPtr outPtr gm smemB) k

end Lz4Sites
