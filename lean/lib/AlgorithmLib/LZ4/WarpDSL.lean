module
public import AlgorithmLib.LZ4.Ptx
meta import AlgorithmLib.LZ4.Ptx
public import AlgorithmLib.LZ4.WarpSched
meta import AlgorithmLib.LZ4.WarpSched
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib

/-- The comparisons a `setp` makes. -/
inductive SCmp where
  | eq | ne | lt | le | ge
  deriving Repr, DecidableEq

def SCmp.run : SCmp → UInt64 → UInt64 → Bool
  | .eq, a, b => a == b
  | .ne, a, b => a != b
  | .lt, a, b => a < b
  | .le, a, b => a ≤ b
  | .ge, a, b => b ≤ a

/-- Single-threaded warp state (cooperation collapsed to sequential semantics). -/
structure WState where
  regs : String → UInt64
  gmem : Array UInt8
  smem : Array UInt8

def WState.setReg (st : WState) (d : String) (v : UInt64) : WState :=
  { st with regs := fun x => if x = d then v else st.regs x }

def WState.stgByte (st : WState) (addr val : UInt64) : WState :=
  { st with gmem := st.gmem.set! addr.toNat val.toUInt8 }

/-- Sequential lane-strided-collapsed copy of `n` bytes within gmem. -/
def copyGmem (g : Array UInt8) (dst src : Nat) : Nat → Array UInt8
  | 0 => g
  | n + 1 => copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n

theorem copyGmem_size : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    (copyGmem g dst src n).size = g.size := by
  intro n
  induction n with
  | zero => intro g dst src; rfl
  | succ n ih =>
      intro g dst src
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).size = g.size
      rw [ih]; simp

/-- `copyGmem` leaves every index below its destination window untouched. -/
theorem copyGmem_getD_lt (j : Nat) : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    j < dst → (copyGmem g dst src n).getD j 0 = g.getD j 0 := by
  intro n
  induction n with
  | zero => intro g dst src _; rfl
  | succ n ih =>
      intro g dst src hj
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).getD j 0
        = g.getD j 0
      rw [ih (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) (by omega)]
      simp [Array.getElem?_setIfInBounds_ne (by omega : dst ≠ j)]

/-- **Copy collective correctness**: a disjoint, in-bounds `copyGmem` of `n` bytes
    writes the source run in order — `out[dst+i] = in[src+i]` for `i < n`.  (The
    DSL `coopCopy`'s sequential model; the byte-level payoff for the literal copy,
    mirroring W2's `coopCopy_segment`.) -/
theorem copyGmem_getD : ∀ (n : Nat) (g : Array UInt8) (dst src i : Nat),
    i < n → dst + n ≤ g.size → (dst + n ≤ src ∨ src + n ≤ dst) →
    (copyGmem g dst src n).getD (dst + i) 0 = g.getD (src + i) 0 := by
  intro n
  induction n with
  | zero => intro g dst src i hi _ _; omega
  | succ n ih =>
      intro g dst src i hi hsize hdisj
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).getD (dst + i) 0
        = g.getD (src + i) 0
      match i with
      | 0 =>
          simp only [Nat.add_zero]
          rw [copyGmem_getD_lt dst n (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1)
                (by omega)]
          simp [Array.getElem?_setIfInBounds_self_of_lt (by omega : dst < g.size)]
      | j + 1 =>
          rw [show dst + (j + 1) = (dst + 1) + j by omega,
              ih (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) j (by omega)
                (by simp; omega) (by omega),
              show (src + 1) + j = src + (j + 1) by omega]
          simp [Array.getElem?_setIfInBounds_ne (by omega : dst ≠ src + (j + 1))]

-- ── Operators / arguments ──

/-- Arithmetic ops (comparisons are `WStmt.setp`, carrying an `SCmp`). -/
inductive WOp
  | add | sub | mul | band | bor | shl | shr | min
  deriving Repr, DecidableEq

def WOp.run : WOp → UInt64 → UInt64 → UInt64
  | .add, a, b => a + b        | .sub, a, b => a - b
  | .mul, a, b => a * b        | .band, a, b => a &&& b
  | .bor, a, b => a ||| b      | .shl, a, b => a <<< b
  | .shr, a, b => a >>> b      | .min, a, b => if a ≤ b then a else b

inductive WArg | reg (n : String) | imm (v : Nat)
  deriving Repr, DecidableEq

def WArg.eval (st : WState) : WArg → UInt64
  | .reg n => st.regs n
  | .imm v => UInt64.ofNat v

/-- The model's `inp` view of gmem: the low `len` bytes. -/
def gmemInp (g : Array UInt8) (len : Nat) : List UInt8 :=
  (List.range len).map (fun i => g.getD i 0)

/-- The input block a warp reads, starting at its own base offset — the
    generalisation the all-warps statement needs. -/
def gmemInpAt (g : Array UInt8) (base len : Nat) : List UInt8 :=
  (List.range len).map (fun i => g.getD (base + i) 0)

theorem gmemInpAt_zero (g : Array UInt8) (len : Nat) : gmemInpAt g 0 len = gmemInp g len := by
  simp [gmemInpAt, gmemInp]

/-- The kernel's Fibonacci hash of the 4 little-endian bytes at `pos`. -/
def wHashK : Nat := 2654435761

def wLoad4 (g : Array UInt8) (pos : Nat) : Nat :=
  (g.getD pos 0).toNat + 256 * (g.getD (pos + 1) 0).toNat
    + 65536 * (g.getD (pos + 2) 0).toNat + 16777216 * (g.getD (pos + 3) 0).toNat

def wHash (g : Array UInt8) (hashLog pos : Nat) : Nat :=
  ((UInt64.ofNat (wLoad4 g pos) * UInt64.ofNat wHashK)
      >>> UInt64.ofNat (32 - hashLog)).toNat % 2 ^ hashLog

/-- The oracle a shared-memory hash table induces: entry `hash(pos)` holds a u16
    `cand+1` (0 = empty).  Mirrors the kernel's `ldsh; sub 1`. -/
def tableOracle (g sm : Array UInt8) (hashLog tbl ib : Nat) (pos : Nat) : Option Nat :=
  let a := tbl + 2 * wHash g hashLog (ib + pos)
  let raw := (sm.getD a 0).toNat + 256 * (sm.getD (a + 1) 0).toNat
  if raw = 0 then none else some (raw - 1)

/-- Insert-after-select: for each lane `k < 32` with `k ≤ upto` and `sp+k <
    searchLim`, write `sp+k+1` (u16 LE) into entry `hash(sp+k)`. -/
def tableInsert (sm g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) : Array UInt8 :=
  (List.range 32).foldl (fun acc k =>
    if k ≤ upto ∧ sp + k < searchLim then
      let a := tbl + 2 * wHash g hashLog (ib + (sp + k))
      let v := sp + k + 1
      (acc.setIfInBounds a (UInt8.ofNat (v % 256))).setIfInBounds (a + 1) (UInt8.ofNat (v / 256 % 256))
    else acc) sm

/-- Window-collective body given the (already-computed) window result `w` — with
    `w` an explicit argument, this def has light equation lemmas (unlike inlining
    the concrete `window`/`tableOracle`, whose eq-lemmas overflow the kernel).
    Where no lane hits, `p0` is the window's end and `cand0` lane 0's candidate
    `c0`, as the lanes compute them. -/
def evalCoopWindowGo (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat) (c0 : UInt64)
    (w : Option (Nat × Nat)) (st : WState) : WState :=
  let s := (st.regs sp).toNat
  match w with
  | some (p, c) =>
      let st1 := ((st.setReg found 1).setReg p0 (UInt64.ofNat p)).setReg cand0 (UInt64.ofNat c)
      { st1 with smem := tableInsert st.smem st.gmem hashLog tbl searchLim ib s (p - s) }
  | none =>
      let st1 := ((st.setReg found 0).setReg p0 (st.regs sp + 32)).setReg cand0 c0
      { st1 with smem := tableInsert st.smem st.gmem hashLog tbl searchLim ib s 32 }

def evalCoopWindow (found p0 cand0 sp : String) (inStride searchLim hashLog tbl : Nat)
    (st : WState) : WState :=
  evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim (st.regs "inBase").toNat
    ((tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat
      (min (st.regs sp).toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat)
    (AlgorithmLib.LZ4WarpFind.window (gmemInpAt st.gmem (st.regs "inBase").toNat inStride)
      (tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat)
      searchLim (st.regs sp).toNat) st

/-- `evalCoopWindow = evalCoopWindowGo` applied to the concrete window. -/
theorem evalCoopWindow_eq_go (found p0 cand0 sp : String) (inStride searchLim hashLog tbl : Nat)
    (st : WState) :
    evalCoopWindow found p0 cand0 sp inStride searchLim hashLog tbl st
      = evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim (st.regs "inBase").toNat
          ((tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat
          (min (st.regs sp).toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat)
          (AlgorithmLib.LZ4WarpFind.window (gmemInpAt st.gmem (st.regs "inBase").toNat inStride)
            (tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat)
            searchLim (st.regs sp).toNat) st := rfl

/-- `evalCoopWindowGo` preserves `gmem` (both branches only touch regs + smem). -/
theorem evalCoopWindowGo_gmem (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat) (c0 : UInt64)
    (w : Option (Nat × Nat)) (st : WState) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib c0 w st).gmem = st.gmem := by
  unfold evalCoopWindowGo; split <;> rfl

/-- `evalCoopWindowGo`'s smem = `tableInsert` at the window-determined `upto`. -/
theorem evalCoopWindowGo_smem (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat) (c0 : UInt64)
    (w : Option (Nat × Nat)) (st : WState) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib c0 w st).smem
      = tableInsert st.smem st.gmem hashLog tbl searchLim ib (st.regs sp).toNat
          (match w with | some (p, _) => p - (st.regs sp).toNat | none => 32) := by
  unfold evalCoopWindowGo; split <;> rfl

/-- `evalCoopWindowGo`'s register update, per register `r`. -/
theorem evalCoopWindowGo_regs (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat) (c0 : UInt64)
    (w : Option (Nat × Nat)) (st : WState) (r : String) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib c0 w st).regs r
      = (match w with
         | some (p, c) =>
             (((st.setReg found 1).setReg p0 (UInt64.ofNat p)).setReg cand0 (UInt64.ofNat c)).regs r
         | none => (((st.setReg found 0).setReg p0 (st.regs sp + 32)).setReg cand0 c0).regs r) := by
  unfold evalCoopWindowGo; split <;> rfl

/-- `eval` body for one extend 32-window. -/
def evalCoopExtendStep (adv p0 cand0 ml : String) (inStride endCap : Nat)
    (st : WState) : WState :=
  st.setReg adv (UInt64.ofNat
    (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt st.gmem (st.regs "inBase").toNat inStride)
      (st.regs p0).toNat (st.regs cand0).toNat endCap (st.regs ml).toNat 32))

/-- Abstract matchfinder loop with a threaded state `smem`: the oracle at each
    step is `mkOracle` of the current state, and `upd` advances the state. Keeping
    `mkOracle`/`upd` abstract keeps the auto-generated equation lemmas light —
    the concrete `tableOracle`/`tableInsert` would otherwise overflow the kernel
    during equation-lemma checking. Validity holds for *any* `mkOracle`/`upd`. -/
def genLoop (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap : Nat) :
    Nat → Array UInt8 → Nat → Nat → AlgorithmLib.LZ4Plan.Plan
  | 0, _, anchor, _ => ⟨[], inp.length - anchor⟩
  | fuel + 1, smem, anchor, s =>
      if s < searchLim then
        match AlgorithmLib.LZ4WarpFind.window inp (mkOracle smem) searchLim s with
        | some (p, c) =>
            let ml := AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap
              (endCap - (p + 4)) 4
            let smem' := upd smem s (p - s)
            let rest := genLoop inp mkOracle upd searchLim endCap fuel smem'
              (p + ml) (p + ml)
            ⟨⟨p - anchor, p - c, ml⟩ :: rest.steps, rest.finalLen⟩
        | none =>
            let smem' := upd smem s 32
            genLoop inp mkOracle upd searchLim endCap fuel smem' anchor (s + 32)
      else ⟨[], inp.length - anchor⟩

/-- The plan the sequential `eval` of `compressorBodyWStmt` computes: `genLoop`
    over the block input, with the shared table as threaded state — oracle
    `tableOracle` of the current table, update `tableInsert` (matching how
    `evalCoopWindow` mutates `smem`). -/
def evalPlan_loop (gmem : Array UInt8) (ib inStride searchLim endCap hashLog tbl : Nat)
    (fuel : Nat) (smem : Array UInt8) (anchor s : Nat) : AlgorithmLib.LZ4Plan.Plan :=
  genLoop (gmemInpAt gmem ib inStride)
    (fun sm => tableOracle gmem sm hashLog tbl ib)
    (fun sm sp upto => tableInsert sm gmem hashLog tbl searchLim ib sp upto)
    searchLim endCap fuel smem anchor s

/-- Top-level eval plan: fuel `inStride` is ample (each step advances `s` by ≥4). -/
def evalPlan (gmem smem0 : Array UInt8) (ib inStride hashLog : Nat) : AlgorithmLib.LZ4Plan.Plan :=
  evalPlan_loop gmem ib inStride (inStride - 12) (inStride - 5) hashLog 0 inStride smem0 0 0

-- ── Statements (flat uniform ops + high-level cooperative primitives) ─────────

inductive WStmt
  | skip
  | seq    (a b : WStmt)
  | bin    (o : WOp) (d a : String) (b : WArg)   -- d := a ⊕ b  (arith)
  | setp   (c : SCmp) (d a : String) (b : WArg)  -- d := (a ⋈ b) ? 1 : 0
  | mov    (d : String) (a : WArg)               -- d := a
  | stgB   (addr val : String)                   -- gmem[addr] := val    (byte)
  | uif    (cond : String) (t e : WStmt)         -- if regs[cond] ≠ 0
  | uwhile (cond : String) (body : WStmt)        -- while regs[cond] ≠ 0 (fuel-bounded)
  | coopCopy (dst src len : String)              -- gmem[dst..] := gmem[src..] (len bytes)
  -- Window-find collective: probe 32 candidates at `sp`, select first hit, insert.
  | coopWindow (found p0 cand0 sp : String) (inStride searchLim hashLog tbl : Nat)
  -- One extend 32-window: `adv := leadingGood inp p0 cand0 endCap ml` over [ml,ml+32).
  | coopExtendStep (adv p0 cand0 ml : String) (inStride endCap : Nat)

/-- **Sequential eval** (fuel bounds `uwhile`). -/
def WStmt.eval : Nat → WStmt → WState → WState
  | _, .skip, st => st
  | fuel, .seq a b, st => b.eval fuel (a.eval fuel st)
  | _, .bin o d a b, st => st.setReg d (o.run (st.regs a) (b.eval st))
  | _, .setp c d a b, st => st.setReg d (if c.run (st.regs a) (b.eval st) then 1 else 0)
  | _, .mov d a, st => st.setReg d (a.eval st)
  | _, .stgB addr val, st => st.stgByte (st.regs addr) (st.regs val)
  | fuel, .uif cond t e, st =>
      if (st.regs cond == 1) then t.eval fuel st else e.eval fuel st
  | 0, .uwhile _ _, st => st
  | fuel + 1, .uwhile cond body, st =>
      if (st.regs cond == 1) then
        WStmt.eval fuel (.uwhile cond body) (body.eval fuel st)
      else st
  | _, .coopCopy dst src len, st =>
      { st with gmem := copyGmem st.gmem (st.regs dst).toNat (st.regs src).toNat (st.regs len).toNat }
  | _, .coopWindow found p0 cand0 sp inStride searchLim hashLog tbl, st =>
      evalCoopWindow found p0 cand0 sp inStride searchLim hashLog tbl st
  | _, .coopExtendStep adv p0 cand0 ml inStride endCap, st =>
      evalCoopExtendStep adv p0 cand0 ml inStride endCap st

/-- The loop's guard becomes false within `fuel` iterations, so the fuel-bounded
    `eval` has actually terminated. Mirrors `eval`'s recursion exactly. -/
def WhileHalts (cond : String) (body : WStmt) : Nat → WState → Prop
  | 0, _ => False
  | fuel + 1, ws =>
      if (ws.regs cond == 1) then WhileHalts cond body fuel (body.eval fuel ws) else True

-- ── A ballot's first set lane ──

/-- Bit `j` of `foldl (acc ⊕= if P a then 2^(f a))` is set iff some selected
    element maps to `j`.  Both `brev32` and `ballotOf` are instances. -/
theorem testBit_foldl_or {α : Type} (P : α → Bool) (f : α → Nat) :
    ∀ (l : List α) (init j : Nat),
      (l.foldl (fun acc a => if P a then acc ||| 2 ^ (f a) else acc) init).testBit j
        = (init.testBit j || l.any (fun a => P a && decide (f a = j))) := by
  intro l
  induction l with
  | nil => intro init j; simp
  | cons a t ih =>
      intro init j
      simp only [List.foldl_cons, List.any_cons]
      rw [ih]
      have hstep : (if P a then init ||| 2 ^ (f a) else init).testBit j
          = (init.testBit j || (P a && decide (f a = j))) := by
        by_cases hP : P a
        · rw [if_pos hP, Nat.testBit_or, Nat.testBit_two_pow, hP, Bool.true_and]
        · rw [if_neg hP]
          have : P a = false := by simpa using hP
          rw [this, Bool.false_and, Bool.or_false]
      rw [hstep, Bool.or_assoc]

/-- Least index `≥ start` in the window `[start, start+bound)` with `g` true;
    `start+bound` if none.  With `(bound, start) = (32, 0)` this is exactly the
    lane `clz∘brev∘ballot` returns (or `32` for an empty ballot). -/
def firstSetNat (g : Nat → Bool) : Nat → Nat → Nat
  | 0, start => start
  | b + 1, start => if g start then start else firstSetNat g b (start + 1)

/-- **The clean bridge**: the least-set-index view equals `firstHit`,
    defaulting to `start+bound` (i.e. `32`) exactly when no lane hits. -/
theorem firstSetNat_eq_firstHit (g : Nat → Bool) :
    ∀ b start, firstSetNat g b start
      = (AlgorithmLib.LZ4WarpSched.firstHit g b start).getD (start + b) := by
  intro b
  induction b with
  | zero => intro start; simp [firstSetNat, AlgorithmLib.LZ4WarpSched.firstHit]
  | succ n ih =>
      intro start
      unfold firstSetNat AlgorithmLib.LZ4WarpSched.firstHit
      by_cases hg : g start = true
      · rw [if_pos hg, if_pos hg]; simp
      · rw [if_neg hg, if_neg hg, ih (start + 1)]
        congr 1; omega

/-- Specialization at the warp: `firstSetNat g 32 0 = (firstHit g 32 0).getD 32`. -/
theorem firstSetNat_warp (g : Nat → Bool) :
    firstSetNat g 32 0 = (AlgorithmLib.LZ4WarpSched.firstHit g 32 0).getD 32 := by
  have := firstSetNat_eq_firstHit g 32 0
  simpa using this

/-- `find?` depends only on the predicate's values at list elements. -/
theorem find?_congr {α : Type} {p q : α → Bool} :
    ∀ (l : List α), (∀ a ∈ l, p a = q a) → l.find? p = l.find? q := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a t ih =>
      intro h
      rw [List.find?_cons, List.find?_cons, h a (by simp)]
      cases q a
      · simp; exact ih (fun b hb => h b (by simp [hb]))
      · simp

/-- The recursive scan `firstSetNat` equals the `find?` over the same window. -/
theorem firstSetNat_eq_find (g : Nat → Bool) :
    ∀ b start, firstSetNat g b start = ((List.range' start b).find? g).getD (start + b) := by
  intro b
  induction b with
  | zero => intro start; simp [firstSetNat, List.range']
  | succ n ih =>
      intro start
      rw [firstSetNat, List.range'_succ]
      by_cases hg : g start = true
      · rw [if_pos hg, List.find?_cons_of_pos hg]; simp
      · rw [if_neg hg, List.find?_cons_of_neg hg, ih (start + 1)]
        congr 1; omega

end AlgorithmLib.LZ4WarpDSL
