import AlgorithmLib.LZ4Simt
import AlgorithmLib.LZ4WarpSched

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt

/-- Single-threaded warp state (cooperation collapsed to sequential semantics). -/
structure WState where
  regs : String → UInt64
  gmem : Array UInt8
  smem : Array UInt8

def WState.setReg (st : WState) (d : String) (v : UInt64) : WState :=
  { st with regs := fun x => if x = d then v else st.regs x }

def WState.stgByte (st : WState) (addr val : UInt64) : WState :=
  { st with gmem := st.gmem.set! addr.toNat val.toUInt8 }

def WState.stshU16 (st : WState) (addr val : UInt64) : WState :=
  let s0 := st.smem.setIfInBounds addr.toNat val.toUInt8
  { st with smem := s0.setIfInBounds (addr + 1).toNat (val >>> 8).toUInt8 }

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

-- ── Operators / arguments (flat, matching `SInstr`) ───────────────────────────

/-- Arithmetic ops (comparisons are `WStmt.setp`, carrying an `SCmp` directly so
    they match the machine's `.setp` exactly — `WOp.run = SOp.run`). -/
inductive WOp
  | add | sub | mul | band | bor | shl | shr | min
  deriving Repr, DecidableEq

def WOp.run : WOp → UInt64 → UInt64 → UInt64
  | .add, a, b => a + b        | .sub, a, b => a - b
  | .mul, a, b => a * b        | .band, a, b => a &&& b
  | .bor, a, b => a ||| b      | .shl, a, b => a <<< b
  | .shr, a, b => a >>> b      | .min, a, b => if a ≤ b then a else b

def WOp.toS : WOp → SOp
  | .add => .add | .sub => .sub | .mul => .mul | .band => .band
  | .bor => .bor | .shl => .shl | .shr => .shr | .min => .min

theorem WOp.run_eq_toS (o : WOp) (a b : UInt64) : o.run a b = o.toS.run a b := by
  cases o <;> rfl

inductive WArg | reg (n : String) | imm (v : Nat)
  deriving Repr, DecidableEq

def WArg.eval (st : WState) : WArg → UInt64
  | .reg n => st.regs n
  | .imm v => UInt64.ofNat v

/-- The `SArg` this compiles to (emit side). -/
def WArg.toS : WArg → SArg
  | .reg n => .reg n
  | .imm v => .imm v

-- ── Collective model helpers (window-find / extend, self-contained on WState) ──

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
    the concrete `window`/`tableOracle`, whose eq-lemmas overflow the kernel). -/
def evalCoopWindowGo (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat)
    (w : Option (Nat × Nat)) (st : WState) : WState :=
  let s := (st.regs sp).toNat
  match w with
  | some (p, c) =>
      let st1 := ((st.setReg found 1).setReg p0 (UInt64.ofNat p)).setReg cand0 (UInt64.ofNat c)
      { st1 with smem := tableInsert st.smem st.gmem hashLog tbl searchLim ib s (p - s) }
  | none =>
      let st1 := st.setReg found 0
      { st1 with smem := tableInsert st.smem st.gmem hashLog tbl searchLim ib s 32 }

def evalCoopWindow (found p0 cand0 sp : String) (inStride searchLim hashLog tbl : Nat)
    (st : WState) : WState :=
  evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim (st.regs "inBase").toNat
    (AlgorithmLib.LZ4WarpFind.window (gmemInpAt st.gmem (st.regs "inBase").toNat inStride)
      (tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat)
      searchLim (st.regs sp).toNat) st

/-- `evalCoopWindow = evalCoopWindowGo` applied to the concrete window. -/
theorem evalCoopWindow_eq_go (found p0 cand0 sp : String) (inStride searchLim hashLog tbl : Nat)
    (st : WState) :
    evalCoopWindow found p0 cand0 sp inStride searchLim hashLog tbl st
      = evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim (st.regs "inBase").toNat
          (AlgorithmLib.LZ4WarpFind.window (gmemInpAt st.gmem (st.regs "inBase").toNat inStride)
            (tableOracle st.gmem st.smem hashLog tbl (st.regs "inBase").toNat)
            searchLim (st.regs sp).toNat) st := rfl

/-- `evalCoopWindowGo` preserves `gmem` (both branches only touch regs + smem). -/
theorem evalCoopWindowGo_gmem (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat)
    (w : Option (Nat × Nat)) (st : WState) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib w st).gmem = st.gmem := by
  unfold evalCoopWindowGo; split <;> rfl

/-- `evalCoopWindowGo`'s smem = `tableInsert` at the window-determined `upto`. -/
theorem evalCoopWindowGo_smem (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat)
    (w : Option (Nat × Nat)) (st : WState) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib w st).smem
      = tableInsert st.smem st.gmem hashLog tbl searchLim ib (st.regs sp).toNat
          (match w with | some (p, _) => p - (st.regs sp).toNat | none => 32) := by
  unfold evalCoopWindowGo; split <;> rfl

/-- `evalCoopWindowGo`'s register update, per register `r`. -/
theorem evalCoopWindowGo_regs (found p0 cand0 sp : String) (hashLog tbl searchLim ib : Nat)
    (w : Option (Nat × Nat)) (st : WState) (r : String) :
    (evalCoopWindowGo found p0 cand0 sp hashLog tbl searchLim ib w st).regs r
      = (match w with
         | some (p, c) =>
             (((st.setReg found 1).setReg p0 (UInt64.ofNat p)).setReg cand0 (UInt64.ofNat c)).regs r
         | none => (st.setReg found 0).regs r) := by
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
  | ldgB   (d addr : String) (off : Nat)         -- d := gmem[addr+off]  (byte)
  | ldshU  (d addr : String)                     -- d := smem[addr]      (u16 LE)
  | stgB   (addr val : String)                   -- gmem[addr] := val    (byte)
  | stshU  (addr val : String)                   -- smem[addr] := val    (u16 LE)
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
  | _, .ldgB d addr off, st =>
      st.setReg d (UInt64.ofNat (st.gmem.getD ((st.regs addr).toNat + off) 0).toNat)
  | _, .ldshU d addr, st =>
      let a := (st.regs addr).toNat
      st.setReg d (UInt64.ofNat ((st.smem.getD a 0).toNat + 256 * (st.smem.getD (a + 1) 0).toNat))
  | _, .stgB addr val, st => st.stgByte (st.regs addr) (st.regs val)
  | _, .stshU addr val, st => st.stshU16 (st.regs addr) (st.regs val)
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

end AlgorithmLib.LZ4WarpDSL
