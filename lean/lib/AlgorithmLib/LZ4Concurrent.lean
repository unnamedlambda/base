import AlgorithmLib.LZ4SimtRSim

set_option maxRecDepth 8192

/-!
  # Many warps, one memory

  Everything else about the compressor is stated for ONE warp against a flat
  global memory: `sstep` is a 32-lane SIMT machine and knows nothing about the
  6400 warps a launch actually starts.  `LaunchAgreesPerWarp` is the assumption
  that bridged the gap — that the final memory of the real launch agrees, on each
  warp's output range, with what that warp would have written alone.

  This file replaces the *composition* half of that assumption with a proof.  The
  instruction set makes it tractable: exactly one instruction reads global memory
  (`ldgo`) and exactly three write it (`stg`, `stgp`, `stg32p`), so "what does
  this step touch" is a short definition rather than a program analysis.

  What is proven here: an interleaving of race-free warps computes, on each
  warp's own region, exactly what that warp computes alone, and leaves everything
  outside all regions untouched.  What remains assumed is that the hardware
  realises *some* interleaving — PTX's documented DRF-SC guarantee, which is
  hardware, not this kernel.
-/

namespace AlgorithmLib.LZ4Simt

open AlgorithmLib

-- ── What a step touches ───────────────────────────────────────────────────────

/-- **What an instruction reads from global memory**, as a predicate rather than
    a list: `List.finRange 32` is a term the elaborator will happily evaluate, and
    a predicate keeps every proof below symbolic.

    `ldgo` is the only instruction in the set that reads global memory, so this is
    exact, not an over-approximation. -/
def ReadsI (i : SInstr) (st : SState) (j : Nat) : Prop :=
  match i with
  | .ldgo _ addr off => ∃ l : Lane, (st.regs addr l).toNat + off = j
  | .ldgop p _ addr off => ∃ l : Lane, st.regs p l = 1 ∧ (st.regs addr l).toNat + off = j
  | _ => False

/-- **What an instruction may write.**  `stg`/`stgp` write one byte per lane,
    `stg32p` four; nothing else touches global memory.  Predicates are ignored —
    a superset of the writes is what a frame needs. -/
def WritesI (i : SInstr) (st : SState) (j : Nat) : Prop :=
  match i with
  | .stg addr _ => ∃ l : Lane, (st.regs addr l).toNat = j
  | .stgp _ addr _ => ∃ l : Lane, (st.regs addr l).toNat = j
  | .stg32p _ addr _ => ∃ l : Lane,
      (st.regs addr l).toNat = j ∨ (st.regs addr l + 1).toNat = j ∨
      (st.regs addr l + 2).toNat = j ∨ (st.regs addr l + 3).toNat = j
  | _ => False

/-- The same, for whichever instruction the pc selects. -/
def Reads (prog : Array SInstr) (st : SState) (j : Nat) : Prop :=
  match prog[st.pc]? with
  | some i => ReadsI i st j
  | none => False

def Writes (prog : Array SInstr) (st : SState) (j : Nat) : Prop :=
  match prog[st.pc]? with
  | some i => WritesI i st j
  | none => False

-- ── `storeBytes`: size, and the frame ─────────────────────────────────────────

private def sbFold (pred : Lane → Bool) (addr val : Lane → UInt64) :
    List Lane → Array UInt8 → Array UInt8
  | [], m => m
  | l :: ls, m => sbFold pred addr val ls (if pred l then m.set! (addr l).toNat (val l).toUInt8 else m)

private theorem storeBytes_eq_sbFold (mem : Array UInt8) (pred : Lane → Bool)
    (addr val : Lane → UInt64) (ls : List Lane) :
    ls.foldl (fun m l => if pred l then m.set! (addr l).toNat (val l).toUInt8 else m) mem
      = sbFold pred addr val ls mem := by
  induction ls generalizing mem with
  | nil => rfl
  | cons l ls ih => rw [List.foldl_cons, ih, sbFold]

private theorem sbFold_size (pred : Lane → Bool) (addr val : Lane → UInt64) :
    ∀ (ls : List Lane) (m : Array UInt8), (sbFold pred addr val ls m).size = m.size := by
  intro ls
  induction ls with
  | nil => intro m; rfl
  | cons l ls ih =>
      intro m
      by_cases h : pred l
      · simp [sbFold, h, ih, Array.set!_eq_setIfInBounds]
      · simp [sbFold, h, ih]

theorem storeBytes_size (mem : Array UInt8) (pred : Lane → Bool) (addr val : Lane → UInt64) :
    (storeBytes mem pred addr val).size = mem.size := by
  rw [storeBytes, storeBytes_eq_sbFold, sbFold_size]

private theorem sbFold_getD_of_ne (pred : Lane → Bool) (addr val : Lane → UInt64) (j : Nat) :
    ∀ (ls : List Lane) (m : Array UInt8),
      (∀ l ∈ ls, (addr l).toNat ≠ j) →
      (sbFold pred addr val ls m).getD j 0 = m.getD j 0 := by
  intro ls
  induction ls with
  | nil => intro m _; rfl
  | cons l ls ih =>
      intro m hne
      have hl : (addr l).toNat ≠ j := hne l (by simp)
      have hrest : ∀ l' ∈ ls, (addr l').toNat ≠ j := fun l' hl' => hne l' (by simp [hl'])
      by_cases h : pred l
      · rw [sbFold, if_pos h, ih _ hrest]
        simp [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
          Array.getElem?_setIfInBounds, hl]
      · rw [sbFold, if_neg h, ih _ hrest]

/-- **The frame.**  A byte no lane addresses comes through a `storeBytes`
    unchanged. -/
theorem storeBytes_getD_of_ne (mem : Array UInt8) (pred : Lane → Bool)
    (addr val : Lane → UInt64) (j : Nat)
    (h : ∀ l : Lane, (addr l).toNat ≠ j) :
    (storeBytes mem pred addr val).getD j 0 = mem.getD j 0 := by
  rw [storeBytes, storeBytes_eq_sbFold]
  exact sbFold_getD_of_ne pred addr val j _ mem (fun l _ => h l)

/-- **`storeBytes` is congruent.**  Two memories of equal size that agree at `j`
    still agree at `j` after the same store: at an address a lane wrote, both
    hold the value that lane supplied; elsewhere, both are unchanged.

    This is what lets a warp's step be replayed against a different memory — the
    concurrent one instead of its own — without a case analysis on which lane
    wrote last. -/
private theorem sbFold_agree (pred : Lane → Bool) (addr val : Lane → UInt64) (j : Nat) :
    ∀ (ls : List Lane) (g g' : Array UInt8), g.size = g'.size →
      g.getD j 0 = g'.getD j 0 →
      (sbFold pred addr val ls g).getD j 0 = (sbFold pred addr val ls g').getD j 0 := by
  intro ls
  induction ls with
  | nil => intro g g' _ h; exact h
  | cons l ls ih =>
      intro g g' hsize h
      by_cases hp : pred l
      · rw [sbFold, sbFold, if_pos hp, if_pos hp]
        refine ih _ _ ?_ ?_
        · simp [Array.set!_eq_setIfInBounds, hsize]
        · by_cases he : (addr l).toNat = j
          · subst he
            simp [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
              Array.getElem?_setIfInBounds, hsize]
          · simp only [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
              Array.getElem?_setIfInBounds, if_neg he]
            simpa [Array.getD_eq_getD_getElem?] using h
      · rw [sbFold, sbFold, if_neg hp, if_neg hp]
        exact ih _ _ hsize h

theorem storeBytes_agree (g g' : Array UInt8) (pred : Lane → Bool)
    (addr val : Lane → UInt64) (j : Nat)
    (hsize : g.size = g'.size) (h : g.getD j 0 = g'.getD j 0) :
    (storeBytes g pred addr val).getD j 0 = (storeBytes g' pred addr val).getD j 0 := by
  rw [storeBytes, storeBytes, storeBytes_eq_sbFold, storeBytes_eq_sbFold]
  exact sbFold_agree pred addr val j _ g g' hsize h

/-- Agreement-with-size, the form that chains: `stg32p` is four stores deep, and
    each layer needs the size equality the previous one produced. -/
theorem storeBytes_agree' (g g' : Array UInt8) (pred : Lane → Bool)
    (addr val : Lane → UInt64) (j : Nat)
    (h : g.size = g'.size ∧ g.getD j 0 = g'.getD j 0) :
    (storeBytes g pred addr val).size = (storeBytes g' pred addr val).size ∧
    (storeBytes g pred addr val).getD j 0 = (storeBytes g' pred addr val).getD j 0 :=
  ⟨by rw [storeBytes_size, storeBytes_size]; exact h.1,
   storeBytes_agree g g' pred addr val j h.1 h.2⟩

-- ── One step: the frame, the size, and replay against another memory ──────────

/-- Global memory never changes size. -/
theorem sstepInstr_gmem_size (prog : Array SInstr) (i : SInstr) (st : SState) :
    (sstepInstr prog i st).gmem.size = st.gmem.size := by
  cases i <;>
    simp only [sstepInstr, SState.setReg, SState.setPc, storeBytes_size]

theorem sstep_gmem_size (prog : Array SInstr) (st : SState) :
    (sstep prog st).gmem.size = st.gmem.size := by
  rw [sstep]; split
  · rfl
  · rfl
  · exact sstepInstr_gmem_size _ _ _

/-- **The frame for one step.**  A byte the instruction does not address is
    unchanged.  Nineteen of the twenty-two instructions do not touch global
    memory at all, which is why this is short. -/
theorem sstepInstr_gmem_frame (prog : Array SInstr) (i : SInstr) (st : SState) (j : Nat)
    (h : ¬ WritesI i st j) :
    (sstepInstr prog i st).gmem.getD j 0 = st.gmem.getD j 0 := by
  cases i with
  | stg addr s =>
      simp only [sstepInstr]
      exact storeBytes_getD_of_ne _ _ _ _ j (fun l hl => h ⟨l, hl⟩)
  | stgp p addr s =>
      simp only [sstepInstr]
      exact storeBytes_getD_of_ne _ _ _ _ j (fun l hl => h ⟨l, hl⟩)
  | stg32p p addr s =>
      have h0 : ∀ l : Lane, (st.regs addr l).toNat ≠ j :=
        fun l hl => h ⟨l, Or.inl hl⟩
      have h1 : ∀ l : Lane, (st.regs addr l + 1).toNat ≠ j :=
        fun l hl => h ⟨l, Or.inr (Or.inl hl)⟩
      have h2 : ∀ l : Lane, (st.regs addr l + 2).toNat ≠ j :=
        fun l hl => h ⟨l, Or.inr (Or.inr (Or.inl hl))⟩
      have h3 : ∀ l : Lane, (st.regs addr l + 3).toNat ≠ j :=
        fun l hl => h ⟨l, Or.inr (Or.inr (Or.inr hl))⟩
      simp only [sstepInstr]
      rw [storeBytes_getD_of_ne _ _ _ _ j h3, storeBytes_getD_of_ne _ _ _ _ j h2,
        storeBytes_getD_of_ne _ _ _ _ j h1, storeBytes_getD_of_ne _ _ _ _ j h0]
  | _ => simp only [sstepInstr, SState.setReg, SState.setPc]



/-- **`storeBytes` only looks at the values of ACTIVE lanes.**  Needed once the
    cooperative copy loads under the same predicate it stores under: a masked lane
    keeps a stale `cpB`, and that value must not be allowed to matter. -/
private theorem sbFold_val_congr (pred : Lane → Bool) (addr v1 v2 : Lane → UInt64)
    (h : ∀ l : Lane, pred l = true → v1 l = v2 l) :
    ∀ (ls : List Lane) (m : Array UInt8),
      sbFold pred addr v1 ls m = sbFold pred addr v2 ls m := by
  intro ls
  induction ls with
  | nil => intro m; rfl
  | cons l ls ih =>
      intro m
      by_cases hp : pred l
      · rw [sbFold, sbFold, if_pos hp, if_pos hp, h l hp, ih]
      · rw [sbFold, sbFold, if_neg hp, if_neg hp, ih]

theorem storeBytes_val_congr (mem : Array UInt8) (pred : Lane → Bool) (addr v1 v2 : Lane → UInt64)
    (h : ∀ l : Lane, pred l = true → v1 l = v2 l) :
    storeBytes mem pred addr v1 = storeBytes mem pred addr v2 := by
  rw [storeBytes, storeBytes, storeBytes_eq_sbFold, storeBytes_eq_sbFold]
  exact sbFold_val_congr pred addr v1 v2 h _ mem

/-- **The frame, respecting the predicate.**  A byte no *active* lane addresses
    comes through unchanged.  `storeBytes_getD_of_ne` is this with the predicate
    thrown away; keeping it is what lets a predicated store be confined at all. -/
private theorem sbFold_getD_of_ne' (pred : Lane → Bool) (addr val : Lane → UInt64) (j : Nat) :
    ∀ (ls : List Lane) (m : Array UInt8),
      (∀ l ∈ ls, pred l = true → (addr l).toNat ≠ j) →
      (sbFold pred addr val ls m).getD j 0 = m.getD j 0 := by
  intro ls
  induction ls with
  | nil => intro m _; rfl
  | cons l ls ih =>
      intro m hne
      have hrest : ∀ l' ∈ ls, pred l' = true → (addr l').toNat ≠ j :=
        fun l' hl' => hne l' (by simp [hl'])
      by_cases h : pred l
      · have hl : (addr l).toNat ≠ j := hne l (by simp) h
        rw [sbFold, if_pos h, ih _ hrest]
        simp [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
          Array.getElem?_setIfInBounds, hl]
      · rw [sbFold, if_neg h, ih _ hrest]

theorem storeBytes_getD_of_ne' (mem : Array UInt8) (pred : Lane → Bool)
    (addr val : Lane → UInt64) (j : Nat)
    (h : ∀ l : Lane, pred l = true → (addr l).toNat ≠ j) :
    (storeBytes mem pred addr val).getD j 0 = mem.getD j 0 := by
  rw [storeBytes, storeBytes_eq_sbFold]
  exact sbFold_getD_of_ne' pred addr val j _ mem (fun l _ => h l)

/-- **What an instruction actually writes.**  `WritesI` deliberately ignores
    predicates, which is the safe direction for a frame (`¬ Writes` ⇒ unchanged)
    and the WRONG direction for confinement: it demands that the addresses of
    MASKED lanes also be in region, and in the cooperative copy those run past
    the end of the literal run.  This is the precise set. -/
def WritesActI (i : SInstr) (st : SState) (j : Nat) : Prop :=
  match i with
  | .stg addr _ => ∃ l : Lane, (st.regs addr l).toNat = j
  | .stgp p addr _ => ∃ l : Lane, st.regs p l = 1 ∧ (st.regs addr l).toNat = j
  | .stg32p p addr _ => ∃ l : Lane, st.regs p l = 1 ∧
      ((st.regs addr l).toNat = j ∨ (st.regs addr l + 1).toNat = j ∨
       (st.regs addr l + 2).toNat = j ∨ (st.regs addr l + 3).toNat = j)
  | _ => False

def WritesAct (prog : Array SInstr) (st : SState) (j : Nat) : Prop :=
  match prog[st.pc]? with
  | some i => WritesActI i st j
  | none => False

/-- The precise set is contained in the over-approximation. -/
theorem writesAct_writes (prog : Array SInstr) (st : SState) (j : Nat)
    (h : WritesAct prog st j) : Writes prog st j := by
  rw [WritesAct] at h
  rw [Writes]
  cases hp : prog[st.pc]? with
  | none => rw [hp] at h; exact absurd h not_false
  | some i =>
      rw [hp] at h
      cases i <;> simp only [WritesActI, WritesI] at h ⊢ <;>
        first
          | exact h
          | (obtain ⟨l, -, hl⟩ := h; exact ⟨l, hl⟩)
          | exact absurd h not_false

private theorem sstepInstr_gmem_frame' (prog : Array SInstr) (i : SInstr) (st : SState) (j : Nat)
    (h : ¬ WritesActI i st j) :
    (sstepInstr prog i st).gmem.getD j 0 = st.gmem.getD j 0 := by
  cases i with
  | stg addr s =>
      simp only [sstepInstr]
      exact storeBytes_getD_of_ne _ _ _ _ j (fun l hl => h ⟨l, hl⟩)
  | stgp p addr s =>
      simp only [sstepInstr]
      refine storeBytes_getD_of_ne' _ _ _ _ j (fun l hp hl => h ⟨l, ?_, hl⟩)
      exact by simpa using hp
  | stg32p p addr s =>
      simp only [WritesActI] at h
      have hne : ∀ l : Lane, (st.regs p l == 1) = true →
          ((st.regs addr l).toNat ≠ j ∧ (st.regs addr l + 1).toNat ≠ j
            ∧ (st.regs addr l + 2).toNat ≠ j ∧ (st.regs addr l + 3).toNat ≠ j) := by
      -- placeholder
        intro l hp
        have hp' : st.regs p l = 1 := by simpa using hp
        exact ⟨fun e => h ⟨l, hp', Or.inl e⟩, fun e => h ⟨l, hp', Or.inr (Or.inl e)⟩,
          fun e => h ⟨l, hp', Or.inr (Or.inr (Or.inl e))⟩,
          fun e => h ⟨l, hp', Or.inr (Or.inr (Or.inr e))⟩⟩
      simp only [sstepInstr]
      rw [storeBytes_getD_of_ne' _ _ _ _ j (fun l hp => (hne l hp).2.2.2),
        storeBytes_getD_of_ne' _ _ _ _ j (fun l hp => (hne l hp).2.2.1),
        storeBytes_getD_of_ne' _ _ _ _ j (fun l hp => (hne l hp).2.1),
        storeBytes_getD_of_ne' _ _ _ _ j (fun l hp => (hne l hp).1)]
  | _ => simp only [sstepInstr, SState.setReg, SState.setPc]

/-- **The frame, against the precise write set.**  Strictly stronger than
    `sstep_gmem_frame`, and it is what confinement of a predicated store needs. -/
theorem sstep_gmem_frame' (prog : Array SInstr) (st : SState) (j : Nat)
    (h : ¬ WritesAct prog st j) :
    (sstep prog st).gmem.getD j 0 = st.gmem.getD j 0 := by
  rw [WritesAct] at h
  rw [sstep]
  split
  · rfl
  · rfl
  · rename_i i hi
    rw [hi] at h
    exact sstepInstr_gmem_frame' _ _ _ _ h

theorem sstep_gmem_frame (prog : Array SInstr) (st : SState) (j : Nat)
    (h : ¬ Writes prog st j) :
    (sstep prog st).gmem.getD j 0 = st.gmem.getD j 0 := by
  rw [Writes] at h
  rw [sstep]
  split
  · rfl
  · rfl
  · rename_i i hi
    rw [hi] at h
    exact sstepInstr_gmem_frame _ _ _ _ h

-- ── Replaying a step against a different memory ───────────────────────────────

/-- **The local state does not notice the swap.**  If the other memory agrees
    everywhere this instruction reads, the registers, shared memory and pc come
    out identical.  Twenty-one of the twenty-two instructions do not read global
    memory at all, so only `ldgo` has content. -/
theorem sstepInstr_local_congr (prog : Array SInstr) (i : SInstr) (st : SState)
    (g : Array UInt8) (hread : ∀ j, ReadsI i st j → g.getD j 0 = st.gmem.getD j 0) :
    (sstepInstr prog i { st with gmem := g }).regs = (sstepInstr prog i st).regs ∧
    (sstepInstr prog i { st with gmem := g }).smem = (sstepInstr prog i st).smem ∧
    (sstepInstr prog i { st with gmem := g }).pc = (sstepInstr prog i st).pc := by
  cases i with
  | ldgo d addr off =>
      refine ⟨?_, rfl, rfl⟩
      simp only [sstepInstr, SState.setReg, SState.setPc]
      funext x
      by_cases hx : x = d
      · subst hx
        funext l
        simp only [if_pos rfl, if_true]
        rw [hread _ ⟨l, rfl⟩]
      · simp only [if_neg hx]
  | ldgop p d addr off =>
      refine ⟨?_, rfl, rfl⟩
      simp only [sstepInstr, SState.setReg, SState.setPc]
      funext x
      by_cases hx : x = d
      · subst hx
        funext l
        simp only [if_pos rfl, if_true]
        by_cases hp : st.regs p l == 1
        · rw [if_pos hp, if_pos hp, hread _ ⟨l, by simpa using hp, rfl⟩]
        · rw [if_neg hp, if_neg hp]
      · simp only [if_neg hx]
  | _ => exact ⟨rfl, rfl, rfl⟩

/-- **The swap is preserved.**  Two memories of equal size agreeing at `j` still
    agree at `j` after the step: what a store writes is a function of registers,
    which the swap leaves alone. -/
theorem sstepInstr_gmem_congr (prog : Array SInstr) (i : SInstr) (st : SState)
    (g : Array UInt8) (j : Nat) (hsize : g.size = st.gmem.size)
    (hj : g.getD j 0 = st.gmem.getD j 0) :
    (sstepInstr prog i { st with gmem := g }).gmem.getD j 0
      = (sstepInstr prog i st).gmem.getD j 0 := by
  cases i with
  | stg addr s =>
      simp only [sstepInstr]
      exact storeBytes_agree _ _ _ _ _ j hsize hj
  | stgp p addr s =>
      simp only [sstepInstr]
      exact storeBytes_agree _ _ _ _ _ j hsize hj
  | stg32p p addr s =>
      simp only [sstepInstr]
      exact (storeBytes_agree' _ _ _ _ _ j (storeBytes_agree' _ _ _ _ _ j
        (storeBytes_agree' _ _ _ _ _ j (storeBytes_agree' _ _ _ _ _ j ⟨hsize, hj⟩)))).2
  | _ => simp only [sstepInstr, SState.setReg, SState.setPc]; exact hj

-- ── The concurrent machine ────────────────────────────────────────────────────

/-- A warp's state minus global memory: what is private to it. -/
structure LState where
  regs : String → Lane → UInt64
  smem : Array UInt8
  pc   : Nat

def SState.loc (st : SState) : LState := ⟨st.regs, st.smem, st.pc⟩

def LState.withMem (l : LState) (g : Array UInt8) : SState :=
  { regs := l.regs, gmem := g, smem := l.smem, pc := l.pc }

/-- Many warps, one memory.  A step picks a warp and runs one instruction of it
    against the shared memory — the sequentially-consistent interleaving model. -/
structure CState (n : Nat) where
  locals : Fin n → LState
  gmem   : Array UInt8

/-- Pointwise update; `Function.update` lives in Mathlib, which this development
    does not depend on. -/
def upd {n : Nat} (f : Fin n → LState) (w : Fin n) (v : LState) : Fin n → LState :=
  fun x => if x = w then v else f x

@[simp] theorem upd_self {n : Nat} (f : Fin n → LState) (w : Fin n) (v : LState) :
    upd f w v w = v := by simp [upd]

@[simp] theorem upd_other {n : Nat} (f : Fin n → LState) (w w' : Fin n) (v : LState)
    (h : w' ≠ w) : upd f w v w' = f w' := by simp [upd, h]

def cstep (prog : Array SInstr) {n : Nat} (cs : CState n) (w : Fin n) : CState n :=
  let st' := sstep prog ((cs.locals w).withMem cs.gmem)
  { locals := upd cs.locals w st'.loc, gmem := st'.gmem }

/-- Run a schedule: any list of warp choices at all. -/
def crun (prog : Array SInstr) {n : Nat} : List (Fin n) → CState n → CState n
  | [], cs => cs
  | w :: ws, cs => crun prog ws (cstep prog cs w)

/-- A warp on its own, `k` steps in. -/
def siter (prog : Array SInstr) : Nat → SState → SState
  | 0, st => st
  | k + 1, st => siter prog k (sstep prog st)

theorem siter_succ (prog : Array SInstr) (k : Nat) (st : SState) :
    siter prog (k + 1) st = sstep prog (siter prog k st) := by
  induction k generalizing st with
  | zero => rfl
  | succ m ih => rw [siter, ih (sstep prog st)]; rfl

theorem siter_gmem_size (prog : Array SInstr) (k : Nat) (st : SState) :
    (siter prog k st).gmem.size = st.gmem.size := by
  induction k generalizing st with
  | zero => rfl
  | succ m ih => rw [siter, ih (sstep prog st), sstep_gmem_size]


-- ── Lifting the step lemmas to `sstep` ────────────────────────────────────────

theorem withMem_loc (st : SState) (g : Array UInt8) :
    st.loc.withMem g = { st with gmem := g } := rfl

/-- `Reads`/`Writes` look only at registers and the pc, so swapping the memory
    cannot change which addresses a step touches. -/
theorem Reads_withMem (prog : Array SInstr) (st : SState) (g : Array UInt8) (j : Nat) :
    Reads prog { st with gmem := g } j ↔ Reads prog st j := Iff.rfl

theorem Writes_withMem (prog : Array SInstr) (st : SState) (g : Array UInt8) (j : Nat) :
    Writes prog { st with gmem := g } j ↔ Writes prog st j := Iff.rfl

theorem WritesAct_withMem (prog : Array SInstr) (st : SState) (g : Array UInt8) (j : Nat) :
    WritesAct prog { st with gmem := g } j ↔ WritesAct prog st j := Iff.rfl

/-- `ret` is a fixpoint of `sstepInstr` too, so the machine is just "run the
    instruction at the pc" — a form with one case instead of three. -/
theorem sstep_eq_instr (prog : Array SInstr) (st : SState) :
    sstep prog st = match prog[st.pc]? with
      | none => st
      | some i => sstepInstr prog i st := by
  show (match prog[st.pc]? with
      | none => st | some .ret => st | some i => sstepInstr prog i st) = _
  cases prog[st.pc]? with
  | none => rfl
  | some i => cases i <;> rfl

theorem sstep_local_congr (prog : Array SInstr) (st : SState) (g : Array UInt8)
    (hread : ∀ j, Reads prog st j → g.getD j 0 = st.gmem.getD j 0) :
    (sstep prog { st with gmem := g }).loc = (sstep prog st).loc := by
  simp only [Reads] at hread
  rw [sstep_eq_instr, sstep_eq_instr]
  show (match prog[st.pc]? with
      | none => ({ st with gmem := g } : SState)
      | some i => sstepInstr prog i { st with gmem := g }).loc = _
  cases hi : prog[st.pc]? with
  | none => rfl
  | some i =>
      simp only [hi] at hread
      obtain ⟨h1, h2, h3⟩ := sstepInstr_local_congr prog i st g hread
      simp only [SState.loc, h1, h2, h3]

theorem sstep_gmem_congr (prog : Array SInstr) (st : SState) (g : Array UInt8) (j : Nat)
    (hsize : g.size = st.gmem.size) (hj : g.getD j 0 = st.gmem.getD j 0) :
    (sstep prog { st with gmem := g }).gmem.getD j 0 = (sstep prog st).gmem.getD j 0 := by
  rw [sstep_eq_instr, sstep_eq_instr]
  show (match prog[st.pc]? with
      | none => ({ st with gmem := g } : SState)
      | some i => sstepInstr prog i { st with gmem := g }).gmem.getD j 0 = _
  cases hi : prog[st.pc]? with
  | none => exact hj
  | some i => exact sstepInstr_gmem_congr prog i st g j hsize hj

-- ── Connecting to `SReaches`, and running past the end ────────────────────────

/-- `SReaches` and `siter` are the same recursion. -/
theorem sreaches_iff_siter (prog : Array SInstr) :
    ∀ (k : Nat) (st st' : SState), SReaches prog k st st' ↔ siter prog k st = st' := by
  intro k
  induction k with
  | zero => intro st st'; exact Iff.rfl
  | succ m ih => intro st st'; exact ih (sstep prog st) st'

/-- A state the machine cannot leave — `ret`, or a pc past the end. -/
def Halted (prog : Array SInstr) (st : SState) : Prop := sstep prog st = st

theorem siter_halted (prog : Array SInstr) (st : SState) (h : Halted prog st) :
    ∀ k, siter prog k st = st := by
  intro k
  induction k with
  | zero => rfl
  | succ m ih => rw [siter, h, ih]

/-- Once a warp has finished, giving it more steps changes nothing — so a
    schedule only has to be *long enough*, not exact. -/
theorem siter_of_halted_ge (prog : Array SInstr) (st st' : SState) (n : Nat)
    (hreach : siter prog n st = st') (hhalt : Halted prog st') :
    ∀ m, n ≤ m → siter prog m st = st' := by
  intro m hm
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hm
  clear hm
  induction d generalizing st' with
  | zero => rw [Nat.add_zero]; exact hreach
  | succ e ih =>
      show siter prog (n + e + 1) st = st'
      rw [siter_succ, ih st' hreach hhalt, hhalt]

-- ── Race-freedom, and what an interleaving computes ───────────────────────────

section Interleaving

variable {n : Nat}

/-- Warp `w` on its own, `k` steps in, from the shared initial memory. -/
def solo (prog : Array SInstr) (init : Fin n → LState) (gm : Array UInt8)
    (w : Fin n) (k : Nat) : SState :=
  siter prog k ((init w).withMem gm)

/-- **Race-freedom**, as the three facts an interleaving argument needs.  Each is
    a statement about the warps running ALONE, which is the only thing the
    single-warp machine can talk about — and, for this kernel, each is either
    already proven (`writes`, from the frame clause of `ShippedCorrect`;
    `disjoint`, from `warp_regions_disjoint`) or the one remaining kernel-level
    obligation (`reads`). -/
structure RaceFree (prog : Array SInstr) (init : Fin n → LState) (gm : Array UInt8)
    (R : Fin n → Nat → Prop) : Prop where
  disjoint : ∀ (w w' : Fin n) (j : Nat), w ≠ w' → R w j → ¬ R w' j
  writes : ∀ (w : Fin n) (k j : Nat), WritesAct prog (solo prog init gm w k) j → R w j
  reads : ∀ (w w' : Fin n) (k j : Nat), w ≠ w' →
    Reads prog (solo prog init gm w k) j → ¬ R w' j

variable {prog : Array SInstr} {init : Fin n → LState} {gm : Array UInt8}
  {R : Fin n → Nat → Prop}

/-- A warp alone changes nothing outside its own region. -/
theorem solo_frame (hrf : RaceFree prog init gm R) (w : Fin n) (k j : Nat)
    (hj : ¬ R w j) :
    (solo prog init gm w k).gmem.getD j 0 = gm.getD j 0 := by
  induction k with
  | zero => rfl
  | succ m ih =>
      rw [solo, siter_succ, ← solo, sstep_gmem_frame', ih]
      exact fun hw => hj (hrf.writes w m j hw)

theorem solo_size (w : Fin n) (k : Nat) :
    (solo prog init gm w k).gmem.size = gm.size := by
  rw [solo, siter_gmem_size]; rfl

/-- **The invariant.**  The concurrent memory agrees with each warp's solo memory
    on that warp's region, agrees with the initial memory everywhere outside all
    regions, and each warp's local state is exactly its solo local state at the
    number of steps the schedule has given it. -/
def Sim (prog : Array SInstr) (init : Fin n → LState) (gm : Array UInt8)
    (R : Fin n → Nat → Prop) (cs : CState n) (cnt : Fin n → Nat) : Prop :=
  cs.gmem.size = gm.size ∧
  (∀ w, cs.locals w = (solo prog init gm w (cnt w)).loc) ∧
  (∀ w j, R w j → cs.gmem.getD j 0 = (solo prog init gm w (cnt w)).gmem.getD j 0) ∧
  (∀ j, (∀ w, ¬ R w j) → cs.gmem.getD j 0 = gm.getD j 0)

/-- The initial concurrent state satisfies it, with every count at zero. -/
theorem sim_init (prog : Array SInstr) (init : Fin n → LState) (gm : Array UInt8)
    (R : Fin n → Nat → Prop) :
    Sim prog init gm R ⟨init, gm⟩ (fun _ => 0) :=
  ⟨rfl, fun _ => rfl, fun _ _ _ => rfl, fun _ _ => rfl⟩


/-- **The invariant survives a scheduling step.**  Whichever warp the scheduler
    picks, the concurrent step reproduces exactly that warp's next solo step:
    the values it reads are the values it would have read alone (its own region
    agrees by the invariant, everything else is untouched by anyone), and the
    values it writes land only in its own region, where no other warp is looking. -/
theorem sim_step (hrf : RaceFree prog init gm R) (cs : CState n) (cnt : Fin n → Nat)
    (h : Sim prog init gm R cs cnt) (w : Fin n) :
    Sim prog init gm R (cstep prog cs w) (fun x => if x = w then cnt x + 1 else cnt x) := by
  obtain ⟨hsize, hloc, hreg, hout⟩ := h
  -- the solo state this step mirrors
  have hcs : (cs.locals w).withMem cs.gmem = { (solo prog init gm w (cnt w)) with gmem := cs.gmem } := by
    rw [hloc w, withMem_loc]
  have hsz : cs.gmem.size = (solo prog init gm w (cnt w)).gmem.size := by rw [hsize, solo_size]
  -- every address this step reads holds the same byte in both memories
  have hread : ∀ j, Reads prog (solo prog init gm w (cnt w)) j → cs.gmem.getD j 0 = (solo prog init gm w (cnt w)).gmem.getD j 0 := by
    intro j hj
    by_cases hR : R w j
    · exact hreg w j hR
    · have hall : ∀ x : Fin n, ¬ R x j := by
        intro x
        by_cases hx : x = w
        · rw [hx]; exact hR
        · exact hrf.reads w x (cnt w) j (fun e => hx e.symm) hj
      rw [hout j hall, solo_frame hrf w (cnt w) j hR]
  -- and every address it writes lies in its own region
  have hwrite : ∀ j, ¬ R w j →
      ¬ WritesAct prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem } j := by
    intro j hR hw
    exact hR (hrf.writes w (cnt w) j
      ((WritesAct_withMem prog (solo prog init gm w (cnt w)) cs.gmem j).mp hw))
  have hstep : cstep prog cs w =
      ⟨upd cs.locals w (sstep prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem }).loc,
       (sstep prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem }).gmem⟩ := by
    rw [cstep, hcs]
  have hnext : solo prog init gm w (cnt w + 1) = sstep prog (solo prog init gm w (cnt w)) :=
    siter_succ prog (cnt w) ((init w).withMem gm)
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hstep]
    show (sstep prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem }).gmem.size = gm.size
    rw [sstep_gmem_size]; exact hsize
  · intro x
    dsimp only
    by_cases hx : x = w
    · subst hx
      rw [hstep]
      show upd cs.locals x _ x = _
      rw [upd_self, if_pos rfl, hnext]
      exact sstep_local_congr prog (solo prog init gm x (cnt x)) cs.gmem hread
    · rw [hstep]
      show upd cs.locals w _ x = _
      rw [upd_other _ _ _ _ hx, if_neg hx]
      exact hloc x
  · intro x j hRx
    dsimp only
    by_cases hx : x = w
    · subst hx
      rw [hstep, if_pos rfl, hnext]
      show (sstep prog { (solo prog init gm x (cnt x)) with gmem := cs.gmem }).gmem.getD j 0 = _
      exact sstep_gmem_congr prog (solo prog init gm x (cnt x)) cs.gmem j hsz (hreg x j hRx)
    · rw [hstep, if_neg hx]
      show (sstep prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem }).gmem.getD j 0 = _
      rw [sstep_gmem_frame']
      · exact hreg x j hRx
      · exact hwrite j (hrf.disjoint x w j (fun e => hx e) hRx)
  · intro j hall
    rw [hstep]
    show (sstep prog { (solo prog init gm w (cnt w)) with gmem := cs.gmem }).gmem.getD j 0 = _
    rw [sstep_gmem_frame']
    · exact hout j hall
    · exact hwrite j (hall w)

/-- How many steps a schedule gives a warp. -/
def schedCount {n : Nat} (sched : List (Fin n)) (w : Fin n) : Nat :=
  sched.foldr (fun x acc => if x = w then acc + 1 else acc) 0

theorem schedCount_cons {n : Nat} (w x : Fin n) (ws : List (Fin n)) :
    schedCount (w :: ws) x = if w = x then schedCount ws x + 1 else schedCount ws x := rfl

/-- **Any schedule at all.**  Running the warps in any order preserves the
    invariant, and each warp has advanced by exactly the number of turns the
    schedule gave it — so "the schedule ran warp `w` to completion" is a
    statement one can actually make. -/
theorem crun_sim (hrf : RaceFree prog init gm R) :
    ∀ (sched : List (Fin n)) (cs : CState n) (cnt : Fin n → Nat),
      Sim prog init gm R cs cnt →
      Sim prog init gm R (crun prog sched cs) (fun x => cnt x + schedCount sched x) := by
  intro sched
  induction sched with
  | nil => intro cs cnt h; exact h
  | cons w ws ih =>
      intro cs cnt h
      have step := sim_step hrf cs cnt h w
      have := ih (cstep prog cs w) (fun x => if x = w then cnt x + 1 else cnt x) step
      have heq : (fun x => (if x = w then cnt x + 1 else cnt x) + schedCount ws x)
          = (fun x => cnt x + schedCount (w :: ws) x) := by
        funext x
        rw [schedCount_cons]
        by_cases hx : x = w
        · rw [if_pos hx, if_pos hx.symm]; omega
        · rw [if_neg hx, if_neg (fun e => hx e.symm)]
      rw [crun, ← heq]
      exact this

/-- The headline, from the initial state. -/
theorem interleaving_agrees (hrf : RaceFree prog init gm R) (sched : List (Fin n)) :
    Sim prog init gm R (crun prog sched ⟨init, gm⟩) (schedCount sched) := by
  have := crun_sim hrf sched ⟨init, gm⟩ (fun _ => 0) (sim_init prog init gm R)
  simpa using this

/-- **What a completed schedule computes.**  If the schedule gives every warp at
    least as many turns as it needs to halt, then the memory it leaves is: each
    warp's own solo result on that warp's region, and the initial memory
    everywhere else.

    This is `LaunchAgreesPerWarp` and `LaunchFrame` together, derived from
    race-freedom rather than assumed about the hardware.  What is left to the
    platform is only that the hardware realises *some* interleaving, which is
    PTX's documented DRF-SC guarantee. -/
theorem schedule_completes (hrf : RaceFree prog init gm R) (sched : List (Fin n))
    (fin : Fin n → SState) (need : Fin n → Nat)
    (hreach : ∀ w, siter prog (need w) ((init w).withMem gm) = fin w)
    (hhalt : ∀ w, Halted prog (fin w))
    (hlong : ∀ w, need w ≤ schedCount sched w) :
    (∀ w j, R w j → (crun prog sched ⟨init, gm⟩).gmem.getD j 0 = (fin w).gmem.getD j 0) ∧
    (∀ j, (∀ w, ¬ R w j) → (crun prog sched ⟨init, gm⟩).gmem.getD j 0 = gm.getD j 0) ∧
    (crun prog sched ⟨init, gm⟩).gmem.size = gm.size := by
  obtain ⟨hs, _, hreg, hout⟩ := interleaving_agrees hrf sched
  refine ⟨?_, hout, hs⟩
  intro w j hR
  rw [hreg w j hR]
  have : solo prog init gm w (schedCount sched w) = fin w :=
    siter_of_halted_ge prog ((init w).withMem gm) (fin w) (need w)
      (hreach w) (hhalt w) _ (hlong w)
  rw [this]

end Interleaving


end AlgorithmLib.LZ4Simt
