import AlgorithmLib.LZ4WarpDSL
import AlgorithmLib.LZ4SimtRSim
set_option maxRecDepth 4096

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt

-- ── Coupling: SIMT state ~ sequential WState over DSL register set `R` ─────────

/-- The machine state `ss` realizes the sequential state `ws`: memory agrees, and
    every DSL register in `R` is warp-uniform with the matching scalar value. -/
def Couple (R : List String) (ss : SState) (ws : WState) : Prop :=
  ss.gmem = ws.gmem ∧ ss.smem = ws.smem ∧
    ∀ r ∈ R, ∀ l : Fin 32, ss.regs r l = ws.regs r

theorem Couple.gmem {R ss ws} (h : Couple R ss ws) : ss.gmem = ws.gmem := h.1
theorem Couple.smem {R ss ws} (h : Couple R ss ws) : ss.smem = ws.smem := h.2.1
theorem Couple.reg {R ss ws} (h : Couple R ss ws) {r} (hr : r ∈ R) (l : Fin 32) :
    ss.regs r l = ws.regs r := h.2.2 r hr l

/-- `WArg` value on the machine (per lane) = its sequential value, when reg args
    are in `R`. -/
theorem warg_eval {R ss ws} (h : Couple R ss ws) (a : WArg)
    (ha : ∀ n, a = .reg n → n ∈ R) (l : Fin 32) :
    ss.get l a.toS = a.eval ws := by
  cases a with
  | reg n => exact h.reg (ha n rfl) l
  | imm v => rfl

-- ── Per-instruction soundness (one `sstep`, cheap) ────────────────────────────

/-- Coupling after setting a single register `d` to a uniform value `v` (on the
    machine, all lanes to `v`; sequentially, `ws.setReg d v`). -/
theorem couple_setReg {R ss ws} (hc : Couple R ss ws) (d : String) (fv : Fin 32 → UInt64)
    (v : UInt64) (hv : ∀ l, fv l = v) (n : Nat) :
    Couple R ((ss.setReg d fv).setPc n) (ws.setReg d v) := by
  refine ⟨hc.1, hc.2.1, ?_⟩
  intro r hr l
  show (if r = d then fv else ss.regs r) l = (if r = d then v else ws.regs r)
  by_cases hrd : r = d
  · simp only [hrd, if_pos rfl]; exact hv l
  · simp only [if_neg hrd]; exact hc.reg hr l

theorem mov_sound (R : List String) (prog : Array SInstr) (d : String) (a : WArg)
    (ss : SState) (ws : WState) (hpc : prog[ss.pc]? = some (.mov d a.toS))
    (hc : Couple R ss ws) (ha : ∀ n, a = .reg n → n ∈ R) :
    Couple R (sstep prog ss) (ws.setReg d (a.eval ws)) ∧ (sstep prog ss).pc = ss.pc + 1 := by
  have hstep : sstep prog ss = (ss.setReg d (fun l => ss.get l a.toS)).setPc (ss.pc + 1) := by
    simp only [sstep, hpc, sstepInstr]
  rw [hstep]
  exact ⟨couple_setReg hc d _ _ (fun l => warg_eval hc a ha l) _, rfl⟩

/-- `sstep` of an arithmetic `bin` preserves the coupling. -/
theorem bin_sound (R : List String) (prog : Array SInstr) (o : WOp) (d a : String) (b : WArg)
    (ss : SState) (ws : WState) (hpc : prog[ss.pc]? = some (.bin o.toS d a b.toS))
    (hc : Couple R ss ws) (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R) :
    Couple R (sstep prog ss) (ws.setReg d (o.run (ws.regs a) (b.eval ws))) ∧
      (sstep prog ss).pc = ss.pc + 1 := by
  have hstep : sstep prog ss =
      (ss.setReg d (fun l => o.toS.run (ss.regs a l) (ss.get l b.toS))).setPc (ss.pc + 1) := by
    simp only [sstep, hpc, sstepInstr]
  rw [hstep]
  refine ⟨couple_setReg hc d _ _ (fun l => ?_) _, rfl⟩
  show o.toS.run (ss.regs a l) (ss.get l b.toS) = o.run (ws.regs a) (b.eval ws)
  rw [hc.reg ha l, warg_eval hc b hb l, WOp.run_eq_toS]

/-- `sstep` of a `setp` (comparison) preserves the coupling. -/
theorem setp_sound (R : List String) (prog : Array SInstr) (c : SCmp) (d a : String) (b : WArg)
    (ss : SState) (ws : WState) (hpc : prog[ss.pc]? = some (.setp c d a b.toS))
    (hc : Couple R ss ws) (ha : a ∈ R) (hb : ∀ n, b = .reg n → n ∈ R) :
    Couple R (sstep prog ss) (ws.setReg d (if c.run (ws.regs a) (b.eval ws) then 1 else 0)) ∧
      (sstep prog ss).pc = ss.pc + 1 := by
  have hstep : sstep prog ss =
      (ss.setReg d (fun l => if c.run (ss.regs a l) (ss.get l b.toS) then 1 else 0)).setPc (ss.pc + 1) := by
    simp only [sstep, hpc, sstepInstr]
  rw [hstep]
  refine ⟨couple_setReg hc d _ _ (fun l => ?_) _, rfl⟩
  show (if c.run (ss.regs a l) (ss.get l b.toS) then 1 else 0)
      = (if c.run (ws.regs a) (b.eval ws) then 1 else 0)
  rw [hc.reg ha l, warg_eval hc b hb l]
theorem foldl_set_const (i : Nat) (v : UInt8) :
    ∀ (l : List (Fin 32)) (mem : Array UInt8),
      l.foldl (fun m _ => m.setIfInBounds i v) (mem.setIfInBounds i v) = mem.setIfInBounds i v := by
  intro l
  induction l with
  | nil => intro mem; rfl
  | cons a t ih =>
      intro mem
      rw [List.foldl_cons, Array.setIfInBounds_setIfInBounds]
      exact ih mem

/-- A cooperative store with all lanes active and a *uniform* address/value (every
    lane writes the same byte to the same slot) collapses to one `setIfInBounds`. -/
theorem storeBytes_uniform (mem : Array UInt8) (addr val : Fin 32 → UInt64) (a0 v0 : UInt64)
    (ha : ∀ l, addr l = a0) (hv : ∀ l, val l = v0) :
    storeBytes mem (fun _ => true) addr val = mem.setIfInBounds a0.toNat v0.toUInt8 := by
  unfold storeBytes
  simp only [ha, hv, if_true, Array.set!_eq_setIfInBounds]
  have hfr : (List.finRange W) = (0 : Fin 32) :: (List.finRange 31).map Fin.succ :=
    List.finRange_succ
  rw [hfr, List.foldl_cons]
  exact foldl_set_const a0.toNat v0.toUInt8 _ mem

/-- `sstep` of a *uniform* global byte store preserves the coupling: all lanes
    write the same byte to the same slot (`addr`/`s` ∈ `R`), collapsing to the
    DSL's single `stgByte` — no per-lane predicate/invariant needed.  (The fully
    uniform kernel makes the header/length write faithful as an all-lane store.) -/
theorem stgU_sound (R : List String) (prog : Array SInstr) (addr s : String)
    (ss : SState) (ws : WState) (hpc : prog[ss.pc]? = some (.stg addr s))
    (hc : Couple R ss ws) (haddr : addr ∈ R) (hs : s ∈ R) :
    Couple R (sstep prog ss) (ws.stgByte (ws.regs addr) (ws.regs s)) ∧
      (sstep prog ss).pc = ss.pc + 1 := by
  have hstep : sstep prog ss =
      { ss with gmem := storeBytes ss.gmem (fun _ => true) (ss.regs addr) (ss.regs s),
                pc := ss.pc + 1 } := by
    simp only [sstep, hpc, sstepInstr]
  rw [hstep]
  refine ⟨⟨?_, hc.smem, ?_⟩, rfl⟩
  · rw [storeBytes_uniform ss.gmem (ss.regs addr) (ss.regs s) (ws.regs addr) (ws.regs s)
          (fun l => hc.reg haddr l) (fun l => hc.reg hs l), hc.gmem]
    simp [WState.stgByte, Array.set!_eq_setIfInBounds]
  · intro r hr l; exact hc.reg hr l

-- ── Control-flow step lemmas (branch/label pc resolution; coupling preserved) ──

/-- `setPc` touches only the pc, so the coupling is preserved. -/
theorem couple_setPc {R ss ws} (hc : Couple R ss ws) (n : Nat) :
    Couple R (ss.setPc n) ws := ⟨hc.1, hc.2.1, hc.2.2⟩

theorem lbl_step (prog : Array SInstr) (ss : SState) (name : String)
    (hpc : prog[ss.pc]? = some (.lbl name)) : sstep prog ss = ss.setPc (ss.pc + 1) := by
  simp only [sstep, hpc, sstepInstr]

theorem bra_step (prog : Array SInstr) (ss : SState) (l : String)
    (hpc : prog[ss.pc]? = some (.bra l)) : sstep prog ss = ss.setPc (sfindLabel prog l) := by
  simp only [sstep, hpc, sstepInstr]

theorem braif_step (prog : Array SInstr) (ss : SState) (p l : String)
    (hpc : prog[ss.pc]? = some (.braif p l)) :
    sstep prog ss = ss.setPc (if ss.regs p 0 == 1 then sfindLabel prog l else ss.pc + 1) := by
  simp only [sstep, hpc, sstepInstr]

theorem braifnot_step (prog : Array SInstr) (ss : SState) (p l : String)
    (hpc : prog[ss.pc]? = some (.braifnot p l)) :
    sstep prog ss = ss.setPc (if ss.regs p 0 == 1 then ss.pc + 1 else sfindLabel prog l) := by
  simp only [sstep, hpc, sstepInstr]

-- ── Composition: straight-line fragments compose through the `SReaches` engine ──

/-- `prog` realizes the instruction list `seg` contiguously starting at `base`. -/
def SegAt (prog : Array SInstr) (base : Nat) (seg : List SInstr) : Prop :=
  ∀ i, i < seg.length → prog[base + i]? = seg[i]?

theorem SegAt.head {prog base seg} (h : SegAt prog base (seg)) (i0 : SInstr)
    (hseg : seg = i0 :: []) : prog[base]? = some i0 := by
  have := h 0 (by rw [hseg]; simp)
  rw [hseg] at this; simpa using this

theorem SegAt.cons {prog base x rest} (h : SegAt prog base (x :: rest)) :
    prog[base]? = some x ∧ SegAt prog (base + 1) rest := by
  refine ⟨?_, ?_⟩
  · have := h 0 (by simp); simpa using this
  · intro i hi
    have hh := h (i + 1) (by simp; omega)
    rw [show base + (i + 1) = base + 1 + i by omega] at hh
    simpa using hh

theorem SegAt.append_left {prog base ea eb} (h : SegAt prog base (ea ++ eb)) :
    SegAt prog base ea := by
  intro i hi
  have hlt : i < (ea ++ eb).length := by rw [List.length_append]; omega
  have := h i hlt
  rwa [List.getElem?_append_left hi] at this

theorem SegAt.append_right {prog base ea eb} (h : SegAt prog base (ea ++ eb)) :
    SegAt prog (base + ea.length) eb := by
  intro j hj
  have hlt : ea.length + j < (ea ++ eb).length := by rw [List.length_append]; omega
  have hh := h (ea.length + j) hlt
  rw [List.getElem?_append_right (by omega)] at hh
  simpa [Nat.add_sub_cancel_left, Nat.add_assoc] using hh

/-- Every label inside `emit` resolves (via the global `sfindLabel` search) to its
    in-segment position `base + k` — the well-formed-layout hypothesis that makes
    the branch targets of `uif`/`uwhile` land where the emitter intends.  Vacuous
    for the label-free straight-line leaves; splits like `SegAt` across `seq`. -/
def LabelsResolve (prog : Array SInstr) (base : Nat) (emit : List SInstr) : Prop :=
  ∀ k name, emit[k]? = some (.lbl name) → sfindLabel prog name = base + k

theorem LabelsResolve.cons {prog base x rest} (h : LabelsResolve prog base (x :: rest)) :
    LabelsResolve prog (base + 1) rest := by
  intro k name hk
  have hh := h (k + 1) name (by simpa using hk)
  rw [hh]; omega

theorem LabelsResolve.append_left {prog base ea eb} (h : LabelsResolve prog base (ea ++ eb)) :
    LabelsResolve prog base ea := by
  intro k name hk
  have hlt : k < ea.length := by
    rcases Nat.lt_or_ge k ea.length with h1 | h1
    · exact h1
    · rw [List.getElem?_eq_none h1] at hk; simp at hk
  exact h k name (by rw [List.getElem?_append_left hlt]; exact hk)

theorem LabelsResolve.append_right {prog base ea eb} (h : LabelsResolve prog base (ea ++ eb)) :
    LabelsResolve prog (base + ea.length) eb := by
  intro k name hk
  have hh := h (ea.length + k) name
    (by rw [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]; exact hk)
  rw [hh, Nat.add_assoc]

-- ── Control: `uif` lowering + soundness ───────────────────────────────────────

/-- Lowering of `uif cond t e` (uniform condition): branch-if-not to the else
    label, then-block, jump past the else, else label, else-block, end label.
    `lElse`/`lEnd` must be globally unique in the assembled program. -/
def uifEmit (cond lElse lEnd : String) (et ee : List SInstr) : List SInstr :=
  .braifnot cond lElse :: (et ++ .bra lEnd :: .lbl lElse :: (ee ++ [.lbl lEnd]))

/-- One forward step from an explicit `sstep` equation. -/
theorem sreaches_one_eq {prog : Array SInstr} {s s' : SState} (h : sstep prog s = s') :
    SReaches prog 1 s s' := h ▸ sreaches_one prog s

theorem uifEmit_length (cond lElse lEnd : String) (et ee : List SInstr) :
    (uifEmit cond lElse lEnd et ee).length = et.length + ee.length + 4 := by
  simp [uifEmit]; omega

-- ── Control: `uwhile` lowering + soundness (induction on fuel) ────────────────

/-- Lowering of `uwhile cond body`: head label, branch-if-not to the end, body,
    jump back to the head, end label.  `lHead`/`lEnd` must be globally unique. -/
def uwhileEmit (cond lHead lEnd : String) (ebody : List SInstr) : List SInstr :=
  .lbl lHead :: .braifnot cond lEnd :: (ebody ++ .bra lHead :: [.lbl lEnd])

theorem uwhileEmit_length (cond lHead lEnd : String) (ebody : List SInstr) :
    (uwhileEmit cond lHead lEnd ebody).length = ebody.length + 4 := by
  simp [uwhileEmit]

/-- The loop's guard becomes false within `fuel` iterations (so the fuel-bounded
    `eval` has actually terminated — the machine, which runs to completion, then
    matches it).  Mirrors `eval`'s recursion exactly. -/
def WhileHalts (cond : String) (body : WStmt) : Nat → WState → Prop
  | 0, _ => False
  | fuel + 1, ws =>
      if (ws.regs cond == 1) then WhileHalts cond body fuel (body.eval fuel ws) else True

end AlgorithmLib.LZ4WarpDSL
