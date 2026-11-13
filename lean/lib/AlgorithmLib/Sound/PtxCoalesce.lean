module
public import AlgorithmLib.Kernel.PtxRun
meta import AlgorithmLib.Kernel.PtxRun
public import AlgorithmLib.Kernel.Coalesce
meta import AlgorithmLib.Kernel.Coalesce
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Sound.PtxCoalesce` — merged registers run every thread to the same world

`Coalesce.run` renames an entry's registers and turns the moves the renaming
leaves copying a register to itself into labels, where its check passes. This
proves that a thread of the renamed entry ends as a thread of the entry does:
in the same world, or with the same error.

A run of the renamed body steps with the run of the body, line for line. At
each line every register live into it (by the liveness the check accepted)
holds, under its new name, the value it holds in the original run; a line's
reads are live, so it reads the same values and writes the same value, under
the written register's new name. A register sharing that name is dead after
the line, by the check's interference rule, so nothing live is lost. A move
turned into a label wrote its source masked to the move's width; the source
holds a value within that width already (`bounds`: every line writing it
writes within its result width, `resWidth`), so the move wrote what the
shared register holds.
-/

namespace AlgorithmLib.Kernel.PtxCoalesce

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Kernel AlgorithmLib.PtxIr
  AlgorithmLib.PtxRun AlgorithmLib.Kernel.Coalesce

/-! ## Renaming and reading -/

/-- A register is not a special one. -/
def Plain (r : Reg) : Prop := ∀ s, r ≠ .special s

theorem ren_special (ρ : Ren) (s : String) : Reg.ren ρ (.special s) = .special s := rfl

theorem ren_plain (ρ : Ren) (r : Reg) (h : Plain r) : Plain (Reg.ren ρ r) := by
  cases r with
  | special t => exact absurd rfl (h t)
  | _ =>
    intro s he
    simp only [Reg.ren] at he
    split at he <;> first | cases he | exact h s he

/-- What an operand reads, from the registers given. -/
def opndV (c : Ctx) (regs : Std.HashMap Reg UInt64) : Opnd → Except String UInt64
  | .reg (.special n) => .ok ((c.special.lookup n).getD 0)
  | .reg r => .ok ((regs.get? r).getD 0)
  | .imm v => .ok (UInt64.ofInt v)
  | .f32 b => .ok (UInt64.ofNat b)
  | .f64 b => .ok (UInt64.ofNat b)
  | .sym n => match c.syms.lookup n with
    | some v => .ok v
    | none => .error s!"unknown symbol {n}"

theorem opnd_run (c : Ctx) (a : Opnd) (s : TS) : (opnd c a).run s = (opndV c s.regs a).map (·, s) := by
  cases a with
  | reg r =>
      cases r <;> simp [opnd, opndV, getReg, StateT.run, MonadState.get, getThe, MonadStateOf.get, StateT.get, pure,
        StateT.pure, bind, StateT.bind, Except.map, Except.pure, Except.bind]
  | sym n =>
      cases hl : c.syms.lookup n <;>
        simp [opnd, opndV, hl, StateT.run, pure, StateT.pure, throw, throwThe, MonadExceptOf.throw,
          StateT.lift, Except.map, Except.pure, bind, Except.bind]
  | _ => rfl

/-- The registers `U` read in `a` hold, under their new names in `b`, what they
    hold in `a`. -/
def Agree (ρ : Ren) (U : List Reg) (a b : Std.HashMap Reg UInt64) : Prop :=
  ∀ r ∈ U, Plain r → (b.get? (Reg.ren ρ r)).getD 0 = (a.get? r).getD 0

theorem Agree.mono {ρ : Ren} {U V : List Reg} {a b : Std.HashMap Reg UInt64} (h : Agree ρ U a b)
    (hs : ∀ r ∈ V, r ∈ U) : Agree ρ V a b := fun r hr => h r (hs r hr)

/-- A plain register reads its value, zero where it has none. -/
theorem opndV_plain (c : Ctx) (regs : Std.HashMap Reg UInt64) (r : Reg) (h : Plain r) :
    opndV c regs (.reg r) = .ok ((regs.get? r).getD 0) := by
  cases r with
  | special s => exact absurd rfl (h s)
  | _ => rfl

theorem opndV_ren_plain (c : Ctx) (ρ : Ren) (r : Reg) (hp : Plain r) (regs regs' : Std.HashMap Reg UInt64)
    (h : Agree ρ [r] regs regs') :
    opndV c regs' (Coalesce.Opnd.ren ρ (.reg r)) = opndV c regs (.reg r) := by
  simp only [Coalesce.Opnd.ren]
  rw [opndV_plain c regs' _ (ren_plain ρ r hp), opndV_plain c regs r hp, h r (by simp) hp]

theorem opndV_ren (c : Ctx) (ρ : Ren) (a : Opnd) (regs regs' : Std.HashMap Reg UInt64)
    (h : Agree ρ (Coalesce.Opnd.regs a) regs regs') :
    opndV c regs' (Coalesce.Opnd.ren ρ a) = opndV c regs a := by
  cases a with
  | reg r =>
      cases r with
      | special s => rfl
      | val c' n => exact opndV_ren_plain c ρ _ (fun _ h => by cases h) regs regs' h
      | own p t n => exact opndV_ren_plain c ρ _ (fun _ h => by cases h) regs regs' h
      | name n => exact opndV_ren_plain c ρ _ (fun _ h => by cases h) regs regs' h
  | _ => rfl

/-- A load's value, as `load` reads it. -/
def loadV (c : Ctx) (regs : Std.HashMap Reg UInt64) (w : World) (sp : Space) (t : Ty) (ad : Addr) :
    Except String UInt64 :=
  if sp == .param then
    match ad.base with
    | .sym n => match c.params.lookup n with
        | some v => .ok v
        | none => .error s!"unknown parameter {n}"
    | _ => .error "a parameter load through a register"
  else do
    let base ← opndV c regs ad.base
    let a := UInt64.ofInt ((base.toNat : Int) + ad.off)
    match w.mem.load (place sp a) (t.width / 8) with
    | some v => .ok v
    | none => .error s!"a load of {t.width / 8} bytes at {place sp a} reaches nothing"

/-- A store's world, as `store` leaves it. -/
def storeV (c : Ctx) (regs : Std.HashMap Reg UInt64) (w : World) (sp : Space) (t : Ty) (ad : Addr) (v : UInt64) :
    Except String World := do
  let base ← opndV c regs ad.base
  let a := place sp (UInt64.ofInt ((base.toNat : Int) + ad.off))
  let n := t.width / 8
  let v := mask t.width v
  match w.mem.store a n v with
  | some m => .ok { w with mem := m, obs := .store a n v :: w.obs }
  | none => .error s!"a store of {n} bytes at {a} reaches nothing"

/-- Whether an atomic access at `ad` is aligned to its width, as `aligned`
    finds. -/
def alignedV (c : Ctx) (regs : Std.HashMap Reg UInt64) (sp : Space) (t : Ty) (ad : Addr) :
    Except String Unit := do
  let base ← opndV c regs ad.base
  let a := place sp (UInt64.ofInt ((base.toNat : Int) + ad.off))
  if a % UInt64.ofNat (t.width / 8) != 0 then .error s!"an atomic access at {a} not aligned to its width"
  else .ok ()

/-- A collective's answer and world, as `coll` leaves them. -/
def collV (c : Ctx) (w : World) (i : Intr) (args : List V) : Except String (Option V × World) :=
  let k := (collCalls w c.start).length
  let w1 := { w with obs := .call (.intr i) args :: w.obs }
  if i == .barrier then
    match c.ans[k]? with
    | some (.smem b) => .ok (none, { w1 with mem := { w1.mem with arena := b } })
    | _ => if c.provisional then .ok (none, w1) else .error "a barrier no law has answered with shared memory"
  else .ok (some (valAt c.ans k), w1)

/-- **One instruction's step**: what follows, the world after, and the register
    it writes with the value, from the registers and world before. -/
def stepV (c : Ctx) (regs : Std.HashMap Reg UInt64) (w : World) :
    Instr → Except String (Next × World × Option (Reg × UInt64))
  | .mov t d a => do
      let x ← opndV c regs a
      pure (.step, w, some (d, mask t.width x))
  | .bin o _ t d a b => do
      let x ← opndV c regs a; let y ← opndV c regs b
      let v ← bin o t x y
      pure (.step, w, some (d, v))
  | .un o _ t d a => do
      let x ← opndV c regs a
      let v ← un o c.hw t x
      pure (.step, w, some (d, v))
  | .fma (some _) t d a b e => do
      if t != .f32 then .error "fma not at f32" else
      let x ← opndV c regs a; let y ← opndV c regs b; let z ← opndV c regs e
      pure (.step, w, some (d, canon (fmaLane ClifTy.f32 (mask 32 x) (mask 32 y) (mask 32 z))))
  | .fma none t d a b e => do
      let x ← opndV c regs a; let y ← opndV c regs b; let z ← opndV c regs e
      pure (.step, w, some (d, mask t.width (x * y + z)))
  | .cvt r dt sty d a => do
      let x ← opndV c regs a
      let v ← cvt r dt sty x
      pure (.step, w, some (d, v))
  | .cvta to sp _ d a => do
      let v ← opndV c regs a
      if sp != .shared then pure (.step, w, some (d, v))
      else if to then
        match decodeAddr v with
        | some (.arena, off) => pure (.step, w, some (d, UInt64.ofNat off))
        | _ => .error "cvta.to.shared of an address outside shared memory"
      else pure (.step, w, some (d, addrOf .arena (mask 32 v).toNat))
  | .setp cc t p a b => do
      let x ← opndV c regs a; let y ← opndV c regs b
      let r ← cmp cc t x y
      pure (.step, w, some (p, if r then 1 else 0))
  | .selp t d a b p => do
      let pv ← opndV c regs p
      let x ← opndV c regs a; let y ← opndV c regs b
      pure (.step, w, some (d, mask t.width (if pv &&& 1 == 1 then x else y)))
  | .prmt d a b s => do
      let x ← opndV c regs a; let y ← opndV c regs b; let sel ← opndV c regs s
      pure (.step, w, some (d, Intr.prmtVal x y sel))
  | .dp4a d a b e => do
      let x ← opndV c regs a; let y ← opndV c regs b; let z ← opndV c regs e
      pure (.step, w, some (d, Intr.dp4aVal x y z))
  | .ld sp _ t d a => do
      let v ← loadV c regs w sp t a
      pure (.step, w, some (d, v))
  | .st sp t a v => do
      let x ← opndV c regs v
      let w' ← storeV c regs w sp t a x
      pure (.step, w', none)
  | .atom sp o _ t d a v => do
      alignedV c regs sp t a
      match o.rmw?, o, d with
      | some k, _, some d => do
          let old ← loadV c regs w sp t a
          let x ← opndV c regs v
          let w' ← storeV c regs w sp t a (rmwNew k (atomTy t) old x)
          pure (.step, w', some (d, old))
      | none, .load, some d => do
          let old ← loadV c regs w sp t a
          pure (.step, w, some (d, old))
      | none, .store, none => do
          let x ← opndV c regs v
          let w' ← storeV c regs w sp t a x
          pure (.step, w', none)
      | _, _, _ => .error "an atom without its destination, or with one it does not write"
  | .redShared a v => do
      let base ← opndV c regs a.base
      let p := place .shared (UInt64.ofInt ((base.toNat : Int) + a.off))
      let x ← opndV c regs v
      let (_, w') ← collV c w .sharedAdd [.sc .i64 p, .sc .i32 (mask 32 x)]
      pure (.step, w', none)
  | .shfl m d a b cl _ => do
      let k ← opndV c regs cl
      if k != 31 then .error "a shuffle within less than a warp" else
      let i ← match m with
        | .bfly => .ok Intr.shflBfly
        | .idx => .ok .shflIdx
        | _ => .error s!"shfl.{m.name} not modelled"
      let x ← opndV c regs a; let y ← opndV c regs b
      let (r, w') ← collV c w i [.sc .i32 (mask 32 x), .sc .i32 (mask 32 y)]
      match r with
      | some (.sc _ v) => pure (.step, w', some (d, mask 32 v))
      | _ => .error "a shuffle answered by no value"
  | .vote d p _ => do
      let x ← opndV c regs p
      let (r, w') ← collV c w .ballot [.sc .i32 (x &&& 1)]
      match r with
      | some (.sc _ v) => pure (.step, w', some (d, mask 32 v))
      | _ => .error "a ballot answered by no value"
  | .mmaW hi d a0 a1 a2 a3 b0 b1 => do
      let x0 ← opndV c regs a0; let x1 ← opndV c regs a1; let x2 ← opndV c regs a2
      let x3 ← opndV c regs a3; let x4 ← opndV c regs b0; let x5 ← opndV c regs b1
      let (r, w') ← collV c w (if hi then .mmaHi else .mmaLo) [.sc .i32 (mask 32 x0), .sc .i32 (mask 32 x1),
        .sc .i32 (mask 32 x2), .sc .i32 (mask 32 x3), .sc .i32 (mask 32 x4), .sc .i32 (mask 32 x5)]
      match r with
      | some (.sc _ v) => pure (.step, w', some (d, v))
      | _ => .error "a wide mma answered by no value"
  | .mmaB hi d a0 a1 a2 a3 b0 b1 e => do
      let x0 ← opndV c regs a0; let x1 ← opndV c regs a1; let x2 ← opndV c regs a2
      let x3 ← opndV c regs a3; let x4 ← opndV c regs b0; let x5 ← opndV c regs b1; let x6 ← opndV c regs e
      let (r, w') ← collV c w (if hi then .mmaBfHi else .mmaBfLo) [.sc .i32 (mask 32 x0), .sc .i32 (mask 32 x1),
        .sc .i32 (mask 32 x2), .sc .i32 (mask 32 x3), .sc .i32 (mask 32 x4), .sc .i32 (mask 32 x5), .sc .i64 x6]
      match r with
      | some (.sc _ v) => pure (.step, w', some (d, v))
      | _ => .error "a bf16 mma answered by no value"
  -- a line writing two registers is in no renamed body (`Coalesce.fresh`)
  | .mmaB2 .. => .error "an mma writing both halves is not renamed"
  | .bar _ => do
      let (_, w') ← collV c w .barrier []
      pure (.step, w', none)
  | .barWarp _ =>
      if (c.special.lookup "ntid.x").getD 0 ≤ 32 then do
        let (_, w') ← collV c w .barrier []
        pure (.step, w', none)
      else .error "a warp barrier in a block of more than one warp"
  | .bra _ l =>
      match c.labels.get? l with
      | some i => pure (.jump i, w, none)
      | none => .error s!"no label {l.text}"
  | .ret => pure (.done, w, none)

/-- The state a step leaves. -/
def after (ts : TS) : Next × World × Option (Reg × UInt64) → Next × TS
  | (n, w, none) => (n, { ts with w })
  | (n, w, some (d, v)) => (n, { regs := ts.regs.insert d v, w })


theorem run_throw {α} (e : String) (s : TS) : (throw e : M α).run s = .error e := rfl
theorem run_pure {α} (x : α) (s : TS) : (pure x : M α).run s = .ok (x, s) := rfl
theorem run_get (s : TS) : (MonadState.get : M TS).run s = .ok (s, s) := rfl
theorem run_set (s s' : TS) : (set s' : M Unit).run s = .ok ((), s') := rfl
theorem run_bind' {α β} (x : M α) (f : α → M β) (s : TS) :
    (x >>= f).run s = match x.run s with
      | .error e => .error e
      | .ok p => (f p.1).run p.2 := by
  rw [StateT.run_bind]; cases x.run s <;> rfl

theorem load_run (c : Ctx) (sp : Space) (t : Ty) (ad : Addr) (s : TS) :
    (load c sp t ad).run s = (loadV c s.regs s.w sp t ad).map (·, s) := by
  by_cases hp : sp == .param
  · simp only [load, loadV, hp, if_true]
    cases hb : ad.base with
    | sym n => cases hl : c.params.lookup n <;> simp only [hl] <;> rfl
    | _ => rfl
  · simp only [load, loadV, hp, if_false, Bool.false_eq_true, run_bind', opnd_run]
    cases opndV c s.regs ad.base with
    | error e => rfl
    | ok x =>
        simp only [Except.map, run_bind', run_get, bind, Except.bind]
        cases hm : s.w.mem.load (place sp (UInt64.ofInt ((x.toNat : Int) + ad.off))) (t.width / 8) <;>
          simp only [hm] <;> rfl

theorem store_run (c : Ctx) (sp : Space) (t : Ty) (ad : Addr) (v : UInt64) (s : TS) :
    (store c sp t ad v).run s = (storeV c s.regs s.w sp t ad v).map (fun w => ((), { s with w })) := by
  simp only [store, storeV, run_bind', opnd_run]
  cases opndV c s.regs ad.base with
  | error e => rfl
  | ok x =>
      simp only [Except.map, run_bind', run_get, bind, Except.bind]
      cases hm : s.w.mem.store (place sp (UInt64.ofInt ((x.toNat : Int) + ad.off))) (t.width / 8)
          (mask t.width v) <;> simp only [hm] <;> rfl

theorem aligned_run (c : Ctx) (sp : Space) (t : Ty) (ad : Addr) (s : TS) :
    (aligned c sp t ad).run s = (alignedV c s.regs sp t ad).map (·, s) := by
  simp only [aligned, alignedV, run_bind', opnd_run]
  cases opndV c s.regs ad.base with
  | error e => rfl
  | ok x =>
      simp only [Except.map, bind, Except.bind]
      split <;> rfl

theorem coll_run (c : Ctx) (i : Intr) (args : List V) (s : TS) :
    (coll c i args).run s = (collV c s.w i args).map (fun (r, w) => (r, { s with w })) := by
  simp only [coll, collV, run_bind', run_get, run_set]
  by_cases hb : i == .barrier
  · simp only [hb, if_true]
    cases ha : c.ans[(collCalls s.w c.start).length]? with
    | none => cases hpv : c.provisional <;> simp only [ha, hpv] <;> rfl
    | some a => cases a <;> cases hpv : c.provisional <;> simp only [ha, hpv] <;> rfl
  · simp only [hb, if_false, Bool.false_eq_true]
    rfl


theorem lift_run {α} (e : Except String α) (s : TS) : (lift e).run s = e.map (·, s) := by
  cases e <;> rfl
theorem run_setReg (d : Reg) (v : UInt64) (s : TS) :
    (setReg d v).run s = .ok ((), { s with regs := s.regs.insert d v }) := rfl

theorem run_sbind {α β} (x : M α) (f : α → M β) (s : TS) :
    (StateT.bind x f).run s = match x.run s with
      | .error e => .error e
      | .ok p => (f p.1).run p.2 := run_bind' x f s
theorem run_spure {α} (x : α) (s : TS) : (StateT.pure x : M α).run s = .ok (x, s) := rfl

-- Reduce a run's binds, operand reads and register writes.
local macro "norm" : tactic => `(tactic| (try simp only [run_bind', run_sbind, run_spure, opnd_run, lift_run, load_run,
  store_run, coll_run, run_setReg, run_pure, run_throw, after, Except.map, bind, Except.bind, pure, Except.pure]))

-- Split on a read and reduce.
local macro "rd " x:term : tactic => `(tactic| (cases hh : $x <;> (try simp only [hh]) <;> norm))

/-- **An instruction's run is its step**, where it writes one register at
    most. -/
theorem exec_run (c : Ctx) (i : Instr) (ts : TS) (h2 : Coalesce.Instr.two i = false) :
    (exec c i).run ts = (stepV c ts.regs ts.w i).map (after ts) := by
  cases i with
  | mov t d a => simp only [exec, stepV]; norm; rd opndV c ts.regs a
  | bin o r t d a b =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd opndV c ts.regs b
      rd bin o t _ _
  | un o r t d a =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd un o c.hw t _
  | fma r t d a b e =>
      cases r with
      | none =>
          simp only [exec, stepV]; norm
          rd opndV c ts.regs a
          rd opndV c ts.regs b
          rd opndV c ts.regs e
      | some rr =>
          by_cases ht : t != .f32
          · simp only [exec, stepV, ht, if_true]; norm
          · simp only [exec, stepV, ht, if_false, Bool.false_eq_true]; norm
            rd opndV c ts.regs a
            rd opndV c ts.regs b
            rd opndV c ts.regs e
  | cvt r dt st d a =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd cvt r dt st _
  | cvta to sp t d a =>
      simp only [exec, stepV]; norm
      cases hh : opndV c ts.regs a with
      | error e => simp only [hh]; norm
      | ok x =>
        simp only [hh]; norm
        by_cases h1 : sp != .shared
        · simp only [h1, if_true]; norm
        · simp only [h1, if_false, Bool.false_eq_true]
          cases to
          · simp only [Bool.false_eq_true, if_false]; norm
          · simp only [if_true]
            cases hd : decodeAddr x with
            | none => simp only [hd]; norm
            | some p => obtain ⟨rg, off⟩ := p; cases rg <;> simp only [hd] <;> norm
  | setp cc t p a b =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd opndV c ts.regs b
      rd cmp cc t _ _
  | selp t d a b p =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs p
      rd opndV c ts.regs a
      rd opndV c ts.regs b
  | prmt d a b x =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd opndV c ts.regs b
      rd opndV c ts.regs x
  | dp4a d a b x =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a
      rd opndV c ts.regs b
      rd opndV c ts.regs x
  | ld sp nc t d a =>
      simp only [exec, stepV]; norm
      rd loadV c ts.regs ts.w sp t a
  | st sp t a v =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs v
      cases hh : storeV c ts.regs ts.w sp t a _ <;> simp only [hh] <;> norm
  | atom sp o od t d a v =>
      simp only [exec, stepV, PtxRun.atom]; norm
      simp only [aligned_run]; norm
      rd alignedV c ts.regs sp t a
      cases o <;> cases d <;> simp only [AtomOp.rmw?] <;> norm
      all_goals first
        | done
        | (rd loadV c ts.regs ts.w sp t a
           rd opndV c ts.regs v
           cases hh : storeV c ts.regs ts.w sp t a _ <;> simp only [hh] <;> norm
           done)
        | (rd loadV c ts.regs ts.w sp t a; done)
        | (rd opndV c ts.regs v
           cases hh : storeV c ts.regs ts.w sp t a _ <;> simp only [hh] <;> norm
           done)
  | redShared a v =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a.base
      rd opndV c ts.regs v
      cases hc : collV c ts.w _ _ <;> simp only [hc] <;> norm
  | shfl m d a b cl k =>
      simp only [exec, stepV]; norm
      cases hh : opndV c ts.regs cl with
      | error e => simp only [hh]; norm
      | ok kk =>
        simp only [hh]; norm
        by_cases h31 : kk != 31
        · simp only [h31, if_true]; norm
        · simp only [h31, if_false, Bool.false_eq_true]
          cases m <;> norm
          all_goals
            rd opndV c ts.regs a
            rd opndV c ts.regs b
            cases hc : collV c ts.w _ _ with
            | error e => simp only [hc]; norm
            | ok rw =>
              obtain ⟨r, w'⟩ := rw
              simp only [hc]; norm
              cases r with
              | none => norm
              | some v => cases v <;> norm
  | mmaW hi d a0 a1 a2 a3 b0 b1 =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs a0
      rd opndV c ts.regs a1
      rd opndV c ts.regs a2
      rd opndV c ts.regs a3
      rd opndV c ts.regs b0
      rd opndV c ts.regs b1
      cases hc : collV c ts.w _ _ with
      | error e => simp only [hc]; norm
      | ok rw =>
        obtain ⟨r, w'⟩ := rw
        simp only [hc]; norm
        cases r with
        | none => norm
        | some v => cases v <;> norm
  | mmaB hi d a0 a1 a2 a3 b0 b1 e =>
      simp only [exec, mmaBStep, stepV]; norm
      rd opndV c ts.regs a0
      rd opndV c ts.regs a1
      rd opndV c ts.regs a2
      rd opndV c ts.regs a3
      rd opndV c ts.regs b0
      rd opndV c ts.regs b1
      rd opndV c ts.regs e
      cases hc : collV c ts.w _ _ with
      | error e => simp only [hc]; norm
      | ok rw =>
        obtain ⟨r, w'⟩ := rw
        simp only [hc]; norm
        cases r with
        | none => norm
        | some v => cases v <;> norm
  | vote d p k =>
      simp only [exec, stepV]; norm
      rd opndV c ts.regs p
      cases hc : collV c ts.w _ _ with
      | error e => simp only [hc]; norm
      | ok rw =>
        obtain ⟨r, w'⟩ := rw
        simp only [hc]; norm
        cases r with
        | none => norm
        | some v => cases v <;> norm
  | bar n =>
      simp only [exec, stepV]; norm
      cases hc : collV c ts.w _ _ <;> simp only [hc] <;> norm
  | barWarp k =>
      by_cases h : (c.special.lookup "ntid.x").getD 0 ≤ 32
      · simp only [exec, stepV, h, if_true]; norm
        cases hc : collV c ts.w _ _ <;> simp only [hc] <;> norm
      · simp only [exec, stepV, h, if_false]; norm
  | bra u l =>
      simp only [exec, stepV]
      cases hl : c.labels.get? l <;> simp only [hl] <;> norm
  | ret => rfl
  | mmaB2 dl dh a0 a1 a2 a3 b0 b1 cl ch => simp [Coalesce.Instr.two] at h2


/-! ## The renamed step -/

/-- A context the same but for its body and labels. -/
abbrev withBody (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) : Ctx := { c with body := b, labels := L }

theorem opndV_with (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (regs : Std.HashMap Reg UInt64) (a : Opnd) :
    opndV (withBody c b L) regs a = opndV c regs a := by
  cases a with
  | reg r => cases r <;> rfl
  | sym n => rfl
  | _ => rfl

theorem loadV_ren (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (ρ : Ren) (regs regs' : Std.HashMap Reg UInt64)
    (w : World) (sp : Space) (t : Ty) (a : Addr) (h : Agree ρ (Coalesce.Opnd.regs a.base) regs regs') :
    loadV (withBody c b L) regs' w sp t (Coalesce.Addr.ren ρ a) = loadV c regs w sp t a := by
  simp only [loadV, Coalesce.Addr.ren, opndV_with, opndV_ren c ρ a.base regs regs' h]
  split
  · cases a with | mk base off => cases base <;> rfl
  · rfl

theorem storeV_ren (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (ρ : Ren) (regs regs' : Std.HashMap Reg UInt64)
    (w : World) (sp : Space) (t : Ty) (a : Addr) (v : UInt64) (h : Agree ρ (Coalesce.Opnd.regs a.base) regs regs') :
    storeV (withBody c b L) regs' w sp t (Coalesce.Addr.ren ρ a) v = storeV c regs w sp t a v := by
  simp only [storeV, Coalesce.Addr.ren, opndV_with, opndV_ren c ρ a.base regs regs' h]

theorem alignedV_ren (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (ρ : Ren) (regs regs' : Std.HashMap Reg UInt64)
    (sp : Space) (t : Ty) (a : Addr) (h : Agree ρ (Coalesce.Opnd.regs a.base) regs regs') :
    alignedV (withBody c b L) regs' sp t (Coalesce.Addr.ren ρ a) = alignedV c regs sp t a := by
  simp only [alignedV, Coalesce.Addr.ren, opndV_with, opndV_ren c ρ a.base regs regs' h]

theorem collV_with (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (w : World) (i : Intr) (args : List V) :
    collV (withBody c b L) w i args = collV c w i args := rfl

/-- The register a step writes, renamed. -/
def renOut (ρ : Ren) : Next × World × Option (Reg × UInt64) → Next × World × Option (Reg × UInt64)
  | (n, w, wr) => (n, w, wr.map fun (d, v) => (Reg.ren ρ d, v))

-- An operand's reads agree, from the instruction's.
local macro "ag" h:term : tactic => `(tactic| exact ($h).mono (by intro r hr; simp [Coalesce.Instr.uses, hr]))

-- Rewrite every operand read of the renamed instruction to the original's.
local macro "rw_reads" : tactic => `(tactic| (simp only [Coalesce.Instr.ren, stepV, opndV_with, collV_with]))

/-- **The renamed instruction steps as the instruction does**, writing the same
    value to the renamed register, where the registers it reads agree. -/
theorem stepV_ren (c : Ctx) (b : Array Line) (L : Std.HashMap Lbl Nat) (ρ : Ren) (i : Instr)
    (hl : ∀ u l, i = .bra u l → L.get? l = c.labels.get? l)
    (regs regs' : Std.HashMap Reg UInt64) (w : World) (hag : Agree ρ (Coalesce.Instr.uses i) regs regs') :
    stepV (withBody c b L) regs' w (Coalesce.Instr.ren ρ i) = (stepV c regs w i).map (renOut ρ) := by
  have R (a : Opnd) (ha : Agree ρ (Coalesce.Opnd.regs a) regs regs') :
      opndV c regs' (Coalesce.Opnd.ren ρ a) = opndV c regs a := opndV_ren c ρ a regs regs' ha
  cases i with
  | mov t d a =>
      rw_reads; rw [R a (by ag hag)]
      cases opndV c regs a <;> rfl
  | bin o r t d a b =>
      rw_reads; rw [R a (by ag hag), R b (by ag hag)]
      cases opndV c regs a <;> try rfl
      cases opndV c regs b <;> try rfl
      simp only [bind, Except.bind]
      cases AlgorithmLib.PtxRun.bin o t _ _ <;> rfl
  | un o r t d a =>
      rw_reads; rw [R a (by ag hag)]
      cases opndV c regs a <;> try rfl
      simp only [bind, Except.bind]
      cases AlgorithmLib.PtxRun.un o c.hw t _ <;> rfl
  | fma r t d a b e =>
      cases r with
      | none =>
          rw_reads; rw [R a (by ag hag), R b (by ag hag), R e (by ag hag)]
          cases opndV c regs a <;> try rfl
          cases opndV c regs b <;> try rfl
          cases opndV c regs e <;> rfl
      | some rr =>
          rw_reads; rw [R a (by ag hag), R b (by ag hag), R e (by ag hag)]
          split
          · rfl
          · cases opndV c regs a <;> try rfl
            cases opndV c regs b <;> try rfl
            cases opndV c regs e <;> rfl
  | cvt r dt st d a =>
      rw_reads; rw [R a (by ag hag)]
      cases opndV c regs a <;> try rfl
      simp only [bind, Except.bind]
      cases AlgorithmLib.PtxRun.cvt r dt st _ <;> rfl
  | cvta to sp t d a =>
      rw_reads; rw [R a (by ag hag)]
      cases opndV c regs a <;> try rfl
      simp only [bind, Except.bind]
      split
      · rfl
      · split
        · split <;> rfl
        · rfl
  | setp cc t p a b =>
      rw_reads; rw [R a (by ag hag), R b (by ag hag)]
      cases opndV c regs a <;> try rfl
      cases opndV c regs b <;> try rfl
      simp only [bind, Except.bind]
      cases AlgorithmLib.PtxRun.cmp cc t _ _ <;> rfl
  | selp t d a b p =>
      rw_reads; rw [R a (by ag hag), R b (by ag hag), R p (by ag hag)]
      cases opndV c regs p <;> try rfl
      cases opndV c regs a <;> try rfl
      cases opndV c regs b <;> rfl
  | prmt d a b x =>
      rw_reads; rw [R a (by ag hag), R b (by ag hag), R x (by ag hag)]
      cases opndV c regs a <;> try rfl
      cases opndV c regs b <;> try rfl
      cases opndV c regs x <;> rfl
  | dp4a d a b x =>
      rw_reads; rw [R a (by ag hag), R b (by ag hag), R x (by ag hag)]
      cases opndV c regs a <;> try rfl
      cases opndV c regs b <;> try rfl
      cases opndV c regs x <;> rfl
  | ld sp nc t d a =>
      rw_reads; rw [loadV_ren c b L ρ regs regs' w sp t a (by ag hag)]
      cases loadV c regs w sp t a <;> rfl
  | st sp t a v =>
      rw_reads; rw [R v (by ag hag)]
      cases opndV c regs v <;> try rfl
      simp only [bind, Except.bind]
      rw [storeV_ren c b L ρ regs regs' w sp t a _ (by ag hag)]
      cases storeV c regs w sp t a _ <;> rfl
  | atom sp o od t d a v =>
      rw_reads
      rw [alignedV_ren c b L ρ regs regs' sp t a (by ag hag)]
      cases alignedV c regs sp t a <;> try rfl
      simp only [bind, Except.bind]
      cases o <;> cases d <;> simp only [AtomOp.rmw?, Option.map] <;> try rfl
      all_goals
        first
        | (rw [loadV_ren c b L ρ regs regs' w sp t a (by ag hag), R v (by ag hag)]
           cases loadV c regs w sp t a <;> try rfl
           cases opndV c regs v <;> try rfl
           simp only [bind, Except.bind]
           rw [storeV_ren c b L ρ regs regs' w sp t a _ (by ag hag)]
           cases storeV c regs w sp t a _ <;> rfl)
        | (rw [loadV_ren c b L ρ regs regs' w sp t a (by ag hag)]
           cases loadV c regs w sp t a <;> rfl)
        | (rw [R v (by ag hag)]
           cases opndV c regs v <;> try rfl
           simp only [bind, Except.bind]
           rw [storeV_ren c b L ρ regs regs' w sp t a _ (by ag hag)]
           cases storeV c regs w sp t a _ <;> rfl)
  | redShared a v =>
      rw_reads; simp only [Coalesce.Addr.ren]; rw [R a.base (by ag hag), R v (by ag hag)]
      cases opndV c regs a.base <;> try rfl
      cases opndV c regs v <;> rfl
  | shfl m d a b cl k =>
      rw_reads; rw [R cl (by ag hag), R a (by ag hag), R b (by ag hag)]
      cases opndV c regs cl <;> try rfl
      simp only [bind, Except.bind]
      split
      · rfl
      · cases m <;> try rfl
        all_goals
          cases opndV c regs a <;> try rfl
          cases opndV c regs b <;> try rfl
          simp only [bind, Except.bind]
          cases collV c w _ _ <;> try rfl
          rename_i rw
          obtain ⟨r, w'⟩ := rw
          cases r with
          | none => rfl
          | some v => cases v <;> rfl
  | mmaW hi d a0 a1 a2 a3 b0 b1 =>
      rw_reads; rw [R a0 (by ag hag), R a1 (by ag hag), R a2 (by ag hag), R a3 (by ag hag), R b0 (by ag hag),
        R b1 (by ag hag)]
      cases opndV c regs a0 <;> try rfl
      cases opndV c regs a1 <;> try rfl
      cases opndV c regs a2 <;> try rfl
      cases opndV c regs a3 <;> try rfl
      cases opndV c regs b0 <;> try rfl
      cases opndV c regs b1 <;> try rfl
      simp only [bind, Except.bind]
      cases collV c w _ _ <;> try rfl
      rename_i rw
      obtain ⟨r, w'⟩ := rw
      cases r with
      | none => rfl
      | some v => cases v <;> rfl
  | mmaB hi d a0 a1 a2 a3 b0 b1 e =>
      rw_reads; rw [R a0 (by ag hag), R a1 (by ag hag), R a2 (by ag hag), R a3 (by ag hag), R b0 (by ag hag),
        R b1 (by ag hag), R e (by ag hag)]
      cases opndV c regs a0 <;> try rfl
      cases opndV c regs a1 <;> try rfl
      cases opndV c regs a2 <;> try rfl
      cases opndV c regs a3 <;> try rfl
      cases opndV c regs b0 <;> try rfl
      cases opndV c regs b1 <;> try rfl
      cases opndV c regs e <;> try rfl
      simp only [bind, Except.bind]
      cases collV c w _ _ <;> try rfl
      rename_i rw
      obtain ⟨r, w'⟩ := rw
      cases r with
      | none => rfl
      | some v => cases v <;> rfl
  | vote d p k =>
      rw_reads; rw [R p (by ag hag)]
      cases opndV c regs p <;> try rfl
      simp only [bind, Except.bind]
      cases collV c w _ _ <;> try rfl
      rename_i rw
      obtain ⟨r, w'⟩ := rw
      cases r with
      | none => rfl
      | some v => cases v <;> rfl
  | bar n =>
      rw_reads
      cases collV c w _ _ <;> rfl
  | barWarp k =>
      rw_reads
      by_cases h : (c.special.lookup "ntid.x").getD 0 ≤ 32
      · simp only [h, if_true]; cases collV c w _ _ <;> rfl
      · simp only [h, if_false]; rfl
  | bra u l =>
      rw_reads
      rw [hl u l rfl]
      cases c.labels.get? l <;> rfl
  | ret => rfl
  | mmaB2 dl dh a0 a1 a2 a3 b0 b1 cl ch => rfl


/-! ## What a step writes fits its result width -/

theorem width_le (t : Ty) : t.width ≤ 64 := by cases t <;> decide

theorem canon_lt (x : UInt64) (h : x.toNat < 2 ^ 32) : (canon x).toNat < 2 ^ 32 := by
  unfold canon; split
  · decide
  · exact h

theorem pickExtreme_mem (b : Bool) (t : ClifTy) (x y : UInt64) : pickExtreme b t x y = x ∨ pickExtreme b t x y = y := by
  unfold pickExtreme
  split
  · exact .inl rfl
  · split
    · exact .inr rfl
    · split
      · split <;> simp
      · split
        · split <;> simp
        · split <;> simp

theorem lt_of_mem {a x y : UInt64} {n : Nat} (h : a = x ∨ a = y) (hx : x.toNat < n) (hy : y.toNat < n) : a.toNat < n := by
  rcases h with rfl | rfl <;> assumption

theorem binF_lt (o : BinOp) (x y v : UInt64) (hx : x.toNat < 2 ^ 32) (hy : y.toNat < 2 ^ 32)
    (h : binF o x y = .ok v) : v.toNat < 2 ^ 32 := by
  cases o <;> simp only [binF] at h <;> (try cases h) <;>
    first
    | exact canon_lt _ (ofF32_lt _)
    | exact canon_lt _ (lt_of_mem (pickExtreme_mem _ _ x y) hx hy)
    | (apply canon_lt; unfold fext; split
       · exact hy
       · split
         · exact hx
         · exact lt_of_mem (pickExtreme_mem _ _ x y) hx hy)

theorem ofIntW_lt (t : Ty) (i : Int) : (ofIntW t.width i).toNat < 2 ^ t.width := PtxRun.mask_lt t _

theorem shr_lt (t : Ty) (x y : UInt64) : (mask t.width x >>> y).toNat < 2 ^ t.width := by
  rw [UInt64.toNat_shiftRight]
  exact Nat.lt_of_le_of_lt (Nat.shiftRight_le _ _) (PtxRun.mask_lt t x)

theorem binI_lt (o : BinOp) (t : Ty) (x y v : UInt64) (ho : o ≠ .mulWide) (h : binI o t x y = .ok v) :
    v.toNat < 2 ^ t.width := by
  cases o with
  | mulWide => exact absurd rfl ho
  | shl =>
      simp only [binI] at h; cases h
      split
      · simp; exact Nat.two_pow_pos _
      · exact PtxRun.mask_lt t _
  | shr =>
      simp only [binI] at h
      split at h <;> cases h
      · exact ofIntW_lt t _
      · split
        · simp; exact Nat.two_pow_pos _
        · exact shr_lt t x y
  | div | rem =>
      simp only [binI] at h
      split at h
      · cases h
      · cases h; exact ofIntW_lt t _
  | min | max =>
      simp only [binI] at h; cases h
      split <;> exact PtxRun.mask_lt t _
  | _ =>
      simp only [binI] at h <;> (try cases h) <;> first | exact PtxRun.mask_lt t _ | exact ofIntW_lt t _


theorem un_lt (o : UnOp) (hw : Hw) (hap : ∀ k x, hw.approx k x &&& widthMask .f32 = hw.approx k x)
    (x v : UInt64) (ho : (o == .neg || o == .abs || o == .sqrt || o == .ex2Approx || o == .lg2Approx ||
      o == .rcpApprox || o == .rsqrtApprox || o == .sqrtApprox || o == .sinApprox || o == .cosApprox) = true)
    (h : AlgorithmLib.PtxRun.un o hw .f32 x = .ok v) : v.toNat < 2 ^ 32 := by
  cases o <;> simp at ho <;>
    simp only [AlgorithmLib.PtxRun.un] at h <;> (try simp only [show (Ty.f32 == Ty.f32) = true from rfl, if_true] at h) <;> cases h
  · exact canon_lt _ (PtxRun.mask32_toNat_lt _)
  · rw [UInt64.toNat_and]
    exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)
  · exact canon_lt _ (ofF32_lt _)
  all_goals exact lt32_of_norm (hap _ _)

theorem lt64 (v : UInt64) : v.toNat < 2 ^ 64 := v.toNat_lt

theorem cvt_lt (r : Option Rnd) (dt st : Ty) (x v : UInt64) (h : cvt r dt st x = .ok v) :
    v.toNat < 2 ^ (Instr.resWidth (.cvt r dt st (.name "") (.imm 0))) := by
  simp only [Instr.resWidth]
  by_cases hds : (dt == st) = true
  · rw [if_pos hds]; exact lt64 v
  rw [if_neg hds]
  by_cases hf : (dt == .f32) = true
  · rw [if_pos hf]
    have hd : dt = .f32 := by simpa using hf
    subst hd
    cases st <;> simp [cvt, Ty.isFloat] at h hds <;> cases h <;>
      first | exact ofF32_lt _ | exact canon_lt _ (ofF32_lt _)
  rw [if_neg hf]
  by_cases hi : (!dt.isFloat && !st.isFloat) = true
  · rw [if_pos hi]
    unfold cvt at h; rw [if_pos hi] at h; cases h
    exact ofIntW_lt dt _
  · rw [if_neg hi]; exact lt64 v


theorem fma32_lt (x y z : UInt64) :
    (canon (fmaLane ClifTy.f32 (mask 32 x) (mask 32 y) (mask 32 z))).toNat < 2 ^ 32 :=
  canon_lt _ (fmaLane32_lt _ _ _ (PtxRun.mask32_toNat_lt x) (PtxRun.mask32_toNat_lt y) (PtxRun.mask32_toNat_lt z))

-- Split a step's hypothesis `h` on one read, closing the failing branch.
set_option hygiene false in
local macro "peel " x:term " as " v:ident : tactic => `(tactic| (
  cases hh : $x
  · (try simp only [hh, bind, Except.bind] at h) <;> (try cases h)
  rename_i $v:ident
  (try simp only [hh, bind, Except.bind, pure, Except.pure] at h)))

/-- **What a step writes fits its instruction's result width**, on hardware whose
    approximations give 32 bits. -/
theorem stepV_bound (c : Ctx) (hap : ∀ k x, c.hw.approx k x &&& widthMask .f32 = c.hw.approx k x)
    (regs : Std.HashMap Reg UInt64) (w : World) (i : Instr) (n : Next) (w' : World) (d : Reg) (v : UInt64)
    (h : stepV c regs w i = .ok (n, w', some (d, v))) : v.toNat < 2 ^ Instr.resWidth i := by
  cases i with
  | mov t d0 a =>
      simp only [stepV] at h
      peel opndV c regs a as v1
      cases h; exact PtxRun.mask_lt t _
  | selp t d0 a b p =>
      simp only [stepV] at h
      peel opndV c regs p as v2
      peel opndV c regs a as v3
      peel opndV c regs b as v4
      cases h; exact PtxRun.mask_lt t _
  | fma r t d0 a b e =>
      cases r with
      | none =>
          simp only [stepV] at h
          peel opndV c regs a as v5
          peel opndV c regs b as v6
          peel opndV c regs e as v7
          cases h; exact PtxRun.mask_lt t _
      | some rr =>
          simp only [stepV] at h
          by_cases ht : (t != Ty.f32) = true
          · rw [if_pos ht] at h; cases h
          rw [if_neg ht] at h
          peel opndV c regs a as v8
          peel opndV c regs b as v9
          peel opndV c regs e as v10
          cases h; exact fma32_lt _ _ _
  | bin o r t d0 a b =>
      simp only [stepV] at h
      peel opndV c regs a as x
      peel opndV c regs b as y
      cases hb : AlgorithmLib.PtxRun.bin o t x y <;>
        simp only [hb, bind, Except.bind, pure, Except.pure] at h
      · cases h
      cases h
      simp only [Instr.resWidth]
      unfold AlgorithmLib.PtxRun.bin at hb
      by_cases htf : t.isFloat = true
      · rw [if_pos htf]; rw [if_pos htf] at hb
        by_cases h32 : (t == .f32) = true
        · rw [if_pos h32]; rw [if_neg (by simpa using h32)] at hb
          exact binF_lt o _ _ _ (PtxRun.mask32_toNat_lt x) (PtxRun.mask32_toNat_lt y) hb
        · rw [if_neg h32]; exact lt64 _
      · rw [if_neg htf]; rw [if_neg htf] at hb
        by_cases hmw : (o == .mulWide) = true
        · rw [if_pos hmw]; exact lt64 _
        · rw [if_neg hmw]; exact binI_lt o t x y _ (by simpa using hmw) hb
  | un o r t d0 a =>
      simp only [stepV] at h
      peel opndV c regs a as x
      cases hu : AlgorithmLib.PtxRun.un o c.hw t x <;>
        simp only [hu, bind, Except.bind, pure, Except.pure] at h
      · cases h
      cases h
      simp only [Instr.resWidth]
      split
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨ht, ho⟩ := hc
        have : t = .f32 := by simpa using ht
        subst this
        exact un_lt o c.hw hap x _ ho hu
      · exact lt64 _
  | cvt r dt st d0 a =>
      simp only [stepV] at h
      peel opndV c regs a as x
      cases hc : cvt r dt st x <;> simp only [hc, bind, Except.bind, pure, Except.pure] at h
      · cases h
      cases h
      exact cvt_lt r dt st x _ hc
  | setp cc t p a b =>
      simp only [stepV] at h
      peel opndV c regs a as x
      peel opndV c regs b as y
      cases hc : cmp cc t x y <;> simp only [hc, bind, Except.bind, pure, Except.pure] at h
      · cases h
      cases h
      simp only [Instr.resWidth]
      split <;> decide
  | shfl m d0 a b cl k =>
      simp only [stepV] at h
      peel opndV c regs cl as kk
      by_cases h31 : (kk != 31) = true
      · rw [if_pos h31] at h; cases h
      rw [if_neg h31] at h
      cases m <;> simp only [bind, Except.bind, pure, Except.pure] at h <;> (try (cases h; done))
      all_goals
        peel opndV c regs a as v18
        peel opndV c regs b as v19
        repeat' (first | (split at h) | (cases h; done))
        all_goals (cases h; exact PtxRun.mask32_toNat_lt _)
  | vote d0 p k =>
      simp only [stepV] at h
      peel opndV c regs p as v20
      repeat' (first | (split at h) | (cases h; done))
      all_goals (cases h; exact PtxRun.mask32_toNat_lt _)
  | _ => exact Nat.lt_of_lt_of_le (UInt64.toNat_lt _) (by simp [Instr.resWidth])


/-! ## The run, line by line -/

theorem labels_eq (body : List Line) : Coalesce.labelsOf body = AlgorithmLib.PtxRun.labelsOf body := rfl

theorem mask_id (t : Ty) (x : UInt64) (h : x.toNat < 2 ^ t.width) : mask t.width x = x := by
  cases t <;> simp only [Ty.width] at h ⊢
  all_goals first
    | (simp only [mask, show (64 : Nat) ≥ 64 from by decide, if_true]; done)
    | (simp only [mask]
       rw [if_neg (by decide)]
       apply UInt64.toNat_inj.mp
       rw [UInt64.toNat_and]
       first
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 1 - 1).toNat = 2 ^ 1 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 8 - 1).toNat = 2 ^ 8 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 16 - 1).toNat = 2 ^ 16 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 32 - 1).toNat = 2 ^ 32 - 1 from rfl]
       rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt h])

/-- A register's bound only grows as more lines are folded in. -/
theorem bounds_fold_mono (L : List Line) (m : Std.HashMap Reg Nat) (d : Reg) (k : Nat)
    (hk : k ≤ m.getD d 0) (hin : d ∈ m) :
    k ≤ (L.foldl (fun m l => match l with
      | .ins _ i => match Instr.def? i with
          | some d => m.insert d (max (m.getD d 0) (Instr.resWidth i))
          | none => m
      | _ => m) m).getD d 64 := by
  induction L generalizing m with
  | nil =>
      simp only [List.foldl_nil]
      obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (Std.HashMap.mem_iff_isSome_getElem?.mp hin)
      rw [Std.HashMap.getD_eq_getD_getElem?, hy] at hk ⊢
      exact hk
  | cons l L ih =>
      simp only [List.foldl_cons]
      apply ih
      · cases l with
        | label _ => exact hk
        | ins g i =>
            simp only
            cases hd : Instr.def? i with
            | none => exact hk
            | some d' =>
                simp only
                by_cases he : d' = d
                · subst he; rw [Std.HashMap.getD_insert_self]; exact Nat.le_trans hk (Nat.le_max_left _ _)
                · rw [Std.HashMap.getD_insert, if_neg (by simpa using he)]; exact hk
      · cases l with
        | label _ => exact hin
        | ins g i =>
            simp only
            cases Instr.def? i with
            | none => exact hin
            | some d' => exact Std.HashMap.mem_insert.mpr (.inr hin)

/-- **Every line writing a register writes within its bound.** -/
theorem bounds_ge (body : List Line) (g : Option (Bool × Reg)) (i : Instr) (d : Reg)
    (hl : Line.ins g i ∈ body) (hd : Instr.def? i = some d) :
    Instr.resWidth i ≤ (bounds body.toArray).getD d 64 := by
  unfold bounds
  rw [List.foldl_toArray' _ _ _ rfl]
  generalize (∅ : Std.HashMap Reg Nat) = m
  induction body generalizing m with
  | nil => cases hl
  | cons l L ih =>
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp hl with rfl | hL
      · apply bounds_fold_mono
        · simp only [hd]; rw [Std.HashMap.getD_insert_self]; exact Nat.le_max_right _ _
        · simp only [hd]; exact Std.HashMap.mem_insert_self
      · exact ih hL _


/-! ### Guards -/

/-- A register's value as `getReg` reads it. -/
def regV (c : Ctx) (regs : Std.HashMap Reg UInt64) : Reg → UInt64
  | .special s => (c.special.lookup s).getD 0
  | r => (regs.get? r).getD 0

/-- Whether a guard lets its line run. -/
def guardV (c : Ctx) (regs : Std.HashMap Reg UInt64) : Option (Bool × Reg) → Bool
  | none => true
  | some (neg, p) => if neg then regV c regs p &&& 1 == 0 else regV c regs p &&& 1 == 1

theorem guard_run (c : Ctx) (g : Option (Bool × Reg)) (ts : TS) :
    (guardOk c g).run ts = .ok (guardV c ts.regs g, ts) := by
  cases g with
  | none => rfl
  | some np =>
      obtain ⟨neg, p⟩ := np
      cases p <;> rfl

theorem regV_plain (c : Ctx) (regs : Std.HashMap Reg UInt64) (r : Reg) (h : Plain r) :
    regV c regs r = (regs.get? r).getD 0 := by
  cases r with
  | special s => exact absurd rfl (h s)
  | _ => rfl

theorem regV_ren (c : Ctx) (ρ : Ren) (p : Reg) (regs regs' : Std.HashMap Reg UInt64) (h : Agree ρ [p] regs regs') :
    regV c regs' (Reg.ren ρ p) = regV c regs p := by
  cases p with
  | special s => rfl
  | val a n =>
      have hp : Plain (.val a n) := fun _ h => by cases h
      rw [regV_plain c regs' _ (ren_plain ρ _ hp), regV_plain c regs _ hp, h _ (by simp) hp]
  | own a b n =>
      have hp : Plain (.own a b n) := fun _ h => by cases h
      rw [regV_plain c regs' _ (ren_plain ρ _ hp), regV_plain c regs _ hp, h _ (by simp) hp]
  | name n =>
      have hp : Plain (.name n) := fun _ h => by cases h
      rw [regV_plain c regs' _ (ren_plain ρ _ hp), regV_plain c regs _ hp, h _ (by simp) hp]

theorem guardV_ren (c : Ctx) (ρ : Ren) (g : Option (Bool × Reg)) (regs regs' : Std.HashMap Reg UInt64)
    (h : Agree ρ (match g with | some (_, p) => [p] | none => []) regs regs') :
    guardV c regs' (g.map fun (n, p) => (n, Reg.ren ρ p)) = guardV c regs g := by
  cases g with
  | none => rfl
  | some np =>
      obtain ⟨neg, p⟩ := np
      simp only [Option.map, guardV, regV_ren c ρ p regs regs' h]


/-! ### Labels -/

/-- One line's part in the label table. -/
def labStep (m : Std.HashMap Lbl Nat) (p : Line × Nat) : Std.HashMap Lbl Nat :=
  match p.1 with
  | .label n => m.insert n p.2
  | _ => m

theorem labelsOf_fold (body : List Line) : AlgorithmLib.PtxRun.labelsOf body = (body.zipIdx).foldl labStep {} := by
  unfold AlgorithmLib.PtxRun.labelsOf labStep; rfl

theorem labels_fold (l0 : Lbl) : ∀ (A B : List Line) (k : Nat) (m m' : Std.HashMap Lbl Nat),
    A.length = B.length →
    (∀ i (ha : i < A.length) (hb : i < B.length), A[i] = .label l0 ↔ B[i] = .label l0) →
    m.get? l0 = m'.get? l0 →
    ((A.zipIdx k).foldl labStep m).get? l0 = ((B.zipIdx k).foldl labStep m').get? l0
  | [], [], _, _, _, _, _, h => h
  | [], _ :: _, _, _, _, hl, _, _ => by cases hl
  | _ :: _, [], _, _, _, hl, _, _ => by cases hl
  | a :: A, b :: B, k, m, m', hl, hi, h => by
      simp only [List.zipIdx_cons, List.foldl_cons]
      apply labels_fold l0 A B (k + 1)
      · simpa using hl
      · intro i ha hb; exact hi (i + 1) (by simpa using ha) (by simpa using hb)
      · have h0 := hi 0 (by simp) (by simp)
        simp only [List.getElem_cons_zero] at h0
        by_cases hA : a = .label l0
        · have hB := h0.mp hA
          subst hA; subst hB
          simp [labStep]
        · have hB : b ≠ .label l0 := fun e => hA (h0.mpr e)
          have ea : (labStep m (a, k)).get? l0 = m.get? l0 := by
            unfold labStep
            cases a with
            | label n =>
                simp only
                rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert,
                  if_neg (by intro e; apply hA; simp at e; rw [e])]
            | ins _ _ => rfl
          have eb : (labStep m' (b, k)).get? l0 = m'.get? l0 := by
            unfold labStep
            cases b with
            | label n =>
                simp only
                rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert,
                  if_neg (by intro e; apply hB; simp at e; rw [e])]
            | ins _ _ => rfl
          rw [ea, eb, h]

/-- A renamed line is a given label exactly where the line was, for a label
    no removed move leaves. -/
theorem renLine_label (ρ : Ren) (bd : Std.HashMap Reg Nat) (pc : Nat) (l : Line) (l0 : Lbl)
    (h0 : ∀ q, l0 ≠ .gone q) : renLine ρ bd pc l = .label l0 ↔ l = .label l0 := by
  cases l with
  | label n => simp [renLine, Line.ren]
  | ins g i =>
      simp only [reduceCtorEq, iff_false]
      unfold renLine
      split
      · split
        · intro e; cases e; exact h0 pc rfl
        · simp [Line.ren]
      · simp [Line.ren]


/-! ### What a step leads to -/

/-- **A step writes its instruction's destination, steps on unless it is a
    branch or a return, and jumps only by a branch to its label's line.** -/
theorem stepV_shape (c : Ctx) (regs : Std.HashMap Reg UInt64) (w : World) (i : Instr) (n : Next) (w' : World)
    (wr : Option (Reg × UInt64)) (h : stepV c regs w i = .ok (n, w', wr)) :
    wr.map (·.1) = Instr.def? i ∧
    (n = .step → (∀ u l, i ≠ .bra u l) ∧ i ≠ .ret) ∧
    (∀ j, n = .jump j → ∃ u l, i = .bra u l ∧ c.labels.get? l = some j) := by
  cases i <;> simp only [stepV, bind, Except.bind, pure, Except.pure] at h <;>
    (repeat' (first | (split at h) | (cases h; done))) <;>
    (cases h) <;> simp_all [Instr.def?]
  rename_i u l _ j hj
  cases u
  · exact .inl ⟨l, ⟨rfl, rfl⟩, hj⟩
  · exact .inr ⟨l, ⟨rfl, rfl⟩, hj⟩


/-! ### What the check guarantees at a line -/

theorem valid_line (body : Array Line) (ρ : Ren) (live : Array (List Reg)) (hv : valid body ρ live = true)
    (pc : Nat) (hpc : pc < body.size) :
    (match body[pc] with
    | .label _ => (liveOut (Coalesce.labelsOf body.toList) live pc body[pc]).all ((live.getD pc []).contains ·)
    | .ins g i =>
        (Line.uses body[pc]).all ((live.getD pc []).contains ·) &&
        (liveOut (Coalesce.labelsOf body.toList) live pc body[pc]).all (fun r =>
          (live.getD pc []).contains r || (g.isNone && Instr.def? i == some r)) &&
        match Instr.def? i with
        | none => true
        | some d => (liveOut (Coalesce.labelsOf body.toList) live pc body[pc]).all fun r =>
            r == d || Reg.ren ρ r != Reg.ren ρ d) = true := by
  simp only [valid, List.all_eq_true, List.mem_range] at hv
  have h := hv pc hpc
  have hg : body.getD pc (.label (.blk 0)) = body[pc] := by simp [Array.getD, hpc]
  rw [hg] at h
  exact h

theorem valid_label (body : Array Line) (ρ : Ren) (live : Array (List Reg)) (hv : valid body ρ live = true)
    (pc : Nat) (hpc : pc < body.size) (n : Lbl) (hl : body[pc] = .label n) :
    ∀ r ∈ liveOut (Coalesce.labelsOf body.toList) live pc (.label n), r ∈ live.getD pc [] := by
  have h := valid_line body ρ live hv pc hpc
  rw [hl] at h
  simp only [List.all_eq_true, List.contains_iff_mem] at h
  exact h

theorem valid_ins (body : Array Line) (ρ : Ren) (live : Array (List Reg)) (hv : valid body ρ live = true)
    (pc : Nat) (hpc : pc < body.size) (g : Option (Bool × Reg)) (i : Instr) (hl : body[pc] = .ins g i) :
    let out := liveOut (Coalesce.labelsOf body.toList) live pc (.ins g i)
    (∀ u ∈ Line.uses (.ins g i), u ∈ live.getD pc []) ∧
    (∀ r ∈ out, r ∈ live.getD pc [] ∨ (g = none ∧ Instr.def? i = some r)) ∧
    (∀ d, Instr.def? i = some d → ∀ r ∈ out, r = d ∨ Reg.ren ρ r ≠ Reg.ren ρ d) := by
  have h := valid_line body ρ live hv pc hpc
  rw [hl] at h
  intro out
  simp only [Bool.and_eq_true, List.all_eq_true, List.contains_iff_mem, Bool.or_eq_true, beq_iff_eq,
    Option.isNone_iff_eq_none] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  refine ⟨h1, h2, fun d hd r hr => ?_⟩
  rw [hd] at h3
  simp only [List.all_eq_true, Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h3
  rcases h3 r hr with e | e
  · exact .inl e
  · exact .inr e

/-- A renamed instruction line is a removed move's label, or the line renamed. -/
theorem renLine_ins (ρ : Ren) (bd : Std.HashMap Reg Nat) (pc : Nat) (g : Option (Bool × Reg)) (i : Instr) :
    (renLine ρ bd pc (.ins g i) = .label (.gone pc) ∧ ∃ t d s, g = none ∧ i = .mov t d (.reg s) ∧
      Reg.ren ρ d = Reg.ren ρ s ∧ renamable d = true ∧ renamable s = true ∧ bd.getD s 64 ≤ t.width) ∨
    renLine ρ bd pc (.ins g i) = Line.ren ρ (.ins g i) := by
  cases g with
  | some p => exact .inr rfl
  | none =>
    cases i with
    | mov t d a =>
        cases a with
        | reg s =>
            by_cases hc : (Reg.ren ρ d == Reg.ren ρ s && renamable d && renamable s && decide (bd.getD s 64 ≤ t.width)) = true
            · left
              refine ⟨by simp only [renLine, hc, if_true], t, d, s, rfl, rfl, ?_⟩
              simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
              exact ⟨hc.1.1.1, hc.1.1.2, hc.1.2, hc.2⟩
            · right
              simp only [renLine, hc, Bool.false_eq_true, if_false]; rfl
        | _ => exact .inr rfl
    | _ => exact .inr rfl


/-! ### One line of a run -/

theorem go_zero (c : Ctx) (pc : Nat) (ts : TS) : (go c 0 pc).run ts = .error "out of steps" := rfl

theorem go_none (c : Ctx) (fuel pc : Nat) (ts : TS) (h : c.body[pc]? = none) :
    (go c (fuel + 1) pc).run ts = .ok ((), ts) := by
  unfold go; simp only [h]; rfl

theorem go_label (c : Ctx) (fuel pc : Nat) (ts : TS) (n : Lbl) (h : c.body[pc]? = some (.label n)) :
    (go c (fuel + 1) pc).run ts = (go c fuel (pc + 1)).run ts := by
  conv => lhs; unfold go
  simp only [h]

/-- The run after a step: its next line, or its end. -/
def cont (c : Ctx) (fuel pc : Nat) (ts : TS) : Next × World × Option (Reg × UInt64) → Except String (Unit × TS)
  | (n, w, wr) =>
    let ts1 := (after ts (n, w, wr)).2
    match n with
    | .step => (go c fuel (pc + 1)).run ts1
    | .jump j => (go c fuel j).run ts1
    | .done => .ok ((), ts1)

theorem go_ins (c : Ctx) (fuel pc : Nat) (ts : TS) (g : Option (Bool × Reg)) (i : Instr)
    (h : c.body[pc]? = some (.ins g i)) (h2 : Coalesce.Instr.two i = false) :
    (go c (fuel + 1) pc).run ts =
      if guardV c ts.regs g then
        match stepV c ts.regs ts.w i with
        | .error e => .error e
        | .ok r => cont c fuel pc ts r
      else (go c fuel (pc + 1)).run ts := by
  conv => lhs; unfold go
  simp only [h, run_bind', run_sbind, guard_run]
  cases hg : guardV c ts.regs g
  · simp only [if_false, Bool.false_eq_true]
  · simp only [if_true, run_bind', exec_run c i _ h2]
    cases hs : stepV c ts.regs ts.w i with
    | error e => rfl
    | ok r =>
        obtain ⟨n, w1, wr⟩ := r
        cases n <;> cases wr <;> rfl


/-! ### The run, simulated -/

/-- Each register holds a value within its bound. -/
def Bnd (bd : Std.HashMap Reg Nat) (regs : Std.HashMap Reg UInt64) : Prop :=
  ∀ r, ((regs.get? r).getD 0).toNat < 2 ^ bd.getD r 64

/-- Two runs' ends: the same world, or the same error. -/
def RelR : Except String (Unit × TS) → Except String (Unit × TS) → Prop
  | .ok (_, s), .ok (_, s') => s.w = s'.w
  | .error x, .error y => x = y
  | _, _ => False

/-- The renamed body. -/
def bodyR (e : Entry) (ρ : Ren) : List Line :=
  e.body.zipIdx.map fun (l, pc) => renLine ρ (bounds e.body.toArray) pc l

theorem bodyR_get (e : Entry) (ρ : Ren) (pc : Nat) :
    (bodyR e ρ).toArray[pc]? = (e.body.toArray[pc]?).map (renLine ρ (bounds e.body.toArray) pc) := by
  simp only [bodyR, List.getElem?_toArray, List.getElem?_map, List.getElem?_zipIdx]
  cases e.body[pc]? <;> simp

theorem bodyR_labels (e : Entry) (ρ : Ren) (l : Lbl) (hl : ∀ q, l ≠ .gone q) :
    (AlgorithmLib.PtxRun.labelsOf (bodyR e ρ)).get? l = (AlgorithmLib.PtxRun.labelsOf e.body).get? l := by
  rw [labelsOf_fold, labelsOf_fold]
  apply labels_fold l _ _ 0 {} {} (by simp [bodyR])
  · intro i ha hb
    simp only [bodyR, List.getElem_map, List.getElem_zipIdx]
    exact renLine_label ρ _ _ _ l hl
  · rfl

theorem bnd_insert (bd : Std.HashMap Reg Nat) (regs : Std.HashMap Reg UInt64) (d : Reg) (v : UInt64)
    (h : Bnd bd regs) (hv : v.toNat < 2 ^ bd.getD d 64) : Bnd bd (regs.insert d v) := by
  intro r
  rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert]
  split
  · rename_i he; simp at he; subst he; simpa using hv
  · rw [← Std.HashMap.get?_eq_getElem?]; exact h r

/-- **A write keeps the registers live after it agreeing**, where each is the
    written one or was live before and shares no name with it. -/
theorem agree_write (ρ : Ren) (inn O : List Reg) (regs regs' : Std.HashMap Reg UInt64) (d : Reg) (v : UInt64)
    (ha : Agree ρ inn regs regs') (hsub : ∀ r ∈ O, r ≠ d → r ∈ inn)
    (hint : ∀ r ∈ O, r = d ∨ Reg.ren ρ r ≠ Reg.ren ρ d) :
    Agree ρ O (regs.insert d v) (regs'.insert (Reg.ren ρ d) v) := by
  intro r hr hp
  by_cases he : r = d
  · subst he
    simp [Std.HashMap.get?_eq_getElem?]
  · have hn := (hint r hr).resolve_left he
    rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert,
      Std.HashMap.getElem?_insert, if_neg (by simpa using fun h => hn h.symm),
      if_neg (by simpa using fun h => he h.symm), ← Std.HashMap.get?_eq_getElem?, ← Std.HashMap.get?_eq_getElem?]
    exact ha r (hsub r hr he) hp


theorem renamable_plain (r : Reg) (h : renamable r = true) : Plain r := by
  cases r <;> simp [renamable] at h <;> exact fun _ e => by cases e

theorem mem_liveOut (L : Std.HashMap Lbl Nat) (live : Array (List Reg)) (pc q : Nat) (l : Line)
    (hq : q ∈ succs L pc l) (r : Reg) (hr : r ∈ live.getD q []) : r ∈ liveOut L live pc l := by
  simp only [liveOut, List.mem_flatMap]; exact ⟨q, hq, hr⟩

/-- No jump of a fresh body is to a removed move's label. -/
theorem fresh_bra (body : List Line) (hf : fresh body = true) (g : Option (Bool × Reg)) (u : Bool) (l : Lbl)
    (hm : Line.ins g (.bra u l) ∈ body) : ∀ q, l ≠ .gone q := by
  intro q he; subst he
  simp only [fresh, List.all_eq_true] at hf
  have := hf _ hm
  simp at this

/-- No line of a fresh body writes two registers. -/
theorem fresh_two (body : List Line) (hf : fresh body = true) (g : Option (Bool × Reg)) (i : Instr)
    (hm : Line.ins g i ∈ body) : Coalesce.Instr.two i = false := by
  simp only [fresh, List.all_eq_true] at hf
  have := hf _ hm
  revert this
  cases i <;> simp [Coalesce.Instr.two]

theorem two_ren (ρ : Ren) (i : Instr) : Coalesce.Instr.two (Instr.ren ρ i) = Coalesce.Instr.two i := by
  cases i <;> rfl

/-- **A run of the renamed body ends as the run of the body does**, from
    states agreeing on the registers live into the line. -/
theorem go_sim (e : Entry) (ρ : Ren) (live : Array (List Reg))
    (hv : valid e.body.toArray ρ live = true) (hf : fresh e.body = true)
    (c : Ctx) (hb : c.body = e.body.toArray) (hl : c.labels = AlgorithmLib.PtxRun.labelsOf e.body)
    (hap : ∀ k x, c.hw.approx k x &&& widthMask .f32 = c.hw.approx k x) :
    ∀ fuel pc (ts ts' : TS), ts.w = ts'.w → Agree ρ (live.getD pc []) ts.regs ts'.regs →
      Bnd (bounds e.body.toArray) ts.regs →
      RelR ((go c fuel pc).run ts)
        ((go (withBody c (bodyR e ρ).toArray (AlgorithmLib.PtxRun.labelsOf (bodyR e ρ))) fuel pc).run ts') := by
  intro fuel
  induction fuel with
  | zero => intro pc ts ts' _ _ _; simp only [go_zero, RelR]
  | succ fuel ih =>
  intro pc ts ts' hw ha hbd
  let bd := bounds e.body.toArray
  let c' := withBody c (bodyR e ρ).toArray (AlgorithmLib.PtxRun.labelsOf (bodyR e ρ))
  have hb' : c'.body[pc]? = (c.body[pc]?).map (renLine ρ bd pc) := by
    show (bodyR e ρ).toArray[pc]? = _; rw [hb, bodyR_get]
  have hL : Coalesce.labelsOf e.body.toArray.toList = c.labels := by rw [hl]; simp; rfl
  cases hcl : c.body[pc]? with
  | none =>
      rw [go_none c fuel pc ts hcl, go_none c' fuel pc ts' (by rw [hb', hcl]; rfl)]
      exact hw
  | some l =>
    rw [hb] at hcl
    have hpc : pc < e.body.toArray.size := (Array.getElem?_eq_some_iff.mp hcl).1
    have hl0 : e.body.toArray[pc] = l := (Array.getElem?_eq_some_iff.mp hcl).2
    have hmem : l ∈ e.body := by
      rw [← hl0]; simp [List.getElem_mem]
    rw [← hb] at hcl
    cases l with
    | label n =>
        rw [go_label c fuel pc ts n hcl, go_label c' fuel pc ts' n (by rw [hb', hcl]; rfl)]
        apply ih (pc + 1) ts ts' hw _ hbd
        exact ha.mono fun r hr => valid_label _ ρ live hv pc hpc n hl0 r
          (mem_liveOut _ live pc (pc + 1) _ (by simp [succs]) r hr)
    | ins g i =>
        obtain ⟨huses, hout, hint⟩ := valid_ins _ ρ live hv pc hpc g i hl0
        rw [hL] at hout hint
        have h2 := fresh_two e.body hf g i hmem
        rw [go_ins c fuel pc ts g i hcl h2]
        rcases renLine_ins ρ bd pc g i with ⟨hr, t, d, s, rfl, rfl, hds, hdr, hsr, hbs⟩ | hr
        · -- a move the renaming removed: the shared register already holds its value
          rw [go_label c' fuel pc ts' (.gone pc) (by rw [hb', hcl]; simp [hr])]
          have hsp := renamable_plain s hsr
          simp only [guardV, if_true, stepV, opndV_plain c ts.regs s hsp, bind, Except.bind, pure, Except.pure,
            cont, after]
          have hsv : ((ts.regs.get? s).getD 0).toNat < 2 ^ t.width :=
            Nat.lt_of_lt_of_le (hbd s) (Nat.pow_le_pow_right (n := 2) (by decide) hbs)
          rw [mask_id t _ hsv]
          refine ih (pc + 1) { regs := ts.regs.insert d ((ts.regs.get? s).getD 0), w := ts.w } ts' hw ?_ ?_
          rotate_left
          · exact bnd_insert bd ts.regs d _ hbd (Nat.lt_of_lt_of_le hsv
              (Nat.pow_le_pow_right (n := 2) (by decide) (bounds_ge e.body none (.mov t d (.reg s)) d hmem rfl)))
          intro r hr hp
          have hro : r ∈ liveOut c.labels live pc (.ins none (.mov t d (.reg s))) :=
            mem_liveOut _ live pc (pc + 1) _ (by simp [succs]) r hr
          show (ts'.regs.get? (Reg.ren ρ r)).getD 0 = ((ts.regs.insert d ((ts.regs.get? s).getD 0)).get? r).getD 0
          by_cases he : r = d
          · rw [he, Std.HashMap.get?_eq_getElem? (m := ts.regs.insert _ _), Std.HashMap.getElem?_insert_self,
              Option.getD_some, hds]
            exact ha s (huses s (by simp [Line.uses, Instr.uses, Coalesce.Opnd.regs])) hsp
          · rw [Std.HashMap.get?_eq_getElem? (m := ts.regs.insert _ _), Std.HashMap.getElem?_insert,
              if_neg (by simpa using fun h => he h.symm), ← Std.HashMap.get?_eq_getElem?]
            rcases hout r hro with hin | ⟨-, hdef⟩
            · exact ha r hin hp
            · simp [Instr.def?] at hdef; exact absurd hdef.symm he
        · rw [go_ins c' fuel pc ts' (g.map fun (n, p) => (n, Reg.ren ρ p)) (Instr.ren ρ i)
            (by rw [hb', hcl]; simp [hr, Line.ren]) (by rw [two_ren]; exact h2)]
          have hgv : guardV c' ts'.regs (g.map fun (n, p) => (n, Reg.ren ρ p)) = guardV c ts.regs g := by
            have := guardV_ren c ρ g ts.regs ts'.regs (ha.mono fun r hr => huses r (by
              cases g with
              | none => simp at hr
              | some np => obtain ⟨n, p⟩ := np; simp at hr; subst hr; simp [Line.uses]))
            exact this
          rw [hgv]
          cases hgd : guardV c ts.regs g with
          | false =>
              simp only [Bool.false_eq_true, if_false]
              have hgs : g ≠ none := by intro h0; subst h0; simp [guardV] at hgd
              apply ih (pc + 1) ts ts' hw _ hbd
              intro r hr hp
              have hro : r ∈ liveOut c.labels live pc (.ins g i) :=
                mem_liveOut _ live pc (pc + 1) _ (by
                  cases g with
                  | none => exact absurd rfl hgs
                  | some _ => cases i <;> simp [succs]) r hr
              rcases hout r hro with hin | ⟨h0, -⟩
              · exact ha r hin hp
              · exact absurd h0 hgs
          | true =>
              simp only [if_true]
              rw [← hw, stepV_ren c _ _ ρ i (fun u l he => by
                subst he
                rw [hl]; exact bodyR_labels e ρ l (fresh_bra e.body hf g u l hmem)) ts.regs ts'.regs ts.w
                (ha.mono fun r hr => huses r (by simp [Line.uses, hr]))]
              cases hs : stepV c ts.regs ts.w i with
              | error x => rfl
              | ok res =>
                  obtain ⟨n, w1, wr⟩ := res
                  obtain ⟨hwr, hstep, hjump⟩ := stepV_shape c ts.regs ts.w i n w1 wr hs
                  -- the next line, and that it follows this one
                  have next : ∀ q, (q ∈ succs c.labels pc (.ins g i)) →
                      RelR ((go c fuel q).run (after ts (n, w1, wr)).2)
                        ((go c' fuel q).run (after ts' (renOut ρ (n, w1, wr))).2) := by
                    intro q hq
                    cases wr with
                    | none =>
                        refine ih q { ts with w := w1 } { ts' with w := w1 } rfl ?_ hbd
                        intro r hr hp
                        rcases hout r (mem_liveOut _ live pc q _ hq r hr) with hin | ⟨-, hdef⟩
                        · exact ha r hin hp
                        · rw [← hwr] at hdef; cases hdef
                    | some dv =>
                        obtain ⟨d, v⟩ := dv
                        have hd : Instr.def? i = some d := by rw [← hwr]; rfl
                        refine ih q { regs := ts.regs.insert d v, w := w1 }
                          { regs := ts'.regs.insert (Reg.ren ρ d) v, w := w1 } rfl ?_
                          (bnd_insert bd ts.regs d v hbd (Nat.lt_of_lt_of_le
                            (stepV_bound c hap ts.regs ts.w i n w1 d v hs)
                            (Nat.pow_le_pow_right (n := 2) (by decide) (bounds_ge e.body g i d hmem hd))))
                        apply agree_write ρ (live.getD pc []) _ ts.regs ts'.regs d v ha
                        · intro r hr hne
                          rcases hout r (mem_liveOut _ live pc q _ hq r hr) with hin | ⟨-, hdef⟩
                          · exact hin
                          · rw [hd] at hdef; cases hdef; exact absurd rfl hne
                        · intro r hr; exact hint d hd r (mem_liveOut _ live pc q _ hq r hr)
                  simp only [Except.map]
                  cases n with
                  | done => cases wr <;> rfl
                  | step =>
                      obtain ⟨hnb, hnr⟩ := hstep rfl
                      exact next (pc + 1) (by cases i <;> simp_all [succs])
                  | jump j =>
                      obtain ⟨u, l, rfl, hj⟩ := hjump j rfl
                      rw [Std.HashMap.get?_eq_getElem?] at hj
                      exact next j (by simp [succs, hj])


/-- **A thread of the coalesced entry ends as a thread of the entry does**: in
    the same world, or with the same error, on hardware whose approximations
    give 32 bits. -/
theorem thread_run (e : Entry) (ans : List Answer) (args : List V) (w : World) (hw : Hw)
    (hap : ∀ k x, hw.approx k x &&& widthMask .f32 = hw.approx k x) (steps : Nat) (prov : Bool) :
    AlgorithmLib.PtxRun.thread (Coalesce.run e) ans args w hw steps prov =
      AlgorithmLib.PtxRun.thread e ans args w hw steps prov := by
  unfold Coalesce.run
  obtain ⟨ρ, live⟩ := plan e
  simp only
  split
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨hv, hf⟩ := hc
    let c := Ctx.ofThread e ans args w hw prov
    have hs := go_sim e ρ live hv hf c rfl rfl hap steps 0 { regs := {}, w } { regs := {}, w } rfl
      (fun r _ _ => by simp) (fun r => by simp; exact Nat.two_pow_pos _)
    unfold AlgorithmLib.PtxRun.thread
    show (do let (_, s) ← (go (withBody c (bodyR e ρ).toArray (AlgorithmLib.PtxRun.labelsOf (bodyR e ρ))) steps 0).run
                { regs := {}, w }
             pure s.w) = (do let (_, s) ← (go c steps 0).run { regs := {}, w }; pure s.w)
    revert hs
    cases (go c steps 0).run { regs := {}, w } <;>
      cases (go (withBody c (bodyR e ρ).toArray (AlgorithmLib.PtxRun.labelsOf (bodyR e ρ))) steps 0).run
        { regs := {}, w } <;> simp [RelR] <;> (try intro h) <;>
      (try simp only [Functor.map, Except.map, h])
  · rfl

/-- **A runner of the coalesced entry is the runner of the entry.** -/
theorem runner_run (e : Entry) (hw : Hw) (hap : ∀ k x, hw.approx k x &&& widthMask .f32 = hw.approx k x)
    (steps : Nat) : AlgorithmLib.PtxRun.runner (Coalesce.run e) hw steps = AlgorithmLib.PtxRun.runner e hw steps := by
  funext prov ans args w
  exact thread_run e ans args w hw hap steps prov

end AlgorithmLib.Kernel.PtxCoalesce
