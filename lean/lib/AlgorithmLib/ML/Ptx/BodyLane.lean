module
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
public import AlgorithmLib.Proof.EndsRules
meta import AlgorithmLib.Proof.EndsRules
public import AlgorithmLib.Proof.EndsCont
meta import AlgorithmLib.Proof.EndsCont
public import AlgorithmLib.Kernel.Warp
meta import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyLane` — one lane of a warp statement, and its code

`laneI`: one lane's run of a warp statement. Its integer registers are 64-bit
words, its float registers `F32`s (the hardware's approximations, the world's
NaN rule, as the code's), its memory the world's; its `k`-th collective is
answered from the answers `ans` as a thread's are (`Kernel.AnswersFrom`): a
ballot, an index shuffle or a butterfly shuffle the word `ans[k]`, a barrier
shared memory as `ans[k]` holds it. An index is a 32-bit word computed as its
code computes it (`idxV`); a counted loop runs its body with the trip as
`x.i`; a `while` loop runs at most its bound and must have stopped by then.
A statement staging floats through shared memory or converting an index has
no lane run.

`EnvI` and `FEnv`: the lowering's environment holds the lane's integer and
float registers. `code_lane`: for every statement, from a lane state the
environment holds, where the lane's run ends the code ends with an
environment holding the lane's registers after (`CodeOk`), and at a loop
body's end goes round with the carries the lane ends with (`TailOk`).
`wpE_bodyI`: a thread of a statement's kernel ends with the world its lane's
run ends with.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace AlgorithmLib.ML.BodyLane

/-- What a lane's integer statements see: its lane and block, each buffer's
    address, shared memory's, and the answers its collectives get from the
    `start`-th observation on. -/
structure LCtx where
  lv : UInt64
  cv : UInt64
  ptr : Buf → UInt64
  sm : UInt64
  ans : List Answer
  start : Nat
  /-- The innermost counted loop's trip. -/
  i : Nat
  /-- The hardware's approximations and NaN. -/
  hw : Hw

/-- A lane's integer registers, its world, and its float registers. -/
structure LI where
  reg : Nat → UInt64
  w : World
  f : Nat → F32

def LI.set (s : LI) (d : Nat) (v : UInt64) : LI := { s with reg := fun x => if x = d then v else s.reg x }

def LI.setF (s : LI) (d : Nat) (v : F32) : LI := { s with f := fun x => if x = d then v else s.f x }

/-- A float expression's value in a lane, at a carrier: registers `reg`, a
    literal as the carrier holds it. -/
def _root_.AlgorithmLib.ML.WFExp.evalL {R : Type} [NumOps R] [NumLit R] (reg : Nat → R) : WFExp → R
  | .reg n => reg n
  | .lit v => NumLit.lit v
  | .add a b => NumOps.add (evalL reg a) (evalL reg b)
  | .mul a b => NumOps.mul (evalL reg a) (evalL reg b)
  | .neg a => NumOps.neg (evalL reg a)
  | .inv a => NumOps.inv (evalL reg a)
  | .exp a => NumOps.exp (evalL reg a)
  | .ex2 a => NumOps.ex2 (evalL reg a)
  | .rsqrt a => NumOps.rsqrt (evalL reg a)
  | .maxW a b => NumOps.max (evalL reg a) (evalL reg b)
  | .geF a b => NumOps.ifGe (evalL reg a) (evalL reg b) (NumLit.lit 1.0) (NumLit.lit 0.0)

/-- A float expression's value in a lane, at `F32` on the hardware `hw`. -/
abbrev evalF (hw : Hw) (f : Nat → F32) (e : WFExp) : F32 := @WFExp.evalL F32 (F32.ops hw) _ f e

/-- The calls a float expression's evaluation makes, in order: each `2^x`
    the hardware's approximation, observed with its argument. -/
def exLog (hw : Hw) (f : Nat → F32) : WFExp → World → World
  | .reg _, w | .lit _, w => w
  | .add a b, w | .mul a b, w | .maxW a b, w | .geF a b, w => exLog hw f b (exLog hw f a w)
  | .neg a, w | .inv a, w | .rsqrt a, w => exLog hw f a w
  | .ex2 a, w => obsCall (exLog hw f a w) (.intr .ex2Approx) [.sc .f32 (evalF hw f a).bits]
  | .exp a, w => obsCall (exLog hw f a w) (.intr .ex2Approx)
      [.sc .f32 (nanFix hw.nan .f32 (ofF32 (f32 (evalF hw f a).bits * f32 log2eBits)))]

/-- An index's 32-bit value in a lane, as its code computes it: wrapping
    arithmetic, an index read the word memory holds there. -/
def idxV (x : LCtx) (m : Mem) : IdxE → Option UInt32
  | .laneId => some x.lv.toUInt32
  | .loopI => some (UInt32.ofNat x.i)
  | .ctaId => some x.cv.toUInt32
  | .lit n => if n < 2 ^ 32 then some (UInt32.ofNat n) else none
  | .ireg _ => some 0
  | .ldIdx b o => do
      let j ← idxV x m o
      let v ← m.load (x.ptr b + off j) 4
      some v.toUInt32
  | .add a b => do some ((← idxV x m a) + (← idxV x m b))
  | .mul a b => do some ((← idxV x m a) * (← idxV x m b))

/-- An integer expression's value in a lane. -/
def _root_.AlgorithmLib.ML.IExp.evalL (x : LCtx) (reg : Nat → UInt64) : IExp → UInt64
  | .reg n => reg n
  | .lit n => UInt64.ofNat n
  | .lane => x.lv
  | .cta => x.cv
  | .ptr b => x.ptr b
  | .bin o a b => o.run (a.evalL x reg) (b.evalL x reg)
  | .cmp c a b => if c.run (a.evalL x reg) (b.evalL x reg) then 1 else 0
  | .ctz a => ctz32 ((a.evalL x reg) &&& 0xffffffff)

/-- A store of `n` bytes of `v` at `a`, as a store statement makes it. -/
def stw (w : World) (a : UInt64) (n : Nat) (v : UInt64) : Option World := do
  let m ← w.mem.store a n v
  some { obsStore w a n v with mem := m }

/-- A collective's call observed. -/
abbrev callW (w : World) (c : Intr) (args : List V) : World := obsCall w (.intr c) args

/-- The word a lane's `k`-th collective is answered with. -/
def ansW (ans : List Answer) (k : Nat) : UInt64 := s32 (wordAt ans k)

/-- A trip of a loop carrying `cs`, from before the loop `s0`. -/
def keepL (cs : List Nat) (s0 s : LI) : LI := { s with reg := fun r => if r ∈ cs then s.reg r else s0.reg r }

/-- After local registers `rs`, from before them `s0`. -/
def restoreL (rs : List Nat) (s0 s : LI) : LI := { s with reg := fun r => if r ∈ rs then s0.reg r else s.reg r }

/-- At most `n` trips of `step` while `go` holds, which must have stopped by then. -/
def whileL (step : LI → Option LI) (go : LI → Bool) : Nat → LI → Option LI
  | 0, s => if go s then none else some s
  | k + 1, s => if go s then (step s).bind (whileL step go k) else some s

/-- **One lane's run of a warp statement**: none where an access reaches
    nothing, a collective is made before the lane's first observation, a
    barrier's answer is not shared memory, a `while` loop has not stopped
    within its bound, a counted loop or a shuffle's mask is not below `2^31`,
    or the statement stages floats through shared memory or converts an
    index. A float register is an `F32`, its operations the hardware `x.hw`'s
    with the world's NaN rule, as the code's are; a counted loop runs its
    body with the trip as `x.i`. -/
def laneI (x : LCtx) : EWStmt → LI → Option LI
  | .skip, s => some s
  | .seq a b, s => do laneI x b (← laneI x a s)
  | .setI d e, s => some (s.set d (e.evalL x s.reg))
  | .ldB d a, s => do
      let v ← s.w.mem.load (a.evalL x s.reg) 1
      some (s.set d v)
  | .stB a v g, s =>
      if g.evalL x s.reg ≠ 0 then do
        let w ← stw s.w (a.evalL x s.reg) 1 (v.evalL x s.reg &&& 0xff)
        some { s with w }
      else some s
  | .ldH d a, s => do
      let v ← s.w.mem.load (x.sm + a.evalL x s.reg) 2
      some (s.set d v)
  | .stH a v g, s =>
      if g.evalL x s.reg ≠ 0 then do
        let w ← stw s.w (x.sm + a.evalL x s.reg) 2 (v.evalL x s.reg &&& 0xffff)
        some { s with w }
      else some s
  | .ballot d p, s =>
      if x.start ≤ s.w.obs.length then
        let k := (collCalls s.w x.start).length
        let s' := s.set d (ansW x.ans k &&& 0xffffffff)
        some { s' with w := callW s.w .ballot [boolV (p.evalL x s.reg != 0)] }
      else none
  | .shflIdx d v sr, s =>
      if x.start ≤ s.w.obs.length then
        let k := (collCalls s.w x.start).length
        let s' := s.set d (ansW x.ans k &&& 0xffffffff)
        some { s' with w := (callW s.w .shflIdx
          [.sc .i32 (v.evalL x s.reg &&& 0xffffffff), .sc .i32 (sr.evalL x s.reg &&& 0xffffffff)]) }
      else none
  | .barrier, s =>
      if x.start ≤ s.w.obs.length then
        match x.ans[(collCalls s.w x.start).length]? with
        | some (.smem b) =>
            let w := callW s.w .barrier []
            some { s with w := { w with mem := { w.mem with arena := b } } }
        | _ => none
      else none
  | .iif c t e, s => if c.evalL x s.reg ≠ 0 then laneI x t s else laneI x e s
  | .iwhile n c cs body, s =>
      whileL (fun s' => (laneI x body s').map (keepL cs s)) (fun s' => c.evalL x s'.reg != 0) n s
  | .locals rs body, s => (laneI x body s).map (restoreL rs s)
  | .setR d e, s =>
      let hw : Hw := { x.hw with nan := s.w.mem.nan }
      some { s with f := (s.setF d (evalF hw s.f e)).f, w := exLog hw s.f e s.w }
  | .shflXor d sr m, s =>
      if m < 2 ^ 31 ∧ x.start ≤ s.w.obs.length then
        let k := (collCalls s.w x.start).length
        some { s with f := (s.setF d ⟨ansW x.ans k⟩).f,
                      w := callW s.w .shflBfly [.sc .i32 (s.f sr).bits, .sc .i32 (s32 (UInt32.ofNat m))] }
      else none
  | .loadIdx d b ix, s => do
      let j ← idxV x s.w.mem ix
      let v ← s.w.mem.load (x.ptr b + off j) 4
      some (s.setF d ⟨v⟩)
  | .loadV4 d0 d1 d2 d3 b ix, s => do
      let j ← idxV x s.w.mem ix
      let a := x.ptr b + off j
      let v0 ← s.w.mem.load a 4
      let v1 ← s.w.mem.load (a + c64 4) 4
      let v2 ← s.w.mem.load (a + c64 8) 4
      let v3 ← s.w.mem.load (a + c64 12) 4
      some ((((s.setF d0 ⟨v0⟩).setF d1 ⟨v1⟩).setF d2 ⟨v2⟩).setF d3 ⟨v3⟩)
  | .storeLane b ix n, s => do
      let j ← idxV x s.w.mem ix
      let w ← stw s.w (x.ptr b + off j) 4 (s.f n).bits
      some { s with w }
  | .storeLane0 b ix n, s =>
      if x.lv = 0 then do
        let j ← idxV x s.w.mem ix
        let w ← stw s.w (x.ptr b + off j) 4 (s.f n).bits
        some { s with w }
      else some s
  | .forN n body, s =>
      if n < 2 ^ 31 then (List.range n).foldlM (fun s j => laneI { x with i := j } body s) s else none
  | .forM b a body, s =>
      if a < 2 ^ 31 then do
        let v ← s.w.mem.load (x.ptr b + off (UInt32.ofNat a)) 4
        if v.toNat < 2 ^ 31 then (List.range v.toNat).foldlM (fun s j => laneI { x with i := j } body s) s
        else none
      else none
  | _, _ => none

/-! ## The lowering's environment -/

section
variable (c : Body.Ctx Slot)

/-- **The environment holds the lane's integer registers**: a value
    register its word, a predicate register its bit, a register kept as a
    32-bit word that word (below `2^32`), one not bound zero. -/
structure EnvI (r : Body.Regs Slot) (reg : Nat → UInt64) (Γ : Env) : Prop where
  val : ∀ n v, r.v.lookup n = some v → Γ[v]? = some (.sc .i64 (reg n))
  flag : ∀ n v, r.p.lookup n = some v → Γ[v]? = some (.sc .i8 (reg n))
  word : ∀ n v, r.w.lookup n = some v → Γ[v]? = some (.sc .i32 (reg n)) ∧ reg n < 2 ^ 32
  bit : ∀ n, c.fl.contains n = true → reg n ≤ 1
  flag0 : ∀ n, c.fl.contains n = true → r.p.lookup n = none → reg n = 0
  val0 : ∀ n, c.fl.contains n = false → r.v.lookup n = none → reg n = 0
  fkind : ∀ n v, r.p.lookup n = some v → c.fl.contains n = true
  vkind : ∀ n v, r.v.lookup n = some v → c.fl.contains n = false
  wkind : ∀ n v, r.w.lookup n = some v → c.fl.contains n = false

/-- **The context's slots hold what the lane sees**: its lane and block as
    32-bit words, the buffers' addresses (zero past the last), shared
    memory's, the loop's trip, and zeros. -/
structure Slots (Γ : Env) (x : LCtx) : Prop where
  lane : Γ[c.lane]? = some (.sc .i32 x.lv)
  lane32 : x.lv < 2 ^ 32
  cta : Γ[c.cta]? = some (.sc .i32 x.cv)
  cta32 : x.cv < 2 ^ 32
  buf : ∀ b, Γ[Body.buf c b]? = some (.sc .i64 (x.ptr b))
  z64 : Γ[c.z64]? = some (.sc .i64 0)
  sm : Γ[c.sm]? = some (.sc .i64 x.sm)
  i : Γ[c.i]? = some (.sc .i32 (s32 (UInt32.ofNat x.i)))
  zf : Γ[c.zf]? = some (.sc .f32 0)
  zi : Γ[c.zi]? = some (.sc .i32 0)

end

variable {cfg : Cfg} {c : Body.Ctx Slot} {x : LCtx}

theorem EnvI.extend {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ Γ' : Env} (h : EnvI c r reg Γ)
    (hx : Extends Γ Γ') : EnvI c r reg Γ' :=
  ⟨fun n v hv => hx.get (h.val n v hv), fun n v hv => hx.get (h.flag n v hv),
    fun n v hv => ⟨hx.get (h.word n v hv).1, (h.word n v hv).2⟩, h.bit, h.flag0, h.val0, h.fkind, h.vkind,
    h.wkind⟩

theorem Slots.extend {Γ Γ' : Env} (h : Slots c Γ x) (hx : Extends Γ Γ') : Slots c Γ' x :=
  ⟨hx.get h.lane, h.lane32, hx.get h.cta, h.cta32, fun b => hx.get (h.buf b), hx.get h.z64,
    hx.get h.sm, hx.get h.i, hx.get h.zf, hx.get h.zi⟩

/-- **The environment holds the lane's float registers**: each one's word in
    its slot, zero's where none is bound, every word 32 bits. -/
structure FEnv (c : Body.Ctx Slot) (r : Body.Regs Slot) (f : Nat → F32) (Γ : Env) : Prop where
  get : ∀ n, Γ[Body.get c r n]? = some (.sc .f32 (f n).bits)
  w32 : ∀ n, (f n).bits.toNat < 2 ^ 32

theorem FEnv.extend {r : Body.Regs Slot} {f : Nat → F32} {Γ Γ' : Env} (h : FEnv c r f Γ) (hx : Extends Γ Γ') :
    FEnv c r f Γ' :=
  ⟨fun n => hx.get (h.get n), h.w32⟩

theorem get_setF (r : Body.Regs Slot) (n : Nat) (v : Slot .f32) (k : Nat) :
    Body.get c (Body.set r n v) k = if k = n then v else Body.get c r k := by
  unfold Body.get Body.set
  by_cases h : k = n
  · subst h; simp [List.lookup]
  · simp only [if_neg h, List.lookup]
    rw [show (k == n) = false from by simpa using h]
    simp only [cond_false]
    congr 1
    generalize r.f = fs
    induction fs with
    | nil => rfl
    | cons p ps ih =>
      obtain ⟨a, y⟩ := p
      by_cases ha : a = n
      · subst ha; simp [List.filter, List.lookup, ih, show (k == a) = false from by simpa using h]
      · simp only [List.filter, show (a != n) = true from by simpa using ha, List.lookup]
        split <;> simp_all

/-- The float registers' relation sees only the float bindings. -/
theorem FEnv.ofF {r r' : Body.Regs Slot} {f : Nat → F32} {Γ : Env} (h : FEnv c r f Γ) (he : r'.f = r.f := by rfl) :
    FEnv c r' f Γ :=
  ⟨fun n => by unfold Body.get; rw [he]; exact h.get n, h.w32⟩

/-- The integer registers' relation sees only the integer bindings. -/
theorem EnvI.ofI {r r' : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (h : EnvI c r reg Γ)
    (hv : r'.v = r.v := by rfl) (hp : r'.p = r.p := by rfl) (hw : r'.w = r.w := by rfl) : EnvI c r' reg Γ := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := h
  exact ⟨by rw [hv]; exact h1, by rw [hp]; exact h2, by rw [hw]; exact h3, h4, by rw [hp]; exact h5,
    by rw [hv]; exact h6, by rw [hp]; exact h7, by rw [hv]; exact h8, by rw [hw]; exact h9⟩

/-- Float register `n` bound to a slot holding `y`. -/
theorem FEnv.set {r : Body.Regs Slot} {f : Nat → F32} {Γ : Env} (h : FEnv c r f Γ) (n : Nat) {v : Slot .f32}
    {y : F32} (hv : Γ[v]? = some (.sc .f32 y.bits)) (hy : y.bits.toNat < 2 ^ 32) :
    FEnv c (Body.set r n v) (fun k => if k = n then y else f k) Γ := by
  refine ⟨fun k => ?_, fun k => ?_⟩
  · rw [get_setF]
    by_cases hk : k = n
    · simp [hk, hv]
    · simp [hk, h.get k]
  · by_cases hk : k = n
    · simp [hk, hy]
    · simp [hk, h.w32 k]


/-- Each operation's value, handed on in a slot of an environment keeping `Γ`. -/
theorem xget {Γ : Env} {w : World} {ty : ClifTy} {v : V} {Q : Slot ty → Env → World → Prop}
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some v → Q s Γ' w) : Q Γ.size (Γ.push v) w :=
  h _ _ (Extends.push _ _) (get_push_size)

/-! ## Registers and expressions -/

theorem shr_mod64 (x y : UInt64) : x >>> (y % 64) = x >>> y := by
  apply UInt64.toNat_inj.mp
  simp [UInt64.toNat_shiftRight]

theorem min_wide (x y : UInt64) :
    UInt64.ofNat (min (x.toNat &&& (widthMask ClifTy.i64).toNat) (y.toNat &&& (widthMask ClifTy.i64).toNat)) =
      if x ≤ y then x else y := by
  have hm : ∀ z : UInt64, z.toNat &&& (widthMask ClifTy.i64).toNat = z.toNat := fun z => by
    rw [← UInt64.toNat_and, and_wide64]
  rw [hm, hm]
  split
  · rename_i h; rw [Nat.min_eq_left (UInt64.le_iff_toNat_le.mp h), UInt64.ofNat_toNat]
  · rename_i h; rw [Nat.min_eq_right (by have := UInt64.lt_iff_toNat_lt.mp (UInt64.not_le.mp h); omega),
      UInt64.ofNat_toNat]

/-- A bit's predicate is the bit. -/
theorem boolV_ne0 {x : UInt64} (h : x ≤ 1) : boolV (x != 0) = .sc .i8 x := by
  rcases flag_cases _ h with h0 | h0 <;> rw [h0] <;> rfl

/-- A bit, chosen between two constants pushed after it. -/
theorem eval_sel01 (m : Mem) (Γ0 : Env) (f : Slot .i8) (b : UInt64) (hb : b ≤ 1)
    (hf : Γ0[f]? = some (.sc .i8 b)) :
    evalOp m ((Γ0.push (.sc .i64 (c64 1))).push (.sc .i64 (c64 0)))
      (.select f Γ0.size (Γ0.push (.sc .i64 (c64 1))).size) = some (.sc .i64 b) := by
  have hf' := ((Extends.push Γ0 (.sc .i64 (c64 1))).push_of (.sc .i64 (c64 0))).get hf
  have h1 := (Extends.push (Γ0.push (.sc .i64 (c64 1))) (.sc .i64 (c64 0))).get (get_push_size (Γ := Γ0) (v := .sc .i64 (c64 1)))
  have h0 := get_push_size (Γ := Γ0.push (.sc .i64 (c64 1))) (v := .sc .i64 (c64 0))
  simp only [evalOp, Sem.get, hf', h1, h0]
  rcases flag_cases _ hb with rfl | rfl <;> rfl

/-- A predicate, as a 1 or a 0. -/
theorem eval_selB (m : Mem) (Γ0 : Env) (f : Slot .i8) (p : Bool) (hf : Γ0[f]? = some (boolV p)) :
    evalOp m ((Γ0.push (.sc .i64 (c64 1))).push (.sc .i64 (c64 0)))
      (.select f Γ0.size (Γ0.push (.sc .i64 (c64 1))).size) = some (.sc .i64 (if p then 1 else 0)) := by
  refine eval_sel01 m Γ0 f _ (by split <;> decide) ?_
  rw [hf]; unfold boolV; split <;> rfl

theorem iconst8_zero {Γ : Env} {w : World} {Q : Slot .i8 → Env → World → Prop}
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i8 0) → Q s Γ' w) :
    wpE cfg (iconst .i8 0 : Prog Slot Lvl (Slot .i8)) Q Γ w :=
  wpE_val (v := .sc .i8 0) (by simp [Op'.erase, evalOp, ofInt]; rfl) (xget h)

/-- **A register as a predicate**: whether it is not zero. -/
theorem wpE_getP {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} {w : World} {n : Nat}
    {Q : Slot .i8 → Env → World → Prop} (hE : EnvI c r reg Γ)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (boolV (reg n != 0)) → Q v Γ' w) :
    wpE cfg (Body.getP c r n) Q Γ w := by
  unfold Body.getP
  by_cases hfl : c.fl.contains n = true
  · rw [if_pos hfl]
    have hb := boolV_ne0 (hE.bit n hfl)
    cases hl : r.p.lookup n with
    | some v => exact wpE_pure (h v Γ (Extends.refl _) (by rw [hb]; exact hE.flag n v hl))
    | none =>
        refine iconst8_zero fun s Γ' hx hs => h s Γ' hx ?_
        rw [hb, hE.flag0 n hfl hl]; exact hs
  · rw [if_neg hfl]
    have hcmp : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (reg n)) →
        wpE cfg (do icmp .ne v (← iconst64 0)) Q Γ' w := by
      intro v Γ' hx hv
      refine wpE_bind (wpE_iconst64 ?_)
      have hv' : (Γ'.push (.sc .i64 (c64 0)))[v]? = some (.sc .i64 (reg n)) := (Extends.push _ _).get hv
      refine wpE_val (v := boolV (reg n != 0)) ?_ (xget fun s Γ'' hx' hs => h s Γ'' ((hx.push_of _).trans hx') hs)
      simp [Op'.erase, evalOp, Sem.get, hv', zipIntCmp, cmpInt, ClifTy.isInt]; rfl
    cases hl : r.v.lookup n with
    | some v => exact wpE_bind (wpE_pure (hcmp v Γ (Extends.refl _) (hE.val n v hl)))
    | none =>
        refine wpE_bind (wpE_iconst64 ?_)
        exact hcmp _ _ (Extends.push _ _) (by rw [hE.val0 n (by simpa using hfl) hl]; simp; rfl)

/-- **A register's value**: its 64 bits, a predicate register its bit. -/
theorem wpE_getI {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} {w : World} {n : Nat}
    {Q : Slot .i64 → Env → World → Prop} (hE : EnvI c r reg Γ)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (reg n)) → Q v Γ' w) :
    wpE cfg (Body.getI c r n) Q Γ w := by
  unfold Body.getI
  by_cases hfl : c.fl.contains n = true
  · rw [if_pos hfl]
    refine wpE_bind (wpE_getP hE fun f Γ1 hx1 hf => ?_)
    refine wpE_bind (wpE_iconst64 ?_)
    refine wpE_bind (wpE_iconst64 ?_)
    refine wpE_val (v := .sc .i64 (reg n)) ?_ ?_
    · exact eval_sel01 _ Γ1 f _ (hE.bit n hfl) (by rw [hf, boolV_ne0 (hE.bit n hfl)])
    · exact h _ _ (((hx1.push_of _).push_of _).push_of _) (get_push_size)
  · rw [if_neg hfl]
    cases hl : r.v.lookup n with
    | some v => exact wpE_pure (h v Γ (Extends.refl _) (hE.val n v hl))
    | none =>
        refine wpE_val (v := .sc .i64 0) (by simp [Op'.erase, evalOp, ofInt]; rfl) ?_
        exact h _ _ (Extends.push _ _) (by rw [hE.val0 n (by simpa using hfl) hl]; simp)

/-- **An integer operation**: its value on the operands'. -/
theorem wpE_opOf {Γ : Env} {w : World} (o : IOp) {a b : Slot .i64} {x y : UInt64}
    {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (o.run x y)) → Q v Γ' w) :
    wpE cfg (Body.opOf o a b) Q Γ w := by
  cases o <;> unfold Body.opOf <;>
    refine wpE_val (v := .sc .i64 _) ?_ (h _ _ (Extends.push _ _) (get_push_size)) <;>
    simp [Op'.erase, evalOp, Sem.get, ha, hb, IOp.run, norm, ibinOp, and_wide64, ClifTy.isInt, bin,
      shiftBin, zipIntBits, ClifTy.width, shr_mod64, min_wide]

theorem cmpOf_run (k : ICmp) (t : ClifTy) (x y : UInt64) : cmpInt (Body.cmpOf k) t x y = k.run x y := by
  cases k <;> simp [Body.cmpOf, cmpInt, ICmp.run] <;> omega

/-! ### 32-bit words -/

theorem widthMask_i32 : widthMask .i32 = 0xffffffff := rfl

theorem toNat_m32 (a : UInt64) : (a &&& 0xffffffff).toNat = a.toNat % 2 ^ 32 := by
  rw [UInt64.toNat_and, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]

theorem m32_of_lt {a : UInt64} (h : a < 2 ^ 32) : a &&& 0xffffffff = a := by
  apply UInt64.toNat_inj.mp
  rw [toNat_m32]
  have : a.toNat < 2 ^ 32 := UInt64.lt_iff_toNat_lt.mp h
  omega

theorem m32_m32 (a : UInt64) : (a &&& 0xffffffff) &&& 0xffffffff = a &&& 0xffffffff := by
  apply UInt64.toNat_inj.mp; simp only [toNat_m32]; omega

theorem m32_lt (a : UInt64) : a &&& 0xffffffff < 2 ^ 32 := by
  apply UInt64.lt_iff_toNat_lt.mpr; rw [toNat_m32]; exact Nat.mod_lt _ (by decide)

theorem m32_add (a b : UInt64) : ((a &&& 0xffffffff) + (b &&& 0xffffffff)) &&& 0xffffffff = (a + b) &&& 0xffffffff := by
  apply UInt64.toNat_inj.mp; simp only [toNat_m32, UInt64.toNat_add]; omega

theorem m32_sub (a b : UInt64) : ((a &&& 0xffffffff) - (b &&& 0xffffffff)) &&& 0xffffffff = (a - b) &&& 0xffffffff := by
  apply UInt64.toNat_inj.mp; simp only [toNat_m32, UInt64.toNat_sub]
  have := a.toNat_lt; have := b.toNat_lt; omega

theorem m32_mul (a b : UInt64) : ((a &&& 0xffffffff) * (b &&& 0xffffffff)) &&& 0xffffffff = (a * b) &&& 0xffffffff := by
  apply UInt64.toNat_inj.mp
  simp only [toNat_m32, UInt64.toNat_mul]
  rw [Nat.mod_mod_of_dvd _ (by decide : 2 ^ 32 ∣ 2 ^ 64), Nat.mod_mod_of_dvd _ (by decide : 2 ^ 32 ∣ 2 ^ 64),
    Nat.mul_mod, Nat.mod_mod, Nat.mod_mod, ← Nat.mul_mod]

theorem m32_and (a b : UInt64) : ((a &&& 0xffffffff) &&& (b &&& 0xffffffff)) &&& 0xffffffff = (a &&& b) &&& 0xffffffff := by
  ac_rfl

theorem m32_or (a b : UInt64) : ((a &&& 0xffffffff) ||| (b &&& 0xffffffff)) &&& 0xffffffff = (a ||| b) &&& 0xffffffff := by
  apply UInt64.toBitVec_inj.mp; simp only [UInt64.toBitVec_and, UInt64.toBitVec_or]
  ext i hi; simp only [BitVec.getElem_and, BitVec.getElem_or]; cases a.toBitVec[i] <;> cases b.toBitVec[i] <;> simp

theorem m32_xor (a b : UInt64) : ((a &&& 0xffffffff) ^^^ (b &&& 0xffffffff)) &&& 0xffffffff = (a ^^^ b) &&& 0xffffffff := by
  apply UInt64.toBitVec_inj.mp; simp only [UInt64.toBitVec_and, UInt64.toBitVec_xor]
  ext i hi; simp only [BitVec.getElem_and, BitVec.getElem_xor]; cases a.toBitVec[i] <;> cases b.toBitVec[i] <;> simp

theorem ctz32_le (y : UInt64) : ctz32 y ≤ 32 := by
  unfold ctz32
  apply UInt64.le_iff_toNat_le.mpr
  have : ((List.range 32).find? (fun k => (y >>> UInt64.ofNat k) &&& 1 == 1)).getD 32 ≤ 32 := by
    cases h : (List.range 32).find? (fun k => (y >>> UInt64.ofNat k) &&& 1 == 1) with
    | none => simp
    | some k => have := List.mem_range.mp (List.mem_of_find?_eq_some h); simp; omega
  simp only [UInt64.toNat_ofNat]
  exact Nat.le_trans (Nat.mod_le _ _) this

theorem ctz32_lt (y : UInt64) : ctz32 y < 2 ^ 32 :=
  UInt64.lt_of_le_of_lt (ctz32_le y) (by decide)

theorem ofInt_c32w (k : Nat) : ofInt .i32 ((k % 2 ^ 32 : Nat) : Int) = .sc .i32 (UInt64.ofNat k &&& 0xffffffff) := by
  simp only [ofInt, norm]
  congr 1
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, UInt64.toNat_and, show (widthMask .i32).toNat = 2 ^ 32 - 1 from rfl,
    show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    Nat.and_two_pow_sub_one_eq_mod, UInt64.toNat_ofNat', UInt64.toNat_ofNat']
  have h1 : (((k % 2 ^ 32 : Nat) : Int).emod (1 <<< 64 : Nat)).toNat = k % 2 ^ 32 := by
    show (((k % 2 ^ 32 : Nat) : Int) % ((1 <<< 64 : Nat) : Int)).toNat = k % 2 ^ 32
    rw [Int.emod_eq_of_lt (by omega) (by rw [show (1 <<< 64 : Nat) = 2 ^ 64 from rfl]; omega)]
    omega
  rw [h1]; omega

/-- A narrow expression's value is below `2^32`. -/
theorem narrow_lt {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (hE : EnvI c r reg Γ) :
    ∀ e : IExp, Body.narrow r e = true → e.evalL x reg < 2 ^ 32
  | .reg n, h => by
      simp only [Body.narrow, Option.isSome_iff_exists] at h
      obtain ⟨v, hv⟩ := h
      exact (hE.word n v hv).2
  | .lit k, h => by
      simp only [Body.narrow, decide_eq_true_eq] at h
      simp only [IExp.evalL]
      have : (UInt64.ofNat k).toNat < 2 ^ 32 := by rw [UInt64.toNat_ofNat']; omega
      exact UInt64.lt_iff_toNat_lt.mpr this
  | .ctz _, _ => ctz32_lt _
  | .lane, h | .cta, h | .ptr _, h | .bin .., h | .cmp .., h => by simp [Body.narrow] at h

theorem cmp_m32 (k : ICmp) {a b : UInt64} (ha : a < 2 ^ 32) (hb : b < 2 ^ 32) :
    k.run (a &&& 0xffffffff) (b &&& 0xffffffff) = k.run a b := by
  rw [m32_of_lt ha, m32_of_lt hb]

/-! ### Expressions -/

theorem wpE_uext32 {Γ : Env} {w : World} {a : Slot .i32} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (uextend64 a) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, widthMask]) (xget h)

theorem wpE_ireduce32 {Γ : Env} {w : World} {a : Slot .i64} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 (x &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (ireduce32 a) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, norm, widthMask]) (xget h)

theorem wpE_ctz32 {Γ : Env} {w : World} {a : Slot .i32} {y : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 (ctz32 y)) → Q s Γ' w) :
    wpE cfg (ctz a) Q Γ w := by
  refine wpE_val ?_ (xget h)
  have hm := m32_of_lt (ctz32_lt y)
  simp only [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, norm]
  rw [← hm]; rfl

theorem wpE_bin32 {Γ : Env} {w : World} {a b : Slot .i32} {x y : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (x &&& 0xffffffff))) (hb : Γ[b]? = some (.sc .i32 (y &&& 0xffffffff)))
    (o : IOp) (ho : o = .add ∨ o = .sub ∨ o = .mul ∨ o = .band ∨ o = .bor ∨ o = .bxor)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 (o.run x y &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (match o with
      | .add => iadd a b | .sub => isub a b | .mul => imul a b
      | .band => band a b | .bor => bor a b | _ => bxor a b : Prog Slot Lvl (Slot .i32)) Q Γ w := by
  rcases ho with rfl | rfl | rfl | rfl | rfl | rfl <;>
    refine wpE_val ?_ (xget h) <;>
    simp only [Op'.erase, evalOp, Sem.get, ha, hb, bin, zipIntBits, ClifTy.isInt, norm, IOp.run] <;>
    simp [m32_add, m32_sub, m32_mul, m32_and, m32_or, m32_xor, widthMask_i32]

theorem wpE_icmp64 {Γ : Env} {w : World} (cc : ICmpCond) {a b : Slot .i64}
    {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (boolV (cmpInt cc .i64 x y)) → Q s Γ' w) :
    wpE cfg (icmp cc a b) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipIntCmp, ClifTy.isInt]) (xget h)

theorem wpE_icmp32 {Γ : Env} {w : World} (cc : ICmpCond) {a b : Slot .i32}
    {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (boolV (cmpInt cc .i32 x y)) → Q s Γ' w) :
    wpE cfg (icmp cc a b) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipIntCmp, ClifTy.isInt]) (xget h)

/-- **An integer expression's code ends with its value**, and its low-32 code
    with its low 32 bits, in a slot of an environment keeping `Γ`. -/
theorem wpE_exp {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) : ∀ e : IExp,
    (∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
      (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 (e.evalL x reg)) → Q v Γ' w) →
      wpE cfg (Body.iexp c r e) Q Γ0 w) ∧
    (∀ {Γ0 : Env} {Q : Slot .i32 → Env → World → Prop}, Extends Γ Γ0 →
      (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i32 (e.evalL x reg &&& 0xffffffff)) → Q v Γ' w) →
      wpE cfg (Body.low32 c r e) Q Γ0 w) := by
  have red : ∀ (e : IExp), (∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
      (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 (e.evalL x reg)) → Q v Γ' w) →
      wpE cfg (Body.iexp c r e) Q Γ0 w) →
      ∀ {Γ0 : Env} {Q : Slot .i32 → Env → World → Prop}, Extends Γ Γ0 →
      (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i32 (e.evalL x reg &&& 0xffffffff)) → Q v Γ' w) →
      wpE cfg (do ireduce32 (← Body.iexp c r e)) Q Γ0 w := by
    intro e hI Γ0 Q hx0 hQ
    exact wpE_bind (hI hx0 fun v Γ1 hx1 hv => wpE_ireduce32 hv fun s Γ2 hx2 hs => hQ s Γ2 (hx1.trans hx2) hs)
  intro e
  induction e with
  | reg n =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 ((IExp.reg n).evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r (.reg n)) Q Γ0 w :=
        fun hx0 hQ => by simp only [Body.iexp]; exact wpE_getI (hE.extend hx0) hQ
      refine ⟨hI, fun {Γ0 Q} hx0 hQ => ?_⟩
      simp only [Body.low32]
      cases hw : r.w.lookup n with
      | some v =>
          have hv := hE.word n v hw
          exact wpE_pure (hQ v Γ0 (Extends.refl _) (by rw [hx0.get hv.1, IExp.evalL, m32_of_lt hv.2]))
      | none =>
          simp only
          exact wpE_bind (wpE_getI (hE.extend hx0) fun v Γ1 hx1 hv =>
            wpE_ireduce32 hv fun s Γ2 hx2 hs => hQ s Γ2 (hx1.trans hx2) (by rw [hs]; rfl))
  | lit k =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 ((IExp.lit k).evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r (.lit k)) Q Γ0 w :=
        fun _ hQ => by
          simp only [Body.iexp]
          exact wpE_iconst64 (xget fun s Γ' hx hs => hQ s Γ' hx (by rw [hs, IExp.evalL, c64_ofNat]))
      refine ⟨hI, fun {Γ0 Q} _ hQ => ?_⟩
      simp only [Body.low32, Body.c32w]
      exact wpE_val (v := .sc .i32 (UInt64.ofNat k &&& 0xffffffff))
        (by simp only [Op'.erase, evalOp]; exact congrArg some (ofInt_c32w k)) (xget hQ)
  | lane =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 (IExp.lane.evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r .lane) Q Γ0 w :=
        fun hx0 hQ => by
          simp only [Body.iexp]
          exact wpE_uext32 (hx0.get hS.lane) fun s Γ' hx hs => hQ s Γ' hx (by rw [hs, m32_of_lt hS.lane32]; rfl)
      exact ⟨hI, fun hx0 hQ => by simp only [Body.low32]; exact red _ hI hx0 hQ⟩
  | cta =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 (IExp.cta.evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r .cta) Q Γ0 w :=
        fun hx0 hQ => by
          simp only [Body.iexp]
          exact wpE_uext32 (hx0.get hS.cta) fun s Γ' hx hs => hQ s Γ' hx (by rw [hs, m32_of_lt hS.cta32]; rfl)
      exact ⟨hI, fun hx0 hQ => by simp only [Body.low32]; exact red _ hI hx0 hQ⟩
  | ptr b =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 ((IExp.ptr b).evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r (.ptr b)) Q Γ0 w :=
        fun hx0 hQ => by
          simp only [Body.iexp]
          exact wpE_pure (hQ _ _ (Extends.refl _) (hx0.get (hS.buf b)))
      exact ⟨hI, fun hx0 hQ => by simp only [Body.low32]; exact red _ hI hx0 hQ⟩
  | bin o a b iha ihb =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 ((IExp.bin o a b).evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r (.bin o a b)) Q Γ0 w := by
        intro Γ0 Q hx0 hQ
        simp only [Body.iexp]
        refine wpE_bind (iha.1 hx0 fun va Γ1 hx1 hva => ?_)
        refine wpE_bind (ihb.1 (hx0.trans hx1) fun vb Γ2 hx2 hvb => ?_)
        exact wpE_opOf o (hx2.get hva) hvb fun v Γ3 hx3 hv => hQ v Γ3 (hx1.trans (hx2.trans hx3)) hv
      refine ⟨hI, fun {Γ0 Q} hx0 hQ => ?_⟩
      have hb32 : ∀ (o' : IOp), o' = .add ∨ o' = .sub ∨ o' = .mul ∨ o' = .band ∨ o' = .bor ∨ o' = .bxor →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i32 (o'.run (a.evalL x reg) (b.evalL x reg) &&& 0xffffffff)) →
            Q v Γ' w) →
          wpE cfg (do
            let p ← Body.low32 c r a
            let q ← Body.low32 c r b
            (match o' with
              | .add => iadd p q | .sub => isub p q | .mul => imul p q
              | .band => band p q | .bor => bor p q | _ => bxor p q : Prog Slot Lvl (Slot .i32))) Q Γ0 w := by
        intro o' ho' hQ'
        refine wpE_bind (iha.2 hx0 fun va Γ1 hx1 hva => ?_)
        refine wpE_bind (ihb.2 (hx0.trans hx1) fun vb Γ2 hx2 hvb => ?_)
        exact wpE_bin32 (hx2.get hva) hvb o' ho' fun v Γ3 hx3 hv => hQ' v Γ3 (hx1.trans (hx2.trans hx3)) hv
      cases o with
      | add => simp only [Body.low32]; exact hb32 .add (by simp) hQ
      | sub => simp only [Body.low32]; exact hb32 .sub (by simp) hQ
      | mul => simp only [Body.low32]; exact hb32 .mul (by simp) hQ
      | band => simp only [Body.low32]; exact hb32 .band (by simp) hQ
      | bor => simp only [Body.low32]; exact hb32 .bor (by simp) hQ
      | bxor => simp only [Body.low32]; exact hb32 .bxor (by simp) hQ
      | shl | shr | umin => simp only [Body.low32]; exact red _ hI hx0 hQ
  | cmp k a b iha ihb =>
      have hI : ∀ {Γ0 : Env} {Q : Slot .i64 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i64 ((IExp.cmp k a b).evalL x reg)) → Q v Γ' w) →
          wpE cfg (Body.iexp c r (.cmp k a b)) Q Γ0 w := by
        intro Γ0 Q hx0 hQ
        have hrest : ∀ (p : Slot .i8) Γ1, Extends Γ0 Γ1 → Γ1[p]? = some (boolV (k.run (a.evalL x reg) (b.evalL x reg))) →
            wpE cfg (do select p (← iconst64 1) (← iconst64 0)) Q Γ1 w := by
          intro p Γ1 hx1 hp
          refine wpE_bind (wpE_iconst64 ?_)
          refine wpE_bind (wpE_iconst64 ?_)
          refine wpE_val (v := .sc .i64 ((IExp.cmp k a b).evalL x reg)) ?_ ?_
          · simp only [Op'.erase]; rw [eval_selB _ Γ1 p _ hp]; rfl
          · exact hQ _ _ (((hx1.push_of _).push_of _).push_of _) (get_push_size)
        simp only [Body.iexp]
        by_cases hn : (Body.narrow r a && Body.narrow r b) = true
        · simp only [hn, if_true]
          have hn' := hn
          simp only [Bool.and_eq_true] at hn'
          refine wpE_bind (iha.2 hx0 fun va Γ2 hx2 hva => ?_)
          refine wpE_bind (ihb.2 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
          refine wpE_bind (wpE_icmp32 _ (hx3.get hva) hvb fun p Γ4 hx4 hp => hrest p Γ4 (hx2.trans (hx3.trans hx4)) ?_)
          rw [hp, cmpOf_run, cmp_m32 k (narrow_lt hE a hn'.1) (narrow_lt hE b hn'.2)]
        · simp only [hn, Bool.false_eq_true, if_false]
          refine wpE_bind (iha.1 hx0 fun va Γ2 hx2 hva => ?_)
          refine wpE_bind (ihb.1 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
          refine wpE_bind (wpE_icmp64 _ (hx3.get hva) hvb fun p Γ4 hx4 hp => hrest p Γ4 (hx2.trans (hx3.trans hx4)) ?_)
          rw [hp, cmpOf_run]
      exact ⟨hI, fun hx0 hQ => by simp only [Body.low32]; exact red _ hI hx0 hQ⟩
  | ctz a ih =>
      have hlow : ∀ {Γ0 : Env} {Q : Slot .i32 → Env → World → Prop}, Extends Γ Γ0 →
          (∀ v Γ', Extends Γ0 Γ' → Γ'[v]? = some (.sc .i32 (ctz32 (a.evalL x reg &&& 0xffffffff))) → Q v Γ' w) →
          wpE cfg (do ctz (← Body.low32 c r a)) Q Γ0 w := by
        intro Γ0 Q hx0 hQ
        refine wpE_bind (ih.2 hx0 fun va Γ1 hx1 hva => ?_)
        exact wpE_ctz32 hva fun v Γ2 hx2 hv => hQ v Γ2 (hx1.trans hx2) hv
      refine ⟨fun hx0 hQ => ?_, fun hx0 hQ => ?_⟩
      · simp only [Body.iexp]
        refine wpE_bind (ih.2 hx0 fun va Γ1 hx1 hva => ?_)
        refine wpE_bind (wpE_ctz32 hva fun v Γ2 hx2 hv => ?_)
        exact wpE_uext32 hv fun s Γ3 hx3 hs => hQ s Γ3 (hx1.trans (hx2.trans hx3))
          (by rw [hs, m32_of_lt (ctz32_lt _)]; rfl)
      · simp only [Body.low32]
        exact hlow hx0 fun v Γ1 hx1 hv => hQ v Γ1 hx1 (by rw [hv, IExp.evalL, m32_of_lt (ctz32_lt _)])

/-! ### Predicates and tests -/

/-- An expression `isBool` says is `0` or `1` is so. -/
theorem isBool_bit {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (hE : EnvI c r reg Γ) :
    ∀ e : IExp, e.isBool c.fl = true → e.evalL x reg ≤ 1
  | .cmp k a b, _ => by simp only [IExp.evalL]; split <;> decide
  | .reg n, h => by simp only [IExp.isBool] at h; exact hE.bit n h
  | .bin .band a b, h | .bin .bor a b, h | .bin .bxor a b, h => by
      simp only [IExp.isBool, Bool.and_eq_true] at h
      have ha := isBool_bit hE a h.1
      have hb := isBool_bit hE b h.2
      simp only [IExp.evalL, IOp.run]
      rcases flag_cases _ ha with h1 | h1 <;> rcases flag_cases _ hb with h2 | h2 <;> rw [h1, h2] <;> decide
  | .lit _, h | .lane, h | .cta, h | .ptr _, h | .ctz _, h => by simp [IExp.isBool] at h
  | .bin .add _ _, h | .bin .sub _ _, h | .bin .mul _ _, h | .bin .shl _ _, h | .bin .shr _ _, h
  | .bin .umin _ _, h => by simp [IExp.isBool] at h

theorem boolV_cmp (k : ICmp) (a b : UInt64) :
    boolV ((if k.run a b then (1 : UInt64) else 0) != 0) = boolV (k.run a b) := by
  cases k.run a b <;> rfl

/-- **A comparison's predicate**, of the 32-bit words where both sides are narrow. -/
theorem wpE_icmpE {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ Γ0 : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) (hx0 : Extends Γ Γ0) (k : ICmp) (a b : IExp)
    {Q : Slot .i8 → Env → World → Prop}
    (h : ∀ p Γ', Extends Γ0 Γ' → Γ'[p]? = some (boolV (k.run (a.evalL x reg) (b.evalL x reg))) → Q p Γ' w) :
    wpE cfg (Body.icmpE c r k a b) Q Γ0 w := by
  unfold Body.icmpE
  by_cases hn : (Body.narrow r a && Body.narrow r b) = true
  · simp only [hn, if_true]
    have hn' := hn
    simp only [Bool.and_eq_true] at hn'
    refine wpE_bind ((wpE_exp hE hS a).2 hx0 fun va Γ2 hx2 hva => ?_)
    refine wpE_bind ((wpE_exp hE hS b).2 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
    refine wpE_icmp32 _ (hx3.get hva) hvb fun p Γ4 hx4 hp => h p Γ4 (hx2.trans (hx3.trans hx4)) ?_
    rw [hp, cmpOf_run, cmp_m32 k (narrow_lt hE a hn'.1) (narrow_lt hE b hn'.2)]
  · simp only [hn, Bool.false_eq_true, if_false]
    refine wpE_bind ((wpE_exp hE hS a).1 hx0 fun va Γ2 hx2 hva => ?_)
    refine wpE_bind ((wpE_exp hE hS b).1 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
    refine wpE_icmp64 _ (hx3.get hva) hvb fun p Γ4 hx4 hp => h p Γ4 (hx2.trans (hx3.trans hx4)) ?_
    rw [hp, cmpOf_run]

theorem wpE_ne0 {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ Γ0 : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) (hx0 : Extends Γ Γ0) (e : IExp)
    {Q : Slot .i8 → Env → World → Prop}
    (h : ∀ p Γ', Extends Γ0 Γ' → Γ'[p]? = some (boolV (e.evalL x reg != 0)) → Q p Γ' w) :
    wpE cfg (do icmp .ne (← Body.iexp c r e) (← iconst64 0)) Q Γ0 w := by
  refine wpE_bind ((wpE_exp hE hS e).1 hx0 fun v Γ1 hx1 hv => ?_)
  refine wpE_bind (wpE_iconst64 ?_)
  refine wpE_icmp64 _ ((Extends.push _ _).get hv) (get_push_size) fun p Γ2 hx2 hp =>
    h p Γ2 (hx1.trans ((Extends.push _ _).trans hx2)) ?_
  rw [hp]; rfl

theorem band8 (p q : UInt64) (hp : p ≤ 1) (hq : q ≤ 1) :
    zipIntBits (boolV (p != 0)) (boolV (q != 0)) (fun _ x y => x &&& y) = some (boolV ((p &&& q) != 0)) := by
  rcases flag_cases _ hp with rfl | rfl <;> rcases flag_cases _ hq with rfl | rfl <;> rfl

theorem bor8 (p q : UInt64) (hp : p ≤ 1) (hq : q ≤ 1) :
    zipIntBits (boolV (p != 0)) (boolV (q != 0)) (fun _ x y => x ||| y) = some (boolV ((p ||| q) != 0)) := by
  rcases flag_cases _ hp with rfl | rfl <;> rcases flag_cases _ hq with rfl | rfl <;> rfl

theorem bxor8 (p q : UInt64) (hp : p ≤ 1) (hq : q ≤ 1) :
    zipIntBits (boolV (p != 0)) (boolV (q != 0)) (fun _ x y => x ^^^ y) = some (boolV ((p ^^^ q) != 0)) := by
  rcases flag_cases _ hp with rfl | rfl <;> rcases flag_cases _ hq with rfl | rfl <;> rfl

/-- **An expression's predicate**: whether it is not zero. -/
theorem wpE_ipred {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) : ∀ (e : IExp) {Γ0 : Env} {Q : Slot .i8 → Env → World → Prop},
    Extends Γ Γ0 → (∀ p Γ', Extends Γ0 Γ' → Γ'[p]? = some (boolV (e.evalL x reg != 0)) → Q p Γ' w) →
    wpE cfg (Body.ipred c r e) Q Γ0 w := by
  intro e
  induction e with
  | cmp k a b _ _ =>
      intro Γ0 Q hx0 h
      simp only [Body.ipred]
      exact wpE_icmpE hE hS hx0 k a b fun p Γ' hx hp => h p Γ' hx (by rw [hp, IExp.evalL, boolV_cmp])
  | reg n =>
      intro Γ0 Q hx0 h
      simp only [Body.ipred]
      exact wpE_getP (hE.extend hx0) h
  | bin o a b iha ihb =>
      intro Γ0 Q hx0 h
      simp only [Body.ipred]
      by_cases hc : ((o == .band || o == .bor || o == .bxor) && (IExp.bin o a b).isBool c.fl) = true
      · rw [if_pos hc]
        simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hc
        obtain ⟨ho, hb⟩ := hc
        have hb' := hb
        have hab : a.isBool c.fl = true ∧ b.isBool c.fl = true := by
          rcases ho with (rfl | rfl) | rfl <;> simpa [IExp.isBool] using hb'
        have ha1 := isBool_bit (x := x) hE a hab.1
        have hb1 := isBool_bit (x := x) hE b hab.2
        refine wpE_bind (iha hx0 fun pa Γ1 hx1 hpa => ?_)
        refine wpE_bind (ihb (hx0.trans hx1) fun pb Γ2 hx2 hpb => ?_)
        rcases ho with (rfl | rfl) | rfl
        · exact wpE_val (by simp only [Op'.erase, evalOp, Sem.get, hx2.get hpa, hpb]; exact band8 _ _ ha1 hb1)
            (xget fun s Γ3 hx3 hs => h s Γ3 (hx1.trans (hx2.trans hx3)) hs)
        · exact wpE_val (by simp only [Op'.erase, evalOp, Sem.get, hx2.get hpa, hpb]; exact bor8 _ _ ha1 hb1)
            (xget fun s Γ3 hx3 hs => h s Γ3 (hx1.trans (hx2.trans hx3)) hs)
        · exact wpE_val (by simp only [Op'.erase, evalOp, Sem.get, hx2.get hpa, hpb]; exact bxor8 _ _ ha1 hb1)
            (xget fun s Γ3 hx3 hs => h s Γ3 (hx1.trans (hx2.trans hx3)) hs)
      · rw [if_neg hc]
        exact wpE_ne0 hE hS hx0 _ h
  | lit _ | lane | cta | ptr _ =>
      intro Γ0 Q hx0 h
      simp only [Body.ipred]
      exact wpE_ne0 hE hS hx0 _ h
  | ctz _ _ =>
      intro Γ0 Q hx0 h
      simp only [Body.ipred]
      exact wpE_ne0 hE hS hx0 _ h

/-- **A test's comparison**: two values of one integer type whose comparison
    is whether the expression is not zero. -/
theorem wpE_condOf {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ Γ0 : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) (hx0 : Extends Γ Γ0) (e : IExp)
    {Q : ((ty : ClifTy) × ICmpCond × Slot ty × Slot ty) → Env → World → Prop}
    (h : ∀ (t : ClifTy) (k : ICmpCond) (a b : Slot t) (X Y : UInt64) (Γ' : Env), Extends Γ0 Γ' →
      t.isInt = true → Γ'[a]? = some (.sc t X) → Γ'[b]? = some (.sc t Y) →
      cmpInt k t X Y = (e.evalL x reg != 0) → Q ⟨t, k, a, b⟩ Γ' w) :
    wpE cfg (Body.condOf c r e) Q Γ0 w := by
  have hgen : (Body.condOf (L := Lvl) c r e = do
      let p ← Body.ipred (L := Lvl) c r e
      pure ⟨.i8, .ne, p, ← Body.no (L := Lvl)⟩) → wpE cfg (Body.condOf c r e) Q Γ0 w := by
    intro he'
    rw [he']
    refine wpE_bind (wpE_ipred hE hS e hx0 fun p Γ1 hx1 hp => ?_)
    have hfin : ∀ (n : Slot .i8) Γ2, Extends Γ1 Γ2 → Γ2[n]? = some (.sc .i8 0) →
        Q ⟨.i8, .ne, p, n⟩ Γ2 w := by
      intro n Γ2 hx2 hn
      refine h .i8 .ne p n (if e.evalL x reg != 0 then 1 else 0) 0 Γ2 (hx1.trans hx2) rfl ?_ hn ?_
      · rw [hx2.get hp]; unfold boolV; rfl
      · cases e.evalL x reg != 0 <;> rfl
    exact wpE_bind (iconst8_zero fun n Γ2 hx2 hn => wpE_pure (hfin n Γ2 hx2 hn))
  cases e with
  | cmp k a b =>
      simp only [Body.condOf]
      by_cases hn : (Body.narrow r a && Body.narrow r b) = true
      · simp only [hn, if_true]
        have hn' := hn
        simp only [Bool.and_eq_true] at hn'
        refine wpE_bind ((wpE_exp hE hS a).2 hx0 fun va Γ2 hx2 hva => ?_)
        refine wpE_bind ((wpE_exp hE hS b).2 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
        refine wpE_pure (h .i32 _ va vb _ _ Γ3 (hx2.trans hx3) rfl (hx3.get hva) hvb ?_)
        rw [cmpOf_run, cmp_m32 k (narrow_lt hE a hn'.1) (narrow_lt hE b hn'.2), IExp.evalL]
        cases k.run (a.evalL x reg) (b.evalL x reg) <;> rfl
      · simp only [hn, Bool.false_eq_true, if_false]
        refine wpE_bind ((wpE_exp hE hS a).1 hx0 fun va Γ2 hx2 hva => ?_)
        refine wpE_bind ((wpE_exp hE hS b).1 (hx0.trans hx2) fun vb Γ3 hx3 hvb => ?_)
        refine wpE_pure (h .i64 _ va vb _ _ Γ3 (hx2.trans hx3) rfl (hx3.get hva) hvb ?_)
        rw [cmpOf_run, IExp.evalL]
        cases k.run (a.evalL x reg) (b.evalL x reg) <;> rfl
  | reg _ | lit _ | lane | cta | ptr _ | bin _ _ _ | ctz _ => exact hgen rfl

/-! ## The environment through a statement -/

theorem lookup_cons_ne {α : Type} {r d : Nat} {v : α} {l : List (Nat × α)} (h : r ≠ d) :
    ((d, v) :: l).lookup r = l.lookup r := by
  simp [List.lookup, show (r == d) = false by simpa using h]

theorem lookup_cons_eq {α : Type} {d : Nat} {v : α} {l : List (Nat × α)} : ((d, v) :: l).lookup d = some v := by
  simp [List.lookup]

theorem lookup_filter_ne {α : Type} {n : Nat} :
    ∀ (l : List (Nat × α)) (r : Nat), (l.filter (·.1 != n)).lookup r = if r = n then none else l.lookup r
  | [], r => by split <;> rfl
  | (d, v) :: l, r => by
      by_cases hd : d = n
      · subst hd
        simp only [List.filter, bne_self_eq_false]
        rw [lookup_filter_ne l r]
        by_cases hr : r = d
        · simp [hr]
        · simp [hr, lookup_cons_ne hr]
      · simp only [List.filter, show (d != n) = true by simpa using hd]
        by_cases hr : r = d
        · subst hr; simp [hd]
        · rw [lookup_cons_ne hr, lookup_filter_ne l r, lookup_cons_ne hr]

theorem EnvI.congr {r : Body.Regs Slot} {reg reg' : Nat → UInt64} {Γ : Env} (h : EnvI c r reg Γ)
    (he : ∀ n, reg n = reg' n) : EnvI c r reg' Γ := by
  have : reg = reg' := funext he
  subst this; exact h

/-- A value register bound to a slot holding its new value. -/
theorem EnvI.setV {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (h : EnvI c r reg Γ) (d : Nat)
    (hd : c.fl.contains d = false) (v : Slot .i64) (X : UInt64) (hv : Γ[v]? = some (.sc .i64 X)) :
    EnvI c (Body.setV r d v) (fun n => if n = d then X else reg n) Γ where
  val n u hu := by
    simp only [Body.setV] at hu
    by_cases hn : n = d
    · subst hn; rw [lookup_cons_eq] at hu; cases hu; simpa using hv
    · rw [lookup_cons_ne hn] at hu; rw [if_neg hn]; exact h.val n u hu
  flag n u hu := by
    have hn : n ≠ d := fun e => by subst e; have := h.fkind n u hu; rw [hd] at this; cases this
    rw [if_neg hn]; exact h.flag n u hu
  word n u hu := by
    simp only [Body.setV] at hu
    rw [lookup_filter_ne] at hu
    by_cases hn : n = d
    · simp [hn] at hu
    · rw [if_neg hn] at hu ⊢; exact h.word n u hu
  bit n hn := by
    have hnd : n ≠ d := fun e => by subst e; rw [hd] at hn; cases hn
    rw [if_neg hnd]; exact h.bit n hn
  flag0 n hn hl := by
    have hnd : n ≠ d := fun e => by subst e; rw [hd] at hn; cases hn
    rw [if_neg hnd]; exact h.flag0 n hn hl
  val0 n hn hl := by
    simp only [Body.setV] at hl
    by_cases hnd : n = d
    · subst hnd; rw [lookup_cons_eq] at hl; cases hl
    · rw [lookup_cons_ne hnd] at hl; rw [if_neg hnd]; exact h.val0 n hn hl
  fkind n u hu := h.fkind n u hu
  vkind n u hu := by
    simp only [Body.setV] at hu
    by_cases hnd : n = d
    · subst hnd; exact hd
    · rw [lookup_cons_ne hnd] at hu; exact h.vkind n u hu
  wkind n u hu := by
    simp only [Body.setV] at hu
    rw [lookup_filter_ne] at hu
    split at hu
    · cases hu
    · exact h.wkind n u hu

/-- A value register bound to a slot holding its new value, a 32-bit word,
    and to a slot holding that word. -/
theorem EnvI.setW {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (h : EnvI c r reg Γ) (d : Nat)
    (hd : c.fl.contains d = false) (v : Slot .i64) (v32 : Slot .i32) (X : UInt64) (hX : X < 2 ^ 32)
    (hv : Γ[v]? = some (.sc .i64 X)) (hv32 : Γ[v32]? = some (.sc .i32 X)) :
    EnvI c (Body.setW r d v v32) (fun n => if n = d then X else reg n) Γ := by
  have h1 := h.setV d hd v X hv
  exact { h1 with
    word := fun n u hu => by
      simp only [Body.setW] at hu
      by_cases hn : n = d
      · subst hn; rw [lookup_cons_eq] at hu; cases hu; simp [hv32, hX]
      · rw [lookup_cons_ne hn, lookup_filter_ne, if_neg hn] at hu; rw [if_neg hn]; exact h.word n u hu
    wkind := fun n u hu => by
      simp only [Body.setW] at hu
      by_cases hn : n = d
      · subst hn; exact hd
      · rw [lookup_cons_ne hn, lookup_filter_ne, if_neg hn] at hu; exact h.wkind n u hu }

/-- A predicate register bound to a slot holding its new bit. -/
theorem EnvI.setP {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} (h : EnvI c r reg Γ) (d : Nat)
    (hd : c.fl.contains d = true) (v : Slot .i8) (X : UInt64) (hX : X ≤ 1) (hv : Γ[v]? = some (.sc .i8 X)) :
    EnvI c (Body.setP r d v) (fun n => if n = d then X else reg n) Γ where
  val n u hu := by
    have hn : n ≠ d := fun e => by subst e; have := h.vkind n u hu; rw [hd] at this; cases this
    rw [if_neg hn]; exact h.val n u hu
  flag n u hu := by
    simp only [Body.setP] at hu
    by_cases hn : n = d
    · subst hn; rw [lookup_cons_eq] at hu; cases hu; simpa using hv
    · rw [lookup_cons_ne hn] at hu; rw [if_neg hn]; exact h.flag n u hu
  word n u hu := by
    have hn : n ≠ d := fun e => by subst e; have := h.wkind n u hu; rw [hd] at this; cases this
    rw [if_neg hn]; exact h.word n u hu
  bit n hn := by
    by_cases hnd : n = d
    · subst hnd; simpa using hX
    · rw [if_neg hnd]; exact h.bit n hn
  flag0 n hn hl := by
    simp only [Body.setP] at hl
    by_cases hnd : n = d
    · subst hnd; rw [lookup_cons_eq] at hl; cases hl
    · rw [lookup_cons_ne hnd] at hl; rw [if_neg hnd]; exact h.flag0 n hn hl
  val0 n hn hl := by
    have hnd : n ≠ d := fun e => by subst e; rw [hd] at hn; cases hn
    rw [if_neg hnd]; exact h.val0 n hn hl
  fkind n u hu := by
    simp only [Body.setP] at hu
    by_cases hnd : n = d
    · subst hnd; exact hd
    · rw [lookup_cons_ne hnd] at hu; exact h.fkind n u hu
  vkind n u hu := h.vkind n u hu
  wkind n u hu := h.wkind n u hu

/-- The values a join or a loop carries: float registers `ws`, then value
    registers `rv`, then predicate registers `rf`. -/
def carried (st : LI) (ws rv rf : List Nat) : List V :=
  ws.map (fun n => .sc .f32 (st.f n).bits) ++
    (rv.map (fun n => .sc .i64 (st.reg n)) ++ rf.map (fun n => .sc .i8 (st.reg n)))

theorem appendV_slots {a b : List ClifTy} (u : Vals Slot a) (v : Vals Slot b) :
    (Body.appendV u v).slots = u.slots ++ v.slots := by
  induction u with
  | nil => rfl
  | cons y ys ih => simp [Body.appendV, Vals.slots, ih]

theorem splitV_carries {b : List ClifTy} : ∀ (a : List ClifTy) (n : Nat),
    Body.splitV a (carriesFrom n (a ++ b)) = (carriesFrom n a, carriesFrom (n + a.length) b)
  | [], n => rfl
  | _ :: a, n => by
      simp only [List.cons_append, carriesFrom, Body.splitV, splitV_carries a (n + 1), List.length_cons]
      rw [Nat.add_assoc, Nat.add_comm 1]

theorem wpE_packV {r : Body.Regs Slot} {reg : Nat → UInt64} {w : World} :
    ∀ (rs : List Nat) {Γ : Env} {Q : Vals Slot (rs.map fun _ => ClifTy.i64) → Env → World → Prop},
    EnvI c r reg Γ →
    (∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (rs.map fun n => .sc .i64 (reg n)) →
      Q pv Γ' w) →
    wpE cfg (Body.packV (L := Lvl) c r rs) Q Γ w
  | [], Γ, Q, _, h => wpE_pure (h _ _ (Extends.refl _) rfl)
  | n :: rs, Γ, Q, hE, h => by
      simp only [Body.packV]
      refine wpE_bind (wpE_getI hE fun v Γ1 hx1 hv => ?_)
      refine wpE_bind (wpE_packV rs (hE.extend hx1) fun pv Γ2 hx2 hpv => ?_)
      refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
      simp [Vals.slots, hx2.get hv, hpv]

theorem wpE_packP {r : Body.Regs Slot} {reg : Nat → UInt64} {w : World} :
    ∀ (rs : List Nat) {Γ : Env} {Q : Vals Slot (rs.map fun _ => ClifTy.i8) → Env → World → Prop},
    EnvI c r reg Γ → (∀ n ∈ rs, c.fl.contains n = true) →
    (∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (rs.map fun n => .sc .i8 (reg n)) →
      Q pv Γ' w) →
    wpE cfg (Body.packP (L := Lvl) c r rs) Q Γ w
  | [], Γ, Q, _, _, h => wpE_pure (h _ _ (Extends.refl _) rfl)
  | n :: rs, Γ, Q, hE, hf, h => by
      simp only [Body.packP]
      refine wpE_bind (wpE_getP hE fun v Γ1 hx1 hv => ?_)
      refine wpE_bind (wpE_packP rs (hE.extend hx1) (fun m hm => hf m (by simp [hm])) fun pv Γ2 hx2 hpv => ?_)
      refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
      rw [boolV_ne0 (hE.bit n (hf n (by simp)))] at hv
      simp [Vals.slots, hx2.get hv, hpv]

theorem packF_slots (r : Body.Regs Slot) : ∀ ws : List Nat, (Body.packF c r ws).slots = ws.map (Body.get c r)
  | [] => rfl
  | n :: ws => by simp [Body.packF, Vals.slots, packF_slots r ws]

theorem packF_read {r : Body.Regs Slot} {f : Nat → F32} {Γ : Env} (hF : FEnv c r f Γ) :
    ∀ ws : List Nat, (ws.map (Body.get c r)).mapM (fun r => Γ[r]?) = some (ws.map fun n => .sc .f32 (f n).bits)
  | [] => rfl
  | n :: ws => by simp [List.mapM_cons, hF.get n, packF_read hF ws]

/-- **The carried registers' values**: float registers `ws`, value registers
    `rv`, predicate registers `rf`. -/
theorem wpE_pack {r : Body.Regs Slot} {st : LI} {w : World} {Γ : Env}
    {ws rv rf : List Nat} {Q : Vals Slot (Body.carryTys ws rv rf) → Env → World → Prop}
    (hE : EnvI c r st.reg Γ) (hF : FEnv c r st.f Γ) (hf : ∀ n ∈ rf, c.fl.contains n = true)
    (h : ∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (carried st ws rv rf) → Q pv Γ' w) :
    wpE cfg (Body.pack (L := Lvl) c r ws rv rf) Q Γ w := by
  simp only [Body.pack]
  refine wpE_bind (wpE_packV rv hE fun a Γ1 hx1 ha => ?_)
  refine wpE_bind (wpE_packP rf (hE.extend hx1) hf fun b Γ2 hx2 hb => ?_)
  refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
  have ha' : a.slots.mapM (fun r => Γ2[r]?) = some (rv.map fun n => .sc .i64 (st.reg n)) := Sem.mapM_ext hx2 ha
  simp [appendV_slots, packF_slots, List.mapM_append, ha', hb, carried,
    packF_read (hF.extend (hx1.trans hx2)) ws]

theorem bindV_ok {Γ : Env} {reg' : Nat → UInt64} :
    ∀ (rs : List Nat) (k : Nat) (r : Body.Regs Slot) (reg : Nat → UInt64), EnvI c r reg Γ →
    (∀ n ∈ rs, c.fl.contains n = false) →
    (∀ (i : Nat) (h : i < rs.length), Γ[k + i]? = some (.sc .i64 (reg' rs[i]))) →
    EnvI c (Body.bindV r rs (carriesFrom k _)) (fun n => if n ∈ rs then reg' n else reg n) Γ
  | [], k, r, reg, hE, _, _ => by simpa [Body.bindV] using hE
  | d :: rs, k, r, reg, hE, hv, hs => by
      simp only [List.map_cons, carriesFrom, Body.bindV]
      have h1 := hE.setV d (hv d (by simp)) k (reg' d) (by simpa using hs 0 (by simp))
      have h2 := bindV_ok rs (k + 1) _ _ h1 (fun n hn => hv n (by simp [hn]))
        (fun i hi => by simpa [Nat.add_assoc, Nat.add_comm 1 i] using hs (i + 1) (by simp; omega))
      exact h2.congr fun n => by by_cases hx : n = d <;> by_cases hm : n ∈ rs <;> simp [hx, hm]

theorem bindP_ok {Γ : Env} {reg' : Nat → UInt64} :
    ∀ (rs : List Nat) (k : Nat) (r : Body.Regs Slot) (reg : Nat → UInt64), EnvI c r reg Γ →
    (∀ n ∈ rs, c.fl.contains n = true) → (∀ n ∈ rs, reg' n ≤ 1) →
    (∀ (i : Nat) (h : i < rs.length), Γ[k + i]? = some (.sc .i8 (reg' rs[i]))) →
    EnvI c (Body.bindP r rs (carriesFrom k _)) (fun n => if n ∈ rs then reg' n else reg n) Γ
  | [], k, r, reg, hE, _, _, _ => by simpa [Body.bindP] using hE
  | d :: rs, k, r, reg, hE, hf, hb, hs => by
      simp only [List.map_cons, carriesFrom, Body.bindP]
      have h1 := hE.setP d (hf d (by simp)) k (reg' d) (hb d (by simp)) (by simpa using hs 0 (by simp))
      have h2 := bindP_ok rs (k + 1) _ _ h1 (fun n hn => hf n (by simp [hn])) (fun n hn => hb n (by simp [hn]))
        (fun i hi => by simpa [Nat.add_assoc, Nat.add_comm 1 i] using hs (i + 1) (by simp; omega))
      exact h2.congr fun n => by by_cases hx : n = d <;> by_cases hm : n ∈ rs <;> simp [hx, hm]

theorem EnvI.bindF {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env} :
    ∀ (ws : List Nat) (vs : Vals Slot (ws.map fun _ => ClifTy.f32)), EnvI c r reg Γ → EnvI c (Body.bindF r ws vs) reg Γ
  | [], _, h => h
  | n :: ws, .cons v vs, h => EnvI.bindF (r := Body.set r n v) ws vs (EnvI.ofI (r := r) h)

theorem FEnv.bindV {r : Body.Regs Slot} {f : Nat → F32} {Γ : Env} :
    ∀ (rs : List Nat) (vs : Vals Slot (rs.map fun _ => ClifTy.i64)), FEnv c r f Γ → FEnv c (Body.bindV r rs vs) f Γ
  | [], _, h => h
  | n :: rs, .cons v vs, h => FEnv.bindV (r := Body.setV r n v) rs vs (FEnv.ofF (r := r) h)

theorem FEnv.bindP {r : Body.Regs Slot} {f : Nat → F32} {Γ : Env} :
    ∀ (rs : List Nat) (vs : Vals Slot (rs.map fun _ => ClifTy.i8)), FEnv c r f Γ → FEnv c (Body.bindP r rs vs) f Γ
  | [], _, h => h
  | n :: rs, .cons v vs, h => FEnv.bindP (r := Body.setP r n v) rs vs (FEnv.ofF (r := r) h)

theorem bindF_ok {Γ : Env} {f' : Nat → F32} :
    ∀ (ws : List Nat) (k : Nat) (r : Body.Regs Slot) (f : Nat → F32), FEnv c r f Γ →
    (∀ n ∈ ws, (f' n).bits.toNat < 2 ^ 32) →
    (∀ (i : Nat) (h : i < ws.length), Γ[k + i]? = some (.sc .f32 (f' ws[i]).bits)) →
    FEnv c (Body.bindF r ws (carriesFrom k _)) (fun n => if n ∈ ws then f' n else f n) Γ
  | [], k, r, f, hF, _, _ => by simpa [Body.bindF] using hF
  | d :: ws, k, r, f, hF, h32, hs => by
      simp only [List.map_cons, carriesFrom, Body.bindF]
      have h1 := hF.set d (v := k) (y := f' d) (by simpa using hs 0 (by simp)) (h32 d (by simp))
      have h2 := bindF_ok ws (k + 1) _ _ h1 (fun n hn => h32 n (by simp [hn]))
        (fun i hi => by simpa [Nat.add_assoc, Nat.add_comm 1 i] using hs (i + 1) (by simp; omega))
      have he : (fun n => if n ∈ ws then f' n else if n = d then f' d else f n) =
          (fun n => if n ∈ d :: ws then f' n else f n) := by
        funext n; by_cases hx : n = d <;> by_cases hm : n ∈ ws <;> simp [hx, hm]
      rw [← he]; exact h2

theorem FEnv.congr {r : Body.Regs Slot} {f f' : Nat → F32} {Γ : Env} (h : FEnv c r f Γ) (he : ∀ n, f n = f' n) :
    FEnv c r f' Γ := by
  have : f = f' := funext he
  subst this; exact h

/-- **The environment after a join or a loop** holds the registers the
    carries bring, read from slot `n` on, and the ones they do not as before. -/
theorem unpack_okG {r : Body.Regs Slot} {st st' : LI} {Γ Γ'' : Env} {n : Nat}
    {ws rv rf : List Nat} (hE : EnvI c r st.reg Γ) (hF : FEnv c r st.f Γ) (hx : Extends Γ Γ'')
    (hread : ∀ i v, (carried st' ws rv rf)[i]? = some v → Γ''[n + i]? = some v)
    (hv : ∀ m ∈ rv, c.fl.contains m = false) (hf : ∀ m ∈ rf, c.fl.contains m = true)
    (hb : ∀ m, c.fl.contains m = true → st'.reg m ≤ 1) (hfr : ∀ m, m ∉ rv → m ∉ rf → st'.reg m = st.reg m)
    (hff : ∀ m, m ∉ ws → st'.f m = st.f m) (h32 : ∀ m, (st'.f m).bits.toNat < 2 ^ 32) :
    EnvI c (Body.unpack r ws rv rf (carriesFrom n (Body.carryTys ws rv rf))) st'.reg Γ'' ∧
      FEnv c (Body.unpack r ws rv rf (carriesFrom n (Body.carryTys ws rv rf))) st'.f Γ'' := by
  simp only [Body.unpack, Body.carryTys]
  rw [splitV_carries, splitV_carries]
  have hF1 := bindF_ok (f' := st'.f) ws n r st.f (hF.extend hx) (fun m _ => h32 m) (fun i hi =>
    hread i _ (by simp [carried, List.getElem?_append_left, hi]))
  have hE1 := EnvI.bindF (r := r) ws (carriesFrom n (ws.map fun _ => ClifTy.f32)) (hE.extend hx)
  have h1 := bindV_ok (reg' := st'.reg) rv (n + (ws.map fun _ => ClifTy.f32).length) _ st.reg hE1 hv (fun i hi => by
    rw [Nat.add_assoc]
    exact hread _ _ (by simp [carried, List.getElem?_append_right, List.getElem?_append_left, hi]))
  have h2 := bindP_ok (reg' := st'.reg) rf (n + (ws.map fun _ => ClifTy.f32).length +
      (rv.map fun _ => ClifTy.i64).length) _ _ h1 hf
    (fun m hm => hb m (hf m hm)) (fun i hi => by
      rw [Nat.add_assoc, Nat.add_assoc]
      exact hread _ _ (by simp [carried, List.getElem?_append_right, hi]))
  refine ⟨h2.congr fun m => ?_, ?_⟩
  · by_cases h1 : m ∈ rf
    · simp [h1]
    · by_cases h2 : m ∈ rv
      · simp [h1, h2]
      · simp [h1, h2, hfr m h2 h1]
  · refine FEnv.bindP rf _ (FEnv.bindV rv _ (hF1.congr fun m => ?_))
    by_cases hm : m ∈ ws
    · simp [hm]
    · simp [hm, hff m hm]

/-- **The environment after a join or a loop**, the carries bound at `n`. -/
theorem unpack_ok {r : Body.Regs Slot} {st st' : LI} {Γ Γ' : Env} {n : Nat}
    {ws rv rf : List Nat} (hE : EnvI c r st.reg Γ) (hF : FEnv c r st.f Γ) (hx : Extends Γ Γ') (hn : Γ.size ≤ n)
    (hv : ∀ m ∈ rv, c.fl.contains m = false) (hf : ∀ m ∈ rf, c.fl.contains m = true)
    (hb : ∀ m, c.fl.contains m = true → st'.reg m ≤ 1) (hfr : ∀ m, m ∉ rv → m ∉ rf → st'.reg m = st.reg m)
    (hff : ∀ m, m ∉ ws → st'.f m = st.f m) (h32 : ∀ m, (st'.f m).bits.toNat < 2 ^ 32) :
    EnvI c (Body.unpack r ws rv rf (carriesFrom n (Body.carryTys ws rv rf))) st'.reg
        (bindAt Γ' n (carried st' ws rv rf)) ∧
      FEnv c (Body.unpack r ws rv rf (carriesFrom n (Body.carryTys ws rv rf))) st'.f
        (bindAt Γ' n (carried st' ws rv rf)) :=
  unpack_okG hE hF (hx.bindAt hn _) (fun i v h => by rw [bindAt_at]; exact h) hv hf hb hfr hff h32

theorem carried_inj {a b : LI} {ws rv rf : List Nat} (h : carried a ws rv rf = carried b ws rv rf) :
    (∀ r ∈ ws, a.f r = b.f r) ∧ (∀ r ∈ rv, a.reg r = b.reg r) ∧ ∀ r ∈ rf, a.reg r = b.reg r := by
  unfold carried at h
  obtain ⟨h0, h12⟩ := List.append_inj h (by simp)
  obtain ⟨h1, h2⟩ := List.append_inj h12 (by simp)
  refine ⟨fun r hr => ?_, fun r hr => by simpa using List.map_inj_left.mp h1 r hr,
    fun r hr => by simpa using List.map_inj_left.mp h2 r hr⟩
  have := List.map_inj_left.mp h0 r hr
  simp only [V.sc.injEq, true_and] at this
  cases ha : a.f r; cases hb : b.f r
  rw [ha, hb] at this; simp only at this; rw [this]

theorem carried_length (st : LI) (ws rv rf : List Nat) :
    (carried st ws rv rf).length = (Body.carryTys ws rv rf).length := by simp [carried, Body.carryTys]

/-- The carries a loop or a join binds, read back from where they are bound. -/
theorem carries_read {Γ : Env} {n : Nat} {xs : List V} : ∀ (tys : List ClifTy) (k : Nat),
    xs.length = k + tys.length →
    (carriesFrom (n + k) tys).slots.mapM (fun r => (bindAt Γ n xs)[r]?) = some (xs.drop k)
  | [], k, h => by simp at h; simp [carriesFrom, Vals.slots, List.drop_eq_nil_of_le (show xs.length ≤ k by omega)]
  | _ :: ts, k, h => by
      have hk : k < xs.length := by simp at h; omega
      have ih := carries_read (Γ := Γ) (n := n) (xs := xs) ts (k + 1) (by simp at h; omega)
      rw [show n + (k + 1) = n + k + 1 by omega] at ih
      simp only [carriesFrom, Vals.slots, List.mapM_cons, bindAt_at, ih, List.getElem?_eq_getElem hk]
      rw [List.drop_eq_getElem_cons hk]; rfl

theorem lookup_restore {α : Type} (rs : List Nat) (l0 l : List (Nat × α)) (n : Nat) :
    (l.filter (·.1 ∉ rs) ++ l0.filter (·.1 ∈ rs)).lookup n = if n ∈ rs then l0.lookup n else l.lookup n := by
  induction l with
  | nil =>
      simp only [List.filter_nil, List.nil_append]
      induction l0 with
      | nil => split <;> rfl
      | cons p l0 ih =>
          obtain ⟨d, v⟩ := p
          by_cases hd : d ∈ rs
          · simp only [List.filter, hd, decide_true]
            by_cases hn : n = d
            · subst hn; simp [hd]
            · rw [lookup_cons_ne hn, ih, lookup_cons_ne hn]
          · simp only [List.filter, hd, decide_false]
            rw [ih]
            by_cases hn : n = d
            · subst hn; simp [hd]
            · rw [lookup_cons_ne hn]
  | cons p l ih =>
      obtain ⟨d, v⟩ := p
      rw [List.filter_cons]
      by_cases hd : d ∈ rs
      · simp only [hd, not_true_eq_false, decide_false, Bool.false_eq_true, if_false]
        rw [ih]
        by_cases hn : n = d
        · subst hn; simp [hd]
        · rw [lookup_cons_ne hn]
      · simp only [hd, not_false_eq_true, decide_true, if_true, List.cons_append]
        by_cases hn : n = d
        · subst hn; simp [hd]
        · rw [lookup_cons_ne hn, ih, lookup_cons_ne hn]

/-- **After local registers**, the environment holds them as before them and
    the others as after. -/
theorem EnvI.restore {r0 r : Body.Regs Slot} {reg0 reg : Nat → UInt64} {Γ0 Γ : Env} (rs : List Nat)
    (h0 : EnvI c r0 reg0 Γ0) (h : EnvI c r reg Γ) (hx : Extends Γ0 Γ) :
    EnvI c (Body.restoreR rs r0 r) (fun n => if n ∈ rs then reg0 n else reg n) Γ where
  val n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hn : n ∈ rs <;> simp only [hn, if_true, if_false] at hu ⊢
    · exact hx.get (h0.val n u hu)
    · exact h.val n u hu
  flag n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hn : n ∈ rs <;> simp only [hn, if_true, if_false] at hu ⊢
    · exact hx.get (h0.flag n u hu)
    · exact h.flag n u hu
  word n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hn : n ∈ rs <;> simp only [hn, if_true, if_false] at hu ⊢
    · exact ⟨hx.get (h0.word n u hu).1, (h0.word n u hu).2⟩
    · exact h.word n u hu
  bit n hn := by by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] <;> first | exact h0.bit n hn | exact h.bit n hn
  flag0 n hn hl := by
    simp only [Body.restoreR, lookup_restore] at hl
    by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] at hl ⊢
    · exact h0.flag0 n hn hl
    · exact h.flag0 n hn hl
  val0 n hn hl := by
    simp only [Body.restoreR, lookup_restore] at hl
    by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] at hl ⊢
    · exact h0.val0 n hn hl
    · exact h.val0 n hn hl
  fkind n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] at hu
    · exact h0.fkind n u hu
    · exact h.fkind n u hu
  vkind n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] at hu
    · exact h0.vkind n u hu
    · exact h.vkind n u hu
  wkind n u hu := by
    simp only [Body.restoreR, lookup_restore] at hu
    by_cases hm : n ∈ rs <;> simp only [hm, if_true, if_false] at hu
    · exact h0.wkind n u hu
    · exact h.wkind n u hu

open Classical in
/-- The least `m ≤ n` that `P` holds of, or `n`. -/
noncomputable def least (P : Nat → Prop) : Nat → Nat
  | 0 => 0
  | n + 1 => if P (least P n) then least P n else n + 1

theorem least_inv (P : Nat → Prop) : ∀ n, (∀ m, m < least P n → ¬P m) ∧ (P (least P n) ∨ least P n = n)
  | 0 => ⟨fun m h => absurd h (Nat.not_lt_zero _), .inr rfl⟩
  | n + 1 => by
      obtain ⟨ih1, ih2⟩ := least_inv P n
      by_cases hp : P (least P n)
      · simp only [least, if_pos hp]; exact ⟨ih1, .inl hp⟩
      · simp only [least, if_neg hp]
        have hn : least P n = n := ih2.resolve_left hp
        refine ⟨fun m hm hpm => ?_, .inr (by simp)⟩
        rcases Nat.lt_or_ge m n with h | h
        · exact ih1 m (by omega) hpm
        · have : m = n := by omega
          subst this; rw [hn] at hp; exact hp hpm

/-- **The least of what `P` holds of**, below a bound one of them is. -/
theorem least_spec {P : Nat → Prop} {k n : Nat} (hk : P k) (hn : k ≤ n) :
    P (least P n) ∧ ∀ m, P m → least P n ≤ m := by
  obtain ⟨h1, h2⟩ := least_inv P n
  have hmin : ∀ m, P m → least P n ≤ m := fun m hm => Nat.not_lt.mp fun h => h1 m h hm
  refine ⟨?_, hmin⟩
  rcases h2 with h | h
  · exact h
  · have := hmin k hk; rw [h] at this ⊢; rw [show n = k by omega]; exact hk

/-! ## Statements -/

theorem wpE_uload8 {Γ : Env} {w : World} {a : Slot .i64} {X v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 X)) (hl : w.mem.load X 1 = some v)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 v) → Q s Γ' w) : wpE cfg (uload8_64 a) Q Γ w :=
  wpE_val (by simp only [Op'.erase, evalOp, Sem.get, ha]; simp [hl, norm, and_wide64]) (xget h)

theorem wpE_uload16 {Γ : Env} {w : World} {a : Slot .i64} {X v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 X)) (hl : w.mem.load X 2 = some v)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 v) → Q s Γ' w) : wpE cfg (uload16_64 a) Q Γ w :=
  wpE_val (by simp only [Op'.erase, evalOp, Sem.get, ha]; simp [hl, norm, and_wide64]) (xget h)

/-- **A byte store ends** where the store reaches memory. -/
theorem wpE_store8 {Γ : Env} {w w' : World} {vS aS : Slot .i64} {X A : UInt64} {Q : Unit → Env → World → Prop}
    (hv : Γ[vS]? = some (.sc .i64 X)) (ha : Γ[aS]? = some (.sc .i64 A))
    (hs : stw w A 1 (X &&& 0xff) = some w') (h : Q () Γ w') : wpE cfg (istore8 vS aS) Q Γ w := by
  simp only [stw] at hs
  obtain ⟨m, hm, hs⟩ := Option.bind_eq_some_iff.mp hs
  obtain rfl := (Option.some.inj hs).symm
  exact wpE_istore8 (by simp [runStmt, Sem.get, hv, ha, hm]) (wpE_ret h)

/-- **A 16-bit shared store ends** where the store reaches memory. -/
theorem wpE_store16 {Γ : Env} {w w' : World} {xS smS aS : Slot .i64} {X S A : UInt64}
    {Q : Unit → Env → World → Prop}
    (hx : Γ[xS]? = some (.sc .i64 X)) (hsm : Γ[smS]? = some (.sc .i64 S)) (ha : Γ[aS]? = some (.sc .i64 A))
    (hs : stw w (S + A) 2 (X &&& 0xffff) = some w') (h : ∀ Γ', Extends Γ Γ' → Q () Γ' w') :
    wpE cfg (do istore16 xS (← iadd smS aS)) Q Γ w := by
  simp only [stw] at hs
  obtain ⟨m, hm, hs⟩ := Option.bind_eq_some_iff.mp hs
  obtain rfl := (Option.some.inj hs).symm
  simp only [istore16]
  refine wpE_bind (wpE_iadd hsm ha ?_)
  have hx2 := (Extends.push Γ (.sc .i64 (S + A))).get hx
  refine wpE_bind (wpE_val (v := .sc .i16 (X &&& 0xffff)) (by
    simp only [Op'.erase, evalOp, Sem.get, hx2]; simp [norm, IExt.admits, ClifTy.isInt, ClifTy.width]; rfl) ?_)
  have hp := (Extends.push _ (.sc .i16 (X &&& 0xffff))).get (get_push_size (Γ := Γ) (v := .sc .i64 (S + A)))
  have hq := get_push_size (Γ := Γ.push (.sc .i64 (S + A))) (v := .sc .i16 (X &&& 0xffff))
  refine wpE_store (w' := { obsStore w (S + A) 2 (X &&& 0xffff) with mem := m })
    (by simp only [runStmt, Sem.get, hp, hq]; simp [tyBytes, ClifTy.width, hm]) (wpE_ret ?_)
  exact h _ (((Extends.push _ _).push_of _))

/-- **A guarded action** runs where the guard is not zero, and is skipped
    where it is. -/
theorem wpE_guarded {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ Γ0 : Env} {w : World}
    (hE : EnvI c r reg Γ) (hS : Slots c Γ x) (hx0 : Extends Γ Γ0) (g : IExp) (act : Prog Slot Lvl Unit)
    {Q : Unit → Env → World → Prop}
    (hact : g.evalL x reg ≠ 0 → ∀ Γ1, Extends Γ0 Γ1 →
      wpE cfg act (fun _ Γ' w' => ∀ Γ'', Extends Γ' Γ'' → Q () Γ'' w') Γ1 w)
    (hskip : g.evalL x reg = 0 → ∀ Γ1, Extends Γ0 Γ1 → Q () Γ1 w) :
    wpE cfg (Body.guarded c r g act) Q Γ0 w := by
  unfold Body.guarded
  split
  · exact wpE_mono (fun _ Γ' w' h => h Γ' (Extends.refl _)) (hact (by simp [IExp.evalL]) Γ0 (Extends.refl _))
  · refine wpE_bind (wpE_condOf hE hS hx0 g fun t k a b X Y Γ1 hx1 hti ha hb hk => ?_)
    simp only [when]
    refine wpE_bind (wpE_ifteT (fun _ Γ' w' => ∀ Γ'', Extends Γ' Γ'' → Q () Γ'' w') hti ha hb ?_ ?_ ?_ ?_)
    · intro hc
      rw [hc] at hk
      refine wpE_bind (wpE_mono (fun _ Γ' w' hq => wpE_pure hq) (hact ?_ _ (hx1.push_of _)))
      simpa using hk.symm
    · intro hc ne hne
      rw [hc] at hk
      exact wpE_pure fun Γ'' hx'' => hskip (by simpa using hk.symm) _
        (((hx1.push_of _).bindAt_of hne []).trans hx'')
    · exact fun jv Γ' w' _ => ⟨[], by cases jv; rfl⟩
    · intro nJ Γ' jv vs w' hle _ hq
      exact wpE_pure (hq _ ((Extends.refl _).bindAt_of hle _))

/-- Every register `isBool`-assigned where it is a predicate register. -/
def Wf (c : Body.Ctx Slot) (s : EWStmt) : Prop :=
  ∀ p ∈ Body.assigns c.fl s, c.fl.contains p.1 = true → p.2 = true

variable (cfg c x) in
/-- **What a statement's code does**: from a lane state the environment holds,
    where the lane's run of the
    statement ends, the code ends with an environment holding the lane's
    registers after. -/
def CodeOk (s : EWStmt) : Prop :=
  ∀ (r : Body.Regs Slot) (Γ : Env) (st st' : LI) (Q : Body.Regs Slot → Env → World → Prop),
    Wf c s → EnvI c r st.reg Γ → FEnv c r st.f Γ → Slots c Γ x → laneI x s st = some st' →
    (∀ r' Γ' , Extends Γ Γ' → EnvI c r' st'.reg Γ' → FEnv c r' st'.f Γ' → Q r' Γ' st'.w) →
    wpE cfg (Body.stmt c r s) Q Γ st.w

theorem s32_lt32 (u : UInt32) : s32 u < 2 ^ 32 := by
  apply UInt64.lt_iff_toNat_lt.mpr
  simp only [s32, UInt32.toNat_toUInt64]
  exact u.toNat_lt

theorem ansW_lt (ans : List Answer) (k : Nat) : ansW ans k < 2 ^ 32 := s32_lt32 _

theorem assign_flag {s : EWStmt} {d : Nat} (hwf : Wf c s) (hm : (d, false) ∈ Body.assigns c.fl s) :
    c.fl.contains d = false := by
  cases h : c.fl.contains d
  · rfl
  · exact absurd (hwf _ hm h) (by simp)

/-! ### Floats and indexes -/

theorem exLog_mem (hw : Hw) (f : Nat → F32) : ∀ (e : WFExp) (w : World), (exLog hw f e w).mem = w.mem := by
  intro e
  induction e with
  | reg _ | lit _ => intro w; rfl
  | add a b iha ihb | mul a b iha ihb | maxW a b iha ihb | geF a b iha ihb =>
      intro w; simp only [exLog]; rw [ihb, iha]
  | neg a ih | inv a ih | rsqrt a ih => intro w; exact ih w
  | ex2 a ih | exp a ih => intro w; simp only [exLog, obsCall]; exact ih w

theorem fl_nan {hw : Hw} {w : World} (hn : w.mem.nan = hw.nan) : w.mem.fl = nanFix hw.nan := by
  show nanFix w.mem.nan = _; rw [hn]

/-- An expression's value is a 32-bit word where the registers are and the
    hardware's approximations give 32 bits. -/
theorem evalF_32 (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) {f : Nat → F32}
    (h : ∀ n, (f n).bits.toNat < 2 ^ 32) : ∀ e : WFExp, (evalF hw f e).bits.toNat < 2 ^ 32 := by
  intro e
  induction e with
  | reg n => exact h n
  | lit v => exact ofF32_lt v
  | add a b _ _ => exact nanFix_lt _ (ofF32_lt _)
  | mul a b _ _ => exact nanFix_lt _ (ofF32_lt _)
  | neg a ih =>
    show ((evalF hw f a).bits ^^^ 0x80000000).toNat < 2 ^ 32
    rw [UInt64.toNat_xor]
    exact Nat.xor_lt_two_pow ih (by decide)
  | inv a _ => exact nanFix_lt _ (ofF32_lt _)
  | exp a _ => exact hw32 _ _
  | ex2 a _ => exact hw32 _ _
  | rsqrt a _ => exact nanFix_lt _ (ofF32_lt _)
  | maxW a b iha ihb =>
    show (@NumOps.max F32 (F32.ops hw) (evalF hw f a) (evalF hw f b)).bits.toNat < 2 ^ 32
    simp only [NumOps.max, NumOps.ifGe, F32.ops]
    split <;> first | assumption | (split <;> assumption)
  | geF a b iha ihb =>
    show (@NumOps.ifGe F32 (F32.ops hw) (evalF hw f a) (evalF hw f b) (NumLit.lit 1.0) (NumLit.lit 0.0)).bits.toNat < 2 ^ 32
    simp only [NumOps.ifGe, F32.ops]
    split <;> first | exact ofF32_lt _ | (split <;> exact ofF32_lt _)

theorem fmax_bits (hw : Hw) (A B : F32) : (@NumOps.max F32 (F32.ops hw) A B).bits =
    if fcmpBits .ge .f32 A.bits B.bits then A.bits else B.bits := by
  simp only [NumOps.max, NumOps.ifGe, F32.ops, fcmpBits, Bool.or_comm]
  split <;> simp_all <;> split <;> rfl

theorem fifGe_bits (hw : Hw) (A B X Y : F32) : (@NumOps.ifGe F32 (F32.ops hw) A B X Y).bits =
    if fcmpBits .ge .f32 A.bits B.bits then X.bits else Y.bits := by
  simp only [NumOps.ifGe, F32.ops, fcmpBits, Bool.or_comm]
  split <;> simp_all <;> split <;> rfl

/-- The hardware's `2^x`, as a thread's locals answer it. -/
theorem wpE_ex2L {hw : Hw} {Γ : Env} {w : World} {a : Slot .f32} {v : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (hex : ∀ v w, cfg.locals Intr.ex2Approx.slot [.sc .f32 v] w = .ok (some (.sc .f32 (hw.approx .ex2Approx v)), w))
    (ha : Γ[a]? = some (.sc .f32 v))
    (h : Q Γ.size (Γ.push (.sc .f32 (hw.approx .ex2Approx v))) (obsCall w (.intr .ex2Approx) [.sc .f32 v])) :
    wpE cfg (callLocal Kernel.Block.ex2Approx %[a] : Prog Slot Lvl (Slot .f32)) Q Γ w :=
  wpE_callLocal (vs := [.sc .f32 v]) (v := .sc .f32 (hw.approx .ex2Approx v))
    (w' := obsCall w (.intr .ex2Approx) [.sc .f32 v]) rfl (by simp [Vals.slots, ha])
    (by simp [callOf, Kernel.Block.ex2Approx, hex, Except.toOption]) (wpE_ret h)

/-- **A float expression's code computes its value at `F32`**: its code ends
    with that value's word, in the world its `2^x` calls make. -/
theorem wpE_fexpL (hw : Hw)
    (hex : ∀ v w, cfg.locals Intr.ex2Approx.slot [.sc .f32 v] w = .ok (some (.sc .f32 (hw.approx .ex2Approx v)), w))
    (r : Body.Regs Slot) (f : Nat → F32) :
    ∀ (e : WFExp) (Γ : Env) (w : World) (Q : Slot .f32 → Env → World → Prop),
      w.mem.nan = hw.nan → (∀ n, Γ[Body.get c r n]? = some (.sc .f32 (f n).bits)) →
      (∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .f32 (evalF hw f e).bits) → Q v Γ' (exLog hw f e w)) →
      wpE cfg (Body.fexp c r e : Prog Slot Lvl (Slot .f32)) Q Γ w := by
  intro e
  induction e with
  | reg n =>
      intro Γ w Q hn hr hQ
      rw [Body.fexp]
      exact wpE_pure (hQ _ _ (Extends.refl _) (hr n))
  | lit v =>
      intro Γ w Q hn _ hQ
      rw [Body.fexp]
      apply wpE_fconst32
      rw [and_f32_lt _ (ofF32_lt v)]
      exact hQ _ _ (Extends.push _ _) (get_push_size)
  | add a b iha ihb =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (ihb Γa _ _ hna (fun n => hEa.get (hr n)) fun rb Γb hEb hrb => ?_)
      refine wpE_fadd32 (hEb.get hra) hrb ?_
      rw [fl_nan (by rw [exLog_mem]; exact hna)]
      exact hQ _ _ ((hEa.trans hEb).push_of _) (get_push_size)
  | mul a b iha ihb =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (ihb Γa _ _ hna (fun n => hEa.get (hr n)) fun rb Γb hEb hrb => ?_)
      refine wpE_fmul32 (hEb.get hra) hrb ?_
      rw [fl_nan (by rw [exLog_mem]; exact hna)]
      exact hQ _ _ ((hEa.trans hEb).push_of _) (get_push_size)
  | neg a iha =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      exact wpE_fneg32 hra (hQ _ _ (hEa.push_of _) (get_push_size))
  | inv a iha =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (wpE_fconst32 ?_)
      rw [and_f32_lt _ (by decide : oneBits.toNat < 2 ^ 32)]
      refine wpE_fdiv32 (get_push_size) (get_push hra) ?_
      rw [fl_nan hna]
      exact hQ _ _ ((hEa.push_of _).push_of _) (get_push_size)
  | exp a iha =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (wpE_fconst32 ?_)
      rw [show log2eBits &&& widthMask .f32 = log2eBits from and_mask32 (ofF32_lt _)]
      refine wpE_bind (wpE_fmul32 (get_push hra) (get_push_size) ?_)
      rw [fl_nan hna]
      exact wpE_ex2L hex (get_push_size)
        (hQ _ _ (((hEa.push_of _).push_of _).push_of _) (get_push_size))
  | ex2 a iha =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      exact wpE_ex2L hex hra (hQ _ _ (hEa.push_of _) (get_push_size))
  | rsqrt a iha =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (wpE_sqrt32 hra ?_)
      rw [fl_nan hna]
      refine wpE_bind (wpE_fconst32 ?_)
      rw [and_f32_lt _ (by decide : oneBits.toNat < 2 ^ 32)]
      refine wpE_fdiv32 (get_push_size) (get_push (get_push_size)) ?_
      rw [fl_nan hna]
      exact hQ _ _ (((hEa.push_of _).push_of _).push_of _) (get_push_size)
  | maxW a b iha ihb =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (ihb Γa _ _ hna (fun n => hEa.get (hr n)) fun rb Γb hEb hrb => ?_)
      refine wpE_bind (wpE_fcmp32 (hEb.get hra) hrb ?_)
      refine wpE_selectF32 (get_push_size) (get_push (hEb.get hra)) (get_push hrb) ?_
      exact hQ _ _ (((hEa.trans hEb).push_of _).push_of _)
        (by show _ = some (V.sc ClifTy.f32 (@NumOps.max F32 (F32.ops hw) _ _).bits); rw [fmax_bits]
            exact get_push_size)
  | geF a b iha ihb =>
      intro Γ w Q hn hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ hn hr fun ra Γa hEa hra => ?_)
      have hna : (exLog hw f a w).mem.nan = hw.nan := by rw [exLog_mem]; exact hn
      refine wpE_bind (ihb Γa _ _ hna (fun n => hEa.get (hr n)) fun rb Γb hEb hrb => ?_)
      refine wpE_bind (wpE_fcmp32 (hEb.get hra) hrb ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      rw [and_f32_lt _ (ofF32_lt _), and_f32_lt _ (ofF32_lt _)]
      refine wpE_selectF32 (get_push (get_push (get_push_size))) (get_push (get_push_size))
        (get_push_size) ?_
      exact hQ _ _ ((((hEa.trans hEb).push_of _).push_of _).push_of _ |>.push_of _)
        (by show _ = some (V.sc ClifTy.f32 (@NumOps.ifGe F32 (F32.ops hw) _ _ _ _).bits); rw [fifGe_bits]
            exact get_push_size)

theorem s32_toUInt32 {v : UInt64} (h : v < 2 ^ 32) : s32 v.toUInt32 = v := by
  apply UInt64.toNat_inj.mp
  have h' : v.toNat < 2 ^ 32 := lt_of_lt_of_eq (UInt64.lt_iff_toNat_lt.mp h) rfl
  simp only [s32, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
  omega

theorem wp_at4L {d : Nat} {J : Post} {base : Slot .i64} {Γ : Env} {w : World} {B : UInt64} {ixs : Slot .i32}
    {j : UInt32} {Q : Slot .i64 → Env → World → Prop}
    (hb : Γ[base]? = some (.sc .i64 B)) (hj : Γ[ixs]? = some (.sc .i32 (s32 j)))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (B + off j)) → Q v Γ' w) :
    wp cfg d J (Body.at4 base ixs : Prog Slot Lvl (Slot .i64)) Q Γ w := by
  simp only [Body.at4, wp_bind]
  refine wp_uext32 hj ?_
  refine wp_ishlImm get_push_size ?_
  refine wp_iadd (get_push (get_push (get_push hb))) get_push_size ?_
  exact h _ _ ((((Extends.refl _).push_of _).push_of _).push_of _ |>.push_of _) get_push_size

theorem wp_loadI32L {d : Nat} {J : Post} {a : Slot .i64} {p v : UInt64} {Γ : Env} {w : World}
    {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 4 = some v)
    (h : Q Γ.size (Γ.push (.sc .i32 v)) w) : wp cfg d J (load32 a) Q Γ w :=
  wp_val (by simp [load32, Op'.erase, evalOp, Sem.get, ha, hl, tyBytes, ClifTy.lanes, ClifTy.width] <;> op_fin) h

theorem at4_opsL (base : Slot .i64) (ix : Slot .i32) : OpsOnly (Body.at4 base ix : Prog Slot Lvl (Slot .i64)) := by
  unfold Body.at4; ops_only

theorem idx_opsL (c : Body.Ctx Slot) : ∀ ix : IdxE, OpsOnly (Body.idx c ix : Prog Slot Lvl (Slot .i32))
  | .laneId => by unfold Body.idx; exact .ret _
  | .loopI => by unfold Body.idx; exact .ret _
  | .ctaId => by unfold Body.idx; exact .ret _
  | .ireg _ => by unfold Body.idx; exact .ret _
  | .lit n => by unfold Body.idx Kernel.Block.c32; ops_only
  | .add a b => by
      unfold Body.idx; exact OpsOnly.bind (idx_opsL c a) fun _ => OpsOnly.bind (idx_opsL c b) fun _ => by ops_only
  | .mul a b => by
      unfold Body.idx; exact OpsOnly.bind (idx_opsL c a) fun _ => OpsOnly.bind (idx_opsL c b) fun _ => by ops_only
  | .ldIdx b o => by
      unfold Body.idx
      exact OpsOnly.bind (idx_opsL c o) fun _ => OpsOnly.bind (at4_opsL _ _) fun _ => by unfold load32; ops_only

/-- **An index's code computes its lane value.** -/
theorem wp_idxV {d : Nat} {J : Post} :
    ∀ (ix : IdxE) (Γ : Env) (w : World) (Q : Slot .i32 → Env → World → Prop) (j : UInt32),
      Slots c Γ x → idxV x w.mem ix = some j →
      (∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i32 (s32 j)) → Q v Γ' w) →
      wp cfg d J (Body.idx c ix : Prog Slot Lvl (Slot .i32)) Q Γ w := by
  intro ix
  induction ix with
  | laneId =>
    intro Γ w Q j hS hj hQ
    simp only [idxV, Option.some.injEq] at hj; subst hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) (by rw [s32_toUInt32 hS.lane32]; exact hS.lane)
  | loopI =>
    intro Γ w Q j hS hj hQ
    simp only [idxV, Option.some.injEq] at hj; subst hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) hS.i
  | ctaId =>
    intro Γ w Q j hS hj hQ
    simp only [idxV, Option.some.injEq] at hj; subst hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) (by rw [s32_toUInt32 hS.cta32]; exact hS.cta)
  | lit n =>
    intro Γ w Q j hS hj hQ
    simp only [idxV] at hj
    split at hj
    · rename_i hn
      obtain rfl := Option.some.inj hj
      rw [Body.idx, Kernel.Block.c32]
      refine wp_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_
      exact hQ _ _ (Extends.push _ _) (by simpa using get_push_size)
    · cases hj
  | ireg n =>
    intro Γ w Q j hS hj hQ
    simp only [idxV, Option.some.injEq] at hj; subst hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) hS.zi
  | add a b iha ihb =>
    intro Γ w Q j hS hj hQ
    simp only [idxV] at hj
    obtain ⟨ja, hja, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain ⟨jb, hjb, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain rfl := Option.some.inj hj
    simp only [Body.idx, wp_bind]
    refine iha Γ w _ ja hS hja fun va Γa hEa hva => ?_
    refine ihb Γa w _ jb (hS.extend hEa) hjb fun vb Γb hEb hvb => ?_
    refine wp_iadd32 (hEb.get hva) hvb ?_
    exact hQ _ _ ((hEa.trans hEb).push_of _) get_push_size
  | mul a b iha ihb =>
    intro Γ w Q j hS hj hQ
    simp only [idxV] at hj
    obtain ⟨ja, hja, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain ⟨jb, hjb, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain rfl := Option.some.inj hj
    simp only [Body.idx, wp_bind]
    refine iha Γ w _ ja hS hja fun va Γa hEa hva => ?_
    refine ihb Γa w _ jb (hS.extend hEa) hjb fun vb Γb hEb hvb => ?_
    refine wp_imul32 (hEb.get hva) hvb ?_
    exact hQ _ _ ((hEa.trans hEb).push_of _) get_push_size
  | ldIdx b o ih =>
    intro Γ w Q j hS hj hQ
    simp only [idxV] at hj
    obtain ⟨jo, hjo, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain ⟨v, hv, hj⟩ := Option.bind_eq_some_iff.mp hj
    obtain rfl := Option.some.inj hj
    simp only [Body.idx, wp_bind]
    refine ih Γ w _ jo hS hjo fun vo Γo hEo hvo => ?_
    refine wp_at4L ((hS.extend hEo).buf b) hvo fun va Γa hEa hva => ?_
    refine wp_loadI32L hva hv ?_
    have hlt : v < 2 ^ 32 := UInt64.lt_iff_toNat_lt.mpr (by simpa using load4_lt hv)
    exact hQ _ _ ((hEo.trans hEa).push_of _) (by rw [s32_toUInt32 hlt]; exact get_push_size)

theorem wpE_idxV {Γ : Env} {w : World} {Q : Slot .i32 → Env → World → Prop} {j : UInt32} (ix : IdxE)
    (hS : Slots c Γ x) (hj : idxV x w.mem ix = some j)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i32 (s32 j)) → Q v Γ' w) :
    wpE cfg (Body.idx c ix : Prog Slot Lvl (Slot .i32)) Q Γ w :=
  wpE_of_wp (idx_opsL c ix) (wp_idxV (d := 0) (J := J0) ix Γ w Q j hS hj h)

theorem wpE_at4L {base : Slot .i64} {Γ : Env} {w : World} {B : UInt64} {ixs : Slot .i32} {j : UInt32}
    {Q : Slot .i64 → Env → World → Prop}
    (hb : Γ[base]? = some (.sc .i64 B)) (hj : Γ[ixs]? = some (.sc .i32 (s32 j)))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (B + off j)) → Q v Γ' w) :
    wpE cfg (Body.at4 base ixs : Prog Slot Lvl (Slot .i64)) Q Γ w :=
  wpE_of_wp (at4_opsL _ _) (wp_at4L (d := 0) (J := J0) hb hj h)

theorem wpE_loadI32L {a : Slot .i64} {p v : UInt64} {Γ : Env} {w : World} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 4 = some v)
    (h : Q Γ.size (Γ.push (.sc .i32 v)) w) : wpE cfg (load32 a) Q Γ w :=
  wpE_of_wp (by unfold load32; ops_only) (wp_loadI32L (d := 0) (J := J0) ha hl h)

section Leaves

theorem code_skip : CodeOk cfg c x .skip := by
  intro r Γ st st' Q _ hE hF _ hr hQ
  simp only [laneI, Option.some.injEq] at hr; subst hr
  exact wpE_pure (hQ r Γ (Extends.refl _) hE hF)

theorem code_setI (d : Nat) (e : IExp) : CodeOk cfg c x (.setI d e) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI, Option.some.injEq] at hr; subst hr
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  by_cases hfl : c.fl.contains d = true
  · rw [if_pos hfl]
    have hb := isBool_bit (x := x) hE e (hwf (d, e.isBool c.fl) (by simp [Body.assigns]) hfl)
    refine wpE_bind (wpE_ipred hE hS e (Extends.refl _) fun p Γ1 hx1 hp => ?_)
    exact wpE_pure (hQ _ _ hx1 ((hE.extend hx1).setP d hfl p _ hb (by rw [hp, boolV_ne0 hb])) (FEnv.ofF (hF.extend hx1)))
  · rw [if_neg hfl]
    have hfl' : c.fl.contains d = false := by simpa using hfl
    have hgen : wpE cfg (do pure (Body.setV r d (← Body.iexp c r e))) Q Γ st.w := by
      refine wpE_bind ((wpE_exp hE hS e).1 (Extends.refl _) fun v Γ1 hx1 hv => ?_)
      exact wpE_pure (hQ _ _ hx1 ((hE.extend hx1).setV d hfl' v _ hv) (FEnv.ofF (hF.extend hx1)))
    cases e with
    | ctz a =>
      refine wpE_bind ((wpE_exp hE hS a).2 (Extends.refl _) fun va Γ1 hx1 hva => ?_)
      refine wpE_bind (wpE_ctz32 hva fun v32 Γ2 hx2 hv32 => ?_)
      refine wpE_bind (wpE_uext32 hv32 fun v Γ3 hx3 hv => ?_)
      refine wpE_pure (hQ _ _ (hx1.trans (hx2.trans hx3)) ?_ (FEnv.ofF (hF.extend (hx1.trans (hx2.trans hx3)))))
      have hlt := ctz32_lt (a.evalL x st.reg &&& 0xffffffff)
      exact (hE.extend (hx1.trans (hx2.trans hx3))).setW d hfl' v v32 _ hlt
        (by rw [hv, m32_of_lt hlt]; rfl) (hx3.get hv32)
    | reg _ | lit _ | lane | cta | ptr _ | bin _ _ _ | cmp _ _ _ => exact hgen

theorem code_ldB (d : Nat) (a : IExp) : CodeOk cfg c x (.ldB d a) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI] at hr
  obtain ⟨v, hl, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind ((wpE_exp hE hS a).1 (Extends.refl _) fun va Γ1 hx1 hva => ?_)
  refine wpE_bind (wpE_uload8 hva hl fun vs Γ2 hx2 hvs => ?_)
  exact wpE_pure (hQ _ _ (hx1.trans hx2)
    ((hE.extend (hx1.trans hx2)).setV d (assign_flag hwf (by simp [Body.assigns])) vs v hvs)
    (FEnv.ofF (hF.extend (hx1.trans hx2))))

theorem code_ldH (d : Nat) (a : IExp) : CodeOk cfg c x (.ldH d a) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI] at hr
  obtain ⟨v, hl, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind ((wpE_exp hE hS a).1 (Extends.refl _) fun va Γ1 hx1 hva => ?_)
  refine wpE_bind (wpE_iadd ((hS.extend hx1).sm) hva ?_)
  refine wpE_bind (wpE_uload16 (get_push_size) hl fun vs Γ2 hx2 hvs => ?_)
  have hx := hx1.trans ((Extends.push _ _).trans hx2)
  exact wpE_pure (hQ _ _ hx
    ((hE.extend hx).setV d (assign_flag hwf (by simp [Body.assigns])) vs v hvs) (FEnv.ofF (hF.extend hx)))

theorem code_stB (a v g : IExp) : CodeOk cfg c x (.stB a v g) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_guarded hE hS (Extends.refl _) g _ (fun hg Γ1 hx1 => ?_) (fun hg Γ1 hx1 => ?_))
  · simp only [laneI, if_pos hg] at hr
    obtain ⟨w1, hw1, hr⟩ := Option.bind_eq_some_iff.mp hr
    obtain rfl := (Option.some.inj hr).symm
    refine wpE_bind ((wpE_exp hE hS v).1 hx1 fun vv Γ2 hx2 hvv => ?_)
    refine wpE_bind ((wpE_exp hE hS a).1 (hx1.trans hx2) fun va Γ3 hx3 hva => ?_)
    exact wpE_store8 (hx3.get hvv) hva hw1 fun Γ'' hx'' =>
      wpE_pure (hQ r Γ'' (hx1.trans (hx2.trans (hx3.trans hx''))) (hE.extend (hx1.trans (hx2.trans (hx3.trans hx''))))
        (hF.extend (hx1.trans (hx2.trans (hx3.trans hx'')))))
  · simp only [laneI, hg, ne_eq, not_true_eq_false, if_false, Option.some.injEq] at hr
    subst hr
    exact wpE_pure (hQ r Γ1 hx1 (hE.extend hx1) (hF.extend hx1))

theorem code_stH (a v g : IExp) : CodeOk cfg c x (.stH a v g) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_guarded hE hS (Extends.refl _) g _ (fun hg Γ1 hx1 => ?_) (fun hg Γ1 hx1 => ?_))
  · simp only [laneI, if_pos hg] at hr
    obtain ⟨w1, hw1, hr⟩ := Option.bind_eq_some_iff.mp hr
    obtain rfl := (Option.some.inj hr).symm
    refine wpE_bind ((wpE_exp hE hS v).1 hx1 fun vv Γ2 hx2 hvv => ?_)
    refine wpE_bind ((wpE_exp hE hS a).1 (hx1.trans hx2) fun va Γ3 hx3 hva => ?_)
    exact wpE_store16 (hx3.get hvv) ((hS.extend (hx1.trans (hx2.trans hx3))).sm) hva hw1 fun Γ'' hx'' Γ4 hx4 =>
      wpE_pure (hQ r Γ4 (hx1.trans (hx2.trans (hx3.trans (hx''.trans hx4))))
        (hE.extend (hx1.trans (hx2.trans (hx3.trans (hx''.trans hx4)))))
        (hF.extend (hx1.trans (hx2.trans (hx3.trans (hx''.trans hx4))))))
  · simp only [laneI, hg, ne_eq, not_true_eq_false, if_false, Option.some.injEq] at hr
    subst hr
    exact wpE_pure (hQ r Γ1 hx1 (hE.extend hx1) (hF.extend hx1))

theorem code_ballot (hA : AnswersFrom x.ans x.start cfg) (hW : Words x.ans) (d : Nat) (p : IExp) :
    CodeOk cfg c x (.ballot d p) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI] at hr
  split at hr
  · rename_i hs
    obtain rfl := (Option.some.inj hr).symm
    simp only [Body.stmt, Body.stmtT, Body.leaf]
    refine wpE_bind (wpE_ipred hE hS p (Extends.refl _) fun pv Γ1 hx1 hp => ?_)
    refine wpE_bind (wpE_coll1 hA (c := .ballot) rfl Body.ballotR rfl hp hs rfl (valAt_words hW _) ?_)
    refine wpE_bind (wpE_uext32 (get_push_size) fun v Γ2 hx2 hv => ?_)
    have hx := hx1.trans ((Extends.push _ _).trans hx2)
    have hlt := ansW_lt x.ans (collCalls st.w x.start).length
    refine wpE_pure (hQ _ _ hx ?_ (FEnv.ofF (hF.extend hx)))
    exact (hE.extend hx).setW d (assign_flag hwf (by simp [Body.assigns])) v _
      (ansW x.ans (collCalls st.w x.start).length &&& 0xffffffff) (m32_lt _) (by rw [hv]; rfl)
      (by rw [hx2.get (get_push_size), m32_of_lt hlt]; rfl)
  · cases hr

theorem code_shflIdx (hA : AnswersFrom x.ans x.start cfg) (hW : Words x.ans) (d : Nat) (v sr : IExp) :
    CodeOk cfg c x (.shflIdx d v sr) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI] at hr
  split at hr
  · rename_i hs
    obtain rfl := (Option.some.inj hr).symm
    simp only [Body.stmt, Body.stmtT, Body.leaf]
    refine wpE_bind ((wpE_exp hE hS v).2 (Extends.refl _) fun vv Γ1 hx1 hvv => ?_)
    refine wpE_bind ((wpE_exp hE hS sr).2 hx1 fun vs Γ2 hx2 hvs => ?_)
    refine wpE_bind (wpE_coll2 hA (c := .shflIdx) rfl Body.shflIdxR rfl (hx2.get hvv) hvs hs rfl
      (valAt_words hW _) ?_)
    refine wpE_bind (wpE_uext32 (get_push_size) fun o Γ3 hx3 ho => ?_)
    have hx := hx1.trans (hx2.trans ((Extends.push _ _).trans hx3))
    have hlt := ansW_lt x.ans (collCalls st.w x.start).length
    refine wpE_pure (hQ _ _ hx ?_ (FEnv.ofF (hF.extend hx)))
    exact (hE.extend hx).setW d (assign_flag hwf (by simp [Body.assigns])) o _
      (ansW x.ans (collCalls st.w x.start).length &&& 0xffffffff) (m32_lt _) (by rw [ho]; rfl)
      (by rw [hx3.get (get_push_size), m32_of_lt hlt]; rfl)
  · cases hr

theorem code_barrier (hA : AnswersFrom x.ans x.start cfg) : CodeOk cfg c x .barrier := by
  intro r Γ st st' Q _ hE hF _ hr hQ
  simp only [laneI] at hr
  split at hr
  · rename_i hs
    split at hr
    · rename_i b hb
      obtain rfl := (Option.some.inj hr).symm
      simp only [Body.stmt, Body.stmtT, Body.leaf]
      exact wpE_bind (wpE_bar hA Kernel.Block.barrier rfl hs rfl hb (wpE_pure (hQ r Γ (Extends.refl _) hE hF)))
    · cases hr
  · cases hr

theorem code_setR (hex : ∀ v w, cfg.locals Intr.ex2Approx.slot [.sc .f32 v] w =
      .ok (some (.sc .f32 (x.hw.approx .ex2Approx v)), w))
    (hw32 : ∀ k v, (x.hw.approx k v).toNat < 2 ^ 32) (d : Nat) (e : WFExp) : CodeOk cfg c x (.setR d e) := by
  intro r Γ st st' Q _ hE hF _ hr hQ
  simp only [laneI, Option.some.injEq] at hr; subst hr
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_fexpL { x.hw with nan := st.w.mem.nan } hex r st.f e Γ st.w _ rfl hF.get
    fun v Γ1 hx1 hv => ?_)
  exact wpE_pure (hQ _ _ hx1 (EnvI.ofI (hE.extend hx1))
    ((hF.extend hx1).set d hv (evalF_32 { x.hw with nan := st.w.mem.nan } hw32 hF.w32 e)))

theorem bits32L {y : F32} (h : y.bits.toNat < 2 ^ 32) : y.bits = s32 y.bits.toUInt32 := by
  apply UInt64.toNat_inj.mp
  simp [s32, Nat.mod_eq_of_lt h]

theorem code_shflXor (hA : AnswersFrom x.ans x.start cfg) (hW : Words x.ans) (d sr m : Nat) :
    CodeOk cfg c x (.shflXor d sr m) := by
  intro r Γ st st' Q _ hE hF _ hr hQ
  simp only [laneI] at hr
  split at hr
  · rename_i hc
    obtain ⟨hm31, hs⟩ := hc
    obtain rfl := (Option.some.inj hr).symm
    simp only [Body.stmt, Body.stmtT, Body.leaf]
    have hss := hF.get sr
    rw [bits32L (hF.w32 sr)] at hss
    refine wpE_bind (wpE_bitcastI32 hss ?_)
    refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
    refine wpE_bind (wpE_coll2 hA (c := .shflBfly) rfl Kernel.Block.shflBfly rfl (get_push (get_push_size))
      (get_push_size) hs (k := (collCalls st.w x.start).length) rfl (valAt_words hW _) ?_)
    refine wpE_bind (wpE_bitcastF32 (get_push_size) ?_)
    have hwe : obsCall st.w (.intr .shflBfly) [.sc .i32 (s32 (st.f sr).bits.toUInt32),
        .sc .i32 (s32 (UInt32.ofNat (Int.ofNat m).toNat))] =
        callW st.w .shflBfly [.sc .i32 (st.f sr).bits, .sc .i32 (s32 (UInt32.ofNat m))] := by
      rw [← bits32L (hF.w32 sr)]; rfl
    rw [hwe]
    exact wpE_pure (hQ _ _ (by extc) (EnvI.ofI (hE.extend (by extc)))
      ((hF.extend (by extc)).set d (y := ⟨ansW x.ans (collCalls st.w x.start).length⟩) (get_push_size)
        (by simpa using UInt64.lt_iff_toNat_lt.mp (ansW_lt x.ans _))))
  · cases hr

theorem code_loadIdx (d : Nat) (b : Buf) (ix : IdxE) : CodeOk cfg c x (.loadIdx d b ix) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [laneI] at hr
  obtain ⟨j, hj, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨v, hv, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_idxV ix hS hj fun vi Γi hEi hvi => ?_)
  refine wpE_bind (wpE_at4L ((hS.extend hEi).buf b) hvi fun va Γa hEa hva => ?_)
  refine wpE_bind (wpE_loadF32 hva hv ?_)
  have hx : Extends Γ (Γa.push (.sc .f32 v)) := (hEi.trans hEa).push_of _
  exact wpE_pure (hQ _ _ hx (EnvI.ofI (hE.extend hx)) ((hF.extend hx).set d (y := ⟨v⟩) (get_push_size) (load4_lt hv)))

theorem code_loadV4 (d0 d1 d2 d3 : Nat) (b : Buf) (ix : IdxE) : CodeOk cfg c x (.loadV4 d0 d1 d2 d3 b ix) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [laneI] at hr
  obtain ⟨j, hj, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨v0, h0, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨v1, h1, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨v2, h2, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨v3, h3, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_idxV ix hS hj fun vi Γi hEi hvi => ?_)
  refine wpE_bind (wpE_at4L ((hS.extend hEi).buf b) hvi fun va Γa hEa hva => ?_)
  refine wpE_bind (wpE_loadF32 hva h0 ?_)
  refine wpE_bind (wpE_iaddImm (get_push hva) ?_)
  refine wpE_bind (wpE_loadF32 (get_push_size) h1 ?_)
  refine wpE_bind (wpE_iaddImm (get_push (get_push (get_push (get_push hva)))) ?_)
  refine wpE_bind (wpE_loadF32 (get_push_size) h2 ?_)
  refine wpE_bind (wpE_iaddImm (get_push (get_push (get_push (get_push (get_push (get_push (get_push hva))))))) ?_)
  refine wpE_bind (wpE_loadF32 (get_push_size) h3 ?_)
  refine wpE_pure (hQ _ _ (by extc) (EnvI.ofI (hE.extend (by extc))) ?_)
  exact (((((hF.extend (by extc)).set d0 (y := ⟨v0⟩) (by look) (load4_lt h0)).set d1 (y := ⟨v1⟩) (by look)
    (load4_lt h1)).set d2 (y := ⟨v2⟩) (by look) (load4_lt h2)).set d3 (y := ⟨v3⟩) (by look) (load4_lt h3))

theorem code_storeLane (b : Buf) (ix : IdxE) (n : Nat) : CodeOk cfg c x (.storeLane b ix n) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [laneI] at hr
  obtain ⟨j, hj, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨w1, hw1, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [stw] at hw1
  obtain ⟨m, hm, hw1⟩ := Option.bind_eq_some_iff.mp hw1
  obtain rfl := (Option.some.inj hw1).symm
  simp only [Body.stmt, Body.stmtT, Body.leaf]
  refine wpE_bind (wpE_idxV ix hS hj fun vi Γi hEi hvi => ?_)
  refine wpE_bind (wpE_at4L ((hS.extend hEi).buf b) hvi fun va Γa hEa hva => ?_)
  refine wpE_bind (wpE_storeF32 ((hF.extend (hEi.trans hEa)).get n) hva hm ?_)
  exact wpE_pure (hQ _ _ (hEi.trans hEa) (hE.extend (hEi.trans hEa)) (hF.extend (hEi.trans hEa)))

theorem code_storeLane0 (b : Buf) (ix : IdxE) (n : Nat) : CodeOk cfg c x (.storeLane0 b ix n) := by
  intro r Γ st st' Q _ hE hF hS hr hQ
  simp only [Body.stmt, Body.stmtT, Body.leaf, Kernel.Block.c32]
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  refine wpE_bind ?_
  simp only [when]
  have hl : (Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat))))[c.lane]? =
      some (.sc .i32 (s32 x.lv.toUInt32)) := by rw [s32_toUInt32 hS.lane32]; exact get_push hS.lane
  refine wpE_bind (wpE_ifteT (Qj := fun _ Γ' u => Extends Γ Γ' ∧ u = st'.w) rfl hl (get_push_size) ?_ ?_
    (fun jv _ _ _ => ⟨[], by cases jv; rfl⟩) ?_)
  · intro hcmp
    rw [cmp32_eq] at hcmp
    have hl0 : x.lv = 0 := by
      have h := beq_iff_eq.mp hcmp
      have := congrArg UInt32.toNat h
      have h32 : x.lv.toNat < 2 ^ 32 := lt_of_lt_of_eq (UInt64.lt_iff_toNat_lt.mp hS.lane32) rfl
      simp at this
      apply UInt64.toNat_inj.mp; simp; omega
    simp only [laneI, hl0, if_true] at hr
    obtain ⟨j, hj, hr⟩ := Option.bind_eq_some_iff.mp hr
    obtain ⟨w1, hw1, hr⟩ := Option.bind_eq_some_iff.mp hr
    obtain rfl := (Option.some.inj hr).symm
    simp only [stw] at hw1
    obtain ⟨m, hm, hw1⟩ := Option.bind_eq_some_iff.mp hw1
    obtain rfl := (Option.some.inj hw1).symm
    have hx0 : Extends Γ ((Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push (boolV true)) :=
      (Extends.push _ _).push_of _
    refine wpE_bind ?_
    refine wpE_bind (wpE_idxV ix (hS.extend hx0) hj fun vi Γi hEi hvi => ?_)
    refine wpE_bind (wpE_at4L ((hS.extend (hx0.trans hEi)).buf b) hvi fun va Γa hEa hva => ?_)
    refine wpE_storeF32 ((hF.extend ((hx0.trans hEi).trans hEa)).get n) hva hm ?_
    exact wpE_pure ⟨(hx0.trans hEi).trans hEa, rfl⟩
  · intro hcmp ne hne
    rw [cmp32_eq] at hcmp
    have hl0 : x.lv ≠ 0 := by
      intro h; rw [h] at hcmp; exact absurd hcmp (by decide)
    simp only [laneI, hl0, if_false, Option.some.injEq] at hr
    subst hr
    exact wpE_pure ⟨((Extends.push _ _).push_of _).bindAt_of hne _, rfl⟩
  · intro nJ Γ' jv vs w' hle hvs ⟨hx, hw'⟩
    subst hw'
    have hn : Γ.size ≤ nJ := Nat.le_trans hx.size hle
    have hrg : st'.reg = st.reg ∧ st'.f = st.f := by
      simp only [laneI] at hr
      split at hr
      · obtain ⟨_, _, hr⟩ := Option.bind_eq_some_iff.mp hr
        obtain ⟨_, _, hr⟩ := Option.bind_eq_some_iff.mp hr
        obtain rfl := (Option.some.inj hr).symm; exact ⟨rfl, rfl⟩
      · obtain rfl := (Option.some.inj hr).symm; exact ⟨rfl, rfl⟩
    have hE' : EnvI c r st'.reg (bindAt Γ' nJ vs) := hrg.1 ▸ hE.extend (hx.bindAt hn _)
    have hF' : FEnv c r st'.f (bindAt Γ' nJ vs) := hrg.2 ▸ hF.extend (hx.bindAt hn _)
    exact wpE_pure (wpE_pure (hQ _ _ (hx.bindAt hn _) hE' hF'))

end Leaves

/-! ## Control -/

theorem mem_eraseDups' {a : Nat} : ∀ {l : List Nat}, a ∈ l.eraseDups ↔ a ∈ l
  | [] => by simp
  | x :: xs => by
    rw [List.eraseDups_cons, List.mem_cons, mem_eraseDups' (l := xs.filter fun b => !b == x), List.mem_filter,
      List.mem_cons]
    by_cases h : a = x <;> simp [h]
termination_by l => l.length
decreasing_by
  simp only [List.length_cons]
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- A loop's run keeps what every trip keeps. -/
theorem whileL_inv (step : LI → Option LI) (go : LI → Bool) (P : LI → Prop)
    (hs : ∀ s1 s2, step s1 = some s2 → P s1 → P s2) :
    ∀ (k : Nat) (s s' : LI), whileL step go k s = some s' → P s → P s'
  | 0, s, s', h, hp => by
      simp only [whileL] at h
      split at h
      · cases h
      · cases h; exact hp
  | k + 1, s, s', h, hp => by
      simp only [whileL] at h
      split at h
      · obtain ⟨s1, h1, h⟩ := Option.bind_eq_some_iff.mp h
        exact whileL_inv step go P hs k s1 s' h (hs s s1 h1 hp)
      · cases h; exact hp

/-- A fold's run keeps what every step keeps. -/
theorem foldlM_invL {α β : Type} (f : β → α → Option β) (P : β → Prop)
    (h : ∀ b a b', f b a = some b' → P b → P b') : ∀ (l : List α) (b b' : β), l.foldlM f b = some b' → P b → P b'
  | [], b, b', hl, hp => by cases hl; exact hp
  | a :: l, b, b', hl, hp => by
      simp only [List.foldlM_cons, bind, Option.bind] at hl
      split at hl
      · cases hl
      rename_i b1 h1
      exact foldlM_invL f P h l b1 b' hl (h b a b1 h1 hp)

/-- **A lane's run changes only the integer registers a statement sets.** -/
theorem laneI_frame : ∀ (s : EWStmt) (x : LCtx) (st st' : LI), laneI x s st = some st' →
    ∀ n, n ∉ Body.setsI s → st'.reg n = st.reg n := by
  intro s
  induction s with
  | skip => intro x st st' h n _; simp [laneI] at h; subst h; rfl
  | seq a b iha ihb =>
      intro x st st' h n hn
      simp only [laneI, bind, Option.bind] at h
      split at h
      · cases h
      rename_i s1 h1
      simp only [Body.setsI, mem_eraseDups', List.mem_append, not_or] at hn
      rw [ihb x s1 st' h n hn.2, iha x st s1 h1 n hn.1]
  | setI d e =>
      intro x st st' h n hn
      simp [laneI] at h; subst h
      simp only [Body.setsI, List.mem_singleton] at hn
      simp [LI.set, hn]
  | ldB d a | ldH d a =>
      intro x st st' h n hn
      simp only [laneI] at h
      obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      simp only [Body.setsI, List.mem_singleton] at hn
      simp [LI.set, hn]
  | stB a v g | stH a v g =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain ⟨w, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; rfl
      · cases h; rfl
  | ballot d p | shflIdx d p _ =>
      intro x st st' h n hn
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm
        simp only [Body.setsI, List.mem_singleton] at hn
        simp [LI.set, hn]
      · cases h
  | barrier =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · split at h
        · obtain rfl := (Option.some.inj h).symm; rfl
        · cases h
      · cases h
  | iif cnd a b iha ihb =>
      intro x st st' h n hn
      simp only [Body.setsI, mem_eraseDups', List.mem_append, not_or] at hn
      simp only [laneI] at h
      split at h
      · exact iha x st st' h n hn.1
      · exact ihb x st st' h n hn.2
  | iwhile k cnd cs body _ =>
      intro x st st' h n hn
      simp only [Body.setsI] at hn
      simp only [laneI] at h
      refine whileL_inv _ _ (fun s => s.reg n = st.reg n) (fun s1 s2 h2 _ => ?_) k st st' h rfl
      obtain ⟨s3, _, h3⟩ := Option.map_eq_some_iff.mp h2
      subst h3
      simp [keepL, hn]
  | locals rs body ih =>
      intro x st st' h n hn
      simp only [laneI] at h
      obtain ⟨s1, h1, h⟩ := Option.map_eq_some_iff.mp h
      subst h
      simp only [restoreL]
      split
      · rfl
      · rename_i hr
        exact ih x st s1 h1 n (fun hm => hn (by simp [Body.setsI, List.mem_filter, hm, hr]))
  | setR d e =>
      intro x st st' h n _
      simp only [laneI, Option.some.injEq] at h; subst h; rfl
  | shflXor d sr m =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm; rfl
      · cases h
  | loadIdx d b ix =>
      intro x st st' h n _
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | loadV4 d0 d1 d2 d3 b ix =>
      intro x st st' h n _
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | storeLane b ix k =>
      intro x st st' h n _
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | storeLane0 b ix k =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; rfl
      · obtain rfl := (Option.some.inj h).symm; rfl
  | forN k body ih =>
      intro x st st' h n hn
      simp only [laneI] at h
      split at h
      · simp only [Body.setsI] at hn
        by_cases hk : k = 0
        · subst hk; simp at h; subst h; rfl
        · rw [if_neg hk] at hn
          exact foldlM_invL _ (fun s => s.reg n = st.reg n) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12 n hn).trans h1)
            _ _ _ h rfl
      · cases h
  | forM b a body ih =>
      intro x st st' h n hn
      simp only [Body.setsI] at hn
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        split at h
        · exact foldlM_invL _ (fun s => s.reg n = st.reg n) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12 n hn).trans h1)
            _ _ _ h rfl
        · cases h
      · cases h
  | stSm | ldSm | cvtIF => intro x st st' h; simp [laneI] at h

/-- **A lane's run changes only the float registers a statement writes.** -/
theorem laneI_fframe : ∀ (s : EWStmt) (x : LCtx) (st st' : LI), laneI x s st = some st' →
    ∀ n, n ∉ Body.written s → st'.f n = st.f n := by
  intro s
  induction s with
  | skip => intro x st st' h n _; simp [laneI] at h; subst h; rfl
  | seq a b iha ihb =>
      intro x st st' h n hn
      simp only [laneI, bind, Option.bind] at h
      split at h
      · cases h
      rename_i s1 h1
      simp only [Body.written, mem_eraseDups', List.mem_append, not_or] at hn
      rw [ihb x s1 st' h n hn.2, iha x st s1 h1 n hn.1]
  | setI d e => intro x st st' h n _; simp [laneI] at h; subst h; rfl
  | ldB d a | ldH d a =>
      intro x st st' h n _
      simp only [laneI] at h
      obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | stB a v g | stH a v g =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain ⟨w, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; rfl
      · cases h; rfl
  | ballot d p | shflIdx d p _ =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm; rfl
      · cases h
  | barrier =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · split at h
        · obtain rfl := (Option.some.inj h).symm; rfl
        · cases h
      · cases h
  | iif cnd a b iha ihb =>
      intro x st st' h n hn
      simp only [Body.written, mem_eraseDups', List.mem_append, not_or] at hn
      simp only [laneI] at h
      split at h
      · exact iha x st st' h n hn.1
      · exact ihb x st st' h n hn.2
  | iwhile k cnd cs body ih =>
      intro x st st' h n hn
      simp only [Body.written] at hn
      simp only [laneI] at h
      refine whileL_inv _ _ (fun s => s.f n = st.f n) (fun s1 s2 h2 h1 => ?_) k st st' h rfl
      obtain ⟨s3, h3, rfl⟩ := Option.map_eq_some_iff.mp h2
      exact (ih x s1 s3 h3 n hn).trans h1
  | locals rs body ih =>
      intro x st st' h n hn
      simp only [laneI] at h
      obtain ⟨s1, h1, h⟩ := Option.map_eq_some_iff.mp h
      subst h
      exact ih x st s1 h1 n hn
  | setR d e =>
      intro x st st' h n hn
      simp only [laneI, Option.some.injEq] at h; subst h
      simp only [Body.written, List.mem_singleton] at hn
      simp [LI.setF, hn]
  | shflXor d sr m =>
      intro x st st' h n hn
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm
        simp only [Body.written, List.mem_singleton] at hn
        simp [LI.setF, hn]
      · cases h
  | loadIdx d b ix =>
      intro x st st' h n hn
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      simp only [Body.written, List.mem_singleton] at hn
      simp [LI.setF, hn]
  | loadV4 d0 d1 d2 d3 b ix =>
      intro x st st' h n hn
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      simp only [Body.written, List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
      simp [LI.setF, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2]
  | storeLane b ix k =>
      intro x st st' h n _
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | storeLane0 b ix k =>
      intro x st st' h n _
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; rfl
      · obtain rfl := (Option.some.inj h).symm; rfl
  | forN k body ih =>
      intro x st st' h n hn
      simp only [Body.written] at hn
      simp only [laneI] at h
      split at h
      · exact foldlM_invL _ (fun s => s.f n = st.f n) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12 n hn).trans h1)
          _ _ _ h rfl
      · cases h
  | forM b a body ih =>
      intro x st st' h n hn
      simp only [Body.written] at hn
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        split at h
        · exact foldlM_invL _ (fun s => s.f n = st.f n) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12 n hn).trans h1)
            _ _ _ h rfl
        · cases h
      · cases h
  | stSm | ldSm | cvtIF => intro x st st' h; simp [laneI] at h

theorem stw_nan {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') :
    w'.mem.nan = w.mem.nan := by
  simp only [stw] at h
  obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
  obtain rfl := (Option.some.inj h).symm
  exact Mem.store_nan hm

/-- **A lane's run keeps the world's NaN rule.** -/
theorem laneI_nan : ∀ (s : EWStmt) (x : LCtx) (st st' : LI), laneI x s st = some st' →
    st'.w.mem.nan = st.w.mem.nan := by
  intro s
  induction s with
  | skip => intro x st st' h; simp [laneI] at h; subst h; rfl
  | seq a b iha ihb =>
      intro x st st' h
      simp only [laneI, bind, Option.bind] at h
      split at h
      · cases h
      rename_i s1 h1
      rw [ihb x s1 st' h, iha x st s1 h1]
  | setI d e => intro x st st' h; simp [laneI] at h; subst h; rfl
  | ldB d a | ldH d a =>
      intro x st st' h
      simp only [laneI] at h
      obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | stB a v g | stH a v g =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; exact stw_nan hw
      · cases h; rfl
  | ballot d p | shflIdx d p _ =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm; rfl
      · cases h
  | barrier =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · split at h
        · obtain rfl := (Option.some.inj h).symm; rfl
        · cases h
      · cases h
  | iif cnd a b iha ihb =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · exact iha x st st' h
      · exact ihb x st st' h
  | iwhile k cnd cs body ih =>
      intro x st st' h
      simp only [laneI] at h
      refine whileL_inv _ _ (fun s => s.w.mem.nan = st.w.mem.nan) (fun s1 s2 h2 h1 => ?_) k st st' h rfl
      obtain ⟨s3, h3, rfl⟩ := Option.map_eq_some_iff.mp h2
      exact (ih x s1 s3 h3).trans h1
  | locals rs body ih =>
      intro x st st' h
      simp only [laneI] at h
      obtain ⟨s1, h1, h⟩ := Option.map_eq_some_iff.mp h
      subst h
      exact ih x st s1 h1
  | setR d e =>
      intro x st st' h
      simp only [laneI, Option.some.injEq] at h; subst h
      exact congrArg Mem.nan (exLog_mem _ st.f e st.w)
  | shflXor d sr m =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · obtain rfl := (Option.some.inj h).symm; rfl
      · cases h
  | loadIdx d b ix =>
      intro x st st' h
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | loadV4 d0 d1 d2 d3 b ix =>
      intro x st st' h
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; rfl
  | storeLane b ix k =>
      intro x st st' h
      simp only [laneI] at h
      obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨w1, hw1, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm; exact stw_nan hw1
  | storeLane0 b ix k =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨w1, hw1, h⟩ := Option.bind_eq_some_iff.mp h
        obtain rfl := (Option.some.inj h).symm; exact stw_nan hw1
      · obtain rfl := (Option.some.inj h).symm; rfl
  | forN k body ih =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · exact foldlM_invL _ (fun s => s.w.mem.nan = st.w.mem.nan) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12).trans h1)
          _ _ _ h rfl
      · cases h
  | forM b a body ih =>
      intro x st st' h
      simp only [laneI] at h
      split at h
      · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
        split at h
        · exact foldlM_invL _ (fun s => s.w.mem.nan = st.w.mem.nan) (fun s1 j s2 h12 h1 => (ih _ s1 s2 h12).trans h1)
            _ _ _ h rfl
        · cases h
      · cases h
  | stSm | ldSm | cvtIF => intro x st st' h; simp [laneI] at h

/-- A statement of register assignments' code is its code. -/
theorem straightCode_eq : ∀ (s : EWStmt), Body.straight s = true → ∀ (r : Body.Regs Slot),
    Body.straightCode (L := Lvl) c r s = Body.stmt c r s := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro h r
      simp only [Body.straight, Bool.and_eq_true] at h
      simp only [Body.straightCode, Body.stmt, Body.stmtT]
      rw [iha h.1 r]
      congr 1
      funext y
      exact ihb h.2 y
  | skip | setR | setI => intro _ _; rfl
  | _ => intro h; simp [Body.straight] at h

variable (cfg c x) in
/-- **What a statement's code does at the end of a loop's body**: where the
    lane's run of it then of the tail's assignments ends, the code goes round
    the loop with the carries the lane ends with. -/
def TailOk (s : EWStmt) : Prop :=
  ∀ (t : Body.Tail Slot Lvl) (r : Body.Regs Slot) (Γ : Env) (st st' : LI) (C : List V → World → Prop),
    Body.straight t.post = true → CodeOk cfg c x t.post → Wf c t.post →
    (∀ n ∈ t.rf, c.fl.contains n = true) → Wf c s →
    EnvI c r st.reg Γ → FEnv c r st.f Γ → Slots c Γ x → laneI x (.seq s t.post) st = some st' →
    ((∀ n, c.fl.contains n = true → st'.reg n ≤ 1) → (∀ n, (st'.f n).bits.toNat < 2 ^ 32) →
      C (carried st' t.ws t.rv t.rf) st'.w) →
    wpC cfg t.lp (Body.stmtT c (some t) r s) C Γ st.w

/-- **Going round from a body's end**: the tail's assignments, then the jump
    with the carries the lane ends with. -/
theorem goRound_ok {t : Body.Tail Slot Lvl} {r : Body.Regs Slot} {Γ : Env} {st st' : LI}
    {C : List V → World → Prop} (hs : Body.straight t.post = true)
    (hpost : CodeOk cfg c x t.post) (hwp : Wf c t.post)
    (hrf : ∀ n ∈ t.rf, c.fl.contains n = true) (hE : EnvI c r st.reg Γ) (hF : FEnv c r st.f Γ)
    (hS : Slots c Γ x) (hr : laneI x t.post st = some st')
    (hC : (∀ n, c.fl.contains n = true → st'.reg n ≤ 1) → (∀ n, (st'.f n).bits.toNat < 2 ^ 32) →
      C (carried st' t.ws t.rv t.rf) st'.w) :
    wpC cfg t.lp (Body.goRound c t r) C Γ st.w := by
  obtain ⟨ws, rv, rf, lp, post⟩ := t
  simp only at hs hpost hwp hrf hr hC ⊢
  unfold Body.goRound
  simp only
  rw [straightCode_eq post hs r]
  refine wpC_bind (hpost r Γ st st' _ hwp hE hF hS hr fun r' Γ' _ hE' hF' => ?_)
  exact wpC_bind (wpE_pack hE' hF' hrf fun pv Γ'' _ hpv => wpC_cont rfl hpv (hC hE'.bit hF'.w32))

/-- A statement whose code at a body's end is its code, then going round,
    goes round. -/
theorem tail_of_code {s : EWStmt} (hA : CodeOk cfg c x s)
    (heq : ∀ (t : Body.Tail Slot Lvl) (r : Body.Regs Slot),
      Body.stmtT c (some t) r s = (Body.stmt c r s >>= fun r' => Body.goRound c t r')) :
    TailOk cfg c x s := by
  intro t r Γ st st' C hs hpost hwp hrf hwf hE hF hS hr hC
  rw [heq]
  simp only [laneI, bind, Option.bind] at hr
  split at hr
  · cases hr
  rename_i s1 h1
  exact wpC_bind (hA r Γ st s1 _ hwf hE hF hS h1 fun r' Γ' hx hE' hF' =>
    goRound_ok hs hpost hwp hrf hE' hF' (hS.extend hx) hr hC)

theorem wf_seq {a b : EWStmt} (h : Wf c (.seq a b)) : Wf c a ∧ Wf c b :=
  ⟨fun p hp => h p (by simp [Body.assigns, hp]), fun p hp => h p (by simp [Body.assigns, hp])⟩

theorem CodeOk.seq {a b : EWStmt} (ha : CodeOk cfg c x a) (hb : CodeOk cfg c x b) : CodeOk cfg c x (.seq a b) := by
  intro r Γ st st' Q hwf hE hF hS hr hQ
  simp only [laneI, bind, Option.bind] at hr
  split at hr
  · cases hr
  rename_i st1 h1
  have hw := wf_seq hwf
  simp only [Body.stmt, Body.stmtT]
  exact wpE_bind (ha r Γ st st1 _ hw.1 hE hF hS h1 fun r1 Γ1 hx1 hE1 hF1 =>
    hb r1 Γ1 st1 st' Q hw.2 hE1 hF1 (hS.extend hx1) hr
      fun r2 Γ2 hx2 hE2 hF2 => hQ r2 Γ2 (hx1.trans hx2) hE2 hF2)

theorem laneI_seq_assoc (a b d : EWStmt) (st : LI) :
    laneI x (.seq (.seq a b) d) st = laneI x (.seq a (.seq b d)) st := by
  simp only [laneI, bind, Option.bind_assoc]

theorem vregs_mem {rs : List Nat} {n : Nat} (h : n ∈ Body.vregs c rs) : c.fl.contains n = false := by
  simpa [Body.vregs] using (List.mem_filter.mp h).2

theorem pregs_mem {rs : List Nat} {n : Nat} (h : n ∈ Body.pregs c rs) : c.fl.contains n = true := by
  simpa [Body.pregs] using (List.mem_filter.mp h).2

theorem not_split {rs : List Nat} {n : Nat} (h1 : n ∉ Body.vregs c rs) (h2 : n ∉ Body.pregs c rs) : n ∉ rs := by
  intro hm
  cases hc : c.fl.contains n
  · exact h1 (List.mem_filter.mpr ⟨hm, by simpa using hc⟩)
  · exact h2 (List.mem_filter.mpr ⟨hm, by simpa using hc⟩)

theorem whileL_stop {step : LI → Option LI} {go : LI → Bool} {s : LI} (h : go s = false) :
    ∀ j, whileL step go j s = some s
  | 0 => by simp [whileL, h]
  | _ + 1 => by simp [whileL, h]

theorem eval_icmp {m : Mem} {Γ : Env} {t : ClifTy} {kk : ICmpCond} {a b : Slot t} {X Y : UInt64}
    (hti : t.isInt = true) (ha : Γ[a]? = some (.sc t X)) (hb : Γ[b]? = some (.sc t Y)) :
    evalOp m Γ (.icmp kk a b) = some (boolV (cmpInt kk t X Y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntCmp, hti]

theorem keepL_carried {cs ws rv rf : List Nat} {s0 s : LI} (hv : ∀ n ∈ rv, n ∈ cs) (hf : ∀ n ∈ rf, n ∈ cs) :
    carried (keepL cs s0 s) ws rv rf = carried s ws rv rf := by
  simp only [carried, keepL]
  congr 2
  · exact List.map_congr_left fun n hn => by simp [hv n hn]
  · exact List.map_congr_left fun n hn => by simp [hf n hn]

theorem uniformToList_carries {ty : ClifTy} : ∀ (n k : Nat),
    Vals.uniformToList (carriesFrom k (List.replicate n ty)) = List.range' k n
  | 0, _ => rfl
  | n + 1, k => by
    simp [List.replicate_succ, carriesFrom, Vals.uniformToList, Vals.head, Vals.tail,
      uniformToList_carries n (k + 1), List.range'_succ]

/-! ### Counted loops -/

/-- A loop's first `j` trips of a lane's run. -/
def tripsL (f : LI → Nat → Option LI) (st : LI) (j : Nat) : LI := ((List.range j).foldlM f st).getD st

theorem tripsL_succ {f : LI → Nat → Option LI} {st st' : LI} {n : Nat}
    (h : (List.range n).foldlM f st = some st') :
    ∀ j, j < n → f (tripsL f st j) j = some (tripsL f st (j + 1)) := by
  intro j hj
  have key : ∀ m, m ≤ n → ∃ s, (List.range m).foldlM f st = some s := by
    intro m hm
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hm
    rw [List.range_add, List.foldlM_append] at h
    cases hm' : (List.range m).foldlM f st with
    | none => rw [hm'] at h; cases h
    | some s => exact ⟨s, rfl⟩
  obtain ⟨s, hs⟩ := key j (by omega)
  obtain ⟨s', hs'⟩ := key (j + 1) hj
  rw [List.range_succ, List.foldlM_append, hs] at hs'
  simp only [List.foldlM_cons, List.foldlM_nil, bind, Option.bind] at hs'
  unfold tripsL
  rw [hs, List.range_succ, List.foldlM_append, hs]
  simp only [Option.getD_some, List.foldlM_cons, List.foldlM_nil, bind, Option.bind]
  cases hf : f s j with
  | none => rw [hf] at hs'; cases hs'
  | some s2 => rfl

theorem tripsL_end {f : LI → Nat → Option LI} {st st' : LI} {n : Nat}
    (h : (List.range n).foldlM f st = some st') : tripsL f st n = st' := by
  unfold tripsL; rw [h]; rfl

theorem EnvI.fl {c c' : Body.Ctx Slot} {r : Body.Regs Slot} {reg : Nat → UInt64} {Γ : Env}
    (h : EnvI c r reg Γ) (hfl : c'.fl = c.fl := by rfl) : EnvI c' r reg Γ := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := h
  exact ⟨h1, h2, h3, by rw [hfl]; exact h4, by rw [hfl]; exact h5, by rw [hfl]; exact h6, by rw [hfl]; exact h7,
    by rw [hfl]; exact h8, by rw [hfl]; exact h9⟩

theorem FEnv.zf {c c' : Body.Ctx Slot} {r : Body.Regs Slot} {f : Nat → F32} {Γ : Env}
    (h : FEnv c r f Γ) (hzf : c'.zf = c.zf := by rfl) : FEnv c' r f Γ :=
  ⟨fun n => by unfold Body.get; rw [hzf]; exact h.get n, h.w32⟩

theorem s32_ofNat_toNatL {j : Nat} (hj : j < 2 ^ 32) : (s32 (UInt32.ofNat j)).toNat = j := by
  simp only [s32, UInt32.toNat_toUInt64, toNat_ofNat32 hj]

/-- **A counted loop carrying registers ends as its trips do**: trips
    `0 … n-1` of `run`, each ending as the lane's trip, end with the lane's
    state after `n`. -/
theorem wpE_loopXL {r : Body.Regs Slot} {Γ : Env} {nS : Slot .i32} {n : Nat} (ws rs : List Nat)
    (run : Body.Ctx Slot → Body.Regs Slot → Prog Slot Lvl (Body.Regs Slot)) (stF : Nat → LI)
    (hnS : Γ[nS]? = some (.sc .i32 (s32 (UInt32.ofNat n)))) (hn : n < 2 ^ 31)
    (hS : Slots c Γ x) (hE : EnvI c r (stF 0).reg Γ) (hF : FEnv c r (stF 0).f Γ)
    (hfr : ∀ j, j ≤ n → ∀ k, k ∉ rs → (stF j).reg k = (stF 0).reg k)
    (hff : ∀ j, j ≤ n → ∀ k, k ∉ ws → (stF j).f k = (stF 0).f k)
    (hrun : ∀ j, j < n → ∀ (sl : Slot .i32) (r' : Body.Regs Slot) (Γ' : Env)
      (Q' : Body.Regs Slot → Env → World → Prop),
      Slots { c with i := sl } Γ' { x with i := j } → EnvI c r' (stF j).reg Γ' → FEnv c r' (stF j).f Γ' →
      (∀ r'' Γ'', Extends Γ' Γ'' → EnvI c r'' (stF (j + 1)).reg Γ'' → FEnv c r'' (stF (j + 1)).f Γ'' →
        Q' r'' Γ'' (stF (j + 1)).w) →
      wpE cfg (run { c with i := sl } r') Q' Γ' (stF j).w)
    {Q : Body.Regs Slot → Env → World → Prop}
    (hQ : ∀ r' Γ', Extends Γ Γ' → EnvI c r' (stF n).reg Γ' → FEnv c r' (stF n).f Γ' → Q r' Γ' (stF n).w) :
    wpE cfg (Body.stmtT.loopX c r nS ws rs run) Q Γ (stF 0).w := by
  simp only [Body.stmtT.loopX, Kernel.Block.c32]
  have hv0 : ∀ m ∈ Body.vregs c rs, c.fl.contains m = false := fun m hm => vregs_mem hm
  have hf0 : ∀ m ∈ Body.pregs c rs, c.fl.contains m = true := fun m hm => pregs_mem hm
  have hout : ∀ m, m ∉ Body.vregs c rs → m ∉ Body.pregs c rs → m ∉ rs := fun m h1 h2 => not_split h1 h2
  generalize Body.vregs c rs = rv at hv0 hout ⊢
  generalize Body.pregs c rs = rf at hf0 hout ⊢
  have hnlt : nS < Γ.size := by
    rcases Nat.lt_or_ge nS Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hnS; cases hnS
  refine wpE_bind (wpE_pack hE hF hf0 fun init Γp hxp hinit => ?_)
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  generalize hΓ0 : Γp.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat))) = Γ0
  have hx0 : Extends Γ Γ0 := by rw [← hΓ0]; exact hxp.push_of _
  have hsz0 : Γ0.size = Γp.size + 1 := by rw [← hΓ0]; simp
  have hnS0 : Γ0[nS]? = some (.sc .i32 (s32 (UInt32.ofNat n))) := hx0.get hnS
  have hlt0 : nS < Γ0.size := by
    rw [hsz0]; exact Nat.lt_of_lt_of_le hnlt (by have := hxp.1; omega)
  let Bits (j : Nat) : Prop :=
    (∀ k, c.fl.contains k = true → (stF j).reg k ≤ 1) ∧ ∀ k, ((stF j).f k).bits.toNat < 2 ^ 32
  let Inv (cs : List V) (wi : World) : Prop := ∃ j, j ≤ n ∧
    cs = V.sc .i32 (s32 (UInt32.ofNat j)) :: carried (stF j) ws rv rf ∧ wi = (stF j).w ∧ Bits j
  let μ (cs : List V) (_ : World) : Nat := match cs with
    | .sc _ y :: _ => n - y.toNat
    | _ => 0
  let Hd (cs : List V) (_ : World) (a : Cond Slot × Vals Slot (Body.carryTys ws rv rf) × Unit) (Γ1 : Env)
      (w1 : World) : Prop :=
    a = (contIfULt Γ0.size nS, carriesFrom (Γ0.size + 1) (Body.carryTys ws rv rf), ()) ∧
    Γ1 = bindAt Γ0 Γ0.size cs ∧ Inv cs w1
  let X (vs : List V) (wx : World) : Prop := vs = carried (stF n) ws rv rf ∧ wx = (stF n).w ∧ Bits n
  have hev : ∀ j (w1 : World) (cs : List V), j ≤ n →
      evalOp w1.mem (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs)) (.icmp .ult Γ0.size nS) =
        some (boolV (cmpInt .ult .i32 (s32 (UInt32.ofNat j)) (s32 (UInt32.ofNat n)))) := by
    intro j w1 cs _
    have h1 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs))[Γ0.size]? =
        some (V.sc .i32 (s32 (UInt32.ofNat j))) := bindAt_head _ _ _ _
    have h2 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs))[nS]? =
        some (V.sc .i32 (s32 (UInt32.ofNat n))) := by
      rw [bindAt_lt hlt0 hlt0]; exact hnS0
    simp [evalOp, Sem.get, h1, h2, zipIntCmp, ClifTy.isInt]
  refine wpE_bind (wpE_wloopN Inv μ Hd X ?_ ?_ ?_ ?_ ?_ ?_)
  · refine ⟨_, ?_, 0, Nat.zero_le _, rfl, rfl, hE.bit, hF.w32⟩
    have hi0 : Γ0[Γp.size]? = some (V.sc .i32 (s32 (UInt32.ofNat 0))) := by
      rw [← hΓ0]; exact get_push_size
    simp only [Vals.slots, List.mapM_cons, hi0, Sem.mapM_ext (hx0.trans (Extends.refl _) |> fun _ =>
      (by rw [← hΓ0]; exact Extends.push _ _ : Extends Γp Γ0)) hinit]
    rfl
  · intro wi cs hinv
    exact wpE_pure ⟨Extends.refl _, rfl, rfl, hinv⟩
  · rintro cs _ a Γ1 w1 ⟨rfl, rfl, j, hj, rfl, -, -⟩
    exact ⟨_, _, by simp only [contIfULt, contIf]; rw [hev j w1 _ hj]; rfl⟩
  · rintro cs _ a Γ1 w1 t f ⟨rfl, rfl, j, hj, rfl, rfl, hbj⟩ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j _ _ hj] at hev'
    have hjn : j = n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        omega
    subst hjn
    refine ⟨carried (stF j) ws rv rf, ?_, rfl, rfl, hbj⟩
    have hrd := carries_read (Γ := Γ0) (n := Γ0.size)
      (xs := V.sc .i32 (s32 (UInt32.ofNat j)) :: carried (stF j) ws rv rf) (Body.carryTys ws rv rf) 1
      (by simp [carried_length]; omega)
    simp only [List.drop_one, List.tail_cons] at hrd
    exact Sem.mapM_ext (Extends.push _ _) hrd
  · rintro cs _ a Γ1 wb t f ⟨rfl, rfl, j, hj, rfl, rfl, hbj⟩ _ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j _ _ hj] at hev'
    have hjn : j < n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        exact hlt
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
    generalize hΓ1 : bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: carried (stF j) ws rv rf) = Γ1
    have hE1 : Extends Γ Γ1 := by rw [← hΓ1]; exact hx0.trans ((Extends.refl _).bindAt_of (Nat.le_refl _) _)
    generalize hΓ2 : bindAt (Γ1.push (.sc t f)) (Γ1.size + 1)
      (V.sc .i32 (s32 (UInt32.ofNat j)) :: carried (stF j) ws rv rf) = Γ2
    have hE2 : Extends Γ Γ2 := by rw [← hΓ2]; exact (hE1.push_of _).trans ((Extends.refl _).bindAt_of (by simp) _)
    have hci : Γ2[Γ1.size + 1]? = some (V.sc .i32 (s32 (UInt32.ofNat j))) := by
      rw [← hΓ2]; exact bindAt_head _ _ _ _
    have hU := unpack_okG (n := Γ1.size + 1 + 1) (st' := stF j) hE hF hE2
      (fun i v hv => by
        rw [← hΓ2, show Γ1.size + 1 + 1 + i = Γ1.size + 1 + (i + 1) by omega, bindAt_at]; simpa using hv)
      hv0 hf0 hbj.1 (fun m h1 h2 => hfr j (by omega) m (hout m h1 h2)) (hff j (by omega)) hbj.2
    simp only [carriesFrom, Vals.head, Vals.tail]
    have hSl : Slots { c with i := Γ1.size + 1 } Γ2 { x with i := j } :=
      ⟨hE2.get hS.lane, hS.lane32, hE2.get hS.cta, hS.cta32, fun b => hE2.get (hS.buf b), hE2.get hS.z64,
        hE2.get hS.sm, hci, hE2.get hS.zf, hE2.get hS.zi⟩
    refine wpE_bind (hrun j hjn _ _ Γ2 _ hSl hU.1 hU.2 fun r'' Γ'' hx'' hE'' hF'' => ?_)
    refine wpE_bind (wpE_iconst32u (by decide) ?_)
    refine wpE_bind (wpE_iadd32 (x := UInt32.ofNat j) (get_push (hx''.get hci)) (get_push_size) ?_)
    refine wpE_bind (wpE_pack (hE''.extend ((Extends.push _ _).push_of _)) (hF''.extend ((Extends.push _ _).push_of _))
      hf0 fun pv Γ3 hx3 hpv => ?_)
    refine wpE_pure ⟨V.sc .i32 (s32 (UInt32.ofNat (j + 1))) :: carried (stF (j + 1)) ws rv rf, ?_,
      ⟨j + 1, by omega, rfl, rfl, hE''.bit, hF''.w32⟩, ?_⟩
    · simp only [Vals.slots, List.mapM_cons]
      rw [hx3.get (get_push_size), hpv]
      simp [UInt32.ofNat_add]
    · show n - (s32 (UInt32.ofNat (j + 1))).toNat < n - (s32 (UInt32.ofNat j)).toNat
      rw [s32_ofNat_toNatL (by omega), s32_ofNat_toNatL (by omega)]
      omega
  · rintro Γb vs w' nE hEb hnE ⟨rfl, rfl, hbn⟩
    have hEb' : Extends Γ Γb := hx0.trans hEb
    have hle : Γ.size ≤ nE := by have h1 := hxp.1; rw [hsz0] at hnE; omega
    have hU := unpack_ok (n := nE) (st' := stF n) hE hF hEb' hle hv0 hf0 hbn.1
      (fun m h1 h2 => hfr n (Nat.le_refl _) m (hout m h1 h2)) (hff n (Nat.le_refl _)) hbn.2
    exact wpE_pure (hQ _ _ (hEb'.bindAt hle _) hU.1 hU.2)

theorem slots_ofFnL {ty : ClifTy} : ∀ (n : Nat) (f : Fin n → Nat),
    (Vals.ofFn (V := Slot) (ty := ty) f).slots = List.ofFn f
  | 0, _ => rfl
  | n + 1, f => by simp [Vals.ofFn, Vals.slots, slots_ofFnL n, List.ofFn_succ]

theorem ofFn_getD_mapL {β : Type} (ws : List Nat) (f : Nat → β) :
    List.ofFn (fun j : Fin ws.length => f (ws.getD j.val 0)) = ws.map f := by
  apply List.ext_getElem <;> simp

theorem mapM_range'L (A : Env) : ∀ (vs : List V) (n : Nat), (∀ j, j < vs.length → A[n + j]? = vs[j]?) →
    (List.range' n vs.length).mapM (fun s => A[s]?) = some vs
  | [], _, _ => rfl
  | v :: vs, n, h => by
    simp only [List.length_cons, List.range'_succ, List.mapM_cons]
    rw [show A[n]? = some v by simpa using h 0 (by simp),
      mapM_range'L A vs (n + 1) (fun j hj => by rw [Nat.add_assoc, Nat.add_comm 1 j]; simpa using h (j + 1) (by simp; omega))]
    rfl

theorem slots_carriesL {ty : ClifTy} : ∀ (m k : Nat), (carriesFrom k (List.replicate m ty)).slots = List.range' k m
  | 0, _ => rfl
  | m + 1, k => by simp [List.replicate_succ, carriesFrom, Vals.slots, slots_carriesL m (k + 1), List.range'_succ]

/-- Float registers bound from slots `ss`, the others as before. -/
theorem FEnv.back {Γ : Env} {f : Nat → F32} (h32 : ∀ k, (f k).bits.toNat < 2 ^ 32) :
    ∀ (ws : List Nat) (ss : List Nat) (r : Body.Regs Slot), ws.length = ss.length →
      (∀ (p k sl : Nat), ws[p]? = some k → ss[p]? = some sl → Γ[sl]? = some (V.sc .f32 (f k).bits)) →
      (∀ k, k ∉ ws → Γ[Body.get c r k]? = some (.sc .f32 (f k).bits)) →
      FEnv c ((ws.zip ss).foldl (fun r p => Body.set r p.1 p.2) r) f Γ
  | [], [], r, _, _, hk => ⟨fun k => hk k (by simp), h32⟩
  | w0 :: ws, s0 :: ss, r, hl, hp, hk => by
    rw [List.zip_cons_cons, List.foldl_cons]
    refine FEnv.back h32 ws ss _ (by simpa using hl) (fun p k sl h1 h2 => hp (p + 1) k sl (by simpa using h1)
      (by simpa using h2)) ?_
    intro k hk'
    rw [get_setF]
    by_cases h : k = w0
    · subst h; simpa using hp 0 k s0 rfl rfl
    · simp only [if_neg h]; exact hk k (by simp [h, hk'])
  | [], _ :: _, _, hl, _, _ => by simp at hl
  | _ :: _, [], _, hl, _, _ => by simp at hl

theorem back_int (r : Body.Regs Slot) : ∀ (ps : List (Nat × Slot .f32)),
    (ps.foldl (fun r p => Body.set r p.1 p.2) r).v = r.v ∧ (ps.foldl (fun r p => Body.set r p.1 p.2) r).p = r.p ∧
    (ps.foldl (fun r p => Body.set r p.1 p.2) r).w = r.w
  | [] => ⟨rfl, rfl, rfl⟩
  | p :: ps => by
    rw [List.foldl_cons]
    exact back_int (Body.set r p.1 p.2) ps

theorem lookups_mapL {r : Body.Regs Slot} {Γ : Env} {f : Nat → F32} (h : FEnv c r f Γ) (ws : List Nat) :
    (ws.map (Body.get c r)).mapM (fun s => Γ[s]?) = some (ws.map fun k => V.sc .f32 (f k).bits) := by
  induction ws with
  | nil => rfl
  | cons k ks ih => simp [List.mapM_cons, h.get k, ih]

/-- **A counted loop of float registers ends as its trips do**: trips
    `0 … n-1` of `run`, each ending as the lane's trip, its integer registers
    left as they were, end with the lane's state after `n`. -/
theorem wpE_loopL {r : Body.Regs Slot} {Γ : Env} {nS : Slot .i32} {n : Nat} (ws : List Nat)
    (run : Body.Ctx Slot → Body.Regs Slot → Prog Slot Lvl (Body.Regs Slot)) (stF : Nat → LI)
    (hnS : Γ[nS]? = some (.sc .i32 (s32 (UInt32.ofNat n)))) (hn : n < 2 ^ 31)
    (hS : Slots c Γ x) (hE : EnvI c r (stF 0).reg Γ) (hF : FEnv c r (stF 0).f Γ)
    (hreg : ∀ j, j ≤ n → (stF j).reg = (stF 0).reg)
    (hff : ∀ j, j ≤ n → ∀ k, k ∉ ws → (stF j).f k = (stF 0).f k)
    (hrun : ∀ j, j < n → ∀ (sl : Slot .i32) (r' : Body.Regs Slot) (Γ' : Env)
      (Q' : Body.Regs Slot → Env → World → Prop),
      Slots { c with i := sl } Γ' { x with i := j } → EnvI c r' (stF j).reg Γ' → FEnv c r' (stF j).f Γ' →
      (∀ r'' Γ'', Extends Γ' Γ'' → EnvI c r'' (stF (j + 1)).reg Γ'' → FEnv c r'' (stF (j + 1)).f Γ'' →
        Q' r'' Γ'' (stF (j + 1)).w) →
      wpE cfg (run { c with i := sl } r') Q' Γ' (stF j).w)
    {Q : Body.Regs Slot → Env → World → Prop}
    (hQ : ∀ r' Γ', Extends Γ Γ' → EnvI c r' (stF n).reg Γ' → FEnv c r' (stF n).f Γ' → Q r' Γ' (stF n).w) :
    wpE cfg (Body.loop c r nS ws run) Q Γ (stF 0).w := by
  simp only [Body.loop, Kernel.Block.c32]
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  have hnlt : nS < Γ.size := by
    rcases Nat.lt_or_ge nS Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hnS; cases hnS
  generalize hΓ0 : Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat))) = Γ0
  have hx0 : Extends Γ Γ0 := by rw [← hΓ0]; exact Extends.push _ _
  have hsz0 : Γ0.size = Γ.size + 1 := by rw [← hΓ0]; simp
  have hlt0 : nS < Γ0.size := by rw [hsz0]; exact Nat.lt_succ_of_lt hnlt
  have hnS0 : Γ0[nS]? = some (.sc .i32 (s32 (UInt32.ofNat n))) := hx0.get hnS
  let vals (j : Nat) : List V := ws.map fun k => V.sc .f32 ((stF j).f k).bits
  let W32 (j : Nat) : Prop := ∀ k, ((stF j).f k).bits.toNat < 2 ^ 32
  let Inv (cs : List V) (wi : World) : Prop := ∃ j, j ≤ n ∧ cs = V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j ∧
    wi = (stF j).w ∧ W32 j
  let μ (cs : List V) (_ : World) : Nat := match cs with
    | .sc _ y :: _ => n - y.toNat
    | _ => 0
  let Hd (cs : List V) (_ : World) (a : Cond Slot × Vals Slot (List.replicate ws.length .f32) × Unit) (Γ1 : Env)
      (w1 : World) : Prop :=
    a = (contIfULt Γ0.size nS, carriesFrom (Γ0.size + 1) (List.replicate ws.length .f32), ()) ∧
    Γ1 = bindAt Γ0 Γ0.size cs ∧ Inv cs w1
  let X (vs : List V) (wx : World) : Prop := vs = vals n ∧ wx = (stF n).w ∧ W32 n
  have hev : ∀ j (w1 : World) (cs : List V), j ≤ n →
      evalOp w1.mem (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs)) (.icmp .ult Γ0.size nS) =
        some (boolV (cmpInt .ult .i32 (s32 (UInt32.ofNat j)) (s32 (UInt32.ofNat n)))) := by
    intro j w1 cs _
    have h1 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs))[Γ0.size]? =
        some (V.sc .i32 (s32 (UInt32.ofNat j))) := bindAt_head _ _ _ _
    have h2 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: cs))[nS]? =
        some (V.sc .i32 (s32 (UInt32.ofNat n))) := by
      rw [bindAt_lt hlt0 hlt0]; exact hnS0
    simp [evalOp, Sem.get, h1, h2, zipIntCmp, ClifTy.isInt]
  have hbackF : ∀ (j : Nat) (Γ2 : Env) (base : Nat), j ≤ n → W32 j → Extends Γ Γ2 →
      (∀ p, p < ws.length → Γ2[base + p]? = (vals j)[p]?) →
      FEnv c (List.foldl (fun r x => Body.set r x.fst x.snd) r
        (ws.zip (carriesFrom base (List.replicate ws.length ClifTy.f32)).uniformToList)) (stF j).f Γ2 := by
    intro j Γ2 base hj hw hE2 hrd
    rw [uniformToList_carries]
    refine FEnv.back hw ws (List.range' base ws.length) r (by simp) ?_ ?_
    · intro p k sl hk hsl
      have hp : p < ws.length := by
        rcases Nat.lt_or_ge p ws.length with h | h
        · exact h
        · rw [List.getElem?_eq_none h] at hk; cases hk
      rw [List.getElem?_range' hp] at hsl
      simp only [Nat.one_mul, Option.some.injEq] at hsl
      subst hsl
      rw [hrd p hp]
      simp [vals, List.getElem?_map, hk]
    · intro k hk
      rw [hff j hj k hk]
      exact hE2.get (hF.get k)
  have hbackE : ∀ (j : Nat) (Γ2 : Env) (ps : List (Nat × Slot .f32)), j ≤ n → Extends Γ Γ2 →
      EnvI c (List.foldl (fun r x => Body.set r x.fst x.snd) r ps) (stF j).reg Γ2 := by
    intro j Γ2 ps hj hE2
    rw [hreg j hj]
    obtain ⟨h1, h2, h3⟩ := back_int r ps
    exact EnvI.ofI (hE.extend hE2) h1 h2 h3
  refine wpE_bind (wpE_wloopN Inv μ Hd X ?_ ?_ ?_ ?_ ?_ ?_)
  · refine ⟨_, ?_, 0, Nat.zero_le _, rfl, rfl, hF.w32⟩
    have hi0 : Γ0[Γ.size]? = some (V.sc .i32 (s32 (UInt32.ofNat 0))) := by rw [← hΓ0]; exact get_push_size
    simp only [Vals.slots, slots_ofFnL, ofFn_getD_mapL, List.mapM_cons]
    rw [hi0, lookups_mapL (hF.extend hx0)]
    rfl
  · intro wi cs hinv
    exact wpE_pure ⟨Extends.refl _, rfl, rfl, hinv⟩
  · rintro cs _ a Γ1 w1 ⟨rfl, rfl, j, hj, rfl, -, -⟩
    exact ⟨_, _, by simp only [contIfULt, contIf]; rw [hev j w1 _ hj]; rfl⟩
  · rintro cs _ a Γ1 w1 t f ⟨rfl, rfl, j, hj, rfl, rfl, hwj⟩ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j _ _ hj] at hev'
    have hjn : j = n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        omega
    subst hjn
    refine ⟨vals j, ?_, rfl, rfl, hwj⟩
    simp only [slots_carriesL]
    have hlen : (vals j).length = ws.length := by simp [vals]
    rw [← hlen]
    rw [mapM_range'L _ (vals j) (Γ0.size + 1) (fun p hp => by
      rw [show Γ0.size + 1 + p = Γ0.size + (p + 1) by omega, Array.getElem?_push,
        if_neg (by rw [bindAt_size]; simp; omega), bindAt_at]
      simp)]
  · rintro cs _ a Γ1 wb t f ⟨rfl, rfl, j, hj, rfl, rfl, hwj⟩ _ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j _ _ hj] at hev'
    have hjn : j < n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        exact hlt
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
    generalize hΓ1 : bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j) = Γ1
    have hE1 : Extends Γ Γ1 := by rw [← hΓ1]; exact hx0.trans ((Extends.refl _).bindAt_of (Nat.le_refl _) _)
    generalize hΓ2 : bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j) = Γ2
    have hE2 : Extends Γ Γ2 := by rw [← hΓ2]; exact (hE1.push_of _).trans ((Extends.refl _).bindAt_of (by simp) _)
    have hci : Γ2[Γ1.size + 1]? = some (V.sc .i32 (s32 (UInt32.ofNat j))) := by
      rw [← hΓ2]; exact bindAt_head _ _ _ _
    have hcv : ∀ p, Γ2[Γ1.size + 1 + 1 + p]? = (vals j)[p]? := by
      intro p; rw [← hΓ2, show Γ1.size + 1 + 1 + p = Γ1.size + 1 + (p + 1) by omega, bindAt_at]; simp
    simp only [carriesFrom, Vals.head, Vals.tail]
    have hSl : Slots { c with i := Γ1.size + 1 } Γ2 { x with i := j } :=
      ⟨hE2.get hS.lane, hS.lane32, hE2.get hS.cta, hS.cta32, fun b => hE2.get (hS.buf b), hE2.get hS.z64,
        hE2.get hS.sm, hci, hE2.get hS.zf, hE2.get hS.zi⟩
    refine wpE_bind (hrun j hjn _ _ Γ2 _ hSl (hbackE j Γ2 _ (by omega) hE2)
      (hbackF j Γ2 _ (by omega) hwj hE2 fun p _ => hcv p) fun r'' Γ'' hx'' hE'' hF'' => ?_)
    refine wpE_bind (wpE_iconst32u (by decide) ?_)
    refine wpE_bind (wpE_iadd32 (x := UInt32.ofNat j) (get_push (hx''.get hci)) (get_push_size) ?_)
    refine wpE_pure ⟨V.sc .i32 (s32 (UInt32.ofNat (j + 1))) :: vals (j + 1), ?_,
      ⟨j + 1, by omega, rfl, rfl, hF''.w32⟩, ?_⟩
    · simp only [Vals.slots, slots_ofFnL, ofFn_getD_mapL, List.mapM_cons, get_push_size]
      rw [lookups_mapL (hF''.extend ((Extends.push _ _).push_of _))]
      simp [vals, UInt32.ofNat_add]
    · show n - (s32 (UInt32.ofNat (j + 1))).toNat < n - (s32 (UInt32.ofNat j)).toNat
      rw [s32_ofNat_toNatL (by omega), s32_ofNat_toNatL (by omega)]
      omega
  · rintro Γb vs w' nE hEb hnE ⟨rfl, rfl, hwn⟩
    have hEb' : Extends Γ Γb := hx0.trans hEb
    have hle : Γ.size ≤ nE := by rw [hsz0] at hnE; omega
    have hE' : Extends Γ (bindAt Γb nE (vals n)) := hEb'.bindAt hle _
    refine wpE_pure (hQ _ _ hE' (hbackE n _ _ (Nat.le_refl _) hE') (hbackF n _ nE (Nat.le_refl _) hwn hE' ?_))
    intro p hp
    rw [bindAt_at]


/-- What a lane's run needs of its context: its collectives answered as a
    thread's are, by words, and its approximations the hardware's, 32 bits. -/
structure LaneOk (cfg : Cfg) (x : LCtx) : Prop where
  ans : AnswersFrom x.ans x.start cfg
  words : Words x.ans
  approx : ApproxFrom x.hw cfg
  w32 : ∀ k v, (x.hw.approx k v).toNat < 2 ^ 32

/-- **A statement's code does what the lane does**, and at the end of a loop's
    body goes round with the carries the lane ends with. -/
theorem code_lane : ∀ (s : EWStmt) (c : Body.Ctx Slot) (x : LCtx), LaneOk cfg x →
    CodeOk cfg c x s ∧ TailOk cfg c x s := by
  intro s
  induction s with
  | skip => intro c x _; exact ⟨code_skip, tail_of_code code_skip fun _ _ => rfl⟩
  | setI d e => intro c x _; exact ⟨code_setI d e, tail_of_code (code_setI d e) fun _ _ => rfl⟩
  | ldB d a => intro c x _; exact ⟨code_ldB d a, tail_of_code (code_ldB d a) fun _ _ => rfl⟩
  | ldH d a => intro c x _; exact ⟨code_ldH d a, tail_of_code (code_ldH d a) fun _ _ => rfl⟩
  | stB a v g => intro c x _; exact ⟨code_stB a v g, tail_of_code (code_stB a v g) fun _ _ => rfl⟩
  | stH a v g => intro c x _; exact ⟨code_stH a v g, tail_of_code (code_stH a v g) fun _ _ => rfl⟩
  | ballot d p =>
      intro c x hL
      exact ⟨code_ballot hL.ans hL.words d p, tail_of_code (code_ballot hL.ans hL.words d p) fun _ _ => rfl⟩
  | shflIdx d v sr =>
      intro c x hL
      exact ⟨code_shflIdx hL.ans hL.words d v sr, tail_of_code (code_shflIdx hL.ans hL.words d v sr) fun _ _ => rfl⟩
  | barrier => intro c x hL; exact ⟨code_barrier hL.ans, tail_of_code (code_barrier hL.ans) fun _ _ => rfl⟩
  | setR d e =>
      intro c x hL
      exact ⟨code_setR (fun v w => hL.approx .ex2Approx .f32 v w rfl) hL.w32 d e, tail_of_code (code_setR (fun v w => hL.approx .ex2Approx .f32 v w rfl) hL.w32 d e) fun _ _ => rfl⟩
  | shflXor d sr m =>
      intro c x hL
      exact ⟨code_shflXor hL.ans hL.words d sr m, tail_of_code (code_shflXor hL.ans hL.words d sr m) fun _ _ => rfl⟩
  | loadIdx d b ix => intro c x _; exact ⟨code_loadIdx d b ix, tail_of_code (code_loadIdx d b ix) fun _ _ => rfl⟩
  | loadV4 d0 d1 d2 d3 b ix =>
      intro c x _
      exact ⟨code_loadV4 d0 d1 d2 d3 b ix, tail_of_code (code_loadV4 d0 d1 d2 d3 b ix) fun _ _ => rfl⟩
  | storeLane b ix n =>
      intro c x _; exact ⟨code_storeLane b ix n, tail_of_code (code_storeLane b ix n) fun _ _ => rfl⟩
  | storeLane0 b ix n =>
      intro c x _; exact ⟨code_storeLane0 b ix n, tail_of_code (code_storeLane0 b ix n) fun _ _ => rfl⟩
  | stSm | ldSm | cvtIF =>
      intro c x _
      refine ⟨fun r Γ st st' Q _ _ _ _ hr => by simp [laneI] at hr, ?_⟩
      intro t r Γ st st' C _ _ _ _ _ _ _ _ hr
      simp [laneI] at hr
  | seq a b iha ihb =>
      intro c x hL
      have iha := iha c x hL
      have ihb := ihb c x hL
      refine ⟨CodeOk.seq iha.1 ihb.1, ?_⟩
      intro t r Γ st st' C hs hpost hwp hrf hwf hE hF hS hr hC
      have hw := wf_seq hwf
      by_cases hb : Body.straight b = true
      · simp only [Body.stmtT, hb, if_true]
        rw [laneI_seq_assoc] at hr
        refine iha.2 { t with post := .seq b t.post } r Γ st st' C (by simp [Body.straight, hb, hs])
          (CodeOk.seq ihb.1 hpost) (fun p hp => ?_) hrf hw.1 hE hF hS hr hC
        simp only [Body.assigns, List.mem_append] at hp
        exact hp.elim (hw.2 p) (hwp p)
      · simp only [Body.stmtT, hb, Bool.false_eq_true, if_false]
        rw [laneI_seq_assoc] at hr
        simp only [laneI, bind, Option.bind] at hr
        split at hr
        · cases hr
        rename_i s1 h1
        exact wpC_bind (iha.1 r Γ st s1 _ hw.1 hE hF hS h1 fun r1 Γ1 hx1 hE1 hF1 =>
          ihb.2 t r1 Γ1 s1 st' C hs hpost hwp hrf hw.2 hE1 hF1 (hS.extend hx1) (by simpa [laneI] using hr) hC)
  | locals rs b ih =>
      intro c x hL
      have ih := ih c x hL
      have hA' : CodeOk cfg c x (.locals rs b) := by
        intro r Γ st st' Q hwf hE hF hS hr hQ
        simp only [laneI] at hr
        obtain ⟨s1, h1, hr⟩ := Option.map_eq_some_iff.mp hr
        subst hr
        simp only [Body.stmt, Body.stmtT]
        exact wpE_bind (ih.1 r Γ st s1 _ (fun p hp => hwf p hp) hE hF hS h1 fun r1 Γ1 hx1 hE1 hF1 =>
          wpE_pure (hQ _ _ hx1 (EnvI.restore rs hE hE1 hx1) (FEnv.ofF hF1)))
      refine ⟨hA', ?_⟩
      intro t r Γ st st' C hs hpost hwp hrf hwf hE hF hS hr hC
      simp only [laneI, bind, Option.bind] at hr
      split at hr
      · cases hr
      rename_i s2 h2
      obtain ⟨s1, h1, hs2⟩ := Option.map_eq_some_iff.mp h2
      subst hs2
      simp only [Body.stmtT]
      exact wpC_bind (ih.1 r Γ st s1 _ (fun p hp => hwf p hp) hE hF hS h1 fun r1 Γ1 hx1 hE1 hF1 =>
        goRound_ok (st := restoreL rs st s1) hs hpost hwp hrf (EnvI.restore rs hE hE1 hx1) (FEnv.ofF hF1)
          (hS.extend hx1) hr hC)
  | iif cnd a b iha ihb =>
      intro c x hL
      have iha := iha c x hL
      have ihb := ihb c x hL
      have hwfs : ∀ {u v : EWStmt}, Wf c (.iif cnd u v) → Wf c u ∧ Wf c v := fun h =>
        ⟨fun p hp => h p (by simp [Body.assigns, hp]), fun p hp => h p (by simp [Body.assigns, hp])⟩
      have hA' : CodeOk cfg c x (.iif cnd a b) := by
        intro r Γ st st' Q hwf hE hF hS hr hQ
        have hwfab := hwfs hwf
        have hfr0 : ∀ n, n ∉ Body.vregs c (Body.setsI a ++ Body.setsI b).eraseDups →
            n ∉ Body.pregs c (Body.setsI a ++ Body.setsI b).eraseDups → st'.reg n = st.reg n :=
          fun n h1 h2 => laneI_frame _ x st st' hr n (not_split h1 h2)
        have hff0 : ∀ n, n ∉ (Body.written a ++ Body.written b).eraseDups → st'.f n = st.f n :=
          fun n h => laneI_fframe _ x st st' hr n h
        have hv0 : ∀ m ∈ Body.vregs c (Body.setsI a ++ Body.setsI b).eraseDups, c.fl.contains m = false :=
          fun m hm => vregs_mem hm
        have hf0 : ∀ m ∈ Body.pregs c (Body.setsI a ++ Body.setsI b).eraseDups, c.fl.contains m = true :=
          fun m hm => pregs_mem hm
        simp only [Body.stmt, Body.stmtT]
        generalize (Body.written a ++ Body.written b).eraseDups = ws at hff0 ⊢
        generalize Body.vregs c (Body.setsI a ++ Body.setsI b).eraseDups = rv at hfr0 hv0 ⊢
        generalize Body.pregs c (Body.setsI a ++ Body.setsI b).eraseDups = rf at hfr0 hf0 ⊢
        refine wpE_bind (wpE_condOf hE hS (Extends.refl _) cnd fun tt k xa xb X Y Γ1 hx1 hti ha hb hk => ?_)
        let Qj : Vals Slot (Body.carryTys ws rv rf) → Env → World → Prop :=
          fun jv Γ' w' => Extends Γ1 Γ' ∧ jv.slots.mapM (fun r => Γ'[r]?) = some (carried st' ws rv rf) ∧
            w' = st'.w ∧ (∀ n, c.fl.contains n = true → st'.reg n ≤ 1) ∧ ∀ n, (st'.f n).bits.toNat < 2 ^ 32
        have harm : ∀ (u : EWStmt) (Γa : Env), Extends Γ1 Γa → Wf c u →
            CodeOk cfg c x u → laneI x u st = some st' →
            wpE cfg (do Body.pack c (← Body.stmtT c none r u) ws rv rf) Qj Γa st.w := by
          intro u Γa hxa hwu hu hrun
          refine wpE_bind (hu r Γa st st' _ hwu (hE.extend (hx1.trans hxa)) (hF.extend (hx1.trans hxa))
            (hS.extend (hx1.trans hxa)) hrun fun r' Γ' hx' hE' hF' => ?_)
          exact wpE_pack hE' hF' hf0 fun pv Γ'' hx'' hpv => ⟨hxa.trans (hx'.trans hx''), hpv, rfl, hE'.bit, hF'.w32⟩
        refine wpE_bind (wpE_ifteT Qj hti ha hb ?_ ?_ ?_ ?_)
        · intro hc
          rw [hc] at hk
          have hne : cnd.evalL x st.reg ≠ 0 := by simpa using hk.symm
          simp only [laneI, if_pos hne] at hr
          exact harm a _ (Extends.push _ _) hwfab.1 iha.1 hr
        · intro hc ne hne
          rw [hc] at hk
          have he : cnd.evalL x st.reg = 0 := by simpa using hk.symm
          simp only [laneI, he, ne_eq, not_true_eq_false, if_false] at hr
          exact harm b _ ((Extends.push Γ1 _).bindAt_of hne []) hwfab.2 ihb.1 hr
        · exact fun jv Γ' w' hq => ⟨_, hq.2.1⟩
        · intro nJ Γ' jv vs w' hle hvs hq
          obtain ⟨hx', hm, rfl, hb, h32⟩ := hq
          rw [hm] at hvs; cases hvs
          have hU := unpack_ok hE hF (hx1.trans hx') (Nat.le_trans (hx1.trans hx').1 hle) hv0 hf0 hb hfr0 hff0 h32
          exact wpE_pure (hQ _ _ ((hx1.trans hx').bindAt_of hle _) hU.1 hU.2)
      refine ⟨hA', ?_⟩
      intro t r Γ st st' C hs hpost hwp hrf hwf hE hF hS hr hC
      have hwfab := hwfs hwf
      simp only [Body.stmtT]
      refine wpC_bind (wpE_condOf hE hS (Extends.refl _) cnd fun tt k xa xb X Y Γ1 hx1 hti ha hb hk => ?_)
      refine wpC_bind_ret (g := fun _ => r) (wpC_ifteT hti ha hb ?_ ?_)
      · intro hc
        rw [hc] at hk
        have hne : cnd.evalL x st.reg ≠ 0 := by simpa using hk.symm
        have hx : Extends Γ (Γ1.push (boolV true)) := hx1.trans (Extends.push _ _)
        have hr' : laneI x (.seq a t.post) st = some st' := by
          simp only [laneI, bind, Option.bind] at hr ⊢; rwa [if_pos hne] at hr
        exact wpC_bind_ret (iha.2 t r _ st st' C hs hpost hwp hrf hwfab.1 (hE.extend hx) (hF.extend hx)
          (hS.extend hx) hr' hC)
      · intro hc ne hne
        rw [hc] at hk
        have he : cnd.evalL x st.reg = 0 := by simpa using hk.symm
        have hx : Extends Γ (bindAt (Γ1.push (boolV false)) ne []) := hx1.trans ((Extends.push Γ1 _).bindAt_of hne [])
        have hr' : laneI x (.seq b t.post) st = some st' := by
          simp only [laneI, bind, Option.bind] at hr ⊢
          rwa [if_neg (by simpa using he)] at hr
        exact wpC_bind_ret (ihb.2 t r _ st st' C hs hpost hwp hrf hwfab.2 (hE.extend hx) (hF.extend hx)
          (hS.extend hx) hr' hC)
  | iwhile k cnd cs body ih =>
      intro c x hL
      have ih := ih c x hL
      have hA' : CodeOk cfg c x (.iwhile k cnd cs body) := by
        intro r Γ st st' Q hwf hE hF hS hr hQ
        have hwb : Wf c body := fun p hp => hwf p hp
        have hv0 : ∀ m ∈ Body.vregs c cs, c.fl.contains m = false := fun m hm => vregs_mem hm
        have hf0 : ∀ m ∈ Body.pregs c cs, c.fl.contains m = true := fun m hm => pregs_mem hm
        have hvc : ∀ m ∈ Body.vregs c cs, m ∈ cs := fun m hm => (List.mem_filter.mp hm).1
        have hfc : ∀ m ∈ Body.pregs c cs, m ∈ cs := fun m hm => (List.mem_filter.mp hm).1
        have hout0 : ∀ m, m ∉ Body.vregs c cs → m ∉ Body.pregs c cs → m ∉ cs := fun m h1 h2 => not_split h1 h2
        have hfb : ∀ (s1 s2 : LI), laneI x body s1 = some s2 → ∀ n, n ∉ Body.written body → s2.f n = s1.f n :=
          fun s1 s2 h n hn => laneI_fframe body x s1 s2 h n hn
        simp only [Body.stmt, Body.stmtT]
        generalize Body.written body = ws at hfb ⊢
        generalize Body.vregs c cs = rv at hv0 hvc hout0 ⊢
        generalize Body.pregs c cs = rf at hf0 hfc hout0 ⊢
        simp only [laneI] at hr
        let step : LI → Option LI := fun s' => (laneI x body s').map (keepL cs st)
        let go : LI → Bool := fun s' => cnd.evalL x s'.reg != 0
        change whileL step go k st = some st' at hr
        -- A lane state the loop reaches: its registers outside the carries
        -- the lane's before the loop, its predicate registers bits, its
        -- floats words.
        let Fr (s1 : LI) : Prop :=
          (∀ n, n ∉ rv → n ∉ rf → s1.reg n = st.reg n) ∧ (∀ n, c.fl.contains n = true → s1.reg n ≤ 1) ∧
          (∀ n, n ∉ ws → s1.f n = st.f n) ∧ ∀ n, (s1.f n).bits.toNat < 2 ^ 32
        let R (cs' : List V) (wi : World) (j : Nat) : Prop := ∃ s1 : LI, cs' = carried s1 ws rv rf ∧
          wi = s1.w ∧ Fr s1 ∧ whileL step go j s1 = some st'
        have huniq : ∀ s1 s2 : LI, carried s1 ws rv rf = carried s2 ws rv rf → s1.w = s2.w → Fr s1 → Fr s2 →
            s1 = s2 := by
          intro s1 s2 hc hw h1 h2
          obtain ⟨hf, hv, hp⟩ := carried_inj hc
          cases s1; cases s2
          simp only at hw; subst hw
          congr 1
          · funext n
            by_cases hrv : n ∈ rv
            · exact hv n hrv
            · by_cases hrf : n ∈ rf
              · exact hp n hrf
              · exact (h1.1 n hrv hrf).trans (h2.1 n hrv hrf).symm
          · funext n
            by_cases hws : n ∈ ws
            · exact hf n hws
            · exact (h1.2.2.1 n hws).trans (h2.2.2.1 n hws).symm
        refine wpE_bind (wpE_pack hE hF hf0 fun init Γp hxp hinit => ?_)
        have hxB : ∀ cs', Extends Γp (bindAt Γp Γp.size cs') := fun cs' =>
          (Extends.refl _).bindAt_of (Nat.le_refl _) cs'
        refine wpE_bind (wpE_wloopC (fun cs' wi => ∃ j, j ≤ k ∧ R cs' wi j) (fun cs' wi => least (R cs' wi) k)
          (fun cs' wi a Γ1 w1 => Extends (bindAt Γp Γp.size cs') Γ1 ∧ ∃ (tt : ClifTy) (kk : ICmpCond)
            (xa xb : Slot tt) (X Y : UInt64) (s1 : LI),
            a = (contIf kk xa xb, carriesFrom Γp.size (Body.carryTys ws rv rf), ()) ∧ tt.isInt = true ∧
            Γ1[xa]? = some (.sc tt X) ∧ Γ1[xb]? = some (.sc tt Y) ∧ cmpInt kk tt X Y = go s1 ∧ w1 = wi ∧
            cs' = carried s1 ws rv rf ∧ wi = s1.w ∧ Fr s1 ∧ ∃ j, j ≤ k ∧ whileL step go j s1 = some st')
          (fun vs w => vs = carried st' ws rv rf ∧ w = st'.w ∧ Fr st') ?_ ?_ ?_ ?_ ?_ ?_)
        · exact ⟨_, hinit, k, Nat.le_refl _, st, rfl, rfl, ⟨fun _ _ _ => rfl, hE.bit, fun _ _ => rfl, hF.w32⟩, hr⟩
        · rintro wi cs' ⟨j, hj, s1, rfl, rfl, hF1, hrun⟩
          have hU := unpack_ok (n := Γp.size) hE hF hxp hxp.1 hv0 hf0 hF1.2.1 hF1.1 hF1.2.2.1 hF1.2.2.2
          refine wpE_bind (wpE_condOf hU.1 (hS.extend (hxp.trans (hxB _))) (Extends.refl _) cnd
            fun tt kk xa xb X Y Γ1 hx1 hti ha hb hk => ?_)
          exact wpE_pure ⟨hx1, hx1, tt, kk, xa, xb, X, Y, s1, rfl, hti, ha, hb, hk, rfl, rfl, rfl, hF1, j, hj, hrun⟩
        · rintro cs' wi a Γ1 w1 ⟨-, tt, kk, xa, xb, X, Y, s1, rfl, hti, ha, hb, -⟩
          exact ⟨_, _, eval_icmp hti ha hb⟩
        · rintro cs' wi a Γ1 w1 t f ⟨hxe, tt, kk, xa, xb, X, Y, s1, rfl, hti, ha, hb, hk, rfl, rfl, rfl, hF1, j, hj,
            hrun⟩ hev hexit
          simp only [contIf] at hev hexit
          rw [eval_icmp hti ha hb] at hev
          have hgo : go s1 = false := by
            rw [← hk]
            cases hc : cmpInt kk tt X Y <;> simp [hc, boolV] at hev ⊢
            obtain ⟨-, rfl⟩ := hev; simp at hexit
          rw [whileL_stop hgo j] at hrun
          have hs1 : s1 = st' := Option.some.inj hrun
          subst hs1
          have hrd := carries_read (Γ := Γp) (n := Γp.size) (xs := carried s1 ws rv rf) (Body.carryTys ws rv rf) 0
            (by simp [carried_length])
          simp only [Nat.add_zero, List.drop_zero] at hrd
          exact ⟨_, Sem.mapM_ext (hxe.push_of _) hrd, rfl, rfl, hF1⟩
        · rintro lp cs' wi a Γ1 wb t f ⟨hxe, tt, kk, xa, xb, X, Y, s1, rfl, hti, ha, hb, hk, rfl, rfl, rfl, hF1, j,
            hj, hrun⟩ _ hev hcont
          simp only [contIf] at hev hcont
          rw [eval_icmp hti ha hb] at hev
          have hgo : go s1 = true := by
            rw [← hk]
            cases hc : cmpInt kk tt X Y <;> simp [hc, boolV] at hev ⊢
            obtain ⟨-, rfl⟩ := hev; simp at hcont
          obtain ⟨hR0, hmin⟩ := least_spec (P := R (carried s1 ws rv rf) s1.w) ⟨s1, rfl, rfl, hF1, hrun⟩ hj
          generalize hk0e : least (R (carried s1 ws rv rf) s1.w) k = k0 at hR0 hmin
          obtain ⟨s1', hcs, hw, hF', hrun0⟩ := hR0
          rw [huniq s1' s1 hcs.symm hw.symm hF' hF1] at hrun0
          have hk0 : k0 ≤ j := hmin j ⟨s1, rfl, rfl, hF1, hrun⟩
          cases k0 with
          | zero => simp [whileL, hgo] at hrun0
          | succ k' =>
          simp only [whileL, hgo, if_true] at hrun0
          obtain ⟨s2', hs2', hrun0⟩ := Option.bind_eq_some_iff.mp hrun0
          obtain ⟨s2, hb2, rfl⟩ := Option.map_eq_some_iff.mp hs2'
          have hxe' : Extends Γ (Γ1.push (.sc t f)) := (hxp.trans ((hxB _).trans hxe)).push_of _
          have hn' : Γ.size ≤ Γ1.size + 1 := by simpa using hxe'.1
          have hU := unpack_ok hE hF hxe' hn' hv0 hf0 hF1.2.1 hF1.1 hF1.2.2.1 hF1.2.2.2
          refine wpC_bind_ret (ih.2 ⟨ws, rv, rf, lp, .skip⟩ _ _ s1 s2 _ rfl code_skip
            (fun p hp => by simp [Body.assigns] at hp) hf0 hwb hU.1 hU.2 (hS.extend (hxe'.bindAt hn' _))
            (by simpa [laneI] using hb2) fun hbit h32 => ?_)
          have hF2 : Fr (keepL cs st s2) := by
            refine ⟨fun n h1 h2 => ?_, fun n hn => ?_, fun n hn => ?_, h32⟩
            · simp [keepL, hout0 n h1 h2]
            · simp only [keepL]
              split
              · exact hbit n hn
              · exact hE.bit n hn
            · exact (hfb s1 s2 hb2 n hn).trans (hF1.2.2.1 n hn)
          have hR2 : R (carried s2 ws rv rf) s2.w k' :=
            ⟨keepL cs st s2, (keepL_carried hvc hfc).symm, rfl, hF2, hrun0⟩
          refine ⟨carried_length _ _ _ _, ⟨k', by omega, hR2⟩, ?_⟩
          have := (least_spec hR2 (show k' ≤ k by omega)).2 k' hR2
          show least (R (carried s2 ws rv rf) s2.w) k < least (R (carried s1 ws rv rf) s1.w) k
          rw [hk0e]; omega
        · rintro Γb vs w' nE hxb hle ⟨rfl, rfl, hF1⟩
          have hU := unpack_ok hE hF (hxp.trans hxb) (Nat.le_trans hxp.1 hle) hv0 hf0 hF1.2.1 hF1.1 hF1.2.2.1
            hF1.2.2.2
          exact wpE_pure (hQ _ _ ((hxp.trans hxb).bindAt (Nat.le_trans hxp.1 hle) _) hU.1 hU.2)
      refine ⟨hA', tail_of_code hA' fun t r => ?_⟩
      simp only [Body.stmt, Body.stmtT, Prog.bind_assoc]
      rfl
  | forN k body ih =>
      intro c x hL
      have hA' : CodeOk cfg c x (.forN k body) := by
        intro r Γ st st' Q hwf hE hF hS hr hQ
        simp only [laneI] at hr
        split at hr
        · rename_i hk
          let f : LI → Nat → Option LI := fun s j => laneI { x with i := j } body s
          have hsucc := tripsL_succ (f := f) hr
          have hfrT : ∀ j, j ≤ k → ∀ m, m ∉ Body.setsI body → (tripsL f st j).reg m = st.reg m := by
            intro j hj m hm
            induction j with
            | zero => rfl
            | succ j ihj => rw [laneI_frame body _ _ _ (hsucc j (by omega)) m hm, ihj (by omega)]
          have hffT : ∀ j, j ≤ k → ∀ m, m ∉ Body.written body → (tripsL f st j).f m = st.f m := by
            intro j hj m hm
            induction j with
            | zero => rfl
            | succ j ihj => rw [laneI_fframe body _ _ _ (hsucc j (by omega)) m hm, ihj (by omega)]
          have hrun : ∀ j, j < k → ∀ (sl : Slot .i32) (r' : Body.Regs Slot) (Γ' : Env)
              (Q' : Body.Regs Slot → Env → World → Prop),
              Slots { c with i := sl } Γ' { x with i := j } → EnvI c r' (tripsL f st j).reg Γ' →
              FEnv c r' (tripsL f st j).f Γ' →
              (∀ r'' Γ'', Extends Γ' Γ'' → EnvI c r'' (tripsL f st (j + 1)).reg Γ'' →
                FEnv c r'' (tripsL f st (j + 1)).f Γ'' → Q' r'' Γ'' (tripsL f st (j + 1)).w) →
              wpE cfg (Body.stmtT { c with i := sl } none r' body) Q' Γ' (tripsL f st j).w := by
            intro j hj sl r' Γ' Q' hSl hE' hF' hQ'
            exact (ih { c with i := sl } { x with i := j } ⟨hL.ans, hL.words, hL.approx, hL.w32⟩).1 r' Γ'
              (tripsL f st j) (tripsL f st (j + 1)) Q' hwf (EnvI.fl hE') (FEnv.zf hF') hSl (hsucc j hj)
              fun r'' Γ'' hx hE'' hF'' => hQ' r'' Γ'' hx (EnvI.fl hE'') (FEnv.zf hF'')
          have hend := tripsL_end hr
          simp only [Body.stmt, Body.stmtT, Kernel.Block.c32]
          by_cases hs0 : Body.setsI (.forN k body) = []
          · rw [if_pos hs0]
            refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
            have hreg : ∀ j, j ≤ k → (tripsL f st j).reg = (tripsL f st 0).reg := by
              intro j hj
              funext m
              by_cases hk0 : k = 0
              · subst hk0; obtain rfl : j = 0 := by omega
                rfl
              · have hb : Body.setsI body = [] := by simpa [Body.setsI, hk0] using hs0
                exact hfrT j hj m (by simp [hb])
            refine wpE_loopL (Body.written body) _ (tripsL f st) (by simpa using get_push_size) hk
              (hS.extend (Extends.push _ _)) (hE.extend (Extends.push _ _)) (hF.extend (Extends.push _ _)) hreg
              (fun j hj m hm => hffT j hj m hm) hrun fun r' Γ' hx hE' hF' => ?_
            rw [hend] at hE' hF' ⊢
            exact hQ r' Γ' ((Extends.push _ _).trans hx) hE' hF'
          · rw [if_neg hs0]
            have hsb : Body.setsI (.forN k body) = Body.setsI body := by
              by_cases hk0 : k = 0
              · simp [Body.setsI, hk0] at hs0
              · simp [Body.setsI, hk0]
            refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
            refine wpE_loopXL (Body.written body) (Body.setsI body) _ (tripsL f st) (by simpa using get_push_size)
              hk (hS.extend (Extends.push _ _)) (hE.extend (Extends.push _ _)) (hF.extend (Extends.push _ _))
              (fun j hj m hm => hfrT j hj m hm) (fun j hj m hm => hffT j hj m hm) hrun fun r' Γ' hx hE' hF' => ?_
            rw [hend] at hE' hF' ⊢
            exact hQ r' Γ' ((Extends.push _ _).trans hx) hE' hF'
        · cases hr
      refine ⟨hA', tail_of_code hA' fun t r => ?_⟩
      simp only [Body.stmt, Body.stmtT]
      split <;> simp only [Prog.bind_assoc]
  | forM b a body ih =>
      intro c x hL
      have hA' : CodeOk cfg c x (.forM b a body) := by
        intro r Γ st st' Q hwf hE hF hS hr hQ
        simp only [laneI] at hr
        split at hr
        · rename_i ha
          obtain ⟨v, hv, hr⟩ := Option.bind_eq_some_iff.mp hr
          split at hr
          · rename_i hk
            let f : LI → Nat → Option LI := fun s j => laneI { x with i := j } body s
            have hsucc := tripsL_succ (f := f) hr
            have hfrT : ∀ j, j ≤ v.toNat → ∀ m, m ∉ Body.setsI body → (tripsL f st j).reg m = st.reg m := by
              intro j hj m hm
              induction j with
              | zero => rfl
              | succ j ihj => rw [laneI_frame body _ _ _ (hsucc j (by omega)) m hm, ihj (by omega)]
            have hffT : ∀ j, j ≤ v.toNat → ∀ m, m ∉ Body.written body → (tripsL f st j).f m = st.f m := by
              intro j hj m hm
              induction j with
              | zero => rfl
              | succ j ihj => rw [laneI_fframe body _ _ _ (hsucc j (by omega)) m hm, ihj (by omega)]
            have hrun : ∀ j, j < v.toNat → ∀ (sl : Slot .i32) (r' : Body.Regs Slot) (Γ' : Env)
                (Q' : Body.Regs Slot → Env → World → Prop),
                Slots { c with i := sl } Γ' { x with i := j } → EnvI c r' (tripsL f st j).reg Γ' →
                FEnv c r' (tripsL f st j).f Γ' →
                (∀ r'' Γ'', Extends Γ' Γ'' → EnvI c r'' (tripsL f st (j + 1)).reg Γ'' →
                  FEnv c r'' (tripsL f st (j + 1)).f Γ'' → Q' r'' Γ'' (tripsL f st (j + 1)).w) →
                wpE cfg (Body.stmtT { c with i := sl } none r' body) Q' Γ' (tripsL f st j).w := by
              intro j hj sl r' Γ' Q' hSl hE' hF' hQ'
              exact (ih { c with i := sl } { x with i := j } ⟨hL.ans, hL.words, hL.approx, hL.w32⟩).1 r' Γ'
                (tripsL f st j) (tripsL f st (j + 1)) Q' hwf (EnvI.fl hE') (FEnv.zf hF') hSl (hsucc j hj)
                fun r'' Γ'' hx hE'' hF'' => hQ' r'' Γ'' hx (EnvI.fl hE'') (FEnv.zf hF'')
            have hend := tripsL_end hr
            have hvlt : v < 2 ^ 32 := UInt64.lt_iff_toNat_lt.mpr (by simpa using load4_lt hv)
            simp only [Body.stmt, Body.stmtT, Kernel.Block.c32]
            refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
            refine wpE_bind (wpE_at4L (j := UInt32.ofNat a) (get_push (hS.buf b)) (by simpa using get_push_size)
              fun va Γa hEa hva => ?_)
            refine wpE_bind (wpE_loadI32L hva hv ?_)
            have hx : Extends Γ (Γa.push (.sc .i32 v)) := ((Extends.push _ _).trans hEa).push_of _
            have hnS : (Γa.push (V.sc .i32 v))[Γa.size]? = some (.sc .i32 (s32 (UInt32.ofNat v.toNat))) := by
              rw [show s32 (UInt32.ofNat v.toNat) = v from by
                apply UInt64.toNat_inj.mp
                have h' : v.toNat < 2 ^ 32 := lt_of_lt_of_eq (UInt64.lt_iff_toNat_lt.mp hvlt) rfl
                simp [s32, Nat.mod_eq_of_lt h']]
              exact get_push_size
            by_cases hs0 : Body.setsI body = []
            · rw [if_pos hs0]
              have hreg : ∀ j, j ≤ v.toNat → (tripsL f st j).reg = (tripsL f st 0).reg := fun j hj =>
                funext fun m => hfrT j hj m (by simp [hs0])
              refine wpE_loopL (Body.written body) _ (tripsL f st) hnS hk (hS.extend hx) (hE.extend hx) (hF.extend hx)
                hreg (fun j hj m hm => hffT j hj m hm) hrun fun r' Γ' hx' hE' hF' => ?_
              rw [hend] at hE' hF' ⊢
              exact hQ r' Γ' (hx.trans hx') hE' hF'
            · rw [if_neg hs0]
              refine wpE_loopXL (Body.written body) (Body.setsI body) _ (tripsL f st) hnS hk (hS.extend hx)
                (hE.extend hx) (hF.extend hx) (fun j hj m hm => hfrT j hj m hm) (fun j hj m hm => hffT j hj m hm)
                hrun fun r' Γ' hx' hE' hF' => ?_
              rw [hend] at hE' hF' ⊢
              exact hQ r' Γ' (hx.trans hx') hE' hF'
          · cases hr
        · cases hr
      refine ⟨hA', tail_of_code hA' fun t r => ?_⟩
      simp only [Body.stmt, Body.stmtT, Prog.bind_assoc]
      congr 1; funext; congr 1; funext; congr 1; funext; congr 1; funext
      split <;> simp only [Prog.bind_assoc]

/-! ## Predicate registers -/

/-- A pass that drops nothing leaves every register it keeps assigned only
    `0` or `1`. -/
theorem flagFix_fix (s : EWStmt) : ∀ (fl : List Nat), Body.flagStep s (Body.flagFix s fl) = Body.flagFix s fl := by
  intro fl
  induction fl using (measure List.length).wf.induction with
  | h fl ih =>
      rw [Body.flagFix]
      split
      · rename_i hlt; exact ih _ hlt
      · rename_i hnl
        simp only [Body.flagStep] at hnl ⊢
        exact List.filter_eq_self.mpr (List.length_filter_eq_length_iff.mp
          (Nat.le_antisymm (List.length_filter_le _ _) (Nat.not_lt.mp hnl)))

/-- **The lowering's predicate registers are assigned only bits.** -/
theorem flagRegs_wf (s : EWStmt) (hc : c.fl = Body.flagRegs s) : Wf c s := by
  intro p hp hf
  rw [hc] at hp hf
  have hfix := flagFix_fix s ((Body.assigns [] s).map (·.1) |>.eraseDups)
  have hm : p.1 ∈ Body.flagRegs s := by simpa using hf
  unfold Body.flagRegs at hm hp
  rw [← hfix] at hm
  simp only [Body.flagStep, List.mem_filter, List.all_eq_true] at hm
  have := hm.2 p hp
  simpa using this

/-! ## A thread's entry -/

theorem drop4_slots (nbuf : Nat) :
    (Vals.drop 4 (carriesFrom 0 (idParams ++ List.replicate nbuf ClifTy.i64))).uniformToList = List.range' 4 nbuf := by
  simp only [idParams, List.cons_append, List.nil_append, carriesFrom, Vals.drop, Vals.tail]
  exact uniformToList_carries nbuf 4

/-- What thread `t` of block `cta` sees, its buffers at `ps`, its collectives
    answered from `ans` after the `start`-th observation. -/
def threadCtx (cta t : Nat) (ps : List UInt64) (ans : List Answer) (start : Nat) (hw : Hw) : LCtx :=
  { lv := (UInt32.ofNat t).toUInt64, cv := (UInt32.ofNat cta).toUInt64, ptr := fun b => ps.getD b 0,
    sm := addrOf .arena 0, ans, start, i := 0, hw }

/-- A lane's state at a thread's start: registers zero, in world `w`. -/
def LI.init (w : World) : LI := ⟨fun _ => 0, w, fun _ => ⟨0⟩⟩

/-- A thread's locals give its lane what it needs. -/
theorem laneOk_thread (cta t : Nat) (ps : List UInt64) (ans : List Answer) (start : Nat) (hw : Hw)
    (hW : Words ans) (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) :
    LaneOk { env := [], locals := threadLocals ans start hw } (threadCtx cta t ps ans start hw) :=
  ⟨answersFrom_thread _ _ _, hW, approxFrom_thread _ _ _, hw32⟩

/-- **A statement's code from its context's slots ends as its lane does**:
    the lane and block in `lane` and `cta`, buffer `k` at slot `4 + k`, shared
    memory's address in `sm`, the loop's trip zero. -/
theorem wpE_runI {cfg : Cfg} (nbuf : Nat) (s : EWStmt) (x : LCtx) (hL : LaneOk cfg x) (hi0 : x.i = 0)
    (w : World) (li' : LI) (hrun : laneI x s (LI.init w) = some li') (Γa : Env) (lane cta : Slot .i32)
    (sm : Slot .i64) (h1 : Γa[lane]? = some (.sc .i32 x.lv)) (hl32 : x.lv < 2 ^ 32)
    (h0 : Γa[cta]? = some (.sc .i32 x.cv)) (hc32 : x.cv < 2 ^ 32)
    (hb : ∀ k, k < nbuf → Γa[4 + k]? = some (.sc .i64 (x.ptr k))) (hpz : ∀ k, nbuf ≤ k → x.ptr k = 0)
    (h3 : Γa[sm]? = some (.sc .i64 x.sm)) :
    wpE cfg (ML.Body.run s lane cta (List.range' 4 nbuf) sm) (fun _ _ u => u = li'.w) Γa w := by
  simp only [ML.Body.run, AlgorithmLib.Kernel.Block.c32]
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  refine wpE_bind (wpE_fconst32 ?_)
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  refine wpE_bind (wpE_iconst64 ?_)
  generalize hG : ((((Γa.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .f32 (0 &&& widthMask .f32))).push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .i64 (c64 0))) = G
  have hE : Extends Γa G := by rw [← hG]; exact (((Extends.push _ _).push_of _).push_of _).push_of _
  have hi : G[Γa.size]? = some (.sc .i32 (s32 (UInt32.ofNat x.i))) := by
    rw [← hG, hi0]; exact get_push (get_push (get_push (get_push_size)))
  have hzf : G[(Γa.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).size]? = some (.sc .f32 0) := by
    rw [← hG]; exact get_push (get_push (get_push_size))
  have hzi : G[((Γa.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .f32 (0 &&& widthMask .f32))).size]? = some (.sc .i32 0) := by
    rw [← hG]; exact get_push (get_push_size)
  have hz : G[(((Γa.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .f32 (0 &&& widthMask .f32))).push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).size]? =
      some (.sc .i64 0) := by rw [← hG]; exact get_push_size
  refine wpE_bind ((code_lane s _ x hL).1 {} G (LI.init w) li' _
    (flagRegs_wf s rfl) ?_ ?_ ?_ hrun fun r' Γ' _ _ _ => wpE_pure rfl)
  · exact ⟨fun n v h => by simp at h, fun n v h => by simp at h, fun n v h => by simp at h,
      fun _ _ => by show (0 : UInt64) ≤ 1; decide,
      fun _ _ _ => rfl, fun _ _ _ => rfl, fun n v h => by simp at h, fun n v h => by simp at h,
      fun n v h => by simp at h⟩
  · exact ⟨fun n => by simpa [Body.get, LI.init] using hzf, fun n => by simp [LI.init]⟩
  · refine ⟨hE.get h1, hl32, hE.get h0, hc32, fun b => ?_, hz, hE.get h3, hi, hzf, hzi⟩
    simp only [ML.Body.buf]
    by_cases hk : b < nbuf
    · rw [List.getD_eq_getElem?_getD, List.getElem?_range' hk]
      simpa [Nat.one_mul] using hE.get (hb b hk)
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by simp; omega)]
      simp only [Option.getD_none]
      rw [hpz b (by omega)]
      exact hz

theorem threadArgs_get {cta bd t : Nat} {ps : List UInt64} {k : Nat} (hk : k < ps.length) :
    (List.toArray (threadArgs cta bd (ps.map (.sc .i64)) t))[4 + k]? = some (.sc .i64 (ps.getD k 0)) := by
  rw [List.getElem?_toArray]
  simp only [threadArgs]
  rw [List.getElem?_append_right (by simp)]
  simp only [List.length_cons, List.length_nil, Nat.add_sub_cancel_left, List.getElem?_map,
    List.getElem?_eq_getElem hk, Option.map_some, List.getD_eq_getElem?_getD, Option.getD_some]

theorem threadCtx_lt (cta t : Nat) (ps : List UInt64) (ans : List Answer) (start : Nat) (hw : Hw) :
    (threadCtx cta t ps ans start hw).lv < 2 ^ 32 ∧ (threadCtx cta t ps ans start hw).cv < 2 ^ 32 := by
  have h1 := (UInt32.ofNat t).toNat_lt
  have h2 := (UInt32.ofNat cta).toNat_lt
  exact ⟨UInt64.lt_iff_toNat_lt.mpr (by simp only [threadCtx]; rw [UInt32.toNat_toUInt64]; exact h1),
    UInt64.lt_iff_toNat_lt.mpr (by simp only [threadCtx]; rw [UInt32.toNat_toUInt64]; exact h2)⟩

theorem threadCtx_pz {cta t : Nat} {ps : List UInt64} {ans : List Answer} {start : Nat} {hw : Hw} {k : Nat}
    (hk : ps.length ≤ k) : (threadCtx cta t ps ans start hw).ptr k = 0 := by
  simp only [threadCtx]
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hk, Option.getD_none]

/-- **A thread of a statement's kernel ends as its lane does**: thread `t` of
    block `cta` of a one-warp-a-block body over `nbuf` buffers at `ps`, its
    collectives answered from `ans` (words) and `2^x` the hardware's (32
    bits), ends with the world its lane's run ends with. -/
theorem wpE_bodyI {cfg : Cfg} (nbuf : Nat) (s : EWStmt) (cta t bd : Nat)
    (ps : List UInt64) (hps : ps.length = nbuf) (w : World) (ans : List Answer) (hw : Hw)
    (hL : LaneOk cfg (threadCtx cta t ps ans w.obs.length hw)) (li' : LI)
    (hrun : laneI (threadCtx cta t ps ans w.obs.length hw) s (LI.init w) = some li') :
    wpE cfg (fnBody (idParams ++ List.replicate nbuf .i64) (ML.Body.body nbuf s))
      (fun _ _ u => u = li'.w) (List.toArray (threadArgs cta bd (ps.map (.sc .i64)) t)) w := by
  rw [fnBody_of (by simp [idParams, HProg.ptrParams])]
  simp only [ML.Body.body]
  refine wpE_entryParams ?_
  rw [drop4_slots]
  simp only [idParams, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.drop, Vals.tail]
  rw [if_pos (Nat.le_refl 1)]
  refine wpE_bind (wpE_pure ?_)
  have hlt := threadCtx_lt cta t ps ans w.obs.length hw
  exact wpE_runI nbuf s _ hL rfl w li' hrun _ 1 0 3 (by simp [threadArgs, threadCtx]) hlt.1
    (by simp [threadArgs, threadCtx]) hlt.2 (fun k hk => threadArgs_get (by omega)) (fun k hk => threadCtx_pz (by omega))
    (by simp [threadArgs, threadCtx])

theorem and31 (t : Nat) (ht : t < 2 ^ 32) :
    UInt32.ofNat t &&& UInt32.ofNat (Int.ofNat 31).toNat = UInt32.ofNat (t % 32) := by
  apply UInt32.toNat_inj.mp
  rw [UInt32.toNat_and, toNat_ofNat32 ht, toNat_ofNat32 (by decide : (Int.ofNat 31).toNat < 2 ^ 32),
    toNat_ofNat32 (by omega : t % 32 < 2 ^ 32), show (Int.ofNat 31).toNat = 2 ^ 5 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod]

/-- **A thread of a statement's kernel ends as its lane does**, `W` warps a
    block: thread `t` of block `cta` runs as lane `t mod 32` of block
    `cta·W + t/32`. -/
theorem wpE_bodyW {cfg : Cfg} (nbuf : Nat) (s : EWStmt) (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31)
    (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32) (hv : cta * W + t / 32 < 2 ^ 32)
    (ps : List UInt64) (hps : ps.length = nbuf) (w : World) (ans : List Answer) (hw : Hw)
    (hL : LaneOk cfg (threadCtx (cta * W + t / 32) (t % 32) ps ans w.obs.length hw)) (li' : LI)
    (hrun : laneI (threadCtx (cta * W + t / 32) (t % 32) ps ans w.obs.length hw) s (LI.init w) = some li') :
    wpE cfg (fnBody (idParams ++ List.replicate nbuf .i64) (ML.Body.body nbuf s W))
      (fun _ _ u => u = li'.w) (List.toArray (threadArgs cta bd (ps.map (.sc .i64)) t)) w := by
  by_cases hW1 : W ≤ 1
  · have hW1' : W = 1 := by omega
    subst hW1'
    have ht32 : t < 32 := by omega
    have e : threadCtx (cta * 1 + t / 32) (t % 32) ps ans w.obs.length hw = threadCtx cta t ps ans w.obs.length hw := by
      rw [Nat.mod_eq_of_lt ht32, Nat.div_eq_of_lt ht32, Nat.mul_one, Nat.add_zero]
    rw [e] at hL hrun
    exact wpE_bodyI nbuf s cta t bd ps hps w ans hw hL li' hrun
  rw [fnBody_of (by simp [idParams, HProg.ptrParams])]
  simp only [ML.Body.body]
  refine wpE_entryParams ?_
  rw [drop4_slots]
  simp only [idParams, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.drop, Vals.tail]
  generalize hΓa : List.toArray (threadArgs cta bd (ps.map (.sc .i64)) t) = Γa
  have h0 : Γa[0]? = some (.sc .i32 (s32 (UInt32.ofNat cta))) := by rw [← hΓa]; simp [threadArgs]
  have h1 : Γa[1]? = some (.sc .i32 (s32 (UInt32.ofNat t))) := by rw [← hΓa]; simp [threadArgs]
  have h3 : Γa[3]? = some (.sc .i64 (addrOf .arena 0)) := by rw [← hΓa]; simp [threadArgs]
  have hb : ∀ k, k < nbuf → Γa[4 + k]? = some (.sc .i64 (ps.getD k 0)) := fun k hk => by
    rw [← hΓa]; exact threadArgs_get (by omega)
  have hlt := threadCtx_lt (cta * W + t / 32) (t % 32) ps ans w.obs.length hw
  rw [if_neg hW1]
  simp only [AlgorithmLib.Kernel.Block.c32]
  refine wpE_bind (wpE_iconst32u (k := Int.ofNat 31) (by decide) ?_)
  refine wpE_bind (wpE_band32 (x := UInt32.ofNat t) (get_push h1) get_push_size ?_)
  refine wpE_bind (wpE_ushrImm32 (by decide) (get_push (get_push h1)) ?_)
  refine wpE_bind (wpE_iconst32u (k := Int.ofNat W) (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
  refine wpE_bind (wpE_imul32 (x := UInt32.ofNat cta) (get_push (get_push (get_push (get_push (get_push h0)))))
    get_push_size ?_)
  refine wpE_bind (wpE_iadd32 get_push_size (get_push (get_push (get_push_size))) ?_)
  refine wpE_bind (wpE_pure ?_)
  refine wpE_runI nbuf s _ hL rfl w li' hrun _ _ _ _ ?_ hlt.1 ?_ hlt.2
    (fun k hk => get_push (get_push (get_push (get_push (get_push (get_push (get_push (hb k hk))))))))
    (fun k hk => threadCtx_pz (by omega))
    (get_push (get_push (get_push (get_push (get_push (get_push (get_push h3)))))))
  · show _ = some (V.sc .i32 (UInt32.ofNat (t % 32)).toUInt64)
    rw [← and31 t ht]; exact get_push (get_push (get_push (get_push (get_push get_push_size))))
  · have hc : UInt32.ofNat cta * UInt32.ofNat (Int.ofNat W).toNat + UInt32.ofNat t >>> UInt32.ofNat (Int.ofNat 5).toNat
        = UInt32.ofNat (cta * W + t / 32) := by
      have hcW : cta * W < 2 ^ 32 := by omega
      have hc32 : cta < 2 ^ 32 := by
        rcases Nat.eq_zero_or_pos cta with h | h
        · omega
        · exact Nat.lt_of_le_of_lt (Nat.le_mul_of_pos_right cta hW) hcW
      have hWn : (Int.ofNat W).toNat = W := rfl
      have h5 : (Int.ofNat 5).toNat = 5 := rfl
      apply UInt32.toNat_inj.mp
      rw [UInt32.toNat_add, UInt32.toNat_mul, UInt32.toNat_shiftRight, hWn, h5, toNat_ofNat32 hc32,
        toNat_ofNat32 (by omega : W < 2 ^ 32), toNat_ofNat32 ht, toNat_ofNat32 (by decide : 5 < 2 ^ 32),
        toNat_ofNat32 hv, Nat.mod_eq_of_lt hcW, Nat.shiftRight_eq_div_pow]
      exact Nat.mod_eq_of_lt (by omega)
    show _ = some (V.sc .i32 (UInt32.ofNat (cta * W + t / 32)).toUInt64)
    rw [← hc]; exact get_push_size

end AlgorithmLib.ML.BodyLane
