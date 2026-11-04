module
public import AlgorithmLib.Kernel.PtxIr
meta import AlgorithmLib.Kernel.PtxIr
public import AlgorithmLib.Kernel.Run
meta import AlgorithmLib.Kernel.Run
public import Std.Data.HashMap
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `PtxRun` — what a PTX entry point does, thread by thread

One thread runs its entry's body from the first line: each instruction on its
registers, a load or store on the launch's memory, a branch to its label. A
shuffle, a ballot or a barrier is a collective: it is recorded as a call, as
`Kernel.Run` reads one, and answered from the answers the thread is given,
so a launch of PTX threads is `Kernel.Run`'s, its answers settled by the
same laws (`launchOn`).

Memory is the launch's world: a global or generic address is one the world
decodes, a `.shared` address an offset into the block's shared memory (the
world's `arena`), and a parameter the value the launch passed for it. The
special registers `%tid.x`, `%ctaid.x` and `%ntid.x` are the thread's.
-/

namespace AlgorithmLib.PtxRun

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Kernel AlgorithmLib.PtxIr

def mask (w : Nat) (x : UInt64) : UInt64 :=
  if w ≥ 64 then x else x &&& ((1 <<< UInt64.ofNat w) - 1)

def toInt (t : Ty) (x : UInt64) : Int :=
  let w := t.width
  let m := mask w x
  if t.isSigned && m.toNat ≥ 2 ^ (w - 1) then (m.toNat : Int) - 2 ^ w else m.toNat

def ofIntW (w : Nat) (i : Int) : UInt64 := mask w (UInt64.ofInt i)

/-- A thread's data: its registers and its world. Where it is in its body is
    `go`'s alone. -/
structure TS where
  regs : Std.HashMap Reg UInt64
  w : World

/-- What an instruction leaves to do next: the next line, the line a branch
    goes to, or nothing, the thread having returned. -/
inductive Next where
  | step
  | jump (pc : Nat)
  | done

/-- What a thread is given: its entry, labels, special registers, parameter
    values, collective answers and the hardware. -/
structure Ctx where
  body : Array Line
  labels : Std.HashMap Lbl Nat
  special : List (String × UInt64)
  params : List (String × UInt64)
  syms : List (String × UInt64)
  ans : List Answer
  hw : Hw
  /-- A round of finding the answers: a barrier with none leaves shared
      memory alone. -/
  provisional : Bool := false
  /-- How many observations the world held when the thread began: its
      collectives are those after. -/
  start : Nat := 0

abbrev M := StateT TS (Except String)

/-- A register's value: a special one the thread's, any other what was last
    written to it, zero before. -/
def getReg (c : Ctx) (n : Reg) : M UInt64 := do
  match n with
  | .special s => pure ((c.special.lookup s).getD 0)
  | _ => pure (((← get).regs.get? n).getD 0)

def setReg (n : Reg) (v : UInt64) : M Unit :=
  modify fun s => { s with regs := s.regs.insert n v }

def opnd (c : Ctx) : Opnd → M UInt64
  | .reg n => getReg c n
  | .imm v => pure (UInt64.ofInt v)
  | .f32 b => pure (UInt64.ofNat b)
  | .f64 b => pure (UInt64.ofNat b)
  | .sym n => match c.syms.lookup n with
      | some v => pure v
      | none => throw s!"unknown symbol {n}"

/-- The world address an access through space `sp` at `a` reaches. -/
def place (sp : Space) (a : UInt64) : UInt64 :=
  match sp with
  | .shared => addrOf .arena (mask 32 a).toNat
  | _ => a

def load (c : Ctx) (sp : Space) (t : Ty) (ad : Addr) : M UInt64 := do
  if sp == .param then
    match ad.base with
    | .sym n => match c.params.lookup n with
        | some v => pure v
        | none => throw s!"unknown parameter {n}"
    | _ => throw "a parameter load through a register"
  else
    let base ← opnd c ad.base
    let a := UInt64.ofInt ((base.toNat : Int) + ad.off)
    match (← get).w.mem.load (place sp a) (t.width / 8) with
    | some v => pure v
    | none => throw s!"a load of {t.width / 8} bytes at {place sp a} reaches nothing"

def store (c : Ctx) (sp : Space) (t : Ty) (ad : Addr) (v : UInt64) : M Unit := do
  let base ← opnd c ad.base
  let a := place sp (UInt64.ofInt ((base.toNat : Int) + ad.off))
  let n := t.width / 8
  let v := mask t.width v
  let s ← get
  match s.w.mem.store a n v with
  | some m => set { s with w := { s.w with mem := m, obs := .store a n v :: s.w.obs } }
  | none => throw s!"a store of {n} bytes at {a} reaches nothing"

/-- A collective: recorded as the call `Kernel.Run` reads, and answered from
    the thread's answers. -/
def coll (c : Ctx) (i : Intr) (args : List V) : M (Option V) := do
  let s ← get
  -- the thread's collectives so far, as `Kernel.Run` counts them
  let k := (collCalls s.w c.start).length
  set { s with w := { s.w with obs := .call (.intr i) args :: s.w.obs } }
  if i == .barrier then
    match c.ans[k]? with
    | some (.smem b) => modify fun s => { s with w := { s.w with mem := { s.w.mem with arena := b } } }; pure none
    | _ => if c.provisional then pure none else throw "a barrier no law has answered with shared memory"
  else pure (some (valAt c.ans k))

def f32b (x : UInt64) : Float32 := f32 x

/-- A float result as the GPU leaves it: every NaN one NaN, `0x7fffffff`,
    whatever NaN produced it. -/
def canon (x : UInt64) : UInt64 := if isNaNBits ClifTy.f32 x then 0x7fffffff else x

/-- PTX's `min`/`max` on floats: a NaN operand gives the other. -/
def fext (wantMax : Bool) (x y : UInt64) : UInt64 :=
  if isNaNBits ClifTy.f32 x then y else if isNaNBits ClifTy.f32 y then x else pickExtreme wantMax ClifTy.f32 x y

/-- A two-operand operation on floats, at `f32`. -/
def binF (o : BinOp) (x y : UInt64) : Except String UInt64 :=
  match o with
  | .add => pure (canon (ofF32 (f32b x + f32b y)))
  | .sub => pure (canon (ofF32 (f32b x - f32b y)))
  | .mul => pure (canon (ofF32 (f32b x * f32b y)))
  | .div => pure (canon (ofF32 (f32b x / f32b y)))
  | .min => pure (canon (fext false x y))
  | .max => pure (canon (fext true x y))
  | .minNaN => pure (canon (pickExtreme false ClifTy.f32 x y))
  | .maxNaN => pure (canon (pickExtreme true ClifTy.f32 x y))
  | _ => .error s!"{o.name} on a float"

/-- A two-operand operation on integers or bits of type `t`. -/
def binI (o : BinOp) (t : Ty) (x y : UInt64) : Except String UInt64 :=
  let w := t.width
  match o with
  | .add => pure (mask w (x + y))
  | .sub => pure (mask w (x - y))
  | .mul | .mulLo => pure (mask w (x * y))
  | .mulWide => pure (mask (2 * w) (UInt64.ofInt (toInt t x * toInt t y)))
  | .mulHi => pure (mask w (UInt64.ofInt ((toInt t x * toInt t y) / 2 ^ w)))
  | .div => if mask w y == 0 then .error "division by zero" else
      pure (ofIntW w (if t.isSigned then (toInt t x).tdiv (toInt t y) else (toInt t x) / (toInt t y)))
  | .rem => if mask w y == 0 then .error "remainder by zero" else
      pure (ofIntW w (if t.isSigned then (toInt t x).tmod (toInt t y) else (toInt t x) % (toInt t y)))
  | .min => pure (if toInt t x ≤ toInt t y then mask w x else mask w y)
  | .max => pure (if toInt t x ≥ toInt t y then mask w x else mask w y)
  | .and => pure (mask w (x &&& y))
  | .or => pure (mask w (x ||| y))
  | .xor => pure (mask w (x ^^^ y))
  -- PTX clamps a shift amount at the width
  | .shl => pure (if (mask 32 y).toNat ≥ w then 0 else mask w (x <<< y))
  | .shr =>
      if t.isSigned then
        let n := min (mask 32 y).toNat (w - 1)
        pure (ofIntW w ((toInt t x) >>> n))
      else pure (if (mask 32 y).toNat ≥ w then 0 else mask w x >>> y)
  | _ => .error s!"{o.name} on an integer"

/-- A two-operand operation at type `t`; a float operand is read as its
    register's 32 bits. -/
def bin (o : BinOp) (t : Ty) (x y : UInt64) : Except String UInt64 :=
  if t.isFloat then
    if t != .f32 then .error "a float operation not at f32" else binF o (mask 32 x) (mask 32 y)
  else binI o t x y

def un (o : UnOp) (hw : Hw) (t : Ty) (x : UInt64) : Except String UInt64 :=
  let w := t.width
  match o with
  | .neg => if t == .f32 then pure (canon (mask 32 (x ^^^ 0x80000000))) else pure (mask w (0 - x))
  | .abs => if t == .f32 then pure (x &&& 0x7fffffff) else pure (ofIntW w (toInt t x).natAbs)
  | .not => pure (mask w (~~~x))
  | .brev => pure (UInt64.ofNat ((List.range w).foldl
      (fun acc i => if x.toNat.testBit i then acc ||| 2 ^ (w - 1 - i) else acc) 0))
  | .clz => pure (UInt64.ofNat (match (List.range w).reverse.find? (fun i => x.toNat.testBit i) with
      | some hi => w - 1 - hi
      | none => w))
  | .popc => pure (UInt64.ofNat ((List.range w).filter (fun i => x.toNat.testBit i)).length)
  | .sqrt => if t == .f32 then pure (canon (ofF32 (f32b x).sqrt)) else .error "sqrt not at f32"
  | .ex2Approx => pure (hw.approx .ex2Approx x)
  | .lg2Approx => pure (hw.approx .lg2Approx x)
  | .rcpApprox => pure (hw.approx .rcpApprox x)
  | .sinApprox => pure (hw.approx .sinApprox x)
  | .cosApprox => pure (hw.approx .cosApprox x)
  | .sqrtApprox => pure (hw.approx .sqrtApprox x)
  | .rsqrtApprox => pure (hw.approx .rsqrtApprox x)

def cmp (c : Cmp) (t : Ty) (x y : UInt64) : Except String Bool :=
  if t.isFloat then
    -- IEEE 754's order on the bits (`fkey`), a NaN unordered
    let a := fkey ClifTy.f32 x; let b := fkey ClifTy.f32 y
    let un := isNaNBits ClifTy.f32 x || isNaNBits ClifTy.f32 y
    match c with
    | .eq => pure (!un && a == b) | .ne => pure (!un && a != b)
    | .lt => pure (!un && decide (a < b)) | .le => pure (!un && decide (a ≤ b))
    | .gt => pure (!un && decide (b < a)) | .ge => pure (!un && decide (b ≤ a))
    | .equ => pure (un || a == b) | .neu => pure (un || a != b)
    | .ltu => pure (un || decide (a < b)) | .leu => pure (un || decide (a ≤ b))
    | .gtu => pure (un || decide (b < a)) | .geu => pure (un || decide (b ≤ a))
    | .nan => pure un | .num => pure !un
    | _ => .error s!"{c.name} on a float"
  else
    let a := toInt t x; let b := toInt t y
    let ua := (mask t.width x).toNat; let ub := (mask t.width y).toNat
    match c with
    | .eq => pure (a == b) | .ne => pure (a != b)
    | .lt => pure (a < b) | .le => pure (a ≤ b) | .gt => pure (a > b) | .ge => pure (a ≥ b)
    | .lo => pure (ua < ub) | .ls => pure (ua ≤ ub) | .hi => pure (ua > ub) | .hs => pure (ua ≥ ub)
    | _ => .error s!"{c.name} on an integer"

def cvt (r : Option Rnd) (dt st : Ty) (x : UInt64) : Except String UInt64 :=
  if !dt.isFloat && !st.isFloat then
    -- an integer conversion: sign- or zero-extend from the source, then wrap
    pure (ofIntW dt.width (toInt st x))
  else if dt == .f32 && !st.isFloat then
    pure (ofF32 (Float32.ofInt (toInt st x)))
  else if !dt.isFloat && st == .f32 && r == some .rzi && dt.isSigned && (dt.width == 32 || dt.width == 64) then
    -- truncation to a signed word is the block semantics' saturating
    -- conversion: NaN to zero, out of range to the nearer end
    match satToSint (if dt.width == 64 then .i64 else .i32) (f32b x).toFloat with
    | .sc _ v => pure v
    | .vec .. => .error "cvt: a vector"
  else if !dt.isFloat && st == .f32 then
    let v := (f32b x).toFloat
    let i : Float := match r with
      | some .rni => (Float.round v)
      | some .rzi => (if v < 0 then v.ceil else v.floor)
      | _ => v
    if v != v then pure 0
    else if dt.isSigned then
      let lo : Int := -(2 ^ (dt.width - 1)); let hi : Int := 2 ^ (dt.width - 1) - 1
      let n : Int := if i ≥ 0 then (i.toUInt64.toNat : Int) else -(((-i).toUInt64.toNat : Nat) : Int)
      pure (ofIntW dt.width (max lo (min hi n)))
    else
      let hi : Nat := 2 ^ dt.width - 1
      pure (UInt64.ofNat (if i ≤ 0 then 0 else min hi i.toUInt64.toNat))
  else if dt == .f32 && st == .f64 then pure (canon (ofF32 (Float.ofBits x).toFloat32))
  else if dt == st then pure x
  else .error s!"cvt.{dt.name}.{st.name} not modelled"

def guardOk (c : Ctx) : Option (Bool × Reg) → M Bool
  | none => pure true
  | some (neg, p) => do
      let v ← getReg c p
      pure (if neg then v &&& 1 == 0 else v &&& 1 == 1)

def lift {α} (e : Except String α) : M α := match e with
  | .ok a => pure a
  | .error m => throw m

/-- One instruction, and what it leaves to do next. -/
def exec (c : Ctx) : Instr → M Next
  | .mov t d a => do setReg d (mask t.width (← opnd c a)); pure .step
  | .bin o _ t d a b => do setReg d (← lift (bin o t (← opnd c a) (← opnd c b))); pure .step
  | .un o _ t d a => do setReg d (← lift (un o c.hw t (← opnd c a))); pure .step
  | .fma (some _) t d a b e => do
      if t != .f32 then throw "fma not at f32"
      setReg d (canon (fmaLane ClifTy.f32 (mask 32 (← opnd c a)) (mask 32 (← opnd c b)) (mask 32 (← opnd c e))))
      pure .step
  | .fma none t d a b e => do
      setReg d (mask t.width ((← opnd c a) * (← opnd c b) + (← opnd c e))); pure .step
  | .cvt r dt sty d a => do setReg d (← lift (cvt r dt sty (← opnd c a))); pure .step
  | .cvta to sp _ d a => do
      let v ← opnd c a
      -- a shared window offset and the generic address of the same byte
      if sp != .shared then setReg d v
      else if to then
        match decodeAddr v with
        | some (.arena, off) => setReg d (UInt64.ofNat off)
        | _ => throw "cvta.to.shared of an address outside shared memory"
      else setReg d (addrOf .arena (mask 32 v).toNat)
      pure .step
  | .setp cc t p a b => do
      setReg p (if ← lift (cmp cc t (← opnd c a) (← opnd c b)) then 1 else 0); pure .step
  | .selp t d a b p => do
      let pv ← opnd c p
      let x ← opnd c a; let y ← opnd c b
      setReg d (mask t.width (if pv &&& 1 == 1 then x else y)); pure .step
  | .prmt d a b s => do
      let x ← opnd c a; let y ← opnd c b; let sel ← opnd c s
      let bytes := (List.range 4).map (fun i => (x >>> UInt64.ofNat (8 * i)) &&& 0xff) ++
        (List.range 4).map (fun i => (y >>> UInt64.ofNat (8 * i)) &&& 0xff)
      let r := (List.range 4).foldl (fun acc i =>
        let k := ((sel >>> UInt64.ofNat (4 * i)) &&& 7).toNat
        acc ||| (bytes.getD k 0 <<< UInt64.ofNat (8 * i))) 0
      setReg d r; pure .step
  | .ld sp _ t d a => do setReg d (← load c sp t a); pure .step
  | .st sp t a v => do store c sp t a (← opnd c v); pure .step
  | .shfl m d a b cl _ => do
      if (← opnd c cl) != 31 then throw "a shuffle within less than a warp"
      let i ← match m with
        | .bfly => pure Intr.shflBfly
        | .idx => pure .shflIdx
        | _ => throw s!"shfl.{m.name} not modelled"
      let r ← coll c i [.sc .i32 (mask 32 (← opnd c a)), .sc .i32 (mask 32 (← opnd c b))]
      match r with
      | some (.sc _ v) => setReg d (mask 32 v); pure .step
      | _ => throw "a shuffle answered by no value"
  | .vote d p _ => do
      let r ← coll c .ballot [.sc .i32 ((← opnd c p) &&& 1)]
      match r with
      | some (.sc _ v) => setReg d (mask 32 v); pure .step
      | _ => throw "a ballot answered by no value"
  | .bar _ => do let _ ← coll c .barrier []; pure .step
  | .barWarp _ =>
      -- the warp's barrier is the block's where the block is one warp
      if (c.special.lookup "ntid.x").getD 0 ≤ 32 then do let _ ← coll c .barrier []; pure .step
      else throw "a warp barrier in a block of more than one warp"
  | .bra _ l => do
      match c.labels.get? l with
      | some i => pure (.jump i)
      | none => throw s!"no label {l.text}"
  | .ret => pure .done

/-- Run from line `pc` with `fuel` steps. -/
def go (c : Ctx) : Nat → Nat → M Unit
  | 0, _ => throw "out of steps"
  | fuel + 1, pc => do
      match c.body[pc]? with
      | none => pure ()
      | some (.label _) => go c fuel (pc + 1)
      | some (.ins g i) =>
          if ← guardOk c g then
            match ← exec c i with
            | .step => go c fuel (pc + 1)
            | .jump j => go c fuel j
            | .done => pure ()
          else go c fuel (pc + 1)

/-- Each label's line: where it is written, the last place if more than one. -/
def labelsOf (body : List Line) : Std.HashMap Lbl Nat :=
  body.zipIdx.foldl (fun m (l, i) => match l with
    | .label n => m.insert n i
    | _ => m) {}

/-- The context thread `args` of entry `e` runs in: its body and labels, its
    identity and parameters from `args`, the shared array at offset zero. -/
def Ctx.ofThread (e : Entry) (ans : List Answer) (args : List V) (w : World) (hw : Hw)
    (provisional : Bool) : Ctx :=
  let bits : V → UInt64 := fun v => (asBits v).getD 0
  { body := e.body.toArray, labels := labelsOf e.body,
    special := [("ctaid.x", bits (args.getD 0 default)), ("tid.x", bits (args.getD 1 default)),
      ("ntid.x", bits (args.getD 2 default))],
    params := e.params.zipIdx.map fun ((_, n), i) => (n, bits (args.getD (4 + i) default)),
    syms := (match e.shared with | some s => [(s.sym, 0)] | none => []) ++ [("smem", 0), ("_smem", 0)],
    ans, hw, provisional, start := w.obs.length }

/-- **One thread** of entry `e`: its identity and parameter values from
    `args` (block, thread, block size, shared memory's address, then the
    entry's parameters in order), its collectives answered from `ans`. -/
def thread (e : Entry) (ans : List Answer) (args : List V) (w : World) (hw : Hw := Hw.rounded)
    (steps : Nat := 100000000) (provisional := false) : Except String World := do
  let (_, s) ← (go (Ctx.ofThread e ans args w hw provisional) steps 0).run { regs := {}, w }
  pure s.w

/-! ## A launch

`Kernel.Run`'s launch, each thread entry `e`'s. -/

/-- Entry `e`'s threads with `steps` of fuel. -/
def runner (e : Entry) (hw : Hw) (steps : Nat) : Kernel.Runner :=
  fun prov ans args w => thread e ans args w hw steps prov

/-- **A launch** of entry `e`: `grid` blocks of `blockDim` threads, each
    `thread`, on device memory `bufs`. -/
def launchOn (e : Entry) (smem grid blockDim : Nat) (params : List V) (bufs : Nat → ByteArray)
    (hw : Hw := Hw.rounded) : Except String (Nat → ByteArray) :=
  Kernel.launchOn (runner e hw 100000000) smem grid blockDim params bufs hw.nan

/-- A word cut to a width is below `2` to that width. -/
theorem mask_lt (t : Ty) (x : UInt64) : (mask t.width x).toNat < 2 ^ t.width := by
  cases t <;> simp only [Ty.width]
  all_goals first
    | (simp only [mask, show (64 : Nat) ≥ 64 from by decide, if_true]; exact x.toNat_lt)
    | (simp only [mask]
       rw [if_neg (by decide), UInt64.toNat_and]
       first
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 1 - 1).toNat = 2 ^ 1 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 8 - 1).toNat = 2 ^ 8 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 16 - 1).toNat = 2 ^ 16 - 1 from rfl]
       | rw [show ((1 : UInt64) <<< UInt64.ofNat 32 - 1).toNat = 2 ^ 32 - 1 from rfl]
       rw [Nat.and_two_pow_sub_one_eq_mod]
       exact Nat.mod_lt _ (by decide))

theorem mask32_toNat_lt (x : UInt64) : (mask 32 x).toNat < 2 ^ 32 := mask_lt .b32 x

end AlgorithmLib.PtxRun
