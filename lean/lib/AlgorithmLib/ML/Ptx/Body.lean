module
public import AlgorithmLib.ML.Machine.WarpEmit
meta import AlgorithmLib.ML.Machine.WarpEmit
public import AlgorithmLib.Kernel.Ptx
meta import AlgorithmLib.Kernel.Ptx
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.ML.Num.Bits
meta import AlgorithmLib.ML.Num.Bits
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.Body` — warp statements as kernel bodies

`body nbuf s` is the body a thread of a one-warp block runs for the warp
statement `s` (`EWStmt`): lane is the thread, the block is the CTA, buffer
`b` is the `b`-th pointer parameter, an index counts 32-bit elements. A float
register is a value the statements thread through; a loop carries the
registers its body writes.

Each operation computes what it means at the carrier `F32` (`F32.ops`):
`inv` is `1 / x` and `rsqrt` `1 / sqrt x`, both exactly rounded; `ex2` is the
hardware's approximation and `exp` is `ex2 (x · log₂ e)`; `geF` is `1.0` where
`a ≥ b` and `0.0` otherwise; a literal is its word.
`maxW` is `x` where `x ≥ y` and `y` otherwise, as `NumOps.max` defines it.
-/

namespace AlgorithmLib.ML.Body

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.ML AlgorithmLib.Kernel AlgorithmLib.HProg.Sem

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def shfl : LocalRef [.i32, .i32] (some .i32) := { callee := .intr .shflBfly }
def bar : LocalRef [] none := { callee := .intr .barrier }
def ex2 : LocalRef [.f32] (some .f32) := { callee := .intr .ex2Approx }

/-- What a statement sees: the thread's lane and block, the buffers'
    addresses, shared memory's, the innermost loop's counter, and zeros. -/
structure Ctx (V : ClifTy → Type) where
  lane : V .i32
  cta : V .i32
  bufs : List (V .i64)
  sm : V .i64
  i : V .i32
  zf : V .f32
  zi : V .i32
  z64 : V .i64

/-- The float registers, by number; one never written reads zero. -/
abbrev Regs (V : ClifTy → Type) := List (Nat × V .f32)

def get (c : Ctx V) (r : Regs V) (n : Nat) : V .f32 := (r.lookup n).getD c.zf
def set (r : Regs V) (n : Nat) (v : V .f32) : Regs V := (n, v) :: r.filter (·.1 != n)

def c32 (k : Nat) : Prog V L (V .i32) := iconst32 (Int.ofNat k)

/-- Element `ix` of the buffer at `base`. -/
def at4 (base : V .i64) (ix : V .i32) : Prog V L (V .i64) := do
  iadd base (← ishlImm (← uextend64 ix) 2)

def buf (c : Ctx V) (b : Nat) : V .i64 := c.bufs.getD b c.z64

def idx (c : Ctx V) : IdxE → Prog V L (V .i32)
  | .laneId => pure c.lane
  | .loopI => pure c.i
  | .ctaId => pure c.cta
  | .lit n => c32 n
  | .ireg _ => pure c.zi
  | .ldIdx b off => do load32 (← at4 (buf c b) (← idx c off))
  | .add a b => do iadd (← idx c a) (← idx c b)
  | .mul a b => do imul (← idx c a) (← idx c b)

def fexp (c : Ctx V) (r : Regs V) : WFExp → Prog V L (V .f32)
  | .reg n => pure (get c r n)
  | .lit v => fconst .f32 (ofF32 v)
  | .add a b => do fadd (← fexp c r a) (← fexp c r b)
  | .mul a b => do fmul (← fexp c r a) (← fexp c r b)
  | .neg a => do fneg (← fexp c r a)
  | .inv a => do
      let x ← fexp c r a
      fdiv (← fconst .f32 oneBits) x
  | .rsqrt a => do
      let s ← sqrt (← fexp c r a)
      fdiv (← fconst .f32 oneBits) s
  | .ex2 a => do callLocal ex2 %[← fexp c r a]
  | .exp a => do
      let x ← fexp c r a
      callLocal ex2 %[← fmul x (← fconst .f32 log2eBits)]
  | .maxW a b => do
      let x ← fexp c r a
      let y ← fexp c r b
      select (← fcmp .ge x y) x y
  | .geF a b => do
      let x ← fexp c r a
      let y ← fexp c r b
      select (← fcmp .ge x y) (← fconst .f32 (ofF32 1.0)) (← fconst .f32 (ofF32 0.0))

/-- The float registers a statement may write. -/
def written : EWStmt → List Nat
  | .setR r _ | .shflXor r _ _ | .loadIdx r _ _ | .ldSm r _ | .cvtIF r _ => [r]
  | .loadV4 a b c d _ _ => [a, b, c, d]
  | .seq a b => (written a ++ written b).eraseDups
  | .forN _ b | .forM _ _ b => written b
  | _ => []

/-- `n` trips of `run`, carrying registers `ws`. -/
def loop (c : Ctx V) (r : Regs V) (n : V .i32) (ws : List Nat)
    (run : Ctx V → Regs V → Prog V L (Regs V)) : Prog V L (Regs V) := do
  let pick (r : Regs V) : Vals V (List.replicate ws.length .f32) :=
    Vals.ofFn fun j => get c r (ws.getD j.val 0)
  let back (r : Regs V) (vs : Vals V (List.replicate ws.length .f32)) : Regs V :=
    (ws.zip (Vals.uniformToList vs)).foldl (fun r (k, v) => set r k v) r
  let i0 ← c32 0
  let res ← wloop (tys := .i32 :: List.replicate ws.length .f32) (.cons i0 (pick r))
    (fun cs => pure (contIfULt cs.head n, cs.tail, ()))
    (fun cs _ => do
      let r' ← run { c with i := cs.head } (back r cs.tail)
      pure (.cons (← iadd cs.head (← c32 1)) (pick r')))
  pure (back r res)

def stmt (c : Ctx V) : Regs V → EWStmt → Prog V L (Regs V)
  | r, .skip => pure r
  | r, .seq a b => do stmt c (← stmt c r a) b
  | r, .setR n e => do pure (set r n (← fexp c r e))
  | r, .shflXor d s m => do
      let o ← callLocal shfl %[← bitcast .i32 (get c r s), ← c32 m]
      pure (set r d (← bitcast .f32 o))
  | r, .loadIdx d b ix => do pure (set r d (← loadF32 (← at4 (buf c b) (← idx c ix))))
  | r, .loadV4 d0 d1 d2 d3 b ix => do
      let a ← at4 (buf c b) (← idx c ix)
      let v0 ← loadF32 a
      let v1 ← loadF32 (← iaddImm a 4)
      let v2 ← loadF32 (← iaddImm a 8)
      let v3 ← loadF32 (← iaddImm a 12)
      pure (set (set (set (set r d0 v0) d1 v1) d2 v2) d3 v3)
  | r, .storeLane0 b ix n => do
      when .eq c.lane (← c32 0) do store (get c r n) (← at4 (buf c b) (← idx c ix))
      pure r
  | r, .storeLane b ix n => do
      store (get c r n) (← at4 (buf c b) (← idx c ix))
      pure r
  | r, .stSm ix n => do
      store (get c r n) (← at4 c.sm (← idx c ix))
      pure r
  | r, .ldSm d ix => do pure (set r d (← loadF32 (← at4 c.sm (← idx c ix))))
  | r, .barrier => do
      callLocalVoid bar %[]
      pure r
  | r, .forN n body => do loop c r (← c32 n) (written body) fun c' r' => stmt c' r' body
  | r, .forM b a body => do
      let n ← load32 (← at4 (buf c b) (← c32 a))
      loop c r n (written body) fun c' r' => stmt c' r' body
  | r, .cvtIF d ix => do pure (set r d (← fcvtFromUint .f32 (← idx c ix)))

/-- A warp statement run as lane `lane` of block `cta`, over buffers `bufs`
    and shared memory `sm`. -/
def run (s : EWStmt) (lane cta : V .i32) (bufs : List (V .i64)) (sm : V .i64) : Prog V L Unit := do
  let c : Ctx V := { lane, cta, bufs, sm, i := ← c32 0, zf := ← fconst .f32 0, zi := ← c32 0, z64 := ← iconst64 0 }
  let _ ← stmt c [] s
  pure ()

/-- **A warp statement's body**, over `nbuf` buffers, for blocks of `warps`
    warps that do not meet: warp `w` of block `b` runs the statement as block
    `b·warps + w`, its lanes as the warp's. -/
def body (nbuf : Nat) (s : EWStmt) (warps : Nat := 1) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate nbuf ClifTy.i64)
  let bufs := Vals.uniformToList (ps.drop 4 : Vals V (List.replicate nbuf .i64))
  let (lane, cta) ← if warps ≤ 1 then pure (ps.snd, ps.head) else do
    let lane ← band ps.snd (← c32 31)
    let w ← ushrImm ps.snd 5
    pure (lane, ← iadd (← imul ps.head (← c32 warps)) w)
  run s lane cta bufs (ps.drop 3).head
end

/-- The kernel: `nbuf` pointer parameters, `smem` bytes of shared memory, and
    `warps` warps a block. -/
def kernel (name : String) (nbuf smem : Nat) (s : EWStmt) (warps : Nat := 1) : Kernel :=
  { name, params := List.replicate nbuf .i64,
    code := emit (body nbuf s warps) (idParams ++ List.replicate nbuf .i64), smem }

/-- **A statement's kernel as PTX text**: a module of its own, or the printer's
    error as a comment, which no driver loads. -/
def text (name : String) (nbuf smem : Nat) (s : EWStmt) (warps : Nat := 1) : String :=
  match (kernel name nbuf smem s warps).ptx with
  | .ok t => t
  | .error e => s!"// {e}"

end AlgorithmLib.ML.Body
