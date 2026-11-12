module
public import AlgorithmLib.Kernel.Ptx
meta import AlgorithmLib.Kernel.Ptx
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Block` — what the threads of a block do together

The pieces a kernel body (`Kernel.Ptx`) is built from when a block's threads
cooperate: the intrinsics as calls, element addressing, loops over a strided
range, and a reduction across the warp (a butterfly) and across the block's
warps (through shared memory, behind one barrier).

A block reduction gives every thread the same value, folded in a fixed order:
each warp's butterfly, then the warps' results in warp order. Two reductions
in one kernel use different shared offsets, so neither needs a second barrier.
-/

namespace AlgorithmLib.Kernel.Block

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def shflBfly : LocalRef [.i32, .i32] (some .i32) := { callee := .intr .shflBfly }
/-- `shfl.sync.idx`: lane `src`'s value, to every lane that asks. -/
def shflIdx : LocalRef [.i32, .i32] (some .i32) := { callee := .intr .shflIdx }
def barrier : LocalRef [] none := { callee := .intr .barrier }
def ex2Approx : LocalRef [.f32] (some .f32) := { callee := .intr .ex2Approx }
def rcpApprox : LocalRef [.f32] (some .f32) := { callee := .intr .rcpApprox }
def lg2Approx : LocalRef [.f32] (some .f32) := { callee := .intr .lg2Approx }
def sinApprox : LocalRef [.f32] (some .f32) := { callee := .intr .sinApprox }
def cosApprox : LocalRef [.f32] (some .f32) := { callee := .intr .cosApprox }
def sqrtApprox : LocalRef [.f32] (some .f32) := { callee := .intr .sqrtApprox }
def rsqrtApprox : LocalRef [.f32] (some .f32) := { callee := .intr .rsqrtApprox }
/-- `dp4a.s32.s32`: `c` plus the dot product of `a`'s and `b`'s signed bytes. -/
def dp4a : LocalRef [.i32, .i32, .i32] (some .i32) := { callee := .intr .dp4a }
/-- `red.shared.add.u32`: `x` added to the shared word at `p`, which every
    thread finds after the next barrier. -/
def sharedAdd : LocalRef [.i64, .i32] none := { callee := .intr .sharedAdd }

def c32 (k : Nat) : Prog V L (V .i32) := iconst32 (Int.ofNat k)
def c64n (k : Nat) : Prog V L (V .i64) := iconst64 (Int.ofNat k)
def f32c (bits : UInt32) : Prog V L (V .f32) := fconst .f32 bits.toUInt64
/-- `a + off` bytes. -/
def at_ (a : V .i64) (off : Nat) : Prog V L (V .i64) := iaddImm a (Int.ofNat off)
def ext (x : V .i32) : Prog V L (V .i64) := uextend64 x
/-- Element `i` of the 4-byte array at `base`, the index widened first. -/
def at4 (base : V .i64) (i : V .i32) : Prog V L (V .i64) := do
  iadd base (← ishlImm (← ext i) 2)

/-- `log₂ e` and `-∞` as `Float32` bits. -/
def log2eBits : UInt32 := 0x3FB8AA3B
def negInfBits : UInt32 := 0xFF800000

/-- `2^(x · log₂ e)` by the hardware's approximation: `exp x`. -/
def expA (x : V .f32) : Prog V L (V .f32) := do
  callLocal ex2Approx %[← fmul x (← f32c log2eBits)]

/-- A float constant. -/
def fl (x : Float) : Prog V L (V .f32) := fconst .f32 x.toFloat32.toBits.toUInt64

/-- Three floats. -/
structure V3 (V : ClifTy → Type) where
  x : V .f32
  y : V .f32
  z : V .f32

def V3.map (f : V .f32 → Prog V L (V .f32)) (a : V3 V) : Prog V L (V3 V) := do
  pure ⟨← f a.x, ← f a.y, ← f a.z⟩
def V3.zip (f : V .f32 → V .f32 → Prog V L (V .f32)) (a b : V3 V) : Prog V L (V3 V) := do
  pure ⟨← f a.x b.x, ← f a.y b.y, ← f a.z b.z⟩
/-- `a · b`, as `a.x b.x` then two fused multiply-adds. -/
def V3.dot (a b : V3 V) : Prog V L (V .f32) := do
  fma a.z b.z (← fma a.y b.y (← fmul a.x b.x))
/-- `a · s + b`, a lane at a time. -/
def V3.fmaS (a : V3 V) (s : V .f32) (b : V3 V) : Prog V L (V3 V) := do
  pure ⟨← fma a.x s b.x, ← fma a.y s b.y, ← fma a.z s b.z⟩
def V3.scale (a : V3 V) (s : V .f32) : Prog V L (V3 V) := a.map (fmul · s)
def V3.lit (x y z : Float) : Prog V L (V3 V) := do pure ⟨← fl x, ← fl y, ← fl z⟩
def V3.sel (c : V .i8) (a b : V3 V) : Prog V L (V3 V) := do
  pure ⟨← select c a.x b.x, ← select c a.y b.y, ← select c a.z b.z⟩
/-- `a` over its length, by the hardware's reciprocal square root. -/
def V3.unit (a : V3 V) : Prog V L (V3 V) := do
  a.scale (← callLocal rsqrtApprox %[← a.dot a])
def V3.neg (a : V3 V) : Prog V L (V3 V) := a.map fneg

/-- `i = start, start + stride, … < n`, carrying a float from `init` through
    `step`. -/
def strided (start n : V .i32) (stride : Nat) (init : V .f32)
    (step : V .i32 → V .f32 → Prog V L (V .f32)) : Prog V L (V .f32) := do
  let r ← wloop %[start, init] (fun cs => pure (contIfULt cs.head n, %[cs.snd], ()))
    (fun cs _ => do pure %[← iadd cs.head (← c32 stride), ← step cs.head cs.snd])
  pure r.head

/-- `body i` for `i = start, start + stride, … < n`. -/
def stridedDo (start n : V .i32) (stride : Nat) (body : V .i32 → Prog V L Unit) :
    Prog V L Unit := do
  let _ ← wloop %[start] (fun cs => pure (contIfULt cs.head n, %[], ()))
    (fun cs _ => do
      body cs.head
      pure %[← iadd cs.head (← c32 stride)])

/-- A float through the warp's butterfly: `op a (lane l xor m's a)` for
    `m = 16, 8, 4, 2, 1`, after which every lane holds the same value. -/
def warpAll (op : V .f32 → V .f32 → Prog V L (V .f32)) (a : V .f32) : Prog V L (V .f32) := do
  let mut x := a
  for m in [16, 8, 4, 2, 1] do
    let o ← callLocal shflBfly %[← bitcast .i32 x, ← c32 m]
    x ← op x (← bitcast .f32 o)
  pure x

/-- **A float reduced over the block**, `warps` warps of it: each warp's
    butterfly, lane 0 of warp `w` storing its result at `sm + off + 4w`, a
    barrier, then every thread folding the `warps` results in warp order. -/
def blockAll (tid : V .i32) (sm : V .i64) (off warps : Nat)
    (op : V .f32 → V .f32 → Prog V L (V .f32)) (a : V .f32) : Prog V L (V .f32) := do
  let w ← warpAll op a
  let slot ← at_ sm off
  when .eq (← band tid (← c32 31)) (← c32 0) do
    store w (← at4 slot (← ushrImm tid 5))
  callLocalVoid barrier %[]
  let mut acc ← loadF32 slot
  for k in List.range' 1 (warps - 1) do
    acc ← op acc (← loadF32 (← at_ slot (4 * k)))
  pure acc

/-- `vote.sync.ballot`: bit `l` set where lane `l` passed true. -/
def ballotR : LocalRef [.i8] (some .i32) := { callee := .intr .ballot }

/-- The set bits of `x` (the PTX compiler has no `popc`): pairs, nibbles,
    bytes, then the bytes summed by a multiply. -/
def popc32 (x : V .i32) : Prog V L (V .i32) := do
  let a ← isub x (← band (← ushrImm x 1) (← c32 0x55555555))
  let b ← iadd (← band a (← c32 0x33333333)) (← band (← ushrImm a 2) (← c32 0x33333333))
  let c ← band (← iadd b (← ushrImm b 4)) (← c32 0x0F0F0F0F)
  ushrImm (← imul c (← c32 0x01010101)) 24

/-- `1` where `c` holds, else `0`. -/
def bit (c : V .i8) : Prog V L (V .i32) := do select c (← c32 1) (← c32 0)

/-- The lanes holding the same 8-bit digit `d` as this one, among the lanes
    where `active` holds: one ballot of `active`, then one a bit of `d`. -/
def peers (d : V .i32) (active : V .i8) : Prog V L (V .i32) := do
  let mut m ← callLocal ballotR %[active]
  for b in List.range 8 do
    let isSet ← icmp .ne (← band (← ushrImm d b) (← c32 1)) (← c32 0)
    let bal ← callLocal ballotR %[isSet]
    m ← band m (← select isSet bal (← bxor bal (← c32 0xFFFFFFFF)))
  pure m

/-- The lowest finite `Float32`: the maximum of a softmax state that has seen
    nothing, and what an input's `-∞` is raised to (`lowest`). With no
    maximum ever `-∞`, no merge subtracts `-∞` from `-∞`, so none makes a
    NaN, and the merge needs no test for it, which would cost the reduction
    a tenth of its time. -/
def lowestBits : UInt32 := 0xFF7FFFFF

/-- An input raised to the lowest finite float: unchanged unless `-∞`. -/
def lowest (x : V .f32) : Prog V L (V .f32) := do fmax x (← f32c lowestBits)

/-- **Two running softmax states merged**: `(m, s)` stands for `s · e^m`,
    `m` finite. The state with the larger maximum (the first on a tie) is
    kept and the other's sum rescaled to it, `s_big + s_small ·
    e^(m_small - m_big)` in one fused multiply-add: the kept sum's own factor
    is `e^0 = 1`, so one `exp` serves. The maxima come from `max` and `min`
    rather than the comparison, so a butterfly's chain from one shuffle of the
    maximum to the next is one instruction. -/
def online (m1 s1 m2 s2 : V .f32) : Prog V L (V .f32 × V .f32) := do
  let hi ← fmax m1 m2
  let lo ← fmin m1 m2
  let ge ← fcmp .ge m1 m2
  let sHi ← select ge s1 s2
  let sLo ← select ge s2 s1
  pure (hi, ← fma sLo (← expA (← fsub lo hi)) sHi)

/-- A running softmax state through the warp's butterfly. -/
def warpOnline (m s : V .f32) : Prog V L (V .f32 × V .f32) := do
  let mut st := (m, s)
  for k in [16, 8, 4, 2, 1] do
    let om ← callLocal shflBfly %[← bitcast .i32 st.1, ← c32 k]
    let os ← callLocal shflBfly %[← bitcast .i32 st.2, ← c32 k]
    st ← online st.1 st.2 (← bitcast .f32 om) (← bitcast .f32 os)
  pure st

/-- **A running softmax state reduced over the block, to thread 0**: each
    warp's butterfly, lane 0 of warp `w` storing its maximum at `sm + 4w` and
    its sum at `sm + 4·warps + 4w`, a barrier, then thread 0 alone merges the
    warps' states in warp order and runs `k` on the result. The merge costs
    fourteen `ex2` a block, which the other warps would otherwise spend issue
    slots on that the block's neighbours on the SM could use. -/
def blockOnline (tid : V .i32) (sm : V .i64) (warps : Nat) (m s : V .f32)
    (k : V .f32 → V .f32 → Prog V L Unit) : Prog V L Unit := do
  let (wm, ws) ← warpOnline m s
  let sSlot ← at_ sm (4 * warps)
  when .eq (← band tid (← c32 31)) (← c32 0) do
    let w ← ushrImm tid 5
    store wm (← at4 sm w)
    store ws (← at4 sSlot w)
  callLocalVoid barrier %[]
  when .eq tid (← c32 0) do
    let mut st := (← loadF32 sm, ← loadF32 sSlot)
    for j in List.range' 1 (warps - 1) do
      st ← online st.1 st.2 (← loadF32 (← at_ sm (4 * j))) (← loadF32 (← at_ sSlot (4 * j)))
    k st.1 st.2

end

end AlgorithmLib.Kernel.Block
