module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
public import AlgorithmLib.Vocab.Libm
meta import AlgorithmLib.Vocab.Libm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Math` — `sinf`, `cosf`, `powf` as CLIF

`Vocab.Libm` states each as a sequence of IEEE operations; here each is that
sequence, instruction for instruction, with the same constants by their bits.
A branch the definition takes, this takes; where the definition computes only
the value one branch needs, this may compute both and select, which leaves the
answer the same. So the functions a program calls answer what `Libm` says, on
every machine.
-/

namespace AlgorithmLib.LibMath

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def f64c (x : Float) : Prog V L (V .f64) := fconst .f64 x.toBits
def f32c (b : UInt32) : Prog V L (V .f32) := fconst .f32 b.toUInt64

/-- `Libm.horner`, innermost first. -/
def horner (cs : List Float) (z : V .f64) : Prog V L (V .f64) := do
  let mut acc ← f64c 0.0
  for c in cs.reverse do
    acc ← fadd (← f64c c) (← fmul z acc)
  pure acc

def sinPoly (r : V .f64) : Prog V L (V .f64) := do
  let z ← fmul r r
  let h ← horner Libm.sinC z
  fadd r (← fmul (← fmul r z) h)

def cosPoly (r : V .f64) : Prog V L (V .f64) := do
  let z ← fmul r r
  let h ← horner Libm.cosC z
  fadd (← f64c 1.0) (← fmul z h)

def shl (a b : V .i64) : Prog V L (V .i64) := ishl a b
def shr (a b : V .i64) : Prog V L (V .i64) := ushr a b

/-- `Libm.bits64`, each case computed and the one that applies selected. -/
def bits64 (p0 p1 p2 off : V .i64) : Prog V L (V .i64) := do
  let c64 ← iconst64 64
  let c128 ← iconst64 128
  let a1 ← bor (← shr p0 off) (← shl p1 (← isub c64 off))
  let a3 ← bor (← shr p1 (← isub off c64)) (← shl p2 (← isub c128 off))
  let a5 ← shr p2 (← isub off c128)
  let r ← select (← icmp .eq off c128) p2 a5
  let r ← select (← icmp .ult off c128) a3 r
  let r ← select (← icmp .eq off c64) p1 r
  select (← icmp .ult off c64) a1 r

/-- `Libm.reduce`: the quadrant and the remainder of `|x|` by `π/2`. -/
def reduce (ax : V .i32) : Prog V L (V .i64 × V .f64) := do
  let ef ← uextend64 (← ushrImm ax 23)
  let m ← bor (← uextend64 (← band ax (← iconst32 0x7fffff))) (← iconst64 0x800000)
  let zero ← iconst64 0
  let big ← icmp .uge ef (← iconst64 152)
  let s1 ← select big (← iaddImm ef (-152)) zero
  let w ← ushrImm s1 6
  let o ← band s1 (← iconst64 63)
  let t (i : Nat) : Prog V L (V .i64) := iconst64 (Libm.twoOverPi[i]!).toInt64.toInt
  let w0 ← icmp .eq w zero
  let t0 ← select w0 (← t 0) (← t 1)
  let t1 ← select w0 (← t 1) (← t 2)
  let t2 ← select w0 (← t 2) (← t 3)
  let o0 ← icmp .eq o zero
  let back ← isub (← iconst64 64) o
  let hi ← select o0 t0 (← bor (← shl t0 o) (← shr t1 back))
  let lo ← select o0 t1 (← bor (← shl t1 o) (← shr t2 back))
  let p0 ← imul m lo
  let c ← umulhi m lo
  let tt ← imul m hi
  let p1 ← iadd tt c
  let p2 ← iadd (← umulhi m hi) (← uextend64 (← icmp .ult p1 tt))
  let q ← select big (← iconst64 126) (← isub (← iconst64 278) ef)
  let f ← bits64 p0 p1 p2 (← iaddImm q (-64))
  let k ← iadd (← band (← bits64 p0 p1 p2 q) (← iconst64 3)) (← ushrImm f 63)
  let k ← band k (← iconst64 3)
  let r ← fmul (← fcvtFromSint .f64 f) (← f64c Libm.pio2Scaled)
  pure (k, r)

/-- The four quadrants' values, selected by `k`. -/
def byQuadrant (k : V .i64) (v0 v1 v2 v3 : V .f64) : Prog V L (V .f64) := do
  let r ← select (← icmp .eq k (← iconst64 2)) v2 v3
  let r ← select (← icmp .eq k (← iconst64 1)) v1 r
  select (← icmp .eq k (← iconst64 0)) v0 r

/-- `Libm.nanOf`: a NaN's bits quietened, or `0x7fc00000` for an infinity. -/
def nanOf (b : V .i32) : Prog V L (V .f32) := do
  let isInf ← icmp .eq (← band b (← iconst32 0x7fffff)) (← iconst32 0)
  bitcast .f32 (← select isInf (← iconst32 0x7fc00000) (← bor b (← iconst32 0x400000)))

/-- An `f32` answered as the `i64` a library function answers. -/
def answer (v : V .f32) : Prog V L (V .i64) := do uextend64 (← bitcast .i32 v)

/-- `sinf` (`cos` for `cos = true`), as `Libm.sinf` and `Libm.cosf`. -/
def trig (cos : Bool) : StatusBody := do
  let (x ::ᵥ .nil) ← entryParams [.f32]
  let b ← bitcast .i32 x
  let ax ← band b (← iconst32 0x7fffffff)
  let r ← ifte (jTys := [.f32]) .uge ax (← iconst32 0x7f800000) (do pure %[← nanOf b]) (do
    let r ← ifte (jTys := [.f32]) .ult ax (← iconst32 0x3f000000)
      (do
        let d ← fpromote x
        pure %[← fdemote (← if cos then cosPoly d else sinPoly d)])
      (do
        let (k, r) ← reduce ax
        let s ← sinPoly r
        let c ← cosPoly r
        let ns ← fneg s
        let nc ← fneg c
        let v ← if cos then byQuadrant k c ns nc s else do
          let v ← byQuadrant k s c ns nc
          select (← icmp .ne (← ushrImm b 31) (← iconst32 0)) (← fneg v) v
        pure %[← fdemote v])
    pure %[r.head])
  answer r.head

/-- `Libm.log2Abs`. -/
def log2Abs (d : V .f64) : Prog V L (V .f64) := do
  let db ← bitcast .i64 d
  let e ← iaddImm (← band (← ushrImm db 52) (← iconst64 0x7ff)) (-1023)
  let m ← bitcast .f64 (← bor (← band db (← iconst64 0x000fffffffffffff))
    (← iconst64 0x3ff0000000000000))
  let big ← fcmp .gt m (← f64c Libm.sqrt2)
  let m ← select big (← fmul m (← f64c 0.5)) m
  let e ← select big (← iaddImm e 1) e
  let s ← fdiv (← fsub m (← f64c 1.0)) (← fadd m (← f64c 1.0))
  let z ← fmul s s
  let s2 ← fadd s s
  let lnm ← fadd s2 (← fmul (← fmul s2 z) (← horner Libm.lnC z))
  fadd (← fcvtFromSint .f64 e) (← fmul lnm (← f64c Libm.invLn2))

/-- `Libm.exp2`. -/
def exp2 (t : V .f64) : Prog V L (V .f64) := do
  let n ← floor (← fadd t (← f64c 0.5))
  let g ← fmul (← fsub t n) (← f64c Libm.ln2)
  let p ← fadd (← f64c 1.0) (← fmul g (← horner Libm.expC g))
  let nb ← ishlImm (← iaddImm (← fcvtToSint .i64 n) 1023) 52
  fmul p (← bitcast .f64 nb)

/-- `powf`, as `Libm.powf`. -/
def powf : StatusBody := do
  let (x ::ᵥ y ::ᵥ .nil) ← entryParams [.f32, .f32]
  let xb ← bitcast .i32 x
  let yb ← bitcast .i32 y
  let m31 ← iconst32 0x7fffffff
  let ax ← band xb m31
  let ay ← band yb m31
  let yd ← fpromote y
  let z32 ← iconst32 0
  let yneg ← icmp .ne (← ushrImm yb 31) z32
  let xneg ← icmp .ne (← ushrImm xb 31) z32
  let one ← f32c 0x3f800000
  let inf ← iconst32 0x7f800000
  let bitsF (v : Slot .i32) : Prog Slot Lvl (Vals Slot [.f32]) := do pure %[← bitcast .f32 v]
  let r ← ifte (jTys := [.f32]) .eq ay z32 (pure %[one]) (do
   let r ← ifte (jTys := [.f32]) .eq xb (← iconst32 0x3f800000) (pure %[one]) (do
    -- a NaN operand wins, the base first
    let r ← ifte (jTys := [.f32]) .ugt (← umax ax ay) inf
      (do pure %[← nanOf (← select (← icmp .ugt ax inf) xb yb)]) (do
     let r ← ifte (jTys := [.f32]) .eq ay inf
      (do
        let r ← ifte (jTys := [.f32]) .eq ax (← iconst32 0x3f800000) (pure %[one]) (do
          let small ← icmp .ult ax (← iconst32 0x3f800000)
          let toZero ← icmp .ne (← bxor small yneg) (← iconst .i8 0)
          bitsF (← select toZero z32 inf))
        pure %[r.head])
      (do
        let yInt ← fcmp .eq (← floor yd) yd
        let half ← fmul yd (← f64c 0.5)
        let odd ← band yInt (← fcmp .ne (← floor half) half)
        let neg ← band odd xneg
        let sign ← select neg (← iconst32 0x80000000) z32
        let r ← ifte (jTys := [.f32]) .eq ax z32
          (do bitsF (← bor sign (← select yneg inf z32)))
          (do
            let r ← ifte (jTys := [.f32]) .eq ax inf
              (do bitsF (← bor sign (← select yneg z32 inf)))
              (do
                let bad ← band xneg (← bxor yInt (← iconst .i8 1))
                let r ← ifte (jTys := [.f32]) .ne bad (← iconst .i8 0)
                  (do bitsF (← iconst32 0x7fc00000))
                  (do
                    let t ← fmul yd (← log2Abs (← fpromote x))
                    let v ← ifte (jTys := [.f64]) .eq (← fcmp .ge t (← f64c 129.0)) (← iconst .i8 0)
                      (do
                        let v ← ifte (jTys := [.f64]) .eq (← fcmp .le t (← f64c (-152.0))) (← iconst .i8 0)
                          (do pure %[← exp2 t])
                          (do pure %[← f64c 0.0])
                        pure %[v.head])
                      (do pure %[← fconst .f64 0x7ff0000000000000])
                    let v ← select (← icmp .ne sign z32) (← fneg v.head) v.head
                    pure %[← fdemote v])
                pure %[r.head])
            pure %[r.head])
        pure %[r.head])
     pure %[r.head])
    pure %[r.head])
   pure %[r.head])
  answer r.head

/-- The function implementing a math entry point. -/
def implOf : Ffi → Option Impl
  | .sinf => some (.status (trig false))
  | .cosf => some (.status (trig true))
  | .powf => some (.status powf)
  | _ => none

end AlgorithmLib.LibMath
