import AlgorithmLib.LZ4
import AlgorithmLib.PTX

namespace AlgorithmLib.LZ4Imp
open AlgorithmLib

abbrev Var := Nat

inductive Operand where
  | v (x : Var)
  | c (n : Nat)

inductive Op where
  | add | sub | band | bor | shl | shr

inductive Cmp where
  | eq | ne | lt | ge

-- Which cooperative-copy schedule the printer should emit for a `coop` block.
inductive CopyKind where
  | lit    -- literal run: input → output
  | matchC -- back-reference match: output → output

inductive Stmt where
  | mov (d : Var) (s : Operand)
  | binop (o : Op) (d a : Var) (b : Operand)
  | ldIn (d a : Var)
  | ldOut (d a : Var)
  | stOut (a s : Var)
  | seq (s t : Stmt)
  | skip
  | whileLoop (c : Cmp) (a : Var) (b : Operand) (body : Stmt)
  | ite (c : Cmp) (a : Var) (b : Operand) (t e : Stmt)
  -- Cooperative wrapper: `s` is the proven serial meaning (used by `eval` and
  -- the single-thread machine model); `k` tells the parallel printer which
  -- warp-cooperative copy to emit instead. Semantically transparent.
  | coop (k : CopyKind) (s : Stmt)

def block : List Stmt → Stmt := fun l => l.foldr .seq .skip

def Op.eval : Op → Nat → Nat → Nat
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | .band, a, b => a &&& b
  | .bor, a, b => a ||| b
  | .shl, a, b => a <<< b
  | .shr, a, b => a >>> b

def Cmp.eval : Cmp → Nat → Nat → Bool
  | .eq, a, b => a == b
  | .ne, a, b => a != b
  | .lt, a, b => Nat.blt a b
  | .ge, a, b => Nat.ble b a

structure St where
  inp  : List UInt8
  out  : List UInt8
  vars : Var → Nat

def St.get (s : St) : Operand → Nat
  | .v x => s.vars x
  | .c n => n

def St.set (s : St) (x : Var) (n : Nat) : St :=
  { s with vars := fun y => if y = x then n else s.vars y }

-- Fuel bounds only the total number of loop iterations; straight-line code is
-- fuel-free, so `seq` composes at the same fuel (recursion is lexicographic
-- on (fuel, statement size)).
def eval : Nat → Stmt → St → Option St
  | _, .mov d s, st => some (st.set d (st.get s))
  | _, .binop o d a b, st => some (st.set d (o.eval (st.vars a) (st.get b)))
  | _, .ldIn d a, st => some (st.set d (st.inp.getD (st.vars a) 0).toNat)
  | _, .ldOut d a, st => some (st.set d (st.out.getD (st.vars a) 0).toNat)
  | _, .stOut a s, st =>
      some { st with out := st.out.set (st.vars a) (UInt8.ofNat (st.vars s)) }
  | f, .seq s t, st => (eval f s st).bind (eval f t)
  | _, .skip, st => some st
  | f, .ite c a b t e, st =>
      if c.eval (st.vars a) (st.get b) then eval f t st else eval f e st
  | 0, .whileLoop c a b _, st =>
      if c.eval (st.vars a) (st.get b) then none else some st
  | f+1, .whileLoop c a b body, st =>
      if c.eval (st.vars a) (st.get b) then
        (eval (f+1) body st).bind (eval f (.whileLoop c a b body))
      else some st
  | f, .coop _ s, st => eval f s st
termination_by f s _ => (f, sizeOf s)
decreasing_by all_goals simp_wf <;> omega

-- The LZ4 block decompressor as data.
def vIP  : Var := 0
def vOP  : Var := 1
def vEND : Var := 2
def vTOK : Var := 3
def vLEN : Var := 4
def vB   : Var := 5
def vOFF : Var := 6
def vSRC : Var := 7
def vRUN : Var := 8
def vTMP : Var := 9

-- LSIC extension: while the read byte is 255, keep adding (do-while via vB=255 seed).
def extWhileBody : Stmt := block [
  .ldIn vB vIP,
  .binop .add vIP vIP (.c 1),
  .binop .add vLEN vLEN (.v vB)]

def extWhile : Stmt := .whileLoop .eq vB (.c 255) extWhileBody

def extLoop : Stmt := .seq (.mov vB (.c 255)) extWhile

def litCopyBody : Stmt := block [
  .ldIn vB vIP,
  .stOut vOP vB,
  .binop .add vIP vIP (.c 1),
  .binop .add vOP vOP (.c 1),
  .binop .sub vLEN vLEN (.c 1)]

def litCopy : Stmt := .whileLoop .ne vLEN (.c 0) litCopyBody

def mcopyBody : Stmt := block [
  .ldOut vB vSRC,
  .stOut vOP vB,
  .binop .add vSRC vSRC (.c 1),
  .binop .add vOP vOP (.c 1),
  .binop .sub vLEN vLEN (.c 1)]

def mcopyLoop : Stmt := .whileLoop .ne vLEN (.c 0) mcopyBody

def matchPart : Stmt := block [
  .ldIn vOFF vIP,
  .binop .add vTMP vIP (.c 1),
  .ldIn vB vTMP,
  .binop .shl vB vB (.c 8),
  .binop .add vOFF vOFF (.v vB),
  .binop .add vIP vIP (.c 2),
  .binop .band vLEN vTOK (.c 15),
  .ite .eq vLEN (.c 15) extLoop .skip,
  .binop .add vLEN vLEN (.c 4),
  .mov vSRC (.v vOP),
  .binop .sub vSRC vSRC (.v vOFF),
  mcopyLoop]

def mainBody : Stmt := block [
  .ldIn vTOK vIP,
  .binop .add vIP vIP (.c 1),
  .binop .shr vLEN vTOK (.c 4),
  .ite .eq vLEN (.c 15) extLoop .skip,
  litCopy,
  .ite .ge vIP (.v vEND) (.mov vRUN (.c 0)) matchPart]

def decompressProg : Stmt := .whileLoop .eq vRUN (.c 1) mainBody

def initVars (inp : List UInt8) : Var → Nat :=
  fun v => if v = vRUN then 1 else if v = vEND then inp.length else 0

def St.init (inp : List UInt8) (cap : Nat) : St :=
  { inp, out := List.replicate cap 0, vars := initVars inp }

def defaultFuel (inp cap : Nat) : Nat := 64 * (inp + cap + 8)

def decompress (inp : List UInt8) (cap : Nat) : Option (List UInt8) :=
  (eval (defaultFuel inp.length cap) decompressProg (St.init inp cap)).map
    fun st => st.out.take (st.vars vOP)

open PTX

end AlgorithmLib.LZ4Imp
