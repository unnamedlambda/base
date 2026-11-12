module
public import AlgorithmLib.Kernel.PtxIr
meta import AlgorithmLib.Kernel.PtxIr
public import Std.Data.HashMap
public import Std.Data.HashSet
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Coalesce` — a kernel's registers merged where they never meet

The compiler (`Kernel.Ptx`) gives every value its own register and passes a
value into a branch's join or a loop's next trip by a move into the join's
own register. `ptxas` keeps many of those moves: where two paths meet, each
arrives with its value in a different register. Hand-written PTX has none,
because every path writes the same register.

This pass renames registers so that a move's destination and its source
share one, wherever no point of the program needs both: a register `d` and a
register `s` merge only if `s` is dead after a move `d := s`, and no write to
either happens while the other is live. The move then copies a register to
itself, which `ptxas` deletes; it is kept in the body, so every line keeps
its place and every label its line.

The renaming is planned without proof (`plan`) and checked (`valid`) before
it is used: the liveness the plan states must satisfy the dataflow equations
at every line, and a register written at a line must not share a name with
another register live after it. `Sound.PtxCoalesce` proves that a renamed
body that passes the check runs every thread to the same world as the body
it came from.
-/

namespace AlgorithmLib.Kernel.Coalesce

open AlgorithmLib.PtxIr

/-- Each label's line: `labelsOf`, which a run jumps by. -/
def labelsOf (body : List Line) : Std.HashMap Lbl Nat :=
  body.zipIdx.foldl (fun m (l, i) => match l with
    | .label n => m.insert n i
    | _ => m) {}

/-- A renaming, as a table: a register it lists becomes its entry. -/
abbrev Ren := Std.HashMap Reg Reg

/-- A register renamed: never a special register, and never into one. -/
def Reg.ren (ρ : Ren) : Reg → Reg
  | .special s => .special s
  | r => match ρ.get? r with
    | some (.val c n) => .val c n
    | some (.own p t n) => .own p t n
    | some (.name n) => .name n
    | _ => r

def Opnd.ren (ρ : Ren) : Opnd → Opnd
  | .reg r => .reg (Reg.ren ρ r)
  | o => o

def Addr.ren (ρ : Ren) (a : Addr) : Addr := { a with base := Opnd.ren ρ a.base }

def Instr.ren (ρ : Ren) : Instr → Instr
  | .mov t d a => .mov t (Reg.ren ρ d) (Opnd.ren ρ a)
  | .bin o r t d a b => .bin o r t (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b)
  | .un o r t d a => .un o r t (Reg.ren ρ d) (Opnd.ren ρ a)
  | .fma r t d a b c => .fma r t (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ c)
  | .cvt r dt st d a => .cvt r dt st (Reg.ren ρ d) (Opnd.ren ρ a)
  | .cvta to sp t d a => .cvta to sp t (Reg.ren ρ d) (Opnd.ren ρ a)
  | .setp c t p a b => .setp c t (Reg.ren ρ p) (Opnd.ren ρ a) (Opnd.ren ρ b)
  | .selp t d a b p => .selp t (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ p)
  | .prmt d a b c => .prmt (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ c)
  | .dp4a d a b c => .dp4a (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ c)
  | .ld sp nc t d a => .ld sp nc t (Reg.ren ρ d) (Addr.ren ρ a)
  | .st sp t a v => .st sp t (Addr.ren ρ a) (Opnd.ren ρ v)
  | .atom sp o od t d a v => .atom sp o od t (d.map (Reg.ren ρ)) (Addr.ren ρ a) (Opnd.ren ρ v)
  | .redShared a v => .redShared (Addr.ren ρ a) (Opnd.ren ρ v)
  | .shfl m d a b c k => .shfl m (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ c) (Opnd.ren ρ k)
  | .vote d p k => .vote (Reg.ren ρ d) (Opnd.ren ρ p) (Opnd.ren ρ k)
  | .mma d a b c => .mma (Reg.ren ρ d) (Opnd.ren ρ a) (Opnd.ren ρ b) (Opnd.ren ρ c)
  | .bar n => .bar n
  | .barWarp k => .barWarp (Opnd.ren ρ k)
  | .bra u l => .bra u l
  | .ret => .ret

def Line.ren (ρ : Ren) : Line → Line
  | .label l => .label l
  | .ins g i => .ins (g.map fun (n, p) => (n, Reg.ren ρ p)) (Instr.ren ρ i)

/-- The registers an operand reads. -/
def Opnd.regs : Opnd → List Reg
  | .reg r => [r]
  | _ => []

/-- The registers an instruction reads. -/
def Instr.uses : Instr → List Reg
  | .mov _ _ a | .un _ _ _ _ a | .cvt _ _ _ _ a | .cvta _ _ _ _ a => Opnd.regs a
  | .bin _ _ _ _ a b | .setp _ _ _ a b => Opnd.regs a ++ Opnd.regs b
  | .fma _ _ _ a b c | .prmt _ a b c | .dp4a _ a b c | .mma _ a b c => Opnd.regs a ++ Opnd.regs b ++ Opnd.regs c
  | .selp _ _ a b p => Opnd.regs a ++ Opnd.regs b ++ Opnd.regs p
  | .ld _ _ _ _ a => Opnd.regs a.base
  | .st _ _ a v | .atom _ _ _ _ _ a v | .redShared a v => Opnd.regs a.base ++ Opnd.regs v
  | .shfl _ _ a b c k => Opnd.regs a ++ Opnd.regs b ++ Opnd.regs c ++ Opnd.regs k
  | .vote _ p k => Opnd.regs p ++ Opnd.regs k
  | .barWarp k => Opnd.regs k
  | .bar _ | .bra _ _ | .ret => []

/-- The register an instruction writes, if any. -/
def Instr.def? : Instr → Option Reg
  | .mov _ d _ | .bin _ _ _ d _ _ | .un _ _ _ d _ | .fma _ _ d _ _ _ | .cvt _ _ _ d _
  | .cvta _ _ _ d _ | .setp _ _ d _ _ | .selp _ d _ _ _ | .prmt d _ _ _ | .dp4a d _ _ _ | .ld _ _ _ d _
  | .shfl _ d _ _ _ _ | .vote d _ _ | .mma d _ _ _ => some d
  | .atom _ _ _ _ d _ _ => d
  | _ => none

/-- A bound on the bits an instruction writes: its result fits this many.
    A float result at `f32` is 32 bits, approximations included (`HwOk`); a
    comparison's is one; where nothing smaller is known, 64. -/
def Instr.resWidth : Instr → Nat
  | .mov t _ _ | .selp t _ _ _ _ | .fma none t _ _ _ _ => t.width
  | .bin o _ t _ _ _ => if t.isFloat then (if t == .f32 then 32 else 64)
      else if o == .mulWide then 64 else t.width
  | .un o _ t _ _ => if t == .f32 && (o == .neg || o == .abs || o == .sqrt || o == .ex2Approx ||
      o == .lg2Approx || o == .rcpApprox || o == .rsqrtApprox || o == .sqrtApprox || o == .sinApprox ||
      o == .cosApprox) then 32 else 64
  | .fma (some _) _ _ _ _ _ => 32
  | .cvt _ dt st _ _ => if dt == st then 64 else if dt == .f32 then 32
      else if !dt.isFloat && !st.isFloat then dt.width else 64
  | .setp .. => 1
  | .shfl .. | .vote .. | .mma .. => 32
  | _ => 64

/-- Each register's bound: the widest any line writing it writes. -/
def bounds (body : Array Line) : Std.HashMap Reg Nat :=
  body.foldl (fun m l => match l with
    | .ins _ i => match Instr.def? i with
        | some d => m.insert d (max (m.getD d 0) (Instr.resWidth i))
        | none => m
    | _ => m) {}

/-- A register the pass may rename: one the compiler named. -/
def renamable : Reg → Bool
  | .val .. | .own .. => true
  | _ => false


/-- A line renamed, where a move left copying a register to itself whose
    source fits the move's width becomes a label. -/
def renLine (ρ : Ren) (bd : Std.HashMap Reg Nat) (pc : Nat) : Line → Line
  | .ins none (.mov t d (.reg s)) =>
      if Reg.ren ρ d == Reg.ren ρ s && renamable d && renamable s && bd.getD s 64 ≤ t.width then .label (.gone pc)
      else .ins none (.mov t (Reg.ren ρ d) (.reg (Reg.ren ρ s)))
  | l => Line.ren ρ l

/-- The registers a line reads: its guard's and its instruction's. -/
def Line.uses : Line → List Reg
  | .label _ => []
  | .ins g i => (match g with | some (_, p) => [p] | none => []) ++ Instr.uses i

/-- The lines a run may go to from line `pc`. -/
def succs (labels : Std.HashMap Lbl Nat) (pc : Nat) : Line → List Nat
  | .label _ => [pc + 1]
  | .ins g (.bra _ l) => (match labels.get? l with | some j => [j] | none => []) ++
      (if g.isSome then [pc + 1] else [])
  | .ins g .ret => if g.isSome then [pc + 1] else []
  | .ins _ _ => [pc + 1]

/-- The registers live after line `pc`: those live into any line it may go to. -/
def liveOut (labels : Std.HashMap Lbl Nat) (live : Array (List Reg)) (pc : Nat) (l : Line) : List Reg :=
  (succs labels pc l).flatMap fun s => live.getD s []

/-- **The check**: at every line of `body`, the registers `live` states live
    into it hold every one it reads and every one live after it that it does
    not write unconditionally; and a register it writes shares its name under
    `ρ` with no other register live after it. -/
def valid (body : Array Line) (ρ : Ren) (live : Array (List Reg)) : Bool :=
  let labels := labelsOf body.toList
  (List.range body.size).all fun pc =>
    let l := body.getD pc (.label (.blk 0))
    let inn := live.getD pc []
    let out := liveOut labels live pc l
    match l with
    | .label _ => out.all (inn.contains ·)
    | .ins g i =>
        (Line.uses l).all (inn.contains ·) &&
        out.all (fun r => inn.contains r || (g.isNone && Instr.def? i == some r)) &&
        match Instr.def? i with
        | none => true
        | some d => out.all fun r => r == d || Reg.ren ρ r != Reg.ren ρ d

/-! ## The plan -/

/-- Each register's declared type. -/
def declTypes (e : Entry) : Std.HashMap Reg Ty :=
  e.regs.foldl (fun m d => match d with
    | .range t pfx n =>
        -- a range is declared as `%pfx<n>`, its registers named without the `%`
        let c := if pfx.startsWith "%" then pfx.drop 1 else pfx
        (List.range n).foldl (fun m i => m.insert (.val c i) t) m
    | .names t ns => ns.foldl (fun m r => m.insert r t) m) {}

/-- The registers live into each line, by iterating the dataflow equations from
    nothing until they settle, at most `fuel` sweeps. -/
def liveness (body : Array Line) (fuel : Nat) : Array (List Reg) := Id.run do
  let labels := labelsOf body.toList
  let n := body.size
  let mut live : Array (Std.HashSet Reg) := Array.replicate (n + 1) {}
  for _ in List.range fuel do
    let mut changed := false
    for k in List.range n do
      let pc := n - 1 - k
      let l := body.getD pc (.label (.blk 0))
      let mut s : Std.HashSet Reg := {}
      for t in succs labels pc l do
        for r in (live.getD t {}).toList do s := s.insert r
      match l with
      | .ins none i => match Instr.def? i with
          | some d => s := s.erase d
          | none => pure ()
      | _ => pure ()
      for u in Line.uses l do s := s.insert u
      if s.size != (live.getD pc {}).size then
        live := live.set! pc s
        changed := true
    if !changed then break
  pure (live.map (·.toList))

/-- **The plan**: the liveness, and a renaming that merges each move's
    destination with its source where the source dies at the move, both have
    one declared type, and no register of the one class is written while one
    of the other is live. -/
def plan (e : Entry) : Ren × Array (List Reg) := Id.run do
  let body := e.body.toArray
  let labels := labelsOf e.body
  let live := liveness body (body.size + 2)
  let tys := declTypes e
  let bd := bounds body
  -- interference: a written register against every other register live after,
  -- as one set of pairs (both orders)
  let mut inter : Std.HashSet (Reg × Reg) := {}
  let mut moves : Array (Reg × Reg) := #[]
  for pc in List.range body.size do
    let l := body.getD pc (.label (.blk 0))
    let out := liveOut labels live pc l
    match l with
    | .ins g i =>
        match Instr.def? i with
        | some d =>
            for r in out do
              if r != d then
                inter := (inter.insert (d, r)).insert (r, d)
            match g, i with
            | none, .mov t d' (.reg s) =>
                -- merged only where the move, left copying a register to itself, can go
                if !out.contains s && bd.getD s 64 ≤ t.width then moves := moves.push (d', s)
            | _, _ => pure ()
        | none => pure ()
    | _ => pure ()
  -- union of classes: two merge where no member of one meets one of the other
  let mut rep : Std.HashMap Reg Reg := {}
  let mut members : Std.HashMap Reg (List Reg) := {}
  let find (rep : Std.HashMap Reg Reg) (r : Reg) : Reg := rep.getD r r
  for (d, s) in moves do
    let a := find rep d
    let b := find rep s
    if a != b && renamable a && renamable b && tys.get? a == tys.get? b && (tys.get? a).isSome then
      let ma := members.getD a [a]
      let mb := members.getD b [b]
      if !(mb.any fun x => ma.any fun y => inter.contains (y, x)) then
        for x in ma do rep := rep.insert x b
        members := members.insert b (mb ++ ma)
  pure (rep, live)

/-- No label of `body`, and no label it jumps to, is a removed move's
    (`Lbl.gone`) or a named one, so the labels removed moves leave are
    distinct from every other, by value and as text. -/
def fresh (body : List Line) : Bool :=
  body.all fun l => match l with
    | .label (.gone _) | .label (.name _) | .ins _ (.bra _ (.gone _)) | .ins _ (.bra _ (.name _)) => false
    | _ => true

/-- **The coalesced entry**: its registers renamed as planned where the plan
    passes the check, a move left copying a register to itself removed (a
    label in its place, so every line keeps its index), and the entry as it
    was where the check fails. -/
def run (e : Entry) : Entry :=
  let (ρ, live) := plan e
  if valid e.body.toArray ρ live && fresh e.body then
    let bd := bounds e.body.toArray
    { e with body := e.body.zipIdx.map fun (l, pc) => renLine ρ bd pc l }
  else e

end AlgorithmLib.Kernel.Coalesce
