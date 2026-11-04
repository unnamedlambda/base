module
public import AlgorithmLib.LZ4.WarpKernel
public import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Kernel.Ptx
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeKernel` — the warp language as a kernel's Code

The warp language (`LZ4WarpDSL.WStmt`) says what one warp does, its registers
the warp's. Each lane of the kernel runs it: a register is a value every lane
holds alike, a loop or a branch carries the registers its body sets, and a
cooperative statement is the lanes' share of it, joined by a ballot or a
shuffle. The kernel is one warp per block; its shared memory is the block's
hash table.
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.LZ4WarpDSL
  AlgorithmLib.Kernel

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def ballot : LocalRef [.i8] (some .i32) := { callee := .intr .ballot }
def shflIdx : LocalRef [.i32, .i32] (some .i32) := { callee := .intr .shflIdx }
def barrier : LocalRef [] none := { callee := .intr .barrier }

/-- The registers' values, the latest binding first: a flag a predicate, any
    other register a 64-bit value. -/
structure RegEnv (V : ClifTy → Type) where
  vals : List (String × V .i64) := []
  flags : List (String × V .i8) := []

def RegEnv.setVal (e : RegEnv V) (r : String) (v : V .i64) : RegEnv V := { e with vals := (r, v) :: e.vals }
def RegEnv.setFlag (e : RegEnv V) (r : String) (v : V .i8) : RegEnv V := { e with flags := (r, v) :: e.flags }

/-- A flag's predicate: a register not a flag tested, as `uif` and `uwhile`
    test it, against one. -/
def RegEnv.getFlag (fl : List String) (e : RegEnv V) (r : String) : Prog V L (V .i8) := do
  if fl.contains r then
    match e.flags.lookup r with
    | some v => pure v
    | none => iconst .i8 0
  else
    let v ← match e.vals.lookup r with
      | some v => pure v
      | none => iconst64 0
    icmp .eq v (← iconst64 1)

/-- A register's 64-bit value: a flag 1 or 0. -/
def RegEnv.get (fl : List String) (e : RegEnv V) (r : String) : Prog V L (V .i64) := do
  if fl.contains r then select (← e.getFlag fl r) (← iconst64 1) (← iconst64 0)
  else
    match e.vals.lookup r with
    | some v => pure v
    | none => iconst64 0

/-- The values of value registers `rs`. -/
def packV (fl : List String) (e : RegEnv V) : (rs : List String) → Prog V L (Vals V (rs.map fun _ => ClifTy.i64))
  | [] => pure %[]
  | r :: rs => do
      let v ← e.get (L := L) fl r
      pure (v ::ᵥ (← packV fl e rs))

/-- The predicates of flags `rs`. -/
def packF (fl : List String) (e : RegEnv V) : (rs : List String) → Prog V L (Vals V (rs.map fun _ => ClifTy.i8))
  | [] => pure %[]
  | r :: rs => do
      let v ← e.getFlag (L := L) fl r
      pure (v ::ᵥ (← packF fl e rs))

def bindV (e : RegEnv V) : (rs : List String) → Vals V (rs.map fun _ => ClifTy.i64) → RegEnv V
  | [], _ => e
  | r :: rs, .cons v vs => bindV (e.setVal r v) rs vs

def bindF (e : RegEnv V) : (rs : List String) → Vals V (rs.map fun _ => ClifTy.i8) → RegEnv V
  | [], _ => e
  | r :: rs, .cons v vs => bindF (e.setFlag r v) rs vs

def appendV {a b : List ClifTy} : Vals V a → Vals V b → Vals V (a ++ b)
  | .nil, ys => ys
  | .cons x xs, ys => .cons x (appendV xs ys)

def splitV {b : List ClifTy} : (a : List ClifTy) → Vals V (a ++ b) → Vals V a × Vals V b
  | [], ys => (.nil, ys)
  | _ :: a, .cons x xs => let (l, r) := splitV a xs; (.cons x l, r)

/-- The carried registers' types: values, then flags. -/
abbrev carryTys (rv rf : List String) : List ClifTy := rv.map (fun _ => ClifTy.i64) ++ rf.map (fun _ => ClifTy.i8)

def pack (fl : List String) (e : RegEnv V) (rv rf : List String) : Prog V L (Vals V (carryTys rv rf)) := do
  pure (appendV (← packV (L := L) fl e rv) (← packF (L := L) fl e rf))

def unpack (e : RegEnv V) (rv rf : List String) (vs : Vals V (carryTys rv rf)) : RegEnv V :=
  let (a, b) := splitV _ vs
  bindF (bindV e rv a) rf b

/-- The registers a statement sets. -/
def setsOf : WStmt → List String
  | .skip | .stgB .. | .coopCopy .. => []
  | .seq a b => setsOf a ++ setsOf b
  | .bin _ d _ _ | .setp _ d _ _ | .mov d _ => [d]
  | .uif _ t e => setsOf t ++ setsOf e
  | .uwhile c b => c :: setsOf b
  | .coopWindow found p0 cand0 .. => [found, p0, cand0]
  | .coopExtendStep adv .. => [adv]

def dedup (rs : List String) : List String := rs.foldl (fun acc r => if acc.contains r then acc else acc ++ [r]) []

/-- Each register a statement sets, and whether by a comparison. -/
def assigns : WStmt → List (String × Bool)
  | .seq a b | .uif _ a b => assigns a ++ assigns b
  | .uwhile _ b => assigns b
  | .setp _ d _ _ => [(d, true)]
  | .bin _ d _ _ | .mov d _ => [(d, false)]
  | .coopWindow found p0 cand0 .. => [(found, true), (p0, false), (cand0, false)]
  | .coopExtendStep adv .. => [(adv, false)]
  | _ => []

/-- The flags: registers only comparisons set. -/
def flagRegs (s : WStmt) : List String :=
  let as := assigns s
  dedup (as.filterMap fun (r, b) => if b && !as.contains (r, false) then some r else none)

def argOf (fl : List String) (e : RegEnv V) : WArg → Prog V L (V .i64)
  | .reg n => e.get fl n
  | .imm v => iconst64 v

def opOf (o : WOp) (a b : V .i64) : Prog V L (V .i64) :=
  match o with
  | .add => iadd a b | .sub => isub a b | .mul => imul a b
  | .band => band a b | .bor => bor a b
  | .shl => ishl a b | .shr => ushr a b
  | .min => umin a b

def cmpOf : SCmp → ICmpCond
  | .eq => .eq | .ne => .ne | .lt => .ult | .le => .ule | .ge => .uge

/-- Four input bytes at `a`, little-endian. -/
def load4 (a : V .i64) : Prog V L (V .i64) := do
  let b0 ← uload8_64 a
  let b1 ← ishlImm (← uload8_64 (← iaddImm a 1)) 8
  let b2 ← ishlImm (← uload8_64 (← iaddImm a 2)) 16
  let b3 ← ishlImm (← uload8_64 (← iaddImm a 3)) 24
  bor (← bor b0 b1) (← bor b2 b3)

/-- A 16-bit shared word at offset `a`. -/
def stsh (sm a v : V .i64) : Prog V L Unit := do istore16 v (← iadd sm a)

/-- One lane's statement that holds no other: `sm` the shared base, `lane` the
    lane, `fl` the flags, `no` a false flag every test compares with. -/
def leafCode (fl : List String) (sm lane : V .i64) (no : V .i8) : WStmt → RegEnv V → Prog V L (RegEnv V)
  | .skip, e => pure e
  | .bin o d a b, e => do pure (e.setVal d (← opOf o (← e.get fl a) (← argOf fl e b)))
  | .setp c d a b, e => do
      let p ← icmp (cmpOf c) (← e.get fl a) (← argOf fl e b)
      if fl.contains d then pure (e.setFlag d p)
      else pure (e.setVal d (← select p (← iconst64 1) (← iconst64 0)))
  | .mov d a, e => do pure (e.setVal d (← argOf fl e a))
  | .stgB addr val, e => do istore8 (← e.get fl val) (← e.get fl addr); pure e
  | .coopCopy dst src len, e => do
      let d ← e.get fl dst; let s ← e.get fl src; let n ← e.get fl len
      let _ ← wloop1 (exitTys := []) (← iconst64 0)
        (fun i => pure (contIf .ult i n, %[], ()))
        (fun i _ => do
          let j ← iadd i lane
          let v ← uload8_64 (← iadd s j)
          when .ult j n do
            istore8 v (← iadd d j)
          pure %[← iaddImm i 32])
      pure e
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog _, e => do
      let inBase ← e.get fl "inBase"; let tbl ← e.get fl "tbl"
      let s ← e.get fl sp
      let posP ← iadd s lane
      let pValid ← icmp .ult posP (← iconst64 searchLim)
      let cap4 ← iconst64 (inStride - 4)
      let rp ← umin posP cap4
      let v32 ← load4 (← iadd inBase rp)
      let hh ← band (← ushrImm (← imul v32 (← iconst64 wHashK)) (32 - hashLog)) (← iconst64 (2 ^ hashLog - 1))
      let addr ← iadd (← ishlImm hh 1) tbl
      let candRaw ← uload16_64 (← iadd sm addr)
      let cand ← iaddImm candRaw (-1)
      let rc ← umin cand cap4
      let vc ← load4 (← iadd inBase rc)
      let hit ← band (← band pValid (← icmp .ne candRaw (← iconst64 0)))
        (← band (← icmp .ult cand posP) (← icmp .eq vc v32))
      let bal ← callLocal ballot %[hit]
      let first ← uextend64 (← ctz bal)
      let ins ← band (← icmp .ule lane first) pValid
      when .ne ins no do
        stsh sm addr (← iaddImm posP 1)
      callLocalVoid barrier %[]
      let c0 ← callLocal shflIdx %[← ireduce32 cand, ← ireduce32 first]
      let e := e.setVal p0 (← iadd s first)
      let e := e.setVal cand0 (← uextend64 c0)
      let fd ← icmp .ne bal (← iconst32 0)
      if fl.contains found then pure (e.setFlag found fd)
      else pure (e.setVal found (← select fd (← iconst64 1) (← iconst64 0)))
  | .coopExtendStep adv p0 cand0 ml _ _, e => do
      let inBase ← e.get fl "inBase"
      let pe ← iadd (← e.get fl p0) (← iadd (← e.get fl ml) lane)
      let pIn ← icmp .ult pe (← e.get fl "ecR")
      let ec1 ← e.get fl "ec1"
      let peC ← umin pe ec1
      let caC ← iadd (← e.get fl cand0) (← isub peC (← e.get fl p0))
      let bP ← uload8_64 (← iadd inBase peC)
      let bC ← uload8_64 (← iadd inBase caC)
      let ok ← band pIn (← icmp .eq bP bC)
      let bal ← callLocal ballot %[ok]
      pure (e.setVal adv (← uextend64 (← ctz (← bxor bal (← iconst32 (-1))))))
  | .seq .., e | .uif .., e | .uwhile .., e => pure e

/-- A statement of `skip`, operations, comparisons and moves alone. -/
def straight : WStmt → Bool
  | .skip | .bin .. | .setp .. | .mov .. => true
  | .seq a b => straight a && straight b
  | _ => false

/-- Straight statements' code. -/
def straightCode (fl : List String) (sm lane : V .i64) (no : V .i8) : WStmt → RegEnv V → Prog V L (RegEnv V)
  | .seq a b, e => do straightCode fl sm lane no b (← straightCode fl sm lane no a e)
  | s, e => leafCode fl sm lane no s e

/-- Where a loop's body ends: the loop's carries `rv` and `rf`, its label, and
    the straight statements still to run before going round. -/
structure Tail (L : List ClifTy → List ClifTy → Type) where
  rv : List String
  rf : List String
  lp : L (carryTys rv rf) (carryTys rv rf)
  post : WStmt

/-- Going round a loop from its body's end: the tail's statements, then a jump
    to the loop's head with the carries. -/
def goRound (fl : List String) (sm lane : V .i64) (no : V .i8) (t : Tail L) (e : RegEnv V) :
    Prog V L (RegEnv V) := do
  let e ← straightCode fl sm lane no t.post e
  .cont t.lp (← pack fl e t.rv t.rf)

/-- **One lane's statement**, at the end of a loop's body where `tl` says so:
    there it goes round the loop itself, each arm of a branch it ends with
    running the straight statements after it and jumping to the loop's head,
    with no join between them. -/
def codeOfT (fl : List String) (sm lane : V .i64) (no : V .i8) :
    Option (Tail L) → WStmt → RegEnv V → Prog V L (RegEnv V)
  | none, .seq a b, e => do codeOfT fl sm lane no none b (← codeOfT fl sm lane no none a e)
  | some t, .seq a b, e =>
      if straight b then codeOfT fl sm lane no (some { t with post := .seq b t.post }) a e
      else do codeOfT fl sm lane no (some t) b (← codeOfT fl sm lane no none a e)
  | none, .uif cond t f, e => do
      let rs := dedup (setsOf t ++ setsOf f)
      let rv := rs.filter (!fl.contains ·); let rf := rs.filter (fl.contains ·)
      let c ← e.getFlag fl cond
      let out ← ifte (jTys := carryTys rv rf) .ne c no
        (do pack fl (← codeOfT fl sm lane no none t e) rv rf)
        (do pack fl (← codeOfT fl sm lane no none f e) rv rf)
      pure (unpack e rv rf out)
  | some tl, .uif cond t f, e => do
      let c ← e.getFlag fl cond
      let _ ← ifte (jTys := []) .ne c no
        (do let _ ← codeOfT fl sm lane no (some tl) t e; pure %[])
        (do let _ ← codeOfT fl sm lane no (some tl) f e; pure %[])
      pure e
  | tl, .uwhile cond body, e => do
      let rs := dedup (cond :: setsOf body)
      let rv := rs.filter (!fl.contains ·); let rf := rs.filter (fl.contains ·)
      let init ← pack fl e rv rf
      let out ← wloopL (exitTys := carryTys rv rf) init
        (fun _ vs => do
          let c ← (unpack e rv rf vs).getFlag fl cond
          pure (contIf .ne c no, vs, ()))
        (fun lp vs _ => do
          let _ ← codeOfT fl sm lane no (some ⟨rv, rf, lp, .skip⟩) body (unpack e rv rf vs)
          pure vs)
      match tl with
      | none => pure (unpack e rv rf out)
      | some t => goRound fl sm lane no t (unpack e rv rf out)
  | none, s, e => leafCode fl sm lane no s e
  | some t, s, e => do goRound fl sm lane no t (← leafCode fl sm lane no s e)

/-- One lane's statement. -/
abbrev codeOf (fl : List String) (sm lane : V .i64) (no : V .i8) : WStmt → RegEnv V → Prog V L (RegEnv V) :=
  codeOfT fl sm lane no none

/-- The compressor: each warp's input block, at `inStride` apart from the
    first parameter, compressed to its output block, the output region placed
    after every input block (`copySlack` past them), as the shipped kernel
    does. -/
def compressor (numBlocks inStride outStride lenOff hashLog : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let cta ← uextend64 ps.head; let tid ← uextend64 ps.snd; let ntid ← uextend64 ps.thd
  let sm := (ps.drop 3).head
  let inP := (ps.drop 4).head
  let gwarp ← ushrImm (← iadd (← imul cta ntid) tid) 5
  let lane ← band tid (← iconst64 31)
  when .ult gwarp (← iconst64 numBlocks) do
    let outP ← iaddImm inP (numBlocks * inStride + copySlack)
    let inBase ← iadd inP (← imul gwarp (← iconst64 inStride))
    let outBase ← iadd outP (← imul gwarp (← iconst64 outStride))
    -- one warp a block: its table is the block's shared memory
    let tbl ← iconst64 0
    -- the table cleared, each lane its share
    let z ← iconst64 0
    let _ ← wloop1 (exitTys := []) lane
      (fun ci => do pure (contIf .ult ci (← iconst64 (2 ^ hashLog)), %[], ()))
      (fun ci _ => do
        stsh sm (← band (← iadd tbl (← ishlImm ci 1)) (← iconst64 ((2 * 2 ^ hashLog - 1 : Nat) : Int))) z
        pure %[← iaddImm ci 32])
    callLocalVoid barrier %[]
    let e : RegEnv V := { vals := [("inBase", inBase), ("outBase", outBase), ("tbl", tbl),
      ("op", z), ("searchPos", z), ("litAnchor", z)] }
    let body := compressorBodyWStmt inStride lenOff hashLog
    let no ← iconst .i8 0
    let _ ← codeOf (flagRegs body) sm lane no body e
    pure ()

end

open AlgorithmLib.Kernel in
/-- The compressor's kernel: the input and output pointers after `idParams`,
    the table (`2 ^ hashLog` 16-bit entries) in shared memory, each block one
    warp. -/
def codeKernel (name : String) (numBlocks inStride outStride lenOff hashLog : Nat) : Kernel :=
  { name, params := [.i64, .i64],
    code := emit (compressor numBlocks inStride outStride lenOff hashLog) (idParams ++ [.i64, .i64]),
    smem := 2 ^ hashLog * 2, warp := true }

end Lz4Code
