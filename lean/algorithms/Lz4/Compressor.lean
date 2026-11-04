module
public import AlgorithmLib.LZ4.WarpKernel
public import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Kernel.Ptx
public import AlgorithmLib.ML.Ptx.Body
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.Compressor` — the compressor in the warp language

`ofW` writes LZ4's warp statement (`LZ4WarpDSL.WStmt`) in the warp language
(`ML.EWStmt`): a register is an integer register, numbered by `R`; an
operation, comparison or move sets it, a byte store is every lane's, a branch
or loop tests a predicate register as itself and any other as equal to one,
and a loop carries the registers it and its body set, at most `F` trips
(`2 ^ 64` in the kernel: more than any 64-bit count needs). A
cooperative statement is the lanes' share of it, its temporaries local to it
(`T` on): the copy each lane's bytes `lane, lane + 32, …`; the window each
lane's probe at `sp + lane`, the ballot of hits, the lowest hit's position,
the inserts up to it, the barrier and the lowest hit's candidate shuffled from
its lane; an extend step each lane's byte compare and the ballot of
mismatches. The lane is read into a register once.

`compressorS` is a block's thread: the block's input and output bases, its
share of clearing the table (entries `lane, lane + 32, …`), the barrier, then
the body. `compressorKernel` compiles it through the body lowering
(`ML.Body`), one warp a block, the table in shared memory.
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog AlgorithmLib.LZ4WarpDSL
  AlgorithmLib.ML AlgorithmLib.Kernel

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

/-- Every register name a statement mentions, the cooperative statements'
    implicit ones too. -/
def namesOf : WStmt → List String
  | .skip => []
  | .seq a b => namesOf a ++ namesOf b
  | .bin _ d a b | .setp _ d a b => d :: a :: (match b with | .reg n => [n] | _ => [])
  | .mov d a => d :: (match a with | .reg n => [n] | _ => [])
  | .stgB a v => [a, v]
  | .uif c t e => c :: namesOf t ++ namesOf e
  | .uwhile c b => c :: namesOf b
  | .coopCopy d s n => [d, s, n]
  | .coopWindow f p c sp .. => [f, p, c, sp, "inBase", "tbl"]
  | .coopExtendStep a p c m .. => [a, p, c, m, "inBase", "ecR", "ec1"]

def opW : WOp → IOp
  | .add => .add | .sub => .sub | .mul => .mul | .band => .band | .bor => .bor
  | .shl => .shl | .shr => .shr | .min => .umin

def cmpW : SCmp → ICmp
  | .eq => .eq | .ne => .ne | .lt => .ult | .le => .ule | .ge => .uge

/-- Statements in order. -/
def seqs : List EWStmt → EWStmt
  | [] => .skip
  | [s] => s
  | s :: r => .seq s (seqs r)

/-- The 64-bit word of an integer, as a literal. -/
def litI (k : Int) : IExp := .lit (k.emod (1 <<< 64)).toNat

section
variable (R : String → Nat) (T : Nat) (F : Nat)

/-- The register holding the lane. -/
def laneR : Nat := T + 41

def argW : WArg → IExp
  | .reg n => .reg (R n)
  | .imm v => .lit v

/-- A copy of `n` bytes from `s` to `d`, each lane its bytes `lane, lane + 32, …`. -/
def copyE (d s n : Nat) : EWStmt :=
  let i := T; let j := T + 1; let v := T + 2
  .locals [i, j, v] (seqs [
    .setI i (.lit 0),
    .iwhile F (.cmp .ult (.reg i) (.reg n)) [i] (seqs [
      .setI j (.bin .add (.reg i) (.reg (laneR T))),
      .ldB v (.bin .add (.reg s) (.reg j)),
      .stB (.bin .add (.reg d) (.reg j)) (.reg v) (.cmp .ult (.reg j) (.reg n)),
      .setI i (.bin .add (.reg i) (.lit 32)) ]) ])

/-- Four bytes at `a`, little-endian, into `d`, through temporaries `t … t + 4`. -/
def load4E (d t : Nat) (a : IExp) : List EWStmt :=
  [ .setI t a,
    .ldB (t + 1) (.reg t),
    .ldB (t + 2) (.bin .add (.reg t) (.lit 1)),
    .setI (t + 2) (.bin .shl (.reg (t + 2)) (.lit 8)),
    .ldB (t + 3) (.bin .add (.reg t) (.lit 2)),
    .setI (t + 3) (.bin .shl (.reg (t + 3)) (.lit 16)),
    .ldB (t + 4) (.bin .add (.reg t) (.lit 3)),
    .setI (t + 4) (.bin .shl (.reg (t + 4)) (.lit 24)),
    .setI d (.bin .bor (.bin .bor (.reg (t + 1)) (.reg (t + 2))) (.bin .bor (.reg (t + 3)) (.reg (t + 4)))) ]

/-- The window's probe: each lane's position `sp + lane` and whether it is
    below the search limit, its four bytes, their hash's table entry and the
    candidate it names, and the candidate's four bytes. Its temporaries are
    `T + 3 …` (`posP`, `pValid`, `rp`, `v32`, `hh`, `addr`, `candRaw`, `cand`,
    `rc`, `vc`, then the four-byte loads' `T + 17 …`). -/
def winPre (sp inStride searchLim hashLog : Nat) : List EWStmt :=
  let b := T + 3
  let posP := b; let pValid := b + 1; let rp := b + 2; let v32 := b + 3; let hh := b + 4
  let addr := b + 5; let candRaw := b + 6; let cand := b + 7; let rc := b + 8; let vc := b + 9
  let t := b + 14
  [ .setI posP (.bin .add (.reg sp) (.reg (laneR T))),
    .setI pValid (.cmp .ult (.reg posP) (litI searchLim)),
    .setI rp (.bin .umin (.reg posP) (litI (inStride - 4))) ] ++
    load4E v32 t (.bin .add (.reg (R "inBase")) (.reg rp)) ++ [
    .setI hh (.bin .band (.bin .shr (.bin .mul (.reg v32) (litI wHashK)) (litI (32 - hashLog)))
      (litI (2 ^ hashLog - 1))),
    .setI addr (.bin .add (.bin .shl (.reg hh) (.lit 1)) (.reg (R "tbl"))),
    .ldH candRaw (.reg addr),
    .setI cand (.bin .add (.reg candRaw) (litI (-1))),
    .setI rc (.bin .umin (.reg cand) (litI (inStride - 4))) ] ++
    load4E vc t (.bin .add (.reg (R "inBase")) (.reg rc))

/-- The window's choice: the hit, the ballot of hits and its lowest lane, the
    inserts up to it, the barrier, the lowest hit's candidate shuffled from its
    lane, and the window's results. -/
def winSuf (found p0 cand0 sp : Nat) : List EWStmt :=
  let b := T + 3
  let posP := b; let pValid := b + 1; let v32 := b + 3
  let addr := b + 5; let candRaw := b + 6; let cand := b + 7; let vc := b + 9
  let hit := b + 10; let bal := b + 11; let first := b + 12; let c0 := b + 13
  [ .setI hit (.bin .band (.bin .band (.reg pValid) (.cmp .ne (.reg candRaw) (.lit 0)))
      (.bin .band (.cmp .ult (.reg cand) (.reg posP)) (.cmp .eq (.reg vc) (.reg v32)))),
    .ballot bal (.reg hit),
    .setI first (.ctz (.reg bal)),
    .stH (.reg addr) (.bin .add (.reg posP) (.lit 1))
      (.bin .band (.cmp .ule (.reg (laneR T)) (.reg first)) (.reg pValid)),
    .barrier,
    .shflIdx c0 (.reg cand) (.reg first),
    .setI p0 (.bin .add (.reg sp) (.reg first)),
    .setI cand0 (.reg c0),
    .setI found (.cmp .ne (.reg bal) (.lit 0)) ]

/-- The window: each lane probes `sp + lane`, the lowest hit wins, the lanes up
    to it insert their positions. -/
def windowE (found p0 cand0 sp inStride searchLim hashLog : Nat) : EWStmt :=
  .locals (List.range' (T + 3) 19)
    (seqs (winPre R T sp inStride searchLim hashLog ++ winSuf T found p0 cand0 sp))

/-- An extend step: each lane compares its byte of the match. -/
def extendE (adv p0 cand0 ml : Nat) : EWStmt :=
  let b := T + 23
  let pe := b; let pIn := b + 1; let peC := b + 2; let caC := b + 3; let bP := b + 4; let bC := b + 5
  let ok := b + 6; let bal := b + 7
  let inBase := R "inBase"
  .locals (List.range' b 8) (seqs [
    .setI pe (.bin .add (.reg p0) (.bin .add (.reg ml) (.reg (laneR T)))),
    .setI pIn (.cmp .ult (.reg pe) (.reg (R "ecR"))),
    .setI peC (.bin .umin (.reg pe) (.reg (R "ec1"))),
    .setI caC (.bin .add (.reg cand0) (.bin .sub (.reg peC) (.reg p0))),
    .ldB bP (.bin .add (.reg inBase) (.reg peC)),
    .ldB bC (.bin .add (.reg inBase) (.reg caC)),
    .setI ok (.bin .band (.reg pIn) (.cmp .eq (.reg bP) (.reg bC))),
    .ballot bal (.reg ok),
    .setI adv (.ctz (.bin .bxor (.reg bal) (.lit 0xffffffff))) ])

/-- A register's test as `uif` and `uwhile` make it: a predicate register
    itself, any other equal to one. -/
def testW (fl : List String) (c : String) : IExp :=
  if fl.contains c then .reg (R c) else .cmp .eq (.reg (R c)) (.lit 1)

/-- **LZ4's warp statement in the warp language**, its predicate registers `fl`. -/
def ofW (fl : List String) : WStmt → EWStmt
  | .skip => .skip
  | .seq a b => .seq (ofW fl a) (ofW fl b)
  | .bin o d a b => .setI (R d) (.bin (opW o) (.reg (R a)) (argW R b))
  | .setp c d a b => .setI (R d) (.cmp (cmpW c) (.reg (R a)) (argW R b))
  | .mov d a => .setI (R d) (argW R a)
  | .stgB a v => .stB (.reg (R a)) (.reg (R v)) (.lit 1)
  | .uif c t e => .iif (testW R fl c) (ofW fl t) (ofW fl e)
  | .uwhile c b => .iwhile F (testW R fl c) ((dedup (c :: setsOf b)).map R) (ofW fl b)
  | .coopCopy d s n => copyE T F (R d) (R s) (R n)
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog _ =>
      windowE R T (R found) (R p0) (R cand0) (R sp) inStride searchLim hashLog
  | .coopExtendStep adv p0 cand0 ml _ _ => extendE R T (R adv) (R p0) (R cand0) (R ml)

end

/-- The registers of the compressor's body, in order. -/
def bodyNames (inStride lenOff hashLog : Nat) : List String :=
  dedup (["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] ++
    namesOf (compressorBodyWStmt inStride lenOff hashLog))

/-- **A block's thread of the compressor**: where its block is one, the
    block's input and output bases, its share of clearing the table, the
    barrier and the body, every register local. -/
def compressorS (numBlocks inStride outStride lenOff hashLog : Nat) : EWStmt :=
  let body := compressorBodyWStmt inStride lenOff hashLog
  let ns := bodyNames inStride lenOff hashLog
  let R : String → Nat := fun x => ns.idxOf x
  let T := ns.length
  let ci := T + 40
  .iif (.cmp .ult .cta (.lit numBlocks)) (.locals (List.range (T + 42)) (seqs [
    .setI (laneR T) .lane,
    .setI (R "inBase") (.bin .add (.ptr 0) (.bin .mul .cta (.lit inStride))),
    .setI (R "outBase") (.bin .add (.bin .add (.ptr 0) (.lit (numBlocks * inStride + copySlack)))
      (.bin .mul .cta (.lit outStride))),
    .setI (R "tbl") (.lit 0),
    .setI ci (.reg (laneR T)),
    .iwhile (2 ^ 64) (.cmp .ult (.reg ci) (.lit (2 ^ hashLog))) [ci] (seqs [
      .stH (.bin .band (.bin .add (.reg (R "tbl")) (.bin .shl (.reg ci) (.lit 1))) (.lit (2 * 2 ^ hashLog - 1)))
        (.lit 0) (.lit 1),
      .setI ci (.bin .add (.reg ci) (.lit 32)) ]),
    .barrier,
    ofW R T (2 ^ 64) (flagRegs body) body ])) .skip

/-- **The compressor's kernel**: the input and output pointers after
    `idParams`, the table (`2 ^ hashLog` 16-bit entries) in shared memory, each
    block one warp. -/
def compressorKernel (name : String) (numBlocks inStride outStride lenOff hashLog : Nat) : Kernel :=
  { name, params := [.i64, .i64],
    code := emit (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog))
      (idParams ++ [.i64, .i64]),
    smem := 2 ^ hashLog * 2, warp := true }

end Lz4Code
