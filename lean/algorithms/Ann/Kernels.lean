module
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The approximate-nearest-neighbour kernels

The device half of base's ANN engine: an IVF index whose rows are int8 codes
of their residuals from their list's centre, scanned by one fused kernel per
(list, tile of queries), the candidates re-ranked exactly.

These are kernel bodies (`Body`), compiled by the PTX compiler; what they
compute is checked against reference outputs, not yet proven.

The engine uses no atomic operation: every place a GPU engine usually takes
one (a histogram, a buffer's next free slot, a sum per centre) is a ballot,
a shuffle-scan or a sort here, so a run's result is a function of its inputs.
-/

set_option maxRecDepth 8000

namespace AnnKernels

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- `+∞` as `Float32` bits. -/
def posInfBits : UInt32 := 0x7F800000

/-- The warps of a 256-thread block. -/
def WARPS : Nat := 8

/-- A float from lane `lane xor m`. -/
def shflF (x : V .f32) (m : Nat) : Prog V L (V .f32) := do
  bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 x, ← c32 m])

/-- A word from lane `lane xor m`. -/
def shflI (x : V .i32) (m : Nat) : Prog V L (V .i32) := do
  callLocal shflBfly %[x, ← c32 m]

/-- Whether `(ka, va)` sorts before `(kb, vb)`: the smaller key, the smaller
    value on equal keys. -/
def before (ka : V .f32) (va : V .i32) (kb : V .f32) (vb : V .i32) : Prog V L (V .i8) := do
  let lt ← bit (← fcmp .lt ka kb)
  let eq ← bit (← fcmp .eq ka kb)
  let vlt ← bit (← icmp .slt va vb)
  icmp .ne (← bor lt (← band eq vlt)) (← c32 0)

/-- `n` registers as a loop's carries, from `xs` (`d` past its end). -/
def packFrom {t} (xs : Array (V t)) (d : V t) (i : Nat) : (n : Nat) → Vals V (List.replicate n t)
  | 0 => .nil
  | n + 1 => .cons (xs.getD i d) (packFrom xs d (i + 1) n)

def pack {t} (xs : Array (V t)) (d : V t) (n : Nat) : Vals V (List.replicate n t) := packFrom xs d 0 n

/-- A loop's `n` carries of one type, as registers. -/
def unpack {t} : (n : Nat) → Vals V (List.replicate n t) → Array (V t)
  | 0, _ => #[]
  | n + 1, .cons v vs => #[v] ++ unpack n vs

/-- The first `as`, and the rest. -/
def splitV : {as bs : List ClifTy} → Vals V (as ++ bs) → Vals V as × Vals V bs
  | [], _, vs => (.nil, vs)
  | _ :: _, _, .cons v vs => let (a, b) := splitV vs; (.cons v a, b)

/-- Two arrays of `n` registers, carried by a loop. -/
def unpack2 {a b} (n : Nat) (vs : Vals V (List.replicate n a ++ List.replicate n b)) :
    Array (V a) × Array (V b) :=
  let (x, y) := splitV vs; (unpack n x, unpack n y)

/-- Register `j` of `xs`, `j` a value alike across the warp. -/
def pick {t} (j : V .i32) (xs : Array (V t)) (d : V t) : Prog V L (V t) := do
  let mut v := xs.getD 0 d
  for i in (List.range xs.size).drop 1 do
    v ← select (← icmp .eq j (← c32 i)) (xs.getD i d) v
  pure v

/-- `xs` with register `j` replaced by `x`. -/
def place {t} (j : V .i32) (xs : Array (V t)) (x : V t) : Prog V L (Array (V t)) := do
  let mut ys := xs
  for i in List.range xs.size do
    ys := ys.set! i (← select (← icmp .eq j (← c32 i)) x (xs.getD i x))
  pure ys

/-- The direction a merge step orders a pair of registers: fixed, or
    ascending where the lane's flag holds. -/
inductive Dir (V : ClifTy → Type) where
  | asc | desc | lane (up : V .i8)

/-- One step of a merge within a lane: registers `r` and `r + stride`
    ordered, `r` the one whose bit `stride` is clear, in the direction `dir r`
    gives. -/
def inLane (per stride : Nat) (ks : Array (V .f32)) (vs : Array (V .i32)) (kd : V .f32) (vd : V .i32)
    (dir : Nat → Prog V L (Dir V)) : Prog V L (Array (V .f32) × Array (V .i32)) := do
  let mut ks := ks
  let mut vs := vs
  for r in List.range per do
    if r &&& stride == 0 then
      let r2 := r + stride
      let a := ks.getD r kd; let av := vs.getD r vd; let b := ks.getD r2 kd; let bv := vs.getD r2 vd
      let lt ← before a av b bv
      let lo ← select lt a b; let lov ← select lt av bv
      let hi ← select lt b a; let hiv ← select lt bv av
      match ← dir r with
      | .asc => ks := (ks.set! r lo).set! r2 hi; vs := (vs.set! r lov).set! r2 hiv
      | .desc => ks := (ks.set! r hi).set! r2 lo; vs := (vs.set! r hiv).set! r2 lov
      | .lane up =>
          ks := (ks.set! r (← select up lo hi)).set! r2 (← select up hi lo)
          vs := (vs.set! r (← select up lov hiv)).set! r2 (← select up hiv lov)
  pure (ks, vs)

/-- **A warp's bitonic sort in registers**: `32 · per` (key, value) pairs,
    element `lane · per + r` in register `r` of lane `lane`, ascending by
    `before`. A stride below `per` exchanges within a lane, a larger one with
    lane `lane xor (stride / per)` by a shuffle, so the warp needs no shared
    memory and no barrier. `per` is a power of two. -/
def warpSort (lane : V .i32) (kd : V .f32) (vd : V .i32) (ks : Array (V .f32)) (vs : Array (V .i32)) :
    Prog V L (Array (V .f32) × Array (V .i32)) := do
  let per := ks.size
  let n := 32 * per
  let mut ks := ks
  let mut vs := vs
  let sizes := (List.range 13).map (2 ^ ·) |>.filter (fun s => 2 ≤ s ∧ s ≤ n)
  for size in sizes do
    let upLane : Option (V .i8) ← if size < per then pure none else do
      let bitv ← band lane (← c32 (size / per))
      pure (some (← icmp .eq bitv (← c32 0)))
    let strides := (List.range 13).map (2 ^ ·) |>.filter (fun s => s < size) |>.reverse
    for stride in strides do
      if stride < per then
        (ks, vs) ← inLane per stride ks vs kd vd (fun r => match upLane with
          | none => pure (if r &&& size == 0 then .asc else .desc)
          | some up => pure (.lane up))
      else
        let m := stride / per
        let lower ← icmp .eq (← band lane (← c32 m)) (← c32 0)
        let up ← match upLane with
          | none => icmp .eq (← c32 0) (← c32 0)
          | some u => pure u
        let takeMin ← icmp .eq (← bit lower) (← bit up)
        for r in List.range per do
          let a := ks.getD r kd; let av := vs.getD r vd
          let b ← shflF a m; let bv ← shflI av m
          let lt ← before a av b bv
          let mn ← select lt a b; let mnv ← select lt av bv
          let mx ← select lt b a; let mxv ← select lt bv av
          ks := ks.set! r (← select takeMin mn mx)
          vs := vs.set! r (← select takeMin mnv mxv)
  pure (ks, vs)

/-- **The nearest centre of each row**: of the `rows × cols` tile `m2dot`
    (row `i` holds `-2 x_i · c_j` for every centre `j`), the index of the
    smallest `cn[j] + m2dot[i, j]`, the lowest index on a tie (as
    `argmin`). A warp a row: each lane strides the row keeping its best, then
    a butterfly keeps the better of each pair, the lower index on a tie.
    `meta` holds `rows` and `cols` as words. -/
def nearestBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd
  let m2 := (ps.drop 4).head; let cn := (ps.drop 5).head
  let out := (ps.drop 6).head; let m := (ps.drop 7).head
  let rows ← load32 m
  let cols ← load32 (← at_ m 4)
  let lane ← band tid (← c32 31)
  let row ← iadd (← imul cta (← c32 WARPS)) (← ushrImm tid 5)
  when .ult row rows do
    let rowBase ← iadd m2 (← ishlImm (← imul (← ext row) (← ext cols)) 2)
    let sentinel ← c32 0x7fffffff
    let r ← wloop %[lane, ← f32c posInfBits, sentinel] (fun cs => pure (contIfULt cs.head cols, %[cs.snd, cs.thd], ()))
      (fun cs _ => do
        let j := cs.head
        let v ← fadd (← loadF32 (← at4 cn j)) (← loadF32 (← at4 rowBase j))
        let lt ← fcmp .lt v cs.snd
        pure %[← iadd j (← c32 32), ← select lt v cs.snd, ← select lt j cs.thd])
    let mut best := r.head
    let mut ix := r.snd
    for k in [16, 8, 4, 2, 1] do
      let b ← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 best, ← c32 k])
      let a ← callLocal shflBfly %[ix, ← c32 k]
      let lt ← fcmp .lt b best
      let eq ← fcmp .eq b best
      let lo ← umin a ix
      ix ← select lt a (← select eq lo ix)
      best ← select lt b best
    when .eq lane (← c32 0) do
      let fixed ← select (← icmp .eq ix sentinel) (← c32 0) ix
      storeI32 fixed (← at4 out row)

/-! ## The scan -/

/-- Candidate entries a query's buffer holds (the scan is built at two:
    `128` for `R ≤ 96`, `256` for `R ≤ 224`, a buffer cut to `R` taking 32
    more), the queries a warp scores at a time, and the warps a list's queries
    are shared among. -/
def CAP : Nat := 128
def CAP2 : Nat := 256
def QPW : Nat := 4
def QW : Nat := 4

/-- The scan's shared memory for `dw` code words a row (one warp's): its
    queries' codes (`dw` words each), then their candidate buffers (keys,
    then rows). -/
def sqOff : Nat := 0
def bkeyOff (dw : Nat) (qpw : Nat := QPW) : Nat := sqOff + qpw * dw * 4
def bvalOff (dw cap : Nat) (qpw : Nat := QPW) : Nat := bkeyOff dw qpw + qpw * cap * 4
def scanSmem (dw cap : Nat) (qpw : Nat := QPW) : Nat := bvalOff dw cap qpw + qpw * cap * 4

/-- The warp's sort of buffer `q` of `cap` entries, `cap / 32` a lane; the
    sorted keys and rows, in the lane's registers. -/
def sortBuffer (cap : Nat) (lane : V .i32) (kd : V .f32) (vd : V .i32) (bkey bval : V .i64) (q : V .i32) :
    Prog V L (Array (V .f32) × Array (V .i32)) := do
  let per := cap / 32
  let base ← iadd (← imul q (← c32 cap)) (← imul lane (← c32 per))
  let mut ks : Array (V .f32) := #[]
  let mut vs : Array (V .i32) := #[]
  for r in List.range per do
    let e ← iadd base (← c32 r)
    ks := ks.push (← loadF32 (← at4 bkey e))
    vs := vs.push (← load32 (← at4 bval e))
  warpSort lane kd vd ks vs


/-- Element `e` of a warp's sorted `32 · per`, `per` a lane (a power of
    two): every lane gets it. -/
def bcastSorted (per : Nat) (ks : Array (V .f32)) (kd : V .f32) (e : V .i32) : Prog V L (V .f32) := do
  let r ← band e (← c32 (per - 1))
  let mut v := kd
  for i in List.range per do
    v ← select (← icmp .eq r (← c32 i)) (ks.getD i kd) v
  bitcast .f32 (← callLocal shflIdx %[← bitcast .i32 v, ← udiv e (← c32 per)])

/-- **The fused scan**, a block of one warp per (list, slot): block `b` takes
    list `b / QW`'s pairs `p0 + s + QW · j` (`j < QPW`, those that exist),
    then the next `QW · QPW`, …, `s = b % QW`, for the `dw`-word codes of an
    artifact's dimension.

    Parameters: the codes (word `w` of row `r` at `((r / 32) · dw + w) · 32 +
    r % 32`), each row's `‖x‖²` term, each list's first and past-last row,
    the queries' int8 codes (`dw` words each) and signed scales, each list's
    first pair and past its last (pairs sorted by list), each pair's query, its
    centre term and its output slot, the output keys and rows (`R` a slot),
    each laid-out row's id, the live bitset over ids, each query's admitted
    bitset over ids (a filtered search), and the meta words (`R`, whether to
    skip rows whose id is not live, and the admitted bitsets' words a query,
    `0` for an unfiltered search).

    The warp scores 32 of the list's rows at a time against its queries, a
    lane a row, four code bytes against four of a query's by one `dp4a`; a row
    whose key beats its query's threshold is appended to the query's buffer
    at the slot a ballot gives it, so a buffer's order is a function of the
    rows alone. A buffer that might not take the next 32 is sorted in
    registers and cut to `R`, its threshold raised to the `R`-th key. Counts
    and thresholds live in registers; a list's queries share no state between
    warps, so the only barriers are the warp's own. A query the warp does not
    have is skipped by a branch alike across the warp. At the end each buffer
    is sorted and its best `R` written to the pair's slot, `+∞` and `-1` past
    the candidates found. -/
def scanBody (dw cap : Nat) (qpw : Nat := QPW) (qw : Nat := QW) (own : Option Bool := none) :
    Prog V L Unit := do
  let per := cap / 32
  let ps ← entryParams (idParams ++ List.replicate 17 .i64)
  let cta := ps.head; let lane := ps.snd; let sm := (ps.drop 3).head
  let codes := (ps.drop 4).head; let rterm := (ps.drop 5).head
  let loff := (ps.drop 6).head; let lend := (ps.drop 7).head
  let tq := (ps.drop 8).head; let tqs := (ps.drop 9).head
  let pst := (ps.drop 10).head; let pen := (ps.drop 11).head; let pq := (ps.drop 12).head
  let pbase := (ps.drop 13).head; let pout := (ps.drop 14).head
  let okey := (ps.drop 15).head; let orow := (ps.drop 16).head
  let lid := (ps.drop 17).head; let live := (ps.drop 18).head; let qbits := (ps.drop 19).head
  let mt := (ps.drop 20).head
  let rR ← load32 mt
  let checkLive ← load32 (← at_ mt 4)
  let qwords ← load32 (← at_ mt 8)
  let unfiltered ← bit (← icmp .eq qwords (← c32 0))
  let unchecked ← bit (← icmp .eq checkLive (← c32 0))
  let l ← udiv cta (← c32 qw)
  let slot ← isub cta (← imul l (← c32 qw))
  let sq ← at_ sm sqOff
  let bkey ← at_ sm (bkeyOff dw qpw); let bval ← at_ sm (bvalOff dw cap qpw)
  let inf ← f32c posInfBits
  let neg1 ← c32 0xFFFFFFFF
  let zero ← c32 0
  -- the block's list and pairs: its own, or (`own`) piece `l` of pair
  -- `l / split`, the `l % split`-th of the pair's list's rows, alone
  let (p0, p1, r0, r1, ownQ, ownBase) ← match own with
    | none => do
        pure (← load32 (← at4 pst l), ← load32 (← at4 pen l), ← load32 (← at4 loff l),
          ← load32 (← at4 lend l), (none : Option (V .i32)), (none : Option (V .f32)))
    | some l2 => do
        let nprobe ← load32 (← at_ mt 12)
        let lists ← load32 (← at_ mt 16)
        let split ← load32 (← at_ mt 20)
        let i ← udiv l split
        let s ← isub l (← imul i split)
        let li ← load32 (← at4 pst i)
        let q ← udiv i nprobe
        let ra ← load32 (← at4 loff li)
        let len ← isub (← load32 (← at4 lend li)) ra
        let ck ← loadF32 (← at4 pen (← iadd (← imul q lists) li))
        let base ← if l2 then fsub ck (← loadF32 (← at4 pq li)) else pure ck
        pure (l, ← iadd l (← c32 1), ← iadd ra (← udiv (← imul len s) split),
          ← iadd ra (← udiv (← imul len (← iadd s (← c32 1))) split), some q, some base)
  let trips ← udiv (← iadd (← isub r1 r0) (← c32 31)) (← c32 32)
  let ltMask ← isub (← ishl (← c32 1) lane) (← c32 1)
  let rLast ← isub rR (← c32 1)
  let _ ← wloop %[← iadd p0 slot] (fun cs => pure (contIfULt cs.head p1, %[], ()))
    (fun cs _ => do
      let pa := cs.head
      -- this warp's queries: pair `pa + qw · j` where it exists (else `pa`)
      let mut on : Array (V .i32) := #[]
      let mut pp : Array (V .i32) := #[]
      for j in List.range qpw do
        let raw ← iadd pa (← c32 (qw * j))
        let has ← bit (← icmp .ult raw p1)
        on := on.push has
        pp := pp.push (← select (← icmp .ne has zero) raw pa)
      let qs ← pp.mapM fun p => match ownQ with
        | some q => pure q
        | none => do load32 (← at4 pq p)
      for j in List.range qpw do
        let q := qs.getD j zero
        stridedDo lane (← c32 dw) 32 fun w => do
          storeI32 (← load32 (← at4 tq (← iadd (← imul q (← c32 dw)) w))) (← at4 sq (← iadd w (← c32 (j * dw))))
      stridedDo lane (← c32 (qpw * cap)) 32 fun i => do
        store inf (← at4 bkey i)
        storeI32 neg1 (← at4 bval i)
      let bases ← pp.mapM fun p => match ownBase with
        | some b => pure b
        | none => do loadF32 (← at4 pbase p)
      let scales ← qs.mapM fun q => do loadF32 (← at4 tqs q)
      callLocalVoid barrier %[]
      -- carried: trip, then each query's count, then each one's threshold
      let init := %[← c32 0] |>.append (pack #[] zero qpw |>.append (pack #[] inf qpw))
      let r ← wloop init (fun cs => pure (contIfULt cs.head trips, %[], ()))
        (fun cs _ => do
          let (cnts, thrs) := unpack2 qpw cs.tail
          let row ← iadd (← iadd r0 (← ishlImm cs.head 5)) lane
          let inList ← icmp .ult row r1
          let rowc ← select inList row (← isub r1 (← c32 1))
          let id ← load32 (← at4 lid rowc)
          let liveBit ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
          let ok ← band (← bit inList) (← bor liveBit unchecked)
          let idw ← ushrImm id 5
          let idb ← band id (← c32 31)
          let cbase ← iadd (← imul (← ushrImm rowc 5) (← c32 (dw * 32))) (← band rowc (← c32 31))
          let mut acc : Array (V .i32) := Array.replicate qpw zero
          for w in List.range dw do
            let cw ← load32 (← at4 codes (← iadd cbase (← c32 (w * 32))))
            for j in List.range qpw do
              let a0 := acc.getD j zero
              let step : Prog V L (V .i32) := do
                callLocal dp4a %[cw, ← load32 (← at_ sq (4 * (j * dw + w))), a0]
              if j == 0 then
                acc := acc.set! j (← step)
              else
                let rb ← ifte (jTys := [.i32]) .ne (on.getD j zero) zero
                  (thn := do pure %[← step]) (els := pure %[a0])
                acc := acc.set! j rb.head
          let rt ← loadF32 (← at4 rterm rowc)
          -- every query's appends, one barrier, then every query's cut
          let mut mid : Array (V .i32) := #[]
          for j in List.range qpw do
            let cnt := cnts.getD j zero
            let thr := thrs.getD j inf
            let body : Prog V L (Vals V [.i32]) := do
              let key ← fadd (← fadd (bases.getD j inf) rt) (← fmul (scales.getD j inf) (← fcvtFromSint .f32 (acc.getD j zero)))
              -- a filtered search admits only the query's items (an
              -- unfiltered one reads word 0, so no read is past the bitsets)
              let qix ← select (← icmp .eq qwords zero) zero (← iadd (← imul (qs.getD j zero) qwords) idw)
              let qw ← load32 (← at4 qbits qix)
              let adm ← bor unfiltered (← band (← ushr qw idb) (← c32 1))
              let pass ← icmp .ne (← band (← band ok adm) (← bit (← fcmp .lt key thr))) zero
              let m ← callLocal ballotR %[pass]
              when .ne (← bit pass) zero do
                let e ← iadd (← c32 (j * cap)) (← iadd cnt (← popcnt (← band m ltMask)))
                store key (← at4 bkey e)
                storeI32 row (← at4 bval e)
              pure %[← iadd cnt (← popcnt m)]
            let res ← if j == 0 then body else
              ifte (jTys := [.i32]) .ne (on.getD j zero) zero (thn := body) (els := pure %[cnt])
            mid := mid.push res.head
          callLocalVoid barrier %[]
          -- a buffer that might not take the next 32: sorted, cut to R, a
          -- query at a time (one copy of the sort), its count and threshold
          -- the registers its index picks
          let mut most := mid.getD 0 zero
          for j in (List.range qpw).drop 1 do most ← umax most (mid.getD j zero)
          let cut ← ifte (jTys := List.replicate qpw .i32 ++ List.replicate qpw .f32) .ugt most (← c32 (cap - 32))
            (els := pure (pack mid zero qpw |>.append (pack thrs inf qpw))) (thn := do
              wloop (%[zero] |>.append (pack mid zero qpw |>.append (pack thrs inf qpw)))
                (fun cs => do pure (contIfULt cs.head (← c32 qpw), cs.tail, ()))
                (fun cs _ => do
                  let j := cs.head
                  let (cn, tn) := unpack2 qpw cs.tail
                  let c2 ← pick j cn zero
                  let thr ← pick j tn inf
                  let res ← ifte (jTys := [.i32, .f32]) .ugt c2 (← c32 (cap - 32))
                    (thn := do
                      let (ks, vs) ← sortBuffer cap lane inf neg1 bkey bval j
                      let base ← iadd (← imul j (← c32 cap)) (← imul lane (← c32 per))
                      for r in List.range per do
                        let e ← iadd (← imul lane (← c32 per)) (← c32 r)
                        let keep ← icmp .ult e rR
                        store (← select keep (ks.getD r inf) inf) (← at4 bkey (← iadd base (← c32 r)))
                        storeI32 (← select keep (vs.getD r neg1) neg1) (← at4 bval (← iadd base (← c32 r)))
                      pure %[rR, ← bcastSorted per ks inf rLast])
                    (els := pure %[c2, thr])
                  let cn ← place j cn res.head
                  let tn ← place j tn res.snd
                  pure (%[← iadd j (← c32 1)] |>.append (pack cn zero qpw |>.append (pack tn inf qpw)))))
          let (newC, newT) := unpack2 qpw cut
          pure (%[← iadd cs.head (← c32 1)] |>.append (pack newC zero qpw |>.append (pack newT inf qpw))))
      let _ := r
      callLocalVoid barrier %[]
      -- the results, a query at a time (one copy of the sort)
      let onA := on.set! 0 (← c32 1)
      stridedDo zero (← c32 qpw) 1 fun j => do
        when .ne (← pick j onA zero) zero do
          let (ks, vs) ← sortBuffer cap lane inf neg1 bkey bval j
          let slotOut ← match own with
            | some _ => pure l
            | none => do load32 (← at4 pout (← pick j pp zero))
          let obase ← imul slotOut rR
          for r in List.range per do
            let e ← iadd (← imul lane (← c32 per)) (← c32 r)
            when .ult e rR do
              store (ks.getD r inf) (← at4 okey (← iadd obase e))
              storeI32 (vs.getD r neg1) (← at4 orow (← iadd obase e))
      callLocalVoid barrier %[]
      pure %[← iadd pa (← c32 (qw * qpw))])

/-! ## Sorting and selection, without atomics

Two lanes of a warp updating one shared word are ordered only by a warp
barrier, so every kernel here whose lanes update shared counters is a block
of one warp (`Kernel.warp`), whose barrier is the warp's. -/

/-- The inclusive sum of `x` over lanes `0 … lane`, by five indexed
    shuffles. -/
def laneScan (lane x : V .i32) : Prog V L (V .i32) := do
  let mut s := x
  for k in [1, 2, 4, 8, 16] do
    let src ← select (← icmp .uge lane (← c32 k)) (← isub lane (← c32 k)) lane
    let y ← callLocal shflIdx %[s, src]
    s ← iadd s (← select (← icmp .uge lane (← c32 k)) y (← c32 0))
  pure s

/-- A float's bits in an order that agrees with the float's (negatives
    flipped, positives with the sign bit set). -/
def orderBits (f : V .f32) : Prog V L (V .i32) := do
  let u ← bitcast .i32 f
  bxor u (← bor (← sshr u (← c32 31)) (← c32 0x80000000))

/-- Elements a block of the radix sort takes: 128 rounds of a warp. -/
def RADIX_E : Nat := 4096

/-- **One pass of the radix sort, its histogram**: block `b` (one warp)
    counts the digits `(key >> shift) & 255` of keys `b · E … (b+1) · E` and
    writes count `d` at `hist[d · blocks + b]`, bin-major, as the offsets'
    scan wants. A digit's count is added by the lowest lane holding it. `meta`:
    `n`, `shift`, `blocks`. -/
def radixHistBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let b := ps.head; let lane := ps.snd; let sm := (ps.drop 3).head
  let keys := (ps.drop 4).head; let hist := (ps.drop 5).head; let mt := (ps.drop 6).head
  let n ← load32 mt
  let shift ← load32 (← at_ mt 4)
  let blocks ← load32 (← at_ mt 8)
  for r in List.range 8 do
    storeI32 (← c32 0) (← at4 sm (← iadd (← ishlImm lane 3) (← c32 r)))
  callLocalVoid barrier %[]
  let start ← imul b (← c32 RADIX_E)
  let _ ← wloop %[← c32 0] (fun cs => do pure (contIfULt cs.head (← c32 (RADIX_E / 32)), %[], ()))
    (fun cs _ => do
      let i ← iadd start (← iadd (← ishlImm cs.head 5) lane)
      let active ← icmp .ult i n
      let key ← load32 (← at4 keys (← select active i (← c32 0)))
      let d ← band (← ushr key shift) (← c32 255)
      let m ← peers d active
      when .ne (← bit active) (← c32 0) do
        when .eq (← ctz m) lane do
          let slot ← at4 sm d
          storeI32 (← iadd (← load32 slot) (← popcnt m)) slot
      callLocalVoid barrier %[]
      pure %[← iadd cs.head (← c32 1)])
  for r in List.range 8 do
    let d ← iadd (← ishlImm lane 3) (← c32 r)
    storeI32 (← load32 (← at4 sm d)) (← at4 hist (← iadd (← imul d blocks) b))

/-- **An exclusive scan in place** of the `n` words at `x`, one block: each
    thread sums its run, thread 0 scans the 256 sums, each thread rewrites its
    run. `meta`: `n`. -/
def exScanBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let mt := (ps.drop 5).head
  let n ← load32 mt
  let per ← udiv (← iadd n (← c32 255)) (← c32 256)
  let lo ← umin (← imul tid per) n
  let hi ← umin (← iadd lo per) n
  let tot ← wloop %[lo, ← c32 0] (fun cs => pure (contIfULt cs.head hi, %[cs.snd], ()))
    (fun cs _ => do pure %[← iadd cs.head (← c32 1), ← iadd cs.snd (← load32 (← at4 x cs.head))])
  storeI32 tot.head (← at4 sm tid)
  callLocalVoid barrier %[]
  when .eq tid (← c32 0) do
    let _ ← wloop %[← c32 0, ← c32 0] (fun cs => do pure (contIfULt cs.head (← c32 256), %[], ()))
      (fun cs _ => do
        let slot ← at4 sm cs.head
        let v ← load32 slot
        storeI32 cs.snd slot
        pure %[← iadd cs.head (← c32 1), ← iadd cs.snd v])
  callLocalVoid barrier %[]
  let _ ← wloop %[lo, ← load32 (← at4 sm tid)] (fun cs => pure (contIfULt cs.head hi, %[], ()))
    (fun cs _ => do
      let slot ← at4 x cs.head
      let v ← load32 slot
      storeI32 cs.snd slot
      pure %[← iadd cs.head (← c32 1), ← iadd cs.snd v])

/-- **One pass of the radix sort, its scatter**: block `b` (one warp) moves
    its keys and values, in order, to their digit's next places from the
    scanned histogram: a stable pass. -/
def radixScatterBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let lane := ps.snd; let sm := (ps.drop 3).head
  let kin := (ps.drop 4).head; let vin := (ps.drop 5).head
  let kout := (ps.drop 6).head; let vout := (ps.drop 7).head
  let hist := (ps.drop 8).head; let mt := (ps.drop 9).head
  let n ← load32 mt
  let shift ← load32 (← at_ mt 4)
  let blocks ← load32 (← at_ mt 8)
  for r in List.range 8 do
    let d ← iadd (← ishlImm lane 3) (← c32 r)
    storeI32 (← load32 (← at4 hist (← iadd (← imul d blocks) b))) (← at4 sm d)
  callLocalVoid barrier %[]
  let ltMask ← isub (← ishl (← c32 1) lane) (← c32 1)
  let start ← imul b (← c32 RADIX_E)
  let _ ← wloop %[← c32 0] (fun cs => do pure (contIfULt cs.head (← c32 (RADIX_E / 32)), %[], ()))
    (fun cs _ => do
      let i ← iadd start (← iadd (← ishlImm cs.head 5) lane)
      let active ← icmp .ult i n
      let ic ← select active i (← c32 0)
      let key ← load32 (← at4 kin ic)
      let val ← load32 (← at4 vin ic)
      let d ← band (← ushr key shift) (← c32 255)
      let m ← peers d active
      let slot ← at4 sm d
      let base ← load32 slot
      when .ne (← bit active) (← c32 0) do
        let pos ← iadd base (← popcnt (← band m ltMask))
        storeI32 key (← at4 kout pos)
        storeI32 val (← at4 vout pos)
      callLocalVoid barrier %[]
      when .ne (← bit active) (← c32 0) do
        when .eq (← ctz m) lane do
          storeI32 (← iadd base (← popcnt m)) slot
      callLocalVoid barrier %[]
      pure %[← iadd cs.head (← c32 1)])

/-- **Each key's run** in the `n` sorted keys: `start[k]` the first index
    holding `k`, `end[k]` past its last (keys absent keep what they held).
    `meta`: `n`. -/
def segmentsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let keys := (ps.drop 4).head; let st := (ps.drop 5).head; let en := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let n ← load32 mt
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i n do
    let k ← load32 (← at4 keys i)
    let isFirst ← bit (← icmp .eq i (← c32 0))
    let prev ← load32 (← at4 keys (← select (← icmp .eq i (← c32 0)) i (← isub i (← c32 1))))
    let differs ← bit (← icmp .ne prev k)
    when .ne (← bor isFirst differs) (← c32 0) do
      storeI32 i (← at4 st k)
    let lastI ← isub n (← c32 1)
    let isLast ← bit (← icmp .eq i lastI)
    let next ← load32 (← at4 keys (← select (← icmp .eq i lastI) i (← iadd i (← c32 1))))
    let differs2 ← bit (← icmp .ne next k)
    when .ne (← bor isLast differs2) (← c32 0) do
      storeI32 (← iadd i (← c32 1)) (← at4 en k)

/-- **The `k` smallest of each row**, in no order: their column indices. A
    block of eight warps a row of `cols` floats: four passes of an 8-bit radix
    select over the keys' order bits find the `k`-th smallest exactly (each
    warp counting its elements' digits into its own histogram, a digit's count
    added by the lowest lane holding it, the eight summed per digit), then the
    block collects the keys below it and as many equal to it as are needed, in
    column order, each warp's place from the warps before it. Every thread
    makes the same number of trips, so the block's barriers inside them hold.
    Shared memory: eight histograms of 256 counts, their sums, the digit found
    and the count below it, each warp's count of a trip. `meta`: `cols`, `k`. -/
def rowSelectBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let row := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let cols ← load32 mt
  let k ← load32 (← at_ mt 4)
  let lane ← band tid (← c32 31)
  let warp ← ushrImm tid 5
  let xr ← iadd x (← ishlImm (← imul (← ext row) (← ext cols)) 2)
  let orow ← iadd out (← ishlImm (← imul (← ext row) (← ext k)) 2)
  let ltMask ← isub (← ishl (← c32 1) lane) (← c32 1)
  let trips ← udiv (← iadd cols (← c32 255)) (← c32 256)
  let hist ← at4 sm (← ishlImm warp 8)
  let tot ← at_ sm (8 * 1024)
  let found ← at_ sm (9 * 1024)
  let wc ← at_ sm (9 * 1024 + 16)
  let mut pfx ← c32 0
  let mut pmask ← c32 0
  let mut need := k
  for shiftN in [24, 16, 8, 0] do
    for r in List.range 8 do
      storeI32 (← c32 0) (← at4 sm (← iadd (← ishlImm tid 3) (← c32 r)))
    callLocalVoid barrier %[]
    let shift ← c32 shiftN
    let _ ← wloop %[← c32 0] (fun cs => pure (contIfULt cs.head trips, %[], ()))
      (fun cs _ => do
        let i ← iadd (← ishlImm cs.head 8) tid
        let inRow ← icmp .ult i cols
        let bits ← orderBits (← loadF32 (← at4 xr (← select inRow i (← c32 0))))
        let inPfx ← bit (← icmp .eq (← band bits pmask) pfx)
        let active ← icmp .ne (← band (← bit inRow) inPfx) (← c32 0)
        let d ← band (← ushr bits shift) (← c32 255)
        -- a warp with nothing left in the prefix skips the digit's peers
        when .ne (← callLocal ballotR %[active]) (← c32 0) do
          let m ← peers d active
          when .ne (← bit active) (← c32 0) do
            when .eq (← ctz m) lane do
              let slot ← at4 hist d
              storeI32 (← iadd (← load32 slot) (← popcnt m)) slot
        callLocalVoid barrier %[]
        pure %[← iadd cs.head (← c32 1)])
    -- each digit's count over the eight warps
    let mut sum ← c32 0
    for w in List.range WARPS do
      sum ← iadd sum (← load32 (← at4 sm (← iadd tid (← c32 (256 * w)))))
    storeI32 sum (← at4 tot tid)
    callLocalVoid barrier %[]
    -- the digit whose cumulative count first reaches `need`: warp 0, a lane
    -- eight digits
    when .eq warp (← c32 0) do
      let mut own ← c32 0
      for r in List.range 8 do
        own ← iadd own (← load32 (← at4 tot (← iadd (← ishlImm lane 3) (← c32 r))))
      let incl ← laneScan lane own
      let mut cum ← isub incl own
      for r in List.range 8 do
        let d ← iadd (← ishlImm lane 3) (← c32 r)
        let c ← load32 (← at4 tot d)
        let next ← iadd cum c
        let lo ← bit (← icmp .ult cum need)
        let hi ← bit (← icmp .uge next need)
        when .ne (← band lo hi) (← c32 0) do
          storeI32 d found
          storeI32 cum (← at_ found 4)
        cum := next
    callLocalVoid barrier %[]
    let dig ← load32 found
    let below ← load32 (← at_ found 4)
    callLocalVoid barrier %[]
    pfx ← bor pfx (← ishlImm dig shiftN)
    pmask ← bor pmask (← c32 (255 * 2 ^ shiftN))
    need ← isub need below
  -- collect: below the threshold, then `need` equal to it, in column order;
  -- a warp's place is the block's count so far and the warps' before it
  let collect (eqPass : Bool) (pos0 cap : V .i32) : Prog V L (V .i32) := do
    let r ← wloop %[← c32 0, pos0, cap] (fun cs => pure (contIfULt cs.head trips, %[cs.snd], ()))
      (fun cs _ => do
        let i ← iadd (← ishlImm cs.head 8) tid
        let inRow ← icmp .ult i cols
        let bits ← orderBits (← loadF32 (← at4 xr (← select inRow i (← c32 0))))
        let hit ← if eqPass then icmp .eq bits pfx else icmp .ult bits pfx
        let want ← icmp .ne (← band (← bit inRow) (← bit hit)) (← c32 0)
        let m ← callLocal ballotR %[want]
        when .eq lane (← c32 0) do
          storeI32 (← popcnt m) (← at4 wc warp)
        callLocalVoid barrier %[]
        let mut before ← c32 0
        let mut all ← c32 0
        for w in List.range WARPS do
          let cw ← load32 (← at4 wc (← c32 w))
          before ← iadd before (← select (← icmp .ult (← c32 w) warp) cw (← c32 0))
          all ← iadd all cw
        callLocalVoid barrier %[]
        let rank ← iadd before (← popcnt (← band m ltMask))
        let take ← band (← bit want) (← bit (← icmp .ult rank cs.thd))
        when .ne take (← c32 0) do
          storeI32 i (← at4 orow (← iadd cs.snd rank))
        let got ← umin all cs.thd
        pure %[← iadd cs.head (← c32 1), ← iadd cs.snd got, ← isub cs.thd got])
    pure r.head
  let below ← collect false (← c32 0) (← c32 0x7FFFFFFF)
  let _ ← collect true below need

/-- **The coarse keys in place**: `‖c‖² - 2 q·c` (L2) or `-q·c` (inner
    product) from the `n` products `q·c` at `dots`, row-major `[queries,
    lists]`, plus each list's mask (`+∞` for a list with no live row, which a
    probe would only waste). `meta`: `n`, `lists`. -/
def coarseKeyBody (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let dots := (ps.drop 4).head; let cn := (ps.drop 5).head; let mask := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let n ← load32 mt
  let lists ← load32 (← at_ mt 4)
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i n do
    let slot ← at4 dots i
    let v ← loadF32 slot
    let l ← isub i (← imul (← udiv i lists) lists)
    let key ← if l2 then fsub (← loadF32 (← at4 cn l)) (← fadd v v) else fneg v
    store (← fadd key (← loadF32 (← at4 mask l))) slot

/-- **The coarse keys of few queries**, a warp a (query, list): the dot of
    query `q` (`d` floats) with centre `l`, then the key as `coarseKeyBody`
    makes it from the product, into `dots[q · lists + l]`. `meta`: `nq`,
    `lists`. -/
def coarseDirectBody (d : Nat) (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let qv := (ps.drop 4).head; let cent := (ps.drop 5).head; let cn := (ps.drop 6).head
  let mask := (ps.drop 7).head; let dots := (ps.drop 8).head; let mt := (ps.drop 9).head
  let nq ← load32 mt
  let lists ← load32 (← at_ mt 4)
  let lane ← band tid (← c32 31)
  let w ← iadd (← imul b (← c32 WARPS)) (← ushrImm tid 5)
  let q ← udiv w lists
  when .ult q nq do
    let l ← isub w (← imul q lists)
    let qr ← iadd qv (← ishlImm (← imul (← ext q) (← c64n d)) 2)
    let cr ← iadd cent (← ishlImm (← imul (← ext l) (← c64n d)) 2)
    let part ← strided lane (← c32 d) 32 (← f32c 0) fun j acc => do
      fadd acc (← fmul (← loadF32 (← at4 qr j)) (← loadF32 (← at4 cr j)))
    let v ← warpAll (fun a b => fadd a b) part
    when .eq lane (← c32 0) do
      let key ← if l2 then fsub (← loadF32 (← at4 cn l)) (← fadd v v) else fneg v
      store (← fadd key (← loadF32 (← at4 mask l))) (← at4 dots w)

/-- **Each list's mask**, a block a list: `0` where some row in it is live,
    `+∞` where none is. -/
def listLiveBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let l := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let loff := (ps.drop 4).head; let lend := (ps.drop 5).head; let lid := (ps.drop 6).head
  let live := (ps.drop 7).head; let mask := (ps.drop 8).head
  let r0 ← load32 (← at4 loff l)
  let r1 ← load32 (← at4 lend l)
  let cnt ← strided (← iadd r0 tid) r1 256 (← f32c 0) fun r acc => do
    let id ← load32 (← at4 lid r)
    let b ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
    fadd acc (← fcvtFromSint .f32 b)
  let tot ← blockAll tid sm 0 WARPS (fun a b => fadd a b) cnt
  when .eq tid (← c32 0) do
    store (← select (← fcmp .gt tot (← f32c 0)) (← f32c 0) (← f32c posInfBits)) (← at4 mask l)

/-- **Which stored rows are live**: `flag[p]` is `1` where the id of row `p`
    is in the live set. `meta`: `n`. -/
def liveFlagBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let ids := (ps.drop 4).head; let live := (ps.drop 5).head; let flag := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let p ← iadd (← imul b (← c32 256)) tid
  when .ult p (← load32 mt) do
    let id ← ireduce32 (← load64 (← iadd ids (← ishlImm (← ext p) 3)))
    let bv ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
    storeI32 bv (← at4 flag p)

/-- **The live rows' positions**, in order: a live row `p` goes to
    `pos[off[p]]`, `off` the scanned flags. `meta`: `n`. -/
def scatterLiveBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let ids := (ps.drop 4).head; let live := (ps.drop 5).head; let off := (ps.drop 6).head
  let pos := (ps.drop 7).head; let mt := (ps.drop 8).head
  let p ← iadd (← imul b (← c32 256)) tid
  when .ult p (← load32 mt) do
    let id ← ireduce32 (← load64 (← iadd ids (← ishlImm (← ext p) 3)))
    let bv ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
    when .ne bv (← c32 0) do
      storeI32 p (← at4 pos (← load32 (← at4 off p)))

/-- **Rows gathered**: row `r` of `xl` (and its id) := row `pos[r]` of `x`
    (and its id), a block a row of `d` floats. -/
def gatherIdxBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let r := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let ids := (ps.drop 5).head; let pos := (ps.drop 6).head
  let xl := (ps.drop 7).head; let idsl := (ps.drop 8).head
  let p ← load32 (← at4 pos r)
  let src ← iadd x (← ishlImm (← imul (← ext p) (← c64n d)) 2)
  let dst ← iadd xl (← ishlImm (← imul (← ext r) (← c64n d)) 2)
  stridedDo tid (← c32 d) 256 fun t => do
    store (← loadF32 (← at4 src t)) (← at4 dst t)
  when .eq tid (← c32 0) do
    storeI64 (← load64 (← iadd ids (← ishlImm (← ext p) 3))) (← iadd idsl (← ishlImm (← ext r) 3))

/-! ## Filtered search -/

/-- **The (tag, item) pairs** of an item → tags list: entry `e` (item `i`'s
    tags at `indptr[i] … indptr[i+1]`) gives key `tags[e]`, value `i` (found by
    binary search in `indptr`). `meta`: `entries`, `items`. -/
def tagPairsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let indptr := (ps.drop 4).head; let tags := (ps.drop 5).head
  let keys := (ps.drop 6).head; let vals := (ps.drop 7).head; let mt := (ps.drop 8).head
  let entries ← load32 mt
  let items ← load32 (← at_ mt 4)
  let e ← iadd (← imul b (← c32 256)) tid
  when .ult e entries do
    -- the last item whose start is at most e
    let r ← wloop %[← c32 0, items] (fun cs => do
        pure (contIfULt (← iadd cs.head (← c32 1)) cs.snd, %[cs.head], ()))
      (fun cs _ => do
        let mid ← ushrImm (← iadd cs.head cs.snd) 1
        let le ← icmp .ule (← load32 (← at4 indptr mid)) e
        pure %[← select le mid cs.head, ← select le cs.snd mid])
    storeI32 (← load32 (← at4 tags e)) (← at4 keys e)
    storeI32 r.head (← at4 vals e)

/-- Whether `x` is among the sorted `items[lo … hi)`: a binary search. -/
def holds (items : V .i64) (lo hi x : V .i32) : Prog V L (V .i32) := do
  let r ← wloop %[lo, hi] (fun cs => pure (contIfULt cs.head cs.snd, %[cs.head], ()))
    (fun cs _ => do
      let mid ← ushrImm (← iadd cs.head cs.snd) 1
      let lt ← icmp .ult (← load32 (← at4 items mid)) x
      pure %[← select lt (← iadd mid (← c32 1)) cs.head, ← select lt cs.snd mid])
  let at' := r.head
  let inRange ← bit (← icmp .ult at' hi)
  let eq ← bit (← icmp .eq (← load32 (← at4 items (← select (← icmp .ult at' hi) at' lo))) x)
  band inRange eq

/-- **Each query's admitted items**, a block a query, in id order: the items
    of its shortest required list (`items[qlo … qlo + qlen)`) found in each of
    its other required lists (`olo[r] … ohi[r]` for `r` in `optr[q] …
    optr[q+1]`), compacted by ballots to `adm[off[q] …]`, their count to
    `cnt[q]`. Shared memory: each warp's count of a trip. -/
def admitBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 9 .i64)
  let q := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let items := (ps.drop 4).head; let qlo := (ps.drop 5).head; let qlen := (ps.drop 6).head
  let optr := (ps.drop 7).head; let olo := (ps.drop 8).head; let ohi := (ps.drop 9).head
  let off := (ps.drop 10).head; let adm := (ps.drop 11).head; let cnt := (ps.drop 12).head
  let lane ← band tid (← c32 31)
  let warp ← ushrImm tid 5
  let ltMask ← isub (← ishl (← c32 1) lane) (← c32 1)
  let lo ← load32 (← at4 qlo q)
  let len ← load32 (← at4 qlen q)
  let o0 ← load32 (← at4 optr q)
  let o1 ← load32 (← at4 optr (← iadd q (← c32 1)))
  let dst ← load32 (← at4 off q)
  let trips ← udiv (← iadd len (← c32 255)) (← c32 256)
  let r ← wloop %[← c32 0, ← c32 0] (fun cs => pure (contIfULt cs.head trips, %[cs.snd], ()))
    (fun cs _ => do
      let i ← iadd (← ishlImm cs.head 8) tid
      let inL ← icmp .ult i len
      let x ← load32 (← at4 items (← iadd lo (← select inL i (← c32 0))))
      let keep ← wloop %[o0, ← bit inL] (fun cs2 => pure (contIfULt cs2.head o1, %[cs2.snd], ()))
        (fun cs2 _ => do
          let h ← holds items (← load32 (← at4 olo cs2.head)) (← load32 (← at4 ohi cs2.head)) x
          pure %[← iadd cs2.head (← c32 1), ← band cs2.snd h])
      let take ← icmp .ne keep.head (← c32 0)
      let m ← callLocal ballotR %[take]
      when .eq lane (← c32 0) do
        storeI32 (← popcnt m) (← at4 sm warp)
      callLocalVoid barrier %[]
      let mut before ← c32 0
      let mut all ← c32 0
      for w in List.range WARPS do
        let cw ← load32 (← at4 sm (← c32 w))
        before ← iadd before (← select (← icmp .ult (← c32 w) warp) cw (← c32 0))
        all ← iadd all cw
      callLocalVoid barrier %[]
      when .ne (← bit take) (← c32 0) do
        storeI32 x (← at4 adm (← iadd dst (← iadd cs.snd (← iadd before (← popcnt (← band m ltMask))))))
      pure %[← iadd cs.head (← c32 1), ← iadd cs.snd all])
  when .eq tid (← c32 0) do
    storeI32 r.head (← at4 cnt q)

/-- `shfl.sync.idx` of a float. -/
def shflIdxF (x : V .f32) (src : V .i32) : Prog V L (V .f32) := do
  bitcast .f32 (← callLocal shflIdx %[← bitcast .i32 x, src])

/-- **The exact answer over a query's admitted items**, a block a query (the
    query `qidx[b]`): each warp scores its share of the items exactly (a lane
    a stride of dimensions, then a butterfly) and keeps its best 32 in
    registers, lane `l` the `l`-th, a new item shifting the larger ones up a
    lane; then warp 0 sorts the eight warps' 256 and writes the best `k`
    ids, `-1` where there were fewer. An item no longer live is skipped.
    `meta`: `k`. -/
def exactBody (d : Nat) (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 10 .i64)
  let b := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let posOf := (ps.drop 5).head; let q := (ps.drop 6).head
  let adm := (ps.drop 7).head; let off := (ps.drop 8).head; let cnt := (ps.drop 9).head
  let qidx := (ps.drop 10).head; let out := (ps.drop 11).head; let live := (ps.drop 12).head
  let mt := (ps.drop 13).head
  let k ← load32 mt
  let lane ← band tid (← c32 31)
  let warp ← ushrImm tid 5
  let inf ← f32c posInfBits
  let neg1 ← c32 0xFFFFFFFF
  let qi ← load32 (← at4 qidx b)
  let qr ← iadd q (← ishlImm (← imul (← ext qi) (← c64n d)) 2)
  let o ← load32 (← at4 off qi)
  let n ← load32 (← at4 cnt qi)
  let mine ← udiv (← iadd (← isub n (← umin warp n)) (← c32 (WARPS - 1))) (← c32 WARPS)
  let lm1 ← select (← icmp .eq lane (← c32 0)) lane (← isub lane (← c32 1))
  let best ← wloop %[← c32 0, inf, neg1] (fun cs => pure (contIfULt cs.head mine, %[cs.snd, cs.thd], ()))
    (fun cs _ => do
      let i ← iadd warp (← ishlImm cs.head 3)
      let id ← load32 (← at4 adm (← iadd o i))
      let p ← load32 (← at4 posOf id)
      let lv ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
      let xr ← iadd x (← ishlImm (← imul (← ext p) (← c64n d)) 2)
      let part ← strided lane (← c32 d) 32 (← f32c 0) fun j acc => do
        let xv ← loadF32 (← at4 xr j)
        let qv ← loadF32 (← at4 qr j)
        if l2 then
          let df ← fsub xv qv
          fma df df acc
        else
          fma xv qv acc
      let tot ← warpAll (fun a b => fadd a b) part
      let key0 ← if l2 then pure tot else fneg tot
      let key ← select (← icmp .ne lv (← c32 0)) key0 inf
      -- insert: lanes holding larger keys move up one, the first takes it
      let kk := cs.snd; let kv := cs.thd
      let bigger ← icmp .ne (← bit (← fcmp .gt kk key)) (← c32 0)
      let mask ← callLocal ballotR %[bigger]
      let pos ← ctz (← bor mask (← c32 0x80000000))
      let prevK ← shflIdxF kk lm1
      let prevV ← callLocal shflIdx %[kv, lm1]
      let atPos ← icmp .eq lane pos
      let after ← icmp .ugt lane pos
      let nk ← select atPos key (← select after prevK kk)
      let nv ← select atPos id (← select after prevV kv)
      let ins ← icmp .ne mask (← c32 0)
      pure %[← iadd cs.head (← c32 1), ← select ins nk kk, ← select ins nv kv])
  store best.head (← at4 sm (← iadd (← ishlImm warp 5) lane))
  storeI32 best.snd (← at4 sm (← iadd (← iadd (← ishlImm warp 5) lane) (← c32 256)))
  callLocalVoid barrier %[]
  when .eq warp (← c32 0) do
    let mut ks : Array (V .f32) := #[]
    let mut vs : Array (V .i32) := #[]
    for r in List.range 8 do
      let e ← iadd (← ishlImm lane 3) (← c32 r)
      ks := ks.push (← loadF32 (← at4 sm e))
      vs := vs.push (← load32 (← at4 sm (← iadd e (← c32 256))))
    let (_, sv) ← warpSort lane inf neg1 ks vs
    for r in List.range 8 do
      let e ← iadd (← ishlImm lane 3) (← c32 r)
      when .ult e k do
        storeI64 (← sextend64 (sv.getD r neg1)) (← iadd out (← ishlImm (← ext (← iadd (← imul qi k) e)) 3))

/-- **Admitted bitsets**, a block a query (query `bidx[b]`, bitset row `b`):
    its admitted items are in id order, so the first item of each word's run
    sets the run's bits and writes the word, and no two threads write one.
    `meta`: `words`. -/
def setBitsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let adm := (ps.drop 4).head; let off := (ps.drop 5).head; let cnt := (ps.drop 6).head
  let bidx := (ps.drop 7).head; let bits := (ps.drop 8).head; let mt := (ps.drop 9).head
  let words ← load32 mt
  let qi ← load32 (← at4 bidx b)
  let o ← load32 (← at4 off qi)
  let n ← load32 (← at4 cnt qi)
  let row ← iadd bits (← ishlImm (← imul (← ext b) (← ext words)) 2)
  stridedDo tid n 256 fun i => do
    let id ← load32 (← at4 adm (← iadd o i))
    let w ← ushrImm id 5
    let prevW ← ushrImm (← load32 (← at4 adm (← iadd o (← select (← icmp .eq i (← c32 0)) i (← isub i (← c32 1)))))) 5
    let first ← bor (← bit (← icmp .eq i (← c32 0))) (← bit (← icmp .ne prevW w))
    when .ne first (← c32 0) do
      let r ← wloop %[i, ← c32 0] (fun cs => do
          let inN ← bit (← icmp .ult cs.head n)
          let same ← bit (← icmp .eq (← ushrImm (← load32 (← at4 adm (← iadd o (← select (← icmp .ult cs.head n) cs.head i)))) 5) w)
          pure (contIf .ne (← band inN same) (← c32 0), %[cs.snd], ()))
        (fun cs _ => do
          let x ← load32 (← at4 adm (← iadd o cs.head))
          pure %[← iadd cs.head (← c32 1), ← bor cs.snd (← ishl (← c32 1) (← band x (← c32 31)))])
      storeI32 r.head (← at4 row w)

/-- Row `r` of `out` := row `idx[r]` of `x`, a block a row of `d` floats. -/
def gatherRowsBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let r := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let idx := (ps.drop 5).head; let out := (ps.drop 6).head
  let src ← iadd x (← ishlImm (← imul (← ext (← load32 (← at4 idx r))) (← c64n d)) 2)
  let dst ← iadd out (← ishlImm (← imul (← ext r) (← c64n d)) 2)
  stridedDo tid (← c32 d) 256 fun t => do
    store (← loadF32 (← at4 src t)) (← at4 dst t)

/-- Each stored row's position by its id: `posOf[ids[p]] := p`. `meta`: `n`. -/
def posOfBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let ids := (ps.drop 4).head; let posOf := (ps.drop 5).head; let mt := (ps.drop 6).head
  let p ← iadd (← imul b (← c32 256)) tid
  when .ult p (← load32 mt) do
    let id ← ireduce32 (← load64 (← iadd ids (← ishlImm (← ext p) 3)))
    storeI32 p (← at4 posOf id)

/-- Each laid-out row's id: `lid[r] := ids[pos[r]]`. `meta`: `rows`. -/
def lidBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let ids := (ps.drop 4).head; let pos := (ps.drop 5).head; let lid := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let r ← iadd (← imul b (← c32 256)) tid
  when .ult r (← load32 mt) do
    let p ← load32 (← at4 pos r)
    storeI32 (← ireduce32 (← load64 (← iadd ids (← ishlImm (← ext p) 3)))) (← at4 lid r)

/-- **The buffered rows' norms**, a warp a row: `‖x‖²` of rows `b0 + j`,
    `j < nb`. `meta`: `b0`, `nb`. -/
def normsBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let nrm := (ps.drop 5).head; let mt := (ps.drop 6).head
  let b0 ← load32 mt
  let nb ← load32 (← at_ mt 4)
  let lane ← band tid (← c32 31)
  let j ← iadd (← imul b (← c32 WARPS)) (← ushrImm tid 5)
  when .ult j nb do
    let xr ← iadd x (← ishlImm (← imul (← ext (← iadd b0 j)) (← c64n d)) 2)
    let part ← strided lane (← c32 d) 32 (← f32c 0) fun t acc => do
      let v ← loadF32 (← at4 xr t)
      fma v v acc
    let tot ← warpAll (fun a b => fadd a b) part
    when .eq lane (← c32 0) do
      store tot (← at4 nrm j)

/-- **The buffered rows' keys in place**, from their products with the
    queries (row-major `[queries, nb]`): `‖x‖² - 2 q·x` (L2) or `-q·x`, `+∞`
    for a row no longer live. `meta`: `n`, `nb`, `b0`. -/
def bufKeyBody (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let dots := (ps.drop 4).head; let nrm := (ps.drop 5).head; let ids := (ps.drop 6).head
  let live := (ps.drop 7).head; let mt := (ps.drop 8).head
  let n ← load32 mt
  let nb ← load32 (← at_ mt 4)
  let b0 ← load32 (← at_ mt 8)
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i n do
    let j ← isub i (← imul (← udiv i nb) nb)
    let slot ← at4 dots i
    let v ← loadF32 slot
    let key ← if l2 then fsub (← loadF32 (← at4 nrm j)) (← fadd v v) else fneg v
    let id ← ireduce32 (← load64 (← iadd ids (← ishlImm (← ext (← iadd b0 j)) 3)))
    let bv ← band (← ushr (← load32 (← at4 live (← ushrImm id 5))) (← band id (← c32 31))) (← c32 1)
    store (← select (← icmp .ne bv (← c32 0)) key (← f32c posInfBits)) slot

/-- **The pairs' fields** from the pairs sorted by list (keys the lists,
    values `q · nprobe + j`): each pair's query and its centre term, the
    coarse key less `‖c‖²` (L2) or the coarse key itself. `meta`: `pairs`,
    `nprobe`, `lists`. -/
def pairFieldsBody (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 7 .i64)
  let b := ps.head; let tid := ps.snd
  let keys := (ps.drop 4).head; let vals := (ps.drop 5).head; let ckey := (ps.drop 6).head
  let cn := (ps.drop 7).head; let pq := (ps.drop 8).head; let pbase := (ps.drop 9).head
  let mt := (ps.drop 10).head
  let n ← load32 mt
  let nprobe ← load32 (← at_ mt 4)
  let lists ← load32 (← at_ mt 8)
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i n do
    let l ← load32 (← at4 keys i)
    let q ← udiv (← load32 (← at4 vals i)) nprobe
    storeI32 q (← at4 pq i)
    let ck ← loadF32 (← at4 ckey (← iadd (← imul q lists) l))
    let base ← if l2 then fsub ck (← loadF32 (← at4 cn l)) else pure ck
    store base (← at4 pbase i)

/-- `x` rounded to the nearest integer, ties to even (`|x| < 2²²`), as an
    integer: adding and subtracting `1.5 · 2²³` leaves `x` rounded. -/
def rint (x : V .f32) : Prog V L (V .i32) := do
  let magic ← f32c 0x4B400000
  fcvtToSint .i32 (← fsub (← fadd x magic) magic)

/-- An int8 code: `x` rounded and clamped to `±127`, as its low byte. -/
def code8 (x : V .f32) : Prog V L (V .i32) := do
  let c ← fmin (← fmax x (← f32c 0xC2FE0000)) (← f32c 0x42FE0000)
  band (← rint c) (← c32 255)

/-- **The queries' codes**, a warp a query of `d` floats: `u = q · scale`,
    the query's scale `max |u| / 127` (at least `1e-30 / 127`), its codes
    `rint(u / scale)` packed four a word (zeros past `d`), and the signed scale
    the scan reads (`sign · scale`). `meta`: `queries`. -/
def quantQueryBody (d : Nat) (sign : UInt32) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let q := (ps.drop 4).head; let scale := (ps.drop 5).head; let tq := (ps.drop 6).head
  let tqs := (ps.drop 7).head; let mt := (ps.drop 8).head
  let dw := (d + 3) / 4
  let lane ← band tid (← c32 31)
  let qi ← iadd (← imul b (← c32 WARPS)) (← ushrImm tid 5)
  when .ult qi (← load32 mt) do
    let qr ← iadd q (← ishlImm (← imul (← ext qi) (← c64n d)) 2)
    let am ← strided lane (← c32 d) 32 (← f32c 0) fun j acc => do
      fmax acc (← fabs (← fmul (← loadF32 (← at4 qr j)) (← loadF32 (← at4 scale j))))
    let amax ← warpAll (fun a b => fmax a b) am
    let ts ← fmul (← fmax amax (← f32c 0x0DA24260)) (← f32c 0x3C010204)
    let inv ← fdiv (← f32c 0x3F800000) ts
    stridedDo lane (← c32 dw) 32 fun w => do
      let mut word ← c32 0
      for i in List.range 4 do
        let e ← iadd (← ishlImm w 2) (← c32 i)
        let inD ← icmp .ult e (← c32 d)
        let v ← fmul (← fmul (← loadF32 (← at4 qr (← select inD e (← c32 0)))) (← loadF32 (← at4 scale (← select inD e (← c32 0))))) inv
        let c ← select inD (← code8 v) (← c32 0)
        word ← bor word (← ishlImm c (8 * i))
      storeI32 word (← at4 tq (← iadd (← imul qi (← c32 dw)) w))
    when .eq lane (← c32 0) do
      store (← fmul ts (← f32c sign)) (← at4 tqs qi)

/-- **The rows' codes and terms**, a thread a laid-out row: row `r` is the
    vector at position `pos[r]` in list `list[r]`; its residual from the
    list's centre, under the per-dimension `scale`, as int8 codes interleaved
    (word `w` at `((r / 32) · dw + w) · 32 + r % 32`), and `‖x‖²` (L2) or `0`
    as its term. `meta`: `rows`. -/
def encodeBody (d : Nat) (l2 : Bool) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 8 .i64)
  let b := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let pos := (ps.drop 5).head; let lst := (ps.drop 6).head
  let cent := (ps.drop 7).head; let scale := (ps.drop 8).head
  let codes := (ps.drop 9).head; let rterm := (ps.drop 10).head
  let mt := (ps.drop 11).head
  let dw := (d + 3) / 4
  let r ← iadd (← imul b (← c32 256)) tid
  when .ult r (← load32 mt) do
    let xr ← iadd x (← ishlImm (← imul (← ext (← load32 (← at4 pos r))) (← c64n d)) 2)
    let cr ← iadd cent (← ishlImm (← imul (← ext (← load32 (← at4 lst r))) (← c64n d)) 2)
    let cbase ← iadd (← imul (← ushrImm r 5) (← c32 (dw * 32))) (← band r (← c32 31))
    let mut nn ← f32c 0
    for w in List.range dw do
      let mut word ← c32 0
      for i in List.range 4 do
        let e := 4 * w + i
        if e < d then
          let xv ← loadF32 (← at_ xr (4 * e))
          nn ← fma xv xv nn
          let res ← fsub xv (← loadF32 (← at_ cr (4 * e)))
          let c ← code8 (← fdiv res (← loadF32 (← at_ scale (4 * e))))
          word ← bor word (← ishlImm c (8 * i))
      storeI32 word (← at4 codes (← iadd cbase (← c32 (w * 32))))
    let term ← if l2 then pure nn else f32c 0
    store term (← at4 rterm r)

/-- **The largest residual of each dimension**, over a block's rows: a
    thread a dimension, rows `b · per …` of the `n`; to `part[b · d + j]`.
    `meta`: `n`, `per`. -/
def resMaxBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let asg := (ps.drop 5).head; let cent := (ps.drop 6).head
  let part := (ps.drop 7).head; let mt := (ps.drop 8).head
  let n ← load32 mt
  let per ← load32 (← at_ mt 4)
  let lo ← umin (← imul b per) n
  let hi ← umin (← iadd lo per) n
  stridedDo tid (← c32 d) 256 fun j => do
    let m ← wloop %[lo, ← f32c 0] (fun cs => pure (contIfULt cs.head hi, %[cs.snd], ()))
      (fun cs _ => do
        let i := cs.head
        let xv ← loadF32 (← at4 x (← iadd (← imul i (← c32 d)) j))
        let c ← load32 (← at4 asg i)
        let cv ← loadF32 (← at4 cent (← iadd (← imul c (← c32 d)) j))
        pure %[← iadd i (← c32 1), ← fmax cs.snd (← fabs (← fsub xv cv))])
    store m.head (← at4 part (← iadd (← imul b (← c32 d)) j))

/-- **The scale of each dimension** from the blocks' largest residuals:
    `max / 127`, at least `1e-12 / 127`. `meta`: `blocks`. -/
def scaleBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd
  let part := (ps.drop 4).head; let scale := (ps.drop 5).head; let mt := (ps.drop 6).head
  let blocks ← load32 mt
  stridedDo tid (← c32 d) 256 fun j => do
    let m ← wloop %[← c32 0, ← f32c 0] (fun cs => pure (contIfULt cs.head blocks, %[cs.snd], ()))
      (fun cs _ => do
        pure %[← iadd cs.head (← c32 1), ← fmax cs.snd (← loadF32 (← at4 part (← iadd (← imul cs.head (← c32 d)) j)))])
    store (← fmul (← fmax m.head (← f32c 0x2B8CBCCC)) (← f32c 0x3C010204)) (← at4 scale j)

/-- **A centre's new value**, a block a centre: the mean of its rows (the
    sample's rows `order[start … end)`, sample row `i` the vector at position
    `i · stride`), or the old centre where it has none; then its `‖c‖²`.
    `meta`: `stride`. -/
def centreBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 7 .i64)
  let c := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let order := (ps.drop 5).head; let st := (ps.drop 6).head
  let en := (ps.drop 7).head; let cent := (ps.drop 8).head; let cn := (ps.drop 9).head
  let mt := (ps.drop 10).head
  let stride ← load32 mt
  let s0 ← load32 (← at4 st c)
  let s1 ← load32 (← at4 en c)
  let cnt ← isub s1 s0
  let cr ← iadd cent (← ishlImm (← imul (← ext c) (← c64n d)) 2)
  let inv ← fdiv (← f32c 0x3F800000) (← fcvtFromSint .f32 (← umax cnt (← c32 1)))
  let nn ← strided tid (← c32 d) 256 (← f32c 0) fun j acc => do
    let sum ← wloop %[s0, ← f32c 0] (fun cs => pure (contIfULt cs.head s1, %[cs.snd], ()))
      (fun cs _ => do
        let row ← imul (← load32 (← at4 order cs.head)) stride
        let v ← loadF32 (← iadd x (← ishlImm (← iadd (← imul (← ext row) (← c64n d)) (← ext j)) 2))
        pure %[← iadd cs.head (← c32 1), ← fadd cs.snd v])
    let slot ← at4 cr j
    let v ← select (← icmp .ugt cnt (← c32 0)) (← fmul sum.head inv) (← loadF32 slot)
    store v slot
    fma v v acc
  let tot ← blockAll tid sm 0 WARPS (fun a b => fadd a b) nn
  when .eq tid (← c32 0) do
    store tot (← at4 cn c)

/-- `buf[i] := i` for `i < n`; `meta`: `n`. -/
def iotaBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let b := ps.head; let tid := ps.snd
  let buf := (ps.drop 4).head; let mt := (ps.drop 5).head
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i (← load32 mt) do
    storeI32 i (← at4 buf i)

/-- Row `j` of `out` := row `j · step` of `x`, a block a row of `d` floats.
    `meta`: `step`. -/
def gatherBody (d : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let j := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let src ← iadd x (← ishlImm (← imul (← ext (← imul j (← load32 mt))) (← c64n d)) 2)
  let dst ← iadd out (← ishlImm (← imul (← ext j) (← c64n d)) 2)
  stridedDo tid (← c32 d) 256 fun t => do
    store (← loadF32 (← at4 src t)) (← at4 dst t)

/-- **The exact re-rank**, a block a query, over two kinds of candidate:
    candidate `c < R` is the index row the scan put at index `sel[q · R + c]`
    of the query's `C` (`R · nprobe`) candidates, its vector row
    `rowPos[row]` of the index's rows `xi` and its id there; candidate
    `R + c`, `c < kb`, is the buffered row `b0 + bsel[q · kb + c]` of the
    store `xs` and its id there. Each warp scores its candidates exactly (a
    lane a stride of dimensions, then a butterfly), and warp 0 sorts all of
    them in registers and writes the best `k` ids, `-1` where there were
    fewer. `meta`: `C`, `R`, `k`, `kb`, `b0`. -/
def rerankBody (d : Nat) (l2 : Bool) (per : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 11 .i64)
  let qi := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let xi := (ps.drop 4).head; let rowPos := (ps.drop 5).head; let idsi := (ps.drop 6).head
  let orow := (ps.drop 7).head; let sel := (ps.drop 8).head; let q := (ps.drop 9).head
  let out := (ps.drop 10).head; let xs := (ps.drop 11).head; let idss := (ps.drop 12).head
  let bsel := (ps.drop 13).head; let mt := (ps.drop 14).head
  let cC ← load32 mt
  let rR ← load32 (← at_ mt 4)
  let k ← load32 (← at_ mt 8)
  let kb ← load32 (← at_ mt 12)
  let b0 ← load32 (← at_ mt 16)
  let lane ← band tid (← c32 31)
  let warp ← ushrImm tid 5
  let inf ← f32c posInfBits
  let neg1 ← c32 0xFFFFFFFF
  let qr ← iadd q (← ishlImm (← imul (← ext qi) (← c64n d)) 2)
  let dist ← at_ sm 0
  let ids ← at_ sm (4 * 32 * per)
  stridedDo warp (← c32 (32 * per)) WARPS fun c => do
    let inR ← icmp .ult c rR
    let cb ← isub c rR
    let inB ← icmp .ne (← band (← bit (← icmp .uge c rR)) (← bit (← icmp .ult cb kb))) (← c32 0)
    let sIx ← load32 (← at4 sel (← iadd (← imul qi rR) (← select inR c (← c32 0))))
    let row ← load32 (← at4 orow (← iadd (← imul qi cC) sIx))
    let rowOk ← icmp .ne (← band (← bit inR) (← bit (← icmp .sge row (← c32 0)))) (← c32 0)
    let ipos ← load32 (← at4 rowPos (← select rowOk row (← c32 0)))
    let bj ← load32 (← at4 bsel (← iadd (← imul qi kb) (← select inB cb (← c32 0))))
    let bOk ← icmp .ne (← band (← bit inB) (← bit (← icmp .sge bj (← c32 0)))) (← c32 0)
    let spos ← iadd b0 (← select bOk bj (← c32 0))
    let valid ← icmp .ne (← bor (← bit rowOk) (← bit bOk)) (← c32 0)
    let src ← select rowOk xi xs
    -- a candidate that is neither reads row 0, so no read is past the rows
    let p ← select rowOk ipos (← select bOk spos (← c32 0))
    let xr ← iadd src (← ishlImm (← imul (← ext p) (← c64n d)) 2)
    let part ← strided lane (← c32 d) 32 (← f32c 0) fun j acc => do
      let xv ← loadF32 (← at4 xr j)
      let qv ← loadF32 (← at4 qr j)
      if l2 then
        let df ← fsub xv qv
        fma df df acc
      else
        fma xv qv acc
    let tot ← warpAll (fun a b => fadd a b) part
    let key ← if l2 then pure tot else fneg tot
    let idSrc ← select rowOk idsi idss
    let id ← ireduce32 (← load64 (← iadd idSrc (← ishlImm (← ext p) 3)))
    when .eq lane (← c32 0) do
      store (← select valid key inf) (← at4 dist c)
      storeI32 (← select valid id neg1) (← at4 ids c)
  callLocalVoid barrier %[]
  when .eq warp (← c32 0) do
    let mut ks : Array (V .f32) := #[]
    let mut vs : Array (V .i32) := #[]
    for r in List.range per do
      let e ← iadd (← imul lane (← c32 per)) (← c32 r)
      ks := ks.push (← loadF32 (← at4 dist e))
      vs := vs.push (← load32 (← at4 ids e))
    let (_, sv) ← warpSort lane inf neg1 ks vs
    for r in List.range per do
      let e ← iadd (← imul lane (← c32 per)) (← c32 r)
      when .ult e k do
        let id := sv.getD r neg1
        storeI64 (← sextend64 id) (← iadd out (← ishlImm (← ext (← iadd (← imul qi k) e)) 3))

end

/-- The fused scan for `dw`-word codes, as a kernel: a block of one warp,
    eight a list. -/
def scan (dw : Nat) (cap : Nat := CAP) : Kernel :=
  { name := s!"ann_scan_{dw}_{cap}", params := List.replicate 17 .i64,
    code := emit (scanBody dw cap) (idParams ++ List.replicate 17 .i64), smem := scanSmem dw cap, warp := true }

/-- **The scan of few queries' pieces**: one query a warp, a warp a block,
    block `v` piece `v` of a pair, which it finds from the pair's selected
    list: its rows the pair's list's cut in `split` alike pieces. Parameters as the scan's, the
    first and past-last pair taking the selected lists and the coarse keys,
    the pairs' queries the centres' terms `‖c‖²`; `meta` adds `nprobe`,
    `lists` and the pieces a pair. -/
def scanOwn (dw cap : Nat) (l2 : Bool) : Kernel :=
  { name := s!"ann_scan_own_{dw}_{cap}_{if l2 then "l2" else "ip"}", params := List.replicate 17 .i64,
    code := emit (scanBody dw cap 1 1 (some l2)) (idParams ++ List.replicate 17 .i64),
    smem := scanSmem dw cap 1, warp := true }

/-- The kernels that take only pointers: name, body, parameter count,
    shared bytes, one warp a block. -/
def k (name : String) (body : Prog Slot Lvl Unit) (np : Nat) (smem : Nat := 0) (warp : Bool := false) : Kernel :=
  { name, params := List.replicate np .i64, code := emit body (idParams ++ List.replicate np .i64), smem, warp }

def radixHist : Kernel := k "ann_radix_hist" radixHistBody 3 1024 true
def exScan : Kernel := k "ann_exscan" exScanBody 2 1024
def radixScatter : Kernel := k "ann_radix_scatter" radixScatterBody 6 1024 true
def segments : Kernel := k "ann_segments" segmentsBody 4
def rowSelect : Kernel := k "ann_row_select" rowSelectBody 3 (9 * 1024 + 16 + 32)
def coarseKey (l2 : Bool) : Kernel := k s!"ann_coarse_key_{if l2 then "l2" else "ip"}" (coarseKeyBody l2) 4
def pairFields (l2 : Bool) : Kernel := k s!"ann_pair_fields_{if l2 then "l2" else "ip"}" (pairFieldsBody l2) 7
def coarseDirect (d : Nat) (l2 : Bool) : Kernel :=
  k s!"ann_coarse_direct_{d}_{if l2 then "l2" else "ip"}" (coarseDirectBody d l2) 6
def quantQuery (d : Nat) (l2 : Bool) : Kernel :=
  k s!"ann_quant_query_{d}_{if l2 then "l2" else "ip"}" (quantQueryBody d (if l2 then 0xC0000000 else 0xBF800000)) 5
def encode (d : Nat) (l2 : Bool) : Kernel := k s!"ann_encode_{d}_{if l2 then "l2" else "ip"}" (encodeBody d l2) 8
def resMax (d : Nat) : Kernel := k s!"ann_res_max_{d}" (resMaxBody d) 5
def scaleK (d : Nat) : Kernel := k s!"ann_scale_{d}" (scaleBody d) 3
def centre (d : Nat) : Kernel := k s!"ann_centre_{d}" (centreBody d) 7 64
def iota : Kernel := k "ann_iota" iotaBody 2
def listLive : Kernel := k "ann_list_live" listLiveBody 5 64
def liveFlag : Kernel := k "ann_live_flag" liveFlagBody 4
def scatterLive : Kernel := k "ann_scatter_live" scatterLiveBody 5
def gatherIdx (d : Nat) : Kernel := k s!"ann_gather_idx_{d}" (gatherIdxBody d) 5
def lidK : Kernel := k "ann_lid" lidBody 4
def tagPairs : Kernel := k "ann_tag_pairs" tagPairsBody 5
def admit : Kernel := k "ann_admit" admitBody 9 64
def exact (d : Nat) (l2 : Bool) : Kernel :=
  k s!"ann_exact_{d}_{if l2 then "l2" else "ip"}" (exactBody d l2) 10 2048
def setBits : Kernel := k "ann_set_bits" setBitsBody 6
def gatherRows (d : Nat) : Kernel := k s!"ann_gather_rows_{d}" (gatherRowsBody d) 3
def posOfK : Kernel := k "ann_pos_of" posOfBody 3
def norms (d : Nat) : Kernel := k s!"ann_norms_{d}" (normsBody d) 3
def bufKey (l2 : Bool) : Kernel := k s!"ann_buf_key_{if l2 then "l2" else "ip"}" (bufKeyBody l2) 5
def gather (d : Nat) : Kernel := k s!"ann_gather_{d}" (gatherBody d) 3
def rerank (d : Nat) (l2 : Bool) (per : Nat) : Kernel :=
  k s!"ann_rerank_{d}_{if l2 then "l2" else "ip"}_{per}" (rerankBody d l2 per) 11 (8 * 32 * per)

/-- The nearest centre of each row of a distance tile, as a kernel: blocks of
    eight warps, a warp a row. -/
def nearest : Kernel :=
  { name := "ann_nearest", params := [.i64, .i64, .i64, .i64],
    code := emit nearestBody (idParams ++ [.i64, .i64, .i64, .i64]) }

/-- The kernels of an engine for `d`-dimensional vectors under L2 (`l2`) or
    inner product, in the order the host names them. -/
def kernelsFor (d : Nat) (l2 : Bool) : List Kernel :=
  [nearest, scan ((d + 3) / 4), radixHist, exScan, radixScatter, iota, segments, rowSelect,
   coarseKey l2, pairFields l2, quantQuery d l2, encode d l2, resMax d, scaleK d, centre d, gather d,
   rerank d l2 1, rerank d l2 2, rerank d l2 4, listLive, liveFlag, scatterLive, gatherIdx d, norms d,
   bufKey l2, lidK, tagPairs, admit, exact d l2, setBits, gatherRows d, posOfK,
   scan ((d + 3) / 4) CAP2, rerank d l2 8, coarseDirect d l2,
   scanOwn ((d + 3) / 4) CAP l2, scanOwn ((d + 3) / 4) CAP2 l2]

/-- An engine's kernels as one module whose entry points carry their names. -/
def moduleFor (d : Nat) (l2 : Bool) : String :=
  match Kernel.module (kernelsFor d l2) with
  | .ok s => s
  | .error e => "ERROR " ++ e

/-- SIFT's engine's module, for the kernel checks. -/
def moduleText : String := moduleFor 128 true

end AnnKernels
