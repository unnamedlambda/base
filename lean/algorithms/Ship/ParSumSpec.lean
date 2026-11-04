import AlgorithmLib.Host.Hoare
import AlgorithmLib.Host.Program
import AlgorithmLib.Host.Threads
import AlgorithmLib.Proof.Typestate
import Demo.ParSum

/-!
# What ParSum computes

`Demo.ParSum`'s `main` writes the wrapping sum of the arena's 512 words, on a
machine whose joins are honest (`ThreadHonest`) and whose threads end. Its
thread function `u0:2` answers the wrapping sum of the 128 words at its
argument and leaves the world as it found it (`worker_triple`); in a thread's
view of a quarter those are the quarter's words (`view_wsum`), and a join of
a thread that wrote nothing hands back the bytes it was lent, so the arena
`main` reads is the one it was called with.

These are partial: where a run returns, it returns this. That `main` never
misuses a call or faults is `Ship.Threads`.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare

namespace ParSumSpec

/-- The wrapping sum of the `k` words at `p`, as memory `m` holds them. -/
def wsum (m : Mem) (p : UInt64) (k : Nat) : UInt64 :=
  (List.range k).foldl (fun s j => s + (m.load (p + UInt64.ofNat (8 * j)) 8).getD 0) 0

theorem wsum_succ (m : Mem) (p : UInt64) (k : Nat) :
    wsum m p (k + 1) = wsum m p k + (m.load (p + UInt64.ofNat (8 * k)) 8).getD 0 := by
  simp [wsum, List.range_succ, List.foldl_append]

-- ---------------------------------------------------------------------------
-- The thread's function
-- ---------------------------------------------------------------------------

/-- The thread's function as the program ships it. -/
def workerCode : Code := (runAns ParSum.worker [.i64]).2.1

theorem workerCode_eq : workerCode =
    [.straight [.op (.iconst .i64 128), .op (.iconst .i64 0), .op (.iconst .i64 0)],
     .loop { pTys := [.i64, .i64], init := [3, 2], flag := 6, exitOnTrue := false, cont := [15, 13],
             exitR := [5], exitTys := [.i64] }
       [.straight [.op (.icmp .ult 4 1)]]
       [.straight [.op (.iconst .i64 3), .op (.ishl 7 9), .op (.iadd 0 10),
          .op (.load { kind := .plain, ty := .i64, aligned := false } 11), .op (.iadd 8 12),
          .op (.iconst .i64 1), .op (.iadd 7 14)]]] := by
  rfl

theorem shl3 {j : Nat} (hj : j < 2 ^ 61) :
    UInt64.ofNat j <<< ((3 : UInt64) % UInt64.ofNat 64) = UInt64.ofNat (8 * j) := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftLeft]
  have h3 : ((3 : UInt64) % UInt64.ofNat 64).toNat % 64 = 3 := by decide
  rw [h3, UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.shiftLeft_eq]
  rw [Nat.mod_eq_of_lt (by omega : j < 2 ^ 64), Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  omega

/-- What the thread's function ends with, from `w0` on the word `p`: the world
    as it was, and in its status slot the wrapping sum of the 128 words at `p`. -/
def workerPost (w0 : World) (p : UInt64) : Post where
  ok Γ w := w = w0 ∧ Γ[16]? = some (.sc .i64 (wsum w0.mem p 128))

/-- The loop's carries: `i` trips done, and the sum of the first `i` words. -/
def workerInv (w0 : World) (p : UInt64) (cs : List V) (w : World) : Prop :=
  w = w0 ∧ ∃ i, i ≤ 128 ∧ cs = [.sc .i64 (UInt64.ofNat i), .sc .i64 (wsum w0.mem p i)]

/-- The slots after the constants. -/
def E0 (p : UInt64) : Env := #[.sc .i64 p, .sc .i64 128, .sc .i64 0, .sc .i64 0]

theorem ofInt3 : ofInt .i64 3 = .sc .i64 3 := rfl
theorem ofInt1 : ofInt .i64 1 = .sc .i64 1 := rfl
theorem ofInt0 : ofInt .i64 0 = .sc .i64 0 := rfl
theorem ofInt8 : ofInt .i64 8 = .sc .i64 8 := rfl
theorem ofInt1024 : ofInt .i64 1024 = .sc .i64 1024 := rfl
theorem ofInt2048 : ofInt .i64 2048 = .sc .i64 2048 := rfl
theorem ofInt3072 : ofInt .i64 3072 = .sc .i64 3072 := rfl

theorem worker_triple (cfg : Cfg) (w0 : World) (p : UInt64) :
    Triple cfg (At #[.sc .i64 p] w0) workerCode (workerPost w0 p) := by
  rw [workerCode_eq]
  refine cons_rule (R := fun Γ w => Γ = E0 p ∧ w = w0) (straight_rule ?_) ?_
  · rintro Γ w ⟨rfl, rfl⟩
    simp [runStmts, runStmt, evalOp, OutSat, E0, ofInt]
    exact ⟨rfl, rfl⟩
  refine cons_rule (R := (workerPost w0 p).ok) ?_ (nil_rule fun _ _ h => h)
  refine loop_rule (fun _ cs w => workerInv w0 p cs w) ?_ ?_
  · rintro Γ w cs ⟨rfl, rfl⟩ hcs
    simp [E0] at hcs
    subst hcs
    exact ⟨rfl, 0, by omega, by simp [wsum]⟩
  · rintro Γ w1 cs w ⟨rfl, -⟩ ⟨hw, i, hi, rfl⟩
    have hsz : (E0 p).size = 4 := rfl
    rw [hsz]
    refine cons_rule (R := fun Γ w' => Γ = (Sem.bindAt (E0 p) 4
        [.sc .i64 (UInt64.ofNat i), .sc .i64 (wsum w0.mem p i)]).push
        (boolV (cmpInt .ult .i64 (UInt64.ofNat i) 128)) ∧ w' = w0) (straight_rule ?_) (nil_rule ?_)
    · rintro Γ w' ⟨rfl, rfl⟩
      simp [runStmts, runStmt, evalOp, OutSat, Sem.get, Sem.bindAt, E0, zipIntCmp, ClifTy.isInt, hw]
    · rintro Γ w' ⟨rfl, rfl⟩
      intro t f hf
      simp [Sem.bindAt, E0] at hf
      simp only [boolV, V.sc.injEq] at hf
      obtain ⟨rfl, rfl⟩ := hf
      have hab : slotsOf (slotsOf (4 + 2) [Piece.straight [Stmt.op (Op.icmp ICmpCond.ult 4 1)]] + 2)
          [Piece.straight [Stmt.op (Op.iconst ClifTy.i64 3), Stmt.op (Op.ishl 7 9), Stmt.op (Op.iadd 0 10),
            Stmt.op (Op.load { ty := ClifTy.i64 } 11), Stmt.op (Op.iadd 8 12), Stmt.op (Op.iconst ClifTy.i64 1),
            Stmt.op (Op.iadd 7 14)]] = 16 := rfl
      have h7 : slotsOf (4 + 2) [Piece.straight [Stmt.op (Op.icmp ICmpCond.ult 4 1)]] = 7 := rfl
      have hlt : (UInt64.ofNat i < 128) ↔ i < 128 := by
        rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]; rfl
      refine ⟨fun hx vs hvs => ?_, fun hx => ?_⟩
      · -- the loop leaves: every word is summed
        have hi' : i = 128 := by
          by_cases h : UInt64.ofNat i < 128
          · simp [cmpInt, h] at hx
          · have := (not_congr hlt).mp h; omega
        subst hi'
        simp [Sem.bindAt, E0] at hvs
        subst hvs
        simp only [List.length_cons, List.length_nil, Nat.reduceAdd] at hab ⊢
        rw [hab]
        refine ⟨rfl, ?_⟩
        simp [Sem.bindAt, E0, Array.getElem_push]
      · have hi' : i < 128 := by
          by_cases h : UInt64.ofNat i < 128
          · exact hlt.mp h
          · simp [cmpInt, h] at hx
        simp only [List.length_cons, List.length_nil, Nat.reduceAdd] at h7 ⊢
        rw [h7]
        refine cons_rule (R := fun Γ w2 => w2 = w' ∧ Γ[15]? = some (.sc .i64 (UInt64.ofNat (i + 1))) ∧
          Γ[13]? = some (.sc .i64 (wsum w'.mem p (i + 1)))) (straight_rule ?_) (nil_rule ?_)
        · rintro Γ w2 ⟨rfl, rfl⟩
          have h3 : UInt64.ofNat i <<< 3 = UInt64.ofNat (8 * i) := shl3 (j := i) (by omega)
          cases hl : w2.mem.load (p + 8 * UInt64.ofNat i) 8 with
          | none =>
            simp [runStmts, runStmt, evalOp, OutSat, Sem.get, Sem.bindAt, E0, ClifTy.isInt, bin, shiftBin, norm64,
              ofInt3, ClifTy.width, h3, ClifTy.lanes, tyBytes, hl]
            rfl
          | some v =>
            simp [runStmts, runStmt, evalOp, OutSat, Sem.get, Sem.bindAt, E0, ClifTy.isInt, bin, shiftBin, norm64,
              ofInt3, ofInt1, ClifTy.width, h3, ClifTy.lanes, tyBytes, hl, wsum_succ]
            done
        · rintro Γ w2 ⟨rfl, h15, h13⟩ next hnext
          simp [h15, h13] at hnext
          subst hnext
          exact ⟨rfl, i + 1, by omega, by simp⟩

/-- The thread's function as a term of the program's table. -/
def workerFn : Fn :=
  let (a, c, e, _) := runAns ParSum.worker [.i64]
  { params := [.i64], code := c, status := a, env := e }

theorem workerFn_code : workerFn.code = workerCode := rfl
theorem workerFn_status : workerFn.status = some 16 := rfl

/-- **A call of the thread's function that returns** answers the wrapping sum
    of the 128 words at its argument and leaves the world as it was. -/
theorem worker_call {c : Cfg} {p : UInt64} {w : World} {r : Option V} {w' : World}
    (h : Sem.callFn c workerFn [.sc .i64 p] w = .ok (r, w')) :
    r = some (.sc .i64 (wsum w.mem p 128)) ∧ w' = w := by
  unfold Sem.callFn at h
  rw [if_neg (fun h => h rfl)] at h
  have ht := worker_triple c w p c.steps #[.sc .i64 p] w ⟨rfl, rfl⟩
  rw [← workerFn_code] at ht
  have hl : [V.sc .i64 p].toArray = #[.sc .i64 p] := rfl
  rw [hl] at h
  split at h
  · rename_i Γ w2 hr
    rw [hr] at ht
    obtain ⟨rfl, h16⟩ := ht
    injection h with h
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    rw [workerFn_status]
    simp [h16]
  all_goals cases h

-- ---------------------------------------------------------------------------
-- What a thread sees
-- ---------------------------------------------------------------------------

/-- The wrapping sum of the `k` words of `a` from byte `o`. -/
def bsum (a : ByteArray) (o k : Nat) : UInt64 :=
  (List.range k).foldl (fun s j => s + Contracts.bytesAt a (o + 8 * j) 8) 0

theorem bsum_add (a : ByteArray) (o m : Nat) : ∀ n, bsum a o (m + n) = bsum a o m + bsum a (o + 8 * m) n
  | 0 => by simp [bsum]
  | n + 1 => by
      rw [← Nat.add_assoc]
      simp only [bsum, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil] at *
      rw [show (List.range (m + n)).foldl (fun s j => s + Contracts.bytesAt a (o + 8 * j) 8) 0 =
          bsum a o (m + n) from rfl, bsum_add a o m n]
      simp only [bsum, UInt64.add_assoc]
      congr 3; omega

/-- **The arena's 512 words, summed**: what `main` writes. -/
def arenaSum (a : ByteArray) : UInt64 := bsum a 0 512

theorem arenaSum_quarters (a : ByteArray) :
    arenaSum a = bsum a 0 128 + bsum a 1024 128 + bsum a 2048 128 + bsum a 3072 128 := by
  unfold arenaSum
  rw [show 512 = 384 + 128 from rfl, bsum_add, show 384 = 256 + 128 from rfl, bsum_add,
    show 256 = 128 + 128 from rfl, bsum_add]

theorem get!_append_right (a b : ByteArray) (i : Nat) : (a ++ b).get! (a.size + i) = b.get! i := by
  apply get!_of_data
  rw [ByteArray.data_append, Array.getElem?_append_right (by simp [ByteArray.size_data])]
  congr 1
  simp [ByteArray.size_data]

/-- A thread's view reads its span's bytes where the program knows them. -/
theorem view_load (t : Thread) (bs : ByteArray) (hb : bs.size = t.len) (h36 : t.off + t.len < 2 ^ 36)
    {j : Nat} (hj : j + 8 ≤ t.len) :
    (t.view bs).mem.load (regionBase .arena + UInt64.ofNat (t.off + j)) 8 = some (Contracts.bytesAt bs j 8) := by
  have hd := decodeAddr_base .arena (t.off + j) (by rw [regionSpan_toNat]; omega)
  have hsz : (Thread.view t bs).mem.arena.size = t.off + t.len := by
    show (ByteArray.mk (Array.replicate t.off 0) ++ bs).size = _
    rw [ByteArray.size_append, hb]
    show (Array.replicate t.off (0 : UInt8)).size + t.len = _
    rw [Array.size_replicate]
  rw [Contracts.load_of hd (by show _ ≤ (Thread.view t bs).mem.arena.size; omega)
    (Mem.readable_of_ne_pinned (by decide) (by simp [Mem.LentOk, Mem.lentClear, Thread.view]))]
  refine congrArg some ?_
  show Contracts.bytesAt (ByteArray.mk (Array.replicate t.off 0) ++ bs) (t.off + j) 8 = _
  unfold Contracts.bytesAt
  refine Contracts.foldr_fun_congr
    (fun i => ((ByteArray.mk (Array.replicate t.off 0) ++ bs).get! (t.off + j + i)).toUInt64)
    (fun i => (bs.get! (j + i)).toUInt64) _ _ fun i _ => congrArg UInt8.toUInt64 ?_
  have hz : (ByteArray.mk (Array.replicate t.off (0 : UInt8))).size = t.off := by
    show (Array.replicate t.off (0 : UInt8)).size = _; rw [Array.size_replicate]
  rw [show t.off + j + i = (ByteArray.mk (Array.replicate t.off 0)).size + (j + i) by omega,
    get!_append_right]

/-- Thread `k`: `u0:2` on quarter `k`'s address, owning quarter `k`. -/
def thr (k : Nat) : Thread := ⟨2, regionBase .arena + UInt64.ofNat (1024 * k), 1024 * k, 1024⟩

/-- **In its view, thread `k`'s words are its quarter's.** -/
theorem view_wsum {a : ByteArray} {k : Nat} (hk : k < 4) (ha : 4096 ≤ a.size) :
    wsum ((thr k).view (a.extract (1024 * k) (1024 * k + 1024))).mem (thr k).arg 128 = bsum a (1024 * k) 128 := by
  have hb : (a.extract (1024 * k) (1024 * k + 1024)).size = (thr k).len := by
    simp only [ByteArray.size_extract, thr]; omega
  unfold wsum bsum
  refine foldl_congr_mem _ _ _ _ fun j hj s => ?_
  have hj := List.mem_range.mp hj
  have e : (thr k).arg + UInt64.ofNat (8 * j) = regionBase .arena + UInt64.ofNat ((thr k).off + 8 * j) := by
    simp only [thr]; rw [UInt64.add_assoc, ← UInt64.ofNat_add]
  rw [e, view_load (thr k) _ hb (by simp only [thr]; omega) (by simp only [thr]; omega)]
  simp only [Option.getD_some]
  refine congrArg (s + ·) ?_
  unfold Contracts.bytesAt
  refine Contracts.foldr_fun_congr
    (fun i => ((a.extract (1024 * k) (1024 * k + 1024)).get! (8 * j + i)).toUInt64)
    (fun i => (a.get! (1024 * k + 8 * j + i)).toUInt64) _ _ fun i hi => congrArg UInt8.toUInt64 ?_
  have hi := List.mem_range.mp hi
  rw [show 1024 * k + 1024 = 1024 * k + 1024 from rfl, extract_get (by omega) (by omega)]
  congr 1; omega

-- ---------------------------------------------------------------------------
-- What a join hands back
-- ---------------------------------------------------------------------------

theorem ba_ext {a b : ByteArray} (hs : a.size = b.size) (h : ∀ i < a.size, a.get! i = b.get! i) : a = b := by
  obtain ⟨a⟩ := a
  obtain ⟨b⟩ := b
  congr 1
  apply Array.ext hs fun i h1 h2 => ?_
  have := h i h1
  simp only [ByteArray.get!, getElem!_pos a i h1, getElem!_pos b i h2] at this
  exact this

/-- The span a thread was lent, cut from its view, is the bytes it was lent. -/
theorem view_extract (t : Thread) (bs : ByteArray) (hb : bs.size = t.len) :
    (t.view bs).mem.arena.extract t.off (t.off + t.len) = bs := by
  have hz : (ByteArray.mk (Array.replicate t.off (0 : UInt8))).size = t.off := by
    show (Array.replicate t.off (0 : UInt8)).size = _; rw [Array.size_replicate]
  have hsz : (t.view bs).mem.arena.size = t.off + t.len := by
    show (ByteArray.mk (Array.replicate t.off 0) ++ bs).size = _
    rw [ByteArray.size_append, hb, hz]
  apply ba_ext (by rw [ByteArray.size_extract, hsz, hb]; omega) fun i hi => ?_
  rw [ByteArray.size_extract, hsz] at hi
  rw [extract_get (by omega) (by omega)]
  show (ByteArray.mk (Array.replicate t.off 0) ++ bs).get! (t.off + i) = _
  rw [show t.off + i = (ByteArray.mk (Array.replicate t.off 0)).size + i by omega, get!_append_right]

/-- Handing back the bytes a thread was lent, as they were, leaves the arena. -/
theorem copySlice_self {a : ByteArray} {o l : Nat} (hl : o + l ≤ a.size) :
    (fitBytes l (a.extract o (o + l))).copySlice 0 a o l = a := by
  have hf := fitBytes_size l (a.extract o (o + l))
  apply ba_ext (copySlice_size hf hl) fun i hi => ?_
  rw [copySlice_size hf hl] at hi
  by_cases hin : o ≤ i ∧ i < o + l
  · obtain ⟨j, rfl⟩ : ∃ j, i = o + j := ⟨i - o, by omega⟩
    rw [copySlice_get_in hf hl (by omega), fitBytes_get (by omega) (by rw [ByteArray.size_extract]; omega),
      extract_get hl (by omega)]
  · exact copySlice_get_out hf hl (by omega)

/-- **An honest join of thread `k`**, whose run ends, hands back the quarter
    it was lent and answers the quarter's sum. -/
theorem join_out {c : Cfg} {L : Locals} {out : Thread → ByteArray → ByteArray × UInt64}
    (hon : ThreadHonest L out) (hL : ∀ a w', L 2 a w' = Sem.callFn c workerFn a w')
    (hend : ∀ bs, ∃ r wc, L 2 [.sc .i64 (thr k).arg] ((thr k).view bs) = .ok (r, wc))
    {a : ByteArray} (hk : k < 4) (ha : 4096 ≤ a.size) :
    out (thr k) (a.extract (1024 * k) (1024 * k + 1024)) =
      (a.extract (1024 * k) (1024 * k + 1024), bsum a (1024 * k) 128) := by
  obtain ⟨r, wc, hr⟩ := hend (a.extract (1024 * k) (1024 * k + 1024))
  have hr' := hr
  rw [hL] at hr'
  obtain ⟨rfl, rfl⟩ := worker_call hr'
  have ho := hon (thr k) _ _ _ hr
  rw [ho, view_wsum hk ha]
  have hb : (a.extract (1024 * k) (1024 * k + 1024)).size = (thr k).len := by
    simp only [ByteArray.size_extract, thr]; omega
  rw [view_extract _ _ hb]

-- ---------------------------------------------------------------------------
-- `main`
-- ---------------------------------------------------------------------------

def mainCode : Code := emit ParSum.code

def thn : List Stmt :=
  [.op (.funcAddr 2), .op (.iconst .i64 0), .op (.iadd 4 8), .op (.iconst .i64 0),
   .call (.ext (.base .threadSpawn)) [7, 9, 10],
   .op (.iconst .i64 1024), .op (.iadd 4 12), .op (.iconst .i64 0),
   .call (.ext (.base .threadSpawn)) [7, 13, 14],
   .op (.iconst .i64 2048), .op (.iadd 4 16), .op (.iconst .i64 0),
   .call (.ext (.base .threadSpawn)) [7, 17, 18],
   .op (.iconst .i64 3072), .op (.iadd 4 20), .op (.iconst .i64 0),
   .call (.ext (.base .threadSpawn)) [7, 21, 22],
   .op (.iconst .i64 0),
   .call (.ext (.base .threadJoin)) [11], .op (.iadd 24 25),
   .call (.ext (.base .threadJoin)) [15], .op (.iadd 26 27),
   .call (.ext (.base .threadJoin)) [19], .op (.iadd 28 29),
   .call (.ext (.base .threadJoin)) [23], .op (.iadd 30 31),
   .store .i64 32 2]

theorem mainCode_eq : mainCode =
    [.straight [.op .memory, .op (.iconst .i64 8), .op (.icmp .ule 5 3)],
     .ite { flag := 6, jTys := [] } [.straight thn] [] [] []] := by
  rfl

/-- What `main` ends with, from `w`: the eight bytes at the output are the
    wrapping sum of the arena's 512 words, as `w` held them. -/
def mainPost (w : World) : Post where
  ok _ w' := w'.mem.load (regionBase .out) 8 = some (arenaSum w.mem.arena)

/-- The slots as the branch starts. -/
def E1 (dl ol : UInt64) : Env :=
  #[.sc .i64 (regionBase .data), .sc .i64 dl, .sc .i64 (regionBase .out), .sc .i64 ol,
    .sc .i64 (regionBase .arena), .sc .i64 8, boolV (cmpInt .ule .i64 8 ol)]

set_option maxHeartbeats 4000000 in
/-- **What `main` computes.** On a machine whose joins are honest about the
    program's function `u0:2` and whose threads end, a run of `main` that
    returns, with room for its answer, leaves in the first eight bytes of its
    output the wrapping sum of the 512 words the arena held when it was
    called. -/
theorem main_triple (cfg c : Cfg) (L : Locals) (dl ol : UInt64) (w : World)
    (hon : ThreadHonest L w.threadOut) (hL : ∀ a w', L 2 a w' = Sem.callFn c workerFn a w')
    (hend : ∀ k < 4, ∀ bs, ∃ r wc, L 2 [.sc .i64 (thr k).arg] ((thr k).view bs) = .ok (r, wc))
    (hth : w.threads = #[]) (hlent : w.mem.lent = []) (ha : 4096 ≤ w.mem.arena.size)
    (hg : w.grants "thread" = true)
    (hlend : ∀ k < 4, w.lend 2 (thr k).arg = some ((thr k).off, (thr k).len))
    (hol : 8 ≤ ol.toNat) :
    Triple cfg (At (runArgs dl ol).toArray w) mainCode (mainPost w) := by
  rw [mainCode_eq]
  refine cons_rule (R := fun Γ w' => Γ = E1 dl ol ∧ w' = w) (straight_rule ?_) ?_
  · rintro Γ w' ⟨rfl, hw⟩
    subst w'
    simp [runStmts, runStmt, evalOp, OutSat, Sem.get, runArgs, E1, zipIntCmp, ClifTy.isInt, ofInt8]
  refine cons_rule (R := (mainPost w).ok) ?_ (nil_rule fun _ _ h => h)
  refine ite_rule ?_ ?_
  · rintro Γ w' t f ⟨rfl, hw⟩ hf -
    subst w'
    refine cons_rule (R := fun _ w' => (mainPost w).ok #[] w') (straight_rule ?_) (nil_rule fun _ _ h _ _ => h)
    rintro Γ w' ⟨rfl, hw⟩
    subst w'
    have l0 := hlend 0 (by decide); have l1 := hlend 1 (by decide)
    have l2 := hlend 2 (by decide); have l3 := hlend 3 (by decide)
    simp [thr] at l0 l1 l2 l3
    have j0 := join_out (k := 0) hon hL (hend 0 (by decide)) (by decide) ha
    have j1 := join_out (k := 1) hon hL (hend 1 (by decide)) (by decide) ha
    have j2 := join_out (k := 2) hon hL (hend 2 (by decide)) (by decide) ha
    have j3 := join_out (k := 3) hon hL (hend 3 (by decide)) (by decide) ha
    simp [thr] at j0 j1 j2 j3
    have c0 := copySlice_self (a := w.mem.arena) (o := 0) (l := 1024) (by omega)
    have c1 := copySlice_self (a := w.mem.arena) (o := 1024) (l := 1024) (by omega)
    have c2 := copySlice_self (a := w.mem.arena) (o := 2048) (l := 1024) (by omega)
    have c3 := copySlice_self (a := w.mem.arena) (o := 3072) (l := 1024) (by omega)
    simp at c0 c1 c2 c3
    have f1 : ¬ (fnBase + 2 < fnBase) := by decide
    have f2 : (fnBase + 2 - fnBase).toNat = 2 := by decide
    have s1 : ¬ w.mem.arena.size < 1024 := by omega
    have s2 : ¬ w.mem.arena.size < 2048 := by omega
    have s3 : ¬ w.mem.arena.size < 3072 := by omega
    have s4 : ¬ w.mem.arena.size < 4096 := by omega
    simp [thn, runStmts, runStmt, Sem.runStmt.runCall, evalOp, OutSat, Sem.get, E1, ClifTy.isInt, bin, norm64,
      ofInt0, ofInt1024, ofInt2048, ofInt3072, callOf, extCall, World.has, Ext.lib, Lib.builtin, baseCall,
      ffiThreadSpawn, ffiThreadJoin, obsCall, asBits, hth, hlent, hg, l0, l1, l2, l3,
      threadOf_handle, j0, j1, j2, j3, c0, c1, c2, c3, s1, s2, s3, s4, Mem.lentClear, f1, f2, tyBytes, ClifTy.width]
    have hd := decodeAddr_base .out 0 (by decide)
    simp at hd
    cases hm : ({ w.mem with lent := [] } : Mem).store (regionBase .out) 8
        (bsum w.mem.arena 0 128 + bsum w.mem.arena 1024 128 + bsum w.mem.arena 2048 128 +
          bsum w.mem.arena 3072 128) with
    | none => simp; rfl
    | some m =>
      simp
      show m.load (regionBase .out) 8 = some (arenaSum w.mem.arena)
      rw [arenaSum_quarters]
      exact Contracts.load_store8 hd (by decide) hm
  · rintro Γ w' t f ⟨rfl, hw⟩ hf hz
    have : (8 : UInt64) ≤ ol := by rw [UInt64.le_iff_toNat_le]; exact hol
    simp [E1, boolV, cmpInt, this] at hf
    obtain ⟨-, rfl⟩ := hf
    simp at hz

/-- **`main` sums the arena on its threads.** A run of `main` that returns,
    with room for its answer, leaves in the first eight bytes of its output
    the wrapping sum of the 512 words the arena held when it was called: on a
    machine whose joins are honest about the program's function `u0:2`
    (`ThreadHonest`, of the term `workerFn` the artifact ships at `u0:2`),
    whose threads end (a join waits for its thread, so a `main` that returns
    had every thread end), and which starts `main` with no thread running and
    the quarters `Ship.Threads` lends. -/
theorem main_computes (cfg c : Cfg) (L : Locals) (dl ol : UInt64) (w : World)
    (hon : ThreadHonest L w.threadOut) (hL : ∀ a w', L 2 a w' = Sem.callFn c workerFn a w')
    (hend : ∀ k < 4, ∀ bs, ∃ r wc, L 2 [.sc .i64 (thr k).arg] ((thr k).view bs) = .ok (r, wc))
    (hth : w.threads = #[]) (hlent : w.mem.lent = []) (ha : 4096 ≤ w.mem.arena.size)
    (hg : w.grants "thread" = true)
    (hlend : ∀ k < 4, w.lend 2 (thr k).arg = some ((thr k).off, (thr k).len))
    (hol : 8 ≤ ol.toNat) {obs : List Obs} {w' : World}
    (h : Sem.run cfg (runArgs dl ol) w (emit ParSum.code) = .ok obs w') :
    w'.mem.load (regionBase .out) 8 = some (arenaSum w.mem.arena) := by
  have ht := main_triple cfg c L dl ol w hon hL hend hth hlent ha hg hlend hol cfg.steps _ w ⟨rfl, rfl⟩
  unfold Sem.run at h
  rw [show emit ParSum.code = mainCode from rfl] at h
  split at h <;> try cases h
  rename_i Γ w2 hr
  rw [hr] at ht
  exact ht

end ParSumSpec

#print axioms ParSumSpec.main_computes
