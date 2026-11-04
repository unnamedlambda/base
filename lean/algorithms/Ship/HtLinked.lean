import AlgorithmLib.Proof.Typestate
import Ship.HtSpec
import Ship.HtRuns

/-!
# A linked program over the hash table, proven as it is linked

`roundTrip` makes a table, sets one key, reads it back and frees the table,
through `HtDiff.lib`: its calls are to the library's functions, the program's
own (`HtDiff.fns`), as an artifact's are once linked. `roundTrip_linked` is
about that program: from any world with no worker running and 256 bytes of
arena, at any call depth, it neither misuses a call nor faults, and it leaves
null in its slot, the table freed. Nothing in it is about the entry points'
model: each call is read by the library function's own statement
(`Ship.HtSpec`'s `_at` theorems, through `LibSpec.wp_callA` and
`LibSpec.wp_callD`), and between calls the program keeps its own memory
(`LibSpec.Outside`). Where `htInit` could not allocate, every later call
answers that there is no context (`create_null`, `insert_null`,
`lookup_null`), and `htCleanup` frees nothing.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace HtLinked
open AlgorithmLib.LibSpec AlgorithmLib.LibHt HtSpec

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- A table made, one key set and read back, and the table freed. -/
def roundTrip (api : HtDiff.Api V L) : Prog V L Unit := do
  let base ← basePtr
  let key ← iaddImm base 16
  let val ← iaddImm base 32
  let out ← iaddImm base 128
  let k8 ← iconst32 8
  api.init base
  let c ← load64 base
  let _ ← api.create c
  api.insert c key k8 val k8
  let _ ← api.lookup c key k8 out
  api.cleanup base

theorem roundTrip_fine :
    Fine (emitGo (fnBody ptrParams (roundTrip HtDiff.lib : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The program's own memory: no worker running, its key and value, and room
    for the value read back. -/
def Mine (w : World) (base : UInt64) (kb vb : List UInt8) : Prop :=
  Calm w ∧ Bytes w (base + c64 16) kb ∧ Bytes w (base + c64 32) vb ∧ Room w (base + c64 128) 8

set_option maxHeartbeats 6400000 in
/-- **The linked program neither misuses a call nor faults, and leaves its
    slot null**: the table's functions are the library's, called at any call
    depth, from any world with no worker running and 256 bytes of arena. -/
theorem roundTrip_linked (env : FnEnv) (steps k : Nat) (w : World) (hc : Calm w) (hlent : w.mem.lent = [])
    (hsz : 256 ≤ (w.mem.region .arena).size) (d dl o ol : UInt64) :
    Does { env, steps, locals := termLocals HtDiff.fns env steps k }
      [.sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol] w
      (roundTrip HtDiff.lib) ptrParams (fun w' => w'.mem.load (regionBase .arena) 8 = some 0) := by
  refine does_of_wp ?_ roundTrip_fine rfl
  simp only [fnBody, show (ptrParams == ptrParams) = true from rfl, if_true, withMemory, wp_memory]
  simp only [roundTrip, HtDiff.lib, HtDiff.lo32, basePtr, AlgorithmLib.Prog.callLocal,
    AlgorithmLib.Prog.callLocalVoid, wp_bind, wp_params, ptrParams, carriesFrom, wp_ret, wp_pure,
    List.cons_append, List.nil_append]
  have h0 : ((#[V.sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol] : Env).push (.sc .i64 (regionBase .arena)))[4]? =
    some (.sc .i64 (regionBase .arena)) := rfl
  generalize hbase : regionBase .arena = base at h0 ⊢
  have hsp : ∀ j, j < 256 → j < regionSpan.toNat := fun j hj => by rw [Sem.regionSpan_toNat]; omega
  have hd0 : decodeAddr base = some (.arena, 0) := by
    have := Sem.decodeAddr_base .arena 0 (hsp 0 (by decide))
    rwa [add_ofNat_zero, hbase] at this
  have hdk : decodeAddr (base + c64 16) = some (.arena, 16) := by
    rw [← hbase, show c64 16 = UInt64.ofNat 16 from by decide]
    exact Sem.decodeAddr_base .arena 16 (hsp 16 (by decide))
  have hdv : decodeAddr (base + c64 32) = some (.arena, 32) := by
    rw [← hbase, show c64 32 = UInt64.ofNat 32 from by decide]
    exact Sem.decodeAddr_base .arena 32 (hsp 32 (by decide))
  have hdo : decodeAddr (base + c64 128) = some (.arena, 128) := by
    rw [← hbase, show c64 128 = UInt64.ofNat 128 from by decide]
    exact Sem.decodeAddr_base .arena 128 (hsp 128 (by decide))
  have hna : Region.arena ≠ .pinned := by decide
  have ok : Out (base + c64 16) := out_of hdk hna
  have ov : Out (base + c64 32) := out_of hdv hna
  have oo : Out (base + c64 128) := out_of hdo hna
  have mk : ∀ n, Misses base (base + c64 16) n := fun n => misses_of hd0 hdk (Or.inr (Or.inl (by decide)))
  have mv : ∀ n, Misses base (base + c64 32) n := fun n => misses_of hd0 hdv (Or.inr (Or.inl (by decide)))
  have mo : ∀ n, Misses base (base + c64 128) n := fun n => misses_of hd0 hdo (Or.inr (Or.inl (by decide)))
  obtain ⟨kb, hkl, hkb⟩ := (room_out (w := w) (n := 8) hdk hna (by omega) (by decide) (Mem.lentOk_nil hlent _ _ _)).bytes
  obtain ⟨vb, hvl, hvb⟩ := (room_out (w := w) (n := 8) hdv hna (by omega) (by decide) (Mem.lentOk_nil hlent _ _ _)).bytes
  have hkne : kb ≠ [] := fun e => by rw [e] at hkl; cases hkl
  have hvne : vb ≠ [] := fun e => by rw [e] at hvl; cases hvl
  have M0 : Mine w base kb vb := ⟨hc, hkb, hvb, room_out hdo hna (by omega) (by decide) (Mem.lentOk_nil hlent _ _ _)⟩
  have keep : ∀ w1 w2, Outside w1 w2 (Misses base) → Mine w1 base kb vb → Mine w2 base kb vb :=
    fun w1 w2 O M => ⟨O.calm M.1, O.keeps.bytes _ kb hkne ⟨M.2.1.inW hkne, ok, mk _⟩ M.2.1,
      O.keeps.bytes _ vb hvne ⟨M.2.2.1.inW hvne, ov, mv _⟩ M.2.2.1,
      O.keeps.room _ 8 (by decide) ⟨M.2.2.2.inW (by decide), oo, mo _⟩ M.2.2.2⟩
  refine wp_iaddImm (by slot) ?_
  refine wp_iaddImm (by slot) ?_
  refine wp_iaddImm (by slot) ?_
  apply wp_iconst32
  -- htInit
  refine wp_callD (cs := [.sc .i64 base]) rfl (by slots)
    (init_at (i := 1) rfl (obsCall w _ _) base .arena 0 hc hd0 hna (Nat.le_trans (by decide) hsz)
      (Mem.lentOk_nil hlent _ _ _) k) ?_
  rintro w1 ⟨⟨c, hl1, hcase⟩, O1⟩
  have M1 := keep _ _ O1 M0
  simp only [wp_ret]
  refine wp_load64 (by slot) hl1 ?_
  rcases hcase with rfl | hA
  · -- no context: every call answers that there is none
    refine wp_callA (cs := [.sc .i64 0]) rfl (by slots) (Ans.calls (i := 3) rfl (fun cfg => create_null cfg _) k) ?_
    rintro x w2 ⟨-, hw2⟩
    have hl2 : w2.mem.load base 8 = some 0 := by rw [hw2]; exact hl1
    have hc2 : Calm w2 := by rw [hw2]; exact M1.1
    clear hw2
    simp only [wp_ret]
    apply wp_iconst64
    refine wp_band (by slot) (by slot) ?_
    refine wp_callD (cs := [.sc .i64 0, .sc .i64 (base + c64 16), ofInt .i32 8, .sc .i64 (base + c64 32),
        ofInt .i32 8]) rfl (by slots) (Does.calls (i := 4) rfl (fun cfg => insert_null cfg _ _ _ _ _) k) ?_
    rintro w3 hw3
    have hl3 : w3.mem.load base 8 = some 0 := by rw [hw3]; exact hl2
    have hc3 : Calm w3 := by rw [hw3]; exact hc2
    clear hw3
    simp only [wp_ret]
    refine wp_callA (cs := [.sc .i64 0, .sc .i64 (base + c64 16), ofInt .i32 8, .sc .i64 (base + c64 128)])
      rfl (by slots) (Ans.calls (i := 5) rfl (fun cfg => lookup_null cfg _ _ _ _) k) ?_
    rintro x' w4 ⟨-, hw4⟩
    have hl4 : w4.mem.load base 8 = some 0 := by rw [hw4]; exact hl3
    have hc4 : Calm w4 := by rw [hw4]; exact hc3
    clear hw4
    simp only [wp_ret]
    apply wp_iconst64
    refine wp_band (by slot) (by slot) ?_
    refine wp_callD (cs := [.sc .i64 base]) rfl (by slots)
      (cleanup_calls (i := 2) rfl (obsCall w4 _ _) base 0 .arena 0 hc4 hd0 hna hl4 (Or.inl rfl) k) ?_
    rintro w5 ⟨h, -⟩
    simp only [wp_ret]
    exact h
  · -- a context holding the empty table
    refine wp_callA (cs := [.sc .i64 c]) rfl (by slots) (create_at (i := 3) rfl (hA.obs _ _) (by decide) k) ?_
    rintro x w2 ⟨-, hA2, O2⟩
    have M2 := keep _ _ (O2.mono fun _ _ _ => trivial) M1
    simp only [wp_ret]
    apply wp_iconst64
    refine wp_band (by slot) (by slot) ?_
    refine wp_callD (cs := [.sc .i64 c, .sc .i64 (base + c64 16), .sc .i32 (UInt64.ofNat 8),
        .sc .i64 (base + c64 32), .sc .i32 (UInt64.ofNat 8)]) rfl (by slots)
      (insert_at (i := 4) rfl (hA2.obs _ _) _ _ _ _ kb vb M2.2.1 M2.2.2.1 (by rw [hkl]) (by rw [hkl]; decide)
        (by rw [hvl]) (by rw [hvl]; decide) ok ov k) ?_
    rintro w3 ⟨hA3, O3⟩
    have M3 := keep _ _ (O3.mono fun _ _ _ => trivial) M2
    simp only [wp_ret]
    obtain ⟨T, hT3, hsize⟩ : ∃ T : Ht, HtAt w3 base c T ∧
        ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb → (T.entries[p]'hp).2.size = 8 := by
      rcases hA3 with h | h
      · rw [if_pos (by decide), set_absent (by simp [keysOf])] at h
        refine ⟨_, h, fun p hp _ => ?_⟩
        have hp0 : p = 0 := by
          have : p < 1 := hp
          omega
        subst hp0
        show (ByteArray.mk vb.toArray).size = 8
        rw [← hvl]
        simp [ByteArray.size]
      · exact ⟨_, h, fun p hp _ => absurd hp (Nat.not_lt_zero p)⟩
    refine wp_callA (cs := [.sc .i64 c, .sc .i64 (base + c64 16), .sc .i32 (UInt64.ofNat 8),
        .sc .i64 (base + c64 128)]) rfl (by slots)
      (lookup_at (i := 5) rfl (hT3.obs _ _) _ _ _ kb M3.2.1 (by rw [hkl]) (by rw [hkl]; decide) oo
        (fun p hp hk => by rw [hsize p hp hk]; exact ⟨M3.2.2.2, mo _⟩) k) ?_
    rintro x' w4 ⟨hA4, -⟩
    simp only [wp_ret]
    apply wp_iconst64
    refine wp_band (by slot) (by slot) ?_
    refine wp_callD (cs := [.sc .i64 base]) rfl (by slots) (cleanup_at (i := 2) rfl (hA4.obs _ _) k) ?_
    rintro w5 ⟨h, -⟩
    simp only [wp_ret]
    exact h

/-- The linked program neither misuses a call nor faults. -/
theorem roundTrip_no_misuse (env : FnEnv) (steps k : Nat) (w : World) (hc : Calm w) (hlent : w.mem.lent = [])
    (hsz : 256 ≤ (w.mem.region .arena).size) (d dl o ol : UInt64) (m : String) :
    Sem.run { env, steps, locals := termLocals HtDiff.fns env steps k }
        [.sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol] w
        (emit (roundTrip HtDiff.lib) ptrParams) ≠ .misuse m ∧
      Sem.run { env, steps, locals := termLocals HtDiff.fns env steps k }
        [.sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol] w
        (emit (roundTrip HtDiff.lib) ptrParams) ≠ .fault m :=
  (roundTrip_linked env steps k w hc hlent hsz d dl o ol).sound m

end HtLinked

#print axioms HtLinked.roundTrip_linked
#print axioms HtLinked.roundTrip_no_misuse
