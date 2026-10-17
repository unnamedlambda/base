import Ship.HtSpec
import AlgorithmLib.Check.Quiet

/-!
# The hash table's functions are what the checker takes them to be

The checker reads a call into `Lib.Ht` by the statements `Check.HtOk` collects,
over what a typestate says of a table (`HtObj`: a slot holding null or a table
of small values, as it stood before writes outside the heap). `htOk_linked` is
those statements of a program whose function table has the library's functions
at their places and is all code that calls only the C library and itself: each
is the function's own statement at a slot (`init_at` … `cleanup_at`, the
`_null` theorems for a slot holding null), with what the call leaves of the
rest of the world (`LibStep`) from what it keeps outside the heap
(`LibSpec.Outside`) and from its being quiet (`Check.termLocals_quiet`).

`HtObj.at` reads a typestate's table as one at its slot now: writes outside the
heap leave a table and its index (`OutSteps.tab`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace HtSpec
open AlgorithmLib.LibSpec AlgorithmLib.LibHt AlgorithmLib.Check

/-- Writes outside the heap leave a table and its index. -/
theorem OutSteps.tab {w0 w : World} {s : UInt64} {T : Ht} {L : Lay} (h : OutSteps w0 w) (hT : Tab w0 s T L)
    (hI : Idx w0 L (keysOf T)) : Tab w s T L ∧ Idx w L (keysOf T) := by
  induction h with
  | refl => exact ⟨hT, hI⟩
  | step _ hw hd hr ih =>
      obtain ⟨hT', hI'⟩ := ih
      exact ⟨hT'.wrote hw (clear_out hd hr),
        hI'.wrote hT'.bi hw (clear_out (s := s) (L := L) hd hr L.ix (L.icap * 8) (by simp [blocks]))⟩

/-- **What a typestate says of a slot is a table at the slot now**: null, or
    a table within the bounds. -/
theorem HtObj.at {w : World} {x : UInt64} {n k : Nat} (h : HtObj w x n k) :
    ∃ s r off, decodeAddr x = some (r, off) ∧ r ≠ .pinned ∧ w.mem.load x 8 = some s ∧
      (s = 0 ∨ ∃ T, HtAt w x s T ∧ (∀ kv ∈ T.entries, kv.2.size ≤ n) ∧ T.handles.length ≤ k) := by
  obtain ⟨s, r, off, hd, hr, hl, hT⟩ := h
  refine ⟨s, r, off, hd, hr, hl, hT.imp id ?_⟩
  rintro ⟨w0, T, L, ho, hT, hI, hb, hc⟩
  obtain ⟨hT', hI'⟩ := ho.tab hT hI
  exact ⟨T, ⟨r, off, L, hd, hr, hl, hT', hI'⟩, hb, hc⟩

/-- A table at a slot, within bounds, is what a typestate says of the slot. -/
theorem HtAt.obj {w : World} {x s : UInt64} {T : Ht} {n k : Nat} (h : HtAt w x s T)
    (hb : ∀ kv ∈ T.entries, kv.2.size ≤ n) (hc : T.handles.length ≤ k) : HtObj w x n k := by
  obtain ⟨r, off, L, hd, hr, hl, hT, hI⟩ := h
  exact ⟨s, r, off, hd, hr, hl, .inr ⟨w, T, L, .refl w, hT, hI, hb, hc⟩⟩

/-- A slot outside the heap holding null is what a typestate says of it. -/
theorem HtObj.null {w : World} {x : UInt64} {r : Region} {off : Nat} (hd : decodeAddr x = some (r, off))
    (hr : r ≠ .pinned) (hl : w.mem.load x 8 = some 0) (n k : Nat) : HtObj w x n k :=
  ⟨0, r, off, hd, hr, hl, .inl rfl⟩

/-- A cell clear, as numbers, of the bytes at `p` misses them. -/
theorem misses_num {x p : UInt64} {n m : Nat} (h : p.toNat + m ≤ x.toNat ∨ x.toNat + 8 ≤ p.toNat) (hn : n ≤ m) :
    Misses x p n := by
  intro _ r off r' off' hx hp
  have e1 := toNat_of_decode hx
  have e2 := toNat_of_decode hp
  by_cases hrr : r = r'
  · subst hrr; right; omega
  · exact .inl hrr

theorem out_cell {c : UInt64} (hc : cellOk c = true) : Out c := by
  obtain ⟨r, off, hd, hp, -⟩ := cellOk_spec hc
  intro r' off' hd'
  rw [hd] at hd'; cases hd'; exact hp

/-- **What a call leaves of the rest of the world**, from what it keeps
    outside the heap and its being quiet. -/
theorem libStep_of {w w' : World} {P : UInt64 → Nat → Prop} {lo hi : Nat} (ho : Outside w w' P)
    (hq : Quiet w w') (hc : Calm w)
    (hP : ∀ c, cellOk c = true → hi ≤ c.toNat ∨ c.toNat + 8 ≤ lo → P c 8) : LibStep w w' lo hi := by
  have hc' := ho.calm hc
  refine ⟨fun r hr => ho.size r hr, hc'.1, by rw [hc'.2]; rfl, fun c v hcell hm hl => ?_, hq.calls⟩
  exact ho.keeps.load c 8 v (by decide) ⟨load_inW hl, out_cell hcell, hP c hcell hm⟩ hl

/-- A call that changed nothing left the rest of the world. -/
theorem libStep_refl {w : World} (hc : Calm w) (lo hi : Nat) : LibStep w w lo hi :=
  ⟨fun _ _ => rfl, hc.1, by rw [hc.2]; rfl, fun _ _ _ _ h => h, ⟨[], rfl⟩⟩

theorem calm_of {w : World} (hz : w.mem.frozen = false) (hi : w.mem.busy.isEmpty = true) : Calm w :=
  ⟨hz, List.isEmpty_iff.mp hi⟩

/-- Room outside pinned memory, where no worker runs, is room. -/
theorem NPRoom.room {w : World} {p : UInt64} {k : Nat} (h : NPRoom w p k) (hz : w.mem.frozen = false) :
    Room w p k ∧ Out p := by
  obtain ⟨r, off, hd, hr, hs, h36⟩ := h
  refine ⟨.inr ⟨r, off, hd, hs, h36, ?_⟩, fun r' off' hd' => by rw [hd] at hd'; cases hd'; exact hr⟩
  simp only [Mem.reachable, hz, Bool.not_false, Bool.true_and, Bool.or_eq_true]
  exact Or.inl (by cases r <;> first | rfl | exact absurd rfl hr)

section linked
variable {fns : Nat → Option Fn} {env : FnEnv} {steps : Nat}

/-- A call whose statement is a `CallsD`, as the checker reads it. -/
theorem libCall_of_callsD {f : Ffi} {vs : List V} {w : World} {Q Q' : World → Prop} {k : Nat}
    (hq : ∀ i f, fns i = some f → quietCode f.code = true)
    (h : CallsD (termLocals fns env steps k) f.libSlot vs w Q) (hQ : ∀ w', Q w' → Quiet w w' → Q' w') :
    LibCall (termLocals fns env steps k) f vs w Q' := by
  unfold CallsD at h
  unfold LibCall
  cases hl : termLocals fns env steps k f.libSlot vs w with
  | error e => rw [hl] at h; exact h
  | ok p =>
      obtain ⟨r, w'⟩ := p
      rw [hl] at h
      exact hQ w' h (termLocals_quiet hq k _ _ _ _ _ hl)

/-- A call whose statement is a `CallsA`, as the checker reads it. -/
theorem libCall_of_callsA {f : Ffi} {vs : List V} {w : World} {Q : UInt64 → World → Prop} {Q' : World → Prop}
    {k : Nat} (hq : ∀ i f, fns i = some f → quietCode f.code = true)
    (h : CallsA (termLocals fns env steps k) f.libSlot vs w Q) (hQ : ∀ x w', Q x w' → Quiet w w' → Q' w') :
    LibCall (termLocals fns env steps k) f vs w Q' := by
  unfold CallsA at h
  unfold LibCall
  cases hl : termLocals fns env steps k f.libSlot vs w with
  | error e => rw [hl] at h; exact h
  | ok p =>
      obtain ⟨r, w'⟩ := p
      rw [hl] at h
      obtain ⟨v, -, hv⟩ := h
      exact hQ v w' hv (termLocals_quiet hq k _ _ _ _ _ hl)

/-- Values within a bound stay within the larger of it and a new value's
    length, set. -/
theorem set_bound {T : Ht} {kB vB : ByteArray} {n : Nat} (hb : ∀ kv ∈ T.entries, kv.2.size ≤ n) :
    ∀ kv ∈ (T.set kB vB).entries, kv.2.size ≤ max n vB.size := by
  intro kv hkv
  unfold Ht.set at hkv
  split at hkv
  · simp only [List.mem_map] at hkv
    obtain ⟨e, he, rfl⟩ := hkv
    split
    · exact Nat.le_max_right _ _
    · exact Nat.le_trans (hb e he) (Nat.le_max_left _ _)
  · simp only [List.mem_append, List.mem_singleton] at hkv
    rcases hkv with h | rfl
    · exact Nat.le_trans (hb kv h) (Nat.le_max_left _ _)
    · exact Nat.le_max_right _ _

theorem set_handles (T : Ht) (kB vB : ByteArray) : (T.set kB vB).handles = T.handles := by
  unfold Ht.set; split <;> rfl

/-- **The hash table's functions, in a linked program, are what the checker
    takes them to be**: where the program's function table has the library's
    functions at their places and every function of it calls only the C
    library and the program's own, at every call depth. -/
theorem htOk_linked (hq : ∀ i f, fns i = some f → quietCode f.code = true)
    (hInit : fns (Ffi.libSlot .htInit) = some (implFn [.i64] (.void init)))
    (hCleanup : fns (Ffi.libSlot .htCleanup) = some (implFn [.i64] (.void cleanup)))
    (hCreate : fns (Ffi.libSlot .htCreate) = some (implFn [.i64] (.status create)))
    (hInsert : fns (Ffi.libSlot .htInsert) =
      some (implFn [.i64, .i64, .i32, .i64, .i32] (.void LibHt.insert)))
    (hLookup : fns (Ffi.libSlot .htLookup) = some (implFn [.i64, .i64, .i32, .i64] (.status lookup)))
    (k : Nat) : HtOk (termLocals fns env steps k) where
  init := fun w x hz hi hcell hroom => by
    obtain ⟨r, off, hd, hr, hs, -⟩ := hroom
    have hc := calm_of hz hi
    refine libCall_of_callsD hq (init_at hInit w x r off hc hd hr hs k) fun w' ⟨⟨s, hl, hT⟩, ho⟩ hquiet => ?_
    refine ⟨?_, libStep_of ho hquiet hc fun c _ hm => misses_num (m := 8) (by omega) (Nat.le_refl _)⟩
    rcases hT with rfl | hT
    · exact HtObj.null hd hr hl 0 0
    · exact hT.obj (fun kv hkv => by cases hkv) (Nat.le_refl _)
  cleanup := fun w x n j hz hi hobj => by
    have hc := calm_of hz hi
    obtain ⟨s, r, off, hd, hr, hl, hT⟩ := hobj.at
    have key : CallsD (termLocals fns env steps k) (Ffi.libSlot .htCleanup) [.sc .i64 x] w (fun w' =>
        w'.mem.load x 8 = some 0 ∧ Outside w w' (Misses x)) := by
      rcases hT with rfl | ⟨T, hat, -, -⟩
      · exact cleanup_calls hCleanup w x 0 r off hc hd hr hl (.inl rfl) k
      · exact cleanup_at hCleanup hat k
    exact libCall_of_callsD hq key fun w' ⟨hl', ho⟩ hquiet =>
      ⟨hl', libStep_of ho hquiet hc fun c _ hm => misses_num (m := 8) (by omega) (Nat.le_refl _)⟩
  create := fun w x s n j hz hi hobj hl hj => by
    have hc := calm_of hz hi
    obtain ⟨s', r, off, hd, hr, hl', hT⟩ := hobj.at
    rw [hl] at hl'; cases hl'
    rcases hT with rfl | ⟨T, hat, hb, hh⟩
    · refine libCall_of_callsA hq (Ans.calls hCreate (fun cfg => create_null cfg w) k) fun v w' ⟨_, e⟩ _ => ?_
      subst e
      exact ⟨HtObj.null hd hr hl n (j + 1), libStep_refl hc 0 0⟩
    · refine libCall_of_callsA hq (create_at hCreate hat (by omega) k) fun v w' ⟨_, hat', ho⟩ hquiet => ?_
      exact ⟨hat'.obj hb (by simp only [List.length_append, List.length_singleton]; omega),
        libStep_of ho hquiet hc fun _ _ _ => trivial⟩
  insert := fun w x s kp kl vp vl n j hz hi hobj hl hkl hvl hrk hrv => by
    have hc := calm_of hz hi
    obtain ⟨s', r, off, hd, hr, hl', hT⟩ := hobj.at
    rw [hl] at hl'; cases hl'
    rcases hT with rfl | ⟨T, hat, hb, hh⟩
    · refine libCall_of_callsD hq (Does.calls hInsert (fun cfg => insert_null cfg w kp kl vp vl) k)
        fun w' e _ => ?_
      subst e
      exact ⟨HtObj.null hd hr hl _ j, libStep_refl hc 0 0⟩
    · obtain ⟨hRk, hOk⟩ := NPRoom.room hrk hz
      obtain ⟨hRv, hOv⟩ := NPRoom.room hrv hz
      obtain ⟨kb, hkb, hBk⟩ := hRk.bytes
      obtain ⟨vb, hvb, hBv⟩ := hRv.bytes
      refine libCall_of_callsD hq (insert_at hInsert hat kp kl vp vl kb vb hBk hBv
        (by rw [hkb]; exact (UInt64.ofNat_toNat (x := kl)).symm) (by omega)
        (by rw [hvb]; exact (UInt64.ofNat_toNat (x := vl)).symm) (by omega) hOk hOv k)
        fun w' ⟨hat', ho⟩ hquiet => ?_
      refine ⟨?_, libStep_of ho hquiet hc fun _ _ _ => trivial⟩
      rcases hat' with hat' | hat'
      · split at hat'
        · refine hat'.obj (fun kv hkv => ?_) (by rw [set_handles]; exact hh)
          have := set_bound (kB := ByteArray.mk kb.toArray) (vB := ByteArray.mk vb.toArray) hb kv hkv
          have hsz : (ByteArray.mk vb.toArray).size = vl.toNat := by
            show vb.toArray.size = _; simp [hvb]
          rw [hsz] at this; exact this
        · exact hat'.obj (fun kv hkv => Nat.le_trans (hb kv hkv) (Nat.le_max_left _ _)) hh
      · exact hat'.obj (fun kv hkv => Nat.le_trans (hb kv hkv) (Nat.le_max_left _ _)) hh
  lookup := fun w x s kp kl rp n j hz hi hobj hl hkl hrk hrr hmiss => by
    have hc := calm_of hz hi
    obtain ⟨s', r, off, hd, hr, hl', hT⟩ := hobj.at
    rw [hl] at hl'; cases hl'
    rcases hT with rfl | ⟨T, hat, hb, hh⟩
    · refine libCall_of_callsA hq (Ans.calls hLookup (fun cfg => lookup_null cfg w kp kl rp) k)
        fun v w' ⟨_, e⟩ _ => ?_
      subst e
      exact ⟨HtObj.null hd hr hl n j, libStep_refl hc _ _⟩
    · obtain ⟨hRk, hOk⟩ := NPRoom.room hrk hz
      obtain ⟨hRr, hOr⟩ := NPRoom.room hrr hz
      obtain ⟨kb, hkb, hBk⟩ := hRk.bytes
      have hroomP : ∀ p (hp : p < T.entries.length), (T.entries[p]'hp).1.data.toList = kb →
          Room w rp (T.entries[p]'hp).2.size ∧ Misses x rp (T.entries[p]'hp).2.size := fun p hp _ => by
        have hle : (T.entries[p]'hp).2.size ≤ n := hb _ (List.getElem_mem hp)
        refine ⟨?_, misses_num hmiss hle⟩
        rcases hRr with h0 | ⟨r', off', hd', hs', h36', hre'⟩
        · exact .inl (by omega)
        · exact .inr ⟨r', off', hd', by omega, by omega, reachable_sub hre' (Nat.le_refl _) (by omega)⟩
      refine libCall_of_callsA hq (lookup_at hLookup hat kp kl rp kb hBk
        (by rw [hkb]; exact (UInt64.ofNat_toNat (x := kl)).symm) (by omega) hOr hroomP k)
        fun v w' ⟨hat', L, hQ⟩ hquiet => ?_
      refine ⟨hat'.obj hb hh, ?_⟩
      unfold LookQ at hQ
      split at hQ
      · rcases hQ with ⟨-, -, e⟩ | ⟨p, hp, -, -, -, -, -, hW⟩
        · subst e; exact libStep_refl hc _ _
        · have hle : (T.entries[p]'hp).2.size ≤ n := hb _ (List.getElem_mem hp)
          have hst := hW.stays
          have hc' := hst.calm hc
          refine ⟨fun rg hrg => hst.size rg hrg, hc'.1, by rw [hc'.2]; rfl, fun c v hcell hm hlc => ?_,
            hquiet.calls⟩
          obtain ⟨rc, oc, hdc, hpc, -⟩ := cellOk_spec hcell
          rw [hW.load hdc fun _ r2 o2 hd2 => ?_]
          · exact hlc
          · have e1 := toNat_of_decode hdc
            have e2 := toNat_of_decode hd2
            by_cases hrr : rc = r2
            · subst hrr; right; omega
            · exact .inl hrr
      · obtain ⟨-, e⟩ := hQ
        subst e; exact libStep_refl hc _ _

end linked

end HtSpec
