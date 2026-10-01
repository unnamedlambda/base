module
public import AlgorithmLib.Host.Hoare
meta import AlgorithmLib.Host.Hoare
public import AlgorithmLib.Host.RaceRefine
meta import AlgorithmLib.Host.RaceRefine
public import AlgorithmLib.Host.StaticCong
meta import AlgorithmLib.Host.StaticCong
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.DevSpec` — the CUDA contracts as specifications

What each supported CUDA call needs so that it is not refused, and what it
leaves, in a vocabulary that says nothing about how a checker will use it.

## The vocabulary

The device tracker records, for every buffer, its last write and the reads
since, each as `(party, tick)`, and a vector clock per party. Everything a
race check asks is one shape of fact: **the accesses `es` happened before a
point whose clock is `c`** --- `Below es c`. The points are a party's clock
(the host is party 0), what a party knows when it issues (`know`: its own clock
and the host's), and an event's recorded clock. Every transfer of ordering the
CUDA API offers is a statement about `Below`:

* an operation on `p` is not a race when what it touches is `Below` `know p`
  (`op_some`), and afterwards what it touched is `Below` `p`'s clock;
* synchronising `p` makes what is `Below` `p`'s clock `Below` the host's;
* recording an event on `p` makes what is `Below` `know p` `Below` the event;
* waiting on an event makes what is `Below` it `Below` the waiter;
* clocks only grow, so `Below` a party's clock stays true (`Grows`).

Ownership of a buffer by a stream, a buffer being quiet, a stream being
drained --- each is a `Below` fact, and a frontend picks the ones it tracks.

## The specifications

For each call, `*_safe`: under its precondition the contract answers --- so the
call is not misuse. And, where a frontend needs to continue from it, what the
answering world is.
-/

namespace AlgorithmLib.HProg.DevSpec

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.Device (Clock.get_join Clock.get_bump Race.clock_setClock readsFold writesFold
  readsFold_other readsFold_mem writesFold_other writesFold_lastW writesFold_reads)
open AlgorithmLib.HProg.Static (KeepsSize MemSame store_frame store_some decodeAddr_add decodeAddr_base)

-- ---------------------------------------------------------------------------
-- Clocks
-- ---------------------------------------------------------------------------

def ClockLe (a b : Clock) : Prop := ∀ i, a.get i ≤ b.get i

theorem ClockLe.refl (a : Clock) : ClockLe a a := fun _ => Nat.le_refl _

theorem ClockLe.trans {a b c : Clock} (h1 : ClockLe a b) (h2 : ClockLe b c) : ClockLe a c :=
  fun i => Nat.le_trans (h1 i) (h2 i)

theorem ClockLe.join_left (a b : Clock) : ClockLe a (a.join b) := fun i => by
  rw [Clock.get_join]; exact Nat.le_max_left _ _

theorem ClockLe.join_right (a b : Clock) : ClockLe b (a.join b) := fun i => by
  rw [Clock.get_join]; exact Nat.le_max_right _ _

theorem ClockLe.bump (a : Clock) (p : Nat) : ClockLe a (a.bump p) := fun i => by
  rw [Clock.get_bump]; split
  · subst i; omega
  · omega

/-- Every access in `es` happened before a point whose clock is `c`. -/
def Below (es : List (Nat × Nat)) (c : Clock) : Prop := ∀ e ∈ es, e.2 ≤ c.get e.1

theorem Below.mono {es : List (Nat × Nat)} {c c' : Clock} (h : Below es c) (hc : ClockLe c c') :
    Below es c' := fun e he => Nat.le_trans (h e he) (hc e.1)

theorem Below.nil (c : Clock) : Below [] c := fun _ h => by cases h

-- ---------------------------------------------------------------------------
-- The tracker
-- ---------------------------------------------------------------------------

/-- The last write to `b`, as a list. -/
def wAcc (r : Race) (b : Nat) : List (Nat × Nat) := (r.lastW.getD b none).toList
/-- The reads of `b` since its last write. -/
def rAcc (r : Race) (b : Nat) : List (Nat × Nat) := r.reads.getD b []

/-- What party `p` knows when it issues an operation: its own clock and the
    host's. -/
def know (r : Race) (p : Nat) : Clock := (r.clock p).join (r.clock hostParty)

/-- The clock an operation on `p` is issued with. -/
def issued (r : Race) (p : Nat) : Clock := (know r p).bump p

/-- Every party's clock is at least what it was. -/
def Grows (r r' : Race) : Prop := ∀ q, ClockLe (r.clock q) (r'.clock q)

theorem Grows.refl (r : Race) : Grows r r := fun _ => ClockLe.refl _

theorem Grows.trans {a b c : Race} (h1 : Grows a b) (h2 : Grows b c) : Grows a c :=
  fun q => (h1 q).trans (h2 q)

/-- What an accepted operation on `p` reading `rs` and writing `ws` leaves. -/
def opped (r : Race) (p : Nat) (rs ws : List Nat) : Race :=
  let c := issued r p
  writesFold (p, c.get p) ws (readsFold (p, c.get p) rs (r.setClock p c))

theorem ordered_of_le {p : Nat} {c : Clock} {e : Nat × Nat} (h : e.2 ≤ c.get e.1) :
    ordered p c e = true := by
  simp [ordered, h]

/-- **An operation is not a race** when the last write of everything it touches,
    and the reads of everything it writes, are below what `p` knows. -/
theorem op_some {r : Race} {p : Nat} {rs ws : List Nat}
    (hw : ∀ b ∈ rs ++ ws, Below (wAcc r b) (know r p))
    (hr : ∀ b ∈ ws, Below (rAcc r b) (know r p)) :
    r.op p rs ws = some (opped r p rs ws) := by
  have hle : ClockLe (know r p) (issued r p) := ClockLe.bump _ _
  have hok : ((rs ++ ws).all (fun b =>
        (((r.setClock p (issued r p)).lastW.getD b none).map (ordered p (issued r p))).getD true)
      && ws.all (fun b => ((r.setClock p (issued r p)).reads.getD b []).all
        (ordered p (issued r p)))) = true := by
    simp only [Race.setClock, Bool.and_eq_true, List.all_eq_true]
    refine ⟨fun b hb => ?_, fun b hb x hx => ?_⟩
    · have := hw b hb
      simp only [wAcc] at this
      cases e : r.lastW.getD b none with
      | none => rfl
      | some x =>
          simp only [Option.map_some, Option.getD_some]
          exact ordered_of_le (Nat.le_trans (this x (by simp [e])) (hle x.1))
    · exact ordered_of_le (Nat.le_trans (hr b hb x hx) (hle x.1))
  have e : r.op p rs ws = (r.setClock p (issued r p)).access p (issued r p) rs ws := rfl
  rw [e, Race.access]
  dsimp only
  simp only [hok, Bool.not_true, Bool.false_eq_true, if_false]
  rfl

theorem opped_clock (r : Race) (p rs ws q) :
    (opped r p rs ws).clock q = if q = p then issued r p else r.clock q := by
  simp only [opped, Race.clock]
  rw [writesFold_other, (readsFold_other _ _ _).1]
  exact Race.clock_setClock r p q _

theorem opped_grows (r : Race) (p rs ws) : Grows r (opped r p rs ws) := fun q => by
  rw [opped_clock]
  split
  · subst q; exact (ClockLe.join_left _ _).trans (ClockLe.bump _ _)
  · exact ClockLe.refl _

theorem opped_wAcc (r : Race) (p rs ws b) :
    wAcc (opped r p rs ws) b = if b ∈ ws then [(p, (issued r p).get p)] else wAcc r b := by
  simp only [wAcc, opped]
  rw [writesFold_lastW, (readsFold_other _ _ _).2]
  split <;> rfl

theorem opped_rAcc_written (r : Race) (p rs ws b) (h : b ∈ ws) :
    rAcc (opped r p rs ws) b = [] := by
  simp only [rAcc, opped]
  rw [writesFold_reads, if_pos h]

theorem opped_rAcc (r : Race) (p rs ws b) (h : b ∉ ws) (x) :
    x ∈ rAcc (opped r p rs ws) b ↔ x ∈ rAcc r b ∨ (b ∈ rs ∧ x = (p, (issued r p).get p)) := by
  simp only [rAcc, opped]
  rw [writesFold_reads, if_neg h, readsFold_mem]
  rfl

theorem issued_get (r : Race) (p : Nat) : (issued r p).get p = (know r p).get p + 1 := by
  simp [issued, Clock.get_bump]

theorem opped_clock_self (r : Race) (p rs ws) : (opped r p rs ws).clock p = issued r p := by
  rw [opped_clock, if_pos rfl]

-- Synchronisation, joins, issues.

theorem sync_clock (r : Race) (p q : Nat) :
    (r.sync p).clock q = if q = hostParty then (r.clock hostParty).join (r.clock p) else r.clock q := by
  simp only [Race.sync, Race.joinInto]
  rw [Race.clock_setClock]

theorem sync_grows (r : Race) (p : Nat) : Grows r (r.sync p) := fun q => by
  rw [sync_clock]; split
  · subst q; exact ClockLe.join_left _ _
  · exact ClockLe.refl _

/-- **Waiting for `p`**: what was below `p`'s clock is below the host's. -/
theorem sync_host (r : Race) (p : Nat) : ClockLe (r.clock p) ((r.sync p).clock hostParty) := by
  rw [sync_clock, if_pos rfl]; exact ClockLe.join_right _ _

theorem sync_lastW (r : Race) (p : Nat) : (r.sync p).lastW = r.lastW := rfl
theorem sync_reads (r : Race) (p : Nat) : (r.sync p).reads = r.reads := rfl

theorem joinInto_clock (r : Race) (p : Nat) (c : Clock) (q : Nat) :
    (r.joinInto p c).clock q = if q = p then (r.clock p).join c else r.clock q := by
  simp only [Race.joinInto]; rw [Race.clock_setClock]

theorem joinInto_grows (r : Race) (p : Nat) (c : Clock) : Grows r (r.joinInto p c) := fun q => by
  rw [joinInto_clock]; split
  · subst q; exact ClockLe.join_left _ _
  · exact ClockLe.refl _

theorem joinInto_learns (r : Race) (p : Nat) (c : Clock) : ClockLe c ((r.joinInto p c).clock p) := by
  rw [joinInto_clock, if_pos rfl]; exact ClockLe.join_right _ _

theorem joinInto_lastW (r : Race) (p : Nat) (c : Clock) : (r.joinInto p c).lastW = r.lastW := rfl
theorem joinInto_reads (r : Race) (p : Nat) (c : Clock) : (r.joinInto p c).reads = r.reads := rfl

theorem issue_eq (r : Race) (p : Nat) : r.issue p = (r.setClock p (issued r p), issued r p) := rfl

theorem setClock_issued_grows (r : Race) (p : Nat) : Grows r (r.setClock p (issued r p)) := fun q => by
  rw [Race.clock_setClock]; split
  · subst q; exact (ClockLe.join_left _ _).trans (ClockLe.bump _ _)
  · exact ClockLe.refl _

/-- What `p` knows is below the clock it issues with, which becomes its own. -/
theorem know_le_issued (r : Race) (p : Nat) : ClockLe (know r p) (issued r p) := ClockLe.bump _ _

theorem host_le_know (r : Race) (p : Nat) : ClockLe (r.clock hostParty) (know r p) :=
  ClockLe.join_right _ _

theorem own_le_know (r : Race) (p : Nat) : ClockLe (r.clock p) (know r p) := ClockLe.join_left _ _

theorem Grows.know_le {r r' : Race} (h : Grows r r') (p : Nat) : ClockLe (know r p) (know r' p) := fun i => by
  show (know r p).get i ≤ (know r' p).get i
  simp only [know, Clock.get_join]
  have := h p i; have := h hostParty i; omega

-- ---------------------------------------------------------------------------
-- What the calls need
-- ---------------------------------------------------------------------------

/-- The context argument is one the contracts answer: null, which they answer
    with `-1`, or the live context. -/
def CtxOk (w : World) (ctx : UInt64) : Prop := ctx = 0 ∨ (ctx = cudaCtx ∧ w.dev.live = true)

/-- An access by `p` to `b` is not a race: the last write is below what `p`
    knows, and for a write so are the reads since. -/
def Ready (r : Race) (p b : Nat) (write : Bool) : Prop :=
  Below (wAcc r b) (know r p) ∧ (write = true → Below (rAcc r b) (know r p))

/-- A launch's bindings can be read, and each that names a live buffer is
    ready for `p` to write. -/
def BindsReady (w : World) (p : Nat) (nBufs bindPtr : UInt64) : Prop :=
  ∃ ids, readIds w.mem bindPtr (asI32 nBufs).toNat = some ids ∧
    ∀ id ∈ ids, (w.dev.get? id).isSome = true → Ready w.dev.race p id.toNat true

theorem cudaCtx_ne_zero : (cudaCtx == 0) = false := by decide

theorem cudaCtxOk_of {w : World} {ctx : UInt64} (h : CtxOk w ctx) :
    cudaCtxOk w ctx = some (ctx != 0) := by
  unfold cudaCtxOk
  rcases h with rfl | ⟨rfl, hl⟩
  · rfl
  · simp [hl, show cudaCtx ≠ 0 by decide]

theorem devOnly_isSome (w : World) (k : Option (Option V × Dev)) :
    (devOnly w k).isSome = k.isSome := by
  unfold devOnly; cases k <;> rfl

theorem op_isSome {r : Race} {p b : Nat} {write : Bool} (h : Ready r p b write) :
    (if write then r.op p [] [b] else r.op p [b] []).isSome = true := by
  cases write with
  | true =>
      simp only [if_true]
      rw [op_some (fun x hx => by simp at hx; subst hx; exact h.1)
        (fun x hx => by simp at hx; subst hx; exact h.2 rfl)]
      rfl
  | false =>
      simp only [Bool.false_eq_true, if_false]
      rw [op_some (fun x hx => by simp at hx; subst hx; exact h.1) (fun x hx => by simp at hx)]
      rfl

theorem syncWrite_isSome {d : Dev} {id : Nat} (b : ByteArray) (h : Ready d.race defaultParty id true) :
    (d.syncWrite id b).isSome = true := by
  unfold Dev.syncWrite
  have := op_isSome h
  simp only [if_true] at this
  cases e : d.race.op defaultParty [] [id] with
  | none => rw [e] at this; cases this
  | some r => simp [bind, Option.bind]

theorem syncRead_isSome {d : Dev} {id : Nat} (h : Ready d.race defaultParty id false) :
    (d.syncRead id).isSome = true := by
  unfold Dev.syncRead
  have := op_isSome h
  simp only [Bool.false_eq_true, if_false] at this
  cases e : d.race.op defaultParty [id] [] with
  | none => rw [e] at this; cases this
  | some r => simp [bind, Option.bind]

theorem keeps_ok {k : Launch → List ByteArray → List ByteArray} (hk : KeepsSize k) (l : Launch)
    (ins : List ByteArray) :
    ((k l ins).length != ins.length || ((k l ins).zip ins).any (fun (o, i) => o.size != i.size)) = false := by
  have h := hk l ins
  have hl : (k l ins).length = ins.length := by simpa using congrArg List.length h
  simp only [hl, bne_self_eq_false, Bool.false_or, List.any_eq_false]
  intro x hx
  obtain ⟨o, i⟩ := x
  have hsz : ∀ (os is : List ByteArray), os.map ByteArray.size = is.map ByteArray.size →
      ∀ o i, (o, i) ∈ os.zip is → o.size = i.size := by
    intro os
    induction os with
    | nil => intro is _ o i hm; simp at hm
    | cons a as ih =>
      intro is his o i hm
      cases is with
      | nil => simp at hm
      | cons b bs =>
        simp only [List.map_cons, List.cons.injEq] at his
        simp only [List.zip_cons_cons, List.mem_cons, Prod.mk.injEq] at hm
        rcases hm with ⟨rfl, rfl⟩ | hm
        · exact his.1
        · exact ih bs his.2 o i hm
  simp only [bne_iff_ne, ne_eq, Decidable.not_not]
  exact hsz _ _ h o i hx

theorem mapM_get_live {d : Dev} : ∀ (ids : List Int), (ids.any (fun id => (d.get? id).isNone)) = false →
    ∃ bs, (ids.map Int.toNat).mapM (fun i => d.get? (Int.ofNat i)) = some bs
  | [], _ => ⟨[], rfl⟩
  | id :: ids, h => by
      simp only [List.any_cons, Bool.or_eq_false_iff] at h
      obtain ⟨bs, hbs⟩ := mapM_get_live ids h.2
      have hid : 0 ≤ id := by
        rcases Int.lt_or_le id 0 with hn | hn
        · have : d.get? id = none := by simp [Dev.get?, hn]
          simp [this] at h
        · exact hn
      obtain ⟨b, hb⟩ : ∃ b, d.get? id = some b := Option.isSome_iff_exists.mp (by simpa using h.1)
      refine ⟨b :: bs, ?_⟩
      simp only [List.map_cons, List.mapM_cons, bind, Option.bind]
      rw [Int.ofNat_eq_natCast, Int.toNat_of_nonneg hid, hb, hbs]
      rfl

theorem runLaunch_isSome {d : Dev} {k : Launch → List ByteArray → List ByteArray} (hk : KeepsSize k)
    (l : Launch) {ids : List Int} (h : (ids.any (fun id => (d.get? id).isNone)) = false) :
    (d.runLaunch k l (ids.map Int.toNat)).isSome = true := by
  obtain ⟨bs, hbs⟩ := mapM_get_live ids h
  unfold Dev.runLaunch
  simp only [bind, Option.bind, hbs, keeps_ok hk, Bool.false_eq_true, if_false, Option.isSome_some]

theorem cudaLaunchOn_isSome {w : World} {p : Nat} {kernel entry : String} {nBufs bindPtr : UInt64}
    {dims : List UInt64} (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hb : BindsReady w p nBufs bindPtr) :
    (cudaLaunchOn w p kernel entry nBufs bindPtr dims).isSome = true := by
  obtain ⟨ids, hids, hready⟩ := hb
  unfold cudaLaunchOn
  simp only [hids, bind, Option.bind]
  split
  · rfl
  · rename_i hany
    have hany' : (ids.any (fun id => (w.dev.get? id).isNone)) = false := by simpa using hany
    have hlive : ∀ id ∈ ids, (w.dev.get? id).isSome = true := by
      intro id hid
      have := List.any_eq_false.mp hany' id hid
      simpa [Option.isSome_iff_ne_none] using this
    have hop : w.dev.race.op p [] (ids.map Int.toNat) =
        some (opped w.dev.race p [] (ids.map Int.toNat)) := by
      apply op_some
      · intro b hb
        simp only [List.nil_append, List.mem_map] at hb
        obtain ⟨id, hid, rfl⟩ := hb
        exact (hready id hid (hlive id hid)).1
      · intro b hb
        simp only [List.mem_map] at hb
        obtain ⟨id, hid, rfl⟩ := hb
        exact (hready id hid (hlive id hid)).2 rfl
    unfold Dev.devOp Dev.devOp.run
    simp only [hcap, hop, bind, Option.bind]
    have hr := runLaunch_isSome (d := { w.dev with race := opped w.dev.race p [] (ids.map Int.toNat) })
      hk ⟨kernel, entry, ids.map Int.toNat, dims, []⟩ (ids := ids) hany'
    cases e : Dev.runLaunch { w.dev with race := opped w.dev.race p [] (ids.map Int.toNat) } w.kernel
        ⟨kernel, entry, ids.map Int.toNat, dims, []⟩ (ids.map Int.toNat) with
    | none => rw [e] at hr; cases hr
    | some d =>
        simp only [hcap] at e
        rw [e]
        rfl

/-- `n` bytes from `a` can be read. -/
def Readable (m : Mem) (a : UInt64) (n : Nat) : Prop := (readBytes m a n).isSome = true

/-- Any `n` bytes can be copied to `a`. -/
def Writable (m : Mem) (a : UInt64) (n : Nat) : Prop :=
  ∀ src : ByteArray, src.size = n → (copyIn m a src).isSome = true

-- ---------------------------------------------------------------------------
-- The calls, not refused
-- ---------------------------------------------------------------------------

/-- Whether a store answers does not depend on the value stored. -/
theorem store_isSome_val {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : (m.store a n v).isSome = true)
    (v' : UInt64) : (m.store a n v').isSome = true := by
  unfold Mem.store at h ⊢
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h ⊢
    split at h
    · cases h
    · rw [if_neg ‹_›]; rfl

theorem init_safe {w : World} {slot : UInt64} (h : (w.mem.store slot 8 cudaCtx).isSome = true) :
    (ffiCudaInit [slot] w).isSome = true := by
  obtain ⟨m1, e1⟩ := Option.isSome_iff_exists.mp h
  obtain ⟨m0, e0⟩ := Option.isSome_iff_exists.mp (store_isSome_val h 0)
  unfold ffiCudaInit
  dsimp only
  split <;> simp [bind, Option.bind, e1, e0]

theorem cleanup_safe {w : World} {slot : UInt64} (h : (w.mem.store slot 8 0).isSome = true) :
    (ffiCudaCleanup [slot] w).isSome = true := by
  unfold ffiCudaCleanup
  cases e : w.mem.store slot 8 0 with
  | none => rw [e] at h; cases h
  | some m => simp only [bind, Option.bind, e]; rfl

theorem createBuffer_safe {w : World} {ctx size : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaCreateBuffer [ctx, size] w).isSome = true := by
  unfold ffiCudaCreateBuffer
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split <;> rfl

theorem sync_safe {w : World} {ctx : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaSync [ctx] w).isSome = true := by
  unfold ffiCudaSync
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split <;> rfl

theorem upload_safe {w : World} {ctx buf src size : UInt64} (hc : CtxOk w ctx)
    (h : ∀ b, w.dev.get? (asI32 buf) = some b → b.size = size.toNat →
      Readable w.mem src size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat true) :
    (ffiCudaUpload [ctx, buf, src, size] w).isSome = true := by
  unfold ffiCudaUpload
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · rename_i b hb
        split
        · rfl
        rename_i hs
        obtain ⟨hr, hready⟩ := h b hb (by simpa using hs)
        obtain ⟨bytes, hbytes⟩ := Option.isSome_iff_exists.mp hr
        rw [hbytes]
        obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp (syncWrite_isSome (d := w.dev) bytes hready)
        simp [hd, devOk]

theorem uploadOffset_safe {w : World} {ctx buf off src size : UInt64} (hc : CtxOk w ctx)
    (h : ∀ b, w.dev.get? (asI32 buf) = some b →
      Readable w.mem src size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat true) :
    (ffiCudaUploadOffset [ctx, buf, off, src, size] w).isSome = true := by
  unfold ffiCudaUploadOffset
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · rename_i b hb
        obtain ⟨hr, hready⟩ := h b hb
        split
        · rfl
        · obtain ⟨bytes, hbytes⟩ := Option.isSome_iff_exists.mp hr
          rw [hbytes]
          obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp
            (syncWrite_isSome (d := w.dev) (overwrite b off.toNat bytes) hready)
          simp [hd, devOk]

theorem download_safe {w : World} {ctx buf dst size : UInt64} (hc : CtxOk w ctx)
    (h : ∀ b, w.dev.get? (asI32 buf) = some b → b.size = size.toNat →
      Writable w.mem dst size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat false) :
    (ffiCudaDownload [ctx, buf, dst, size] w).isSome = true := by
  unfold ffiCudaDownload
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · rename_i b hb
        split
        · rfl
        rename_i hs
        have hs : b.size = size.toNat := by simpa using hs
        obtain ⟨hw, hready⟩ := h b hb hs
        obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp (syncRead_isSome (d := w.dev) hready)
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hw b hs)
        simp [hd, hm]

theorem downloadOffset_safe {w : World} {ctx buf off dst size : UInt64} (hc : CtxOk w ctx)
    (h : ∀ b, w.dev.get? (asI32 buf) = some b →
      Writable w.mem dst size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat false) :
    (ffiCudaDownloadOffset [ctx, buf, off, dst, size] w).isSome = true := by
  unfold ffiCudaDownloadOffset
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · rename_i b hb
        obtain ⟨hw, hready⟩ := h b hb
        split
        · rfl
        · rename_i hin
          obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp (syncRead_isSome (d := w.dev) hready)
          have hsz : (b.extract off.toNat (off.toNat + size.toNat)).size = size.toNat := by
            simp only [ByteArray.size_extract]; omega
          obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hw _ hsz)
          simp [hd, hm]

theorem freeBuffer_safe {w : World} {ctx buf : UInt64} (hc : CtxOk w ctx)
    (h : (w.dev.get? (asI32 buf)).isSome = true → Ready w.dev.race defaultParty (asI32 buf).toNat true) :
    (ffiCudaFreeBuffer [ctx, buf] w).isSome = true := by
  unfold ffiCudaFreeBuffer
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · rename_i b hb
        have := op_isSome (h (by simp [hb]))
        simp only [if_true] at this
        obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp this
        simp [hr, devOk]

theorem streamCreate_safe {w : World} {ctx : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaStreamCreate [ctx] w).isSome = true := by
  unfold ffiCudaStreamCreate
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split <;> rfl

theorem capturing_none {d : Dev} (h : d.capture = none) (p : Nat) : d.capturing p = false := by
  simp [Dev.capturing, h]

theorem streamSync_safe {w : World} {ctx sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) :
    (ffiCudaStreamSync [ctx, sid] w).isSome = true := by
  unfold ffiCudaStreamSync
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · rfl
    · simp [capturing_none hcap, devOk]

theorem streamDestroy_safe {w : World} {ctx sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) :
    (ffiCudaStreamDestroy [ctx, sid] w).isSome = true := by
  unfold ffiCudaStreamDestroy
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · simp [capturing_none hcap, devOk]

theorem eventCreate_safe {w : World} {ctx : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaEventCreate [ctx] w).isSome = true := by
  unfold ffiCudaEventCreate
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split <;> rfl

theorem eventDestroy_safe {w : World} {ctx eid : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaEventDestroy [ctx, eid] w).isSome = true := by
  unfold ffiCudaEventDestroy
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split <;> rfl

theorem eventRecord_safe {w : World} {ctx eid sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) :
    (ffiCudaEventRecord [ctx, eid, sid] w).isSome = true := by
  unfold ffiCudaEventRecord
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · simp [hcap, devOk]
    · rfl

/-- Waiting on an event: no capture is in progress, and the event does not mark
    a point inside one. -/
theorem streamWaitEvent_safe {w : World} {ctx sid eid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) (hev : ∀ c, w.dev.event? (asI32 eid) ≠ some (some (c, true))) :
    (ffiCudaStreamWaitEvent [ctx, sid, eid] w).isSome = true := by
  unfold ffiCudaStreamWaitEvent
  rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · rename_i p ev hp he
      split
      · rfl
      · rename_i clk c hc'
        simp [hcap] at hc'
      · simp [capturing_none hcap, devOk]
      · rename_i clk hn
        exact absurd he (hev clk)
    · rfl

theorem launch_safe {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz : UInt64}
    (hc : CtxOk w ctx) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hs : (readCStrAt w.mem kptr).isSome = true) (hb : BindsReady w defaultParty nBufs bindPtr) :
    (ffiCudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w).isSome = true := by
  unfold ffiCudaLaunch
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.mp hs
      simp only [hk']
      exact cudaLaunchOn_isSome hk hcap hb

theorem launchNamed_safe {w : World} {ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz : UInt64}
    (hc : CtxOk w ctx) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hs : (readCStrAt w.mem kptr).isSome = true) (hn : (readCStrAt w.mem namePtr).isSome = true)
    (hb : BindsReady w defaultParty nBufs bindPtr) :
    (ffiCudaLaunchNamed [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w).isSome = true := by
  unfold ffiCudaLaunchNamed
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.mp hs
      obtain ⟨e, he⟩ := Option.isSome_iff_exists.mp hn
      simp only [hk', he]
      exact cudaLaunchOn_isSome hk hcap hb

theorem launchOnStream_safe {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64}
    (hc : CtxOk w ctx) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hs : (readCStrAt w.mem kptr).isSome = true)
    (hb : ∀ p, w.dev.party? (asI32 sid) = some p → BindsReady w p nBufs bindPtr) :
    (ffiCudaLaunchOnStream [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w).isSome = true := by
  unfold ffiCudaLaunchOnStream
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.mp hs
      simp only [hk']
      split
      · rfl
      · rename_i p hp
        exact cudaLaunchOn_isSome hk hcap (hb p hp)

theorem launchNamedOnStream_safe {w : World}
    {ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64}
    (hc : CtxOk w ctx) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hs : (readCStrAt w.mem kptr).isSome = true) (hn : (readCStrAt w.mem namePtr).isSome = true)
    (hb : ∀ p, w.dev.party? (asI32 sid) = some p → BindsReady w p nBufs bindPtr) :
    (ffiCudaLaunchNamedOnStream [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid]
      w).isSome = true := by
  unfold ffiCudaLaunchNamedOnStream
  rw [devOnly_isSome]
  dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.mp hs
      obtain ⟨e, he⟩ := Option.isSome_iff_exists.mp hn
      simp only [hk', he]
      split
      · rfl
      · rename_i p hp
        exact cudaLaunchOn_isSome hk hcap (hb p hp)

-- ---------------------------------------------------------------------------
-- Host memory
--
-- What a load, a string read, a binding read or a store at an arena address
-- sees is the arena's bytes and whether memory is frozen, nothing else. So a
-- fact about one can be computed on `arenaMem A`, a closed memory, and holds
-- of every memory whose arena is `A`.
-- ---------------------------------------------------------------------------

/-- A memory holding only an arena. -/
def arenaMem (A : ByteArray) : Mem := { arena := A, data := ByteArray.empty, out := ByteArray.empty }

theorem arena_ne_pinned : (Region.arena != Region.pinned) = true := rfl

theorem load_arena {m : Mem} {A : ByteArray} (hA : m.arena = A) (hf : m.frozen = false)
    {a : UInt64} {off : Nat} (hd : decodeAddr a = some (.arena, off)) (n : Nat) :
    m.load a n = (arenaMem A).load a n := by
  simp only [Mem.load, hd, bind, Option.bind, Mem.region, hA, Mem.readable, hf, arenaMem,
    arena_ne_pinned, Bool.true_or, Bool.not_false, Bool.true_and]

theorem readCStr_arena {m : Mem} {A : ByteArray} (hA : m.arena = A) {a : UInt64} {off : Nat}
    (hd : decodeAddr a = some (.arena, off)) : readCStr m a = readCStr (arenaMem A) a := by
  simp only [readCStr, hd, bind, Option.bind, Mem.region, hA, arenaMem]

theorem readIds_arena {m : Mem} {A : ByteArray} (hA : m.arena = A) (hf : m.frozen = false)
    {p : UInt64} {off n : Nat} (hd : decodeAddr p = some (.arena, off))
    (hn : off + 4 * n < regionSpan.toNat) : readIds m p n = readIds (arenaMem A) p n := by
  unfold readIds
  rw [Static.mapM_congr_mem _ _ _ fun i hi => load_arena hA hf
    (decodeAddr_add hd (by have := List.mem_range.mp hi; omega)) 4]

theorem copyOut_arena {m : Mem} {A : ByteArray} (hA : m.arena = A) (hf : m.frozen = false)
    {a : UInt64} {off n : Nat} (hd : decodeAddr a = some (.arena, off))
    (hn : off + n < regionSpan.toNat) : copyOut m a n = copyOut (arenaMem A) a n := by
  unfold copyOut
  suffices ∀ (l : List Nat), (∀ i ∈ l, i < n) → ∀ acc,
      l.foldlM (fun (acc : ByteArray) i => do
        let b ← m.load (a + UInt64.ofNat i) 1
        pure (acc.push b.toUInt8)) acc =
      l.foldlM (fun (acc : ByteArray) i => do
        let b ← (arenaMem A).load (a + UInt64.ofNat i) 1
        pure (acc.push b.toUInt8)) acc from
    this _ (fun i hi => List.mem_range.mp hi) _
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons i l ih =>
      intro hl acc
      simp only [List.foldlM_cons]
      rw [load_arena hA hf (decodeAddr_add hd (by have := hl i (List.mem_cons_self ..); omega)) 1]
      cases (arenaMem A).load (a + UInt64.ofNat i) 1 with
      | none => rfl
      | some b => exact ih (fun j hj => hl j (List.mem_cons_of_mem _ hj)) _

theorem store_arena {m : Mem} {A : ByteArray} (hA : m.arena = A) (hf : m.frozen = false)
    {a : UInt64} {off : Nat} (hd : decodeAddr a = some (.arena, off)) (n : Nat) (v : UInt64) :
    m.store a n v = ((arenaMem A).store a n v).map (fun m' => { m with arena := m'.arena }) := by
  simp only [Mem.store, hd, bind, Option.bind, Mem.region, hA, Mem.reachable, hf, arenaMem,
    arena_ne_pinned, Bool.true_or, Bool.not_false, Bool.true_and]
  split
  · rfl
  · cases m; subst hA; simp_all [Mem.setRegion]

theorem store_isSome_of {m : Mem} {a : UInt64} {r : Region} {off n : Nat} (v : UInt64)
    (hd : decodeAddr a = some (r, off)) (hr : r ≠ .pinned) (hf : m.frozen = false)
    (hn : off + n ≤ (m.region r).size) : (m.store a n v).isSome = true := by
  simp only [Mem.store, hd, bind, Option.bind, Mem.reachable, hf]
  have : (r != Region.pinned) = true := by cases r <;> first | rfl | exact absurd rfl hr
  simp only [this, Bool.not_false, Bool.true_and, Bool.true_or, Bool.not_true, Bool.or_false]
  rw [if_neg (by simp; omega)]
  rfl

/-- **A copy into ordinary memory that fits is not refused.** -/
theorem copyIn_isSome_of {m : Mem} {a : UInt64} {r : Region} {off : Nat} {src : ByteArray}
    (hd : decodeAddr a = some (r, off)) (hr : r ≠ .pinned) (hf : m.frozen = false)
    (hn : off + src.size ≤ (m.region r).size) (hs : off + src.size ≤ regionSpan.toNat) :
    (copyIn m a src).isSome = true := by
  unfold copyIn
  suffices ∀ k, k ≤ src.size → ∃ m', (List.range k).foldlM
      (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m = some m' ∧
      MemSame m m' by
    obtain ⟨m', h, _⟩ := this _ (Nat.le_refl _)
    rw [h]; rfl
  intro k
  induction k with
  | zero => intro _; exact ⟨m, rfl, Static.MemSame.refl m⟩
  | succ k ih =>
      intro hk
      obtain ⟨m1, h1, hs1⟩ := ih (by omega)
      rw [List.range_succ, List.foldlM_append, h1]
      simp only [Option.bind_eq_bind, Option.bind_some, List.foldlM_cons, List.foldlM_nil]
      have hd' := decodeAddr_add hd (i := k) (by omega)
      have hsome := store_isSome_of (m := m1) (n := 1) (src.get! k).toUInt64 hd' hr (hs1.frozen ▸ hf)
        (by rw [← hs1.size r]; omega)
      obtain ⟨m2, h2⟩ := Option.isSome_iff_exists.mp hsome
      rw [h2]
      obtain ⟨_, _, _, hf2⟩ := store_frame h2
      exact ⟨m2, rfl, hs1.trans hf2.same⟩

/-- Whether an address falls in the arena. -/
def inArena (a : UInt64) : Bool :=
  match decodeAddr a with
  | some (.arena, _) => true
  | _ => false

theorem inArena_decode {a : UInt64} (h : inArena a = true) : ∃ off, decodeAddr a = some (.arena, off) := by
  unfold inArena at h
  split at h
  · rename_i off hd; exact ⟨off, hd⟩
  · cases h

/-- What a load, a store, a string read, a binding read and a byte read at an
    arena address see: named so a rewrite to them cannot loop. -/
def arenaLoad (A : ByteArray) (a : UInt64) (n : Nat) : Option UInt64 := (arenaMem A).load a n
def arenaStore (A : ByteArray) (a : UInt64) (n : Nat) (v : UInt64) : Option ByteArray :=
  ((arenaMem A).store a n v).map (·.arena)
def arenaCStr (A : ByteArray) (a : UInt64) : Option String := readCStr (arenaMem A) a
def arenaIds (A : ByteArray) (p : UInt64) (n : Nat) : Option (List Int) := readIds (arenaMem A) p n
def arenaBytes (A : ByteArray) (a : UInt64) (n : Nat) : Option ByteArray := copyOut (arenaMem A) a n

theorem load_inArena {m : Mem} {a : UInt64} (ha : inArena a = true) (hf : m.frozen = false) (n : Nat) :
    m.load a n = arenaLoad m.arena a n := by
  obtain ⟨_, hd⟩ := inArena_decode ha
  exact load_arena rfl hf hd n

theorem store_inArena {m : Mem} {a : UInt64} (ha : inArena a = true) (hf : m.frozen = false)
    (n : Nat) (v : UInt64) :
    m.store a n v = (arenaStore m.arena a n v).map (fun A => { m with arena := A }) := by
  obtain ⟨_, hd⟩ := inArena_decode ha
  rw [store_arena rfl hf hd, arenaStore, Option.map_map]
  rfl

theorem readCStrAt_inArena {m : Mem} {a : UInt64} (ha : inArena a = true) :
    readCStrAt m a = arenaCStr m.arena a := by
  obtain ⟨_, hd⟩ := inArena_decode ha
  exact readCStr_arena rfl hd

/-- Reads within an arena's worth of `p`. -/
theorem readIds_inArena {m : Mem} {p : UInt64} {n : Nat} {off : Nat}
    (hd : decodeAddr p = some (.arena, off)) (hn : off + 4 * n < regionSpan.toNat)
    (hf : m.frozen = false) : readIds m p n = arenaIds m.arena p n :=
  readIds_arena rfl hf hd hn

theorem readBytes_inArena {m : Mem} {a : UInt64} {off n : Nat}
    (hd : decodeAddr a = some (.arena, off)) (hn : off + n < regionSpan.toNat)
    (hf : m.frozen = false) : readBytes m a n = arenaBytes m.arena a n :=
  copyOut_arena rfl hf hd hn

-- ---------------------------------------------------------------------------
-- Dispatch
-- ---------------------------------------------------------------------------

/-- A supported call, made outside a worker's lifetime, is its contract on the
    argument bits. -/
theorem callOf_supported {f : IR.Ffi} (hf : Static.supported f = true) (lc : Locals) (vs : List V)
    {w : World} (hz : w.mem.frozen = false) :
    callOf lc (.ffi f) vs w = (vs.mapM asBits).bind (fun bits => callBits f bits w) := by
  simp only [callOf, hz, Static.supported_ne_spawn f hf, Bool.false_and, Bool.false_eq_true,
    if_false, callImport, Static.ofCname_supported f hf, bind, Option.bind, callFfi]

-- ---------------------------------------------------------------------------
-- What the calls leave
-- ---------------------------------------------------------------------------

theorem retire_nil {m : Mem} (h : m.busy = []) (c : Clock) : m.retire c = m := by
  cases m; simp_all [Mem.retire]

theorem retire_same (m : Mem) (c : Clock) (h : m.busy = []) : MemSame m (m.retire c) := by
  rw [retire_nil h]; exact Static.MemSame.refl m

/-- A copy into memory changes bytes and nothing else. -/
theorem copyIn_same {m m' : Mem} {a : UInt64} {src : ByteArray} (h : copyIn m a src = some m') :
    MemSame m m' := by
  unfold copyIn at h
  suffices ∀ (l : List Nat) (m0 m1 : Mem), MemSame m m0 → l.foldlM
      (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m0 = some m1 →
      MemSame m m1 from this _ m m' (Static.MemSame.refl m) h
  intro l
  induction l with
  | nil => intro m0 m1 hs h; cases h; exact hs
  | cons i l ih =>
      intro m0 m1 hs h
      simp only [List.foldlM_cons] at h
      cases h1 : m0.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64 with
      | none => rw [h1] at h; cases h
      | some m2 =>
          rw [h1] at h
          obtain ⟨_, _, _, hf⟩ := store_frame h1
          exact ih m2 m1 (hs.trans hf.same) h

/-- The device after an operation by `p` that may have written `ids`: the same
    objects, buffers the same lengths, and the race record unchanged or that
    operation's. -/
structure LaunchPost (d d' : Dev) (p : Nat) (ids : List Nat) : Prop where
  live : d'.live = d.live
  streams : d'.streams = d.streams
  capture : d'.capture = d.capture
  events : d'.events = d.events
  sizes : ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size
  race : d'.race = d.race ∨ d'.race = opped d.race p [] ids

theorem LaunchPost.refl (d : Dev) (p : Nat) (ids : List Nat) : LaunchPost d d p ids :=
  ⟨rfl, rfl, rfl, rfl, fun _ => rfl, Or.inl rfl⟩

theorem get?_put_size (d : Dev) (id : Nat) (b : ByteArray) (i : Int) (hb : ∀ x, d.get? id = some x → x.size = b.size)
    (hlive : (d.get? id).isSome = true) :
    ((d.put id (some b)).get? i).map ByteArray.size = (d.get? i).map ByteArray.size := by
  have hlt : id < d.bufs.size := by
    rcases e : d.bufs[id]? with _ | x
    · simp [Dev.get?, e] at hlive
    · exact (Array.getElem?_eq_some_iff.mp e).1
  simp only [Dev.get?, Dev.put]
  split
  · rfl
  · rename_i hi
    by_cases h : i.toNat = id
    · rw [h]
      simp only [Array.getElem?_setIfInBounds, hlt, if_true, Option.join_some, Option.map_some]
      obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp hlive
      have hx' : (d.bufs[id]?).join = some x := by
        have := hx; simp only [Dev.get?] at this; simpa using this
      rw [hx', Option.map_some, hb x hx]
    · simp [Ne.symm h]

theorem mapM_get_sizes {d d' : Dev}
    (hs : ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size) :
    ∀ (js : List Nat) (xs : List ByteArray), js.mapM (fun i => d.get? (Int.ofNat i)) = some xs →
      ∃ ys, js.mapM (fun i => d'.get? (Int.ofNat i)) = some ys ∧
        ys.map ByteArray.size = xs.map ByteArray.size
  | [], xs, h => by simp at h; subst h; exact ⟨[], rfl, rfl⟩
  | j :: js, xs, h => by
      simp only [List.mapM_cons] at h
      obtain ⟨y, hy, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨ys, hys, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      obtain ⟨zs, hzs, hzs'⟩ := mapM_get_sizes hs js ys hys
      have hj := hs (Int.ofNat j)
      rw [hy] at hj
      obtain ⟨z, hz, hzsz⟩ : ∃ z, d'.get? (Int.ofNat j) = some z ∧ z.size = y.size := by
        cases e : d'.get? (Int.ofNat j) with
        | none => rw [e] at hj; cases hj
        | some z => rw [e] at hj; exact ⟨z, rfl, by simpa using hj⟩
      refine ⟨z :: zs, ?_, ?_⟩
      · simp only [List.mapM_cons, hz, hzs, bind, Option.bind, pure]
      · simp [hzsz, hzs']

theorem runLaunch_sizes {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} (hk : KeepsSize k)
    {l : Launch} {ids : List Nat} (h : d.runLaunch k l ids = some d') :
    d'.live = d.live ∧ d'.streams = d.streams ∧ d'.capture = d.capture ∧ d'.events = d.events ∧
      d'.race = d.race ∧ ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, hins, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [keeps_ok hk, Bool.false_eq_true, if_false, Option.some.injEq] at h
  subst h
  have hsz := hk l ins
  suffices ∀ (ids : List Nat) (outs ins : List ByteArray) (d : Dev),
      ids.mapM (fun i => d.get? (Int.ofNat i)) = some ins → outs.map ByteArray.size = ins.map ByteArray.size →
      let d' := (ids.zip outs).foldl (fun d (id, o) => d.put id (some o)) d
      d'.live = d.live ∧ d'.streams = d.streams ∧ d'.capture = d.capture ∧ d'.events = d.events ∧
        d'.race = d.race ∧ ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size from
    this ids _ ins d hins hsz
  intro ids
  induction ids with
  | nil => intro outs ins d _ _; exact ⟨rfl, rfl, rfl, rfl, rfl, fun _ => rfl⟩
  | cons id ids ih =>
      intro outs ins d hm hs
      cases outs with
      | nil => exact ⟨rfl, rfl, rfl, rfl, rfl, fun _ => rfl⟩
      | cons o os =>
          simp only [List.mapM_cons] at hm
          obtain ⟨x, hx, hm⟩ := Option.bind_eq_some_iff.mp hm
          obtain ⟨xs, hxs, hm⟩ := Option.bind_eq_some_iff.mp hm
          simp only [Option.pure_def, Option.some.injEq] at hm
          subst hm
          simp only [List.map_cons, List.cons.injEq] at hs
          simp only [List.zip_cons_cons, List.foldl_cons]
          have hput : ∀ i, ((d.put id (some o)).get? i).map ByteArray.size = (d.get? i).map ByteArray.size :=
            fun i => get?_put_size d id o i (fun y hy => by
              have : d.get? id = d.get? (Int.ofNat id) := rfl
              rw [this, hx] at hy; cases hy; exact hs.1.symm) (by
              have : d.get? id = d.get? (Int.ofNat id) := rfl
              rw [this, hx]; rfl)
          obtain ⟨ys, hys, hyssz⟩ := mapM_get_sizes hput ids xs hxs
          obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih os ys (d.put id (some o)) hys (by rw [hs.2, hyssz])
          exact ⟨h1, h2, h3, h4, h5, fun i => (h6 i).trans (hput i)⟩

theorem cudaLaunchOn_post {w : World} {p : Nat} {kernel entry : String} {nBufs bindPtr : UInt64}
    {dims : List UInt64} {r : Option V} {d' : Dev} (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (h : cudaLaunchOn w p kernel entry nBufs bindPtr dims = some (r, d')) :
    d' = w.dev ∨ ∃ ids, readIds w.mem bindPtr (asI32 nBufs).toNat = some ids ∧
      LaunchPost w.dev d' p (ids.map Int.toNat) := by
  unfold cudaLaunchOn at h
  obtain ⟨ids, hids, h⟩ := Option.bind_eq_some_iff.mp h
  split at h
  · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at h; exact h.2.symm
  · right
    refine ⟨ids, hids, ?_⟩
    obtain ⟨d1, hd1, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [devOk, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    unfold Dev.devOp Dev.devOp.run at hd1
    simp only [hcap] at hd1
    obtain ⟨r1, hr1, hd1⟩ := Option.bind_eq_some_iff.mp hd1
    have hr1' : r1 = opped w.dev.race p [] (ids.map Int.toNat) := by
      unfold Race.op at hr1
      obtain ⟨_, he⟩ := Device.access_eq hr1
      rw [he]; rfl
    subst hr1'
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := runLaunch_sizes hk hd1
    exact ⟨h1, h2, h3.trans (by simp [hcap]), h4, h6, Or.inr h5⟩

theorem devOnly_some {w w' : World} {k : Option (Option V × Dev)} {r : Option V}
    (h : devOnly w k = some (r, w')) :
    ∃ d, k = some (r, d) ∧ w' = { w with dev := d, mem := w.mem.retire (d.race.clock hostParty) } := by
  unfold devOnly at h
  cases k with
  | none => cases h
  | some x =>
      obtain ⟨r', d⟩ := x
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨d, rfl, rfl⟩

/-- **What a launch on a stream leaves**: the device unchanged, or the launch
    on the stream's party over the buffers its bindings name. -/
theorem launchOnStream_post {w w' : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64}
    {r : Option V} (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (h : ffiCudaLaunchOnStream [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w = some (r, w')) :
    ∃ d, w' = { w with dev := d, mem := w.mem.retire (d.race.clock hostParty) } ∧
      (d = w.dev ∨ ∃ p ids, w.dev.party? (asI32 sid) = some p ∧
        readIds w.mem bindPtr (asI32 nBufs).toNat = some ids ∧ LaunchPost w.dev d p (ids.map Int.toNat)) := by
  obtain ⟨d, hk', rfl⟩ := devOnly_some h
  refine ⟨d, rfl, ?_⟩
  dsimp only at hk'
  split at hk'
  · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk'; exact hk'.2.symm
  · obtain ⟨ok, hok, hk'⟩ := Option.bind_eq_some_iff.mp hk'
    split at hk'
    · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk'; exact hk'.2.symm
    · obtain ⟨kernel, _, hk'⟩ := Option.bind_eq_some_iff.mp hk'
      split at hk'
      · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk'; exact hk'.2.symm
      · rename_i p hp
        rcases cudaLaunchOn_post hk hcap hk' with he | ⟨ids, hids, hpost⟩
        · exact Or.inl he
        · exact Or.inr ⟨p, ids, hp, hids, hpost⟩

/-- **What a stream sync leaves**: the device unchanged, or the host caught up
    with the stream's party. -/
theorem streamSync_post {w w' : World} {ctx sid : UInt64} {r : Option V}
    (h : ffiCudaStreamSync [ctx, sid] w = some (r, w')) :
    ∃ d, w' = { w with dev := d, mem := w.mem.retire (d.race.clock hostParty) } ∧
      (d = w.dev ∨ ∃ p, w.dev.party? (asI32 sid) = some p ∧ d = { w.dev with race := w.dev.race.sync p }) := by
  obtain ⟨d, hk', rfl⟩ := devOnly_some h
  refine ⟨d, rfl, ?_⟩
  obtain ⟨ok, hok, hk'⟩ := Option.bind_eq_some_iff.mp hk'
  split at hk'
  · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk'; exact hk'.2.symm
  · split at hk'
    · left; simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk'; exact hk'.2.symm
    · rename_i p hp
      split at hk'
      · cases hk'
      · simp only [devOk, Option.some.injEq, Prod.mk.injEq] at hk'
        exact Or.inr ⟨p, hp, hk'.2.symm⟩

/-- A download changes the host's bytes, not its memory's shape. -/
theorem download_mem {w w' : World} {ctx buf dst size : UInt64} {r : Option V}
    (h : ffiCudaDownload [ctx, buf, dst, size] w = some (r, w')) : MemSame w.mem w'.mem := by
  unfold ffiCudaDownload at h
  dsimp only at h
  split at h
  · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; rw [← h.2]; exact Static.MemSame.refl _
  · obtain ⟨ok, _, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; rw [← h.2]; exact Static.MemSame.refl _
    · split at h
      · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; rw [← h.2]; exact Static.MemSame.refl _
      · split at h
        · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; rw [← h.2]; exact Static.MemSame.refl _
        · obtain ⟨d, _, h⟩ := Option.bind_eq_some_iff.mp h
          obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          rw [← h.2]
          exact copyIn_same hm

-- ---------------------------------------------------------------------------
-- How readiness moves
-- ---------------------------------------------------------------------------

def belowB (es : List (Nat × Nat)) (c : Clock) : Bool := es.all fun e => decide (e.2 ≤ c.get e.1)

theorem below_of_belowB {es : List (Nat × Nat)} {c : Clock} (h : belowB es c = true) : Below es c := by
  intro e he
  have := List.all_eq_true.mp h e he
  simpa using this

def readyB (r : Race) (p b : Nat) (write : Bool) : Bool :=
  belowB (wAcc r b) (know r p) && (!write || belowB (rAcc r b) (know r p))

theorem ready_of_readyB {r : Race} {p b : Nat} {write : Bool} (h : readyB r p b write = true) :
    Ready r p b write := by
  simp only [readyB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true'] at h
  refine ⟨below_of_belowB h.1, fun hw => ?_⟩
  rcases h.2 with h2 | h2
  · rw [hw] at h2; cases h2
  · exact below_of_belowB h2

/-- **After an operation by `p` that wrote `b`, `p` may write `b` again.** -/
theorem ready_opped_self (r : Race) (p : Nat) (rs ws : List Nat) {b : Nat} (hb : b ∈ ws) :
    Ready (opped r p rs ws) p b true := by
  have hk : ClockLe (issued r p) (know (opped r p rs ws) p) := by
    rw [← opped_clock_self r p rs ws]; exact own_le_know _ _
  refine ⟨?_, fun _ => ?_⟩
  · rw [opped_wAcc, if_pos hb]
    intro e he
    simp only [List.mem_singleton] at he
    subst he
    exact hk p
  · rw [opped_rAcc_written _ _ _ _ _ hb]; exact Below.nil _

/-- **After the host waits for `p`, whatever `p` was ready for, anyone is.** -/
theorem ready_sync {r : Race} {p b : Nat} {wr : Bool} (h : Ready r p b wr) (q : Nat) (wr' : Bool)
    (hw : wr' = true → wr = true) : Ready (r.sync p) q b wr' := by
  have hk : ClockLe (know r p) (know (r.sync p) q) := by
    intro i
    have h1 := sync_host r p i
    have h2 := host_le_know (r.sync p) q i
    simp only [know, Clock.get_join] at h1 h2 ⊢
    have := (sync_grows r p) hostParty i
    omega
  refine ⟨h.1.mono hk, fun hw' => (h.2 (hw hw')).mono hk⟩

theorem ready_weaken {r : Race} {p b : Nat} (h : Ready r p b true) : Ready r p b false :=
  ⟨h.1, fun h' => by cases h'⟩

theorem streamSync_eff {w : World} {sid : UInt64} {p : Nat} (hl : w.dev.live = true)
    (hcap : w.dev.capture = none) (hp : w.dev.party? (asI32 sid) = some p) :
    ffiCudaStreamSync [cudaCtx, sid] w =
      some (some (ofInt .i32 0),
        { w with
          dev := { w.dev with race := w.dev.race.sync p }
          mem := w.mem.retire ((w.dev.race.sync p).clock hostParty) }) := by
  simp [ffiCudaStreamSync, devOnly, cudaCtxOk, hl, hp, capturing_none hcap, devOk,
    show cudaCtx ≠ 0 by decide]

end AlgorithmLib.HProg.DevSpec
