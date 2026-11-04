module
public import AlgorithmLib.Check.CudaRule
meta import AlgorithmLib.Check.CudaRule
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.CudaSound` — what `cudaRow` decides is what the driver does

**`cudaRow_sound`**: where `cudaRow` gives a typestate and an answer for a
driver call, the driver answers that call on the bits the arguments carry,
in a world the new typestate describes, with what the row says it answers.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState Part holds_of_mem OpsLive OpLive VendorKeeps runVendor_isSome)

/-- What every row needs of the world: the driver loaded and no capture in
    progress. -/
theorem cudaBase_world {S : TState} {w : World} (hS : S.holds w) (h : cudaBase S = true) :
    w.has .cuda = true ∧ w.dev.capture = none := by
  simp only [cudaBase, Bool.and_eq_true] at h
  have h1 : (Fact.part (.has .cuda) true).holds w := holds_of_mem hS h.1
  have h2 : (Fact.part .capturing false).holds w := holds_of_mem hS h.2
  simp only [Fact.holds, Part.get] at h1 h2
  exact ⟨h1, by cases hc : w.dev.capture <;> simp_all⟩

theorem cudaDrv_step {f : CudaFn} {bits : List UInt64} {w : World} (hc : w.dev.capture = none) :
    cudaDrv f bits w = cudaStep f bits w := by
  simp [cudaDrv, hc]

/-- Setting the driver initialised keeps every fact. -/
theorem holds_drvInit {x : Fact} {w : World} (h : x.holds w) :
    x.holds { w with dev := { w.dev with drvInit := true } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | drvInit => rfl
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- What a driver call the checker accepts leaves of the rest of the world:
    every region but the pinned one as large, every handle held still held,
    every heap block still live. -/
structure CuKept (w w' : World) : Prop where
  sizes : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r
  held : ∀ (k : Contracts.Held) x, k.live w x = true → k.live w' x = true
  blk : ∀ x n, Contracts.Blk w x n → Contracts.Blk w' x n
  obs : w'.obs = w.obs

/-- Every allocation live in `d` is live in `d'`. -/
def LiveKept (d d' : Dev) : Prop :=
  ∀ j : Nat, ((d.bufs[j]?).join).isSome = true → ((d'.bufs[j]?).join).isSome = true

theorem _root_.AlgorithmLib.HProg.Contracts.OpsLive.mono {d d' : Dev} {ops : Array DevOp} (h : OpsLive d ops) (hk : LiveKept d d') :
    OpsLive d' ops := fun op hop =>
  match h op hop with
  | .inl ⟨l, ids, he, hl⟩ => .inl ⟨l, ids, he, fun j hj => hk j (hl j hj)⟩
  | .inr ⟨c, ins, out, he, hlast, hl⟩ => .inr ⟨c, ins, out, he, hlast, fun j hj => hk j (hl j hj)⟩

/-- Every allocation's size kept keeps every allocation live. -/
theorem LiveKept.of_sizes {d d' : Dev}
    (h : ∀ (j n : Nat), (∃ c : ByteArray, (d.bufs[j]?).join = some c ∧ c.size = n) →
      ∃ c : ByteArray, (d'.bufs[j]?).join = some c ∧ c.size = n) : LiveKept d d' := by
  intro j hj
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp hj
  obtain ⟨c', hc', -⟩ := h j c.size ⟨c, hc, rfl⟩
  rw [hc']; rfl

theorem LiveKept.push (d : Dev) (b : Option ByteArray) : LiveKept d { d with bufs := d.bufs.push b } := by
  intro j hj
  have hi : j < d.bufs.size := by
    rcases Nat.lt_or_ge j d.bufs.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at hj; cases hj
  show ((d.bufs.push b)[j]?).join.isSome = true
  rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at hj

/-- A step that changes only the device keeps the rest. -/
theorem CuKept.dev (w : World) (d : Dev) : CuKept w { w with dev := d } :=
  ⟨fun _ _ => rfl, fun k _ h => by cases k <;> exact h, fun _ _ h => h.same rfl (Nat.le_refl _) rfl, rfl⟩

/-- What a driver row leaves: the driver's answer and world, the typestate
    the row computes holding there, the answer it says, and the rest kept. -/
def RowOk (f : CudaFn) (bits : List UInt64) (w : World) (S' : TState) (av : AV) : Prop :=
  w.has .cuda = true ∧ ∃ r w', cudaDrv f bits w = some (r, w') ∧ S'.holds w' ∧
    (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧ CuKept w w'

theorem cuOk_desc (Γ : Env) (w : World) : cuOk.Desc Γ w (ofInt .i32 cuSuccess) := rfl

theorem row_init {S S' : TState} {w : World} {Γ : Env} {fl : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [fl] bits) (h : cudaRow S .init [fl] = some (S', av)) :
    RowOk .init bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨hbase, hfl⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    match bits, hb with
    | [x], ⟨hx, _⟩ =>
      obtain rfl := known_bits hx hfl
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := { w.dev with drvInit := true } }, ?_, ?_,
        fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]; rfl
      · intro y hy
        simp only [List.mem_cons, List.mem_filter, bne_iff_ne, ne_eq, decide_eq_true_eq] at hy
        rcases hy with rfl | ⟨hy, -⟩
        · rfl
        · exact holds_drvInit (hS y hy)
      · cases hv; exact cuOk_desc Γ _
  · cases h

theorem retainedOf_mem {S : TState} {k : Nat} (h : retainedOf S = some k) : Fact.retained k ∈ S := by
  unfold retainedOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

theorem nbufsOf_mem {S : TState} {k : Nat} (h : nbufsOf S = some k) : Fact.nbufs k ∈ S := by
  unfold nbufsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

/-- A driver call's store of eight bytes through the pointer it is handed
    leaves what a program's own store would. -/
theorem afterStore_world {S : TState} {w : World} {a v : UInt64} {ao : Option UInt64} {m : Mem}
    (hS : S.holds w) (hm : w.mem.store a 8 v = some m) (ha : ∀ x, ao = some x → x = a) :
    (Contracts.afterStore ao (some v) 8 S).holds { w with mem := m } ∧ CuKept w { w with mem := m } := by
  have hsz : m.sizes = w.mem.sizes := Contracts.store_sizes_eq hm
  refine ⟨Contracts.afterStore_sound hS (fun p => ?_) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl, rfl, rfl⟩
    hsz (fun _ => hm) ha (fun _ e => by cases e; rfl) (by decide), fun r _ => by rw [hsz], fun k _ h => by
      cases k <;> exact h, fun _ _ h => h.same rfl (Nat.le_of_eq (congrFun hsz .pinned).symm)
        (Contracts.store_live hm), rfl⟩
  cases p <;> first
    | rfl
    | exact Contracts.store_frozen hm
    | (show m.busy.isEmpty = w.mem.busy.isEmpty; rw [Contracts.store_busy hm])

/-- Retaining or releasing the context keeps every fact but how many times
    it is retained. -/
theorem holds_retained {x : Fact} {w : World} (k : Nat) (hx : ∀ n, x ≠ .retained n) (h : x.holds w) :
    x.holds { w with dev := { w.dev with retained := k } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | retained n => exact absurd rfl (hx n)
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- Making the context current keeps every fact but whether it is. -/
theorem holds_live {x : Fact} {w : World} (hx : ∀ b, x ≠ .part .cuda b) (h : x.holds w) :
    x.holds { w with dev := { w.dev with live := true } } := by
  cases x with
  | part p b => cases p <;> first | exact h | exact absurd rfl (hx b)
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- A new allocation keeps every fact but how many there are. -/
theorem holds_push {x : Fact} {w : World} (b : Option ByteArray) (hx : ∀ n, x ≠ .nbufs n) (h : x.holds w) :
    x.holds { w with dev := { w.dev with bufs := w.dev.bufs.push b } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | devBuf a n => exact Device.BufIs.push h b
  | nbufs n => exact absurd rfl (hx n)
  | alloc i n =>
      obtain ⟨c, hc, hs⟩ := h
      have hi : i < w.dev.bufs.size := by
        rcases Nat.lt_or_ge i w.dev.bufs.size with hi | hi
        · exact hi
        · rw [Array.getElem?_eq_none hi] at hc; cases hc
      refine ⟨c, ?_, hs⟩
      show (w.dev.bufs.push b)[i]?.join = some c
      rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at hc
  | graph i => obtain ⟨ops, hg, hl⟩ := h; exact ⟨ops, hg, hl.mono (LiveKept.push _ _)⟩
  | exec i => obtain ⟨ops, hg, hl⟩ := h; exact ⟨ops, hg, hl.mono (LiveKept.push _ _)⟩
  | cap q j r evs =>
      obtain ⟨c, hc, ho, hj, hr, hl, hev⟩ := h
      exact ⟨c, hc, ho, hj, hr, hl.mono (LiveKept.push _ _), hev⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem CuKept.trans {w w' w'' : World} (h : CuKept w w') (h' : CuKept w' w'') : CuKept w w'' :=
  ⟨fun r hr => (h'.sizes r hr).trans (h.sizes r hr), fun k x hl => h'.held k x (h.held k x hl),
    fun x n hb => h'.blk x n (h.blk x n hb), h'.obs.trans h.obs⟩

/-- A fact a filter on its constructor kept is not of that constructor. -/
theorem not_retained {y : Fact} (h : (match y with | .retained _ => false | _ => true) = true) :
    ∀ n, y ≠ .retained n := by
  intro n e; subst e; simp at h

theorem not_cuda {y : Fact} (h : (match y with | .part .cuda _ => false | _ => true) = true) :
    ∀ b, y ≠ .part .cuda b := by
  intro b e; subst e; simp at h

theorem not_nbufs {y : Fact} (h : (match y with | .nbufs _ => false | _ => true) = true) :
    ∀ n, y ≠ .nbufs n := by
  intro n e; subst e; simp at h

theorem row_ctxSetCurrent {S S' : TState} {w : World} {Γ : Env} {c : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [c] bits) (h : cudaRow S .ctxSetCurrent [c] = some (S', av)) :
    RowOk .ctxSetCurrent bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i k hk
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
      obtain ⟨⟨hbase, hcc⟩, hk0⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hret : w.dev.retained = k := hS _ (retainedOf_mem hk)
      match bits, hb with
      | [x], ⟨hx, _⟩ =>
        obtain rfl := known_bits hx hcc
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := { w.dev with live := true } }, ?_, ?_,
          fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]; simp [cudaStep, hret, hk0, cudaCtx]
        · intro y hy
          simp only [List.mem_cons, List.mem_filter] at hy
          rcases hy with rfl | ⟨hy, hn⟩
          · rfl
          · exact holds_live (not_cuda hn) (hS y hy)
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_primaryCtxRetain {S S' : TState} {w : World} {Γ : Env} {p dv : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [p, dv] bits) (h : cudaRow S .primaryCtxRetain [p, dv] = some (S', av)) :
    RowOk .primaryCtxRetain bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i k hk
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨hbase, hinit⟩, hdv⟩, hmem⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hret : w.dev.retained = k := hS _ (retainedOf_mem hk)
      have hdi : w.dev.drvInit = true := hS _ (List.contains_iff_mem.mp hinit)
      match bits, hb with
      | [pa, x], ⟨hpa, hx, _⟩ =>
        obtain rfl := known_bits hx hdv
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem cudaCtx)
        obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
        refine ⟨hhas, some (ofInt .i32 cuSuccess),
          { w with mem := m, dev := { w.dev with retained := w.dev.retained + 1 } }, ?_, ?_,
          fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
        · rw [cudaDrv_step hcap]; simp [cudaStep, hdi, hm]; rfl
        · intro y hy
          simp only [List.mem_cons, List.mem_filter] at hy
          rcases hy with rfl | ⟨hy, hn⟩
          · show w.dev.retained + 1 = k + 1; rw [hret]
          · exact holds_retained (w := { w with mem := m }) _ (not_retained hn) (hst y hy)
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_memAlloc {S S' : TState} {w : World} {Γ : Env} {p sz : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [p, sz] bits) (h : cudaRow S .memAlloc [p, sz] = some (S', av)) :
    RowOk .memAlloc bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i n z hn hz
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨⟨⟨⟨hbase, hlive⟩, hmem⟩, hz0⟩, hz36⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hcnt : w.dev.bufs.size = n := hS _ (nbufsOf_mem hn)
      subst hcnt
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      match bits, hb with
      | [pa, x], ⟨hpa, hx, _⟩ =>
        obtain ⟨t, hx⟩ := hx
        obtain ⟨y, hy, hlo, hhi, -⟩ := AV.num_desc hx hz
        cases hy
        replace hz0 : 0 < x.toNat := Nat.lt_of_lt_of_le hz0 hlo
        replace hz36 : x.toNat < 2 ^ 36 := Nat.lt_of_le_of_lt hhi hz36
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (devPtr w.dev.bufs.size))
        obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
        have hz0' : (x == 0) = false := by
          simp only [beq_eq_false_iff_ne, ne_eq]; intro e; subst e; simp at hz0
        refine ⟨hhas, some (ofInt .i32 cuSuccess),
          { w with mem := m, dev := { w.dev with
            bufs := w.dev.bufs.push (some (exactly (w.devFill w.dev.bufs.size x.toNat) x.toNat)) } }, ?_, ?_,
          fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
        · rw [cudaDrv_step hcap]; simp [cudaStep, hl, hz0', Nat.not_le.mpr hz36, hm]
        · intro y hy
          simp only [List.mem_cons, List.mem_append, List.mem_filter] at hy
          rcases hy with rfl | (hy | ⟨hy, hnb⟩)
          · show (w.dev.bufs.push _).size = w.dev.bufs.size + 1; rw [Array.size_push]
          · split at hy
            · rename_i hlh
              simp only [List.mem_singleton] at hy
              subst hy
              have hxe : x.toNat = z.lo := Nat.le_antisymm (hlh ▸ hhi) hlo
              refine ⟨exactly (w.devFill w.dev.bufs.size x.toNat) x.toNat, ?_, hxe ▸ exactly_size _ _⟩
              show (w.dev.bufs.push _)[w.dev.bufs.size]?.join = _
              rw [Array.getElem?_push_size]; rfl
            · cases hy
          · exact holds_push (w := { w with mem := m }) _ (not_nbufs hnb) (hst y hy)
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- **An allocation the row accepts has the size asked for**: allocation `k`,
    the count before it, holds that many bytes after the call. -/
theorem memAlloc_sized {S S' : TState} {w : World} {Γ : Env} {p sz : AV} {bits : List UInt64} {av : AV}
    {k : Nat} {r : Option V} {w' : World}
    (hS : S.holds w) (hb : ArgBits Γ w [p, sz] bits) (h : cudaRow S .memAlloc [p, sz] = some (S', av))
    (hk : nbufsOf S = some k) (hs : cudaDrv .memAlloc bits w = some (r, w')) :
    (Fact.alloc k (bits.getD 1 0).toNat).holds w' := by
  simp only [cudaRow] at h
  split at h
  · rename_i n z hn hz
    rw [hk] at hn; cases hn
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨⟨⟨⟨hbase, hlive⟩, hmem⟩, hz0⟩, hz36⟩ := hc
      obtain ⟨-, hcap⟩ := cudaBase_world hS hbase
      have hcnt : w.dev.bufs.size = k := hS _ (nbufsOf_mem hk)
      subst hcnt
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      match bits, hb with
      | [pa, x], ⟨hpa, hx, _⟩ =>
        obtain ⟨t, hx⟩ := hx
        obtain ⟨y, hy, hlo, hhi, -⟩ := AV.num_desc hx hz
        cases hy
        replace hz0 : 0 < x.toNat := Nat.lt_of_lt_of_le hz0 hlo
        replace hz36 : x.toNat < 2 ^ 36 := Nat.lt_of_le_of_lt hhi hz36
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (devPtr w.dev.bufs.size))
        have hz0' : (x == 0) = false := by
          simp only [beq_eq_false_iff_ne, ne_eq]; intro e; subst e; simp at hz0
        rw [cudaDrv_step hcap] at hs
        simp [cudaStep, hl, hz0', Nat.not_le.mpr hz36, hm] at hs
        obtain ⟨-, rfl⟩ := hs
        refine ⟨exactly (w.devFill w.dev.bufs.size x.toNat) x.toNat, ?_, exactly_size _ _⟩
        show (w.dev.bufs.push _)[w.dev.bufs.size]?.join = _
        rw [Array.getElem?_push_size]; rfl
    · cases h
  · cases h

/-- **A step on the device that leaves its shape**: streams, events, the
    context, the count of allocations and every allocation's size as they
    were, keeps every fact but those that read the tracker's clocks, given
    the default stream's order where the typestate had it. -/
theorem holds_devKeep {x : Fact} {w : World} {d : Dev} (h : x.holds w) (hx : raceKeeps x = true)
    (hs : d.streams = w.dev.streams) (he : d.events = w.dev.events) (hl : d.live = w.dev.live)
    (hc : d.capture = w.dev.capture) (hi : d.drvInit = w.dev.drvInit) (hr : d.retained = w.dev.retained)
    (hn : d.bufs.size = w.dev.bufs.size) (hmo : d.modules = w.dev.modules) (hfu : d.funcs = w.dev.funcs)
    (hdb : ∀ i n, Device.BufIs i n w.dev → Device.BufIs i n d)
    (hal : ∀ (j : Nat) (n : Nat), x = .alloc j n → (∃ c : ByteArray, (w.dev.bufs[j]?).join = some c ∧ c.size = n) →
      ∃ c : ByteArray, (d.bufs[j]?).join = some c ∧ c.size = n)
    (hseq : x = .devSeq → Device.TrackOk d ∧ Device.DefaultOnly d.race)
    (hlive : x.isGraph = true → LiveKept w.dev d)
    (hbl : d.blas = w.dev.blas := by first | rfl | assumption)
    (hgr : d.graphs = w.dev.graphs := by first | rfl | assumption)
    (hex : d.execs = w.dev.execs := by first | rfl | assumption) :
    x.holds { w with dev := d } := by
  cases x with
  | part p b =>
      cases p
      case cuda => show d.live = b; rw [hl]; exact h
      case capturing => show d.capture.isSome = b; rw [hc]; exact h
      all_goals exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact hseq rfl
  | devBuf a n => exact hdb _ _ h
  | streams l => show d.streams.toList = l; rw [hs]; exact h
  | events n => show d.events.size = n; rw [he]; exact h
  | evLive e =>
      have h' : (w.dev.event? e).isSome = true := h
      show (d.event? e).isSome = true; unfold Dev.event? at h' ⊢; rw [he]; exact h'
  | readyAll _ => simp [raceKeeps] at hx
  | evAll _ => simp [raceKeeps] at hx
  | drvInit => show d.drvInit = true; rw [hi]; exact h
  | retained n => show d.retained = n; rw [hr]; exact h
  | nbufs n => show d.bufs.size = n; rw [hn]; exact h
  | alloc j n => exact hal j n rfl h
  | nmods n => show d.modules.size = n; rw [hmo]; exact h
  | nfuncs n => show d.funcs.size = n; rw [hfu]; exact h
  | mod i t => show ∃ s, d.modules[i]? = _ ∧ _; rw [hmo]; exact h
  | fn i sz => show ∃ mi e t, d.funcs[i]? = some (mi, e) ∧ d.modules[mi]? = some (some t) ∧ _; rw [hmo, hfu]; exact h
  | nblas n => show d.blas.size = n; rw [hbl]; exact h
  | blasOn i t => show d.blas[i]? = _; rw [hbl]; exact h
  | ngraphs n => show d.graphs.size = n; rw [hgr]; exact h
  | nexecs n => show d.execs.size = n; rw [hex]; exact h
  | graph i =>
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show d.graphs[i]? = _; rw [hgr]; exact hg, hl.mono (hlive rfl)⟩
  | exec i =>
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show d.execs[i]? = _; rw [hex]; exact hg, hl.mono (hlive rfl)⟩
  | cap q j r evs =>
      obtain ⟨c, hcc, ho, hj, hr, hl, hev⟩ := h
      exact ⟨c, by show d.capture = _; rw [hc]; exact hcc, ho, hj, hr, hl.mono (hlive rfl),
        fun e he' => by unfold Dev.event? at hev ⊢; rw [he]; exact hev e he'⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- Every allocation's size as it was carries both buffer facts. -/
theorem sizes_keep {d d' : Dev} (hn : d'.bufs.size = d.bufs.size)
    (hb : ∀ j : Nat, (d'.bufs[j]?.join).map ByteArray.size = (d.bufs[j]?.join).map ByteArray.size) :
    (∀ i n, Device.BufIs i n d → Device.BufIs i n d') ∧
      ∀ (j : Nat) (n : Nat), (∃ c : ByteArray, (d.bufs[j]?).join = some c ∧ c.size = n) →
        ∃ c : ByteArray, (d'.bufs[j]?).join = some c ∧ c.size = n := by
  refine ⟨fun i n h => Device.BufIs.of_sizes h (Nat.le_of_eq hn.symm) fun j => ?_, fun j n ⟨c, hc, hs⟩ => ?_⟩
  · unfold Dev.get?
    split
    · rfl
    · exact hb j.toNat
  · have := hb j
    rw [hc] at this
    obtain ⟨c', hc', hs'⟩ := Option.map_eq_some_iff.mp this
    exact ⟨c', hc', hs'.trans hs⟩

/-- The allocation the typestate places `n` bytes from `p` in, as the model
    finds it. -/
theorem allocFor_range {S : TState} {w : World} {p : UInt64} {n : Nat} (hS : S.holds w)
    (h : allocFor S p n = true) : ∃ id off b, w.dev.range? p n = some (id, off, b) := by
  unfold allocFor at h
  split at h
  · rename_i id off hp
    obtain ⟨y, hy, hm⟩ := List.any_eq_true.mp h
    cases y with
    | alloc i sz =>
        simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hm
        obtain ⟨rfl, hle⟩ := hm
        obtain ⟨b, hb, hsz⟩ := (hS _ hy : (Fact.alloc i sz).holds w)
        refine ⟨i, off, b, ?_⟩
        simp only [Dev.range?, hp, Option.bind_eq_bind, Option.bind_some, hb]
        rw [if_pos (by omega)]
    | _ => simp at hm
  · cases h

/-- Replacing a live allocation's contents by as many bytes keeps every
    allocation's size. -/
theorem put_sizes {d : Dev} {id : Nat} {c : ByteArray}
    (h : (d.bufs[id]?.join).map ByteArray.size = some c.size) (j : Nat) :
    ((d.put id (some c)).bufs[j]?.join).map ByteArray.size = (d.bufs[j]?.join).map ByteArray.size := by
  simp only [Dev.put, Array.getElem?_setIfInBounds]
  by_cases e : id = j
  · subst e
    by_cases hj : id < d.bufs.size
    · simp only [hj, if_true, Option.join_some, Option.map_some]; exact h.symm
    · simp [hj]
  · simp [e]

theorem put_size (d : Dev) (id : Nat) (c : Option ByteArray) : (d.put id c).bufs.size = d.bufs.size := by
  simp [Dev.put]

theorem row_memcpyHtoD {S S' T : TState} {w : World} {Γ : Env} {dst src n : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [dst, src, n] bits)
    (hT : T.holds w) (hv : cudaVal T .memcpyHtoD (bits.map some) = true)
    (h : cudaRow S .memcpyHtoD [dst, src, n] = some (S', av)) :
    RowOk .memcpyHtoD bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨⟨⟨hbase, hlive⟩, hseq⟩, hmem⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
    match bits, hb with
    | [x, y, z], ⟨hx, hy, hz, _⟩ =>
        simp only [List.map_cons, List.map_nil, cudaVal, argAt, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.join_some] at hv
        obtain ⟨id, off, buf, hr⟩ := allocFor_range hT hv
        obtain ⟨bytes, hbytes⟩ := Option.isSome_iff_exists.mp (memB_readable hS hy hz hmem)
        have hbsz : bytes.size = z.toNat := Sem.copyOut_size hbytes
        have hready := (Contracts.readyIn_sound hS (hseq)).ready id true
        obtain ⟨(r : Race), hop⟩ := Option.isSome_iff_exists.mp (ExtContracts.op_isSome_w hready)
        have hfit : off + z.toNat ≤ buf.size := by
          simp only [Dev.range?, Option.bind_eq_bind] at hr
          obtain ⟨⟨i, o⟩, hp, hr⟩ := Option.bind_eq_some_iff.mp hr
          obtain ⟨c, hc, hr⟩ := Option.bind_eq_some_iff.mp hr
          split at hr
          · cases hr; omega
          · cases hr
        have hlb : (w.dev.bufs[id]?.join) = some buf := by
          simp only [Dev.range?, Option.bind_eq_bind] at hr
          obtain ⟨⟨i, o⟩, hp, hr⟩ := Option.bind_eq_some_iff.mp hr
          obtain ⟨c, hc, hr⟩ := Option.bind_eq_some_iff.mp hr
          split at hr
          · cases hr; exact hc
          · cases hr
        let d' : Dev := { (w.dev.put id (some (overwrite buf off bytes))) with race := r.sync defaultParty }
        have hsw : w.dev.syncWrite id (overwrite buf off bytes) = some d' := by
          simp only [Dev.syncWrite, hop, Option.bind_eq_bind, Option.bind_some]; rfl
        have hsizes := put_sizes (d := w.dev) (id := id) (c := overwrite buf off bytes)
          (by rw [hlb, Device.overwrite_size (by omega)]; rfl)
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]
          simp only [cudaStep, hl, Bool.not_true, Bool.false_eq_true, if_false, hr, hbytes, hsw,
            Option.bind_eq_bind, Option.bind_some, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
        · intro f hf
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
          have hk2 := sizes_keep (d := w.dev) (d' := d') (put_size _ _ _) hsizes
          exact holds_devKeep (hS f hf) hk' rfl rfl rfl rfl rfl rfl (put_size _ _ _) rfl rfl hk2.1
            (fun j n _ => hk2.2 j n) (hlive := fun _ => LiveKept.of_sizes hk2.2)
            (fun e => by subst e; exact ⟨Device.syncWrite_track hsw (hS _ hf).1, Device.syncWrite_seq hsw (hS _ hf).2⟩)
        · cases hv; exact cuOk_desc Γ _
  · cases h

/-- The tracker after an operation on the default stream is still one the
    race proof covers. -/
theorem track_op {d d' : Dev} {r : Race} {rs ws : List Nat} (ht : Device.TrackOk d)
    (hop : d.race.op defaultParty rs ws = some r) (h1 : d'.race = r) (h2 : d'.capture = d.capture)
    (h3 : d'.events = d.events) : Device.TrackOk d' :=
  Device.track_race ht (fun _ h => .op defaultParty rs ws h hop) d' h1 h2 h3

theorem creached_foldSync : ∀ (ps : List Nat) (r : Race) (evs : List Clock), Device.CReached r evs →
    Device.CReached (ps.foldl (fun r p => r.sync p) r) evs
  | [], _, _, h => h
  | p :: ps, r, evs, h => creached_foldSync ps _ _ (.learnParty hostParty p h)

theorem defaultOnly_foldSync : ∀ (ps : List Nat) (r : Race), Device.DefaultOnly r →
    Device.DefaultOnly (ps.foldl (fun r p => r.sync p) r)
  | [], _, h => h
  | p :: ps, r, h => defaultOnly_foldSync ps _ (h.sync p)

/-- Where the typestate has the default stream's order, the default stream may
    write any allocation. -/
theorem op_write {S : TState} {w : World} (hS : S.holds w) (hseq : S.contains .devSeq = true) (id : Nat) :
    ∃ r : Race, w.dev.race.op defaultParty [] [id] = some r :=
  Option.isSome_iff_exists.mp (ExtContracts.op_isSome_w
    ((Contracts.readyIn_sound hS (Contracts.readyIn_seq hseq)).ready id true))

theorem range_alloc {d : Dev} {p : UInt64} {n id off : Nat} {b : ByteArray}
    (hr : d.range? p n = some (id, off, b)) : (d.bufs[id]?).join = some b ∧ off + n ≤ b.size := by
  simp only [Dev.range?, Option.bind_eq_bind] at hr
  obtain ⟨⟨i, o⟩, hp, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨c, hc, hr⟩ := Option.bind_eq_some_iff.mp hr
  split at hr
  · cases hr; exact ⟨hc, by omega⟩
  · cases hr

theorem row_memsetD8 {S S' : TState} {w : World} {Γ : Env} {dst uc n : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [dst, uc, n] bits)
    (h : cudaRow S .memsetD8 [dst, uc, n] = some (S', av)) :
    RowOk .memsetD8 bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i dv k hd hk
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hseq⟩, hal⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      have hsq : (Fact.devSeq).holds w := hS _ (List.contains_iff_mem.mp hseq)
      match bits, hb with
      | [x, u, z], ⟨hx, _, hz, _⟩ =>
        obtain rfl := known_bits hx hd
        obtain rfl := known_bits hz hk
        obtain ⟨id, off, buf, hr⟩ := allocFor_range hS hal
        obtain ⟨hlb, hfit⟩ := range_alloc hr
        obtain ⟨r, hop⟩ := op_write hS hseq id
        let fill := ByteArray.mk (Array.replicate z.toNat (u &&& 0xff).toUInt8)
        have hfs : fill.size = z.toNat := by simp [fill, ByteArray.size]
        let d' : Dev := ({ w.dev with race := r }).put id (some (overwrite buf off fill))
        have hsizes := put_sizes (d := { w.dev with race := r }) (id := id) (c := overwrite buf off fill)
          (by show (w.dev.bufs[id]?.join).map _ = _; rw [hlb, Device.overwrite_size (by omega)]; rfl)
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hr, hop,
            Option.bind_eq_bind, Option.bind_some, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
          rfl
        · intro f hf
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
          have hk2 := sizes_keep (d := w.dev) (d' := d') (put_size _ _ _) hsizes
          exact holds_devKeep (hS f hf) hk' rfl rfl rfl rfl rfl rfl (put_size _ _ _) rfl rfl hk2.1
            (fun j n _ => hk2.2 j n) (hlive := fun _ => LiveKept.of_sizes hk2.2)
            (fun e => by
              subst e
              exact ⟨Device.put_track (track_op (d' := { w.dev with race := r }) hsq.1 hop rfl rfl rfl) _ _,
                Device.DefaultOnly.op hsq.2 hop⟩)
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_ctxSynchronize {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [] bits)
    (h : cudaRow S .ctxSynchronize [] = some (S', av)) :
    RowOk .ctxSynchronize bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨⟨hbase, hlive⟩, hseq⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
    have hsq : (Fact.devSeq).holds w := hS _ (List.contains_iff_mem.mp hseq)
    match bits, hb with
    | [], _ =>
      let d' : Dev := { w.dev with race := w.dev.race.syncAll }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, devOnly, cuRes, Option.map_some,
          DevSpec.retire_nil hbusy]
        rfl
      · intro f hf
        obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
        exact holds_devKeep (hS f hf) hk' rfl rfl rfl rfl rfl rfl rfl rfl rfl (fun _ _ h => h) (fun _ _ _ h => h)
          (hlive := fun _ _ h => h)
          (fun e => by
            subst e
            exact ⟨Device.track_race hsq.1 (fun evs h => creached_foldSync _ _ evs h) d' rfl rfl rfl,
              defaultOnly_foldSync _ _ hsq.2⟩)
      · cases hv; exact cuOk_desc Γ _
  · cases h

/-- A copy into host memory leaves every part but the ones memory's state
    decides as it found them, and those too. -/
theorem copyIn_parts {w : World} {a : UInt64} {src : ByteArray} {m : Mem} (h : copyIn w.mem a src = some m)
    (p : Part) : p.get { w with mem := m } = p.get w := by
  cases p <;> first
    | rfl
    | exact Contracts.copyIn_frozen h
    | (show m.busy.isEmpty = w.mem.busy.isEmpty; rw [Contracts.copyIn_busy h])

/-- **What a copy into host memory keeps**: the facts `copyKeeps` keeps. -/
theorem copyKeeps_sound {S : TState} {w : World} {a : UInt64} {src : ByteArray} {m : Mem} (hS : S.holds w)
    (h : copyIn w.mem a src = some m) (hn : src.size ≤ 2 ^ 64) :
    TState.holds (S.filter (copyKeeps a src.size)) { w with mem := m } := by
  intro x hx
  obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
  simp only [copyKeeps, Bool.and_eq_true, Bool.not_eq_true'] at hk
  obtain ⟨hk, hht⟩ := hk
  have hx0 := hS x hm
  have hsz : m.sizes = w.mem.sizes := Contracts.copyIn_sizes_eq h
  cases x with
  | part p b => exact (copyIn_parts h p).trans hx0
  | room r k => show k ≤ m.sizes r; rw [hsz]; exact hx0
  | held k a => cases k <;> exact hx0
  | opened k a => cases k <;> exact hx0
  | cstr _ => simp [Contracts.storeKeeps] at hk
  | htAt _ _ _ => simp at hht
  | roomArg r x => show x.toNat ≤ m.sizes r ∧ _; rw [hsz]; exact hx0
  | pinnedUsed n => show m.sizes .pinned ≤ n; rw [hsz]; exact hx0
  | cstrIn c k =>
      simp only [Contracts.storeKeeps, decide_eq_true_eq] at hk
      have hz' : m.frozen = false := by rw [Contracts.copyIn_frozen h]; exact hx0.1
      have hw := hx0.2.2.1
      refine Contracts.cstrIn_kept hx0 hz' (fun rg _ => congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
      refine copyIn_other h _ fun j hj e => ?_
      have e2 := congrArg UInt64.toNat e
      rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
        Nat.mod_eq_of_lt (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt (by omega : j < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : c.toNat + i < 2 ^ 64)] at e2
      rcases hk with h1 | ⟨h1, h2⟩
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
  | text c t k _ =>
      simp only [Contracts.storeKeeps, decide_eq_true_eq] at hk
      have hz' : m.frozen = false := by rw [Contracts.copyIn_frozen h]; exact hx0.1.1
      have hw := hx0.1.2.2.1
      refine Contracts.text_kept hx0 hz' (fun rg _ => congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
      refine copyIn_other h _ fun j hj e => ?_
      have e2 := congrArg UInt64.toNat e
      rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
        Nat.mod_eq_of_lt (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt (by omega : j < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : c.toNat + i < 2 ^ 64)] at e2
      rcases hk with h1 | ⟨h1, h2⟩
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
  | cell c v =>
      simp only [Contracts.storeKeeps, Bool.and_eq_true, List.all_eq_true, List.mem_range,
        Bool.not_eq_true'] at hk
      obtain ⟨hc, hmiss⟩ := hk
      obtain ⟨rg, off, hd, hpin, hb⟩ := Contracts.cellOk_spec hc
      have hz' : m.frozen = false := by rw [Contracts.copyIn_frozen h]; exact Contracts.frozen_of_load hd hpin hx0
      refine Contracts.load_kept hd hpin hb hx0 hz' (congrFun hsz rg) fun i hi => Unchanged.of_eq ?_
      refine copyIn_other h _ fun j hj e => ?_
      have := within_of hn ⟨j, hj, e⟩
      rw [hmiss i hi] at this; cases this
  | wg1 _ s => cases s <;> exact hx0
  | wgText c k _ =>
      simp only [Contracts.storeKeeps, decide_eq_true_eq] at hk
      have hz' : m.frozen = false := by rw [Contracts.copyIn_frozen h]; exact hx0.1
      have hw := hx0.2.1
      refine Contracts.wgText_kept hx0 hz' (fun rg _ => congrFun hsz rg) rfl fun i hi => Unchanged.of_eq ?_
      refine copyIn_other h _ fun j hj e => ?_
      have e2 := congrArg UInt64.toNat e
      rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
        Nat.mod_eq_of_lt (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt (by omega : j < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : c.toNat + i < 2 ^ 64)] at e2
      rcases hk with h1 | ⟨h1, h2⟩
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
      · rw [Nat.mod_eq_of_lt (by omega : a.toNat + j < 2 ^ 64)] at e2; omega
  | _ => exact hx0

/-- What `afterCopy` leaves holds after the copy. -/
theorem afterCopy_sound {S : TState} {w : World} {ao : Option UInt64} {a : UInt64} {src : ByteArray} {m : Mem}
    (hS : S.holds w) (h : copyIn w.mem a src = some m) (ha : ∀ x, ao = some x → x = a) (hn : src.size ≤ 2 ^ 64) :
    (afterCopy ao src.size S).holds { w with mem := m } ∧ CuKept w { w with mem := m } := by
  have hsz : m.sizes = w.mem.sizes := Contracts.copyIn_sizes_eq h
  refine ⟨?_, fun r _ => by rw [hsz], fun k _ h => by cases k <;> exact h,
    fun _ _ hb => hb.same rfl (Nat.le_of_eq (congrFun hsz .pinned).symm) (copyIn_meta h).1, rfl⟩
  cases ao with
  | some x =>
      obtain rfl := ha x rfl
      exact copyKeeps_sound hS h hn
  | none =>
      exact Contracts.nonCell_kept hS (copyIn_parts h) (fun k _ => by cases k <;> rfl) ⟨rfl, rfl, rfl, rfl, rfl⟩ hsz

theorem allocBase_spec {S : TState} {p : UInt64} {id : Nat} (h : allocBase S p = some id) :
    devPtrOf p = some (id, 0) ∧ ∃ sz, Fact.alloc id sz ∈ S := by
  unfold allocBase at h
  split at h
  · rename_i i hp
    split at h
    · rename_i hany
      cases h
      obtain ⟨z, hz, hzm⟩ := List.any_eq_true.mp hany
      cases z with
      | alloc i' sz => simp only [beq_iff_eq] at hzm; subst hzm; exact ⟨hp, sz, hz⟩
      | _ => simp at hzm
    · cases h
  · cases h

theorem row_memFree {S S' : TState} {w : World} {Γ : Env} {p : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [p] bits)
    (h : cudaRow S .memFree [p] = some (S', av)) :
    RowOk .memFree bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i id hid
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨hbase, hlive⟩, hseq⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      have hsq : (Fact.devSeq).holds w := hS _ (List.contains_iff_mem.mp hseq)
      match bits, hb with
      | [x], ⟨hx, _⟩ =>
        obtain ⟨y, hpy, hy⟩ := Option.bind_eq_some_iff.mp hid
        obtain rfl := known_bits hx hpy
        obtain ⟨hp, sz, hz⟩ := allocBase_spec hy
        obtain ⟨buf, hbuf, -⟩ := (hS _ hz : (Fact.alloc id sz).holds w)
        have hr : w.dev.range? x 0 = some (id, 0, buf) := by
          simp only [Dev.range?, hp, Option.bind_eq_bind, Option.bind_some, hbuf]
          rw [if_pos (by omega)]
        obtain ⟨r, hop⟩ := op_write hS hseq id
        let d' : Dev := { (w.dev.put id none) with race := r }
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hr, hop, bne_self_eq_false,
            Option.bind_eq_bind, Option.bind_some, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
          rfl
        · intro f hf
          obtain ⟨hf, hna⟩ := List.mem_filter.mp hf
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
          refine holds_devKeep (hS f hf) hk' rfl rfl rfl rfl rfl rfl (put_size _ _ _) rfl rfl
            (fun a n h => Device.BufIs.put_none h id) (fun j n e ⟨c, hc, hs⟩ => ?_)
            (fun e => by subst e; exact ⟨track_op hsq.1 hop rfl rfl rfl, Device.DefaultOnly.op hsq.2 hop⟩)
            (hlive := fun hg => by cases f <;> simp [Fact.isGraph] at hg <;> simp at hna)
          subst e
          have hj : j ≠ id := by simpa using hna
          refine ⟨c, ?_, hs⟩
          show ((w.dev.put id none).bufs[j]?).join = some c
          simp only [Dev.put, Array.getElem?_setIfInBounds, Ne.symm hj]; exact hc
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_memcpyDtoH {S S' T : TState} {w : World} {Γ : Env} {dst src n : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [dst, src, n] bits)
    (hT : T.holds w) (hv : cudaVal T .memcpyDtoH (bits.map some) = true)
    (h : cudaRow S .memcpyDtoH [dst, src, n] = some (S', av)) :
    RowOk .memcpyDtoH bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨⟨⟨hbase, hlive⟩, hseq⟩, hmem⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
    match bits, hb with
    | [x, y, z], ⟨hx, hy, hz, _⟩ =>
        simp only [List.map_cons, List.map_nil, cudaVal, argAt, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.join_some] at hv
        obtain ⟨id, off, buf, hr⟩ := allocFor_range hT hv
        obtain ⟨hlb, hfit⟩ := range_alloc hr
        have hready := (Contracts.readyIn_sound hS (hseq)).ready id false
        obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (ExtContracts.op_isSome_r hready)
        let src' := buf.extract off (off + z.toNat)
        have hsz' : src'.size = z.toNat := by simp [src', ByteArray.size_extract]; omega
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memB_writable hS hx hz hmem src' hsz')
        let d' : Dev := { w.dev with race := r.sync defaultParty }
        have hsr : w.dev.syncRead id = some d' := by
          simp only [Dev.syncRead, hop, Option.bind_eq_bind, Option.bind_some]; rfl
        have hm' : copyIn w.mem x (buf.extract off (off + z.toNat)) = some m := hm
        have hle : src'.size ≤ 2 ^ 64 := by rw [hsz']; exact Nat.le_of_lt (UInt64.toNat_lt _)
        have hkc := (afterCopy_sound (ao := none) hS hm (fun _ e => by cases e) hle).2
        have hcp : (match n.bits with
            | some k => afterCopy dst.bits k.toNat S
            | none => afterCopy none 0 S).holds { w with mem := m } := by
          cases hn : n.bits with
          | some k =>
              obtain rfl := known_bits hz hn
              have := (afterCopy_sound (ao := dst.bits) hS hm (fun q hq => (known_bits hx hq).symm) hle).1
              rw [hsz'] at this
              exact this
          | none => exact (afterCopy_sound (ao := none) hS hm (fun _ e => by cases e) hle).1
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with mem := m, dev := d' }, ?_, ?_, fun Γ v hv => ?_,
          hkc.trans (CuKept.dev _ _)⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hr, hsr, hm',
            Option.bind_eq_bind, Option.bind_some]
        · intro f hf
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
          exact holds_devKeep (w := { w with mem := m }) (hcp f hf) hk' rfl rfl rfl rfl rfl rfl rfl rfl rfl
            (fun _ _ h => h) (fun _ _ _ h => h)
            (hlive := fun _ _ h => h)
            (fun e => by
              subst e
              have h0 := hcp _ hf
              exact ⟨Device.syncRead_track hsr h0.1, Device.syncRead_seq hsr h0.2⟩)
        · cases hv; exact cuOk_desc Γ _
  · cases h

theorem nmodsOf_mem {S : TState} {k : Nat} (h : nmodsOf S = some k) : Fact.nmods k ∈ S := by
  unfold nmodsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

theorem nfuncsOf_mem {S : TState} {k : Nat} (h : nfuncsOf S = some k) : Fact.nfuncs k ∈ S := by
  unfold nfuncsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

theorem textOf_mem {S : TState} {a : UInt64} {t : String} (h : textOf S a = some t) :
    ∃ n sg, Fact.text a (some t) n sg ∈ S := by
  unfold textOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> try simp at hk
  rename_i a' s' n sg
  cases s' <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact ⟨_, _, hx⟩

theorem sigsOf_mem {S : TState} {a : UInt64} {sg : List (String × List Nat)} (h : sigsOf S a = some sg) :
    ∃ t n, Fact.text a t n sg ∈ S := by
  unfold sigsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact ⟨_, _, hx⟩

theorem mem_of_lookup {α β : Type} [BEq α] [LawfulBEq α] {l : List (α × β)} {a : α} {b : β} (h : l.lookup a = some b) :
    (a, b) ∈ l := by
  induction l with
  | nil => cases h
  | cons p l ih =>
      obtain ⟨k, v⟩ := p
      simp only [List.lookup] at h
      split at h
      · rename_i hk
        cases h
        have : a = k := by simpa using hk
        subst this; exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (ih h)

theorem modOf_mem {S : TState} {i : Nat} {t : List (String × List Nat)} (h : modOf S i = some t) :
    Fact.mod i t ∈ S := by
  unfold modOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact hx

theorem fnOf_mem {S : TState} {i : Nat} {sz : List Nat} (h : fnOf S i = some sz) : Fact.fn i sz ∈ S := by
  unfold fnOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact hx

theorem cellOf_mem {S : TState} {a v : UInt64} (h : cellOf S a = some v) : Fact.cell a v ∈ S := by
  unfold cellOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact hx

theorem not_nmods {y : Fact} (h : (match y with | .nmods _ => false | _ => true) = true) :
    ∀ n, y ≠ .nmods n := by
  intro n e; subst e; simp at h

theorem not_nfuncs {y : Fact} (h : (match y with | .nfuncs _ => false | _ => true) = true) :
    ∀ n, y ≠ .nfuncs n := by
  intro n e; subst e; simp at h

theorem getElem?_push_some {α : Type} {a : Array (Option α)} {i : Nat} {x : α} (b : Option α)
    (h : a[i]? = some (some x)) : (a.push b)[i]? = some (some x) := by
  have hi : i < a.size := by
    rcases Nat.lt_or_ge i a.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at h

theorem getElem?_push_pair {a : Array (Nat × String)} {i : Nat} {x : Nat × String} (b : Nat × String)
    (h : a[i]? = some x) : (a.push b)[i]? = some x := by
  have hi : i < a.size := by
    rcases Nat.lt_or_ge i a.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at h

/-- A module loaded keeps every fact but how many there are. -/
theorem holds_pushMod {x : Fact} {w : World} (t : String) (hx : ∀ n, x ≠ .nmods n) (h : x.holds w) :
    x.holds { w with dev := { w.dev with modules := w.dev.modules.push (some t) } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | nmods n => exact absurd rfl (hx n)
  | mod i u =>
      obtain ⟨s, hm, hs⟩ := h
      exact ⟨s, getElem?_push_some _ hm, hs⟩
  | fn i sz =>
      obtain ⟨mi, e, u, hf, hm, hs⟩ := h
      exact ⟨mi, e, u, hf, getElem?_push_some _ hm, hs⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- A function looked up keeps every fact but how many there are. -/
theorem holds_pushFn {x : Fact} {w : World} (f : Nat × String) (hx : ∀ n, x ≠ .nfuncs n) (h : x.holds w) :
    x.holds { w with dev := { w.dev with funcs := w.dev.funcs.push f } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | nfuncs n => exact absurd rfl (hx n)
  | fn i sz =>
      obtain ⟨mi, e, u, hf, hm, hs⟩ := h
      exact ⟨mi, e, u, getElem?_push_pair _ hf, hm, hs⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem paramSizes_hasEntry {ptx e : String} {sz : List Nat} (h : ptxParamSizes ptx e = some sz) :
    ptxHasEntry ptx e = true := by
  unfold ptxParamSizes at h
  unfold ptxHasEntry
  cases hp : afterFirst (".entry " ++ e ++ "(").toList ptx.toList with
  | none => rw [hp] at h; cases h
  | some _ => rfl

theorem row_memcpyDtoD {S S' T : TState} {w : World} {Γ : Env} {dst src n : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [dst, src, n] bits)
    (hT : T.holds w) (hv : cudaVal T .memcpyDtoD (bits.map some) = true)
    (h : cudaRow S .memcpyDtoD [dst, src, n] = some (S', av)) :
    RowOk .memcpyDtoD bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨⟨hbase, hlive⟩, hseq⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
    have hsq : (Fact.devSeq).holds w := hS _ (List.contains_iff_mem.mp hseq)
    match bits, hb with
    | [x, y, z], ⟨hx, hy, hz, _⟩ =>
        simp only [List.map_cons, List.map_nil, cudaVal, argAt, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.join_some, Bool.and_eq_true] at hv
        obtain ⟨did, doff, dbuf, hrd⟩ := allocFor_range hT hv.1
        obtain ⟨sid, soff, sbuf, hrs⟩ := allocFor_range hT hv.2
        obtain ⟨hlbd, hfitd⟩ := range_alloc hrd
        obtain ⟨-, hfits⟩ := range_alloc hrs
        have hknow := Contracts.readyIn_sound hS (Contracts.readyIn_seq hseq)
        obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp
          (ExtContracts.op_isSome_rw (hknow.ready sid false) (hknow.ready did true))
        let c := overwrite dbuf doff (sbuf.extract soff (soff + z.toNat))
        have hes : (sbuf.extract soff (soff + z.toNat)).size = z.toNat := by
          simp only [ByteArray.size_extract]; omega
        let d' : Dev := ({ w.dev with race := r }).put did (some c)
        have hsizes := put_sizes (d := { w.dev with race := r }) (id := did) (c := c)
          (by show (w.dev.bufs[did]?.join).map _ = _; rw [hlbd, Device.overwrite_size (by omega)]; rfl)
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hrs, hrd, hop, hlbd,
            Option.bind_eq_bind, Option.bind_some, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
          rfl
        · intro f hf
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf
          have hk2 := sizes_keep (d := w.dev) (d' := d') (put_size _ _ _) hsizes
          exact holds_devKeep (hS f hf) hk' rfl rfl rfl rfl rfl rfl (put_size _ _ _) rfl rfl hk2.1
            (fun j n _ => hk2.2 j n) (hlive := fun _ => LiveKept.of_sizes hk2.2)
            (fun e => by
              subst e
              exact ⟨Device.put_track (track_op (d' := { w.dev with race := r }) hsq.1 hop rfl rfl rfl) _ _,
                Device.DefaultOnly.op hsq.2 hop⟩)
        · cases hv; exact cuOk_desc Γ _
  · cases h

theorem row_moduleLoadData {S S' : TState} {w : World} {Γ : Env} {p img : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [p, img] bits) (h : cudaRow S .moduleLoadData [p, img] = some (S', av)) :
    RowOk .moduleLoadData bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i k sg hk ht
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨hbase, hlive⟩, hmem⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hcnt : w.dev.modules.size = k := hS _ (nmodsOf_mem hk)
      subst hcnt
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨a, ha, hta⟩ := Option.bind_eq_some_iff.mp ht
      obtain ⟨t0, n, htn⟩ := sigsOf_mem hta
      obtain ⟨-, t, hrd, -, hsg⟩ := (hS _ htn : (Fact.text a t0 n sg).holds w)
      match bits, hb with
      | [pa, x], ⟨hpa, hx, _⟩ =>
        obtain rfl := known_bits hx ha
        obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kModule w.dev.modules.size))
        obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
        refine ⟨hhas, some (ofInt .i32 cuSuccess),
          { w with mem := m, dev := { w.dev with modules := w.dev.modules.push (some t) } }, ?_, ?_,
          fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hrd, hm, Option.bind_eq_bind, Option.bind_some]
        · intro y hy
          simp only [List.mem_cons, List.mem_filter] at hy
          rcases hy with rfl | rfl | ⟨hy, hnb⟩
          · show (w.dev.modules.push _).size = w.dev.modules.size + 1; rw [Array.size_push]
          · exact ⟨t, by show (w.dev.modules.push _)[w.dev.modules.size]? = _; rw [Array.getElem?_push_size],
              hsg⟩
          · exact holds_pushMod (w := { w with mem := m }) _ (not_nmods hnb) (hst y hy)
        · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_moduleGetFunction {S S' : TState} {w : World} {Γ : Env} {p hm pn : AV} {bits : List UInt64}
    {av : AV} (hS : S.holds w) (hb : ArgBits Γ w [p, hm, pn] bits)
    (h : cudaRow S .moduleGetFunction [p, hm, pn] = some (S', av)) :
    RowOk .moduleGetFunction bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i k mi name hk hmi hname
    split at h
    · rename_i sizes hsz
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨⟨hbase, hlive⟩, hmem⟩ := hc
        cases h
        obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
        have hcnt : w.dev.funcs.size = k := hS _ (nfuncsOf_mem hk)
        subst hcnt
        have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
        obtain ⟨hv, hhv, hmi'⟩ := Option.bind_eq_some_iff.mp hmi
        obtain ⟨q, hq, hnq⟩ := Option.bind_eq_some_iff.mp hname
        obtain ⟨n, sg', htn⟩ := textOf_mem hnq
        obtain ⟨-, t', hrd, hnm, -⟩ := (hS _ htn : (Fact.text q (some name) n sg').holds w)
        obtain rfl := hnm name rfl
        obtain ⟨sg, hsg, hlk⟩ := Option.bind_eq_some_iff.mp hsz
        obtain ⟨t, hmod, hall⟩ := (hS _ (modOf_mem hsg) : (Fact.mod mi sg).holds w)
        have hps : ptxParamSizes t name = some sizes := hall (name, sizes) (mem_of_lookup hlk)
        have hent := paramSizes_hasEntry hps
        match bits, hb with
        | [pa, x, y], ⟨hpa, hx, hy, _⟩ =>
          obtain rfl := known_bits hx hhv
          obtain rfl := known_bits hy hq
          obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kFunc w.dev.funcs.size))
          obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
          refine ⟨hhas, some (ofInt .i32 cuSuccess),
            { w with mem := m, dev := { w.dev with funcs := w.dev.funcs.push (mi, name) } }, ?_, ?_,
            fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
          · rw [cudaDrv_step hcap]
            have hnl : (!w.dev.live) = false := by simp [hl]
            simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hmi', hmod, Option.join_some, hrd, hent,
              Bool.not_true, hm, Option.bind_eq_bind, Option.bind_some]
          · intro y hy
            simp only [List.mem_cons, List.mem_filter] at hy
            rcases hy with rfl | rfl | ⟨hy, hnb⟩
            · show (w.dev.funcs.push _).size = w.dev.funcs.size + 1; rw [Array.size_push]
            · exact ⟨mi, name, t, by show (w.dev.funcs.push _)[w.dev.funcs.size]? = _; rw [Array.getElem?_push_size],
                hmod, hps⟩
            · exact holds_pushFn (w := { w with mem := m }) _ (not_nfuncs hnb) (hst y hy)
          · cases hv; exact cuOk_desc Γ _
      · cases h
    · cases h
  · cases h

/-- A launch writes only buffers: modules, functions, the driver's own
    state and how many allocations there are, it leaves. -/
theorem runLaunch_rest {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} {l : Launch}
    {ids : List Nat} (h : d.runLaunch k l ids = some d') :
    d'.modules = d.modules ∧ d'.funcs = d.funcs ∧ d'.drvInit = d.drvInit ∧ d'.retained = d.retained ∧
      d'.bufs.size = d.bufs.size ∧ d'.blas = d.blas := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    suffices ∀ (ps : List (Nat × ByteArray)) (d : Dev),
        let d' := ps.foldl (fun d (id, o) => d.put id (some o)) d
        d'.modules = d.modules ∧ d'.funcs = d.funcs ∧ d'.drvInit = d.drvInit ∧ d'.retained = d.retained ∧
          d'.bufs.size = d.bufs.size ∧ d'.blas = d.blas from this _ d
    intro ps
    induction ps with
    | nil => intro d; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
    | cons x ps ih =>
        intro d
        obtain ⟨a, b, c, e, f, g⟩ := ih (d.put x.1 (some x.2))
        exact ⟨a, b, c, e, f.trans (put_size _ _ _), g⟩

/-- An operation on the device with no capture in progress: the tracker's
    step, then the operation. -/
theorem devOp_launch {d d' : Dev} {w : World} {p : Nat} {l : Launch} {ids : List Nat} (hc : d.capture = none)
    (h : d.devOp w p (.launch l ids) [] ids = some d') :
    ∃ r, d.race.op p [] ids = some r ∧ ({ d with race := r }).runLaunch w.kernel l ids = some d' := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨r, hr, h⟩

/-- An operation by the stream a capture began on: recorded in the capture,
    checked against its own order. -/
theorem devOp_captured {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat} {c : Capture}
    (hc : d.capture = some c) (ho : c.origin = p) (h : d.devOp w p op rs ws = some d') :
    ∃ r, c.race.op p rs ws = some r ∧ d' = { d with capture := some { c with race := r, ops := c.ops.push op } } := by
  unfold Dev.devOp at h
  rw [hc] at h
  have hcp : d.capturing p = true := by simp [Dev.capturing, hc, ho]
  simp only [hcp, if_true] at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨r, hr, (Option.some.inj h).symm⟩

/-- Every allocation a launch binds is live: it is one a parameter points into. -/
theorem launch_ids_live {d : Dev} {args : List UInt64} (j : Nat)
    (hj : j ∈ (args.filterMap fun a => (d.range? a 0).map (·.1)).eraseDups) :
    ((d.bufs[j]?).join).isSome = true := by
  obtain ⟨a, -, ha⟩ := List.mem_filterMap.mp (ExtContracts.mem_of_mem_eraseDups hj)
  obtain ⟨⟨j', off, b⟩, hr, hj'⟩ := Option.map_eq_some_iff.mp ha
  cases hj'
  rw [(ExtContracts.range?_live hr).1]; rfl

/-- What a capture in progress needs of the world: the driver loaded. -/
theorem capBase_world {S : TState} {w : World} (hS : S.holds w) (h : capBase S = true) :
    w.has .cuda = true := by
  simp only [capBase, Bool.and_eq_true] at h
  have h1 : (Fact.part (.has .cuda) true).holds w := holds_of_mem hS h.1
  exact h1

theorem cudaDrv_under {f : CudaFn} {bits : List UInt64} {w : World} (hf : f.underCapture = true) :
    cudaDrv f bits w = cudaStep f bits w := by
  simp [cudaDrv, hf]

/-- **A step that changes only what a capture in progress has recorded**
    keeps every fact but the capture's own, given the tracker's order where
    the typestate had it. -/
theorem holds_capKeep {x : Fact} {w : World} {c c' : Capture} (hcap : w.dev.capture = some c)
    (h : x.holds w) (hx : ∀ q j r evs, x ≠ .cap q j r evs)
    (hseq : x = .devSeq → Device.TrackOk { w.dev with capture := some c' } ∧ Device.DefaultOnly w.dev.race) :
    x.holds { w with dev := { w.dev with capture := some c' } } := by
  cases x with
  | part p b =>
      cases p
      case capturing => show (some c').isSome = b; have h' : w.dev.capture.isSome = b := h; rw [hcap] at h'; exact h'
      all_goals exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact hseq rfl
  | cap q j r evs => exact absurd rfl (hx q j r evs)
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem streams_getD {d : Dev} {l : List Bool} (h : d.streams.toList = l) (i : Nat) :
    d.streams.getD i false = l.getD i false := by
  rw [← h, Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD, Array.getElem?_toList]

/-- The party the typestate names for a stream handle is the one the driver
    finds. -/
theorem drvParty_sound {S : TState} {w : World} {h : UInt64} {q : Nat} (hS : S.holds w)
    (hd : drvParty S h = some q) : w.dev.streamParty? h = some q := by
  unfold drvParty at hd
  split at hd
  · rename_i h0; cases hd; simp [Dev.streamParty?, h0]
  · rename_i h0
    split at hd
    · rename_i id l hid hl
      split at hd
      · rename_i hg
        cases hd
        have hg' : w.dev.streams.getD id false = true := by
          rw [streams_getD (Contracts.streamsOf_holds hS hl)]; exact hg
        rw [Array.getD_eq_getD_getElem?] at hg'
        simp [Dev.streamParty?, h0, hid, hg']
      · cases hd
    · cases hd

theorem get?_nat (d : Dev) (j : Nat) : d.get? (Int.ofNat j) = (d.bufs[j]?).join := by
  unfold Dev.get?
  rw [if_neg (by simp)]
  rfl

theorem argBits_cons {Γ : Env} {w : World} {a : AV} {as : List AV} {bits : List UInt64}
    (h : ArgBits Γ w (a :: as) bits) :
    ∃ x xs, bits = x :: xs ∧ (∃ t, a.Desc Γ w (.sc t x)) ∧ ArgBits Γ w as xs := by
  cases bits with
  | nil => exact h.elim
  | cons x xs => exact ⟨x, xs, rfl, h.1, h.2⟩

theorem argBits_nil {Γ : Env} {w : World} {bits : List UInt64} (h : ArgBits Γ w [] bits) : bits = [] := by
  cases bits with
  | nil => rfl
  | cons _ _ => exact h.elim

theorem capOf_mem {S : TState} {q : Nat} {j : List Nat} {r : Race} {evs : List (Nat × Clock)}
    (h : capOf S = some (q, j, r, evs)) : Fact.cap q j r evs ∈ S := by
  unfold capOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl, rfl, rfl⟩ := hk
  exact hx

/-- The allocation `allocAt?` places a pointer in is the one the device does. -/
theorem allocAt?_spec {S : TState} {w : World} (hS : S.holds w) {x : UInt64} {o : Option Nat}
    (h : allocAt? S x = some o) : (w.dev.range? x 0).map (·.1) = o := by
  unfold allocAt? at h
  unfold Dev.range?
  cases hp : devPtrOf x with
  | none => rw [hp] at h; cases h; rfl
  | some p =>
      obtain ⟨id, off⟩ := p
      rw [hp] at h
      simp only at h
      split at h
      · rename_i n hn
        obtain ⟨y, hy, hk⟩ := List.exists_of_findSome?_eq_some hn
        cases y <;> simp at hk
        rename_i i m
        obtain ⟨rfl, rfl⟩ := hk
        obtain ⟨b, hb, hbs⟩ := (hS _ hy : (Fact.alloc i m).holds w)
        cases h
        simp only [Option.bind_eq_bind, Option.bind_some, hb, Nat.add_zero, hbs]
        split <;> rfl
      · split at h
        · rename_i k hk
          have hsz : w.dev.bufs.size = k := hS _ (nbufsOf_mem hk)
          split at h
          · rename_i hle
            cases h
            have : w.dev.bufs[id]? = none := Array.getElem?_eq_none (by omega)
            simp [this]
          · cases h
        · cases h

/-- **The allocations `launchIds` names are the ones the launch binds.** -/
theorem launchIds_spec {S : TState} {w : World} (hS : S.holds w) {P : UInt64} {sizes : List Nat}
    {ids : List Nat} {args : List UInt64} (h : launchIds S P sizes = some ids)
    (ha : kernelArgs w.mem P sizes = some args) :
    (args.filterMap fun a => (w.dev.range? a 0).map (·.1)).eraseDups = ids := by
  unfold launchIds at h
  unfold kernelArgs at ha
  obtain ⟨os, hos, hid⟩ := Option.bind_eq_some_iff.mp h
  cases hid
  suffices hgen : ∀ (l : List (Nat × Nat)) (os : List (Option Nat)) (as : List UInt64),
      l.mapM (fun (sz, i) => do
        let a ← cellOf S (P + UInt64.ofNat (8 * i))
        if sz == 8 then allocAt? S (← cellOf S a) else none) = some os →
      l.mapM (fun (sz, i) => do
        let at_ ← w.mem.load (P + UInt64.ofNat (8 * i)) 8
        w.mem.load at_ sz) = some as →
      as.filterMap (fun a => (w.dev.range? a 0).map (·.1)) = os.filterMap id by
    rw [hgen _ os args hos ha]
  intro l
  induction l with
  | nil =>
      intro os as h1 h2
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h1 h2
      subst h1 h2; rfl
  | cons hd tl ih =>
      intro os as h1 h2
      obtain ⟨sz, i⟩ := hd
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h1 h2
      obtain ⟨o, ho, os', hos', rfl⟩ := h1
      obtain ⟨v, hv, as', has', rfl⟩ := h2
      obtain ⟨a, ha, ho⟩ := ho
      split at ho
      · rename_i hsz
        have hsz : sz = 8 := by simpa using hsz
        subst hsz
        cases hy : cellOf S a with
        | none => rw [hy] at ho; cases ho
        | some y =>
        rw [hy] at ho
        have hl1 : w.mem.load (P + UInt64.ofNat (8 * i)) 8 = some a := hS _ (cellOf_mem ha)
        obtain ⟨a', ha', hv⟩ := hv
        rw [hl1, Option.some.injEq] at ha'
        subst ha'
        have hl2 : w.mem.load a 8 = some y := hS _ (cellOf_mem hy)
        rw [hl2, Option.some.injEq] at hv
        subst hv
        have hsp := allocAt?_spec hS ho
        rw [List.filterMap_cons, List.filterMap_cons, ih os' as' hos' has', hsp]
        cases o <;> rfl
      · cases ho

/-- A party a capture's fact says it records is one the device says is
    capturing. -/
theorem capturing_of {d : Dev} {c : Capture} {q : Nat} {j : List Nat} {p : Nat} (hcap : d.capture = some c)
    (hq : c.origin = q) (hj : c.joined = j) (h : capIn q j p = true) : d.capturing p = true := by
  simp only [Dev.capturing, hcap, hq, hj]; simpa only [capIn] using h

set_option maxHeartbeats 3200000 in
theorem row_launchKernel {S S' T : TState} {w : World} {Γ : Env}
    {hf gx gy gz bx by_ bz sh hs pr ex : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [hf, gx, gy, gz, bx, by_, bz, sh, hs, pr, ex] bits)
    (hT : T.holds w) (hv : cudaVal T .launchKernel (bits.map some) = true)
    (h : cudaRow S .launchKernel [hf, gx, gy, gz, bx, by_, bz, sh, hs, pr, ex] = some (S', av)) :
    RowOk .launchKernel bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i fi P hfi hP
    split at h
    · rename_i sizes q hsz hq
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        obtain ⟨⟨⟨⟨⟨hbase, hlive⟩, hrq⟩, horc⟩, hex0⟩, hargs⟩ := hc
        cases h
        obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
        have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
        have hks : KeepsSize w.kernel := (hS _ (List.contains_iff_mem.mp horc) : (Fact.oracles).holds w).1
        obtain ⟨mi, e, ptx, hfun, hmod, hps⟩ := (hS _ (fnOf_mem hsz) : (Fact.fn fi sizes).holds w)
        obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
        obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
        obtain ⟨x2, b, rfl, h2, hb⟩ := argBits_cons hb
        obtain ⟨x3, b, rfl, h3, hb⟩ := argBits_cons hb
        obtain ⟨x4, b, rfl, h4, hb⟩ := argBits_cons hb
        obtain ⟨x5, b, rfl, h5, hb⟩ := argBits_cons hb
        obtain ⟨x6, b, rfl, h6, hb⟩ := argBits_cons hb
        obtain ⟨x7, b, rfl, h7, hb⟩ := argBits_cons hb
        obtain ⟨x8, b, rfl, h8, hb⟩ := argBits_cons hb
        obtain ⟨x9, b, rfl, h9, hb⟩ := argBits_cons hb
        obtain ⟨x10, b, rfl, h10, hb⟩ := argBits_cons hb
        obtain rfl := argBits_nil hb
        obtain ⟨y, hy, hfi'⟩ := Option.bind_eq_some_iff.mp hfi
        obtain rfl := known_bits h0 hy
        obtain rfl := known_bits h9 hP
        obtain ⟨y8, hy8, hq'⟩ := Option.bind_eq_some_iff.mp hq
        obtain rfl := known_bits h8 hy8
        have hsp := drvParty_sound hS hq'
        obtain rfl := known_bits h10 hex0
        have hdz := hv
        simp only [List.map_cons, List.map_nil, cudaVal, argAt, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.join_some, dimOk, List.range, List.range.loop, List.all_cons,
          List.all_nil] at hdz
        have hka : (kernelArgs w.mem x9 sizes).isSome = true := by
          unfold kernelArgs
          refine Contracts.mapM_isSome_of fun q hq => ?_
          obtain ⟨sz, i⟩ := q
          have hq' := List.all_eq_true.mp hargs _ hq
          simp only at hq'
          split at hq'
          · rename_i a ha
            have hcell : w.mem.load (x9 + UInt64.ofNat (8 * i)) 8 = some a := hS _ (cellOf_mem ha)
            simp only [hcell, Option.bind_eq_bind, Option.bind_some]
            exact Contracts.load_of_state hS hq'
          · cases hq'
        obtain ⟨args, hargs'⟩ := Option.isSome_iff_exists.mp hka
        have hknow := Contracts.readyIn_sound hS hrq
        have hlo : ExtContracts.LaunchOk w x0 x8 x9 :=
          ⟨fi, mi, e, ptx, q, sizes, args, hfi', hfun, hmod, hsp, hps,
            hargs', fun _ _ id _ _ _ => by simp only [Dev.raceFor, hcap]; exact hknow.ready id true⟩
        have hpre : ExtContracts.cudaStepPre .launchKernel [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 0] w :=
          ⟨hl, rfl, hks, hlo⟩
        obtain ⟨⟨r, w'⟩, hstep⟩ := Option.isSome_iff_exists.mp (ExtContracts.cudaStepPre_safe _ _ _ hpre)
        have hstep0 := hstep
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, bne_iff_ne, ne_eq] at hdz
        have z1 : (x1 &&& 4294967295 == 0) = false := by simpa using hdz.1
        have z2 : (x2 &&& 4294967295 == 0) = false := by simpa using hdz.2.1
        have z3 : (x3 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.1
        have z4 : (x4 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.1
        have z5 : (x5 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.2.1
        have z6 : (x6 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.2.2
        simp only [cudaStep, hnl, Bool.false_or, bne_self_eq_false, Bool.false_eq_true, if_false, hfi', hfun,
          hmod, Option.join_some, hsp, List.map_cons, List.map_nil,
          List.any_cons, List.any_nil, z1, z2, z3, z4, z5,
          z6, Bool.or_false, Bool.or_self, hps, hargs', Option.bind_eq_bind, Option.bind_some,
          Option.some_bind] at hstep
        unfold devOnly at hstep
        obtain ⟨⟨r0, d'⟩, hk, he⟩ := Option.map_eq_some_iff.mp hstep
        obtain ⟨d'', hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
        simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
        obtain ⟨rfl, rfl⟩ := hcu
        simp only [Prod.mk.injEq] at he
        obtain ⟨rfl, rfl⟩ := he
        obtain ⟨rr, hop, hrl⟩ := devOp_launch hcap hdo
        obtain ⟨hlv, hst, hcp, hev, hrace, hsz'⟩ := DevSpec.runLaunch_sizes hks hrl
        obtain ⟨hmo, hfu, hdi, hre, hn, hbl⟩ := runLaunch_rest hrl
        refine ⟨hhas, some (ofInt .i32 cuSuccess),
          { w with dev := d'', mem := w.mem.retire (d''.race.clock hostParty) }, ?_, ?_, fun Γ v hv => ?_, ?_⟩
        · rw [cudaDrv_step hcap]; exact hstep0
        · refine Contracts.TState.holds_app (fun f hf0 => ?_)
            (Contracts.TState.holds_if fun _ => Contracts.TState.holds_one ?_)
          rotate_left
          · show Device.KnowsAll d''.race q
            rw [hrace]; exact hknow.op_self hop
          obtain ⟨hf1, hdq⟩ := List.mem_filter.mp hf0
          obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf1
          rw [DevSpec.retire_nil hbusy]
          have hkg := Device.devOp_keepsGraphs hdo
          refine holds_devKeep (hS f hf) hk' hst hev hlv hcp hdi hre hn hmo hfu
            (fun i n hb => Device.devOp_bufs hdo hb) (fun j n _ ⟨c, hc, hcs⟩ => ?_)
            (hlive := fun _ => LiveKept.of_sizes fun j n ⟨c, hc, hcs⟩ => by
              have := hsz' (Int.ofNat j)
              rw [get?_nat, get?_nat, hc] at this
              obtain ⟨c', hc', hcs'⟩ := Option.map_eq_some_iff.mp this
              exact ⟨c', hc', hcs'.trans hcs⟩)
            (hgr := hkg.1) (hex := hkg.2)
            (fun e => by
              subst e
              have hq0 : q = defaultParty := by simpa using hdq
              subst hq0
              have hsq : (Fact.devSeq).holds w := hS _ hf
              exact ⟨Device.devOp_track hdo hsq.1, Device.devOp_seq hdo hsq.2⟩)
          have := hsz' (Int.ofNat j)
          rw [get?_nat, get?_nat] at this
          rw [hc] at this
          obtain ⟨c', hc', hcs'⟩ := Option.map_eq_some_iff.mp this
          exact ⟨c', hc', hcs'.trans hcs⟩
        · cases hv; exact cuOk_desc Γ _
        · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
      · split at h
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          obtain ⟨⟨⟨⟨hbase, hlive⟩, horc⟩, hex0⟩, hargs⟩ := hc
          split at h
          · rename_i cq jn r evs ids hco hids
            split at h
            · rename_i hin
              split at h
              · rename_i r' hop
                cases h
                have hhas := capBase_world hS hbase
                have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
                obtain ⟨mi, e, ptx, hfun, hmod, hps⟩ := (hS _ (fnOf_mem hsz) : (Fact.fn fi sizes).holds w)
                obtain ⟨c, hcap, hcq, hcj, hcr, hol, hev⟩ := (hS _ (capOf_mem hco) : (Fact.cap cq jn r evs).holds w)
                obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
                obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
                obtain ⟨x2, b, rfl, h2, hb⟩ := argBits_cons hb
                obtain ⟨x3, b, rfl, h3, hb⟩ := argBits_cons hb
                obtain ⟨x4, b, rfl, h4, hb⟩ := argBits_cons hb
                obtain ⟨x5, b, rfl, h5, hb⟩ := argBits_cons hb
                obtain ⟨x6, b, rfl, h6, hb⟩ := argBits_cons hb
                obtain ⟨x7, b, rfl, h7, hb⟩ := argBits_cons hb
                obtain ⟨x8, b, rfl, h8, hb⟩ := argBits_cons hb
                obtain ⟨x9, b, rfl, h9, hb⟩ := argBits_cons hb
                obtain ⟨x10, b, rfl, h10, hb⟩ := argBits_cons hb
                obtain rfl := argBits_nil hb
                obtain ⟨y, hy, hfi'⟩ := Option.bind_eq_some_iff.mp hfi
                obtain rfl := known_bits h0 hy
                obtain rfl := known_bits h9 hP
                obtain ⟨y8, hy8, hq'⟩ := Option.bind_eq_some_iff.mp hq
                obtain rfl := known_bits h8 hy8
                have hsp := drvParty_sound hS hq'
                obtain rfl := known_bits h10 hex0
                have hdz := hv
                simp only [List.map_cons, List.map_nil, cudaVal, argAt, List.getElem?_cons_zero, List.getElem?_cons_succ,
                  Option.join_some, dimOk, List.range, List.range.loop, List.all_cons, List.all_nil] at hdz
                have hka : (kernelArgs w.mem x9 sizes).isSome = true := by
                  unfold kernelArgs
                  refine Contracts.mapM_isSome_of fun q hq => ?_
                  obtain ⟨sz, i⟩ := q
                  have hq' := List.all_eq_true.mp hargs _ hq
                  simp only at hq'
                  split at hq'
                  · rename_i a ha
                    have hcell : w.mem.load (x9 + UInt64.ofNat (8 * i)) 8 = some a := hS _ (cellOf_mem ha)
                    simp only [hcell, Option.bind_eq_bind, Option.bind_some]
                    exact Contracts.load_of_state hS hq'
                  · cases hq'
                obtain ⟨args, hargs'⟩ := Option.isSome_iff_exists.mp hka
                have hids' := launchIds_spec hS hids hargs'
                have hcp : w.dev.capturing q = true := capturing_of hcap hcq hcj hin
                let l : Launch := ⟨ptx, e, ids, [x1 &&& 0xffffffff, x2 &&& 0xffffffff, x3 &&& 0xffffffff,
                  x4 &&& 0xffffffff, x5 &&& 0xffffffff, x6 &&& 0xffffffff], args⟩
                let d'' : Dev := { w.dev with capture := some { c with race := r', ops := c.ops.push (.launch l ids) } }
                have hdo : w.dev.devOp w q (.launch l ids) [] ids = some d'' := by
                  unfold Dev.devOp
                  simp only [hcap, hcp, if_true, hcr, hop, Option.bind_eq_bind, Option.bind_some]
                  rfl
                have hnl : (!w.dev.live) = false := by simp [hl]
                simp only [List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true, bne_iff_ne, ne_eq] at hdz
                have z1 : (x1 &&& 4294967295 == 0) = false := by simpa using hdz.1
                have z2 : (x2 &&& 4294967295 == 0) = false := by simpa using hdz.2.1
                have z3 : (x3 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.1
                have z4 : (x4 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.1
                have z5 : (x5 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.2.1
                have z6 : (x6 &&& 4294967295 == 0) = false := by simpa using hdz.2.2.2.2.2
                have hstep : cudaStep .launchKernel [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 0] w =
                    devOnly w (cuRes d'' cuSuccess) := by
                  simp only [cudaStep, hnl, Bool.false_or, bne_self_eq_false, Bool.false_eq_true, if_false, hfi', hfun,
                    hmod, Option.join_some, hsp, List.map_cons, List.map_nil,
                    List.any_cons, List.any_nil, z1, z2, z3, z4, z5,
                    z6, Bool.or_false, Bool.or_self, hps, hargs', Option.bind_eq_bind, Option.bind_some,
                    Option.some_bind]
                  rw [hids']
                  show devOnly w ((w.dev.devOp w q (.launch l ids) [] ids).bind fun x => cuRes x cuSuccess) = _
                  rw [hdo]; rfl
                have hdrv := hstep
                rw [← cudaDrv_under (f := .launchKernel) rfl] at hdrv
                refine ⟨hhas, _, _, hdrv, ?_, fun Γ v hv => ?_, ?_⟩
                · rw [DevSpec.retire_nil hbusy]
                  refine Contracts.TState.holds_cons ?_ (fun f hf => ?_)
                  · refine ⟨_, rfl, hcq, hcj, rfl, fun o ho => ?_, hev⟩
                    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at ho
                    rcases ho with ho | rfl
                    · exact hol o ho
                    · exact .inl ⟨_, _, rfl, fun j hj => by rw [← hids'] at hj; exact launch_ids_live j hj⟩
                  · obtain ⟨hf, hnc⟩ := List.mem_filter.mp hf
                    exact holds_capKeep hcap (hS f hf) (fun q' j' r'' e' he' => by subst he'; simp at hnc) (fun e => by
                      subst e
                      have hsq : (Fact.devSeq).holds w := hS _ hf
                      exact ⟨Device.devOp_track hdo hsq.1, hsq.2⟩)
                · cases hv; exact cuOk_desc Γ _
                · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
              · cases h
            · cases h
          · cases h
        · cases h
    · cases h
  · cases h

/-- **A step on the device's streams, events and tracker**: what reads the
    streams, events or tracker survives where the step leaves what it reads
    and only grows the clocks; everything else the step leaves alone. -/
theorem holds_devStep {x : Fact} {w : World} {d : Dev} (h : x.holds w)
    (hs : (∃ l, x = .streams l) → d.streams = w.dev.streams)
    (hn : (∃ n, x = .events n) → d.events.size = w.dev.events.size)
    (he : ∀ e : Nat, (x = .evLive e ∨ x = .evAll e) → d.event? e = w.dev.event? e)
    (hlw : d.race.lastW = w.dev.race.lastW) (hrd : d.race.reads = w.dev.race.reads)
    (hg : DevSpec.Grows w.dev.race d.race)
    (hl : d.live = w.dev.live) (hc : d.capture = w.dev.capture) (hb : d.bufs = w.dev.bufs)
    (hmo : d.modules = w.dev.modules) (hfu : d.funcs = w.dev.funcs) (hi : d.drvInit = w.dev.drvInit)
    (hr : d.retained = w.dev.retained)
    (hseq : x = .devSeq → Device.TrackOk d ∧ Device.DefaultOnly d.race)
    (hbl : d.blas = w.dev.blas := by first | rfl | assumption)
    (hgr : d.graphs = w.dev.graphs := by first | rfl | assumption)
    (hex : d.execs = w.dev.execs := by first | rfl | assumption)
    (hevs : d.events = w.dev.events ∨ w.dev.capture = none := by
      first | exact .inl rfl | exact .inr (by assumption)) :
    x.holds { w with dev := d } := by
  have hlk : LiveKept w.dev d := fun j h => by rw [hb]; exact h
  by_cases hd : x.isDev = true
  · exact Fact.holds_step (w' := { w with dev := d }) hd h hs hn he hlw hrd hg
  cases x with
  | part p b =>
      cases p <;> first
        | (show d.live = b; rw [hl]; exact h)
        | (show d.capture.isSome = b; rw [hc]; exact h)
        | exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact hseq rfl
  | devBuf a n => show Device.BufIs _ _ d; unfold Device.BufIs Dev.get? at *; rw [hb]; exact h
  | drvInit => show d.drvInit = true; rw [hi]; exact h
  | retained n => show d.retained = n; rw [hr]; exact h
  | nbufs n => show d.bufs.size = n; rw [hb]; exact h
  | alloc j n => show ∃ c, (d.bufs[j]?).join = some c ∧ _; rw [hb]; exact h
  | nmods n => show d.modules.size = n; rw [hmo]; exact h
  | nfuncs n => show d.funcs.size = n; rw [hfu]; exact h
  | mod i t => show ∃ s, d.modules[i]? = _ ∧ _; rw [hmo]; exact h
  | fn i sz => show ∃ mi e t, d.funcs[i]? = some (mi, e) ∧ d.modules[mi]? = some (some t) ∧ _; rw [hmo, hfu]; exact h
  | nblas n => show d.blas.size = n; rw [hbl]; exact h
  | blasOn i t => show d.blas[i]? = _; rw [hbl]; exact h
  | ngraphs n => show d.graphs.size = n; rw [hgr]; exact h
  | nexecs n => show d.execs.size = n; rw [hex]; exact h
  | graph i =>
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show d.graphs[i]? = _; rw [hgr]; exact hg, hl.mono hlk⟩
  | exec i =>
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show d.execs[i]? = _; rw [hex]; exact hg, hl.mono hlk⟩
  | cap q j r evs =>
      obtain ⟨c, hcc, ho, hj, hr, hl, hev⟩ := h
      rcases hevs with hevs | hevs
      · exact ⟨c, by show d.capture = _; rw [hc]; exact hcc, ho, hj, hr, hl.mono hlk,
          fun e he' => by unfold Dev.event? at hev ⊢; rw [hevs]; exact hev e he'⟩
      · rw [hevs] at hcc; cases hcc
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => simp [Fact.isDev] at hd
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem event?_nat' (d : Dev) (e : Nat) : d.event? e = (d.events[e]?).join := by
  unfold Dev.event?; rw [if_neg (by simp)]; rfl

theorem evOf_spec {S : TState} {w : World} {h : UInt64} {e : Nat} (hS : S.holds w) (hd : evOf S h = some e) :
    handleOf kEvent h = some e ∧ ∃ ev, (w.dev.events[e]?).join = some ev := by
  unfold evOf at hd
  cases he : handleOf kEvent h with
  | none => rw [he] at hd; cases hd
  | some e' =>
      rw [he] at hd
      simp only at hd
      by_cases hc : S.contains (.evLive e') = true
      · rw [if_pos hc] at hd
        obtain rfl : e' = e := Option.some.inj hd
        have hl : (w.dev.event? e').isSome = true := hS _ (List.contains_iff_mem.mp hc)
        rw [event?_nat'] at hl
        obtain ⟨ev, hev⟩ := Option.isSome_iff_exists.mp hl
        exact ⟨rfl, ev, hev⟩
      · rw [if_neg hc] at hd; cases hd

theorem row_streamSynchronize {S S' : TState} {w : World} {Γ : Env} {hh : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [hh] bits)
    (h : cudaRow S .streamSynchronize [hh] = some (S', av)) : RowOk .streamSynchronize bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i q hq
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h0 hy
      have hsp := drvParty_sound hS hq'
      let d' : Dev := { w.dev with race := w.dev.race.sync q }
      have hnc : w.dev.capturing q = false := by simp [Dev.capturing, hcap]
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hsp, hnc, Option.bind_eq_bind, Option.bind_some,
          devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
        rfl
      · refine Contracts.TState.holds_app (fun f hf => ?_) (Contracts.TState.holds_if fun hrd => Contracts.TState.holds_one ?_)
        · exact holds_devStep (hS f hf) (fun _ => rfl) (fun _ => rfl) (fun _ _ => rfl) rfl rfl
            (DevSpec.joinInto_grows _ _ _) rfl rfl rfl rfl rfl rfl rfl
            (fun e => by
              subst e
              have h0 := hS _ hf
              exact ⟨Device.track_race h0.1 (fun _ hc => .learnParty hostParty q hc) d' rfl rfl rfl,
                h0.2.sync q⟩)
        · exact (Contracts.readyIn_sound hS hrd).sync
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- A new event, unrecorded, leaves the tracker one the race proof covers. -/
theorem track_pushEvent {d d' : Dev} (ht : Device.TrackOk d) (hr : d'.race = d.race) (hc : d'.capture = d.capture)
    (he : d'.events = d.events.push (some none)) : Device.TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  refine ⟨⟨evs, hr ▸ c1, Device.evOk_push c2 he⟩, ?_⟩
  rw [hc]
  cases hcap : d.capture with
  | none => rw [hcap] at c3; exact Device.evOk_push c3 he
  | some c => rw [hcap] at c3; obtain ⟨evs', h1, h2⟩ := c3; exact ⟨evs', h1, Device.evOk_push h2 he⟩

/-- An event destroyed leaves the tracker one the race proof covers. -/
theorem track_dropEvent {d d' : Dev} (ht : Device.TrackOk d) (hr : d'.race = d.race) (hc : d'.capture = d.capture)
    (e : Nat) (he : d'.events = d.events.set! e none) : Device.TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  refine ⟨⟨evs, hr ▸ c1, Device.evOk_set c2 e none (fun _ h => nomatch h) he⟩, ?_⟩
  rw [hc]
  cases hcap : d.capture with
  | none => rw [hcap] at c3; exact Device.evOk_set c3 e none (fun _ h => nomatch h) he
  | some c =>
      rw [hcap] at c3; obtain ⟨evs', h1, h2⟩ := c3
      exact ⟨evs', h1, Device.evOk_set h2 e none (fun _ h => nomatch h) he⟩

/-- Advancing a party's clock adds no access. -/
theorem defaultOnly_issue {r : Race} (h : Device.DefaultOnly r) (p : Nat) : Device.DefaultOnly (r.issue p).1 :=
  fun b e he => h b e (by simpa [Race.issue, Device.wAcc_setClock, Device.rAcc_setClock] using he)

theorem evLive_lt {w : World} {e : Nat} (h : (w.dev.event? e).isSome = true) : e < w.dev.events.size := by
  rw [event?_nat'] at h
  rcases Nat.lt_or_ge e w.dev.events.size with hl | hl
  · exact hl
  · rw [Array.getElem?_eq_none hl] at h; cases h

theorem row_streamCreate {S S' : TState} {w : World} {Γ : Env} {p fl : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [p, fl] bits) (h : cudaRow S .streamCreate [p, fl] = some (S', av)) :
    RowOk .streamCreate bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i l hlst
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hfl⟩, hmem⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      have hl' := Contracts.streamsOf_holds hS hlst
      have hsz : w.dev.streams.size = l.length := by rw [← hl', Array.length_toList]
      obtain ⟨pa, b, rfl, hpa, hb⟩ := argBits_cons hb
      obtain ⟨x, b, rfl, hx, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain rfl := known_bits hx hfl
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kStream w.dev.streams.size))
      obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
      let c := (w.dev.race.clock defaultParty).join (w.dev.race.clock hostParty)
      let r' := w.dev.race.joinInto (streamParty w.dev.streams.size) c
      let d' : Dev := { w.dev with streams := w.dev.streams.push true, race := r' }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with mem := m, dev := d' }, ?_, ?_, fun Γ v hv => ?_,
        hk1.trans (CuKept.dev _ _)⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_or, show ((1 : UInt64) &&& 0xffffffff != 1) = false by decide,
          Bool.false_eq_true, if_false, hm, Option.bind_eq_bind, Option.bind_some]
        rfl
      · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_app (fun f hf => ?_)
          (Contracts.TState.holds_if fun hrd => Contracts.TState.holds_one ?_))
        · show (w.dev.streams.push true).toList = _; rw [Array.toList_push, hl']
        · obtain ⟨hf, hns⟩ := List.mem_filter.mp hf
          rw [← hsz] at hf
          exact holds_devStep (w := { w with mem := m }) (hst f hf) (fun ⟨l', e⟩ => by subst e; simp at hns)
            (fun _ => rfl) (fun _ _ => rfl) rfl rfl (DevSpec.joinInto_grows _ _ _) rfl rfl rfl rfl rfl rfl rfl
            (fun e => by
              subst e
              have h0 := hst _ hf
              exact ⟨Device.track_race h0.1 (fun _ hc => .learnTwo _ defaultParty hostParty hc) d' rfl rfl rfl,
                h0.2.joinInto _ _⟩)
        · show Device.KnowsAll (w.dev.race.joinInto _ c) _
          rw [hsz]
          exact (Contracts.readyIn_sound hS hrd).learn _ (DevSpec.ClockLe.join_left _ _)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_streamDestroy {S S' : TState} {w : World} {Γ : Env} {hh : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [hh] bits) (h : cudaRow S .streamDestroy [hh] = some (S', av)) :
    RowOk .streamDestroy bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i x l hx hlst
    split at h
    · rename_i id hid
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
        obtain ⟨⟨⟨hbase, hlive⟩, hx0⟩, hg⟩ := hc
        cases h
        obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
        have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
        have hl' := Contracts.streamsOf_holds hS hlst
        obtain ⟨y, b, rfl, hy, hb⟩ := argBits_cons hb
        obtain rfl := argBits_nil hb
        obtain rfl := known_bits hy hx
        have hg' : w.dev.streams.getD id false = true := by rw [streams_getD hl']; exact hg
        let d' : Dev := { w.dev with streams := w.dev.streams.set! id false }
        refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · rw [cudaDrv_step hcap]
          have hnl : (!w.dev.live) = false := by simp [hl]
          have hy0 : (y == 0) = false := by simpa using hx0
          simp only [cudaStep, hnl, hy0, Bool.false_or, Bool.false_eq_true, if_false, hid, hg', Bool.not_true,
            Option.bind_eq_bind, Option.bind_some]
          rfl
        · refine Contracts.TState.holds_cons ?_ (fun f hf => ?_)
          · show (w.dev.streams.set! id false).toList = _; simp [Array.set!, hl']
          · obtain ⟨hf, hns⟩ := List.mem_filter.mp hf
            exact holds_devStep (hS f hf) (fun ⟨l', e⟩ => by subst e; simp at hns) (fun _ => rfl) (fun _ _ => rfl)
              rfl rfl (DevSpec.Grows.refl _) rfl rfl rfl rfl rfl rfl rfl
              (fun e => by subst e; have h0 := hS _ hf; exact ⟨h0.1.congr rfl rfl rfl, h0.2⟩)
        · cases hv; exact cuOk_desc Γ _
      · cases h
    · cases h
  · cases h

theorem row_eventCreate {S S' : TState} {w : World} {Γ : Env} {p fl : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [p, fl] bits) (h : cudaRow S .eventCreate [p, fl] = some (S', av)) :
    RowOk .eventCreate bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i n hn
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hfl⟩, hmem⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      have hn' := Contracts.eventsOf_holds hS hn
      subst hn'
      obtain ⟨pa, b, rfl, hpa, hb⟩ := argBits_cons hb
      obtain ⟨x, b, rfl, hx, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain rfl := known_bits hx hfl
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kEvent w.dev.events.size))
      obtain ⟨hst, hk1⟩ := afterStore_world (ao := p.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
      let d' : Dev := { w.dev with events := w.dev.events.push (some none) }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with mem := m, dev := d' }, ?_, ?_, fun Γ v hv => ?_,
        hk1.trans (CuKept.dev _ _)⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_or, show ((0 : UInt64) &&& 0xffffffff != 0) = false by decide,
          Bool.false_eq_true, if_false, hm, Option.bind_eq_bind, Option.bind_some]
        rfl
      · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_cons ?_ (fun f hf => ?_))
        · show (w.dev.events.push _).size = _; rw [Array.size_push]
        · show (d'.event? _).isSome = true
          rw [event?_nat']; show ((w.dev.events.push _)[w.dev.events.size]?).join.isSome = true
          rw [Array.getElem?_push_size]; rfl
        · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          have h0 := hst f hf
          refine holds_devStep (w := { w with mem := m }) h0 (fun _ => rfl) (fun ⟨n, e⟩ => by subst e; simp at hne)
            (fun e he => ?_) rfl rfl (DevSpec.Grows.refl _) rfl rfl rfl rfl rfl rfl rfl
            (fun e => by subst e; exact ⟨track_pushEvent h0.1 rfl rfl rfl, h0.2⟩)
          have hlt : e < w.dev.events.size := by
            rcases he with rfl | rfl
            · exact evLive_lt (w := { w with mem := m }) h0
            · obtain ⟨clk, hc, -⟩ := h0; exact evLive_lt (w := { w with mem := m }) (by rw [hc]; rfl)
          rw [event?_nat', event?_nat']
          show ((w.dev.events.push _)[e]?).join = _
          rw [Array.getElem?_push_lt hlt, Array.getElem?_eq_getElem hlt]
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- Setting one event's slot leaves every other event as it was. -/
theorem event?_set_ne {d d' : Dev} {e e' : Nat} {v : Option (Option (Clock × Bool))}
    (h : d'.events = d.events.set! e v) (hne : e' ≠ e) : d'.event? e' = d.event? e' := by
  rw [event?_nat', event?_nat', h, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (Ne.symm hne)]

theorem event_lt {w : World} {e : Nat} {ev : Option (Clock × Bool)} (h : (w.dev.events[e]?).join = some ev) :
    e < w.dev.events.size := by
  rcases Nat.lt_or_ge e w.dev.events.size with hl | hl
  · exact hl
  · rw [Array.getElem?_eq_none hl] at h; cases h

/-- **An event recorded inside a capture** keeps every fact but the
    capture's and the event's covering, given the tracker's order where the
    typestate had it. -/
theorem holds_capRecord {x : Fact} {w : World} {c c' : Capture} {e : Nat} {v : Option (Clock × Bool)}
    (hcap : w.dev.capture = some c) (hlt : e < w.dev.events.size) (hv : v = v)
    (h : x.holds w) (hx : ∀ q j r evs, x ≠ .cap q j r evs) (hxe : x ≠ .evAll e)
    (hseq : x = .devSeq → Device.TrackOk { w.dev with capture := some c', events := w.dev.events.set! e (some v) }) :
    x.holds { w with dev := { w.dev with capture := some c', events := w.dev.events.set! e (some v) } } := by
  have hne : ∀ e', e' ≠ e → Dev.event? { w.dev with capture := some c', events := w.dev.events.set! e (some v) } e' =
      w.dev.event? e' := fun e' h => event?_set_ne rfl h
  cases x with
  | part p b =>
      cases p
      case capturing => show (some c').isSome = b; have h' : w.dev.capture.isSome = b := h; rw [hcap] at h'; exact h'
      all_goals exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨hseq rfl, h.2⟩
  | events n => show (w.dev.events.set! e _).size = n; rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]; exact h
  | evLive e' =>
      by_cases he : e' = e
      · subst he
        show (Dev.event? _ e').isSome = true
        rw [event?_nat']
        show ((w.dev.events.set! e' _)[e']?).join.isSome = true
        rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_self_of_lt hlt]; rfl
      · show (Dev.event? _ e').isSome = true; rw [hne e' he]; exact h
  | evAll e' =>
      have he : e' ≠ e := fun h => hxe (by rw [h])
      obtain ⟨clk, hc, hcv⟩ := h
      exact ⟨clk, by rw [hne e' he]; exact hc, hcv⟩
  | cap q j r evs => exact absurd rfl (hx q j r evs)
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- A stream that waits on a point inside the capture joins it, and the
    capture's tracker stays one the proof covers. -/
theorem track_wait_inner {d : Dev} (ht : Device.TrackOk d) {c : Capture} (hcap : d.capture = some c)
    (p : Nat) {e : Nat} {clk : Clock} (hev : d.event? e = some (some (clk, true))) (d' : Dev)
    (h1 : d'.race = d.race) (h3 : d'.events = d.events) {j : List Nat}
    (h2 : d'.capture = some { c with race := (c.race.issue p).1.joinInto p clk, joined := j }) :
    Device.TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  rw [hcap] at c3
  obtain ⟨evs', c4, c5⟩ := c3
  have hi : Device.CReached (c.race.issue p).1 evs' := Device.CReached.op p [] [] c4 (Device.Race.op_nil c.race p)
  refine ⟨⟨evs, h1 ▸ c1, fun i k hi' => c2 i k (h3 ▸ hi')⟩, ?_⟩
  rw [h2]
  refine ⟨evs', ?_, fun i k hi' => c5 i k (h3 ▸ hi')⟩
  rcases Device.event_clock c5 hev with rfl | hin
  · exact .learnNothing p hi
  · exact .learnEvent p clk hi hin

theorem row_eventRecord {S S' : TState} {w : World} {Γ : Env} {hv hh : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [hv, hh] bits)
    (h : cudaRow S .eventRecord [hv, hh] = some (S', av)) : RowOk .eventRecord bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i e q he hq
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, he'⟩ := Option.bind_eq_some_iff.mp he
      obtain rfl := known_bits h0 hy
      obtain ⟨y', hy', hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h1 hy'
      have hsp := drvParty_sound hS hq'
      obtain ⟨hhe, ev, hev⟩ := evOf_spec hS he'
      have hlt := event_lt hev
      let ic := w.dev.race.issue q
      let d' : Dev := { w.dev with race := ic.1, events := w.dev.events.set! e (some (some (ic.2, false))), capture := none }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, Dev.eventOf?, hhe, hev, hsp, Option.bind_eq_bind,
          Option.bind_some, Dev.recordAt, hcap, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
        rfl
      · refine Contracts.TState.holds_app (fun f hf => ?_)
          (Contracts.TState.holds_if fun hrd => Contracts.TState.holds_one ?_)
        · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          have hne' : f ≠ .evAll e := by simpa using hne
          have h0 := hS f hf
          by_cases hfl : f = .evLive e
          · subst hfl
            show (d'.event? e).isSome = true
            rw [event?_nat']
            show ((w.dev.events.set! e _)[e]?).join.isSome = true
            rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_self_of_lt hlt]; rfl
          · exact holds_devStep (d := d') h0 (fun _ => rfl) (fun _ => Array.size_setIfInBounds)
              (fun e' hx' => event?_set_ne rfl (by rintro rfl; rcases hx' with rfl | rfl <;> contradiction))
              rfl rfl (DevSpec.setClock_issued_grows _ _) rfl hcap.symm rfl rfl rfl rfl rfl
              (fun hx => by
                subst hx
                have h1 : Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race := h0
                exact ⟨Device.track_record_outer h1.1 (Or.inr trivial) q e d' rfl hcap.symm rfl,
                  defaultOnly_issue h1.2 q⟩)
        · refine ⟨ic.2, ?_, Device.Covers.issue (Contracts.readyIn_sound hS hrd)⟩
          rw [event?_nat']
          show ((w.dev.events.set! e _)[e]?).join = _
          rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_self_of_lt hlt]; rfl
      · cases hv; exact cuOk_desc Γ _
    · split at h
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨hbase, hlive⟩ := hc
        split at h
        · rename_i cq jn r evs hco
          split at h
          · rename_i hin
            cases h
            have hhas := capBase_world hS hbase
            have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
            obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
            obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
            obtain rfl := argBits_nil hb
            obtain ⟨y, hy, he'⟩ := Option.bind_eq_some_iff.mp he
            obtain rfl := known_bits h0 hy
            obtain ⟨y', hy', hq'⟩ := Option.bind_eq_some_iff.mp hq
            obtain rfl := known_bits h1 hy'
            have hsp := drvParty_sound hS hq'
            obtain ⟨hhe, ev, hev⟩ := evOf_spec hS he'
            have hlt := event_lt hev
            obtain ⟨c, hcap, rfl, rfl, rfl, hol, hevs⟩ := (hS _ (capOf_mem hco) : (Fact.cap cq jn r evs).holds w)
            have hcp : w.dev.capturing q = true := capturing_of hcap rfl rfl hin
            let ic := c.race.issue q
            let c1 : Capture := { c with race := ic.1 }
            let d' : Dev := { w.dev with capture := some c1, events := w.dev.events.set! e (some (some (ic.2, true))) }
            have hrec : w.dev.recordAt e q = d' := by
              unfold Dev.recordAt; simp only [hcap, hcp, if_true]; rfl
            refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d', mem := w.mem.retire (d'.race.clock hostParty) },
              ?_, ?_, fun Γ v hv => ?_, ?_⟩
            · rw [cudaDrv_under rfl]
              have hnl : (!w.dev.live) = false := by simp [hl]
              simp only [cudaStep, hnl, Bool.false_eq_true, if_false, Dev.eventOf?, hhe, hev, hsp,
                Option.bind_eq_bind, Option.bind_some, hrec]
              rfl
            · rw [DevSpec.retire_nil hbusy]
              refine Contracts.TState.holds_cons ?_ (fun f hf => ?_)
              · refine ⟨_, rfl, rfl, rfl, rfl, hol, fun x hx => ?_⟩
                simp only [List.mem_cons, List.mem_filter, bne_iff_ne, ne_eq] at hx
                rcases hx with rfl | ⟨hx, hne⟩
                · rw [event?_nat']
                  show ((w.dev.events.set! e _)[e]?).join = _
                  rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_self_of_lt hlt]; rfl
                · rw [event?_set_ne (d := w.dev) (d' := d') rfl hne]; exact hevs x hx
              · obtain ⟨hf1, hnc⟩ := List.mem_filter.mp hf
                obtain ⟨hf, hne⟩ := List.mem_filter.mp hf1
                exact holds_capRecord hcap hlt rfl (hS f hf)
                  (fun q' j' r' e' he' => by subst he'; simp at hnc) (by simpa using hne)
                  (fun hx => by
                    subst hx
                    have h1 : Device.TrackOk w.dev := (hS _ hf : (Fact.devSeq).holds w).1
                    exact Device.track_record_inner h1 hcap q e d' rfl rfl rfl)
            · cases hv; exact cuOk_desc Γ _
            · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
          · cases h
        · cases h
      · cases h
  · cases h

theorem row_streamWaitEvent {S S' : TState} {w : World} {Γ : Env} {hh hv fl : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [hh, hv, fl] bits)
    (h : cudaRow S .streamWaitEvent [hh, hv, fl] = some (S', av)) : RowOk .streamWaitEvent bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i q e hq he
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hfl⟩, hall⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain ⟨x2, b, rfl, h2, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h0 hy
      obtain ⟨y', hy', he'⟩ := Option.bind_eq_some_iff.mp he
      obtain rfl := known_bits h1 hy'
      obtain rfl := known_bits h2 hfl
      have hsp := drvParty_sound hS hq'
      obtain ⟨hhe, -⟩ := evOf_spec hS he'
      obtain ⟨clk, hclk, hcov⟩ : (Fact.evAll e).holds w := hS _ (List.contains_iff_mem.mp hall)
      have hev : (w.dev.events[e]?).join = some (some (clk, false)) := by rw [← event?_nat']; exact hclk
      have hnc : w.dev.capturing q = false := by simp [Dev.capturing, hcap]
      let d' : Dev := { w.dev with race := (w.dev.race.issue q).1.joinInto q clk }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_or, show ((0 : UInt64) &&& 0xffffffff != 0) = false by decide,
          Bool.false_eq_true, if_false, hsp, Dev.eventOf?, hhe, hev, Option.bind_eq_bind, Option.bind_some,
          Dev.waitOn, hnc, devOnly, cuRes, Option.map_some, DevSpec.retire_nil hbusy]
        rfl
      · refine Contracts.TState.holds_app (fun f hf => ?_) (Contracts.TState.holds_one ?_)
        · have h0 := hS f hf
          exact holds_devStep (d := d') h0 (fun _ => rfl) (fun _ => rfl) (fun _ _ => rfl) rfl rfl
            ((DevSpec.setClock_issued_grows _ _).trans (DevSpec.joinInto_grows _ _ _)) rfl rfl rfl rfl rfl rfl rfl
            (fun hx => by
              subst hx
              have h1 : Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race := h0
              obtain ⟨⟨evs, c1, c2⟩, c3⟩ := h1.1
              have hi : Device.CReached (w.dev.race.issue q).1 evs :=
                Device.CReached.op q [] [] c1 (Device.Race.op_nil _ q)
              refine ⟨⟨⟨evs, ?_, c2⟩, c3⟩, (defaultOnly_issue h1.2 q).joinInto _ _⟩
              rcases Device.event_clock c2 hclk with hz | hin
              · subst hz; exact .learnNothing q hi
              · exact .learnEvent q clk hi hin)
        · exact Device.Covers.wait hcov q
      · cases hv; exact cuOk_desc Γ _
    · split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hc
        obtain ⟨⟨⟨hbase, hlive⟩, hfl⟩, hnd⟩ := hc
        split at h
        · rename_i cq jn r evs hco
          split at h
          · rename_i clk hlk
            cases h
            have hhas := capBase_world hS hbase
            have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
            obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
            obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
            obtain ⟨x2, b, rfl, h2, hb⟩ := argBits_cons hb
            obtain rfl := argBits_nil hb
            obtain ⟨y, hy, hq'⟩ := Option.bind_eq_some_iff.mp hq
            obtain rfl := known_bits h0 hy
            obtain ⟨y', hy', he'⟩ := Option.bind_eq_some_iff.mp he
            obtain rfl := known_bits h1 hy'
            obtain rfl := known_bits h2 hfl
            have hsp := drvParty_sound hS hq'
            obtain ⟨hhe, -⟩ := evOf_spec hS he'
            obtain ⟨c, hcap, rfl, rfl, rfl, hol, hevs⟩ := (hS _ (capOf_mem hco) : (Fact.cap cq jn r evs).holds w)
            have hevc : w.dev.event? e = some (some (clk, true)) := hevs (e, clk) (mem_of_lookup hlk)
            have hev : (w.dev.events[e]?).join = some (some (clk, true)) := by rw [← event?_nat']; exact hevc
            let j' : List Nat := if capIn c.origin c.joined q then c.joined else q :: c.joined
            let c' : Capture := { c with race := (c.race.issue q).1.joinInto q clk, joined := j' }
            have hwait : w.dev.waitOn q (some (clk, true)) = some { w.dev with capture := some c' } := by
              unfold Dev.waitOn; simp only [hcap]; rfl
            refine ⟨hhas, some (ofInt .i32 cuSuccess),
              { w with dev := { w.dev with capture := some c' },
                       mem := w.mem.retire (w.dev.race.clock hostParty) }, ?_, ?_, fun Γ v hv => ?_, ?_⟩
            · rw [cudaDrv_under rfl]
              have hnl : (!w.dev.live) = false := by simp [hl]
              simp only [cudaStep, hnl, Bool.false_or, show ((0 : UInt64) &&& 0xffffffff != 0) = false by decide,
                Bool.false_eq_true, if_false, hsp, Dev.eventOf?, hhe, hev, Option.bind_eq_bind, Option.bind_some,
                hwait]
              rfl
            · rw [DevSpec.retire_nil hbusy]
              refine Contracts.TState.holds_cons ⟨c', rfl, rfl, rfl, rfl, hol, hevs⟩ (fun f hf => ?_)
              obtain ⟨hf, hnc⟩ := List.mem_filter.mp hf
              exact holds_capKeep hcap (hS f hf) (fun q' j' r' e' he' => by subst he'; simp at hnc) (fun hx => by
                subst hx
                have h1 : Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race := hS _ hf
                exact ⟨track_wait_inner h1.1 hcap q hevc _ rfl rfl rfl, h1.2⟩)
            · cases hv; exact cuOk_desc Γ _
            · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
          · cases h
        · cases h
      · cases h
  · cases h

theorem row_eventDestroy {S S' : TState} {w : World} {Γ : Env} {hv : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [hv] bits) (h : cudaRow S .eventDestroy [hv] = some (S', av)) :
    RowOk .eventDestroy bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i e he
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, he'⟩ := Option.bind_eq_some_iff.mp he
      obtain rfl := known_bits h0 hy
      obtain ⟨hhe, ev, hev⟩ := evOf_spec hS he'
      let d' : Dev := { w.dev with events := w.dev.events.set! e none }
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d' }, ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, Dev.eventOf?, hhe, hev, Option.bind_eq_bind,
          Option.bind_some]
        rfl
      · intro f hf
        obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hne
        have h0 := hS f hf
        exact holds_devStep (d := d') h0 (fun _ => rfl) (fun _ => Array.size_setIfInBounds)
          (fun e' hx' => event?_set_ne rfl (by rintro rfl; rcases hx' with rfl | rfl <;> simp_all))
          rfl rfl (DevSpec.Grows.refl _) rfl rfl rfl rfl rfl rfl rfl
          (fun hx => by
            subst hx
            have h1 : Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race := h0
            exact ⟨track_dropEvent h1.1 rfl rfl e rfl, h1.2⟩)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- What a store of a known value leaves holds of a store of any value. -/
theorem afterStore_forget {S : TState} {w : World} {ao : Option UInt64} {v : UInt64}
    (h : (Contracts.afterStore ao (some v) 8 S).holds w) : (Contracts.afterStore ao none 8 S).holds w := by
  intro x hx
  apply h
  cases ao with
  | none => exact hx
  | some a =>
      simp only [Contracts.afterStore] at hx ⊢
      split
      · exact List.mem_cons_of_mem _ hx
      · exact hx

theorem row_memGetInfo {S S' : TState} {w : World} {Γ : Env} {pf pt : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [pf, pt] bits) (h : cudaRow S .memGetInfo [pf, pt] = some (S', av)) :
    RowOk .memGetInfo bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true] at hc
    obtain ⟨⟨⟨hbase, hlive⟩, hm1⟩, hm2⟩ := hc
    cases h
    obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
    have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
    obtain ⟨x, b, rfl, hx, hb⟩ := argBits_cons hb
    obtain ⟨y, b, rfl, hy, hb⟩ := argBits_cons hb
    obtain rfl := argBits_nil hb
    obtain ⟨m1, h1⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hx hm1 w.memInfo.1)
    obtain ⟨hst1, hk1⟩ := afterStore_world (ao := pf.bits) hS h1 (fun z hz => (known_bits hx hz).symm)
    obtain ⟨r, off, hdec, hp, hs, hz, -⟩ := memN_spec hS hy hm2
    have hs' : off + 8 ≤ m1.sizes r := by rw [Contracts.store_sizes_eq h1]; exact hs
    have hz' : m1.frozen = false := by rw [Contracts.store_frozen h1]; exact hz
    obtain ⟨m2, h2⟩ := Option.isSome_iff_exists.mp (Contracts.store_isSome hdec hp hs' hz' w.memInfo.2)
    obtain ⟨hst2, hk2⟩ := afterStore_world (w := { w with mem := m1 }) (ao := pt.bits) (afterStore_forget hst1) h2
      (fun z hz => (known_bits hy hz).symm)
    refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with mem := m2 }, ?_, afterStore_forget hst2,
      fun Γ v hv => ?_, hk1.trans hk2⟩
    · rw [cudaDrv_step hcap]
      have hnl : (!w.dev.live) = false := by simp [hl]
      simp only [cudaStep, hnl, Bool.false_eq_true, if_false, h1, h2, Option.bind_eq_bind, Option.bind_some]
    · cases hv; exact cuOk_desc Γ _
  · cases h

theorem graphOf_spec {S : TState} {w : World} {h : UInt64} {g : Nat} (hS : S.holds w)
    (hd : graphOf S h = some g) : handleOf kGraph h = some g ∧ (Fact.graph g).holds w := by
  unfold graphOf at hd
  cases he : handleOf kGraph h with
  | none => rw [he] at hd; cases hd
  | some g' =>
      rw [he] at hd
      simp only at hd
      by_cases hc : S.contains (.graph g') = true
      · rw [if_pos hc] at hd
        obtain rfl : g' = g := Option.some.inj hd
        exact ⟨rfl, hS _ (List.contains_iff_mem.mp hc)⟩
      · rw [if_neg hc] at hd; cases hd

theorem execOf_spec {S : TState} {w : World} {h : UInt64} {e : Nat} (hS : S.holds w)
    (hd : execOf S h = some e) : handleOf kExec h = some e ∧ (Fact.exec e).holds w := by
  unfold execOf at hd
  cases he : handleOf kExec h with
  | none => rw [he] at hd; cases hd
  | some e' =>
      rw [he] at hd
      simp only at hd
      by_cases hc : S.contains (.exec e') = true
      · rw [if_pos hc] at hd
        obtain rfl : e' = e := Option.some.inj hd
        exact ⟨rfl, hS _ (List.contains_iff_mem.mp hc)⟩
      · rw [if_neg hc] at hd; cases hd

/-- Destroying graph `g` keeps every fact but its own. -/
theorem holds_setGraph {x : Fact} {w : World} {g : Nat} (h : x.holds w) (hx : x ≠ .graph g) :
    x.holds { w with dev := { w.dev with graphs := w.dev.graphs.set! g none } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | ngraphs n => show (w.dev.graphs.setIfInBounds g none).size = n; rw [Array.size_setIfInBounds]; exact h
  | graph i =>
      have hi : g ≠ i := fun e => hx (e ▸ rfl)
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show (w.dev.graphs.setIfInBounds g none)[i]? = _; rw [Array.getElem?_setIfInBounds_ne hi]; exact hg, hl⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

/-- Destroying instantiated graph `e` keeps every fact but its own. -/
theorem holds_setExec {x : Fact} {w : World} {e : Nat} (h : x.holds w) (hx : x ≠ .exec e) :
    x.holds { w with dev := { w.dev with execs := w.dev.execs.set! e none } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | nexecs n => show (w.dev.execs.setIfInBounds e none).size = n; rw [Array.size_setIfInBounds]; exact h
  | exec i =>
      have hi : e ≠ i := fun e' => hx (e' ▸ rfl)
      obtain ⟨ops, hg, hl⟩ := h
      exact ⟨ops, by show (w.dev.execs.setIfInBounds e none)[i]? = _; rw [Array.getElem?_setIfInBounds_ne hi]; exact hg, hl⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem row_graphDestroy {S S' : TState} {w : World} {Γ : Env} {hg : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [hg] bits) (h : cudaRow S .graphDestroy [hg] = some (S', av)) :
    RowOk .graphDestroy bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i g hgo
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hgo'⟩ := Option.bind_eq_some_iff.mp hgo
      obtain rfl := known_bits h0 hy
      obtain ⟨hhg, ops, hgr, -⟩ := graphOf_spec hS hgo'
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := { w.dev with graphs := w.dev.graphs.set! g none } },
        ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hhg, hgr, Option.join_some, Option.bind_eq_bind,
          Option.bind_some]
      · intro f hf
        obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
        exact holds_setGraph (hS f hf) (by simpa using hne)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_graphExecDestroy {S S' : TState} {w : World} {Γ : Env} {he : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [he] bits) (h : cudaRow S .graphExecDestroy [he] = some (S', av)) :
    RowOk .graphExecDestroy bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i e heo
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, heo'⟩ := Option.bind_eq_some_iff.mp heo
      obtain rfl := known_bits h0 hy
      obtain ⟨hhe, ops, hex, -⟩ := execOf_spec hS heo'
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := { w.dev with execs := w.dev.execs.set! e none } },
        ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hhe, hex, Option.join_some, Option.bind_eq_bind,
          Option.bind_some]
      · intro f hf
        obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
        exact holds_setExec (hS f hf) (by simpa using hne)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem row_graphUpload {S S' : TState} {w : World} {Γ : Env} {he hh : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [he, hh] bits) (h : cudaRow S .graphUpload [he, hh] = some (S', av)) :
    RowOk .graphUpload bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i e q heo hq
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbase, hlive⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, heo'⟩ := Option.bind_eq_some_iff.mp heo
      obtain rfl := known_bits h0 hy
      obtain ⟨y1, hy1, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h1 hy1
      obtain ⟨hhe, ops, hex, -⟩ := execOf_spec hS heo'
      have hsp := drvParty_sound hS hq'
      refine ⟨hhas, some (ofInt .i32 cuSuccess), w, ?_, hS, fun Γ v hv => ?_, CuKept.dev w w.dev⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hhe, hex, hsp, Option.join_some,
          Option.bind_eq_bind, Option.bind_some]
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- **Beginning a capture** keeps every fact but that none is in progress. -/
theorem holds_capStart {x : Fact} {w : World} {c : Capture} (hcap : w.dev.capture = none) (h : x.holds w)
    (hx : x ≠ .part .capturing false) (hc : c.race = {}) :
    x.holds { w with dev := { w.dev with capture := some c } } := by
  cases x with
  | part p b =>
      cases p
      case capturing =>
        have h' : w.dev.capture.isSome = b := h
        rw [hcap] at h'
        subst h'
        exact absurd rfl hx
      all_goals exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq =>
      obtain ⟨⟨c1, c3⟩, h2⟩ := h
      refine ⟨⟨c1, ?_⟩, h2⟩
      rw [hcap] at c3
      show ∃ evs, Device.CReached c.race evs ∧ Device.EvOk evs true _
      rw [hc]
      exact ⟨[], .init, c3⟩
  | cap q j r evs =>
      obtain ⟨c', hc', -⟩ := h
      rw [hcap] at hc'
      cases hc'
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem row_beginCapture {S S' : TState} {w : World} {Γ : Env} {hh mo : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [hh, mo] bits)
    (h : cudaRow S .beginCapture [hh, mo] = some (S', av)) :
    RowOk .beginCapture bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i q m hq hm
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hmode⟩, hnd⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h0 hy
      obtain rfl := known_bits h1 hm
      have hsp := drvParty_sound hS hq'
      let c : Capture := ⟨q, [], {}, #[]⟩
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := { w.dev with capture := some c } },
        ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        have hm2 : (x1 &&& 0xffffffff != 2) = false := by simp [hmode]
        have hd2 : (q == defaultParty) = false := by simp [hnd]
        simp only [cudaStep, hnl, hm2, hcap, Option.isSome_none, Bool.or_false, Bool.false_or,
          Bool.false_eq_true, if_false, hsp, hd2, Option.bind_eq_bind, Option.bind_some, devOnly, cuRes,
          Option.map_some, DevSpec.retire_nil hbusy]
        rfl
      · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_cons rfl fun f hf => ?_)
        · exact ⟨c, rfl, rfl, rfl, rfl, fun op ho => by simp [c] at ho, fun e he => by simp at he⟩
        · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          exact holds_capStart hcap (hS f hf) (by simpa using hne) rfl
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

theorem nexecsOf_mem {S : TState} {k : Nat} (h : nexecsOf S = some k) : Fact.nexecs k ∈ S := by
  unfold nexecsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

theorem ngraphsOf_mem {S : TState} {k : Nat} (h : ngraphsOf S = some k) : Fact.ngraphs k ∈ S := by
  unfold ngraphsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

/-- A new instantiated graph keeps every fact but how many there are. -/
theorem holds_pushExec {x : Fact} {w : World} (o : Option (Array DevOp)) (hx : ∀ n, x ≠ .nexecs n)
    (h : x.holds w) : x.holds { w with dev := { w.dev with execs := w.dev.execs.push o } } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | nexecs n => exact absurd rfl (hx n)
  | exec i =>
      obtain ⟨ops, hg, hl⟩ := h
      have hi : i < w.dev.execs.size := by
        rcases Nat.lt_or_ge i w.dev.execs.size with hi | hi
        · exact hi
        · rw [Array.getElem?_eq_none hi] at hg; cases hg
      refine ⟨ops, ?_, hl⟩
      show (w.dev.execs.push o)[i]? = _
      rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at hg
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem row_graphInstantiate {S S' : TState} {w : World} {Γ : Env} {pe hg fl : AV} {bits : List UInt64}
    {av : AV} (hS : S.holds w) (hb : ArgBits Γ w [pe, hg, fl] bits)
    (h : cudaRow S .graphInstantiate [pe, hg, fl] = some (S', av)) : RowOk .graphInstantiate bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i g n hgo hn
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hfl⟩, hmem⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨pa, b, rfl, hpa, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain ⟨x2, b, rfl, h2, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hgo'⟩ := Option.bind_eq_some_iff.mp hgo
      obtain rfl := known_bits h1 hy
      obtain rfl := known_bits h2 hfl
      obtain ⟨hhg, ops, hgr, hol⟩ := graphOf_spec hS hgo'
      have hsz : w.dev.execs.size = n := hS _ (nexecsOf_mem hn)
      subst hsz
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kExec w.dev.execs.size))
      obtain ⟨hst, hk1⟩ := afterStore_world (ao := pe.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
      refine ⟨hhas, some (ofInt .i32 cuSuccess),
        { w with mem := m, dev := { w.dev with execs := w.dev.execs.push (some ops) } },
        ?_, ?_, fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, bne_self_eq_false, hhg, hgr, Option.join_some, hm,
          Option.bind_eq_bind, Option.bind_some, Bool.or_false]
      · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_cons ?_ fun f hf => ?_)
        · refine ⟨ops, ?_, hol⟩
          show (w.dev.execs.push _)[w.dev.execs.size]? = _
          rw [Array.getElem?_push_size]
        · show (w.dev.execs.push _).size = _
          rw [Array.size_push]
        · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          exact holds_pushExec (w := { w with mem := m }) _ (fun k e => by subst e; simp at hne) (hst f hf)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- **Ending a capture** and keeping what it recorded as a new graph keeps
    every fact but how many graphs there are and those about the capture. -/
theorem holds_endCap {x : Fact} {w : World} (o : Option (Array DevOp)) (h : x.holds w)
    (hx1 : ∀ n, x ≠ .ngraphs n) (hx2 : ∀ q j r evs, x ≠ .cap q j r evs) (hx3 : ∀ b, x ≠ .part .capturing b) :
    x.holds { w with dev := { w.dev.endCapture with graphs := w.dev.graphs.push o } } := by
  cases x with
  | part p b =>
      cases p
      case capturing => exact absurd rfl (hx3 b)
      all_goals exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | devSeq =>
      obtain ⟨⟨⟨evs, c1, c2⟩, _⟩, h2⟩ := h
      exact ⟨⟨⟨evs, c1, fun i clk hi => Device.evOk_endCapture_false c2 i clk hi⟩,
        fun i clk hi => Device.evOk_endCapture_true i clk hi⟩, h2⟩
  | events n =>
      show (w.dev.events.map _).size = n
      rw [Array.size_map]; exact h
  | evLive e =>
      have h' : ((w.dev.events[e]?).join).isSome = true := by rw [← event?_nat']; exact h
      show (Dev.event? _ e).isSome = true
      rw [event?_nat']
      show ((w.dev.endCapture.events[e]?).join).isSome = true
      simp only [Dev.endCapture, Array.getElem?_map]
      cases hx : w.dev.events[e]? with
      | none => rw [hx] at h'; cases h'
      | some y =>
          rw [hx] at h'
          simp only [Option.map_some, Option.join_some] at h' ⊢
          split <;> simp_all
  | evAll e =>
      obtain ⟨clk, hc, hcv⟩ := h
      refine ⟨clk, ?_, hcv⟩
      rw [event?_nat'] at hc ⊢
      show (w.dev.endCapture.events[e]?).join = _
      simp only [Dev.endCapture, Array.getElem?_map]
      cases hx : w.dev.events[e]? with
      | none => rw [hx] at hc; cases hc
      | some y =>
          rw [hx] at hc
          simp only [Option.join_some] at hc
          subst hc
          simp
  | ngraphs n => exact absurd rfl (hx1 n)
  | cap q j r evs => exact absurd rfl (hx2 q j r evs)
  | graph i =>
      obtain ⟨ops, hg, hl⟩ := h
      have hi : i < w.dev.graphs.size := by
        rcases Nat.lt_or_ge i w.dev.graphs.size with hi | hi
        · exact hi
        · rw [Array.getElem?_eq_none hi] at hg; cases hg
      refine ⟨ops, ?_, hl⟩
      show (w.dev.graphs.push o)[i]? = _
      rw [Array.getElem?_push_lt hi]; rwa [Array.getElem?_eq_getElem hi] at hg
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem row_endCapture {S S' : TState} {w : World} {Γ : Env} {hh pg : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hb : ArgBits Γ w [hh, pg] bits) (h : cudaRow S .endCapture [hh, pg] = some (S', av)) :
    RowOk .endCapture bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i q n cq jn r evs hq hn hco
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨⟨⟨hbase, hlive⟩, hqc⟩, hrej⟩, hmem⟩ := hc
      subst hqc
      cases h
      have hhas := capBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      obtain ⟨c, hcap, hco', hcj, hcr, hol, -⟩ := (hS _ (capOf_mem hco) : (Fact.cap q jn r evs).holds w)
      subst hcj hcr
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨pa, b, rfl, hpa, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h0 hy
      have hsp := drvParty_sound hS hq'
      have hsz : w.dev.graphs.size = n := hS _ (ngraphsOf_mem hn)
      subst hsz
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (memN_slot8 hS hpa hmem (drvHandle kGraph w.dev.graphs.size))
      obtain ⟨hst, hk1⟩ := afterStore_world (ao := pg.bits) hS hm (fun y hy => (known_bits hpa hy).symm)
      have hrj : c.rejoined = true := by simp only [Capture.rejoined] at hrej ⊢; rw [hco']; exact hrej
      refine ⟨hhas, some (ofInt .i32 cuSuccess),
        { w with mem := m, dev := { w.dev.endCapture with graphs := w.dev.graphs.push (some c.ops) } },
        ?_, ?_, fun Γ v hv => ?_, hk1.trans (CuKept.dev _ _)⟩
      · rw [cudaDrv_under rfl]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hsp, hcap, hco', bne_self_eq_false, hrj,
          Bool.not_true, hm, Option.bind_eq_bind, Option.bind_some]
      · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_cons ?_
          (Contracts.TState.holds_cons rfl fun f hf => ?_))
        · refine ⟨c.ops, ?_, hol⟩
          show (w.dev.graphs.push _)[w.dev.graphs.size]? = _
          rw [Array.getElem?_push_size]
        · show (w.dev.graphs.push _).size = _
          rw [Array.size_push]
        · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          exact holds_endCap (w := { w with mem := m }) _ (hst f hf) (fun k e => by subst e; simp at hne)
            (fun _ _ _ _ e => by subst e; simp at hne) (fun k e => by subst e; simp at hne)
      · cases hv; exact cuOk_desc Γ _
    · cases h
  · cases h

/-- What a run of a graph's launches leaves: the device as it was but its
    allocations' contents, each the size it was. -/
structure BufsOnly (d d' : Dev) : Prop where
  live : d'.live = d.live
  streams : d'.streams = d.streams
  capture : d'.capture = d.capture
  events : d'.events = d.events
  race : d'.race = d.race
  modules : d'.modules = d.modules
  funcs : d'.funcs = d.funcs
  drvInit : d'.drvInit = d.drvInit
  retained : d'.retained = d.retained
  size : d'.bufs.size = d.bufs.size
  blas : d'.blas = d.blas
  graphs : d'.graphs = d.graphs
  execs : d'.execs = d.execs
  sizes : ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size

theorem BufsOnly.refl (d : Dev) : BufsOnly d d :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ => rfl⟩

theorem BufsOnly.trans {a b c : Dev} (h1 : BufsOnly a b) (h2 : BufsOnly b c) : BufsOnly a c :=
  ⟨h2.live.trans h1.live, h2.streams.trans h1.streams, h2.capture.trans h1.capture,
   h2.events.trans h1.events, h2.race.trans h1.race, h2.modules.trans h1.modules, h2.funcs.trans h1.funcs,
   h2.drvInit.trans h1.drvInit, h2.retained.trans h1.retained, h2.size.trans h1.size, h2.blas.trans h1.blas,
   h2.graphs.trans h1.graphs, h2.execs.trans h1.execs, fun i => (h2.sizes i).trans (h1.sizes i)⟩

theorem runLaunch_bufsOnly {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} {l : Launch}
    {ids : List Nat} (hk : KeepsSize k) (h : d.runLaunch k l ids = some d') : BufsOnly d d' := by
  obtain ⟨hlv, hst, hcp, hev, hrace, hsz⟩ := DevSpec.runLaunch_sizes hk h
  obtain ⟨hmo, hfu, hdi, hre, hn, hbl⟩ := runLaunch_rest h
  obtain ⟨hg, hx⟩ := Device.runLaunch_keepsGraphs h
  exact ⟨hlv, hst, hcp, hev, hrace, hmo, hfu, hdi, hre, hn, hbl, hg, hx, hsz⟩

/-- A vendor routine leaves the device as it was but its output's contents,
    at the size it was. -/
theorem runVendor_bufsOnly {d d' : Dev} {v : VendorCall → List ByteArray → ByteArray} {c : VendorCall}
    {ins : List Nat} {out : Nat} (h : d.runVendor v c ins out = some d') : BufsOnly d d' := by
  unfold Dev.runVendor at h
  obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨old, hold, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  rename_i hc
  simp only [Option.some.injEq] at h
  subst h
  have hs : (v c bs).size = old.size := by simpa using hc
  have hlt : out < d.bufs.size := by
    rcases Nat.lt_or_ge out d.bufs.size with hl | hl
    · exact hl
    · simp [Dev.get?, Array.getElem?_eq_none hl] at hold
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, Array.size_setIfInBounds, rfl, rfl, rfl, fun i => ?_⟩
  unfold Dev.get? Dev.put
  split
  · rfl
  · rename_i hi
    by_cases he : i.toNat = out
    · rw [he, Array.getElem?_setIfInBounds_self_of_lt hlt]
      have hold' : (d.bufs[out]?).join = some old := by simpa [Dev.get?] using hold
      rw [hold']; simp [hs]
    · rw [Array.getElem?_setIfInBounds_ne (Ne.symm he)]

/-- Where every element of a list has an answer, `mapM` answers each. -/
theorem mapM_eq_map_getD {α β : Type} [Inhabited β] {g : α → Option β} :
    ∀ (l : List α), (∀ a ∈ l, (g a).isSome = true) → l.mapM g = some (l.map fun a => (g a).getD default)
  | [], _ => rfl
  | x :: xs, h => by
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (h x List.mem_cons_self)
      rw [List.mapM_cons, hb, mapM_eq_map_getD xs (fun a ha => h a (List.mem_cons_of_mem _ ha))]
      simp [hb]

/-- **A graph whose every node is a launch or a vendor routine over live
    allocations runs**, and leaves the device as it was but its allocations'
    contents. -/
theorem ops_run (w : World) (hks : KeepsSize w.kernel) (hvk : VendorKeeps w.vendor) :
    ∀ (ops : List DevOp) (d : Dev), (∀ op ∈ ops, OpLive d op) →
    ∃ d', ops.foldlM (fun d op => match op with
      | .launch l ids => d.runLaunch w.kernel l ids
      | .vendor c ins out => d.runVendor w.vendor c ins out) d = some d' ∧ BufsOnly d d'
  | [], d, _ => ⟨d, rfl, BufsOnly.refl d⟩
  | op :: ops, d, h => by
      have hrun : ∃ d1, (match op with
          | .launch l ids => d.runLaunch w.kernel l ids
          | .vendor c ins out => d.runVendor w.vendor c ins out) = some d1 ∧ BufsOnly d d1 := by
        rcases h _ List.mem_cons_self with ⟨l, ids, rfl, hl⟩ | ⟨c, ins, out, rfl, hlast, hl⟩
        · obtain ⟨d1, h1⟩ := Option.isSome_iff_exists.mp (ExtContracts.runLaunch_isSome_nat hks l hl)
          exact ⟨d1, h1, runLaunch_bufsOnly hks h1⟩
        · have hg : ∀ j ∈ ins, (d.get? (Int.ofNat j)).isSome = true := fun j hj => hl j hj
          have hm := mapM_eq_map_getD (g := fun j => d.get? (Int.ofNat j)) ins hg
          have hout : ∃ o, d.get? (Int.ofNat out) = some o :=
            Option.isSome_iff_exists.mp (hg out (List.mem_of_getLast? hlast))
          obtain ⟨o, ho⟩ := hout
          have hl' : (ins.map fun a => (d.get? (Int.ofNat a)).getD default).getLast? = some o := by
            rw [List.getLast?_map, hlast]
            show some ((d.get? (Int.ofNat out)).getD default) = some o
            rw [ho]; rfl
          obtain ⟨d1, h1⟩ := Option.isSome_iff_exists.mp (runVendor_isSome hvk hm ho hl')
          exact ⟨d1, h1, runVendor_bufsOnly h1⟩
      obtain ⟨d1, h1, hb1⟩ := hrun
      have hlive : ∀ op ∈ ops, OpLive d1 op := fun op hop =>
        OpsLive.mono (ops := #[op]) (fun o ho => by
            simp only [List.mem_singleton] at ho; subst ho
            exact h _ (List.mem_cons_of_mem _ hop))
          (fun j hj => by
            have := hb1.sizes (Int.ofNat j)
            rw [get?_nat, get?_nat] at this
            obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp hj
            rw [hc] at this
            obtain ⟨c', hc', -⟩ := Option.map_eq_some_iff.mp this
            rw [hc']; rfl) op (by simp)
      obtain ⟨d', h2, hb2⟩ := ops_run w hks hvk ops d1 hlive
      refine ⟨d', ?_, hb1.trans hb2⟩
      simp only [List.foldlM_cons, h1, Option.bind_eq_bind, Option.bind_some]
      exact h2

/-- A party that knows every access may issue any operation. -/
theorem op_isSome_all {r : Race} {p : Nat} {rs ws : List Nat} (h : Device.KnowsAll r p) :
    (r.op p rs ws).isSome = true := by
  rw [DevSpec.op_some (fun x _ => (h.ready x true).1) (fun x _ => (h.ready x true).2 rfl)]
  rfl

theorem row_graphLaunch {S S' : TState} {w : World} {Γ : Env} {he hh : AV} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w [he, hh] bits)
    (h : cudaRow S .graphLaunch [he, hh] = some (S', av)) : RowOk .graphLaunch bits w S' av := by
  simp only [cudaRow] at h
  split at h
  · rename_i e q heo hq
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨⟨hbase, hlive⟩, hrq⟩, horc⟩ := hc
      cases h
      obtain ⟨hhas, hcap⟩ := cudaBase_world hS hbase
      have hl : w.dev.live = true := hS _ (List.contains_iff_mem.mp hlive)
      have hks : KeepsSize w.kernel := (hS _ (List.contains_iff_mem.mp horc) : (Fact.oracles).holds w).1
      obtain ⟨x0, b, rfl, h0, hb⟩ := argBits_cons hb
      obtain ⟨x1, b, rfl, h1, hb⟩ := argBits_cons hb
      obtain rfl := argBits_nil hb
      obtain ⟨y, hy, heo'⟩ := Option.bind_eq_some_iff.mp heo
      obtain rfl := known_bits h0 hy
      obtain ⟨y1, hy1, hq'⟩ := Option.bind_eq_some_iff.mp hq
      obtain rfl := known_bits h1 hy1
      obtain ⟨hhe, ops, hex, hol⟩ := execOf_spec hS heo'
      have hsp := drvParty_sound hS hq'
      have hknow := Contracts.readyIn_sound hS hrq
      obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp
        (op_isSome_all (rs := (graphAccess ops).1) (ws := (graphAccess ops).2) hknow)
      have hvk : VendorKeeps w.vendor := (hS _ (List.contains_iff_mem.mp horc) : (Fact.oracles).holds w).2
      obtain ⟨d', hfold, hbo⟩ := ops_run w hks hvk ops.toList { w.dev with race := r } (fun op hop => hol op hop)
      have hrg : w.dev.runGraph w ops q = some d' := by
        unfold Dev.runGraph
        simp only [hop, Option.bind_eq_bind, Option.bind_some]
        rw [← Array.foldlM_toList]
        exact hfold
      refine ⟨hhas, some (ofInt .i32 cuSuccess), { w with dev := d', mem := w.mem.retire (d'.race.clock hostParty) },
        ?_, ?_, fun Γ v hv => ?_, ?_⟩
      · rw [cudaDrv_step hcap]
        have hnl : (!w.dev.live) = false := by simp [hl]
        simp only [cudaStep, hnl, Bool.false_eq_true, if_false, hhe, hex, Option.join_some, hsp, hrg,
          Option.bind_eq_bind, Option.bind_some, devOnly, cuRes, Option.map_some]
      · rw [DevSpec.retire_nil hbusy]
        refine Contracts.TState.holds_app (fun f hf0 => ?_)
          (Contracts.TState.holds_if fun _ => Contracts.TState.holds_one ?_)
        rotate_left
        · show Device.KnowsAll d'.race q
          rw [hbo.race]; exact hknow.op_self hop
        obtain ⟨hf1, hdq⟩ := List.mem_filter.mp hf0
        obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf1
        have hal : ∀ (j n : Nat), (∃ c : ByteArray, (w.dev.bufs[j]?).join = some c ∧ c.size = n) →
            ∃ c : ByteArray, (d'.bufs[j]?).join = some c ∧ c.size = n := fun j n ⟨c, hc, hcs⟩ => by
          have := hbo.sizes (Int.ofNat j)
          rw [get?_nat, get?_nat] at this
          rw [show ({ w.dev with race := r } : Dev).bufs = w.dev.bufs from rfl, hc] at this
          obtain ⟨c', hc', hcs'⟩ := Option.map_eq_some_iff.mp this
          exact ⟨c', hc', hcs'.trans hcs⟩
        exact holds_devKeep (hS f hf) hk' hbo.streams hbo.events hbo.live hbo.capture hbo.drvInit hbo.retained
          hbo.size hbo.modules hbo.funcs
          (fun i n hb => Device.BufIs.of_sizes hb (Nat.le_of_eq hbo.size.symm) (fun j => hbo.sizes j))
          (fun j n _ => hal j n)
          (fun e => by
            subst e
            have hq0 : q = defaultParty := by simpa using hdq
            subst hq0
            have hsq : (Fact.devSeq).holds w := hS _ hf
            exact ⟨track_op hsq.1 hop hbo.race hbo.capture hbo.events, hbo.race ▸ Device.DefaultOnly.op hsq.2 hop⟩)
          (hlive := fun _ => LiveKept.of_sizes hal) (hbl := hbo.blas) (hgr := hbo.graphs) (hex := hbo.execs)
      · cases hv; exact cuOk_desc Γ _
      · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
    · cases h
  · cases h

/-- **What `cudaRow` decides is what the driver does**: where it gives a
    typestate and an answer for a call, the driver answers the call on the
    bits the arguments carry, the new typestate holds of the world it leaves,
    the answer is the one the row gives, and the rest of the world is kept. -/
theorem cudaRow_sound {S S' T : TState} {w : World} {Γ : Env} {f : CudaFn} {as : List AV}
    {bits : List UInt64} {av : AV} (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w as bits)
    (hT : T.holds w) (hv : cudaVal T f (bits.map some) = true)
    (h : cudaRow S f as = some (S', av)) : RowOk f bits w S' av := by
  have h0 := h
  unfold cudaRow at h
  split at h
  · exact row_init hS hb h0
  · exact row_primaryCtxRetain hS hb h0
  · exact row_ctxSetCurrent hS hb h0
  · exact row_memAlloc hS hb h0
  · exact row_memcpyHtoD hS hbusy hb hT hv h0
  · exact row_memcpyDtoH hS hbusy hb hT hv h0
  · exact row_memsetD8 hS hbusy hb h0
  · exact row_ctxSynchronize hS hbusy hb h0
  · exact row_memFree hS hbusy hb h0
  · exact row_memcpyDtoD hS hbusy hb hT hv h0
  · exact row_memGetInfo hS hb h0
  · exact row_moduleLoadData hS hb h0
  · exact row_moduleGetFunction hS hb h0
  · exact row_launchKernel hS hbusy hb hT hv h0
  · exact row_streamCreate hS hb h0
  · exact row_streamSynchronize hS hbusy hb h0
  · exact row_streamDestroy hS hb h0
  · exact row_eventCreate hS hb h0
  · exact row_eventRecord hS hbusy hb h0
  · exact row_streamWaitEvent hS hbusy hb h0
  · exact row_eventDestroy hS hb h0
  · exact row_beginCapture hS hbusy hb h0
  · exact row_endCapture hS hb h0
  · exact row_graphInstantiate hS hb h0
  · exact row_graphUpload hS hb h0
  · exact row_graphLaunch hS hbusy hb h0
  · exact row_graphExecDestroy hS hb h0
  · exact row_graphDestroy hS hb h0
  · cases h


-- ---------------------------------------------------------------------------
-- What a driver call keeps of the device's allocations
-- ---------------------------------------------------------------------------

/-- Every live allocation of `d` is live in `d'`, of the same size. -/
def AllocsKept (d d' : Dev) : Prop :=
  ∀ (j : Nat) (b : ByteArray), (d.bufs[j]?).join = some b → ∃ b', (d'.bufs[j]?).join = some b' ∧ b'.size = b.size

theorem AllocsKept.of_sizes {d d' : Dev}
    (h : ∀ j : Nat, (d'.bufs[j]?.join).map ByteArray.size = (d.bufs[j]?.join).map ByteArray.size) :
    AllocsKept d d' := fun j b hb => by
  have := h j
  rw [hb] at this
  obtain ⟨b', hb', hs⟩ := Option.map_eq_some_iff.mp this
  exact ⟨b', hb', hs⟩

theorem AllocsKept.bufs {d d' : Dev} (h : d'.bufs = d.bufs) : AllocsKept d d' :=
  fun j b hb => ⟨b, by rw [h]; exact hb, rfl⟩

theorem devOnly_dev {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) : ∃ d, k = some (r, d) ∧ w'.dev = d := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hk, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  exact ⟨d, hk, rfl⟩

/-- A launch keeps every allocation, recorded by a capture or run where
    kernels keep the sizes of the buffers they write. -/
theorem devOp_launch_keepsAllocs {d d' : Dev} {w : World} {p : Nat} {l : Launch} {ids rs ws : List Nat}
    (hks : KeepsSize w.kernel) (h : d.devOp w p (.launch l ids) rs ws = some d') : AllocsKept d d' := by
  unfold Dev.devOp at h
  have run_ok : ∀ {d'}, Dev.devOp.run d w p (.launch l ids) rs ws = some d' → AllocsKept d d' := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨-, -, -, -, -, hsz⟩ := DevSpec.runLaunch_sizes hks h
    exact AllocsKept.of_sizes fun j => by
      have := hsz (Int.ofNat j)
      rw [get?_nat, get?_nat] at this
      exact this
  split at h
  · split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq] at h
      subst h
      exact AllocsKept.bufs rfl
    · exact run_ok h
  · exact run_ok h

/-- An allocation leaves every other as it was: the new one goes at the end. -/
theorem memAlloc_keepsAllocs {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : cudaStep .memAlloc bits w = some (r, w')) : AllocsKept w.dev w'.dev := by
  simp only [cudaStep] at h
  split at h
  · split at h
    · cases h
    · split at h
      · cases h; exact AllocsKept.bufs rfl
      · obtain ⟨m, -, h⟩ := Option.bind_eq_some_iff.mp h
        cases h
        intro j b hb
        refine ⟨b, ?_, rfl⟩
        have hj : j < w.dev.bufs.size := by
          rcases Nat.lt_or_ge j w.dev.bufs.size with hj | hj
          · exact hj
          · rw [Array.getElem?_eq_none hj] at hb; cases hb
        show (w.dev.bufs.push _)[j]?.join = some b
        rw [Array.getElem?_push_lt hj]
        rwa [Array.getElem?_eq_getElem hj] at hb
  · cases h

/-- **An allocation, a copy, a fill, a launch or a wait keeps every
    allocation** live at its size, where kernels keep the sizes of the
    buffers they write. -/
theorem cudaStep_keepsAllocs {f : CudaFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hf : f.keepsAllocs = true) (hks : KeepsSize w.kernel) (h : cudaStep f bits w = some (r, w')) :
    AllocsKept w.dev w'.dev := by
  by_cases hm : f = .memAlloc
  · subst hm; exact memAlloc_keepsAllocs h
  cases f <;> simp only [CudaFn.keepsAllocs, Bool.false_eq_true] at hf <;> (try exact absurd rfl hm) <;>
    revert h <;> simp only [cudaStep] <;> split <;> intro h
  all_goals try (cases h; done)
  -- ctxSynchronize
  · split at h
    · cases h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    cases hk
    exact AllocsKept.bufs rfl
  -- memcpyHtoD
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, hr, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨bs, hbs, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    obtain ⟨d1, hsw, hcu⟩ := Option.bind_eq_some_iff.mp hk
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
    obtain ⟨-, rfl⟩ := hcu
    simp only [Dev.syncWrite, Option.bind_eq_bind] at hsw
    obtain ⟨rr, -, hsw⟩ := Option.bind_eq_some_iff.mp hsw
    cases hsw
    obtain ⟨hlb, hfit⟩ := range_alloc hr
    have hsz := Sem.copyOut_size hbs
    exact AllocsKept.of_sizes (put_sizes (by rw [hlb, Device.overwrite_size (by omega)]; rfl))
  -- memcpyDtoH
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d', hsr, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, -, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    simp only [Dev.syncRead, Option.bind_eq_bind] at hsr
    obtain ⟨rr, -, hsr⟩ := Option.bind_eq_some_iff.mp hsr
    cases hsr
    exact AllocsKept.bufs rfl
  -- memcpyDtoD
  · split at h
    · cases h
    obtain ⟨⟨sid, soff, sb⟩, hrs, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨did, doff, db0⟩, hrd, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨rr, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    obtain ⟨db, hdb, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨-, rfl⟩ := hk
    obtain ⟨hlb, hfit⟩ := range_alloc hrd
    obtain ⟨-, hfits⟩ := range_alloc hrs
    change (w.dev.bufs[did]?).join = some db at hdb
    rw [hlb] at hdb
    cases hdb
    exact AllocsKept.of_sizes (put_sizes (d := { w.dev with race := rr })
      (by show (w.dev.bufs[did]?.join).map _ = _
          rw [hlb, Device.overwrite_size (by simp only [ByteArray.size_extract]; omega)]; rfl))
  -- memsetD8
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, hr, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨rr, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨-, rfl⟩ := hk
    obtain ⟨hlb, hfit⟩ := range_alloc hr
    exact AllocsKept.of_sizes (put_sizes (d := { w.dev with race := rr })
      (by show (w.dev.bufs[id]?.join).map _ = _
          rw [hlb, Device.overwrite_size (by
            have e : ∀ (k : Nat) (x : UInt8), (ByteArray.mk (Array.replicate k x)).size = k :=
              fun k x => Array.size_replicate
            rw [e]; omega)]; rfl))
  -- launchKernel
  · split at h
    · cases h
    obtain ⟨fi, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨mi, entry⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨ptx, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact AllocsKept.bufs rfl
    obtain ⟨sizes, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨args, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    obtain ⟨d1, hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
    obtain ⟨-, rfl⟩ := hcu
    exact devOp_launch_keepsAllocs hks hdo
  -- streamSynchronize
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · cases h
    obtain ⟨d, hk, hd⟩ := devOnly_dev h
    rw [hd]
    cases hk
    exact AllocsKept.bufs rfl

end AlgorithmLib.Check
