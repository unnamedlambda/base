module
public import AlgorithmLib.Check.WgpuSound
meta import AlgorithmLib.Check.WgpuSound
public import AlgorithmLib.Check.VendorRule
meta import AlgorithmLib.Check.VendorRule
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.BlasSound` — what `blasRow` decides is what cuBLAS does

**`blasRow_sound`**: where `blasRow` gives a typestate and an answer for a
cuBLAS call, cuBLAS answers that call on the bits the arguments carry, in a
world the new typestate describes, with what the row says it answers.
**`vendorRow_sound`** says the same of every vendor call, driver or cuBLAS.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState Part holds_of_mem)

/-- What a cuBLAS row leaves: cuBLAS's answer and world, the typestate the
    row computes holding there, the answer it says, and the rest kept. -/
def BlasOk (f : CublasFn) (bits : List UInt64) (w : World) (S' : TState) (av : AV) : Prop :=
  w.has .cublas = true ∧ ∃ r w', cublasCall f bits w = some (r, w') ∧ S'.holds w' ∧
    (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧ CuKept w w'

theorem blasBase_world {S : TState} {w : World} (hS : S.holds w) (h : blasBase S = true) :
    w.has .cublas = true ∧ w.has .cuda = true ∧ w.dev.capture = none ∧ w.dev.live = true := by
  simp only [blasBase, Bool.and_eq_true] at h
  have h1 : (Fact.part (.has .cublas) true).holds w := holds_of_mem hS h.1.1
  obtain ⟨hc, hcap⟩ := cudaBase_world hS h.1.2
  have hl : (Fact.part .cuda true).holds w := holds_of_mem hS h.2
  exact ⟨h1, hc, hcap, hl⟩

theorem nblasOf_mem {S : TState} {k : Nat} (h : nblasOf S = some k) : Fact.nblas k ∈ S := by
  unfold nblasOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  subst hk; exact hx

theorem blasOnOf_mem {S : TState} {i : Nat} {s : UInt64} (h : blasOnOf S i = some s) : Fact.blasOn i s ∈ S := by
  unfold blasOnOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl⟩ := hk
  exact hx

theorem blasParty_sound {S : TState} {w : World} {h : UInt64} {q : Nat} (hS : S.holds w)
    (hd : blasParty S h = some q) : w.dev.blasParty? h = some q := by
  unfold blasParty at hd
  obtain ⟨i, hi, hd⟩ := Option.bind_eq_some_iff.mp hd
  obtain ⟨s, hs, hd⟩ := Option.bind_eq_some_iff.mp hd
  have hb : w.dev.blas[i]? = some (some s) := hS _ (blasOnOf_mem hs)
  simp only [Dev.blasParty?, hi, hb, Option.join_some, Option.bind_eq_bind, Option.bind_some]
  exact drvParty_sound hS hd

theorem operandOf_spec {S : TState} {w : World} {p : UInt64} {e span id : Nat} (hS : S.holds w)
    (h : operandOf S p e span = some id) : ∃ off, w.dev.operand? p e span = some (id, off) := by
  unfold operandOf at h
  split at h
  · rename_i id' off hp
    split at h
    · rename_i hc
      cases h
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨id2, off2, b, hr⟩ := allocFor_range hS hc.1
      have hr' := hr
      simp only [Dev.range?, hp, Option.bind_eq_bind, Option.bind_some] at hr'
      obtain ⟨b', hb', hr'⟩ := Option.bind_eq_some_iff.mp hr'
      split at hr'
      · cases hr'
        refine ⟨off / e, ?_⟩
        simp only [Dev.operand?, hr, Option.bind_eq_bind, Option.bind_some, hc.2, beq_self_eq_true, if_true]
      · cases hr'
    · cases h
  · cases h

/-- A party that knows of every access may make any operation. -/
theorem KnowsAll.opReady {r : Race} {p : Nat} (h : Device.KnowsAll r p) (rs ws : List Nat) :
    Contracts.OpReady r p rs ws :=
  ⟨fun b _ => (h.ready b true).1, fun b _ => (h.ready b true).2 rfl⟩

/-- A vendor routine on the device with no capture in progress: the
    tracker's step, then the routine, whose output keeps its size. -/
theorem devOp_vendor {d d' : Dev} {w : World} {p : Nat} {c : VendorCall} {ins rs ws : List Nat} {out : Nat}
    (hc : d.capture = none) (h : d.devOp w p (.vendor c ins out) rs ws = some d') :
    ∃ r nb, d.race.op p rs ws = some r ∧ (d.bufs[out]?.join).map ByteArray.size = some nb.size ∧
      d' = ({ d with race := r }).put out (some nb) := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  unfold Dev.runVendor at h
  obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨old, hold, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · rename_i hsz
    cases h
    refine ⟨r, _, hr, ?_, rfl⟩
    have hold' : (d.bufs[out]?).join = some old := by simpa [Dev.get?] using hold
    rw [hold']
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hsz
    simp [hsz]

/-- A typestate after one device operation by `q` that keeps every size,
    every handle and the driver's state holds. -/
theorem holds_afterOp {S : TState} {w : World} {q : Nat} {r : Race} {out : Nat} {nb : ByteArray} {d' : Dev}
    (hS : S.holds w) (hbusy : w.mem.busy = []) {rs ws : List Nat} (hop : w.dev.race.op q rs ws = some r)
    (hsz : (w.dev.bufs[out]?.join).map ByteArray.size = some nb.size)
    (hd : d' = Dev.put { w.dev with race := r } out (some nb)) :
    (afterOp S q).holds { w with dev := d', mem := w.mem.retire (d'.race.clock hostParty) } := by
  subst hd
  rw [DevSpec.retire_nil hbusy]
  have hsizes := put_sizes (d := { w.dev with race := r }) (id := out) (c := nb) hsz
  refine Contracts.TState.holds_app (fun f hf0 => ?_)
    (Contracts.TState.holds_if fun hrd => Contracts.TState.holds_one ?_)
  · obtain ⟨hf1, hdq⟩ := List.mem_filter.mp hf0
    obtain ⟨hf, hk'⟩ := List.mem_filter.mp hf1
    have hk2 := sizes_keep (d := w.dev) (d' := Dev.put { w.dev with race := r } out (some nb))
      (Dev.put_bufs_size _ _ _) hsizes
    exact holds_devKeep (d := Dev.put { w.dev with race := r } out (some nb)) (hS f hf) hk' rfl rfl rfl rfl rfl rfl (Dev.put_bufs_size _ _ _) rfl rfl hk2.1
      (fun j n _ => hk2.2 j n) (hlive := fun _ => LiveKept.of_sizes hk2.2)
      (fun e => by
        subst e
        have hq0 : q = defaultParty := by simpa using hdq
        subst hq0
        have hsq : Device.TrackOk w.dev ∧ Device.DefaultOnly w.dev.race := hS _ hf
        exact ⟨Device.put_track (track_op (d' := { w.dev with race := r }) hsq.1 hop rfl rfl rfl) _ _,
          Device.DefaultOnly.op hsq.2 hop⟩)
  · exact (Contracts.readyIn_sound hS hrd).op_self hop

/-- What the arguments' patterns know of their values, their bits know. -/
theorem argBits_argAt {Γ : Env} {w : World} :
    ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits →
      ∀ i x, argAt (as.map AV.bits) i = some x → argAt (bits.map some) i = some x
  | [], [], _, i, x, h => by simp [argAt] at h
  | a :: as, y :: ys, ⟨hd, hb⟩, i, x, h => by
      cases i with
      | zero =>
          simp only [argAt, List.map_cons, List.getElem?_cons_zero, Option.join_some] at h ⊢
          rw [known_bits hd h]
      | succ i =>
          simp only [argAt, List.map_cons, List.getElem?_cons_succ] at h ⊢
          exact argBits_argAt hb i x h
  | [], _ :: _, h, _, _, _ => h.elim
  | _ :: _, [], h, _, _, _ => h.elim

theorem getAll_mono {os os' : List (Option UInt64)} (h : ∀ i x, argAt os i = some x → argAt os' i = some x) :
    ∀ {idxs : List Nat} {xs : List UInt64}, getAll os idxs = some xs → getAll os' idxs = some xs
  | [], xs, hx => hx
  | i :: idxs, xs, hx => by
      unfold getAll at hx ⊢
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at hx ⊢
      obtain ⟨a, ha, xs', hxs', rfl⟩ := hx
      exact ⟨a, h i a ha, xs', getAll_mono h hxs', rfl⟩

/-- **`blasIds` only reads what it knows**: values known more widely name the
    same allocations. -/
theorem blasIds_mono {T : TState} {f : CublasFn} {os os' : List (Option UInt64)} {is : List Nat}
    (h : ∀ i x, argAt os i = some x → argAt os' i = some x) (hb : blasIds T f os = some is) :
    blasIds T f os' = some is := by
  unfold blasIds at hb ⊢
  obtain ⟨xs, hxs, hb⟩ := Option.bind_eq_some_iff.mp hb
  rw [getAll_mono h hxs]
  exact hb

theorem operand?_live {d : Dev} {p : UInt64} {e span id off : Nat} (h : d.operand? p e span = some (id, off)) :
    ((d.bufs[id]?).join).isSome = true := by
  unfold Dev.operand? at h
  obtain ⟨⟨id', off', b⟩, hr, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
    rw [(ExtContracts.range?_live hr).1]; rfl
  · cases h

/-- **A vendor routine a capture records**, ordered against the capture by
    the allocations it reads, its output last: the device records it, and the
    capture's fact names what it records. -/
theorem capVendor {S : TState} {w : World} (hS : S.holds w) (hbusy : w.mem.busy = [])
    {cq : Nat} {jn : List Nat} {rc : Race} {evs : List (Nat × Clock)} (hco : capOf S = some (cq, jn, rc, evs))
    {q : Nat} (hin : capIn cq jn q = true) {ins : List Nat} {out : Nat} (hlast : ins.getLast? = some out)
    (hlive : ∀ j ∈ ins, ((w.dev.bufs[j]?).join).isSome = true) {r' : Race} (hop : rc.op q ins [out] = some r')
    (cv : VendorCall) :
    ∃ d'', w.dev.devOp w q (.vendor cv ins out) ins [out] = some d'' ∧
      (withCap S cq jn r' evs).holds { w with dev := d'', mem := w.mem.retire (d''.race.clock hostParty) } ∧
      CuKept w { w with dev := d'', mem := w.mem.retire (d''.race.clock hostParty) } := by
  obtain ⟨c, hcap, rfl, rfl, rfl, hol, hevs⟩ := (hS _ (capOf_mem hco) : (Fact.cap cq jn rc evs).holds w)
  have hcp : w.dev.capturing q = true := capturing_of hcap rfl rfl hin
  let d'' : Dev := { w.dev with capture := some { c with race := r', ops := c.ops.push (.vendor cv ins out) } }
  have hdo : w.dev.devOp w q (.vendor cv ins out) ins [out] = some d'' := by
    unfold Dev.devOp
    simp only [hcap, hcp, if_true, hop, Option.bind_eq_bind, Option.bind_some]
    rfl
  refine ⟨d'', hdo, ?_, by rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _⟩
  rw [DevSpec.retire_nil hbusy]
  refine Contracts.TState.holds_cons ⟨_, rfl, rfl, rfl, rfl, fun o ho => ?_, hevs⟩ (fun f hf => ?_)
  · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at ho
    rcases ho with ho | rfl
    · exact hol o ho
    · exact .inr ⟨_, _, _, rfl, hlast, hlive⟩
  · obtain ⟨hf, hnc⟩ := List.mem_filter.mp hf
    exact holds_capKeep hcap (hS f hf) (fun q' j' r'' e' he' => by subst he'; simp at hnc) (fun e => by
      subst e
      have hsq : (Fact.devSeq).holds w := hS _ hf
      exact ⟨Device.devOp_track hdo hsq.1, hsq.2⟩)

/-- **A product the row accepts runs**, and leaves what the row says. -/
theorem gemm_sound {S S' T : TState} {w : World} {av : AV} {op : String} {ea eb ec : Nat}
    {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hT : T.holds w) {ids : Option (List Nat)}
    (hr : productRow S h pa pb ids = some (S', av))
    (hval : gemmVal T ea eb ec ta tb m n k A lda sa B ldb sb C ldc sc batch = true)
    (hid : ∀ is, ids = some is → gemmIds S ea eb ec ta tb m n k A lda sa B ldb sb C ldc sc batch = some is) :
    w.has .cublas = true ∧ w.dev.live = true ∧ ∃ r w',
      gemmCall w op ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch = some (r, w') ∧
      S'.holds w' ∧ (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧ CuKept w w' := by
  unfold productRow at hr
  split at hr
  · rename_i q hq
    split at hr
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨⟨⟨hbase, hrq⟩, hor⟩, hma⟩, hmb⟩ := hc
      cases hr
      unfold gemmVal at hval
      dsimp only at hval
      split at hval
      rotate_left
      · cases hval
      rename_i ia ib ic hia hib hic
      simp only [Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hval
      obtain ⟨⟨⟨hinv, hun⟩, hca⟩, hcb⟩ := hval
      obtain ⟨hhas, -, hcap, hl⟩ := blasBase_world hS hbase
      have hp := blasParty_sound hS hq
      obtain ⟨oa, hA⟩ := operandOf_spec hT hia
      obtain ⟨ob, hB⟩ := operandOf_spec hT hib
      obtain ⟨oc, hC⟩ := operandOf_spec hT hic
      have hknow := Contracts.readyIn_sound hS hrq
      have hla := Contracts.load_of_state hS hma
      have hlb := Contracts.load_of_state hS hmb
      have hrun : ExtContracts.GemmRuns w ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch := by
        refine ⟨q, hp, fun _ => ⟨hun, ia, oa, ib, ob, ic, oc, hA, hB, hC, hca, hcb, hla, hlb, ?_⟩⟩
        simp only [Dev.raceFor, hcap]
        exact KnowsAll.opReady hknow _ _
      have hv := Contracts.vendorKeeps_of_state hS hor
      obtain ⟨⟨r, w'⟩, hstep⟩ := Option.isSome_iff_exists.mp (ExtContracts.gemmCall_isSome (op := op) hv hrun)
      refine ⟨hhas, hl, r, w', hstep, ?_⟩
      obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hla
      obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hlb
      have e1 : (ic == ia || ic == ib) = false := by simp [hca, hcb]
      unfold gemmCall at hstep
      simp only [hp, hinv, hun, hA, hB, hC, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
        Bool.false_eq_true, if_false] at hstep
      unfold devOnly at hstep
      obtain ⟨⟨r0, d'⟩, hk, he⟩ := Option.map_eq_some_iff.mp hstep
      obtain ⟨d'', hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
      simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
      obtain ⟨rfl, rfl⟩ := hcu
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨rr, nb, hop, hsz, hd⟩ := devOp_vendor hcap hdo
      refine ⟨holds_afterOp hS hbusy hop hsz hd, fun Γ v hv => ?_, ?_⟩
      · cases hv; exact cuOk_desc Γ _
      · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
    · split at hr
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨⟨⟨hbase, hor⟩, hma⟩, hmb⟩ := hc
        split at hr
        · rename_i cq jn rc evs is hco
          split at hr
          · rename_i hin
            split at hr
            · rename_i r' hop
              cases hr
              unfold gemmVal at hval
              dsimp only at hval
              split at hval
              rotate_left
              · cases hval
              rename_i ia ib ic hia hib hic
              simp only [Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hval
              obtain ⟨⟨⟨hinv, hun⟩, hca⟩, hcb⟩ := hval
              simp only [blasCapBase, Bool.and_eq_true] at hbase
              have hhas : w.has .cublas = true := holds_of_mem hS hbase.1.1
              have hl : w.dev.live = true := holds_of_mem hS hbase.2
              have hp := blasParty_sound hS hq
              obtain ⟨oa, hA⟩ := operandOf_spec hT hia
              obtain ⟨ob, hB⟩ := operandOf_spec hT hib
              obtain ⟨oc, hC⟩ := operandOf_spec hT hic
              have hg := hid is rfl
              unfold gemmIds at hg
              dsimp only at hg
              split at hg
              rotate_left
              · cases hg
              rename_i j1 j2 j3 hj1 hj2 hj3
              cases hg
              obtain ⟨_, k1⟩ := operandOf_spec hS hj1
              obtain ⟨_, k2⟩ := operandOf_spec hS hj2
              obtain ⟨_, k3⟩ := operandOf_spec hS hj3
              rw [hA] at k1; rw [hB] at k2; rw [hC] at k3
              cases k1; cases k2; cases k3
              have hla := Contracts.load_of_state hS hma
              have hlb := Contracts.load_of_state hS hmb
              obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hla
              obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hlb
              have e1 : (ic == ia || ic == ib) = false := by simp [hca, hcb]
              obtain ⟨d'', hdo, hS', hk⟩ := capVendor hS hbusy hco hin (ins := [ia, ib, ic]) (out := ic) rfl
                (fun j hj => by
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
                  rcases hj with rfl | rfl | rfl
                  · exact operand?_live hA
                  · exact operand?_live hB
                  · exact operand?_live hC) hop
                ⟨op, [ta, tb, m, n, k, al, be, sa, sb, sc, batch, UInt64.ofNat oa, UInt64.ofNat ob,
                  UInt64.ofNat oc, lda, ldb, ldc]⟩
              refine ⟨hhas, hl, some (ofInt .i32 blasOk), _, ?_, hS', fun Γ v hv => ?_, hk⟩
              · unfold gemmCall
                simp only [hp, hinv, hun, hA, hB, hC, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
                  Bool.false_eq_true, if_false, hdo, devOnly, cuRes, Option.map_some]
              · cases hv; exact cuOk_desc Γ _
            · cases hr
          · cases hr
        · cases hr
      · cases hr
  · cases hr

/-- **A matrix-vector product the row accepts runs**, and leaves what the row
    says. -/
theorem sgemv_sound {S S' T : TState} {w : World} {av : AV}
    {h trans m n pa A lda x incx pb y incy : UInt64}
    (hS : S.holds w) (hbusy : w.mem.busy = []) (hT : T.holds w)
    {ids : Option (List Nat)}
    (hr : productRow S h pa pb ids = some (S', av)) (hval : sgemvVal T trans m n A lda x incx y incy = true)
    (hid : ∀ is, ids = some is → sgemvIds S trans m n A x y = some is) :
    w.has .cublas = true ∧ w.dev.live = true ∧ ∃ r w',
      sgemvCall w h trans m n pa A lda x incx pb y incy = some (r, w') ∧
      S'.holds w' ∧ (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧ CuKept w w' := by
  unfold productRow at hr
  split at hr
  · rename_i q hq
    split at hr
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨⟨⟨hbase, hrq⟩, hor⟩, hma⟩, hmb⟩ := hc
      cases hr
      unfold sgemvVal at hval
      dsimp only at hval
      split at hval
      rotate_left
      · cases hval
      rename_i ia ix iy hia hix hiy
      simp only [Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hval
      obtain ⟨⟨⟨hinv, hun⟩, hya⟩, hyx⟩ := hval
      obtain ⟨hhas, -, hcap, hl⟩ := blasBase_world hS hbase
      have hp := blasParty_sound hS hq
      obtain ⟨oa, hA⟩ := operandOf_spec hT hia
      obtain ⟨ox, hX⟩ := operandOf_spec hT hix
      obtain ⟨oy, hY⟩ := operandOf_spec hT hiy
      have hknow := Contracts.readyIn_sound hS hrq
      have hla := Contracts.load_of_state hS hma
      have hlb := Contracts.load_of_state hS hmb
      have hrun : ExtContracts.SgemvRuns w h trans m n pa A lda x incx pb y incy := by
        refine ⟨q, hp, fun _ => ⟨hun, ia, oa, ix, ox, iy, oy, hA, hX, hY, hya, hyx, hla, hlb, ?_⟩⟩
        simp only [Dev.raceFor, hcap]
        exact KnowsAll.opReady hknow _ _
      have hv := Contracts.vendorKeeps_of_state hS hor
      obtain ⟨⟨r, w'⟩, hstep⟩ := Option.isSome_iff_exists.mp (ExtContracts.sgemvCall_isSome hv hrun)
      refine ⟨hhas, hl, r, w', hstep, ?_⟩
      obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hla
      obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hlb
      have e1 : (iy == ia || iy == ix) = false := by
        simp [hya, hyx]
      unfold sgemvCall at hstep
      simp only [hp, hinv, hun, hA, hX, hY, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
        Bool.false_eq_true, if_false] at hstep
      unfold devOnly at hstep
      obtain ⟨⟨r0, d'⟩, hk, he⟩ := Option.map_eq_some_iff.mp hstep
      obtain ⟨d'', hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
      simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
      obtain ⟨rfl, rfl⟩ := hcu
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨rr, nb, hop, hsz, hd⟩ := devOp_vendor hcap hdo
      refine ⟨holds_afterOp hS hbusy hop hsz hd, fun Γ v hv => ?_, ?_⟩
      · cases hv; exact cuOk_desc Γ _
      · rw [DevSpec.retire_nil hbusy]; exact CuKept.dev w _
    · split at hr
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨⟨⟨hbase, hor⟩, hma⟩, hmb⟩ := hc
        split at hr
        · rename_i cq jn rc evs is hco
          split at hr
          · rename_i hin
            split at hr
            · rename_i r' hop
              cases hr
              unfold sgemvVal at hval
              dsimp only at hval
              split at hval
              rotate_left
              · cases hval
              rename_i ia ix iy hia hix hiy
              simp only [Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hval
              obtain ⟨⟨⟨hinv, hun⟩, hya⟩, hyx⟩ := hval
              simp only [blasCapBase, Bool.and_eq_true] at hbase
              have hhas : w.has .cublas = true := holds_of_mem hS hbase.1.1
              have hl : w.dev.live = true := holds_of_mem hS hbase.2
              have hp := blasParty_sound hS hq
              obtain ⟨oa, hA⟩ := operandOf_spec hT hia
              obtain ⟨ox, hX⟩ := operandOf_spec hT hix
              obtain ⟨oy, hY⟩ := operandOf_spec hT hiy
              have hg := hid is rfl
              unfold sgemvIds at hg
              dsimp only at hg
              split at hg
              rotate_left
              · cases hg
              rename_i j1 j2 j3 hj1 hj2 hj3
              cases hg
              obtain ⟨_, k1⟩ := operandOf_spec hS hj1
              obtain ⟨_, k2⟩ := operandOf_spec hS hj2
              obtain ⟨_, k3⟩ := operandOf_spec hS hj3
              rw [hA] at k1; rw [hX] at k2; rw [hY] at k3
              cases k1; cases k2; cases k3
              have hla := Contracts.load_of_state hS hma
              have hlb := Contracts.load_of_state hS hmb
              obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hla
              obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hlb
              have e1 : (iy == ia || iy == ix) = false := by simp [hya, hyx]
              obtain ⟨d'', hdo, hS', hk⟩ := capVendor hS hbusy hco hin (ins := [ia, ix, iy]) (out := iy) rfl
                (fun j hj => by
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
                  rcases hj with rfl | rfl | rfl
                  · exact operand?_live hA
                  · exact operand?_live hX
                  · exact operand?_live hY) hop
                ⟨"sgemv", [trans, m, n, al, be, UInt64.ofNat oa, UInt64.ofNat ox, UInt64.ofNat oy]⟩
              refine ⟨hhas, hl, some (ofInt .i32 blasOk), _, ?_, hS', fun Γ v hv => ?_, hk⟩
              · unfold sgemvCall
                simp only [hp, hinv, hun, hA, hX, hY, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
                  Bool.false_eq_true, if_false, hdo, devOnly, cuRes, Option.map_some]
              · cases hv; exact cuOk_desc Γ _
            · cases hr
          · cases hr
        · cases hr
      · cases hr
  · cases hr

/-- The bits a list of known values carries are the ones it says. -/
theorem argBits_eq_bits {Γ : Env} {w : World} :
    ∀ {as : List AV} {xs bits : List UInt64}, ArgBits Γ w as bits → as.mapM AV.bits = some xs → bits = xs
  | [], xs, bits, hb, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; exact ArgBits.nil_inv hb
  | a :: as, xs, bits, hb, h => by
      obtain ⟨x, xs', rfl, hx, hb⟩ := ArgBits.cons_inv hb
      rw [List.mapM_cons] at h
      obtain ⟨y, hy, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨ys, hys, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      rw [known_bits hx hy, argBits_eq_bits hb hys]

theorem cublasCall_gemm {w : World} (hl : w.dev.live = true) :
    (∀ h ta tb m n k pa A lda B ldb pb C ldc,
      cublasCall .sgemm [h, ta, tb, m, n, k, pa, A, lda, B, ldb, pb, C, ldc] w =
        gemmCall w "sgemmStridedBatched" 4 4 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1) ∧
    (∀ h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch,
      cublasCall .sgemmStridedBatched [h, ta, tb, m, n, k, pa, A, lda, sa, B, ldb, sb, pb, C, ldc, sc, batch] w =
        gemmCall w "sgemmStridedBatched" 4 4 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch) ∧
    (∀ h ta tb m n k pa A at_ lda B bt ldb pb C ct ldc compute algo, bf16In32Out at_ bt ct compute algo = true →
      cublasCall .gemmEx [h, ta, tb, m, n, k, pa, A, at_, lda, B, bt, ldb, pb, C, ct, ldc, compute, algo] w =
        gemmCall w "gemmStridedBatchedExBf16" 2 2 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1) ∧
    (∀ h ta tb m n k pa A at_ lda sa B bt ldb sb pb C ct ldc sc batch compute algo,
      bf16In32Out at_ bt ct compute algo = true →
      cublasCall .gemmStridedBatchedEx
          [h, ta, tb, m, n, k, pa, A, at_, lda, sa, B, bt, ldb, sb, pb, C, ct, ldc, sc, batch, compute, algo] w =
        gemmCall w "gemmStridedBatchedExBf16" 2 2 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch) := by
  have hnl : (!w.dev.live) = false := by simp [hl]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intros <;>
    simp only [cublasCall, hnl, Bool.false_eq_true, if_false, Bool.false_or, Bool.not_true, *]

/-- Changing only cuBLAS's handles keeps every fact but those about them. -/
theorem holds_blas {x : Fact} {w : World} {bl : Array (Option UInt64)} (h : x.holds w)
    (hn : ∀ n, x = .nblas n → bl.size = w.dev.blas.size)
    (hb : ∀ i s, x = .blasOn i s → bl[i]? = some (some s)) :
    x.holds { w with dev := { w.dev with blas := bl } } := by
  cases x with
  | nblas n => show bl.size = n; rw [hn n rfl]; exact h
  | blasOn i s => exact hb i s rfl
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | devSeq => exact ⟨h.1.congr rfl rfl rfl, h.2⟩
  | wg1 _ s => cases s <;> exact h
  | _ => exact h

theorem lt_of_getElem?_some {α : Type} {xs : Array α} {i : Nat} {v : α} (h : xs[i]? = some v) : i < xs.size := by
  rcases Nat.lt_or_ge i xs.size with hl | hl
  · exact hl
  · rw [Array.getElem?_eq_none hl] at h; cases h

theorem blasHandleRow_sound {S S' : TState} {w : World} {Γ : Env} {f : CublasFn} {as : List AV}
    {bits : List UInt64} {av : AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : blasHandleRow S f as = some (S', av)) : BlasOk f bits w S' av := by
  unfold blasHandleRow at h
  split at h
  · rename_i p hm
    obtain rfl := argBits_eq_bits hb hm
    split at h
    · rename_i k hk
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨hbase, hroom⟩ := hc
        cases h
        obtain ⟨hhas, -, hcap, hl⟩ := blasBase_world hS hbase
        have hcnt : w.dev.blas.size = k := hS _ (nblasOf_mem hk)
        subst hcnt
        obtain ⟨m, hm'⟩ := Option.isSome_iff_exists.mp
          (Contracts.slot8_of_state hS hroom (drvHandle kBlas w.dev.blas.size))
        obtain ⟨hst, hk1⟩ := afterStore_world (ao := some p) hS hm' (fun x hx => (Option.some.inj hx).symm)
        refine ⟨hhas, some (ofInt .i32 blasOk),
          { w with mem := m, dev := { w.dev with blas := w.dev.blas.push (some 0) } }, ?_, ?_, fun Γ v hv => ?_,
          hk1.trans (CuKept.dev _ _)⟩
        · have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cublasCall, hnl, hcap, Option.isSome_none, Bool.or_false, Bool.false_eq_true, if_false, hm',
            Option.bind_eq_bind, Option.bind_some]
        · refine Contracts.TState.holds_cons ?_ (Contracts.TState.holds_cons ?_ (fun f hf => ?_))
          · show (w.dev.blas.push _).size = _; rw [Array.size_push]
          · show (w.dev.blas.push _)[w.dev.blas.size]? = _; rw [Array.getElem?_push_size]
          · obtain ⟨hf, hnb⟩ := List.mem_filter.mp hf
            have h0 := hst f hf
            refine holds_blas (w := { w with mem := m }) (bl := w.dev.blas.push (some 0)) h0
              (fun n e => by subst e; simp at hnb) (fun j t e => ?_)
            subst e
            have hj : w.dev.blas[j]? = some (some t) := h0
            rw [Array.getElem?_push_lt (lt_of_getElem?_some hj)]
            simpa [Array.getElem?_eq_getElem (lt_of_getElem?_some hj)] using hj
        · cases hv; exact cuOk_desc Γ _
      · cases h
    · cases h
  · rename_i hh hm
    obtain rfl := argBits_eq_bits hb hm
    split at h
    · rename_i i hi
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨hbase, hon⟩ := hc
        cases h
        obtain ⟨hhas, -, hcap, hl⟩ := blasBase_world hS hbase
        obtain ⟨s, hs⟩ := Option.isSome_iff_exists.mp hon
        have hbs : w.dev.blas[i]? = some (some s) := hS _ (blasOnOf_mem hs)
        refine ⟨hhas, some (ofInt .i32 blasOk), { w with dev := { w.dev with blas := w.dev.blas.set! i none } },
          ?_, ?_, fun Γ v hv => ?_, CuKept.dev w _⟩
        · have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cublasCall, hnl, hcap, Option.isSome_none, Bool.or_false, Bool.false_eq_true, if_false, hi,
            hbs, Option.join_some, Option.bind_eq_bind, Option.bind_some]
        · intro f hf
          obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
          refine holds_blas (bl := w.dev.blas.set! i none) (hS f hf) (fun n e => Array.size_setIfInBounds)
            (fun j t e => ?_)
          subst e
          have hj : j ≠ i := by simpa using hne
          rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (Ne.symm hj)]; exact hS _ hf
        · cases hv; exact cuOk_desc Γ _
      · cases h
    · cases h
  · rename_i hh sv hm
    obtain rfl := argBits_eq_bits hb hm
    split at h
    · rename_i i hi
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨⟨hbase, hon⟩, hpq⟩ := hc
        cases h
        obtain ⟨hhas, -, hcap, hl⟩ := blasBase_world hS hbase
        obtain ⟨s, hs⟩ := Option.isSome_iff_exists.mp hon
        have hbs : w.dev.blas[i]? = some (some s) := hS _ (blasOnOf_mem hs)
        have hlt := lt_of_getElem?_some hbs
        obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp hpq
        have hsp := drvParty_sound hS hq
        refine ⟨hhas, some (ofInt .i32 blasOk),
          { w with dev := { w.dev with blas := w.dev.blas.set! i (some sv) } }, ?_, ?_, fun Γ v hv => ?_,
          CuKept.dev w _⟩
        · have hnl : (!w.dev.live) = false := by simp [hl]
          simp only [cublasCall, hnl, hcap, Option.isSome_none, Bool.or_false, Bool.false_eq_true, if_false, hi,
            hbs, Option.join_some, hsp, Option.bind_eq_bind, Option.bind_some]
        · refine Contracts.TState.holds_cons ?_ (fun f hf => ?_)
          · show (w.dev.blas.set! i _)[i]? = _
            rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_self_of_lt hlt]
          · obtain ⟨hf, hne⟩ := List.mem_filter.mp hf
            refine holds_blas (bl := w.dev.blas.set! i (some sv)) (hS f hf) (fun n e => Array.size_setIfInBounds)
              (fun j t e => ?_)
            subst e
            have hj : j ≠ i := by simpa using hne
            rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (Ne.symm hj)]; exact hS _ hf
        · cases hv; exact cuOk_desc Γ _
      · cases h
    · cases h
  · cases h

theorem blasRow_sound {S S' T : TState} {w : World} {Γ : Env} {f : CublasFn} {as : List AV}
    {bits : List UInt64} {av : AV} (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w as bits)
    (hT : T.holds w) (hv : blasVal T f (bits.map some) = true)
    (h : blasRow S f as = some (S', av)) : BlasOk f bits w S' av := by
  unfold blasRow at h
  split at h
  · split at h
    · rename_i hx px qx eh ep eq
      have hb0 := hb
      obtain ⟨y0, b, rfl, h0, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y1, b, rfl, h1, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y2, b, rfl, h2, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y3, b, rfl, h3, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y4, b, rfl, h4, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y5, b, rfl, h5, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y6, b, rfl, h6, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y7, b, rfl, h7, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y8, b, rfl, h8, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y9, b, rfl, h9, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y10, b, rfl, h10, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y11, b, rfl, h11, hb⟩ := ArgBits.cons_inv hb
      obtain rfl := ArgBits.nil_inv hb
      obtain rfl := known_bits h0 eh
      obtain rfl := known_bits h4 ep
      obtain rfl := known_bits h9 eq
      simp [blasVal, getAll, argAt] at hv
      obtain ⟨hhas, hl, r, w', hstep, hS', hans, hk⟩ := sgemv_sound hS hbusy hT h hv
        (fun is hi => by have := blasIds_mono (argBits_argAt hb0) hi; simpa [blasIds, blasIdxs, blasIdsOf, getAll, argAt] using this)
      have hnl : (!w.dev.live) = false := by simp [hl]
      exact ⟨hhas, r, w', by simp only [cublasCall, hnl, Bool.false_eq_true, if_false]; exact hstep, hS', hans, hk⟩
    · cases h
  · split at h
    · rename_i hx px qx eh ep eq
      have hb0 := hb
      obtain ⟨y0, b, rfl, h0, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y1, b, rfl, h1, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y2, b, rfl, h2, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y3, b, rfl, h3, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y4, b, rfl, h4, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y5, b, rfl, h5, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y6, b, rfl, h6, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y7, b, rfl, h7, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y8, b, rfl, h8, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y9, b, rfl, h9, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y10, b, rfl, h10, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y11, b, rfl, h11, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y12, b, rfl, h12, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y13, b, rfl, h13, hb⟩ := ArgBits.cons_inv hb
      obtain rfl := ArgBits.nil_inv hb
      obtain rfl := known_bits h0 eh
      obtain rfl := known_bits h6 ep
      obtain rfl := known_bits h11 eq
      simp [blasVal, getAll, argAt] at hv
      obtain ⟨hhas, hl, r, w', hstep, hS', hans, hk⟩ := gemm_sound (op := "sgemmStridedBatched") hS hbusy hT h hv
        (fun is hi => by have := blasIds_mono (argBits_argAt hb0) hi; simpa [blasIds, blasIdxs, blasIdsOf, getAll, argAt] using this)
      exact ⟨hhas, r, w', by rw [(cublasCall_gemm hl).1]; exact hstep, hS', hans, hk⟩
    · cases h
  · split at h
    · rename_i hx px qx eh ep eq
      have hb0 := hb
      obtain ⟨y0, b, rfl, h0, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y1, b, rfl, h1, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y2, b, rfl, h2, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y3, b, rfl, h3, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y4, b, rfl, h4, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y5, b, rfl, h5, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y6, b, rfl, h6, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y7, b, rfl, h7, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y8, b, rfl, h8, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y9, b, rfl, h9, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y10, b, rfl, h10, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y11, b, rfl, h11, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y12, b, rfl, h12, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y13, b, rfl, h13, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y14, b, rfl, h14, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y15, b, rfl, h15, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y16, b, rfl, h16, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y17, b, rfl, h17, hb⟩ := ArgBits.cons_inv hb
      obtain rfl := ArgBits.nil_inv hb
      obtain rfl := known_bits h0 eh
      obtain rfl := known_bits h6 ep
      obtain rfl := known_bits h13 eq
      simp [blasVal, getAll, argAt] at hv
      obtain ⟨hhas, hl, r, w', hstep, hS', hans, hk⟩ := gemm_sound (op := "sgemmStridedBatched") hS hbusy hT h hv
        (fun is hi => by have := blasIds_mono (argBits_argAt hb0) hi; simpa [blasIds, blasIdxs, blasIdsOf, getAll, argAt] using this)
      exact ⟨hhas, r, w', by rw [(cublasCall_gemm hl).2.1]; exact hstep, hS', hans, hk⟩
    · cases h
  · split at h
    · rename_i hx px qx eh ep eq
      have hb0 := hb
      obtain ⟨y0, b, rfl, h0, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y1, b, rfl, h1, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y2, b, rfl, h2, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y3, b, rfl, h3, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y4, b, rfl, h4, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y5, b, rfl, h5, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y6, b, rfl, h6, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y7, b, rfl, h7, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y8, b, rfl, h8, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y9, b, rfl, h9, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y10, b, rfl, h10, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y11, b, rfl, h11, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y12, b, rfl, h12, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y13, b, rfl, h13, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y14, b, rfl, h14, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y15, b, rfl, h15, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y16, b, rfl, h16, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y17, b, rfl, h17, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y18, b, rfl, h18, hb⟩ := ArgBits.cons_inv hb
      obtain rfl := ArgBits.nil_inv hb
      obtain rfl := known_bits h0 eh
      obtain rfl := known_bits h6 ep
      obtain rfl := known_bits h13 eq
      simp [blasVal, getAll, argAt] at hv
      obtain ⟨hbf, hv⟩ := hv
      obtain ⟨hhas, hl, r, w', hstep, hS', hans, hk⟩ := gemm_sound (op := "gemmStridedBatchedExBf16") hS hbusy hT h hv
        (fun is hi => by have := blasIds_mono (argBits_argAt hb0) hi; simpa [blasIds, blasIdxs, blasIdsOf, getAll, argAt] using this)
      exact ⟨hhas, r, w', by rw [(cublasCall_gemm hl).2.2.1 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hbf]; exact hstep,
        hS', hans, hk⟩
    · cases h
  · split at h
    · rename_i hx px qx eh ep eq
      have hb0 := hb
      obtain ⟨y0, b, rfl, h0, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y1, b, rfl, h1, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y2, b, rfl, h2, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y3, b, rfl, h3, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y4, b, rfl, h4, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y5, b, rfl, h5, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y6, b, rfl, h6, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y7, b, rfl, h7, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y8, b, rfl, h8, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y9, b, rfl, h9, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y10, b, rfl, h10, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y11, b, rfl, h11, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y12, b, rfl, h12, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y13, b, rfl, h13, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y14, b, rfl, h14, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y15, b, rfl, h15, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y16, b, rfl, h16, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y17, b, rfl, h17, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y18, b, rfl, h18, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y19, b, rfl, h19, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y20, b, rfl, h20, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y21, b, rfl, h21, hb⟩ := ArgBits.cons_inv hb
      obtain ⟨y22, b, rfl, h22, hb⟩ := ArgBits.cons_inv hb
      obtain rfl := ArgBits.nil_inv hb
      obtain rfl := known_bits h0 eh
      obtain rfl := known_bits h6 ep
      obtain rfl := known_bits h15 eq
      simp [blasVal, getAll, argAt] at hv
      obtain ⟨hbf, hv⟩ := hv
      obtain ⟨hhas, hl, r, w', hstep, hS', hans, hk⟩ := gemm_sound (op := "gemmStridedBatchedExBf16") hS hbusy hT h hv
        (fun is hi => by have := blasIds_mono (argBits_argAt hb0) hi; simpa [blasIds, blasIdxs, blasIdsOf, getAll, argAt] using this)
      exact ⟨hhas, r, w', by
        rw [(cublasCall_gemm hl).2.2.2 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hbf]; exact hstep, hS', hans, hk⟩
    · cases h
  · exact blasHandleRow_sound hS hb h

/-- What a vendor library does with a call. -/
def vendorStep : Ext → List UInt64 → World → Option (Option V × World)
  | .cuda f => cudaDrv f
  | .cublas f => cublasCall f
  | .wgpu f => wgpuCall f
  | _ => fun _ _ => none

/-- A vendor routine keeps every allocation: it writes its output at its size,
    recorded by a capture or run. -/
theorem devOp_vendor_keepsAllocs {d d' : Dev} {w : World} {p : Nat} {c : VendorCall} {ins : List Nat}
    {out : Nat} {rs ws : List Nat} (h : d.devOp w p (.vendor c ins out) rs ws = some d') : AllocsKept d d' := by
  unfold Dev.devOp at h
  have run_ok : ∀ {d'}, Dev.devOp.run d w p (.vendor c ins out) rs ws = some d' → AllocsKept d d' := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    unfold Dev.runVendor at h
    obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨old, hold, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · cases h
    rename_i hc
    simp only [Option.some.injEq] at h
    subst h
    have hsz : (w.vendor c bs).size = old.size := by simpa using hc
    have hold' : (d.bufs[out]?).join = some old := by
      have := hold
      rw [get?_nat] at this
      exact this
    exact AllocsKept.of_sizes (put_sizes (d := { d with race := r })
      (by show (d.bufs[out]?.join).map _ = _; rw [hold', hsz]; rfl))
  split at h
  · split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq] at h
      subst h
      exact AllocsKept.bufs rfl
    · exact run_ok h
  · exact run_ok h

theorem sgemvCall_keepsAllocs {w : World} {h trans m n pa A lda x incx pb y incy : UInt64} {r : Option V}
    {w' : World} (hc : sgemvCall w h trans m n pa A lda x incx pb y incy = some (r, w')) :
    AllocsKept w.dev w'.dev := by
  unfold sgemvCall at hc
  obtain ⟨p, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  · simp only [Option.some.injEq, Prod.mk.injEq] at hc
    obtain ⟨-, rfl⟩ := hc
    exact AllocsKept.bufs rfl
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  dsimp only at hc
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨d, hk, hd⟩ := Device.devOnly_dev hc
  rw [hd]
  obtain ⟨d1, hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
  obtain ⟨-, rfl⟩ := hcu
  exact devOp_vendor_keepsAllocs hdo

theorem gemmCall_keepsAllocs {w : World} {op : String} {ea eb ec : Nat}
    {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64} {r : Option V} {w' : World}
    (hc : gemmCall w op ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch = some (r, w')) :
    AllocsKept w.dev w'.dev := by
  unfold gemmCall at hc
  obtain ⟨p, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  · simp only [Option.some.injEq, Prod.mk.injEq] at hc
    obtain ⟨-, rfl⟩ := hc
    exact AllocsKept.bufs rfl
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  dsimp only at hc
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨d, hk, hd⟩ := Device.devOnly_dev hc
  rw [hd]
  obtain ⟨d1, hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
  obtain ⟨-, rfl⟩ := hcu
  exact devOp_vendor_keepsAllocs hdo

theorem AllocsKept.trans {d d' d'' : Dev} (h1 : AllocsKept d d') (h2 : AllocsKept d' d'') :
    AllocsKept d d'' := fun j b hb => by
  obtain ⟨b', hb', hs⟩ := h1 j b hb
  obtain ⟨b'', hb'', hs'⟩ := h2 j b' hb'
  exact ⟨b'', hb'', hs'.trans hs⟩

/-- A pointer-array batch keeps every allocation, member by member. -/
theorem batchOps_keepsAllocs {w : World} {p : Nat} {ta tb m n k alpha beta : UInt64} {arrs : List Nat} :
    ∀ {ms : List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat))} {d d' : Dev},
      batchOps w p ta tb m n k alpha beta arrs ms d = some d' → AllocsKept d d'
  | [], d, d', h => by
      simp only [batchOps, Option.some.injEq] at h
      subst h
      exact AllocsKept.bufs rfl
  | (a, b, c) :: ms, d, d', h => by
      simp only [batchOps] at h
      obtain ⟨d1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      exact (devOp_vendor_keepsAllocs h1).trans (batchOps_keepsAllocs h)

theorem sgemmBatchedCall_keepsAllocs {w : World} {h ta tb m n k pa PA lda PB ldb pb PC ldc batch : UInt64}
    {r : Option V} {w' : World}
    (hc : sgemmBatchedCall w h ta tb m n k pa PA lda PB ldb pb PC ldc batch = some (r, w')) :
    AllocsKept w.dev w'.dev := by
  unfold sgemmBatchedCall at hc
  obtain ⟨p, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  · simp only [Option.some.injEq, Prod.mk.injEq] at hc
    obtain ⟨-, rfl⟩ := hc
    exact AllocsKept.bufs rfl
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  · cases hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  obtain ⟨d, hk, hd⟩ := Device.devOnly_dev hc
  rw [hd]
  obtain ⟨d1, hdo, hcu⟩ := Option.bind_eq_some_iff.mp hk
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hcu
  obtain ⟨-, rfl⟩ := hcu
  exact batchOps_keepsAllocs hdo
/-- Every cuBLAS call keeps every allocation. -/
theorem cublasCall_keepsAllocs {f : CublasFn} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : cublasCall f bits w = some (r, w')) : AllocsKept w.dev w'.dev := by
  unfold cublasCall at h
  split at h
  all_goals (try dsimp only at h)
  all_goals (try (cases h; done))
  all_goals split at h
  all_goals (try (cases h; done))
  all_goals first
    | (simp only [bind, Option.bind_eq_some_iff, Option.some.injEq, Prod.mk.injEq] at h
       first
         | obtain ⟨_, -, -, rfl⟩ := h
         | obtain ⟨_, -, _, -, -, rfl⟩ := h
         | obtain ⟨_, -, _, -, _, -, -, rfl⟩ := h
       exact fun j b hb => ⟨b, hb, rfl⟩)
    | with_reducible exact sgemvCall_keepsAllocs h
    | with_reducible exact gemmCall_keepsAllocs h
    | with_reducible exact sgemmBatchedCall_keepsAllocs h

/-- **A vendor call that keeps every allocation does**, where kernels keep
    the sizes of the buffers they write. -/
theorem vendorStep_keepsAllocs {e : Ext} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hk : e.keepsAllocs = true) (hks : KeepsSize w.kernel) (h : vendorStep e bits w = some (r, w')) :
    AllocsKept w.dev w'.dev := by
  cases e <;> simp only [Ext.keepsAllocs, Bool.false_eq_true] at hk
  case cuda f =>
    simp only [vendorStep, cudaDrv] at h
    split at h
    · cases h
    · exact cudaStep_keepsAllocs hk hks h
  case cublas f =>
    simp only [vendorStep] at h
    exact cublasCall_keepsAllocs h

/-- What a vendor row leaves: the library's answer and world, the typestate
    the row computes holding there, the answer it says, and the rest kept. -/
def VendorOk (e : Ext) (bits : List UInt64) (w : World) (S' : TState) (av : AV) : Prop :=
  w.has e.lib = true ∧ ∃ r w', vendorStep e bits w = some (r, w') ∧ S'.holds w' ∧
    (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧ CuKeptBut e.kills w w'

/-- The values the checker knows of a call's arguments, where it knows them,
    are the call's bits. -/
def Agree : List (Option UInt64) → List UInt64 → Prop
  | [], [] => True
  | o :: os, b :: bs => (∀ x, o = some x → x = b) ∧ Agree os bs
  | _, _ => False

theorem allocFor_append {S X : TState} {p : UInt64} {n : Nat} (h : allocFor S p n = true) :
    allocFor (S ++ X) p n = true := by
  unfold allocFor at h ⊢
  split at h
  · simp only [List.any_append, h, Bool.true_or]
  · cases h

theorem operandOf_append {S X : TState} {p : UInt64} {e span id : Nat} (h : operandOf S p e span = some id) :
    operandOf (S ++ X) p e span = some id := by
  unfold operandOf at h ⊢
  split at h
  · split at h
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      rw [if_pos (by simp only [Bool.and_eq_true]; exact ⟨allocFor_append hc.1, hc.2⟩)]
      exact h
    · cases h
  · cases h

theorem sgemvVal_mono {S X : TState} {trans m n A lda x incx y incy : UInt64}
    (h : sgemvVal S trans m n A lda x incx y incy = true) : sgemvVal (S ++ X) trans m n A lda x incx y incy = true := by
  unfold sgemvVal at h ⊢
  dsimp only at h ⊢
  split at h
  · rename_i ia ix iy e1 e2 e3
    rw [operandOf_append e1, operandOf_append e2, operandOf_append e3]
    exact h
  · cases h

theorem gemmVal_mono {S X : TState} {ea eb ec : Nat} {ta tb m n k A lda sa B ldb sb C ldc sc batch : UInt64}
    (h : gemmVal S ea eb ec ta tb m n k A lda sa B ldb sb C ldc sc batch = true) :
    gemmVal (S ++ X) ea eb ec ta tb m n k A lda sa B ldb sb C ldc sc batch = true := by
  unfold gemmVal at h ⊢
  dsimp only at h ⊢
  split at h
  · rename_i ia ib ic e1 e2 e3
    rw [operandOf_append e1, operandOf_append e2, operandOf_append e3]
    exact h
  · cases h

/-- A cuBLAS value check the facts pass, more facts pass. -/
theorem blasVal_mono {S X : TState} {f : CublasFn} {os : List (Option UInt64)} (h : blasVal S f os = true) :
    blasVal (S ++ X) f os = true := by
  cases f
  all_goals simp only [blasVal] at h ⊢
  all_goals first
    | rfl
    | (split at h
       · first
           | with_reducible exact sgemvVal_mono h
           | with_reducible exact gemmVal_mono h
           | (simp only [Bool.and_eq_true] at h ⊢; exact ⟨h.1, gemmVal_mono h.2⟩)
       · cases h)

/-- A value check the facts pass, more facts pass. -/
theorem cudaVal_mono {S X : TState} {f : CudaFn} {os : List (Option UInt64)} (h : cudaVal S f os = true) :
    cudaVal (S ++ X) f os = true := by
  cases f
  case memcpyHtoD | memcpyDtoH =>
    simp only [cudaVal] at h ⊢
    split at h
    · rename_i a b e1 e2
      exact allocFor_append h
    · cases h
  case memcpyDtoD =>
    simp only [cudaVal] at h ⊢
    split at h
    · rename_i a b c e1 e2 e3
      simp only [Bool.and_eq_true] at h ⊢
      exact ⟨allocFor_append h.1, allocFor_append h.2⟩
    · cases h
  case launchKernel => exact h
  all_goals simp [cudaVal]

/-- An argument the checker knows the value of has that value in the call. -/
theorem argAt_agree : ∀ {os : List (Option UInt64)} {bits : List UInt64} (i : Nat), Agree os bits →
    ∀ x, argAt os i = some x → argAt (bits.map some) i = some x
  | [], [], _, _, x, hx => by simp [argAt] at hx
  | o :: os, b :: bs, 0, ⟨h, _⟩, x, hx => by
      simp only [argAt, List.getElem?_cons_zero, Option.join_some] at hx
      simp only [argAt, List.map_cons, List.getElem?_cons_zero, Option.join_some]
      rw [h x hx]
  | o :: os, b :: bs, i + 1, ⟨_, h⟩, x, hx => by
      simp only [argAt, List.map_cons, List.getElem?_cons_succ] at hx ⊢
      exact argAt_agree i h x hx
  | [], _ :: _, _, h, _, _ => h.elim
  | _ :: _, [], _, h, _, _ => h.elim

/-- What the value check of a call decides from the values the checker knows
    it decides from the call's bits. -/
theorem cudaVal_known {T : TState} {f : CudaFn} {os : List (Option UInt64)} {bits : List UInt64}
    (hag : Agree os bits) (h : cudaVal T f os = true) :
    cudaVal T f (bits.map some) = true := by
  have hA := fun i x hx => argAt_agree (bits := bits) i hag x hx
  cases f
  case memcpyHtoD | memcpyDtoH =>
    simp only [cudaVal] at h ⊢
    split at h
    · rename_i a b e1 e2
      rw [hA _ a e1, hA _ b e2]
      exact h
    · cases h
  case memcpyDtoD =>
    simp only [cudaVal] at h ⊢
    split at h
    · rename_i a b c e1 e2 e3
      rw [hA _ a e1, hA _ b e2, hA _ c e3]
      exact h
    · cases h
  case launchKernel =>
    simp only [cudaVal, List.all_eq_true] at h ⊢
    intro i hi
    have hd := h i hi
    cases e : argAt os (i + 1) with
    | none => rw [e] at hd; cases hd
    | some x => rw [hA _ x e]; rw [e] at hd; exact hd
  all_goals simp [cudaVal]

/-- Arguments the checker knows the values of have those values in the call. -/
theorem getAll_agree {os : List (Option UInt64)} {bits : List UInt64} (hag : Agree os bits) :
    ∀ (idxs : List Nat) (vs : List UInt64), getAll os idxs = some vs → getAll (bits.map some) idxs = some vs
  | [], vs, h => h
  | i :: is, vs, h => by
      simp only [getAll, List.mapM_cons, Option.bind_eq_bind] at h ⊢
      obtain ⟨x, hx, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨xs, hxs, h⟩ := Option.bind_eq_some_iff.mp h
      rw [argAt_agree i hag x hx]
      have := getAll_agree hag is xs hxs
      simp only [getAll] at this
      simp only [Option.bind_some, this]
      exact h

theorem blasVal_known {T : TState} {f : CublasFn} {os : List (Option UInt64)} {bits : List UInt64}
    (hag : Agree os bits) (h : blasVal T f os = true) : blasVal T f (bits.map some) = true := by
  cases f
  all_goals simp only [blasVal] at h ⊢
  all_goals first
    | rfl
    | (split at h
       · rename_i e
         rw [getAll_agree hag _ _ e]
         exact h
       · cases h)

theorem vendorVal_known {T : TState} {e : Ext} {os : List (Option UInt64)} {bits : List UInt64}
    (hag : Agree os bits) (h : vendorVal T e os = true) :
    vendorVal T e (bits.map some) = true := by
  cases e with
  | cuda f => exact cudaVal_known hag h
  | cublas f => exact blasVal_known hag h
  | _ => rfl

/-- The bits the checker knows of a call's arguments are the call's. -/
theorem argBits_agree {Γ : Env} {w : World} : ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits →
    Agree (as.map (·.bits)) bits
  | [], [], _ => trivial
  | a :: as, b :: bs, ⟨hd, hr⟩ => ⟨fun x (hx : a.bits = some x) => (known_bits hd hx).symm, argBits_agree hr⟩

theorem devPtrOf_spec {p : UInt64} {i o : Nat} (h : devPtrOf p = some (i, o)) :
    p.toNat = (6 + i) * 2 ^ 36 + o ∧ o < 2 ^ 36 ∧ i < 268435449 := by
  unfold devPtrOf at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    omega
  · cases h

theorem devPtrOf_of {p : UInt64} {i o : Nat} (hp : p.toNat = (6 + i) * 2 ^ 36 + o) (ho : o < 2 ^ 36)
    (hi : i < 268435449) : devPtrOf p = some (i, o) := by
  unfold devPtrOf
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  rw [if_pos (by omega)]
  congr 2 <;> omega

theorem allocFor_spec {T : TState} {p : UInt64} {n : Nat} (h : allocFor T p n = true) :
    ∃ i o sz, devPtrOf p = some (i, o) ∧ Fact.alloc i sz ∈ T ∧ o + n ≤ sz := by
  unfold allocFor at h
  split at h
  · rename_i i o hp
    obtain ⟨x, hx, hx'⟩ := List.any_eq_true.mp h
    cases x with
    | alloc j sz =>
        simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hx'
        obtain ⟨rfl, hle⟩ := hx'
        exact ⟨j, o, sz, hp, hx, hle⟩
    | _ => simp at hx'
  · cases h

theorem allocFor_of {T : TState} {p : UInt64} {n i o sz : Nat} (hp : devPtrOf p = some (i, o))
    (ha : Fact.alloc i sz ∈ T) (hle : o + n ≤ sz) : allocFor T p n = true := by
  unfold allocFor
  rw [hp]
  exact List.any_eq_true.mpr ⟨_, ha, by simp [hle]⟩

theorem allocForR_sound {T : TState} {lo hi n : Nat} {x : UInt64} {m : Nat} (h : allocForR T lo hi n = true)
    (hx : lo ≤ x.toNat ∧ x.toNat ≤ hi) (hm : m ≤ n) : allocFor T x m = true := by
  unfold allocForR at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hle, hhi⟩, h⟩ := h
  split at h
  · rename_i i o1 j o2 e1 e2
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, ha⟩ := h
    have hlo : (UInt64.ofNat lo).toNat = lo := by simp; omega
    have hhi' : (UInt64.ofNat hi).toNat = hi := by simp; omega
    obtain ⟨a1, a2, a3⟩ := devPtrOf_spec e1
    obtain ⟨i', o', sz, hp, hal, hle'⟩ := allocFor_spec ha
    rw [e2] at hp; cases hp
    obtain ⟨b1, b2, b3⟩ := devPtrOf_spec e2
    rw [hlo] at a1; rw [hhi'] at b1
    refine allocFor_of (sz := sz) (devPtrOf_of (o := x.toNat - (6 + i) * 2 ^ 36) (by omega) (by omega) a3) hal ?_
    omega
  · cases h

/-- A value within what a range says. -/
def InRng : Rng → UInt64 → Prop
  | some (lo, hi, al), x => lo ≤ x.toNat ∧ x.toNat ≤ hi ∧ al ∣ x.toNat
  | none, _ => True

theorem rng_val {rs : List Rng} {bits : List UInt64} (hl : rs.length = bits.length)
    (hw : ∀ i x, bits[i]? = some x → InRng (rngAt rs i) x) {i lo hi al : Nat} (hr : rngAt rs i = some (lo, hi, al)) :
    ∃ x, argAt (bits.map some) i = some x ∧ lo ≤ x.toNat ∧ x.toNat ≤ hi ∧ al ∣ x.toNat := by
  have hi' : i < bits.length := by
    rw [← hl]
    rcases Nat.lt_or_ge i rs.length with h | h
    · exact h
    · simp [rngAt, List.getElem?_eq_none h] at hr
  obtain ⟨x, hx⟩ : ∃ x, bits[i]? = some x := ⟨bits[i], List.getElem?_eq_getElem hi'⟩
  have := hw i x hx
  rw [hr] at this
  exact ⟨x, by simp [argAt, hx], this⟩

/-- **A range check holds at the values the ranges allow.** -/
theorem cudaValR_sound {T : TState} {f : CudaFn} {rs : List Rng} {bits : List UInt64}
    (hl : rs.length = bits.length) (hw : ∀ i x, bits[i]? = some x → InRng (rngAt rs i) x)
    (h : cudaValR T f rs = true) : cudaVal T f (bits.map some) = true := by
  have hv := fun {i lo hi al} (hr : rngAt rs i = some (lo, hi, al)) => rng_val hl hw hr
  cases f
  case memcpyHtoD =>
    simp only [cudaValR] at h
    split at h
    · rename_i dl dh _da _nl nh _na e1 e2
      obtain ⟨d, hd, hd1, hd2, -⟩ := hv e1
      obtain ⟨n, hn, -, hn2, -⟩ := hv e2
      simp only [cudaVal, hd, hn]
      exact allocForR_sound h ⟨hd1, hd2⟩ hn2
    · cases h
  case memcpyDtoH =>
    simp only [cudaValR] at h
    split at h
    · rename_i dl dh _da _nl nh _na e1 e2
      obtain ⟨d, hd, hd1, hd2, -⟩ := hv e1
      obtain ⟨n, hn, -, hn2, -⟩ := hv e2
      simp only [cudaVal, hd, hn]
      exact allocForR_sound h ⟨hd1, hd2⟩ hn2
    · cases h
  case memcpyDtoD =>
    simp only [cudaValR] at h
    split at h
    · rename_i dl dh _da sl sh _sa _nl nh _na e1 e2 e3
      simp only [Bool.and_eq_true] at h
      obtain ⟨d, hd, hd1, hd2, -⟩ := hv e1
      obtain ⟨s', hs, hs1, hs2, -⟩ := hv e2
      obtain ⟨n, hn, -, hn2, -⟩ := hv e3
      simp only [cudaVal, hd, hs, hn, Bool.and_eq_true]
      exact ⟨allocForR_sound h.1 ⟨hd1, hd2⟩ hn2, allocForR_sound h.2 ⟨hs1, hs2⟩ hn2⟩
    · cases h
  case launchKernel =>
    simp only [cudaValR, cudaVal, List.all_eq_true] at h ⊢
    intro i hi
    have hd := h i hi
    unfold dimR at hd
    split at hd
    · rename_i lo hi' _al e
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hd
      obtain ⟨x, hx, h1, h2, -⟩ := hv e
      rw [hx]
      simp only [dimOk, bne_iff_ne, ne_eq]
      intro h0
      have : (x &&& 4294967295).toNat = x.toNat := by
        simp only [UInt64.toNat_and]
        rw [show (4294967295 : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
        omega
      rw [h0] at this
      simp at this
      omega
    · cases hd
  all_goals simp [cudaVal]

theorem exactR_sound {r : Rng} {v x : UInt64} (h : exactR r = some v) (hx : InRng r x) : x = v := by
  unfold exactR at h
  split at h
  · rename_i lo hi al
    split at h
    · rename_i hc
      cases h
      obtain ⟨h1, h2, -⟩ := hx
      apply UInt64.toNat_inj.mp
      simp only [UInt64.toNat_ofNat']
      rw [Nat.mod_eq_of_lt hc.2]
      omega
    · cases h
  · cases h

/-- A value the range at a place pins down is the value there. -/
theorem exact_at {rs : List Rng} {bits : List UInt64} (hl : rs.length = bits.length)
    (hw : ∀ i x, bits[i]? = some x → InRng (rngAt rs i) x) {i : Nat} {v : UInt64}
    (he : exactR (rngAt rs i) = some v) : argAt (bits.map some) i = some v := by
  cases hr : rngAt rs i with
  | none => rw [hr] at he; cases he
  | some t =>
      obtain ⟨lo, hi, al⟩ := t
      obtain ⟨x, hx, h1, h2, h3⟩ := rng_val hl hw hr
      rw [hr] at he
      rw [hx, exactR_sound he ⟨h1, h2, h3⟩]

/-- An allocation every pointer of a range lies in is the one its first does. -/
theorem allocForR_id {T : TState} {lo hi n : Nat} {x : UInt64} {i o : Nat} (h : allocForR T lo hi n = true)
    (hx : lo ≤ x.toNat ∧ x.toNat ≤ hi) (hlo : devPtrOf (UInt64.ofNat lo) = some (i, o)) :
    ∃ off, devPtrOf x = some (i, off) := by
  unfold allocForR at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hle, hhi⟩, h⟩ := h
  split at h
  · rename_i i' o1 j o2 e1 e2
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, -⟩ := h
    rw [e1] at hlo
    cases hlo
    have hlo' : (UInt64.ofNat lo).toNat = lo := by simp; omega
    have hhi' : (UInt64.ofNat hi).toNat = hi := by simp; omega
    obtain ⟨a1, a2, a3⟩ := devPtrOf_spec e1
    obtain ⟨b1, b2, b3⟩ := devPtrOf_spec e2
    rw [hlo'] at a1; rw [hhi'] at b1
    exact ⟨_, devPtrOf_of (o := x.toNat - (6 + i) * 2 ^ 36) (by omega) (by omega) a3⟩
  · cases h

/-- **An operand a range check places is one at every value it allows.** -/
theorem operandOfR_sound {T : TState} {r : Rng} {e span id : Nat} {x : UInt64}
    (h : operandOfR T r e span = some id) (hx : InRng r x) : operandOf T x e span = some id := by
  unfold operandOfR at h
  split at h
  · rename_i lo hi al
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
      obtain ⟨⟨⟨ha, he⟩, hal⟩, h36⟩ := hc
      obtain ⟨h1, h2, h3⟩ := hx
      cases hd : devPtrOf (UInt64.ofNat lo) with
      | none => rw [hd] at h; cases h
      | some p =>
          obtain ⟨i, o⟩ := p
          rw [hd] at h
          cases h
          obtain ⟨off, hoff⟩ := allocForR_id ha ⟨h1, h2⟩ hd
          have hal' := allocForR_sound ha ⟨h1, h2⟩ (Nat.le_refl _)
          obtain ⟨c1, c2, c3⟩ := devPtrOf_spec hoff
          -- `e` divides the pointer and the allocation's start, so its offset
          have hex : e ∣ x.toNat := Nat.dvd_trans (Nat.dvd_of_mod_eq_zero hal) h3
          have he36 : e ∣ 2 ^ 36 := Nat.dvd_of_mod_eq_zero h36
          have hbase : e ∣ (6 + i) * 2 ^ 36 := Nat.dvd_mul_left_of_dvd he36 _
          have hoff' : e ∣ off := by
            have : off = x.toNat - (6 + i) * 2 ^ 36 := by omega
            rw [this]; exact Nat.dvd_sub hex hbase
          unfold operandOf
          rw [hoff]
          simp only [hal', Bool.true_and, Nat.mod_eq_zero_of_dvd hoff', beq_self_eq_true, if_true]
    · cases h
  · cases h

/-- **A ranged cuBLAS check holds at the values the ranges allow.** -/
theorem blasValR_sound {T : TState} {f : CublasFn} {rs : List Rng} {bits : List UInt64}
    (hl : rs.length = bits.length) (hw : ∀ i x, bits[i]? = some x → InRng (rngAt rs i) x)
    (h : blasValR T f rs = true) : blasVal T f (bits.map some) = true := by
  cases f
  case sgemv =>
    simp only [blasValR, sgemvValR] at h
    split at h
    · rename_i trans m n lda incx incy e1 e2 e3 e6 e8 e11
      split at h
      · rename_i ia ix iy hA hX hY
        have h5 : ∃ t, rngAt rs 5 = some t := by
          cases hr : rngAt rs 5 with
          | none => rw [hr] at hA; cases hA
          | some t => exact ⟨t, rfl⟩
        have h7 : ∃ t, rngAt rs 7 = some t := by
          cases hr : rngAt rs 7 with
          | none => rw [hr] at hX; cases hX
          | some t => exact ⟨t, rfl⟩
        have h10 : ∃ t, rngAt rs 10 = some t := by
          cases hr : rngAt rs 10 with
          | none => rw [hr] at hY; cases hY
          | some t => exact ⟨t, rfl⟩
        obtain ⟨⟨alo, ahi, aal⟩, ha⟩ := h5
        obtain ⟨⟨xlo, xhi, xal⟩, hxr⟩ := h7
        obtain ⟨⟨ylo, yhi, yal⟩, hyr⟩ := h10
        obtain ⟨A, hAa, a1, a2, a3⟩ := rng_val hl hw ha
        obtain ⟨X, hXa, x1, x2, x3⟩ := rng_val hl hw hxr
        obtain ⟨Y, hYa, y1, y2, y3⟩ := rng_val hl hw hyr
        rw [ha] at hA; rw [hxr] at hX; rw [hyr] at hY
        have oA := operandOfR_sound hA ⟨a1, a2, a3⟩
        have oX := operandOfR_sound hX ⟨x1, x2, x3⟩
        have oY := operandOfR_sound hY ⟨y1, y2, y3⟩
        simp only [blasVal, getAll, List.mapM_cons, List.mapM_nil, exact_at hl hw e1, exact_at hl hw e2,
          exact_at hl hw e3, exact_at hl hw e6, exact_at hl hw e8, exact_at hl hw e11, hAa, hXa, hYa,
          Option.bind_eq_bind, Option.bind_some, Option.pure_def]
        simp only [sgemvVal, oA, oX, oY]
        exact h
      · cases h
    · cases h
  all_goals simp [blasValR] at h

/-- The ranges of arguments the checker describes hold their bits. -/
theorem argBits_rng {Γ : Env} {w : World} :
    ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits →
      (as.map AV.rng).length = bits.length ∧ ∀ i x, bits[i]? = some x → InRng (rngAt (as.map AV.rng) i) x
  | [], [], _ => ⟨rfl, fun _ _ h => by simp at h⟩
  | a :: as, b :: bs, ⟨⟨t, hd⟩, hr⟩ => by
      obtain ⟨hl, hw⟩ := argBits_rng hr
      refine ⟨by simp [hl], fun i x hx => ?_⟩
      cases i with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
          subst hx
          simp only [rngAt, List.map_cons, List.getElem?_cons_zero, Option.join_some, AV.rng]
          cases hn : a.num with
          | none => simp [InRng]
          | some n =>
              obtain ⟨y, hy, h1, h2, h3, -⟩ := AV.num_desc hd hn
              simp only [V.sc.injEq] at hy
              obtain ⟨-, rfl⟩ := hy
              exact ⟨h1, h2, h3⟩
      | succ i =>
          simp only [List.getElem?_cons_succ] at hx
          simpa [rngAt] using hw i x hx
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

/-- **What `vendorRow` decides is what the vendor library does**, given what
    `vendorVal` decides of the call's bits. -/
theorem vendorRow_sound {S S' T : TState} {w : World} {Γ : Env} {e : Ext} {as : List AV}
    {bits : List UInt64} {av : AV} (hS : S.holds w) (hbusy : w.mem.busy = []) (hb : ArgBits Γ w as bits)
    (hT : T.holds w) (hv : vendorVal T e (bits.map some) = true)
    (h : vendorRow S e as = some (S', av)) : VendorOk e bits w S' av := by
  cases e with
  | cuda f =>
      obtain ⟨hh, r, w', h1, h2, h3, h4⟩ := cudaRow_sound hS hbusy hb hT hv h
      exact ⟨hh, r, w', h1, h2, h3, h4.but _⟩
  | cublas f =>
      obtain ⟨hh, r, w', h1, h2, h3, h4⟩ := blasRow_sound hS hbusy hb hT hv h
      exact ⟨hh, r, w', h1, h2, h3, h4.but _⟩
  | wgpu f =>
      obtain ⟨r, w', h1, h2, h3, h4⟩ := wgRow_sound hS hb h
      exact ⟨rfl, r, w', h1, h2, h3, h4⟩
  | _ => simp [vendorRow] at h

end AlgorithmLib.Check
