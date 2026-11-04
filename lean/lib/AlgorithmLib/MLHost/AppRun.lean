module
public import AlgorithmLib.MLHost.AppTrace
meta import AlgorithmLib.MLHost.AppTrace
public import AlgorithmLib.Check.DevRun
meta import AlgorithmLib.Check.DevRun
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A call a record admits, as a step of device memory

A program in the standard layout launches kernel `j` with the call
`Drv.App.launchRec j` records. A call that record admits, answered outside a
capture, left device memory as the launch's step: the kernel the function
handle names, its module's text, bound to the allocations the parameter
values it read point into (`launchRec_done`). An upload a record admits
wrote the bytes it read over the allocation it names (`htod_done`), and a
download copied the allocation's bytes out (`dtoh_done`).
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.Check AlgorithmLib.Device

theorem mapM_asBits_of_argsOk : ∀ {ws : List UInt64} {vs : List V}, argsOk (ws.map ArgPat.is) vs →
    vs.mapM asBits = some ws
  | [], [], _ => rfl
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim
  | w :: ws, v :: vs, ⟨hv, h⟩ => by
      have hv' : asBits v = some w := hv
      simp only [List.mapM_cons, hv', mapM_asBits_of_argsOk h, Option.bind_eq_bind, Option.bind_some]; rfl

theorem argsOk_length : ∀ {os : List ArgPat} {vs : List V}, argsOk os vs → vs.length = os.length
  | [], [], _ => rfl
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim
  | _ :: _, _ :: _, ⟨_, h⟩ => by simp [argsOk_length h]

theorem argsOk_split {os os' : List ArgPat} {vs vs' : List V} (hl : vs.length = os.length)
    (h : argsOk (os ++ os') (vs ++ vs')) : argsOk os vs ∧ argsOk os' vs' := by
  induction os generalizing vs with
  | nil => cases vs with
    | nil => exact ⟨trivial, h⟩
    | cons _ _ => simp at hl
  | cons o os ih => cases vs with
    | nil => simp at hl
    | cons v vs =>
      obtain ⟨hv, h⟩ := h
      obtain ⟨h1, h2⟩ := ih (by simpa using hl) h
      exact ⟨⟨hv, h1⟩, h2⟩

theorem mapM_asBits_length : ∀ {vs : List V} {bs : List UInt64}, vs.mapM asBits = some bs → bs.length = vs.length
  | [], bs, h => by simp at h; subst h; rfl
  | v :: vs, bs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind] at h
      obtain ⟨b, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨bs', hbs, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      simp [mapM_asBits_length hbs]

theorem cudaStep_launch_length {bits : List UInt64} {w : World} {x : Option V × World}
    (h : cudaStep .launchKernel bits w = some x) : bits.length = 11 := by
  simp only [cudaStep] at h
  split at h
  · rfl
  · cases h

/-- The launch a record of kernel `j` names: its grid and block, its
    parameter values the pointers to the allocations it is handed. -/
def recLaunchArgs (k : Drv.KSpec) : List UInt64 := k.args.map devPtr

def recDims (k : Drv.KSpec) : List UInt64 :=
  DrvLaunch.dimsOf (UInt64.ofNat k.grid.1) (UInt64.ofNat k.grid.2.1) (UInt64.ofNat k.grid.2.2)
    (UInt64.ofNat k.block.1) (UInt64.ofNat k.block.2.1) (UInt64.ofNat k.block.2.2)

/-- **A call a launch record admits, answered outside a capture, is the
    launch's step**: the function the record names is entry `entry` of a
    module of text `ptx`, and device memory afterwards is that kernel's
    launch over the allocations the record's parameter values point into. -/
theorem launchRec_done {a : Drv.App} {j : Nat} {k : Drv.KSpec} (hk : a.kernels[j]? = some k) (hj : j < 2 ^ 48)
    {w1 w2 : World} {vs : List V} {r : Option V}
    (hhas : w1.has (Ext.cuda .launchKernel).lib = true) (hcap : w1.dev.capture = none)
    (hdims : (recDims k).any (· == 0) = false)
    (hm : (a.launchRec j).Matches (.ext (.cuda .launchKernel), vs ++ callWords w1 (.ext (.cuda .launchKernel)) vs))
    (hx : extCall (.cuda .launchKernel) vs (obsCall w1 (.ext (.cuda .launchKernel)) vs) = some (r, w2)) :
    ∃ mi entry ptx, w1.dev.funcs[j]? = some (mi, entry) ∧ (w1.dev.modules[mi]?).join = some ptx ∧
      devMem w2.dev = (launchStep w1.kernel
        ⟨ptx, entry, DrvLaunch.bindsOf w1.dev (recLaunchArgs k), recDims k, recLaunchArgs k,
          w1.dev.writesOf (w1.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩
        (DrvLaunch.bindsOf w1.dev (recLaunchArgs k))).run (devMem w1.dev) := by
  obtain ⟨-, ha⟩ := hm
  simp only [Drv.App.launchRec, hk] at ha
  have hw1 : (obsCall w1 (.ext (.cuda .launchKernel)) vs).has (Ext.cuda .launchKernel).lib = true := hhas
  simp only [extCall, hw1, Bool.not_true, Bool.false_eq_true, if_false, Option.bind_eq_bind] at hx
  obtain ⟨bits, hbits, hcd⟩ := Option.bind_eq_some_iff.mp hx
  have hcap' : (obsCall w1 (.ext (.cuda .launchKernel)) vs).dev.capture = none := hcap
  have hl : vs.length = 11 := by
    rw [← mapM_asBits_length hbits]
    unfold cudaDrv at hcd
    simp only [hcap', Option.isSome_none, Bool.false_and, Bool.false_eq_true, if_false] at hcd
    exact cudaStep_launch_length hcd
  obtain ⟨h11, hcw⟩ := argsOk_split (os := [_, _, _, _, _, _, _, _, _, _, _]) (by simpa using hl) ha
  have hb := mapM_asBits_of_argsOk (ws := [drvHandle kFunc j, _, _, _, _, _, _, 0, 0, _, 0]) h11
  rw [hbits] at hb
  cases hb
  obtain ⟨args, fi, mi, entry, ptx, hlw, h1, h2, h3, hmem⟩ := DrvLaunch.drvLaunch_done hcap' hdims hcd
  rw [handleOf_drvHandle (by decide : kFunc < 2 ^ 8) hj] at h1
  cases h1
  -- the parameter values read are the ones the record names
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, rfl⟩ := eq_of_length_eleven vs hl
  obtain ⟨hv0, -, -, -, -, -, -, -, -, hv9, -⟩ := h11
  have hv0' : asBits v0 = some (drvHandle kFunc j) := hv0
  have hv9' : asBits v9 = some (arenaAt (a.paramsOff j)) := hv9
  have hlw' : launchWords w1.dev w1.mem (drvHandle kFunc j) (arenaAt (a.paramsOff j)) = some args := hlw
  have hcw' : callWords w1 (.ext (.cuda .launchKernel)) [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
      = args.map (V.sc .i64) := by
    show (match asBits v0, asBits v9 with
      | some f, some p => ((launchWords w1.dev w1.mem f p).getD []).map (V.sc .i64)
      | _, _ => []) = _
    simp only [hv0', hv9', hlw', Option.getD_some]
  rw [hcw'] at hcw
  have e := words_exact (ws := recLaunchArgs k) (by simpa [recLaunchArgs, List.map_map] using hcw)
  have e' : args = recLaunchArgs k := by
    simpa [List.map_map, Function.comp_def, asBits] using e
  subst e'
  exact ⟨mi, entry, ptx, h2, h3, hmem⟩

theorem cublas_strided_length {bits : List UInt64} {w : World} {x : Option V × World}
    (h : cublasCall .sgemmStridedBatched bits w = some x) : bits.length = 18 := by
  unfold cublasCall at h
  split at h <;> first | rfl | (rename_i heq; cases heq) | cases h

/-- What a strided product is observed reading: the scalars at its seventh
    and fourteenth arguments. -/
theorem callWords_strided {w : World} {vs : List V} {bs : List UInt64} (h : vs.mapM asBits = some bs)
    (hl : vs.length = 18) :
    callWords w (.ext (.cublas .sgemmStridedBatched)) vs =
      [.sc .i32 ((w.mem.load (bs.getD 6 0) 4).getD 0), .sc .i32 ((w.mem.load (bs.getD 13 0) 4).getD 0)] := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, rfl⟩ :=
    eq_of_length_eighteen vs hl
  simp only [List.mapM_cons, List.mapM_nil, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq] at h
  obtain ⟨b0, -, _, ⟨b1, -, _, ⟨b2, -, _, ⟨b3, -, _, ⟨b4, -, _, ⟨b5, -, _, ⟨b6, h6, _, ⟨b7, -, _, ⟨b8, -, _,
    ⟨b9, -, _, ⟨b10, -, _, ⟨b11, -, _, ⟨b12, -, _, ⟨b13, h13, _, ⟨b14, -, _, ⟨b15, -, _, ⟨b16, -, _,
    ⟨b17, -, _, rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩,
    rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
  simp only [callWords, h6, h13, Option.bind_some, List.getD_eq_getElem?_getD]
  rfl

/-- A pointer to the start of an allocation names it as an operand, from
    its first element. -/
theorem operand_devPtr {d : Dev} {i e span x o : Nat} (hi : i < 2 ^ 16)
    (h : d.operand? (devPtr i) e span = some (x, o)) : x = i ∧ o = 0 := by
  unfold Dev.operand? at h
  cases hr : d.range? (devPtr i) (e * span) with
  | none => rw [hr] at h; cases h
  | some t =>
      obtain ⟨id, off, b⟩ := t
      rw [hr] at h
      unfold Dev.range? at hr
      rw [devPtrOf_devPtr hi] at hr
      simp only [Option.bind_eq_bind, Option.bind_some] at hr
      obtain ⟨b', -, hr⟩ := Option.bind_eq_some_iff.mp hr
      split at hr
      · simp only [Option.some.injEq, Prod.mk.injEq] at hr
        obtain ⟨rfl, rfl, -⟩ := hr
        simp only [Option.bind_eq_bind, Option.bind_some, Nat.zero_mod, beq_self_eq_true, if_true,
          Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨rfl, Nat.zero_div _⟩
      · cases hr

/-- The vendor call a strided product `g` at a batch of one makes, reading
    scalars `s0`, `s1`, its operands from elements `oa`, `ob`, `oc`. -/
def stridedCall (s0 s1 : UInt64) (oa ob oc : Nat) : Gemm → VendorCall
  | (ta, tb, m, n, k, _, _, _) =>
      ⟨"sgemmStridedBatched", [UInt64.ofNat ta, UInt64.ofNat tb, UInt64.ofNat m, UInt64.ofNat n, UInt64.ofNat k,
        s0, s1, 0, 0, 0, 1, UInt64.ofNat oa, UInt64.ofNat ob, UInt64.ofNat oc,
        UInt64.ofNat (if ta = 1 then k else m), UInt64.ofNat (if tb = 1 then n else k), UInt64.ofNat m]⟩

/-- **A call a strided product's record admits, answered outside a capture,
    on arguments cuBLAS takes, is the vendor's step** for the product, with
    the scalars the record names. -/
theorem stridedRec_done {a : Drv.App} {h s0 s1 : UInt64} {ta tb m n k A B C : Nat} {w1 w2 : World} {vs : List V}
    {r : Option V}
    (hhas : w1.has (Ext.cublas .sgemmStridedBatched).lib = true) (hcap : w1.dev.capture = none)
    (hv : gemmInvalid (UInt64.ofNat ta) (UInt64.ofNat tb) (UInt64.ofNat m) (UInt64.ofNat n) (UInt64.ofNat k)
      (UInt64.ofNat (if ta = 1 then k else m)) (UInt64.ofNat (if tb = 1 then n else k)) (UInt64.ofNat m) 1 = false)
    (hb16 : A < 2 ^ 16 ∧ B < 2 ^ 16 ∧ C < 2 ^ 16)
    (hm : (a.stridedRec h [s0, s1] (ta, tb, m, n, k, A, B, C)).Matches
      (.ext (.cublas .sgemmStridedBatched), vs ++ callWords w1 (.ext (.cublas .sgemmStridedBatched)) vs))
    (hx : extCall (.cublas .sgemmStridedBatched) vs (obsCall w1 (.ext (.cublas .sgemmStridedBatched)) vs) = some (r, w2)) :
    devMem w2.dev =
      (vendorStep w1.vendor (stridedCall s0 s1 0 0 0 (ta, tb, m, n, k, A, B, C)) [A, B, C] C).run (devMem w1.dev) := by
  obtain ⟨-, ha⟩ := hm
  simp only [Drv.App.stridedRec, List.map_append] at ha
  obtain ⟨bits, hbits, hcd⟩ := extCall_cublas (w := obsCall w1 _ vs) hhas hx
  have hl : vs.length = 18 := by rw [← mapM_asBits_length hbits]; exact cublas_strided_length hcd
  obtain ⟨h18, hsc⟩ := argsOk_split (by rw [hl]; rfl) ha
  rw [mapM_asBits_of_argsOk h18] at hbits
  cases hbits
  simp only [Drv.App.stridedWords] at hcd
  obtain ⟨alpha, beta, ia, oa, ib, ob, ic, oc, hal, hbe, hA, hB, hC, hmem⟩ := strided_done (w := obsCall w1 _ _)
    (h := h) (ta := UInt64.ofNat ta) (tb := UInt64.ofNat tb) (m := UInt64.ofNat m) (n := UInt64.ofNat n)
    (k := UInt64.ofNat k) (pa := regionBase .arena + UInt64.ofNat a.ONE) (A := devPtr A)
    (lda := UInt64.ofNat (if ta = 1 then k else m)) (sa := 0) (B := devPtr B)
    (ldb := UInt64.ofNat (if tb = 1 then n else k)) (sb := 0) (pb := regionBase .arena + UInt64.ofNat a.ZERO)
    (C := devPtr C) (ldc := UInt64.ofNat m) (sc := 0) (batch := 1) hcap hv hcd
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, rfl⟩ :=
    eq_of_length_eighteen vs hl
  have hcw := callWords_strided (w := w1) (mapM_asBits_of_argsOk h18) rfl
  simp only [Drv.App.stridedWords, List.getD_eq_getElem?_getD, List.getElem?_cons_succ, List.getElem?_cons_zero,
    Option.getD_some] at hcw
  have hal' : w1.mem.load (regionBase .arena + UInt64.ofNat a.ONE) 4 = some alpha := hal
  have hbe' : w1.mem.load (regionBase .arena + UInt64.ofNat a.ZERO) 4 = some beta := hbe
  rw [hal', hbe', Option.getD_some, Option.getD_some] at hcw
  rw [hcw] at hsc
  obtain ⟨h0, h1, -⟩ := hsc
  have e0 : alpha = s0 := Option.some.inj h0
  have e1 : beta = s1 := Option.some.inj h1
  subst e0 e1
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.1 hA
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.2.1 hB
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.2.2 hC
  exact hmem

/-- A context wait leaves device memory as it was. -/
theorem syncRec_done {w1 w2 : World} {vs : List V} {r : Option V} (hcap : w1.dev.capture = none)
    (hx : extCall (.cuda .ctxSynchronize) vs (obsCall w1 (.ext (.cuda .ctxSynchronize)) vs) = some (r, w2)) :
    devMem w2.dev = devMem w1.dev := by
  simp only [extCall] at hx
  split at hx
  · simp only [Option.some.injEq, Prod.mk.injEq] at hx
    obtain ⟨-, rfl⟩ := hx; rfl
  · simp only [Option.bind_eq_bind] at hx
    obtain ⟨bits, -, hcd⟩ := Option.bind_eq_some_iff.mp hx
    have hcap' : (obsCall w1 (.ext (.cuda .ctxSynchronize)) vs).dev.capture = none := hcap
    rw [cudaDrv_step hcap'] at hcd
    simp only [cudaStep] at hcd
    split at hcd
    · split at hcd
      · cases hcd
      · obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
        simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
        obtain ⟨-, rfl⟩ := hkd
        rfl
    · cases hcd

/-- A step of a run that launches kernels, calls strided products and waits. -/
inductive DevCall where
  | kernel (j : Nat)
  | gemm (g : Gemm)
  | sync

instance : DecidableEq DevCall := fun a b =>
  match a, b with
  | .kernel i, .kernel j =>
      if h : i = j then isTrue (h ▸ rfl) else isFalse (fun e => by cases e; exact h rfl)
  | .gemm (a1, a2, a3, a4, a5, a6, a7, a8), .gemm (b1, b2, b3, b4, b5, b6, b7, b8) =>
      if h : a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6 ∧ a7 = b7 ∧ a8 = b8 then
        isTrue (by obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := h; rfl)
      else isFalse (fun e => by cases e; exact h ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)
  | .sync, .sync => isTrue rfl
  | .kernel _, .gemm _ | .kernel _, .sync | .gemm _, .kernel _ | .gemm _, .sync
  | .sync, .kernel _ | .sync, .gemm _ => isFalse (fun e => by cases e)

/-- The call a step makes, its products on handle `h` reading scalars
    `s0`, `s1`. -/
def DevCall.record (a : Drv.App) (h s0 s1 : UInt64) : DevCall → CallRec
  | .kernel j => a.launchRec j
  | .gemm g => a.stridedRec h [s0, s1] g
  | .sync => ⟨.ext (.cuda .ctxSynchronize), []⟩

/-- What a step's record must be for its call to be read: a kernel of the
    layout with a grid of no zero dimension, a product on arguments cuBLAS
    takes. -/
def DevCall.Ok (a : Drv.App) : DevCall → Prop
  | .kernel j => ∃ k, a.kernels[j]? = some k ∧ j < 2 ^ 48 ∧ (recDims k).any (· == 0) = false
  | .gemm (ta, tb, m, n, k, A, B, C) =>
      gemmInvalid (UInt64.ofNat ta) (UInt64.ofNat tb) (UInt64.ofNat m) (UInt64.ofNat n) (UInt64.ofNat k)
        (UInt64.ofNat (if ta = 1 then k else m)) (UInt64.ofNat (if tb = 1 then n else k)) (UInt64.ofNat m) 1 = false ∧
      A < 2 ^ 16 ∧ B < 2 ^ 16 ∧ C < 2 ^ 16
  | .sync => True

/-- `DevCall.Ok`, decided. -/
def DevCall.okB (a : Drv.App) : DevCall → Bool
  | .kernel j => match a.kernels[j]? with
      | some k => decide (j < 2 ^ 48) && !(recDims k).any (· == 0)
      | none => false
  | .gemm (ta, tb, m, n, k, A, B, C) =>
      !gemmInvalid (UInt64.ofNat ta) (UInt64.ofNat tb) (UInt64.ofNat m) (UInt64.ofNat n) (UInt64.ofNat k)
        (UInt64.ofNat (if ta = 1 then k else m)) (UInt64.ofNat (if tb = 1 then n else k)) (UInt64.ofNat m) 1 &&
      decide (A < 2 ^ 16) && decide (B < 2 ^ 16) && decide (C < 2 ^ 16)
  | .sync => true

theorem DevCall.okB_ok {a : Drv.App} {s : DevCall} (h : s.okB a = true) : s.Ok a := by
  cases s with
  | kernel j =>
      simp only [DevCall.okB] at h
      split at h
      · rename_i k hk
        simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at h
        exact ⟨k, hk, h.1, h.2⟩
      · cases h
  | gemm g =>
      obtain ⟨ta, tb, m, n, k, A, B, C⟩ := g
      simp only [DevCall.okB, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at h
      exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩
  | sync => trivial

/-- **What a step does to device memory**, read against the world `w0` the
    run started from: a kernel's launch step, the vendor's step for a product
    with scalars `s0`, `s1`, or nothing. -/
def DevCall.Step (a : Drv.App) (w0 : World) (s0 s1 : UInt64) : DevCall → DevMem → DevMem → Prop
  | .kernel j, m, m' => ∃ k mi entry ptx, a.kernels[j]? = some k ∧ w0.dev.funcs[j]? = some (mi, entry) ∧
      (w0.dev.modules[mi]?).join = some ptx ∧
      m' = (launchStep w0.kernel
        ⟨ptx, entry, DrvLaunch.bindsOf w0.dev (recLaunchArgs k), recDims k, recLaunchArgs k,
          w0.dev.writesOf (w0.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩
        (DrvLaunch.bindsOf w0.dev (recLaunchArgs k))).run m
  | .gemm g, m, m' => m' = (vendorStep w0.vendor (stridedCall s0 s1 0 0 0 g) [g.2.2.2.2.2.1, g.2.2.2.2.2.2.1,
      g.2.2.2.2.2.2.2] g.2.2.2.2.2.2.2).run m
  | .sync, m, m' => m' = m

/-- Steps taken in order. -/
inductive DevCall.Steps (a : Drv.App) (w0 : World) (s0 s1 : UInt64) : List DevCall → DevMem → DevMem → Prop
  | nil (m : DevMem) : DevCall.Steps a w0 s0 s1 [] m m
  | snoc {ss : List DevCall} {s : DevCall} {m m1 m2 : DevMem} :
      DevCall.Steps a w0 s0 s1 ss m m1 → DevCall.Step a w0 s0 s1 s m1 m2 → DevCall.Steps a w0 s0 s1 (ss ++ [s]) m m2

theorem DevCall.Steps.nil_inv {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {m m' : DevMem}
    (h : DevCall.Steps a w0 s0 s1 [] m m') : m' = m := by
  generalize e : ([] : List DevCall) = l at h
  cases h with
  | nil => rfl
  | snoc => simp at e

/-- The last of several steps, taken apart. -/
theorem DevCall.Steps.snoc_inv {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {ss : List DevCall} {s : DevCall}
    {m m2 : DevMem} (h : DevCall.Steps a w0 s0 s1 (ss ++ [s]) m m2) :
    ∃ m1, DevCall.Steps a w0 s0 s1 ss m m1 ∧ DevCall.Step a w0 s0 s1 s m1 m2 := by
  generalize e : ss ++ [s] = l at h
  cases h with
  | nil => simp at e
  | snoc h1 h2 =>
      obtain ⟨rfl, e2⟩ := List.append_inj' e rfl
      cases e2
      exact ⟨_, h1, h2⟩

theorem recsMatch_snoc : ∀ {rs : List CallRec} {cs : List (Callee × List V)} {x : Callee × List V},
    recsMatch rs (cs ++ [x]) → ∃ rs0 r, rs = rs0 ++ [r] ∧ recsMatch rs0 cs ∧ r.Matches x
  | [r], [], x, ⟨hm, _⟩ => ⟨[], r, rfl, trivial, hm⟩
  | [], [], _, h => h.elim
  | _ :: _ :: _, [], _, ⟨_, h⟩ => h.elim
  | r :: rs, c :: cs, x, ⟨hm, h⟩ => by
      obtain ⟨rs0, r', rfl, h0, hr⟩ := recsMatch_snoc h
      exact ⟨r :: rs0, r', rfl, ⟨hm, h0⟩, hr⟩
  | [], _ :: _, _, h => h.elim

theorem recsMatch_nil_right : ∀ {rs : List CallRec}, recsMatch rs [] → rs = []
  | [], _ => rfl
  | _ :: _, h => h.elim

/-- **A run through calls a list of steps' records admits is those steps**:
    from a world with the driver loaded, and cuBLAS where a step is a
    product, no capture in progress and kernels that keep the sizes of what
    they write, device memory at the end is the steps', in order, from device
    memory at the start, the device otherwise as it was. -/
theorem extRun_steps {a : Drv.App} {h s0 s1 : UInt64} {w0 : World}
    (hcu : w0.has (Ext.cuda .launchKernel).lib = true)
    (hcap : w0.dev.capture = none) (hks : KeepsSize w0.kernel) :
    ∀ {cs : List (Callee × List V)} {w : World}, ExtRun w0 cs w →
      ∀ ss : List DevCall, recsMatch (ss.map (DevCall.record a h s0 s1)) cs → (∀ s ∈ ss, DevCall.Ok a s) →
        (∀ g, DevCall.gemm g ∈ ss → w0.has (Ext.cublas .sgemmStridedBatched).lib = true) →
        DevCall.Steps a w0 s0 s1 ss (devMem w0.dev) (devMem w.dev) ∧ DevSame w0.dev w.dev ∧ Fixed w0 w := by
  intro cs w hr
  induction hr with
  | nil hd ho hf =>
      intro ss hm _ _
      have : ss = [] := by
        have := recsMatch_nil_right hm; simpa using this
      subst this
      refine ⟨?_, ?_, hf⟩
      · rw [hd]; exact .nil _
      · rw [← hd]; exact DevSame.refl _
  | host _ hd ho hf _ ih =>
      intro ss hm hok hbl
      obtain ⟨hs, hsame, hf0⟩ := ih ss hm hok hbl
      refine ⟨by rw [hd]; exact hs, by rw [hd]; exact hsame, hf0.trans hf⟩
  | @call w1 w2 cs e vs r _ hx ho ih =>
      intro ss hm hok hbl
      obtain ⟨rs0, r0, hrs, hm0, hr0⟩ := recsMatch_snoc hm
      obtain ⟨ss0, sl, rfl, hss0, rfl⟩ : ∃ ss0 sl, ss = ss0 ++ [sl] ∧ ss0.map (DevCall.record a h s0 s1) = rs0 ∧
          DevCall.record a h s0 s1 sl = r0 := by
        obtain ⟨l1, l2, e1, e2, e3⟩ := List.map_eq_append_iff.mp hrs
        obtain ⟨sl, rfl, rfl⟩ := List.map_eq_singleton_iff.mp e3
        exact ⟨l1, sl, e1, e2, rfl⟩
      obtain ⟨hs, hsame, hf0⟩ := ih ss0 (hss0 ▸ hm0) (fun s hs => hok s (List.mem_append_left _ hs))
        (fun g hg => hbl g (List.mem_append_left _ hg))
      have hokl := hok sl (List.mem_append_right _ (List.mem_singleton_self _))
      have hcap1 : w1.dev.capture = none := by rw [hsame.capture]; exact hcap
      have hks1 : KeepsSize w1.kernel := by rw [hf0.1]; exact hks
      have hpres : w1.present = w0.present := hf0.2.2
      have has1 : ∀ l : Lib, w0.has l = true → w1.has l = true := fun l hl => by
        unfold World.has at hl ⊢
        rw [hpres]; exact hl
      cases sl with
      | kernel j =>
          obtain ⟨k, hk, hj, hdims⟩ := hokl
          have he : Callee.ext e = .ext (.cuda .launchKernel) := by
            have := hr0.1; simp only [DevCall.record, Drv.App.launchRec, hk] at this; exact this.symm
          injection he with he
          subst he
          obtain ⟨mi, entry, ptx, hfn, hmod, hmem⟩ := launchRec_done hk hj (has1 _ hcu) hcap1 hdims hr0 hx
          obtain ⟨bits, -, hcd⟩ := extCall_cuda (w := obsCall w1 _ vs) (has1 _ hcu) hx
          have hkp : CallKeeps w1 w2 := launch_keeps (w := obsCall w1 _ vs) hcap1 hks1 hcd
          have hfn0 : w0.dev.funcs[j]? = some (mi, entry) := by rw [← hsame.funcs]; exact hfn
          have hmod0 : (w0.dev.modules[mi]?).join = some ptx := by rw [← hsame.modules]; exact hmod
          refine ⟨.snoc hs ⟨k, mi, entry, ptx, hk, hfn0, hmod0, ?_⟩, hsame.trans hkp.1, hf0.trans hkp.2⟩
          rw [hmem, hf0.1, bindsOf_same hsame, hsame.kernelWrites, writesOf_same hsame]
      | gemm g =>
          have hblg := hbl g (List.mem_append_right _ (List.mem_singleton_self _))
          obtain ⟨ta, tb, m, n, k, A, B, C⟩ := g
          have he : Callee.ext e = .ext (.cublas .sgemmStridedBatched) := hr0.1.symm
          injection he with he
          subst he
          have hmem := stridedRec_done (has1 _ hblg) hcap1 hokl.1 hokl.2 hr0 hx
          obtain ⟨bits, hbits, hcd⟩ := extCall_cublas (w := obsCall w1 _ vs) (has1 _ hblg) hx
          have hl : vs.length = 18 := by rw [← mapM_asBits_length hbits]; exact cublas_strided_length hcd
          have ha := hr0.2
          simp only [DevCall.record, Drv.App.stridedRec, List.map_append] at ha
          obtain ⟨h18, -⟩ := argsOk_split (by rw [hl]; rfl) ha
          rw [mapM_asBits_of_argsOk h18] at hbits
          cases hbits
          simp only [Drv.App.stridedWords] at hcd
          have hkp : CallKeeps w1 w2 := strided_keeps (w := obsCall w1 _ vs) hcap1 hks1 hcd
          exact ⟨.snoc hs (by rw [hmem, hf0.2.1]; rfl), hsame.trans hkp.1, hf0.trans hkp.2⟩
      | sync =>
          have he : Callee.ext e = .ext (.cuda .ctxSynchronize) := hr0.1.symm
          injection he with he
          subst he
          have hmem := syncRec_done hcap1 hx
          obtain ⟨bits, -, hcd⟩ := extCall_cuda (f := .ctxSynchronize) (w := obsCall w1 _ vs) (has1 _ hcu) hx
          have hkp : CallKeeps w1 w2 := ctxSync_keeps (w := obsCall w1 _ vs) hcap1 hcd
          exact ⟨.snoc hs hmem, hsame.trans hkp.1, hf0.trans hkp.2⟩

theorem writeAll_size : ∀ (ws : List (Nat × ByteArray)) (m : DevMem),
    (∀ p ∈ ws, p.2.size = (m p.1).size) → ∀ g, (writeAll m ws g).size = (m g).size
  | [], _, _, _ => rfl
  | (b, x) :: ws, m, h, g => by
      have hx : x.size = (m b).size := h (b, x) List.mem_cons_self
      have h' : ∀ p ∈ ws, p.2.size = (setBuf m b x p.1).size := fun p hp => by
        rw [h p (List.mem_cons_of_mem _ hp)]
        unfold setBuf
        split
        · rename_i e; rw [e, hx]
        · rfl
      simp only [writeAll]
      rw [writeAll_size ws _ h' g]
      unfold setBuf
      split
      · rename_i e; rw [e, hx]
      · rfl

/-- **A launch keeps every buffer's size**: it replaces bound buffers only
    with answers of their own sizes. -/
theorem launchStep_size (k : Launch → List ByteArray → List ByteArray) (l : Launch) (ids : List Nat)
    (m : DevMem) (g : Nat) : ((launchStep k l ids).run m g).size = (m g).size := by
  simp only [launchStep]
  split
  · rfl
  · rename_i hs
    have hs' : sizesKept (k l (ids.map m)) (ids.map m) = false := by simpa using hs
    simp only [sizesKept, Bool.or_eq_false_iff, bne_eq_false_iff_eq, List.any_eq_false] at hs'
    obtain ⟨hlen, hall⟩ := hs'
    have hlen' : (k l (ids.map m)).length = ids.length := by simpa using hlen
    refine writeAll_size _ m (fun p hp => ?_) g
    obtain ⟨n, hn1, hn⟩ := List.getElem_of_mem (List.mem_filter.mp hp).1
    have hn1' : n < ids.length := by simp at hn1; omega
    have hn2 : n < (k l (ids.map m)).length := by omega
    have hp' : p = (ids[n], (k l (ids.map m))[n]) := by rw [← hn]; simp
    subst hp'
    have hm : ((k l (ids.map m))[n], (ids.map m)[n]'(by simp; omega)) ∈ (k l (ids.map m)).zip (ids.map m) :=
      List.mem_iff_getElem.mpr ⟨n, by simp; omega, by simp⟩
    have := hall _ hm
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at this
    simpa using this

/-- A product keeps every buffer's size: it replaces its output only with an
    answer of its size. -/
theorem vendorStep_size (v : VendorCall → List ByteArray → ByteArray) (c : VendorCall) (ins : List Nat)
    (out : Nat) (m : DevMem) (g : Nat) : ((Device.vendorStep v c ins out).run m g).size = (m g).size := by
  simp only [Device.vendorStep]
  split
  · rfl
  · rename_i hs
    unfold setBuf
    split
    · rename_i e; subst e; simpa using hs
    · rfl

/-- **Steps keep every buffer's size.** -/
theorem DevCall.Steps.size {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {ss : List DevCall} {m m' : DevMem}
    (h : DevCall.Steps a w0 s0 s1 ss m m') (g : Nat) : (m' g).size = (m g).size := by
  induction h with
  | nil => rfl
  | @snoc ss s m0 m1 m2 _ hs ih =>
      cases s with
      | kernel j =>
          obtain ⟨_, _, _, _, _, _, _, rfl⟩ := hs
          rw [launchStep_size, ih]
      | gemm g' => rw [hs, vendorStep_size, ih]
      | sync => rw [hs, ih]

/-- `memcpyHtoD` and `memcpyDtoH` are observed with their arguments alone. -/
theorem callWords_htod (w : World) (vs : List V) : callWords w (.ext (.cuda .memcpyHtoD)) vs = [] := by
  unfold callWords; split <;> first | rfl | (rename_i h; cases h) | skip
theorem callWords_dtoh (w : World) (vs : List V) : callWords w (.ext (.cuda .memcpyDtoH)) vs = [] := by
  unfold callWords; split <;> first | rfl | (rename_i h; cases h) | skip

/-- Three values a record of three exact or bounded places admits. -/
theorem args_three {o0 o1 o2 : ArgPat} {vs : List V} (h : argsOk [o0, o1, o2] vs) :
    ∃ v0 v1 v2, vs = [v0, v1, v2] ∧ argOk o0 v0 ∧ argOk o1 v1 ∧ argOk o2 v2 := by
  rcases vs with _ | ⟨v0, _ | ⟨v1, _ | ⟨v2, _ | ⟨v3, vs⟩⟩⟩⟩
  · exact h.elim
  · exact h.2.elim
  · exact h.2.2.elim
  · exact ⟨v0, v1, v2, rfl, h.1, h.2.1, h.2.2.1⟩
  · exact h.2.2.2.elim

/-- A same-size write of a live allocation keeps the device but for that
    allocation's bytes. -/
theorem syncWrite_same {d d' : Dev} {id : Nat} {c : ByteArray}
    (h : (d.bufs[id]?.join).map ByteArray.size = some c.size) (hw : d.syncWrite id c = some d') : DevSame d d' := by
  unfold Dev.syncWrite at hw
  obtain ⟨r, -, hw⟩ := Option.bind_eq_some_iff.mp hw
  cases hw
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, Dev.put_bufs_size _ _ _, rfl, rfl, rfl, fun i => ?_, rfl⟩
  simp only [Dev.get?]
  split
  · rfl
  · exact put_sizes h _

/-- **The upload a record admits**: a `memcpyHtoD` to `dst` from `src` of a
    length within `[lo, hi]`, answered, read that many bytes at `src` and
    wrote them over the allocation `dst` falls in, at its offset, keeping the
    device otherwise. -/
theorem htod_done {w1 w2 : World} {vs : List V} {r : Option V} {dst src : UInt64} {lo hi : Nat}
    (hhas : w1.has (Ext.cuda .memcpyHtoD).lib = true) (hcap : w1.dev.capture = none)
    (hm : CallRec.Matches ⟨.ext (.cuda .memcpyHtoD), [.is dst, .is src, .within lo hi]⟩
      (.ext (.cuda .memcpyHtoD), vs ++ callWords w1 (.ext (.cuda .memcpyHtoD)) vs))
    (hx : extCall (.cuda .memcpyHtoD) vs (obsCall w1 (.ext (.cuda .memcpyHtoD)) vs) = some (r, w2)) :
    ∃ id off B bytes, w1.dev.range? dst bytes.size = some (id, off, B) ∧ lo ≤ bytes.size ∧ bytes.size ≤ hi ∧
      readBytes w1.mem src bytes.size = some bytes ∧
      devMem w2.dev = setBuf (devMem w1.dev) id (overwrite B off bytes) ∧ CallKeeps w1 w2 := by
  obtain ⟨-, ha⟩ := hm
  rw [callWords_htod, List.append_nil] at ha
  obtain ⟨v0, v1, v2, rfl, h0, h1, h2⟩ := args_three ha
  obtain ⟨n, hn, hlo, hhi⟩ := h2
  obtain ⟨bits, hbits, hcd⟩ := extCall_cuda (w := obsCall w1 _ _) hhas hx
  have h0' : asBits v0 = some dst := h0
  have h1' : asBits v1 = some src := h1
  simp only [List.mapM_cons, List.mapM_nil, h0', h1', hn, Option.bind_eq_bind, Option.bind_some,
    Option.pure_def, Option.some.injEq] at hbits
  subst hbits
  have hcap' : (obsCall w1 (.ext (.cuda .memcpyHtoD)) [v0, v1, v2]).dev.capture = none := hcap
  rw [cudaDrv_step hcap'] at hcd
  simp only [cudaStep, Option.bind_eq_bind] at hcd
  split at hcd
  · cases hcd
  · obtain ⟨⟨id, off, B⟩, hr, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    obtain ⟨bytes, hb, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
    obtain ⟨d1, hsw, hkd⟩ := Option.bind_eq_some_iff.mp hkd
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
    obtain ⟨-, hd⟩ := hkd
    subst hd
    replace hr : w1.dev.range? dst n.toNat = some (id, off, B) := hr
    replace hb : readBytes w1.mem src n.toNat = some bytes := hb
    replace hsw : w1.dev.syncWrite id (overwrite B off bytes) = some d1 := hsw
    have hsz : bytes.size = n.toNat := copyOut_size hb
    have hr' : w1.dev.range? dst bytes.size = some (id, off, B) := by rw [hsz]; exact hr
    obtain ⟨hlb, hfit⟩ := ExtContracts.range?_live hr
    have hidl : id < w1.dev.bufs.size := by
      refine Nat.lt_of_not_le fun hc => ?_
      rw [Array.getElem?_eq_none hc] at hlb; cases hlb
    have hsame : DevSame w1.dev d1 :=
      syncWrite_same (by rw [hlb, Device.overwrite_size (by omega)]; rfl) hsw
    refine ⟨id, off, B, bytes, hr', by omega, by omega, by rw [hsz]; exact hb, ?_, hsame, Fixed.refl _⟩
    unfold Dev.syncWrite at hsw
    obtain ⟨rr, -, hsw⟩ := Option.bind_eq_some_iff.mp hsw
    cases hsw
    exact devMem_put _ _ _ hidl

/-- **The download a record admits**: a `memcpyDtoH` to `dst` from `src` of
    `n` bytes, answered, copied those bytes of the allocation `src` falls in
    to `dst`, keeping the device but for its clocks. -/
theorem dtoh_done {w1 w2 : World} {vs : List V} {r : Option V} {dst src n : UInt64}
    (hhas : w1.has (Ext.cuda .memcpyDtoH).lib = true) (hcap : w1.dev.capture = none)
    (hm : CallRec.Matches ⟨.ext (.cuda .memcpyDtoH), [.is dst, .is src, .is n]⟩
      (.ext (.cuda .memcpyDtoH), vs ++ callWords w1 (.ext (.cuda .memcpyDtoH)) vs))
    (hx : extCall (.cuda .memcpyDtoH) vs (obsCall w1 (.ext (.cuda .memcpyDtoH)) vs) = some (r, w2)) :
    ∃ id off B, w1.dev.range? src n.toNat = some (id, off, B) ∧
      copyIn w1.mem dst (B.extract off (off + n.toNat)) = some w2.mem ∧ devMem w2.dev = devMem w1.dev := by
  obtain ⟨-, ha⟩ := hm
  rw [callWords_dtoh, List.append_nil] at ha
  obtain ⟨v0, v1, v2, rfl, h0, h1, h2⟩ := args_three ha
  obtain ⟨bits, hbits, hcd⟩ := extCall_cuda (w := obsCall w1 _ _) hhas hx
  have h0' : asBits v0 = some dst := h0
  have h1' : asBits v1 = some src := h1
  have h2' : asBits v2 = some n := h2
  simp only [List.mapM_cons, List.mapM_nil, h0', h1', h2', Option.bind_eq_bind, Option.bind_some,
    Option.pure_def, Option.some.injEq] at hbits
  subst hbits
  have hcap' : (obsCall w1 (.ext (.cuda .memcpyDtoH)) [v0, v1, v2]).dev.capture = none := hcap
  rw [cudaDrv_step hcap'] at hcd
  simp only [cudaStep, Option.bind_eq_bind] at hcd
  split at hcd
  · cases hcd
  · obtain ⟨⟨id, off, B⟩, hr, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    obtain ⟨d1, hrd, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    obtain ⟨m, hci, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    cases hcd
    replace hr : w1.dev.range? src n.toNat = some (id, off, B) := hr
    replace hrd : w1.dev.syncRead id = some d1 := hrd
    refine ⟨id, off, B, hr, hci, ?_⟩
    unfold Dev.syncRead at hrd
    obtain ⟨rr, -, hrd⟩ := Option.bind_eq_some_iff.mp hrd
    cases hrd
    rfl

/-- Where an allocation range falls, and the bytes it found. -/
theorem range_eq {d : Dev} {p : UInt64} {n id off : Nat} {B : ByteArray}
    (h : d.range? p n = some (id, off, B)) : devPtrOf p = some (id, off) ∧ devMem d id = B := by
  unfold Dev.range? at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨⟨i, o⟩, hp, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨c, hc, h⟩ := Option.bind_eq_some_iff.mp h
  split at h
  · cases h; exact ⟨hp, by simp [devMem, hc]⟩
  · cases h

end AlgorithmLib.ML
