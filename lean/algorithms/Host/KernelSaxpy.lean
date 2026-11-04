module
public import Host.KernelCorpus
meta import Host.KernelCorpus
public import AlgorithmLib.Kernel.Elementwise
meta import AlgorithmLib.Kernel.Elementwise
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
public import AlgorithmLib.Kernel.Host
meta import AlgorithmLib.Kernel.Host
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# saxpy, from one thread's proof to what a block leaves

The corpus's `saxpy` kernel (`KernelCorpus.saxpy`, the one
`tools/kernel_agree.py` runs on the GPU) proven: one thread's body by the
program logic (`thread_wp`), what a thread that finishes leaves
(`thread_spec`), and, as an elementwise kernel (`elem`), what a launch leaves
(`launch_spec`): every `y[j]` its grid covers holds `fma(a, x[j], y[j])`. The
launch law (`Kernel.Runs`) carries this to what the host sees after launching
the printed kernel: `drv_launch_saxpy` is a host program's `cuLaunchKernel`
of its text, after which every `y[j]` the grid covers, read from the device
buffer, holds `fma(a, x[j], y[j])`. The scalar `a` is taken to be an f32 word (below `2^32`),
as the kernel's f32 parameter holds one.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace KernelSaxpy

def params : List ClifTy := [.i64, .i64, .i32, .f32]

/-- The kernel: the corpus's body, as the function of its parameters. -/
def K : Kernel := { name := "saxpy", params, code := emit KernelCorpus.saxpy (idParams ++ params) }

attribute [local irreducible] wp

/-- **What one thread does**: thread `i < n` stores `fma(a, x[i], y[i])`, a
    NaN as the platform leaves it, at `y[i]`; the rest leave memory as it was. -/
theorem thread_wp (cfg : Cfg) (w : World) (b t bd n : UInt32) (px py ab : UInt64) (sm : UInt64)
    (xv yv : UInt64) (m : Mem)
    (hx : (b * bd + t) < n → w.mem.load (px + (s32 ((b * bd + t)) <<< (c64 2 % 64))) 4 = some xv)
    (hy : (b * bd + t) < n → w.mem.load (py + (s32 ((b * bd + t)) <<< (c64 2 % 64))) 4 = some yv)
    (hs : (b * bd + t) < n →
      w.mem.store (py + (s32 ((b * bd + t)) <<< (c64 2 % 64))) 4 (w.mem.fl .f32 (fmaLane .f32 ab xv yv)) = some m) :
    wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody (idParams ++ params) KernelCorpus.saxpy)
      (fun _ _ w' => if (b * bd + t) < n then w'.mem = m ∧ ∃ p x, w'.obs = .store p 4 x :: w.obs
                     else w' = w)
      #[.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab] w := by
  rw [fnBody_of (by decide +kernel)]
  simp only [KernelCorpus.saxpy, KernelCorpus.gid, entryParams, wp_bind, wp_params, wp_ret]
  simp only [idParams, params, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.thd,
    Vals.fth, Vals.fif, Vals.tail, Vals.drop, Nat.zero_add, Nat.reduceAdd]
  have h4 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab])[4]? = some (.sc .i64 px) := rfl
  have h5 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab])[5]? = some (.sc .i64 py) := rfl
  have h6 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab])[6]? = some (.sc .i32 (s32 n)) := rfl
  have h7 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab])[7]? = some (.sc .f32 ab) := rfl
  refine wp_imul32 (x := b) (y := bd) rfl rfl ?_
  refine wp_iadd32 (x := b * bd) (y := t) (by look) rfl ?_
  simp only [when, wp_bind]
  refine wp_ifteT (Qj := fun _ _ w' => if (b * bd + t) < n then w'.mem = m ∧ ∃ p x, w'.obs = .store p 4 x :: w.obs
                     else w' = w) rfl (by look) (by look) ?_ ?_ ?_
  · intro hc
    have hlt : (b * bd + t) < n := by rw [cmp32_ult] at hc; exact of_decide_eq_true hc
    repeat step
    refine wp_loadF32 (by look) (hx hlt) ?_
    repeat step
    refine wp_loadF32 (by look) (hy hlt) ?_
    rw [wp_bind]
    refine wp_fma32 (x := ab) (y := xv) (z := yv) (by look) (by look) (by look) ?_
    refine wp_storeF32 (by look) (by look) (hs hlt) ?_
    rw [wp_pure]
    simp [hlt, obsStore]
  · intro hc ne hne
    have hlt : ¬ (b * bd + t) < n := by rw [cmp32_ult] at hc; exact of_decide_eq_false hc
    rw [wp_pure]
    simp only [hlt, if_false]
  · intro nJ Γ' jv vs w' hle hvs hq
    rw [wp_pure]
    exact hq

theorem fine : Fine (emitGo (fnBody (idParams ++ params) KernelCorpus.saxpy) ⟨(idParams ++ params).length, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **What a saxpy thread leaves**, when it finishes. -/
theorem thread_spec (b t bd n : UInt32) (px py ab sm : UInt64) (xv yv : UInt64) (m : Mem) (w u : World)
    (hx : (b * bd + t) < n → w.mem.load (px + (s32 (b * bd + t) <<< (c64 2 % 64))) 4 = some xv)
    (hy : (b * bd + t) < n → w.mem.load (py + (s32 (b * bd + t) <<< (c64 2 % 64))) 4 = some yv)
    (hs : (b * bd + t) < n →
      w.mem.store (py + (s32 (b * bd + t) <<< (c64 2 % 64))) 4 (w.mem.fl .f32 (fmaLane .f32 ab xv yv)) = some m)
    {ans : List Answer} {steps : Nat}
    (hr : thread K.code ans [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab] w Hw.rounded steps = .ok u) :
    if (b * bd + t) < n then u.mem = m ∧ ∃ p x, u.obs = .store p 4 x :: w.obs else u = w := by
  refine thread_of_triple (Q := fun u => if (b * bd + t) < n then u.mem = m ∧ ∃ p x, u.obs = .store p 4 x :: w.obs else u = w) (fun cfg => ?_) hr
  have h1 := emit_triple (cfg := cfg) (J := { ok := fun _ _ => True, faultOk := false })
    (P0 := At (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab]) w)
    (PT.conseq (P' := fun Γ' w' => Γ' = (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab]) ∧ w' = w)
      (wp_sound cfg 0 _ (fnBody (idParams ++ params) KernelCorpus.saxpy) _)
      (fun Γ' w' h => by rw [h.1, h.2]; exact thread_wp cfg w b t bd n px py ab sm xv yv m hx hy hs)
      (fun _ _ _ h => h)) fine (fun Γ' w' h => ⟨by rw [h.1]; rfl, h⟩)
  exact h1

/-- `x` and `y`: device buffers 0 and 1. -/
abbrev px : UInt64 := addrOf (.buf ⟨0, by decide⟩) 0
abbrev py : UInt64 := addrOf (.buf ⟨1, by decide⟩) 0
def args (n : UInt32) (ab : UInt64) : List V := [.sc .i64 px, .sc .i64 py, .sc .i32 (s32 n), .sc .f32 ab]

/-- **saxpy as an elementwise kernel**: thread `j < n` reads `x[j]` and `y[j]`
    and stores `fma(a, x[j], y[j])` at `y[j]`. -/
def elem (n : UInt32) (ab : UInt64) (X Y : UInt32 → UInt64) : Elem where
  out j := py + off (UInt32.ofNat j)
  val j := nanFix gpuNaN .f32 (fmaLane .f32 ab (X (UInt32.ofNat j)) (Y (UInt32.ofNat j)))
  active j := decide (UInt32.ofNat j < n)
  reads j := [(px + off (UInt32.ofNat j), X (UInt32.ofNat j)), (py + off (UInt32.ofNat j), Y (UInt32.ofNat j))]

/-- Its threads are apart: each its own element of `y`, reading `x` (another
    buffer) and only its own `y`. -/
theorem elem_apart (n : UInt32) (ab : UInt64) (X Y : UInt32 → UInt64) (N : Nat) (hN : N ≤ 2 ^ 32) :
    (elem n ab X Y).Apart N where
  buf j _ _ := ⟨_, _, decode_elem _ _⟩
  outs i j hi hj hne _ _ := by
    refine ⟨_, _, _, _, decode_elem _ _, decode_elem _ _, Or.inr ?_⟩
    rw [toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)]; omega
  reads i j hi hj hne _ _ p hp := by
    simp only [elem, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact ⟨_, _, _, _, decode_elem _ _, decode_elem _ _, Or.inl (by simp)⟩
    · refine ⟨_, _, _, _, decode_elem _ _, decode_elem _ _, Or.inr ?_⟩
      rw [toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)]; omega

/-- Its threads do what `thread_spec` says. -/
theorem elem_threads (n : UInt32) (ab : UInt64) (X Y : UInt32 → UInt64) (grid bd : Nat) :
    Elem.Threads K (args n ab) grid bd (elem n ab X Y) := by
  intro b t _ _ s s' ans steps hnan hP hr
  by_cases ha : UInt32.ofNat (b * bd + t) < n
  · have hact : (elem n ab X Y).active (b * bd + t) = true := by simp only [elem, decide_eq_true_eq]; exact ha
    obtain ⟨hrd, hs⟩ := hP hact
    have hx := hrd _ List.mem_cons_self
    have hy := hrd _ (List.mem_cons_of_mem _ List.mem_cons_self)
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs ((elem n ab X Y).val (b * bd + t)))
    have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) n px py ab (addrOf .arena 0)
      (X (UInt32.ofNat (b * bd + t))) (Y (UInt32.ofNat (b * bd + t))) m s s'
      (fun _ => by rw [gi_eq]; exact hx) (fun _ => by rw [gi_eq]; exact hy)
      (fun _ => by rw [gi_eq, show s.mem.fl = nanFix gpuNaN from by show nanFix s.mem.nan = _; rw [hnan]]; exact hm) hr
    rw [gi_eq, if_pos ha] at h
    obtain ⟨hmem, p, x, hobs⟩ := h
    refine ⟨?_, collCalls_store hobs⟩
    unfold Elem.eff put
    rw [if_pos hact, hm, Option.getD_some, hmem]
  · have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) n px py ab (addrOf .arena 0)
      0 0 s.mem s s' (fun h => absurd (gi_eq b bd t ▸ h) ha) (fun h => absurd (gi_eq b bd t ▸ h) ha)
      (fun h => absurd (gi_eq b bd t ▸ h) ha) hr
    rw [gi_eq, if_neg ha] at h
    subst h
    refine ⟨?_, collCalls_self s'⟩
    unfold Elem.eff
    rw [if_neg (by simp only [elem, decide_eq_true_eq]; exact ha)]

/-- **What a launch of saxpy leaves**: after `grid` blocks of `bd` threads,
    each element `j < n` of the grid holds `fma(a, x[j], y[j])` in `y`. -/
theorem launch_spec (n : UInt32) (ab : UInt64) (X Y : UInt32 → UInt64) (grid bd : Nat)
    (hsz : grid * bd ≤ 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hpre : ∀ j, j < grid * bd → (elem n ab X Y).Pre (kmem bufs) j)
    (hab : ab.toNat < 2 ^ 32) (hr : LaunchRuns K grid bd (args n ab) bufs bufs') :
    ∀ j, j < grid * bd → UInt32.ofNat j < n →
      ({ kmem bufs with bufs := bufs' } : Mem).load (py + off (UInt32.ofNat j)) 4 =
        some (nanFix gpuNaN .f32 (fmaLane .f32 ab (X (UInt32.ofNat j)) (Y (UInt32.ofNat j)))) :=
  fun j hj ha => Elem.launch_spec rfl (elem_apart n ab X Y _ hsz) (elem_threads n ab X Y grid bd)
    bufs bufs' hpre (fun i hi hact => by
      obtain ⟨hrd, -⟩ := hpre i hi hact
      exact nanFix_lt _ (fmaLane32_lt _ _ _ hab (load4_lt (hrd _ List.mem_cons_self))
        (load4_lt (hrd _ (List.mem_cons_of_mem _ List.mem_cons_self))))) hr j hj (by simp [elem, ha])

/-! ## Launched by a host program -/

open AlgorithmLib.HProg.DevSpec AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.Kernel

/-- The parameter sizes the driver reads off saxpy's printed text. -/
theorem K_sizes : ptxParamSizes K.text K.name = some [8, 8, 4, 4] := by native_decide

/-- The memory a saxpy launch binds: device buffers `0` (`x`) and `1` (`y`). -/
def pick (d : Dev) (i : Nat) : ByteArray := if i = 0 ∨ i = 1 then devMem d i else .empty

/-- **A host program's launch of saxpy.** A `cuLaunchKernel` of saxpy's
    printed text, its parameters `x` and `y` (device buffers `0` and `1`), `n`
    and `a`, from a world where the call answers, on a device that runs the
    text as `Kernel.Runs` says, answers; and where the threads run, every
    `y[j]` the grid covers holds `fma(a, x[j], y[j])` afterwards. -/
theorem drv_launch_saxpy {w : World} {hf gx bx shmem hstream params : UInt64} {fi mi p : Nat}
    {n : UInt32} {ab : UInt64} {X Y : UInt32 → UInt64} {B0 B1 : ByteArray}
    (hruns : Kernel.Runs w.kernel K)
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, K.name))
    (h3 : w.dev.modules[mi]? = some (some K.text)) (h4 : w.dev.streamParty? hstream = some p)
    (h6 : kernelArgs w.mem params [8, 8, 4, 4] = some [px, py, s32 n, ab])
    (hb0 : w.dev.bufs[0]? = some (some B0)) (hb1 : w.dev.bufs[1]? = some (some B1))
    (hr0 : Ready w.dev.race p 0 true) (hr1 : Ready w.dev.race p 1 true)
    (hdims : (DrvLaunch.dimsOf gx 1 1 bx 1 1).any (· == 0) = false)
    (hall : ∀ i : Nat, ((w.dev.kernelWrites K.text K.name).bind (·[i]?)).getD true = true)
    (hsz : (gx &&& 0xffffffff).toNat * (bx &&& 0xffffffff).toNat ≤ 2 ^ 32)
    (hab : ab.toNat < 2 ^ 32)
    (hpre : ∀ j, j < (gx &&& 0xffffffff).toNat * (bx &&& 0xffffffff).toNat →
      (elem n ab X Y).Pre (kmem (pick w.dev)) j) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, 1, 1, bx, 1, 1, shmem, hstream, params, 0] w = some (r, w') ∧
      ((∃ bufs, LaunchRuns K (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (args n ab) (pick w.dev) bufs) →
        ∀ j, j < (gx &&& 0xffffffff).toNat * (bx &&& 0xffffffff).toNat → UInt32.ofNat j < n →
          (kmem (devMem w'.dev)).load (py + off (UInt32.ofNat j)) 4 =
            some (nanFix gpuNaN .f32 (fmaLane .f32 ab (X (UInt32.ofNat j)) (Y (UInt32.ofNat j))))) := by
  have hn : (s32 n).toNat < 2 ^ 32 := by simp [s32]; exact n.toNat_lt
  have hbinds : DrvLaunch.bindsOf w.dev [px, py, s32 n, ab] = [0, 1] := by
    have hx : devPtrOf px = some (0, 0) := by decide
    have hy : devPtrOf py = some (1, 0) := by decide
    simp [DrvLaunch.bindsOf, Dev.range?, hx, hy, devPtrOf_small hn, devPtrOf_small hab, hb0, hb1]
    decide
  have hx : devPtrOf px = some (0, 0) := by decide
  have hy : devPtrOf py = some (1, 0) := by decide
  obtain ⟨r, w', h, hM⟩ := drv_launch (k := K) (args := [px, py, s32 n, ab]) (shmem := shmem)
    hruns hl hcap hk h1 h2 h3 h4 K_sizes h6 hdims
    (fun a ha id o b hra => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
      rcases ha with rfl | rfl | rfl | rfl
      · simp [Dev.range?, hx, hb0] at hra; rw [← hra.1]; exact hr0
      · simp [Dev.range?, hy, hb1] at hra; rw [← hra.1]; exact hr1
      · simp [Dev.range?, devPtrOf_small hn] at hra
      · simp [Dev.range?, devPtrOf_small hab] at hra) hall
  refine ⟨r, w', h, fun hex j hj ha => ?_⟩
  have hargs : K.argsOf (drvLaunchOf K w.dev gx bx [px, py, s32 n, ab]) = args n ab := by
    simp [Kernel.argsOf, drvLaunchOf, K, KernelSaxpy.params, args, norm, widthMask, and_mask32 hn,
      and_mask32 hab, and_mask64]
  have hmem : bindsMem (drvLaunchOf K w.dev gx bx [px, py, s32 n, ab])
      ((drvLaunchOf K w.dev gx bx [px, py, s32 n, ab]).binds.map (devMem w.dev)) = pick w.dev := by
    funext i
    simp only [bindsMem, drvLaunchOf, hbinds, pick]
    rcases i with _ | _ | i
    · rfl
    · rfl
    · have : List.idxOf? (i + 1 + 1) [0, 1] = none := by simp [List.idxOf?_eq_none_iff]
      rw [this]; simp
  obtain ⟨bufs, hb, hout⟩ := hM (by rw [hargs, hmem]; exact hex)
  rw [hargs, hmem] at hb
  simp only [drvLaunchOf, hbinds, List.map_cons, List.map_nil, List.cons.injEq] at hout
  rw [← launch_spec n ab X Y _ _ hsz (pick w.dev) bufs hpre hab hb j hj ha]
  exact load_buf_eq (decode_elem ⟨1, by decide⟩ _) hout.2.1

end KernelSaxpy
