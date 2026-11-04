module
public import AlgorithmLib.Kernel.Proof
meta import AlgorithmLib.Kernel.Proof
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Elementwise` — a kernel whose threads each store one word

Most kernels are elementwise: thread `j` reads some words and stores one. If
no thread's word overlaps another's word or reads (`Elem.Apart`), a launch
leaves every active thread's word its value (`Elem.launch_spec`), whatever the
blocks and their order. A kernel proves its thread (`Elem.Threads`), by the
program logic, and nothing else.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec

namespace AlgorithmLib.Kernel

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

theorem collCalls_store {w u : World} {p x : UInt64} {n : Nat} (h : u.obs = .store p n x :: w.obs) :
    collCalls u w.obs.length = [] := by
  simp [collCalls, collOf, h]

theorem collCalls_self (w : World) : collCalls w w.obs.length = [] := by
  unfold collCalls
  rw [show w.obs.length = w.obs.reverse.length by simp, List.drop_length]
  rfl

theorem store_arena {m m' : Mem} {a v : UInt64} {n : Nat} {i : Fin 268435449} {off : Nat}
    (hd : decodeAddr a = some (.buf i, off)) (h : m.store a n v = some m') : m'.arena = m.arena := by
  obtain ⟨r, o, hd', hF⟩ := store_frame h
  rw [hd] at hd'; cases hd'
  exact hF.other .arena (by simp)

theorem foldl_put_filter (addr val : Nat → UInt64) (a : Nat → Bool) : ∀ (l : List Nat) (m : Mem),
    l.foldl (fun m t => if a t then put addr val t m else m) m = (l.filter a).foldl (fun m t => put addr val t m) m
  | [], _ => rfl
  | t :: ts, m => by
    by_cases h : a t
    · simp [List.foldl_cons, h, foldl_put_filter addr val a ts]
    · simp [List.foldl_cons, h, foldl_put_filter addr val a ts]

theorem world_arena (s : World) {A : ByteArray} (h : s.mem.arena = A) :
    ({ s with mem := { s.mem with arena := A } } : World) = s := by subst h; rfl

theorem mem_arena_self (x : Mem) {A : ByteArray} (h : x.arena = A) : ({ x with arena := A } : Mem) = x := by
  subst h; rfl

theorem load_bufs {m₁ m₂ : Mem} {a : UInt64} {n : Nat} {j : Fin 268435449} {o : Nat}
    (hd : decodeAddr a = some (.buf j, o)) (h : m₁.bufs = m₂.bufs) : m₁.load a n = m₂.load a n := by
  simp [Mem.load, hd, Mem.region, Mem.readable, h]

/-- A load from buffer `j` depends on that buffer alone. -/
theorem load_buf_eq {m₁ m₂ : Mem} {a : UInt64} {n : Nat} {j : Fin 268435449} {o : Nat}
    (hd : decodeAddr a = some (.buf j, o)) (h : m₁.bufs j.val = m₂.bufs j.val) :
    m₁.load a n = m₂.load a n := by
  simp [Mem.load, hd, Mem.region, Mem.readable, h]

/-- Two block-then-thread loops are one loop over the global index. -/
theorem foldl_grid {α : Type} (g : Nat → α → α) (bd : Nat) : ∀ (grid : Nat) (a : α),
    (List.range grid).foldl (fun a b => (List.range bd).foldl (fun a t => g (b * bd + t) a) a) a =
      (List.range (grid * bd)).foldl (fun a j => g j a) a
  | 0, a => by simp
  | k + 1, a => by
    rw [List.range_succ, List.foldl_append, foldl_grid g bd k a, List.foldl_cons, List.foldl_nil,
      show (k + 1) * bd = k * bd + bd by rw [Nat.succ_mul], List.range_add, List.foldl_append, List.foldl_map]

/-- Thread `t` of block `b`'s 32-bit index is the global index. -/
theorem gi_eq (b bd t : Nat) :
    UInt32.ofNat b * UInt32.ofNat bd + UInt32.ofNat t = UInt32.ofNat (b * bd + t) := by
  simp [UInt32.ofNat_add, UInt32.ofNat_mul]

/-- Byte `o` of device buffer `i`. -/
theorem decode_buf (i : Fin 268435449) (o : Nat) (ho : o < 2 ^ 36) :
    decodeAddr (addrOf (.buf i) 0 + UInt64.ofNat o) = some (.buf i, o) := by
  rw [addrOf, show regionBase (.buf i) + UInt64.ofNat 0 = regionBase (.buf i) by simp]
  exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact ho)

/-- The offset of word `i` of a buffer. -/
abbrev off (i : UInt32) : UInt64 := s32 i <<< (c64 2 % 64)

theorem off_eq (i : UInt32) : off i = UInt64.ofNat (4 * i.toNat) := by
  unfold off
  rw [show c64 2 % 64 = (2 : UInt64) from by decide]
  apply UInt64.toNat_inj.mp
  have := i.toNat_lt
  simp [s32, UInt64.toNat_shiftLeft, Nat.shiftLeft_eq]
  omega

/-- Word `i` of buffer `j`. -/
theorem decode_elem (j : Fin 268435449) (i : UInt32) :
    decodeAddr (addrOf (.buf j) 0 + off i) = some (.buf j, 4 * i.toNat) := by
  have := i.toNat_lt
  rw [off_eq, addrOf, show regionBase (.buf j) + UInt64.ofNat 0 = regionBase (.buf j) by simp]
  exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; omega)

theorem toNat_ofNat32 {j : Nat} (hj : j < 2 ^ 32) : (UInt32.ofNat j).toNat = j := by
  simp [Nat.mod_eq_of_lt hj]

/-- The memory a launch starts from. -/
def kmem (bufs : Nat → ByteArray) (nan : ClifTy → UInt64 → UInt64 := gpuNaN) : Mem :=
  { arena := .empty, data := .empty, out := .empty, bufs, nan }

-- ---------------------------------------------------------------------------
-- Elementwise kernels
-- ---------------------------------------------------------------------------

/-- **An elementwise kernel's threads**: thread `j` (block `b`, thread `t`, at
    `j = b · blockDim + t`), when `active`, reads four-byte words `reads j` each
    holding a known value and stores `val j` at `out j`, a word of a device
    buffer; otherwise it does nothing. -/
structure Elem where
  out : Nat → UInt64
  val : Nat → UInt64
  active : Nat → Bool
  reads : Nat → List (UInt64 × UInt64)

namespace Elem
variable (E : Elem)

/-- What thread `j` needs of memory: its reads hold their values, and its word
    has room. -/
def Pre (m : Mem) (j : Nat) : Prop :=
  E.active j → (∀ p ∈ E.reads j, m.load p.1 4 = some p.2) ∧ ∀ v, (m.store (E.out j) 4 v).isSome

/-- What thread `j` does to memory. -/
def eff (j : Nat) (m : Mem) : Mem := if E.active j then put E.out E.val j m else m

/-- No two threads below `N` disturb each other: their words are apart, and
    no thread's word overlaps another's reads. Each word is a device buffer's. -/
structure Apart (N : Nat) : Prop where
  buf : ∀ j, j < N → E.active j → ∃ i o, decodeAddr (E.out j) = some (.buf i, o)
  outs : ∀ i j, i < N → j < N → i ≠ j → E.active i → E.active j → Apart4 E.out i j
  reads : ∀ i j, i < N → j < N → i ≠ j → E.active i → E.active j → ∀ p ∈ E.reads j,
    ∃ r off r' off', decodeAddr (E.out i) = some (r, off) ∧ decodeAddr p.1 = some (r', off') ∧
      (r' ≠ r ∨ off' + 4 ≤ off ∨ off + 4 ≤ off')

variable {E}

theorem eff_arena {N : Nat} (hA : E.Apart N) {j : Nat} (hj : j < N) (m : Mem) :
    (E.eff j m).arena = m.arena := by
  unfold eff put
  split
  · rename_i ha
    obtain ⟨i, o, hd⟩ := hA.buf j hj ha
    cases h : m.store (E.out j) 4 (E.val j) with
    | none => rfl
    | some m' => exact store_arena hd h
  · rfl

/-- A thread's effect leaves the platform's NaN rule. -/
theorem eff_nan (j : Nat) (m : Mem) : (E.eff j m).nan = m.nan := by
  unfold eff put
  split
  · cases h : m.store (E.out j) 4 (E.val j) with
    | none => rfl
    | some m' => exact Mem.store_nan h
  · rfl

theorem foldl_eff_nan : ∀ (l : List Nat) (m : Mem), (l.foldl (fun m j => E.eff j m) m).nan = m.nan
  | [], _ => rfl
  | j :: js, m => by rw [List.foldl_cons, foldl_eff_nan js, eff_nan]

theorem eff_keep {N : Nat} (hA : E.Apart N) {i j : Nat} (hi : i < N) (hj : j < N) (hne : j ≠ i) {m : Mem}
    (h : E.Pre m i) : E.Pre (E.eff j m) i := by
  intro hai
  obtain ⟨hr, hs⟩ := h hai
  unfold eff put
  split
  · rename_i haj
    cases hst : m.store (E.out j) 4 (E.val j) with
    | none => exact ⟨hr, hs⟩
    | some m' =>
      obtain ⟨_, _, _, hF⟩ := store_frame hst
      simp only [Option.getD_some]
      refine ⟨fun p hp => ?_, fun v => ?_⟩
      · obtain ⟨r, off, r', off', h1, h2, hc⟩ := hA.reads j i hj hi hne haj hai p hp
        rw [load_store_clear hst h1 h2 hc, hr p hp]
      · rw [← MemSame.store_isSome hF.same _ 4 v v]; exact hs v
  · exact ⟨hr, hs⟩

theorem foldl_eff_keep {N : Nat} (hA : E.Apart N) {i : Nat} (hi : i < N) : ∀ (l : List Nat) (m : Mem),
    i ∉ l → (∀ j ∈ l, j < N) → E.Pre m i → E.Pre (l.foldl (fun m j => E.eff j m) m) i
  | [], _, _, _, h => h
  | j :: js, m, hni, hl, h => by
    rw [List.foldl_cons]
    exact foldl_eff_keep hA hi js _ (fun e => hni (List.mem_cons_of_mem _ e)) (fun x hx => hl x (List.mem_cons_of_mem _ hx))
      (eff_keep hA hi (hl j List.mem_cons_self) (fun e => hni (e ▸ List.mem_cons_self)) h)

theorem foldl_eff_arena {N : Nat} (hA : E.Apart N) : ∀ (l : List Nat) (m : Mem), (∀ j ∈ l, j < N) →
    (l.foldl (fun m j => E.eff j m) m).arena = m.arena
  | [], _, _ => rfl
  | j :: js, m, hl => by
    rw [List.foldl_cons, foldl_eff_arena hA js _ (fun x hx => hl x (List.mem_cons_of_mem _ hx)),
      eff_arena hA (hl j List.mem_cons_self)]

section Launch
variable (k : Kernel) (params : List V) (grid bd : Nat)

/-- What every thread of the launch does, as its proof says: from memory where
    its precondition holds, on the hardware's float arithmetic, a thread that
    finishes leaves that memory with its effect, and makes no collective. -/
def Threads (E : Elem) : Prop :=
  ∀ b t, b < grid → t < bd → ∀ (s s' : World) (ans : List Answer) (steps : Nat),
    s.mem.nan = gpuNaN → E.Pre s.mem (b * bd + t) → thread k.code ans (threadArgs b bd params t) s Hw.rounded steps = .ok s' →
    s'.mem = E.eff (b * bd + t) s.mem ∧ collCalls s' s.obs.length = []

/-- What every block of the launch does, as its proof says: from memory where
    every thread's precondition holds and shared memory is zeroed, a block
    that finishes leaves its threads' effects, in order. -/
def Blocks (E : Elem) : Prop :=
  ∀ b, b < grid → ∀ (w w' : World), w.mem.arena = ByteArray.empty → w.mem.nan = gpuNaN →
    (∀ t, t < bd → E.Pre w.mem (b * bd + t)) → BlockRuns k b bd params Hw.rounded w w' →
    w'.mem = (List.range bd).foldl (fun m t => E.eff (b * bd + t) m) w.mem

variable {k params grid bd}

theorem lt_grid {b t : Nat} (hb : b < grid) (ht : t < bd) : b * bd + t < grid * bd := by
  have : (b + 1) * bd ≤ grid * bd := Nat.mul_le_mul_right _ hb
  rw [Nat.succ_mul] at this; omega

/-- **A block of an elementwise kernel**: its threads' effects, in order. -/
theorem block_mem (hA : E.Apart (grid * bd)) (hth : Threads k params grid bd E)
    {b : Nat} (hb : b < grid) {w w' : World} (harena : w.mem.arena = ByteArray.mk (Array.replicate k.smem 0))
    (hnan : w.mem.nan = gpuNaN) (hpre : ∀ t, t < bd → E.Pre w.mem (b * bd + t)) (hr : BlockRuns k b bd params Hw.rounded w w') :
    w'.mem = (List.range bd).foldl (fun m t => E.eff (b * bd + t) m) w.mem := by
  have hw0 : ({ w with mem := { w.mem with arena := ByteArray.mk (Array.replicate k.smem 0) } } : World) = w :=
    world_arena w harena
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, -, -, rfl⟩ := hr
  rw [hw0] at hfold
  let f (t : Nat) (m : Mem) : Mem := if t < bd then E.eff (b * bd + t) m else m
  have hmem := fold_mem (S := PassSt)
    (fun t s => passStep (k.run Hw.rounded steps) (ByteArray.mk (Array.replicate k.smem 0)) (threadArgs b bd params) ans s t)
    (fun s => s.1.mem)
    (fun t m => t < bd ∧ m.arena = ByteArray.mk (Array.replicate k.smem 0) ∧ E.Pre m (b * bd + t) ∧
      m.nan = gpuNaN) f
    (fun t s s' hP h => by
      obtain ⟨w1, c1, s1⟩ := s
      simp only [passStep, Kernel.run, bind, Except.bind] at h
      split at h
      · cases h
      rename_i u hu
      cases h
      have h' : thread k.code (ans t) (threadArgs b bd params t) w1 Hw.rounded steps = .ok u :=
        (congrArg (fun z => thread k.code (ans t) (threadArgs b bd params t) z Hw.rounded steps)
          (world_arena (A := ByteArray.mk (Array.replicate k.smem 0)) w1 hP.2.1)).symm.trans hu
      simp only [f, if_pos hP.1]; exact (hth b t hb hP.1 w1 u (ans t) steps hP.2.2.2 hP.2.2.1 h').1)
    (fun t u m htu hP => by
      refine ⟨hP.1, ?_, ?_, ?_⟩
      · by_cases ht : t < bd
        · simp only [f, if_pos ht]; rw [eff_arena hA (lt_grid hb ht)]; exact hP.2.1
        · simp only [f, if_neg ht]; exact hP.2.1
      · by_cases ht : t < bd
        · simp only [f, if_pos ht]
          exact eff_keep hA (lt_grid hb hP.1) (lt_grid hb ht) (by omega) hP.2.2.1
        · simp only [f, if_neg ht]; exact hP.2.2.1
      · by_cases ht : t < bd
        · simp only [f, if_pos ht]; rw [eff_nan]; exact hP.2.2.2
        · simp only [f, if_neg ht]; exact hP.2.2.2)
    (List.range bd) _ _ List.nodup_range
    (fun t ht => ⟨List.mem_range.mp ht, harena, hpre t (List.mem_range.mp ht), hnan⟩) hfold
  simp only at hmem
  show { w''.mem with arena := w.mem.arena } = _
  rw [hmem, mem_arena_self _ (by
    rw [foldl_congr_mem _ (fun m t => E.eff (b * bd + t) m) _ _ (fun t ht m => by
        simp only [f, if_pos (List.mem_range.mp ht)])]
    rw [← List.foldl_map (f := (b * bd + ·)) (g := fun m j => E.eff j m),
      foldl_eff_arena hA _ _ (fun j hj => by
        obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj; exact lt_grid hb (List.mem_range.mp ht))])]
  exact foldl_congr_mem _ _ _ _ (fun t ht m => by simp only [f, if_pos (List.mem_range.mp ht)])

/-- Memory after blocks run in order is each block's effect in turn, where
    each block's effect keeps what the later blocks need. -/
theorem blocksRun_mem {k' : Kernel} {bd' : Nat} {params' : List V} {hw : Hw} (P : Nat → Mem → Prop)
    (F : Nat → Mem → Mem)
    (hrun : ∀ b s s', P b s.mem → BlockRuns k' b bd' params' hw s s' → s'.mem = F b s.mem)
    (hkeep : ∀ b b' m, b ≠ b' → P b' m → P b' (F b m)) :
    ∀ (c : Nat) (s s' : World), (∀ b ∈ List.range c, P b s.mem) → BlocksRun k' bd' params' hw c s s' →
      s'.mem = (List.range c).foldl (fun m b => F b m) s.mem := by
  intro c
  induction c with
  | zero => intro s s' _ h; subst h; rfl
  | succ c ih =>
      intro s s' hP h
      obtain ⟨v, hv, hb⟩ := h
      have hv' := ih s v (fun b hb' => hP b (List.mem_range.mpr (by have := List.mem_range.mp hb'; omega))) hv
      have key : ∀ (l : List Nat) m, (∀ b ∈ l, b ≠ c) → P c m → P c (l.foldl (fun m b => F b m) m) := by
        intro l
        induction l with
        | nil => intro m _ h; exact h
        | cons a l ihl =>
            intro m hl h
            exact ihl _ (fun b hb => hl b (List.mem_cons_of_mem _ hb)) (hkeep a c m (hl a List.mem_cons_self) h)
      have hPc : P c v.mem := by
        rw [hv']
        exact key _ _ (fun b hb => by have := List.mem_range.mp hb; omega) (hP c (List.mem_range.mpr (by omega)))
      rw [hrun c v s' hPc hb, hv', List.range_succ, List.foldl_append]
      rfl

/-- **What a launch leaves, from its blocks**: after `grid` blocks of `bd`
    threads, every active thread's word holds its value. -/
theorem launch_blocks (hA : E.Apart (grid * bd)) (hblk : Blocks k params grid bd E)
    (bufs bufs' : Nat → ByteArray) (hpre : ∀ j, j < grid * bd → E.Pre (kmem bufs) j)
    (hval : ∀ j, j < grid * bd → E.active j → (E.val j).toNat < 2 ^ 32)
    (hr : LaunchRuns k grid bd params bufs bufs') :
    ∀ j, j < grid * bd → E.active j → ({ kmem bufs with bufs := bufs' } : Mem).load (E.out j) 4 = some (E.val j) := by
  intro j hj ha
  have har0 : (kmem bufs).arena = ByteArray.empty := rfl
  obtain ⟨w', hfold, rfl⟩ := hr
  let G (b : Nat) (m : Mem) : Mem := (List.range bd).foldl (fun m t => E.eff (b * bd + t) m) m
  let F (b : Nat) (m : Mem) : Mem := if b < grid then G b m else m
  have hG : ∀ b m, G b m = ((List.range bd).map (b * bd + ·)).foldl (fun m j => E.eff j m) m := by
    intro b m; simp only [G, List.foldl_map]
  have hmem := blocksRun_mem (k' := k) (bd' := bd) (params' := params) (hw := Hw.rounded)
    (fun b m => b < grid ∧ m.arena = ByteArray.empty ∧ m.nan = gpuNaN ∧ ∀ t, t < bd → E.Pre m (b * bd + t))
    F
    (fun b s s' hP h => by
      rw [hblk b hP.1 _ _ hP.2.1 hP.2.2.1 hP.2.2.2 h]
      simp only [F, G, if_pos hP.1])
    (fun b b' m hne hP => by
      obtain ⟨hb', har, hnn, hpj⟩ := hP
      by_cases hb : b < grid
      · simp only [F, if_pos hb]
        have hl : ∀ j ∈ (List.range bd).map (b * bd + ·), j < grid * bd := fun j hj => by
          obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj; exact lt_grid hb (List.mem_range.mp ht)
        refine ⟨hb', by rw [hG, foldl_eff_arena hA _ _ hl]; exact har, by rw [hG, foldl_eff_nan]; exact hnn,
          fun t' ht' => ?_⟩
        rw [hG]
        refine foldl_eff_keep hA (lt_grid hb' ht') _ _ ?_ hl (hpj t' ht')
        intro hm
        obtain ⟨t, ht, e⟩ := List.mem_map.mp hm
        have ht := List.mem_range.mp ht
        rcases Nat.lt_or_gt_of_ne hne with h | h
        · have : (b + 1) * bd ≤ b' * bd := Nat.mul_le_mul_right _ h
          rw [Nat.succ_mul] at this; omega
        · have : (b' + 1) * bd ≤ b * bd := Nat.mul_le_mul_right _ h
          rw [Nat.succ_mul] at this; omega
      · simp only [F, if_neg hb]; exact ⟨hb', har, hnn, hpj⟩)
    grid _ w'
    (fun b hb => ⟨List.mem_range.mp hb, har0, rfl, fun t ht => hpre _ (lt_grid (List.mem_range.mp hb) ht)⟩) hfold
  obtain ⟨i, o, hd⟩ := hA.buf j hj ha
  refine (load_bufs (m₁ := ({ kmem bufs with bufs := w'.mem.bufs } : Mem)) (m₂ := w'.mem) (n := 4) hd rfl).trans ?_
  rw [hmem]
  show ((List.range grid).foldl (fun m b => F b m) (kmem bufs)).load (E.out j) 4 = _
  rw [show (List.range grid).foldl (fun m b => F b m) (kmem bufs) =
      (List.range grid).foldl (fun m b => G b m) (kmem bufs) from
    foldl_congr_mem _ _ _ _ (fun b hb m => if_pos (List.mem_range.mp hb))]
  simp only [G]
  rw [foldl_grid (fun j m => E.eff j m) bd grid (kmem bufs)]
  rw [show (List.range (grid * bd)).foldl (fun m j => E.eff j m) (kmem bufs) =
      ((List.range (grid * bd)).filter E.active).foldl (fun m j => put E.out E.val j m) (kmem bufs) by
    rw [← foldl_put_filter]; rfl]
  refine load_puts _ _ _ _ (List.nodup_range.filter _)
    (fun x hx y hy hxy => ?_) (fun x hx => ?_) j (by simp [hj, ha])
  · simp at hx hy; exact hA.outs x y hx.1 hy.1 hxy hx.2 hy.2
  · simp at hx; exact ⟨(hpre x hx.1 hx.2).2 _, hval x hx.1 hx.2⟩

/-- **What a launch of an elementwise kernel leaves**: after `grid` blocks of
    `bd` threads, every active thread's word holds its value. -/
theorem launch_spec (hsm : k.smem = 0) (hA : E.Apart (grid * bd)) (hth : Threads k params grid bd E)
    (bufs bufs' : Nat → ByteArray) (hpre : ∀ j, j < grid * bd → E.Pre (kmem bufs) j)
    (hval : ∀ j, j < grid * bd → E.active j → (E.val j).toNat < 2 ^ 32)
    (hr : LaunchRuns k grid bd params bufs bufs') :
    ∀ j, j < grid * bd → E.active j → ({ kmem bufs with bufs := bufs' } : Mem).load (E.out j) 4 = some (E.val j) :=
  launch_blocks hA (fun _ hb w _ harena hnan hpre hr => block_mem hA hth hb (by rw [harena, hsm]; rfl) hnan hpre hr)
    bufs bufs' hpre hval hr

end Launch

end Elem

end AlgorithmLib.Kernel
