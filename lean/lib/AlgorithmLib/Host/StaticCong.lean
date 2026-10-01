module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.StaticCong` — contracts that cannot tell data apart

Two worlds *agree up to data* (`Same K`) when they have the same memory layout,
the same bytes in the ranges `K`, the same device but for the contents of its
buffers, and kernels that keep buffer lengths. For each foreign call
`Host.Static` makes against its canonical world, `cong_*` proves that two such
worlds both refuse it or both accept it, answer the same, and still agree after
--- on `K`, less what the call overwrote, plus what it wrote the same on both.
A call that reads memory to decide what to do (a launch reads its kernel text
and bindings) is covered only where `K` holds those bytes.
-/

namespace AlgorithmLib.HProg.Static

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- Worlds that agree up to data
-- ---------------------------------------------------------------------------

/-- The bytes replaced by zeros, the length kept. -/
def zeroed (b : ByteArray) : ByteArray := ⟨Array.replicate b.size 0⟩

/-- The device with every buffer's contents forgotten, its length kept. -/
def Dev.erase (d : Dev) : Dev := { d with bufs := d.bufs.map (·.map zeroed) }

/-- Host byte ranges, as `(region, offset, length)`. -/
abbrev Known := List (Region × Nat × Nat)

/-- Two memories that differ at most in their bytes. -/
structure MemSame (m₁ m₂ : Mem) : Prop where
  size : ∀ r, (m₁.region r).size = (m₂.region r).size
  live : m₁.pinnedLive = m₂.pinnedLive
  busy : m₁.busy = m₂.busy
  frozen : m₁.frozen = m₂.frozen

/-- The bytes of `K` are the same in both. -/
def Agree (K : Known) (m₁ m₂ : Mem) : Prop :=
  ∀ r o n, (r, o, n) ∈ K → ∀ i < n, (m₁.region r).get! (o + i) = (m₂.region r).get! (o + i)

/-- A kernel oracle that hands back buffers of the lengths it was given. -/
def KeepsSize (k : Launch → List ByteArray → List ByteArray) : Prop :=
  ∀ l bs, (k l bs).map ByteArray.size = bs.map ByteArray.size

/-- **Two worlds that agree up to data**: the same memory layout and the same
    bytes in `K`, the same device but for the contents of its buffers, and
    kernels that keep lengths. -/
structure Same (K : Known) (w₁ w₂ : World) : Prop where
  mem : MemSame w₁.mem w₂.mem
  agree : Agree K w₁.mem w₂.mem
  dev : Dev.erase w₁.dev = Dev.erase w₂.dev
  kernel₁ : KeepsSize w₁.kernel
  kernel₂ : KeepsSize w₂.kernel
  cudaDevice : w₁.cudaDevice = w₂.cudaDevice

/-- Two answers of a contract, from worlds that agree up to data: both refuse,
    or both answer the same and still agree, on `K`. -/
def ResSame (K : Known) : Option (Option V × World) → Option (Option V × World) → Prop
  | none, none => True
  | some (r₁, w₁), some (r₂, w₂) => r₁ = r₂ ∧ Same K w₁ w₂
  | _, _ => False

/-- The same for a contract's device half. -/
def DevRes : Option (Option V × Dev) → Option (Option V × Dev) → Prop
  | none, none => True
  | some (r₁, d₁), some (r₂, d₂) => r₁ = r₂ ∧ Dev.erase d₁ = Dev.erase d₂
  | _, _ => False

theorem MemSame.refl (m : Mem) : MemSame m m := ⟨fun _ => rfl, rfl, rfl, rfl⟩

theorem MemSame.symm {m₁ m₂ : Mem} (h : MemSame m₁ m₂) : MemSame m₂ m₁ :=
  ⟨fun r => (h.size r).symm, h.live.symm, h.busy.symm, h.frozen.symm⟩

theorem Agree.mono {K K' : Known} {m₁ m₂ : Mem} (h : Agree K m₁ m₂) (hs : ∀ x ∈ K', x ∈ K) :
    Agree K' m₁ m₂ := fun r o n hx => h r o n (hs _ hx)

theorem Agree.nil (m₁ m₂ : Mem) : Agree [] m₁ m₂ := fun _ _ _ hx => by cases hx

theorem MemSame.retire {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (c : Clock) :
    MemSame (m₁.retire c) (m₂.retire c) :=
  ⟨fun r => by have := h.size r; cases r <;> exact this, h.live, by simp only [Mem.retire, h.busy], h.frozen⟩

theorem Agree.retire {K : Known} {m₁ m₂ : Mem} (h : Agree K m₁ m₂) (c₁ c₂ : Clock) :
    Agree K (m₁.retire c₁) (m₂.retire c₂) := by
  intro r o n hx i hi
  have := h r o n hx i hi
  cases r <;> exact this

-- The device's fields other than the buffers survive erasing.
theorem erase_live {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.live = d₂.live :=
  (congrArg Dev.live h : (Dev.erase _).live = (Dev.erase _).live)
theorem erase_race {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.race = d₂.race :=
  (congrArg Dev.race h : (Dev.erase _).race = (Dev.erase _).race)
theorem erase_streams {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.streams = d₂.streams :=
  (congrArg Dev.streams h : (Dev.erase _).streams = (Dev.erase _).streams)
theorem erase_events {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.events = d₂.events :=
  (congrArg Dev.events h : (Dev.erase _).events = (Dev.erase _).events)
theorem erase_capture {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.capture = d₂.capture :=
  (congrArg Dev.capture h : (Dev.erase _).capture = (Dev.erase _).capture)
theorem erase_pinned {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.pinned = d₂.pinned :=
  (congrArg Dev.pinned h : (Dev.erase _).pinned = (Dev.erase _).pinned)
theorem erase_graphs {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.graphs = d₂.graphs :=
  (congrArg Dev.graphs h : (Dev.erase _).graphs = (Dev.erase _).graphs)
theorem erase_drvInit {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.drvInit = d₂.drvInit :=
  (congrArg Dev.drvInit h : (Dev.erase _).drvInit = (Dev.erase _).drvInit)
theorem erase_retained {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.retained = d₂.retained :=
  (congrArg Dev.retained h : (Dev.erase _).retained = (Dev.erase _).retained)
theorem erase_modules {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.modules = d₂.modules :=
  (congrArg Dev.modules h : (Dev.erase _).modules = (Dev.erase _).modules)
theorem erase_funcs {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.funcs = d₂.funcs :=
  (congrArg Dev.funcs h : (Dev.erase _).funcs = (Dev.erase _).funcs)
theorem erase_blas {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.blas = d₂.blas :=
  (congrArg Dev.blas h : (Dev.erase _).blas = (Dev.erase _).blas)
theorem erase_execs {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) : d₁.execs = d₂.execs :=
  (congrArg Dev.execs h : (Dev.erase _).execs = (Dev.erase _).execs)
theorem erase_bufs {d₁ d₂ : Dev} (h : Dev.erase d₁ = Dev.erase d₂) :
    d₁.bufs.map (·.map zeroed) = d₂.bufs.map (·.map zeroed) :=
  (congrArg Dev.bufs h : (Dev.erase _).bufs = (Dev.erase _).bufs)

theorem devOnly_same {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    {k₁ k₂ : Option (Option V × Dev)} (hk : DevRes k₁ k₂) :
    ResSame K (devOnly w₁ k₁) (devOnly w₂ k₂) := by
  match k₁, k₂, hk with
  | none, none, _ => trivial
  | some (r₁, d₁), some (r₂, d₂), ⟨hr, hd⟩ =>
    refine ⟨hr, ?_, ?_, hd, hs.kernel₁, hs.kernel₂, hs.cudaDevice⟩
    · have : d₁.race = d₂.race :=
  (congrArg Dev.race hd : (Dev.erase _).race = (Dev.erase _).race)
      simp only [this]; exact hs.mem.retire _
    · exact hs.agree.retire _ _

/-- Close `Dev.erase d₁' = Dev.erase d₂'` where each side updates its device
    the same way, from `hd : Dev.erase d₁ = Dev.erase d₂`. -/
macro "erase_eq " hd:term : tactic => `(tactic| (
  simp only [Dev.erase, Array.map_push, erase_live $hd, erase_race $hd, erase_streams $hd,
    erase_events $hd, erase_capture $hd, erase_pinned $hd, erase_graphs $hd, erase_bufs $hd,
    erase_drvInit $hd, erase_retained $hd, erase_modules $hd, erase_funcs $hd,
    erase_blas $hd, erase_execs $hd]))

theorem cudaCtxOk_same {w₁ w₂ : World} (h : Dev.erase w₁.dev = Dev.erase w₂.dev) (ctx : UInt64) :
    cudaCtxOk w₁ ctx = cudaCtxOk w₂ ctx := by
  simp only [cudaCtxOk, erase_live h]

-- ---------------------------------------------------------------------------
-- Memory
-- ---------------------------------------------------------------------------

/-- Whether `K` holds `n` bytes from `off` in region `r`. -/
def covers (K : Known) (r : Region) (off n : Nat) : Bool :=
  K.any fun (r', o, l) => decide (r' = r) && decide (o ≤ off) && decide (off + n ≤ o + l)

/-- Whether `K` holds the `n` bytes a load at `a` reads. An address no region
    decodes reads nothing. -/
def coversAt (K : Known) (a : UInt64) (n : Nat) : Bool :=
  match decodeAddr a with
  | some (r, off) => covers K r off n
  | none => true

theorem covers_agree {K : Known} {m₁ m₂ : Mem} (h : Agree K m₁ m₂) {r : Region} {off n : Nat}
    (hc : covers K r off n = true) : ∀ i < n, (m₁.region r).get! (off + i) = (m₂.region r).get! (off + i) := by
  intro i hi
  simp only [covers, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨r', o, l⟩, hx, ⟨rfl, h1⟩, h2⟩ := hc
  dsimp only at h1 h2
  have := h r' o l hx (off + i - o) (by omega)
  rwa [show o + (off + i - o) = off + i by omega] at this

theorem MemSame.readable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.readable r off n = m₂.readable r off n := by
  simp only [Mem.readable, h.frozen, h.live, h.busy]

theorem MemSame.reachable {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) (off n : Nat) :
    m₁.reachable r off n = m₂.reachable r off n := by
  simp only [Mem.reachable, h.frozen, h.live, h.busy]

theorem load_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) :
    (m₁.load a n).isSome = (m₂.load a n).isSome := by
  unfold Mem.load
  cases decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [bind, Option.bind, h.size r, h.readable r off n]
    split <;> rfl

theorem foldr_congr_mem {α β : Type} (l : List α) (f g : α → β → β) (b : β)
    (h : ∀ x ∈ l, ∀ acc, f x acc = g x acc) : l.foldr f b = l.foldr g b := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldr_cons]
    rw [ih fun y hy => h y (List.mem_cons_of_mem x hy), h x (List.mem_cons_self ..)]

theorem load_eq {K : Known} {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂) {a : UInt64}
    {n : Nat} (hc : coversAt K a n = true) : m₁.load a n = m₂.load a n := by
  unfold Mem.load
  unfold coversAt at hc
  cases hd : decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    rw [hd] at hc
    have hb := covers_agree hk hc
    simp only [bind, Option.bind, h.size r, h.readable r off n]
    split
    · rfl
    · congr 1
      exact foldr_congr_mem _ _ _ _ fun i hi acc => by rw [hb i (List.mem_range.mp hi)]

theorem foldlM_isSome {α β γ : Type} (f : α → γ → Option α) (g : β → γ → Option β)
    (hfg : ∀ a b x, (f a x).isSome = (g b x).isSome) :
    ∀ (l : List γ) (a : α) (b : β), (l.foldlM f a).isSome = (l.foldlM g b).isSome
  | [], _, _ => rfl
  | x :: xs, a, b => by
    simp only [List.foldlM_cons]
    have := hfg a b x
    cases hf : f a x <;> cases hg : g b x <;> rw [hf, hg] at this <;> simp at this ⊢
    exact foldlM_isSome f g hfg xs _ _

theorem copyOut_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) :
    (copyOut m₁ a n).isSome = (copyOut m₂ a n).isSome := by
  unfold copyOut
  apply foldlM_isSome
  intro acc acc' i
  have := load_isSome h (a + UInt64.ofNat i) 1
  cases h1 : m₁.load (a + UInt64.ofNat i) 1 <;> cases h2 : m₂.load (a + UInt64.ofNat i) 1 <;>
    rw [h1, h2] at this <;> simp_all

theorem set!_size (b : ByteArray) (i : Nat) (v : UInt8) : (b.set! i v).size = b.size := by
  cases b; simp [ByteArray.set!, ByteArray.size, Array.set!_eq_setIfInBounds]

theorem get!_set!_ne (b : ByteArray) (i j : Nat) (v : UInt8) (h : i ≠ j) :
    (b.set! i v).get! j = b.get! j := by
  cases b with | mk bs =>
  simp [ByteArray.set!, ByteArray.get!, Array.set!_eq_setIfInBounds, getElem!_def, h]

theorem get!_set!_same (b : ByteArray) (i : Nat) (v : UInt8) (h : i < b.size) :
    (b.set! i v).get! i = v := by
  cases b with | mk bs =>
  simp only [ByteArray.size] at h
  simp [ByteArray.set!, ByteArray.get!, Array.set!_eq_setIfInBounds, h]

/-- The bytes a run of `set!` from `off` leaves: the written ones inside the
    range, the old ones outside it. -/
theorem get!_setRun (bs : ByteArray) (off : Nat) (f : Nat → UInt8) :
    ∀ (n j : Nat), off + n ≤ bs.size →
      ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).get! j
        = if off ≤ j ∧ j < off + n then f (j - off) else bs.get! j := by
  intro n
  induction n with
  | zero => intro j _; simp only [List.range_zero, List.foldl_nil]; rw [if_neg (by omega)]
  | succ n ih =>
    intro j hn
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    by_cases hj : j = off + n
    · subst hj
      have hsz : ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).size
          = bs.size := by
        clear ih
        induction n with
        | zero => rfl
        | succ n ih2 =>
          rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
            set!_size, ih2 (by omega)]
      rw [get!_set!_same _ _ _ (by rw [hsz]; omega)]
      simp
    · rw [get!_set!_ne _ _ _ _ (Ne.symm hj), ih j (by omega)]
      by_cases h1 : off ≤ j ∧ j < off + n
      · rw [if_pos h1, if_pos ⟨h1.1, by omega⟩]
      · rw [if_neg h1, if_neg (by omega)]

theorem size_setRun (bs : ByteArray) (off : Nat) (f : Nat → UInt8) :
    ∀ n, ((List.range n).foldl (fun (b : ByteArray) i => b.set! (off + i) (f i)) bs).size = bs.size
  | 0 => rfl
  | n + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
      set!_size, size_setRun bs off f n]

theorem region_setRegion_self (m : Mem) (r : Region) (b : ByteArray) : (m.setRegion r b).region r = b := by
  cases r <;> rfl

theorem region_setRegion_ne (m : Mem) {r r' : Region} (b : ByteArray) (h : r' ≠ r) :
    (m.setRegion r b).region r' = m.region r' := by
  cases r <;> cases r' <;> first | rfl | exact absurd rfl h

/-- What a store does to memory, region by region. -/
theorem store_some {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    ∃ r off, decodeAddr a = some (r, off) ∧ off + n ≤ (m.region r).size ∧
      m' = m.setRegion r ((List.range n).foldl (fun (b : ByteArray) i =>
        b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) (m.region r)) := by
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r, off⟩ := p
    rw [hd] at h
    simp only [bind, Option.bind] at h
    split at h
    · cases h
    · rename_i hc
      injection h with h
      refine ⟨r, off, rfl, ?_, h.symm⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hc
      omega

theorem store_isSome {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (a : UInt64) (n : Nat) (v₁ v₂ : UInt64) :
    (m₁.store a n v₁).isSome = (m₂.store a n v₂).isSome := by
  unfold Mem.store
  cases decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [bind, Option.bind, h.size r, h.reachable r off n]
    split <;> rfl

theorem MemSame.setRegion {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (r : Region) {b₁ b₂ : ByteArray}
    (hb : b₁.size = b₂.size) : MemSame (m₁.setRegion r b₁) (m₂.setRegion r b₂) := by
  refine ⟨fun r' => ?_, ?_, ?_, ?_⟩
  · by_cases hr : r' = r
    · subst hr; rw [region_setRegion_self, region_setRegion_self, hb]
    · rw [region_setRegion_ne _ _ hr, region_setRegion_ne _ _ hr, h.size r']
  all_goals cases r <;> first | exact h.live | exact h.busy | exact h.frozen

/-- A store to one side only: the layout still agrees, and so do the bytes it
    did not touch. -/
theorem store_left {K : Known} {m₁ m₂ m₁' : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂)
    {a : UInt64} {n : Nat} {v : UInt64} (hs : m₁.store a n v = some m₁') :
    ∃ r off, decodeAddr a = some (r, off) ∧ MemSame m₁' m₂ ∧
      Agree (K.filter fun (r', o, l) => !(decide (r' = r) && decide (o < off + n) && decide (off < o + l)))
        m₁' m₂ := by
  obtain ⟨r, off, hd, hn, rfl⟩ := store_some hs
  refine ⟨r, off, hd, ?_, ?_⟩
  · have := h.setRegion r (b₁ := (List.range n).foldl (fun (b : ByteArray) i =>
        b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) (m₁.region r))
        (b₂ := m₂.region r) (by rw [size_setRun, h.size r])
    have e : m₂.setRegion r (m₂.region r) = m₂ := by cases r <;> rfl
    rwa [e] at this
  · intro r' o l hx i hi
    simp only [List.mem_filter, Bool.not_eq_true', Bool.and_eq_false_iff,
      decide_eq_false_iff_not, Nat.not_lt] at hx
    obtain ⟨hx, hdis⟩ := hx
    have hb := hk r' o l hx i hi
    by_cases hr : r' = r
    · subst hr
      have hd' : ¬ o < off + n ∨ ¬ off < o + l := by simpa using hdis
      rw [region_setRegion_self, get!_setRun _ _ _ _ _ hn, if_neg (by omega)]
      exact hb
    · rw [region_setRegion_ne _ _ hr]; exact hb

/-- The same store on both sides: the layout still agrees, and the stored bytes
    join what agrees. -/
theorem store_both {K : Known} {m₁ m₂ m₁' m₂' : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂)
    {a : UInt64} {n : Nat} {v : UInt64} (hs₁ : m₁.store a n v = some m₁')
    (hs₂ : m₂.store a n v = some m₂') :
    ∃ r off, decodeAddr a = some (r, off) ∧ MemSame m₁' m₂' ∧ Agree ((r, off, n) :: K) m₁' m₂' := by
  obtain ⟨r, off, hd, hn, rfl⟩ := store_some hs₁
  obtain ⟨r2, off2, hd2, hn2, rfl⟩ := store_some hs₂
  rw [hd] at hd2; injection hd2 with hd2; injection hd2 with e1 e2; subst e1; subst e2
  refine ⟨r, off, hd, h.setRegion r (by rw [size_setRun, size_setRun, h.size r]), ?_⟩
  intro r' o l hx i hi
  by_cases hr : r' = r
  · subst hr
    rw [region_setRegion_self, region_setRegion_self, get!_setRun _ _ _ _ _ hn,
      get!_setRun _ _ _ _ _ hn2]
    split
    · rfl
    · rcases List.mem_cons.mp hx with hx | hx
      · injection hx with _ hx; injection hx with e1 e2; subst e1; subst e2; omega
      · exact hk r' o l hx i hi
  · rw [region_setRegion_ne _ _ hr, region_setRegion_ne _ _ hr]
    rcases List.mem_cons.mp hx with hx | hx
    · injection hx with e; exact absurd e hr
    · exact hk r' o l hx i hi

theorem MemSame.trans {m₁ m₂ m₃ : Mem} (h₁ : MemSame m₁ m₂) (h₂ : MemSame m₂ m₃) : MemSame m₁ m₃ :=
  ⟨fun r => (h₁.size r).trans (h₂.size r), h₁.live.trans h₂.live, h₁.busy.trans h₂.busy,
    h₁.frozen.trans h₂.frozen⟩

/-- `K` without what overlaps `n` bytes from `off` in region `r`. -/
def kill (r : Region) (off n : Nat) (K : Known) : Known :=
  K.filter fun (r', o, l) => !(decide (r' = r) && decide (o < off + n) && decide (off < o + l))

/-- `m'` is `m` with at most `n` bytes from `off` in region `r` changed. -/
structure StoreFrame (m m' : Mem) (r : Region) (off n : Nat) : Prop where
  same : MemSame m m'
  other : ∀ r', r' ≠ r → m'.region r' = m.region r'
  outside : ∀ j, j < off ∨ off + n ≤ j → (m'.region r).get! j = (m.region r).get! j

theorem StoreFrame.agree {K : Known} {m m' m₂ : Mem} {r : Region} {off n : Nat}
    (hf : StoreFrame m m' r off n) (h : MemSame m m₂) (hk : Agree K m m₂) :
    MemSame m' m₂ ∧ Agree (kill r off n K) m' m₂ := by
  refine ⟨hf.same.symm.trans h, ?_⟩
  intro r' o l hx i hi
  simp only [kill, List.mem_filter, Bool.not_eq_true', Bool.and_eq_false_iff,
    decide_eq_false_iff_not, Nat.not_lt] at hx
  obtain ⟨hx, hdis⟩ := hx
  have hb := hk r' o l hx i hi
  by_cases hr : r' = r
  · subst hr
    have hd' : ¬ o < off + n ∨ ¬ off < o + l := by simpa using hdis
    rw [hf.outside _ (by omega)]; exact hb
  · rw [hf.other _ hr]; exact hb

theorem StoreFrame.trans {m m' m'' : Mem} {r : Region} {off n n' : Nat}
    (h₁ : StoreFrame m m' r off n) (h₂ : StoreFrame m' m'' r off n') (hle : n ≤ n') :
    StoreFrame m m'' r off n' :=
  ⟨h₁.same.trans h₂.same, fun r' hr => (h₂.other r' hr).trans (h₁.other r' hr),
    fun j hj => (h₂.outside j hj).trans (h₁.outside j (by omega))⟩

theorem StoreFrame.refl (m : Mem) (r : Region) (off n : Nat) : StoreFrame m m r off n :=
  ⟨MemSame.refl m, fun _ _ => rfl, fun _ _ => rfl⟩

theorem StoreFrame.widen {m m' : Mem} {r : Region} {off n off' n' : Nat}
    (h : StoreFrame m m' r off n) (h1 : off' ≤ off) (h2 : off + n ≤ off' + n') :
    StoreFrame m m' r off' n' :=
  ⟨h.same, h.other, fun j hj => h.outside j (by omega)⟩

theorem store_frame {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    ∃ r off, decodeAddr a = some (r, off) ∧ StoreFrame m m' r off n := by
  obtain ⟨r, off, hd, hn, rfl⟩ := store_some h
  refine ⟨r, off, hd, ⟨?_, fun r' hr => region_setRegion_ne _ _ hr, fun j hj => ?_⟩⟩
  · have := (MemSame.refl m).setRegion r (b₁ := m.region r)
      (b₂ := (List.range n).foldl (fun (b : ByteArray) i =>
        b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) (m.region r))
      (by rw [size_setRun])
    have e : m.setRegion r (m.region r) = m := by cases r <;> rfl
    rwa [e] at this
  · rw [region_setRegion_self, get!_setRun _ _ _ _ _ hn, if_neg (by omega)]

-- Addresses: a region's base plus an offset inside its span decodes to both.

theorem regionSpan_toNat : regionSpan.toNat = 2 ^ 36 := by decide

theorem regionBase_toNat (r : Region) : (regionBase r).toNat + regionSpan.toNat ≤ 2 ^ 39 := by
  cases r <;> decide

theorem addrCond (b a : UInt64) :
    (decide (a ≥ b) && decide (a - b < regionSpan))
      = decide (b.toNat ≤ a.toNat ∧ a.toNat < b.toNat + 2 ^ 36) := by
  rw [Bool.eq_iff_iff, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, decide_eq_true_eq]
  simp only [ge_iff_le, UInt64.le_iff_toNat_le, UInt64.lt_iff_toNat_lt, regionSpan_toNat]
  constructor
  · rintro ⟨h1, h2⟩
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)] at h2; omega
  · rintro ⟨h1, h2⟩
    refine ⟨h1, ?_⟩
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr h1)]; omega

theorem decodeAddr_base (r : Region) (k : Nat) (hk : k < regionSpan.toNat) :
    decodeAddr (regionBase r + UInt64.ofNat k) = some (r, k) := by
  rw [regionSpan_toNat] at hk
  have hb := regionBase_toNat r
  rw [regionSpan_toNat] at hb
  have hx : (regionBase r + UInt64.ofNat k).toNat = (regionBase r).toNat + k := by
    rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := k) (by omega),
      Nat.mod_eq_of_lt (by omega)]
  have hsub : ((regionBase r + UInt64.ofNat k) - regionBase r).toNat = k := by
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr (by omega)), hx]; omega
  unfold decodeAddr
  generalize regionBase r + UInt64.ofNat k = a at hx hsub
  simp only [List.findSome?, addrCond]
  have e1 : (regionBase .arena).toNat = 2 ^ 36 := by decide
  have e2 : (regionBase .data).toNat = 2 * 2 ^ 36 := by decide
  have e3 : (regionBase .out).toNat = 3 * 2 ^ 36 := by decide
  have e4 : (regionBase .pinned).toNat = 4 * 2 ^ 36 := by decide
  cases r <;> rw [e1, e2, e3, e4] <;> simp only [e1, e2, e3, e4] at hx <;>
    simp (disch := omega) only [decide_eq_true, decide_eq_false, if_true, if_false,
      Bool.false_eq_true, hsub]

theorem decodeAddr_some {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a = regionBase r + UInt64.ofNat off ∧ off < regionSpan.toNat := by
  unfold decodeAddr at h
  simp only [List.findSome?, addrCond] at h
  have key : ∀ b : UInt64, decide (b.toNat ≤ a.toNat ∧ a.toNat < b.toNat + 2 ^ 36) = true →
      (a - b).toNat = off → a = b + UInt64.ofNat off ∧ off < regionSpan.toNat := by
    intro b hc ho
    simp only [decide_eq_true_eq] at hc
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hc.1)] at ho
    refine ⟨?_, by rw [regionSpan_toNat]; omega⟩
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := off) (by omega)]
    have := a.toNat_lt
    rw [Nat.mod_eq_of_lt (by omega)]; omega
  by_cases c1 : decide ((regionBase .arena).toNat ≤ a.toNat ∧ a.toNat < (regionBase .arena).toNat + 2 ^ 36) = true
  · rw [if_pos c1] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c1 h2
  rw [if_neg c1] at h
  by_cases c2 : decide ((regionBase .data).toNat ≤ a.toNat ∧ a.toNat < (regionBase .data).toNat + 2 ^ 36) = true
  · rw [if_pos c2] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c2 h2
  rw [if_neg c2] at h
  by_cases c3 : decide ((regionBase .out).toNat ≤ a.toNat ∧ a.toNat < (regionBase .out).toNat + 2 ^ 36) = true
  · rw [if_pos c3] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c3 h2
  rw [if_neg c3] at h
  by_cases c4 : decide ((regionBase .pinned).toNat ≤ a.toNat ∧ a.toNat < (regionBase .pinned).toNat + 2 ^ 36) = true
  · rw [if_pos c4] at h; simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, h2⟩ := h; exact key _ c4 h2
  rw [if_neg c4] at h
  cases h

theorem decodeAddr_add {a : UInt64} {r : Region} {off i : Nat} (h : decodeAddr a = some (r, off))
    (hi : off + i < regionSpan.toNat) : decodeAddr (a + UInt64.ofNat i) = some (r, off + i) := by
  obtain ⟨rfl, _⟩ := decodeAddr_some h
  rw [UInt64.add_assoc, ← UInt64.ofNat_add]
  exact decodeAddr_base r _ hi

theorem foldlM_rel {α β γ : Type} (R : α → β → Prop) (f : α → γ → Option α) (g : β → γ → Option β)
    (hfg : ∀ a b x b', R a b → g b x = some b' → ∃ a', f a x = some a' ∧ R a' b') :
    ∀ (l : List γ) (a : α) (b : β) (b' : β), R a b → l.foldlM g b = some b' →
      ∃ a', l.foldlM f a = some a' ∧ R a' b'
  | [], a, b, b', hr, h => by cases h; exact ⟨a, rfl, hr⟩
  | x :: xs, a, b, b', hr, h => by
    simp only [List.foldlM_cons] at h ⊢
    cases hg : g b x with
    | none => rw [hg] at h; cases h
    | some b1 =>
      rw [hg] at h
      obtain ⟨a1, hf, hr1⟩ := hfg a b x b1 hr hg
      rw [hf]
      exact foldlM_rel R f g hfg xs a1 b1 b' hr1 h

/-- A copy into memory writes the bytes from its address and nothing else. -/
theorem copyIn_frame {m m' : Mem} {a : UInt64} {src : ByteArray} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hn : off + src.size ≤ regionSpan.toNat)
    (h : copyIn m a src = some m') : StoreFrame m m' r off src.size := by
  unfold copyIn at h
  suffices ∀ n, n ≤ src.size → ∀ m', (List.range n).foldlM
      (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m = some m' →
      StoreFrame m m' r off n from this _ (Nat.le_refl _) m' h
  intro n
  induction n with
  | zero => intro _ m' h; cases h; exact StoreFrame.refl m r off 0
  | succ n ih =>
    intro hn' m' h
    rw [List.range_succ, List.foldlM_append] at h
    cases h1 : (List.range n).foldlM
        (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m with
    | none => rw [h1] at h; cases h
    | some m1 =>
      rw [h1] at h
      simp only [Option.bind_eq_bind, Option.bind_some, List.foldlM_cons, List.foldlM_nil] at h
      have hf1 := ih (by omega) m1 h1
      cases h2 : m1.store (a + UInt64.ofNat n) 1 (src.get! n).toUInt64 with
      | none => rw [h2] at h; cases h
      | some m2 =>
        rw [h2] at h; cases h
        obtain ⟨r2, off2, hd2, hf2⟩ := store_frame h2
        rw [decodeAddr_add hd (by omega)] at hd2
        injection hd2 with hd2; injection hd2 with e1 e2; subst e1; subst e2
        exact hf1.trans (hf2.widen (by omega) (by omega)) (by omega)

theorem copyIn_isSome {m₁ m₂ m₂' : Mem} (h : MemSame m₁ m₂) {a : UInt64} {s₁ s₂ : ByteArray}
    (hsz : s₁.size = s₂.size) (h2 : copyIn m₂ a s₂ = some m₂') :
    ∃ m₁', copyIn m₁ a s₁ = some m₁' ∧ MemSame m₁' m₂' := by
  unfold copyIn at h2 ⊢
  rw [hsz]
  refine foldlM_rel MemSame _ _ ?_ _ m₁ m₂ m₂' h h2
  intro x y i y' hxy hy
  have hs := store_isSome hxy (a + UInt64.ofNat i) 1 (s₁.get! i).toUInt64 (s₂.get! i).toUInt64
  rw [hy] at hs
  cases hx : x.store (a + UInt64.ofNat i) 1 (s₁.get! i).toUInt64 with
  | none => rw [hx] at hs; cases hs
  | some x' =>
    refine ⟨x', rfl, ?_⟩
    obtain ⟨_, _, _, hfx⟩ := store_frame hx
    obtain ⟨_, _, _, hfy⟩ := store_frame hy
    exact hfx.same.symm.trans (hxy.trans hfy.same)

theorem Agree.symm {K : Known} {m₁ m₂ : Mem} (h : Agree K m₁ m₂) : Agree K m₂ m₁ :=
  fun r o n hx i hi => (h r o n hx i hi).symm

theorem kill_sub (r : Region) (off n : Nat) (K : Known) : ∀ x ∈ kill r off n K, x ∈ K :=
  fun _ hx => (List.mem_filter.mp hx).1

/-- What a copy of `n` bytes into `a` leaves known: `K` without the copied
    range, or nothing where the range is not one region's. -/
def killCopy (a : UInt64) (n : Nat) (K : Known) : Known :=
  match decodeAddr a with
  | some (r, off) => if off + n ≤ regionSpan.toNat then kill r off n K else []
  | none => []

theorem copyIn_both {K : Known} {m₁ m₂ m₂' : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂)
    {a : UInt64} {s₁ s₂ : ByteArray} (hsz : s₁.size = s₂.size) (h2 : copyIn m₂ a s₂ = some m₂') :
    ∃ m₁', copyIn m₁ a s₁ = some m₁' ∧ MemSame m₁' m₂' ∧ Agree (killCopy a s₂.size K) m₁' m₂' := by
  obtain ⟨m₁', h1, hs'⟩ := copyIn_isSome h hsz h2
  refine ⟨m₁', h1, hs', ?_⟩
  unfold killCopy
  cases hd : decodeAddr a with
  | none => exact Agree.nil _ _
  | some p =>
    obtain ⟨r, off⟩ := p
    dsimp only
    split
    · rename_i hn
      have f1 := copyIn_frame hd (hsz ▸ hn) h1
      have f2 := copyIn_frame hd hn h2
      obtain ⟨hm1, ha1⟩ := f1.agree h hk
      obtain ⟨_, ha2⟩ := f2.agree hm1.symm ha1.symm
      rw [hsz] at ha2
      refine (ha2.symm).mono ?_
      intro x hx
      simp only [kill, List.mem_filter] at hx ⊢
      exact ⟨hx, hx.2⟩
    · exact Agree.nil _ _

theorem copyOut_size {m : Mem} {a : UInt64} {n : Nat} {b : ByteArray} (h : copyOut m a n = some b) :
    b.size = n := by
  unfold copyOut at h
  suffices ∀ (l : List Nat) (acc b : ByteArray), l.foldlM (fun (acc : ByteArray) i => do
      let x ← m.load (a + UInt64.ofNat i) 1
      pure (acc.push x.toUInt8)) acc = some b → b.size = acc.size + l.length by
    have := this _ _ _ h; simpa using this
  intro l
  induction l with
  | nil => intro acc b h; cases h; simp
  | cons x xs ih =>
    intro acc b h
    simp only [List.foldlM_cons] at h
    cases hl : m.load (a + UInt64.ofNat x) 1 with
    | none => simp [hl] at h
    | some v =>
      simp only [hl, Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h
      have := ih _ _ h
      have hp : (acc.push v.toUInt8).size = acc.size + 1 := by
        cases acc; simp [ByteArray.push, ByteArray.size]
      simp only [List.length_cons] at this ⊢
      omega

theorem copyOut_same {m₁ m₂ : Mem} (h : MemSame m₁ m₂) {a : UInt64} {n : Nat} {b₂ : ByteArray}
    (h2 : copyOut m₂ a n = some b₂) : ∃ b₁, copyOut m₁ a n = some b₁ ∧ b₁.size = b₂.size := by
  have hs := copyOut_isSome h a n
  rw [h2] at hs
  cases h1 : copyOut m₁ a n with
  | none => rw [h1] at hs; cases hs
  | some b₁ => exact ⟨b₁, rfl, by rw [copyOut_size h1, copyOut_size h2]⟩

-- The device's buffers, up to their contents.

theorem zeroed_size (b : ByteArray) : (zeroed b).size = b.size :=
  Array.size_replicate ..

theorem zeroed_eq_iff (b₁ b₂ : ByteArray) : zeroed b₁ = zeroed b₂ ↔ b₁.size = b₂.size := by
  constructor
  · intro h
    have := congrArg ByteArray.size h
    rwa [zeroed_size, zeroed_size] at this
  · intro h; unfold zeroed; rw [h]

theorem get?_erase (d : Dev) (id : Int) : (Dev.erase d).get? id = (d.get? id).map zeroed := by
  unfold Dev.get?
  split
  · rfl
  · show ((d.bufs.map (·.map zeroed))[id.toNat]?).join = ((d.bufs[id.toNat]?).join).map zeroed
    rw [Array.getElem?_map]
    cases d.bufs[id.toNat]? with
    | none => rfl
    | some x => cases x <;> rfl

/-- The buffer an id names, on two devices that agree up to contents: on both
    or on neither, with the same length. -/
theorem get?_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (id : Int) :
    (d₁.get? id = none ∧ d₂.get? id = none) ∨
      ∃ b₁ b₂, d₁.get? id = some b₁ ∧ d₂.get? id = some b₂ ∧ b₁.size = b₂.size := by
  have h := congrArg (fun d => Dev.get? d id) hd
  simp only [get?_erase] at h
  cases h1 : d₁.get? id <;> cases h2 : d₂.get? id <;> rw [h1, h2] at h <;> simp at h
  · exact Or.inl ⟨rfl, rfl⟩
  · exact Or.inr ⟨_, _, rfl, rfl, (zeroed_eq_iff _ _).mp h⟩

theorem erase_put (d : Dev) (id : Nat) (b : Option ByteArray) :
    Dev.erase (d.put id b) = { Dev.erase d with bufs := (Dev.erase d).bufs.setIfInBounds id (b.map zeroed) } := by
  simp [Dev.erase, Dev.put, Array.map_setIfInBounds]

theorem erase_put_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (id : Nat) {b₁ b₂ : ByteArray}
    (hb : b₁.size = b₂.size) : Dev.erase (d₁.put id (some b₁)) = Dev.erase (d₂.put id (some b₂)) := by
  simp only [erase_put, hd, Option.map_some, (zeroed_eq_iff _ _).mpr hb]

theorem erase_put_none {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (id : Nat) :
    Dev.erase (d₁.put id none) = Dev.erase (d₂.put id none) := by
  rw [erase_put, erase_put, hd]

theorem bufs_size_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) : d₁.bufs.size = d₂.bufs.size := by
  have := congrArg (fun d => d.bufs.size) hd
  simpa [Dev.erase] using this

theorem party?_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (sid : Int) :
    d₁.party? sid = d₂.party? sid := by
  simp only [Dev.party?, erase_streams hd]

theorem capturing_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (p : Nat) :
    d₁.capturing p = d₂.capturing p := by
  simp only [Dev.capturing, erase_capture hd]

theorem event?_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (e : Int) :
    d₁.event? e = d₂.event? e := by
  simp only [Dev.event?, erase_events hd]

-- ---------------------------------------------------------------------------
-- Per contract
-- ---------------------------------------------------------------------------

/-- Two outcomes of a device step. -/
def OptDev : Option Dev → Option Dev → Prop
  | none, none => True
  | some d₁, some d₂ => Dev.erase d₁ = Dev.erase d₂
  | _, _ => False

theorem syncWrite_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (id : Nat) {b₁ b₂ : ByteArray}
    (hb : b₁.size = b₂.size) : OptDev (d₁.syncWrite id b₁) (d₂.syncWrite id b₂) := by
  unfold Dev.syncWrite
  rw [erase_race hd]
  cases d₂.race.op defaultParty [] [id] with
  | none => trivial
  | some r =>
    show Dev.erase { (d₁.put id (some b₁)) with race := _ } = Dev.erase { (d₂.put id (some b₂)) with race := _ }
    have := erase_put_same hd id hb
    simp only [Dev.erase, Dev.put] at this ⊢
    simp only [Dev.mk.injEq] at this ⊢
    exact ⟨this.1, this.2.1, this.2.2.1, this.2.2.2.1, this.2.2.2.2.1, trivial, this.2.2.2.2.2.2⟩

theorem syncRead_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) (id : Nat) :
    OptDev (d₁.syncRead id) (d₂.syncRead id) := by
  unfold Dev.syncRead
  rw [erase_race hd]
  cases d₂.race.op defaultParty [id] [] with
  | none => trivial
  | some r => show Dev.erase { d₁ with race := _ } = Dev.erase { d₂ with race := _ }; erase_eq hd

theorem cong_cudaSync {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaSync bits w₁) (ffiCudaSync bits w₂) := by
  unfold ffiCudaSync
  apply devOnly_same hs
  have hd := hs.dev
  split
  · rw [cudaCtxOk_same hd]
    cases cudaCtxOk w₂ _ with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hd⟩
      · exact ⟨rfl, by erase_eq hd⟩
  · trivial

theorem cong_cudaCreateBuffer {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaCreateBuffer bits w₁) (ffiCudaCreateBuffer bits w₂) := by
  unfold ffiCudaCreateBuffer
  apply devOnly_same hs
  have hd := hs.dev
  rcases bits with _ | ⟨ctx, _ | ⟨size, _ | _⟩⟩ <;> try trivial
  simp only [cudaCtxOk_same hd, bufs_size_same hd]
  split
  · exact ⟨rfl, hd⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hd⟩
      · exact ⟨rfl, by erase_eq hd⟩

theorem cong_cudaUpload {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaUpload bits w₁) (ffiCudaUpload bits w₂) := by
  unfold ffiCudaUpload
  apply devOnly_same hs
  have hd := hs.dev
  rcases bits with _ | ⟨ctx, _ | ⟨buf, _ | ⟨src, _ | ⟨size, _ | _⟩⟩⟩⟩ <;> try trivial
  simp only [cudaCtxOk_same hd]
  split
  · exact ⟨rfl, hd⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hd⟩
      · rcases get?_same hd (asI32 buf) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
        · simp only [h1, h2]; exact ⟨rfl, hd⟩
        · simp only [h1, h2, hb]
          split
          · trivial
          · have hi := copyOut_isSome hs.mem src size.toNat
            unfold readBytes
            cases h4 : copyOut w₂.mem src size.toNat with
            | none =>
              rw [h4] at hi
              cases h3 : copyOut w₁.mem src size.toNat with
              | none => trivial
              | some _ => rw [h3] at hi; cases hi
            | some y₂ =>
              obtain ⟨y₁, h3, hy⟩ := copyOut_same hs.mem h4
              simp only [h3, bind, Option.bind]
              have hw := syncWrite_same hd (asI32 buf).toNat hy
              cases e₂ : w₂.dev.syncWrite (asI32 buf).toNat y₂ with
              | none =>
                rw [e₂] at hw
                cases e₁ : w₁.dev.syncWrite (asI32 buf).toNat y₁ with
                | none => trivial
                | some _ => rw [e₁] at hw; cases hw
              | some d₂ =>
                rw [e₂] at hw
                cases e₁ : w₁.dev.syncWrite (asI32 buf).toNat y₁ with
                | none => rw [e₁] at hw; cases hw
                | some d₁ => rw [e₁] at hw; exact ⟨rfl, hw⟩

theorem cong_cudaFreeBuffer {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaFreeBuffer bits w₁) (ffiCudaFreeBuffer bits w₂) := by
  unfold ffiCudaFreeBuffer
  apply devOnly_same hs
  have hd := hs.dev
  rcases bits with _ | ⟨ctx, _ | ⟨buf, _ | _⟩⟩ <;> try trivial
  simp only [cudaCtxOk_same hd, erase_race hd]
  split
  · exact ⟨rfl, hd⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hd⟩
      · rcases get?_same hd (asI32 buf) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
        · simp only [h1, h2]; exact ⟨rfl, hd⟩
        · simp only [h1, h2]
          cases w₂.dev.race.op defaultParty [] [(asI32 buf).toNat] with
          | none => trivial
          | some r =>
            refine ⟨rfl, ?_⟩
            have := erase_put_none hd (asI32 buf).toNat
            simp only [Dev.erase, Dev.put] at this ⊢
            simp only [Dev.mk.injEq] at this ⊢
            exact ⟨this.1, this.2.1, this.2.2.1, this.2.2.2.1, this.2.2.2.2.1, trivial, this.2.2.2.2.2.2⟩

theorem Same.mono {K K' : Known} {w₁ w₂ : World} (h : Same K w₁ w₂) (hs : ∀ x ∈ K', x ∈ K) :
    Same K' w₁ w₂ := ⟨h.mem, h.agree.mono hs, h.dev, h.kernel₁, h.kernel₂, h.cudaDevice⟩

theorem killCopy_sub (a : UInt64) (n : Nat) (K : Known) : ∀ x ∈ killCopy a n K, x ∈ K := by
  intro x hx
  unfold killCopy at hx
  split at hx
  · split at hx
    · exact kill_sub _ _ _ _ _ hx
    · cases hx
  · cases hx

theorem copyIn_none {m₁ m₂ : Mem} (h : MemSame m₁ m₂) {a : UInt64} {s₁ s₂ : ByteArray}
    (hsz : s₁.size = s₂.size) (h2 : copyIn m₂ a s₂ = none) : copyIn m₁ a s₁ = none := by
  cases h1 : copyIn m₁ a s₁ with
  | none => rfl
  | some m₁' =>
    obtain ⟨m₂', h2', _⟩ := copyIn_isSome h.symm hsz.symm h1
    rw [h2] at h2'; cases h2'

theorem cong_cudaDownload {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (ctx buf dst size : UInt64) :
    ResSame (killCopy dst size.toNat K) (ffiCudaDownload [ctx, buf, dst, size] w₁)
      (ffiCudaDownload [ctx, buf, dst, size] w₂) := by
  unfold ffiCudaDownload
  have hd := hs.dev
  have hk := hs.mono (killCopy_sub dst size.toNat K)
  simp only [cudaCtxOk_same hd]
  split
  · exact ⟨rfl, hk⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hk⟩
      · rcases get?_same hd (asI32 buf) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
        · simp only [h1, h2]; exact ⟨rfl, hk⟩
        · simp only [h1, h2, hb]
          split
          · trivial
          · rename_i hsz
            have hsz' : b₂.size = size.toNat := by simpa using hsz
            have hr := syncRead_same hd (asI32 buf).toNat
            cases e₂ : w₂.dev.syncRead (asI32 buf).toNat with
            | none =>
              rw [e₂] at hr
              cases e₁ : w₁.dev.syncRead (asI32 buf).toNat with
              | none => trivial
              | some _ => rw [e₁] at hr; cases hr
            | some d₂ =>
              rw [e₂] at hr
              cases e₁ : w₁.dev.syncRead (asI32 buf).toNat with
              | none => rw [e₁] at hr; cases hr
              | some d₁ =>
                rw [e₁] at hr
                simp only [bind, Option.bind]
                cases c₂ : copyIn w₂.mem dst b₂ with
                | none => rw [copyIn_none hs.mem hb c₂]; trivial
                | some m₂ =>
                  obtain ⟨m₁, c₁, hm, ha⟩ := copyIn_both hs.mem hs.agree hb c₂
                  rw [c₁]
                  refine ⟨rfl, hm, ?_, hr, hs.kernel₁, hs.kernel₂, hs.cudaDevice⟩
                  rwa [hsz'] at ha

theorem cong_cudaDownloadOffset {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    (ctx buf off dst size : UInt64) :
    ResSame (killCopy dst size.toNat K) (ffiCudaDownloadOffset [ctx, buf, off, dst, size] w₁)
      (ffiCudaDownloadOffset [ctx, buf, off, dst, size] w₂) := by
  unfold ffiCudaDownloadOffset
  have hd := hs.dev
  have hk := hs.mono (killCopy_sub dst size.toNat K)
  simp only [cudaCtxOk_same hd]
  split
  · exact ⟨rfl, hk⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hk⟩
      · rcases get?_same hd (asI32 buf) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
        · simp only [h1, h2]; exact ⟨rfl, hk⟩
        · simp only [h1, h2, hb]
          split
          · exact ⟨rfl, hk⟩
          · rename_i hsz
            have hr := syncRead_same hd (asI32 buf).toNat
            cases e₂ : w₂.dev.syncRead (asI32 buf).toNat with
            | none =>
              rw [e₂] at hr
              cases e₁ : w₁.dev.syncRead (asI32 buf).toNat with
              | none => trivial
              | some _ => rw [e₁] at hr; cases hr
            | some d₂ =>
              rw [e₂] at hr
              cases e₁ : w₁.dev.syncRead (asI32 buf).toNat with
              | none => rw [e₁] at hr; cases hr
              | some d₁ =>
                rw [e₁] at hr
                simp only [bind, Option.bind]
                have hx : (b₁.extract off.toNat (off.toNat + size.toNat)).size
                    = (b₂.extract off.toNat (off.toNat + size.toNat)).size := by
                  rw [ByteArray.size_extract, ByteArray.size_extract, hb]
                have hx2 : (b₂.extract off.toNat (off.toNat + size.toNat)).size = size.toNat := by
                  rw [ByteArray.size_extract]; simp only [gt_iff_lt, Nat.not_lt] at hsz; omega
                cases c₂ : copyIn w₂.mem dst (b₂.extract off.toNat (off.toNat + size.toNat)) with
                | none => rw [copyIn_none hs.mem hx c₂]; trivial
                | some m₂ =>
                  obtain ⟨m₁, c₁, hm, ha⟩ := copyIn_both hs.mem hs.agree hx c₂
                  rw [c₁]
                  refine ⟨rfl, hm, ?_, hr, hs.kernel₁, hs.kernel₂, hs.cudaDevice⟩
                  rwa [hx2] at ha

/-- What `cudaInit` and `cudaCleanup` leave known: the slot they store. -/
def addSlot (slot : UInt64) (K : Known) : Known :=
  match decodeAddr slot with
  | some (r, off) => (r, off, 8) :: K
  | none => K

theorem store_slot {K : Known} {m₁ m₂ m₁' m₂' : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂)
    {slot v : UInt64} (h1 : m₁.store slot 8 v = some m₁') (h2 : m₂.store slot 8 v = some m₂') :
    MemSame m₁' m₂' ∧ Agree (addSlot slot K) m₁' m₂' := by
  obtain ⟨r, off, hd, hm, ha⟩ := store_both h hk h1 h2
  unfold addSlot; rw [hd]; exact ⟨hm, ha⟩

theorem store_same_none {m₁ m₂ : Mem} (h : MemSame m₁ m₂) {a : UInt64} {n : Nat} {v : UInt64}
    (h2 : m₂.store a n v = none) : m₁.store a n v = none := by
  have := store_isSome h a n v v
  rw [h2] at this
  cases h1 : m₁.store a n v with
  | none => rfl
  | some _ => rw [h1] at this; cases this

theorem cong_cudaInit {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (slot : UInt64) :
    ResSame (addSlot slot K) (ffiCudaInit [slot] w₁) (ffiCudaInit [slot] w₂) := by
  unfold ffiCudaInit
  rw [← hs.cudaDevice]
  cases hd : w₁.cudaDevice
  · simp only [Bool.false_eq_true, if_false, bind, Option.bind]
    cases h2 : w₂.mem.store slot 8 0 with
    | none => rw [store_same_none hs.mem h2]; trivial
    | some m₂ =>
      have := store_isSome hs.mem slot 8 0 0
      rw [h2] at this
      cases h1 : w₁.mem.store slot 8 0 with
      | none => rw [h1] at this; cases this
      | some m₁ =>
        obtain ⟨hm, ha⟩ := store_slot hs.mem hs.agree h1 h2
        exact ⟨rfl, hm, ha, rfl, hs.kernel₁, hs.kernel₂, rfl⟩
  simp only [if_true, bind, Option.bind]
  cases h2 : w₂.mem.store slot 8 cudaCtx with
  | none => rw [store_same_none hs.mem h2]; trivial
  | some m₂ =>
    have := store_isSome hs.mem slot 8 cudaCtx cudaCtx
    rw [h2] at this
    cases h1 : w₁.mem.store slot 8 cudaCtx with
    | none => rw [h1] at this; cases this
    | some m₁ =>
      obtain ⟨hm, ha⟩ := store_slot hs.mem hs.agree h1 h2
      exact ⟨rfl, hm, ha, rfl, hs.kernel₁, hs.kernel₂, rfl⟩

theorem cong_cudaCleanup {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (slot : UInt64) :
    ResSame (addSlot slot K) (ffiCudaCleanup [slot] w₁) (ffiCudaCleanup [slot] w₂) := by
  unfold ffiCudaCleanup
  simp only [bind, Option.bind]
  cases h2 : w₂.mem.store slot 8 0 with
  | none => rw [store_same_none hs.mem h2]; trivial
  | some m₂ =>
    have := store_isSome hs.mem slot 8 0 0
    rw [h2] at this
    cases h1 : w₁.mem.store slot 8 0 with
    | none => rw [h1] at this; cases this
    | some m₁ =>
      obtain ⟨hm, ha⟩ := store_slot hs.mem hs.agree h1 h2
      refine ⟨rfl, ⟨fun r => ?_, rfl, hm.busy, hm.frozen⟩, fun r o n hx i hi => ?_, rfl, hs.kernel₁, hs.kernel₂, hs.cudaDevice⟩
      · have := hm.size r; cases r <;> exact this
      · have := ha r o n hx i hi; cases r <;> exact this

theorem cong_cudaUploadOffset {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaUploadOffset bits w₁) (ffiCudaUploadOffset bits w₂) := by
  unfold ffiCudaUploadOffset
  apply devOnly_same hs
  have hd := hs.dev
  rcases bits with _ | ⟨ctx, _ | ⟨buf, _ | ⟨off, _ | ⟨src, _ | ⟨size, _ | _⟩⟩⟩⟩⟩ <;> try trivial
  simp only [cudaCtxOk_same hd]
  split
  · exact ⟨rfl, hd⟩
  · cases cudaCtxOk w₂ ctx with
    | none => trivial
    | some ok =>
      cases ok
      · exact ⟨rfl, hd⟩
      · rcases get?_same hd (asI32 buf) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
        · simp only [h1, h2]; exact ⟨rfl, hd⟩
        · simp only [h1, h2, hb]
          split
          · exact ⟨rfl, hd⟩
          · have hi := copyOut_isSome hs.mem src size.toNat
            unfold readBytes
            cases h4 : copyOut w₂.mem src size.toNat with
            | none =>
              rw [h4] at hi
              cases h3 : copyOut w₁.mem src size.toNat with
              | none => trivial
              | some _ => rw [h3] at hi; cases hi
            | some y₂ =>
              obtain ⟨y₁, h3, hy⟩ := copyOut_same hs.mem h4
              simp only [h3, bind, Option.bind]
              have hov : (overwrite b₁ off.toNat y₁).size = (overwrite b₂ off.toNat y₂).size := by
                simp only [overwrite, ByteArray.size_append, ByteArray.size_extract, hb, hy]
              have hw := syncWrite_same hd (asI32 buf).toNat hov
              cases e₂ : w₂.dev.syncWrite (asI32 buf).toNat (overwrite b₂ off.toNat y₂) with
              | none =>
                rw [e₂] at hw
                cases e₁ : w₁.dev.syncWrite (asI32 buf).toNat (overwrite b₁ off.toNat y₁) with
                | none => trivial
                | some _ => rw [e₁] at hw; cases hw
              | some d₂ =>
                rw [e₂] at hw
                cases e₁ : w₁.dev.syncWrite (asI32 buf).toNat (overwrite b₁ off.toNat y₁) with
                | none => rw [e₁] at hw; cases hw
                | some d₁ => rw [e₁] at hw; exact ⟨rfl, hw⟩

/-- A device contract that reads nothing but the device's bookkeeping: rewrite
    every field it reads onto the second device, then follow both through the
    same branches. -/
macro "dev_cong" hs:term : tactic => `(tactic| (
  apply devOnly_same $hs
  have hd := ($hs).dev
  simp only [cudaCtxOk_same hd, party?_same hd, capturing_same hd, event?_same hd, erase_race hd,
    erase_capture hd, erase_streams hd, erase_events hd, bufs_size_same hd, bind, Option.bind]
  repeat' (first
    | exact ⟨rfl, hd⟩
    | exact ⟨rfl, by erase_eq hd⟩
    | trivial
    | (split <;> try simp_all only []))))

theorem cong_cudaStreamCreate {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaStreamCreate bits w₁) (ffiCudaStreamCreate bits w₂) := by
  unfold ffiCudaStreamCreate
  dev_cong hs

theorem cong_cudaStreamSync {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaStreamSync bits w₁) (ffiCudaStreamSync bits w₂) := by
  unfold ffiCudaStreamSync
  dev_cong hs

theorem cong_cudaStreamDestroy {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaStreamDestroy bits w₁) (ffiCudaStreamDestroy bits w₂) := by
  unfold ffiCudaStreamDestroy
  dev_cong hs

theorem cong_cudaEventCreate {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaEventCreate bits w₁) (ffiCudaEventCreate bits w₂) := by
  unfold ffiCudaEventCreate
  dev_cong hs

theorem cong_cudaEventDestroy {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaEventDestroy bits w₁) (ffiCudaEventDestroy bits w₂) := by
  unfold ffiCudaEventDestroy
  dev_cong hs

theorem cong_cudaEventRecord {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaEventRecord bits w₁) (ffiCudaEventRecord bits w₂) := by
  unfold ffiCudaEventRecord
  dev_cong hs

theorem cong_cudaStreamWaitEvent {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64) :
    ResSame K (ffiCudaStreamWaitEvent bits w₁) (ffiCudaStreamWaitEvent bits w₂) := by
  unfold ffiCudaStreamWaitEvent
  dev_cong hs

-- Launches read their kernel text, entry point and bindings from memory.

/-- Where the string at `a` ends, in the region it is in: the bytes up to and
    including its terminating zero. -/
def strSpan (m : Mem) (a : UInt64) : Option (Region × Nat × Nat) := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let n ← (List.range (bs.size - off)).find? (fun i => bs.get! (off + i) == 0)
  some (r, off, n + 1)

/-- Whether `K` holds the string at `a`, terminator included. -/
def strKnown (K : Known) (m : Mem) (a : UInt64) : Bool :=
  match strSpan m a with
  | some (r, off, n) => covers K r off n
  | none => false

theorem readCStr_eq {K : Known} {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂) {a : UInt64}
    (hc : strKnown K m₂ a = true) : readCStr m₁ a = readCStr m₂ a := by
  unfold strKnown strSpan at hc
  unfold readCStr
  cases hd : decodeAddr a with
  | none => rfl
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, bind, Option.bind] at hc ⊢
    rw [h.size r]
    cases hf : (List.range ((m₂.region r).size - off)).find? (fun i => (m₂.region r).get! (off + i) == 0) with
    | none => rw [hf] at hc; cases hc
    | some n =>
      rw [hf] at hc
      simp only at hc
      have hb := covers_agree hk hc
      have hf1 : (List.range ((m₂.region r).size - off)).find? (fun i => (m₁.region r).get! (off + i) == 0)
          = some n := by
        rw [List.find?_range_eq_some] at hf ⊢
        obtain ⟨h1, h2, h3⟩ := hf
        refine ⟨by rw [hb n (by omega)]; exact h1, h2, fun j hj => ?_⟩
        rw [hb j (by omega)]; exact h3 j hj
      rw [hf1]
      simp only
      congr 2
      apply List.map_congr_left
      intro i hi
      exact hb i (by have := List.mem_range.mp hi; omega)

theorem mapM_congr_mem {α β : Type} (l : List α) (f g : α → Option β) (h : ∀ x ∈ l, f x = g x) :
    l.mapM f = l.mapM g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.mapM_cons]
    rw [h x (List.mem_cons_self ..), ih fun y hy => h y (List.mem_cons_of_mem x hy)]

/-- Whether `K` holds the `n` bindings a launch reads from `p`. -/
def idsKnown (K : Known) (p : UInt64) (n : Nat) : Bool :=
  (List.range n).all fun i => coversAt K (p + UInt64.ofNat (4 * i)) 4

theorem readIds_eq {K : Known} {m₁ m₂ : Mem} (h : MemSame m₁ m₂) (hk : Agree K m₁ m₂) {p : UInt64}
    {n : Nat} (hc : idsKnown K p n = true) : readIds m₁ p n = readIds m₂ p n := by
  unfold readIds
  rw [mapM_congr_mem _ _ (fun i => m₂.load (p + UInt64.ofNat (4 * i)) 4)]
  intro i hi
  exact load_eq h hk (List.all_eq_true.mp hc i hi)

theorem mapM_get_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) :
    ∀ (ids : List Nat), ((ids.mapM fun i => d₁.get? (Int.ofNat i)) = none ∧
        (ids.mapM fun i => d₂.get? (Int.ofNat i)) = none) ∨
      ∃ bs₁ bs₂, (ids.mapM fun i => d₁.get? (Int.ofNat i)) = some bs₁ ∧
        (ids.mapM fun i => d₂.get? (Int.ofNat i)) = some bs₂ ∧ bs₁.map ByteArray.size = bs₂.map ByteArray.size
  | [] => Or.inr ⟨[], [], rfl, rfl, rfl⟩
  | i :: is => by
    simp only [List.mapM_cons, bind, Option.bind]
    rcases get?_same hd (Int.ofNat i) with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, hb⟩
    · left; rw [h1, h2]; exact ⟨rfl, rfl⟩
    · rw [h1, h2]
      rcases mapM_get_same hd is with ⟨h3, h4⟩ | ⟨bs₁, bs₂, h3, h4, hbs⟩
      · left; rw [h3, h4]; exact ⟨rfl, rfl⟩
      · right
        rw [h3, h4]
        exact ⟨b₁ :: bs₁, b₂ :: bs₂, rfl, rfl, by simp [hb, hbs]⟩

theorem foldl_put_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) :
    ∀ (ids : List Nat) (os₁ os₂ : List ByteArray), os₁.map ByteArray.size = os₂.map ByteArray.size →
      Dev.erase ((ids.zip os₁).foldl (fun d (id, o) => d.put id (some o)) d₁)
        = Dev.erase ((ids.zip os₂).foldl (fun d (id, o) => d.put id (some o)) d₂)
  | [], _, _, _ => by simpa using hd
  | _ :: _, [], [], _ => by simpa using hd
  | _ :: _, [], _ :: _, h => by simp at h
  | _ :: _, _ :: _, [], h => by simp at h
  | id :: ids, o₁ :: os₁, o₂ :: os₂, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    simp only [List.zip_cons_cons, List.foldl_cons]
    exact foldl_put_same (erase_put_same hd id h.1) ids os₁ os₂ h.2

theorem runLaunch_same {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂) {k₁ k₂ : Launch → List ByteArray → List ByteArray}
    (hk₁ : KeepsSize k₁) (hk₂ : KeepsSize k₂) (l : Launch) (ids : List Nat) :
    OptDev (d₁.runLaunch k₁ l ids) (d₂.runLaunch k₂ l ids) := by
  unfold Dev.runLaunch
  simp only [bind, Option.bind]
  have keeps : ∀ (k : Launch → List ByteArray → List ByteArray), KeepsSize k → ∀ ins : List ByteArray,
      ((k l ins).length != ins.length || ((k l ins).zip ins).any (fun (o, i) => o.size != i.size)) = false := by
    intro k hk ins
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
  rcases mapM_get_same hd ids with ⟨h1, h2⟩ | ⟨bs₁, bs₂, h1, h2, hbs⟩
  · rw [h1, h2]; trivial
  · rw [h1, h2]
    simp only [keeps k₁ hk₁, keeps k₂ hk₂, Bool.false_eq_true, if_false]
    apply foldl_put_same hd
    rw [hk₁, hk₂, hbs]

theorem devOp_launch_same {w₁ w₂ : World} {d₁ d₂ : Dev} (hd : Dev.erase d₁ = Dev.erase d₂)
    (hk₁ : KeepsSize w₁.kernel) (hk₂ : KeepsSize w₂.kernel) (p : Nat) (l : Launch) (ids rs ws : List Nat) :
    OptDev (d₁.devOp w₁ p (.launch l ids) rs ws) (d₂.devOp w₂ p (.launch l ids) rs ws) := by
  unfold Dev.devOp Dev.devOp.run
  cases hc : d₂.capture with
  | none =>
    have hc₁ : d₁.capture = none := (erase_capture hd).trans hc
    simp only [hc₁, erase_race hd, bind, Option.bind]
    cases d₂.race.op p rs ws with
    | none => trivial
    | some r => exact runLaunch_same (by erase_eq hd) hk₁ hk₂ l ids
  | some c =>
    have hc₁ : d₁.capture = some c := (erase_capture hd).trans hc
    simp only [hc₁, capturing_same hd, erase_race hd, bind, Option.bind]
    split
    · cases c.race.op p rs ws with
      | none => trivial
      | some r => show Dev.erase _ = Dev.erase _; erase_eq hd
    · cases d₂.race.op p rs ws with
      | none => trivial
      | some r => exact runLaunch_same (by erase_eq hd) hk₁ hk₂ l ids

theorem cudaLaunchOn_same {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (p : Nat) (kernel entry : String)
    (nBufs bindPtr : UInt64) (dims : List UInt64) (hc : idsKnown K bindPtr (asI32 nBufs).toNat = true) :
    DevRes (cudaLaunchOn w₁ p kernel entry nBufs bindPtr dims)
      (cudaLaunchOn w₂ p kernel entry nBufs bindPtr dims) := by
  unfold cudaLaunchOn
  have hd := hs.dev
  rw [readIds_eq hs.mem hs.agree hc]
  simp only [bind, Option.bind]
  cases readIds w₂.mem bindPtr (asI32 nBufs).toNat with
  | none => trivial
  | some ids =>
    simp only
    have hany : ids.any (fun id => (w₁.dev.get? id).isNone) = ids.any (fun id => (w₂.dev.get? id).isNone) := by
      congr 1; funext id
      rcases get?_same hd id with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, _⟩ <;> simp [h1, h2]
    rw [hany]
    split
    · exact ⟨rfl, hd⟩
    · have := devOp_launch_same hd hs.kernel₁ hs.kernel₂ p
        ⟨kernel, entry, ids.map Int.toNat, dims, []⟩ (ids.map Int.toNat) [] (ids.map Int.toNat)
      cases e₂ : w₂.dev.devOp w₂ p _ [] _ with
      | none =>
        rw [e₂] at this
        cases e₁ : w₁.dev.devOp w₁ p _ [] _ with
        | none => trivial
        | some _ => rw [e₁] at this; cases this
      | some d₂ =>
        rw [e₂] at this
        cases e₁ : w₁.dev.devOp w₁ p _ [] _ with
        | none => rw [e₁] at this; cases this
        | some d₁ => rw [e₁] at this; exact ⟨rfl, this⟩

/-- What a launch reads from memory must be known: its kernel text, its entry
    point when named, and its bindings. -/
theorem cong_cudaLaunch {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    (ctx kptr nBufs bindPtr gx gy gz bx by_ bz : UInt64)
    (hkptr : strKnown K w₂.mem kptr = true)
    (hi : idsKnown K bindPtr (asI32 nBufs).toNat = true) :
    ResSame K (ffiCudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w₁)
      (ffiCudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w₂) := by
  unfold ffiCudaLaunch
  apply devOnly_same hs
  have hd := hs.dev
  simp only [cudaCtxOk_same hd, party?_same hd, readCStrAt, readCStr_eq hs.mem hs.agree hkptr, bind, Option.bind]
  repeat' (first
    | exact ⟨rfl, hd⟩
    | exact cudaLaunchOn_same hs _ _ _ _ _ _ hi
    | trivial
    | (split <;> try simp_all only []))

theorem cong_cudaLaunchNamed {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    (ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz : UInt64)
    (hkptr : strKnown K w₂.mem kptr = true)
    (hnamePtr : strKnown K w₂.mem namePtr = true)
    (hi : idsKnown K bindPtr (asI32 nBufs).toNat = true) :
    ResSame K (ffiCudaLaunchNamed [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w₁)
      (ffiCudaLaunchNamed [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w₂) := by
  unfold ffiCudaLaunchNamed
  apply devOnly_same hs
  have hd := hs.dev
  simp only [cudaCtxOk_same hd, party?_same hd, readCStrAt, readCStr_eq hs.mem hs.agree hkptr, readCStr_eq hs.mem hs.agree hnamePtr, bind, Option.bind]
  repeat' (first
    | exact ⟨rfl, hd⟩
    | exact cudaLaunchOn_same hs _ _ _ _ _ _ hi
    | trivial
    | (split <;> try simp_all only []))

theorem cong_cudaLaunchOnStream {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    (ctx kptr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64)
    (hkptr : strKnown K w₂.mem kptr = true)
    (hi : idsKnown K bindPtr (asI32 nBufs).toNat = true) :
    ResSame K (ffiCudaLaunchOnStream [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w₁)
      (ffiCudaLaunchOnStream [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w₂) := by
  unfold ffiCudaLaunchOnStream
  apply devOnly_same hs
  have hd := hs.dev
  simp only [cudaCtxOk_same hd, party?_same hd, readCStrAt, readCStr_eq hs.mem hs.agree hkptr, bind, Option.bind]
  repeat' (first
    | exact ⟨rfl, hd⟩
    | exact cudaLaunchOn_same hs _ _ _ _ _ _ hi
    | trivial
    | (split <;> try simp_all only []))

theorem cong_cudaLaunchNamedOnStream {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂)
    (ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64)
    (hkptr : strKnown K w₂.mem kptr = true)
    (hnamePtr : strKnown K w₂.mem namePtr = true)
    (hi : idsKnown K bindPtr (asI32 nBufs).toNat = true) :
    ResSame K (ffiCudaLaunchNamedOnStream [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w₁)
      (ffiCudaLaunchNamedOnStream [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w₂) := by
  unfold ffiCudaLaunchNamedOnStream
  apply devOnly_same hs
  have hd := hs.dev
  simp only [cudaCtxOk_same hd, party?_same hd, readCStrAt, readCStr_eq hs.mem hs.agree hkptr, readCStr_eq hs.mem hs.agree hnamePtr, bind, Option.bind]
  repeat' (first
    | exact ⟨rfl, hd⟩
    | exact cudaLaunchOn_same hs _ _ _ _ _ _ hi
    | trivial
    | (split <;> try simp_all only []))

-- ---------------------------------------------------------------------------
-- The calls the checker makes
-- ---------------------------------------------------------------------------

/-- The foreign calls `Host.Static` makes against its canonical world. -/
def supportedList : List IR.Ffi :=
  [.cudaInit, .cudaCleanup, .cudaCreateBuffer, .cudaUpload, .cudaUploadOffset,
   .cudaDownload, .cudaDownloadOffset, .cudaFreeBuffer, .cudaSync,
   .cudaStreamCreate, .cudaStreamSync, .cudaStreamDestroy,
   .cudaEventCreate, .cudaEventRecord, .cudaEventDestroy, .cudaStreamWaitEvent,
   .cudaLaunch, .cudaLaunchNamed, .cudaLaunchOnStream, .cudaLaunchNamedOnStream]

def supported (f : IR.Ffi) : Bool := decide (f ∈ supportedList)

/-- What a call reads from memory to decide what to do, known in `K`. -/
def need (f : IR.Ffi) (bits : List UInt64) (K : Known) (m : Mem) : Bool :=
  match f, bits with
  | .cudaLaunch, [_, kptr, nBufs, bindPtr, _, _, _, _, _, _] =>
      strKnown K m kptr && idsKnown K bindPtr (asI32 nBufs).toNat
  | .cudaLaunchNamed, [_, kptr, namePtr, nBufs, bindPtr, _, _, _, _, _, _] =>
      strKnown K m kptr && strKnown K m namePtr && idsKnown K bindPtr (asI32 nBufs).toNat
  | .cudaLaunchOnStream, [_, kptr, nBufs, bindPtr, _, _, _, _, _, _, _] =>
      strKnown K m kptr && idsKnown K bindPtr (asI32 nBufs).toNat
  | .cudaLaunchNamedOnStream, [_, kptr, namePtr, nBufs, bindPtr, _, _, _, _, _, _, _] =>
      strKnown K m kptr && strKnown K m namePtr && idsKnown K bindPtr (asI32 nBufs).toNat
  | _, _ => true

/-- What stays known after a call: `K`, less what it overwrote, plus what it
    wrote the same on every world. -/
def post (f : IR.Ffi) (bits : List UInt64) (K : Known) : Known :=
  match f, bits with
  | .cudaInit, [slot] => addSlot slot K
  | .cudaCleanup, [slot] => addSlot slot K
  | .cudaDownload, [_, _, dst, size] => killCopy dst size.toNat K
  | .cudaDownloadOffset, [_, _, _, dst, size] => killCopy dst size.toNat K
  | _, _ => K

theorem cong_cudaLaunch_bits {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64)
    (hn : need .cudaLaunch bits K w₂.mem = true) :
    ResSame K (ffiCudaLaunch bits w₁) (ffiCudaLaunch bits w₂) := by
  match bits, hn with
  | [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9], hn =>
    simp only [need, Bool.and_eq_true] at hn
    exact cong_cudaLaunch hs _ _ _ _ _ _ _ _ _ _ hn.1 hn.2
  | [], _ => exact devOnly_same hs trivial
  | [_], _ => exact devOnly_same hs trivial
  | [_, _], _ => exact devOnly_same hs trivial
  | [_, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _ => exact devOnly_same hs trivial

theorem cong_cudaLaunchNamed_bits {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64)
    (hn : need .cudaLaunchNamed bits K w₂.mem = true) :
    ResSame K (ffiCudaLaunchNamed bits w₁) (ffiCudaLaunchNamed bits w₂) := by
  match bits, hn with
  | [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], hn =>
    simp only [need, Bool.and_eq_true] at hn
    exact cong_cudaLaunchNamed hs _ _ _ _ _ _ _ _ _ _ _ hn.1.1 hn.1.2 hn.2
  | [], _ => exact devOnly_same hs trivial
  | [_], _ => exact devOnly_same hs trivial
  | [_, _], _ => exact devOnly_same hs trivial
  | [_, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _ => exact devOnly_same hs trivial

theorem cong_cudaLaunchOnStream_bits {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64)
    (hn : need .cudaLaunchOnStream bits K w₂.mem = true) :
    ResSame K (ffiCudaLaunchOnStream bits w₁) (ffiCudaLaunchOnStream bits w₂) := by
  match bits, hn with
  | [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], hn =>
    simp only [need, Bool.and_eq_true] at hn
    exact cong_cudaLaunchOnStream hs _ _ _ _ _ _ _ _ _ _ _ hn.1 hn.2
  | [], _ => exact devOnly_same hs trivial
  | [_], _ => exact devOnly_same hs trivial
  | [_, _], _ => exact devOnly_same hs trivial
  | [_, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _ => exact devOnly_same hs trivial

theorem cong_cudaLaunchNamedOnStream_bits {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (bits : List UInt64)
    (hn : need .cudaLaunchNamedOnStream bits K w₂.mem = true) :
    ResSame K (ffiCudaLaunchNamedOnStream bits w₁) (ffiCudaLaunchNamedOnStream bits w₂) := by
  match bits, hn with
  | [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], hn =>
    simp only [need, Bool.and_eq_true] at hn
    exact cong_cudaLaunchNamedOnStream hs _ _ _ _ _ _ _ _ _ _ _ _ hn.1.1 hn.1.2 hn.2
  | [], _ => exact devOnly_same hs trivial
  | [_], _ => exact devOnly_same hs trivial
  | [_, _], _ => exact devOnly_same hs trivial
  | [_, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | [_, _, _, _, _, _, _, _, _, _, _], _ => exact devOnly_same hs trivial
  | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _ => exact devOnly_same hs trivial

theorem callBits_cong {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (f : IR.Ffi)
    (hf : supported f = true) (bits : List UInt64) (hn : need f bits K w₂.mem = true) :
    ResSame (post f bits K) (callBits f bits w₁) (callBits f bits w₂) := by
  have hm : f ∈ supportedList := of_decide_eq_true hf
  simp only [supportedList, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · rcases bits with _ | ⟨slot, _ | _⟩
    · trivial
    · exact cong_cudaInit hs slot
    · trivial
  · rcases bits with _ | ⟨slot, _ | _⟩
    · trivial
    · exact cong_cudaCleanup hs slot
    · trivial
  · exact cong_cudaCreateBuffer hs bits
  · exact cong_cudaUpload hs bits
  · exact cong_cudaUploadOffset hs bits
  · rcases bits with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | _⟩⟩⟩⟩
    all_goals first | exact cong_cudaDownload hs _ _ _ _ | trivial
  · rcases bits with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _ | _⟩⟩⟩⟩⟩
    all_goals first | exact cong_cudaDownloadOffset hs _ _ _ _ _ | trivial
  · exact cong_cudaFreeBuffer hs bits
  · exact cong_cudaSync hs bits
  · exact cong_cudaStreamCreate hs bits
  · exact cong_cudaStreamSync hs bits
  · exact cong_cudaStreamDestroy hs bits
  · exact cong_cudaEventCreate hs bits
  · exact cong_cudaEventRecord hs bits
  · exact cong_cudaEventDestroy hs bits
  · exact cong_cudaStreamWaitEvent hs bits
  · exact cong_cudaLaunch_bits hs bits hn
  · exact cong_cudaLaunchNamed_bits hs bits hn
  · exact cong_cudaLaunchOnStream_bits hs bits hn
  · exact cong_cudaLaunchNamedOnStream_bits hs bits hn

theorem ofCname_supported (f : IR.Ffi) (hf : supported f = true) : IR.Ffi.ofCname f.cname = some f := by
  have hm : f ∈ supportedList := of_decide_eq_true hf
  simp only [supportedList, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl

theorem supported_ne_spawn (f : IR.Ffi) (hf : supported f = true) : (f == .threadSpawn) = false := by
  have hm : f ∈ supportedList := of_decide_eq_true hf
  simp only [supportedList, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl

/-- **A supported call, as `Sem` makes it, cannot tell data apart.** -/
theorem callOf_cong {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) {f : IR.Ffi}
    (hf : supported f = true) (vs : List V) {bits : List UInt64} (hb : vs.mapM asBits = some bits)
    (hn : need f bits K w₂.mem = true) (lc₁ lc₂ : Locals) :
    ResSame (post f bits K) (callOf lc₁ (.ffi f) vs w₁) (callOf lc₂ (.ffi f) vs w₂) := by
  simp only [callOf, hs.mem.frozen, supported_ne_spawn f hf, Bool.false_eq_true, if_false]
  split
  · trivial
  · simp only [callImport, ofCname_supported f hf, bind, Option.bind, callFfi, hb]
    exact callBits_cong hs f hf bits hn

theorem Same.obsCall {K : Known} {w₁ w₂ : World} (hs : Same K w₁ w₂) (c : Callee) (vs : List V) :
    Same K (obsCall w₁ c vs) (obsCall w₂ c vs) :=
  ⟨hs.mem, hs.agree, hs.dev, hs.kernel₁, hs.kernel₂, hs.cudaDevice⟩

end AlgorithmLib.HProg.Static
