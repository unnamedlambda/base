module
public import AlgorithmLib.Kernel.Run
meta import AlgorithmLib.Kernel.Run
public import AlgorithmLib.Proof.LibSpec
meta import AlgorithmLib.Proof.LibSpec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Proof` — from a thread's proof to what a launch leaves

A thread is a body, so it is proven with the program logic: `thread_of_triple`
reads a body's triple as what a thread that finishes leaves. A block of quiet
threads is its threads in order (`block_quiet`); `fold_mem` composes their
effects on memory, given that none disturbs another's precondition; and
`load_puts` says what threads that each store their own word leave: every word
its value, which is an elementwise kernel's whole claim.

These are partial: they speak of runs that finish. A kernel that does not
finish hangs the GPU and the host waiting on it, so nothing is claimed of it.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec

/-- **A thread that finishes ends where its body's proof says.** -/
theorem thread_of_triple {code : Code} {args : List V} {w u : World} {Q : World → Prop}
    (h : ∀ cfg : Cfg, Triple cfg (At args.toArray w) code { ok := fun _ w' => Q w', faultOk := false })
    {ans : List Answer} {hw : Hw} {steps : Nat} (hr : thread code ans args w hw steps = .ok u) : Q u := by
  unfold thread at hr
  have ht := h { env := [], locals := threadLocals ans w.obs.length hw } steps args.toArray w ⟨rfl, rfl⟩
  cases hc : runCode steps { env := [], locals := threadLocals ans w.obs.length hw } args.toArray w code <;>
    rw [hc] at hr ht <;> simp_all [CodeRes.Sat]

-- ---------------------------------------------------------------------------
-- The parallel rule, for runs that may fail
-- ---------------------------------------------------------------------------

/-- Threads in order, each from where the last left: if the fold finishes,
    its memory is each thread's effect on memory, composed in that order,
    provided each thread's precondition survives the others. -/
theorem fold_mem {S : Type} (run : Nat → S → Except String S) (mem : S → Mem)
    (P : Nat → Mem → Prop) (f : Nat → Mem → Mem)
    (hrun : ∀ t s s', P t (mem s) → run t s = .ok s' → mem s' = f t (mem s))
    (hkeep : ∀ t u m, t ≠ u → P u m → P u (f t m)) :
    ∀ (ts : List Nat) (s s' : S), ts.Nodup → (∀ t ∈ ts, P t (mem s)) →
      ts.foldlM (fun s t => run t s) s = .ok s' → mem s' = ts.foldl (fun m t => f t m) (mem s)
  | [], s, s', _, _, h => by cases h; rfl
  | t :: ts, s, s', hnd, hP, h => by
    simp only [List.foldlM_cons] at h
    cases hr : run t s with
    | error e => rw [hr] at h; cases h
    | ok s1 =>
      rw [hr] at h
      have h1 := hrun t s s1 (hP t List.mem_cons_self) hr
      have hnt : t ∉ ts := (List.nodup_cons.mp hnd).1
      rw [List.foldl_cons, ← h1]
      exact fold_mem run mem P f hrun hkeep ts s1 s' (List.nodup_cons.mp hnd).2
        (fun u hu => by rw [h1]; exact hkeep t u _ (fun e => hnt (e ▸ hu)) (hP u (List.mem_cons_of_mem _ hu))) h

-- ---------------------------------------------------------------------------
-- Stores at distinct words
-- ---------------------------------------------------------------------------

section Words
variable (addr : Nat → UInt64) (val : Nat → UInt64)

/-- Thread `t`'s store of its word, where it succeeds. -/
def put (t : Nat) (m : Mem) : Mem := (m.store (addr t) 4 (val t)).getD m

/-- The words of two threads do not overlap. -/
def Apart4 (t u : Nat) : Prop :=
  ∃ r off r' off', decodeAddr (addr t) = some (r, off) ∧ decodeAddr (addr u) = some (r', off') ∧
    (r' ≠ r ∨ off' + 4 ≤ off ∨ off + 4 ≤ off')

theorem put_frame {t : Nat} {m : Mem} (h : (m.store (addr t) 4 (val t)).isSome) :
    ∃ m', m.store (addr t) 4 (val t) = some m' ∧ put addr val t m = m' := by
  obtain ⟨m', hm⟩ := Option.isSome_iff_exists.mp h
  exact ⟨m', hm, by simp [put, hm]⟩

/-- A word no later store touches reads what it read. -/
theorem load_puts_other {u : Nat} : ∀ (ts : List Nat) (m : Mem), (∀ t ∈ ts, Apart4 addr t u) →
    (ts.foldl (fun m t => put addr val t m) m).load (addr u) 4 = m.load (addr u) 4
  | [], _, _ => rfl
  | t :: ts, m, h => by
    rw [List.foldl_cons, load_puts_other ts _ (fun x hx => h x (List.mem_cons_of_mem _ hx))]
    obtain ⟨r, off, r', off', hd, hd', hc⟩ := h t List.mem_cons_self
    unfold put
    cases hs : m.store (addr t) 4 (val t) with
    | none => rfl
    | some m' => exact load_store_clear hs hd hd' hc

/-- A store that succeeds still succeeds after other stores. -/
theorem store_isSome_puts {u : Nat} {v : UInt64} : ∀ (ts : List Nat) (m : Mem),
    (m.store (addr u) 4 v).isSome → ((ts.foldl (fun m t => put addr val t m) m).store (addr u) 4 v).isSome
  | [], _, h => h
  | t :: ts, m, h => by
    rw [List.foldl_cons]
    refine store_isSome_puts ts _ ?_
    unfold put
    cases hs : m.store (addr t) 4 (val t) with
    | none => exact h
    | some m' =>
      obtain ⟨_, _, _, hF⟩ := store_frame hs
      show (m'.store (addr u) 4 v).isSome = true
      rw [← MemSame.store_isSome hF.same (addr u) 4 v v]; exact h

/-- **Each thread's word holds its value**, after every thread has stored. -/
theorem load_puts : ∀ (ts : List Nat) (m : Mem), ts.Nodup →
    (∀ t ∈ ts, ∀ u ∈ ts, t ≠ u → Apart4 addr t u) →
    (∀ t ∈ ts, (m.store (addr t) 4 (val t)).isSome ∧ (val t).toNat < 2 ^ 32) →
    ∀ u ∈ ts, (ts.foldl (fun m t => put addr val t m) m).load (addr u) 4 = some (val u)
  | [], _, _, _, _, u, hu => absurd hu List.not_mem_nil
  | t :: ts, m, hnd, hap, hok, u, hu => by
    rw [List.foldl_cons]
    have hnt : t ∉ ts := (List.nodup_cons.mp hnd).1
    rcases List.mem_cons.mp hu with rfl | hu'
    · rw [load_puts_other addr val ts _ (fun x hx => hap x (List.mem_cons_of_mem _ hx) u List.mem_cons_self
          (fun e => hnt (e ▸ hx)))]
      obtain ⟨m', hm, hput⟩ := put_frame addr val (hok u List.mem_cons_self).1
      rw [hput]; exact load_store4 (hok u List.mem_cons_self).2 hm
    · refine load_puts ts _ (List.nodup_cons.mp hnd).2
        (fun x hx y hy => hap x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy))
        (fun x hx => ⟨?_, (hok x (List.mem_cons_of_mem _ hx)).2⟩) u hu'
      have := store_isSome_puts addr val [t] m (hok x (List.mem_cons_of_mem _ hx)).1
      simpa using this

/-- Bytes read little-endian into a word: `k` of them give less than `2^(8k)`. -/
theorem bytes_lt (l : List Nat) (g : Nat → UInt8) (hl : l.length ≤ 8) :
    (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (g i).toUInt64) (0 : UInt64)).toNat < 2 ^ (8 * l.length) := by
  induction l with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldr_cons, List.length_cons] at hl ⊢
    have ih := ih (by omega)
    generalize List.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (g i).toUInt64) (0 : UInt64) l = acc at ih ⊢
    rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
    have hg : (g i).toUInt64.toNat < 2 ^ 8 := by simpa using (g i).toNat_lt
    apply Nat.or_lt_two_pow
    · have : acc.toNat <<< (8 % 64 : UInt64).toNat % 2 ^ 64 ≤ acc.toNat <<< 8 := by
        simp [Nat.mod_le]
      calc _ ≤ acc.toNat <<< 8 := this
        _ < 2 ^ (8 * (l.length + 1)) := by
          rw [Nat.shiftLeft_eq, Nat.mul_add, Nat.pow_add, Nat.mul_one]
          exact Nat.mul_lt_mul_of_pos_right ih (Nat.two_pow_pos _)
    · exact Nat.lt_of_lt_of_le hg (Nat.pow_le_pow_right (by decide) (by omega))

/-- A 4-byte load gives a 32-bit word. -/
theorem load4_lt {m : Mem} {a v : UInt64} (h : m.load a 4 = some v) : v.toNat < 2 ^ 32 := by
  unfold Mem.load at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, bind, Option.bind] at h
    split at h
    · cases h
    · cases h
      exact bytes_lt (List.range 4) _ (by simp)

end Words

end AlgorithmLib.Kernel
