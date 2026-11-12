module
public import AlgorithmLib.Kernel.Ptx
meta import AlgorithmLib.Kernel.Ptx
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Kernel.Hw
meta import AlgorithmLib.Kernel.Hw
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Run` — what a launch does

One thread is its body run by the host semantics (`Sem.run`) on its identity
and the kernel's arguments. Threads meet only at collectives, and a
collective's answer is the one thing a thread cannot compute alone, so each
thread is run with its answers given, and the answers are fixed by laws that
read what every thread passed:

* **warp law**: lanes `32w … 32w+31` make the same collectives in the same
  order; at its `k`-th, lane `l` gets lane `l xor m`'s 32-bit word (`shflBfly`),
  lane `m`'s (`shflIdx`), or the mask of lanes that passed non-zero
  (`ballot`);
* **barrier law**: after a block's `b`-th barrier, shared memory is what it
  was, with every thread's shared stores before that barrier applied.

A thread's shuffles and ballots are always answered: the value its list
holds, or zero (`valAt`). A barrier is answered by the shared memory its list
holds; one with none leaves shared memory alone in a round of finding the
answers (`provisional`), and is stuck otherwise. A collective's arguments depend only on the
answers before it, so the laws have one solution, reached in one round per
collective. A block takes answers only once they are settled, the laws giving
back what the threads were answered (`block_settled`), so no thread of a
block's run was answered by default; the threads then run one after another
with them. That they may: a kernel's threads touch disjoint
memory between collectives, so the order does not matter
(`Kernel.launch_any_order` states this for any order).

Shared memory is the `arena` region of the kernel's world; buffer `i` bound to
a launch is region `.buf i`. A block's shared memory starts zeroed.

A block and a launch are the same whatever runs a thread (`Runner`): a
kernel's Code (`Kernel.run`, `LaunchRuns`) or the PTX it compiles to
(`PtxRun.runner`), so the laws and `block_settled` are stated once.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- What a collective answers one thread: a value, or for a barrier the
    block's shared memory. -/
inductive Answer where
  | val (v : V)
  | smem (b : ByteArray)
  deriving DecidableEq

/-- The collective an observation records, if any. -/
def collOf : Obs → Option (Intr × List V)
  | .call (.intr c) a => if c.isColl then some (c, a) else none
  | _ => none

/-- The collectives a run made since the `start`-th observation, in order. -/
def collCalls (w : World) (start : Nat) : List (Intr × List V) :=
  (w.obs.reverse.drop start).filterMap collOf

/-- One observation's part in a thread's shared stores: a barrier passed, or
    a store into shared memory, with how many barriers it had passed. -/
def shStep : Nat × List (Nat × UInt64 × Nat × UInt64) → Obs → Nat × List (Nat × UInt64 × Nat × UInt64)
  | (b, acc), .call (.intr .barrier) _ => (b + 1, acc)
  | (b, acc), .store a n v => match decodeAddr a with
      | some (.arena, _) => (b, (b, a, n, v) :: acc)
      | _ => (b, acc)
  | s, _ => s

/-- A thread's stores into shared memory, each with how many barriers it
    had passed. -/
def sharedStores (w : World) (start : Nat) : List (Nat × UInt64 × Nat × UInt64) :=
  ((w.obs.reverse.drop start).foldl shStep (0, [])).2.reverse

/-- What a collective other than a barrier answers at its `k`-th: the value
    `ans[k]` holds, zero where it holds none. -/
def valAt (ans : List Answer) (k : Nat) : V :=
  match ans[k]? with
  | some (.val v) => v
  | _ => .sc .i32 0

/-- The shared memory a thread's `k`-th collective is answered, empty where
    it is not shared memory. -/
def smemAt (ans : List Answer) (k : Nat) : ByteArray :=
  match ans[k]? with
  | some (.smem b) => b
  | _ => .empty

/-- What a thread's calls answer, its collectives counted from the `start`-th
    observation: its `k`-th collective from `ans[k]` — a barrier by the shared
    memory it holds, stuck where it holds none unless the run is a round of
    finding the answers (`provisional`), when it leaves shared memory alone;
    any other collective by the value `valAt` gives, zero where it holds none —
    an approximating intrinsic as the hardware `hw` does, and `dp4a` exactly. A run whose
    answers the laws do not give back is not a block's (`BlockRuns`). -/
def threadLocals (ans : List Answer) (start : Nat) (hw : Hw) (provisional := false) : Locals := fun i vs w' =>
  match Intr.all.find? (·.slot == i) with
  | some c =>
    if c.isColl then
      if c = .barrier then
        match ans[(collCalls w' start).length - 1]? with
        | some (.smem b) => .ok (none, { w' with mem := { w'.mem with arena := b } })
        | _ => if provisional then .ok (none, w')
            else .error (.stuck "a barrier no law has answered with shared memory")
      else .ok (some (valAt ans ((collCalls w' start).length - 1)), w')
    else
      match vs with
      | [.sc _ x] => .ok (some (.sc .f32 (hw.approx c x)), w')
      | [.sc _ a, .sc _ b, .sc _ e] =>
          if c = .dp4a then .ok (some (.sc .i32 (Intr.dp4aVal a b e)), w')
          else .error (.stuck "an approximation of something not a scalar")
      | _ => .error (.stuck "an approximation of something not a scalar")
  | none => .error (.stuck s!"calls u0:{i}, which a kernel does not have")

/-- One thread: its code run on `args` from `w` with `steps` of fuel, its
    calls answered by `threadLocals`; leaving a loop it is not in is stuck. -/
def thread (code : Code) (ans : List Answer) (args : List V) (w : World)
    (hw : Hw := Hw.rounded) (steps : Nat := 100000000) (provisional := false) : Except String World :=
  match runCode steps { env := [], locals := threadLocals ans w.obs.length hw provisional } args.toArray w code with
  | .ok _ w' => .ok w'
  | .stuck m => .error s!"stuck: {m}"
  | .misuse m => .error s!"misuse: {m}"
  | .fault m => .error s!"fault: {m}"
  | .brk d _ _ _ => .error s!"stuck: br {d} leaves the function body"
  | .cont d _ _ => .error s!"stuck: continue {d} leaves the function body"

/-- Lane `l`'s value at the `k`-th collective of every lane of its warp. -/
def laneArg (calls : Nat → List (Intr × List V)) (warp lane k : Nat) (c : Intr) :
    Except String (List V) :=
  match (calls (32 * warp + lane))[k]? with
  | some (c', a) => if c' == c then .ok a else .error s!"lanes of warp {warp} disagree at collective {k}"
  | none => .error s!"lane {lane} of warp {warp} makes fewer collectives"

/-- Memory with the stores of phase `b` (those made after `b` barriers)
    applied, in order. -/
def phase (all : List (Nat × UInt64 × Nat × UInt64)) (b : Nat) (cur : Mem) : Mem :=
  all.foldl (fun c x => if x.1 = b then (c.store x.2.1 x.2.2.1 x.2.2.2).getD c else c) cur

/-- A phase's stores keep the spans threads own. -/
theorem phase_lent (all : List (Nat × UInt64 × Nat × UInt64)) (b : Nat) : ∀ cur : Mem,
    (phase all b cur).lent = cur.lent := by
  unfold phase
  induction all with
  | nil => intro cur; rfl
  | cons x xs ih =>
      intro cur
      rw [List.foldl_cons, ih]
      split
      · cases h : cur.store x.2.1 x.2.2.1 x.2.2.2 with
        | none => rfl
        | some m => exact Mem.store_lent h
      · rfl

/-- Shared memory after each of a block's barriers: the block's shared stores
    applied phase by phase. -/
def snapshots (threads : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) : Array ByteArray :=
  let all := (List.range threads).flatMap stores
  let nb := (List.range threads).foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) 0
  ((List.range nb).foldl (fun (acc : Mem × Array ByteArray) b =>
      let cur := phase all b acc.1
      (cur, acc.2.push cur.arena))
    (({ arena := smem0, data := .empty, out := .empty } : Mem), #[])).2

/-- The 32-bit word a lane passed first, as a shuffle moves it. -/
def lane32 : List V → V
  | .sc _ x :: _ => .sc .i32 x.toUInt32.toUInt64
  | _ => .sc .i32 0

/-- The law's answer to one collective of lane `lane` of warp `warp`, its
    `k`-th, after `nbar` barriers. -/
def answerStep (calls : Nat → List (Intr × List V)) (snaps : Array ByteArray) (smem0 : ByteArray)
    (warp lane : Nat) : Nat × Array Answer → (Intr × List V) × Nat → Except String (Nat × Array Answer)
  | (nbar, out), ((c, args), k) =>
      match c, args with
      | .barrier, _ => pure (nbar + 1, out.push (Answer.smem (snaps[nbar]?.getD smem0)))
      | .shflBfly, [_, .sc _ m] => do
          let a ← laneArg calls warp ((lane ^^^ m.toNat) % 32) k c
          pure (nbar, out.push (.val (lane32 a)))
      | .shflIdx, [_, .sc _ m] => do
          let a ← laneArg calls warp (m.toNat % 32) k c
          pure (nbar, out.push (.val (lane32 a)))
      | .ballot, _ => do
          let bits ← (List.range 32).foldlM (fun acc l => do
            match ← laneArg calls warp l k c with
            | [v] => pure (if (asBits v).getD 0 != 0 then acc ||| (1 <<< l.toUInt32) else acc)
            | _ => throw "a ballot of other than one value") (0 : UInt32)
          pure (nbar, out.push (.val (.sc .i32 bits.toUInt64)))
      | _, _ => throw s!"{c.name} on arguments it does not take"

/-- What the laws answer thread `t`'s collectives, in order. -/
def answersOf (calls : Nat → List (Intr × List V)) (snaps : Array ByteArray) (smem0 : ByteArray)
    (t : Nat) : Except String (List Answer) := do
  let (_, out) ← (calls t).zipIdx.foldlM (answerStep calls snaps smem0 (t / 32) (t % 32)) (0, #[])
  pure out.toList

/-- **The laws**, from what every thread of a block passed. -/
def answers (threads : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) : Except String (Nat → List Answer) := do
  let per ← (List.range threads).mapM (answersOf calls (snapshots threads smem0 calls stores) smem0)
  pure fun t => per.getD t []

/-- The collectives a thread's run made, none if it failed. -/
def runCalls (r : Except String World) (start : Nat) : List (Intr × List V) :=
  match r with
  | .ok u => collCalls u start
  | .error _ => []

/-- The shared stores a thread's run made, none if it failed. -/
def runStores (r : Except String World) (start : Nat) : List (Nat × UInt64 × Nat × UInt64) :=
  match r with
  | .ok u => sharedStores u start
  | .error _ => []

/-- What runs one thread of a block: provisionally or not (a round of finding
    the answers, or the pass), on its answers, arguments and world. A block and
    a launch are the same whatever runs their threads: a kernel's Code
    (`Kernel.run`) or the PTX it compiles to (`PtxRun.launchOn`). -/
abbrev Runner := Bool → List Answer → List V → World → Except String World

/-- A thread of kernel `k`: its code with `steps` of fuel and the hardware's
    approximations `hw`. -/
def Kernel.run (k : Kernel) (hw : Hw) (steps : Nat) : Runner :=
  fun prov ans args w => thread k.code ans args w hw steps prov

/-- One round of finding a block's answers: every thread run with the answers
    so far, the answers the laws give from what they passed, and the most
    collectives any thread made. A thread that fails here is reported by the
    last pass, which runs every thread again. -/
def round (run : Runner) (blockDim : Nat) (smem0 : ByteArray) (args : Nat → List V) (w0 : World)
    (ans : Nat → List Answer) : Except String ((Nat → List Answer) × Nat) := do
  let runs := (List.range blockDim).map fun t => run true (ans t) (args t) w0
  let calls (t : Nat) := runCalls (runs.getD t (.error "")) w0.obs.length
  let stores (t : Nat) := runStores (runs.getD t (.error "")) w0.obs.length
  let ans' ← answers blockDim smem0 calls stores
  pure (ans', (List.range blockDim).foldl (fun m t => max m (calls t).length) 0)

/-- `n` rounds from `ans`. -/
def rounds (step : (Nat → List Answer) → Except String ((Nat → List Answer) × Nat)) :
    Nat → (Nat → List Answer) → Except String (Nat → List Answer)
  | 0, ans => pure ans
  | n + 1, ans => do
      let (ans', _) ← step ans
      rounds step n ans'

/-- The arguments thread `t` of block `blockIdx` is run on: its numbers as the
    32-bit registers `%ctaid.x`, `%tid.x` and `%ntid.x` hold them. -/
def threadArgs (blockIdx blockDim : Nat) (params : List V) (t : Nat) : List V :=
  [.sc .i32 (UInt32.ofNat blockIdx).toUInt64, .sc .i32 (UInt32.ofNat t).toUInt64,
   .sc .i32 (UInt32.ofNat blockDim).toUInt64, .sc .i64 (addrOf .arena 0)] ++ params

/-- What a pass of a block's threads carries: the world, and the collectives
    and shared stores each thread made. -/
abbrev PassSt := World × (Nat → List (Intr × List V)) × (Nat → List (Nat × UInt64 × Nat × UInt64))

/-- Thread `t` of a pass, run with its answers after the threads before it,
    and what it passed. -/
def passStep (run : Runner) (smem0 : ByteArray) (args : Nat → List V) (ans : Nat → List Answer) :
    PassSt → Nat → Except String PassSt
  | (w', calls, stores), t => do
    let u ← run false (ans t) (args t) { w' with mem := { w'.mem with arena := smem0 } }
    pure (u, fun t' => if t' = t then collCalls u w'.obs.length else calls t',
      fun t' => if t' = t then sharedStores u w'.obs.length else stores t')

/-- **One block**: its threads in order, each with its answers, which the laws
    must give back from what those runs passed; answers they do not give back
    are an error, so a block is never answered wrongly. The answers are found
    by rounds from none, one more than the collectives the first round's
    threads make (each round settles one more: a collective's arguments depend
    only on the answers before it). -/
def block (run : Runner) (smem blockIdx blockDim : Nat) (params : List V) (w : World) :
    Except String World := do
  let smem0 := ByteArray.mk (Array.replicate smem 0)
  let w0 : World := { w with mem := { w.mem with arena := smem0 } }
  let args := threadArgs blockIdx blockDim params
  let step := fun ans => round run blockDim smem0 args w0 ans
  let (_, n) ← step fun _ => []
  let ans ← rounds step (n + 1) fun _ => []
  let (w', calls, stores) ← (List.range blockDim).foldlM (passStep run smem0 args ans)
    (w0, fun _ => [], fun _ => [])
  let ans' ← answers blockDim smem0 calls stores
  unless (List.range blockDim).all (fun t => decide (ans' t = ans t)) do
    throw "the laws' answers are not settled"
  pure { w' with mem := { w'.mem with arena := w.mem.arena } }

/-- The address of the buffer a launch binds at position `i`. -/
def bufAddr (i : Nat) (h : i < 268435449 := by decide) : UInt64 := addrOf (.buf ⟨i, h⟩) 0

/-- **A launch**: `grid` blocks of `blockDim` threads, each with `smem` bytes
    of shared memory, on device memory `bufs`, buffer by number, its float
    arithmetic leaving NaNs as `nan` says, and what it holds after. -/
def launchOn (run : Runner) (smem grid blockDim : Nat) (params : List V) (bufs : Nat → ByteArray)
    (nan : ClifTy → UInt64 → UInt64) : Except String (Nat → ByteArray) := do
  let w : World := { mem := { arena := .empty, data := .empty, out := .empty, bufs, nan } }
  let w' ← (List.range grid).foldlM (fun w b => block run smem b blockDim params w) w
  pure w'.mem.bufs

/-- A launch of kernel `k`, buffer `i` the `i`-th of a list. -/
def launch (k : Kernel) (grid blockDim : Nat) (params : List V) (bufs : List ByteArray) :
    Except String (List ByteArray) := do
  let out ← launchOn (k.run Hw.rounded 100000000) k.smem grid blockDim params (fun i => bufs.getD i .empty)
    Hw.rounded.nan
  pure ((List.range bufs.length).map out)

-- ---------------------------------------------------------------------------
-- The parallel rule
-- ---------------------------------------------------------------------------

/-! Threads run one after another in some order. Each thread's proof says:
from any state where its precondition holds, it ends in `f t s`. If no
thread's effect disturbs another's precondition, and effects commute, every
order ends in the same state. -/

section Parallel
variable {S : Type}

/-- Threads `ts` run in order; each may fail. -/
def runAll (run : Nat → S → Option S) : List Nat → S → Option S
  | [], s => some s
  | t :: ts, s => (run t s).bind (runAll run ts)

/-- What the threads' specified effects compose to, in order. -/
def effAll (f : Nat → S → S) : List Nat → S → S
  | [], s => s
  | t :: ts, s => effAll f ts (f t s)

theorem runAll_eq (run : Nat → S → Option S) (P : Nat → S → Prop) (f : Nat → S → S)
    (hrun : ∀ t s, P t s → run t s = some (f t s))
    (hkeep : ∀ t u s, t ≠ u → P u s → P u (f t s)) :
    ∀ (ts : List Nat) (s : S), ts.Nodup → (∀ t ∈ ts, P t s) → runAll run ts s = some (effAll f ts s) := by
  intro ts
  induction ts with
  | nil => intro s _ _; rfl
  | cons t ts ih =>
    intro s hnd hP
    have ht : t ∉ ts := (List.nodup_cons.mp hnd).1
    simp only [runAll, effAll, hrun t s (hP t List.mem_cons_self), Option.bind_some]
    exact ih _ (List.nodup_cons.mp hnd).2
      (fun u hu => hkeep t u s (fun h => ht (h ▸ hu)) (hP u (List.mem_cons_of_mem _ hu)))

/-- Commuting effects compose to the same state in any order. -/
theorem effAll_perm (f : Nat → S → S) (hcomm : ∀ t u s, t ≠ u → f u (f t s) = f t (f u s)) :
    ∀ {ts us : List Nat}, ts.Nodup → ts.Perm us → ∀ s, effAll f ts s = effAll f us s := by
  intro ts us hnd hp
  induction hp with
  | nil => intro s; rfl
  | cons t _ ih => intro s; exact ih (List.nodup_cons.mp hnd).2 _
  | swap t u l =>
    intro s
    have htu : u ≠ t := fun h => (List.nodup_cons.mp hnd).1 (h ▸ List.mem_cons_self)
    simp only [effAll, hcomm u t s htu]
  | trans _ _ ih1 ih2 => intro s; rw [ih1 hnd, ih2 ((List.Perm.nodup_iff (by assumption)).mp hnd)]

/-- **The parallel rule**: every order of the threads ends in the same state,
    the one the specified effects compose to. -/
theorem launch_any_order (run : Nat → S → Option S) (P : Nat → S → Prop) (f : Nat → S → S)
    (hrun : ∀ t s, P t s → run t s = some (f t s))
    (hkeep : ∀ t u s, t ≠ u → P u s → P u (f t s))
    (hcomm : ∀ t u s, t ≠ u → f u (f t s) = f t (f u s))
    (n : Nat) (order : List Nat) (hord : order.Perm (List.range n)) (s : S)
    (hP : ∀ t < n, P t s) :
    runAll run order s = some (effAll f (List.range n) s) := by
  have hnd : order.Nodup := hord.nodup_iff.mpr List.nodup_range
  rw [runAll_eq run P f hrun hkeep order s hnd
    (fun t ht => hP t (List.mem_range.mp (hord.subset ht))), effAll_perm f hcomm hnd hord]

end Parallel

-- ---------------------------------------------------------------------------
-- Quiet kernels
-- ---------------------------------------------------------------------------

/-! A kernel whose threads make no collectives is its threads run in order:
the laws have nothing to answer. -/

section Quiet

theorem mapM_congr_mem {α β : Type} (f g : α → Except String β) : ∀ (l : List α),
    (∀ x ∈ l, f x = g x) → l.mapM f = l.mapM g
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.mapM_cons]
    rw [h x List.mem_cons_self, mapM_congr_mem f g xs (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem foldl_keep {α β : Type} (l : List β) : ∀ a : α, l.foldl (fun m _ => m) a = a := by
  induction l with
  | nil => intro a; rfl
  | cons x xs ih => intro a; exact ih a

theorem answers_quiet (n : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) (hc : ∀ t, t < n → calls t = []) :
    answers n smem0 calls stores = .ok fun _ => [] := by
  unfold answers
  rw [mapM_congr_mem _ (fun _ => (pure [] : Except String (List Answer))) _ (fun t ht => by
    simp [answersOf, hc t (List.mem_range.mp ht)])]
  have hm : ∀ l : List Nat, List.mapM (fun (_ : Nat) => (pure [] : Except String (List Answer))) l =
      .ok (l.map fun _ => []) := by
    intro l; induction l with
    | nil => rfl
    | cons x xs ih => simp only [List.mapM_cons, ih, List.map_cons]; rfl
  rw [hm]
  show Except.ok (fun t => ((List.range n).map fun _ => ([] : List Answer)).getD t []) = _
  congr 1; funext t
  cases h : (List.range n)[t]? <;> simp_all [List.getD]

/-- Thread `t`'s run among a round's, `t` one of them. -/
theorem runs_getD {α : Type} (f : Nat → α) (n t : Nat) (d : α) (ht : t < n) :
    ((List.range n).map f).getD t d = f t := by
  simp [List.getD_eq_getElem?_getD, ht]

theorem round_quiet (run : Runner) (bd : Nat) (smem0 : ByteArray) (args : Nat → List V) (w0 : World)
    (ans : Nat → List Answer)
    (hc : ∀ t, t < bd → runCalls (run true (ans t) (args t) w0) w0.obs.length = []) :
    round run bd smem0 args w0 ans = .ok (fun _ => [], 0) := by
  have hc' : ∀ t, t < bd → runCalls (((List.range bd).map fun t => run true (ans t) (args t) w0).getD t
      (.error "")) w0.obs.length = [] := fun t ht => by rw [runs_getD _ _ _ _ ht]; exact hc t ht
  unfold round
  dsimp only
  rw [answers_quiet _ _ _ _ hc']
  simp only [bind, Except.bind]
  rw [foldl_congr_mem _ (fun m _ => m) _ _ (fun t ht m => by rw [hc' t (List.mem_range.mp ht)]; simp), foldl_keep]
  rfl

theorem rounds_quiet (step : (Nat → List Answer) → Except String ((Nat → List Answer) × Nat))
    (hs : step (fun _ => []) = .ok (fun _ => [], 0)) : ∀ n, rounds step n (fun _ => []) = .ok fun _ => []
  | 0 => rfl
  | n + 1 => by simp only [rounds, hs, bind, Except.bind]; exact rounds_quiet step hs n

/-- A block's threads in order, no collective answered. -/
def blockPlain (run : Runner) (smem blockIdx blockDim : Nat) (params : List V) (w : World) :
    Except String World := do
  let smem0 := ByteArray.mk (Array.replicate smem 0)
  let w0 : World := { w with mem := { w.mem with arena := smem0 } }
  let w' ← (List.range blockDim).foldlM (fun w' t =>
    run false [] (threadArgs blockIdx blockDim params t) { w' with mem := { w'.mem with arena := smem0 } }) w0
  pure { w' with mem := { w'.mem with arena := w.mem.arena } }

/-- A pass's world is its threads' runs in order. -/
theorem pass_world (run : Runner) (smem0 : ByteArray) (args : Nat → List V) (ans : Nat → List Answer) :
    ∀ (ts : List Nat) (s : PassSt),
      (ts.foldlM (passStep run smem0 args ans) s).map (·.1) =
        ts.foldlM (fun w' t => run false (ans t) (args t) { w' with mem := { w'.mem with arena := smem0 } }) s.1
  | [], _ => rfl
  | t :: ts, (w', calls, stores) => by
    simp only [List.foldlM_cons, passStep, bind, Except.bind]
    cases run false (ans t) (args t) { w' with mem := { w'.mem with arena := smem0 } } with
    | error e => rfl
    | ok u => exact pass_world run smem0 args ans ts _

/-- What holds of every state of a pass, from each thread's step. -/
theorem pass_inv {run : Runner} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer}
    (P : PassSt → Prop) : ∀ (ts : List Nat) (s r : PassSt),
      (∀ s t s', t ∈ ts → P s → passStep run smem0 args ans s t = .ok s' → P s') →
      P s → ts.foldlM (passStep run smem0 args ans) s = .ok r → P r
  | [], _, _, _, hs, hr => by cases hr; exact hs
  | t :: ts, s, r, h, hs, hr => by
    simp only [List.foldlM_cons, bind, Except.bind] at hr
    split at hr
    · cases hr
    · rename_i s' hs'
      exact pass_inv P ts s' r (fun s t s' ht => h s t s' (List.mem_cons_of_mem _ ht))
        (h s t s' List.mem_cons_self hs hs') hr

/-- **A quiet kernel's block is its threads in order**: when no thread that
    finishes, whatever it is answered and wherever it starts, has made a
    collective, the laws answer nothing. -/
theorem block_quiet (run : Runner) (smem b bd : Nat) (params : List V) (w : World)
    (hq : ∀ t, t < bd → ∀ ans s u (prov : Bool), run prov ans (threadArgs b bd params t) s = .ok u →
      collCalls u s.obs.length = []) :
    block run smem b bd params w = blockPlain run smem b bd params w := by
  have hr : ∀ t, t < bd → ∀ ans, runCalls (run true ans (threadArgs b bd params t)
      { w with mem := { w.mem with arena := ByteArray.mk (Array.replicate smem 0) } }) w.obs.length = [] := by
    intro t ht ans
    unfold runCalls
    split
    · rename_i u hu; have h := hq t ht ans _ u true hu; exact h
    · rfl
  unfold block blockPlain
  dsimp only
  rw [round_quiet _ _ _ _ _ _ (fun t ht => hr t ht _)]
  simp only [bind, Except.bind]
  rw [rounds_quiet _ (round_quiet _ _ _ _ _ _ (fun t ht => hr t ht _))]
  dsimp only
  have hw := pass_world run (ByteArray.mk (Array.replicate smem 0)) (threadArgs b bd params) (fun _ => [])
    (List.range bd)
    ({ w with mem := { w.mem with arena := ByteArray.mk (Array.replicate smem 0) } }, fun _ => [], fun _ => [])
  cases hp : (List.range bd).foldlM (passStep run (ByteArray.mk (Array.replicate smem 0)) (threadArgs b bd params)
      (fun _ => []))
      ({ w with mem := { w.mem with arena := ByteArray.mk (Array.replicate smem 0) } }, fun _ => [], fun _ => []) with
  | error e =>
    rw [hp] at hw
    simp only at hw ⊢
    rw [← hw]; rfl
  | ok r =>
    rw [hp] at hw
    obtain ⟨w', calls, stores⟩ := r
    have hc : ∀ t, calls t = [] :=
      pass_inv (fun s => ∀ t, s.2.1 t = []) (List.range bd) _ _
        (fun s t s' ht hs hst => by
          obtain ⟨w1, c1, s1⟩ := s
          simp only [passStep, bind, Except.bind] at hst
          split at hst
          · cases hst
          · rename_i u hu
            cases hst
            intro t'
            by_cases e : t' = t
            · simp only [e, if_true]
              have h := hq t (List.mem_range.mp ht) _ _ _ false hu; exact h
            · simp only [e, if_false]; exact hs t')
        (fun _ => rfl) hp
    simp only [Except.map] at hw
    rw [← hw]
    simp only [answers_quiet _ _ _ _ (fun t _ => hc t)]
    rw [if_pos (by simp)]
    rfl

end Quiet

/-- **A block's answers are settled**: a block that runs ran its threads in
    order with answers the laws give back from what those runs passed. -/
theorem block_settled {run : Runner} {smem b bd : Nat} {params : List V} {w u : World}
    (h : block run smem b bd params w = .ok u) :
    let smem0 := ByteArray.mk (Array.replicate smem 0)
    let w0 : World := { w with mem := { w.mem with arena := smem0 } }
    ∃ ans w' calls stores ans',
      (List.range bd).foldlM (passStep run smem0 (threadArgs b bd params) ans) (w0, fun _ => [], fun _ => []) =
        .ok (w', calls, stores) ∧
      answers bd smem0 calls stores = .ok ans' ∧ (∀ t, t < bd → ans' t = ans t) ∧
      u = { w' with mem := { w'.mem with arena := w.mem.arena } } := by
  intro smem0 w0
  unfold block at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i ans _
  split at h
  · cases h
  rename_i r hr
  obtain ⟨w', calls, stores⟩ := r
  dsimp only at h
  split at h
  · cases h
  rename_i ans' hans
  by_cases hc : ((List.range bd).all fun t => decide (ans' t = ans t)) = true
  · rw [if_pos hc] at h
    simp only [pure, Except.pure] at h
    refine ⟨ans, w', calls, stores, ans', hr, hans, fun t ht => ?_, (Except.ok.inj h).symm⟩
    have := List.all_eq_true.mp hc t (List.mem_range.mpr ht)
    simpa using this
  · rw [if_neg hc] at h
    cases h

/-! ## What a launch is

A block runs when its threads, in order, each run with its answers to the
collectives it makes, and the laws give those answers back from what the runs
passed: the answers settle. A launch runs its blocks in order. These are what
a launch means, whatever runs its threads at some fuel (`run : Nat → Runner`);
`block` and `launchOn` compute it, finding the answers by rounds
(`block_runs`, `launchOn_runs`). A kernel's launch is its Code's
(`BlockRuns`, `LaunchRuns`). -/

/-- **A block runs** from `w` to `u`: its threads in order, each with its
    answers, which the laws give back from what those runs passed; each run
    ends, at one fuel. -/
def RunsBlock (run : Nat → Runner) (smem b bd : Nat) (params : List V) (w u : World) : Prop :=
  let smem0 := ByteArray.mk (Array.replicate smem 0)
  let w0 : World := { w with mem := { w.mem with arena := smem0 } }
  ∃ steps ans w' calls stores ans',
    (List.range bd).foldlM (passStep (run steps) smem0 (threadArgs b bd params) ans)
      (w0, fun _ => [], fun _ => []) = .ok (w', calls, stores) ∧
    answers bd smem0 calls stores = .ok ans' ∧ (∀ t, t < bd → ans' t = ans t) ∧
    u = { w' with mem := { w'.mem with arena := w.mem.arena } }

theorem block_runs {run : Nat → Runner} {steps smem b bd : Nat} {params : List V} {w u : World}
    (h : block (run steps) smem b bd params w = .ok u) : RunsBlock run smem b bd params w u :=
  ⟨steps, block_settled h⟩

/-- The first `c` blocks of a grid run, in order. -/
def RunsBlocks (run : Nat → Runner) (smem bd : Nat) (params : List V) : Nat → World → World → Prop
  | 0, w, u => u = w
  | c + 1, w, u => ∃ v, RunsBlocks run smem bd params c w v ∧ RunsBlock run smem c bd params v u

/-- **A launch runs**: `grid` blocks of `blockDim` threads, in order, on
    device memory `bufs`, buffer by number, its float arithmetic leaving NaNs
    as `nan` says, leaving `bufs'`. -/
def RunsLaunch (run : Nat → Runner) (smem grid blockDim : Nat) (params : List V) (bufs bufs' : Nat → ByteArray)
    (nan : ClifTy → UInt64 → UInt64) : Prop :=
  ∃ u, RunsBlocks run smem blockDim params grid { mem := { arena := .empty, data := .empty, out := .empty, bufs, nan } } u ∧
    u.mem.bufs = bufs'

/-- A block of kernel `k` runs, its Code each thread. -/
abbrev BlockRuns (k : Kernel) (b bd : Nat) (params : List V) (hw : Hw) (w u : World) : Prop :=
  RunsBlock (k.run hw) k.smem b bd params w u

/-- The first `c` blocks of kernel `k`'s grid run. -/
abbrev BlocksRun (k : Kernel) (bd : Nat) (params : List V) (hw : Hw) (c : Nat) (w u : World) : Prop :=
  RunsBlocks (k.run hw) k.smem bd params c w u

/-- **A launch of kernel `k` runs**, its Code each thread, the hardware's
    approximations `hw`'s. -/
abbrev LaunchRuns (k : Kernel) (grid blockDim : Nat) (params : List V) (bufs bufs' : Nat → ByteArray)
    (hw : Hw := Hw.rounded) : Prop :=
  RunsLaunch (k.run hw) k.smem grid blockDim params bufs bufs' hw.nan

/-- **What `launchOn` computes is a run of the launch.** -/
theorem launchOn_runs {run : Nat → Runner} {steps smem grid bd : Nat} {params : List V} {bufs bufs' : Nat → ByteArray}
    {nan : ClifTy → UInt64 → UInt64}
    (h : launchOn (run steps) smem grid bd params bufs nan = .ok bufs') :
    RunsLaunch run smem grid bd params bufs bufs' nan := by
  unfold launchOn at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i W hfold
  cases h
  refine ⟨W, ?_, rfl⟩
  suffices hs : ∀ c W, (List.range c).foldlM (fun w b => block (run steps) smem b bd params w)
      ({ mem := { arena := .empty, data := .empty, out := .empty, bufs, nan } } : World) = .ok W →
      RunsBlocks run smem bd params c { mem := { arena := .empty, data := .empty, out := .empty, bufs, nan } } W from
    hs grid W hfold
  intro c
  induction c with
  | zero =>
      intro W h
      simp only [List.range_zero, List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
      exact h.symm
  | succ c ih =>
      intro W h
      rw [List.range_succ, List.foldlM_append] at h
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      rename_i Wp hp
      simp only [List.foldlM_cons, List.foldlM_nil, bind, Except.bind] at h
      split at h
      · cases h
      rename_i Wb hb
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨Wp, ih Wp hp, block_runs hb⟩

end AlgorithmLib.Kernel
