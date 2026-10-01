module
public import Host.StaticChecks
meta import Host.StaticChecks
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.StaticHoare
meta import AlgorithmLib.Host.StaticHoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgHoareChecks

open HProgCudaCorpus (SRC KERNEL MEM image)
open HProgStaticChecks (BA launchOn)

/-- Launch on one stream as many times as the first input byte says; with
    `synced`, wait for the stream before downloading. -/
def byInput (synced : Bool) : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  cudaInit ptr
  let c ← cudaCtx ptr
  let s1 ← c.streamCreate
  let bX ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bX (← iconst64 SRC) (← iconst64 64)
  storeI32 bX (← iadd ptr (← iconst64 BA))
  let n ← uload8_64 (← dataPtr)
  forLoop n fun _ => do
    let _ ← launchOn ptr c s1
  if synced then
    let _ ← c.streamSync s1
  let _ ← ffi .cudaDownload %[c.ptr, bX, out, ← iconst64 64]
  cudaCleanup ptr

-- ---------------------------------------------------------------------------
-- The emitted body
-- ---------------------------------------------------------------------------

open AlgorithmLib.HProg (Stmt Piece Op Loop Code)
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.DevSpec
open HProgStaticChecks (canon imageKnown args)

def ld64 (a : Nat) : Stmt := .op (.load { kind := .plain, ty := .i64, notrapAligned := false } a)

/-- Context, stream, buffer, upload, binding, the trip count, the counter. -/
def setup : List Stmt :=
  [.op (.iconst .i64 16), .op (.iadd 0 5), .callVoid (.ffi .cudaInit) [6],
   .op (.iconst .i64 16), .op (.iadd 0 7), ld64 8,
   .call (.ffi .cudaStreamCreate) [9],
   .op (.iconst .i64 64), .op (.iconst .i64 16), .op (.iadd 0 12), ld64 13,
   .call (.ffi .cudaCreateBuffer) [14, 11],
   .op (.iconst .i64 256), .op (.iconst .i64 64), .op (.iconst .i64 16), .op (.iadd 0 18), ld64 19,
   .op (.iadd 0 16),
   .call (.ffi .cudaUpload) [20, 15, 21, 17],
   .op (.iconst .i64 1792), .op (.iadd 0 23), .store .i32 15 24,
   .op (.load { kind := .uload8, ty := .i64, notrapAligned := false } 1),
   .op (.iconst .i64 0)]

def head : Code := [.straight [.op (.icmp .ult 27 25)]]

/-- One trip: a launch on the stream, then the counter. -/
def trip : List Stmt :=
  [.op (.iconst .i32 1), .op (.iconst .i64 384), .op (.iadd 0 31), .op (.iconst .i64 1792),
   .op (.iadd 0 33), .op (.iconst .i32 64),
   .call (.ffi .cudaLaunchOnStream) [9, 32, 30, 34, 30, 30, 30, 35, 30, 30, 10],
   .op (.iconst .i64 1), .op (.iadd 29 37)]

def theLoop : Loop :=
  { pTys := [.i64], init := [26], flag := 28, exitOnTrue := false, cont := [38], exitR := [],
    exitTys := [] }

def finish : List Stmt :=
  [.call (.ffi .cudaStreamSync) [9, 10], .op (.iconst .i64 64),
   .call (.ffi .cudaDownload) [9, 15, 3, 40], .op (.iconst .i64 16), .op (.iadd 0 42),
   .callVoid (.ffi .cudaCleanup) [43]]

def code : Code := [.straight setup, .loop theLoop head [.straight trip], .straight finish]

theorem emit_eq : Prog.emit (byInput true) = code := by rfl

-- ---------------------------------------------------------------------------
-- The setup, by the interpreter
-- ---------------------------------------------------------------------------

def a0 : Static.AS := Static.entry args canon imageKnown
def aS : Static.AS := (goOf (Static.aStmts a0 setup)).getD a0

theorem setup_go : goOf (Static.aStmts a0 setup) = some aS := by
  have h : (goOf (Static.aStmts a0 setup)).isSome = true := by native_decide
  unfold aS
  cases e : goOf (Static.aStmts a0 setup) with
  | none => rw [e] at h; cases h
  | some a => rfl

def knownAt (E : Static.AEnv) (i : Nat) : ClifTy → UInt64 → Bool
  | .i64, b => match E[i]? with
    | some (.known (.sc .i64 b')) => b' == b
    | _ => false
  | .i32, b => match E[i]? with
    | some (.known (.sc .i32 b')) => b' == b
    | _ => false
  | _, _ => false

theorem knownAt_eq {E : Static.AEnv} {i : Nat} {t : ClifTy} {b : UInt64} (h : knownAt E i t b = true) :
    E[i]? = some (.known (.sc t b)) := by
  cases t <;> simp only [knownAt] at h <;> try cases h
  all_goals
    split at h
    · rename_i b' he
      simp only [beq_iff_eq] at h
      rw [he, h]
    · cases h

-- What the interpreter knows once the setup has run, computed.
theorem aS_size : aS.env.size = 27 := by native_decide
theorem aS_0 : knownAt aS.env 0 .i64 0x1000000000 = true := by native_decide
theorem aS_3 : knownAt aS.env 3 .i64 0x3000000000 = true := by native_decide
theorem aS_9 : knownAt aS.env 9 .i64 cudaCtx = true := by native_decide
theorem aS_10 : knownAt aS.env 10 .i32 0 = true := by native_decide
theorem aS_15 : knownAt aS.env 15 .i32 0 = true := by native_decide
theorem aS_live : aS.w.dev.live = true := by native_decide
theorem aS_cap : aS.w.dev.capture.isNone = true := by native_decide
theorem aS_stream : aS.w.dev.streams.getD 0 false = true := by native_decide
theorem aS_buf : (aS.w.dev.get? 0).map ByteArray.size = some 64 := by native_decide
theorem aS_ready : readyB aS.w.dev.race 2 0 true = true := by native_decide
theorem aS_frozen : aS.w.mem.frozen = false := by native_decide
theorem aS_busy : aS.w.mem.busy.isEmpty = true := by native_decide
theorem aS_out : aS.w.mem.out.size = 64 := by native_decide
theorem aS_arena : 24 ≤ aS.w.mem.arena.size := by native_decide
theorem aS_kstr : Static.strKnown aS.known aS.w.mem 0x1000000180 = true := by native_decide
theorem aS_kstr' : (readCStr aS.w.mem 0x1000000180).isSome = true := by native_decide
theorem aS_ids : Static.idsKnown aS.known 0x1000000700 1 = true := by native_decide
theorem aS_ids' : readIds aS.w.mem 0x1000000700 1 = some [0] := by native_decide

abbrev vA : V := .sc .i64 0x1000000000
abbrev vO : V := .sc .i64 0x3000000000
abbrev vC : V := .sc .i64 cudaCtx
abbrev vZ : V := .sc .i32 0

/-- The slots the loop and the finish read. -/
def fsF : List (Nat × V) := [(0, vA), (3, vO), (9, vC), (10, vZ), (15, vZ)]

theorem has_of_rel {Γ : Env} {w : World} (h : Static.Rel aS Γ w) : Has Γ 27 fsF := by
  refine ⟨h.1.1.symm.trans aS_size, fun p hp => ?_⟩
  simp only [fsF, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl
  · exact h.1.known (knownAt_eq aS_0)
  · exact h.1.known (knownAt_eq aS_3)
  · exact h.1.known (knownAt_eq aS_9)
  · exact h.1.known (knownAt_eq aS_10)
  · exact h.1.known (knownAt_eq aS_15)

-- ---------------------------------------------------------------------------
-- The loop invariant
-- ---------------------------------------------------------------------------

/-- What holds at the head of every trip: host memory as the setup left it, the
    device with its context, its stream and its buffer, and the stream free to
    write the buffer. Nothing about how many trips have run. -/
structure Inv (w : World) : Prop where
  mem : Static.MemSame w.mem aS.w.mem
  agree : Static.Agree aS.known w.mem aS.w.mem
  live : w.dev.live = true
  cap : w.dev.capture = none
  stream : w.dev.streams.getD 0 false = true
  buf : (w.dev.get? 0).map ByteArray.size = some 64
  ready : Ready w.dev.race 2 0 true
  kernel : Static.KeepsSize w.kernel

theorem inv_of_rel {Γ : Env} {w : World} (h : Static.Rel aS Γ w) : Inv w := by
  have hs := h.2
  refine ⟨hs.mem, hs.agree, (Static.erase_live hs.dev).trans aS_live, ?_, ?_, ?_, ?_, hs.kernel₁⟩
  · rw [Static.erase_capture hs.dev]; simpa using aS_cap
  · rw [Static.erase_streams hs.dev]; exact aS_stream
  · rcases Static.get?_same hs.dev 0 with ⟨h1, h2⟩ | ⟨b₁, b₂, h1, h2, h3⟩
    · have := aS_buf; rw [h2] at this; cases this
    · have := aS_buf; rw [h2] at this; rw [h1, Option.map_some, h3]; exact this
  · rw [Static.erase_race hs.dev]; exact ready_of_readyB aS_ready

theorem Inv.frozen {w : World} (h : Inv w) : w.mem.frozen = false := h.mem.frozen.trans aS_frozen

theorem Inv.busy {w : World} (h : Inv w) : w.mem.busy = [] := by
  rw [h.mem.busy]; simpa using aS_busy

theorem asI32_zero : asI32 0 = 0 := by decide

theorem Inv.party {w : World} (h : Inv w) : w.dev.party? (asI32 0) = some 2 := by
  have := h.stream
  simp only [Dev.party?, asI32_zero]
  simp [this, streamParty]

theorem fsF_lt : ∀ p ∈ fsF, p.1 < 16 := by decide

theorem has_bindAt {Γ : Env} {k n : Nat} (h : Has Γ k fsF) (hk : 16 ≤ k) (hkn : k ≤ n) (vs : List V) :
    Has (bindAt Γ n vs) (n + vs.length) fsF := by
  refine ⟨bindAt_size (by rw [h.1]; exact hkn), fun p hp => ?_⟩
  have := fsF_lt p hp
  rw [bindAt_lt (by omega) (by rw [h.1]; omega)]
  exact h.2 p hp

-- ---------------------------------------------------------------------------
-- One launch keeps the invariant
-- ---------------------------------------------------------------------------

def vsL : List V :=
  [vC, .sc .i64 0x1000000180, .sc .i32 1, .sc .i64 0x1000000700, .sc .i32 1, .sc .i32 1, .sc .i32 1,
   .sc .i32 64, .sc .i32 1, .sc .i32 1, vZ]

def bitsL : List UInt64 := [cudaCtx, 0x1000000180, 1, 0x1000000700, 1, 1, 1, 64, 1, 1, 0]

theorem asI32_one : (asI32 1).toNat = 1 := by decide

theorem Inv.obs {w : World} (h : Inv w) (c : Callee) (vs : List V) : Inv (obsCall w c vs) :=
  ⟨h.mem, h.agree, h.live, h.cap, h.stream, h.buf, h.ready, h.kernel⟩

theorem norm_i64 (x : UInt64) : norm .i64 x = .sc .i64 x := by
  simp only [norm, widthMask]
  rw [show (0xffffffffffffffff : UInt64) = -1 by decide, UInt64.and_neg_one]

theorem launch_ok (cfg : Cfg) (w : World) (hw : Inv w) :
    ∃ r w', callOf cfg.locals (.ffi .cudaLaunchOnStream) vsL
      (obsCall w (.ffi .cudaLaunchOnStream) vsL) = some (r, w') ∧ Inv w' := by
  have hw1 := hw.obs (.ffi .cudaLaunchOnStream) vsL
  rw [callOf_supported rfl _ _ hw1.frozen]
  have hb : vsL.mapM asBits = some bitsL := rfl
  rw [hb]
  show ∃ r w', ffiCudaLaunchOnStream bitsL (obsCall w (.ffi .cudaLaunchOnStream) vsL) = some (r, w') ∧ Inv w'
  generalize obsCall w (.ffi .cudaLaunchOnStream) vsL = w1 at hw1
  have hk : (readCStrAt w1.mem 0x1000000180).isSome = true := by
    show (readCStr w1.mem 0x1000000180).isSome = true
    rw [Static.readCStr_eq hw1.mem hw1.agree aS_kstr]; exact aS_kstr'
  have hids : readIds w1.mem 0x1000000700 (asI32 1).toNat = some [0] := by
    rw [asI32_one, Static.readIds_eq hw1.mem hw1.agree aS_ids]; exact aS_ids'
  have hbinds : ∀ p, w1.dev.party? (asI32 0) = some p → BindsReady w1 p 1 0x1000000700 := by
    intro p hp
    rw [hw1.party] at hp; cases hp
    exact ⟨[0], hids, fun id hid _ => by simp at hid; subst hid; exact hw1.ready⟩
  have hsafe := launchOnStream_safe (w := w1) (ctx := cudaCtx) (gx := 1) (gy := 1) (gz := 1) (bx := 64)
    (by_ := 1) (bz := 1) (Or.inr ⟨rfl, hw1.live⟩) hw1.kernel
    hw1.cap hk hbinds
  obtain ⟨⟨r, w'⟩, hr⟩ := Option.isSome_iff_exists.mp hsafe
  refine ⟨r, w', hr, ?_⟩
  obtain ⟨d, rfl, hd⟩ := launchOnStream_post hw1.kernel hw1.cap hr
  have hm : w1.mem.retire (d.race.clock hostParty) = w1.mem := retire_nil hw1.busy _
  have hmem : Static.MemSame (w1.mem.retire (d.race.clock hostParty)) aS.w.mem := by
    rw [hm]; exact hw1.mem
  have hagree : Static.Agree aS.known (w1.mem.retire (d.race.clock hostParty)) aS.w.mem := by
    rw [hm]; exact hw1.agree
  rcases hd with rfl | ⟨p, ids, hp, hids', hpost⟩
  · exact ⟨hmem, hagree, hw1.live, hw1.cap, hw1.stream, hw1.buf, hw1.ready, hw1.kernel⟩
  · rw [hw1.party] at hp; cases hp
    rw [hids] at hids'; cases hids'
    refine ⟨hmem, hagree, hpost.live.trans hw1.live, hpost.capture.trans hw1.cap,
      hpost.streams ▸ hw1.stream, (hpost.sizes 0).trans hw1.buf, ?_, hw1.kernel⟩
    rcases hpost.race with h | h
    · rw [h]; exact hw1.ready
    · rw [h]; exact ready_opped_self _ _ _ _ (by simp)

-- ---------------------------------------------------------------------------
-- One trip
-- ---------------------------------------------------------------------------

/-- The loop's exit: the finish's slots, and the invariant. -/
def afterLoop : Post := { ok := fun Γ w => Has Γ 39 fsF ∧ Inv w }

/-- The invariant, on one carry. -/
def I : List V → World → Prop := fun cs w => cs.length = 1 ∧ Inv w

macro "ev_const" : tactic =>
  `(tactic| (intro _ _ _ _ hv; simp [evalOp, ofInt, norm, widthMask] at hv; exact hv.symm))

theorem body_ok (cfg : Cfg) (Γ1 : Env) (h1 : Has Γ1 29 fsF) (cs : List V) (hcs : cs.length = 1)
    (w : World) (hw : Inv w) :
    Triple cfg (At (bindAt Γ1 29 cs) w) [.straight trip] (bodyPost I afterLoop theLoop 39) := by
  have hb : Has (bindAt Γ1 29 cs) 30 fsF := by
    have := has_bindAt h1 (by omega) (Nat.le_refl 29) cs; rwa [hcs] at this
  refine cons_rule (R := fun Γ2 w2 => Has Γ2 39 fsF ∧ Inv w2) ?_ (nil_rule ?_)
  · apply straight_rule
    intro Γ' w' ⟨e1, e2⟩
    subst e1; subst e2
    refine (?_ : Stmts cfg (fun Γ w => Has Γ 30 fsF ∧ Inv w) trip _) _ _ ⟨hb, hw⟩
    refine stmts_cons (op_step (.sc .i32 1) (by ev_const)) ?_
    refine stmts_cons (op_step (.sc .i64 384) (by ev_const)) ?_
    refine stmts_cons (op_step (.sc .i64 0x1000000180) ?_) ?_
    · intro Γ m v hh hv
      simp only [evalOp, bin, Sem.get, hh.get (i := 0) (v := vA) (by simp [fsF]),
        hh.get (i := 31) (v := .sc .i64 384) (by simp)] at hv
      simp (config := {decide := true}) [norm_i64] at hv; exact hv.symm
    refine stmts_cons (op_step (.sc .i64 1792) (by ev_const)) ?_
    refine stmts_cons (op_step (.sc .i64 0x1000000700) ?_) ?_
    · intro Γ m v hh hv
      simp only [evalOp, bin, Sem.get, hh.get (i := 0) (v := vA) (by simp [fsF]),
        hh.get (i := 33) (v := .sc .i64 1792) (by simp)] at hv
      simp (config := {decide := true}) [norm_i64] at hv; exact hv.symm
    refine stmts_cons (op_step (.sc .i32 64) (by ev_const)) ?_
    refine stmts_cons (call_step (J' := Inv) ?_) ?_
    · intro Γ w vs hh hw hvs
      have g9 := hh.get (i := 9) (v := vC) (by simp [fsF])
      have g10 := hh.get (i := 10) (v := vZ) (by simp [fsF])
      have g30 := hh.get (i := 30) (v := .sc .i32 1) (by simp)
      have g32 := hh.get (i := 32) (v := .sc .i64 0x1000000180) (by simp)
      have g34 := hh.get (i := 34) (v := .sc .i64 0x1000000700) (by simp)
      have g35 := hh.get (i := 35) (v := .sc .i32 64) (by simp)
      simp only [List.mapM_cons, List.mapM_nil, g9, g10, g30, g32, g34, g35, bind, Option.bind,
        pure, Option.some.injEq] at hvs
      subst hvs
      exact launch_ok cfg w hw
    refine stmts_cons (op_step (.sc .i64 1) (by ev_const)) ?_
    refine stmts_cons (op_rule (Q := fun Γ w => Has Γ 39 fsF ∧ Inv w) ?_) (stmts_nil fun _ _ h => h)
    intro Γ w v ⟨hh, hw⟩ _
    exact ⟨(hh.push' v).drop (fun p hp => by simp [hp]), hw⟩
  · intro Γ2 w2 ⟨_, hw2⟩ next hn
    simp only [theLoop, List.mapM_cons, List.mapM_nil] at hn
    refine ⟨?_, hw2⟩
    cases e : Γ2[38]? with
    | none => simp [e, bind, Option.bind] at hn
    | some x => simp [e, bind, Option.bind, pure] at hn; subst hn; rfl

theorem trip_ok (cfg : Cfg) (Γ : Env) (hΓ : Has Γ 27 fsF) (cs : List V) (hcs : cs.length = 1)
    (w : World) (hw : Inv w) :
    Triple cfg (At (bindAt Γ 27 cs) w) head
      (headPost cfg I afterLoop theLoop head [.straight trip] 27 39 cs) := by
  have hh : Has (bindAt Γ 27 cs) 28 fsF := by
    have := has_bindAt hΓ (by omega) (Nat.le_refl 27) cs; rwa [hcs] at this
  refine cons_rule (R := fun Γ1 w1 => Has Γ1 29 fsF ∧ w1 = w) ?_ (nil_rule ?_)
  · apply straight_rule
    refine stmts_cons (op_rule fun Γ' w' v ⟨e1, e2⟩ _ => ?_) (stmts_nil fun _ _ h => h)
    subst e1; subst e2; exact ⟨hh.push' v, rfl⟩
  · intro Γ1 w1 ⟨h1, e⟩
    subst e
    intro t f _
    refine ⟨fun _ vs hvs => ?_, fun _ => ?_⟩
    · have : vs = [] := by cases hvs; rfl
      subst this
      exact ⟨has_bindAt h1 (by omega) (by omega) [], hw⟩
    · show Triple cfg (At (bindAt Γ1 (slotsOf (27 + 1) head) cs) w1) [.straight trip] _
      have : slotsOf (27 + 1) head = 29 := by rfl
      rw [this]
      exact body_ok cfg Γ1 h1 cs hcs w1 hw

-- ---------------------------------------------------------------------------
-- The finish: wait, download, clean up
-- ---------------------------------------------------------------------------

/-- After the wait: the default stream may read the buffer. -/
structure Synced (w : World) : Prop where
  mem : Static.MemSame w.mem aS.w.mem
  live : w.dev.live = true
  buf : (w.dev.get? 0).map ByteArray.size = some 64
  ready : Ready w.dev.race defaultParty 0 false

theorem aS_decode_out : decodeAddr 0x3000000000 = some (.out, 0) := by decide
theorem aS_decode_ctx : decodeAddr 0x1000000010 = some (.arena, 16) := by decide

theorem finish_ok (cfg : Cfg) :
    Stmts cfg (fun Γ w => Has Γ 39 fsF ∧ Inv w) finish (fun _ _ => True) := by
  refine stmts_cons (call_step (J' := Synced) ?_) ?_
  · intro Γ w vs hh hw hvs
    have g9 := hh.get (i := 9) (v := vC) (by simp [fsF])
    have g10 := hh.get (i := 10) (v := vZ) (by simp [fsF])
    simp only [List.mapM_cons, List.mapM_nil, g9, g10, bind, Option.bind, pure] at hvs
    cases hvs
    have hw1 := hw.obs (.ffi .cudaStreamSync) [vC, vZ]
    rw [callOf_supported rfl _ _ hw1.frozen]
    show ∃ r w', ffiCudaStreamSync [cudaCtx, 0] (obsCall w (.ffi .cudaStreamSync) [vC, vZ]) = some (r, w') ∧ _
    generalize obsCall w (.ffi .cudaStreamSync) [vC, vZ] = w1 at hw1
    rw [streamSync_eff hw1.live hw1.cap hw1.party]
    refine ⟨_, _, rfl, ?_, hw1.live, hw1.buf, ready_sync hw1.ready defaultParty false (fun h => by cases h)⟩
    show Static.MemSame (w1.mem.retire _) aS.w.mem
    rw [retire_nil hw1.busy]; exact hw1.mem
  refine stmts_cons (op_step (.sc .i64 64) (by ev_const)) ?_
  refine stmts_cons (call_step (J' := fun w => Static.MemSame w.mem aS.w.mem) ?_) ?_
  · intro Γ w vs hh hw hvs
    have g3 := hh.get (i := 3) (v := vO) (by simp [fsF])
    have g9 := hh.get (i := 9) (v := vC) (by simp [fsF])
    have g15 := hh.get (i := 15) (v := vZ) (by simp [fsF])
    have g40 := hh.get (i := 40) (v := .sc .i64 64) (by simp)
    simp only [List.mapM_cons, List.mapM_nil, g3, g9, g15, g40, bind, Option.bind, pure] at hvs
    cases hvs
    have hfz : (obsCall w (.ffi .cudaDownload) [vC, vZ, vO, .sc .i64 64]).mem.frozen = false :=
      hw.mem.frozen.trans aS_frozen
    rw [callOf_supported rfl _ _ hfz]
    show ∃ r w', ffiCudaDownload [cudaCtx, 0, 0x3000000000, 64]
      (obsCall w (.ffi .cudaDownload) [vC, vZ, vO, .sc .i64 64]) = some (r, w') ∧ _
    have hw' : Synced (obsCall w (.ffi .cudaDownload) [vC, vZ, vO, .sc .i64 64]) :=
      ⟨hw.mem, hw.live, hw.buf, hw.ready⟩
    clear hw
    generalize obsCall w (.ffi .cudaDownload) [vC, vZ, vO, .sc .i64 64] = w1 at hw' hfz
    have hw := hw'
    have hsafe := download_safe (w := w1) (ctx := cudaCtx) (buf := 0) (dst := 0x3000000000) (size := 64)
      (Or.inr ⟨rfl, hw.live⟩) (fun b hb _ => by
        rw [asI32_zero] at hb ⊢
        refine ⟨fun src hs => ?_, hw.ready⟩
        refine copyIn_isSome_of aS_decode_out (by decide) hfz ?_ (by rw [hs]; decide)
        rw [hs, hw.mem.size .out]; simp only [Mem.region, aS_out]; decide)
    obtain ⟨⟨r, w'⟩, hr⟩ := Option.isSome_iff_exists.mp hsafe
    exact ⟨r, w', hr, (download_mem hr).symm.trans hw.mem⟩
  refine stmts_cons (op_step (.sc .i64 16) (by ev_const)) ?_
  refine stmts_cons (op_step (.sc .i64 0x1000000010) ?_) ?_
  · intro Γ m v hh hv
    simp only [evalOp, bin, Sem.get, hh.get (i := 0) (v := vA) (by simp [fsF]),
      hh.get (i := 42) (v := .sc .i64 16) (by simp)] at hv
    simp (config := {decide := true}) [norm_i64] at hv; exact hv.symm
  refine stmts_cons (callVoid_step (J' := fun _ => True) ?_) (stmts_nil fun _ _ _ => trivial)
  intro Γ w vs hh hw hvs
  have g43 := hh.get (i := 43) (v := .sc .i64 0x1000000010) (by simp)
  simp only [List.mapM_cons, List.mapM_nil, g43, bind, Option.bind, pure] at hvs
  cases hvs
  have hfz : (obsCall w (.ffi .cudaCleanup) [.sc .i64 0x1000000010]).mem.frozen = false :=
    hw.frozen.trans aS_frozen
  rw [callOf_supported rfl _ _ hfz]
  show ∃ r w', ffiCudaCleanup [0x1000000010] (obsCall w (.ffi .cudaCleanup) [.sc .i64 0x1000000010])
    = some (r, w') ∧ True
  have hw' : Static.MemSame (obsCall w (.ffi .cudaCleanup) [.sc .i64 0x1000000010]).mem aS.w.mem := hw
  clear hw
  generalize obsCall w (.ffi .cudaCleanup) [.sc .i64 0x1000000010] = w1 at hw' hfz
  have hw := hw'
  have hsafe := cleanup_safe (w := w1) (store_isSome_of 0 aS_decode_ctx (by decide) hfz (by
    rw [hw.size .arena]; show 16 + 8 ≤ aS.w.mem.arena.size; exact aS_arena))
  obtain ⟨⟨r, w'⟩, hr⟩ := Option.isSome_iff_exists.mp hsafe
  exact ⟨r, w', hr, trivial⟩

-- ---------------------------------------------------------------------------
-- The whole body
-- ---------------------------------------------------------------------------

theorem code_triple (cfg : Cfg) (w : World) (hs : Static.Same imageKnown w canon) :
    Triple cfg (At args.toArray w) code { ok := fun _ _ => True } := by
  refine cons_rule (R := Static.Rel aS) ?_ ?_
  · apply straight_rule
    intro Γ w' ⟨e1, e2⟩
    subst e1; subst e2
    exact static_stmts cfg setup_go _ _ (Static.entry_rel args hs)
  refine cons_rule (R := fun Γ w => Has Γ 39 fsF ∧ Inv w) ?_ ?_
  · apply loop_rule (fun _ cs w => cs.length = 1 ∧ Inv w)
    · intro Γ w cs hr hcs
      refine ⟨?_, inv_of_rel hr⟩
      simp only [theLoop, List.mapM_cons, List.mapM_nil] at hcs
      cases e : Γ[26]? with
      | none => simp [e, bind, Option.bind] at hcs
      | some x => simp [e, bind, Option.bind, pure] at hcs; subst hcs; rfl
    · intro Γ w0 cs w hr ⟨hcs, hw⟩
      have hΓ := has_of_rel hr
      rw [hΓ.1]
      have : slotsOf (slotsOf (27 + theLoop.pTys.length) head + theLoop.pTys.length) [.straight trip] = 39 := by
        rfl
      rw [this]
      exact trip_ok cfg Γ hΓ cs hcs w hw
  refine cons_rule (R := fun _ _ => True) ?_ (nil_rule fun _ _ _ => trivial)
  exact straight_rule (finish_ok cfg)

/-- **No run misuses a foreign call, however many times the input says to
    launch.** The trip count is the first input byte, which the interpreter
    cannot follow; the loop is proven once, by its invariant. -/
theorem byInput_no_misuse {w : World} (hs : Static.Same imageKnown w canon) (cfg : Cfg) (m : String) :
    Sem.run cfg args w (Prog.emit (byInput true)) ≠ .misuse m := by
  rw [emit_eq]
  exact run_safe (code_triple cfg w hs) m

-- The interpreter refuses the body: it would have to unroll a loop whose count
-- it does not know.
#guard !HProgStaticChecks.check (byInput true)

-- Without the wait the download races the launches, on every input that
-- launches at all.
#guard HProgStaticChecks.isMisuse (HProgStaticChecks.runOn (byInput false) 1)
#guard HProgStaticChecks.isMisuse (HProgStaticChecks.runOn (byInput false) 200)
#guard !HProgStaticChecks.isMisuse (HProgStaticChecks.runOn (byInput false) 0)
#guard !HProgStaticChecks.isMisuse (HProgStaticChecks.runOn (byInput true) 0)
#guard !HProgStaticChecks.isMisuse (HProgStaticChecks.runOn (byInput true) 200)

end HProgHoareChecks
