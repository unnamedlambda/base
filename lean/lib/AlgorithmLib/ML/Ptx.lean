import AlgorithmLib.ML.Warp

namespace AlgorithmLib.ML

/-- A float register: either an emitter temporary (`%f`) or a machine register
    (`%fw`).  Disjoint by construction. -/
inductive PReg where
  | tmp : Nat → PReg     -- %f
  | mach : Nat → PReg    -- %fw
  deriving DecidableEq

/-- Machine state for the emitted fragment.  The two float files are separate
    fields — the aliasing bug is unrepresentable. -/
structure PState where
  f    : Nat → Lane → Float32
  fw   : Nat → Lane → Float32
  mem  : Buf → Nat → Float32

namespace PState

def get (ps : PState) : PReg → Lane → Float32
  | .tmp n => ps.f n
  | .mach n => ps.fw n

def set (ps : PState) : PReg → (Lane → Float32) → PState
  | .tmp n, v  => { ps with f := fun x => if x = n then v else ps.f x }
  | .mach n, v => { ps with fw := fun x => if x = n then v else ps.fw x }

@[simp] theorem get_set_same (ps : PState) (r : PReg) (v : Lane → Float32) :
    (ps.set r v).get r = v := by
  cases r <;> simp [get, set]

@[simp] theorem get_set_tmp_mach (ps : PState) (n m : Nat) (v : Lane → Float32) :
    (ps.set (.tmp n) v).get (.mach m) = ps.get (.mach m) := by simp [get, set]

@[simp] theorem get_set_mach_tmp (ps : PState) (n m : Nat) (v : Lane → Float32) :
    (ps.set (.mach n) v).get (.tmp m) = ps.get (.tmp m) := by simp [get, set]

@[simp] theorem get_set_tmp_other (ps : PState) (n m : Nat) (v : Lane → Float32)
    (h : m ≠ n) : (ps.set (.tmp n) v).get (.tmp m) = ps.get (.tmp m) := by
  simp [get, set, h]

@[simp] theorem get_set_mach_other (ps : PState) (n m : Nat) (v : Lane → Float32)
    (h : m ≠ n) : (ps.set (.mach n) v).get (.mach m) = ps.get (.mach m) := by
  simp [get, set, h]

@[simp] theorem mem_set (ps : PState) (r : PReg) (v : Lane → Float32) :
    (ps.set r v).mem = ps.mem := by cases r <;> rfl

end PState

/-- The instruction fragment the emitter actually produces.  Each carries the
    *exact* semantics of the PTX opcode it names. -/
inductive PInstr where
  | movImm  : PReg → Float32 → PInstr            -- mov.f32 d, 0f…
  | mov     : PReg → PReg → PInstr             -- mov.f32 d, s
  | addF    : PReg → PReg → PReg → PInstr      -- add.f32
  | mulF    : PReg → PReg → PReg → PInstr      -- mul.f32
  | negF    : PReg → PReg → PInstr             -- neg.f32
  | divRn   : PReg → PReg → PReg → PInstr      -- div.rn.f32  (exact)
  | sqrtRn  : PReg → PReg → PInstr             -- sqrt.rn.f32 (exact)
  | ex2     : PReg → PReg → PInstr             -- ex2.approx.f32 (declared approx)
  | maxF    : PReg → PReg → PReg → PInstr      -- max.f32
  | setGeF  : PReg → PReg → PReg → PInstr      -- set.ge.f32.f32  (1.0 / 0.0)
  | shflBfly : PReg → PReg → Nat → PInstr      -- shfl.sync.bfly.b32

/-- Instruction semantics, in the *same* `NumOps Float32` operations the spec
    uses.  This is what makes a wrong opcode a proof failure. -/
def PInstr.step (i : PInstr) (ps : PState) : PState :=
  match i with
  | .movImm d v      => ps.set d (fun _ => v)
  | .mov d s         => ps.set d (ps.get s)
  | .addF d a b      => ps.set d (fun l => NumOps.add (ps.get a l) (ps.get b l))
  | .mulF d a b      => ps.set d (fun l => NumOps.mul (ps.get a l) (ps.get b l))
  | .negF d a        => ps.set d (fun l => NumOps.neg (ps.get a l))
  | .divRn d a b     => ps.set d (fun l => (ps.get a l) / (ps.get b l))
  | .sqrtRn d a      => ps.set d (fun l => Float32.sqrt (ps.get a l))
  -- `ex2.approx.f32` computes 2^x, NOT e^x.  It is `NumOps.ex2` — the *same*
  -- opaque function the spec language names, so this is an exact equality like
  -- every other opcode, not an approximation.
  | .ex2 d a         => ps.set d (fun l => NumOps.ex2 (ps.get a l))
  | .maxF d a b      => ps.set d (fun l => NumOps.max (ps.get a l) (ps.get b l))
  | .setGeF d a b    => ps.set d (fun l => NumOps.ifGe (ps.get a l) (ps.get b l) 1.0 0.0)
  | .shflBfly d s m  => ps.set d (fun l => ps.get s (xorLane l m))

def prun (is : List PInstr) (ps : PState) : PState := is.foldl (fun s i => i.step s) ps

@[simp] theorem prun_nil (ps : PState) : prun [] ps = ps := rfl
@[simp] theorem prun_cons (i : PInstr) (is : List PInstr) (ps : PState) :
    prun (i :: is) ps = prun is (i.step ps) := rfl
theorem prun_append (a b : List PInstr) (ps : PState) :
    prun (a ++ b) ps = prun b (prun a ps) := by
  simp [prun, List.foldl_append]


-- ---------------------------------------------------------------------------
-- A verified emitter for the exactly-realisable fragment
-- ---------------------------------------------------------------------------

/-- The fragment the hardware realises, and the emitter therefore covers.

    Only `exp` is excluded, and for a concrete reason: there is no `e^x`
    instruction.  `ex2` **is** included — `ex2.approx.f32` is `NumOps.ex2`,
    named opaquely on both sides exactly like `Float32.exp` is everywhere else,
    so it lowers with an exact equality and no hypothesis.

    Turning `exp` into `ex2` is a *rewrite* (`expandExp`), performed before
    emission and carrying its own declared approximation.  That keeps every
    theorem below this line unconditional. -/
def WFExp.ExpFree : WFExp → Prop
  | .reg _     => True
  | .lit _     => True
  | .add a b   => a.ExpFree ∧ b.ExpFree
  | .mul a b   => a.ExpFree ∧ b.ExpFree
  | .neg a     => a.ExpFree
  | .inv a     => a.ExpFree
  | .rsqrt a   => a.ExpFree
  | .maxW a b  => a.ExpFree ∧ b.ExpFree
  | .geF a b   => a.ExpFree ∧ b.ExpFree
  | .ex2 a     => a.ExpFree
  | .exp _     => False

/-- Emit straight-line code for an expression, threading a temporary counter.
    Machine registers are `.mach`; temporaries are `.tmp` — different
    constructors, so the aliasing bug cannot be written. -/
def emitP : Nat → WFExp → PReg × List PInstr × Nat
  | n, .reg r  => (.mach r, [], n)
  | n, .lit v  => (.tmp n, [.movImm (.tmp n) v], n + 1)
  | n, .add a b =>
      (.tmp (emitP (emitP n a).2.2 b).2.2,
       (emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
         ++ [.addF (.tmp (emitP (emitP n a).2.2 b).2.2)
                   (emitP n a).1 (emitP (emitP n a).2.2 b).1],
       (emitP (emitP n a).2.2 b).2.2 + 1)
  | n, .mul a b =>
      (.tmp (emitP (emitP n a).2.2 b).2.2,
       (emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
         ++ [.mulF (.tmp (emitP (emitP n a).2.2 b).2.2)
                   (emitP n a).1 (emitP (emitP n a).2.2 b).1],
       (emitP (emitP n a).2.2 b).2.2 + 1)
  | n, .maxW a b =>
      (.tmp (emitP (emitP n a).2.2 b).2.2,
       (emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
         ++ [.maxF (.tmp (emitP (emitP n a).2.2 b).2.2)
                   (emitP n a).1 (emitP (emitP n a).2.2 b).1],
       (emitP (emitP n a).2.2 b).2.2 + 1)
  | n, .geF a b =>
      (.tmp (emitP (emitP n a).2.2 b).2.2,
       (emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
         ++ [.setGeF (.tmp (emitP (emitP n a).2.2 b).2.2)
                     (emitP n a).1 (emitP (emitP n a).2.2 b).1],
       (emitP (emitP n a).2.2 b).2.2 + 1)
  | n, .neg a =>
      (.tmp (emitP n a).2.2,
       (emitP n a).2.1 ++ [.negF (.tmp (emitP n a).2.2) (emitP n a).1],
       (emitP n a).2.2 + 1)
  | n, .inv a =>
      (.tmp ((emitP n a).2.2 + 1),
       (emitP n a).2.1 ++ [.movImm (.tmp (emitP n a).2.2) 1.0,
         .divRn (.tmp ((emitP n a).2.2 + 1)) (.tmp (emitP n a).2.2) (emitP n a).1],
       (emitP n a).2.2 + 2)
  | n, .rsqrt a =>
      (.tmp ((emitP n a).2.2 + 2),
       (emitP n a).2.1 ++ [.sqrtRn (.tmp (emitP n a).2.2) (emitP n a).1,
         .movImm (.tmp ((emitP n a).2.2 + 1)) 1.0,
         .divRn (.tmp ((emitP n a).2.2 + 2)) (.tmp ((emitP n a).2.2 + 1))
                (.tmp (emitP n a).2.2)],
       (emitP n a).2.2 + 3)
  -- `ex2.approx.f32`, one instruction, exactly `NumOps.ex2`.
  | n, .ex2 a =>
      (.tmp (emitP n a).2.2,
       (emitP n a).2.1 ++ [.ex2 (.tmp (emitP n a).2.2) (emitP n a).1],
       (emitP n a).2.2 + 1)
  -- `exp` has no instruction.  This case exists only for totality and is
  -- excluded by `ExpFree`; `expandExp` removes it before emission.
  | n, .exp a =>
      (.tmp (emitP n a).2.2,
       (emitP n a).2.1 ++ [.ex2 (.tmp (emitP n a).2.2) (emitP n a).1],
       (emitP n a).2.2 + 1)

/-- Read the warp state a `PState`'s machine file represents. -/
def PState.toWSt (ps : PState) : WSt :=
  { regs := ps.fw, mem := ps.mem, smem := fun _ => 0.0 }


-- ---------------------------------------------------------------------------
-- Soundness of the lowering
-- ---------------------------------------------------------------------------

/-- Registers the emitter is not allowed to disturb: every machine register,
    and every temporary already in use. -/
def PReg.Below (n : Nat) : PReg → Prop
  | .mach _ => True
  | .tmp m  => m < n

/-- Writing a fresh temporary leaves every register already in use alone. -/
theorem PState.get_set_below (ps : PState) (n : Nat) (v : Lane → Float32) (r : PReg)
    (h : r.Below n) : (ps.set (.tmp n) v).get r = ps.get r := by
  cases r with
  | mach m => simp [PState.get, PState.set]
  | tmp m => exact PState.get_set_tmp_other ps n m v (by simp [PReg.Below] at h; omega)

theorem emitP_counter : ∀ (e : WFExp) (n : Nat), n ≤ (emitP n e).2.2 := by
  intro e
  induction e with
  | reg r => intro n; exact Nat.le_refl n
  | lit v => intro n; show n ≤ n + 1; omega
  | add a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitP n a).2.2
      show n ≤ (emitP (emitP n a).2.2 b).2.2 + 1; omega
  | mul a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitP n a).2.2
      show n ≤ (emitP (emitP n a).2.2 b).2.2 + 1; omega
  | maxW a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitP n a).2.2
      show n ≤ (emitP (emitP n a).2.2 b).2.2 + 1; omega
  | geF a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitP n a).2.2
      show n ≤ (emitP (emitP n a).2.2 b).2.2 + 1; omega
  | neg a ih   => intro n; have h := ih n; show n ≤ (emitP n a).2.2 + 1; omega
  | inv a ih   => intro n; have h := ih n; show n ≤ (emitP n a).2.2 + 2; omega
  | rsqrt a ih => intro n; have h := ih n; show n ≤ (emitP n a).2.2 + 3; omega
  | exp a ih   => intro n; have h := ih n; show n ≤ (emitP n a).2.2 + 1; omega
  | ex2 a ih   => intro n; have h := ih n; show n ≤ (emitP n a).2.2 + 1; omega

/-- **The emitter never touches a machine register or a live temporary.**

    This is the theorem the register-aliasing bug violated.  With `%f` and `%fw`
    as separate fields it is provable; with both mapped to `%f` it is false, and
    the proof would not close. -/
theorem emitP_frame : ∀ (e : WFExp) (n : Nat) (ps : PState) (r : PReg),
    r.Below n → (prun (emitP n e).2.1 ps).get r = ps.get r := by
  intro e
  induction e with
  | reg _ => intro n ps r _; rfl
  | lit v =>
      intro n ps r hr
      show (PInstr.step (.movImm (.tmp n) v) ps).get r = _
      cases r with
      | mach m => simp [PInstr.step]
      | tmp m => exact PState.get_set_tmp_other ps n m _ (by simp [PReg.Below] at hr; omega)
  | add a b iha ihb =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
              ++ [PInstr.addF (.tmp _) _ _]) ps).get r = _
      rw [prun_append, prun_append]
      have hb := emitP_counter b (emitP n a).2.2
      have ha := emitP_counter a n
      have hrb : r.Below (emitP (emitP n a).2.2 b).2.2 := by
        cases r with
        | mach _ => trivial
        | tmp m => simp [PReg.Below] at hr ⊢; omega
      have hra : r.Below (emitP n a).2.2 := by
        cases r with
        | mach _ => trivial
        | tmp m => simp [PReg.Below] at hr ⊢; omega
      show (PInstr.step (.addF (.tmp _) _ _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]
                  rw [ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP (emitP n a).2.2 b).2.2 := by
            simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
  | mul a b iha ihb =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
              ++ [PInstr.mulF (.tmp _) _ _]) ps).get r = _
      rw [prun_append, prun_append]
      have hb := emitP_counter b (emitP n a).2.2
      have ha := emitP_counter a n
      have hra : r.Below (emitP n a).2.2 := by
        cases r with
        | mach _ => trivial
        | tmp m => simp [PReg.Below] at hr ⊢; omega
      show (PInstr.step (.mulF (.tmp _) _ _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]
                  rw [ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP (emitP n a).2.2 b).2.2 := by
            simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
  | maxW a b iha ihb =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
              ++ [PInstr.maxF (.tmp _) _ _]) ps).get r = _
      rw [prun_append, prun_append]
      have hb := emitP_counter b (emitP n a).2.2
      have ha := emitP_counter a n
      have hra : r.Below (emitP n a).2.2 := by
        cases r with
        | mach _ => trivial
        | tmp m => simp [PReg.Below] at hr ⊢; omega
      show (PInstr.step (.maxF (.tmp _) _ _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]
                  rw [ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP (emitP n a).2.2 b).2.2 := by
            simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
  | geF a b iha ihb =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1
              ++ [PInstr.setGeF (.tmp _) _ _]) ps).get r = _
      rw [prun_append, prun_append]
      have hb := emitP_counter b (emitP n a).2.2
      have ha := emitP_counter a n
      have hra : r.Below (emitP n a).2.2 := by
        cases r with
        | mach _ => trivial
        | tmp m => simp [PReg.Below] at hr ⊢; omega
      show (PInstr.step (.setGeF (.tmp _) _ _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]
                  rw [ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP (emitP n a).2.2 b).2.2 := by
            simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ihb (emitP n a).2.2 (prun (emitP n a).2.1 ps) _ hra, iha n ps _ hr]
  | neg a ih =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ [PInstr.negF (.tmp _) _]) ps).get r = _
      rw [prun_append]
      have ha := emitP_counter a n
      show (PInstr.step (.negF (.tmp _) _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]; rw [ih n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP n a).2.2 := by simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ih n ps _ hr]
  | inv a ih =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ [PInstr.movImm (.tmp _) 1.0,
              PInstr.divRn (.tmp _) (.tmp _) _]) ps).get r = _
      rw [prun_append]
      have ha := emitP_counter a n
      simp only [prun_cons, prun_nil]
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]; rw [ih n ps _ hr]
      | tmp m =>
          have h1 : m ≠ (emitP n a).2.2 + 1 := by simp [PReg.Below] at hr; omega
          have h2 : m ≠ (emitP n a).2.2 := by simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ h1, PState.get_set_tmp_other _ _ _ _ h2,
              ih n ps _ hr]
  | rsqrt a ih =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ [PInstr.sqrtRn (.tmp _) _,
              PInstr.movImm (.tmp _) 1.0, PInstr.divRn (.tmp _) (.tmp _) (.tmp _)]) ps).get r = _
      rw [prun_append]
      have ha := emitP_counter a n
      simp only [prun_cons, prun_nil]
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]; rw [ih n ps _ hr]
      | tmp m =>
          have h1 : m ≠ (emitP n a).2.2 + 2 := by simp [PReg.Below] at hr; omega
          have h2 : m ≠ (emitP n a).2.2 + 1 := by simp [PReg.Below] at hr; omega
          have h3 : m ≠ (emitP n a).2.2 := by simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ h1, PState.get_set_tmp_other _ _ _ _ h2,
              PState.get_set_tmp_other _ _ _ _ h3, ih n ps _ hr]
  | exp a ih =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ [PInstr.ex2 (.tmp _) _]) ps).get r = _
      rw [prun_append]
      have ha := emitP_counter a n
      show (PInstr.step (.ex2 (.tmp _) _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]; rw [ih n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP n a).2.2 := by simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ih n ps _ hr]
  | ex2 a ih =>
      intro n ps r hr
      show (prun ((emitP n a).2.1 ++ [PInstr.ex2 (.tmp _) _]) ps).get r = _
      rw [prun_append]
      have ha := emitP_counter a n
      show (PInstr.step (.ex2 (.tmp _) _) _).get r = _
      cases r with
      | mach m => simp only [PInstr.step, PState.get_set_tmp_mach]; rw [ih n ps _ hr]
      | tmp m =>
          have hm : m ≠ (emitP n a).2.2 := by simp [PReg.Below] at hr; omega
          simp only [PInstr.step]
          rw [PState.get_set_tmp_other _ _ _ _ hm, ih n ps _ hr]

/-- The machine file is preserved wholesale — a corollary of `emitP_frame`, and
    the statement that makes the emitted code composable. -/
theorem emitP_fw (e : WFExp) (n : Nat) (ps : PState) :
    (prun (emitP n e).2.1 ps).fw = ps.fw := by
  funext m
  exact emitP_frame e n ps (.mach m) trivial

/-- No instruction in this fragment writes memory. -/
theorem PInstr.step_mem (i : PInstr) (ps : PState) : (i.step ps).mem = ps.mem := by
  cases i <;> simp [PInstr.step]

theorem prun_mem : ∀ (is : List PInstr) (ps : PState), (prun is ps).mem = ps.mem := by
  intro is
  induction is with
  | nil => intro ps; rfl
  | cons i is ih => intro ps; rw [prun_cons, ih, PInstr.step_mem]

theorem emitP_mem (e : WFExp) (n : Nat) (ps : PState) :
    (prun (emitP n e).2.1 ps).mem = ps.mem := prun_mem _ ps

/-- The warp state an emitted fragment sees is unchanged by running it. -/
theorem emitP_toWSt (e : WFExp) (n : Nat) (ps : PState) :
    (prun (emitP n e).2.1 ps).toWSt = ps.toWSt := by
  show PState.toWSt _ = PState.toWSt _
  unfold PState.toWSt
  rw [emitP_fw e n ps, emitP_mem e n ps]

/-- The result register lies inside the range the emitter claimed. -/
theorem emitP_res_below : ∀ (e : WFExp) (n : Nat),
    ((emitP n e).1).Below ((emitP n e).2.2) := by
  intro e
  cases e with
  | reg r => intro n; trivial
  | lit v => intro n; show n < n + 1; omega
  | add a b => intro n; show (emitP (emitP n a).2.2 b).2.2 < _ + 1; omega
  | mul a b => intro n; show (emitP (emitP n a).2.2 b).2.2 < _ + 1; omega
  | maxW a b => intro n; show (emitP (emitP n a).2.2 b).2.2 < _ + 1; omega
  | geF a b => intro n; show (emitP (emitP n a).2.2 b).2.2 < _ + 1; omega
  | neg a => intro n; show (emitP n a).2.2 < _ + 1; omega
  | inv a => intro n; show (emitP n a).2.2 + 1 < _ + 2; omega
  | rsqrt a => intro n; show (emitP n a).2.2 + 2 < _ + 3; omega
  | exp a => intro n; show (emitP n a).2.2 < _ + 1; omega
  | ex2 a => intro n; show (emitP n a).2.2 < _ + 1; omega


/-- **A4, for the straight-line fragment: the emitted PTX computes the
    expression.**

    The result register holds exactly `e.eval`, in every lane, with no epsilon.
    Both bugs the unproven lowering shipped are excluded structurally: machine
    registers live in a different field from temporaries (so aliasing is not
    expressible), and every opcode carries the `NumOps Float32` operation it
    implements (so `rcp.approx` where the model says exact division fails to
    typecheck as a proof).

    `ex2` is covered — it is `NumOps.ex2` on both sides, exactly like every
    other opcode.  Only `exp` is excluded, because no instruction computes it;
    `expandExp` rewrites it away first, carrying the approximation openly. -/
theorem emitP_sound : ∀ (e : WFExp), e.ExpFree → ∀ (n : Nat) (ps : PState) (l : Lane),
    (prun (emitP n e).2.1 ps).get (emitP n e).1 l = e.eval ps.toWSt l := by
  intro e
  induction e with
  | reg r => intro _ n ps l; rfl
  | lit v =>
      intro _ n ps l
      show (PInstr.step (.movImm (.tmp n) v) ps).get (.tmp n) l = _
      simp only [PInstr.step, PState.get_set_same]; rfl
  | add a b iha ihb =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1 ++ [_]) ps).get
             (.tmp (emitP (emitP n a).2.2 b).2.2) l = _
      rw [prun_append, prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.add _ _ = NumOps.add _ _
      rw [emitP_frame b (emitP n a).2.2 (prun (emitP n a).2.1 ps) (emitP n a).1
            (emitP_res_below a n),
          iha hx.1 n ps l,
          ihb hx.2 (emitP n a).2.2 (prun (emitP n a).2.1 ps) l,
          emitP_toWSt a n ps]
  | mul a b iha ihb =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1 ++ [_]) ps).get
             (.tmp (emitP (emitP n a).2.2 b).2.2) l = _
      rw [prun_append, prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.mul _ _ = NumOps.mul _ _
      rw [emitP_frame b (emitP n a).2.2 (prun (emitP n a).2.1 ps) (emitP n a).1
            (emitP_res_below a n),
          iha hx.1 n ps l,
          ihb hx.2 (emitP n a).2.2 (prun (emitP n a).2.1 ps) l,
          emitP_toWSt a n ps]
  | maxW a b iha ihb =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1 ++ [_]) ps).get
             (.tmp (emitP (emitP n a).2.2 b).2.2) l = _
      rw [prun_append, prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.max _ _ = NumOps.max _ _
      rw [emitP_frame b (emitP n a).2.2 (prun (emitP n a).2.1 ps) (emitP n a).1
            (emitP_res_below a n),
          iha hx.1 n ps l,
          ihb hx.2 (emitP n a).2.2 (prun (emitP n a).2.1 ps) l,
          emitP_toWSt a n ps]
  | geF a b iha ihb =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ (emitP (emitP n a).2.2 b).2.1 ++ [_]) ps).get
             (.tmp (emitP (emitP n a).2.2 b).2.2) l = _
      rw [prun_append, prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.ifGe _ _ 1.0 0.0 = NumOps.ifGe _ _ 1.0 0.0
      rw [emitP_frame b (emitP n a).2.2 (prun (emitP n a).2.1 ps) (emitP n a).1
            (emitP_res_below a n),
          iha hx.1 n ps l,
          ihb hx.2 (emitP n a).2.2 (prun (emitP n a).2.1 ps) l,
          emitP_toWSt a n ps]
  | neg a ih =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ [_]) ps).get (.tmp (emitP n a).2.2) l = _
      rw [prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.neg _ = NumOps.neg _
      rw [ih hx n ps l]
  | inv a ih =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ [_, _]) ps).get (.tmp ((emitP n a).2.2 + 1)) l = _
      rw [prun_append]; simp only [prun_cons, prun_nil]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      simp only [PInstr.step, PState.get_set_same,
                 PState.get_set_tmp_other _ _ _ _ (show (emitP n a).2.2 ≠ (emitP n a).2.2 + 1 by omega)]
      rw [PState.get_set_below _ (emitP n a).2.2 _ _ (emitP_res_below a n), ih hx n ps l]; rfl
  | rsqrt a ih =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ [_, _, _]) ps).get (.tmp ((emitP n a).2.2 + 2)) l = _
      rw [prun_append]; simp only [prun_cons, prun_nil]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      simp only [PInstr.step, PState.get_set_same,
                 PState.get_set_tmp_other _ _ _ _ (show (emitP n a).2.2 ≠ (emitP n a).2.2 + 1 by omega),
                 PState.get_set_tmp_other _ _ _ _ (show (emitP n a).2.2 + 1 ≠ (emitP n a).2.2 + 2 by omega),
                 PState.get_set_tmp_other _ _ _ _ (show (emitP n a).2.2 ≠ (emitP n a).2.2 + 2 by omega)]
      rw [ih hx n ps l]; rfl
  | exp a ih => intro hx _ _ _; exact absurd hx (by simp [WFExp.ExpFree])
  | ex2 a ih =>
      intro hx n ps l
      show (prun ((emitP n a).2.1 ++ [_]) ps).get (.tmp (emitP n a).2.2) l = _
      rw [prun_append]
      show (PState.set _ (.tmp _) _).get (.tmp _) l = _
      rw [PState.get_set_same]
      show NumOps.ex2 _ = NumOps.ex2 _
      rw [ih hx n ps l]


-- ---------------------------------------------------------------------------
-- Lowering statements: the butterfly reduction's control-free core
-- ---------------------------------------------------------------------------

/-- The statement fragment this lowering covers: register assignment, cross-lane
    shuffle, and sequencing.  That is exactly the shape of a warp reduction
    once its loop has been peeled — `warpReduceSum` is five `warpRound`s. -/
def WStmt.SFrag : WStmt → Prop
  | .skip          => True
  | .seq a b       => a.SFrag ∧ b.SFrag
  | .setR _ e      => e.ExpFree
  | .shflXor _ _ _ => True
  | _              => False

/-- Lower a statement.  `n` is the first free temporary. -/
def emitS (n : Nat) : WStmt → List PInstr
  | .skip          => []
  | .seq a b       => emitS n a ++ emitS n b
  | .setR r e      => (emitP n e).2.1 ++ [.mov (.mach r) (emitP n e).1]
  | .shflXor d s m => [.shflBfly (.mach d) (.mach s) m]
  | _              => []

theorem emitS_mem : ∀ (s : WStmt) (n : Nat) (ps : PState),
    (prun (emitS n s) ps).mem = ps.mem := by
  intro s n ps; exact prun_mem _ ps

/-- Register-only statements touch neither global nor shared memory. -/
theorem SFrag_mem : ∀ (s : WStmt), s.SFrag → ∀ (st : WSt),
    (s.run st).mem = st.mem ∧ (s.run st).smem = st.smem := by
  intro s
  induction s with
  | skip => intro _ st; exact ⟨rfl, rfl⟩
  | seq a b iha ihb =>
      intro hf st
      have h1 := iha hf.1 st
      have h2 := ihb hf.2 (a.run st)
      refine ⟨?_, ?_⟩
      · show (b.run (a.run st)).mem = st.mem
        rw [h2.1, h1.1]
      · show (b.run (a.run st)).smem = st.smem
        rw [h2.2, h1.2]
  | setR r e => intro _ st; exact ⟨rfl, rfl⟩
  | shflXor d s m => intro _ st; exact ⟨rfl, rfl⟩
  | loadIdx _ _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | loadV4 _ _ _ _ _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | forN _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | storeLane0 _ _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | storeLane _ _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | stSmem _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | ldSmem _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | barrier => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | extern _ _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])
  | setLaneF _ _ => intro hf _; exact absurd hf (by simp [WStmt.SFrag])

/-- **The lowered statement reproduces the machine's register file.**

    Composed with `emitP_sound`, this proves the emitted PTX for a warp
    reduction's body computes what `WStmt.run` says — the whole chain from
    `Expr` down to instructions, for the control-free fragment. -/
theorem emitS_sound : ∀ (s : WStmt), s.SFrag → ∀ (n : Nat) (ps : PState),
    (prun (emitS n s) ps).fw = (s.run ps.toWSt).regs := by
  intro s
  induction s with
  | skip => intro _ n ps; rfl
  | seq a b iha ihb =>
      intro hf n ps
      show (prun (emitS n a ++ emitS n b) ps).fw = _
      rw [prun_append]
      have hmem : (prun (emitS n a) ps).toWSt = (a.run ps.toWSt) := by
        have hsm := SFrag_mem a hf.1 ps.toWSt
        refine WSt.ext ?_ ?_ ?_
        · show (prun (emitS n a) ps).fw = (a.run ps.toWSt).regs
          exact iha hf.1 n ps
        · show (prun (emitS n a) ps).mem = (a.run ps.toWSt).mem
          rw [emitS_mem a n ps, hsm.1]; rfl
        · show (fun _ => (0.0:Float32)) = (a.run ps.toWSt).smem
          rw [hsm.2]; rfl
      rw [ihb hf.2 n (prun (emitS n a) ps), hmem]
      rfl
  | setR r e =>
      intro hf n ps
      show (prun ((emitP n e).2.1 ++ [PInstr.mov (.mach r) (emitP n e).1]) ps).fw = _
      rw [prun_append]
      funext m l
      show ((PInstr.mov (.mach r) (emitP n e).1).step (prun (emitP n e).2.1 ps)).fw m l = _
      by_cases hm : m = r
      · subst hm
        show (PState.set _ (.mach m) _).fw m l = _
        have : ((prun (emitP n e).2.1 ps).set (.mach m)
                  ((prun (emitP n e).2.1 ps).get (emitP n e).1)).get (.mach m) l
              = (prun (emitP n e).2.1 ps).get (emitP n e).1 l :=
          congrFun (PState.get_set_same _ (.mach m) _) l
        show ((prun (emitP n e).2.1 ps).set (.mach m) _).get (.mach m) l = _
        rw [this, emitP_sound e hf n ps l]
        show _ = (WSt.setReg ps.toWSt m _).regs m l
        rw [WSt.regs_setReg_same]
      · show ((prun (emitP n e).2.1 ps).set (.mach r) _).get (.mach m) l = _
        rw [PState.get_set_mach_other _ r m _ hm]
        show (prun (emitP n e).2.1 ps).fw m l = (WSt.setReg ps.toWSt r _).regs m l
        rw [WSt.regs_setReg_other _ r m _ hm, emitP_fw e n ps]
        rfl
  | shflXor d s m =>
      intro _ n ps
      funext x l
      show ((PInstr.shflBfly (.mach d) (.mach s) m).step ps).fw x l = _
      by_cases hx : x = d
      · subst hx
        show ((ps.set (.mach x) _).get (.mach x)) l = _
        rw [PState.get_set_same]
        show ps.get (.mach s) (xorLane l m) = (WSt.setReg ps.toWSt x _).regs x l
        rw [WSt.regs_setReg_same]
        rfl
      · show ((ps.set (.mach d) _).get (.mach x)) l = _
        rw [PState.get_set_mach_other _ d x _ hx]
        show ps.fw x l = (WSt.setReg ps.toWSt d _).regs x l
        rw [WSt.regs_setReg_other _ d x _ hx]
        rfl
  | loadIdx _ _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | loadV4 _ _ _ _ _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | forN _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | storeLane0 _ _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | storeLane _ _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | stSmem _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | ldSmem _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | barrier => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | extern _ _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])
  | setLaneF _ _ => intro hf _ _; exact absurd hf (by simp [WStmt.SFrag])

end AlgorithmLib.ML
