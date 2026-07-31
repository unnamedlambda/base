import AlgorithmLib.ML.StageFrame

/-!
  # Buffer binding: from kernel-local slots to the pipeline's buffers

  A `StageSpec` describes a kernel over the buffers *the kernel names* — `0`,
  `1`, `2`, the parameters `emitProvenKernelN` declares.  Those numbers are
  local: RMSNorm writes its buffer `2` and so does the SiLU gate, and they are
  not the same memory.  Composing two stages into a `Pipeline` therefore could
  not be done at all, however many kernels had stages — which is exactly where
  the stage work stalled.

  What the host actually does is *bind*: a launch supplies a pointer per slot,
  and two launches of one kernel with different binds are two different steps.
  The machine model already gives each buffer its own address space, so the bind
  is a **renaming of buffer numbers**, not a reindexing of addresses — and that
  makes the bridge a structural induction rather than an aliasing argument.

  * `EWStmt.renameBuf f` — the kernel with its slots relabelled.
  * `WSt.pull f` — the local view of a global state: `pull f st` at buffer `b`
    is `st` at buffer `f b`.
  * `EWStmt.run_renameBuf` — **the simulation**: running the renamed kernel and
    then taking the view is running the original kernel on the view.  Needs `f`
    injective, which is exactly "distinct slots get distinct buffers", i.e. the
    bind list has no duplicates.
  * `StageSpec.rename` — a stage, moved into the pipeline's numbering.

  Integer memory is renamed too (`forM`'s trip count, `ldIdx`'s gather table
  both name a buffer), which is why the stage carries `him`: the global integer
  memory, seen through the bind, must be the one the local stage was proven
  against.
-/

namespace AlgorithmLib.ML

/-- Distinct slots bind to distinct buffers.  Spelled out rather than taken
    from `Function.Injective` so this file needs no import. -/
def BufInj (f : Buf → Buf) : Prop := ∀ b b', f b = f b' → b = b'

theorem BufInj.ne {f : Buf → Buf} (hf : BufInj f) {b b' : Buf} (h : b ≠ b') :
    f b ≠ f b' := fun hc => h (hf b b' hc)

-- ---------------------------------------------------------------------------
-- Renaming
-- ---------------------------------------------------------------------------

def IdxE.renameBuf (f : Buf → Buf) : IdxE → IdxE
  | .ldIdx b off => .ldIdx (f b) (off.renameBuf f)
  | .add a b     => .add (a.renameBuf f) (b.renameBuf f)
  | .mul a b     => .mul (a.renameBuf f) (b.renameBuf f)
  | x            => x

/-- Evaluating a renamed address against global integer memory is evaluating the
    original against the *view* of it. -/
theorem IdxE.eval_renameBuf (f : Buf → Buf) (cta i : Nat) (l : Lane)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ ix : IdxE,
      (ix.renameBuf f).eval cta i l ir im = ix.eval cta i l ir (fun b => im (f b)) := by
  intro ix
  induction ix with
  | ldIdx b off ih => show im (f b) _ = im (f b) _; rw [ih]
  | add a b iha ihb => show _ + _ = _ + _; rw [iha, ihb]
  | mul a b iha ihb => show _ * _ = _ * _; rw [iha, ihb]
  | _ => rfl

def EWStmt.renameBuf (f : Buf → Buf) : EWStmt → EWStmt
  | .skip                 => .skip
  | .seq x y              => .seq (x.renameBuf f) (y.renameBuf f)
  | .setR r e             => .setR r e
  | .shflXor d s m        => .shflXor d s m
  | .loadIdx d b ix       => .loadIdx d (f b) (ix.renameBuf f)
  | .loadV4 a b c d bu ix => .loadV4 a b c d (f bu) (ix.renameBuf f)
  | .storeLane0 b ix r    => .storeLane0 (f b) (ix.renameBuf f) r
  | .storeLane b ix r     => .storeLane (f b) (ix.renameBuf f) r
  | .stSm ix r            => .stSm (ix.renameBuf f) r
  | .ldSm d ix            => .ldSm d (ix.renameBuf f)
  | .barrier              => .barrier
  | .forN n body          => .forN n (body.renameBuf f)
  | .forM bu a body       => .forM (f bu) a (body.renameBuf f)
  | .cvtIF d ix           => .cvtIF d (ix.renameBuf f)

/-- The renamed kernel writes exactly the renamed buffers — so `run_otherBuf`
    still applies, which is what covers buffers the bind never mentions. -/
theorem EWStmt.wbufs_renameBuf (f : Buf → Buf) :
    ∀ s : EWStmt, (s.renameBuf f).wbufs = s.wbufs.map f := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      show (x.renameBuf f).wbufs ++ (y.renameBuf f).wbufs = _
      rw [ihx, ihy]
      exact List.map_append.symm
  | forN _ _ ih => exact ih
  | forM _ _ _ ih => exact ih
  | _ => rfl

-- ---------------------------------------------------------------------------
-- The local view of a global state
-- ---------------------------------------------------------------------------

/-- `pull f st` is what the kernel sees: its slot `b` is the global buffer
    `f b`.  Registers and shared memory are per-warp and unaffected by a bind. -/
def WSt.pull (f : Buf → Buf) (st : WSt) : WSt :=
  { st with mem := fun b => st.mem (f b) }

@[simp] theorem WSt.pull_mem (f : Buf → Buf) (st : WSt) (b : Buf) :
    (WSt.pull f st).mem b = st.mem (f b) := rfl
@[simp] theorem WSt.pull_regs (f : Buf → Buf) (st : WSt) : (WSt.pull f st).regs = st.regs := rfl
@[simp] theorem WSt.pull_smem (f : Buf → Buf) (st : WSt) : (WSt.pull f st).smem = st.smem := rfl

theorem WSt.pull_setReg (f : Buf → Buf) (st : WSt) (r : Nat) (g : Lane → Float32) :
    WSt.pull f (st.setReg r g) = (WSt.pull f st).setReg r g := rfl

/-- A float expression reads registers only, and a bind does not touch those. -/
theorem WFExp.eval_pull (f : Buf → Buf) (st : WSt) (l : Lane) :
    ∀ e : WFExp, e.eval (WSt.pull f st) l = e.eval st l := by
  intro e
  induction e with
  | reg _ => rfl
  | lit _ => rfl
  | add _ _ iha ihb => show NumOps.add _ _ = NumOps.add _ _; rw [iha, ihb]
  | mul _ _ iha ihb => show NumOps.mul _ _ = NumOps.mul _ _; rw [iha, ihb]
  | neg _ ih => show NumOps.neg _ = NumOps.neg _; rw [ih]
  | inv _ ih => show NumOps.inv _ = NumOps.inv _; rw [ih]
  | exp _ ih => show NumOps.exp _ = NumOps.exp _; rw [ih]
  | ex2 _ ih => show NumOps.ex2 _ = NumOps.ex2 _; rw [ih]
  | rsqrt _ ih => show NumOps.rsqrt _ = NumOps.rsqrt _; rw [ih]
  | maxW _ _ iha ihb => show NumOps.max _ _ = NumOps.max _ _; rw [iha, ihb]
  | geF _ _ iha ihb => show NumOps.ifGe _ _ _ _ = NumOps.ifGe _ _ _ _; rw [iha, ihb]

/-- The key commutation: a global store to `f b`, viewed locally, is a local
    store to `b`.  Injectivity is what stops a store to one slot showing up in
    another. -/
theorem WSt.pull_store1 {f : Buf → Buf} (hf : BufInj f) (st : WSt) (b : Buf)
    (a : Nat) (v : Float32) :
    WSt.pull f (st.store1 (f b) a v) = (WSt.pull f st).store1 b a v := by
  refine WSt.ext rfl ?_ rfl
  funext c j
  show (if f c = f b ∧ j = a then v else st.mem (f c) j)
     = (if c = b ∧ j = a then v else st.mem (f c) j)
  by_cases hc : c = b
  · rw [hc]; simp
  · rw [if_neg (fun h => hc (hf c b h.1)), if_neg (fun h => hc h.1)]

/-- …and therefore for the whole per-lane store fold. -/
theorem pull_storeFold {f : Buf → Buf} (hf : BufInj f) (b : Buf) (ix : Lane → Nat)
    (v : Lane → Float32) :
    ∀ (L : List Lane) (st : WSt),
      WSt.pull f (L.foldl (fun s l => s.store1 (f b) (ix l) (v l)) st)
        = L.foldl (fun s l => s.store1 b (ix l) (v l)) (WSt.pull f st) := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons l L ih =>
      intro st
      show WSt.pull f (L.foldl _ (st.store1 (f b) (ix l) (v l))) = _
      rw [ih, WSt.pull_store1 hf]
      rfl

-- ---------------------------------------------------------------------------
-- The simulation
-- ---------------------------------------------------------------------------

/-- Loops: the body commutes at every iteration, so the fold does. -/
theorem pull_forN (f : Buf → Buf) (g g' : Nat → WStmt)
    (h : ∀ j (st : WSt), WSt.pull f ((g' j).run st) = (g j).run (WSt.pull f st)) :
    ∀ (n : Nat) (st : WSt),
      WSt.pull f ((WStmt.forN n g').run st) = (WStmt.forN n g).run (WSt.pull f st) := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro st
      show WSt.pull f ((List.range (n + 1)).foldl (fun s j => (g' j).run s) st)
         = (List.range (n + 1)).foldl (fun s j => (g j).run s) (WSt.pull f st)
      rw [List.range_succ, List.foldl_append, List.foldl_append,
          List.foldl_cons, List.foldl_nil, List.foldl_cons, List.foldl_nil, h n _]
      exact congrArg (fun s => (g n).run s) (ih st)

/-- **Running the bound kernel, viewed through the bind, is running the kernel.**

    The statement a pipeline needs: it says a launch of the *renamed* kernel on
    global memory does, slot by slot, exactly what the stage was proven to do —
    so every theorem about the local stage transfers, rather than being
    reproven per binding.

    The global integer memory is seen through the same rename, which is the
    honest treatment of `forM` and `ldIdx`: those name a buffer too, and a bind
    that moved the float buffers while leaving the meta buffer behind would be a
    different kernel. -/
theorem EWStmt.run_renameBuf {f : Buf → Buf} (hf : BufInj f) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt) (i : Nat) (st : WSt),
      WSt.pull f (((s.renameBuf f).elabAt cta i ir im).run st)
        = (s.elabAt cta i ir (fun b => im (f b))).run (WSt.pull f st) := by
  intro s
  induction s with
  | skip => intro _ _; rfl
  | seq x y ihx ihy =>
      intro i st
      show WSt.pull f (((y.renameBuf f).elabAt cta i ir im).run
             (((x.renameBuf f).elabAt cta i ir im).run st)) = _
      rw [ihy i _]
      exact congrArg (fun s' => (y.elabAt cta i ir (fun b => im (f b))).run s') (ihx i st)
  | setR r e =>
      intro i st
      show WSt.pull f (st.setReg r (fun l => e.eval st l)) = _
      rw [WSt.pull_setReg]
      exact congrArg _ (funext fun l => (WFExp.eval_pull f st l e).symm)
  | shflXor _ _ _ => intro _ _; rfl
  | loadIdx d b ix =>
      intro i st
      show WSt.pull f (st.setReg d (fun l => st.mem (f b)
             ((ix.renameBuf f).eval cta i l ir im))) = _
      rw [WSt.pull_setReg]
      refine congrArg _ (funext fun l => ?_)
      rw [IdxE.eval_renameBuf]
      rfl
  | loadV4 a b c d bu ix =>
      intro i st
      show WSt.pull f ((((st.setReg a _).setReg b _).setReg c _).setReg d _) = _
      rw [WSt.pull_setReg, WSt.pull_setReg, WSt.pull_setReg, WSt.pull_setReg]
      show ((((WSt.pull f st).setReg a (fun l => st.mem (f bu)
                ((ix.renameBuf f).eval cta i l ir im))).setReg b _).setReg c _).setReg d _ = _
      simp only [IdxE.eval_renameBuf]
      rfl
  | storeLane0 b ix r =>
      intro i st
      show WSt.pull f (st.store1 (f b) ((ix.renameBuf f).eval cta i ⟨0, by decide⟩ ir im)
             (st.regs r ⟨0, by decide⟩)) = _
      rw [WSt.pull_store1 hf, IdxE.eval_renameBuf]
      rfl
  | storeLane b ix r =>
      intro i st
      show WSt.pull f ((List.finRange W).foldl
             (fun s l => s.store1 (f b) ((ix.renameBuf f).eval cta i l ir im)
               (st.regs r l)) st) = _
      rw [pull_storeFold hf]
      show (List.finRange W).foldl
             (fun s l => s.store1 b ((ix.renameBuf f).eval cta i l ir im)
               (st.regs r l)) (WSt.pull f st) = _
      simp only [IdxE.eval_renameBuf]
      rfl
  | stSm ix r =>
      intro i st
      show WSt.pull f { st with smem := _ } = _
      simp only [IdxE.eval_renameBuf]
      rfl
  | ldSm d ix =>
      intro i st
      show WSt.pull f (st.setReg d (fun l => st.smem
             ((ix.renameBuf f).eval cta i l ir im))) = _
      rw [WSt.pull_setReg]
      refine congrArg _ (funext fun l => ?_)
      rw [IdxE.eval_renameBuf]
      rfl
  | barrier => intro _ _; rfl
  | forN n body ih => intro _ st; exact pull_forN f _ _ (fun j s' => ih j s') n st
  | forM bu a body ih =>
      intro _ st
      show WSt.pull f ((WStmt.forN (im (f bu) a) _).run st) = _
      exact pull_forN f _ _ (fun j s' => ih j s') (im (f bu) a) st
  | cvtIF d ix =>
      intro i st
      show WSt.pull f (st.setReg d (fun l => NumOps.ofNat
             ((ix.renameBuf f).eval cta i l ir im))) = _
      rw [WSt.pull_setReg]
      refine congrArg _ (funext fun l => ?_)
      rw [IdxE.eval_renameBuf]

/-- The read-back form: a global buffer in the image of the bind holds what the
    local stage put in the corresponding slot. -/
theorem EWStmt.run_renameBuf_at {f : Buf → Buf} (hf : BufInj f) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (s : EWStmt) (i : Nat) (st : WSt)
    (b : Buf) (a : Nat) :
    (((s.renameBuf f).elabAt cta i ir im).run st).mem (f b) a
      = ((s.elabAt cta i ir (fun c => im (f c))).run (WSt.pull f st)).mem b a := by
  have h := EWStmt.run_renameBuf hf cta ir im s i st
  exact congrFun (congrFun (congrArg WSt.mem h) b) a

-- ---------------------------------------------------------------------------
-- Which integer buffers an address reads
-- ---------------------------------------------------------------------------

/-! A bind has to carry the *meta* buffer along with the float buffers, or a
    stage would describe a launch reading its sequence length from wherever the
    old numbering put it.  But most kernels read no integer memory at all, and
    demanding the obligation from them would make every element-wise stage
    unbindable — the requirement `gim (f b) = 0` for every `b` is simply false.

    So the obligation is asked only about the buffers the kernel names. -/

def IdxE.ibufs : IdxE → List Buf
  | .ldIdx b off => b :: off.ibufs
  | .add a b     => a.ibufs ++ b.ibufs
  | .mul a b     => a.ibufs ++ b.ibufs
  | _            => []

def EWStmt.ibufs : EWStmt → List Buf
  | .seq x y              => x.ibufs ++ y.ibufs
  | .loadIdx _ _ ix       => ix.ibufs
  | .loadV4 _ _ _ _ _ ix  => ix.ibufs
  | .storeLane0 _ ix _    => ix.ibufs
  | .storeLane _ ix _     => ix.ibufs
  | .stSm ix _            => ix.ibufs
  | .ldSm _ ix            => ix.ibufs
  | .cvtIF _ ix           => ix.ibufs
  | .forN _ body          => body.ibufs
  | .forM bu _ body       => bu :: body.ibufs
  | _                     => []

theorem IdxE.eval_imCongr (cta i : Nat) (l : Lane) (ir : Nat → Lane → Nat)
    (im im' : Buf → Nat → Nat) :
    ∀ (ix : IdxE), (∀ b ∈ ix.ibufs, im b = im' b) →
      ix.eval cta i l ir im = ix.eval cta i l ir im' := by
  intro ix
  induction ix with
  | ldIdx b off ih =>
      intro h
      show im b _ = im' b _
      rw [ih (fun c hc => h c (List.mem_cons_of_mem _ hc)),
          h b (List.mem_cons_self ..)]
  | add a b iha ihb =>
      intro h
      show _ + _ = _ + _
      rw [iha (fun c hc => h c (List.mem_append_left _ hc)),
          ihb (fun c hc => h c (List.mem_append_right _ hc))]
  | mul a b iha ihb =>
      intro h
      show _ * _ = _ * _
      rw [iha (fun c hc => h c (List.mem_append_left _ hc)),
          ihb (fun c hc => h c (List.mem_append_right _ hc))]
  | _ => intro _; rfl

/-- **A kernel elaborates the same against any integer memory agreeing on the
    buffers it names.** -/
theorem EWStmt.elabAt_imCongr (cta : Nat) (ir : Nat → Lane → Nat)
    (im im' : Buf → Nat → Nat) :
    ∀ (s : EWStmt) (i : Nat), (∀ b ∈ s.ibufs, im b = im' b) →
      s.elabAt cta i ir im = s.elabAt cta i ir im' := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      intro i h
      show WStmt.seq _ _ = WStmt.seq _ _
      rw [ihx i (fun c hc => h c (List.mem_append_left _ hc)),
          ihy i (fun c hc => h c (List.mem_append_right _ hc))]
  | loadIdx d b ix =>
      intro i h
      show WStmt.loadIdx d b _ = WStmt.loadIdx d b _
      exact congrArg _ (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | loadV4 a b c d bu ix =>
      intro i h
      show WStmt.loadV4 a b c d bu _ = WStmt.loadV4 a b c d bu _
      exact congrArg _ (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | storeLane0 b ix r =>
      intro i h
      show WStmt.storeLane0 b _ r = WStmt.storeLane0 b _ r
      rw [IdxE.eval_imCongr cta i ⟨0, by decide⟩ ir im im' ix h]
  | storeLane b ix r =>
      intro i h
      show WStmt.storeLane b _ r = WStmt.storeLane b _ r
      exact congrArg (fun g => WStmt.storeLane b g r)
        (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | stSm ix r =>
      intro i h
      show WStmt.stSmem _ r = WStmt.stSmem _ r
      exact congrArg (fun g => WStmt.stSmem g r)
        (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | ldSm d ix =>
      intro i h
      show WStmt.ldSmem d _ = WStmt.ldSmem d _
      exact congrArg _ (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | cvtIF d ix =>
      intro i h
      show WStmt.setLaneF d _ = WStmt.setLaneF d _
      exact congrArg _ (funext fun l => IdxE.eval_imCongr cta i l ir im im' ix h)
  | forN n body ih =>
      intro _ h
      show WStmt.forN n _ = WStmt.forN n _
      exact congrArg _ (funext fun j => ih j h)
  | forM bu a body ih =>
      intro _ h
      show WStmt.forN (im bu a) _ = WStmt.forN (im' bu a) _
      rw [h bu (List.mem_cons_self ..)]
      exact congrArg _ (funext fun j =>
        ih j (fun c hc => h c (List.mem_cons_of_mem _ hc)))
  | _ => intro _ _; rfl

-- ---------------------------------------------------------------------------
-- A stage, bound
-- ---------------------------------------------------------------------------

open Classical in
/-- **The same kernel, as a step of the pipeline.**

    `f` is the launch's bind: slot `b` of the kernel is buffer `f b` of the
    model.  Everything the local stage proved transfers — the frame condition,
    the value, the ownership — with no new reasoning about the kernel, because
    `run_renameBuf` already said the two runs agree.

    `him` is the obligation that makes this honest: the model's integer memory,
    seen through the same bind, must be the one the local stage was proven
    against.  Without it a bind could move the float buffers and leave the meta
    buffer pointing elsewhere, and the stage would describe a launch that reads
    its sequence length from the wrong place. -/
def StageSpec.rename (S : StageSpec) (f : Buf → Buf) (hf : BufInj f)
    (gim : Buf → Nat → Nat) (him : ∀ b ∈ S.ew.ibufs, gim (f b) = S.imem b) : StageSpec where
  ew    := S.ew.renameBuf f
  iregs := S.iregs
  imem  := gim
  grid  := S.grid
  out   := f S.out
  dom   := S.dom
  val   := fun m cta a => S.val (fun b => m (f b)) cta a
  frame := by
    intro cta st b a h
    have hgim : S.ew.elabAt cta 0 S.iregs (fun c => gim (f c))
        = S.ew.elabAt cta 0 S.iregs S.imem :=
      EWStmt.elabAt_imCongr cta S.iregs _ _ S.ew 0 him
    by_cases hb : ∃ c, f c = b
    · obtain ⟨c, hc⟩ := hb
      subst hc
      rw [EWStmt.run_renameBuf_at hf cta S.iregs gim S.ew 0 st c a, hgim]
      refine (S.frame cta (WSt.pull f st) c a ?_).trans rfl
      rcases h with h | h
      · exact Or.inl (fun hcon => h (by rw [hcon]))
      · exact Or.inr h
    · refine congrFun (EWStmt.run_otherBuf b cta S.iregs gim (S.ew.renameBuf f)
        (fun c hc => ?_) 0 st) a
      rw [EWStmt.wbufs_renameBuf] at hc
      obtain ⟨x, _, hx⟩ := List.mem_map.mp hc
      exact fun hcon => hb ⟨x, hx.trans hcon⟩
  value := by
    intro cta st a hdom
    have hgim : S.ew.elabAt cta 0 S.iregs (fun c => gim (f c))
        = S.ew.elabAt cta 0 S.iregs S.imem :=
      EWStmt.elabAt_imCongr cta S.iregs _ _ S.ew 0 him
    rw [EWStmt.run_renameBuf_at hf cta S.iregs gim S.ew 0 st S.out a, hgim]
    exact S.value cta (WSt.pull f st) a hdom
  valOnly := by
    intro m m' cta a hdom hb hout
    exact S.valOnly (fun c => m (f c)) (fun c => m' (f c)) cta a hdom
      (fun c hc => hb (f c) (hf.ne hc)) hout

@[simp] theorem StageSpec.rename_grid (S : StageSpec) (f hf gim him) :
    (S.rename f hf gim him).grid = S.grid := rfl
@[simp] theorem StageSpec.rename_dom (S : StageSpec) (f hf gim him) :
    (S.rename f hf gim him).dom = S.dom := rfl
@[simp] theorem StageSpec.rename_out (S : StageSpec) (f : Buf → Buf) (hf gim him) :
    (S.rename f hf gim him).out = f S.out := rfl

/-- Binding does not disturb ownership: `dom` and `grid` are carried across
    unchanged, so a stage that was exclusive stays exclusive. -/
theorem StageSpec.rename_exclusive (S : StageSpec) (f : Buf → Buf) (hf : BufInj f)
    (gim : Buf → Nat → Nat) (him : ∀ b ∈ S.ew.ibufs, gim (f b) = S.imem b)
    (hex : S.Exclusive) :
    (S.rename f hf gim him).Exclusive := hex

/-- **The bind that does nothing** — a sanity instantiation, so `rename` cannot
    be vacuously true of a `f` no launch could use. -/
theorem rename_id (S : StageSpec) (him : ∀ b ∈ S.ew.ibufs, S.imem b = S.imem b) :
    (S.rename id (fun _ _ h => h) S.imem him).out = S.out := rfl

-- ---------------------------------------------------------------------------
-- Concrete binds
-- ---------------------------------------------------------------------------

/-- **A three-slot bind.**  The kernel's `%rd0/%rd1/%rd2` — the parameters
    `emitProvenKernelN "main" 3` declares — become model buffers `a`, `b`, `c`.

    Slots the kernel does not have are sent past `1000`, which is not padding:
    `run_renameBuf` needs the map injective on *all* of `Buf`, because `pull` is
    total.  Keeping the model's own buffers below `1000` is therefore the side
    condition, and it is what the `< 1000` hypotheses below check. -/
def bind3 (a b c : Buf) : Buf → Buf := fun x =>
  if x = 0 then a else if x = 1 then b else if x = 2 then c else x + 1000

/-- The one arithmetic fact each case needs, isolated so the case analysis is
    mechanical. -/
theorem bind3_val (a b c : Buf) (x : Buf) (hx : ¬ x = 0) (hx1 : ¬ x = 1) (hx2 : ¬ x = 2) :
    bind3 a b c x = x + 1000 := by
  show (if x = 0 then a else if x = 1 then b else if x = 2 then c else x + 1000) = _
  rw [if_neg hx, if_neg hx1, if_neg hx2]

@[simp] theorem bind3_0 (a b c : Buf) : bind3 a b c 0 = a := rfl
@[simp] theorem bind3_1 (a b c : Buf) : bind3 a b c 1 = b := rfl
@[simp] theorem bind3_2 (a b c : Buf) : bind3 a b c 2 = c := rfl

/-- A model buffer and an out-of-range slot can never coincide. -/
theorem bind3_far {v y : Buf} (hv : v < 1000) (h : v = y + 1000) : False :=
  absurd (h ▸ hv) (Nat.not_lt.mpr (Nat.le_add_left 1000 y))

theorem bind3_inj {a b c : Buf} (ha : a < 1000) (hb : b < 1000) (hc : c < 1000)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : BufInj (bind3 a b c) := by
  intro x y h
  by_cases hx0 : x = 0
  · subst hx0
    by_cases hy0 : y = 0
    · exact hy0.symm
    · by_cases hy1 : y = 1
      · exact absurd h (by rw [bind3_0, hy1, bind3_1]; exact hab)
      · by_cases hy2 : y = 2
        · exact absurd h (by rw [bind3_0, hy2, bind3_2]; exact hac)
        · rw [bind3_0, bind3_val a b c y hy0 hy1 hy2] at h; exact (bind3_far ha h).elim
  · by_cases hx1 : x = 1
    · subst hx1
      by_cases hy0 : y = 0
      · exact absurd h (by rw [bind3_1, hy0, bind3_0]; exact fun hcon => hab hcon.symm)
      · by_cases hy1 : y = 1
        · exact hy1.symm
        · by_cases hy2 : y = 2
          · exact absurd h (by rw [bind3_1, hy2, bind3_2]; exact hbc)
          · rw [bind3_1, bind3_val a b c y hy0 hy1 hy2] at h; exact (bind3_far hb h).elim
    · by_cases hx2 : x = 2
      · subst hx2
        by_cases hy0 : y = 0
        · exact absurd h (by rw [bind3_2, hy0, bind3_0]; exact fun hcon => hac hcon.symm)
        · by_cases hy1 : y = 1
          · exact absurd h (by rw [bind3_2, hy1, bind3_1]; exact fun hcon => hbc hcon.symm)
          · by_cases hy2 : y = 2
            · exact hy2.symm
            · rw [bind3_2, bind3_val a b c y hy0 hy1 hy2] at h; exact (bind3_far hc h).elim
      · rw [bind3_val a b c x hx0 hx1 hx2] at h
        by_cases hy0 : y = 0
        · rw [hy0, bind3_0] at h; exact (bind3_far ha h.symm).elim
        · by_cases hy1 : y = 1
          · rw [hy1, bind3_1] at h; exact (bind3_far hb h.symm).elim
          · by_cases hy2 : y = 2
            · rw [hy2, bind3_2] at h; exact (bind3_far hc h.symm).elim
            · rw [bind3_val a b c y hy0 hy1 hy2] at h; exact Nat.add_right_cancel h


end AlgorithmLib.ML
