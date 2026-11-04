module
public import AlgorithmLib.Sound.PtxSound
public import AlgorithmLib.Kernel.Run
public import AlgorithmLib.Host.Sound
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Sound.PtxThread` — a kernel thread's PTX ends where its body does

A kernel's body, run by the host semantics as one of its threads
(`Kernel.thread`), and the entry it compiles to, run by PTX's (`PtxRun.thread`):
when the first ends, the second does, in a world related to the first's
(`WRel`). The body runs as its blocks (`compile_core`); the prologue puts the
thread's arguments where the entry block's parameters are held
(`prologue_sim`); the blocks run as their lowerings (`run_sim`), each
instruction meeting its obligation (`insts_ok`).
-/

namespace AlgorithmLib.Kernel.PtxSound

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem Ptx

/-- Worlds the same apart from their observations are related from where
    they are. -/
theorem wrel_start (w pw : World) (h : { w with obs := [] } = { pw with obs := [] }) :
    WRel w.obs.length pw.obs.length w pw := by
  refine ⟨h, Nat.le_refl _, Nat.le_refl _, ?_⟩
  rw [List.drop_eq_nil_of_le (by simp), List.drop_eq_nil_of_le (by simp)]

/-- A thread's arguments: its identity, the shared base and the kernel's own,
    each the bits `xs` give, of its type. -/
def threadArgsOf (k : Kernel) (xs : List UInt64) : List V :=
  ((idParams ++ k.params).zip xs).map fun (t, x) => .sc t x

/-- A lookup among keys without repeats finds the entry of its key. -/
theorem lookup_nodup {β : Type} : ∀ (L : List (String × β)) (i : Nat) (key : String) (v : β),
    (L.map (·.1)).Nodup → L[i]? = some (key, v) → L.lookup key = some v
  | [], _, _, _, _, h => by simp at h
  | (k', v') :: L, 0, key, v, _, h => by simp at h; obtain ⟨rfl, rfl⟩ := h; simp [List.lookup]
  | (k', v') :: L, i + 1, key, v, hnd, h => by
      simp only [List.map_cons, List.nodup_cons] at hnd
      simp only [List.getElem?_cons_succ] at h
      have hne : key ≠ k' := by
        intro he; subst he
        exact hnd.1 (List.mem_map.mpr ⟨(key, v), List.mem_of_getElem? h, rfl⟩)
      simp only [List.lookup, show (key == k') = false by simpa using hne]
      exact lookup_nodup L i key v hnd.2 h

/-- A thread's argument's bits. -/
theorem bits_args (k : Kernel) (xs : List UInt64) (i : Nat) (hi : i < xs.length)
    (hlen : xs.length = 4 + k.params.length) :
    (asBits ((threadArgsOf k xs).getD i default)).getD 0 = xs.getD i 0 := by
  have hl : i < (idParams ++ k.params).length := by simp [idParams]; omega
  have hz : i < ((idParams ++ k.params).zip xs).length := by simp only [List.length_zip]; omega
  simp [threadArgsOf, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hz, asBits, List.getElem_zip,
    List.getElem?_eq_getElem hi]

/-- An entry's parameters, named by position: each name finds its value. -/
theorem lookup_names {β γ : Type} (ps : List β) (g : β → γ) (vals : Nat → UInt64)
    (hnd : (ps.zipIdx.map fun (_, i) => s!"p{i}").Nodup) (i : Nat) (hi : i < ps.length) :
    ((ps.zipIdx.map fun x => (g x.1, s!"p{x.2}")).zipIdx.map fun x => (x.1.2, vals x.2)).lookup s!"p{i}" =
      some (vals i) := by
  apply lookup_nodup _ i
  · have : ((ps.zipIdx.map fun x => (g x.1, s!"p{x.2}")).zipIdx.map fun x => (x.1.2, vals x.2)).map (·.1) =
        ps.zipIdx.map fun (_, i) => s!"p{i}" := by
      apply List.ext_getElem (by simp)
      intro j h1 h2; simp [List.getElem_zipIdx]
    rw [this]; exact hnd
  · simp [List.getElem?_zipIdx, hi]

/-- **A kernel thread's PTX ends where its body does.** A thread of kernel `k`
    whose arguments are bits cut to their types, the shared base at the shared
    array's start, its answers values a thread can take, on a GPU (`HwOk`),
    from memory with the GPU's NaN rule and shared memory within reach, run by
    the host semantics from `w` to world `w'`, a block of at most 32 threads if
    each is one warp: the entry `k` compiles to, run by
    PTX's on the same arguments and answers from a world the same as `w` apart
    from its observations, ends, in a world related to `w'`. -/
theorem thread_sound (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e)
    (ans : List Answer) (xs : List UInt64) (w w' pw : World) (hw : Hw) (prov : Bool)
    (hpw : { w with obs := [] } = { pw with obs := [] })
    (hlen : xs.length = 4 + k.params.length)
    (hnorm : ∀ i (hi : i < (idParams ++ k.params).length) (hx : i < xs.length),
      xs[i] &&& widthMask (idParams ++ k.params)[i] = xs[i])
    (h3 : xs[3]? = some (addrOf .arena 0)) (hans : AnsOk ans) (hhw : HwOk hw) (hnan : pw.mem.nan = hw.nan)
    (harena : pw.mem.arena.size < 2 ^ 32) (hwarp : k.warp = true → xs.getD 2 0 ≤ 32)
    (budget : Nat) (hrun : Kernel.thread k.code ans (threadArgsOf k xs) w hw budget prov = .ok w') :
    ∃ fuel w'', PtxRun.thread e ans (threadArgsOf k xs) pw hw fuel prov = .ok w'' ∧
      WRel w.obs.length pw.obs.length w' w'' := by
  obtain ⟨f, hf, hnd, hdef, htyp, hwin, htys, htk, hent, hpok, hconds, hdefs, herr, hlab, hpn, rfl, -⟩ :=
    entry_facts k e he
  have hf0 := hf
  let F := k.factsOf f
  let params := idParams ++ k.params
  let args := threadArgsOf k xs
  let lc := threadLocals ans w.obs.length hw prov
  have hfb : F.blocks = f.blocks := by
    show (k.factsOf f).blocks = f.blocks
    unfold Kernel.factsOf; dsimp only; split <;> rfl
  -- the body compiled, and its run as blocks
  unfold compileChecked at hf
  split at hf
  · cases hf
  split at hf
  · cases hf
  rename_i _ hsc
  have hfc : f = compileBody 0 k.code .nil params := by cases hf; rfl
  have hsc' : scopeOk params k.code = true := by simpa using hsc
  have hlen' : params.length = args.length := by
    simp [params, args, threadArgsOf, hlen, idParams]; omega
  unfold Kernel.thread at hrun
  split at hrun
  case h_2 | h_3 | h_4 | h_5 | h_6 => cases hrun
  rename_i Γf w2 hres
  cases hrun
  obtain ⟨steps, hbr⟩ := compile_core 0 .nil params k.code none { env := [], locals := lc } _ args w Γf w'
    (by rw [retOk_none]; exact hsc') hlen' hres
  rw [← hfc] at hbr
  -- the entry block
  obtain ⟨b0, rest, hb⟩ : ∃ b0 rest, F.blocks = b0 :: rest := by
    cases h : F.blocks with
    | nil => simp [entryOk, F, h] at hent
    | cons b0 rest => exact ⟨b0, rest, rfl⟩
  have hent2 := hent
  simp only [entryOk, show (k.factsOf f).blocks = b0 :: rest from hb, Bool.and_eq_true, beq_iff_eq] at hent2
  obtain ⟨⟨⟨hid0, hmap0⟩, -⟩, -⟩ := hent2
  have hlenb : b0.params.length = xs.length := by
    have := congrArg List.length hmap0; simp [idParams] at this; omega
  -- the context, its layout, and what its blocks have
  let c := PtxRun.Ctx.ofThread (k.assemble F) ans args pw hw prov
  have hc : c.body = ((Ptx.body F k.params).flatMap (·.lines.toList)).toArray := rfl
  have hls : c.labels = PtxRun.labelsOf ((Ptx.body F k.params).flatMap (·.lines.toList)) := rfl
  obtain ⟨pos, start, hL, hpos0, hmem⟩ :=
    body_layout F k.params b0 rest hb (by rw [hfb]; exact hnd) c hc hlab hls
  have hdef' := hdef
  simp only [definedOk, Bool.and_eq_true, List.all_eq_true] at hdef'
  obtain ⟨h0, hblk⟩ := hdef'
  have hG : Good F lc c w.obs.length pos start (definedIn f) := {
    lay := hL
    ok := insts_ok F lc c w.obs.length hdefs hwin htys htk rfl hans hhw (fun hw2 => by
      have hFw : F.warp = k.warp := by show (k.factsOf f).warp = k.warp; unfold Kernel.factsOf; rfl
      have hn : (c.special.lookup "ntid.x").getD 0 = xs.getD 2 0 := by
        rw [← bits_args k xs 2 (by omega) hlen]; simp [c, args, PtxRun.Ctx.ofThread, List.lookup]
      rw [hn]; exact hwarp (hFw ▸ hw2))
    low := fun b hbm j hj => herr _ (hmem b hbm j hj)
    inn := fun b hbm => by
      have := hblk b (hfb ▸ hbm)
      split at this
      · rename_i inn hi; exact ⟨inn, hi, this⟩
      · cases this
    params := hpok, typed := htyp, conds := hconds }
  -- the prologue
  obtain ⟨regs, hpro, hS⟩ := prologue_sim F k.params b0 rest hb hent hpok c pw args xs hlenb.symm
    (by
      show threadArgsOf k xs = _
      simp only [threadArgsOf, ← hmap0, List.zip_map_left, List.map_map]
      rfl)
    (fun i hi hx => by
      have := hnorm i (by rw [← hmap0]; simpa using hi) hx
      have ht : (idParams ++ k.params)[i]'(by rw [← hmap0]; simpa using hi) = b0.params[i].2 := by
        simp only [← hmap0, List.getElem_map]
      rwa [ht] at this)
    h3
    (fun i s hs => by
      by_cases h1 : s = "ctaid.x"
      · subst h1; simp [List.lookup] at hs; subst hs
        rw [← bits_args k xs 0 (by omega) hlen]; simp [c, args, PtxRun.Ctx.ofThread, List.lookup]
      by_cases h2 : s = "tid.x"
      · subst h2; simp [List.lookup] at hs; subst hs
        rw [← bits_args k xs 1 (by omega) hlen]; simp [c, args, PtxRun.Ctx.ofThread, List.lookup]
      by_cases h3' : s = "ntid.x"
      · subst h3'; simp [List.lookup] at hs; subst hs
        rw [← bits_args k xs 2 (by omega) hlen]; simp [c, args, PtxRun.Ctx.ofThread, List.lookup]
      · have e1 : (s == "ctaid.x") = false := by simpa using h1
        have e2 : (s == "tid.x") = false := by simpa using h2
        have e3 : (s == "ntid.x") = false := by simpa using h3'
        simp [List.lookup, e1, e2, e3] at hs)
    (by simp [c, PtxRun.Ctx.ofThread, Kernel.assemble, List.lookup])
    (fun i hi => by
      rw [show i + 4 = 4 + i by omega, ← bits_args k xs (4 + i) (by omega) hlen]
      exact lookup_names k.params
        (fun t => (if t == .f32 then .f32 else if t == .f64 then .f64 else Ptx.uTy t.width : PtxIr.Ty))
        (fun j => (asBits ((threadArgsOf k xs).getD (4 + j) default)).getD 0) hpn i hi)
  -- the blocks, from block 0
  obtain ⟨s0, hs0, hs0e⟩ : ∃ s0, (definedIn f).get? 0 = some s0 ∧ s0.isEmpty = true := by
    revert h0; cases (definedIn f).get? 0 with
    | none => intro h; cases h
    | some s0 => intro h; exact ⟨s0, rfl, h⟩
  have hfind : f.blocks.find? (·.ref.id == 0) = some b0 := by
    rw [← hfb, hb]; simp [hid0]
  obtain ⟨N, ts', hW', hgo⟩ := run_sim F lc c w.obs.length pos start (definedIn f) hG f hfb (memOkInv c hans) steps ⟨#[], w⟩ 0 args
    none w' hbr b0 _ { regs, w := pw } s0 hfind hs0
    (fun n hn => by
      exfalso
      have := Std.HashSet.isEmpty_iff_forall_not_mem.mp hs0e n
      exact this (Std.HashSet.mem_iff_contains.mpr hn))
    (fun p hp => ⟨p, hp, rfl⟩) hS (wrel_start w pw hpw) ⟨hnan, harena⟩
  -- the prologue's lines, block 0's label, then the blocks
  let P := (prologueInstrs F k.params b0).length
  have hbody : (Ptx.body F k.params).flatMap (·.lines.toList) =
      (prologueInstrs F k.params b0).map (PtxIr.Line.ins none) ++
        (blocksBody F 0 F.blocks).flatMap (·.lines.toList) := by
    simp [Ptx.body, hb, prologue]
  have hat : ∀ j (hj : j < P), c.body[0 + j]? = some (.ins none (prologueInstrs F k.params b0)[j]) := by
    intro j hj
    rw [hc, hbody]
    simp only [List.getElem?_toArray, Nat.zero_add]
    rw [List.getElem?_append_left (by simpa using hj)]
    simp [hj]
  have hlab0 : c.body[P]? = some (.label (.blk b0.ref.id)) := by
    have := (hL.blk b0 (by rw [hb]; exact List.mem_cons_self)).1
    rw [hpos0] at this
    simpa [prologue, P] using this
  refine ⟨N + 1 + P, ts'.w, ?_, hW'⟩
  have hg1 := go_steps c (prologueInstrs F k.params b0) 0 (N + 1) { regs := {}, w := pw } { regs, w := pw } hat hpro
  have hg2 := go_label c N P (.blk b0.ref.id) { regs, w := pw } hlab0
  have hg3 := hgo 0
  have hpos : pos 0 = P := by rw [← hid0, hpos0]; simp [prologue, P]
  rw [hpos, Nat.zero_add] at hg3
  simp only [PtxRun.thread]
  show (do let (_, s) ← (PtxRun.go c (N + 1 + P) 0).run { regs := {}, w := pw }; pure s.w) = _
  rw [hg1, Nat.zero_add, hg2, hg3]
  rfl

/-- **A PTX thread leaves the world's NaN rule.** -/
theorem thread_nan (e : PtxIr.Entry) (ans : List Answer) (args : List V) (w : World) (hw : Hw) (F : Nat)
    (prov : Bool) (u : World) (h : PtxRun.thread e ans args w hw F prov = .ok u) : u.mem.nan = w.mem.nan := by
  unfold PtxRun.thread at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, s⟩ := r
    cases h
    exact go_keeps (nanInv _ w.mem.nan) F 0 { regs := {}, w } () s hr rfl

end AlgorithmLib.Kernel.PtxSound
