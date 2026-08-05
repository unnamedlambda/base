import Lz4Interleave
import Lz4Geo
import AlgorithmLib.LZ4Confine
import AlgorithmLib.LZ4OpBound
import AlgorithmLib.LZ4Checkpoint

set_option maxRecDepth 8192

/-!
  # Where the kernel touches global memory

  `Lz4Interleave.KernelConfined` quantifies over every step of every warp.  Most
  of those steps cannot touch global memory at all: of the twenty-two
  instructions in the set, one reads it and three write it.  So the obligation is
  really about the handful of program points where such an instruction sits, and
  this file locates them — by enumerating them from the shipped kernel, not by
  reading the generator's source.

  That matters because the claim it replaces was a claim about the source.  The
  ledger said "all five `ldgo` sites use `inBase`-derived address registers";
  `shipped32_load_sites` below says there are **twelve** `ldgo` sites, using five
  distinct address registers.  The sentence was counting registers and calling
  them sites.  Nothing checked it, so nothing caught it.

  What this buys: `reads_at_site` and `writes_at_site` turn "for every reachable
  state" into "at these twelve pcs, for these five registers", which is a
  located, finite obligation rather than an open-ended one.  `load_at_site`,
  `sbAddr_is_outBase_add_op`, `la_at_store` and `cpDo_at_store` then say what
  each of those registers *holds* at the instant its instruction runs, so all
  twenty-eight sites are placed.

  It still does not discharge `KernelConfined`: placing an address is not
  bounding it, and the offsets' ranges are a per-step invariant the body
  simulation does not currently expose.  `CursorAtSites` is what is left.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

/-- Positions in a program at which `f` fires, paired with what it extracted. -/
def siteAux {α : Type} (f : SInstr → Option α) : Nat → List SInstr → List (Nat × α)
  | _, [] => []
  | n, i :: is =>
      match f i with
      | some a => (n, a) :: siteAux f (n + 1) is
      | none => siteAux f (n + 1) is

/-- Every `stg`/`stgp`/`stg32p` in the program, with its address register. -/
def storeSites (p : Array SInstr) : List (Nat × String) :=
  siteAux (fun i => match i with
    | .stg addr _ => some addr
    | .stgp _ addr _ => some addr
    | .stg32p _ addr _ => some addr
    | _ => none) 0 p.toList

/-- Every `ldgo`, with its address register and byte offset. -/
def loadSites (p : Array SInstr) : List (Nat × String × Nat) :=
  siteAux (fun i => match i with
    | .ldgo _ addr off => some (addr, off)
    | .ldgop _ _ addr off => some (addr, off)
    | _ => none) 0 p.toList

theorem mem_siteAux {α : Type} (f : SInstr → Option α) :
    ∀ (l : List SInstr) (base n : Nat) (i : SInstr) (a : α),
      l[n]? = some i → f i = some a → (base + n, a) ∈ siteAux f base l := by
  intro l
  induction l with
  | nil => intro base n i a h; simp at h
  | cons x xs ih =>
      intro base n i a hn ha
      cases n with
      | zero =>
          have hx : x = i := by simpa using hn
          subst hx
          rw [siteAux, ha]
          simp
      | succ m =>
          have hm : xs[m]? = some i := by simpa using hn
          have := ih (base + 1) m i a hm ha
          have harith : base + 1 + m = base + (m + 1) := by omega
          rw [harith] at this
          rw [siteAux]
          cases hfx : f x with
          | none => simpa [hfx] using this
          | some b => simp only [hfx, List.mem_cons]; exact Or.inr this

/-- **Every global read happens at a `loadSites` entry**, and the address is that
    site's register plus that site's offset, in some lane. -/
theorem reads_at_site (p : Array SInstr) (st : SState) (j : Nat) (h : Reads p st j) :
    ∃ (r : String) (off : Nat), (st.pc, r, off) ∈ loadSites p ∧
      ∃ l : Lane, (st.regs r l).toNat + off = j := by
  rw [Reads] at h
  cases hp : p[st.pc]? with
  | none => rw [hp] at h; exact absurd h not_false
  | some i =>
      rw [hp] at h
      cases i with
      | ldgo d addr off =>
          obtain ⟨l, hl⟩ := h
          refine ⟨addr, off, ?_, l, hl⟩
          have : p.toList[st.pc]? = some (SInstr.ldgo d addr off) := by
            simpa using hp
          simpa using mem_siteAux _ p.toList 0 st.pc _ (addr, off) this rfl
      | ldgop q d addr off =>
          obtain ⟨l, -, hl⟩ := h
          refine ⟨addr, off, ?_, l, hl⟩
          have : p.toList[st.pc]? = some (SInstr.ldgop q d addr off) := by
            simpa using hp
          simpa using mem_siteAux _ p.toList 0 st.pc _ (addr, off) this rfl
      | _ => exact absurd h not_false

/-- **Every global write happens at a `storeSites` entry**, at that site's
    register or one of the three bytes above it. -/
theorem writes_at_site (p : Array SInstr) (st : SState) (j : Nat) (h : Writes p st j) :
    ∃ r : String, (st.pc, r) ∈ storeSites p ∧ ∃ l : Lane,
      (st.regs r l).toNat = j ∨ (st.regs r l + 1).toNat = j ∨
      (st.regs r l + 2).toNat = j ∨ (st.regs r l + 3).toNat = j := by
  rw [Writes] at h
  cases hp : p[st.pc]? with
  | none => rw [hp] at h; exact absurd h not_false
  | some i =>
      rw [hp] at h
      have site : ∀ (addr : String),
          (fun i => match i with
            | SInstr.stg addr _ => some addr
            | SInstr.stgp _ addr _ => some addr
            | SInstr.stg32p _ addr _ => some addr
            | _ => none) i = some addr → (st.pc, addr) ∈ storeSites p := by
        intro addr hf
        have : p.toList[st.pc]? = some i := by simpa using hp
        simpa using mem_siteAux _ p.toList 0 st.pc i addr this hf
      cases i with
      | stg addr s => exact ⟨addr, site addr rfl, h.imp (fun l hl => Or.inl hl)⟩
      | stgp q addr s => exact ⟨addr, site addr rfl, h.imp (fun l hl => Or.inl hl)⟩
      | stg32p q addr s => exact ⟨addr, site addr rfl, h⟩
      | _ => exact absurd h not_false


-- ── The shipped kernel's sites ────────────────────────────────────────────────
-- Read off the emitted program.  `native_decide` for the same reason `Lz4Host`
-- uses it: the kernel is an `Array SInstr` of 274 entries built by the DSL, and
-- kernel reduction over it is not viable.

theorem numBlk32 : (WP.mk 15).numBlk = 6400 := by decide

theorem shipped32_size : (WP.mk 15).kernel.size = 274 := by decide

/-- **Twelve `ldgo` sites, five distinct address registers.**  `rpA`/`rcA` read
    the four bytes of a match candidate, `aP`/`aC` a single byte each during the
    extend, and `cpSo` the copy source.  `load_at_site` places all twelve; four
    of the five registers turn out to be `inBase`-derived at the load and the
    fifth, `cpSo`, is not — see there. -/
theorem shipped32_load_sites :
    loadSites (WP.mk 15).kernel =
      [(47, "rpA", 0), (48, "rpA", 1), (49, "rpA", 2), (50, "rpA", 3),
       (66, "rcA", 0), (67, "rcA", 1), (68, "rcA", 2), (69, "rcA", 3),
       (110, "aP", 0), (111, "aC", 0), (164, "cpSo", 0), (250, "cpSo", 0)] := by
  decide

/-- **Sixteen store sites, four distinct address registers.**  `sbAddr` is the
    token/literal emitter, `cpDo` the match copy destination, and `la0`–`la3` the
    four bytes of the length field the tail writes. -/
theorem shipped32_store_sites :
    storeSites (WP.mk 15).kernel =
      [(130, "sbAddr"), (140, "sbAddr"), (147, "sbAddr"), (165, "cpDo"),
       (173, "sbAddr"), (178, "sbAddr"), (188, "sbAddr"), (195, "sbAddr"),
       (216, "sbAddr"), (226, "sbAddr"), (233, "sbAddr"), (251, "cpDo"),
       (259, "la0"), (263, "la1"), (267, "la2"), (271, "la3")] := by
  decide

/-- The 64 KiB kernel has the same shape — the geometry is a parameter of the
    DSL, not a different program. -/
theorem shipped64_load_sites :
    loadSites (WP.mk 16).kernel = loadSites (WP.mk 15).kernel := by decide

theorem shipped64_store_sites :
    storeSites (WP.mk 16).kernel = storeSites (WP.mk 15).kernel := by decide

-- ── From program points to registers ─────────────────────────────────────────

/-- The five registers that ever hold a global load address. -/
def loadRegs : List String := ["rpA", "rcA", "aP", "aC", "cpSo"]

/-- The six that ever hold a global store address. -/
def storeRegs : List String := ["sbAddr", "cpDo", "la0", "la1", "la2", "la3"]

theorem load_regs32 :
    ∀ s ∈ loadSites (WP.mk 15).kernel, s.2.1 ∈ loadRegs ∧ s.2.2 ≤ 3 := by
  rw [shipped32_load_sites]; decide

theorem store_regs32 :
    ∀ s ∈ storeSites (WP.mk 15).kernel, s.2 ∈ storeRegs := by
  rw [shipped32_store_sites]; decide

/-- Whether a lane's store actually happens: `stgp`/`stg32p` are predicated, the
    rest always write.  The cooperative copy masks the lanes past the end of the
    literal run, and their addresses run out of region — so confinement can only
    be about active lanes. -/
def ActiveAt (prog : Array SInstr) (st : SState) (l : Lane) : Prop :=
  match prog[st.pc]? with
  | some (SInstr.stgp p _ _) => st.regs p l = 1
  | some (SInstr.stg32p p _ _) => st.regs p l = 1
  | _ => True

/-- **The obligation, with the program eliminated.**  No instructions — only the
    nine address registers, at the program points where a memory instruction
    actually uses them, and the values they may hold there.

    This is what `KernelConfined` reduces to.

    ## Why it is conditioned on the site

    The first version of this said "at every step, for every address register,
    the value is in range" — with no reference to the pc.  That statement is
    **false**, and `unconditioned_form_is_false` below proves it is: at `k = 0`
    the machine has not run, every scratch register still holds the launch
    default `0`, and `outRegion 15 outPtr w 0` fails for any non-zero `outPtr`.
    So the discharge of `KernelConfined` from it was vacuous — an obligation
    nobody could ever meet, sitting under a theorem that therefore said nothing.

    Nothing caught that.  `#print axioms` cannot see it, the trust scan cannot
    see it (an open obligation is *allowed* to be open), and the shape reads
    exactly like a dataflow fact.  It surfaced only on trying to prove it, which
    is the same lesson as instantiating a theorem at concrete values.

    The conditioned form below is what `writes_at_site`/`reads_at_site` actually
    hand you, and it is what the confinement argument actually needs: a register
    only has to be in range at the instant it is used as an address. -/
structure RegConfined (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) : Prop where
  /-- At a load site, the address it reads is below the output allocation. -/
  loads : ∀ (w : Fin (WP.mk b).numBlk) (k : Nat) (r : String) (off : Nat),
    ((siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)).pc, r, off)
      ∈ loadSites (WP.mk b).kernel →
    ∀ l : Lane,
      (((siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat
        + off) < outPtr
  /-- At a store site, the byte it writes is inside the warp's own output stride.

      The byte is exact, not a four-byte window.  Demanding the three bytes ABOVE
      the address — to match `writes_at_site`'s four-way disjunction — would be
      **false**: every store in this kernel is a 1-byte `stg`, and the four
      length-field stores sit at `outBase + lenOff … +3`, so those phantom bytes
      run past `outRegion`'s end at `outBase + lenOff + 4`.  `writes_at_site32`
      keeps the byte exact and nothing downstream needs more — see
      `kernelConfined_of_regConfined32`. -/
  stores : ∀ (w : Fin (WP.mk b).numBlk) (k : Nat) (r : String),
    ((siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)).pc, r)
      ∈ storeSites (WP.mk b).kernel →
    ∀ l : Lane,
      ActiveAt (WP.mk b).kernel (siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)) l →
      Lz4Interleave.outRegion b outPtr w.val
        ((siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat

/-- **The unconditioned form is false**, so conditioning on the site is not a
    convenience — it is the difference between an obligation and a vacuity.

    Stated at the shipped geometry with the placement the host actually produces
    (`outPtr = inPtr + totIn`), and witnessed at the launch state itself: `k = 0`,
    where `sbAddr` still holds `0`. -/
theorem unconditioned_form_is_false (gm : Array UInt8) (smemB : List UInt8) :
    ¬ (∀ (w : Fin (WP.mk 15).numBlk) (k : Nat) (r : String), r ∈ storeRegs →
        ∀ l : Lane,
          Lz4Interleave.outRegion 15 (WP.mk 15).totIn w.val
            (((siter (WP.mk 15).kernel k (initSt w.val 0 (WP.mk 15).totIn gm smemB)).regs
              r l).toNat)) := by
  intro h
  have hw : (0 : Nat) < (WP.mk 15).numBlk := by decide
  have := h ⟨0, hw⟩ 0 "sbAddr" (by decide) 0
  -- at `k = 0` the state is the launch state and `sbAddr` is still the default 0
  have hz : ((siter (WP.mk 15).kernel 0 (initSt 0 0 (WP.mk 15).totIn gm smemB)).regs
      "sbAddr" (0 : Lane)).toNat = 0 := rfl
  rw [hz] at this
  obtain ⟨hge, _⟩ := this
  have : (WP.mk 15).totIn = 209715200 := by decide
  omega

-- ── The one store register that carries the output cursor ────────────────────

/-- The ten store sites that use `sbAddr`.  Read off `shipped32_store_sites`. -/
def sbAddrSites : List Nat := [130, 140, 147, 173, 178, 188, 195, 216, 226, 233]

/-- **At every one of them, `sbAddr` holds `outBase + op`.**

    Each is one instruction below a `bin add sbAddr outBase op`, and none of them
    is a label or `ret`, so `add_above_holds_at'` says the machine can only have
    arrived by executing that assignment.  `decide` checks both facts against
    the emitted array, per site.

    What this buys: the *store* half of `RegConfined` for `sbAddr` — sixteen of
    the twenty-eight memory sites — now reduces to a single inequality about a
    single register, `op ≤ lenOff`, with no mention of addresses, sites or
    program points. -/
theorem sbAddr_is_outBase_add_op (w inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (k q : Nat) (hq : q ∈ sbAddrSites)
    (hpc : (siter (WP.mk 15).kernel (k + 1)
      (initSt w inPtr outPtr gm smemB)).pc = q) (l : Lane) :
    (siter (WP.mk 15).kernel (k + 1) (initSt w inPtr outPtr gm smemB)).regs "sbAddr" l
      = (siter (WP.mk 15).kernel (k + 1) (initSt w inPtr outPtr gm smemB)).regs "outBase" l
        + (siter (WP.mk 15).kernel (k + 1) (initSt w inPtr outPtr gm smemB)).regs "op" l := by
  rw [siter_succ] at hpc ⊢
  have go : ∀ r : Nat, r ∈ sbAddrSites →
      ((WP.mk 15).kernel[r]?.map fallthroughOnlyB) = some true ∧
      (WP.mk 15).kernel[r - 1]? = some (.bin .add "sbAddr" "outBase" (.reg "op")) ∧
      0 < r := by decide
  obtain ⟨h1, h2, h3⟩ := go q hq
  exact add_above_holds_at' _ q "sbAddr" "outBase" "op" h1 h2 (by decide) (by decide) h3 _ hpc l

-- ── The tail's four length-field stores ──────────────────────────────────────

/-- **The four length-field stores are confined for structural reasons.**

    The tail is straight-line code:

        257:  bin add la0 outBase 35072      -- `lenOff`, an immediate
        258:  bin band lb op 255
        259:  stg la0 lb                     -- and likewise at 263, 267, 271
        260:  bin add la1 la0 1

    so `la_j` is `outBase + lenOff + j` outright.  `outRegion` reaches
    `outBase + lenOff + 4`, so these four need **no bound on the output cursor
    at all** — which is worth saying, because it means the cursor bound is owed
    only by the ten `sbAddr` stores and the two `cpDo` stores, not by all
    sixteen.

    Each line below is `add_imm_carried` at a program point, with its side
    conditions decided against the emitted array. -/
theorem la_at_store (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) (k : Nat) :
    let S := siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)
    ∀ l : Lane,
      (S.pc = 259 → S.regs "la0" l = S.regs "outBase" l + UInt64.ofNat (WP.mk 15).lenOff) ∧
      (S.pc = 263 → S.regs "la1" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 15).lenOff + 1)) ∧
      (S.pc = 267 → S.regs "la2" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 15).lenOff + 2)) ∧
      (S.pc = 271 → S.regs "la3" l = S.regs "outBase" l + UInt64.ofNat ((WP.mk 15).lenOff + 3)) := by
  intro S l
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have base : ∀ (q n : Nat), q - n - 1 = 257 → n + 1 ≤ q →
      (∀ t, t < n + 1 → ((WP.mk 15).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "la0")) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "outBase")) = some true) →
      S.pc = q → S.regs "la0" l = S.regs "outBase" l + UInt64.ofNat 35072 := by
    intro q n h257 hn hft hd ha hpc
    exact add_imm_carried _ _ hinit "la0" "outBase" 35072 q n hft hd ha
      (by rw [h257]; decide) (by decide) hn k hpc l
  have step : ∀ (d : String) (j q n : Nat), n + 1 ≤ q →
      (WP.mk 15).kernel[q - n - 1]? = some (.bin .add d "la0" (.imm j)) →
      (∀ t, t < n + 1 → ((WP.mk 15).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "la0")) = some true) →
      d ≠ "la0" → S.pc = q → S.regs d l = S.regs "la0" l + UInt64.ofNat j := by
    intro d j q n hn hpre hft hd ha hne hpc
    exact add_imm_carried _ _ hinit d "la0" j q n hft hd ha hpre hne hn k hpc l
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · exact base 259 1 rfl (by omega) (by decide) (by decide) (by decide) h
  · rw [step "la1" 1 263 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 263 5 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 15).lenOff + 1 = 35072 + 1 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]
  · rw [step "la2" 2 267 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 267 9 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 15).lenOff + 2 = 35072 + 2 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]
  · rw [step "la3" 3 271 2 (by omega) (by decide) (by decide) (by decide) (by decide)
        (by decide) h,
      base 271 13 rfl (by omega) (by decide) (by decide) (by decide) h]
    rw [show (WP.mk 15).lenOff + 3 = 35072 + 3 from rfl, UInt64.ofNat_add,
      UInt64.add_assoc]

-- ── The two literal-copy stores ──────────────────────────────────────────────

/-- **At both `cpDo` store sites, the address is `cpDst + cpI + lane`.**

    `cpDo` is the one address register that is not set in a single instruction.
    It accumulates — `cpDst + cpI`, then `+ lane` — and the accumulating step
    assigns to its own source, which is why `add_above_holds_at'` cannot phrase
    it and `binr_pair_carried` exists.

    The base is left as `cpDst` (resp. `cpDstF`) rather than expanded to
    `outBase + op`.  That expansion is *not* available by the same argument:
    `cpDst` is set at pc 152, outside the copy loop, and pc 156 is the loop
    header label — so the run from 152 to 165 is not fallthrough-only and the
    value is loop-carried.  Saying `cpDst + cpI + lane` is what the trace
    actually supports; claiming `outBase + op + cpI + lane` here would be
    asserting a loop invariant that nothing has proved.

    With this, all sixteen store sites are placed: ten at `outBase + op`, four
    at `outBase + lenOff + j`, and these two at a cursor plus a copy offset. -/
theorem cpDo_at_store (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (l : Lane) :
    let S := siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)
    (S.pc = 165 → S.regs "cpDo" l = S.regs "cpDst" l + S.regs "cpI" l + S.regs "lane" l) ∧
    (S.pc = 251 → S.regs "cpDo" l = S.regs "cpDstF" l + S.regs "cpI" l + S.regs "lane" l) := by
  intro S
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have go : ∀ (dst : String) (q : Nat), q - 5 - 2 = q - 7 →
      (WP.mk 15).kernel[q - 5 - 2]? = some (.binr .add "cpDo" dst "cpI") →
      (WP.mk 15).kernel[q - 5 - 1]? = some (.binr .add "cpDo" "cpDo" "lane") →
      (∀ t, t < 7 → ((WP.mk 15).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < 5 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "cpDo")) = some true) →
      (∀ t, t < 7 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some dst)) = some true) →
      (∀ t, t < 7 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "cpI")) = some true) →
      (∀ t, t < 7 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some "lane")) = some true) →
      7 ≤ q → S.pc = q →
      S.regs "cpDo" l = S.regs dst l + S.regs "cpI" l + S.regs "lane" l := by
    intro dst q _ h1 h2 hft hnwd hnwa hnwb hnwc hq hpc
    exact binr_pair_carried _ _ hinit .add "cpDo" dst "cpI" "lane" q 5
      hft hnwd hnwa hnwb hnwc h1 h2 (by omega) k hpc l
  exact ⟨go "cpDst" 165 rfl (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by omega),
    go "cpDstF" 251 rfl (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by omega)⟩

-- ── The twelve load sites ────────────────────────────────────────────────────

/-- **At every one of the twelve `ldgo` sites, the address is placed.**

    Four shapes, and the file header's original claim — "all five `ldgo` sites
    use `inBase`-derived address registers" — turns out to be right about ten of
    the twelve and wrong about the other two:

    * `rpA` (47–50) is `inBase + rp`, set once at 46 and read four times;
    * `rcA` (66–69) is `inBase + rc`, likewise from 65;
    * `aP` (110) is `inBase + peD` and `aC` (111) is `inBase + caD`;
    * `cpSo` (164, 250) is **not** `inBase`-derived at this point — it is
      `cpSrc + cpI + lane`, the same accumulating shape as `cpDo`.  `cpSrc` is
      `inBase + litAnchor`, but that is set outside the copy loop, so the same
      loop-carried limit applies as for `cpDst`.

    Together with `sbAddr_is_outBase_add_op`, `la_at_store` and `cpDo_at_store`,
    all twenty-eight memory sites are now located: none of `RegConfined` is an
    open question about *which* register or *where*.  What remains is the range
    of the offsets, which is `CursorAtSites` below. -/
theorem load_at_site (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (l : Lane) :
    let S := siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)
    (∀ q ∈ [47, 48, 49, 50], S.pc = q → S.regs "rpA" l = S.regs "inBase" l + S.regs "rp" l) ∧
    (∀ q ∈ [66, 67, 68, 69], S.pc = q → S.regs "rcA" l = S.regs "inBase" l + S.regs "rc" l) ∧
    (S.pc = 110 → S.regs "aP" l = S.regs "inBase" l + S.regs "peD" l) ∧
    (S.pc = 111 → S.regs "aC" l = S.regs "inBase" l + S.regs "caD" l) ∧
    (S.pc = 164 → S.regs "cpSo" l = S.regs "cpSrc" l + S.regs "cpI" l + S.regs "lane" l) ∧
    (S.pc = 250 → S.regs "cpSo" l = S.regs "cpSrcF" l + S.regs "cpI" l + S.regs "lane" l) := by
  intro S
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  -- one source instruction, read `n` steps later
  have one : ∀ (d a b : String) (q n : Nat),
      (WP.mk 15).kernel[q - n - 1]? = some (.binr .add d a b) →
      (∀ t, t < n + 1 → ((WP.mk 15).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some a)) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some b)) = some true) →
      d ≠ a → d ≠ b → n + 1 ≤ q → S.pc = q →
      S.regs d l = S.regs a l + S.regs b l := by
    intro d a b q n hpre hft hd ha hb hda hdb hn hpc
    exact binr_carried _ _ hinit .add d a b q n hft hd ha hb hpre hda hdb hn k hpc l
  -- two accumulating instructions, read `n` steps after the second
  have two : ∀ (d a b c : String) (q n : Nat),
      (WP.mk 15).kernel[q - n - 2]? = some (.binr .add d a b) →
      (WP.mk 15).kernel[q - n - 1]? = some (.binr .add d d c) →
      (∀ t, t < n + 2 → ((WP.mk 15).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some a)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some b)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 15).kernel[q - t - 1]?.map
        (fun i => destOf i != some c)) = some true) →
      n + 2 ≤ q → S.pc = q →
      S.regs d l = S.regs a l + S.regs b l + S.regs c l := by
    intro d a b c q n h1 h2 hft hd ha hb hc hn hpc
    exact binr_pair_carried _ _ hinit .add d a b c q n hft hd ha hb hc h1 h2 hn k hpc l
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl | rfl | rfl
    · exact one "rpA" "inBase" "rp" 47 0 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 48 1 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 49 2 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 50 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl | rfl | rfl
    · exact one "rcA" "inBase" "rc" 66 0 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 67 1 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 68 2 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 69 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
  · exact one "aP" "inBase" "peD" 110 2 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by omega)
  · exact one "aC" "inBase" "caD" 111 1 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by omega)
  · exact two "cpSo" "cpSrc" "cpI" "lane" 164 2 (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by omega)
  · exact two "cpSo" "cpSrcF" "cpI" "lane" 250 2 (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by omega)


theorem load_at_site64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (l : Lane) :
    let S := siter (WP.mk 16).kernel k (initSt w inPtr outPtr gm smemB)
    (∀ q ∈ [47, 48, 49, 50], S.pc = q → S.regs "rpA" l = S.regs "inBase" l + S.regs "rp" l) ∧
    (∀ q ∈ [66, 67, 68, 69], S.pc = q → S.regs "rcA" l = S.regs "inBase" l + S.regs "rc" l) ∧
    (S.pc = 110 → S.regs "aP" l = S.regs "inBase" l + S.regs "peD" l) ∧
    (S.pc = 111 → S.regs "aC" l = S.regs "inBase" l + S.regs "caD" l) ∧
    (S.pc = 164 → S.regs "cpSo" l = S.regs "cpSrc" l + S.regs "cpI" l + S.regs "lane" l) ∧
    (S.pc = 250 → S.regs "cpSo" l = S.regs "cpSrcF" l + S.regs "cpI" l + S.regs "lane" l) := by
  intro S
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  -- one source instruction, read `n` steps later
  have one : ∀ (d a b : String) (q n : Nat),
      (WP.mk 16).kernel[q - n - 1]? = some (.binr .add d a b) →
      (∀ t, t < n + 1 → ((WP.mk 16).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some a)) = some true) →
      (∀ t, t < n → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some b)) = some true) →
      d ≠ a → d ≠ b → n + 1 ≤ q → S.pc = q →
      S.regs d l = S.regs a l + S.regs b l := by
    intro d a b q n hpre hft hd ha hb hda hdb hn hpc
    exact binr_carried _ _ hinit .add d a b q n hft hd ha hb hpre hda hdb hn k hpc l
  -- two accumulating instructions, read `n` steps after the second
  have two : ∀ (d a b c : String) (q n : Nat),
      (WP.mk 16).kernel[q - n - 2]? = some (.binr .add d a b) →
      (WP.mk 16).kernel[q - n - 1]? = some (.binr .add d d c) →
      (∀ t, t < n + 2 → ((WP.mk 16).kernel[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some d)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some a)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some b)) = some true) →
      (∀ t, t < n + 2 → ((WP.mk 16).kernel[q - t - 1]?.map
        (fun i => destOf i != some c)) = some true) →
      n + 2 ≤ q → S.pc = q →
      S.regs d l = S.regs a l + S.regs b l + S.regs c l := by
    intro d a b c q n h1 h2 hft hd ha hb hc hn hpc
    exact binr_pair_carried _ _ hinit .add d a b c q n hft hd ha hb hc h1 h2 hn k hpc l
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl | rfl | rfl
    · exact one "rpA" "inBase" "rp" 47 0 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 48 1 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 49 2 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rpA" "inBase" "rp" 50 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl | rfl | rfl
    · exact one "rcA" "inBase" "rc" 66 0 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 67 1 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 68 2 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
    · exact one "rcA" "inBase" "rc" 69 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by omega)
  · exact one "aP" "inBase" "peD" 110 2 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by omega)
  · exact one "aC" "inBase" "caD" 111 1 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by omega)
  · exact two "cpSo" "cpSrc" "cpI" "lane" 164 2 (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by omega)
  · exact two "cpSo" "cpSrcF" "cpI" "lane" 250 2 (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by omega)


/-- **The load-address and store-site facts, as a class.**  Both are immediate-free
    — they name registers and opcodes only — so they hold verbatim at every block
    geometry; each instance is machine-checked against its own kernel. -/
class Loads (p : Array AlgorithmLib.LZ4Simt.SInstr) : Prop where
  loadAt : ∀ (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
      (k : Nat) (l : AlgorithmLib.LZ4Simt.Lane),
    let St := AlgorithmLib.LZ4Simt.siter p k (initSt w inPtr outPtr gm smemB)
    (∀ q ∈ [47, 48, 49, 50], St.pc = q → St.regs "rpA" l = St.regs "inBase" l + St.regs "rp" l) ∧
    (∀ q ∈ [66, 67, 68, 69], St.pc = q → St.regs "rcA" l = St.regs "inBase" l + St.regs "rc" l) ∧
    (St.pc = 110 → St.regs "aP" l = St.regs "inBase" l + St.regs "peD" l) ∧
    (St.pc = 111 → St.regs "aC" l = St.regs "inBase" l + St.regs "caD" l) ∧
    (St.pc = 164 → St.regs "cpSo" l = St.regs "cpSrc" l + St.regs "cpI" l + St.regs "lane" l) ∧
    (St.pc = 250 → St.regs "cpSo" l = St.regs "cpSrcF" l + St.regs "cpI" l + St.regs "lane" l)
  storeCpDo : ∀ x ∈ storeSites p,
    x.2 = "cpDo" → x.1 = 165 ∨ x.1 = 251
  /-- The twelve load sites.  Offsets are byte indices, not geometry immediates,
      so this list is the same at every stride. -/
  loadSitesEq : loadSites p =
    [(47, "rpA", 0), (48, "rpA", 1), (49, "rpA", 2), (50, "rpA", 3),
     (66, "rcA", 0), (67, "rcA", 1), (68, "rcA", 2), (69, "rcA", 3),
     (110, "aP", 0), (111, "aC", 0), (164, "cpSo", 0), (250, "cpSo", 0)]

/-- The shipped 32 KiB kernel's load and store sites. -/
instance loads32 : Loads (WP.mk 15).kernel where
  loadAt := load_at_site
  storeCpDo := by rw [shipped32_store_sites]; decide
  loadSitesEq := shipped32_load_sites

/-- The same, machine-checked against the 64 KiB kernel. -/
instance loads64 : Loads (WP.mk 16).kernel where
  loadAt := load_at_site64
  storeCpDo := by decide
  loadSitesEq := by decide

-- ── What the ten `sbAddr` stores still need, named ───────────────────────────

/-- **The whole residue of `RegConfined`'s store half, in one place.**

    `sbAddr_is_outBase_add_op` proved the address is `outBase + op` at each of the
    ten sites, `la_at_store` closed the four tail stores outright, and
    `cpDo_at_store` placed the remaining two on `cpDst + cpI + lane` — so all
    sixteen store sites are located and none of them is open.  What is left is
    not a statement about addresses, sites or program points at all — it is two
    facts about two registers at the instants the stores happen:

    * `opLe` — the output cursor has not passed `lenOff`.  `LZ4OpBound` reduces
      this to exposing the per-step machine/eval correspondence that already
      lives inside `warpKernelDSL_sstep_roundtrips`' proof; the cursor-advance
      chain above it (`op_le_of_add` and its instances) is done.

      `shipped32_op_writes` narrows what could go wrong: in the whole kernel
      there are thirteen writes to `op`, one `mov op, 0` in the prologue and
      twelve `add op, op, X`.  So the cursor cannot be *assigned* a bad value —
      the only failure mode left is being pushed past the budget, and the
      invariant that rules that out is `op + |remaining plan| ≤ lenOff`, which is
      forward-preserved (plain `op ≤ lenOff` is not) and therefore has to be
      threaded through the body simulation rather than recovered from its
      endpoints.  That threading is what `LZ4StepDescent` starts.

    "The output base is still the one the prologue computed" is not a field here:
    it is proven, as `outBase_at_store_site`. -/
structure CursorAtSites (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) : Prop where
  opLe : ∀ (w : Fin (WP.mk b).numBlk) (k : Nat) (l : Lane),
    (siter (WP.mk b).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc ∈ sbAddrSites →
    ((siter (WP.mk b).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).regs "op" l).toNat
      ≤ (WP.mk b).lenOff

-- ── The prologue never stands at a store site ───────────────────────────────

/-- Every branch below pc 25 targets the `OOB` label at 272 — decided. -/
theorem shape_lo : ∀ t, t < 25 → stepShapeB (WP.mk 15).kernel 272 t = true := by decide

/-- …and from 272 on there are no branches at all (272 is a label, 273 a `ret`,
    and the program ends). -/
theorem shape_hi : ∀ q, 272 ≤ q → stepShapeB (WP.mk 15).kernel 272 q = true := by
  intro q hq
  rcases Nat.lt_or_ge q 274 with h | h
  · have : q = 272 ∨ q = 273 := by omega
    rcases this with rfl | rfl <;> decide
  · rw [stepShapeB, Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]

/-- **For the first 25 steps the machine is below pc 25 or past 272** — so it is
    never at a store site, all of which lie in `[130, 272)`.

    This is the prologue bridge, and it needs no simulation of the prologue: the
    pc can only advance by one or jump to the single escape the guard uses, which
    `pc_next` says and `shape_lo`/`shape_hi` decide.  An earlier attempt tried to
    prove `pc k = k` by strengthening four slice lemmas in `LZ4Prologue`; this
    replaces all of that. -/
theorem prologue_pc_shape (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k, k ≤ 25 →
      (siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).pc ≤ k ∨
      272 ≤ (siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).pc := by
  intro k
  induction k with
  | zero => intro _; exact Or.inl (Nat.le_refl _)
  | succ m ih =>
      intro hm
      have hstep := ih (by omega)
      rw [siter_succ]
      rcases hstep with hlo | hhi
      · have hs : stepShapeB (WP.mk 15).kernel 272
            (siter (WP.mk 15).kernel m (initSt w inPtr outPtr gm smemB)).pc = true :=
          shape_lo _ (by omega)
        rcases pc_next (WP.mk 15).kernel 272 _ _ rfl hs with e | e | e <;> omega
      · have hs : stepShapeB (WP.mk 15).kernel 272
            (siter (WP.mk 15).kernel m (initSt w inPtr outPtr gm smemB)).pc = true :=
          shape_hi _ hhi
        rcases pc_next (WP.mk 15).kernel 272 _ _ rfl hs with e | e | e <;> omega

/-- **Hence the prologue is never at a store site.**  `CursorAtSites`' obligations
    are vacuous below step 25, which is what lets `RegConfined` be stated from the
    post-prologue state without weakening it. -/
theorem prologue_not_at_store_site (w inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (k : Nat) (hk : k ≤ 25) (r : String) :
    ¬ ((siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).pc, r)
        ∈ storeSites (WP.mk 15).kernel := by
  intro hmem
  have hrange : ∀ s ∈ storeSites (WP.mk 15).kernel, 130 ≤ s.1 ∧ s.1 < 272 := by
    rw [shipped32_store_sites]; decide
  obtain ⟨h1, h2⟩ := hrange _ hmem
  rcases prologue_pc_shape w inPtr outPtr gm smemB k hk with e | e <;> omega

-- ── `outBaseAt`, discharged ──────────────────────────────────────────────────

/-- `siter` and `snsteps` are the same iteration under two names: the
    concurrency development and the straight-line simulation each defined one.
    Nothing depended on their agreement until a prologue result had to be used
    inside a trace argument. -/
theorem siter_eq_snsteps (p : Array SInstr) :
    ∀ (n : Nat) (st : SState), siter p n st = snsteps p n st := by
  intro n
  induction n with
  | zero => intro st; rfl
  | succ m ih => intro st; simp only [siter, snsteps]; exact ih _

theorem noDest20 : noDestFrom (WP.mk 15).kernel "outBase" 20 = true := by
  rw [noDestFrom_eq _ _ _ (by rw [shipped32_size]; omega)]; decide
theorem noExit20 : noExitBelow (WP.mk 15).kernel 20 = true := by
  rw [noExitBelow_eq _ _ (by rw [shipped32_size]; omega)]; decide
theorem nbpos : 0 < (WP.mk 15).numBlk := by decide
theorem nb64 : (WP.mk 15).numBlk < 2 ^ 64 := by decide

set_option maxHeartbeats 2000000 in
/-- **The second field of `CursorAtSites` is not an obligation.**

    At every `sbAddr` store site the output base is exactly the one the launch
    geometry dictates.  Three pieces already existed and only had to meet:

    * `prologue_pc_shape` — for `k ≤ 25` the pc is at most `k` or already at the
      out-of-bounds exit, so a site in `[130, 272)` forces `26 ≤ k`;
    * `head25` — after exactly 25 steps the machine stands at pc 25 with
      `outBase = outPtr + w * outStride`, the value the prologue computes;
    * `regs_const_from` at `lo = 20` — `outBase` is written at one program point
      (pc 19) and nothing below 20 is ever branched to, so from pc 25 onward the
      register cannot change.

    `siter_add` splits the trace at 25 and the three compose.  Note the bound is
    `20`, not `39`: the constancy argument does not care where the prologue
    *ends*, only where `outBase` is last written — which is why this does not
    need the clear loop's step-exact trajectory, and `head38`/`prologue_couple`
    (indexed by `25 + 8 * clearIters hL + 8`) never enter it. -/
theorem outBase_at_store_site (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 15).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (k : Nat) (l : Fin 32)
    (h130 : 130 ≤ (siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).pc)
    (h272 : (siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).pc < 272) :
    (siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)).regs "outBase" l
      = UInt64.ofNat (outPtr + w * (WP.mk 15).outStride) := by
  have hk : 26 ≤ k := by
    rcases Nat.lt_or_ge k 26 with hlt | hge
    · rcases prologue_pc_shape w inPtr outPtr gm smemB k (by omega) with e | e <;> omega
    · exact hge
  have hker : (WP.mk 15).kernel = AlgorithmLib.LZ4WarpDSL.warpKernelDSL
      (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog :=
    rfl
  obtain ⟨hpc25, -, -, -, hob, -⟩ :=
    AlgorithmLib.LZ4WarpDSL.head25 (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride
      (WP.mk 15).lenOff wHashLog w inPtr outPtr gm smemB nbpos nb64 hw hw64 hderive
  rw [← hker, ← siter_eq_snsteps] at hpc25 hob
  have hsplit : siter (WP.mk 15).kernel k (initSt w inPtr outPtr gm smemB)
      = siter (WP.mk 15).kernel (k - 25)
          (siter (WP.mk 15).kernel 25 (initSt w inPtr outPtr gm smemB)) := by
    rw [← siter_add, show 25 + (k - 25) = k from by omega]
  have hconst := regs_const_from (WP.mk 15).kernel "outBase" 20 noDest20 noExit20
    (siter (WP.mk 15).kernel 25 (initSt w inPtr outPtr gm smemB)) (by rw [hpc25]; omega) (k - 25)
  rw [hsplit, hconst.1]
  exact (show rOutBase = "outBase" from rfl) ▸ hob l

/-- **The ten `sbAddr` stores are confined, given exactly that.**  No further
    reasoning about the program: the address is `outBase + op` by
    `sbAddr_is_outBase_add_op`, and the two register facts place it. -/
theorem sbAddr_confined_of_cursor (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (h : CursorAtSites 15 inPtr outPtr gm smemB)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (l : Lane)
    (hpc : (siter (WP.mk 15).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc
      ∈ sbAddrSites)
    (htop : outPtr + w.val * (WP.mk 15).outStride + (WP.mk 15).lenOff + 4 < 2 ^ 64) :
    Lz4Interleave.outRegion 15 outPtr w.val
      (((siter (WP.mk 15).kernel (k + 1)
        (initSt w.val inPtr outPtr gm smemB)).regs "sbAddr" l).toNat) := by
  obtain ⟨q, hq, hpcq⟩ : ∃ q, q ∈ sbAddrSites ∧
      (siter (WP.mk 15).kernel (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = q :=
    ⟨_, hpc, rfl⟩
  have hadd := sbAddr_is_outBase_add_op w.val inPtr outPtr gm smemB k q hq hpcq l
  have hob : ((siter (WP.mk 15).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l).toNat
      = outPtr + w.val * (WP.mk 15).outStride := by
    have hr : ∀ q ∈ sbAddrSites, 130 ≤ q ∧ q < 272 := by decide
    rw [outBase_at_store_site w.val inPtr outPtr gm smemB w.isLt
      (by have h := w.isLt; have hn := numBlk32; omega) hderive (k + 1) l
      (hr _ hpc).1 (hr _ hpc).2]
    exact UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega)
  have hop := h.opLe w k l hpc
  have hnw : ((siter (WP.mk 15).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "outBase" l).toNat
    + ((siter (WP.mk 15).kernel (k + 1)
      (initSt w.val inPtr outPtr gm smemB)).regs "op" l).toNat < 2 ^ 64 := by
    rw [hob]; omega
  rw [hadd, toNat_add_of_lt _ _ hnw, hob]
  exact ⟨by omega, by omega⟩

-- ── `outBase` survives the whole body ────────────────────────────────────────

/-- **The output base is written once, in the prologue, and never again.**

    `outBase` is set at pc 19 and no instruction from the main-loop head (39)
    onward touches it; nor does any branch at or above 39 target a point below
    it, so the region is closed and the frame is sound.  Both facts are `decide`
    scans of the emitted array.

    Superseded for the store sites by `outBase_at_store_site`, which splits the
    trace at 25 instead of 39 and so needs no assumption about where the trace
    starts.  Kept because the load half will want the same frame at `inBase`, and
    because it is the general statement: this one holds from *any* post-prologue
    state, not only from the launch state. -/
theorem outBase_const_after_prologue (st0 : AlgorithmLib.LZ4Simt.SState)
    (h0 : 39 ≤ st0.pc) (k : Nat) :
    (siter (WP.mk 15).kernel k st0).regs "outBase" = st0.regs "outBase" ∧
    39 ≤ (siter (WP.mk 15).kernel k st0).pc :=
  regs_const_from (WP.mk 15).kernel "outBase" 39
    (by rw [noDestFrom_eq _ _ _ (by rw [shipped32_size]; omega)]; decide)
    (by rw [noExitBelow_eq _ _ (by rw [shipped32_size]; omega)]; decide) st0 h0 k

/-- The same for `inBase`, which the load half will need. -/
theorem inBase_const_after_prologue (st0 : AlgorithmLib.LZ4Simt.SState)
    (h0 : 39 ≤ st0.pc) (k : Nat) :
    (siter (WP.mk 15).kernel k st0).regs "inBase" = st0.regs "inBase" ∧
    39 ≤ (siter (WP.mk 15).kernel k st0).pc :=
  regs_const_from (WP.mk 15).kernel "inBase" 39
    (by rw [noDestFrom_eq _ _ _ (by rw [shipped32_size]; omega)]; decide)
    (by rw [noExitBelow_eq _ _ (by rw [shipped32_size]; omega)]; decide) st0 h0 k

-- ── What can move the output cursor at all ────────────────────────────────────

/-- Every instruction that writes the output cursor `op`, with the instruction
    itself.  Same `siteAux` enumeration as `storeSites`/`loadSites`: read off the
    shipped program, not off the generator. -/
def opWriteSites (p : Array SInstr) : List (Nat × SInstr) :=
  siteAux (fun i => if destOf i == some "op" then some i else none) 0 p.toList

/-- **Thirteen writes to `op` in the whole 274-instruction kernel**, and twelve of
    them are accumulations.

    This is the part of `CursorAtSites` that does not need the emit loop's plan:
    the cursor is initialised once, in the prologue, and thereafter is only ever
    `op := op + X`.  There is no reset, no reload from memory, and no assignment
    of an unrelated value — so on the machine `op` moves in one direction, and the
    only way `op ≤ lenOff` can fail at a store site is by the cursor being *pushed
    past* the budget, never by it jumping somewhere arbitrary.

    Ten of the twelve accumulations sit immediately after the ten `sbAddr` stores
    (`sbAddrSites`), one (170) skips the literal run the copy loop just wrote, and
    one (256) skips the final match copy. -/
theorem shipped32_op_writes :
    opWriteSites (WP.mk 15).kernel =
      [ (37,  .mov "op" (.imm 0)),
        (131, .bin .add "op" "op" (.imm 1)),
        (141, .bin .add "op" "op" (.imm 1)),
        (148, .bin .add "op" "op" (.imm 1)),
        (170, .bin .add "op" "op" (.reg "litLen")),
        (174, .bin .add "op" "op" (.imm 1)),
        (179, .bin .add "op" "op" (.imm 1)),
        (189, .bin .add "op" "op" (.imm 1)),
        (196, .bin .add "op" "op" (.imm 1)),
        (217, .bin .add "op" "op" (.imm 1)),
        (227, .bin .add "op" "op" (.imm 1)),
        (234, .bin .add "op" "op" (.imm 1)),
        (256, .bin .add "op" "op" (.reg "fLen")) ] := by
  decide

/-- **The cursor accumulates: every write past the prologue is `op := op + X`.**
    Stated as a check over the enumeration so a later change to the emitter that
    introduced a reset would fail this build rather than silently invalidate the
    monotonicity half of `CursorAtSites`. -/
theorem shipped32_op_accumulates :
    ((opWriteSites (WP.mk 15).kernel).filter (fun s => decide (37 < s.1))).all
      (fun s => match s.2 with
                | .bin .add "op" "op" _ => true
                | _ => false) = true := by
  rw [shipped32_op_writes]; decide

/-- The 64 KiB kernel writes `op` at exactly the same points — the geometry is a
    parameter of the DSL, not a different program. -/
theorem shipped64_op_writes :
    opWriteSites (WP.mk 16).kernel = opWriteSites (WP.mk 15).kernel := by decide

-- ── The tail: the final literal run, bounded by its own potential ────────────

/-- The straight-line run from the `loopC` exit (pc 208) to the final token
    store (pc 216).  Closed under the emitted program's control flow, so
    `regs_const_on` decides that nothing in it writes `op`. -/
def tailPre : List Nat := [208, 209, 210, 211, 212, 213, 214, 215, 216]

theorem tailPre_iv : tailPre = (List.range 9).map (· + 208) := rfl

theorem tailPre_closed : PcClosed (WP.mk 15).kernel tailPre [216] :=
  tailPre_iv ▸ ivClosed_at (WP.mk 15).kernel 208 9 [216] shipped32_size (by omega) (by decide)

/-- **The final token store happens at the cursor the loop left.**  Nothing
    between the loop exit and pc 216 writes `op`, so the first `sbAddr` store of
    the tail is at exactly `outBase + op_exit` — and `op_exit ≤ lenOff` is
    immediate from the tail's own byte count (`op_exit + 1 + |encNib fLen| +
    fLen = encode.length ≤ lenOff`), with no potential argument needed.

    This is the first of the ten `sbAddr` sites to close from the machine side. -/
theorem op_const_to_216 (st : SState) (h0 : st.pc = 208) (k : Nat)
    (hne : ∀ j, j < k → (siter (WP.mk 15).kernel j st).pc ∉ [216]) :
    (siter (WP.mk 15).kernel k st).regs "op" = st.regs "op" :=
  regs_const_on (WP.mk 15).kernel "op" tailPre [216] tailPre_closed (by decide) st
    (by rw [h0]; decide) k hne

-- ── The tail's LSIC loop: the potential that bounds sites 226 and 233 ────────

/-- The tail's length-extension loop with its exit run, pcs 222–234:
    `222 lbl; 223 braifnot lsicC→231; 224 c255:=255; 225 sbAddr:=outBase+op;
     226 stg; 227 op+=1; 228 litExtraF-=255; 229 setp lsicC; 230 bra→222;
     231 lbl; 232 sbAddr:=outBase+op; 233 stg; 234 op+=1`. -/
def lsicFS : List Nat := [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]

theorem lsicFS_iv : lsicFS = (List.range 13).map (· + 222) := rfl

theorem lsicFS_closed : PcClosed (WP.mk 15).kernel lsicFS [234] :=
  lsicFS_iv ▸ ivClosed_at (WP.mk 15).kernel 222 13 [234] shipped32_size (by omega) (by decide)

/-- **Bytes the tail's LSIC still owes at each program point.**  `op + lsicRem`
    is then constant along the region: the `op += 1` at 227 is paid for by the
    drop from `lx/255 + 1` to `lx/255`, and the `litExtraF -= 255` at 228 pays it
    back — which is why `op ≤ lenOff` alone is not preserved but this is. -/
def lsicRem (q : Nat) (lx : Nat) : Nat :=
  if q = 228 then lx / 255
  else if 231 ≤ q then 1
  else lx / 255 + 1

/-- **The invariant carried through the tail's LSIC loop.**

    The potential alone is not enough.  Step 228 is `litExtraF -= 255`, and its
    effect on `lx / 255` is a *decrease* only when `255 ≤ litExtraF`; below that
    the `UInt64` subtraction wraps and the potential jumps.  That fact is the
    loop guard, so it has to ride along — as does lane-uniformity of
    `litExtraF`, because the branch at 223 reads lane 0 while the bound is
    needed at every lane.  The guard clause has to hold at pc 230 as well as at
    222/223: 230 is the `bra` back to the head, and it is where the freshly
    recomputed `lsicC` has to be tied to `litExtraF` for the next iteration. -/
def LsicInv (l : Lane) (B : Nat) (st : SState) : Prop :=
  (st.regs "op" l).toNat + lsicRem st.pc ((st.regs "litExtraF" l).toNat) ≤ B
  ∧ st.regs "litExtraF" l = st.regs "litExtraF" 0
  ∧ (st.pc = 222 ∨ st.pc = 223 ∨ st.pc = 230 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "litExtraF" 0).toNat))
  ∧ (224 ≤ st.pc → st.pc ≤ 228 → 255 ≤ (st.regs "litExtraF" l).toNat)

/-- At the two `sbAddr` stores of the tail's LSIC run the invariant already
    bounds the cursor: `lsicRem` is `≥ 1` everywhere except pc 228, and neither
    226 nor 233 is 228. -/
theorem lsicInv_op_le (l : Lane) (B : Nat) (st : SState)
    (h : LsicInv l B st) (hq : st.pc = 226 ∨ st.pc = 233) :
    (st.regs "op" l).toNat < B := by
  have h1 := h.1
  rcases hq with e | e
  · rw [e] at h1
    have hr : lsicRem 226 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
    omega
  · rw [e] at h1
    have hr : lsicRem 233 ((st.regs "litExtraF" l).toNat) = 1 := rfl
    omega

-- The per-instruction verification conditions for the tail's LSIC loop.
-- `maxRecDepth` is for the `decide`s that read an instruction out of the
-- 274-entry emitted array; nothing here is deep recursion of its own.
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

abbrev K := (WP.mk 15).kernel

/-- `true` unless the instruction is a 4-byte store. -/
def noWide (o : Option SInstr) : Bool :=
  match o with | some (SInstr.stg32p _ _ _) => false | _ => true

theorem shipped32_no_stg32p_lt : ∀ q : Nat, q < 274 → noWide K[q]? = true := by
  have h : K.toList.all (fun i => noWide (some i)) = true := by decide
  intro q hq
  have hlt : q < K.size := by rw [shipped32_size]; exact hq
  rw [getElem?_pos K q hlt]
  exact List.all_eq_true.mp h K[q] (by simpa using Array.getElem_mem hlt)

/-- **No 4-byte store in the shipped kernel** — all sixteen store sites are 1-byte
    `stg`.  Decided against the emitted array. -/
theorem shipped32_no_stg32p : ∀ q : Nat, noWide K[q]? = true := by
  intro q
  rcases Nat.lt_or_ge q 274 with h | h
  · exact shipped32_no_stg32p_lt q h
  · rw [Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]; rfl

/-- **A write lands on the address register of an ACTIVE lane.**  `writes_at_site`
    flattens every store into a four-byte disjunction and ignores predicates,
    which is right for a frame and wrong for confinement: it would demand that the
    three bytes above each store, and the addresses of masked lanes, be in region
    too — both false in this kernel. -/
theorem writes_at_site32 (st : SState) (j : Nat) (h : WritesAct K st j) :
    ∃ r : String, (st.pc, r) ∈ storeSites K ∧ ∃ l : Lane,
      ActiveAt K st l ∧ (st.regs r l).toNat = j := by
  rw [WritesAct] at h
  have hno := shipped32_no_stg32p st.pc
  cases hp : K[st.pc]? with
  | none => rw [hp] at h; exact absurd h not_false
  | some i =>
      rw [hp] at h
      rw [hp] at hno
      have site : ∀ (addr : String),
          (fun i => match i with
            | SInstr.stg addr _ => some addr
            | SInstr.stgp _ addr _ => some addr
            | SInstr.stg32p _ addr _ => some addr
            | _ => none) i = some addr → (st.pc, addr) ∈ storeSites K := by
        intro addr hf
        have : K.toList[st.pc]? = some i := by simpa using hp
        simpa using mem_siteAux _ K.toList 0 st.pc i addr this hf
      cases i with
      | stg addr s =>
          obtain ⟨l, hl⟩ := h
          exact ⟨addr, site addr rfl, l, by rw [ActiveAt, hp]; trivial, hl⟩
      | stgp q addr s =>
          obtain ⟨l, hq, hl⟩ := h
          exact ⟨addr, site addr rfl, l, by rw [ActiveAt, hp]; exact hq, hl⟩
      | stg32p q addr s => rw [noWide] at hno; exact absurd hno (by decide)
      | _ => exact absurd h not_false

/-- **`RegConfined` discharges `KernelConfined`** at the shipped 32 KiB geometry.
    Everything about *where* the kernel touches memory is proven; what is assumed
    is only *what the address registers hold when they are used*. -/
theorem kernelConfined_of_regConfined32 (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (h : RegConfined 15 inPtr outPtr gm smemB) :
    Lz4Interleave.KernelConfined 15 inPtr outPtr gm smemB where
  writes := by
    intro w k j hw
    obtain ⟨r, hsite, l, hact, hl⟩ := writes_at_site32 _ j hw
    rw [← hl]
    exact h.stores w k r hsite l hact
  reads := by
    intro w w' k j _hne hr hreg
    obtain ⟨r, off, hsite, l, hl⟩ := reads_at_site _ _ j hr
    have hoff : off ≤ 3 := (load_regs32 _ hsite).2
    have hlt := h.loads w k r off hsite l
    rw [hl] at hlt
    obtain ⟨hge, _⟩ := hreg
    have : outPtr ≤ outPtr + w'.val * (WP.mk 15).outStride := Nat.le_add_right _ _
    omega




theorem lsic_frame (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litExtraF" ∨ r = "lsicC" →
      (sstep K st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRem q' x ≤ lsicRem st.pc x)
    (h3' : (q' = 222 ∨ q' = 223 ∨ q' = 230) → (st.pc = 222 ∨ st.pc = 223 ∨ st.pc = 230))
    (h4' : 224 ≤ q' → q' ≤ 228 → 224 ≤ st.pc ∧ st.pc ≤ 228)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litExtraF" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "litExtraF" l (Or.inr (Or.inl rfl)), hfr "litExtraF" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "litExtraF" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "litExtraF" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsic_at222 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 222)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lh18") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 223 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at224 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 224)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 225 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at225 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 225)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 226 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at226 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 226)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 227 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at231 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 231)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lx19") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 232 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at232 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 232)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 233 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at233 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 233)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "litExtraF") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "litExtraF"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 234 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsic_at230 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 230)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Lh18") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (sfindLabel K "Lh18") := by rw [sstep, hp]; rfl
  refine lsic_frame l B st 222 (by rw [hstep]; show sfindLabel K "Lh18" = 222; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsic_at223 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 223)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "lsicC" "Lx19") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K "Lx19") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K "Lx19" = 231 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K st).pc = 224 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRem 224 ((st.regs "litExtraF" l).toNat)
          = lsicRem 223 ((st.regs "litExtraF" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K st).pc = 231 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRem 223 ((st.regs "litExtraF" l).toNat)
          = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRem 231 ((st.regs "litExtraF" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsic_at227 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 227)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 228 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "litExtraF" j = st.regs "litExtraF" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRem 227 ((st.regs "litExtraF" l).toNat)
      = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRem 228 ((st.regs "litExtraF" l).toNat)
      = (st.regs "litExtraF" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsic_at228 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 228)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .sub "litExtraF" "litExtraF" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "litExtraF"
      (fun l => SOp.sub.run (st.regs "litExtraF" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "litExtraF" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K st).pc = 229 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K st).regs "litExtraF" j = st.regs "litExtraF" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRem 228 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 := rfl
    have hr229 : lsicRem 229 ((st.regs "litExtraF" l).toNat - 255)
        = ((st.regs "litExtraF" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsic_at229 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 229)
    (h : LsicInv l B st) : LsicInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp .ge "lsicC" "litExtraF" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "litExtraF" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 230 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "litExtraF" j = st.regs "litExtraF" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRem 229 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRem 230 ((st.regs "litExtraF" l).toNat)
        = (st.regs "litExtraF" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "litExtraF" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "litExtraF" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

/-- **The tail's LSIC loop preserves its invariant, at every one of its pcs.** -/
theorem lsicFS_hstep (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicFS) (hex : st.pc ∉ [234]) (h : LsicInv l B st) :
    LsicInv l B (sstep K st) := by
  simp only [lsicFS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsic_at222 l B st e h
  · exact lsic_at223 l B st e h
  · exact lsic_at224 l B st e h
  · exact lsic_at225 l B st e h
  · exact lsic_at226 l B st e h
  · exact lsic_at227 l B st hB e h
  · exact lsic_at228 l B st e h
  · exact lsic_at229 l B st e h
  · exact lsic_at230 l B st e h
  · exact lsic_at231 l B st e h
  · exact lsic_at232 l B st e h
  · exact lsic_at233 l B st e h
  · exact absurd (by simp [e]) hex

/-- **Sites 226 and 233 are confined.**  From a state at the loop head (pc 222)
    satisfying the invariant, the output cursor is below `B` at every visit to
    either `sbAddr` store, for the whole run of the loop. -/
theorem lsic_op_lt (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 222) (h : LsicInv l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K j st).pc ∉ [234])
    (hq : (siter K k st).pc = 226 ∨ (siter K k st).pc = 233) :
    ((siter K k st).regs "op" l).toNat < B :=
  lsicInv_op_le l B _
    (inv_on K (LsicInv l B) lsicFS [234] lsicFS_closed
      (fun s hsm hexs hh => lsicFS_hstep l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

-- ── The body's literal-length LSIC loop: sites 140 and 147 ────────────────
-- The same thirteen instructions as the tail's run at pcs 222-234, on
-- `litExtra` instead of `litExtraF`.  The potential omits the bytes the token
-- still owes after this run (the literal copy, the two offset bytes and the
-- match LSIC), which only makes the bound weaker and so is free.

/-- The tail's length-extension loop with its exit run, pcs 136–148:
    `136 lbl; 137 braifnot lsicC→145; 138 c255:=255; 139 sbAddr:=outBase+op;
     140 stg; 141 op+=1; 142 litExtraF-=255; 143 setp lsicC; 144 bra→136;
     145 lbl; 146 sbAddr:=outBase+op; 147 stg; 148 op+=1`. -/
def lsicLS : List Nat := [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148]

theorem lsicLS_iv : lsicLS = (List.range 13).map (· + 136) := rfl

theorem lsicLS_closed : PcClosed (WP.mk 15).kernel lsicLS [148] :=
  lsicLS_iv ▸ ivClosed_at (WP.mk 15).kernel 136 13 [148] shipped32_size (by omega) (by decide)

/-- **Bytes the tail's LSIC still owes at each program point.**  `op + lsicRemL`
    is then constant along the region: the `op += 1` at 141 is paid for by the
    drop from `lx/255 + 1` to `lx/255`, and the `litExtraF -= 255` at 142 pays it
    back — which is why `op ≤ lenOff` alone is not preserved but this is. -/
def lsicRemL (q : Nat) (lx : Nat) : Nat :=
  if q = 142 then lx / 255
  else if 145 ≤ q then 1
  else lx / 255 + 1

/-- **The invariant carried through the tail's LSIC loop.**

    The potential alone is not enough.  Step 142 is `litExtraF -= 255`, and its
    effect on `lx / 255` is a *decrease* only when `255 ≤ litExtraF`; below that
    the `UInt64` subtraction wraps and the potential jumps.  That fact is the
    loop guard, so it has to ride along — as does lane-uniformity of
    `litExtraF`, because the branch at 137 reads lane 0 while the bound is
    needed at every lane.  The guard clause has to hold at pc 144 as well as at
    136/137: 144 is the `bra` back to the head, and it is where the freshly
    recomputed `lsicC` has to be tied to `litExtraF` for the next iteration. -/
def LsicInvL (l : Lane) (B : Nat) (st : SState) : Prop :=
  (st.regs "op" l).toNat + lsicRemL st.pc ((st.regs "litExtra" l).toNat) ≤ B
  ∧ st.regs "litExtra" l = st.regs "litExtra" 0
  -- NOT at 143/191: those are the `setp`s that recompute `lsicC`, so the
  -- register is stale there and the equivalence genuinely fails.
  ∧ (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "litExtra" 0).toNat))
  ∧ (138 ≤ st.pc → st.pc ≤ 142 → 255 ≤ (st.regs "litExtra" l).toNat)

/-- At the two `sbAddr` stores of the tail's LSIC run the invariant already
    bounds the cursor: `lsicRemL` is `≥ 1` everywhere except pc 142, and neither
    140 nor 147 is 142. -/
theorem lsicLInv_op_le (l : Lane) (B : Nat) (st : SState)
    (h : LsicInvL l B st) (hq : st.pc = 140 ∨ st.pc = 147) :
    (st.regs "op" l).toNat < B := by
  have h1 := h.1
  rcases hq with e | e
  · rw [e] at h1
    have hr : lsicRemL 140 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [e] at h1
    have hr : lsicRemL 147 ((st.regs "litExtra" l).toNat) = 1 := rfl
    omega

-- The per-instruction verification conditions for the tail's LSIC loop.
-- `maxRecDepth` is for the `decide`s that read an instruction out of the
-- 274-entry emitted array; nothing here is deep recursion of its own.
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000


theorem lsicL_frame (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litExtra" ∨ r = "lsicC" →
      (sstep K st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRemL q' x ≤ lsicRemL st.pc x)
    (h3' : (q' = 136 ∨ q' = 137 ∨ q' = 144) → (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4' : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "litExtra" l (Or.inr (Or.inl rfl)), hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsicL_at222 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 136)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lh8") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 137 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at224 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 138)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 139 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at225 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 139)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 140 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at226 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 140)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 141 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at231 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 145)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lx9") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 146 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at232 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 146)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 147 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at233 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 147)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "litExtra") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "litExtra"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 148 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicL_at230 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 144)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Lh8") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (sfindLabel K "Lh8") := by rw [sstep, hp]; rfl
  refine lsicL_frame l B st 136 (by rw [hstep]; show sfindLabel K "Lh8" = 136; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsicL_at223 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 137)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "lsicC" "Lx9") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K "Lx9") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K "Lx9" = 145 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K st).pc = 138 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "litExtra" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRemL 138 ((st.regs "litExtra" l).toNat)
          = lsicRemL 137 ((st.regs "litExtra" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K st).pc = 145 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRemL 137 ((st.regs "litExtra" l).toNat)
          = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRemL 145 ((st.regs "litExtra" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsicL_at227 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 141)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 142 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "litExtra" j = st.regs "litExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "litExtra" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRemL 141 ((st.regs "litExtra" l).toNat)
      = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRemL 142 ((st.regs "litExtra" l).toNat)
      = (st.regs "litExtra" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsicL_at228 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 142)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .sub "litExtra" "litExtra" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "litExtra"
      (fun l => SOp.sub.run (st.regs "litExtra" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "litExtra" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K st).pc = 143 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K st).regs "litExtra" j = st.regs "litExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRemL 142 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 := rfl
    have hr229 : lsicRemL 143 ((st.regs "litExtra" l).toNat - 255)
        = ((st.regs "litExtra" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsicL_at229 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 143)
    (h : LsicInvL l B st) : LsicInvL l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp .ge "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "litExtra" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 144 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "litExtra" j = st.regs "litExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRemL 143 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRemL 144 ((st.regs "litExtra" l).toNat)
        = (st.regs "litExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "litExtra" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "litExtra" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

/-- **The tail's LSIC loop preserves its invariant, at every one of its pcs.** -/
theorem lsicLS_hstep (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicLS) (hex : st.pc ∉ [148]) (h : LsicInvL l B st) :
    LsicInvL l B (sstep K st) := by
  simp only [lsicLS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsicL_at222 l B st e h
  · exact lsicL_at223 l B st e h
  · exact lsicL_at224 l B st e h
  · exact lsicL_at225 l B st e h
  · exact lsicL_at226 l B st e h
  · exact lsicL_at227 l B st hB e h
  · exact lsicL_at228 l B st e h
  · exact lsicL_at229 l B st e h
  · exact lsicL_at230 l B st e h
  · exact lsicL_at231 l B st e h
  · exact lsicL_at232 l B st e h
  · exact lsicL_at233 l B st e h
  · exact absurd (by simp [e]) hex

/-- **Sites 140 and 147 are confined.**  From a state at the loop head (pc 136)
    satisfying the invariant, the output cursor is below `B` at every visit to
    either `sbAddr` store, for the whole run of the loop. -/
theorem lsicL_op_lt (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 136) (h : LsicInvL l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K j st).pc ∉ [148])
    (hq : (siter K k st).pc = 140 ∨ (siter K k st).pc = 147) :
    ((siter K k st).regs "op" l).toNat < B :=
  lsicLInv_op_le l B _
    (inv_on K (LsicInvL l B) lsicLS [148] lsicLS_closed
      (fun s hsm hexs hh => lsicLS_hstep l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq


-- ── The body's match-length LSIC loop: sites 188 and 195 ──────────────────
-- Third instance of the same run, on `matExtra`, at pcs 184-196.

/-- The tail's length-extension loop with its exit run, pcs 184–196:
    `184 lbl; 185 braifnot lsicC→193; 186 c255:=255; 187 sbAddr:=outBase+op;
     188 stg; 189 op+=1; 190 litExtraF-=255; 191 setp lsicC; 192 bra→184;
     193 lbl; 194 sbAddr:=outBase+op; 195 stg; 196 op+=1`. -/
def lsicMS : List Nat := [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]

theorem lsicMS_iv : lsicMS = (List.range 13).map (· + 184) := rfl

theorem lsicMS_closed : PcClosed (WP.mk 15).kernel lsicMS [196] :=
  lsicMS_iv ▸ ivClosed_at (WP.mk 15).kernel 184 13 [196] shipped32_size (by omega) (by decide)

/-- **Bytes the tail's LSIC still owes at each program point.**  `op + lsicRemM`
    is then constant along the region: the `op += 1` at 189 is paid for by the
    drop from `lx/255 + 1` to `lx/255`, and the `litExtraF -= 255` at 190 pays it
    back — which is why `op ≤ lenOff` alone is not preserved but this is. -/
def lsicRemM (q : Nat) (lx : Nat) : Nat :=
  if q = 190 then lx / 255
  else if 193 ≤ q then 1
  else lx / 255 + 1

/-- **The invariant carried through the tail's LSIC loop.**

    The potential alone is not enough.  Step 190 is `litExtraF -= 255`, and its
    effect on `lx / 255` is a *decrease* only when `255 ≤ litExtraF`; below that
    the `UInt64` subtraction wraps and the potential jumps.  That fact is the
    loop guard, so it has to ride along — as does lane-uniformity of
    `litExtraF`, because the branch at 185 reads lane 0 while the bound is
    needed at every lane.  The guard clause has to hold at pc 192 as well as at
    184/185: 192 is the `bra` back to the head, and it is where the freshly
    recomputed `lsicC` has to be tied to `litExtraF` for the next iteration. -/
def LsicInvM (l : Lane) (B : Nat) (st : SState) : Prop :=
  (st.regs "op" l).toNat + lsicRemM st.pc ((st.regs "matExtra" l).toNat) ≤ B
  ∧ st.regs "matExtra" l = st.regs "matExtra" 0
  ∧ (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "matExtra" 0).toNat))
  ∧ (186 ≤ st.pc → st.pc ≤ 190 → 255 ≤ (st.regs "matExtra" l).toNat)

/-- At the two `sbAddr` stores of the tail's LSIC run the invariant already
    bounds the cursor: `lsicRemM` is `≥ 1` everywhere except pc 190, and neither
    188 nor 195 is 190. -/
theorem lsicMInv_op_le (l : Lane) (B : Nat) (st : SState)
    (h : LsicInvM l B st) (hq : st.pc = 188 ∨ st.pc = 195) :
    (st.regs "op" l).toNat < B := by
  have h1 := h.1
  rcases hq with e | e
  · rw [e] at h1
    have hr : lsicRemM 188 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [e] at h1
    have hr : lsicRemM 195 ((st.regs "matExtra" l).toNat) = 1 := rfl
    omega

-- The per-instruction verification conditions for the tail's LSIC loop.
-- `maxRecDepth` is for the `decide`s that read an instruction out of the
-- 274-entry emitted array; nothing here is deep recursion of its own.
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000


theorem lsicM_frame (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "matExtra" ∨ r = "lsicC" →
      (sstep K st).regs r j = st.regs r j)
    (hrem : ∀ x : Nat, lsicRemM q' x ≤ lsicRemM st.pc x)
    (h3' : (q' = 184 ∨ q' = 185 ∨ q' = 192) → (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h4' : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "matExtra" l (Or.inr (Or.inl rfl))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _) _) h1
  · rw [hfr "matExtra" l (Or.inr (Or.inl rfl)), hfr "matExtra" 0 (Or.inr (Or.inl rfl))]
    exact h2
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr rfl)), hfr "matExtra" 0 (Or.inr (Or.inl rfl))]
    exact h3 (h3' hq)
  · rw [hpc']; intro ha hb
    rw [hfr "matExtra" l (Or.inr (Or.inl rfl))]
    exact h4 (h4' ha hb).1 (h4' ha hb).2

theorem lsicM_at222 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 184)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lh14") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 185 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at224 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 186)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "c255" (fun l => st.get l (SArg.imm 255))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 187 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at225 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 187)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 188 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at226 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 188)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "c255"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 189 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at231 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 193)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lx15") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 194 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at232 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 194)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "sbAddr" (fun l => SOp.add.run (st.regs "outBase" l) (st.get l (SArg.reg "op")))).setPc (st.pc + 1) := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 195 (by rw [hstep, he]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at233 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 195)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "matExtra") := by rw [he]; decide
  have hstep : sstep K st = { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs "sbAddr") (st.regs "matExtra"), pc := st.pc + 1 } := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 196 (by rw [hstep, he])
    (fun r j _ => by rw [hstep])
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h

theorem lsicM_at230 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 192)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Lh14") := by rw [he]; decide
  have hstep : sstep K st = st.setPc (sfindLabel K "Lh14") := by rw [sstep, hp]; rfl
  refine lsicM_frame l B st 184 (by rw [hstep]; show sfindLabel K "Lh14" = 184; decide)
    (fun r j hr => by rw [hstep]; rcases hr with rfl | rfl | rfl <;> rfl)
    (by rw [he]; intro x; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega) h


theorem lsicM_at223 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 185)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "lsicC" "Lx15") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K "Lx15") := by
    rw [sstep, hp]; rfl
  have hlbl : sfindLabel K "Lx15" = 193 := by decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hfr : ∀ (r : String) (j : Lane), (sstep K st).regs r j = st.regs r j := by
    intro r j; rw [hstep]; rfl
  rw [he] at h1
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · have hpc' : (sstep K st).pc = 186 := by rw [hstep, he, if_pos hg]; rfl
    have hlx : 255 ≤ (st.regs "matExtra" l).toNat := by
      rw [h2]; exact (h3 (Or.inr (Or.inl he))).mp hg
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr : lsicRemM 186 ((st.regs "matExtra" l).toNat)
          = lsicRemM 185 ((st.regs "matExtra" l).toNat) := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc', hfr]; intro _ _; exact hlx
  · have hpc' : (sstep K st).pc = 193 := by rw [hstep, if_neg hg, hlbl]; rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr, hfr]
      have hr223 : lsicRemM 185 ((st.regs "matExtra" l).toNat)
          = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
      have hr231 : lsicRemM 193 ((st.regs "matExtra" l).toNat) = 1 := rfl
      omega
    · rw [hfr, hfr]; exact h2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ hb; omega

theorem lsicM_at227 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 189)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .add "op" "op" (SArg.imm 1)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 190 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j + 1 := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "matExtra" j = st.regs "matExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : ∀ j : Lane, (sstep K st).regs "lsicC" j = st.regs "lsicC" j := by
    intro j; rw [hstep]; rfl
  have hlx : 255 ≤ (st.regs "matExtra" l).toNat := h4 (by omega) (by omega)
  rw [he] at h1
  have hr227 : lsicRemM 189 ((st.regs "matExtra" l).toNat)
      = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
  have hr228 : lsicRemM 190 ((st.regs "matExtra" l).toNat)
      = (st.regs "matExtra" l).toNat / 255 := rfl
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hle : (st.regs "op" l).toNat + 1 ≤ B := by omega
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l, hopN]; omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc', hlx' l]; intro _ _; exact hlx

theorem lsicM_at228 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 190)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin .sub "matExtra" "matExtra" (SArg.imm 255)) := by
    rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "matExtra"
      (fun l => SOp.sub.run (st.regs "matExtra" l) (st.get l (SArg.imm 255)))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hlx : 255 ≤ (st.regs "matExtra" l).toNat := h4 (by omega) (by omega)
  have hpc' : (sstep K st).pc = 191 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlxs : ∀ j : Lane,
      (sstep K st).regs "matExtra" j = st.regs "matExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hsubN : ∀ a : UInt64, 255 ≤ a.toNat → (a - UInt64.ofNat 255).toNat = a.toNat - 255 := by
    intro a ha
    rw [UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
      show 2 ^ 64 - 255 + a.toNat = 2 ^ 64 + (a.toNat - 255) from by
        have := a.toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlxs l, hsubN _ hlx]
    have hr228 : lsicRemM 190 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 := rfl
    have hr229 : lsicRemM 191 ((st.regs "matExtra" l).toNat - 255)
        = ((st.regs "matExtra" l).toNat - 255) / 255 + 1 := rfl
    have hdiv := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) hlx
    omega
  · rw [hlxs l, hlxs 0, h2]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _ hb; omega

theorem lsicM_at229 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 191)
    (h : LsicInvM l B st) : LsicInvM l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp .ge "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨h1, h2, h3, h4⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun l => if SCmp.ge.run (st.regs "matExtra" l) (st.get l (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 192 := by rw [hstep, he]; rfl
  have hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j := by
    intro j; rw [hstep]; rfl
  have hlx' : ∀ j : Lane, (sstep K st).regs "matExtra" j = st.regs "matExtra" j := by
    intro j; rw [hstep]; rfl
  have hlc' : (sstep K st).regs "lsicC" 0
      = (if SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) then 1 else 0) := by
    rw [hstep]; rfl
  rw [he] at h1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hlx' l]
    have hr229 : lsicRemM 191 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
    have hr230 : lsicRemM 192 ((st.regs "matExtra" l).toNat)
        = (st.regs "matExtra" l).toNat / 255 + 1 := rfl
    omega
  · rw [hlx' l, hlx' 0]; exact h2
  · rw [hpc', hlc', hlx' 0]; intro _
    have hiff : SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) = true
        ↔ 255 ≤ (st.regs "matExtra" 0).toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le,
        show ((UInt64.ofNat 255).toNat) = 255 from by decide]
    by_cases hc : SCmp.ge.run (st.regs "matExtra" 0) (UInt64.ofNat 255) = true
    · rw [if_pos hc]
      exact ⟨fun _ => hiff.mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro hcon; exact absurd hcon (by decide)
      · intro hn; exact absurd (hiff.mpr hn) hc
  · rw [hpc']; intro _ hb; omega

/-- **The tail's LSIC loop preserves its invariant, at every one of its pcs.** -/
theorem lsicMS_hstep (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ lsicMS) (hex : st.pc ∉ [196]) (h : LsicInvM l B st) :
    LsicInvM l B (sstep K st) := by
  simp only [lsicMS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact lsicM_at222 l B st e h
  · exact lsicM_at223 l B st e h
  · exact lsicM_at224 l B st e h
  · exact lsicM_at225 l B st e h
  · exact lsicM_at226 l B st e h
  · exact lsicM_at227 l B st hB e h
  · exact lsicM_at228 l B st e h
  · exact lsicM_at229 l B st e h
  · exact lsicM_at230 l B st e h
  · exact lsicM_at231 l B st e h
  · exact lsicM_at232 l B st e h
  · exact lsicM_at233 l B st e h
  · exact absurd (by simp [e]) hex

/-- **Sites 188 and 195 are confined.**  From a state at the loop head (pc 184)
    satisfying the invariant, the output cursor is below `B` at every visit to
    either `sbAddr` store, for the whole run of the loop. -/
theorem lsicM_op_lt (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 184) (h : LsicInvM l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K j st).pc ∉ [196])
    (hq : (siter K k st).pc = 188 ∨ (siter K k st).pc = 195) :
    ((siter K k st).regs "op" l).toNat < B :=
  lsicMInv_op_le l B _
    (inv_on K (LsicInvM l B) lsicMS [196] lsicMS_closed
      (fun s hsm hexs hh => lsicMS_hstep l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

-- ── The token store at pc 130 ────────────────────────────────────────────────

/-- Everything the `loopC` body can reach before it emits a token: the head and
    guard (40–41), the window/extend/match search (42–129) and the not-found
    branch (203–207).  Its two exits are the token store itself (130) and the
    loop's own exit to the tail (208). -/
def bodyPre : List Nat :=
  (List.range 90).map (· + 40) ++ [203, 204, 205, 206, 207, 130, 208]

theorem bodyPre_closed : PcClosed (WP.mk 15).kernel bodyPre [130, 208] := by decide

/-- **The token store happens at the cursor the loop head left.**  No instruction
    in `bodyPre` writes `op` — the first write after the prologue is the `op += 1`
    at 131 — so the first `sbAddr` store of an iteration is at `outBase + op_head`.
    With `op_head ≤ lenOff` from the emit loop's own bound, site 130 is confined
    without any potential argument, exactly as site 216 was in the tail. -/
theorem op_const_to_130 (st : SState) (h0 : st.pc = 40) (k : Nat)
    (hne : ∀ j, j < k → (siter (WP.mk 15).kernel j st).pc ∉ [130, 208]) :
    (siter (WP.mk 15).kernel k st).regs "op" = st.regs "op" :=
  regs_const_on (WP.mk 15).kernel "op" bodyPre [130, 208] bodyPre_closed (by decide) st
    (by rw [h0]; decide) k hne

-- ── The whole token emit as one potential: the remaining body sites ──────────

/-- LSIC bytes for a length field: one byte per full 255 plus the remainder
    byte, and nothing at all below 15 (the nibble holds it). -/
def lsicLen (n : Nat) : Nat := if 15 ≤ n then (n - 15) / 255 + 1 else 0

/-- **Bytes the token still owes at each program point of the emit**, pcs
    129–197.  `op + tokRem` is constant along the whole run: every `op` write is
    paid for by a drop here, and every register update that shrinks a length
    field is matched by the corresponding drop.

    This is one region rather than three chained ones because the chaining does
    not work: an LSIC potential that omits the bytes owed *after* it (which is
    what makes the standalone `lsicL`/`lsicM`/tail instances cheap) cannot hand
    its successor an entry condition. -/
def tokRem (q : Nat) (litLen litExtra mlm matExtra : Nat) : Nat :=
  if q ≤ 131 then 1 + lsicLen litLen + litLen + 2 + lsicLen mlm
  else if q ≤ 134 then lsicLen litLen + litLen + 2 + lsicLen mlm
  else if q ≤ 141 then litExtra / 255 + 1 + litLen + 2 + lsicLen mlm
  else if q = 142 then litExtra / 255 + litLen + 2 + lsicLen mlm
  else if q ≤ 148 then litExtra / 255 + 1 + litLen + 2 + lsicLen mlm
  else if q ≤ 170 then litLen + 2 + lsicLen mlm
  else if q ≤ 174 then 2 + lsicLen mlm
  else if q ≤ 179 then 1 + lsicLen mlm
  else if q ≤ 182 then lsicLen mlm
  else if q ≤ 189 then matExtra / 255 + 1
  else if q = 190 then matExtra / 255
  else if q ≤ 196 then matExtra / 255 + 1
  else 0

/-- At every `sbAddr` store of the token emit the potential still owes at least
    one byte, so the invariant's `op + tokRem ≤ B` gives `op < B` there. -/
theorem tokRem_pos (q : Nat) (hq : q = 130 ∨ q = 140 ∨ q = 147 ∨ q = 173 ∨ q = 178
    ∨ q = 188 ∨ q = 195) (a b c d : Nat) : 1 ≤ tokRem q a b c d := by
  rcases hq with e | e | e | e | e | e | e
  · subst e; show 1 ≤ 1 + lsicLen a + a + 2 + lsicLen c; omega
  · subst e; show 1 ≤ b / 255 + 1 + a + 2 + lsicLen c; omega
  · subst e; show 1 ≤ b / 255 + 1 + a + 2 + lsicLen c; omega
  · subst e; show 1 ≤ 2 + lsicLen c; omega
  · subst e; show 1 ≤ 1 + lsicLen c; omega
  · subst e; show 1 ≤ d / 255 + 1; omega
  · subst e; show 1 ≤ d / 255 + 1; omega

/-- The token emit and its two exits (`197` is the `bra` past the match LSIC,
    `198` the `pMatBig` else-label). -/
def tokS : List Nat := [129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198]

theorem tokS_iv : tokS = (List.range 70).map (· + 129) := rfl

theorem tokS_closed : PcClosed (WP.mk 15).kernel tokS [197, 198] :=
  tokS_iv ▸ ivClosed_at (WP.mk 15).kernel 129 70 [197, 198] shipped32_size (by omega) (by decide)

/-- **The invariant carried through the whole token emit.**  One potential plus
    the four guards the arithmetic needs (the two LSIC counters must be `≥ 255`
    where they are decremented, and the two length fields must be `≥ 15` where
    their extension bytes are being counted), plus lane-uniformity, because the
    branches read lane 0 while the bound is needed at every lane. -/
def TokInv (l : Lane) (B : Nat) (st : SState) : Prop :=
  (st.regs "op" l).toNat + tokRem st.pc ((st.regs "litLen" l).toNat)
      ((st.regs "litExtra" l).toNat) ((st.regs "mlm" l).toNat)
      ((st.regs "matExtra" l).toNat) ≤ B
  ∧ (∀ r : String, r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra"
      ∨ r = "lsicC" → st.regs r l = st.regs r 0)
  -- NOT at 143/191: those are the `setp`s that recompute `lsicC`, so the
  -- register is stale there and the equivalence genuinely fails.
  ∧ (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "litExtra" 0).toNat))
  ∧ (138 ≤ st.pc → st.pc ≤ 142 → 255 ≤ (st.regs "litExtra" l).toNat)
  ∧ (st.pc = 134 → 15 ≤ (st.regs "litLen" l).toNat)
  ∧ (st.pc = 133 →
      (((st.regs "pLitBig" 0) == 1) = true ↔ 15 ≤ (st.regs "litLen" 0).toNat))
  ∧ (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "matExtra" 0).toNat))
  ∧ (186 ≤ st.pc → st.pc ≤ 190 → 255 ≤ (st.regs "matExtra" l).toNat)
  ∧ (st.pc = 182 → 15 ≤ (st.regs "mlm" l).toNat)
  ∧ (st.pc = 181 →
      (((st.regs "pMatBig" 0) == 1) = true ↔ 15 ≤ (st.regs "mlm" 0).toNat))

/-- The invariant bounds the cursor at all seven `sbAddr` stores of the emit. -/
theorem tokInv_op_lt (l : Lane) (B : Nat) (st : SState) (h : TokInv l B st)
    (hq : st.pc = 130 ∨ st.pc = 140 ∨ st.pc = 147 ∨ st.pc = 173 ∨ st.pc = 178
      ∨ st.pc = 188 ∨ st.pc = 195) :
    (st.regs "op" l).toNat < B := by
  have h1 := h.1
  have hp := tokRem_pos st.pc hq ((st.regs "litLen" l).toNat)
    ((st.regs "litExtra" l).toNat) ((st.regs "mlm" l).toNat) ((st.regs "matExtra" l).toNat)
  omega

-- The per-instruction verification conditions for the token emit.  Forty-five
-- of the sixty-eight are pure frame steps and go through `tok_frame`; the
-- rest either move the cursor, shrink a length field, or establish a guard
-- for their successor, and are proven individually below.

theorem tok_frame (l : Lane) (B : Nat) (st : SState) (q' : Nat)
    (hpc' : (sstep K st).pc = q')
    (hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
        ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j)
    (hrem : ∀ a b c d : Nat, tokRem q' a b c d ≤ tokRem st.pc a b c d)
    (h3 : (q' = 136 ∨ q' = 137 ∨ q' = 144) →
      (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4 : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h5 : q' = 134 → st.pc = 134)
    (h6 : q' = 133 → st.pc = 133)
    (h7 : (q' = 184 ∨ q' = 185 ∨ q' = 192) →
      (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h8 : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h9 : q' = 182 → st.pc = 182)
    (h10 : q' = 181 → st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact Nat.le_trans (Nat.add_le_add_left (hrem _ _ _ _) _) c1
  · intro r hr
    have hr8 : r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra"
        ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" := by
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl <;> simp
    rw [hfr r l hr8, hfr r 0 hr8]; exact c2 r hr
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]; exact c3 (h3 hq)
  · rw [hpc']; intro ha hb; rw [hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl)))]
    exact c4 (h4 ha hb).1 (h4 ha hb).2
  · rw [hpc']; intro hq; rw [hfr "litLen" l (Or.inr (Or.inl rfl))]; exact c5 (h5 hq)
  · rw [hpc']; intro hq
    rw [hfr "pLitBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))), hfr "litLen" 0 (Or.inr (Or.inl rfl))]; exact c6 (h6 hq)
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact c7 (h7 hq)
  · rw [hpc']; intro ha hb; rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact c8 (h8 ha hb).1 (h8 ha hb).2
  · rw [hpc']; intro hq; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact c9 (h9 hq)
  · rw [hpc']; intro hq
    rw [hfr "pMatBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl)))))))), hfr "mlm" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact c10 (h10 hq)

theorem tok_at129 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 129)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 130 (by rw [sstep, hp]; show st.pc + 1 = 130; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at130 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 130)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "tok") := by rw [he]; decide
  refine tok_frame l B st 131 (by rw [sstep, hp]; show st.pc + 1 = 131; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at136 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 136)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lh8") := by rw [he]; decide
  refine tok_frame l B st 137 (by rw [sstep, hp]; show st.pc + 1 = 137; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at138 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 138)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame l B st 139 (by rw [sstep, hp]; show st.pc + 1 = 139; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at139 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 139)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 140 (by rw [sstep, hp]; show st.pc + 1 = 140; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at140 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 140)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  refine tok_frame l B st 141 (by rw [sstep, hp]; show st.pc + 1 = 141; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at144 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 144)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Lh8") := by rw [he]; decide
  refine tok_frame l B st 136 (by rw [sstep, hp]; show sfindLabel K "Lh8" = 136; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at145 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 145)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lx9") := by rw [he]; decide
  refine tok_frame l B st 146 (by rw [sstep, hp]; show st.pc + 1 = 146; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at146 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 146)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 147 (by rw [sstep, hp]; show st.pc + 1 = 147; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at147 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 147)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "litExtra") := by rw [he]; decide
  refine tok_frame l B st 148 (by rw [sstep, hp]; show st.pc + 1 = 148; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at149 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 149)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Ln7") := by rw [he]; decide
  refine tok_frame l B st 151 (by rw [sstep, hp]; show sfindLabel K "Ln7" = 151; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at150 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 150)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Le6") := by rw [he]; decide
  refine tok_frame l B st 151 (by rw [sstep, hp]; show st.pc + 1 = 151; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at151 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 151)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Ln7") := by rw [he]; decide
  refine tok_frame l B st 152 (by rw [sstep, hp]; show st.pc + 1 = 152; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at152 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 152)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "cpDst" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 153 (by rw [sstep, hp]; show st.pc + 1 = 153; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at153 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 153)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "cpSrc" "inBase" (SArg.reg "litAnchor")) := by rw [he]; decide
  refine tok_frame l B st 154 (by rw [sstep, hp]; show st.pc + 1 = 154; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at154 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 154)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "cpI" (SArg.imm 0)) := by rw [he]; decide
  refine tok_frame l B st 155 (by rw [sstep, hp]; show st.pc + 1 = 155; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at155 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 155)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame l B st 156 (by rw [sstep, hp]; show st.pc + 1 = 156; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at156 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 156)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Ch10") := by rw [he]; decide
  refine tok_frame l B st 157 (by rw [sstep, hp]; show st.pc + 1 = 157; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at158 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 158)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.binr (.add) "cpDo" "cpDst" "cpI") := by rw [he]; decide
  refine tok_frame l B st 159 (by rw [sstep, hp]; show st.pc + 1 = 159; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at159 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 159)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.binr (.add) "cpDo" "cpDo" "lane") := by rw [he]; decide
  refine tok_frame l B st 160 (by rw [sstep, hp]; show st.pc + 1 = 160; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at160 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 160)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.binr (.add) "cpSo" "cpSrc" "cpI") := by rw [he]; decide
  refine tok_frame l B st 161 (by rw [sstep, hp]; show st.pc + 1 = 161; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at161 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 161)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.binr (.add) "cpSo" "cpSo" "lane") := by rw [he]; decide
  refine tok_frame l B st 162 (by rw [sstep, hp]; show st.pc + 1 = 162; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at162 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 162)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.binr (.add) "cpJ" "cpI" "lane") := by rw [he]; decide
  refine tok_frame l B st 163 (by rw [sstep, hp]; show st.pc + 1 = 163; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at163 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 163)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.lt) "cpP" "cpJ" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame l B st 164 (by rw [sstep, hp]; show st.pc + 1 = 164; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at164 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 164)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.ldgo "cpB" "cpSo" 0) := by rw [he]; decide
  refine tok_frame l B st 165 (by rw [sstep, hp]; show st.pc + 1 = 165; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at165 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 165)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stgp "cpP" "cpDo" "cpB") := by rw [he]; decide
  refine tok_frame l B st 166 (by rw [sstep, hp]; show st.pc + 1 = 166; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at166 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 166)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "cpI" "cpI" (SArg.imm 32)) := by rw [he]; decide
  refine tok_frame l B st 167 (by rw [sstep, hp]; show st.pc + 1 = 167; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at167 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 167)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.lt) "cpCont" "cpI" (SArg.reg "litLen")) := by rw [he]; decide
  refine tok_frame l B st 168 (by rw [sstep, hp]; show st.pc + 1 = 168; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at168 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 168)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Ch10") := by rw [he]; decide
  refine tok_frame l B st 156 (by rw [sstep, hp]; show sfindLabel K "Ch10" = 156; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at169 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 169)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Cx11") := by rw [he]; decide
  refine tok_frame l B st 170 (by rw [sstep, hp]; show st.pc + 1 = 170; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at171 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 171)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.band) "offLo" "off0" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame l B st 172 (by rw [sstep, hp]; show st.pc + 1 = 172; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at172 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 172)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 173 (by rw [sstep, hp]; show st.pc + 1 = 173; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at173 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 173)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "offLo") := by rw [he]; decide
  refine tok_frame l B st 174 (by rw [sstep, hp]; show st.pc + 1 = 174; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at175 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 175)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.shr) "offHi" "off0" (SArg.imm 8)) := by rw [he]; decide
  refine tok_frame l B st 176 (by rw [sstep, hp]; show st.pc + 1 = 176; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at176 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 176)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.band) "offHi" "offHi" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame l B st 177 (by rw [sstep, hp]; show st.pc + 1 = 177; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at177 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 177)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 178 (by rw [sstep, hp]; show st.pc + 1 = 178; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at178 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 178)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "offHi") := by rw [he]; decide
  refine tok_frame l B st 179 (by rw [sstep, hp]; show st.pc + 1 = 179; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at184 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 184)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lh14") := by rw [he]; decide
  refine tok_frame l B st 185 (by rw [sstep, hp]; show st.pc + 1 = 185; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at186 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 186)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.mov "c255" (SArg.imm 255)) := by rw [he]; decide
  refine tok_frame l B st 187 (by rw [sstep, hp]; show st.pc + 1 = 187; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at187 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 187)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 188 (by rw [sstep, hp]; show st.pc + 1 = 188; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at188 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 188)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "c255") := by rw [he]; decide
  refine tok_frame l B st 189 (by rw [sstep, hp]; show st.pc + 1 = 189; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at192 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 192)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bra "Lh14") := by rw [he]; decide
  refine tok_frame l B st 184 (by rw [sstep, hp]; show sfindLabel K "Lh14" = 184; decide)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at193 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 193)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.lbl "Lx15") := by rw [he]; decide
  refine tok_frame l B st 194 (by rw [sstep, hp]; show st.pc + 1 = 194; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at194 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 194)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) := by rw [he]; decide
  refine tok_frame l B st 195 (by rw [sstep, hp]; show st.pc + 1 = 195; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at195 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 195)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.stg "sbAddr" "matExtra") := by rw [he]; decide
  refine tok_frame l B st 196 (by rw [sstep, hp]; show st.pc + 1 = 196; omega)
    (fun r j hr => by rw [sstep, hp]; first | rfl | (rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl))
    (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) h

/-- The seven `op += 1` steps of the emit.  Each is paid for by a drop of one in
    `tokRem`, so the potential is unchanged; the no-wrap side condition comes
    free from the potential itself (`op + rem ≤ B < 2 ^ 64` and `rem ≥ 1`). -/
theorem tok_op1 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (q' : Nat)
    (hpc' : (sstep K st).pc = q')
    (hop' : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j + 1)
    (hfr : ∀ (r : String) (j : Lane), r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
        ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j)
    (hrem : ∀ a b c d : Nat, tokRem q' a b c d + 1 ≤ tokRem st.pc a b c d)
    (h3 : (q' = 136 ∨ q' = 137 ∨ q' = 144) →
      (st.pc = 136 ∨ st.pc = 137 ∨ st.pc = 144))
    (h4 : 138 ≤ q' → q' ≤ 142 → 138 ≤ st.pc ∧ st.pc ≤ 142)
    (h5 : q' = 134 → st.pc = 134) (h6 : q' = 133 → st.pc = 133)
    (h7 : (q' = 184 ∨ q' = 185 ∨ q' = 192) →
      (st.pc = 184 ∨ st.pc = 185 ∨ st.pc = 192))
    (h8 : 186 ≤ q' → q' ≤ 190 → 186 ≤ st.pc ∧ st.pc ≤ 190)
    (h9 : q' = 182 → st.pc = 182) (h10 : q' = 181 → st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hr := hrem (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
    (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
  have hopN : ((st.regs "op" l) + 1).toNat = (st.regs "op" l).toNat + 1 := by
    have hb := (st.regs "op" l).toNat_lt
    have hL := (toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hop' l, hopN, hfr "litLen" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl)),
      hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    omega
  · intro r hr2
    rcases hr2 with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hop' l, hop' 0, c2 "op" (Or.inl rfl)]
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
      hfr "litExtra" 0 (Or.inr (Or.inl rfl))]
    exact c3 (h3 hq)
  · rw [hpc']; intro ha hb2; rw [hfr "litExtra" l (Or.inr (Or.inl rfl))]
    exact c4 (h4 ha hb2).1 (h4 ha hb2).2
  · rw [hpc']; intro hq; rw [hfr "litLen" l (Or.inl rfl)]; exact c5 (h5 hq)
  · rw [hpc']; intro hq
    rw [hfr "pLitBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))),
      hfr "litLen" 0 (Or.inl rfl)]
    exact c6 (h6 hq)
  · rw [hpc']; intro hq
    rw [hfr "lsicC" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
      hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact c7 (h7 hq)
  · rw [hpc']; intro ha hb2
    rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact c8 (h8 ha hb2).1 (h8 ha hb2).2
  · rw [hpc']; intro hq; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl)))]; exact c9 (h9 hq)
  · rw [hpc']; intro hq
    rw [hfr "pMatBig" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))),
      hfr "mlm" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact c10 (h10 hq)

theorem tokRem_drop131 : ∀ a b c d : Nat,
    tokRem 132 a b c d + 1 ≤ tokRem 131 a b c d := by
  intro a b c d
  have e1 : tokRem 131 a b c d = 1 + lsicLen a + a + 2 + lsicLen c := rfl
  have e2 : tokRem 132 a b c d = lsicLen a + a + 2 + lsicLen c := rfl
  omega

theorem tokRem_drop141 : ∀ a b c d : Nat,
    tokRem 142 a b c d + 1 ≤ tokRem 141 a b c d := by
  intro a b c d
  have e1 : tokRem 141 a b c d = b / 255 + 1 + a + 2 + lsicLen c := rfl
  have e2 : tokRem 142 a b c d = b / 255 + a + 2 + lsicLen c := rfl
  omega

theorem tokRem_drop148 : ∀ a b c d : Nat,
    tokRem 149 a b c d + 1 ≤ tokRem 148 a b c d := by
  intro a b c d
  have e1 : tokRem 148 a b c d = b / 255 + 1 + a + 2 + lsicLen c := rfl
  have e2 : tokRem 149 a b c d = a + 2 + lsicLen c := rfl
  omega

theorem tokRem_drop174 : ∀ a b c d : Nat,
    tokRem 175 a b c d + 1 ≤ tokRem 174 a b c d := by
  intro a b c d
  have e1 : tokRem 174 a b c d = 2 + lsicLen c := rfl
  have e2 : tokRem 175 a b c d = 1 + lsicLen c := rfl
  omega

theorem tokRem_drop179 : ∀ a b c d : Nat,
    tokRem 180 a b c d + 1 ≤ tokRem 179 a b c d := by
  intro a b c d
  have e1 : tokRem 179 a b c d = 1 + lsicLen c := rfl
  have e2 : tokRem 180 a b c d = lsicLen c := rfl
  omega

theorem tokRem_drop189 : ∀ a b c d : Nat,
    tokRem 190 a b c d + 1 ≤ tokRem 189 a b c d := by
  intro a b c d
  have e1 : tokRem 189 a b c d = d / 255 + 1 := rfl
  have e2 : tokRem 190 a b c d = d / 255 := rfl
  omega

theorem tokRem_drop196 : ∀ a b c d : Nat,
    tokRem 197 a b c d + 1 ≤ tokRem 196 a b c d := by
  intro a b c d
  have e1 : tokRem 196 a b c d = d / 255 + 1 := rfl
  have e2 : tokRem 197 a b c d = 0 := rfl
  omega

theorem tok_at131 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 131)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 132 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop131)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at141 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 141)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 142 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop141)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at148 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 148)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 149 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop148)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at174 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 174)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 175 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop174)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at179 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 179)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 180 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop179)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at189 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 189)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 190 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop189)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at196 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 196)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.imm 1)) := by rw [he]; decide
  have hstep : sstep K st = (st.setReg "op"
      (fun l => SOp.add.run (st.regs "op" l) (st.get l (SArg.imm 1)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  refine tok_op1 l B st hB 197 (by rw [hstep, he]; rfl)
    (fun j => by rw [hstep]; rfl)
    (fun r j hr => by rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl)
    (by rw [he]; exact tokRem_drop196)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
    (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) h

/-- What a `setp.ge` against an immediate leaves in the predicate register, as an
    arithmetic statement.  Shared by all six guard-establishing steps. -/
theorem setp_ge_iff (a : UInt64) (V : Nat) (hV : (UInt64.ofNat V).toNat = V) :
    (((if SCmp.ge.run a (UInt64.ofNat V) then (1 : UInt64) else 0) == 1) = true)
      ↔ V ≤ a.toNat := by
  by_cases hc : SCmp.ge.run a (UInt64.ofNat V) = true
  · rw [if_pos hc]
    have hle : V ≤ a.toNat := by
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le, hV] at hc
      exact hc
    exact ⟨fun _ => hle, fun _ => rfl⟩
  · rw [if_neg hc]
    constructor
    · intro hcon; exact absurd hcon (by decide)
    · intro hn
      exfalso; apply hc
      simp only [SCmp.run, decide_eq_true_eq, UInt64.le_iff_toNat_le, hV]
      exact hn

/-- pc 135: `setp.ge lsicC, litExtra, 255` — establishes the literal-LSIC guard
    that pc 142's wrap-free subtraction will need, seven steps later. -/
theorem tok_at135 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 135)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 136 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "litExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
      hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 136 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
        (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = tokRem 135 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
        (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact setp_ge_iff (st.regs "litExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at143 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 143)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "lsicC" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 144 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "litExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 144 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 143 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litExtra" 0 (Or.inr (Or.inr (Or.inl rfl)))]
    exact setp_ge_iff (st.regs "litExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at183 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 183)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 184 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "matExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 184 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 183 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact setp_ge_iff (st.regs "matExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at191 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 191)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "lsicC" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "lsicC"
      (fun j => if SCmp.ge.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 192 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "lsicC" j
      = (if SCmp.ge.run (st.regs "matExtra" j) (UInt64.ofNat 255) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 192 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 191 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hlc l, hlc 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "matExtra" 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    exact setp_ge_iff (st.regs "matExtra" 0) 255 (by decide)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at132 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 132)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "pLitBig" "litLen" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "pLitBig"
      (fun j => if SCmp.ge.run (st.regs "litLen" j) (st.get j (SArg.imm 15)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 133 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "pLitBig" j
      = (if SCmp.ge.run (st.regs "litLen" j) (UInt64.ofNat 15) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 133 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 132 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "litLen" 0 (Or.inr (Or.inl rfl))]
    exact setp_ge_iff (st.regs "litLen" 0) 15 (by decide)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at180 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 180)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.setp (.ge) "pMatBig" "mlm" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "pMatBig"
      (fun j => if SCmp.ge.run (st.regs "mlm" j) (st.get j (SArg.imm 15)) then 1
                else 0)).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 181 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hlc : ∀ j : Lane, (sstep K st).regs "pMatBig" j
      = (if SCmp.ge.run (st.regs "mlm" j) (UInt64.ofNat 15) then 1 else 0) := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    have e : tokRem 181 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 180 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro _
    rw [hlc 0, hfr "mlm" 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    exact setp_ge_iff (st.regs "mlm" 0) 15 (by decide)

/-- A wrap-free `UInt64` subtraction, given that the subtrahend fits. -/
theorem uint64_sub_toNat (a : UInt64) (V : Nat) (hV : (UInt64.ofNat V).toNat = V)
    (h : V ≤ a.toNat) : (a - UInt64.ofNat V).toNat = a.toNat - V := by
  rw [UInt64.toNat_sub, hV,
    show 2 ^ 64 - V + a.toNat = 2 ^ 64 + (a.toNat - V) from by
      have := a.toNat_lt; have := (UInt64.ofNat V).toNat_lt; omega,
    Nat.add_mod_left, Nat.mod_eq_of_lt (by have := a.toNat_lt; omega)]

theorem tokRem_sub142 (a b c d : Nat) (h : 255 ≤ b) :
    tokRem 143 a (b - 255) c d ≤ tokRem 142 a b c d := by
  have e1 : tokRem 142 a b c d = b / 255 + a + 2 + lsicLen c := rfl
  have e2 : tokRem 143 a (b - 255) c d = (b - 255) / 255 + 1 + a + 2 + lsicLen c := rfl
  have hd := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) h
  omega

theorem tokRem_sub190 (a b c d : Nat) (h : 255 ≤ d) :
    tokRem 191 a b c (d - 255) ≤ tokRem 190 a b c d := by
  have e1 : tokRem 190 a b c d = d / 255 := rfl
  have e2 : tokRem 191 a b c (d - 255) = (d - 255) / 255 + 1 := rfl
  have hd := Nat.div_eq_sub_div (Nat.zero_lt_succ 254) h
  omega

theorem tokRem_sub134 (a b c d : Nat) (h : 15 ≤ a) :
    tokRem 135 a (a - 15) c d ≤ tokRem 134 a b c d := by
  have e1 : tokRem 134 a b c d = lsicLen a + a + 2 + lsicLen c := rfl
  have e2 : tokRem 135 a (a - 15) c d = (a - 15) / 255 + 1 + a + 2 + lsicLen c := rfl
  have e3 : lsicLen a = (a - 15) / 255 + 1 := by simp only [lsicLen, if_pos h]
  omega

theorem tokRem_sub182 (a b c d : Nat) (h : 15 ≤ c) :
    tokRem 183 a b c (c - 15) ≤ tokRem 182 a b c d := by
  have e1 : tokRem 182 a b c d = lsicLen c := rfl
  have e2 : tokRem 183 a b c (c - 15) = (c - 15) / 255 + 1 := rfl
  have e3 : lsicLen c = (c - 15) / 255 + 1 := by simp only [lsicLen, if_pos h]
  omega

theorem tok_at134 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 134)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.sub) "litExtra" "litLen" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "litExtra"
      (fun j => SOp.sub.run (st.regs "litLen" j) (st.get j (SArg.imm 15)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 135 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K st).regs "litExtra" j
      = st.regs "litLen" j - UInt64.ofNat 15 := by
    intro j; rw [hstep]; rfl
  have hguard : 15 ≤ (st.regs "litLen" l).toNat := c5 he
  have hDN : ((sstep K st).regs "litExtra" l).toNat = (st.regs "litLen" l).toNat - 15 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub134 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hD l, hD 0, c2 "litLen" (Or.inr (Or.inl rfl))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at142 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 142)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.sub) "litExtra" "litExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "litExtra"
      (fun j => SOp.sub.run (st.regs "litExtra" j) (st.get j (SArg.imm 255)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 143 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K st).regs "litExtra" j
      = st.regs "litExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hguard : 255 ≤ (st.regs "litExtra" l).toNat := c4 (by rw [he]; omega) (by rw [he]; omega)
  have hDN : ((sstep K st).regs "litExtra" l).toNat = (st.regs "litExtra" l).toNat - 255 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))), hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub142 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hD l, hD 0, c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at182 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 182)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.sub) "matExtra" "mlm" (SArg.imm 15)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "matExtra"
      (fun j => SOp.sub.run (st.regs "mlm" j) (st.get j (SArg.imm 15)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 183 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K st).regs "matExtra" j
      = st.regs "mlm" j - UInt64.ofNat 15 := by
    intro j; rw [hstep]; rfl
  have hguard : 15 ≤ (st.regs "mlm" l).toNat := c9 he
  have hDN : ((sstep K st).regs "matExtra" l).toNat = (st.regs "mlm" l).toNat - 15 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub182 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hD l, hD 0, c2 "mlm" (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

theorem tok_at190 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 190)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.sub) "matExtra" "matExtra" (SArg.imm 255)) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "matExtra"
      (fun j => SOp.sub.run (st.regs "matExtra" j) (st.get j (SArg.imm 255)))).setPc (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 191 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hD : ∀ j : Lane, (sstep K st).regs "matExtra" j
      = st.regs "matExtra" j - UInt64.ofNat 255 := by
    intro j; rw [hstep]; rfl
  have hguard : 255 ≤ (st.regs "matExtra" l).toNat := c8 (by rw [he]; omega) (by rw [he]; omega)
  have hDN : ((sstep K st).regs "matExtra" l).toNat = (st.regs "matExtra" l).toNat - 255 := by
    rw [hD l]; exact uint64_sub_toNat _ _ (by decide) hguard
  rw [he] at c1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)), hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hDN]
    exact Nat.le_trans (Nat.add_le_add_left (tokRem_sub190 _ _ _ _ hguard) _) c1
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
      exact c2 _ (Or.inl rfl)
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hD l, hD 0, c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

/-- pc 170: `op += litLen`, the literal run the copy loop has just written.  The
    only cursor move that is not by one; no-wrap still comes from the potential,
    since `op + litLen + 2 + |encNib mlm| ≤ B < 2 ^ 64`. -/
theorem tok_at170 (l : Lane) (B : Nat) (st : SState) (hB : B < 2 ^ 64) (he : st.pc = 170)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.bin (.add) "op" "op" (SArg.reg "litLen")) := by rw [he]; decide
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
  have hstep : sstep K st = (st.setReg "op"
      (fun j => SOp.add.run (st.regs "op" j) (st.get j (SArg.reg "litLen")))).setPc
      (st.pc + 1) := by
    rw [sstep, hp]; rfl
  have hpc' : (sstep K st).pc = 171 := by rw [hstep, he]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j hr; rw [hstep]; rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have hop : ∀ j : Lane, (sstep K st).regs "op" j = st.regs "op" j + st.regs "litLen" j := by
    intro j; rw [hstep]; rfl
  rw [he] at c1
  have e170 : tokRem 170 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
      (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = (st.regs "litLen" l).toNat + 2 + lsicLen (st.regs "mlm" l).toNat := rfl
  have e171 : tokRem 171 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat
      (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat
      = 2 + lsicLen (st.regs "mlm" l).toNat := rfl
  have hopN : ((st.regs "op" l) + (st.regs "litLen" l)).toNat
      = (st.regs "op" l).toNat + (st.regs "litLen" l).toNat := by
    rw [UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc', hop l, hopN, hfr "litLen" l (Or.inl rfl), hfr "litExtra" l (Or.inr (Or.inl rfl)),
      hfr "mlm" l (Or.inr (Or.inr (Or.inl rfl))),
      hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hop l, hop 0, c2 "op" (Or.inl rfl), c2 "litLen" (Or.inr (Or.inl rfl))]
    · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]; exact c2 _ (Or.inr (Or.inl rfl))
    · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
      exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))),
        hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro ha hb; omega
  · rw [hpc']; intro hq; exact absurd hq (by omega)
  · rw [hpc']; intro hq; exact absurd hq (by omega)

/-- pc 157: the literal copy loop's guard.  Both targets are inside the copy
    region, where the potential is constant, so both sides are frame steps. -/
theorem tok_at157 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 157)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "cpCont" "Cx11") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "cpCont" 0 == 1 then st.pc + 1 else sfindLabel K "Cx11") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm"
      ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "cpCont" 0 == 1) = true
  · exact tok_frame l B st 158 (by rw [hstep, he, if_pos hg]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h
  · exact tok_frame l B st 169
      (by rw [hstep, if_neg hg, show sfindLabel K "Cx11" = 169 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h


theorem tokRem_le150 (a b c d : Nat) : tokRem 150 a b c d ≤ tokRem 133 a b c d := by
  have e1 : tokRem 133 a b c d = lsicLen a + a + 2 + lsicLen c := rfl
  have e2 : tokRem 150 a b c d = a + 2 + lsicLen c := rfl
  omega

theorem tokRem_le198 (a b c d : Nat) : tokRem 198 a b c d ≤ tokRem 181 a b c d := by
  have e1 : tokRem 181 a b c d = lsicLen c := rfl
  have e2 : tokRem 198 a b c d = 0 := rfl
  omega


theorem tok_at133 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 133)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "pLitBig" "Le6") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "pLitBig" 0 == 1 then st.pc + 1 else sfindLabel K "Le6") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "pLitBig" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 15 ≤ (st.regs "litLen" l).toNat := by
      rw [c2 "litLen" (Or.inr (Or.inl rfl))]; exact (c6 he).mp hg
    have hpc' : (sstep K st).pc = 134 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 134 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 133 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro _; rw [hfr "litLen" l (Or.inr (Or.inl rfl))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame l B st 150
      (by rw [hstep, if_neg hg, show sfindLabel K "Le6" = 150 from by decide]; rfl) hfr
      (by rw [he]; exact tokRem_le150) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at137 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 137)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "lsicC" "Lx9") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K "Lx9") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 255 ≤ (st.regs "litExtra" l).toNat := by
      rw [c2 "litExtra" (Or.inr (Or.inr (Or.inl rfl)))]; exact (c3 (Or.inr (Or.inl he))).mp hg
    have hpc' : (sstep K st).pc = 138 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 138 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 137 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ _; rw [hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl)))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame l B st 145
      (by rw [hstep, if_neg hg, show sfindLabel K "Lx9" = 145 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at181 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 181)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "pMatBig" "Le12") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "pMatBig" 0 == 1 then st.pc + 1 else sfindLabel K "Le12") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "pMatBig" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 15 ≤ (st.regs "mlm" l).toNat := by
      rw [c2 "mlm" (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact (c10 he).mp hg
    have hpc' : (sstep K st).pc = 182 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 182 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 181 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro _; rw [hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame l B st 198
      (by rw [hstep, if_neg hg, show sfindLabel K "Le12" = 198 from by decide]; rfl) hfr
      (by rw [he]; exact tokRem_le198) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h

theorem tok_at185 (l : Lane) (B : Nat) (st : SState) (he : st.pc = 185)
    (h : TokInv l B st) : TokInv l B (sstep K st) := by
  have hp : K[st.pc]? = some (.braifnot "lsicC" "Lx15") := by rw [he]; decide
  have hstep : sstep K st
      = st.setPc (if st.regs "lsicC" 0 == 1 then st.pc + 1 else sfindLabel K "Lx15") := by
    rw [sstep, hp]; rfl
  have hfr : ∀ (r : String) (j : Lane), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "mlm" ∨ r = "matExtra" ∨ r = "lsicC" ∨ r = "pLitBig" ∨ r = "pMatBig" →
      (sstep K st).regs r j = st.regs r j := by
    intro r j _; rw [hstep]; rfl
  by_cases hg : (st.regs "lsicC" 0 == 1) = true
  · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := h
    have hg2 : 255 ≤ (st.regs "matExtra" l).toNat := by
      rw [c2 "matExtra" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact (c7 (Or.inr (Or.inl he))).mp hg
    have hpc' : (sstep K st).pc = 186 := by rw [hstep, he, if_pos hg]; rfl
    rw [he] at c1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpc', hfr "op" l (Or.inl rfl), hfr "litLen" l (Or.inr (Or.inl rfl)),
        hfr "litExtra" l (Or.inr (Or.inr (Or.inl rfl))), hfr "mlm" l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))),
        hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
      have e : tokRem 186 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat = tokRem 185 (st.regs "litLen" l).toNat (st.regs "litExtra" l).toNat (st.regs "mlm" l).toNat (st.regs "matExtra" l).toNat := rfl
      omega
    · intro r hr
      rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
      · rw [hfr _ l (Or.inl rfl), hfr _ 0 (Or.inl rfl)]
        exact c2 _ (Or.inl rfl)
      · rw [hfr _ l (Or.inr (Or.inl rfl)), hfr _ 0 (Or.inr (Or.inl rfl))]
        exact c2 _ (Or.inr (Or.inl rfl))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inl rfl))), hfr _ 0 (Or.inr (Or.inr (Or.inl rfl)))]
        exact c2 _ (Or.inr (Or.inr (Or.inl rfl)))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inl rfl)))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inl rfl))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      · rw [hfr _ l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))), hfr _ 0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))]
        exact c2 _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (rfl))))))
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro ha hb; omega
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro _ _; rw [hfr "matExtra" l (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]; exact hg2
    · rw [hpc']; intro hq; exact absurd hq (by omega)
    · rw [hpc']; intro hq; exact absurd hq (by omega)
  · exact tok_frame l B st 193
      (by rw [hstep, if_neg hg, show sfindLabel K "Lx15" = 193 from by decide]; rfl) hfr
      (by rw [he]; intro a b c d; exact Nat.le_of_eq rfl) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega) (by rw [he]; omega)
      (by rw [he]; omega) (by rw [he]; omega) h


/-- **The token emit preserves its invariant, at every one of its sixty-eight
    program points.**  Forty-five are frame steps, seven move the cursor by one,
    four shrink a length field, six establish a guard for their successor, five
    are branches, and one is the literal-run cursor jump. -/
theorem tokS_hstep (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (hs : st.pc ∈ tokS) (hex : st.pc ∉ [197, 198]) (h : TokInv l B st) :
    TokInv l B (sstep K st) := by
  simp only [tokS, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e | e
  · exact tok_at129 l B st e h
  · exact tok_at130 l B st e h
  · exact tok_at131 l B st hB e h
  · exact tok_at132 l B st e h
  · exact tok_at133 l B st e h
  · exact tok_at134 l B st e h
  · exact tok_at135 l B st e h
  · exact tok_at136 l B st e h
  · exact tok_at137 l B st e h
  · exact tok_at138 l B st e h
  · exact tok_at139 l B st e h
  · exact tok_at140 l B st e h
  · exact tok_at141 l B st hB e h
  · exact tok_at142 l B st e h
  · exact tok_at143 l B st e h
  · exact tok_at144 l B st e h
  · exact tok_at145 l B st e h
  · exact tok_at146 l B st e h
  · exact tok_at147 l B st e h
  · exact tok_at148 l B st hB e h
  · exact tok_at149 l B st e h
  · exact tok_at150 l B st e h
  · exact tok_at151 l B st e h
  · exact tok_at152 l B st e h
  · exact tok_at153 l B st e h
  · exact tok_at154 l B st e h
  · exact tok_at155 l B st e h
  · exact tok_at156 l B st e h
  · exact tok_at157 l B st e h
  · exact tok_at158 l B st e h
  · exact tok_at159 l B st e h
  · exact tok_at160 l B st e h
  · exact tok_at161 l B st e h
  · exact tok_at162 l B st e h
  · exact tok_at163 l B st e h
  · exact tok_at164 l B st e h
  · exact tok_at165 l B st e h
  · exact tok_at166 l B st e h
  · exact tok_at167 l B st e h
  · exact tok_at168 l B st e h
  · exact tok_at169 l B st e h
  · exact tok_at170 l B st hB e h
  · exact tok_at171 l B st e h
  · exact tok_at172 l B st e h
  · exact tok_at173 l B st e h
  · exact tok_at174 l B st hB e h
  · exact tok_at175 l B st e h
  · exact tok_at176 l B st e h
  · exact tok_at177 l B st e h
  · exact tok_at178 l B st e h
  · exact tok_at179 l B st hB e h
  · exact tok_at180 l B st e h
  · exact tok_at181 l B st e h
  · exact tok_at182 l B st e h
  · exact tok_at183 l B st e h
  · exact tok_at184 l B st e h
  · exact tok_at185 l B st e h
  · exact tok_at186 l B st e h
  · exact tok_at187 l B st e h
  · exact tok_at188 l B st e h
  · exact tok_at189 l B st hB e h
  · exact tok_at190 l B st e h
  · exact tok_at191 l B st e h
  · exact tok_at192 l B st e h
  · exact tok_at193 l B st e h
  · exact tok_at194 l B st e h
  · exact tok_at195 l B st e h
  · exact tok_at196 l B st hB e h
  · exact absurd (by simp [e]) hex
  · exact absurd (by simp [e]) hex

/-- **All seven `sbAddr` stores of the token emit are confined.**  From a state
    at the emit entry (pc 129) satisfying the invariant, the output cursor stays
    below `B` at every visit to any of them, for the whole run. -/
theorem tok_op_lt (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (st : SState)
    (h0 : st.pc = 129) (h : TokInv l B st) (k : Nat)
    (hne : ∀ j, j < k → (siter K j st).pc ∉ [197, 198])
    (hq : (siter K k st).pc = 130 ∨ (siter K k st).pc = 140 ∨ (siter K k st).pc = 147
      ∨ (siter K k st).pc = 173 ∨ (siter K k st).pc = 178 ∨ (siter K k st).pc = 188
      ∨ (siter K k st).pc = 195) :
    ((siter K k st).regs "op" l).toNat < B :=
  tokInv_op_lt l B _
    (inv_on K (TokInv l B) tokS [197, 198] tokS_closed
      (fun s hsm hexs hh => tokS_hstep l B hB s hsm hexs hh) st (by rw [h0]; decide) h k hne) hq

-- ── The loopC body, as a region ──────────────────────────────────────────────

/-- The `loopC` body and the `bra` that closes it: pcs 42–207.  `uwhileEmit`
    lays the loop out as `40 lbl Lh0; 41 braifnot loopC Lx1; body 42..206;
    207 bra Lh0; 208 lbl Lx1`, so this is `bodyEmit` plus its single exit.

    What it is for: the loop-head checkpoint needs "the body never visits pc 40",
    and that is exactly `40 ∉ loopBodyS` together with the closure below. -/
def loopBodyS : List Nat := ((List.range 166).map (· + 42))

theorem loopBodyS_closed : PcClosed (WP.mk 15).kernel loopBodyS [207] :=
  ivClosed_at (WP.mk 15).kernel 42 166 [207] shipped32_size (by omega) (by decide)

/-- The loop head is not in the body — the fact `allSteps_off_site` consumes. -/
theorem head_not_in_loopBodyS : (40 ∈ loopBodyS) = False := by decide


-- ── The token-emit entry, at the shipped kernel ──────────────────────────────

/-- `MB` — the base of `wEmitMatchSeqEmit` inside the `loopC` body — is pc 124:
    the body starts at 42, `coopWindowEmit` is 51 wide, the `uif found` branch adds
    one, the four found movs four, and the extend `uwhile` twenty-four
    (`coopExtendEmit` 18 + `bin`/`setp` 2, plus the four scaffolding instructions).
    `42 + 51 + 1 + 4 + 24 + 2 = 124`, one below the token region `tokS`. -/
theorem mb_succs : AlgorithmLib.LZ4Simt.succsOf K 124 = [125] := by decide

/-- `hMBtop` at the shipped kernel: 124's only successor is above it. -/
theorem mb_top : ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K 124, 124 < q' := by decide

/-- `hMBno` at the shipped kernel, in the decidable bounded form: nothing in the
    match-sequence range steps back to its own entry. -/
theorem mb_noentry_lt : ∀ q', q' < 203 → 124 ≤ q' →
    124 ∉ AlgorithmLib.LZ4Simt.succsOf K q' := by decide

/-- …and the hypothesis shape the simulation chain asks for. -/
theorem mb_noentry (hi : Nat) (hhi : hi ≤ 202) :
    ∀ q', 124 ≤ q' → q' ≤ hi → 124 ∉ AlgorithmLib.LZ4Simt.succsOf K q' :=
  fun q' h1 h2 => mb_noentry_lt q' (by omega) h1


end Lz4Sites
