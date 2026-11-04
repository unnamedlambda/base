module
public import AlgorithmLib.Host.DevBufs
meta import AlgorithmLib.Host.DevBufs
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Buffers ready for a stream, and the events that carry it

A party *knows every access* the tracker records (`KnowsAll`) when each is at
or below what the party knows: then every buffer is ready for it, to read or to
write. An event's clock *covers* the accesses (`Covers`) when each is at or
below that clock joined with the host's: a stream that waits on the event then
knows every access.

How each step keeps or makes these: an operation on `p` keeps `p` knowing all
(`KnowsAll.op_self`); clocks that only grow keep every party knowing all
(`KnowsAll.grow`); the host waiting for `p` makes the host know all, and the
host knowing all makes every party know all (`KnowsAll.sync`,
`KnowsAll.of_host`); a stream created behind the default stream knows what the
default stream knows (`KnowsAll.learn`); recording an event on a party that
knows all makes it cover the accesses (`Covers.issue`), and waiting on it makes
the waiter know all (`Covers.wait`).
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.DevSpec (wAcc rAcc know Ready Below ClockLe Grows issued Grows.know_le
  know_le_issued host_le_know own_le_know)

/-- Every access the tracker records is one `p` knows of. -/
def KnowsAll (r : Race) (p : Nat) : Prop :=
  ∀ b e, (e ∈ wAcc r b ∨ e ∈ rAcc r b) → e.2 ≤ (know r p).get e.1

/-- An event's clock, joined with the host's, is past every access. -/
def Covers (r : Race) (clk : Clock) : Prop :=
  ∀ b e, (e ∈ wAcc r b ∨ e ∈ rAcc r b) → e.2 ≤ (clk.join (r.clock hostParty)).get e.1

/-- **A party that knows every access finds every buffer ready.** -/
theorem KnowsAll.ready {r : Race} {p : Nat} (h : KnowsAll r p) (b : Nat) (write : Bool) : Ready r p b write :=
  ⟨fun e he => h b e (.inl he), fun _ e he => h b e (.inr he)⟩

/-- **The default stream knows every access** where every access is its own
    and at or below its own clock. -/
theorem KnowsAll.of_seq {r : Race} (hO : OwnBelow r) (hD : DefaultOnly r) : KnowsAll r defaultParty := by
  intro b e he
  have h1 := hO b e he
  rw [hD b e he] at h1 ⊢
  simp only [know, Clock.get_join]
  omega

/-- A tracker that records nothing is known to everyone. -/
theorem KnowsAll.empty (p : Nat) : KnowsAll {} p := by
  intro b e he; simp [wAcc, rAcc] at he

/-- The same accesses, and clocks that only grow, keep a party knowing all. -/
theorem KnowsAll.grow {r r' : Race} {p : Nat} (h : KnowsAll r p) (hw : r'.lastW = r.lastW)
    (hr : r'.reads = r.reads) (hg : Grows r r') : KnowsAll r' p := by
  intro b e he
  have he' : e ∈ wAcc r b ∨ e ∈ rAcc r b := by simpa [wAcc, rAcc, hw, hr] using he
  exact Nat.le_trans (h b e he') (hg.know_le p e.1)

/-- **The host knowing every access, everyone does.** -/
theorem KnowsAll.of_host {r : Race} (h : KnowsAll r hostParty) (q : Nat) : KnowsAll r q := by
  intro b e he
  have := h b e he
  have h2 := host_le_know r q e.1
  simp only [know, Clock.get_join] at this h2 ⊢
  omega

/-- **An operation on `p` keeps `p` knowing every access**: its own access is
    at its own clock. -/
theorem KnowsAll.op_self {r r' : Race} {p : Nat} {rs ws : List Nat} (h : KnowsAll r p)
    (hop : r.op p rs ws = some r') : KnowsAll r' p := by
  have hc : ∀ q, r'.clock q = if q = p then issued r p else r.clock q := fun q => by
    rw [op_clock hop, Race.clock_setClock]; rfl
  have hg : Grows r r' := fun q i => by
    rw [hc q]; split
    · next hq => subst hq; exact Nat.le_trans (own_le_know r q i) (know_le_issued r q i)
    · exact Nat.le_refl _
  intro b e he
  rcases op_accs hop b e he with he | rfl
  · exact Nat.le_trans (h b e he) (hg.know_le p e.1)
  · show (issued r p).get p ≤ (know r' p).get p
    have h1 := own_le_know r' p p
    rw [hc p, if_pos rfl] at h1
    exact h1

/-- **The host waiting for `p`, the host knows what `p` knew.** -/
theorem KnowsAll.sync {r : Race} {p : Nat} (h : KnowsAll r p) : KnowsAll (r.sync p) hostParty := by
  intro b e he
  have := h b e he
  have h1 := HProg.DevSpec.sync_host r p e.1
  have h2 := HProg.DevSpec.sync_grows r p hostParty e.1
  simp only [know, Clock.get_join] at this ⊢
  omega

/-- **A party that learns a clock past what `p` knows knows every access `p`
    knew of.** -/
theorem KnowsAll.learn {r : Race} {p : Nat} (h : KnowsAll r p) (q : Nat) {c : Clock}
    (hc : ClockLe (know r p) (c.join (r.clock hostParty))) : KnowsAll (r.joinInto q c) q := by
  intro b e he
  have := Nat.le_trans (h b e he) (hc e.1)
  have h1 := HProg.DevSpec.joinInto_learns r q c e.1
  have h2 := HProg.DevSpec.joinInto_grows r q c hostParty e.1
  simp only [know, Clock.get_join] at this h1 h2 ⊢
  omega

/-- **An event recorded on a party that knows every access covers them.** -/
theorem Covers.issue {r : Race} {p : Nat} (h : KnowsAll r p) : Covers (r.issue p).1 (r.issue p).2 := by
  intro b e he
  have he' : e ∈ wAcc r b ∨ e ∈ rAcc r b := by simpa [Race.issue, Race.setClock, wAcc, rAcc] using he
  have := Nat.le_trans (h b e he') (know_le_issued r p e.1)
  show e.2 ≤ ((issued r p).join _).get e.1
  rw [Clock.get_join]; exact Nat.le_trans this (Nat.le_max_left _ _)

/-- The same accesses, and a host clock that only grows, keep an event's clock
    covering them. -/
theorem Covers.grow {r r' : Race} {clk : Clock} (h : Covers r clk) (hw : r'.lastW = r.lastW)
    (hr : r'.reads = r.reads) (hg : Grows r r') : Covers r' clk := by
  intro b e he
  have he' : e ∈ wAcc r b ∨ e ∈ rAcc r b := by simpa [wAcc, rAcc, hw, hr] using he
  have := h b e he'
  have h2 := hg hostParty e.1
  simp only [Clock.get_join] at this ⊢
  omega

/-- **A party that waits on an event covering every access knows every access.** -/
theorem Covers.wait {r : Race} {clk : Clock} (h : Covers r clk) (p : Nat) :
    KnowsAll (((r.issue p).1).joinInto p clk) p := by
  intro b e he
  have he' : e ∈ wAcc r b ∨ e ∈ rAcc r b := by
    simpa [Race.issue, Race.setClock, Race.joinInto, wAcc, rAcc] using he
  have := h b e he'
  show e.2 ≤ (know ((r.setClock p (issued r p)).joinInto p clk) p).get e.1
  have h1 := HProg.DevSpec.joinInto_learns (r.setClock p (issued r p)) p clk e.1
  have h2 := HProg.DevSpec.joinInto_grows (r.setClock p (issued r p)) p clk hostParty e.1
  have h3 := HProg.DevSpec.setClock_issued_grows r p hostParty e.1
  simp only [know, Clock.get_join] at this h1 h2 h3 ⊢
  omega

-- ---------------------------------------------------------------------------
-- What each call does to streams, events and the tracker
-- ---------------------------------------------------------------------------

/-- A device whose streams and events are as they were, whose tracker records
    the same accesses, and whose clocks only grew. -/
def DevGrows (d d' : Dev) : Prop :=
  d'.streams = d.streams ∧ d'.events = d.events ∧ d'.race.lastW = d.race.lastW ∧
    d'.race.reads = d.race.reads ∧ Grows d.race d'.race

theorem DevGrows.refl (d : Dev) : DevGrows d d := ⟨rfl, rfl, rfl, rfl, Grows.refl _⟩

theorem devOnly_dev {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) : ∃ d, k = some (r, d) ∧ w'.dev = d := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hkd, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  exact ⟨d, hkd, rfl⟩

theorem putAll_streams : ∀ (ws : List (Nat × ByteArray)) (d : Dev),
    (ws.foldl (fun d (x : Nat × ByteArray) => d.put x.1 (some x.2)) d).streams = d.streams
  | [], _ => rfl
  | p :: ws, d => putAll_streams ws (d.put p.1 (some p.2))

theorem runLaunch_streams {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} {l : Launch}
    {ids : List Nat} (h : d.runLaunch k l ids = some d') : d'.streams = d.streams := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    exact putAll_streams _ d

/-- A launch on `p`: the device as it was, or one operation of `p` over the
    same streams and events. -/
def LaunchStep (p : Nat) (d d' : Dev) : Prop :=
  d' = d ∨ ∃ ns, d.race.op p [] ns = some d'.race ∧ d'.streams = d.streams ∧ d'.events = d.events

/-- A synchronous copy: the device as it was, or one operation of the default
    stream that the host then waits for, over the same streams and events. -/
def CopyStep (d d' : Dev) : Prop :=
  d' = d ∨ ∃ rs ws r1, d.race.op defaultParty rs ws = some r1 ∧ d'.race = r1.sync defaultParty ∧
    d'.streams = d.streams ∧ d'.events = d.events

theorem syncWrite_dev {d d' : Dev} {id : Nat} {b : ByteArray} (h : d.syncWrite id b = some d') : CopyStep d d' := by
  unfold Dev.syncWrite at h
  obtain ⟨r1, hr1, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact .inr ⟨_, _, r1, hr1, rfl, rfl, rfl⟩

theorem syncRead_dev {d d' : Dev} {id : Nat} (h : d.syncRead id = some d') : CopyStep d d' := by
  unfold Dev.syncRead at h
  obtain ⟨r1, hr1, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact .inr ⟨_, _, r1, hr1, rfl, rfl, rfl⟩

end AlgorithmLib.Device
