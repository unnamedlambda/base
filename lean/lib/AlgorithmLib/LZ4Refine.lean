import AlgorithmLib.LZ4Imp

namespace AlgorithmLib.LZ4Imp
open AlgorithmLib

-- ── Fuel monotonicity ─────────────────────────────────────────────────────────

theorem eval_mono (g : Nat) : ∀ (f : Nat) (s : Stmt) (st st' : St),
    eval f s st = some st' → f ≤ g → eval g s st = some st' := by
  induction g with
  | zero =>
    intro f s st st' h hf
    have hf0 : f = 0 := by omega
    subst hf0
    exact h
  | succ g ihg =>
    intro f s st st'
    induction s generalizing f st st' with
    | mov d src => intro h _; simp only [eval] at h ⊢; exact h
    | binop o d a b => intro h _; simp only [eval] at h ⊢; exact h
    | ldIn d a => intro h _; simp only [eval] at h ⊢; exact h
    | ldOut d a => intro h _; simp only [eval] at h ⊢; exact h
    | stOut a s => intro h _; simp only [eval] at h ⊢; exact h
    | skip => intro h _; simp only [eval] at h ⊢; exact h
    | seq s t ihs iht =>
      intro h hf
      simp only [eval] at h ⊢
      cases hx : eval f s st with
      | none => rw [hx] at h; simp at h
      | some mid =>
        rw [hx, Option.bind_some] at h
        rw [ihs f st mid hx hf, Option.bind_some]
        exact iht f mid st' h hf
    | ite c a b t e iht ihe =>
      intro h hf
      simp only [eval] at h ⊢
      by_cases hc : Cmp.eval c (st.vars a) (st.get b)
      · rw [if_pos hc] at h ⊢; exact iht f st st' h hf
      · rw [if_neg hc] at h ⊢; exact ihe f st st' h hf
    | whileLoop c a b body ihb =>
      intro h hf
      by_cases hc : Cmp.eval c (st.vars a) (st.get b)
      · cases f with
        | zero => simp [eval, hc] at h
        | succ f =>
          simp only [eval, if_pos hc] at h ⊢
          cases hx : eval (f + 1) body st with
          | none => rw [hx] at h; simp at h
          | some mid =>
            rw [hx, Option.bind_some] at h
            rw [ihb (f + 1) st mid hx (by omega), Option.bind_some]
            exact ihg f _ mid st' h (by omega)
      · cases f with
        | zero => simp only [eval, if_neg hc] at h ⊢; exact h
        | succ f => simp only [eval, if_neg hc] at h ⊢; exact h
    | coop k s ihs =>
      intro h hf; simp only [eval] at h ⊢; exact ihs f st st' h hf

private theorem getD_append_left : ∀ (l r : List UInt8) (k : Nat), k < l.length →
    (l ++ r).getD k 0 = l.getD k 0
  | [], _, k, hk => by simp at hk
  | _ :: _, _, 0, _ => rfl
  | _ :: l, r, k + 1, hk => by
    simp only [List.cons_append, List.getD_cons_succ]
    exact getD_append_left l r k (by simpa using hk)

private theorem getD_append_at : ∀ (pre rest : List UInt8) (d : UInt8),
    (pre ++ rest).getD pre.length d = rest.getD 0 d
  | [], rest, d => rfl
  | a :: pre, rest, d => by simpa using getD_append_at pre rest d

private theorem getD_middle : ∀ (pre mid suf : List UInt8) (k : Nat), k < mid.length →
    (pre ++ (mid ++ suf)).getD (pre.length + k) 0 = mid.getD k 0
  | [], mid, suf, k, hk => by simpa using getD_append_left mid suf k hk
  | a :: pre, mid, suf, k, hk => by
    simp only [List.cons_append, List.length_cons]
    rw [show pre.length + 1 + k = (pre.length + k) + 1 from by omega]
    simp only [List.getD_cons_succ]
    exact getD_middle pre mid suf k hk

private theorem set_at_end : ∀ (o : List UInt8) {n : Nat}, 0 < n → ∀ (x : UInt8),
    (o ++ List.replicate n (0 : UInt8)).set o.length x
      = (o ++ [x]) ++ List.replicate (n - 1) (0 : UInt8)
  | [], n, hn, x => by
    obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
    simp [List.replicate_succ]
  | b :: o, n, hn, x => by
    simp only [List.cons_append, List.length_cons, List.set_cons_succ]
    rw [set_at_end o hn x]

private theorem toNat_byte {n : Nat} (h : n < 256) : (UInt8.ofNat n).toNat = n := by
  simp
  omega

-- ── State helpers ─────────────────────────────────────────────────────────────

theorem St.ext {a b : St} (h1 : a.inp = b.inp) (h2 : a.out = b.out)
    (h3 : ∀ v, a.vars v = b.vars v) : a = b := by
  cases a; cases b
  simp only [St.mk.injEq] at *
  exact ⟨h1, h2, funext h3⟩

@[simp] theorem set_out (st : St) (x n) : (st.set x n).out = st.out := rfl
@[simp] theorem set_vars_self (st : St) (x n) : (st.set x n).vars x = n := by
  simp [St.set]
theorem set_vars_ne (st : St) {x y : Var} (n) (h : y ≠ x) :
    (st.set x n).vars y = st.vars y := by
  simp [St.set, h]

-- ── Spec length lemmas ────────────────────────────────────────────────────────

theorem copyMatch_length (off : Nat) : ∀ (n : Nat) (o : List UInt8),
    (LZ4.copyMatch o off n).length = o.length + n
  | 0, o => rfl
  | n + 1, o => by
    rw [show LZ4.copyMatch o off (n + 1)
          = LZ4.copyMatch (o ++ [o.getD (o.length - off) 0]) off n from rfl,
        copyMatch_length off n]
    simp only [List.length_append, List.length_cons, List.length_nil]
    omega

theorem expandSeq_length (o : List UInt8) (s : LZ4.Seq) :
    (LZ4.expandSeq o s).length = o.length + s.lits.length + s.mlen := by
  rw [show LZ4.expandSeq o s = LZ4.copyMatch (o ++ s.lits) s.offset s.mlen from rfl,
      copyMatch_length]
  simp only [List.length_append]

theorem foldl_expand_ge (seqs : List LZ4.Seq) : ∀ (o : List UInt8),
    o.length ≤ (seqs.foldl LZ4.expandSeq o).length := by
  induction seqs with
  | nil => intro o; simp
  | cons s ss ih =>
    intro o
    rw [List.foldl_cons]
    have := ih (LZ4.expandSeq o s)
    rw [expandSeq_length] at this
    omega

-- ── Loop-body step functions ──────────────────────────────────────────────────
-- Each loop body is straight-line; its effect is a total function on states,
-- computed once and reused by the loop inductions.

def litStep (st : St) : St :=
  { inp := st.inp,
    out := st.out.set (st.vars vOP) (st.inp.getD (st.vars vIP) 0),
    vars := fun v =>
      if v = vLEN then st.vars vLEN - 1
      else if v = vOP then st.vars vOP + 1
      else if v = vIP then st.vars vIP + 1
      else if v = vB then (st.inp.getD (st.vars vIP) 0).toNat
      else st.vars v }

def mcopyStep (st : St) : St :=
  { inp := st.inp,
    out := st.out.set (st.vars vOP) (st.out.getD (st.vars vSRC) 0),
    vars := fun v =>
      if v = vLEN then st.vars vLEN - 1
      else if v = vOP then st.vars vOP + 1
      else if v = vSRC then st.vars vSRC + 1
      else if v = vB then (st.out.getD (st.vars vSRC) 0).toNat
      else st.vars v }

def extStep (st : St) : St :=
  { inp := st.inp,
    out := st.out,
    vars := fun v =>
      if v = vLEN then st.vars vLEN + (st.inp.getD (st.vars vIP) 0).toNat
      else if v = vIP then st.vars vIP + 1
      else if v = vB then (st.inp.getD (st.vars vIP) 0).toNat
      else st.vars v }

theorem litStep_inp (st : St) : (litStep st).inp = st.inp := rfl
theorem litStep_out (st : St) :
    (litStep st).out = st.out.set (st.vars vOP) (st.inp.getD (st.vars vIP) 0) := rfl
theorem litStep_ip (st : St) : (litStep st).vars vIP = st.vars vIP + 1 := by
  simp [litStep, vIP, vOP, vLEN]
theorem litStep_op (st : St) : (litStep st).vars vOP = st.vars vOP + 1 := by
  simp [litStep, vOP, vLEN]
theorem litStep_len (st : St) : (litStep st).vars vLEN = st.vars vLEN - 1 := by
  simp [litStep, vLEN]
theorem litStep_other (st : St) {v : Var} (h0 : v ≠ vIP) (h1 : v ≠ vOP)
    (h4 : v ≠ vLEN) (h5 : v ≠ vB) : (litStep st).vars v = st.vars v := by
  simp only [litStep, vIP, vOP, vLEN, vB] at h0 h1 h4 h5 ⊢
  simp [h0, h1, h4, h5]

theorem mcopyStep_out (st : St) :
    (mcopyStep st).out = st.out.set (st.vars vOP) (st.out.getD (st.vars vSRC) 0) := rfl
theorem mcopyStep_src (st : St) : (mcopyStep st).vars vSRC = st.vars vSRC + 1 := by
  simp [mcopyStep, vSRC, vOP, vLEN]
theorem mcopyStep_op (st : St) : (mcopyStep st).vars vOP = st.vars vOP + 1 := by
  simp [mcopyStep, vOP, vLEN]
theorem mcopyStep_len (st : St) : (mcopyStep st).vars vLEN = st.vars vLEN - 1 := by
  simp [mcopyStep, vLEN]
theorem mcopyStep_other (st : St) {v : Var} (h7 : v ≠ vSRC) (h1 : v ≠ vOP)
    (h4 : v ≠ vLEN) (h5 : v ≠ vB) : (mcopyStep st).vars v = st.vars v := by
  simp only [mcopyStep, vSRC, vOP, vLEN, vB] at h7 h1 h4 h5 ⊢
  simp [h7, h1, h4, h5]

theorem extStep_inp (st : St) : (extStep st).inp = st.inp := rfl
theorem extStep_ip (st : St) : (extStep st).vars vIP = st.vars vIP + 1 := by
  simp [extStep, vIP, vLEN]
theorem extStep_len (st : St) :
    (extStep st).vars vLEN = st.vars vLEN + (st.inp.getD (st.vars vIP) 0).toNat := by
  simp [extStep, vLEN]
theorem extStep_b (st : St) :
    (extStep st).vars vB = (st.inp.getD (st.vars vIP) 0).toNat := by
  simp [extStep, vIP, vLEN, vB]
theorem extStep_other (st : St) {v : Var} (h0 : v ≠ vIP) (h4 : v ≠ vLEN)
    (h5 : v ≠ vB) : (extStep st).vars v = st.vars v := by
  simp only [extStep, vIP, vLEN, vB] at h0 h4 h5 ⊢
  simp [h0, h4, h5]

theorem litCopyBody_eval (f : Nat) (st : St) :
    eval f litCopyBody st = some (litStep st) := by
  simp only [litCopyBody, block, List.foldr_cons, List.foldr_nil, eval,
             Option.bind_some, St.get, Op.eval]
  refine congrArg some (St.ext rfl ?_ fun v => ?_)
  · simp [St.set, litStep, vB, vOP, UInt8.ofNat_toNat]
  · simp only [St.set, litStep, vIP, vOP, vLEN, vB]
    by_cases h4 : v = 4 <;> by_cases h1 : v = 1 <;> by_cases h0 : v = 0 <;>
      by_cases h5 : v = 5 <;> simp [h4, h1, h0, h5]

theorem mcopyBody_eval (f : Nat) (st : St) :
    eval f mcopyBody st = some (mcopyStep st) := by
  simp only [mcopyBody, block, List.foldr_cons, List.foldr_nil, eval,
             Option.bind_some, St.get, Op.eval]
  refine congrArg some (St.ext rfl ?_ fun v => ?_)
  · simp [St.set, mcopyStep, vB, vOP, vSRC, UInt8.ofNat_toNat]
  · simp only [St.set, mcopyStep, vOP, vLEN, vB, vSRC]
    by_cases h4 : v = 4 <;> by_cases h1 : v = 1 <;> by_cases h7 : v = 7 <;>
      by_cases h5 : v = 5 <;> simp [h4, h1, h7, h5]

theorem extWhileBody_eval (f : Nat) (st : St) :
    eval f extWhileBody st = some (extStep st) := by
  simp only [extWhileBody, block, List.foldr_cons, List.foldr_nil, eval,
             Option.bind_some, St.get, Op.eval]
  refine congrArg some (St.ext rfl rfl fun v => ?_)
  simp only [St.set, extStep, vIP, vLEN, vB]
  by_cases h4 : v = 4 <;> by_cases h0 : v = 0 <;> by_cases h5 : v = 5 <;>
    simp [h4, h0, h5]

-- ── The literal-copy loop streams bytes input → output ────────────────────────

theorem litCopy_spec : ∀ (xs pre suf o : List UInt8) (cap : Nat) (st : St),
    st.inp = pre ++ (xs ++ suf) →
    st.vars vIP = pre.length →
    st.vars vLEN = xs.length →
    st.vars vOP = o.length →
    st.out = o ++ List.replicate (cap - o.length) 0 →
    o.length + xs.length ≤ cap →
    ∃ st', eval xs.length litCopy st = some st' ∧
      st'.inp = st.inp ∧
      st'.out = (o ++ xs) ++ List.replicate (cap - (o.length + xs.length)) 0 ∧
      st'.vars vIP = pre.length + xs.length ∧
      st'.vars vOP = o.length + xs.length ∧
      st'.vars vLEN = 0 ∧
      ∀ v, v ≠ vIP → v ≠ vOP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v
  | [], pre, suf, o, cap, st => by
    intro hinp hip hlen hop hout hcap
    refine ⟨st, ?_, rfl, by simpa using hout, by simpa using hip,
            by simpa using hop, by simpa using hlen, fun v _ _ _ _ => rfl⟩
    simp [litCopy, eval, hlen, St.get, Cmp.eval]
  | x :: xs, pre, suf, o, cap, st => by
    intro hinp hip hlen hop hout hcap
    have hcap' : o.length + (xs.length + 1) ≤ cap := by simpa using hcap
    have hbyte : st.inp.getD (st.vars vIP) 0 = x := by
      rw [hinp, hip, show pre.length = pre.length + 0 from rfl,
          getD_middle pre (x :: xs) suf 0 (by simp)]
      rfl
    obtain ⟨st', hev, hinp', hout', hip', hop', hlen', hframe'⟩ :=
      litCopy_spec xs (pre ++ [x]) suf (o ++ [x]) cap (litStep st)
        (by rw [litStep_inp, hinp]; simp)
        (by rw [litStep_ip, hip]; simp)
        (by rw [litStep_len, hlen]; simp)
        (by rw [litStep_op, hop]; simp)
        (by rw [litStep_out, hbyte, hout, hop, set_at_end o (by omega) x,
                show (o ++ [x]).length = o.length + 1 from by simp,
                show cap - o.length - 1 = cap - (o.length + 1) from by omega])
        (by simp only [List.length_append, List.length_cons, List.length_nil]; omega)
    refine ⟨st', ?_, hinp', ?_, ?_, ?_, hlen', ?_⟩
    · show eval (x :: xs).length litCopy st = some st'
      have hcond : Cmp.eval .ne (st.vars vLEN) (st.get (.c 0)) = true := by
        simp [hlen, St.get, Cmp.eval]
      simp only [List.length_cons, litCopy, eval]
      rw [if_pos hcond, litCopyBody_eval, Option.bind_some]
      exact hev
    · rw [hout', show (o ++ [x]) ++ xs = o ++ (x :: xs) from by simp,
          show (o ++ [x]).length + xs.length = o.length + (x :: xs).length from by
            simp only [List.length_append, List.length_cons, List.length_nil]; omega]
    · rw [hip']
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega
    · rw [hop']
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega
    · intro v h0 h1 h4 h5
      rw [hframe' v h0 h1 h4 h5, litStep_other st h0 h1 h4 h5]

-- ── The match-copy loop is the spec's `copyMatch` ─────────────────────────────

theorem mcopy_spec : ∀ (m : Nat) (o : List UInt8) (off cap : Nat) (st : St),
    1 ≤ off → off ≤ o.length →
    st.vars vSRC = o.length - off →
    st.vars vOP = o.length →
    st.vars vLEN = m →
    st.out = o ++ List.replicate (cap - o.length) 0 →
    o.length + m ≤ cap →
    ∃ st', eval m mcopyLoop st = some st' ∧
      st'.inp = st.inp ∧
      st'.out = LZ4.copyMatch o off m
                  ++ List.replicate (cap - (o.length + m)) 0 ∧
      st'.vars vOP = o.length + m ∧
      st'.vars vLEN = 0 ∧
      ∀ v, v ≠ vSRC → v ≠ vOP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v
  | 0, o, off, cap, st => by
    intro _ _ _ hop hlen hout hcap
    refine ⟨st, ?_, rfl, by simpa [LZ4.copyMatch] using hout, by simpa using hop,
            hlen, fun v _ _ _ _ => rfl⟩
    simp [mcopyLoop, eval, hlen, St.get, Cmp.eval]
  | m + 1, o, off, cap, st => by
    intro hoff1 hoff2 hsrc hop hlen hout hcap
    have hlt : o.length - off < o.length := by omega
    have hbyte : st.out.getD (st.vars vSRC) 0 = o.getD (o.length - off) 0 := by
      rw [hout, hsrc, getD_append_left o _ _ hlt]
    obtain ⟨st', hev, hinp', hout', hop', hlen', hframe'⟩ :=
      mcopy_spec m (o ++ [o.getD (o.length - off) 0]) off cap (mcopyStep st)
        hoff1
        (by simp only [List.length_append, List.length_cons, List.length_nil]; omega)
        (by rw [mcopyStep_src, hsrc]
            simp only [List.length_append, List.length_cons, List.length_nil]
            omega)
        (by rw [mcopyStep_op, hop]
            simp only [List.length_append, List.length_cons, List.length_nil])
        (by rw [mcopyStep_len, hlen]; omega)
        (by rw [mcopyStep_out, hbyte, hout, hop, set_at_end o (by omega) _,
                show (o ++ [o.getD (o.length - off) 0]).length = o.length + 1 from by simp,
                show cap - o.length - 1 = cap - (o.length + 1) from by omega])
        (by simp only [List.length_append, List.length_cons, List.length_nil]; omega)
    refine ⟨st', ?_, hinp', ?_, ?_, hlen', ?_⟩
    · show eval (m + 1) mcopyLoop st = some st'
      have hcond : Cmp.eval .ne (st.vars vLEN) (st.get (.c 0)) = true := by
        simp [hlen, St.get, Cmp.eval]
      simp only [mcopyLoop, eval]
      rw [if_pos hcond, mcopyBody_eval, Option.bind_some]
      exact hev
    · rw [hout',
          show LZ4.copyMatch (o ++ [o.getD (o.length - off) 0]) off m
            = LZ4.copyMatch o off (m + 1) from rfl,
          show (o ++ [o.getD (o.length - off) 0]).length + m = o.length + (m + 1) from by
            simp only [List.length_append, List.length_cons, List.length_nil]; omega]
    · rw [hop']
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega
    · intro v h7 h1 h4 h5
      rw [hframe' v h7 h1 h4 h5, mcopyStep_other st h7 h1 h4 h5]

-- ── The LSIC extension loop parses `LZ4.ext n` ────────────────────────────────

theorem extWhile_small (n : Nat) (hn : n ≤ 254) (pre suf : List UInt8) (st : St)
    (hinp : st.inp = pre ++ (LZ4.ext n ++ suf))
    (hip : st.vars vIP = pre.length)
    (hvb : st.vars vB = 255) :
    ∃ st', eval 1 extWhile st = some st' ∧
      st'.inp = st.inp ∧ st'.out = st.out ∧
      st'.vars vIP = pre.length + 1 ∧
      st'.vars vLEN = st.vars vLEN + n ∧
      st'.vars vB = n ∧
      ∀ v, v ≠ vIP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v := by
  have hext : LZ4.ext n = [UInt8.ofNat n] := by
    simp [LZ4.ext, Nat.div_eq_of_lt (by omega : n < 255),
          Nat.mod_eq_of_lt (by omega : n < 255)]
  have hbyte : st.inp.getD (st.vars vIP) 0 = UInt8.ofNat n := by
    rw [hinp, hext, hip, show pre.length = pre.length + 0 from rfl,
        getD_middle pre [UInt8.ofNat n] suf 0 (by simp)]
    rfl
  have hbn : (st.inp.getD (st.vars vIP) 0).toNat = n := by
    rw [hbyte]; exact toNat_byte (by omega)
  refine ⟨extStep st, ?_, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · have hcond : Cmp.eval .eq (st.vars vB) (st.get (.c 255)) = true := by
      simp [hvb, St.get, Cmp.eval]
    have hcond' : Cmp.eval .eq ((extStep st).vars vB) ((extStep st).get (.c 255))
        = false := by
      rw [extStep_b, hbn]
      simp [St.get, Cmp.eval]
      omega
    simp only [extWhile, eval]
    rw [if_pos hcond, extWhileBody_eval, Option.bind_some]
    simp only [eval, hcond']
    simp
  · rw [extStep_ip, hip]
  · rw [extStep_len, hbn]
  · rw [extStep_b, hbn]
  · intro v h0 h4 h5
    exact extStep_other st h0 h4 h5

theorem extWhile_spec : ∀ (k n : Nat), n ≤ 255 * k + 254 →
    ∀ (pre suf : List UInt8) (st : St),
    st.inp = pre ++ (LZ4.ext n ++ suf) →
    st.vars vIP = pre.length →
    st.vars vB = 255 →
    ∃ st', eval (n / 255 + 1) extWhile st = some st' ∧
      st'.inp = st.inp ∧ st'.out = st.out ∧
      st'.vars vIP = pre.length + (n / 255 + 1) ∧
      st'.vars vLEN = st.vars vLEN + n ∧
      st'.vars vB = n % 255 ∧
      ∀ v, v ≠ vIP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v
  | 0, n, hn, pre, suf, st => by
    intro hinp hip hvb
    have hd : n / 255 = 0 := Nat.div_eq_of_lt (by omega)
    have hm : n % 255 = n := Nat.mod_eq_of_lt (by omega)
    obtain ⟨st', hev, h1, h2, h3, h4, h5, h6⟩ :=
      extWhile_small n (by omega) pre suf st hinp hip hvb
    exact ⟨st', by rw [hd]; exact hev, h1, h2, by rw [hd]; exact h3, h4,
           by rw [hm]; exact h5, h6⟩
  | k + 1, n, hn, pre, suf, st => by
    intro hinp hip hvb
    by_cases hsmall : n ≤ 254
    · have hd : n / 255 = 0 := Nat.div_eq_of_lt (by omega)
      have hm : n % 255 = n := Nat.mod_eq_of_lt (by omega)
      obtain ⟨st', hev, h1, h2, h3, h4, h5, h6⟩ :=
        extWhile_small n hsmall pre suf st hinp hip hvb
      exact ⟨st', by rw [hd]; exact hev, h1, h2, by rw [hd]; exact h3, h4,
             by rw [hm]; exact h5, h6⟩
    · have hge : 255 ≤ n := by omega
      have hext : LZ4.ext n = (255 : UInt8) :: LZ4.ext (n - 255) := by
        have hd : n / 255 = (n - 255) / 255 + 1 := by omega
        have hm : n % 255 = (n - 255) % 255 := by omega
        simp [LZ4.ext, hd, hm, List.replicate_succ]
      have hbyte : st.inp.getD (st.vars vIP) 0 = (255 : UInt8) := by
        rw [hinp, hext, hip, show pre.length = pre.length + 0 from rfl,
            getD_middle pre ((255 : UInt8) :: LZ4.ext (n - 255)) suf 0 (by simp)]
        rfl
      have hbn : (st.inp.getD (st.vars vIP) 0).toNat = 255 := by
        rw [hbyte]; rfl
      obtain ⟨st', hev, h1, h2, h3, h4, h5, h6⟩ :=
        extWhile_spec k (n - 255) (by omega) (pre ++ [(255 : UInt8)]) suf
          (extStep st)
          (by rw [extStep_inp, hinp, hext]; simp)
          (by rw [extStep_ip, hip]; simp)
          (by rw [extStep_b, hbn])
      have hfuel : n / 255 + 1 = ((n - 255) / 255 + 1) + 1 := by omega
      refine ⟨st', ?_, h1, h2, ?_, ?_, ?_, ?_⟩
      · rw [hfuel]
        have hcond : Cmp.eval .eq (st.vars vB) (st.get (.c 255)) = true := by
          simp [hvb, St.get, Cmp.eval]
        simp only [extWhile, eval]
        rw [if_pos hcond, extWhileBody_eval, Option.bind_some]
        exact hev
      · rw [h3]
        simp only [List.length_append, List.length_cons, List.length_nil]
        omega
      · rw [h4, extStep_len, hbn]
        omega
      · rw [h5]
        omega
      · intro v h0 h4' h5'
        rw [h6 v h0 h4' h5', extStep_other st h0 h4' h5']

theorem extLoop_spec (n : Nat) (pre suf : List UInt8) (st : St)
    (hinp : st.inp = pre ++ (LZ4.ext n ++ suf))
    (hip : st.vars vIP = pre.length) :
    ∃ st', eval (n / 255 + 1) extLoop st = some st' ∧
      st'.inp = st.inp ∧ st'.out = st.out ∧
      st'.vars vIP = pre.length + (n / 255 + 1) ∧
      st'.vars vLEN = st.vars vLEN + n ∧
      st'.vars vB = n % 255 ∧
      ∀ v, v ≠ vIP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v := by
  obtain ⟨st', hev, h1, h2, h3, h4, h5, h6⟩ :=
    extWhile_spec n n (by omega) pre suf (st.set vB 255)
      (by simpa using hinp)
      (by rw [set_vars_ne st 255 (by decide : vIP ≠ vB)]; exact hip)
      (by simp)
  refine ⟨st', ?_, h1, h2, h3, ?_, h5, ?_⟩
  · simp only [extLoop, eval, Option.bind_some, St.get]
    exact hev
  · rw [h4, set_vars_ne st 255 (by decide : vLEN ≠ vB)]
  · intro v h0 h4' h5'
    rw [h6 v h0 h4' h5', set_vars_ne st 255 h5']

theorem eval_seq (f : Nat) (a b : Stmt) (st : St) :
    eval f (a.seq b) st = (eval f a st).bind (fun s => eval f b s) := by
  simp only [eval]

-- Straight-line step evaluators (fuel-agnostic).
theorem eval_ldIn' (f : Nat) (d a : Var) (st : St) :
    eval f (.ldIn d a) st = some (st.set d (st.inp.getD (st.vars a) 0).toNat) := by
  simp only [eval]
theorem eval_binop' (f : Nat) (o : Op) (d a : Var) (b : Operand) (st : St) :
    eval f (.binop o d a b) st = some (st.set d (o.eval (st.vars a) (st.get b))) := by
  simp only [eval]
theorem eval_mov' (f : Nat) (d : Var) (s : Operand) (st : St) :
    eval f (.mov d s) st = some (st.set d (st.get s)) := by
  simp only [eval]

-- ── Length-field encoding shape ───────────────────────────────────────────────

theorem ext_length (m : Nat) : (LZ4.ext m).length = m / 255 + 1 := by
  simp [LZ4.ext]

theorem encNib_lt {n : Nat} (h : n < 15) : LZ4.encNib n = [] := by
  simp [LZ4.encNib, h]

theorem encNib_ge {n : Nat} (h : 15 ≤ n) : LZ4.encNib n = LZ4.ext (n - 15) := by
  simp [LZ4.encNib, Nat.not_lt.mpr h]

theorem encNib_length (n : Nat) :
    (LZ4.encNib n).length = if n < 15 then 0 else (n - 15) / 255 + 1 := by
  by_cases h : n < 15
  · rw [encNib_lt h]; simp [h]
  · rw [encNib_ge (by omega), ext_length]; simp [h]

-- ── The optional length-extension `ite` decodes one `encNib` field ────────────

theorem readExt_spec (n : Nat) (pre suf : List UInt8) (Fw : Nat) (st : St)
    (hinp : st.inp = pre ++ (LZ4.encNib n ++ suf))
    (hip : st.vars vIP = pre.length)
    (hlen : st.vars vLEN = min n 15)
    (hfuel : (n - 15) / 255 + 1 ≤ Fw) :
    ∃ st', eval Fw (.ite .eq vLEN (.c 15) extLoop .skip) st = some st' ∧
      st'.inp = st.inp ∧ st'.out = st.out ∧
      st'.vars vIP = pre.length + (LZ4.encNib n).length ∧
      st'.vars vLEN = n ∧
      ∀ v, v ≠ vIP → v ≠ vLEN → v ≠ vB → st'.vars v = st.vars v := by
  by_cases h : n < 15
  · have hmin : min n 15 = n := by omega
    refine ⟨st, ?_, rfl, rfl, ?_, ?_, fun v _ _ _ => rfl⟩
    · have hcond : Cmp.eval .eq (st.vars vLEN) (st.get (.c 15)) = false := by
        rw [hlen, hmin]; simp [St.get, Cmp.eval]; omega
      simp [eval, hcond]
    · rw [encNib_lt h]; simp [hip]
    · rw [hlen, hmin]
  · have hmin : min n 15 = 15 := by omega
    have hge : 15 ≤ n := by omega
    have henc : LZ4.encNib n = LZ4.ext (n - 15) := encNib_ge hge
    obtain ⟨st', hev, hinp', hout', hip', hlen', hb', hframe'⟩ :=
      extLoop_spec (n - 15) pre suf st (by rw [hinp, henc]) hip
    refine ⟨st', ?_, hinp', hout', ?_, ?_, ?_⟩
    · have hcond : Cmp.eval .eq (st.vars vLEN) (st.get (.c 15)) = true := by
        rw [hlen, hmin]; simp [St.get, Cmp.eval]
      simp only [eval, hcond, if_true]
      exact eval_mono Fw _ extLoop st st' hev hfuel
    · rw [hip', encNib_length, if_neg h]
    · rw [hlen', hlen, hmin]; omega
    · intro v hv hl hb
      exact hframe' v hv hl hb

-- ── Block-level validity for the refinement ───────────────────────────────────
-- Every match offset must point within the output produced so far. This is the
-- decode-side wellformedness the byte-format `Seq.WF` cannot express (it is
-- per-sequence, this is cumulative).

def ValidFrom : List UInt8 → List LZ4.Seq → Prop
  | _, [] => True
  | acc, s :: ss => s.offset ≤ acc.length + s.lits.length ∧ ValidFrom (LZ4.expandSeq acc s) ss

-- ── Loop-header straight-line step (ldIn vTOK; vIP+=1; vLEN = vTOK>>4) ─────────

def hdrState (st : St) : St :=
  { inp := st.inp, out := st.out,
    vars := fun v =>
      if v = vLEN then (st.inp.getD (st.vars vIP) 0).toNat >>> 4
      else if v = vIP then st.vars vIP + 1
      else if v = vTOK then (st.inp.getD (st.vars vIP) 0).toNat
      else st.vars v }

theorem hdr_eval (f : Nat) (rest : Stmt) (st : St) :
    eval f ((Stmt.ldIn vTOK vIP).seq
              ((Stmt.binop .add vIP vIP (.c 1)).seq
                ((Stmt.binop .shr vLEN vTOK (.c 4)).seq rest))) st
      = eval f rest (hdrState st) := by
  rw [eval_seq, eval_ldIn', Option.bind_some,
      eval_seq, eval_binop', Option.bind_some,
      eval_seq, eval_binop', Option.bind_some]
  congr 1

theorem eval_while_stop (g : Nat) (c : Cmp) (a : Var) (b : Operand) (body : Stmt)
    (st : St) (h : Cmp.eval c (st.vars a) (st.get b) = false) :
    eval g (.whileLoop c a b body) st = some st := by
  cases g <;> simp [eval, h]

-- ── matchPart straight-line header (offset decode + match-len nibble) ──────────

def matchHdrState (st : St) : St :=
  { inp := st.inp, out := st.out,
    vars := fun v =>
      if v = vLEN then st.vars vTOK &&& 15
      else if v = vIP then st.vars vIP + 2
      else if v = vOFF then (st.inp.getD (st.vars vIP) 0).toNat
                            + (st.inp.getD (st.vars vIP + 1) 0).toNat <<< 8
      else if v = vB then (st.inp.getD (st.vars vIP + 1) 0).toNat <<< 8
      else if v = vTMP then st.vars vIP + 1
      else st.vars v }

theorem matchHdr_eval (f : Nat) (rest : Stmt) (st : St) :
    eval f ((Stmt.ldIn vOFF vIP).seq
              ((Stmt.binop .add vTMP vIP (.c 1)).seq
                ((Stmt.ldIn vB vTMP).seq
                  ((Stmt.binop .shl vB vB (.c 8)).seq
                    ((Stmt.binop .add vOFF vOFF (.v vB)).seq
                      ((Stmt.binop .add vIP vIP (.c 2)).seq
                        ((Stmt.binop .band vLEN vTOK (.c 15)).seq rest))))))) st
      = eval f rest (matchHdrState st) := by
  rw [eval_seq, eval_ldIn', Option.bind_some, eval_seq, eval_binop', Option.bind_some,
      eval_seq, eval_ldIn', Option.bind_some, eval_seq, eval_binop', Option.bind_some,
      eval_seq, eval_binop', Option.bind_some, eval_seq, eval_binop', Option.bind_some,
      eval_seq, eval_binop', Option.bind_some]
  congr 1
  refine St.ext rfl rfl fun v => ?_
  simp only [matchHdrState, St.get, Op.eval, St.set]
  by_cases h4 : v = vLEN <;> by_cases h0 : v = vIP <;> by_cases h6 : v = vOFF <;>
    by_cases h5 : v = vB <;> by_cases h9 : v = vTMP <;>
    simp_all [vLEN, vIP, vOFF, vB, vTMP, vTOK]

theorem matchHdrState_inp (st : St) : (matchHdrState st).inp = st.inp := rfl
theorem matchHdrState_out (st : St) : (matchHdrState st).out = st.out := rfl
theorem matchHdrState_ip (st : St) : (matchHdrState st).vars vIP = st.vars vIP + 2 := by
  simp [matchHdrState, vIP, vLEN]
theorem matchHdrState_len (st : St) :
    (matchHdrState st).vars vLEN = st.vars vTOK &&& 15 := by
  simp [matchHdrState, vLEN]
theorem matchHdrState_off (st : St) :
    (matchHdrState st).vars vOFF = (st.inp.getD (st.vars vIP) 0).toNat
      + (st.inp.getD (st.vars vIP + 1) 0).toNat <<< 8 := by
  simp [matchHdrState, vOFF, vLEN, vIP]
theorem matchHdrState_other (st : St) {v : Var} (h4 : v ≠ vLEN) (h0 : v ≠ vIP)
    (h6 : v ≠ vOFF) (h5 : v ≠ vB) (h9 : v ≠ vTMP) :
    (matchHdrState st).vars v = st.vars v := by
  simp only [matchHdrState]
  rw [if_neg h4, if_neg h0, if_neg h6, if_neg h5, if_neg h9]

theorem eval_skip (f : Nat) (st : St) : eval f .skip st = some st := by simp only [eval]

theorem eval_ite_false (f : Nat) (c : Cmp) (a : Var) (b : Operand) (t e : Stmt) (st : St)
    (h : Cmp.eval c (st.vars a) (st.get b) = false) : eval f (.ite c a b t e) st = eval f e st := by
  cases f <;> simp [eval, h]

-- matchPart tail: vLEN += 4; vSRC := vOP; vSRC := vSRC - vOFF
def preCopyState (st : St) : St :=
  { inp := st.inp, out := st.out,
    vars := fun v =>
      if v = vSRC then st.vars vOP - st.vars vOFF
      else if v = vLEN then st.vars vLEN + 4
      else st.vars v }

theorem preCopy_eval (f : Nat) (rest : Stmt) (st : St) :
    eval f ((Stmt.binop .add vLEN vLEN (.c 4)).seq
              ((Stmt.mov vSRC (.v vOP)).seq
                ((Stmt.binop .sub vSRC vSRC (.v vOFF)).seq rest))) st
      = eval f rest (preCopyState st) := by
  rw [eval_seq, eval_binop', Option.bind_some, eval_seq, eval_mov', Option.bind_some,
      eval_seq, eval_binop', Option.bind_some]
  congr 1
  refine St.ext rfl rfl fun v => ?_
  simp only [preCopyState, St.get, Op.eval, St.set]
  by_cases h7 : v = vSRC <;> by_cases h4 : v = vLEN <;> by_cases h1 : v = vOP <;>
    by_cases h6 : v = vOFF <;> simp_all [vSRC, vLEN, vOP, vOFF]

theorem preCopyState_inp (st : St) : (preCopyState st).inp = st.inp := rfl
theorem preCopyState_out (st : St) : (preCopyState st).out = st.out := rfl
theorem preCopyState_src (st : St) :
    (preCopyState st).vars vSRC = st.vars vOP - st.vars vOFF := by simp [preCopyState, vSRC]
theorem preCopyState_len (st : St) : (preCopyState st).vars vLEN = st.vars vLEN + 4 := by
  simp [preCopyState, vLEN, vSRC]
theorem preCopyState_other (st : St) {v : Var} (h7 : v ≠ vSRC) (h4 : v ≠ vLEN) :
    (preCopyState st).vars v = st.vars v := by
  simp only [preCopyState]; rw [if_neg h7, if_neg h4]

theorem hdrState_inp (st : St) : (hdrState st).inp = st.inp := rfl
theorem hdrState_out (st : St) : (hdrState st).out = st.out := rfl
theorem hdrState_ip (st : St) : (hdrState st).vars vIP = st.vars vIP + 1 := by
  simp [hdrState, vIP, vLEN]
theorem hdrState_len (st : St) :
    (hdrState st).vars vLEN = (st.inp.getD (st.vars vIP) 0).toNat >>> 4 := by
  simp [hdrState, vLEN]
theorem hdrState_tok (st : St) :
    (hdrState st).vars vTOK = (st.inp.getD (st.vars vIP) 0).toNat := by
  simp [hdrState, vTOK, vLEN, vIP]
theorem hdrState_other (st : St) {v : Var} (h4 : v ≠ vLEN) (h0 : v ≠ vIP)
    (h3 : v ≠ vTOK) : (hdrState st).vars v = st.vars v := by
  simp only [hdrState]; rw [if_neg h4, if_neg h0, if_neg h3]

-- ── The main-loop refinement ──────────────────────────────────────────────────

theorem mainLoop_spec :
    ∀ (seqs : List LZ4.Seq) (final acc pre : List UInt8) (Fw : Nat) (st : St),
      (∀ s ∈ seqs, s.WF) →
      ValidFrom acc seqs →
      ((seqs.foldl LZ4.expandSeq acc) ++ final).length + seqs.length + 2 ≤ Fw →
      st.inp = pre ++ (seqs.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final) →
      st.vars vIP = pre.length →
      st.vars vRUN = 1 →
      st.vars vEND = st.inp.length →
      st.vars vOP = acc.length →
      st.out = acc ++ List.replicate
        (((seqs.foldl LZ4.expandSeq acc) ++ final).length - acc.length) 0 →
      ∃ st', eval Fw decompressProg st = some st' ∧
        st'.out = (seqs.foldl LZ4.expandSeq acc) ++ final ∧
        st'.vars vOP = ((seqs.foldl LZ4.expandSeq acc) ++ final).length := by
  intro seqs
  induction seqs with
  | nil =>
    intro final acc pre Fw st _ _ hfuel hinp hip hrun hend hop hout
    simp only [List.foldl_nil, List.length_nil, Nat.add_zero] at hfuel hout ⊢
    have hcapval : (acc ++ final).length = acc.length + final.length := by simp
    obtain ⟨fw, rfl⟩ : ∃ fw, Fw = fw + 1 := ⟨Fw - 1, by omega⟩
    have hinp2 : st.inp = pre ++ LZ4.encodeFinal final := by simpa using hinp
    have hbyte : st.inp.getD (st.vars vIP) 0 = UInt8.ofNat (min final.length 15 * 16) := by
      rw [hinp2, hip, getD_append_at]; rfl
    have htokN : (st.inp.getD (st.vars vIP) 0).toNat = min final.length 15 * 16 := by
      rw [hbyte]; exact toNat_byte (by omega)
    -- enter the loop and run the header
    rw [decompressProg]
    have hcond1 : Cmp.eval .eq (st.vars vRUN) (st.get (.c 1)) = true := by
      rw [hrun]; simp [St.get, Cmp.eval]
    simp only [eval, hcond1, if_true]
    rw [mainBody]
    simp only [block, List.foldr_cons, List.foldr_nil]
    rw [hdr_eval]
    -- projections of the post-header state
    have h3inp := hdrState_inp st
    have h3out := hdrState_out st
    have h3ip : (hdrState st).vars vIP = pre.length + 1 := by rw [hdrState_ip, hip]
    have h3len : (hdrState st).vars vLEN = min final.length 15 := by
      rw [hdrState_len, htokN, Nat.shiftRight_eq_div_pow, show (2:Nat)^4 = 16 from rfl]; omega
    have h3op : (hdrState st).vars vOP = acc.length := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hop]
    have h3run : (hdrState st).vars vRUN = 1 := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hrun]
    have h3end : (hdrState st).vars vEND = st.inp.length := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hend]
    have hfl_le : final.length ≤ (acc ++ final).length := by rw [hcapval]; omega
    -- readExt for the final-literal length
    obtain ⟨s4, hs4ev, hs4inp, hs4out, hs4ip, hs4len, hs4frame⟩ :=
      readExt_spec final.length (pre ++ [UInt8.ofNat (min final.length 15 * 16)]) final (fw + 1)
        (hdrState st)
        (by rw [h3inp, hinp2]; simp [LZ4.encodeFinal, List.append_assoc])
        (by rw [h3ip]; simp)
        h3len
        (by omega)
    rw [eval_seq, hs4ev, Option.bind_some]
    have hs4op : s4.vars vOP = acc.length := by
      rw [hs4frame vOP (by decide) (by decide) (by decide), h3op]
    have hs4end : s4.vars vEND = st.inp.length := by
      rw [hs4frame vEND (by decide) (by decide) (by decide), h3end]
    -- litCopy of the final literals
    obtain ⟨s5, hs5ev, hs5inp, hs5out, hs5ip, hs5op, hs5len, hs5frame⟩ :=
      litCopy_spec final (pre ++ [UInt8.ofNat (min final.length 15 * 16)] ++ LZ4.encNib final.length)
        [] acc ((acc ++ final).length) s4
        (by rw [hs4inp, h3inp, hinp2]; simp [LZ4.encodeFinal, List.append_assoc])
        (by rw [hs4ip]; simp only [List.length_append])
        hs4len
        (by rw [hs4op])
        (by rw [hs4out, h3out, hout])
        (by omega)
    rw [eval_seq, eval_mono (fw+1) final.length litCopy s4 s5 hs5ev (by omega), Option.bind_some]
    -- the final `ite`: vIP has reached vEND, take the `mov vRUN 0` branch
    have hs5end : s5.vars vEND = st.inp.length := by
      rw [hs5frame vEND (by decide) (by decide) (by decide) (by decide), hs4end]
    have hs5ip_full : s5.vars vIP = st.inp.length := by
      rw [hs5ip, hinp2]
      simp only [List.length_append, List.length_cons, List.length_nil]
      rw [show LZ4.encodeFinal final
            = UInt8.ofNat (min final.length 15 * 16) :: (LZ4.encNib final.length ++ final) from rfl]
      simp only [List.length_cons, List.length_append, encNib_length]
      by_cases h : final.length < 15 <;> simp [h] <;> omega
    have hcond2 : Cmp.eval .ge (s5.vars vIP) (s5.get (.v vEND)) = true := by
      simp [Cmp.eval, St.get, hs5ip_full, hs5end]
    rw [eval_seq]
    simp only [eval, hcond2, if_true, Option.bind_some]
    -- close the outer while (vRUN = 0 now)
    have hcond3 : Cmp.eval .eq ((s5.set vRUN (s5.get (.c 0))).vars vRUN)
        ((s5.set vRUN (s5.get (.c 0))).get (.c 1)) = false := by
      simp [St.get, Cmp.eval]
    refine ⟨s5.set vRUN (s5.get (.c 0)), eval_while_stop fw _ _ _ _ _ hcond3, ?_, ?_⟩
    · rw [set_out, hs5out]; simp [hcapval]
    · rw [set_vars_ne _ _ (by decide : vOP ≠ vRUN), hs5op]; omega
  | cons s ss ih =>
    intro final acc pre Fw st hwf hvalid hfuel hinp hip hrun hend hop hout
    obtain ⟨ho1, ho2, hm4⟩ := hwf s (List.mem_cons_self)
    obtain ⟨hoff_in, hvalid_tl⟩ := hvalid
    obtain ⟨fw, rfl⟩ : ∃ fw, Fw = fw + 1 := ⟨Fw - 1, by
      simp only [List.length_cons] at hfuel; omega⟩
    -- length / fuel facts
    have hLs : (LZ4.expandSeq acc s).length = acc.length + s.lits.length + s.mlen :=
      expandSeq_length acc s
    have hfoldcons : (s :: ss).foldl LZ4.expandSeq acc = ss.foldl LZ4.expandSeq (LZ4.expandSeq acc s) :=
      List.foldl_cons
    have hcap_ge : (LZ4.expandSeq acc s).length ≤ ((s :: ss).foldl LZ4.expandSeq acc ++ final).length := by
      rw [hfoldcons, List.length_append]
      have := foldl_expand_ge ss (LZ4.expandSeq acc s); omega
    have hlencons : (s :: ss).length = ss.length + 1 := List.length_cons
    -- the encoded input, split off `encodeSeq s`
    have hinpS : st.inp = pre ++ (LZ4.encodeSeq s ++
        (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final)) := by
      rw [hinp]; simp [List.flatMap_cons, List.append_assoc]
    have htokN : (st.inp.getD (st.vars vIP) 0).toNat
        = min s.lits.length 15 * 16 + min (s.mlen - 4) 15 := by
      have hb : st.inp.getD (st.vars vIP) 0
          = UInt8.ofNat (min s.lits.length 15 * 16 + min (s.mlen - 4) 15) := by
        rw [hinpS, hip, getD_append_at]; rfl
      rw [hb]; exact toNat_byte (by omega)
    -- header
    rw [decompressProg]
    have hcond1 : Cmp.eval .eq (st.vars vRUN) (st.get (.c 1)) = true := by
      rw [hrun]; simp [St.get, Cmp.eval]
    simp only [eval, hcond1, if_true]
    rw [mainBody]; simp only [block, List.foldr_cons, List.foldr_nil]
    rw [hdr_eval]
    have h3inp := hdrState_inp st
    have h3ip : (hdrState st).vars vIP = pre.length + 1 := by rw [hdrState_ip, hip]
    have h3len : (hdrState st).vars vLEN = min s.lits.length 15 := by
      rw [hdrState_len, htokN, Nat.shiftRight_eq_div_pow, show (2:Nat)^4 = 16 from rfl]; omega
    have h3op : (hdrState st).vars vOP = acc.length := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hop]
    have h3run : (hdrState st).vars vRUN = 1 := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hrun]
    have h3end : (hdrState st).vars vEND = st.inp.length := by
      rw [hdrState_other st (by decide) (by decide) (by decide), hend]
    have h3tok : (hdrState st).vars vTOK
        = min s.lits.length 15 * 16 + min (s.mlen - 4) 15 := by rw [hdrState_tok, htokN]
    -- readExt (literal length)
    obtain ⟨s4, hs4ev, hs4inp, hs4out, hs4ip, hs4len, hs4frame⟩ :=
      readExt_spec s.lits.length (pre ++ [UInt8.ofNat (min s.lits.length 15 * 16 + min (s.mlen - 4) 15)])
        (s.lits ++ ([UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)]
          ++ (LZ4.encNib (s.mlen - 4) ++ (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final))))
        (fw + 1) (hdrState st)
        (by rw [h3inp, hinpS]; simp [LZ4.encodeSeq, List.append_assoc])
        (by rw [h3ip]; simp)
        h3len (by omega)
    rw [eval_seq, hs4ev, Option.bind_some]
    have hs4tok : s4.vars vTOK = min s.lits.length 15 * 16 + min (s.mlen - 4) 15 := by
      rw [hs4frame vTOK (by decide) (by decide) (by decide), h3tok]
    have hs4op : s4.vars vOP = acc.length := by
      rw [hs4frame vOP (by decide) (by decide) (by decide), h3op]
    have hs4run : s4.vars vRUN = 1 := by
      rw [hs4frame vRUN (by decide) (by decide) (by decide), h3run]
    have hs4end : s4.vars vEND = st.inp.length := by
      rw [hs4frame vEND (by decide) (by decide) (by decide), h3end]
    -- litCopy
    obtain ⟨s5, hs5ev, hs5inp, hs5out, hs5ip, hs5op, hs5len, hs5frame⟩ :=
      litCopy_spec s.lits
        (pre ++ [UInt8.ofNat (min s.lits.length 15 * 16 + min (s.mlen - 4) 15)]
          ++ LZ4.encNib s.lits.length)
        ([UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)]
          ++ (LZ4.encNib (s.mlen - 4) ++ (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final)))
        acc ((s :: ss).foldl LZ4.expandSeq acc ++ final).length s4
        (by rw [hs4inp, h3inp, hinpS]; simp [LZ4.encodeSeq, List.append_assoc])
        (by rw [hs4ip]; simp only [List.length_append])
        hs4len (by rw [hs4op])
        (by rw [hs4out, hdrState_out, hout])
        (by rw [hLs] at hcap_ge; omega)
    rw [eval_seq, eval_mono (fw+1) s.lits.length litCopy s4 s5 hs5ev (by rw [hLs] at hcap_ge; omega),
        Option.bind_some]
    have hs5inp_st : s5.inp = st.inp := by rw [hs5inp, hs4inp, h3inp]
    have hs5tok : s5.vars vTOK = min s.lits.length 15 * 16 + min (s.mlen - 4) 15 := by
      rw [hs5frame vTOK (by decide) (by decide) (by decide) (by decide), hs4tok]
    have hs5run : s5.vars vRUN = 1 := by
      rw [hs5frame vRUN (by decide) (by decide) (by decide) (by decide), hs4run]
    have hs5end : s5.vars vEND = st.inp.length := by
      rw [hs5frame vEND (by decide) (by decide) (by decide) (by decide), hs4end]
    -- vIP now points at the offset bytes; decompose st.inp there
    have hAlen : s5.vars vIP = (((pre ++ [UInt8.ofNat (min s.lits.length 15 * 16
        + min (s.mlen - 4) 15)]) ++ LZ4.encNib s.lits.length) ++ s.lits).length := by
      rw [hs5ip]
      simp only [List.length_append, List.length_cons, List.length_nil]
    have hAdecomp : st.inp = (((pre ++ [UInt8.ofNat (min s.lits.length 15 * 16
        + min (s.mlen - 4) 15)]) ++ LZ4.encNib s.lits.length) ++ s.lits)
        ++ ([UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)]
          ++ (LZ4.encNib (s.mlen - 4) ++ (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final))) := by
      rw [hinpS]; simp [LZ4.encodeSeq, List.append_assoc]
    have hro1 : st.inp.getD (s5.vars vIP) 0 = UInt8.ofNat (s.offset % 256) := by
      rw [hAdecomp, hAlen, getD_append_at]; rfl
    have hro2 : st.inp.getD (s5.vars vIP + 1) 0 = UInt8.ofNat (s.offset / 256) := by
      rw [hAdecomp, hAlen]
      exact getD_middle _ [UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)] _ 1 (by simp)
    -- the final ite: still inside the block, offset bytes remain ⇒ take matchPart
    have hs5ip_lt : s5.vars vIP < st.inp.length := by
      rw [hAlen, hAdecomp]; simp [List.length_append]
    have hcond2 : Cmp.eval .ge (s5.vars vIP) (s5.get (.v vEND)) = false := by
      simp only [Cmp.eval, St.get, hs5end]
      exact Bool.eq_false_iff.mpr
        (fun h => (Nat.not_le_of_gt hs5ip_lt) (Nat.le_of_ble_eq_true h))
    rw [eval_seq, eval_ite_false _ _ _ _ _ _ _ hcond2]
    -- matchPart
    rw [matchPart]; simp only [block, List.foldr_cons, List.foldr_nil]
    rw [matchHdr_eval]
    have hm_inp : (matchHdrState s5).inp = st.inp := by rw [matchHdrState_inp, hs5inp_st]
    have hm_ip : (matchHdrState s5).vars vIP = s5.vars vIP + 2 := matchHdrState_ip s5
    have hm_len : (matchHdrState s5).vars vLEN = min (s.mlen - 4) 15 := by
      rw [matchHdrState_len, hs5tok]
      rw [Nat.and_two_pow_sub_one_eq_mod _ 4]
      omega
    have hm_off : (matchHdrState s5).vars vOFF = s.offset := by
      rw [matchHdrState_off, hs5inp_st, hro1, hro2, toNat_byte (by omega), toNat_byte (by omega),
          Nat.shiftLeft_eq, show (2:Nat)^8 = 256 from rfl]; omega
    have hm_op : (matchHdrState s5).vars vOP = acc.length + s.lits.length := by
      rw [matchHdrState_other s5 (by decide) (by decide) (by decide) (by decide) (by decide),
          hs5op]
    have hm_run : (matchHdrState s5).vars vRUN = 1 := by
      rw [matchHdrState_other s5 (by decide) (by decide) (by decide) (by decide) (by decide),
          hs5run]
    have hm_end : (matchHdrState s5).vars vEND = st.inp.length := by
      rw [matchHdrState_other s5 (by decide) (by decide) (by decide) (by decide) (by decide),
          hs5end]
    have hm_out : (matchHdrState s5).out = acc ++ s.lits
        ++ List.replicate (((s :: ss).foldl LZ4.expandSeq acc ++ final).length
          - (acc.length + s.lits.length)) 0 := by
      rw [matchHdrState_out, hs5out]
    -- readExt (match length)
    obtain ⟨s7, hs7ev, hs7inp, hs7out, hs7ip, hs7len, hs7frame⟩ :=
      readExt_spec (s.mlen - 4)
        (((pre ++ [UInt8.ofNat (min s.lits.length 15 * 16 + min (s.mlen - 4) 15)]
          ++ LZ4.encNib s.lits.length) ++ s.lits)
          ++ [UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)])
        (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final) (fw + 1) (matchHdrState s5)
        (by rw [hm_inp, hAdecomp]; simp [List.append_assoc])
        (by
          rw [hm_ip, hAlen]
          simp only [List.length_append, List.length_cons, List.length_nil]
        )
        hm_len (by rw [hLs] at hcap_ge; omega)
    rw [eval_seq, hs7ev, Option.bind_some]
    have hs7op : s7.vars vOP = acc.length + s.lits.length := by
      rw [hs7frame vOP (by decide) (by decide) (by decide), hm_op]
    have hs7off : s7.vars vOFF = s.offset := by
      rw [hs7frame vOFF (by decide) (by decide) (by decide), hm_off]
    have hs7run : s7.vars vRUN = 1 := by
      rw [hs7frame vRUN (by decide) (by decide) (by decide), hm_run]
    have hs7end : s7.vars vEND = st.inp.length := by
      rw [hs7frame vEND (by decide) (by decide) (by decide), hm_end]
    have hs7out : s7.out = acc ++ s.lits
        ++ List.replicate (((s :: ss).foldl LZ4.expandSeq acc ++ final).length
          - (acc.length + s.lits.length)) 0 := by rw [hs7out, hm_out]
    have hs7inp_st : s7.inp = st.inp := by rw [hs7inp, hm_inp]
    -- preCopy (vLEN += 4; vSRC := vOP - vOFF)
    rw [preCopy_eval]
    have hpc_src : (preCopyState s7).vars vSRC = (acc ++ s.lits).length - s.offset := by
      rw [preCopyState_src, hs7op, hs7off]; simp [List.length_append]
    have hpc_op : (preCopyState s7).vars vOP = (acc ++ s.lits).length := by
      rw [preCopyState_other s7 (by decide) (by decide), hs7op]; simp [List.length_append]
    have hpc_len : (preCopyState s7).vars vLEN = s.mlen := by
      rw [preCopyState_len, hs7len]; omega
    have hpc_out : (preCopyState s7).out = (acc ++ s.lits)
        ++ List.replicate (((s :: ss).foldl LZ4.expandSeq acc ++ final).length
          - (acc ++ s.lits).length) 0 := by
      rw [preCopyState_out, hs7out]; simp [List.length_append]
    -- mcopy
    obtain ⟨s8, hs8ev, hs8inp, hs8out, hs8op, hs8len, hs8frame⟩ :=
      mcopy_spec s.mlen (acc ++ s.lits) s.offset
        ((s :: ss).foldl LZ4.expandSeq acc ++ final).length (preCopyState s7)
        ho1 (by simp [List.length_append]; omega) hpc_src hpc_op hpc_len hpc_out
        (by
          have hcopyLen : (acc ++ s.lits).length + s.mlen = (LZ4.expandSeq acc s).length := by
            rw [hLs]
            simp [List.length_append]
          omega)
    rw [eval_seq, eval_mono (fw+1) s.mlen mcopyLoop (preCopyState s7) s8 hs8ev
        (by rw [hLs] at hcap_ge; omega), Option.bind_some]
    -- the trailing skips collapse
    rw [eval_skip, Option.bind_some, eval_skip, Option.bind_some]
    -- s8 is positioned for the tail; assemble IH preconditions
    have hs8inp_st : s8.inp = st.inp := by
      rw [hs8inp, preCopyState_inp, hs7inp_st]
    have hexpand : LZ4.copyMatch (acc ++ s.lits) s.offset s.mlen = LZ4.expandSeq acc s := rfl
    have hs8out' : s8.out = LZ4.expandSeq acc s
        ++ List.replicate (((s :: ss).foldl LZ4.expandSeq acc ++ final).length
          - (LZ4.expandSeq acc s).length) 0 := by
      rw [hs8out, hexpand, hLs, List.length_append]; simp [List.length_append]
    have hs8op' : s8.vars vOP = (LZ4.expandSeq acc s).length := by
      rw [hs8op, hLs, List.length_append]
    have hs8ip : s8.vars vIP = pre.length + (LZ4.encodeSeq s).length := by
      rw [hs8frame vIP (by decide) (by decide) (by decide) (by decide),
          preCopyState_other s7 (by decide) (by decide), hs7ip]
      rw [show LZ4.encodeSeq s = UInt8.ofNat (min s.lits.length 15 * 16 + min (s.mlen - 4) 15)
            :: (LZ4.encNib s.lits.length ++ s.lits
              ++ [UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)]
              ++ LZ4.encNib (s.mlen - 4)) from rfl]
      simp [List.length_append]
      omega
    have hs8run : s8.vars vRUN = 1 := by
      rw [hs8frame vRUN (by decide) (by decide) (by decide) (by decide),
          preCopyState_other s7 (by decide) (by decide), hs7run]
    have hs8end : s8.vars vEND = s8.inp.length := by
      rw [hs8frame vEND (by decide) (by decide) (by decide) (by decide),
          preCopyState_other s7 (by decide) (by decide), hs7end, hs8inp_st]
    -- close the outer while via IH
    have goalRW : ((s :: ss).foldl LZ4.expandSeq acc ++ final)
        = (ss.foldl LZ4.expandSeq (LZ4.expandSeq acc s) ++ final) := by rw [hfoldcons]
    rw [goalRW]
    have hIH := ih final (LZ4.expandSeq acc s) (pre ++ LZ4.encodeSeq s) fw s8
      (fun x hx => hwf x (List.mem_cons_of_mem _ hx)) hvalid_tl
      (by
        have : (ss.foldl LZ4.expandSeq (LZ4.expandSeq acc s) ++ final).length
            = ((s :: ss).foldl LZ4.expandSeq acc ++ final).length := by rw [hfoldcons]
        rw [this]; omega)
      (by
        rw [hs8inp_st, hinpS, List.append_assoc])
      (by rw [hs8ip]; simp [List.length_append])
      hs8run hs8end hs8op' hs8out'
    simpa [decompressProg, mainBody, block, List.foldr_cons, List.foldr_nil] using hIH

private theorem seqs_length_le_encode_length (seqs : List LZ4.Seq) (final : List UInt8) :
    seqs.length ≤ (seqs.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final).length := by
  induction seqs with
  | nil => simp
  | cons s ss ih =>
    calc
      (s :: ss).length = ss.length + 1 := by simp
      _ ≤ (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final).length + 1 :=
        Nat.succ_le_succ ih
      _ ≤ (LZ4.encodeSeq s).length
          + (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final).length := by
        have hs : 1 ≤ (LZ4.encodeSeq s).length := by simp [LZ4.encodeSeq]
        omega
      _ = (LZ4.encodeSeq s ++ (ss.flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final)).length := by
        simp
      _ = ((s :: ss).flatMap LZ4.encodeSeq ++ LZ4.encodeFinal final).length := by
        simp [List.flatMap_cons, List.append_assoc]

/-- End-to-end refinement: for any byte-format well-formed block whose match
    offsets point into bytes already emitted, the executable Lean decompressor
    applied to the spec encoder returns the spec expansion. -/
theorem decompress_encode (b : LZ4.Block) (hwf : b.WF) (hvalid : ValidFrom [] b.seqs) :
    decompress b.encode b.expand.length = some b.expand := by
  unfold decompress
  let st0 := St.init b.encode b.expand.length
  have hmain := mainLoop_spec b.seqs b.final [] []
    (defaultFuel b.encode.length b.expand.length) st0
    hwf hvalid
    (by
      have hseqs := seqs_length_le_encode_length b.seqs b.final
      simp only [LZ4.Block.expand, LZ4.Block.encode, defaultFuel]
      omega)
    (by simp [st0, St.init, LZ4.Block.encode])
    (by simp [st0, St.init, initVars, vIP, vRUN, vEND])
    (by simp [st0, St.init, initVars, vRUN])
    (by simp [st0, St.init, initVars, vRUN, vEND, LZ4.Block.encode])
    (by simp [st0, St.init, initVars, vRUN, vEND, vOP])
    (by simp [st0, St.init, LZ4.Block.expand])
  obtain ⟨st', hev, hout, hop⟩ := hmain
  rw [hev]
  simp only [Option.map_some]
  congr
  rw [hout, hop]
  simpa [List.length_append, LZ4.Block.expand] using
    (List.take_length (l := List.foldl LZ4.expandSeq [] b.seqs ++ b.final))

-- ── The warp-cooperative program is proven equal to the serial one ────────────

end AlgorithmLib.LZ4Imp
