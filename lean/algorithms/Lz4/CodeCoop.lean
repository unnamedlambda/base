module
public import Lz4.CodeLane
public import AlgorithmLib.Kernel.Warp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeCoop` — one lane's share of a cooperative statement

What the code `Lz4Code.codeOf` writes for a cooperative statement does in one
lane (`coopL`), its collectives answered from the lane's answers `ans` as a
thread's are (`Kernel.AnswersFrom`), each a word (`Kernel.Words`): the copy
its bytes `lane, lane + 32, …`; the window its probe at `sp + lane`, the
ballot of hits, the lowest hit's position, its insert, the barrier, and the
lowest hit's candidate shuffled from that lane; an extend step its byte
compare and the ballot of mismatches. `coopL_ends`: the code ends as `coopL`
says, so `codeOf_lane` holds with `coopL` for the cooperative statements
(`codeOf_lane_coop`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

/-- The word a lane's `k`-th collective is answered with. -/
def ansW (ans : List Answer) (k : Nat) : UInt64 := s32 (wordAt ans k)

/-- A 32-bit word's trailing zeros, 32 where it is zero, as `ctz` leaves it. -/
def ctz32 (x : UInt64) : UInt64 :=
  UInt64.ofNat (((List.range 32).find? fun i => (x >>> UInt64.ofNat i) &&& 1 == 1).getD 32) &&& 0xffffffff

/-- Four bytes at `a`, little-endian. -/
def load4L (m : Mem) (a : UInt64) : Option UInt64 := do
  let b0 ← m.load a 1
  let b1 ← m.load (a + c64 1) 1
  let b2 ← m.load (a + c64 2) 1
  let b3 ← m.load (a + c64 3) 1
  some ((b0 ||| b1 <<< c64 8) ||| (b2 <<< c64 16 ||| b3 <<< c64 24))

/-- One trip of a lane's share of a copy: byte `i + lane` loaded, and
    stored where it is below `n`. -/
def copyStep (d s n lv i : UInt64) (w : World) : Option World :=
  if i + lv < n then do
    let v ← w.mem.load (s + (i + lv)) 1
    stw w (d + (i + lv)) 1 (v &&& 0xff)
  else do
    let _ ← w.mem.load (s + (i + lv)) 1
    some w

/-- A step that loads, then goes on where it was, ends where it was. -/
theorem bind_const_some {α β : Type} {o : Option α} {w w1 : β} (h : o.bind (fun _ => some w) = some w1) :
    w1 = w := by
  cases o <;> simp at h; exact h.symm

/-- A lane's share of a copy of `n` bytes from `s` to `d`: bytes `i + lane`
    for `i` from `i0` by 32, `f` trips of fuel. -/
def copyL (d s n lv : UInt64) : Nat → UInt64 → World → Option World
  | 0, i, w => if i < n then none else some w
  | f + 1, i, w =>
      if i < n then do
        let w1 ← copyStep d s n lv i w
        copyL d s n lv f (i + c64 32) w1
      else some w

/-- A store made where `c` holds. -/
def stwIf (c : Bool) (w : World) (a : UInt64) (n : Nat) (v : UInt64) : Option World :=
  if c then stw w a n v else some w

/-- A collective's call observed. -/
abbrev callW (w : World) (c : Intr) (args : List V) : World := obsCall w (.intr c) args

/-- **One lane's share of a cooperative statement**, its `k`-th collective
    answered from `ans[k]`, counted from the `start`-th observation: none
    where an access reaches nothing, a barrier's answer is not shared memory,
    or the world has fewer observations than the lane started with. -/
def coopL (ans : List Answer) (start : Nat) (sm lv : UInt64) : WStmt → LSt → Option LSt
  | .coopCopy dst src len, st => do
      let w ← copyL (st.regs dst) (st.regs src) (st.regs len) lv ((st.regs len).toNat + 1) (c64 0) st.w
      some { st with w }
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog _, st =>
      if start ≤ st.w.obs.length then do
        let inBase := st.regs "inBase"
        let tbl := st.regs "tbl"
        let s := st.regs sp
        let posP := s + lv
        let pValid := posP < c64 searchLim
        let cap4 := c64 (inStride - 4)
        let rp := if posP ≤ cap4 then posP else cap4
        let v32 ← load4L st.w.mem (inBase + rp)
        let hh := ((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)
        let addr := (hh <<< c64 1) + tbl
        let candRaw ← st.w.mem.load (sm + addr) 2
        let cand := candRaw + c64 (-1)
        let rc := if cand ≤ cap4 then cand else cap4
        let vc ← load4L st.w.mem (inBase + rc)
        let hit := (pValid && candRaw != c64 0) && (cand < posP && vc == v32)
        let w1 := callW st.w .ballot [boolV hit]
        let bal := ansW ans (collCalls st.w start).length
        let first := ctz32 bal &&& 0xffffffff
        let w2 ← stwIf (lv ≤ first && pValid) w1 (sm + addr) 2 ((posP + c64 1) &&& 0xffff)
        match ans[(collCalls w2 start).length]? with
        | some (.smem b) =>
            let w3 := callW w2 .barrier []
            let w3 := { w3 with mem := { w3.mem with arena := b } }
            let w4 := callW w3 .shflIdx [.sc .i32 (cand &&& 0xffffffff), .sc .i32 (first &&& 0xffffffff)]
            let c0 := ansW ans (collCalls w3 start).length &&& 0xffffffff
            some { regs := (((LSt.set ⟨st.regs, w4⟩ p0 (s + first)).set cand0 c0).set found
                     (if bal != 0 then 1 else 0)).regs, w := w4 }
        | _ => none
      else none
  | .coopExtendStep adv p0 cand0 ml _ _, st =>
      if start ≤ st.w.obs.length then do
        let inBase := st.regs "inBase"
        let pe := st.regs p0 + (st.regs ml + lv)
        let pIn := pe < st.regs "ecR"
        let peC := if pe ≤ st.regs "ec1" then pe else st.regs "ec1"
        let caC := st.regs cand0 + (peC - st.regs p0)
        let bP ← st.w.mem.load (inBase + peC) 1
        let bC ← st.w.mem.load (inBase + caC) 1
        let ok := pIn && bP == bC
        let bal := ansW ans (collCalls st.w start).length
        some (LSt.set ⟨st.regs, callW st.w .ballot [boolV ok]⟩ adv
          (ctz32 ((bal ^^^ 0xffffffff) &&& 0xffffffff) &&& 0xffffffff))
      else none
  | _, _ => none

section
variable {cfg : Cfg} {Γ : Env} {w : World}

theorem wpE_icmp64 {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i64 x y))) w) : wpE cfg (icmp cc a b) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipIntCmp, ClifTy.isInt]) h

theorem wpE_icmp32 {cc : ICmpCond} {a b : Slot .i32} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i32 x y))) w) : wpE cfg (icmp cc a b) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipIntCmp, ClifTy.isInt]) h

theorem wpE_band8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (p && q))) w) : wpE cfg (band a b) Q Γ w := by
  refine wpE_val ?_ h
  simp only [Op'.erase, evalOp, Sem.get, ha, hb]
  cases p <;> cases q <;> rfl

theorem wpE_uload8 {a : Slot .i64} {x v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hl : w.mem.load x 1 = some v)
    (h : Q Γ.size (Γ.push (.sc .i64 v)) w) : wpE cfg (uload8_64 a) Q Γ w :=
  wpE_val (by simp only [Op'.erase, evalOp, Sem.get, ha]; simp [hl, norm, and_wide64]) h

theorem wpE_uload16 {a : Slot .i64} {x v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hl : w.mem.load x 2 = some v)
    (h : Q Γ.size (Γ.push (.sc .i64 v)) w) : wpE cfg (uload16_64 a) Q Γ w :=
  wpE_val (by simp only [Op'.erase, evalOp, Sem.get, ha]; simp [hl, norm, and_wide64]) h

theorem wpE_ushrImm64 {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x >>> c64 imm))) w) :
    wpE cfg (ushrImm a imm) Q Γ w := by
  refine wpE_bind (wpE_iconst64 ?_)
  have ha' := (Extends.push Γ (.sc .i64 (c64 imm))).get ha
  refine wpE_val ?_ h
  simp [Op'.erase, evalOp, Sem.get, ha', shiftBin, ClifTy.isInt, ClifTy.width, norm, and_wide64, shr_mod64]

theorem wpE_ctz32 {a : Slot .i32} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (h : Q Γ.size (Γ.push (.sc .i32 (ctz32 x))) w) : wpE cfg (ctz a) Q Γ w := by
  refine wpE_val ?_ h
  simp [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, norm, widthMask, ctz32]

theorem wpE_uext32' {a : Slot .i32} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (h : Q Γ.size (Γ.push (.sc .i64 (x &&& 0xffffffff))) w) :
    wpE cfg (uextend64 a) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, widthMask]) h

theorem wpE_ireduce32 {a : Slot .i64} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (h : Q Γ.size (Γ.push (.sc .i32 (x &&& 0xffffffff))) w) :
    wpE cfg (ireduce32 a) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, un, ha, ClifTy.isInt, ClifTy.width, norm, widthMask]) h

/-! Each operation's rule, its value handed on in a slot of an environment
keeping `Γ`. -/

theorem xget {v : V} {Q : Slot ty → Env → World → Prop}
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some v → Q s Γ' w) : Q Γ.size (Γ.push v) w :=
  h _ _ (Extends.push _ _) (get_push_last _ _)

theorem xiadd {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x + y)) → Q s Γ' w) : wpE cfg (iadd a b) Q Γ w :=
  wpE_iadd ha hb (xget h)

theorem xicmp64 {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (boolV (cmpInt cc .i64 x y)) → Q s Γ' w) :
    wpE cfg (icmp cc a b) Q Γ w :=
  wpE_icmp64 ha hb (xget h)

theorem xband8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (boolV (p && q)) → Q s Γ' w) : wpE cfg (band a b) Q Γ w :=
  wpE_band8 ha hb (xget h)

theorem xuload8 {a : Slot .i64} {x v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hl : w.mem.load x 1 = some v)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 v) → Q s Γ' w) : wpE cfg (uload8_64 a) Q Γ w :=
  wpE_uload8 ha hl (xget h)

theorem xuload16 {a : Slot .i64} {x v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hl : w.mem.load x 2 = some v)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 v) → Q s Γ' w) : wpE cfg (uload16_64 a) Q Γ w :=
  wpE_uload16 ha hl (xget h)

theorem xctz32 {a : Slot .i32} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 (ctz32 x)) → Q s Γ' w) : wpE cfg (ctz a) Q Γ w :=
  wpE_ctz32 ha (xget h)

theorem xuext32 {a : Slot .i32} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (uextend64 a) Q Γ w :=
  wpE_uext32' ha (xget h)

theorem xireduce32 {a : Slot .i64} {x : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 (x &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (ireduce32 a) Q Γ w :=
  wpE_ireduce32 ha (xget h)

theorem xiconst32 {k : Int} {Q : Slot .i32 → Env → World → Prop}
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (ofInt .i32 k) → Q s Γ' w) : wpE cfg (iconst32 k) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp]) (xget h)

theorem xbxor32 {a b : Slot .i32} {x y : UInt64} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i32 ((x ^^^ y) &&& 0xffffffff)) → Q s Γ' w) :
    wpE cfg (bxor a b) Q Γ w :=
  wpE_val (by simp [Op'.erase, evalOp, Sem.get, ha, hb, zipIntBits, ClifTy.isInt, norm, widthMask]) (xget h)

theorem xmul {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x * y)) → Q s Γ' w) : wpE cfg (imul a b) Q Γ w :=
  wpE_opOf .mul ha hb h

theorem xsub {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x - y)) → Q s Γ' w) : wpE cfg (isub a b) Q Γ w :=
  wpE_opOf .sub ha hb h

theorem xband64 {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x &&& y)) → Q s Γ' w) : wpE cfg (band a b) Q Γ w :=
  wpE_opOf .band ha hb h

theorem xbor64 {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x ||| y)) → Q s Γ' w) : wpE cfg (bor a b) Q Γ w :=
  wpE_opOf .bor ha hb h

theorem xumin {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (if x ≤ y then x else y)) → Q s Γ' w) :
    wpE cfg (umin a b) Q Γ w :=
  wpE_opOf .min ha hb h

theorem ofInt_m1 : ofInt .i32 (-1) = .sc .i32 0xffffffff := by decide

/-- **An extend step's code ends as the lane's share of it.** -/
theorem coopL_extend {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg) (hW : Words ans)
    {fl : List String} {sm lane : Slot .i64} {no : Slot .i8} {smv lv : UInt64}
    {adv p0 cand0 ml : String} {inStride endCap : Nat} {e : RegEnv Slot} {st st' : LSt}
    {Q : RegEnv Slot → Env → World → Prop}
    (hwf : Wf fl (.coopExtendStep adv p0 cand0 ml inStride endCap)) (hE : EnvOk fl e st.regs Γ)
    (hS : Slots sm lane no smv lv Γ)
    (hr : coopL ans start smv lv (.coopExtendStep adv p0 cand0 ml inStride endCap) st = some st')
    (hQ : ∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) :
    wpE cfg (codeOf fl sm lane no (.coopExtendStep adv p0 cand0 ml inStride endCap) e) Q Γ st.w := by
  simp only [coopL] at hr
  have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at hr
  simp only [hs, if_true] at hr
  obtain ⟨bP, hbP, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨bC, hbC, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  simp only [codeOf, codeOfT, leafCode]
  refine wpE_bind (wpE_get hE fun vIB Γ1 hx1 hIB => ?_)
  refine wpE_bind (wpE_get (hE.extend hx1) fun vP Γ2 hx2 hP => ?_)
  refine wpE_bind (wpE_get (hE.extend (hx1.trans hx2)) fun vM Γ3 hx3 hM => ?_)
  refine wpE_bind (xiadd hM (hS.extend (hx1.trans (hx2.trans hx3))).lane fun vML Γ4 hx4 hML => ?_)
  refine wpE_bind (xiadd ((hx3.trans hx4).get hP) hML fun vPE Γ5 hx5 hPE => ?_)
  have hx15 := hx1.trans (hx2.trans (hx3.trans (hx4.trans hx5)))
  refine wpE_bind (wpE_get (hE.extend hx15) fun vR Γ6 hx6 hR => ?_)
  refine wpE_bind (xicmp64 (cc := .ult) (hx6.get hPE) hR fun vIn Γ7 hx7 hIn => ?_)
  refine wpE_bind (wpE_get (hE.extend (hx15.trans (hx6.trans hx7))) fun vE1 Γ8 hx8 hE1 => ?_)
  refine wpE_bind (xumin ((hx6.trans (hx7.trans hx8)).get hPE) hE1 fun vPC Γ9 hx9 hPC => ?_)
  have hx19 := hx15.trans (hx6.trans (hx7.trans (hx8.trans hx9)))
  refine wpE_bind (wpE_get (hE.extend hx19) fun vC Γ10 hx10 hC => ?_)
  refine wpE_bind (wpE_get (hE.extend (hx19.trans hx10)) fun vP' Γ11 hx11 hP' => ?_)
  refine wpE_bind (xsub ((hx10.trans hx11).get hPC) hP' fun vD Γ12 hx12 hD => ?_)
  refine wpE_bind (xiadd ((hx11.trans hx12).get hC) hD fun vCA Γ13 hx13 hCA => ?_)
  have hx913 := hx10.trans (hx11.trans (hx12.trans hx13))
  have hx213 := (hx2.trans (hx3.trans (hx4.trans (hx5.trans (hx6.trans (hx7.trans (hx8.trans hx9))))))).trans hx913
  refine wpE_bind (xiadd (hx213.get hIB) (hx913.get hPC) fun vA1 Γ14 hx14 hA1 => ?_)
  refine wpE_bind (xuload8 hA1 hbP fun vBP Γ15 hx15' hBP => ?_)
  refine wpE_bind (xiadd ((hx213.trans (hx14.trans hx15')).get hIB) ((hx14.trans hx15').get hCA)
    fun vA2 Γ16 hx16 hA2 => ?_)
  refine wpE_bind (xuload8 hA2 hbC fun vBC Γ17 hx17 hBC => ?_)
  refine wpE_bind (xicmp64 (cc := .eq) ((hx16.trans hx17).get hBP) hBC fun vEq Γ18 hx18 hEq => ?_)
  refine wpE_bind (xband8 ((hx913.trans (hx14.trans (hx15'.trans (hx16.trans (hx17.trans hx18))))).get
    ((hx8.trans hx9).get hIn)) hEq fun vOk Γ19 hx19' hOk => ?_)
  have hx119 := hx19.trans (hx913.trans (hx14.trans (hx15'.trans (hx16.trans (hx17.trans (hx18.trans hx19'))))))
  refine wpE_bind (wpE_coll1 hA (c := .ballot) rfl ballot rfl hOk hs rfl (valAt_words hW _) ?_)
  refine wpE_bind (xiconst32 fun vM1 Γ20 hx20 hM1 => ?_)
  rw [ofInt_m1] at hM1
  refine wpE_bind (xbxor32 (hx20.get (get_push_last _ _)) hM1 fun vX Γ21 hx21 hX => ?_)
  refine wpE_bind (xctz32 hX fun vZ Γ22 hx22 hZ => ?_)
  refine wpE_bind (xuext32 hZ fun vU Γ23 hx23 hU => ?_)
  have hx := (hx119.trans (Extends.push _ _)).trans (hx20.trans (hx21.trans (hx22.trans hx23)))
  refine wpE_pure (hQ _ _ hx ?_)
  exact (hE.extend hx).setVal adv (hwf adv (by simp [assigns])) vU _ hU


theorem xiconst64 {k : Int} {Q : Slot .i64 → Env → World → Prop}
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (c64 k)) → Q s Γ' w) : wpE cfg (iconst64 k) Q Γ w :=
  wpE_iconst64 (xget h)

theorem xiaddImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x + c64 imm)) → Q s Γ' w) :
    wpE cfg (iaddImm a imm) Q Γ w :=
  wpE_iaddImm ha (h _ _ ((Extends.push _ _).push_of _) (get_push_last _ _))

theorem xishlImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x <<< c64 imm)) → Q s Γ' w) :
    wpE cfg (ishlImm a imm) Q Γ w :=
  wpE_ishlImm ha (by rw [shl_mod64]; exact h _ _ ((Extends.push _ _).push_of _) (get_push_last _ _))

theorem xushrImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (x >>> c64 imm)) → Q s Γ' w) :
    wpE cfg (ushrImm a imm) Q Γ w :=
  wpE_ushrImm64 ha (h _ _ ((Extends.push _ _).push_of _) (get_push_last _ _))

theorem xload4 {a : Slot .i64} {x v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hl : load4L w.mem x = some v)
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 v) → Q s Γ' w) : wpE cfg (load4 a) Q Γ w := by
  simp only [load4L] at hl
  obtain ⟨b0, h0, hl⟩ := Option.bind_eq_some_iff.mp hl
  obtain ⟨b1, h1, hl⟩ := Option.bind_eq_some_iff.mp hl
  obtain ⟨b2, h2, hl⟩ := Option.bind_eq_some_iff.mp hl
  obtain ⟨b3, h3, hl⟩ := Option.bind_eq_some_iff.mp hl
  obtain rfl := (Option.some.inj hl).symm
  unfold load4
  refine wpE_bind (xuload8 ha h0 fun s0 Γ1 hx1 hs0 => ?_)
  refine wpE_bind (xiaddImm (hx1.get ha) fun a1 Γ2 hx2 ha1 => ?_)
  refine wpE_bind (xuload8 ha1 h1 fun s1 Γ3 hx3 hs1 => ?_)
  refine wpE_bind (xishlImm hs1 fun t1 Γ4 hx4 ht1 => ?_)
  have hx14 := hx1.trans (hx2.trans (hx3.trans hx4))
  refine wpE_bind (xiaddImm (hx14.get ha) fun a2 Γ5 hx5 ha2 => ?_)
  refine wpE_bind (xuload8 ha2 h2 fun s2 Γ6 hx6 hs2 => ?_)
  refine wpE_bind (xishlImm hs2 fun t2 Γ7 hx7 ht2 => ?_)
  have hx47 := hx14.trans (hx5.trans (hx6.trans hx7))
  refine wpE_bind (xiaddImm (hx47.get ha) fun a3 Γ8 hx8 ha3 => ?_)
  refine wpE_bind (xuload8 ha3 h3 fun s3 Γ9 hx9 hs3 => ?_)
  refine wpE_bind (xishlImm hs3 fun t3 Γ10 hx10 ht3 => ?_)
  have hx710 := hx8.trans (hx9.trans hx10)
  have hx210 := (hx2.trans (hx3.trans (hx4.trans (hx5.trans (hx6.trans hx7))))).trans hx710
  refine wpE_bind (xbor64 (hx210.get hs0) ((hx5.trans (hx6.trans (hx7.trans hx710))).get ht1)
    fun o1 Γ11 hx11 ho1 => ?_)
  refine wpE_bind (xbor64 ((hx710.trans hx11).get ht2) (hx11.get ht3) fun o2 Γ12 hx12 ho2 => ?_)
  exact xbor64 (hx12.get ho1) ho2 fun s Γ' hx' hs' =>
    h s Γ' ((hx47.trans (hx710.trans (hx11.trans hx12))).trans hx') hs'

theorem stw_obs {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') :
    w'.obs.length = w.obs.length + 1 := by
  simp only [stw] at h
  obtain ⟨m, _, h⟩ := Option.bind_eq_some_iff.mp h
  obtain rfl := (Option.some.inj h).symm
  simp [obsStore]

/-- **Code under a test ends** where it ends when the test holds, and what
    follows holds after it or without it. -/
theorem wpE_whenT {t : ClifTy} {cc : ICmpCond} {a b : Slot t} {x y : UInt64} {thn : Prog Slot Lvl Unit}
    {Q : Unit → Env → World → Prop} (P : Env → World → Prop) (hti : t.isInt = true)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y))
    (ht : cmpInt cc t x y = true → wpE cfg thn (fun _ Γ' w' => P Γ' w') (Γ.push (boolV true)) w)
    (he : cmpInt cc t x y = false → ∀ Γ', Extends Γ Γ' → P Γ' w)
    (hP : ∀ Γ' w', P Γ' w' → ∀ n, Γ'.size ≤ n → Q () (bindAt Γ' n []) w') :
    wpE cfg (when cc a b thn) Q Γ w := by
  unfold when
  refine wpE_bind (wpE_ifteT (fun _ Γ' w' => P Γ' w') hti ha hb ?_ ?_ ?_ ?_)
  · exact fun hc => wpE_bind (wpE_mono (fun _ Γ' w' h => wpE_pure h) (ht hc))
  · exact fun hc ne hne => wpE_pure (he hc _ ((Extends.push _ _).bindAt_of hne []))
  · exact fun jv _ _ _ => ⟨[], by cases jv; rfl⟩
  · intro nJ Γ' jv vs w' hle hvs hq
    have : vs = [] := by cases jv; simpa [Vals.slots] using hvs.symm
    subst this
    exact wpE_pure (hP Γ' w' hq nJ hle)

/-- **Code under a flag ends** where it ends when the flag is set. -/
theorem wpE_when {c no : Slot .i8} {p : Bool} {thn : Prog Slot Lvl Unit} {Q : Unit → Env → World → Prop}
    (P : Env → World → Prop) (hc : Γ[c]? = some (boolV p)) (hno : Γ[no]? = some (.sc .i8 0))
    (ht : p = true → wpE cfg thn (fun _ Γ' w' => P Γ' w') (Γ.push (boolV true)) w)
    (he : p = false → ∀ Γ', Extends Γ Γ' → P Γ' w)
    (hP : ∀ Γ' w', P Γ' w' → ∀ n, Γ'.size ≤ n → Q () (bindAt Γ' n []) w') :
    wpE cfg (when .ne c no thn) Q Γ w :=
  wpE_whenT (x := if p then 1 else 0) (y := 0) P rfl (by rw [hc]; rfl) hno
    (fun hcmp => ht (by cases p <;> simp_all [cmpInt]))
    (fun hcmp => he (by cases p <;> simp_all [cmpInt])) hP

/-- A flag's 1 or 0, as a select between two constants makes it. -/
theorem xsel01 {f : Slot .i8} {p : Bool} {Q : Slot .i64 → Env → World → Prop}
    (hf : Γ[f]? = some (boolV p))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (.sc .i64 (if p then 1 else 0)) → Q s Γ' w) :
    wpE cfg (do select f (← iconst64 1) (← iconst64 0)) Q Γ w := by
  refine wpE_bind (wpE_iconst64 ?_)
  refine wpE_bind (wpE_iconst64 ?_)
  refine wpE_val (v := .sc .i64 (if p then 1 else 0)) ?_ (h _ _ (((Extends.push _ _).push_of _).push_of _)
    (get_push_last _ _))
  exact eval_sel01 _ Γ f _ (by cases p <;> decide) (by rw [hf]; cases p <;> rfl)


theorem xicmp32 {cc : ICmpCond} {a b : Slot .i32} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x)) (hb : Γ[b]? = some (.sc .i32 y))
    (h : ∀ s Γ', Extends Γ Γ' → Γ'[s]? = some (boolV (cmpInt cc .i32 x y)) → Q s Γ' w) :
    wpE cfg (icmp cc a b) Q Γ w :=
  wpE_icmp32 ha hb (xget h)

/-- **A 16-bit shared store ends** where the store reaches memory. -/
theorem wpE_stsh {smS aS vS : Slot .i64} {smv a x : UInt64} {w' : World} {Q : Unit → Env → World → Prop}
    (hsm : Γ[smS]? = some (.sc .i64 smv)) (ha : Γ[aS]? = some (.sc .i64 a)) (hv : Γ[vS]? = some (.sc .i64 x))
    (hs : stw w (smv + a) 2 (x &&& 0xffff) = some w') (h : ∀ Γ', Extends Γ Γ' → Q () Γ' w') :
    wpE cfg (stsh smS aS vS) Q Γ w := by
  simp only [stw] at hs
  obtain ⟨m, hm, hs⟩ := Option.bind_eq_some_iff.mp hs
  obtain rfl := (Option.some.inj hs).symm
  simp only [stsh, istore16]
  refine wpE_bind (wpE_iadd hsm ha ?_)
  have hv2 := (Extends.push Γ (.sc .i64 (smv + a))).get hv
  refine wpE_bind (wpE_val (v := .sc .i16 (x &&& 0xffff)) (by
    simp only [Op'.erase, evalOp, Sem.get, hv2]; simp [norm, IExt.admits, ClifTy.isInt, ClifTy.width]; rfl) ?_)
  have hp := (Extends.push _ (.sc .i16 (x &&& 0xffff))).get (get_push_last Γ (.sc .i64 (smv + a)))
  have hq := get_push_last (Γ.push (.sc .i64 (smv + a))) (.sc .i16 (x &&& 0xffff))
  refine wpE_store (w' := { obsStore w (smv + a) 2 (x &&& 0xffff) with mem := m })
    (by simp only [runStmt, Sem.get, hp, hq]; simp [tyBytes, ClifTy.width, hm]) (wpE_ret ?_)
  exact h _ (((Extends.push _ _).push_of _))

theorem stwIf_true {c : Bool} {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (hc : c = true)
    (h : stwIf c w a n v = some w') : stw w a n v = some w' := by
  subst hc; exact h

theorem stwIf_false {c : Bool} {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (hc : c = false)
    (h : stwIf c w a n v = some w') : w' = w := by
  subst hc; exact (Option.some.inj h).symm

theorem stwIf_obs {c : Bool} {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stwIf c w a n v = some w') :
    w.obs.length ≤ w'.obs.length := by
  unfold stwIf at h
  split at h
  · rw [stw_obs h]; omega
  · cases h; omega

/-- A slot's value, read through the step-wise extensions in scope. -/
syntax "lift" : tactic
macro_rules
  | `(tactic| lift) => `(tactic| first | assumption | (refine Extends.get (by assumption) ?_; lift))

/-- An extension through the step-wise extensions in scope. -/
syntax "liftx" : tactic
macro_rules
  | `(tactic| liftx) => `(tactic| first | exact Extends.refl _ | (refine Extends.trans ?_ (by assumption); liftx))

/-- A named slot fact, read through the step-wise extensions in scope. -/
syntax "liftf " term : tactic
macro_rules
  | `(tactic| liftf $h) => `(tactic| exact Extends.get (by liftx) $h)

set_option maxHeartbeats 1000000 in
/-- **A window's code ends as the lane's share of it.** -/
theorem coopL_window {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg) (hW : Words ans)
    {fl : List String} {sm lane : Slot .i64} {no : Slot .i8} {smv lv : UInt64}
    {found p0 cand0 sp : String} {inStride searchLim hashLog tbl : Nat} {e : RegEnv Slot} {st st' : LSt}
    {Q : RegEnv Slot → Env → World → Prop}
    (hwf : Wf fl (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl)) (hE : EnvOk fl e st.regs Γ)
    (hS : Slots sm lane no smv lv Γ)
    (hr : coopL ans start smv lv (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) st = some st')
    (hQ : ∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) :
    wpE cfg (codeOf fl sm lane no (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) e) Q Γ st.w := by
  simp only [coopL] at hr
  have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at hr
  simp only [hs, if_true] at hr
  obtain ⟨v32, hv32, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨candRaw, hcr, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨vc, hvc, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨w2, hw2, hr⟩ := Option.bind_eq_some_iff.mp hr
  split at hr
  case h_2 => cases hr
  rename_i b hb
  obtain rfl := (Option.some.inj hr).symm
  have hsm := hS.sm; have hlane := hS.lane; have hno := hS.no
  simp only [codeOf, codeOfT, leafCode]
  refine wpE_bind (wpE_get hE fun vIB Γ1 hx1 hIB => ?_)
  refine wpE_bind (wpE_get (hE.extend (by liftx)) fun vT Γ2 hx2 hT => ?_)
  refine wpE_bind (wpE_get (hE.extend (by liftx)) fun vS Γ3 hx3 hSp => ?_)
  refine wpE_bind (xiadd hSp (by lift) fun vPos Γ4 hx4 hPos => ?_)
  refine wpE_bind (xiconst64 fun vSL Γ5 hx5 hSL => ?_)
  refine wpE_bind (xicmp64 (cc := .ult) (by lift) hSL fun vPV Γ6 hx6 hPV => ?_)
  refine wpE_bind (xiconst64 fun vC4 Γ7 hx7 hC4 => ?_)
  refine wpE_bind (xumin (by lift) hC4 fun vRP Γ8 hx8 hRP => ?_)
  refine wpE_bind (xiadd (by lift) hRP fun vA Γ9 hx9 hA9 => ?_)
  refine wpE_bind (xload4 hA9 hv32 fun vV Γ10 hx10 hV => ?_)
  refine wpE_bind (xiconst64 fun vK Γ11 hx11 hK => ?_)
  refine wpE_bind (xmul (by lift) hK fun vM Γ12 hx12 hM => ?_)
  refine wpE_bind (xushrImm hM fun vSh Γ13 hx13 hSh => ?_)
  refine wpE_bind (xiconst64 fun vMk Γ14 hx14 hMk => ?_)
  refine wpE_bind (xband64 (by lift) hMk fun vHH Γ15 hx15 hHH => ?_)
  refine wpE_bind (xishlImm hHH fun vHS Γ16 hx16 hHS => ?_)
  refine wpE_bind (xiadd hHS (by lift) fun vAd Γ17 hx17 hAd => ?_)
  refine wpE_bind (xiadd (by lift) hAd fun vSA Γ18 hx18 hSA => ?_)
  refine wpE_bind (xuload16 hSA hcr fun vCR Γ19 hx19 hCR => ?_)
  refine wpE_bind (xiaddImm hCR fun vCd Γ20 hx20 hCd => ?_)
  refine wpE_bind (xumin hCd (by lift) fun vRC Γ21 hx21 hRC => ?_)
  refine wpE_bind (xiadd (by lift) hRC fun vA2 Γ22 hx22 hA22 => ?_)
  refine wpE_bind (xload4 hA22 hvc fun vVC Γ23 hx23 hVC => ?_)
  refine wpE_bind (xiconst64 fun vZ0 Γ24 hx24 hZ0 => ?_)
  refine wpE_bind (xicmp64 (cc := .ne) (by lift) hZ0 fun vNZ Γ25 hx25 hNZ => ?_)
  refine wpE_bind (xband8 (by lift) hNZ fun vB1 Γ26 hx26 hB1 => ?_)
  refine wpE_bind (xicmp64 (cc := .ult) (by lift) (by lift) fun vLt Γ27 hx27 hLt => ?_)
  refine wpE_bind (xicmp64 (cc := .eq) (by lift) (by lift) fun vEq Γ28 hx28 hEq => ?_)
  refine wpE_bind (xband8 (by lift) hEq fun vB2 Γ29 hx29 hB2 => ?_)
  refine wpE_bind (xband8 (by lift) hB2 fun vHit Γ30 hx30 hHit => ?_)
  have hPV30 := (show Extends Γ6 Γ30 by liftx).get hPV
  have hPos30 := (show Extends Γ4 Γ30 by liftx).get hPos
  have hCd30 := (show Extends Γ20 Γ30 by liftx).get hCd
  have hAd30 := (show Extends Γ17 Γ30 by liftx).get hAd
  have hSp30 := (show Extends Γ3 Γ30 by liftx).get hSp
  have hc30 : Extends Γ Γ30 := by liftx
  have hlane30 : Γ30[lane]? = some (.sc .i64 lv) := hc30.get hlane
  have hsm30 : Γ30[sm]? = some (.sc .i64 smv) := hc30.get hsm
  have hno30 : Γ30[no]? = some (.sc .i8 0) := hc30.get hno
  have hE30 := hE.extend hc30
  clear hc30
  refine wpE_bind (wpE_coll1 hA (c := .ballot) rfl ballot rfl hHit hs rfl (valAt_words hW _) ?_)
  have hx31 : Extends Γ30 (Γ30.push (.sc .i32 (s32 (wordAt ans (collCalls st.w start).length)))) := Extends.push _ _
  have hBal := get_push_last Γ30 (.sc .i32 (s32 (wordAt ans (collCalls st.w start).length)))
  refine wpE_bind (xctz32 hBal fun vCz Γ32 hx32 hCz => ?_)
  refine wpE_bind (xuext32 hCz fun vF Γ33 hx33 hF => ?_)
  refine wpE_bind (xicmp64 (cc := .ule) (by liftf hlane30) hF fun vLe Γ34 hx34 hLe => ?_)
  refine wpE_bind (xband8 hLe (by liftf hPV30) fun vIns Γ35 hx35 hIns => ?_)
  have hno35 : Γ35[no]? = some (.sc .i8 0) := by liftf hno30
  have hs1 : start ≤ (callW st.w .ballot [boolV ((cmpInt .ult .i64 (st.regs sp + lv) (c64 searchLim) &&
      cmpInt .ne .i64 candRaw (c64 0)) && (cmpInt .ult .i64 (candRaw + c64 (-1)) (st.regs sp + lv) &&
      cmpInt .eq .i64 vc v32))]).obs.length := by
    simp only [callW, obsCall_length]; omega
  have hs2 : start ≤ w2.obs.length := Nat.le_trans hs1 (stwIf_obs hw2)
  refine wpE_bind (wpE_when (fun Γ' w' => Extends Γ35 Γ' ∧ w' = w2) hIns hno35 ?_ ?_ ?_)
  · intro hp
    have hxT : Extends Γ35 (Γ35.push (boolV true)) := Extends.push _ _
    refine wpE_bind (xiaddImm (by liftf hPos30) fun vP1 Γa hxa hP1 => ?_)
    exact wpE_stsh (by liftf hsm30) (by liftf hAd30) hP1 (stwIf_true hp hw2) fun Γ' hx' =>
      ⟨hxT.trans (hxa.trans hx'), rfl⟩
  · intro hp Γ' hx'
    exact ⟨hx', (stwIf_false hp hw2).symm⟩
  · rintro Γ36 w' ⟨hx36, hw'⟩ n hn
    have hw2' := hw'.symm
    subst hw2'
    have hx36' : Extends Γ36 (bindAt Γ36 n []) := (Extends.refl _).bindAt_of hn []
    refine wpE_bind (wpE_bar hA barrier rfl hs2 rfl hb ?_)
    refine wpE_bind (xireduce32 (by liftf hCd30) fun vR1 Γ37 hx37 hR1 => ?_)
    refine wpE_bind (xireduce32 (by liftf hF) fun vR2 Γ38 hx38 hR2 => ?_)
    refine wpE_bind (wpE_coll2 hA (c := .shflIdx) rfl shflIdx rfl (by liftf hR1) hR2
      (by simp only [obsCall_length]; omega) rfl (valAt_words hW _) ?_)
    have hx39 := Extends.push Γ38 (.sc .i32 (s32 (wordAt ans (collCalls ({ obsCall w2 (.intr .barrier) [] with
      mem := { (obsCall w2 (.intr .barrier) []).mem with arena := b } }) start).length)))
    have hC0 := get_push_last Γ38 (.sc .i32 (s32 (wordAt ans (collCalls ({ obsCall w2 (.intr .barrier) [] with
      mem := { (obsCall w2 (.intr .barrier) []).mem with arena := b } }) start).length)))
    refine wpE_bind (xiadd (by liftf hSp30) (by liftf hF) fun vP0 Γ40 hx40 hP0 => ?_)
    refine wpE_bind (xuext32 (by liftf hC0) fun vC0 Γ41 hx41 hC0e => ?_)
    refine wpE_bind (xiconst32 fun vZ Γ42 hx42 hZ => ?_)
    rw [show ofInt .i32 0 = .sc .i32 0 by decide] at hZ
    refine wpE_bind (xicmp32 (cc := .ne) (by liftf hBal) hZ fun vFd Γ43 hx43 hFd => ?_)
    have hP0' := (show Extends Γ40 Γ43 by liftx).get hP0
    have hC0' := (show Extends Γ41 Γ43 by liftx).get hC0e
    have hc43 : Extends Γ Γ43 := by liftx
    have hv2 := ((hE.extend hc43).setVal p0 (hwf p0 (by simp [assigns])) vP0 _ hP0').setVal cand0
      (hwf cand0 (by simp [assigns])) vC0 _ hC0'
    by_cases hf : fl.contains found = true
    · simp only [if_pos hf]
      refine wpE_pure (hQ _ _ hc43 ?_)
      exact hv2.setFlag found hf vFd _ (by split <;> decide) (by rw [hFd]; rfl)
    · simp only [hf, Bool.false_eq_true, if_false]
      refine wpE_bind (wpE_iconst64 ?_)
      refine wpE_bind (wpE_iconst64 ?_)
      have hx44 := ((Extends.push Γ43 (.sc .i64 (c64 1))).push_of (.sc .i64 (c64 0)))
      refine wpE_op (eval_sel01 _ Γ43 vFd (if s32 (wordAt ans (collCalls st.w start).length) != 0 then 1 else 0)
        (by split <;> decide) (by rw [hFd]; rfl)) ?_
      refine wpE_pure (hQ _ _ ((hc43.trans hx44).push_of _) ?_)
      exact (hv2.extend (hx44.push_of _)).setVal found (by simpa using hf) _ _ (get_push_last _ _)


theorem copyL_done {d s n lv i : UInt64} {w : World} (h : ¬ i < n) : ∀ k, copyL d s n lv k i w = some w
  | 0 => by simp [copyL, h]
  | _ + 1 => by simp [copyL, h]

/-- **A copy's code ends as the lane's share of it.** -/
theorem coopL_copy {ans : List Answer} {start : Nat}
    {fl : List String} {sm lane : Slot .i64} {no : Slot .i8} {smv lv : UInt64}
    {dst src len : String} {e : RegEnv Slot} {st st' : LSt} {Q : RegEnv Slot → Env → World → Prop}
    (hE : EnvOk fl e st.regs Γ) (hS : Slots sm lane no smv lv Γ)
    (hr : coopL ans start smv lv (.coopCopy dst src len) st = some st')
    (hQ : ∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) :
    wpE cfg (codeOf fl sm lane no (.coopCopy dst src len) e) Q Γ st.w := by
  simp only [coopL] at hr
  obtain ⟨wEnd, hcp, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain rfl := (Option.some.inj hr).symm
  have hlane := hS.lane
  simp only [codeOf, codeOfT, leafCode, wloop1]
  refine wpE_bind (wpE_get hE fun vD Γ1 hx1 hD => ?_)
  refine wpE_bind (wpE_get (hE.extend hx1) fun vS Γ2 hx2 hSr => ?_)
  refine wpE_bind (wpE_get (hE.extend (hx1.trans hx2)) fun vN Γ3 hx3 hN => ?_)
  refine wpE_bind (xiconst64 fun vI0 Γ4 hx4 hI0 => ?_)
  have hD4 := (show Extends Γ1 Γ4 by liftx).get hD
  have hS4 := (show Extends Γ2 Γ4 by liftx).get hSr
  have hN4 := (show Extends Γ3 Γ4 by liftx).get hN
  have hc4 : Extends Γ Γ4 := by liftx
  have hL4 := hc4.get hlane
  clear hx1 hx2 hx3 hx4
  let F := (st.regs len).toNat + 1
  let R (cs : List V) (wi : World) (k : Nat) : Prop :=
    ∃ i, cs = [.sc .i64 i] ∧ copyL (st.regs dst) (st.regs src) (st.regs len) lv k i wi = some wEnd
  have hread : ∀ (Γ1 : Env) (i : UInt64), Extends (bindAt Γ4 Γ4.size [.sc .i64 i]) Γ1 →
      Γ1[Γ4.size]? = some (.sc .i64 i) := fun Γ1 i hx => hx.get (by simpa using bindAt_at Γ4 Γ4.size [.sc .i64 i] 0)
  have hkeep : ∀ (Γ1 : Env) (cs : List V) {x : Nat} {v : V}, Extends (bindAt Γ4 Γ4.size cs) Γ1 → Γ4[x]? = some v →
      Γ1[x]? = some v := fun Γ1 cs x v hx h => hx.get (get_bindAt (Nat.le_refl _) h)
  refine wpE_bind (wpE_wloopN (fun cs wi => ∃ k, k ≤ F ∧ R cs wi k) (fun cs wi => least (R cs wi) F)
    (fun cs wi a Γ1 w1 => Extends (bindAt Γ4 Γ4.size cs) Γ1 ∧ ∃ i, cs = [.sc .i64 i] ∧
      a = (contIf .ult Γ4.size vN, %[], ()) ∧ w1 = wi ∧ ∃ k, k ≤ F ∧
        copyL (st.regs dst) (st.regs src) (st.regs len) lv k i wi = some wEnd)
    (fun _ w => w = wEnd) ?_ ?_ ?_ ?_ ?_ ?_)
  · exact ⟨_, by simp [Vals.slots, hI0], F, Nat.le_refl _, c64 0, rfl, hcp⟩
  · rintro wi cs ⟨k, hk, i, rfl, hrun⟩
    exact wpE_pure ⟨Extends.refl _, Extends.refl _, i, rfl, rfl, rfl, k, hk, hrun⟩
  · rintro cs wi a Γ1 w1 ⟨hx, i, rfl, rfl, rfl, -⟩
    exact ⟨_, _, by simp only [contIf]; simp [evalOp, Sem.get, hread Γ1 i hx, hkeep Γ1 _ hx hN4, zipIntCmp,
      ClifTy.isInt]; rfl⟩
  · rintro cs wi a Γ1 w1 t f ⟨hx, i, rfl, rfl, rfl, k, hk, hrun⟩ hev hexit
    simp only [contIf] at hev hexit
    rw [show evalOp w1.mem Γ1 (.icmp .ult Γ4.size vN) = some (boolV (cmpInt .ult .i64 i (st.regs len))) by
      simp [evalOp, Sem.get, hread Γ1 i hx, hkeep Γ1 _ hx hN4, zipIntCmp, ClifTy.isInt]] at hev
    have hlt : ¬ i < st.regs len := by
      intro h
      simp [h, boolV, cmpInt] at hev
      obtain ⟨-, rfl⟩ := hev; simp at hexit
    rw [copyL_done hlt k] at hrun
    exact ⟨[], rfl, Option.some.inj hrun⟩
  · rintro cs wi a Γ1 wb t f ⟨hx, i, rfl, rfl, rfl, k, hk, hrun⟩ _ hev hcont
    simp only [contIf] at hev hcont
    rw [show evalOp wb.mem Γ1 (.icmp .ult Γ4.size vN) = some (boolV (cmpInt .ult .i64 i (st.regs len))) by
      simp [evalOp, Sem.get, hread Γ1 i hx, hkeep Γ1 _ hx hN4, zipIntCmp, ClifTy.isInt]] at hev
    have hlt : i < st.regs len := by
      refine Classical.byContradiction fun h => ?_
      simp [h, boolV, cmpInt] at hev
      obtain ⟨-, rfl⟩ := hev; simp at hcont
    obtain ⟨hR0, hmin⟩ := least_spec (P := R [.sc .i64 i] wb) ⟨i, rfl, hrun⟩ hk
    generalize hk0e : least (R [.sc .i64 i] wb) F = k0 at hR0 hmin
    obtain ⟨i', hi', hrun0⟩ := hR0
    obtain rfl : i' = i := by simpa using hi'.symm
    have hk0 : k0 ≤ k := hmin k ⟨i', rfl, hrun⟩
    cases k0 with
    | zero => simp [copyL, hlt] at hrun0
    | succ k' =>
    simp only [copyL, hlt, if_true] at hrun0
    obtain ⟨w1, hw1, hrun1⟩ := Option.bind_eq_some_iff.mp hrun0
    have hxa := (hx.push_of (.sc t f))
    have hxb : Extends (Γ1.push (.sc t f)) (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i']) :=
      (Extends.refl _).bindAt_of (by simp) _
    have hI : (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'])[Γ1.size + 1]? = some (.sc .i64 i') := by
      simpa using bindAt_at (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'] 0
    have hkb : ∀ {x : Nat} {v : V}, Γ4[x]? = some v →
        (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) [.sc .i64 i'])[x]? = some v := fun h =>
      hxb.get (hxa.get (get_bindAt (Nat.le_refl _) h))
    simp only [carriesFrom, Vals.head]
    refine wpE_bind (xiadd hI (hkb hL4) fun vJ Γ5 hx5 hJ => ?_)
    -- the byte loads whether or not it is stored
    have hload : ∃ v, wb.mem.load (st.regs src + (i' + lv)) 1 = some v ∧
        (i' + lv < st.regs len → stw wb (st.regs dst + (i' + lv)) 1 (v &&& 0xff) = some w1) ∧
        (¬ i' + lv < st.regs len → w1 = wb) := by
      simp only [copyStep] at hw1
      split at hw1
      · rename_i hj
        obtain ⟨v, hv, h⟩ := Option.bind_eq_some_iff.mp hw1
        exact ⟨v, hv, fun _ => h, fun hn => absurd hj hn⟩
      · rename_i hj
        obtain ⟨v, hv, h⟩ := Option.bind_eq_some_iff.mp hw1
        exact ⟨v, hv, fun hn => absurd hn hj, fun _ => (Option.some.inj h).symm⟩
    obtain ⟨v, hv, hst, hsk⟩ := hload
    refine wpE_bind (xiadd (hx5.get (hkb hS4)) hJ fun vA Γ6 hx6 hA => ?_)
    refine wpE_bind (xuload8 hA hv fun vV Γ7 hx7 hV => ?_)
    have hx57 : Extends Γ5 Γ7 := hx6.trans hx7
    refine wpE_bind (wpE_whenT (fun Γ' w' => Extends Γ5 Γ' ∧ w' = w1) rfl (hx57.get hJ)
      (hx57.get (hx5.get (hkb hN4))) ?_ ?_ ?_)
    · intro hc
      have hjl : i' + lv < st.regs len := by simpa [cmpInt] using hc
      have hw1 := hst hjl
      have hxT : Extends Γ7 (Γ7.push (boolV true)) := Extends.push _ _
      refine wpE_bind (xiadd (hxT.get (hx57.get (hx5.get (hkb hD4)))) (hxT.get (hx57.get hJ)) fun vB Γ8 hx8 hB => ?_)
      simp only [stw] at hw1
      obtain ⟨m, hm, hw1⟩ := Option.bind_eq_some_iff.mp hw1
      obtain rfl := Option.some.inj hw1
      refine wpE_istore8 (w' := { obsStore wb (st.regs dst + (i' + lv)) 1 (v &&& 0xff) with mem := m })
        (by simp [runStmt, Sem.get, hx8.get (hxT.get hV), hB, hm]) (wpE_ret ⟨hx57.trans (hxT.trans hx8), rfl⟩)
    · intro hc Γ' hx'
      have hjl : ¬ i' + lv < st.regs len := by simpa [cmpInt] using hc
      exact ⟨hx57.trans hx', (hsk hjl).symm⟩
    · rintro Γ6 w' ⟨hx6, hw'⟩ n hn
      have hw1' := hw'.symm
      subst hw1'
      have hx7 : Extends Γ6 (bindAt Γ6 n []) := (Extends.refl _).bindAt_of hn []
      refine wpE_bind (xiaddImm (by liftf hI) fun vI Γ8 hx8 hI8 => ?_)
      have hR2 : R [.sc .i64 (i' + c64 32)] w1 k' := ⟨_, rfl, hrun1⟩
      refine wpE_pure ⟨[.sc .i64 (i' + c64 32)], by simp [Vals.slots, hI8], ⟨k', by omega, hR2⟩, ?_⟩
      have := (least_spec hR2 (show k' ≤ F by omega)).2 k' hR2
      show least (R [.sc .i64 (i' + c64 32)] w1) F < least (R [.sc .i64 i'] wb) F
      rw [hk0e]; omega
  · rintro Γb vs w' nE hxb hle rfl
    exact wpE_pure (hQ _ _ ((hc4.trans hxb).bindAt (Nat.le_trans hc4.1 hle) _)
      (hE.extend ((hc4.trans hxb).bindAt (Nat.le_trans hc4.1 hle) _)))


/-- A cooperative statement changes only the registers it sets. -/
theorem coopL_frame (ans : List Answer) (start : Nat) (smv lv : UInt64) :
    ∀ s st st', coopL ans start smv lv s st = some st' → ∀ r, r ∉ setsOf s → st'.regs r = st.regs r := by
  intro s st st' hr r hn
  cases s with
  | coopCopy dst src len =>
      simp only [coopL] at hr
      obtain ⟨w, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain rfl := (Option.some.inj hr).symm
      rfl
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      simp only [coopL] at hr
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun h => by simp [h] at hr
      simp only [hs, if_true] at hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      split at hr
      · obtain rfl := (Option.some.inj hr).symm
        simp only [setsOf, List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
        simp [LSt.set, hn.1, hn.2.1, hn.2.2]
      · cases hr
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      simp only [coopL] at hr
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun h => by simp [h] at hr
      simp only [hs, if_true] at hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain ⟨_, -, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain rfl := (Option.some.inj hr).symm
      simp only [setsOf, List.mem_cons, List.not_mem_nil, or_false] at hn
      simp [LSt.set, hn]
  | _ => simp [coopL] at hr

/-- **The code a statement writes ends as one lane's run of it**, its
    collectives answered from the lane's answers as a thread's are. -/
theorem codeOf_lane_coop {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg) (hW : Words ans)
    (fl : List String) (sm lane : Slot .i64) (no : Slot .i8) (smv lv : UInt64) :
    ∀ (s : WStmt) (f : Nat) (e : RegEnv Slot) (Γ : Env) (st st' : LSt) (Q : RegEnv Slot → Env → World → Prop),
      Wf fl s → EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ →
      laneRun (coopL ans start smv lv) smv f s st = some st' →
      (∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) →
      wpE cfg (codeOf fl sm lane no s e) Q Γ st.w :=
  codeOf_lane (coopL ans start smv lv) fl sm lane no smv lv
    (fun s e Γ st st' Q hm hwf hE hS hr hQ => by
      cases s with
      | coopCopy => exact coopL_copy hE hS hr hQ
      | coopWindow => exact coopL_window hA hW hwf hE hS hr hQ
      | coopExtendStep => exact coopL_extend hA hW hwf hE hS hr hQ
      | _ => cases hm)
    (coopL_frame ans start smv lv)


end

end Lz4Code
