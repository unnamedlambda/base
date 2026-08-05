import AlgorithmLib.LZ4Ptx

namespace AlgorithmLib.LZ4Simt
open AlgorithmLib

/-- Warp width. -/
abbrev W : Nat := 32
abbrev Lane := Fin W

-- ── Instruction set (the warp-uniform fragment the kernels emit) ──────────────

inductive SArg where
  | reg (n : String)
  | imm (v : Nat)
  deriving Repr, DecidableEq

inductive SOp where
  | add | sub | band | bor | shl | shr | min | mul
  deriving Repr, DecidableEq

inductive SCmp where
  | eq | ne | lt | le | ge
  deriving Repr, DecidableEq

inductive SInstr where
  | mov   (d : String) (a : SArg)
  | bin   (o : SOp) (d a : String) (b : SArg)   -- reg ⊕ (reg|imm)
  | binr  (o : SOp) (d a b : String)            -- reg ⊕ reg (min, mul, var-shift)
  | cvt32 (d a : String)                         -- d = a & 0xffffffff
  | brev  (d a : String)                         -- brev.b32 (low 32 bits)
  | clz   (d a : String)                         -- clz.b32 (low 32 bits)
  | bnot  (d a : String)                         -- not.b32 (bitwise complement)
  | setp  (c : SCmp) (p a : String) (b : SArg)
  | andp  (d a b : String)                       -- and.pred
  | selp  (d a b p : String)                     -- d = (p ? a : b), per lane
  | ldgo  (d addr : String) (off : Nat)          -- ld.global.u8 [addr+off]
  | ldgop (p d addr : String) (off : Nat)        -- @p ld.global.u8 [addr+off]
  | stg   (addr s : String)                      -- st.global.u8 (all lanes)
  | stgp  (p addr s : String)                    -- @p st.global.u8
  | stg32p (p addr s : String)                   -- @p st.global.u32 (LE)
  | ldsh  (d addr : String)                      -- ld.shared.u16
  | stsh  (addr s : String)                      -- st.shared.u16 (all lanes)
  | stshp (p addr s : String)                    -- @p st.shared.u16
  | vote  (d p : String)                         -- vote.sync.ballot.b32
  | shfl  (d s src : String)                     -- shfl.sync.idx.b32
  | barwarp                                      -- bar.warp.sync (no-op)
  | braif    (p lbl : String)
  | braifnot (p lbl : String)
  | bra   (lbl : String)
  | lbl   (name : String)
  | ret
  deriving Repr, DecidableEq

-- ── Bit primitives (32-bit brev/clz; item-2 facts reason about these) ──────────

/-- Reverse the low 32 bits (`brev.b32`).  Routed through `Nat.testBit` (faithful;
    validated concretely in `LZ4SimtBits`) so the bit-facts have a clean core. -/
def brev32 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((List.range 32).foldl
    (fun acc i => if x.toNat.testBit i then acc ||| 2 ^ (31 - i) else acc) 0)

/-- Count leading zeros of the low 32 bits (`clz.b32`); 32 if all clear. -/
def clz32 (x : UInt64) : UInt64 :=
  UInt64.ofNat (match (List.range 32).reverse.find? (fun i => x.toNat.testBit i) with
    | some hi => 31 - hi
    | none => 32)

-- ── Machine state and small-step semantics ────────────────────────────────────

structure SState where
  regs : String → Lane → UInt64
  gmem : Array UInt8
  smem : Array UInt8
  pc   : Nat

def SState.get (st : SState) (l : Lane) : SArg → UInt64
  | .reg n => st.regs n l
  | .imm v => UInt64.ofNat v

def SState.setReg (st : SState) (d : String) (f : Lane → UInt64) : SState :=
  { st with regs := fun x => if x = d then f else st.regs x }

def SState.setPc (st : SState) (n : Nat) : SState := { st with pc := n }

def SOp.run : SOp → UInt64 → UInt64 → UInt64
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | .band, a, b => a &&& b
  | .bor, a, b => a ||| b
  | .shl, a, b => a <<< b
  | .shr, a, b => a >>> b
  | .min, a, b => if a ≤ b then a else b
  | .mul, a, b => a * b

def SCmp.run : SCmp → UInt64 → UInt64 → Bool
  | .eq, a, b => a == b
  | .ne, a, b => a != b
  | .lt, a, b => a < b
  | .le, a, b => a ≤ b
  | .ge, a, b => b ≤ a

-- `List.findIdx` (structural) rather than `Array.findIdx?` so label pcs reduce by
-- `rfl` symbolically over abstract-config programs (Array.findIdx? does not).
def sfindLabel (prog : Array SInstr) (name : String) : Nat :=
  prog.toList.findIdx (fun i => match i with | .lbl n => n == name | _ => false)

/-- The source lane of a `shfl.idx`: low 5 bits of the requested index. -/
def toLane (v : UInt64) : Lane := ⟨v.toNat % W, Nat.mod_lt _ (by decide)⟩

/-- The ballot of predicate register `p`: bit `l` set iff lane `l` holds 1. -/
def ballotOf (regs : String → Lane → UInt64) (p : String) : UInt64 :=
  UInt64.ofNat ((List.finRange W).foldl
    (fun acc l => if regs p l == 1 then acc ||| 2 ^ l.val else acc) 0)

/-- Apply a per-lane byte store to memory in lane order (later lane wins). -/
def storeBytes (mem : Array UInt8) (pred : Lane → Bool) (addr val : Lane → UInt64) :
    Array UInt8 :=
  (List.finRange W).foldl
    (fun m l => if pred l then m.set! (addr l).toNat (val l).toUInt8 else m) mem

/-- One non-`ret` instruction; lockstep across all lanes. -/
def sstepInstr (prog : Array SInstr) (i : SInstr) (st : SState) : SState :=
  match i with
  | .mov d a => (st.setReg d (fun l => st.get l a)).setPc (st.pc + 1)
  | .bin o d a b => (st.setReg d (fun l => o.run (st.regs a l) (st.get l b))).setPc (st.pc + 1)
  | .binr o d a b => (st.setReg d (fun l => o.run (st.regs a l) (st.regs b l))).setPc (st.pc + 1)
  | .cvt32 d a => (st.setReg d (fun l => st.regs a l &&& 0xffffffff)).setPc (st.pc + 1)
  | .brev d a => (st.setReg d (fun l => brev32 (st.regs a l))).setPc (st.pc + 1)
  | .clz d a => (st.setReg d (fun l => clz32 (st.regs a l))).setPc (st.pc + 1)
  | .bnot d a => (st.setReg d (fun l => ~~~ (st.regs a l))).setPc (st.pc + 1)
  | .setp c p a b =>
      (st.setReg p (fun l => if c.run (st.regs a l) (st.get l b) then 1 else 0)).setPc (st.pc + 1)
  | .andp d a b =>
      (st.setReg d (fun l => if st.regs a l == 1 ∧ st.regs b l == 1 then 1 else 0)).setPc (st.pc + 1)
  | .selp d a b p =>
      (st.setReg d (fun l => if st.regs p l == 1 then st.regs a l else st.regs b l)).setPc (st.pc + 1)
  | .ldgo d addr off =>
      (st.setReg d (fun l =>
        UInt64.ofNat (st.gmem.getD ((st.regs addr l).toNat + off) 0).toNat)).setPc (st.pc + 1)
  | .ldgop p d addr off =>
      -- a masked lane does not read: it keeps whatever `d` held
      (st.setReg d (fun l =>
        if st.regs p l == 1 then
          UInt64.ofNat (st.gmem.getD ((st.regs addr l).toNat + off) 0).toNat
        else st.regs d l)).setPc (st.pc + 1)
  | .stg addr s =>
      { st with gmem := storeBytes st.gmem (fun _ => true) (st.regs addr) (st.regs s),
                pc := st.pc + 1 }
  | .stgp p addr s =>
      { st with gmem := storeBytes st.gmem (fun l => st.regs p l == 1) (st.regs addr) (st.regs s),
                pc := st.pc + 1 }
  | .stg32p p addr s =>
      -- 4-byte little-endian store per predicated lane
      let g0 := storeBytes st.gmem (fun l => st.regs p l == 1) (st.regs addr) (st.regs s)
      let g1 := storeBytes g0 (fun l => st.regs p l == 1)
                  (fun l => st.regs addr l + 1) (fun l => st.regs s l >>> 8)
      let g2 := storeBytes g1 (fun l => st.regs p l == 1)
                  (fun l => st.regs addr l + 2) (fun l => st.regs s l >>> 16)
      let g3 := storeBytes g2 (fun l => st.regs p l == 1)
                  (fun l => st.regs addr l + 3) (fun l => st.regs s l >>> 24)
      { st with gmem := g3, pc := st.pc + 1 }
  | .ldsh d addr =>
      (st.setReg d (fun l =>
        let a := (st.regs addr l).toNat
        UInt64.ofNat ((st.smem.getD a 0).toNat + 256 * (st.smem.getD (a + 1) 0).toNat))).setPc (st.pc + 1)
  | .stsh addr s =>
      -- 2-byte little-endian shared store, all lanes
      let s0 := storeBytes st.smem (fun _ => true) (st.regs addr) (st.regs s)
      let s1 := storeBytes s0 (fun _ => true)
                  (fun l => st.regs addr l + 1) (fun l => st.regs s l >>> 8)
      { st with smem := s1, pc := st.pc + 1 }
  | .stshp p addr s =>
      -- 2-byte little-endian shared store per predicated lane
      let s0 := storeBytes st.smem (fun l => st.regs p l == 1) (st.regs addr) (st.regs s)
      let s1 := storeBytes s0 (fun l => st.regs p l == 1)
                  (fun l => st.regs addr l + 1) (fun l => st.regs s l >>> 8)
      { st with smem := s1, pc := st.pc + 1 }
  | .vote d p =>
      let b := ballotOf st.regs p
      (st.setReg d (fun _ => b)).setPc (st.pc + 1)
  | .shfl d s src =>
      (st.setReg d (fun l => st.regs s (toLane (st.regs src l)))).setPc (st.pc + 1)
  | .barwarp => st.setPc (st.pc + 1)
  | .braif p l => st.setPc (if st.regs p 0 == 1 then sfindLabel prog l else st.pc + 1)
  | .braifnot p l => st.setPc (if st.regs p 0 == 1 then st.pc + 1 else sfindLabel prog l)
  | .bra l => st.setPc (sfindLabel prog l)
  | .lbl _ => st.setPc (st.pc + 1)
  | .ret => st

end AlgorithmLib.LZ4Simt
