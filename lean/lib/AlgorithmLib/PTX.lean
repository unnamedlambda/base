import Lean
import Std
open Lean

namespace AlgorithmLib.PTX

/-!
  PTX assembly DSL for NVIDIA GPU kernels.

  Phantom-typed register references prevent mixing u32/u64/f32.
  The builder accumulates instruction strings; `buildKernel` prepends
  the register-count header once the body is fully known.
-/

inductive RK | pred | u32 | u64 | f32

structure Reg (k : RK) where
  raw : String

-- Special system registers
def tidX  : Reg .u32 := ⟨"%tid.x"⟩
def tidY  : Reg .u32 := ⟨"%tid.y"⟩
def ctaX  : Reg .u32 := ⟨"%ctaid.x"⟩
def ctaY  : Reg .u32 := ⟨"%ctaid.y"⟩
def ntidX : Reg .u32 := ⟨"%ntid.x"⟩
def ntidY : Reg .u32 := ⟨"%ntid.y"⟩

structure PTXState where
  nPred : Nat := 0
  nR    : Nat := 0
  nRd   : Nat := 0
  nF    : Nat := 0
  lines : Array String := #[]

abbrev PTX := StateM PTXState

private def emit (s : String) : PTX Unit :=
  modify fun st => { st with lines := st.lines.push ("    " ++ s ++ ";") }

def label (name : String) : PTX Unit :=
  modify fun st => { st with lines := st.lines.push (name ++ ":") }

-- Register allocation
def freshP : PTX (Reg .pred) := do
  let s ← get; set { s with nPred := s.nPred + 1 }; return ⟨s!"%p{s.nPred}"⟩

def freshR : PTX (Reg .u32) := do
  let s ← get; set { s with nR := s.nR + 1 }; return ⟨s!"%r{s.nR}"⟩

def freshRd : PTX (Reg .u64) := do
  let s ← get; set { s with nRd := s.nRd + 1 }; return ⟨s!"%rd{s.nRd}"⟩

def freshF : PTX (Reg .f32) := do
  let s ← get; set { s with nF := s.nF + 1 }; return ⟨s!"%f{s.nF}"⟩

-- Float constant formatting (PTX hex float literal)
def f32Bits (bits : UInt32) : String :=
  let h i :=
    let d := (bits.toNat >>> (i * 4)) &&& 0xf
    if d < 10 then Char.ofNat (d + '0'.toNat) else Char.ofNat (d - 10 + 'a'.toNat)
  s!"0f{h 7}{h 6}{h 5}{h 4}{h 3}{h 2}{h 1}{h 0}"

-- Common float constants
def f32_0     : UInt32 := 0x00000000  -- 0.0f
def f32_1     : UInt32 := 0x3f800000  -- 1.0f
def f32_log2e : UInt32 := 0x3fb8aa3b  -- log2(e)
def f32_eps   : UInt32 := 0x3727c5ac  -- ~1e-5 (RMS norm epsilon)

-- ── Load / Store ──────────────────────────────────────────────────────────────

def ldParam64 (dst : Reg .u64) (p : String)  : PTX Unit := emit s!"ld.param.u64 {dst.raw}, [{p}]"
def ldGlobalF (dst : Reg .f32) (a : Reg .u64): PTX Unit := emit s!"ld.global.f32 {dst.raw}, [{a.raw}]"
def ldGlobalFO (dst : Reg .f32) (a : Reg .u64) (off : Int) : PTX Unit :=
  emit s!"ld.global.f32 {dst.raw}, [{a.raw}+{off}]"
def ldGlobalU  (dst : Reg .u32) (a : Reg .u64): PTX Unit := emit s!"ld.global.u32 {dst.raw}, [{a.raw}]"
def ldGlobalUO (dst : Reg .u32) (a : Reg .u64) (off : Nat) : PTX Unit :=
  emit s!"ld.global.u32 {dst.raw}, [{a.raw}+{off}]"
def ldGlobalU64 (dst : Reg .u64) (a : Reg .u64): PTX Unit := emit s!"ld.global.u64 {dst.raw}, [{a.raw}]"
def ldGlobalS64 (dst : Reg .u64) (a : Reg .u64): PTX Unit := emit s!"ld.global.s64 {dst.raw}, [{a.raw}]"
def stGlobalF  (a : Reg .u64) (v : Reg .f32)  : PTX Unit := emit s!"st.global.f32 [{a.raw}], {v.raw}"
def stGlobalFO (a : Reg .u64) (off : Int) (v : Reg .f32) : PTX Unit :=
  emit s!"st.global.f32 [{a.raw}+{off}], {v.raw}"
def stGlobalU64 (a : Reg .u64) (v : Reg .u64) : PTX Unit := emit s!"st.global.u64 [{a.raw}], {v.raw}"
def stGlobalS64 (a : Reg .u64) (v : Reg .u64) : PTX Unit := emit s!"st.global.s64 [{a.raw}], {v.raw}"
def ldSharedF  (dst : Reg .f32) (b : Reg .u32) (off : Nat) : PTX Unit :=
  if off = 0 then emit s!"ld.shared.f32 {dst.raw}, [{b.raw}]"
  else emit s!"ld.shared.f32 {dst.raw}, [{b.raw}+{off}]"
def ldSharedFD (dst : Reg .f32) (a : Reg .u32): PTX Unit := emit s!"ld.shared.f32 {dst.raw}, [{a.raw}]"
def stSharedF  (b : Reg .u32) (off : Nat) (v : Reg .f32) : PTX Unit :=
  if off = 0 then emit s!"st.shared.f32 [{b.raw}], {v.raw}"
  else emit s!"st.shared.f32 [{b.raw}+{off}], {v.raw}"
def stSharedFD (a : Reg .u32) (v : Reg .f32)  : PTX Unit := emit s!"st.shared.f32 [{a.raw}], {v.raw}"

-- ── Integer (u32) ─────────────────────────────────────────────────────────────

def movR  (d s : Reg .u32)               : PTX Unit := emit s!"mov.u32 {d.raw}, {s.raw}"
def movRC (d : Reg .u32) (n : Nat)       : PTX Unit := emit s!"mov.u32 {d.raw}, {n}"
def addR  (d a b : Reg .u32)             : PTX Unit := emit s!"add.u32 {d.raw}, {a.raw}, {b.raw}"
def addRI (d a : Reg .u32) (n : Nat)     : PTX Unit := emit s!"add.u32 {d.raw}, {a.raw}, {n}"
def mulLoR (d a b : Reg .u32)            : PTX Unit := emit s!"mul.lo.u32 {d.raw}, {a.raw}, {b.raw}"
def madLoS (d a b c : Reg .u32)          : PTX Unit := emit s!"mad.lo.s32 {d.raw}, {a.raw}, {b.raw}, {c.raw}"
def madLoU (d a b c : Reg .u32)          : PTX Unit := emit s!"mad.lo.u32 {d.raw}, {a.raw}, {b.raw}, {c.raw}"
def madLoRC (d a : Reg .u32) (b : Nat) (c : Reg .u32) : PTX Unit :=
  emit s!"mad.lo.u32 {d.raw}, {a.raw}, {b}, {c.raw}"
def shlR  (d a : Reg .u32) (n : Nat)     : PTX Unit := emit s!"shl.b32 {d.raw}, {a.raw}, {n}"
def shrR  (d a : Reg .u32) (n : Nat)     : PTX Unit := emit s!"shr.u32 {d.raw}, {a.raw}, {n}"
def andR  (d a : Reg .u32) (n : Nat)     : PTX Unit := emit s!"and.b32 {d.raw}, {a.raw}, {n}"

-- ── Integer (u64) ─────────────────────────────────────────────────────────────

def movRd  (d s : Reg .u64)              : PTX Unit := emit s!"mov.u64 {d.raw}, {s.raw}"
def addRd  (d a b : Reg .u64)            : PTX Unit := emit s!"add.u64 {d.raw}, {a.raw}, {b.raw}"
def addRdI (d a : Reg .u64) (n : Nat)   : PTX Unit := emit s!"add.u64 {d.raw}, {a.raw}, {n}"
def shlRd  (d a : Reg .u64) (n : Nat)   : PTX Unit := emit s!"shl.b64 {d.raw}, {a.raw}, {n}"
def mulLoRd (d a b : Reg .u64)           : PTX Unit := emit s!"mul.lo.u64 {d.raw}, {a.raw}, {b.raw}"
def addS64  (d a b : Reg .u64)           : PTX Unit := emit s!"add.s64 {d.raw}, {a.raw}, {b.raw}"
def mulLoS64 (d a b : Reg .u64)          : PTX Unit := emit s!"mul.lo.s64 {d.raw}, {a.raw}, {b.raw}"

-- ── Conversions ───────────────────────────────────────────────────────────────

def cvtU64 (d : Reg .u64) (s : Reg .u32): PTX Unit := emit s!"cvt.u64.u32 {d.raw}, {s.raw}"
def cvtS64 (d : Reg .u64) (s : Reg .u32): PTX Unit := emit s!"cvt.s64.s32 {d.raw}, {s.raw}"
def cvtF32 (d : Reg .f32) (s : Reg .u32): PTX Unit := emit s!"cvt.rn.f32.u32 {d.raw}, {s.raw}"

-- ── Float arithmetic ──────────────────────────────────────────────────────────

def movF  (d s : Reg .f32)               : PTX Unit := emit s!"mov.f32 {d.raw}, {s.raw}"
def movFC (d : Reg .f32) (bits : UInt32) : PTX Unit := emit s!"mov.f32 {d.raw}, {f32Bits bits}"
def addF  (d a b : Reg .f32)             : PTX Unit := emit s!"add.f32 {d.raw}, {a.raw}, {b.raw}"
def subF  (d a b : Reg .f32)             : PTX Unit := emit s!"sub.f32 {d.raw}, {a.raw}, {b.raw}"
def mulF  (d a b : Reg .f32)             : PTX Unit := emit s!"mul.f32 {d.raw}, {a.raw}, {b.raw}"
def maxF  (d a b : Reg .f32)             : PTX Unit := emit s!"max.f32 {d.raw}, {a.raw}, {b.raw}"
def negF  (d a : Reg .f32)               : PTX Unit := emit s!"neg.f32 {d.raw}, {a.raw}"
def fmaRn (d a b c : Reg .f32)           : PTX Unit := emit s!"fma.rn.f32 {d.raw}, {a.raw}, {b.raw}, {c.raw}"
def divRn (d a b : Reg .f32)             : PTX Unit := emit s!"div.rn.f32 {d.raw}, {a.raw}, {b.raw}"
def ex2   (d a : Reg .f32)               : PTX Unit := emit s!"ex2.approx.f32 {d.raw}, {a.raw}"
def rcp   (d a : Reg .f32)               : PTX Unit := emit s!"rcp.approx.f32 {d.raw}, {a.raw}"
def rsqrt (d a : Reg .f32)               : PTX Unit := emit s!"rsqrt.approx.f32 {d.raw}, {a.raw}"

-- ── Control flow ──────────────────────────────────────────────────────────────

def setpGe  (p : Reg .pred) (a b : Reg .u32)  : PTX Unit := emit s!"setp.ge.u32 {p.raw}, {a.raw}, {b.raw}"
def setpGeI (p : Reg .pred) (a : Reg .u32) (n : Nat) : PTX Unit := emit s!"setp.ge.u32 {p.raw}, {a.raw}, {n}"
def setpLt  (p : Reg .pred) (a b : Reg .u32)  : PTX Unit := emit s!"setp.lt.u32 {p.raw}, {a.raw}, {b.raw}"
def setpLtI (p : Reg .pred) (a : Reg .u32) (n : Nat) : PTX Unit := emit s!"setp.lt.u32 {p.raw}, {a.raw}, {n}"
def setpNe0 (p : Reg .pred) (a : Reg .u32)    : PTX Unit := emit s!"setp.ne.u32 {p.raw}, {a.raw}, 0"
def andPred (d a b : Reg .pred)                : PTX Unit := emit s!"and.pred {d.raw}, {a.raw}, {b.raw}"
def braIf   (p : Reg .pred) (lbl : String)    : PTX Unit :=
  modify fun st => { st with lines := st.lines.push s!"    @{p.raw} bra {lbl};" }
def braIfNot (p : Reg .pred) (lbl : String)   : PTX Unit :=
  modify fun st => { st with lines := st.lines.push s!"    @!{p.raw} bra {lbl};" }
def bra     (lbl : String)                     : PTX Unit :=
  modify fun st => { st with lines := st.lines.push s!"    bra {lbl};" }
def barSync : PTX Unit := emit "bar.sync 0"
def ptxRet  : PTX Unit := modify fun st => { st with lines := st.lines.push "    ret;" }

-- Warp shuffle (bfly reduction step)
def shflBfly (dst src : Reg .f32) (mask : Nat) : PTX Unit :=
  emit s!"shfl.sync.bfly.b32 {dst.raw}, {src.raw}, {mask}, 31, 0xffffffff"

-- ── Common combinators ────────────────────────────────────────────────────────

-- Load a u64 param and return its register
def ldParam (name : String) : PTX (Reg .u64) := do
  let r ← freshRd; ldParam64 r name; return r

-- Standard warp-ID decomposition: (tid, warpId, laneId)
def getWarpIds : PTX (Reg .u32 × Reg .u32 × Reg .u32) := do
  let tid ← freshR;    movR tid tidX
  let wid ← freshR;    shrR wid tid 5
  let lid ← freshR;    andR lid tid 31
  return (tid, wid, lid)

-- Compute element pointer: base + idx*4
def elemAddr (base : Reg .u64) (idx : Reg .u32) : PTX (Reg .u64) := do
  let i ← freshRd; cvtU64 i idx
  let b ← freshRd; shlRd b i 2
  let a ← freshRd; addRd a base b
  return a

-- Get smem base address into a fresh u32 register (uses default _smem name)
def smemBase : PTX (Reg .u32) := do
  let r ← freshR; emit s!"mov.u32 {r.raw}, _smem"; return r

-- Get smem base with a custom name (for kernels using non-default smem declarations)
def smemBaseNamed (name : String) : PTX (Reg .u32) := do
  let r ← freshR; emit s!"mov.u32 {r.raw}, {name}"; return r

-- Warp-level butterfly: 5 steps (16,8,4,2,1), applying op(acc,acc,tmp)
def warpButterfly (acc tmp : Reg .f32) (op : Reg .f32 → Reg .f32 → Reg .f32 → PTX Unit) : PTX Unit :=
  for mask in [16, 8, 4, 2, 1] do
    shflBfly tmp acc mask
    op acc acc tmp

-- Warp reduce: sum
def warpReduceSum (acc tmp : Reg .f32) : PTX Unit := warpButterfly acc tmp addF

-- Warp reduce: max
def warpReduceMax (acc tmp : Reg .f32) : PTX Unit := warpButterfly acc tmp maxF

-- Warp reduce: online softmax (max, sum) pair
-- acc0=max, acc1=sum, t0/t1=shfl temps, nm/adj=scratch, log2e=log2e constant
def warpReduceOnline (acc0 acc1 t0 t1 nm adj log2e : Reg .f32) : PTX Unit := do
  for mask in [16, 8, 4, 2, 1] do
    shflBfly t0 acc0 mask; shflBfly t1 acc1 mask
    maxF nm acc0 t0
    subF adj acc0 nm; mulF adj adj log2e; ex2 adj adj; mulF acc1 acc1 adj
    subF adj t0   nm; mulF adj adj log2e; ex2 adj adj; mulF t1   t1   adj
    addF acc1 acc1 t1; movF acc0 nm

-- Lane 0 writes warp result(s) to smem[warpId*4 + extraOff], then bar.sync
-- `writes smemAddr` is called only by lane 0; smemAddr = _smem (no warp offset applied)
def lane0WriteSmem (laneId warpId : Reg .u32) (skipLbl : String)
    (writes : Reg .u32 → PTX Unit) : PTX Unit := do
  let p ← freshP; setpNe0 p laneId; braIf p skipLbl
  let sBase ← smemBase
  let off ← freshR; shlR off warpId 2
  let addr ← freshR; addR addr sBase off
  writes addr
  label skipLbl; barSync

-- Thread 0 operation + bar.sync
def thread0Op (tid : Reg .u32) (skipLbl : String) (op : PTX Unit) : PTX Unit := do
  let p ← freshP; setpNe0 p tid; braIf p skipLbl
  op; label skipLbl; barSync

-- Cross-warp reduce: 8 warp results at sBase[0,4,...,28] + baseOff
-- Accumulates into acc using op(acc, acc, tmp); returns acc
def crossWarp8 (acc tmp : Reg .f32) (sBase : Reg .u32) (baseOff : Nat)
    (op : Reg .f32 → Reg .f32 → Reg .f32 → PTX Unit) : PTX Unit := do
  ldSharedF acc sBase baseOff
  for i in List.range 7 do
    ldSharedF tmp sBase (baseOff + (i + 1) * 4)
    op acc acc tmp

-- Cross-warp online softmax reduce (8 warps)
-- maxes at sBase[mOff+0..28], sums at sBase[sOff+0..28]
def crossWarp8Online (mAcc sAcc mTmp sTmp nm adj log2e : Reg .f32)
    (sBase : Reg .u32) (mOff sOff : Nat) : PTX Unit := do
  ldSharedF mAcc sBase mOff
  ldSharedF sAcc sBase sOff
  for i in List.range 7 do
    ldSharedF mTmp sBase (mOff + (i + 1) * 4)
    ldSharedF sTmp sBase (sOff + (i + 1) * 4)
    maxF nm mAcc mTmp
    subF adj mAcc nm; mulF adj adj log2e; ex2 adj adj; mulF sAcc sAcc adj
    subF adj mTmp nm; mulF adj adj log2e; ex2 adj adj; mulF sTmp sTmp adj
    addF sAcc sAcc sTmp; movF mAcc nm

-- Stride loop: init i=tid, loop body, i+=stride, repeat; returns loop-index register
def strideLoop (tid : Reg .u32) (n : Reg .u32) (stride : Nat)
    (loopLbl doneLbl : String) (body : Reg .u32 → PTX Unit) : PTX Unit := do
  let p ← freshP
  let i ← freshR; movR i tid
  label loopLbl
  setpGe p i n; braIf p doneLbl
  body i
  addRI i i stride; bra loopLbl
  label doneLbl

-- Grid-stride setup: returns (global_idx, out-of-bounds pred)
def gridStrideSetup (n : Reg .u32) (oobLbl : String) : PTX (Reg .u32 × Reg .pred) := do
  let bid ← freshR; movR bid ctaX
  let bsz ← freshR; movR bsz ntidX
  let tid ← freshR; movR tid tidX
  let gid ← freshR; madLoS gid bid bsz tid
  let p ← freshP; setpGe p gid n; braIf p oobLbl
  return (gid, p)

-- ── Reusable kernel building blocks ──────────────────────────────────────────

-- Core RMS norm body given registers for pointers, thread IDs, count, and n-as-float bits
-- 1-block, 256-thread kernel; smem must be ≥ 36 bytes
def rmsNormBody (xPtr wPtr yPtr : Reg .u64) (tid warpId laneId : Reg .u32)
    (nReg : Reg .u32) (nAsBits : UInt32) (pfx : String) : PTX Unit := do
  -- loop1: accumulate sum of squares
  let acc ← freshF; movFC acc f32_0
  let tmp ← freshF
  strideLoop tid nReg 256 (pfx ++ "loop1") (pfx ++ "done1") fun i => do
    let addr ← elemAddr xPtr i
    ldGlobalF tmp addr; fmaRn acc tmp tmp acc
  warpReduceSum acc tmp
  lane0WriteSmem laneId warpId (pfx ++ "skip1") fun wAddr => stSharedFD wAddr acc
  -- thread 0: sum 8 warps, divide, add eps, rsqrt
  thread0Op tid (pfx ++ "skip2") do
    let sBase ← smemBase
    let total ← freshF
    crossWarp8 total tmp sBase 0 addF
    let nf ← freshF; movFC nf nAsBits
    divRn total total nf
    let eps ← freshF; movFC eps f32_eps
    addF total total eps; rsqrt total total
    stSharedF sBase 32 total
  -- loop2: normalize
  let sBase2 ← smemBase
  let scale ← freshF; ldSharedF scale sBase2 32
  strideLoop tid nReg 256 (pfx ++ "loop2") (pfx ++ "done2") fun j => do
    let xAddr ← elemAddr xPtr j
    let wAddr ← elemAddr wPtr j
    let yAddr ← elemAddr yPtr j
    let xi ← freshF; ldGlobalF xi xAddr
    let wi ← freshF; ldGlobalF wi wAddr
    mulF xi xi scale; mulF xi xi wi; stGlobalF yAddr xi

-- ── Module assembly ───────────────────────────────────────────────────────────

structure KernelSpec where
  name   : String
  params : List String   -- all u64 (pointer) params
  body   : PTX Unit

structure PTXModuleConfig where
  version   : String := "8.0"
  target    : String := "sm_86"
  smemSize  : Nat    := 0
  smemAlign : Nat    := 4
  smemName  : String := "_smem"

def buildKernel (k : KernelSpec) : String :=
  let (_, st) := k.body.run {}
  let paramList := String.intercalate ",\n" (k.params.map fun p => "    .param .u64 " ++ p)
  let regDecls :=
    (if st.nPred > 0 then "    .reg .pred %p<" ++ toString st.nPred ++ ">;\n" else "") ++
    (if st.nR  > 0 then "    .reg .u32  %r<" ++ toString st.nR  ++ ">;\n" else "") ++
    (if st.nRd > 0 then "    .reg .u64  %rd<" ++ toString st.nRd ++ ">;\n" else "") ++
    (if st.nF  > 0 then "    .reg .f32  %f<" ++ toString st.nF  ++ ">;\n" else "")
  let body := st.lines.toList.foldl (fun acc l => acc ++ l ++ "\n") ""
  ".visible .entry " ++ k.name ++ "(\n" ++ paramList ++ "\n)\n{\n" ++ regDecls ++ "\n" ++ body ++ "}\n"

def buildModuleWith (cfg : PTXModuleConfig) (kernels : List KernelSpec) : String :=
  let hdr := s!".version {cfg.version}\n.target {cfg.target}\n.address_size 64\n\n"
  let smem := if cfg.smemSize > 0
    then s!".shared .align {cfg.smemAlign} .b8 {cfg.smemName}[{cfg.smemSize}];\n\n"
    else ""
  hdr ++ smem ++ String.intercalate "\n" (kernels.map buildKernel)

def buildModule (smemSize : Nat) (kernels : List KernelSpec) : String :=
  buildModuleWith { smemSize } kernels


end AlgorithmLib.PTX
