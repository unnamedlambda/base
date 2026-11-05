module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
public import Bench.ImageAsm
meta import Bench.ImageAsm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace ImageBench

/-!
  # A 4K video frame downscaled to 1080p

  `resize` takes a 3840x2160 YUV 4:2:0 frame and writes the 1920x1080 one: each
  plane halved along both axes by an antialiased bilinear filter, whose taps
  along an axis are `1 3 3 1` over the four source pixels nearest an output's
  centre. A tap past the edge is dropped and the rest renormalised, so an
  output is `sum w_r w_c p(r, c) / sum w_r w_c`, rounded half up.

  The taps are separable: along each source row they give 16-bit sums, and an
  output row combines four rows of sums. `resize` makes an output row with one
  call of AVX2 machine code (`ImageAsm.rowpair`), mapped once by `asm_load`
  when the CPU has AVX2: it sums the two source rows the row adds, keeping the
  previous two rows' sums, and combines all four. `resize_clif`, and `resize`
  where the machine code did not load, do the same in CLIF with two loops,
  `hrow` for a row's sums, kept in a ring of four rows, and `vcol` to combine
  them. The outermost ring of outputs is computed one pixel at a time, each
  with its own constant weights: the top row first, each other row's two ends
  just after its interior, while its source rows are in cache, and the bottom
  row last.

  The sums and the loops' arguments live in the caller's output, after the
  frame, so calls on different outputs may run at once.
-/

-- ---------------------------------------------------------------------------
-- The frame
-- ---------------------------------------------------------------------------

def SRC_W : Nat := 3840
def SRC_H : Nat := 2160

/-- Each plane of a `w`x`h` frame: its width, height and offset. -/
def planes (w h : Nat) : List (Nat × Nat × Nat) :=
  [(w, h, 0), (w / 2, h / 2, w * h), (w / 2, h / 2, w * h + w / 2 * (h / 2))]

def frameBytes (w h : Nat) : Nat := w * h * 3 / 2

-- ---------------------------------------------------------------------------
-- Memory
-- ---------------------------------------------------------------------------

def natives : List X86.Body := [ImageAsm.rowpair, ImageAsm.boxrow, ImageAsm.lzh, ImageAsm.lzv]

def ROWPAIR_SLOT : Nat := 0
def BOXROW_SLOT : Nat := 1
def LZH_SLOT : Nat := 2
def LZV_SLOT : Nat := 3

/-- `0x00`--`0x40` are the runtime's context slots. From `SLOT_OFF`, one `i64`
    per body: where `asm_load` mapped it, or 0. At `NAME_OFF`, `"avx2"`,
    NUL-terminated, for `base_cpu_has`. From `CODE_OFF`, the bodies, each on a
    64-byte boundary. -/
def SLOT_OFF : Nat := 0x100
def NAME_OFF : Nat := 0x140
def CODE_OFF : Nat := 0x200

example : SLOT_OFF + 8 * natives.length ≤ NAME_OFF ∧ NAME_OFF + 5 ≤ CODE_OFF ∧
    CODE_OFF % 64 = 0 := by
  decide

def bodyBytes (b : X86.Body) : List UInt8 :=
  match X86.assemble b with
  | .ok bs => bs
  | .error e => panic! e

#guard natives.all fun b => (X86.assemble b).isOk

def align64 (n : Nat) : Nat := (n + 63) / 64 * 64

def codeOffs : List Nat :=
  (natives.foldl (fun (acc : List Nat × Nat) b =>
    (acc.1 ++ [acc.2], align64 (acc.2 + (bodyBytes b).length))) ([], CODE_OFF)).1

def CODE_END : Nat := natives.foldl (fun acc b => align64 (acc + (bodyBytes b).length)) CODE_OFF

-- The GPU state, after the bodies: the device, its primary context, the
-- module and its kernel, two streams, two device frames in and two out for the
-- pipeline, a launch's two arguments and the array of their addresses, the
-- kernel's name and its PTX, NUL-terminated.
def GPU_OFF : Nat := CODE_END
def G_DEV : Nat := GPU_OFF
def G_CTX : Nat := GPU_OFF + 0x08
def G_MOD : Nat := GPU_OFF + 0x10
def G_FN : Nat := GPU_OFF + 0x18
def G_STREAM : Nat := GPU_OFF + 0x20
def G_DIN : Nat := GPU_OFF + 0x30
def G_DOUT : Nat := GPU_OFF + 0x40
def G_ARGS : Nat := GPU_OFF + 0x50
def G_ARGP : Nat := GPU_OFF + 0x60
def G_NAME : Nat := GPU_OFF + 0x80
def G_PTX : Nat := GPU_OFF + 0xC0

/-- Threads a launch runs: one for every four output bytes. -/
def THREADS : Nat := frameBytes (SRC_W / 2) (SRC_H / 2) / 4
def BLOCK : Nat := 256
def GRID : Nat := (THREADS + BLOCK - 1) / BLOCK

section
open AlgorithmLib.Kernel AlgorithmLib.Kernel.Block
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- `box2(src, dst)`: the whole frame halved by averaging each 2x2 block,
    rounded half up, as NPP's supersampling does at exactly half size. A thread
    makes four outputs of one plane from two aligned eight-byte loads: the even
    and odd bytes of both rows summed in 16-bit lanes, then each lane's low
    byte packed (the shift by 2 carries a lane's low bits into the one below,
    which the mask clears). -/
def boxBody : Prog V L Unit := do
  let (w, h) := (SRC_W, SRC_H)
  let yT := w / 2 * (h / 2) / 4
  let uT := yT + w / 4 * (h / 4) / 4
  let uSrc := w * h
  let vSrc := uSrc + w / 2 * (h / 2)
  let uDst := w / 2 * (h / 2)
  let vDst := uDst + w / 4 * (h / 4)
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let src := (ps.drop 4).head; let dst := (ps.drop 5).head
  let g ← iadd (← imul ps.head ps.thd) ps.snd
  when .ult g (← c32 THREADS) do
    let inU ← icmp .uge g (← c32 yT)
    let inV ← icmp .uge g (← c32 uT)
    let rw ← select inU (← c32 (w / 4)) (← c32 (w / 2))
    let base ← select inV (← c32 uT) (← select inU (← c32 yT) (← c32 0))
    let so ← select inV (← c32 vSrc) (← select inU (← c32 uSrc) (← c32 0))
    let dOff ← select inV (← c32 vDst) (← select inU (← c32 uDst) (← c32 0))
    let o ← ishlImm (← isub g base) 2
    let row ← udiv o rw
    let col ← isub o (← imul row rw)
    let sw ← ishlImm rw 1
    let off ← iadd (← iadd (← ishlImm (← imul row sw) 1) (← ishlImm col 1)) so
    let pa ← iadd src (← ext off)
    let a ← load64 pa
    let b ← load64 (← iadd pa (← ext sw))
    let m ← c64n 0x00FF00FF00FF00FF
    let s ← iadd (← band a m) (← band b m)
    let s ← iadd s (← band (← ushrImm a 8) m)
    let s ← iadd s (← band (← ushrImm b 8) m)
    let x ← band (← ushrImm (← iadd s (← c64n 0x0002000200020002)) 2) m
    let y ← band (← bor x (← ushrImm x 8)) (← c64n 0x0000FFFF0000FFFF)
    let z ← bor y (← ushrImm y 16)
    storeI32 (← ireduce32 z) (← iadd dst (← ext (← iadd o dOff)))

/-- `box2`'s module, printed by the PTX compiler. -/
def boxPtx : String :=
  match ({ name := "box2", params := [.i64, .i64],
           code := emit boxBody (idParams ++ [.i64, .i64]) } : Kernel).ptx with
  | .ok s => s
  | .error _ => ""

/-- `box2`'s module prints. -/
theorem box_ptx_prints : 0 < boxPtx.utf8ByteSize := by native_decide

end

-- ---------------------------------------------------------------------------
-- Lanczos3's weights
-- ---------------------------------------------------------------------------

def PI : Float := 3.141592653589793

/-- Lanczos3 at `x`: `sinc x * sinc (x / 3)` inside `|x| < 3`. -/
def l3 (x : Float) : Float :=
  let sinc (t : Float) : Float := if t == 0 then 1 else Float.sin (PI * t) / (PI * t)
  if x.abs < 3 then sinc x * sinc (x / 3) else 0

/-- The nearest integer to `x`. -/
def roundInt (x : Float) : Int :=
  if x ≥ 0 then (x.round.toUInt64.toNat : Int) else -((-x).round.toUInt64.toNat : Int)

/-- The twelve taps of a 2:1 output, at source offsets `-5` to `6` from twice
    its index, as 14-bit weights summing to exactly 16384: Lanczos3 at each
    tap's distance from the output's centre, in output pixels, over the taps
    `keep` keeps, normalised, rounded, and the rounding's remainder put on the
    largest. -/
def lzWeights (keep : Nat → Bool) : List Int :=
  let raw := (List.range 12).map fun k => if keep k then l3 ((k.toFloat - 5.5) / 2) else 0
  let sum := raw.foldl (· + ·) 0
  let w := raw.map fun r => roundInt (r / sum * 16384)
  let big := ((List.range 12).zip w).foldl (fun (bk, bw) (k, x) => if x > bw then (k, x) else (bk, bw)) (0, -100000)
  let err := 16384 - w.foldl (· + ·) 0
  w.set big.1 (w[big.1]! + err)

/-- The weights where all twelve taps lie inside the plane. -/
def LZ_IN : List Int := lzWeights fun _ => true

/-- Output `e` from the first edge (`e < 3`) keeps the taps from `5 - 2e`;
    output `e` from the last keeps the taps to `6 + 2e`. -/
def lzLo (e : Nat) : List Int := lzWeights fun k => k + 2 * e ≥ 5
def lzHi (e : Nat) : List Int := lzWeights fun k => k ≤ 6 + 2 * e

#guard LZ_IN.foldl (· + ·) 0 == 16384
#guard (List.range 3).all fun e => (lzLo e).foldl (· + ·) 0 == 16384 && (lzHi e).foldl (· + ·) 0 == 16384
#guard LZ_IN == LZ_IN.reverse

/-- A table of twelve weights as `vpmaddwd` takes them: six 32-byte vectors,
    each eight copies of a pair `(w[2p], w[2p+1])` as two 16-bit words. -/
def lzTable (w : List Int) : List UInt8 :=
  (List.range 6).flatMap fun p =>
    let lo := (w[2 * p]! % 65536).toNat
    let hi := (w[2 * p + 1]! % 65536).toNat
    (List.replicate 8 (X86.le 2 lo ++ X86.le 2 hi)).flatten

/-- The tables: all taps inside, then the first three edge outputs', then the
    last three's. -/
def LZ_TABLES : List (List Int) :=
  [LZ_IN] ++ (List.range 3).map lzLo ++ (List.range 3).map lzHi

def LZ_TAB_OFF : Nat := align64 (G_PTX + boxPtx.utf8ByteSize + 1)
def LZ_TAB : Nat := 192

def MEM_SIZE : Nat := LZ_TAB_OFF + LZ_TABLES.length * LZ_TAB

/-- The caller's output: the frame, then from `RING_OFF` the sums: for
    `resize_clif` four rows of `RING_ROW` bytes, a 16-bit sum for every output
    of the widest plane; for `resize` two pairs of rows of `PAIR_ROW` bytes,
    as `rowpair` lays them out; for `lanczos` sixteen rows of `RING_ROW`
    bytes. At `DUMMY_OFF`, an output row the first pair's call writes and
    nothing reads; at `ARGS_OFF`, the loops' arguments, at most sixteen. -/
def OUT_FRAME : Nat := frameBytes (SRC_W / 2) (SRC_H / 2)
def RING_ROW : Nat := align64 (2 * (SRC_W / 2))
def PAIR_ROW : Nat := 64 * ((SRC_W / 2 - 2 + 15) / 16)
def RING_OFF : Nat := align64 OUT_FRAME
def DUMMY_OFF : Nat := RING_OFF + max (16 * RING_ROW) (2 * PAIR_ROW)
def ARGS_OFF : Nat := DUMMY_OFF + align64 (SRC_W / 2)
def OUT_NEED : Nat := ARGS_OFF + 8 * 16

def writeAt (m : List UInt8) (off : Nat) (bs : List UInt8) : List UInt8 :=
  m.take off ++ bs ++ m.drop (off + bs.length)

def initialMemory : List UInt8 :=
  let m := writeAt (zeros MEM_SIZE) NAME_OFF ("avx2".toUTF8.toList ++ [0])
  let m := writeAt (writeAt m G_NAME ("box2".toUTF8.toList ++ [0])) G_PTX (boxPtx.toUTF8.toList ++ [0])
  let m := writeAt m LZ_TAB_OFF (LZ_TABLES.flatMap lzTable)
  (natives.zip codeOffs).foldl (fun m (b, off) => writeAt m off (bodyBytes b)) m

def loads : List (Nat × Nat × Nat) :=
  (natives.zip (codeOffs.zip (List.range natives.length))).map
    fun (b, off, k) => (off, (bodyBytes b).length, k)

/-- Map every body, on an x86-64 CPU with AVX2. The caller's output gets how
    many bodies there are at 0 and each slot at `8 * (1 + slot)`, so it can see
    that all loaded. -/
def asmLoad : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let one ← iconst32 1
  when .ule (← iconst64 (8 * (natives.length + 1))) (← outLen) do
    storeI64 (← iconst64 natives.length) out
    when .eq (← nativeArch) one do
      when .eq (← cpuHas (← iaddImm base NAME_OFF) (← iconst64 5)) one do
        for (off, len, k) in loads do
          let addr ← nativeLoad (← iaddImm base off) (← iconst64 len)
          storeI64 addr (← iaddImm base (SLOT_OFF + 8 * k))
    for k in List.range natives.length do
      storeI64 (← load64 (← iaddImm base (SLOT_OFF + 8 * k))) (← iaddImm out (8 * (k + 1)))

-- ---------------------------------------------------------------------------
-- The outermost outputs
-- ---------------------------------------------------------------------------

/-- The taps along an axis: all four inside the plane, the last three at the
    first output, the first three at the last. -/
def TAPS_IN : List Nat := [1, 3, 3, 1]
def TAPS_LO : List Nat := [3, 3, 1]
def TAPS_HI : List Nat := [1, 3, 3]

/-- The output at `o` from the source pixels from `p`: `rows` taps down, each
    `w` bytes on, by `cols` taps across, divided by the weights' sum and
    rounded half up. The weights, and so the divisor, are constants. -/
def edgeOut (p o : V .i64) (w : Nat) (rows cols : List Nat) : Prog V L Unit := do
  let mut sum ← iconst64 0
  for (i, wr) in (List.range rows.length).zip rows do
    for (j, wc) in (List.range cols.length).zip cols do
      let x ← uload8_64 (← iaddImm p (i * w + j))
      let t ← if wr * wc == 1 then pure x else do imul x (← iconst64 (wr * wc))
      sum ← iadd sum t
  let den := rows.sum * cols.sum
  let v ← udiv (← iaddImm (← ishlImm sum 1) den) (← iconst64 (2 * den))
  istore8 v o

/-- Output row `oy` of the plane, whose taps down are `rows` from source row
    `r`: every output of it, one at a time. -/
def edgeRow (src dst : V .i64) (w : Nat) (oy r : Nat) (rows : List Nat) : Prog V L Unit := do
  let ow := w / 2
  let p ← iaddImm src (r * w)
  let o ← iaddImm dst (oy * ow)
  edgeOut p o w rows TAPS_LO
  forLoopFromTo (← iconst64 1) (← iconst64 (ow - 1)) fun ox => do
    edgeOut (← iadd p (← iaddImm (← ishlImm ox 1) (-1))) (← iadd o ox) w rows TAPS_IN
  edgeOut (← iaddImm p (w - 3)) (← iaddImm o (ow - 1)) w rows TAPS_HI

/-- The first and last outputs of output row `oy`, which is neither the first
    nor the last. -/
def rowEnds (src dst : V .i64) (w : Nat) (oy : V .i64) : Prog V L Unit := do
  let ow := w / 2
  let p ← iadd src (← imul (← iaddImm (← ishlImm oy 1) (-1)) (← iconst64 w))
  let o ← iadd dst (← imul oy (← iconst64 ow))
  edgeOut p o w TAPS_IN TAPS_LO
  edgeOut (← iaddImm p (w - 3)) (← iaddImm o (ow - 1)) w TAPS_IN TAPS_HI

-- ---------------------------------------------------------------------------
-- The plane through the loops
-- ---------------------------------------------------------------------------

/-- Ring row `r mod 4`, from its second sum: the first interior output. -/
def ringAt (r : V .i64) : Prog V L (V .i64) := do
  let k ← band r (← iconst64 3)
  iadd (← iaddImm (← outPtr) (RING_OFF + 2)) (← imul k (← iconst64 RING_ROW))

/-- The two loops: `hrow s dst n` and `vcol args`, `args` holding four rows of
    sums, the output and the count. -/
structure Loops (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  hrow : V .i64 → V .i64 → V .i64 → Prog V L Unit
  vcol : V .i64 → Prog V L Unit

/-- The loops in CLIF, one output a trip: what the machine code computes. -/
def clifLoops : Loops V L where
  hrow s d n := do
    forLoop n fun j => do
      let p ← iadd s (← ishlImm j 1)
      let a ← uload8_64 p
      let b ← uload8_64 (← iaddImm p 1)
      let c ← uload8_64 (← iaddImm p 2)
      let e ← uload8_64 (← iaddImm p 3)
      let v ← iadd (← iadd a e) (← imul (← iadd b c) (← iconst64 3))
      istore16 v (← iadd d (← ishlImm j 1))
  vcol args := do
    let h0 ← load64 args
    let h1 ← load64 (← iaddImm args 8)
    let h2 ← load64 (← iaddImm args 16)
    let h3 ← load64 (← iaddImm args 24)
    let o ← load64 (← iaddImm args 32)
    let n ← load64 (← iaddImm args 40)
    forLoop n fun j => do
      let at_ (h : V .i64) : Prog V L (V .i64) := do uload16_64 (← iadd h (← ishlImm j 1))
      let v ← iadd (← iadd (← at_ h0) (← at_ h3)) (← imul (← iadd (← at_ h1) (← at_ h2)) (← iconst64 3))
      istore8 (← ushrImm (← iaddImm v 32) 6) (← iadd o j)

/-- The loops in CLIF four outputs a trip, in 16-bit lanes of a 64-bit
    word: `hrow` takes eight source bytes at `2j` and eight at `2j + 2`, whose
    even and odd bytes are four outputs' four taps; `vcol` adds four words of
    sums lane by lane and packs the four results to bytes. A count that is not
    a multiple of four ends with a trip over the last four outputs, which
    writes some twice with the same values. -/
def swarLoops : Loops V L where
  hrow s d n := do
    let m ← iconst64 0x00FF00FF00FF00FF
    let last ← iaddImm n (-4)
    forLoop (← ushrImm (← iaddImm n 3) 2) fun c => do
      let j ← umin (← ishlImm c 2) last
      let p ← iadd s (← ishlImm j 1)
      let a ← load64 p
      let b ← load64 (← iaddImm p 2)
      let ea ← band a m
      let oa ← band (← ushrImm a 8) m
      let eb ← band b m
      let ob ← band (← ushrImm b 8) m
      let h ← iadd (← iadd ea ob) (← imul (← iadd oa eb) (← iconst64 3))
      storeI64 h (← iadd d (← ishlImm j 1))
  vcol args := do
    let h0 ← load64 args
    let h1 ← load64 (← iaddImm args 8)
    let h2 ← load64 (← iaddImm args 16)
    let h3 ← load64 (← iaddImm args 24)
    let o ← load64 (← iaddImm args 32)
    let n ← load64 (← iaddImm args 40)
    let m ← iconst64 0x00FF00FF00FF00FF
    let half ← iconst64 0x0020002000200020
    let last ← iaddImm n (-4)
    forLoop (← ushrImm (← iaddImm n 3) 2) fun c => do
      let j ← umin (← ishlImm c 2) last
      let at_ (h : V .i64) : Prog V L (V .i64) := do load64 (← iadd h (← ishlImm j 1))
      let v ← iadd (← iadd (← at_ h0) (← at_ h3)) (← imul (← iadd (← at_ h1) (← at_ h2)) (← iconst64 3))
      let v ← band (← ushrImm (← iadd v half) 6) m
      let v ← band (← bor v (← ushrImm v 8)) (← iconst64 0x0000FFFF0000FFFF)
      let v ← bor v (← ushrImm v 16)
      istore32 v (← iadd o j)

/-- The plane by `rowpair` at `rf`: each output row but the outermost from one
    call, the sums alternating between two buffers, then its two ends while
    its source rows are in cache. -/
def planeAsm (rf src dst : V .i64) (w h : Nat) : Prog V L Unit := do
  let (ow, oh) := (w / 2, h / 2)
  edgeRow src dst w 0 0 TAPS_LO
  let out ← outPtr
  let x ← iaddImm out RING_OFF
  let y ← iaddImm out (RING_OFF + PAIR_ROW)
  let args ← iaddImm out ARGS_OFF
  let z ← iconst64 0
  -- row `a` and the one after it, their first taps at source column 1
  let call (a prev next o : V .i64) : Prog V L Unit := do
    storeI64 (← iaddImm (← iadd src (← imul a (← iconst64 w))) 1) args
    storeI64 (← iconst64 w) (← iaddImm args 8)
    storeI64 prev (← iaddImm args 16)
    storeI64 next (← iaddImm args 24)
    storeI64 o (← iaddImm args 32)
    storeI64 (← iconst64 (ow - 2)) (← iaddImm args 40)
    let _ ← nativeCall rf args z z z
  -- rows 1 and 2's sums into `x`; the output row this writes is not one
  call (← iconst64 1) y x (← iaddImm out DUMMY_OFF)
  forLoopFromTo (← iconst64 1) (← iconst64 (oh - 1)) fun oy => do
    let odd ← icmp .ne (← band oy (← iconst64 1)) z
    let prev ← select odd x y
    let next ← select odd y x
    call (← iaddImm (← ishlImm oy 1) 1) prev next (← iaddImm (← iadd dst (← imul oy (← iconst64 ow))) 1)
    rowEnds src dst w oy
  edgeRow src dst w (oh - 1) (h - 3) TAPS_HI

/-- The plane by the loops in `lp`, as `planeAsm` orders it. -/
def planeFast (lp : Loops V L) (src dst : V .i64) (w h : Nat) : Prog V L Unit := do
  let (ow, oh) := (w / 2, h / 2)
  edgeRow src dst w 0 0 TAPS_LO
  let bottom ← iconst64 (oh - 1)
  -- source row `r`'s sums, for outputs 1 to ow - 2: their first taps start
  -- at source column 1
  let hsum (r : V .i64) : Prog V L Unit := do
    let s ← iaddImm (← iadd src (← imul r (← iconst64 w))) 1
    lp.hrow s (← ringAt r) (← iconst64 (ow - 2))
  hsum (← iconst64 1)
  hsum (← iconst64 2)
  let args ← iaddImm (← outPtr) ARGS_OFF
  forLoopFromTo (← iconst64 1) bottom fun oy => do
    let r0 ← iaddImm (← ishlImm oy 1) (-1)
    hsum (← iaddImm r0 2)
    hsum (← iaddImm r0 3)
    for k in List.range 4 do
      storeI64 (← ringAt (← iaddImm r0 k)) (← iaddImm args (8 * k))
    storeI64 (← iaddImm (← iadd dst (← imul oy (← iconst64 ow))) 1) (← iaddImm args 32)
    storeI64 (← iconst64 (ow - 2)) (← iaddImm args 40)
    lp.vcol args
    rowEnds src dst w oy
  edgeRow src dst w (oh - 1) (h - 3) TAPS_HI

/-- Each plane by `plane`, when the input holds a frame and the output room
    for the frame and the ring. -/
def eachPlane (plane : V .i64 → V .i64 → Nat → Nat → Prog V L Unit) : Prog V L Unit := do
  let src ← dataPtr
  let dst ← outPtr
  when .ule (← iconst64 (frameBytes SRC_W SRC_H)) (← dataLen) do
    when .ule (← iconst64 OUT_NEED) (← outLen) do
      for ((w, h, so), (_, _, d_o)) in (planes SRC_W SRC_H).zip (planes (SRC_W / 2) (SRC_H / 2)) do
        plane (← iaddImm src so) (← iaddImm dst d_o) w h

/-- The frame through the machine code, or through the four-a-trip loops in
    CLIF where it did not load. -/
def resize : Prog V L Unit := do
  let base ← basePtr
  let rf ← load64 (← iaddImm base (SLOT_OFF + 8 * ROWPAIR_SLOT))
  eachPlane fun s d w h => do
    let _ ← ifte (jTys := []) .ne rf (← iconst64 0)
      (thn := do planeAsm rf s d w h; pure %[])
      (els := do planeFast swarLoops s d w h; pure %[])

/-- The frame through the loops in CLIF, one output a trip. -/
def resizeClif : Prog V L Unit :=
  eachPlane fun s d w h => planeFast clifLoops s d w h

/-- The frame through the loops in CLIF, four outputs a trip. -/
def resizeSwar : Prog V L Unit :=
  eachPlane fun s d w h => planeFast swarLoops s d w h

-- ---------------------------------------------------------------------------
-- The GPU, through the CUDA driver called directly
-- ---------------------------------------------------------------------------

/-- Set the GPU up on the calling thread: the device's primary context, the
    module from the PTX and its kernel, two streams, and two device frames in
    and two out. Each step's `CUresult` goes to the output, four bytes apiece,
    and a step runs only when the one before answered 0. -/
def gpuInit : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let at_ (k : Nat) : Prog V L (V .i64) := iaddImm base k
  let z32 ← iconst32 0
  let steps : List (Prog V L (V .i32)) :=
    [do ext (.cuda .init) %[z32],
     do ext (.cuda .deviceGet) %[← at_ G_DEV, z32],
     do ext (.cuda .primaryCtxRetain) %[← at_ G_CTX, ← load32 (← at_ G_DEV)],
     do ext (.cuda .ctxSetCurrent) %[← load64 (← at_ G_CTX)],
     do ext (.cuda .moduleLoadData) %[← at_ G_MOD, ← at_ G_PTX],
     do ext (.cuda .moduleGetFunction) %[← at_ G_FN, ← load64 (← at_ G_MOD), ← at_ G_NAME],
     do ext (.cuda .streamCreate) %[← at_ G_STREAM, ← iconst32 1],
     do ext (.cuda .streamCreate) %[← at_ (G_STREAM + 8), ← iconst32 1],
     do ext (.cuda .memAlloc) %[← at_ G_DIN, ← iconst64 (frameBytes SRC_W SRC_H)],
     do ext (.cuda .memAlloc) %[← at_ (G_DIN + 8), ← iconst64 (frameBytes SRC_W SRC_H)],
     do ext (.cuda .memAlloc) %[← at_ G_DOUT, ← iconst64 (frameBytes (SRC_W / 2) (SRC_H / 2))],
     do ext (.cuda .memAlloc) %[← at_ (G_DOUT + 8), ← iconst64 (frameBytes (SRC_W / 2) (SRC_H / 2))]]
  when .ule (← iconst64 (4 * steps.length)) (← outLen) do
    storeI64 (← at_ G_ARGS) (← at_ G_ARGP)
    storeI64 (← at_ (G_ARGS + 8)) (← at_ (G_ARGP + 8))
    let rec go : List (Prog V L (V .i32)) → Nat → Prog V L Unit
      | [], _ => pure ()
      | st :: rest, k => do
        let rc ← st
        store rc (← iaddImm out (4 * k))
        when .eq rc z32 (go rest (k + 1))
    go steps 0

/-- Launch `box2` from device frame `src` to `dst` on `stream`. -/
def launchBox (src dst stream : V .i64) : Prog V L (V .i32) := do
  let base ← basePtr
  storeI64 src (← iaddImm base G_ARGS)
  storeI64 dst (← iaddImm base (G_ARGS + 8))
  let one ← iconst32 1
  ext (.cuda .launchKernel) %[← load64 (← iaddImm base G_FN), ← iconst32 GRID, one, one,
    ← iconst32 BLOCK, one, one, ← iconst32 0, stream, ← iaddImm base G_ARGP, ← iconst64 0]

/-- Frames already on the device: the input is `(src, dst)` pairs of device
    addresses; every pair is launched on the first stream, then the stream is
    waited for once. The output gets how many launches failed and the wait's
    `CUresult`, four bytes each. -/
def gpuBox : Prog V L Unit := do
  let base ← basePtr
  let input ← dataPtr
  let out ← outPtr
  when .ule (← iconst64 8) (← outLen) do
    let stream ← load64 (← iaddImm base G_STREAM)
    let z32 ← iconst32 0
    let fails ← forLoopAcc (← ushrImm (← dataLen) 4) z32 fun i acc => do
      let p ← iadd input (← ishlImm i 4)
      let rc ← launchBox (← load64 p) (← load64 (← iaddImm p 8)) stream
      iadd acc (← select (← icmp .ne rc z32) (← iconst32 1) z32)
    store fails out
    store (← ext (.cuda .streamSynchronize) %[stream]) (← iaddImm out 4)

/-- Frames in page-locked host memory, through the device and back: the input
    is two host outputs, then the host address of each frame. Frame `i` goes on
    stream `i mod 2`: uploaded to that stream's device frame, launched, and
    downloaded to that stream's host output, so one stream's copies overlap the
    other's work. Both streams are waited for at the end. The output gets how
    many calls failed and the two waits' `CUresult`s, four bytes each. -/
def gpuPipe : Prog V L Unit := do
  let base ← basePtr
  let input ← dataPtr
  let out ← outPtr
  when .ule (← iconst64 12) (← outLen) do
    when .ule (← iconst64 16) (← dataLen) do
      let z32 ← iconst32 0
      let n ← isub (← ushrImm (← dataLen) 3) (← iconst64 2)
      let fails ← forLoopAcc n z32 fun i acc => do
        let k ← ishlImm (← band i (← iconst64 1)) 3
        let stream ← load64 (← iadd (← iaddImm base G_STREAM) k)
        let din ← load64 (← iadd (← iaddImm base G_DIN) k)
        let dout ← load64 (← iadd (← iaddImm base G_DOUT) k)
        let host ← load64 (← iadd input k)
        let frame ← load64 (← iadd (← iaddImm input 16) (← ishlImm i 3))
        let a ← ext (.cuda .memcpyHtoDAsync) %[din, frame, ← iconst64 (frameBytes SRC_W SRC_H), stream]
        let b ← launchBox din dout stream
        let c ← ext (.cuda .memcpyDtoHAsync) %[host, dout, ← iconst64 (frameBytes (SRC_W / 2) (SRC_H / 2)), stream]
        let bad ← bor (← bor (← icmp .ne a z32) (← icmp .ne b z32)) (← icmp .ne c z32)
        iadd acc (← select bad (← iconst32 1) z32)
      store fails out
      for k in [0, 1] do
        let rc ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base (G_STREAM + 8 * k))]
        store rc (← iaddImm out (4 + 4 * k))

-- ---------------------------------------------------------------------------
-- The 2x2 average on the CPU
-- ---------------------------------------------------------------------------

/-- The plane's 2x2 average in CLIF, four outputs a trip: eight bytes of each
    source row, whose even and odd bytes summed in 16-bit lanes give four
    blocks' sums, rounded, shifted and packed to bytes. -/
def boxPlaneClif (src dst : V .i64) (w h : Nat) : Prog V L Unit := do
  let ow := w / 2
  let m ← iconst64 0x00FF00FF00FF00FF
  let two ← iconst64 0x0002000200020002
  forLoop (← iconst64 (h / 2)) fun oy => do
    let a ← iadd src (← imul oy (← iconst64 (2 * w)))
    let b ← iaddImm a w
    let o ← iadd dst (← imul oy (← iconst64 ow))
    forLoop (← iconst64 (ow / 4)) fun c => do
      let x ← load64 (← iadd a (← ishlImm c 3))
      let y ← load64 (← iadd b (← ishlImm c 3))
      let e ← iadd (← band x m) (← band y m)
      let od ← iadd (← band (← ushrImm x 8) m) (← band (← ushrImm y 8) m)
      let v ← band (← ushrImm (← iadd (← iadd e od) two) 2) m
      let v ← band (← bor v (← ushrImm v 8)) (← iconst64 0x0000FFFF0000FFFF)
      let v ← bor v (← ushrImm v 16)
      istore32 v (← iadd o (← ishlImm c 2))

/-- The plane's 2x2 average by `boxrow` at `bf`, one call an output row. -/
def boxPlaneAsm (bf src dst : V .i64) (w h : Nat) : Prog V L Unit := do
  let ow := w / 2
  let args ← iaddImm (← outPtr) ARGS_OFF
  let z ← iconst64 0
  storeI64 (← iconst64 w) (← iaddImm args 8)
  storeI64 (← iconst64 ow) (← iaddImm args 24)
  forLoop (← iconst64 (h / 2)) fun oy => do
    storeI64 (← iadd src (← imul oy (← iconst64 (2 * w)))) args
    storeI64 (← iadd dst (← imul oy (← iconst64 ow))) (← iaddImm args 16)
    let _ ← nativeCall bf args z z z

/-- Every plane's widths are a multiple of eight, as the CLIF's trips need. -/
example : (planes SRC_W SRC_H).all (fun (w, _, _) => w % 8 == 0) = true := by decide

/-- The frame's 2x2 average through the machine code, or through the CLIF
    where it did not load. -/
def box : Prog V L Unit := do
  let bf ← load64 (← iaddImm (← basePtr) (SLOT_OFF + 8 * BOXROW_SLOT))
  eachPlane fun s d w h => do
    let _ ← ifte (jTys := []) .ne bf (← iconst64 0)
      (thn := do boxPlaneAsm bf s d w h; pure %[])
      (els := do boxPlaneClif s d w h; pure %[])

/-- The frame's 2x2 average through the CLIF. -/
def boxClif : Prog V L Unit :=
  eachPlane fun s d w h => boxPlaneClif s d w h

-- ---------------------------------------------------------------------------
-- Lanczos3
-- ---------------------------------------------------------------------------

/-- The two loops: `hrow s dst n`, a source row's sums for `n` outputs from
    the tap under the first, and `vrow args`, an output row from the twelve
    rows, the weights, the output and the count at `args`. -/
structure LzLoops (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  hrow : V .i64 → V .i64 → V .i64 → Prog V L Unit
  vrow : V .i64 → Prog V L Unit

/-- `sum w_k * load k` over the taps with a weight. -/
def taps12 (w : List Int) (load : Nat → Prog V L (V .i64)) : Prog V L (V .i64) := do
  let mut acc ← iconst64 0
  for (k, wk) in (List.range 12).zip w do
    if wk != 0 then
      acc ← iadd acc (← imul (← load k) (← iconst64 wk))
  return acc

/-- A sum's rounding to a word: `(acc + 128) >> 8`, arithmetic. -/
def toWord (acc : V .i64) : Prog V L (V .i64) := do
  sshr (← iaddImm acc 128) (← iconst64 8)

/-- The loops as the machine code at `hf` and `vf`, with the inside weights. -/
def lzNative (hf vf : V .i64) : LzLoops V L where
  hrow s d n := do
    let _ ← nativeCall hf s d n (← iaddImm (← basePtr) LZ_TAB_OFF)
  vrow args := do
    let z ← iconst64 0
    let _ ← nativeCall vf args z z z

/-- The loops in CLIF, one output a trip, computing what the machine code
    does. -/
def lzClif : LzLoops V L where
  hrow s d n := do
    forLoop n fun j => do
      let p ← iadd s (← ishlImm j 1)
      let acc ← taps12 LZ_IN fun k => do uload8_64 (← iaddImm p k)
      istore16 (← toWord acc) (← iadd d (← ishlImm j 1))
  vrow args := do
    let tab ← load64 (← iaddImm args 96)
    let o ← load64 (← iaddImm args 104)
    let n ← load64 (← iaddImm args 112)
    let rows ← (List.range 12).mapM fun (k : Nat) => do load64 (← iaddImm args (8 * k : Nat))
    let ws ← (List.range 12).mapM fun (k : Nat) => do
      sload16_64 (← iaddImm tab (32 * (k / 2) + 2 * (k % 2) : Nat))
    let lo ← iconst64 0
    let hi ← iconst64 255
    forLoop n fun t => do
      let mut acc ← iconst64 0
      for (r, w) in rows.zip ws do
        acc ← iadd acc (← imul w (← sload16_64 (← iadd r (← ishlImm t 1))))
      let v ← sshr (← iaddImm acc 0x80000) (← iconst64 20)
      istore8 (← smax lo (← smin hi v)) (← iadd o t)

/-- Ring row `r mod 16` of Lanczos3's sums. -/
def lzRow (r : V .i64) : Prog V L (V .i64) := do
  iadd (← iaddImm (← outPtr) RING_OFF) (← imul (← band r (← iconst64 15)) (← iconst64 RING_ROW))

/-- The plane by `lp`. Each source row's sums go to the ring as the output
    rows need them: the inside outputs' by `lp.hrow`, the three at each end
    one at a time with their own weights. Each output row then takes its
    twelve rows, clamped to the plane, with the weights for its distance from
    the plane's first and last rows. -/
def planeLz (lp : LzLoops V L) (src dst : V .i64) (w h : Nat) : Prog V L Unit := do
  let (ow, oh) := (w / 2, h / 2)
  let hsum (r : V .i64) : Prog V L Unit := do
    let row ← iadd src (← imul r (← iconst64 w))
    let d ← lzRow r
    lp.hrow (← iaddImm row 1) (← iaddImm d 6) (← iconst64 (ow - 6))
    for e in List.range 3 do
      -- output e's taps from source column 2e - 5; output ow - 1 - e's from w - 7 - 2e
      let lo ← taps12 (lzLo e) fun k => do uload8_64 (← iaddImm row (2 * e + k - 5))
      istore16 (← toWord lo) (← iaddImm d (2 * e))
      let hi ← taps12 (lzHi e) fun k => do uload8_64 (← iaddImm row (w - 7 - 2 * e + k))
      istore16 (← toWord hi) (← iaddImm d (2 * (ow - 1 - e)))
  for r in List.range 5 do
    hsum (← iconst64 r)
  let args ← iaddImm (← outPtr) ARGS_OFF
  let tabs ← iaddImm (← basePtr) LZ_TAB_OFF
  let zero ← iconst64 0
  let last ← iconst64 (h - 1)
  forLoop (← iconst64 oh) fun y => do
    let y2 ← ishlImm y 1
    for e in [5, 6] do
      let r ← iaddImm y2 e
      when .ult r (← iconst64 h) (hsum r)
    -- the table: 1 + y for the first three rows, 4 + (oh - 1 - y) for the last three
    let top ← icmp .ult y (← iconst64 3)
    let bot ← icmp .uge y (← iconst64 (oh - 3))
    let idx ← select top (← iaddImm y 1) (← select bot (← isub (← iconst64 (oh + 3)) y) zero)
    storeI64 (← iadd tabs (← imul idx (← iconst64 LZ_TAB))) (← iaddImm args 96)
    for k in List.range 12 do
      let r ← iaddImm y2 (k - 5 : Int)
      let r ← select (← icmp .slt r zero) zero (← umin r last)
      storeI64 (← lzRow r) (← iaddImm args (8 * k))
    storeI64 (← iadd dst (← imul y (← iconst64 ow))) (← iaddImm args 104)
    storeI64 (← iconst64 ow) (← iaddImm args 112)
    lp.vrow args

/-- The frame's Lanczos3 through the machine code, or through the CLIF where
    it did not load. -/
def lanczos : Prog V L Unit := do
  let base ← basePtr
  let hf ← load64 (← iaddImm base (SLOT_OFF + 8 * LZH_SLOT))
  let vf ← load64 (← iaddImm base (SLOT_OFF + 8 * LZV_SLOT))
  let both ← band (← icmp .ne hf (← iconst64 0)) (← icmp .ne vf (← iconst64 0))
  eachPlane fun s d w h => do
    let _ ← ifte (jTys := []) .ne both (← iconst .i8 0)
      (thn := do planeLz (lzNative hf vf) s d w h; pure %[])
      (els := do planeLz lzClif s d w h; pure %[])

/-- The frame's Lanczos3 through the CLIF. -/
def lanczosClif : Prog V L Unit :=
  eachPlane fun s d w h => planeLz lzClif s d w h

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "asm_load" (Prog.compileProg 0 asmLoad),
     Prog.entry "resize" (Prog.compileProg 1 resize),
     Prog.entry "resize_clif" (Prog.compileProg 2 resizeClif),
     Prog.entry "gpu_init" (Prog.compileProg 3 gpuInit),
     Prog.entry "gpu_box" (Prog.compileProg 4 gpuBox),
     Prog.entry "gpu_pipe" (Prog.compileProg 5 gpuPipe),
     Prog.entry "resize_swar" (Prog.compileProg 6 resizeSwar),
     Prog.entry "box" (Prog.compileProg 7 box),
     Prog.entry "box_clif" (Prog.compileProg 8 boxClif),
     Prog.entry "lanczos" (Prog.compileProg 9 lanczos),
     Prog.entry "lanczos_clif" (Prog.compileProg 10 lanczosClif)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "image_bench" {
    functions := clif,
    required_memory := MEM_SIZE,
    initial_memory := initialMemory
  }]

end ImageBench

def Bench.Image.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie ImageBench.clifIR
  emitArtifacts outDir (ImageBench.artifacts clif)

#eval ShipScan.check "Bench.Image" `Bench.Image.main
