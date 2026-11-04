module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace Scene

structure PositiveNat where
  value : Nat
  isPositive : value > 0

structure Channel where
  value : Nat
  inRange : value ≤ 255

def PositiveNat.mkChecked (value : Nat) (isPositive : value > 0) : PositiveNat :=
  { value, isPositive }

def Channel.mkChecked (value : Nat) (inRange : value ≤ 255) : Channel :=
  { value, inRange }

instance : Coe PositiveNat Nat where
  coe n := n.value

instance : Coe Channel Nat where
  coe c := c.value

structure Color where
  red : Channel
  green : Channel
  blue : Channel

def checkedColor
    (red green blue : Nat)
    (redInRange : red ≤ 255)
    (greenInRange : green ≤ 255)
    (blueInRange : blue ≤ 255) : Color :=
  {
    red := Channel.mkChecked red redInRange
    green := Channel.mkChecked green greenInRange
    blue := Channel.mkChecked blue blueInRange
  }

structure ScenePalette where
  groundLight : Color
  groundDark : Color
  wall : Color
  diffuseSphere : Color
  metalSphere : Color

structure SceneSpec where
  width : PositiveNat
  height : PositiveNat
  samples : PositiveNat
  maxBounces : PositiveNat
  palette : ScenePalette
  filename : String

def checkedScene
    (width height samples maxBounces : Nat)
    (palette : ScenePalette)
    (filename : String)
    (widthPositive : width > 0)
    (heightPositive : height > 0)
    (samplesPositive : samples > 0)
    (bouncesPositive : maxBounces > 0) : SceneSpec :=
  {
    width := PositiveNat.mkChecked width widthPositive
    height := PositiveNat.mkChecked height heightPositive
    samples := PositiveNat.mkChecked samples samplesPositive
    maxBounces := PositiveNat.mkChecked maxBounces bouncesPositive
    palette
    filename
  }

def imageWidth (spec : SceneSpec) : Nat := spec.width.value
def imageHeight (spec : SceneSpec) : Nat := spec.height.value
def sampleCount (spec : SceneSpec) : Nat := spec.samples.value
def bounceCount (spec : SceneSpec) : Nat := spec.maxBounces.value

def pixelCount (spec : SceneSpec) : Nat := imageWidth spec * imageHeight spec
def pixelBytes (spec : SceneSpec) : Nat := pixelCount spec * 4

def bmpHeader (spec : SceneSpec) : List UInt8 :=
  let fileSize : Nat := 54 + pixelBytes spec
  let bfType := [0x42, 0x4D]
  let bfSize := uint32ToBytes (UInt32.ofNat fileSize)
  let bfReserved := [0, 0, 0, 0]
  let bfOffBits := uint32ToBytes 54
  let biSize := uint32ToBytes 40
  let biWidth := uint32ToBytes (UInt32.ofNat (imageWidth spec))
  let biHeight := int32ToBytes (Int.negSucc (imageHeight spec - 1))
  let biPlanes := [1, 0]
  let biBitCount := [32, 0]
  let biCompression := uint32ToBytes 0
  let biSizeImage := uint32ToBytes (UInt32.ofNat (pixelBytes spec))
  let biXPelsPerMeter := uint32ToBytes 2835
  let biYPelsPerMeter := uint32ToBytes 2835
  let biClrUsed := uint32ToBytes 0
  let biClrImportant := uint32ToBytes 0
  bfType ++ bfSize ++ bfReserved ++ bfOffBits ++
  biSize ++ biWidth ++ biHeight ++ biPlanes ++ biBitCount ++
  biCompression ++ biSizeImage ++ biXPelsPerMeter ++ biYPelsPerMeter ++
  biClrUsed ++ biClrImportant

/-! ## The renderer

A path tracer, a thread a pixel and blocks of 256 threads a 16×16 tile. For
each of the spec's samples a ray leaves a jittered lens point through a
jittered point of the pixel; at each of up to `maxBounces` bounces it finds
the nearest of a ground plane (`y = -1`, checkered), a back wall (`z = -10`)
and three spheres (glass, gold, a blue diffuse), adds fog, and on a diffuse
or metal surface the light of three point lights, the first behind a shadow
test against the spheres. A diffuse surface scatters the ray about its
normal, the gold sphere reflects it with a little fuzz, and the glass one
reflects or refracts it by Schlick's approximation. After the second bounce
a path survives with probability its throughput's largest channel, clamped to
`[0.1, 0.97]`, and is reweighted by its inverse. The samples' mean is tone
mapped (`c / (c + 1)`, then a square root), vignetted, and stored as a BGRA
word.

Random numbers are a 32-bit LCG per pixel, a draw being its next state over
`2^32`. Where they lie in the draw order is part of the picture: the four a
hit draws for the lights are drawn before its material draws its own. The
first light's position is built from three of the four, and the other two
lights from the fourth's slot after it holds the first light's unnormalised
`x` offset, as the picture has always been made. -/

namespace Body
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The generator's next state, and its draw in `[0, 1)`. -/
def draw (rng : V .i32) : Prog V L (V .i32 × V .f32) := do
  let r ← iadd (← imul rng (← c32 1664525)) (← c32 1013904223)
  pure (r, ← fmul (← fcvtFromUint .f32 r) (← f32c 0x2F800000))

/-- `n` draws, each less a half: the scaling and the subtraction one fused
    multiply-add. -/
def draws (rng : V .i32) (n : Nat) : Prog V L (V .i32 × List (V .f32)) := do
  let mut r := rng
  let mut xs : List (V .f32) := []
  let sc ← f32c 0x2F800000
  let h ← fl (-0.5)
  for _ in List.range n do
    let r' ← iadd (← imul r (← c32 1664525)) (← c32 1013904223)
    r := r'
    xs := xs ++ [← fma (← fcvtFromUint .f32 r') sc h]
  pure (r, xs)

/-- Two flags both set. -/
def both (a b : V .i8) : Prog V L (V .i8) := band a b

/-- The nearest hit so far: its distance, kind and normal. -/
structure Hit (V : ClifTy → Type) where
  t : V .f32
  kind : V .i32
  n : V3 V

def Hit.vals (h : Hit V) : Vals V [.f32, .i32, .f32, .f32, .f32] := %[h.t, h.kind, h.n.x, h.n.y, h.n.z]
def Hit.ofVals (v : Vals V [.f32, .i32, .f32, .f32, .f32]) : Hit V :=
  ⟨v.head, v.snd, ⟨v.thd, (v.drop 3).head, (v.drop 4).head⟩⟩

/-- Whether a hit at `t` is nearer than `h` and past `0.001`. -/
def Hit.nearer (h : Hit V) (t : V .f32) : Prog V L (V .i8) := do
  both (← fcmp .gt t (← fl 0.001)) (← fcmp .lt t h.t)

/-- A plane `coord = pos` met along one axis, whose normal is `n`. -/
def plane (h : Hit V) (o d : V .f32) (pos : Float) (k : Nat) (n : V3 V) : Prog V L (Hit V) := do
  let ok ← bor (← fcmp .ge d (← fl 1.0e-6)) (← fcmp .le d (← fl (-1.0e-6)))
  let t ← fdiv (← fsub (← fl pos) o) d
  let c ← both ok (← h.nearer t)
  pure ⟨← select c t h.t, ← select c (← c32 k) h.kind, ← V3.sel c n h.n⟩

/-- A sphere at `-k` of squared radius `r2` and reciprocal radius `ir`: the
    nearer root past `0.001`, or the farther one, and there the normal
    `(o + k + t d) / r`. A ray that misses (a negative discriminant) skips the
    root, which is most rays at most spheres, and one that hits nothing nearer
    skips the normal. -/
def sphere (h : Hit V) (o d : V3 V) (k : V3 V) (r2 ir : Float) (kind : Nat) : Prog V L (Hit V) := do
  let oc ← V3.zip fadd o k
  let b ← oc.dot d
  let disc ← fma b b (← fsub (← fl r2) (← oc.dot oc))
  let r ← ifte (jTys := [.f32, .i32, .f32, .f32, .f32]) .ne (← fcmp .ge disc (← fl 0.0)) (← iconst .i8 0)
    (do
      let sq : V .f32 ← callLocal sqrtApprox %[disc]
      let t0 ← fneg (← fadd b sq)
      let t1 ← fsub sq b
      let t ← select (← fcmp .gt t0 (← fl 0.001)) t0 t1
      let c ← h.nearer t
      let n ← (← d.fmaS t oc).scale (← fl ir)
      pure (Hit.vals ⟨← select c t h.t, ← select c (← c32 kind) h.kind, ← V3.sel c n h.n⟩))
    (pure h.vals)
  pure (Hit.ofVals r)

/-- The spheres: the offset to each centre, its squared radius and reciprocal
    radius, by kind. -/
def spheres : List (Nat × (Float × Float × Float) × Float × Float) :=
  [(3, (1.35, 0.05, 4.7), 0.9025, 1 / 0.95), (4, (-1.35, 0.25, 4.2), 0.5625, 1 / 0.75),
   (5, (-0.1, -0.15, 6.5), 1.3225, 1 / 1.15)]

/-- Whether a ray from `p` along `l` meets the sphere at `-k` before `-ndist`. -/
def blocks (p l : V3 V) (k : V3 V) (r2 : Float) (ndist : V .f32) : Prog V L (V .i8) := do
  let oc ← V3.zip fadd p k
  let b ← oc.dot l
  let disc ← fma b b (← fsub (← fl r2) (← oc.dot oc))
  -- a ray that misses (a negative discriminant) skips the root
  let r ← ifte (jTys := [.i32]) .ne (← fcmp .ge disc (← fl 0.0)) (← iconst .i8 0)
    (do
      let sq : V .f32 ← callLocal sqrtApprox %[disc]
      -- the root `t = -(b + √disc)` lies in `(0.001, dist)`, `ndist = -dist`
      let s ← fadd b sq
      let hit ← both (← fcmp .lt s (← fl (-0.001))) (← fcmp .gt s ndist)
      pure %[← select hit (← c32 1) (← c32 0)])
    (pure %[← c32 0])
  icmp .ne r.head (← c32 0)

/-- A point light at `pos` seen from `p` with normal `n`: the light's offset,
    its squared distance, the offset's direction and `max (n · l̂) 0`. -/
def toLight (p n pos : V3 V) : Prog V L (V3 V × V .f32 × V3 V × V .f32) := do
  let l ← V3.zip fsub pos p
  let d2 ← l.dot l
  let lh ← l.scale (← callLocal rsqrtApprox %[d2])
  pure (l, d2, lh, ← fmax (← n.dot lh) (← fl 0.0))

/-- `acc + w · c`, where `ok`, `w = ndl / d2`. -/
def addLight (acc : V3 V) (ok : V .i8) (ndl d2 : V .f32) (c : Float × Float × Float) :
    Prog V L (V3 V) := do
  let w ← fmul ndl (← callLocal rcpApprox %[d2])
  let lit ← V3.mk (← fma w (← fl c.1) acc.x) (← fma w (← fl c.2.1) acc.y) <$> fma w (← fl c.2.2) acc.z
  V3.sel ok lit acc

/-- An 8-bit channel's value as a float in `[0, 1]`. -/
def chan (c : Channel) : Float := Float.ofNat c.value / 255.0

def colorLit (c : Color) : Prog V L (V3 V) := V3.lit (chan c.red) (chan c.green) (chan c.blue)

/-- A path's state between bounces: the bounce (the limit once the path has
    ended), the generator, the ray, the throughput and the radiance. -/
abbrev PathTys : List ClifTy := .i32 :: .i32 :: List.replicate 12 .f32

structure Path (V : ClifTy → Type) where
  bounce : V .i32
  rng : V .i32
  o : V3 V
  d : V3 V
  thr : V3 V
  rad : V3 V

def Path.vals (p : Path V) : Vals V PathTys :=
  %[p.bounce, p.rng, p.o.x, p.o.y, p.o.z, p.d.x, p.d.y, p.d.z, p.thr.x, p.thr.y, p.thr.z,
    p.rad.x, p.rad.y, p.rad.z]

def Path.ofVals (v : Vals V PathTys) : Path V :=
  let f := Vals.uniformToList (v.drop 2 : Vals V (List.replicate 12 .f32))
  let g (i : Nat) := f.getD i v.thd
  { bounce := v.head, rng := v.snd, o := ⟨g 0, g 1, g 2⟩, d := ⟨g 3, g 4, g 5⟩,
    thr := ⟨g 6, g 7, g 8⟩, rad := ⟨g 9, g 10, g 11⟩ }

set_option maxRecDepth 8000 in
/-- One bounce of a live path. -/
def bounce (spec : SceneSpec) (q : Path V) : Prog V L (Path V) := do
  let o := q.o; let d := q.d
  let zero ← fl 0.0
  let n0 : V3 V := ⟨zero, zero, zero⟩
  let h0 : Hit V := ⟨← fl 1.0e20, ← c32 0, n0⟩
  let h1 ← plane h0 o.y d.y (-1.0) 1 (← V3.lit 0 1 0)
  let mut h ← plane h1 o.z d.z (-10.0) 2 (← V3.lit 0 0 1)
  for (k, off, r2, ir) in spheres do
    h ← sphere h o d (← V3.lit off.1 off.2.1 off.2.2) r2 ir k
  let r ← ifte (jTys := PathTys) .eq h.kind (← c32 0)
    (-- the sky, and the path ends
     do
      let s ← fmul (← fadd d.y (← fl 1.0)) (← fl 0.5)
      let sky ← V3.mk (← fma s (← fl 0.35) (← fl 0.04)) (← fma s (← fl 0.45) (← fl 0.06))
        <$> fma s (← fl 0.75) (← fl 0.12)
      let rad' ← V3.mk (← fma q.thr.x sky.x q.rad.x) (← fma q.thr.y sky.y q.rad.y) <$> fma q.thr.z sky.z q.rad.z
      let stop ← c32 (bounceCount spec)
      pure (Path.vals { q with bounce := stop, rad := rad' }))
    (do
      let n := h.n
      let p ← d.fmaS h.t o
      let po ← n.fmaS (← fl 0.001) p
      -- fog
      let fog ← fmin (← fmul h.t (← fl 0.03)) (← fl 0.35)
      let rad0 ← V3.mk (← fma q.thr.x (← fmul fog (← fl 0.04)) q.rad.x)
        (← fma q.thr.y (← fmul fog (← fl 0.07)) q.rad.y) <$> fma q.thr.z (← fmul fog (← fl 0.12)) q.rad.z
      -- the lights
      let (rng1, rs) ← draws q.rng 4
      let r0 := rs.getD 0 zero; let r1 := rs.getD 1 zero; let r2 := rs.getD 2 zero
      let posA ← V3.mk (← fma r0 (← fl 1.25) (← fl (-4.0))) (← fma r1 (← fl 0.35) (← fl 5.5))
        <$> fma r2 (← fl 1.25) (← fl (-2.0))
      let (lA, dA, lhA, ndlA) ← toLight po n posA
      -- the shadow test only where the light faces the surface
      let a ← ifte (jTys := [.f32, .f32, .f32]) .ne (← fcmp .gt ndlA zero) (← iconst .i8 0)
        (do
          let nd ← fmul (← callLocal sqrtApprox %[dA]) (← fl (-0.999))
          let b1 ← blocks po lhA (← V3.lit 1.35 0.05 4.7) 0.9025 nd
          let b2 ← blocks po lhA (← V3.lit (-1.35) 0.25 4.2) 0.5625 nd
          let b3 ← blocks po lhA (← V3.lit (-0.1) (-0.15) 6.5) 1.3225 nd
          let lit ← bxor (← bor b1 (← bor b2 b3)) (← iconst .i8 1)
          let acc ← addLight n0 lit ndlA dA (90, 74, 58)
          pure %[acc.x, acc.y, acc.z])
        (pure %[zero, zero, zero])
      let accA : V3 V := ⟨a.head, a.snd, a.thd⟩
      let posB ← V3.mk (← fma r2 (← fl 1.35) (← fl 4.5)) (← fma lA.x (← fl 0.45) (← fl 4.0))
        <$> fma r1 (← fl 1.45) (← fl (-7.0))
      let (_, dB, _, ndlB) ← toLight po n posB
      let accB ← addLight accA (← fcmp .gt ndlB zero) ndlB dB (40, 66, 110)
      let posC ← V3.mk (← fma r0 (← fl 2.0) zero) (← fma r1 (← fl 0.6) (← fl 6.5))
        <$> fma lA.x (← fl 1.8) (← fl 2.5)
      let (_, dC, _, ndlC) ← toLight po n posC
      let li ← addLight accB (← fcmp .gt ndlC zero) ndlC dC (28, 40, 72)
      -- the material
      let m ← ifte (jTys := PathTys) .eq h.kind (← c32 3)
        (-- glass: reflect or refract by Schlick's approximation
         do
          let cosD ← d.dot n
          let inside ← fcmp .gt cosD zero
          let nn ← V3.sel inside (← n.neg) n
          let cosi ← select inside cosD (← fneg cosD)
          let eta ← select inside (← fl 1.45) (← fl (1 / 1.45))
          let x ← fsub (← fl 1.0) cosi
          let x2 ← fmul x x
          let x5 ← fmul (← fmul x2 x2) x
          let fr ← fma x5 (← fl 0.96) (← fl 0.04)
          let (rng2, u) ← draw rng1
          let one ← fl 1.0
          let k ← fsub one (← fmul (← fsub one (← fmul cosi cosi)) (← fmul eta eta))
          let refl ← bor (← fcmp .lt k zero) (← fcmp .lt u fr)
          let rr ← ifte (jTys := PathTys) .ne refl (← iconst .i8 0)
            (do
              let c2 ← fmul (← d.dot nn) (← fl (-2.0))
              let d' ← nn.fmaS c2 d
              let thr : V3 V := ⟨← fmul q.thr.x (← fl 0.99), ← fmul q.thr.y (← fl 0.99), q.thr.z⟩
              pure (Path.vals { q with rng := rng2, o := po, d := d', rad := rad0, thr }))
            (do
              let sq : V .f32 ← callLocal sqrtApprox %[k]
              let ed ← d.scale eta
              let c2 ← fsub (← fmul eta cosi) sq
              let d' ← nn.fmaS c2 ed
              let thr : V3 V := ⟨← fmul q.thr.x (← fl 0.98), ← fmul q.thr.y (← fl 0.99), q.thr.z⟩
              pure (Path.vals { q with rng := rng2, o := p, d := d', rad := rad0, thr }))
          pure rr)
        (do
          let mm ← ifte (jTys := PathTys) .eq h.kind (← c32 4)
            (-- gold: a fuzzed reflection
             do
              let rad1 ← V3.mk (← fma q.thr.x (← fmul li.x (← fl 0.15)) rad0.x)
                (← fma q.thr.y (← fmul li.y (← fl 0.13)) rad0.y) <$> fma q.thr.z (← fmul li.z (← fl 0.09)) rad0.z
              let c2 ← fmul (← d.dot n) (← fl (-2.0))
              let d1 ← n.fmaS c2 d
              let (rng2, fz) ← draws rng1 3
              let fuzz : V3 V := ⟨fz.getD 0 zero, fz.getD 1 zero, fz.getD 2 zero⟩
              let d2 ← fuzz.fmaS (← fl 0.08) d1
              let thr ← q.thr.zip fmul (← colorLit spec.palette.metalSphere)
              let d3 ← d2.unit
              pure (Path.vals { q with rng := rng2, o := po, d := d3, rad := rad1, thr }))
            (-- diffuse: the ground's checker, the wall or the blue sphere
             do
              let cv ← ifte (jTys := [.f32, .f32, .f32]) .eq h.kind (← c32 1)
                (do
                  let ix ← fconv .toSint .i32 p.x
                  let iz ← fconv .toSint .i32 p.z
                  let odd ← icmp .ne (← band (← bxor ix iz) (← c32 1)) (← c32 0)
                  let g ← V3.sel odd (← colorLit spec.palette.groundLight) (← colorLit spec.palette.groundDark)
                  pure %[g.x, g.y, g.z])
                (do
                  let o ← V3.sel (← icmp .eq h.kind (← c32 2)) (← colorLit spec.palette.wall)
                    (← colorLit spec.palette.diffuseSphere)
                  pure %[o.x, o.y, o.z])
              let col : V3 V := ⟨cv.head, cv.snd, cv.thd⟩
              let e ← li.zip fmul col
              let rad1 ← V3.mk (← fma q.thr.x e.x rad0.x) (← fma q.thr.y e.y rad0.y) <$> fma q.thr.z e.z rad0.z
              let thr ← q.thr.zip fmul col
              let (rng2, js) ← draws rng1 3
              let jit : V3 V := ⟨js.getD 0 zero, js.getD 1 zero, js.getD 2 zero⟩
              let d1 ← n.fmaS (← fl 1.4) jit
              let d2 ← d1.unit
              pure (Path.vals { q with rng := rng2, o := po, d := d2, rad := rad1, thr }))
          pure mm)
      -- Russian roulette after the second bounce
      let s := Path.ofVals m
      let next ← iadd q.bounce (← c32 1)
      let rr ← ifte (jTys := [.i32, .i32, .f32, .f32, .f32]) .uge q.bounce (← c32 2)
        (do
          let pr ← fmax (← fmin (← fmax (← fmax s.thr.x s.thr.y) s.thr.z) (← fl 0.97)) (← fl 0.10)
          let (rng3, u) ← draw s.rng
          let inv : V .f32 ← callLocal rcpApprox %[pr]
          let dead ← fcmp .gt u pr
          pure %[← select dead (← c32 (bounceCount spec)) next, rng3,
            ← fmul s.thr.x inv, ← fmul s.thr.y inv, ← fmul s.thr.z inv])
        (pure %[next, s.rng, s.thr.x, s.thr.y, s.thr.z])
      let thr : V3 V := ⟨rr.thd, (rr.drop 3).head, (rr.drop 4).head⟩
      pure (Path.vals { s with bounce := rr.head, rng := rr.snd, thr }))
  pure (Path.ofVals r)

set_option maxRecDepth 8000 in
/-- **One pixel**, written to `out`. -/
def pixel (spec : SceneSpec) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64])
  let cta := ps.head; let tid := ps.snd; let out := (ps.drop 4).head
  let W := imageWidth spec; let H := imageHeight spec
  let tilesX := (W + 15) / 16
  let by_ ← udiv cta (← c32 tilesX)
  let px ← iadd (← ishlImm (← isub cta (← imul by_ (← c32 tilesX))) 4) (← band tid (← c32 15))
  let py ← iadd (← ishlImm by_ 4) (← ushrImm tid 4)
  when .ult px (← c32 W) do
  when .ult py (← c32 H) do
    let fx ← fcvtFromUint .f32 px
    let fy ← fcvtFromUint .f32 py
    let fw ← fl (Float.ofNat W)
    let fh ← fl (Float.ofNat H)
    let rng0 ← bxor (← iadd (← imul (← bxor px py) (← c32 1973)) (← c32 9277)) (← c32 26699)
    let zero ← fl 0.0
    let nS ← c32 (sampleCount spec)
    let nB ← c32 (bounceCount spec)
    let r ← wloop %[← c32 0, rng0, zero, zero, zero]
      (fun cs => pure (contIfULt cs.head nS, %[cs.snd, cs.thd, (cs.drop 3).head, (cs.drop 4).head], ()))
      (fun cs _ => do
        -- the camera ray
        let (ra, jx) ← draw cs.snd
        let (rb, jy) ← draw ra
        let (rc, lx) ← draw rb
        let (rd, ly) ← draw rc
        let u ← fmul (← fma (← fdiv (← fadd fx jx) fw) (← fl 2.0) (← fl (-1.0)))
          (← fl (Float.ofNat W / Float.ofNat H))
        let v ← fma (← fdiv (← fadd fy jy) fh) (← fl (-2.0)) (← fl 1.0)
        let ox ← fmul (← fadd lx (← fl (-0.5))) (← fl 0.07)
        let oy ← fma (← fadd ly (← fl (-0.5))) (← fl 0.07) (← fl 1.15)
        let o : V3 V := ⟨ox, oy, ← fl 2.7⟩
        let dir : V3 V := ⟨← fsub (← fmul u (← fl 4.8125)) ox,
          ← fsub (← fma v (← fl 4.8125) (← fl 1.15)) oy, ← fsub (← fl (-5.0)) o.z⟩
        let one ← fl 1.0
        let d0 ← dir.unit
        let b0 ← c32 0
        let q0 : Path V := ⟨b0, rd, o, d0, ⟨one, one, one⟩, ⟨zero, zero, zero⟩⟩
        let fin ← wloop q0.vals (fun c => pure (contIfULt c.head nB, c, ()))
          (fun c _ => do pure (Path.vals (← bounce spec (Path.ofVals c))))
        let e := Path.ofVals fin
        pure %[← iadd cs.head (← c32 1), e.rng, ← fadd cs.thd e.rad.x, ← fadd (cs.drop 3).head e.rad.y,
          ← fadd (cs.drop 4).head e.rad.z])
    -- tone map, vignette, and the BGRA word
    let inv ← fl (1.0 / Float.ofNat (sampleCount spec))
    let vx ← fsub (← fdiv fx fw) (← fl 0.5)
    let vy ← fsub (← fdiv fy fh) (← fl 0.5)
    let vig ← fmax (← fma (← fma vy vy (← fmul vx vx)) (← fl (-1.85)) (← fl 1.0)) (← fl 0.72)
    let tone (c : V .f32) : Prog V L (V .i32) := do
      let m ← fmul c inv
      let t : V .f32 ← callLocal sqrtApprox %[← fdiv m (← fadd m (← fl 1.0))]
      let w ← fmul (← fmul (← fmin t (← fl 1.0)) vig) (← fl 255.0)
      fconv .toSint .i32 (← fadd w (← fl 0.5))
    let rr ← tone r.snd
    let gg ← tone r.thd
    let bb ← tone (r.drop 3).head
    let word ← bor (← bor bb (← ishlImm gg 8)) (← bor (← ishlImm rr 16) (← c32 0xFF000000))
    storeI32 word (← at4 out (← iadd (← imul py (← c32 W)) px))

end

def kernel (spec : SceneSpec) : Kernel :=
  { name := "main", params := [.i64], code := emit (pixel spec) (idParams ++ [.i64]) }

end Body

/-- The renderer's module, or empty where the printer refuses it. -/
def ptxSource (spec : SceneSpec) : String :=
  match (Body.kernel spec).ptx with
  | .ok s => s
  | .error _ => ""

def ptxOff : Nat := 0x0100
def ptxRegion : Nat := 131072
def bindOff : Nat := ptxOff + ptxRegion
def filenameOff : Nat := bindOff + 16
def filenameRegion : Nat := 256
def clifIrOff : Nat := filenameOff + filenameRegion
def clifIrRegion : Nat := 4096
def bmpHeaderOff : Nat := clifIrOff + clifIrRegion
def pixelsOff : Nat := bmpHeaderOff + 54

open AlgorithmLib.Prog


/-- **The renderer in the driver's standard layout**: one allocation for the
    pixels and one kernel over it, a block of 256 threads per 16×16 tile. Its state
    lies in the region `clifIrOff` keeps. -/
def app (spec : SceneSpec) : Drv.App :=
  { bufs := [pixelBytes spec],
    kernels := [⟨ptxOff, ptxRegion, [0], ((imageWidth spec + 15) / 16 * ((imageHeight spec + 15) / 16), 1, 1),
      (256, 1, 1), []⟩],
    memSize := pixelsOff + pixelBytes spec, org := clifIrOff }

/-- Bring the driver up, render into the allocation, download the pixels
    beside the BMP header, free the allocation and write the file. -/
def code (spec : SceneSpec) : Prog V L Unit :=
  let a := app spec
  do
    let base ← basePtr
    a.setup base
    a.run base 0
    a.download base (← iaddImm base pixelsOff) 0 (pixelBytes spec)
    Drv.free base (a.bufSlot 0)
    let total ← iconst64 (54 + pixelBytes spec)
    let _ ← writeFile0 base filenameOff filenameRegion bmpHeaderOff total


def clifIrSource (spec : SceneSpec) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 (code spec))]

def payloads (spec : SceneSpec) : List UInt8 :=
  (app spec).initialMemory [ptxSource spec]
    [(filenameOff, stringToBytes spec.filename), (bmpHeaderOff, bmpHeader spec)]

def config (spec : SceneSpec) (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := (app spec).memSize,
  initial_memory := payloads spec
}

def algorithm : UInt32 := IR.mainFnIdx

def renderScene (spec : SceneSpec) : Except String (Artifact × UInt32) := do
  return (config spec (← clifIrSource spec), algorithm)

def defaultPalette : ScenePalette := {
  groundLight := checkedColor 189 191 204 (by decide) (by decide) (by decide)
  groundDark := checkedColor 26 28 33 (by decide) (by decide) (by decide)
  wall := checkedColor 117 87 77 (by decide) (by decide) (by decide)
  diffuseSphere := checkedColor 26 56 199 (by decide) (by decide) (by decide)
  metalSphere := checkedColor 245 199 82 (by decide) (by decide) (by decide)
}

def sunsetPalette : ScenePalette := {
  groundLight := checkedColor 226 184 126 (by decide) (by decide) (by decide)
  groundDark := checkedColor 38 31 40 (by decide) (by decide) (by decide)
  wall := checkedColor 148 75 70 (by decide) (by decide) (by decide)
  diffuseSphere := checkedColor 42 116 172 (by decide) (by decide) (by decide)
  metalSphere := checkedColor 255 175 71 (by decide) (by decide) (by decide)
}

def studioPalette : ScenePalette := {
  groundLight := checkedColor 204 207 202 (by decide) (by decide) (by decide)
  groundDark := checkedColor 35 38 40 (by decide) (by decide) (by decide)
  wall := checkedColor 92 108 102 (by decide) (by decide) (by decide)
  diffuseSphere := checkedColor 154 68 118 (by decide) (by decide) (by decide)
  metalSphere := checkedColor 222 222 210 (by decide) (by decide) (by decide)
}

def defaultScene : SceneSpec :=
  checkedScene 1280 720 128 5 defaultPalette "scene.bmp"
    (by decide) (by decide) (by decide) (by decide)

def previewScene : SceneSpec :=
  checkedScene 640 360 16 3 sunsetPalette "scene.bmp"
    (by decide) (by decide) (by decide) (by decide)

def studioScene : SceneSpec :=
  checkedScene 1920 1080 96 6 studioPalette "scene.bmp"
    (by decide) (by decide) (by decide) (by decide)

/-- The shipped scene's module prints, and fits its region with its
    terminating zero. -/
theorem ptx_fits : 0 < (ptxSource defaultScene).utf8ByteSize ∧
    (ptxSource defaultScene).utf8ByteSize < ptxRegion := by
  native_decide

-- To render a different valid scene, change `main` below to:
--
--   let (cfg, alg) := Scene.renderScene Scene.previewScene
--
-- or:
--
--   let (cfg, alg) := Scene.renderScene Scene.studioScene
--
-- These intentionally invalid examples show the dependent checks. Uncomment
-- one at a time and Lean rejects it before PTX or CLIF is generated.
--
--   def invalidZeroWidth : SceneSpec :=
--     checkedScene 0 720 128 5 defaultPalette "bad.bmp"
--       (by decide) (by decide) (by decide) (by decide)
--
--   def invalidZeroSamples : SceneSpec :=
--     checkedScene 1280 720 0 5 defaultPalette "bad.bmp"
--       (by decide) (by decide) (by decide) (by decide)
--
--   def invalidColor : Color :=
--     checkedColor 256 32 64 (by decide) (by decide) (by decide)
--
--   def invalidZeroBounces : SceneSpec :=
--     checkedScene 1280 720 128 0 defaultPalette "bad.bmp"
--       (by decide) (by decide) (by decide) (by decide)

end Scene

def Demo.Scene.main (args : List String) : IO Unit := do
  let (cfg, alg) ← Prog.orDie (Scene.renderScene Scene.defaultScene)
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[artifactEntry "scene_app" cfg]

#eval ShipScan.check "Demo.Scene" `Demo.Scene.main
