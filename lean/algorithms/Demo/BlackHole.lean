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

set_option maxRecDepth 8192
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace BlackHole

structure PositiveNat where
  value : Nat
  isPositive : value > 0

def PositiveNat.mkChecked (value : Nat) (isPositive : value > 0) : PositiveNat :=
  { value, isPositive }

instance : Coe PositiveNat Nat where
  coe n := n.value

/-- ============================================================
    Black hole scene specification.

    A type-checked Schwarzschild geometry + accretion disk renderer.
    All numerical sanity (positivity, ISCO, ordering) is enforced
    by the `checked` constructor's proofs — Lean refuses to build
    PTX for a spec with an inside-out disk or a camera inside the
    event horizon.

    Coordinate convention:
    * Black hole at the origin.
    * Schwarzschild radius `rs` (= 2M) is the natural length unit;
      we typically fix rs = 1.0 and scale everything else relative
      to it.
    * The accretion disk lies in the y = 0 (xz) plane with normal
      +y, between radii `diskInner` and `diskOuter`.
    * Camera at `(0, cameraHeight, cameraDistance)` looking at the
      origin (Cx = 0 keeps the plane-of-motion math clean).
    ============================================================ -/
structure BlackHoleSpec where
  width : PositiveNat
  height : PositiveNat
  schwarzschildRadius : Float
  diskInner : Float
  diskOuter : Float
  cameraHeight : Float
  cameraDistance : Float
  fovYDeg : Float
  stepCount : PositiveNat
  samplesPerPixel : PositiveNat
  dPhi : Float
  rMax : Float
  filename : String

def checkedBlackHole
    (width height : Nat) (stepCount samplesPerPixel : Nat)
    (schwarzschildRadius diskInner diskOuter
     cameraHeight cameraDistance fovYDeg dPhi rMax : Float)
    (filename : String)
    (widthPos : width > 0)
    (heightPos : height > 0)
    (stepsPos : stepCount > 0)
    (samplesPos : samplesPerPixel > 0)
    (_rsPos : schwarzschildRadius > 0.0)
    (_diskOrdered : diskOuter > diskInner)
    (_isco : diskInner ≥ 3.0 * schwarzschildRadius)
    (_cameraOutside : cameraDistance > diskOuter)
    (_dPhiPos : dPhi > 0.0)
    (_rMaxLarge : rMax > diskOuter) : BlackHoleSpec :=
  {
    width := PositiveNat.mkChecked width widthPos
    height := PositiveNat.mkChecked height heightPos
    stepCount := PositiveNat.mkChecked stepCount stepsPos
    samplesPerPixel := PositiveNat.mkChecked samplesPerPixel samplesPos
    schwarzschildRadius
    diskInner
    diskOuter
    cameraHeight
    cameraDistance
    fovYDeg
    dPhi
    rMax
    filename
  }

def imageWidth (spec : BlackHoleSpec) : Nat := spec.width.value
def imageHeight (spec : BlackHoleSpec) : Nat := spec.height.value
def stepCount (spec : BlackHoleSpec) : Nat := spec.stepCount.value
def samplesPerPixel (spec : BlackHoleSpec) : Nat := spec.samplesPerPixel.value
def pixelCount (spec : BlackHoleSpec) : Nat := imageWidth spec * imageHeight spec
def pixelBytes (spec : BlackHoleSpec) : Nat := pixelCount spec * 4

/-- ============================================================
    Precomputed scalar constants derived from the spec.

    These are all classical-Newtonian / metric-derived quantities
    we'd compute once on the CPU before launching the kernel.
    Doing it in Lean lets us bake them as `mov.f32` immediates and
    avoid the per-thread setup cost.
    ============================================================ -/

def cameraR0 (spec : BlackHoleSpec) : Float :=
  let h := spec.cameraHeight
  let d := spec.cameraDistance
  Float.sqrt (h * h + d * d)

/-- Camera basis vectors. Camera at `(0, h, D)`, looking at origin,
    world-up `(0, 1, 0)`. Algebra:

    forward = -C/|C|                      = (0, -h/L, -D/L)
    right   = normalize(forward × up)     = (1, 0, 0)          (since Cx=0, D>0)
    up_cam  = right × forward             = (0, D/L, -h/L)
-/
def cameraForward (spec : BlackHoleSpec) : Float × Float × Float :=
  let L := cameraR0 spec
  (0.0, -spec.cameraHeight / L, -spec.cameraDistance / L)

def cameraRight (_spec : BlackHoleSpec) : Float × Float × Float :=
  (1.0, 0.0, 0.0)

def cameraUp (spec : BlackHoleSpec) : Float × Float × Float :=
  let L := cameraR0 spec
  (0.0, spec.cameraDistance / L, -spec.cameraHeight / L)

/-- The fixed plane-of-motion basis vector pointing from the BH out
    to the camera: `u1 = C / |C|`. -/
def planeU1 (spec : BlackHoleSpec) : Float × Float × Float :=
  let L := cameraR0 spec
  (0.0, spec.cameraHeight / L, spec.cameraDistance / L)

def tanHalfFov (spec : BlackHoleSpec) : Float :=
  Float.tan (spec.fovYDeg * 0.017453292519943295 * 0.5)

def aspectRatio (spec : BlackHoleSpec) : Float :=
  Float.ofNat (imageWidth spec) / Float.ofNat (imageHeight spec)

def cosDPhi (spec : BlackHoleSpec) : Float := Float.cos spec.dPhi
def sinDPhi (spec : BlackHoleSpec) : Float := Float.sin spec.dPhi

/-- ============================================================
    BMP header (identical to the legacy renderer).
    ============================================================ -/
def bmpHeader (spec : BlackHoleSpec) : List UInt8 :=
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

/-! ## The kernels

Two kernels in one module, a thread a pixel and blocks of 256 threads a
16×16 tile, block `b` at tile `(b mod tilesX, b / tilesX)`.

`render_hdr` traces `samplesPerPixel` jittered rays a pixel. A ray's path is
a Schwarzschild null geodesic in its plane of motion, integrated in
`u = 1/r` over `φ` by semi-implicit Euler (`u'' + u = 3M u²`), the plane's
`cos φ`, `sin φ` advanced by the step's rotation. It ends at the horizon
(`r < rs`, black), past `rMax` (escape: a hashed starfield over a faint
galactic glow along the photon's final direction), or crossing the disk's
plane between its radii (the crossing interpolated from the last step; the
disk's colour is a temperature ramp warped and modulated by four octaves of
value noise, boosted by Doppler beaming and gravitational redshift `D⁴ g⁴`,
limb-brightened by the crossing's angle and tinted by `D g`). The samples'
mean is written as linear HDR, 16 bytes a pixel.

`composite_bloom` gathers the bright-pass (`max (c − 4.5) 0`) of each
pixel's 121×121 neighbourhood under three Gaussians at once, adds the
weighted glow to the pixel, then exposes, tone maps (ACES), vignettes,
takes the square root for gamma, adds a hashed grain and packs BGRA. -/

namespace Body
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- A pixel's tile coordinates and whether it lies in the image. -/
def pixelOf (W H : Nat) (cta tid : V .i32) : Prog V L (V .i32 × V .i32) := do
  let tilesX := (W + 15) / 16
  let by_ ← udiv cta (← c32 tilesX)
  let px ← iadd (← ishlImm (← isub cta (← imul by_ (← c32 tilesX))) 4) (← band tid (← c32 15))
  let py ← iadd (← ishlImm by_ 4) (← ushrImm tid 4)
  pure (px, py)

/-- `x − ⌊x⌋` for a non-negative `x`, as the hash wants it: through a 32-bit
    truncation read back unsigned. -/
def fract (x : V .f32) : Prog V L (V .f32) := do
  fsub x (← fcvtFromUint .f32 (← fconv .toSint .i32 x))

/-- A hash's fraction in `[0, 1)`: its bits over `2^32`. -/
def hashFrac (h : V .i32) : Prog V L (V .f32) := do
  fract (← fabs (← fmul (← fcvtFromUint .f32 h) (← f32c 0x2F800000)))

/-- `x · (x · (3 − 2x))`, the smoothstep weight. -/
def smooth (x : V .f32) : Prog V L (V .f32) := do
  fmul x (← fmul x (← fma x (← fl (-2.0)) (← fl 3.0)))

/-- One octave of value noise at `(sx, sy)`: the four corners' hashes,
    blended by smoothstep weights. -/
def octave (sx sy : V .f32) (s0 s1 s2 : Nat) : Prog V L (V .f32) := do
  let ix ← fconv .toSint .i32 sx
  let iy ← fconv .toSint .i32 sy
  let fx ← fsub sx (← fcvtFromUint .f32 ix)
  let fy ← fsub sy (← fcvtFromUint .f32 iy)
  let wx ← smooth fx
  let wy ← smooth fy
  let a ← imul ix (← c32 s0)
  let b ← imul iy (← c32 s1)
  let a1 ← iadd a (← c32 s0)
  let b1 ← iadd b (← c32 s1)
  let h (p q : V .i32) : Prog V L (V .f32) := do hashFrac (← imul (← bxor p q) (← c32 s2))
  let h00 ← h a b
  let h10 ← h a1 b
  let h01 ← h a b1
  let h11 ← h a1 b1
  let m0 ← fma wx (← fsub h10 h00) h00
  let m1 ← fma wx (← fsub h11 h01) h01
  fma wy (← fsub m1 m0) m0

/-- The plane's in-plane vectors at angle `φ`: `a = cos φ·u1 + sin φ·u2` and
    `b = −sin φ·u1 + cos φ·u2`, `u1 = (0, p1y, p1z)`. -/
def planeAB (c s p1y p1z : V .f32) (u2 : V3 V) (np1y np1z : V .f32) : Prog V L (V3 V × V3 V) := do
  let a : V3 V := ⟨← fmul s u2.x, ← fma s u2.y (← fmul c p1y), ← fma s u2.z (← fmul c p1z)⟩
  let b : V3 V := ⟨← fmul c u2.x, ← fma c u2.y (← fmul s np1y), ← fma c u2.z (← fmul s np1z)⟩
  pure (a, b)

/-- The geodesic's tangent at `r`, `φ`: `dr/dφ · a + r · b`, `dr/dφ = −r² du`. -/
def tangent (r du : V .f32) (a b : V3 V) : Prog V L (V3 V) := do
  let drdphi ← fneg (← fmul (← fmul r r) du)
  pure ⟨← fma r b.x (← fmul drdphi a.x), ← fma r b.y (← fmul drdphi a.y), ← fma r b.z (← fmul drdphi a.z)⟩

/-- An escaping photon's colour: the background floor, the galactic glow
    and, in one cell in about 1700, a star. -/
def escapeColor (d : V3 V) : Prog V L (V3 V) := do
  let c0 ← V3.lit 0.0008 0.0006 0.0014
  let gd ← fabs (← fma d.z (← fl (-0.375)) (← fmul d.y (← fl 0.927)))
  let g1 ← fmul gd (← fl 3.125)
  let g : V .f32 ← callLocal ex2Approx %[← fmul (← fmul g1 g1) (← fl (-1.4426950408889634))]
  let c1 : V3 V := ⟨← fma g (← fl 0.045) c0.x, ← fma g (← fl 0.036) c0.y, ← fma g (← fl 0.024) c0.z⟩
  let cell (x : V .f32) : Prog V L (V .i32) := do
    fconv .toSint .i32 (← fma x (← fl 700.0) (← fl 700.0))
  let hx ← imul (← cell d.x) (← c32 374761393)
  let hy ← imul (← cell d.y) (← c32 668265263)
  let hz ← imul (← cell d.z) (← c32 2147483647)
  let h ← imul (← bxor (← bxor hx hy) hz) (← c32 1274126177)
  let r ← ifte (jTys := [.f32, .f32, .f32]) .ne (← fcmp .gt (← hashFrac h) (← fl 0.9994)) (← iconst .i8 0)
    (do
      let h2 ← iadd (← imul h (← c32 1664525)) (← c32 1013904223)
      let f ← hashFrac h2
      let f2 ← fmul f f
      let inten ← fma (← fmul f2 f2) (← fl 3200.0) (← fl 40.0)
      let h3 ← iadd (← imul h2 (← c32 22695477)) (← c32 1)
      let t ← hashFrac h3
      pure %[← fma inten (← fma t (← fl 0.3) (← fl 0.9)) c1.x,
        ← fma inten (← fma t (← fl (-0.1)) (← fl 1.05)) c1.y,
        ← fma inten (← fma t (← fl (-0.4)) (← fl 1.25)) c1.z])
    (pure %[c1.x, c1.y, c1.z])
  pure ⟨r.head, r.snd, r.thd⟩

/-- The disk's colour where the photon crossed it at `r`, `φ`. -/
def diskColor (spec : BlackHoleSpec) (r du c s p1y p1z np1y np1z : V .f32) (u2 : V3 V) : Prog V L (V3 V) := do
  let (a, b) ← planeAB c s p1y p1z u2 np1y np1z
  let px ← fmul r a.x
  let pz ← fmul r a.z
  -- four octaves of value noise on (Px, Pz)
  let octs : List (Float × Float × Nat × Nat × Nat × Float) :=
    [(0.55, 1.54, 374761393, 668265263, 1274126177, 0.78), (1.30, 1.30, 1597334677, 3812015801, 2654435769, 0.15),
     (3.10, 3.10, 2654435761, 40503, 2246822519, 0.05), (7.40, 7.40, 1865811581, 506832829, 3033965713, 0.02)]
  let mut fbm ← fl 0.0
  for (kx, ky, s0, s1, s2, amp) in octs do
    let o ← octave (← fma px (← fl kx) (← fl 4096.0)) (← fma pz (← fl ky) (← fl 4096.0)) s0 s1 s2
    fbm ← fma o (← fl amp) fbm
  let pert ← fma fbm (← fl 2.4) (← fl (-1.2))
  let rp ← fsub (← fadd r pert) (← fl spec.diskInner)
  let t0 ← fdiv rp (← fl (spec.diskOuter - spec.diskInner))
  let t ← fmin (← fmax t0 (← fl 0.0)) (← fl 1.0)
  let bright ← fma fbm (← fl 2.30) (← fl 0.10)
  let base : V3 V := ⟨← fmul (← fma t (← fl (-0.60)) (← fl 1.20)) bright,
    ← fmul (← fma t (← fl (-0.73)) (← fl 0.95)) bright, ← fmul (← fma t (← fl (-0.50)) (← fl 0.55)) bright⟩
  -- the photon's direction, against the traced tangent
  let n ← (← (← tangent r du a b).unit).neg
  -- β: the orbital speed √(M/r) along the photon's direction
  let ox ← fdiv (← fneg pz) r
  let oz ← fdiv px r
  let v : V .f32 ← callLocal sqrtApprox %[← fdiv (← fl (spec.schwarzschildRadius * 0.5)) r]
  let beta ← fmul v (← fma oz n.z (← fmul ox n.x))
  let one ← fl 1.0
  let sq1 : V .f32 ← callLocal sqrtApprox %[← fmax (← fma (← fneg v) v one) (← fl 0.0001)]
  let dd ← fdiv sq1 (← fmax (← fsub one beta) (← fl 0.05))
  let d2 ← fmul dd dd
  let gq : V .f32 ← callLocal sqrtApprox %[← fmax (← fsub one (← fdiv (← fl spec.schwarzschildRadius) r))
    (← fl 0.0001)]
  let g2 ← fmul gq gq
  let boost0 ← fmul (← fmul d2 d2) (← fmul g2 g2)
  let limb ← fmin (← fdiv (← fl 0.45) (← fadd (← fabs n.y) (← fl 0.12))) (← fl 2.6)
  let boost ← fmul boost0 limb
  let sh ← fmin (← fmax (← fmul dd gq) (← fl 0.25)) (← fl 4.0)
  let rs : V .f32 ← callLocal sqrtApprox %[sh]
  let inv ← fdiv one rs
  pure ⟨← fmul (← fmul base.x boost) inv, ← fmul base.y boost, ← fmul (← fmul base.z boost) rs⟩

/-- The integration's state: the step (the limit once the path has ended),
    the outcome (0 horizon, 1 disk, 2 escape), `u`, `du/dφ`, `cos φ`,
    `sin φ`, the last step's `y`, and `r`. -/
abbrev OrbTys : List ClifTy := [.i32, .i32, .f32, .f32, .f32, .f32, .f32, .f32]

set_option maxRecDepth 8000 in
/-- **One pixel of the HDR render**, written to `hdr`. -/
def render (spec : BlackHoleSpec) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64])
  let cta := ps.head; let tid := ps.snd; let hdr := (ps.drop 4).head
  let W := imageWidth spec; let H := imageHeight spec
  let (px, py) ← pixelOf W H cta tid
  when .ult px (← c32 W) do
  when .ult py (← c32 H) do
    let (fx, fy, fz) := cameraForward spec
    let (_, uy, uz) := cameraUp spec
    let (_, p1yF, p1zF) := planeU1 spec
    let r0 := cameraR0 spec
    let p1y ← fl p1yF; let p1z ← fl p1zF
    let np1y ← fl (-p1yF); let np1z ← fl (-p1zF)
    let fxp ← fcvtFromUint .f32 px
    let fyp ← fcvtFromUint .f32 py
    let rng0 ← bxor (← iadd (← imul (← bxor px py) (← c32 1973)) (← c32 9277)) (← c32 26699)
    let zero ← fl 0.0
    let nS ← c32 (samplesPerPixel spec)
    let nSteps ← c32 (stepCount spec)
    let res ← wloop %[← c32 0, rng0, zero, zero, zero]
      (fun cs => pure (contIfULt cs.head nS, %[cs.thd, (cs.drop 3).head, (cs.drop 4).head], ()))
      (fun cs _ => do
        let jit (rng : V .i32) : Prog V L (V .i32 × V .f32) := do
          let r ← iadd (← imul rng (← c32 1664525)) (← c32 1013904223)
          pure (r, ← fadd (← hashFrac r) (← fl (-0.5)))
        let (ra, jx) ← jit cs.snd
        let (rb, jy) ← jit ra
        let sx ← fadd (← fadd fxp jx) (← fl 0.5)
        let sy ← fadd (← fadd fyp jy) (← fl 0.5)
        let tanH ← fl (tanHalfFov spec)
        let u0 ← fma (← fdiv sx (← fl (Float.ofNat W))) (← fl 2.0) (← fl (-1.0))
        let v0 ← fma (← fdiv sy (← fl (Float.ofNat H))) (← fl (-2.0)) (← fl 1.0)
        let u ← fmul (← fmul u0 (← fl (aspectRatio spec))) tanH
        let v ← fmul v0 tanH
        let rd ← (V3.mk (← fadd u (← fl fx)) (← fma v (← fl uy) (← fl fy)) (← fma v (← fl uz) (← fl fz))).unit
        -- the plane of motion: u1 towards the camera, u2 the ray's part across it
        let du1 ← fma rd.z p1z (← fmul rd.y p1y)
        let ndu1 ← fneg du1
        let perp : V3 V := ⟨rd.x, ← fma ndu1 p1y rd.y, ← fma ndu1 p1z rd.z⟩
        let pp ← perp.dot perp
        let u2 ← perp.scale (← callLocal rsqrtApprox %[pp])
        let du2 : V .f32 ← callLocal sqrtApprox %[pp]
        let uInit ← fl (1.0 / r0)
        let duInit ← fneg (← fmul uInit (← fdiv du1 du2))
        -- the geodesic
        let orb ← wloop (%[← c32 0, ← c32 2, uInit, duInit, ← fl 1.0, zero, ← fl spec.cameraHeight, ← fl r0] :
            Vals V OrbTys)
          (fun o => pure (contIfULt o.head nSteps, o, ()))
          (fun o _ => do
            let uu := o.thd; let du := (o.drop 3).head; let cph := (o.drop 4).head; let sph := (o.drop 5).head
            let prevY := (o.drop 6).head; let rPrev := (o.drop 7).head
            let t ← fsub (← fmul (← fmul uu uu) (← fl (3.0 * spec.schwarzschildRadius * 0.5))) uu
            let dPhi ← fl spec.dPhi
            let du' ← fma dPhi t du
            let u' ← fma dPhi du' uu
            let cD ← fl (cosDPhi spec)
            let sph' ← fma cph (← fl (sinDPhi spec)) (← fmul sph cD)
            let cph' ← fma cph cD (← fmul sph (← fl (-(sinDPhi spec))))
            let r ← fdiv (← fl 1.0) u'
            let stop ← c32 (stepCount spec)
            let next ← iadd o.head (← c32 1)
            let cy ← fmul r (← fma sph' u2.y (← fmul cph' p1y))
            -- horizon, escape, a disk crossing, or on: the rare cases off the common path
            let horizon ← fcmp .lt r (← fl spec.schwarzschildRadius)
            let escape ← fcmp .gt r (← fl spec.rMax)
            let cross ← fcmp .lt (← fmul prevY cy) zero
            let cont : Vals V OrbTys := %[next, o.snd, u', du', cph', sph', cy, r]
            ifte (jTys := OrbTys) .ne (← bor horizon (← bor escape cross)) (← iconst .i8 0)
              (ifte (jTys := OrbTys) .ne horizon (← iconst .i8 0)
                (pure %[stop, ← c32 0, u', du', cph', sph', cy, r])
                (ifte (jTys := OrbTys) .ne escape (← iconst .i8 0)
                  (pure %[stop, o.snd, u', du', cph', sph', cy, r])
                  (do
                    let tt ← fdiv prevY (← fsub prevY cy)
                    let rHit ← fma tt (← fsub r rPrev) rPrev
                    let inDisk ← band (← fcmp .ge rHit (← fl spec.diskInner)) (← fcmp .le rHit (← fl spec.diskOuter))
                    ifte (jTys := OrbTys) .ne inDisk (← iconst .i8 0)
                      (pure %[stop, ← c32 1, u', ← fma tt (← fsub du' du) du, ← fma tt (← fsub cph' cph) cph,
                        ← fma tt (← fsub sph' sph) sph, cy, rHit])
                      (pure cont))))
              (pure cont))
        let outcome := orb.snd
        let du := (orb.drop 3).head; let cph := (orb.drop 4).head; let sph := (orb.drop 5).head
        let r := (orb.drop 7).head
        let col ← ifte (jTys := [.f32, .f32, .f32]) .eq outcome (← c32 0)
          (pure %[zero, zero, zero])
          (ifte (jTys := [.f32, .f32, .f32]) .eq outcome (← c32 1)
            (do
              let k ← diskColor spec r du cph sph p1y p1z np1y np1z u2
              pure %[k.x, k.y, k.z])
            (do
              let (a, b) ← planeAB cph sph p1y p1z u2 np1y np1z
              let k ← escapeColor (← (← tangent r du a b).unit)
              pure %[k.x, k.y, k.z]))
        pure %[← iadd cs.head (← c32 1), rb, ← fadd cs.thd col.head, ← fadd (cs.drop 3).head col.snd,
          ← fadd (cs.drop 4).head col.thd])
    let inv ← fl (1.0 / Float.ofNat (samplesPerPixel spec))
    let at_ ← iadd hdr (← ishlImm (← ext (← iadd (← imul py (← c32 W)) px)) 4)
    store (← fmul res.head inv) at_
    store (← fmul res.snd inv) (← iaddImm at_ 4)
    store (← fmul res.thd inv) (← iaddImm at_ 8)

set_option maxRecDepth 8000 in
/-- **One pixel of the composite**: bloom gathered from `hdr`, tone mapped
    and packed to `bgra`. -/
def composite (spec : BlackHoleSpec) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let cta := ps.head; let tid := ps.snd
  let hdr := (ps.drop 4).head; let bgra := (ps.drop 5).head
  let W := imageWidth spec; let H := imageHeight spec
  let (px, py) ← pixelOf W H cta tid
  when .ult px (← c32 W) do
  when .ult py (← c32 H) do
    let radius : Nat := 60
    let diam := 2 * radius + 1
    let sig : List Float := [1.1, 10.0, 32.0]
    let ks := sig.map fun s => -1.4426950408889634 / (2.0 * s * s)
    let norms := (sig.zip [0.48, 0.06, 0.012]).map fun (s, w) => w / (2.0 * 3.141592653589793 * s * s)
    let idx ← iadd (← imul py (← c32 W)) px
    let own ← iadd hdr (← ishlImm (← ext idx) 4)
    let oR ← loadF32 own
    let oG ← loadF32 (← iaddImm own 4)
    let oB ← loadF32 (← iaddImm own 8)
    let zero ← fl 0.0
    let thr ← fl 4.5
    let k0 ← fl (ks.getD 0 0); let k1 ← fl (ks.getD 1 0); let k2 ← fl (ks.getD 2 0)
    let nDiam ← c32 diam
    let negR ← c32 (2 ^ 32 - radius)
    let nyBase ← iadd py negR
    let nxBase ← iadd px negR
    let fR ← fl (-(Float.ofNat radius))
    let accTys : List ClifTy := .i32 :: List.replicate 9 .f32
    let z9 : Vals V (List.replicate 9 .f32) := Vals.ofFn fun _ => zero
    let rows ← wloop (tys := accTys) (.cons (← c32 0) z9)
      (fun cs => pure (contIfULt cs.head nDiam, cs.tail, ()))
      (fun cs _ => do
        let dy := cs.head
        let ny ← iadd nyBase dy
        let accs := cs.tail
        let next ← iadd dy (← c32 1)
        let r ← ifte (jTys := List.replicate 9 .f32) .ult ny (← c32 H)
          (do
            let dyf ← fadd (← fcvtFromUint .f32 dy) fR
            let dy2 ← fmul dyf dyf
            let rowBase ← imul ny (← c32 W)
            let cols ← wloop (tys := accTys) (.cons (← c32 0) accs)
              (fun ds => pure (contIfULt ds.head nDiam, ds.tail, ()))
              (fun ds _ => do
                let dx := ds.head
                let nx ← iadd nxBase dx
                let a := Vals.uniformToList (ds.tail : Vals V (List.replicate 9 .f32))
                let g (i : Nat) := a.getD i zero
                let r2 ← ifte (jTys := List.replicate 9 .f32) .ult nx (← c32 W)
                  (do
                    let dxf ← fadd (← fcvtFromUint .f32 dx) fR
                    let d2 ← fma dxf dxf dy2
                    let ws : V .f32 ← callLocal ex2Approx %[← fmul d2 k0]
                    let wm : V .f32 ← callLocal ex2Approx %[← fmul d2 k1]
                    let wl : V .f32 ← callLocal ex2Approx %[← fmul d2 k2]
                    let nb ← iadd hdr (← ishlImm (← ext (← iadd rowBase nx)) 4)
                    let bR ← fmax (← fsub (← loadF32 nb) thr) zero
                    let bG ← fmax (← fsub (← loadF32 (← iaddImm nb 4)) thr) zero
                    let bB ← fmax (← fsub (← loadF32 (← iaddImm nb 8)) thr) zero
                    let out := [← fma bR ws (g 0), ← fma bG ws (g 1), ← fma bB ws (g 2),
                      ← fma bR wm (g 3), ← fma bG wm (g 4), ← fma bB wm (g 5),
                      ← fma bR wl (g 6), ← fma bG wl (g 7), ← fma bB wl (g 8)]
                    pure (Vals.ofFn fun i => out.getD i.val zero))
                  (pure ds.tail)
                pure (.cons (← iadd dx (← c32 1)) r2))
            pure cols)
          (pure accs)
        pure (.cons next r))
    let a := Vals.uniformToList (rows : Vals V (List.replicate 9 .f32))
    let g (i : Nat) := a.getD i zero
    let nS ← fl (norms.getD 0 0); let nM ← fl (norms.getD 1 0); let nL ← fl (norms.getD 2 0)
    let bloom (i : Nat) : Prog V L (V .f32) := do
      fma (g (6 + i)) nL (← fma (g (3 + i)) nM (← fmul (g i) nS))
    let exposure ← fl 0.24
    let cR ← fmul (← fadd oR (← bloom 0)) exposure
    let cG ← fmul (← fadd oG (← bloom 1)) exposure
    let cB ← fmul (← fadd oB (← bloom 2)) exposure
    let aces (x : V .f32) : Prog V L (V .f32) := do
      let num ← fmul x (← fma x (← fl 2.51) (← fl 0.03))
      let den ← fma x (← fma x (← fl 2.43) (← fl 0.59)) (← fl 0.14)
      fmin (← fmax (← fdiv num den) zero) (← fl 1.0)
    let vx ← fma (← fcvtFromUint .f32 px) (← fl (1.0 / (Float.ofNat W * 0.5))) (← fl (-1.0))
    let vy ← fma (← fcvtFromUint .f32 py) (← fl (1.0 / (Float.ofNat H * 0.5))) (← fl (-1.0))
    let vig ← fmax (← fma (← fma vy vy (← fmul vx vx)) (← fl (-0.22)) (← fl 1.0)) (← fl 0.65)
    let hg ← imul (← bxor (← imul px (← c32 374761393)) (← imul py (← c32 1597334677))) (← c32 1274126177)
    let grain ← fma (← hashFrac hg) (← fl 0.014) (← fl (-0.007))
    let finish (x : V .f32) : Prog V L (V .i32) := do
      let m ← fmul (← aces x) vig
      let s : V .f32 ← callLocal sqrtApprox %[m]
      let c ← fmin (← fmax (← fadd s grain) zero) (← fl 1.0)
      fconv .toSint .i32 (← fma c (← fl 255.0) (← fl 0.5))
    let rr ← finish cR
    let gg ← finish cG
    let bb ← finish cB
    let word ← bor (← bor bb (← ishlImm gg 8)) (← bor (← ishlImm rr 16) (← c32 0xFF000000))
    storeI32 word (← iadd bgra (← ishlImm (← ext idx) 2))

end

def kernels (spec : BlackHoleSpec) : List Kernel :=
  [{ name := "render_hdr", params := [.i64], code := emit (render spec) (idParams ++ [.i64]) },
   { name := "composite_bloom", params := [.i64, .i64], code := emit (composite spec) (idParams ++ [.i64, .i64]) }]

end Body

/-- The two kernels' module, or empty where the printer refuses one. -/
def ptxSource (spec : BlackHoleSpec) : String :=
  match AlgorithmLib.Kernel.Kernel.module (Body.kernels spec) with
  | .ok s => s
  | .error _ => ""

/-- ============================================================
    Memory layout — extended for the 2-kernel HDR → bloom →
    BGRA pipeline.  We now hold:

      * one PTX text region with two `.entry` functions
      * two kernel-name regions and two binding-descriptor regions,
        which nothing reads
      * the existing filename, BMP-header, and final BGRA-pixel
        regions

    The HDR scratch buffer itself lives on the device only; CPU
    memory just holds the binding descriptors that reference it.
    ============================================================ -/
def ptxOff : Nat := 0x0100
def ptxRegion : Nat := 262144
def nameOffRender : Nat := ptxOff + ptxRegion
def nameRegion : Nat := 32
def nameOffComposite : Nat := nameOffRender + nameRegion
def bindOffA : Nat := nameOffComposite + nameRegion
def bindRegionA : Nat := 16
def bindOffB : Nat := bindOffA + bindRegionA
def bindRegionB : Nat := 16
def filenameOff : Nat := bindOffB + bindRegionB
def filenameRegion : Nat := 256
def clifIrOff : Nat := filenameOff + filenameRegion
def clifIrRegion : Nat := 4096
def bmpHeaderOff : Nat := clifIrOff + clifIrRegion
def pixelsOff : Nat := bmpHeaderOff + 54

def hdrPixelBytes (spec : BlackHoleSpec) : Nat := pixelCount spec * 16

open AlgorithmLib.Prog


/-- **The renderer in the driver's standard layout**, a block of 256 threads
    a 16×16 tile: the HDR scratch (RGB
    `f32`, padded to 16 bytes a pixel) and the final BGRA pixels as two
    allocations, and two entry points of one module, the HDR render over the
    scratch and the bloom composite over both. Its state lies in the region
    `clifIrOff` keeps. -/
def app (spec : BlackHoleSpec) : Drv.App :=
  let grid := ((imageWidth spec + 15) / 16 * ((imageHeight spec + 15) / 16), 1, 1)
  { bufs := [hdrPixelBytes spec, pixelBytes spec],
    kernels := [⟨ptxOff, ptxRegion, [0], grid, (256, 1, 1), []⟩, ⟨ptxOff, ptxRegion, [0, 1], grid, (256, 1, 1), []⟩],
    entries := ["render_hdr", "composite_bloom"],
    memSize := pixelsOff + pixelBytes spec, org := clifIrOff }

/-- Bring the driver up, render the HDR image, composite it, download the
    pixels beside the BMP header, free both allocations and write the file.
    The default stream orders the two launches. -/
def code (spec : BlackHoleSpec) : Prog V L Unit :=
  let a := app spec
  do
    let base ← basePtr
    a.setup base
    a.run base 0
    a.run base 1
    a.download base (← iaddImm base pixelsOff) 1 (pixelBytes spec)
    Drv.free base (a.bufSlot 0)
    Drv.free base (a.bufSlot 1)
    let total ← iconst64 (54 + pixelBytes spec)
    let _ ← writeFile0 base filenameOff filenameRegion bmpHeaderOff total


-- Generic in the spec, so there is no instance to `decide` at; the
-- obligation is a parameter instead, discharged at the spec that ships.
def clifIrSource (spec : BlackHoleSpec) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 (code spec))]

def payloads (spec : BlackHoleSpec) : List UInt8 :=
  (app spec).initialMemory [ptxSource spec, ptxSource spec]
    [(filenameOff, stringToBytes spec.filename), (bmpHeaderOff, bmpHeader spec)]

def config (spec : BlackHoleSpec) (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := (app spec).memSize,
  initial_memory := payloads spec
}

def algorithm : UInt32 := IR.mainFnIdx

def renderScene (spec : BlackHoleSpec) : Except String (Artifact × UInt32) := do
  return (config spec (← clifIrSource spec), algorithm)

/-- ============================================================
    Preset specs.  Each must pass every dependent check in
    `checkedBlackHole`; flip a parameter to nonsense and Lean
    refuses to elaborate the file.
    ============================================================ -/
def defaultBlackHole : BlackHoleSpec :=
  -- Schwarzschild radius rs = 1 sets the unit; disk runs from
  -- the ISCO (3·rs) out to 12·rs.  Camera tilted ~16° above the
  -- equatorial plane and well outside the disk so we get a wide
  -- lensed view (Interstellar-style) with the back-disk wrapping
  -- over the top of the BH.
  checkedBlackHole
    (width := 1280) (height := 720) (stepCount := 1500) (samplesPerPixel := 512)
    (schwarzschildRadius := 1.0)
    (diskInner := 3.0)
    (diskOuter := 12.0)
    (cameraHeight := 1.4)
    (cameraDistance := 16.0)
    (fovYDeg := 45.0)
    (dPhi := 0.004)
    (rMax := 100.0)
    (filename := "blackhole.bmp")
    (by decide) (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    (by native_decide) (by native_decide) (by native_decide)

/-- The shipped spec's module prints, and fits its region with its
    terminating zero. -/
theorem ptx_fits : 0 < (ptxSource defaultBlackHole).utf8ByteSize ∧
    (ptxSource defaultBlackHole).utf8ByteSize < ptxRegion := by
  native_decide

end BlackHole

def Demo.BlackHole.main (args : List String) : IO Unit := do
  let (cfg, alg) ← Prog.orDie (BlackHole.renderScene BlackHole.defaultBlackHole)
  let jsonEntry := artifactEntry "blackhole_app" cfg
  let outputDir ← requireOutputDir args
  emitArtifacts outputDir #[jsonEntry]

#eval ShipScan.check "Demo.BlackHole" `Demo.BlackHole.main
