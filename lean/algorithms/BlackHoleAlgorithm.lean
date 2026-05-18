import AlgorithmLib
set_option maxRecDepth 8192
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Algorithm

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

def fImm (x : Float) : FImm := AlgorithmLib.PTX.FImm.float x
def fNat (n : Nat) : FImm := AlgorithmLib.PTX.FImm.nat n

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

/-- ============================================================
    PTX register naming helpers + named slots for state we touch
    from multiple emitter blocks.
    ============================================================ -/
private def pReg (n : Nat) : Reg .pred := ⟨s!"%p{n}"⟩
private def uReg (n : Nat) : Reg .u32 := ⟨s!"%r{n}"⟩
private def dReg (n : Nat) : Reg .u64 := ⟨s!"%rd{n}"⟩
private def fReg (n : Nat) : Reg .f32 := ⟨s!"%f{n}"⟩

namespace BHReg
-- Pixel + output address.
def pixelX : Reg .u32 := uReg 4
def pixelY : Reg .u32 := uReg 5

-- Per-pixel ray direction (in world space, unit length).
def rdx : Reg .f32 := fReg 50
def rdy : Reg .f32 := fReg 51
def rdz : Reg .f32 := fReg 52

-- Plane-of-motion second basis vector u2 (3-vector).
def u2x : Reg .f32 := fReg 53
def u2y : Reg .f32 := fReg 54
def u2z : Reg .f32 := fReg 55

-- ODE state.
def u    : Reg .f32 := fReg 80     -- 1/r
def du   : Reg .f32 := fReg 81     -- du/dφ
def cphi : Reg .f32 := fReg 82     -- cos φ
def sphi : Reg .f32 := fReg 83     -- sin φ
def prevY : Reg .f32 := fReg 84    -- y-coordinate of previous step
def rCur  : Reg .f32 := fReg 85    -- 1/u_curr cached after step
def stepIdx : Reg .u32 := uReg 30

-- Outcome flag (0 = horizon, 1 = disk, 2 = escape).
def outcome : Reg .u32 := uReg 31

-- Final linear RGB (pre-tonemap).
def cR : Reg .f32 := fReg 200
def cG : Reg .f32 := fReg 201
def cB : Reg .f32 := fReg 202
end BHReg

/-- ============================================================
    Emit block 1 — kernel header, pixel coords, constants,
    primary ray, plane-of-motion basis (u2 + initial u, du).

    Camera basis vectors and plane u1 are precomputed in Lean and
    burnt in as f32 immediates.
    ============================================================ -/
private def emitBHSetupAndRay (spec : BlackHoleSpec) : PTX Unit := do
  declPredRegs 48
  declU32Regs 128
  declU64Regs 16
  declF32Regs 512
  -- Block / thread → pixel coords.
  movR (uReg 0) ctaX
  movR (uReg 1) ctaY
  movR (uReg 2) tidX
  movR (uReg 3) tidY
  madLoRC (BHReg.pixelX) (uReg 0) (16) (uReg 2)
  madLoRC (BHReg.pixelY) (uReg 1) (16) (uReg 3)
  setpGeI (pReg 0) (BHReg.pixelX) (imageWidth spec)
  braIf (pReg 0) "DONE"
  setpGeI (pReg 1) (BHReg.pixelY) (imageHeight spec)
  braIf (pReg 1) "DONE"
  -- Output pixel address = data_ptr + (y·W + x) · 4.
  -- HDR pixel base (16 B/pixel) — final write at the end of the
  -- sample loop will recompute the address since we don't use it
  -- elsewhere in this kernel.
  madLoRC (uReg 6) (BHReg.pixelY) (imageWidth spec) (BHReg.pixelX)
  -- ── Reusable constants (HOISTED out of sample loop) ──────
  -- Geometry / physics.
  let rs := spec.schwarzschildRadius
  let m  := rs * 0.5
  movFI (fReg 20) (rs : Float)                  -- rs
  movFI (fReg 21) (3.0 * m : Float)             -- 3M  (used in ODE: u'' = -u + 3M·u²)
  movFI (fReg 22) (spec.diskInner : Float)
  movFI (fReg 23) (spec.diskOuter : Float)
  movFI (fReg 24) (spec.rMax : Float)
  movFI (fReg 25) (spec.dPhi : Float)
  movFI (fReg 26) (cosDPhi spec : Float)
  movFI (fReg 27) (sinDPhi spec : Float)
  movFI (fReg 28) (m : Float)                   -- M (for Doppler v_orbit = √(M/r))
  -- Camera basis (precomputed, Cx = 0 so right = (1,0,0)).
  let (fx, fy, fz) := cameraForward spec
  let (_, _, _) := cameraRight spec
  let (_, uy, uz) := cameraUp spec
  let (_, p1y, p1z) := planeU1 spec
  let r0 := cameraR0 spec
  movFI (fReg 30) (fx : Float)
  movFI (fReg 31) (fy : Float)
  movFI (fReg 32) (fz : Float)
  movFI (fReg 33) (uy : Float)                  -- up_cam.y
  movFI (fReg 34) (uz : Float)                  -- up_cam.z
  movFI (fReg 35) (p1y : Float)                 -- u1.y
  movFI (fReg 36) (p1z : Float)                 -- u1.z
  movFI (fReg 37) (r0 : Float)                  -- |C|
  movFI (fReg 38) (1.0 / r0 : Float)            -- 1/|C| = u_init
  movFI (fReg 39) (spec.cameraHeight : Float)   -- C.y (= y at φ=0)
  movFI (fReg 40) (tanHalfFov spec : Float)
  movFI (fReg 41) (aspectRatio spec : Float)
  movFI (fReg 42) (Float.ofNat (imageWidth spec) : Float)
  movFI (fReg 43) (Float.ofNat (imageHeight spec) : Float)
  movFI (fReg 44) (0.0 : Float)
  movFI (fReg 45) (1.0 : Float)
  movFI (fReg 46) (255.0 : Float)
  -- fReg 47 = 2^-32 — the inverse-uint32 scale used everywhere we
  -- convert a hash result to a [0,1) fract.  Loaded as the exact
  -- IEEE-754 bit pattern because `Float.toString` of 2.328e-10
  -- rounds to "0.000000" (only 6 decimal places of precision),
  -- which would silently zero every hash → fract conversion.
  movFI (fReg 47) (0x2F800000 : UInt32)
  -- Integer pixel coords as float, used as the center of each sub-pixel
  -- jitter range.
  cvtF32 (fReg 0) (BHReg.pixelX)
  cvtF32 (fReg 1) (BHReg.pixelY)
  -- Per-pixel accumulator (sum of all sample-contributions).
  movFI (fReg 10) (0.0 : Float)
  movFI (fReg 11) (0.0 : Float)
  movFI (fReg 12) (0.0 : Float)
  -- RNG state — distinct per pixel so jitter is decorrelated.
  xorRR (uReg 25) (BHReg.pixelX) (BHReg.pixelY)
  madLoRII (uReg 25) (uReg 25) (1973) (9277)
  xorRI (uReg 25) (uReg 25) (26699)
  -- Sample counter.
  movRC (uReg 26) 0
  label "SAMPLE_LOOP"
  setpGeI (pReg 11) (uReg 26) (samplesPerPixel spec)
  braIf (pReg 11) "SAMPLE_DONE"
  -- ── Sub-pixel jitter (two independent [-0.5, +0.5) offsets) ──
  madLoRII (uReg 25) (uReg 25) (1664525) (1013904223)
  cvtF32 (fReg 4) (uReg 25)
  mulF (fReg 4) (fReg 4) (fReg 47)
  absF (fReg 4) (fReg 4)
  cvtRziS32F32 (uReg 27) (fReg 4)
  cvtF32 (fReg 14) (uReg 27)
  subF (fReg 4) (fReg 4) (fReg 14)
  addFI (fReg 4) (fReg 4) (-0.5 : Float)
  madLoRII (uReg 25) (uReg 25) (1664525) (1013904223)
  cvtF32 (fReg 5) (uReg 25)
  mulF (fReg 5) (fReg 5) (fReg 47)
  absF (fReg 5) (fReg 5)
  cvtRziS32F32 (uReg 27) (fReg 5)
  cvtF32 (fReg 14) (uReg 27)
  subF (fReg 5) (fReg 5) (fReg 14)
  addFI (fReg 5) (fReg 5) (-0.5 : Float)
  -- Jittered pixel coords: integer center + 0.5 + jitter.
  addF (fReg 6) (fReg 0) (fReg 4)
  addFI (fReg 6) (fReg 6) (0.5 : Float)
  addF (fReg 7) (fReg 1) (fReg 5)
  addFI (fReg 7) (fReg 7) (0.5 : Float)
  -- ── Primary ray direction (NDC → world) ──────────────────
  divRn (fReg 2) (fReg 6) (fReg 42)
  divRn (fReg 3) (fReg 7) (fReg 43)
  fmaFII (fReg 2) (fReg 2) (2.0 : Float) (-1.0 : Float)
  fmaFII (fReg 3) (fReg 3) (-2.0 : Float) (1.0 : Float)
  mulF (fReg 2) (fReg 2) (fReg 41)
  mulF (fReg 2) (fReg 2) (fReg 40)
  mulF (fReg 3) (fReg 3) (fReg 40)
  -- rd = forward + u·right + v·up_cam
  -- right = (1, 0, 0): rd.x = u
  movF (fReg 60) (fReg 2)                       -- rd.x_pre = u
  addF (fReg 60) (fReg 60) (fReg 30)            -- + fx (= 0, but keep generic)
  -- rd.y_pre = fy + v · up_cam.y
  fmaRn (fReg 61) (fReg 3) (fReg 33) (fReg 31)
  -- rd.z_pre = fz + v · up_cam.z
  fmaRn (fReg 62) (fReg 3) (fReg 34) (fReg 32)
  -- Normalize.
  mulF (fReg 63) (fReg 60) (fReg 60)
  fmaRn (fReg 63) (fReg 61) (fReg 61) (fReg 63)
  fmaRn (fReg 63) (fReg 62) (fReg 62) (fReg 63)
  rsqrt (fReg 64) (fReg 63)
  mulF (BHReg.rdx) (fReg 60) (fReg 64)
  mulF (BHReg.rdy) (fReg 61) (fReg 64)
  mulF (BHReg.rdz) (fReg 62) (fReg 64)
  -- ── Plane-of-motion second basis u2 ──────────────────────
  -- u1 = (0, u1.y, u1.z)   (precomputed in fReg 35, 36)
  -- d_dot_u1 = rd · u1 = rd.y · u1.y + rd.z · u1.z
  mulF (fReg 70) (BHReg.rdy) (fReg 35)
  fmaRn (fReg 70) (BHReg.rdz) (fReg 36) (fReg 70)   -- fReg 70 = d·u1
  -- perp = rd − (d·u1) · u1
  movF (fReg 71) (BHReg.rdx)                    -- perp.x = rd.x − (d·u1)·0
  negF (fReg 72) (fReg 70)
  fmaRn (fReg 72) (fReg 72) (fReg 35) (BHReg.rdy)   -- perp.y = rd.y − (d·u1)·u1.y
  negF (fReg 73) (fReg 70)
  fmaRn (fReg 73) (fReg 73) (fReg 36) (BHReg.rdz)   -- perp.z = rd.z − (d·u1)·u1.z
  -- |perp|
  mulF (fReg 74) (fReg 71) (fReg 71)
  fmaRn (fReg 74) (fReg 72) (fReg 72) (fReg 74)
  fmaRn (fReg 74) (fReg 73) (fReg 73) (fReg 74)
  rsqrt (fReg 75) (fReg 74)
  sqrtApprox (fReg 76) (fReg 74)                -- |perp| (d·u2 = |perp|)
  mulF (BHReg.u2x) (fReg 71) (fReg 75)
  mulF (BHReg.u2y) (fReg 72) (fReg 75)
  mulF (BHReg.u2z) (fReg 73) (fReg 75)
  -- Initial ODE state:
  --   u₀  = 1/r₀
  --   du₀ = −u₀ · (d·u1) / (d·u2)
  movF (BHReg.u) (fReg 38)
  divRn (fReg 77) (fReg 70) (fReg 76)           -- (d·u1)/(d·u2)
  mulF (BHReg.du) (fReg 38) (fReg 77)
  negF (BHReg.du) (BHReg.du)
  -- φ trig accumulator: cos 0 = 1, sin 0 = 0.
  movFI (BHReg.cphi) (1.0 : Float)
  movFI (BHReg.sphi) (0.0 : Float)
  -- Previous y = camera.y.
  movF (BHReg.prevY) (fReg 39)
  -- Cached current r = r₀.
  movF (BHReg.rCur) (fReg 37)
  -- Default outcome = escape (2). Overwritten on horizon (0) / disk (1).
  movRC (BHReg.outcome) 2
  -- Step counter.
  movRC (BHReg.stepIdx) 0

/-- ============================================================
    Emit block 2 — geodesic integration loop.

    Symplectic-Euler stepping of
        u'' + u = 3M·u²       (Schwarzschild null geodesic in
                                the orbital plane, u = 1/r)
    with per-step tests for:
      * horizon       (r < rs)              → outcome = 0
      * accretion-disk crossing (sign(y) flips and r∈[r_in,r_out])
                                            → outcome = 1
      * escape        (r > r_max)            → outcome = 2

    cos/sin of φ are advanced by a rotation matrix:
        cos_new = cos·cosΔ − sin·sinΔ
        sin_new = sin·cosΔ + cos·sinΔ
    avoiding any per-step trig call.
    ============================================================ -/
private def emitBHIntegrationLoop (spec : BlackHoleSpec) : PTX Unit := do
  label "BH_LOOP"
  setpGeI (pReg 2) (BHReg.stepIdx) (stepCount spec)
  braIf (pReg 2) "BH_LOOP_END"
  -- Snapshot previous step's state so a disk-crossing detection can
  -- linearly interpolate r/φ/du to the actual y = 0 plane. Without
  -- this the hit point is quantized to one of the integrator's φ
  -- samples — visible as stair-stepped disk edges.
  movF (fReg 86) (BHReg.rCur)                   -- prev_r
  movF (fReg 87) (BHReg.cphi)                   -- prev_cos
  movF (fReg 88) (BHReg.sphi)                   -- prev_sin
  movF (fReg 89) (BHReg.du)                     -- prev_du
  -- Semi-implicit Euler step.
  mulF (fReg 100) (BHReg.u) (BHReg.u)           -- u²
  mulF (fReg 101) (fReg 100) (fReg 21)          -- 3M·u²
  subF (fReg 101) (fReg 101) (BHReg.u)          -- 3M·u² − u
  fmaRn (BHReg.du) (fReg 25) (fReg 101) (BHReg.du)
  fmaRn (BHReg.u) (fReg 25) (BHReg.du) (BHReg.u)
  -- Advance trig by dφ:
  --   cos_new = cos·cosΔ − sin·sinΔ
  --   sin_new = sin·cosΔ + cos·sinΔ
  mulF (fReg 104) (BHReg.sphi) (fReg 26)
  fmaRn (fReg 104) (BHReg.cphi) (fReg 27) (fReg 104)
  mulF (fReg 105) (BHReg.cphi) (fReg 26)
  mulF (fReg 106) (BHReg.sphi) (fReg 27)
  subF (BHReg.cphi) (fReg 105) (fReg 106)
  movF (BHReg.sphi) (fReg 104)
  -- r = 1/u
  movFI (fReg 107) (1.0 : Float)
  divRn (BHReg.rCur) (fReg 107) (BHReg.u)
  -- Horizon: r < rs ?
  setpLtF (pReg 3) (BHReg.rCur) (fReg 20)
  braIf (pReg 3) "BH_HORIZON"
  -- Escape: r > rMax ?
  setpGtF (pReg 4) (BHReg.rCur) (fReg 24)
  braIf (pReg 4) "BH_LOOP_END"
  -- Disk crossing: y(φ) = r·(cos·u1.y + sin·u2.y). Sign flip and the
  -- interpolated crossing radius must lie in [r_in, r_out].
  mulF (fReg 108) (BHReg.cphi) (fReg 35)
  fmaRn (fReg 108) (BHReg.sphi) (BHReg.u2y) (fReg 108)
  mulF (fReg 109) (BHReg.rCur) (fReg 108)       -- curr_y
  mulF (fReg 110) (BHReg.prevY) (fReg 109)
  setpLtFI (pReg 5) (fReg 110) (0.0 : Float)
  braIfNot (pReg 5) "BH_NO_DISK"
  -- Compute interpolation parameter t = prev_y / (prev_y − curr_y).
  subF (fReg 111) (BHReg.prevY) (fReg 109)
  divRn (fReg 90) (BHReg.prevY) (fReg 111)
  -- Interpolated r and bound check (skip if hit lies outside disk).
  subF (fReg 91) (BHReg.rCur) (fReg 86)
  fmaRn (fReg 91) (fReg 90) (fReg 91) (fReg 86)   -- r_hit
  setpLtF (pReg 6) (fReg 91) (fReg 22)
  braIf (pReg 6) "BH_NO_DISK"
  setpGtF (pReg 7) (fReg 91) (fReg 23)
  braIf (pReg 7) "BH_NO_DISK"
  -- Overwrite state with the lerped (prev + t·(curr − prev)) values.
  -- Color computation below uses BHReg.cphi/sphi/rCur/du directly.
  movF (BHReg.rCur) (fReg 91)
  subF (fReg 91) (BHReg.cphi) (fReg 87)
  fmaRn (BHReg.cphi) (fReg 90) (fReg 91) (fReg 87)
  subF (fReg 91) (BHReg.sphi) (fReg 88)
  fmaRn (BHReg.sphi) (fReg 90) (fReg 91) (fReg 88)
  subF (fReg 91) (BHReg.du) (fReg 89)
  fmaRn (BHReg.du) (fReg 90) (fReg 91) (fReg 89)
  movRC (BHReg.outcome) 1
  bra "BH_LOOP_END"
  label "BH_NO_DISK"
  movF (BHReg.prevY) (fReg 109)
  addRI (BHReg.stepIdx) (BHReg.stepIdx) (1)
  bra "BH_LOOP"
  label "BH_HORIZON"
  movRC (BHReg.outcome) 0
  label "BH_LOOP_END"

/-- ============================================================
    Emit block 3 — outcome → color.

    horizon  → pitch-black (sealing the event horizon).
    disk     → temperature-graded ring with Doppler beaming +
               gravitational redshift. Standard formula for a
               thermal (∝T⁴) source observed through a relativistic
               flow:

                   I_obs / I_emit = D⁴ · g_grav⁴

               where D = √(1−v²)/(1−β) is the Doppler factor and
               g_grav = √(1−rs/r) is the gravitational redshift.

    escape   → procedural starfield + faint nebula tint based on
               the photon's asymptotic direction.
    ============================================================ -/
private def emitBHColorAndWrite (spec : BlackHoleSpec) : PTX Unit := do
  setpEqI (pReg 8) (BHReg.outcome) 0
  braIf (pReg 8) "BH_C_HORIZON"
  setpEqI (pReg 9) (BHReg.outcome) 1
  braIf (pReg 9) "BH_C_DISK"
  -- ── ESCAPE → starfield ───────────────────────────────────
  -- Asymptotic direction T = drdφ · (cos·u1 + sin·u2) + r · (−sin·u1 + cos·u2)
  -- with drdφ = −du · r²  (since dr/dφ = −(1/u²)·du and r = 1/u).
  mulF (fReg 120) (BHReg.rCur) (BHReg.rCur)
  mulF (fReg 120) (fReg 120) (BHReg.du)
  negF (fReg 120) (fReg 120)                     -- fReg 120 = drdφ
  -- (cos · u1.x + sin · u2.x); u1.x = 0.
  mulF (fReg 121) (BHReg.sphi) (BHReg.u2x)       -- (cos·u1.x + sin·u2.x)
  -- (cos · u1.y + sin · u2.y)
  mulF (fReg 122) (BHReg.cphi) (fReg 35)
  fmaRn (fReg 122) (BHReg.sphi) (BHReg.u2y) (fReg 122)
  -- (cos · u1.z + sin · u2.z)
  mulF (fReg 123) (BHReg.cphi) (fReg 36)
  fmaRn (fReg 123) (BHReg.sphi) (BHReg.u2z) (fReg 123)
  -- (−sin · u1.x + cos · u2.x); u1.x = 0
  mulF (fReg 124) (BHReg.cphi) (BHReg.u2x)
  -- (−sin · u1.y + cos · u2.y)
  mulF (fReg 125) (BHReg.sphi) (fReg 35)
  negF (fReg 125) (fReg 125)
  fmaRn (fReg 125) (BHReg.cphi) (BHReg.u2y) (fReg 125)
  -- (−sin · u1.z + cos · u2.z)
  mulF (fReg 126) (BHReg.sphi) (fReg 36)
  negF (fReg 126) (fReg 126)
  fmaRn (fReg 126) (BHReg.cphi) (BHReg.u2z) (fReg 126)
  -- T = drdφ · (cos·u1 + sin·u2) + r · (−sin·u1 + cos·u2)
  mulF (fReg 127) (fReg 120) (fReg 121)
  fmaRn (fReg 127) (BHReg.rCur) (fReg 124) (fReg 127)
  mulF (fReg 128) (fReg 120) (fReg 122)
  fmaRn (fReg 128) (BHReg.rCur) (fReg 125) (fReg 128)
  mulF (fReg 129) (fReg 120) (fReg 123)
  fmaRn (fReg 129) (BHReg.rCur) (fReg 126) (fReg 129)
  -- Normalize.
  mulF (fReg 130) (fReg 127) (fReg 127)
  fmaRn (fReg 130) (fReg 128) (fReg 128) (fReg 130)
  fmaRn (fReg 130) (fReg 129) (fReg 129) (fReg 130)
  rsqrt (fReg 131) (fReg 130)
  mulF (fReg 132) (fReg 127) (fReg 131)         -- dx_out
  mulF (fReg 133) (fReg 128) (fReg 131)         -- dy_out
  mulF (fReg 134) (fReg 129) (fReg 131)         -- dz_out
  -- Deep-space background floor: near-pure black.
  movFI (BHReg.cR) (0.0008 : Float)
  movFI (BHReg.cG) (0.0006 : Float)
  movFI (BHReg.cB) (0.0014 : Float)
  -- ── Milky Way galactic-plane glow ───────────────────────
  -- Add a soft diffuse band along a fixed "galactic plane".
  -- For each escape direction d, distance to plane = |d · n̂_g|
  -- where n̂_g is the galactic-plane normal.  Smaller distance →
  -- brighter, with warm cream tint (the unresolved stellar
  -- continuum of a real Milky Way).  This is what makes the void
  -- between stars feel like a real night sky rather than just
  -- empty CG.
  -- n̂_g chosen to tilt across the frame ≈ 22° from horizontal.
  --   n̂_g = (0.0, 0.927, -0.375)   (normalized)
  mulFI (fReg 187) (fReg 133) (0.927 : Float)
  fmaFIR (fReg 187) (fReg 134) (-0.375 : Float) (fReg 187)
  absF (fReg 187) (fReg 187)                    -- |d · n̂_g|, the perpendicular distance
  -- Soft Gaussian-ish band: glow = 0.06 * exp(-(distance / 0.32)² · log2e)
  mulFI (fReg 188) (fReg 187) (3.125 : Float)
  mulF (fReg 188) (fReg 188) (fReg 188)
  mulFI (fReg 188) (fReg 188) (-1.4426950408889634 : Float)
  ex2 (fReg 188) (fReg 188)
  -- Very faint warm tint along the galactic plane.  On a wide-field
  -- astro photo, the MW band reads as a barely-perceptible gradient
  -- of warm grey — not a bright glow.  These amplitudes (peak ~0.05
  -- HDR) end up well below the bloom threshold and read as the dim
  -- diffuse continuum of unresolved background stars.
  fmaFIR (BHReg.cR) (fReg 188) (0.045 : Float) (BHReg.cR)
  fmaFIR (BHReg.cG) (fReg 188) (0.036 : Float) (BHReg.cG)
  fmaFIR (BHReg.cB) (fReg 188) (0.024 : Float) (BHReg.cB)
  -- Starfield direction quantization.  Multiplier 700 ≈ 1 sub-pixel
  -- per cell at our FOV, so individual stars resolve to point-like
  -- ~1px features once bloom is applied (rather than the 4–5 px blocky
  -- chunks the coarser quantization produced).
  fmaFII (fReg 138) (fReg 132) (700.0 : Float) (700.0 : Float)
  fmaFII (fReg 139) (fReg 133) (700.0 : Float) (700.0 : Float)
  fmaFII (fReg 140) (fReg 134) (700.0 : Float) (700.0 : Float)
  cvtRziS32F32 (uReg 40) (fReg 138)
  cvtRziS32F32 (uReg 41) (fReg 139)
  cvtRziS32F32 (uReg 42) (fReg 140)
  madLoRII (uReg 43) (uReg 40) (374761393) (0)
  madLoRII (uReg 44) (uReg 41) (668265263) (0)
  xorRR (uReg 43) (uReg 43) (uReg 44)
  madLoRII (uReg 44) (uReg 42) (2147483647) (0)
  xorRR (uReg 43) (uReg 43) (uReg 44)
  madLoRII (uReg 43) (uReg 43) (1274126177) (0)
  -- Threshold: take low 16 bits, if < 80 → star.
  xorRI (uReg 45) (uReg 43) (0)
  -- bitmask 0xFFFF via mad: keep low bits via shift would be ideal; use simple
  -- modulo 65536 via (h & 0xFFFF) — emulate with subtract loop is overkill.
  -- Instead: compute (h % 1009) which gives a small integer, then threshold.
  -- Use float-modulo trick: cast → frac → scale. Cheap enough.
  cvtF32 (fReg 141) (uReg 43)
  mulF (fReg 141) (fReg 141) (fReg 47)  -- divide by 2^32
  absF (fReg 141) (fReg 141)
  -- fract via subtracting floor: just take fractional via subtracting cvt-back.
  cvtRziS32F32 (uReg 46) (fReg 141)
  cvtF32 (fReg 142) (uReg 46)
  subF (fReg 141) (fReg 141) (fReg 142)
  -- ~10% of escape rays hit a star cell.  Combined with the long-tail
  -- intensity distribution below, ~1% land on a "bright" cell that
  -- survives bloom thresholding to become a visible point of light.
  -- ~0.2% of cells = stars.  Combined with 700-fine quantization
  -- that gives ~one star per few thousand pixels, matching the visual
  -- density of a wide-field astrophotography frame.
  -- Star density tuned to match a wide-field astronomical exposure:
  -- enough to read as "space" but not "globular cluster".  Real dark
  -- sky has roughly 1 mag-6 star per 10 arcmin² → at our FOV ~50–80
  -- visible bright stars per frame.
  setpGtFI (pReg 10) (fReg 141) (0.9994 : Float)
  braIfNot (pReg 10) "BH_C_DONE"
  -- Star intensity by another random read.
  madLoRII (uReg 43) (uReg 43) (1664525) (1013904223)
  cvtF32 (fReg 143) (uReg 43)
  mulF (fReg 143) (fReg 143) (fReg 47)
  absF (fReg 143) (fReg 143)
  cvtRziS32F32 (uReg 46) (fReg 143)
  cvtF32 (fReg 144) (uReg 46)
  subF (fReg 143) (fReg 143) (fReg 144)
  -- Star intensity ∝ fract^4 → mostly dim with a few very bright outliers.
  mulF (fReg 145) (fReg 143) (fReg 143)
  mulF (fReg 145) (fReg 145) (fReg 145)
  -- Star intensity distribution.  Per-sample contribution must be
  -- huge to survive 384× averaging: the fract^4 distribution gives
  -- many dim stars (~30 HDR) and a long tail of very-bright outliers
  -- (~2000 HDR) so the brightest stars produce visible bloom halos
  -- after dilution.
  -- Star intensity: very long tail.  With sparse hits (most pixels
  -- have 0–2 samples landing in a star cell out of 384), only a high
  -- per-sample contribution makes individual stars visible.  fract⁴
  -- means most are dim (~80 HDR) but the brightest ~5% reach ~8000.
  -- Star intensity: fewer stars → can push each one slightly brighter
  -- without flooding the frame.  fract⁴ keeps the bulk dim with a
  -- long tail of standout bright outliers (just like real night-sky
  -- magnitude distribution).
  mulFI (fReg 145) (fReg 145) (3200.0 : Float)
  addFI (fReg 145) (fReg 145) (40.0 : Float)
  -- Stars get a slight color shift (blue-white to amber) sampled from the hash.
  madLoRII (uReg 43) (uReg 43) (22695477) (1)
  cvtF32 (fReg 146) (uReg 43)
  mulF (fReg 146) (fReg 146) (fReg 47)
  absF (fReg 146) (fReg 146)
  cvtRziS32F32 (uReg 46) (fReg 146)
  cvtF32 (fReg 147) (uReg 46)
  subF (fReg 146) (fReg 146) (fReg 147)
  -- fReg 146 ∈ [0,1) → tint balance
  fmaFII (fReg 148) (fReg 146) (0.3 : Float) (0.9 : Float)        -- R = 0.9..1.2
  fmaFII (fReg 149) (fReg 146) (-0.1 : Float) (1.05 : Float)      -- G = 0.95..1.05
  fmaFII (fReg 144) (fReg 146) (-0.4 : Float) (1.25 : Float)      -- B = 0.85..1.25
  fmaRn (BHReg.cR) (fReg 145) (fReg 148) (BHReg.cR)
  fmaRn (BHReg.cG) (fReg 145) (fReg 149) (BHReg.cG)
  fmaRn (BHReg.cB) (fReg 145) (fReg 144) (BHReg.cB)
  bra "BH_C_DONE"
  -- Event horizon → pitch black (no photons escape).
  label "BH_C_HORIZON"
  movFI (BHReg.cR) (0.0 : Float)
  movFI (BHReg.cG) (0.0 : Float)
  movFI (BHReg.cB) (0.0 : Float)
  bra "BH_C_DONE"
  -- ── DISK HIT → temperature × Doppler × redshift ──────────
  label "BH_C_DISK"
  -- 3D hit point P = r·(cos·u1 + sin·u2). u1.x = 0.
  mulF (fReg 150) (BHReg.sphi) (BHReg.u2x)                  -- (cos·u1.x + sin·u2.x)
  mulF (fReg 150) (BHReg.rCur) (fReg 150)                   -- Px
  mulF (fReg 151) (BHReg.cphi) (fReg 35)
  fmaRn (fReg 151) (BHReg.sphi) (BHReg.u2y) (fReg 151)
  mulF (fReg 151) (BHReg.rCur) (fReg 151)                   -- Py (≈ 0 by construction)
  mulF (fReg 152) (BHReg.cphi) (fReg 36)
  fmaRn (fReg 152) (BHReg.sphi) (BHReg.u2z) (fReg 152)
  mulF (fReg 152) (BHReg.rCur) (fReg 152)                   -- Pz
  -- ── Turbulent noise for the disk surface ──────────────────
  -- Four octaves of smooth (bilinear) value noise on (Px, Pz),
  -- accumulated as fBm.  This is what kills the "concentric bands"
  -- look — by perturbing the radial temperature lookup with noise
  -- instead of evaluating it on the raw r, the temperature gradient
  -- becomes streaky and clumpy like real turbulent accretion-disk gas.
  --
  -- Each octave: hash the 4 corners of the cell containing (sx, sy),
  -- compute smoothstep-weighted bilinear interp.  Result ∈ [0, 1].
  --
  -- Inline 4 times.  Scales and seeds chosen so octaves decorrelate
  -- visually.  Output:
  --   fReg 230 = fBm value in [0, 1]
  --
  -- Helper layout per octave (all temps overwritten between octaves):
  --   fReg 232 = sx, fReg 233 = sy          (scaled coords)
  --   uReg 70 = ix,  uReg 71 = iy           (integer floor)
  --   fReg 234 = fx, fReg 235 = fy          (fractional part)
  --   fReg 236 = sfx, fReg 237 = sfy        (smoothstep weights)
  --   uReg 72..75 = ix*s0, iy*s1, (ix+1)*s0, (iy+1)*s1
  --   fReg 240..243 = corner values h00, h10, h01, h11
  --   fReg 244 = mx0, fReg 245 = mx1, fReg 246 = octave value
  --   fReg 230 = running fBm sum
  movFI (fReg 230) (0.0 : Float)
  -- ── Octave macros via repetition ────────────────────────────
  -- (Octave i): freq_i, amp_i, seed triplet, x-stretch, y-stretch
  --   1: 0.55,  0.50,  (374761393, 668265263, 1274126177), (1.0, 2.8)
  --   2: 1.30,  0.28,  (1597334677, 3812015801, 2654435769), (1.0, 1.0)
  --   3: 3.10,  0.14,  (2654435761, 40503,     2246822519), (1.0, 1.0)
  --   4: 7.40,  0.08,  (1865811581, 506832829, 3033965713), (1.0, 1.0)
  -- ─── Octave 1 ───
  fmaFII (fReg 232) (fReg 150) (0.55 : Float) (4096.0 : Float)
  fmaFII (fReg 233) (fReg 152) (1.54 : Float) (4096.0 : Float)
  cvtRziS32F32 (uReg 70) (fReg 232)
  cvtRziS32F32 (uReg 71) (fReg 233)
  cvtF32 (fReg 234) (uReg 70)
  cvtF32 (fReg 235) (uReg 71)
  subF (fReg 234) (fReg 232) (fReg 234)
  subF (fReg 235) (fReg 233) (fReg 235)
  -- smoothstep: x*x*(3 - 2x)
  fmaFII (fReg 236) (fReg 234) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 236) (fReg 234) (fReg 236)
  mulF (fReg 236) (fReg 234) (fReg 236)
  fmaFII (fReg 237) (fReg 235) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 237) (fReg 235) (fReg 237)
  mulF (fReg 237) (fReg 235) (fReg 237)
  -- Hash 4 corners: h(a,b) = ((a*s0) xor (b*s1)) * s2, fract via low bits
  madLoRII (uReg 72) (uReg 70) (374761393) (0)
  madLoRII (uReg 73) (uReg 71) (668265263) (0)
  addRI (uReg 74) (uReg 72) (374761393)
  addRI (uReg 75) (uReg 73) (668265263)
  xorRR (uReg 76) (uReg 72) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (1274126177) (0)
  cvtF32 (fReg 240) (uReg 76)
  mulF (fReg 240) (fReg 240) (fReg 47)
  absF (fReg 240) (fReg 240)
  cvtRziS32F32 (uReg 76) (fReg 240)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 240) (fReg 240) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (1274126177) (0)
  cvtF32 (fReg 241) (uReg 76)
  mulF (fReg 241) (fReg 241) (fReg 47)
  absF (fReg 241) (fReg 241)
  cvtRziS32F32 (uReg 76) (fReg 241)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 241) (fReg 241) (fReg 244)
  xorRR (uReg 76) (uReg 72) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (1274126177) (0)
  cvtF32 (fReg 242) (uReg 76)
  mulF (fReg 242) (fReg 242) (fReg 47)
  absF (fReg 242) (fReg 242)
  cvtRziS32F32 (uReg 76) (fReg 242)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 242) (fReg 242) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (1274126177) (0)
  cvtF32 (fReg 243) (uReg 76)
  mulF (fReg 243) (fReg 243) (fReg 47)
  absF (fReg 243) (fReg 243)
  cvtRziS32F32 (uReg 76) (fReg 243)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 243) (fReg 243) (fReg 244)
  -- Bilinear blend
  subF (fReg 244) (fReg 241) (fReg 240)
  fmaRn (fReg 244) (fReg 236) (fReg 244) (fReg 240)
  subF (fReg 245) (fReg 243) (fReg 242)
  fmaRn (fReg 245) (fReg 236) (fReg 245) (fReg 242)
  subF (fReg 246) (fReg 245) (fReg 244)
  fmaRn (fReg 246) (fReg 237) (fReg 246) (fReg 244)
  fmaFIR (fReg 230) (fReg 246) (0.78 : Float) (fReg 230)
  -- ─── Octave 2 ───
  fmaFII (fReg 232) (fReg 150) (1.30 : Float) (4096.0 : Float)
  fmaFII (fReg 233) (fReg 152) (1.30 : Float) (4096.0 : Float)
  cvtRziS32F32 (uReg 70) (fReg 232)
  cvtRziS32F32 (uReg 71) (fReg 233)
  cvtF32 (fReg 234) (uReg 70)
  cvtF32 (fReg 235) (uReg 71)
  subF (fReg 234) (fReg 232) (fReg 234)
  subF (fReg 235) (fReg 233) (fReg 235)
  fmaFII (fReg 236) (fReg 234) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 236) (fReg 234) (fReg 236)
  mulF (fReg 236) (fReg 234) (fReg 236)
  fmaFII (fReg 237) (fReg 235) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 237) (fReg 235) (fReg 237)
  mulF (fReg 237) (fReg 235) (fReg 237)
  madLoRII (uReg 72) (uReg 70) (1597334677) (0)
  madLoRII (uReg 73) (uReg 71) (3812015801) (0)
  addRI (uReg 74) (uReg 72) (1597334677)
  addRI (uReg 75) (uReg 73) (3812015801)
  xorRR (uReg 76) (uReg 72) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (2654435769) (0)
  cvtF32 (fReg 240) (uReg 76)
  mulF (fReg 240) (fReg 240) (fReg 47)
  absF (fReg 240) (fReg 240)
  cvtRziS32F32 (uReg 76) (fReg 240)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 240) (fReg 240) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (2654435769) (0)
  cvtF32 (fReg 241) (uReg 76)
  mulF (fReg 241) (fReg 241) (fReg 47)
  absF (fReg 241) (fReg 241)
  cvtRziS32F32 (uReg 76) (fReg 241)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 241) (fReg 241) (fReg 244)
  xorRR (uReg 76) (uReg 72) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (2654435769) (0)
  cvtF32 (fReg 242) (uReg 76)
  mulF (fReg 242) (fReg 242) (fReg 47)
  absF (fReg 242) (fReg 242)
  cvtRziS32F32 (uReg 76) (fReg 242)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 242) (fReg 242) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (2654435769) (0)
  cvtF32 (fReg 243) (uReg 76)
  mulF (fReg 243) (fReg 243) (fReg 47)
  absF (fReg 243) (fReg 243)
  cvtRziS32F32 (uReg 76) (fReg 243)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 243) (fReg 243) (fReg 244)
  subF (fReg 244) (fReg 241) (fReg 240)
  fmaRn (fReg 244) (fReg 236) (fReg 244) (fReg 240)
  subF (fReg 245) (fReg 243) (fReg 242)
  fmaRn (fReg 245) (fReg 236) (fReg 245) (fReg 242)
  subF (fReg 246) (fReg 245) (fReg 244)
  fmaRn (fReg 246) (fReg 237) (fReg 246) (fReg 244)
  fmaFIR (fReg 230) (fReg 246) (0.15 : Float) (fReg 230)
  -- ─── Octave 3 ───
  fmaFII (fReg 232) (fReg 150) (3.10 : Float) (4096.0 : Float)
  fmaFII (fReg 233) (fReg 152) (3.10 : Float) (4096.0 : Float)
  cvtRziS32F32 (uReg 70) (fReg 232)
  cvtRziS32F32 (uReg 71) (fReg 233)
  cvtF32 (fReg 234) (uReg 70)
  cvtF32 (fReg 235) (uReg 71)
  subF (fReg 234) (fReg 232) (fReg 234)
  subF (fReg 235) (fReg 233) (fReg 235)
  fmaFII (fReg 236) (fReg 234) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 236) (fReg 234) (fReg 236)
  mulF (fReg 236) (fReg 234) (fReg 236)
  fmaFII (fReg 237) (fReg 235) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 237) (fReg 235) (fReg 237)
  mulF (fReg 237) (fReg 235) (fReg 237)
  madLoRII (uReg 72) (uReg 70) (2654435761) (0)
  madLoRII (uReg 73) (uReg 71) (40503) (0)
  addRI (uReg 74) (uReg 72) (2654435761)
  addRI (uReg 75) (uReg 73) (40503)
  xorRR (uReg 76) (uReg 72) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (2246822519) (0)
  cvtF32 (fReg 240) (uReg 76)
  mulF (fReg 240) (fReg 240) (fReg 47)
  absF (fReg 240) (fReg 240)
  cvtRziS32F32 (uReg 76) (fReg 240)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 240) (fReg 240) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (2246822519) (0)
  cvtF32 (fReg 241) (uReg 76)
  mulF (fReg 241) (fReg 241) (fReg 47)
  absF (fReg 241) (fReg 241)
  cvtRziS32F32 (uReg 76) (fReg 241)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 241) (fReg 241) (fReg 244)
  xorRR (uReg 76) (uReg 72) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (2246822519) (0)
  cvtF32 (fReg 242) (uReg 76)
  mulF (fReg 242) (fReg 242) (fReg 47)
  absF (fReg 242) (fReg 242)
  cvtRziS32F32 (uReg 76) (fReg 242)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 242) (fReg 242) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (2246822519) (0)
  cvtF32 (fReg 243) (uReg 76)
  mulF (fReg 243) (fReg 243) (fReg 47)
  absF (fReg 243) (fReg 243)
  cvtRziS32F32 (uReg 76) (fReg 243)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 243) (fReg 243) (fReg 244)
  subF (fReg 244) (fReg 241) (fReg 240)
  fmaRn (fReg 244) (fReg 236) (fReg 244) (fReg 240)
  subF (fReg 245) (fReg 243) (fReg 242)
  fmaRn (fReg 245) (fReg 236) (fReg 245) (fReg 242)
  subF (fReg 246) (fReg 245) (fReg 244)
  fmaRn (fReg 246) (fReg 237) (fReg 246) (fReg 244)
  fmaFIR (fReg 230) (fReg 246) (0.05 : Float) (fReg 230)
  -- ─── Octave 4 ───
  fmaFII (fReg 232) (fReg 150) (7.40 : Float) (4096.0 : Float)
  fmaFII (fReg 233) (fReg 152) (7.40 : Float) (4096.0 : Float)
  cvtRziS32F32 (uReg 70) (fReg 232)
  cvtRziS32F32 (uReg 71) (fReg 233)
  cvtF32 (fReg 234) (uReg 70)
  cvtF32 (fReg 235) (uReg 71)
  subF (fReg 234) (fReg 232) (fReg 234)
  subF (fReg 235) (fReg 233) (fReg 235)
  fmaFII (fReg 236) (fReg 234) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 236) (fReg 234) (fReg 236)
  mulF (fReg 236) (fReg 234) (fReg 236)
  fmaFII (fReg 237) (fReg 235) (-2.0 : Float) (3.0 : Float)
  mulF (fReg 237) (fReg 235) (fReg 237)
  mulF (fReg 237) (fReg 235) (fReg 237)
  madLoRII (uReg 72) (uReg 70) (1865811581) (0)
  madLoRII (uReg 73) (uReg 71) (506832829) (0)
  addRI (uReg 74) (uReg 72) (1865811581)
  addRI (uReg 75) (uReg 73) (506832829)
  xorRR (uReg 76) (uReg 72) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (3033965713) (0)
  cvtF32 (fReg 240) (uReg 76)
  mulF (fReg 240) (fReg 240) (fReg 47)
  absF (fReg 240) (fReg 240)
  cvtRziS32F32 (uReg 76) (fReg 240)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 240) (fReg 240) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 73)
  madLoRII (uReg 76) (uReg 76) (3033965713) (0)
  cvtF32 (fReg 241) (uReg 76)
  mulF (fReg 241) (fReg 241) (fReg 47)
  absF (fReg 241) (fReg 241)
  cvtRziS32F32 (uReg 76) (fReg 241)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 241) (fReg 241) (fReg 244)
  xorRR (uReg 76) (uReg 72) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (3033965713) (0)
  cvtF32 (fReg 242) (uReg 76)
  mulF (fReg 242) (fReg 242) (fReg 47)
  absF (fReg 242) (fReg 242)
  cvtRziS32F32 (uReg 76) (fReg 242)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 242) (fReg 242) (fReg 244)
  xorRR (uReg 76) (uReg 74) (uReg 75)
  madLoRII (uReg 76) (uReg 76) (3033965713) (0)
  cvtF32 (fReg 243) (uReg 76)
  mulF (fReg 243) (fReg 243) (fReg 47)
  absF (fReg 243) (fReg 243)
  cvtRziS32F32 (uReg 76) (fReg 243)
  cvtF32 (fReg 244) (uReg 76)
  subF (fReg 243) (fReg 243) (fReg 244)
  subF (fReg 244) (fReg 241) (fReg 240)
  fmaRn (fReg 244) (fReg 236) (fReg 244) (fReg 240)
  subF (fReg 245) (fReg 243) (fReg 242)
  fmaRn (fReg 245) (fReg 236) (fReg 245) (fReg 242)
  subF (fReg 246) (fReg 245) (fReg 244)
  fmaRn (fReg 246) (fReg 237) (fReg 246) (fReg 244)
  fmaFIR (fReg 230) (fReg 246) (0.02 : Float) (fReg 230)
  -- Normalize fBm (Σ amplitudes = 1.0).  fReg 230 ∈ [0, 1].
  -- Now perturb the radial temperature lookup with it.  This warps
  -- isothermal annuli into turbulent flow lines.
  -- Noise → [-1.2, 1.2].  Lower perturbation amplitude keeps the
  -- disk's lensed edges crisp (large amplitude was making the
  -- temperature gradient mush at the inner / outer boundaries).
  fmaFII (fReg 231) (fReg 230) (2.4 : Float) (-1.2 : Float)
  addF (fReg 153) (BHReg.rCur) (fReg 231)                     -- r_perturbed
  subF (fReg 153) (fReg 153) (fReg 22)
  subF (fReg 154) (fReg 23) (fReg 22)
  divRn (fReg 155) (fReg 153) (fReg 154)
  maxFI (fReg 155) (fReg 155) (0.0 : Float)
  minFI (fReg 155) (fReg 155) (1.0 : Float)
  -- Inner (white-hot) → outer (ember red).  Below bloom threshold
  -- at rest, so disk midtones don't glow — only the Doppler-beamed
  -- approaching limb (×5–9 in D⁴) crosses the threshold.
  --   baseR = lerp(1.20, 0.60, t)
  fmaFII (fReg 156) (fReg 155) (-0.60 : Float) (1.20 : Float)
  --   baseG = lerp(0.95, 0.22, t)
  fmaFII (fReg 157) (fReg 155) (-0.73 : Float) (0.95 : Float)
  --   baseB = lerp(0.55, 0.05, t)
  fmaFII (fReg 158) (fReg 155) (-0.50 : Float) (0.55 : Float)
  -- And modulate brightness multiplicatively in [0.10, 2.40] using
  -- the same fBm — wider range reveals the turbulent gas clumps
  -- against the temperature gradient even after bloom mixes things.
  fmaFII (fReg 247) (fReg 230) (2.30 : Float) (0.10 : Float)
  mulF (fReg 156) (fReg 156) (fReg 247)
  mulF (fReg 157) (fReg 157) (fReg 247)
  mulF (fReg 158) (fReg 158) (fReg 247)
  -- Geodesic tangent at hit (3D), normalized — used to take β = v · n̂.
  mulF (fReg 160) (BHReg.rCur) (BHReg.rCur)
  mulF (fReg 160) (fReg 160) (BHReg.du)
  negF (fReg 160) (fReg 160)                                -- drdφ
  -- a-vector = (cos·u1 + sin·u2);  b-vector = (−sin·u1 + cos·u2)
  -- T = drdφ · a + r · b
  mulF (fReg 161) (BHReg.sphi) (BHReg.u2x)
  mulF (fReg 162) (BHReg.cphi) (fReg 35)
  fmaRn (fReg 162) (BHReg.sphi) (BHReg.u2y) (fReg 162)
  mulF (fReg 163) (BHReg.cphi) (fReg 36)
  fmaRn (fReg 163) (BHReg.sphi) (BHReg.u2z) (fReg 163)
  mulF (fReg 164) (BHReg.cphi) (BHReg.u2x)
  mulF (fReg 165) (BHReg.sphi) (fReg 35)
  negF (fReg 165) (fReg 165)
  fmaRn (fReg 165) (BHReg.cphi) (BHReg.u2y) (fReg 165)
  mulF (fReg 166) (BHReg.sphi) (fReg 36)
  negF (fReg 166) (fReg 166)
  fmaRn (fReg 166) (BHReg.cphi) (BHReg.u2z) (fReg 166)
  mulF (fReg 167) (fReg 160) (fReg 161)
  fmaRn (fReg 167) (BHReg.rCur) (fReg 164) (fReg 167)
  mulF (fReg 168) (fReg 160) (fReg 162)
  fmaRn (fReg 168) (BHReg.rCur) (fReg 165) (fReg 168)
  mulF (fReg 169) (fReg 160) (fReg 163)
  fmaRn (fReg 169) (BHReg.rCur) (fReg 166) (fReg 169)
  mulF (fReg 170) (fReg 167) (fReg 167)
  fmaRn (fReg 170) (fReg 168) (fReg 168) (fReg 170)
  fmaRn (fReg 170) (fReg 169) (fReg 169) (fReg 170)
  rsqrt (fReg 171) (fReg 170)
  mulF (fReg 167) (fReg 167) (fReg 171)                     -- n̂.x
  mulF (fReg 168) (fReg 168) (fReg 171)                     -- n̂.y
  mulF (fReg 169) (fReg 169) (fReg 171)                     -- n̂.z
  -- Photon direction is opposite to the traced geodesic tangent.
  negF (fReg 167) (fReg 167)
  negF (fReg 168) (fReg 168)
  negF (fReg 169) (fReg 169)
  -- Orbital tangent at P = (−Pz, 0, Px) / |P_xz|.   |P_xz| ≈ r since Py≈0.
  negF (fReg 172) (fReg 152)                                -- −Pz
  divRn (fReg 172) (fReg 172) (BHReg.rCur)
  movFI (fReg 173) (0.0 : Float)
  divRn (fReg 174) (fReg 150) (BHReg.rCur)                  -- Px/r
  -- v_orbit = √(M/r) (Keplerian; first-order approximation).
  divRn (fReg 175) (fReg 28) (BHReg.rCur)
  sqrtApprox (fReg 175) (fReg 175)                          -- v
  -- β = v · (orbit_tangent · photon_direction)
  mulF (fReg 176) (fReg 172) (fReg 167)
  fmaRn (fReg 176) (fReg 173) (fReg 168) (fReg 176)
  fmaRn (fReg 176) (fReg 174) (fReg 169) (fReg 176)
  mulF (fReg 176) (fReg 175) (fReg 176)                     -- β
  -- Doppler factor D = √(1 − v²)/(1 − β)
  mulF (fReg 177) (fReg 175) (fReg 175)
  subFIR (fReg 177) (1.0 : Float) (fReg 177)
  maxFI (fReg 177) (fReg 177) (0.0001 : Float)
  sqrtApprox (fReg 177) (fReg 177)                          -- √(1−v²)
  subFIR (fReg 178) (1.0 : Float) (fReg 176)
  maxFI (fReg 178) (fReg 178) (0.05 : Float)
  divRn (fReg 179) (fReg 177) (fReg 178)                    -- D
  -- D⁴
  mulF (fReg 180) (fReg 179) (fReg 179)
  mulF (fReg 180) (fReg 180) (fReg 180)
  -- Gravitational redshift g = √(1 − rs/r)
  divRn (fReg 181) (fReg 20) (BHReg.rCur)
  subFIR (fReg 181) (1.0 : Float) (fReg 181)
  maxFI (fReg 181) (fReg 181) (0.0001 : Float)
  sqrtApprox (fReg 181) (fReg 181)
  -- g⁴
  mulF (fReg 182) (fReg 181) (fReg 181)
  mulF (fReg 182) (fReg 182) (fReg 182)
  -- Total boost = D⁴ · g⁴ (Doppler beaming × gravitational redshift,
  -- bolometric, for a thermal source).
  mulF (fReg 183) (fReg 180) (fReg 182)
  -- Limb brightening: an optically-thin emitting disk traversed by a
  -- ray contributes emission proportional to path length through the
  -- disk.  Steep crossing (large |n̂.y|) → short path → dim;
  -- grazing crossing (small |n̂.y|) → long path → bright.  This is
  -- the physical reason for the iconic bright thin "edges" of the
  -- lensed disk in real renders / the EHT M87 image.  Use the
  -- normalized geodesic-tangent y-component (already in fReg 168;
  -- earlier negated to point along photon direction so flip back).
  negF (fReg 184) (fReg 168)                    -- |n̂.y| via abs
  absF (fReg 184) (fReg 184)
  addFI (fReg 184) (fReg 184) (0.12 : Float)    -- regularize near grazing
  movFI (fReg 185) (0.45 : Float)
  divRn (fReg 184) (fReg 185) (fReg 184)        -- limb factor ∈ [≈0.3, 3.8]
  minFI (fReg 184) (fReg 184) (2.6 : Float)
  mulF (fReg 183) (fReg 183) (fReg 184)
  mulF (BHReg.cR) (fReg 156) (fReg 183)
  mulF (BHReg.cG) (fReg 157) (fReg 183)
  mulF (BHReg.cB) (fReg 158) (fReg 183)
  -- Doppler colour shift.  Relativistic Doppler doesn't just brighten
  -- — it also blue-shifts the spectrum of the approaching limb and
  -- red-shifts the receding one.  Approximate this by tinting RGB by
  -- a `shift = D · g` (combined Doppler + gravitational frequency
  -- ratio) along a hot-blue ↔ cool-red axis.
  --   shift > 1  →  multiply B by ~shift, R by ~1/shift
  --   shift < 1  →  vice versa
  -- fReg 179 = D (already computed), fReg 181 = g.
  mulF (fReg 185) (fReg 179) (fReg 181)            -- shift = D·g
  maxFI (fReg 185) (fReg 185) (0.25 : Float)
  minFI (fReg 185) (fReg 185) (4.0 : Float)
  -- Use √shift as the tint exponent (cheap; gives ~[0.5, 2.0] colour
  -- shift over D·g ∈ [0.25, 4]):
  --   B *= √shift   (blueshift on approach, dim on recede)
  --   R *= 1/√shift (redshift on recede, dim on approach)
  sqrtApprox (fReg 186) (fReg 185)                 -- √shift
  movFI (fReg 187) (1.0 : Float)
  divRn (fReg 188) (fReg 187) (fReg 186)           -- 1/√shift
  mulF (BHReg.cR) (BHReg.cR) (fReg 188)
  mulF (BHReg.cB) (BHReg.cB) (fReg 186)
  -- ── Per-sample tail: accumulate into pixel sum, next sample ──
  label "BH_C_DONE"
  addF (fReg 10) (fReg 10) (BHReg.cR)
  addF (fReg 11) (fReg 11) (BHReg.cG)
  addF (fReg 12) (fReg 12) (BHReg.cB)
  addRI (uReg 26) (uReg 26) (1)
  bra "SAMPLE_LOOP"
  -- ── All samples done — average, write linear HDR triple ─────
  -- The second kernel (composite_bloom) does threshold-gather,
  -- Gaussian bloom convolution, ACES tonemap, gamma, and BGRA pack
  -- using these HDR floats as input.
  label "SAMPLE_DONE"
  movFI (fReg 230) (1.0 / Float.ofNat (samplesPerPixel spec) : Float)
  mulF (fReg 10) (fReg 10) (fReg 230)
  mulF (fReg 11) (fReg 11) (fReg 230)
  mulF (fReg 12) (fReg 12) (fReg 230)
  -- HDR pixel = 16 bytes (RGB f32 + 4-byte pad for alignment).
  -- Recompute address: hdr_ptr + (y*W + x) * 16.
  ldParam64 (dReg 4) "hdr_ptr"
  madLoRC (uReg 6) (BHReg.pixelY) (imageWidth spec) (BHReg.pixelX)
  mulWideRI (dReg 5) (uReg 6) (16)
  addS64 (dReg 6) (dReg 4) (dReg 5)
  stGlobalF (dReg 6) (fReg 10)
  stGlobalFO (dReg 6) 4 (fReg 11)
  stGlobalFO (dReg 6) 8 (fReg 12)
  label "DONE"
  ptxRet

/-- ============================================================
    Emit block — bloom composite kernel.

    For each output pixel, scan a `(2R+1)²` window of HDR neighbors,
    take the "bright pass" (`max(c − T, 0)` per channel for a
    luminance threshold `T`), Gaussian-weight by distance, and
    accumulate into a soft glow.  Add the glow back to the original
    HDR, then run ACES tonemap and gamma.

    Cost is O(R²) per pixel — at R=24 that's ~2400 HDR reads per
    output pixel.  720p × 2400 = 2.2 G reads, comfortably under
    a second on any consumer GPU.

    The cooler look-and-feel of the result vs. the raw render is
    almost entirely the bloom (and the more aggressive HDR range
    upstream that bloom unlocks — without tonemap clamp, bright
    Doppler-beamed limb pixels carry values of ~5–30 instead of
    being clamped at 1.0, so the bright-pass actually has something
    to spread).
    ============================================================ -/
private def emitBloomComposite (spec : BlackHoleSpec) : PTX Unit := do
  declPredRegs 32
  declU32Regs 64
  declU64Regs 16
  declF32Regs 128
  -- Multi-scale bloom: one wide gather pass that builds three
  -- Gaussian sums at different σ simultaneously.  Combining a
  -- tight glow (σ≈8 px) for crisp halos with a huge soft veil
  -- (σ≈48 px) for the long-falloff "light spilling into the
  -- atmosphere" look — exactly the layered bloom modern engines
  -- use.  Done in one gather to keep memory bandwidth amortized.
  -- Bloom radii for realistic point-source response.  Real space
  -- imaging (telescope or modern cinema sensor) shows:
  --   σ=1.5 px → near-pinpoint stars with very faint halo
  --   σ=10 px → subtle warm glow around bright regions (atmospheric
  --             scatter in an Earth telescope; defocus blur otherwise)
  --   σ=32 px → long-falloff energy veil, mostly invisible except
  --             around the brightest disk-limb pixels
  let bloomRadius : Nat := 60
  let bloomDiameter : Nat := 2 * bloomRadius + 1
  let sigmaSmall : Float := 1.1
  let sigmaMedium : Float := 10.0
  let sigmaLarge : Float := 32.0
  -- exp(d²·k) with k = -log₂(e) / (2σ²), used via ex2.approx.
  let kSmall : Float := -1.4426950408889634 / (2.0 * sigmaSmall * sigmaSmall)
  let kMedium : Float := -1.4426950408889634 / (2.0 * sigmaMedium * sigmaMedium)
  let kLarge : Float := -1.4426950408889634 / (2.0 * sigmaLarge * sigmaLarge)
  -- Per-σ normalization (≈ 1 / 2π σ²).
  let normSmall : Float := 1.0 / (2.0 * 3.141592653589793 * sigmaSmall * sigmaSmall)
  let normMedium : Float := 1.0 / (2.0 * 3.141592653589793 * sigmaMedium * sigmaMedium)
  let normLarge : Float := 1.0 / (2.0 * 3.141592653589793 * sigmaLarge * sigmaLarge)
  -- Realistic bloom: small layer is the only meaningfully visible
  -- one — gives stars their tight halo and the bright disk limb a
  -- subtle warm rim.  Medium/large are barely perceptible (mimicking
  -- the very small amount of scatter inside a real camera optical
  -- path).  No anamorphic streaks — those are a cinema-lens artifact,
  -- not a property of telescope or space-photography imaging.
  let wSmall : Float := 0.48
  let wMedium : Float := 0.06
  let wLarge : Float := 0.012
  -- ex2(x * invLog2 * (-)) — we use ex2.approx for exp(d²·invTwoSigmaSq).
  -- Block / thread → pixel.
  movR (uReg 0) ctaX
  movR (uReg 1) ctaY
  movR (uReg 2) tidX
  movR (uReg 3) tidY
  madLoRC (uReg 4) (uReg 0) (16) (uReg 2)       -- px
  madLoRC (uReg 5) (uReg 1) (16) (uReg 3)       -- py
  setpGeI (pReg 0) (uReg 4) (imageWidth spec)
  braIf (pReg 0) "BC_DONE"
  setpGeI (pReg 1) (uReg 5) (imageHeight spec)
  braIf (pReg 1) "BC_DONE"
  -- HDR base pointer and own-pixel address.
  ldParam64 (dReg 0) "hdr_ptr"
  ldParam64 (dReg 1) "bgra_ptr"
  madLoRC (uReg 6) (uReg 5) (imageWidth spec) (uReg 4)
  mulWideRI (dReg 2) (uReg 6) (16)
  addS64 (dReg 3) (dReg 0) (dReg 2)             -- self HDR addr
  ldGlobalF (fReg 0) (dReg 3)                   -- own R
  ldGlobalFO (fReg 1) (dReg 3) 4                -- own G
  ldGlobalFO (fReg 2) (dReg 3) 8                -- own B
  -- Bloom accumulators (3 σ layers × 3 channels = 9 floats).
  movFI (fReg 3)  (0.0 : Float)   -- small R
  movFI (fReg 4)  (0.0 : Float)   -- small G
  movFI (fReg 5)  (0.0 : Float)   -- small B
  movFI (fReg 6)  (0.0 : Float)   -- medium R
  movFI (fReg 7)  (0.0 : Float)   -- medium G
  movFI (fReg 8)  (0.0 : Float)   -- medium B
  movFI (fReg 9)  (0.0 : Float)   -- large R
  movFI (fReg 60) (0.0 : Float)   -- large G
  movFI (fReg 61) (0.0 : Float)   -- large B
  -- Bright-pass threshold (linear HDR).  Only pixels above this glow
  -- — keeps the disk's midtones intact, mirrors real lens bloom which
  -- only affects highlight pixels.
  movFI (fReg 11) (4.5 : Float)
  -- fReg 47 = 2^-32 for hash→fract conversion (sensor grain).  Same
  -- workaround as the render kernel: exact IEEE bits because
  -- `Float.toString` would round this to 0.000000.
  movFI (fReg 47) (0x2F800000 : UInt32)
  -- Pre-loaded log2-space Gaussian coefficients per layer.
  movFI (fReg 70) (kSmall : Float)
  movFI (fReg 71) (kMedium : Float)
  movFI (fReg 72) (kLarge : Float)
  -- ny_base = py + (2^32 - R), nx_base = px + (2^32 - R)  — u32-wrap trick
  -- gives signed offsets within the bounds check below.
  let negR : Nat := (Nat.pow 2 32) - bloomRadius
  addRI (uReg 10) (uReg 5) negR                 -- ny_base
  addRI (uReg 11) (uReg 4) negR                 -- nx_base
  movRC (uReg 12) 0                             -- dy_idx
  label "BC_LOOP_Y"
  setpGeI (pReg 2) (uReg 12) bloomDiameter
  braIf (pReg 2) "BC_LOOP_Y_END"
  -- ny = ny_base + dy_idx
  addR (uReg 13) (uReg 10) (uReg 12)
  setpGeI (pReg 3) (uReg 13) (imageHeight spec)
  braIf (pReg 3) "BC_SKIP_Y"
  movRC (uReg 14) 0                             -- dx_idx
  -- dy as float (signed): dy = dy_idx - R  → as float, just cvt then subtract R.
  cvtF32 (fReg 20) (uReg 12)
  addFI (fReg 20) (fReg 20) (- (Float.ofNat bloomRadius) : Float)
  mulF (fReg 21) (fReg 20) (fReg 20)            -- dy²
  label "BC_LOOP_X"
  setpGeI (pReg 4) (uReg 14) bloomDiameter
  braIf (pReg 4) "BC_LOOP_X_END"
  addR (uReg 15) (uReg 11) (uReg 14)            -- nx
  setpGeI (pReg 5) (uReg 15) (imageWidth spec)
  braIf (pReg 5) "BC_SKIP_X"
  -- Squared distance d² = dx² + dy² (shared across all three layers).
  cvtF32 (fReg 22) (uReg 14)
  addFI (fReg 22) (fReg 22) (- (Float.ofNat bloomRadius) : Float)
  fmaRn (fReg 23) (fReg 22) (fReg 22) (fReg 21)
  -- Three Gaussian weights from one d², via ex2(d² · k).
  mulF (fReg 73) (fReg 23) (fReg 70)
  ex2 (fReg 73) (fReg 73)                        -- w_small
  mulF (fReg 74) (fReg 23) (fReg 71)
  ex2 (fReg 74) (fReg 74)                        -- w_medium
  mulF (fReg 75) (fReg 23) (fReg 72)
  ex2 (fReg 75) (fReg 75)                        -- w_large
  -- neighbor addr
  madLoRC (uReg 16) (uReg 13) (imageWidth spec) (uReg 15)
  mulWideRI (dReg 8) (uReg 16) (16)
  addS64 (dReg 9) (dReg 0) (dReg 8)
  ldGlobalF (fReg 24) (dReg 9)
  ldGlobalFO (fReg 25) (dReg 9) 4
  ldGlobalFO (fReg 26) (dReg 9) 8
  -- Bright-pass on max-channel luminance: a soft knee at threshold.
  subF (fReg 24) (fReg 24) (fReg 11)
  maxFI (fReg 24) (fReg 24) (0.0 : Float)
  subF (fReg 25) (fReg 25) (fReg 11)
  maxFI (fReg 25) (fReg 25) (0.0 : Float)
  subF (fReg 26) (fReg 26) (fReg 11)
  maxFI (fReg 26) (fReg 26) (0.0 : Float)
  -- Accumulate into all three layers.
  fmaRn (fReg 3)  (fReg 24) (fReg 73) (fReg 3)
  fmaRn (fReg 4)  (fReg 25) (fReg 73) (fReg 4)
  fmaRn (fReg 5)  (fReg 26) (fReg 73) (fReg 5)
  fmaRn (fReg 6)  (fReg 24) (fReg 74) (fReg 6)
  fmaRn (fReg 7)  (fReg 25) (fReg 74) (fReg 7)
  fmaRn (fReg 8)  (fReg 26) (fReg 74) (fReg 8)
  fmaRn (fReg 9)  (fReg 24) (fReg 75) (fReg 9)
  fmaRn (fReg 60) (fReg 25) (fReg 75) (fReg 60)
  fmaRn (fReg 61) (fReg 26) (fReg 75) (fReg 61)
  label "BC_SKIP_X"
  addRI (uReg 14) (uReg 14) (1)
  bra "BC_LOOP_X"
  label "BC_LOOP_X_END"
  label "BC_SKIP_Y"
  addRI (uReg 12) (uReg 12) (1)
  bra "BC_LOOP_Y"
  label "BC_LOOP_Y_END"
  -- Normalize each Gaussian layer and apply per-layer weight, then sum.
  let nS : Float := normSmall * wSmall
  let nM : Float := normMedium * wMedium
  let nL : Float := normLarge * wLarge
  movFI (fReg 30) (nS : Float)
  movFI (fReg 31) (nM : Float)
  movFI (fReg 32) (nL : Float)
  mulF (fReg 3)  (fReg 3)  (fReg 30)
  mulF (fReg 4)  (fReg 4)  (fReg 30)
  mulF (fReg 5)  (fReg 5)  (fReg 30)
  fmaRn (fReg 3) (fReg 6)  (fReg 31) (fReg 3)
  fmaRn (fReg 4) (fReg 7)  (fReg 31) (fReg 4)
  fmaRn (fReg 5) (fReg 8)  (fReg 31) (fReg 5)
  fmaRn (fReg 3) (fReg 9)  (fReg 32) (fReg 3)
  fmaRn (fReg 4) (fReg 60) (fReg 32) (fReg 4)
  fmaRn (fReg 5) (fReg 61) (fReg 32) (fReg 5)
  -- Composite: HDR + bloom
  addF (fReg 40) (fReg 0) (fReg 3)
  addF (fReg 41) (fReg 1) (fReg 4)
  addF (fReg 42) (fReg 2) (fReg 5)
  -- Exposure tuned so receding-limb Doppler dim crushes to black
  -- and approaching-limb peaks at near-white through ACES.  This
  -- is what makes the brightness asymmetry actually read as
  -- dramatic instead of subtle gradient.
  mulFI (fReg 40) (fReg 40) (0.24 : Float)
  mulFI (fReg 41) (fReg 41) (0.24 : Float)
  mulFI (fReg 42) (fReg 42) (0.24 : Float)
  -- ACES tonemap per channel.
  fmaFII (fReg 50) (fReg 40) (2.51 : Float) (0.03 : Float)
  mulF (fReg 50) (fReg 40) (fReg 50)
  fmaFII (fReg 51) (fReg 40) (2.43 : Float) (0.59 : Float)
  mulF (fReg 51) (fReg 40) (fReg 51)
  addFI (fReg 51) (fReg 51) (0.14 : Float)
  divRn (fReg 40) (fReg 50) (fReg 51)
  maxFI (fReg 40) (fReg 40) (0.0 : Float)
  minFI (fReg 40) (fReg 40) (1.0 : Float)
  fmaFII (fReg 50) (fReg 41) (2.51 : Float) (0.03 : Float)
  mulF (fReg 50) (fReg 41) (fReg 50)
  fmaFII (fReg 51) (fReg 41) (2.43 : Float) (0.59 : Float)
  mulF (fReg 51) (fReg 41) (fReg 51)
  addFI (fReg 51) (fReg 51) (0.14 : Float)
  divRn (fReg 41) (fReg 50) (fReg 51)
  maxFI (fReg 41) (fReg 41) (0.0 : Float)
  minFI (fReg 41) (fReg 41) (1.0 : Float)
  fmaFII (fReg 50) (fReg 42) (2.51 : Float) (0.03 : Float)
  mulF (fReg 50) (fReg 42) (fReg 50)
  fmaFII (fReg 51) (fReg 42) (2.43 : Float) (0.59 : Float)
  mulF (fReg 51) (fReg 42) (fReg 51)
  addFI (fReg 51) (fReg 51) (0.14 : Float)
  divRn (fReg 42) (fReg 50) (fReg 51)
  maxFI (fReg 42) (fReg 42) (0.0 : Float)
  minFI (fReg 42) (fReg 42) (1.0 : Float)
  -- ── Vignetting (real wide-FOV lens darkens toward corners) ──
  -- Compute normalised radial distance from frame centre and apply
  -- a soft falloff.  ~30% darken at the corners, no falloff at
  -- centre — matches a moderately fast cinema lens.
  cvtF32 (fReg 28) (uReg 4)
  cvtF32 (fReg 29) (uReg 5)
  fmaFII (fReg 28) (fReg 28) ((1.0 / (Float.ofNat (imageWidth spec) * 0.5)) : Float) (-1.0 : Float)
  fmaFII (fReg 29) (fReg 29) ((1.0 / (Float.ofNat (imageHeight spec) * 0.5)) : Float) (-1.0 : Float)
  mulF (fReg 28) (fReg 28) (fReg 28)
  fmaRn (fReg 28) (fReg 29) (fReg 29) (fReg 28)
  fmaFII (fReg 28) (fReg 28) (-0.22 : Float) (1.0 : Float)
  maxFI (fReg 28) (fReg 28) (0.65 : Float)
  mulF (fReg 40) (fReg 40) (fReg 28)
  mulF (fReg 41) (fReg 41) (fReg 28)
  mulF (fReg 42) (fReg 42) (fReg 28)
  -- Gamma ≈ 2.0 (sqrt).
  sqrtApprox (fReg 40) (fReg 40)
  sqrtApprox (fReg 41) (fReg 41)
  sqrtApprox (fReg 42) (fReg 42)
  -- ── Sensor grain (subtle additive Gaussian-ish noise) ───────
  -- A small per-pixel deterministic noise that mimics CMOS read
  -- noise.  Real long-exposure astro images and digital cinema
  -- frames always show this — the absence of it is what makes
  -- pure-CG images read as "too clean".  Amplitude ~1.5/255.
  madLoRII (uReg 50) (uReg 4) (374761393) (0)
  madLoRII (uReg 51) (uReg 5) (1597334677) (0)
  xorRR (uReg 50) (uReg 50) (uReg 51)
  madLoRII (uReg 50) (uReg 50) (1274126177) (0)
  cvtF32 (fReg 56) (uReg 50)
  mulF (fReg 56) (fReg 56) (fReg 47)
  absF (fReg 56) (fReg 56)
  cvtRziS32F32 (uReg 50) (fReg 56)
  cvtF32 (fReg 57) (uReg 50)
  subF (fReg 56) (fReg 56) (fReg 57)
  fmaFII (fReg 56) (fReg 56) (0.014 : Float) (-0.007 : Float)
  addF (fReg 40) (fReg 40) (fReg 56)
  addF (fReg 41) (fReg 41) (fReg 56)
  addF (fReg 42) (fReg 42) (fReg 56)
  maxFI (fReg 40) (fReg 40) (0.0 : Float)
  maxFI (fReg 41) (fReg 41) (0.0 : Float)
  maxFI (fReg 42) (fReg 42) (0.0 : Float)
  minFI (fReg 40) (fReg 40) (1.0 : Float)
  minFI (fReg 41) (fReg 41) (1.0 : Float)
  minFI (fReg 42) (fReg 42) (1.0 : Float)
  -- Pack BGRA and store.
  movFI (fReg 52) (255.0 : Float)
  mulF (fReg 53) (fReg 40) (fReg 52)
  mulF (fReg 54) (fReg 41) (fReg 52)
  mulF (fReg 55) (fReg 42) (fReg 52)
  cvtRniU32F32 (uReg 30) (fReg 53)
  cvtRniU32F32 (uReg 31) (fReg 54)
  cvtRniU32F32 (uReg 32) (fReg 55)
  shlR (uReg 31) (uReg 31) (8)
  shlR (uReg 30) (uReg 30) (16)
  movRC (uReg 33) 0xFF000000
  orRR (uReg 34) (uReg 32) (uReg 31)
  orRR (uReg 35) (uReg 34) (uReg 30)
  orRR (uReg 36) (uReg 35) (uReg 33)
  mulWideRI (dReg 10) (uReg 6) (4)
  addS64 (dReg 11) (dReg 1) (dReg 10)
  stGlobalU32 (dReg 11) (uReg 36)
  label "BC_DONE"
  ptxRet

def emitBHKernel (spec : BlackHoleSpec) : PTX Unit := do
  emitBHSetupAndRay spec
  emitBHIntegrationLoop spec
  emitBHColorAndWrite spec

def ptxSource (spec : BlackHoleSpec) : String :=
  buildModuleWith { version := "7.0", target := "sm_50" } [
    { name := "render_hdr"
      params := ["hdr_ptr"]
      body := do emitBHKernel spec },
    { name := "composite_bloom"
      params := ["hdr_ptr", "bgra_ptr"]
      body := do emitBloomComposite spec }
  ]

/-- ============================================================
    Memory layout — extended for the 2-kernel HDR → bloom →
    BGRA pipeline.  We now hold:

      * one PTX text region with two `.entry` functions
      * two kernel-name strings (passed to `cl_cuda_launch_named`)
      * two binding-descriptor regions (1-buffer for `render_hdr`,
        2-buffer for `composite_bloom`), written at runtime by
        the CLIF stub once `cudaCreateBuffer` returns IDs
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

open AlgorithmLib.IR in
def clifIrSource (spec : BlackHoleSpec) : String := buildProgram do
  let fnWrite ← declareFileWrite
  let cuda ← declareCudaFFI

  let ptr ← entryBlock
  cudaInit cuda ptr
  -- Allocate device buffers: HDR scratch (RGB f32, padded to 16 B/pixel)
  -- and final BGRA u32 output.
  let hdrSz ← iconst64 (hdrPixelBytes spec)
  let bgraSz ← iconst64 (pixelBytes spec)
  let hdrBuf ← cudaCreateBuffer cuda ptr hdrSz
  let bgraBuf ← cudaCreateBuffer cuda ptr bgraSz
  -- Write the two binding descriptors into host memory at the
  -- pre-reserved offsets.  Each is just a packed list of i32
  -- buffer IDs; the FFI side reads N=`nBufs` of them.
  storeI32 hdrBuf (← absAddr ptr bindOffA)
  storeI32 hdrBuf (← absAddr ptr bindOffB)
  storeI32 bgraBuf (← absAddr ptr (bindOffB + 4))
  -- Launch kernel A: HDR render.
  let ptxOffV ← iconst64 ptxOff
  let nameAV ← iconst64 nameOffRender
  let nBufs1 ← iconst32 1
  let bindAV ← iconst64 bindOffA
  let gridX ← iconst32 ((imageWidth spec + 15) / 16)
  let gridY ← iconst32 ((imageHeight spec + 15) / 16)
  let one32 ← iconst32 1
  let blk16 ← iconst32 16
  let _ ← cudaLaunchNamed cuda ptr ptxOffV nameAV nBufs1 bindAV gridX gridY one32 blk16 blk16 one32
  let _ ← cudaSync cuda ptr
  -- Launch kernel B: bloom composite (HDR + bloom → BGRA).
  let nameBV ← iconst64 nameOffComposite
  let nBufs2 ← iconst32 2
  let bindBV ← iconst64 bindOffB
  let _ ← cudaLaunchNamed cuda ptr ptxOffV nameBV nBufs2 bindBV gridX gridY one32 blk16 blk16 one32
  let _ ← cudaSync cuda ptr
  -- Download final BGRA pixels and write the BMP file.
  let pxOffV ← iconst64 pixelsOff
  let _ ← cudaDownload cuda ptr bgraBuf pxOffV bgraSz
  cudaCleanup cuda ptr
  let total ← iconst64 (54 + pixelBytes spec)
  let _ ← writeFile0 ptr fnWrite filenameOff bmpHeaderOff total
  ret

def payloads (spec : BlackHoleSpec) : List UInt8 :=
  let reserved := zeros ptxOff
  let ptxBytes := padTo (stringToBytes (ptxSource spec)) ptxRegion
  let nameA := padTo (stringToBytes "render_hdr") nameRegion
  let nameB := padTo (stringToBytes "composite_bloom") nameRegion
  let bindAPad := zeros bindRegionA
  let bindBPad := zeros bindRegionB
  let filenameBytes := padTo (stringToBytes spec.filename) filenameRegion
  let clifPad := zeros clifIrRegion
  reserved ++ ptxBytes ++ nameA ++ nameB ++ bindAPad ++ bindBPad ++
    filenameBytes ++ clifPad ++ bmpHeader spec

def config (spec : BlackHoleSpec) : BaseConfig := {
  cranelift_ir := clifIrSource spec,
  memory_size := (payloads spec).length + pixelBytes spec,
  context_offset := 0,
  initial_memory := payloads spec
}

def algorithm : Algorithm := {
  fn_idx := IR.mainFnIdx
}

def renderScene (spec : BlackHoleSpec) : BaseConfig × Algorithm :=
  (config spec, algorithm)

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

def previewBlackHole : BlackHoleSpec :=
  checkedBlackHole
    (width := 640) (height := 360) (stepCount := 400) (samplesPerPixel := 4)
    (schwarzschildRadius := 1.0)
    (diskInner := 3.0)
    (diskOuter := 10.0)
    (cameraHeight := 3.0)
    (cameraDistance := 12.0)
    (fovYDeg := 55.0)
    (dPhi := 0.012)
    (rMax := 80.0)
    (filename := "blackhole.bmp")
    (by decide) (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    (by native_decide) (by native_decide) (by native_decide)

def edgeOnBlackHole : BlackHoleSpec :=
  checkedBlackHole
    (width := 1280) (height := 720) (stepCount := 1000) (samplesPerPixel := 32)
    (schwarzschildRadius := 1.0)
    (diskInner := 3.0)
    (diskOuter := 14.0)
    (cameraHeight := 1.2)
    (cameraDistance := 18.0)
    (fovYDeg := 45.0)
    (dPhi := 0.005)
    (rMax := 120.0)
    (filename := "blackhole.bmp")
    (by decide) (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    (by native_decide) (by native_decide) (by native_decide)

end Algorithm

def main (args : List String) : IO Unit := do
  let (cfg, alg) := Algorithm.renderScene Algorithm.defaultBlackHole
  let jsonEntry := toJsonEntry "blackhole_app" cfg alg
  let outputDir ← requireOutputDir args
  emitArtifacts outputDir #[jsonEntry]
