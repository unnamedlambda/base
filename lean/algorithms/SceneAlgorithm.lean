import AlgorithmLib
set_option maxRecDepth 8192
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Algorithm

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

def f32Nat (n : Nat) : FImm :=
  AlgorithmLib.PTX.FImm.nat n

def f32Float (x : Float) : FImm :=
  AlgorithmLib.PTX.FImm.float x

def channelF32 (channel : Channel) : FImm :=
  f32Float (Float.ofNat channel.value / 255.0)

def colorF32 (color : Color) : FImm × FImm × FImm :=
  (channelF32 color.red, channelF32 color.green, channelF32 color.blue)

def invSamplesF32 (spec : SceneSpec) : FImm :=
  f32Float (1.0 / Float.ofNat (sampleCount spec))

def aspectF32 (spec : SceneSpec) : FImm :=
  f32Float (Float.ofNat (imageWidth spec) / Float.ofNat (imageHeight spec))

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

private def pReg (n : Nat) : Reg .pred := ⟨s!"%p{n}"⟩
private def uReg (n : Nat) : Reg .u32 := ⟨s!"%r{n}"⟩
private def dReg (n : Nat) : Reg .u64 := ⟨s!"%rd{n}"⟩
private def fReg (n : Nat) : Reg .f32 := ⟨s!"%f{n}"⟩

namespace SceneReg

def pixelX : Reg .u32 := uReg 4
def pixelY : Reg .u32 := uReg 5
def sampleIdx : Reg .u32 := uReg 20
def rng : Reg .u32 := uReg 21
def bounceIdx : Reg .u32 := uReg 30
def hitKind : Reg .u32 := uReg 31
def rayOx : Reg .f32 := fReg 30
def rayOy : Reg .f32 := fReg 31
def rayOz : Reg .f32 := fReg 32
def rayDx : Reg .f32 := fReg 33
def rayDy : Reg .f32 := fReg 34
def rayDz : Reg .f32 := fReg 35
def throughputR : Reg .f32 := fReg 40
def throughputG : Reg .f32 := fReg 41
def throughputB : Reg .f32 := fReg 42
def radianceR : Reg .f32 := fReg 43
def radianceG : Reg .f32 := fReg 44
def radianceB : Reg .f32 := fReg 45
def hitT : Reg .f32 := fReg 50
def normalX : Reg .f32 := fReg 51
def normalY : Reg .f32 := fReg 52
def normalZ : Reg .f32 := fReg 53

end SceneReg

private def emitKernelSetupAndSampleLoop (spec : SceneSpec) : PTX Unit := do
  declPredRegs 64
  declU32Regs 192
  declU64Regs 32
  declF32Regs 512
  movR (uReg 0) ctaX
  movR (uReg 1) ctaY
  movR (uReg 2) tidX
  movR (uReg 3) tidY
  madLoRC (SceneReg.pixelX) (uReg 0) (16) (uReg 2)
  madLoRC (SceneReg.pixelY) (uReg 1) (16) (uReg 3)
  setpGeI (pReg 0) (SceneReg.pixelX) (imageWidth spec)
  braIf (pReg 0) "DONE"
  setpGeI (pReg 1) (SceneReg.pixelY) (imageHeight spec)
  braIf (pReg 1) "DONE"
  ldParam64 (dReg 0) "out_ptr"
  madLoRC (uReg 6) (SceneReg.pixelY) (imageWidth spec) (SceneReg.pixelX)
  mulWideRI (dReg 1) (uReg 6) (4)
  addS64 (dReg 2) (dReg 0) (dReg 1)
  movFI (fReg 400) (0x00000000 : UInt32)
  movFI (fReg 401) (0x3F800000 : UInt32)
  movFI (fReg 402) (0x40000000 : UInt32)
  movFI (fReg 403) (0x3F000000 : UInt32)
  movFI (fReg 404) (0xBF000000 : UInt32)
  movFI (fReg 405) (0x437F0000 : UInt32)
  movFI (fReg 406) (0x2F800000 : UInt32)
  movFI (fReg 407) (0x3A83126F : UInt32)
  movFI (fReg 408) (f32Nat (imageWidth spec))
  movFI (fReg 409) (f32Nat (imageHeight spec))
  movFI (fReg 410) (16.0 : Float)
  movFI (fReg 411) (-1.6 : Float)
  movFI (fReg 412) (-1.0 : Float)
  movFI (fReg 413) (-10.0 : Float)
  movFI (fReg 414) (0.001 : Float)
  movFI (fReg 415) ("1.0e20" : String)
  movFI (fReg 416) (invSamplesF32 spec)
  movFI (fReg 417) (0.03 : Float)
  movFI (fReg 418) (0.97 : Float)
  movFI (fReg 419) (0.08 : Float)
  movFI (fReg 420) (1.45 : Float)
  movFI (fReg 421) (0.6896552 : Float)
  movFI (fReg 422) (0.04 : Float)
  movFI (fReg 423) (0.96 : Float)
  movFI (fReg 424) (aspectF32 spec)
  cvtF32 (fReg 0) (SceneReg.pixelX)
  cvtF32 (fReg 1) (SceneReg.pixelY)
  movRC (SceneReg.sampleIdx) 0
  xorRR (SceneReg.rng) (SceneReg.pixelX) (SceneReg.pixelY)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1973) (9277)
  xorRI (SceneReg.rng) (SceneReg.rng) (26699)
  movFI (fReg 10) (0.0 : Float)
  movFI (fReg 11) (0.0 : Float)
  movFI (fReg 12) (0.0 : Float)
  label "SAMPLE_LOOP"
  setpGeI (pReg 2) (SceneReg.sampleIdx) (sampleCount spec)
  braIf (pReg 2) "SAMPLE_DONE"
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 20) (SceneReg.rng)
  mulF (fReg 20) (fReg 20) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 21) (SceneReg.rng)
  mulF (fReg 21) (fReg 21) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 26) (SceneReg.rng)
  mulF (fReg 26) (fReg 26) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 27) (SceneReg.rng)
  mulF (fReg 27) (fReg 27) (fReg 406)
  addF (fReg 22) (fReg 0) (fReg 20)
  addF (fReg 23) (fReg 1) (fReg 21)

private def emitPrimaryRayAndInitialIntersections (spec : SceneSpec) : PTX Unit := do
  divRn (fReg 24) (fReg 22) (fReg 408)
  divRn (fReg 25) (fReg 23) (fReg 409)
  mulF (fReg 24) (fReg 24) (fReg 402)
  mulF (fReg 25) (fReg 25) (fReg 402)
  addFI (fReg 24) (fReg 24) (-1.0 : Float)
  subFIR (fReg 25) (1.0 : Float) (fReg 25)
  mulF (fReg 24) (fReg 24) (fReg 424)
  addFI (fReg 28) (fReg 26) (-0.5 : Float)
  addFI (fReg 29) (fReg 27) (-0.5 : Float)
  mulFI (fReg 28) (fReg 28) (0.07 : Float)
  mulFI (fReg 29) (fReg 29) (0.07 : Float)
  movF (SceneReg.rayOx) (fReg 28)
  addFIR (SceneReg.rayOy) (1.15 : Float) (fReg 29)
  movFI (SceneReg.rayOz) (2.7 : Float)
  mulFI (SceneReg.rayDx) (fReg 24) (4.8125 : Float)
  mulFI (SceneReg.rayDy) (fReg 25) (4.8125 : Float)
  movFI (SceneReg.rayDz) (-5.0 : Float)
  subF (SceneReg.rayDx) (SceneReg.rayDx) (SceneReg.rayOx)
  addFI (SceneReg.rayDy) (SceneReg.rayDy) (1.15 : Float)
  subF (SceneReg.rayDy) (SceneReg.rayDy) (SceneReg.rayOy)
  subF (SceneReg.rayDz) (SceneReg.rayDz) (SceneReg.rayOz)
  mulF (fReg 36) (SceneReg.rayDx) (SceneReg.rayDx)
  fmaRn (fReg 36) (SceneReg.rayDy) (SceneReg.rayDy) (fReg 36)
  fmaRn (fReg 36) (SceneReg.rayDz) (SceneReg.rayDz) (fReg 36)
  rsqrt (fReg 37) (fReg 36)
  mulF (SceneReg.rayDx) (SceneReg.rayDx) (fReg 37)
  mulF (SceneReg.rayDy) (SceneReg.rayDy) (fReg 37)
  mulF (SceneReg.rayDz) (SceneReg.rayDz) (fReg 37)
  movFI (SceneReg.throughputR) (1.0 : Float)
  movFI (SceneReg.throughputG) (1.0 : Float)
  movFI (SceneReg.throughputB) (1.0 : Float)
  movFI (SceneReg.radianceR) (0.0 : Float)
  movFI (SceneReg.radianceG) (0.0 : Float)
  movFI (SceneReg.radianceB) (0.0 : Float)
  movRC (SceneReg.bounceIdx) 0
  label "BOUNCE_LOOP"
  setpGeI (pReg 3) (SceneReg.bounceIdx) (bounceCount spec)
  braIf (pReg 3) "PATH_DONE"
  movF (SceneReg.hitT) (fReg 415)
  movRC (SceneReg.hitKind) 0
  movFI (SceneReg.normalX) (0.0 : Float)
  movFI (SceneReg.normalY) (0.0 : Float)
  movFI (SceneReg.normalZ) (0.0 : Float)
  movFI (fReg 54) (0.0 : Float)
  movFI (fReg 55) (0.0 : Float)
  movFI (fReg 56) (0.0 : Float)
  absF (fReg 57) (SceneReg.rayDy)
  setpLtFI (pReg 4) (fReg 57) (1.0e-6 : Float)
  braIf (pReg 4) "GROUND_DONE"
  subF (fReg 58) (fReg 412) (SceneReg.rayOy)
  divRn (fReg 59) (fReg 58) (SceneReg.rayDy)
  setpLeF (pReg 5) (fReg 59) (fReg 414)
  braIf (pReg 5) "GROUND_DONE"
  setpGeF (pReg 6) (fReg 59) (SceneReg.hitT)
  braIf (pReg 6) "GROUND_DONE"
  movF (SceneReg.hitT) (fReg 59)
  movRC (SceneReg.hitKind) 1
  movFI (SceneReg.normalX) (0.0 : Float)
  movFI (SceneReg.normalY) (1.0 : Float)
  movFI (SceneReg.normalZ) (0.0 : Float)
  label "GROUND_DONE"
  absF (fReg 60) (SceneReg.rayDz)
  setpLtFI (pReg 7) (fReg 60) (1.0e-6 : Float)
  braIf (pReg 7) "WALL_DONE"
  subF (fReg 61) (fReg 413) (SceneReg.rayOz)
  divRn (fReg 62) (fReg 61) (SceneReg.rayDz)
  setpLeF (pReg 8) (fReg 62) (fReg 414)
  braIf (pReg 8) "WALL_DONE"
  setpGeF (pReg 9) (fReg 62) (SceneReg.hitT)

private def emitSphereIntersectionPassA (_spec : SceneSpec) : PTX Unit := do
  braIf (pReg 9) "WALL_DONE"
  movF (SceneReg.hitT) (fReg 62)
  movRC (SceneReg.hitKind) 2
  movFI (SceneReg.normalX) (0.0 : Float)
  movFI (SceneReg.normalY) (0.0 : Float)
  movFI (SceneReg.normalZ) (1.0 : Float)
  label "WALL_DONE"
  addFI (fReg 63) (SceneReg.rayOx) (1.35 : Float)
  addFI (fReg 64) (SceneReg.rayOy) (0.05 : Float)
  addFI (fReg 65) (SceneReg.rayOz) (4.7 : Float)
  mulF (fReg 66) (fReg 63) (SceneReg.rayDx)
  fmaRn (fReg 66) (fReg 64) (SceneReg.rayDy) (fReg 66)
  fmaRn (fReg 66) (fReg 65) (SceneReg.rayDz) (fReg 66)
  mulF (fReg 67) (fReg 63) (fReg 63)
  fmaRn (fReg 67) (fReg 64) (fReg 64) (fReg 67)
  fmaRn (fReg 67) (fReg 65) (fReg 65) (fReg 67)
  addFI (fReg 67) (fReg 67) (-0.9025 : Float)
  mulF (fReg 68) (fReg 66) (fReg 66)
  subF (fReg 68) (fReg 68) (fReg 67)
  setpLtFI (pReg 10) (fReg 68) (0.0 : Float)
  braIf (pReg 10) "GLASS_DONE"
  sqrtApprox (fReg 69) (fReg 68)
  negF (fReg 70) (fReg 66)
  subF (fReg 71) (fReg 70) (fReg 69)
  addF (fReg 72) (fReg 70) (fReg 69)
  setpGtF (pReg 11) (fReg 71) (fReg 414)
  braIf (pReg 11) "GLASS_T0"
  movF (fReg 71) (fReg 72)
  label "GLASS_T0"
  setpLeF (pReg 12) (fReg 71) (fReg 414)
  braIf (pReg 12) "GLASS_DONE"
  setpGeF (pReg 13) (fReg 71) (SceneReg.hitT)
  braIf (pReg 13) "GLASS_DONE"
  movF (SceneReg.hitT) (fReg 71)
  movRC (SceneReg.hitKind) 3
  fmaRn (fReg 73) (SceneReg.rayDx) (fReg 71) (fReg 63)
  fmaRn (fReg 74) (SceneReg.rayDy) (fReg 71) (fReg 64)
  fmaRn (fReg 75) (SceneReg.rayDz) (fReg 71) (fReg 65)
  mulFI (SceneReg.normalX) (fReg 73) (1.0526316 : Float)
  mulFI (SceneReg.normalY) (fReg 74) (1.0526316 : Float)
  mulFI (SceneReg.normalZ) (fReg 75) (1.0526316 : Float)
  label "GLASS_DONE"
  addFI (fReg 76) (SceneReg.rayOx) (-1.35 : Float)
  addFI (fReg 77) (SceneReg.rayOy) (0.25 : Float)
  addFI (fReg 78) (SceneReg.rayOz) (4.2 : Float)
  mulF (fReg 79) (fReg 76) (SceneReg.rayDx)
  fmaRn (fReg 79) (fReg 77) (SceneReg.rayDy) (fReg 79)
  fmaRn (fReg 79) (fReg 78) (SceneReg.rayDz) (fReg 79)
  mulF (fReg 80) (fReg 76) (fReg 76)
  fmaRn (fReg 80) (fReg 77) (fReg 77) (fReg 80)
  fmaRn (fReg 80) (fReg 78) (fReg 78) (fReg 80)
  addFI (fReg 80) (fReg 80) (-0.5625 : Float)
  mulF (fReg 81) (fReg 79) (fReg 79)
  subF (fReg 81) (fReg 81) (fReg 80)
  setpLtFI (pReg 14) (fReg 81) (0.0 : Float)
  braIf (pReg 14) "GOLD_DONE"
  sqrtApprox (fReg 82) (fReg 81)
  negF (fReg 83) (fReg 79)
  subF (fReg 84) (fReg 83) (fReg 82)
  addF (fReg 85) (fReg 83) (fReg 82)
  setpGtF (pReg 15) (fReg 84) (fReg 414)
  braIf (pReg 15) "GOLD_T0"
  movF (fReg 84) (fReg 85)
  label "GOLD_T0"
  setpLeF (pReg 16) (fReg 84) (fReg 414)
  braIf (pReg 16) "GOLD_DONE"
  setpGeF (pReg 17) (fReg 84) (SceneReg.hitT)
  braIf (pReg 17) "GOLD_DONE"
  movF (SceneReg.hitT) (fReg 84)
  movRC (SceneReg.hitKind) 4
  fmaRn (fReg 86) (SceneReg.rayDx) (fReg 84) (fReg 76)
  fmaRn (fReg 87) (SceneReg.rayDy) (fReg 84) (fReg 77)
  fmaRn (fReg 88) (SceneReg.rayDz) (fReg 84) (fReg 78)
  mulFI (SceneReg.normalX) (fReg 86) (1.3333334 : Float)
  mulFI (SceneReg.normalY) (fReg 87) (1.3333334 : Float)
  mulFI (SceneReg.normalZ) (fReg 88) (1.3333334 : Float)

private def emitHitDispatchAndLightSamplingSetup (_spec : SceneSpec) : PTX Unit := do
  label "GOLD_DONE"
  addFI (fReg 89) (SceneReg.rayOx) (-0.1 : Float)
  addFI (fReg 90) (SceneReg.rayOy) (-0.15 : Float)
  addFI (fReg 91) (SceneReg.rayOz) (6.5 : Float)
  mulF (fReg 92) (fReg 89) (SceneReg.rayDx)
  fmaRn (fReg 92) (fReg 90) (SceneReg.rayDy) (fReg 92)
  fmaRn (fReg 92) (fReg 91) (SceneReg.rayDz) (fReg 92)
  mulF (fReg 93) (fReg 89) (fReg 89)
  fmaRn (fReg 93) (fReg 90) (fReg 90) (fReg 93)
  fmaRn (fReg 93) (fReg 91) (fReg 91) (fReg 93)
  addFI (fReg 93) (fReg 93) (-1.3225 : Float)
  mulF (fReg 94) (fReg 92) (fReg 92)
  subF (fReg 94) (fReg 94) (fReg 93)
  setpLtFI (pReg 18) (fReg 94) (0.0 : Float)
  braIf (pReg 18) "BLUE_DONE"
  sqrtApprox (fReg 95) (fReg 94)
  negF (fReg 96) (fReg 92)
  subF (fReg 97) (fReg 96) (fReg 95)
  addF (fReg 98) (fReg 96) (fReg 95)
  setpGtF (pReg 19) (fReg 97) (fReg 414)
  braIf (pReg 19) "BLUE_T0"
  movF (fReg 97) (fReg 98)
  label "BLUE_T0"
  setpLeF (pReg 20) (fReg 97) (fReg 414)
  braIf (pReg 20) "BLUE_DONE"
  setpGeF (pReg 21) (fReg 97) (SceneReg.hitT)
  braIf (pReg 21) "BLUE_DONE"
  movF (SceneReg.hitT) (fReg 97)
  movRC (SceneReg.hitKind) 5
  fmaRn (fReg 99) (SceneReg.rayDx) (fReg 97) (fReg 89)
  fmaRn (fReg 100) (SceneReg.rayDy) (fReg 97) (fReg 90)
  fmaRn (fReg 101) (SceneReg.rayDz) (fReg 97) (fReg 91)
  mulFI (SceneReg.normalX) (fReg 99) (0.86956525 : Float)
  mulFI (SceneReg.normalY) (fReg 100) (0.86956525 : Float)
  mulFI (SceneReg.normalZ) (fReg 101) (0.86956525 : Float)
  label "BLUE_DONE"
  setpEqI (pReg 22) (SceneReg.hitKind) (0)
  braIf (pReg 22) "SHADE_SKY"
  fmaRn (fReg 102) (SceneReg.rayDx) (SceneReg.hitT) (SceneReg.rayOx)
  fmaRn (fReg 103) (SceneReg.rayDy) (SceneReg.hitT) (SceneReg.rayOy)
  fmaRn (fReg 104) (SceneReg.rayDz) (SceneReg.hitT) (SceneReg.rayOz)
  fmaRn (fReg 105) (SceneReg.normalX) (fReg 414) (fReg 102)
  fmaRn (fReg 106) (SceneReg.normalY) (fReg 414) (fReg 103)
  fmaRn (fReg 107) (SceneReg.normalZ) (fReg 414) (fReg 104)
  mulF (fReg 108) (SceneReg.hitT) (fReg 417)
  minFI (fReg 108) (fReg 108) (0.35 : Float)
  mulFI (fReg 109) (fReg 108) (0.04 : Float)
  mulFI (fReg 210) (fReg 108) (0.07 : Float)
  mulFI (fReg 211) (fReg 108) (0.12 : Float)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 109) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 210) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 211) (SceneReg.radianceB)
  movFI (fReg 110) (0.0 : Float)
  movFI (fReg 111) (0.0 : Float)
  movFI (fReg 112) (0.0 : Float)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 113) (SceneReg.rng)
  mulF (fReg 113) (fReg 113) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 114) (SceneReg.rng)
  mulF (fReg 114) (fReg 114) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 115) (SceneReg.rng)
  mulF (fReg 115) (fReg 115) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 116) (SceneReg.rng)
  mulF (fReg 116) (fReg 116) (fReg 406)
  addFI (fReg 113) (fReg 113) (-0.5 : Float)
  addFI (fReg 114) (fReg 114) (-0.5 : Float)
  addFI (fReg 115) (fReg 115) (-0.5 : Float)

private def emitDirectLightVisibilityA (_spec : SceneSpec) : PTX Unit := do
  addFI (fReg 116) (fReg 116) (-0.5 : Float)
  fmaFII (fReg 117) (fReg 113) (1.25 : Float) (-4.0 : Float)
  fmaFII (fReg 118) (fReg 114) (0.35 : Float) (5.5 : Float)
  fmaFII (fReg 119) (fReg 115) (1.25 : Float) (-2.0 : Float)
  subF (fReg 116) (fReg 117) (fReg 105)
  subF (fReg 117) (fReg 118) (fReg 106)
  subF (fReg 118) (fReg 119) (fReg 107)
  mulF (fReg 119) (fReg 116) (fReg 116)
  fmaRn (fReg 119) (fReg 117) (fReg 117) (fReg 119)
  fmaRn (fReg 119) (fReg 118) (fReg 118) (fReg 119)
  rsqrt (fReg 120) (fReg 119)
  mulF (fReg 121) (fReg 116) (fReg 120)
  mulF (fReg 122) (fReg 117) (fReg 120)
  mulF (fReg 123) (fReg 118) (fReg 120)
  mulF (fReg 124) (SceneReg.normalX) (fReg 121)
  fmaRn (fReg 124) (SceneReg.normalY) (fReg 122) (fReg 124)
  fmaRn (fReg 124) (SceneReg.normalZ) (fReg 123) (fReg 124)
  maxFI (fReg 124) (fReg 124) (0.0 : Float)
  setpLeFI (pReg 23) (fReg 124) (0.0 : Float)
  braIf (pReg 23) "LIGHTA_DONE"
  movFI (fReg 125) (0.999 : Float)
  sqrtApprox (fReg 126) (fReg 119)
  mulF (fReg 126) (fReg 126) (fReg 125)
  addFI (fReg 127) (fReg 105) (1.35 : Float)
  addFI (fReg 128) (fReg 106) (0.05 : Float)
  addFI (fReg 129) (fReg 107) (4.7 : Float)
  mulF (fReg 130) (fReg 127) (fReg 121)
  fmaRn (fReg 130) (fReg 128) (fReg 122) (fReg 130)
  fmaRn (fReg 130) (fReg 129) (fReg 123) (fReg 130)
  mulF (fReg 131) (fReg 127) (fReg 127)
  fmaRn (fReg 131) (fReg 128) (fReg 128) (fReg 131)
  fmaRn (fReg 131) (fReg 129) (fReg 129) (fReg 131)
  addFI (fReg 131) (fReg 131) (-0.9025 : Float)
  mulF (fReg 132) (fReg 130) (fReg 130)
  subF (fReg 132) (fReg 132) (fReg 131)
  setpGeFI (pReg 24) (fReg 132) (0.0 : Float)
  braIfNot (pReg 24) "LIGHTA_GOLD"
  sqrtApprox (fReg 133) (fReg 132)
  negF (fReg 134) (fReg 130)
  subF (fReg 135) (fReg 134) (fReg 133)
  setpGtF (pReg 25) (fReg 135) (fReg 414)
  braIfNot (pReg 25) "LIGHTA_GOLD"
  setpLtF (pReg 26) (fReg 135) (fReg 126)
  braIf (pReg 26) "LIGHTA_DONE"
  label "LIGHTA_GOLD"
  addFI (fReg 136) (fReg 105) (-1.35 : Float)
  addFI (fReg 137) (fReg 106) (0.25 : Float)
  addFI (fReg 138) (fReg 107) (4.2 : Float)
  mulF (fReg 139) (fReg 136) (fReg 121)
  fmaRn (fReg 139) (fReg 137) (fReg 122) (fReg 139)
  fmaRn (fReg 139) (fReg 138) (fReg 123) (fReg 139)
  mulF (fReg 140) (fReg 136) (fReg 136)
  fmaRn (fReg 140) (fReg 137) (fReg 137) (fReg 140)
  fmaRn (fReg 140) (fReg 138) (fReg 138) (fReg 140)
  addFI (fReg 140) (fReg 140) (-0.5625 : Float)
  mulF (fReg 141) (fReg 139) (fReg 139)
  subF (fReg 141) (fReg 141) (fReg 140)
  setpGeFI (pReg 27) (fReg 141) (0.0 : Float)
  braIfNot (pReg 27) "LIGHTA_BLUE"
  sqrtApprox (fReg 142) (fReg 141)
  negF (fReg 143) (fReg 139)
  subF (fReg 144) (fReg 143) (fReg 142)
  setpGtF (pReg 28) (fReg 144) (fReg 414)
  braIfNot (pReg 28) "LIGHTA_BLUE"
  setpLtF (pReg 29) (fReg 144) (fReg 126)
  braIf (pReg 29) "LIGHTA_DONE"
  label "LIGHTA_BLUE"
  addFI (fReg 145) (fReg 105) (-0.1 : Float)
  addFI (fReg 146) (fReg 106) (-0.15 : Float)
  addFI (fReg 147) (fReg 107) (6.5 : Float)
  mulF (fReg 148) (fReg 145) (fReg 121)
  fmaRn (fReg 148) (fReg 146) (fReg 122) (fReg 148)
  fmaRn (fReg 148) (fReg 147) (fReg 123) (fReg 148)
  mulF (fReg 149) (fReg 145) (fReg 145)
  fmaRn (fReg 149) (fReg 146) (fReg 146) (fReg 149)
  fmaRn (fReg 149) (fReg 147) (fReg 147) (fReg 149)

private def emitDirectLightVisibilityBAndDispatch (_spec : SceneSpec) : PTX Unit := do
  addFI (fReg 149) (fReg 149) (-1.3225 : Float)
  mulF (fReg 150) (fReg 148) (fReg 148)
  subF (fReg 150) (fReg 150) (fReg 149)
  setpGeFI (pReg 30) (fReg 150) (0.0 : Float)
  braIfNot (pReg 30) "LIGHTA_APPLY"
  sqrtApprox (fReg 151) (fReg 150)
  negF (fReg 152) (fReg 148)
  subF (fReg 153) (fReg 152) (fReg 151)
  setpGtF (pReg 31) (fReg 153) (fReg 414)
  braIfNot (pReg 31) "LIGHTA_APPLY"
  setpLtF (pReg 32) (fReg 153) (fReg 126)
  braIf (pReg 32) "LIGHTA_DONE"
  label "LIGHTA_APPLY"
  rcp (fReg 154) (fReg 119)
  mulF (fReg 155) (fReg 124) (fReg 154)
  fmaFIR (fReg 110) (fReg 155) (90.0 : Float) (fReg 110)
  fmaFIR (fReg 111) (fReg 155) (74.0 : Float) (fReg 111)
  fmaFIR (fReg 112) (fReg 155) (58.0 : Float) (fReg 112)
  label "LIGHTA_DONE"
  fmaFII (fReg 156) (fReg 115) (1.35 : Float) (4.5 : Float)
  fmaFII (fReg 157) (fReg 116) (0.45 : Float) (4.0 : Float)
  fmaFII (fReg 158) (fReg 114) (1.45 : Float) (-7.0 : Float)
  subF (fReg 159) (fReg 156) (fReg 105)
  subF (fReg 160) (fReg 157) (fReg 106)
  subF (fReg 161) (fReg 158) (fReg 107)
  mulF (fReg 162) (fReg 159) (fReg 159)
  fmaRn (fReg 162) (fReg 160) (fReg 160) (fReg 162)
  fmaRn (fReg 162) (fReg 161) (fReg 161) (fReg 162)
  rsqrt (fReg 163) (fReg 162)
  mulF (fReg 164) (fReg 159) (fReg 163)
  mulF (fReg 165) (fReg 160) (fReg 163)
  mulF (fReg 166) (fReg 161) (fReg 163)
  mulF (fReg 167) (SceneReg.normalX) (fReg 164)
  fmaRn (fReg 167) (SceneReg.normalY) (fReg 165) (fReg 167)
  fmaRn (fReg 167) (SceneReg.normalZ) (fReg 166) (fReg 167)
  maxFI (fReg 167) (fReg 167) (0.0 : Float)
  setpLeFI (pReg 33) (fReg 167) (0.0 : Float)
  braIf (pReg 33) "LIGHTB_DONE"
  rcp (fReg 168) (fReg 162)
  mulF (fReg 169) (fReg 167) (fReg 168)
  fmaFIR (fReg 110) (fReg 169) (40.0 : Float) (fReg 110)
  fmaFIR (fReg 111) (fReg 169) (66.0 : Float) (fReg 111)
  fmaFIR (fReg 112) (fReg 169) (110.0 : Float) (fReg 112)
  label "LIGHTB_DONE"
  fmaFII (fReg 170) (fReg 113) (2.0 : Float) (0.0 : Float)
  fmaFII (fReg 171) (fReg 114) (0.6 : Float) (6.5 : Float)
  fmaFII (fReg 172) (fReg 116) (1.8 : Float) (2.5 : Float)
  subF (fReg 173) (fReg 170) (fReg 105)
  subF (fReg 174) (fReg 171) (fReg 106)
  subF (fReg 175) (fReg 172) (fReg 107)
  mulF (fReg 176) (fReg 173) (fReg 173)
  fmaRn (fReg 176) (fReg 174) (fReg 174) (fReg 176)
  fmaRn (fReg 176) (fReg 175) (fReg 175) (fReg 176)
  rsqrt (fReg 177) (fReg 176)
  mulF (fReg 178) (fReg 173) (fReg 177)
  mulF (fReg 179) (fReg 174) (fReg 177)
  mulF (fReg 180) (fReg 175) (fReg 177)
  mulF (fReg 181) (SceneReg.normalX) (fReg 178)
  fmaRn (fReg 181) (SceneReg.normalY) (fReg 179) (fReg 181)
  fmaRn (fReg 181) (SceneReg.normalZ) (fReg 180) (fReg 181)
  maxFI (fReg 181) (fReg 181) (0.0 : Float)
  setpLeFI (pReg 34) (fReg 181) (0.0 : Float)
  braIf (pReg 34) "LIGHTC_DONE"
  rcp (fReg 182) (fReg 176)
  mulF (fReg 183) (fReg 181) (fReg 182)
  fmaFIR (fReg 110) (fReg 183) (28.0 : Float) (fReg 110)
  fmaFIR (fReg 111) (fReg 183) (40.0 : Float) (fReg 111)
  fmaFIR (fReg 112) (fReg 183) (72.0 : Float) (fReg 112)
  label "LIGHTC_DONE"
  setpEqI (pReg 35) (SceneReg.hitKind) (1)
  braIf (pReg 35) "SHADE_GROUND"
  setpEqI (pReg 36) (SceneReg.hitKind) (2)
  braIf (pReg 36) "SHADE_WALL"
  setpEqI (pReg 37) (SceneReg.hitKind) (3)
  braIf (pReg 37) "SHADE_GLASS"

private def emitMaterialShading (spec : SceneSpec) : PTX Unit := do
  setpEqI (pReg 38) (SceneReg.hitKind) (4)
  braIf (pReg 38) "SHADE_GOLD"
  bra "SHADE_BLUE"
  label "SHADE_GROUND"
  cvtRziS32F32 (uReg 40) (fReg 102)
  cvtRziS32F32 (uReg 41) (fReg 104)
  andR (uReg 42) (uReg 40) (1)
  andR (uReg 43) (uReg 41) (1)
  xorRR (uReg 44) (uReg 42) (uReg 43)
  setpEqI (pReg 39) (uReg 44) (0)
  braIf (pReg 39) "GROUND_LIGHT"
  movFI (fReg 170) (channelF32 spec.palette.groundLight.red)
  movFI (fReg 171) (channelF32 spec.palette.groundLight.green)
  movFI (fReg 172) (channelF32 spec.palette.groundLight.blue)
  bra "GROUND_APPLY"
  label "GROUND_LIGHT"
  movFI (fReg 170) (channelF32 spec.palette.groundDark.red)
  movFI (fReg 171) (channelF32 spec.palette.groundDark.green)
  movFI (fReg 172) (channelF32 spec.palette.groundDark.blue)
  label "GROUND_APPLY"
  mulF (fReg 212) (fReg 110) (fReg 170)
  mulF (fReg 213) (fReg 111) (fReg 171)
  mulF (fReg 214) (fReg 112) (fReg 172)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 212) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 213) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 214) (SceneReg.radianceB)
  mulF (SceneReg.throughputR) (SceneReg.throughputR) (fReg 170)
  mulF (SceneReg.throughputG) (SceneReg.throughputG) (fReg 171)
  mulF (SceneReg.throughputB) (SceneReg.throughputB) (fReg 172)
  bra "DIFFUSE_BOUNCE"
  label "SHADE_WALL"
  movFI (fReg 173) (channelF32 spec.palette.wall.red)
  movFI (fReg 174) (channelF32 spec.palette.wall.green)
  movFI (fReg 175) (channelF32 spec.palette.wall.blue)
  mulF (fReg 212) (fReg 110) (fReg 173)
  mulF (fReg 213) (fReg 111) (fReg 174)
  mulF (fReg 214) (fReg 112) (fReg 175)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 212) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 213) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 214) (SceneReg.radianceB)
  mulF (SceneReg.throughputR) (SceneReg.throughputR) (fReg 173)
  mulF (SceneReg.throughputG) (SceneReg.throughputG) (fReg 174)
  mulF (SceneReg.throughputB) (SceneReg.throughputB) (fReg 175)
  bra "DIFFUSE_BOUNCE"
  label "SHADE_BLUE"
  movFI (fReg 176) (channelF32 spec.palette.diffuseSphere.red)
  movFI (fReg 177) (channelF32 spec.palette.diffuseSphere.green)
  movFI (fReg 178) (channelF32 spec.palette.diffuseSphere.blue)
  mulF (fReg 212) (fReg 110) (fReg 176)
  mulF (fReg 213) (fReg 111) (fReg 177)
  mulF (fReg 214) (fReg 112) (fReg 178)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 212) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 213) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 214) (SceneReg.radianceB)
  mulF (SceneReg.throughputR) (SceneReg.throughputR) (fReg 176)
  mulF (SceneReg.throughputG) (SceneReg.throughputG) (fReg 177)
  mulF (SceneReg.throughputB) (SceneReg.throughputB) (fReg 178)
  bra "DIFFUSE_BOUNCE"
  label "SHADE_GOLD"
  movFI (fReg 179) (channelF32 spec.palette.metalSphere.red)
  movFI (fReg 180) (channelF32 spec.palette.metalSphere.green)
  movFI (fReg 181) (channelF32 spec.palette.metalSphere.blue)
  mulFI (fReg 212) (fReg 110) (0.15 : Float)
  mulFI (fReg 213) (fReg 111) (0.13 : Float)
  mulFI (fReg 214) (fReg 112) (0.09 : Float)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 212) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 213) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 214) (SceneReg.radianceB)
  mulF (fReg 182) (SceneReg.rayDx) (SceneReg.normalX)
  fmaRn (fReg 182) (SceneReg.rayDy) (SceneReg.normalY) (fReg 182)
  fmaRn (fReg 182) (SceneReg.rayDz) (SceneReg.normalZ) (fReg 182)
  mulFI (fReg 183) (fReg 182) (2.0 : Float)
  negF (fReg 226) (SceneReg.normalX)
  negF (fReg 227) (SceneReg.normalY)
  negF (fReg 228) (SceneReg.normalZ)
  fmaRn (SceneReg.rayDx) (fReg 226) (fReg 183) (SceneReg.rayDx)

private def emitMetalAndGlassSetup (_spec : SceneSpec) : PTX Unit := do
  fmaRn (SceneReg.rayDy) (fReg 227) (fReg 183) (SceneReg.rayDy)
  fmaRn (SceneReg.rayDz) (fReg 228) (fReg 183) (SceneReg.rayDz)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 184) (SceneReg.rng)
  mulF (fReg 184) (fReg 184) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 185) (SceneReg.rng)
  mulF (fReg 185) (fReg 185) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 186) (SceneReg.rng)
  mulF (fReg 186) (fReg 186) (fReg 406)
  addFI (fReg 184) (fReg 184) (-0.5 : Float)
  addFI (fReg 185) (fReg 185) (-0.5 : Float)
  addFI (fReg 186) (fReg 186) (-0.5 : Float)
  fmaRn (SceneReg.rayDx) (fReg 184) (fReg 419) (SceneReg.rayDx)
  fmaRn (SceneReg.rayDy) (fReg 185) (fReg 419) (SceneReg.rayDy)
  fmaRn (SceneReg.rayDz) (fReg 186) (fReg 419) (SceneReg.rayDz)
  mulF (fReg 187) (SceneReg.rayDx) (SceneReg.rayDx)
  fmaRn (fReg 187) (SceneReg.rayDy) (SceneReg.rayDy) (fReg 187)
  fmaRn (fReg 187) (SceneReg.rayDz) (SceneReg.rayDz) (fReg 187)
  rsqrt (fReg 188) (fReg 187)
  mulF (SceneReg.rayDx) (SceneReg.rayDx) (fReg 188)
  mulF (SceneReg.rayDy) (SceneReg.rayDy) (fReg 188)
  mulF (SceneReg.rayDz) (SceneReg.rayDz) (fReg 188)
  mulF (SceneReg.throughputR) (SceneReg.throughputR) (fReg 179)
  mulF (SceneReg.throughputG) (SceneReg.throughputG) (fReg 180)
  mulF (SceneReg.throughputB) (SceneReg.throughputB) (fReg 181)
  movF (SceneReg.rayOx) (fReg 105)
  movF (SceneReg.rayOy) (fReg 106)
  movF (SceneReg.rayOz) (fReg 107)
  bra "RR_STEP"
  label "SHADE_GLASS"
  mulF (fReg 189) (SceneReg.rayDx) (SceneReg.normalX)
  fmaRn (fReg 189) (SceneReg.rayDy) (SceneReg.normalY) (fReg 189)
  fmaRn (fReg 189) (SceneReg.rayDz) (SceneReg.normalZ) (fReg 189)
  setpGtFI (pReg 39) (fReg 189) (0.0 : Float)
  braIf (pReg 39) "GLASS_INSIDE"
  movF (fReg 190) (SceneReg.normalX)
  movF (fReg 191) (SceneReg.normalY)
  movF (fReg 192) (SceneReg.normalZ)
  negF (fReg 193) (fReg 189)
  movF (fReg 194) (fReg 421)
  bra "GLASS_COMMON"
  label "GLASS_INSIDE"
  negF (fReg 190) (SceneReg.normalX)
  negF (fReg 191) (SceneReg.normalY)
  negF (fReg 192) (SceneReg.normalZ)
  movF (fReg 193) (fReg 189)
  movF (fReg 194) (fReg 420)
  label "GLASS_COMMON"
  subFIR (fReg 195) (1.0 : Float) (fReg 193)
  mulF (fReg 196) (fReg 195) (fReg 195)
  mulF (fReg 196) (fReg 196) (fReg 196)
  mulF (fReg 196) (fReg 196) (fReg 195)
  fmaRn (fReg 197) (fReg 196) (fReg 423) (fReg 422)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 198) (SceneReg.rng)
  mulF (fReg 198) (fReg 198) (fReg 406)
  mulF (fReg 199) (fReg 193) (fReg 193)
  subFIR (fReg 200) (1.0 : Float) (fReg 199)
  mulF (fReg 201) (fReg 194) (fReg 194)
  mulF (fReg 200) (fReg 200) (fReg 201)
  subFIR (fReg 202) (1.0 : Float) (fReg 200)
  setpLtFI (pReg 40) (fReg 202) (0.0 : Float)
  braIf (pReg 40) "GLASS_REFLECT"
  setpLtF (pReg 41) (fReg 198) (fReg 197)
  braIf (pReg 41) "GLASS_REFLECT"
  sqrtApprox (fReg 203) (fReg 202)
  mulF (fReg 204) (fReg 194) (SceneReg.rayDx)
  mulF (fReg 205) (fReg 194) (SceneReg.rayDy)
  mulF (fReg 206) (fReg 194) (SceneReg.rayDz)
  mulF (fReg 207) (fReg 194) (fReg 193)
  subF (fReg 207) (fReg 207) (fReg 203)
  fmaRn (SceneReg.rayDx) (fReg 190) (fReg 207) (fReg 204)
  fmaRn (SceneReg.rayDy) (fReg 191) (fReg 207) (fReg 205)
  fmaRn (SceneReg.rayDz) (fReg 192) (fReg 207) (fReg 206)
  mulFI (SceneReg.throughputR) (SceneReg.throughputR) (0.98 : Float)
  mulFI (SceneReg.throughputG) (SceneReg.throughputG) (0.99 : Float)

private def emitGlassResolveAndDiffuseBounce (_spec : SceneSpec) : PTX Unit := do
  mulFI (SceneReg.throughputB) (SceneReg.throughputB) (1.0 : Float)
  movF (SceneReg.rayOx) (fReg 102)
  movF (SceneReg.rayOy) (fReg 103)
  movF (SceneReg.rayOz) (fReg 104)
  bra "RR_STEP"
  label "GLASS_REFLECT"
  mulF (fReg 208) (SceneReg.rayDx) (fReg 190)
  fmaRn (fReg 208) (SceneReg.rayDy) (fReg 191) (fReg 208)
  fmaRn (fReg 208) (SceneReg.rayDz) (fReg 192) (fReg 208)
  mulFI (fReg 209) (fReg 208) (2.0 : Float)
  negF (fReg 229) (fReg 190)
  negF (fReg 230) (fReg 191)
  negF (fReg 231) (fReg 192)
  fmaRn (SceneReg.rayDx) (fReg 229) (fReg 209) (SceneReg.rayDx)
  fmaRn (SceneReg.rayDy) (fReg 230) (fReg 209) (SceneReg.rayDy)
  fmaRn (SceneReg.rayDz) (fReg 231) (fReg 209) (SceneReg.rayDz)
  mulFI (SceneReg.throughputR) (SceneReg.throughputR) (0.99 : Float)
  mulFI (SceneReg.throughputG) (SceneReg.throughputG) (0.99 : Float)
  mulFI (SceneReg.throughputB) (SceneReg.throughputB) (1.0 : Float)
  movF (SceneReg.rayOx) (fReg 105)
  movF (SceneReg.rayOy) (fReg 106)
  movF (SceneReg.rayOz) (fReg 107)
  bra "RR_STEP"
  label "DIFFUSE_BOUNCE"
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 210) (SceneReg.rng)
  mulF (fReg 210) (fReg 210) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 211) (SceneReg.rng)
  mulF (fReg 211) (fReg 211) (fReg 406)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 212) (SceneReg.rng)
  mulF (fReg 212) (fReg 212) (fReg 406)
  addFI (fReg 210) (fReg 210) (-0.5 : Float)
  addFI (fReg 211) (fReg 211) (-0.5 : Float)
  addFI (fReg 212) (fReg 212) (-0.5 : Float)
  fmaFIR (SceneReg.rayDx) (SceneReg.normalX) (1.4 : Float) (fReg 210)
  fmaFIR (SceneReg.rayDy) (SceneReg.normalY) (1.4 : Float) (fReg 211)
  fmaFIR (SceneReg.rayDz) (SceneReg.normalZ) (1.4 : Float) (fReg 212)
  mulF (fReg 213) (SceneReg.rayDx) (SceneReg.rayDx)
  fmaRn (fReg 213) (SceneReg.rayDy) (SceneReg.rayDy) (fReg 213)
  fmaRn (fReg 213) (SceneReg.rayDz) (SceneReg.rayDz) (fReg 213)
  rsqrt (fReg 214) (fReg 213)
  mulF (SceneReg.rayDx) (SceneReg.rayDx) (fReg 214)
  mulF (SceneReg.rayDy) (SceneReg.rayDy) (fReg 214)
  mulF (SceneReg.rayDz) (SceneReg.rayDz) (fReg 214)
  movF (SceneReg.rayOx) (fReg 105)
  movF (SceneReg.rayOy) (fReg 106)
  movF (SceneReg.rayOz) (fReg 107)
  bra "RR_STEP"
  label "SHADE_SKY"
  addFI (fReg 215) (SceneReg.rayDy) (1.0 : Float)
  mulFI (fReg 215) (fReg 215) (0.5 : Float)
  mulFI (fReg 216) (fReg 215) (0.35 : Float)
  mulFI (fReg 217) (fReg 215) (0.45 : Float)
  mulFI (fReg 218) (fReg 215) (0.75 : Float)
  addFI (fReg 216) (fReg 216) (0.04 : Float)
  addFI (fReg 217) (fReg 217) (0.06 : Float)
  addFI (fReg 218) (fReg 218) (0.12 : Float)
  fmaRn (SceneReg.radianceR) (SceneReg.throughputR) (fReg 216) (SceneReg.radianceR)
  fmaRn (SceneReg.radianceG) (SceneReg.throughputG) (fReg 217) (SceneReg.radianceG)
  fmaRn (SceneReg.radianceB) (SceneReg.throughputB) (fReg 218) (SceneReg.radianceB)
  bra "PATH_DONE"
  label "RR_STEP"
  setpLtI (pReg 42) (SceneReg.bounceIdx) (2)
  braIf (pReg 42) "RR_SKIP"
  maxF (fReg 216) (SceneReg.throughputR) (SceneReg.throughputG)
  maxF (fReg 216) (fReg 216) (SceneReg.throughputB)
  minF (fReg 216) (fReg 216) (fReg 418)
  maxFI (fReg 216) (fReg 216) (0.10 : Float)
  madLoRII (SceneReg.rng) (SceneReg.rng) (1664525) (1013904223)
  cvtF32 (fReg 217) (SceneReg.rng)
  mulF (fReg 217) (fReg 217) (fReg 406)
  setpGtF (pReg 43) (fReg 217) (fReg 216)
  braIf (pReg 43) "PATH_DONE"
  rcp (fReg 218) (fReg 216)
  mulF (SceneReg.throughputR) (SceneReg.throughputR) (fReg 218)

private def emitSkyRrAndFinalize (_spec : SceneSpec) : PTX Unit := do
  mulF (SceneReg.throughputG) (SceneReg.throughputG) (fReg 218)
  mulF (SceneReg.throughputB) (SceneReg.throughputB) (fReg 218)
  label "RR_SKIP"
  addRI (SceneReg.bounceIdx) (SceneReg.bounceIdx) (1)
  bra "BOUNCE_LOOP"
  label "PATH_DONE"
  addF (fReg 10) (fReg 10) (SceneReg.radianceR)
  addF (fReg 11) (fReg 11) (SceneReg.radianceG)
  addF (fReg 12) (fReg 12) (SceneReg.radianceB)
  addRI (SceneReg.sampleIdx) (SceneReg.sampleIdx) (1)
  bra "SAMPLE_LOOP"
  label "SAMPLE_DONE"
  mulF (fReg 10) (fReg 10) (fReg 416)
  mulF (fReg 11) (fReg 11) (fReg 416)
  mulF (fReg 12) (fReg 12) (fReg 416)
  addFI (fReg 220) (fReg 10) (1.0 : Float)
  addFI (fReg 221) (fReg 11) (1.0 : Float)
  addFI (fReg 222) (fReg 12) (1.0 : Float)
  divRn (fReg 10) (fReg 10) (fReg 220)
  divRn (fReg 11) (fReg 11) (fReg 221)
  divRn (fReg 12) (fReg 12) (fReg 222)
  sqrtApprox (fReg 10) (fReg 10)
  sqrtApprox (fReg 11) (fReg 11)
  sqrtApprox (fReg 12) (fReg 12)
  minFI (fReg 10) (fReg 10) (1.0 : Float)
  minFI (fReg 11) (fReg 11) (1.0 : Float)
  minFI (fReg 12) (fReg 12) (1.0 : Float)
  divRn (fReg 223) (fReg 0) (fReg 408)
  divRn (fReg 224) (fReg 1) (fReg 409)
  subFI (fReg 223) (fReg 223) (0.5 : Float)
  subFI (fReg 224) (fReg 224) (0.5 : Float)
  mulF (fReg 225) (fReg 223) (fReg 223)
  fmaRn (fReg 225) (fReg 224) (fReg 224) (fReg 225)
  mulFI (fReg 225) (fReg 225) (1.85 : Float)
  subFIR (fReg 226) (1.0 : Float) (fReg 225)
  maxFI (fReg 226) (fReg 226) (0.72 : Float)
  mulF (fReg 10) (fReg 10) (fReg 226)
  mulF (fReg 11) (fReg 11) (fReg 226)
  mulF (fReg 12) (fReg 12) (fReg 226)
  mulF (fReg 223) (fReg 10) (fReg 405)
  mulF (fReg 224) (fReg 11) (fReg 405)
  mulF (fReg 225) (fReg 12) (fReg 405)
  cvtRniU32F32 (uReg 60) (fReg 223)
  cvtRniU32F32 (uReg 61) (fReg 224)
  cvtRniU32F32 (uReg 62) (fReg 225)
  shlR (uReg 61) (uReg 61) (8)
  shlR (uReg 60) (uReg 60) (16)
  movRC (uReg 63) 0xFF000000
  orRR (uReg 64) (uReg 62) (uReg 61)
  orRR (uReg 65) (uReg 64) (uReg 60)
  orRR (uReg 66) (uReg 65) (uReg 63)
  stGlobalU32 (dReg 2) (uReg 66)
  label "DONE"
  ptxRet

def emitSceneKernel (spec : SceneSpec) : PTX Unit := do
  emitKernelSetupAndSampleLoop spec
  emitPrimaryRayAndInitialIntersections spec
  emitSphereIntersectionPassA spec
  emitHitDispatchAndLightSamplingSetup spec
  emitDirectLightVisibilityA spec
  emitDirectLightVisibilityBAndDispatch spec
  emitMaterialShading spec
  emitMetalAndGlassSetup spec
  emitGlassResolveAndDiffuseBounce spec
  emitSkyRrAndFinalize spec

def ptxSource (spec : SceneSpec) : String :=
  buildModuleWith { version := "7.0", target := "sm_50" } [{
    name := "main"
    params := ["out_ptr"]
    body := do
      emitSceneKernel spec
  }]

def ptxOff : Nat := 0x0100
def ptxRegion : Nat := 131072
def bindOff : Nat := ptxOff + ptxRegion
def filenameOff : Nat := bindOff + 16
def filenameRegion : Nat := 256
def clifIrOff : Nat := filenameOff + filenameRegion
def clifIrRegion : Nat := 4096
def bmpHeaderOff : Nat := clifIrOff + clifIrRegion
def pixelsOff : Nat := bmpHeaderOff + 54

open AlgorithmLib.IR in
def clifIrSource (spec : SceneSpec) : String := buildProgram do
  let fnWrite ← declareFileWrite
  let cuda ← declareCudaFFI

  let ptr ← entryBlock
  cudaInit cuda ptr
  let dataSz ← iconst64 (pixelBytes spec)
  let bufId ← cudaCreateBuffer cuda ptr dataSz
  let ptxOffV ← iconst64 ptxOff
  let nBufs ← iconst32 1
  let bindOffV ← iconst64 bindOff
  let gridX ← iconst32 ((imageWidth spec + 15) / 16)
  let gridY ← iconst32 ((imageHeight spec + 15) / 16)
  let one32 ← iconst32 1
  let blk16 ← iconst32 16
  let _ := bufId
  let _ ← cudaLaunch cuda ptr ptxOffV nBufs bindOffV gridX gridY one32 blk16 blk16 one32
  let pxOffV ← iconst64 pixelsOff
  let _ ← cudaDownload cuda ptr bufId pxOffV dataSz
  cudaCleanup cuda ptr
  let total ← iconst64 (54 + pixelBytes spec)
  let _ ← writeFile0 ptr fnWrite filenameOff bmpHeaderOff total
  ret

def payloads (spec : SceneSpec) : List UInt8 :=
  let reserved := zeros ptxOff
  let ptxBytes := padTo (stringToBytes (ptxSource spec)) ptxRegion
  let bindDesc := uint32ToBytes 0
  let bindPad := zeros (filenameOff - bindOff - bindDesc.length)
  let filenameBytes := padTo (stringToBytes spec.filename) filenameRegion
  let clifPad := zeros clifIrRegion
  reserved ++ ptxBytes ++ bindDesc ++ bindPad ++ filenameBytes ++ clifPad ++ bmpHeader spec

def config (spec : SceneSpec) : BaseConfig := {
  cranelift_ir := clifIrSource spec,
  memory_size := (payloads spec).length + pixelBytes spec,
  context_offset := 0,
  initial_memory := payloads spec
}

def algorithm : Algorithm := {
  fn_idx := IR.mainFnIdx
}

def renderScene (spec : SceneSpec) : BaseConfig × Algorithm :=
  (config spec, algorithm)

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

-- To render a different valid scene, change `main` below to:
--
--   let (cfg, alg) := Algorithm.renderScene Algorithm.previewScene
--
-- or:
--
--   let (cfg, alg) := Algorithm.renderScene Algorithm.studioScene
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

end Algorithm

def main (args : List String) : IO Unit := do
  let (cfg, alg) := Algorithm.renderScene Algorithm.defaultScene
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "scene_app" cfg alg]
