import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

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

def f32Nat (n : Nat) : String :=
  s!"{n}.0"

def f32Float (x : Float) : String :=
  toString x

def channelF32 (channel : Channel) : String :=
  f32Float (Float.ofNat channel.value / 255.0)

def colorF32 (color : Color) : String × String × String :=
  (channelF32 color.red, channelF32 color.green, channelF32 color.blue)

def invSamplesF32 (spec : SceneSpec) : String :=
  f32Float (1.0 / Float.ofNat (sampleCount spec))

def aspectF32 (spec : SceneSpec) : String :=
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

namespace Ptx

inductive Line where
  | blank
  | comment (text : String)
  | decl (text : String)
  | instr (text : String)
  | label (name : String)
  | raw (text : String)

def blank : Line := .blank
def comment (text : String) : Line := .comment text
def decl (text : String) : Line := .decl text
def instr (text : String) : Line := .instr text
def label (name : String) : Line := .label name
def raw (text : String) : Line := .raw text

structure Program (paramCount : Nat) where
  version : String
  target : String
  addressSize : Nat
  entryName : String
  params : List String
  params_length : params.length = paramCount
  body : List Line

def renderLine : Line → String
  | .blank => ""
  | .comment text => s!"    // {text}"
  | .decl text => s!"    {text}"
  | .instr text => s!"    {text}"
  | .label name => s!"{name}:"
  | .raw text => text

def renderParam (param : String) : String :=
  s!"    .param {param}"

def renderProgram {paramCount : Nat} (program : Program paramCount) : String :=
  let paramLines := program.params.map renderParam
  let header := [
    s!".version {program.version}",
    s!".target {program.target}",
    s!".address_size {program.addressSize}",
    "",
    s!".visible .entry {program.entryName}("
  ]
  let params := match paramLines with
    | [] => []
    | first :: rest =>
      let rec commaParams : List String → List String
        | [] => []
        | [last] => [last]
        | line :: more => (line ++ ",") :: commaParams more
      commaParams (first :: rest)
  String.intercalate "\n" (header ++ params ++ [")", "{"] ++ program.body.map renderLine ++ ["}", ""])

end Ptx

def ptxProgram (spec : SceneSpec) : Ptx.Program 1 := {
  version := "7.0"
  target := "sm_50"
  addressSize := 64
  entryName := "main"
  params := [".u64 out_ptr"]
  params_length := rfl
  body := [
  Ptx.decl ".reg .pred %p<64>;",
  Ptx.decl ".reg .b32 %r<192>;",
  Ptx.decl ".reg .b64 %rd<32>;",
  Ptx.decl ".reg .f32 %f<512>;",
  Ptx.blank,
  Ptx.instr "mov.u32 %r0, %ctaid.x;",
  Ptx.instr "mov.u32 %r1, %ctaid.y;",
  Ptx.instr "mov.u32 %r2, %tid.x;",
  Ptx.instr "mov.u32 %r3, %tid.y;",
  Ptx.instr "mad.lo.u32 %r4, %r0, 16, %r2;",
  Ptx.instr "mad.lo.u32 %r5, %r1, 16, %r3;",
  Ptx.instr s!"setp.ge.u32 %p0, %r4, {imageWidth spec};",
  Ptx.instr "@%p0 bra DONE;",
  Ptx.instr s!"setp.ge.u32 %p1, %r5, {imageHeight spec};",
  Ptx.instr "@%p1 bra DONE;",
  Ptx.blank,
  Ptx.instr "ld.param.u64 %rd0, [out_ptr];",
  Ptx.instr s!"mad.lo.u32 %r6, %r5, {imageWidth spec}, %r4;",
  Ptx.instr "mul.wide.u32 %rd1, %r6, 4;",
  Ptx.instr "add.s64 %rd2, %rd0, %rd1;",
  Ptx.blank,
  Ptx.comment "constants",
  Ptx.instr "mov.f32 %f400, 0f00000000;",
  Ptx.instr "mov.f32 %f401, 0f3F800000;",
  Ptx.instr "mov.f32 %f402, 0f40000000;",
  Ptx.instr "mov.f32 %f403, 0f3F000000;",
  Ptx.instr "mov.f32 %f404, 0fBF000000;",
  Ptx.instr "mov.f32 %f405, 0f437F0000;",
  Ptx.instr "mov.f32 %f406, 0f2F800000;",
  Ptx.instr "mov.f32 %f407, 0f3A83126F;",
  Ptx.instr s!"mov.f32 %f408, {f32Nat (imageWidth spec)};",
  Ptx.instr s!"mov.f32 %f409, {f32Nat (imageHeight spec)};",
  Ptx.instr "mov.f32 %f410, 16.0;",
  Ptx.instr "mov.f32 %f411, -1.6;",
  Ptx.instr "mov.f32 %f412, -1.0;",
  Ptx.instr "mov.f32 %f413, -10.0;",
  Ptx.instr "mov.f32 %f414, 0.001;",
  Ptx.instr "mov.f32 %f415, 1.0e20;",
  Ptx.instr s!"mov.f32 %f416, {invSamplesF32 spec};",
  Ptx.instr "mov.f32 %f417, 0.03;",
  Ptx.instr "mov.f32 %f418, 0.97;",
  Ptx.instr "mov.f32 %f419, 0.08;",
  Ptx.instr "mov.f32 %f420, 1.45;",
  Ptx.instr "mov.f32 %f421, 0.6896552;",
  Ptx.instr "mov.f32 %f422, 0.04;",
  Ptx.instr "mov.f32 %f423, 0.96;",
  Ptx.instr s!"mov.f32 %f424, {aspectF32 spec};",
  Ptx.blank,
  Ptx.instr "cvt.rn.f32.u32 %f0, %r4;",
  Ptx.instr "cvt.rn.f32.u32 %f1, %r5;",
  Ptx.blank,
  Ptx.instr "mov.u32 %r20, 0;",
  Ptx.instr "xor.b32 %r21, %r4, %r5;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1973, 9277;",
  Ptx.instr "xor.b32 %r21, %r21, 26699;",
  Ptx.blank,
  Ptx.instr "mov.f32 %f10, 0.0;",
  Ptx.instr "mov.f32 %f11, 0.0;",
  Ptx.instr "mov.f32 %f12, 0.0;",
  Ptx.blank,
  Ptx.label "SAMPLE_LOOP",
  Ptx.instr s!"setp.ge.u32 %p2, %r20, {sampleCount spec};",
  Ptx.instr "@%p2 bra SAMPLE_DONE;",
  Ptx.blank,
  Ptx.comment "jitter",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f20, %r21;",
  Ptx.instr "mul.f32 %f20, %f20, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f21, %r21;",
  Ptx.instr "mul.f32 %f21, %f21, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f26, %r21;",
  Ptx.instr "mul.f32 %f26, %f26, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f27, %r21;",
  Ptx.instr "mul.f32 %f27, %f27, %f406;",
  Ptx.blank,
  Ptx.instr "add.f32 %f22, %f0, %f20;",
  Ptx.instr "add.f32 %f23, %f1, %f21;",
  Ptx.instr "div.rn.f32 %f24, %f22, %f408;",
  Ptx.instr "div.rn.f32 %f25, %f23, %f409;",
  Ptx.instr "mul.f32 %f24, %f24, %f402;",
  Ptx.instr "mul.f32 %f25, %f25, %f402;",
  Ptx.instr "add.f32 %f24, %f24, -1.0;",
  Ptx.instr "sub.f32 %f25, 1.0, %f25;",
  Ptx.instr "mul.f32 %f24, %f24, %f424;",
  Ptx.blank,
  Ptx.instr "add.f32 %f28, %f26, -0.5;",
  Ptx.instr "add.f32 %f29, %f27, -0.5;",
  Ptx.instr "mul.f32 %f28, %f28, 0.07;",
  Ptx.instr "mul.f32 %f29, %f29, 0.07;",
  Ptx.blank,
  Ptx.instr "mov.f32 %f30, %f28;",
  Ptx.instr "add.f32 %f31, 1.15, %f29;",
  Ptx.instr "mov.f32 %f32, 2.7;",
  Ptx.blank,
  Ptx.instr "mul.f32 %f33, %f24, 4.8125;",
  Ptx.instr "mul.f32 %f34, %f25, 4.8125;",
  Ptx.instr "mov.f32 %f35, -5.0;",
  Ptx.instr "sub.f32 %f33, %f33, %f30;",
  Ptx.instr "add.f32 %f34, %f34, 1.15;",
  Ptx.instr "sub.f32 %f34, %f34, %f31;",
  Ptx.instr "sub.f32 %f35, %f35, %f32;",
  Ptx.instr "mul.f32 %f36, %f33, %f33;",
  Ptx.instr "fma.rn.f32 %f36, %f34, %f34, %f36;",
  Ptx.instr "fma.rn.f32 %f36, %f35, %f35, %f36;",
  Ptx.instr "rsqrt.approx.f32 %f37, %f36;",
  Ptx.instr "mul.f32 %f33, %f33, %f37;",
  Ptx.instr "mul.f32 %f34, %f34, %f37;",
  Ptx.instr "mul.f32 %f35, %f35, %f37;",
  Ptx.blank,
  Ptx.instr "mov.f32 %f40, 1.0;",
  Ptx.instr "mov.f32 %f41, 1.0;",
  Ptx.instr "mov.f32 %f42, 1.0;",
  Ptx.instr "mov.f32 %f43, 0.0;",
  Ptx.instr "mov.f32 %f44, 0.0;",
  Ptx.instr "mov.f32 %f45, 0.0;",
  Ptx.instr "mov.u32 %r30, 0;",
  Ptx.blank,
  Ptx.label "BOUNCE_LOOP",
  Ptx.instr s!"setp.ge.u32 %p3, %r30, {bounceCount spec};",
  Ptx.instr "@%p3 bra PATH_DONE;",
  Ptx.blank,
  Ptx.comment "Scene intersection",
  Ptx.instr "mov.f32 %f50, %f415;",
  Ptx.instr "mov.u32 %r31, 0;",
  Ptx.instr "mov.f32 %f51, 0.0;",
  Ptx.instr "mov.f32 %f52, 0.0;",
  Ptx.instr "mov.f32 %f53, 0.0;",
  Ptx.instr "mov.f32 %f54, 0.0;",
  Ptx.instr "mov.f32 %f55, 0.0;",
  Ptx.instr "mov.f32 %f56, 0.0;",
  Ptx.blank,
  Ptx.comment "ground plane y = -1",
  Ptx.instr "abs.f32 %f57, %f34;",
  Ptx.instr "setp.lt.f32 %p4, %f57, 1.0e-6;",
  Ptx.instr "@%p4 bra GROUND_DONE;",
  Ptx.instr "sub.f32 %f58, %f412, %f31;",
  Ptx.instr "div.rn.f32 %f59, %f58, %f34;",
  Ptx.instr "setp.le.f32 %p5, %f59, %f414;",
  Ptx.instr "@%p5 bra GROUND_DONE;",
  Ptx.instr "setp.ge.f32 %p6, %f59, %f50;",
  Ptx.instr "@%p6 bra GROUND_DONE;",
  Ptx.instr "mov.f32 %f50, %f59;",
  Ptx.instr "mov.u32 %r31, 1;",
  Ptx.instr "mov.f32 %f51, 0.0;",
  Ptx.instr "mov.f32 %f52, 1.0;",
  Ptx.instr "mov.f32 %f53, 0.0;",
  Ptx.label "GROUND_DONE",
  Ptx.blank,
  Ptx.comment "back wall z = -10",
  Ptx.instr "abs.f32 %f60, %f35;",
  Ptx.instr "setp.lt.f32 %p7, %f60, 1.0e-6;",
  Ptx.instr "@%p7 bra WALL_DONE;",
  Ptx.instr "sub.f32 %f61, %f413, %f32;",
  Ptx.instr "div.rn.f32 %f62, %f61, %f35;",
  Ptx.instr "setp.le.f32 %p8, %f62, %f414;",
  Ptx.instr "@%p8 bra WALL_DONE;",
  Ptx.instr "setp.ge.f32 %p9, %f62, %f50;",
  Ptx.instr "@%p9 bra WALL_DONE;",
  Ptx.instr "mov.f32 %f50, %f62;",
  Ptx.instr "mov.u32 %r31, 2;",
  Ptx.instr "mov.f32 %f51, 0.0;",
  Ptx.instr "mov.f32 %f52, 0.0;",
  Ptx.instr "mov.f32 %f53, 1.0;",
  Ptx.label "WALL_DONE",
  Ptx.blank,
  Ptx.comment "glass sphere (-1.35, -0.05, -4.7), r=0.95",
  Ptx.instr "add.f32 %f63, %f30, 1.35;",
  Ptx.instr "add.f32 %f64, %f31, 0.05;",
  Ptx.instr "add.f32 %f65, %f32, 4.7;",
  Ptx.instr "mul.f32 %f66, %f63, %f33;",
  Ptx.instr "fma.rn.f32 %f66, %f64, %f34, %f66;",
  Ptx.instr "fma.rn.f32 %f66, %f65, %f35, %f66;",
  Ptx.instr "mul.f32 %f67, %f63, %f63;",
  Ptx.instr "fma.rn.f32 %f67, %f64, %f64, %f67;",
  Ptx.instr "fma.rn.f32 %f67, %f65, %f65, %f67;",
  Ptx.instr "add.f32 %f67, %f67, -0.9025;",
  Ptx.instr "mul.f32 %f68, %f66, %f66;",
  Ptx.instr "sub.f32 %f68, %f68, %f67;",
  Ptx.instr "setp.lt.f32 %p10, %f68, 0.0;",
  Ptx.instr "@%p10 bra GLASS_DONE;",
  Ptx.instr "sqrt.approx.f32 %f69, %f68;",
  Ptx.instr "neg.f32 %f70, %f66;",
  Ptx.instr "sub.f32 %f71, %f70, %f69;",
  Ptx.instr "add.f32 %f72, %f70, %f69;",
  Ptx.instr "setp.gt.f32 %p11, %f71, %f414;",
  Ptx.instr "@%p11 bra GLASS_T0;",
  Ptx.instr "mov.f32 %f71, %f72;",
  Ptx.label "GLASS_T0",
  Ptx.instr "setp.le.f32 %p12, %f71, %f414;",
  Ptx.instr "@%p12 bra GLASS_DONE;",
  Ptx.instr "setp.ge.f32 %p13, %f71, %f50;",
  Ptx.instr "@%p13 bra GLASS_DONE;",
  Ptx.instr "mov.f32 %f50, %f71;",
  Ptx.instr "mov.u32 %r31, 3;",
  Ptx.instr "fma.rn.f32 %f73, %f33, %f71, %f63;",
  Ptx.instr "fma.rn.f32 %f74, %f34, %f71, %f64;",
  Ptx.instr "fma.rn.f32 %f75, %f35, %f71, %f65;",
  Ptx.instr "mul.f32 %f51, %f73, 1.0526316;",
  Ptx.instr "mul.f32 %f52, %f74, 1.0526316;",
  Ptx.instr "mul.f32 %f53, %f75, 1.0526316;",
  Ptx.label "GLASS_DONE",
  Ptx.blank,
  Ptx.comment "gold sphere (1.35, -0.25, -4.2), r=0.75",
  Ptx.instr "add.f32 %f76, %f30, -1.35;",
  Ptx.instr "add.f32 %f77, %f31, 0.25;",
  Ptx.instr "add.f32 %f78, %f32, 4.2;",
  Ptx.instr "mul.f32 %f79, %f76, %f33;",
  Ptx.instr "fma.rn.f32 %f79, %f77, %f34, %f79;",
  Ptx.instr "fma.rn.f32 %f79, %f78, %f35, %f79;",
  Ptx.instr "mul.f32 %f80, %f76, %f76;",
  Ptx.instr "fma.rn.f32 %f80, %f77, %f77, %f80;",
  Ptx.instr "fma.rn.f32 %f80, %f78, %f78, %f80;",
  Ptx.instr "add.f32 %f80, %f80, -0.5625;",
  Ptx.instr "mul.f32 %f81, %f79, %f79;",
  Ptx.instr "sub.f32 %f81, %f81, %f80;",
  Ptx.instr "setp.lt.f32 %p14, %f81, 0.0;",
  Ptx.instr "@%p14 bra GOLD_DONE;",
  Ptx.instr "sqrt.approx.f32 %f82, %f81;",
  Ptx.instr "neg.f32 %f83, %f79;",
  Ptx.instr "sub.f32 %f84, %f83, %f82;",
  Ptx.instr "add.f32 %f85, %f83, %f82;",
  Ptx.instr "setp.gt.f32 %p15, %f84, %f414;",
  Ptx.instr "@%p15 bra GOLD_T0;",
  Ptx.instr "mov.f32 %f84, %f85;",
  Ptx.label "GOLD_T0",
  Ptx.instr "setp.le.f32 %p16, %f84, %f414;",
  Ptx.instr "@%p16 bra GOLD_DONE;",
  Ptx.instr "setp.ge.f32 %p17, %f84, %f50;",
  Ptx.instr "@%p17 bra GOLD_DONE;",
  Ptx.instr "mov.f32 %f50, %f84;",
  Ptx.instr "mov.u32 %r31, 4;",
  Ptx.instr "fma.rn.f32 %f86, %f33, %f84, %f76;",
  Ptx.instr "fma.rn.f32 %f87, %f34, %f84, %f77;",
  Ptx.instr "fma.rn.f32 %f88, %f35, %f84, %f78;",
  Ptx.instr "mul.f32 %f51, %f86, 1.3333334;",
  Ptx.instr "mul.f32 %f52, %f87, 1.3333334;",
  Ptx.instr "mul.f32 %f53, %f88, 1.3333334;",
  Ptx.label "GOLD_DONE",
  Ptx.blank,
  Ptx.comment "blue sphere (0.1, 0.15, -6.5), r=1.15",
  Ptx.instr "add.f32 %f89, %f30, -0.1;",
  Ptx.instr "add.f32 %f90, %f31, -0.15;",
  Ptx.instr "add.f32 %f91, %f32, 6.5;",
  Ptx.instr "mul.f32 %f92, %f89, %f33;",
  Ptx.instr "fma.rn.f32 %f92, %f90, %f34, %f92;",
  Ptx.instr "fma.rn.f32 %f92, %f91, %f35, %f92;",
  Ptx.instr "mul.f32 %f93, %f89, %f89;",
  Ptx.instr "fma.rn.f32 %f93, %f90, %f90, %f93;",
  Ptx.instr "fma.rn.f32 %f93, %f91, %f91, %f93;",
  Ptx.instr "add.f32 %f93, %f93, -1.3225;",
  Ptx.instr "mul.f32 %f94, %f92, %f92;",
  Ptx.instr "sub.f32 %f94, %f94, %f93;",
  Ptx.instr "setp.lt.f32 %p18, %f94, 0.0;",
  Ptx.instr "@%p18 bra BLUE_DONE;",
  Ptx.instr "sqrt.approx.f32 %f95, %f94;",
  Ptx.instr "neg.f32 %f96, %f92;",
  Ptx.instr "sub.f32 %f97, %f96, %f95;",
  Ptx.instr "add.f32 %f98, %f96, %f95;",
  Ptx.instr "setp.gt.f32 %p19, %f97, %f414;",
  Ptx.instr "@%p19 bra BLUE_T0;",
  Ptx.instr "mov.f32 %f97, %f98;",
  Ptx.label "BLUE_T0",
  Ptx.instr "setp.le.f32 %p20, %f97, %f414;",
  Ptx.instr "@%p20 bra BLUE_DONE;",
  Ptx.instr "setp.ge.f32 %p21, %f97, %f50;",
  Ptx.instr "@%p21 bra BLUE_DONE;",
  Ptx.instr "mov.f32 %f50, %f97;",
  Ptx.instr "mov.u32 %r31, 5;",
  Ptx.instr "fma.rn.f32 %f99, %f33, %f97, %f89;",
  Ptx.instr "fma.rn.f32 %f100, %f34, %f97, %f90;",
  Ptx.instr "fma.rn.f32 %f101, %f35, %f97, %f91;",
  Ptx.instr "mul.f32 %f51, %f99, 0.86956525;",
  Ptx.instr "mul.f32 %f52, %f100, 0.86956525;",
  Ptx.instr "mul.f32 %f53, %f101, 0.86956525;",
  Ptx.label "BLUE_DONE",
  Ptx.blank,
  Ptx.instr "setp.eq.u32 %p22, %r31, 0;",
  Ptx.instr "@%p22 bra SHADE_SKY;",
  Ptx.blank,
  Ptx.instr "fma.rn.f32 %f102, %f33, %f50, %f30;",
  Ptx.instr "fma.rn.f32 %f103, %f34, %f50, %f31;",
  Ptx.instr "fma.rn.f32 %f104, %f35, %f50, %f32;",
  Ptx.instr "fma.rn.f32 %f105, %f51, %f414, %f102;",
  Ptx.instr "fma.rn.f32 %f106, %f52, %f414, %f103;",
  Ptx.instr "fma.rn.f32 %f107, %f53, %f414, %f104;",
  Ptx.blank,
  Ptx.comment "Fog / atmosphere",
  Ptx.instr "mul.f32 %f108, %f50, %f417;",
  Ptx.instr "min.f32 %f108, %f108, 0.35;",
  Ptx.instr "mul.f32 %f109, %f108, 0.04;",
  Ptx.instr "mul.f32 %f210, %f108, 0.07;",
  Ptx.instr "mul.f32 %f211, %f108, 0.12;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f109, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f210, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f211, %f45;",
  Ptx.blank,
  Ptx.comment "Direct lighting accumulator",
  Ptx.instr "mov.f32 %f110, 0.0;",
  Ptx.instr "mov.f32 %f111, 0.0;",
  Ptx.instr "mov.f32 %f112, 0.0;",
  Ptx.blank,
  Ptx.comment "Per-bounce light jitter for softer highlights/shadows.",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f113, %r21;",
  Ptx.instr "mul.f32 %f113, %f113, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f114, %r21;",
  Ptx.instr "mul.f32 %f114, %f114, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f115, %r21;",
  Ptx.instr "mul.f32 %f115, %f115, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f116, %r21;",
  Ptx.instr "mul.f32 %f116, %f116, %f406;",
  Ptx.instr "add.f32 %f113, %f113, -0.5;",
  Ptx.instr "add.f32 %f114, %f114, -0.5;",
  Ptx.instr "add.f32 %f115, %f115, -0.5;",
  Ptx.instr "add.f32 %f116, %f116, -0.5;",
  Ptx.blank,
  Ptx.comment "Light A at (-4, 5.5, -2), warm",
  Ptx.instr "fma.rn.f32 %f117, %f113, 1.25, -4.0;",
  Ptx.instr "fma.rn.f32 %f118, %f114, 0.35, 5.5;",
  Ptx.instr "fma.rn.f32 %f119, %f115, 1.25, -2.0;",
  Ptx.instr "sub.f32 %f116, %f117, %f105;",
  Ptx.instr "sub.f32 %f117, %f118, %f106;",
  Ptx.instr "sub.f32 %f118, %f119, %f107;",
  Ptx.instr "mul.f32 %f119, %f116, %f116;",
  Ptx.instr "fma.rn.f32 %f119, %f117, %f117, %f119;",
  Ptx.instr "fma.rn.f32 %f119, %f118, %f118, %f119;",
  Ptx.instr "rsqrt.approx.f32 %f120, %f119;",
  Ptx.instr "mul.f32 %f121, %f116, %f120;",
  Ptx.instr "mul.f32 %f122, %f117, %f120;",
  Ptx.instr "mul.f32 %f123, %f118, %f120;",
  Ptx.instr "mul.f32 %f124, %f51, %f121;",
  Ptx.instr "fma.rn.f32 %f124, %f52, %f122, %f124;",
  Ptx.instr "fma.rn.f32 %f124, %f53, %f123, %f124;",
  Ptx.instr "max.f32 %f124, %f124, 0.0;",
  Ptx.instr "setp.le.f32 %p23, %f124, 0.0;",
  Ptx.instr "@%p23 bra LIGHTA_DONE;",
  Ptx.comment "shadow against spheres only",
  Ptx.instr "mov.f32 %f125, 0.999;",
  Ptx.instr "sqrt.approx.f32 %f126, %f119;",
  Ptx.instr "mul.f32 %f126, %f126, %f125;",
  Ptx.comment "glass shadow",
  Ptx.instr "add.f32 %f127, %f105, 1.35;",
  Ptx.instr "add.f32 %f128, %f106, 0.05;",
  Ptx.instr "add.f32 %f129, %f107, 4.7;",
  Ptx.instr "mul.f32 %f130, %f127, %f121;",
  Ptx.instr "fma.rn.f32 %f130, %f128, %f122, %f130;",
  Ptx.instr "fma.rn.f32 %f130, %f129, %f123, %f130;",
  Ptx.instr "mul.f32 %f131, %f127, %f127;",
  Ptx.instr "fma.rn.f32 %f131, %f128, %f128, %f131;",
  Ptx.instr "fma.rn.f32 %f131, %f129, %f129, %f131;",
  Ptx.instr "add.f32 %f131, %f131, -0.9025;",
  Ptx.instr "mul.f32 %f132, %f130, %f130;",
  Ptx.instr "sub.f32 %f132, %f132, %f131;",
  Ptx.instr "setp.ge.f32 %p24, %f132, 0.0;",
  Ptx.instr "@!%p24 bra LIGHTA_GOLD;",
  Ptx.instr "sqrt.approx.f32 %f133, %f132;",
  Ptx.instr "neg.f32 %f134, %f130;",
  Ptx.instr "sub.f32 %f135, %f134, %f133;",
  Ptx.instr "setp.gt.f32 %p25, %f135, %f414;",
  Ptx.instr "@!%p25 bra LIGHTA_GOLD;",
  Ptx.instr "setp.lt.f32 %p26, %f135, %f126;",
  Ptx.instr "@%p26 bra LIGHTA_DONE;",
  Ptx.label "LIGHTA_GOLD",
  Ptx.instr "add.f32 %f136, %f105, -1.35;",
  Ptx.instr "add.f32 %f137, %f106, 0.25;",
  Ptx.instr "add.f32 %f138, %f107, 4.2;",
  Ptx.instr "mul.f32 %f139, %f136, %f121;",
  Ptx.instr "fma.rn.f32 %f139, %f137, %f122, %f139;",
  Ptx.instr "fma.rn.f32 %f139, %f138, %f123, %f139;",
  Ptx.instr "mul.f32 %f140, %f136, %f136;",
  Ptx.instr "fma.rn.f32 %f140, %f137, %f137, %f140;",
  Ptx.instr "fma.rn.f32 %f140, %f138, %f138, %f140;",
  Ptx.instr "add.f32 %f140, %f140, -0.5625;",
  Ptx.instr "mul.f32 %f141, %f139, %f139;",
  Ptx.instr "sub.f32 %f141, %f141, %f140;",
  Ptx.instr "setp.ge.f32 %p27, %f141, 0.0;",
  Ptx.instr "@!%p27 bra LIGHTA_BLUE;",
  Ptx.instr "sqrt.approx.f32 %f142, %f141;",
  Ptx.instr "neg.f32 %f143, %f139;",
  Ptx.instr "sub.f32 %f144, %f143, %f142;",
  Ptx.instr "setp.gt.f32 %p28, %f144, %f414;",
  Ptx.instr "@!%p28 bra LIGHTA_BLUE;",
  Ptx.instr "setp.lt.f32 %p29, %f144, %f126;",
  Ptx.instr "@%p29 bra LIGHTA_DONE;",
  Ptx.label "LIGHTA_BLUE",
  Ptx.instr "add.f32 %f145, %f105, -0.1;",
  Ptx.instr "add.f32 %f146, %f106, -0.15;",
  Ptx.instr "add.f32 %f147, %f107, 6.5;",
  Ptx.instr "mul.f32 %f148, %f145, %f121;",
  Ptx.instr "fma.rn.f32 %f148, %f146, %f122, %f148;",
  Ptx.instr "fma.rn.f32 %f148, %f147, %f123, %f148;",
  Ptx.instr "mul.f32 %f149, %f145, %f145;",
  Ptx.instr "fma.rn.f32 %f149, %f146, %f146, %f149;",
  Ptx.instr "fma.rn.f32 %f149, %f147, %f147, %f149;",
  Ptx.instr "add.f32 %f149, %f149, -1.3225;",
  Ptx.instr "mul.f32 %f150, %f148, %f148;",
  Ptx.instr "sub.f32 %f150, %f150, %f149;",
  Ptx.instr "setp.ge.f32 %p30, %f150, 0.0;",
  Ptx.instr "@!%p30 bra LIGHTA_APPLY;",
  Ptx.instr "sqrt.approx.f32 %f151, %f150;",
  Ptx.instr "neg.f32 %f152, %f148;",
  Ptx.instr "sub.f32 %f153, %f152, %f151;",
  Ptx.instr "setp.gt.f32 %p31, %f153, %f414;",
  Ptx.instr "@!%p31 bra LIGHTA_APPLY;",
  Ptx.instr "setp.lt.f32 %p32, %f153, %f126;",
  Ptx.instr "@%p32 bra LIGHTA_DONE;",
  Ptx.label "LIGHTA_APPLY",
  Ptx.instr "rcp.approx.f32 %f154, %f119;",
  Ptx.instr "mul.f32 %f155, %f124, %f154;",
  Ptx.instr "fma.rn.f32 %f110, %f155, 90.0, %f110;",
  Ptx.instr "fma.rn.f32 %f111, %f155, 74.0, %f111;",
  Ptx.instr "fma.rn.f32 %f112, %f155, 58.0, %f112;",
  Ptx.label "LIGHTA_DONE",
  Ptx.blank,
  Ptx.comment "Light B at (4.5, 4.0, -7.0), cool",
  Ptx.instr "fma.rn.f32 %f156, %f115, 1.35, 4.5;",
  Ptx.instr "fma.rn.f32 %f157, %f116, 0.45, 4.0;",
  Ptx.instr "fma.rn.f32 %f158, %f114, 1.45, -7.0;",
  Ptx.instr "sub.f32 %f159, %f156, %f105;",
  Ptx.instr "sub.f32 %f160, %f157, %f106;",
  Ptx.instr "sub.f32 %f161, %f158, %f107;",
  Ptx.instr "mul.f32 %f162, %f159, %f159;",
  Ptx.instr "fma.rn.f32 %f162, %f160, %f160, %f162;",
  Ptx.instr "fma.rn.f32 %f162, %f161, %f161, %f162;",
  Ptx.instr "rsqrt.approx.f32 %f163, %f162;",
  Ptx.instr "mul.f32 %f164, %f159, %f163;",
  Ptx.instr "mul.f32 %f165, %f160, %f163;",
  Ptx.instr "mul.f32 %f166, %f161, %f163;",
  Ptx.instr "mul.f32 %f167, %f51, %f164;",
  Ptx.instr "fma.rn.f32 %f167, %f52, %f165, %f167;",
  Ptx.instr "fma.rn.f32 %f167, %f53, %f166, %f167;",
  Ptx.instr "max.f32 %f167, %f167, 0.0;",
  Ptx.instr "setp.le.f32 %p33, %f167, 0.0;",
  Ptx.instr "@%p33 bra LIGHTB_DONE;",
  Ptx.instr "rcp.approx.f32 %f168, %f162;",
  Ptx.instr "mul.f32 %f169, %f167, %f168;",
  Ptx.instr "fma.rn.f32 %f110, %f169, 40.0, %f110;",
  Ptx.instr "fma.rn.f32 %f111, %f169, 66.0, %f111;",
  Ptx.instr "fma.rn.f32 %f112, %f169, 110.0, %f112;",
  Ptx.label "LIGHTB_DONE",
  Ptx.blank,
  Ptx.comment "Light C at (0, 6.5, 2.5), cool top fill.",
  Ptx.instr "fma.rn.f32 %f170, %f113, 2.0, 0.0;",
  Ptx.instr "fma.rn.f32 %f171, %f114, 0.6, 6.5;",
  Ptx.instr "fma.rn.f32 %f172, %f116, 1.8, 2.5;",
  Ptx.instr "sub.f32 %f173, %f170, %f105;",
  Ptx.instr "sub.f32 %f174, %f171, %f106;",
  Ptx.instr "sub.f32 %f175, %f172, %f107;",
  Ptx.instr "mul.f32 %f176, %f173, %f173;",
  Ptx.instr "fma.rn.f32 %f176, %f174, %f174, %f176;",
  Ptx.instr "fma.rn.f32 %f176, %f175, %f175, %f176;",
  Ptx.instr "rsqrt.approx.f32 %f177, %f176;",
  Ptx.instr "mul.f32 %f178, %f173, %f177;",
  Ptx.instr "mul.f32 %f179, %f174, %f177;",
  Ptx.instr "mul.f32 %f180, %f175, %f177;",
  Ptx.instr "mul.f32 %f181, %f51, %f178;",
  Ptx.instr "fma.rn.f32 %f181, %f52, %f179, %f181;",
  Ptx.instr "fma.rn.f32 %f181, %f53, %f180, %f181;",
  Ptx.instr "max.f32 %f181, %f181, 0.0;",
  Ptx.instr "setp.le.f32 %p34, %f181, 0.0;",
  Ptx.instr "@%p34 bra LIGHTC_DONE;",
  Ptx.instr "rcp.approx.f32 %f182, %f176;",
  Ptx.instr "mul.f32 %f183, %f181, %f182;",
  Ptx.instr "fma.rn.f32 %f110, %f183, 28.0, %f110;",
  Ptx.instr "fma.rn.f32 %f111, %f183, 40.0, %f111;",
  Ptx.instr "fma.rn.f32 %f112, %f183, 72.0, %f112;",
  Ptx.label "LIGHTC_DONE",
  Ptx.blank,
  Ptx.instr "setp.eq.u32 %p35, %r31, 1;",
  Ptx.instr "@%p35 bra SHADE_GROUND;",
  Ptx.instr "setp.eq.u32 %p36, %r31, 2;",
  Ptx.instr "@%p36 bra SHADE_WALL;",
  Ptx.instr "setp.eq.u32 %p37, %r31, 3;",
  Ptx.instr "@%p37 bra SHADE_GLASS;",
  Ptx.instr "setp.eq.u32 %p38, %r31, 4;",
  Ptx.instr "@%p38 bra SHADE_GOLD;",
  Ptx.instr "bra SHADE_BLUE;",
  Ptx.blank,
  Ptx.label "SHADE_GROUND",
  Ptx.instr "cvt.rzi.s32.f32 %r40, %f102;",
  Ptx.instr "cvt.rzi.s32.f32 %r41, %f104;",
  Ptx.instr "and.b32 %r42, %r40, 1;",
  Ptx.instr "and.b32 %r43, %r41, 1;",
  Ptx.instr "xor.b32 %r44, %r42, %r43;",
  Ptx.instr "setp.eq.u32 %p39, %r44, 0;",
  Ptx.instr "@%p39 bra GROUND_LIGHT;",
  Ptx.instr s!"mov.f32 %f170, {channelF32 spec.palette.groundLight.red};",
  Ptx.instr s!"mov.f32 %f171, {channelF32 spec.palette.groundLight.green};",
  Ptx.instr s!"mov.f32 %f172, {channelF32 spec.palette.groundLight.blue};",
  Ptx.instr "bra GROUND_APPLY;",
  Ptx.label "GROUND_LIGHT",
  Ptx.instr s!"mov.f32 %f170, {channelF32 spec.palette.groundDark.red};",
  Ptx.instr s!"mov.f32 %f171, {channelF32 spec.palette.groundDark.green};",
  Ptx.instr s!"mov.f32 %f172, {channelF32 spec.palette.groundDark.blue};",
  Ptx.label "GROUND_APPLY",
  Ptx.instr "mul.f32 %f212, %f110, %f170;",
  Ptx.instr "mul.f32 %f213, %f111, %f171;",
  Ptx.instr "mul.f32 %f214, %f112, %f172;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f212, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f213, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f214, %f45;",
  Ptx.instr "mul.f32 %f40, %f40, %f170;",
  Ptx.instr "mul.f32 %f41, %f41, %f171;",
  Ptx.instr "mul.f32 %f42, %f42, %f172;",
  Ptx.instr "bra DIFFUSE_BOUNCE;",
  Ptx.blank,
  Ptx.label "SHADE_WALL",
  Ptx.instr s!"mov.f32 %f173, {channelF32 spec.palette.wall.red};",
  Ptx.instr s!"mov.f32 %f174, {channelF32 spec.palette.wall.green};",
  Ptx.instr s!"mov.f32 %f175, {channelF32 spec.palette.wall.blue};",
  Ptx.instr "mul.f32 %f212, %f110, %f173;",
  Ptx.instr "mul.f32 %f213, %f111, %f174;",
  Ptx.instr "mul.f32 %f214, %f112, %f175;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f212, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f213, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f214, %f45;",
  Ptx.instr "mul.f32 %f40, %f40, %f173;",
  Ptx.instr "mul.f32 %f41, %f41, %f174;",
  Ptx.instr "mul.f32 %f42, %f42, %f175;",
  Ptx.instr "bra DIFFUSE_BOUNCE;",
  Ptx.blank,
  Ptx.label "SHADE_BLUE",
  Ptx.instr s!"mov.f32 %f176, {channelF32 spec.palette.diffuseSphere.red};",
  Ptx.instr s!"mov.f32 %f177, {channelF32 spec.palette.diffuseSphere.green};",
  Ptx.instr s!"mov.f32 %f178, {channelF32 spec.palette.diffuseSphere.blue};",
  Ptx.instr "mul.f32 %f212, %f110, %f176;",
  Ptx.instr "mul.f32 %f213, %f111, %f177;",
  Ptx.instr "mul.f32 %f214, %f112, %f178;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f212, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f213, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f214, %f45;",
  Ptx.instr "mul.f32 %f40, %f40, %f176;",
  Ptx.instr "mul.f32 %f41, %f41, %f177;",
  Ptx.instr "mul.f32 %f42, %f42, %f178;",
  Ptx.instr "bra DIFFUSE_BOUNCE;",
  Ptx.blank,
  Ptx.label "SHADE_GOLD",
  Ptx.instr s!"mov.f32 %f179, {channelF32 spec.palette.metalSphere.red};",
  Ptx.instr s!"mov.f32 %f180, {channelF32 spec.palette.metalSphere.green};",
  Ptx.instr s!"mov.f32 %f181, {channelF32 spec.palette.metalSphere.blue};",
  Ptx.instr "mul.f32 %f212, %f110, 0.15;",
  Ptx.instr "mul.f32 %f213, %f111, 0.13;",
  Ptx.instr "mul.f32 %f214, %f112, 0.09;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f212, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f213, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f214, %f45;",
  Ptx.instr "mul.f32 %f182, %f33, %f51;",
  Ptx.instr "fma.rn.f32 %f182, %f34, %f52, %f182;",
  Ptx.instr "fma.rn.f32 %f182, %f35, %f53, %f182;",
  Ptx.instr "mul.f32 %f183, %f182, 2.0;",
  Ptx.instr "neg.f32 %f226, %f51;",
  Ptx.instr "neg.f32 %f227, %f52;",
  Ptx.instr "neg.f32 %f228, %f53;",
  Ptx.instr "fma.rn.f32 %f33, %f226, %f183, %f33;",
  Ptx.instr "fma.rn.f32 %f34, %f227, %f183, %f34;",
  Ptx.instr "fma.rn.f32 %f35, %f228, %f183, %f35;",
  Ptx.comment "roughness jitter",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f184, %r21;",
  Ptx.instr "mul.f32 %f184, %f184, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f185, %r21;",
  Ptx.instr "mul.f32 %f185, %f185, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f186, %r21;",
  Ptx.instr "mul.f32 %f186, %f186, %f406;",
  Ptx.instr "add.f32 %f184, %f184, -0.5;",
  Ptx.instr "add.f32 %f185, %f185, -0.5;",
  Ptx.instr "add.f32 %f186, %f186, -0.5;",
  Ptx.instr "fma.rn.f32 %f33, %f184, %f419, %f33;",
  Ptx.instr "fma.rn.f32 %f34, %f185, %f419, %f34;",
  Ptx.instr "fma.rn.f32 %f35, %f186, %f419, %f35;",
  Ptx.instr "mul.f32 %f187, %f33, %f33;",
  Ptx.instr "fma.rn.f32 %f187, %f34, %f34, %f187;",
  Ptx.instr "fma.rn.f32 %f187, %f35, %f35, %f187;",
  Ptx.instr "rsqrt.approx.f32 %f188, %f187;",
  Ptx.instr "mul.f32 %f33, %f33, %f188;",
  Ptx.instr "mul.f32 %f34, %f34, %f188;",
  Ptx.instr "mul.f32 %f35, %f35, %f188;",
  Ptx.instr "mul.f32 %f40, %f40, %f179;",
  Ptx.instr "mul.f32 %f41, %f41, %f180;",
  Ptx.instr "mul.f32 %f42, %f42, %f181;",
  Ptx.instr "mov.f32 %f30, %f105;",
  Ptx.instr "mov.f32 %f31, %f106;",
  Ptx.instr "mov.f32 %f32, %f107;",
  Ptx.instr "bra RR_STEP;",
  Ptx.blank,
  Ptx.label "SHADE_GLASS",
  Ptx.instr "mul.f32 %f189, %f33, %f51;",
  Ptx.instr "fma.rn.f32 %f189, %f34, %f52, %f189;",
  Ptx.instr "fma.rn.f32 %f189, %f35, %f53, %f189;",
  Ptx.instr "setp.gt.f32 %p39, %f189, 0.0;",
  Ptx.instr "@%p39 bra GLASS_INSIDE;",
  Ptx.instr "mov.f32 %f190, %f51;",
  Ptx.instr "mov.f32 %f191, %f52;",
  Ptx.instr "mov.f32 %f192, %f53;",
  Ptx.instr "neg.f32 %f193, %f189;",
  Ptx.instr "mov.f32 %f194, %f421;",
  Ptx.instr "bra GLASS_COMMON;",
  Ptx.label "GLASS_INSIDE",
  Ptx.instr "neg.f32 %f190, %f51;",
  Ptx.instr "neg.f32 %f191, %f52;",
  Ptx.instr "neg.f32 %f192, %f53;",
  Ptx.instr "mov.f32 %f193, %f189;",
  Ptx.instr "mov.f32 %f194, %f420;",
  Ptx.label "GLASS_COMMON",
  Ptx.instr "sub.f32 %f195, 1.0, %f193;",
  Ptx.instr "mul.f32 %f196, %f195, %f195;",
  Ptx.instr "mul.f32 %f196, %f196, %f196;",
  Ptx.instr "mul.f32 %f196, %f196, %f195;",
  Ptx.instr "fma.rn.f32 %f197, %f196, %f423, %f422;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f198, %r21;",
  Ptx.instr "mul.f32 %f198, %f198, %f406;",
  Ptx.instr "mul.f32 %f199, %f193, %f193;",
  Ptx.instr "sub.f32 %f200, 1.0, %f199;",
  Ptx.instr "mul.f32 %f201, %f194, %f194;",
  Ptx.instr "mul.f32 %f200, %f200, %f201;",
  Ptx.instr "sub.f32 %f202, 1.0, %f200;",
  Ptx.instr "setp.lt.f32 %p40, %f202, 0.0;",
  Ptx.instr "@%p40 bra GLASS_REFLECT;",
  Ptx.instr "setp.lt.f32 %p41, %f198, %f197;",
  Ptx.instr "@%p41 bra GLASS_REFLECT;",
  Ptx.instr "sqrt.approx.f32 %f203, %f202;",
  Ptx.instr "mul.f32 %f204, %f194, %f33;",
  Ptx.instr "mul.f32 %f205, %f194, %f34;",
  Ptx.instr "mul.f32 %f206, %f194, %f35;",
  Ptx.instr "mul.f32 %f207, %f194, %f193;",
  Ptx.instr "sub.f32 %f207, %f207, %f203;",
  Ptx.instr "fma.rn.f32 %f33, %f190, %f207, %f204;",
  Ptx.instr "fma.rn.f32 %f34, %f191, %f207, %f205;",
  Ptx.instr "fma.rn.f32 %f35, %f192, %f207, %f206;",
  Ptx.instr "mul.f32 %f40, %f40, 0.98;",
  Ptx.instr "mul.f32 %f41, %f41, 0.99;",
  Ptx.instr "mul.f32 %f42, %f42, 1.0;",
  Ptx.instr "mov.f32 %f30, %f102;",
  Ptx.instr "mov.f32 %f31, %f103;",
  Ptx.instr "mov.f32 %f32, %f104;",
  Ptx.instr "bra RR_STEP;",
  Ptx.label "GLASS_REFLECT",
  Ptx.instr "mul.f32 %f208, %f33, %f190;",
  Ptx.instr "fma.rn.f32 %f208, %f34, %f191, %f208;",
  Ptx.instr "fma.rn.f32 %f208, %f35, %f192, %f208;",
  Ptx.instr "mul.f32 %f209, %f208, 2.0;",
  Ptx.instr "neg.f32 %f229, %f190;",
  Ptx.instr "neg.f32 %f230, %f191;",
  Ptx.instr "neg.f32 %f231, %f192;",
  Ptx.instr "fma.rn.f32 %f33, %f229, %f209, %f33;",
  Ptx.instr "fma.rn.f32 %f34, %f230, %f209, %f34;",
  Ptx.instr "fma.rn.f32 %f35, %f231, %f209, %f35;",
  Ptx.instr "mul.f32 %f40, %f40, 0.99;",
  Ptx.instr "mul.f32 %f41, %f41, 0.99;",
  Ptx.instr "mul.f32 %f42, %f42, 1.0;",
  Ptx.instr "mov.f32 %f30, %f105;",
  Ptx.instr "mov.f32 %f31, %f106;",
  Ptx.instr "mov.f32 %f32, %f107;",
  Ptx.instr "bra RR_STEP;",
  Ptx.blank,
  Ptx.label "DIFFUSE_BOUNCE",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f210, %r21;",
  Ptx.instr "mul.f32 %f210, %f210, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f211, %r21;",
  Ptx.instr "mul.f32 %f211, %f211, %f406;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f212, %r21;",
  Ptx.instr "mul.f32 %f212, %f212, %f406;",
  Ptx.instr "add.f32 %f210, %f210, -0.5;",
  Ptx.instr "add.f32 %f211, %f211, -0.5;",
  Ptx.instr "add.f32 %f212, %f212, -0.5;",
  Ptx.instr "fma.rn.f32 %f33, %f51, 1.4, %f210;",
  Ptx.instr "fma.rn.f32 %f34, %f52, 1.4, %f211;",
  Ptx.instr "fma.rn.f32 %f35, %f53, 1.4, %f212;",
  Ptx.instr "mul.f32 %f213, %f33, %f33;",
  Ptx.instr "fma.rn.f32 %f213, %f34, %f34, %f213;",
  Ptx.instr "fma.rn.f32 %f213, %f35, %f35, %f213;",
  Ptx.instr "rsqrt.approx.f32 %f214, %f213;",
  Ptx.instr "mul.f32 %f33, %f33, %f214;",
  Ptx.instr "mul.f32 %f34, %f34, %f214;",
  Ptx.instr "mul.f32 %f35, %f35, %f214;",
  Ptx.instr "mov.f32 %f30, %f105;",
  Ptx.instr "mov.f32 %f31, %f106;",
  Ptx.instr "mov.f32 %f32, %f107;",
  Ptx.instr "bra RR_STEP;",
  Ptx.blank,
  Ptx.label "SHADE_SKY",
  Ptx.instr "add.f32 %f215, %f34, 1.0;",
  Ptx.instr "mul.f32 %f215, %f215, 0.5;",
  Ptx.instr "mul.f32 %f216, %f215, 0.35;",
  Ptx.instr "mul.f32 %f217, %f215, 0.45;",
  Ptx.instr "mul.f32 %f218, %f215, 0.75;",
  Ptx.instr "add.f32 %f216, %f216, 0.04;",
  Ptx.instr "add.f32 %f217, %f217, 0.06;",
  Ptx.instr "add.f32 %f218, %f218, 0.12;",
  Ptx.instr "fma.rn.f32 %f43, %f40, %f216, %f43;",
  Ptx.instr "fma.rn.f32 %f44, %f41, %f217, %f44;",
  Ptx.instr "fma.rn.f32 %f45, %f42, %f218, %f45;",
  Ptx.instr "bra PATH_DONE;",
  Ptx.blank,
  Ptx.label "RR_STEP",
  Ptx.instr "setp.lt.u32 %p42, %r30, 2;",
  Ptx.instr "@%p42 bra RR_SKIP;",
  Ptx.instr "max.f32 %f216, %f40, %f41;",
  Ptx.instr "max.f32 %f216, %f216, %f42;",
  Ptx.instr "min.f32 %f216, %f216, %f418;",
  Ptx.instr "max.f32 %f216, %f216, 0.10;",
  Ptx.instr "mad.lo.u32 %r21, %r21, 1664525, 1013904223;",
  Ptx.instr "cvt.rn.f32.u32 %f217, %r21;",
  Ptx.instr "mul.f32 %f217, %f217, %f406;",
  Ptx.instr "setp.gt.f32 %p43, %f217, %f216;",
  Ptx.instr "@%p43 bra PATH_DONE;",
  Ptx.instr "rcp.approx.f32 %f218, %f216;",
  Ptx.instr "mul.f32 %f40, %f40, %f218;",
  Ptx.instr "mul.f32 %f41, %f41, %f218;",
  Ptx.instr "mul.f32 %f42, %f42, %f218;",
  Ptx.label "RR_SKIP",
  Ptx.instr "add.u32 %r30, %r30, 1;",
  Ptx.instr "bra BOUNCE_LOOP;",
  Ptx.blank,
  Ptx.label "PATH_DONE",
  Ptx.instr "add.f32 %f10, %f10, %f43;",
  Ptx.instr "add.f32 %f11, %f11, %f44;",
  Ptx.instr "add.f32 %f12, %f12, %f45;",
  Ptx.instr "add.u32 %r20, %r20, 1;",
  Ptx.instr "bra SAMPLE_LOOP;",
  Ptx.blank,
  Ptx.label "SAMPLE_DONE",
  Ptx.instr "mul.f32 %f10, %f10, %f416;",
  Ptx.instr "mul.f32 %f11, %f11, %f416;",
  Ptx.instr "mul.f32 %f12, %f12, %f416;",
  Ptx.blank,
  Ptx.instr "add.f32 %f220, %f10, 1.0;",
  Ptx.instr "add.f32 %f221, %f11, 1.0;",
  Ptx.instr "add.f32 %f222, %f12, 1.0;",
  Ptx.instr "div.rn.f32 %f10, %f10, %f220;",
  Ptx.instr "div.rn.f32 %f11, %f11, %f221;",
  Ptx.instr "div.rn.f32 %f12, %f12, %f222;",
  Ptx.instr "sqrt.approx.f32 %f10, %f10;",
  Ptx.instr "sqrt.approx.f32 %f11, %f11;",
  Ptx.instr "sqrt.approx.f32 %f12, %f12;",
  Ptx.instr "min.f32 %f10, %f10, 1.0;",
  Ptx.instr "min.f32 %f11, %f11, 1.0;",
  Ptx.instr "min.f32 %f12, %f12, 1.0;",
  Ptx.blank,
  Ptx.instr "div.rn.f32 %f223, %f0, %f408;",
  Ptx.instr "div.rn.f32 %f224, %f1, %f409;",
  Ptx.instr "sub.f32 %f223, %f223, 0.5;",
  Ptx.instr "sub.f32 %f224, %f224, 0.5;",
  Ptx.instr "mul.f32 %f225, %f223, %f223;",
  Ptx.instr "fma.rn.f32 %f225, %f224, %f224, %f225;",
  Ptx.instr "mul.f32 %f225, %f225, 1.85;",
  Ptx.instr "sub.f32 %f226, 1.0, %f225;",
  Ptx.instr "max.f32 %f226, %f226, 0.72;",
  Ptx.instr "mul.f32 %f10, %f10, %f226;",
  Ptx.instr "mul.f32 %f11, %f11, %f226;",
  Ptx.instr "mul.f32 %f12, %f12, %f226;",
  Ptx.blank,
  Ptx.instr "mul.f32 %f223, %f10, %f405;",
  Ptx.instr "mul.f32 %f224, %f11, %f405;",
  Ptx.instr "mul.f32 %f225, %f12, %f405;",
  Ptx.instr "cvt.rni.u32.f32 %r60, %f223;",
  Ptx.instr "cvt.rni.u32.f32 %r61, %f224;",
  Ptx.instr "cvt.rni.u32.f32 %r62, %f225;",
  Ptx.instr "shl.b32 %r61, %r61, 8;",
  Ptx.instr "shl.b32 %r60, %r60, 16;",
  Ptx.instr "mov.u32 %r63, 0xFF000000;",
  Ptx.instr "or.b32 %r64, %r62, %r61;",
  Ptx.instr "or.b32 %r65, %r64, %r60;",
  Ptx.instr "or.b32 %r66, %r65, %r63;",
  Ptx.instr "st.global.u32 [%rd2], %r66;",
  Ptx.blank,
  Ptx.label "DONE",
  Ptx.instr "ret;"
  ]
}

def ptxSource (spec : SceneSpec) : String :=
  Ptx.renderProgram (ptxProgram spec)


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
  callVoid cuda.fnInit [ptr]
  let dataSz ← iconst64 (pixelBytes spec)
  let bufId ← call cuda.fnCreateBuffer [ptr, dataSz]
  let ptxOffV ← iconst64 ptxOff
  let nBufs ← iconst32 1
  let bindOffV ← iconst64 bindOff
  let gridX ← iconst32 ((imageWidth spec + 15) / 16)
  let gridY ← iconst32 ((imageHeight spec + 15) / 16)
  let one32 ← iconst32 1
  let blk16 ← iconst32 16
  let _ := bufId
  let _ ← call cuda.fnLaunch
    [ptr, ptxOffV, nBufs, bindOffV, gridX, gridY, one32, blk16, blk16, one32]
  let pxOffV ← iconst64 pixelsOff
  let _ ← call cuda.fnDownload [ptr, bufId, pxOffV, dataSz]
  callVoid cuda.fnCleanup [ptr]
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
  actions := [IR.clifCallAction],
  cranelift_units := 0,
  timeout_ms := some 300000
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

def main : IO Unit := do
  let (cfg, alg) := Algorithm.renderScene Algorithm.defaultScene
  let json := toJsonPair cfg alg
  IO.println (Json.compress json)
