import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Algorithm

def imageWidth : Nat := 1280
def imageHeight : Nat := 720

def pixelCount : Nat := imageWidth * imageHeight
def pixelBytes : Nat := pixelCount * 4

def bmpHeader : List UInt8 :=
  let fileSize : Nat := 54 + pixelBytes
  let bfType := [0x42, 0x4D]
  let bfSize := uint32ToBytes (UInt32.ofNat fileSize)
  let bfReserved := [0, 0, 0, 0]
  let bfOffBits := uint32ToBytes 54
  let biSize := uint32ToBytes 40
  let biWidth := uint32ToBytes (UInt32.ofNat imageWidth)
  let biHeight := int32ToBytes (Int.negSucc (imageHeight - 1))
  let biPlanes := [1, 0]
  let biBitCount := [32, 0]
  let biCompression := uint32ToBytes 0
  let biSizeImage := uint32ToBytes (UInt32.ofNat pixelBytes)
  let biXPelsPerMeter := uint32ToBytes 2835
  let biYPelsPerMeter := uint32ToBytes 2835
  let biClrUsed := uint32ToBytes 0
  let biClrImportant := uint32ToBytes 0
  bfType ++ bfSize ++ bfReserved ++ bfOffBits ++
  biSize ++ biWidth ++ biHeight ++ biPlanes ++ biBitCount ++
  biCompression ++ biSizeImage ++ biXPelsPerMeter ++ biYPelsPerMeter ++
  biClrUsed ++ biClrImportant

def ptxSource : String := r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
    .param .u64 out_ptr
)
{
    .reg .pred %p<64>;
    .reg .b32 %r<192>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<512>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ctaid.y;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %tid.y;
    mad.lo.u32 %r4, %r0, 16, %r2;
    mad.lo.u32 %r5, %r1, 16, %r3;
    setp.ge.u32 %p0, %r4, 1280;
    @%p0 bra DONE;
    setp.ge.u32 %p1, %r5, 720;
    @%p1 bra DONE;

    ld.param.u64 %rd0, [out_ptr];
    mad.lo.u32 %r6, %r5, 1280, %r4;
    mul.wide.u32 %rd1, %r6, 4;
    add.s64 %rd2, %rd0, %rd1;

    // constants
    mov.f32 %f400, 0f00000000;
    mov.f32 %f401, 0f3F800000;
    mov.f32 %f402, 0f40000000;
    mov.f32 %f403, 0f3F000000;
    mov.f32 %f404, 0fBF000000;
    mov.f32 %f405, 0f437F0000;
    mov.f32 %f406, 0f2F800000;
    mov.f32 %f407, 0f3A83126F;
    mov.f32 %f408, 1280.0;
    mov.f32 %f409, 720.0;
    mov.f32 %f410, 16.0;
    mov.f32 %f411, -1.6;
    mov.f32 %f412, -1.0;
    mov.f32 %f413, -10.0;
    mov.f32 %f414, 0.001;
    mov.f32 %f415, 1.0e20;
    mov.f32 %f416, 0.0078125;
    mov.f32 %f417, 0.03;
    mov.f32 %f418, 0.97;
    mov.f32 %f419, 0.08;
    mov.f32 %f420, 1.45;
    mov.f32 %f421, 0.6896552;
    mov.f32 %f422, 0.04;
    mov.f32 %f423, 0.96;
    mov.f32 %f424, 1.7777778;

    cvt.rn.f32.u32 %f0, %r4;
    cvt.rn.f32.u32 %f1, %r5;

    mov.u32 %r20, 0;
    xor.b32 %r21, %r4, %r5;
    mad.lo.u32 %r21, %r21, 1973, 9277;
    xor.b32 %r21, %r21, 26699;

    mov.f32 %f10, 0.0;
    mov.f32 %f11, 0.0;
    mov.f32 %f12, 0.0;

SAMPLE_LOOP:
    setp.ge.u32 %p2, %r20, 128;
    @%p2 bra SAMPLE_DONE;

    // jitter
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f20, %r21;
    mul.f32 %f20, %f20, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f21, %r21;
    mul.f32 %f21, %f21, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f26, %r21;
    mul.f32 %f26, %f26, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f27, %r21;
    mul.f32 %f27, %f27, %f406;

    add.f32 %f22, %f0, %f20;
    add.f32 %f23, %f1, %f21;
    div.rn.f32 %f24, %f22, %f408;
    div.rn.f32 %f25, %f23, %f409;
    mul.f32 %f24, %f24, %f402;
    mul.f32 %f25, %f25, %f402;
    add.f32 %f24, %f24, -1.0;
    sub.f32 %f25, 1.0, %f25;
    mul.f32 %f24, %f24, %f424;

    add.f32 %f28, %f26, -0.5;
    add.f32 %f29, %f27, -0.5;
    mul.f32 %f28, %f28, 0.07;
    mul.f32 %f29, %f29, 0.07;

    mov.f32 %f30, %f28;
    add.f32 %f31, 1.15, %f29;
    mov.f32 %f32, 2.7;

    mul.f32 %f33, %f24, 4.8125;
    mul.f32 %f34, %f25, 4.8125;
    mov.f32 %f35, -5.0;
    sub.f32 %f33, %f33, %f30;
    add.f32 %f34, %f34, 1.15;
    sub.f32 %f34, %f34, %f31;
    sub.f32 %f35, %f35, %f32;
    mul.f32 %f36, %f33, %f33;
    fma.rn.f32 %f36, %f34, %f34, %f36;
    fma.rn.f32 %f36, %f35, %f35, %f36;
    rsqrt.approx.f32 %f37, %f36;
    mul.f32 %f33, %f33, %f37;
    mul.f32 %f34, %f34, %f37;
    mul.f32 %f35, %f35, %f37;

    mov.f32 %f40, 1.0;
    mov.f32 %f41, 1.0;
    mov.f32 %f42, 1.0;
    mov.f32 %f43, 0.0;
    mov.f32 %f44, 0.0;
    mov.f32 %f45, 0.0;
    mov.u32 %r30, 0;

BOUNCE_LOOP:
    setp.ge.u32 %p3, %r30, 5;
    @%p3 bra PATH_DONE;

    // Scene intersection
    mov.f32 %f50, %f415;
    mov.u32 %r31, 0;
    mov.f32 %f51, 0.0;
    mov.f32 %f52, 0.0;
    mov.f32 %f53, 0.0;
    mov.f32 %f54, 0.0;
    mov.f32 %f55, 0.0;
    mov.f32 %f56, 0.0;

    // ground plane y = -1
    abs.f32 %f57, %f34;
    setp.lt.f32 %p4, %f57, 1.0e-6;
    @%p4 bra GROUND_DONE;
    sub.f32 %f58, %f412, %f31;
    div.rn.f32 %f59, %f58, %f34;
    setp.le.f32 %p5, %f59, %f414;
    @%p5 bra GROUND_DONE;
    setp.ge.f32 %p6, %f59, %f50;
    @%p6 bra GROUND_DONE;
    mov.f32 %f50, %f59;
    mov.u32 %r31, 1;
    mov.f32 %f51, 0.0;
    mov.f32 %f52, 1.0;
    mov.f32 %f53, 0.0;
GROUND_DONE:

    // back wall z = -10
    abs.f32 %f60, %f35;
    setp.lt.f32 %p7, %f60, 1.0e-6;
    @%p7 bra WALL_DONE;
    sub.f32 %f61, %f413, %f32;
    div.rn.f32 %f62, %f61, %f35;
    setp.le.f32 %p8, %f62, %f414;
    @%p8 bra WALL_DONE;
    setp.ge.f32 %p9, %f62, %f50;
    @%p9 bra WALL_DONE;
    mov.f32 %f50, %f62;
    mov.u32 %r31, 2;
    mov.f32 %f51, 0.0;
    mov.f32 %f52, 0.0;
    mov.f32 %f53, 1.0;
WALL_DONE:

    // glass sphere (-1.35, -0.05, -4.7), r=0.95
    add.f32 %f63, %f30, 1.35;
    add.f32 %f64, %f31, 0.05;
    add.f32 %f65, %f32, 4.7;
    mul.f32 %f66, %f63, %f33;
    fma.rn.f32 %f66, %f64, %f34, %f66;
    fma.rn.f32 %f66, %f65, %f35, %f66;
    mul.f32 %f67, %f63, %f63;
    fma.rn.f32 %f67, %f64, %f64, %f67;
    fma.rn.f32 %f67, %f65, %f65, %f67;
    add.f32 %f67, %f67, -0.9025;
    mul.f32 %f68, %f66, %f66;
    sub.f32 %f68, %f68, %f67;
    setp.lt.f32 %p10, %f68, 0.0;
    @%p10 bra GLASS_DONE;
    sqrt.approx.f32 %f69, %f68;
    neg.f32 %f70, %f66;
    sub.f32 %f71, %f70, %f69;
    add.f32 %f72, %f70, %f69;
    setp.gt.f32 %p11, %f71, %f414;
    @%p11 bra GLASS_T0;
    mov.f32 %f71, %f72;
GLASS_T0:
    setp.le.f32 %p12, %f71, %f414;
    @%p12 bra GLASS_DONE;
    setp.ge.f32 %p13, %f71, %f50;
    @%p13 bra GLASS_DONE;
    mov.f32 %f50, %f71;
    mov.u32 %r31, 3;
    fma.rn.f32 %f73, %f33, %f71, %f63;
    fma.rn.f32 %f74, %f34, %f71, %f64;
    fma.rn.f32 %f75, %f35, %f71, %f65;
    mul.f32 %f51, %f73, 1.0526316;
    mul.f32 %f52, %f74, 1.0526316;
    mul.f32 %f53, %f75, 1.0526316;
GLASS_DONE:

    // gold sphere (1.35, -0.25, -4.2), r=0.75
    add.f32 %f76, %f30, -1.35;
    add.f32 %f77, %f31, 0.25;
    add.f32 %f78, %f32, 4.2;
    mul.f32 %f79, %f76, %f33;
    fma.rn.f32 %f79, %f77, %f34, %f79;
    fma.rn.f32 %f79, %f78, %f35, %f79;
    mul.f32 %f80, %f76, %f76;
    fma.rn.f32 %f80, %f77, %f77, %f80;
    fma.rn.f32 %f80, %f78, %f78, %f80;
    add.f32 %f80, %f80, -0.5625;
    mul.f32 %f81, %f79, %f79;
    sub.f32 %f81, %f81, %f80;
    setp.lt.f32 %p14, %f81, 0.0;
    @%p14 bra GOLD_DONE;
    sqrt.approx.f32 %f82, %f81;
    neg.f32 %f83, %f79;
    sub.f32 %f84, %f83, %f82;
    add.f32 %f85, %f83, %f82;
    setp.gt.f32 %p15, %f84, %f414;
    @%p15 bra GOLD_T0;
    mov.f32 %f84, %f85;
GOLD_T0:
    setp.le.f32 %p16, %f84, %f414;
    @%p16 bra GOLD_DONE;
    setp.ge.f32 %p17, %f84, %f50;
    @%p17 bra GOLD_DONE;
    mov.f32 %f50, %f84;
    mov.u32 %r31, 4;
    fma.rn.f32 %f86, %f33, %f84, %f76;
    fma.rn.f32 %f87, %f34, %f84, %f77;
    fma.rn.f32 %f88, %f35, %f84, %f78;
    mul.f32 %f51, %f86, 1.3333334;
    mul.f32 %f52, %f87, 1.3333334;
    mul.f32 %f53, %f88, 1.3333334;
GOLD_DONE:

    // blue sphere (0.1, 0.15, -6.5), r=1.15
    add.f32 %f89, %f30, -0.1;
    add.f32 %f90, %f31, -0.15;
    add.f32 %f91, %f32, 6.5;
    mul.f32 %f92, %f89, %f33;
    fma.rn.f32 %f92, %f90, %f34, %f92;
    fma.rn.f32 %f92, %f91, %f35, %f92;
    mul.f32 %f93, %f89, %f89;
    fma.rn.f32 %f93, %f90, %f90, %f93;
    fma.rn.f32 %f93, %f91, %f91, %f93;
    add.f32 %f93, %f93, -1.3225;
    mul.f32 %f94, %f92, %f92;
    sub.f32 %f94, %f94, %f93;
    setp.lt.f32 %p18, %f94, 0.0;
    @%p18 bra BLUE_DONE;
    sqrt.approx.f32 %f95, %f94;
    neg.f32 %f96, %f92;
    sub.f32 %f97, %f96, %f95;
    add.f32 %f98, %f96, %f95;
    setp.gt.f32 %p19, %f97, %f414;
    @%p19 bra BLUE_T0;
    mov.f32 %f97, %f98;
BLUE_T0:
    setp.le.f32 %p20, %f97, %f414;
    @%p20 bra BLUE_DONE;
    setp.ge.f32 %p21, %f97, %f50;
    @%p21 bra BLUE_DONE;
    mov.f32 %f50, %f97;
    mov.u32 %r31, 5;
    fma.rn.f32 %f99, %f33, %f97, %f89;
    fma.rn.f32 %f100, %f34, %f97, %f90;
    fma.rn.f32 %f101, %f35, %f97, %f91;
    mul.f32 %f51, %f99, 0.86956525;
    mul.f32 %f52, %f100, 0.86956525;
    mul.f32 %f53, %f101, 0.86956525;
BLUE_DONE:

    setp.eq.u32 %p22, %r31, 0;
    @%p22 bra SHADE_SKY;

    fma.rn.f32 %f102, %f33, %f50, %f30;
    fma.rn.f32 %f103, %f34, %f50, %f31;
    fma.rn.f32 %f104, %f35, %f50, %f32;
    fma.rn.f32 %f105, %f51, %f414, %f102;
    fma.rn.f32 %f106, %f52, %f414, %f103;
    fma.rn.f32 %f107, %f53, %f414, %f104;

    // Fog / atmosphere
    mul.f32 %f108, %f50, %f417;
    min.f32 %f108, %f108, 0.35;
    mul.f32 %f109, %f108, 0.04;
    mul.f32 %f210, %f108, 0.07;
    mul.f32 %f211, %f108, 0.12;
    fma.rn.f32 %f43, %f40, %f109, %f43;
    fma.rn.f32 %f44, %f41, %f210, %f44;
    fma.rn.f32 %f45, %f42, %f211, %f45;

    // Direct lighting accumulator
    mov.f32 %f110, 0.0;
    mov.f32 %f111, 0.0;
    mov.f32 %f112, 0.0;

    // Per-bounce light jitter for softer highlights/shadows.
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f113, %r21;
    mul.f32 %f113, %f113, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f114, %r21;
    mul.f32 %f114, %f114, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f115, %r21;
    mul.f32 %f115, %f115, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f116, %r21;
    mul.f32 %f116, %f116, %f406;
    add.f32 %f113, %f113, -0.5;
    add.f32 %f114, %f114, -0.5;
    add.f32 %f115, %f115, -0.5;
    add.f32 %f116, %f116, -0.5;

    // Light A at (-4, 5.5, -2), warm
    fma.rn.f32 %f117, %f113, 1.25, -4.0;
    fma.rn.f32 %f118, %f114, 0.35, 5.5;
    fma.rn.f32 %f119, %f115, 1.25, -2.0;
    sub.f32 %f116, %f117, %f105;
    sub.f32 %f117, %f118, %f106;
    sub.f32 %f118, %f119, %f107;
    mul.f32 %f119, %f116, %f116;
    fma.rn.f32 %f119, %f117, %f117, %f119;
    fma.rn.f32 %f119, %f118, %f118, %f119;
    rsqrt.approx.f32 %f120, %f119;
    mul.f32 %f121, %f116, %f120;
    mul.f32 %f122, %f117, %f120;
    mul.f32 %f123, %f118, %f120;
    mul.f32 %f124, %f51, %f121;
    fma.rn.f32 %f124, %f52, %f122, %f124;
    fma.rn.f32 %f124, %f53, %f123, %f124;
    max.f32 %f124, %f124, 0.0;
    setp.le.f32 %p23, %f124, 0.0;
    @%p23 bra LIGHTA_DONE;
    // shadow against spheres only
    mov.f32 %f125, 0.999;
    sqrt.approx.f32 %f126, %f119;
    mul.f32 %f126, %f126, %f125;
    // glass shadow
    add.f32 %f127, %f105, 1.35;
    add.f32 %f128, %f106, 0.05;
    add.f32 %f129, %f107, 4.7;
    mul.f32 %f130, %f127, %f121;
    fma.rn.f32 %f130, %f128, %f122, %f130;
    fma.rn.f32 %f130, %f129, %f123, %f130;
    mul.f32 %f131, %f127, %f127;
    fma.rn.f32 %f131, %f128, %f128, %f131;
    fma.rn.f32 %f131, %f129, %f129, %f131;
    add.f32 %f131, %f131, -0.9025;
    mul.f32 %f132, %f130, %f130;
    sub.f32 %f132, %f132, %f131;
    setp.ge.f32 %p24, %f132, 0.0;
    @!%p24 bra LIGHTA_GOLD;
    sqrt.approx.f32 %f133, %f132;
    neg.f32 %f134, %f130;
    sub.f32 %f135, %f134, %f133;
    setp.gt.f32 %p25, %f135, %f414;
    @!%p25 bra LIGHTA_GOLD;
    setp.lt.f32 %p26, %f135, %f126;
    @%p26 bra LIGHTA_DONE;
LIGHTA_GOLD:
    add.f32 %f136, %f105, -1.35;
    add.f32 %f137, %f106, 0.25;
    add.f32 %f138, %f107, 4.2;
    mul.f32 %f139, %f136, %f121;
    fma.rn.f32 %f139, %f137, %f122, %f139;
    fma.rn.f32 %f139, %f138, %f123, %f139;
    mul.f32 %f140, %f136, %f136;
    fma.rn.f32 %f140, %f137, %f137, %f140;
    fma.rn.f32 %f140, %f138, %f138, %f140;
    add.f32 %f140, %f140, -0.5625;
    mul.f32 %f141, %f139, %f139;
    sub.f32 %f141, %f141, %f140;
    setp.ge.f32 %p27, %f141, 0.0;
    @!%p27 bra LIGHTA_BLUE;
    sqrt.approx.f32 %f142, %f141;
    neg.f32 %f143, %f139;
    sub.f32 %f144, %f143, %f142;
    setp.gt.f32 %p28, %f144, %f414;
    @!%p28 bra LIGHTA_BLUE;
    setp.lt.f32 %p29, %f144, %f126;
    @%p29 bra LIGHTA_DONE;
LIGHTA_BLUE:
    add.f32 %f145, %f105, -0.1;
    add.f32 %f146, %f106, -0.15;
    add.f32 %f147, %f107, 6.5;
    mul.f32 %f148, %f145, %f121;
    fma.rn.f32 %f148, %f146, %f122, %f148;
    fma.rn.f32 %f148, %f147, %f123, %f148;
    mul.f32 %f149, %f145, %f145;
    fma.rn.f32 %f149, %f146, %f146, %f149;
    fma.rn.f32 %f149, %f147, %f147, %f149;
    add.f32 %f149, %f149, -1.3225;
    mul.f32 %f150, %f148, %f148;
    sub.f32 %f150, %f150, %f149;
    setp.ge.f32 %p30, %f150, 0.0;
    @!%p30 bra LIGHTA_APPLY;
    sqrt.approx.f32 %f151, %f150;
    neg.f32 %f152, %f148;
    sub.f32 %f153, %f152, %f151;
    setp.gt.f32 %p31, %f153, %f414;
    @!%p31 bra LIGHTA_APPLY;
    setp.lt.f32 %p32, %f153, %f126;
    @%p32 bra LIGHTA_DONE;
LIGHTA_APPLY:
    rcp.approx.f32 %f154, %f119;
    mul.f32 %f155, %f124, %f154;
    fma.rn.f32 %f110, %f155, 90.0, %f110;
    fma.rn.f32 %f111, %f155, 74.0, %f111;
    fma.rn.f32 %f112, %f155, 58.0, %f112;
LIGHTA_DONE:

    // Light B at (4.5, 4.0, -7.0), cool
    fma.rn.f32 %f156, %f115, 1.35, 4.5;
    fma.rn.f32 %f157, %f116, 0.45, 4.0;
    fma.rn.f32 %f158, %f114, 1.45, -7.0;
    sub.f32 %f159, %f156, %f105;
    sub.f32 %f160, %f157, %f106;
    sub.f32 %f161, %f158, %f107;
    mul.f32 %f162, %f159, %f159;
    fma.rn.f32 %f162, %f160, %f160, %f162;
    fma.rn.f32 %f162, %f161, %f161, %f162;
    rsqrt.approx.f32 %f163, %f162;
    mul.f32 %f164, %f159, %f163;
    mul.f32 %f165, %f160, %f163;
    mul.f32 %f166, %f161, %f163;
    mul.f32 %f167, %f51, %f164;
    fma.rn.f32 %f167, %f52, %f165, %f167;
    fma.rn.f32 %f167, %f53, %f166, %f167;
    max.f32 %f167, %f167, 0.0;
    setp.le.f32 %p33, %f167, 0.0;
    @%p33 bra LIGHTB_DONE;
    rcp.approx.f32 %f168, %f162;
    mul.f32 %f169, %f167, %f168;
    fma.rn.f32 %f110, %f169, 40.0, %f110;
    fma.rn.f32 %f111, %f169, 66.0, %f111;
    fma.rn.f32 %f112, %f169, 110.0, %f112;
LIGHTB_DONE:

    // Light C at (0, 6.5, 2.5), cool top fill.
    fma.rn.f32 %f170, %f113, 2.0, 0.0;
    fma.rn.f32 %f171, %f114, 0.6, 6.5;
    fma.rn.f32 %f172, %f116, 1.8, 2.5;
    sub.f32 %f173, %f170, %f105;
    sub.f32 %f174, %f171, %f106;
    sub.f32 %f175, %f172, %f107;
    mul.f32 %f176, %f173, %f173;
    fma.rn.f32 %f176, %f174, %f174, %f176;
    fma.rn.f32 %f176, %f175, %f175, %f176;
    rsqrt.approx.f32 %f177, %f176;
    mul.f32 %f178, %f173, %f177;
    mul.f32 %f179, %f174, %f177;
    mul.f32 %f180, %f175, %f177;
    mul.f32 %f181, %f51, %f178;
    fma.rn.f32 %f181, %f52, %f179, %f181;
    fma.rn.f32 %f181, %f53, %f180, %f181;
    max.f32 %f181, %f181, 0.0;
    setp.le.f32 %p34, %f181, 0.0;
    @%p34 bra LIGHTC_DONE;
    rcp.approx.f32 %f182, %f176;
    mul.f32 %f183, %f181, %f182;
    fma.rn.f32 %f110, %f183, 28.0, %f110;
    fma.rn.f32 %f111, %f183, 40.0, %f111;
    fma.rn.f32 %f112, %f183, 72.0, %f112;
LIGHTC_DONE:

    setp.eq.u32 %p35, %r31, 1;
    @%p35 bra SHADE_GROUND;
    setp.eq.u32 %p36, %r31, 2;
    @%p36 bra SHADE_WALL;
    setp.eq.u32 %p37, %r31, 3;
    @%p37 bra SHADE_GLASS;
    setp.eq.u32 %p38, %r31, 4;
    @%p38 bra SHADE_GOLD;
    bra SHADE_BLUE;

SHADE_GROUND:
    cvt.rzi.s32.f32 %r40, %f102;
    cvt.rzi.s32.f32 %r41, %f104;
    and.b32 %r42, %r40, 1;
    and.b32 %r43, %r41, 1;
    xor.b32 %r44, %r42, %r43;
    setp.eq.u32 %p39, %r44, 0;
    @%p39 bra GROUND_LIGHT;
    mov.f32 %f170, 0.74;
    mov.f32 %f171, 0.75;
    mov.f32 %f172, 0.80;
    bra GROUND_APPLY;
GROUND_LIGHT:
    mov.f32 %f170, 0.10;
    mov.f32 %f171, 0.11;
    mov.f32 %f172, 0.13;
GROUND_APPLY:
    mul.f32 %f212, %f110, %f170;
    mul.f32 %f213, %f111, %f171;
    mul.f32 %f214, %f112, %f172;
    fma.rn.f32 %f43, %f40, %f212, %f43;
    fma.rn.f32 %f44, %f41, %f213, %f44;
    fma.rn.f32 %f45, %f42, %f214, %f45;
    mul.f32 %f40, %f40, %f170;
    mul.f32 %f41, %f41, %f171;
    mul.f32 %f42, %f42, %f172;
    bra DIFFUSE_BOUNCE;

SHADE_WALL:
    mov.f32 %f173, 0.46;
    mov.f32 %f174, 0.34;
    mov.f32 %f175, 0.30;
    mul.f32 %f212, %f110, %f173;
    mul.f32 %f213, %f111, %f174;
    mul.f32 %f214, %f112, %f175;
    fma.rn.f32 %f43, %f40, %f212, %f43;
    fma.rn.f32 %f44, %f41, %f213, %f44;
    fma.rn.f32 %f45, %f42, %f214, %f45;
    mul.f32 %f40, %f40, %f173;
    mul.f32 %f41, %f41, %f174;
    mul.f32 %f42, %f42, %f175;
    bra DIFFUSE_BOUNCE;

SHADE_BLUE:
    mov.f32 %f176, 0.10;
    mov.f32 %f177, 0.22;
    mov.f32 %f178, 0.78;
    mul.f32 %f212, %f110, %f176;
    mul.f32 %f213, %f111, %f177;
    mul.f32 %f214, %f112, %f178;
    fma.rn.f32 %f43, %f40, %f212, %f43;
    fma.rn.f32 %f44, %f41, %f213, %f44;
    fma.rn.f32 %f45, %f42, %f214, %f45;
    mul.f32 %f40, %f40, %f176;
    mul.f32 %f41, %f41, %f177;
    mul.f32 %f42, %f42, %f178;
    bra DIFFUSE_BOUNCE;

SHADE_GOLD:
    mov.f32 %f179, 0.96;
    mov.f32 %f180, 0.78;
    mov.f32 %f181, 0.32;
    mul.f32 %f212, %f110, 0.15;
    mul.f32 %f213, %f111, 0.13;
    mul.f32 %f214, %f112, 0.09;
    fma.rn.f32 %f43, %f40, %f212, %f43;
    fma.rn.f32 %f44, %f41, %f213, %f44;
    fma.rn.f32 %f45, %f42, %f214, %f45;
    mul.f32 %f182, %f33, %f51;
    fma.rn.f32 %f182, %f34, %f52, %f182;
    fma.rn.f32 %f182, %f35, %f53, %f182;
    mul.f32 %f183, %f182, 2.0;
    neg.f32 %f226, %f51;
    neg.f32 %f227, %f52;
    neg.f32 %f228, %f53;
    fma.rn.f32 %f33, %f226, %f183, %f33;
    fma.rn.f32 %f34, %f227, %f183, %f34;
    fma.rn.f32 %f35, %f228, %f183, %f35;
    // roughness jitter
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f184, %r21;
    mul.f32 %f184, %f184, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f185, %r21;
    mul.f32 %f185, %f185, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f186, %r21;
    mul.f32 %f186, %f186, %f406;
    add.f32 %f184, %f184, -0.5;
    add.f32 %f185, %f185, -0.5;
    add.f32 %f186, %f186, -0.5;
    fma.rn.f32 %f33, %f184, %f419, %f33;
    fma.rn.f32 %f34, %f185, %f419, %f34;
    fma.rn.f32 %f35, %f186, %f419, %f35;
    mul.f32 %f187, %f33, %f33;
    fma.rn.f32 %f187, %f34, %f34, %f187;
    fma.rn.f32 %f187, %f35, %f35, %f187;
    rsqrt.approx.f32 %f188, %f187;
    mul.f32 %f33, %f33, %f188;
    mul.f32 %f34, %f34, %f188;
    mul.f32 %f35, %f35, %f188;
    mul.f32 %f40, %f40, %f179;
    mul.f32 %f41, %f41, %f180;
    mul.f32 %f42, %f42, %f181;
    mov.f32 %f30, %f105;
    mov.f32 %f31, %f106;
    mov.f32 %f32, %f107;
    bra RR_STEP;

SHADE_GLASS:
    mul.f32 %f189, %f33, %f51;
    fma.rn.f32 %f189, %f34, %f52, %f189;
    fma.rn.f32 %f189, %f35, %f53, %f189;
    setp.gt.f32 %p39, %f189, 0.0;
    @%p39 bra GLASS_INSIDE;
    mov.f32 %f190, %f51;
    mov.f32 %f191, %f52;
    mov.f32 %f192, %f53;
    neg.f32 %f193, %f189;
    mov.f32 %f194, %f421;
    bra GLASS_COMMON;
GLASS_INSIDE:
    neg.f32 %f190, %f51;
    neg.f32 %f191, %f52;
    neg.f32 %f192, %f53;
    mov.f32 %f193, %f189;
    mov.f32 %f194, %f420;
GLASS_COMMON:
    sub.f32 %f195, 1.0, %f193;
    mul.f32 %f196, %f195, %f195;
    mul.f32 %f196, %f196, %f196;
    mul.f32 %f196, %f196, %f195;
    fma.rn.f32 %f197, %f196, %f423, %f422;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f198, %r21;
    mul.f32 %f198, %f198, %f406;
    mul.f32 %f199, %f193, %f193;
    sub.f32 %f200, 1.0, %f199;
    mul.f32 %f201, %f194, %f194;
    mul.f32 %f200, %f200, %f201;
    sub.f32 %f202, 1.0, %f200;
    setp.lt.f32 %p40, %f202, 0.0;
    @%p40 bra GLASS_REFLECT;
    setp.lt.f32 %p41, %f198, %f197;
    @%p41 bra GLASS_REFLECT;
    sqrt.approx.f32 %f203, %f202;
    mul.f32 %f204, %f194, %f33;
    mul.f32 %f205, %f194, %f34;
    mul.f32 %f206, %f194, %f35;
    mul.f32 %f207, %f194, %f193;
    sub.f32 %f207, %f207, %f203;
    fma.rn.f32 %f33, %f190, %f207, %f204;
    fma.rn.f32 %f34, %f191, %f207, %f205;
    fma.rn.f32 %f35, %f192, %f207, %f206;
    mul.f32 %f40, %f40, 0.98;
    mul.f32 %f41, %f41, 0.99;
    mul.f32 %f42, %f42, 1.0;
    mov.f32 %f30, %f102;
    mov.f32 %f31, %f103;
    mov.f32 %f32, %f104;
    bra RR_STEP;
GLASS_REFLECT:
    mul.f32 %f208, %f33, %f190;
    fma.rn.f32 %f208, %f34, %f191, %f208;
    fma.rn.f32 %f208, %f35, %f192, %f208;
    mul.f32 %f209, %f208, 2.0;
    neg.f32 %f229, %f190;
    neg.f32 %f230, %f191;
    neg.f32 %f231, %f192;
    fma.rn.f32 %f33, %f229, %f209, %f33;
    fma.rn.f32 %f34, %f230, %f209, %f34;
    fma.rn.f32 %f35, %f231, %f209, %f35;
    mul.f32 %f40, %f40, 0.99;
    mul.f32 %f41, %f41, 0.99;
    mul.f32 %f42, %f42, 1.0;
    mov.f32 %f30, %f105;
    mov.f32 %f31, %f106;
    mov.f32 %f32, %f107;
    bra RR_STEP;

DIFFUSE_BOUNCE:
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f210, %r21;
    mul.f32 %f210, %f210, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f211, %r21;
    mul.f32 %f211, %f211, %f406;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f212, %r21;
    mul.f32 %f212, %f212, %f406;
    add.f32 %f210, %f210, -0.5;
    add.f32 %f211, %f211, -0.5;
    add.f32 %f212, %f212, -0.5;
    fma.rn.f32 %f33, %f51, 1.4, %f210;
    fma.rn.f32 %f34, %f52, 1.4, %f211;
    fma.rn.f32 %f35, %f53, 1.4, %f212;
    mul.f32 %f213, %f33, %f33;
    fma.rn.f32 %f213, %f34, %f34, %f213;
    fma.rn.f32 %f213, %f35, %f35, %f213;
    rsqrt.approx.f32 %f214, %f213;
    mul.f32 %f33, %f33, %f214;
    mul.f32 %f34, %f34, %f214;
    mul.f32 %f35, %f35, %f214;
    mov.f32 %f30, %f105;
    mov.f32 %f31, %f106;
    mov.f32 %f32, %f107;
    bra RR_STEP;

SHADE_SKY:
    add.f32 %f215, %f34, 1.0;
    mul.f32 %f215, %f215, 0.5;
    mul.f32 %f216, %f215, 0.35;
    mul.f32 %f217, %f215, 0.45;
    mul.f32 %f218, %f215, 0.75;
    add.f32 %f216, %f216, 0.04;
    add.f32 %f217, %f217, 0.06;
    add.f32 %f218, %f218, 0.12;
    fma.rn.f32 %f43, %f40, %f216, %f43;
    fma.rn.f32 %f44, %f41, %f217, %f44;
    fma.rn.f32 %f45, %f42, %f218, %f45;
    bra PATH_DONE;

RR_STEP:
    setp.lt.u32 %p42, %r30, 2;
    @%p42 bra RR_SKIP;
    max.f32 %f216, %f40, %f41;
    max.f32 %f216, %f216, %f42;
    min.f32 %f216, %f216, %f418;
    max.f32 %f216, %f216, 0.10;
    mad.lo.u32 %r21, %r21, 1664525, 1013904223;
    cvt.rn.f32.u32 %f217, %r21;
    mul.f32 %f217, %f217, %f406;
    setp.gt.f32 %p43, %f217, %f216;
    @%p43 bra PATH_DONE;
    rcp.approx.f32 %f218, %f216;
    mul.f32 %f40, %f40, %f218;
    mul.f32 %f41, %f41, %f218;
    mul.f32 %f42, %f42, %f218;
RR_SKIP:
    add.u32 %r30, %r30, 1;
    bra BOUNCE_LOOP;

PATH_DONE:
    add.f32 %f10, %f10, %f43;
    add.f32 %f11, %f11, %f44;
    add.f32 %f12, %f12, %f45;
    add.u32 %r20, %r20, 1;
    bra SAMPLE_LOOP;

SAMPLE_DONE:
    mul.f32 %f10, %f10, %f416;
    mul.f32 %f11, %f11, %f416;
    mul.f32 %f12, %f12, %f416;

    add.f32 %f220, %f10, 1.0;
    add.f32 %f221, %f11, 1.0;
    add.f32 %f222, %f12, 1.0;
    div.rn.f32 %f10, %f10, %f220;
    div.rn.f32 %f11, %f11, %f221;
    div.rn.f32 %f12, %f12, %f222;
    sqrt.approx.f32 %f10, %f10;
    sqrt.approx.f32 %f11, %f11;
    sqrt.approx.f32 %f12, %f12;
    min.f32 %f10, %f10, 1.0;
    min.f32 %f11, %f11, 1.0;
    min.f32 %f12, %f12, 1.0;

    div.rn.f32 %f223, %f0, %f408;
    div.rn.f32 %f224, %f1, %f409;
    sub.f32 %f223, %f223, 0.5;
    sub.f32 %f224, %f224, 0.5;
    mul.f32 %f225, %f223, %f223;
    fma.rn.f32 %f225, %f224, %f224, %f225;
    mul.f32 %f225, %f225, 1.85;
    sub.f32 %f226, 1.0, %f225;
    max.f32 %f226, %f226, 0.72;
    mul.f32 %f10, %f10, %f226;
    mul.f32 %f11, %f11, %f226;
    mul.f32 %f12, %f12, %f226;

    mul.f32 %f223, %f10, %f405;
    mul.f32 %f224, %f11, %f405;
    mul.f32 %f225, %f12, %f405;
    cvt.rni.u32.f32 %r60, %f223;
    cvt.rni.u32.f32 %r61, %f224;
    cvt.rni.u32.f32 %r62, %f225;
    shl.b32 %r61, %r61, 8;
    shl.b32 %r60, %r60, 16;
    mov.u32 %r63, 0xFF000000;
    or.b32 %r64, %r62, %r61;
    or.b32 %r65, %r64, %r60;
    or.b32 %r66, %r65, %r63;
    st.global.u32 [%rd2], %r66;

DONE:
    ret;
}
"#

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
def clifIrSource : String := buildProgram do
  let fnWrite ← declareFileWrite
  let cuda ← declareCudaFFI

  let ptr ← entryBlock
  callVoid cuda.fnInit [ptr]
  let dataSz ← iconst64 pixelBytes
  let bufId ← call cuda.fnCreateBuffer [ptr, dataSz]
  let ptxOffV ← iconst64 ptxOff
  let nBufs ← iconst32 1
  let bindOffV ← iconst64 bindOff
  let gridX ← iconst32 ((imageWidth + 15) / 16)
  let gridY ← iconst32 ((imageHeight + 15) / 16)
  let one32 ← iconst32 1
  let blk16 ← iconst32 16
  let _ := bufId
  let _ ← call cuda.fnLaunch
    [ptr, ptxOffV, nBufs, bindOffV, gridX, gridY, one32, blk16, blk16, one32]
  let pxOffV ← iconst64 pixelsOff
  let _ ← call cuda.fnDownload [ptr, bufId, pxOffV, dataSz]
  callVoid cuda.fnCleanup [ptr]
  let total ← iconst64 (54 + pixelBytes)
  let _ ← writeFile0 ptr fnWrite filenameOff bmpHeaderOff total
  ret

def payloads : List UInt8 :=
  let reserved := zeros ptxOff
  let ptxBytes := padTo ((stringToBytes ptxSource) ++ [0]) ptxRegion
  let bindDesc := uint32ToBytes 0
  let bindPad := zeros (filenameOff - bindOff - bindDesc.length)
  let filenameBytes := padTo (stringToBytes "scene.bmp") filenameRegion
  let clifPad := zeros clifIrRegion
  reserved ++ ptxBytes ++ bindDesc ++ bindPad ++ filenameBytes ++ clifPad ++ bmpHeader

def config : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + pixelBytes,
  context_offset := 0,
  initial_memory := payloads
}

def algorithm : Algorithm := {
  actions := [IR.clifCallAction],
  cranelift_units := 0,
  timeout_ms := some 300000
}

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.config Algorithm.algorithm
  IO.println (Json.compress json)
