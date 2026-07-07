import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.WGSL

namespace Algorithm

-- ---------------------------------------------------------------------------
-- 4096x4096 Cornell box path tracer rendered on GPU, written to BMP
--
-- Single CLIF function orchestrates: GPU compute -> file write.
-- 2D dispatch with workgroup_size(16, 16) = 256 threads per group.
-- dispatch(4096/16, 4096/16, 1) = (256, 256, 1).
-- 32 samples per pixel, up to 5 bounces with direct + indirect lighting.
-- ---------------------------------------------------------------------------

def imageWidth : Nat := 4096
def imageHeight : Nat := 4096
def maxBounces : Nat := 5
def numSamples : Nat := 32

def pixelCount : Nat := imageWidth * imageHeight
def pixelBytes : Nat := pixelCount * 4

-- ---------------------------------------------------------------------------
-- BMP header (54 bytes) — 32-bit BGRA, top-down
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- WGSL compute shader: Cornell box path tracer
--
-- 2D dispatch with workgroup_size(16, 16). gid.x = column, gid.y = row.
-- Each thread traces 16 jittered samples with up to 5 bounces.
-- Scene: 5 planes (walls) + 2 AABBs (boxes) + area light on ceiling.
-- Lambertian diffuse with direct lighting and cosine-weighted indirect.
-- ---------------------------------------------------------------------------

def cornellBoxShader : String :=
  let pixels : AlgorithmLib.WGSL.Expr (.arr .u32) := ⟨"pixels"⟩
  let imgW : AlgorithmLib.WGSL.Expr .u32 := ⟨"IMG_W"⟩
  let imgH : AlgorithmLib.WGSL.Expr .u32 := ⟨"IMG_H"⟩
  let numSamplesE : AlgorithmLib.WGSL.Expr .u32 := ⟨"NUM_SAMPLES"⟩
  let maxBouncesE : AlgorithmLib.WGSL.Expr .u32 := ⟨"MAX_BOUNCES"⟩
  let v3 (x y z : String) : AlgorithmLib.WGSL.Expr .vec3f := mkVec3f (litF x) (litF y) (litF z)
  let pcgE (v : AlgorithmLib.WGSL.Expr .u32) : AlgorithmLib.WGSL.Expr .u32 := call1 "pcg" v
  let randFE (seed : AlgorithmLib.WGSL.Expr (.ptrFn .u32)) : AlgorithmLib.WGSL.Expr .f32 := call1 "rand_f" seed
  let cosineHemisphereE (n : AlgorithmLib.WGSL.Expr .vec3f) (seed : AlgorithmLib.WGSL.Expr (.ptrFn .u32)) : AlgorithmLib.WGSL.Expr .vec3f :=
    call2 "cosine_hemisphere" n seed
  let rayAabbE
      (ro rd bmin bmax : AlgorithmLib.WGSL.Expr .vec3f)
      (outNormal : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f)) : AlgorithmLib.WGSL.Expr .f32 :=
    call5 "ray_aabb" ro rd bmin bmax outNormal
  let sceneHitE
      (ro rd : AlgorithmLib.WGSL.Expr .vec3f)
      (outN outColor outEmit : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f)) : AlgorithmLib.WGSL.Expr .f32 :=
    call5 "scene_hit" ro rd outN outColor outEmit
  buildShader
    [{ binding := 0, name := "pixels", ty := .arr .u32 }]
    []
    [.constU "IMG_W" imageWidth,
     .constU "IMG_H" imageHeight,
     .constU "NUM_SAMPLES" numSamples,
     .constU "MAX_BOUNCES" maxBounces,
     .fn {
       name := "pcg",
       params := [{ name := "v", ty := .u32 }],
       retTy := some .u32,
       body := do
         let v : AlgorithmLib.WGSL.Expr .u32 := ⟨"v"⟩
         let s ← varV (v * litU 747796405 + litU 2891336453)
         let w ← letV ((bxorU (shrU s (shrU s (litU 28) + litU 4)) s) * litU 277803737)
         retE (bxorU (shrU w (litU 22)) w)
     },
     .fn {
       name := "rand_f",
       params := [{ name := "seed", ty := .ptrFn .u32 }],
       retTy := some .f32,
       body := do
         let seed : AlgorithmLib.WGSL.Expr (.ptrFn .u32) := ⟨"seed"⟩
         derefAssign seed (pcgE (deref seed))
         retE (f32OfU (deref seed) / litF "4294967295.0")
     },
     .fn {
       name := "cosine_hemisphere",
       params := [{ name := "n", ty := .vec3f }, { name := "seed", ty := .ptrFn .u32 }],
       retTy := some .vec3f,
       body := do
         let n : AlgorithmLib.WGSL.Expr .vec3f := ⟨"n"⟩
         let seed : AlgorithmLib.WGSL.Expr (.ptrFn .u32) := ⟨"seed"⟩
         let u1 ← letV (randFE seed)
         let u2 ← letV (randFE seed)
         let r ← letV (wSqrt u1)
         let theta ← letV (litF "6.283185307" * u2)
         let x ← letV (r * wCos theta)
         let y ← letV (r * wSin theta)
         let z ← letV (wSqrt (litF "1.0" - u1))
         let up ← varV (v3 "0.0" "1.0" "0.0")
         ifB (gtE (wAbs (vy n)) (litF "0.999")) do
           assign up (v3 "1.0" "0.0" "0.0")
         let tangent ← letV (wNorm (wCross up n))
         let bitangent ← letV (wCross n tangent)
         retE (wNorm (tangent * x + bitangent * y + n * z))
     },
     .fn {
       name := "ray_aabb",
       params := [{ name := "ro", ty := .vec3f }, { name := "rd", ty := .vec3f },
                  { name := "bmin", ty := .vec3f }, { name := "bmax", ty := .vec3f },
                  { name := "out_normal", ty := .ptrFn .vec3f }],
       retTy := some .f32,
       body := do
         let ro : AlgorithmLib.WGSL.Expr .vec3f := ⟨"ro"⟩
         let rd : AlgorithmLib.WGSL.Expr .vec3f := ⟨"rd"⟩
         let bmin : AlgorithmLib.WGSL.Expr .vec3f := ⟨"bmin"⟩
         let bmax : AlgorithmLib.WGSL.Expr .vec3f := ⟨"bmax"⟩
         let outNormal : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f) := ⟨"out_normal"⟩
         let invD ← letV (splatV3 (litF "1.0") / rd)
         let t1 ← letV ((bmin - ro) * invD)
         let t2 ← letV ((bmax - ro) * invD)
         let tminV ← letV (wMinV3 t1 t2)
         let tmaxV ← letV (wMaxV3 t1 t2)
         let tmin ← letV (wMax (wMax (vx tminV) (vy tminV)) (vz tminV))
         let tmax ← letV (wMin (wMin (vx tmaxV) (vy tmaxV)) (vz tmaxV))
         ifB ((gtE tmin tmax) .|| (ltE tmax (litF "0.001"))) do
           retE (litF "-1.0")
         let t ← letV (wSelect tmin tmax (ltE tmin (litF "0.001")))
         let hit ← letV (ro + t * rd)
         let ce ← letV ((bmin + bmax) * litF "0.5")
         let d ← letV ((hit - ce) / (bmax - bmin))
         let ad ← letV (wAbsV3 d)
         ifElse ((gtE (vx ad) (vy ad)) .&& (gtE (vx ad) (vz ad)))
           (derefAssign outNormal (mkVec3f (wSign (vx d)) (litF "0.0") (litF "0.0")))
           (ifElse (gtE (vy ad) (vz ad))
             (derefAssign outNormal (mkVec3f (litF "0.0") (wSign (vy d)) (litF "0.0")))
             (derefAssign outNormal (mkVec3f (litF "0.0") (litF "0.0") (wSign (vz d)))))
         retE t
     },
     .fn {
       name := "scene_hit",
       params := [{ name := "ro", ty := .vec3f }, { name := "rd", ty := .vec3f },
                  { name := "out_n", ty := .ptrFn .vec3f },
                  { name := "out_color", ty := .ptrFn .vec3f },
                  { name := "out_emit", ty := .ptrFn .vec3f }],
       retTy := some .f32,
       body := do
         let ro : AlgorithmLib.WGSL.Expr .vec3f := ⟨"ro"⟩
         let rd : AlgorithmLib.WGSL.Expr .vec3f := ⟨"rd"⟩
         let outN : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f) := ⟨"out_n"⟩
         let outColor : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f) := ⟨"out_color"⟩
         let outEmit : AlgorithmLib.WGSL.Expr (.ptrFn .vec3f) := ⟨"out_emit"⟩
         let white := v3 "0.73" "0.73" "0.73"
         let closest ← varV (litF "1e20")
         derefAssign outEmit (v3 "0.0" "0.0" "0.0")
         let inUnitXZ (p : AlgorithmLib.WGSL.Expr .vec3f) : AlgorithmLib.WGSL.Expr .bool :=
           (geE (vx p) (litF "0.0")) .&& (leE (vx p) (litF "1.0")) .&& (geE (vz p) (litF "0.0")) .&& (leE (vz p) (litF "1.0"))
         let inUnitXY (p : AlgorithmLib.WGSL.Expr .vec3f) : AlgorithmLib.WGSL.Expr .bool :=
           (geE (vx p) (litF "0.0")) .&& (leE (vx p) (litF "1.0")) .&& (geE (vy p) (litF "0.0")) .&& (leE (vy p) (litF "1.0"))
         let inUnitYZ (p : AlgorithmLib.WGSL.Expr .vec3f) : AlgorithmLib.WGSL.Expr .bool :=
           (geE (vy p) (litF "0.0")) .&& (leE (vy p) (litF "1.0")) .&& (geE (vz p) (litF "0.0")) .&& (leE (vz p) (litF "1.0"))
         let considerPlane :=
           fun (t : AlgorithmLib.WGSL.Expr .f32)
               (inside : AlgorithmLib.WGSL.Expr .vec3f → AlgorithmLib.WGSL.Expr .bool)
               (normal color : AlgorithmLib.WGSL.Expr .vec3f) => do
             ifB ((gtE t (litF "0.001")) .&& (ltE t closest)) do
               let p ← letV (ro + t * rd)
               ifB (inside p) do
                 assign closest t
                 derefAssign outN normal
                 derefAssign outColor color
         let tFloor ← letV (-vy ro / vy rd)
         considerPlane tFloor inUnitXZ (v3 "0.0" "1.0" "0.0") white
         let tCeil ← letV ((litF "1.0" - vy ro) / vy rd)
         ifB ((gtE tCeil (litF "0.001")) .&& (ltE tCeil closest)) do
           let p ← letV (ro + tCeil * rd)
           ifB (inUnitXZ p) do
             assign closest tCeil
             derefAssign outN (v3 "0.0" "-1.0" "0.0")
             ifElse ((gtE (vx p) (litF "0.35")) .&& (ltE (vx p) (litF "0.65")) .&& (gtE (vz p) (litF "0.35")) .&& (ltE (vz p) (litF "0.65")))
               (do
                 derefAssign outColor (v3 "0.78" "0.78" "0.78")
                 derefAssign outEmit (v3 "15.0" "15.0" "15.0"))
               (derefAssign outColor white)
         let tBack ← letV (-vz ro / vz rd)
         considerPlane tBack inUnitXY (v3 "0.0" "0.0" "1.0") white
         let tLeft ← letV (-vx ro / vx rd)
         considerPlane tLeft inUnitYZ (v3 "1.0" "0.0" "0.0") (v3 "0.65" "0.05" "0.05")
         let tRight ← letV ((litF "1.0" - vx ro) / vx rd)
         considerPlane tRight inUnitYZ (v3 "-1.0" "0.0" "0.0") (v3 "0.12" "0.45" "0.15")
         let tFront ← letV ((litF "1.0" - vz ro) / vz rd)
         considerPlane tFront inUnitXY (v3 "0.0" "0.0" "-1.0") white
         let bn ← varVT .vec3f
         let tTall ← letV (rayAabbE ro rd (v3 "0.53" "0.0" "0.09") (v3 "0.83" "0.60" "0.38") (addrOf bn))
         ifB ((gtE tTall (litF "0.001")) .&& (ltE tTall closest)) do
           assign closest tTall
           derefAssign outN bn
           derefAssign outColor white
         let tShort ← letV (rayAabbE ro rd (v3 "0.13" "0.0" "0.37") (v3 "0.43" "0.30" "0.67") (addrOf bn))
         ifB ((gtE tShort (litF "0.001")) .&& (ltE tShort closest)) do
           assign closest tShort
           derefAssign outN bn
           derefAssign outColor white
         retE closest
     }]
    { wgX := 16, wgY := 16 }
    do
      let px ← letV gidX
      let py ← letV gidY
      ifB ((geE px imgW) .|| (geE py imgH)) retV
      let idx ← letV (py * imgW + px)
      let camPos ← letV (v3 "0.5" "0.5" "0.9")
      let camTarget ← letV (v3 "0.5" "0.4" "0.0")
      let camFwd ← letV (wNorm (camTarget - camPos))
      let camRight ← letV (wNorm (wCross camFwd (v3 "0.0" "1.0" "0.0")))
      let camUp ← letV (wCross camRight camFwd)
      let fov ← letV (litF "0.55")
      let accum ← varV (v3 "0.0" "0.0" "0.0")
      let seed ← varV (pcgE (idx * litU 1973 + px * litU 9277 + py * litU 26699))
      forU "s" (litU 0) (fun s => ltE s numSamplesE) (fun s => s + litU 1) fun _ => do
        let jx ← letV ((f32OfU px + randFE (addrOf seed)) / f32OfU imgW)
        let jy ← letV ((f32OfU py + randFE (addrOf seed)) / f32OfU imgH)
        let uv ← letV (mkVec2f (jx - litF "0.5") (litF "0.5" - jy))
        let rayDir ← varV (wNorm (camFwd + fov * v2x uv * camRight + fov * v2y uv * camUp))
        let rayOrg ← varV camPos
        let throughput ← varV (v3 "1.0" "1.0" "1.0")
        let color ← varV (v3 "0.0" "0.0" "0.0")
        forU "bounce" (litU 0) (fun bounce => ltE bounce maxBouncesE) (fun bounce => bounce + litU 1) fun bounce => do
          let hitN ← varVT .vec3f
          let hitColor ← varVT .vec3f
          let hitEmit ← varVT .vec3f
          let t ← letV (sceneHitE rayOrg rayDir (addrOf hitN) (addrOf hitColor) (addrOf hitEmit))
          ifB (geE t (litF "1e19")) breakS
          assign color (color + throughput * hitEmit)
          let hitP ← letV (rayOrg + t * rayDir + litF "0.001" * hitN)
          let lightU ← letV (litF "0.35" + randFE (addrOf seed) * litF "0.30")
          let lightV ← letV (litF "0.35" + randFE (addrOf seed) * litF "0.30")
          let lightPos ← letV (mkVec3f lightU (litF "0.999") lightV)
          let toLight ← letV (lightPos - hitP)
          let lightDist ← letV (wLen toLight)
          let lightDir ← letV (toLight / lightDist)
          let nDotL ← letV (wMax (wDot hitN lightDir) (litF "0.0"))
          ifB (gtE nDotL (litF "0.0")) do
            let shadowN ← varVT .vec3f
            let shadowC ← varVT .vec3f
            let shadowE ← varVT .vec3f
            let st ← letV (sceneHitE hitP lightDir (addrOf shadowN) (addrOf shadowC) (addrOf shadowE))
            ifB (geE st (lightDist - litF "0.01")) do
              let lightArea ← letV (litF "0.09")
              let lightNDot ← letV (wMax (-vy lightDir) (litF "0.0"))
              let solidAngle ← letV (lightArea * lightNDot / (lightDist * lightDist))
              let lightIntensity ← letV (v3 "15.0" "15.0" "15.0")
              assign color (color + throughput * hitColor * lightIntensity * nDotL * solidAngle / litF "3.14159265")
          ifB (gtE bounce (litU 1)) do
            let pContinue ← letV (wMax (wMax (vx hitColor) (vy hitColor)) (vz hitColor))
            ifB (gtE (randFE (addrOf seed)) pContinue) breakS
            assign throughput (throughput / pContinue)
          assign throughput (throughput * hitColor)
          assign rayOrg hitP
          assign rayDir (cosineHemisphereE hitN (addrOf seed))
        assign accum (accum + color)
      let finalColor ← letV (accum / f32OfU numSamplesE)
      let mapped ← letV (wPow (wClampV3 finalColor (v3 "0.0" "0.0" "0.0") (v3 "1.0" "1.0" "1.0")) (v3 "0.45454545" "0.45454545" "0.45454545"))
      let ri ← letV (u32OfF (wClamp (vx mapped * litF "255.0") (litF "0.0") (litF "255.0")))
      let gi ← letV (u32OfF (wClamp (vy mapped * litF "255.0") (litF "0.0") (litF "255.0")))
      let bi ← letV (u32OfF (wClamp (vz mapped * litF "255.0") (litF "0.0") (litF "255.0")))
      assign (arrIdx pixels idx) (borU (borU bi (shlU gi (litU 8))) (borU (shlU ri (litU 16)) (shlU (litU 0xFF) (litU 24))))

-- ---------------------------------------------------------------------------
-- CLIF IR: GPU init -> buffer -> pipeline -> dispatch -> download -> cleanup -> file write
-- ---------------------------------------------------------------------------

-- Payload layout
def hdrBase : Nat := 0x40
def bindDesc_off : Nat := 0x100
def shader_off : Nat := 0x200
def shaderRegionSize : Nat := 16384   -- 16KB for the larger ray tracing shader
def filename_off : Nat := shader_off + shaderRegionSize
def filenameRegionSize : Nat := 256
def flag_off : Nat := filename_off + filenameRegionSize
def clifIr_off : Nat := flag_off + 64
def clifIrRegionSize : Nat := 4096
def bmpHeader_off : Nat := clifIr_off + clifIrRegionSize
def pixels_off : Nat := bmpHeader_off + 54

def wgX : Nat := imageWidth / 16    -- 256
def wgY : Nat := imageHeight / 16   -- 256

open AlgorithmLib.IR in
def clifIrSource : String := buildProgram do
  let gpu ← declareGpuFFI
  let fnWr ← declareFileWrite
  let ptr ← entryBlock
  gpuInit gpu ptr
  let dataSz ← iconst64 pixelBytes
  let bufId  ← gpuCreateBuffer gpu ptr dataSz
  let shOff  ← iconst64 shader_off
  let bdOff  ← iconst64 bindDesc_off
  let one32  ← iconst32 1
  let pipeId ← gpuCreatePipeline gpu ptr shOff bdOff one32
  let wgx    ← iconst32 wgX
  let wgy    ← iconst32 wgY
  let _      ← gpuDispatch gpu ptr pipeId wgx wgy one32
  let pxOff  ← iconst64 pixels_off
  let _      ← gpuDownload gpu ptr bufId pxOff dataSz
  gpuCleanup gpu ptr
  let total  ← iconst64 (54 + pixelBytes)
  let _      ← writeFile0 ptr fnWr filename_off bmpHeader_off total
  ret

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros hdrBase
  let hdrPad := zeros (bindDesc_off - hdrBase)
  let bindDesc := uint32ToBytes 0 ++ uint32ToBytes 0
  let bindPad := zeros (shader_off - bindDesc_off - 8)
  let shaderBytes := padTo (stringToBytes cornellBoxShader) shaderRegionSize
  let filenameBytes := padTo (stringToBytes "cornell_box.bmp") filenameRegionSize
  let flagBytes := uint64ToBytes 0
  let flagPad := zeros (clifIr_off - flag_off - 8)
  let clifPad := zeros clifIrRegionSize
  let bmpBytes := bmpHeader
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ filenameBytes ++ flagBytes ++ flagPad ++
  clifPad ++ bmpBytes

-- ---------------------------------------------------------------------------
-- Algorithm definition
-- ---------------------------------------------------------------------------

def raytraceConfig : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + pixelBytes,
  initial_memory := payloads
}

def raytraceAlgorithm : Algorithm := {
    fn_idx := IR.mainFnIdx
  }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "raytrace_app" Algorithm.raytraceConfig Algorithm.raytraceAlgorithm]
