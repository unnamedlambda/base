import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- 4096x4096 Cornell box path tracer rendered on GPU, written to BMP
--
-- Single CLIF function orchestrates: GPU compute -> file write.
-- 2D dispatch with workgroup_size(16, 16) = 256 threads per group.
-- dispatch(4096/16, 4096/16, 1) = (256, 256, 1).
-- 16 samples per pixel, up to 5 bounces with direct + indirect lighting.
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
  let w := toString imageWidth
  let h := toString imageHeight
  let spp := toString numSamples
  let bounces := toString maxBounces
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> pixels: array<u32>;\n\n" ++

  -- PCG random number generator
  "fn pcg(v: u32) -> u32 {\n" ++
  "    var s = v * 747796405u + 2891336453u;\n" ++
  "    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;\n" ++
  "    return (w >> 22u) ^ w;\n" ++
  "}\n\n" ++

  "fn rand_f(seed: ptr<function, u32>) -> f32 {\n" ++
  "    *seed = pcg(*seed);\n" ++
  "    return f32(*seed) / 4294967295.0;\n" ++
  "}\n\n" ++

  -- Cosine-weighted hemisphere sampling
  "fn cosine_hemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {\n" ++
  "    let u1 = rand_f(seed);\n" ++
  "    let u2 = rand_f(seed);\n" ++
  "    let r = sqrt(u1);\n" ++
  "    let theta = 6.283185307 * u2;\n" ++
  "    let x = r * cos(theta);\n" ++
  "    let y = r * sin(theta);\n" ++
  "    let z = sqrt(1.0 - u1);\n" ++
  "    var up = vec3<f32>(0.0, 1.0, 0.0);\n" ++
  "    if (abs(n.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }\n" ++
  "    let tangent = normalize(cross(up, n));\n" ++
  "    let bitangent = cross(n, tangent);\n" ++
  "    return normalize(tangent * x + bitangent * y + n * z);\n" ++
  "}\n\n" ++

  -- Ray-AABB intersection returning distance and setting normal
  "fn ray_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>,\n" ++
  "            out_normal: ptr<function, vec3<f32>>) -> f32 {\n" ++
  "    let inv_d = 1.0 / rd;\n" ++
  "    let t1 = (bmin - ro) * inv_d;\n" ++
  "    let t2 = (bmax - ro) * inv_d;\n" ++
  "    let tmin_v = min(t1, t2);\n" ++
  "    let tmax_v = max(t1, t2);\n" ++
  "    let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);\n" ++
  "    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);\n" ++
  "    if (tmin > tmax || tmax < 0.001) { return -1.0; }\n" ++
  "    let t = select(tmin, tmax, tmin < 0.001);\n" ++
  "    let hit = ro + t * rd;\n" ++
  "    let ce = (bmin + bmax) * 0.5;\n" ++
  "    let d = (hit - ce) / (bmax - bmin);\n" ++
  "    let ad = abs(d);\n" ++
  "    if (ad.x > ad.y && ad.x > ad.z) {\n" ++
  "        *out_normal = vec3<f32>(sign(d.x), 0.0, 0.0);\n" ++
  "    } else if (ad.y > ad.z) {\n" ++
  "        *out_normal = vec3<f32>(0.0, sign(d.y), 0.0);\n" ++
  "    } else {\n" ++
  "        *out_normal = vec3<f32>(0.0, 0.0, sign(d.z));\n" ++
  "    }\n" ++
  "    return t;\n" ++
  "}\n\n" ++

  -- Full scene intersection: 5 planes + 2 AABBs + area light
  "fn scene_hit(ro: vec3<f32>, rd: vec3<f32>,\n" ++
  "             out_n: ptr<function, vec3<f32>>,\n" ++
  "             out_color: ptr<function, vec3<f32>>,\n" ++
  "             out_emit: ptr<function, vec3<f32>>) -> f32 {\n" ++
  "    var closest = 1e20;\n" ++
  "    var n: vec3<f32>;\n" ++
  "    *out_emit = vec3<f32>(0.0, 0.0, 0.0);\n" ++

  -- Floor (y = 0): white
  "    let t_floor = -ro.y / rd.y;\n" ++
  "    if (t_floor > 0.001 && t_floor < closest) {\n" ++
  "        let p = ro + t_floor * rd;\n" ++
  "        if (p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0) {\n" ++
  "            closest = t_floor;\n" ++
  "            *out_n = vec3<f32>(0.0, 1.0, 0.0);\n" ++
  "            *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Ceiling (y = 1): white + area light check
  "    let t_ceil = (1.0 - ro.y) / rd.y;\n" ++
  "    if (t_ceil > 0.001 && t_ceil < closest) {\n" ++
  "        let p = ro + t_ceil * rd;\n" ++
  "        if (p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0) {\n" ++
  "            closest = t_ceil;\n" ++
  "            *out_n = vec3<f32>(0.0, -1.0, 0.0);\n" ++
  "            if (p.x > 0.35 && p.x < 0.65 && p.z > 0.35 && p.z < 0.65) {\n" ++
  "                *out_color = vec3<f32>(0.78, 0.78, 0.78);\n" ++
  "                *out_emit = vec3<f32>(15.0, 15.0, 15.0);\n" ++
  "            } else {\n" ++
  "                *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "            }\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Back wall (z = 0): white
  "    let t_back = -ro.z / rd.z;\n" ++
  "    if (t_back > 0.001 && t_back < closest) {\n" ++
  "        let p = ro + t_back * rd;\n" ++
  "        if (p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0) {\n" ++
  "            closest = t_back;\n" ++
  "            *out_n = vec3<f32>(0.0, 0.0, 1.0);\n" ++
  "            *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Left wall (x = 0): red
  "    let t_left = -ro.x / rd.x;\n" ++
  "    if (t_left > 0.001 && t_left < closest) {\n" ++
  "        let p = ro + t_left * rd;\n" ++
  "        if (p.y >= 0.0 && p.y <= 1.0 && p.z >= 0.0 && p.z <= 1.0) {\n" ++
  "            closest = t_left;\n" ++
  "            *out_n = vec3<f32>(1.0, 0.0, 0.0);\n" ++
  "            *out_color = vec3<f32>(0.65, 0.05, 0.05);\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Right wall (x = 1): green
  "    let t_right = (1.0 - ro.x) / rd.x;\n" ++
  "    if (t_right > 0.001 && t_right < closest) {\n" ++
  "        let p = ro + t_right * rd;\n" ++
  "        if (p.y >= 0.0 && p.y <= 1.0 && p.z >= 0.0 && p.z <= 1.0) {\n" ++
  "            closest = t_right;\n" ++
  "            *out_n = vec3<f32>(-1.0, 0.0, 0.0);\n" ++
  "            *out_color = vec3<f32>(0.12, 0.45, 0.15);\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Front wall (z = 1): white (closes the box)
  "    let t_front = (1.0 - ro.z) / rd.z;\n" ++
  "    if (t_front > 0.001 && t_front < closest) {\n" ++
  "        let p = ro + t_front * rd;\n" ++
  "        if (p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0) {\n" ++
  "            closest = t_front;\n" ++
  "            *out_n = vec3<f32>(0.0, 0.0, -1.0);\n" ++
  "            *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "        }\n" ++
  "    }\n" ++

  -- Tall box (right-rear): white AABB
  "    var bn: vec3<f32>;\n" ++
  "    let t_tall = ray_aabb(ro, rd,\n" ++
  "        vec3<f32>(0.53, 0.0, 0.09),\n" ++
  "        vec3<f32>(0.83, 0.60, 0.38), &bn);\n" ++
  "    if (t_tall > 0.001 && t_tall < closest) {\n" ++
  "        closest = t_tall;\n" ++
  "        *out_n = bn;\n" ++
  "        *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "    }\n" ++

  -- Short box (left-front): white AABB
  "    let t_short = ray_aabb(ro, rd,\n" ++
  "        vec3<f32>(0.13, 0.0, 0.37),\n" ++
  "        vec3<f32>(0.43, 0.30, 0.67), &bn);\n" ++
  "    if (t_short > 0.001 && t_short < closest) {\n" ++
  "        closest = t_short;\n" ++
  "        *out_n = bn;\n" ++
  "        *out_color = vec3<f32>(0.73, 0.73, 0.73);\n" ++
  "    }\n" ++

  "    return closest;\n" ++
  "}\n\n" ++

  -- Main compute entry point
  "@compute @workgroup_size(16, 16)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  s!"    let width: u32 = {w}u;\n" ++
  s!"    let height: u32 = {h}u;\n" ++
  "    let px = gid.x;\n" ++
  "    let py = gid.y;\n" ++
  "    if (px >= width || py >= height) { return; }\n" ++
  "    let idx = py * width + px;\n" ++

  -- Camera
  "    let cam_pos = vec3<f32>(0.5, 0.5, 0.9);\n" ++
  "    let cam_target = vec3<f32>(0.5, 0.4, 0.0);\n" ++
  "    let cam_fwd = normalize(cam_target - cam_pos);\n" ++
  "    let cam_right = normalize(cross(cam_fwd, vec3<f32>(0.0, 1.0, 0.0)));\n" ++
  "    let cam_up = cross(cam_right, cam_fwd);\n" ++
  "    let fov = 0.55;\n" ++

  -- Multi-sample path tracing
  s!"    let num_samples: u32 = {spp}u;\n" ++
  "    var accum = vec3<f32>(0.0, 0.0, 0.0);\n" ++
  "    var seed: u32 = pcg(idx * 1973u + px * 9277u + py * 26699u);\n" ++
  "    for (var s: u32 = 0u; s < num_samples; s = s + 1u) {\n" ++
  "        let jx = (f32(px) + rand_f(&seed)) / f32(width);\n" ++
  "        let jy = (f32(py) + rand_f(&seed)) / f32(height);\n" ++
  "        let uv = vec2<f32>(jx - 0.5, 0.5 - jy);\n" ++
  "        var ray_dir = normalize(cam_fwd + fov * uv.x * cam_right + fov * uv.y * cam_up);\n" ++
  "        var ray_org = cam_pos;\n" ++
  "        var throughput = vec3<f32>(1.0, 1.0, 1.0);\n" ++
  "        var color = vec3<f32>(0.0, 0.0, 0.0);\n" ++

  s!"        for (var bounce: u32 = 0u; bounce < {bounces}u; bounce = bounce + 1u) " ++ "{\n" ++
  "            var hit_n: vec3<f32>;\n" ++
  "            var hit_color: vec3<f32>;\n" ++
  "            var hit_emit: vec3<f32>;\n" ++
  "            let t = scene_hit(ray_org, ray_dir, &hit_n, &hit_color, &hit_emit);\n" ++
  "            if (t >= 1e19) { break; }\n" ++

  -- Emission
  "            color = color + throughput * hit_emit;\n" ++

  -- Direct lighting: shadow ray to area light
  "            let hit_p = ray_org + t * ray_dir + 0.001 * hit_n;\n" ++
  "            let light_u = 0.35 + rand_f(&seed) * 0.30;\n" ++
  "            let light_v = 0.35 + rand_f(&seed) * 0.30;\n" ++
  "            let light_pos = vec3<f32>(light_u, 0.999, light_v);\n" ++
  "            let to_light = light_pos - hit_p;\n" ++
  "            let light_dist = length(to_light);\n" ++
  "            let light_dir = to_light / light_dist;\n" ++
  "            let n_dot_l = max(dot(hit_n, light_dir), 0.0);\n" ++
  "            if (n_dot_l > 0.0) {\n" ++
  "                var shadow_n: vec3<f32>;\n" ++
  "                var shadow_c: vec3<f32>;\n" ++
  "                var shadow_e: vec3<f32>;\n" ++
  "                let st = scene_hit(hit_p, light_dir, &shadow_n, &shadow_c, &shadow_e);\n" ++
  "                if (st >= light_dist - 0.01) {\n" ++
  "                    let light_area = 0.09;\n" ++
  "                    let light_n_dot = max(-light_dir.y, 0.0);\n" ++
  "                    let solid_angle = light_area * light_n_dot / (light_dist * light_dist);\n" ++
  "                    let light_intensity = vec3<f32>(15.0, 15.0, 15.0);\n" ++
  "                    color = color + throughput * hit_color * light_intensity * n_dot_l * solid_angle / 3.14159265;\n" ++
  "                }\n" ++
  "            }\n" ++

  -- Russian roulette after bounce 2
  "            if (bounce > 1u) {\n" ++
  "                let p_continue = max(max(hit_color.x, hit_color.y), hit_color.z);\n" ++
  "                if (rand_f(&seed) > p_continue) { break; }\n" ++
  "                throughput = throughput / p_continue;\n" ++
  "            }\n" ++

  -- Indirect bounce
  "            throughput = throughput * hit_color;\n" ++
  "            ray_org = hit_p;\n" ++
  "            ray_dir = cosine_hemisphere(hit_n, &seed);\n" ++
  "        }\n" ++
  "        accum = accum + color;\n" ++
  "    }\n" ++

  -- Gamma correction and BGRA output
  "    let final_color = accum / f32(num_samples);\n" ++
  "    let mapped = pow(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));\n" ++
  "    let ri = u32(clamp(mapped.x * 255.0, 0.0, 255.0));\n" ++
  "    let gi = u32(clamp(mapped.y * 255.0, 0.0, 255.0));\n" ++
  "    let bi = u32(clamp(mapped.z * 255.0, 0.0, 255.0));\n" ++
  "    pixels[idx] = bi | (gi << 8u) | (ri << 16u) | (0xFFu << 24u);\n" ++
  "}\n"

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

def clifIrSource : String :=
  let sh := toString shader_off
  let bd := toString bindDesc_off
  let fn_out := toString filename_off
  let bmp := toString bmpHeader_off
  let bmpTotalSize := toString (54 + pixelBytes)
  let px := toString pixels_off
  let dataSz := toString pixelBytes
  let wgXStr := toString wgX
  let wgYStr := toString wgY
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i64, i64, i32) -> i32 system_v\n" ++
  "    sig3 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    fn0 = %cl_gpu_init sig0\n" ++
  "    fn1 = %cl_gpu_create_buffer sig1\n" ++
  "    fn2 = %cl_gpu_create_pipeline sig2\n" ++
  "    fn3 = %cl_gpu_dispatch sig4\n" ++
  "    fn4 = %cl_gpu_download sig3\n" ++
  "    fn5 = %cl_gpu_cleanup sig0\n" ++
  "    fn6 = %cl_file_write sig5\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    call fn0(v0)\n" ++
  s!"    v1 = iconst.i64 {dataSz}\n" ++
  "    v2 = call fn1(v0, v1)\n" ++
  s!"    v3 = iconst.i64 {sh}\n" ++
  s!"    v4 = iconst.i64 {bd}\n" ++
  "    v5 = iconst.i32 1\n" ++
  "    v6 = call fn2(v0, v3, v4, v5)\n" ++
  s!"    v7 = iconst.i32 {wgXStr}\n" ++
  s!"    v8 = iconst.i32 {wgYStr}\n" ++
  "    v9 = call fn3(v0, v6, v7, v8, v5)\n" ++
  s!"    v10 = iconst.i64 {px}\n" ++
  "    v11 = call fn4(v0, v2, v10, v1)\n" ++
  "    call fn5(v0)\n" ++
  s!"    v12 = iconst.i64 {fn_out}\n" ++
  s!"    v13 = iconst.i64 {bmp}\n" ++
  "    v14 = iconst.i64 0\n" ++
  s!"    v15 = iconst.i64 {bmpTotalSize}\n" ++
  "    v16 = call fn6(v0, v12, v13, v14, v15)\n" ++
  "    return\n" ++
  "}\n"

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

def raytraceConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + pixelBytes,
  context_offset := 0
}

def raytraceAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.raytraceConfig Algorithm.raytraceAlgorithm
  IO.println (Json.compress json)
