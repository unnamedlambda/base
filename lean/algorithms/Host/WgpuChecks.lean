import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate

/-!
# A program calling wgpu directly neither misuses it nor faults

wgpu's own contract is the one the checker reads (`Check.WgpuRule`): the
program makes an instance, its adapter, a device and its queue, a storage
buffer and a buffer to read it back through, a compute pipeline over a shader
with one binding, and a bind group; records a dispatch and a copy in one
encoder, submits them, and reads the result into the caller's output.

The theorem states the platform it needs: wgpu finds an adapter, accepts the
shader at `SHADER_OFF` (`wgText … true`), and the program has made no object
yet, so its handles are counted from the first.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuChecks

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- Where the layout's entry, the pipeline layout's group and the bind
    group's entry are written, and where the entry point's name and the
    shader's source are. -/
def ENTS : Nat := 0x40
def ARR : Nat := 0x48
def BG : Nat := 0x50
def NAME_OFF : Nat := 0x80
def SHADER_OFF : Nat := 0x100
def SHADER_LEN : Nat := 200

/-- How many bytes the buffer holds and the caller gets back. -/
def N : Nat := 256

def dispatchOnce : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let z ← iconst64 0
  let z32 ← iconst32 0
  let one ← iconst32 1
  let inst ← ext (.wgpu .createInstance) %[]
  let ad ← ext (.wgpu .requestAdapter) %[inst]
  let dev ← ext (.wgpu .requestDevice) %[ad]
  let q ← ext (.wgpu .deviceGetQueue) %[dev]
  let buf ← ext (.wgpu .deviceCreateBuffer) %[dev, ← iconst64 (128 + 8 + 4), ← iconst64 N]
  let st ← ext (.wgpu .deviceCreateBuffer) %[dev, ← iconst64 (1 + 8), ← iconst64 N]
  let sh ← ext (.wgpu .deviceCreateShaderModule) %[dev, ← iaddImm base SHADER_OFF, ← iconst64 SHADER_LEN]
  storeI64 z (← iaddImm base ENTS)
  let bgl ← ext (.wgpu .deviceCreateBindGroupLayout) %[dev, ← iaddImm base ENTS, ← iconst64 1]
  storeI64 bgl (← iaddImm base ARR)
  let pl ← ext (.wgpu .deviceCreatePipelineLayout) %[dev, ← iaddImm base ARR, ← iconst64 1]
  let pipe ← ext (.wgpu .deviceCreateComputePipeline) %[dev, pl, sh, ← iaddImm base NAME_OFF, ← iconst64 4]
  storeI64 z (← iaddImm base BG)
  storeI64 buf (← iaddImm base (BG + 8))
  let bg ← ext (.wgpu .deviceCreateBindGroup) %[dev, bgl, ← iaddImm base BG, ← iconst64 1]
  let enc ← ext (.wgpu .deviceCreateCommandEncoder) %[dev]
  let pass ← ext (.wgpu .encoderBeginComputePass) %[enc]
  let _ ← ext (.wgpu .computePassSetPipeline) %[pass, pipe]
  let _ ← ext (.wgpu .computePassSetBindGroup) %[pass, z32, bg]
  let _ ← ext (.wgpu .computePassDispatch) %[pass, one, one, one]
  let _ ← ext (.wgpu .computePassEnd) %[pass]
  let _ ← ext (.wgpu .encoderCopyBufferToBuffer) %[enc, buf, z, st, z, ← iconst64 N]
  let cb ← ext (.wgpu .encoderFinish) %[enc]
  let _ ← ext (.wgpu .queueSubmit) %[q, cb]
  let _ ← ext (.wgpu .bufferRead) %[dev, st, z, out, ← iconst64 N]

abbrev Start : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .idle true, .room .arena 4096, .room .out N,
   .nwg 0, .wg1 .commandEncoder none, .wg1 .computePass none, .wg1 .commandBuffer none,
   .wgText (regionBase .arena + UInt64.ofNat SHADER_OFF) SHADER_LEN true,
   .wgText (regionBase .arena + UInt64.ofNat NAME_OFF) 4 false]

ship_entry dispatchOnce : dispatchOnce under Start

#print axioms dispatchOnce_no_misuse

end WgpuChecks
