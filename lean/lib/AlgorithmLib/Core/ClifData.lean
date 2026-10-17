module
public import Lean
public import AlgorithmLib.Core.Cbor
meta import AlgorithmLib.Core.Cbor
public import AlgorithmLib.Core.Enum
meta import AlgorithmLib.Core.Enum
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The CLIF program an artifact carries

The instruction set the generators emit, as data, together with the CBOR
`base_types::clif` reads. Separate from `IR.lean` because `Core.Setup`
carries a `Program` and `IR` builds one — both need these types and neither
should import the other.
-/

namespace AlgorithmLib

namespace IR

/-- CLIF value types -/
inductive ClifTy where
  | i8 | i16 | i32 | i64
  | f32 | f64
  | f32x4 | i8x16
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq ClifTy where
  eq_of_beq {a b} h := by cases a <;> cases b <;> first | rfl | exact absurd h (by decide)
  rfl {a} := by cases a <;> decide

/-- An SSA value reference -/
structure Val where
  id : Nat
  deriving Repr, BEq

/-- A block reference -/
structure BlockRef where
  id : Nat
  deriving Repr, BEq

/-- Comparison condition codes -/
inductive ICmpCond where
  | eq | ne | uge | ugt | ule | ult | slt | sle | sgt | sge
  deriving Repr, BEq

/-- Float comparison conditions -/
inductive FloatCC where
  | eq | ne | lt | le | gt | ge
  deriving Repr, BEq

/-- Two-operand integer instructions whose operands and result share one
    integer type: signed division and both remainders, the four min/max, and
    the high halves of the two multiplications. -/
inductive IBin where
  | sdiv | urem | srem | smin | smax | umin | umax | umulhi | smulhi
  deriving Repr, BEq, DecidableEq

/-- Shift-shaped instructions: the amount may be any integer width and is
    taken modulo the shifted operand's. -/
inductive IShift where
  | sshr | rotl | rotr
  deriving Repr, BEq, DecidableEq

/-- One-operand integer instructions whose result has the operand's type.
    `cls` is absent: x64 has no lowering for it, so a body using it would not
    build everywhere. -/
inductive IUn where
  | bnot | iabs | clz | bswap | bitrev
  deriving Repr, BEq, DecidableEq

/-- Two-operand float instructions, scalar or lane-wise. -/
inductive FBin where
  | fdiv | fcopysign
  deriving Repr, BEq, DecidableEq

/-- One-operand float instructions, scalar or lane-wise. `nearest` rounds half
    to even. -/
inductive FUn where
  | sqrt | fabs | ceil | floor | trunc | nearest
  deriving Repr, BEq, DecidableEq

/-- Conversions to a named type. `toSint` saturates, as `fcvtToUint` does; a
    conversion that traps on an out-of-range value would abort the process. -/
inductive FConv where
  | toSint | fromUint | demote
  deriving Repr, BEq, DecidableEq

/-- Integer width changes to a named type: narrowing keeps the low bits, and
    the two widenings differ in what fills the new top bits. A store narrower
    than its value is a narrowing followed by a store at the narrow type, which
    Cranelift emits as one instruction. -/
inductive IExt where
  | reduce | uextend | sextend
  deriving Repr, BEq, DecidableEq

/-- The read-modify-write an `atomic_rmw` performs on the old value and its
    operand. -/
inductive AtomicRmw where
  | add | sub | and | nand | or | xor | xchg | umin | umax | smin | smax
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- CLIF's atomic instructions, at an integer width.

    They travel as a callee rather than as instructions of their own: each takes
    arguments and may answer a value exactly as a call does, so the term, its
    checker and the compilation proof treat them as calls, and the engine emits
    the instruction in place of the call. -/
inductive Atomic where
  | load (ty : ClifTy)
  | store (ty : ClifTy)
  | rmw (ty : ClifTy) (k : AtomicRmw)
  | cas (ty : ClifTy)
  | fence
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- **A C library a program may call directly**, by the files each operating
    system names it with, in the order they are tried.

    Every one is ABI-portable: the same symbols, signatures, constants and
    layouts on each target, so an artifact calls it identically everywhere.
    It is loaded when a program first names it; a library that is not there
    makes its functions absent rather than the program unloadable. The C
    library, wgpu, the engine's own functions (`base`, the `Ffi` entries) and
    its window, CPU, serial and USB libraries are built into it
    (`Lib.builtin`): always there, bound to the functions the engine links. -/
inductive Lib where
  | c | cuda | cublas | usb | window | cpu | wgpu | serial | base
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def Lib.name : Lib → String
  | .c => "c" | .cuda => "cuda" | .cublas => "cublas" | .usb => "usb"
  | .window => "window" | .cpu => "cpu" | .wgpu => "wgpu" | .serial => "serial"
  | .base => "base"

/-- Built into the engine: present on every platform, with no file to find. -/
def Lib.builtin : Lib → Bool
  | .c | .usb | .window | .cpu | .wgpu | .serial | .base => true
  | _ => false

/-- The C library's memory functions: pointers and sizes only, so the same
    call on every target. `memcmp` is not here: the value it answers beyond
    its sign differs between implementations. `calloc` and `free` are the
    heap: zeroed memory, and its release. -/
inductive CFn where
  | memcpy | memmove | memset | strlen | calloc | free
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- The CUDA driver API, called directly: context, memory, modules, launches
    and streams. Each answers a `CUresult`, `0` for success; an object it
    creates is written through the pointer it is handed. -/
inductive CudaFn where
  | init | deviceGet | primaryCtxRetain | primaryCtxRelease | ctxSetCurrent | ctxSynchronize
  | memGetInfo | memAlloc | memFree | memcpyHtoD | memcpyDtoH | memcpyDtoD | memsetD8
  | moduleLoadData | moduleGetFunction | moduleUnload | launchKernel
  | streamCreate | streamSynchronize | streamDestroy
  | eventCreate | eventRecord | streamWaitEvent | eventSynchronize | eventDestroy
  | eventElapsedTime
  | memAllocHost | memFreeHost | memcpyHtoDAsync | memcpyDtoHAsync
  | beginCapture | endCapture | graphInstantiate | graphLaunch | graphExecDestroy | graphDestroy
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- Every driver function, in declaration order. -/
def CudaFn.all : List CudaFn :=
  [.init, .deviceGet, .primaryCtxRetain, .primaryCtxRelease, .ctxSetCurrent, .ctxSynchronize,
   .memGetInfo, .memAlloc, .memFree, .memcpyHtoD, .memcpyDtoH, .memcpyDtoD, .memsetD8,
   .moduleLoadData, .moduleGetFunction, .moduleUnload, .launchKernel,
   .streamCreate, .streamSynchronize, .streamDestroy,
   .eventCreate, .eventRecord, .streamWaitEvent, .eventSynchronize, .eventDestroy,
   .eventElapsedTime, .memAllocHost, .memFreeHost, .memcpyHtoDAsync, .memcpyDtoHAsync,
   .beginCapture, .endCapture, .graphInstantiate, .graphLaunch, .graphExecDestroy, .graphDestroy]

theorem CudaFn.mem_all (f : CudaFn) : f ∈ CudaFn.all := by cases f <;> simp [CudaFn.all]

/-- cuBLAS, called directly: handles, the stream a handle runs on, and the
    single-precision and bf16-in, f32-out products. Each answers a
    `cublasStatus_t`, `0` for success. -/
inductive CublasFn where
  | create | destroy | setStream
  | sgemv | sgemm | sgemmStridedBatched | gemmEx | gemmStridedBatchedEx | sgemmBatched
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def CublasFn.all : List CublasFn :=
  [.create, .destroy, .setStream, .sgemv, .sgemm, .sgemmStridedBatched, .gemmEx,
   .gemmStridedBatchedEx, .sgemmBatched]

theorem CublasFn.mem_all (f : CublasFn) : f ∈ CublasFn.all := by cases f <;> simp [CublasFn.all]

/-- The kinds of object wgpu hands a program, each released by its own
    symbol (`wgpu<Kind>Release`). -/
inductive WgObj where
  | instance | adapter | device | queue | buffer | shaderModule | bindGroupLayout
  | pipelineLayout | computePipeline | bindGroup | commandEncoder | computePass
  | commandBuffer | surface | texture | textureView | renderPipeline | renderPass
  | errorScope
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def WgObj.all : List WgObj :=
  [.instance, .adapter, .device, .queue, .buffer, .shaderModule, .bindGroupLayout,
   .pipelineLayout, .computePipeline, .bindGroup, .commandEncoder, .computePass,
   .commandBuffer, .surface, .texture, .textureView, .renderPipeline, .renderPass, .errorScope]

/-- The kind's name, as its release symbol spells it after `wgpu`. -/
def WgObj.cname : WgObj → String
  | .instance => "Instance" | .adapter => "Adapter" | .device => "Device" | .queue => "Queue"
  | .buffer => "Buffer" | .shaderModule => "ShaderModule" | .bindGroupLayout => "BindGroupLayout"
  | .pipelineLayout => "PipelineLayout" | .computePipeline => "ComputePipeline"
  | .bindGroup => "BindGroup" | .commandEncoder => "CommandEncoder"
  | .computePass => "ComputePassEncoder" | .commandBuffer => "CommandBuffer"
  | .surface => "Surface" | .texture => "Texture" | .textureView => "TextureView"
  | .renderPipeline => "RenderPipeline" | .renderPass => "RenderPassEncoder"
  | .errorScope => "ErrorScope"

/-- wgpu, through the engine's shims over the `wgpu` crate
    (`base/src/ffi/wgpu.rs`): the objects a compute dispatch and a present to
    a window need, each made by one call whose descriptor is its arguments.
    What a call reads from program memory is a run of words or of bytes at an
    address and a length it is handed; what it writes is the bytes a buffer
    read asks for. -/
inductive WgpuFn where
  | createInstance | requestAdapter | requestDevice | release (o : WgObj)
  | deviceGetQueue | deviceCreateBuffer | deviceCreateShaderModule
  | deviceCreateBindGroupLayout | deviceCreatePipelineLayout | deviceCreateComputePipeline
  | deviceCreateBindGroup | deviceCreateCommandEncoder | deviceCreateRenderPipeline
  | devicePushErrorScope | errorScopePop | renderPipelineGetBindGroupLayout
  | encoderBeginComputePass | computePassSetPipeline | computePassSetBindGroup
  | computePassDispatch | computePassEnd
  | encoderCopyBufferToBuffer | encoderFinish
  | encoderBeginRenderPass | renderPassSetPipeline | renderPassSetBindGroup
  | renderPassDraw | renderPassEnd
  | queueSubmit | queueWriteBuffer | bufferRead
  | instanceCreateSurface | surfaceConfigure
  | surfaceGetCurrentTexture | surfacePresent | textureCreateView
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def WgpuFn.all : List WgpuFn :=
  [.createInstance, .requestAdapter, .requestDevice] ++ WgObj.all.map .release ++
  [.deviceGetQueue, .deviceCreateBuffer, .deviceCreateShaderModule,
   .deviceCreateBindGroupLayout, .deviceCreatePipelineLayout, .deviceCreateComputePipeline,
   .deviceCreateBindGroup, .deviceCreateCommandEncoder, .deviceCreateRenderPipeline,
   .devicePushErrorScope, .errorScopePop, .renderPipelineGetBindGroupLayout,
   .encoderBeginComputePass, .computePassSetPipeline, .computePassSetBindGroup,
   .computePassDispatch, .computePassEnd, .encoderCopyBufferToBuffer, .encoderFinish,
   .encoderBeginRenderPass, .renderPassSetPipeline, .renderPassSetBindGroup,
   .renderPassDraw, .renderPassEnd, .queueSubmit, .queueWriteBuffer, .bufferRead,
   .instanceCreateSurface, .surfaceConfigure,
   .surfaceGetCurrentTexture, .surfacePresent, .textureCreateView]

theorem WgpuFn.mem_all (f : WgpuFn) : f ∈ WgpuFn.all := by
  cases f
  case release o => cases o <;> simp [WgpuFn.all, WgObj.all]
  all_goals simp [WgpuFn.all]

/-- The engine's window library (`base/src/ffi/window.rs`, over winit): the
    thread's event loop, windows, their events as 32-byte records, and their
    size in pixels. A wgpu surface is made from a window's handle. Where it
    answers `bool` the result is an `i8`. -/
inductive WindowFn where
  | init | open | close | pump | poll | pixels
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def WindowFn.all : List WindowFn := [.init, .open, .close, .pump, .poll, .pixels]

theorem WindowFn.mem_all (f : WindowFn) : f ∈ WindowFn.all := by cases f <;> simp [WindowFn.all]

/-- The engine's CPU library (`base/src/ffi/cpu.rs`): how many logical CPUs
    are online, the physical core and package of each, and pinning the
    calling thread to one or releasing it. Each answers an `int`, `-1` where
    the system does not say or does not grant it. -/
inductive CpuFn where
  | count | core | package | pin | unpin
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def CpuFn.all : List CpuFn := [.count, .core, .package, .pin, .unpin]

theorem CpuFn.mem_all (f : CpuFn) : f ∈ CpuFn.all := by cases f <;> simp [CpuFn.all]

/-- The engine's serial library (`base/src/ffi/serial.rs`, over the
    `serialport` crate): the ports the system lists and their names, and a
    port opened by name, read, written, asked what is waiting, and closed. -/
inductive SerialFn where
  | count | name | open | close | read | write | pending
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def SerialFn.all : List SerialFn := [.count, .name, .open, .close, .read, .write, .pending]

theorem SerialFn.mem_all (f : SerialFn) : f ∈ SerialFn.all := by cases f <;> simp [SerialFn.all]

/-- The engine's USB library (`base/src/ffi/usb.rs`, over the `nusb`
    crate): the devices the system lists and what each is, a device opened
    by its place, its interfaces claimed and released, and control, bulk and
    interrupt transfers through a claimed interface, each waited for. -/
inductive UsbFn where
  | count | info | open | close | claim | release | control | bulk | interrupt
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def UsbFn.all : List UsbFn := [.count, .info, .open, .close, .claim, .release, .control, .bulk, .interrupt]

theorem UsbFn.mem_all (f : UsbFn) : f ∈ UsbFn.all := by cases f <;> simp [UsbFn.all]

/-- **A function of a C library**, by family. `present l` is not a symbol: it
    answers `1` when `l` loaded and `0` when it did not. -/
inductive Ext where
  | present (l : Lib)
  | c (f : CFn)
  | cuda (f : CudaFn)
  | cublas (f : CublasFn)
  | wgpu (f : WgpuFn)
  | window (f : WindowFn)
  | cpu (f : CpuFn)
  | serial (f : SerialFn)
  | usb (f : UsbFn)
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def Ext.lib : Ext → Lib
  | .present l => l
  | .c _ => .c
  | .cuda _ => .cuda
  | .cublas _ => .cublas
  | .wgpu _ => .wgpu
  | .window _ => .window
  | .cpu _ => .cpu
  | .serial _ => .serial
  | .usb _ => .usb

/-- The symbol the library exports, empty for the presence probe. -/
def Ext.symbol : Ext → String
  | .present _ => ""
  | .c .memcpy => "memcpy" | .c .memmove => "memmove" | .c .memset => "memset"
  | .c .strlen => "strlen" | .c .calloc => "calloc" | .c .free => "free"
  | .cuda f => match f with
    | .init => "cuInit" | .deviceGet => "cuDeviceGet"
    | .primaryCtxRetain => "cuDevicePrimaryCtxRetain"
    | .primaryCtxRelease => "cuDevicePrimaryCtxRelease_v2"
    | .ctxSetCurrent => "cuCtxSetCurrent" | .ctxSynchronize => "cuCtxSynchronize"
    | .memGetInfo => "cuMemGetInfo_v2" | .memAlloc => "cuMemAlloc_v2" | .memFree => "cuMemFree_v2"
    | .memcpyHtoD => "cuMemcpyHtoD_v2" | .memcpyDtoH => "cuMemcpyDtoH_v2"
    | .memcpyDtoD => "cuMemcpyDtoD_v2" | .memsetD8 => "cuMemsetD8_v2"
    | .moduleLoadData => "cuModuleLoadData" | .moduleGetFunction => "cuModuleGetFunction"
    | .moduleUnload => "cuModuleUnload" | .launchKernel => "cuLaunchKernel"
    | .streamCreate => "cuStreamCreate" | .streamSynchronize => "cuStreamSynchronize"
    | .streamDestroy => "cuStreamDestroy_v2"
    | .eventCreate => "cuEventCreate" | .eventRecord => "cuEventRecord"
    | .streamWaitEvent => "cuStreamWaitEvent" | .eventSynchronize => "cuEventSynchronize"
    | .eventDestroy => "cuEventDestroy_v2" | .eventElapsedTime => "cuEventElapsedTime"
    | .memAllocHost => "cuMemAllocHost_v2" | .memFreeHost => "cuMemFreeHost"
    | .memcpyHtoDAsync => "cuMemcpyHtoDAsync_v2" | .memcpyDtoHAsync => "cuMemcpyDtoHAsync_v2"
    | .beginCapture => "cuStreamBeginCapture_v2" | .endCapture => "cuStreamEndCapture"
    | .graphInstantiate => "cuGraphInstantiateWithFlags" | .graphLaunch => "cuGraphLaunch"
    | .graphExecDestroy => "cuGraphExecDestroy" | .graphDestroy => "cuGraphDestroy"
  | .cublas f => match f with
    | .create => "cublasCreate_v2" | .destroy => "cublasDestroy_v2"
    | .setStream => "cublasSetStream_v2" | .sgemv => "cublasSgemv_v2" | .sgemm => "cublasSgemm_v2"
    | .sgemmStridedBatched => "cublasSgemmStridedBatched" | .gemmEx => "cublasGemmEx"
    | .gemmStridedBatchedEx => "cublasGemmStridedBatchedEx" | .sgemmBatched => "cublasSgemmBatched"
  | .wgpu f => match f with
    | .createInstance => "wgpuCreateInstance"
    | .requestAdapter => "wgpuInstanceRequestAdapter"
    | .requestDevice => "wgpuAdapterRequestDevice"
    | .release o => "wgpu" ++ o.cname ++ "Release"
    | .deviceGetQueue => "wgpuDeviceGetQueue"
    | .deviceCreateBuffer => "wgpuDeviceCreateBuffer"
    | .deviceCreateShaderModule => "wgpuDeviceCreateShaderModule"
    | .deviceCreateBindGroupLayout => "wgpuDeviceCreateBindGroupLayout"
    | .deviceCreatePipelineLayout => "wgpuDeviceCreatePipelineLayout"
    | .deviceCreateComputePipeline => "wgpuDeviceCreateComputePipeline"
    | .deviceCreateBindGroup => "wgpuDeviceCreateBindGroup"
    | .deviceCreateCommandEncoder => "wgpuDeviceCreateCommandEncoder"
    | .deviceCreateRenderPipeline => "wgpuDeviceCreateRenderPipeline"
    | .devicePushErrorScope => "wgpuDevicePushErrorScope"
    | .errorScopePop => "wgpuErrorScopePop"
    | .renderPipelineGetBindGroupLayout => "wgpuRenderPipelineGetBindGroupLayout"
    | .encoderBeginComputePass => "wgpuCommandEncoderBeginComputePass"
    | .computePassSetPipeline => "wgpuComputePassSetPipeline"
    | .computePassSetBindGroup => "wgpuComputePassSetBindGroup"
    | .computePassDispatch => "wgpuComputePassDispatchWorkgroups"
    | .computePassEnd => "wgpuComputePassEnd"
    | .encoderCopyBufferToBuffer => "wgpuCommandEncoderCopyBufferToBuffer"
    | .encoderFinish => "wgpuCommandEncoderFinish"
    | .encoderBeginRenderPass => "wgpuCommandEncoderBeginRenderPass"
    | .renderPassSetPipeline => "wgpuRenderPassSetPipeline"
    | .renderPassSetBindGroup => "wgpuRenderPassSetBindGroup"
    | .renderPassDraw => "wgpuRenderPassDraw"
    | .renderPassEnd => "wgpuRenderPassEnd"
    | .queueSubmit => "wgpuQueueSubmit"
    | .queueWriteBuffer => "wgpuQueueWriteBuffer"
    | .bufferRead => "wgpuBufferRead"
    | .instanceCreateSurface => "wgpuInstanceCreateSurface"
    | .surfaceConfigure => "wgpuSurfaceConfigure"
    | .surfaceGetCurrentTexture => "wgpuSurfaceGetCurrentTexture"
    | .surfacePresent => "wgpuSurfaceTexturePresent"
    | .textureCreateView => "wgpuSurfaceTextureCreateView"
  | .window f => match f with
    | .init => "base_window_init" | .open => "base_window_open" | .close => "base_window_close"
    | .pump => "base_window_pump" | .poll => "base_window_poll" | .pixels => "base_window_pixels"
  | .cpu f => match f with
    | .count => "base_cpu_count" | .core => "base_cpu_core" | .package => "base_cpu_package"
    | .pin => "base_cpu_pin" | .unpin => "base_cpu_unpin"
  | .serial f => "base_serial_" ++ match f with
    | .count => "count" | .name => "name" | .open => "open" | .close => "close"
    | .read => "read" | .write => "write" | .pending => "pending"
  | .usb f => "base_usb_" ++ match f with
    | .count => "count" | .info => "info" | .open => "open" | .close => "close"
    | .claim => "claim" | .release => "release" | .control => "control"
    | .bulk => "bulk" | .interrupt => "interrupt"

/-- Parameters and result, as the C declaration gives them: a pointer or a
    `size_t` is an `i64`, an `int` an `i32`. -/
def Ext.sig : Ext → List ClifTy × Option ClifTy
  | .present _ => ([], some .i32)
  | .c .memcpy | .c .memmove => ([.i64, .i64, .i64], some .i64)
  | .c .memset => ([.i64, .i32, .i64], some .i64)
  | .c .strlen => ([.i64], some .i64)
  | .c .calloc => ([.i64, .i64], some .i64)
  | .c .free => ([.i64], none)
  -- `CUresult`, `CUdevice`, `unsigned int` are `i32`; handles, pointers,
  -- `CUdeviceptr` and `size_t` are `i64`. `cuMemsetD8`'s `unsigned char`
  -- travels zero-extended as an `i32`, which a register reads the same.
  | .cuda f => (match f with
    | .init => [.i32] | .deviceGet => [.i64, .i32] | .primaryCtxRetain => [.i64, .i32]
    | .primaryCtxRelease => [.i32] | .ctxSetCurrent => [.i64] | .ctxSynchronize => []
    | .memGetInfo => [.i64, .i64] | .memAlloc => [.i64, .i64] | .memFree => [.i64]
    | .memcpyHtoD | .memcpyDtoH | .memcpyDtoD => [.i64, .i64, .i64]
    | .memsetD8 => [.i64, .i32, .i64]
    | .moduleLoadData => [.i64, .i64] | .moduleGetFunction => [.i64, .i64, .i64]
    | .moduleUnload => [.i64]
    | .launchKernel => [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i64]
    | .streamCreate => [.i64, .i32] | .streamSynchronize | .streamDestroy => [.i64]
    | .eventCreate => [.i64, .i32] | .eventRecord => [.i64, .i64]
    | .streamWaitEvent => [.i64, .i64, .i32]
    | .eventSynchronize | .eventDestroy => [.i64]
    | .eventElapsedTime => [.i64, .i64, .i64]
    | .memAllocHost => [.i64, .i64] | .memFreeHost => [.i64]
    | .memcpyHtoDAsync | .memcpyDtoHAsync => [.i64, .i64, .i64, .i64]
    | .beginCapture => [.i64, .i32] | .endCapture => [.i64, .i64]
    | .graphInstantiate => [.i64, .i64, .i64] | .graphLaunch => [.i64, .i64]
    | .graphExecDestroy | .graphDestroy => [.i64],
    some .i32)
  -- `cublasOperation_t`, `cudaDataType`, `cublasComputeType_t`,
  -- `cublasGemmAlgo_t` and `int` are `i32`; `long long` strides are `i64`.
  | .cublas f => (match f with
    | .create | .destroy => [.i64] | .setStream => [.i64, .i64]
    | .sgemv => [.i64, .i32, .i32, .i32, .i64, .i64, .i32, .i64, .i32, .i64, .i64, .i32]
    | .sgemm => [.i64, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i32, .i64, .i32, .i64, .i64, .i32]
    | .sgemmStridedBatched =>
        [.i64, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i32, .i64, .i64, .i32, .i64, .i64,
         .i64, .i32, .i64, .i32]
    | .gemmEx =>
        [.i64, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i32, .i32, .i64, .i32, .i32, .i64,
         .i64, .i32, .i32, .i32, .i32]
    | .gemmStridedBatchedEx =>
        [.i64, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i32, .i32, .i64, .i64, .i32, .i32,
         .i64, .i64, .i64, .i32, .i32, .i64, .i32, .i32, .i32]
    -- the three operands are device arrays of device pointers
    | .sgemmBatched =>
        [.i64, .i32, .i32, .i32, .i32, .i32, .i64, .i64, .i32, .i64, .i32, .i64, .i64, .i32, .i32],
    some .i32)
  -- Objects, addresses, lengths, counts, sizes, offsets and buffer usages
  -- are `i64`; group indices, workgroup and draw counts, formats and a
  -- surface's size, `i32`; a status `i32`.
  | .wgpu f => match f with
    | .createInstance => ([], some .i64)
    | .requestAdapter | .requestDevice | .deviceGetQueue | .deviceCreateCommandEncoder
    | .devicePushErrorScope | .encoderBeginComputePass | .encoderFinish
    | .surfaceGetCurrentTexture | .textureCreateView => ([.i64], some .i64)
    | .release _ | .computePassEnd | .renderPassEnd | .surfacePresent => ([.i64], none)
    | .errorScopePop => ([.i64], some .i32)
    | .deviceCreateBuffer | .deviceCreateShaderModule | .deviceCreateBindGroupLayout
    | .deviceCreatePipelineLayout => ([.i64, .i64, .i64], some .i64)
    | .deviceCreateComputePipeline => ([.i64, .i64, .i64, .i64, .i64], some .i64)
    | .deviceCreateBindGroup => ([.i64, .i64, .i64, .i64], some .i64)
    | .deviceCreateRenderPipeline => ([.i64, .i64, .i64, .i64, .i64, .i64, .i32], some .i64)
    | .renderPipelineGetBindGroupLayout => ([.i64, .i32], some .i64)
    | .computePassSetPipeline | .renderPassSetPipeline | .queueSubmit => ([.i64, .i64], none)
    | .computePassSetBindGroup | .renderPassSetBindGroup => ([.i64, .i32, .i64], none)
    | .computePassDispatch => ([.i64, .i32, .i32, .i32], none)
    | .encoderCopyBufferToBuffer => ([.i64, .i64, .i64, .i64, .i64, .i64], none)
    | .encoderBeginRenderPass => ([.i64, .i64, .i32], some .i64)
    | .renderPassDraw => ([.i64, .i32, .i32, .i32, .i32], none)
    | .queueWriteBuffer => ([.i64, .i64, .i64, .i64, .i64], none)
    | .bufferRead => ([.i64, .i64, .i64, .i64, .i64], some .i32)
    | .instanceCreateSurface => ([.i64, .i64], some .i64)
    | .surfaceConfigure => ([.i64, .i64, .i32, .i32, .i32], none)
  -- `bool` is an `i8`, `int32_t` an `i32`; a window, a title and a record
  -- pointer `i64`s.
  | .window f => match f with
    | .init => ([], some .i8)
    | .open => ([.i64, .i32, .i32], some .i64)
    | .close => ([.i64], none)
    | .pump => ([], none)
    | .poll => ([.i64, .i64], some .i8)
    | .pixels => ([.i64], some .i64)
  -- A CPU's number, and every answer, is an `int`.
  | .cpu f => match f with
    | .count | .unpin => ([], some .i32)
    | .core | .package | .pin => ([.i32], some .i32)
  -- A port, a device, a buffer, a length and a count are `i64`s; an index,
  -- a baud rate, a timeout, an interface, an endpoint and the fields of a
  -- SETUP packet `i32`s.
  | .serial f => match f with
    | .count => ([], some .i32)
    | .name => ([.i32, .i64, .i64], some .i64)
    | .open => ([.i64, .i32, .i32], some .i64)
    | .close => ([.i64], none)
    | .read | .write => ([.i64, .i64, .i64], some .i64)
    | .pending => ([.i64], some .i64)
  | .usb f => match f with
    | .count => ([], some .i32)
    | .info => ([.i32, .i32], some .i64)
    | .open => ([.i32], some .i64)
    | .close => ([.i64], none)
    | .claim => ([.i64, .i32], some .i64)
    | .release => ([.i64], none)
    | .control => ([.i64, .i32, .i32, .i32, .i32, .i64, .i32, .i32], some .i64)
    | .bulk | .interrupt => ([.i64, .i32, .i64, .i64, .i32], some .i64)

/-- The C declaration's parameter types, for the functions a header declares
    more than once in C++ (as overloads that take older enums): what selects
    the C one. -/
def Ext.cParams : Ext → Option (List String)
  | .cublas .gemmEx => some
      ["cublasHandle_t", "cublasOperation_t", "cublasOperation_t", "int", "int", "int",
       "const void*", "const void*", "cudaDataType", "int", "const void*", "cudaDataType", "int",
       "const void*", "void*", "cudaDataType", "int", "cublasComputeType_t", "cublasGemmAlgo_t"]
  | .cublas .gemmStridedBatchedEx => some
      ["cublasHandle_t", "cublasOperation_t", "cublasOperation_t", "int", "int", "int",
       "const void*", "const void*", "cudaDataType", "int", "long long int", "const void*",
       "cudaDataType", "int", "long long int", "const void*", "void*", "cudaDataType", "int",
       "long long int", "int", "cublasComputeType_t", "cublasGemmAlgo_t"]
  | _ => none

/-- Which load instruction, independent of the type it yields. -/
inductive LoadKind where
  | plain | uload8 | uload16 | uload32 | sload8 | sload16 | sload32
  deriving Repr, BEq

/-- A load: what to read, as what type, under which memory flags. -/
structure LoadOp where
  kind : LoadKind := .plain
  ty : ClifTy
  /-- On a vector load, the program states the address is aligned to the
      vector's size: one that says so of an address that is not is undefined
      (`evalOp` faults), as an aligned SSE operand faults. On a scalar it
      states nothing, and changes no code. No accessor sets it yet: the input
      and output may start anywhere, and a load from the memory sets it only
      where its alignment is proven. -/
  aligned : Bool := false
  deriving Repr, BEq

/-- One runtime entry point. -/
inductive Ffi where
  | fileReadToPtr | fileWriteFromPtr
  | stdinReadline | stdoutWrite
  | gpuInit | gpuCreateBuffer | gpuCreatePipeline | gpuUpload | gpuDownload
  | gpuDispatch | gpuCleanup | gpuUploadPtr | gpuDownloadPtr
  | windowInit | windowOpen | windowPoll | windowPresentGpuBuffer | windowCleanup
  | lmdbInit | lmdbOpen | lmdbBeginWriteTxn | lmdbPut | lmdbCommitWriteTxn
  | lmdbCursorScan | lmdbCleanup
  | htCreate | htLookup | htInsert | htIncrement | htCount | htGetEntry
  | htCleanup | htInit
  | sinf | cosf | powf
  | threadInit | threadSpawn | threadJoin | threadCleanup
  | cudaInit | cudaCreateBuffer | cudaUpload | cudaUploadOffset | cudaUploadAsync
  | cudaUploadOffsetAsync | cudaDownload | cudaDownloadOffset | cudaDownloadAsync
  | cudaFreeBuffer | cudaStreamCreate | cudaStreamSync | cudaStreamDestroy
  | cudaEventCreate | cudaEventRecord | cudaStreamWaitEvent | cudaEventElapsedMsBits
  | cudaEventDestroy | cudaGraphBeginCapture | cudaGraphEndCapture | cudaGraphUpload
  | cudaGraphLaunch | cudaGraphDestroy | cudaPinnedAlloc | cudaPinnedPtr
  | cudaPinnedFree | cudaLaunch | cudaLaunchNamed | cudaLaunchOnStream
  | cudaLaunchNamedOnStream | cudaSync | cudaCleanup
  | cublasSgemv | cublasSgemvOnStream | cublasSgemm | cublasSgemmOnStream
  | cublasPtrArray | cublasSgemmBatchedOnStream
  | cudaPinnedPtrAt | cudaMemInfoFree | cudaMemInfoTotal | cublasGemmExBf16
  | cublasGemmStridedBatchedExBf16
  | nativeLoad | nativeFree | nativeArch | cpuHas
  | fileCreateDirAll | threadStart | threadFinish
  | memLock | memUnlock | memAdviseHuge | threadPriority
  | memWait | memWake | clockNow
  | fileSize | fileSetLen
  deriving Repr, DecidableEq, Inhabited, Lean.ToExpr

namespace Ffi

/-- The C symbol the JIT resolves. -/
def cname : Ffi → String
  | .fileReadToPtr => "cl_file_read_to_ptr"
  | .fileWriteFromPtr => "cl_file_write_from_ptr"
  | .stdinReadline => "cl_stdin_read_line"
  | .stdoutWrite => "cl_stdout_write"
  | .gpuInit => "cl_gpu_init"
  | .gpuCreateBuffer => "cl_gpu_create_buffer"
  | .gpuCreatePipeline => "cl_gpu_create_pipeline"
  | .gpuUpload => "cl_gpu_upload"
  | .gpuDownload => "cl_gpu_download"
  | .gpuDispatch => "cl_gpu_dispatch"
  | .gpuCleanup => "cl_gpu_cleanup"
  | .gpuUploadPtr => "cl_gpu_upload_ptr"
  | .gpuDownloadPtr => "cl_gpu_download_ptr"
  | .windowInit => "cl_window_init"
  | .windowOpen => "cl_window_open"
  | .windowPoll => "cl_window_poll"
  | .windowPresentGpuBuffer => "cl_window_present_gpu_buffer"
  | .windowCleanup => "cl_window_cleanup"
  | .lmdbInit => "cl_lmdb_init"
  | .lmdbOpen => "cl_lmdb_open"
  | .lmdbBeginWriteTxn => "cl_lmdb_begin_write_txn"
  | .lmdbPut => "cl_lmdb_put"
  | .lmdbCommitWriteTxn => "cl_lmdb_commit_write_txn"
  | .lmdbCursorScan => "cl_lmdb_cursor_scan"
  | .lmdbCleanup => "cl_lmdb_cleanup"
  | .htCreate => "ht_create"
  | .htLookup => "ht_lookup"
  | .htInsert => "ht_insert"
  | .htIncrement => "ht_increment"
  | .htCount => "ht_count"
  | .htGetEntry => "ht_get_entry"
  | .htCleanup => "cl_ht_cleanup"
  | .htInit => "cl_ht_init"
  | .sinf => "cl_sinf"
  | .cosf => "cl_cosf"
  | .powf => "cl_powf"
  | .threadInit => "cl_thread_init"
  | .threadSpawn => "cl_thread_spawn"
  | .threadJoin => "cl_thread_join"
  | .threadCleanup => "cl_thread_cleanup"
  | .cudaInit => "cl_cuda_init"
  | .cudaCreateBuffer => "cl_cuda_create_buffer"
  | .cudaUpload => "cl_cuda_upload_ptr"
  | .cudaUploadOffset => "cl_cuda_upload_ptr_offset"
  | .cudaUploadAsync => "cl_cuda_upload_ptr_async"
  | .cudaUploadOffsetAsync => "cl_cuda_upload_ptr_offset_async"
  | .cudaDownload => "cl_cuda_download_ptr"
  | .cudaDownloadOffset => "cl_cuda_download_ptr_offset"
  | .cudaDownloadAsync => "cl_cuda_download_ptr_async"
  | .cudaFreeBuffer => "cl_cuda_free_buffer"
  | .cudaStreamCreate => "cl_cuda_stream_create"
  | .cudaStreamSync => "cl_cuda_stream_sync"
  | .cudaStreamDestroy => "cl_cuda_stream_destroy"
  | .cudaEventCreate => "cl_cuda_event_create"
  | .cudaEventRecord => "cl_cuda_event_record"
  | .cudaStreamWaitEvent => "cl_cuda_stream_wait_event"
  | .cudaEventElapsedMsBits => "cl_cuda_event_elapsed_ms_bits"
  | .cudaEventDestroy => "cl_cuda_event_destroy"
  | .cudaGraphBeginCapture => "cl_cuda_graph_begin_capture"
  | .cudaGraphEndCapture => "cl_cuda_graph_end_capture"
  | .cudaGraphUpload => "cl_cuda_graph_upload"
  | .cudaGraphLaunch => "cl_cuda_graph_launch"
  | .cudaGraphDestroy => "cl_cuda_graph_destroy"
  | .cudaPinnedAlloc => "cl_cuda_pinned_alloc"
  | .cudaPinnedPtr => "cl_cuda_pinned_ptr"
  | .cudaPinnedFree => "cl_cuda_pinned_free"
  | .cudaLaunch => "cl_cuda_launch"
  | .cudaLaunchNamed => "cl_cuda_launch_named"
  | .cudaLaunchOnStream => "cl_cuda_launch_on_stream"
  | .cudaLaunchNamedOnStream => "cl_cuda_launch_named_on_stream"
  | .cudaSync => "cl_cuda_sync"
  | .cudaCleanup => "cl_cuda_cleanup"
  | .cublasSgemv => "cl_cublas_sgemv"
  | .cublasSgemvOnStream => "cl_cublas_sgemv_on_stream"
  | .cublasSgemm => "cl_cublas_sgemm_strided_batched"
  | .cublasSgemmOnStream => "cl_cublas_sgemm_strided_batched_on_stream"
  | .cublasPtrArray => "cl_cublas_ptr_array"
  | .cublasSgemmBatchedOnStream => "cl_cublas_sgemm_batched_on_stream"
  | .cudaPinnedPtrAt => "cl_cuda_pinned_ptr_at"
  | .cudaMemInfoFree => "cl_cuda_mem_info_free"
  | .cudaMemInfoTotal => "cl_cuda_mem_info_total"
  | .cublasGemmExBf16 => "cl_cublas_gemm_ex_bf16"
  | .cublasGemmStridedBatchedExBf16 => "cl_cublas_gemm_strided_batched_ex_bf16"
  | .nativeLoad => "cl_native_load"
  | .nativeFree => "cl_native_free"
  | .nativeArch => "cl_native_arch"
  | .cpuHas => "cl_cpu_has"
  | .fileCreateDirAll => "cl_file_create_dir_all"
  | .threadStart => "cl_thread_start"
  | .threadFinish => "cl_thread_finish"
  | .memLock => "cl_mem_lock"
  | .memUnlock => "cl_mem_unlock"
  | .memAdviseHuge => "cl_mem_advise_huge"
  | .threadPriority => "cl_thread_priority"
  | .memWait => "cl_mem_wait"
  | .memWake => "cl_mem_wake"
  | .clockNow => "cl_clock_now"
  | .fileSize => "cl_file_size"
  | .fileSetLen => "cl_file_set_len"

/-- Parameters and result, exactly as `base/src/ffi/` takes them. -/
def sig : Ffi → List ClifTy × Option ClifTy
  | .fileReadToPtr => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .fileWriteFromPtr => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .stdinReadline => ([.i64, .i64], some .i64)
  | .stdoutWrite => ([.i64, .i64], some .i64)
  | .gpuInit => ([.i64], none)
  | .gpuCreateBuffer => ([.i64, .i64], some .i32)
  | .gpuCreatePipeline => ([.i64, .i64, .i64, .i32], some .i32)
  | .gpuUpload => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDownload => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDispatch => ([.i64, .i32, .i32, .i32, .i32], some .i32)
  | .gpuCleanup => ([.i64], none)
  | .gpuUploadPtr => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDownloadPtr => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .windowInit => ([.i64], none)
  | .windowOpen => ([.i64, .i64, .i64, .i64, .i64, .i64, .i64], some .i32)
  | .windowPoll => ([.i64, .i64, .i32], some .i32)
  | .windowPresentGpuBuffer => ([.i64, .i64, .i32], some .i32)
  | .windowCleanup => ([.i64], none)
  | .lmdbInit => ([.i64], none)
  | .lmdbOpen => ([.i64, .i64, .i32], some .i32)
  | .lmdbBeginWriteTxn => ([.i64, .i32], some .i32)
  | .lmdbPut => ([.i64, .i32, .i64, .i32, .i64, .i32], some .i32)
  | .lmdbCommitWriteTxn => ([.i64, .i32], some .i32)
  | .lmdbCursorScan => ([.i64, .i32, .i64, .i32, .i32, .i64, .i64], some .i32)
  | .lmdbCleanup => ([.i64], none)
  | .htCreate => ([.i64], some .i32)
  | .htLookup => ([.i64, .i64, .i32, .i64], some .i32)
  | .htInsert => ([.i64, .i64, .i32, .i64, .i32], none)
  | .htIncrement => ([.i64, .i64, .i32, .i64], some .i64)
  | .htCount => ([.i64], some .i32)
  | .htGetEntry => ([.i64, .i32, .i64, .i64], some .i32)
  | .htCleanup => ([.i64], none)
  | .htInit => ([.i64], none)
  | .sinf => ([.f32], some .f32)
  | .cosf => ([.f32], some .f32)
  | .powf => ([.f32, .f32], some .f32)
  | .threadInit => ([.i64], none)
  | .threadSpawn => ([.i64, .i64, .i64], some .i64)
  | .threadJoin => ([.i64, .i64], some .i64)
  | .threadCleanup => ([.i64], none)
  | .cudaInit => ([.i64], none)
  | .cudaCreateBuffer => ([.i64, .i64], some .i32)
  | .cudaUpload => ([.i64, .i32, .i64, .i64], some .i32)
  | .cudaUploadOffset => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .cudaUploadAsync => ([.i64, .i32, .i64, .i64, .i32], some .i32)
  | .cudaUploadOffsetAsync => ([.i64, .i32, .i64, .i64, .i64, .i32], some .i32)
  | .cudaDownload => ([.i64, .i32, .i64, .i64], some .i32)
  | .cudaDownloadOffset => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .cudaDownloadAsync => ([.i64, .i32, .i64, .i64, .i32], some .i32)
  | .cudaFreeBuffer => ([.i64, .i32], some .i32)
  | .cudaStreamCreate => ([.i64], some .i32)
  | .cudaStreamSync => ([.i64, .i32], some .i32)
  | .cudaStreamDestroy => ([.i64, .i32], some .i32)
  | .cudaEventCreate => ([.i64], some .i32)
  | .cudaEventRecord => ([.i64, .i32, .i32], some .i32)
  | .cudaStreamWaitEvent => ([.i64, .i32, .i32], some .i32)
  | .cudaEventElapsedMsBits => ([.i64, .i32, .i32], some .i32)
  | .cudaEventDestroy => ([.i64, .i32], some .i32)
  | .cudaGraphBeginCapture => ([.i64, .i32], some .i32)
  | .cudaGraphEndCapture => ([.i64, .i32], some .i32)
  | .cudaGraphUpload => ([.i64, .i32, .i32], some .i32)
  | .cudaGraphLaunch => ([.i64, .i32, .i32], some .i32)
  | .cudaGraphDestroy => ([.i64, .i32], some .i32)
  | .cudaPinnedAlloc => ([.i64, .i64], some .i32)
  | .cudaPinnedPtr => ([.i64, .i32], some .i64)
  | .cudaPinnedFree => ([.i64, .i32], some .i32)
  | .cudaLaunch => ([.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchNamed =>
      ([.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchOnStream =>
      ([.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchNamedOnStream =>
      ([.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaSync => ([.i64], some .i32)
  | .cudaCleanup => ([.i64], none)
  | .cublasSgemv => ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cublasSgemvOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  -- The three `.i64`s after the batch count are element offsets into the A, B
  -- and C operands; a zero trailing `ld_*` asks for the default leading
  -- dimension. An offset moves the pointer and leaves the matrix the call
  -- contracts alone, so one buffer can hold several operands without touching
  -- what `Law.cublasIsMatvec` says a contraction computes.
  | .cublasSgemm =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
        .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  | .cublasSgemmOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
        .i32, .i64, .i32, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  | .cublasPtrArray => ([.i64, .i32, .i32, .i32, .i64], some .i32)
  | .cublasSgemmBatchedOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32,
        .i32], some .i32)
  -- `(ctx, pinned_id, off, len)`: the pinned pool's host address at `off`, or
  -- `-1` when `off + len` runs past the allocation. The bound is the point --
  -- `cudaUploadOffsetAsync` checks the device range it writes but takes its
  -- source as a bare address, so an unchecked offset into a multi-gigabyte pool
  -- uploads whatever the process has there and calls it a weight.
  | .cudaPinnedPtrAt => ([.i64, .i32, .i64, .i64], some .i64)
  -- Free and total device memory. What a card has left after weights, caches
  -- and the driver's own reservations is not a number that can be written down
  -- ahead of the machine, so the sizes that depend on it are read here.
  | .cudaMemInfoFree => ([.i64], some .i64)
  | .cudaMemInfoTotal => ([.i64], some .i64)
  -- `cublasGemmEx` over bf16 operands with an f32 accumulator and result.
  -- Same argument shape as `.cublasSgemm` minus the batching: transposes, the
  -- three dimensions, alpha/A/B, beta/C, then the three element offsets and the
  -- three leading dimensions. BOTH inputs are bf16 -- cuBLAS rejects a mixed
  -- (bf16, f32) pair -- so `off_a` and `off_b` count 2-byte elements while
  -- `off_c` counts 4-byte ones.
  | .cublasGemmExBf16 =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32,
        .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  -- `.cublasSgemmBatchedOnStream`'s shape with `.cublasGemmExBf16`'s element
  -- types: transposes, dimensions, then (alpha, A, strideA), (B, strideB),
  -- (beta, C, strideC), the batch count, and the offsets and leading
  -- dimensions. Strides and offsets count elements, and an element is two
  -- bytes on both inputs and four on the result.
  | .cublasGemmStridedBatchedExBf16 =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64,
        .i32, .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  -- ffi/native.rs: machine code the program carries as data. `load` takes the
  -- bytes' address and length and answers where they now run (0 if they could
  -- not be placed); `arch` and `cpuHas` (a NUL-terminated feature name) decide
  -- which bytes to carry in. Running them is not an import: it is a call with
  -- `Callee.native`.
  | .nativeLoad => ([.i64, .i64], some .i64)
  | .nativeFree => ([.i64], none)
  | .nativeArch => ([], some .i32)
  | .cpuHas => ([.i64, .i64], some .i32)
  -- `(path)`: the directory at the path, and every one missing on the way;
  -- `0`, or `-1` when that cannot be done.
  | .fileCreateDirAll => ([.i64, .i64], some .i64)
  -- `(fn, arg)`: function `fn` of the program run on `arg` on a new thread,
  -- answered as a handle, or `-1`; `(handle)`: that thread waited for and the
  -- handle released, once: `0`, or `-1`.
  | .threadStart => ([.i64, .i64], some .i64)
  | .threadFinish => ([.i64], some .i64)
  -- The performance controls: lock `(ptr, len)` in physical memory, unlock
  -- it, ask for huge pages under it, and raise (`1`) or lower (`-1`) the
  -- calling thread's priority a step, or reset it (`0`). Each answers `0`,
  -- or `-1` when the system declines; none changes what memory holds.
  | .memLock => ([.i64, .i64], some .i32)
  | .memUnlock => ([.i64, .i64], some .i32)
  | .memAdviseHuge => ([.i64, .i64], some .i32)
  | .threadPriority => ([.i32], some .i32)
  -- ffi/sync.rs: `(addr, expected, timeout)` sleeps while the 32-bit word at
  -- `addr` holds `expected`, until a wake on `addr`, the timeout in
  -- nanoseconds (none if negative), or for no reason: `0` woken, `1` the word
  -- did not hold it, `2` timed out, `-1` an address not 4-byte aligned.
  -- `(addr, count)` wakes up to `count` waiters there (all if negative),
  -- answering how many; `()` reads a clock that never goes backwards, in
  -- nanoseconds. None changes what memory holds.
  | .memWait => ([.i64, .i32, .i64], none)
  | .memWake => ([.i64, .i32], none)
  | .clockNow => ([], some .i64)
  | .fileSize => ([.i64, .i64], some .i64)
  | .fileSetLen => ([.i64, .i64, .i64], some .i64)
  -- What wgpu answers through a callback, waited for: `(instance)` its
  -- high-performance adapter, or `0`; `(adapter)` a device, or `0`;
  -- `(device, buffer, offset, dst, size)` the buffer's bytes copied to `dst`
  -- through a mapping, `0` or `-1`; `(device)` the innermost error scope
  -- popped, `0` when it caught nothing and `-1` otherwise.

def params (f : Ffi) : List ClifTy := f.sig.1
def result (f : Ffi) : Option ClifTy := f.sig.2

/-- Every entry point, in the order that fixes the ids. -/
def all : List Ffi :=
  [.fileReadToPtr, .fileWriteFromPtr,
   .stdinReadline, .stdoutWrite,
   .gpuInit, .gpuCreateBuffer, .gpuCreatePipeline, .gpuUpload, .gpuDownload,
   .gpuDispatch, .gpuCleanup, .gpuUploadPtr, .gpuDownloadPtr,
   .windowInit, .windowOpen, .windowPoll, .windowPresentGpuBuffer, .windowCleanup,
   .lmdbInit, .lmdbOpen, .lmdbBeginWriteTxn, .lmdbPut, .lmdbCommitWriteTxn,
   .lmdbCursorScan, .lmdbCleanup,
   .htCreate, .htLookup, .htInsert, .htIncrement, .htCount, .htGetEntry,
   .htCleanup, .htInit,
   .sinf, .cosf, .powf,
   .threadInit, .threadSpawn, .threadJoin, .threadCleanup,
   .cudaInit, .cudaCreateBuffer, .cudaUpload, .cudaUploadOffset, .cudaUploadAsync,
   .cudaUploadOffsetAsync, .cudaDownload, .cudaDownloadOffset, .cudaDownloadAsync,
   .cudaFreeBuffer, .cudaStreamCreate, .cudaStreamSync, .cudaStreamDestroy,
   .cudaEventCreate, .cudaEventRecord, .cudaStreamWaitEvent, .cudaEventElapsedMsBits,
   .cudaEventDestroy, .cudaGraphBeginCapture, .cudaGraphEndCapture, .cudaGraphUpload,
   .cudaGraphLaunch, .cudaGraphDestroy, .cudaPinnedAlloc, .cudaPinnedPtr,
   .cudaPinnedFree, .cudaLaunch, .cudaLaunchNamed, .cudaLaunchOnStream,
   .cudaLaunchNamedOnStream, .cudaSync, .cudaCleanup,
   .cublasSgemv, .cublasSgemvOnStream, .cublasSgemm, .cublasSgemmOnStream,
   .cublasPtrArray, .cublasSgemmBatchedOnStream,
   .cudaPinnedPtrAt, .cudaMemInfoFree, .cudaMemInfoTotal, .cublasGemmExBf16,
   .cublasGemmStridedBatchedExBf16,
   .nativeLoad, .nativeFree, .nativeArch, .cpuHas, .fileCreateDirAll, .threadStart, .threadFinish,
   .memLock, .memUnlock, .memAdviseHuge, .threadPriority,
   .memWait, .memWake, .clockNow, .fileSize, .fileSetLen]

/-- The registry of entry points: `all` misses none and repeats none, so the
    ids below are a numbering of `Ffi` and not merely of a list beside it. -/
instance : Enum Ffi where
  all := all
  complete f := by cases f <;> decide
  nodup := by decide

/-- `f`'s position in `all`. -/
def id (f : Ffi) : Nat := all.idxOf f

/-- Distinct entry points carry distinct ids. -/
theorem id_injective {f g : Ffi} (h : f.id = g.id) : f = g := Enum.index_injective h

/-- The entry point a name resolves to, if it is one. -/
def ofCname (s : String) : Option Ffi := all.find? (·.cname == s)

/-- **Entry points that are functions of the program.** A library written in
    CLIF implements each over the C and vendor libraries, and a call to one is
    a call to that function: the entry point has no meaning of its own. -/
def inLib : Ffi → Bool
  | .htCreate | .htLookup | .htInsert | .htIncrement | .htCount | .htGetEntry | .htCleanup | .htInit => true
  | _ => false

/-- Where the library function for `f` sits among a program's functions,
    past every index a program gives its own. -/
def libSlot (f : Ffi) : Nat := 2 ^ 32 + f.id

/-- **Entry points the engine answers**, by their contracts: every one but
    the thread spawn, which runs a function of the program, and those a
    library implements. -/
def engine (f : Ffi) : Bool := f != .threadSpawn && !f.inLib

end Ffi

/-- What a call names.

    Each arm is the identity its owner gives it: an import is named by the
    engine's own entry point, a defined function by its place in this artifact,
    which is closed. Machine code the program placed itself has no name at
    all: its address is a value, the call's first argument. Nothing here is an
    index into a table this format invented, which is what a call carrying a
    position into a per-function callee list was.

    An import that the engine does not provide is unrepresentable rather than
    refused at load: `Ffi` is the only way to name one. -/
inductive Callee where
  /-- A function of the engine's own, `Lib.base`'s `cname`, called at the
      signature `Ffi.sig` states, as any built-in library's is. -/
  | ffi (f : Ffi)
  /-- Another function of this same program, by its `u0:N` position. -/
  | local (index : Nat)
  /-- Machine code at the address in the call's first argument --- one
      `nativeLoad` answered --- called on the other four under the
      architecture's C convention (System V on x86-64 whatever the OS), and
      answering an `i64`. A plain indirect call: nothing of the engine's runs
      between the caller and the code. -/
  | native
  /-- The host's function at the address in the call's first argument, called
      on the rest at `params`, answering `result`, under the host's C
      convention: how a host hands a program a callback, or a C function of
      its own. A plain indirect call. -/
  | host (params : List ClifTy) (result : Option ClifTy)
  /-- A CLIF atomic instruction, which the engine emits in place of a call. -/
  | atomic (a : Atomic)
  /-- A function of a C library, called directly. -/
  | ext (e : Ext)
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- A single CLIF instruction -/
inductive Inst where
  | iconst (dst : Val) (ty : ClifTy) (value : Int)
  /-- The address of the program's memory (`symbol_value` of `memory`). -/
  | memory (dst : Val)
  /-- The address of a function (`func_addr`), without calling it. -/
  | funcAddr (dst : Val) (callee : Callee)
  | iadd (dst : Val) (a b : Val)
  | isub (dst : Val) (a b : Val)
  | imul (dst : Val) (a b : Val)
  | udiv (dst : Val) (a b : Val)
  | ineg (dst : Val) (a : Val)
  | ishl (dst : Val) (a b : Val)
  | ushr (dst : Val) (a b : Val)
  | band (dst : Val) (a b : Val)
  | bandNot (dst : Val) (a b : Val)
  | bor (dst : Val) (a b : Val)
  | bxor (dst : Val) (a b : Val)
  | ireduce32 (dst : Val) (a : Val)
  | uextend64 (dst : Val) (a : Val)
  | sextend64 (dst : Val) (a : Val)
  | store (val addr : Val)
  | istore8 (val addr : Val)
  | load (dst : Val) (op : LoadOp) (addr : Val)
  | icmp (dst : Val) (cond : ICmpCond) (a b : Val)
  | select (dst : Val) (cond a b : Val)
  | call (dst : Option Val) (callee : Callee) (args : List Val)
  | jump (target : BlockRef) (args : List Val)
  | brif (cond : Val) (thenBlk : BlockRef) (thenArgs : List Val)
         (elseBlk : BlockRef) (elseArgs : List Val)
  /-- `return v`, or `return` when a function answers nothing. A body whose
      `ret` carries a value is a body whose signature returns an `i64`: the
      runtime reads the signature off the body rather than being told. -/
  | ret (value : Option Val)
  -- Float / SIMD
  | fconst (dst : Val) (ty : ClifTy) (bits : UInt64)
  | fadd (dst a b : Val)
  | fsub (dst a b : Val)
  | fmul (dst a b : Val)
  | fmax (dst a b : Val)
  | fmin (dst a b : Val)
  | fpromote (dst a : Val)
  | splat (dst : Val) (ty : ClifTy) (src : Val)
  | extractlane (dst : Val) (src : Val) (lane : Nat)
  | storeTyped (ty : ClifTy) (val addr : Val)
  -- Additional float / int ops
  | fneg (dst a : Val)
  | fcvtFromSint (dst : Val) (ty : ClifTy) (src : Val)
  /-- Saturating float-to-unsigned conversion.  Saturating rather than trapping
      so an out-of-range value clamps instead of aborting the process — the
      only consumer is a token id read back from a kernel that computed it as
      an exactly-representable integer. -/
  | fcvtToUint (dst : Val) (ty : ClifTy) (src : Val)
  | fcmp (dst : Val) (cond : FloatCC) (a b : Val)
  | bitcast (dst : Val) (ty : ClifTy) (src : Val)
  /-- Lane-wise `c ? a : b` on the *bits* of `c`, which is how a vector
      comparison's all-ones/all-zeros mask is consumed. -/
  | bitselect (dst c a b : Val)
  | ctz (dst a : Val)
  | popcnt (dst a : Val)
  | vhighBits (dst a : Val)
  | ibin (dst : Val) (k : IBin) (a b : Val)
  | ishift (dst : Val) (k : IShift) (a b : Val)
  | iun (dst : Val) (k : IUn) (a : Val)
  | fbin (dst : Val) (k : FBin) (a b : Val)
  | fun1 (dst : Val) (k : FUn) (a : Val)
  | fconv (dst : Val) (k : FConv) (ty : ClifTy) (a : Val)
  /-- `a * b + c`, rounded once. -/
  | fma (dst a b c : Val)
  | iext (dst : Val) (k : IExt) (ty : ClifTy) (a : Val)
  deriving BEq

/-- A finalized block -/
structure BlockData where
  ref : BlockRef
  params : List (Val × ClifTy)
  insts : List Inst


-- ---------------------------------------------------------------------------
-- The emitted program
-- ---------------------------------------------------------------------------

/-- One function of the emitted program.

    `index` is the `u0:N` it was compiled at, which is its position in the
    artifact and is not written out: `Prog.program` checks the two agree.
    `name` is the name a host calls it by, and a function without one is
    the program's own. -/
structure FuncData where
  index : Nat
  blocks : List BlockData
  name : Option String := none

-- ---------------------------------------------------------------------------
-- Serialization
--
-- The CBOR `base_types::clif` reads (see `Cbor`): externally tagged enums, an
-- instruction's fields in the order its Cranelift format gives them, newtypes
-- as their number. Field names and their order are the Rust ones; a
-- disagreement is a build failure, because the build re-encodes every
-- artifact it decodes.
-- ---------------------------------------------------------------------------

open Cbor

instance : ToCbor Val where
  cbor v := nat v.id
instance : ToCbor BlockRef where
  cbor b := nat b.id

instance : ToCbor ClifTy where
  cbor t := text <| match t with
    | .i8 => "i8" | .i16 => "i16" | .i32 => "i32" | .i64 => "i64"
    | .f32 => "f32" | .f64 => "f64" | .f32x4 => "f32x4" | .i8x16 => "i8x16"

/-! CLIF's own names. An instruction travels as a Cranelift instruction format
    and an opcode of it, spelled as CLIF spells it, with its controlling type
    named only where CLIF does not take it from an operand; condition codes and
    atomic operations by CLIF's names too. `base_types::clif::Inst` reads it. -/

def ICmpCond.clif : ICmpCond → String
  | .eq => "eq" | .ne => "ne" | .uge => "uge" | .ugt => "ugt" | .ule => "ule"
  | .ult => "ult" | .slt => "slt" | .sle => "sle" | .sgt => "sgt" | .sge => "sge"

def FloatCC.clif : FloatCC → String
  | .eq => "eq" | .ne => "ne" | .lt => "lt" | .le => "le" | .gt => "gt" | .ge => "ge"

def LoadKind.clif : LoadKind → String
  | .plain => "load" | .uload8 => "uload8" | .uload16 => "uload16" | .uload32 => "uload32"
  | .sload8 => "sload8" | .sload16 => "sload16" | .sload32 => "sload32"

def IBin.clif : IBin → String
  | .sdiv => "sdiv" | .urem => "urem" | .srem => "srem" | .smin => "smin" | .smax => "smax"
  | .umin => "umin" | .umax => "umax" | .umulhi => "umulhi" | .smulhi => "smulhi"

def IShift.clif : IShift → String
  | .sshr => "sshr" | .rotl => "rotl" | .rotr => "rotr"

def IUn.clif : IUn → String
  | .bnot => "bnot" | .iabs => "iabs" | .clz => "clz" | .bswap => "bswap" | .bitrev => "bitrev"

def FBin.clif : FBin → String
  | .fdiv => "fdiv" | .fcopysign => "fcopysign"

def FUn.clif : FUn → String
  | .sqrt => "sqrt" | .fabs => "fabs" | .ceil => "ceil" | .floor => "floor"
  | .trunc => "trunc" | .nearest => "nearest"

def IExt.clif : IExt → String
  | .reduce => "ireduce" | .uextend => "uextend" | .sextend => "sextend"

def FConv.clif : FConv → String
  | .toSint => "fcvt_to_sint_sat" | .fromUint => "fcvt_from_uint" | .demote => "fdemote"

def AtomicRmw.clif : AtomicRmw → String
  | .add => "add" | .sub => "sub" | .and => "and" | .nand => "nand" | .or => "or"
  | .xor => "xor" | .xchg => "xchg" | .umin => "umin" | .umax => "umax"
  | .smin => "smin" | .smax => "smax"

/-- The value a definition binds, if it binds one. -/
def Inst.dst? : Inst → Option Val
  | .iconst d .. | .iadd d .. | .isub d .. | .imul d .. | .udiv d .. | .ineg d ..
  | .ishl d .. | .ushr d .. | .band d .. | .bandNot d .. | .bor d .. | .bxor d ..
  | .ireduce32 d .. | .uextend64 d .. | .sextend64 d .. | .load d .. | .icmp d ..
  | .select d .. | .fconst d .. | .fadd d .. | .fsub d .. | .fmul d .. | .fmax d ..
  | .fmin d .. | .fpromote d .. | .splat d .. | .extractlane d .. | .fneg d ..
  | .fcvtFromSint d .. | .fcvtToUint d .. | .fcmp d .. | .bitcast d .. | .bitselect d ..
  | .ctz d .. | .popcnt d .. | .vhighBits d .. | .ibin d .. | .ishift d .. | .iun d ..
  | .fbin d .. | .fun1 d .. | .fconv d .. | .fma d .. | .iext d .. | .memory d | .funcAddr d .. => some d
  | .call d .. => d
  | .store .. | .istore8 .. | .jump .. | .brif .. | .ret .. | .storeTyped .. => none

/-- One past the largest value a function binds. -/
def FuncData.nextVal (f : FuncData) : Nat :=
  f.blocks.foldl (fun n b =>
    let n := b.params.foldl (fun n (v, _) => max n (v.id + 1)) n
    b.insts.foldl (fun n i => match Inst.dst? i with
      | some v => max n (v.id + 1)
      | none => n) n) 0

/-- Whether a function answers a value: whether a `ret` of it carries one. -/
def FuncData.answers (f : FuncData) : Bool :=
  f.blocks.any fun b => b.insts.any fun | .ret (some _) => true | _ => false

/-- A C function a program calls, as the artifact declares it once: its
    library, its symbol, and the signature to call it at. -/
structure Primitive where
  lib : String
  symbol : String
  sig : List ClifTy × Option ClifTy
  deriving BEq

/-- The primitive a callee is, if it is one. -/
def Callee.primitive? : Callee → Option Primitive
  | .ffi f => some ⟨Lib.base.name, f.cname, f.sig⟩
  | .ext e => some ⟨e.lib.name, e.symbol, e.sig⟩
  | _ => none

/-- Whether an atomic answers a value. -/
def Atomic.answers : Atomic → Bool
  | .load _ | .rmw .. | .cas _ => true
  | _ => false

/-- Whether calling `c` answers a value, given whether each function of the
    artifact does. -/
def Callee.answers (fns : List Bool) : Callee → Bool
  | .ffi f => f.sig.2.isSome
  | .ext e => e.sig.2.isSome
  | .local k => fns.getD k false
  | .native => true
  | .host _ r => r.isSome
  | .atomic a => a.answers

/-- The artifact's primitives, each once, in the order its functions first
    call them. A symbol two calls give different signatures is two entries,
    which the engine refuses: it is one function declared twice. -/
def primitivesOf (fs : List FuncData) : List Primitive :=
  fs.foldl (fun acc f => f.blocks.foldl (fun acc b => b.insts.foldl (fun acc i =>
    let c? := match i with
      | .call _ c _ => some c
      | .funcAddr _ c => some c
      | _ => none
    match c?.bind Callee.primitive? with
    | some p => if acc.contains p then acc else acc ++ [p]
    | none => acc) acc) acc) []

/-- What writing a function needs from the artifact around it. -/
structure Wire where
  /-- The artifact's primitives, which a call names by position. -/
  prims : List Primitive
  /-- Whether each function of the artifact answers a value. -/
  answers : List Bool

/-- A memory access's flags, as CLIF's words: `aligned` where it states its
    address aligned. -/
def flagsCbor (aligned : Bool) : W Unit := array (if aligned then ["aligned"] else []) text

/-- The type an instruction names, where CLIF does not take it from an operand. -/
def tyCbor (t : Option ClifTy) : W Unit := option cbor t

/-- Every value an instruction binds. -/
def resultsCbor (rs : List Val) : W Unit := array rs cbor

/-- A one-operand instruction (`Unary`). -/
def unaryCbor (d : Val) (op : String) (t : Option ClifTy) (a : Val) : W Unit :=
  variant "Unary" [resultsCbor [d], text op, tyCbor t, cbor a]

/-- A two-operand instruction (`Binary`), its type an operand's. -/
def binaryCbor (d : Val) (op : String) (a b : Val) : W Unit :=
  variant "Binary" [resultsCbor [d], text op, tyCbor none, cbor a, cbor b]

/-- A three-operand instruction (`Ternary`), its type an operand's. -/
def ternaryCbor (d : Val) (op : String) (a b c : Val) : W Unit :=
  variant "Ternary" [resultsCbor [d], text op, tyCbor none, cbor a, cbor b, cbor c]

/-- A store at no offset (`Store`): the value, then the address. -/
def storeCbor (op : String) (v a : Val) : W Unit :=
  variant "Store" [resultsCbor [], text op, tyCbor none, flagsCbor false, cbor v, cbor a, nat 0]

/-- A branch's destination: the block, and what it is passed. -/
def blockCallCbor (t : BlockRef) (args : List Val) : W Unit := do
  head 4 2; cbor t; array args cbor

/-- A parameter or return, passed as its type alone. -/
def paramCbor (t : ClifTy) : W Unit := do
  head 4 2; cbor t; array ([] : List String) text

/-- A signature: its convention, what it takes, and what it returns. -/
def sigCbor (convention : String) (params returns : List ClifTy) : W Unit :=
  struct
    [("convention", text convention),
     ("params", array params paramCbor),
     ("returns", array returns paramCbor)]

/-- A primitive, as the artifact's table declares it. -/
def primitiveCbor (p : Primitive) : W Unit :=
  struct
    [("lib", text p.lib),
     ("symbol", text p.symbol),
     ("signature", sigCbor "Host" p.sig.1 p.sig.2.toList)]

/-- A callee a direct call or a `func_addr` names: a function of the artifact,
    or a primitive, by position. -/
def calleeCbor (w : Wire) : Callee → W Unit
  | .local i => newtypeVariant "Function" (nat i)
  | c => match c.primitive? with
    | some p => match w.prims.findIdx? (· == p) with
      | some i => newtypeVariant "Primitive" (nat i)
      | none => throw s!"{p.lib}!{p.symbol} is not in the artifact's primitives"
    | none => throw "an indirect callee or an atomic names no function"

/-- What a call binds: the value the program names, or, where it names none
    for a callee that answers one, a fresh value, since CLIF names every
    result. -/
def callResults (d : Option Val) (answers : Bool) (fresh : Nat) : List Val × Nat :=
  match d with
  | some v => ([v], fresh)
  | none => if answers then ([⟨fresh⟩], fresh + 1) else ([], fresh)

/-- An atomic, which states its address aligned: the model faults one that
    is not, as AArch64 does. -/
def atomicCbor (rs : List Val) : Atomic → List Val → W Unit
  | .load t, [a] =>
      variant "LoadNoOffset" [resultsCbor rs, text "atomic_load", tyCbor (some t), flagsCbor true, cbor a]
  | .store _, [v, a] =>
      variant "StoreNoOffset" [resultsCbor rs, text "atomic_store", tyCbor none, flagsCbor true, cbor v, cbor a]
  | .rmw t k, [a, x] =>
      variant "AtomicRmw"
        [resultsCbor rs, text "atomic_rmw", tyCbor (some t), flagsCbor true, text k.clif, cbor a, cbor x]
  | .cas _, [a, e, x] =>
      variant "AtomicCas" [resultsCbor rs, text "atomic_cas", tyCbor none, flagsCbor true, cbor a, cbor e, cbor x]
  | .fence, [] => variant "NullAry" [resultsCbor rs, text "fence", tyCbor none]
  -- An atomic on the wrong number of arguments: written under a name that is
  -- no opcode, which the engine refuses at load.
  | _, _ => variant "NullAry" [resultsCbor rs, text "atomic on the wrong number of arguments", tyCbor none]

/-- A call through an address: the callee's signature under `convention`,
    then the address, then the arguments. -/
def indirectCbor (rs : List Val) (convention : String) (params : List ClifTy) (result : Option ClifTy) :
    List Val → W Unit
  | f :: args =>
      variant "CallIndirect"
        [resultsCbor rs, text "call_indirect", tyCbor none, sigCbor convention params result.toList,
         cbor f, array args cbor]
  | [] => throw "an indirect call without its address"

/-- One instruction other than a call, in the shape `base_types::clif::Inst`
    reads: a Cranelift format and an opcode of it.

    Stores and loads carry a byte offset on the Rust side that no builder here
    emits yet, so it is written as zero. No access states its address aligned
    but an atomic's: the input and output may start anywhere. -/
def Inst.plainCbor (w : Wire) : Inst → W Unit
  | .iconst d t v => variant "UnaryImm" [resultsCbor [d], text "iconst", tyCbor (some t), i64 v]
  | .memory d => variant "UnaryGlobalValue" [resultsCbor [d], text "symbol_value", tyCbor (some .i64), text "Arena"]
  | .funcAddr d c => variant "FuncAddr" [resultsCbor [d], text "func_addr", tyCbor (some .i64), calleeCbor w c]
  | .iadd d a b => binaryCbor d "iadd" a b
  | .isub d a b => binaryCbor d "isub" a b
  | .imul d a b => binaryCbor d "imul" a b
  | .udiv d a b => binaryCbor d "udiv" a b
  | .ineg d a => unaryCbor d "ineg" none a
  | .ishl d a b => binaryCbor d "ishl" a b
  | .ushr d a b => binaryCbor d "ushr" a b
  | .band d a b => binaryCbor d "band" a b
  | .bandNot d a b => binaryCbor d "band_not" a b
  | .bor d a b => binaryCbor d "bor" a b
  | .bxor d a b => binaryCbor d "bxor" a b
  | .ireduce32 d a => unaryCbor d "ireduce" (some .i32) a
  | .uextend64 d a => unaryCbor d "uextend" (some .i64) a
  | .sextend64 d a => unaryCbor d "sextend" (some .i64) a
  | .store v a => storeCbor "store" v a
  | .istore8 v a => storeCbor "istore8" v a
  | .storeTyped _ v a => storeCbor "store" v a
  -- `uload32` and `sload32` answer an `i64` whatever: CLIF takes their type
  -- from the address.
  | .load d op a =>
    let t := match op.kind with | .uload32 | .sload32 => none | _ => some op.ty
    variant "Load" [resultsCbor [d], text op.kind.clif, tyCbor t, flagsCbor op.aligned, cbor a, nat 0]
  | .icmp d c a b => variant "IntCompare" [resultsCbor [d], text "icmp", tyCbor none, text c.clif, cbor a, cbor b]
  | .fcmp d c a b => variant "FloatCompare" [resultsCbor [d], text "fcmp", tyCbor none, text c.clif, cbor a, cbor b]
  | .select d c a b => ternaryCbor d "select" c a b
  | .bitselect d c a b => ternaryCbor d "bitselect" c a b
  | .fma d a b c => ternaryCbor d "fma" a b c
  | .call .. => throw "a call is written with what it binds"
  | .jump t args => variant "Jump" [resultsCbor [], text "jump", tyCbor none, blockCallCbor t args]
  | .brif c tb ta eb ea =>
    variant "Brif" [resultsCbor [], text "brif", tyCbor none, cbor c, blockCallCbor tb ta, blockCallCbor eb ea]
  | .ret v => variant "MultiAry" [resultsCbor [], text "return", tyCbor none, array v.toList cbor]
  | .fconst d .f32 bits =>
    variant "UnaryIeee32" [resultsCbor [d], text "f32const", tyCbor none, nat (bits &&& 0xFFFFFFFF).toNat]
  | .fconst d _ bits => variant "UnaryIeee64" [resultsCbor [d], text "f64const", tyCbor none, nat bits.toNat]
  | .fadd d a b => binaryCbor d "fadd" a b
  | .fsub d a b => binaryCbor d "fsub" a b
  | .fmul d a b => binaryCbor d "fmul" a b
  | .fmax d a b => binaryCbor d "fmax" a b
  | .fmin d a b => binaryCbor d "fmin" a b
  | .fpromote d a => unaryCbor d "fpromote" (some .f64) a
  | .splat d t s => unaryCbor d "splat" (some t) s
  | .extractlane d s lane =>
    variant "BinaryImm8" [resultsCbor [d], text "extractlane", tyCbor none, cbor s, nat lane]
  | .fneg d a => unaryCbor d "fneg" none a
  | .fcvtFromSint d t s => unaryCbor d "fcvt_from_sint" (some t) s
  | .fcvtToUint d t s => unaryCbor d "fcvt_to_uint_sat" (some t) s
  | .bitcast d t s =>
    variant "LoadNoOffset" [resultsCbor [d], text "bitcast", tyCbor (some t), flagsCbor false, cbor s]
  | .ctz d a => unaryCbor d "ctz" none a
  | .popcnt d a => unaryCbor d "popcnt" none a
  | .vhighBits d a => unaryCbor d "vhigh_bits" (some .i32) a
  | .ibin d k a b => binaryCbor d k.clif a b
  | .ishift d k a b => binaryCbor d k.clif a b
  | .iun d k a => unaryCbor d k.clif none a
  | .fbin d k a b => binaryCbor d k.clif a b
  | .fun1 d k a => unaryCbor d k.clif none a
  | .fconv d k t a => unaryCbor d k.clif (some t) a
  | .iext d k t a => unaryCbor d k.clif (some t) a

/-- One instruction, given the next fresh value, answering the one after
    what it used: a call binds a fresh value for a result the program does not
    name. -/
def Inst.toCbor (w : Wire) (fresh : Nat) : Inst → W Nat
  | .call d (.atomic k) args => do
    let (rs, fresh) := callResults d k.answers fresh
    atomicCbor rs k args
    pure fresh
  | .call d .native args => do
    let (rs, fresh) := callResults d true fresh
    indirectCbor rs "Arch" [.i64, .i64, .i64, .i64] (some .i64) args
    pure fresh
  | .call d (.host ps r) args => do
    let (rs, fresh) := callResults d r.isSome fresh
    indirectCbor rs "Host" ps r args
    pure fresh
  | .call d c args => do
    let (rs, fresh) := callResults d (c.answers w.answers) fresh
    variant "Call" [resultsCbor rs, text "call", tyCbor none, calleeCbor w c, array args cbor]
    pure fresh
  | i => do
    i.plainCbor w
    pure fresh

/-- A block: its name, its parameters, its instructions. -/
def BlockData.toCbor (w : Wire) (b : BlockData) (fresh : Nat) : W Nat := do
  head 5 3
  text "reference"; cbor b.ref
  text "params"; array b.params fun (v, t) => do head 4 2; cbor v; cbor t
  text "insts"; head 4 b.insts.length
  b.insts.foldlM (fun fresh i => i.toCbor w fresh) fresh

/-- A function: its name if it is an entry point, its signature, and its
    blocks. A named function's signature is `BaseFunction`'s; any other function
    takes what its entry block does and answers an `i64` where a `ret` of it
    carries one. -/
def FuncData.toCbor (w : Wire) (f : FuncData) : W Unit := do
  let params := (f.blocks.head?.map fun b => b.params.map (·.2)).getD []
  head 5 4
  text "name"; option text f.name
  text "signature"
  if f.name.isSome then sigCbor "Host" [.i64, .i64, .i64, .i64] [.i64]
  else sigCbor "Host" params (if f.answers then [.i64] else [])
  text "stack_slots"; array ([] : List Nat) nat
  text "blocks"; head 4 f.blocks.length
  let _ ← f.blocks.foldlM (fun fresh b => b.toCbor w fresh) f.nextVal

end IR

end AlgorithmLib
