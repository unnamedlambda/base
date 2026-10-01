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
    library, wgpu and the engine's window, CPU, serial and USB libraries are
    built into it (`Lib.builtin`): always there, bound to the functions the
    engine links. -/
inductive Lib where
  | c | cuda | cublas | usb | window | cpu | wgpu | serial
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

def Lib.name : Lib → String
  | .c => "c" | .cuda => "cuda" | .cublas => "cublas" | .usb => "usb"
  | .window => "window" | .cpu => "cpu" | .wgpu => "wgpu" | .serial => "serial"

def Lib.files : Lib → List (String × List String)
  | .cuda => [("linux", ["libcuda.so.1", "libcuda.so"]), ("windows", ["nvcuda.dll"])]
  | .cublas => [("linux", ["libcublas.so.12", "libcublas.so"]), ("windows", ["cublas64_12.dll"])]
  | .c | .usb | .window | .cpu | .wgpu | .serial => []

/-- Built into the engine: present on every platform, with no file to find. -/
def Lib.builtin : Lib → Bool
  | .c | .usb | .window | .cpu | .wgpu | .serial => true
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
    crate): the devices the system lists and what each is, and a device
    opened by its place, its interfaces claimed and released, and its
    control, bulk and interrupt transfers, each waited for. -/
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
    | .claim | .release => ([.i64, .i32], some .i32)
    | .control => ([.i64, .i32, .i32, .i32, .i32, .i64, .i32, .i32], some .i64)
    | .bulk | .interrupt => ([.i64, .i32, .i32, .i64, .i64, .i32], some .i64)

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
  /-- `notrap aligned`; the float and vector accessors set it. -/
  notrapAligned : Bool := false
  deriving Repr, BEq

/-- One runtime entry point. -/
inductive Ffi where
  | fileRead | fileWrite | fileReadToPtr | fileWriteFromPtr
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
  -- Appended, and appending is the rule: a callee's id is its position in
  -- `all`, so inserting one renumbers every shipped artifact's calls.
  | cudaPinnedPtrAt | cudaMemInfoFree | cudaMemInfoTotal | cublasGemmExBf16
  | cublasGemmStridedBatchedExBf16
  | nativeLoad | nativeFree | nativeArch | cpuHas
  | fileCreateDirAll | threadStart | threadFinish
  | memLock | memUnlock | memAdviseHuge | threadPriority
  deriving Repr, DecidableEq, Inhabited, Lean.ToExpr

namespace Ffi

/-- The C symbol the JIT resolves. -/
def cname : Ffi → String
  | .fileRead => "cl_file_read"
  | .fileWrite => "cl_file_write"
  | .fileReadToPtr => "cl_file_read_to_ptr"
  | .fileWriteFromPtr => "cl_file_write_from_ptr"
  | .stdinReadline => "cl_stdin_readline"
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

/-- Parameters and result, exactly as `base/src/ffi/` takes them. -/
def sig : Ffi → List ClifTy × Option ClifTy
  | .fileRead => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .fileWrite => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .fileReadToPtr => ([.i64, .i64, .i64, .i64], some .i64)
  | .fileWriteFromPtr => ([.i64, .i64, .i64, .i64], some .i64)
  | .stdinReadline => ([.i64, .i64, .i64], some .i64)
  | .stdoutWrite => ([.i64, .i64, .i64], some .i64)
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
  | .nativeFree => ([.i64], some .i32)
  | .nativeArch => ([], some .i32)
  | .cpuHas => ([.i64], some .i32)
  -- `(path)`: the directory at the path, and every one missing on the way;
  -- `0`, or `-1` when that cannot be done.
  | .fileCreateDirAll => ([.i64], some .i64)
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
  -- What wgpu answers through a callback, waited for: `(instance)` its
  -- high-performance adapter, or `0`; `(adapter)` a device, or `0`;
  -- `(device, buffer, offset, dst, size)` the buffer's bytes copied to `dst`
  -- through a mapping, `0` or `-1`; `(device)` the innermost error scope
  -- popped, `0` when it caught nothing and `-1` otherwise.

def params (f : Ffi) : List ClifTy := f.sig.1
def result (f : Ffi) : Option ClifTy := f.sig.2

/-- Every entry point, in the order that fixes the ids. -/
def all : List Ffi :=
  [.fileRead, .fileWrite, .fileReadToPtr, .fileWriteFromPtr,
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
   .memLock, .memUnlock, .memAdviseHuge, .threadPriority]

/-- The registry of entry points: `all` misses none and repeats none, so the
    ids below are a numbering of `Ffi` and not merely of a list beside it. -/
instance : Enum Ffi where
  all := all
  complete f := by cases f <;> decide
  nodup := by decide

/-- The callee id every artifact carries for `f`. -/
def id (f : Ffi) : Nat := all.idxOf f

/-- Distinct entry points carry distinct ids. -/
theorem id_injective {f g : Ffi} (h : f.id = g.id) : f = g := Enum.index_injective h

/-- The entry point a name resolves to, if it is one. -/
def ofCname (s : String) : Option Ffi := all.find? (·.cname == s)

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
  /-- An entry point of the engine, resolved by its `cname` through the JIT's
      symbol table. -/
  | ffi (f : Ffi)
  /-- Another function of this same program, by its `u0:N` position. -/
  | local (index : Nat)
  /-- Machine code at the address in the call's first argument --- one
      `nativeLoad` answered --- called on the other four under the
      architecture's C convention (System V on x86-64 whatever the OS), and
      answering an `i64`. A plain indirect call: nothing of the engine's runs
      between the caller and the code. -/
  | native
  /-- A CLIF atomic instruction, which the engine emits in place of a call. -/
  | atomic (a : Atomic)
  /-- A function of a C library, called directly. -/
  | ext (e : Ext)
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- A single CLIF instruction -/
inductive Inst where
  | iconst (dst : Val) (ty : ClifTy) (value : Int)
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
    `entryName` is the name a host calls it by, and a function without one is
    the program's own. -/
structure FuncData where
  index : Nat
  blocks : List BlockData
  entryName : Option String := none

-- ---------------------------------------------------------------------------
-- Serialization
--
-- The CBOR `base_types::clif` reads (see `Cbor`): externally tagged enums,
-- tuple variants whose fields are in constructor order, newtypes as their
-- number. Field names and their order are the Rust ones; a disagreement is a
-- build failure, because the build re-encodes every artifact it decodes.
-- ---------------------------------------------------------------------------

open Cbor

instance : ToCbor Val where
  cbor v := nat v.id
instance : ToCbor BlockRef where
  cbor b := nat b.id

instance : ToCbor ClifTy where
  cbor t := text <| match t with
    | .i8 => "I8" | .i16 => "I16" | .i32 => "I32" | .i64 => "I64"
    | .f32 => "F32" | .f64 => "F64" | .f32x4 => "F32x4" | .i8x16 => "I8x16"

instance : ToCbor ICmpCond where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .uge => "Uge" | .ugt => "Ugt" | .ule => "Ule"
    | .ult => "Ult" | .slt => "Slt" | .sle => "Sle" | .sgt => "Sgt" | .sge => "Sge"

instance : ToCbor FloatCC where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .lt => "Lt" | .le => "Le" | .gt => "Gt" | .ge => "Ge"

instance : ToCbor LoadKind where
  cbor k := text <| match k with
    | .plain => "Plain" | .uload8 => "Uload8" | .uload16 => "Uload16"
    | .uload32 => "Uload32" | .sload8 => "Sload8" | .sload16 => "Sload16"
    | .sload32 => "Sload32"

instance : ToCbor IBin where
  cbor k := text <| match k with
    | .sdiv => "Sdiv" | .urem => "Urem" | .srem => "Srem" | .smin => "Smin"
    | .smax => "Smax" | .umin => "Umin" | .umax => "Umax" | .umulhi => "Umulhi"
    | .smulhi => "Smulhi"

instance : ToCbor IShift where
  cbor k := text <| match k with
    | .sshr => "Sshr" | .rotl => "Rotl" | .rotr => "Rotr"

instance : ToCbor IUn where
  cbor k := text <| match k with
    | .bnot => "Bnot" | .iabs => "Iabs" | .clz => "Clz"
    | .bswap => "Bswap" | .bitrev => "Bitrev"

instance : ToCbor FBin where
  cbor k := text <| match k with
    | .fdiv => "Fdiv" | .fcopysign => "Fcopysign"

instance : ToCbor FUn where
  cbor k := text <| match k with
    | .sqrt => "Sqrt" | .fabs => "Fabs" | .ceil => "Ceil" | .floor => "Floor"
    | .trunc => "Trunc" | .nearest => "Nearest"

instance : ToCbor IExt where
  cbor k := text <| match k with
    | .reduce => "Reduce" | .uextend => "Uextend" | .sextend => "Sextend"

instance : ToCbor FConv where
  cbor k := text <| match k with
    | .toSint => "ToSint" | .fromUint => "FromUint" | .demote => "Demote"

instance : ToCbor LoadOp where
  cbor op := struct
    [("kind", cbor op.kind),
     ("ty", cbor op.ty),
     ("notrap_aligned", bool op.notrapAligned)]

/-- An import travels as the symbol the engine resolves, which is the name
    `Ffi` already fixes; a defined function as its position. The constructor
    does not travel: the wire says what the engine needs to resolve, and the
    restriction to entry points the engine has belongs on this side of it. -/
instance : ToCbor AtomicRmw where
  cbor k := text <| match k with
    | .add => "Add" | .sub => "Sub" | .and => "And" | .nand => "Nand" | .or => "Or"
    | .xor => "Xor" | .xchg => "Xchg" | .umin => "Umin" | .umax => "Umax"
    | .smin => "Smin" | .smax => "Smax"

instance : ToCbor Atomic where
  cbor
    | .load t => newtypeVariant "Load" (cbor t)
    | .store t => newtypeVariant "Store" (cbor t)
    | .rmw t k => variant "Rmw" [cbor t, cbor k]
    | .cas t => newtypeVariant "Cas" (cbor t)
    | .fence => text "Fence"

/-- A C function travels with everything the engine needs to bind it: the
    library and its files, the symbol, and the signature to call it at. -/
def Ext.toCbor (e : Ext) : W Unit :=
  struct
    [("lib", text e.lib.name),
     ("files", array e.lib.files fun (os, fs) => do head 4 2; text os; array fs text),
     ("symbol", text e.symbol),
     ("params", array e.sig.1 cbor),
     ("result", option cbor e.sig.2)]

instance : ToCbor Callee where
  cbor
    | .ffi f   => newtypeVariant "Import" (text f.cname)
    | .local i => newtypeVariant "Local" (nat i)
    | .native  => text "Native"
    | .atomic a => newtypeVariant "Atomic" (cbor a)
    | .ext e => newtypeVariant "Extern" (Ext.toCbor e)

/-- One instruction, in the shape `base_types::clif::Inst` reads.

    Stores and loads carry a byte offset on the Rust side that no builder here
    emits yet, so it is written as zero. -/
def Inst.toCbor : Inst → W Unit
  | .iconst d t v => variant "Iconst" [cbor d, cbor t, i64 v]
  | .iadd d a b => variant "Iadd" [cbor d, cbor a, cbor b]
  | .isub d a b => variant "Isub" [cbor d, cbor a, cbor b]
  | .imul d a b => variant "Imul" [cbor d, cbor a, cbor b]
  | .udiv d a b => variant "Udiv" [cbor d, cbor a, cbor b]
  | .ineg d a => variant "Ineg" [cbor d, cbor a]
  | .ishl d a b => variant "Ishl" [cbor d, cbor a, cbor b]
  | .ushr d a b => variant "Ushr" [cbor d, cbor a, cbor b]
  | .band d a b => variant "Band" [cbor d, cbor a, cbor b]
  | .bandNot d a b => variant "BandNot" [cbor d, cbor a, cbor b]
  | .bor d a b => variant "Bor" [cbor d, cbor a, cbor b]
  | .bxor d a b => variant "Bxor" [cbor d, cbor a, cbor b]
  | .ireduce32 d a => variant "Ireduce32" [cbor d, cbor a]
  | .uextend64 d a => variant "Uextend64" [cbor d, cbor a]
  | .sextend64 d a => variant "Sextend64" [cbor d, cbor a]
  | .store v a => variant "Store" [cbor v, cbor a, nat 0]
  | .istore8 v a => variant "Istore8" [cbor v, cbor a, nat 0]
  | .load d op a => variant "Load" [cbor d, cbor op, cbor a, nat 0]
  | .icmp d c a b => variant "Icmp" [cbor d, cbor c, cbor a, cbor b]
  | .select d c a b => variant "Select" [cbor d, cbor c, cbor a, cbor b]
  | .call d f args =>
    variant "Call" [option cbor d,
                   cbor f, array args cbor]
  | .jump t args => variant "Jump" [cbor t, array args cbor]
  | .brif c tb ta eb ea =>
    variant "Brif" [cbor c, cbor tb, array ta cbor,
                   cbor eb, array ea cbor]
  -- A newtype variant, so the payload sits directly under the tag rather than
  -- in an array the way the tuple variants above do.
  | .ret v => newtypeVariant "Ret" (option cbor v)
  | .fconst d t bits => variant "Fconst" [cbor d, cbor t, nat bits.toNat]
  | .fadd d a b => variant "Fadd" [cbor d, cbor a, cbor b]
  | .fsub d a b => variant "Fsub" [cbor d, cbor a, cbor b]
  | .fmul d a b => variant "Fmul" [cbor d, cbor a, cbor b]
  | .fmax d a b => variant "Fmax" [cbor d, cbor a, cbor b]
  | .fmin d a b => variant "Fmin" [cbor d, cbor a, cbor b]
  | .fpromote d a => variant "Fpromote" [cbor d, cbor a]
  | .splat d t s => variant "Splat" [cbor d, cbor t, cbor s]
  | .extractlane d s lane => variant "Extractlane" [cbor d, cbor s, nat lane]
  | .storeTyped t v a => variant "StoreTyped" [cbor t, cbor v, cbor a, nat 0]
  | .fneg d a => variant "Fneg" [cbor d, cbor a]
  | .fcvtFromSint d t s => variant "FcvtFromSint" [cbor d, cbor t, cbor s]
  | .fcvtToUint d t s => variant "FcvtToUint" [cbor d, cbor t, cbor s]
  | .fcmp d c a b => variant "Fcmp" [cbor d, cbor c, cbor a, cbor b]
  | .bitcast d t s => variant "Bitcast" [cbor d, cbor t, cbor s]
  | .bitselect d c a b => variant "Bitselect" [cbor d, cbor c, cbor a, cbor b]
  | .ctz d a => variant "Ctz" [cbor d, cbor a]
  | .popcnt d a => variant "Popcnt" [cbor d, cbor a]
  | .vhighBits d a => variant "VhighBits" [cbor d, cbor a]
  | .ibin d k a b => variant "Ibin" [cbor d, cbor k, cbor a, cbor b]
  | .ishift d k a b => variant "Ishift" [cbor d, cbor k, cbor a, cbor b]
  | .iun d k a => variant "Iun" [cbor d, cbor k, cbor a]
  | .fbin d k a b => variant "Fbin" [cbor d, cbor k, cbor a, cbor b]
  | .fun1 d k a => variant "Fun1" [cbor d, cbor k, cbor a]
  | .fconv d k t a => variant "Fconv" [cbor d, cbor k, cbor t, cbor a]
  | .fma d a b c => variant "Fma" [cbor d, cbor a, cbor b, cbor c]
  | .iext d k t a => variant "Iext" [cbor d, cbor k, cbor t, cbor a]



instance : ToCbor Inst where
  cbor := Inst.toCbor

instance : ToCbor BlockData where
  cbor b := struct
    [("reference", cbor b.ref),
     ("params", array b.params fun (v, t) => do head 4 2; cbor v; cbor t),
     ("insts", array b.insts cbor)]


instance : ToCbor FuncData where
  cbor f := struct
    [("entry_name", option text f.entryName),
     ("blocks", array f.blocks cbor)]

end IR

end AlgorithmLib
