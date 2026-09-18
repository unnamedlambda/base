import AlgorithmLib.Core
import AlgorithmLib.Layout
import AlgorithmLib.IR

/-!
# The runtime surface, as data

Every entry point the runtime exports, one constructor each. What a callee is —
its C symbol, its signature, how it resolves — are total functions of the
constructor, so a signature is written in exactly one place and two files
cannot describe the same symbol differently.

The constructors live in `ClifData`, beside the instructions, because a call
names one: `Callee.ffi` takes an `Ffi`, so an artifact calling something the
engine does not provide is unrepresentable rather than refused at load. What
travels is the `cname`; `all` fixes only the order `id` reports, which nothing
shipped depends on.

The executable contracts — what a call *does*, transcribed from
`base/src/ffi/` — are in `HProgSem`; which memory a call may write is in
`HProgFrames`. This file is only who exists and how to call them.
-/

namespace AlgorithmLib.IR


namespace FFI

/-- A named group of entry points, so a body can be checked against the part of
    the table it uses. -/
inductive Bundle where
  | fileIO | gpu | window | lmdb | ht | math | thread | cuda | cublas
  deriving Repr, BEq

end FFI

/-- The bundle an entry point belongs to. -/
def Ffi.bundle : Ffi → FFI.Bundle
  | .fileRead | .fileWrite | .fileReadToPtr | .fileWriteFromPtr
  | .stdinReadline | .stdoutWrite => .fileIO
  | .gpuInit | .gpuCreateBuffer | .gpuCreatePipeline | .gpuUpload | .gpuDownload
  | .gpuDispatch | .gpuCleanup | .gpuUploadPtr | .gpuDownloadPtr => .gpu
  | .windowInit | .windowOpen | .windowPoll | .windowPresentGpuBuffer
  | .windowCleanup => .window
  | .lmdbInit | .lmdbOpen | .lmdbBeginWriteTxn | .lmdbPut | .lmdbCommitWriteTxn
  | .lmdbCursorScan | .lmdbCleanup => .lmdb
  | .htCreate | .htLookup | .htInsert | .htIncrement | .htCount | .htGetEntry
  | .htCleanup | .htInit => .ht
  | .sinf | .cosf | .powf => .math
  | .threadInit | .threadSpawn | .threadJoin | .threadCleanup => .thread
  | .cublasSgemv | .cublasSgemvOnStream | .cublasSgemm | .cublasSgemmOnStream
  | .cublasPtrArray | .cublasSgemmBatchedOnStream => .cublas
  | _ => .cuda

/-- The index of the entry point every application emits as `u0:1`, with `u0:0`
    reserved as a no-op stub. This is the number a host calls. -/
def mainFnIdx : UInt32 := u32 1

end AlgorithmLib.IR
