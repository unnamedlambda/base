module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What a call may write

A `Frame` says which memory a call may disturb, and only that: a call writes
at an address built from its own arguments and leaves the rest of memory
alone. It does not say what it writes.

Every C library function has one (`Ext.frame`). None is taken on trust:
`ExtContracts` proves each model writes only inside its frame
(`ext_respects_frame`).
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.HProg.Sem

/-- The memory a call may write. Argument positions are zero-based and count
    the callee's own parameters, so they line up with the `args` list. -/
inductive Frame where
  /-- Writes no addressable memory. The result, a file, a socket or the device
      is the whole effect. -/
  | none
  /-- Writes `[args[dst], args[dst] + args[len])`. -/
  | at (dst len : Nat)
  /-- Writes `[args[base] + args[off], … + args[len])` — the shared-memory form,
      where the caller passes an offset rather than a pointer. -/
  | atOff (base off len : Nat)
  /-- Writes exactly `n` bytes at `args[dst]`. -/
  | fixed (dst n : Nat)
  /-- Writes exactly `n` bytes at `args[a]` and `n` at `args[b]`: two out
      parameters of one width. -/
  | fixed2 (a b n : Nat)
  /-- Writes at `args[dst]`, at most `size` bytes for each of `args[count]`
      records, read as an `i32`: fewer may be written, never more. -/
  | atRecords (dst count size : Nat)
  /-- Writes at `args[dst]`, for a length decided by state the call reads
      rather than by any argument. -/
  | dataDependent (dst : Nat)
  /-- Writes at `args[a]` and `args[b]`, both data-dependent. -/
  | dataDependent2 (a b : Nat)
  /-- May write anywhere: the call runs code this model cannot read, so no byte
      of memory is known to survive it. -/
  | anywhere
  deriving Repr, BEq

/-- The memory a library function's call may write for the engine: none. The
    functions are `Lib.Ht`'s and `Lib.Math`'s, the program's own, and what
    they write is their bodies'. -/
def frame : IR.LibFn → Frame := fun _ => .none

/-- **What a C library function may write**, by its parameters. The copies
    write their destination for their count; the driver writes each object it
    creates through the one pointer it is handed, and a download its
    destination. The probe and everything else write no host memory. -/
def _root_.AlgorithmLib.IR.Ext.frame : IR.Ext → Frame
  | .present _ => .none
  | .c .memcpy | .c .memmove | .c .memset => .at 0 2
  | .c .strlen | .c .calloc | .c .free => .none
  | .cuda .deviceGet => .fixed 0 4
  | .cuda .primaryCtxRetain | .cuda .memAlloc | .cuda .moduleLoadData
  | .cuda .moduleGetFunction | .cuda .streamCreate | .cuda .eventCreate
  | .cuda .memAllocHost | .cuda .graphInstantiate => .fixed 0 8
  | .cuda .endCapture => .fixed 1 8
  | .cuda .eventElapsedTime => .fixed 0 4
  | .cuda .memGetInfo => .fixed2 0 1 8
  | .cuda .memcpyDtoH | .cuda .memcpyDtoHAsync => .at 0 2
  | .cuda .init | .cuda .primaryCtxRelease | .cuda .ctxSetCurrent | .cuda .ctxSynchronize
  | .cuda .memFree | .cuda .memcpyHtoD | .cuda .memcpyDtoD | .cuda .memsetD8
  | .cuda .moduleUnload | .cuda .launchKernel | .cuda .streamSynchronize
  | .cuda .streamDestroy | .cuda .eventRecord | .cuda .streamWaitEvent
  | .cuda .eventSynchronize | .cuda .eventDestroy | .cuda .memFreeHost
  | .cuda .memcpyHtoDAsync | .cuda .beginCapture | .cuda .graphUpload | .cuda .graphLaunch
  | .cuda .graphExecDestroy | .cuda .graphDestroy => .none
  | .cublas .create => .fixed 0 8
  | .cublas _ => .none
  | .cpu _ => .none
  -- A file read or a line of standard input lands in memory; the engine's
  -- other functions write none.
  | .base .fileReadToPtr => .at 2 4
  | .base .stdinReadline => .at 0 1
  | .base _ => .none
  -- The serial library writes a port's name and what a read takes; the USB
  -- library what a transfer brings back from the device.
  | .serial .name | .serial .read => .at 1 2
  | .serial _ => .none
  | .usb .control => .at 5 6
  | .usb .bulk | .usb .interrupt => .at 2 3
  | .usb _ => .none
  -- wgpu writes host memory only where a texture is acquired: the
  -- `WGPUSurfaceTexture` it is handed.
  | .wgpu .bufferRead => .at 3 4
  | .wgpu _ => .none
  -- The window library writes the record it polls.
  | .window .poll => .fixed 1 32
  | .window _ => .none

-- ---------------------------------------------------------------------------
-- What one program assumes
-- ---------------------------------------------------------------------------

/-- What a program assumes by calling `c`, named: an import's declared frame,
    and for machine code `anywhere`, since it is the program's own code and
    nothing here reads what it does. A call to one of the program's own
    functions assumes nothing: its body is part of the program. -/
def calleeFrame : IR.Callee → Option (String × Option Frame)
  | .lib f   => some (f.cname, some (frame f))
  | .local _ => none
  | .native  => some ("native code", some .anywhere)
  -- A host's function likewise: nothing here reads what it does.
  | .host _ _ => some ("host function", some .anywhere)
  -- An atomic is the program's own memory access, like a load or a store.
  | .atomic _ => none
  -- A C library function is foreign code, with its frame from the table.
  | .ext e => some (s!"{e.lib.name}!{e.symbol}", some e.frame)

/-- **The FFI a program actually assumes.**

    Not every entry point that exists — only the callees this body calls, which
    `callsOf` already reads off the term. A program that computes and stores
    assumes nothing at all; the histogram assumes two.

    This is what makes the FFI part of the trusted base per-program and usually
    near-empty, instead of a fixed eighty-item liability every artifact
    carries. -/
def footprint (c : Code) : List (String × Option Frame) :=
  (callsOf c).foldl
    (fun acc fn =>
      match calleeFrame fn with
      | none => acc
      | some (n, f) => if acc.any (·.1 == n) then acc else acc ++ [(n, f)])
    []

/-- A program whose every callee has a declared frame — otherwise something it
    calls has no statement about what it may write, and no proof about that
    program can be complete. -/
def footprintComplete (c : Code) : Bool :=
  (footprint c).all (·.2.isSome)

/-- Rendered for the build log, so each artifact reports its own assumptions. -/
def footprintReport (c : Code) : String :=
  match footprint c with
  | [] => "assumes no FFI"
  | fs => s!"assumes {fs.length} FFI frame(s): " ++
          String.intercalate ", " (fs.map fun (n, f) =>
            match f with | some _ => n | none => n ++ " (NO FRAME)")

end AlgorithmLib.HProg
