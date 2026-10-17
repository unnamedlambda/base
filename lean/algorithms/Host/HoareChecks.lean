module
public import Host.StaticChecks
meta import Host.StaticChecks
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

/-!
# A loop of launches the checker proves once

`byInput` launches on one stream as many times as the first input byte says,
then, with `synced`, waits for the stream before downloading. The checker
carries the stream's readiness round the loop: at the head, the host knows
every access, or the stream does, and either lets the next launch on the
stream go; after the loop, the wait makes the host know them all again, so the
download is not a race.
-/

@[expose] public section

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgHoareChecks

open HProgCudaCorpus (SRC KERNEL MEM image)
open HProgStaticChecks (BA launchOn)

/-- Launch on one stream as many times as the first input byte says; with
    `synced`, wait for the stream before downloading. -/
def byInput (synced : Bool) : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  cudaInit ptr
  let c ← cudaCtx ptr
  let s1 ← c.streamCreate
  let bX ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bX (← iconst64 SRC) (← iconst64 64)
  storeI32 bX (← iadd ptr (← iconst64 BA))
  let n ← uload8_64 (← dataPtr)
  forLoop n fun _ => do
    let _ ← launchOn ptr c s1
  if synced then
    let _ ← c.streamSync s1
  let _ ← ffi .cudaDownload %[c.ptr, bX, out, ← iconst64 64]
  cudaCleanup ptr

open HProgStaticChecks (Device0 args check isMisuse runOn)

/-- **No run misuses a foreign call, however many times the input says to
    launch.** -/
theorem byInput_no_misuse (cfg : Sem.Cfg) (w : Sem.World) (h : Device0 w) (m : String) :
    Sem.run cfg args w (Prog.emit (byInput true)) ≠ .misuse m := by
  check_body

#guard check (byInput true)
#guard !check (byInput false)

-- Without the wait the download races the launches, on every input that
-- launches at all.
#guard isMisuse (runOn (byInput false) 1)
#guard isMisuse (runOn (byInput false) 200)
#guard !isMisuse (runOn (byInput false) 0)
#guard !isMisuse (runOn (byInput true) 0)
#guard !isMisuse (runOn (byInput true) 200)

end HProgHoareChecks
