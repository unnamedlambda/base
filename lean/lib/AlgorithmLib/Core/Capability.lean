module
public import AlgorithmLib.Core.ClifData
meta import AlgorithmLib.Core.ClifData
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Capabilities: what CLIF cannot spell

**The gap table.** CLIF spells integer and float arithmetic, 128-bit vectors,
memory, atomics and control flow, identically on every target. What it does not
spell is listed in `Gap`, each with its one home: machine code the program
carries (`Callee.native`, chosen at run time by `nativeArch` and `cpuHas`),
which is the only place target-specific code may live. Floating-point modes are
the one gap that also sits outside the model: the float semantics assumes
round-to-nearest with subnormals kept, so a leaf that changes the modes must
restore them before it returns.
-/

namespace AlgorithmLib.Capability

open AlgorithmLib.IR

/-- What CLIF cannot spell. -/
inductive Gap where
  /-- Vectors wider than 128 bits: AVX2, AVX-512, SVE. -/
  | wideSimd
  /-- Prefetch hints and non-temporal stores. -/
  | cacheControl
  /-- CRC32, AES, SHA and carry-less multiplication. -/
  | cryptoCrc
  /-- Bit deposit and extract (`pdep`, `pext`). -/
  | bitDeposit
  /-- The cycle counter (`rdtsc`, `cntvct_el0`). -/
  | cycleCounter
  /-- Rounding modes, flush-to-zero and denormals-are-zero. -/
  | fpModes
  deriving Repr, BEq, DecidableEq, Inhabited

def Gap.all : List Gap := [.wideSimd, .cacheControl, .cryptoCrc, .bitDeposit, .cycleCounter, .fpModes]

/-- Where a gap's code lives. -/
inductive Home where
  /-- Machine code the program carries, loaded by `nativeLoad` and called with
      `Callee.native`; the program picks the bytes by `nativeArch` and
      `cpuHas`, and carries a CLIF path for a machine without the feature. -/
  | leaf
  /-- A leaf, and outside the float model: the leaf restores the modes it
      changes before it returns. -/
  | leafRestoring
  deriving Repr, BEq, DecidableEq

/-- **The decision, per gap.** Every gap is a leaf; no gap is a new CLIF
    instruction, a new engine entry point, or an OS call. -/
def Gap.home : Gap → Home
  | .fpModes => .leafRestoring
  | _ => .leaf

/-- The `cpuHas` names that say a machine has the instructions, by
    architecture: x86-64's CPUID names and AArch64's feature names. -/
def Gap.features : Gap → List (String × List String)
  | .wideSimd => [("x86_64", ["avx2", "avx512f"]), ("aarch64", ["sve"])]
  | .cacheControl => [("x86_64", ["sse2"]), ("aarch64", [])]
  | .cryptoCrc => [("x86_64", ["sse4.2", "aes", "sha", "pclmulqdq"]),
                   ("aarch64", ["crc", "aes", "sha2", "pmull"])]
  | .bitDeposit => [("x86_64", ["bmi2"]), ("aarch64", [])]
  | .cycleCounter => [("x86_64", []), ("aarch64", [])]
  | .fpModes => [("x86_64", []), ("aarch64", [])]

end AlgorithmLib.Capability
