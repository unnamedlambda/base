module
import all Init.Data.Repr
@[expose] public section

/-!
# The hash table's layout

Where `Lib.Ht`'s context block keeps each field, how long an entry is, and the
two multipliers its hash mixes with. Below both the library that writes the
table (`Surface.LibHt`) and the statement of what memory holding one is
(`Host.HtRep`), so each reads the same numbers and neither imports the other.
-/

namespace AlgorithmLib.LibHt

def CREATED : Nat := 0x00
def COUNT : Nat := 0x08
def ECAP : Nat := 0x10
def ENTRIES : Nat := 0x18
def ICAP : Nat := 0x20
def INDEX : Nat := 0x28
/-- Eight bytes of scratch: the value an increment inserts. -/
def SCRATCH : Nat := 0x30
/-- The arena every key and value lives in: address, bytes used, capacity. -/
def ARENA : Nat := 0x38
def AUSED : Nat := 0x40
def ACAP : Nat := 0x48
def SIZE : Nat := 0x50

/-- An entry: where its key and its value sit in the arena, and their
    lengths. Offsets, not addresses, since the arena moves when it grows. -/
def ENTRY : Nat := 32

def MIX : Int := 0x9E3779B97F4A7C15 - 2 ^ 64
def FIN : Int := 0xBF58476D1CE4E5B9 - 2 ^ 64

end AlgorithmLib.LibHt
