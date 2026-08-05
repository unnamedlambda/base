import AlgorithmLib.LZ4Simt
set_option maxRecDepth 8192

namespace AlgorithmLib.LZ4Simt
open AlgorithmLib

/-- Special/pre-set registers of the initial state. -/
def rTidX : String := "%tid.x"
def rCtaX : String := "%ctaid.x"
def rNtidX : String := "%ntid.x"
def rInPtr : String := "in_ptr"
def rOutPtr : String := "out_ptr"

/-- Semantically-named persistent registers (the ones the RSim invariants couple
    to `warpFind`'s state).  Named rather than fresh so the invariants are
    statable; the machine treats names opaquely so this is free. -/
def rLane      : String := "lane"
def rPL0       : String := "pL0"
def rInBase    : String := "inBase"
def rOutBase   : String := "outBase"

/-- **Slack between the input and the output.**  The cooperative copy loop loads on
    all 32 lanes and stores only where `cpI + lane < len`, so a masked lane reads up
    to `lane = 31` bytes past the end of the literal run — and the final literal run
    flushes to the block end.  Without slack the LAST warp's over-read lands in warp
    0's output region: a real (if benign, the value is discarded) data race, and it
    makes `Lz4Sites.RegConfined.loads` false.  `31` suffices; `32` is the warp width. -/
def copySlack : Nat := 32
def rTbl       : String := "tbl"
def rLitAnchor : String := "litAnchor"
def rSearchPos : String := "searchPos"
def rOp        : String := "op"

/-- Prologue segment: ids → oob guard → base pointers → shared-table clear loop →
    cursor init → `loop` head (39 instructions; `loop` at index 38). -/
def prologueInstrs (numBlocks inStride outStride hashLog : Nat) : List SInstr :=
  let entries := 2 ^ hashLog
  let tableBytes := entries * 2
  [ .mov "inP" (.reg rInPtr),
    -- The output base is DERIVED from the input base, not taken from the second
    -- parameter: one allocation holds the input followed by the output, so
    -- `inPtr + totIn ≤ outPtr` holds by construction rather than by assumption
    -- about two independent `cudaMalloc` results.  Same instruction count as the
    -- `mov` it replaces, so nothing is paid for it.
    .bin .add "outP" rInPtr (.imm (numBlocks * inStride + copySlack)),
    .mov "tid" (.reg rTidX), .mov "ctab" (.reg rCtaX), .mov "ntid" (.reg rNtidX),
    .binr .mul "gtid" "ctab" "ntid", .binr .add "gtid" "gtid" "tid",
    .bin .shr "gwarp" "gtid" (.imm 5), .bin .band rLane "tid" (.imm 31),
    .bin .shr "lwarp" "tid" (.imm 5),
    .setp .ge "oob" "gwarp" (.imm numBlocks), .braif "oob" "OOB",
    .setp .eq rPL0 rLane (.imm 0),
    .mov "gwD" (.reg "gwarp"),
    .mov "inOff" (.imm inStride), .binr .mul "inOff" "gwD" "inOff",
    .mov "outOff" (.imm outStride), .binr .mul "outOff" "gwD" "outOff",
    .binr .add rInBase "inP" "inOff", .binr .add rOutBase "outP" "outOff",
    .mov "smem" (.imm 0), .bin .mul "tblOff" "lwarp" (.imm tableBytes),
    .binr .add rTbl "smem" "tblOff",
    .mov "ci" (.reg rLane), .mov "z" (.imm 0),
    .lbl "clr",
    .setp .ge "pcnd" "ci" (.imm entries), .braif "pcnd" "clrDone",
    .bin .shl "ca" "ci" (.imm 1), .binr .add "ca" "ca" rTbl, .stsh "ca" "z",
    .bin .add "ci" "ci" (.imm 32), .bra "clr",
    .lbl "clrDone", .barwarp,
    .mov rLitAnchor (.imm 0), .mov rSearchPos (.imm 0), .mov rOp (.imm 0),
    .lbl "loop" ]

end AlgorithmLib.LZ4Simt
