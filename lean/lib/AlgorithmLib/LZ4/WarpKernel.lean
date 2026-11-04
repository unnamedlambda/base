module
public import AlgorithmLib.LZ4.WarpDSL
meta import AlgorithmLib.LZ4.WarpDSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib

/-- **Slack between the input and the output.** A cooperative copy loads on
    all 32 lanes and stores only below its length, so a lane reads up to 31
    bytes past the end of the literal run, and the final literal run reaches the
    block's end. The slack keeps the last block's over-read inside the
    allocation and out of the first block's output. `31` suffices; `32` is the
    warp width. -/
def copySlack : Nat := 32

-- ── The compressor's warp program ──

/-- Fold a list of statements into a right-nested `seq`. -/
def wseq : List WStmt → WStmt
  | [] => .skip
  | [s] => s
  | s :: rest => .seq s (wseq rest)

/-- Store the low byte of `val` at `outBase+op`, then `op += 1` (lane-0 store). -/
def wStoreByte (val : String) : WStmt := wseq
  [ .bin .add "sbAddr" "outBase" (.reg "op"),
    .stgB "sbAddr" val,
    .bin .add "op" "op" (.imm 1) ]

/-- LSIC length-extension bytes for the count in register `n` (`n` is consumed). -/
def wEmitLSIC (n : String) : WStmt := wseq
  [ .setp .ge "lsicC" n (.imm 255),
    .uwhile "lsicC" (wseq
      [ .mov "c255" (.imm 255),
        wStoreByte "c255",
        .bin .sub n n (.imm 255),
        .setp .ge "lsicC" n (.imm 255) ]),
    wStoreByte n ]

/-- The token byte: `(min litLen 15) << 4 | tokLo`. -/
def wEmitToken (litLen tokLo : String) : WStmt := wseq
  [ .bin .min "tokHi" litLen (.imm 15),
    .bin .shl "tok" "tokHi" (.imm 4),
    .bin .bor "tok" "tok" (.reg tokLo),
    wStoreByte "tok" ]

/-- Full match sequence: token, literal LSIC, literal copy, 2 offset bytes, match LSIC. -/
def wEmitMatchSeq (litStart litLen off ml : String) : WStmt := wseq
  [ .bin .sub "mlm" ml (.imm 4),
    .bin .min "tokLo" "mlm" (.imm 15),
    wEmitToken litLen "tokLo",
    .setp .ge "pLitBig" litLen (.imm 15),
    .uif "pLitBig" (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip,
    .bin .add "cpDst" "outBase" (.reg "op"),
    .bin .add "cpSrc" "inBase" (.reg litStart),
    .coopCopy "cpDst" "cpSrc" litLen,
    .bin .add "op" "op" (.reg litLen),
    .bin .band "offLo" off (.imm 255),
    wStoreByte "offLo",
    .bin .shr "offHi" off (.imm 8),
    .bin .band "offHi" "offHi" (.imm 255),
    wStoreByte "offHi",
    .setp .ge "pMatBig" "mlm" (.imm 15),
    .uif "pMatBig" (wseq [ .bin .sub "matExtra" "mlm" (.imm 15), wEmitLSIC "matExtra" ]) .skip ]

/-- Final literal run: token (matchLo=0), literal LSIC, literal copy. -/
def wEmitFinalSeq (litStart litLen : String) : WStmt := wseq
  [ .mov "zero" (.imm 0),
    wEmitToken litLen "zero",
    .setp .ge "pLitBigF" litLen (.imm 15),
    .uif "pLitBigF" (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip,
    .bin .add "cpDstF" "outBase" (.reg "op"),
    .bin .add "cpSrcF" "inBase" (.reg litStart),
    .coopCopy "cpDstF" "cpSrcF" litLen,
    .bin .add "op" "op" (.reg litLen) ]

/-- The compressor main loop + finalize + length store, as a `WStmt`.  Assumes the
    prologue has set `inBase,outBase,op,searchPos,litAnchor,lane,pL0,tbl`. -/
def compressorBodyWStmt (inStride lenOff hashLog : Nat) : WStmt :=
  let searchLim := inStride - 12
  let endCap := inStride - 5
  wseq
  [ .setp .lt "loopC" "searchPos" (.imm searchLim),
    .uwhile "loopC" (wseq
      [ .coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm endCap),
            .mov "ec1" (.imm (endCap - 1)),
            .mov "ml" (.imm 4),
            .setp .ge "extC" "ml" (.imm 0),   -- enter loop (tautology: ml ≥ 0), predicate-typed
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
        .setp .lt "loopC" "searchPos" (.imm searchLim) ]),
    .mov "fLen" (.imm inStride),
    .bin .sub "fLen" "fLen" (.reg "litAnchor"),
    wEmitFinalSeq "litAnchor" "fLen",
    .bin .add "la0" "outBase" (.imm lenOff),
    .bin .band "lb" "op" (.imm 255), .stgB "la0" "lb",
    .bin .add "la1" "la0" (.imm 1), .bin .shr "ls" "op" (.imm 8), .bin .band "lb" "ls" (.imm 255), .stgB "la1" "lb",
    .bin .add "la2" "la0" (.imm 2), .bin .shr "ls" "op" (.imm 16), .bin .band "lb" "ls" (.imm 255), .stgB "la2" "lb",
    .bin .add "la3" "la0" (.imm 3), .bin .shr "ls" "op" (.imm 24), .bin .band "lb" "ls" (.imm 255), .stgB "la3" "lb" ]
