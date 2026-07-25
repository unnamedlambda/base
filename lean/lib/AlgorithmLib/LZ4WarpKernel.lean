import AlgorithmLib.LZ4WarpColl
import AlgorithmLib.LZ4SimtSerialize

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt

-- ── Collective lowerings (match the machine segments in `compressorBodyInstrs`) ─

/-- `coopExtendStep` → 14-instr per-lane compare + `vote;bnot;brev;clz` reduction. -/
def coopExtendEmit (adv : String) : List SInstr :=
  [ .binr .add "idx" "ml" "lane",
    .binr .add "pe" "p0" "idx",
    .setp .lt "pIn" "pe" (.reg "ecR"),
    .binr .min "peC" "pe" "ec1",
    .binr .sub "dfe" "peC" "p0",
    .binr .add "caC" "cand0" "dfe",
    .mov "peD" (.reg "peC"),
    .binr .add "aP" "inBase" "peD",
    .mov "caD" (.reg "caC"),
    .binr .add "aC" "inBase" "caD",
    .ldgo "bP" "aP" 0,
    .ldgo "bC" "aC" 0,
    .setp .eq "pEqB" "bP" (.reg "bC"),
    .andp "pOk" "pIn" "pEqB",
    .vote "balOk" "pOk",
    .bnot "mis" "balOk",
    .brev "revM" "mis",
    .clz adv "revM" ]

/-- `coopWindow` → per-lane probe (hash → table lookup → 4-byte verify), select
    first hit (`vote;brev;clz`), insert-after-select, and extract `(found,p0,cand0)`.
    Mirrors `compressorBodyInstrs` lines 464–496 (the machine's window-find). -/
def coopWindowEmit (found p0 cand0 sp : String) (inStride searchLim hashLog : Nat) : List SInstr :=
  let hashShift := 32 - hashLog
  let mask := 2 ^ hashLog - 1
  [ .binr .add "posP" sp "lane",
    .setp .lt "pValid" "posP" (.imm searchLim),
    .mov "cap4" (.imm (inStride - 4)),
    .binr .min "rp" "posP" "cap4",
    .binr .add "rpA" "inBase" "rp",
    -- load4 at inBase+rp → v32
    .ldgo "b0" "rpA" 0, .ldgo "b1" "rpA" 1, .ldgo "b2" "rpA" 2, .ldgo "b3" "rpA" 3,
    .bin .shl "b1" "b1" (.imm 8), .bin .shl "b2" "b2" (.imm 16), .bin .shl "b3" "b3" (.imm 24),
    .bin .bor "v32" "b0" (.reg "b1"), .bin .bor "v32" "v32" (.reg "b2"),
    .bin .bor "v32" "v32" (.reg "b3"),
    .bin .mul "hh" "v32" (.imm wHashK),
    .bin .shr "hh" "hh" (.imm hashShift),
    .bin .band "hh" "hh" (.imm mask),
    .bin .shl "hh" "hh" (.imm 1),
    .binr .add "addr" "hh" "tbl",
    .ldsh "candRaw" "addr",
    .bin .sub "cand" "candRaw" (.imm 1),
    .binr .min "rc" "cand" "cap4",
    .binr .add "rcA" "inBase" "rc",
    .ldgo "c0" "rcA" 0, .ldgo "c1" "rcA" 1, .ldgo "c2" "rcA" 2, .ldgo "c3" "rcA" 3,
    .bin .shl "c1" "c1" (.imm 8), .bin .shl "c2" "c2" (.imm 16), .bin .shl "c3" "c3" (.imm 24),
    .bin .bor "vc" "c0" (.reg "c1"), .bin .bor "vc" "vc" (.reg "c2"), .bin .bor "vc" "vc" (.reg "c3"),
    .setp .ne "pNE" "candRaw" (.imm 0),
    .setp .lt "pCO" "cand" (.reg "posP"),
    .setp .eq "pEq" "vc" (.reg "v32"),
    .andp "pH1" "pValid" "pNE", .andp "pH2" "pH1" "pCO", .andp "pHit" "pH2" "pEq",
    .vote "bal" "pHit", .brev "rev" "bal", .clz "fl" "rev",
    .setp .le "pLe" "lane" (.reg "fl"), .andp "pIns" "pLe" "pValid",
    .bin .add "pp1" "posP" (.imm 1), .stshp "pIns" "addr" "pp1", .barwarp,
    .binr .add p0 sp "fl",
    .shfl cand0 "cand" "fl",
    .setp .ne found "bal" (.imm 0) ]

-- ── The lowering function ─────────────────────────────────────────────────────

abbrev EmitM := StateM Nat

def freshLabel (pfx : String) : EmitM String := do
  let n ← get; set (n + 1); pure (pfx ++ toString n)

/-- Lower a `WStmt` to `SInstr`, generating fresh branch labels for `uif`/`uwhile`. -/
def WStmt.emitM : WStmt → EmitM (List SInstr)
  | .skip => pure []
  | .seq a b => do let ea ← a.emitM; let eb ← b.emitM; pure (ea ++ eb)
  | .bin o d a b => pure [.bin o.toS d a b.toS]     -- SArg form (matches simSL_bin)
  | .setp c d a b => pure [.setp c d a b.toS]
  | .mov d a => pure [.mov d a.toS]
  | .ldgB d addr off => pure [.ldgo d addr off]
  | .ldshU d addr => pure [.ldsh d addr]
  | .stgB addr val => pure [.stg addr val]          -- all-lane (matches simSL_stgB; scalar uniform)
  | .stshU addr val => pure [.stsh addr val]
  | .uif cond t e => do
      let lElse ← freshLabel "Le"; let lEnd ← freshLabel "Ln"
      let et ← t.emitM; let ee ← e.emitM
      pure (uifEmit cond lElse lEnd et ee)
  | .uwhile cond body => do
      let lHead ← freshLabel "Lh"; let lEnd ← freshLabel "Lx"
      let eb ← body.emitM
      pure (uwhileEmit cond lHead lEnd eb)
  | .coopCopy dst src len => do
      -- 32-lane strided copy loop: while i<len { out[dst+i+lane]=in[src+i+lane]; i+=32 }
      let lH ← freshLabel "Ch"; let lX ← freshLabel "Cx"
      pure (
        [ .mov "cpI" (.imm 0), .setp .lt "cpCont" "cpI" (.reg len) ] ++
        uwhileEmit "cpCont" lH lX
          [ .binr .add "cpDo" dst "cpI", .binr .add "cpDo" "cpDo" "lane",
            .binr .add "cpSo" src "cpI", .binr .add "cpSo" "cpSo" "lane",
            .binr .add "cpJ" "cpI" "lane", .setp .lt "cpP" "cpJ" (.reg len),
            .ldgo "cpB" "cpSo" 0, .stgp "cpP" "cpDo" "cpB",
            .bin .add "cpI" "cpI" (.imm 32),
            .setp .lt "cpCont" "cpI" (.reg len) ] )
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog _ =>
      pure (coopWindowEmit found p0 cand0 sp inStride searchLim hashLog)
  | .coopExtendStep adv _ _ _ _ _ => pure (coopExtendEmit adv)

def WStmt.emit (s : WStmt) : List SInstr := (s.emitM.run 0).1

-- ── The compressor body as a `WStmt` (uniform emit sequences + collectives) ────

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

/-- The emitted body program (fresh labels from 0). -/
def compressorBodyEmit (inStride lenOff hashLog : Nat) : List SInstr :=
  (compressorBodyWStmt inStride lenOff hashLog).emit

/-- The full DSL compressor kernel: reused prologue (ids, base pointers, table
    clear, ending at `loop`) followed by the emitted `WStmt` body. -/
def warpKernelDSL (numBlocks inStride outStride lenOff hashLog : Nat) : Array SInstr :=
  (AlgorithmLib.LZ4Simt.prologueInstrs numBlocks inStride outStride hashLog
    ++ compressorBodyEmit inStride lenOff hashLog
    ++ ([SInstr.lbl "OOB", SInstr.ret])).toArray
