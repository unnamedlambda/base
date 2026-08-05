import Lz4Stores

set_option maxRecDepth 8192

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

/-- Lane-uniformity of the shipped 32 KiB kernel — the `Shape.uniShape` witness. -/
theorem uni_ok : K.toList.all (unifOK uniR) = true := by decide

-- ── The shipped kernel's witnesses: every `decide` in the chain lives here ──

theorem preShapeOK_true : preShapeB K = true := by
  simp only [preShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem guardShapeOK_true : guardShapeB K 32768 = true := by
  simp only [guardShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem loopShapeOK_true : loopShapeB K = true := by
  simp only [loopShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem selShapeOK_true : selShapeB K 32768 = true := by
  simp only [selShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem selFrameOK_true : selFrameB K = true := by
  simp only [selFrameB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem entryShapeOK_true : entryShapeB K = true := by
  simp only [entryShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem extShapeOK_true : extShapeB K 32768 = true := by
  simp only [extShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem extLoadShapeOK_true : extLoadShapeB K = true := by
  simp only [extLoadShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem advShapeOK_true : advShapeB K = true := by
  simp only [advShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem mlShapeOK_true : mlShapeB K = true := by
  simp only [mlShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem caShapeOK_true : caShapeB K = true := by
  simp only [caShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem matchShapeOK_true : matchShapeB K = true := by
  simp only [matchShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem litShapeOK_true : litShapeB K = true := by
  simp only [litShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem cpShapeOK_true : cpShapeB K = true := by
  simp only [cpShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem tailShapeOK_true : tailShapeB K = true := by
  simp only [tailShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem laShapeOK_true : laShapeB K = true := by
  simp only [laShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem ftShapeOK_true : ftShapeB K 32768 = true := by
  simp only [ftShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem cp2ShapeOK_true : cp2ShapeB K = true := by
  simp only [cp2ShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem ibShapeOK_true : ibShapeB K 32768 = true := by
  simp only [ibShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

theorem ibRegOK_true : ibRegOK K = true := by
  simp only [ibRegOK, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K shipped32_size, cfgRegion_eq K shipped32_size, cfgAllRegion_eq, winG_eqG K shipped32_size, win_eqG K shipped32_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

/-- **The shipped 32 KiB kernel has the program shape.** -/
instance shape32 : Shape K where
  size := shipped32_size
  preShape := preShapeOK_true
  entryShape := entryShapeOK_true
  uniShape := uni_ok
  lx1 := by decide
  lh4 := by decide
  ibReg := ibRegOK_true
  loopShape := loopShapeOK_true
  selFrame := selFrameOK_true
  extLoadShape := extLoadShapeOK_true
  advShape := advShapeOK_true
  mlShape := mlShapeOK_true
  caShape := caShapeOK_true
  matchShape := matchShapeOK_true
  litShape := litShapeOK_true
  cpShape := cpShapeOK_true
  tailShape := tailShapeOK_true
  laShape := laShapeOK_true
  cp2Shape := cp2ShapeOK_true

/-- **…and is a geometry at stride S.** -/
instance geo32 : Geo K 32768 :=
  { shape32 with
    sBound := (by omega),
    winShape := winShapeB_32,
    guardShape := guardShapeOK_true,
    selShape := selShapeOK_true,
    extShape := extShapeOK_true,
    ftShape := ftShapeOK_true,
    ibShape := ibShapeOK_true }

end Lz4Sites
