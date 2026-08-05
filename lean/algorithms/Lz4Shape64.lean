import Lz4ExtShape

set_option maxRecDepth 8192

/-!
# The 64 KiB geometry satisfies every shape premise of the 32 KiB proof

`Lz4Extend` states its program-shape facts as `Bool`-valued predicates over an
arbitrary program `p` and block stride `S`.  The shipped 32 KiB kernel
discharges them via the `…ShapeOK_true` lemmas.  This file discharges the very
same predicates for the 64 KiB kernel, `(WP.mk 16).kernel`, at `S = 65536`.

Each is one `decide`: the two kernels have the same 274 pcs and the same
opcodes, and differ only in immediates, so every check that the 32 KiB proof
performs is a check the 64 KiB kernel also passes — but that is a claim about
the emitted arrays, and this file is where it is *checked* rather than assumed.
-/

namespace Lz4Sites
open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

/-- The 64 KiB kernel. -/
abbrev K64 := (WP.mk 16).kernel

theorem shipped64_size : K64.size = 274 := by rfl

theorem preShapeB_64 : preShapeB K64 = true := by
  simp only [preShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem guardShapeB_64 : guardShapeB K64 65536 = true := by
  simp only [guardShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem loopShapeB_64 : loopShapeB K64 = true := by
  simp only [loopShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem selShapeB_64 : selShapeB K64 65536 = true := by
  simp only [selShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem selFrameB_64 : selFrameB K64 = true := by
  simp only [selFrameB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem extShapeB_64 : extShapeB K64 65536 = true := by
  simp only [extShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem extLoadShapeB_64 : extLoadShapeB K64 = true := by
  simp only [extLoadShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem advShapeB_64 : advShapeB K64 = true := by
  simp only [advShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem mlShapeB_64 : mlShapeB K64 = true := by
  simp only [mlShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem caShapeB_64 : caShapeB K64 = true := by
  simp only [caShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem matchShapeB_64 : matchShapeB K64 = true := by
  simp only [matchShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem litShapeB_64 : litShapeB K64 = true := by
  simp only [litShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem cpShapeB_64 : cpShapeB K64 = true := by
  simp only [cpShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem tailShapeB_64 : tailShapeB K64 = true := by
  simp only [tailShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem laShapeB_64 : laShapeB K64 = true := by
  simp only [laShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem ftShapeB_64 : ftShapeB K64 65536 = true := by
  simp only [ftShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem cp2ShapeB_64 : cp2ShapeB K64 = true := by
  simp only [cp2ShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide
theorem ibShapeB_64 : ibShapeB K64 65536 = true := by
  simp only [ibShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide


/-- **The 64 KiB kernel has the same program shape.** -/
theorem entryShapeB_64 : entryShapeB K64 = true := by
  simp only [entryShapeB, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
    Nat.reduceLeDiff, Nat.reduceAdd]
  decide

instance shape64 : Shape K64 where
  size := shipped64_size
  preShape := preShapeB_64
  entryShape := entryShapeB_64
  uniShape := by decide
  lx1 := by decide
  lh4 := by decide
  ibReg := by
    simp only [ibRegOK, loopS, extS, matchS, litS, tailS, laS, ftS, ibS, cfg_eqG K64 shipped64_size, cfgRegion_eq K64 shipped64_size, cfgAllRegion_eq, winG_eqG K64 shipped64_size, win_eqG K64 shipped64_size,
      Nat.reduceLeDiff, Nat.reduceAdd]
    decide
  loopShape := loopShapeB_64
  selFrame := selFrameB_64
  extLoadShape := extLoadShapeB_64
  advShape := advShapeB_64
  mlShape := mlShapeB_64
  caShape := caShapeB_64
  matchShape := matchShapeB_64
  litShape := litShapeB_64
  cpShape := cpShapeB_64
  tailShape := tailShapeB_64
  laShape := laShapeB_64
  cp2Shape := cp2ShapeB_64

/-- **…and is a geometry at stride 65536.** -/
instance geo64 : Geo K64 65536 :=
  { shape64 with
    sBound := (by omega),
    winShape := winShapeB_64,
    guardShape := guardShapeB_64,
    selShape := selShapeB_64,
    extShape := extShapeB_64,
    ftShape := ftShapeB_64,
    ibShape := ibShapeB_64 }

end Lz4Sites
