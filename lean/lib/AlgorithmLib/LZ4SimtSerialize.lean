import AlgorithmLib.LZ4SimtEmit

namespace AlgorithmLib.LZ4Simt
open Std

-- ── Register classification ───────────────────────────────────────────────────

/-- Special SM registers (read via `mov.u32` then widened). -/
def isSpecialReg (n : String) : Bool :=
  n == "%tid.x" || n == "%ctaid.x" || n == "%ntid.x"

/-- Kernel parameters (loaded via `ld.param.u64`). -/
def isParamReg (n : String) : Bool := n == "in_ptr" || n == "out_ptr"

private def pushIf (acc : List String) (n : String) : List String :=
  if acc.contains n then acc else n :: acc

/-- Predicate registers = destinations of `setp`/`andp`.  (Branches/predicated
    stores *use* them.) -/
def predRegsOf (prog : Array SInstr) : List String :=
  prog.foldl (init := []) fun acc i =>
    match i with
    | .setp _ p _ _ => pushIf acc p
    | .andp d _ _   => pushIf acc d
    | _             => acc

/-- Every non-special, non-param, non-predicate register mentioned — the `.b64`
    virtual registers to declare. -/
def valueRegsOf (prog : Array SInstr) (preds : List String) : List String :=
  let add := fun (acc : List String) (n : String) =>
    if isSpecialReg n || isParamReg n || preds.contains n then acc else pushIf acc n
  let addA := fun (acc : List String) (a : SArg) =>
    match a with | .reg n => add acc n | .imm _ => acc
  prog.foldl (init := []) fun acc i =>
    match i with
    | .mov d a          => addA (add acc d) a
    | .bin _ d a b      => addA (add (add acc d) a) b
    | .binr _ d a b     => add (add (add acc d) a) b
    | .cvt32 d a | .brev d a | .clz d a | .bnot d a => add (add acc d) a
    | .setp _ _ a b     => addA (add acc a) b
    | .andp _ _ _       => acc
    | .selp d a b _     => add (add (add acc d) a) b
    | .ldgo d addr _    => add (add acc d) addr
    | .ldgop p d addr _ => add (add (add acc p) d) addr
    | .stg addr s | .stsh addr s => add (add acc addr) s
    | .stgp _ addr s | .stshp _ addr s | .stg32p _ addr s => add (add acc addr) s
    | .ldsh d addr      => add (add acc d) addr
    | .vote d p         => add (add acc d) p
    | .shfl d s src     => add (add (add acc d) s) src
    | _                 => acc

-- ── Per-instruction rendering (all-`b64`; typed mem/collective ops) ────────────

def sopStr : SOp → String
  | .add => "add.s64" | .sub => "sub.s64" | .band => "and.b64" | .bor => "or.b64"
  | .shl => "shl.b64" | .shr => "shr.u64" | .min => "min.u64" | .mul => "mul.lo.s64"

def scmpStr : SCmp → String
  | .eq => "eq" | .ne => "ne" | .lt => "lt" | .le => "le" | .ge => "ge"

def sargStr : SArg → String
  | .reg n => n
  | .imm v => toString v

/-- Render one `SInstr` as one-or-more PTX lines (no leading indent). -/
def instrLines : SInstr → List String
  | .mov d (.reg s) =>
      if isSpecialReg s then ["mov.u32 %wtmp, " ++ s, "cvt.u64.u32 " ++ d ++ ", %wtmp"]
      else ["mov.b64 " ++ d ++ ", " ++ s]
  | .mov d (.imm v)     =>
      -- The model sets the shared-table base register `smem` to the abstract
      -- flat-array base `0`; on hardware that base is the `.shared` symbol
      -- `_smem`.  Sound because every table access is `smem + offset`, so the
      -- base cancels between the clear/insert writes and the probe reads — only
      -- consistency matters, which `sstep`'s flat `smem` array captures.
      if d == "smem" && v == 0 then ["mov.u64 smem, _smem"]
      else ["mov.u64 " ++ d ++ ", " ++ toString v]
  | .bin o d a b        => [sopStr o ++ " " ++ d ++ ", " ++ a ++ ", " ++ sargStr b]
  | .binr o d a b       => [sopStr o ++ " " ++ d ++ ", " ++ a ++ ", " ++ b]
  | .cvt32 d a          => ["and.b64 " ++ d ++ ", " ++ a ++ ", 4294967295"]
  | .brev d a           => ["cvt.u32.u64 %wtmp, " ++ a, "brev.b32 %wtmp, %wtmp",
                            "cvt.u64.u32 " ++ d ++ ", %wtmp"]
  | .clz d a            => ["cvt.u32.u64 %wtmp, " ++ a, "clz.b32 %wtmp2, %wtmp",
                            "cvt.u64.u32 " ++ d ++ ", %wtmp2"]
  | .bnot d a           => ["not.b64 " ++ d ++ ", " ++ a]
  | .setp c p a b       => ["setp." ++ scmpStr c ++ ".u64 " ++ p ++ ", " ++ a ++ ", " ++ sargStr b]
  | .andp d a b         => ["and.pred " ++ d ++ ", " ++ a ++ ", " ++ b]
  | .selp d a b p       => ["selp.b64 " ++ d ++ ", " ++ a ++ ", " ++ b ++ ", " ++ p]
  | .ldgo d addr off    =>
      let a := if off == 0 then "[" ++ addr ++ "]" else "[" ++ addr ++ "+" ++ toString off ++ "]"
      ["ld.global.u8 " ++ d ++ ", " ++ a]
  | .ldgop p d addr off =>
      let a := if off == 0 then "[" ++ addr ++ "]" else "[" ++ addr ++ "+" ++ toString off ++ "]"
      ["@" ++ p ++ " ld.global.u8 " ++ d ++ ", " ++ a]
  | .stg addr s         => ["st.global.u8 [" ++ addr ++ "], " ++ s]
  | .stgp p addr s      => ["@" ++ p ++ " st.global.u8 [" ++ addr ++ "], " ++ s]
  | .stg32p p addr s    => ["@" ++ p ++ " st.global.u32 [" ++ addr ++ "], " ++ s]
  | .ldsh d addr        => ["ld.shared.u16 " ++ d ++ ", [" ++ addr ++ "]"]
  | .stsh addr s        => ["st.shared.u16 [" ++ addr ++ "], " ++ s]
  | .stshp p addr s     => ["@" ++ p ++ " st.shared.u16 [" ++ addr ++ "], " ++ s]
  | .vote d p           => ["vote.sync.ballot.b32 %wtmp, " ++ p ++ ", 4294967295",
                            "cvt.u64.u32 " ++ d ++ ", %wtmp"]
  | .shfl d s src       => -- shfl.sync.idx.b32 d, data, srcLane, clamp(0x1f), mask
                           ["cvt.u32.u64 %wtmp, " ++ src,
                            "cvt.u32.u64 %wtmp2, " ++ s,
                            "shfl.sync.idx.b32 %wtmp2, %wtmp2, %wtmp, 31, 4294967295",
                            "cvt.u64.u32 " ++ d ++ ", %wtmp2"]
  | .barwarp            => ["bar.warp.sync 4294967295"]
  | .braif p lbl        => ["@" ++ p ++ " bra " ++ lbl ++ ";"]
  | .braifnot p lbl     => ["@!" ++ p ++ " bra " ++ lbl ++ ";"]
  | .bra lbl            => ["bra " ++ lbl ++ ";"]
  | .lbl name           => [name ++ ":"]
  | .ret                => ["ret;"]

/-- A body line gets `;` unless it is a label or already terminated. -/
private def term (s : String) : String :=
  if s.endsWith ":" || s.endsWith ";" then s else s ++ ";"

-- ── Module scaffold ───────────────────────────────────────────────────────────

/-- Full PTX module for a serialized `Array SInstr`.  `smemBytes` = shared size,
    `smemName` the label the `.shared` region is addressed through (the model uses
    a `smemBase` register; we point it at `_smem`). -/
def serializeKernel (prog : Array SInstr) (smemBytes : Nat) : String :=
  let preds := predRegsOf prog
  let vals  := valueRegsOf prog preds
  let decl := fun (kind : String) (rs : List String) =>
    if rs.isEmpty then "" else rs.foldl (fun s r => s ++ "    .reg " ++ kind ++ " " ++ r ++ ";\n") ""
  let header :=
    ".version 8.0\n.target sm_86\n.address_size 64\n\n" ++
    ".visible .entry main(\n    .param .u64 in_ptr_p,\n    .param .u64 out_ptr_p\n) {\n"
  let regDecls :=
    "    .reg .b64 in_ptr, out_ptr;\n" ++
    "    .reg .b32 %wtmp, %wtmp2;\n" ++
    decl ".b64" vals ++ decl ".pred" preds ++
    (if smemBytes > 0 then "    .shared .align 4 .b8 _smem[" ++ toString smemBytes ++ "];\n" else "")
  let paramLoad :=
    "    ld.param.u64 in_ptr, [in_ptr_p];\n    ld.param.u64 out_ptr, [out_ptr_p];\n"
  -- Emit the body, deduplicating labels.  A repeated `.lbl name` is provably
  -- unreachable (`sfindLabel` = first occurrence, so every branch targets the
  -- first), hence inert in `sstep` — but ptxas rejects duplicate labels, so
  -- later occurrences get a fresh suffix.  Branches keep the original name and
  -- still resolve to the first (name-preserving) occurrence.  Behavior-preserving.
  -- NOTE: `compProgSExplicit` has a duplicate `loop` (pc 38 target, pc 43 inert) —
  -- a transcription wart worth cleaning at the source (rename the pc-43 label).
  let (_, body) := prog.foldl (init := (([] : List String), "")) fun (seen, s) i =>
    match i with
    | .lbl name =>
        let nm := if seen.contains name then name ++ "_dup" ++ toString seen.length else name
        (name :: seen, s ++ "    " ++ nm ++ ":\n")
    | _ => (seen, (instrLines i).foldl (fun s l => s ++ "    " ++ term l ++ "\n") s)
  header ++ regDecls ++ paramLoad ++ body ++ "}\n"

end AlgorithmLib.LZ4Simt
