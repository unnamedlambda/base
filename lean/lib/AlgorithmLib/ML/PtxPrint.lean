import AlgorithmLib.ML.PtxFlat

namespace AlgorithmLib.ML

open AlgorithmLib.PTX

/-- The module-scope shared-memory array the emitted kernels address. -/
def smemSym : String := "_smem"

-- ---------------------------------------------------------------------------
-- Literals and registers
-- ---------------------------------------------------------------------------

private def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat (48 + n) else Char.ofNat (65 + (n - 10))

/-- 8 uppercase hex digits — the `0f…` literal form PTX wants. -/
def hex8 (v : UInt32) : String :=
  let n := v.toNat
  String.mk ((List.range 8).map (fun i => hexDigit ((n / (16 ^ (7 - i))) % 16)))

/-- A `Float32` as its f32 bit pattern.  The model carries `Float32` and the
    hardware register is binary32, so this is an exact re-encoding — nothing
    rounds here.  (It used to: the model carried binary64 and this call silently
    narrowed, so the "exact Float equality" theorems above were stating
    something the silicon did not do.) -/
def f32Lit (v : Float32) : String := "0f" ++ hex8 v.toBits

def pregName : PReg → String
  | .tmp n  => s!"%f{n}"
  | .mach n => s!"%fw{n}"

-- ---------------------------------------------------------------------------
-- Instruction encoding — the trusted table
-- ---------------------------------------------------------------------------

/-- The float fragment.  One line each, and each line is the opcode whose
    semantics `PInstr.step` gives. -/
def pinstrText : PInstr → String
  | .movImm d v   => s!"    mov.f32 {pregName d}, {f32Lit v};"
  | .mov d s      => s!"    mov.f32 {pregName d}, {pregName s};"
  | .addF d a b   => s!"    add.rn.f32 {pregName d}, {pregName a}, {pregName b};"
  | .mulF d a b   => s!"    mul.rn.f32 {pregName d}, {pregName a}, {pregName b};"
  | .maxF d a b   => s!"    max.f32 {pregName d}, {pregName a}, {pregName b};"
  | .setGeF d a b   => s!"    set.ge.f32.f32 {pregName d}, {pregName a}, {pregName b};"
  | .negF d a     => s!"    neg.f32 {pregName d}, {pregName a};"
  | .divRn d a b  => s!"    div.rn.f32 {pregName d}, {pregName a}, {pregName b};"
  | .sqrtRn d a   => s!"    sqrt.rn.f32 {pregName d}, {pregName a};"
  | .ex2 d a      => s!"    ex2.approx.f32 {pregName d}, {pregName a};"
  | .shflBfly d s m =>
      s!"    shfl.sync.bfly.b32 {pregName d}, {pregName s}, {m}, 31, 0xffffffff;"

/-- Byte address of element `%r{ix}` of buffer `b`, into the scratch `%rd60`. -/
private def addrLines (b ix : Nat) : List String :=
  [s!"    mul.wide.u32 %rd61, %r{ix}, 4;",
   s!"    add.u64 %rd60, %rd{b}, %rd61;"]

/-- Byte address of element `%r{ix}` of shared memory, into `%r1020`. -/
private def smemLines (ix : Nat) : List String :=
  [s!"    mov.u32 %r1020, {smemSym};",
   s!"    mul.lo.u32 %r1021, %r{ix}, 4;",
   s!"    add.u32 %r1020, %r1020, %r1021;"]

def siText : SI → List String
  | .fp p          => [pinstrText p]
  | .movIC d c     => [s!"    mov.u32 %r{d}, {c};"]
  | .movLane d     => [s!"    mov.u32 %r{d}, %laneid;"]
  | .movCta d      => [s!"    mov.u32 %r{d}, %ctaid.x;"]
  | .addR d a b    => [s!"    add.u32 %r{d}, %r{a}, %r{b};"]
  | .mulR d a b    => [s!"    mul.lo.u32 %r{d}, %r{a}, %r{b};"]
  | .addRC d a c   => [s!"    add.u32 %r{d}, %r{a}, {c};"]
  | .setpEqC p a c => [s!"    setp.eq.u32 %p{p}, %r{a}, {c};"]
  | .setpGeC p a c => [s!"    setp.ge.u32 %p{p}, %r{a}, {c};"]
  | .setpGeR p a b  => [s!"    setp.ge.u32 %p{p}, %r{a}, %r{b};"]
  | .ldGI d b ix   => addrLines b ix ++ [s!"    ld.global.u32 %r{d}, [%rd60];"]
  | .cvtIF d ix    => [s!"    cvt.rn.f32.u32 {pregName d}, %r{ix};"]
  | .ldG d b ix    => addrLines b ix ++ [s!"    ld.global.f32 {pregName d}, [%rd60];"]
  | .ldGV4 d0 d1 d2 d3 b ix =>
      addrLines b ix ++
        [s!"    ld.global.v4.f32 \{{pregName d0}, {pregName d1}, {pregName d2}, {pregName d3}}, [%rd60];"]
  | .stG b ix r    => addrLines b ix ++ [s!"    st.global.f32 [%rd60], {pregName r};"]
  | .stGIf p b ix r =>
      addrLines b ix ++ [s!"    @%p{p} st.global.f32 [%rd60], {pregName r};"]
  | .stS ix r      => smemLines ix ++ [s!"    st.shared.f32 [%r1020], {pregName r};"]
  | .ldS d ix      => smemLines ix ++ [s!"    ld.shared.f32 {pregName d}, [%r1020];"]
  | .bar           => ["    bar.warp.sync 0xffffffff;"]
  | .ext op _ _    => [s!"    // extern {op.name}: dispatched from the host"]
  | .loop _ _ _    => ["    // unreachable: loops are flattened before printing"]

def fiText : FI → List String
  | .si i      => siText i
  | .jmp t     => [s!"    bra L{t};"]
  | .jmpIf p t => [s!"    @%p{p} bra L{t};"]

/-! ### What the printer is held to

    This module carries no semantic theorem: the chain is proven down to the
    `List FI`, and the text encoding is trusted.  One property is both
    load-bearing and provable, so it is proved here.  `flatKernel` emits
    **absolute instruction indices** as branch targets, so a missing or
    misplaced label `L{t}` would send every branch elsewhere while every theorem
    still held.

    The body is therefore an explicit recursion carrying the index — a shape an
    induction can see — and `programText_label` proves every in-range label is
    emitted. -/

/-- The instruction lines, labelled from `start`. -/
def bodyLines (start : Nat) : List FI → List String
  | []      => []
  | x :: xs => (s!"L{start}:" :: fiText x) ++ bodyLines (start + 1) xs

/-- Render the program.  Instruction `i` is preceded by the label `L{i}`, so a
    branch target — which *is* an instruction index — always names exactly the
    instruction the machine model jumps to. -/
def programText (P : List FI) : List String :=
  bodyLines 0 P ++ [s!"L{P.length}:"]

/-- **Every instruction is labelled with its own index.**

    Together with `FlatTargetsOkB` — a `decide` that every branch target is in
    range — this gives: *every branch in the emitted text resolves to the label
    of the instruction the machine model jumps to*.  That is not full printer
    correctness and is not claimed to be. -/
theorem bodyLines_label : ∀ (P : List FI) (start i : Nat), i < P.length →
    s!"L{start + i}:" ∈ bodyLines start P := by
  intro P
  induction P with
  | nil => intro _ i h; exact absurd h (by simp)
  | cons x xs ih =>
      intro start i h
      rcases Nat.eq_zero_or_pos i with hi | hi
      · subst hi
        exact List.mem_append_left _ (by simp [bodyLines])
      · obtain ⟨i', hi'⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
        subst hi'
        refine List.mem_append_right _ ?_
        have : start + (i' + 1) = (start + 1) + i' := by omega
        rw [this]
        exact ih (start + 1) i' (by simpa using Nat.lt_of_succ_lt_succ h)

theorem programText_label (P : List FI) (i : Nat) (h : i < P.length) :
    s!"L{i}:" ∈ programText P := by
  refine List.mem_append_left _ ?_
  have := bodyLines_label P 0 i h
  simpa using this

/-- The end label is emitted too, so a branch past the last instruction — which
    is how `flatKernel` encodes "fall through to the end" — also resolves. -/
theorem programText_endLabel (P : List FI) :
    s!"L{P.length}:" ∈ programText P := List.mem_append_right _ (by simp)

/-- **The two `SI` constructors the printer cannot render.**

    `siText` renders `.loop` and `.ext` as *comments*, so either one reaching the
    printer would give a silently wrong kernel rather than an error.
    `EWStmt.Flat` does not cover this: these are machine-level constructors,
    downstream of it.  `flatKernel` should never emit them — loops are flattened
    and externs are host-dispatched — and this check turns that into a build
    failure. -/
def SI.printableB : SI → Bool
  | .loop _ _ _ => false
  | .ext _ _ _  => false
  | _           => true

def FI.printableB : FI → Bool
  | .si i      => i.printableB
  | .jmp _     => true
  | .jmpIf _ _ => true

/-- No unrenderable instruction reaches the text.  A `decide` per kernel. -/
def FlatPrintableB (P : List FI) : Bool := P.all FI.printableB

/-- Every branch target of an instruction names a real label. -/
def FI.targetOkB (n : Nat) : FI → Bool
  | .si _      => true
  | .jmp t     => decide (t ≤ n)
  | .jmpIf _ t => decide (t ≤ n)

/-- …for a whole program.  A `decide` at any concrete kernel. -/
def FlatTargetsOkB (P : List FI) : Bool := P.all (FI.targetOkB P.length)

-- ---------------------------------------------------------------------------
-- The module
-- ---------------------------------------------------------------------------

/-- Register declarations.  Counts are virtual — ptxas allocates — so they are
    set generously; the scratch indices used above sit above anything the
    emitter allocates. -/
private def regDecls : String :=
  "    .reg .pred %p<1024>;\n    .reg .u32  %r<1024>;\n"
    ++ "    .reg .u64  %rd<64>;\n    .reg .f32  %f<1024>;\n    .reg .f32  %fw<1024>;\n"

/-- Emit a complete PTX module for a proven kernel.

    `flatKernel_sound` proves the instruction list this renders computes
    `s.elabIn cta`; this function is the text encoding of that list. -/
def emitProvenKernelN (name : String) (nbuf : Nat) (smemBytes : Nat) (s : EWStmt) : String :=
  -- `expandEW` first: it discharges the emitter's exactness precondition for
  -- *any* input (`expandEW_expFree`), so what is printed is always inside the
  -- proven fragment.
  let P := flatKernel (expandEW s)
  let params := String.intercalate ",\n" ((List.range nbuf).map (fun i => s!"    .param .u64 b{i}_ptr"))
  let loads := String.intercalate "\n"
    ((List.range nbuf).map (fun i => s!"    ld.param.u64 %rd{i}, [b{i}_ptr];"))
  let body := String.intercalate "\n" (programText P)
  let smem := if smemBytes > 0 then s!".shared .align 4 .b8 {smemSym}[{smemBytes}];\n\n" else ""
  ".version 7.5\n.target sm_75\n.address_size 64\n\n" ++ smem
    ++ ".visible .entry " ++ name ++ "(\n" ++ params ++ "\n)\n{\n"
    ++ regDecls ++ "\n" ++ loads ++ "\n" ++ body ++ "\n    ret;\n}\n"

/-- Two-buffer convenience wrapper — the shape all three demo kernels use. -/
def emitProvenKernel (name : String) (s : EWStmt) : String :=
  emitProvenKernelN name 2 0 s

end AlgorithmLib.ML
