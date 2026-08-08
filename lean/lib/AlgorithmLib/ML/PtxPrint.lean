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

/-- **The first register index in each class the emitter keeps for itself.**

    Address arithmetic needs somewhere to put a byte offset, and shared-memory
    addressing needs somewhere to put a symbol — neither is a register the
    statement asked for, so both are taken from the top of the file.  Buffer
    `b`'s base pointer is `%rd{b}` and a statement's own integers are `%r{d}`,
    so anything a kernel names has to stay below this: at it, a load would
    compute its address out of a base pointer the same instruction had just
    overwritten.  `FlatRegsOkB` and the caller's buffer-count guard are what
    check that. -/
def PTX_ADDR_SCRATCH : Nat := 1020

/-- Byte address of element `%r{ix}` of buffer `b`, into the scratch
    `%rd{PTX_ADDR_SCRATCH}`. -/
private def addrLines (b ix : Nat) : List String :=
  [s!"    mul.wide.u32 %rd{PTX_ADDR_SCRATCH + 1}, %r{ix}, 4;",
   s!"    add.u64 %rd{PTX_ADDR_SCRATCH}, %rd{b}, %rd{PTX_ADDR_SCRATCH + 1};"]

/-- Byte address of element `%r{ix}` of shared memory, into
    `%r{PTX_ADDR_SCRATCH}`. -/
private def smemLines (ix : Nat) : List String :=
  [s!"    mov.u32 %r{PTX_ADDR_SCRATCH}, {smemSym};",
   s!"    mul.lo.u32 %r{PTX_ADDR_SCRATCH + 1}, %r{ix}, 4;",
   s!"    add.u32 %r{PTX_ADDR_SCRATCH}, %r{PTX_ADDR_SCRATCH}, %r{PTX_ADDR_SCRATCH + 1};"]

/-- **The buffers a program stores to.**

    A kernel writes one buffer — the stage's `out` — and the disjointness of
    that buffer from the ones it reads is what `StageSpec.Exclusive` proves.
    Reading it back off the emitted instructions is what lets the printer act
    on it without being told. -/
def SI.storesTo : SI → Option Buf
  | .stG b _ _     => some b
  | .stGIf _ b _ _ => some b
  | _              => none

def FI.storesTo : FI → Option Buf
  | .si i => i.storesTo
  | _     => none

def flatStores (P : List FI) : List Buf := P.filterMap FI.storesTo

/-- **A load the kernel can take from the read-only path.**

    `ld.global.nc` reads through the non-coherent cache, which is sound exactly
    when nothing writes the data while the kernel runs.  A kernel stores only
    to the buffers `flatStores` names, and launches are serialised, so a buffer
    outside that list is not written for the kernel's lifetime.  The condition
    is decided from the instructions themselves rather than assumed. -/
def flatReadOnly (P : List FI) (b : Buf) : Bool := !(flatStores P).contains b

def siText (ro : Buf → Bool) : SI → List String
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
  | .ldGI d b ix   =>
      addrLines b ix ++
        [s!"    ld.global{if ro b then ".nc" else ""}.u32 %r{d}, [%rd{PTX_ADDR_SCRATCH}];"]
  | .cvtIF d ix    => [s!"    cvt.rn.f32.u32 {pregName d}, %r{ix};"]
  | .ldG d b ix    =>
      addrLines b ix ++
        [s!"    ld.global{if ro b then ".nc" else ""}.f32 {pregName d}, [%rd{PTX_ADDR_SCRATCH}];"]
  | .ldGV4 d0 d1 d2 d3 b ix =>
      addrLines b ix ++
        [s!"    ld.global{if ro b then ".nc" else ""}.v4.f32 \{{pregName d0}, {pregName d1}, {pregName d2}, {pregName d3}}, [%rd{PTX_ADDR_SCRATCH}];"]
  | .stG b ix r    => addrLines b ix ++ [s!"    st.global.f32 [%rd{PTX_ADDR_SCRATCH}], {pregName r};"]
  | .stGIf p b ix r =>
      addrLines b ix ++ [s!"    @%p{p} st.global.f32 [%rd{PTX_ADDR_SCRATCH}], {pregName r};"]
  | .stS ix r      => smemLines ix ++ [s!"    st.shared.f32 [%r{PTX_ADDR_SCRATCH}], {pregName r};"]
  | .ldS d ix      => smemLines ix ++ [s!"    ld.shared.f32 {pregName d}, [%r{PTX_ADDR_SCRATCH}];"]
  | .bar           => ["    bar.warp.sync 0xffffffff;"]
  | .ext op _ _    => [s!"    // extern {op.name}: dispatched from the host"]
  | .loop _ _ _    => ["    // unreachable: loops are flattened before printing"]

def fiText (ro : Buf → Bool) : FI → List String
  | .si i      => siText ro i
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
def bodyLines (ro : Buf → Bool) (start : Nat) : List FI → List String
  | []      => []
  | x :: xs => (s!"L{start}:" :: fiText ro x) ++ bodyLines ro (start + 1) xs

/-- Render the program.  Instruction `i` is preceded by the label `L{i}`, so a
    branch target — which *is* an instruction index — always names exactly the
    instruction the machine model jumps to. -/
def programText (P : List FI) (ro : Buf → Bool := flatReadOnly P) : List String :=
  bodyLines ro 0 P ++ [s!"L{P.length}:"]

/-- **Which loads take the read-only path, as a schedule rather than a rule.**

    `.nc` is sound for any buffer the program does not store to, but it is not
    always faster: a buffer read once and never revisited gains nothing from a
    cache and can lose to the ordinary path.  Both emissions render the same
    `List FI`, so the choice is a schedule — measured, not argued — and this is
    the knob a generator turns.

    `all` is the default and the safe end: every buffer that may take the path,
    does. `under k` restricts it to buffers below `k` bytes, which is where
    reuse across blocks actually lives. -/
inductive ROPolicy where
  | all
  | under (bytes : Nat) (sizeOf : Buf → Nat)
  | none

def ROPolicy.pred (p : ROPolicy) (P : List FI) : Buf → Bool
  | b => match p with
    | .all         => flatReadOnly P b
    | .under k sz  => flatReadOnly P b && decide (sz b ≤ k)
    | .none        => false

/-- **Every instruction is labelled with its own index.**

    Together with `FlatTargetsOkB` — a `decide` that every branch target is in
    range — this gives: *every branch in the emitted text resolves to the label
    of the instruction the machine model jumps to*.  That is not full printer
    correctness and is not claimed to be. -/
theorem bodyLines_label (ro : Buf → Bool) :
    ∀ (P : List FI) (start i : Nat), i < P.length →
    s!"L{start + i}:" ∈ bodyLines ro start P := by
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

/-- **Every register a kernel names is one the emitter left it.**

    The scratch indices at `PTX_ADDR_SCRATCH` are written by address arithmetic
    the statement never asked for, so a statement naming one of them would have
    its own value overwritten between the address computation and the access
    that uses it — text `ptxas` accepts and hardware addresses out of range.
    The buffer side of the same hazard is the caller's `BufBelow` check; this is
    the register side, and it is a `decide` per kernel. -/
def PReg.okB : PReg → Bool
  | .tmp n  => decide (n < PTX_ADDR_SCRATCH)
  | .mach n => decide (n < PTX_ADDR_SCRATCH)

def PInstr.regsOkB : PInstr → Bool
  | .movImm d _   => d.okB
  | .mov d s      => d.okB && s.okB
  | .addF d a b   => d.okB && a.okB && b.okB
  | .mulF d a b   => d.okB && a.okB && b.okB
  | .maxF d a b   => d.okB && a.okB && b.okB
  | .setGeF d a b => d.okB && a.okB && b.okB
  | .negF d a     => d.okB && a.okB
  | .divRn d a b  => d.okB && a.okB && b.okB
  | .sqrtRn d a   => d.okB && a.okB
  | .ex2 d a      => d.okB && a.okB
  | .shflBfly d s _ => d.okB && s.okB

def iOk (n : Nat) : Bool := decide (n < PTX_ADDR_SCRATCH)

def SI.regsOkB : SI → Bool
  | .fp p          => p.regsOkB
  | .movIC d _     => iOk d
  | .movLane d     => iOk d
  | .movCta d      => iOk d
  | .addR d a b    => iOk d && iOk a && iOk b
  | .mulR d a b    => iOk d && iOk a && iOk b
  | .addRC d a _   => iOk d && iOk a
  | .setpEqC p a _ => iOk p && iOk a
  | .setpGeC p a _ => iOk p && iOk a
  | .setpGeR p a b => iOk p && iOk a && iOk b
  | .ldGI d b ix   => iOk d && iOk b && iOk ix
  | .cvtIF d ix    => d.okB && iOk ix
  | .ldG d b ix    => d.okB && iOk b && iOk ix
  | .ldGV4 d0 d1 d2 d3 b ix =>
      d0.okB && d1.okB && d2.okB && d3.okB && iOk b && iOk ix
  | .stG b ix r    => iOk b && iOk ix && r.okB
  | .stGIf p b ix r => iOk p && iOk b && iOk ix && r.okB
  | .stS ix r      => iOk ix && r.okB
  | .ldS d ix      => d.okB && iOk ix
  | .bar           => true
  | .ext _ _ _     => true
  | .loop _ _ _    => true

def FI.regsOkB : FI → Bool
  | .si i      => i.regsOkB
  | .jmp _     => true
  | .jmpIf p _ => iOk p

/-- …for a whole program. -/
def FlatRegsOkB (P : List FI) : Bool := P.all FI.regsOkB

-- ---------------------------------------------------------------------------
-- The module
-- ---------------------------------------------------------------------------

/-- **How many virtual registers of each class the preamble declares.**

    A buffer becomes a `%rd`, so a kernel binding more buffers than this
    references a register the module never declared and `ptxas` rejects the
    text — after every proof about it has passed.  Naming the number is what
    lets a caller check its buffer count against it at build time. -/
def PTX_REG_BUDGET : Nat := 1024

/-- Register declarations.  Counts are virtual — ptxas allocates — so they are
    set generously; the scratch indices used above sit above anything the
    emitter allocates. -/
private def regDecls : String :=
  "    .reg .pred %p<" ++ toString PTX_REG_BUDGET ++ ">;\n"
    ++ "    .reg .u32  %r<" ++ toString PTX_REG_BUDGET ++ ">;\n"
    ++ "    .reg .u64  %rd<" ++ toString PTX_REG_BUDGET ++ ">;\n"
    ++ "    .reg .f32  %f<" ++ toString PTX_REG_BUDGET ++ ">;\n"
    ++ "    .reg .f32  %fw<" ++ toString PTX_REG_BUDGET ++ ">;\n"

/-- Emit a complete PTX module for a proven kernel.

    `flatKernel_sound` proves the instruction list this renders computes
    `s.elabIn cta`; this function is the text encoding of that list. -/
def emitProvenKernelN (name : String) (nbuf : Nat) (smemBytes : Nat) (s : EWStmt)
    (rop : ROPolicy := .all) : String :=
  -- `expandEW` first: it discharges the emitter's exactness precondition for
  -- *any* input (`expandEW_expFree`), so what is printed is always inside the
  -- proven fragment.
  let P := flatKernel (expandEW s)
  let params := String.intercalate ",\n" ((List.range nbuf).map (fun i => s!"    .param .u64 b{i}_ptr"))
  let loads := String.intercalate "\n"
    ((List.range nbuf).map (fun i => s!"    ld.param.u64 %rd{i}, [b{i}_ptr];"))
  let body := String.intercalate "\n" (programText P (rop.pred P))
  let smem := if smemBytes > 0 then s!".shared .align 4 .b8 {smemSym}[{smemBytes}];\n\n" else ""
  ".version 7.5\n.target sm_75\n.address_size 64\n\n" ++ smem
    ++ ".visible .entry " ++ name ++ "(\n" ++ params ++ "\n)\n{\n"
    ++ regDecls ++ "\n" ++ loads ++ "\n" ++ body ++ "\n    ret;\n}\n"

/-- Two-buffer convenience wrapper — the shape all three demo kernels use. -/
def emitProvenKernel (name : String) (s : EWStmt) : String :=
  emitProvenKernelN name 2 0 s

end AlgorithmLib.ML
