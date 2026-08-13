import AlgorithmLib.ClifData
import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- IR builder state -/
structure IRState where
  nextVal : Nat := 0
  nextBlock : Nat := 0
  nextSig : Nat := 0
  nextFn : Nat := 0
  currentBlock : Option BlockRef := none
  currentBlockParams : List (Val × ClifTy) := []
  currentInsts : List Inst := []  -- reverse order for O(1) prepend
  sigs : List SigDecl := []
  fns : List FnDecl := []
  blocks : List BlockData := []

/-- The IR builder monad -/
abbrev IRBuilder := StateM IRState

/-- **Every block, including the one still being emitted.**

    A builder's last block sits in `currentInsts` and never reaches `blocks`, so
    anything reading `IRState.blocks` directly sees a program with its final
    block missing. Every analysis must go through this. -/
def IRState.allBlocks (s : IRState) : List BlockData :=
  match s.currentBlock with
  | none      => s.blocks
  | some bref => s.blocks ++ [{ ref := bref
                                params := s.currentBlockParams
                                insts := s.currentInsts.reverse }]

-- ---------------------------------------------------------------------------
-- FFI declarations
-- ---------------------------------------------------------------------------

/-- Declare a CLIF signature -/
def declareSig (params : List ClifTy) (result : Option ClifTy) : IRBuilder SigRef := do
  let s ← get
  let ref : SigRef := { id := s.nextSig }
  let decl : SigDecl := { ref := ref, params := params, result := result }
  set { s with
    nextSig := s.nextSig + 1
    sigs := s.sigs ++ [decl]
  }
  pure ref

/-- Declare an FFI function with a new signature -/
def declareFFI (name : String) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .import name, sig := sig : FnDecl }]
  }
  pure ref

-- ---------------------------------------------------------------------------
-- Top-level builders
-- ---------------------------------------------------------------------------

/-- A function that does nothing, at a given index: one block taking the shared
    memory pointer and returning. -/
def noopAt (funcIdx : Nat) : FuncData :=
  { index := funcIdx, sigs := [], fns := [],
    blocks := [{ ref := { id := 0 },
                 params := [({ id := 0 }, ClifTy.i64)],
                 insts := [Inst.ret] }] }

/-- The standard noop function u0:0 -/
def noopFunction : FuncData := noopAt 0

/-- Assemble functions into a program. They must be in `u0:N` order: the
    runtime resolves call targets by treating the index as a `FuncId`. -/
def program (fs : List FuncData) : Program := { functions := fs }

/-- Declare a colocated FFI function (intra-module call, e.g. colocated %ht_create) -/
def declareColocatedFFI (name : String) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .import name, sig := sig, colocated := true : FnDecl }]
  }
  pure ref

/-- Declare a call to another function of this same program, by `u0:N` index. -/
def declareLocal (index : Nat) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .local index, sig := sig, colocated := true : FnDecl }]
  }
  pure ref

end IR

end AlgorithmLib
