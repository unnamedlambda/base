module
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Typed handles

A resource id is an `i32` and a context is an `i64`, so at the value level an
environment and a byte count are interchangeable --- and passing one where
another belongs is a wrong answer, not an error at build time. Each resource
here is its own type over the surface's value type, and each wrapper takes and
returns the types it means: a `dbPut` takes a `DbEnv`, and nothing else
elaborates.

The wrappers emit exactly the calls the untyped ones do; a handle's `id` is
the value, for a body that needs the number itself.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR

/-- The families that hold a context in a slot. -/
inductive Family where
  | ht | lmdb

/-- A family's context, loaded from its slot: the value its calls take first. -/
structure Ctx (V : ClifTy → Type) (fam : Family) where
  ptr : V .i64

/-- An open LMDB environment. -/
structure DbEnv (V : ClifTy → Type) where
  id : V .i32

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

-- LMDB

def Ctx.dbOpen (c : Ctx V .lmdb) (path : V .i64) (mapMb : V .i32) : Prog V L (DbEnv V) := do
  pure ⟨← ffi .lmdbOpen %[c.ptr, path, mapMb]⟩

def Ctx.dbBegin (c : Ctx V .lmdb) (e : DbEnv V) : Prog V L (V .i32) :=
  ffi .lmdbBeginWriteTxn %[c.ptr, e.id]

def Ctx.dbPut (c : Ctx V .lmdb) (e : DbEnv V) (key : V .i64) (keyLen : V .i32)
    (val : V .i64) (valLen : V .i32) : Prog V L (V .i32) :=
  ffi .lmdbPut %[c.ptr, e.id, key, keyLen, val, valLen]

def Ctx.dbCommit (c : Ctx V .lmdb) (e : DbEnv V) : Prog V L (V .i32) :=
  ffi .lmdbCommitWriteTxn %[c.ptr, e.id]

def Ctx.dbScan (c : Ctx V .lmdb) (e : DbEnv V) (key : V .i64) (keyLen maxEntries : V .i32)
    (result room : V .i64) : Prog V L (V .i32) :=
  ffi .lmdbCursorScan %[c.ptr, e.id, key, keyLen, maxEntries, result, room]

end

end AlgorithmLib.Prog
