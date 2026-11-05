module
public import AlgorithmLib.Host.SerialLib
meta import AlgorithmLib.Host.SerialLib
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The file library, called directly

What each `FsFn` does: the engine's functions over `std::fs`
(`base/src/builtin/fs.rs`). A path names contents (`FS`), so a handle keeps
what it was opened on through a rename or a removal of its path. A handle
lives outside every region (`fsFileAddr`, `fsDirAddr`), so a program passes
it and never loads through it.

A regular file opens for what its mode sets — `1` read, `2` write, `4`
create, `8` cut to nothing — and a handle refuses with `-1` what it was not
opened for. A read stops only at the file's end; a write writes every byte.
A directory opens to a handle that lists its entries in the system's order
(`listOrder`), a name's whole length answered and as much of it written as
fits.

**Undefined.** A handle not open, a path field not all there, a buffer
without room for what is read or bytes to write.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

inductive FsCall where
  | open (path : String) (read write create trunc : Bool)
  | close (k : Nat)
  | read (k : Nat) (len off : Nat)
  | write (k : Nat) (data : ByteArray) (off : Nat)
  | setLen (k : Nat) (len : Nat)
  | createDir (path : String)
  | removeFile (path : String)
  | removeDir (path : String)
  | rename (src dst : String)
  | dirOpen (path : String)
  | dirNext (k : Nat)
  | dirClose (k : Nat)
  /-- An answer that changes nothing. -/
  | answer (v : V)

/-- The open file at `a`. -/
def World.fsFile? (w : World) (a : UInt64) : Option Nat :=
  (List.range w.openFiles.size).find? fun k => fsFileAddr k == a && (w.openFiles[k]?.map (·.live)).getD false

/-- The open directory at `a`. -/
def World.fsDir? (w : World) (a : UInt64) : Option Nat :=
  (List.range w.openDirs.size).find? fun k => fsDirAddr k == a && (w.openDirs[k]?.map (·.live)).getD false

/-- The contents file `k` is open on. -/
def World.fsContents (w : World) (k : Nat) : ByteArray :=
  ((w.openFiles[k]?).bind fun f => w.fs.nodes[f.node]?).getD ByteArray.empty

/-- An offset or length the C side reads as an `int64_t` and refuses when it
    is negative. -/
def negI64 (x : UInt64) : Bool := x ≥ 0x8000000000000000

def fsNeg : FsCall := .answer (ofInt .i64 (-1))

section
variable (w : World)

/-- The path in a field, refused with `r` where it is not UTF-8 or empty. -/
def fsPath (path cap : UInt64) (r : FsCall) (k : String → FsCall) : Option FsCall := do
  let p ← readField w.mem path cap
  match p with
  | some p => if p == "" then some r else some (k p)
  | none => some r

def fsDOpen : List UInt64 → Option FsCall
  | [path, cap, mode] =>
      let m := asI32 mode
      fsPath w path cap (.answer (.sc .i64 0)) fun p =>
        let (r, wr, c, t) := (m % 2 == 1, m / 2 % 2 == 1, m / 4 % 2 == 1, m / 8 % 2 == 1)
        if m < 0 || m ≥ 16 || !(r || wr) || ((c || t) && !wr) then .answer (.sc .i64 0)
        else .open p r wr c t
  | _ => none

def fsDClose : List UInt64 → Option FsCall
  | [h] => (w.fsFile? h).map .close
  | _ => none

/-- Room for the bytes read: up to `len`, as many as the file has from `off`.
    A negative length or offset, or a file not opened to read, answers `-1`. -/
def fsDRead : List UInt64 → Option FsCall
  | [h, buf, len, off] => do
      let k ← w.fsFile? h
      if negI64 len || negI64 off || !((w.openFiles[k]?.map (·.read)).getD false) then some fsNeg
      else do
        let n := ((w.fsContents k).extract off.toNat (off.toNat + len.toNat)).size
        let _ ← copyIn w.mem buf (ByteArray.mk (Array.replicate n 0))
        some (.read k len.toNat off.toNat)
  | _ => none

def fsDWrite : List UInt64 → Option FsCall
  | [h, buf, len, off] => do
      let k ← w.fsFile? h
      if negI64 len || negI64 off || !((w.openFiles[k]?.map (·.write)).getD false) then some fsNeg
      else do
        let bs ← readBytes w.mem buf len.toNat
        some (.write k bs off.toNat)
  | _ => none

def fsDSize : List UInt64 → Option FsCall
  | [h] => do
      let k ← w.fsFile? h
      let n := ((w.openFiles[k]?).bind fun f => w.fs.nodes[f.node]?).map (·.size)
      some (.answer (ofInt .i64 (n.getD 0)))
  | _ => none

def fsDSetLen : List UInt64 → Option FsCall
  | [h, len] => do
      let k ← w.fsFile? h
      if negI64 len || !((w.openFiles[k]?.map (·.write)).getD false) then some fsNeg
      else some (.setLen k len.toNat)
  | _ => none

def fsDSync : List UInt64 → Option FsCall
  | [h] => do
      let k ← w.fsFile? h
      some (.answer (ofInt .i64 (if (w.openFiles[k]?.map (·.write)).getD false then 0 else -1)))
  | _ => none

def fsDPathCall (k : String → FsCall) : List UInt64 → Option FsCall
  | [path, cap] => fsPath w path cap fsNeg k
  | _ => none

def fsDRename : List UInt64 → Option FsCall
  | [a, ac, b, bc] => do
      let p ← readField w.mem a ac
      let q ← readField w.mem b bc
      match p, q with
      | some p, some q => if p == "" || q == "" then some fsNeg else some (.rename p q)
      | _, _ => some fsNeg
  | _ => none

def fsDDirOpen : List UInt64 → Option FsCall
  | [path, cap] => fsPath w path cap (.answer (.sc .i64 0)) .dirOpen
  | _ => none

/-- The next name, with room for as much of it as fits in `cap`. -/
def fsDDirNext : List UInt64 → Option FsCall
  | [d, out, cap] => do
      let k ← w.fsDir? d
      match (w.openDirs[k]?).map (·.left) with
      | some (name :: _) =>
          let c := min name.utf8ByteSize (lenOf cap)
          (copyIn w.mem out (ByteArray.mk (Array.replicate c 0))).map fun _ => .dirNext k
      | _ => some (.answer (ofInt .i64 0))
  | _ => none

def fsDDirClose : List UInt64 → Option FsCall
  | [d] => (w.fsDir? d).map .dirClose
  | _ => none

end

/-- **What a file library call decodes to**, or `none` where the library
    does not define it. -/
def fsDecode (f : FsFn) (bits : List UInt64) (w : World) : Option FsCall :=
  match f with
  | .open => fsDOpen w bits
  | .close => fsDClose w bits
  | .read => fsDRead w bits
  | .write => fsDWrite w bits
  | .size => fsDSize w bits
  | .setLen => fsDSetLen w bits
  | .sync => fsDSync w bits
  | .createDir => fsDPathCall w .createDir bits
  | .removeFile => fsDPathCall w .removeFile bits
  | .removeDir => fsDPathCall w .removeDir bits
  | .rename => fsDRename w bits
  | .dirOpen => fsDDirOpen w bits
  | .dirNext => fsDDirNext w bits
  | .dirClose => fsDDirClose w bits

/-- The names directly inside directory `p`: its files and directories. -/
def FS.children (fs : FS) (p : String) : List String :=
  let inside (q : String) : Option String :=
    if q.startsWith (p ++ "/") then
      let rest := q.drop (p.length + 1)
      if rest == "" || rest.contains '/' then none else some rest
    else none
  ((fs.names.map (·.1) ++ fs.dirs).filterMap inside).mergeSort (fun a b => decide (a ≤ b))

/-- Whether anything is inside directory `p`. -/
def FS.hasChildren (fs : FS) (p : String) : Bool :=
  (fs.names.map (·.1) ++ fs.dirs).any (·.startsWith (p ++ "/"))

/-- **What a decoded file call does to the files.** It always answers, and it
    leaves program memory alone: what a read or a listing writes is written
    in `fsCall`, where the caller asked. -/
def fsEffect (c : FsCall) (w : World) : Option V × World :=
  match c with
  | .open p r wr cr t =>
      if w.fs.dirs.contains p then (some (.sc .i64 0), w) else
      match w.fs.node? p with
      | some n =>
          (some (.sc .i64 (fsFileAddr w.openFiles.size)),
           { w with fs := if t then { w.fs with nodes := w.fs.nodes.set! n ByteArray.empty } else w.fs,
                    openFiles := w.openFiles.push { node := n, read := r, write := wr } })
      | none =>
          if cr && w.creatable p then
            (some (.sc .i64 (fsFileAddr w.openFiles.size)),
             { w with fs := { w.fs with names := (p, w.fs.nodes.size) :: w.fs.names,
                                        nodes := w.fs.nodes.push ByteArray.empty },
                      openFiles := w.openFiles.push { node := w.fs.nodes.size, read := r, write := wr } })
          else (some (.sc .i64 0), w)
  | .close k => (none, { w with openFiles := w.openFiles.modify k fun f => { f with live := false } })
  | .read k len off =>
      let c := w.fsContents k
      (some (ofInt .i64 ((c.extract off (off + len)).size)), w)
  | .write k data off =>
      let n := ((w.openFiles[k]?).map (·.node)).getD 0
      (some (ofInt .i64 data.size),
       { w with fs := { w.fs with nodes := w.fs.nodes.modify n (spliceAt · data off) } })
  | .setLen k len =>
      let n := ((w.openFiles[k]?).map (·.node)).getD 0
      let cut (b : ByteArray) := if len ≤ b.size then b.extract 0 len
        else b ++ ByteArray.mk (Array.replicate (len - b.size) 0)
      (some (ofInt .i64 0), { w with fs := { w.fs with nodes := w.fs.nodes.modify n cut } })
  | .createDir p =>
      if (w.fs.node? p).isSome || w.fs.dirs.contains p || !w.creatable p then (some (ofInt .i64 (-1)), w)
      else (some (ofInt .i64 0), { w with fs := { w.fs with dirs := p :: w.fs.dirs } })
  | .removeFile p =>
      if (w.fs.node? p).isSome then (some (ofInt .i64 0), { w with fs := { w.fs with names := w.fs.names.filter (·.1 != p) } })
      else (some (ofInt .i64 (-1)), w)
  | .removeDir p =>
      if w.fs.dirs.contains p && !w.fs.hasChildren p then
        (some (ofInt .i64 0), { w with fs := { w.fs with dirs := w.fs.dirs.filter (· != p) } })
      else (some (ofInt .i64 (-1)), w)
  | .rename p q =>
      match w.fs.node? p with
      | some n =>
          if w.fs.dirs.contains q then (some (ofInt .i64 (-1)), w)
          else (some (ofInt .i64 0), { w with fs := { w.fs with names := (q, n) :: w.fs.names.filter (fun e => e.1 != p && e.1 != q) } })
      | none => (some (ofInt .i64 (-1)), w)
  | .dirOpen p =>
      if w.fs.dirs.contains p then
        (some (.sc .i64 (fsDirAddr w.openDirs.size)),
         { w with openDirs := w.openDirs.push { left := w.listOrder (w.fs.children p) } })
      else (some (.sc .i64 0), w)
  | .dirNext k =>
      match (w.openDirs[k]?).map (·.left) with
      | some (name :: rest) =>
          (some (ofInt .i64 name.utf8ByteSize),
           { w with openDirs := w.openDirs.modify k fun d => { d with left := rest } })
      | _ => (some (ofInt .i64 0), w)
  | .dirClose k => (none, { w with openDirs := w.openDirs.modify k fun d => { d with live := false } })
  | .answer v => (some v, w)

/-- What a read call writes: the bytes it takes from the file at `h`. -/
def World.fsReadBytes (w : World) (h len off : UInt64) : ByteArray :=
  match w.fsFile? h with
  | some k =>
      if negI64 len || negI64 off || !((w.openFiles[k]?.map (·.read)).getD false) then ByteArray.empty
      else (w.fsContents k).extract off.toNat (off.toNat + len.toNat)
  | none => ByteArray.empty

/-- What a listing call writes: as much of the next name as fits in `cap`. -/
def World.fsNameBytes (w : World) (d cap : UInt64) : ByteArray :=
  match (w.fsDir? d).bind fun k => (w.openDirs[k]?).map (·.left) with
  | some (name :: _) => name.toUTF8.extract 0 (lenOf cap)
  | _ => ByteArray.empty

theorem fsReadBytes_size (w : World) (h len off : UInt64) : (w.fsReadBytes h len off).size ≤ len.toNat := by
  unfold World.fsReadBytes; split
  · split
    · simp
    · rw [ByteArray.size_extract]; omega
  · simp

theorem fsNameBytes_size (w : World) (d cap : UInt64) : (w.fsNameBytes d cap).size ≤ cap.toNat := by
  unfold World.fsNameBytes; split
  · exact Nat.le_trans (extract_zero_size_le _ _) (lenOf_le cap)
  · simp

/-- **What a file library call does**: decoded, then carried out; the bytes a
    read takes and a listed name are written where the caller asked. -/
def fsCall (f : FsFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (fsDecode f bits w).map fun c =>
    let e := fsEffect c w
    match f, bits with
    | .read, [h, buf, len, off] => (e.1, { e.2 with mem := (copyIn w.mem buf (w.fsReadBytes h len off)).getD w.mem })
    | .dirNext, [d, out, cap] => (e.1, { e.2 with mem := (copyIn w.mem out (w.fsNameBytes d cap)).getD w.mem })
    | _, _ => e

theorem fsEffect_mem (c : FsCall) (w : World) : (fsEffect c w).2.mem = w.mem := by
  cases c <;> simp only [fsEffect]
  all_goals (repeat' split) <;> rfl

end AlgorithmLib.HProg.Sem
