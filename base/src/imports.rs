//! What a program may call outside itself, and at what signature.
//!
//! The table is the whole of it: a program naming anything else is refused
//! when it is built, rather than resolved against whatever the process happens
//! to have loaded. Each signature is read off the Rust function's own type, so
//! it is not a second statement that could disagree with the function — the
//! `as` cast in each entry is checked by the compiler.

use std::sync::OnceLock;

use base_types::clif::ClifTy;

use crate::ffi::{file, native, os, stdio, thread};

/// A host function a program may import.
pub(crate) struct Import {
    pub(crate) name: &'static str,
    /// The function's address. Code, never written, so sharing it between
    /// threads is sound.
    pub(crate) addr: usize,
    pub(crate) params: Vec<ClifTy>,
    pub(crate) result: Option<ClifTy>,
}

/// How a Rust parameter or result type crosses into CLIF.
///
/// Every pointer is an `i64`: the platform is 64-bit, and a program holds an
/// address as a plain integer.
trait Abi {
    const TY: ClifTy;
}

impl Abi for i32 {
    const TY: ClifTy = ClifTy::I32;
}
impl Abi for u32 {
    const TY: ClifTy = ClifTy::I32;
}
impl Abi for i64 {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for u64 {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for usize {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for f32 {
    const TY: ClifTy = ClifTy::F32;
}
impl Abi for f64 {
    const TY: ClifTy = ClifTy::F64;
}
impl<T> Abi for *mut T {
    const TY: ClifTy = ClifTy::I64;
}
impl<T> Abi for *const T {
    const TY: ClifTy = ClifTy::I64;
}

/// What a function answers: nothing, or one value.
trait Answer {
    const TY: Option<ClifTy>;
}

impl Answer for () {
    const TY: Option<ClifTy> = None;
}
impl<T: Abi> Answer for T {
    const TY: Option<ClifTy> = Some(T::TY);
}

/// A C function pointer, and the signature its type spells.
trait HostFn: Copy {
    fn addr(self) -> usize;
    fn params() -> Vec<ClifTy>;
    fn result() -> Option<ClifTy>;
}

macro_rules! host_fn {
    ($($a:ident)*) => {
        impl<R: Answer, $($a: Abi),*> HostFn for unsafe extern "C" fn($($a),*) -> R {
            fn addr(self) -> usize {
                self as usize
            }
            fn params() -> Vec<ClifTy> {
                vec![$($a::TY),*]
            }
            fn result() -> Option<ClifTy> {
                R::TY
            }
        }
    };
}

host_fn!();
host_fn!(A);
host_fn!(A B);
host_fn!(A B C);
host_fn!(A B C D);
host_fn!(A B C D E);
host_fn!(A B C D E F);
host_fn!(A B C D E F G);
host_fn!(A B C D E F G H);
host_fn!(A B C D E F G H I);
host_fn!(A B C D E F G H I J);
host_fn!(A B C D E F G H I J K);
host_fn!(A B C D E F G H I J K L);
host_fn!(A B C D E F G H I J K L M);
host_fn!(A B C D E F G H I J K L M N);
host_fn!(A B C D E F G H I J K L M N O);
host_fn!(A B C D E F G H I J K L M N O P);
host_fn!(A B C D E F G H I J K L M N O P Q);
host_fn!(A B C D E F G H I J K L M N O P Q S);
host_fn!(A B C D E F G H I J K L M N O P Q S T);
host_fn!(A B C D E F G H I J K L M N O P Q S T U);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V W);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V W X);

fn entry<F: HostFn>(name: &'static str, f: F) -> Import {
    Import { name, addr: f.addr(), params: F::params(), result: F::result() }
}

/// Every function a program may import.
pub(crate) fn imports() -> &'static [Import] {
    static TABLE: OnceLock<Vec<Import>> = OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
        // Native code the program carries: place it, unmap it, and the two
        // questions that decide which code to carry in (see ffi/native.rs).
        // Calling it is not an import: it is `Callee::Native`.
        entry("cl_native_load", native::cl_native_load as unsafe extern "C" fn(*const u8, i64) -> i64),
        entry("cl_native_free", native::cl_native_free as unsafe extern "C" fn(i64) -> i32),
        entry("cl_native_arch", native::cl_native_arch as unsafe extern "C" fn() -> i32),
        entry("cl_cpu_has", native::cl_cpu_has as unsafe extern "C" fn(*const u8) -> i32),
        // File + math + stdio
        entry("cl_file_read", file::cl_file_read as unsafe extern "C" fn(*mut u8, i64, i64, i64, i64) -> i64),
        entry("cl_file_read_to_ptr", file::cl_file_read_to_ptr as unsafe extern "C" fn(*const u8, *mut u8, i64, i64) -> i64),
        entry("cl_file_write", file::cl_file_write as unsafe extern "C" fn(*mut u8, i64, i64, i64, i64) -> i64),
        entry("cl_file_write_from_ptr", file::cl_file_write_from_ptr as unsafe extern "C" fn(*const u8, *const u8, i64, i64) -> i64),
        entry("cl_file_create_dir_all", file::cl_file_create_dir_all as unsafe extern "C" fn(*const u8) -> i64),
        entry("cl_stdin_readline", stdio::cl_stdin_readline as unsafe extern "C" fn(*mut u8, i64, i64) -> i64),
        entry("cl_stdout_write", stdio::cl_stdout_write as unsafe extern "C" fn(*mut u8, i64, i64) -> i64),

        // Threads
        entry("cl_thread_start", thread::cl_thread_start as unsafe extern "C" fn(i64, *mut u8) -> i64),
        entry("cl_thread_finish", thread::cl_thread_finish as unsafe extern "C" fn(i64) -> i64),
        // Performance controls
        entry("cl_mem_lock", os::cl_mem_lock as unsafe extern "C" fn(*const u8, i64) -> i32),
        entry("cl_mem_unlock", os::cl_mem_unlock as unsafe extern "C" fn(*const u8, i64) -> i32),
        entry("cl_mem_advise_huge", os::cl_mem_advise_huge as unsafe extern "C" fn(*const u8, i64) -> i32),
        entry("cl_thread_priority", os::cl_thread_priority as unsafe extern "C" fn(i32) -> i32),
        ]
    })
}

/// The import a program names, if base provides one by that name.
pub(crate) fn lookup(name: &str) -> Option<&'static Import> {
    imports().iter().find(|i| i.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_types::clif::{Block, BlockRef, Callee, Function, Inst, Val};

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = imports().iter().map(|i| i.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
    }

    /// Every import, named by a program and with its address taken, links: the
    /// table names only functions that exist, at the signature the table itself
    /// hands the JIT.
    #[test]
    fn every_import_links_at_its_own_signature() {
        let mut insts = Vec::new();
        for (n, import) in imports().iter().enumerate() {
            insts.push(Inst::FuncAddr(
                Val(100 + n as u32),
                Callee::Import(import.name.to_string()),
            ));
        }
        insts.push(Inst::Ret(None));
        let f = Function {
            entry_name: None,
            blocks: vec![Block { reference: BlockRef(0), params: vec![(Val(0), ClifTy::I64)], insts }],
        };
        if let Err(e) = crate::jit::compile(&[f]) {
            panic!("{e}");
        }
    }
}
