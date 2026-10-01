//! Serial ports, through the `serialport` crate: the C functions a program's
//! `Ext.serial` calls reach, by symbol.
//!
//! Seven calls: how many ports the system lists and the name of each, and a
//! port opened by name — at a baud rate, eight data bits, no parity, one stop
//! bit and no flow control — then read, written, asked how much is waiting,
//! and closed. A port is the program's: `base_serial_open` hands back a box it
//! passes to every other call, and `base_serial_close` drops. Nothing is kept
//! here, and the ports are listed afresh at each call. `Host/SerialLib.lean`
//! states them.

use std::ffi::c_char;
use std::io::{Read, Write};
use std::time::Duration;

use serialport::SerialPort;

/// An open port, as the program holds it.
pub(crate) struct Port(Box<dyn SerialPort>);

fn names() -> Vec<String> {
    serialport::available_ports().map(|ps| ps.into_iter().map(|p| p.port_name).collect()).unwrap_or_default()
}

/// How many ports the system lists.
unsafe extern "C" fn base_serial_count() -> i32 {
    names().len().min(i32::MAX as usize) as i32
}

/// Port `i`'s name, as many of its bytes as fit in `cap` written at `out`:
/// the name's whole length, or `-1` for a port not listed.
unsafe extern "C" fn base_serial_name(i: i32, out: *mut u8, cap: i64) -> i64 {
    let names = names();
    let Some(name) = usize::try_from(i).ok().and_then(|i| names.get(i)) else { return -1 };
    let n = name.len().min(cap.max(0) as usize);
    std::ptr::copy_nonoverlapping(name.as_ptr(), out, n);
    name.len() as i64
}

/// The port named by the C string at `path`, at `baud`, with reads waiting
/// up to `timeout_ms` for a first byte; null where it does not open.
unsafe extern "C" fn base_serial_open(path: *const c_char, baud: i32, timeout_ms: i32) -> *mut Port {
    let path = super::read_cstr_ptr(path as *const u8);
    if path.is_empty() || baud <= 0 || timeout_ms < 0 {
        return std::ptr::null_mut();
    }
    let builder = serialport::new(path, baud as u32).timeout(Duration::from_millis(timeout_ms as u64));
    match builder.open() {
        Ok(p) => Box::into_raw(Box::new(Port(p))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// `port` closed.
unsafe extern "C" fn base_serial_close(port: *mut Port) {
    drop(Box::from_raw(port));
}

/// Up to `len` bytes that have arrived, read into `buf`: how many, `0` when
/// none came within the timeout, or `-1`.
unsafe extern "C" fn base_serial_read(port: *mut Port, buf: *mut u8, len: i64) -> i64 {
    let Ok(len) = usize::try_from(len) else { return -1 };
    match (*port).0.read(std::slice::from_raw_parts_mut(buf, len)) {
        Ok(n) => n as i64,
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => 0,
        Err(_) => -1,
    }
}

/// The `len` bytes at `buf` written to `port`, all of them: `len`, or `-1`.
unsafe extern "C" fn base_serial_write(port: *mut Port, buf: *const u8, len: i64) -> i64 {
    let Ok(n) = usize::try_from(len) else { return -1 };
    let p = &mut (*port).0;
    match p.write_all(std::slice::from_raw_parts(buf, n)).and_then(|()| p.flush()) {
        Ok(()) => len,
        Err(_) => -1,
    }
}

/// How many bytes have arrived and wait to be read, or `-1`.
unsafe extern "C" fn base_serial_pending(port: *mut Port) -> i64 {
    match (*port).0.bytes_to_read() {
        Ok(n) => i64::from(n),
        Err(_) => -1,
    }
}

/// The address of `symbol`, or `None` for one this library does not have.
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "base_serial_count" => base_serial_count as usize,
        "base_serial_name" => base_serial_name as usize,
        "base_serial_open" => base_serial_open as usize,
        "base_serial_close" => base_serial_close as usize,
        "base_serial_read" => base_serial_read as usize,
        "base_serial_write" => base_serial_write as usize,
        "base_serial_pending" => base_serial_pending as usize,
        _ => return None,
    })
}
