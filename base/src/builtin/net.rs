//! TCP and UDP through `std::net`, on handles: a name resolved to addresses; a
//! listener bound, asked its address and accepted from without waiting; a
//! stream connected or accepted, sent on, received from, shut for writing and
//! closed; a UDP socket bound, sent from and received on.
//!
//! An address is a 24-byte record: the family (`4` or `6`) at byte 0, the port
//! at 2 and the IPv6 scope at 4, little-endian, and the IP at 8 (an IPv4
//! address in its first four bytes). A stream or socket waits at most the
//! timeout it was made with for what it receives and sends; a receive that
//! times out answers `-2`.

use std::ffi::c_char;
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, SocketAddrV6, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Duration;

use super::read_field;

/// The bytes of an address record.
const ADDR: usize = 24;

/// The largest datagram UDP carries.
const DATAGRAM: usize = 65536;

/// A listener, as the program holds it.
pub(crate) struct Listener(TcpListener);
/// A connected stream.
pub(crate) struct Stream(TcpStream);
/// A bound UDP socket.
pub(crate) struct Udp(UdpSocket);

/// The address in the record at `a`, or `None` for a family not `4` or `6`.
unsafe fn addr_in(a: *const u8) -> Option<SocketAddr> {
    let r = std::slice::from_raw_parts(a, ADDR);
    let port = u16::from_le_bytes([r[2], r[3]]);
    match r[0] {
        4 => Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(r[8], r[9], r[10], r[11])), port)),
        6 => {
            let ip: [u8; 16] = r[8..24].try_into().ok()?;
            let scope = u32::from_le_bytes([r[4], r[5], r[6], r[7]]);
            Some(SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::from(ip), port, 0, scope)))
        }
        _ => None,
    }
}

/// `a` as the record written at `out`.
unsafe fn addr_out(a: SocketAddr, out: *mut u8) {
    let mut r = [0u8; ADDR];
    r[2..4].copy_from_slice(&a.port().to_le_bytes());
    match a {
        SocketAddr::V4(v4) => {
            r[0] = 4;
            r[8..12].copy_from_slice(&v4.ip().octets());
        }
        SocketAddr::V6(v6) => {
            r[0] = 6;
            r[4..8].copy_from_slice(&v6.scope_id().to_le_bytes());
            r[8..24].copy_from_slice(&v6.ip().octets());
        }
    }
    std::ptr::copy_nonoverlapping(r.as_ptr(), out, ADDR);
}

/// A timeout of `ms` milliseconds, or `None` where it is not positive.
fn timeout(ms: i32) -> Option<Duration> {
    u64::try_from(ms).ok().filter(|&m| m > 0).map(Duration::from_millis)
}

/// A receive's answer: the bytes, `-2` for a timeout, `-1` otherwise.
fn received(r: std::io::Result<usize>) -> i64 {
    match r {
        Ok(n) => n as i64,
        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => -2,
        Err(_) => -1,
    }
}

/// The addresses the name in the `cap`-byte field at `name` ([`read_field`])
/// resolves to, through the system's resolver, each with port `0`: how many,
/// as many of their records as fit in `out_cap` bytes written at `out`; `-1`
/// where it does not resolve.
unsafe extern "C" fn base_net_resolve(name: *const u8, cap: i64, out: *mut u8, out_cap: i64) -> i64 {
    let Some(name) = read_field(name, cap).filter(|n| !n.is_empty()) else { return -1 };
    let Ok(found) = (name, 0u16).to_socket_addrs() else { return -1 };
    let found: Vec<SocketAddr> = found.collect();
    let room = usize::try_from(out_cap).unwrap_or(0) / ADDR;
    for (k, a) in found.iter().take(room).enumerate() {
        addr_out(*a, out.add(k * ADDR));
    }
    found.len() as i64
}

/// A stream connected to the address in the record at `addr`, sending and
/// receiving with a timeout of `timeout_ms`; null where it does not connect
/// within it, or the timeout is not positive.
unsafe extern "C" fn base_net_tcp_connect(addr: *const u8, timeout_ms: i32) -> *mut Stream {
    let (Some(a), Some(t)) = (addr_in(addr), timeout(timeout_ms)) else { return std::ptr::null_mut() };
    match TcpStream::connect_timeout(&a, t) {
        Ok(s) if s.set_read_timeout(Some(t)).is_ok() && s.set_write_timeout(Some(t)).is_ok() => {
            Box::into_raw(Box::new(Stream(s)))
        }
        _ => std::ptr::null_mut(),
    }
}

/// A listener bound to the address in the record at `addr` (port `0` for one
/// the system picks); null where it does not bind.
unsafe extern "C" fn base_net_tcp_listen(addr: *const u8) -> *mut Listener {
    let Some(a) = addr_in(addr) else { return std::ptr::null_mut() };
    match TcpListener::bind(a) {
        Ok(l) if l.set_nonblocking(true).is_ok() => Box::into_raw(Box::new(Listener(l))),
        _ => std::ptr::null_mut(),
    }
}

/// The address `l` is bound to, written as a record at `out`: `0`, or `-1`.
unsafe extern "C" fn base_net_tcp_local(l: *mut Listener, out: *mut u8) -> i64 {
    match (*l).0.local_addr() {
        Ok(a) => {
            addr_out(a, out);
            0
        }
        Err(_) => -1,
    }
}

/// A connection waiting on `l`, accepted as a stream that sends and receives
/// with a timeout of `timeout_ms`; null where none is waiting, it fails, or
/// the timeout is not positive. It does not wait.
unsafe extern "C" fn base_net_tcp_accept(l: *mut Listener, timeout_ms: i32) -> *mut Stream {
    let Some(t) = timeout(timeout_ms) else { return std::ptr::null_mut() };
    match (*l).0.accept() {
        Ok((s, _))
            if s.set_nonblocking(false).is_ok()
                && s.set_read_timeout(Some(t)).is_ok()
                && s.set_write_timeout(Some(t)).is_ok() =>
        {
            Box::into_raw(Box::new(Stream(s)))
        }
        _ => std::ptr::null_mut(),
    }
}

/// The `len` bytes at `buf` sent on `s`, all of them: `len`, or `-1`.
unsafe extern "C" fn base_net_tcp_send(s: *mut Stream, buf: *const u8, len: i64) -> i64 {
    let Ok(n) = usize::try_from(len) else { return -1 };
    let data = if n == 0 { &[][..] } else { std::slice::from_raw_parts(buf, n) };
    match (*s).0.write_all(data) {
        Ok(()) => len,
        Err(_) => -1,
    }
}

/// Bytes that have arrived on `s`, at most `len`, read into `buf`: how many,
/// waiting up to the stream's timeout for a first; `0` once the other end has
/// shut it, `-2` when nothing came in time, `-1` otherwise.
unsafe extern "C" fn base_net_tcp_recv(s: *mut Stream, buf: *mut u8, len: i64) -> i64 {
    let Ok(n) = usize::try_from(len) else { return -1 };
    if n == 0 {
        return 0;
    }
    loop {
        match (*s).0.read(std::slice::from_raw_parts_mut(buf, n)) {
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            r => return received(r),
        }
    }
}

/// `s` shut for sending, so the other end receives its end: `0`, or `-1`.
unsafe extern "C" fn base_net_tcp_shutdown(s: *mut Stream) -> i64 {
    match (*s).0.shutdown(Shutdown::Write) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// `s` closed.
unsafe extern "C" fn base_net_tcp_close(s: *mut Stream) {
    drop(Box::from_raw(s));
}

/// `l` closed.
unsafe extern "C" fn base_net_tcp_listener_close(l: *mut Listener) {
    drop(Box::from_raw(l));
}

/// A UDP socket bound to the address in the record at `addr` (port `0` for
/// one the system picks), receiving with a timeout of `timeout_ms`; null
/// where it does not bind or the timeout is not positive.
unsafe extern "C" fn base_net_udp_bind(addr: *const u8, timeout_ms: i32) -> *mut Udp {
    let (Some(a), Some(t)) = (addr_in(addr), timeout(timeout_ms)) else { return std::ptr::null_mut() };
    match UdpSocket::bind(a) {
        Ok(u) if u.set_read_timeout(Some(t)).is_ok() => Box::into_raw(Box::new(Udp(u))),
        _ => std::ptr::null_mut(),
    }
}

/// The address `u` is bound to, written as a record at `out`: `0`, or `-1`.
unsafe extern "C" fn base_net_udp_local(u: *mut Udp, out: *mut u8) -> i64 {
    match (*u).0.local_addr() {
        Ok(a) => {
            addr_out(a, out);
            0
        }
        Err(_) => -1,
    }
}

/// The `len` bytes at `buf` sent from `u` as one datagram to the address in
/// the record at `addr`: `len`, or `-1`.
unsafe extern "C" fn base_net_udp_send(u: *mut Udp, buf: *const u8, len: i64, addr: *const u8) -> i64 {
    let (Ok(n), Some(a)) = (usize::try_from(len), addr_in(addr)) else { return -1 };
    let data = if n == 0 { &[][..] } else { std::slice::from_raw_parts(buf, n) };
    match (*u).0.send_to(data, a) {
        Ok(k) if k == n => len,
        _ => -1,
    }
}

std::thread_local! {
    /// A whole datagram, received before the part that fits is copied out.
    static DGRAM: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0; DATAGRAM]);
}

/// The next datagram to arrive at `u`, waiting up to its timeout: the
/// sender's record written at `buf` and as much of the datagram as fits in
/// the `len` bytes there after it, answering the datagram's whole length;
/// `-2` when none came in time, `-1` otherwise or for room under a record.
unsafe extern "C" fn base_net_udp_recv(u: *mut Udp, buf: *mut c_char, len: i64) -> i64 {
    let Ok(room) = usize::try_from(len) else { return -1 };
    if room < ADDR {
        return -1;
    }
    DGRAM.with(|d| {
        let mut d = d.borrow_mut();
        loop {
            match (*u).0.recv_from(&mut d) {
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Ok((n, from)) => {
                    addr_out(from, buf.cast());
                    std::ptr::copy_nonoverlapping(d.as_ptr(), buf.add(ADDR).cast(), n.min(room - ADDR));
                    return n as i64;
                }
                Err(e) => return received(Err(e)),
            }
        }
    })
}

/// `u` closed.
unsafe extern "C" fn base_net_udp_close(u: *mut Udp) {
    drop(Box::from_raw(u));
}

crate::builtin::symbols!(
    base_net_resolve,
    base_net_tcp_connect,
    base_net_tcp_listen,
    base_net_tcp_local,
    base_net_tcp_accept,
    base_net_tcp_send,
    base_net_tcp_recv,
    base_net_tcp_shutdown,
    base_net_tcp_close,
    base_net_tcp_listener_close,
    base_net_udp_bind,
    base_net_udp_local,
    base_net_udp_send,
    base_net_udp_recv,
    base_net_udp_close,
);

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback() -> [u8; ADDR] {
        let mut r = [0u8; ADDR];
        r[0] = 4;
        r[8..12].copy_from_slice(&[127, 0, 0, 1]);
        r
    }

    #[test]
    fn a_stream_carries_bytes_both_ways_and_its_end() {
        let at = loopback();
        let mut bound = [0u8; ADDR];
        unsafe {
            let l = base_net_tcp_listen(at.as_ptr());
            assert!(!l.is_null());
            assert_eq!(base_net_tcp_local(l, bound.as_mut_ptr()), 0);
            assert_ne!(u16::from_le_bytes([bound[2], bound[3]]), 0);
            let c = base_net_tcp_connect(bound.as_ptr(), 2000);
            assert!(!c.is_null());
            let mut s = std::ptr::null_mut();
            for _ in 0..200 {
                s = base_net_tcp_accept(l, 2000);
                if !s.is_null() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(!s.is_null());
            assert_eq!(base_net_tcp_send(c, b"ping".as_ptr(), 4), 4);
            let mut got = [0u8; 8];
            assert_eq!(base_net_tcp_recv(s, got.as_mut_ptr(), 8), 4);
            assert_eq!(&got[..4], b"ping");
            assert_eq!(base_net_tcp_shutdown(c), 0);
            assert_eq!(base_net_tcp_recv(s, got.as_mut_ptr(), 8), 0);
            base_net_tcp_close(c);
            base_net_tcp_close(s);
            assert!(base_net_tcp_accept(l, 10).is_null(), "none waiting");
            base_net_tcp_listener_close(l);
        }
    }

    #[test]
    fn a_datagram_arrives_with_its_sender() {
        let at = loopback();
        let (mut a, mut b) = ([0u8; ADDR], [0u8; ADDR]);
        unsafe {
            let x = base_net_udp_bind(at.as_ptr(), 2000);
            let y = base_net_udp_bind(at.as_ptr(), 50);
            assert_eq!(base_net_udp_local(x, a.as_mut_ptr()), 0);
            assert_eq!(base_net_udp_local(y, b.as_mut_ptr()), 0);
            assert_eq!(base_net_udp_send(y, b"hi!".as_ptr(), 3, a.as_ptr()), 3);
            let mut got = [0u8; ADDR + 2];
            assert_eq!(base_net_udp_recv(x, got.as_mut_ptr().cast(), got.len() as i64), 3);
            assert_eq!(&got[..ADDR], &b);
            assert_eq!(&got[ADDR..], b"hi");
            assert_eq!(base_net_udp_recv(y, got.as_mut_ptr().cast(), got.len() as i64), -2);
            base_net_udp_close(x);
            base_net_udp_close(y);
        }
    }
}
