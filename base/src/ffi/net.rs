use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::net::{TcpListener, TcpStream};

use super::{clear_ctx_slot, read_cstr_ptr, read_ctx_mut, read_ctx_ref, write_ctx_slot};

pub(crate) struct CraneliftNetContext {
    connections: HashMap<u32, TcpStream>,
    listeners: HashMap<u32, TcpListener>,
    next_handle: u32,
}

pub(crate) unsafe extern "C" fn cl_net_init(ctx_slot_ptr: *mut *mut CraneliftNetContext) {
    let ctx = Box::new(CraneliftNetContext {
        connections: HashMap::new(),
        listeners: HashMap::new(),
        next_handle: 1,
    });
    let _ = write_ctx_slot(ctx_slot_ptr, Box::into_raw(ctx));
}

pub(crate) unsafe extern "C" fn cl_net_listen(
    ctx_ptr: *mut CraneliftNetContext,
    addr_ptr: *const u8,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftNetContext>(ctx_ptr) else {
        return 0;
    };
    let addr = read_cstr_ptr(addr_ptr);
    match TcpListener::bind(&addr) {
        Ok(listener) => {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.listeners.insert(handle, listener);
            handle as i64
        }
        Err(_) => 0,
    }
}

pub(crate) unsafe extern "C" fn cl_net_connect(
    ctx_ptr: *mut CraneliftNetContext,
    addr_ptr: *const u8,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftNetContext>(ctx_ptr) else {
        return 0;
    };
    let addr = read_cstr_ptr(addr_ptr);
    match TcpStream::connect(&addr) {
        Ok(stream) => {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.connections.insert(handle, stream);
            handle as i64
        }
        Err(_) => 0,
    }
}

pub(crate) unsafe extern "C" fn cl_net_listener_port(
    ctx_ptr: *const CraneliftNetContext,
    listener: i64,
) -> i64 {
    let Some(ctx) = read_ctx_ref::<CraneliftNetContext>(ctx_ptr) else {
        return -1;
    };
    match ctx.listeners.get(&(listener as u32)) {
        Some(l) => match l.local_addr() {
            Ok(a) => a.port() as i64,
            Err(_) => -1,
        },
        None => -1,
    }
}

pub(crate) unsafe extern "C" fn cl_net_accept(
    ctx_ptr: *mut CraneliftNetContext,
    listener: i64,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftNetContext>(ctx_ptr) else {
        return 0;
    };
    if let Some(l) = ctx.listeners.get(&(listener as u32)) {
        if let Ok((stream, _)) = l.accept() {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.connections.insert(handle, stream);
            return handle as i64;
        }
    }
    0
}

pub(crate) unsafe extern "C" fn cl_net_send(
    ctx_ptr: *mut CraneliftNetContext,
    conn: i64,
    src_ptr: *const u8,
    size: i64,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftNetContext>(ctx_ptr) else {
        return -1;
    };
    if let Some(stream) = ctx.connections.get_mut(&(conn as u32)) {
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        match IoWrite::write_all(stream, data) {
            Ok(_) => return 0,
            Err(_) => return -1,
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_net_recv(
    ctx_ptr: *mut CraneliftNetContext,
    conn: i64,
    dst_ptr: *mut u8,
    size: i64,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftNetContext>(ctx_ptr) else {
        return -1;
    };
    if let Some(stream) = ctx.connections.get_mut(&(conn as u32)) {
        let buf = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        let mut total = 0;
        while total < size as usize {
            match IoRead::read(stream, &mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(_) => return -1,
            }
        }
        return total as i64;
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_net_cleanup(ctx_slot_ptr: *mut *mut CraneliftNetContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftNetContext>(ctx_slot_ptr);
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn init_then_cleanup_lifecycle() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe {
            cl_net_init(&mut slot);
            assert!(!slot.is_null(), "init should populate the slot");
            cl_net_cleanup(&mut slot);
            assert!(slot.is_null(), "cleanup should null the slot");
        }
    }

    #[test]
    fn listen_returns_positive_handle() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let addr = CString::new("127.0.0.1:0").unwrap();
        unsafe {
            cl_net_init(&mut slot);
            let h = cl_net_listen(slot, addr.as_ptr() as *const u8);
            assert!(h > 0);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn listen_bad_address_returns_zero() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let bad = CString::new("not-a-valid-address:99999").unwrap();
        unsafe {
            cl_net_init(&mut slot);
            assert_eq!(cl_net_listen(slot, bad.as_ptr() as *const u8), 0);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn connect_unreachable_returns_zero() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let addr = CString::new("127.0.0.1:1").unwrap();
        unsafe {
            cl_net_init(&mut slot);
            assert_eq!(cl_net_connect(slot, addr.as_ptr() as *const u8), 0);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn distinct_listens_return_distinct_handles() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let addr = CString::new("127.0.0.1:0").unwrap();
        unsafe {
            cl_net_init(&mut slot);
            let h1 = cl_net_listen(slot, addr.as_ptr() as *const u8);
            let h2 = cl_net_listen(slot, addr.as_ptr() as *const u8);
            assert!(h1 > 0 && h2 > 0 && h1 != h2);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn listener_port_returns_assigned_port() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let addr = CString::new("127.0.0.1:0").unwrap();
        unsafe {
            cl_net_init(&mut slot);
            let h = cl_net_listen(slot, addr.as_ptr() as *const u8);
            let port = cl_net_listener_port(slot, h);
            assert!(port > 0 && port <= 65535);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn listener_port_invalid_handle_returns_neg1() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe {
            cl_net_init(&mut slot);
            assert_eq!(cl_net_listener_port(slot, 999), -1);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn listen_accept_send_recv_roundtrip() {
        let addr = CString::new("127.0.0.1:0").unwrap();
        let payload = b"roundtrip payload";

        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe {
            cl_net_init(&mut slot);
            let listen_h = cl_net_listen(slot, addr.as_ptr() as *const u8);
            assert!(listen_h > 0);
            let port = cl_net_listener_port(slot, listen_h) as u16;

            let client = std::thread::spawn(move || {
                let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
                s.write_all(payload).unwrap();
                let mut echo = [0u8; 17];
                s.read_exact(&mut echo).unwrap();
                echo
            });

            let conn_h = cl_net_accept(slot, listen_h);
            assert!(conn_h > 0);

            let mut buf = [0u8; 17];
            let n = cl_net_recv(slot, conn_h, buf.as_mut_ptr(), buf.len() as i64);
            assert_eq!(n, payload.len() as i64);
            assert_eq!(&buf, payload);

            let sent = cl_net_send(slot, conn_h, buf.as_ptr(), buf.len() as i64);
            assert_eq!(sent, 0);

            let echo = client.join().unwrap();
            assert_eq!(&echo, payload);

            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn connect_send_recv_roundtrip() {
        // Mirror of the above, but using cl_net_connect against a server-side
        // std TcpListener so we exercise the connect path end-to-end.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let addr = CString::new(format!("127.0.0.1:{port}")).unwrap();
        let payload = b"connect roundtrip";

        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 17];
            s.read_exact(&mut buf).unwrap();
            s.write_all(&buf).unwrap();
        });

        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe {
            cl_net_init(&mut slot);
            let conn_h = cl_net_connect(slot, addr.as_ptr() as *const u8);
            assert!(conn_h > 0);

            let sent = cl_net_send(slot, conn_h, payload.as_ptr(), payload.len() as i64);
            assert_eq!(sent, 0);

            let mut buf = [0u8; 17];
            let n = cl_net_recv(slot, conn_h, buf.as_mut_ptr(), buf.len() as i64);
            assert_eq!(n, payload.len() as i64);
            assert_eq!(&buf, payload);

            cl_net_cleanup(&mut slot);
        }

        server.join().unwrap();
    }

    #[test]
    fn send_recv_on_invalid_handle_returns_neg1() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        let buf = [0u8; 4];
        unsafe {
            cl_net_init(&mut slot);
            assert_eq!(cl_net_send(slot, 42, buf.as_ptr(), 4), -1);
            let mut dst = [0u8; 4];
            assert_eq!(cl_net_recv(slot, 42, dst.as_mut_ptr(), 4), -1);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn accept_on_invalid_handle_returns_zero() {
        let mut slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe {
            cl_net_init(&mut slot);
            assert_eq!(cl_net_accept(slot, 999), 0);
            cl_net_cleanup(&mut slot);
        }
    }

    #[test]
    fn null_ctx_pointers_return_zero_or_neg1() {
        let null_ctx = std::ptr::null_mut::<CraneliftNetContext>();
        let addr = CString::new("127.0.0.1:0").unwrap();
        let buf = [0u8; 4];
        unsafe {
            assert_eq!(cl_net_listen(null_ctx, addr.as_ptr() as *const u8), 0);
            assert_eq!(cl_net_listener_port(null_ctx as *const _, 1), -1);
            assert_eq!(cl_net_connect(null_ctx, addr.as_ptr() as *const u8), 0);
            assert_eq!(cl_net_accept(null_ctx, 1), 0);
            assert_eq!(cl_net_send(null_ctx, 1, buf.as_ptr(), 4), -1);
            let mut dst = [0u8; 4];
            assert_eq!(cl_net_recv(null_ctx, 1, dst.as_mut_ptr(), 4), -1);
        }
    }

    #[test]
    fn cleanup_on_null_slot_is_noop() {
        let mut null_slot: *mut CraneliftNetContext = std::ptr::null_mut();
        unsafe { cl_net_cleanup(&mut null_slot) };
        assert!(null_slot.is_null());
    }
}
