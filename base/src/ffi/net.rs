use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::net::{TcpListener, TcpStream};

use super::{clear_ctx_slot, read_cstr_ptr, read_ctx_mut, write_ctx_slot};

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
