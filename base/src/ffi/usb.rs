//! USB devices, through the `nusb` crate: the C functions a program's
//! `Ext.usb` calls reach, by symbol.
//!
//! Nine blocking calls: how many devices the system lists and what each is —
//! its bus, address, vendor and product — and a device opened by its place in
//! the list, then its interfaces claimed and released and its control, bulk
//! and interrupt transfers made, each waited for. A device is the program's:
//! `base_usb_open` hands back a box it passes to every other call, holding
//! the interfaces claimed through it, and `base_usb_close` drops it and them.
//! The devices are listed afresh at each call. `Host/UsbLib.lean` states
//! them.
//!
//! A transfer answers how many bytes moved, or `-1`. One from the device must
//! ask for a whole number of the endpoint's packets: the device may send up
//! to that much, and a request that ends inside a packet is refused rather
//! than have what overflows it dropped.

use std::collections::HashMap;
use std::time::Duration;

use nusb::transfer::{
    Buffer, Bulk, BulkOrInterrupt, ControlIn, ControlOut, ControlType, In, Interrupt, Out, Recipient,
};
use nusb::{Device, DeviceInfo, Interface, MaybeFuture};

/// An open device, as the program holds it, and the interfaces claimed.
pub(crate) struct Dev {
    device: Device,
    claimed: HashMap<u8, Interface>,
}

fn listed() -> Vec<DeviceInfo> {
    nusb::list_devices().wait().map(|it| it.collect()).unwrap_or_default()
}

fn nth(i: i32) -> Option<DeviceInfo> {
    usize::try_from(i).ok().and_then(|i| listed().into_iter().nth(i))
}

/// A bus identifier as a number: decimal, or hexadecimal after `0x`.
fn bus_number(id: &str) -> i64 {
    match id.strip_prefix("0x") {
        Some(hex) => i64::from_str_radix(hex, 16).unwrap_or(-1),
        None => id.parse().unwrap_or(-1),
    }
}

/// How many devices the system lists.
unsafe extern "C" fn base_usb_count() -> i32 {
    listed().len().min(i32::MAX as usize) as i32
}

/// Device `i`'s bus number (`which = 0`) — where the system numbers its
/// buses, `-1` where it names them otherwise — address (`1`), vendor (`2`) or
/// product (`3`); `-1` for a device not listed or a `which` not here.
unsafe extern "C" fn base_usb_info(i: i32, which: i32) -> i64 {
    let Some(d) = nth(i) else { return -1 };
    match which {
        0 => bus_number(d.bus_id()),
        1 => i64::from(d.device_address()),
        2 => i64::from(d.vendor_id()),
        3 => i64::from(d.product_id()),
        _ => -1,
    }
}

/// Device `i` opened, or null — for a device not listed, or one this
/// process may not open.
unsafe extern "C" fn base_usb_open(i: i32) -> *mut Dev {
    match nth(i).map(|d| d.open().wait()) {
        Some(Ok(device)) => Box::into_raw(Box::new(Dev { device, claimed: HashMap::new() })),
        _ => std::ptr::null_mut(),
    }
}

/// `dev` closed, and every interface claimed through it released.
unsafe extern "C" fn base_usb_close(dev: *mut Dev) {
    drop(Box::from_raw(dev));
}

/// Interface `iface` of `dev` claimed, detaching a kernel driver from it
/// where the system has one: `0`, or `-1`.
unsafe extern "C" fn base_usb_claim(dev: *mut Dev, iface: i32) -> i32 {
    let Ok(n) = u8::try_from(iface) else { return -1 };
    let d = &mut *dev;
    if d.claimed.contains_key(&n) {
        return 0;
    }
    match d.device.detach_and_claim_interface(n).wait() {
        Ok(i) => {
            d.claimed.insert(n, i);
            0
        }
        Err(_) => -1,
    }
}

/// Interface `iface` of `dev` released: `0`, or `-1` for one not claimed.
unsafe extern "C" fn base_usb_release(dev: *mut Dev, iface: i32) -> i32 {
    let claimed = u8::try_from(iface).ok().and_then(|n| (*dev).claimed.remove(&n));
    if claimed.is_some() {
        0
    } else {
        -1
    }
}

/// A control transfer on `dev`'s default endpoint, its direction, kind and
/// recipient from `request_type` as the SETUP packet has them, and `len`
/// bytes of data at `data`: how many moved, or `-1`. It goes through an
/// interface claimed where there is one, which Windows requires.
unsafe extern "C" fn base_usb_control(
    dev: *mut Dev,
    request_type: i32,
    request: i32,
    value: i32,
    index: i32,
    data: *mut u8,
    len: i32,
    timeout_ms: i32,
) -> i64 {
    let (Ok(len), Ok(ms)) = (u16::try_from(len), u64::try_from(timeout_ms)) else { return -1 };
    let control_type = match (request_type >> 5) & 3 {
        0 => ControlType::Standard,
        1 => ControlType::Class,
        2 => ControlType::Vendor,
        _ => return -1,
    };
    let recipient = match request_type & 0x1f {
        0 => Recipient::Device,
        1 => Recipient::Interface,
        2 => Recipient::Endpoint,
        3 => Recipient::Other,
        _ => return -1,
    };
    let (request, value, index) = (request as u8, value as u16, index as u16);
    let timeout = Duration::from_millis(ms);
    let d = &*dev;
    let via = d.claimed.values().next();
    if request_type & 0x80 != 0 {
        let setup = ControlIn { control_type, recipient, request, value, index, length: len };
        let got = match via {
            Some(i) => i.control_in(setup, timeout).wait(),
            #[cfg(not(windows))]
            None => d.device.control_in(setup, timeout).wait(),
            #[cfg(windows)]
            None => return -1,
        };
        match got {
            Ok(bytes) => {
                let n = bytes.len().min(len as usize);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, n);
                n as i64
            }
            Err(_) => -1,
        }
    } else {
        let out = if len == 0 { &[][..] } else { std::slice::from_raw_parts(data, len as usize) };
        let setup = ControlOut { control_type, recipient, request, value, index, data: out };
        let sent = match via {
            Some(i) => i.control_out(setup, timeout).wait(),
            #[cfg(not(windows))]
            None => d.device.control_out(setup, timeout).wait(),
            #[cfg(windows)]
            None => return -1,
        };
        match sent {
            Ok(()) => i64::from(len),
            Err(_) => -1,
        }
    }
}

/// A bulk or interrupt transfer of `len` bytes at `data` on `endpoint` of
/// interface `iface`, claimed: how many moved, or `-1`.
unsafe fn data_transfer<T: BulkOrInterrupt>(
    dev: *mut Dev,
    iface: i32,
    endpoint: i32,
    data: *mut u8,
    len: i64,
    timeout_ms: i32,
) -> i64 {
    let (Ok(n), Ok(ep), Ok(len), Ok(ms)) =
        (u8::try_from(iface), u8::try_from(endpoint), usize::try_from(len), u64::try_from(timeout_ms))
    else {
        return -1;
    };
    let Some(intf) = (*dev).claimed.get(&n) else { return -1 };
    let timeout = Duration::from_millis(ms);
    if ep & 0x80 != 0 {
        let Ok(mut e) = intf.endpoint::<T, In>(ep) else { return -1 };
        let packet = e.max_packet_size();
        if len == 0 || packet == 0 || len % packet != 0 {
            return -1;
        }
        let done = e.transfer_blocking(Buffer::new(len), timeout);
        if done.status.is_err() {
            return -1;
        }
        let got = &done.buffer[..done.actual_len.min(len)];
        std::ptr::copy_nonoverlapping(got.as_ptr(), data, got.len());
        got.len() as i64
    } else {
        let Ok(mut e) = intf.endpoint::<T, Out>(ep) else { return -1 };
        let out: &[u8] = if len == 0 { &[] } else { std::slice::from_raw_parts(data, len) };
        let done = e.transfer_blocking(Buffer::from(out), timeout);
        match done.status {
            Ok(()) => done.actual_len as i64,
            Err(_) => -1,
        }
    }
}

unsafe extern "C" fn base_usb_bulk(dev: *mut Dev, iface: i32, endpoint: i32, data: *mut u8, len: i64, timeout_ms: i32) -> i64 {
    data_transfer::<Bulk>(dev, iface, endpoint, data, len, timeout_ms)
}

unsafe extern "C" fn base_usb_interrupt(
    dev: *mut Dev,
    iface: i32,
    endpoint: i32,
    data: *mut u8,
    len: i64,
    timeout_ms: i32,
) -> i64 {
    data_transfer::<Interrupt>(dev, iface, endpoint, data, len, timeout_ms)
}

/// The address of `symbol`, or `None` for one this library does not have.
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "base_usb_count" => base_usb_count as usize,
        "base_usb_info" => base_usb_info as usize,
        "base_usb_open" => base_usb_open as usize,
        "base_usb_close" => base_usb_close as usize,
        "base_usb_claim" => base_usb_claim as usize,
        "base_usb_release" => base_usb_release as usize,
        "base_usb_control" => base_usb_control as usize,
        "base_usb_bulk" => base_usb_bulk as usize,
        "base_usb_interrupt" => base_usb_interrupt as usize,
        _ => return None,
    })
}
