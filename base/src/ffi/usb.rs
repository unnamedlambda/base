//! USB through `nusb`: listing, opening a device, claiming an interface, and
//! blocking control, bulk and interrupt transfers. A transfer answers the
//! bytes moved, or `-1`.

use std::time::Duration;

use nusb::transfer::{
    Buffer, Bulk, BulkOrInterrupt, ControlIn, ControlOut, ControlType, In, Interrupt, Out, Recipient,
};
use nusb::{Device, DeviceInfo, Interface, MaybeFuture};

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
unsafe extern "C" fn base_usb_open(i: i32) -> *mut Device {
    match nth(i).map(|d| d.open().wait()) {
        Some(Ok(device)) => Box::into_raw(Box::new(device)),
        _ => std::ptr::null_mut(),
    }
}

/// `dev` closed. An interface claimed through it stays usable until it is
/// released.
///
/// # Safety
///
/// `dev` is a device `base_usb_open` answered, not closed already.
unsafe extern "C" fn base_usb_close(dev: *mut Device) {
    drop(Box::from_raw(dev));
}

/// Interface `iface` of `dev` claimed, detaching a kernel driver from it
/// where the system has one: its handle, or null where the system does not
/// grant it or the number is not one.
///
/// # Safety
///
/// `dev` is an open device.
unsafe extern "C" fn base_usb_claim(dev: *mut Device, iface: i32) -> *mut Interface {
    let Ok(n) = u8::try_from(iface) else { return std::ptr::null_mut() };
    match (*dev).detach_and_claim_interface(n).wait() {
        Ok(i) => Box::into_raw(Box::new(i)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// `intf` released, which ends the system's claim for every handle to it.
///
/// # Safety
///
/// `intf` is an interface `base_usb_claim` answered, not released already.
unsafe extern "C" fn base_usb_release(intf: *mut Interface) {
    drop(Box::from_raw(intf));
}

/// A control transfer through `intf` on its device's default endpoint, its
/// direction, kind and recipient from `request_type` as the SETUP packet has
/// them, and `len` bytes of data at `data`: how many moved, or `-1`.
///
/// # Safety
///
/// `intf` is a held interface, and `data` has `len` bytes: readable for a
/// transfer to the device, writable for one from it.
unsafe extern "C" fn base_usb_control(
    intf: *mut Interface,
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
    let intf = &*intf;
    if request_type & 0x80 != 0 {
        let setup = ControlIn { control_type, recipient, request, value, index, length: len };
        match intf.control_in(setup, timeout).wait() {
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
        match intf.control_out(setup, timeout).wait() {
            Ok(()) => i64::from(len),
            Err(_) => -1,
        }
    }
}

/// A bulk or interrupt transfer of `len` bytes at `data` on `endpoint` of
/// `intf`: how many moved, or `-1`. One from the device asks for a whole
/// number of the endpoint's packets, or the crate fails it.
///
/// # Safety
///
/// As [`base_usb_control`].
unsafe fn data_transfer<T: BulkOrInterrupt>(
    intf: *mut Interface,
    endpoint: i32,
    data: *mut u8,
    len: i64,
    timeout_ms: i32,
) -> i64 {
    // A transfer is one buffer of the crate's, whose length is a `u32`.
    let (Ok(ep), Ok(len), Ok(ms)) =
        (u8::try_from(endpoint), u32::try_from(len), u64::try_from(timeout_ms))
    else {
        return -1;
    };
    let (len, timeout) = (len as usize, Duration::from_millis(ms));
    let intf = &*intf;
    if ep & 0x80 != 0 {
        let Ok(mut e) = intf.endpoint::<T, In>(ep) else { return -1 };
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

unsafe extern "C" fn base_usb_bulk(intf: *mut Interface, endpoint: i32, data: *mut u8, len: i64, timeout_ms: i32) -> i64 {
    data_transfer::<Bulk>(intf, endpoint, data, len, timeout_ms)
}

unsafe extern "C" fn base_usb_interrupt(
    intf: *mut Interface,
    endpoint: i32,
    data: *mut u8,
    len: i64,
    timeout_ms: i32,
) -> i64 {
    data_transfer::<Interrupt>(intf, endpoint, data, len, timeout_ms)
}

crate::ffi::symbols!(
    base_usb_count,
    base_usb_info,
    base_usb_open,
    base_usb_close,
    base_usb_claim,
    base_usb_release,
    base_usb_control,
    base_usb_bulk,
    base_usb_interrupt,
);
