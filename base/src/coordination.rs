use portable_atomic::{AtomicU64, Ordering};
use std::sync::atomic::fence;
use tracing::debug;

pub const MAILBOX_CLOSED: u64 = u64::MAX;

pub(crate) struct Mailbox(AtomicU64);

pub(crate) enum MailboxPoll {
    Empty,
    Work { src: u32, count: u32, flag: u32 },
    Closed,
}

impl Mailbox {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Post work to the mailbox.
    /// - count=0: direct dispatch — `src` is the CLIF function index
    /// - count>0: batch dispatch — `src` is the first action descriptor index,
    ///   worker iterates actions\[src..src+count\] reading fn_idx and mem_offset from each
    pub fn post(&self, src: u32, count: u32, flag: u32) {
        debug!(src, count, flag, "mailbox_post");
        let packed = ((src as u64) << 43) | ((count as u64) << 22) | (flag as u64);
        // Ensure packed != 0 (empty sentinel). When src=0, count=0, flag=0 the packed
        // value would be 0. Set the top bit to disambiguate.
        let packed = if packed == 0 { 1u64 << 63 } else { packed };
        let mut spin_count = 0u32;
        loop {
            match self
                .0
                .compare_exchange(0, packed, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(_) => spin_backoff(&mut spin_count),
            }
        }
    }

    pub fn shutdown(&self) {
        debug!("mailbox_shutdown");
        self.0.store(MAILBOX_CLOSED, Ordering::Release);
    }

    pub fn poll(&self) -> MailboxPoll {
        let packed = self.0.swap(0, Ordering::AcqRel);
        if packed == 0 {
            return MailboxPoll::Empty;
        }
        if packed == MAILBOX_CLOSED {
            return MailboxPoll::Closed;
        }
        // Undo the top-bit sentinel for the src=0,count=0,flag=0 case
        let packed = if packed == (1u64 << 63) { 0 } else { packed };
        let src = (packed >> 43) as u32;
        let count = ((packed >> 22) & 0x1F_FFFF) as u32;
        let flag = (packed & 0x3F_FFFF) as u32;
        MailboxPoll::Work { src, count, flag }
    }
}

pub(crate) fn spin_backoff(spin_count: &mut u32) {
    *spin_count += 1;
    if *spin_count < 100 {
        std::hint::spin_loop();
    } else if *spin_count < 1000 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(std::time::Duration::from_micros(1));
    }
}

pub(crate) struct SharedMemory {
    pub(crate) ptr: *mut u8,
}

unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    pub fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    pub unsafe fn read(&self, offset: usize, size: usize) -> &[u8] {
        std::slice::from_raw_parts(self.ptr.add(offset), size)
    }

    // Use a true atomic op when the pointer is naturally aligned; fall back to
    // an unaligned read + fence otherwise.
    pub unsafe fn load_u64(&self, offset: usize, order: Ordering) -> u64 {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x7 == 0 {
            return (*(ptr as *const AtomicU64)).load(order);
        }
        let value = std::ptr::read_unaligned(ptr as *const u64);
        if matches!(
            order,
            Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst
        ) {
            fence(Ordering::Acquire);
        }
        value
    }

    pub unsafe fn store_u64(&self, offset: usize, value: u64, order: Ordering) {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x7 == 0 {
            (*(ptr as *const AtomicU64)).store(value, order);
            return;
        }
        if matches!(
            order,
            Ordering::Release | Ordering::AcqRel | Ordering::SeqCst
        ) {
            fence(Ordering::Release);
        }
        std::ptr::write_unaligned(ptr as *mut u64, value);
    }

    // CAS requires natural alignment — there is no unaligned fallback.
    pub unsafe fn cas64(&self, offset: usize, expected: u64, new: u64) -> u64 {
        let ptr = self.ptr.add(offset);
        debug_assert!(
            (ptr as usize) & 0x7 == 0,
            "cas64: pointer not 8-byte aligned (offset {offset})"
        );
        (*(ptr as *const AtomicU64))
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }
}

pub(crate) fn order_from_u32(raw: u32) -> Ordering {
    match raw {
        1 => Ordering::Acquire,
        2 => Ordering::Release,
        3 => Ordering::AcqRel,
        4 => Ordering::SeqCst,
        _ => Ordering::Relaxed,
    }
}

pub(crate) unsafe fn load_sized(
    shared: &SharedMemory,
    offset: usize,
    size: u32,
    order: Ordering,
) -> u64 {
    match size {
        1 => shared.read(offset, 1)[0] as u64,
        2 => u16::from_le_bytes(shared.read(offset, 2)[0..2].try_into().unwrap()) as u64,
        4 => u32::from_le_bytes(shared.read(offset, 4)[0..4].try_into().unwrap()) as u64,
        8 => shared.load_u64(offset, order),
        _ => 0,
    }
}
