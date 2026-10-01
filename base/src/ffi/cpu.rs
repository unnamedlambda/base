//! Which CPUs there are and where a thread runs: the C functions a program's
//! `Ext.cpu` calls reach, by symbol.
//!
//! Five stateless calls over what the operating system itself answers: how
//! many logical CPUs are online, the physical core and package of each, and
//! pinning the calling thread to one of them or releasing it. Each answers
//! `-1` where the system does not say or does not grant it — a CPU that is
//! not there, an OS with no such thing (macOS neither reports cores here nor
//! releases a thread, and pins only as a hint) — and none touches program
//! memory. Pinning is the `core_affinity` crate's; the rest is the system's
//! own calls. `Host/CpuLib.lean` states them.

/// The logical CPUs online, at least `1`.
unsafe extern "C" fn base_cpu_count() -> i32 {
    online().max(1)
}

#[cfg(unix)]
unsafe fn online() -> i32 {
    libc::sysconf(libc::_SC_NPROCESSORS_ONLN).clamp(0, i32::MAX as libc::c_long) as i32
}

#[cfg(windows)]
unsafe fn online() -> i32 {
    use windows_sys::Win32::System::Threading::{GetActiveProcessorCount, ALL_PROCESSOR_GROUPS};
    GetActiveProcessorCount(ALL_PROCESSOR_GROUPS as u16) as i32
}

/// A number the system keeps for CPU `cpu` under `topology/`, or `-1`.
#[cfg(target_os = "linux")]
fn topology(cpu: i32, file: &str) -> i32 {
    if cpu < 0 {
        return -1;
    }
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{file}"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(-1)
}

/// Which relation of `kind` — a core or a package, in the order the system
/// lists them — holds CPU `cpu` of the first processor group, or `-1`.
#[cfg(windows)]
unsafe fn relation(cpu: i32, kind: windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP) -> i32 {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    if !(0..64).contains(&cpu) {
        return -1;
    }
    let mut len = 0u32;
    GetLogicalProcessorInformationEx(kind, std::ptr::null_mut(), &mut len);
    let mut buf = vec![0u8; len as usize];
    if GetLogicalProcessorInformationEx(kind, buf.as_mut_ptr().cast(), &mut len) == 0 {
        return -1;
    }
    let (mut off, mut k) = (0usize, 0i32);
    while off < len as usize {
        let rec = &*(buf.as_ptr().add(off) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
        let p = &rec.Anonymous.Processor;
        let groups = std::slice::from_raw_parts(p.GroupMask.as_ptr(), p.GroupCount as usize);
        if groups.iter().any(|g| g.Group == 0 && g.Mask & (1usize << cpu) != 0) {
            return k;
        }
        off += rec.Size as usize;
        k += 1;
    }
    -1
}

/// The physical core CPU `cpu` is a thread of, numbered within its package;
/// `-1` for a CPU that is not there or a system that does not say.
unsafe extern "C" fn base_cpu_core(cpu: i32) -> i32 {
    #[cfg(target_os = "linux")]
    return topology(cpu, "core_id");
    #[cfg(windows)]
    return relation(cpu, windows_sys::Win32::System::SystemInformation::RelationProcessorCore);
    #[cfg(not(any(target_os = "linux", windows)))]
    return {
        let _ = cpu;
        -1
    };
}

/// The package (socket) CPU `cpu` is on; `-1` as for a core.
unsafe extern "C" fn base_cpu_package(cpu: i32) -> i32 {
    #[cfg(target_os = "linux")]
    return topology(cpu, "physical_package_id");
    #[cfg(windows)]
    return relation(cpu, windows_sys::Win32::System::SystemInformation::RelationProcessorPackage);
    #[cfg(not(any(target_os = "linux", windows)))]
    return {
        let _ = cpu;
        -1
    };
}

/// The calling thread kept on CPU `cpu` alone, through `core_affinity`:
/// `0`, or `-1`. On macOS it is a hint to the scheduler, which answers `0`
/// and may still move the thread.
unsafe extern "C" fn base_cpu_pin(cpu: i32) -> i32 {
    if cpu < 0 || cpu >= online() {
        return -1;
    }
    if core_affinity::set_for_current(core_affinity::CoreId { id: cpu as usize }) {
        0
    } else {
        -1
    }
}

/// The calling thread free to run on any CPU its process may: `0`, or `-1`.
unsafe extern "C" fn base_cpu_unpin() -> i32 {
    #[cfg(target_os = "linux")]
    {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for cpu in 0..(online().max(0) as usize).min(libc::CPU_SETSIZE as usize) {
            libc::CPU_SET(cpu, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0 {
            0
        } else {
            -1
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, GetCurrentThread, GetProcessAffinityMask, SetThreadAffinityMask,
        };
        let (mut process, mut system) = (0usize, 0usize);
        if GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) != 0
            && SetThreadAffinityMask(GetCurrentThread(), process) != 0
        {
            0
        } else {
            -1
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        -1
    }
}

/// The address of `symbol`, or `None` for one this library does not have.
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "base_cpu_count" => base_cpu_count as usize,
        "base_cpu_core" => base_cpu_core as usize,
        "base_cpu_package" => base_cpu_package as usize,
        "base_cpu_pin" => base_cpu_pin as usize,
        "base_cpu_unpin" => base_cpu_unpin as usize,
        _ => return None,
    })
}
