"""Energy over a window, from the machine's cumulative counters; shared by the
suites.

    from meter import METER
    start = METER.read()
    ...                                  # the measured work
    METER.since(start)                   # {window_s, gpu_j, cpu_pkg_j}

gpu: the board's total energy (NVML `nvmlDeviceGetTotalEnergyConsumption`,
millijoules since the driver loaded): the cumulative counter, not sampled
power, so short bursts between samples are counted. It advances every 100 ms,
so a window must be seconds long to resolve.
cpu: RAPL package energy (`/sys/class/powercap/intel-rapl:0/energy_uj`),
readable only where its permissions allow; on Zen it is the package's
modelled energy, without DRAM.
A counter that cannot be read is None, never zero.
"""

import os
import time


class Meter:
    RAPL = "/sys/class/powercap/intel-rapl:0"

    def __init__(self):
        self.nvml = self.handle = None
        try:
            import ctypes

            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            handle = ctypes.c_void_p()
            if nvml.nvmlInit_v2() == 0 and nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                self.nvml, self.handle = nvml, handle
        except OSError:
            pass
        try:
            self.cpu_wrap = int(open(os.path.join(self.RAPL, "max_energy_range_uj")).read())
            self.cpu_uj()
        except OSError:
            self.cpu_wrap = None

    def gpu_mj(self):
        import ctypes

        if self.nvml is None:
            return None
        e = ctypes.c_ulonglong()
        return e.value if self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(e)) == 0 else None

    def cpu_uj(self):
        return int(open(os.path.join(self.RAPL, "energy_uj")).read())

    def read(self):
        cpu = self.cpu_uj() if self.cpu_wrap is not None else None
        return time.perf_counter(), self.gpu_mj(), cpu

    def since(self, start) -> dict:
        """{window_s, gpu_j, cpu_pkg_j} from `start` (a `read()`) to now."""
        t1, g1, c1 = self.read()
        t0, g0, c0 = start
        cpu = None
        if c0 is not None and c1 is not None:
            cpu = ((c1 - c0) % self.cpu_wrap) / 1e6
        return dict(
            window_s=t1 - t0,
            gpu_j=(g1 - g0) / 1e3 if g0 is not None and g1 is not None else None,
            cpu_pkg_j=cpu,
        )


METER = Meter()
