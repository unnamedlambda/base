# Machine ceilings

`measure.py` measures this machine's ceilings and writes them, one result line
each, to `results/<host>.jsonl`:
- flash: sequential read, 4 KiB random read latency and IOPS from 32 threads,
  and fsync latency, for the NVMe at `/mnt/nvme` and the SATA SSD at `/`;
- DRAM copy bandwidth on one core;
- GPU device copy bandwidth and PCIe transfers (pinned memory).

Suites read these for their rooflines and their derived tables. They are
measurements, not configuration, so `ann/run.sh` re-measures them on every
invocation (about 30 s). The GPU part needs cupy: run with a suite's
environment, for example `../ann/.venv/bin/python measure.py`.
