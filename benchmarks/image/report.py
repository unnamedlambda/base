"""The image suite's headline rows for the top-level report (../suite.py),
as JSON lines: from the newest run, frames/s of every engine on each table
(4K to 1080p by the antialiased bilinear filter, the 2x2 average and
Lanczos3) at one core, at 6 threads and on the GPU. The memory floors are
this machine's ceiling, not engines, and stay in the suite's own report
(the tables run.sh prints)."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def headline():
    runs = sorted(glob.glob(os.path.join(HERE, "results", "*", "fps.jsonl")))
    if not runs:
        return []
    out = []
    for line in open(runs[-1]):
        r = json.loads(line)
        if r["engine"].startswith("memory floor") or (r["mode"] == "threads" and r["threads"] != 6):
            continue
        mode = "6 threads" if r["mode"] == "threads" else r["mode"]
        out.append(dict(suite="image", workload=f"4K to 1080p, {r['table']}", measure=mode, unit="frames/s",
                        higher=True, engine=r["engine"], value=r["fps"], base=r["engine"].startswith("Base"),
                        run=os.path.basename(os.path.dirname(runs[-1]))))
    return out


if __name__ == "__main__":
    for row in headline():
        print(json.dumps(row))
