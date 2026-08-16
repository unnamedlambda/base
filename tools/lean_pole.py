#!/usr/bin/env python3
"""The longest dependency chain in a Lean build, with per-module times.

A build's wall clock is bounded below by its slowest chain, not by its total
work, so "which module is expensive" is the wrong question once the machine has
cores to spare. This answers "which chain is long", which is the one that moves
the wall clock.

Feed it a `lake build` log and the roots of the source tree:

    tools/lean_pole.py build.log lean/lib lean/algorithms

Modules the log does not mention — replayed from cache, so untimed — count as
zero, which makes a partial log understate a chain rather than invent one.
"""

import re
import sys
from pathlib import Path

# `✔ [12/34] Built Foo.Bar (1.2s)`, and the ⚠/ℹ the linter and cache produce.
BUILT = re.compile(r"^[^\[]*\[\d+/\d+\] Built ([\w.]+)(?::\S+)? \(([\d.]+)s\)")
IMPORT = re.compile(r"^import\s+([\w.]+)")


def times(log: Path) -> dict[str, float]:
    """Seconds per module, keeping the largest when a module is reported twice
    (a module and its `:c.o` share a name in some lake versions)."""
    out: dict[str, float] = {}
    for line in log.read_text(errors="replace").splitlines():
        m = BUILT.match(line)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), 0.0), float(m.group(2)))
    return out


def imports(roots: list[Path]) -> dict[str, list[str]]:
    """Module name to the modules it imports, read from the sources.

    A module's name is its path below the package root with separators as dots,
    which is the same convention lake reports it under.
    """
    graph: dict[str, list[str]] = {}
    for root in roots:
        for path in root.rglob("*.lean"):
            if ".lake" in path.parts or path.name == "lakefile.lean":
                continue
            rel = path.relative_to(root).with_suffix("")
            name = ".".join(rel.parts)
            deps = [m.group(1) for line in path.read_text(errors="replace").splitlines()
                    if (m := IMPORT.match(line))]
            graph[name] = deps
    return graph


def longest(graph: dict[str, list[str]], cost: dict[str, float]):
    """The heaviest chain ending at each module, memoised over the graph.

    Iterative rather than recursive: the chains here run to dozens of modules
    and the graph is read from disk, so a cycle in a malformed tree would
    otherwise blow the stack instead of being ignored.
    """
    best: dict[str, tuple[float, list[str]]] = {}
    for start in graph:
        stack = [(start, False)]
        seen = set()
        while stack:
            node, expanded = stack.pop()
            if node in best:
                continue
            if expanded:
                total, chain = 0.0, []
                for dep in graph.get(node, []):
                    if dep in best and best[dep][0] > total:
                        total, chain = best[dep]
                best[node] = (total + cost.get(node, 0.0), chain + [node])
                seen.discard(node)
                continue
            if node in seen:          # a cycle: treat it as a leaf
                best[node] = (cost.get(node, 0.0), [node])
                continue
            seen.add(node)
            stack.append((node, True))
            for dep in graph.get(node, []):
                if dep not in best and dep in graph:
                    stack.append((dep, False))
    return best


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    log, roots = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    cost = times(log)
    graph = imports(roots)
    if not cost:
        sys.exit(f"{log} reports no build times — was everything replayed from cache?")

    best = longest(graph, cost)
    total, chain = max(best.values(), key=lambda v: v[0])

    print(f"{len(cost)} modules timed, {sum(cost.values()):.0f}s of module time")
    print(f"critical path: {total:.0f}s over {len(chain)} modules\n")
    for name in chain:
        t = cost.get(name, 0.0)
        if t >= 0.5:
            print(f"  {t:7.1f}s  {name}")

    print("\nheaviest modules off the path:")
    on = set(chain)
    for name, t in sorted(cost.items(), key=lambda kv: -kv[1])[:40]:
        if name not in on and t >= 5.0:
            print(f"  {t:7.1f}s  {name}")


if __name__ == "__main__":
    main()
