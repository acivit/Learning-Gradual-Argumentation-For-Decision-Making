"""
Parser for .dl argumentation-framework files.

Expected syntax (one fact per line, Prolog-style; trailing '.' optional):

    arg(name, weight).
    att(source, target).
    sup(source, target).

- `arg(x, w)` declares an argument named `x` with a base/initial weight `w`.
  The weight is optional -- `arg(x)` also works and defaults w to 0.5.
- `att(a, b)` is a directed *attack* from a to b.
- `sup(a, b)` is a directed *support* from a to b.

Blank lines and lines starting with '%' or '#' are ignored, so you can keep
comments in the .dl file.
"""

import re
from collections import deque
from dataclasses import dataclass


ARG_RE = re.compile(r"arg\(\s*([^,\s()]+)\s*(?:,\s*([\-0-9.eE]+)\s*)?\)")
ATT_RE = re.compile(r"att\(\s*([^,\s()]+)\s*,\s*([^,\s()]+)\s*\)")
SUP_RE = re.compile(r"sup\(\s*([^,\s()]+)\s*,\s*([^,\s()]+)\s*\)")


@dataclass
class DLGraph:
    arguments: dict  # name -> base weight (float)
    edges: list       # list of (source, target, kind) with kind in {"att", "sup"}

    def parents(self, name):
        return [s for s, t, _ in self.edges if t == name]

    def edge_kind(self, source, target):
        for s, t, k in self.edges:
            if s == source and t == target:
                return k
        return None


def parse_dl_file(path):
    with open(path, "r") as f:
        text = f.read()
    return parse_dl_text(text)


def parse_dl_text(text):
    arguments = {}
    edges = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue

        m = ARG_RE.search(line)
        if m:
            name = m.group(1)
            weight = float(m.group(2)) if m.group(2) is not None else 0.5
            arguments[name] = weight
            continue

        m = ATT_RE.search(line)
        if m:
            edges.append((m.group(1), m.group(2), "att"))
            continue

        m = SUP_RE.search(line)
        if m:
            edges.append((m.group(1), m.group(2), "sup"))
            continue

    # A node that only ever shows up as an edge endpoint (never declared via
    # arg(...)) still needs a base weight -- default it.
    for s, t, _ in edges:
        arguments.setdefault(s, 0.5)
        arguments.setdefault(t, 0.5)

    return DLGraph(arguments=arguments, edges=edges)


def topological_layers(graph: DLGraph):
    """
    Assigns every argument an integer 'layer' = 1 + max(layer(parent)),
    with leaves (no incoming edges) at layer 0.

    Returns (layer_of, order) where order is a valid topological ordering.
    Raises ValueError if the graph has a cycle (QBAFs used for gradual
    semantics are expected to be acyclic).
    """
    children = {n: [] for n in graph.arguments}
    indeg = {n: 0 for n in graph.arguments}
    for s, t, _ in graph.edges:
        children[s].append(t)
        indeg[t] += 1

    queue = deque([n for n, d in indeg.items() if d == 0])
    layer_of = {n: 0 for n in queue}
    order = []
    indeg_work = dict(indeg)

    while queue:
        n = queue.popleft()
        order.append(n)
        for c in children[n]:
            layer_of[c] = max(layer_of.get(c, 0), layer_of[n] + 1)
            indeg_work[c] -= 1
            if indeg_work[c] == 0:
                queue.append(c)

    if len(order) != len(graph.arguments):
        raise ValueError("The .dl graph contains a cycle; cannot topologically order it.")

    return layer_of, order
