"""
GraphMLP: a differentiable network whose neurons and connections mirror a
parsed argumentation graph (DLGraph) exactly.

Design (mirrors the semantics of your BAG / Backprop setup):
  - One scalar neuron per argument.
  - One trainable weight per att()/sup() edge -- and *only* those edges.
    This is deliberately NOT a dense layer-to-layer MLP: a real dense MLP
    would connect every neuron in layer i to every neuron in layer i+1,
    which is not "the same connections" as a sparse argumentation graph.
    What's implemented here is a general feed-forward network over the
    graph's own DAG (so it naturally handles skip connections too).
  - Leaf arguments (no incoming edges) are *inputs*: their value is
    supplied from outside, exactly like `apply_context_to_bag` pushes
    `base_scores` onto leaf `initial_weight` in the original script.
  - Every non-leaf argument computes:

        value = sigmoid( bias + sum_i  weight_i * parent_value_i )

  - Weights live in [-1, 1], matching your semantics. Rather than storing
    the weight directly and clamping it back into range after every
    optimizer step, each edge stores an *unconstrained* real parameter
    theta, and the effective weight used in the forward pass is
    `tanh(theta)`. This keeps the weight in [-1, 1] by construction:
      - No post-hoc clamp() call is needed, and there's no boundary for
        the optimizer to "overshoot into" and get stuck against.
      - The transform is smooth and differentiable everywhere (clamp()
        has zero gradient exactly at the boundary and a discontinuous
        gradient there, which is part of what causes weights to get
        stuck once they hit +-1 under repeated updates).
    Attack edges are initialized with theta corresponding to a small
    negative weight, support edges to a small positive weight -- but the
    sign is *not* permanently pinned to attack/support after that; it's
    free to move anywhere in (-1, 1) if the data supports it. If you'd
    rather keep attack edges permanently non-positive and support edges
    permanently non-negative (i.e. only the *magnitude* is learned, not
    the polarity), see `fixed_polarity=True` below.
"""

import torch
import torch.nn as nn

from utils.dl_parser import DLGraph, topological_layers


def _atanh(x):
    # torch.atanh exists in modern torch, but guard against x = +-1 exactly.
    x = max(min(x, 1 - 1e-4), -1 + 1e-4)
    return 0.5 * torch.log(torch.tensor((1 + x) / (1 - x)))


class GraphMLP(nn.Module):
    def __init__(self, graph: DLGraph, frozen_edges=None, frozen_value=0.0,
                 fixed_polarity=False, seed=None):
        """
        graph: a DLGraph from dl_parser.parse_dl_file / parse_dl_text
        frozen_edges: optional iterable of (source, target) tuples whose
            *effective* weight should be held fixed at `frozen_value` and
            excluded from training (mirrors the `frozen` edges in the
            original Backprop code; there the frozen value was 0.5 on a
            [0, 1] scale, i.e. "neutral" -- the equivalent neutral point
            on a [-1, 1] scale is 0.0, which is the default here).
        fixed_polarity: if True, attack edges are constrained to stay in
            [-1, 0] and support edges to stay in [0, 1] for the whole of
            training (only magnitude is learned, not sign). If False
            (default), each edge's weight is a free value in (-1, 1) and
            can change sign during training if the data supports it.
        seed: optional torch random seed, for reproducible initialization.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.graph = graph
        self.layer_of, self.topo_order = topological_layers(graph)
        self.leaves = [n for n in graph.arguments if self.layer_of[n] == 0]
        self.non_leaves = [n for n in self.topo_order if n not in self.leaves]
        self.fixed_polarity = fixed_polarity

        frozen_edges = set(frozen_edges or [])
        self.edge_raw = nn.ParameterDict()
        self.edge_kind = {}
        self._frozen = set()

        for s, t, kind in graph.edges:
            key = self._edge_key(s, t)
            self.edge_kind[key] = kind
            if (s, t) in frozen_edges:
                raw = _atanh(frozen_value)
                p = nn.Parameter(raw, requires_grad=False)
                self._frozen.add(key)
            else:
                sign = 1.0 if kind == "sup" else -1.0
                magnitude = torch.empty(()).uniform_(0.05, 0.4)
                init = sign * magnitude  # small, sign-appropriate start
                p = nn.Parameter(_atanh(init.item()))
            self.edge_raw[key] = p

        self.bias = nn.ParameterDict()
        for n in self.non_leaves:
            init = graph.arguments.get(n, 0.5)
            self.bias[n] = nn.Parameter(torch.tensor(float(init)))

        self.last_values = {}
        self.last_pre_activations = {}

    @staticmethod
    def _edge_key(s, t):
        # ParameterDict keys can't contain '.', so use a safe separator.
        return f"{s}__{t}"

    # def effective_weight(self, source, target):
    #     """The actual, bounded [-1, 1] weight currently used for an edge."""
    #     key = self._edge_key(source, target)
    #     raw = self.edge_raw[key]
    #     if self.fixed_polarity:
    #         mag = torch.sigmoid(raw)  # in (0, 1)
    #         sign = 1.0 if self.edge_kind[key] == "sup" else -1.0
    #         return sign * mag
    #     return torch.tanh(raw)

    def effective_weight(self, source, target):
        """Unbounded raw weight."""
        key = self._edge_key(source, target)
        raw = self.edge_raw[key]
        return raw  # No tanh, no sigmoid - completely unbounded

    def forward(self, leaf_values: dict):
        """
        leaf_values: dict {leaf_name: float or 0-d tensor}. Any leaf not
            present falls back to its base weight declared in the .dl file.

        Returns: dict {argument_name: tensor} with every node's activation,
        so you can read out e.g. values["enter"]. Pre-sigmoid totals (the
        "logits") are cached in self.last_pre_activations for use with
        BCEWithLogitsLoss -- see predict_logit().
        """
        values = {}
        pre_activations = {}
        for n in self.leaves:
            v = leaf_values.get(n, self.graph.arguments.get(n, 0.5))
            values[n] = v if torch.is_tensor(v) else torch.tensor(float(v))

        for n in self.topo_order:
            if n in values:
                continue  # leaf, already set
            total = self.bias[n]
            for p in self.graph.parents(n):
                w = self.effective_weight(p, n)
                total = total + w * values[p]
            pre_activations[n] = total
            values[n] = torch.sigmoid(total)

        self.last_values = values
        self.last_pre_activations = pre_activations
        return values

    def predict(self, leaf_values: dict, output_name="enter"):
        return self.forward(leaf_values)[output_name]

    def predict_logit(self, leaf_values: dict, output_name="enter"):
        """
        Pre-sigmoid score for the output node. Use this with
        nn.BCEWithLogitsLoss (optionally with pos_weight=...) instead of
        applying sigmoid yourself + nn.BCELoss -- it's numerically more
        stable and is what makes pos_weight-based class balancing correct.
        """
        self.forward(leaf_values)
        return self.last_pre_activations[output_name]