"""
Backpropagation for a gradual argumentation framework using SumAggregation
and MLPBasedInfluence semantics.

Design notes
------------
- Nodes = Argument objects. Each edge (Attack/Support) has its own trainable
  weight, stored on `edge.initial_weight` (kept in sync with `edge.strength`,
  since edges never pass through the influence/aggregation functions
  themselves - they are plain scalar multipliers, matching how
  SumAggregation.aggregate_strength uses `state[edge]`).
- forward() computes, in topological order, the strength of every argument:
    leaf arguments (no incoming attacks/supports) -> strength = initial_weight
    otherwise      -> strength = MLPBasedInfluence.compute_strength(
                                      initial_weight, SumAggregation(...))
- backward() propagates dL/ds(v) from the loss backwards through the graph
  in reverse topological order, exactly like standard backprop through a
  feedforward net with sigmoid activations:
      delta(v)      = dL/ds(v) * s(v) * (1 - s(v))      # 0 at the boundary
                                                          # cases where
                                                          # compute_strength
                                                          # is constant
      dL/d(theta_att)  = -delta(target) * s(source)      for an attack edge
      dL/d(theta_sup)  = +delta(target) * s(source)      for a support edge
      dL/ds(source)   += delta(target) * (-theta_att)    via each attack
      dL/ds(source)   += delta(target) * (+theta_sup)    via each support
- Only edges (attack/support weights) are learned; node `initial_weight`s
  are left untouched, per the request ("learn the edge weights").

Limitation
----------
This implementation assumes a standard (first-order) BAG: attacks/supports
go between Arguments only. Higher-order frameworks (attacks/supports on
attacks/supports) are not handled and will raise NotImplementedError.
"""

import matplotlib.pyplot as plt
import math
import os
from collections import deque

from Uncertainpy.src.uncertainpy.gradual.Argument import Argument
from Uncertainpy.src.uncertainpy.gradual.Attack import Attack
from Uncertainpy.src.uncertainpy.gradual.Support import Support
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.SumAggregation import SumAggregation
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.MLPBasedInfluence import MLPBasedInfluence
import Uncertainpy.src.uncertainpy.gradual as grad


class Backprop:

    def __init__(self, bag, aggregation=None, influence=None):
        self.bag = bag
        self.aggregation = aggregation or SumAggregation()
        self.influence = influence or MLPBasedInfluence()

        # One snapshot per recorded step: {edge: signed_weight}, keyed by the
        # Attack/Support object itself (so it's edge.strength you get back,
        # not a recomputation from source/target). Note: if an edge flips
        # type via recompute_edge_sign(), the old edge object is removed and
        # a new one created for the same (source, target) pair, so its
        # history will appear as two separate series (see
        # plot_weight_evolution, which dedupes the legend for this case).
        self.weight_history = []
        self.base_score_history = []
        self._record_weights()  # step 0 = initial weights, before any training

    # ------------------------------------------------------------------ #
    # Graph utilities
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(self): #### not necessary, code repeated... we can use the one from the example_code.py. 
        """
        Computes and stores `strength` on every Argument in the BAG.

        Returns:
            state: dict mapping every Argument AND every Attack/Support to
                   its current scalar value (strength for arguments, weight
                   for edges) - the same shape SumAggregation expects.
            order: the topological order used (handy for backward()).
        """

        order = grad.algorithms.computeTopOrder(self.bag)
        order = [arg for arg in order if isinstance(arg, Argument)]  # filter out non-Argument nodes
        strength_values = grad.algorithms.computeStrengthValues(self.bag, self.aggregation, self.influence)

        return strength_values, order

    # ------------------------------------------------------------------ #
    # Weight history (for plotting)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _edge_label(edge):
        return (str(edge.source), str(edge.target))

    def _record_weights(self):
        """Append the current signed weight of every edge to weight_history."""
        snapshot = {}
        for edge in self.bag.attacks:
            snapshot[(str(edge.source.name), str(edge.target.name))] = -edge.initial_weight
        for edge in self.bag.supports:
            snapshot[(str(edge.source.name), str(edge.target.name))] = edge.initial_weight

        self.weight_history.append(snapshot)

    # def plot_weight_evolution(self, save_path=None, figsize=(11, 6),
    #                            edges=None, show_legend=True):
    #     """
    #     Plots the evolution of every edge's signed weight across training
    #     steps recorded so far (step 0 = initial weights, before any
    #     train_step call).

    #     Note: weight_history is keyed by the edge object itself. If an edge
    #     flips type via recompute_edge_sign(), the old object is removed and
    #     a new one created for the same (source, target) pair - so a flipped
    #     edge shows up as two line segments (a gap where the old object no
    #     longer exists / the new one doesn't exist yet), not one continuous
    #     line. Both segments share one legend entry (deduped by source ->
    #     target label) so it still reads as a single connection.

    #     Args:
    #         save_path: if given, saves the figure to this path (e.g. a .png).
    #         edges: optional filter - a single Attack/Support edge, a single
    #             Argument/name (plots that node's incoming edges), or an
    #             iterable of either. Handy for e.g.
    #             `self.frozen_edges("output")` to check the frozen layer
    #             stayed flat, or `bag.arguments["enter"]` for one node.
    #         show_legend: set False when there are many edges and the legend
    #             gets unwieldy.

    #     Returns:
    #         the matplotlib Figure.
    #     """
        

    #     if not self.weight_history:
    #         raise ValueError("No weight history recorded yet.")

    #     # Normalize `edges` filter into a set of edge objects.
    #     edge_filter = None
    #     if edges is not None:
    #         if isinstance(edges, (str, Argument, Attack, Support)):
    #             edges = [edges]
    #         edge_filter = set()
    #         for e in edges:
    #             if isinstance(e, (Attack, Support)):
    #                 edge_filter.add(e)
    #             else:  # Argument or name -> its incoming edges
    #                 arg = self.bag.arguments[e] if isinstance(e, str) else e
    #                 edge_filter.update(arg.attacks)
    #                 edge_filter.update(arg.supports)

    #     all_edges = set()
    #     for snap in self.weight_history:
    #         all_edges.update(snap.keys())
    #     if edge_filter is not None:
    #         all_edges &= edge_filter

    #     # Group edges by (source, target) label so flipped edges (new
    #     # object, same connection) share one legend entry.
    #     by_label = {}
    #     for edge in all_edges:
    #         by_label.setdefault(self._edge_label(edge), []).append(edge)

    #     fig, ax = plt.subplots(figsize=figsize)
    #     n_steps = len(self.weight_history)

    #     def first_seen(edge):
    #         for i, snap in enumerate(self.weight_history):
    #             if edge in snap:
    #                 return i
    #         return n_steps

    #     for label, edge_group in sorted(by_label.items(), key=lambda kv: str(kv[1][0])):
    #         edge_group = sorted(edge_group, key=first_seen)
    #         first = True
    #         for edge in edge_group:
    #             ys = [snap.get(edge, math.nan) for snap in self.weight_history]
    #             ax.plot(range(n_steps), ys, marker='o', markersize=2, linewidth=1,
    #                      color=None if first else ax.lines[-1].get_color(),
    #                      label=str(edge) if first else None)
    #             first = False

    #     ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    #     ax.set_xlabel("Training step")
    #     ax.set_ylabel("Signed edge weight  (+ support / - attack)")
    #     ax.set_title("Edge weight evolution")
    #     if show_legend:
    #         ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
    #     fig.tight_layout()

    #     if save_path:
    #         fig.savefig(save_path, dpi=150, bbox_inches='tight')

    #     return fig

    def plot_weight_evolution(self, save_path=None, figsize=(11, 6),
                           edges=None, show_legend=True,
                           split_labels=("robot_urgency, enter",
                                         "breaking_queue, enter",
                                         "inside_comfort, enter")):
        """
        Plots the evolution of every edge's signed weight across training
        steps recorded so far (step 0 = initial weights, before any
        train_step call).

        weight_history is now keyed by (source_name, target_name) string
        tuples, which stay stable even if an edge flips type via
        recompute_edge_sign() - so each connection is always one
        continuous line.

        Splits the output into two figures: one containing only the edges
        whose "source, target" label is in `split_labels`, and one
        containing everything else.

        Args:
            save_path: if given, saves the figures to this path with
                "_highlight" / "_rest" suffixes inserted before the
                extension (e.g. "weights.png" -> "weights_highlight.png",
                "weights_rest.png").
            edges: optional filter - a single Attack/Support edge, a single
                Argument/name (plots that node's incoming edges), or an
                iterable of either.
            show_legend: set False when there are many edges and the legend
                gets unwieldy.
            split_labels: iterable of "source, target" labels that go into
                the first ("highlight") figure. Everything else goes into
                the second ("rest") figure.

        Returns:
            a tuple (fig_highlight, fig_rest) of matplotlib Figures.
        """

        if not self.weight_history:
            raise ValueError("No weight history recorded yet.")

        # Normalize `edges` filter into a set of (source, target) label keys.
        key_filter = None
        if edges is not None:
            if isinstance(edges, (str, Argument, Attack, Support)):
                edges = [edges]
            key_filter = set()
            for e in edges:
                if isinstance(e, (Attack, Support)):
                    key_filter.add((str(e.source), str(e.target)))
                else:  # Argument or name -> its incoming edges
                    arg = self.bag.arguments[e] if isinstance(e, str) else e
                    for edge in list(arg.attacks) + list(arg.supports):
                        key_filter.add((str(edge.source), str(edge.target)))

        all_keys = set()
        for snap in self.weight_history:
            all_keys.update(snap.keys())
        if key_filter is not None:
            all_keys &= key_filter

        split_labels = set(split_labels)
        keys_highlight = {k for k in all_keys if f"{k[0]}, {k[1]}" in split_labels}
        keys_rest = all_keys - keys_highlight

        n_steps = len(self.weight_history)

        def plot_group(keys_subset, title):
            fig, ax = plt.subplots(figsize=figsize)
            for key in sorted(keys_subset):
                ys = [snap.get(key, math.nan) for snap in self.weight_history]
                ax.plot(range(n_steps), ys, marker='o', markersize=2, linewidth=1,
                        label=f"{key[0]} -> {key[1]}")

            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Training step")
            ax.set_ylabel("Signed edge weight  (+ support / - attack)")
            ax.set_title(title)
            if show_legend:
                ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5))
            fig.tight_layout()
            return fig

        fig_highlight = plot_group(keys_highlight, "Edge weight evolution (selected edges)")
        #fig_rest = plot_group(keys_rest, "Edge weight evolution (other edges)")

        if save_path:
            base, ext = os.path.splitext(save_path)
            fig_highlight.savefig(f"{base}_highlight{ext}", dpi=150, bbox_inches='tight')
            #fig_rest.savefig(f"{base}_rest{ext}", dpi=150, bbox_inches='tight')

        return fig_highlight#, fig_rest

    # ------------------------------------------------------------------ #
    # Frozen edges (intermediate layer -> output)
    # ------------------------------------------------------------------ #

    def frozen_edges(self, outputs):
        """
        Returns the set of edges that feed directly into the given output
        node(s) - i.e. the intermediate-layer -> output edges. These are
        the edges whose weights should stay fixed while the rest of the
        BAG keeps learning.

        Args:
            outputs: a single Argument/name, or an iterable of them.

        Returns:
            set of Attack/Support edges.
        """
        if isinstance(outputs, (str, Argument)):
            outputs = [outputs]

        frozen = set()
        for out in outputs:
            arg = self.bag.arguments[out] if isinstance(out, str) else out
            frozen.update(arg.attacks)
            frozen.update(arg.supports)
        return frozen

    # ------------------------------------------------------------------ #
    # Backward pass
    # ------------------------------------------------------------------ #

    def backward(self, order, loss_grad, df_quad=False):
        """
        Args:
            order: topological order returned by forward().
            loss_grad: dict {argument: dL/ds(argument)}.
            df_quad: if True, assumes forward() computed DF-QuAD strengths
                    (probabilistic-sum aggregation + combination function)
                    and backpropagates through those instead of the
                    sigmoid-MLP semantics.

        Returns:
            edge_grad: dict {Attack/Support: dL/d(edge weight)}
                    (always empty if df_quad=True: DF-QuAD has no edge weights)
            node_grad: dict {Argument: dL/d(initial_weight)}
            delta:     dict {Argument: dL/ds(argument)}
        """
        if df_quad:
            return self._backward_df_quad(order, loss_grad)

        arguments = self.bag.get_arguments()
        delta = {arg: 0.0 for arg in arguments}
        for arg, g in loss_grad.items():
            delta[arg] += g

        edge_grad = {}
        node_grad = {}

        for arg in reversed(order):
            s = arg.strength
            w0 = arg.initial_weight
            has_incoming = bool(arg.attacks) or bool(arg.supports)

            local_grad = delta[arg] * s * (1 - s)

            if has_incoming and w0 != 0 and (1 - w0) != 0:
                node_grad[arg] = local_grad / (w0 * (1 - w0))

            for att in arg.attacks:  # incoming attacks: source -> arg
                src = att.source
                theta = att.initial_weight
                edge_grad[att] = edge_grad.get(att, 0.0) - local_grad * src.strength
                delta[src] += local_grad * (-theta)

            for sup in arg.supports:  # incoming supports: source -> arg
                src = sup.source
                theta = sup.initial_weight
                edge_grad[sup] = edge_grad.get(sup, 0.0) + local_grad * src.strength
                delta[src] += local_grad * theta

        return edge_grad, node_grad, delta


    @staticmethod
    def _prod_excluding(values, i):
        """prod_{k != i} (1 - values[k]); computed directly, no division."""
        p = 1.0
        for k, v in enumerate(values):
            if k != i:
                p *= (1.0 - v)
        return p


    def _backward_df_quad(self, order, loss_grad):
        arguments = self.bag.get_arguments()
        delta = {arg: 0.0 for arg in arguments}
        for arg, g in loss_grad.items():
            delta[arg] += g

        edge_grad = {}   # DF-QuAD has no edge weights
        node_grad = {}

        for arg in reversed(order):      # parents before children, so delta[arg] is complete
            g = delta[arg]
            v0 = arg.initial_weight      # base score BS(arg)

            att_s = [att.source.strength for att in arg.attacks]
            sup_s = [sup.source.strength for sup in arg.supports]

            # F(.) = 1 - prod(1 - v); empty product = 1 -> 0
            Va = 1.0 - math.prod(1.0 - v for v in att_s)
            Vs = 1.0 - math.prod(1.0 - v for v in sup_s)

            d = Vs - Va
            beta = v0 if Va >= Vs else 1.0 - v0   # same branch convention as Va >= Vs

            # dL/dBS(arg) = dL/ds * (1 - |delta|)
            # (for leaves this is just dL/ds, since delta = 0)
            if arg.attacks or arg.supports:
                node_grad[arg] = g * (1.0 - abs(d))

            # propagate to attackers (negative sign) and supporters (positive sign)
            for i, att in enumerate(arg.attacks):
                G = self._prod_excluding(att_s, i)
                delta[att.source] -= g * beta * G

            for i, sup in enumerate(arg.supports):
                G = self._prod_excluding(sup_s, i)
                delta[sup.source] += g * beta * G

        return edge_grad, node_grad, delta

    # ------------------------------------------------------------------ #
    # Weight update
    # ------------------------------------------------------------------ #

    def step(self, edge_grad, lr=0.01, weight_bounds=(None, None), frozen_edges=None,
             node_grad=None, bias_lr=None, bias_bounds=(None, None), frozen_nodes=None):
        """
        Gradient-descent update of edge weights, applied in place to the
        Attack/Support objects stored in the BAG. Optionally also updates
        the `initial_weight` ("bias") of intermediate Argument nodes.

        This is making a clipping between 0 and 1.

        Args:
            frozen_edges: optional set of Attack/Support edges to skip -
                their initial_weight/strength are left untouched even
                though edge_grad still contains a (unused) gradient for
                them. Use `self.frozen_edges(output_name)` to build this
                for the intermediate-layer -> output edges.
            node_grad: optional dict {Argument: dL/d(initial_weight)} from
                backward(). If given, updates each argument's bias by
                gradient descent, same as edges.
            bias_lr: learning rate for node biases. Defaults to `lr`.
            bias_bounds: (lo, hi) clipping for biases, analogous to
                weight_bounds. Note: don't let a bias hit exactly 0 or 1 -
                compute_strength treats those as constant boundaries, which
                zeroes its own future gradient and can get it stuck (use
                something like (1e-6, 1 - 1e-6) instead of (0, 1)).
            frozen_nodes: optional set of Argument nodes to skip - always
                include the output node(s) here, since node_grad still
                contains an (unused) gradient for them. Leaf/input nodes
                are already absent from node_grad, so they don't need to
                be listed.
        """
        frozen_edges = frozen_edges or set()
        lo, hi = weight_bounds

        # FIX FOR WEIGHT DECAY: CHECK THE LINES
        # max_grad = 1 # decay
        for edge, grad in edge_grad.items():
            if edge in frozen_edges:
                continue

            # grad = max(-max_grad, min(max_grad, grad))  # decay: clip the gradient to avoid exploding gradients
            new_w = edge.initial_weight - lr * grad
            if lo is not None:
                new_w = max(lo, new_w)
            if hi is not None:
                new_w = min(hi, new_w)
            edge.initial_weight = new_w
            edge.strength = new_w  # edges' strength == their weight, always

        if node_grad:
            frozen_nodes = frozen_nodes or set()
            blr = bias_lr if bias_lr is not None else lr
            blo, bhi = bias_bounds
            for arg, grad in node_grad.items():
                if arg in frozen_nodes:
                    continue
                new_w0 = arg.initial_weight - blr * grad
                if blo is not None:
                    new_w0 = max(blo, new_w0)
                if bhi is not None:
                    new_w0 = min(bhi, new_w0)
                arg.initial_weight = new_w0
                # arg.strength is NOT set here (unlike edges): it depends
                # on the aggregate of incoming edges too, so it gets
                # recomputed properly on the next forward() call.

        # for edge, grad in edge_grad.items():
        #     new_w = math.tanh(edge.initial_weight - lr * grad)
        #     edge.initial_weight = new_w
        #     edge.strength = new_w  # edges' strength == their weight, always

    # ------------------------------------------------------------------ #
    # Convenience: single binary-classification output, BCE loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def bce_loss_grad(strength, target, eps=1e-12):
        """
        dL/ds for L = -(y log s + (1-y) log(1-s)).
        Dividing by s(1-s) here is intentional: backward() multiplies by
        s(1-s) again internally (the sigmoid derivative), so the two cancel
        and the well-known result delta(output) = s - y falls out naturally.
        """
        
        s = min(max(strength, eps), 1 - eps)
        return (s - target) / (s * (1 - s))

    def train_step(self, output_name, target, lr=0.01, weight_bounds=(None, None),
                   freeze_output_layer=True, learn_biases=True, bias_lr=None,
                   bias_bounds=(1e-6, 1 - 1e-6), frozen_nodes=None, normalise_weights=False, use_weight_decay=False, lambda_l2=0.001):
        """
        One full forward + backward + update step for a single-output
        binary classification setup.

        Args:
            freeze_output_layer: if True (default), the edges connecting
                the intermediate layer directly to `output_name` are kept
                fixed at their current value - only the rest of the BAG's
                edge weights are learned. Set to False to learn every edge.
            learn_biases: if True (default), also gradient-descent-updates
                the `initial_weight` ("bias") of every intermediate
                argument (i.e. every non-leaf argument except the output
                itself). Leaf/input arguments never learn a bias, since
                their strength IS their initial_weight directly (no
                compute_strength involved). Set to False to keep the old
                edges-only behaviour.
            bias_lr: learning rate for node biases. Defaults to `lr`.
            bias_bounds: clipping range for biases. Defaults to a hair
                inside (0, 1) rather than the closed interval, since
                landing exactly on 0 or 1 makes compute_strength constant
                there and zeroes the bias's own future gradient.

        Returns the BCE loss (float) computed *before* the weight update,
        so you can track a learning curve across calls.
        """
        state, order = self.forward()
        output = self.bag.arguments[output_name]
        s = output.strength

        eps = 1e-12
        s_clamped = min(max(s, eps), 1 - eps)
        loss = -(
            target * math.log(s_clamped) + (1 - target) * math.log(1 - s_clamped)
        )

        frozen = self.frozen_edges(output) if freeze_output_layer else set()

        loss_grad = {output: self.bce_loss_grad(s, target, eps)}
        edge_grad, node_grad, delta = self.backward(order, loss_grad)

        # # FIX FOR WEIGHT DECAY: add L2 regularization term to the gradients of the edges that are not frozen
        # if use_weight_decay:
        #     for edge in self.bag.attacks + self.bag.supports:
        #         if edge not in frozen:
        #             edge_grad[edge] = edge_grad.get(edge, 0.0) + 2* lambda_l2 * edge.initial_weight

            

        end_frozen_nodes = []
        for arg in frozen_nodes:
            if arg in self.bag.arguments:
                end_frozen_nodes.append(self.bag.arguments[arg])
        if not learn_biases:
            node_grad = None
        self.step(edge_grad, lr=lr, weight_bounds=weight_bounds, frozen_edges=frozen,
                  node_grad=node_grad, bias_lr=bias_lr, bias_bounds=bias_bounds,
                  frozen_nodes=end_frozen_nodes)

        
        self.recompute_edge_sign(frozen_edges=frozen)
        if normalise_weights:
            self.normalise_edge_weights(frozen_edges=frozen)
        self._record_weights()

        return loss


    # ------------------------------------------------------------------ #

    def normalise_edge_weights(self, frozen_edges=None):
        """
        Normalise the weights of the edges, so that the sum of the absolute values of the weights of the edges is equal to 1.
        Frozen edges are left untouched (they weren't updated by step(),
        so their weight can't have changed, but we skip them explicitly for
        safety/clarity).
        """
        frozen_edges = frozen_edges or set()

        # total_weight = sum(abs(edge.initial_weight) for edge in self.bag.attacks + self.bag.supports if edge not in frozen_edges)
        # if total_weight == 0:
        #     return  # Avoid division by zero

        for edge in self.bag.attacks + self.bag.supports:
            if edge not in frozen_edges:
                # edge.initial_weight /= total_weight
                # edge.strength = edge.initial_weight  # edges' strength == their weight, always
                edge.initial_weight = 1/(1+math.exp(-edge.initial_weight))
                edge.strength = edge.initial_weight  # edges' strength == their weight, always


    def recompute_edge_sign(self, frozen_edges=None):
        """
        After a weight update, recompute the sign of the edges, converting the negative attacks into positive supports, and otherwise.

        Frozen edges are left untouched (they weren't updated by step(),
        so their sign can't have flipped, but we skip them explicitly for
        safety/clarity).
        """
        frozen_edges = frozen_edges or set()

        attacks_into_supports = [
            att for att in self.bag.attacks
            if att.strength < 0 and att not in frozen_edges
        ]
        supports_into_attacks = [
            sup for sup in self.bag.supports
            if sup.strength < 0 and sup not in frozen_edges
        ]

        for att in attacks_into_supports:
            self.bag.add_support(att.source, att.target, support_weight=-att.initial_weight)
            self.bag.remove_attack(att)
            print("Converted attack to support:", att, "->", self.bag.supports[-1])

        for sup in supports_into_attacks:
            self.bag.add_attack(sup.source, sup.target, attack_weight=-sup.initial_weight)
            self.bag.remove_support(sup)
            print("Converted support to attack:", sup, "->", self.bag.attacks[-1])
        # self.forward()  # no és necessari!!

    def keep_constraints(self):
        """
        Keeping constraints, but this messes up with the gradients...
        """
        for att in self.bag.arguments["enter"].attacks:
            att.initial_weight = 0.5
            att.strength = 0.5

        for sup in self.bag.arguments["enter"].supports:
            sup.initial_weight = 0.5
            sup.strength = 0.5




    def __str__(self):
        return __class__.__name__