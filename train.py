import ast
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Uncertainpy.src.uncertainpy.gradual as grad
from scipy.stats import gaussian_kde

from Backprop import Backprop
from settings import frozen_nodes, interpretable_arguments, EDGE_DISCOUNT_TARGET, EDGE_DISCOUNT_VARIANTS, NORMALISED_MODEL_NAME, ABLATION_VARIANTS, BASELINE_MODEL_NAME
from utils.compute_attributions import compute_argument_gradient
from utils.set_base_scores_from_context import set_base_scores_from_context, set_base_scores_original

import bag_test_runner
from analyse_experiments_rl import analyse
from baseline_models.rl_mab_models import train_all_profiles, make_rl_mab_decide_fns

agg_f = grad.semantics.modular.SumAggregation()
inf_f = grad.semantics.modular.MLPBasedInfluence()

inf_lin = grad.semantics.modular.LinearInfluence()

agg_prod = grad.semantics.modular.ProductAggregation()
inf_quad = grad.semantics.modular.QuadraticMaximumInfluence()
inf_eul = grad.semantics.modular.EulerBasedInfluence()


# ---------------------------------------------------------------------- #
# Post-training ablation configuration
# ---------------------------------------------------------------------- #
# Only ONE model is ever trained (the baseline, exactly as before - nothing
# pinned, no changes to the training loop). The two entries below are then
# built AFTER training by deep-copying the trained bag and pinning
# inside_comfort / robot_urgency / breaking_queue to fixed constants, so you
# can see how forcing those concepts' importance changes an already-trained
# model's behaviour - no retraining involved.
#


# ---------------------------------------------------------------------- #
# Post-training edge-discount configuration (two more models)
# ---------------------------------------------------------------------- #
# Instead of pinning an argument's own base score, these two variants
# multiply the WEIGHT OF THE EDGE (attack or support - whichever it is in
# your .dl file) from each of inside_comfort / robot_urgency / breaking_queue
# INTO "enter" by a per-source factor, post-training, on a deep copy of the
# trained bag - mirroring the argument-pinning variants above (0.75/0.25
# pairs), but discounting/boosting the edge weight instead of pinning the
# node's own base score:
#   - edge_discount_comfort_down_urgency_up: inside_comfort x0.5,
#     breaking_queue x0.5, robot_urgency x1.5
#   - edge_discount_comfort_up_urgency_down:  inside_comfort x1.5,
#     breaking_queue x1.5, robot_urgency x0.5




def prepare_data(row, dataset_name, original=False):
    """Extract context and target from a dataset row"""
    context = {
        "number_people_in_elevator": row["number_people_in_elevator"],
        "elevator_capacity": row["elevator_capacity"],
        "timeout": row["timeout"],
        "robot_time_waiting": row["robot_time_waiting"],
        "next_time_over_timeout": row["next_time_over_timeout"],
        "people_with_impairments": row["people_with_impairments"],
        "number_users_before_robot": row["number_users_before_robot"],
        "people_with_priority": row["people_with_priority"],
        "users_before_robot": row["users_before_robot"],
        "waiting_users_did_not_enter": row["waiting_users_did_not_enter"],
        "robot_profile": dataset_name,
    }

    base_scores = set_base_scores_from_context(context) if not original else set_base_scores_original(context)
    target = 1 if row["decision"] == "enter" else 0
    return context, base_scores, target


def apply_context_to_bag(bag, row, dataset_name=None, fixed_arguments=None, original=False):
    """
    Computes the base scores for a dataset row (or a simulation-time
    robot_information dict - same field names) and pushes them onto the
    corresponding leaf arguments' initial_weight in the BAG.

    fixed_arguments: optional dict of {argument_name: value}. If given,
    these arguments' initial_weight is pinned to the given constant AFTER
    the context-derived values are applied, overriding whatever the context
    (or, for arguments not otherwise touched here, the random hidden init)
    would have set - this is how the ablation variants in ABLATION_VARIANTS
    are implemented (they call this with fixed_arguments set; training never
    does).

    Returns the row's binary target (1 = "enter", 0 = otherwise). During
    simulation (bag_test_runner / make_bag_decide_fn) the caller doesn't
    need the target, so it can be ignored there.
    """
    _, base_scores, target = prepare_data(row, dataset_name, original=original)

    # input layer
    bag.arguments["next_time_over_timeout"].initial_weight = base_scores["next_time_over_timeout"]
    bag.arguments["percentage_timeout"].initial_weight = base_scores["percentage_timeout"]
    bag.arguments["important_role_in_queue"].initial_weight = base_scores["important_role_in_queue"]
    bag.arguments["amount_users_queue"].initial_weight = base_scores["amount_users_queue"]
    bag.arguments["waiting_users_did_not_enter"].initial_weight = base_scores["waiting_users_did_not_enter"]
    bag.arguments["percentage_stretchment"].initial_weight = base_scores["percentage_stretchment"]
    bag.arguments["people_impairment"].initial_weight = base_scores["people_impairment"]

    if "next_time_not_over_timeout" in bag.arguments:
        bag.arguments["next_time_not_over_timeout"].initial_weight = base_scores["next_time_not_over_timeout"]

    # intermediate layer
    bag.arguments["inside_comfort"].initial_weight = base_scores["inside_comfort"]
    bag.arguments["breaking_queue"].initial_weight = base_scores["breaking_queue"]
    bag.arguments["robot_urgency"].initial_weight = base_scores["robot_urgency"]

    if fixed_arguments:
        for arg_name, value in fixed_arguments.items():
            bag.arguments[arg_name].initial_weight = value

    return target



def set_edge_weights_to_1(bag):
    """Sets all attack/support edge weights to 1.0, post-training, on a deep
    copy of the trained bag - this is how the "Normalised Edges" variant is
    implemented (it calls this after training; training never does)."""
    bag_copy = copy.deepcopy(bag)
    for edge in list(bag_copy.attacks) + list(bag_copy.supports):
        for attr in ["initial_weight", "weight", "strength"]:
            if hasattr(edge, attr):
                setattr(edge, attr, 1.0)
    return bag_copy


def make_bag_decide_fn(bag, backprop, dataset_name=None, fixed_arguments=None, threshold=0.5,):
    """Builds a decide_fn(robot_information) -> "enter"/"no enter" for
    bag_test_runner.run_tests(), backed by a trained bag. Forward-pass only
    (no learning happens during simulation)."""
    def decide(robot_information):
        # apply_context_to_bag needs a "decision" key (used only to compute
        # the training target, via prepare_data); simulation steps don't
        # have a ground-truth label, so a dummy value is supplied and the
        # returned target is simply discarded here.
        robot_information.setdefault("decision", "no enter")
        apply_context_to_bag(bag, robot_information, dataset_name=dataset_name, fixed_arguments=fixed_arguments)
        backprop.forward()
        score = bag.arguments["enter"].strength
        return "enter" if score >= threshold else "no enter"
    return decide


def make_bag_decide_original_fn(bag, backprop, dataset_name, fixed_arguments=None, threshold=0.5,):
    """Builds a decide_fn(robot_information) -> "enter"/"no enter" for
    bag_test_runner.run_tests(), backed by a trained bag. Forward-pass only
    (no learning happens during simulation)."""
    def decide(robot_information):
        # apply_context_to_bag needs a "decision" key (used only to compute
        # the training target, via prepare_data); simulation steps don't
        # have a ground-truth label, so a dummy value is supplied and the
        # returned target is simply discarded here.
        robot_information.setdefault("decision", "no enter")
        apply_context_to_bag(bag, robot_information, dataset_name=dataset_name, fixed_arguments=fixed_arguments, original=True)
        set_edge_weights_to_1(bag)
        backprop.forward()
        score = bag.arguments["enter"].strength
        return "enter" if score >= threshold else "no enter"
    return decide


def print_class_balance(df, label_col="decision", title="Class balance"):
    counts = df[label_col].value_counts()
    proportions = df[label_col].value_counts(normalize=True)

    print(f"{title} (n={len(df)}):")
    for label in counts.index:
        print(f"  {label}: {counts[label]} rows ({proportions[label] * 100:.2f}%)")

    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    print(f"  Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("  -> Warning: this split is imbalanced. Accuracy alone can be misleading here; "
              "keep an eye on precision/recall/F1, and on the confusion matrix.")


def stratified_train_test_split(dataset, label_col="decision", runs_col="run", test_size=0.2, runs_size=0.1, seed=42):
    """
    First splits the dataset into training and evaluation and post-evaluation (runs),
    the runs are used to evaluate the model after training and test, to evaluate different metrics of the model in complete runs of the scenario.
    Splits each class independently and recombines, so the train/test split
    preserves (approximately) the same class ratio as the full dataset.
    """
    train_parts = []
    test_parts = []

    number_runs = dataset[runs_col].max()

    runs_selected = random.sample(range(number_runs + 1), int(runs_size * number_runs))

    run_parts = dataset[dataset[runs_col].isin(runs_selected)]
    trainable_parts = dataset[~dataset[runs_col].isin(runs_selected)]

    for _, group in trainable_parts.groupby(label_col):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_test = int(round(len(shuffled) * test_size))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df, run_parts


def balanced_train_test_split(
    dataset,
    label_col="decision",
    runs_col="run",
    test_size=0.2,
    runs_size=0.1,
    seed=42,
    samples_per_class=None,
):
    """
    Like stratified_train_test_split, but balances the classes by
    undersampling each class to the same size before splitting, so the
    resulting train/test sets have (approximately) equal representation
    per class instead of preserving the original imbalance.

    Parameters
    ----------
    samples_per_class : int, optional
        Number of rows to keep per class. If None, defaults to the size
        of the smallest class (i.e. full undersampling to the minority
        class size). If set higher than a class's available rows, that
        class is capped at its available rows (with a warning).

    Returns
    -------
    train_df, test_df, run_parts
    """
    rng = random.Random(seed)

    number_runs = dataset[runs_col].max()
    runs_selected = rng.sample(range(number_runs + 1), int(runs_size * number_runs))

    run_parts = dataset[dataset[runs_col].isin(runs_selected)]
    trainable_parts = dataset[~dataset[runs_col].isin(runs_selected)]

    # --- Step 1: determine target size per class ---
    class_counts = trainable_parts.groupby(label_col).size()
    target_n = samples_per_class if samples_per_class is not None else class_counts.min()

    # --- Step 2: undersample each class to target_n ---
    balanced_parts = []
    for label, group in trainable_parts.groupby(label_col):
        n = min(target_n, len(group))
        if n < target_n:
            print(f"Warning: class '{label}' has only {len(group)} rows, "
                  f"less than requested {target_n}. Using {n} instead.")
        balanced_parts.append(group.sample(n=n, random_state=seed))

    balanced_df = pd.concat(balanced_parts)

    # --- Step 3: stratified split on the now-balanced data ---
    train_parts = []
    test_parts = []

    for _, group in balanced_df.groupby(label_col):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_test = int(round(len(shuffled) * test_size))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df, run_parts


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------- #
# Training (baseline only - this is unchanged from before, nothing pinned,
# runs exactly once).
# ---------------------------------------------------------------------- #
def train_baseline(
    case_file,
    train_df,
    test_df,
    model_name=BASELINE_MODEL_NAME,
    dataset_name="complacent",
    seed=42,
    n_epochs=10,
    weight_bounds=(None, None), # if set to 0, 1, it does not learn
    base_score_bounds=(1e-6, 1 - 1e-6), # sadly imposed to satisfy semantics
    freeze_output_layer=True,
    top_weight=0.1, # low if normalising during training to allow gradients, but does not work much
    random_hidden_weights=True,
    learn_base_scores=True,
    base_scores_lr=0.01,
    edges_lr=0.01,
    plot_weights=True,
    normalise_weights=False, # Does not work, we normalise after training better
    use_weight_decay=False, # L2 regularisation
    results_dir="results",
):
    """Trains one bag from scratch (no fixed/pinned arguments) and returns
    (bag, backprop). Ablated variants are built afterward, post-hoc, by
    copy_and_pin_arguments() below - this function never sees them."""
    print("#" * 70)
    print(f"# Training: {model_name}")
    print("#" * 70)

    random.seed(seed)
    np.random.seed(seed)

    bag = grad.BAG(case_file)
    backprop = Backprop(bag, agg_f, inf_f)
    frozen = backprop.frozen_edges("enter")

    # if ((dataset_name == "complacent" or dataset_name == "comfort-blind") and "hidden_5.dl" in model_name) or (dataset_name == "complacent" and "hidden_5.dl" in model_name):
    #     n_epochs = 20

    for attack in bag.attacks:
        if attack in frozen:
            attack.initial_weight = 1
            attack.strength = 1
            continue
        else:
            top_init = top_weight if weight_bounds[1] is None else weight_bounds[1]
            weight = random.uniform(0.0, top_init)
            attack.initial_weight = weight
            attack.strength = weight

    for support in bag.supports:
        if support in frozen:
            support.initial_weight = 1
            support.strength = 1
            continue
        else:
            top_init = top_weight if weight_bounds[0] is None else weight_bounds[0]
            weight = random.uniform(0.0, top_init)
            support.initial_weight = weight
            support.strength = weight

    if random_hidden_weights:
        for argument in bag.arguments:
            if argument not in interpretable_arguments:
                weight = random.uniform(0.0, 1.0)
                bag.arguments[argument].initial_weight = weight
                bag.arguments[argument].initial_strenght = weight

    print("=" * 60)
    print_class_balance(train_df, title=f"[{model_name}] Train set class balance")
    print("-" * 60)
    print_class_balance(test_df, title=f"[{model_name}] Test set class balance")
    print("=" * 60)

    epoch_losses = []
    for epoch in range(1, n_epochs + 1):
        cummulative_loss = []
        for index, row in train_df.iterrows():
            target = apply_context_to_bag(bag, row, dataset_name)  # no fixed_arguments during training

            loss = backprop.train_step(
                output_name="enter", target=target, lr=edges_lr, weight_bounds=weight_bounds,
                freeze_output_layer=freeze_output_layer, learn_biases=learn_base_scores,
                bias_lr=base_scores_lr, bias_bounds=base_score_bounds, frozen_nodes=frozen_nodes,
                normalise_weights=normalise_weights, use_weight_decay=use_weight_decay
            )
            cummulative_loss.append(loss)
            bag = backprop.bag

        avg_loss = sum(cummulative_loss) / len(cummulative_loss)
        print(f"[{model_name}] Epoch {epoch:3d}/{n_epochs}  avg loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Training Loss over Epochs - {model_name}')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_training_loss.png'))
    plt.close()

    if plot_weights:
        backprop.plot_weight_evolution(save_path=os.path.join(results_dir, f"{model_name}_weight_evolution.png"))

    return bag, backprop


def copy_and_pin_arguments(bag, fixed_arguments):
    """Builds an ablated COPY of an already-trained bag: deep-copies it (so
    the original trained model is untouched) and pins fixed_arguments'
    initial_weight to constants, post-training - no learning happens here.
    Returns (bag_copy, backprop_copy)."""
    bag_copy = copy.deepcopy(bag)
    backprop_copy = Backprop(bag_copy, agg_f, inf_f)
    for arg_name, value in fixed_arguments.items():
        bag_copy.arguments[arg_name].initial_weight = value
    return bag_copy, backprop_copy


# -- Edge (attack/support) endpoint + weight introspection -------------- #
# See the CAVEAT comment above EDGE_DISCOUNT_TARGET: these attribute names
# are best-effort guesses, not confirmed against your actual library.
_EDGE_ENDPOINT_ATTR_CANDIDATES = [
    ("attacker", "attacked"),
    ("supporter", "supported"),
    ("source", "target"),
    ("from_argument", "to_argument"),
    ("premise", "conclusion"),
]
_EDGE_WEIGHT_ATTRS = ["initial_weight", "weight", "strength"]


def _edge_endpoints(edge):
    """Best-effort extraction of (source_name, target_name) for an
    attack/support edge object. Tries several attribute-naming conventions
    in _EDGE_ENDPOINT_ATTR_CANDIDATES; raises AttributeError listing the
    object's real attributes if none match, so you can add the correct pair
    there instead of silently guessing wrong."""
    for src_attr, tgt_attr in _EDGE_ENDPOINT_ATTR_CANDIDATES:
        if hasattr(edge, src_attr) and hasattr(edge, tgt_attr):
            src = getattr(edge, src_attr)
            tgt = getattr(edge, tgt_attr)
            src_name = getattr(src, "name", src)
            tgt_name = getattr(tgt, "name", tgt)
            return src_name, tgt_name
    raise AttributeError(
        "Could not find source/target endpoint attributes on this attack/support edge "
        f"object using any of {_EDGE_ENDPOINT_ATTR_CANDIDATES}. "
        f"Available attributes on this object: {[a for a in dir(edge) if not a.startswith('_')]}. "
        "Add the correct attribute-name pair to _EDGE_ENDPOINT_ATTR_CANDIDATES in claude_train.py."
    )


def _discount_edge(edge, factor):
    """Multiplies whichever of initial_weight/weight/strength exist on this
    edge object by `factor`. Multiple attributes are discounted (if
    present) because it isn't confirmed which one forward() actually reads
    for the live edge weight - see the CAVEAT comment above."""
    touched = []
    for attr in _EDGE_WEIGHT_ATTRS:
        if hasattr(edge, attr):
            setattr(edge, attr, getattr(edge, attr) * factor)
            touched.append(attr)
    return touched


def find_edges_into(bag, target_name, source_names):
    """Finds every attack/support edge in `bag` whose target is target_name
    and whose source is in source_names."""
    source_names = set(source_names)
    matches = []
    for edge in list(bag.attacks) + list(bag.supports):
        src_name, tgt_name = _edge_endpoints(edge)
        if tgt_name == target_name and src_name in source_names:
            matches.append(edge)
    return matches


def copy_and_discount_edges(bag, source_factors, target_name=EDGE_DISCOUNT_TARGET):
    """Builds an edge-discounted COPY of an already-trained bag: deep-copies
    it and multiplies the weight of each edge source -> target_name by that
    source's own factor from source_factors (a {source_name: factor} dict,
    so different sources can get different discounts), post-training - no
    learning happens here, same "no retraining" approach as
    copy_and_pin_arguments() but on edges instead of argument base scores.
    Raises ValueError if any requested source has no direct edge into
    target_name (it might connect through an intermediate node instead, in
    which case discounting it directly isn't meaningful)."""
    bag_copy = copy.deepcopy(bag)
    backprop_copy = Backprop(bag_copy, agg_f, inf_f)
    edges = find_edges_into(bag_copy, target_name, source_factors.keys())

    found_sources = set()
    for edge in edges:
        src_name, _ = _edge_endpoints(edge)
        factor = source_factors[src_name]
        found_sources.add(src_name)
        touched_attrs = _discount_edge(edge, factor)
        print(f"  discounted edge {src_name} -> {target_name} by x{factor} "
              f"(attributes updated: {touched_attrs})")

    missing = set(source_factors.keys()) - found_sources
    if missing:
        raise ValueError(
            f"copy_and_discount_edges: no direct {target_name!r}-edge found for: {sorted(missing)}. "
            "Check that these arguments actually attack/support "
            f"{target_name!r} directly in your .dl file (they might connect through an "
            "intermediate node instead, in which case this function isn't the right tool)."
        )
    return bag_copy, backprop_copy

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _normalise_edge(edge, max_weight):
    """Applies the sigmoid to whichever of initial_weight/weight/strength
    exist on this edge object, squashing it into (0, 1). Same
    multiple-attribute approach as _discount_edge(), for the same reason:
    it isn't confirmed which attribute forward() actually reads."""
    touched = []
    for attr in _EDGE_WEIGHT_ATTRS:
        if hasattr(edge, attr):
            #setattr(edge, attr, _sigmoid(getattr(edge, attr)))
            setattr(edge,attr, getattr(edge, attr) / max_weight) # normalise to (0, 1) by dividing by the max weight
            touched.append(attr)
    return touched

def copy_and_normalise_edges(bag):
    """Builds a normalised COPY of an already-trained bag: deep-copies it
    and applies a sigmoid to EVERY attack/support edge's weight (squashing
    each into (0, 1)), post-training - no learning happens here, same "no
    retraining" approach as the other post-hoc variants above."""
    bag_copy = copy.deepcopy(bag)
    backprop_copy = Backprop(bag_copy, agg_f, inf_f)
    all_edges = list(bag_copy.attacks) + list(bag_copy.supports)
    max_weight = max(getattr(edge, "initial_weight", 0) for edge in all_edges)
    for edge in list(bag_copy.attacks) + list(bag_copy.supports):
        _normalise_edge(edge, max_weight)
    return bag_copy, backprop_copy

def plot_attribution_distributions(attributions_df, model_name, results_dir):
    """Plots and saves the distribution (histogram + KDE) of every
    argument's attribution across the whole test set, as one grid of
    subplots (one subplot per argument), for a single model.

    attributions_df: wide-format DataFrame as produced inside
    evaluate_on_test_set() - one row per test example, one column per
    argument (plus an "index" column, which is ignored here).

    Saves to results_dir/<model_name>_attributions_distribution.png and
    returns that path.
    """
    argument_cols = [c for c in attributions_df.columns if c != "index"]
    n_args = len(argument_cols)
    if n_args == 0:
        return None

    n_cols = min(4, n_args)
    n_rows = int(np.ceil(n_args / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i, argument in enumerate(argument_cols):
        ax = axes[i // n_cols][i % n_cols]
        #values = attributions_df[argument].to_numpy(dtype=float)
        raw = attributions_df[argument].to_numpy()
        strengths = np.array([t[0] for t in raw], dtype=float)
        attributions = np.array([t[1] for t in raw], dtype=float)
        values = attributions
        ax.hist(values, bins=20, density=True, alpha=0.5, color="tab:blue")
        if values.max() > values.min():
            x = np.linspace(values.min(), values.max(), 500)
            try:
                kde = gaussian_kde(values)
                ax.plot(x, kde(x), color="tab:blue")
            except Exception:
                pass  # e.g. near-constant values -> singular KDE covariance
        ax.set_title(argument, fontsize=9)
        ax.set_xlabel("Attribution")
        ax.set_ylabel("Density")

    # Hide any unused subplot cells (grid may not divide evenly).
    for j in range(n_args, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Argument attribution distributions - {model_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"{model_name}_attributions_distribution.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path

def plot_attribution_distributions_by_strength(
    attributions_df,
    model_name,
    results_dir,
    n_strength_bins=4,
    max_discrete_strengths=6,
    min_points_for_kde=5,
):
    """For every argument, plots the distribution (histogram outline + KDE)
    of its attribution conditioned on the argument's strength, as one grid
    of subplots (one per argument), for a single model.

    Each cell of attributions_df is a (strength, attribution) tuple.
    - If an argument has <= max_discrete_strengths unique strengths, one
      curve is drawn per unique strength.
    - Otherwise, strengths are split into n_strength_bins quantile bins
      and one curve is drawn per bin.

    Saves to results_dir/<model_name>_attributions_by_strength.png and
    returns that path.
    """
    argument_cols = [c for c in attributions_df.columns if c != "index"]
    n_args = len(argument_cols)
    if n_args == 0:
        return None

    n_cols = min(4, n_args)
    n_rows = int(np.ceil(n_args / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
    )

    for i, argument in enumerate(argument_cols):
        ax = axes[i // n_cols][i % n_cols]

        raw = attributions_df[argument].to_numpy()
        strengths = np.array([t[0] for t in raw], dtype=float)
        attributions = np.array([t[1] for t in raw], dtype=float)

        df = pd.DataFrame({"strength": strengths, "attribution": attributions})

        # Group by strength: discrete values, or quantile bins if continuous.
        if df["strength"].nunique() <= max_discrete_strengths:
            df["group"] = df["strength"].map(lambda s: f"{s:.2f}")
            order = sorted(df["group"].unique(), key=float)
        else:
            binned = pd.qcut(df["strength"], q=n_strength_bins, duplicates="drop")
            df["group"] = binned.astype(str)
            order = [str(c) for c in binned.cat.categories]

        # Common x-range and bins so groups are directly comparable.
        lo, hi = df["attribution"].min(), df["attribution"].max()
        bins = np.linspace(lo, hi, 21) if hi > lo else 20
        x = np.linspace(lo, hi, 500) if hi > lo else None

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
        for color, group in zip(colors, order):
            vals = df.loc[df["group"] == group, "attribution"].to_numpy()
            if len(vals) == 0:
                continue
            label = f"{group} (n={len(vals)})"
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    alpha=0.6, color=color)
            if x is not None and len(vals) >= min_points_for_kde and vals.max() > vals.min():
                try:
                    ax.plot(x, gaussian_kde(vals)(x), color=color, label=label)
                    continue
                except Exception:
                    pass  # near-constant values -> singular KDE covariance
            # Fallback: no KDE, still register the legend entry
            ax.plot([], [], color=color, label=label)

        ax.set_title(argument, fontsize=9)
        ax.set_xlabel("Attribution")
        ax.set_ylabel("Density")
        ax.legend(title="Strength", fontsize=7, title_fontsize=8)

    for j in range(n_args, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Argument attribution distributions by strength - {model_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"{model_name}_attributions_by_strength.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------- #
# Evaluation on the held-out test set - used for the trained baseline AND
# for each post-hoc ablated variant, so you get comparable metrics/plots
# for all of them.
# ---------------------------------------------------------------------- #
def evaluate_on_test_set(bag, backprop, test_df, model_name, dataset_name, fixed_arguments=None, results_dir="results"):
    print("=" * 60)
    print(f"[{model_name}] Evaluating on held-out test set...")

    tp = fp = tn = fn = 0
    test_results = []
    scores_plot = []
    attribution_rows = []  # one dict per test row: {"index": ..., arg1: ..., arg2: ..., ...}
    argument_names = [a for a in bag.arguments if a != "enter"]
    for index, row in test_df.iterrows():
        target = apply_context_to_bag(bag, row, dataset_name, fixed_arguments=fixed_arguments)
        backprop.forward()
        score = bag.arguments["enter"].strength
        scores_plot.append(score)
        pred = 1 if score >= 0.5 else 0

        dict_for_test_results = {"index": index, "score": score, "pred": pred, "true": target}
        for argument in argument_names:
            if argument not in frozen_nodes:
                dict_for_test_results[argument] = bag.arguments[argument].strength
        test_results.append(dict_for_test_results)

        # Argument attributions for THIS test row - the whole point of doing
        # this here (instead of once, after the loop, on whatever state the
        # bag happened to be left in) is that every row has its own
        # context-dependent base scores, so its attributions differ too.
        row_attributions = {"index": index}
        for argument in argument_names:
            row_attributions[argument] = (bag.arguments[argument].strength, compute_argument_gradient(bag, argument, "enter", agg_f, inf_f, h=1e-5))
        attribution_rows.append(row_attributions)

        if pred == 1 and target == 1:
            tp += 1
        elif pred == 1 and target == 0:
            fp += 1
        elif pred == 0 and target == 0:
            tn += 1
        else:
            fn += 1

    scores_plot = np.asarray(scores_plot)
    plt.figure()
    plt.hist(scores_plot, bins=20, density=True, alpha=0.5)
    if scores_plot.max() > scores_plot.min():
        x = np.linspace(scores_plot.min(), scores_plot.max(), 500)
        kde = gaussian_kde(scores_plot)
        plt.plot(x, kde(x))
    plt.xlabel("Enter Strength")
    plt.ylabel("Distribution")
    plt.title(f"Score distribution - {model_name}")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_distribution.png'))
    plt.close()

    metrics = compute_metrics(tp, fp, tn, fn)
    print(f"[{model_name}] Test set results ('enter' = positive class):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 score:  {metrics['f1']:.4f}")
    print(f"  Confusion matrix -> TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(os.path.join(results_dir, f"{model_name}_test_predictions.csv"), index=False)

    # Attributions for every argument, for every row in the test set (wide
    # format: one row per test example, one column per argument).
    attributions_df = pd.DataFrame(attribution_rows)
    attributions_path = os.path.join(results_dir, f"{model_name}_attributions.csv")
    attributions_df.to_csv(attributions_path, index=False)
    plot_attribution_distributions(attributions_df, model_name, results_dir)
    plot_attribution_distributions_by_strength(attributions_df, model_name, results_dir,)

    # Mean attribution per argument across the whole test set (used for the
    # summary .txt below - more representative than a single-context
    # snapshot, since it's averaged over every row's own context).
    mean_attributions = (
        attributions_df[argument_names]
        .apply(lambda col: col.apply(lambda x: x[1]))
        .mean()
        .to_dict()
    )#attributions_df[argument_names].mean().to_dict()

    file_path = os.path.join(results_dir, f"{model_name}.txt")
    with open(file_path, "w") as file:
        file.write(f"Test set results for {model_name} ('enter' = positive class):\n")
        if fixed_arguments:
            file.write(f"  Pinned (post-training) arguments: {fixed_arguments}\n")
        file.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        file.write(f"  Precision: {metrics['precision']:.4f}\n")
        file.write(f"  Recall:    {metrics['recall']:.4f}\n")
        file.write(f"  F1 score:  {metrics['f1']:.4f}\n")
        file.write(f"  Confusion matrix -> TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")

        for item in bag.arguments:
            file.write(str(bag.arguments[item]) + "\n")
        for item in bag.attacks:
            file.write(str(item) + "\n")
        for item in bag.supports:
            file.write(str(item) + "\n")

        file.write("Attributions (averaged over the whole test set - see "
                   f"{os.path.basename(attributions_path)} for the per-example values)\n")
        for argument in mean_attributions.keys():
            file.write(argument + "- base score: " + str(round(bag.arguments[argument].initial_weight, 3))
                       + ", mean attribution: " + str(round(mean_attributions[argument], 3)) + "\n")

    return metrics


# ---------------------------------------------------------------------- #
# Full pipeline for ONE (dataset, .dl file) combination.
#
# Everything that used to live directly under `if __name__ == "__main__":`
# is now wrapped in this function so it can be re-run for every dataset x
# .dl-file combination, each writing into its own results sub-folder:
#
#   results/
#     <dataset_name>/
#       <dl_file_name>/
#         <BASELINE_MODEL_NAME>_training_loss.png
#         <model_name>_distribution.png
#         <model_name>_test_predictions.csv
#         <model_name>.txt
#         combined_decision_log.csv
#         test_outcomes.csv
#         robot_metrics_summary.csv
#         plots/
# ---------------------------------------------------------------------- #
def run_pipeline(dataset_name, dataset_path, dl_name, case_file, results_root="results", n_epochs=10, n_test_runs=500,
                 rl_mab=None):
    """rl_mab: optional (models, weights) tuple for THIS dataset/profile, as
    returned (per profile) by rl_mab_models.train_all_profiles(). If given,
    every RL/MAB/MORL model is simulated alongside the bag models and ends up
    in the combined log, the metrics summary and the radar charts. If None,
    the pipeline behaves exactly as before (bag models only)."""
    results_dir = os.path.join(results_root, dataset_name, dl_name)
    os.makedirs(results_dir, exist_ok=True)

    print("#" * 70)
    print(f"# DATASET: {dataset_name}  |  .dl FILE: {dl_name}")
    print(f"# case_file: {case_file}")
    print(f"# dataset_path: {dataset_path}")
    print(f"# results_dir: {results_dir}")
    print("#" * 70)

    dataset = pd.read_csv(dataset_path)

    # Full-dataset balance, before any split
    print("=" * 60)
    print_class_balance(dataset, title="Full dataset class balance")
    print("=" * 60)

    columns_to_convert = ['users_before_robot']
    for col in columns_to_convert:
        dataset[col] = dataset[col].apply(ast.literal_eval)

    # Split ONCE and reuse for every variant, so the only difference between
    # models is fixed_arguments, not the data they trained/tested on.
    # train_df, test_df, exploration_runs = stratified_train_test_split(
    #     dataset, label_col="decision", runs_col="run", test_size=0.2, runs_size=0.1, seed=42
    # )

    train_df, test_df, exploration_runs = balanced_train_test_split(
        dataset, label_col="decision", runs_col="run", test_size=0.2, runs_size=0.1, seed=42
    )

    
    # ------------------------------------------------------------------ #
    # Step 1: train ONE model (the baseline), exactly as before, nothing
    # pinned. Then build the ablated variants by copying the TRAINED bag
    # and pinning inside_comfort/robot_urgency/breaking_queue afterward -
    # no retraining.
    # ------------------------------------------------------------------ #
    bag, backprop = train_baseline(
        case_file, train_df, test_df, model_name=dataset_name + BASELINE_MODEL_NAME, dataset_name=dataset_name, n_epochs=n_epochs, results_dir=results_dir,
    )

    trained_variants = {dataset_name + BASELINE_MODEL_NAME: (bag, backprop, None)}
    for variant in ABLATION_VARIANTS:
        bag_copy, backprop_copy = copy_and_pin_arguments(bag, variant["fixed_arguments"])
        trained_variants[variant["model_name"]] = (bag_copy, backprop_copy, variant["fixed_arguments"])

    print("=" * 60)
    print("Building edge-discount variants (post-training, no pinned arguments)")
    print("=" * 60)
    for variant in EDGE_DISCOUNT_VARIANTS:
        print(f"[{variant['model_name']}]")
        bag_copy, backprop_copy = copy_and_discount_edges(
            bag, variant["source_factors"], target_name=EDGE_DISCOUNT_TARGET,
        )
        # fixed_arguments=None: nothing about the argument base scores is
        # pinned for these variants, only the edge weights were changed.
        trained_variants[variant["model_name"]] = (bag_copy, backprop_copy, None)

    print("=" * 60)
    print("Building normalised-edges variant (sigmoid on every edge weight, post-training)")
    print("=" * 60)

    # bag_copy, backprop_copy = copy_and_normalise_edges(bag)
    # trained_variants[NORMALISED_MODEL_NAME] = (bag_copy, backprop_copy, None)

    # Evaluate all of them on the (shared) held-out test set.
    for model_name, (bag_v, backprop_v, fixed_arguments) in trained_variants.items():
        if "trained" in model_name:
            evaluate_on_test_set(
                bag_v, backprop_v, test_df, model_name, dataset_name=dataset_name, fixed_arguments=fixed_arguments, results_dir=results_dir,
            )

    # ------------------------------------------------------------------ #
    # Step 2: simulate every trained variant with bag_test_runner and
    # combine into one long-format log, exactly like the old project's
    # run_full_pipeline.py did for its RL/AF models.
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print(f"STEP 2 - simulating {len(trained_variants)} models x {n_test_runs} runs each")
    print("=" * 70)

    # decide_fns = {
    #     model_name: make_bag_decide_fn(bag_v, backprop_v, fixed_arguments=fixed_arguments)
    #     for model_name, (bag_v, backprop_v, fixed_arguments) in trained_variants.items()
    # }

    decide_fns = {}

    for model_name, (bag_v, backprop_v, fixed_arguments) in trained_variants.items():
        decide_fns[model_name] = make_bag_decide_fn(bag_v, backprop_v, dataset_name, fixed_arguments=fixed_arguments)
        if model_name == dataset_name + BASELINE_MODEL_NAME:
            bag_original = grad.BAG(case_file_original)
            decide_fns["baseline_QE"] = make_bag_decide_original_fn(bag_original, Backprop(bag_original, agg_f, inf_quad), dataset_name, fixed_arguments=fixed_arguments)
            # decide_fns["baseline_DF"] = make_bag_decide_original_fn(bag_original, Backprop(bag_original, agg_prod, inf_lin), dataset_name, fixed_arguments=fixed_arguments)
            # decide_fns["baseline_EB"] = make_bag_decide_original_fn(bag_original, Backprop(bag_original, agg_f, inf_eul), dataset_name, fixed_arguments=fixed_arguments)


    # RL / MAB / MORL models (trained once per profile in __main__): same
    # decide_fn interface as the bag models, so they go through exactly the
    # same simulation and end up in the same combined log.
    if rl_mab is not None:
        rl_models, rl_weights = rl_mab
        rl_decide_fns = make_rl_mab_decide_fns(rl_models, rl_weights)
        decide_fns.update(rl_decide_fns)
        print(f"  Added {len(rl_decide_fns)} RL/MAB models to the simulation: {list(rl_decide_fns)}")
    else:
        print("  No RL/MAB models for this profile - simulating bag models only.")

    outcomes_df, log_df = bag_test_runner.run_tests_all_models(
        decide_fns, n_test_runs=n_test_runs, robot_profile=dataset_name,
    )
    for model_name in decide_fns:
        entered_pct = outcomes_df.loc[outcomes_df["model"] == model_name, "entered"].mean() * 100
        print(f"  [{model_name}] entered {entered_pct:.1f}% of {n_test_runs} shared runs")

    log_path = os.path.join(results_dir, "combined_decision_log.csv")
    outcomes_path = os.path.join(results_dir, "test_outcomes.csv")
    log_df.to_csv(log_path, index=False)
    outcomes_df.to_csv(outcomes_path, index=False)
    print(f"Saved {len(log_df)} rows -> {log_path}")
    print(f"Saved {len(outcomes_df)} rows -> {outcomes_path}")

    # ------------------------------------------------------------------ #
    # Step 3: full metrics + plots, including the radar charts, via the
    # (unmodified) analyse_experiments_rl.analyse(). It already supports an
    # arbitrary number of models per profile, so the variants above are
    # compared exactly like the old project compared its RL/AF models. The
    # RL/MAB models are in the same log, so they appear in the radar charts
    # too (see RL_MAB_MODEL_NAMES in analyse_experiments_rl.py).
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 3 - analysis + plots (including radar charts)")
    print("=" * 70)
    summary_path = os.path.join(results_dir, "robot_metrics_summary.csv")
    plots_dir = os.path.join(results_dir, "plots")
    analyse(log_path, summary_path, plots_dir=plots_dir)

    return {
        "results_dir": results_dir,
        "summary_path": summary_path,
        "log_path": log_path,
        "outcomes_path": outcomes_path,
    }


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    # Fill these in:
    #   - DATASETS: one entry per dataset you want to train on. `name` is
    #     only used to name the results sub-folder, `path` is the CSV file.
    #   - DL_FILES: one entry per .dl file. `name` is used to name the
    #     results sub-folder, `path` is the .dl file itself.
    #
    # Every dataset is trained against every .dl file, i.e. with 4 datasets
    # and 9 .dl files you get 36 full pipeline runs, each writing into its
    # own results/<dataset_name>/<dl_name>/ folder.
    # ------------------------------------------------------------------ #

    case_file_original = "envs/elevator_scenario_old.dl"  # used for the original baseline variants (QE, DF, EB)
    DATASETS = [
            {"name": "complacent", "path": "data/argumentation_QE_complacent.csv"},
            {"name": "comfort-blind", "path": "data/argumentation_QE_cb.csv"},
            {"name": "comfort-caring", "path": "data/argumentation_QE_cc.csv"},
            {"name": "time-efficient", "path": "data/argumentation_QE_te.csv"},
        ]

    dl_files_path = "envs/"
    DL_FILES = [
        {"name": "normal", "path": "envs/elevator_scenario.dl"},
        {"name": "hidden_5", "path": dl_files_path + "hidden_5.dl"},
        {"name": "hidden_10", "path": dl_files_path + "hidden_10.dl"},
        # {"name": "hidden_20", "path": dl_files_path + "hidden_20.dl"},
        # {"name": "hidden_50", "path": dl_files_path + "hidden_50.dl"},
        {"name": "hidden_5_fc", "path": dl_files_path + "hidden_5_fc.dl"},
        {"name": "hidden_10_fc", "path": dl_files_path + "hidden_10_fc.dl"},
        # {"name": "hidden_20_fc", "path": dl_files_path + "hidden_20_fc.dl"},
    ]

    RESULTS_ROOT = "results"
    N_EPOCHS = 5#10
    N_TEST_RUNS = 500  # simulated elevator-waiting episodes per model, for step 2

    # RL / MAB / MORL training budget (episodes for RL/MORL, pulls for bandits;
    # same budget for every model, as in run_all_experiments.py).
    RL_EPISODES = 3000
    RL_EVAL_EPISODES = 500
    RL_SEED = 0

    # ------------------------------------------------------------------ #
    # STEP 0: train the RL / MAB / MORL models ONCE per profile, before
    # anything else. They don't depend on the .dl file, so the result is
    # re-used for every .dl file of the same profile.
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 0 - training RL / MAB / MORL models for every profile")
    print("=" * 70)
    rl_mab_by_profile = train_all_profiles(
        [d["name"] for d in DATASETS],
        episodes=RL_EPISODES, eval_episodes=RL_EVAL_EPISODES, seed=RL_SEED,
        outdir=os.path.join(RESULTS_ROOT, "rl_mab"),
    )

    run_summaries = []
    for dataset_cfg in DATASETS:
        for dl_cfg in DL_FILES:
            summary = run_pipeline(
                dataset_name=dataset_cfg["name"],
                dataset_path=dataset_cfg["path"],
                dl_name=dl_cfg["name"],
                case_file=dl_cfg["path"],
                results_root=RESULTS_ROOT,
                n_epochs=N_EPOCHS,
                n_test_runs=N_TEST_RUNS,
                rl_mab=rl_mab_by_profile.get(dataset_cfg["name"]),
            )
            run_summaries.append({"dataset": dataset_cfg["name"], "dl_file": dl_cfg["name"], **summary})

    print("=" * 70)
    print(f"ALL RUNS COMPLETE - {len(run_summaries)} (dataset x .dl file) combinations")
    print("=" * 70)
    for run in run_summaries:
        print(f"  [{run['dataset']} / {run['dl_file']}] -> {run['results_dir']}")

    # A flat index of every run, handy for quickly locating any result.
    index_path = os.path.join(RESULTS_ROOT, "run_index.csv")
    pd.DataFrame(run_summaries).to_csv(index_path, index=False)
    print(f"Saved run index -> {index_path}")