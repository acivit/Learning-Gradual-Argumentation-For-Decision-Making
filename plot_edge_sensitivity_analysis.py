"""
Plots the edge-sensitivity CSVs written by train.py
(results/edge_sensitivity_metrics_<profile>.csv) and saves the figures.

Run:  python plot_edge_sensitivity.py
Everything you may want to rename / recolour is in the CONFIG section below.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ====================================================================== #
# CONFIG - edit this section
# ====================================================================== #

# ---- Input / output -------------------------------------------------- #
RESULTS_DIR = "results"
CSV_GLOB = "edge_sensitivity_metrics_*.csv"          # searched inside RESULTS_DIR
OUTPUT_DIR = os.path.join(RESULTS_DIR, "edge_sensitivity_plots_custom")
FILE_FORMAT = "png"                            # "png", "pdf", "svg", ...
DPI = 200

# ---- Layout ---------------------------------------------------------- #
# "grid"     -> one figure per (profile, edge) with a subplot per metric
# "separate" -> one figure per (profile, edge, metric)
# Use a list to produce both, e.g. ["grid", "separate"].
PLOT_MODES = ["grid"]
GRID_COLUMNS = 3
SUBPLOT_SIZE = (4.2, 3.2)                      # width, height of each subplot
SEPARATE_FIGSIZE = (6, 4)
SHOW_LEGEND = True
SHOW_GRID = True
SHOW_BASELINE_LINE = True                      # dotted vertical line at the base value
MARKER = "o"                                   # marker for single-profile figures
LINE_WIDTH = 1.3

# ---- Several robot profiles in the same figure ----------------------- #
# One figure per group and per edge; each group name -> list of profiles
# (the <profile> part of the CSV filename). Colour = dl_file (shared with the
# single-profile figures), line style + marker = profile.
# Use {} to disable. Profiles whose CSV is missing are skipped.
PROFILE_GROUPS = {
    "all": ["complacent", "time-efficient", "comfort-caring", "comfort-blind"],
    # "two": ["complacent", "comfort-blind"],
}
GROUP_LABELS = {
    # "all": "All robot profiles",             # default: profile names joined by ", "
}
PLOT_INDIVIDUAL = True                         # also keep the one-figure-per-profile plots
# Line style / marker per profile. Missing ones use the default cycles below.
PROFILE_STYLES = {
    # "complacent": {"linestyle": "-", "marker": "o"},
}
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P"]
COMBINED_LEGEND_FONTSIZE = 8

# ---- Edges (x axes) -------------------------------------------------- #
# Keys must match the names used in train.py / settings.py.
EDGE_BASE = {"inside_comfort": 1, "breaking_queue": 1, "robot_urgency": 1}
EDGE_LABELS = {
    "inside_comfort": "Passenger Comfort → Enter",
    "breaking_queue": "Queue Compliance → enter",
    "robot_urgency": "Task Urgency → Enter",
}
EDGE_FIGURE = {
    "inside_comfort": "Passenger Comfort",
    "breaking_queue": "Queue Compliance",
    "robot_urgency": "Task Urgency",
}

# ---- Metrics (y axes) ------------------------------------------------ #
# None = plot every numeric column of the CSV (except the edge_factor_* ones).
# Or give a list of column names, in the order you want them.
METRICS = None
METRICS_TO_SKIP = []                           # only used when METRICS is None
# Display names; columns not listed here keep their CSV name.


# ---- Lines (one per value of HUE_COL) -------------------------------- #
HUE_COL = "dl_file"                            # set to None for a single line
# Display names and colours per HUE value. Missing ones use the default cycle.
HUE_LABELS = {
    "normal": "Original",
    "hidden_5": "5 Hidden SD",
    "hidden_5_fc": "5 Hidden FC",
    "hidden_10": "10 Hidden SD",
    "hidden_10_fc": "10 Hidden FC",
}
HUE_COLORS = {
    # "my_file.dl": "tab:blue",
}
DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

# ---- Titles ---------------------------------------------------------- #
# Display names for the robot profile (taken from the CSV filename).
PROFILE_LABELS = {
    "complacent": "Socially Compliant",
    "comfort-blind": "Comfort-blind",
    "comfort-caring": "Comfort-caring",
    "time-efficient": "Time-efficient",
}
FIGURE_TITLE = "Edge Weight Sensitivity: {edge} (profile: {profile})"   # {edge}, {profile}
SEPARATE_TITLE = "{metric} vs {edge} (profile: {profile})"

METRIC_LABELS = {
    "success_pct": "Success %",
    "skipped_queue_pct": "Skipped Queue %",
    "priority_violations_pct": "Priority Violations %",
    "impairment_squishments_pct": "PC Forced Movements %",
    "elevators_waited_mean": "Elevators Waited (mean)",
}

# ====================================================================== #
# End of CONFIG
# ====================================================================== #


def _profile_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = CSV_GLOB.split("*")[0]
    return name[len(prefix):] if name.startswith(prefix) else name


def _select_metrics(df):
    if METRICS is not None:
        missing = [m for m in METRICS if m not in df.columns]
        if missing:
            print(f"  Warning: metrics not in CSV, skipped: {missing}")
        return [m for m in METRICS if m in df.columns]
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith("edge_factor_")
            and c not in METRICS_TO_SKIP]


def _color_for(hue_value, hue_values):
    if hue_value in HUE_COLORS:
        return HUE_COLORS[hue_value]
    return DEFAULT_COLORS[list(hue_values).index(hue_value) % len(DEFAULT_COLORS)]


def _style_for(profile, profile_names):
    """Line style + marker of a profile (only differs between profiles in
    combined figures; a single profile keeps a solid line and MARKER)."""
    if len(profile_names) == 1:
        return {"linestyle": "-", "marker": MARKER}
    i = list(profile_names).index(profile)
    style = {"linestyle": DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
             "marker": DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)]}
    style.update(PROFILE_STYLES.get(profile, {}))
    return style


def _draw_metric(ax, data, metric, source, hue_values, profile_names):
    """data: {profile: slice_df}. One line per (profile, hue value)."""
    x_col = f"edge_factor_{source}"
    for profile in profile_names:
        slice_df = data[profile]
        style = _style_for(profile, profile_names)
        if HUE_COL is None or HUE_COL not in slice_df.columns:
            groups = [(None, slice_df)]
        else:
            groups = [(h, slice_df[slice_df[HUE_COL] == h]) for h in hue_values]
        for hue, g in groups:
            if g.empty:
                continue
            curve = g.groupby(x_col)[metric].mean().sort_index()
            kwargs = {}
            if hue is not None:
                kwargs = dict(color=_color_for(hue, hue_values), label=HUE_LABELS.get(hue, str(hue)))
            elif len(profile_names) > 1:
                kwargs = dict(color="black")
            ax.plot(curve.index, curve.values, markersize=3, linewidth=LINE_WIDTH,
                    **style, **kwargs)
    if SHOW_BASELINE_LINE:
        ax.axvline(EDGE_BASE[source], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Edge Weight")#EDGE_LABELS.get(source, source))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if SHOW_GRID:
        ax.grid(alpha=0.3)


def _add_combined_legends(fig, hue_values, profile_names):
    """Figure-level legends below the plots: colours = dl_file, styles = profile."""
    if not SHOW_LEGEND:
        return
    handles_hue = []
    if HUE_COL is not None:
        handles_hue = [Line2D([0], [0], color=_color_for(h, hue_values), linewidth=LINE_WIDTH,
                              label=HUE_LABELS.get(h, str(h))) for h in hue_values]
    handles_profile = []
    for p in profile_names:
        st = _style_for(p, profile_names)
        handles_profile.append(Line2D([0], [0], color="black", linewidth=LINE_WIDTH, markersize=4,
                                      label=PROFILE_LABELS.get(p, p), **st))
    if handles_hue:
        leg = fig.legend(handles=handles_hue, loc="lower center", bbox_to_anchor=(0.5, 0.04),
                         ncol=len(handles_hue), fontsize=COMBINED_LEGEND_FONTSIZE, frameon=False)
        fig.add_artist(leg)
    fig.legend(handles=handles_profile, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles_profile), fontsize=COMBINED_LEGEND_FONTSIZE, frameon=False)


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.{FILE_FORMAT}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_profiles():
    """Returns {profile: dataframe with the edge factors present}."""
    profiles = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, CSV_GLOB))):
        profile = _profile_from_path(path)
        df = pd.read_csv(path)
        print(f"{path}: {len(df)} rows")
        factor_cols = [f"edge_factor_{s}" for s in EDGE_BASE]
        if not all(c in df.columns for c in factor_cols):
            print("  edge_factor_* columns missing, skipping.")
            continue
        df = df.dropna(subset=factor_cols)
        if df.empty:
            print("  nothing to plot, skipping.")
            continue
        profiles[profile] = df
    return profiles


def plot_profiles(profiles, profile_names, file_prefix, label):
    """Plots the given profiles together (one profile = a normal figure)."""
    df_all = pd.concat([profiles[p] for p in profile_names], ignore_index=True)
    metrics = _select_metrics(df_all)
    if not metrics:
        print("  nothing to plot, skipping.")
        return
    hue_values = []
    if HUE_COL and HUE_COL in df_all.columns:
        present = set(df_all[HUE_COL].dropna().unique())
        # Order of HUE_LABELS first (legend / colour order), then any other value.
        hue_values = [h for h in HUE_LABELS if h in present] + sorted(present - set(HUE_LABELS))
    combined = len(profile_names) > 1
    bottom = 0.10 if combined and SHOW_LEGEND else 0.0

    for source in EDGE_BASE:
        # Only edges that were actually varied in the sensitivity analysis.
        if df_all[f"edge_factor_{source}"].nunique() < 2:
            continue
        data = {}
        for p in profile_names:
            mask = np.ones(len(profiles[p]), dtype=bool)
            for other in EDGE_BASE:
                if other != source:
                    mask &= (profiles[p][f"edge_factor_{other}"] == EDGE_BASE[other]).to_numpy()
            data[p] = profiles[p][mask]
        if all(d.empty for d in data.values()):
            continue
        edge_label = EDGE_LABELS.get(source, source)

        if "grid" in PLOT_MODES:
            n_cols = min(GRID_COLUMNS, len(metrics))
            n_rows = int(np.ceil(len(metrics) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                                     figsize=(SUBPLOT_SIZE[0] * n_cols, SUBPLOT_SIZE[1] * n_rows))
            for i, metric in enumerate(metrics):
                ax = axes[i // n_cols][i % n_cols]
                _draw_metric(ax, data, metric, source, hue_values, profile_names)
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
                #ax.set_ylabel("")
                if SHOW_LEGEND and hue_values and i == 0 and not combined:
                    ax.legend(fontsize=8)
            for j in range(len(metrics), n_rows * n_cols):
                axes[j // n_cols][j % n_cols].axis("off")
            fig.suptitle(FIGURE_TITLE.format(edge=edge_label, profile=label))
            fig.tight_layout(rect=[0, bottom, 1, 0.96])
            if combined:
                _add_combined_legends(fig, hue_values, profile_names)
            _save(fig, f"{file_prefix}_{source}_grid")

        if "separate" in PLOT_MODES:
            for metric in metrics:
                fig, ax = plt.subplots(figsize=SEPARATE_FIGSIZE)
                _draw_metric(ax, data, metric, source, hue_values, profile_names)
                ax.set_title(SEPARATE_TITLE.format(
                    metric=METRIC_LABELS.get(metric, metric), edge=edge_label, profile=label))
                if SHOW_LEGEND and hue_values and not combined:
                    ax.legend(fontsize=8)
                fig.tight_layout(rect=[0, bottom, 1, 1])
                if combined:
                    _add_combined_legends(fig, hue_values, profile_names)
                _save(fig, f"{file_prefix}_{source}_{metric}")


def main():
    profiles = load_profiles()
    if not profiles:
        print(f"No usable files matching {CSV_GLOB} in {RESULTS_DIR}/")
        return

    if PLOT_INDIVIDUAL:
        for profile in profiles:
            plot_profiles(profiles, [profile], profile, PROFILE_LABELS.get(profile, profile))

    for group, members in PROFILE_GROUPS.items():
        available = [p for p in members if p in profiles]
        missing = [p for p in members if p not in profiles]
        if missing:
            print(f"Group '{group}': no CSV for {missing}, skipped.")
        if len(available) < 2:
            print(f"Group '{group}': needs at least 2 profiles, skipping.")
            continue
        label = GROUP_LABELS.get(group, ", ".join(PROFILE_LABELS.get(p, p) for p in available))
        plot_profiles(profiles, available, f"group_{group}", label)


if __name__ == "__main__":
    main()