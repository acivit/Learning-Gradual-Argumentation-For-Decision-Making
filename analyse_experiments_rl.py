"""
analyse_experiments.py
------------------------

Metrics computed:

    success_pct                - % of runs where the model ever decided "enter"
    skipped_queue_pct          - of the successful runs, % where there was a
                                  queue (number_users_before_robot > 0) that
                                  the robot entered ahead of
    priority_violations_pct    - of the successful runs, % where a priority
                                  passenger was still waiting when the robot entered
    impairment_squishments_pct - of the successful runs, % where people with
                                  reduced mobility were present in the elevator
                                  when the robot entered
    waiting_time_mean          - mean robot_time_waiting at the moment of
                                  entering, over the successful runs

Plots
-----
Three figures are saved to --plots-dir (default: alongside --out):

  metrics_by_model_and_profile.png
      One horizontal grouped-bar panel per metric (success, skipped queue,
      priority violations, impairment squishments, mean wait time), models
      on the y-axis, one bar per profile. This is the "just tell me the
      number for every model/profile/metric combination" view.

  tradeoff_waiting_vs_violations.png
      One scatter subplot per profile: mean wait time (x) vs. how often the
      model violated a queue/priority/comfort norm on the way in (y),
      averaged across the three violation metrics. Point size encodes
      success rate (bigger = more reliably got the robot in at all). This is
      the "what does this model actually trade off" view - fast-and-rude
      models land top-left, slow-and-considerate models land bottom-right.
      Models that never once entered can't have a wait time or a violation
      rate, so they're listed in a text box instead of plotted.

  success_vs_waiting_time.png
      One scatter subplot per profile: success rate (x) vs. mean wait time
      (y). Answers "of the models that do get the robot in, which ones are
      both reliable AND fast?" (bottom-right is the sweet spot).

  Radar plots
      Include the radar plot for the metrics. 
Usage
-----
    python analyse_experiments.py
    python analyse_experiments.py --log pipeline_outputs/combined_decision_log.csv --out pipeline_outputs/robot_metrics_summary.csv
"""

import argparse
import csv
import os

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from settings import frozen_nodes, interpretable_arguments, EDGE_DISCOUNT_TARGET, EDGE_DISCOUNT_VARIANTS, NORMALISED_MODEL_NAME, ABLATION_VARIANTS, BASELINE_MODEL_NAME


def compute_statistics(metrics, robot_profiles, models):
    """Compute statistics from the metrics data"""
    results = {}
    for profile in robot_profiles:
        results[profile] = {}
        for model in models:
            profile_model_data = metrics[profile][model]
            
            # Calculate percentages (already in 0-100 range)
            success_rate = np.mean(profile_model_data["success"]) * 100 if profile_model_data["success"] else 0
            skipped_rate = np.mean(profile_model_data["skipped_queue"]) * 100 if profile_model_data["skipped_queue"] else 0
            priority_rate = np.mean(profile_model_data["priority_violations"]) * 100 if profile_model_data["priority_violations"] else 0
            impairment_rate = np.mean(profile_model_data["impairment_squishments"]) * 100 if profile_model_data["impairment_squishments"] else 0
            elevators_waited = np.mean(profile_model_data["elevators_waited"]) if profile_model_data["elevators_waited"] else 0
            # Mean waiting time (only for successful entries)
            waiting_times = profile_model_data["waiting_time"]
            waiting_mean = np.mean(waiting_times) if waiting_times else 0
            
            results[profile][model] = {
                "success": success_rate,
                "skipped_queue": skipped_rate,
                "priority_violations": priority_rate,
                "impairment_squishments": impairment_rate,
                "waiting_time_mean": waiting_mean,
                "elevators_waited_mean": elevators_waited,
            }
    return results


METRIC_FIELDS = ["success", "skipped_queue", "priority_violations", "impairment_squishments", #"waiting_time_mean", 
                 "elevators_waited_mean"]
METRIC_LABELS = {
    "success": "Entered before timeout (%)",
    "skipped_queue": "Entered ahead of a waiting queue (%)",
    "priority_violations": "Entered while a priority passenger waited (%)",
    "impairment_squishments": "Entered with people with reduced mobility present (%)",
    "waiting_time_mean": "Mean wait time until entering",
    "elevators_waited_mean": "Mean number of elevators waited "
}
VIOLATION_FIELDS = ["skipped_queue", "priority_violations", "impairment_squishments"]

# RL / MAB / MORL models (trained in rl_mab_models.py). They are identified by
# their exact model name in the log. NOTE: keep in sync with
# RL_MAB_MODEL_NAMES in rl_mab_models.py.
RL_MAB_MODEL_NAMES = [
    "tabular_q", "dqn", "contextual_bandit", "neural_contextual_bandit",
    "vector_q_morl", "multihead_dqn_morl",
]
# Nicer legend labels for the radar charts (the log/CSVs keep the raw names).
RL_MAB_LABELS = {
    "tabular_q": "Tabular Q-learning (RL)",
    "dqn": "DQN (RL)",
    "contextual_bandit": "Contextual bandit (MAB)",
    "neural_contextual_bandit": "Neural contextual bandit (MAB)",
    "vector_q_morl": "Vector Q (MORL)",
    "multihead_dqn_morl": "Multi-head DQN (MORL)",
}


def _model_colors(models):
    """One consistent color per model, reused across every plot so a model
    is always the same color no matter which figure you're looking at."""
    sorted_models = sorted(models)
    cmap = plt.get_cmap("tab10" if len(sorted_models) <= 10 else "tab20")
    return {m: cmap(i % cmap.N) for i, m in enumerate(sorted_models)}


def plot_metrics_by_model_and_profile(results, robot_profiles, models, plots_dir):
    """One horizontal grouped-bar panel per metric: models on the y-axis,
    one bar per profile, so you can directly compare every model/profile
    combination for a single metric at a glance."""
    sorted_models = sorted(models)
    profile_cmap = plt.get_cmap("Set2")
    profile_colors = {p: profile_cmap(i / max(len(robot_profiles) - 1, 1)) for i, p in enumerate(robot_profiles)}

    ncols = 2
    nrows = -(-len(METRIC_FIELDS) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows))
    axes = np.array(axes).flatten()

    y_pos = np.arange(len(sorted_models))
    n_profiles = len(robot_profiles)
    bar_h = 0.8 / n_profiles

    for ax, field in zip(axes, METRIC_FIELDS):
        for j, profile in enumerate(robot_profiles):
            values = [results[profile][m][field] for m in sorted_models]
            values = [0 if np.isnan(v) else v for v in values]  # nan (never entered) -> 0-length bar
            offset = (j - (n_profiles - 1) / 2) * bar_h
            ax.barh(y_pos + offset, values, height=bar_h, color=profile_colors[profile], label=profile)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_models, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(METRIC_LABELS[field])
        ax.set_title(METRIC_LABELS[field], fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[len(METRIC_FIELDS):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(robot_profiles),
               bbox_to_anchor=(0.5, 1.02), title="Robot profile", fontsize=10)
    fig.suptitle("Outcome metrics by model and profile\n(bars at 0 = model never entered before timeout, so the metric is undefined)",
                 y=1.09, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_by_model_and_profile.png"), dpi=130, bbox_inches="tight")
    plt.close()


def plot_tradeoff_waiting_vs_violations(results, robot_profiles, models, plots_dir):
    """Per profile: mean wait time (x) vs. average queue/priority/comfort
    violation rate (y), point size = success rate. Shows the fundamental
    speed-vs-consideration trade-off each model strikes."""
    colors = _model_colors(models)
    sorted_models = sorted(models)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ax, profile in zip(axes, robot_profiles):
        never_entered = []
        for m in sorted_models:
            res = results[profile][m]
            wt = res["waiting_time_mean"]
            viol_values = [res[f] for f in VIOLATION_FIELDS if not np.isnan(res[f])]
            if np.isnan(wt) or not viol_values:
                never_entered.append(m)
                continue
            viol = float(np.mean(viol_values))
            success = res["success"]
            size = 40 + 4 * success  # bigger marker = more reliable (higher success rate)
            ax.scatter(wt, viol, s=size, color=colors[m], edgecolor="black", linewidth=0.6,
                       alpha=0.85, zorder=3)
            ax.annotate(m, (wt, viol), fontsize=7.5, xytext=(5, 4), textcoords="offset points")

        ax.set_title(profile, fontsize=12)
        ax.set_xlabel("Mean wait time until entering")
        ax.set_ylabel("Avg. queue/priority/comfort violation rate (%)")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-5)

        if never_entered:
            ax.text(0.02, 0.02, "Never entered:\n" + "\n".join(never_entered),
                    transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                    bbox=dict(boxstyle="round", facecolor="mistyrose", edgecolor="gray", alpha=0.8))

    for ax in axes[len(robot_profiles):]:
        ax.axis("off")

    fig.suptitle("Speed vs. social/comfort cost trade-off\n"
                 "(bottom-left = fast AND considerate; marker size = success rate)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tradeoff_waiting_vs_violations.png"), dpi=130)
    plt.close()


def plot_success_vs_waiting_time(results, robot_profiles, models, plots_dir):
    """Per profile: success rate (x) vs. mean wait time (y) - which models
    are both reliable (high success) and fast (low wait time)?"""
    colors = _model_colors(models)
    sorted_models = sorted(models)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for ax, profile in zip(axes, robot_profiles):
        for m in sorted_models:
            res = results[profile][m]
            success = res["success"]
            wt = res["waiting_time_mean"]
            if np.isnan(wt):
                continue  # 0% success -> no wait time to plot (it's already visible on the x-axis)
            ax.scatter(success, wt, s=70, color=colors[m], edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(m, (success, wt), fontsize=7.5, xytext=(5, 4), textcoords="offset points")
        ax.set_title(profile, fontsize=12)
        ax.set_xlabel("Success rate (% entered before timeout)")
        ax.set_ylabel("Mean wait time until entering")
        ax.set_xlim(-5, 105)
        ax.grid(alpha=0.3)

    for ax in axes[len(robot_profiles):]:
        ax.axis("off")

    fig.suptitle("Reliability vs. speed\n(bottom-right = reliably gets the robot in, fast)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "success_vs_waiting_time.png"), dpi=130)
    plt.close()


def make_plots(results, robot_profiles, models, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    plot_metrics_by_model_and_profile(results, robot_profiles, models, plots_dir)
    plot_tradeoff_waiting_vs_violations(results, robot_profiles, models, plots_dir)
    plot_success_vs_waiting_time(results, robot_profiles, models, plots_dir)
    print(f"Saved plots -> {plots_dir}")

def prepare_dataframe_for_plots(results, robot_profiles, models):
    """Convert results dictionary to tidy DataFrame for plotting"""
    rows = []
    for profile in robot_profiles:
        for model in models:
            res = results[profile][model]
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "success",
                "value": res["success"]
            })
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "skipped_queue",
                "value": res["skipped_queue"]
            })
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "priority_violations",
                "value": res["priority_violations"]
            })
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "impairment_squishments",
                "value": res["impairment_squishments"]
            })
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "waiting_time_mean",
                "value": res["waiting_time_mean"]
            })
            rows.append({
                "robot_profile": profile,
                "model": model,
                "metric": "elevators_waited_mean",
                "value": res["elevators_waited_mean"]
            })
    return pd.DataFrame(rows)



def normalize_waiting_time(df_pivot):
    """
    Normalize only the waiting_time_mean column to 0-1 scale.
    Returns the normalized value where 1 = best (lowest waiting time).
    """
    df_normalized = df_pivot.copy()
    
    # Get min and max of waiting time from full dataset
    min_val_time = 0#df_pivot['waiting_time_mean'].min()
    max_val_time = 200#df_pivot['waiting_time_mean'].max()

    min_val_elevators = 0
    max_val_elevators = 4
    
    # print(f"\nNormalizing waiting_time: min={min_val:.2f}, max={max_val:.2f}")
    
    if max_val_time > min_val_time:
        # Normalize to 0-1
        df_normalized['waiting_time_norm'] = (df_pivot['waiting_time_mean'] - min_val_time) / (max_val_time - min_val_time)
        # Invert so that lower waiting time = higher score (1 = best)
        #df_normalized['waiting_time_norm'] = 1 - df_normalized['waiting_time_norm']
    else:
        # If all values are the same, set to 0.5
        df_normalized['waiting_time_norm'] = 0.5

    df_normalized['elevators_waited_norm'] = (df_pivot['elevators_waited_mean'] - min_val_elevators) / (max_val_elevators - min_val_elevators)
    
    return df_normalized



def plot_faceted_barchart(df, plots_dir):
    """Create faceted bar chart with one panel per metric"""
    # Set nice labels for metrics
    metric_labels = {
        "success": "Success Rate (%)",
        "skipped_queue": "Skipped Queue (%)",
        "priority_violations": "Priority Violations (%)",
        "impairment_squishments": "Impairment Squishments (%)",
        "waiting_time_mean": "Mean Waiting Time (s)",
        "elevators_waited_mean": "Mean Elevators Waited",
    }
    
    # Create the plot
    g = sns.catplot(
        data=df, 
        x='model', 
        y='value', 
        hue='robot_profile',
        col='metric', 
        kind='bar',
        height=4, 
        aspect=0.8,
        col_wrap=3,  # 3 columns, 2 rows for 5 metrics
        sharey=False,  # Each metric has its own y-scale
        legend_out=True
    )
    
    # Update axis labels and titles
    for ax, metric in zip(g.axes.flat, df['metric'].unique()):
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_labels.get(metric, "Value"))
        # Rotate x-labels if needed
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust legend
    g._legend.set_title("Robot Profile")
    g._legend.set_bbox_to_anchor((1.05, 0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(plots_dir, "faceted_barchart.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved faceted bar chart to: {save_path}")
    plt.close()

def plot_normalized_heatmap(df_pivot, plots_dir):
    """
    Create a heatmap with per-column normalized values.
    Each metric is normalized independently using its own min/max.
    Expects df_pivot to already be in pivoted format with columns:
    ['robot_profile', 'model', 'success', 'skipped_queue', 'priority_violations', 
     'impairment_squishments', 'waiting_time_mean']
    """
    # Create a combined index for row labels
    df_pivot['profile_model'] = df_pivot['robot_profile'] + " | " + df_pivot['model']
    
    # Extract just the metric columns for normalization
    metric_cols = ['success', 'skipped_queue', 'priority_violations', 
                   'impairment_squishments', #'waiting_time_mean', 
                   'elevators_waited_mean']
    
    # Normalize each column independently using MinMaxScaler
    # This scales each column to [0,1] using its own min and max
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_pivot[metric_cols])
    
    # Create DataFrame with normalized values
    df_normalized = pd.DataFrame(
        normalized_data, 
        columns=metric_cols,
        index=df_pivot['profile_model']
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_normalized, 
        annot=True,  # Show values in cells
        fmt='.2f',
        cmap='RdYlGn',  # Red (bad) to Green (good)
        cbar_kws={'label': 'Normalized Score (0=worst, 1=best)'},
        linewidths=0.5,
        square=True
    )
    plt.title("Performance Heatmap (Per-Column Normalization)", fontsize=14, pad=20)
    plt.xlabel("Metrics")
    plt.ylabel("Robot Profile | Model")
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(plots_dir, "normalized_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalized heatmap to: {save_path}")
    plt.close()


def plot_radar_charts_individual(df_pivot, df_plot_arg, plots_dir, argumentation_models):
    """
    Create individual radar charts for each profile, only for argumentation models.
    Percentages are shown as-is (0-100), waiting time is normalized to 0-1.
    """
    if df_plot_arg.empty:
        print("Warning: No argumentation models found in the data. Skipping radar charts.")
        return
    
    # Get the full dataset with normalized waiting time
    df_full_normalized = normalize_waiting_time(df_pivot)
    
    # Filter for argumentation models
    df_arg_filtered = df_plot_arg[df_plot_arg['model'].isin(argumentation_models)]
    
    # Pivot the argumentation data
    df_arg_pivot = df_arg_filtered.pivot_table(
        index=['robot_profile', 'model'], 
        columns='metric', 
        values='value'
    ).reset_index()
    
    # Get the normalized waiting time for these specific rows
    # We need to merge with the normalized data
    df_arg_with_norm = df_arg_pivot.merge(
        df_full_normalized[['robot_profile', 'model', #'waiting_time_norm',
                             'elevators_waited_norm']],
        on=['robot_profile', 'model'],
        how='left'
    )
    
    # Now we need to normalize each percentage metric to 0-1 for radar plotting
    # We want them on the same scale as the normalized waiting time (0-1)
    # metric_cols = ['success', 'skipped_queue', 'priority_violations', 'impairment_squishments']
    
    # for col in metric_cols:
    #     min_val = df_arg_with_norm[col].min()
    #     max_val = df_arg_with_norm[col].max()
    #     if max_val > min_val:
    #         df_arg_with_norm[col + '_norm'] = (df_arg_with_norm[col] - min_val) / (max_val - min_val)
    #     else:
    #         df_arg_with_norm[col + '_norm'] = 0.5
    
    # Get unique profiles and models
    profiles = df_arg_with_norm['robot_profile'].unique()
    models = df_arg_with_norm['model'].unique()
    
    print(f"\nCreating radar charts for argumentation models: {models}")
    print("Percentages normalized to 0-1 for radar display, waiting time normalized 0-1 (1=best)")
    
    # Metric names for radar
    metric_names = ['Success', 'Queues\nSkipped', "Priority Access\nViolation Rate"#'Priority\nViolations', 
                   "Impairment\nSqueezements"#'Impairment\nSquishments', #'Waiting\nTime (inv)',
                    "Mean Elevators\nWaited"]
    
    # Number of metrics
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Define a color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Create individual radar chart for each profile
    for profile in profiles:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        
        profile_data = df_arg_with_norm[df_arg_with_norm['robot_profile'] == profile]
        
        # Plot each model for this profile
        for idx, (_, row) in enumerate(profile_data.iterrows()):
            model = row['model']
            values = [row['success']/100.0, row['skipped_queue']/100.0, 
                     row['priority_violations']/100.0, row['impairment_squishments']/100.0,
                     #row['waiting_time_norm'],
                      row["elevators_waited_norm"]]
            values += values[:1]  # Close the loop
            
            color = colors[idx % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Set the axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
        ax.grid(True)
        
        # Set title
        plt.title(f"Argumentation Models - Robot Profile: {profile}", size=14, pad=30)
        
        # Add legend with offset to avoid occlusion
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 0.9), 
                  title="Models", title_fontsize=11, fontsize=10)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        # Save individual radar chart
        safe_profile_name = str(profile).replace(' ', '_').replace('/', '_')
        save_path = os.path.join(plots_dir, f"radar_chart_argumentation_{safe_profile_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved argumentation radar chart for profile '{profile}' to: {save_path}")
        plt.close()
    
    print(f"Created {len(profiles)} individual radar charts for argumentation models")

# def plot_radar_charts_combined(df_pivot, df_plot, plots_dir):
#     """
#     Create separate radar charts for each profile using the same format as individual charts.
#     Percentages are shown as-is (0-100), waiting time is normalized to 0-1.
#     Includes ALL models, not just argumentation ones.
#     """
#     if df_plot.empty:
#         print("Warning: No data available for combined radar charts. Skipping.")
#         return
    
#     # Get the full dataset with normalized waiting time
#     df_full_normalized = normalize_waiting_time(df_pivot)
    
#     # Pivot the data for all models
#     df_all_pivot = df_plot.pivot_table(
#         index=['robot_profile', 'model'], 
#         columns='metric', 
#         values='value'
#     ).reset_index()
    
#     # Get the normalized waiting time for these specific rows
#     df_all_with_norm = df_all_pivot.merge(
#         df_full_normalized[['robot_profile', 'model', #'waiting_time_norm',
#                              'elevators_waited_norm']],
#         on=['robot_profile', 'model'],
#         how='left'
#     )
    
#     # Now we need to normalize each percentage metric to 0-1 for radar plotting
#     # We want them on the same scale as the normalized waiting time (0-1)
#     metric_cols = ['success', 'skipped_queue', 'priority_violations', 'impairment_squishments']
    
#     for col in metric_cols:
#         min_val = df_all_with_norm[col].min()
#         max_val = df_all_with_norm[col].max()
#         if max_val > min_val:
#             df_all_with_norm[col + '_norm'] = (df_all_with_norm[col] - min_val) / (max_val - min_val)
#         else:
#             df_all_with_norm[col + '_norm'] = 0.5
    
#     # Get unique profiles and models
#     profiles = df_all_with_norm['robot_profile'].unique()
#     models = df_all_with_norm['model'].unique()
    
#     print(f"\nCreating radar charts with ALL models: {models}")
#     print("Percentages normalized to 0-1 for radar display, waiting time normalized 0-1 (1=best)")
    
#     # Metric names for radar
#     metric_names = ['Success', 'Skipped\nQueue', 'Priority\nViolations', 
#                    'Impairment\nSquishments', #'Waiting\nTime (inv)',
#                      'Elevators\nWaited']
    
#     # Number of metrics
#     N = len(metric_names)
#     angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#     angles += angles[:1]  # Close the loop
    
#     # Define a color palette for models
#     colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
#     # Create individual radar chart for each profile
#     for profile in profiles:
#         plt.figure(figsize=(8, 8))
#         ax = plt.subplot(111, projection='polar')
        
#         profile_data = df_all_with_norm[df_all_with_norm['robot_profile'] == profile]
        
#         # Plot each model for this profile
#         for idx, (_, row) in enumerate(profile_data.iterrows()):
#             model = row['model']
#             values = [row['success']/100.0, row['skipped_queue']/100.0, 
#                      row['priority_violations']/100.0, row['impairment_squishments']/100.0,
#                      #row['waiting_time_norm'], 
#                      row["elevators_waited_norm"]]
#             values += values[:1]  # Close the loop
            
#             color = colors[idx % len(colors)]
#             ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, alpha=0.8)
#             ax.fill(angles, values, alpha=0.15, color=color)
        
#         # Set the axis labels
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(metric_names, size=11)
#         ax.set_ylim(0, 1)
#         ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#         ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
#         ax.grid(True)
        
#         # Set title
#         plt.title(f"All Models - Robot Profile: {profile}", size=14, pad=30)
        
#         # Add legend with offset to avoid occlusion
#         plt.legend(loc='upper left', bbox_to_anchor=(1.2, 0.9), 
#                   title="Models", title_fontsize=11, fontsize=10)
        
#         # Adjust layout to prevent text cutoff
#         plt.tight_layout()
        
#         # Save individual radar chart
#         safe_profile_name = str(profile).replace(' ', '_').replace('/', '_')
#         save_path = os.path.join(plots_dir, f"radar_chart_all_models_{safe_profile_name}.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved combined radar chart for profile '{profile}' to: {save_path}")
#         plt.close()
    
#     print(f"Created {len(profiles)} radar charts with all models")


def _draw_radar(profile_data, profile, subset_models, model_to_color, model_to_style,
                title_prefix, file_prefix, plots_dir,
                r_floor=0.03, max_jitter_deg=1.5):
    """Draw and save one radar chart for `profile`, restricted to `subset_models`."""
    subset_models = list(subset_models)
    data = profile_data[profile_data['model'].isin(subset_models)]
    if data.empty:
        print(f"Warning: no data for '{title_prefix}' on profile '{profile}'. Skipping.")
        return

    metric_names = ['Success', 'Skipped\nQueue', 'Priority\nViolations',
                    'PC Forced\nMovements', 'Elevators\nWaited']
    N = len(metric_names)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Jitter is spread over the models actually shown in this plot
    n = max(len(subset_models), 1)
    model_to_jitter = {
        m: np.deg2rad(-max_jitter_deg + 2 * max_jitter_deg * (i / max(n - 1, 1)))
        for i, m in enumerate(subset_models)
    }

    plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, projection='polar')

    for _, row in data.iterrows():
        model = row['model']
        raw_values = [
            row['success'] / 100.0,
            row['skipped_queue'] / 100.0,
            row['priority_violations'] / 100.0,
            row['impairment_squishments'] / 100.0,
            row['elevators_waited_norm'],
        ]
        values = [max(v, r_floor) if pd.notna(v) else r_floor for v in raw_values]
        values += values[:1]

        angles = (base_angles + model_to_jitter[model]).tolist()
        angles += angles[:1]

        color = ""
        color_set = False
        for variant in EDGE_DISCOUNT_VARIANTS:
            if variant["model_name"] == model:
                color = variant["color"]
                color_set = True
            
        for variant in ABLATION_VARIANTS:
            if variant["model_name"] == model:
                color = variant["color"]
                color_set = True

        if not color_set:
            color = model_to_color[model]
        marker, linestyle = model_to_style[model]

        label = model
        if "trained" in model:
            label = "Trained Model"
            color = "cyan"

        if label == "baseline_QE":
            label = "Baseline (Quadratic Energy Semantics)"
        label = RL_MAB_LABELS.get(label, label)
        ax.plot(angles, values, marker=marker, linestyle=linestyle,
                linewidth=2, markersize=6, label=label, color=color, alpha=0.85)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(base_angles)
    ax.set_xticklabels(metric_names, size=16)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True)

    plt.title(f"{title_prefix} - Robot Profile: {profile}", size=14, pad=30)
    plt.legend(loc='upper left', bbox_to_anchor=(1.25, 0.9),
               title="Models", title_fontsize=11, fontsize=9)
    plt.tight_layout()

    safe_profile_name = str(profile).replace(' ', '_').replace('/', '_')
    save_path = os.path.join(plots_dir, f"{file_prefix}_{safe_profile_name}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved '{title_prefix}' radar chart for profile '{profile}' to: {save_path}")
    plt.close()


def plot_radar_charts_combined(df_pivot, df_plot, plots_dir):
    """
    For each robot profile, create radar charts for:
      1. All models (bag models + baselines + RL/MAB models)
      2. baseline_QE vs. the model(s) whose name ends with ' trained'
      3. The bag models without the baselines (and without RL/MAB)
      4. RL/MAB models only
      5. Trained model(s) vs. the RL/MAB models
      6. baseline_QE + trained model(s) + RL/MAB models
    """
    if df_plot.empty:
        print("Warning: No data available for combined radar charts. Skipping.")
        return

    df_full_normalized = normalize_waiting_time(df_pivot)

    df_all_pivot = df_plot.pivot_table(
        index=['robot_profile', 'model'],
        columns='metric',
        values='value'
    ).reset_index()

    df_all_with_norm = df_all_pivot.merge(
        df_full_normalized[['robot_profile', 'model', 'elevators_waited_norm']],
        on=['robot_profile', 'model'],
        how='left'
    )

    profiles = df_all_with_norm['robot_profile'].unique()
    models = list(df_all_with_norm['model'].unique())

    # Consistent styling across all the radar plots. With the RL/MAB models
    # there are >10 models, so use tab20 (tab10 would start repeating colours).
    cmap = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(models))]
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    linestyle_cycle = ['-', '--', '-.', ':']
    model_to_color = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    model_to_style = {
        m: (marker_cycle[i % len(marker_cycle)], linestyle_cycle[i % len(linestyle_cycle)])
        for i, m in enumerate(models)
    }

    # ---- Model subsets ----
    baseline_qe = [m for m in models if m == "baseline_QE"]
    trained = [m for m in models if str(m).endswith(" trained")]
    baseline_vs_trained = baseline_qe + trained
    rl_mab = [m for m in models if m in RL_MAB_MODEL_NAMES]
    # "Trained vs Contested" keeps its original meaning (bag models only,
    # no baselines) - the RL/MAB models get their own charts below.
    no_baselines = [m for m in models if "baseline" not in str(m).lower() and m not in RL_MAB_MODEL_NAMES]
    trained_vs_rl_mab = trained + rl_mab
    baseline_trained_vs_rl_mab = baseline_qe + trained + rl_mab

    if not baseline_qe:
        print("Warning: 'baseline_QE' not found among models.")
    if not trained:
        print("Warning: no model ending with ' trained' found.")
    if not rl_mab:
        print("Warning: no RL/MAB models found in the log - RL/MAB radar charts will be skipped.")

    print(f"\nAll models: {models}")
    print(f"baseline_QE vs trained: {baseline_vs_trained}")
    print(f"Without baselines: {no_baselines}")
    print(f"RL/MAB models: {rl_mab}")

    plots = [
        (models,              "All Models",                  "radar_chart_all_models"),
        (baseline_vs_trained, "Baseline vs Trained",         "radar_chart_baselineQE_vs_trained"),
        (no_baselines,        "Trained vs Contested",   "radar_chart_no_baselines"),
        # RL / MAB / MORL comparisons
        (rl_mab,                     "RL / MAB models",                  "radar_chart_rl_mab"),
        (trained_vs_rl_mab,          "Trained vs RL / MAB models",       "radar_chart_trained_vs_rl_mab"),
        (baseline_trained_vs_rl_mab, "Baseline vs Trained vs RL / MAB",  "radar_chart_baseline_trained_vs_rl_mab"),
    ]

    for profile in profiles:
        profile_data = df_all_with_norm[df_all_with_norm['robot_profile'] == profile]
        for subset, title, prefix in plots:
            if not subset:
                continue
            _draw_radar(profile_data, profile, subset, model_to_color, model_to_style,
                        title, prefix, plots_dir)

    print(f"Created radar charts for {len(profiles)} profiles")

# def plot_radar_charts_combined(df_pivot, df_plot, plots_dir):
#     """
#     Create separate radar charts for each profile using the same format as individual charts.
#     Percentages are shown as-is (0-100), waiting time is normalized to 0-1.
#     Includes ALL models, not just argumentation ones.

#     Fixes vs original:
#     - Models with all-zero metrics now render as a small visible polygon instead of
#       collapsing invisibly to the center point.
#     - Models with identical metric vectors no longer fully occlude one another: each
#       model gets a tiny deterministic angular jitter plus a distinct marker/linestyle,
#       so overlapping shapes remain individually visible and legend-identifiable.
#     """
#     if df_plot.empty:
#         print("Warning: No data available for combined radar charts. Skipping.")
#         return

#     df_full_normalized = normalize_waiting_time(df_pivot)

#     df_all_pivot = df_plot.pivot_table(
#         index=['robot_profile', 'model'],
#         columns='metric',
#         values='value'
#     ).reset_index()

#     df_all_with_norm = df_all_pivot.merge(
#         df_full_normalized[['robot_profile', 'model', 'elevators_waited_norm']],
#         on=['robot_profile', 'model'],
#         how='left'
#     )

#     metric_cols = ['success', 'skipped_queue', 'priority_violations', 'impairment_squishments']

#     for col in metric_cols:
#         min_val = df_all_with_norm[col].min()
#         max_val = df_all_with_norm[col].max()
#         if max_val > min_val:
#             df_all_with_norm[col + '_norm'] = (df_all_with_norm[col] - min_val) / (max_val - min_val)
#         else:
#             df_all_with_norm[col + '_norm'] = 0.5

#     profiles = df_all_with_norm['robot_profile'].unique()
#     models = df_all_with_norm['model'].unique()

#     print(f"\nCreating radar charts with ALL models: {models}")
#     print("Percentages normalized to 0-1 for radar display, waiting time normalized 0-1 (1=best)")

#     metric_names = ['Success', 'Skipped\nQueue', 'Priority\nViolations',
#                      'Impairment\nSquishments', 'Elevators\nWaited']

#     N = len(metric_names)
#     base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

#     colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
#     # Distinct marker + linestyle per model so identical/overlapping traces stay
#     # distinguishable even at a glance in the legend and on the chart.
#     marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
#     linestyle_cycle = ['-', '--', '-.', ':']
#     model_to_style = {
#         model: (marker_cycle[i % len(marker_cycle)], linestyle_cycle[i % len(linestyle_cycle)])
#         for i, model in enumerate(models)
#     }
#     model_to_color = {model: colors[i % len(colors)] for i, model in enumerate(models)}

#     # Radius floor: an all-zero model still draws as a small but visible polygon
#     # instead of an invisible point at the origin.
#     r_floor = 0.03
#     # Small angular jitter (degrees, converted to radians), deterministic per model,
#     # so two models with identical values don't perfectly overlap.
#     max_jitter_deg = 2.5
#     n_models = max(len(models), 1)
#     model_to_jitter = {
#         model: np.deg2rad(-max_jitter_deg + (2 * max_jitter_deg) * (i / max(n_models - 1, 1)))
#         for i, model in enumerate(models)
#     }

#     for profile in profiles:
#         plt.figure(figsize=(9, 8))
#         ax = plt.subplot(111, projection='polar')

#         profile_data = df_all_with_norm[df_all_with_norm['robot_profile'] == profile]

#         for _, row in profile_data.iterrows():
#             model = row['model']
#             raw_values = [
#                 row['success'] / 100.0,
#                 row['skipped_queue'] / 100.0,
#                 row['priority_violations'] / 100.0,
#                 row['impairment_squishments'] / 100.0,
#                 row['elevators_waited_norm'],
#             ]
#             # Apply floor so all-zero (or near-zero) traces are still visible.
#             values = [max(v, r_floor) if pd.notna(v) else r_floor for v in raw_values]
#             values += values[:1]

#             jitter = model_to_jitter[model]
#             angles = (base_angles + jitter).tolist()
#             angles += angles[:1]

#             color = model_to_color[model]
#             marker, linestyle = model_to_style[model]

#             ax.plot(angles, values, marker=marker, linestyle=linestyle,
#                      linewidth=2, markersize=6, label=model, color=color, alpha=0.85)
#             ax.fill(angles, values, alpha=0.08, color=color)

#         # Ticks stay at the true (unjittered) angle positions for readability.
#         ax.set_xticks(base_angles)
#         ax.set_xticklabels(metric_names, size=11)
#         ax.set_ylim(0, 1)
#         ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#         ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
#         ax.grid(True)

#         plt.title(f"Robot Profile: {profile}", size=14, pad=30)
#         plt.legend(loc='upper left', bbox_to_anchor=(1.25, 0.9),
#                    title="Models", title_fontsize=11, fontsize=9)
#         plt.tight_layout()

#         safe_profile_name = str(profile).replace(' ', '_').replace('/', '_')
#         save_path = os.path.join(plots_dir, f"radar_chart_all_models_{safe_profile_name}.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved combined radar chart for profile '{profile}' to: {save_path}")
#         plt.close()

#     print(f"Created {len(profiles)} radar charts with all models")

def get_argumentation_models(models):
    """Filter models to only those with 'argumentation' in the name"""
    return [model for model in models if 'argumentation' in model.lower()]

def analyse(log_path, out_path, plots_dir=None):
    df = pd.read_csv(log_path)
    print(df.head())

    robot_profiles = sorted(df["robot_profile"].unique().tolist())
    models = sorted(df["model"].unique().tolist())
    print(f"\nProfiles found: {robot_profiles}")
    print(f"Models found:   {models}\n")
    
    # Get argumentation models
    argumentation_models = get_argumentation_models(models)
    print(f"Argumentation models found: {argumentation_models}\n")

    metrics = {
        profile: {
            model: {
                "success": [],
                "waiting_time": [],
                "skipped_queue": [],
                "priority_violations": [],
                "impairment_squishments": [],
                "elevators_waited": [],
            }
            for model in models
        }
        for profile in robot_profiles
    }

    for robot_profile in robot_profiles:
        profile_df = df[df["robot_profile"] == robot_profile]

        for model in models:
            model_df = profile_df[profile_df["model"] == model]
            if model_df.empty:
                continue
            number_runs = int(model_df["run"].max())

            for run in range(number_runs + 1):
                actual_run = model_df[model_df["run"] == run].sort_values("iteration")
                if actual_run.empty:
                    continue

                entered_rows = actual_run[actual_run["decision"] == "enter"]
                if len(entered_rows) > 0:
                    row = entered_rows.iloc[0]
                    metrics[robot_profile][model]["success"].append(1)
                    metrics[robot_profile][model]["waiting_time"].append(float(row["robot_time_waiting"]))
                    metrics[robot_profile][model]["elevators_waited"].append(len(actual_run)-1)

                    number_users_before_robot = int(row["number_users_before_robot"])
                    case = int(row["case"]) if "case" in row and not pd.isna(row["case"]) else 0
                    if case == 0 and number_users_before_robot > 0:
                        metrics[robot_profile][model]["skipped_queue"].append(1)
                    elif number_users_before_robot == 0:
                        metrics[robot_profile][model]["skipped_queue"].append(0)

                    if bool(row["people_with_priority"]):
                        metrics[robot_profile][model]["priority_violations"].append(1)
                    else:
                        metrics[robot_profile][model]["priority_violations"].append(0)

                    if bool(row["people_with_impairments"]):
                        metrics[robot_profile][model]["impairment_squishments"].append(1)
                    else:
                        metrics[robot_profile][model]["impairment_squishments"].append(0)
                else:
                    metrics[robot_profile][model]["success"].append(0)

    results = compute_statistics(metrics, robot_profiles, models)

    # Print results nicely

    # Write results to CSV file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "robot_profile", "model",
            "success_pct", "skipped_queue_pct",
            "priority_violations_pct", "impairment_squishments_pct",
            #"waiting_time_mean", 
            "elevators_waited_mean"
        ])
        for profile in robot_profiles:
            for model in models:
                res = results[profile][model]
                writer.writerow([
                    profile, model,
                    res["success"],
                    res["skipped_queue"],
                    res["priority_violations"],
                    res["impairment_squishments"],
                    #res["waiting_time_mean"],
                    res["elevators_waited_mean"]
                ])

    print(f"CSV file saved as '{out_path}'")
    
    # -----------------------------------------------------------------
    # Generate plots (if plots_dir is provided)
    # -----------------------------------------------------------------
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare tidy dataframe for plotting
        df_plot = prepare_dataframe_for_plots(results, robot_profiles, models)
        
        # Create pivot table for normalization
        df_full_pivot = df_plot.pivot_table(
            index=['robot_profile', 'model'], 
            columns='metric', 
            values='value'
        ).reset_index()
        df_full_pivot['profile_model'] = df_full_pivot['robot_profile'] + " | " + df_full_pivot['model']
        
        # Generate all plot types
        print("\nGenerating plots...")
        
        # Faceted bar chart (no normalization needed)
        plot_faceted_barchart(df_plot, plots_dir)
        
        # Heatmap with per-column normalization
        plot_normalized_heatmap(df_full_pivot, plots_dir)

        make_plots(results, robot_profiles, models, plots_dir)
        
        # For radar charts, compute per-column normalization parameters from full dataset
        #norm_params = get_per_column_normalization_params(df_full_pivot)
        
        # Filter for argumentation models
        df_plot_arg = df_plot[df_plot['model'].isin(argumentation_models)]
        
        # Generate individual radar charts only for argumentation models
        # using per-column normalization
        # plot_radar_charts_individual(df_full_pivot, df_plot_arg, plots_dir, 
        #                             argumentation_models)#, norm_params)

        plot_radar_charts_combined(df_full_pivot, df_plot, plots_dir)
        
        print(f"\nAll plots saved to: {plots_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-profile, per-model success/queue/priority/comfort metrics.")
    parser.add_argument("--log", type=str, default="pipeline_outputs/combined_decision_log.csv",
                         help="Path to the combined_decision_log.csv produced by run_full_pipeline.py")
    parser.add_argument("--out", type=str, default="pipeline_outputs/robot_metrics_summary.csv")
    args = parser.parse_args()

    analyse(args.log, args.out, plots_dir=os.path.join(os.path.dirname(args.out), "plots"))