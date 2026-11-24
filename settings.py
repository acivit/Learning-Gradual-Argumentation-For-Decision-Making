EDGE_DISCOUNT_TARGET = "enter"
EDGE_DISCOUNT_VARIANTS = [
    {
        "model_name": "Edge Weight: (x0.5 comfort, x0.5 queue, x1.25 urgency)",
        "source_factors": {
            "inside_comfort": 0.5, # 0.5,
            "breaking_queue": 0.5, # 0.5,
            "robot_urgency": 1.25, # 1.1,
        },
        "color": "tab:red"
    },
    {
        "model_name": "Edge Weight: (x1.25 comfort, x0.5 queue, x0.5 urgency)",
        "source_factors": {
            "inside_comfort": 1.25, # 1.5,
            "breaking_queue": 0.5, # 0.9,
            "robot_urgency": 0.5, # 0.9,
        },
        "color": "tab:pink"
    },
    {
        "model_name": "Edge Weight: (x1 comfort, x0 queue, x1 urgency)",
        "source_factors": {
            "inside_comfort": 1,
            "breaking_queue": 0,
            "robot_urgency": 1,
        },
        "color": "tab:brown"
    },
]

NORMALISED_MODEL_NAME = "Normalised Edges" # removed below because we don't learn

ABLATION_VARIANTS = [
    {
        "model_name": "Base Scores: (0.75 for norms, 0.25 for urgency)",
        "fixed_arguments": {
            "inside_comfort": 0.75,
            "robot_urgency": 0.25,
            "breaking_queue": 0.75,
        },
        "color": "tab:orange"
    },
    {
        "model_name": "Base Scores: (0.25 for norms, 0.75 for urgency)",
        "fixed_arguments": {
            "inside_comfort": 0.25,
            "robot_urgency": 0.75,
            "breaking_queue": 0.75,
        },
        "color": "tab:blue"
    },
]
BASELINE_MODEL_NAME = " trained"#"comfort-blind (as the baseline)"


IMPORTANCES_TEST = {
    "complacent": {
        "robot_urgency": 0.5, "percentage_stretchment": 0.75, "people_impairment": 0.75,
        "next_time_over_timeout": 0.25, "still_time_for_timeout": 0.25, "percentage_timeout": 0.25,
        "breaking_queue": 0.5, "each_user_in_queue": 1, "important_role_in_queue": 0.75,
        "inside_comfort": 0.5, "waiting_users_did_not_enter": 0.25,
    },
    "time-efficient": {
        "robot_urgency": 0.5, "percentage_stretchment": 0.25, "people_impairment": 0.25,
        "next_time_over_timeout": 1, "still_time_for_timeout": 0.25, "percentage_timeout": 1,
        "breaking_queue": 0.25, "each_user_in_queue": 0.25, "important_role_in_queue": 0.25,
        "inside_comfort": 0.5, "waiting_users_did_not_enter": 0.25,
    },
    "comfort-caring": {
        "robot_urgency": 0.5, "percentage_stretchment": 1, "people_impairment": 1,
        "next_time_over_timeout": 0.25, "still_time_for_timeout": 0.25, "percentage_timeout": 0.25,
        "breaking_queue": 0.5, "each_user_in_queue": 0.25, "important_role_in_queue": 0.25,
        "inside_comfort": 0.5, "waiting_users_did_not_enter": 0.25,
    },
    "comfort-blind": {
        "robot_urgency": 0.5, "percentage_stretchment": 0.25, "people_impairment": 0.25,
        "next_time_over_timeout": 0.75, "still_time_for_timeout": 0.25, "percentage_timeout": 0.75,
        "breaking_queue": 0.5, "each_user_in_queue": 1, "important_role_in_queue": 0.75,
        "inside_comfort": 0.5, "waiting_users_did_not_enter": 0.25,
    },
}

interpretable_arguments = [
    "enter",
    "robot_urgency",
    "breaking_queue",
    "inside_comfort",
    "percentage_stretchment",
    "people_impairment",
    "next_time_over_timeout",
    "percentage_timeout",
    "important_role_in_queue",
    "waiting_users_did_not_enter",
    "amount_users_queue",
]

frozen_nodes = [
    "enter",
    "robot_urgency",
    "breaking_queue",
    "inside_comfort",
]

# ---------------------------------------------------------------------- #
# Exhaustive edge-weight sweep (post-training, no retraining)
# ---------------------------------------------------------------------- #
# Every variant is a {source: factor} dict applied to the edges source ->
# EDGE_DISCOUNT_TARGET of the trained bag (same mechanism as
# EDGE_DISCOUNT_VARIANTS above). Everything starts at EDGE_SWEEP_BASE; the
# sources listed in EDGE_SWEEP_VALUES are then varied.
#
# EDGE_SWEEP_MODE:
#   "one_at_a_time": vary each listed source on its own, the others stay at
#                    their EDGE_SWEEP_BASE value (n_values per source).
#   "grid":          full cartesian product of all listed sources' values.
EDGE_SWEEP_ENABLED = True
EDGE_SWEEP_MODE = "one_at_a_time"
EDGE_SWEEP_BASE = {
    "inside_comfort": 1,
    "breaking_queue": 1,
    "robot_urgency": 1,
}
EDGE_SWEEP_VALUES = {
    # 0, 0.25, ..., 2 and then 3
    "robot_urgency": [round(0.25 * i, 2) for i in range(0, 9)] + [3],
    # To sweep the other edges too, add e.g.:
    # "inside_comfort": [round(0.25 * i, 2) for i in range(0, 9)] + [3],
    # "breaking_queue": [round(0.25 * i, 2) for i in range(0, 9)] + [3],
}
# Also simulate the RL/MAB/MORL models in the sweep run, so the CSV has them
# as a reference row. Off by default (slower).
EDGE_SWEEP_INCLUDE_RL_MAB = False