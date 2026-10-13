"""
rl_mab_models.py
----------------
Glue between the RL / MAB / MORL code (simple_rl.py, multi_armed_bandit.py,
multi_objective_rl.py, elevator_env.py) and claude_train.py.

What it does
------------
1. train_rl_mab_models(profile)  - trains the SAME six models that
   run_all_experiments.py trained, for one robot profile:
       tabular_q, dqn, contextual_bandit, neural_contextual_bandit,
       vector_q_morl, multihead_dqn_morl
   (the stateless bandit is deliberately left out, as in the old pipeline).

2. train_all_profiles(profiles)  - does (1) once per profile, up front, and
   saves a small training summary CSV + learning-curve plots. RL training
   doesn't depend on the .dl file, so claude_train.py trains once per
   profile and re-uses the result for every .dl file.

3. make_rl_mab_decide_fns(...)   - wraps each trained model as a
   decide_fn(robot_information) -> "enter" / "no enter", the exact interface
   bag_test_runner.run_tests_all_models() already uses for the bag models.
   That's what puts the RL/MAB models into combined_decision_log.csv and
   therefore into analyse_experiments_rl.analyse() and the radar charts.

The profile names must be keys of simple_rl.IMPORTANCES_TEST
(complacent, time-efficient, comfort-caring, comfort-blind), which is also
what claude_train.py's DATASETS use.
"""

import os
import random
import time

import numpy as np
import pandas as pd

from elevator_env import ElevatorEnv, ElevatorState
from baseline_models.simple_rl import IMPORTANCES_TEST, train_tabular, train_dqn, moving_average
from baseline_models.multi_armed_bandit import train_contextual_bandit, train_neural_bandit
from baseline_models.multi_objective_rl import weights_from_importances, train_vector_q, train_multihead_dqn

# NOTE: keep in sync with RL_MAB_MODEL_NAMES in analyse_experiments_rl.py
RL_MAB_MODEL_NAMES = [
    "tabular_q", "dqn", "contextual_bandit", "neural_contextual_bandit",
    "vector_q_morl", "multihead_dqn_morl",
]

# Used for the model->category column in the training summary.
MODEL_CATEGORY = {
    "tabular_q": "single_objective_rl", "dqn": "single_objective_rl",
    "contextual_bandit": "bandit", "neural_contextual_bandit": "bandit",
    "vector_q_morl": "multi_objective_rl", "multihead_dqn_morl": "multi_objective_rl",
}


# ---------------------------------------------------------------------- #
# Greedy action for any model type, from an ElevatorState
# ---------------------------------------------------------------------- #
def greedy_action(model_name, models, weights, state):
    """0 = wait, 1 = enter. Identical to the act_fns in run_all_experiments.py."""
    model = models[model_name]
    if model_name == "tabular_q":
        return model.act(state.discrete(), greedy=True)
    if model_name == "dqn":
        return model.act(state.vector(), greedy=True)
    if model_name == "contextual_bandit":
        ctx = state.discrete()
        if ctx in model.values and np.any(model.counts[ctx] > 0):
            return int(np.argmax(model.values[ctx]))
        return 0  # unseen context -> wait
    if model_name == "neural_contextual_bandit":
        return int(np.argmax(model.net.predict(state.vector())))
    if model_name == "vector_q_morl":
        return model.act(state.discrete(), weights, greedy=True)
    if model_name == "multihead_dqn_morl":
        return model.act(state.vector(), weights, greedy=True)
    raise ValueError(f"unknown model_name {model_name!r}")


# ---------------------------------------------------------------------- #
# Evaluation (full episodes in ElevatorEnv, same procedure for every model)
# ---------------------------------------------------------------------- #
def rollout_policy(env, act_fn, n_episodes=500, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    scalar_sum, vec_sum = 0.0, np.zeros(3, dtype=np.float64)
    enter_count = timeout_count = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = entered = False
        while not done:
            state, reward, r_vec, done, info = env.step(act_fn(state))
            scalar_sum += reward
            vec_sum += r_vec
            entered = entered or info["entered"]
            timeout_count += int(info["timed_out"])
        enter_count += int(entered)
    n = n_episodes
    return {
        "mean_return": scalar_sum / n,
        "return_urgency": vec_sum[0] / n,
        "return_comfort": vec_sum[1] / n,
        "return_queue": vec_sum[2] / n,
        "enter_rate": enter_count / n,
        "timeout_rate": timeout_count / n,
    }


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train_rl_mab_models(profile_name, importances=None, episodes=3000, eval_episodes=500, seed=0, verbose=True):
    """Trains all six models for one profile.

    Returns (models, weights, info):
      models  - {model_name: trained agent / bandit}
      weights - MORL scalarisation weights for this profile (np.ndarray, 3)
      info    - {"curves": {model_name: [scalar reward per episode/pull]},
                 "rows":   [one summary dict per model]}
    """
    if importances is None:
        if profile_name not in IMPORTANCES_TEST:
            raise ValueError(
                f"No importances for profile {profile_name!r}. Known profiles: "
                f"{list(IMPORTANCES_TEST)}. Add it to simple_rl.IMPORTANCES_TEST "
                "or pass `importances=` explicitly."
            )
        importances = IMPORTANCES_TEST[profile_name]

    env = ElevatorEnv(importances=importances)
    weights = weights_from_importances(importances)
    models, curves, rows = {}, {}, []

    def finish(name, agent, hist, t0, act_fn, extra=""):
        dt = time.time() - t0
        models[name] = agent
        curves[name] = np.asarray(hist, dtype=np.float64) if np.ndim(hist) == 1 else np.asarray(hist).sum(axis=1)
        metrics = rollout_policy(env, act_fn, eval_episodes)
        rows.append({"profile": profile_name, "model": name, "category": MODEL_CATEGORY[name],
                     "train_episodes_or_pulls": episodes, "eval_episodes": eval_episodes,
                     **metrics, "train_time_sec": round(dt, 3), "extra_info": extra})
        if verbose:
            print(f"    {name:26s} {dt:6.1f}s  eval mean_return={metrics['mean_return']:+.3f}  "
                  f"enter_rate={metrics['enter_rate']:.2f}")

    def act(name):
        return lambda state: greedy_action(name, models, weights, state)

    if verbose:
        print(f"  [RL/MAB] training profile '{profile_name}' ({episodes} episodes/pulls per model)")

    t0 = time.time()
    agent, hist = train_tabular(env, episodes=episodes, seed=seed)
    models["tabular_q"] = agent
    finish("tabular_q", agent, hist, t0, act("tabular_q"), f"states={len(agent.Q)}")

    t0 = time.time()
    agent, hist = train_dqn(env, episodes=episodes, seed=seed)
    models["dqn"] = agent
    finish("dqn", agent, hist, t0, act("dqn"))

    t0 = time.time()
    agent, hist = train_contextual_bandit(env, pulls=episodes, seed=seed)
    models["contextual_bandit"] = agent
    finish("contextual_bandit", agent, hist, t0, act("contextual_bandit"), f"contexts={len(agent.values)}")

    t0 = time.time()
    agent, hist = train_neural_bandit(env, pulls=episodes, seed=seed)
    models["neural_contextual_bandit"] = agent
    finish("neural_contextual_bandit", agent, hist, t0, act("neural_contextual_bandit"))

    t0 = time.time()
    agent, hist = train_vector_q(env, weights, episodes=episodes, seed=seed)
    models["vector_q_morl"] = agent
    finish("vector_q_morl", agent, hist, t0, act("vector_q_morl"),
           f"states={len(agent.Q)} weights={np.round(weights, 3).tolist()}")

    t0 = time.time()
    agent, hist = train_multihead_dqn(env, weights, episodes=episodes, seed=seed)
    models["multihead_dqn_morl"] = agent
    finish("multihead_dqn_morl", agent, hist, t0, act("multihead_dqn_morl"),
           f"weights={np.round(weights, 3).tolist()}")

    return models, weights, {"curves": curves, "rows": rows}


def _save_learning_curves(curves_by_profile, plots_dir):
    import matplotlib.pyplot as plt  # local import: keeps this module importable without a display

    os.makedirs(plots_dir, exist_ok=True)
    for profile, curves in curves_by_profile.items():
        plt.figure(figsize=(9, 5))
        for name, hist in curves.items():
            w = int(max(5, min(100, len(hist) * 0.05)))
            plt.plot(moving_average(list(hist), w=w), label=name)
        plt.title(f"RL/MAB reward during training - profile: {profile}")
        plt.xlabel("episode / pull (smoothed)")
        plt.ylabel("reward")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"reward_evolution_{profile}.png"), dpi=130)
        plt.close()


def train_all_profiles(profiles, episodes=3000, eval_episodes=500, seed=0, outdir="results/rl_mab", verbose=True):
    """Trains the six RL/MAB models once for each profile in `profiles`.

    Returns {profile: (models, weights)}. Profiles with no entry in
    simple_rl.IMPORTANCES_TEST are skipped with a warning (they'll just be
    absent from the radar charts). Also writes
    <outdir>/rl_mab_training_summary.csv and <outdir>/plots/reward_evolution_<profile>.png
    """
    os.makedirs(outdir, exist_ok=True)
    trained, all_rows, all_curves = {}, [], {}
    for profile in profiles:
        if profile not in IMPORTANCES_TEST:
            print(f"[rl_mab_models] WARNING: no importances for profile {profile!r}; "
                  "skipping RL/MAB training for it.")
            continue
        models, weights, info = train_rl_mab_models(profile, episodes=episodes, eval_episodes=eval_episodes,
                                                    seed=seed, verbose=verbose)
        trained[profile] = (models, weights)
        all_rows += info["rows"]
        all_curves[profile] = info["curves"]

    if all_rows:
        path = os.path.join(outdir, "rl_mab_training_summary.csv")
        pd.DataFrame(all_rows).to_csv(path, index=False)
        _save_learning_curves(all_curves, os.path.join(outdir, "plots"))
        print(f"[rl_mab_models] Saved training summary -> {path}")
    return trained


# ---------------------------------------------------------------------- #
# Hooking the trained models into the simulation (decide_fn interface)
# ---------------------------------------------------------------------- #
def robot_information_to_state(robot_information, max_users_waiting=10, task_urgent=False,
                               default_non_stretched_capacity=7):
    """robot_information (the dict the bag models read) -> ElevatorState.

    The simulation has no notion of task urgency, so `task_urgent` is a fixed
    value here (default False, same as af_experimentation_runner.py did).
    Fields that may be absent in robot_information fall back to sensible
    defaults instead of raising.
    """
    ri = robot_information
    return ElevatorState(
        task_urgent=task_urgent,
        timeout=float(ri["timeout"]),
        robot_time_waiting=float(ri["robot_time_waiting"]),
        number_users_before_robot=int(ri["number_users_before_robot"]),
        people_with_priority=bool(ri["people_with_priority"]),
        elevator_capacity=int(ri["elevator_capacity"]),
        elevator_non_stretched_capacity=int(ri.get("elevator_non_stretched_capacity", default_non_stretched_capacity)),
        max_users_waiting=max_users_waiting,
        number_people_in_elevator=int(ri["number_people_in_elevator"]),
        people_with_impairments=bool(ri["people_with_impairments"]),
        next_time_over_timeout=bool(ri["next_time_over_timeout"]),
        waiting_users_did_not_enter=bool(ri.get("waiting_users_did_not_enter", False)),
    )


def make_rl_mab_decide_fns(models, weights, task_urgent=False, max_users_waiting=10):
    """{model_name: decide_fn(robot_information) -> 'enter' / 'no enter'} for
    every trained model. Forward pass only - nothing is learned here, and
    robot_information is never modified."""
    def build(name):
        def decide(robot_information):
            state = robot_information_to_state(robot_information, max_users_waiting=max_users_waiting,
                                               task_urgent=task_urgent)
            return "enter" if greedy_action(name, models, weights, state) == 1 else "no enter"
        return decide

    return {name: build(name) for name in models}
