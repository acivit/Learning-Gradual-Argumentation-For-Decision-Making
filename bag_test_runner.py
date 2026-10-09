"""
bag_test_runner.py
-------------------
Simulation/testing runner for a trained `grad.BAG` model, adapted from the
old project's af_experimentation_runner.py.

Key difference vs. the original: this new project has no robot profiles and
no src.argumentation_framework / activate_arguments / settings.py / .dl-file
based AF pipeline. Instead, decisions come straight from the bag you trained
in claude_train.py (grad.BAG + Backprop), via a `decide_fn(robot_information)
-> "enter"/"no enter"` callback built by claude_train.make_bag_decide_fn().

Per-iteration simulation mechanics (doors open -> decide -> if "no enter",
advance the queue -> if elevator not full, decide again -> otherwise wait for
the next elevator) are copied unchanged from af_experimentation_runner.py's
simulate_model_run/_advance_queue - only the AF-specific decide_fn builder
and the profile/RL-model looping were removed, since this runner is
model-agnostic: it just calls whatever decide_fn you give it.

`generate_users` / `robot_timeout` come from elevator_env.py (itself already
a self-contained adaptation of the original project's
utils.generate_environment), so this file has no dependency on the old
project at all.

Usage
-----
    from bag_test_runner import run_tests

    outcomes_df, log_df = run_tests(decide_fn, model_name="argumentation_baseline",
                                     n_test_runs=300)
"""

import copy
import random

import numpy as np
import pandas as pd

from elevator_env import generate_users, robot_timeout

# Long-format per-iteration log columns. "robot_profile" is kept (always
# "default" here) purely so the log is a drop-in fit for
# analyse_experiments_rl.py, which groups by (robot_profile, model).
STATE_COLUMNS = [
    "run", "iteration", "robot_profile", "number_people_in_elevator", "timeout",
    "robot_time_waiting", "next_time_over_timeout", "people_with_impairments",
    "number_users_before_robot", "people_with_priority", "users_before_robot",
    "elevator_capacity", "case", "waiting_users_did_not_enter",
]
DF_COLUMNS = STATE_COLUMNS + ["model", "decision"]


def _advance_queue(robot_information, elevator_occupancy, users_before_robot,
                    elevator_capacity, elevator_non_stretched_capacity):
    """Unchanged from af_experimentation_runner.py: users who fit without
    stretching leave the queue first, then users who don't mind stretching,
    until either the elevator is full or the queue is empty. Returns
    (elevator_occupancy, users_before_robot, elevator_full)."""
    elevator_full = False
    number_people_fit_in_elevator = elevator_capacity - elevator_occupancy
    number_people_fit_non_stretched = elevator_non_stretched_capacity - elevator_occupancy

    while number_people_fit_non_stretched > 0 and users_before_robot:
        users_before_robot = users_before_robot[1:]
        users_roles = [u["role"] for u in users_before_robot]
        elevator_occupancy += 1
        number_people_fit_non_stretched = elevator_non_stretched_capacity - elevator_occupancy
        number_people_fit_in_elevator = elevator_capacity - elevator_occupancy
        robot_information["users_before_robot"] = users_before_robot
        robot_information["number_users_before_robot"] = len(users_before_robot)
        robot_information["people_with_priority"] = "priority" in users_roles

    if number_people_fit_in_elevator > 0:
        for user in list(users_before_robot):
            if user["does_not_care_about_stretching"]:
                elevator_occupancy += 1
                users_before_robot.remove(user)
                users_roles = [u["role"] for u in users_before_robot]
                robot_information["users_before_robot"] = users_before_robot
                robot_information["number_users_before_robot"] = len(users_before_robot)
                robot_information["people_with_priority"] = "priority" in users_roles
                if elevator_occupancy >= elevator_capacity:
                    elevator_full = True
                    break

    return elevator_occupancy, users_before_robot, elevator_full


def simulate_model_run(
    decide_fn,
    robot_information,
    elevator_capacity=10,
    elevator_mean_occupancy=5,
    elevator_std_occupancy=2,
    elevator_non_stretched_capacity=7,
    wait_time_shape=60,
    wait_time_scale=1,
    max_iterations=1000,
    log_rows=None,
    model_name=None,
):
    """One independent trajectory through the per-iteration mechanics
    (unchanged from af_experimentation_runner.py), starting from
    `robot_information` (already seeded with run/timeout/queue). Stops as
    soon as `decide_fn` returns "enter", or once robot_time_waiting reaches
    the timeout."""
    users_before_robot = robot_information["users_before_robot"]
    robot_time_waiting = robot_information["robot_time_waiting"]
    timeout = robot_information["timeout"]
    iteration = 0
    entered = False
    timed_out = False

    while True:
        iteration += 1
        if iteration > max_iterations:
            timed_out = True  # safety valve against pathological non-terminating cases
            break

        robot_information["iteration"] = iteration
        robot_information["case"] = "0"
        robot_information["robot_time_waiting"] = robot_time_waiting

        next_elevator_time = np.random.gamma(wait_time_shape, wait_time_scale)
        robot_information["next_elevator_time"] = robot_time_waiting + next_elevator_time
        robot_information["next_time_over_timeout"] = robot_information["next_elevator_time"] > timeout
        robot_information["waiting_users_did_not_enter"] = False

        elevator_occupancy = np.random.normal(elevator_mean_occupancy, elevator_std_occupancy)
        elevator_occupancy = int(round(elevator_occupancy))
        elevator_occupancy = max(
            elevator_non_stretched_capacity - len(users_before_robot),
            min(elevator_occupancy, elevator_capacity - 1),
        )
        people_with_impairments = random.choice([True, False])
        robot_information["number_people_in_elevator"] = elevator_occupancy
        robot_information["people_with_impairments"] = people_with_impairments

        decision = decide_fn(robot_information)
        if log_rows is not None:
            log_rows.append({**{k: robot_information.get(k) for k in STATE_COLUMNS}, "model": model_name, "decision": decision})

        if decision == "enter":
            entered = True
            break

        elevator_full = False
        if len(users_before_robot) > 0:
            elevator_occupancy, users_before_robot, elevator_full = _advance_queue(
                robot_information, elevator_occupancy, users_before_robot,
                elevator_capacity, elevator_non_stretched_capacity,
            )

            if not elevator_full:
                robot_information["number_people_in_elevator"] = elevator_occupancy
                robot_information["waiting_users_did_not_enter"] = True
                decision2 = decide_fn(robot_information)
                if log_rows is not None:
                    log_rows.append({**{k: robot_information.get(k) for k in STATE_COLUMNS}, "model": model_name, "decision": decision2})
                if decision2 == "enter":
                    entered = True
                    break
                robot_time_waiting += next_elevator_time
            else:
                robot_time_waiting += next_elevator_time
        else:
            robot_time_waiting += next_elevator_time

        robot_information["robot_time_waiting"] = robot_time_waiting
        if robot_time_waiting >= timeout:
            timed_out = True
            break

    return {
        "entered": entered,
        "timed_out": timed_out,
        "iterations": iteration,
        "final_robot_time_waiting": robot_time_waiting,
        "timeout": timeout,
    }


def run_tests(
    decide_fn,
    model_name,
    n_test_runs=300,
    robot_profile="default",
    min_timeout=100,
    max_timeout=250,
    elevator_capacity=10,
    elevator_non_stretched_capacity=7,
    elevator_mean_occupancy=5,
    elevator_std_occupancy=2,
    wait_time_shape=60,
    wait_time_scale=1,
    max_users_waiting=10,
    max_iterations=1000,
    keep_iteration_log=True,
):
    """Single-model convenience wrapper around run_tests_all_models() below,
    for when you only care about one model. NOTE: if you're comparing
    several models, call run_tests_all_models() directly instead of calling
    this once per model - each call to this function generates its own
    independent random scenarios, so run i for one model and run i for
    another model would NOT be the same starting situation, which makes the
    per-run comparison unfair. See run_tests_all_models()'s docstring."""
    outcomes_df, log_df = run_tests_all_models(
        {model_name: decide_fn}, n_test_runs=n_test_runs, robot_profile=robot_profile,
        min_timeout=min_timeout, max_timeout=max_timeout, elevator_capacity=elevator_capacity,
        elevator_non_stretched_capacity=elevator_non_stretched_capacity,
        elevator_mean_occupancy=elevator_mean_occupancy, elevator_std_occupancy=elevator_std_occupancy,
        wait_time_shape=wait_time_shape, wait_time_scale=wait_time_scale,
        max_users_waiting=max_users_waiting, max_iterations=max_iterations,
        keep_iteration_log=keep_iteration_log,
    )
    return outcomes_df, log_df


def run_tests_all_models(
    decide_fns,
    n_test_runs=300,
    robot_profile="default",
    min_timeout=100,
    max_timeout=250,
    elevator_capacity=10,
    elevator_non_stretched_capacity=7,
    elevator_mean_occupancy=5,
    elevator_std_occupancy=2,
    wait_time_shape=60,
    wait_time_scale=1,
    max_users_waiting=10,
    max_iterations=1000,
    keep_iteration_log=True,
):
    """Fair multi-model comparison, mirroring the original project's
    af_experimentation_runner.run_tests_for_profile(): for each of
    n_test_runs, the STARTING scenario (timeout, queue composition, robot's
    position in the queue) is generated ONCE and shared by every model in
    decide_fns. Each model then gets its own deep copy of that shared
    starting point and is simulated forward independently via
    simulate_model_run() - so if a model decides "enter", ITS run stops
    there, while the others keep going with their own subsequent random
    draws (elevator arrivals, queue state) until they enter or time out.
    Only the starting point is shared; what happens after each model's first
    decision is independent, exactly like the original.

    decide_fns: {model_name: decide_fn} - build these with
    claude_train.make_bag_decide_fn() for each trained/ablated bag variant.

    Returns (outcomes_df, log_df):
      outcomes_df - one row per (run, model): entered, timed_out, iterations,
                    final_robot_time_waiting, timeout
      log_df      - long-format per-iteration-per-model rows (None if
                    keep_iteration_log=False)
    """
    outcome_rows = []
    log_rows = [] if keep_iteration_log else None

    for i in range(n_test_runs):
        timeout = robot_timeout(min_timeout=min_timeout, max_timeout=max_timeout)
        users, robot_position = generate_users(max_users_waiting)
        base_users_before_robot = [u for u in users if u["position_in_queue"] < robot_position]
        base_roles = [u["role"] for u in base_users_before_robot]

        base_info = {
            "run": i,
            "robot_profile": robot_profile,
            "robot_time_waiting": 0,
            "timeout": timeout,
            "number_users_before_robot": len(base_users_before_robot),
            "people_with_priority": "priority" in base_roles,
            "elevator_capacity": elevator_capacity,
            "elevator_non_stretched_capacity": elevator_non_stretched_capacity,
        }

        for model_name, decide_fn in decide_fns.items():
            robot_information = copy.deepcopy(base_info)
            robot_information["users_before_robot"] = copy.deepcopy(base_users_before_robot)

            outcome = simulate_model_run(
                decide_fn, robot_information,
                elevator_capacity=elevator_capacity,
                elevator_mean_occupancy=elevator_mean_occupancy,
                elevator_std_occupancy=elevator_std_occupancy,
                elevator_non_stretched_capacity=elevator_non_stretched_capacity,
                wait_time_shape=wait_time_shape, wait_time_scale=wait_time_scale,
                max_iterations=max_iterations,
                log_rows=log_rows, model_name=model_name,
            )
            outcome_rows.append({"run": i, "robot_profile": robot_profile, "model": model_name, **outcome})

    outcomes_df = pd.DataFrame(outcome_rows)
    log_df = pd.DataFrame(log_rows, columns=DF_COLUMNS) if keep_iteration_log else None
    return outcomes_df, log_df