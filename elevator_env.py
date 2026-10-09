"""
elevator_env.py
----------------
Standalone RL environment for the "should the robot enter the elevator?"
decision problem described in generate_environment.py / experimentation.py.

This does NOT depend on the argumentation-framework code (ArgumentationFramework,
activate_arguments, ElevatorSimulation, the .dl case file) since those files
were not provided. Instead, the same `importances` dictionaries used in
experimentation.py's `importances_test` are used directly as the weights of a
hand-written reward function, so you can train agents right now with e.g.
importances_test["complacent"], importances_test["time-efficient"], etc.

If/when you plug the real argumentation-framework back in, you only need to
replace `ElevatorEnv._resolve_reward()` with a call to
`get_arguments_and_decision(robot_information, importances)` and translate its
decision ("enter" / "no enter" / "tie") plus your own notion of "how good was
that decision" into the reward. Everything else (state generation, episode
loop, action space) can stay the same.

Action space
------------
0 = WAIT  (robot stays in queue, another elevator cycle happens)
1 = ENTER (robot attempts to get in now; episode ends)

State
-----
A dict of interpretable fields plus a `.vector()` helper that turns the state
into a fixed-length numeric numpy array (for the neural-net agents) and a
`.discrete()` helper that bins it into a small tuple (for tabular agents).

Reward
------
Every step returns:
  - `reward`        : scalar reward, already weighted by `importances`
                       (sum of the three objectives below) -> used by the
                       tabular/neural single-objective agents and the bandit.
  - `reward_vector` : (r_urgency, r_comfort, r_queue) -> used by the
                       multi-objective RL script.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Functions copied/adapted from generate_environment.py so this file has no
# import-path dependency on the rest of your project. Behaviour is identical.
# ----------------------------------------------------------------------


def robot_task_importance():
    return random.choice(["high", "low"])


def task_urgency():
    return random.choice(["urgent", "not_urgent"])


def robot_timeout(min_timeout=100, max_timeout=250):
    return random.uniform(min_timeout, max_timeout)


def user_waiting():
    urgency_persona = random.choice(["urgent", "not_urgent"])
    does_not_care_about_stretching = random.choice([True, False])
    role_persona = random.choice(["priority", "passager"])
    return urgency_persona, role_persona, does_not_care_about_stretching


def generate_users(max_users=10):
    num_users = np.random.randint(0, max_users + 1)
    users = []
    for _ in range(num_users):
        urgency_persona, role_persona, does_not_care_about_stretching = user_waiting()
        users.append(
            {
                "urgency_persona": urgency_persona,
                "role": role_persona,
                "does_not_care_about_stretching": does_not_care_about_stretching,
            }
        )
    possible_positions = list(range(num_users + 1))
    robot_position_in_queue = random.randint(0, num_users)
    possible_positions.remove(robot_position_in_queue)
    for i, user in enumerate(users):
        user_position = random.choice(possible_positions)
        users[i]["position_in_queue"] = user_position
        possible_positions.remove(user_position)
    return users, robot_position_in_queue


# ----------------------------------------------------------------------
# Default importances (identical structure to experimentation.py's
# `importances_test`, so you can import that dict directly instead).
# ----------------------------------------------------------------------

DEFAULT_IMPORTANCES: Dict[str, float] = {
    "robot_urgency": 0.5,
    "percentage_stretchment": 0.5,
    "people_impairment": 0.5,
    "next_time_over_timeout": 0.5,
    "still_time_for_timeout": 0.25,
    "percentage_timeout": 0.5,
    "breaking_queue": 0.5,
    "each_user_in_queue": 0.5,
    "important_role_in_queue": 0.5,
    "inside_comfort": 0.5,
    "waiting_users_did_not_enter": 0.25,
}


@dataclass
class EnvConfig:
    elevator_capacity: int = 10
    elevator_non_stretched_capacity: int = 7
    elevator_mean_occupancy: int = 5
    elevator_std_occupancy: int = 2
    max_users_waiting: int = 10
    min_timeout: float = 100.0
    max_timeout: float = 250.0
    wait_time_shape: float = 60.0  # gamma alpha, matches experimentation.py
    wait_time_scale: float = 1.0  # gamma beta
    max_steps: int = 40  # safety cap on iterations per episode


class ElevatorEnv:
    """A small episodic environment: each episode is one "robot waiting for
    the elevator" scenario, ending either when the robot enters (action=1)
    or when its timeout expires while it keeps waiting (action=0)."""

    ACTIONS = (0, 1)  # 0 = wait, 1 = enter

    def __init__(self, importances: Dict[str, float] = None, config: EnvConfig = None):
        self.importances = importances or DEFAULT_IMPORTANCES
        self.cfg = config or EnvConfig()
        self.state = None
        self._step_count = 0

    # -- episode lifecycle -------------------------------------------------

    def set_importances(self, importances: Dict[str, float]):
        """Swap the reward weighting (e.g. switch robot profile) without
        rebuilding the environment."""
        self.importances = importances

    def reset(self) -> "ElevatorState":
        cfg = self.cfg
        users, robot_position = generate_users(cfg.max_users_waiting)
        users_before_robot = [u for u in users if u["position_in_queue"] < robot_position]

        self.state = ElevatorState(
            task_urgent=(task_urgency() == "urgent"),
            timeout=robot_timeout(cfg.min_timeout, cfg.max_timeout),
            robot_time_waiting=0.0,
            number_users_before_robot=len(users_before_robot),
            people_with_priority=any(u["role"] == "priority" for u in users_before_robot),
            elevator_capacity=cfg.elevator_capacity,
            elevator_non_stretched_capacity=cfg.elevator_non_stretched_capacity,
            max_users_waiting=cfg.max_users_waiting,
        )
        self._draw_elevator_arrival()
        self._step_count = 0
        return self.state

    def _draw_elevator_arrival(self):
        """Simulate the doors opening: how many people already inside, and
        whether anyone with reduced mobility is present."""
        cfg = self.cfg
        occ = np.random.normal(cfg.elevator_mean_occupancy, cfg.elevator_std_occupancy)
        occ = int(round(occ))
        occ = max(
            cfg.elevator_non_stretched_capacity - self.state.number_users_before_robot,
            min(occ, cfg.elevator_capacity - 1),
        )
        occ = max(0, occ)
        self.state.number_people_in_elevator = occ
        self.state.people_with_impairments = random.choice([True, False])
        self.state.next_time_over_timeout = (
            self.state.robot_time_waiting + self._peek_next_wait() > self.state.timeout
        )

    def _peek_next_wait(self) -> float:
        return np.random.gamma(self.cfg.wait_time_shape, self.cfg.wait_time_scale)

    # -- reward -------------------------------------------------------------

    def _objective_rewards(self, action: int) -> Tuple[float, float, float]:
        """Returns (r_urgency, r_comfort, r_queue) for the given action in the
        *current* state, weighted by self.importances. This is the function
        to replace with the real argumentation-framework decision if/when
        that code is available."""
        s = self.state
        imp = self.importances

        pct_timeout = min(1.0, s.robot_time_waiting / max(s.timeout, 1e-6))
        #urgency_factor = 1.0 if s.task_urgent else 0.3
        urgency_pressure = (
            #0.5 * urgency_factor
            + 0.6 * pct_timeout # 0.3
            + 0.4 * (1.0 if s.next_time_over_timeout else 0.0) # 0.2
        )

        would_need_stretch = (
            s.number_people_in_elevator + 1 > s.elevator_non_stretched_capacity
        )
        crowding = s.number_people_in_elevator / max(s.elevator_capacity, 1)
        comfort_cost = (
            imp["inside_comfort"] * crowding
            + imp["percentage_stretchment"] * (1.0 if would_need_stretch else 0.0)
            + imp["people_impairment"]
            * (1.0 if (s.people_with_impairments and would_need_stretch) else 0.0)
        )

        queue_pressure = (
            imp["breaking_queue"] * (1.0 if s.number_users_before_robot > 0 else 0.0)
            + imp["each_user_in_queue"]
            * (s.number_users_before_robot / max(s.max_users_waiting, 1))
            + imp["important_role_in_queue"] * (1.0 if s.people_with_priority else 0.0)
        )

        stuck_penalty = imp["waiting_users_did_not_enter"] * (
            1.0 if getattr(s, "waiting_users_did_not_enter", False) else 0.0
        )
        still_time_bonus = imp["still_time_for_timeout"] * (1.0 - pct_timeout)

        if action == 1:  # ENTER
            r_urgency = imp["robot_urgency"] * urgency_pressure + imp["percentage_timeout"] * pct_timeout
            r_comfort = -comfort_cost
            r_queue = -queue_pressure
        else:  # WAIT
            r_urgency = -imp["robot_urgency"] * urgency_pressure - stuck_penalty + 0.3 * still_time_bonus
            r_comfort = 0.3 * comfort_cost
            r_queue = 0.5 * queue_pressure

        return r_urgency, r_comfort, r_queue

    # -- stepping -------------------------------------------------------------

    def step(self, action: int):
        assert action in self.ACTIONS
        s = self.state
        self._step_count += 1

        r_urgency, r_comfort, r_queue = self._objective_rewards(action)
        reward_vector = np.array([r_urgency, r_comfort, r_queue], dtype=np.float32)
        reward = float(reward_vector.sum())

        info = {"entered": action == 1, "timed_out": False}

        if action == 1:
            done = True
        else:
            wait = self._peek_next_wait()
            s.robot_time_waiting += wait
            s.waiting_users_did_not_enter = True
            timed_out = s.robot_time_waiting >= s.timeout or self._step_count >= self.cfg.max_steps
            if timed_out:
                # Failing to ever get in is bad for every objective.
                reward_vector -= np.array([1.0, 0.0, 0.0], dtype=np.float32)
                reward -= 1.0
                info["timed_out"] = True
                done = True
            else:
                self._draw_elevator_arrival()
                done = False

        return s, reward, reward_vector, done, info


@dataclass
class ElevatorState:
    task_urgent: bool
    timeout: float
    robot_time_waiting: float
    number_users_before_robot: int
    people_with_priority: bool
    elevator_capacity: int
    elevator_non_stretched_capacity: int
    max_users_waiting: int
    number_people_in_elevator: int = 0
    people_with_impairments: bool = False
    next_time_over_timeout: bool = False
    waiting_users_did_not_enter: bool = False

    def vector(self) -> np.ndarray:
        """Numeric feature vector for neural-net agents, roughly normalised to [0, 1]."""
        return np.array(
            [
                self.number_people_in_elevator / max(self.elevator_capacity, 1),
                float(self.people_with_impairments),
                self.number_users_before_robot / max(self.max_users_waiting, 1),
                float(self.people_with_priority),
                min(1.0, self.robot_time_waiting / max(self.timeout, 1e-6)),
                float(self.next_time_over_timeout),
                float(self.task_urgent),
                float(
                    self.number_people_in_elevator + 1
                    > self.elevator_non_stretched_capacity
                ),
            ],
            dtype=np.float32,
        )

    def discrete(self) -> Tuple:
        """Coarse binning of the state for tabular Q-learning / bandit tables."""
        occ_bin = min(4, self.number_people_in_elevator // 2)  # 0..4
        pct_timeout_bin = min(3, int(4 * self.robot_time_waiting / max(self.timeout, 1e-6)))  # 0..3
        queue_bin = min(2, self.number_users_before_robot // 3)  # 0..2
        return (
            occ_bin,
            int(self.people_with_impairments),
            queue_bin,
            int(self.people_with_priority),
            pct_timeout_bin,
            int(self.next_time_over_timeout),
            int(self.task_urgent),
        )
