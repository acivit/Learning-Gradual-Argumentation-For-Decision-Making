"""
multi_objective_rl.py
-----------------------
Multi-objective RL (MORL) for the elevator-entry decision, with three
objectives (matching elevator_env.py's `reward_vector`):

    0. robot_urgency   - getting the robot's task done on time
    1. inside_comfort  - not crowding/stretching the elevator, especially
                          around people with reduced mobility
    2. breaking_queue  - respecting the physical queue / priority passengers

Two agents are included:

  1. VectorQAgent      - tabular Q-learning that keeps a *separate* Q-value
                          per objective, Q_k(s, a) for k in {0,1,2}, each
                          bootstrapped with its own reward stream. Actions are
                          selected by linearly scalarizing with a weight
                          vector `w` (by default derived from the same
                          `importances` dict used elsewhere, so profiles stay
                          comparable): argmax_a sum_k w_k * Q_k(s, a).
                          This is the standard "linear scalarization" MORL
                          approach and lets you sweep `w` after training to
                          see the trade-off curve (a crude Pareto front)
                          without retraining.

  2. MultiHeadDQNAgent - the neural equivalent: one tiny MLP with 2*3=6
                          outputs (one Q-value per action per objective),
                          trained with per-objective TD targets, actions
                          chosen the same way (scalarize, then argmax).

Usage
-----
    python multi_objective_rl.py
    python multi_objective_rl.py --profile comfort-caring --episodes 3000
    python multi_objective_rl.py --sweep        # show the trade-off curve
"""

import argparse
import random
from collections import defaultdict, deque

import numpy as np

from elevator_env import ElevatorEnv
from baseline_models.micro_mlp import MicroMLP
from baseline_models.simple_rl import IMPORTANCES_TEST

OBJECTIVES = ("robot_urgency", "inside_comfort", "breaking_queue")


def weights_from_importances(importances) -> np.ndarray:
    """Turn a profile's importances dict into a 3-vector of scalarization
    weights, one per objective, normalised to sum to 1."""
    w = np.array(
        [
            importances["robot_urgency"],
            importances["inside_comfort"],
            importances["breaking_queue"],
        ],
        dtype=np.float64,
    )
    return w / w.sum()


# ---------------------------------------------------------------------------
# 1. Tabular vector Q-learning with linear scalarization
# ---------------------------------------------------------------------------
class VectorQAgent:
    def __init__(self, n_actions=2, n_objectives=3, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999):
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        # Q[state] -> array shape (n_actions, n_objectives)
        self.Q = defaultdict(lambda: np.zeros((n_actions, n_objectives), dtype=np.float32))

    def scalarized_values(self, state_key, weights):
        return self.Q[state_key] @ weights  # shape (n_actions,)

    def act(self, state_key, weights, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.scalarized_values(state_key, weights)))

    def update(self, s, a, r_vec, s_next, done, weights):
        if done:
            target = r_vec
        else:
            best_a = int(np.argmax(self.scalarized_values(s_next, weights)))
            target = r_vec + self.gamma * self.Q[s_next][best_a]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_vector_q(env: ElevatorEnv, weights: np.ndarray, episodes=4000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    agent = VectorQAgent()
    history = []  # list of 3-vectors, undiscounted episode return per objective
    for ep in range(episodes):
        state = env.reset()
        s_key = state.discrete()
        done = False
        ep_return = np.zeros(3, dtype=np.float64)
        while not done:
            a = agent.act(s_key, weights)
            next_state, _, r_vec, done, info = env.step(a)
            s_next_key = next_state.discrete()
            agent.update(s_key, a, r_vec, s_next_key, done, weights)
            s_key = s_next_key
            ep_return += r_vec
        agent.decay_epsilon()
        history.append(ep_return)
    return agent, np.array(history)


# ---------------------------------------------------------------------------
# 2. Multi-head neural MORL (tiny MLP, 2 actions x 3 objectives = 6 outputs)
# ---------------------------------------------------------------------------
class MultiHeadDQNAgent:
    def __init__(self, n_inputs=8, n_actions=2, n_objectives=3, lr=0.01, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999,
                 buffer_size=2000, batch_size=32, target_update_every=50):
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.buffer = deque(maxlen=buffer_size)
        n_out = n_actions * n_objectives
        self.online = MicroMLP(n_inputs, n_out, n_hidden=24, lr=lr, seed=0)
        self.target = MicroMLP(n_inputs, n_out, n_hidden=24, lr=lr, seed=0)
        self._sync_target()
        self._train_calls = 0

    def _sync_target(self):
        self.target.W1, self.target.b1 = self.online.W1.copy(), self.online.b1.copy()
        self.target.W2, self.target.b2 = self.online.W2.copy(), self.online.b2.copy()

    def _reshape(self, flat):
        # flat: (..., n_actions * n_objectives) -> (..., n_actions, n_objectives)
        return flat.reshape(*flat.shape[:-1], self.n_actions, self.n_objectives)

    def scalarized_values(self, x, weights):
        q = self._reshape(self.online.predict(x))  # (n_actions, n_objectives)
        return q @ weights

    def act(self, x, weights, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.scalarized_values(x, weights)))

    def remember(self, x, a, r_vec, x_next, done):
        self.buffer.append((x, a, r_vec, x_next, done))

    def learn(self, weights):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        xs = np.stack([b[0] for b in batch])
        x_next = np.stack([b[3] for b in batch])

        q_now_flat = self.online.predict(xs)
        q_now = self._reshape(q_now_flat)
        q_next = self._reshape(self.target.predict(x_next))  # (batch, n_actions, n_objectives)
        scal_next = q_next @ weights  # (batch, n_actions)
        best_next = np.argmax(scal_next, axis=1)

        targets = q_now.copy()
        mask = np.zeros_like(q_now)
        for i, (_, a, r_vec, _, done) in enumerate(batch):
            if done:
                td_target = r_vec
            else:
                td_target = r_vec + self.gamma * q_next[i, best_next[i]]
            targets[i, a] = td_target
            mask[i, a] = 1.0

        self.online.train_step(
            xs,
            targets.reshape(len(batch), -1),
            action_mask=mask.reshape(len(batch), -1),
        )
        self._train_calls += 1
        if self._train_calls % self.target_update_every == 0:
            self._sync_target()

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_multihead_dqn(env: ElevatorEnv, weights: np.ndarray, episodes=4000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    agent = MultiHeadDQNAgent()
    history = []
    for ep in range(episodes):
        state = env.reset()
        x = state.vector()
        done = False
        ep_return = np.zeros(3, dtype=np.float64)
        while not done:
            a = agent.act(x, weights)
            next_state, _, r_vec, done, info = env.step(a)
            x_next = next_state.vector()
            agent.remember(x, a, r_vec, x_next, done)
            agent.learn(weights)
            x = x_next
            ep_return += r_vec
        agent.decay_epsilon()
        history.append(ep_return)
    return agent, np.array(history)


# ---------------------------------------------------------------------------
# Evaluation + weight-sweep (Pareto trade-off) helpers
# ---------------------------------------------------------------------------
def evaluate_vector(env, agent, weights, n_episodes=500, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    totals = np.zeros(3, dtype=np.float64)
    enter_rate = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        entered = False
        ep_ret = np.zeros(3, dtype=np.float64)
        while not done:
            a = agent.act(state.discrete(), weights, greedy=True)
            state, _, r_vec, done, info = env.step(a)
            ep_ret += r_vec
            entered = entered or info["entered"]
        totals += ep_ret
        enter_rate += int(entered)
    return totals / n_episodes, enter_rate / n_episodes


def weight_sweep(env, episodes_per_point=1500, n_points=5, seed=0):
    """Train a fresh VectorQAgent for a handful of scalarization weight
    vectors, to show how the objective trade-off shifts (a coarse Pareto
    front). Same reward function (importances) throughout; only the
    scalarization weights used for action-selection/bootstrapping change."""
    print("Weight sweep (urgency, comfort, queue) -> mean per-objective episode return\n")
    for alpha in np.linspace(0.0, 1.0, n_points):
        # simple 2-objective sweep between urgency and comfort, queue fixed low,
        # then renormalised - illustrative, not exhaustive.
        w = np.array([alpha, 1 - alpha, 0.3])
        w = w / w.sum()
        agent, _ = train_vector_q(env, w, episodes=episodes_per_point, seed=seed)
        totals, enter_rate = evaluate_vector(env, agent, w)
        print(f"  w={w.round(2)} -> urgency={totals[0]:+.2f} comfort={totals[1]:+.2f} "
              f"queue={totals[2]:+.2f} enter_rate={enter_rate:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-objective RL for the elevator-entry decision.")
    parser.add_argument("--profile", choices=list(IMPORTANCES_TEST.keys()), default="complacent")
    parser.add_argument("--agent", choices=["vector_q", "multihead_dqn", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="Run a scalarization-weight sweep instead.")
    args = parser.parse_args()

    importances = IMPORTANCES_TEST[args.profile]
    env = ElevatorEnv(importances=importances)
    weights = weights_from_importances(importances)

    if args.sweep:
        weight_sweep(env)
        raise SystemExit

    print(f"Training on profile='{args.profile}' for {args.episodes} episodes")
    print(f"Scalarization weights (urgency, comfort, queue) = {weights.round(3)}\n")

    if args.agent in ("vector_q", "both"):
        agent, history = train_vector_q(env, weights, episodes=args.episodes, seed=args.seed)
        print("[VectorQAgent]")
        print(f"  per-objective return, first 100 eps avg: {history[:100].mean(axis=0).round(3)}")
        print(f"  per-objective return, last 100 eps avg : {history[-100:].mean(axis=0).round(3)}")
        totals, enter_rate = evaluate_vector(env, agent, weights)
        print(f"  eval (greedy): per-objective mean return={totals.round(3)} enter_rate={enter_rate:.2f}")
        print(f"  learned states: {len(agent.Q)}\n")

    if args.agent in ("multihead_dqn", "both"):
        agent, history = train_multihead_dqn(env, weights, episodes=args.episodes, seed=args.seed)
        print("[MultiHeadDQNAgent]")
        print(f"  per-objective return, first 100 eps avg: {history[:100].mean(axis=0).round(3)}")
        print(f"  per-objective return, last 100 eps avg : {history[-100:].mean(axis=0).round(3)}")

        def act_greedy_wrapper(state_vec, w):
            return agent.act(state_vec, w, greedy=True)

        random.seed(123)
        np.random.seed(123)
        totals = np.zeros(3, dtype=np.float64)
        enter_rate = 0
        n_eval = 500
        for _ in range(n_eval):
            state = env.reset()
            done = False
            entered = False
            ep_ret = np.zeros(3, dtype=np.float64)
            while not done:
                a = act_greedy_wrapper(state.vector(), weights)
                state, _, r_vec, done, info = env.step(a)
                ep_ret += r_vec
                entered = entered or info["entered"]
            totals += ep_ret
            enter_rate += int(entered)
        totals /= n_eval
        enter_rate /= n_eval
        print(f"  eval (greedy): per-objective mean return={totals.round(3)} enter_rate={enter_rate:.2f}")
