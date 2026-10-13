"""
simple_rl.py
------------
Trains a simple RL agent to decide WAIT (0) vs ENTER (1) on the elevator
environment (elevator_env.py), with the reward shaped by a `importances`
dict of the same form as `importances_test[...]` in experimentation.py.

Two agents are included:
  1. TabularQAgent  - classic epsilon-greedy Q-learning over a discretised
                      state space. Easiest to inspect/debug.
  2. DQNAgent       - the same idea but with a tiny neural net (MicroMLP)
                      approximating Q(s, a) directly from the continuous
                      state vector, plus a small replay buffer + target
                      network, i.e. a minimal DQN.

Usage
-----
    python simple_rl.py                      # trains both agents on the
                                              # "complacent" style importances
    python simple_rl.py --profile time-efficient --episodes 3000 --agent dqn
"""

import argparse
import random
from collections import defaultdict, deque

import numpy as np

from elevator_env import ElevatorEnv, DEFAULT_IMPORTANCES
from baseline_models.micro_mlp import MicroMLP

# The four robot profiles from experimentation.py's importances_test, copied
# here so this file runs standalone. Feel free to import the real dict
# instead: `from experimentation import importances_test`
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


# ---------------------------------------------------------------------------
# 1. Tabular Q-learning
# ---------------------------------------------------------------------------
class TabularQAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.999):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, state_key, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state_key]))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_tabular(env: ElevatorEnv, episodes=4000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    agent = TabularQAgent()
    history = []
    for ep in range(episodes):
        state = env.reset()
        s_key = state.discrete()
        done = False
        ep_return = 0.0
        while not done:
            a = agent.act(s_key)
            next_state, reward, _, done, info = env.step(a)
            s_next_key = next_state.discrete()
            agent.update(s_key, a, reward, s_next_key, done)
            s_key = s_next_key
            ep_return += reward
        agent.decay_epsilon()
        history.append(ep_return)
    return agent, history


# ---------------------------------------------------------------------------
# 2. Minimal DQN (tiny numpy MLP, no torch required)
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, n_inputs=8, n_actions=2, lr=0.01, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999,
                 buffer_size=2000, batch_size=32, target_update_every=50):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.buffer = deque(maxlen=buffer_size)
        self.online = MicroMLP(n_inputs, n_actions, n_hidden=16, lr=lr, seed=0)
        self.target = MicroMLP(n_inputs, n_actions, n_hidden=16, lr=lr, seed=0)
        self._sync_target()
        self._train_calls = 0

    def _sync_target(self):
        self.target.W1, self.target.b1 = self.online.W1.copy(), self.online.b1.copy()
        self.target.W2, self.target.b2 = self.online.W2.copy(), self.online.b2.copy()

    def act(self, x, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.online.predict(x)
        return int(np.argmax(q))

    def remember(self, x, a, r, x_next, done):
        self.buffer.append((x, a, r, x_next, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        xs = np.stack([b[0] for b in batch])
        x_next = np.stack([b[3] for b in batch])
        q_now = self.online.predict(xs)
        q_next = self.target.predict(x_next)
        targets = q_now.copy()
        mask = np.zeros_like(q_now)
        for i, (_, a, r, _, done) in enumerate(batch):
            td_target = r if done else r + self.gamma * np.max(q_next[i])
            targets[i, a] = td_target
            mask[i, a] = 1.0
        self.online.train_step(xs, targets, action_mask=mask)
        self._train_calls += 1
        if self._train_calls % self.target_update_every == 0:
            self._sync_target()

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_dqn(env: ElevatorEnv, episodes=4000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    agent = DQNAgent()
    history = []
    for ep in range(episodes):
        state = env.reset()
        x = state.vector()
        done = False
        ep_return = 0.0
        while not done:
            a = agent.act(x)
            next_state, reward, _, done, info = env.step(a)
            x_next = next_state.vector()
            agent.remember(x, a, reward, x_next, done)
            agent.learn()
            x = x_next
            ep_return += reward
        agent.decay_epsilon()
        history.append(ep_return)
    return agent, history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(env, agent, kind, n_episodes=500, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    returns, enter_rate, timeout_rate = [], 0, 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        entered = False
        while not done:
            if kind == "tabular":
                a = agent.act(state.discrete(), greedy=True)
            else:
                a = agent.act(state.vector(), greedy=True)
            state, reward, _, done, info = env.step(a)
            ep_return += reward
            if info["entered"]:
                entered = True
            if info["timed_out"]:
                timeout_rate += 1
        returns.append(ep_return)
        enter_rate += int(entered)
    return {
        "mean_return": float(np.mean(returns)),
        "enter_rate": enter_rate / n_episodes,
        "timeout_rate": timeout_rate / n_episodes,
    }


def moving_average(x, w=100):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple RL agent for the elevator-entry decision.")
    parser.add_argument("--profile", choices=list(IMPORTANCES_TEST.keys()), default="complacent",
                         help="Which importances profile (reward weighting) to train against.")
    parser.add_argument("--agent", choices=["tabular", "dqn", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    importances = IMPORTANCES_TEST[args.profile]
    env = ElevatorEnv(importances=importances)

    print(f"Training on profile='{args.profile}' for {args.episodes} episodes\n")

    if args.agent in ("tabular", "both"):
        agent, history = train_tabular(env, episodes=args.episodes, seed=args.seed)
        ma = moving_average(history)
        print("[TabularQAgent]")
        print(f"  return, first 100 eps avg : {np.mean(history[:100]):.3f}")
        print(f"  return, last 100 eps avg  : {np.mean(history[-100:]):.3f}")
        stats = evaluate(env, agent, "tabular")
        print(f"  eval (greedy policy): mean_return={stats['mean_return']:.3f} "
              f"enter_rate={stats['enter_rate']:.2f} timeout_rate={stats['timeout_rate']:.2f}")
        print(f"  learned states: {len(agent.Q)}\n")

    if args.agent in ("dqn", "both"):
        agent, history = train_dqn(env, episodes=args.episodes, seed=args.seed)
        print("[DQNAgent]")
        print(f"  return, first 100 eps avg : {np.mean(history[:100]):.3f}")
        print(f"  return, last 100 eps avg  : {np.mean(history[-100:]):.3f}")
        stats = evaluate(env, agent, "dqn")
        print(f"  eval (greedy policy): mean_return={stats['mean_return']:.3f} "
              f"enter_rate={stats['enter_rate']:.2f} timeout_rate={stats['timeout_rate']:.2f}")
