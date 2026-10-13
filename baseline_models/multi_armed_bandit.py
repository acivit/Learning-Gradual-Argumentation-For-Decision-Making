"""
multi_armed_bandit.py
----------------------
Casts the "enter vs wait" decision as a multi-armed bandit rather than a full
RL problem: instead of bootstrapping future returns (like simple_rl.py does),
a bandit only ever learns from the *immediate* reward of the arm it pulls.
Three variants are included, in increasing order of realism:

  1. StatelessBandit      - two arms {wait, enter}, no context at all.
                             epsilon-greedy and UCB1 action-value estimation.
                             This answers "which action is best on average
                             across all situations?" - useful as a naive
                             baseline, but it can't adapt to context (e.g. an
                             urgent task vs. a relaxed one).

  2. ContextualBandit      - same two arms, but a separate value estimate is
                             kept per discretised context (state bucket),
                             still using simple running-average updates
                             (no bootstrapping/gamma).

  3. NeuralContextualBandit - a tiny MLP (MicroMLP) regresses the expected
                             immediate reward given (state vector, action),
                             trained purely on observed one-step rewards.
                             This is the "small neural net" option for the
                             bandit setting.

Every pull uses the same `importances`-weighted reward from elevator_env.py,
so results are directly comparable to simple_rl.py and multi_objective_rl.py.

Usage
-----
    python multi_armed_bandit.py
    python multi_armed_bandit.py --profile time-efficient --pulls 5000
"""

import argparse
import random
from collections import defaultdict

import numpy as np

from elevator_env import ElevatorEnv
from baseline_models.micro_mlp import MicroMLP
from baseline_models.simple_rl import IMPORTANCES_TEST  # reuse the same profile dicts


# ---------------------------------------------------------------------------
# 1. Stateless multi-armed bandit (epsilon-greedy and UCB1)
# ---------------------------------------------------------------------------
class StatelessBandit:
    def __init__(self, n_arms=2, strategy="epsilon_greedy", eps=0.1, ucb_c=1.5):
        self.n_arms = n_arms
        self.strategy = strategy
        self.eps = eps
        self.ucb_c = ucb_c
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)  # running average reward
        self.t = 0

    def select_arm(self):
        self.t += 1
        if self.strategy == "epsilon_greedy":
            if random.random() < self.eps or np.any(self.counts == 0):
                return int(np.random.randint(self.n_arms))
            return int(np.argmax(self.values))
        elif self.strategy == "ucb1":
            if np.any(self.counts == 0):
                return int(np.argmin(self.counts))  # try untried arms first
            bonus = self.ucb_c * np.sqrt(np.log(self.t) / self.counts)
            return int(np.argmax(self.values + bonus))
        else:
            raise ValueError(f"unknown strategy {self.strategy}")

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def train_stateless_bandit(env: ElevatorEnv, pulls=3000, strategy="epsilon_greedy", seed=0):
    """Each pull is one env.step() (not one full episode). We keep stepping
    through episodes back-to-back until we've taken `pulls` steps, so the
    bandit's average-reward estimate reflects the *whole* state distribution
    an evaluation rollout will encounter (including states reached only after
    some waiting), not just the very first state of each episode."""
    random.seed(seed)
    np.random.seed(seed)
    bandit = StatelessBandit(n_arms=2, strategy=strategy)
    rewards = []
    state = env.reset()
    for _ in range(pulls):
        arm = bandit.select_arm()
        next_state, reward, _, done, _ = env.step(arm)
        bandit.update(arm, reward)
        rewards.append(reward)
        state = env.reset() if done else next_state
    return bandit, rewards


# ---------------------------------------------------------------------------
# 2. Tabular contextual bandit
# ---------------------------------------------------------------------------
class ContextualBandit:
    def __init__(self, n_arms=2, eps=0.1):
        self.n_arms = n_arms
        self.eps = eps
        self.counts = defaultdict(lambda: np.zeros(n_arms, dtype=np.int64))
        self.values = defaultdict(lambda: np.zeros(n_arms, dtype=np.float64))

    def select_arm(self, context):
        if random.random() < self.eps or np.any(self.counts[context] == 0):
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values[context]))

    def update(self, context, arm, reward):
        self.counts[context][arm] += 1
        n = self.counts[context][arm]
        self.values[context][arm] += (reward - self.values[context][arm]) / n


def train_contextual_bandit(env: ElevatorEnv, pulls=3000, seed=0):
    """As with train_stateless_bandit, each pull is one env.step(), stepping
    through episodes back-to-back, so the bandit gets to see (and learn
    values for) contexts throughout an episode - not only the state right
    after reset()."""
    random.seed(seed)
    np.random.seed(seed)
    bandit = ContextualBandit(n_arms=2)
    rewards = []
    state = env.reset()
    for _ in range(pulls):
        context = state.discrete()
        arm = bandit.select_arm(context)
        next_state, reward, _, done, _ = env.step(arm)
        bandit.update(context, arm, reward)
        rewards.append(reward)
        state = env.reset() if done else next_state
    return bandit, rewards


# ---------------------------------------------------------------------------
# 3. Neural contextual bandit (tiny MLP, single-step targets, no bootstrapping)
# ---------------------------------------------------------------------------
class NeuralContextualBandit:
    def __init__(self, n_inputs=8, n_arms=2, lr=0.02, eps=0.1, seed=0):
        self.n_arms = n_arms
        self.eps = eps
        self.net = MicroMLP(n_inputs, n_arms, n_hidden=16, lr=lr, seed=seed)

    def select_arm(self, x):
        if random.random() < self.eps:
            return int(np.random.randint(self.n_arms))
        q = self.net.predict(x)
        return int(np.argmax(q))

    def update(self, x, arm, reward):
        target = self.net.predict(x).copy()
        mask = np.zeros(self.n_arms, dtype=np.float32)
        target[arm] = reward  # pure one-step target, no discounting
        mask[arm] = 1.0
        self.net.train_step(x, target, action_mask=mask)


def train_neural_bandit(env: ElevatorEnv, pulls=3000, seed=0):
    """Same full-trajectory reasoning as train_contextual_bandit, but with
    the neural function approximator so it can generalise across similar
    contexts instead of needing an exact bucket match."""
    random.seed(seed)
    np.random.seed(seed)
    bandit = NeuralContextualBandit()
    rewards = []
    state = env.reset()
    x = state.vector()
    for _ in range(pulls):
        arm = bandit.select_arm(x)
        next_state, reward, _, done, _ = env.step(arm)
        bandit.update(x, arm, reward)
        rewards.append(reward)
        state = env.reset() if done else next_state
        x = state.vector()
    return bandit, rewards


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
def summarize(name, bandit, rewards, tail=500):
    print(f"[{name}]")
    print(f"  mean reward, first {tail} pulls : {np.mean(rewards[:tail]):.3f}")
    print(f"  mean reward, last {tail} pulls  : {np.mean(rewards[-tail:]):.3f}")
    if isinstance(bandit, StatelessBandit):
        print(f"  arm value estimates (wait, enter): {bandit.values.round(3)}")
        print(f"  arm pull counts     (wait, enter): {bandit.counts}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-armed bandits for the elevator-entry decision.")
    parser.add_argument("--profile", choices=list(IMPORTANCES_TEST.keys()), default="complacent")
    parser.add_argument("--pulls", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    importances = IMPORTANCES_TEST[args.profile]
    env = ElevatorEnv(importances=importances)

    print(f"Training bandits on profile='{args.profile}' for {args.pulls} pulls\n")

    bandit, rewards = train_stateless_bandit(env, pulls=args.pulls, strategy="epsilon_greedy", seed=args.seed)
    summarize("StatelessBandit (epsilon-greedy)", bandit, rewards)

    bandit, rewards = train_stateless_bandit(env, pulls=args.pulls, strategy="ucb1", seed=args.seed)
    summarize("StatelessBandit (UCB1)", bandit, rewards)

    bandit, rewards = train_contextual_bandit(env, pulls=args.pulls, seed=args.seed)
    summarize("ContextualBandit (tabular)", bandit, rewards)
    print(f"  contexts seen: {len(bandit.values)}\n")

    bandit, rewards = train_neural_bandit(env, pulls=args.pulls, seed=args.seed)
    summarize("NeuralContextualBandit", bandit, rewards)
