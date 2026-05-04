import random
import math
from multi_agent_bandits.core.agent import Agent


class RiskAverseEpsilonGreedyAgent(Agent):
    """
    Epsilon-greedy agent that penalizes arms with high observed reward variance.

    Score: estimated_mean - risk_aversion * estimated_std

    Higher risk_aversion means the agent avoids volatile arms more strongly.
    """

    def __init__(self, n_arms, epsilon=0.1, risk_aversion=0.5, name=None):
        super().__init__(n_arms, name=name)

        self.epsilon = epsilon
        self.risk_aversion = risk_aversion

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.squared_values = [0.0] * n_arms
        self.last_arm = None

    def choose_arm(self, available_arms=None):
        if available_arms is None:
            available_arms = list(range(self.n_arms))

        if random.random() < self.epsilon:
            self.last_arm = random.choice(available_arms)
            return self.last_arm

        #try unvisited available arms first
        for arm in available_arms:
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        self.last_arm = max(available_arms, key=self._risk_adjusted_score)
        return self.last_arm

    def _risk_adjusted_score(self, arm):
        mean = self.values[arm]

        variance = self.squared_values[arm] - mean ** 2
        variance = max(0.0, variance)

        std = math.sqrt(variance)

        return mean - self.risk_aversion * std

    def update(self, reward):
        arm = self.last_arm

        self.counts[arm] += 1
        step = 1 / self.counts[arm]

        self.values[arm] += step * (reward - self.values[arm])
        self.squared_values[arm] += step * ((reward ** 2) - self.squared_values[arm])