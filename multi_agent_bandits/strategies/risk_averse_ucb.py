import math
from multi_agent_bandits.core.agent import Agent


class RiskAverseUCBAgent(Agent):
    """
    UCB agent with risk aversion.

    Standard UCB score: estimated_mean + exploration_bonus

    Risk-averse UCB score: estimated_mean + exploration_bonus - risk_aversion * estimated_std

    This means the agent still explores, but is less attracted to volatile arms.
    """

    def __init__(self, n_arms, risk_aversion=0.5, name=None):
        super().__init__(n_arms, name=name)

        self.risk_aversion = risk_aversion

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.squared_values = [0.0] * n_arms

        self.total_steps = 0
        self.last_arm = None

    def choose_arm(self, available_arms=None):
        self.total_steps += 1

        if available_arms is None:
            available_arms = list(range(self.n_arms))

        #try each available arm at least once
        for arm in available_arms:
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        scores = {}

        for arm in available_arms:
            mean = self.values[arm]

            exploration_bonus = math.sqrt(
                (2 * math.log(self.total_steps)) / self.counts[arm])

            variance = self.squared_values[arm] - mean ** 2
            variance = max(0.0, variance)

            std = math.sqrt(variance)

            scores[arm] = mean + exploration_bonus - self.risk_aversion * std

        self.last_arm = max(available_arms, key=lambda a: scores[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm

        self.counts[arm] += 1
        step = 1 / self.counts[arm]

        self.values[arm] += step * (reward - self.values[arm])
        self.squared_values[arm] += step * ((reward ** 2) - self.squared_values[arm])