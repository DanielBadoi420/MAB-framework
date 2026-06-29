import random
from multi_agent_bandits.core.agent import Agent

class RandomAgent(Agent):
    def __init__(self, n_arms, name=None):
        super().__init__(n_arms, name=name)

    def choose_arm(self, available_arms = None):
        if available_arms is None:
            available_arms = list(range(self.n_arms))

        return random.choice(available_arms)
