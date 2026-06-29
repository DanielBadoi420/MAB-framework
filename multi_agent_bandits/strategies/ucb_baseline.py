import math
import random
from multi_agent_bandits.core.agent import Agent


class UCB_BaselineAgent(Agent):
    """
    Baseline UCB implementation.
    Chooses arm based on: estimated_value + exploration_bonus,
    where the exploration bonus shrinks as an arm is selected more often.
    """

    def __init__(self, n_arms, name = None):
        super().__init__(n_arms, name=name)

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_steps = 0
        self.last_arm = None

    def choose_arm(self, available_arms = None):
        self.total_steps += 1

        if available_arms is None:
            available_arms = list(range(self.n_arms))

        #try each available arm at least once, but in random order.
        unpulled = [arm for arm in available_arms if self.counts[arm] == 0]
        if unpulled:
            self.last_arm = random.choice(unpulled)
            return self.last_arm

        ucb_scores = {}

        for arm in available_arms:
            bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
            ucb_scores[arm] = self.values[arm] + bonus

        #randomise the order before taking max, so exact ties are broken randomly.
        shuffled_arms = list(available_arms)
        random.shuffle(shuffled_arms)

        self.last_arm = max(shuffled_arms, key=lambda a: ucb_scores[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm

        if arm is None:
            return

        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])
