class Agent:
    """
    Base interface for all agents.
    """

    def __init__(self, n_arms, name=None):
        self.n_arms = n_arms
        self.name = name if name is not None else self.__class__.__name__

    def choose_arm(self, available_arms=None):
        raise NotImplementedError

    def update(self, reward):
        pass
