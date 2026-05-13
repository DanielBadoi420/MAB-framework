from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.reward_sharing import linear_share


class CircuitBreakerEnvironment(Environment):
    """
    Multi-agent bandit environment with circuit breaker logic.

    Circuit breaker rule:
        If the number of agents choosing an arm reaches breaker_threshold,
        the arm is halted for halt_duration timesteps.

    Modes:
        transparent_breakers=False:
            Agents do not observe halted arms.
            They may choose halted arms and receive halted_reward.

        transparent_breakers=True:
            Agents observe which arms are available.
            The environment passes available_arms to choose_arm().
    """

    def __init__(
        self,
        n_agents,
        arms,
        collision_policy=linear_share,
        breaker_threshold=2,
        halt_duration=10,
        halted_reward=0.0,
        transparent_breakers=True,
    ):
        super().__init__(n_agents, arms, collision_policy)

        self.breaker_threshold = breaker_threshold
        self.halt_duration = halt_duration
        self.halted_reward = halted_reward
        self.transparent_breakers = transparent_breakers

        self.timestep = 0

        #cooldowns[arm] > 0 means the arm is halted.
        self.cooldowns = [0] * self.n_arms

        #pe-arm total number of times each breaker has triggered.
        self.trigger_counts = [0] * self.n_arms

        #logs, one entry per timestep.
        self.breaker_triggers_log = []
        self.halted_arms_log = []
        self.available_arms_log = []
        self.collision_count_log = []
        self.disabled_choice_count_log = []
        self.global_reward_log = []
        self.market_wide_halt_log = []
        self.n_available_arms_log = []

    def update_cooldowns(self):
        """
        Decrease cooldown timer for all halted arms.
        """
        for arm in range(self.n_arms):
            if self.cooldowns[arm] > 0:
                self.cooldowns[arm] -= 1

    def is_halted(self, arm):
        """
        Return True if an arm is currently halted.
        """
        return self.cooldowns[arm] > 0

    def get_halted_arms(self):
        """
        Return a list of all currently halted arms.
        """
        return [arm for arm in range(self.n_arms) if self.is_halted(arm)]

    def get_available_arms(self):
        return [arm for arm in range(self.n_arms) if not self.is_halted(arm)]

    def choose_agent_arms(self, agents, available_arms):
        """
        Choose arms for all agents.

        In transparent mode, agents receive available_arms.
        In opaque mode, agents choose without observing halted arms.
        """
        if self.transparent_breakers:
            return [
                agent.choose_arm(available_arms=available_arms)
                for agent in agents
            ]

        return [agent.choose_arm() for agent in agents]

    def step(self, agents):
        self.timestep += 1
        self.update_cooldowns()

        halted_arms = self.get_halted_arms()
        available_arms = self.get_available_arms()

        if self.transparent_breakers and len(available_arms) == 0:
            choices = [None] * len(agents)
            rewards = [0.0] * len(agents)

            self.breaker_triggers_log.append([])
            self.halted_arms_log.append(halted_arms)
            self.available_arms_log.append(available_arms)
            self.collision_count_log.append(0)
            self.disabled_choice_count_log.append(0)
            self.global_reward_log.append(sum(rewards))
            self.market_wide_halt_log.append(1)
            self.n_available_arms_log.append(0)

            return choices, rewards

        choices = self.choose_agent_arms(agents, available_arms)

        collisions = {}
        for agent_idx, arm in enumerate(choices):
            collisions.setdefault(arm, []).append(agent_idx)

        rewards = [0.0] * len(agents)

        triggered_arms = []
        collision_count = 0
        disabled_choice_count = 0

        for arm, agent_ids in collisions.items():

            if len(agent_ids) > 1:
                collision_count += 1

            #case 1: agents chose a halted arm.
            if self.is_halted(arm):
                disabled_choice_count += len(agent_ids)

                for agent_id in agent_ids:
                    rewards[agent_id] = self.halted_reward

                continue

            #case 2: arm is active, so sample normal reward.
            raw_reward = self.sample_reward(arm)

            if len(agent_ids) == 1:
                rewards[agent_ids[0]] = raw_reward
            else:
                shares = self.collision_policy(raw_reward, len(agent_ids))

                for idx, agent_id in enumerate(agent_ids):
                    rewards[agent_id] = shares[idx]

            #case 3: trigger breaker if crowding is too high.
            if len(agent_ids) >= self.breaker_threshold:
                self.cooldowns[arm] = self.halt_duration
                self.trigger_counts[arm] += 1
                triggered_arms.append(arm)

        for agent, reward in zip(agents, rewards):
            agent.update(reward)

        self.breaker_triggers_log.append(triggered_arms)
        self.halted_arms_log.append(halted_arms)
        self.available_arms_log.append(available_arms)
        self.collision_count_log.append(collision_count)
        self.disabled_choice_count_log.append(disabled_choice_count)
        self.global_reward_log.append(sum(rewards))
        self.market_wide_halt_log.append(0)
        self.n_available_arms_log.append(len(available_arms))

        return choices, rewards