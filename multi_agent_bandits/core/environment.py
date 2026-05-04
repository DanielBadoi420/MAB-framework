from multi_agent_bandits.core.reward_sharing import linear_share
from multi_agent_bandits.core.arm import Arm

class Environment:
    """
    Extendable multi-agent bandit environment.
    Agents choose arms -> collisions are handled -> generate rewards.
    """
    def __init__(self, n_agents, arms, collision_policy=linear_share):
        self.n_agents = n_agents
        self.arms = arms
        self.n_arms = len(arms)
        self.collision_policy = collision_policy

        self.collision_count_log = []
        self.global_reward_log = []

    def sample_reward(self, arm_idx):
        return self.arms[arm_idx].sample()

    def step(self, agents):
        choices = [agent.choose_arm() for agent in agents]

        collisions = {}
        for i, arm in enumerate(choices):
            collisions.setdefault(arm, []).append(i)

        rewards = [0.0] * len(agents)

        collision_count = 0

        for arm, agent_ids in collisions.items():
            if len(agent_ids) > 1:
                collision_count += 1

            raw_reward = self.sample_reward(arm)

            if len(agent_ids) == 1:
                rewards[agent_ids[0]] = raw_reward
            else:
                shares = self.collision_policy(raw_reward, len(agent_ids))
                for idx, a_id in enumerate(agent_ids):
                    rewards[a_id] = shares[idx]

        for agent, reward in zip(agents, rewards):
            agent.update(reward)

        self.collision_count_log.append(collision_count)
        self.global_reward_log.append(sum(rewards))

        return choices, rewards
