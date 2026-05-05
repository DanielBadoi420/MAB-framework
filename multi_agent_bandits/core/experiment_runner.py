import os
import csv
import matplotlib.pyplot as plt

class ExperimentRunner:
    def __init__(self, env, agents, timestep_limit=1000, save_dir=None):
        self.env = env
        self.agents = agents
        self.T = timestep_limit
        self.choices_log = []
        self.rewards_log = []
        self.total_rewards = [0.0] * len(agents)
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def print_experiment_info(self):
        print("==== Experiment Setup ====")
        print(f"Timesteps: {self.T}")
        print(f"Number of arms: {self.env.n_arms}")
        print("Arms:")
        for i, arm in enumerate(self.env.arms):
            print(f"  Arm {i}: mean={arm.mean}, sd={arm.sd}")
        print(f"Collision policy: {self.env.collision_policy.__name__}")
        print(f"Number of agents: {len(self.agents)}")
        print("Agents:")
        for i, ag in enumerate(self.agents):
            print(f"  - Agent {i}: {ag.name}")
        print("==========================")

    def run(self, plot_rewards=False, plot_frequencies=False, verbose = True):
        if verbose:
            self.print_experiment_info()

        for t in range(self.T):
            choices, rewards = self.env.step(self.agents)
            self.choices_log.append(choices)
            self.rewards_log.append(rewards)
            for i, r in enumerate(rewards):
                self.total_rewards[i] += r

        if self.save_dir:
            self.save_logs()

        if plot_rewards:
            self.plot_reward_trajectories()

        if plot_frequencies:
            self.plot_arm_frequencies()

        return self.choices_log, self.rewards_log

    def save_logs(self):
        choices_path = os.path.join(self.save_dir, "choices.csv")
        with open(choices_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"agent_{i}" for i in range(len(self.agents))])
            writer.writerows(self.choices_log)

        rewards_path = os.path.join(self.save_dir, "rewards.csv")
        with open(rewards_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"agent_{i}" for i in range(len(self.agents))])
            writer.writerows(self.rewards_log)

        metadata_path = os.path.join(self.save_dir, "metadata.csv")
        with open(metadata_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["timesteps", self.T])
            writer.writerow(["n_arms", self.env.n_arms])
            for i, arm in enumerate(self.env.arms):
                writer.writerow([f"arm_{i}_mean", arm.mean])
                writer.writerow([f"arm_{i}_sd", arm.sd])
            writer.writerow(["collision_policy", self.env.collision_policy.__name__])
            writer.writerow(["n_agents", len(self.agents)])
            for i, ag in enumerate(self.agents):
                writer.writerow([f"agent_{i}_name", ag.name])

        summary_path = os.path.join(self.save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("==== Experiment Summary ====\n")
            f.write(f"Timesteps: {self.T}\n")
            f.write(f"Number of arms: {self.env.n_arms}\n")
            f.write("Arms:\n")
            for i, arm in enumerate(self.env.arms):
                f.write(f"  Arm {i}: mean={arm.mean}, sd={arm.sd}\n")
            f.write(f"Collision policy: {self.env.collision_policy.__name__}\n")
            f.write(f"Number of agents: {len(self.agents)}\n")
            f.write("Agents:\n")
            for i, ag in enumerate(self.agents):
                f.write(f"  - Agent {i}: {ag.name}\n")
            f.write("Total rewards:\n")
            for i, total in enumerate(self.total_rewards):
                f.write(f"  Agent {i} ({self.agents[i].name}): {total:.2f}\n")
            f.write("============================\n")

    def print_summary(self):
        print("Experiment Summary")
        for i, total in enumerate(self.total_rewards):
            print(f"Agent {i} ({self.agents[i].name}) total reward: {total:.2f}")
        print("--------------")

    def plot_reward_trajectories(self):
        import numpy as np
        plt.figure(figsize=(12, 5))
        for agent_idx, agent in enumerate(self.agents):
            rewards = np.array([step[agent_idx] for step in self.rewards_log])
            cumavg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
            plt.plot(cumavg, label=agent.name)
        plt.title("Cumulative average reward")
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_arm_frequencies(self):
        import numpy as np
        n_agents = len(self.agents)
        n_arms = self.env.n_arms
        counts = np.zeros((n_agents, n_arms), dtype=int)
        for step in self.choices_log:
            for agent_idx, arm in enumerate(step):
                if arm is None:
                    continue
                counts[agent_idx, arm] += 1
        fig, axes = plt.subplots(n_agents, 1, figsize=(8, 3 * n_agents), sharex=True)
        if n_agents == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.bar(range(n_arms), counts[i], color=f"C{i}")
            ax.set_ylabel("Count")
            ax.set_title(f"Arm frequencies — {self.agents[i].name}")
        axes[-1].set_xlabel("Arm")
        plt.tight_layout()
        plt.show()
