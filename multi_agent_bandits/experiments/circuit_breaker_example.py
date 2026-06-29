from multi_agent_bandits.core.circuit_breaker_environment import CircuitBreakerEnvironment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm

from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.strategies.risk_averse_epsilon_greedy import RiskAverseEpsilonGreedyAgent
from multi_agent_bandits.strategies.risk_averse_ucb import RiskAverseUCBAgent


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):

    n_agents = 5

    arms = [
        Arm(mean=1.0, sd=0.2),  #low return, low risk
        Arm(mean=2.0, sd=1.5),  #high return, high risk
        Arm(mean=1.2, sd=0.3),  #safe
    ]

    env = CircuitBreakerEnvironment(
        n_agents=n_agents,
        arms=arms,
        breaker_threshold=4,
        halt_duration=5,
        halted_reward=0.0,
        transparent_breakers=True
    )

    agents = [
        RandomAgent(env.n_arms),
        EpsilonGreedyAgent(env.n_arms),
        UCB_BaselineAgent(env.n_arms),
        RiskAverseEpsilonGreedyAgent(env.n_arms, epsilon=0.1, risk_aversion=0.5),
        RiskAverseUCBAgent(env.n_arms, risk_aversion=0.5)
    ]

    runner = ExperimentRunner(
        env,
        agents,
        timestep_limit=steps,
        save_dir=save_dir
    )

    runner.run(
        plot_rewards=plot_rewards,
        plot_frequencies=plot_frequencies
    )

    runner.print_summary()

    print("Circuit breaker triggers per arm:")
    for arm, count in enumerate(env.trigger_counts):
        print(f"Arm {arm}: {count} triggers")
    print(f"Total collisions: {sum(env.collision_count_log)}")
    print(f"Total halted-arm choices: {sum(env.disabled_choice_count_log)}")
    print(f"Average global reward per timestep: {sum(env.global_reward_log) / len(env.global_reward_log):.3f}")


if __name__ == "__main__":
    main(
        steps=1000,
        plot_rewards=True,
        plot_frequencies=True
    )

