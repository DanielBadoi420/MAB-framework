import os
import csv
import random
import statistics

from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.circuit_breaker_environment import CircuitBreakerEnvironment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner

from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.risk_averse_epsilon_greedy import RiskAverseEpsilonGreedyAgent
from multi_agent_bandits.strategies.risk_averse_ucb import RiskAverseUCBAgent


def set_seed(seed):
    random.seed(seed)


def make_arms():
    """
    Financial-market-inspired arms.

    Arm 0: low return, low risk
    Arm 1: high return, high risk
    Arm 2: medium return, medium risk
    Arm 3: high return, very high risk
    Arm 4: safe asset
    """
    return [
        Arm(mean=1.0, sd=0.2),
        Arm(mean=2.0, sd=1.5),
        Arm(mean=1.5, sd=0.7),
        Arm(mean=1.8, sd=2.0),
        Arm(mean=1.2, sd=0.3),
    ]


def make_agents(n_arms):
    return [
        RandomAgent(n_arms),
        EpsilonGreedyAgent(n_arms, epsilon=0.1),
        UCB_BaselineAgent(n_arms),
        RiskAverseEpsilonGreedyAgent(n_arms, epsilon=0.1, risk_aversion=0.5),
        RiskAverseUCBAgent(n_arms, risk_aversion=0.5),
    ]


def gini(values):
    """
    Compute Gini coefficient for reward inequality.
    0 means perfectly equal rewards.
    Higher values mean more inequality.
    """
    values = [v for v in values if v >= 0]

    if len(values) == 0:
        return 0.0

    total = sum(values)

    if total == 0:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    weighted_sum = 0.0
    for i, value in enumerate(sorted_values, start=1):
        weighted_sum += i * value

    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def reward_volatility(global_reward_log):
    """
    Standard deviation of global reward over time.
    """
    if len(global_reward_log) <= 1:
        return 0.0

    return statistics.stdev(global_reward_log)


def total_triggers(env):
    """
    Circuit breaker environments have trigger_counts.
    Baseline environments do not.
    """
    if hasattr(env, "trigger_counts"):
        return sum(env.trigger_counts)

    return 0


def total_halted_choices(env):
    """
    Circuit breaker environments have disabled_choice_count_log.
    Baseline environments do not.
    """
    if hasattr(env, "disabled_choice_count_log"):
        return sum(env.disabled_choice_count_log)

    return 0


def make_environment(condition, arms, n_agents):
    """
    Creates either baseline environment or circuit breaker environment.
    """

    if condition["type"] == "baseline":
        return Environment(
            n_agents=n_agents,
            arms=arms,
        )

    if condition["type"] == "circuit_breaker":
        return CircuitBreakerEnvironment(
            n_agents=n_agents,
            arms=arms,
            breaker_threshold=condition["threshold"],
            halt_duration=condition["halt_duration"],
            halted_reward=condition["halted_reward"],
            transparent_breakers=condition["transparent"],
        )

    raise ValueError(f"Unknown condition type: {condition['type']}")


def run_single_experiment(condition, seed, steps):
    """
    Runs one experiment and returns one row of summary results.
    """
    set_seed(seed)

    arms = make_arms()
    n_agents = 5
    agents = make_agents(n_arms=len(arms))

    env = make_environment(
        condition=condition,
        arms=arms,
        n_agents=n_agents,
    )

    runner = ExperimentRunner(
        env=env,
        agents=agents,
        timestep_limit=steps,
        save_dir=None,
    )

    # Important: no plots in batch mode.
    runner.run(
        plot_rewards=False,
        plot_frequencies=False,
        verbose=False
    )

    total_reward = sum(runner.total_rewards)
    avg_global_reward = total_reward / steps

    total_collisions = sum(env.collision_count_log)
    avg_collisions_per_step = total_collisions / steps

    global_reward_log = env.global_reward_log
    volatility = reward_volatility(global_reward_log)

    row = {
        "condition_name": condition["name"],
        "condition_type": condition["type"],
        "seed": seed,
        "steps": steps,
        "n_agents": n_agents,
        "n_arms": len(arms),

        "threshold": condition.get("threshold", ""),
        "halt_duration": condition.get("halt_duration", ""),
        "halted_reward": condition.get("halted_reward", ""),
        "transparent": condition.get("transparent", ""),

        "total_reward": total_reward,
        "avg_global_reward": avg_global_reward,
        "reward_volatility": volatility,

        "total_collisions": total_collisions,
        "avg_collisions_per_step": avg_collisions_per_step,

        "total_halted_choices": total_halted_choices(env),
        "total_triggers": total_triggers(env),

        "gini_total_rewards": gini(runner.total_rewards),
    }

    for i, reward in enumerate(runner.total_rewards):
        row[f"agent_{i}_reward"] = reward
        row[f"agent_{i}_name"] = agents[i].name

    return row


def write_results_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_condition_summary(rows):
    """
    Prints compact summary aggregated by condition.
    """
    condition_names = sorted(set(row["condition_name"] for row in rows))

    print("\n==== Batch Summary ====")

    for condition_name in condition_names:
        condition_rows = [
            row for row in rows
            if row["condition_name"] == condition_name
        ]

        avg_rewards = [row["avg_global_reward"] for row in condition_rows]
        collisions = [row["total_collisions"] for row in condition_rows]
        halted_choices = [row["total_halted_choices"] for row in condition_rows]
        triggers = [row["total_triggers"] for row in condition_rows]
        ginis = [row["gini_total_rewards"] for row in condition_rows]

        print(f"\nCondition: {condition_name}")
        print(f"  Runs: {len(condition_rows)}")
        print(f"  Avg global reward: {statistics.mean(avg_rewards):.3f} ± {statistics.stdev(avg_rewards):.3f}")
        print(f"  Collisions: {statistics.mean(collisions):.1f} ± {statistics.stdev(collisions):.1f}")
        print(f"  Halted choices: {statistics.mean(halted_choices):.1f} ± {statistics.stdev(halted_choices):.1f}")
        print(f"  Triggers: {statistics.mean(triggers):.1f} ± {statistics.stdev(triggers):.1f}")
        print(f"  Gini: {statistics.mean(ginis):.3f} ± {statistics.stdev(ginis):.3f}")


def main(
    steps=10000,
    save_dir="results/circuit_breaker_batch",
    n_seeds=30,
    plot_rewards=False,
    plot_frequencies=False,
):
    """
    Batch experiment for circuit breaker thesis.
    """

    output_path = os.path.join(save_dir, "summary_results.csv")

    conditions = [
        {
            "name": "baseline",
            "type": "baseline",
        },
    ]

    for threshold in [3, 4]:
        for halt_duration in [1, 3, 5]:
            for transparent in [False, True]:
                mode = "transparent" if transparent else "opaque"

                conditions.append({
                    "name": f"cb_{mode}_threshold{threshold}_halt{halt_duration}",
                    "type": "circuit_breaker",
                    "threshold": threshold,
                    "halt_duration": halt_duration,
                    "halted_reward": 0.0,
                    "transparent": transparent,
                })

    rows = []

    for condition in conditions:
        for seed in range(n_seeds):
            print(f"Running condition={condition['name']}, seed={seed}")

            row = run_single_experiment(
                condition=condition,
                seed=seed,
                steps=steps,
            )

            rows.append(row)

    write_results_csv(rows, output_path)
    print_condition_summary(rows)

    print(f"\nSaved results to: {output_path}")