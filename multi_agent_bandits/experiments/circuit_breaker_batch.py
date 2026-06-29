import os
import csv
import random
import statistics
import math

from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.circuit_breaker_environment import CircuitBreakerEnvironment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.reward_sharing import linear_share, zero_on_collision

from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.risk_averse_epsilon_greedy import RiskAverseEpsilonGreedyAgent
from multi_agent_bandits.strategies.risk_averse_ucb import RiskAverseUCBAgent


def set_seed(seed):
    random.seed(seed)


def make_arms(n_arms):
    """
    Financial-market-inspired arms.

    Arm 0: low return, low risk
    Arm 1: high return, high risk
    Arm 2: medium return, medium risk
    Arm 3: high return, very high risk
    Arm 4: safe asset

    If n_arms > 5, additional arms are added with varied
    risk-return profiles.
    """
    arm_specs = [
        (1.0, 0.2),  # low return, low risk
        (2.0, 1.5),  # high return, high risk
        (1.5, 0.7),  # medium return, medium risk
        (1.8, 2.0),  # high return, very high risk
        (1.2, 0.3),  # safe asset

        #repeated assets for larger markets
        (1.0, 0.2),
        (2.0, 1.5),
        (1.5, 0.7),
        (1.8, 2.0),
        (1.2, 0.3)
    ]

    if n_arms > len(arm_specs):
        raise ValueError(f"Requested {n_arms} arms, but only {len(arm_specs)} arm specifications are defined.")

    return [Arm(mean=mean, sd=sd) for mean, sd in arm_specs[:n_arms]]


def make_agents(n_arms, n_agents):
    '''
    Create a scalable population of agents.
    The population repeats the same five strategy types:
        - RandomAgent
        - EpsilonGreedyAgent
        - UCB_BaselineAgent
        - RiskAverseEpsilonGreedyAgent
        - RiskAverseUCBAgent
    '''
    base_agent_factories = [
        (
            "RandomAgent",
            lambda name: RandomAgent(
                n_arms,
                name=name
            )
        ),
        (
            "EpsilonGreedyAgent",
            lambda name: EpsilonGreedyAgent(
                n_arms,
                epsilon=0.1,
                name=name
            )
        ),
        (
            "UCB_BaselineAgent",
            lambda name: UCB_BaselineAgent(
                n_arms,
                name=name
            )
        ),
        (
            "RiskAverseEpsilonGreedyAgent",
            lambda name: RiskAverseEpsilonGreedyAgent(
                n_arms,
                epsilon=0.1,
                risk_aversion=0.5,
                name=name
            )
        ),
        (
            "RiskAverseUCBAgent",
            lambda name: RiskAverseUCBAgent(
                n_arms,
                risk_aversion=0.5,
                name=name
            )
        ),
     ]

    agents = []
    type_counts = {}

    while len(agents) < n_agents:
        for base_name, factory in base_agent_factories:
            if len(agents) >= n_agents:
                break

            type_counts[base_name] = type_counts.get(base_name, 0) + 1
            agent_name = f"{base_name}_{type_counts[base_name]}"

            agents.append(factory(agent_name))

    return agents


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


def compute_threshold(n_agents, threshold_ratio):
    '''
    Convert a threshold ratio into an integer number of agents.
    '''
    return max(1, math.ceil(n_agents * threshold_ratio))


def get_collision_policy(policy_name):

    if policy_name == "linear_share":
        return linear_share

    if policy_name == "zero_on_collision":
        return zero_on_collision

    raise ValueError(f"Unknown collision policy: {policy_name}")


def make_environment(condition, arms, n_agents):
    """
    Creates either baseline environment or circuit breaker environment.
    """

    collision_policy_name = condition.get("collision_policy", "linear_share")
    collision_policy = get_collision_policy(collision_policy_name)

    if condition["type"] == "baseline":
        return Environment(
            n_agents=n_agents,
            arms=arms,
            collision_policy=collision_policy
        )

    if condition["type"] == "circuit_breaker":
        threshold = condition.get("threshold")

        if threshold is None:
            threshold = compute_threshold(
                n_agents=n_agents,
                threshold_ratio=condition["threshold_ratio"]
            )

        return CircuitBreakerEnvironment(
            n_agents=n_agents,
            arms=arms,
            collision_policy=collision_policy,
            breaker_threshold=threshold,
            halt_duration=condition["halt_duration"],
            halted_reward=condition["halted_reward"],
            transparent_breakers=condition["transparent"]
        )

    raise ValueError(f"Unknown condition type: {condition['type']}")


def run_single_experiment(condition, seed, steps):
    """
    Runs one experiment and returns one row of summary results.
    """
    set_seed(seed)

    n_arms = condition.get("n_arms", 5)
    arms = make_arms(n_arms)
    n_agents = condition.get("n_agents", 5)
    agents = make_agents(n_arms=len(arms), n_agents=n_agents)

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
    avg_reward_per_agent_per_step = total_reward / (steps * n_agents)

    total_collisions = sum(env.collision_count_log)
    avg_collisions_per_step = total_collisions / steps

    global_reward_log = env.global_reward_log
    volatility = reward_volatility(global_reward_log)

    threshold = condition.get("threshold")

    if condition["type"] == "circuit_breaker" and threshold is None:
        threshold = compute_threshold(n_agents=n_agents, threshold_ratio=condition["threshold_ratio"])

    row = {
        "condition_name": condition["name"],
        "condition_type": condition["type"],
        "market_name": condition.get("market_name", ""),
        "collision_policy": condition.get("collision_policy", "linear_share"),
        "seed": seed,
        "steps": steps,
        "n_agents": n_agents,
        "n_arms": len(arms),

        "threshold": threshold if threshold is not None else "",
        "threshold_ratio": condition.get("threshold_ratio", ""),
        "halt_duration": condition.get("halt_duration", ""),
        "halted_reward": condition.get("halted_reward", ""),
        "transparent": condition.get("transparent", ""),

        "total_reward": total_reward,
        "avg_global_reward": avg_global_reward,
        "avg_reward_per_agent_per_step": avg_reward_per_agent_per_step,
        "reward_volatility": volatility,

        "total_collisions": total_collisions,
        "avg_collisions_per_step": avg_collisions_per_step,

        "total_halted_choices": total_halted_choices(env),
        "total_triggers": total_triggers(env),

        "market_wide_halt_steps": market_wide_halt_steps(env),
        "avg_available_arms": avg_available_arms(env),

        "gini_total_rewards": gini(runner.total_rewards),
    }

    for i, reward in enumerate(runner.total_rewards):
        row[f"agent_{i}_reward"] = reward
        row[f"agent_{i}_name"] = agents[i].name

    return row


def write_results_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
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
    This runs experiments across different market densities:
        - low_congestion:  5 agents / 10 arms
        - balanced:        5 agents / 5 arms
        - high_congestion: 10 agents / 5 arms

    For each market density, it runs:
        - baseline
        - opaque circuit breakers
        - transparent circuit breakers
    """

    output_path = os.path.join(save_dir, "summary_results.csv")

    market_configs = [
        {
            "market_name": "low_congestion",
            "n_agents": 5,
            "n_arms": 10,
        },
        {
            "market_name": "balanced",
            "n_agents": 5,
            "n_arms": 5,
        },
        {
            "market_name": "high_congestion",
            "n_agents": 10,
            "n_arms": 5,
        },
    ]

    threshold_ratios = [0.6, 0.8]
    halt_durations = [3, 5, 10]
    transparent_options = [False, True]

    conditions = []

    for market_config in market_configs:
        market_name = market_config["market_name"]
        n_agents = market_config["n_agents"]
        n_arms = market_config["n_arms"]

        #baseline condition for each market density
        conditions.append({
            "name": f"{market_name}_baseline",
            "type": "baseline",
            "market_name": market_name,
            "n_agents": n_agents,
            "n_arms": n_arms,
        })

        #conditions for each market density
        for threshold_ratio in threshold_ratios:
            for halt_duration in halt_durations:
                for transparent in transparent_options:
                    mode = "transparent" if transparent else "opaque"
                    ratio_label = str(threshold_ratio).replace(".", "")

                    conditions.append({
                        "name": f"{market_name}_cb_{mode}_ratio{ratio_label}_halt{halt_duration}",
                        "type": "circuit_breaker",
                        "market_name": market_name,
                        "n_agents": n_agents,
                        "n_arms": n_arms,
                        "threshold_ratio": threshold_ratio,
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


def market_wide_halt_steps(env):
    if hasattr(env, "market_wide_halt_log"):
        return sum(env.market_wide_halt_log)
    return 0


def avg_available_arms(env):
    if hasattr(env, "n_available_arms_log") and len(env.n_available_arms_log) > 0:
        return sum(env.n_available_arms_log) / len(env.n_available_arms_log)
    return env.n_arms