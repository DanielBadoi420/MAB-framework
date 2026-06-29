from multi_agent_bandits.experiments.circuit_breaker_batch import (
    run_single_experiment,
    write_results_csv,
    print_condition_summary,
)


def main(
    steps=10000,
    save_dir="results/circuit_breaker_collision_batch",
    n_seeds=30,
    plot_rewards=False,
    plot_frequencies=False,
):
    """
    Robustness experiment comparing collision policies.

    Compares:
        - linear_share
        - zero_on_collision

    under selected market densities and circuit breaker settings.
    """

    import os

    output_path = os.path.join(save_dir, "summary_results.csv")

    market_configs = [
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

    collision_policies = [
        "linear_share",
        "zero_on_collision",
    ]

    threshold_ratios = [0.6, 0.8]
    halt_durations = [3, 5]
    transparent_options = [False, True]

    conditions = []

    for market_config in market_configs:
        market_name = market_config["market_name"]
        n_agents = market_config["n_agents"]
        n_arms = market_config["n_arms"]

        for collision_policy in collision_policies:

            #baseline for each market and collision policy
            conditions.append({
                "name": f"{market_name}_{collision_policy}_baseline",
                "type": "baseline",
                "market_name": market_name,
                "n_agents": n_agents,
                "n_arms": n_arms,
                "collision_policy": collision_policy,
            })

            for threshold_ratio in threshold_ratios:
                for halt_duration in halt_durations:
                    for transparent in transparent_options:
                        mode = "transparent" if transparent else "opaque"
                        ratio_label = str(threshold_ratio).replace(".", "")

                        conditions.append({
                            "name": f"{market_name}_{collision_policy}_cb_{mode}_ratio{ratio_label}_halt{halt_duration}",
                            "type": "circuit_breaker",
                            "market_name": market_name,
                            "n_agents": n_agents,
                            "n_arms": n_arms,
                            "collision_policy": collision_policy,
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