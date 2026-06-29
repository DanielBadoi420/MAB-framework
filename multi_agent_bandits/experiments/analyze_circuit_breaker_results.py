import csv
import math
import os
from collections import defaultdict


INPUT_PATH = "results/final_market_density_linear/summary_results.csv"
OUTPUT_DIR = "results/final_market_density_linear"

AGGREGATED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "aggregated_results.csv")
AGENT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "agent_results.csv")


METRICS = [
    "avg_global_reward",
    "avg_reward_per_agent_per_step",
    "reward_volatility",
    "total_collisions",
    "avg_collisions_per_step",
    "total_halted_choices",
    "total_triggers",
    "gini_total_rewards",
    "market_wide_halt_steps",
    "avg_available_arms",
]


def safe_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    values = [v for v in values if v is not None]

    if not values:
        return None

    return sum(values) / len(values)


def std(values):
    values = [v for v in values if v is not None]

    if len(values) < 2:
        return 0.0

    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)

    return math.sqrt(variance)


def ci95(values):
    """
    95% confidence interval half-width:
        1.96 * standard_error
    """
    values = [v for v in values if v is not None]

    if len(values) < 2:
        return 0.0

    return 1.96 * std(values) / math.sqrt(len(values))


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def infer_baseline_name(row):
    """
    Find the correct baseline for a condition.

    Main linear-share experiment:
        balanced_cb_... -> balanced_baseline

    Collision-policy experiment:
        balanced_zero_on_collision_cb_... -> balanced_zero_on_collision_baseline
        high_congestion_linear_share_cb_... -> high_congestion_linear_share_baseline
    """

    condition_name = row["condition_name"]

    if condition_name.endswith("_baseline"):
        return condition_name

    market_name = row.get("market_name", "")
    collision_policy = row.get("collision_policy", "")

    if condition_name.startswith(f"{market_name}_cb_"):
        return f"{market_name}_baseline"

    if collision_policy:
        return f"{market_name}_{collision_policy}_baseline"

    return f"{market_name}_baseline"


def aggregate_conditions(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["condition_name"]].append(row)

    aggregated_rows = []

    # First pass: calculate ordinary aggregated metrics.
    for condition_name, condition_rows in grouped.items():
        first = condition_rows[0]

        output_row = {
            "condition_name": condition_name,
            "condition_type": first.get("condition_type", ""),
            "market_name": first.get("market_name", ""),
            "collision_policy": first.get("collision_policy", ""),
            "n_seeds": len(condition_rows),
            "steps": first.get("steps", ""),
            "n_agents": first.get("n_agents", ""),
            "n_arms": first.get("n_arms", ""),
            "threshold": first.get("threshold", ""),
            "threshold_ratio": first.get("threshold_ratio", ""),
            "halt_duration": first.get("halt_duration", ""),
            "halted_reward": first.get("halted_reward", ""),
            "transparent": first.get("transparent", ""),
        }

        for metric in METRICS:
            values = [safe_float(row.get(metric)) for row in condition_rows]

            output_row[f"{metric}_mean"] = mean(values)
            output_row[f"{metric}_std"] = std(values)
            output_row[f"{metric}_ci95"] = ci95(values)

        aggregated_rows.append(output_row)

    # Build lookup so we can calculate relative efficiency.
    reward_lookup = {
        row["condition_name"]: safe_float(row["avg_reward_per_agent_per_step_mean"])
        for row in aggregated_rows
    }

    # Second pass: calculate relative efficiency.
    for row in aggregated_rows:
        baseline_name = infer_baseline_name(row)
        baseline_reward = reward_lookup.get(baseline_name)
        condition_reward = safe_float(row["avg_reward_per_agent_per_step_mean"])

        row["baseline_condition"] = baseline_name

        if baseline_reward is None or baseline_reward == 0 or condition_reward is None:
            row["relative_efficiency"] = None
        else:
            row["relative_efficiency"] = condition_reward / baseline_reward

    return aggregated_rows


def aggregate_agents(rows):
    """
    Aggregates per-agent rewards by condition_name and agent_name.

    This works with variable numbers of agents because it checks all
    agent_*_reward and agent_*_name columns that exist in the CSV.
    """

    grouped = defaultdict(list)

    for row in rows:
        condition_name = row["condition_name"]

        for key, value in row.items():
            if key.startswith("agent_") and key.endswith("_reward"):
                parts = key.split("_")

                # Example key: agent_0_reward
                agent_index = parts[1]
                name_key = f"agent_{agent_index}_name"

                agent_name = row.get(name_key, "")

                if agent_name == "":
                    continue

                reward = safe_float(value)

                if reward is None:
                    continue

                grouped[(condition_name, agent_name)].append(reward)

    output_rows = []

    for (condition_name, agent_name), rewards in grouped.items():
        output_rows.append({
            "condition_name": condition_name,
            "agent_name": agent_name,
            "mean_reward": mean(rewards),
            "std_reward": std(rewards),
            "ci95_reward": ci95(rewards),
            "n_observations": len(rewards),
        })

    return output_rows


def write_csv(rows, path):
    if not rows:
        print(f"No rows to write for {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
):
    aggregated_output_path = os.path.join(output_dir, "aggregated_results.csv")
    agent_output_path = os.path.join(output_dir, "agent_results.csv")

    rows = read_rows(input_path)

    aggregated_rows = aggregate_conditions(rows)
    agent_rows = aggregate_agents(rows)

    write_csv(aggregated_rows, aggregated_output_path)
    write_csv(agent_rows, agent_output_path)

    print(f"Read raw results from: {input_path}")
    print(f"Saved aggregated results to: {aggregated_output_path}")
    print(f"Saved agent results to: {agent_output_path}")


if __name__ == "__main__":
    main()