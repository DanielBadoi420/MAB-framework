import pandas as pd

INPUT_PATH = "results/collision_policy_comparison/summary_results.csv"
OUTPUT_PATH_1 = "results/collision_policy_comparison/aggregated_results.csv"
OUTPUT_PATH_2 = "results/collision_policy_comparison/agent_results.csv"

df = pd.read_csv(INPUT_PATH)

metrics = [
    "avg_global_reward",
    "avg_reward_per_agent_per_step",
    "reward_volatility",
    "total_collisions",
    "avg_collisions_per_step",
    "total_halted_choices",
    "total_triggers",
    "gini_total_rewards",
    "market_wide_halt_steps",
    "avg_available_arms"
]

agent_columns = [
    ("RandomAgent", "agent_0_reward"),
    ("EpsilonGreedyAgent", "agent_1_reward"),
    ("UCB_BaselineAgent", "agent_2_reward"),
    ("RiskAverseEpsilonGreedyAgent", "agent_3_reward"),
    ("RiskAverseUCBAgent", "agent_4_reward"),
]

rows = []

for condition_name, condition_df in df.groupby("condition_name"):
    for agent_name, reward_col in agent_columns:
        rows.append({
            "condition_name": condition_name,
            "agent_name": agent_name,
            "mean_reward": condition_df[reward_col].mean(),
            "std_reward": condition_df[reward_col].std(),
        })

summary = df.groupby("condition_name")[metrics].agg(["mean", "std"])

#flatten column names
summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(OUTPUT_PATH_1, index=False)

print(summary.to_string(index=False))
print(f"\nSaved aggregated results to: {OUTPUT_PATH_1}")

agent_summary = pd.DataFrame(rows)
agent_summary.to_csv(OUTPUT_PATH_2, index=False)

print(agent_summary.to_string(index=False))
print(f"\nSaved agent results to: {OUTPUT_PATH_2}")