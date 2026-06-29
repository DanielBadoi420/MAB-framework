import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Helpers

MARKET_ORDER = ["low_congestion", "balanced", "high_congestion"]
MARKET_LABELS = {
    "low_congestion": "Low congestion",
    "balanced": "Balanced",
    "high_congestion": "High congestion",
}

MODE_LABELS = {
    False: "Opaque",
    True: "Transparent",
}

RATIO_ORDER = [0.6, 0.8]
HALT_ORDER = [3, 5, 10]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_bool(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ["true", "1", "yes"]:
        return True
    if s in ["false", "0", "no"]:
        return False
    return np.nan


def load_results(path):
    df = pd.read_csv(path)

    #convert numeric columns safely
    numeric_cols = [
        "threshold",
        "threshold_ratio",
        "halt_duration",
        "halted_reward",
        "avg_global_reward_mean",
        "avg_global_reward_std",
        "avg_global_reward_ci95",
        "avg_reward_per_agent_per_step_mean",
        "avg_reward_per_agent_per_step_std",
        "avg_reward_per_agent_per_step_ci95",
        "reward_volatility_mean",
        "reward_volatility_std",
        "reward_volatility_ci95",
        "total_collisions_mean",
        "total_collisions_std",
        "total_collisions_ci95",
        "avg_collisions_per_step_mean",
        "avg_collisions_per_step_std",
        "avg_collisions_per_step_ci95",
        "total_halted_choices_mean",
        "total_halted_choices_std",
        "total_halted_choices_ci95",
        "total_triggers_mean",
        "total_triggers_std",
        "total_triggers_ci95",
        "gini_total_rewards_mean",
        "gini_total_rewards_std",
        "gini_total_rewards_ci95",
        "market_wide_halt_steps_mean",
        "market_wide_halt_steps_std",
        "market_wide_halt_steps_ci95",
        "avg_available_arms_mean",
        "avg_available_arms_std",
        "avg_available_arms_ci95",
        "relative_efficiency",
        "n_agents",
        "n_arms",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "transparent" in df.columns:
        df["transparent"] = df["transparent"].apply(clean_bool)

    return df


def fill_relative_efficiency(df):
    """
    Fill missing relative_efficiency values using: avg_global_reward_mean / baseline avg_global_reward_mean matched by market_name + collision_policy.
    """
    if "relative_efficiency" not in df.columns:
        df["relative_efficiency"] = np.nan

    baseline_rows = df[df["condition_type"] == "baseline"].copy()

    baseline_map = {}
    for _, row in baseline_rows.iterrows():
        key = (row.get("market_name"), row.get("collision_policy"))
        baseline_map[key] = row.get("avg_global_reward_mean")

    def compute_rel(row):
        if pd.notna(row["relative_efficiency"]):
            return row["relative_efficiency"]

        key = (row.get("market_name"), row.get("collision_policy"))
        baseline_val = baseline_map.get(key)

        if pd.isna(baseline_val) or baseline_val == 0:
            return np.nan
        if pd.isna(row.get("avg_global_reward_mean")):
            return np.nan

        return row["avg_global_reward_mean"] / baseline_val

    df["relative_efficiency"] = df.apply(compute_rel, axis=1)
    return df


def short_condition_label(row):
    if row["condition_type"] == "baseline":
        return "Baseline"
    ratio = row["threshold_ratio"]
    halt = row["halt_duration"]
    return f"r={ratio:.1f}, h={int(halt)}"


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()



#Figure 1a / 1b
def plot_h1_density(linear_df, outdir):
    baseline = linear_df[
        (linear_df["condition_type"] == "baseline") &
        (linear_df["collision_policy"] == "linear_share")
    ].copy()

    baseline["market_name"] = pd.Categorical(
        baseline["market_name"],
        categories=MARKET_ORDER,
        ordered=True
    )
    baseline = baseline.sort_values("market_name")

    labels = [MARKET_LABELS[m] for m in baseline["market_name"]]
    x = np.arange(len(labels))

    #figure 1a: reward per agent per timestep
    plt.figure(figsize=(8, 5))
    y = baseline["avg_reward_per_agent_per_step_mean"].values
    yerr = baseline["avg_reward_per_agent_per_step_ci95"].values \
        if "avg_reward_per_agent_per_step_ci95" in baseline.columns else None

    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Average reward per agent per timestep")
    savefig(os.path.join(outdir, "figure_1a_density_reward.png"))

    #figure 1b: collisions per step
    plt.figure(figsize=(8, 5))
    y = baseline["avg_collisions_per_step_mean"].values
    yerr = baseline["avg_collisions_per_step_ci95"].values \
        if "avg_collisions_per_step_ci95" in baseline.columns else None

    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Average collisions per step")
    savefig(os.path.join(outdir, "figure_1b_density_collisions.png"))



#Figure 2
def plot_h2_strictness(linear_df, outdir):
    df = linear_df[
        (linear_df["condition_type"] == "circuit_breaker") &
        (linear_df["collision_policy"] == "linear_share") &
        (linear_df["transparent"] == True)
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, market in zip(axes, MARKET_ORDER):
        sub = df[df["market_name"] == market].copy()

        for ratio in RATIO_ORDER:
            s = sub[sub["threshold_ratio"] == ratio].copy()
            s = s.sort_values("halt_duration")
            if len(s) == 0:
                continue

            ax.errorbar(
                s["halt_duration"],
                s["relative_efficiency"],
                yerr=None,
                marker="o",
                capsize=4,
                label=f"ratio={ratio:.1f}"
            )

        ax.axhline(1.0, linestyle="--")
        ax.set_title(MARKET_LABELS[market])
        ax.set_xlabel("Halt duration")
        ax.set_xticks(HALT_ORDER)

    axes[0].set_ylabel("Relative efficiency")
    axes[-1].legend()

    savefig(os.path.join(outdir, "figure_2_strictness_transparent_efficiency.png"))



#Figure 3
def plot_h3_mode_efficiency(linear_df, outdir):
    df = linear_df[
        (linear_df["condition_type"] == "circuit_breaker") &
        (linear_df["collision_policy"] == "linear_share") &
        (linear_df["market_name"] == "balanced")
    ].copy()

    df = df.sort_values(["threshold_ratio", "halt_duration", "transparent"])

    conditions = []
    opaque_vals = []
    transparent_vals = []

    for ratio in RATIO_ORDER:
        for halt in HALT_ORDER:
            cond = df[
                (df["threshold_ratio"] == ratio) &
                (df["halt_duration"] == halt)
            ]

            op = cond[cond["transparent"] == False]
            tr = cond[cond["transparent"] == True]

            if len(op) == 0 or len(tr) == 0:
                continue

            conditions.append(f"r={ratio:.1f}\nh={halt}")
            opaque_vals.append(op.iloc[0]["relative_efficiency"])
            transparent_vals.append(tr.iloc[0]["relative_efficiency"])

    x = np.arange(len(conditions))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, opaque_vals, width, label="Opaque")
    plt.bar(x + width / 2, transparent_vals, width, label="Transparent")
    plt.axhline(1.0, linestyle="--")
    plt.xticks(x, conditions)
    plt.ylabel("Relative efficiency")
    plt.xlabel("Condition")
    plt.legend(loc="lower left")
    savefig(os.path.join(outdir, "figure_3_mode_comparison_efficiency_balanced.png"))






#Figure 4
def plot_h3_halted_choices(linear_df, outdir):
    df = linear_df[
        (linear_df["condition_type"] == "circuit_breaker") &
        (linear_df["collision_policy"] == "linear_share") &
        (linear_df["market_name"] == "balanced")
    ].copy()

    conditions = []
    opaque_vals = []
    transparent_vals = []

    for ratio in RATIO_ORDER:
        for halt in HALT_ORDER:
            cond = df[
                (df["threshold_ratio"] == ratio) &
                (df["halt_duration"] == halt)
            ]

            op = cond[cond["transparent"] == False]
            tr = cond[cond["transparent"] == True]

            if len(op) == 0 or len(tr) == 0:
                continue

            conditions.append(f"r={ratio:.1f}\nh={halt}")
            opaque_vals.append(op.iloc[0]["total_halted_choices_mean"])
            transparent_vals.append(tr.iloc[0]["total_halted_choices_mean"])

    x = np.arange(len(conditions))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, opaque_vals, width, label="Opaque")
    plt.bar(x + width / 2, transparent_vals, width, label="Transparent")
    plt.xticks(x, conditions)
    plt.ylabel("Total halted-arm choices")
    plt.xlabel("Condition")
    plt.legend()
    savefig(os.path.join(outdir, "figure_4_halted_choices_balanced.png"))



#Figure 5
def plot_h2_h3_available_arms(linear_df, outdir):
    df = linear_df[
        (linear_df["condition_type"] == "circuit_breaker") &
        (linear_df["collision_policy"] == "linear_share") &
        (linear_df["transparent"] == True)
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, market in zip(axes, MARKET_ORDER):
        sub = df[df["market_name"] == market].copy()

        for ratio in RATIO_ORDER:
            s = sub[sub["threshold_ratio"] == ratio].copy()
            s = s.sort_values("halt_duration")
            if len(s) == 0:
                continue

            ax.errorbar(
                s["halt_duration"],
                s["avg_available_arms_mean"],
                yerr=s["avg_available_arms_ci95"] if "avg_available_arms_ci95" in s.columns else None,
                marker="o",
                capsize=4,
                label=f"ratio={ratio:.1f}"
            )

        ax.set_title(MARKET_LABELS[market])
        ax.set_xlabel("Halt duration")
        ax.set_xticks(HALT_ORDER)

    axes[0].set_ylabel("Average available arms")
    axes[-1].legend()

    savefig(os.path.join(outdir, "figure_5_available_arms_transparent.png"))



#Figure 6a / 6b
def plot_h4_collision_policy(collision_df, outdir):
    selected = [
        ("baseline", None, None, None, "Baseline"),
        ("circuit_breaker", False, 0.8, 3, "Opaque\nr=0.8,h=3"),
        ("circuit_breaker", True, 0.8, 3, "Transparent\nr=0.8,h=3"),
        ("circuit_breaker", False, 0.6, 5, "Opaque\nr=0.6,h=5"),
        ("circuit_breaker", True, 0.6, 5, "Transparent\nr=0.6,h=5"),
    ]

    markets = ["balanced", "high_congestion"]
    policies = ["linear_share", "zero_on_collision"]
    policy_labels = {
        "linear_share": "Linear share",
        "zero_on_collision": "Zero on collision",
    }

    #figure 6a: relative efficiency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, market in zip(axes, markets):
        x = np.arange(len(selected))
        width = 0.35

        for i, policy in enumerate(policies):
            vals = []
            for condition_type, transparent, ratio, halt, _ in selected:
                sub = collision_df[
                    (collision_df["market_name"] == market) &
                    (collision_df["collision_policy"] == policy) &
                    (collision_df["condition_type"] == condition_type)
                ].copy()

                if condition_type == "circuit_breaker":
                    sub = sub[
                        (sub["transparent"] == transparent) &
                        (sub["threshold_ratio"] == ratio) &
                        (sub["halt_duration"] == halt)
                    ]

                vals.append(sub.iloc[0]["relative_efficiency"] if len(sub) > 0 else np.nan)

            ax.bar(x + (i - 0.5) * width, vals, width, label=policy_labels[policy])

        ax.axhline(1.0, linestyle="--")
        ax.set_title(MARKET_LABELS[market])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, _, _, _, lab in selected])

    axes[0].set_ylabel("Relative efficiency")
    axes[0].legend()
    savefig(os.path.join(outdir, "figure_6a_collision_policy_efficiency.png"))

    #Figure 6b: inequality (Gini)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, market in zip(axes, markets):
        x = np.arange(len(selected))
        width = 0.35

        for i, policy in enumerate(policies):
            vals = []
            for condition_type, transparent, ratio, halt, _ in selected:
                sub = collision_df[
                    (collision_df["market_name"] == market) &
                    (collision_df["collision_policy"] == policy) &
                    (collision_df["condition_type"] == condition_type)
                ].copy()

                if condition_type == "circuit_breaker":
                    sub = sub[
                        (sub["transparent"] == transparent) &
                        (sub["threshold_ratio"] == ratio) &
                        (sub["halt_duration"] == halt)
                    ]

                vals.append(sub.iloc[0]["gini_total_rewards_mean"] if len(sub) > 0 else np.nan)

            ax.bar(x + (i - 0.5) * width, vals, width, label=policy_labels[policy])

        ax.set_title(MARKET_LABELS[market])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, _, _, _, lab in selected])

    axes[0].set_ylabel("Gini coefficient of total rewards")
    axes[0].legend()
    savefig(os.path.join(outdir, "figure_6b_collision_policy_gini.png"))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-agg", required=True, help="Path to linear-share aggregated_results.csv")
    parser.add_argument("--collision-agg", required=True, help="Path to collision comparison aggregated_results.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out)

    linear_df = load_results(args.linear_agg)
    collision_df = load_results(args.collision_agg)

    linear_df = fill_relative_efficiency(linear_df)
    collision_df = fill_relative_efficiency(collision_df)

    plot_h1_density(linear_df, args.out)
    plot_h2_strictness(linear_df, args.out)
    plot_h3_mode_efficiency(linear_df, args.out)
    plot_h3_halted_choices(linear_df, args.out)
    plot_h2_h3_available_arms(linear_df, args.out)
    plot_h4_collision_policy(collision_df, args.out)

    print("\nSaved figures to:", args.out)
    print("figure_1a_density_reward.png")
    print("figure_1b_density_collisions.png")
    print("figure_2_strictness_transparent_efficiency.png")
    print("figure_3_mode_comparison_efficiency_balanced.png")
    print("figure_4_halted_choices_balanced.png")
    print("figure_5_available_arms_transparent.png")
    print("figure_6a_collision_policy_efficiency.png")
    print("figure_6b_collision_policy_gini.png")


if __name__ == "__main__":
    main()