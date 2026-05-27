import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT FILES
# ============================================================

LINEAR_SHARE_FILE = "results/final_market_density_linear/aggregated_results.csv"
COLLISION_FILE = "results/collision_policy_comparison/aggregated_results.csv"

OUTPUT_DIR = "results/final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_results(path):
    df = pd.read_csv(path)

    numeric_cols = [
        "n_seeds",
        "steps",
        "n_agents",
        "n_arms",
        "threshold",
        "threshold_ratio",
        "halt_duration",
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
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "collision_policy" not in df.columns:
        df["collision_policy"] = "linear_share"

    df["collision_policy"] = df["collision_policy"].fillna("linear_share")

    df["transparent"] = df["transparent"].astype(str).str.strip()

    df.loc[df["transparent"] == "True", "transparent_clean"] = True
    df.loc[df["transparent"] == "False", "transparent_clean"] = False
    df.loc[df["transparent"].isin(["nan", "", "NaN"]), "transparent_clean"] = None

    return df


def recompute_relative_efficiency(df):
    """
    Recomputes relative efficiency safely.

    Relative efficiency =
        condition avg reward per agent per step /
        matching baseline avg reward per agent per step

    Matching baseline is based on:
        market_name + collision_policy
    """

    df = df.copy()

    baseline_lookup = {}

    baselines = df[df["condition_type"] == "baseline"]

    for _, row in baselines.iterrows():
        key = (row["market_name"], row["collision_policy"])
        baseline_lookup[key] = row["avg_reward_per_agent_per_step_mean"]

    computed_values = []

    for _, row in df.iterrows():
        key = (row["market_name"], row["collision_policy"])
        baseline_value = baseline_lookup.get(key)

        if baseline_value is None or pd.isna(baseline_value) or baseline_value == 0:
            computed_values.append(None)
        else:
            computed_values.append(
                row["avg_reward_per_agent_per_step_mean"] / baseline_value
            )

    df["relative_efficiency_fixed"] = computed_values

    return df


def load_all():
    linear_df = load_results(LINEAR_SHARE_FILE)
    collision_df = load_results(COLLISION_FILE)

    linear_df = recompute_relative_efficiency(linear_df)
    collision_df = recompute_relative_efficiency(collision_df)

    print("Loaded files:")
    print(f"Linear-share rows: {len(linear_df)}")
    print(f"Collision rows:    {len(collision_df)}")
    print()

    print("Linear-share relative efficiency check:")
    print(
        linear_df[
            [
                "condition_name",
                "market_name",
                "collision_policy",
                "condition_type",
                "threshold_ratio",
                "halt_duration",
                "transparent",
                "relative_efficiency",
                "relative_efficiency_fixed",
            ]
        ].head(20)
    )

    return linear_df, collision_df


# ============================================================
# HELPERS
# ============================================================

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def nice_name(x):
    return str(x).replace("_", " ").title()


def cb_only(df):
    return df[df["condition_type"] == "circuit_breaker"].copy()


def baseline_only(df):
    return df[df["condition_type"] == "baseline"].copy()


def yerr(df, mean_col):
    ci_col = mean_col.replace("_mean", "_ci95")
    if ci_col in df.columns:
        return df[ci_col]
    return None


def transparency_label(value):
    if value is True:
        return "Transparent"
    if value is False:
        return "Opaque"
    return "Baseline"


def condition_label(row):
    if row["condition_type"] == "baseline":
        return "Baseline"

    mode = transparency_label(row["transparent_clean"])
    ratio = row["threshold_ratio"]
    halt = int(row["halt_duration"])

    return f"{mode}\nratio={ratio}, halt={halt}"


# ============================================================
# FIGURE 1A — BASELINE REWARD BY MARKET DENSITY
# ============================================================

def figure_1a_baseline_reward(linear_df):
    data = baseline_only(linear_df).copy()

    order = ["low_congestion", "balanced", "high_congestion"]
    data["order"] = data["market_name"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("order")

    plt.figure(figsize=(8, 5))
    plt.bar(
        data["market_name"].map(nice_name),
        data["avg_reward_per_agent_per_step_mean"],
        yerr=yerr(data, "avg_reward_per_agent_per_step_mean"),
        capsize=4,
    )

    plt.title("Figure 1A — Baseline reward by market density")
    plt.xlabel("Market setting")
    plt.ylabel("Average reward per agent per step")

    savefig("figure_1a_baseline_reward_by_density.png")


# ============================================================
# FIGURE 1B — BASELINE COLLISIONS BY MARKET DENSITY
# ============================================================

def figure_1b_baseline_collisions(linear_df):
    data = baseline_only(linear_df).copy()

    order = ["low_congestion", "balanced", "high_congestion"]
    data["order"] = data["market_name"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("order")

    plt.figure(figsize=(8, 5))
    plt.bar(
        data["market_name"].map(nice_name),
        data["avg_collisions_per_step_mean"],
        yerr=yerr(data, "avg_collisions_per_step_mean"),
        capsize=4,
    )

    plt.title("Figure 1B — Baseline collisions by market density")
    plt.xlabel("Market setting")
    plt.ylabel("Average collisions per step")

    savefig("figure_1b_baseline_collisions_by_density.png")


# ============================================================
# FIGURE 2 — MAIN CIRCUIT BREAKER HEATMAPS
# ============================================================

def figure_2_heatmaps(linear_df):
    data = cb_only(linear_df)

    data = data.dropna(
        subset=[
            "market_name",
            "threshold_ratio",
            "halt_duration",
            "transparent_clean",
            "relative_efficiency_fixed",
        ]
    )

    if data.empty:
        print("Figure 2 skipped: no valid circuit-breaker rows.")
        return

    for market in sorted(data["market_name"].unique()):
        for transparent_value in [False, True]:
            subset = data[
                (data["market_name"] == market)
                & (data["transparent_clean"] == transparent_value)
            ].copy()

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="threshold_ratio",
                columns="halt_duration",
                values="relative_efficiency_fixed",
                aggfunc="mean",
            )

            pivot = pivot.sort_index().sort_index(axis=1)

            plt.figure(figsize=(7, 5))
            plt.imshow(pivot.values, aspect="auto")

            plt.xticks(
                range(len(pivot.columns)),
                [str(int(x)) for x in pivot.columns],
            )
            plt.yticks(
                range(len(pivot.index)),
                [str(x) for x in pivot.index],
            )

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.values[i, j]
                    if pd.notna(value):
                        plt.text(j, i, f"{value:.2f}", ha="center", va="center")

            mode = "transparent" if transparent_value else "opaque"

            plt.title(f"Figure 2 — Relative efficiency heatmap: {nice_name(market)} ({mode})")
            plt.xlabel("Halt duration")
            plt.ylabel("Threshold ratio")
            plt.colorbar(label="Relative efficiency")

            savefig(f"figure_2_heatmap_{market}_{mode}.png")


# ============================================================
# FIGURE 3 — TRANSPARENT COLLAPSE
# ============================================================

def figure_3_transparent_collapse(linear_df):
    data = cb_only(linear_df)

    data = data[
        data["transparent_clean"] == True
    ].copy()

    data = data.dropna(
        subset=[
            "market_name",
            "threshold_ratio",
            "halt_duration",
            "relative_efficiency_fixed",
        ]
    )

    if data.empty:
        print("Figure 3 skipped: no valid transparent circuit-breaker rows.")
        return

    plt.figure(figsize=(10, 6))

    for market in sorted(data["market_name"].unique()):
        for ratio in sorted(data["threshold_ratio"].unique()):
            subset = data[
                (data["market_name"] == market)
                & (data["threshold_ratio"] == ratio)
            ].copy()

            subset = subset.sort_values("halt_duration")

            if subset.empty:
                continue

            plt.errorbar(
                subset["halt_duration"],
                subset["relative_efficiency_fixed"],
                yerr=subset["avg_reward_per_agent_per_step_ci95"]
                / subset["avg_reward_per_agent_per_step_mean"]
                * subset["relative_efficiency_fixed"],
                marker="o",
                capsize=4,
                linewidth=2,
                label=f"{nice_name(market)}, ratio={ratio}",
            )

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title("Figure 3 — Collapse under transparent circuit breakers")
    plt.xlabel("Halt duration")
    plt.ylabel("Relative efficiency")
    plt.xticks([3, 5, 10])
    plt.legend()

    savefig("figure_3_transparent_collapse.png")


# ============================================================
# FIGURE 4 — HALTED-ARM CHOICES
# ============================================================

def figure_4_halted_choices(linear_df, market="balanced", ratio=0.6):
    data = cb_only(linear_df)

    data = data[
        (data["market_name"] == market)
        & (data["threshold_ratio"] == ratio)
    ].copy()

    data = data.dropna(subset=["halt_duration", "transparent_clean"])
    data = data.sort_values(["transparent_clean", "halt_duration"])

    if data.empty:
        print("Figure 4 skipped: no matching rows.")
        return

    labels = data.apply(condition_label, axis=1)

    plt.figure(figsize=(10, 5))
    plt.bar(
        labels,
        data["total_halted_choices_mean"],
        yerr=yerr(data, "total_halted_choices_mean"),
        capsize=4,
    )

    plt.title(f"Figure 4 — Halted-arm choices: {nice_name(market)}, ratio={ratio}")
    plt.xlabel("Condition")
    plt.ylabel("Total halted-arm choices")
    plt.xticks(rotation=45, ha="right")

    savefig(f"figure_4_halted_choices_{market}_ratio{ratio}.png")


# ============================================================
# FIGURE 5 — TRIGGER COUNTS
# ============================================================

def figure_5_trigger_counts(linear_df, market="balanced"):
    data = cb_only(linear_df)

    data = data[data["market_name"] == market].copy()
    data = data.dropna(subset=["threshold_ratio", "halt_duration", "transparent_clean"])
    data = data.sort_values(["threshold_ratio", "halt_duration", "transparent_clean"])

    if data.empty:
        print("Figure 5 skipped: no matching rows.")
        return

    labels = data.apply(condition_label, axis=1)

    plt.figure(figsize=(12, 5))
    plt.bar(
        labels,
        data["total_triggers_mean"],
        yerr=yerr(data, "total_triggers_mean"),
        capsize=4,
    )

    plt.title(f"Figure 5 — Circuit breaker trigger count: {nice_name(market)}")
    plt.xlabel("Condition")
    plt.ylabel("Total trigger count")
    plt.xticks(rotation=45, ha="right")

    savefig(f"figure_5_trigger_counts_{market}.png")


# ============================================================
# FIGURE 6 — COLLISION POLICY ROBUSTNESS
# ============================================================

def figure_6_collision_policy_robustness(collision_df, market="balanced"):
    data = collision_df.copy()

    data = data[data["market_name"] == market].copy()

    selected = []

    for _, row in data.iterrows():
        if row["condition_type"] == "baseline":
            selected.append("Baseline")
        elif row["threshold_ratio"] == 0.8 and row["halt_duration"] == 3:
            selected.append("Mild breaker\nratio=0.8, halt=3")
        elif row["threshold_ratio"] == 0.6 and row["halt_duration"] == 5:
            selected.append("Strict breaker\nratio=0.6, halt=5")
        else:
            selected.append(None)

    data["setting"] = selected
    data = data.dropna(subset=["setting"])

    order = [
        ("linear_share", "Baseline"),
        ("linear_share", "Mild breaker\nratio=0.8, halt=3"),
        ("linear_share", "Strict breaker\nratio=0.6, halt=5"),
        ("zero_on_collision", "Baseline"),
        ("zero_on_collision", "Mild breaker\nratio=0.8, halt=3"),
        ("zero_on_collision", "Strict breaker\nratio=0.6, halt=5"),
    ]

    data["order"] = data.apply(
        lambda row: order.index((row["collision_policy"], row["setting"]))
        if (row["collision_policy"], row["setting"]) in order
        else None,
        axis=1,
    )

    data = data.dropna(subset=["order"]).sort_values("order")

    labels = data.apply(
        lambda row: f"{row['collision_policy']}\n{row['setting']}",
        axis=1,
    )

    plt.figure(figsize=(11, 5))
    plt.bar(labels, data["relative_efficiency_fixed"])
    plt.axhline(1.0, linestyle="--", linewidth=1)

    plt.title(f"Figure 6 — Collision policy robustness: {nice_name(market)}")
    plt.xlabel("Condition")
    plt.ylabel("Relative efficiency")
    plt.xticks(rotation=45, ha="right")

    savefig(f"figure_6_collision_policy_robustness_{market}.png")


# ============================================================
# FIGURE 7 — AVERAGE AVAILABLE ARMS
# ============================================================

def figure_7_available_arms(linear_df, market="balanced"):
    data = cb_only(linear_df)

    data = data[data["market_name"] == market].copy()
    data = data.dropna(subset=["threshold_ratio", "halt_duration", "transparent_clean"])
    data = data.sort_values(["threshold_ratio", "halt_duration", "transparent_clean"])

    if data.empty:
        print("Figure 7 skipped: no matching rows.")
        return

    labels = data.apply(condition_label, axis=1)

    plt.figure(figsize=(12, 5))
    plt.bar(
        labels,
        data["avg_available_arms_mean"],
        yerr=yerr(data, "avg_available_arms_mean"),
        capsize=4,
    )

    plt.title(f"Figure 7 — Average available arms: {nice_name(market)}")
    plt.xlabel("Condition")
    plt.ylabel("Average available arms")
    plt.xticks(rotation=45, ha="right")

    savefig(f"figure_7_available_arms_{market}.png")


# ============================================================
# MAIN
# ============================================================

def main():
    linear_df, collision_df = load_all()

    figure_1a_baseline_reward(linear_df)
    figure_1b_baseline_collisions(linear_df)

    figure_2_heatmaps(linear_df)
    figure_3_transparent_collapse(linear_df)

    figure_4_halted_choices(linear_df, market="balanced", ratio=0.6)
    figure_5_trigger_counts(linear_df, market="balanced")
    figure_6_collision_policy_robustness(collision_df, market="balanced")
    figure_7_available_arms(linear_df, market="balanced")

    print()
    print("Done.")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()