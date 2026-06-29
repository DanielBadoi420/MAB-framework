import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal, wilcoxon


def clean_bool(x):
    if pd.isna(x) or x == "":
        return np.nan
    return str(x).strip().lower() == "true"


def load_csv(path):
    df = pd.read_csv(path)

    if "transparent" in df.columns:
        df["transparent"] = df["transparent"].apply(clean_bool)

    numeric_cols = [
        "seed", "threshold_ratio", "halt_duration",
        "avg_reward_per_agent_per_step",
        "avg_collisions_per_step",
        "total_triggers",
        "total_halted_choices",
        "avg_available_arms",
        "gini_total_rewards",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def holm_correction(p_values):
    """
    Simple Holm correction.
    Returns adjusted p-values in the original order.
    """
    p_values = np.array(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n)

    running_max = 0

    for rank, idx in enumerate(order):
        corrected = (n - rank) * p_values[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def add_result(results, hypothesis, comparison, metric, test, statistic, p_value, note):
    if p_value is None or pd.isna(p_value):
        significant = False
    else:
        significant = p_value < 0.05

    results.append({
        "hypothesis": hypothesis,
        "comparison": comparison,
        "metric": metric,
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
        "significant_raw_0.05": significant,
        "note": note,
    })


def test_h1_market_density(linear_df, results):
    """
    H1:
    Compare baseline low, balanced, high markets.
    Kruskal-Wallis is used because there are three independent groups.
    """
    baseline = linear_df[
        (linear_df["condition_type"] == "baseline") &
        (linear_df["collision_policy"] == "linear_share")
    ]

    metrics = [
        "avg_reward_per_agent_per_step",
        "avg_collisions_per_step",
    ]

    for metric in metrics:
        groups = []

        for market in ["low_congestion", "balanced", "high_congestion"]:
            values = baseline[baseline["market_name"] == market][metric].dropna()
            if len(values) > 0:
                groups.append(values)

        if len(groups) == 3:
            stat, p = kruskal(*groups)
            add_result(
                results,
                "H1",
                "Baseline market density: low vs balanced vs high",
                metric,
                "Kruskal-Wallis",
                stat,
                p,
                "Tests whether baseline market density changes this metric."
            )


def test_h2_strictness(linear_df, results):
    """
    H2:
    Compare mild transparent breaker with strict transparent breaker.
    Paired by seed within each market.
    """
    metrics = [
        "avg_reward_per_agent_per_step",
        "avg_available_arms",
        "total_triggers",
    ]

    for market in ["low_congestion", "balanced", "high_congestion"]:
        mild = linear_df[
            (linear_df["market_name"] == market) &
            (linear_df["condition_type"] == "circuit_breaker") &
            (linear_df["transparent"] == True) &
            (linear_df["threshold_ratio"] == 0.8) &
            (linear_df["halt_duration"] == 3)
        ]

        strict = linear_df[
            (linear_df["market_name"] == market) &
            (linear_df["condition_type"] == "circuit_breaker") &
            (linear_df["transparent"] == True) &
            (linear_df["threshold_ratio"] == 0.6) &
            (linear_df["halt_duration"] == 5)
        ]

        for metric in metrics:
            paired = mild[["seed", metric]].merge(
                strict[["seed", metric]],
                on="seed",
                suffixes=("_mild", "_strict")
            ).dropna()

            if len(paired) > 0:
                stat, p = wilcoxon(
                    paired[f"{metric}_mild"],
                    paired[f"{metric}_strict"]
                )

                add_result(
                    results,
                    "H2",
                    f"{market}: mild transparent ratio 0.8 halt 3 vs strict transparent ratio 0.6 halt 5",
                    metric,
                    "Wilcoxon signed-rank",
                    stat,
                    p,
                    "Paired by seed. Tests whether stricter transparent breakers change this metric."
                )


def test_h3_transparent_vs_opaque(linear_df, results):
    """
    H3:
    Compare transparent and opaque modes with same market, threshold ratio, halt duration.
    Paired by seed.
    """
    metrics = [
        "avg_reward_per_agent_per_step",
        "avg_collisions_per_step",
        "total_halted_choices",
        "total_triggers",
        "avg_available_arms",
    ]

    markets = ["low_congestion", "balanced", "high_congestion"]
    ratios = [0.6, 0.8]
    halts = [3, 5, 10]

    for market in markets:
        for ratio in ratios:
            for halt in halts:
                opaque = linear_df[
                    (linear_df["market_name"] == market) &
                    (linear_df["condition_type"] == "circuit_breaker") &
                    (linear_df["transparent"] == False) &
                    (linear_df["threshold_ratio"] == ratio) &
                    (linear_df["halt_duration"] == halt)
                ]

                transparent = linear_df[
                    (linear_df["market_name"] == market) &
                    (linear_df["condition_type"] == "circuit_breaker") &
                    (linear_df["transparent"] == True) &
                    (linear_df["threshold_ratio"] == ratio) &
                    (linear_df["halt_duration"] == halt)
                ]

                for metric in metrics:
                    paired = opaque[["seed", metric]].merge(
                        transparent[["seed", metric]],
                        on="seed",
                        suffixes=("_opaque", "_transparent")
                    ).dropna()

                    if len(paired) > 0:
                        stat, p = wilcoxon(
                            paired[f"{metric}_opaque"],
                            paired[f"{metric}_transparent"]
                        )

                        add_result(
                            results,
                            "H3",
                            f"{market}: opaque vs transparent, ratio {ratio}, halt {halt}",
                            metric,
                            "Wilcoxon signed-rank",
                            stat,
                            p,
                            "Paired by seed. Tests whether intervention mode changes this metric."
                        )


def test_h4_collision_policy(collision_df, results):
    """
    H4:
    Compare linear_share and zero_on_collision.
    Paired by seed and matched condition.
    """
    metrics = [
        "avg_reward_per_agent_per_step",
        "avg_collisions_per_step",
        "gini_total_rewards",
    ]

    condition_keys = [
        "market_name",
        "condition_type",
        "threshold_ratio",
        "halt_duration",
        "transparent",
    ]

    linear = collision_df[collision_df["collision_policy"] == "linear_share"]
    zero = collision_df[collision_df["collision_policy"] == "zero_on_collision"]

    for metric in metrics:
        paired = linear[condition_keys + ["seed", metric]].merge(
            zero[condition_keys + ["seed", metric]],
            on=condition_keys + ["seed"],
            suffixes=("_linear", "_zero")
        ).dropna()

        if len(paired) > 0:
            stat, p = wilcoxon(
                paired[f"{metric}_linear"],
                paired[f"{metric}_zero"]
            )

            add_result(
                results,
                "H4",
                "Linear sharing vs zero-on-collision across matched conditions",
                metric,
                "Wilcoxon signed-rank",
                stat,
                p,
                "Paired by seed and matched condition. Tests whether collision policy changes this metric."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-summary", required=True)
    parser.add_argument("--collision-summary", required=True)
    parser.add_argument("--out", default="thesis_stats")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    linear_df = load_csv(args.linear_summary)
    collision_df = load_csv(args.collision_summary)

    results = []

    test_h1_market_density(linear_df, results)
    test_h2_strictness(linear_df, results)
    test_h3_transparent_vs_opaque(linear_df, results)
    test_h4_collision_policy(collision_df, results)

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df["p_value_holm"] = np.nan
        valid_mask = results_df["p_value"].notna()

        if valid_mask.sum() > 0:
            results_df.loc[valid_mask, "p_value_holm"] = holm_correction(
                results_df.loc[valid_mask, "p_value"]
            )

        results_df["significant_0.05"] = results_df["p_value_holm"] < 0.05

    output_path = os.path.join(args.out, "statistical_tests_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"Saved statistical test results to: {output_path}")


if __name__ == "__main__":
    main()