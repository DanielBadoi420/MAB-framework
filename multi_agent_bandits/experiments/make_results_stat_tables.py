import argparse
import os
import pandas as pd
import numpy as np


def clean_bool(x):
    if pd.isna(x) or x == "":
        return np.nan
    return str(x).strip().lower() == "true"


def load(path):
    df = pd.read_csv(path)
    if "transparent" in df.columns:
        df["transparent"] = df["transparent"].apply(clean_bool)
    return df


def f(x, d=3):
    if pd.isna(x):
        return ""
    return f"{x:.{d}f}"


def p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}"


def ci(mean, ci95):
    if pd.isna(mean) or pd.isna(ci95):
        return ""
    return f"[{mean - ci95:.3f}, {mean + ci95:.3f}]"


def row(df, market, condition_type, policy="linear_share", transparent=None, ratio=None, halt=None):
    sub = df[
        (df["market_name"] == market) &
        (df["condition_type"] == condition_type) &
        (df["collision_policy"] == policy)
    ]

    if transparent is not None:
        sub = sub[sub["transparent"] == transparent]

    if ratio is not None:
        sub = sub[np.isclose(sub["threshold_ratio"], ratio)]

    if halt is not None:
        sub = sub[sub["halt_duration"] == halt]

    return sub.iloc[0]


def test(stats, hyp, metric, text):
    sub = stats[
        (stats["hypothesis"] == hyp) &
        (stats["metric"] == metric) &
        (stats["comparison"].str.contains(text, regex=False, na=False))
    ]
    return sub.iloc[0]


def save(df, out, name):
    df.to_csv(os.path.join(out, name + ".csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", required=True)
    parser.add_argument("--collision", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out", default="simple_results_tables")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    linear = load(args.linear)
    collision = load(args.collision)
    stats = load(args.stats)

    markets = {
        "low_congestion": "Low congestion",
        "balanced": "Balanced",
        "high_congestion": "High congestion",
    }

    # 4.2 / Figure 1a: reward by market density
    rows = []
    for m, label in markets.items():
        r = row(linear, m, "baseline")
        rows.append({
            "Market type": label,
            "Metric": "Average reward per agent per timestep",
            "Mean": f(r["avg_reward_per_agent_per_step_mean"]),
            "95% CI": ci(r["avg_reward_per_agent_per_step_mean"], r["avg_reward_per_agent_per_step_ci95"]),
        })
    save(pd.DataFrame(rows), args.out, "table_4_2_reward_density")

    # 4.2 / Figure 1b: collisions by market density
    rows = []
    for m, label in markets.items():
        r = row(linear, m, "baseline")
        rows.append({
            "Market type": label,
            "Metric": "Average collisions per timestep",
            "Mean": f(r["avg_collisions_per_step_mean"]),
            "95% CI": ci(r["avg_collisions_per_step_mean"], r["avg_collisions_per_step_ci95"]),
        })
    save(pd.DataFrame(rows), args.out, "table_4_2_collision_density")

    # 4.2 statistical tests
    rows = []
    for metric in ["avg_reward_per_agent_per_step", "avg_collisions_per_step"]:
        t = test(stats, "H1", metric, "Baseline market density")
        rows.append({
            "Metric": metric,
            "Test": t["test"],
            "Statistic": f(t["statistic"], 2),
            "Holm-adjusted p": p(t["p_value_holm"]),
        })
    save(pd.DataFrame(rows), args.out, "table_4_2_density_tests")

    # 4.3 strictness: mild transparent 0.8/halt3 vs strict transparent 0.6/halt5
    rows = []
    metrics = {
        "avg_reward_per_agent_per_step": "Average reward per agent",
        "avg_available_arms": "Average available arms",
        "total_triggers": "Total triggers",
    }

    for m, label in markets.items():
        mild = row(linear, m, "circuit_breaker", transparent=True, ratio=0.8, halt=3)
        strict = row(linear, m, "circuit_breaker", transparent=True, ratio=0.6, halt=5)
        comp = f"{m}: mild transparent ratio 0.8 halt 3 vs strict transparent ratio 0.6 halt 5"

        for metric, nice in metrics.items():
            t = test(stats, "H2", metric, comp)
            rows.append({
                "Market type": label,
                "Metric": nice,
                "Mild mean": f(mild[metric + "_mean"]),
                "Strict mean": f(strict[metric + "_mean"]),
                "Statistic": f(t["statistic"], 2),
                "Holm-adjusted p": p(t["p_value_holm"]),
            })

    save(pd.DataFrame(rows), args.out, "table_4_3_strictness")

    # 4.4 transparent vs opaque: key descriptive values from Results text
    rows = []
    for halt in [3, 5, 10]:
        opaque = row(linear, "balanced", "circuit_breaker", transparent=False, ratio=0.6, halt=halt)
        trans = row(linear, "balanced", "circuit_breaker", transparent=True, ratio=0.6, halt=halt)

        rows.append({
            "Setting": f"Balanced, ratio 0.6, halt {halt}",
            "Opaque relative efficiency": f(opaque["relative_efficiency"]),
            "Transparent relative efficiency": f(trans["relative_efficiency"]),
            "Opaque halted choices mean": f(opaque["total_halted_choices_mean"]),
        })

    save(pd.DataFrame(rows), args.out, "table_4_4_mode_descriptives")

    # 4.4 transparent vs opaque: key statistical tests only
    rows = []
    key_settings = [
        ("balanced", "Balanced, ratio 0.6, halt 5", 0.6, 5),
        ("high_congestion", "High congestion, ratio 0.6, halt 10", 0.6, 10),
    ]

    mode_metrics = {
        "avg_reward_per_agent_per_step": "Average reward per agent",
        "avg_collisions_per_step": "Average collisions per timestep",
        "total_halted_choices": "Total halted choices",
        "total_triggers": "Total triggers",
        "avg_available_arms": "Average available arms",
    }

    for m, label, ratio, halt in key_settings:
        comp = f"{m}: opaque vs transparent, ratio {ratio}, halt {halt}"

        for metric, nice in mode_metrics.items():
            t = test(stats, "H3", metric, comp)
            rows.append({
                "Setting": label,
                "Metric": nice,
                "Statistic": f(t["statistic"], 2),
                "Holm-adjusted p": p(t["p_value_holm"]),
            })

    save(pd.DataFrame(rows), args.out, "table_4_4_mode_tests")

    # 4.5 collision policy: key descriptive values
    rows = []
    selected = [
        ("balanced", "Baseline", "baseline", None, None, None),
        ("balanced", "Transparent ratio 0.6 halt 5", "circuit_breaker", True, 0.6, 5),
        ("high_congestion", "Baseline", "baseline", None, None, None),
        ("high_congestion", "Transparent ratio 0.6 halt 5", "circuit_breaker", True, 0.6, 5),
    ]

    for m, setting, ctype, transparent, ratio, halt in selected:
        for policy in ["linear_share", "zero_on_collision"]:
            r = row(collision, m, ctype, policy=policy, transparent=transparent, ratio=ratio, halt=halt)

            rows.append({
                "Market type": markets[m],
                "Setting": setting,
                "Collision policy": "Linear sharing" if policy == "linear_share" else "Zero-on-collision",
                "Relative efficiency": f(r["relative_efficiency"]),
                "Average reward per agent": f(r["avg_reward_per_agent_per_step_mean"]),
                "Gini": f(r["gini_total_rewards_mean"]),
            })

    save(pd.DataFrame(rows), args.out, "table_4_5_collision_descriptives")

    # 4.5 collision policy statistical tests
    rows = []
    for metric, nice in {
        "avg_reward_per_agent_per_step": "Average reward per agent",
        "avg_collisions_per_step": "Average collisions per timestep",
        "gini_total_rewards": "Gini inequality",
    }.items():
        t = test(stats, "H4", metric, "Linear sharing vs zero-on-collision")
        rows.append({
            "Metric": nice,
            "Statistic": f(t["statistic"], 2),
            "Holm-adjusted p": p(t["p_value_holm"]),
        })

    save(pd.DataFrame(rows), args.out, "table_4_5_collision_tests")

    print(f"Saved simple tables to: {args.out}")


if __name__ == "__main__":
    main()