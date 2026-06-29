import argparse
import os
import pandas as pd
import numpy as np


def clean_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return x


def load_results(path):
    df = pd.read_csv(path)

    if "transparent" in df.columns:
        df["transparent"] = df["transparent"].apply(clean_bool)

    numeric_cols = [
        "n_agents", "n_arms", "threshold_ratio", "halt_duration",
        "avg_reward_per_agent_per_step_mean",
        "avg_collisions_per_step_mean",
        "relative_efficiency",
        "total_halted_choices_mean",
        "total_triggers_mean",
        "avg_available_arms_mean",
        "gini_total_rewards_mean",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_table(df, outdir, filename):
    path = os.path.join(outdir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def get_row(df, market, condition_type="baseline", policy="linear_share",
            transparent=None, ratio=None, halt=None):
    sub = df[
        (df["market_name"] == market) &
        (df["condition_type"] == condition_type)
    ].copy()

    if "collision_policy" in sub.columns:
        sub = sub[sub["collision_policy"] == policy]

    if condition_type == "circuit_breaker":
        sub = sub[
            (sub["transparent"] == transparent) &
            (sub["threshold_ratio"] == ratio) &
            (sub["halt_duration"] == halt)
        ]

    if len(sub) == 0:
        return None

    return sub.iloc[0]


def make_table_1_experimental_design(linear_df, outdir):
    rows = []

    market_info = [
        ("low_congestion", "Low congestion", "More arms than agents", "Tests behaviour when agents can spread across opportunities."),
        ("balanced", "Balanced", "Equal number of agents and arms", "Main reference case where crowding can emerge but is not structurally unavoidable."),
        ("high_congestion", "High congestion", "More agents than arms", "Tests behaviour when competition for arms is structurally stronger."),
    ]

    for market, label, density_description, purpose in market_info:
        row = get_row(linear_df, market, condition_type="baseline")

        if row is None:
            continue

        n_agents = int(row["n_agents"])
        n_arms = int(row["n_arms"])
        ratio = n_agents / n_arms

        rows.append({
            "Market setting": label,
            "Agents": n_agents,
            "Arms": n_arms,
            "Agent-to-arm ratio": f"{ratio:.2f}",
            "Density interpretation": density_description,
            "Purpose in the experiment": purpose,
        })

    table = pd.DataFrame(rows)
    save_table(table, outdir, "table_1_experimental_design_overview.csv")


def make_table_2_breaker_design(linear_df, outdir):
    rows = []

    breaker_settings = [
        (0.8, 3, "Mild", "High threshold and short halt. Intervention is rare and brief."),
        (0.8, 5, "Mild to moderate", "High threshold but longer halt. Intervention remains selective."),
        (0.8, 10, "Moderate", "High threshold with long halt. Can matter when triggers repeat."),
        (0.6, 3, "Moderate", "Lower threshold but short halt. Intervention is easier to trigger but recovery is quick."),
        (0.6, 5, "Strict", "Lower threshold and medium halt. Can strongly reduce available arms."),
        (0.6, 10, "Very strict", "Lower threshold and long halt. Highest risk of repeated restriction or collapse."),
    ]

    for ratio, halt, strictness, interpretation in breaker_settings:
        balanced_transparent = get_row(
            linear_df,
            market="balanced",
            condition_type="circuit_breaker",
            transparent=True,
            ratio=ratio,
            halt=halt,
        )

        balanced_opaque = get_row(
            linear_df,
            market="balanced",
            condition_type="circuit_breaker",
            transparent=False,
            ratio=ratio,
            halt=halt,
        )

        rel_transparent = ""
        rel_opaque = ""

        if balanced_transparent is not None:
            rel_transparent = f"{balanced_transparent['relative_efficiency']:.3f}"

        if balanced_opaque is not None:
            rel_opaque = f"{balanced_opaque['relative_efficiency']:.3f}"

        rows.append({
            "Threshold ratio": ratio,
            "Halt duration": halt,
            "Strictness label": strictness,
            "Design interpretation": interpretation,
            "Balanced opaque relative efficiency": rel_opaque,
            "Balanced transparent relative efficiency": rel_transparent,
        })

    table = pd.DataFrame(rows)
    save_table(table, outdir, "table_2_circuit_breaker_design_interpretation.csv")


def make_table_3_key_findings(linear_df, collision_df, outdir):
    rows = []

    # Finding 1: density and collisions
    low = get_row(linear_df, "low_congestion")
    high = get_row(linear_df, "high_congestion")

    if low is not None and high is not None:
        collision_multiplier = (
            high["avg_collisions_per_step_mean"] /
            low["avg_collisions_per_step_mean"]
        )

        rows.append({
            "Result theme": "Market density",
            "Main observation": "High-congestion markets produced many more collisions than low-congestion markets.",
            "Key metric": "Average collisions per timestep",
            "Numerical summary": f"{low['avg_collisions_per_step_mean']:.3f} to {high['avg_collisions_per_step_mean']:.3f} "
                                 f"({collision_multiplier:.1f}× increase)",
            "Interpretation": "The density manipulation worked: more agents per arm increased crowding."
        })

    # Finding 2: transparent strict collapse
    balanced_strict = get_row(
        linear_df,
        "balanced",
        condition_type="circuit_breaker",
        transparent=True,
        ratio=0.6,
        halt=10,
    )

    if balanced_strict is not None:
        rows.append({
            "Result theme": "Strict transparent breakers",
            "Main observation": "The strict transparent setting sharply reduced efficiency in the balanced market.",
            "Key metric": "Relative efficiency",
            "Numerical summary": f"{balanced_strict['relative_efficiency']:.3f}",
            "Interpretation": "When halted arms are removed from choice, strict intervention can leave agents with too few available options."
        })

    # Finding 3: opaque mechanism
    opaque_strict = get_row(
        linear_df,
        "balanced",
        condition_type="circuit_breaker",
        transparent=False,
        ratio=0.6,
        halt=10,
    )

    transparent_strict = get_row(
        linear_df,
        "balanced",
        condition_type="circuit_breaker",
        transparent=True,
        ratio=0.6,
        halt=10,
    )

    if opaque_strict is not None and transparent_strict is not None:
        rows.append({
            "Result theme": "Opaque vs transparent mode",
            "Main observation": "Opaque mode generated halted-arm choices, while transparent mode did not.",
            "Key metric": "Total halted-arm choices",
            "Numerical summary": f"Opaque: {opaque_strict['total_halted_choices_mean']:.1f}; "
                                 f"Transparent: {transparent_strict['total_halted_choices_mean']:.1f}",
            "Interpretation": "Opaque agents can choose blocked arms and learn indirectly from zero rewards; transparent agents avoid blocked arms directly."
        })

    # Finding 4: available arms mechanism
    high_strict_transparent = get_row(
        linear_df,
        "high_congestion",
        condition_type="circuit_breaker",
        transparent=True,
        ratio=0.6,
        halt=10,
    )

    if high_strict_transparent is not None:
        rows.append({
            "Result theme": "Available-arm restriction",
            "Main observation": "Strict transparent intervention greatly reduced the average number of available arms.",
            "Key metric": "Average available arms",
            "Numerical summary": f"{high_strict_transparent['avg_available_arms_mean']:.3f} available arms on average",
            "Interpretation": "The efficiency loss is explained by the mechanism of removing too many arms from the active choice set."
        })

    # Finding 5: collision policy and inequality
    linear_high = get_row(collision_df, "high_congestion", policy="linear_share")
    zero_high = get_row(collision_df, "high_congestion", policy="zero_on_collision")

    if linear_high is not None and zero_high is not None:
        rows.append({
            "Result theme": "Collision policy",
            "Main observation": "Zero-on-collision produced much higher reward inequality than linear sharing.",
            "Key metric": "Gini coefficient of total rewards",
            "Numerical summary": f"Linear share: {linear_high['gini_total_rewards_mean']:.3f}; "
                                 f"Zero-on-collision: {zero_high['gini_total_rewards_mean']:.3f}",
            "Interpretation": "When collisions give zero reward, crowding becomes more damaging and rewards become less evenly distributed."
        })

    table = pd.DataFrame(rows)
    save_table(table, outdir, "table_3_interpretive_key_findings.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-agg", required=True)
    parser.add_argument("--collision-agg", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    linear_df = load_results(args.linear_agg)
    collision_df = load_results(args.collision_agg)

    make_table_1_experimental_design(linear_df, args.out)
    make_table_2_breaker_design(linear_df, args.out)
    make_table_3_key_findings(linear_df, collision_df, args.out)

    print("\nDone. Created explanatory thesis tables.")


if __name__ == "__main__":
    main()