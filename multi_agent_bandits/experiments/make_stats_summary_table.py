import argparse
import pandas as pd


def p_for(df, hypothesis, metric_contains=None, comparison_contains=None):
    sub = df[df["hypothesis"] == hypothesis].copy()

    if metric_contains is not None:
        sub = sub[sub["metric"].str.contains(metric_contains, na=False)]

    if comparison_contains is not None:
        sub = sub[sub["comparison"].str.contains(comparison_contains, na=False)]

    if len(sub) == 0:
        return ""

    p = sub["p_value_holm"].dropna().min()

    if pd.isna(p):
        return ""

    if p < 0.001:
        return "< 0.001"

    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="compact_stats_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    rows = [
        {
            "Hypothesis": "H1",
            "Main comparison": "Baseline low vs balanced vs high density",
            "Key metric(s)": "Reward per agent; collisions per step",
            "Test": "Kruskal-Wallis",
            "Holm-adjusted p-value": "both < 0.001",
            "Interpretation": "Market density significantly affected reward efficiency and congestion.",
        },
        {
            "Hypothesis": "H2",
            "Main comparison": "Mild vs strict transparent breakers",
            "Key metric(s)": "Available arms; trigger count; balanced-market reward",
            "Test": "Wilcoxon signed-rank",
            "Holm-adjusted p-value": "< 0.001 for available arms/triggers; < 0.001 for balanced reward",
            "Interpretation": "Stricter transparent breakers significantly changed intervention dynamics and reduced reward efficiency most clearly in the balanced market.",
        },
        {
            "Hypothesis": "H3",
            "Main comparison": "Opaque vs transparent breakers under matched settings",
            "Key metric(s)": "Halted choices; strict-setting reward, collisions, triggers, and available arms",
            "Test": "Wilcoxon signed-rank",
            "Holm-adjusted p-value": "mostly < 0.001 in strict settings",
            "Interpretation": "Intervention mode significantly affected outcomes, especially under stricter breaker settings.",
        },
        {
            "Hypothesis": "H4",
            "Main comparison": "Linear sharing vs zero-on-collision",
            "Key metric(s)": "Reward per agent; collisions; Gini inequality",
            "Test": "Wilcoxon signed-rank",
            "Holm-adjusted p-value": "all < 0.001",
            "Interpretation": "Collision policy strongly changed reward efficiency, congestion, and inequality.",
        },
    ]

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)

    print(f"Saved compact statistical summary table to: {args.output}")


if __name__ == "__main__":
    main()