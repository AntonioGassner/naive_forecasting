import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_tpgnn_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns
    expected = {"horizon", "MAE", "RMSE", "MAPE"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"TPGNN CSV missing columns: {missing}")
    # Normalize horizon to int
    df["horizon"] = df["horizon"].astype(int)
    return df


def load_naive_results(json_path: str) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)
    # Convert to DataFrame
    rows = []
    for item in data:
        # Defensive: only keep entries with required keys
        if not {"approach", "horizon", "MAE", "RMSE", "MAPE"}.issubset(item.keys()):
            continue
        rows.append({
            "approach": item["approach"],
            "variant": item.get("variant", ""),
            "horizon": int(item["horizon"]),
            "MAE": float(item["MAE"]),
            "RMSE": float(item["RMSE"]),
            "MAPE": float(item["MAPE"]),
        })
    if not rows:
        raise ValueError("No valid entries found in naive_approaches_results.json")
    df = pd.DataFrame(rows)
    return df


def prep_comparison(
    tpgnn_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    approach_key: str,
    horizons: List[int],
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of metric -> DataFrame with columns: horizon, TPGNN, Naive, diff
    """
    out: Dict[str, pd.DataFrame] = {}
    # Filter approach
    na_sub = naive_df[naive_df["approach"] == approach_key].copy()
    if na_sub.empty:
        raise ValueError(f"No entries for approach '{approach_key}' in naive results")

    # Aggregate in case multiple variants are present for same approach/horizon
    # We simply take the first occurrence. Alternatively, could choose best variant per metric.
    na_sub = na_sub.sort_values(["horizon", "variant"]).drop_duplicates(subset=["horizon"], keep="first")

    # Merge with tpgnn on horizon for each metric
    for metric in ["MAE", "RMSE", "MAPE"]:
        left = tpgnn_df[["horizon", metric]].rename(columns={metric: "TPGNN"})
        right = na_sub[["horizon", metric]].rename(columns={metric: "Naive"})
        merged = pd.merge(left, right, on="horizon", how="inner")
        merged = merged[merged["horizon"].isin(horizons)].sort_values("horizon")
        merged["diff"] = merged["Naive"] - merged["TPGNN"]  # positive => TPGNN better (lower)
        out[metric] = merged.reset_index(drop=True)
    return out


def plot_dumbbell(
    data: pd.DataFrame,
    title: str,
    metric: str,
    out_path: str,
    palette: Tuple[str, str] = ("#1f77b4", "#ff7f0e"),  # Naive, TPGNN
    label_offset: float = 0.38,
):
    """
    Render a horizontal dumbbell chart for a single metric across horizons.
    """
    if data.empty:
        raise ValueError("No data to plot")

    horizons = data["horizon"].tolist()
    y_pos = np.arange(len(horizons))
    y_offset = 0.15  # move all rows slightly down
    y_row = y_pos + y_offset

    x_naive = data["Naive"].values
    x_tpgnn = data["TPGNN"].values

    fig_h = max(2.5, 0.6 * len(horizons) + 1.5)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))

    # Range with some margins
    x_min = float(min(np.min(x_naive), np.min(x_tpgnn)))
    x_max = float(max(np.max(x_naive), np.max(x_tpgnn)))
    padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    ax.set_xlim(x_min - padding, x_max + padding)

    # Reserve extra vertical space below the last horizon (avoids label overlap with axis)
    extra_pad = 0.8  # tweak 0.6–1.0 as desired (bottom padding)
    top_pad = 0.2    # extra space at the top
    ax.set_ylim(-0.5 - top_pad, len(horizons) - 1 + extra_pad)
    ax.invert_yaxis()  # top = smallest horizon

    # Connecting lines
    for i in range(len(horizons)):
        ax.hlines(y=y_row[i], xmin=x_tpgnn[i], xmax=x_naive[i], color="#b0b0b0", lw=3, zorder=1)

    # Points
    ax.scatter(x_naive, y_row, color=palette[0], s=60, label="Naive", zorder=2)
    ax.scatter(x_tpgnn, y_row, color=palette[1], s=60, label="TPGNN", zorder=3)

    # Difference labels near segment mid-point, placed above the points
    mids = (x_naive + x_tpgnn) / 2.0
    for i, mid in enumerate(mids):
        d = float(data.loc[i, "diff"])  # Naive - TPGNN (lower is better)
        denom = max(abs(x_naive[i]), 1e-9)
        pct = (d / denom) * 100.0
        label = f"Δ={d:+.2f} ({pct:+.1f}%)"
        ax.text(
            mid,
            y_row[i] - (label_offset * 0.5),
            label,
            fontsize=9,
            ha="center",
            va="bottom",
            color="#333333",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    # Aesthetics
    ax.set_yticks(y_row)
    ax.set_yticklabels([f"H{h}" for h in horizons])
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_for_approach(
    approach_key: str,
    comps: Dict[str, pd.DataFrame],
    out_dir: str,
):
    # one image per metric
    for metric, df in comps.items():
        title = f"{approach_key.replace('_', ' ')} vs TPGNN — {metric} (lower is better)"
        out_path = os.path.join(out_dir, f"dumbbell_{approach_key.lower()}_{metric.lower()}.png")
        plot_dumbbell(df, title=title, metric=metric, out_path=out_path)


def plot_panel_for_approach(
    approach_key: str,
    comps: Dict[str, pd.DataFrame],
    out_path: str,
):
    metrics = ["MAE", "RMSE", "MAPE"]
    # Use horizons from any metric df
    sample_df = comps[metrics[0]]
    horizons = sample_df["horizon"].tolist()
    y_pos = np.arange(len(horizons))

    fig_h = max(2.8, 0.6 * len(horizons) + 1.6)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics) / 1.5, fig_h), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    palette = ("#1f77b4", "#ff7f0e")  # Naive, TPGNN

    for ax, metric in zip(axes, metrics):
        df = comps[metric]
        x_naive = df["Naive"].values
        x_tpgnn = df["TPGNN"].values

        # Small consistent downward shift for aesthetics
        y_offset = 0.15
        y_row = y_pos + y_offset

        x_min = float(min(np.min(x_naive), np.min(x_tpgnn)))
        x_max = float(max(np.max(x_naive), np.max(x_tpgnn)))
        padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
        ax.set_xlim(x_min - padding, x_max + padding)

        for i in range(len(horizons)):
            ax.hlines(y=y_row[i], xmin=x_tpgnn[i], xmax=x_naive[i], color="#b0b0b0", lw=3, zorder=1)

        ax.scatter(x_naive, y_row, color=palette[0], s=60, label="Naive", zorder=2)
        ax.scatter(x_tpgnn, y_row, color=palette[1], s=60, label="TPGNN", zorder=3)

        # Compute symmetric y-limits based on shifted rows so points are centered nicely
        extra_bottom = 0.8  # space below last row
        extra_top = 0.2     # space above first row
        y_min = float(np.min(y_row)) - 0.5 - extra_top
        y_max = float(np.max(y_row)) + extra_bottom
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()

        # Difference labels near segment mid-point, placed above the points
        mids = (x_naive + x_tpgnn) / 2.0
        for i, mid in enumerate(mids):
            d = float(df.loc[i, "diff"])  # Naive - TPGNN
            denom = max(abs(x_naive[i]), 1e-9)
            pct = (d / denom) * 100.0
            label = f"Δ={d:+.2f} ({pct:+.1f}%)"
            ax.text(
                mid,
                y_row[i] - (0.38 * 0.5),
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                color="#333333",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        ax.set_yticks(y_row)
        ax.set_yticklabels([f"H{h}" for h in horizons])
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Horizon")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(f"{approach_key.replace('_', ' ')} vs TPGNN (lower is better)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Dumbbell charts: TPGNN vs Naive baselines")
    default_root = os.path.dirname(os.path.dirname(__file__))
    parser.add_argument("--tpgnn_csv", type=str,
                        default=os.path.join(default_root, "results", "tpgnn_baseline.csv"))
    parser.add_argument("--naive_json", type=str,
                        default=os.path.join(default_root, "results", "naive_approaches_results.json"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(default_root, "charts"))
    parser.add_argument("--horizons", type=int, nargs="*", default=[3, 6, 9, 12])
    parser.add_argument("--per_metric_pngs", action="store_true",
                        help="Also write one PNG per metric instead of only panel per approach")
    args = parser.parse_args()

    tpgnn_df = load_tpgnn_results(args.tpgnn_csv)
    naive_df = load_naive_results(args.naive_json)

    approaches = [
        "Holt_Winters",
        "Seasonal_Naive",
        "Hour_of_Week",
    ]

    # Generate figures
    for ak in approaches:
        comps = prep_comparison(tpgnn_df, naive_df, ak, args.horizons)
        # Multi-panel (one per metric) in a single image
        out_panel = os.path.join(args.out_dir, f"panel_{ak.lower()}.png")
        plot_panel_for_approach(ak, comps, out_panel)
        # Optional per-metric images
        if args.per_metric_pngs:
            plot_all_for_approach(ak, comps, args.out_dir)

    print(f"Wrote dumbbell charts to: {args.out_dir}")


if __name__ == "__main__":
    main()


