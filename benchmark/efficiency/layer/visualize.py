import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS = {
    "dense": "#1f77b4",  # Blue
    "best_perf": "#ff7f0e",  # Orange
    "best_acc": "#2ca02c",  # Green
    "attention": "#d62728",  # Red
    "linear": "#9467bd",  # Purple
    "mlp": "#8c564b",  # Brown
}


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="visualization/plots", help="Output directory for plots")
    parser.add_argument("--is_decode", action="store_true")
    parser.add_argument(
        "--max_seq_k",
        type=int,
        default=192,
        help="Maximum sequence length to plot in K tokens (set <=0 to disable)",
    )
    parser.add_argument(
        "--seq_step_k",
        type=int,
        default=16,
        help="Sequence-length step to include in plot in K tokens (set <=0 to disable)",
    )
    args = parser.parse_args()

    # Style configuration for IEEE double-column compliance
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )

    if not os.path.exists(args.input_file):
        print(f"File {args.input_file} not found.")
        return

    with open(args.input_file, "r") as f:
        data = json.load(f)

    max_seq_len = args.max_seq_k * 1024 if args.max_seq_k > 0 else None
    seq_step = args.seq_step_k * 1024 if args.seq_step_k > 0 else None

    # Flatten data for pandas
    rows = []
    for entry in data:
        row = entry["metrics"].copy()
        row["cache_type"] = entry["cache_type"]
        row["seq_len"] = entry["seq_len"]
        if max_seq_len is not None and entry["seq_len"] > max_seq_len:
            continue
        if seq_step is not None and entry["seq_len"] % seq_step != 0:
            continue
        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(args.output_dir, exist_ok=True)

    if "Attention Core" not in df.columns or "Linear Projections" not in df.columns or "Layer Total" not in df.columns:
        print("Required columns (Attention Core, Linear Projections, Layer Total) not found in data.")
        return

    # Calculate 'Other' time
    df["Other Time"] = df["Layer Total"] - df["Attention Core"] - df["Linear Projections"]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    seq_lens = sorted(df["seq_len"].unique())
    cache_types = list(df["cache_type"].unique())

    x_positions = list(range(len(seq_lens)))
    num_cache = max(len(cache_types), 1)
    group_width = 0.84
    bar_width = group_width / num_cache

    # Draw grouped stacked bars where each cache type keeps its color,
    # and components are distinguished by hatch/opacity.
    for cache_idx, cache_type in enumerate(cache_types):
        subset = df[df["cache_type"] == cache_type].sort_values("seq_len").set_index("seq_len")
        subset = subset.reindex(seq_lens)

        attention_vals = subset["Attention Core"].fillna(0).to_list()
        linear_vals = subset["Linear Projections"].fillna(0).to_list()
        other_vals = subset["Other Time"].fillna(0).to_list()

        offset = (cache_idx - (num_cache - 1) / 2.0) * bar_width
        x_cache = [x + offset for x in x_positions]
        color = COLORS.get(cache_type, "gray")

        ax1.bar(
            x_cache,
            other_vals,
            width=bar_width,
            color=color,
            alpha=0.55,
            hatch="..",
            edgecolor="black",
            linewidth=0.3,
        )
        ax1.bar(
            x_cache,
            linear_vals,
            width=bar_width,
            bottom=other_vals,
            color=color,
            alpha=0.75,
            hatch="//",
            edgecolor="black",
            linewidth=0.3,
        )
        ax1.bar(
            x_cache,
            attention_vals,
            width=bar_width,
            bottom=[o + l for o, l in zip(other_vals, linear_vals)],
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.3,
        )

    ax1.set_xlabel("Sequence Length", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Time (ms)", fontsize=14, fontweight="bold")

    # Filter ticks to only print at configured intervals.
    if seq_step is not None:
        filtered_ticks = [t for t in seq_lens if t % seq_step == 0]
    else:
        filtered_ticks = seq_lens
    if filtered_ticks:
        filtered_positions = [seq_lens.index(t) for t in filtered_ticks]
        ax1.set_xticks(filtered_positions)
        ax1.set_xticklabels([f"{int(t/1024)}K" for t in filtered_ticks])
    else:
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f"{int(t/1024)}K" if t % 1024 == 0 else str(t) for t in seq_lens])
    plt.setp(ax1.get_xticklabels(), ha="center", fontsize=14)

    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Legend: colors for cache types + hatch styles for components
    legend_elements = []
    for c_type in cache_types:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markerfacecolor=COLORS.get(c_type, "gray"),
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=8,
                label=c_type,
            )
        )
    legend_elements.extend(
        [
            Patch(facecolor="lightgray", edgecolor="black", linewidth=0.6, hatch="..", alpha=0.55, label="Other"),
            Patch(facecolor="lightgray", edgecolor="black", linewidth=0.6, hatch="//", alpha=0.75, label="Linear"),
            Patch(facecolor="lightgray", edgecolor="black", linewidth=0.6, alpha=0.95, label="Attention"),
        ]
    )
    if not args.is_decode:
        ax1.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            fontsize=14,
        )

    plt.tight_layout()
    save_fig(
        fig, os.path.join(args.output_dir, "prefill_breakdown.png" if not args.is_decode else "decode_breakdown.png")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
