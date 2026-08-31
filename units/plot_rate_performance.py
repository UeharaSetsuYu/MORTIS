from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RATES = np.arange(0.1, 1.01, 0.1)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Data_Mining\Code\99zzzz_Paper2_Mortis\Figure\Rate_Performance"
)

# All values are percentages and are rounded to two decimal places.
# Missing experiments are represented by np.nan and are not interpolated.
DATA = {
    "CPSPAN": {
        "ACC": ([66.21, 65.22, 64.31, 65.34, 64.39, 64.35, 62.94, 62.28, 63.19, np.nan],
                [1.32, 2.24, 2.91, 5.84, 2.67, 3.14, 6.82, 4.45, 3.19, np.nan]),
        "ARI": ([55.32, 54.06, 53.08, 53.77, 52.34, 52.44, 50.59, 51.14, 50.05, np.nan],
                [0.83, 3.09, 2.50, 5.29, 1.13, 1.80, 6.49, 3.87, 3.41, np.nan]),
        "NMI": ([78.78, 78.59, 77.77, 77.96, 76.51, 77.51, 76.10, 76.64, 75.09, np.nan],
                [0.32, 1.66, 0.63, 2.60, 0.35, 1.54, 2.89, 1.49, 2.11, np.nan]),
    },
    "DCP": {
        "ACC": ([40.83, 43.68, 40.42, 40.66, 34.95, 29.56, 27.91, 28.20, 27.33, np.nan],
                [7.10, 6.98, 3.20, 3.37, 1.47, 1.09, 1.25, 0.76, 1.37, np.nan]),
        "ARI": ([33.67, 34.82, 28.78, 28.36, 12.91, 7.43, 8.10, 9.01, 8.46, np.nan],
                [4.44, 5.08, 2.04, 1.78, 2.23, 2.05, 0.96, 1.43, 1.21, np.nan]),
        "NMI": ([66.04, 66.42, 61.34, 61.45, 47.40, 41.94, 39.77, 40.68, 39.13, np.nan],
                [3.04, 3.08, 2.61, 1.72, 1.55, 1.24, 0.52, 1.60, 1.32, np.nan]),
    },
    "SMILE": {
        "ACC": ([82.74, 79.80, 81.18, 77.64, 77.18, 76.50, 75.30, 68.98, 58.88, 36.73],
                [0.78, 1.49, 0.62, 2.30, 2.41, 2.33, 2.80, 1.85, 3.42, 1.23]),
        "ARI": ([73.34, 69.96, 70.88, 67.06, 66.57, 65.27, 63.10, 55.29, 44.33, 21.00],
                [0.87, 1.16, 0.69, 2.10, 2.08, 2.09, 2.20, 2.31, 3.12, 1.28]),
        "NMI": ([84.67, 83.28, 83.60, 81.70, 81.49, 80.74, 79.26, 74.97, 67.77, 52.67],
                [0.68, 0.82, 0.18, 0.87, 1.06, 1.03, 1.48, 1.80, 2.15, 1.79]),
    },
    "CAMERA": {
        "ACC": ([79.96, 78.26, 75.11, 75.41, 71.55, 70.23, 65.84, 60.87, 53.62, 38.84],
                [2.29, 2.25, 0.85, 1.15, 2.73, 4.10, 4.92, 2.00, 4.14, 1.75]),
        "ARI": ([70.26, 68.60, 64.45, 63.23, 59.33, 58.38, 53.62, 46.53, 36.93, 20.76],
                [2.21, 2.41, 1.32, 1.17, 2.02, 5.04, 4.69, 2.32, 2.94, 2.12]),
        "NMI": ([85.53, 84.81, 82.26, 81.30, 79.30, 77.76, 76.30, 70.92, 65.33, 52.35],
                [0.99, 1.28, 1.37, 0.79, 1.13, 2.40, 2.53, 1.46, 1.60, 2.08]),
    },
    "CPMN": {
        "ACC": ([63.77, 59.67, 58.55, 55.32, 51.93, 48.49, 46.00, 43.44, 40.21, 37.85],
                [2.49, 0.79, 2.29, 1.44, 2.52, 1.93, 3.13, 1.42, 1.77, 2.09]),
        "ARI": ([50.33, 43.61, 41.47, 37.10, 30.75, 28.17, 25.21, 22.88, 20.25, 17.40],
                [2.04, 1.11, 1.27, 1.06, 1.44, 1.51, 2.41, 1.00, 0.83, 1.30]),
        "NMI": ([73.94, 69.18, 67.23, 63.89, 59.87, 57.47, 55.34, 53.62, 51.31, 49.58],
                [1.24, 0.74, 0.39, 0.66, 1.27, 0.81, 1.67, 0.79, 1.11, 1.61]),
    },
    "PMIMC": {
        "ACC": ([46.54, 44.93, 43.64, 44.97, 43.81, 46.09, 47.21, 49.11, 42.28, np.nan],
                [6.57, 4.51, 4.66, 4.06, 5.42, 3.17, 4.53, 5.29, 0.40, np.nan]),
        "ARI": ([63.22, 62.69, 60.81, 62.74, 60.43, 62.69, 63.86, 63.54, 58.79, np.nan],
                [5.08, 3.75, 3.99, 2.41, 4.89, 3.20, 3.96, 3.90, 1.03, np.nan]),
        "NMI": ([31.92, 30.43, 28.55, 30.58, 28.08, 31.99, 32.38, 32.80, 26.58, np.nan],
                [5.95, 3.89, 4.30, 3.10, 5.62, 4.47, 4.39, 4.76, 0.93, np.nan]),
    },
    "GLGC": {
        "ACC": ([72.17, 70.89, 70.35, 64.64, 64.64, 55.57, 47.70, 41.99, 24.43, 36.69],
                [1.19, 2.22, 1.81, 3.42, 3.42, 4.65, 1.87, 1.44, 1.69, 1.53]),
        "ARI": ([58.12, 57.46, 54.57, 47.82, 47.82, 35.52, 19.77, 7.93, 3.50, 20.96],
                [0.97, 1.43, 2.53, 4.20, 4.20, 6.45, 3.28, 1.57, 0.84, 1.94]),
        "NMI": ([77.16, 76.59, 74.48, 70.67, 70.67, 63.96, 56.14, 49.84, 35.59, 53.08],
                [0.41, 0.69, 1.66, 2.80, 2.80, 3.65, 2.00, 1.06, 2.30, 1.96]),
    },
    "IDMVC-DIA": {
        "ACC": ([68.74, 60.95, 60.29, 53.66, 50.35, 45.92, 42.32, 36.36, 36.89, np.nan],
                [2.62, 4.02, 4.40, 4.29, 3.81, 5.19, 3.44, 2.54, 1.49, np.nan]),
        "ARI": ([55.97, 47.74, 46.23, 39.62, 35.57, 32.61, 28.31, 22.60, 21.10, np.nan],
                [2.70, 3.41, 3.56, 4.42, 3.97, 3.88, 2.00, 2.48, 1.00, np.nan]),
        "NMI": ([75.60, 70.14, 68.54, 64.71, 61.27, 59.40, 55.86, 51.28, 51.29, np.nan],
                [1.23, 2.21, 1.97, 2.84, 3.04, 2.79, 1.92, 2.61, 0.73, np.nan]),
    },
    "ProImp": {
        "ACC": ([71.51, 71.55, 71.84, 67.49, 68.07, 66.21, 64.31, 61.53, 50.06, np.nan],
                [2.02, 2.58, 1.91, 4.04, 2.69, 3.22, 3.36, 2.20, 0.86, np.nan]),
        "ARI": ([59.66, 58.86, 59.38, 55.78, 53.68, 52.59, 49.94, 45.95, 32.83, np.nan],
                [1.28, 2.45, 1.87, 3.64, 2.56, 3.16, 2.50, 1.61, 0.93, np.nan]),
        "NMI": ([78.83, 78.64, 79.12, 77.29, 75.09, 74.99, 73.32, 70.37, 61.27, np.nan],
                [1.03, 1.23, 1.05, 1.55, 1.27, 1.93, 1.72, 1.07, 0.90, np.nan]),
    },
    "Ours": {
        "ACC": ([85.18, 83.77, 82.77, 80.79, 79.01, 77.68, 76.11, 69.98, 60.87, 46.67],
                [0.41, 1.23, 1.58, 0.88, 1.63, 2.05, 1.85, 1.44, 2.08, 0.55]),
        "ARI": ([77.43, 75.26, 73.77, 71.38, 67.95, 66.62, 64.41, 57.48, 47.04, 31.34],
                [0.70, 0.53, 1.16, 1.11, 1.82, 1.88, 1.67, 0.95, 2.80, 1.26]),
        "NMI": ([89.02, 87.53, 86.78, 85.48, 83.95, 83.47, 81.14, 77.78, 73.78, 65.59],
                [0.65, 0.19, 0.74, 0.75, 0.91, 0.90, 0.78, 0.83, 1.77, 0.92]),
    },
}


COLORS = {
    "CPSPAN": "#4C78A8",
    "DCP": "#F58518",
    "SMILE": "#54A24B",
    "CAMERA": "#B279A2",
    "CPMN": "#E45756",
    "PMIMC": "#72B7B2",
    "GLGC": "#FF9DA6",
    "IDMVC-DIA": "#9D755D",
    "ProImp": "#7F7F7F",
    "Ours": "#A83C32",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "--", "-"]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def validate_data() -> None:
    if "GLANCE" in DATA:
        raise ValueError("GLANCE must not be included in the plotted data.")
    for method, metrics in DATA.items():
        for metric in ("ACC", "ARI", "NMI"):
            means, stds = metrics[metric]
            if len(means) != len(RATES) or len(stds) != len(RATES):
                raise ValueError(f"{method}/{metric} must contain 10 mean/std values.")


def plot_metric(metric: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.79)

    for index, (method, metrics) in enumerate(DATA.items()):
        mean = np.asarray(metrics[metric][0], dtype=float)
        std = np.asarray(metrics[metric][1], dtype=float)
        is_ours = method == "Ours"

        ax.fill_between(
            RATES,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.13 if is_ours else 0.07,
            linewidth=0,
            zorder=1 if is_ours else 0,
        )
        ax.plot(
            RATES,
            mean,
            label=method,
            color=COLORS[method],
            # marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            linewidth=2.6 if is_ours else 1.55,
            markersize=8.0 if is_ours else 5.0,
            markerfacecolor=COLORS[method] if is_ours else "white",
            markeredgewidth=1.1,
            zorder=4 if is_ours else 2,
        )

    ax.set_xlabel("Missing Rate")
    ax.set_ylabel("Clustering Performance (%)")
    ax.set_xlim(0.075, 1.025)
    ax.set_ylim(0, 100)
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{rate:.1f}" for rate in RATES])
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, linestyle="--", alpha=0.8)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        if text.get_text() == "Ours":
            text.set_fontweight("bold")

    output_stem = output_dir / f"DHA_{metric}_rate_performance"
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DHA clustering performance across missing rates."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    validate_data()
    for metric in ("ACC", "ARI", "NMI"):
        plot_metric(metric, args.output_dir)
    print(f"Saved six figure files to: {args.output_dir}")


if __name__ == "__main__":
    main()
