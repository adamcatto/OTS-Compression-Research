"""Generate the standalone G1 result figures from results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAUSAL_COLOR = "#0072B2"
FRONTIER_COLOR = "#D55E00"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#20242B"


def _load_results(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("results CSV is empty")
    return {
        "step": np.array([int(row["step"]) for row in rows]),
        "causal_r2": np.array([float(row["causal_gradient_energy_r2"]) for row in rows]),
        "frontier_r2": np.array([float(row["frontier_gradient_energy_r2"]) for row in rows]),
        "causal_bpv": np.array([float(row["causal_bits_per_value"]) for row in rows]),
        "frontier_bpv": np.array([float(row["frontier_bits_per_value"]) for row in rows]),
    }


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8A919C")
    axis.tick_params(colors=TEXT_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def plot_fidelity(data: dict[str, np.ndarray], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    figure.subplots_adjust(left=0.10, right=0.73, bottom=0.20, top=0.88)
    _style_axes(axis)
    steps = data["step"]
    axis.plot(
        steps,
        data["causal_r2"],
        color=CAUSAL_COLOR,
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="Exact-prior causal only",
    )
    axis.plot(
        steps,
        data["frontier_r2"],
        color=FRONTIER_COLOR,
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="+ paid current-SVD factors",
    )
    axis.axhline(0.99, color="#30343B", linestyle="--", linewidth=1.4, label="R² = 0.99 fidelity gate")
    axis.axhline(0.90, color="#7B8492", linestyle=":", linewidth=1.5, label="R² = 0.90 continuation cutoff")
    for values, color, offset in (
        (data["causal_r2"], CAUSAL_COLOR, -0.004),
        (data["frontier_r2"], FRONTIER_COLOR, 0.004),
    ):
        for step, value in zip(steps, values):
            axis.annotate(
                f"{value:.4f}",
                (step, value),
                xytext=(0, 8 if offset > 0 else -14),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=color,
                fontsize=9,
                fontweight="semibold",
            )
    axis.set_xticks(steps)
    axis.set_ylim(0.895, 1.003)
    axis.set_xlabel("Training step", color=TEXT_COLOR)
    axis.set_ylabel("Global gradient energy R²", color=TEXT_COLOR)
    axis.set_title("G1 fidelity degrades late in training", loc="left", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    figure.text(0.10, 0.035, "Every stream is within the 0.32 bits/value hard budget (actual: 0.3191–0.3193 bpv).", color="#555D68", fontsize=9)
    axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def plot_gate_gap(data: dict[str, np.ndarray], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.88)
    _style_axes(axis)
    steps = data["step"]
    positions = np.arange(len(steps), dtype=float)
    width = 0.34
    causal_gap = 100.0 * (0.99 - data["causal_r2"])
    frontier_gap = 100.0 * (0.99 - data["frontier_r2"])
    causal_bars = axis.bar(positions - width / 2, causal_gap, width, color=CAUSAL_COLOR, label="Exact-prior causal only")
    frontier_bars = axis.bar(positions + width / 2, frontier_gap, width, color=FRONTIER_COLOR, label="+ paid current-SVD factors")
    axis.axhline(0.0, color="#30343B", linewidth=1.3)
    for bars in (causal_bars, frontier_bars):
        for bar in bars:
            height = bar.get_height()
            axis.annotate(
                f"{height:+.2f} pp",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5 if height >= 0 else -6),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color=TEXT_COLOR,
            )
    axis.set_xticks(positions, [str(step) for step in steps])
    axis.set_ylim(-0.75, max(6.1, float(np.max(causal_gap)) + 0.7))
    axis.set_xlabel("Training step", color=TEXT_COLOR)
    axis.set_ylabel("Shortfall from R² = 0.99 (percentage points)", color=TEXT_COLOR)
    axis.set_title("The fidelity deficit grows despite a full rate budget", loc="left", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    figure.text(0.10, 0.035, "Negative values clear the fidelity gate; positive values miss it.", color="#555D68", fontsize=9)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_results(args.results)
    plot_fidelity(data, args.output_dir / "fidelity_by_step.png")
    plot_gate_gap(data, args.output_dir / "fidelity_gate_gap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
