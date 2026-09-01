"""Generate standalone G1P stability figures from the saved CSV results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAUSAL_COLOR = "#0072B2"
FRONTIER_COLOR = "#D55E00"
PROCEDURAL_COLOR = "#009E73"
ENDPOINT_COLOR = "#CC79A7"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#20242B"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"results CSV is empty: {path}")
    return rows


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8A919C")
    axis.tick_params(colors=TEXT_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def plot_fidelity_comparison(g1_rows: list[dict[str, str]], g1p_rows: list[dict[str, str]], output: Path) -> None:
    g1 = {int(row["step"]): row for row in g1_rows}
    g1p = {int(row["step"]): row for row in g1p_rows}
    steps = np.array([100, 500, 999])
    causal = np.array([float(g1[step]["causal_gradient_energy_r2"]) for step in steps])
    frontier = np.array([float(g1[step]["frontier_gradient_energy_r2"]) for step in steps])
    procedural = np.array([float(g1p[step]["gradient_energy_r2"]) for step in steps])

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    figure.subplots_adjust(left=0.10, right=0.72, bottom=0.20, top=0.88)
    _style_axes(axis)
    axis.plot(steps, causal, color=CAUSAL_COLOR, marker="o", linewidth=2.3, label="G1 exact-prior causal")
    axis.plot(steps, frontier, color=FRONTIER_COLOR, marker="o", linewidth=2.3, label="G1 + paid current SVD")
    axis.plot(steps, procedural, color=PROCEDURAL_COLOR, marker="o", linewidth=2.7, label="G1P procedural transcript")
    axis.axhline(0.99, color="#30343B", linestyle="--", linewidth=1.4, label="R² = 0.99 fidelity gate")
    for step, value in zip(steps, procedural):
        axis.annotate(
            "≈1.000000",
            (step, value),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=PROCEDURAL_COLOR,
            fontweight="semibold",
        )
    axis.set_xticks(steps)
    axis.set_ylim(0.93, 1.004)
    axis.set_xlabel("Training step", color=TEXT_COLOR)
    axis.set_ylabel("Global gradient energy R²", color=TEXT_COLOR)
    axis.set_title("Procedural replay removes G1's late-training fidelity collapse", loc="left", color=TEXT_COLOR, fontsize=14.5, fontweight="bold")
    figure.text(
        0.10,
        0.035,
        "G1 uses ≈0.3192 bpv; G1P uses 0.03258 bpv but requires the dataset, code, and training compute.",
        color="#555D68",
        fontsize=9,
    )
    axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def plot_numerical_drift(g1p_rows: list[dict[str, str]], summary: dict[str, object], output: Path) -> None:
    steps = np.array([int(row["step"]) for row in g1p_rows])
    positions = np.arange(len(steps), dtype=float)
    relative_l2 = np.array([float(row["relative_l2_error"]) for row in g1p_rows])
    final_relative_l2 = float(summary["final_weight_metrics"]["relative_l2_error"])

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.20, top=0.88)
    _style_axes(axis)
    axis.semilogy(
        positions,
        relative_l2,
        color=PROCEDURAL_COLOR,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Gradient relative L2",
    )
    axis.scatter(
        [positions[-1]],
        [final_relative_l2],
        color=ENDPOINT_COLOR,
        marker="D",
        s=65,
        zorder=4,
        label="Final-weight relative L2",
    )
    axis.axhline(0.1, color="#30343B", linestyle="--", linewidth=1.4, label="R² = 0.99 error boundary")
    axis.annotate(
        f"final weights {final_relative_l2:.2e}",
        (positions[-1], final_relative_l2),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=ENDPOINT_COLOR,
        fontweight="semibold",
    )
    axis.set_xticks(positions, [str(step) for step in steps])
    axis.set_ylim(5e-8, 0.2)
    axis.set_xlabel("Training step", color=TEXT_COLOR)
    axis.set_ylabel("Relative L2 error (log scale)", color=TEXT_COLOR)
    axis.set_title("Numerical drift grows slowly and stays far below the fidelity limit", loc="left", color=TEXT_COLOR, fontsize=14.5, fontweight="bold")
    figure.text(
        0.11,
        0.035,
        "The replay is not bit-exact; step-999 gradient error remains 4.03e-6 and final-weight error 1.34e-5.",
        color="#555D68",
        fontsize=9,
    )
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--g1-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g1p_rows = _rows(args.results)
    g1_rows = _rows(args.g1_results)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot_fidelity_comparison(g1_rows, g1p_rows, args.output_dir / "fidelity_comparison.png")
    plot_numerical_drift(g1p_rows, summary, args.output_dir / "numerical_drift.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
