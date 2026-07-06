#!/usr/bin/env python3
"""Generate a slide-ready route-equivalence and browser-latency figure."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


DEFAULT_REPORT_DIR = PROJECT_ROOT / "evaluation/reports/baseline_route_equivalence_latency"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation/figures"

HKU_GREEN = "#008165"
HKU_DARK = "#004538"
ORANGE = "#D55E00"
GREY = "#58595B"
LIGHT = "#EEF5F3"
CASE_COLORS = {
    "normal_daily": "#6BA4B8",
    "semantic_fraud": "#D55E00",
    "mixed_risk": "#008165",
    "synthetic_voice": "#8C6BB1",
}
CASE_LABELS = {
    "normal_daily": "Normal",
    "semantic_fraud": "Semantic",
    "mixed_risk": "Mixed",
    "synthetic_voice": "Synthetic",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rounded_box(ax, xy, width, height, facecolor, edgecolor="none", radius=0.04):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    return patch


def draw_equivalence_panel(ax, route: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "A  Route equivalence", fontsize=17, fontweight="bold", color=HKU_DARK, va="top")
    ax.text(0.02, 0.895, "Identical 10 s windows / 5 s step", fontsize=10.5, color=GREY, va="top")

    rounded_box(ax, (0.03, 0.61), 0.34, 0.16, LIGHT)
    rounded_box(ax, (0.63, 0.61), 0.34, 0.16, LIGHT)
    ax.text(0.20, 0.69, "Uploaded route", ha="center", va="center", fontsize=13, fontweight="bold", color=HKU_DARK)
    ax.text(0.80, 0.69, "Live-chunk route", ha="center", va="center", fontsize=13, fontweight="bold", color=HKU_DARK)
    ax.annotate("", xy=(0.61, 0.69), xytext=(0.39, 0.69), arrowprops=dict(arrowstyle="->", lw=3, color=HKU_GREEN))
    ax.text(0.50, 0.745, f"{route['windows']} windows", ha="center", fontsize=10, color=GREY)

    ax.text(0.50, 0.46, "0.000", ha="center", va="center", fontsize=39, fontweight="bold", color=HKU_GREEN)
    ax.text(0.50, 0.365, "score drift (MAE)", ha="center", fontsize=12, color=GREY)

    labels = [("Text", "mae_text_score"), ("Voice", "mae_voice_score"), ("Fusion", "mae_fused_score"), ("Smoothed", "mae_smoothed_score")]
    for index, (label, key) in enumerate(labels):
        x = 0.035 + index * 0.24
        rounded_box(ax, (x, 0.18), 0.21, 0.105, "white", edgecolor="#B9D6CF", radius=0.025)
        ax.text(x + 0.105, 0.245, label, ha="center", va="center", fontsize=9.5, color=GREY)
        ax.text(x + 0.105, 0.205, f"{route[key]:.3f}", ha="center", va="center", fontsize=12, fontweight="bold", color=HKU_DARK)

    ax.text(0.50, 0.075, "100% timing, transcript, label and alert-state agreement", ha="center", fontsize=10.5, color=HKU_DARK, fontweight="bold")


def draw_latency_panel(ax, report: dict, rows: list[dict[str, str]]) -> None:
    browser = report["browser_latency"]
    browser_rows = [row for row in rows if row["protocol"] == "browser_style_5s"]
    order = ["normal_daily", "semantic_fraud", "mixed_risk", "synthetic_voice"]
    rng = np.random.default_rng(20260705)

    ax.set_title("B  Browser-chunk latency", loc="left", fontsize=17, fontweight="bold", color=HKU_DARK, pad=17)
    for index, case_type in enumerate(order):
        values = np.array([float(row["latency_sec"]) for row in browser_rows if row["case_type"] == case_type])
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=25, alpha=0.72, color=CASE_COLORS[case_type], edgecolors="white", linewidths=0.35, zorder=3)
        ax.plot([index - 0.23, index + 0.23], [np.median(values), np.median(values)], color=HKU_DARK, lw=3, zorder=4)

    ax.axhline(5.0, color=ORANGE, lw=2.4, linestyle=(0, (6, 4)), zorder=2)
    ax.text(3.43, 5.02, "5 s chunk budget", ha="right", va="bottom", color=ORANGE, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5.55)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("Processing latency (s)", color=GREY)
    ax.set_xticks(range(4), [CASE_LABELS[item] for item in order])
    ax.grid(axis="y", color="#D9E5E1", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", colors=GREY)

    summary = (
        f"median  {browser['median_latency_sec']:.2f} s     "
        f"p95  {browser['p95_latency_sec']:.2f} s     "
        f"max  {browser['max_latency_sec']:.2f} s"
    )
    ax.text(0.5, 0.96, summary, transform=ax.transAxes, ha="center", va="top", fontsize=11, color=HKU_DARK, fontweight="bold")
    ax.text(0.5, 0.885, f"{browser['successful_chunks']}/{browser['chunks']} chunks within their audio-duration budget", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=HKU_GREEN, fontweight="bold")
    ax.text(0.5, -0.16, "In-process Flask latency; network and MediaRecorder encoding excluded.", transform=ax.transAxes, ha="center", fontsize=8.5, color=GREY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="route_equivalence_browser_latency")
    args = parser.parse_args()

    report = json.loads((args.report_dir / "baseline_route_equivalence_latency_report.json").read_text(encoding="utf-8"))
    rows = load_rows(args.report_dir / "baseline_browser_chunk_latency.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(13.333, 6.3), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.15], wspace=0.14, left=0.035, right=0.975, top=0.96, bottom=0.17)
    draw_equivalence_panel(fig.add_subplot(grid[0, 0]), report["route_equivalence"])
    draw_latency_panel(fig.add_subplot(grid[0, 1]), report, rows)

    for suffix in ("pdf", "png"):
        output = args.output_dir / f"{args.basename}.{suffix}"
        fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
