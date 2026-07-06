#!/usr/bin/env python3
"""Generate the slide-ready ablation and window/step trade-off figure."""

from __future__ import annotations

import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


REPORT_ROOT = PROJECT_ROOT / "evaluation/reports"
OUTPUT_DIR = PROJECT_ROOT / "evaluation/figures"
HKU_GREEN = "#008165"
HKU_DARK = "#004538"
ORANGE = "#D55E00"
BLUE = "#6BA4B8"
GREY = "#58595B"


def read_ablation(run_name: str) -> dict[str, dict[str, float]]:
    path = REPORT_ROOT / run_name / "dynamic_eval_ablation_summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["variant"]: {
            "f1": float(row["f1"]),
            "alert_recall": float(row["fraud_alert_recall"]),
            "normal_alert_fpr": float(row["normal_alert_false_positive_rate"]),
        }
        for row in rows
    }


def main() -> None:
    baseline = read_ablation("final_baseline_w10_s5")
    variants = ["text_only", "voice_only", "fusion_without_smoothing", "fusion_with_smoothing"]
    labels = ["Text only", "Voice only", "Fixed fusion", "+ smoothing"]

    settings = [
        ("5 / 2.5 s", read_ablation("final_baseline_w5_s2p5")["fusion_with_smoothing"]),
        ("10 / 5 s", baseline["fusion_with_smoothing"]),
        ("20 / 10 s", read_ablation("final_baseline_w20_s10")["fusion_with_smoothing"]),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x = np.arange(len(variants))
    width = 0.34
    f1 = [baseline[item]["f1"] * 100 for item in variants]
    recall = [baseline[item]["alert_recall"] * 100 for item in variants]
    bars1 = ax.bar(x - width / 2, f1, width, color=BLUE, label="Final F1")
    bars2 = ax.bar(x + width / 2, recall, width, color=HKU_GREEN, label="Fraud alert recall")
    ax.set_title("A  Decision-layer ablation", loc="left", fontsize=15, fontweight="bold", color=HKU_DARK)
    ax.set_ylabel("Score (%)", color=GREY)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    for bars in (bars1, bars2):
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, color=GREY)

    ax = axes[1]
    sx = np.arange(len(settings))
    alert_recall = [item[1]["alert_recall"] * 100 for item in settings]
    alert_fpr = [item[1]["normal_alert_fpr"] * 100 for item in settings]
    ax.plot(sx, alert_recall, color=HKU_GREEN, marker="o", markersize=9, linewidth=2.6, label="Fraud alert recall")
    ax.plot(sx, alert_fpr, color=ORANGE, marker="o", markersize=9, linewidth=2.6, label="Normal alert FPR")
    ax.axvspan(0.72, 1.28, color=HKU_GREEN, alpha=0.09)
    ax.text(1, 91, "Selected", ha="center", color=HKU_DARK, fontweight="bold", fontsize=10)
    for i, value in enumerate(alert_recall):
        ax.text(i, value + 3.2, f"{value:.0f}%", ha="center", color=HKU_DARK, fontsize=9, fontweight="bold")
    for i, value in enumerate(alert_fpr):
        ax.text(i, value - 6.0, f"{value:.0f}%", ha="center", color=ORANGE, fontsize=9, fontweight="bold")
    ax.set_title("B  Window / step trade-off", loc="left", fontsize=15, fontweight="bold", color=HKU_DARK)
    ax.set_ylabel("Rate (%)", color=GREY)
    ax.set_ylim(0, 100)
    ax.set_xticks(sx, [item[0] for item in settings])
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", color="#D9E5E1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(colors=GREY)

    fig.text(
        0.5,
        0.015,
        "10 s / 5 s balances final stability, alert coverage, and transient false-alarm exposure.",
        ha="center",
        color=HKU_DARK,
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.19)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = OUTPUT_DIR / f"ablation_window_tradeoff.{suffix}"
        fig.savefig(output, dpi=240, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
