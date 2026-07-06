#!/usr/bin/env python3
"""Generate the controlled SV_long_05 timeline figure used in the defence deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "evaluation/predictions/final_learned_late_fusion_w10_s5/sample_level_cv_predictions.json"
)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Prediction file does not contain records: {path}")
    return records


def find_sample(records: List[Dict[str, Any]], sample_id: str) -> Dict[str, Any]:
    matches = [record for record in records if str(record.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {sample_id}, found {len(matches)}")
    return matches[0]


def generate(record: Dict[str, Any], output_pdf: Path, output_png: Path) -> None:
    timeline = record.get("timeline") or []
    if not timeline:
        raise ValueError("Selected sample has an empty timeline.")

    end_times = [float(point["end_sec"]) for point in timeline]
    text_scores = [float(point.get("text_score", 0.0)) for point in timeline]
    voice_scores = [float(point.get("voice_score", 0.0)) for point in timeline]
    fixed_scores = [float(point.get("smoothed_score", 0.0)) for point in timeline]
    learned_scores = [float(point.get("learned_late_fusion_score", 0.0)) for point in timeline]

    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(10.8, 4.45), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")
    ax.axhspan(70, 100, color="#008165", alpha=0.055, zorder=0)
    ax.axhline(70, color="#58595B", linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)

    ax.plot(end_times, text_scores, color="#6BA4B8", marker="o", markersize=4.2,
            linewidth=1.8, label="Text risk", zorder=3)
    ax.plot(end_times, voice_scores, color="#D55E00", marker="o", markersize=4.5,
            linewidth=2.2, label="Voice risk", zorder=4)
    ax.plot(end_times, fixed_scores, color="#004538", marker="s", markersize=4.2,
            linewidth=2.2, label="Fixed fusion + smoothing", zorder=5)
    ax.plot(end_times, learned_scores, color="#008165", marker="D", markersize=4.2,
            linewidth=2.4, label="Learned late fusion (OOF)", zorder=6)

    ax.text(43.7, 72.2, "Alert threshold = 70", ha="right", va="bottom",
            color="#58595B", fontsize=8.8)
    fixed_peak_index = max(range(len(fixed_scores)), key=fixed_scores.__getitem__)
    ax.annotate(
        f"Fixed peak {fixed_scores[fixed_peak_index]:.2f}\n(no alert)",
        xy=(end_times[fixed_peak_index], fixed_scores[fixed_peak_index]),
        xytext=(30, -43), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#004538", "lw": 1.0},
        color="#004538", fontsize=9, ha="left",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#004538", "alpha": 0.95},
    )
    ax.set_xlim(8, max(end_times) + 2.5)
    ax.set_ylim(0, 104)
    ax.set_xlabel("Window end time (s)")
    ax.set_ylabel("Risk score")
    ax.set_xticks([10, 15, 20, 25, 30, 35, 40, 46.18])
    ax.set_xticklabels(["10", "15", "20", "25", "30", "35", "40", "46.18"])
    ax.set_yticks([0, 20, 40, 60, 70, 80, 100])
    ax.grid(axis="both", color="#D7D7D7", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A8A8A")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--sample-id", default="SV_long_05")
    parser.add_argument("--output-pdf", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    record = find_sample(load_records(args.predictions.resolve()), args.sample_id)
    generate(record, args.output_pdf.resolve(), args.output_png.resolve())
    print(f"sample_id={args.sample_id}")
    print(f"output_pdf={args.output_pdf.resolve()}")
    print(f"output_png={args.output_png.resolve()}")


if __name__ == "__main__":
    main()
