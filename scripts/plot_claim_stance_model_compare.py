#!/usr/bin/env python3
"""Side-by-side UMAP comparison across embedding model sizes.

Builds a 2×2 figure:

    rows = model size (0.6B, 4B)
    cols = prompt mode (vanilla, stance_instruct)

Reuses already-computed UMAP coords from ``results/claim_stance_umap*`` when
present; otherwise fails with a clear message to run plot_claim_stance_umap.py
first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

STANCE_COLORS = {
    "supportive": "#2ca02c",
    "critical": "#d62728",
    "neutral": "#7f7f7f",
}
STANCE_LABELS = {
    "supportive": "Supportive",
    "critical": "Critical",
    "neutral": "Neutral",
}
ASPECT_MARKERS = {
    "economic": "o",
    "environmental": "s",
    "infrastructure": "^",
    "geopolitical": "D",
    "local_community": "v",
    "technological": "P",
}
ASPECT_LABELS = {
    "economic": "Economic",
    "environmental": "Environmental",
    "infrastructure": "Infrastructure",
    "geopolitical": "Geopolitical",
    "local_community": "Local community",
    "technological": "Technological",
}

DEFAULT_PANELS = [
    {
        "label": "0.6B · vanilla",
        "umap_dir": "results/claim_stance_umap/vanilla",
        "info": "results/claim_stance_umap/umap_info.json",
        "mode": "vanilla",
    },
    {
        "label": "0.6B · stance-instruct",
        "umap_dir": "results/claim_stance_umap/stance_instruct",
        "info": "results/claim_stance_umap/umap_info.json",
        "mode": "stance_instruct",
    },
    {
        "label": "4B · vanilla",
        "umap_dir": "results/claim_stance_umap_qwen3-embedding-4b/vanilla",
        "info": "results/claim_stance_umap_qwen3-embedding-4b/umap_info.json",
        "mode": "vanilla",
    },
    {
        "label": "4B · stance-instruct",
        "umap_dir": "results/claim_stance_umap_qwen3-embedding-4b/stance_instruct",
        "info": "results/claim_stance_umap_qwen3-embedding-4b/umap_info.json",
        "mode": "stance_instruct",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scatter(ax: plt.Axes, rows: list[dict], title: str, metrics: dict | None) -> None:
    coords = np.asarray([[r["x"], r["y"]] for r in rows], dtype=np.float64)
    stances = [r["stance"] for r in rows]
    aspects = [r["aspect"] for r in rows]
    for aspect, marker in ASPECT_MARKERS.items():
        for stance, color in STANCE_COLORS.items():
            mask = np.asarray(
                [(s == stance and a == aspect) for s, a in zip(stances, aspects)]
            )
            pts = coords[mask]
            if len(pts) == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                marker=marker,
                s=18,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.2,
                zorder=2,
            )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if metrics:
        sil = metrics.get("silhouette_cosine")
        nca = metrics.get("nearest_centroid_accuracy")
        if sil is not None and nca is not None:
            ax.text(
                0.98,
                0.02,
                f"sil={sil:.3f}\nnca={nca:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#ddd",
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "claim_stance_umap_model_compare.png",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    metric_table = []
    for ax, panel in zip(axes.ravel(), DEFAULT_PANELS):
        umap_dir = ROOT / panel["umap_dir"]
        coords_path = umap_dir / "umap_coords.jsonl"
        info_path = ROOT / panel["info"]
        if not coords_path.exists():
            raise SystemExit(
                f"Missing {coords_path}. Run plot_claim_stance_umap.py for that model first."
            )
        rows = load_jsonl(coords_path)
        metrics = None
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            metrics = info.get("stance_separation", {}).get(panel["mode"])
        scatter(ax, rows, panel["label"], metrics)
        if metrics:
            metric_table.append(
                {
                    "panel": panel["label"],
                    "silhouette_cosine": metrics.get("silhouette_cosine"),
                    "nearest_centroid_accuracy": metrics.get(
                        "nearest_centroid_accuracy"
                    ),
                    "centroid_cosine": metrics.get("centroid_cosine"),
                }
            )

    # Shared legends on the figure
    stance_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=c,
            markerfacecolor=c,
            linestyle="None",
            markersize=8,
            label=STANCE_LABELS[s],
        )
        for s, c in STANCE_COLORS.items()
    ]
    aspect_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="gray",
            markerfacecolor="gray",
            linestyle="None",
            markersize=8,
            label=ASPECT_LABELS[a],
        )
        for a, m in ASPECT_MARKERS.items()
    ]
    fig.legend(
        handles=stance_handles,
        title="Stance (color)",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.35, 0.02),
        frameon=True,
        fontsize=9,
    )
    fig.legend(
        handles=aspect_handles,
        title="Aspect (shape)",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.72, 0.02),
        frameon=True,
        fontsize=8,
    )
    fig.suptitle(
        "Claim–stance UMAP by model size × prompt mode\n"
        "(color=stance, shape=aspect)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary_path = args.out.with_name(args.out.stem + "_metrics.json")
    summary_path.write_text(
        json.dumps({"panels": metric_table, "figure": str(args.out.relative_to(ROOT))}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.out}")
    print(f"Saved {pdf}")
    print(f"Saved {summary_path}")
    for row in metric_table:
        print(
            f"  {row['panel']}: sil={row['silhouette_cosine']:.3f} "
            f"nca={row['nearest_centroid_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
