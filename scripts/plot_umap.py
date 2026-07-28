#!/usr/bin/env python3
"""UMAP 2D scatter of post embeddings: marker=topic, color=language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_DIR = ROOT / "data" / "embeddings" / "qwen3-embedding-0.6b"
DEFAULT_OUT_DIR = ROOT / "results"

TOPIC_MARKERS = {
    "ai_coding_innovations": "o",
    "ai_copyright_theft": "s",
    "ai_mass_surveillance": "^",
}

TOPIC_LABELS = {
    "ai_coding_innovations": "AI coding innovations",
    "ai_copyright_theft": "AI copyright theft",
    "ai_mass_surveillance": "AI mass surveillance",
}

LANGUAGE_COLORS = {
    "en": "#1f77b4",
    "es": "#ff7f0e",
    "ar": "#2ca02c",
    "zh": "#d62728",
}

LANGUAGE_LABELS = {
    "en": "English",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese",
}


def load_ids(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    emb_path = args.emb_dir / "embeddings.npy"
    ids_path = args.emb_dir / "ids.json"
    embeddings = np.load(emb_path)
    ids = load_ids(ids_path)
    if len(ids) != embeddings.shape[0]:
        raise SystemExit(
            f"Row mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} ids"
        )

    topics = [row["topic"] for row in ids]
    languages = [row["language"] for row in ids]

    print(
        f"Running UMAP on {embeddings.shape} "
        f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, seed={args.seed})"
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    coords = reducer.fit_transform(embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coords_path = args.out_dir / "umap_coords.npy"
    np.save(coords_path, coords.astype(np.float32))

    meta_path = args.out_dir / "umap_coords.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ids):
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "post_id": row["post_id"],
                        "topic": row["topic"],
                        "language": row["language"],
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for topic, marker in TOPIC_MARKERS.items():
        for lang, color in LANGUAGE_COLORS.items():
            mask = [
                (t == topic and lang_ == lang)
                for t, lang_ in zip(topics, languages)
            ]
            pts = coords[np.asarray(mask)]
            if len(pts) == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                marker=marker,
                s=36,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )

    topic_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            markerfacecolor="gray",
            linestyle="None",
            markersize=9,
            label=TOPIC_LABELS.get(topic, topic),
        )
        for topic, marker in TOPIC_MARKERS.items()
    ]
    lang_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            linestyle="None",
            markersize=9,
            label=LANGUAGE_LABELS.get(lang, lang),
        )
        for lang, color in LANGUAGE_COLORS.items()
    ]

    legend_topics = ax.legend(
        handles=topic_handles,
        title="Topic (shape)",
        loc="upper left",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(legend_topics)
    ax.legend(
        handles=lang_handles,
        title="Language (color)",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )

    ax.set_title("Qwen3-Embedding-0.6B — UMAP of toy posts")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    png_path = args.out_dir / "umap_scatter.png"
    pdf_path = args.out_dir / "umap_scatter.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    info = {
        "method": "UMAP",
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": "cosine",
        "random_state": args.seed,
        "n_points": int(embeddings.shape[0]),
        "embedding_source": str(args.emb_dir.relative_to(ROOT)),
        "encoding": {
            "topic_markers": TOPIC_MARKERS,
            "language_colors": LANGUAGE_COLORS,
        },
        "files": {
            "umap_scatter.png": "2D scatter plot (PNG)",
            "umap_scatter.pdf": "2D scatter plot (PDF)",
            "umap_coords.npy": "(n, 2) float32 UMAP coordinates",
            "umap_coords.jsonl": "Coordinates with topic/language metadata",
        },
    }
    (args.out_dir / "umap_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Saved plot -> {png_path}")
    print(f"Saved plot -> {pdf_path}")
    print(f"Saved coords -> {coords_path}")


if __name__ == "__main__":
    main()
