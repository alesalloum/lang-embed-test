#!/usr/bin/env python3
"""UMAP scatter of the 600 library posts, colored by KMeans hard/soft labels.

Reuses ``results/umap_coords.npy`` when present (same embedding rows / seed as
``plot_umap.py``). Optionally recomputes UMAP if ``umap-learn`` is installed.

Hard plot
    Color = hard cluster label; marker shape = ground-truth topic.

Soft plot
    Face color = RGB blend of soft weights ``(w0, w1, w2)`` mapped to the three
    cluster colors (mixture reads as soft membership). Marker = GT topic.

Example::

    python scripts/plot_cluster_umap.py
    python scripts/plot_cluster_umap.py --recompute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_DIR = ROOT / "data" / "embeddings" / "qwen3-embedding-0.6b"
DEFAULT_ASSIGN = ROOT / "results" / "topic_attention_k3" / "post_assignments.jsonl"
DEFAULT_UMAP_COORDS = ROOT / "results" / "umap_coords.npy"
DEFAULT_OUT_DIR = ROOT / "results" / "topic_attention_k3"

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

# Distinct cluster palette (avoid purple-default bias; clear on light bg).
CLUSTER_COLORS = np.array(
    [
        [0.12, 0.47, 0.71],  # blue  — cluster 0
        [0.84, 0.37, 0.00],  # orange — cluster 1
        [0.17, 0.63, 0.17],  # green — cluster 2
    ],
    dtype=np.float64,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_library_assignments(path: Path) -> list[dict]:
    rows = [r for r in load_jsonl(path) if r.get("source", "library") == "library"]
    if not rows:
        # Fallback: first block without user_id (library posts).
        rows = [r for r in load_jsonl(path) if r.get("user_id") in (None, "", "null")]
    if not rows:
        raise SystemExit(f"No library post assignments in {path}")
    return rows


def soft_rgb(weights: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map soft weight rows ``(n, K)`` to RGB via palette blend (K colors)."""
    k = min(weights.shape[1], palette.shape[0])
    w = weights[:, :k]
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return (w @ palette[:k]).clip(0.0, 1.0)


def get_coords(
    emb_dir: Path,
    coords_path: Path,
    recompute: bool,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    if not recompute and coords_path.exists():
        coords = np.load(coords_path)
        print(f"Loaded cached UMAP coords {coords.shape} from {coords_path}")
        return coords.astype(np.float64)

    try:
        import umap
    except ImportError as exc:
        if coords_path.exists():
            coords = np.load(coords_path)
            print(
                f"umap-learn missing ({exc}); falling back to cached coords "
                f"{coords_path}"
            )
            return coords.astype(np.float64)
        raise SystemExit(
            "Need umap-learn to compute coords, or provide --umap-coords "
            f"(tried {coords_path})"
        ) from exc

    emb = np.load(emb_dir / "embeddings.npy")
    print(
        f"Recomputing UMAP on {emb.shape} "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed})"
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(emb).astype(np.float64)


def scatter_by_topic(
    ax,
    coords: np.ndarray,
    topics: list[str],
    facecolors: np.ndarray,
    edgecolors: str | np.ndarray = "white",
) -> None:
    for topic, marker in TOPIC_MARKERS.items():
        idx = [i for i, t in enumerate(topics) if t == topic]
        if not idx:
            continue
        pts = coords[idx]
        fc = facecolors[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=fc,
            marker=marker,
            s=42,
            alpha=0.9,
            edgecolors=edgecolors,
            linewidths=0.35,
            zorder=2,
        )


def style_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_legends(ax, n_clusters: int, soft: bool = False) -> None:
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
    if soft:
        cluster_handles = [
            Patch(
                facecolor=CLUSTER_COLORS[k],
                edgecolor="none",
                label=f"cluster {k} (soft basis)",
            )
            for k in range(n_clusters)
        ]
        cluster_title = "Soft mix (RGB blend)"
    else:
        cluster_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=CLUSTER_COLORS[k],
                markerfacecolor=CLUSTER_COLORS[k],
                linestyle="None",
                markersize=9,
                label=f"cluster {k}",
            )
            for k in range(n_clusters)
        ]
        cluster_title = "Hard label (color)"

    leg_t = ax.legend(
        handles=topic_handles,
        title="GT topic (shape)",
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg_t)
    ax.legend(
        handles=cluster_handles,
        title=cluster_title,
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGN)
    parser.add_argument("--umap-coords", type=Path, default=DEFAULT_UMAP_COORDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute UMAP instead of loading cached coords.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    assigns = load_library_assignments(args.assignments)
    coords = get_coords(
        args.emb_dir,
        args.umap_coords,
        args.recompute,
        args.seed,
        args.n_neighbors,
        args.min_dist,
    )
    if coords.shape[0] != len(assigns):
        raise SystemExit(
            f"Row mismatch: {coords.shape[0]} UMAP coords vs "
            f"{len(assigns)} library assignments"
        )

    soft_cols = sorted(
        (c for c in assigns[0] if c.startswith("soft_w") and c[7:].isdigit()),
        key=lambda c: int(c[7:]),
    )
    n_clusters = len(soft_cols)
    hard = np.array([int(r["hard_label"]) for r in assigns], dtype=np.int32)
    soft = np.array([[float(r[c]) for c in soft_cols] for r in assigns], dtype=np.float64)
    topics = [r.get("topic", "?") for r in assigns]
    languages = [r.get("language") for r in assigns]

    palette = CLUSTER_COLORS
    if n_clusters > palette.shape[0]:
        # Extend with matplotlib tab colors if K > 3.
        import matplotlib as mpl

        cmap = mpl.colormaps["tab10"]
        palette = np.array([cmap(i)[:3] for i in range(n_clusters)], dtype=np.float64)

    hard_colors = palette[hard]
    soft_colors = soft_rgb(soft, palette)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- hard ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter_by_topic(ax, coords, topics, hard_colors)
    style_axes(ax, "Library posts — UMAP colored by hard KMeans label")
    add_legends(ax, n_clusters, soft=False)
    fig.tight_layout()
    hard_png = args.out_dir / "umap_hard_clusters.png"
    hard_pdf = args.out_dir / "umap_hard_clusters.pdf"
    fig.savefig(hard_png, bbox_inches="tight")
    fig.savefig(hard_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {hard_png}")

    # --- soft ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter_by_topic(ax, coords, topics, soft_colors)
    style_axes(ax, "Library posts — UMAP colored by soft cluster weights (RGB blend)")
    add_legends(ax, n_clusters, soft=True)
    fig.tight_layout()
    soft_png = args.out_dir / "umap_soft_clusters.png"
    soft_pdf = args.out_dir / "umap_soft_clusters.pdf"
    fig.savefig(soft_png, bbox_inches="tight")
    fig.savefig(soft_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {soft_png}")

    # Side-by-side panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
    scatter_by_topic(axes[0], coords, topics, hard_colors)
    style_axes(axes[0], "Hard label")
    add_legends(axes[0], n_clusters, soft=False)
    scatter_by_topic(axes[1], coords, topics, soft_colors)
    style_axes(axes[1], "Soft weights (RGB blend)")
    add_legends(axes[1], n_clusters, soft=True)
    fig.suptitle("Library posts — UMAP by KMeans assignment", fontsize=13, y=1.01)
    fig.tight_layout()
    both_png = args.out_dir / "umap_hard_soft_clusters.png"
    fig.savefig(both_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {both_png}")

    # Coords + assignment sidecar for the 600 library rows
    meta_path = args.out_dir / "umap_cluster_coords.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(assigns):
            out = {
                "index": i,
                "post_id": row.get("post_id"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "hard_label": int(row["hard_label"]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
            for c in soft_cols:
                out[c] = float(row[c])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    info = {
        "n_points": len(assigns),
        "n_clusters": n_clusters,
        "umap_coords": str(args.umap_coords),
        "recomputed": bool(args.recompute),
        "assignments": str(args.assignments),
        "encoding": {
            "hard": "face color = CLUSTER_COLORS[hard_label]; marker = GT topic",
            "soft": "face color = soft_w @ CLUSTER_COLORS (RGB blend); marker = GT topic",
            "topic_markers": TOPIC_MARKERS,
            "cluster_colors_rgb": palette[:n_clusters].tolist(),
        },
        "files": {
            "umap_hard_clusters.png": "Hard-label colored UMAP",
            "umap_soft_clusters.png": "Soft-weight RGB-blend UMAP",
            "umap_hard_soft_clusters.png": "Side-by-side hard vs soft",
            "umap_cluster_coords.jsonl": "UMAP x/y + hard/soft assignments",
        },
        "language_counts": {
            str(lang): int(sum(1 for L in languages if L == lang))
            for lang in sorted({L for L in languages if L is not None})
        },
    }
    (args.out_dir / "umap_cluster_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved coords sidecar → {meta_path}")


if __name__ == "__main__":
    main()
