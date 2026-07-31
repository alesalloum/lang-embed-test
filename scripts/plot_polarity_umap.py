#!/usr/bin/env python3
"""UMAP 2D scatters for phenomenon-polarity embeddings (vanilla vs instruct).

Ground-truth encoding:
  - **color** = polarity (pro / against / neutral)
  - **marker shape** = aspect (economic, environmental, …)

Writes per-mode plots plus a side-by-side comparison figure under
``results/polarity_umap/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_ROOT = ROOT / "data" / "embeddings" / "polarity_qwen3-embedding-0.6b"
DEFAULT_OUT_DIR = ROOT / "results" / "polarity_umap"

POLARITY_COLORS = {
    "pro": "#2ca02c",
    "against": "#d62728",
    "neutral": "#7f7f7f",
}

POLARITY_LABELS = {
    "pro": "Pro",
    "against": "Against",
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

MODE_TITLES = {
    "vanilla": "Vanilla (no instruction)",
    "polarity_instruct": "Polarity-instructed",
}


def load_ids(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def polarity_separation_metrics(embeddings: np.ndarray, polarities: list[str]) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(polarities)
    out: dict = {
        "polarities": list(le.classes_),
        "silhouette_cosine": None,
        "nearest_centroid_accuracy": None,
    }
    if len(set(polarities)) < 2:
        return out
    try:
        out["silhouette_cosine"] = float(
            silhouette_score(embeddings, y, metric="cosine")
        )
    except Exception as exc:  # noqa: BLE001
        out["silhouette_error"] = str(exc)

    clf = NearestCentroid(metric="euclidean")
    clf.fit(embeddings, y)
    pred = clf.predict(embeddings)
    out["nearest_centroid_accuracy"] = float((pred == y).mean())

    cents = {}
    for label in le.classes_:
        mask = np.asarray(polarities) == label
        v = embeddings[mask].mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-12)
        cents[label] = v
    cos = {}
    labels = list(le.classes_)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            cos[f"{a}__{b}"] = float(np.dot(cents[a], cents[b]))
    out["centroid_cosine"] = cos
    return out


def aspect_separation_metrics(embeddings: np.ndarray, aspects: list[str]) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(aspects)
    out: dict = {
        "aspects": list(le.classes_),
        "silhouette_cosine": None,
        "nearest_centroid_accuracy": None,
    }
    if len(set(aspects)) < 2:
        return out
    try:
        out["silhouette_cosine"] = float(
            silhouette_score(embeddings, y, metric="cosine")
        )
    except Exception as exc:  # noqa: BLE001
        out["silhouette_error"] = str(exc)
    clf = NearestCentroid(metric="euclidean")
    clf.fit(embeddings, y)
    out["nearest_centroid_accuracy"] = float((clf.predict(embeddings) == y).mean())
    return out


def run_umap(
    embeddings: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def scatter_polarity_aspect(
    ax: plt.Axes,
    coords: np.ndarray,
    polarities: list[str],
    aspects: list[str],
    title: str,
) -> None:
    for aspect, marker in ASPECT_MARKERS.items():
        for polarity, color in POLARITY_COLORS.items():
            mask = np.asarray(
                [
                    (p == polarity and a == aspect)
                    for p, a in zip(polarities, aspects)
                ]
            )
            pts = coords[mask]
            if len(pts) == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                marker=marker,
                s=28,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )

    polarity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            linestyle="None",
            markersize=8,
            label=POLARITY_LABELS.get(polarity, polarity),
        )
        for polarity, color in POLARITY_COLORS.items()
    ]
    aspect_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            markerfacecolor="gray",
            linestyle="None",
            markersize=8,
            label=ASPECT_LABELS.get(aspect, aspect),
        )
        for aspect, marker in ASPECT_MARKERS.items()
    ]

    leg1 = ax.legend(
        handles=polarity_handles,
        title="Polarity (color)",
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=aspect_handles,
        title="Aspect (shape)",
        loc="lower left",
        frameon=True,
        fontsize=7,
        title_fontsize=8,
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_mode_outputs(
    out_dir: Path,
    mode: str,
    coords: np.ndarray,
    ids: list[dict],
    polarity_metrics: dict,
    aspect_metrics: dict,
    emb_dir: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Path:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    np.save(mode_dir / "umap_coords.npy", coords.astype(np.float32))
    with (mode_dir / "umap_coords.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(ids):
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "post_id": row["post_id"],
                        "aspect": row["aspect"],
                        "polarity": row["polarity"],
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    polarities = [r["polarity"] for r in ids]
    aspects = [r["aspect"] for r in ids]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter_polarity_aspect(
        ax,
        coords,
        polarities,
        aspects,
        title=f"Qwen3-Embedding-0.6B — {MODE_TITLES.get(mode, mode)}",
    )
    sil = polarity_metrics.get("silhouette_cosine")
    nca = polarity_metrics.get("nearest_centroid_accuracy")
    a_sil = aspect_metrics.get("silhouette_cosine")
    if sil is not None and nca is not None:
        ax.text(
            0.98,
            0.02,
            (
                f"polarity silhouette (cos)={sil:.3f}\n"
                f"nearest-centroid acc={nca:.3f}\n"
                f"aspect silhouette (cos)={a_sil:.3f}"
                if a_sil is not None
                else f"polarity silhouette (cos)={sil:.3f}\nnearest-centroid acc={nca:.3f}"
            ),
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
    fig.tight_layout()
    png_path = mode_dir / "umap_scatter.png"
    pdf_path = mode_dir / "umap_scatter.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    info = {
        "mode": mode,
        "method": "UMAP",
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": "cosine",
        "random_state": seed,
        "n_points": int(coords.shape[0]),
        "embedding_source": str(emb_dir.relative_to(ROOT)),
        "encoding": {
            "polarity_colors": POLARITY_COLORS,
            "aspect_markers": ASPECT_MARKERS,
        },
        "polarity_separation": polarity_metrics,
        "aspect_separation": aspect_metrics,
        "files": {
            "umap_scatter.png": "2D scatter (PNG)",
            "umap_scatter.pdf": "2D scatter (PDF)",
            "umap_coords.npy": "(n, 2) float32",
            "umap_coords.jsonl": "Coordinates + polarity/aspect",
        },
    }
    (mode_dir / "umap_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{mode}] plot -> {png_path}")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vanilla", "polarity_instruct"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode_coords: dict[str, np.ndarray] = {}
    mode_ids: dict[str, list[dict]] = {}
    mode_polarity_metrics: dict[str, dict] = {}
    mode_aspect_metrics: dict[str, dict] = {}

    for mode in args.modes:
        emb_dir = args.emb_root / mode
        emb_path = emb_dir / "embeddings.npy"
        ids_path = emb_dir / "ids.json"
        if not emb_path.exists():
            raise SystemExit(f"Missing embeddings: {emb_path}")
        embeddings = np.load(emb_path)
        ids = load_ids(ids_path)
        if len(ids) != embeddings.shape[0]:
            raise SystemExit(
                f"Row mismatch for {mode}: {embeddings.shape[0]} emb vs {len(ids)} ids"
            )
        polarities = [r["polarity"] for r in ids]
        aspects = [r["aspect"] for r in ids]
        print(
            f"[{mode}] UMAP on {embeddings.shape} "
            f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"
        )
        p_metrics = polarity_separation_metrics(embeddings, polarities)
        a_metrics = aspect_separation_metrics(embeddings, aspects)
        print(
            f"[{mode}] polarity sil={p_metrics.get('silhouette_cosine')} "
            f"nca={p_metrics.get('nearest_centroid_accuracy')} "
            f"aspect sil={a_metrics.get('silhouette_cosine')}"
        )
        coords = run_umap(embeddings, args.seed, args.n_neighbors, args.min_dist)
        save_mode_outputs(
            args.out_dir,
            mode,
            coords,
            ids,
            p_metrics,
            a_metrics,
            emb_dir,
            args.seed,
            args.n_neighbors,
            args.min_dist,
        )
        mode_coords[mode] = coords
        mode_ids[mode] = ids
        mode_polarity_metrics[mode] = p_metrics
        mode_aspect_metrics[mode] = a_metrics

    if "vanilla" in mode_coords and "polarity_instruct" in mode_coords:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
        for ax, mode in zip(axes, ["vanilla", "polarity_instruct"]):
            ids = mode_ids[mode]
            scatter_polarity_aspect(
                ax,
                mode_coords[mode],
                [r["polarity"] for r in ids],
                [r["aspect"] for r in ids],
                title=MODE_TITLES[mode],
            )
            m = mode_polarity_metrics[mode]
            a = mode_aspect_metrics[mode]
            sil = m.get("silhouette_cosine")
            nca = m.get("nearest_centroid_accuracy")
            a_sil = a.get("silhouette_cosine")
            if sil is not None and nca is not None:
                ax.text(
                    0.98,
                    0.02,
                    f"pol sil={sil:.3f}  nca={nca:.3f}\naspect sil={a_sil:.3f}",
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
        fig.suptitle(
            "AI datacenter polarity posts — UMAP (color=polarity, shape=aspect)",
            fontsize=13,
            y=1.02,
        )
        fig.tight_layout()
        cmp_png = args.out_dir / "umap_compare_vanilla_vs_instruct.png"
        cmp_pdf = args.out_dir / "umap_compare_vanilla_vs_instruct.pdf"
        fig.savefig(cmp_png, bbox_inches="tight")
        fig.savefig(cmp_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare] plot -> {cmp_png}")

    summary = {
        "embedding_root": str(args.emb_root.relative_to(ROOT)),
        "modes": args.modes,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "random_state": args.seed,
        "encoding": {
            "polarity_colors": POLARITY_COLORS,
            "aspect_markers": ASPECT_MARKERS,
        },
        "polarity_separation": mode_polarity_metrics,
        "aspect_separation": mode_aspect_metrics,
        "files": {
            "umap_compare_vanilla_vs_instruct.png": "Side-by-side comparison",
            "vanilla/": "Per-mode UMAP + metrics",
            "polarity_instruct/": "Per-mode UMAP + metrics",
        },
    }
    (args.out_dir / "umap_info.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Phenomenon-polarity UMAP plots",
                "",
                "Qwen3-Embedding-0.6B projections of the AI-datacenter polarity set.",
                "",
                "## Encoding",
                "",
                "- **Color** = ground-truth `polarity` (pro / against / neutral)",
                "- **Marker shape** = ground-truth `aspect`",
                "",
                "## Modes",
                "",
                "| Dir | Embedding |",
                "| --- | --- |",
                "| `vanilla/` | No instruction |",
                "| `polarity_instruct/` | Instructed to encode polarity |",
                "| `umap_compare_vanilla_vs_instruct.png` | Side-by-side |",
                "",
                "Metrics in `umap_info.json` are computed in the original",
                "embedding space (not UMAP).",
                "",
                "```bash",
                "python scripts/embed_polarity.py",
                "python scripts/plot_polarity_umap.py",
                "```",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary -> {args.out_dir / 'umap_info.json'}")


if __name__ == "__main__":
    main()
