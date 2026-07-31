#!/usr/bin/env python3
"""UMAP 2D scatters for claim–stance embeddings (vanilla vs stance-instruct).

Ground-truth encoding (mirrors topic/language plots):
  - **color** = stance (supportive / critical / neutral)
  - **marker shape** = aspect (economic, environmental, …)

Writes per-mode plots plus a side-by-side comparison figure under
``results/claim_stance_umap/``.
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
DEFAULT_EMB_ROOT = (
    ROOT / "data" / "embeddings" / "claims_stances_qwen3-embedding-0.6b"
)
DEFAULT_OUT_DIR = ROOT / "results" / "claim_stance_umap"

STANCE_COLORS = {
    "supportive": "#2ca02c",  # green
    "critical": "#d62728",  # red
    "neutral": "#7f7f7f",  # gray
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

MODE_TITLES = {
    "vanilla": "Vanilla (no instruction)",
    "stance_instruct": "Stance-instructed",
}


def load_ids(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def stance_separation_metrics(embeddings: np.ndarray, stances: list[str]) -> dict:
    """Simple GT-stance separability diagnostics in embedding space."""
    le = LabelEncoder()
    y = le.fit_transform(stances)
    out: dict = {
        "stances": list(le.classes_),
        "silhouette_cosine": None,
        "nearest_centroid_accuracy": None,
    }
    if len(set(stances)) < 2:
        return out
    try:
        out["silhouette_cosine"] = float(
            silhouette_score(embeddings, y, metric="cosine")
        )
    except Exception as exc:  # noqa: BLE001
        out["silhouette_error"] = str(exc)

    clf = NearestCentroid(metric="euclidean")
    # embeddings are L2-normalized → euclidean NN-centroid ≈ cosine
    clf.fit(embeddings, y)
    pred = clf.predict(embeddings)
    out["nearest_centroid_accuracy"] = float((pred == y).mean())

    # Mean pairwise cosine between stance centroids
    cents = {}
    for label in le.classes_:
        mask = np.asarray(stances) == label
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


def scatter_stance_aspect(
    ax: plt.Axes,
    coords: np.ndarray,
    stances: list[str],
    aspects: list[str],
    title: str,
) -> None:
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
                s=28,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )

    stance_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            linestyle="None",
            markersize=8,
            label=STANCE_LABELS.get(stance, stance),
        )
        for stance, color in STANCE_COLORS.items()
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
        handles=stance_handles,
        title="Stance (color)",
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
    metrics: dict,
    emb_dir: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    model_label: str,
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
                        "claim_id": row["claim_id"],
                        "aspect": row["aspect"],
                        "stance": row["stance"],
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stances = [r["stance"] for r in ids]
    aspects = [r["aspect"] for r in ids]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter_stance_aspect(
        ax,
        coords,
        stances,
        aspects,
        title=f"{model_label} — {MODE_TITLES.get(mode, mode)}",
    )
    # Annotate silhouette on the figure
    sil = metrics.get("silhouette_cosine")
    nca = metrics.get("nearest_centroid_accuracy")
    if sil is not None and nca is not None:
        ax.text(
            0.98,
            0.02,
            f"stance silhouette (cos)={sil:.3f}\nnearest-centroid acc={nca:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ddd"),
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
        "embedding_source": str(emb_dir.resolve().relative_to(ROOT)),
        "model_label": model_label,
        "encoding": {
            "stance_colors": STANCE_COLORS,
            "aspect_markers": ASPECT_MARKERS,
        },
        "stance_separation": metrics,
        "files": {
            "umap_scatter.png": "2D scatter (PNG)",
            "umap_scatter.pdf": "2D scatter (PDF)",
            "umap_coords.npy": "(n, 2) float32",
            "umap_coords.jsonl": "Coordinates + claim/stance/aspect",
        },
    }
    (mode_dir / "umap_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{mode}] plot -> {png_path}")
    return png_path


def infer_model_label(emb_root: Path) -> str:
    for mode in ("vanilla", "stance_instruct"):
        info_path = emb_root / mode / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            model = info.get("model")
            if model:
                return str(model).split("/")[-1]
    # Fallback from directory name: claims_stances_qwen3-embedding-4b
    name = emb_root.name
    if name.startswith("claims_stances_"):
        slug = name[len("claims_stances_") :]
        parts = slug.split("-")
        return "-".join(p.upper() if p in {"qwen3"} else p for p in parts)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: results/claim_stance_umap or …_<model-slug>).",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Label used in plot titles (default: inferred from embeddings info.json).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vanilla", "stance_instruct"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    model_label = args.model_label or infer_model_label(args.emb_root)
    if args.out_dir is None:
        if args.emb_root.resolve() == DEFAULT_EMB_ROOT.resolve():
            args.out_dir = DEFAULT_OUT_DIR
        else:
            slug = args.emb_root.name.replace("claims_stances_", "")
            if slug and slug != args.emb_root.name:
                args.out_dir = ROOT / "results" / f"claim_stance_umap_{slug}"
            else:
                args.out_dir = DEFAULT_OUT_DIR

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode_coords: dict[str, np.ndarray] = {}
    mode_ids: dict[str, list[dict]] = {}
    mode_metrics: dict[str, dict] = {}

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
        stances = [r["stance"] for r in ids]
        aspects = [r["aspect"] for r in ids]
        print(
            f"[{mode}] UMAP on {embeddings.shape} "
            f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"
        )
        metrics = stance_separation_metrics(embeddings, stances)
        print(
            f"[{mode}] silhouette={metrics.get('silhouette_cosine')} "
            f"centroid_acc={metrics.get('nearest_centroid_accuracy')} "
            f"centroid_cos={metrics.get('centroid_cosine')}"
        )
        coords = run_umap(
            embeddings, args.seed, args.n_neighbors, args.min_dist
        )
        save_mode_outputs(
            args.out_dir,
            mode,
            coords,
            ids,
            metrics,
            emb_dir,
            args.seed,
            args.n_neighbors,
            args.min_dist,
            model_label,
        )
        mode_coords[mode] = coords
        mode_ids[mode] = ids
        mode_metrics[mode] = metrics

    # Side-by-side comparison when both modes present
    if "vanilla" in mode_coords and "stance_instruct" in mode_coords:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
        for ax, mode in zip(axes, ["vanilla", "stance_instruct"]):
            ids = mode_ids[mode]
            scatter_stance_aspect(
                ax,
                mode_coords[mode],
                [r["stance"] for r in ids],
                [r["aspect"] for r in ids],
                title=MODE_TITLES[mode],
            )
            m = mode_metrics[mode]
            sil = m.get("silhouette_cosine")
            nca = m.get("nearest_centroid_accuracy")
            if sil is not None and nca is not None:
                ax.text(
                    0.98,
                    0.02,
                    f"sil={sil:.3f}  nca={nca:.3f}",
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
            f"{model_label} — AI datacenter claim–stance UMAP "
            "(color=stance, shape=aspect)",
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
        "model_label": model_label,
        "embedding_root": str(args.emb_root.resolve().relative_to(ROOT)),
        "modes": args.modes,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "random_state": args.seed,
        "encoding": {
            "stance_colors": STANCE_COLORS,
            "aspect_markers": ASPECT_MARKERS,
        },
        "stance_separation": mode_metrics,
        "files": {
            "umap_compare_vanilla_vs_instruct.png": "Side-by-side comparison",
            "vanilla/": "Per-mode UMAP + metrics",
            "stance_instruct/": "Per-mode UMAP + metrics",
        },
    }
    (args.out_dir / "umap_info.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Claim–stance UMAP plots ({model_label})",
                "",
                f"{model_label} projections of the AI-datacenter claim–stance set.",
                "",
                "## Encoding",
                "",
                "- **Color** = ground-truth `stance` (supportive / critical / neutral)",
                "- **Marker shape** = ground-truth `aspect`",
                "",
                "## Modes",
                "",
                "| Dir | Embedding |",
                "| --- | --- |",
                "| `vanilla/` | No instruction |",
                "| `stance_instruct/` | Instructed to encode stance |",
                "| `umap_compare_vanilla_vs_instruct.png` | Side-by-side |",
                "",
                "Each mode folder also stores `umap_info.json` with silhouette /",
                "nearest-centroid stance-separability metrics in the original",
                "embedding space (not UMAP).",
                "",
                "```bash",
                f"python scripts/embed_claim_stances.py --model <model-id>",
                f"python scripts/plot_claim_stance_umap.py --emb-root {args.emb_root.resolve().relative_to(ROOT)}",
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
