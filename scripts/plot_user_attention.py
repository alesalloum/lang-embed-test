#!/usr/bin/env python3
"""Lightweight plots for user topic-attention vectors.

Produces:
  * global topic mass bar chart
  * 2D simplex (ternary-style) projection of length-3 attention vectors,
    colored by dominant topic

Skipped gracefully if matplotlib is missing. For K != 3 the simplex view is
skipped (bars still drawn).

Example::

    python scripts/plot_user_attention.py
    python scripts/plot_user_attention.py --attention-type soft
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "results" / "topic_attention_k3"
DEFAULT_OUT_DIR = ROOT / "results" / "topic_attention_k3"


def load_user_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def weights_matrix(rows: list[dict]) -> tuple["np.ndarray", int]:
    import numpy as np

    w_cols = sorted(
        (c for c in rows[0].keys() if c.startswith("w") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    W = np.array([[float(r[c]) for c in w_cols] for r in rows], dtype=np.float64)
    return W, len(w_cols)


def simplex_xy(W: "np.ndarray") -> "np.ndarray":
    """Map 3-simplex weights to 2D equilateral triangle coordinates."""
    import numpy as np

    # Vertices: (0,0), (1,0), (0.5, √3/2)
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    return W[:, 0:1] * v0 + W[:, 1:2] * v1 + W[:, 2:3] * v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--attention-type",
        choices=("soft", "hard"),
        default="soft",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        print(f"Skipping plots (missing dependency: {exc})")
        return

    csv_path = args.in_dir / f"user_attention_{args.attention_type}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}; run cluster_user_attention.py first.")

    rows = load_user_csv(csv_path)
    if not rows:
        raise SystemExit(f"No user rows in {csv_path}")

    W, k = weights_matrix(rows)
    mass = W.mean(axis=0)
    dominant = W.argmax(axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- global topic mass ---
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(k)
    colors = [f"C{i}" for i in range(k)]
    ax.bar(xs, mass, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"topic {i}" for i in xs])
    ax.set_ylabel("mean user weight")
    ax.set_ylim(0, max(0.5, float(mass.max()) * 1.25))
    ax.set_title(f"Global topic mass ({args.attention_type})")
    for i, v in enumerate(mass):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    bar_path = args.out_dir / f"topic_mass_{args.attention_type}.png"
    fig.savefig(bar_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {bar_path}")

    # --- simplex / 2D view (K=3 only) ---
    if k != 3:
        print(f"Skipping simplex plot (K={k}, need 3).")
        return

    xy = simplex_xy(W)
    # slight jitter for overlapping points at vertices
    rng = np.random.default_rng(args.seed)
    xy = xy + rng.normal(0, 0.004, size=xy.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    for c in range(3):
        mask = dominant == c
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=18,
            alpha=0.65,
            c=f"C{c}",
            label=f"dominant topic {c} (n={int(mask.sum())})",
            edgecolors="none",
        )
    # triangle outline
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.0)
    ax.text(-0.02, -0.04, "w0", ha="right", va="top")
    ax.text(1.02, -0.04, "w1", ha="left", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "w2", ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"User attention simplex ({args.attention_type})")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    sim_path = args.out_dir / f"user_attention_simplex_{args.attention_type}.png"
    fig.savefig(sim_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {sim_path}")

    # Also dump a tiny sidecar for notebooks
    qc_path = args.in_dir / "qc_summary.json"
    meta = {"attention_type": args.attention_type, "n_users": len(rows), "k": k}
    if qc_path.exists():
        meta["qc_mode"] = json.loads(qc_path.read_text(encoding="utf-8")).get("mode")
    (args.out_dir / f"plot_info_{args.attention_type}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
