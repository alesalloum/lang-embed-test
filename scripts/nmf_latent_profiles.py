#!/usr/bin/env python3
"""NMF latent-profile recovery and rank selection via KL elbow curves.

Loads an existing user×microtopic attention matrix ``X`` (rows are probability
distributions), fits ``sklearn.decomposition.NMF`` for ``k = 1 .. k_max`` with
KL / multiplicative updates, and reports reconstruction metrics + elbow plots.

Example::

    python scripts/nmf_latent_profiles.py
    python scripts/nmf_latent_profiles.py \\
        --attention-csv results/topic_attention_k20/user_attention_soft.csv \\
        --out-dir results/nmf_latent_profiles \\
        --k-max 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTENTION = ROOT / "results" / "topic_attention_k20" / "user_attention_soft.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "nmf_latent_profiles"

EPS = 1e-12


# ---------------------------------------------------------------------------
# Data loading / normalization
# ---------------------------------------------------------------------------


def load_attention_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load ``X`` (N_users × N_microtopics) and metadata from attention CSV."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty attention CSV: {path}")

    w_cols = sorted(
        (c for c in rows[0].keys() if c.startswith("w") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not w_cols:
        raise SystemExit(f"No w* weight columns in {path}")

    X = np.array([[float(r[c]) for c in w_cols] for r in rows], dtype=np.float64)
    user_ids = [r.get("user_id", str(i)) for i, r in enumerate(rows)]
    return X, w_cols, user_ids


def row_normalize(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    """Ensure each row is a probability distribution (sums to 1)."""
    row_sums = X.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        bad = int(np.sum(row_sums.ravel() <= 0))
        raise SystemExit(f"{name}: {bad} row(s) have non-positive mass")
    Xn = X / row_sums
    max_err = float(np.max(np.abs(Xn.sum(axis=1) - 1.0)))
    print(f"{name} shape={Xn.shape}  row-sum max|err| after normalize={max_err:.3e}")
    return Xn


def post_normalize_factors(
    W: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalize profiles ``H``; rescale ``W`` columns to preserve ``WH``.

    ``H_norm[r]`` sums to 1 (latent profile over microtopics).
    ``W_norm = W * H.row_sums.T`` so ``W_norm @ H_norm ≈ W @ H``.
    """
    h_sums = H.sum(axis=1, keepdims=True)  # (k, 1)
    h_sums = np.maximum(h_sums, EPS)
    H_norm = H / h_sums
    W_norm = W * h_sums.T
    return W_norm, H_norm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def kl_divergence(X: np.ndarray, Y: np.ndarray) -> float:
    """Generalized KL: ``sum(X log(X/Y) - X + Y)`` with flooring at ``EPS``."""
    Xf = np.maximum(X, EPS)
    Yf = np.maximum(Y, EPS)
    return float(np.sum(Xf * np.log(Xf / Yf) - Xf + Yf))


def reconstruction_perplexity(X: np.ndarray, Y: np.ndarray) -> float:
    """``exp(-1/N sum(X log(Y + eps)))`` with ``N = n_users``."""
    n_users = X.shape[0]
    cross_entropy = -float(np.sum(X * np.log(Y + EPS))) / n_users
    return float(np.exp(cross_entropy))


def mean_row_baseline(X: np.ndarray) -> np.ndarray:
    """``X_bar``: identical rows equal to the column-wise mean attention vector."""
    x_bar = X.mean(axis=0, keepdims=True)
    return np.repeat(x_bar, X.shape[0], axis=0)


# ---------------------------------------------------------------------------
# NMF sweep
# ---------------------------------------------------------------------------


def fit_nmf(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, NMF]:
    model = NMF(
        n_components=n_components,
        solver="mu",
        beta_loss="kullback-leibler",
        init="nndsvda",
        max_iter=1000,
        random_state=42,
    )
    W = model.fit_transform(X)
    H = model.components_
    return W, H, model


def evaluate_rank(X: np.ndarray, k: int, kl_baseline: float) -> dict[str, Any]:
    W, H, model = fit_nmf(X, k)
    W_norm, H_norm = post_normalize_factors(W, H)
    Y = W_norm @ H_norm
    kl_loss = kl_divergence(X, Y)
    deviance_explained = 1.0 - (kl_loss / kl_baseline) if kl_baseline > 0 else 0.0
    perplexity = reconstruction_perplexity(X, Y)
    return {
        "k": k,
        "kl_loss": kl_loss,
        "deviance_explained": deviance_explained,
        "deviance_explained_pct": 100.0 * deviance_explained,
        "perplexity": perplexity,
        "n_iter": int(model.n_iter_),
        "reconstruction_err": float(model.reconstruction_err_),
        "W_norm": W_norm,
        "H_norm": H_norm,
    }


def select_elbow_k(ks: list[int], deviance_pct: list[float]) -> dict[str, Any]:
    """Knee / elbow via max perpendicular distance to chord (k_min → k_max).

    Also reports successive absolute gains so the flattening region is visible.
    """
    ks_arr = np.asarray(ks, dtype=np.float64)
    ys = np.asarray(deviance_pct, dtype=np.float64)
    gains = np.diff(ys)

    if len(ks) < 3:
        elbow = int(ks[int(np.argmax(ys))])
        return {
            "elbow_k": elbow,
            "method": "argmax_deviance (too few points for knee)",
            "marginal_gains_pct": gains.tolist(),
        }

    # Normalize to unit square for geometry that is not scale-dominated.
    x = (ks_arr - ks_arr[0]) / max(ks_arr[-1] - ks_arr[0], EPS)
    y = (ys - ys[0]) / max(ys[-1] - ys[0], EPS)
    # Distance from each point to the chord from first → last.
    # Line: p0 + t (p1 - p0); perpendicular distance in 2D.
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    denom = max(np.hypot(dx, dy), EPS)
    dist = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / denom
    # Prefer interior knees; ignore endpoints for the argmax.
    dist[0] = -np.inf
    dist[-1] = -np.inf
    elbow_idx = int(np.argmax(dist))
    elbow_k = int(ks[elbow_idx])

    # Sanity: if gains stay large through the end, prefer last strong gain drop.
    # Find first k where marginal gain falls below 50% of the largest early gain
    # and absolute gain < 5 percentage points.
    if gains.size:
        early = float(np.max(gains[: min(3, len(gains))]))
        thresh = max(0.5 * early, 5.0)
        flat_idx = None
        for i, g in enumerate(gains):
            if g < thresh and g < 5.0:
                flat_idx = i  # gain from ks[i] → ks[i+1] is small → elbow at ks[i]
                break
        gain_elbow = int(ks[flat_idx]) if flat_idx is not None else elbow_k
    else:
        gain_elbow = elbow_k

    # Prefer agreement; otherwise take the more conservative (smaller) of the two.
    recommended = min(elbow_k, gain_elbow)
    return {
        "elbow_k": recommended,
        "knee_distance_k": elbow_k,
        "gain_flatten_k": gain_elbow,
        "method": "min(knee_distance, gain_flatten)",
        "marginal_gains_pct": [float(g) for g in gains],
        "knee_distances": [
            None if not np.isfinite(d) else float(d) for d in dist
        ],
    }


# ---------------------------------------------------------------------------
# Reporting / plots
# ---------------------------------------------------------------------------


def print_metrics_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'k':>3}  {'KL_Loss':>14}  {'Deviance_Explained_%':>20}  {'Perplexity':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['k']:3d}  {r['kl_loss']:14.6f}  "
            f"{r['deviance_explained_pct']:20.4f}  {r['perplexity']:12.6f}"
        )


def plot_elbow(
    rows: list[dict[str, Any]],
    elbow_k: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    deviance = [r["deviance_explained_pct"] for r in rows]
    perplexity = [r["perplexity"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(ks, deviance, marker="o", color="#1f4e79", linewidth=2)
    ax.axvline(elbow_k, color="#c45c26", linestyle="--", linewidth=1.4, label=f"elbow k={elbow_k}")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Fraction of KL-deviance explained (%)")
    ax.set_title("KL-deviance explained vs k")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ks, perplexity, marker="o", color="#2f6b3a", linewidth=2)
    ax.axvline(elbow_k, color="#c45c26", linestyle="--", linewidth=1.4, label=f"elbow k={elbow_k}")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Reconstruction perplexity")
    ax.set_title("Perplexity vs k")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle("NMF rank selection on user attention", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out_path}")


def save_factors(
    out_dir: Path,
    best: dict[str, Any],
    microtopic_cols: list[str],
) -> None:
    np.save(out_dir / f"W_norm_k{best['k']}.npy", best["W_norm"])
    np.save(out_dir / f"H_norm_k{best['k']}.npy", best["H_norm"])

    # H as CSV: one latent profile per row
    h_path = out_dir / f"H_norm_k{best['k']}.csv"
    with h_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component"] + microtopic_cols)
        for i, row in enumerate(best["H_norm"]):
            writer.writerow([f"profile_{i}"] + [f"{v:.8f}" for v in row])
    print(f"Wrote factors for recommended k={best['k']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--attention-csv", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k-min", type=int, default=1)
    args = parser.parse_args()

    if not args.attention_csv.exists():
        raise SystemExit(
            f"Missing {args.attention_csv}. "
            "Run cluster_user_attention.py with --n-clusters >= k-max first."
        )

    X_raw, w_cols, user_ids = load_attention_matrix(args.attention_csv)
    X = row_normalize(X_raw, name="X")

    n_users, n_micro = X.shape
    k_max = min(args.k_max, n_micro, n_users)
    if k_max < args.k_max:
        print(
            f"Note: capping k_max at {k_max} "
            f"(min of requested={args.k_max}, n_microtopics={n_micro}, n_users={n_users})"
        )
    if args.k_min < 1 or args.k_min > k_max:
        raise SystemExit(f"Invalid k range: [{args.k_min}, {k_max}]")

    X_bar = mean_row_baseline(X)
    kl_baseline = kl_divergence(X, X_bar)
    print(f"Baseline D_kl(X || X_bar) = {kl_baseline:.6f}")
    print(
        f"Fitting NMF for k={args.k_min}..{k_max} "
        "(solver=mu, beta_loss=kullback-leibler, init=nndsvda, max_iter=1000, rs=42)"
    )

    results: list[dict[str, Any]] = []
    for k in range(args.k_min, k_max + 1):
        row = evaluate_rank(X, k, kl_baseline)
        print(
            f"  k={k:2d}  KL={row['kl_loss']:.6f}  "
            f"expl={row['deviance_explained_pct']:.2f}%  "
            f"ppl={row['perplexity']:.4f}  n_iter={row['n_iter']}"
        )
        results.append(row)

    print()
    print_metrics_table(results)

    elbow_info = select_elbow_k(
        [r["k"] for r in results],
        [r["deviance_explained_pct"] for r in results],
    )
    elbow_k = int(elbow_info["elbow_k"])
    best = next(r for r in results if r["k"] == elbow_k)

    print()
    print(f"Recommended elbow / optimal k = {elbow_k}")
    print(
        f"  method={elbow_info['method']}; "
        f"marginal gains (% points) = "
        f"{[round(g, 3) for g in elbow_info['marginal_gains_pct']]}"
    )
    print(
        f"  at k={elbow_k}: KL_Loss={best['kl_loss']:.6f}, "
        f"Deviance_Explained={best['deviance_explained_pct']:.2f}%, "
        f"Perplexity={best['perplexity']:.6f}"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "nmf_elbow_curves.png"
    plot_elbow(results, elbow_k, plot_path)
    save_factors(out_dir, best, w_cols)

    summary = {
        "attention_csv": str(args.attention_csv),
        "n_users": n_users,
        "n_microtopics": n_micro,
        "microtopic_columns": w_cols,
        "k_min": args.k_min,
        "k_max": k_max,
        "nmf_hyperparameters": {
            "solver": "mu",
            "beta_loss": "kullback-leibler",
            "init": "nndsvda",
            "max_iter": 1000,
            "random_state": 42,
        },
        "kl_baseline_X_bar": kl_baseline,
        "row_sum_check_max_abs_err": float(np.max(np.abs(X.sum(axis=1) - 1.0))),
        "metrics": [
            {
                "k": r["k"],
                "kl_loss": r["kl_loss"],
                "deviance_explained": r["deviance_explained"],
                "deviance_explained_pct": r["deviance_explained_pct"],
                "perplexity": r["perplexity"],
                "n_iter": r["n_iter"],
            }
            for r in results
        ],
        "elbow": elbow_info,
        "recommended_k": elbow_k,
        "notes": (
            "X is soft user attention over KMeans microtopics. "
            "With topic-bridge mode, user rows are mixtures of a small number of "
            "topic→microtopic signatures, so the effective rank is often near the "
            "number of ground-truth topics (here 3)."
        ),
    }
    (out_dir / "nmf_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Compact text table for quick inspection
    table_path = out_dir / "nmf_metrics_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "KL_Loss", "Deviance_Explained_%", "Perplexity"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "k": r["k"],
                    "KL_Loss": f"{r['kl_loss']:.8f}",
                    "Deviance_Explained_%": f"{r['deviance_explained_pct']:.6f}",
                    "Perplexity": f"{r['perplexity']:.8f}",
                }
            )

    readme = f"""# NMF latent-profile rank selection

Fitted `sklearn.decomposition.NMF` (MU + KL, `init=nndsvda`, `max_iter=1000`,
`random_state=42`) on user attention
[`{args.attention_csv}`]({args.attention_csv.relative_to(ROOT) if args.attention_csv.is_relative_to(ROOT) else args.attention_csv})
shape **({n_users} × {n_micro})**.

## Recommended rank

**k = {elbow_k}** ({elbow_info['method']})

| Metric | Value |
| --- | --- |
| KL loss | {best['kl_loss']:.6f} |
| Deviance explained | {best['deviance_explained_pct']:.2f}% |
| Perplexity | {best['perplexity']:.6f} |

## Artifacts

| File | Description |
| --- | --- |
| `nmf_elbow_curves.png` / `.pdf` | Side-by-side deviance-explained & perplexity vs k |
| `nmf_metrics.json` | Full metric sweep + elbow metadata |
| `nmf_metrics_table.csv` | Compact `[k, KL_Loss, Deviance_Explained_%, Perplexity]` |
| `W_norm_k{elbow_k}.npy` | User×component weights (columns scaled) |
| `H_norm_k{elbow_k}.npy` / `.csv` | Latent profiles over microtopics (rows sum to 1) |

## Reproduce

```bash
python scripts/cluster_user_attention.py --n-clusters 20 --out-dir results/topic_attention_k20
python scripts/nmf_latent_profiles.py \\
  --attention-csv results/topic_attention_k20/user_attention_soft.csv \\
  --out-dir results/nmf_latent_profiles --k-max 10
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote summary → {out_dir / 'nmf_metrics.json'}")


if __name__ == "__main__":
    main()
