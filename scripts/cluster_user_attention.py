#!/usr/bin/env python3
"""Cluster post embeddings (KMeans) and build per-user topic attention vectors.

Pipeline
--------
1. Load on-disk post embeddings (do not re-embed). L2-normalize if needed.
2. Fit spherical KMeans: L2-normalized vectors + sklearn KMeans (cosine geometry).
3. Assign every post a hard label and temperature-scaled soft weights over
   cosine similarities to the K centroids (or pure hard one-hot).
4. Aggregate per user → length-K attention vectors (L1-normalized). Optional
   Dirichlet shrinkage toward the global topic prior for low-activity users.
5. Write centroids, post assignments, user attention tables, and a QC summary.

Data modes
----------
* **Aligned meta** — ``metadata.jsonl`` / posts file rows match ``embeddings.npy``
  and include ``user_id``. Assignments come directly from each post's vector.
* **User-posts + topic bridge** (default for this repo) — embeddings cover the
  multilingual library posts (no ``user_id``). ``user_posts.jsonl`` has
  ``user_id`` + ``topic`` but no embeddings. After clustering, each ground-truth
  ``topic`` is mapped to the mean soft/hard assignment of library posts with
  that topic (optionally language-filtered, default ``en``). User posts inherit
  those weights, then are aggregated. Use this only as a histogram-pipeline
  check until user-post embeddings exist.
* **User embeddings** — ``--user-embeddings`` + ``--user-posts`` with matching
  row order: predict labels from the fitted centroids (no topic bridge).

Output schema
-------------
``centroids.npy``
    ``(K, d)`` float32 L2-normalized cluster centers.

``train_config.json``
    Fit hyperparameters and paths (``n_clusters``, ``random_state``, ``n_init``,
    ``tau``, ``alpha``, ``assignment_variant``, subsample size, etc.).

``post_assignments.jsonl`` / ``.csv``
    One row per assigned post::

        post_id, user_id, hard_label, soft_w0 .. soft_w{K-1}
        [, topic, language, source]

    ``user_id`` may be null for library posts. Soft columns are always present
    (one-hot when ``--assignment-variant hard``).

``user_attention_soft.jsonl`` / ``.csv``
``user_attention_hard.jsonl`` / ``.csv``
    One row per user::

        user_id, n_posts, w0 .. w{K-1}, assignment_type
        [, profile_id, profile_label, topic_distribution,
           empirical_topic_distribution, gt_w0 .. gt_w{K-1}]

    Weights sum to 1. ``assignment_type`` is ``soft`` or ``hard``.
    When ``data/users.jsonl`` is available, ground-truth interest labels are
    joined on ``user_id`` for manual checks. ``gt_w*`` are the empirical
    topic mix remapped into cluster order (via topic→cluster bridge when used).

``qc_summary.json``
    Cluster sizes, inertia, pairwise centroid cosine, user-weight entropy
    stats, fraction of users with ``max(w) > vertex_threshold``, and nearest
    library ``post_id``s per centroid when texts/meta are available.

Example
-------
::

    pip install -r requirements.txt
    python scripts/cluster_user_attention.py
    python scripts/cluster_user_attention.py --n-clusters 3 --tau 0.1 --alpha 0
    python scripts/plot_user_attention.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_DIR = ROOT / "data" / "embeddings" / "qwen3-embedding-0.6b"
DEFAULT_USER_POSTS = ROOT / "data" / "user_posts.jsonl"
DEFAULT_USERS = ROOT / "data" / "users.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "topic_attention_k3"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fieldnames}
            writer.writerow(out)


def load_embedding_meta(emb_dir: Path) -> list[dict]:
    meta_path = emb_dir / "metadata.jsonl"
    ids_path = emb_dir / "ids.json"
    if meta_path.exists():
        return load_jsonl(meta_path)
    if ids_path.exists():
        return json.loads(ids_path.read_text(encoding="utf-8"))
    raise SystemExit(f"No metadata.jsonl or ids.json in {emb_dir}")


# ---------------------------------------------------------------------------
# Geometry / clustering
# ---------------------------------------------------------------------------


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def maybe_normalize(embeddings: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, bool]:
    """L2-normalize rows if they are not already ~unit length."""
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    already = bool(np.all(np.abs(norms - 1.0) < eps))
    if already:
        return embeddings.astype(np.float32), True
    return l2_normalize(embeddings), False


def fit_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
    subsample: int | None,
) -> tuple[KMeans, np.ndarray, int]:
    """Fit KMeans on (optionally subsampled) L2-normalized embeddings."""
    n = embeddings.shape[0]
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        train = embeddings[idx]
        n_fit = int(subsample)
    else:
        train = embeddings
        n_fit = n

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        algorithm="lloyd",
    )
    model.fit(train)
    # Re-normalize centroids so cosine == dot product at prediction time.
    centroids = l2_normalize(model.cluster_centers_)
    model.cluster_centers_ = centroids.astype(np.float64)
    return model, centroids, n_fit


def cosine_to_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix ``(n, K)`` for L2-normalized inputs."""
    return embeddings.astype(np.float64) @ centroids.astype(np.float64).T


def soft_from_cosine(cos: np.ndarray, tau: float) -> np.ndarray:
    """Temperature-scaled softmax over cosine similarities."""
    if tau <= 0:
        raise ValueError("tau must be > 0 for soft assignments")
    logits = cos / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def hard_one_hot(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], n_clusters), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def assign_posts(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard labels, soft weights, and cosine matrix."""
    cos = cosine_to_centroids(embeddings, centroids)
    hard = np.argmax(cos, axis=1).astype(np.int32)
    soft = soft_from_cosine(cos, tau)
    return hard, soft, cos


# ---------------------------------------------------------------------------
# User aggregation
# ---------------------------------------------------------------------------


def topic_weight_lookup(
    meta: Sequence[dict],
    soft: np.ndarray,
    hard_oh: np.ndarray,
    language_filter: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Mean soft assignment + majority hard one-hot per ground-truth topic."""
    soft_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    hard_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(meta):
        topic = row.get("topic")
        if topic is None:
            continue
        if language_filter and row.get("language") not in (None, language_filter):
            continue
        soft_acc[topic].append(soft[i])
        hard_acc[topic].append(hard_oh[i])
    if not soft_acc:
        raise SystemExit(
            "Topic bridge failed: no library posts with `topic` "
            f"(language_filter={language_filter!r})."
        )
    soft_map = {t: np.mean(np.stack(v), axis=0) for t, v in soft_acc.items()}
    hard_map: dict[str, np.ndarray] = {}
    for t, vecs in hard_acc.items():
        mean_h = np.mean(np.stack(vecs), axis=0)
        # Pure hard bridge: majority cluster among library posts of this topic.
        hard_map[t] = hard_one_hot(
            np.array([int(np.argmax(mean_h))], dtype=np.int32),
            mean_h.shape[0],
        )[0]
    counts = {t: len(v) for t, v in soft_acc.items()}
    return soft_map, hard_map, counts


def aggregate_user_weights(
    user_ids: Sequence[str],
    weights: np.ndarray,
    n_clusters: int,
    assignment_type: str,
    alpha: float,
    global_prior: np.ndarray | None = None,
) -> list[dict]:
    """Sum post weights per user, optional Dirichlet shrinkage, L1-normalize."""
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_clusters, dtype=np.float64))
    counts: Counter[str] = Counter()
    for uid, w in zip(user_ids, weights):
        sums[uid] += w
        counts[uid] += 1

    if global_prior is None:
        total = np.zeros(n_clusters, dtype=np.float64)
        n_all = 0
        for uid, s in sums.items():
            total += s
            n_all += counts[uid]
        global_prior = total / max(n_all, 1)

    rows: list[dict] = []
    for uid in sorted(sums.keys()):
        n = counts[uid]
        s = sums[uid]
        if alpha > 0:
            # Dirichlet shrinkage: (n * mean + alpha * prior) / (n + alpha)
            s = s + alpha * global_prior
            denom = float(n + alpha)
        else:
            denom = float(s.sum()) if s.sum() > 0 else 1.0
        w = s / denom
        row = {
            "user_id": uid,
            "n_posts": int(n),
            "assignment_type": assignment_type,
        }
        for k in range(n_clusters):
            row[f"w{k}"] = float(w[k])
        rows.append(row)
    return rows


def load_user_ground_truth(path: Path) -> dict[str, dict]:
    """Index users.jsonl by user_id for GT profile joins."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for row in load_jsonl(path):
        uid = row.get("user_id")
        if uid:
            out[uid] = row
    return out


def topic_to_cluster_map(bridge_info: dict[str, Any] | None) -> dict[str, int]:
    """Map GT topic id → cluster index from topic-bridge hard means."""
    if not bridge_info:
        return {}
    hard_means = bridge_info.get("topic_mean_hard") or {}
    mapping: dict[str, int] = {}
    for topic, vec in hard_means.items():
        mapping[topic] = int(np.argmax(np.asarray(vec, dtype=np.float64)))
    return mapping


def attach_user_ground_truth(
    rows: list[dict],
    users_by_id: dict[str, dict],
    topic_to_cluster: dict[str, int],
    n_clusters: int,
) -> list[dict]:
    """Join profile_id / GT distributions onto attention rows."""
    if not rows:
        return rows
    enriched: list[dict] = []
    missing = 0
    for row in rows:
        out = dict(row)
        gt = users_by_id.get(row["user_id"])
        if gt is None:
            missing += 1
            out["profile_id"] = None
            out["profile_label"] = None
            out["topic_distribution"] = None
            out["empirical_topic_distribution"] = None
        else:
            out["profile_id"] = gt.get("profile_id")
            out["profile_label"] = gt.get("profile_label")
            out["topic_distribution"] = gt.get("topic_distribution")
            out["empirical_topic_distribution"] = gt.get(
                "empirical_topic_distribution"
            )
            # Cluster-aligned GT mix for side-by-side comparison with w*.
            emp = gt.get("empirical_topic_distribution") or {}
            if topic_to_cluster and emp:
                gt_w = np.zeros(n_clusters, dtype=np.float64)
                for topic, p in emp.items():
                    if topic in topic_to_cluster:
                        gt_w[topic_to_cluster[topic]] += float(p)
                for k in range(n_clusters):
                    out[f"gt_w{k}"] = float(gt_w[k])
        enriched.append(out)
    if missing:
        print(f"Warning: {missing} attention users missing from users ground truth.")
    return enriched


def entropy(w: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(w, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------


def pairwise_centroid_cosine(centroids: np.ndarray) -> list[list[float]]:
    sim = centroids.astype(np.float64) @ centroids.astype(np.float64).T
    return [[float(sim[i, j]) for j in range(sim.shape[1])] for i in range(sim.shape[0])]


def nearest_posts_per_centroid(
    cos: np.ndarray,
    meta: Sequence[dict],
    top_n: int = 5,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for k in range(cos.shape[1]):
        order = np.argsort(-cos[:, k])[:top_n]
        items = []
        for i in order:
            row = meta[int(i)]
            items.append(
                {
                    "post_id": row.get("post_id"),
                    "cosine": float(cos[int(i), k]),
                    "topic": row.get("topic"),
                    "language": row.get("language"),
                    "text": (row.get("text") or "")[:160],
                }
            )
        out[str(k)] = items
    return out


def qc_user_weights(
    users: Sequence[dict],
    n_clusters: int,
    vertex_threshold: float,
) -> dict[str, Any]:
    if not users:
        return {"n_users": 0}
    W = np.array([[u[f"w{k}"] for k in range(n_clusters)] for u in users], dtype=np.float64)
    ents = np.array([entropy(w) for w in W])
    dom = W.max(axis=1)
    dominant = W.argmax(axis=1)
    near_vertex = float(np.mean(dom > vertex_threshold))
    frac_per_vertex = {
        str(k): float(np.mean((dom > vertex_threshold) & (dominant == k)))
        for k in range(n_clusters)
    }
    return {
        "n_users": len(users),
        "mean_n_posts": float(np.mean([u["n_posts"] for u in users])),
        "entropy": {
            "mean": float(ents.mean()),
            "std": float(ents.std()),
            "p10": float(np.percentile(ents, 10)),
            "p50": float(np.percentile(ents, 50)),
            "p90": float(np.percentile(ents, 90)),
            "max_entropy": float(math.log(n_clusters)),
        },
        "vertex_threshold": vertex_threshold,
        "fraction_near_vertex_max_w": near_vertex,
        "fraction_near_each_vertex": frac_per_vertex,
        "mean_weights": [float(x) for x in W.mean(axis=0)],
        "global_topic_mass": [float(x) for x in W.mean(axis=0)],
    }


def print_qc(qc: dict) -> None:
    print("\n=== QC summary ===")
    print(f"n_posts_clustered: {qc['n_posts_clustered']}")
    print(f"n_fit: {qc['n_fit']}")
    print(f"inertia: {qc['inertia']:.6f}")
    print(f"cluster_sizes: {qc['cluster_sizes']}")
    print("pairwise_centroid_cosine:")
    for row in qc["pairwise_centroid_cosine"]:
        print("  " + " ".join(f"{v:7.4f}" for v in row))
    if qc.get("topic_bridge"):
        print(f"topic_bridge: {json.dumps(qc['topic_bridge'], ensure_ascii=False)}")
    for key in ("user_attention_soft", "user_attention_hard"):
        block = qc.get(key)
        if not block:
            continue
        print(f"\n{key}:")
        print(f"  n_users={block['n_users']} mean_n_posts={block.get('mean_n_posts')}")
        print(f"  entropy mean/p50/p90={block['entropy']['mean']:.4f}/"
              f"{block['entropy']['p50']:.4f}/{block['entropy']['p90']:.4f} "
              f"(max={block['entropy']['max_entropy']:.4f})")
        print(
            f"  fraction max(w)>{block['vertex_threshold']}: "
            f"{block['fraction_near_vertex_max_w']:.4f}"
        )
        print(f"  fraction near each vertex: {block['fraction_near_each_vertex']}")
        print(f"  global topic mass: {block['global_topic_mass']}")
    if qc.get("nearest_posts_per_centroid"):
        print("\nnearest posts per centroid:")
        for k, items in qc["nearest_posts_per_centroid"].items():
            print(f"  cluster {k}:")
            for it in items[:3]:
                text = (it.get("text") or "").replace("\n", " ")
                print(
                    f"    {it.get('post_id')} cos={it['cosine']:.4f} "
                    f"topic={it.get('topic')} lang={it.get('language')} | {text[:80]}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_post_assignment_rows(
    meta: Sequence[dict] | None,
    hard: np.ndarray,
    soft: np.ndarray,
    n_clusters: int,
    source: str,
    user_ids: Sequence[str | None] | None = None,
    post_ids: Sequence[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    n = hard.shape[0]
    for i in range(n):
        m = meta[i] if meta is not None and i < len(meta) else {}
        uid = None
        if user_ids is not None:
            uid = user_ids[i]
        else:
            uid = m.get("user_id")
        pid = post_ids[i] if post_ids is not None else m.get("post_id", str(i))
        row: dict[str, Any] = {
            "post_id": pid,
            "user_id": uid,
            "hard_label": int(hard[i]),
            "source": source,
        }
        if m.get("topic") is not None:
            row["topic"] = m.get("topic")
        if m.get("language") is not None:
            row["language"] = m.get("language")
        for k in range(n_clusters):
            row[f"soft_w{k}"] = float(soft[i, k])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Override path to embeddings.npy (default: <emb-dir>/embeddings.npy).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="If set and < n_posts, fit KMeans on this many random posts, then predict all.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Softmax temperature over cosine similarities (smaller → harder).",
    )
    parser.add_argument(
        "--assignment-variant",
        choices=("soft", "hard"),
        default="soft",
        help="Stored soft_w* columns: temperature softmax, or hard one-hot.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Dirichlet shrinkage strength toward global prior (0 = off).",
    )
    parser.add_argument(
        "--user-posts",
        type=Path,
        default=DEFAULT_USER_POSTS,
        help="User posts JSONL with user_id (and topic for bridge mode). "
        "Pass an empty string to disable.",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=DEFAULT_USERS,
        help="Users JSONL with ground-truth profile_id / topic_distribution "
        "joined onto attention tables. Empty string to skip.",
    )
    parser.add_argument(
        "--user-embeddings",
        type=Path,
        default=None,
        help="Optional embeddings.npy aligned row-wise with --user-posts.",
    )
    parser.add_argument(
        "--bridge-language",
        default="en",
        help="Language filter when building topic→cluster bridge from library posts. "
        "Empty string = all languages.",
    )
    parser.add_argument(
        "--vertex-threshold",
        type=float,
        default=0.8,
        help="QC: fraction of users with max(w) above this threshold.",
    )
    parser.add_argument("--top-nearest", type=int, default=5)
    args = parser.parse_args()

    emb_path = args.embeddings or (args.emb_dir / "embeddings.npy")
    if not emb_path.exists():
        raise SystemExit(f"Embeddings not found: {emb_path}")

    embeddings_raw = np.load(emb_path)
    embeddings, was_normalized = maybe_normalize(embeddings_raw)
    meta = load_embedding_meta(args.emb_dir)
    if len(meta) != embeddings.shape[0]:
        raise SystemExit(
            f"Row mismatch: {embeddings.shape[0]} embeddings vs {len(meta)} meta rows"
        )

    print(
        f"Loaded embeddings {embeddings.shape} from {emb_path} "
        f"(already_normalized={was_normalized})"
    )
    print(
        f"Fitting KMeans n_clusters={args.n_clusters} n_init={args.n_init} "
        f"random_state={args.random_state} subsample={args.subsample}"
    )

    model, centroids, n_fit = fit_kmeans(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
        subsample=args.subsample,
    )
    hard, soft_temp, cos = assign_posts(embeddings, centroids, tau=args.tau)
    hard_oh = hard_one_hot(hard, args.n_clusters)
    soft_stored = soft_temp if args.assignment_variant == "soft" else hard_oh

    # Inertia on all points under final (normalized) centroids.
    # sklearn inertia_ is on the fit subset with its centers; recompute for clarity.
    inertia = float(((embeddings.astype(np.float64) - centroids[hard]) ** 2).sum())

    lib_rows = build_post_assignment_rows(
        meta, hard, soft_stored, args.n_clusters, source="library"
    )

    user_posts_path: Path | None
    if str(args.user_posts) in ("", "None", "none"):
        user_posts_path = None
    else:
        user_posts_path = args.user_posts

    post_rows = list(lib_rows)
    user_soft_rows: list[dict] = []
    user_hard_rows: list[dict] = []
    bridge_info: dict[str, Any] | None = None
    mode = "library_only"

    has_user_id_in_meta = any(r.get("user_id") for r in meta)
    if has_user_id_in_meta:
        mode = "aligned_meta"
        uids = [r.get("user_id") for r in meta]
        # Drop rows without user_id from user aggregation.
        mask = [u is not None for u in uids]
        if any(mask):
            idx = np.array(mask)
            user_soft_rows = aggregate_user_weights(
                [u for u, m in zip(uids, mask) if m],
                soft_temp[idx],
                args.n_clusters,
                "soft",
                args.alpha,
            )
            user_hard_rows = aggregate_user_weights(
                [u for u, m in zip(uids, mask) if m],
                hard_oh[idx],
                args.n_clusters,
                "hard",
                args.alpha,
            )

    if user_posts_path is not None and user_posts_path.exists():
        user_posts = load_jsonl(user_posts_path)
        if not user_posts:
            raise SystemExit(f"No records in {user_posts_path}")
        if not all("user_id" in r for r in user_posts):
            raise SystemExit("user posts must include user_id")

        if args.user_embeddings is not None:
            mode = "user_embeddings"
            uemb_raw = np.load(args.user_embeddings)
            uemb, _ = maybe_normalize(uemb_raw)
            if uemb.shape[0] != len(user_posts):
                raise SystemExit(
                    f"User embedding rows {uemb.shape[0]} != posts {len(user_posts)}"
                )
            if uemb.shape[1] != embeddings.shape[1]:
                raise SystemExit(
                    f"User embedding dim {uemb.shape[1]} != library dim {embeddings.shape[1]}"
                )
            u_hard, u_soft, _ = assign_posts(uemb, centroids, tau=args.tau)
            u_hard_oh = hard_one_hot(u_hard, args.n_clusters)
            u_stored = u_soft if args.assignment_variant == "soft" else u_hard_oh
            user_assign_rows = build_post_assignment_rows(
                user_posts,
                u_hard,
                u_stored,
                args.n_clusters,
                source="user_embeddings",
            )
            post_rows.extend(user_assign_rows)
            uids = [r["user_id"] for r in user_posts]
            user_soft_rows = aggregate_user_weights(
                uids, u_soft, args.n_clusters, "soft", args.alpha
            )
            user_hard_rows = aggregate_user_weights(
                uids, u_hard_oh, args.n_clusters, "hard", args.alpha
            )
        elif not has_user_id_in_meta:
            mode = "topic_bridge"
            lang = args.bridge_language or None
            soft_map, hard_map, topic_counts = topic_weight_lookup(
                meta, soft_temp, hard_oh, language_filter=lang
            )
            bridge_info = {
                "language_filter": lang,
                "topic_post_counts": topic_counts,
                "topic_mean_soft": {t: v.tolist() for t, v in soft_map.items()},
                "topic_mean_hard": {t: v.tolist() for t, v in hard_map.items()},
            }
            missing = sorted({r["topic"] for r in user_posts if r["topic"] not in soft_map})
            if missing:
                raise SystemExit(f"User posts reference topics missing from bridge: {missing}")

            u_soft = np.stack([soft_map[r["topic"]] for r in user_posts])
            u_hard_w = np.stack([hard_map[r["topic"]] for r in user_posts])
            # Hard label = argmax of bridged hard-mean (or soft); keep discrete label too.
            u_hard = np.argmax(u_hard_w, axis=1).astype(np.int32)
            u_stored = u_soft if args.assignment_variant == "soft" else hard_one_hot(
                u_hard, args.n_clusters
            )
            user_assign_rows = build_post_assignment_rows(
                user_posts,
                u_hard,
                u_stored,
                args.n_clusters,
                source="topic_bridge",
            )
            post_rows.extend(user_assign_rows)
            uids = [r["user_id"] for r in user_posts]
            user_soft_rows = aggregate_user_weights(
                uids, u_soft, args.n_clusters, "soft", args.alpha
            )
            user_hard_rows = aggregate_user_weights(
                uids, u_hard_w, args.n_clusters, "hard", args.alpha
            )
            print(
                f"Topic bridge from library posts "
                f"(language={lang!r}): {topic_counts}"
            )
    elif user_posts_path is not None and not user_posts_path.exists():
        print(f"Warning: user posts not found ({user_posts_path}); skipping user attention.")

    # Cluster sizes on library embeddings
    cluster_sizes = {str(k): int((hard == k).sum()) for k in range(args.n_clusters)}

    qc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "n_posts_clustered": int(embeddings.shape[0]),
        "n_fit": n_fit,
        "embedding_dim": int(embeddings.shape[1]),
        "n_clusters": args.n_clusters,
        "inertia": inertia,
        "sklearn_inertia_fit_subset": float(model.inertia_),
        "cluster_sizes": cluster_sizes,
        "pairwise_centroid_cosine": pairwise_centroid_cosine(centroids),
        "already_normalized_on_disk": was_normalized,
        "tau": args.tau,
        "alpha": args.alpha,
        "assignment_variant": args.assignment_variant,
        "vertex_threshold": args.vertex_threshold,
        "topic_bridge": bridge_info,
        "nearest_posts_per_centroid": nearest_posts_per_centroid(
            cos, meta, top_n=args.top_nearest
        ),
        "user_attention_soft": qc_user_weights(
            user_soft_rows, args.n_clusters, args.vertex_threshold
        )
        if user_soft_rows
        else None,
        "user_attention_hard": qc_user_weights(
            user_hard_rows, args.n_clusters, args.vertex_threshold
        )
        if user_hard_rows
        else None,
    }

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "centroids.npy", centroids.astype(np.float32))

    train_config = {
        "created_at": qc["created_at"],
        "emb_dir": str(args.emb_dir),
        "embeddings": str(emb_path),
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "n_init": args.n_init,
        "subsample": args.subsample,
        "n_fit": n_fit,
        "n_posts": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "tau": args.tau,
        "alpha": args.alpha,
        "assignment_variant": args.assignment_variant,
        "bridge_language": args.bridge_language,
        "user_posts": str(user_posts_path) if user_posts_path else None,
        "user_embeddings": str(args.user_embeddings) if args.user_embeddings else None,
        "mode": mode,
        "algorithm": "sklearn.cluster.KMeans on L2-normalized vectors (spherical)",
        "seed_notes": "numpy default_rng(random_state) for subsample; KMeans random_state set.",
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2) + "\n", encoding="utf-8"
    )

    soft_cols = [f"soft_w{k}" for k in range(args.n_clusters)]
    post_fields = ["post_id", "user_id", "hard_label", *soft_cols, "topic", "language", "source"]
    write_jsonl(out_dir / "post_assignments.jsonl", post_rows)
    write_csv(out_dir / "post_assignments.csv", post_rows, post_fields)

    users_path: Path | None
    if str(args.users) in ("", "None", "none"):
        users_path = None
    else:
        users_path = args.users
    users_by_id = load_user_ground_truth(users_path) if users_path else {}
    t2c = topic_to_cluster_map(bridge_info)
    if users_by_id:
        user_soft_rows = attach_user_ground_truth(
            user_soft_rows, users_by_id, t2c, args.n_clusters
        )
        user_hard_rows = attach_user_ground_truth(
            user_hard_rows, users_by_id, t2c, args.n_clusters
        )
        print(f"Joined ground-truth profiles from {users_path} ({len(users_by_id)} users)")

    w_cols = [f"w{k}" for k in range(args.n_clusters)]
    gt_w_cols = [f"gt_w{k}" for k in range(args.n_clusters)]
    user_fields = [
        "user_id",
        "profile_id",
        "profile_label",
        "n_posts",
        *w_cols,
        *gt_w_cols,
        "assignment_type",
        "topic_distribution",
        "empirical_topic_distribution",
    ]

    def _csv_ready(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            row = dict(r)
            for key in ("topic_distribution", "empirical_topic_distribution"):
                if isinstance(row.get(key), dict):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            out.append(row)
        return out

    write_jsonl(out_dir / "user_attention_soft.jsonl", user_soft_rows)
    write_csv(out_dir / "user_attention_soft.csv", _csv_ready(user_soft_rows), user_fields)
    write_jsonl(out_dir / "user_attention_hard.jsonl", user_hard_rows)
    write_csv(out_dir / "user_attention_hard.csv", _csv_ready(user_hard_rows), user_fields)

    (out_dir / "qc_summary.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nWrote artifacts → {out_dir}")
    print_qc(qc)


if __name__ == "__main__":
    main()
