# lang-embed-test

Toy data and experiments for multilingual embedding models: do same-meaning posts cluster by **topic**, or do they split by **language** under k-means?

## Data

See [`data/`](data/README.md):

- 3 topics × 50 posts × 4 languages (en, es, ar, zh) = **600** texts
- Topics: AI coding innovations, AI copyright theft, AI mass surveillance
- Parallel translations share a `post_id`
- **1000 English users** with predefined interest **profiles** (ground-truth
  topic distributions) and **100 posts each** (**100_000** user posts)
- **AI datacenter claim–stance set**: 500 claims × 3 stances (supportive/neutral/critical) under [`data/claims_stances/`](data/claims_stances/)

Regenerate multilingual posts with:

```bash
pip install deep-translator
python scripts/generate_toy_posts.py
```

Regenerate English users + their posts with:

```bash
python3 scripts/generate_english_users.py
```

Regenerate AI datacenter claim–stance posts with:

```bash
python3 scripts/generate_ai_datacenter_claims.py
```

User profiles (`coding_heavy`, `copyright_heavy`, `surveillance_heavy`,
`balanced`, and dual-interest splits) are defined in
`scripts/generate_english_users.py` / `data/user_profiles.json`. Each user
stores `topic_distribution` (generative GT) and realized `topic_counts`.

## Embeddings

Precomputed document embeddings for all 600 texts live under
[`data/embeddings/qwen3-embedding-0.6b/`](data/embeddings/qwen3-embedding-0.6b/)
(`Qwen/Qwen3-Embedding-0.6B`, L2-normalized, no query prompt).

Regenerate with:

```bash
pip install -r requirements.txt
python scripts/embed_posts.py
```

Claim–stance posts are embedded under
[`data/embeddings/claims_stances_qwen3-embedding-0.6b/`](data/embeddings/claims_stances_qwen3-embedding-0.6b/)
and
[`data/embeddings/claims_stances_qwen3-embedding-4b/`](data/embeddings/claims_stances_qwen3-embedding-4b/):

- `vanilla/` — no instruction
- `stance_instruct/` — Instruct/Query prompt asking the model to encode
  supportive / critical / neutral stance

```bash
python scripts/embed_claim_stances.py
python scripts/embed_claim_stances.py --model Qwen/Qwen3-Embedding-4B --dtype float16 --batch-size 4
```

## UMAP plot

2D UMAP scatter (topic = marker shape, language = color) is written to
[`results/`](results/):

```bash
python scripts/plot_umap.py
```

Claim–stance UMAP (stance = color, aspect = marker shape; vanilla vs
stance-instruct) lands in
[`results/claim_stance_umap/`](results/claim_stance_umap/) (0.6B) and
[`results/claim_stance_umap_qwen3-embedding-4b/`](results/claim_stance_umap_qwen3-embedding-4b/)
(4B). Cross-size comparison:

```bash
python scripts/plot_claim_stance_umap.py
python scripts/plot_claim_stance_umap.py --emb-root data/embeddings/claims_stances_qwen3-embedding-4b
python scripts/plot_claim_stance_model_compare.py
```

## Topic clustering + user attention (K=3)

Fit spherical KMeans (`K=3`) on the on-disk Qwen post embeddings, assign hard
labels + temperature-scaled soft weights, then build per-user attention
histograms (L1-normalized). Optional Dirichlet shrinkage via `--alpha`.

```bash
pip install -r requirements.txt
python scripts/cluster_user_attention.py
python scripts/cluster_user_attention.py --n-clusters 3 --tau 0.1 --n-init 10 --alpha 0
python scripts/plot_user_attention.py
```

Artifacts land in [`results/topic_attention_k3/`](results/topic_attention_k3/).

UMAP of the **600 library posts** colored by hard label vs soft weights
(reuses cached `results/umap_coords.npy` when present):

```bash
python scripts/plot_cluster_umap.py
```

### Output schema

| File | Schema |
| --- | --- |
| `centroids.npy` | `(K, d)` float32 L2-normalized centers |
| `train_config.json` | seed, `n_clusters`, `n_init`, `tau`, `alpha`, paths, mode |
| `post_assignments.jsonl` / `.csv` | `post_id`, `user_id`, `hard_label`, `soft_w0..w{K-1}` (+ `topic`/`language`/`source` when available) |
| `user_attention_soft.jsonl` / `.csv` | `user_id`, `profile_id`, `profile_label`, `n_posts`, `w0..w{K-1}`, `gt_w0..gt_w{K-1}`, `assignment_type=soft`, GT topic dists |
| `user_attention_hard.jsonl` / `.csv` | same with `assignment_type=hard` |
| `qc_summary.json` | cluster sizes, inertia, centroid cosines, user entropy / vertex fractions, nearest posts |

**Mode note:** library embeddings (`data/embeddings/...`) have no `user_id`.
With default `--user-posts data/user_posts.jsonl` and no `--user-embeddings`,
user posts inherit mean cluster weights of English library posts that share
the same ground-truth `topic` (topic bridge). Pass `--user-embeddings` aligned
with `--user-posts` for direct embedding assignment when those vectors exist.
