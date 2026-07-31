# NMF latent-profile rank selection

Fitted `sklearn.decomposition.NMF` (MU + KL, `init=nndsvda`, `max_iter=1000`,
`random_state=42`) on user attention
[`/workspace/results/topic_attention_k20/user_attention_soft.csv`](results/topic_attention_k20/user_attention_soft.csv)
shape **(1000 × 20)**.

## Recommended rank

**k = 3** (min(knee_distance, gain_flatten))

| Metric | Value |
| --- | --- |
| KL loss | 0.007947 |
| Deviance explained | 99.99% |
| Perplexity | 16.598720 |

## Artifacts

| File | Description |
| --- | --- |
| `nmf_elbow_curves.png` / `.pdf` | Side-by-side deviance-explained & perplexity vs k |
| `nmf_metrics.json` | Full metric sweep + elbow metadata |
| `nmf_metrics_table.csv` | Compact `[k, KL_Loss, Deviance_Explained_%, Perplexity]` |
| `W_norm_k3.npy` | User×component weights (columns scaled) |
| `H_norm_k3.npy` / `.csv` | Latent profiles over microtopics (rows sum to 1) |

## Reproduce

```bash
python scripts/cluster_user_attention.py --n-clusters 20 --out-dir results/topic_attention_k20
python scripts/nmf_latent_profiles.py \
  --attention-csv results/topic_attention_k20/user_attention_soft.csv \
  --out-dir results/nmf_latent_profiles --k-max 10
```
