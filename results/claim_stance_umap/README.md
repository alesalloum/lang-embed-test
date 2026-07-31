# Claim–stance UMAP plots

Qwen3-Embedding-0.6B projections of the AI-datacenter claim–stance set.

## Encoding

- **Color** = ground-truth `stance` (supportive / critical / neutral)
- **Marker shape** = ground-truth `aspect`

## Modes

| Dir | Embedding |
| --- | --- |
| `vanilla/` | No instruction |
| `stance_instruct/` | Instructed to encode stance |
| `umap_compare_vanilla_vs_instruct.png` | Side-by-side |

Each mode folder also stores `umap_info.json` with silhouette /
nearest-centroid stance-separability metrics in the original
embedding space (not UMAP).

```bash
python scripts/embed_claim_stances.py
python scripts/plot_claim_stance_umap.py
```

```json
{
  "embedding_root": "data/embeddings/claims_stances_qwen3-embedding-0.6b",
  "modes": [
    "vanilla",
    "stance_instruct"
  ],
  "n_neighbors": 15,
  "min_dist": 0.1,
  "random_state": 42,
  "encoding": {
    "stance_colors": {
      "supportive": "#2ca02c",
      "critical": "#d62728",
      "neutral": "#7f7f7f"
    },
    "aspect_markers": {
      "economic": "o",
      "environmental": "s",
      "infrastructure": "^",
      "geopolitical": "D",
      "local_community": "v",
      "technological": "P"
    }
  },
  "stance_separation": {
    "vanilla": {
      "stances": [
        "critical",
        "neutral",
        "supportive"
      ],
      "silhouette_cosine": 0.015408582054078579,
      "nearest_centroid_accuracy": 0.9086666666666666,
      "centroid_cosine": {
        "critical__neutral": 0.9659225940704346,
        "critical__supportive": 0.9625169634819031,
        "neutral__supportive": 0.9670848250389099
      }
    },
    "stance_instruct": {
      "stances": [
        "critical",
        "neutral",
        "supportive"
      ],
      "silhouette_cosine": 0.27057015895843506,
      "nearest_centroid_accuracy": 0.978,
      "centroid_cosine": {
        "critical__neutral": 0.8889997005462646,
        "critical__supportive": 0.8841314315795898,
        "neutral__supportive": 0.9035146236419678
      }
    }
  },
  "files": {
    "umap_compare_vanilla_vs_instruct.png": "Side-by-side comparison",
    "vanilla/": "Per-mode UMAP + metrics",
    "stance_instruct/": "Per-mode UMAP + metrics"
  }
}
```
