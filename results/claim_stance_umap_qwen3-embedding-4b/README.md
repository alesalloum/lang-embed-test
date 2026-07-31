# Claim–stance UMAP plots (Qwen3-Embedding-4B)

Qwen3-Embedding-4B projections of the AI-datacenter claim–stance set.

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
python scripts/embed_claim_stances.py --model <model-id>
python scripts/plot_claim_stance_umap.py --emb-root data/embeddings/claims_stances_qwen3-embedding-4b
```

```json
{
  "model_label": "Qwen3-Embedding-4B",
  "embedding_root": "data/embeddings/claims_stances_qwen3-embedding-4b",
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
      "silhouette_cosine": 0.022412169724702835,
      "nearest_centroid_accuracy": 0.8926666666666667,
      "centroid_cosine": {
        "critical__neutral": 0.9607948064804077,
        "critical__supportive": 0.9448055028915405,
        "neutral__supportive": 0.9604377746582031
      }
    },
    "stance_instruct": {
      "stances": [
        "critical",
        "neutral",
        "supportive"
      ],
      "silhouette_cosine": 0.6335034370422363,
      "nearest_centroid_accuracy": 0.996,
      "centroid_cosine": {
        "critical__neutral": 0.8181226253509521,
        "critical__supportive": 0.8127896785736084,
        "neutral__supportive": 0.8406574130058289
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
