# Phenomenon-polarity UMAP plots

Qwen3-Embedding-0.6B projections of the AI-datacenter polarity set.

## Encoding

- **Color** = ground-truth `polarity` (pro / against / neutral)
- **Marker shape** = ground-truth `aspect`

## Modes

| Dir | Embedding |
| --- | --- |
| `vanilla/` | No instruction |
| `polarity_instruct/` | Instructed to encode polarity |
| `umap_compare_vanilla_vs_instruct.png` | Side-by-side |

Metrics in `umap_info.json` are computed in the original
embedding space (not UMAP).

```bash
python scripts/embed_polarity.py
python scripts/plot_polarity_umap.py
```

```json
{
  "embedding_root": "data/embeddings/polarity_qwen3-embedding-0.6b",
  "modes": [
    "vanilla",
    "polarity_instruct"
  ],
  "n_neighbors": 15,
  "min_dist": 0.1,
  "random_state": 42,
  "encoding": {
    "polarity_colors": {
      "pro": "#2ca02c",
      "against": "#d62728",
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
  "polarity_separation": {
    "vanilla": {
      "polarities": [
        "against",
        "neutral",
        "pro"
      ],
      "silhouette_cosine": 0.07226518541574478,
      "nearest_centroid_accuracy": 0.9911111111111112,
      "centroid_cosine": {
        "against__neutral": 0.9365310668945312,
        "against__pro": 0.9184142351150513,
        "neutral__pro": 0.9002298712730408
      }
    },
    "polarity_instruct": {
      "polarities": [
        "against",
        "neutral",
        "pro"
      ],
      "silhouette_cosine": 0.5821539163589478,
      "nearest_centroid_accuracy": 0.9988888888888889,
      "centroid_cosine": {
        "against__neutral": 0.8047730922698975,
        "against__pro": 0.7703126072883606,
        "neutral__pro": 0.7688924074172974
      }
    }
  },
  "aspect_separation": {
    "vanilla": {
      "aspects": [
        "economic",
        "environmental",
        "geopolitical",
        "infrastructure",
        "local_community",
        "technological"
      ],
      "silhouette_cosine": -0.004358428996056318,
      "nearest_centroid_accuracy": 0.6188888888888889
    },
    "polarity_instruct": {
      "aspects": [
        "economic",
        "environmental",
        "geopolitical",
        "infrastructure",
        "local_community",
        "technological"
      ],
      "silhouette_cosine": -0.012404077686369419,
      "nearest_centroid_accuracy": 0.4066666666666667
    }
  },
  "files": {
    "umap_compare_vanilla_vs_instruct.png": "Side-by-side comparison",
    "vanilla/": "Per-mode UMAP + metrics",
    "polarity_instruct/": "Per-mode UMAP + metrics"
  }
}
```
