# lang-embed-test

Toy data and experiments for multilingual embedding models: do same-meaning posts cluster by **topic**, or do they split by **language** under k-means?

## Data

See [`data/`](data/README.md):

- 3 topics × 50 posts × 4 languages (en, es, ar, zh) = **600** texts
- Topics: AI coding innovations, AI copyright theft, AI mass surveillance
- Parallel translations share a `post_id`
- **1000 English users**, each with one post per topic (**3000** user posts)

Regenerate multilingual posts with:

```bash
pip install deep-translator
python scripts/generate_toy_posts.py
```

Regenerate English users + their posts with:

```bash
python scripts/generate_english_users.py
```

## Embeddings

Precomputed document embeddings for all 600 texts live under
[`data/embeddings/qwen3-embedding-0.6b/`](data/embeddings/qwen3-embedding-0.6b/)
(`Qwen/Qwen3-Embedding-0.6B`, L2-normalized, no query prompt).

Regenerate with:

```bash
pip install -r requirements.txt
python scripts/embed_posts.py
```

## UMAP plot

2D UMAP scatter (topic = marker shape, language = color) is written to
[`results/`](results/):

```bash
python scripts/plot_umap.py
```
