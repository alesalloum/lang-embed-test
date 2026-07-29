# Multilingual toy posts

Artificial social-media posts for studying multilingual embeddings and k-means.

## Topics

- `ai_coding_innovations`: AI innovations in coding
- `ai_copyright_theft`: AI theft of artistic and authors' copyrights
- `ai_mass_surveillance`: AI-enabled mass surveillance

## Languages

- `en`: English
- `ar`: Arabic
- `es`: Spanish
- `zh`: Chinese (Simplified)

## Scale

- 3 topics × 50 posts × 4 languages = **600** text records
- 150 unique meanings (`post_id`), each with parallel translations

## Files

| File | Format |
| --- | --- |
| `posts.json` | Nested: each post has `texts.{en,es,ar,zh}` |
| `posts.jsonl` | Flat: one JSON object per language version |
| `posts.csv` | Flat CSV with the same columns |

## How it was built

English posts were written as imaginary social-media style text. Spanish, Arabic, and Chinese versions were produced with machine translation (`deep-translator` / Google Translate) via `scripts/generate_toy_posts.py`, with a few manual fixes for obvious MT failures.

## Embeddings

Saved under [`embeddings/qwen3-embedding-0.6b/`](embeddings/qwen3-embedding-0.6b/):

- Model: `Qwen/Qwen3-Embedding-0.6B`
- All 600 flat records from `posts.jsonl`
- L2-normalized document embeddings (no query prompt)
- `embeddings.npy` row `i` aligns with `metadata.jsonl` / `ids.json` index `i`

```bash
python scripts/embed_posts.py
```

## Intended use

Embed all 600 texts, run k-means with k=3, and compare clusters to `topic` (desired) versus `language` (undesired language silos).

```json
{
  "files": {
    "posts.json": "Nested posts with en/es/ar/zh texts per post_id",
    "posts.jsonl": "Flat records (one line per language version)",
    "posts.csv": "Same flat schema as JSONL"
  },
  "topics": [
    "ai_coding_innovations",
    "ai_copyright_theft",
    "ai_mass_surveillance"
  ],
  "languages": [
    "en",
    "ar",
    "es",
    "zh"
  ],
  "counts": {
    "posts_per_topic": 50,
    "topics": 3,
    "languages": 4,
    "unique_meanings": 150,
    "total_text_records": 600
  },
  "clustering_notes": "Ground-truth cluster label for semantic clustering is `topic`. Each `post_id` appears in 4 languages with the same meaning. A good multilingual embedder + k-means (k=3) should group records by topic, not by language."
}
```

## English users (synthetic)

1000 English-only synthetic users, each with one post per topic (3 topics → 3000 posts). Language fields are present so non-English users can be added later.

Regenerate with:

```bash
python scripts/generate_english_users.py
```

### Scale

- **1000** users (`language=en`)
- **3000** posts (1 per topic per user)
- Topics: same three as the multilingual toy set

### Files

| File | Format |
| --- | --- |
| `users.json` | Nested user list + metadata |
| `users.jsonl` | One user per line |
| `users.csv` | Flat users |
| `user_posts.json` | Posts nested by `user_id` |
| `user_posts.jsonl` | Flat posts with `user_id` |
| `user_posts.csv` | Flat CSV |

### User schema

`user_id`, `username`, `display_name`, `language`, `language_name`

### Post schema

`post_id`, `user_id`, `topic`, `topic_label`, `language`, `language_name`, `text`
