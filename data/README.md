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

Labeled English users for **recovering interest profiles from posts**.

Each user has a predefined `profile_id` and ground-truth `topic_distribution`. Their posts are sampled from that distribution (100 posts / user).

Regenerate with:

```bash
python3 scripts/generate_english_users.py
```

### Scale

- **1000** users (`language=en`)
- **100** posts per user → **100000** posts
- **7** interest profiles (round-robin assignment)

### Profiles (ground truth)

| profile_id | label | P(coding / copyright / surveillance) |
| --- | --- | --- |
| `coding_heavy` | Mostly AI coding innovations | 0.80 / 0.10 / 0.10 |
| `copyright_heavy` | Mostly AI copyright / theft concerns | 0.10 / 0.80 / 0.10 |
| `surveillance_heavy` | Mostly AI mass surveillance | 0.10 / 0.10 / 0.80 |
| `balanced` | Even across all topics | 0.33 / 0.33 / 0.33 |
| `coding_copyright` | Split: coding + copyright | 0.45 / 0.45 / 0.10 |
| `coding_surveillance` | Split: coding + surveillance | 0.45 / 0.10 / 0.45 |
| `copyright_surveillance` | Split: copyright + surveillance | 0.10 / 0.45 / 0.45 |

### Files

| File | Format |
| --- | --- |
| `user_profiles.json` | Profile catalog + distributions |
| `users.json` / `users.jsonl` / `users.csv` | Users with profile + GT dist + empirical counts |
| `user_posts.jsonl` / `user_posts.csv` | Flat posts (`user_id`, `profile_id`, `topic`, `text`) |
| `user_posts.json` | Aggregate metadata only (not nested texts) |

### User schema (key fields)

- `profile_id` / `profile_label` — discrete interest type
- `topic_distribution` — generative ground truth over topics
- `topic_counts` / `empirical_topic_distribution` — realized post mix

### Intended use

Embed or model each user's posts, estimate a topic mixture, and compare to `topic_distribution` / `profile_id` (heavy vs balanced vs split).

## AI datacenter claim–stance set

Claim–stance posts for **stance embedding** tests (supportive / neutral / critical)
on a single topic: increasing AI datacenters.

See [`claims_stances/`](claims_stances/) (`claims.json` / `claims.jsonl` / `claims.csv`).

Regenerate with:

```bash
python3 scripts/generate_ai_datacenter_claims.py
```

- **500** claims × **3** stances = **1500** English posts
- Aspects: economic, environmental, infrastructure, geopolitical, local community, technological
- Flat ground-truth label for stance separation: `stance`; shared meaning key: `claim_id`

### Embeddings + UMAP

Precomputed under [`embeddings/claims_stances_qwen3-embedding-0.6b/`](embeddings/claims_stances_qwen3-embedding-0.6b/):

| Mode | Path | Prompt |
| --- | --- | --- |
| vanilla | `vanilla/` | none (document mode) |
| stance_instruct | `stance_instruct/` | Instruct to encode supportive/critical/neutral |

```bash
python scripts/embed_claim_stances.py
python scripts/plot_claim_stance_umap.py
```

UMAP plots (color = stance, shape = aspect) are in
[`../results/claim_stance_umap/`](../results/claim_stance_umap/).

