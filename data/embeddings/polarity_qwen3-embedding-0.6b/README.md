# Phenomenon-polarity embeddings (Qwen3-Embedding-0.6B)

Two embedding runs over `data/polarity_posts/posts.jsonl`:

| Subdir | Mode |
| --- | --- |
| `vanilla/` | No instruction (document / passage mode) |
| `polarity_instruct/` | Instructed to encode pro / against / neutral polarity |

Regenerate:

```bash
python scripts/embed_polarity.py
```

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": "data/polarity_posts/posts.jsonl",
  "modes": [
    "vanilla",
    "polarity_instruct"
  ],
  "n_records": 900,
  "out_root": "data/embeddings/polarity_qwen3-embedding-0.6b",
  "polarity_task": "Identify the polarity of the social media post toward increasing AI datacenters as pro (supportive of expansion), against (opposed to expansion), or neutral; encode that polarity as the primary semantic signal while preserving topical content."
}
```
