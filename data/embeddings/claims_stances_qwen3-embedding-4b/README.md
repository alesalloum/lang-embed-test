# Claim–stance embeddings (Qwen3-Embedding-4B)

Two embedding runs over `data/claims_stances/claims.jsonl`:

| Subdir | Mode |
| --- | --- |
| `vanilla/` | No instruction (document / passage mode) |
| `stance_instruct/` | Instructed to encode supportive / critical / neutral stance |

Regenerate:

```bash
python scripts/embed_claim_stances.py --model Qwen/Qwen3-Embedding-4B --dtype float16
```

```json
{
  "model": "Qwen/Qwen3-Embedding-4B",
  "dtype": "float16",
  "input": "data/claims_stances/claims.jsonl",
  "modes": [
    "vanilla",
    "stance_instruct"
  ],
  "n_records": 1500,
  "out_root": "data/embeddings/claims_stances_qwen3-embedding-4b",
  "stance_task": "Identify the attitudinal stance of the social media post toward a claim about increasing AI datacenters as supportive, critical, or neutral; encode that stance as the primary semantic signal while preserving topic content."
}
```
