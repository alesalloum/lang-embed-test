# Claim–stance embeddings (`stance_instruct`)

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **Mode:** `stance_instruct`
- **Records:** 1500
- **Dim:** 1024
- **Normalized:** yes (L2)
- **Prompt:** `Instruct: Identify the attitudinal stance of the social media post toward a claim about increasing AI datacenters as supportive, critical, or neutral; encode that stance as the primary semantic signal while preserving topic content.
Query:`

| File | Description |
| --- | --- |
| `embeddings.npy` | `(n, d)` float32 matrix |
| `metadata.jsonl` | Row metadata + text |
| `ids.json` | Compact ids for plotting |
| `info.json` | Run metadata |
