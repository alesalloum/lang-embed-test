# Polarity embeddings (`polarity_instruct`)

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **Mode:** `polarity_instruct`
- **Records:** 900
- **Dim:** 1024
- **Normalized:** yes (L2)
- **Prompt:** `Instruct: Identify the polarity of the social media post toward increasing AI datacenters as pro (supportive of expansion), against (opposed to expansion), or neutral; encode that polarity as the primary semantic signal while preserving topical content.
Query:`

| File | Description |
| --- | --- |
| `embeddings.npy` | `(n, d)` float32 matrix |
| `metadata.jsonl` | Row metadata + text |
| `ids.json` | Compact ids for plotting |
| `info.json` | Run metadata |
