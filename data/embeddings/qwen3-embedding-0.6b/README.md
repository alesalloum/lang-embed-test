# Embeddings: `Qwen/Qwen3-Embedding-0.6B`

- **Records:** 600
- **Dim:** 1024
- **Normalized:** yes (L2)
- **Prompt:** none (document / passage mode)

| File | Description |
| --- | --- |
| `embeddings.npy` | `(n, d)` float32 matrix |
| `metadata.jsonl` | Row metadata + text |
| `ids.json` | Compact ids for clustering |
| `info.json` | Run metadata |

Load:

```python
import json
import numpy as np

emb = np.load("data/embeddings/qwen3-embedding-0.6b/embeddings.npy")
ids = json.loads(Path("data/embeddings/qwen3-embedding-0.6b/ids.json").read_text())
```
