#!/usr/bin/env python3
"""Embed all toy posts with Qwen/Qwen3-Embedding-0.6B and save vectors + metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "posts.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "embeddings" / "qwen3-embedding-0.6b"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No records found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SentenceTransformer (default: auto).",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    texts = [r["text"] for r in records]
    print(f"Loaded {len(records)} texts from {args.input}")
    print(f"Loading model {args.model!r} …")

    model_kwargs: dict = {}
    load_kwargs: dict = {
        "model_kwargs": model_kwargs,
        "tokenizer_kwargs": {"padding_side": "left"},
    }
    if args.device:
        load_kwargs["device"] = args.device

    model = SentenceTransformer(args.model, **load_kwargs)

    # Posts are documents (not retrieval queries), so encode without the query prompt.
    print(f"Encoding with batch_size={args.batch_size} …")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise SystemExit(
            f"Unexpected embedding shape {embeddings.shape}, expected ({len(records)}, d)"
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "embeddings.npy"
    np.save(npy_path, embeddings)

    meta_rows = []
    for i, rec in enumerate(records):
        meta_rows.append(
            {
                "index": i,
                "post_id": rec["post_id"],
                "topic": rec["topic"],
                "topic_label": rec["topic_label"],
                "language": rec["language"],
                "language_name": rec["language_name"],
                "text": rec["text"],
            }
        )

    meta_jsonl = out_dir / "metadata.jsonl"
    with meta_jsonl.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compact sidecar without full text (handy for clustering notebooks).
    ids_path = out_dir / "ids.json"
    ids_path.write_text(
        json.dumps(
            [
                {
                    "index": r["index"],
                    "post_id": r["post_id"],
                    "topic": r["topic"],
                    "language": r["language"],
                }
                for r in meta_rows
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    info = {
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.relative_to(ROOT)),
        "n_records": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "prompt": None,
        "notes": (
            "Document embeddings (no query prompt). L2-normalized. "
            "Row i in embeddings.npy matches metadata.jsonl / ids.json index i."
        ),
        "files": {
            "embeddings.npy": f"shape ({embeddings.shape[0]}, {embeddings.shape[1]}) float32",
            "metadata.jsonl": "One JSON object per row, includes full text",
            "ids.json": "Compact index/post_id/topic/language for each row",
        },
        "topics": sorted({r["topic"] for r in records}),
        "languages": sorted({r["language"] for r in records}),
    }
    info_path = out_dir / "info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Embeddings: `{args.model}`",
                "",
                f"- **Records:** {len(records)}",
                f"- **Dim:** {embeddings.shape[1]}",
                "- **Normalized:** yes (L2)",
                "- **Prompt:** none (document / passage mode)",
                "",
                "| File | Description |",
                "| --- | --- |",
                "| `embeddings.npy` | `(n, d)` float32 matrix |",
                "| `metadata.jsonl` | Row metadata + text |",
                "| `ids.json` | Compact ids for clustering |",
                "| `info.json` | Run metadata |",
                "",
                "Load:",
                "",
                "```python",
                "import json",
                "import numpy as np",
                "",
                f'emb = np.load("{npy_path.relative_to(ROOT)}")',
                f'ids = json.loads(Path("{ids_path.relative_to(ROOT)}").read_text())',
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved {embeddings.shape} -> {npy_path}")
    print(f"Metadata -> {meta_jsonl}")
    print(f"Info -> {info_path}")


if __name__ == "__main__":
    main()
