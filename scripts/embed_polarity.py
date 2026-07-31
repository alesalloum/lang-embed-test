#!/usr/bin/env python3
"""Embed AI-datacenter phenomenon-polarity posts with Qwen3-Embedding-0.6B.

Produces two embedding runs for comparison:

1. **vanilla** — document / passage mode (no instruction prompt)
2. **polarity_instruct** — Qwen instruction format that asks the model to
   encode polarity toward AI datacenter expansion (pro / against / neutral)

Outputs land under::

    data/embeddings/polarity_qwen3-embedding-0.6b/{vanilla,polarity_instruct}/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "polarity_posts" / "posts.jsonl"
DEFAULT_OUT_ROOT = ROOT / "data" / "embeddings" / "polarity_qwen3-embedding-0.6b"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

POLARITY_TASK = (
    "Identify the polarity of the social media post toward increasing AI "
    "datacenters as pro (supportive of expansion), against (opposed to "
    "expansion), or neutral; encode that polarity as the primary semantic "
    "signal while preserving topical content."
)


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


def load_model(model_id: str, device: str | None) -> SentenceTransformer:
    load_kwargs: dict = {
        "processor_kwargs": {"padding_side": "left"},
    }
    if device:
        load_kwargs["device"] = device
    try:
        return SentenceTransformer(model_id, **load_kwargs)
    except TypeError:
        load_kwargs.pop("processor_kwargs", None)
        load_kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        return SentenceTransformer(model_id, **load_kwargs)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    prompt: str | None,
) -> np.ndarray:
    encode_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if prompt is not None:
        encode_kwargs["prompt"] = prompt
    embeddings = model.encode(texts, **encode_kwargs)
    return np.asarray(embeddings, dtype=np.float32)


def write_run(
    out_dir: Path,
    embeddings: np.ndarray,
    records: list[dict],
    *,
    model_id: str,
    input_path: Path,
    mode: str,
    prompt: str | None,
    task: str | None,
) -> None:
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
                "aspect": rec["aspect"],
                "aspect_label": rec["aspect_label"],
                "polarity": rec["polarity"],
                "polarity_label": rec["polarity_label"],
                "language": rec.get("language", "en"),
                "language_name": rec.get("language_name", "English"),
                "text": rec["text"],
            }
        )

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "ids.json").write_text(
        json.dumps(
            [
                {
                    "index": r["index"],
                    "post_id": r["post_id"],
                    "topic": r["topic"],
                    "aspect": r["aspect"],
                    "polarity": r["polarity"],
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
        "model": model_id,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.relative_to(ROOT)),
        "n_records": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "prompt": prompt,
        "task_instruction": task,
        "notes": (
            "Vanilla = document embeddings with no instruction. "
            "polarity_instruct = Qwen Instruct/Query prefix asking the model "
            "to encode pro vs against vs neutral polarity toward AI "
            "datacenter expansion. "
            "Row i in embeddings.npy matches metadata.jsonl / ids.json index i."
        ),
        "files": {
            "embeddings.npy": f"shape ({embeddings.shape[0]}, {embeddings.shape[1]}) float32",
            "metadata.jsonl": "One JSON object per row, includes full text",
            "ids.json": "Compact index/post_id/aspect/polarity for each row",
        },
        "polarities": sorted({r["polarity"] for r in records}),
        "aspects": sorted({r["aspect"] for r in records}),
    }
    (out_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Polarity embeddings (`{mode}`)",
                "",
                f"- **Model:** `{model_id}`",
                f"- **Mode:** `{mode}`",
                f"- **Records:** {len(records)}",
                f"- **Dim:** {embeddings.shape[1]}",
                "- **Normalized:** yes (L2)",
                (
                    f"- **Prompt:** `{prompt}`"
                    if prompt
                    else "- **Prompt:** none (vanilla document mode)"
                ),
                "",
                "| File | Description |",
                "| --- | --- |",
                "| `embeddings.npy` | `(n, d)` float32 matrix |",
                "| `metadata.jsonl` | Row metadata + text |",
                "| `ids.json` | Compact ids for plotting |",
                "| `info.json` | Run metadata |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[{mode}] Saved {embeddings.shape} -> {npy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("vanilla", "polarity_instruct"),
        default=["vanilla", "polarity_instruct"],
    )
    parser.add_argument("--task", default=POLARITY_TASK)
    args = parser.parse_args()

    records = load_records(args.input)
    texts = [r["text"] for r in records]
    print(f"Loaded {len(records)} texts from {args.input}")
    print(f"Loading model {args.model!r} …")
    model = load_model(args.model, args.device)

    modes = {
        "vanilla": {"prompt": None, "task": None},
        "polarity_instruct": {
            "prompt": f"Instruct: {args.task}\nQuery:",
            "task": args.task,
        },
    }

    for mode in args.modes:
        cfg = modes[mode]
        print(f"\n=== Encoding mode={mode!r} batch_size={args.batch_size} ===")
        if cfg["prompt"]:
            print(f"Prompt prefix: {cfg['prompt'][:140]}…")
        embeddings = encode_texts(model, texts, args.batch_size, cfg["prompt"])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise SystemExit(
                f"Unexpected embedding shape {embeddings.shape}, "
                f"expected ({len(records)}, d)"
            )
        write_run(
            args.out_root / mode,
            embeddings,
            records,
            model_id=args.model,
            input_path=args.input,
            mode=mode,
            prompt=cfg["prompt"],
            task=cfg["task"],
        )

    summary = {
        "model": args.model,
        "input": str(args.input.relative_to(ROOT)),
        "modes": list(args.modes),
        "n_records": len(records),
        "out_root": str(args.out_root.relative_to(ROOT)),
        "polarity_task": args.task,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "README.md").write_text(
        "\n".join(
            [
                "# Phenomenon-polarity embeddings (Qwen3-Embedding-0.6B)",
                "",
                "Two embedding runs over `data/polarity_posts/posts.jsonl`:",
                "",
                "| Subdir | Mode |",
                "| --- | --- |",
                "| `vanilla/` | No instruction (document / passage mode) |",
                "| `polarity_instruct/` | Instructed to encode pro / against / neutral polarity |",
                "",
                "Regenerate:",
                "",
                "```bash",
                "python scripts/embed_polarity.py",
                "```",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nDone. Outputs under {args.out_root}")


if __name__ == "__main__":
    main()
