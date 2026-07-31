#!/usr/bin/env python3
"""Embed AI-datacenter claim–stance posts with a Qwen3 Embedding model.

Produces two embedding runs for comparison:

1. **vanilla** — document / passage mode (no instruction prompt)
2. **stance_instruct** — Qwen instruction format that asks the model to
   encode attitudinal stance (supportive / critical / neutral)

Default model is ``Qwen/Qwen3-Embedding-0.6B``. Pass ``--model`` for larger
variants (e.g. ``Qwen/Qwen3-Embedding-4B``). Outputs land under::

    data/embeddings/claims_stances_<model-slug>/{vanilla,stance_instruct}/
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "claims_stances" / "claims.jsonl"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def model_slug(model_id: str) -> str:
    """Qwen/Qwen3-Embedding-4B -> qwen3-embedding-4b"""
    name = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def default_out_root(model_id: str) -> Path:
    return ROOT / "data" / "embeddings" / f"claims_stances_{model_slug(model_id)}"

# Custom task instruction for the stance-aware run (English, per Qwen guidance).
STANCE_TASK = (
    "Identify the attitudinal stance of the social media post toward a claim "
    "about increasing AI datacenters as supportive, critical, or neutral; "
    "encode that stance as the primary semantic signal while preserving topic content."
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


def load_model(
    model_id: str,
    device: str | None,
    dtype: str | None,
) -> SentenceTransformer:
    model_kwargs: dict = {}
    if dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise SystemExit(f"Unsupported --dtype {dtype!r}")
        model_kwargs["torch_dtype"] = dtype_map[dtype]

    load_kwargs: dict = {
        "processor_kwargs": {"padding_side": "left"},
        "model_kwargs": model_kwargs,
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
        # sentence-transformers prepends `prompt` to each text. Qwen format:
        # Instruct: <task>\nQuery:<text>
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
                "claim_id": rec["claim_id"],
                "topic": rec["topic"],
                "topic_label": rec["topic_label"],
                "aspect": rec["aspect"],
                "aspect_label": rec["aspect_label"],
                "stance": rec["stance"],
                "claim": rec["claim"],
                "language": rec.get("language", "en"),
                "language_name": rec.get("language_name", "English"),
                "text": rec["text"],
            }
        )

    meta_jsonl = out_dir / "metadata.jsonl"
    with meta_jsonl.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids_path = out_dir / "ids.json"
    ids_path.write_text(
        json.dumps(
            [
                {
                    "index": r["index"],
                    "post_id": r["post_id"],
                    "claim_id": r["claim_id"],
                    "topic": r["topic"],
                    "aspect": r["aspect"],
                    "stance": r["stance"],
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
            "stance_instruct = Qwen Instruct/Query prefix asking the model to "
            "encode supportive vs critical vs neutral stance. "
            "Row i in embeddings.npy matches metadata.jsonl / ids.json index i."
        ),
        "files": {
            "embeddings.npy": f"shape ({embeddings.shape[0]}, {embeddings.shape[1]}) float32",
            "metadata.jsonl": "One JSON object per row, includes full text + claim",
            "ids.json": "Compact index/post_id/claim_id/aspect/stance for each row",
        },
        "stances": sorted({r["stance"] for r in records}),
        "aspects": sorted({r["aspect"] for r in records}),
    }
    (out_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Claim–stance embeddings (`{mode}`)",
                "",
                f"- **Model:** `{model_id}`",
                f"- **Mode:** `{mode}`",
                f"- **Records:** {len(records)}",
                f"- **Dim:** {embeddings.shape[1]}",
                "- **Normalized:** yes (L2)",
                f"- **Prompt:** `{prompt}`" if prompt else "- **Prompt:** none (vanilla document mode)",
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
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root (default: data/embeddings/claims_stances_<model-slug>/).",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float16", "bfloat16", "float32"),
        help="Optional torch dtype for model weights (use float16 for larger models on CPU).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("vanilla", "stance_instruct"),
        default=["vanilla", "stance_instruct"],
        help="Which embedding modes to run (default: both).",
    )
    parser.add_argument(
        "--task",
        default=STANCE_TASK,
        help="Task instruction used for stance_instruct mode.",
    )
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = default_out_root(args.model)

    records = load_records(args.input)
    texts = [r["text"] for r in records]
    print(f"Loaded {len(records)} texts from {args.input}")
    print(f"Loading model {args.model!r} (dtype={args.dtype}) …")
    model = load_model(args.model, args.device, args.dtype)

    modes = {
        "vanilla": {
            "prompt": None,
            "task": None,
        },
        "stance_instruct": {
            # Official Qwen format: Instruct: …\nQuery:<text>
            "prompt": f"Instruct: {args.task}\nQuery:",
            "task": args.task,
        },
    }

    for mode in args.modes:
        cfg = modes[mode]
        print(f"\n=== Encoding mode={mode!r} batch_size={args.batch_size} ===")
        if cfg["prompt"]:
            print(f"Prompt prefix: {cfg['prompt'][:120]}…")
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
        "dtype": args.dtype,
        "input": str(args.input.relative_to(ROOT)),
        "modes": list(args.modes),
        "n_records": len(records),
        "out_root": str(args.out_root.relative_to(ROOT)),
        "stance_task": args.task,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    (args.out_root / "README.md").write_text(
        "\n".join(
            [
                f"# Claim–stance embeddings ({short})",
                "",
                "Two embedding runs over `data/claims_stances/claims.jsonl`:",
                "",
                "| Subdir | Mode |",
                "| --- | --- |",
                "| `vanilla/` | No instruction (document / passage mode) |",
                "| `stance_instruct/` | Instructed to encode supportive / critical / neutral stance |",
                "",
                "Regenerate:",
                "",
                "```bash",
                f"python scripts/embed_claim_stances.py --model {args.model}"
                + (f" --dtype {args.dtype}" if args.dtype else ""),
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
