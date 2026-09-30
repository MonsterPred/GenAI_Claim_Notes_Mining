"""Task distillation: frontier-model labels -> training data for the cheap tiers.

Every Tier 2 note label is appended to a pseudo-label store. On the next run the
Tier 1 centroid model is rebuilt with those examples (so the same kind of note
stops escalating), and `export-finetune` turns the store into a chat-format JSONL
for fine-tuning a self-hosted 3B-8B model (e.g. with Hugging Face TRL / Unsloth),
which can then be plugged in as the Tier 1 "ollama" backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import NoteResult

FINETUNE_SYSTEM = "Classify the insurance claim note. Reply with JSON: {\"category\", \"sentiment\", \"reason_code\"}."


def _key(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def append_pseudo_labels(path: Path, claim_id: str, results: list[NoteResult], model: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            existing = {json.loads(line)["key"] for line in f if line.strip()}
    added = 0
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            if r.resolved_by != "tier2" or not r.tier2:
                continue
            k = _key(r.chunk.text)
            if k in existing:
                continue
            f.write(json.dumps({
                "key": k,
                "claim_id": claim_id,
                "note_id": r.chunk.note_id,
                "text": r.chunk.text,
                "category": r.tier2["category"],
                "sentiment": r.tier2["sentiment"],
                "reason_code": r.tier2["reason_code"],
                "teacher": model,
            }) + "\n")
            existing.add(k)
            added += 1
    return added


def export_finetune(pseudo_path: Path, out_path: Path) -> int:
    if not pseudo_path.exists():
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with pseudo_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            target = {k: row[k] for k in ("category", "sentiment", "reason_code")}
            dst.write(json.dumps({"messages": [
                {"role": "system", "content": FINETUNE_SYSTEM},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": json.dumps(target)},
            ]}) + "\n")
            n += 1
    return n
