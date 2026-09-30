"""Aggressive context distillation & chunk filtering before any frontier-LLM call.

1. Strip boilerplate (page footers, table header rows, panel intros).
2. Drop raw table rows / key-value panels - Tier 0 already parsed them into JSON facts.
3. Deduplicate sentences.
4. Rank the rest with BM25 against the Tier 2 task queries and keep the top slice,
   in original chronological order, anchored by each log entry's date/title line.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .ingest import ClaimDocument
from .tier1_nlp import tokenize

TASK_QUERIES = {
    "liability": "liability disputed fault conflicting statement witness signal yield comparative negligence speed",
    "causation": "pre-existing degenerative prior injury causation aggravation peer review chronic baseline",
    "damages": "demand offer counteroffer settlement billed reasonable wage lien reserve authorized",
    "complaint": "complaint denied inflated excessive unsupported asserts attorney counsel refused",
    "treatment": "injection MRI surgery fusion treatment maximum medical improvement pain",
    "exposure": "Stowers bad faith excess exposure policy limits deadline verdict",
    "resolution": "settlement agreed agreement release executed payment issued closed recovery",
    "authority": "reserve increased supervisor approval authority bracket committee",
}
BOILERPLATE = [
    re.compile(r"^Claim File: .+ Page \d+ of \d+$"),
    re.compile(r"^(?:Date of Svc|Provider|Item / Period|Coverage Reserve Line|Date Party)\b"),
    re.compile(r"(?:is|are) (?:audited|finalized) (?:below|as follows)\b", re.I),
    re.compile(r"^Record of formal written and verbal negotiation", re.I),
]
MONEY = re.compile(r"\$[\d,]+")
PANEL_LABEL = re.compile(r"[A-Z][A-Za-z/ #.]{2,40}:\s")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
ABBREV_END = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|vs|v|Co|Inc|Est|No|St|Jr|Indem|Adj|Exp)|\b[A-Z])\.$")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in SENT_SPLIT.split(text):
        if out and ABBREV_END.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


@dataclass
class Sentence:
    note_id: str
    anchor: str
    text: str
    score: float = 0.0


@dataclass
class DistilledContext:
    text: str
    stats: dict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tfs = [Counter(d) for d in docs]

    def score(self, query: list[str], i: int) -> float:
        tf, dl = self.tfs[i], len(self.docs[i])
        s = 0.0
        for q in query:
            if q in tf:
                f = tf[q]
                s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def _is_structured(sentence: str) -> bool:
    """Table rows (3+ amounts) and key/value panels (3+ labels) - already parsed by Tier 0."""
    return len(MONEY.findall(sentence)) >= 3 or len(PANEL_LABEL.findall(sentence)) >= 3


def distill(doc: ClaimDocument, cfg: dict) -> DistilledContext:
    sentences: list[Sentence] = []
    seen: set[str] = set()
    dropped = Counter()
    for chunk in doc.chunks:
        anchor = (
            f"[{chunk.note_id} {chunk.date} {chunk.title} - {chunk.author}]"
            if chunk.kind == "log"
            else f"[{chunk.note_id} Section {chunk.section_no}: {chunk.section_title}]"
        )
        for s in split_sentences(chunk.text):
            s = s.strip()
            key = " ".join(tokenize(s))
            if not key:
                continue
            if any(rx.search(s) for rx in BOILERPLATE):
                dropped["boilerplate"] += 1
            elif _is_structured(s):
                dropped["tables_panels_replaced_by_facts"] += 1
            elif key in seen:
                dropped["duplicates"] += 1
            else:
                seen.add(key)
                sentences.append(Sentence(chunk.note_id, anchor, s))

    bm25 = BM25([tokenize(s.text) for s in sentences], cfg.get("bm25_k1", 1.5), cfg.get("bm25_b", 0.75))
    queries = [tokenize(q) for q in TASK_QUERIES.values()]
    for i, s in enumerate(sentences):
        s.score = max(bm25.score(q, i) for q in queries)

    keep_n = max(cfg.get("min_keep", 25), int(len(sentences) * cfg.get("keep_ratio", 0.35)))
    ranked = sorted(range(len(sentences)), key=lambda i: -sentences[i].score)
    keep = {i for i in ranked[:keep_n] if sentences[i].score > 0}
    dropped["low_relevance"] = len(sentences) - len(keep)

    # Re-assemble chronologically; every log entry keeps its anchor line so the
    # model can still build a full timeline even when its body was filtered out.
    lines, last_anchor = [], None
    log_anchors = {c.note_id for c in doc.chunks if c.kind == "log"}
    by_note: dict[str, list[str]] = {}
    for i, s in enumerate(sentences):
        if i in keep:
            by_note.setdefault(s.note_id, []).append(s.text)
    for chunk in doc.chunks:
        kept = by_note.get(chunk.note_id)
        if not kept and chunk.note_id not in log_anchors:
            continue
        anchor = next((s.anchor for s in sentences if s.note_id == chunk.note_id), None)
        if anchor and anchor != last_anchor:
            lines.append(anchor)
            last_anchor = anchor
        lines.extend(f"  {t}" for t in kept or [])
    text = "\n".join(lines)

    raw_chars = len(doc.raw_text)
    stats = {
        "raw_chars": raw_chars,
        "distilled_chars": len(text),
        "approx_raw_tokens": raw_chars // 4,
        "approx_distilled_tokens": len(text) // 4,
        "reduction_pct": round(100 * (1 - len(text) / raw_chars), 1),
        "sentences_total": len(sentences) + sum(v for k, v in dropped.items() if k != "low_relevance"),
        "sentences_kept": len(keep),
        "dropped": dict(dropped),
    }
    return DistilledContext(text=text, stats=stats)
