"""Tier 1 - lightweight / local classification and sentiment.

Default backend is a TF-IDF nearest-centroid (Rocchio) classifier trained from
the taxonomy seed phrases plus any pseudo-labels distilled from Tier 2 runs
(see distillation.py). It runs on CPU in microseconds and needs no downloads.

An optional backend calls a local small language model (Llama-3 / Mistral 3B-8B)
through Ollama; any failure falls back to the centroid model.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from .models import Tier1Result

STOPWORDS = set(
    """a an and are as at be been by for from had has have he her his in is it its of on or our she
    that the their them they this to was were which while who will with we via per not no""".split()
)
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s")


def _stem(tok: str) -> str:
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    return [_stem(t) for t in toks if t not in STOPWORDS and len(t) > 2]


# --------------------------------------------------------------------------- #
# Sentiment (domain lexicon with negation scope)
# --------------------------------------------------------------------------- #
NEG_TERMS = {
    "dispute", "disputed", "conflicting", "contested", "deny", "denies", "denied", "injury", "injuries",
    "pain", "trauma", "concussion", "laceration", "fracture", "inflated", "excessive", "unsupported",
    "aggravation", "herniation", "radiculopathy", "failed", "hazard", "crush", "unreasonable",
    "delinquency", "severe", "suffering", "unjust", "bleed", "impingement", "degenerative", "chronic",
}
POS_TERMS = {
    "confirmed", "affirmed", "agreed", "resolved", "satisfied", "satisfy", "successful", "successfully",
    "relief", "improvement", "returned", "reassured", "verified", "concluded", "cleared", "approved",
    "reasonable", "corroborates", "supported", "compliance", "preserving", "accepted", "frees",
}
NEGATORS = {"no", "not", "zero", "without", "neither", "negative", "non", "never"}


def sentiment(text: str) -> tuple[str, float]:
    words = re.findall(r"[a-z]+", text.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        polarity = 1 if w in POS_TERMS else -1 if w in NEG_TERMS else 0
        if not polarity:
            continue
        if any(prev in NEGATORS for prev in words[max(0, i - 3): i]):
            polarity = -polarity
        pos += polarity > 0
        neg += polarity < 0
    if pos + neg == 0:
        return "neutral", 0.0
    score = (pos - neg) / (pos + neg)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return label, round(score, 3)


# --------------------------------------------------------------------------- #
# Centroid classifier
# --------------------------------------------------------------------------- #
class CentroidClassifier:
    def __init__(self, taxonomy: dict, pseudo_labels: Iterable[dict] = (), pseudo_weight: float = 1.0,
                 temperature: float = 0.10):
        self.temperature = temperature
        docs: list[tuple[str, list[str], float]] = []
        for cat, spec in taxonomy["categories"].items():
            docs.append((cat, tokenize(spec["description"]), 1.0))
            docs.extend((cat, tokenize(s), 1.0) for s in spec["seeds"])
        self.n_pseudo = 0
        for row in pseudo_labels:
            if row.get("category") in taxonomy["categories"]:
                docs.append((row["category"], tokenize(row["text"]), pseudo_weight))
                self.n_pseudo += 1

        df = Counter(t for _, toks, _ in docs for t in set(toks))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}

        sums: dict[str, Counter] = {}
        for cat, toks, w in docs:
            vec = self._vec(toks)
            acc = sums.setdefault(cat, Counter())
            for t, v in vec.items():
                acc[t] += v * w
        self.centroids = {cat: self._norm(v) for cat, v in sums.items()}
        self.categories = list(self.centroids)

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return self._norm({t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items() if t in self.idf})

    @staticmethod
    def _norm(v: dict[str, float]) -> dict[str, float]:
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / z for k, x in v.items()}

    def predict(self, text: str, restrict_to: Optional[list[str]] = None) -> list[tuple[str, float]]:
        vec = self._vec(tokenize(text))
        cats = [c for c in (restrict_to or self.categories) if c in self.centroids]
        sims = {c: sum(vec.get(t, 0.0) * w for t, w in self.centroids[c].items()) for c in cats}
        # Softmax over cosine similarities -> calibrated-ish probabilities.
        m = max(sims.values())
        exps = {c: math.exp((s - m) / self.temperature) for c, s in sims.items()}
        z = sum(exps.values())
        return sorted(((c, e / z) for c, e in exps.items()), key=lambda x: -x[1])


class OllamaClassifier:
    """Local SLM backend (e.g. Llama-3-8B / Mistral-7B on CPU/T4) via Ollama's HTTP API."""

    def __init__(self, url: str, model: str, categories: list[str], timeout_s: int = 60):
        self.url, self.model, self.categories, self.timeout_s = url, model, categories, timeout_s

    def predict(self, text: str, restrict_to: Optional[list[str]] = None) -> list[tuple[str, float]]:
        cats = restrict_to or self.categories
        prompt = (
            "Classify this insurance claim note into exactly one category.\n"
            f"Categories: {', '.join(cats)}\n"
            'Reply as JSON: {"category": "<one category>", "confidence": <0..1>}\n\n'
            f"Note: {text[:3000]}"
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "format": "json", "stream": False}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(json.loads(resp.read())["response"])
        cat, conf = out["category"], float(out.get("confidence", 0.0))
        if cat not in cats:
            raise ValueError(f"SLM returned unknown category {cat!r}")
        rest = (1 - conf) / max(1, len(cats) - 1)
        return [(cat, conf)] + [(c, rest) for c in cats if c != cat]


class Tier1:
    def __init__(self, taxonomy: dict, cfg: dict, pseudo_label_path: Optional[Path] = None):
        self.cfg = cfg
        pseudo = _read_jsonl(pseudo_label_path) if pseudo_label_path else []
        self.centroid = CentroidClassifier(
            taxonomy, pseudo, cfg.get("pseudo_label_weight", 1.0), cfg.get("temperature", 0.10)
        )
        self.slm = None
        if cfg.get("backend") == "ollama":
            o = cfg["ollama"]
            self.slm = OllamaClassifier(o["url"], o["model"], self.centroid.categories, o.get("timeout_s", 60))

    @property
    def n_pseudo_labels(self) -> int:
        return self.centroid.n_pseudo

    def classify(self, text: str, candidates: Optional[list[str]] = None) -> Tier1Result:
        backend = "centroid"
        ranked = None
        if self.slm:
            try:
                ranked, backend = self.slm.predict(text, candidates), f"ollama:{self.slm.model}"
            except Exception:  # noqa: BLE001 - any SLM failure degrades to the local model
                ranked = None
        if ranked is None:
            ranked = self.centroid.predict(text, candidates)
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else ("", 0.0)
        label, score = sentiment(text)
        return Tier1Result(
            category=top[0],
            confidence=round(top[1], 4),
            margin=round(top[1] - second[1], 4),
            top_k=ranked[:3],
            sentiment=label,
            sentiment_score=score,
            backend=backend,
        )

    def is_confident(self, r: Tier1Result) -> bool:
        return r.confidence >= self.cfg["confidence_threshold"] and r.margin >= self.cfg["margin_threshold"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
