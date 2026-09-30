"""Cascading router: Tier 0 (rules) -> Tier 1 (local NLP) -> Tier 2 (frontier LLM)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic

from . import tier0_rules
from .distill import distill
from .distillation import append_pseudo_labels
from .ingest import load_claim_pdf
from .models import NoteResult
from .report import render_markdown
from .tier1_nlp import Tier1, sentiment
from .tier2_llm import Tier2Error

log = logging.getLogger(__name__)


@dataclass
class RunOutput:
    claim_id: str
    out_dir: Path
    notes: list[NoteResult]
    facts: dict[str, Any]
    checks: list[dict[str, Any]]
    synthesis: Optional[dict[str, Any]]
    report: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(pdf_path: str | Path, cfg: dict, use_llm: bool = True, out_dir: Optional[str | Path] = None) -> RunOutput:
    timings: dict[str, float] = {}
    taxonomy = load_config(cfg["taxonomy_path"])
    pseudo_path = Path(cfg["pseudo_label_path"])

    t = time.perf_counter()
    doc = load_claim_pdf(pdf_path)
    timings["ingest_s"] = time.perf_counter() - t
    claim_id = doc.header.get("claim_id", Path(pdf_path).stem)

    # ---------------- Tier 0 ----------------
    t = time.perf_counter()
    facts = tier0_rules.extract_tables(doc)
    results = [NoteResult(chunk=c, tier0=tier0_rules.analyze_note(c)) for c in doc.chunks]
    checks = tier0_rules.reconcile(facts, [(r.chunk, r.tier0) for r in results])
    for r in results:
        if r.tier0.category:
            r.category, r.resolved_by = r.tier0.category, "tier0"
            r.sentiment = sentiment(r.chunk.text)[0]
    timings["tier0_s"] = time.perf_counter() - t

    # ---------------- Tier 1 ----------------
    t = time.perf_counter()
    tier1 = Tier1(taxonomy, cfg["tier1"], pseudo_path)
    escalate: list[NoteResult] = []
    for r in results:
        if r.resolved_by == "tier0":
            continue
        cands = r.tier0.candidate_categories or None
        r.tier1 = tier1.classify(r.chunk.text, cands)
        r.sentiment = r.tier1.sentiment
        if tier1.is_confident(r.tier1):
            r.category, r.resolved_by = r.tier1.category, "tier1"
        else:
            if cands:
                r.escalation_reasons.append(f"ambiguous_rules:{'|'.join(cands)}")
            r.escalation_reasons.append(f"low_confidence(p={r.tier1.confidence:.2f},margin={r.tier1.margin:.2f})")
            escalate.append(r)
    timings["tier1_s"] = time.perf_counter() - t

    # ---------------- Context distillation ----------------
    t = time.perf_counter()
    ctx = distill(doc, cfg["distill"])
    timings["distill_s"] = time.perf_counter() - t

    # ---------------- Tier 2 ----------------
    synthesis, tier2_errors, tier2 = None, [], None
    llm_on = use_llm and cfg["tier2"].get("enabled", True)
    t = time.perf_counter()
    if llm_on:
        try:
            tier2 = _make_tier2(taxonomy, cfg["tier2"])
        except Tier2Error as e:
            tier2_errors.append(f"tier2 init: {e}")
            log.error("Tier 2 unavailable: %s", e)
            llm_on = False
    if llm_on:
        if escalate:
            hints = [(r.chunk, {"flags": sorted(r.tier0.flags),
                                "tier1_top": [[c, round(p, 2)] for c, p in r.tier1.top_k]}) for r in escalate]
            try:
                labels = tier2.label_notes(hints)
                for r in escalate:
                    if lab := labels.get(r.chunk.note_id):
                        r.tier2 = lab.model_dump()
                        r.category, r.sentiment, r.resolved_by = lab.category, lab.sentiment, "tier2"
            except (anthropic.APIStatusError, anthropic.APIConnectionError, Tier2Error) as e:
                tier2_errors.append(f"label_notes: {type(e).__name__}: {e}")
                log.error("Tier 2 note labeling failed: %s", e)
        try:
            index = [{"id": r.chunk.note_id, "date": r.chunk.date, "title": r.chunk.title or r.chunk.section_title,
                      "cat": r.category, "sent": r.sentiment} for r in results]
            failed = [c for c in checks if c["status"] == "FAIL"]
            synth = tier2.synthesize(claim_id, {**facts, "reconciliation_failures": failed}, index, ctx.text)
            synthesis = synth.model_dump() if synth else None
        except (anthropic.APIStatusError, anthropic.APIConnectionError, Tier2Error) as e:
            tier2_errors.append(f"synthesis: {type(e).__name__}: {e}")
            log.error("Tier 2 synthesis failed: %s", e)
    timings["tier2_s"] = time.perf_counter() - t

    for r in results:  # escalated but never answered by Tier 2 -> keep best local guess, flagged
        if r.resolved_by == "unresolved" and r.tier1:
            r.category, r.resolved_by = r.tier1.category, "tier1_unconfirmed"

    added = 0
    if tier2 is not None:
        added = append_pseudo_labels(pseudo_path, claim_id, results, cfg["tier2"]["model"])

    out = RunOutput(claim_id, Path(out_dir or cfg["output_dir"]) / claim_id, results, facts, checks, synthesis)
    out.report = _report(out, doc, ctx, tier1, tier2, timings, tier2_errors, added, llm_on)
    _write(out, ctx.text)
    return out


def _make_tier2(taxonomy: dict, cfg: dict):
    from .tier2_llm import Tier2Client
    return Tier2Client(taxonomy, cfg)


def _report(out: RunOutput, doc, ctx, tier1: Tier1, tier2, timings, errors, added, llm_on) -> dict:
    routing: dict[str, int] = {}
    for r in out.notes:
        routing[r.resolved_by] = routing.get(r.resolved_by, 0) + 1
    usage = [asdict(u) for u in tier2.usage] if tier2 else []
    totals = {k: sum(u[k] for u in usage) for k in
              ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "cost_usd")}
    totals["cost_usd"] = round(totals["cost_usd"], 4)

    # Naive baseline: every note sent to the frontier model on its own, plus a synthesis
    # call over the raw file, no caching / distillation. Estimated at ~4 chars per token.
    sys_tokens = len(tier2.system_prompt) // 4 if tier2 else 0
    baseline_in = sum(sys_tokens + len(r.chunk.text) // 4 for r in out.notes) + sys_tokens + len(doc.raw_text) // 4
    return {
        "claim_id": out.claim_id,
        "source": doc.source,
        "pages": doc.page_count,
        "notes_total": len(out.notes),
        "routing": routing,
        "llm_enabled": llm_on,
        "tier2_backend": tier2.backend if tier2 else None,
        "tier2_billing": tier2.billing if tier2 else None,
        "llm_calls": len(usage),
        "tier1_backend": tier1.cfg.get("backend"),
        "tier1_pseudo_labels_loaded": tier1.n_pseudo_labels,
        "pseudo_labels_added": added,
        "reconciliation": {"pass": sum(c["status"] == "pass" for c in out.checks),
                           "fail": sum(c["status"] == "FAIL" for c in out.checks)},
        "distillation": ctx.stats,
        "token_usage": {"calls": usage, "totals": totals,
                        "naive_baseline_input_tokens_est": baseline_in if tier2 else None},
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
        "errors": errors,
    }


def _write(out: RunOutput, context_text: str) -> None:
    out.out_dir.mkdir(parents=True, exist_ok=True)
    with (out.out_dir / "notes.jsonl").open("w", encoding="utf-8") as f:
        for r in out.notes:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    (out.out_dir / "facts.json").write_text(
        json.dumps({**out.facts, "reconciliation": out.checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    if out.synthesis:
        (out.out_dir / "synthesis.json").write_text(json.dumps(out.synthesis, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    (out.out_dir / "distilled_context.txt").write_text(context_text, encoding="utf-8")
    (out.out_dir / "run_report.json").write_text(json.dumps(out.report, indent=2), encoding="utf-8")
    (out.out_dir / "report.md").write_text(render_markdown(out), encoding="utf-8")
