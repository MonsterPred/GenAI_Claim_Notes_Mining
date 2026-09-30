"""Human-readable Markdown report for one pipeline run."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import RunOutput


def _money(v) -> str:
    return f"${v:,.2f}" if isinstance(v, (int, float)) else str(v)


def render_markdown(out: "RunOutput") -> str:
    rep, facts, syn = out.report, out.facts, out.synthesis
    h = facts.get("header", {})
    L: list[str] = [f"# Claim Notes Mining Report - {out.claim_id}", ""]
    L += [f"- **Insured:** {h.get('insured_name', '-')}  |  **Claimant:** {h.get('claimant_name', '-')}",
          f"- **Date of loss:** {h.get('date_of_loss', '-')}  |  **Policy limits:** {h.get('policy_limits', '-')}",
          f"- **Source:** {rep['source']} ({rep['pages']} pages, {rep['notes_total']} notes)", ""]

    L += ["## Cascade routing", "", "| Resolved by | Notes |", "|---|---|"]
    order = ["tier0", "tier1", "tier2", "tier1_unconfirmed", "unresolved"]
    L += [f"| {k} | {rep['routing'][k]} |" for k in order if k in rep["routing"]]
    L += ["", f"Tier 1 backend: `{rep['tier1_backend']}` (pseudo-labels loaded: {rep['tier1_pseudo_labels_loaded']}; "
          f"added this run: {rep['pseudo_labels_added']})", ""]

    L += ["## Note classification", "", "| Note | Date | Title / Section | Category | Tier | Conf. | Sentiment |",
          "|---|---|---|---|---|---|---|"]
    for r in out.notes:
        c = r.chunk
        conf = f"{r.tier1.confidence:.2f}" if r.tier1 else ("rule" if r.resolved_by == "tier0" else "")
        title = (c.title or c.section_title).replace("|", "/")
        L.append(f"| {c.note_id} | {c.date or ''} | {title} | {r.category} | {r.resolved_by} | {conf} | {r.sentiment} |")
    L.append("")

    esc = [r for r in out.notes if r.escalation_reasons]
    if esc:
        L += ["### Escalations to Tier 2", ""]
        for r in esc:
            extra = f" -> **{r.tier2['category']}** ({r.tier2['reason_code']}): {r.tier2['rationale']}" if r.tier2 else ""
            L.append(f"- `{r.chunk.note_id}` {'; '.join(r.escalation_reasons)}{extra}")
        L.append("")

    L += ["## Deterministic reconciliation (Tier 0)", ""]
    fails = [c for c in out.checks if c["status"] == "FAIL"]
    L.append(f"{rep['reconciliation']['pass']} checks passed, {rep['reconciliation']['fail']} failed.")
    L.append("")
    for c in fails:
        L.append(f"- **{c['check']}**: expected {_money(c['expected'])}, found {_money(c['actual'])}. {c.get('detail', '')}")
    if fails:
        L.append("")

    if nego := facts.get("negotiation"):
        L += ["## Negotiation trail", "", "| Date | Party | Type | Amount |", "|---|---|---|---|"]
        L += [f"| {n['date']} | {n['party']} | {n['type']} | {_money(n['amount'])} |" for n in nego]
        if sd := facts.get("settlement_distribution"):
            L.append(f"| | | **Settled** | **{_money(sd.get('Gross BI Settlement'))}** |")
        L.append("")

    if syn:
        L += ["## Claim synthesis (Tier 2)", "", syn["summary"], "",
              f"- **Outcome:** {syn['outcome']}  |  **Final severity:** {syn['final_severity']}/5  |  "
              f"**Insured liability:** {syn['liability_split_insured_pct']}%",
              f"- **Causation:** {syn['causation_finding']}",
              f"- **Risk flags:** {', '.join(syn['risk_flags'])}", ""]
        if syn["disputes"]:
            L += ["### Disputes detected", "", "| Type | Parties | Positions | Status | Code | Evidence |",
                  "|---|---|---|---|---|---|"]
            for d in syn["disputes"]:
                L.append(f"| {d['dispute_type']} | {', '.join(d['parties'])} | {d['position_summary']} | "
                         f"{d['status']} | {d['reason_code']} | {', '.join(d['evidence_note_ids'])} |")
            L.append("")
        if syn["complaint_root_causes"]:
            L += ["### Complaint root causes", ""]
            L += [f"- **{rc['issue']}** - {rc['root_cause']} ({', '.join(rc['evidence_note_ids'])})"
                  for rc in syn["complaint_root_causes"]]
            L.append("")
        L += ["### Chronology", "", "| Date | Event | Notes |", "|---|---|---|"]
        L += [f"| {e['date']} | {e['event']} | {', '.join(e['note_ids'])} |" for e in syn["chronology"]]
        L.append("")

    d = rep["distillation"]
    L += ["## Cost controls", "",
          f"- **Context distillation:** {d['approx_raw_tokens']:,} -> {d['approx_distilled_tokens']:,} est. tokens "
          f"({d['reduction_pct']}% reduction); dropped {d['dropped']}"]
    tu = rep["token_usage"]
    if tu["calls"]:
        t = tu["totals"]
        L += [f"- **LLM calls:** {rep['llm_calls']} (notes escalated: {len(esc)} of {rep['notes_total']})",
              f"- **Tokens:** input {t['input_tokens']:,} | cache write {t['cache_creation_input_tokens']:,} | "
              f"cache read {t['cache_read_input_tokens']:,} | output {t['output_tokens']:,}",
              f"- **Backend:** `{rep['tier2_backend']}` - billed to {rep['tier2_billing']}",
              f"- **Cost:** ${t['cost_usd']:.4f}  (naive all-LLM baseline input est.: "
              f"{tu['naive_baseline_input_tokens_est']:,} tokens)"]
    else:
        L.append("- **LLM calls:** none (offline run)")
    L += [f"- **Timings (s):** {rep['timings_s']}", ""]
    if rep["errors"]:
        L += ["## Errors", ""] + [f"- {e}" for e in rep["errors"]] + [""]
    return "\n".join(L)
