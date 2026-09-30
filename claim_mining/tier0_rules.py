"""Tier 0 - deterministic layer (regex, dictionaries, metadata filters).

Zero token cost, sub-millisecond per note. Three jobs:
  1. classify a note when exactly one high-precision rule fires,
  2. extract entities / flags from every note (fed to Tiers 1-2 as facts),
  3. parse the file's structured tables and reconcile their arithmetic.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .ingest import ClaimDocument
from .models import NoteChunk, Tier0Result

# --------------------------------------------------------------------------- #
# Decisive category rules
# --------------------------------------------------------------------------- #
# Log titles are system-generated activity codes, so title rules are the most
# precise signal. Order does not matter: if >1 category fires the note is
# ambiguous and falls through to Tier 1.
TITLE_RULES: list[tuple[str, str, str]] = [
    ("T_LOR", "ATTORNEY_REPRESENTATION", r"\bLETTER OF REPRESENTATION\b|\(LOR\)"),
    ("T_LIAB", "LIABILITY_INVESTIGATION", r"RECORDED STATEMENT|POLICE (?:CRASH )?REPORT|WITNESS STATEMENT|RECONSTRUCTION"),
    ("T_PD", "PROPERTY_DAMAGE", r"PROPERTY DAMAGE|APPRAISAL|TOTAL LOSS|TOW(?:ING)? INVOICE"),
    ("T_SUBRO", "LIEN_SUBROGATION", r"SUBROGATION|\bLIEN\b"),
    ("T_NEGO", "DEMAND_NEGOTIATION", r"DEMAND PACKAGE|COUNTEROFFER|ROUNDTABLE|NEGOTIATION"),
    ("T_SETTLE", "SETTLEMENT_RELEASE", r"SETTLEMENT AGREEMENT|\bRELEASE\b"),
    ("T_CLOSE", "FILE_CLOSURE", r"CLOSING|CLOSEOUT|FILE CLOSURE|SIGN-?OFF"),
    ("T_INSCOMM", "INSURED_COMMUNICATION", r"(?:TELECONFERENCE|CALL|CONTACT|MEETING) WITH INSURED"),
    ("T_XC", "EXTRACONTRACTUAL_EXPOSURE", r"STOWERS|EXTRACONTRACTUAL|EXCESS|ADVICE LETTER"),
    ("T_DMG", "DAMAGES_AUDIT", r"\bWAGE\b|LETTER OF PROTECTION|\(LOP\)|BILL(?:ING)? AUDIT|SPECIAL DAMAGES"),
    ("T_CAUSE", "MEDICAL_CAUSATION_REVIEW", r"CAUSATION|PEER REVIEW"),
    ("T_ISO", "PRIOR_CLAIMS_ISO", r"\bISO\b|PRIOR (?:CLAIM|MEDICAL)"),
    ("T_RSV", "RESERVE_CHANGE", r"\bRESERVE\b"),
    ("T_MED", "MEDICAL_TREATMENT", r"\bMRI\b|TREATMENT|INJECTION|MEDICAL (?:RECORDS|HISTORY)"),
    ("T_FNOL", "FNOL_INTAKE", r"FIRST NOTICE|\bFNOL\b|INITIAL ASSIGNMENT|TRIAGE"),
    ("T_COV", "COVERAGE_VERIFICATION", r"\bCOVERAGE\b"),
]
# Free-text rules: only unmistakable phrases (explicit attorney letters, ISO hits,
# statute names, standard invoices). Used when a note has no system title.
TEXT_RULES: list[tuple[str, str, str]] = [
    ("X_LOR", "ATTORNEY_REPRESENTATION", r"letter of representation|representation letter"),
    ("X_ISO", "PRIOR_CLAIMS_ISO", r"\bISO ClaimSearch\b"),
    ("X_STOWERS", "EXTRACONTRACTUAL_EXPOSURE", r"\bStowers\b"),
    ("X_TOW", "PROPERTY_DAMAGE", r"towing invoice|tow bill"),
    ("X_RSV", "RESERVE_CHANGE", r"reserve (?:increased|decreased|reduced|raised) from"),
    ("X_REL", "SETTLEMENT_RELEASE", r"executed release|notarized release"),
]
_TITLE_RES = [(rid, cat, re.compile(p)) for rid, cat, p in TITLE_RULES]
_TEXT_RES = [(rid, cat, re.compile(p, re.I)) for rid, cat, p in TEXT_RULES]

# --------------------------------------------------------------------------- #
# Flags and dictionaries
# --------------------------------------------------------------------------- #
FLAG_PATTERNS: dict[str, str] = {
    "attorney_involved": r"\battorney\b|\bcounsel\b|letter of representation",
    "dispute_signal": r"disputed?|conflicting|contested|asserts? 100% liability|denies|failed to yield|inflated",
    "pre_existing_signal": r"pre-existing|degenerative|chronic|prior (?:chronic|lumbar|cervical|back)",
    "prior_claims": r"\bISO\b|prior (?:loss|claim|worker's compensation)",
    "credibility_concern": r"initially denied|denied any prior|misrepresent",
    "invasive_treatment": r"injection|\bESI\b|surgery|surgical|fusion|disc replacement",
    "lop_treatment": r"letter of protection|\(LOP\)|\bLOP\b",
    "lien": r"\blien\b|ERISA|Medicare",
    "subrogation": r"subrogation",
    "total_loss": r"total loss",
    "towing": r"\btow(?:ing|ed)?\b",
    "bad_faith_signal": r"Stowers|bad[- ]faith|extracontractual|excess (?:risk|verdict|exposure)",
    "reserve_activity": r"\breserve",
    "policy_limits_demand": r"policy limits",
    "deadline": r"deadline|within \d+ (?:calendar )?days",
}
_FLAG_RES = {k: re.compile(p, re.I) for k, p in FLAG_PATTERNS.items()}

PROCEDURES = {
    "MRI": r"\bMRI\b", "CT": r"\bCT\b", "ESI": r"\bESI\b|epidural steroid injection",
    "physical_therapy": r"physical therapy|\bPT\b", "suture": r"sutur", "fusion": r"\bfusion\b",
    "disc_replacement": r"disc replacement", "ambulance": r"ambulance|\bEMS\b",
}
CARRIERS = {
    "Geico": r"\bGeico\b", "State Farm": r"State Farm", "Travelers": r"\bTravelers\b",
    "BCBSTX": r"BCBSTX|Blue Cross Blue Shield", "Optum": r"\bOptum\b",
    "Medicare/CMS": r"\bMedicare\b|\bCMS\b", "The Hartford": r"The Hartford",
}
_PROC_RES = {k: re.compile(p, re.I) for k, p in PROCEDURES.items()}
_CARRIER_RES = {k: re.compile(p) for k, p in CARRIERS.items()}

MONEY_RE = re.compile(r"(?<![\w$])([+-]?)\$([\d,]+(?:\.\d{1,2})?)(k\b)?")
PCT_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)%")
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
STATUTE_RE = re.compile(r"CPRC § [\d.]+\d|\bStowers\b|#TX-[A-Z]+-\d+")
SPINE_RE = re.compile(r"\b([CTL]\d)-([CTL]?\d)\b")
REF_RE = re.compile(r"#\s?([A-Z0-9][A-Z0-9-]{4,})")
RESERVE_CHANGE_RE = re.compile(
    r"reserve (?:increased|decreased|reduced|raised|adjusted) from \$([\d,]+) to \$([\d,]+)", re.I
)
RESERVE_INIT_RE = re.compile(r"\$([\d,]+) (BI|PD) indemnity", re.I)


def money(value: str) -> float:
    m = MONEY_RE.search(value)
    if not m:
        raise ValueError(f"no money amount in {value!r}")
    amount = float(m.group(2).replace(",", ""))
    if m.group(3):
        amount *= 1000
    return -amount if m.group(1) == "-" else amount


# --------------------------------------------------------------------------- #
# Per-note analysis
# --------------------------------------------------------------------------- #
def analyze_note(chunk: NoteChunk) -> Tier0Result:
    text = chunk.text
    title_hits = [(rid, cat) for rid, cat, rx in _TITLE_RES if chunk.title and rx.search(chunk.title)]
    text_hits = [(rid, cat) for rid, cat, rx in _TEXT_RES if rx.search(text)]

    decisive = title_hits or text_hits
    cats = sorted({cat for _, cat in decisive})
    category = cats[0] if len(cats) == 1 else None

    flags = {k: True for k, rx in _FLAG_RES.items() if rx.search(text) or (chunk.title and rx.search(chunk.title))}
    entities = _entities(text)
    if "reserve_changes" in entities or "reserve_initial" in entities:
        flags["reserve_activity"] = True

    return Tier0Result(
        category=category,
        fired_rules=[rid for rid, _ in title_hits + text_hits],
        candidate_categories=cats,
        flags=flags,
        entities=entities,
    )


def _entities(text: str) -> dict[str, Any]:
    ent: dict[str, Any] = {}
    amounts = []
    for m in MONEY_RE.finditer(text):
        ctx = text[max(0, m.start() - 60): m.start()].split()[-6:]
        amounts.append({"amount": money(m.group(0)), "context": " ".join(ctx)})
    if amounts:
        ent["amounts"] = amounts
    if pcts := sorted({float(p) for p in PCT_RE.findall(text)}):
        ent["percentages"] = pcts
    if dates := sorted(set(DATE_RE.findall(text))):
        ent["dates"] = dates
    if statutes := sorted(set(STATUTE_RE.findall(text))):
        ent["statutes"] = statutes
    if spine := sorted({f"{a}-{b if b[0].isalpha() else a[0] + b}" for a, b in SPINE_RE.findall(text)}):
        ent["spinal_levels"] = spine
    if procs := [k for k, rx in _PROC_RES.items() if rx.search(text)]:
        ent["procedures"] = procs
    if carriers := [k for k, rx in _CARRIER_RES.items() if rx.search(text)]:
        ent["carriers"] = carriers
    if refs := sorted(set(REF_RE.findall(text))):
        ent["reference_ids"] = refs
    if changes := RESERVE_CHANGE_RE.findall(text):
        ent["reserve_changes"] = [{"from": money("$" + a), "to": money("$" + b)} for a, b in changes]
    if "reserve established" in text.lower():
        if init := RESERVE_INIT_RE.findall(text):
            ent["reserve_initial"] = {line.upper(): money("$" + amt) for amt, line in init}
    return ent


# --------------------------------------------------------------------------- #
# Document-level structured tables
# --------------------------------------------------------------------------- #
AMT = r"[+-]?\$[\d,]+\.\d{2}"


def _kv(text: str, labels: list[str], stop: list[str] = ()) -> list[tuple[str, str]]:
    """Split a flattened key/value panel on known labels (label regex fragments).

    `stop` names the narrative block that follows the panel, so the last value ends there.
    """
    rx = re.compile(r"(" + "|".join([*labels, *stop]) + r")\s*(?:\([^)]*\))?\s*:\s*")
    hits = list(rx.finditer(text))
    out = []
    for i, m in enumerate(hits):
        if m.group(1) in stop:
            break
        end = hits[i + 1].start() if i + 1 < len(hits) else min(len(text), m.end() + 80)
        out.append((m.group(1).replace("\\", ""), text[m.end(): end].strip()))
    return out


def _find_section(doc: ClaimDocument, needle: str) -> str:
    for no, title in doc.sections.items():
        if needle.lower() in title.lower():
            return doc.section_text.get(no, "")
    return ""


def extract_tables(doc: ClaimDocument) -> dict[str, Any]:
    facts: dict[str, Any] = {"header": doc.header}

    recon = _find_section(doc, "Reconstruction")
    if recon:
        facts["reconstruction"] = dict(_kv(recon, [
            "Signal Sequence", "Posted Speed Limit", "Pavement Conditions", r"Lighting / Visibility",
            "Insured Speed at Impact", "Claimant Speed at Impact", r"Dashcam / Traffic Cam",
            r"Comparative Fault Est\.",
        ], stop=["Reconstructionist Field Note"]))

    iso = _find_section(doc, "Prior Claim")
    if iso:
        priors, cur = [], None
        for label, value in _kv(iso, ["Prior Loss Date", r"Carrier / Claim #", "Reported Injury",
                                      "Resolution Status", "Medical Causation Impact"]):
            if label == "Prior Loss Date":
                cur = {"loss_date": value}
                priors.append(cur)
            elif cur is not None and label != "Medical Causation Impact":
                cur[label.lower().replace(" / ", "_").replace(" #", "").replace(" ", "_")] = value
        for p in priors:
            if m := MONEY_RE.search(p.get("resolution_status", "")):
                p["settled_amount"] = money(m.group(0))
        facts["prior_claims"] = priors

    wage = _find_section(doc, "Wage Loss")
    if m := re.search(rf"Total Economic Wage Claim ({AMT}) ({AMT})", wage):
        # Table body = header row .. totals row. Row amounts alternate claimed/audited;
        # rates such as "$1,650/wk" are excluded by the negative lookahead.
        body = wage[wage.find("Variance & Rationale"): m.start()]
        amounts = [money(a) for a in re.findall(r"\$[\d,]+(?:\.\d{2})?(?![\d,]*/)", body)]
        facts["wage_claim"] = {
            "claimed_total": money(m.group(1)),
            "audited_total": money(m.group(2)),
            "claimed_items": amounts[0::2],
            "audited_items": amounts[1::2],
        }

    med = _find_section(doc, "Medical Special Damages")
    if med:
        rows = re.findall(rf"(\$[\d,]+\.\d{{2}}) (\$[\d,]+\.\d{{2}}) (\$[\d,]+\.\d{{2}})", med)
        if rows:
            *items, total = [tuple(money(x) for x in r) for r in rows]
            facts["medical_specials"] = {
                "billed_total": total[0], "allowed_total": total[1], "paid_total": total[2],
                "rows": [{"billed": b, "allowed": a, "paid": p} for b, a, p in items],
            }

    stowers = _find_section(doc, "Stowers")
    if stowers:
        facts["stowers"] = dict(_kv(stowers, [
            "Policy Limits", "Claimant Demand", "Demand Deadline", "Release Terms", "Liability Status",
            "Jury Verdict Verdict Risk", "Excess Verdict Exposure", "Legal Conclusion",
        ], stop=["Coverage Counsel Assessment"]))

    nego = _find_section(doc, "Negotiation")
    if nego:
        facts["negotiation"] = [
            {"date": d, "party": p, "type": t, "amount": money(a)}
            for d, p, t, a in re.findall(rf"(\d{{2}}/\d{{2}}/\d{{4}}) (Plaintiff|Carrier) \(([^)]+)\) ({AMT})", nego)
        ]

    dist = _find_section(doc, "Release Terms")
    if dist:
        kv = _kv(dist, [
            "Gross BI Settlement", r"Attorney Fee \([\d.]+%\)", "Legal Case Expenses", "BCBSTX Health Lien",
            "Sugar Land EMS Out-of-Pocket", "LOP Provider Compromise", "Net Recovery to Claimant",
            "Release Form Type",
        ], stop=["Indemnification & Hold Harmless Clause"])
        facts["settlement_distribution"] = {
            k: (money(v) if MONEY_RE.search(v) and k != "Release Form Type" else v) for k, v in kv
        }

    ledger = _find_section(doc, "Financial Ledger")
    if ledger:
        body = ledger.split("Redundancy", 1)[-1]
        facts["reserve_ledger"] = [
            {"line": line.strip(), "initial": money(a), "peak": money(b), "final_paid": money(c), "redundancy": money(d)}
            for line, a, b, c, d in re.findall(rf"([A-Z][A-Za-z .()]+?) ({AMT}) ({AMT}) ({AMT}) ({AMT})", body)
        ]
    return facts


# --------------------------------------------------------------------------- #
# Deterministic reconciliation (arithmetic + cross-note consistency)
# --------------------------------------------------------------------------- #
def _check(name: str, expected: float, actual: float, tol: float = 0.01) -> dict[str, Any]:
    return {"check": name, "expected": round(expected, 2), "actual": round(actual, 2),
            "status": "pass" if abs(expected - actual) <= tol else "FAIL"}


def reconcile(facts: dict[str, Any], notes: list[tuple[NoteChunk, Tier0Result]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    if ms := facts.get("medical_specials"):
        for col in ("billed", "allowed", "paid"):
            checks.append(_check(f"medical_specials.{col}_sum", ms[f"{col}_total"], sum(r[col] for r in ms["rows"])))

    if wc := facts.get("wage_claim"):
        if wc.get("claimed_items"):
            checks.append(_check("wage_claim.claimed_sum", wc["claimed_total"], sum(wc["claimed_items"])))
        if wc.get("audited_items"):
            checks.append(_check("wage_claim.audited_sum", wc["audited_total"], sum(wc["audited_items"])))

    if sd := facts.get("settlement_distribution"):
        gross, net = sd.get("Gross BI Settlement"), sd.get("Net Recovery to Claimant")
        deductions = [v for k, v in sd.items() if isinstance(v, float) and v < 0]
        if gross is not None and net is not None:
            checks.append(_check("settlement.net_equals_gross_minus_deductions", net, gross + sum(deductions)))

    if rl := facts.get("reserve_ledger"):
        lines = [r for r in rl if not r["line"].startswith("Total")]
        total = next((r for r in rl if r["line"].startswith("Total")), None)
        for r in lines:
            if r["redundancy"] or r["peak"] != r["final_paid"]:
                checks.append(_check(f"reserve_ledger.{r['line']}.redundancy", r["peak"] - r["final_paid"], r["redundancy"]))
        if total:
            for col in ("initial", "peak", "final_paid"):
                checks.append(_check(f"reserve_ledger.total_{col}", total[col], sum(r[col] for r in lines)))

    # Reserve trail continuity: each "increased from $X" must equal the last known reserve.
    last_bi: Optional[float] = None
    for chunk, t0 in notes:
        ent = t0.entities
        if init := ent.get("reserve_initial", {}).get("BI"):
            last_bi = init
        for ch in ent.get("reserve_changes", []):
            if last_bi is not None:
                c = _check(f"reserve_trail.{chunk.note_id}.prior_BI_reserve", last_bi, ch["from"])
                if c["status"] == "FAIL":
                    c["detail"] = "Reserve change references a starting value with no supporting note (undocumented reserve change)."
                checks.append(c)
            last_bi = ch["to"]

    # Authority vs. action: authorized opening offer vs. first carrier offer actually extended.
    nego = facts.get("negotiation", [])
    first_offer = next((n for n in nego if n["party"] == "Carrier"), None)
    for chunk, _ in notes:
        if m := re.search(r"Authorized opening counteroffer at (\$[\d,]+)", chunk.text):
            if first_offer:
                c = _check(f"authority.{chunk.note_id}.opening_offer", money(m.group(1)), first_offer["amount"])
                if c["status"] == "FAIL":
                    c["detail"] = f"First carrier offer on {first_offer['date']} differs from the committee-authorized opening offer."
                checks.append(c)
    return checks
