"""Data structures shared across tiers.

Internal records are plain dataclasses; the schemas the frontier LLM must emit
are Pydantic models so they can be passed straight to structured outputs
(constrained decoding -> short, machine-readable keys, no free-form prose).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Category = Literal[
    "FNOL_INTAKE",
    "COVERAGE_VERIFICATION",
    "LIABILITY_INVESTIGATION",
    "ATTORNEY_REPRESENTATION",
    "MEDICAL_TREATMENT",
    "MEDICAL_CAUSATION_REVIEW",
    "PRIOR_CLAIMS_ISO",
    "RESERVE_CHANGE",
    "PROPERTY_DAMAGE",
    "DAMAGES_AUDIT",
    "LIEN_SUBROGATION",
    "DEMAND_NEGOTIATION",
    "EXTRACONTRACTUAL_EXPOSURE",
    "INSURED_COMMUNICATION",
    "SETTLEMENT_RELEASE",
    "FILE_CLOSURE",
]
Sentiment = Literal["negative", "neutral", "positive"]
DisputeType = Literal["none", "liability", "causation", "damages", "coverage", "lien", "bad_faith"]
ReasonCode = Literal[
    "L-10", "L-14", "M-20", "M-22", "D-30", "D-32", "A-40", "X-50", "S-60", "R-70", "P-80", "N-00"
]


# --------------------------------------------------------------------------- #
# Internal records
# --------------------------------------------------------------------------- #
@dataclass
class NoteChunk:
    """One unit of analysis: a timestamped adjuster log entry or a section narrative."""

    note_id: str
    kind: Literal["log", "section"]
    section_no: int
    section_title: str
    page: int
    text: str
    date: Optional[str] = None
    time: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None


@dataclass
class Tier0Result:
    category: Optional[str]  # set only when exactly one decisive rule fired
    fired_rules: list[str] = field(default_factory=list)
    candidate_categories: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tier1Result:
    category: str
    confidence: float
    margin: float
    top_k: list[tuple[str, float]]
    sentiment: str
    sentiment_score: float
    backend: str


@dataclass
class NoteResult:
    chunk: NoteChunk
    tier0: Tier0Result
    tier1: Optional[Tier1Result] = None
    tier2: Optional[dict[str, Any]] = None
    resolved_by: str = "unresolved"
    category: Optional[str] = None
    sentiment: Optional[str] = None
    escalation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.tier1:
            d["tier1"]["top_k"] = [[c, round(s, 4)] for c, s in self.tier1.top_k]
        return d


@dataclass
class UsageRecord:
    call: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Tier 2 structured-output schemas
# --------------------------------------------------------------------------- #
class NoteLabel(BaseModel):
    note_id: str
    category: Category
    secondary: list[Category] = Field(description="Other applicable categories, at most 2.")
    sentiment: Sentiment
    severity: int = Field(description="1 (routine) to 5 (critical exposure).")
    dispute_type: DisputeType
    reason_code: ReasonCode
    rationale: str = Field(description="At most 20 words.")


class NoteLabelBatch(BaseModel):
    labels: list[NoteLabel]


class ChronoEvent(BaseModel):
    date: str = Field(description="MM/DD/YYYY")
    event: str = Field(description="At most 15 words.")
    note_ids: list[str]


class Dispute(BaseModel):
    dispute_type: DisputeType
    parties: list[str]
    position_summary: str = Field(description="Each party's position, at most 40 words.")
    evidence_note_ids: list[str]
    status: Literal["open", "resolved_by_settlement", "resolved_other", "unclear"]
    reason_code: ReasonCode


class RootCause(BaseModel):
    issue: str = Field(description="Claimant/member complaint or friction point, at most 12 words.")
    root_cause: str = Field(description="At most 25 words.")
    evidence_note_ids: list[str]


class ClaimSynthesis(BaseModel):
    claim_id: str
    summary: str = Field(description="Neutral file summary, at most 120 words.")
    chronology: list[ChronoEvent] = Field(description="Key milestones only, at most 15, in date order.")
    disputes: list[Dispute]
    complaint_root_causes: list[RootCause]
    liability_split_insured_pct: Optional[int]
    causation_finding: str = Field(description="At most 40 words.")
    outcome: Literal["open", "settled", "denied", "litigated", "closed_no_pay"]
    final_severity: int = Field(description="1 (routine) to 5 (critical exposure).")
    risk_flags: list[str] = Field(description="Short snake_case flags, at most 8.")
