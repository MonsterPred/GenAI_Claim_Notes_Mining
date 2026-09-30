"""PDF -> normalized text -> header fields + note chunks.

The claim file is a rendered report: a header block, numbered sections, and
timestamped adjuster log entries of the form

    05/13/2026 08:30 CST | INITIAL ASSIGNMENT & TRIAGE M. Davies (Adjuster)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from .models import NoteChunk

FOOTER_RE = re.compile(r"^Claim File: .+ Page \d+ of \d+$")
SECTION_RE = re.compile(r"^(?P<no>\d{1,2})\. (?P<title>[A-Z][^\n]{3,})$")
# Author must contain a "X. Lastname" token with lowercase letters, so an
# all-caps name inside the title, e.g. "(DR. H. LANGFORD, MD)", stays in the title.
LOG_HEADER_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2}/\d{4}) (?P<time>\d{2}:\d{2}) (?P<tz>[A-Z]{2,4}) \| "
    r"(?P<title>[A-Z0-9 &/\-().,'#:]+?) (?P<author>[A-Z]\. [A-Z][a-z]+.*)$"
)
HEADER_FIELDS = {
    "claim_id": r"Claim #:\s*(\S+)",
    "policy_id": r"Policy #:\s*(\S+)",
    "adjuster": r"Adjuster:\s*(.+)",
    "insured_name": r"Insured Name:\s*(.+)",
    "policy_period": r"Policy Period:\s*(.+)",
    "policy_limits": r"Policy Limits:\s*(.+)",
    "insured_vehicle": r"Insured Vehicle:\s*(.+)",
    "date_of_loss": r"Date of Loss \(DOL\):\s*(.+)",
    "loss_location": r"Loss Location:\s*(.+)",
    "claimant_name": r"Claimant Name:\s*(.+)",
    "claimant_attorney": r"Claimant Attorney:\s*(.+)",
}


@dataclass
class ClaimDocument:
    source: str
    page_count: int
    raw_text: str  # normalized full text, footers included (for baseline measurement)
    header: dict[str, str]
    sections: dict[int, str]  # section no -> title
    section_text: dict[int, str]  # section no -> full collapsed text (tables included)
    chunks: list[NoteChunk] = field(default_factory=list)


def normalize(text: str) -> str:
    # NFKC folds ligatures (ﬁ, ﬂ, ﬃ) back into plain letters.
    text = unicodedata.normalize("NFKC", text)
    return text.replace(" ", " ").replace("–", "-").replace("—", "-")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_claim_pdf(path: str | Path) -> ClaimDocument:
    reader = PdfReader(str(path))
    lines: list[tuple[int, str]] = []
    raw_pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        raw_pages.append(text)
        for line in text.splitlines():
            line = line.strip()
            if line and not FOOTER_RE.match(line):
                lines.append((page_no, line))

    header_text = "\n".join(l for _, l in lines[:40])
    header = {}
    for key, pattern in HEADER_FIELDS.items():
        m = re.search(pattern, header_text)
        if m:
            header[key] = m.group(1).strip()

    doc = ClaimDocument(
        source=str(path),
        page_count=len(reader.pages),
        raw_text="\n".join(raw_pages),
        header=header,
        sections={},
        section_text={},
    )
    _segment(doc, lines)
    return doc


def _segment(doc: ClaimDocument, lines: list[tuple[int, str]]) -> None:
    section_no, section_title = 0, "Header"
    buf: list[str] = []
    buf_page = 1
    current: dict | None = None  # open log entry
    section_lines: dict[int, list[str]] = {0: []}
    log_count = 0

    def flush_section_narrative() -> None:
        nonlocal buf
        text = collapse(" ".join(buf))
        if text and section_no > 0:
            doc.chunks.append(
                NoteChunk(
                    note_id=f"S{section_no:02d}",
                    kind="section",
                    section_no=section_no,
                    section_title=section_title,
                    page=buf_page,
                    text=text,
                )
            )
        buf = []

    def flush_log() -> None:
        nonlocal current
        if current:
            current["text"] = collapse(" ".join(current["text"]))
            doc.chunks.append(NoteChunk(**current))
            current = None

    for page, line in lines:
        sec = SECTION_RE.match(line)
        if sec and int(sec.group("no")) == section_no + 1:
            flush_log()
            flush_section_narrative()
            section_no, section_title = int(sec.group("no")), sec.group("title").strip()
            doc.sections[section_no] = section_title
            section_lines[section_no] = []
            buf_page = page
            continue

        section_lines[section_no].append(line)
        log = LOG_HEADER_RE.match(line)
        if log:
            flush_log()
            log_count += 1
            current = dict(
                note_id=f"N{log_count:02d}",
                kind="log",
                section_no=section_no,
                section_title=section_title,
                page=page,
                date=log.group("date"),
                time=f"{log.group('time')} {log.group('tz')}",
                title=log.group("title").strip(),
                author=log.group("author").strip(),
                text=[],
            )
        elif current is not None:
            current["text"].append(line)
        else:
            if not buf:
                buf_page = page
            buf.append(line)

    flush_log()
    flush_section_narrative()
    doc.section_text = {no: collapse(" ".join(ls)) for no, ls in section_lines.items()}
    doc.chunks.sort(key=lambda c: (c.section_no, c.kind != "section", c.note_id))
