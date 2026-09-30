# GenAI Claim Notes Mining

A tiered text-mining pipeline for insurance claim files (PDF). Each note is sent to the cheapest
tier that can handle it with confidence. The frontier LLM is called only for the notes that are
still ambiguous and for tasks that need reasoning across the whole file.

```
[Claim file PDF] -> ingest (normalize, segment into log entries + section narratives)
        |
   Tier 0  Regex / rules / dictionaries      -> decisive match (e.g. LOR, subrogation), entities,
        |                                       table parsing, arithmetic reconciliation
        | (no rule, or conflicting rules)
   Tier 1  Local NLP (TF-IDF centroid,       -> routine classification + sentiment polarity
        |   or a local SLM via Ollama)
        | (low confidence / small margin)
   Tier 2  Claude (structured JSON output)   -> escalated note labels + claim-level synthesis
                                                (chronology, disputes, root causes, causation)
```

## Cost-reduction strategies and where they live

| Strategy | Implementation |
|---|---|
| Cascading routing | [claim_mining/pipeline.py](claim_mining/pipeline.py): Tier 0 -> 1 -> 2 per note, with escalation reasons recorded |
| Context distillation and chunk filtering | [claim_mining/distill.py](claim_mining/distill.py): strips boilerplate, replaces tables/panels with Tier-0 JSON facts, dedupes, and keeps the top BM25 sentences per task query |
| Prompt caching | [claim_mining/tier2_llm.py](claim_mining/tier2_llm.py): one byte-stable system prompt (taxonomy, reason codes, few-shot) with `cache_control` |
| Constrained output | [claim_mining/models.py](claim_mining/models.py): Pydantic schemas -> structured outputs, with enums and word limits |
| Task distillation | [claim_mining/distillation.py](claim_mining/distillation.py): Tier-2 labels -> pseudo-labels that retrain Tier 1 on the next run, plus `export-finetune` for SLM fine-tuning |

Tier 0 also reconciles the file's numbers deterministically: table column sums, the settlement
net-to-claimant, reserve redundancy, reserve-trail continuity, and authorized offer vs. offer actually
extended. On the sample file it flags an undocumented BI reserve change ($15k -> $25k is never noted)
and an opening offer ($24,500) that differs from the committee's authorization ($22,500).

## Setup

```bash
pip install -r requirements.txt
```

Tier 2 has two backends (`tier2.backend` in [config/pipeline.json](config/pipeline.json)):

- **`claude_code`** (default): runs headless Claude Code (`claude -p`) on your claude.ai
  subscription login. It needs no API credit; usage counts against your plan's limits. The pipeline
  removes `ANTHROPIC_API_KEY` from the subprocess environment so the CLI uses the subscription. The
  model is the CLI's default unless `claude_code_model` is set. The cost shown is Claude Code's
  notional list-price figure, not a charge.
- **`api`**: Anthropic Messages API via the Python SDK (`ANTHROPIC_API_KEY` or `ant auth login`).
  This needs prepaid credit in the Claude Console and is billed separately from subscriptions.

## Usage

```bash
# Full pipeline (Tiers 0-2)
python -m claim_mining run "Personal Auto Bodily Injury Claim File.pdf"

# Tiers 0-1 only: zero token cost, no credentials needed
python -m claim_mining run "Personal Auto Bodily Injury Claim File.pdf" --no-llm

# Pseudo-labels -> chat-format JSONL for fine-tuning a 3B-8B model
python -m claim_mining export-finetune --out data/finetune_tier1.jsonl

# Tests (offline; Tier 2 runs through the real SDK with a mocked network)
python -m unittest discover -s tests -t .
```

Outputs go to `outputs/<claim_id>/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable summary: routing, per-note labels, reconciliation, negotiation trail, synthesis, token/cost ledger |
| `notes.jsonl` | One record per note with the Tier 0 flags/entities, Tier 1 scores, Tier 2 label and resolving tier |
| `facts.json` | Header fields, parsed tables and reconciliation checks |
| `synthesis.json` | Claim-level Tier 2 output (only when the LLM ran) |
| `distilled_context.txt` | The exact context sent for synthesis |
| `run_report.json` | Routing counts, distillation stats, per-call token usage and cost, timings |

## Configuration

- [config/pipeline.json](config/pipeline.json): Tier 1 backend (`centroid` or `ollama`), softmax temperature and
  escalation thresholds; distillation keep ratio; Tier 2 model, effort per task, batch size, and
  pricing used by the cost ledger.
- [config/taxonomy.json](config/taxonomy.json): the categories (with seed phrases for Tier 1), dispute types
  and reason codes. The Tier 2 prompt and schemas are built from the same taxonomy, so keep
  `Category` / `ReasonCode` in `models.py` in sync with it.

To use a local SLM for Tier 1, run Ollama (for example `ollama pull llama3.1:8b`) and set
`"backend": "ollama"`. If the SLM call fails, Tier 1 falls back to the centroid model.

## Extending to other claim files

Tier 0 title rules assume system-generated activity titles in the form
`MM/DD/YYYY HH:MM TZ | TITLE Author`. For a different claim system, adapt `LOG_HEADER_RE` in
[ingest.py](claim_mining/ingest.py) and the table parsers in `extract_tables()`. The rest of the pipeline
works on `NoteChunk`s and doesn't depend on the source layout.
