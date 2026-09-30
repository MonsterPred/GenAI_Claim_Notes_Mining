"""CLI.

    python -m claim_mining run "Personal Auto Bodily Injury Claim File.pdf"
    python -m claim_mining run file.pdf --no-llm          # Tiers 0-1 only, zero token cost
    python -m claim_mining export-finetune                # pseudo-labels -> SLM fine-tuning JSONL
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .distillation import export_finetune
from .pipeline import load_config, run


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="claim_mining", description="Tiered GenAI claim-note mining pipeline")
    p.add_argument("--config", default="config/pipeline.json")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Mine one claim-file PDF")
    r.add_argument("pdf")
    r.add_argument("--no-llm", action="store_true", help="Skip Tier 2 (no API calls)")
    r.add_argument("--out", help="Output directory (default from config)")

    e = sub.add_parser("export-finetune", help="Export pseudo-labels as chat-format fine-tuning JSONL")
    e.add_argument("--out", default="data/finetune_tier1.jsonl")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.cmd == "export-finetune":
        n = export_finetune(Path(cfg["pseudo_label_path"]), Path(args.out))
        print(f"Wrote {n} examples to {args.out}")
        return 0

    out = run(args.pdf, cfg, use_llm=not args.no_llm, out_dir=args.out)
    rep = out.report
    print(f"Claim {out.claim_id}: {rep['notes_total']} notes -> routing {rep['routing']}")
    print(f"Reconciliation: {rep['reconciliation']}; distillation: {rep['distillation']['reduction_pct']}% smaller")
    if rep["llm_calls"]:
        t = rep["token_usage"]["totals"]
        print(f"LLM: {rep['llm_calls']} calls, in={t['input_tokens']} cache_w={t['cache_creation_input_tokens']} "
              f"cache_r={t['cache_read_input_tokens']} out={t['output_tokens']}, cost=${t['cost_usd']:.4f} "
              f"[{rep['tier2_billing']}]")
    for err in rep["errors"]:
        print(f"ERROR {err}", file=sys.stderr)
    print(f"Outputs: {out.out_dir}")
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
