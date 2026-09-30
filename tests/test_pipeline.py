"""Offline tests. Tier 2 runs through the real Anthropic SDK request/parse path with
the network replaced by a mock transport, so no API credits are needed.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anthropic
import httpx2

from claim_mining import pipeline, tier0_rules
from claim_mining.distill import distill, split_sentences
from claim_mining.ingest import load_claim_pdf
from claim_mining.tier1_nlp import sentiment
from claim_mining.tier2_llm import Tier2Client

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "Personal Auto Bodily Injury Claim File.pdf"


def load_cfg() -> dict:
    cfg = json.loads((ROOT / "config" / "pipeline.json").read_text())
    cfg["taxonomy_path"] = str(ROOT / cfg["taxonomy_path"])
    return cfg


class TestTiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = load_claim_pdf(PDF)
        cls.notes = {c.note_id: c for c in cls.doc.chunks}

    def test_ingest_segments_all_log_entries(self):
        logs = [c for c in self.doc.chunks if c.kind == "log"]
        self.assertEqual(len(logs), 26)
        self.assertEqual(self.doc.header["claim_id"], "PA-2026-883491-BI")
        n11 = self.notes["N11"]
        self.assertEqual(n11.title, "PEER REVIEW CONSULTATION (DR. H. LANGFORD, MD)")
        self.assertEqual(n11.author, "M. Davies (Adjuster)")
        self.assertNotIn("ﬁ", self.doc.raw_text)  # ligatures normalized

    def test_tier0_decisive_and_ambiguous(self):
        lor = tier0_rules.analyze_note(self.notes["N05"])
        self.assertEqual(lor.category, "ATTORNEY_REPRESENTATION")
        mixed = tier0_rules.analyze_note(self.notes["N08"])
        self.assertIsNone(mixed.category)
        self.assertEqual(set(mixed.candidate_categories), {"MEDICAL_TREATMENT", "RESERVE_CHANGE"})
        self.assertEqual(mixed.entities["reserve_changes"], [{"from": 25000.0, "to": 65000.0}])

    def test_tier0_tables_and_reconciliation(self):
        facts = tier0_rules.extract_tables(self.doc)
        self.assertEqual(facts["settlement_distribution"]["Net Recovery to Claimant"], 17416.67)
        self.assertEqual([n["amount"] for n in facts["negotiation"]][:2], [225000.0, 24500.0])
        self.assertEqual(facts["reconstruction"]["Comparative Fault Est."], "60% Insured / 40% Claimant")
        checks = tier0_rules.reconcile(facts, [(c, tier0_rules.analyze_note(c)) for c in self.doc.chunks])
        failed = {c["check"] for c in checks if c["status"] == "FAIL"}
        self.assertEqual(failed, {"reserve_trail.N08.prior_BI_reserve", "authority.N17.opening_offer"})

    def test_sentiment_negation(self):
        self.assertEqual(sentiment("Claimant disputes liability and denies fault.")[0], "negative")
        self.assertEqual(sentiment("Negative acute fracture; no injury claimed.")[0], "positive")

    def test_distillation(self):
        ctx = distill(self.doc, load_cfg()["distill"])
        self.assertGreater(ctx.stats["reduction_pct"], 50)
        for c in self.doc.chunks:  # every log entry keeps its timeline anchor
            if c.kind == "log":
                self.assertIn(f"[{c.note_id} {c.date}", ctx.text)
        self.assertEqual(split_sentences("Seen by Dr. Chen today. Follow-up set."),
                         ["Seen by Dr. Chen today.", "Follow-up set."])


# --------------------------------------------------------------------------- #
# Tier 2 through the real SDK with a mocked network
# --------------------------------------------------------------------------- #
def _fake_api(captured: list):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        captured.append({"headers": dict(request.headers), "body": body})
        props = body["output_config"]["format"]["schema"].get("properties", {})
        prompt = body["messages"][0]["content"]
        if "labels" in props:
            ids = re.findall(r'<note id="(\w+)"', prompt)
            payload = {"labels": [{
                "note_id": i, "category": "DAMAGES_AUDIT", "secondary": [], "sentiment": "neutral",
                "severity": 2, "dispute_type": "damages", "reason_code": "D-30", "rationale": "test",
            } for i in ids]}
        else:
            payload = {
                "claim_id": "PA-2026-883491-BI", "summary": "test", "chronology": [], "disputes": [],
                "complaint_root_causes": [], "liability_split_insured_pct": 60, "causation_finding": "test",
                "outcome": "settled", "final_severity": 3, "risk_flags": ["lop_treatment"],
            }
        cached = len(captured) > 1
        return httpx2.Response(200, json={
            "id": f"msg_{len(captured)}", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 300,
                      "cache_creation_input_tokens": 0 if cached else 1500,
                      "cache_read_input_tokens": 1500 if cached else 0},
        })
    return handler


class TestTier2Offline(unittest.TestCase):
    def test_end_to_end_with_mocked_api(self):
        captured: list = []
        client = anthropic.Anthropic(api_key="test-key",
                                     http_client=httpx2.Client(transport=httpx2.MockTransport(_fake_api(captured))))
        cfg = load_cfg()
        cfg["tier2"]["backend"] = "api"
        with tempfile.TemporaryDirectory() as tmp:
            cfg["pseudo_label_path"] = str(Path(tmp) / "pseudo.jsonl")
            make = lambda taxonomy, c: Tier2Client(taxonomy, c, client=client)  # noqa: E731
            with mock.patch.object(pipeline, "_make_tier2", make):
                out = pipeline.run(PDF, cfg, use_llm=True, out_dir=tmp)

                # Request shape: caching, constrained output, effort, server-side fallback.
                first = captured[0]
                self.assertIn("server-side-fallback-2026-07-01", first["headers"]["anthropic-beta"])
                self.assertEqual(first["body"]["fallbacks"], "default")
                self.assertEqual(first["body"]["system"][0]["cache_control"], {"type": "ephemeral"})
                self.assertEqual(first["body"]["output_config"]["format"]["type"], "json_schema")
                self.assertEqual(first["body"]["output_config"]["effort"], "low")
                self.assertEqual(captured[-1]["body"]["output_config"]["effort"], "medium")
                self.assertEqual(captured[0]["body"]["system"], captured[-1]["body"]["system"])

                # Routing and outputs.
                self.assertEqual(out.report["routing"].get("tier2"), 4)
                self.assertEqual(out.synthesis["outcome"], "settled")
                self.assertEqual(out.report["token_usage"]["totals"]["cache_read_input_tokens"], 1500)
                for name in ("notes.jsonl", "facts.json", "synthesis.json", "report.md", "run_report.json"):
                    self.assertTrue((out.out_dir / name).exists(), name)

                # Task distillation: labels stored, then loaded by Tier 1 on the next run.
                lines = Path(cfg["pseudo_label_path"]).read_text().splitlines()
                self.assertEqual(len(lines), 4)
                again = pipeline.run(PDF, cfg, use_llm=True, out_dir=tmp)
                self.assertEqual(again.report["tier1_pseudo_labels_loaded"], 4)
                rows = [json.loads(l) for l in Path(cfg["pseudo_label_path"]).read_text().splitlines()]
                self.assertEqual(len(rows), 4 + again.report["pseudo_labels_added"])
                self.assertEqual(len({r["key"] for r in rows}), len(rows))  # no duplicate notes


class TestClaudeCodeBackend(unittest.TestCase):
    """Subscription backend: `claude -p` subprocess mocked, output format as emitted by the CLI."""

    def test_claude_code_backend(self):
        calls: list = []

        def fake_run(cmd, input, env, **kw):
            calls.append({"cmd": cmd, "env": env})
            schema = json.loads(cmd[cmd.index("--json-schema") + 1])
            if "labels" in schema["properties"]:
                ids = re.findall(r'<note id="(\w+)"', input)
                payload = {"labels": [{
                    "note_id": i, "category": "FILE_CLOSURE", "secondary": [], "sentiment": "neutral",
                    "severity": 1, "dispute_type": "none", "reason_code": "N-00", "rationale": "test",
                } for i in ids]}
            else:
                payload = {
                    "claim_id": "X", "summary": "s", "chronology": [], "disputes": [], "complaint_root_causes": [],
                    "liability_split_insured_pct": None, "causation_finding": "c", "outcome": "settled",
                    "final_severity": 2, "risk_flags": [],
                }
            out = {"is_error": False, "structured_output": payload, "total_cost_usd": 0.05,
                   "terminal_reason": "completed",
                   "modelUsage": {"claude-opus-5": {"inputTokens": 10, "outputTokens": 400,
                                                    "cacheCreationInputTokens": 2000, "cacheReadInputTokens": 0},
                                  "claude-haiku-4-5-20251001": {"inputTokens": 900, "outputTokens": 15,
                                                                "cacheCreationInputTokens": 0,
                                                                "cacheReadInputTokens": 0}}}
            return mock.Mock(stdout=json.dumps(out), stderr="", returncode=0)

        cfg = load_cfg()
        cfg["tier2"]["backend"] = "claude_code"
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-should-not-leak"}), \
                mock.patch("claim_mining.tier2_llm.shutil.which", return_value="claude"), \
                mock.patch("claim_mining.tier2_llm.subprocess.run", side_effect=fake_run):
            cfg["pseudo_label_path"] = str(Path(tmp) / "pseudo.jsonl")
            out = pipeline.run(PDF, cfg, use_llm=True, out_dir=tmp)

        self.assertEqual(len(calls), 2)  # one label batch + one synthesis
        for c in calls:
            self.assertNotIn("ANTHROPIC_API_KEY", c["env"])  # forces subscription auth
            self.assertIn("-p", c["cmd"])
            self.assertEqual(c["cmd"][c["cmd"].index("--tools") + 1], "")
        self.assertEqual(out.report["routing"].get("tier2"), 4)
        self.assertEqual(out.report["tier2_backend"], "claude_code")
        self.assertEqual(out.report["token_usage"]["calls"][0]["model"], "claude-opus-5")
        self.assertEqual(out.report["token_usage"]["totals"]["output_tokens"], 830)
        self.assertEqual(out.synthesis["outcome"], "settled")


if __name__ == "__main__":
    unittest.main()
