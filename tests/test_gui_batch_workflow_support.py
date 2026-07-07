import json
import os
import tempfile
import unittest

from gui_qt.batch_workflow_support import (
    build_submit_cli_args,
    format_cost_estimate_facts,
    load_cost_estimate_facts_from_manifest,
    resolve_submit_max_cost,
)
from gui_qt.doctor_report import doctor_report_to_parsed, summarize_doctor_report
from gui_qt.translation_workflow import TranslationWorkflow


class GuiBatchWorkflowSupportTests(unittest.TestCase):
    def test_resolve_submit_max_cost(self):
        self.assertIsNone(resolve_submit_max_cost({}))
        self.assertIsNone(resolve_submit_max_cost({"batch": {"submit_max_cost": 0}}))
        self.assertEqual(
            resolve_submit_max_cost({"batch": {"submit_max_cost": 12.5}}),
            12.5,
        )

    def test_build_submit_cli_args_appends_max_cost(self):
        self.assertEqual(
            build_submit_cli_args("pkg/manifest.json"),
            ["submit", "pkg/manifest.json"],
        )
        self.assertEqual(
            build_submit_cli_args("pkg/manifest.json", 5),
            ["submit", "pkg/manifest.json", "--max-cost", "5"],
        )

    def test_format_cost_estimate_facts(self):
        facts = format_cost_estimate_facts(
            {
                "model": "gemini-3.1-flash-lite",
                "estimated_input_tokens": 1000,
                "estimated_output_tokens_max": 2000,
                "estimated_cost_min": 0.1,
                "estimated_cost_max": 0.9,
                "currency": "USD",
            }
        )
        self.assertIn("估算模型：gemini-3.1-flash-lite", facts)
        self.assertIn("估算成本：0.1000 至 0.9000 USD", facts)

    def test_load_cost_estimate_facts_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cost_estimate": {
                            "estimated_input_tokens": 10,
                            "estimated_output_tokens_max": 20,
                            "estimated_cost_min": 0.01,
                            "estimated_cost_max": 0.02,
                            "currency": "USD",
                        }
                    },
                    handle,
                )
            facts = load_cost_estimate_facts_from_manifest(manifest_path)
            self.assertIn("估算输入 token：10", facts)

    def test_translation_workflow_submit_uses_max_cost(self):
        workflow = TranslationWorkflow.start_new(submit_max_cost=3.5)
        workflow.manifest_path = r"C:\package\manifest.json"
        workflow._pending_steps[:] = ["submit", "status"]

        self.assertEqual(
            workflow.current_step().args,
            ["submit", r"C:\package\manifest.json", "--max-cost", "3.5"],
        )

    def test_doctor_report_shows_tl_subdir(self):
        parsed = doctor_report_to_parsed(
            {
                "base_dir": "C:/work",
                "tl_dir": "C:/work/game/tl/schinese",
                "tl_subdir": "game/tl/schinese",
                "tl_exists": True,
                "language": "schinese",
                "mode": "existing_tl_only",
                "counts": {},
                "warnings": [],
                "recommendations": [],
            }
        )
        summary = summarize_doctor_report(parsed, exit_code=0, api_key_count=1)
        self.assertTrue(any("TL 路径：game/tl/schinese" in fact for fact in summary.facts))


if __name__ == "__main__":
    unittest.main()