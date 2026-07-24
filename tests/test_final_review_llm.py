# -*- coding: utf-8 -*-
"""Tests for final-review LLM execution and resume (#255 PR B)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import final_review as fr
import final_review_llm as frl


def _unit(
    unit_id: str = "fr-u1",
    *,
    status: str = fr.STATUS_PENDING,
    items: list[dict] | None = None,
    context_digest: str = "ctx-1",
    model: str = "test-model",
) -> dict:
    if items is None:
        items = [
            {
                "id": "id-a",
                "identity_v2": "id-a",
                "file_rel_path": "a.rpy",
                "source": "Hello",
                "current_translation": "你好",
            },
            {
                "id": "id-b",
                "identity_v2": "id-b",
                "file_rel_path": "a.rpy",
                "source": "World",
                "current_translation": "世界",
            },
        ]
    items_digest = fr.digest_translation_items(items)
    item_ids = [i["id"] for i in items]
    input_digest = fr.compute_unit_input_digest(
        item_ids=item_ids,
        items_digest=items_digest,
        context_digest=context_digest,
        model=model,
        prompt_schema_version=fr.PROMPT_SCHEMA_VERSION,
        chunk_index=1,
        file_rel_path="a.rpy",
    )
    return {
        "unit_id": unit_id,
        "status": status,
        "file_rel_path": "a.rpy",
        "chunk_index": 1,
        "item_ids": item_ids,
        "item_count": len(items),
        "items": items,
        "items_digest": items_digest,
        "input_digest": input_digest,
        "context_digest": context_digest,
        "snapshot_digest": "snap-1",
        "model": model,
        "prompt_schema_version": fr.PROMPT_SCHEMA_VERSION,
        "error": "",
        "finding_count": 0,
        "completed_at": "",
    }


class PromptAndRequestTests(unittest.TestCase):
    def test_batch_request_key_is_unit_id(self):
        unit = _unit("fr-abc")
        row = frl.build_batch_request(unit, model="gemini-test")
        self.assertEqual(row["key"], "fr-abc")
        self.assertIn("request", row)
        self.assertEqual(
            row["request"]["generation_config"]["response_mime_type"],
            "application/json",
        )
        schema = row["request"]["generation_config"]["response_json_schema"]
        self.assertIn("findings", schema["properties"])


class ParseFindingsTests(unittest.TestCase):
    def test_parse_findings_and_strip_applied_claim(self):
        unit = _unit()
        text = json.dumps(
            {
                "findings": [
                    {
                        "item_id": "id-a",
                        "finding_type": "mistranslation",
                        "severity": "high",
                        "reason": "tone",
                        "suggested_revision": "您好",
                        "revision_state": "applied",
                    }
                ]
            },
            ensure_ascii=False,
        )
        findings, error = frl.parse_unit_findings(text, unit, model="m")
        self.assertEqual(error, "")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["finding_type"], "mistranslation")
        self.assertEqual(findings[0]["revision_state"], fr.REVISION_STATE_NONE)
        self.assertEqual(findings[0]["identity_v2"], "id-a")

    def test_empty_findings_is_success(self):
        unit = _unit()
        findings, error = frl.parse_unit_findings('{"findings":[]}', unit)
        self.assertEqual(error, "")
        self.assertEqual(findings, [])

    def test_malformed_json_is_error_not_zero_findings(self):
        unit = _unit()
        findings, error = frl.parse_unit_findings("not-json{{{", unit)
        self.assertTrue(error.startswith("failed_to_parse_model_json"))
        self.assertEqual(findings, [])

    def test_missing_findings_key_is_error(self):
        unit = _unit()
        findings, error = frl.parse_unit_findings('{"issues_count": 0}', unit)
        self.assertIn("missing findings", error)
        self.assertEqual(findings, [])

    def test_apply_unit_result_failed_not_done(self):
        unit = _unit()
        updated, findings = frl.apply_unit_result(unit, response_text="{{{")
        self.assertEqual(updated["status"], fr.STATUS_FAILED)
        self.assertEqual(findings, [])
        self.assertNotEqual(updated["status"], fr.STATUS_DONE)
        fr.assert_failure_not_done(updated)

    def test_apply_unit_result_empty_findings_done(self):
        unit = _unit()
        updated, findings = frl.apply_unit_result(unit, response_text='{"findings":[]}')
        self.assertEqual(updated["status"], fr.STATUS_DONE)
        self.assertEqual(updated["finding_count"], 0)
        self.assertEqual(findings, [])


class ResumePlanTests(unittest.TestCase):
    def test_skip_done_same_digest(self):
        unit = fr.mark_unit_done(_unit(), finding_count=0)
        plan = frl.plan_units_for_run([unit], force=False, live_context_digest="ctx-1")
        self.assertEqual(plan["skip_count"], 1)
        self.assertEqual(plan["run_count"], 0)

    def test_force_reruns_done(self):
        unit = fr.mark_unit_done(_unit(), finding_count=0)
        plan = frl.plan_units_for_run([unit], force=True, live_context_digest="ctx-1")
        self.assertEqual(plan["run_count"], 1)
        self.assertEqual(plan["skip_count"], 0)
        self.assertEqual(plan["to_run"][0]["status"], fr.STATUS_PENDING)

    def test_failed_is_requed(self):
        unit = fr.mark_unit_failed(_unit(), "parse_error")
        plan = frl.plan_units_for_run([unit], force=False, live_context_digest="ctx-1")
        self.assertEqual(plan["run_count"], 1)

    def test_context_change_stales_done(self):
        unit = fr.mark_unit_done(_unit(context_digest="ctx-1"), finding_count=0)
        plan = frl.plan_units_for_run(
            [unit], force=False, live_context_digest="ctx-CHANGED"
        )
        self.assertEqual(plan["run_count"], 1)
        # Recomputed live digest differs → queued
        self.assertEqual(plan["to_run"][0]["status"], fr.STATUS_PENDING)


class IngestRowsTests(unittest.TestCase):
    def test_ingest_mixed_done_and_failed(self):
        u1 = _unit("u1")
        u2 = _unit("u2")
        rows = [
            {
                "key": "u1",
                "response_text": json.dumps(
                    {
                        "findings": [
                            {
                                "item_id": "id-a",
                                "finding_type": "terminology",
                                "severity": "medium",
                                "reason": "term",
                            }
                        ]
                    }
                ),
            },
            {"key": "u2", "response_text": "not-json"},
        ]
        result = frl.ingest_result_rows([u1, u2], rows, model="m")
        by_id = {u["unit_id"]: u for u in result["units"]}
        self.assertEqual(by_id["u1"]["status"], fr.STATUS_DONE)
        self.assertEqual(by_id["u2"]["status"], fr.STATUS_FAILED)
        self.assertEqual(result["summary"]["done_units"], 1)
        self.assertEqual(result["summary"]["failed_units"], 1)
        self.assertEqual(result["summary"]["finding_count"], 1)
        self.assertEqual(result["summary"]["campaign_status"], fr.STATUS_FAILED)

    def test_missing_result_for_running_is_failed(self):
        u1 = dict(_unit("u1"), status=fr.STATUS_RUNNING)
        result = frl.ingest_result_rows([u1], [], model="m")
        self.assertEqual(result["units"][0]["status"], fr.STATUS_FAILED)
        self.assertIn("missing_result_row", result["units"][0]["error"])


class PackageResumeIngestTests(unittest.TestCase):
    def _write_package(self, tmp: str, units: list[dict], findings: list | None = None):
        items = []
        for unit in units:
            items.extend(unit.get("items") or [])
        readiness = fr.evaluate_readiness(
            pending_task_count=0, review_item_count=max(1, len(items))
        )
        snap = fr.build_context_snapshot(translation_items=items or [{"id": "x", "source": "s", "current_translation": "t"}])
        # Align unit digests with snap context when possible
        for unit in units:
            unit["context_digest"] = snap["context_digest"]
            unit["input_digest"] = fr.compute_unit_input_digest(
                item_ids=unit["item_ids"],
                items_digest=unit["items_digest"],
                context_digest=snap["context_digest"],
                model=unit.get("model") or "",
                prompt_schema_version=unit.get("prompt_schema_version") or fr.PROMPT_SCHEMA_VERSION,
                chunk_index=int(unit.get("chunk_index") or 0),
                file_rel_path=str(unit.get("file_rel_path") or ""),
            )
        package_dir = os.path.join(tmp, "fr_pkg")
        os.makedirs(package_dir, exist_ok=True)
        manifest = fr.build_campaign_manifest(
            package_dir=package_dir,
            display_name="test",
            snapshot=snap,
            units=units,
            readiness=readiness,
            model="test-model",
        )
        fr.write_campaign_package(
            package_dir,
            manifest=manifest,
            snapshot=snap,
            units=units,
            findings=findings or [],
        )
        return package_dir, snap

    def test_resume_skips_done_writes_only_pending_requests(self):
        done = fr.mark_unit_done(_unit("done-u", context_digest="will-replace"), finding_count=0)
        pending = _unit("pend-u", context_digest="will-replace")
        with tempfile.TemporaryDirectory() as tmp:
            package_dir, snap = self._write_package(tmp, [done, pending])
            result = frl.prepare_resume_requests(
                package_dir,
                force=False,
                live_context_digest=snap["context_digest"],
                model="test-model",
            )
            self.assertEqual(result["run_count"], 1)
            self.assertEqual(result["skip_count"], 1)
            self.assertEqual(result["to_run_unit_ids"], ["pend-u"])
            requests_path = Path(package_dir) / fr.REQUESTS_JSONL_FILENAME
            rows = [
                json.loads(line)
                for line in requests_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([r["key"] for r in rows], ["pend-u"])

    def test_force_resume_all(self):
        done = fr.mark_unit_done(_unit("done-u"), finding_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            package_dir, snap = self._write_package(tmp, [done])
            result = frl.prepare_resume_requests(
                package_dir,
                force=True,
                live_context_digest=snap["context_digest"],
            )
            self.assertEqual(result["run_count"], 1)
            self.assertEqual(result["skip_count"], 0)

    def test_ingest_package_persists_findings(self):
        unit = _unit("u1")
        with tempfile.TemporaryDirectory() as tmp:
            package_dir, _snap = self._write_package(tmp, [unit])
            result_path = Path(package_dir) / "results.jsonl"
            result_path.write_text(
                json.dumps(
                    {
                        "key": "u1",
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "text": json.dumps(
                                                    {
                                                        "findings": [
                                                            {
                                                                "item_id": "id-a",
                                                                "finding_type": "omission",
                                                                "severity": "high",
                                                                "reason": "missing clause",
                                                            }
                                                        ]
                                                    }
                                                )
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            out = frl.ingest_results_into_package(
                package_dir,
                result_path=str(result_path),
                model="m",
            )
            self.assertEqual(out["summary"]["done_units"], 1)
            self.assertEqual(out["summary"]["finding_count"], 1)
            loaded = fr.load_campaign_package(package_dir)
            self.assertEqual(loaded["units"][0]["status"], fr.STATUS_DONE)
            self.assertEqual(len(loaded["findings"]), 1)
            self.assertEqual(loaded["findings"][0]["finding_type"], "omission")

    def test_sync_generate_helper_zero_model_calls_on_resume_skip(self):
        unit = fr.mark_unit_done(_unit("u1"), finding_count=0)
        calls = {"n": 0}

        def gen(_system: str, _user: str) -> str:
            calls["n"] += 1
            return '{"findings":[]}'

        # force=False and done → no generate calls
        plan = frl.plan_units_for_run([unit], force=False, live_context_digest="ctx-1")
        self.assertEqual(plan["run_count"], 0)
        result = frl.run_units_with_generate(
            [unit], gen, force=False, live_context_digest="ctx-1"
        )
        self.assertEqual(calls["n"], 0)
        # units stay done
        self.assertEqual(result["units"][0]["status"], fr.STATUS_DONE)

        result2 = frl.run_units_with_generate(
            [unit], gen, force=True, live_context_digest="ctx-1"
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(result2["units"][0]["status"], fr.STATUS_DONE)


class DiagnosticsCommandTests(unittest.TestCase):
    def test_diagnostics_lists_resume_and_ingest(self):
        from gui_qt.diagnostics_context import build_cli_commands

        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\jobs\manifest.json",
            manifest={"mode": "final_review", "job_name": ""},
        )
        labels = [c.label for c in commands]
        self.assertIn("最终审校·续跑 requests", labels)
        self.assertIn("最终审校·摄入结果", labels)
        text = "\n".join(c.command for c in commands)
        self.assertIn("final-review-resume", text)
        self.assertIn("final-review-ingest-results", text)


if __name__ == "__main__":
    unittest.main()
