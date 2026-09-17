"""Preflight scan/plan tests (#348 P3 increment A)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
import model_profile
import model_profiles_editor as editor
import model_routing_reader as reader
import sync_run_store
import sync_translation_preview
import translation_plan
import translator_runtime as runtime
from atomic_io import file_sha256, sha256_text
from sync_run_contracts import build_run_id


def routing_section(strategy: str = "sync") -> dict:
    section = editor.add_provider(
        editor.empty_section(),
        label="Gemini",
        adapter="gemini",
        provider="gemini",
        credential_kind="api_keys_json",
        credential_name="api_keys",
        credential_env_name="GEMINI_API_KEY",
    )
    section = editor.add_profile(
        section,
        label="Gemini Main",
        provider_id="gemini",
        model="gemini-3.5-flash",
    )
    return editor.set_defaults(
        section,
        primary_profile_id="gemini-main",
        execution_strategy=strategy,
    )


def coverage_report(
    *,
    status: str = "ready",
    counts: dict | None = None,
    invariant_errors=(),
    source_changed: bool = False,
):
    """Minimal coverage report shape consumed by the preflight gate."""

    return SimpleNamespace(
        coverage_status=status,
        classification_counts=dict(counts or {}),
        invariant_errors=tuple(invariant_errors),
        source_changed_during_scan=source_changed,
        coverage_digest="coverage-digest",
    )


def fake_context(
    section: dict,
    *,
    plan=None,
    requests=None,
    coverage_report_value=None,
    coverage_inventory=None,
    plan_fingerprint="plan-fingerprint-1",
):
    plan = plan or reader.read_routing_plan({"model_routing": section})
    chunks = [SimpleNamespace(chunk_id="chunk-1"), SimpleNamespace(chunk_id="chunk-2")]
    if requests is None:
        requests = [
            SimpleNamespace(
                expected_ids=["a", "b"],
                system_instruction="system prompt",
                user_prompt="user prompt",
            ),
            SimpleNamespace(
                expected_ids=["c"],
                system_instruction="system prompt",
                user_prompt="user prompt",
            ),
        ]
    identity = SimpleNamespace(
        to_dict=lambda: {
            "engine": "renpy",
            "adapter_version": "1",
            "project_identity_digest": "digest",
            "source_snapshot_fingerprint": "fingerprint",
            "file_digests": {"a.rpy": "sha"},
        }
    )
    documents = [SimpleNamespace(file_rel_path="a.rpy")]
    return SimpleNamespace(
        plan_build=SimpleNamespace(
            plan=SimpleNamespace(
                chunks=chunks,
                source_identity=identity,
                plan_fingerprint=plan_fingerprint,
            ),
            requests=list(requests),
        ),
        routing_plan=plan,
        adapter_snapshot=SimpleNamespace(
            project=SimpleNamespace(source_documents=documents),
            report=coverage_report_value,
            inventory=coverage_inventory,
        ),
        pending_jobs=[{"file_rel_path": "a.rpy", "tasks": [{}]}],
    )


class PreflightCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = routing_section()
        self.parser = batch.build_arg_parser()
        self._log_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._log_tmp.cleanup)
        self.legacy_patches = (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", self.section),
            mock.patch.object(runtime, "BASE_DIR", "C:/game/work"),
            mock.patch.object(runtime, "TL_DIR", "C:/game/work/game/tl/schinese"),
            mock.patch.object(runtime, "GENERATION_TARGET_LANGUAGE", "schinese"),
            mock.patch.object(runtime, "LOG_DIR", self._log_tmp.name),
            mock.patch.object(runtime, "MAX_ITEMS", 60),
            mock.patch.object(runtime, "MAX_CHARS", 18000),
            mock.patch.object(runtime, "SYNC_RAG_ENABLED", False),
            mock.patch.object(runtime, "SYNC_STORY_MEMORY_ENABLED", False),
            mock.patch.object(runtime, "SYNC_SOURCE_INDEX_ENABLED", False),
            mock.patch.object(
                runtime, "SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF", False
            ),
        )
        for patcher in self.legacy_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_payload_reports_counts_strategy_and_risks(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["strategy"], "sync")
        self.assertEqual(payload["plan_contract"], "shared_translation_plan")
        self.assertEqual(payload["profile"]["id"], "gemini-main")
        self.assertEqual(payload["counts"]["files_with_pending"], 1)
        self.assertEqual(payload["counts"]["pending_items"], 3)
        self.assertEqual(payload["counts"]["chunks"], 2)
        self.assertEqual(payload["chunk_policy"], {"max_items": 60, "max_chars": 18000})
        self.assertEqual(payload["source_snapshot"]["file_count"], 1)
        self.assertEqual(payload["credential_available"], True)

    def test_payload_includes_cost_coverage_and_quality_summaries(self) -> None:
        context = fake_context(
            self.section,
            coverage_report_value=coverage_report(
                status="ready", counts={"translatable": 3}
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            missing_cursor = str(Path(tmp) / "latest_manifest.txt")
            with mock.patch.object(batch, "LATEST_MANIFEST_FILE", missing_cursor):
                payload = self._run_preflight(context)

        self.assertEqual(payload["cost"]["status"], "known")
        self.assertEqual(payload["cost"]["strategy"], "sync")
        self.assertEqual(payload["cost"]["scope"], "current_plan")
        self.assertGreater(payload["cost"]["input_tokens"], 0)
        self.assertGreaterEqual(
            payload["cost"]["estimated_cost_max"],
            payload["cost"]["estimated_cost_min"],
        )
        self.assertEqual(payload["coverage"]["scope"], "current_scan")
        self.assertEqual(payload["coverage"]["completion"], "confirmed")
        self.assertEqual(
            payload["coverage"]["classification_counts"]["translatable"], 3
        )
        self.assertEqual(payload["quality_summary"]["status"], "not_available")
        self.assertEqual(
            payload["quality_summary"]["reason"],
            "durable_sync_run_missing",
        )

    def test_coverage_summary_keeps_live_status_and_gate_separate(self) -> None:
        context = fake_context(
            self.section,
            coverage_report_value=coverage_report(
                status="ready", counts={"translatable": 3}
            ),
            coverage_inventory=object(),
        )
        gate = SimpleNamespace(
            status="review_missing",
            confirmed=False,
            coverage_status="ready",
            review_status="missing",
            to_dict=lambda: {
                "status": "review_missing",
                "confirmed": False,
                "coverage_status": "ready",
                "review_status": "missing",
            },
        )
        with mock.patch.object(batch, "evaluate_coverage_gate", return_value=gate):
            payload = self._run_preflight(context)

        self.assertEqual(payload["coverage"]["status"], "ready")
        self.assertFalse(payload["coverage"]["confirmed"])
        self.assertEqual(payload["coverage"]["gate"]["status"], "review_missing")
        self.assertEqual(payload["coverage"]["review_status"], "missing")

        blocked_context = fake_context(
            self.section,
            coverage_report_value=coverage_report(status="block", counts={}),
        )
        blocked_payload = self._run_preflight(blocked_context)
        self.assertEqual(blocked_payload["coverage"]["status"], "block")

    def test_cost_summary_is_unknown_for_unpriced_model(self) -> None:
        section = routing_section()
        section["profiles"]["gemini-main"]["model"] = "unpriced-model-xyz"
        section["profiles"]["gemini-main"]["models"] = ["unpriced-model-xyz"]
        context = fake_context(section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", section),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["cost"]["status"], "unknown")
        self.assertEqual(payload["cost"]["strategy"], "sync")
        self.assertEqual(payload["cost"]["reason"], "pricing_unavailable")
        self.assertIsNone(payload["cost"]["estimated_cost_min"])

    def test_quality_summary_matches_project_and_plan_fingerprint(self) -> None:
        import hashlib

        def summarize(strategy="gemini_batch"):
            return batch.summarize_preflight_quality(
                plan_fingerprint="plan-fingerprint-1",
                base_dir=runtime.BASE_DIR,
                tl_dir=runtime.TL_DIR,
                strategy=strategy,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "quality_findings.jsonl"
            report_path.write_text(
                json.dumps({"reason_code": "dup", "severity": "medium"}) + "\n"
                + json.dumps({"reason_code": "missing", "severity": "high"}) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest = {
                "mode": batch.MANIFEST_MODE_TRANSLATION,
                "base_dir": runtime.BASE_DIR,
                "tl_dir": runtime.TL_DIR,
                "translation_plan": {"plan_fingerprint": "plan-fingerprint-1"},
                "last_quality_findings_path": str(report_path),
                "last_check_at": "2026-09-15T00:00:00",
                "last_check_summary": {"quality_findings_sha256": digest},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            cursor = root / "latest_manifest.txt"
            cursor.write_text(str(manifest_path), encoding="utf-8")

            with mock.patch.object(batch, "LATEST_MANIFEST_FILE", str(cursor)):
                summary = summarize()
                self.assertEqual(summary["status"], "available")
                self.assertEqual(summary["finding_count"], 2)
                self.assertEqual(summary["severity_counts"]["high"], 1)
                self.assertEqual(summary["severity_counts"]["medium"], 1)
                self.assertEqual(
                    summary["matched_by"], ["project", "plan_fingerprint"]
                )

                sync_summary = summarize(strategy="sync")
                self.assertEqual(sync_summary["status"], "not_available")
                self.assertEqual(
                    sync_summary["reason"],
                    "durable_sync_run_missing",
                )

                manifest["translation_plan"]["plan_fingerprint"] = "other-plan"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(summarize()["reason"], "plan_mismatch")

                manifest["translation_plan"]["plan_fingerprint"] = "plan-fingerprint-1"
                manifest["base_dir"] = "C:/other/project"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(summarize()["reason"], "project_mismatch")

                manifest["base_dir"] = runtime.BASE_DIR
                manifest["mode"] = "revision"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(summarize()["reason"], "mode_mismatch")

                manifest["mode"] = batch.MANIFEST_MODE_TRANSLATION
                manifest["last_quality_findings_path"] = str(root / "missing.jsonl")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(summarize()["reason"], "report_missing")

                manifest["last_quality_findings_path"] = str(report_path)
                manifest["last_check_summary"]["quality_findings_sha256"] = "0" * 64
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(summarize()["reason"], "report_digest_mismatch")

                manifest_path.write_text("{not json", encoding="utf-8")
                self.assertEqual(summarize()["status"], "unknown")
                self.assertEqual(summarize()["reason"], "manifest_unreadable")

                cursor.unlink()
                self.assertEqual(summarize()["status"], "not_available")
                self.assertEqual(summarize()["reason"], "latest_manifest_missing")

    def _run_preflight(self, context):
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            redirect_stdout(io.StringIO()),
        ):
            return batch.run_translate_preflight(args)

    def _capture_preflight_text(self, context, extra_args=None):
        argv = ["translate-preflight", *(extra_args or ["--strategy", "sync"])]
        args = self.parser.parse_args(argv)
        stdout = io.StringIO()
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            redirect_stdout(stdout),
        ):
            payload = batch.run_translate_preflight(args)
        return payload, stdout.getvalue()

    def test_zero_pending_with_unconfirmed_coverage_blocks(self) -> None:
        context = fake_context(
            self.section,
            requests=[],
            coverage_report_value=coverage_report(
                status="block",
                counts={"unknown": 1, "parse_error": 1},
            ),
        )
        payload = self._run_preflight(context)
        risks = {risk["code"]: risk for risk in payload["risks"]}
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(risks["COVERAGE_UNCONFIRMED"]["severity"], "error")
        self.assertNotIn("NO_PENDING_WORK", risks)

    def test_zero_pending_with_confirmed_coverage_stays_info(self) -> None:
        context = fake_context(
            self.section,
            requests=[],
            coverage_report_value=coverage_report(
                status="attention",
                counts={"unsupported": 1},
            ),
        )
        payload = self._run_preflight(context)
        risks = {risk["code"]: risk for risk in payload["risks"]}
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(risks["NO_PENDING_WORK"]["severity"], "info")
        self.assertNotIn("COVERAGE_UNCONFIRMED", risks)

    def test_zero_pending_without_coverage_evidence_warns(self) -> None:
        context = fake_context(self.section, requests=[])
        payload = self._run_preflight(context)
        risks = {risk["code"]: risk for risk in payload["risks"]}
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(risks["COVERAGE_EVIDENCE_MISSING"]["severity"], "warning")
        self.assertNotIn("NO_PENDING_WORK", risks)

    def test_retrieval_risk_uses_the_selected_strategy_flags(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(
            ["translate-preflight", "--strategy", "gemini_batch"]
        )
        batch_flags = {
            "rag_enabled": True,
            "source_index_enabled": False,
            "story_memory_enabled": False,
            "project_analysis_inject_enabled": False,
        }
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch(
                "project_context_settings.resolve_batch_context_flags",
                return_value=batch_flags,
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertTrue(payload["context_sources"]["rag"])
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertIn("RETRIEVAL_PREFLIGHT_SKIPPED", codes)

        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch(
                "project_context_settings.resolve_batch_context_flags",
                return_value={**batch_flags, "rag_enabled": False},
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertFalse(payload["context_sources"]["rag"])
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertNotIn("RETRIEVAL_PREFLIGHT_SKIPPED", codes)

    def test_batch_strategy_uses_the_batch_translation_profile(self) -> None:
        section = routing_section()
        section = editor.add_profile(
            section,
            label="Batch Gemini",
            provider_id="gemini",
            model="gemini-3.5-flash-batch",
        )
        batch_profile_id = next(
            profile_id
            for profile_id in section["profiles"]
            if profile_id != "gemini-main"
        )
        section["legacy_entrypoints"] = {
            "sync_profile_id": "gemini-main",
            "batch_profile_id": batch_profile_id,
        }
        batch_plan = reader.read_routing_plan(
            {"model_routing": section},
            legacy_execution="gemini_batch",
        )
        context = fake_context(section)
        args = self.parser.parse_args(
            ["translate-preflight", "--strategy", "gemini_batch"]
        )

        with (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", section),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(
                batch,
                "freeze_runtime_routing_plan",
                return_value=batch_plan,
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["profile"]["id"], batch_profile_id)
        self.assertEqual(payload["profile"]["model"], "gemini-3.5-flash-batch")
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertNotIn("STRATEGY_NOT_SUPPORTED", codes)

    def test_batch_missing_credential_is_warning_only(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(
            ["translate-preflight", "--strategy", "gemini_batch"]
        )
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "ready")
        credential_risks = [
            risk
            for risk in payload["risks"]
            if risk["code"] == "CREDENTIAL_UNAVAILABLE"
        ]
        self.assertEqual(len(credential_risks), 1)
        self.assertEqual(credential_risks[0]["severity"], "warning")

    def test_batch_strategy_without_v1_section_resolves_the_batch_plan(self) -> None:
        batch_plan = reader.read_routing_plan({"model_routing": self.section})
        context = fake_context(self.section)
        args = self.parser.parse_args(
            ["translate-preflight", "--strategy", "gemini_batch"]
        )

        with (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", None),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(
                batch,
                "freeze_runtime_routing_plan",
                return_value=batch_plan,
            ) as freeze_batch,
        ):
            payload = batch.run_translate_preflight(args)

        freeze_batch.assert_called_once()
        self.assertEqual(
            payload["profile"]["id"],
            batch_plan.routes[model_profile.STAGE_TRANSLATION].profile_id,
        )

    def test_profile_argument_changes_the_resolved_preflight_profile(self) -> None:
        section = routing_section()
        section = editor.add_profile(
            section,
            label="Alternate Gemini",
            provider_id="gemini",
            model="gemini-3.5-pro",
        )
        alternate_id = next(
            profile_id
            for profile_id in section["profiles"]
            if profile_id != "gemini-main"
        )
        args = self.parser.parse_args(
            [
                "translate-preflight",
                "--strategy",
                "sync",
                "--profile",
                alternate_id,
            ]
        )

        section["legacy_entrypoints"] = {
            "sync_profile_id": "gemini-main",
            "batch_profile_id": "gemini-main",
        }
        section["routes"] = {
            "final_review": {
                "profile_id": "gemini-main",
                "strategy": "gemini_batch",
            }
        }

        def resolved_context(**_kwargs):
            plan = reader.resolve_runtime_plan(
                {"model_routing": section},
                execution="sync",
            )
            return fake_context(section, plan=plan)

        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                side_effect=resolved_context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["profile"]["id"], alternate_id)
        self.assertEqual(payload["profile"]["model"], "gemini-3.5-pro")

    def test_prepare_skip_is_reported_as_an_info_risk(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(runtime, "PREP_ENABLED", True),
        ):
            payload = batch.run_translate_preflight(args)

        risks = {
            risk["code"]: risk
            for risk in payload["risks"]
        }
        self.assertIn("PREPARE_PREFLIGHT_SKIPPED", risks)
        self.assertEqual(risks["PREPARE_PREFLIGHT_SKIPPED"]["severity"], "info")

    def test_enabled_retrieval_without_binding_blocks_preflight(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(runtime, "SYNC_RAG_ENABLED", True),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "blocked")
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertIn("EMBEDDING_PROFILE_UNAVAILABLE", codes)

    def test_legacy_config_keeps_legacy_retrieval_without_binding(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", None),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(runtime, "SYNC_RAG_ENABLED", True),
        ):
            payload = batch.run_translate_preflight(args)

        codes = {risk["code"] for risk in payload["risks"]}
        self.assertNotIn("EMBEDDING_PROFILE_UNAVAILABLE", codes)

    def test_graph_only_retrieval_does_not_require_embedding_binding(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            mock.patch.object(runtime, "SYNC_STORY_MEMORY_ENABLED", True),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "ready")
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertNotIn("EMBEDDING_PROFILE_UNAVAILABLE", codes)
        self.assertIn("RETRIEVAL_PREFLIGHT_SKIPPED", codes)

    def test_missing_credential_blocks_preflight(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(["translate-preflight", "--strategy", "sync"])
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="",
            ),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "blocked")
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertIn("CREDENTIAL_UNAVAILABLE", codes)

    def test_batch_strategy_with_litellm_profile_is_refused(self) -> None:
        section = editor.add_provider(
            editor.empty_section(),
            label="Acme",
            adapter="litellm",
            provider="acme",
            base_url="https://acme.example/v1",
            credential_kind="none",
        )
        section = editor.add_profile(
            section,
            label="Acme Main",
            provider_id="acme",
            model="acme/model-a",
        )
        section = editor.set_defaults(
            section,
            primary_profile_id="acme-main",
            execution_strategy="sync",
        )
        with mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", section):
            context = fake_context(section)
            args = self.parser.parse_args(
                ["translate-preflight", "--strategy", "gemini_batch"]
            )
            with (
                mock.patch.object(
                    runtime,
                    "prepare_sync_translation_execution_context",
                    return_value=context,
                ),
                mock.patch(
                    "model_capability_probe.default_credential_loader",
                    return_value="",
                ),
            ):
                payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "blocked")
        codes = {risk["code"] for risk in payload["risks"]}
        self.assertIn("STRATEGY_NOT_SUPPORTED", codes)

    def test_help_mentions_cost_coverage_and_quality_summaries(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(stdout):
            self.parser.parse_args(["translate-preflight", "--help"])
        help_text = stdout.getvalue().lower()
        listing = self.parser.format_help().lower()
        self.assertIn("cost", help_text)
        self.assertIn("coverage", help_text)
        self.assertIn("quality", help_text)
        self.assertIn("unknown", help_text)
        self.assertIn("translate-preflight", listing)
        self.assertIn("cost", listing)

    def test_text_output_shows_known_cost_and_coverage(self) -> None:
        context = fake_context(
            self.section,
            coverage_report_value=coverage_report(
                status="ready", counts={"translatable": 3}
            ),
        )
        payload, text = self._capture_preflight_text(context)
        self.assertEqual(payload["cost"]["status"], "known")
        cost_line = next(line for line in text.splitlines() if "成本：" in line)
        coverage_line = next(line for line in text.splitlines() if "文本覆盖：" in line)
        quality_line = next(line for line in text.splitlines() if "质量摘要：" in line)
        self.assertIn("gemini-3.5-flash", cost_line)
        self.assertIn("–", cost_line)
        self.assertIn("本次初译计划", cost_line)
        self.assertIn("final review / repair / embedding", cost_line)
        self.assertIn("translatable=3", coverage_line)
        self.assertIn("没有可验证的已有质量报告", quality_line)
        self.assertNotIn("免费", cost_line)
        self.assertNotIn("质量通过", quality_line)
        self.assertNotIn("覆盖完成", coverage_line)

    def test_text_output_unknown_cost_does_not_show_zero(self) -> None:
        section = routing_section()
        section["profiles"]["gemini-main"]["model"] = "unpriced-model-xyz"
        section["profiles"]["gemini-main"]["models"] = ["unpriced-model-xyz"]
        context = fake_context(section)
        with mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", section):
            payload, text = self._capture_preflight_text(context)
        self.assertEqual(payload["cost"]["status"], "unknown")
        cost_line = next(line for line in text.splitlines() if "成本：" in line)
        self.assertIn("无法估算", cost_line)
        self.assertIn("没有可用价格表", cost_line)
        self.assertNotIn("免费", cost_line)
        self.assertNotIn("0.0", cost_line)
        self.assertNotIn("¥0", cost_line)

    def test_text_output_quality_available_stale_and_not_available(self) -> None:
        import hashlib

        context = fake_context(
            self.section,
            coverage_report_value=coverage_report(
                status="ready", counts={"translatable": 3}
            ),
            plan_fingerprint="plan-fingerprint-1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "quality_findings.jsonl"
            report_path.write_text(
                json.dumps({"reason_code": "dup", "severity": "medium"}) + "\n"
                + json.dumps({"reason_code": "missing", "severity": "high"}) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest = {
                "mode": batch.MANIFEST_MODE_TRANSLATION,
                "base_dir": runtime.BASE_DIR,
                "tl_dir": runtime.TL_DIR,
                "translation_plan": {"plan_fingerprint": "plan-fingerprint-1"},
                "last_quality_findings_path": str(report_path),
                "last_check_at": "2026-09-15T00:00:00",
                "last_check_summary": {"quality_findings_sha256": digest},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            cursor = root / "latest_manifest.txt"
            cursor.write_text(str(manifest_path), encoding="utf-8")

            with mock.patch.object(batch, "LATEST_MANIFEST_FILE", str(cursor)):
                _payload, available_text = self._capture_preflight_text(
                    context,
                    extra_args=["--strategy", "gemini_batch"],
                )
                available_line = next(
                    line for line in available_text.splitlines() if "质量摘要：" in line
                )
                self.assertIn("finding 2", available_line)
                self.assertIn("high=1", available_line)
                self.assertNotIn("质量通过", available_line)

                manifest["translation_plan"]["plan_fingerprint"] = "other-plan"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                _payload, stale_text = self._capture_preflight_text(
                    context,
                    extra_args=["--strategy", "gemini_batch"],
                )
                stale_line = next(
                    line for line in stale_text.splitlines() if "质量摘要：" in line
                )
                self.assertIn("过期", stale_line)
                self.assertNotIn("finding", stale_line)
                self.assertNotIn("质量通过", stale_line)

        _payload, missing_text = self._capture_preflight_text(
            context,
            extra_args=["--strategy", "sync"],
        )
        missing_line = next(
            line for line in missing_text.splitlines() if "质量摘要：" in line
        )
        self.assertIn("没有可验证的已有质量报告", missing_line)
        self.assertNotIn("质量通过", missing_line)

    def test_json_mode_suppresses_human_preflight_stdout(self) -> None:
        context = fake_context(self.section)
        args = self.parser.parse_args(
            ["translate-preflight", "--strategy", "sync", "--output", "json"]
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=context,
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader",
                return_value="key",
            ),
            mock.patch.object(
                batch,
                "_read_translator_config_object",
                return_value={},
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            payload = batch.run_translate_preflight(args)

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(stdout.getvalue(), "")

    def test_machine_envelope_keeps_stdout_parseable(self) -> None:
        payload = {
            "status": "ready",
            "strategy": "sync",
            "counts": {"pending_items": 1},
        }
        envelope = batch.build_machine_success_envelope(
            "translate-preflight",
            payload,
            self.parser.parse_args(["translate-preflight", "--strategy", "sync"]),
        )
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "ready")
        self.assertEqual(envelope["result"]["counts"]["pending_items"], 1)
        self.assertNotIn("status", envelope["result"])
        cli_contract.parse_result_envelope(json.dumps(envelope))


class DurableSyncPreflightQualityTests(unittest.TestCase):
    """#488 S1.5: durable Sync quality summaries read bound preview artifacts."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project_root = self.root / "game"
        self.tl_dir = self.project_root / "tl" / "schinese"
        self.tl_dir.mkdir(parents=True)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self._log_patch = mock.patch.object(runtime, "LOG_DIR", str(self.log_dir))
        self._log_patch.start()
        self.addCleanup(self._log_patch.stop)
        self.source_text = '    "Hello"\n'
        self.preview_text = '    "你好..."\n'
        self.plan_payload = self._make_plan_payload()
        self.plan_fingerprint = str(self.plan_payload["plan_fingerprint"])
        self.store, _ = sync_run_store.SyncRunStore.bootstrap(
            self.log_dir / "sync_runs",
            build_run_id(),
            plan=self.plan_payload,
            requests=[
                {
                    "request_id": "req-1",
                    "plan_id": "plan-488-s15",
                    "expected_ids": ["item-1"],
                    "prompt_fingerprint": "p" * 16,
                    "request_fingerprint": "r" * 16,
                }
            ],
        )

    def _make_plan_payload(self) -> dict:
        payload = {
            "schema_version": translation_plan.PLAN_SCHEMA_VERSION,
            "plan_id": "plan-488-s15",
            "run_id": "",
            "execution_strategy": translation_plan.STRATEGY_SYNC,
            "source_identity": {
                "engine": "renpy",
                "adapter_version": "v1",
                "file_digests": {"a.rpy": sha256_text(self.source_text)},
            },
            "config_fingerprint": "b" * 16,
            "model_profile_snapshot": {"provider": "fake"},
            "chunk_policy": {},
            "context_policy": {},
            "chunks": [],
            "request_summaries": [],
            "artifacts": {},
        }
        fingerprint_payload = dict(payload)
        fingerprint_payload.pop("run_id", None)
        fingerprint_payload.pop("plan_fingerprint", None)
        payload["plan_fingerprint"] = translation_plan.short_fingerprint(
            translation_plan.canonical_json(fingerprint_payload)
        )
        return payload

    def _create_preview(self) -> Path:
        preview_path, _ = sync_translation_preview.create_sync_preview(
            log_dir=self.store.run_dir,
            project_root=self.project_root,
            tl_dir=self.tl_dir,
            files=[
                {
                    "relative_path": "a.rpy",
                    "source_text": self.source_text,
                    "preview_text": self.preview_text,
                    "progress_entries": ["item-1"],
                    "translated_items": 1,
                    "quality_subjects": [
                        {
                            "item_id": "item-1",
                            "file_rel_path": "a.rpy",
                            "line": 0,
                            "line_number": 1,
                            "source": "Hello",
                            "translation": "你好...",
                        }
                    ],
                }
            ],
            translation_plan_payload=dict(self.plan_payload),
            request_ids=[],
            durable_check_binding={"writeback_gate": {"decision": "allow"}},
        )
        preview = Path(preview_path)
        self.store.put_artifact(
            kind="preview_manifest",
            relative_path=str(
                preview.resolve().relative_to(self.store.run_dir.resolve())
            ),
            sha256_digest=file_sha256(preview),
            schema_version=sync_translation_preview.VERSION,
        )
        return preview

    def _summarize(
        self,
        *,
        strategy: str = "sync",
        plan_fingerprint: str | None = None,
        base_dir=None,
        tl_dir=None,
    ) -> dict:
        return batch.summarize_preflight_quality(
            plan_fingerprint=plan_fingerprint or self.plan_fingerprint,
            base_dir=str(base_dir or self.project_root),
            tl_dir=str(tl_dir or self.tl_dir),
            strategy=strategy,
        )

    def test_available_summary_reads_bound_durable_sync_preview(self) -> None:
        preview = self._create_preview()
        summary = self._summarize()
        self.assertEqual(summary["status"], "available")
        self.assertEqual(summary["source"], "durable_sync_preview")
        self.assertEqual(summary["manifest_path"], str(preview.resolve()))
        self.assertTrue(summary["generated_at"])
        self.assertGreater(summary["finding_count"], 0)
        self.assertGreaterEqual(summary["severity_counts"]["medium"], 1)
        self.assertTrue(summary["reason_counts"])
        self.assertEqual(summary["matched_by"], ["project", "plan_fingerprint"])

    def test_missing_run_is_not_available(self) -> None:
        empty_log_dir = self.root / "empty-logs"
        with mock.patch.object(runtime, "LOG_DIR", str(empty_log_dir)):
            summary = self._summarize()
        self.assertEqual(summary["status"], "not_available")
        self.assertEqual(summary["reason"], "durable_sync_run_missing")
        self.assertEqual(summary["source"], "none")

    def test_run_without_check_is_not_available(self) -> None:
        summary = self._summarize()
        self.assertEqual(summary["status"], "not_available")
        self.assertEqual(summary["reason"], "durable_sync_check_not_run")
        self.assertEqual(summary["source"], "none")

    def test_project_and_plan_binding_mismatch_is_stale(self) -> None:
        self._create_preview()
        plan_mismatch = self._summarize(plan_fingerprint="other-plan")
        self.assertEqual(plan_mismatch["status"], "stale")
        self.assertEqual(plan_mismatch["reason"], "plan_mismatch")
        project_mismatch = self._summarize(base_dir=str(self.root / "other-game"))
        self.assertEqual(project_mismatch["status"], "stale")
        self.assertEqual(project_mismatch["reason"], "project_mismatch")

    def test_preview_artifact_digest_mismatch_is_stale(self) -> None:
        preview = self._create_preview()
        preview.write_text("{}\n", encoding="utf-8")
        summary = self._summarize()
        self.assertEqual(summary["status"], "stale")
        self.assertEqual(summary["reason"], "preview_manifest_mismatch")

    def test_report_digest_mismatch_is_stale(self) -> None:
        preview = self._create_preview()
        manifest = sync_translation_preview.load_sync_preview(str(preview))
        report = (
            Path(manifest["_manifest_path"]).parent
            / manifest["last_quality_findings_path"]
        )
        report.write_text("{}\n", encoding="utf-8")
        summary = self._summarize()
        self.assertEqual(summary["status"], "stale")
        self.assertEqual(summary["reason"], "report_digest_mismatch")

    def test_missing_report_is_stale(self) -> None:
        preview = self._create_preview()
        manifest = sync_translation_preview.load_sync_preview(str(preview))
        report = (
            Path(manifest["_manifest_path"]).parent
            / manifest["last_quality_findings_path"]
        )
        report.unlink()
        summary = self._summarize()
        self.assertEqual(summary["status"], "stale")
        self.assertEqual(summary["reason"], "report_missing")

    def test_corrupt_preview_manifest_is_unknown(self) -> None:
        preview = self._create_preview()
        preview.write_text("{not json\n", encoding="utf-8")
        self.store.put_artifact(
            kind="preview_manifest",
            relative_path=str(
                preview.resolve().relative_to(self.store.run_dir.resolve())
            ),
            sha256_digest=file_sha256(preview),
            schema_version=sync_translation_preview.VERSION,
        )
        summary = self._summarize()
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual(summary["reason"], "preview_manifest_unreadable")
        self.assertEqual(summary["source"], "durable_sync_preview")

    def test_invalid_preview_artifact_path_is_unknown(self) -> None:
        import sqlite3

        connection = sqlite3.connect(str(self.store.db_path))
        try:
            with connection:
                connection.execute(
                    "INSERT INTO artifacts("
                    " run_id, kind, relative_path, sha256, schema_version, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self.store.run_id,
                        "preview_manifest",
                        "../escape/manifest.json",
                        "a" * 64,
                        1,
                        "2026-01-01T00:00:00Z",
                    ),
                )
        finally:
            connection.close()
        summary = self._summarize()
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual(summary["reason"], "preview_manifest_unreadable")
        self.assertEqual(summary["source"], "durable_sync_preview")

    def test_non_sync_strategy_keeps_explicit_reason(self) -> None:
        summary = self._summarize(strategy="other-strategy")
        self.assertEqual(summary["status"], "not_available")
        self.assertEqual(
            summary["reason"], "quality_source_not_supported_for_strategy"
        )


class PreflightNoProviderCallTests(unittest.TestCase):
    def test_preflight_plan_skips_retrieval_stores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)
            (tl_dir / "script.rpy").write_text('    "Hello"\n', encoding="utf-8")
            log_dir = root / "logs"
            patches = (
                mock.patch.object(runtime, "BASE_DIR", str(root)),
                mock.patch.object(runtime, "TL_DIR", str(tl_dir)),
                mock.patch.object(runtime, "TL_SUBDIR", "game/tl/schinese"),
                mock.patch.object(runtime, "LOG_DIR", str(log_dir)),
                mock.patch.object(runtime, "PROGRESS_LOG", str(log_dir / "progress.json")),
                mock.patch.object(runtime, "PREP_ENABLED", True),
                mock.patch.object(runtime, "INCLUDE_FILES", []),
                mock.patch.object(runtime, "INCLUDE_PREFIXES", []),
                mock.patch.object(runtime, "load_config"),
                mock.patch.object(runtime, "load_translator_settings"),
                mock.patch.object(runtime, "load_glossary"),
                mock.patch.object(runtime, "load_progress", return_value={}),
                mock.patch.object(
                    runtime,
                    "run_prepare_steps",
                    side_effect=AssertionError("preflight must not run prepare"),
                ),
                mock.patch.object(
                    runtime,
                    "maybe_update_sync_rag_store",
                    side_effect=AssertionError("preflight must not update RAG stores"),
                ),
                mock.patch.object(
                    runtime,
                    "embed_texts",
                    side_effect=AssertionError("preflight must not call embeddings"),
                ),
                mock.patch.object(
                    runtime,
                    "embed_sync_query_text",
                    side_effect=AssertionError("preflight must not call embeddings"),
                ),
                mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", None),
                mock.patch.object(runtime, "SYNC_BACKEND", "gemini"),
                mock.patch.object(runtime, "MODELS", ["gemini-3.1-flash-lite"]),
                mock.patch.object(runtime, "CURRENT_MODEL_INDEX", 0),
                mock.patch.object(runtime, "GENERATION_TARGET_LANGUAGE", "schinese"),
                mock.patch.object(runtime, "SYNC_RAG_ENABLED", True),
                mock.patch.object(runtime, "SYNC_STORY_MEMORY_ENABLED", True),
                mock.patch.object(runtime, "SYNC_SOURCE_INDEX_ENABLED", True),
                mock.patch.object(
                    runtime, "SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF", True
                ),
                mock.patch.object(
                    runtime,
                    "retrieve_sync_history_hits",
                    side_effect=AssertionError("retrieval must not run in preflight"),
                ),
                mock.patch.object(
                    runtime,
                    "retrieve_sync_source_hits",
                    side_effect=AssertionError("retrieval must not run in preflight"),
                ),
                mock.patch.object(
                    runtime,
                    "retrieve_sync_story_hits",
                    side_effect=AssertionError("retrieval must not run in preflight"),
                ),
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                context = runtime.prepare_sync_translation_execution_context(
                    require_provider=False,
                    preflight=True,
                )

        self.assertEqual(len(context.plan_build.requests), 1)
        self.assertEqual(len(context.plan_build.plan.chunks), 1)
        self.assertTrue(context.pending_jobs)
        capture = context.captures[0]
        self.assertEqual(capture["rag_stats"]["reason"], "preflight_skipped")
        self.assertEqual(capture["source_index_stats"]["reason"], "preflight_skipped")
        self.assertEqual(capture["retrieval_blocks_text"], "")
        self.assertEqual(capture["analysis_blocks_text"], "")


if __name__ == "__main__":
    unittest.main()
