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
import translator_runtime as runtime


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


def fake_context(section: dict, *, plan=None):
    plan = plan or reader.read_routing_plan({"model_routing": section})
    chunks = [SimpleNamespace(chunk_id="chunk-1"), SimpleNamespace(chunk_id="chunk-2")]
    requests = [
        SimpleNamespace(expected_ids=["a", "b"]),
        SimpleNamespace(expected_ids=["c"]),
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
            plan=SimpleNamespace(chunks=chunks, source_identity=identity),
            requests=requests,
        ),
        routing_plan=plan,
        adapter_snapshot=SimpleNamespace(
            project=SimpleNamespace(source_documents=documents)
        ),
        pending_jobs=[{"file_rel_path": "a.rpy", "tasks": [{}]}],
    )


class PreflightCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = routing_section()
        self.parser = batch.build_arg_parser()
        self.legacy_patches = (
            mock.patch.object(runtime, "MODEL_ROUTING_CONFIG", self.section),
            mock.patch.object(runtime, "BASE_DIR", "C:/game/work"),
            mock.patch.object(runtime, "TL_DIR", "C:/game/work/game/tl/schinese"),
            mock.patch.object(runtime, "GENERATION_TARGET_LANGUAGE", "schinese"),
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
