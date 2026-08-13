import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import gemini_translate_batch as batch
import model_usage_ledger as usage
from gui_qt.diagnostics_context import build_diagnostics_context
from sync_model_backend import SYNC_EXECUTION_MODE, SyncGenerationResult


class ModelUsageLedgerTests(unittest.TestCase):
    def _write_jsonl(self, directory, rows):
        path = os.path.join(directory, "results.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _manifest(
        self,
        game_root,
        result_path,
        *,
        mode="translation",
        execution="batch",
        provider="",
        model="gemini-test",
    ):
        package_dir = os.path.dirname(result_path)
        return {
            "_manifest_path": os.path.join(package_dir, "manifest.json"),
            "_package_dir": package_dir,
            "mode": mode,
            "execution": execution,
            "provider": provider,
            "model": model if execution == "sync" else "",
            "batch_model": model,
            "base_dir": game_root,
            "tl_dir": os.path.join(game_root, "game", "tl", "schinese"),
            "display_name": f"{mode}-fixture",
            "result_jsonl_path": result_path,
            "settings": {"thinking_level": "minimal"},
        }

    def test_batch_gemini_import_is_offline_idempotent_and_keeps_raw_usage(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "chunk-1",
                        "response": {
                            "responseId": "gemini-response-1",
                            "usageMetadata": {
                                "promptTokenCount": 100,
                                "candidatesTokenCount": 40,
                                "thoughtsTokenCount": 10,
                                "cachedContentTokenCount": 20,
                                "totalTokenCount": 150,
                            },
                        },
                    }
                ],
            )
            manifest = self._manifest(game_root, result_path)
            pricing = {
                "currency": "USD",
                "models": {
                    "gemini-test": {
                        "input_per_million": 1.0,
                        "output_per_million": 2.0,
                    }
                },
            }

            first = usage.import_manifest_results(
                manifest, result_path=result_path, pricing_config=pricing
            )
            second = usage.import_manifest_results(
                manifest, result_path=result_path, pricing_config=pricing
            )
            report = usage.query_usage(game_root)
            record = usage.UsageLedger(game_root).load()["records"][0]

            self.assertEqual(first["inserted_records"], 1)
            self.assertEqual(second["inserted_records"], 0)
            self.assertEqual(second["duplicate_records"], 1)
            self.assertEqual(report["totals"]["calls"], 1)
            self.assertEqual(report["totals"]["prompt_tokens"], 100)
            self.assertEqual(report["totals"]["completion_tokens"], 40)
            self.assertEqual(report["totals"]["thoughts_tokens"], 10)
            self.assertEqual(report["totals"]["cached_tokens"], 20)
            self.assertEqual(report["totals"]["total_tokens"], 150)
            self.assertEqual(
                report["totals"]["estimated_cost"]["values"]["USD"],
                0.0002,
            )
            self.assertEqual(report["totals"]["actual_cost"]["values"], {})
            self.assertEqual(
                record["provider_usage"]["cachedContentTokenCount"], 20
            )
            self.assertEqual(record["task_mode"], "translation")
            self.assertEqual(record["stage"], "batch_translation")
            self.assertTrue(os.path.isfile(usage.usage_ledger_path(game_root)))

    def test_sync_gemini_fixture_maps_revision_stage(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "rv-1",
                        "provider": "gemini",
                        "model": "gemini-sync",
                        "execution_mode": "sync",
                        "usage_metadata": {
                            "promptTokenCount": 25,
                            "candidatesTokenCount": 5,
                            "totalTokenCount": 30,
                        },
                        "response": {"responseId": "gemini-sync-response-1"},
                    }
                ],
            )
            manifest = self._manifest(
                game_root,
                result_path,
                mode="revision",
                execution="sync",
                provider="gemini",
                model="gemini-sync",
            )

            summary = usage.import_manifest_results(manifest, result_path=result_path)
            report = usage.query_usage(game_root, task="revision", provider="gemini")
            record = usage.UsageLedger(game_root).load()["records"][0]

            self.assertEqual(summary["inserted_records"], 1)
            self.assertEqual(report["totals"]["total_tokens"], 30)
            self.assertEqual(record["task_mode"], "revision")
            self.assertEqual(record["stage"], "sync_revision")
            self.assertEqual(record["execution_mode"], "sync")

    def test_sync_targeted_retry_attempts_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "chunk-1",
                        "provider": "litellm",
                        "model": "openai/test",
                        "execution_mode": "sync",
                        "response": {
                            "id": "first",
                            "_hidden_params": {"response_cost": 0.001},
                        },
                        "provider_response_attempts": [
                            {
                                "kind": "first_pass",
                                "usage_metadata": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 4,
                                    "total_tokens": 14,
                                },
                            },
                            {
                                "kind": "targeted_retry",
                                "item_ids": ["b"],
                                "response": {"id": "retry"},
                                "usage_metadata": {
                                    "prompt_tokens": 6,
                                    "completion_tokens": 2,
                                    "total_tokens": 8,
                                },
                            },
                        ],
                    }
                ],
            )
            manifest = self._manifest(
                game_root,
                result_path,
                execution="sync",
                provider="litellm",
                model="openai/test",
            )

            summary = usage.import_manifest_results(manifest, result_path=result_path)
            report = usage.query_usage(game_root)
            records = usage.UsageLedger(game_root).load()["records"]

            self.assertEqual(summary["inserted_records"], 2)
            self.assertEqual(report["totals"]["calls"], 2)
            self.assertEqual(report["totals"]["total_tokens"], 22)
            self.assertEqual(records[0]["actual_cost"], 0.001)
            self.assertEqual(records[1]["source"]["attempt_kind"], "targeted_retry")
            self.assertEqual(records[1]["source"]["item_ids"], ["b"])

    def test_litellm_fixture_maps_snake_case_cache_reasoning_and_actual_cost(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "kw-1",
                        "provider": "litellm",
                        "model": "openai/test-model",
                        "execution_mode": "sync",
                        "usage_metadata": {
                            "prompt_tokens": 8,
                            "completion_tokens": 4,
                            "total_tokens": 12,
                            "prompt_tokens_details": {"cached_tokens": 3},
                            "completion_tokens_details": {
                                "reasoning_tokens": 2,
                                "text_tokens": 2,
                            },
                        },
                        "response": {
                            "id": "chatcmpl-fixture-1",
                            "_hidden_params": {"response_cost": 0.00125},
                        },
                    }
                ],
            )
            manifest = self._manifest(
                game_root,
                result_path,
                mode="keyword_extraction",
                execution="sync",
                provider="litellm",
                model="openai/test-model",
            )

            usage.import_manifest_results(manifest, result_path=result_path)
            report = usage.query_usage(game_root, task="keyword", provider="litellm")
            record = usage.UsageLedger(game_root).load()["records"][0]

            self.assertEqual(report["totals"]["prompt_tokens"], 8)
            self.assertEqual(report["totals"]["completion_tokens"], 4)
            self.assertEqual(report["totals"]["reasoning_tokens"], 2)
            self.assertEqual(report["totals"]["text_output_tokens"], 2)
            self.assertEqual(report["totals"]["thoughts_tokens"], 2)
            self.assertEqual(report["totals"]["cached_tokens"], 3)
            self.assertEqual(
                report["totals"]["actual_cost"]["values"]["USD"],
                0.00125,
            )
            self.assertIsNone(record["estimated_cost"])
            self.assertEqual(record["actual_cost_source"], "_hidden_params.response_cost")
            self.assertEqual(record["stage"], "sync_keyword")

    def test_reasoning_budget_diagnostics_detect_empty_exhausted_output(self):
        diagnostics = usage.response_budget_diagnostics(
            response_text="",
            finish_reason="length",
            usage_metadata={
                "completion_tokens": 64,
                "completion_tokens_details": {
                    "reasoning_tokens": 64,
                    "text_tokens": 0,
                },
            },
            max_output_tokens=64,
        )

        self.assertTrue(diagnostics["empty_text"])
        self.assertTrue(diagnostics["truncated"])
        self.assertTrue(diagnostics["reasoning_budget_pressure"])
        self.assertEqual(diagnostics["reason_code"], "reasoning_budget_exhausted")

        record = usage.build_usage_record(
            game_root="C:/fixture",
            task_mode="translation",
            stage="sync_translation",
            provider="litellm",
            model="openai/test",
            usage_metadata={
                "completion_tokens": 64,
                "completion_tokens_details": {
                    "reasoning_tokens": 64,
                    "text_tokens": 0,
                },
            },
            response_diagnostics=diagnostics,
        )
        totals = usage.aggregate_usage_records([record])
        lines = usage.format_usage_report(
            {
                "project": {"game_root": "C:/fixture"},
                "ledger_path": "C:/fixture/usage.json",
                "totals": totals,
            }
        )
        self.assertTrue(any("Reasoning share" in line and "100.0%" in line for line in lines))
        self.assertTrue(
            any("reasoning_budget_pressure=1" in line for line in lines)
        )

    def test_visible_text_is_not_derived_for_ambiguous_provider_counters(self):
        normalized = usage.normalize_usage_metadata(
            {
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        )

        self.assertEqual(normalized["completion_tokens"], 20)
        self.assertEqual(normalized["reasoning_tokens"], 5)
        self.assertIsNone(normalized["text_output_tokens"])

    def test_snake_case_gemini_usage_includes_thoughts_in_output_budget(self):
        normalized = usage.normalize_usage_metadata(
            {
                "prompt_token_count": 10,
                "candidates_token_count": 4,
                "thoughts_token_count": 6,
            }
        )

        self.assertEqual(normalized["completion_tokens"], 4)
        self.assertEqual(normalized["reasoning_tokens"], 6)
        self.assertEqual(normalized["text_output_tokens"], 4)
        self.assertEqual(normalized["billable_output_tokens"], 10)
        self.assertEqual(normalized["total_tokens"], 20)

    def test_safe_diagnostic_metadata_rejects_unstructured_or_secret_fields(self):
        self.assertEqual(
            usage.normalize_response_diagnostics(
                {"reason_code": "provider said secret value"}
            ),
            {},
        )
        self.assertEqual(
            usage.normalize_response_diagnostics({"reason_code": "secret-value"}),
            {},
        )
        normalized = usage.normalize_request_metadata(
            {
                "provider": "openai",
                "credential_identity": "raw-secret-key",
                "credential_source": "secret-value",
                "credential_attempts": ["openai#1:****1234", "raw-secret-key"],
                "ignored_provider_options": ["thinking_config", "secret-value"],
            }
        )
        self.assertEqual(normalized["provider"], "openai")
        self.assertNotIn("credential_identity", normalized)
        self.assertNotIn("credential_source", normalized)
        self.assertEqual(
            normalized["credential_attempts"],
            ["openai#1:****1234"],
        )
        self.assertEqual(
            normalized["ignored_provider_options"],
            ["thinking_config"],
        )

    def test_usage_response_cost_without_currency_stays_unknown_currency(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "kw-cost-only",
                        "provider": "litellm",
                        "model": "openai/test-model",
                        "execution_mode": "sync",
                        "usage_metadata": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                            "response_cost": 0.42,
                        },
                        "response": {"id": "chatcmpl-cost-only"},
                    }
                ],
            )
            manifest = self._manifest(
                game_root,
                result_path,
                mode="keyword_extraction",
                execution="sync",
                provider="litellm",
                model="openai/test-model",
            )

            usage.import_manifest_results(manifest, result_path=result_path)
            record = usage.UsageLedger(game_root).load()["records"][0]
            report = usage.query_usage(game_root)

            self.assertEqual(record["actual_cost"], 0.42)
            self.assertIsNone(record["actual_cost_currency"])
            self.assertEqual(record["actual_cost_source"], "usage.response_cost")
            # Missing currency buckets under "unknown", never the literal "None".
            self.assertEqual(
                report["totals"]["actual_cost"]["values"],
                {"unknown": 0.42},
            )
            self.assertNotIn("None", report["totals"]["actual_cost"]["values"])

    def test_non_numeric_schema_version_raises_usage_ledger_error(self):
        with tempfile.TemporaryDirectory() as game_root:
            ledger = usage.UsageLedger(game_root)
            os.makedirs(os.path.dirname(ledger.path), exist_ok=True)
            with open(ledger.path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": "x",
                        "project": ledger.project,
                        "records": [],
                    },
                    handle,
                )
            with self.assertRaises(usage.UsageLedgerError):
                ledger.load()

    def test_missing_usage_stays_unknown_instead_of_zero(self):
        with tempfile.TemporaryDirectory() as game_root:
            usage.record_generation_usage(
                game_root=game_root,
                task_mode="repair",
                stage="repair",
                provider="gemini",
                model="gemini-test",
                usage_metadata={},
                response_payload={"responseId": "unknown-usage-1"},
                operation_id="repair-operation",
                run_id="repair-run",
                source_key="repair-1",
            )
            report = usage.query_usage(game_root)

            self.assertEqual(report["totals"]["calls"], 1)
            self.assertIsNone(report["totals"]["prompt_tokens"])
            self.assertIsNone(report["totals"]["completion_tokens"])
            self.assertIsNone(report["totals"]["total_tokens"])
            self.assertEqual(report["totals"]["total_tokens_unknown_records"], 1)
            self.assertEqual(report["totals"]["estimated_cost"]["values"], {})
            self.assertEqual(report["totals"]["actual_cost"]["values"], {})

    def test_concurrent_writers_keep_both_records(self):
        with tempfile.TemporaryDirectory() as game_root:
            barrier = threading.Barrier(2)
            original_load = usage.UsageLedger.load

            def slow_load(ledger):
                payload = original_load(ledger)
                time.sleep(0.05)
                return payload

            def write_record(index):
                barrier.wait()
                return usage.record_generation_usage(
                    game_root=game_root,
                    task_mode="translation",
                    stage="sync_translation",
                    provider="gemini",
                    model="gemini-test",
                    usage_metadata={"totalTokenCount": index + 1},
                    response_payload={"responseId": f"concurrent-{index}"},
                    operation_id="concurrent-operation",
                    run_id="concurrent-run",
                    source_key=f"record-{index}",
                )

            with (
                mock.patch.object(usage.UsageLedger, "load", slow_load),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                results = list(executor.map(write_record, range(2)))

            report = usage.query_usage(game_root)
            ledger = usage.UsageLedger(game_root)

            self.assertEqual(sum(row["inserted_records"] for row in results), 2)
            self.assertEqual(report["totals"]["records"], 2)
            self.assertEqual(report["totals"]["total_tokens"], 3)
            self.assertFalse(os.path.exists(ledger.lock_path))

    def test_project_identity_prevents_cross_root_mixing(self):
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = usage.record_generation_usage(
                game_root=first_root,
                task_mode="analysis",
                stage="label",
                provider="gemini",
                model="gemini-test",
                usage_metadata={"totalTokenCount": 7},
                response_payload={"responseId": "same-provider-id"},
                run_id="analysis-1",
                source_key="label-a",
            )
            second = usage.record_generation_usage(
                game_root=second_root,
                task_mode="analysis",
                stage="label",
                provider="gemini",
                model="gemini-test",
                usage_metadata={"totalTokenCount": 7},
                response_payload={"responseId": "same-provider-id"},
                run_id="analysis-1",
                source_key="label-a",
            )

            self.assertEqual(first["inserted_records"], 1)
            self.assertEqual(second["inserted_records"], 1)
            with self.assertRaises(usage.UsageLedgerError):
                usage.UsageLedger(
                    second_root,
                    path=usage.usage_ledger_path(first_root),
                ).load()

    def test_same_provider_response_id_from_different_models_stays_distinct(self):
        with tempfile.TemporaryDirectory() as game_root:
            for model in ("provider/model-a", "provider/model-b"):
                usage.record_generation_usage(
                    game_root=game_root,
                    task_mode="translation",
                    stage="sync_translation",
                    provider="litellm",
                    model=model,
                    usage_metadata={"total_tokens": 5},
                    response_payload={"id": "shared-fixture-id"},
                    run_id="shared-id-run",
                    source_key=model,
                )

            report = usage.query_usage(game_root, provider="litellm")

            self.assertEqual(report["totals"]["calls"], 2)
            self.assertEqual(report["totals"]["total_tokens"], 10)

    def test_report_filters_and_groups_by_task_stage_provider_model(self):
        with tempfile.TemporaryDirectory() as game_root:
            for response_id, task, stage, provider, model, tokens in (
                ("r1", "translation", "batch_translation", "gemini", "m1", 10),
                ("r2", "analysis", "brief", "litellm", "m2", 20),
            ):
                usage.record_generation_usage(
                    game_root=game_root,
                    task_mode=task,
                    stage=stage,
                    provider=provider,
                    model=model,
                    usage_metadata={"total_tokens": tokens},
                    response_payload={"id": response_id},
                    run_id=f"run-{response_id}",
                    source_key=response_id,
                )

            all_report = usage.query_usage(game_root)
            filtered = usage.query_usage(
                game_root,
                task="analysis",
                stage="brief",
                provider="litellm",
                model="m2",
            )

            self.assertEqual(len(all_report["groups"]), 2)
            self.assertEqual(filtered["totals"]["records"], 1)
            self.assertEqual(filtered["totals"]["total_tokens"], 20)
            self.assertEqual(filtered["groups"][0]["task_mode"], "analysis")

    def test_retry_merge_import_expands_original_result_lineage(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            parent_dir = os.path.join(package, "parent")
            retry_dir = os.path.join(package, "retry")
            os.makedirs(parent_dir)
            os.makedirs(retry_dir)
            parent_result = self._write_jsonl(
                parent_dir,
                [
                    {
                        "key": "chunk-1",
                        "response": {
                            "responseId": "parent-response",
                            "usageMetadata": {"totalTokenCount": 10},
                        },
                    }
                ],
            )
            retry_result = self._write_jsonl(
                retry_dir,
                [
                    {
                        "key": "chunk-2",
                        "response": {
                            "responseId": "retry-response",
                            "usageMetadata": {"totalTokenCount": 20},
                        },
                    }
                ],
            )
            retry_manifest = self._manifest(game_root, retry_result)
            retry_manifest_path = os.path.join(retry_dir, "manifest.json")
            retry_manifest["_manifest_path"] = retry_manifest_path
            with open(retry_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(retry_manifest, handle)

            parent_manifest = self._manifest(game_root, os.path.join(parent_dir, "merged.jsonl"))
            parent_manifest["retry_merge_history"] = [
                {
                    "previous_result_jsonl_path": parent_result,
                    "retry_manifest": retry_manifest_path,
                    "retry_result_jsonl_path": retry_result,
                }
            ]

            with mock.patch.object(
                batch, "_read_translator_config_object", return_value={}
            ):
                first = batch.import_manifest_usage(parent_manifest)
                second = batch.import_manifest_usage(parent_manifest)
            report = usage.query_usage(game_root)

            self.assertEqual(first["inserted_records"], 2)
            self.assertEqual(first["result_path"], "")
            self.assertEqual(first["result_paths"], [parent_result, retry_result])
            self.assertEqual(second["inserted_records"], 0)
            self.assertEqual(second["duplicate_records"], 2)
            self.assertEqual(report["totals"]["calls"], 2)
            self.assertEqual(report["totals"]["total_tokens"], 30)

    def test_probe_records_only_successful_provider_responses(self):
        with tempfile.TemporaryDirectory() as package:
            manifest = {
                "_manifest_path": os.path.join(package, "manifest.json"),
                "_package_dir": package,
                "batch_model": "gemini-test",
                "settings": {"thinking_level": "minimal"},
                "chunks": [{"key": "chunk-1", "items": [{"id": "item-1"}]}],
            }
            request_rows = [
                {
                    "key": "chunk-1",
                    "request": {"contents": [], "generation_config": {}},
                }
            ]
            response_text = json.dumps(
                [{"id": "item-1", "translation": "译文"}], ensure_ascii=False
            )
            success = SyncGenerationResult(
                provider="gemini",
                model="gemini-test",
                execution_mode=SYNC_EXECUTION_MODE,
                response_payload={"responseId": "probe-response"},
                response_text=response_text,
                finish_reason="STOP",
                usage_metadata={"totalTokenCount": 9},
            )

            class SuccessBackend:
                def __init__(self, *_args, **_kwargs):
                    pass

                def generate(self, _request):
                    return success

            with (
                mock.patch.object(batch, "load_manifest", return_value=manifest),
                mock.patch.object(batch, "load_request_rows", return_value=request_rows),
                mock.patch.object(batch, "create_batch_client", return_value=object()),
                mock.patch.object(batch, "GeminiSyncBackend", SuccessBackend),
                mock.patch.object(
                    batch, "record_generation_usage_best_effort"
                ) as recorder,
            ):
                success_summary = batch.probe_requests("unused", limit=1)

            self.assertEqual(success_summary["request_errors"], 0)
            self.assertEqual(success_summary["parse_ok"], 1)
            recorder.assert_called_once()
            self.assertEqual(recorder.call_args.kwargs["stage"], "probe")

            class FailingBackend:
                def __init__(self, *_args, **_kwargs):
                    pass

                def generate(self, _request):
                    raise RuntimeError("provider failed")

            with (
                mock.patch.object(batch, "load_manifest", return_value=manifest),
                mock.patch.object(batch, "load_request_rows", return_value=request_rows),
                mock.patch.object(batch, "create_batch_client", return_value=object()),
                mock.patch.object(batch, "GeminiSyncBackend", FailingBackend),
                mock.patch.object(
                    batch, "record_generation_usage_best_effort"
                ) as failed_recorder,
            ):
                failed_summary = batch.probe_requests("unused", limit=1)

            self.assertEqual(failed_summary["request_errors"], 1)
            failed_recorder.assert_not_called()

    def test_probe_rejects_missing_or_empty_manifest_chunk_before_provider_call(self):
        request_rows = [
            {
                "key": "chunk-1",
                "request": {"contents": [], "generation_config": {}},
            }
        ]
        invalid_manifests = (
            (
                {
                    "_manifest_path": "manifest.json",
                    "chunks": [],
                },
                "PROBE_REQUEST_CHUNK_MISSING",
            ),
            (
                {
                    "_manifest_path": "manifest.json",
                    "chunks": [{"key": "chunk-1", "items": []}],
                },
                "PROBE_REQUEST_CHUNK_EMPTY",
            ),
        )

        for manifest, code_name in invalid_manifests:
            with self.subTest(code_name=code_name):
                with (
                    mock.patch.object(batch, "load_manifest", return_value=manifest),
                    mock.patch.object(
                        batch, "load_request_rows", return_value=request_rows
                    ),
                    mock.patch.object(batch, "create_batch_client") as create_client,
                    self.assertRaises(batch.cli_contract.MachineContractError) as error,
                ):
                    batch.probe_requests("unused", limit=1)

                self.assertEqual(error.exception.code_name, code_name)
                create_client.assert_not_called()

    def test_probe_rejects_request_row_without_key_before_provider_call(self):
        manifest = {
            "_manifest_path": "manifest.json",
            "chunks": [{"key": "chunk-1", "items": [{"id": "item-1"}]}],
        }
        request_rows = [
            {"request": {"contents": [], "generation_config": {}}},
        ]

        with (
            mock.patch.object(batch, "load_manifest", return_value=manifest),
            mock.patch.object(batch, "load_request_rows", return_value=request_rows),
            mock.patch.object(batch, "create_batch_client") as create_client,
            self.assertRaises(batch.cli_contract.MachineContractError) as error,
        ):
            batch.probe_requests("unused", limit=1)

        self.assertEqual(
            error.exception.code_name,
            "PROBE_REQUEST_CHUNK_MISSING",
        )
        self.assertEqual(error.exception.details["key"], "")
        create_client.assert_not_called()

    def test_repair_records_only_successful_provider_responses(self):
        with tempfile.TemporaryDirectory() as package:
            report_path = os.path.join(package, "remaining.jsonl")
            job = {
                "key": "repair-1",
                "file_path": "script.rpy",
                "items": [
                    {
                        "id": "item-1",
                        "line": 1,
                        "text": "Hello",
                        "start": 0,
                        "end": 7,
                        "prefix": "",
                        "quote": '"',
                    }
                ],
            }
            response_text = json.dumps(
                [{"id": "item-1", "translation": "译文"}], ensure_ascii=False
            )
            response = {
                "provider": "gemini",
                "model": "gemini-test",
                "execution_mode": "sync",
                "response_payload": {"responseId": "repair-response"},
                "response_text": response_text,
                "finish_reason": "STOP",
                "usage_metadata": {"totalTokenCount": 12},
            }

            common_patches = (
                mock.patch.object(
                    batch, "load_repair_report_items", return_value=[{"line": 1}]
                ),
                mock.patch.object(
                    batch, "build_repair_jobs", return_value=([job], [])
                ),
                mock.patch.object(
                    batch,
                    "build_repair_request",
                    return_value={"key": "repair-1", "request": {}},
                ),
                mock.patch.object(batch, "REPAIR_RUNS_DIR", package),
                mock.patch.object(batch, "STORY_MEMORY_ENABLED", False),
                mock.patch.object(
                    batch.legacy,
                    "validate_translation",
                    return_value=(False, "fixture validation failure"),
                ),
            )
            with contextlib.ExitStack() as stack:
                for patcher in common_patches:
                    stack.enter_context(patcher)
                stack.enter_context(
                    mock.patch.object(batch, "run_sync_request", return_value=response)
                )
                recorder = stack.enter_context(mock.patch.object(
                    batch, "record_generation_usage_best_effort"
                ))
                success_summary = batch.repair_remaining_items(report_path)

            self.assertEqual(success_summary["request_errors"], 0)
            recorder.assert_called_once()
            self.assertEqual(recorder.call_args.kwargs["stage"], "repair")

            common_failure_patches = (
                mock.patch.object(
                    batch, "load_repair_report_items", return_value=[{"line": 1}]
                ),
                mock.patch.object(
                    batch, "build_repair_jobs", return_value=([job], [])
                ),
                mock.patch.object(
                    batch,
                    "build_repair_request",
                    return_value={"key": "repair-1", "request": {}},
                ),
                mock.patch.object(batch, "REPAIR_RUNS_DIR", package),
                mock.patch.object(batch, "STORY_MEMORY_ENABLED", False),
            )
            with contextlib.ExitStack() as stack:
                for patcher in common_failure_patches:
                    stack.enter_context(patcher)
                stack.enter_context(mock.patch.object(
                    batch, "run_sync_request", side_effect=RuntimeError("provider failed")
                ))
                failed_recorder = stack.enter_context(mock.patch.object(
                    batch, "record_generation_usage_best_effort"
                ))
                failed_summary = batch.repair_remaining_items(report_path)

            self.assertEqual(failed_summary["request_errors"], 1)
            failed_recorder.assert_not_called()


    def test_cli_parser_exposes_json_filters_and_import_target(self):
        report_args = batch.build_arg_parser().parse_args(
            [
                "usage-report",
                "--task",
                "translation",
                "--provider",
                "gemini",
                "--group-by",
                "provider,model",
                "--json",
            ]
        )
        import_args = batch.build_arg_parser().parse_args(
            ["usage-import", "C:/pkg/manifest.json", "--json"]
        )

        self.assertEqual(report_args.command, "usage-report")
        self.assertTrue(report_args.json)
        self.assertEqual(report_args.group_by, "provider,model")
        self.assertEqual(import_args.target, "C:/pkg/manifest.json")
        self.assertTrue(import_args.json)

    def test_usage_import_json_dispatch_is_offline_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as game_root, tempfile.TemporaryDirectory() as package:
            result_path = self._write_jsonl(
                package,
                [
                    {
                        "key": "chunk-1",
                        "response": {
                            "responseId": "offline-cli-response",
                            "usageMetadata": {"totalTokenCount": 17},
                        },
                    }
                ],
            )
            manifest = self._manifest(game_root, result_path)
            parser = batch.build_arg_parser()
            args = parser.parse_args(
                ["usage-import", manifest["_manifest_path"], "--json"]
            )
            output = io.StringIO()
            previous_base_dir = batch.legacy.BASE_DIR
            batch.legacy.BASE_DIR = ""
            try:
                with (
                    mock.patch.object(batch, "initialize_batch_logging"),
                    mock.patch.object(batch.legacy, "load_config"),
                    mock.patch.object(batch.legacy, "load_translator_settings"),
                    mock.patch.object(batch, "load_batch_settings"),
                    mock.patch.object(batch, "load_manifest", return_value=manifest),
                    mock.patch.object(
                        batch, "_read_translator_config_object", return_value={}
                    ),
                    mock.patch.object(
                        batch,
                        "create_batch_client",
                        side_effect=AssertionError("usage-import must stay offline"),
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    returned = batch.dispatch_command(parser, args)
            finally:
                batch.legacy.BASE_DIR = previous_base_dir

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["inserted_records"], 1)
            self.assertEqual(payload["report"]["totals"]["calls"], 1)
            self.assertEqual(returned["report"]["totals"]["total_tokens"], 17)


    def test_gui_diagnostics_wraps_public_project_summary(self):
        with tempfile.TemporaryDirectory() as game_root:
            usage.record_generation_usage(
                game_root=game_root,
                task_mode="translation",
                stage="sync_translation",
                provider="litellm",
                model="provider/model",
                usage_metadata={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
                response_payload={"id": "gui-summary-1"},
                run_id="gui-run",
                source_key="gui-1",
            )

            context = build_diagnostics_context(
                latest_manifest_path=None,
                manifest=None,
                batch_script_path="gemini_translate_batch.py",
                logs_dir="logs",
                game_root=game_root,
            )

            self.assertTrue(any("累计 1 次调用" in fact for fact in context.facts))
            self.assertTrue(any("12 token" in fact for fact in context.facts))

    def test_gui_diagnostics_surfaces_corrupt_ledger(self):
        with tempfile.TemporaryDirectory() as game_root:
            ledger_path = usage.usage_ledger_path(game_root)
            os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
            with open(ledger_path, "w", encoding="utf-8") as handle:
                handle.write("{not-json")

            context = build_diagnostics_context(
                latest_manifest_path=None,
                manifest=None,
                batch_script_path="gemini_translate_batch.py",
                logs_dir="logs",
                game_root=game_root,
            )

            self.assertTrue(
                any("模型用量账本读取失败" in fact for fact in context.facts)
            )

    def test_sync_path_buffers_records_until_flush(self):
        import translator_runtime as runtime
        from sync_model_backend import SyncGenerationResult, SYNC_EXECUTION_MODE

        with tempfile.TemporaryDirectory() as game_root:
            previous_base = runtime.BASE_DIR
            previous_backend = runtime.SYNC_BACKEND
            runtime.BASE_DIR = game_root
            runtime.SYNC_BACKEND = "litellm"
            usage_buffer = []
            fake_result = SyncGenerationResult(
                provider="litellm",
                model="provider/model",
                execution_mode=SYNC_EXECUTION_MODE,
                response_payload={"id": "buffered-1"},
                response_text='[{"id":"a","translation":"你好"}]',
                finish_reason="stop",
                usage_metadata={"totalTokenCount": 7},
            )
            fake_backend = mock.Mock()
            fake_backend.generate.return_value = fake_result
            try:
                with mock.patch(
                    "litellm_sync_backend.LiteLLMSyncBackend",
                    return_value=fake_backend,
                ), mock.patch.object(
                    runtime, "get_current_model", return_value="provider/model"
                ):
                    runtime.call_gemini_sdk(
                        "prompt",
                        [{"id": "a", "text": "Hello"}],
                        usage_run_id="sync-run-1",
                        usage_buffer=usage_buffer,
                        usage_operation_id="sync-run-1",
                    )
                self.assertEqual(len(usage_buffer), 1)
                self.assertFalse(os.path.exists(usage.usage_ledger_path(game_root)))

                usage.UsageLedger(game_root).add_records(usage_buffer)
                report = usage.query_usage(game_root)
                self.assertEqual(report["totals"]["records"], 1)
                self.assertEqual(report["totals"]["total_tokens"], 7)
                self.assertEqual(
                    usage.UsageLedger(game_root).load()["records"][0]["operation_id"],
                    "sync-run-1",
                )
            finally:
                runtime.BASE_DIR = previous_base
                runtime.SYNC_BACKEND = previous_backend

    def test_runtime_sync_usage_summary_preserves_unknown_text_tokens(self):
        import translator_runtime as runtime

        records = [
            usage.build_usage_record(
                game_root="C:/fixture",
                task_mode="translation",
                stage="sync_translation",
                provider="litellm",
                model="openai/test",
                usage_metadata={
                    "prompt_tokens": 3,
                    "completion_tokens": 23,
                    "completion_tokens_details": {"reasoning_tokens": 17},
                    "total_tokens": 26,
                },
                response_diagnostics={
                    "completion_tokens": 23,
                    "reasoning_tokens": 17,
                    "empty_text": False,
                    "truncated": False,
                    "reasoning_budget_pressure": False,
                },
            )
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runtime.print_sync_usage_summary(records)

        self.assertIn(
            "Sync output tokens: completion=23 reasoning=17 text=unknown",
            output.getvalue(),
        )

    def test_import_best_effort_does_not_swallow_system_exit(self):
        with self.assertRaises(SystemExit):
            with mock.patch.object(
                batch,
                "import_manifest_usage",
                side_effect=SystemExit("contract failure"),
            ):
                batch.import_manifest_usage_best_effort({"_manifest_path": "x"})


if __name__ == "__main__":
    unittest.main()
