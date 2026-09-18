"""Offline tests for final-review sync execution (#431 S3)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import MappingProxyType
from unittest import mock

import cli_contract
import final_review as fr
import final_review_sync as frs
import gemini_translate_batch as batch
import model_profiles_editor as editor
import model_routing_reader as routing_reader
import translator_runtime as runtime
from sync_model_backend import SyncBackendError, SyncGenerationResult


def review_items(count: int = 3) -> list[dict]:
    return [
        {
            "id": f"id-{index}",
            "identity_v2": f"id-{index}",
            "file_rel_path": f"script_{index}.rpy",
            "source": f"source {index}",
            "current_translation": f"译文 {index}",
            "line_number": index + 1,
        }
        for index in range(count)
    ]


def build_package(root: str, *, count: int = 3, strategy: str = "sync") -> dict:
    items = review_items(count)
    snapshot = fr.build_context_snapshot(translation_items=items)
    units = fr.build_review_units(
        items,
        chunk_size=1,
        context_digest=snapshot["context_digest"],
        snapshot_digest=snapshot["snapshot_digest"],
        model="test-model",
    )
    readiness = fr.evaluate_readiness(
        pending_task_count=0,
        review_item_count=len(items),
        require_zero_pending=True,
    )
    manifest = fr.build_campaign_manifest(
        package_dir=root,
        display_name="test-sync-campaign",
        snapshot=snapshot,
        units=units,
        readiness=readiness,
        model="test-model",
        chunk_size=1,
        execution_strategy=strategy,
        input_jsonl_path="",
    )
    fr.write_campaign_package(
        root,
        manifest=manifest,
        snapshot=snapshot,
        units=units,
        findings=[],
    )
    return fr.load_campaign_package(root)


def response_for(unit: dict, *, with_finding: bool = False) -> str:
    findings = []
    if with_finding:
        findings.append(
            {
                "item_id": unit["items"][0]["id"],
                "finding_type": "style_drift",
                "severity": "low",
                "reason": "test finding",
            }
        )
    return json.dumps({"findings": findings}, ensure_ascii=False)


class SyncPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _unit(self) -> dict:
        package_dir = os.path.join(self.tmp.name, "payload")
        return build_package(package_dir, count=1)["units"][0]

    def test_sync_payload_reuses_review_prompt_and_schema(self) -> None:
        unit = self._unit()
        payload = frs.build_sync_request_payload(
            unit,
            shared_context={"macro_setting": "macro text"},
            model="test-model",
            structured_output_mode="json_object",
        )

        self.assertIn("source 0", payload["contents"])
        self.assertIn("macro text", payload["contents"])
        self.assertTrue(payload["system_instruction"])
        config = payload["generation_config"]
        self.assertEqual(config["structured_output_mode"], "json_object")
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertIn("findings", config["response_json_schema"]["properties"])

    def test_sync_payload_requires_unit_id(self) -> None:
        with self.assertRaises(frs.FinalReviewSyncError):
            frs.build_sync_request_payload({})

    def test_mapping_proxy_shared_context_is_accepted(self) -> None:
        payload = frs.build_sync_request_payload(
            self._unit(),
            shared_context=MappingProxyType({"macro_setting": "proxy"}),
        )
        self.assertIn("proxy", payload["contents"])


class RunSyncCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _package_dir(self, name: str = "campaign") -> str:
        return os.path.join(self.tmp.name, name)

    def test_completes_units_and_skips_done_on_rerun(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=3)
        seen: list[str] = []

        def generate(payload):
            seen.append(str(payload["contents"]))
            return {
                "response_text": json.dumps({"findings": []}),
                "provider": "fake",
                "model": "test-model",
                "usage_metadata": {"total_tokens": 3},
            }

        result = frs.run_sync_campaign(package_dir, generate=generate)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(result["done_delta"], 3)
        self.assertEqual(result["failed_delta"], 0)
        self.assertEqual(len(seen), 3)
        package = fr.load_campaign_package(package_dir)
        self.assertEqual(
            [unit["status"] for unit in package["units"]],
            [fr.STATUS_DONE, fr.STATUS_DONE, fr.STATUS_DONE],
        )

        result2 = frs.run_sync_campaign(package_dir, generate=generate)
        self.assertEqual(result2["status"], "no_work")
        self.assertEqual(result2["skip_count"], 3)
        self.assertEqual(len(seen), 3)

        result3 = frs.run_sync_campaign(package_dir, generate=generate, force=True)
        self.assertEqual(result3["status"], "completed")
        self.assertEqual(result3["done_delta"], 3)
        self.assertEqual(len(seen), 6)

    def test_live_context_change_converges_to_no_work(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=2)
        calls = {"count": 0}

        def generate(_payload):
            calls["count"] += 1
            return {"response_text": json.dumps({"findings": []})}

        first = frs.run_sync_campaign(
            package_dir,
            generate=generate,
            live_context_digest="live-context-after-build",
        )
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["done_delta"], 2)

        second = frs.run_sync_campaign(
            package_dir,
            generate=generate,
            live_context_digest="live-context-after-build",
        )
        self.assertEqual(second["status"], "no_work")
        self.assertEqual(second["skip_count"], 2)
        self.assertEqual(calls["count"], 2)

    def test_records_findings_from_sync_responses(self) -> None:
        package_dir = self._package_dir()
        package = build_package(package_dir, count=2)
        units = {str(unit["unit_id"]): unit for unit in package["units"]}
        order = list(units)
        calls = {"index": 0}

        def generate(_payload):
            unit = units[order[calls["index"]]]
            calls["index"] += 1
            return {"response_text": response_for(unit, with_finding=True)}

        result = frs.run_sync_campaign(package_dir, generate=generate)

        self.assertEqual(result["finding_count"], 2)
        loaded = fr.load_campaign_package(package_dir)
        self.assertEqual(len(loaded["findings"]), 2)
        self.assertEqual(
            {finding["finding_type"] for finding in loaded["findings"]},
            {"style_drift"},
        )

    def test_unit_failure_does_not_stop_later_units(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=3)
        calls = {"count": 0}

        def generate(_payload):
            calls["count"] += 1
            if calls["count"] == 2:
                raise SyncBackendError("provider_error")
            return {"response_text": json.dumps({"findings": []})}

        result = frs.run_sync_campaign(package_dir, generate=generate)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(result["done_delta"], 2)
        self.assertEqual(result["failed_delta"], 1)
        self.assertEqual(calls["count"], 3)
        statuses = [unit["status"] for unit in fr.load_campaign_package(package_dir)["units"]]
        self.assertEqual(statuses.count(fr.STATUS_DONE), 2)
        self.assertEqual(statuses.count(fr.STATUS_FAILED), 1)
        failed = next(
            unit
            for unit in fr.load_campaign_package(package_dir)["units"]
            if unit["status"] == fr.STATUS_FAILED
        )
        self.assertTrue(str(failed["error"]).startswith("sync_provider_error"))

    def test_systemic_failures_abort_campaign(self) -> None:
        for category in (
            "authentication",
            "missing_dependency",
            "unsupported_capability",
        ):
            with self.subTest(category=category):
                package_dir = self._package_dir(f"campaign-{category}")
                build_package(package_dir, count=3)
                calls = {"count": 0}

                def generate(_payload):
                    calls["count"] += 1
                    raise SyncBackendError(category)

                result = frs.run_sync_campaign(package_dir, generate=generate)

                self.assertEqual(result["status"], "aborted")
                self.assertEqual(result["abort_category"], category)
                self.assertEqual(calls["count"], 1)
                self.assertEqual(result["run_count"], 1)
                self.assertEqual(result["deferred_count"], 2)
                self.assertEqual(len(result["planned_unit_ids"]), 3)
                self.assertEqual(len(result["to_run_unit_ids"]), 3)
                self.assertEqual(len(result["attempted_unit_ids"]), 1)
                statuses = [
                    unit["status"]
                    for unit in fr.load_campaign_package(package_dir)["units"]
                ]
                self.assertEqual(statuses.count(fr.STATUS_FAILED), 1)
                self.assertEqual(statuses.count(fr.STATUS_PENDING), 2)

    def test_dry_run_and_limit_do_not_call_provider(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=3)
        calls = {"count": 0}

        def generate(_payload):
            calls["count"] += 1
            return {"response_text": json.dumps({"findings": []})}

        dry = frs.run_sync_campaign(package_dir, generate=generate, dry_run=True, limit=1)
        self.assertEqual(dry["status"], "dry_run")
        self.assertEqual(dry["run_count"], 1)
        self.assertEqual(dry["deferred_count"], 2)
        self.assertEqual(len(dry["planned_unit_ids"]), 3)
        self.assertEqual(len(dry["to_run_unit_ids"]), 1)
        self.assertEqual(dry["attempted_unit_ids"], [])
        self.assertEqual(calls["count"], 0)

        limited = frs.run_sync_campaign(package_dir, generate=generate, limit=2)
        self.assertEqual(limited["run_count"], 2)
        self.assertEqual(limited["deferred_count"], 1)
        self.assertEqual(len(limited["planned_unit_ids"]), 3)
        self.assertEqual(len(limited["to_run_unit_ids"]), 2)
        self.assertEqual(len(limited["attempted_unit_ids"]), 2)
        self.assertEqual(calls["count"], 2)

    def test_negative_limit_is_rejected_without_provider_calls(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=1)
        calls = {"count": 0}

        with self.assertRaises(frs.FinalReviewSyncError):
            frs.run_sync_campaign(
                package_dir,
                generate=lambda _payload: calls.__setitem__("count", calls["count"] + 1),
                limit=-1,
            )
        self.assertEqual(calls["count"], 0)

    def test_internal_payload_error_is_not_a_provider_failure(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=1)
        package = fr.load_campaign_package(package_dir)
        units = [dict(package["units"][0])]
        units[0].pop("unit_id")
        fr.write_campaign_package(
            package_dir,
            manifest=dict(package["manifest"]),
            snapshot=dict(package["snapshot"]),
            units=units,
            findings=[],
        )

        with self.assertRaises(frs.FinalReviewSyncError):
            frs.run_sync_campaign(
                package_dir,
                generate=lambda _payload: {"response_text": "{}"},
            )

        persisted = fr.load_campaign_package(package_dir)
        self.assertEqual(persisted["units"][0]["status"], fr.STATUS_PENDING)

    def test_batch_package_is_rejected(self) -> None:
        package_dir = self._package_dir()
        build_package(package_dir, count=1, strategy=fr.EXECUTION_STRATEGY_GEMINI_BATCH)

        with self.assertRaises(frs.FinalReviewSyncError):
            frs.run_sync_campaign(
                package_dir,
                generate=lambda _payload: {"response_text": "{}"},
            )


class StatusContractTests(unittest.TestCase):
    def test_sync_package_manifest_records_strategy_and_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "campaign")
            build_package(package_dir, count=1)
            manifest = fr.load_campaign_package(package_dir)["manifest"]
            self.assertEqual(manifest["execution_strategy"], "sync")
            self.assertEqual(manifest["input_jsonl_path"], "")
            self.assertEqual(
                manifest["final_review_settings"]["execution_strategy"],
                "sync",
            )

    def test_status_exposes_execution_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "campaign")
            build_package(package_dir, count=1)
            status = fr.collect_campaign_status(package_dir)
            self.assertEqual(
                status["execution_strategy"],
                fr.EXECUTION_STRATEGY_SYNC,
            )
            self.assertIn(
                "execution: sync",
                fr.format_status_text(status),
            )


class RunFinalReviewRunSyncCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _sync_section(self) -> tuple[dict, str]:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="OpenAI",
            adapter="openai_compatible",
            provider="openai",
            base_url="https://api.example/v1",
            credential_kind="none",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="Main",
            provider_id=provider_id,
            model="gpt-4.1-mini",
        )
        profile_id = editor.profile_ids(section)[0]
        section = editor.set_defaults(
            section,
            primary_profile_id=profile_id,
            execution_strategy="sync",
        )
        section = editor.set_route(
            section,
            "final_review",
            enabled=True,
            profile_id=profile_id,
            strategy="sync",
        )
        return section, profile_id

    def _package_with_routing(self) -> tuple[str, str]:
        package_dir = os.path.join(self.tmp.name, "campaign")
        build_package(package_dir, count=1)
        section, profile_id = self._sync_section()
        plan = routing_reader.read_routing_plan({"model_routing": section})
        package = fr.load_campaign_package(package_dir)
        manifest = dict(package["manifest"])
        manifest["model_routing"] = plan.to_manifest_dict()
        fr.write_campaign_package(
            package_dir,
            manifest=manifest,
            snapshot=dict(package["snapshot"]),
            units=list(package["units"]),
            findings=list(package["findings"]),
        )
        return package_dir, profile_id

    def test_run_final_review_run_sync_ingests_units(self) -> None:
        package_dir, profile_id = self._package_with_routing()
        context = {
            "context_digest": "live-context",
            "snapshot_digest": "live-snapshot",
            "prompt_context": {},
        }
        with (
            mock.patch.object(
                batch,
                "_collect_final_review_context_snapshot",
                return_value=context,
            ),
            mock.patch.object(
                batch,
                "run_sync_request",
                return_value={
                    "response_text": json.dumps({"findings": []}),
                    "provider": "fake",
                    "model": "gpt-4.1-mini",
                    "usage_metadata": {"total_tokens": 4},
                },
            ),
            mock.patch.object(batch, "remember_latest_manifest"),
        ):
            result = batch.run_final_review_run_sync(package_dir)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["profile_id"], profile_id)
        self.assertEqual(result["done_delta"], 1)
        self.assertEqual(result["execution_strategy"], "sync")
        loaded = fr.load_campaign_package(package_dir)
        self.assertEqual(loaded["units"][0]["status"], fr.STATUS_DONE)

    def test_batch_only_commands_reject_sync_package(self) -> None:
        package_dir, _profile_id = self._package_with_routing()
        manifest_path = fr.load_campaign_package(package_dir)["paths"]["manifest"]

        with self.assertRaises(cli_contract.MachineContractError) as resume_error:
            batch.run_final_review_resume(manifest_path)
        self.assertEqual(resume_error.exception.code_name, "FINAL_REVIEW_USE_RUN_SYNC")

        with self.assertRaises(cli_contract.MachineContractError) as ingest_error:
            batch.run_final_review_ingest_results(manifest_path)
        self.assertEqual(ingest_error.exception.code_name, "FINAL_REVIEW_USE_RUN_SYNC")

    def test_run_sync_rejects_batch_package_with_stable_code(self) -> None:
        package_dir = os.path.join(self.tmp.name, "batch-campaign")
        build_package(package_dir, count=1, strategy=fr.EXECUTION_STRATEGY_GEMINI_BATCH)
        manifest_path = fr.load_campaign_package(package_dir)["paths"]["manifest"]

        with self.assertRaises(cli_contract.MachineContractError) as captured:
            batch.run_final_review_run_sync(manifest_path)

        self.assertEqual(captured.exception.code_name, "FINAL_REVIEW_NOT_SYNC")

    def test_aborted_sync_is_not_retryable_for_systemic_categories(self) -> None:
        package_dir, _profile_id = self._package_with_routing()
        context = {
            "context_digest": "live-context",
            "snapshot_digest": "live-snapshot",
            "prompt_context": {},
        }
        with (
            mock.patch.object(
                batch,
                "_collect_final_review_context_snapshot",
                return_value=context,
            ),
            mock.patch.object(
                batch,
                "run_sync_request",
                side_effect=SyncBackendError("authentication"),
            ),
            mock.patch.object(batch, "remember_latest_manifest"),
            mock.patch.object(batch, "print_banner"),
        ):
            with self.assertRaises(cli_contract.MachineContractError) as captured:
                batch.run_final_review_run_sync(package_dir)

        self.assertEqual(captured.exception.code_name, "FINAL_REVIEW_SYNC_ABORTED")
        self.assertFalse(captured.exception.retryable)

    def test_sync_campaign_missing_frozen_plan_is_rejected(self) -> None:
        package_dir, _profile_id = self._package_with_routing()
        package = fr.load_campaign_package(package_dir)
        manifest = dict(package["manifest"])
        manifest.pop("model_routing", None)
        fr.write_campaign_package(
            package_dir,
            manifest=manifest,
            snapshot=dict(package["snapshot"]),
            units=list(package["units"]),
            findings=list(package["findings"]),
        )

        with self.assertRaises(cli_contract.MachineContractError) as captured:
            batch.run_final_review_run_sync(package_dir)

        self.assertEqual(
            captured.exception.code_name,
            "FINAL_REVIEW_PLAN_MISSING",
        )

    def test_negative_limit_returns_stable_usage_error(self) -> None:
        package_dir, _profile_id = self._package_with_routing()

        with self.assertRaises(cli_contract.MachineContractError) as captured:
            batch.run_final_review_run_sync(package_dir, limit=-1)

        self.assertEqual(
            captured.exception.code_name,
            "FINAL_REVIEW_LIMIT_INVALID",
        )
        self.assertEqual(
            captured.exception.semantic_exit_code,
            cli_contract.EXIT_USAGE,
        )

    def test_machine_envelope_exposes_sync_result_fields(self) -> None:
        result = {
            "status": "completed",
            "package_dir": "C:/tmp/review",
            "manifest_path": "C:/tmp/review/manifest.json",
            "execution_strategy": "sync",
            "profile_id": "openai-sync",
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "run_count": 2,
            "skip_count": 1,
            "deferred_count": 0,
            "done_delta": 2,
            "failed_delta": 0,
            "finding_count": 3,
            "to_run_unit_ids": ["u1", "u2"],
            "dry_run": False,
            "limit": 0,
            "campaign_status": {"status": "done"},
        }
        envelope = batch.build_machine_success_envelope(
            "final-review-run-sync",
            result,
            mock.Mock(),
        )

        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "completed")
        payload = envelope["result"]
        self.assertEqual(payload["execution_strategy"], "sync")
        self.assertEqual(payload["profile_id"], "openai-sync")
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["skip_count"], 1)
        self.assertEqual(payload["done_delta"], 2)
        self.assertEqual(payload["finding_count"], 3)
        self.assertEqual(payload["campaign_status"], {"status": "done"})

    def test_route_profile_without_strategy_keeps_batch_behavior(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="Gemini Batch",
            provider_id=provider_id,
            model="gemini-3.5-flash",
        )
        profile_id = editor.profile_ids(section)[0]
        section = editor.set_defaults(
            section,
            primary_profile_id=profile_id,
            execution_strategy="gemini_batch",
        )
        section["legacy_entrypoints"] = {"batch_profile_id": profile_id}
        section["routes"] = {
            "project_analysis": {"profile_id": profile_id, "strategy": "sync"},
            "final_review": {"profile_id": profile_id},
        }

        with runtime.runtime_config_scope(
            runtime.RuntimeConfig(model_routing_config=section)
        ):
            plan = batch.freeze_final_review_routing_plan()

        self.assertEqual(plan.routes["final_review"].strategy.value, "gemini_batch")
        self.assertEqual(plan.routes["final_review"].profile_id, profile_id)

    def test_missing_final_review_route_does_not_follow_sync_default(self) -> None:
        section, _profile_id = self._sync_section()
        section = editor.set_route(section, "final_review", enabled=False)
        self.assertNotIn("final_review", section.get("routes", {}))

        with runtime.runtime_config_scope(
            runtime.RuntimeConfig(model_routing_config=section)
        ):
            with self.assertRaises(cli_contract.MachineContractError):
                batch.freeze_final_review_routing_plan()

    def test_freeze_final_review_plan_accepts_v1_sync_without_legacy_entrypoints(self) -> None:
        section, _profile_id = self._sync_section()
        self.assertNotIn("legacy_entrypoints", section)

        with runtime.runtime_config_scope(
            runtime.RuntimeConfig(model_routing_config=section)
        ):
            plan = batch.freeze_final_review_routing_plan()

        self.assertEqual(plan.routes["final_review"].strategy.value, "sync")

    def test_run_sync_request_direct_adapter_skips_gemini_client(self) -> None:
        section, _profile_id = self._sync_section()
        plan = routing_reader.read_routing_plan({"model_routing": section})
        route = plan.routes["final_review"]

        class FakeBackend:
            def generate(self, request):
                return SyncGenerationResult(
                    provider="openai",
                    model=request.model,
                    execution_mode="sync",
                    response_payload={"ok": True},
                    response_text='{"findings": []}',
                )

        with (
            mock.patch.object(
                batch.model_profile,
                "build_sync_backend",
                return_value=FakeBackend(),
            ) as factory,
            mock.patch.object(
                batch,
                "create_batch_client",
                side_effect=AssertionError("direct adapter must not create a Gemini client"),
            ),
        ):
            raw = batch.run_sync_request(
                {"contents": "review this", "generation_config": {}},
                route,
                plan=plan,
            )

        self.assertEqual(raw["response_text"], '{"findings": []}')
        self.assertEqual(raw["provider"], "openai")
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
