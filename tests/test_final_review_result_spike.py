import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import final_review as fr
import final_review_llm as frl


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "final_review_result_cases.json"
)


def _review_items() -> list[dict]:
    return [
        {
            "id": "id-a",
            "identity_v2": "id-a",
            "file_rel_path": "tl/script.rpy",
            "source": "Hello.",
            "current_translation": "你好。",
        },
        {
            "id": "id-b",
            "identity_v2": "id-b",
            "file_rel_path": "tl/script.rpy",
            "source": "Goodbye.",
            "current_translation": "再见。",
        },
    ]


def _fixture_unit() -> dict:
    items = _review_items()
    items_digest = fr.digest_translation_items(items)
    context_digest = "fixture-context"
    return {
        "unit_id": "fixture-unit",
        "status": fr.STATUS_RUNNING,
        "file_rel_path": "tl/script.rpy",
        "chunk_index": 1,
        "item_ids": [item["id"] for item in items],
        "item_count": len(items),
        "items": items,
        "items_digest": items_digest,
        "input_digest": fr.compute_unit_input_digest(
            item_ids=[item["id"] for item in items],
            items_digest=items_digest,
            context_digest=context_digest,
            model="fixture-model",
            chunk_index=1,
            file_rel_path="tl/script.rpy",
        ),
        "context_digest": context_digest,
        "model": "fixture-model",
        "prompt_schema_version": fr.PROMPT_SCHEMA_VERSION,
        "error": "",
        "finding_count": 0,
        "completed_at": "",
    }


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class ResultFixtureSpikeTests(unittest.TestCase):
    def test_fixed_fixtures_characterize_current_ingest_decisions(self):
        fixture = _load_fixture()
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["unit_item_count"], len(_fixture_unit()["items"]))

        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                row = {"key": "fixture-unit", **case["result_row"]}
                result = frl.ingest_result_rows([_fixture_unit()], [row])
                unit = result["units"][0]

                self.assertEqual(unit["status"], case["current_status"])
                self.assertEqual(
                    unit["finding_count"], case["current_finding_count"]
                )
                current_error = str(unit.get("error") or "")
                if case["current_error_contains"]:
                    self.assertIn(case["current_error_contains"], current_error)
                else:
                    self.assertEqual(current_error, "")

                current_decision = (
                    "accept" if unit["status"] == fr.STATUS_DONE else "reject"
                )
                if current_decision == case["expected_decision"]:
                    assessment = (
                        "correct_accept"
                        if current_decision == "accept"
                        else "correct_reject"
                    )
                else:
                    assessment = (
                        "false_accept"
                        if current_decision == "accept"
                        else "false_reject"
                    )
                self.assertEqual(assessment, case["decision_assessment"])

    def test_candidate_codes_and_evidence_scopes_are_explicit(self):
        fixture = _load_fixture()
        mapped_codes = {case["candidate_code"] for case in fixture["cases"]}
        self.assertEqual(mapped_codes, set(fixture["candidate_codes"]))
        self.assertEqual(
            {
                case["id"]
                for case in fixture["cases"]
                if case["evidence_scope"] == "current_contract"
                if case["decision_assessment"] == "false_accept"
            },
            {
                "schema_missing_reason",
                "duplicate_item",
            },
        )
        self.assertEqual(
            {
                case["id"]
                for case in fixture["cases"]
                if case["evidence_scope"] == "hypothetical_receipt"
            },
            {"reviewed_count_mismatch"},
        )
        self.assertFalse(
            any(
                case["decision_assessment"] == "false_reject"
                and case["evidence_scope"] == "current_contract"
                for case in fixture["cases"]
            )
        )


class TargetedResumeContractTests(unittest.TestCase):
    def test_resume_requeues_only_failed_unit_and_preserves_done_semantics(self):
        items = _review_items()
        snapshot = fr.build_context_snapshot(
            translation_items=items,
            glossary_enabled=False,
        )
        units = fr.build_review_units(
            items,
            chunk_size=1,
            context_digest=snapshot["context_digest"],
            snapshot_digest=snapshot["snapshot_digest"],
            model="fixture-model",
        )
        by_item_id = {unit["item_ids"][0]: unit for unit in units}
        done = fr.mark_unit_done(by_item_id["id-a"], finding_count=1)
        failed = fr.mark_unit_failed(by_item_id["id-b"], "fixture_parse_error")
        done_before = copy.deepcopy(done)

        finding = fr.normalize_finding(
            {
                "identity_v2": "id-a",
                "file_rel_path": done["file_rel_path"],
                "source": done["items"][0]["source"],
                "current_translation": done["items"][0]["current_translation"],
                "finding_type": "terminology",
                "severity": "medium",
                "reason": "fixture finding retained across targeted resume",
            },
            review_unit_id=done["unit_id"],
            review_unit_digest=done["input_digest"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "final_review_spike")
            readiness = fr.evaluate_readiness(
                pending_task_count=0,
                review_item_count=len(items),
            )
            manifest = fr.build_campaign_manifest(
                package_dir=package_dir,
                display_name="issue-309-fixture",
                snapshot=snapshot,
                units=[done, failed],
                readiness=readiness,
                model="fixture-model",
                chunk_size=1,
            )
            fr.write_campaign_package(
                package_dir,
                manifest=manifest,
                snapshot=snapshot,
                units=[done, failed],
                findings=[finding],
            )

            result = frl.prepare_resume_requests(
                package_dir,
                live_context_digest=snapshot["context_digest"],
                model="fixture-model",
            )
            self.assertEqual(result["run_count"], 1)
            self.assertEqual(result["skip_count"], 1)
            self.assertEqual(result["to_run_unit_ids"], [failed["unit_id"]])

            request_rows = [
                json.loads(line)
                for line in (
                    Path(package_dir) / fr.REQUESTS_JSONL_FILENAME
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["key"] for row in request_rows], [failed["unit_id"]]
            )

            package = fr.load_campaign_package(package_dir)
            units_after = {
                unit["unit_id"]: unit for unit in package["units"]
            }
            done_after = units_after[done["unit_id"]]
            for field in (
                "unit_id",
                "status",
                "input_digest",
                "context_digest",
                "items_digest",
                "item_ids",
                "items",
                "finding_count",
                "completed_at",
                "error",
            ):
                with self.subTest(field=field):
                    self.assertEqual(done_after[field], done_before[field])

            self.assertEqual(
                units_after[failed["unit_id"]]["status"], fr.STATUS_RUNNING
            )
            self.assertEqual(len(package["findings"]), 1)
            self.assertEqual(
                package["findings"][0]["finding_id"], finding["finding_id"]
            )


if __name__ == "__main__":
    unittest.main()
