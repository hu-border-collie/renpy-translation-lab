import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project_analysis as pa
from gui_qt.project_analysis_workflow import (
    ProjectAnalysisWorkflow,
    discover_keyword_summary_path,
)
from gui_qt.work_modes import WorkMode, workbench_nav_for_work_mode, WorkbenchNavItem
from gui_qt.workflow_factory import create_workflow

try:
    from gui_qt.project_analysis_review_dialog import (
        build_project_analysis_review_data,
    )
except ImportError as exc:
    build_project_analysis_review_data = None  # type: ignore[assignment]
    REVIEW_IMPORT_ERROR = exc
else:
    REVIEW_IMPORT_ERROR = None


class ProjectAnalysisWorkflowTests(unittest.TestCase):
    def test_full_sequence_is_ingest_build_generate(self):
        workflow = ProjectAnalysisWorkflow.start_new(
            keyword_summary_path="C:/reports/keyword_chunk_summaries.jsonl"
        )

        self.assertEqual(
            workflow.current_step().args,
            [
                "project-analysis-ingest-keywords",
                "--summary-jsonl",
                "C:/reports/keyword_chunk_summaries.jsonl",
            ],
        )
        self.assertTrue(workflow.complete_current_step(0, "{}").should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["project-analysis-build-structure"],
        )
        self.assertTrue(workflow.complete_current_step(0, "{}").should_continue)
        self.assertEqual(workflow.current_step().args, ["project-analysis-generate"])
        update = workflow.complete_current_step(0, "{}")
        self.assertEqual(update.status, "done")
        self.assertFalse(update.should_continue)

    def test_failed_stage_stops_and_can_restart_from_artifacts(self):
        workflow = ProjectAnalysisWorkflow.start_new(build=True, generate=True)
        update = workflow.complete_current_step(1, "failed")

        self.assertEqual(update.status, "failed")
        self.assertIsNone(workflow.current_step())
        self.assertIn("重新开始", update.message)

    def test_factory_and_navigation_recognize_project_analysis(self):
        workflow = create_workflow(WorkMode.PROJECT_ANALYSIS)

        self.assertIsInstance(workflow, ProjectAnalysisWorkflow)
        self.assertEqual(
            workbench_nav_for_work_mode(WorkMode.PROJECT_ANALYSIS),
            WorkbenchNavItem.CONTEXT,
        )

    def test_discovery_prefers_newest_current_project_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "game" / "work"
            root.mkdir(parents=True)
            package = Path(tmp) / "package"
            package.mkdir()
            manifest_path = package / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            old = package / "keyword_chunk_summaries.jsonl"
            old.write_text("{}\n", encoding="utf-8")
            copied = root.parent / "extracted_keywords" / "keyword_chunk_summaries.jsonl"
            copied.parent.mkdir()
            copied.write_text("{}\n", encoding="utf-8")
            os.utime(old, (1, 1))
            os.utime(copied, (2, 2))

            result = discover_keyword_summary_path(
                game_root=str(root),
                manifest_path=str(manifest_path),
                manifest={},
            )

            self.assertEqual(Path(result), copied)


@unittest.skipIf(
    build_project_analysis_review_data is None,
    f"GUI dependencies unavailable: {REVIEW_IMPORT_ERROR}",
)
class ProjectAnalysisReviewTests(unittest.TestCase):
    def test_review_timestamp_preserves_freshness_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("draft", published=False)
            manifest = pa.empty_manifest(store_dir=tmp)
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF] = {
                "id": "project_brief",
                "status": pa.STATUS_REVIEW_REQUIRED,
                "draft_present": True,
                "published_present": False,
                "lineage": pa.empty_lineage(source_fingerprint="fresh-fp"),
            }
            store.save_manifest(manifest)

            result = pa.mark_project_brief_reviewed(
                tmp,
                reviewed_at="2026-07-24T00:00:00Z",
            )
            saved = store.load_manifest()
            brief = saved["artifacts"][pa.KIND_PROJECT_BRIEF]

            self.assertEqual(result["reviewed_at"], "2026-07-24T00:00:00Z")
            self.assertEqual(brief["status"], pa.STATUS_REVIEW_REQUIRED)
            self.assertEqual(brief["lineage"]["source_fingerprint"], "fresh-fp")
            self.assertEqual(
                brief["lineage"]["reviewed_at"],
                "2026-07-24T00:00:00Z",
            )

    def test_review_data_keeps_full_diff_and_actual_injection_result(self):
        class FakeStore:
            store_dir = "C:/store"

            def load_brief_text(self, *, published):
                return "published\n" if published else "draft\n" + ("x" * 900)

            def load_manifest(self):
                return {
                    "artifacts": {
                        pa.KIND_PROJECT_BRIEF: {
                            "lineage": pa.empty_lineage(reviewed_at="2026-07-24T00:00:00Z")
                        }
                    }
                }

            def load_summaries(self, kind):
                if kind == pa.KIND_LABEL:
                    return [
                        {
                            "id": "label:start",
                            "kind": kind,
                            "status": "draft",
                            "summary": "evidence summary",
                            "source_files": ["script.rpy"],
                            "line_span": [3, 8],
                            "evidence_item_ids": ["item-1"],
                        }
                    ]
                return []

            def load_routes(self):
                return []

        injection = {
            "text": "published",
            "injectable": True,
            "reason": "",
            "diagnostics": "fingerprint=fresh",
            "status": {},
        }
        with (
            patch(
                "gui_qt.project_analysis_review_dialog.resolve_project_analysis_store",
                return_value=FakeStore(),
            ),
            patch(
                "gui_qt.project_analysis_review_dialog.load_injectable_project_brief",
                return_value=injection,
            ) as load_preview,
        ):
            data = build_project_analysis_review_data(
                base_dir="C:/game/work",
                live_fingerprint="fresh",
                inject_enabled=True,
                max_brief_chars=321,
            )

        self.assertGreater(len(data["draft"]), 500)
        self.assertIn("+draft", data["diff"])
        self.assertEqual(data["records"][0]["line_span"], [3, 8])
        self.assertEqual(data["injection"], injection)
        load_preview.assert_called_once_with(
            "C:/store",
            expected_source_fingerprint="fresh",
            max_chars=321,
            enabled=True,
        )


if __name__ == "__main__":
    unittest.main()
