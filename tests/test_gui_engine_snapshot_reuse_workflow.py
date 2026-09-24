"""End-to-end GUI reuse actions through the existing P4 and Batch services."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
import translation_core
from engine_adapters.contracts import Occurrence, OpaqueLocator
from engine_adapters.coverage import digest_json
import engine_adapters.versioning as versioning

from tests import gui_test_support

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from gui_qt.engine_snapshot_dialog import EngineSnapshotDialog
    from gui_qt.engine_snapshot_worker import EngineSnapshotTaskResult
except ImportError as exc:
    QObject = None  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    EngineSnapshotDialog = None  # type: ignore[assignment,misc]
    EngineSnapshotTaskResult = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


if QObject is not None:

    class _InlineTaskWorker(QObject):
        """Run the real task synchronously while keeping the QThread result shape."""

        completed = Signal(object)

        def __init__(self, task, parent=None) -> None:
            super().__init__(parent)
            self._task = task

        def isRunning(self) -> bool:  # noqa: N802 - Qt spelling
            return False

        def start(self) -> None:
            try:
                payload = self._task()
            except Exception as exc:  # noqa: BLE001 - mirror the GUI worker boundary
                result = EngineSnapshotTaskResult(
                    ok=False,
                    error=str(exc),
                    error_code=type(exc).__name__,
                )
            else:
                result = EngineSnapshotTaskResult(ok=True, payload=payload)
            self.completed.emit(result)

        def requestInterruption(self) -> None:  # noqa: N802 - Qt spelling
            return None

        def wait(self, *_args) -> bool:
            return True

else:
    _InlineTaskWorker = object  # type: ignore[assignment,misc]


class _SnapshotFactory:
    """Build small valid P4 artifacts without importing another test case."""

    versioning = versioning

    def _snapshot(self, version: str, specs: list[dict]):
        project_fingerprint = f"project-{version}"
        source_fingerprint = f"source-{version}"
        occurrences = []
        for spec in specs:
            key = str(spec["key"])
            source = str(spec["source"])
            line_number = int(spec.get("line", 1))
            unit = translation_core.TranslationUnit(
                id=f"unit-{version}-{key}",
                mode=translation_core.MODE_TRANSLATION,
                text=source,
                source=source,
                current_translation=f"translated-{version}-{key}",
                file_rel_path=str(spec.get("file", "chapter.rpy")),
                line=max(0, line_number - 1),
                line_number=line_number,
                start=4,
                end=4 + len(source),
            )
            occurrences.append(
                Occurrence(
                    occurrence_id=f"occ-{version}-{key}",
                    engine="renpy",
                    project_snapshot_fingerprint=project_fingerprint,
                    content_fingerprint=digest_json(
                        {"content": spec.get("content", f"{version}-{key}")}
                    ),
                    candidate_id=f"candidate-{version}-{key}",
                    locator=OpaqueLocator(
                        engine="renpy",
                        locator_schema_version=1,
                        locator={"stable_key": spec.get("locator", f"{version}-{key}")},
                    ),
                    unit=unit,
                )
            )

        files = sorted({str(spec.get("file", "chapter.rpy")) for spec in specs})
        coverage = versioning.CoverageBinding.create(
            coverage_digest=f"coverage-{version}",
            coverage_status="ready",
            coverage_schema_version=1,
            inventory_digest=f"inventory-{version}",
            source_fingerprint=source_fingerprint,
            candidate_count=len(specs),
            classification_counts={"translatable": len(specs)},
            review_digest=f"review-{version}",
            review_status="human_reviewed",
            review_policy="agent_or_human",
            review_policy_satisfied=True,
            unresolved_findings=0,
        )
        return versioning.create_project_snapshot(
            versioning.GameVersion(version_id=version),
            engine="renpy",
            adapter_version="test-adapter",
            localization_mode="hybrid",
            target_language="schinese",
            source_fingerprint=source_fingerprint,
            project_snapshot_fingerprint=project_fingerprint,
            source_files=[
                {
                    "file_rel_path": path,
                    "size": 10,
                    "sha256": digest_json({"version": version, "path": path}),
                }
                for path in files
            ],
            coverage=coverage,
            occurrences=occurrences,
            generated_at="2026-08-15T00:00:00+00:00",
        )


@gui_test_support.skip_unless_gui(EngineSnapshotDialog is None, IMPORT_ERROR)
class GuiReuseWorkflowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.game_root = self.root / "game"
        self.tl_dir = self.game_root / "tl" / "schinese"
        self.tl_dir.mkdir(parents=True)
        self.game_file = self.tl_dir / "synthetic.rpy"
        self.game_file.write_text(
            "translate schinese synthetic_route:\n"
            '    e "SYNTHETIC_SOURCE_ALPHA."\n'
            '    e "SYNTHETIC_SOURCE_BETA."\n',
            encoding="utf-8",
        )

        self.old_batch_state = {
            "base_dir": batch.legacy.BASE_DIR,
            "tl_dir": batch.legacy.TL_DIR,
            "include_files": set(batch.legacy.INCLUDE_FILES),
            "include_prefixes": set(batch.legacy.INCLUDE_PREFIXES),
            "log_dir": batch.LOG_DIR,
            "jobs_dir": batch.BATCH_JOBS_DIR,
            "repair_dir": batch.REPAIR_RUNS_DIR,
            "sync_dir": batch.SYNC_RUNS_DIR,
            "latest": batch.LATEST_MANIFEST_FILE,
            "progress": batch.PROGRESS_LOG,
            "snapshots": batch.PROJECT_SNAPSHOTS_DIR,
            "reuse": batch.PROJECT_REUSE_DIR,
            "rag_enabled": batch.RAG_ENABLED,
            "rag_store": batch._RAG_STORE,
            "story_enabled": batch.STORY_MEMORY_ENABLED,
            "story_graph": batch._STORY_GRAPH,
            "story_graph_path": batch._STORY_GRAPH_PATH,
        }
        self.log_dir = self.root / "logs"
        self.jobs_dir = self.log_dir / "batch_jobs"
        batch.legacy.BASE_DIR = str(self.root)
        batch.legacy.TL_DIR = str(self.tl_dir)
        batch.legacy.INCLUDE_FILES = {"synthetic.rpy"}
        batch.legacy.INCLUDE_PREFIXES = set()
        batch.LOG_DIR = str(self.log_dir)
        batch.BATCH_JOBS_DIR = str(self.jobs_dir)
        batch.REPAIR_RUNS_DIR = str(self.log_dir / "repair_runs")
        batch.SYNC_RUNS_DIR = str(self.log_dir / "sync_runs")
        batch.LATEST_MANIFEST_FILE = str(self.jobs_dir / "latest_manifest.txt")
        batch.PROGRESS_LOG = str(self.log_dir / "translation_progress_batch.json")
        batch.PROJECT_SNAPSHOTS_DIR = str(self.log_dir / "project_snapshots")
        batch.PROJECT_REUSE_DIR = str(self.log_dir / "translation_reuse")
        batch.RAG_ENABLED = False
        batch._RAG_STORE = None
        batch.STORY_MEMORY_ENABLED = False
        batch._STORY_GRAPH = None
        batch._STORY_GRAPH_PATH = ""

        self.addCleanup(self._restore_batch_state)
        self.fixture = _SnapshotFactory()

        self._worker_patch = mock.patch(
            "gui_qt.engine_snapshot_dialog.EngineSnapshotTaskWorker",
            _InlineTaskWorker,
        )
        self._worker_patch.start()
        self.addCleanup(self._worker_patch.stop)
        self.dialog = EngineSnapshotDialog(game_root=str(self.game_root), start_tab=2)
        self.addCleanup(self._close_dialog)

    def _restore_batch_state(self) -> None:
        batch.legacy.BASE_DIR = self.old_batch_state["base_dir"]
        batch.legacy.TL_DIR = self.old_batch_state["tl_dir"]
        batch.legacy.INCLUDE_FILES = self.old_batch_state["include_files"]
        batch.legacy.INCLUDE_PREFIXES = self.old_batch_state["include_prefixes"]
        batch.LOG_DIR = self.old_batch_state["log_dir"]
        batch.BATCH_JOBS_DIR = self.old_batch_state["jobs_dir"]
        batch.REPAIR_RUNS_DIR = self.old_batch_state["repair_dir"]
        batch.SYNC_RUNS_DIR = self.old_batch_state["sync_dir"]
        batch.LATEST_MANIFEST_FILE = self.old_batch_state["latest"]
        batch.PROGRESS_LOG = self.old_batch_state["progress"]
        batch.PROJECT_SNAPSHOTS_DIR = self.old_batch_state["snapshots"]
        batch.PROJECT_REUSE_DIR = self.old_batch_state["reuse"]
        batch.RAG_ENABLED = self.old_batch_state["rag_enabled"]
        batch._RAG_STORE = self.old_batch_state["rag_store"]
        batch.STORY_MEMORY_ENABLED = self.old_batch_state["story_enabled"]
        batch._STORY_GRAPH = self.old_batch_state["story_graph"]
        batch._STORY_GRAPH_PATH = self.old_batch_state["story_graph_path"]

    def _close_dialog(self) -> None:
        if getattr(self, "dialog", None) is not None:
            self.dialog.close()
            self.dialog.deleteLater()

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_manifest(path: Path, manifest: dict) -> None:
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_package_items(self, manifest_path: Path) -> tuple[dict, list[dict]]:
        manifest = self._read_manifest(manifest_path)
        items = []
        for chunk in manifest.get("chunks") or []:
            items.extend(chunk.get("items") or [])
        return manifest, items

    def _snapshot_for_manifest(self, version: str, manifest: dict):
        chunks = list(manifest.get("chunks") or [])
        specs = []
        for index, item in enumerate(
            item for chunk in chunks for item in chunk.get("items") or []
        ):
            key = "alpha" if index == 0 else "beta"
            source = str(item.get("source") or item.get("text") or "")
            chunk = next(
                chunk
                for chunk in chunks
                if item in (chunk.get("items") or [])
            )
            unit = translation_core.unit_from_manifest_item(
                item,
                mode=translation_core.MODE_TRANSLATION,
                chunk=chunk,
            )
            specs.append(
                {
                    "key": key,
                    "source": source,
                    "file": str(chunk.get("file_rel_path") or "synthetic.rpy"),
                    "line": int(unit.display_line_number or unit.line + 1),
                    "locator": f"synthetic-locator-{key}",
                    "content": f"synthetic-content-{key}",
                    "unit": unit,
                }
            )

        template = self.fixture._snapshot(
            version,
            [
                {key: value for key, value in spec.items() if key != "unit"}
                for spec in specs
            ],
        )
        occurrences = []
        for spec in specs:
            unit = spec["unit"]
            key = str(spec["key"])
            occurrences.append(
                Occurrence(
                    occurrence_id=f"occ-{version}-{key}",
                    engine="renpy",
                    project_snapshot_fingerprint=template.project_snapshot_fingerprint,
                    content_fingerprint=digest_json(
                        {"content": f"synthetic-content-{key}"}
                    ),
                    candidate_id=f"candidate-{version}-{key}",
                    locator=OpaqueLocator(
                        engine="renpy",
                        locator_schema_version=1,
                        locator={
                            "file_rel_path": spec["file"],
                            "line_hint": unit.display_line_number,
                            "stable_key": f"synthetic-locator-{key}",
                        },
                    ),
                    unit=unit,
                )
            )
        return self.fixture.versioning.create_project_snapshot(
            template.game_version,
            engine=template.engine,
            adapter_version=template.adapter_version,
            localization_mode=template.localization_mode,
            target_language=template.target_language,
            source_fingerprint=template.source_fingerprint,
            project_snapshot_fingerprint=template.project_snapshot_fingerprint,
            source_files=template.source_files,
            coverage=template.coverage,
            occurrences=occurrences,
            generated_at=template.generated_at,
        )

    def _write_result_row(
        self,
        manifest_path: Path,
        manifest: dict,
        translations_by_id: dict[str, str],
    ) -> Path:
        manifest["result_jsonl_path"] = "results.jsonl"
        rows = []
        for chunk in manifest.get("chunks") or []:
            translations = [
                {"id": item["id"], "translation": translations_by_id[item["id"]]}
                for item in chunk.get("items") or []
                if item["id"] in translations_by_id
            ]
            rows.append(
                {
                    "key": chunk["key"],
                    "normalized_response": {"items": translations},
                }
            )
        result_path = manifest_path.parent / "results.jsonl"
        result_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self._write_manifest(manifest_path, manifest)
        return result_path

    def _make_p4_package(self) -> tuple[Path, Path, str, str]:
        base_manifest_path = Path(
            batch.create_batch_package(
                display_name_override="synthetic-reuse-base",
                skip_prepare=True,
            )
        )
        target_package = self.jobs_dir / "synthetic-reuse-target"
        shutil.copytree(base_manifest_path.parent, target_package)
        target_manifest_path = target_package / "manifest.json"
        base_manifest, base_items = self._load_package_items(base_manifest_path)
        target_manifest, target_items = self._load_package_items(target_manifest_path)
        self.assertEqual(len(base_items), 2)
        self.assertEqual(len(target_items), 2)
        self.assertEqual(
            [item["text"] for item in base_items],
            ["SYNTHETIC_SOURCE_ALPHA.", "SYNTHETIC_SOURCE_BETA."],
        )

        long_translation = "原创合成旧译文甲" * 30
        self.alpha_target_id = str(target_items[0]["id"])
        self.beta_target_id = str(target_items[1]["id"])
        self._write_result_row(
            base_manifest_path,
            base_manifest,
            {
                str(base_items[0]["id"]): long_translation,
                str(base_items[1]["id"]): "原创合成旧译文乙。",
            },
        )
        self._write_result_row(
            target_manifest_path,
            target_manifest,
            {self.beta_target_id: "原创已有目标译文乙。"},
        )
        base_snapshot = self._snapshot_for_manifest("1.0", base_manifest)
        target_snapshot = self._snapshot_for_manifest("2.0", target_manifest)

        base_snapshot_paths = self.fixture.versioning.export_project_snapshot(
            base_snapshot,
            self.root / "snapshots" / "base",
        )
        target_snapshot_paths = self.fixture.versioning.export_project_snapshot(
            target_snapshot,
            self.root / "snapshots" / "target",
        )
        records = batch.run_translation_records_export(
            base_snapshot_paths.snapshot_path,
            str(base_manifest_path),
            output_dir=str(self.root / "records"),
        )
        reconciliation = self.fixture.versioning.reconcile_project_snapshots(
            base_snapshot,
            target_snapshot,
        )
        reconciliation_paths = self.fixture.versioning.export_reconciliation_report(
            reconciliation,
            self.root / "reconciliation",
        )
        candidates = batch.run_reuse_candidates_build(
            base_snapshot_paths.snapshot_path,
            target_snapshot_paths.snapshot_path,
            reconciliation_paths.report_path,
            records["paths"]["manifest"],
            output_dir=str(self.root / "reuse-original"),
        )
        self.target_item_ids = {self.alpha_target_id, self.beta_target_id}
        return (
            Path(candidates["paths"]["output_dir"]),
            target_manifest_path,
            long_translation,
            str(candidates["paths"]["report"]),
        )

    def _select_candidate(self, predicate) -> dict:
        for row in range(self.dialog.reuse_table.rowCount()):
            item = self.dialog.reuse_table.item(row, 0)
            candidate_id = str(item.data(256) or "")
            candidate = self.dialog._reuse_candidates_by_id[candidate_id]
            if predicate(candidate):
                self.dialog.reuse_table.selectRow(row)
                return candidate
        self.fail("expected candidate was not present on the current page")

    def test_gui_decisions_export_and_check_apply_use_shared_services(self) -> None:
        original_package, target_manifest_path, long_translation, original_report = (
            self._make_p4_package()
        )
        original_files = {
            path.relative_to(original_package): path.read_bytes()
            for path in original_package.rglob("*")
            if path.is_file()
        }
        original_game_bytes = self.game_file.read_bytes()

        self.dialog.reuse_path_edit.setText(str(original_package))
        self.dialog.load_reuse()
        self.assertEqual(self.dialog.reuse_table.rowCount(), 2)
        accepted_candidate = self._select_candidate(
            lambda item: item.get("reference_translation_full") == long_translation
        )
        self.assertGreater(len(long_translation), 160)
        self.assertIn(long_translation, self.dialog.reuse_detail.toPlainText())
        self.assertEqual(self.dialog.reuse_reviewer_edit.text(), "")

        self.dialog.reuse_reviewer_edit.setText("原创人类审阅者甲")
        self.dialog.reuse_note_edit.setText("原创接受审计备注")
        self.dialog.reuse_accept_btn.click()
        first_decided_package = Path(self.dialog.reuse_path_edit.text())
        self.assertNotEqual(first_decided_package, original_package)
        accepted = self.dialog._reuse_candidates_by_id[accepted_candidate["candidate_id"]]
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["decision"]["reviewer_type"], "human")
        self.assertEqual(accepted["decision"]["reviewer_name"], "原创人类审阅者甲")
        self.assertEqual(accepted["decision"]["note"], "原创接受审计备注")
        self._select_candidate(
            lambda item: item["candidate_id"] == accepted_candidate["candidate_id"]
        )
        self.assertIn("原创接受审计备注", self.dialog.reuse_detail.toPlainText())
        self.assertEqual(
            {
                path.relative_to(original_package): path.read_bytes()
                for path in original_package.rglob("*")
                if path.is_file()
            },
            original_files,
        )
        self.assertEqual(Path(original_report).read_bytes(), original_files[Path("reuse_report.json")])

        rejected_candidate = self._select_candidate(
            lambda item: item["candidate_id"] != accepted_candidate["candidate_id"]
        )
        self.dialog.reuse_reviewer_edit.setText("原创人类审阅者乙")
        self.dialog.reuse_note_edit.setText("原创拒绝审计备注")
        self.dialog.reuse_reject_btn.click()
        self.assertNotEqual(Path(self.dialog.reuse_path_edit.text()), first_decided_package)
        rejected = self.dialog._reuse_candidates_by_id[rejected_candidate["candidate_id"]]
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["decision"]["reviewer_name"], "原创人类审阅者乙")
        self.assertEqual(rejected["decision"]["note"], "原创拒绝审计备注")

        self.dialog.reuse_manifest_edit.setText(str(target_manifest_path))
        self.assertTrue(self.dialog.reuse_export_btn.isEnabled())
        self.dialog.reuse_export_btn.click()

        exported_manifest = self._read_manifest(target_manifest_path)
        result_path = target_manifest_path.parent / exported_manifest["result_jsonl_path"]
        result_rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        translations = {
            item["id"]: item["translation"]
            for row in result_rows
            for item in row["normalized_response"]["translations"]
        }
        self.assertEqual(translations[self.alpha_target_id], long_translation)
        self.assertEqual(translations[self.beta_target_id], "原创已有目标译文乙。")
        self.assertEqual(len(exported_manifest["reuse_export_history"]), 1)
        self.assertIn("check", self.dialog.reuse_status_label.text())
        self.assertEqual(self.game_file.read_bytes(), original_game_bytes)

        with self.assertRaises(SystemExit):
            batch.apply_results(str(target_manifest_path))
        checked = batch.check_results(str(target_manifest_path))
        self.assertEqual(
            checked["last_check_summary"]["writeback_gate"]["decision"],
            "allow",
        )
        batch.apply_results(str(target_manifest_path))
        self.assertNotEqual(self.game_file.read_bytes(), original_game_bytes)
        self.assertIn(long_translation, self.game_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
