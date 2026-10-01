"""#539: standalone keyword-candidate open + review entry (关键词 / 术语).

The tests drive the real coordinator wiring (``MainWindow``) against an original
offline fixture: a tiny Ren'Py script, an external ``keyword_candidates.jsonl``
and a project ``glossary.json``. They assert real glossary/script file results
instead of only checking whether a button is enabled.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDialog

    from gui_qt.app import MainWindow
    from gui_qt.doctor_report import DoctorSummary
    from gui_qt.keyword_merge_dialog import KeywordMergeDialog
    from gui_qt.user_copy import KEYWORD_CANDIDATE_COPY
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    QDialog = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    KeywordMergeDialog = None  # type: ignore[assignment,misc]
    DoctorSummary = None  # type: ignore[assignment,misc]
    KEYWORD_CANDIDATE_COPY = None  # type: ignore[assignment,misc]
    WorkMode = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support

_EMPTY_GLOSSARY = {
    "preserve_terms": [],
    "non_translatable": [],
    "normalize_map": {},
}
_DEFAULT_CANDIDATES = (
    {
        "source": "Magic Academy",
        "suggested_target": "魔法学院",
        "category": "organization",
        "confidence": 0.92,
    },
    {
        "source": "Luna",
        "suggested_target": "露娜",
        "category": "character",
        "confidence": 0.88,
    },
    {
        "source": "Eldoria",
        "suggested_target": "艾尔多利亚",
        "category": "place",
        "confidence": 0.81,
    },
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scripted_review_dialog(
    *,
    created: list,
    checked_rows: tuple[int, ...] | None = None,
    overwrite: bool = False,
    mutate=None,
):
    """Real review dialog with a scripted user: adjust checks, then confirm.

    ``exec`` mirrors a real modal review: the coordinator keeps ownership of the
    merge service and only sees the dialog result.
    """

    class _ScriptedReviewDialog(KeywordMergeDialog):
        def exec(self) -> object:  # noqa: A003 - Qt API name
            created.append(self)
            if checked_rows is not None:
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 0)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if row in checked_rows
                        else Qt.CheckState.Unchecked
                    )
            self.overwrite_check.setChecked(overwrite)
            if mutate is not None:
                mutate()
            self._on_write()
            if self.result is not None and self.result.summary.wrote_glossary:
                return QDialog.DialogCode.Accepted
            return QDialog.DialogCode.Rejected

    return _ScriptedReviewDialog


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiKeywordCandidateEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        cls._app = app if app is not None else QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        temp = tempfile.TemporaryDirectory(prefix="rtl-539-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

        self.game_root = self.root / "work"
        (self.game_root / "game").mkdir(parents=True)
        self.script_path = self.game_root / "game" / "script.rpy"
        self.script_path.write_text(
            'label start:\n    luna "Hello from the academy."\n',
            encoding="utf-8",
        )
        self.glossary_path = self.game_root / "glossary.json"
        self._write_glossary(_EMPTY_GLOSSARY)

        # External candidate file: outside the project on purpose, like a file
        # produced elsewhere and opened through 打开候选文件.
        self.candidates_path = self.root / "keyword_candidates.jsonl"
        _write_jsonl(self.candidates_path, list(_DEFAULT_CANDIDATES))

        self._active_root = self.game_root
        self.window.state.get_game_root = lambda: str(self._active_root)  # type: ignore[method-assign]
        # Hermetic: no developer config, no real manifest history lookup.
        for patcher in (
            mock.patch.object(self.window.state, "load_translator_config", return_value={}),
            mock.patch.object(
                self.window.state,
                "get_latest_manifest_path_for_mode",
                return_value=None,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="可以提取关键词。",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.page = self.window.keywords_page
        self.window._sync_keywords_page_controls()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    # --- fixture helpers -------------------------------------------------

    def _use_project(self, root: Path | str) -> None:
        self._active_root = str(root)

    def _write_glossary(self, data: dict, path: Path | None = None) -> None:
        target = path or self.glossary_path
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_glossary(self, path: Path | None = None) -> dict:
        target = path or self.glossary_path
        return json.loads(target.read_text(encoding="utf-8"))

    def _backups(self) -> list[Path]:
        return sorted(self.game_root.glob("glossary.json.bak-*"))

    def _open_candidate(self, path: Path | None = None) -> None:
        candidates = path or self.candidates_path
        with mock.patch(
            "gui_qt.app.QFileDialog.getOpenFileName",
            return_value=(str(candidates), ""),
        ):
            self.page.open_candidates_btn.click()

    # --- entry reachability (#539 root cause) ----------------------------

    def test_empty_page_exposes_standalone_open_entry(self) -> None:
        """Without any extraction result the open entry is reachable and usable."""
        self.assertEqual(self.window._resolve_keyword_merge_candidates_path(), "")
        self.assertIs(self.page.page_stack.currentWidget(), self.page.content_page)
        self.assertTrue(self.page.open_candidates_btn.isEnabled())
        self.assertFalse(self.page.open_candidates_btn.isHidden())
        self.assertEqual(
            self.page.open_candidates_btn.text(),
            KEYWORD_CANDIDATE_COPY["open_action"],
        )
        self.assertEqual(
            self.page.merge_btn.text(),
            KEYWORD_CANDIDATE_COPY["merge_action"],
        )
        self.assertFalse(self.page.merge_btn.isEnabled())
        self.assertIn(
            KEYWORD_CANDIDATE_COPY["open_action"],
            self.page.result_hint.text(),
        )

    def test_merge_without_candidate_explains_instead_of_picking(self) -> None:
        """The merge action must not be the hidden file picker anymore (#539)."""
        with mock.patch(
            "gui_qt.app.QFileDialog.getOpenFileName",
            return_value=(str(self.candidates_path), ""),
        ) as picker, mock.patch(
            "gui_qt.app.message_box_information",
        ) as info:
            self.window._on_open_keyword_merge()

        picker.assert_not_called()
        info.assert_called_once()
        self.assertEqual(
            info.call_args.args[2],
            KEYWORD_CANDIDATE_COPY["merge_no_candidates"],
        )
        self.assertIsNone(self.window._keyword_candidate_selection)

    # --- open → inspect → partial check → merge --------------------------

    def test_open_shows_source_count_and_target_then_enables_review(self) -> None:
        script_before = self.script_path.read_bytes()
        glossary_before = self.glossary_path.read_bytes()

        self._open_candidate()

        info = self.page.candidate_info_label.text()
        self.assertFalse(self.page.candidate_info_label.isHidden())
        self.assertIn(KEYWORD_CANDIDATE_COPY["source_external"], info)
        self.assertIn(KEYWORD_CANDIDATE_COPY["format_name"], info)
        self.assertIn(str(self.candidates_path), info)
        self.assertIn("3 条（可审核 3 条）", info)
        self.assertIn(str(self.glossary_path), info)
        self.assertEqual(
            self.window._resolve_keyword_merge_candidates_path(),
            str(self.candidates_path),
        )
        self.assertTrue(self.page.merge_btn.isEnabled())
        # Opening only reads: neither the glossary nor the game script changes.
        self.assertEqual(self.glossary_path.read_bytes(), glossary_before)
        self.assertEqual(self.script_path.read_bytes(), script_before)
        self.assertEqual(self._backups(), [])

    def test_cancelled_file_picker_keeps_loaded_candidate(self) -> None:
        self._open_candidate()
        with mock.patch(
            "gui_qt.app.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ):
            self.page.open_candidates_btn.click()

        self.assertIsNotNone(self.window._keyword_candidate_selection)
        self.assertEqual(
            self.window._resolve_keyword_merge_candidates_path(),
            str(self.candidates_path),
        )
        self.assertTrue(self.page.merge_btn.isEnabled())

    def test_partial_selection_merges_only_checked_entries(self) -> None:
        script_before = self.script_path.read_bytes()
        self._open_candidate()

        created: list = []
        dialog_cls = _scripted_review_dialog(created=created, checked_rows=(0, 2))
        with mock.patch("gui_qt.app.KeywordMergeDialog", dialog_cls), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_question",
            return_value="yes",
        ):
            self.page.merge_btn.click()

        self.assertEqual(len(created), 1)
        data = self._read_glossary()
        self.assertEqual(
            data["normalize_map"],
            {"Magic Academy": "魔法学院", "Eldoria": "艾尔多利亚"},
        )
        # The unchecked candidate is never written.
        self.assertNotIn("Luna", data["normalize_map"])
        self.assertEqual(len(self._backups()), 1)
        # The game script is untouched by a glossary merge.
        self.assertEqual(self.script_path.read_bytes(), script_before)

    def test_review_cancel_writes_nothing(self) -> None:
        self._open_candidate()
        with mock.patch.object(
            KeywordMergeDialog,
            "exec",
            return_value=QDialog.DialogCode.Rejected,
        ):
            self.page.merge_btn.click()

        self.assertEqual(self._read_glossary(), _EMPTY_GLOSSARY)
        self.assertEqual(self._backups(), [])

    def test_preview_only_writes_nothing(self) -> None:
        self._open_candidate()
        created: list = []

        class _PreviewOnlyDialog(KeywordMergeDialog):
            def exec(self) -> object:  # noqa: A003 - Qt API name
                created.append(self)
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 0)
                    if item is not None:
                        item.setCheckState(Qt.CheckState.Checked)
                self._on_preview()
                return QDialog.DialogCode.Rejected

        with mock.patch("gui_qt.app.KeywordMergeDialog", _PreviewOnlyDialog):
            self.page.merge_btn.click()

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].result is not None)
        self.assertTrue(created[0].result.dry_run)
        self.assertEqual(self._read_glossary(), _EMPTY_GLOSSARY)
        self.assertEqual(self._backups(), [])

    # --- invalid / empty inputs ------------------------------------------

    def test_invalid_candidate_file_cannot_enable_merge(self) -> None:
        broken = self.root / "broken_candidates.jsonl"
        broken.write_text("{not json}\n", encoding="utf-8")

        with mock.patch(
            "gui_qt.app.QFileDialog.getOpenFileName",
            return_value=(str(broken), ""),
        ), mock.patch("gui_qt.app.message_box_warning") as warning:
            self.page.open_candidates_btn.click()

        warning.assert_called_once()
        self.assertEqual(
            warning.call_args.args[1],
            KEYWORD_CANDIDATE_COPY["invalid_title"],
        )
        self.assertIsNone(self.window._keyword_candidate_selection)
        self.assertFalse(self.page.merge_btn.isEnabled())
        self.assertEqual(self._read_glossary(), _EMPTY_GLOSSARY)

    def test_empty_candidate_file_cannot_enable_merge(self) -> None:
        empty = self.root / "empty_candidates.jsonl"
        empty.write_text("\n", encoding="utf-8")

        with mock.patch(
            "gui_qt.app.QFileDialog.getOpenFileName",
            return_value=(str(empty), ""),
        ), mock.patch("gui_qt.app.message_box_information") as info:
            self.page.open_candidates_btn.click()

        info.assert_called_once()
        self.assertEqual(
            info.call_args.args[1],
            KEYWORD_CANDIDATE_COPY["empty_title"],
        )
        self.assertIsNone(self.window._keyword_candidate_selection)
        self.assertFalse(self.page.merge_btn.isEnabled())
        self.assertEqual(self._read_glossary(), _EMPTY_GLOSSARY)

    # --- conflict + overwrite confirmation -------------------------------

    def _conflicting_project(self) -> None:
        glossary = dict(_EMPTY_GLOSSARY)
        glossary["normalize_map"] = {"Magic Academy": "魔法学园"}
        self._write_glossary(glossary)

    def test_conflict_is_skipped_without_explicit_overwrite(self) -> None:
        self._conflicting_project()
        self._open_candidate()
        created: list = []

        with mock.patch(
            "gui_qt.app.KeywordMergeDialog",
            _scripted_review_dialog(created=created, checked_rows=(0,)),
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_question",
            return_value="yes",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_information",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_warning",
        ) as warning:
            self.page.merge_btn.click()

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], "没有可写入项")
        data = self._read_glossary()
        # The conflicting entry keeps its existing translation.
        self.assertEqual(data["normalize_map"]["Magic Academy"], "魔法学园")
        self.assertEqual(self._backups(), [])

    def test_conflict_overwrites_only_with_explicit_opt_in(self) -> None:
        self._conflicting_project()
        self._open_candidate()
        created: list = []

        with mock.patch(
            "gui_qt.app.KeywordMergeDialog",
            _scripted_review_dialog(
                created=created,
                checked_rows=(0,),
                overwrite=True,
            ),
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_question",
            return_value="yes",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_information",
        ):
            self.page.merge_btn.click()

        data = self._read_glossary()
        self.assertEqual(data["normalize_map"]["Magic Academy"], "魔法学院")
        self.assertEqual(len(self._backups()), 1)

    # --- context isolation ------------------------------------------------

    def test_switching_candidate_file_never_reuses_previous_review(self) -> None:
        self._open_candidate()
        second = self.root / "second_keyword_candidates.jsonl"
        _write_jsonl(
            second,
            [
                {
                    "source": "Orin",
                    "suggested_target": "奥林",
                    "category": "character",
                    "confidence": 0.9,
                }
            ],
        )
        self._open_candidate(second)
        self.assertEqual(
            self.window._resolve_keyword_merge_candidates_path(),
            str(second),
        )

        created: list = []
        with mock.patch(
            "gui_qt.app.KeywordMergeDialog",
            _scripted_review_dialog(created=created, checked_rows=(0,)),
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_question",
            return_value="yes",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_information",
        ):
            self.page.merge_btn.click()

        dialog = created[0]
        offered = {
            dialog.table.item(row, 1).text()
            for row in range(dialog.table.rowCount())
        }
        self.assertEqual(offered, {"Orin"})
        data = self._read_glossary()
        self.assertEqual(data["normalize_map"], {"Orin": "奥林"})

    def test_project_switch_clears_opened_candidate(self) -> None:
        self._open_candidate()
        self.assertTrue(self.page.merge_btn.isEnabled())

        other_root = self.root / "other_project"
        (other_root / "game").mkdir(parents=True)

        def _fake_set_game_root(path):
            self._use_project(Path(path))
            return Path(path), False

        with (
            mock.patch.object(
                self.window.state,
                "set_game_root",
                side_effect=_fake_set_game_root,
            ),
            mock.patch.object(self.window.runner, "is_running", return_value=False),
            mock.patch.object(self.window, "_is_doctor_running", return_value=False),
            mock.patch.object(self.window, "_load_config_to_ui"),
            mock.patch.object(self.window, "_refresh_diagnostics_context"),
            mock.patch.object(self.window, "_invalidate_manifest_caches"),
            mock.patch.object(self.window, "_apply_work_mode_ui"),
        ):
            self.assertTrue(self.window._switch_game_root(str(other_root)))

        self.assertIsNone(self.window._keyword_candidate_selection)
        self.assertEqual(self.window._resolve_keyword_merge_candidates_path(), "")
        self.assertFalse(self.page.merge_btn.isEnabled())
        self.assertEqual(self.page.candidate_info_label.text(), "")
        self.assertTrue(self.page.candidate_info_label.isHidden())

    def test_project_change_drops_candidate_without_reuse(self) -> None:
        """Even without the explicit reset, a stale selection is never reused."""
        self._open_candidate()
        other_root = self.root / "other_project"
        (other_root / "game").mkdir(parents=True)
        self._use_project(other_root)

        self.assertEqual(self.window._resolve_keyword_merge_candidates_path(), "")
        self.assertIsNone(self.window._keyword_candidate_selection)
        self.window._sync_keywords_page_controls()
        self.assertFalse(self.page.merge_btn.isEnabled())

    def test_late_review_write_is_refused_after_target_change(self) -> None:
        self._open_candidate()
        other_root = self.root / "other_project"
        (other_root / "game").mkdir(parents=True)

        created: list = []
        with mock.patch(
            "gui_qt.app.KeywordMergeDialog",
            _scripted_review_dialog(
                created=created,
                checked_rows=(0,),
                mutate=lambda: self._use_project(other_root),
            ),
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_question",
            return_value="yes",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_information",
        ), mock.patch(
            "gui_qt.keyword_merge_dialog.message_box_warning",
        ) as warning:
            self.page.merge_btn.click()

        self.assertEqual(len(created), 1)
        warning.assert_called_once()
        self.assertEqual(
            warning.call_args.args[1],
            KEYWORD_CANDIDATE_COPY["stale_title"],
        )
        self.assertEqual(
            warning.call_args.args[2],
            KEYWORD_CANDIDATE_COPY["stale_project"],
        )
        self.assertEqual(self._read_glossary(), _EMPTY_GLOSSARY)
        self.assertEqual(self._backups(), [])
        self.assertFalse((other_root / "glossary.json").exists())

    # --- running / project gate limits -----------------------------------

    def test_running_task_blocks_open_entry_with_reason(self) -> None:
        self.window._set_task_running(True)
        self.window._sync_keywords_page_controls()

        self.assertFalse(self.page.open_candidates_btn.isEnabled())
        self.assertEqual(
            self.page.open_candidates_btn.toolTip(),
            KEYWORD_CANDIDATE_COPY["open_running"],
        )
        with mock.patch("gui_qt.app.QFileDialog.getOpenFileName") as picker, mock.patch(
            "gui_qt.app.message_box_information",
        ) as info:
            self.page.open_candidates_btn.click()
            self.window._on_open_keyword_candidates()
        picker.assert_not_called()
        info.assert_called_once()
        self.assertEqual(
            info.call_args.args[2],
            KEYWORD_CANDIDATE_COPY["open_running"],
        )

        self.window._set_task_running(False)
        self.window._sync_keywords_page_controls()
        self.assertTrue(self.page.open_candidates_btn.isEnabled())

    def test_entry_blocked_until_project_check_passes(self) -> None:
        self.window._doctor_check_completed = False
        ready, reason = self.window._keyword_candidate_open_state(running=False)
        self.assertFalse(ready)
        self.assertEqual(reason, KEYWORD_CANDIDATE_COPY["open_project_not_ready"])

        self.window._doctor_check_completed = True
        self._use_project("")
        ready, reason = self.window._keyword_candidate_open_state(running=False)
        self.assertFalse(ready)
        self.assertEqual(reason, KEYWORD_CANDIDATE_COPY["open_no_project"])

    # --- existing extraction-driven path (#539 requirement 6) -------------

    def test_extraction_candidate_path_still_reviews_and_merges(self) -> None:
        """A batch/sync extraction result keeps its own end-to-end merge."""
        extracted = self.game_root / "keyword_candidates.jsonl"
        _write_jsonl(extracted, list(_DEFAULT_CANDIDATES))
        manifest = {
            "mode": "keyword_extraction",
            "keyword_export": {"jsonl_path": str(extracted)},
        }
        script_before = self.script_path.read_bytes()

        with mock.patch.object(
            self.window,
            "_latest_keyword_extraction_manifest",
            return_value=(str(self.game_root / "manifest.json"), manifest),
        ):
            self.window._sync_keywords_page_controls()
            self.assertEqual(
                self.window._resolve_keyword_merge_candidates_path(),
                str(extracted),
            )
            self.assertTrue(self.page.merge_btn.isEnabled())

            created: list = []
            with mock.patch(
                "gui_qt.app.KeywordMergeDialog",
                _scripted_review_dialog(created=created, checked_rows=(1,)),
            ), mock.patch(
                "gui_qt.keyword_merge_dialog.message_box_question",
                return_value="yes",
            ):
                self.page.merge_btn.click()

        self.assertEqual(len(created), 1)
        self.assertEqual(self._read_glossary()["normalize_map"], {"Luna": "露娜"})
        self.assertEqual(self.script_path.read_bytes(), script_before)

    def test_new_extraction_result_supersedes_manual_pick(self) -> None:
        self._open_candidate()
        extracted = self.game_root / "keyword_candidates.jsonl"
        _write_jsonl(extracted, list(_DEFAULT_CANDIDATES))

        self.window._remember_keyword_merge_candidates(str(extracted))

        self.assertIsNone(self.window._keyword_candidate_selection)
        self.assertEqual(
            self.window._resolve_keyword_merge_candidates_path(),
            str(extracted),
        )


if __name__ == "__main__":
    unittest.main()
