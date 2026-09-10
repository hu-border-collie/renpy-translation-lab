"""Durable Sync GUI page and MainWindow wiring tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import cli_contract

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.sync_translation_workflow import SyncTranslationWorkflow
    from gui_qt.workbench import WorkbenchPageActions
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    SyncTranslationWorkflow = None  # type: ignore[assignment,misc]
    WorkbenchPageActions = None  # type: ignore[assignment,misc]
    WorkMode = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support

RUN_ID = "sync-run-v1-20260910T120000Z"


def snapshot(
    *,
    status: str = "running",
    next_action: str = "resume",
    unknown: int = 0,
) -> dict:
    return {
        "run_id": RUN_ID,
        "run_status": status,
        "revision": 2,
        "plan": {"plan_id": "plan-1", "plan_fingerprint": "abc"},
        "cancellation": {"requested": False},
        "progress": {
            "requests": {
                "total": 4,
                "pending": 1,
                "in_flight": 0,
                "succeeded": 3,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "superseded": 0,
                "outcome_unknown": unknown,
                "cancelled": 0,
            },
            "items": {"expected": 10, "accepted": 9, "unresolved": 1},
            "attempts": {"total": 4, "unknown": 0, "late_ignored": 0},
            "usage": {"known_calls": 4, "total_tokens": 800},
        },
        "next_action": next_action,
    }


def envelope(
    command: str,
    *,
    status: str = "completed",
    result: dict | None = None,
    artifacts: dict | None = None,
) -> str:
    return json.dumps(
        cli_contract.success_envelope(
            command,
            status=status,
            result=result or {},
            artifacts=artifacts or {},
        ),
        ensure_ascii=False,
    )


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str]]] = []

    def run(self, script, args) -> bool:
        self.calls.append((Path(script), list(args)))
        return True


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class DurableSyncPageWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.page = self.window.sync_translation_page

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_resume_and_cancel_buttons_use_their_callbacks(self) -> None:
        calls: list[str] = []
        self.page.set_action_callbacks(
            WorkbenchPageActions(
                resume=lambda: calls.append("resume"),
                cancel=lambda: calls.append("cancel"),
            )
        )
        self.page.set_run_snapshot(snapshot(status="running"))

        self.assertTrue(self.page.resume_btn.isEnabled())
        self.assertTrue(self.page.cancel_btn.isEnabled())
        self.page.resume_btn.click()
        self.page.cancel_btn.click()

        self.assertEqual(calls, ["resume", "cancel"])

    def test_cancel_disabled_for_terminal_or_running_run(self) -> None:
        self.page.set_action_callbacks(
            WorkbenchPageActions(cancel=lambda: None, resume=lambda: None)
        )
        self.page.set_run_snapshot(snapshot(status="completed", next_action="check"))
        self.assertFalse(self.page.cancel_btn.isEnabled())

        self.page.set_run_snapshot(snapshot(status="running"))
        self.assertTrue(self.page.cancel_btn.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.cancel_btn.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.cancel_btn.isEnabled())

    def test_derive_button_tracks_recovery_state(self) -> None:
        calls: list[str] = []
        self.page.set_action_callbacks(
            WorkbenchPageActions(derive=lambda: calls.append("derive"))
        )
        self.page.set_run_snapshot(snapshot(status="running", next_action="resume"))
        self.assertFalse(self.page.derive_btn.isEnabled())

        self.page.set_run_snapshot(snapshot(status="failed", next_action="derive"))
        self.assertTrue(self.page.derive_btn.isEnabled())
        self.page.derive_btn.click()
        self.assertEqual(calls, ["derive"])

        self.page.set_run_snapshot(snapshot(status="completed", next_action="check"))
        self.assertFalse(self.page.derive_btn.isEnabled())

    def test_durable_preview_enables_apply(self) -> None:
        self.page.set_durable_preview(RUN_ID)

        self.assertEqual(self.page.preview_run_id(), RUN_ID)
        self.assertTrue(self.page.apply_btn.isEnabled())
        self.assertIn(RUN_ID, self.page.preview_status.text())

        self.page.clear_preview()
        self.assertEqual(self.page.preview_run_id(), "")
        self.assertFalse(self.page.apply_btn.isEnabled())

    def test_run_status_renders_public_snapshot(self) -> None:
        self.page.set_run_snapshot(snapshot(status="running"))

        text = self.page.run_status_label.text()
        self.assertIn(RUN_ID, text)
        self.assertIn("执行中", text)

    def test_reset_project_clears_run_state(self) -> None:
        self.page.set_run_snapshot(snapshot(status="running"))
        self.page.set_durable_preview(RUN_ID)

        self.page.reset_project()

        self.assertEqual(self.page.run_id(), "")
        self.assertEqual(self.page.preview_run_id(), "")
        self.assertFalse(self.page.cancel_btn.isEnabled())


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class DurableSyncAppWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.runner = _FakeRunner()
        self.window.runner = self.runner
        self.window._doctor_check_completed = True
        self.window._doctor_summary_status = "ready"
        self.window.state.get_game_root = lambda: "C:/game/work"  # type: ignore[method-assign]
        self.window.state.get_batch_script_path = lambda: Path(  # type: ignore[method-assign]
            "C:/tool/gemini_translate_batch.py"
        )
        self.window._confirm_unsaved_config_before_workflow = lambda: True  # type: ignore[method-assign]
        self.window._append_log = lambda _text: None  # type: ignore[method-assign]
        self.window._clear_log_view = lambda: None  # type: ignore[method-assign]
        self.window._show_workbench_log_drawer = lambda: None  # type: ignore[method-assign]
        self.window._refresh_diagnostics_context = lambda: None  # type: ignore[method-assign]
        self.window._sync_work_modes_requiring_api_key = lambda: frozenset()  # type: ignore[method-assign]

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def _finish_step(self, output: str, exit_code: int = 0) -> None:
        self.window._workflow_step_output_lines = [output]
        self.window._on_workflow_step_finished(exit_code)

    def test_start_sync_translation_runs_durable_sync_start(self) -> None:
        self.window._on_start_translation()

        self.assertEqual(len(self.runner.calls), 1)
        script, args = self.runner.calls[0]
        self.assertEqual(script, Path("C:/tool/gemini_translate_batch.py"))
        self.assertEqual(args, ["sync-start", "--output", "json", "--non-interactive"])
        self.assertIsInstance(self.window._workflow, SyncTranslationWorkflow)

    def test_completed_start_chains_check_and_exposes_preview(self) -> None:
        self.window._on_start_translation()
        self._finish_step(
            envelope(
                "sync-start",
                result={
                    **snapshot(status="completed", next_action="check"),
                },
                artifacts={"run_dir": "C:/runs/run1"},
            )
        )

        # should_continue schedules the check step on the next event-loop turn.
        QApplication.processEvents()

        self.assertEqual(len(self.runner.calls), 2)
        self.assertEqual(
            self.runner.calls[1][1],
            ["check", RUN_ID, "--output", "json", "--non-interactive"],
        )
        page = self.window.sync_translation_page
        self.assertEqual(page.run_id(), RUN_ID)
        self.assertIn(RUN_ID, page.run_status_label.text())

        self._finish_step(
            envelope(
                "check",
                status="safe",
                result={
                    "check": {
                        "check_status": "ready",
                        "safety_level": "safe",
                        "writeback_gate": {"decision": "allow"},
                    }
                },
            )
        )

        self.assertEqual(page.preview_run_id(), RUN_ID)
        self.assertTrue(page.apply_btn.isEnabled())
        self.assertFalse(self.window.kill_btn.isEnabled())
        self.assertIn("运行 ID", page.status_section.facts_label.text())

    def test_stopping_local_worker_locates_run_without_resuming(self) -> None:
        self.window._on_start_translation()

        self._finish_step("Traceback: local worker killed\n", exit_code=1)
        QApplication.processEvents()

        # The locator query is read-only: no resume, no cancel, no check.
        self.assertEqual(
            self.runner.calls[1][1],
            ["sync-status", "--latest", "--output", "json", "--non-interactive"],
        )
        commands = [args[0] for _script, args in self.runner.calls]
        self.assertNotIn("sync-resume", commands)
        self.assertNotIn("sync-cancel", commands)

        self._finish_step(
            envelope(
                "sync-status",
                status="running",
                result=snapshot(status="running", next_action="resume"),
            )
        )
        page = self.window.sync_translation_page
        self.assertEqual(page.run_id(), RUN_ID)
        self.assertTrue(page.resume_btn.isEnabled())
        self.assertTrue(page.cancel_btn.isEnabled())
        self.assertIsNone(self.window._workflow)

    def test_resume_button_queries_latest_durable_run(self) -> None:
        self.window._on_resume_durable_sync()

        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(
            self.runner.calls[0][1],
            ["sync-status", "--latest", "--output", "json", "--non-interactive"],
        )

        self._finish_step(
            envelope(
                "sync-status",
                status="running",
                result=snapshot(status="running", next_action="resume"),
            )
        )
        QApplication.processEvents()

        self.assertEqual(
            self.runner.calls[1][1],
            ["sync-resume", RUN_ID, "--output", "json", "--non-interactive"],
        )

    def test_derive_without_unknown_runs_plain_sync_derive(self) -> None:
        self.window.sync_translation_page.set_run_snapshot(
            snapshot(status="failed", next_action="derive")
        )
        with mock.patch("gui_qt.app.message_box_question", return_value="yes"):
            self.window._on_derive_durable_sync()

        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(
            self.runner.calls[0][1],
            ["sync-derive", RUN_ID, "--output", "json", "--non-interactive"],
        )

    def test_derive_with_unknown_uses_explicit_exclude(self) -> None:
        self.window.sync_translation_page.set_run_snapshot(
            snapshot(status="failed", next_action="derive", unknown=2)
        )
        with mock.patch.object(
            self.window,
            "_prompt_derive_options",
            return_value={"exclude_unknown": True},
        ):
            self.window._on_derive_durable_sync()

        self.assertEqual(
            self.runner.calls[0][1],
            [
                "sync-derive",
                RUN_ID,
                "--exclude-unknown",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_resume_requires_the_environment_check(self) -> None:
        self.window._doctor_check_completed = False
        self.window._doctor_summary_status = ""

        with mock.patch("gui_qt.app.message_box_information") as info:
            self.window._on_resume_durable_sync()

        info.assert_called_once()
        self.assertEqual(self.runner.calls, [])

    def test_cancel_not_terminal_cancel_requested_is_reported_as_waiting(self) -> None:
        self.window.sync_translation_page.set_run_snapshot(snapshot(status="running"))
        with mock.patch("gui_qt.app.message_box_question", return_value="yes"):
            self.window._on_cancel_durable_sync()

        self._finish_step(
            envelope(
                "sync-cancel",
                status="cancel_requested",
                result=snapshot(status="cancel_requested", next_action="wait_cancel"),
            )
        )
        self.assertIn("等待", self.window._workflow_heading_text)
        self.assertTrue(self.window.sync_translation_page.cancel_btn.isEnabled())

    def test_cancel_button_confirms_then_runs_sync_cancel(self) -> None:
        self.window.sync_translation_page.set_run_snapshot(snapshot(status="running"))
        with mock.patch(
            "gui_qt.app.message_box_question",
            return_value="yes",
        ):
            self.window._on_cancel_durable_sync()

        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(
            self.runner.calls[0][1],
            ["sync-cancel", RUN_ID, "--output", "json", "--non-interactive"],
        )

    def test_apply_button_runs_durable_apply(self) -> None:
        self.window.sync_translation_page.set_durable_preview(RUN_ID)
        with mock.patch(
            "gui_qt.app.message_box_question",
            return_value="yes",
        ):
            self.window._on_apply_sync_translation()

        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(
            self.runner.calls[0][1],
            ["apply", RUN_ID, "--output", "json", "--non-interactive"],
        )

    def test_apply_without_checked_preview_is_refused(self) -> None:
        with mock.patch("gui_qt.app.message_box_information") as info:
            self.window._on_apply_sync_translation()

        info.assert_called_once()
        self.assertEqual(self.runner.calls, [])


if __name__ == "__main__":
    unittest.main()
