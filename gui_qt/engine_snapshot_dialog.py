"""Read-only GUI dialog for engine capabilities, snapshots, diffs and reuse."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .engine_snapshot_actions import (
    collect_engine_snapshot_overview,
    load_reuse_candidates_overview,
    reconcile_snapshots,
)
from .engine_snapshot_worker import EngineSnapshotTaskResult, EngineSnapshotTaskWorker
from .user_copy import ENGINE_SNAPSHOT_COPY, engine_snapshot_label

_EVIDENCE_FLAGS = (
    "source_equal",
    "speaker_equal",
    "context_before_equal",
    "context_after_equal",
    "file_moved",
    "line_changed",
)


class EngineSnapshotDialog(QDialog):
    """Read-only view over P3 snapshots/reconciliation and P4 reuse candidates."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        game_root: str = "",
        start_tab: int = 0,
    ) -> None:
        super().__init__(parent)
        self._game_root = str(game_root or "")
        self._snapshot_root_override = ""
        self._worker: EngineSnapshotTaskWorker | None = None
        self._overview: dict[str, Any] = {}
        self._diff: dict[str, Any] = {}
        self._reuse: dict[str, Any] = {}

        self.setWindowTitle(ENGINE_SNAPSHOT_COPY["dialog_title"])
        self.resize(1040, 700)

        outer = QVBoxLayout(self)
        intro = QLabel(ENGINE_SNAPSHOT_COPY["dialog_intro"])
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("engine_snapshot_tabs")
        self.tabs.addTab(self._build_overview_tab(), ENGINE_SNAPSHOT_COPY["tab_overview"])
        self.tabs.addTab(self._build_diff_tab(), ENGINE_SNAPSHOT_COPY["tab_diff"])
        self.tabs.addTab(self._build_reuse_tab(), ENGINE_SNAPSHOT_COPY["tab_reuse"])
        outer.addWidget(self.tabs, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self.close_btn = QPushButton(ENGINE_SNAPSHOT_COPY["close"])
        self.close_btn.setObjectName("engine_snapshot_close_btn")
        self.close_btn.clicked.connect(self.reject)
        close_row.addWidget(self.close_btn)
        outer.addLayout(close_row)

        index = max(0, min(int(start_tab or 0), self.tabs.count() - 1))
        self.tabs.setCurrentIndex(index)
        self.refresh_overview()

    # -- construction ----------------------------------------------------

    def _build_overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.engine_summary_label = QLabel("")
        self.engine_summary_label.setWordWrap(True)
        self.engine_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.engine_summary_label.setObjectName("engine_snapshot_summary_label")
        layout.addWidget(self.engine_summary_label)
        self.snapshot_root_label = QLabel("")
        self.snapshot_root_label.setWordWrap(True)
        self.snapshot_root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.snapshot_root_label)
        self.snapshot_table = self._make_table(ENGINE_SNAPSHOT_COPY["snapshot_columns"])
        self.snapshot_table.setObjectName("engine_snapshot_table")
        layout.addWidget(self.snapshot_table, 1)
        row = QHBoxLayout()
        self.refresh_overview_btn = QPushButton(ENGINE_SNAPSHOT_COPY["refresh"])
        self.refresh_overview_btn.setObjectName("engine_snapshot_refresh_btn")
        self.refresh_overview_btn.clicked.connect(self.refresh_overview)
        row.addWidget(self.refresh_overview_btn)
        self.overview_status_label = QLabel("")
        self.overview_status_label.setWordWrap(True)
        row.addWidget(self.overview_status_label, 1)
        layout.addLayout(row)
        self._overview_buttons = (self.refresh_overview_btn,)
        return tab

    def _build_diff_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.diff_base_combo = self._make_path_combo("engine_snapshot_diff_base_combo")
        self.diff_target_combo = self._make_path_combo("engine_snapshot_diff_target_combo")
        form.addRow(
            ENGINE_SNAPSHOT_COPY["diff_base"],
            self._combo_row(self.diff_base_combo, lambda: self._choose_snapshot_dir(self.diff_base_combo)),
        )
        form.addRow(
            ENGINE_SNAPSHOT_COPY["diff_target"],
            self._combo_row(self.diff_target_combo, lambda: self._choose_snapshot_dir(self.diff_target_combo)),
        )
        layout.addLayout(form)
        action_row = QHBoxLayout()
        self.diff_run_btn = QPushButton(ENGINE_SNAPSHOT_COPY["diff_run"])
        self.diff_run_btn.setObjectName("engine_snapshot_diff_run_btn")
        self.diff_run_btn.clicked.connect(self.run_diff)
        action_row.addWidget(self.diff_run_btn)
        self.diff_status_label = QLabel("")
        self.diff_status_label.setWordWrap(True)
        action_row.addWidget(self.diff_status_label, 1)
        layout.addLayout(action_row)
        layout.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["diff_summary"]))
        self.diff_summary = QPlainTextEdit()
        self.diff_summary.setReadOnly(True)
        self.diff_summary.setMaximumHeight(130)
        self.diff_summary.setObjectName("engine_snapshot_diff_summary")
        layout.addWidget(self.diff_summary)
        layout.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["diff_items"]))
        self.diff_table = self._make_table(ENGINE_SNAPSHOT_COPY["diff_columns"])
        self.diff_table.setObjectName("engine_snapshot_diff_table")
        layout.addWidget(self.diff_table, 1)
        self._diff_buttons = (self.diff_run_btn,)
        return tab

    def _build_reuse_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["reuse_path"]))
        self.reuse_path_edit = QLineEdit("")
        self.reuse_path_edit.setObjectName("engine_snapshot_reuse_path_edit")
        row.addWidget(self.reuse_path_edit, 1)
        self.reuse_choose_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_choose"])
        self.reuse_choose_btn.setObjectName("engine_snapshot_reuse_choose_btn")
        self.reuse_choose_btn.clicked.connect(self._choose_reuse_dir)
        row.addWidget(self.reuse_choose_btn)
        self.reuse_load_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_load"])
        self.reuse_load_btn.setObjectName("engine_snapshot_reuse_load_btn")
        self.reuse_load_btn.clicked.connect(self.load_reuse)
        row.addWidget(self.reuse_load_btn)
        layout.addLayout(row)
        self.reuse_status_label = QLabel("")
        self.reuse_status_label.setWordWrap(True)
        layout.addWidget(self.reuse_status_label)
        layout.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["reuse_summary"]))
        self.reuse_summary_label = QLabel("")
        self.reuse_summary_label.setWordWrap(True)
        self.reuse_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.reuse_summary_label)
        self.reuse_table = self._make_table(ENGINE_SNAPSHOT_COPY["reuse_columns"])
        self.reuse_table.setObjectName("engine_snapshot_reuse_table")
        layout.addWidget(self.reuse_table, 1)
        self.reuse_open_review_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_open_review"])
        self.reuse_open_review_btn.setObjectName("engine_snapshot_reuse_review_btn")
        self.reuse_open_review_btn.setEnabled(False)
        self.reuse_open_review_btn.clicked.connect(self._open_reuse_review)
        review_row = QHBoxLayout()
        review_row.addWidget(self.reuse_open_review_btn)
        review_row.addStretch(1)
        layout.addLayout(review_row)
        self._reuse_buttons = (
            self.reuse_choose_btn,
            self.reuse_load_btn,
            self.reuse_open_review_btn,
        )
        return tab

    @staticmethod
    def _make_table(columns: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    @staticmethod
    def _make_path_combo(object_name: str) -> QComboBox:
        combo = QComboBox()
        combo.setObjectName(object_name)
        combo.setEditable(False)
        return combo

    @staticmethod
    def _combo_row(combo: QComboBox, choose: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo, 1)
        button = QPushButton(ENGINE_SNAPSHOT_COPY["diff_choose"])
        button.clicked.connect(choose)
        layout.addWidget(button)
        return row

    # -- overview --------------------------------------------------------

    def refresh_overview(self) -> None:
        if self._worker_running():
            return
        self._set_controls_enabled(False)
        self.overview_status_label.setText(ENGINE_SNAPSHOT_COPY["loading"])
        self._start_task(
            lambda: collect_engine_snapshot_overview(
                game_root=self._game_root,
                snapshot_root=self._snapshot_root_override,
            ),
            self._on_overview_loaded,
        )

    def _on_overview_loaded(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            return
        payload = dict(result.payload or {})
        self._overview = payload
        capabilities = payload.get("capabilities") or {}
        writeback = capabilities.get("declarative_writeback") or []
        lines = [
            f"engine：{payload.get('engine') or ''} / adapter {payload.get('adapter_version') or ''}",
            "protocol："
            f"{payload.get('protocol_version') or ''}；locator schema："
            f"{payload.get('locator_schema_version') or ''}",
            "mode："
            f"{capabilities.get('selected_localization_mode') or ''}；"
            f"native catalog：{capabilities.get('native_catalog')}；"
            f"relocation：{capabilities.get('relocation')}",
            "declarative writeback：" + ("、".join(str(item) for item in writeback) or "(none)"),
            f"{ENGINE_SNAPSHOT_COPY['behavior_digest']}：{payload.get('behavior_digest') or ''}",
        ]
        self.engine_summary_label.setText("\n".join(lines))
        self.snapshot_root_label.setText(
            f"{ENGINE_SNAPSHOT_COPY['snapshot_root']}：{payload.get('snapshot_root') or ''}；"
            f"{ENGINE_SNAPSHOT_COPY['snapshot_count']}：{payload.get('snapshot_count') or 0}；"
            f"{ENGINE_SNAPSHOT_COPY['coverage_review_path']}："
            f"{payload.get('coverage_review_path') or '(unset)'}"
        )
        snapshots = list(payload.get("snapshots") or [])
        rows = []
        for item in snapshots:
            rows.append(
                (
                    str(item.get("version_id") or item.get("name") or ""),
                    str(item.get("generated_at") or ""),
                    f"{item.get('engine') or ''} / {item.get('adapter_version') or ''}",
                    str(item.get("coverage_status") or ""),
                    str(item.get("review_status") or ""),
                    str(item.get("snapshot_digest") or "")[:16],
                    str(item.get("manifest_path") or item.get("path") or ""),
                )
            )
        self._fill_table(self.snapshot_table, rows)
        self.overview_status_label.setText("")
        self._populate_diff_combos(snapshots)
        if not snapshots:
            self.overview_status_label.setText(ENGINE_SNAPSHOT_COPY["no_snapshots"])

    def _populate_diff_combos(self, snapshots: list[dict[str, Any]]) -> None:
        for combo in (self.diff_base_combo, self.diff_target_combo):
            combo.clear()
        for index, item in enumerate(snapshots):
            label = str(item.get("version_id") or item.get("name") or "")
            generated = str(item.get("generated_at") or "")
            if generated:
                label += f"（{generated}）"
            combo = self.diff_target_combo if index == 0 else self.diff_base_combo
            combo.addItem(label, str(item.get("manifest_path") or item.get("path") or ""))
            combo.setItemData(combo.count() - 1, str(item.get("manifest_path") or ""), Qt.ItemDataRole.ToolTipRole)

    # -- version diff ----------------------------------------------------

    def _choose_snapshot_dir(self, combo: QComboBox) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            ENGINE_SNAPSHOT_COPY["diff_choose"],
            str(self._game_root or ""),
        )
        selected = str(selected or "").strip()
        if not selected:
            return
        label = selected
        combo.addItem(label, selected)
        combo.setCurrentIndex(combo.count() - 1)

    def run_diff(self) -> None:
        if self._worker_running():
            return
        base = str(self.diff_base_combo.currentData() or "")
        target = str(self.diff_target_combo.currentData() or "")
        if not base or not target:
            self.diff_status_label.setText(ENGINE_SNAPSHOT_COPY["diff_empty"])
            return
        self.diff_status_label.setText(ENGINE_SNAPSHOT_COPY["diff_running"])
        self._set_controls_enabled(False)
        self._start_task(
            lambda: reconcile_snapshots(base, target),
            self._on_diff_loaded,
        )

    def _on_diff_loaded(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            return
        payload = dict(result.payload or {})
        self._diff = payload
        summary = payload.get("summary") or {}
        lines = [
            f"{payload.get('base_version_id') or ''} → {payload.get('target_version_id') or ''}",
            f"status：{payload.get('status') or ''}",
        ]
        for key in sorted(summary):
            lines.append(f"{key}：{summary[key]}")
        coverage_changes = payload.get("coverage_changes") or {}
        if coverage_changes:
            lines.append("coverage changes：" + str(coverage_changes))
        lines.append(
            f"items：{payload.get('item_count') or 0}"
            + (
                f"（显示前 {payload.get('item_limit')} 条）"
                if int(payload.get("item_count") or 0) > int(payload.get("item_limit") or 0)
                else ""
            )
        )
        self.diff_summary.setPlainText("\n".join(lines))
        rows = []
        for item in payload.get("items") or []:
            rows.append(
                (
                    engine_snapshot_label("disposition_labels", str(item.get("disposition") or "")),
                    engine_snapshot_label("match_kind_labels", str(item.get("match_kind") or "")),
                    f"{float(item.get('confidence') or 0.0):.3f}",
                    str(item.get("base_locator") or ""),
                    str(item.get("target_locator") or ""),
                    self._diff_evidence_text(item),
                )
            )
        self._fill_table(self.diff_table, rows)
        self.diff_status_label.setText("")

    @staticmethod
    def _diff_evidence_text(item: Mapping[str, Any]) -> str:
        evidence = dict(item.get("evidence") or {})
        labels = ENGINE_SNAPSHOT_COPY["evidence_labels"]
        parts = [
            f"{labels.get(key, key)}={str(evidence[key]).lower()}"
            for key in _EVIDENCE_FLAGS
            if key in evidence
        ]
        if "source_similarity" in evidence:
            parts.append(f"{labels['source_similarity']}={evidence['source_similarity']}")
        candidates = list(item.get("candidate_locators") or [])
        if candidates:
            shown = "、".join(str(value) for value in candidates[:3])
            suffix = "…" if len(candidates) > 3 else ""
            parts.append(f"候选：{shown}{suffix}")
        return "；".join(parts)

    # -- reuse -----------------------------------------------------------

    def _choose_reuse_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            ENGINE_SNAPSHOT_COPY["reuse_choose"],
            str(self._game_root or ""),
        )
        selected = str(selected or "").strip()
        if selected:
            self.reuse_path_edit.setText(selected)

    def load_reuse(self) -> None:
        if self._worker_running():
            return
        path = str(self.reuse_path_edit.text() or "").strip()
        if not path:
            return
        self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_running"])
        self._set_controls_enabled(False)
        self._start_task(
            lambda: load_reuse_candidates_overview(path),
            self._on_reuse_loaded,
        )

    def _on_reuse_loaded(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            return
        payload = dict(result.payload or {})
        self._reuse = payload
        summary = payload.get("summary") or {}
        lines = [
            f"status：{payload.get('status') or ''}",
            f"candidates：{payload.get('candidate_count') or 0}",
            f"{payload.get('base_version_id') or ''} → {payload.get('target_version_id') or ''}",
        ]
        if summary:
            lines.append("summary：" + "、".join(f"{key}={summary[key]}" for key in sorted(summary)))
        stale_reasons = list(payload.get("stale_reasons") or [])
        if stale_reasons:
            lines.append("stale：" + "、".join(stale_reasons))
        self.reuse_summary_label.setText("\n".join(lines))
        rows = []
        for candidate in payload.get("candidates") or []:
            rows.append(
                (
                    str(candidate.get("candidate_id") or "")[:20],
                    engine_snapshot_label("reuse_class_labels", str(candidate.get("reuse_class") or "")),
                    engine_snapshot_label("reuse_status_labels", str(candidate.get("status") or "")),
                    f"{float(candidate.get('confidence') or 0.0):.3f}",
                    self._reuse_origin_text(candidate),
                    self._reuse_target_text(candidate),
                    self._reuse_evidence_text(candidate),
                )
            )
        self._fill_table(self.reuse_table, rows)
        self.reuse_status_label.setText("")
        review_exists = bool(payload.get("review_exists"))
        self.reuse_open_review_btn.setEnabled(review_exists)
        self.reuse_open_review_btn.setToolTip(
            str(payload.get("review_path") or "")
            if review_exists
            else ENGINE_SNAPSHOT_COPY["reuse_review_missing"]
        )

    @staticmethod
    def _reuse_origin_text(candidate: Mapping[str, Any]) -> str:
        origin = str(candidate.get("reference_origin") or "")
        if candidate.get("reference_only"):
            origin = (origin + "；仅参考").strip("；")
        translation = str(candidate.get("effective_translation") or candidate.get("reference_translation") or "")
        return f"{origin}：{translation}" if origin else translation

    @staticmethod
    def _reuse_target_text(candidate: Mapping[str, Any]) -> str:
        targets = list(candidate.get("candidate_target_occurrence_ids") or [])
        if targets:
            shown = "、".join(str(value) for value in targets[:2])
            suffix = f" 等 {len(targets)} 个" if len(targets) > 2 else ""
            return shown + suffix
        return str(candidate.get("target_occurrence_id") or "")

    @staticmethod
    def _reuse_evidence_text(candidate: Mapping[str, Any]) -> str:
        evidence = dict(candidate.get("evidence") or {})
        labels = ENGINE_SNAPSHOT_COPY["evidence_labels"]
        parts = [
            f"{labels.get(key, key)}={evidence[key]}" for key in sorted(evidence)[:6]
        ]
        if len(evidence) > 6:
            parts.append("…")
        return "；".join(parts)

    def _open_reuse_review(self) -> None:
        path = str((self._reuse or {}).get("review_path") or "")
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # -- task plumbing ---------------------------------------------------

    def _worker_running(self) -> bool:
        worker = self._worker
        return worker is not None and worker.isRunning()

    def _start_task(
        self,
        task: Callable[[], Any],
        handler: Callable[[EngineSnapshotTaskResult], None],
    ) -> None:
        worker = EngineSnapshotTaskWorker(task, parent=self)
        worker.completed.connect(
            lambda result, current_handler=handler: self._on_task_completed(
                result,
                current_handler,
            )
        )
        self._worker = worker
        worker.start()

    def _on_task_completed(
        self,
        result: EngineSnapshotTaskResult,
        handler: Callable[[EngineSnapshotTaskResult], None],
    ) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_controls_enabled(True)
        handler(result)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in (
            *getattr(self, "_overview_buttons", ()),
            *getattr(self, "_diff_buttons", ()),
            *getattr(self, "_reuse_buttons", ()),
        ):
            button.setEnabled(bool(enabled))
        if enabled and hasattr(self, "reuse_open_review_btn"):
            self.reuse_open_review_btn.setEnabled(
                bool(getattr(self, "_reuse", {}).get("review_exists"))
            )

    def _show_error(self, result: EngineSnapshotTaskResult) -> None:
        detail = result.error or result.error_code or "unknown error"
        QMessageBox.warning(self, ENGINE_SNAPSHOT_COPY["error_title"], detail)

    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
        table.setRowCount(0)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(2000)
        super().closeEvent(event)
