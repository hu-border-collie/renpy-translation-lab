"""Read-only GUI dialog for engine capabilities, snapshots, diffs and reuse."""

from __future__ import annotations

import json
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
    MAX_REUSE_CANDIDATES,
    collect_engine_snapshot_overview,
    load_reuse_candidates_overview,
    reconcile_snapshots,
    export_reuse_results_to_manifest,
    submit_reuse_candidate_decision,
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
    """Browse P3/P4 artifacts and submit reuse actions through shared services."""

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
        self._reuse_context_revision = 0
        self._editor_revision = 0
        self._reuse_page_offset = 0
        self._awaiting_task_result = False

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
        self.reuse_path_edit.textChanged.connect(self._on_reuse_path_changed)
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

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["reuse_manifest"]))
        self.reuse_manifest_edit = QLineEdit("")
        self.reuse_manifest_edit.setObjectName("engine_snapshot_reuse_manifest_edit")
        self.reuse_manifest_edit.setPlaceholderText(
            ENGINE_SNAPSHOT_COPY["reuse_manifest_placeholder"]
        )
        self.reuse_manifest_edit.textChanged.connect(self._on_reuse_manifest_changed)
        target_row.addWidget(self.reuse_manifest_edit, 1)
        self.reuse_manifest_choose_btn = QPushButton(
            ENGINE_SNAPSHOT_COPY["reuse_manifest_choose"]
        )
        self.reuse_manifest_choose_btn.setObjectName(
            "engine_snapshot_reuse_manifest_choose_btn"
        )
        self.reuse_manifest_choose_btn.clicked.connect(self._choose_reuse_manifest)
        target_row.addWidget(self.reuse_manifest_choose_btn)
        layout.addLayout(target_row)

        self.reuse_status_label = QLabel("")
        self.reuse_status_label.setWordWrap(True)
        layout.addWidget(self.reuse_status_label)
        self.reuse_operation_label = QLabel("")
        self.reuse_operation_label.setWordWrap(True)
        self.reuse_operation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.reuse_operation_label)
        layout.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["reuse_summary"]))
        self.reuse_summary_label = QLabel("")
        self.reuse_summary_label.setWordWrap(True)
        self.reuse_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.reuse_summary_label)
        self.reuse_scope_note = QLabel(ENGINE_SNAPSHOT_COPY["reuse_scope_note"])
        self.reuse_scope_note.setWordWrap(True)
        layout.addWidget(self.reuse_scope_note)
        self.reuse_table = self._make_table(ENGINE_SNAPSHOT_COPY["reuse_columns"])
        self.reuse_table.setObjectName("engine_snapshot_reuse_table")
        self.reuse_table.itemSelectionChanged.connect(self._on_reuse_selection_changed)
        layout.addWidget(self.reuse_table, 1)
        page_row = QHBoxLayout()
        self.reuse_previous_page_btn = QPushButton(
            ENGINE_SNAPSHOT_COPY["reuse_previous_page"]
        )
        self.reuse_previous_page_btn.setObjectName(
            "engine_snapshot_reuse_previous_page_btn"
        )
        self.reuse_previous_page_btn.clicked.connect(
            lambda: self._change_reuse_page(-MAX_REUSE_CANDIDATES)
        )
        page_row.addWidget(self.reuse_previous_page_btn)
        self.reuse_page_label = QLabel("")
        self.reuse_page_label.setObjectName("engine_snapshot_reuse_page_label")
        page_row.addWidget(self.reuse_page_label, 1)
        self.reuse_next_page_btn = QPushButton(
            ENGINE_SNAPSHOT_COPY["reuse_next_page"]
        )
        self.reuse_next_page_btn.setObjectName("engine_snapshot_reuse_next_page_btn")
        self.reuse_next_page_btn.clicked.connect(
            lambda: self._change_reuse_page(MAX_REUSE_CANDIDATES)
        )
        page_row.addWidget(self.reuse_next_page_btn)
        layout.addLayout(page_row)

        layout.addWidget(QLabel(ENGINE_SNAPSHOT_COPY["reuse_detail"]))
        self.reuse_detail = QPlainTextEdit()
        self.reuse_detail.setReadOnly(True)
        self.reuse_detail.setMaximumHeight(150)
        self.reuse_detail.setObjectName("engine_snapshot_reuse_detail")
        layout.addWidget(self.reuse_detail)

        review_form = QFormLayout()
        self.reuse_reviewer_edit = QLineEdit("")
        self.reuse_reviewer_edit.setObjectName("engine_snapshot_reuse_reviewer_edit")
        self.reuse_reviewer_edit.setPlaceholderText(
            ENGINE_SNAPSHOT_COPY["reuse_reviewer_placeholder"]
        )
        self.reuse_reviewer_edit.textEdited.connect(self._on_reuse_editor_edited)
        review_form.addRow(ENGINE_SNAPSHOT_COPY["reuse_reviewer"], self.reuse_reviewer_edit)
        self.reuse_note_edit = QLineEdit("")
        self.reuse_note_edit.setObjectName("engine_snapshot_reuse_note_edit")
        self.reuse_note_edit.setPlaceholderText(
            ENGINE_SNAPSHOT_COPY["reuse_note_placeholder"]
        )
        self.reuse_note_edit.textEdited.connect(self._on_reuse_editor_edited)
        review_form.addRow(ENGINE_SNAPSHOT_COPY["reuse_note"], self.reuse_note_edit)
        self.reuse_target_combo = QComboBox()
        self.reuse_target_combo.setObjectName("engine_snapshot_reuse_target_combo")
        self.reuse_target_combo.currentIndexChanged.connect(self._on_reuse_target_changed)
        review_form.addRow(ENGINE_SNAPSHOT_COPY["reuse_ambiguous_target"], self.reuse_target_combo)
        layout.addLayout(review_form)

        action_row = QHBoxLayout()
        self.reuse_accept_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_accept"])
        self.reuse_accept_btn.setObjectName("engine_snapshot_reuse_accept_btn")
        self.reuse_accept_btn.clicked.connect(lambda: self.submit_reuse_decision("accept"))
        action_row.addWidget(self.reuse_accept_btn)
        self.reuse_reject_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_reject"])
        self.reuse_reject_btn.setObjectName("engine_snapshot_reuse_reject_btn")
        self.reuse_reject_btn.clicked.connect(lambda: self.submit_reuse_decision("reject"))
        action_row.addWidget(self.reuse_reject_btn)
        self.reuse_cancel_decision_btn = QPushButton(
            ENGINE_SNAPSHOT_COPY["reuse_cancel_decision"]
        )
        self.reuse_cancel_decision_btn.setObjectName(
            "engine_snapshot_reuse_cancel_decision_btn"
        )
        self.reuse_cancel_decision_btn.clicked.connect(self.cancel_reuse_decision)
        action_row.addWidget(self.reuse_cancel_decision_btn)
        action_row.addStretch(1)
        self.reuse_export_btn = QPushButton(ENGINE_SNAPSHOT_COPY["reuse_export"])
        self.reuse_export_btn.setObjectName("engine_snapshot_reuse_export_btn")
        self.reuse_export_btn.clicked.connect(self.export_reuse_results)
        action_row.addWidget(self.reuse_export_btn)
        layout.addLayout(action_row)

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
            self.reuse_manifest_choose_btn,
            self.reuse_previous_page_btn,
            self.reuse_next_page_btn,
            self.reuse_accept_btn,
            self.reuse_reject_btn,
            self.reuse_cancel_decision_btn,
            self.reuse_export_btn,
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
        offset = self._reuse_page_offset
        self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_running"])
        self._set_controls_enabled(False)
        self._start_task(
            lambda: load_reuse_candidates_overview(
                path,
                candidate_offset=offset,
            ),
            self._on_reuse_loaded,
        )

    def _on_reuse_loaded(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            return
        payload = dict(result.payload or {})
        self._reuse = payload
        self._clear_reuse_editor(clear_selection=True)
        summary = payload.get("summary") or {}
        lines = [
            f"status：{payload.get('status') or ''}",
            f"candidates：{payload.get('candidate_count') or 0}",
            f"{payload.get('base_version_id') or ''} → {payload.get('target_version_id') or ''}",
        ]
        candidate_count = int(payload.get("candidate_count") or 0)
        candidate_limit = int(payload.get("candidate_limit") or 0)
        if candidate_count > candidate_limit:
            lines.append(
                ENGINE_SNAPSHOT_COPY["reuse_display_limited"].format(
                    count=candidate_limit,
                    total=candidate_count,
                )
            )
        if candidate_count:
            lines.append(
                ENGINE_SNAPSHOT_COPY["reuse_page_summary"].format(
                    start=payload.get("candidate_start") or 0,
                    end=payload.get("candidate_end") or 0,
                    total=candidate_count,
                )
            )
        if summary:
            lines.append("summary：" + "、".join(f"{key}={summary[key]}" for key in sorted(summary)))
        stale_reasons = list(payload.get("stale_reasons") or [])
        if stale_reasons:
            lines.append("stale：" + "、".join(stale_reasons))
        self.reuse_summary_label.setText("\n".join(lines))
        rows = []
        self._reuse_candidates_by_id = {}
        for candidate in payload.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            self._reuse_candidates_by_id[candidate_id] = candidate
            rows.append(
                (
                    candidate_id[:20],
                    engine_snapshot_label("reuse_class_labels", str(candidate.get("reuse_class") or "")),
                    engine_snapshot_label("reuse_status_labels", str(candidate.get("status") or "")),
                    f"{float(candidate.get('confidence') or 0.0):.3f}",
                    self._reuse_origin_text(candidate),
                    self._reuse_target_text(candidate),
                    self._reuse_evidence_text(candidate),
                )
            )
        self._fill_table(self.reuse_table, rows)
        for row_index, candidate_id in enumerate(self._reuse_candidates_by_id):
            item = self.reuse_table.item(row_index, 0)
            if item is not None:
                item.setData(Qt.ItemDataRole.UserRole, candidate_id)
        self.reuse_status_label.setText("")
        self.reuse_page_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_page_label"].format(
                start=payload.get("candidate_start") or 0,
                end=payload.get("candidate_end") or 0,
                total=candidate_count,
            )
        )
        review_exists = bool(payload.get("review_exists"))
        self.reuse_open_review_btn.setEnabled(review_exists)
        self.reuse_open_review_btn.setToolTip(
            str(payload.get("review_path") or "")
            if review_exists
            else ENGINE_SNAPSHOT_COPY["reuse_review_missing"]
        )
        self._refresh_reuse_action_states()

    def _choose_reuse_manifest(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            ENGINE_SNAPSHOT_COPY["reuse_manifest_choose"],
            str(self._game_root or ""),
            ENGINE_SNAPSHOT_COPY["reuse_manifest_filter"],
        )
        selected = str(selected or "").strip()
        if selected:
            self.reuse_manifest_edit.setText(selected)

    def _current_reuse_candidate(self) -> dict[str, Any] | None:
        row = self.reuse_table.currentRow()
        item = self.reuse_table.item(row, 0) if row >= 0 else None
        candidate_id = str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""
        return getattr(self, "_reuse_candidates_by_id", {}).get(candidate_id)

    def _on_reuse_selection_changed(self) -> None:
        had_unsubmitted_input = bool(
            self.reuse_reviewer_edit.text()
            or self.reuse_note_edit.text()
            or self.reuse_target_combo.currentData()
        )
        self._clear_reuse_editor(clear_selection=False)
        if had_unsubmitted_input:
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_candidate_changed"]
            )
        candidate = self._current_reuse_candidate()
        if candidate is None:
            self._refresh_reuse_action_states()
            return
        target_ids = list(candidate.get("candidate_target_occurrence_ids") or [])
        is_ambiguous = str(candidate.get("reuse_class") or "") == "ambiguous"
        self.reuse_target_combo.blockSignals(True)
        self.reuse_target_combo.clear()
        self.reuse_target_combo.addItem(
            ENGINE_SNAPSHOT_COPY["reuse_target_placeholder"], ""
        )
        if is_ambiguous:
            for target_id in target_ids:
                self.reuse_target_combo.addItem(str(target_id), str(target_id))
        elif candidate.get("target_occurrence_id"):
            self.reuse_target_combo.addItem(
                str(candidate["target_occurrence_id"]),
                str(candidate["target_occurrence_id"]),
            )
        self.reuse_target_combo.setCurrentIndex(0)
        self.reuse_target_combo.setEnabled(is_ambiguous)
        self.reuse_target_combo.blockSignals(False)
        self._editor_revision += 1
        self.reuse_detail.setPlainText(self._reuse_candidate_detail(candidate))
        self._refresh_reuse_action_states()

    @staticmethod
    def _reuse_candidate_detail(candidate: Mapping[str, Any]) -> str:
        targets = list(candidate.get("candidate_target_occurrence_ids") or [])
        if not targets and candidate.get("target_occurrence_id"):
            targets = [str(candidate.get("target_occurrence_id"))]
        lines = [
            f"candidate_id：{candidate.get('candidate_id') or ''}",
            f"类别：{candidate.get('reuse_class') or ''}",
            f"状态：{candidate.get('status') or ''}",
            "旧译文：\n" + str(candidate.get("reference_translation_full") or ""),
            "当前有效译文：\n"
            + str(
                candidate.get("effective_translation_full")
                or candidate.get("reference_translation_full")
                or ""
            ),
            "目标 occurrence：\n" + ("\n".join(targets) if targets else "(none)"),
            "证据：\n"
            + json.dumps(candidate.get("evidence") or {}, ensure_ascii=False, indent=2),
            "当前决定：\n"
            + json.dumps(candidate.get("decision") or {}, ensure_ascii=False, indent=2),
            f"审计记录（{len(candidate.get('audit') or [])}）：\n"
            + json.dumps(candidate.get("audit") or [], ensure_ascii=False, indent=2),
        ]
        return "\n\n".join(lines)

    def _clear_reuse_editor(self, *, clear_selection: bool) -> None:
        if clear_selection:
            self.reuse_table.blockSignals(True)
            self.reuse_table.clearSelection()
            self.reuse_table.setCurrentCell(-1, -1)
            self.reuse_table.blockSignals(False)
        self.reuse_detail.clear()
        self.reuse_reviewer_edit.clear()
        self.reuse_note_edit.clear()
        self.reuse_target_combo.blockSignals(True)
        self.reuse_target_combo.clear()
        self.reuse_target_combo.addItem(
            ENGINE_SNAPSHOT_COPY["reuse_target_placeholder"], ""
        )
        self.reuse_target_combo.setCurrentIndex(0)
        self.reuse_target_combo.setEnabled(False)
        self.reuse_target_combo.blockSignals(False)
        self._editor_revision += 1

    def _on_reuse_path_changed(self, _value: str) -> None:
        self._reuse_context_revision += 1
        self._reuse_page_offset = 0
        self._reuse = {}
        self._reuse_candidates_by_id = {}
        self._fill_table(self.reuse_table, [])
        self._clear_reuse_editor(clear_selection=True)
        self.reuse_summary_label.clear()
        self.reuse_page_label.clear()
        self.reuse_operation_label.clear()
        self.reuse_open_review_btn.setEnabled(False)
        self.reuse_status_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_package_changed"]
        )
        self._refresh_reuse_action_states()

    def _on_reuse_manifest_changed(self, _value: str) -> None:
        self._reuse_context_revision += 1
        self._clear_reuse_editor(clear_selection=True)
        self.reuse_operation_label.clear()
        self.reuse_status_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_manifest_changed"]
        )
        self._refresh_reuse_action_states()

    def _on_reuse_editor_edited(self, _value: str) -> None:
        self._editor_revision += 1

    def _on_reuse_target_changed(self, _index: int) -> None:
        self._editor_revision += 1
        self._refresh_reuse_action_states()

    def _refresh_reuse_action_states(self) -> None:
        if not hasattr(self, "reuse_accept_btn"):
            return
        candidate = self._current_reuse_candidate()
        fresh = bool(self._reuse) and not self._reuse.get("stale_reasons") and (
            str(self._reuse.get("status") or "") != "stale"
        )
        pending = bool(candidate) and str(candidate.get("status") or "") == "pending"
        idle = not self._worker_running()
        self.reuse_accept_btn.setEnabled(idle and fresh and pending)
        self.reuse_reject_btn.setEnabled(idle and fresh and pending)
        self.reuse_cancel_decision_btn.setEnabled(
            idle
            and bool(candidate or self.reuse_reviewer_edit.text() or self.reuse_note_edit.text())
        )
        summary = self._reuse.get("summary") or {}
        has_direct_accept = int(summary.get("accepted_direct_reuse") or 0) > 0
        self.reuse_export_btn.setEnabled(
            idle
            and fresh
            and has_direct_accept
            and bool(str(self.reuse_manifest_edit.text() or "").strip())
        )
        self.reuse_open_review_btn.setEnabled(
            idle and bool(self._reuse.get("review_exists"))
        )
        candidate_count = int(self._reuse.get("candidate_count") or 0)
        candidate_limit = int(self._reuse.get("candidate_limit") or MAX_REUSE_CANDIDATES)
        self.reuse_previous_page_btn.setEnabled(
            idle and bool(self._reuse) and self._reuse_page_offset > 0
        )
        self.reuse_next_page_btn.setEnabled(
            idle
            and bool(self._reuse)
            and self._reuse_page_offset + candidate_limit < candidate_count
        )

    def _change_reuse_page(self, delta: int) -> None:
        if self._worker_running() or not self._reuse:
            return
        total = int(self._reuse.get("candidate_count") or 0)
        limit = max(1, int(self._reuse.get("candidate_limit") or MAX_REUSE_CANDIDATES))
        offset = max(0, min(self._reuse_page_offset + int(delta), max(0, total - 1)))
        offset = (offset // limit) * limit
        if offset == self._reuse_page_offset:
            return
        self._reuse_page_offset = offset
        self._reuse_context_revision += 1
        self._clear_reuse_editor(clear_selection=True)
        self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_running"])
        self._set_controls_enabled(False)
        path = str(self.reuse_path_edit.text() or "").strip()
        self._start_task(
            lambda: load_reuse_candidates_overview(
                path,
                candidate_offset=offset,
            ),
            self._on_reuse_loaded,
        )

    def submit_reuse_decision(self, action: str) -> None:
        if self._worker_running():
            return
        candidate = self._current_reuse_candidate()
        if candidate is None:
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_select_candidate"]
            )
            return
        reviewer = str(self.reuse_reviewer_edit.text() or "").strip()
        if not reviewer:
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_reviewer_required"]
            )
            return
        target_id = ""
        if action == "accept" and str(candidate.get("reuse_class") or "") == "ambiguous":
            target_id = str(self.reuse_target_combo.currentData() or "").strip()
            if not target_id:
                self.reuse_status_label.setText(
                    ENGINE_SNAPSHOT_COPY["reuse_ambiguous_target_required"]
                )
                return
        source = str(self.reuse_path_edit.text() or "").strip()
        candidate_id = str(candidate.get("candidate_id") or "")
        note = str(self.reuse_note_edit.text() or "").strip()
        self.reuse_status_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_decision_running"]
        )
        self._set_controls_enabled(False)
        self._start_task(
            lambda: submit_reuse_candidate_decision(
                source,
                candidate_id,
                action,
                reviewer,
                note=note,
                target_occurrence_id=target_id,
            ),
            self._on_reuse_decision_submitted,
        )

    def _on_reuse_decision_submitted(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_failed"])
            return
        payload = dict(result.payload or {})
        paths = dict(payload.get("paths") or {})
        output_dir = str(paths.get("output_dir") or "")
        self.reuse_status_label.setText("")
        if output_dir:
            self.reuse_path_edit.setText(output_dir)
        self.reuse_operation_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_decision_saved"].format(
                status=payload.get("status") or "",
                output=output_dir or paths.get("report") or "",
            )
        )
        if output_dir:
            self.load_reuse()

    def cancel_reuse_decision(self) -> None:
        self._clear_reuse_editor(clear_selection=True)
        self.reuse_status_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_decision_cancelled"]
        )
        self._refresh_reuse_action_states()

    def export_reuse_results(self) -> None:
        if self._worker_running():
            return
        source = str(self.reuse_path_edit.text() or "").strip()
        manifest = str(self.reuse_manifest_edit.text() or "").strip()
        if not manifest:
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_manifest_required"]
            )
            return
        self.reuse_status_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_export_running"]
        )
        self._set_controls_enabled(False)
        self._start_task(
            lambda: export_reuse_results_to_manifest(source, manifest),
            self._on_reuse_results_exported,
        )

    def _on_reuse_results_exported(self, result: EngineSnapshotTaskResult) -> None:
        if not result.ok:
            self._show_error(result)
            self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_failed"])
            return
        payload = dict(result.payload or {})
        self.reuse_operation_label.setText(
            ENGINE_SNAPSHOT_COPY["reuse_export_saved"].format(
                reused=payload.get("reused_items") or 0,
                result=payload.get("result_jsonl_path") or "",
                manifest=payload.get("manifest_path") or "",
            )
        )
        self.reuse_status_label.setText(ENGINE_SNAPSHOT_COPY["reuse_check_required"])
        self._refresh_reuse_action_states()

    def _async_context_key(self) -> tuple[str, str, str, int, int, int]:
        return (
            str(self._game_root or ""),
            str(self.reuse_path_edit.text() or "").strip()
            if hasattr(self, "reuse_path_edit")
            else "",
            str(self.reuse_manifest_edit.text() or "").strip()
            if hasattr(self, "reuse_manifest_edit")
            else "",
            self._reuse_context_revision,
            self._editor_revision,
            self._reuse_page_offset,
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
        expected_context = self._async_context_key()
        worker = EngineSnapshotTaskWorker(task, parent=self)
        worker.completed.connect(
            lambda result, current_handler=handler: self._on_task_completed(
                result,
                current_handler,
                expected_context,
            )
        )
        self._worker = worker
        self._awaiting_task_result = True
        worker.start()

    def _on_task_completed(
        self,
        result: EngineSnapshotTaskResult,
        handler: Callable[[EngineSnapshotTaskResult], None],
        expected_context: tuple[str, str, str, int, int, int],
    ) -> None:
        worker = self._worker
        self._worker = None
        self._awaiting_task_result = False
        if worker is not None:
            worker.deleteLater()
        self._set_controls_enabled(True)
        if expected_context != self._async_context_key():
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_stale_result_ignored"]
            )
            return
        handler(result)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in (
            *getattr(self, "_overview_buttons", ()),
            *getattr(self, "_diff_buttons", ()),
            *getattr(self, "_reuse_buttons", ()),
        ):
            button.setEnabled(bool(enabled))
        if hasattr(self, "reuse_table"):
            self.reuse_path_edit.setEnabled(bool(enabled))
            self.reuse_manifest_edit.setEnabled(bool(enabled))
            self.reuse_table.setEnabled(bool(enabled))
            self.reuse_reviewer_edit.setEnabled(bool(enabled))
            self.reuse_note_edit.setEnabled(bool(enabled))
            candidate = self._current_reuse_candidate()
            self.reuse_target_combo.setEnabled(
                bool(enabled)
                and bool(candidate)
                and str(candidate.get("reuse_class") or "") == "ambiguous"
            )
            self._refresh_reuse_action_states()

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
        running = worker is not None and worker.isRunning()
        finished_result_pending = bool(
            worker is not None
            and self._awaiting_task_result
            and callable(getattr(worker, "isFinished", None))
            and worker.isFinished()
        )
        if running or finished_result_pending:
            self.reuse_status_label.setText(
                ENGINE_SNAPSHOT_COPY["reuse_wait_to_close"]
            )
            event.ignore()
            return
        super().closeEvent(event)
