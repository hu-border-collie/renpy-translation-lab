"""Bounded ordinary-translation review controls for the revision page."""
from __future__ import annotations

import json
from itertools import chain
from typing import Any, Callable

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

import review_index
import review_workspace as service

from ..user_copy import REVIEW_WORKSPACE_COPY as COPY

_ACTIVE_WORKERS: set[QThread] = set()


class ReviewLoadWorker(QThread):
    loaded = Signal(object, object, object)
    failed = Signal(object, str)

    def __init__(self, corpus_path: str, game_root: str, tl_dir: str, token: object) -> None:
        super().__init__()
        self.corpus_path = corpus_path
        self.game_root = game_root
        self.tl_dir = tl_dir
        self.token = token

    def run(self) -> None:
        try:
            manifest, entries = service.open_workspace(
                self.corpus_path, expected_game_root=self.game_root,
                expected_tl_dir=self.tl_dir,
            )
            self.loaded.emit(self.token, manifest, entries)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.failed.emit(self.token, str(exc))


class ReviewWorkspaceWidget(QWidget):
    """Review one exported corpus; all displayed rows are from a bounded page."""

    proposal_ready = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("review_workspace")
        self._generation = 0
        self._current_context: Callable[[], str] = lambda: ""
        self.manifest: dict[str, Any] | None = None
        self.entries: list[dict[str, Any]] = []
        self.index_path = ""
        self.drafts: dict[str, dict[str, Any]] = {}
        self.page_number = 0
        self.page_rows: list[dict[str, Any]] = []
        self.current_id = ""
        self._editing = False
        self._dirty = False
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(450)
        self._autosave.timeout.connect(self.save_current_draft)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(250)
        self._filter_timer.timeout.connect(self._filter_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(QLabel(COPY["title"]))
        self.message = QLabel(COPY["empty"])
        self.message.setObjectName("review_workspace_message")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        filters = QHBoxLayout()
        self.file_filter = QLineEdit()
        self.file_filter.setObjectName("review_file_filter")
        self.file_filter.setPlaceholderText(COPY["file_filter"])
        self.lifecycle_filter = QComboBox()
        self.lifecycle_filter.setObjectName("review_lifecycle_filter")
        for label, value in COPY["lifecycles"]:
            self.lifecycle_filter.addItem(label, value)
        self.findings_filter = QComboBox()
        for label, value in COPY["finding_options"]:
            self.findings_filter.addItem(label, value)
        self.severity_filter = QComboBox()
        for label, value in COPY["severities"]:
            self.severity_filter.addItem(label, value)
        self.speaker_filter = QLineEdit()
        self.speaker_filter.setPlaceholderText(COPY["speaker_filter"])
        self.query_filter = QLineEdit()
        self.query_filter.setPlaceholderText(COPY["query_filter"])
        for widget in (self.file_filter, self.lifecycle_filter, self.findings_filter,
                       self.severity_filter, self.speaker_filter, self.query_filter):
            filters.addWidget(widget)
        layout.addLayout(filters)
        for combo in (self.lifecycle_filter, self.findings_filter, self.severity_filter):
            combo.currentIndexChanged.connect(self._filter_changed)
        for field in (self.file_filter, self.speaker_filter, self.query_filter):
            field.textChanged.connect(lambda _text: self._filter_timer.start())

        self.table = QTableWidget(0, 5)
        self.table.setObjectName("review_entries_table")
        self.table.setHorizontalHeaderLabels(COPY["columns"])
        self.table.setMinimumHeight(170)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table)
        nav = QHBoxLayout()
        self.previous_btn = QPushButton(COPY["previous"])
        self.next_btn = QPushButton(COPY["next"])
        self.next_open_btn = QPushButton(COPY["next_open"])
        self.page_label = QLabel("")
        for widget in (self.previous_btn, self.page_label, self.next_btn, self.next_open_btn):
            nav.addWidget(widget)
        self.previous_btn.clicked.connect(lambda: self._navigate(-1))
        self.next_btn.clicked.connect(lambda: self._navigate(1))
        self.next_open_btn.clicked.connect(self._next_open)
        layout.addLayout(nav)

        self.details = QTextEdit()
        self.details.setObjectName("review_entry_details")
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(100)
        layout.addWidget(self.details)
        form = QFormLayout()
        self.proposed = QTextEdit()
        self.proposed.setObjectName("review_proposed_translation")
        self.proposed.setMinimumHeight(70)
        self.reason = QLineEdit()
        self.reason.setObjectName("review_reason")
        self.reviewer = QLineEdit()
        self.reviewer.setObjectName("review_reviewer")
        self.note = QLineEdit()
        self.note.setObjectName("review_note")
        form.addRow(COPY["proposed"], self.proposed)
        form.addRow(COPY["reason"], self.reason)
        form.addRow(COPY["reviewer"], self.reviewer)
        form.addRow(COPY["note"], self.note)
        layout.addLayout(form)
        self.proposed.textChanged.connect(self._draft_changed)
        self.reason.textChanged.connect(self._draft_changed)
        actions = QGridLayout()
        self.save_btn = QPushButton(COPY["save_draft"])
        self.ignored_btn = QPushButton(COPY["ignored"])
        self.resolved_btn = QPushButton(COPY["resolved"])
        self.reopen_btn = QPushButton(COPY["reopen"])
        self.select_page_btn = QPushButton(COPY["select_page"])
        self.export_btn = QPushButton(COPY["export"])
        self.export_btn.setToolTip(COPY["export_tooltip"])
        for index, widget in enumerate((self.save_btn, self.ignored_btn, self.resolved_btn,
                                        self.reopen_btn, self.select_page_btn, self.export_btn)):
            actions.addWidget(widget, index // 3, index % 3)
        self.save_btn.clicked.connect(self.save_current_draft)
        self.ignored_btn.clicked.connect(lambda: self._save_decision("ignored"))
        self.resolved_btn.clicked.connect(lambda: self._save_decision("resolved"))
        self.reopen_btn.clicked.connect(lambda: self._save_decision("open"))
        self.select_page_btn.clicked.connect(self.table.selectAll)
        self.export_btn.clicked.connect(self._export_selected)
        layout.addLayout(actions)
        self._set_editor_enabled(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for widget in (self.proposed, self.reason, self.reviewer, self.note,
                       self.save_btn, self.ignored_btn, self.resolved_btn, self.reopen_btn):
            widget.setEnabled(enabled)
        self.select_page_btn.setEnabled(bool(self.page_rows))
        self.export_btn.setEnabled(bool(self.page_rows))

    def reset(self) -> None:
        self._generation += 1
        if not self._save_if_dirty():
            message = QMessageBox(self)
            message.setWindowTitle(COPY["draft_error"])
            message.setText(COPY["unsaved_warning"])
            message.setDetailedText(
                f"{COPY['proposed']}:\n{self.proposed.toPlainText()}\n\n"
                f"{COPY['reason']}:\n{self.reason.text()}"
            )
            message.exec()
        self._filter_timer.stop()
        self.manifest = None
        self.entries = []
        self.index_path = ""
        self.drafts = {}
        self.page_rows = []
        self.current_id = ""
        self._dirty = False
        self.table.setRowCount(0)
        self.details.clear()
        self._set_editor_enabled(False)
        self.message.setText(COPY["empty"])

    def load(self, corpus_path: str, game_root: str, tl_dir: str, current_context: Callable[[], str]) -> None:
        self.reset()
        self._current_context = current_context
        self._generation += 1
        token = (self._generation, current_context())
        self.message.setText(COPY["loading"])
        worker = ReviewLoadWorker(corpus_path, game_root, tl_dir, token)
        _ACTIVE_WORKERS.add(worker)
        worker.loaded.connect(self._loaded)
        worker.failed.connect(self._failed)
        worker.finished.connect(lambda: _ACTIVE_WORKERS.discard(worker))
        worker.start()

    def _current_token(self, token: object) -> bool:
        return token == (self._generation, self._current_context())

    def _loaded(self, token: object, manifest: object, entries: object) -> None:
        if not self._current_token(token):
            return
        self.manifest = manifest
        self.entries = entries
        self.index_path = str(manifest.get("_manifest_path") or "")
        try:
            self.drafts = service.load_drafts(self.index_path, manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.message.setText(f"{COPY['draft_error']} {exc}")
            self.manifest = None
            self.entries = []
            return
        self.page_number = 0
        self._render_page()

    def _failed(self, token: object, message: str) -> None:
        if self._current_token(token):
            self.message.setText(f"{COPY['load_error']} {message}")

    def _save_if_dirty(self) -> bool:
        if self._dirty or self._autosave.isActive():
            self._autosave.stop()
            return self.save_current_draft()
        return True

    def _draft_changed(self) -> None:
        if not self._editing and self.current_id and self.manifest:
            self._dirty = True
            self._autosave.start()

    def save_current_draft(self) -> bool:
        if not self.current_id or not self.manifest:
            return True
        try:
            draft = service.save_draft(
                self.index_path, self.manifest, self.entries, self.current_id,
                self.proposed.toPlainText(), self.reason.text(),
            )
            self.drafts[self.current_id] = draft
            self._dirty = False
            self.message.setText(COPY["draft_saved"])
            return True
        except (OSError, ValueError) as exc:
            self.message.setText(f"{COPY['draft_error']} {exc}")
            return False

    def _filter_changed(self, _value: object = None) -> None:
        if not self._save_if_dirty():
            return
        self.page_number = 0
        self._render_page()

    def _render_page(self) -> None:
        if self.manifest is None:
            return
        total, rows = service.filter_page(
            self.entries, page=self.page_number,
            file=self.file_filter.text(),
            lifecycle=str(self.lifecycle_filter.currentData() or ""),
            findings=str(self.findings_filter.currentData() or ""),
            severity=str(self.severity_filter.currentData() or ""),
            speaker=self.speaker_filter.text(), query=self.query_filter.text(),
        )
        self.page_rows = list(rows)
        self.current_id = ""
        self._editing = True
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for number, row in enumerate(rows):
            review = row.get("review") or {}
            values = (
                f"{row.get('file_rel_path') or ''}:{row.get('display_line') or ''}",
                str(row.get("speaker_id") or COPY["missing"]),
                str(row.get("source") or "").replace("\n", " ")[:100],
                str(row.get("current_translation") or "").replace("\n", " ")[:100],
                f"{review.get('lifecycle') or 'open'} / {len(row.get('quality_findings') or [])}",
            )
            for column, value in enumerate(values):
                self.table.setItem(number, column, QTableWidgetItem(value))
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.details.clear()
        self.proposed.clear()
        self.reason.clear()
        self._editing = False
        self._set_editor_enabled(False)
        self.page_label.setText(COPY["page"].format(page=self.page_number + 1, total=total))
        self.previous_btn.setEnabled(self.page_number > 0)
        self.next_btn.setEnabled((self.page_number + 1) * service.PAGE_SIZE < total)
        self.message.setText(COPY["loaded"].format(total=total))

    def _selection_changed(self) -> None:
        if not self._save_if_dirty():
            return
        selected = sorted({item.row() for item in self.table.selectedItems()})
        if len(selected) != 1:
            self.current_id = ""
            self._set_editor_enabled(False)
            return
        row = self.page_rows[selected[0]]
        self.current_id = str(row.get("occurrence_id") or "")
        review = row.get("review") or {}
        record = row.get("translation_record")
        context = row.get("context") or {}
        findings = row.get("quality_findings") or []
        glossary_findings = [
            finding for finding in findings
            if "glossary" in str(finding.get("reason_code") or "").casefold()
        ]
        detail = [
            f"identity: {self.current_id}",
            f"{COPY['source']}: {row.get('source') or COPY['missing']}",
            f"{COPY['current']}: {row.get('current_translation') or COPY['missing']}",
            f"{COPY['speaker']}: {row.get('speaker_id') or COPY['missing']}",
            f"{COPY['context']}: {json.dumps(context, ensure_ascii=False) if context else COPY['missing']}",
            f"{COPY['findings']}: {json.dumps(findings, ensure_ascii=False) if findings else COPY['missing']}",
            f"{COPY['provenance']}: {json.dumps(record, ensure_ascii=False) if record else COPY['missing']}",
            f"{COPY['glossary']}: {json.dumps(glossary_findings, ensure_ascii=False) if glossary_findings else COPY['missing']}",
            f"{COPY['decision']}: {json.dumps(review, ensure_ascii=False)}",
        ]
        self.details.setPlainText("\n".join(detail))
        draft = self.drafts.get(self.current_id) or {}
        self._editing = True
        self.proposed.setPlainText(str(draft.get("proposed_translation") or row.get("current_translation") or ""))
        self.reason.setText(str(draft.get("reason") or ""))
        self._editing = False
        self._set_editor_enabled(True)
        if draft and not service.draft_is_current(draft, row):
            self.message.setText(COPY["stale_draft"])

    def _navigate(self, delta: int) -> None:
        if not self._save_if_dirty():
            return
        self.page_number += delta
        self._render_page()

    def _next_open(self) -> None:
        if not self._save_if_dirty():
            return
        current_position = next(
            (index for index, row in enumerate(self.entries)
             if row.get("occurrence_id") == self.current_id),
            -1,
        )
        for combo in (self.lifecycle_filter, self.findings_filter, self.severity_filter):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        for field in (self.file_filter, self.speaker_filter, self.query_filter):
            field.blockSignals(True)
            field.clear()
            field.blockSignals(False)
        for index in chain(
            range(current_position + 1, len(self.entries)),
            range(0, current_position + 1),
        ):
            row = self.entries[index]
            if (row.get("review") or {}).get("lifecycle") in ("open", "needs_recheck"):
                self.page_number = index // service.PAGE_SIZE
                self._render_page()
                self.table.selectRow(index % service.PAGE_SIZE)
                return

    def _save_decision(self, lifecycle: str) -> None:
        if not self._save_if_dirty():
            return
        reviewer = self.reviewer.text().strip()
        if not reviewer:
            self.message.setText(COPY["reviewer_required"])
            return
        try:
            service.save_decision(
                self.index_path, self.entries, self.current_id, lifecycle,
                reviewer, self.note.text(),
            )
            self.manifest, self.entries = review_index.load_review_index(self.index_path)
            self._render_page()
            self.message.setText(COPY["decision_saved"])
        except (OSError, ValueError) as exc:
            self.message.setText(f"{COPY['decision_error']} {exc}")

    def _export_selected(self) -> None:
        self._save_if_dirty()
        selected = sorted({item.row() for item in self.table.selectedItems()})
        if not selected:
            self.message.setText(COPY["selection_required"])
            return
        occurrence_ids = [str(self.page_rows[index]["occurrence_id"]) for index in selected]
        try:
            path = service.export_proposals(self.index_path, self.manifest or {}, self.entries, occurrence_ids)
            corpus_meta = (self.manifest or {}).get("inputs") or {}
            corpus_path = str((corpus_meta.get("corpus_manifest") or {}).get("path") or "")
            self.message.setText(COPY["proposal_saved"].format(count=len(selected)))
            self.proposal_ready.emit(path, corpus_path)
        except (OSError, ValueError) as exc:
            self.message.setText(f"{COPY['proposal_error']} {exc}")
