"""Human review workspace for Project Analysis artifacts."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QGuiApplication,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from project_analysis import (
    KIND_CHUNK,
    KIND_LABEL,
    KIND_PROJECT_BRIEF,
    KIND_ROUTE,
    load_injectable_project_brief,
    mark_project_brief_reviewed,
    normalize_lineage,
    resolve_project_analysis_store,
)
from .user_copy import PROJECT_ANALYSIS_COPY


class UnifiedDiffHighlighter(QSyntaxHighlighter):
    """Use restrained semantic colors to make a full unified diff scannable."""

    def highlightBlock(self, text: str) -> None:
        style = QTextCharFormat()
        if text.startswith("+") and not text.startswith("+++"):
            style.setForeground(QColor("#18794e"))
            style.setBackground(QColor("#e7f6ec"))
        elif text.startswith("-") and not text.startswith("---"):
            style.setForeground(QColor("#b42318"))
            style.setBackground(QColor("#fdecec"))
        elif text.startswith("@@"):
            style.setForeground(QColor("#175cd3"))
        else:
            return
        self.setFormat(0, len(text), style)


def _full_unified_diff(published: str, draft: str) -> str:
    if not published and not draft:
        return "尚无 draft 或 published brief。"
    if published == draft:
        return "draft 与 published 完全一致。"
    return "".join(
        difflib.unified_diff(
            published.splitlines(keepends=True),
            draft.splitlines(keepends=True),
            fromfile="published",
            tofile="draft",
        )
    )


def build_project_analysis_review_data(
    *,
    base_dir: str,
    live_fingerprint: str,
    inject_enabled: bool,
    max_brief_chars: int,
) -> dict[str, Any]:
    """Load full review content and the exact brief that translation would inject."""
    store = resolve_project_analysis_store(base_dir=base_dir)
    draft = store.load_brief_text(published=False)
    published = store.load_brief_text(published=True)
    manifest = store.load_manifest() or {}
    brief = (manifest.get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {}
    records: list[dict[str, Any]] = []
    for kind, rows in (
        (KIND_CHUNK, store.load_summaries(KIND_CHUNK)),
        (KIND_LABEL, store.load_summaries(KIND_LABEL)),
        (KIND_ROUTE, store.load_routes()),
    ):
        for row in rows:
            record = dict(row)
            record["kind"] = kind
            records.append(record)
    injection = load_injectable_project_brief(
        store.store_dir,
        expected_source_fingerprint=live_fingerprint,
        max_chars=max_brief_chars,
        enabled=inject_enabled,
    )
    return {
        "store_dir": store.store_dir,
        "draft": draft,
        "published": published,
        "diff": _full_unified_diff(published, draft),
        "records": records,
        "reviewed_at": normalize_lineage(brief.get("lineage")).get("reviewed_at") or "",
        "injection": injection,
        "max_brief_chars": max_brief_chars,
        "injection_truncated": bool(
            injection.get("injectable")
            and max_brief_chars > 0
            and len(published.strip()) > max_brief_chars
        ),
    }


class ProjectAnalysisReviewDialog(QDialog):
    """Searchable full-content review, evidence navigation, and publish handoff."""

    def __init__(
        self,
        *,
        base_dir: str,
        live_fingerprint: str,
        inject_enabled: bool,
        max_brief_chars: int = 4000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.base_dir = str(base_dir)
        self.requested_action = ""
        self.data = build_project_analysis_review_data(
            base_dir=self.base_dir,
            live_fingerprint=live_fingerprint,
            inject_enabled=inject_enabled,
            max_brief_chars=max_brief_chars,
        )
        self.setWindowTitle(PROJECT_ANALYSIS_COPY["review_title"])
        self.resize(1040, 720)

        root = QVBoxLayout(self)
        heading = QLabel(PROJECT_ANALYSIS_COPY["review_heading"])
        heading.setObjectName("dialog_heading")
        root.addWidget(heading)
        self.review_status = QLabel()
        self.review_status.setWordWrap(True)
        root.addWidget(self.review_status)
        self._refresh_review_status()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_diff_tab(), "完整差异")
        self.tabs.addTab(self._build_brief_tab(), "摘要全文")
        self.tabs.addTab(self._build_evidence_tab(), "证据与定位")
        self.tabs.addTab(self._build_injection_tab(), "实际注入预览")
        root.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.review_btn = QPushButton(PROJECT_ANALYSIS_COPY["review_confirm"])
        self.review_btn.setObjectName("secondary_btn")
        self.review_btn.clicked.connect(self._record_review)
        footer.addWidget(self.review_btn)
        footer.addStretch(1)
        if self.data["published"].strip():
            self.unpublish_btn = QPushButton("停止用于翻译")
            self.unpublish_btn.setObjectName("secondary_btn")
            self.unpublish_btn.clicked.connect(self._request_unpublish)
            footer.addWidget(self.unpublish_btn)
        self.publish_btn = QPushButton(PROJECT_ANALYSIS_COPY["review_publish"])
        self.publish_btn.setObjectName("primary_btn")
        self.publish_btn.setEnabled(bool(self.data["draft"].strip()))
        self.publish_btn.clicked.connect(self._request_publish)
        footer.addWidget(self.publish_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _readonly(self, text: str) -> QPlainTextEdit:
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        view.setPlainText(text)
        return view

    def _build_diff_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel("“+” 是当前 draft 新增内容，“-” 是相对已发布版本删除的内容。")
        note.setWordWrap(True)
        layout.addWidget(note)
        diff_view = self._readonly(self.data["diff"])
        self._diff_highlighter = UnifiedDiffHighlighter(diff_view.document())
        layout.addWidget(diff_view, 1)
        return page

    def _build_brief_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        for title, text in (
            ("当前 draft", self.data["draft"]),
            ("已发布版本", self.data["published"]),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(f"{title} · {len(text)} 字符"))
            column.addWidget(self._readonly(text or "（空）"), 1)
            layout.addLayout(column, 1)
        return page

    def _build_evidence_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索 ID、摘要、来源文件或 evidence item…")
        self.search.textChanged.connect(self._filter_records)
        layout.addWidget(self.search)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.record_list = QListWidget()
        self.record_list.setMinimumWidth(300)
        self.record_list.currentItemChanged.connect(self._show_record)
        splitter.addWidget(self.record_list)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        self.record_detail = self._readonly("")
        detail_layout.addWidget(self.record_detail, 1)
        actions = QHBoxLayout()
        self.open_source_btn = QPushButton("打开来源文件")
        self.open_source_btn.clicked.connect(self._open_selected_source)
        actions.addWidget(self.open_source_btn)
        self.copy_location_btn = QPushButton("复制文件与行号")
        self.copy_location_btn.clicked.connect(self._copy_selected_location)
        actions.addWidget(self.copy_location_btn)
        actions.addStretch(1)
        detail_layout.addLayout(actions)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self._populate_records()
        return page

    def _build_injection_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        injection = self.data["injection"]
        state = "会注入" if injection.get("injectable") else "不会注入"
        truncation = " · 已按上限截断" if self.data["injection_truncated"] else ""
        reason = str(injection.get("reason") or "通过发布与 freshness 门禁")
        header = QLabel(
            f"当前翻译行为：{state}{truncation} · 上限 {self.data['max_brief_chars']} 字符\n"
            f"原因：{reason}\n诊断：{injection.get('diagnostics') or '无'}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        layout.addWidget(self._readonly(str(injection.get("text") or "（无注入内容）")), 1)
        return page

    def _populate_records(self, query: str = "") -> None:
        current_id = ""
        if self.record_list.currentItem() is not None:
            current_id = str(self.record_list.currentItem().data(Qt.ItemDataRole.UserRole) or "")
        self.record_list.clear()
        needle = query.casefold().strip()
        for index, record in enumerate(self.data["records"]):
            haystack = " ".join(
                [
                    str(record.get("id") or ""),
                    str(record.get("title") or ""),
                    str(record.get("summary") or ""),
                    " ".join(record.get("source_files") or []),
                    " ".join(record.get("evidence_item_ids") or []),
                ]
            ).casefold()
            if needle and needle not in haystack:
                continue
            kind = str(record.get("kind") or "artifact")
            label = str(record.get("title") or record.get("id") or "未命名")
            item = QListWidgetItem(f"{kind} · {label}")
            item.setData(Qt.ItemDataRole.UserRole, str(index))
            self.record_list.addItem(item)
            if str(index) == current_id:
                self.record_list.setCurrentItem(item)
        if self.record_list.currentRow() < 0 and self.record_list.count():
            self.record_list.setCurrentRow(0)

    def _filter_records(self, text: str) -> None:
        self._populate_records(text)

    def _selected_record(self) -> dict[str, Any] | None:
        item = self.record_list.currentItem()
        if item is None:
            return None
        try:
            return self.data["records"][int(item.data(Qt.ItemDataRole.UserRole))]
        except (TypeError, ValueError, IndexError):
            return None

    def _show_record(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        del current
        record = self._selected_record()
        if record is None:
            self.record_detail.clear()
            return
        span = record.get("line_span") or []
        span_text = f"{span[0]}-{span[1]}" if len(span) == 2 else "未记录"
        sources = "\n".join(f"- {path}" for path in record.get("source_files") or []) or "- 未记录"
        evidence = (
            "\n".join(f"- {item}" for item in record.get("evidence_item_ids") or []) or "- 未记录"
        )
        self.record_detail.setPlainText(
            f"ID：{record.get('id') or ''}\n类型：{record.get('kind') or ''}\n"
            f"状态：{record.get('status') or ''}\n行号：{span_text}\n\n"
            f"来源文件：\n{sources}\n\nEvidence items：\n{evidence}\n\n"
            f"摘要全文：\n{record.get('summary') or '（空）'}"
        )

    def _source_location(self) -> tuple[Path | None, str]:
        record = self._selected_record() or {}
        files = record.get("source_files") or []
        if not files:
            return None, ""
        path = Path(str(files[0]))
        if not path.is_absolute():
            path = Path(self.base_dir) / path
        span = record.get("line_span") or []
        suffix = f":{span[0]}" if len(span) == 2 else ""
        return path, f"{path}{suffix}"

    def _open_selected_source(self) -> None:
        path, _location = self._source_location()
        if path is None or not path.is_file():
            QMessageBox.information(self, "来源不可用", "该条目没有可打开的来源文件。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _copy_selected_location(self) -> None:
        _path, location = self._source_location()
        if location:
            QGuiApplication.clipboard().setText(location)

    def _refresh_review_status(self) -> None:
        reviewed_at = str(self.data.get("reviewed_at") or "")
        self.review_status.setText(
            f"审查记录：{reviewed_at if reviewed_at else '尚未确认'} · "
            f"存储：{self.data['store_dir']}"
        )

    def _record_review(self) -> bool:
        try:
            result = mark_project_brief_reviewed(base_dir=self.base_dir)
        except Exception as exc:
            QMessageBox.warning(self, "无法记录审查", str(exc))
            return False
        self.data["reviewed_at"] = result["reviewed_at"]
        self._refresh_review_status()
        return True

    def _request_publish(self) -> None:
        if (
            QMessageBox.question(
                self,
                PROJECT_ANALYSIS_COPY["publish_confirm_title"],
                PROJECT_ANALYSIS_COPY["publish_confirm_body"],
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        if self._record_review():
            self.requested_action = "publish"
            self.accept()

    def _request_unpublish(self) -> None:
        if (
            QMessageBox.question(
                self,
                PROJECT_ANALYSIS_COPY["unpublish_confirm_title"],
                PROJECT_ANALYSIS_COPY["unpublish_confirm_body"],
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.requested_action = "unpublish"
            self.accept()
