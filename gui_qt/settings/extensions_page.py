"""Independent Extensions Settings page (#202 Phase D).

Owns relation-analyzer chrome. The OptionalFeatureInstallController, pip
QProcess, and docs open stay on the host. No translator_config fields.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QWidget,
)

from .page_chrome import build_settings_scroll_page, settings_group
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

EXTENSIONS_PAGE_KEY = "extensions"
_EXTENSIONS_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == EXTENSIONS_PAGE_KEY
)
EXTENSIONS_NAV_LABEL = _EXTENSIONS_SPEC.nav_label
EXTENSIONS_CONFIG_KEYS = _EXTENSIONS_SPEC.config_keys
EXTENSIONS_IMMEDIATE_ACTION_IDS = _EXTENSIONS_SPEC.immediate_action_ids

EXTENSIONS_WIDGET_ATTRS: tuple[str, ...] = (
    "relation_analyzer_status_label",
    "relation_analyzer_failure_label",
    "relation_analyzer_install_btn",
    "relation_analyzer_docs_btn",
    "relation_analyzer_install_progress",
)


class ExtensionsSettingsPage(QObject):
    """Settings page for optional feature chrome."""

    page_key = EXTENSIONS_PAGE_KEY
    nav_label = EXTENSIONS_NAV_LABEL
    config_keys = EXTENSIONS_CONFIG_KEYS
    immediate_action_ids = EXTENSIONS_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        on_install_relation_analyzer: Callable[[], None] | None = None,
        on_open_docs: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._on_install_relation_analyzer = on_install_relation_analyzer
        self._on_open_docs = on_open_docs
        self.widget, self.body = self._build_widgets()

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in EXTENSIONS_WIDGET_ATTRS:
            target[name] = getattr(self, name)

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        return None

    def collect(self) -> dict[str, object]:
        return {}

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        return None

    def focus_issue(self, issue: SettingsIssue) -> bool:
        if issue.field_key in {"install_relation_analyzer", "extensions"}:
            self.relation_analyzer_install_btn.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        # Installs run in the background and do not share the translation
        # task lock; host `_apply_relation_analyzer_status` owns the button.
        return None

    def _emit_install(self) -> None:
        if callable(self._on_install_relation_analyzer):
            self._on_install_relation_analyzer()

    def _emit_open_docs(self) -> None:
        if callable(self._on_open_docs):
            self._on_open_docs()

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_extensions")
        hint = QLabel(
            "扩展是按需安装的可选能力。是否启用取决于当前 Python 环境中的安装状态，"
            "不会单独保存“已启用”开关。安装在后台进行，不会阻止普通翻译任务。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        card, card_layout = settings_group("关系分析器")
        card.setObjectName("relation_analyzer_extension_card")

        purpose = QLabel(
            "从 Ren'Py TL 目录提取人物关系与语义相似度图。"
            "包含 NumPy、Matplotlib、scikit-learn、Pillow 等科学/图像组件；"
            "通过独立 CLI（extract_relations.py）运行，不会把重型库加载进 GUI 进程。"
        )
        purpose.setWordWrap(True)
        purpose.setObjectName("config_hint_label")
        card_layout.addWidget(purpose)

        components = QLabel(
            "组件：numpy · matplotlib · scikit-learn · pillow（及 scipy 传递依赖）"
        )
        components.setWordWrap(True)
        components.setObjectName("config_hint_label")
        card_layout.addWidget(components)

        self.relation_analyzer_status_label = QLabel()
        self.relation_analyzer_status_label.setObjectName(
            "relation_analyzer_status_label"
        )
        self.relation_analyzer_status_label.setWordWrap(True)
        card_layout.addWidget(self.relation_analyzer_status_label)

        self.relation_analyzer_failure_label = QLabel()
        self.relation_analyzer_failure_label.setObjectName(
            "relation_analyzer_failure_label"
        )
        self.relation_analyzer_failure_label.setWordWrap(True)
        self.relation_analyzer_failure_label.setVisible(False)
        card_layout.addWidget(self.relation_analyzer_failure_label)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.relation_analyzer_install_btn = QPushButton("安装并启用")
        self.relation_analyzer_install_btn.setObjectName(
            "relation_analyzer_install_btn"
        )
        self.relation_analyzer_install_btn.clicked.connect(self._emit_install)
        action_layout.addWidget(self.relation_analyzer_install_btn)
        self.relation_analyzer_docs_btn = QPushButton("使用说明")
        self.relation_analyzer_docs_btn.setObjectName("secondary_btn")
        self.relation_analyzer_docs_btn.clicked.connect(self._emit_open_docs)
        action_layout.addWidget(self.relation_analyzer_docs_btn)
        action_layout.addStretch(1)
        card_layout.addWidget(action_row)

        self.relation_analyzer_install_progress = QProgressBar()
        self.relation_analyzer_install_progress.setObjectName(
            "relation_analyzer_install_progress"
        )
        self.relation_analyzer_install_progress.setTextVisible(True)
        self.relation_analyzer_install_progress.setFormat("正在后台安装关系分析器…")
        self.relation_analyzer_install_progress.setVisible(False)
        card_layout.addWidget(self.relation_analyzer_install_progress)

        layout.addWidget(card)
        layout.addStretch(1)
        return page, body
