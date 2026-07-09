"""Workbench task category and work-task definitions for the optional GUI shell."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskCategory(str, Enum):
    TRANSLATION = "translation"
    ANALYSIS_PREP = "analysis_prep"
    MAINTENANCE = "maintenance"


class WorkMode(str, Enum):
    BATCH_TRANSLATION = "batch_translation"
    SYNC_TRANSLATION = "sync_translation"
    KEYWORD_EXTRACTION = "keyword_extraction"
    SYNC_KEYWORD_EXTRACTION = "sync_keyword_extraction"
    BOOTSTRAP_RAG = "bootstrap_rag"
    BOOTSTRAP_SOURCE_INDEX = "bootstrap_source_index"
    REVISION = "revision"
    SYNC_REVISION = "sync_revision"


@dataclass(frozen=True)
class TaskCategorySpec:
    category: TaskCategory
    label: str
    work_modes: tuple[WorkMode, ...]


@dataclass(frozen=True)
class WorkModeSpec:
    mode: WorkMode
    category: TaskCategory
    label: str
    start_button_label: str
    resume_button_label: str
    task_group_label: str
    progress_tab_label: str
    writeback_tab_label: str
    idle_workflow_heading: str
    idle_workflow_message: str
    supports_resume: bool
    supports_translation_writeback: bool
    implemented: bool
    is_bootstrap: bool
    bootstrap_kind: str
    manifest_mode: str | None
    not_implemented_message: str


TASK_CATEGORY_SPECS: dict[TaskCategory, TaskCategorySpec] = {
    TaskCategory.TRANSLATION: TaskCategorySpec(
        category=TaskCategory.TRANSLATION,
        label="翻译",
        work_modes=(
            WorkMode.BATCH_TRANSLATION,
            WorkMode.SYNC_TRANSLATION,
        ),
    ),
    TaskCategory.ANALYSIS_PREP: TaskCategorySpec(
        category=TaskCategory.ANALYSIS_PREP,
        label="分析与准备",
        work_modes=(
            WorkMode.BOOTSTRAP_RAG,
            WorkMode.BOOTSTRAP_SOURCE_INDEX,
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.SYNC_KEYWORD_EXTRACTION,
        ),
    ),
    TaskCategory.MAINTENANCE: TaskCategorySpec(
        category=TaskCategory.MAINTENANCE,
        label="维护",
        work_modes=(
            WorkMode.REVISION,
            WorkMode.SYNC_REVISION,
        ),
    ),
}


TASK_CATEGORY_ORDER: tuple[TaskCategory, ...] = (
    TaskCategory.TRANSLATION,
    TaskCategory.ANALYSIS_PREP,
    TaskCategory.MAINTENANCE,
)


WORK_MODE_SPECS: dict[WorkMode, WorkModeSpec] = {
    WorkMode.BATCH_TRANSLATION: WorkModeSpec(
        mode=WorkMode.BATCH_TRANSLATION,
        category=TaskCategory.TRANSLATION,
        label="批量翻译",
        start_button_label="开始翻译",
        resume_button_label="继续翻译",
        task_group_label="翻译任务",
        progress_tab_label="翻译进度",
        writeback_tab_label="写回",
        idle_workflow_heading="尚未开始翻译任务",
        idle_workflow_message="完成环境检查后，可以开始批量翻译流程。",
        supports_resume=True,
        supports_translation_writeback=True,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode="translation",
        not_implemented_message="",
    ),
    WorkMode.SYNC_TRANSLATION: WorkModeSpec(
        mode=WorkMode.SYNC_TRANSLATION,
        category=TaskCategory.TRANSLATION,
        label="同步翻译",
        start_button_label="开始翻译",
        resume_button_label="继续翻译",
        task_group_label="同步任务",
        progress_tab_label="翻译进度",
        writeback_tab_label="写回说明",
        idle_workflow_heading="尚未开始同步翻译",
        idle_workflow_message="适合小范围试译或局部修改；可能直接改项目文件，请先备份。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode=None,
        not_implemented_message="",
    ),
    WorkMode.KEYWORD_EXTRACTION: WorkModeSpec(
        mode=WorkMode.KEYWORD_EXTRACTION,
        category=TaskCategory.ANALYSIS_PREP,
        label="批量关键词",
        start_button_label="提取关键词",
        resume_button_label="继续提取",
        task_group_label="分析任务",
        progress_tab_label="提取进度",
        writeback_tab_label="结果说明",
        idle_workflow_heading="尚未开始批量关键词提取",
        idle_workflow_message=(
            "会扫描翻译文本，批量提取术语与剧情概要；"
            "只生成报告，不修改游戏脚本。"
        ),
        supports_resume=True,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode="keyword_extraction",
        not_implemented_message="",
    ),
    WorkMode.SYNC_KEYWORD_EXTRACTION: WorkModeSpec(
        mode=WorkMode.SYNC_KEYWORD_EXTRACTION,
        category=TaskCategory.ANALYSIS_PREP,
        label="同步关键词",
        start_button_label="提取关键词",
        resume_button_label="继续提取",
        task_group_label="同步分析任务",
        progress_tab_label="提取进度",
        writeback_tab_label="结果说明",
        idle_workflow_heading="尚未开始同步关键词提取",
        idle_workflow_message="适合小范围即时生成术语与剧情报告；不修改游戏脚本。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode=None,
        not_implemented_message="",
    ),
    WorkMode.BOOTSTRAP_RAG: WorkModeSpec(
        mode=WorkMode.BOOTSTRAP_RAG,
        category=TaskCategory.ANALYSIS_PREP,
        label="预建记忆库",
        start_button_label="预建记忆库",
        resume_button_label="",
        task_group_label="预建任务",
        progress_tab_label="预建进度",
        writeback_tab_label="说明",
        idle_workflow_heading="尚未预建记忆库",
        idle_workflow_message="扫描已有译文并写入本地记忆库；需先在配置页启用并保存配置。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=True,
        bootstrap_kind="rag",
        manifest_mode=None,
        not_implemented_message="",
    ),
    WorkMode.BOOTSTRAP_SOURCE_INDEX: WorkModeSpec(
        mode=WorkMode.BOOTSTRAP_SOURCE_INDEX,
        category=TaskCategory.ANALYSIS_PREP,
        label="预建原文索引",
        start_button_label="预建原文索引",
        resume_button_label="",
        task_group_label="预建任务",
        progress_tab_label="预建进度",
        writeback_tab_label="说明",
        idle_workflow_heading="尚未预建原文索引",
        idle_workflow_message="只索引翻译模板里的原文，不修改游戏脚本；需先在配置页启用并保存配置。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=True,
        bootstrap_kind="source_index",
        manifest_mode=None,
        not_implemented_message="",
    ),
    WorkMode.REVISION: WorkModeSpec(
        mode=WorkMode.REVISION,
        category=TaskCategory.MAINTENANCE,
        label="订正",
        start_button_label="生成订正预览",
        resume_button_label="继续订正",
        task_group_label="订正任务",
        progress_tab_label="订正进度",
        writeback_tab_label="写回订正",
        idle_workflow_heading="尚未开始订正流程",
        idle_workflow_message="会批量生成订正预览；确认后再单独写回，与普通翻译写回分开。",
        supports_resume=True,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode="revision",
        not_implemented_message="",
    ),
    WorkMode.SYNC_REVISION: WorkModeSpec(
        mode=WorkMode.SYNC_REVISION,
        category=TaskCategory.MAINTENANCE,
        label="同步订正",
        start_button_label="生成订正预览",
        resume_button_label="继续订正",
        task_group_label="同步订正任务",
        progress_tab_label="订正进度",
        writeback_tab_label="写回订正",
        idle_workflow_heading="尚未开始同步订正",
        idle_workflow_message="适合小范围订正预览；默认只出报告，不会自动写回，请先备份。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=True,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode=None,
        not_implemented_message="",
    ),
}


def task_category_spec(category: TaskCategory | str) -> TaskCategorySpec:
    if isinstance(category, TaskCategory):
        return TASK_CATEGORY_SPECS[category]
    return TASK_CATEGORY_SPECS[TaskCategory(category)]


def work_mode_spec(mode: WorkMode | str) -> WorkModeSpec:
    if isinstance(mode, WorkMode):
        return WORK_MODE_SPECS[mode]
    return WORK_MODE_SPECS[WorkMode(mode)]


def normalize_work_mode(value: WorkMode | str | None) -> WorkMode:
    if isinstance(value, WorkMode):
        return value
    if isinstance(value, str) and value.strip():
        return WorkMode(value.strip())
    return WorkMode.BATCH_TRANSLATION


def normalize_task_category(value: TaskCategory | str | None) -> TaskCategory:
    if isinstance(value, TaskCategory):
        return value
    if isinstance(value, str) and value.strip():
        return TaskCategory(value.strip())
    return TaskCategory.TRANSLATION


def task_category_for_work_mode(mode: WorkMode | str) -> TaskCategory:
    return work_mode_spec(mode).category


def work_modes_for_category(category: TaskCategory | str) -> tuple[WorkMode, ...]:
    return task_category_spec(category).work_modes


def default_work_mode_for_category(category: TaskCategory | str) -> WorkMode:
    modes = work_modes_for_category(category)
    for mode in modes:
        if work_mode_spec(mode).implemented:
            return mode
    return modes[0] if modes else WorkMode.BATCH_TRANSLATION


def manifest_mode_for_work_mode(mode: WorkMode | str) -> str | None:
    return work_mode_spec(mode).manifest_mode


def work_mode_from_manifest_mode(manifest_mode: object) -> WorkMode | None:
    text = manifest_mode.strip() if isinstance(manifest_mode, str) else ""
    if not text or text == "translation":
        return WorkMode.BATCH_TRANSLATION
    for spec in WORK_MODE_SPECS.values():
        if spec.manifest_mode == text:
            return spec.mode
    return None


def bootstrap_disabled_message(kind: str) -> str:
    if kind == "rag":
        return "请先在配置页勾选「启用记忆库」，并点击「保存参数配置」。"
    return "请先在配置页勾选「启用原文索引」，并点击「保存参数配置」。"


_BOOTSTRAP_DISABLED_HINTS = (
    bootstrap_disabled_message("rag"),
    bootstrap_disabled_message("source_index"),
)


def work_mode_hint_texts() -> tuple[str, ...]:
    """All strings that may appear in the work-mode selector hint label."""
    texts: list[str] = []
    for spec in WORK_MODE_SPECS.values():
        if spec.idle_workflow_message.strip():
            texts.append(spec.idle_workflow_message.strip())
        if spec.not_implemented_message.strip():
            texts.append(spec.not_implemented_message.strip())
    texts.extend(_BOOTSTRAP_DISABLED_HINTS)
    return tuple(texts)


# ---------------------------------------------------------------------------
# Workbench left-nav (GUI IA P1a / #160)
# ---------------------------------------------------------------------------


class WorkbenchNavItem(str, Enum):
    """Top-level workbench navigation entries (not the same as WorkMode)."""

    BATCH_TRANSLATION = "batch_translation"
    SYNC_TRANSLATION = "sync_translation"
    KEYWORDS = "keywords"
    REVISION = "revision"
    CONTEXT = "context"


@dataclass(frozen=True)
class WorkbenchNavSpec:
    item: WorkbenchNavItem
    label: str
    work_modes: tuple[WorkMode, ...]
    show_submode: bool


WORKBENCH_NAV_SPECS: dict[WorkbenchNavItem, WorkbenchNavSpec] = {
    WorkbenchNavItem.BATCH_TRANSLATION: WorkbenchNavSpec(
        item=WorkbenchNavItem.BATCH_TRANSLATION,
        label="批量翻译",
        work_modes=(WorkMode.BATCH_TRANSLATION,),
        show_submode=False,
    ),
    WorkbenchNavItem.SYNC_TRANSLATION: WorkbenchNavSpec(
        item=WorkbenchNavItem.SYNC_TRANSLATION,
        label="同步翻译",
        work_modes=(WorkMode.SYNC_TRANSLATION,),
        show_submode=False,
    ),
    WorkbenchNavItem.KEYWORDS: WorkbenchNavSpec(
        item=WorkbenchNavItem.KEYWORDS,
        label="关键词 / 术语",
        work_modes=(WorkMode.KEYWORD_EXTRACTION, WorkMode.SYNC_KEYWORD_EXTRACTION),
        show_submode=True,
    ),
    WorkbenchNavItem.REVISION: WorkbenchNavSpec(
        item=WorkbenchNavItem.REVISION,
        label="订正",
        work_modes=(WorkMode.REVISION, WorkMode.SYNC_REVISION),
        show_submode=True,
    ),
    WorkbenchNavItem.CONTEXT: WorkbenchNavSpec(
        item=WorkbenchNavItem.CONTEXT,
        label="上下文库",
        work_modes=(WorkMode.BOOTSTRAP_RAG, WorkMode.BOOTSTRAP_SOURCE_INDEX),
        show_submode=True,
    ),
}


WORKBENCH_NAV_ORDER: tuple[WorkbenchNavItem, ...] = (
    WorkbenchNavItem.BATCH_TRANSLATION,
    WorkbenchNavItem.SYNC_TRANSLATION,
    WorkbenchNavItem.KEYWORDS,
    WorkbenchNavItem.REVISION,
    WorkbenchNavItem.CONTEXT,
)


def workbench_nav_spec(item: WorkbenchNavItem | str) -> WorkbenchNavSpec:
    if isinstance(item, WorkbenchNavItem):
        return WORKBENCH_NAV_SPECS[item]
    return WORKBENCH_NAV_SPECS[WorkbenchNavItem(item)]


def workbench_nav_for_work_mode(mode: WorkMode | str) -> WorkbenchNavItem:
    mode = normalize_work_mode(mode)
    for item in WORKBENCH_NAV_ORDER:
        if mode in workbench_nav_spec(item).work_modes:
            return item
    return WorkbenchNavItem.BATCH_TRANSLATION


def default_work_mode_for_nav(item: WorkbenchNavItem | str) -> WorkMode:
    modes = workbench_nav_spec(item).work_modes
    for mode in modes:
        if work_mode_spec(mode).implemented:
            return mode
    return modes[0]