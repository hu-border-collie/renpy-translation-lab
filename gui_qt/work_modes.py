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
    BOOTSTRAP_RAG = "bootstrap_rag"
    BOOTSTRAP_SOURCE_INDEX = "bootstrap_source_index"
    REVISION = "revision"


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
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.BOOTSTRAP_RAG,
            WorkMode.BOOTSTRAP_SOURCE_INDEX,
        ),
    ),
    TaskCategory.MAINTENANCE: TaskCategorySpec(
        category=TaskCategory.MAINTENANCE,
        label="维护",
        work_modes=(WorkMode.REVISION,),
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
        label="Batch 翻译",
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
        idle_workflow_message="适合小范围即时翻译、局部修复和 smoke test；写回行为遵循同步脚本现有规则，请先备份项目。",
        supports_resume=False,
        supports_translation_writeback=False,
        implemented=False,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode=None,
        not_implemented_message="同步翻译模式的图形界面仍在开发中，请暂时使用 gemini_translate.py。",
    ),
    WorkMode.KEYWORD_EXTRACTION: WorkModeSpec(
        mode=WorkMode.KEYWORD_EXTRACTION,
        category=TaskCategory.ANALYSIS_PREP,
        label="关键词提取",
        start_button_label="提取关键词",
        resume_button_label="继续提取",
        task_group_label="分析任务",
        progress_tab_label="提取进度",
        writeback_tab_label="结果说明",
        idle_workflow_heading="尚未开始关键词提取",
        idle_workflow_message="只生成术语与剧情候选报告，不会修改游戏 .rpy 文件。",
        supports_resume=True,
        supports_translation_writeback=False,
        implemented=False,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode="keyword_extraction",
        not_implemented_message="关键词提取模式的图形界面仍在开发中，请暂时使用 build-keywords 或 sync-keywords。",
    ),
    WorkMode.BOOTSTRAP_RAG: WorkModeSpec(
        mode=WorkMode.BOOTSTRAP_RAG,
        category=TaskCategory.ANALYSIS_PREP,
        label="预建 RAG 库",
        start_button_label="预建 RAG 库",
        resume_button_label="",
        task_group_label="预建任务",
        progress_tab_label="预建进度",
        writeback_tab_label="说明",
        idle_workflow_heading="尚未预建 RAG 库",
        idle_workflow_message="扫描已有译文并写入 Batch 记忆库；需先在配置页启用 RAG 并保存。",
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
        idle_workflow_message="只索引 TL 模板原文，不修改 .rpy；需先在配置页启用原文索引并保存。",
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
        writeback_tab_label="订正写回",
        idle_workflow_heading="尚未开始订正流程",
        idle_workflow_message="订正会生成修订建议；写回与普通翻译分离，需单独确认。",
        supports_resume=True,
        supports_translation_writeback=False,
        implemented=False,
        is_bootstrap=False,
        bootstrap_kind="",
        manifest_mode="revision",
        not_implemented_message="订正模式的图形界面仍在开发中，请暂时使用 build-revisions 或 sync-revisions。",
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