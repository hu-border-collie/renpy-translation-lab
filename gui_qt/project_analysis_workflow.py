"""Project Analysis lifecycle workflow used by the GUI context library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .translation_workflow import WorkflowStep, WorkflowUpdate

PROGRESS_PREFIX = "PROJECT_ANALYSIS_PROGRESS "


def _generation_stage_label(stage: str) -> str:
    return {
        "label": "场景摘要",
        "route": "路线摘要",
        "brief": "项目摘要",
        "complete": "全部完成",
    }.get(str(stage or ""), "生成中")


def generation_facts_from_output(output: str) -> list[str]:
    """Render the latest machine progress event as compact GUI facts."""
    latest: dict[str, Any] = {}
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line.startswith(PROGRESS_PREFIX):
            continue
        try:
            event = json.loads(line[len(PROGRESS_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            latest = event
    usage = latest.get("usage") if isinstance(latest.get("usage"), dict) else {}
    if not latest:
        return []
    facts = [
        f"生成进度：{_generation_stage_label(str(latest.get('stage') or ''))} "
        f"{latest.get('completed', 0)}/{latest.get('total', 0)}",
        f"模型请求：{usage.get('requests', 0)} · "
        f"输入 Token {usage.get('input_tokens', 0)} · "
        f"输出 Token {usage.get('output_tokens', 0)}",
    ]
    if usage.get("estimated_cost") is not None:
        facts.append(
            f"估算成本：{float(usage.get('estimated_cost') or 0.0):.6f} "
            f"{usage.get('currency') or 'USD'}（按当前配置价格表）"
        )
    return facts


STEP_TEXT = {
    "project-analysis-ingest-keywords": (
        "正在导入剧情概要",
        "正在把已确认的关键词剧情概要导入项目分析存储。",
    ),
    "project-analysis-build-structure": (
        "正在构建项目结构",
        "正在读取剧情节点、跳转与路线；不会修改游戏脚本。",
    ),
    "project-analysis-generate": (
        "正在生成项目摘要",
        "正在基于当前结构生成待审查的项目摘要。",
    ),
}


class ProjectAnalysisWorkflow:
    """Run an explicit, restartable ingest -> build -> generate sequence."""

    manifest_path = ""

    def __init__(self, steps: list[tuple[str, list[str]]]) -> None:
        self._steps = list(steps)

    @classmethod
    def start_new(
        cls,
        *,
        keyword_summary_path: str = "",
        build: bool = True,
        generate: bool = True,
    ) -> "ProjectAnalysisWorkflow":
        steps: list[tuple[str, list[str]]] = []
        summary_path = str(keyword_summary_path or "").strip()
        if summary_path:
            steps.append(
                (
                    "project-analysis-ingest-keywords",
                    ["project-analysis-ingest-keywords", "--summary-jsonl", summary_path],
                )
            )
        if build:
            steps.append(("project-analysis-build-structure", ["project-analysis-build-structure"]))
        if generate:
            steps.append(("project-analysis-generate", ["project-analysis-generate"]))
        return cls(steps)

    def current_step(self) -> WorkflowStep | None:
        if not self._steps:
            return None
        key, args = self._steps[0]
        heading, message = STEP_TEXT[key]
        return WorkflowStep(key=key, args=list(args), heading=heading, message=message)

    def step_keys(self) -> tuple[str, ...]:
        """Return the concrete remaining step sequence for GUI progress chrome."""
        return tuple(key for key, _args in self._steps)

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._steps:
            return WorkflowUpdate(
                status="failed",
                heading="项目分析流程状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )
        key, _args = self._steps.pop(0)
        if exit_code != 0:
            self._steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="项目分析流程中断",
                message=f"{STEP_TEXT[key][0]}没有正常完成；可从当前产物阶段重新开始。",
                facts=[],
            )
        next_step = self.current_step()
        if next_step is not None:
            return WorkflowUpdate(
                status="running",
                heading=next_step.heading,
                message=next_step.message,
                facts=[],
                should_continue=True,
            )
        return WorkflowUpdate(
            status="done",
            heading="项目分析摘要待审查",
            message="结构与摘要已生成。请审查全文、差异、证据与翻译使用预览后再启用。",
            facts=generation_facts_from_output(output),
        )


def keyword_summary_path_from_manifest(
    manifest_path: str,
    manifest: Mapping[str, Any] | None,
) -> str:
    """Resolve an exported keyword summary path from one batch manifest."""
    export = manifest.get("keyword_export") if isinstance(manifest, Mapping) else None
    if isinstance(export, Mapping):
        configured = str(export.get("summary_jsonl_path") or "").strip()
        configured_path = Path(configured) if configured else None
        if configured_path is not None and not configured_path.is_absolute() and manifest_path:
            configured_path = Path(manifest_path).parent / configured_path
        if configured_path is not None and configured_path.is_file():
            return str(configured_path)
    if manifest_path:
        candidate = Path(manifest_path).parent / "keyword_chunk_summaries.jsonl"
        if candidate.is_file():
            return str(candidate)
    return ""


def discover_keyword_summary_path(
    *,
    game_root: str,
    manifest_path: str = "",
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Return the newest known summary for the current project, if any."""
    candidates: list[Path] = []
    from_manifest = keyword_summary_path_from_manifest(manifest_path, manifest)
    if from_manifest:
        candidates.append(Path(from_manifest))
    root = Path(game_root) if game_root else None
    if root is not None:
        exported = root.parent / "extracted_keywords" / "keyword_chunk_summaries.jsonl"
        if exported.is_file():
            candidates.append(exported)
    if not candidates:
        return ""
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))
