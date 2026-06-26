"""User-facing summaries for GUI sync translation runs."""
from __future__ import annotations

import re

from .translation_workflow import WorkflowUpdate


def summarize_sync_translation_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="同步翻译中断",
            message=_failure_message(output),
            facts=_collect_facts(output),
        )

    if "ERROR: No valid API keys found!" in output:
        return WorkflowUpdate(
            status="failed",
            heading="未配置 API Key",
            message="请先配置有效 API Key，或设置 GEMINI_API_KEY 环境变量。",
            facts=[],
        )

    facts = _collect_facts(output)
    files_done = len(re.findall(r"^\s*Done with .+\.$", output, re.MULTILINE))
    lines_to_translate = sum(
        int(match.group(1))
        for match in re.finditer(r"Found (\d+) lines to translate\.", output)
    )
    translated_count = sum(
        int(match.group(1))
        for match in re.finditer(r"Translated (\d+)/\d+ items\.", output)
    )

    if _tl_dir_missing(output):
        return WorkflowUpdate(
            status="failed",
            heading="翻译目录不存在",
            message="翻译目录不存在；请先运行环境检查或准备工作目录。",
            facts=facts,
        )

    if files_done == 0 and lines_to_translate == 0:
        return WorkflowUpdate(
            status="done",
            heading="没有待翻译内容",
            message="当前项目没有需要同步翻译的新行。",
            facts=facts,
        )

    if lines_to_translate > 0 and translated_count == 0:
        return WorkflowUpdate(
            status="failed",
            heading="同步翻译未完成",
            message="有待翻译行但未成功写回任何译文，请查看下方原始输出或失败日志。",
            facts=facts,
        )

    message = "同步翻译已完成；译文已按脚本现有规则写回项目文件，建议在游戏中抽查。"
    if files_done:
        message = f"已处理 {files_done} 个文件。{message}"
    if lines_to_translate > 0 and translated_count < lines_to_translate:
        message = (
            f"部分完成（已翻译 {translated_count}/{lines_to_translate} 行）。{message}"
        )
        return WorkflowUpdate(
            status="warning",
            heading="同步翻译部分完成",
            message=message,
            facts=facts,
        )

    return WorkflowUpdate(
        status="done",
        heading="同步翻译完成",
        message=message,
        facts=facts,
    )


def _tl_dir_missing(output: str) -> bool:
    return "TL_DIR does not exist" in output or "WARNING: TL_DIR does not exist" in output


def _failure_message(output: str) -> str:
    if "ERROR: No valid API keys found!" in output:
        return "请先配置有效 API Key，或设置 GEMINI_API_KEY 环境变量。"
    if _tl_dir_missing(output):
        return "翻译目录不存在；请先运行环境检查或准备工作目录。"
    return "同步翻译没有正常完成，请查看下方原始输出。"


def _collect_facts(output: str) -> list[str]:
    facts: list[str] = []
    for pattern, label in (
        (r"^Found (\d+) files\.$", "待处理文件"),
        (r"^Progress log: (.+)$", "进度日志"),
    ):
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            if label == "待处理文件":
                facts.append(f"{label}：{match.group(1)} 个")
            else:
                facts.append(f"{label}：{match.group(1).strip()}")
    return facts