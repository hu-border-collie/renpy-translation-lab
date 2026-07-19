"""User-facing summaries for GUI sync translation runs."""
from __future__ import annotations

import re

from .translation_workflow import WorkflowUpdate


def summarize_sync_translation_output(
    output: str,
    exit_code: int,
    *,
    operation: str = "preview",
) -> WorkflowUpdate:
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
    if operation == "apply":
        if "Sync translation apply complete." not in output:
            return WorkflowUpdate(
                status="failed",
                heading="同步翻译写回未完成",
                message="写回命令没有返回完成标记，请查看原始输出。",
                facts=facts,
            )
        return WorkflowUpdate(
            status="done",
            heading="同步翻译已写回",
            message="预览已通过项目与源文件复核，并原子写入项目脚本。",
            facts=facts,
        )

    files_done = len(re.findall(r"^\s*Previewed .+\.$", output, re.MULTILINE))
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

    if "Sync preview manifest:" not in output or "Preview status: safe" not in output:
        return WorkflowUpdate(
            status="failed",
            heading="同步翻译预览未完成",
            message="没有生成可写回的安全预览，请查看下方原始输出。",
            facts=facts,
        )

    message = "同步翻译预览已生成；项目脚本尚未修改。请检查 diff 后再确认写回。"
    if files_done:
        message = f"已处理 {files_done} 个文件。{message}"
    if lines_to_translate > 0 and translated_count < lines_to_translate:
        message = (
            f"部分完成（已翻译 {translated_count}/{lines_to_translate} 行）。{message}"
        )
        return WorkflowUpdate(
            status="warning",
            heading="同步翻译预览部分完成",
            message=message,
            facts=facts,
        )

    return WorkflowUpdate(
        status="done",
        heading="同步翻译预览可写回",
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
        (r"^Sync preview manifest: (.+)$", "预览清单"),
        (r"^Sync preview report: (.+)$", "差异报告"),
        (r"^Applied files: (\d+)$", "已写回文件"),
    ):
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            if label in {"待处理文件", "已写回文件"}:
                facts.append(f"{label}：{match.group(1)} 个")
            else:
                facts.append(f"{label}：{match.group(1).strip()}")
    return facts
