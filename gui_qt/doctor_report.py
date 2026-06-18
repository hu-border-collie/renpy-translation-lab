"""User-facing summaries for the GUI doctor command."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DoctorSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]


MODE_MESSAGES = {
    "can_generate_template": "Ren'Py 模板生成环境可用；如翻译模板尚不存在，需要先生成或刷新模板。",
    "existing_tl_only": "已有翻译文件可处理；模板生成环境不可用，后续依赖现有 TL 文件。",
    "blocked_missing_template": "缺少可处理的翻译文件，也无法自动生成模板。",
}


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _parse_counts(raw_counts: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in re.findall(r"([a-z_]+)=(-?\d+)", raw_counts):
        counts[key] = int(value)
    return counts


def parse_doctor_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {"warnings": []}
    in_warnings = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "Warnings:":
            in_warnings = True
            continue

        if in_warnings:
            if line.startswith("- "):
                parsed.setdefault("warnings", []).append(line[2:].strip())
                continue
            in_warnings = False

        if line.startswith("- Base dir:"):
            parsed["base_dir"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- TL dir:"):
            match = re.match(r"- TL dir:\s*(.*?)\s*\(exists:\s*(True|False)\)", line)
            if match:
                parsed["tl_dir"] = match.group(1).strip()
                parsed["tl_exists"] = _parse_bool(match.group(2))
            continue

        if line.startswith("- Language:"):
            parsed["language"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- Template generation:"):
            parsed["template_generation"] = line.split(":", 1)[1].strip()
            parsed["can_generate_template"] = line.startswith("- Template generation: available")
            continue

        if line.startswith("- Mode:"):
            parsed["mode"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- TL scan:"):
            parsed["counts"] = _parse_counts(line.split(":", 1)[1])
            continue

    return parsed


def summarize_doctor_output(
    output: str,
    exit_code: int,
    api_key_count: int | None = None,
    api_key_source: str = "",
) -> DoctorSummary:
    parsed = parse_doctor_output(output)
    warnings = [
        warning
        for warning in parsed.get("warnings", [])
        if isinstance(warning, str) and warning.strip()
    ]
    counts = parsed.get("counts") if isinstance(parsed.get("counts"), dict) else {}
    mode = parsed.get("mode") if isinstance(parsed.get("mode"), str) else ""

    facts: list[str] = []
    if parsed.get("base_dir"):
        facts.append(f"项目目录：{parsed['base_dir']}")
    if parsed.get("tl_dir"):
        exists_text = "存在" if parsed.get("tl_exists") is True else "不存在"
        facts.append(f"翻译目录：{exists_text}")
    if parsed.get("language"):
        facts.append(f"目标语言：{parsed['language']}")
    if mode:
        facts.append(f"检查模式：{mode}")
    if counts:
        rpy_files = int(counts.get("rpy_files", 0))
        old_lines = int(counts.get("old_lines", 0))
        new_lines = int(counts.get("new_lines", 0))
        facts.append(f"扫描到 {rpy_files} 个 .rpy 文件，old/new 行数 {old_lines}/{new_lines}")

    findings = list(warnings)
    if mode == "can_generate_template":
        if parsed.get("tl_exists") is False:
            findings.append(
                "翻译目录尚不存在；doctor 可以生成模板，但还没有可检查的 TL 文件。"
                "请先生成或刷新翻译模板后重新检查。"
            )
        elif counts and int(counts.get("rpy_files", 0)) == 0:
            findings.append(
                "翻译目录中没有可处理的 .rpy 文件；请先生成或刷新翻译模板后重新检查。"
            )
    if api_key_count is not None:
        if api_key_count > 0:
            if api_key_source == "environment":
                facts.append(f"API Key：已通过环境变量配置 {api_key_count} 个")
            else:
                facts.append(f"API Key：已配置 {api_key_count} 个")
        else:
            findings.append("尚未配置 API Key；doctor 不调用 Gemini，但后续翻译任务需要 API Key。")

    if exit_code != 0:
        return DoctorSummary(
            status="blocked",
            heading="项目检查失败",
            message="命令行检查没有正常完成，请查看下方诊断输出。",
            facts=facts,
            findings=findings,
        )

    if mode == "blocked_missing_template":
        status = "blocked"
        heading = "需要先准备翻译模板"
    elif findings:
        status = "warning"
        heading = "检查完成，但有需要处理的事项"
    else:
        status = "ready"
        heading = "项目检查通过"

    message = MODE_MESSAGES.get(mode, "项目检查已完成。")
    return DoctorSummary(
        status=status,
        heading=heading,
        message=message,
        facts=facts,
        findings=findings,
    )


def running_summary() -> DoctorSummary:
    return DoctorSummary(
        status="running",
        heading="正在检查项目",
        message="正在运行 doctor；完成后这里会显示可读摘要。",
        facts=[],
        findings=[],
    )


def idle_summary() -> DoctorSummary:
    return DoctorSummary(
        status="idle",
        heading="尚未运行项目检查",
        message="选择游戏 work 目录后运行 doctor。",
        facts=[],
        findings=[],
    )


def stale_summary() -> DoctorSummary:
    return DoctorSummary(
        status="stale",
        heading="项目已切换，请重新运行检查",
        message="当前摘要已清空；请针对新的 work 目录重新运行 doctor。",
        facts=[],
        findings=[],
    )
