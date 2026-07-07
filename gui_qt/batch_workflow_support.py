"""Shared Batch workflow helpers used by GUI and CLI argument planning."""
from __future__ import annotations

import json
import os
from typing import Any

import batch_submit_recovery


def resolve_submit_max_cost(translator_config: dict[str, Any] | None) -> float | None:
    if not isinstance(translator_config, dict):
        return None
    batch = translator_config.get("batch")
    if not isinstance(batch, dict):
        return None
    raw = batch.get("submit_max_cost")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _format_max_cost(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def load_manifest_dict(manifest_path: str) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def get_uncertain_submit_kind(manifest_path: str) -> str | None:
    manifest = load_manifest_dict(manifest_path)
    if not manifest or manifest.get("job_name"):
        return None
    package_dir = os.path.dirname(os.path.abspath(manifest_path))
    uncertain_state = batch_submit_recovery.get_uncertain_submit_state(
        manifest,
        package_dir=package_dir,
    )
    if not uncertain_state:
        return None
    kind = uncertain_state.get("kind")
    return kind.strip() if isinstance(kind, str) and kind.strip() else None


def plan_unsubmitted_workflow_steps(manifest_path: str) -> list[str]:
    kind = get_uncertain_submit_kind(manifest_path)
    if kind == "job_created_uncommitted":
        return ["recover-submit", "status"]
    return ["submit", "status"]


def build_submit_cli_args(
    manifest_path: str,
    submit_max_cost: float | None = None,
    *,
    resume: bool | None = None,
) -> list[str]:
    args = ["submit", manifest_path]
    if resume is None:
        resume = get_uncertain_submit_kind(manifest_path) == "upload_pending_job_create"
    if resume:
        args.append("--resume")
    if submit_max_cost is not None and submit_max_cost > 0:
        args.extend(["--max-cost", _format_max_cost(submit_max_cost)])
    return args


def build_recover_submit_cli_args(manifest_path: str) -> list[str]:
    return ["recover-submit", manifest_path]


def format_cost_estimate_facts(estimate: dict[str, Any] | None) -> list[str]:
    if not isinstance(estimate, dict):
        return []

    facts: list[str] = []
    model = estimate.get("model")
    if isinstance(model, str) and model.strip():
        facts.append(f"估算模型：{model.strip()}")

    input_tokens = estimate.get("estimated_input_tokens")
    if isinstance(input_tokens, int) and input_tokens >= 0:
        facts.append(f"估算输入 token：{input_tokens}")

    output_tokens = estimate.get("estimated_output_tokens_max")
    if isinstance(output_tokens, int) and output_tokens >= 0:
        facts.append(f"估算输出 token（上限）：{output_tokens}")

    currency = estimate.get("currency") if isinstance(estimate.get("currency"), str) else "USD"
    min_cost = estimate.get("estimated_cost_min")
    max_cost = estimate.get("estimated_cost_max")
    if isinstance(min_cost, (int, float)) and isinstance(max_cost, (int, float)):
        facts.append(f"估算成本：{float(min_cost):.4f} 至 {float(max_cost):.4f} {currency}")
    return facts


def load_target_language_facts_from_manifest(manifest_path: str) -> list[str]:
    manifest = load_manifest_dict(manifest_path)
    if not manifest:
        return []

    facts: list[str] = []
    tl_subdir = manifest.get("tl_subdir")
    if isinstance(tl_subdir, str) and tl_subdir.strip():
        facts.append(f"TL 路径：{tl_subdir.strip()}")
    target_language = manifest.get("target_language")
    if isinstance(target_language, str) and target_language.strip():
        facts.append(f"目标语言：{target_language.strip()}")
    return facts


def load_cost_estimate_facts_from_manifest(manifest_path: str) -> list[str]:
    manifest = load_manifest_dict(manifest_path)
    if not manifest:
        return []
    estimate = manifest.get("cost_estimate")
    return format_cost_estimate_facts(estimate if isinstance(estimate, dict) else None)


def load_uncertain_submit_facts_from_manifest(manifest_path: str) -> list[str]:
    manifest = load_manifest_dict(manifest_path)
    if not manifest:
        return []
    package_dir = os.path.dirname(os.path.abspath(manifest_path))
    uncertain_state = batch_submit_recovery.get_uncertain_submit_state(
        manifest,
        package_dir=package_dir,
    )
    facts = batch_submit_recovery.format_uncertain_submit_hints(uncertain_state)
    return [f"提交恢复：{fact}" for fact in facts]


def output_blocked_by_max_cost(output: str) -> bool:
    return "Submit blocked by --max-cost" in output


def output_blocked_by_uncertain_submit(output: str) -> bool:
    return batch_submit_recovery.BLOCKED_MESSAGE_PREFIX in output


def uncertain_submit_failure_message(output: str) -> str:
    if "recover-submit" in output.lower() or batch_submit_recovery.RECOVER_HINT in output:
        return (
            "检测到未完成的提交状态，且可能存在已创建的远端任务。"
            "请先运行 recover-submit 恢复任务，再刷新状态。"
        )
    if "--resume" in output:
        return (
            "检测到输入文件已上传但尚未创建批量任务。"
            "请使用带 --resume 的 submit 继续创建任务，或使用 --force 重新开始。"
        )
    return (
        "提交被未完成状态拦截。请先恢复或确认上次提交，避免重复创建付费任务。"
    )