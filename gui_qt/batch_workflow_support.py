"""Shared Batch workflow helpers used by GUI and CLI argument planning."""
from __future__ import annotations

import json
from typing import Any


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


def build_submit_cli_args(
    manifest_path: str,
    submit_max_cost: float | None = None,
) -> list[str]:
    args = ["submit", manifest_path]
    if submit_max_cost is not None and submit_max_cost > 0:
        args.extend(["--max-cost", _format_max_cost(submit_max_cost)])
    return args


def _format_max_cost(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


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


def load_cost_estimate_facts_from_manifest(manifest_path: str) -> list[str]:
    if not manifest_path:
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, dict):
        return []
    estimate = manifest.get("cost_estimate")
    return format_cost_estimate_facts(estimate if isinstance(estimate, dict) else None)


def output_blocked_by_max_cost(output: str) -> bool:
    return "Submit blocked by --max-cost" in output