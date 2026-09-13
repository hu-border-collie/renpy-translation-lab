"""Pure coverage/review helpers shared by GUI workers and doctor summaries.

The GUI must present the same coverage + independent-review contract as the
CLI (#424 P6 Slice 3c). This module intentionally has no PySide6 import so the
contract and formatting can be tested without a Qt install; the Qt worker
wrapper lives in :mod:`gui_qt.coverage_worker`.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from .user_copy import (
    COVERAGE_REVIEW_COPY,
    coverage_gate_status_label,
    coverage_review_policy_label,
    coverage_review_status_label,
    coverage_status_label,
)


@dataclass(frozen=True)
class CoverageReviewImportResult:
    """Outcome of one ``coverage-review-import`` call."""

    ok: bool
    payload: dict[str, Any] | None = None
    error: str = ""
    error_code: str = ""
    suggested_action: str = ""
    retryable: bool = False

    def user_message(self) -> str:
        """Return one user-facing message without parsing free-form output."""
        if self.ok and self.payload is not None:
            return str(self.payload.get("review_path") or "")
        parts = [self.error or "导入覆盖核对结果失败。"]
        if self.error_code:
            parts.append(f"错误码：{self.error_code}")
        if self.suggested_action:
            parts.append(self.suggested_action)
        return "\n".join(parts)


def import_coverage_review(review_path: str) -> CoverageReviewImportResult:
    """Validate and install one completed review through the CLI contract."""
    source = str(review_path or "").strip()
    if not source:
        return CoverageReviewImportResult(
            ok=False,
            error="没有选择覆盖核对 JSON 文件。",
            error_code="COVERAGE_REVIEW_FILE_REQUIRED",
            suggested_action="请选择 coverage 包中的 coverage_review.json 后重试。",
        )
    try:
        import cli_contract
        import gemini_translate_batch as batch_mod
    except Exception as exc:  # pragma: no cover - defensive GUI boundary
        return CoverageReviewImportResult(
            ok=False,
            error=f"无法加载覆盖核对模块：{exc}",
            error_code="COVERAGE_REVIEW_IMPORT_UNAVAILABLE",
        )
    try:
        payload = batch_mod.run_coverage_review_import(
            SimpleNamespace(file=source, dry_run=False)
        )
    except cli_contract.MachineContractError as exc:
        return CoverageReviewImportResult(
            ok=False,
            error=str(exc),
            error_code=exc.code_name,
            suggested_action=exc.suggested_action,
            retryable=exc.retryable,
        )
    except Exception as exc:  # pragma: no cover - defensive GUI boundary
        return CoverageReviewImportResult(
            ok=False,
            error=f"导入覆盖核对结果失败：{exc}",
            error_code="COVERAGE_REVIEW_IMPORT_FAILED",
        )
    if not isinstance(payload, Mapping):
        return CoverageReviewImportResult(
            ok=False,
            error="覆盖核对导入未返回结构化结果。",
            error_code="COVERAGE_REVIEW_IMPORT_FAILED",
        )
    return CoverageReviewImportResult(ok=True, payload=dict(payload))


def _classification_counts_text(counts: Mapping[str, Any]) -> str:
    labels = COVERAGE_REVIEW_COPY["classification_labels"]
    parts = []
    for key in sorted(counts):
        try:
            value = int(counts.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            continue
        parts.append(f"{labels.get(str(key), str(key))}={value}")
    return "、".join(parts)


def _gate_reason_labels(gate: Mapping[str, Any]) -> list[str]:
    labels = COVERAGE_REVIEW_COPY["reason_labels"]
    result = []
    for reason in gate.get("reasons") or ():
        text = str(reason or "")
        if text:
            result.append(labels.get(text, text))
    return result


def _locator_text(locator: Any) -> str:
    payload = locator.get("locator") if isinstance(locator, Mapping) else None
    if not isinstance(payload, Mapping):
        payload = locator if isinstance(locator, Mapping) else {}
    file_rel_path = str(payload.get("file_rel_path") or payload.get("path") or "")
    line = None
    for key in ("line", "line_hint", "line_number", "ordinal"):
        value = payload.get(key)
        if value is not None:
            line = str(value)
            break
    if not file_rel_path:
        return "(locator unavailable)"
    if line:
        return f"{file_rel_path}:{line}"
    return file_rel_path


def format_coverage_facts(coverage: Mapping[str, Any] | None) -> list[str]:
    """Return DoctorSummary primary facts for the coverage/review block."""
    if not isinstance(coverage, Mapping) or not coverage:
        return []
    facts: list[str] = []
    status = str(coverage.get("status") or "unknown")
    completion = str(coverage.get("completion") or "unconfirmed")
    completion_label = COVERAGE_REVIEW_COPY["completion_labels"].get(
        completion,
        completion,
    )
    facts.append(f"文本覆盖：{coverage_status_label(status)}；{completion_label}")

    counts = coverage.get("classification_counts")
    if isinstance(counts, Mapping):
        counts_text = _classification_counts_text(counts)
        if counts_text:
            facts.append(f"覆盖分类：{counts_text}")

    gate = coverage.get("gate") if isinstance(coverage.get("gate"), Mapping) else {}
    gate_status = str(gate.get("status") or "")
    review_status = str(
        gate.get("review_status") or coverage.get("review_status") or "unknown"
    )
    if gate_status:
        review_line = f"独立核对：{coverage_review_status_label(review_status)}"
        policy = str(gate.get("review_policy") or coverage.get("review_policy") or "")
        if policy:
            review_line += f"；策略：{coverage_review_policy_label(policy)}"
        facts.append(review_line)
        if gate.get("confirmed") is True:
            facts.append(f"覆盖核对门禁：{coverage_gate_status_label('confirmed')}")
        else:
            reasons = _gate_reason_labels(gate)
            detail = "、".join(reasons) if reasons else "尚未满足"
            facts.append(
                f"覆盖核对门禁：{coverage_gate_status_label(gate_status)}（{detail}）"
            )

    unresolved = int(
        coverage.get("unresolved_candidate_count")
        or coverage.get("unresolved_findings")
        or 0
    )
    if unresolved > 0:
        facts.append(f"未解决候选：{unresolved} 条（定位见「更多详情」）")
    return facts


def format_coverage_detail_facts(coverage: Mapping[str, Any] | None) -> list[str]:
    """Return detail lines (paths/locators) for the collapsible details area."""
    if not isinstance(coverage, Mapping) or not coverage:
        return []
    details: list[str] = []
    review_path = str(coverage.get("review_path") or "")
    if review_path:
        details.append(f"核对文件：{review_path}")
    coverage_digest = str(coverage.get("coverage_digest") or "")
    if coverage_digest:
        details.append(f"coverage digest：{coverage_digest}")
    for candidate in coverage.get("unresolved_candidates") or ():
        if not isinstance(candidate, Mapping):
            continue
        classification = str(candidate.get("classification") or "unknown")
        location = _locator_text(candidate.get("locator"))
        reasons = ",".join(str(item) for item in candidate.get("reason_codes") or ())
        line = f"未解决候选：{location} [{classification}]"
        if reasons:
            line += f" {reasons}"
        details.append(line)
    return details
