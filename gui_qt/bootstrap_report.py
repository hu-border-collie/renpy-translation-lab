"""User-facing summaries for GUI bootstrap-rag / bootstrap-source-index commands."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .duration_format import format_remaining_duration_zh
from .summary_helpers import extend_facts_with_notices
from .user_copy import format_bootstrap_fact
from typing import Any


@dataclass(frozen=True)
class BootstrapSummary:
    kind: str
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]


@dataclass(frozen=True)
class BootstrapProgressState:
    kind: str = "source_index"
    phase: str = "starting"
    total_segments: int = 0
    stored_segments: int = 0
    embedding_total: int = 0
    embedding_done: int = 0
    reused_embeddings: int = 0


_SOURCE_INDEX_PRE_RUN_TOTAL_RE = re.compile(
    r"- Total segments scanned from files:\s*(\d+)"
)
_SOURCE_INDEX_PRE_RUN_PENDING_RE = re.compile(
    r"- New/updated segments \(need embeddings\):\s*(\d+)"
)
_SOURCE_INDEX_PRE_RUN_REUSED_RE = re.compile(
    r"- Unchanged segments \(reusing embeddings\):\s*(\d+)"
)
_SOURCE_INDEX_REUSED_WRITTEN_RE = re.compile(
    r"Reused embeddings written:\s*(\d+)"
)
_SOURCE_INDEX_EMBEDDING_TOTAL_RE = re.compile(
    r"Generating embeddings for\s*(\d+)\s*segments"
)
_SOURCE_INDEX_EMBEDDING_PROGRESS_RE = re.compile(
    r"Source index embedding progress:\s*(\d+)/(\d+)\s*scanned,\s*(\d+)\s*embedded,\s*(\d+)\s*stored"
)


RAG_SUMMARY_HEADER = "RAG bootstrap summary:"
SOURCE_INDEX_SUMMARY_HEADER = "Source Index bootstrap final summary:"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def read_batch_context_flags(config: dict[str, Any]) -> dict[str, bool]:
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
    rag = batch.get("rag")
    if not isinstance(rag, dict):
        rag = {}
    source_index = batch.get("source_index")
    if not isinstance(source_index, dict):
        source_index = {}
    return {
        "rag_enabled": coerce_bool(rag.get("enabled"), False),
        "source_index_enabled": coerce_bool(source_index.get("enabled"), False),
        "bootstrap_on_build": coerce_bool(rag.get("bootstrap_on_build"), True),
    }


def _parse_summary_values(output: str, header: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == header:
            in_section = True
            continue
        if in_section and line.startswith("- "):
            match = re.match(r"-\s*([a-z_]+):\s*(.*)$", line)
            if match:
                values[match.group(1)] = match.group(2).strip()
            continue
        if in_section and not line.startswith("- "):
            break
    return values


def _parse_int_field(values: dict[str, str], key: str, default: int = 0) -> int:
    raw = values.get(key, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _format_fact_values(values: dict[str, str], keys: tuple[str, ...]) -> list[str]:
    facts: list[str] = []
    for key in keys:
        if key in values:
            facts.append(format_bootstrap_fact(key, values[key]))
    return facts


def idle_bootstrap_summary() -> BootstrapSummary:
    return BootstrapSummary(
        kind="",
        status="idle",
        heading="尚未预建上下文库",
        message="如果项目已有部分译文，可先预建记忆库；如果译文很少，可预建原文索引。",
        facts=[],
        findings=[],
    )


def create_bootstrap_progress_state(kind: str) -> BootstrapProgressState:
    return BootstrapProgressState(kind=kind or "source_index")


def update_bootstrap_progress_from_line(
    line: str,
    state: BootstrapProgressState,
) -> BootstrapProgressState:
    if state.kind != "source_index":
        return state

    text = line.strip()
    if not text:
        return state

    match = _SOURCE_INDEX_PRE_RUN_TOTAL_RE.search(text)
    if match:
        return BootstrapProgressState(
            kind=state.kind,
            phase=state.phase,
            total_segments=int(match.group(1)),
            stored_segments=state.stored_segments,
            embedding_total=state.embedding_total,
            embedding_done=state.embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    match = _SOURCE_INDEX_PRE_RUN_PENDING_RE.search(text)
    if match:
        return BootstrapProgressState(
            kind=state.kind,
            phase=state.phase,
            total_segments=state.total_segments,
            stored_segments=state.stored_segments,
            embedding_total=int(match.group(1)),
            embedding_done=state.embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    match = _SOURCE_INDEX_PRE_RUN_REUSED_RE.search(text)
    if match:
        return BootstrapProgressState(
            kind=state.kind,
            phase="reusing",
            total_segments=state.total_segments,
            stored_segments=state.stored_segments,
            embedding_total=state.embedding_total,
            embedding_done=state.embedding_done,
            reused_embeddings=int(match.group(1)),
        )

    match = _SOURCE_INDEX_REUSED_WRITTEN_RE.search(text)
    if match:
        stored = int(match.group(1))
        return BootstrapProgressState(
            kind=state.kind,
            phase="embedding",
            total_segments=state.total_segments,
            stored_segments=stored,
            embedding_total=state.embedding_total,
            embedding_done=state.embedding_done,
            reused_embeddings=max(state.reused_embeddings, stored),
        )

    match = _SOURCE_INDEX_EMBEDDING_TOTAL_RE.search(text)
    if match:
        return BootstrapProgressState(
            kind=state.kind,
            phase="embedding",
            total_segments=state.total_segments,
            stored_segments=state.stored_segments,
            embedding_total=int(match.group(1)),
            embedding_done=state.embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    match = _SOURCE_INDEX_EMBEDDING_PROGRESS_RE.search(text)
    if match:
        embedding_done = int(match.group(3))
        embedding_total = int(match.group(2))
        stored = int(match.group(4))
        return BootstrapProgressState(
            kind=state.kind,
            phase="embedding",
            total_segments=state.total_segments,
            stored_segments=stored,
            embedding_total=embedding_total,
            embedding_done=embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    if text.startswith("Pruning ") and "stale segments" in text:
        return BootstrapProgressState(
            kind=state.kind,
            phase="pruning",
            total_segments=state.total_segments,
            stored_segments=state.stored_segments,
            embedding_total=state.embedding_total,
            embedding_done=state.embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    if text.startswith("Sync complete."):
        total = state.total_segments
        stored = state.stored_segments
        if total > 0 and stored < total:
            stored = total
        return BootstrapProgressState(
            kind=state.kind,
            phase="done",
            total_segments=total,
            stored_segments=stored,
            embedding_total=state.embedding_total,
            embedding_done=state.embedding_done,
            reused_embeddings=state.reused_embeddings,
        )

    return state


_BOOTSTRAP_ETA_MIN_ELAPSED_SECONDS = 3.0
_BOOTSTRAP_ETA_MIN_PROGRESS_SEGMENTS = 16


@dataclass
class BootstrapProgressTracker:
    started_at: float | None = None
    last_stored_segments: int = 0
    last_sample_at: float | None = None
    seconds_per_segment: float | None = None

    def reset(self) -> None:
        self.started_at = None
        self.last_stored_segments = 0
        self.last_sample_at = None
        self.seconds_per_segment = None

    def observe(
        self,
        state: BootstrapProgressState,
        now: float | None = None,
    ) -> None:
        if state.kind != "source_index" or state.total_segments <= 0:
            return
        if now is None:
            now = time.monotonic()

        stored = max(state.stored_segments, 0)
        if self.started_at is None:
            if stored <= 0:
                return
            self.started_at = now
            self.last_stored_segments = stored
            self.last_sample_at = now
            return

        if stored <= self.last_stored_segments:
            return

        delta_stored = stored - self.last_stored_segments
        last_sample_at = self.last_sample_at
        delta_time = now - (last_sample_at if last_sample_at is not None else now)
        if delta_stored <= 0 or delta_time <= 0:
            return

        sample_rate = delta_time / delta_stored
        if self.seconds_per_segment is None:
            self.seconds_per_segment = sample_rate
        else:
            self.seconds_per_segment = (
                self.seconds_per_segment * 0.6 + sample_rate * 0.4
            )
        self.last_stored_segments = stored
        self.last_sample_at = now

    def estimate_remaining_seconds(
        self,
        state: BootstrapProgressState,
        now: float | None = None,
    ) -> int | None:
        if state.kind != "source_index" or state.total_segments <= 0:
            return None
        if now is None:
            now = time.monotonic()

        stored = min(max(state.stored_segments, 0), state.total_segments)
        remaining_segments = state.total_segments - stored
        if remaining_segments <= 0:
            return 0

        used_fallback_rate = False
        seconds_per_segment = self.seconds_per_segment
        if seconds_per_segment is None and self.started_at is not None:
            elapsed = now - self.started_at
            if (
                stored >= _BOOTSTRAP_ETA_MIN_PROGRESS_SEGMENTS
                and elapsed >= _BOOTSTRAP_ETA_MIN_ELAPSED_SECONDS
            ):
                seconds_per_segment = elapsed / stored
                used_fallback_rate = True

        if seconds_per_segment is None:
            return None

        estimate = remaining_segments * seconds_per_segment
        if not used_fallback_rate and self.last_sample_at is not None:
            estimate -= now - self.last_sample_at
        return max(0, int(estimate + 0.5))


def create_bootstrap_progress_tracker() -> BootstrapProgressTracker:
    return BootstrapProgressTracker()


def format_bootstrap_progress_bar_label(
    state: BootstrapProgressState,
    remaining_seconds: int | None = None,
) -> str:
    if state.total_segments <= 0 and state.stored_segments <= 0:
        return "正在扫描原文…"

    total = max(state.total_segments, 1)
    stored = min(max(state.stored_segments, 0), total)
    percent = (stored * 100) // total
    label = f"{percent}%（{stored}/{total}）"
    if (
        remaining_seconds is not None
        and remaining_seconds > 0
        and stored < total
    ):
        label += f" · {format_remaining_duration_zh(remaining_seconds)}"
    return label


def format_bootstrap_progress_facts(state: BootstrapProgressState) -> list[str]:
    if state.kind != "source_index":
        return []

    facts: list[str] = []
    if state.total_segments > 0:
        percent = (state.stored_segments * 100) // state.total_segments
        facts.append(
            f"入库进度：{state.stored_segments}/{state.total_segments} 片段（{percent}%）"
        )
    if state.embedding_total > 0:
        facts.append(
            f"本轮向量生成：{state.embedding_done}/{state.embedding_total}"
        )
    elif state.reused_embeddings > 0 and state.phase in {"reusing", "embedding", "starting"}:
        facts.append(f"已复用向量：{state.reused_embeddings} 片段")

    if state.phase == "pruning":
        facts.append("正在清理失效片段…")
    return facts


def running_bootstrap_summary(kind: str) -> BootstrapSummary:
    if kind == "source_index":
        return BootstrapSummary(
            kind=kind,
            status="running",
            heading="正在预建原文索引",
            message="正在扫描翻译模板原文并生成向量索引，请稍候。",
            facts=[],
            findings=[],
        )
    return BootstrapSummary(
        kind=kind or "rag",
        status="running",
        heading="正在预建记忆库",
        message="正在扫描已有译文并更新记忆库，请稍候。",
        facts=[],
        findings=[],
    )


def stale_bootstrap_summary() -> BootstrapSummary:
    return BootstrapSummary(
        kind="",
        status="stale",
        heading="预建状态已过期",
        message="项目或配置已切换，请针对当前项目重新运行预建库。",
        facts=[],
        findings=[],
    )


def summarize_rag_bootstrap_output(output: str, exit_code: int) -> BootstrapSummary:
    if "RAG is disabled" in output:
        return BootstrapSummary(
            kind="rag",
            status="warning",
            heading="记忆库未启用",
            message="请先在配置页启用记忆库并保存配置，再运行预建记忆库。",
            facts=[],
            findings=[],
        )

    values = _parse_summary_values(output, RAG_SUMMARY_HEADER)
    facts = _format_fact_values(
        values,
        (
            "store_dir",
            "scan_scope",
            "files_scanned",
            "scanned",
            "embedded",
            "upserted",
            "history_records_before",
            "history_records_after",
            "external_seed_records",
        ),
    )
    findings: list[str] = []
    if values.get("error"):
        findings.append(values["error"])

    if exit_code != 0:
        return BootstrapSummary(
            kind="rag",
            status="failed",
            heading="预建记忆库失败",
            message="预建记忆库未成功完成，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if values.get("error"):
        return BootstrapSummary(
            kind="rag",
            status="failed",
            heading="预建记忆库失败",
            message="预建过程中出现错误，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    scanned = _parse_int_field(values, "scanned")
    upserted = _parse_int_field(values, "upserted")
    embedded = _parse_int_field(values, "embedded")
    external_seed_records = _parse_int_field(values, "external_seed_records")
    if scanned == 0 and external_seed_records == 0:
        return BootstrapSummary(
            kind="rag",
            status="warning",
            heading="预建记忆库完成（无新记录）",
            message="未扫描到可写入的译文记录。若项目尚无译文，可先翻译一部分。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if upserted > 0 or embedded > 0 or external_seed_records > 0:
        return BootstrapSummary(
            kind="rag",
            status="ready",
            heading="预建记忆库完成",
            message="记忆库已刷新，可以开始翻译任务。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    return BootstrapSummary(
        kind="rag",
        status="ready",
        heading="预建记忆库完成",
        message="预建流程已完成；如需更新记忆库，可在译文变更后再次运行。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
    )


def summarize_source_index_bootstrap_output(output: str, exit_code: int) -> BootstrapSummary:
    values = _parse_summary_values(output, SOURCE_INDEX_SUMMARY_HEADER)
    facts = _format_fact_values(
        values,
        (
            "store_dir",
            "files_scanned",
            "scanned",
            "embedded",
            "upserted",
            "reused_embeddings",
            "stale_count",
            "pruned",
            "history_records_before",
            "history_records_after",
        ),
    )
    findings: list[str] = []
    if values.get("error"):
        findings.append(values["error"])

    if exit_code != 0:
        return BootstrapSummary(
            kind="source_index",
            status="failed",
            heading="原文索引预建失败",
            message="预建原文索引未成功完成，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if values.get("error"):
        return BootstrapSummary(
            kind="source_index",
            status="failed",
            heading="原文索引预建失败",
            message="预建过程中出现错误，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    scanned = _parse_int_field(values, "scanned")
    upserted = _parse_int_field(values, "upserted")
    embedded = _parse_int_field(values, "embedded")
    if scanned == 0:
        return BootstrapSummary(
            kind="source_index",
            status="warning",
            heading="原文索引预建完成（无新记录）",
            message="未扫描到可索引的原文片段。请确认翻译模板已生成且项目路径正确。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if upserted > 0 or embedded > 0:
        return BootstrapSummary(
            kind="source_index",
            status="ready",
            heading="原文索引预建完成",
            message="原文索引已刷新，后续翻译时可检索相关剧情原文。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    return BootstrapSummary(
        kind="source_index",
        status="ready",
        heading="原文索引预建完成",
        message="索引库已是最新状态，无需新增向量。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
    )