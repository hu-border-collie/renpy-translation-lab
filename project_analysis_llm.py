"""Project Analysis LLM map-reduce (#254 PR B).

Refines draft label / route / project brief summaries with an injectable
``SyncModelBackend``. Structure (routes/labels) still comes from
``project_analysis_routes`` + ``project_analysis_generate.build_structure_drafts``.

Does not write glossary.json, story_graph.json, or .rpy files.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Protocol, Sequence

from project_analysis import (
    KIND_LABEL,
    KIND_PROJECT_BRIEF,
    KIND_ROUTE,
    STATUS_DRAFT,
    STATUS_FAILED,
    STATUS_PUBLISHED,
    STATUS_STALE,
    ProjectAnalysisError,
    digest_upstream_artifacts,
    empty_lineage,
    evaluate_record_status,
    normalize_lineage,
    normalize_summary_record,
    resolve_project_analysis_store,
    sha256_text,
    utc_now_iso,
)
from sync_model_backend import SyncGenerationRequest, SyncModelBackend, SyncGenerationResult

PROMPT_SCHEMA_VERSION = "project-analysis-llm-v1"

GenerateFn = Callable[[SyncGenerationRequest], SyncGenerationResult]


class AnalysisLlmConfig(Protocol):
    model: str
    thinking_level: str
    max_label_summary_chars: int
    max_route_summary_chars: int
    max_brief_chars: int
    max_input_chars_per_request: int
    max_output_tokens: int


def default_llm_config(
    *,
    model: str = "",
    thinking_level: str = "",
    max_label_summary_chars: int = 800,
    max_route_summary_chars: int = 1200,
    max_brief_chars: int = 4000,
    max_input_chars_per_request: int = 12000,
    max_output_tokens: int = 2048,
) -> dict[str, Any]:
    return {
        "model": str(model or "").strip(),
        "thinking_level": str(thinking_level or "").strip(),
        "max_label_summary_chars": int(max_label_summary_chars),
        "max_route_summary_chars": int(max_route_summary_chars),
        "max_brief_chars": int(max_brief_chars),
        "max_input_chars_per_request": int(max_input_chars_per_request),
        "max_output_tokens": int(max_output_tokens),
    }


def merge_llm_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(config or {})
    return default_llm_config(
        model=str(raw.get("model") or ""),
        thinking_level=str(raw.get("thinking_level") or ""),
        max_label_summary_chars=int(raw.get("max_label_summary_chars") or 800),
        max_route_summary_chars=int(raw.get("max_route_summary_chars") or 1200),
        max_brief_chars=int(raw.get("max_brief_chars") or 4000),
        max_input_chars_per_request=int(raw.get("max_input_chars_per_request") or 12000),
        max_output_tokens=int(raw.get("max_output_tokens") or 2048),
    )


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _build_request(
    *,
    model: str,
    system: str,
    user: str,
    thinking_level: str = "",
    max_output_tokens: int = 2048,
) -> SyncGenerationRequest:
    config: dict[str, Any] = {
        "system_instruction": system,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.2,
    }
    if thinking_level:
        config["thinking_config"] = {"thinking_level": thinking_level}
    return SyncGenerationRequest(
        model=model,
        contents=[{"role": "user", "parts": [{"text": user}]}],
        config=config,
    )


def complete_analysis_text(
    generate: GenerateFn,
    *,
    model: str,
    system: str,
    user: str,
    thinking_level: str = "",
    max_output_tokens: int = 2048,
) -> tuple[str, Mapping[str, Any]]:
    """Call backend; return (text, usage_metadata)."""
    if not model:
        raise ProjectAnalysisError("analysis LLM model is not configured")
    result = generate(
        _build_request(
            model=model,
            system=system,
            user=user,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
        )
    )
    text = str(getattr(result, "response_text", "") or "").strip()
    if not text:
        raise ProjectAnalysisError("analysis LLM returned empty text")
    usage = dict(getattr(result, "usage_metadata", None) or {})
    return text, usage


def _label_system() -> str:
    return (
        "你是 Ren'Py 视觉小说项目分析助手。根据给定的 label 结构与证据摘要，"
        "写一段简短中文 label 摘要。规则：\n"
        "1) 只使用提供的材料，不要编造未出现的情节、人物关系或结局；\n"
        "2) 保留分支/未解析跳转的不确定性，明确写 unresolved 时不要猜目标；\n"
        "3) 不要输出 JSON，不要写术语表或修改建议；\n"
        "4) 控制长度，聚焦该 label 的功能与关键选择。"
    )


def _route_system() -> str:
    return (
        "你是 Ren'Py 分支路线分析助手。根据路线上各 label 摘要，写该路线的中文概述。\n"
        "规则：只依据材料；不虚构单一真结局；shared label 与 unresolved 要点名；"
        "不要输出 JSON；不要编造人物关系。"
    )


def _brief_system() -> str:
    return (
        "你是 Ren'Py 项目 brief 撰写助手。根据各路线摘要，写一份中文项目 brief 草稿，"
        "供人工审核后发布。规则：\n"
        "1) 只依据材料；\n"
        "2) 说明主要路线/分支与 unresolved 风险；\n"
        "3) 不要自动写入术语表或关系图；\n"
        "4) 使用 Markdown 小标题；不要声称已发布。"
    )


def refine_label_record(
    record: Mapping[str, Any],
    *,
    generate: GenerateFn,
    config: Mapping[str, Any],
    source_fingerprint: str,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    rec = normalize_summary_record(record, default_kind=KIND_LABEL)
    cfg = merge_llm_config(config)
    model_name = model or cfg["model"]
    raw = rec.get("summary") or ""
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    user = (
        f"Label id: {rec.get('label_id') or rec.get('id')}\n"
        f"Source files: {', '.join(rec.get('source_files') or [])}\n"
        f"Evidence item ids: {', '.join(rec.get('evidence_item_ids') or [])}\n"
        f"Unresolved outgoing: {meta.get('unresolved_outgoing')}\n"
        f"Outgoing targets: {meta.get('outgoing_targets')}\n\n"
        f"Current draft notes (may be mechanical):\n{_clip(str(raw), cfg['max_input_chars_per_request'])}\n"
    )
    text, _usage = complete_analysis_text(
        generate,
        model=model_name,
        system=_label_system(),
        user=user,
        thinking_level=cfg["thinking_level"],
        max_output_tokens=cfg["max_output_tokens"],
    )
    text = _clip(text, cfg["max_label_summary_chars"])
    out = dict(rec)
    out["summary"] = text
    out["status"] = STATUS_DRAFT
    out["source_checksum"] = sha256_text(text)
    out["lineage"] = empty_lineage(
        source_fingerprint=source_fingerprint
        or (normalize_lineage(rec.get("lineage")).get("source_fingerprint") or ""),
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        provider=provider,
        model=model_name,
        thinking_level=cfg["thinking_level"],
        upstream_dependency_digest=digest_upstream_artifacts(
            rec.get("upstream_artifact_ids") or []
        ),
        generated_at=utc_now_iso(),
    )
    return normalize_summary_record(out, default_kind=KIND_LABEL)


def refine_route_record(
    record: Mapping[str, Any],
    label_by_id: Mapping[str, Mapping[str, Any]],
    *,
    generate: GenerateFn,
    config: Mapping[str, Any],
    source_fingerprint: str,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    rec = normalize_summary_record(record, default_kind=KIND_ROUTE)
    cfg = merge_llm_config(config)
    model_name = model or cfg["model"]
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    parts: list[str] = []
    for lid in meta.get("label_ids") or []:
        label = label_by_id.get(f"label:{lid}") or label_by_id.get(str(lid))
        if label:
            parts.append(f"### {lid}\n{label.get('summary') or ''}")
        else:
            parts.append(f"### {lid}\n(missing label summary)")
    body = "\n\n".join(parts)
    user = (
        f"Route id: {rec.get('id')}\n"
        f"Entry: {meta.get('entry_label')}\n"
        f"Unresolved: {meta.get('unresolved')}\n"
        f"Shared labels: {meta.get('shared_labels')}\n"
        f"Evidence item ids: {', '.join(rec.get('evidence_item_ids') or [])}\n\n"
        f"Label summaries:\n{_clip(body, cfg['max_input_chars_per_request'])}\n"
    )
    text, _usage = complete_analysis_text(
        generate,
        model=model_name,
        system=_route_system(),
        user=user,
        thinking_level=cfg["thinking_level"],
        max_output_tokens=cfg["max_output_tokens"],
    )
    text = _clip(text, cfg["max_route_summary_chars"])
    out = dict(rec)
    out["summary"] = text
    out["status"] = STATUS_DRAFT
    out["source_checksum"] = sha256_text(text)
    out["lineage"] = empty_lineage(
        source_fingerprint=source_fingerprint
        or (normalize_lineage(rec.get("lineage")).get("source_fingerprint") or ""),
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        provider=provider,
        model=model_name,
        thinking_level=cfg["thinking_level"],
        upstream_dependency_digest=digest_upstream_artifacts(
            rec.get("upstream_artifact_ids") or []
        ),
        generated_at=utc_now_iso(),
    )
    return normalize_summary_record(out, default_kind=KIND_ROUTE)


def refine_project_brief(
    routes: Sequence[Mapping[str, Any]],
    *,
    generate: GenerateFn,
    config: Mapping[str, Any],
    source_fingerprint: str,
    provider: str = "",
    model: str = "",
    unresolved_count: int = 0,
) -> str:
    cfg = merge_llm_config(config)
    model_name = model or cfg["model"]
    parts = []
    for route in routes:
        rid = route.get("id") or ""
        meta = route.get("metadata") if isinstance(route.get("metadata"), dict) else {}
        parts.append(
            f"## {rid}\nentry={meta.get('entry_label')} unresolved={meta.get('unresolved')}\n"
            f"{route.get('summary') or ''}"
        )
    user = (
        f"Unresolved dynamic edges (count): {unresolved_count}\n"
        f"Route count: {len(routes)}\n\n"
        f"{_clip(chr(10).join(parts), cfg['max_input_chars_per_request'])}\n"
    )
    text, _usage = complete_analysis_text(
        generate,
        model=model_name,
        system=_brief_system(),
        user=user,
        thinking_level=cfg["thinking_level"],
        max_output_tokens=cfg["max_output_tokens"],
    )
    return _clip(text, cfg["max_brief_chars"])


def generation_signature(
    *,
    source_fingerprint: str = "",
    provider: str = "",
    model: str = "",
    thinking_level: str = "",
    upstream_dependency_digest: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
) -> dict[str, str]:
    """Full generation signature used for cache/skip decisions."""
    return {
        "prompt_schema_version": str(prompt_schema_version or ""),
        "source_fingerprint": str(source_fingerprint or ""),
        "provider": str(provider or ""),
        "model": str(model or ""),
        "thinking_level": str(thinking_level or ""),
        "upstream_dependency_digest": str(upstream_dependency_digest or ""),
    }


def lineage_matches_generation_signature(
    lineage: Mapping[str, Any],
    expected: Mapping[str, str],
) -> bool:
    """Return True when stored lineage matches the full generation signature."""
    current = normalize_lineage(lineage)
    for key, value in expected.items():
        if not value:
            # Empty expected field is not a wildcard for model/provider/thinking:
            # only source_fingerprint may be empty when caller cannot compute it.
            if key == "source_fingerprint":
                continue
            if not str(current.get(key) or ""):
                continue
            return False
        if str(current.get(key) or "") != value:
            return False
    # Model must be non-empty for a cached LLM result.
    if not str(current.get("model") or ""):
        return False
    if str(current.get("prompt_schema_version") or "") != str(
        expected.get("prompt_schema_version") or PROMPT_SCHEMA_VERSION
    ):
        return False
    return True


def _needs_llm_refresh(
    record: Mapping[str, Any],
    *,
    expected_signature: Mapping[str, str],
    force: bool,
) -> bool:
    if force:
        return True
    status = str(record.get("status") or STATUS_DRAFT)
    lineage = normalize_lineage(record.get("lineage"))
    # Recompute expected upstream from the record when not provided by caller.
    expected = dict(expected_signature)
    if not expected.get("upstream_dependency_digest"):
        expected["upstream_dependency_digest"] = digest_upstream_artifacts(
            record.get("upstream_artifact_ids") or []
        )
    if status in {STATUS_STALE, STATUS_FAILED}:
        return True
    if status == STATUS_DRAFT:
        return not lineage_matches_generation_signature(lineage, expected)
    if status == STATUS_PUBLISHED:
        effective = evaluate_record_status(
            record,
            expected_source_fingerprint=expected.get("source_fingerprint") or "",
            expected_upstream_digest=expected.get("upstream_dependency_digest") or "",
            expected_prompt_schema_version=expected.get("prompt_schema_version") or "",
            expected_provider=expected.get("provider") or "",
            expected_model=expected.get("model") or "",
            expected_thinking_level=expected.get("thinking_level") or "",
        )
        return effective != STATUS_PUBLISHED
    return True


def run_mapreduce_drafts(
    *,
    store_dir: str | None = None,
    base_dir: str | None = None,
    generate: GenerateFn | None = None,
    backend: SyncModelBackend | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    """LLM-refine label → route → brief drafts in the analysis store.

    Requires structure drafts already present (run build-structure first).
    """
    if generate is None:
        if backend is None:
            raise ProjectAnalysisError("run_mapreduce_drafts requires generate= or backend=")

        def generate(request: SyncGenerationRequest) -> SyncGenerationResult:
            return backend.generate(request)

    cfg = merge_llm_config(config)
    if model:
        cfg["model"] = model
    if not cfg["model"]:
        raise ProjectAnalysisError(
            "analysis model is empty; set batch.project_analysis.model or --model"
        )

    store = resolve_project_analysis_store(store_dir, base_dir=base_dir)
    labels = store.load_summaries(KIND_LABEL)
    routes = store.load_routes()
    if not labels and not routes:
        raise ProjectAnalysisError(
            "no label/route drafts found; run project-analysis-build-structure first"
        )

    # Prefer structure fingerprint from existing records / manifest.
    source_fp = ""
    if labels:
        source_fp = str(
            (normalize_lineage(labels[0].get("lineage")).get("source_fingerprint") or "")
        )
    if not source_fp and routes:
        source_fp = str(
            (normalize_lineage(routes[0].get("lineage")).get("source_fingerprint") or "")
        )
    if not source_fp:
        manifest = store.load_manifest() or {}
        brief = (manifest.get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {}
        source_fp = str(
            (normalize_lineage(brief.get("lineage")).get("source_fingerprint") or "")
        )

    prov = provider or getattr(backend, "provider", "") or ""
    model_name = cfg["model"]

    base_sig = generation_signature(
        source_fingerprint=source_fp,
        provider=prov,
        model=model_name,
        thinking_level=cfg["thinking_level"],
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
    )

    labels_out: list[dict[str, Any]] = []
    labels_refined = 0
    labels_skipped = 0
    for rec in labels:
        if not _needs_llm_refresh(rec, expected_signature=base_sig, force=force):
            labels_out.append(normalize_summary_record(rec, default_kind=KIND_LABEL))
            labels_skipped += 1
            continue
        labels_out.append(
            refine_label_record(
                rec,
                generate=generate,
                config=cfg,
                source_fingerprint=source_fp,
                provider=prov,
                model=model_name,
            )
        )
        labels_refined += 1

    label_by_id = {r["id"]: r for r in labels_out}
    for r in labels_out:
        if r.get("label_id"):
            label_by_id[str(r["label_id"])] = r

    routes_out: list[dict[str, Any]] = []
    routes_refined = 0
    routes_skipped = 0
    for rec in routes:
        if not _needs_llm_refresh(rec, expected_signature=base_sig, force=force):
            routes_out.append(normalize_summary_record(rec, default_kind=KIND_ROUTE))
            routes_skipped += 1
            continue
        routes_out.append(
            refine_route_record(
                rec,
                label_by_id,
                generate=generate,
                config=cfg,
                source_fingerprint=source_fp,
                provider=prov,
                model=model_name,
            )
        )
        routes_refined += 1

    store.save_summaries(KIND_LABEL, labels_out)
    store.save_routes(routes_out)

    unresolved = sum(
        1
        for r in routes_out
        if (r.get("metadata") or {}).get("unresolved")
    )
    brief_text = refine_project_brief(
        routes_out,
        generate=generate,
        config=cfg,
        source_fingerprint=source_fp,
        provider=prov,
        model=model_name,
        unresolved_count=unresolved,
    )
    # Keep a short machine header so humans see LLM origin.
    if not brief_text.lstrip().startswith("#"):
        brief_text = "# Project Analysis Brief (LLM draft)\n\n" + brief_text
    store.save_brief_text(brief_text, published=False)

    identity = {}
    previous = store.load_manifest()
    if previous:
        identity = dict(previous.get("project_identity") or {})
    manifest = store.rebuild_manifest(
        project_identity=identity,
        expected_source_fingerprint=source_fp,
    )
    brief_entry = dict((manifest.get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {})
    upstream = [r["id"] for r in routes_out] + [r["id"] for r in labels_out]
    brief_entry.update(
        {
            "status": STATUS_DRAFT,
            "draft_present": True,
            "id": "project_brief",
            "lineage": empty_lineage(
                source_fingerprint=source_fp,
                prompt_schema_version=PROMPT_SCHEMA_VERSION,
                provider=prov,
                model=model_name,
                thinking_level=cfg["thinking_level"],
                upstream_dependency_digest=digest_upstream_artifacts(upstream),
                generated_at=utc_now_iso(),
            ),
        }
    )
    if identity:
        manifest["project_identity"] = identity
    manifest.setdefault("artifacts", {})[KIND_PROJECT_BRIEF] = brief_entry
    store.save_manifest(manifest)

    return {
        "store_dir": store.store_dir,
        "source_fingerprint": source_fp,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "model": model_name,
        "provider": prov,
        "labels_refined": labels_refined,
        "labels_skipped": labels_skipped,
        "routes_refined": routes_refined,
        "routes_skipped": routes_skipped,
        "brief_draft_chars": len(brief_text),
        "force": bool(force),
    }
