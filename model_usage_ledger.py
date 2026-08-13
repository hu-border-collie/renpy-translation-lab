"""Project-scoped, provider-neutral model usage ledger.

The ledger stores usage summaries only. Response text is used to derive a
stable deduplication fingerprint but is never persisted in the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import atomic_io
import batch_cost_estimate

SCHEMA_VERSION = 1
USAGE_DIRECTORY_NAME = "translation_usage"
USAGE_LEDGER_FILENAME = "usage_ledger.json"

TASK_MODE_ALIASES = {
    "translation": "translation",
    "revision": "revision",
    "revisions": "revision",
    "keyword": "keyword",
    "keywords": "keyword",
    "keyword_extraction": "keyword",
    "repair": "repair",
    "analysis": "analysis",
    "final_review": "analysis",
    "project_analysis": "analysis",
    "probe": "analysis",
    "compare_variants": "analysis",
}

GROUP_FIELD_ALIASES = {
    "task": "task_mode",
    "task_mode": "task_mode",
    "stage": "stage",
    "provider": "provider",
    "model": "model",
    "run": "run_id",
    "run_id": "run_id",
    "operation": "operation_id",
    "operation_id": "operation_id",
    "execution": "execution_mode",
    "execution_mode": "execution_mode",
}

TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "thoughts_tokens",
    "cached_tokens",
)


class UsageLedgerError(RuntimeError):
    """Raised when a usage ledger cannot be read, validated, or written."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_game_root(game_root: str | os.PathLike[str] | None) -> str:
    if not game_root:
        raise UsageLedgerError("game_root is required for the model usage ledger")
    raw = os.fspath(game_root)
    try:
        return str(Path(raw).expanduser().resolve(strict=False))
    except OSError:
        return os.path.abspath(raw)


def project_identity(game_root: str | os.PathLike[str] | None) -> dict[str, str]:
    canonical = canonical_game_root(game_root)
    normalized = os.path.normcase(canonical)
    return {
        "game_root": canonical,
        "project_id": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def usage_ledger_path(game_root: str | os.PathLike[str] | None) -> str:
    root = canonical_game_root(game_root)
    return os.path.join(root, USAGE_DIRECTORY_NAME, USAGE_LEDGER_FILENAME)


def new_run_id(prefix: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "-" for char in prefix)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{clean or 'usage'}-{stamp}-{uuid.uuid4().hex[:12]}"


def normalize_task_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return TASK_MODE_ALIASES.get(normalized, normalized or "analysis")


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _json_safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    normalized = _json_safe(dict(value))
    return normalized if isinstance(normalized, dict) else {}


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_provider_usage(response_payload: Any) -> dict[str, Any]:
    """Extract a provider usage mapping without assuming Gemini or OpenAI casing."""
    payload = response_payload if isinstance(response_payload, Mapping) else {}
    nested = payload.get("response")
    if isinstance(nested, Mapping):
        payload = nested
    for key in ("usageMetadata", "usage_metadata", "usage"):
        usage = payload.get(key)
        if isinstance(usage, Mapping):
            return _json_safe_mapping(usage)
    return {}


def extract_response_id(response_payload: Any) -> str:
    payload = response_payload if isinstance(response_payload, Mapping) else {}
    candidates = [payload]
    nested = payload.get("response")
    if isinstance(nested, Mapping):
        candidates.insert(0, nested)
    for candidate in candidates:
        for key in ("responseId", "response_id", "id", "request_id"):
            value = candidate.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
    return ""


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _first_int(mapping: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> int | None:
    for path in paths:
        parsed = _nonnegative_int(_nested_value(mapping, path))
        if parsed is not None:
            return parsed
    return None


def normalize_usage_metadata(usage_metadata: Mapping[str, Any] | None) -> dict[str, int | bool | None]:
    """Normalize Gemini, LiteLLM/OpenAI, and generic token counters.

    Missing values remain ``None``. ``total_tokens`` is derived only when the
    available component counters make that derivation unambiguous.
    """
    usage = dict(usage_metadata or {})
    prompt = _first_int(
        usage,
        (
            ("promptTokenCount",),
            ("prompt_token_count",),
            ("prompt_tokens",),
            ("input_tokens",),
            ("inputTokenCount",),
        ),
    )
    completion = _first_int(
        usage,
        (
            ("candidatesTokenCount",),
            ("candidates_token_count",),
            ("completion_tokens",),
            ("output_tokens",),
            ("outputTokenCount",),
        ),
    )
    thoughts = _first_int(
        usage,
        (
            ("thoughtsTokenCount",),
            ("thoughts_token_count",),
            ("thoughts_tokens",),
            ("reasoning_tokens",),
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ),
    )
    cached = _first_int(
        usage,
        (
            ("cachedContentTokenCount",),
            ("cached_content_token_count",),
            ("cached_tokens",),
            ("cache_read_input_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        ),
    )
    total = _first_int(
        usage,
        (
            ("totalTokenCount",),
            ("total_token_count",),
            ("total_tokens",),
        ),
    )
    total_derived = False
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
        if "candidatesTokenCount" in usage and thoughts is not None:
            total += thoughts
        total_derived = True

    billable_output = None
    if total is not None and prompt is not None:
        billable_output = max(0, total - prompt)
    elif completion is not None:
        billable_output = completion
        if "candidatesTokenCount" in usage and thoughts is not None:
            billable_output += thoughts

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "thoughts_tokens": thoughts,
        "cached_tokens": cached,
        "billable_output_tokens": billable_output,
        "total_tokens_derived": total_derived,
    }


def _nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def extract_actual_cost(
    usage_metadata: Mapping[str, Any] | None,
    response_payload: Any = None,
) -> tuple[float | None, str | None, str | None]:
    usage = dict(usage_metadata or {})
    for key in ("actual_cost", "total_cost", "response_cost", "cost"):
        parsed = _nonnegative_float(usage.get(key))
        if parsed is not None:
            currency = usage.get("cost_currency") or usage.get("currency")
            return parsed, str(currency or "").strip() or None, f"usage.{key}"

    payload = response_payload if isinstance(response_payload, Mapping) else {}
    hidden = payload.get("_hidden_params")
    if isinstance(hidden, Mapping):
        parsed = _nonnegative_float(hidden.get("response_cost"))
        if parsed is not None:
            currency = (
                hidden.get("response_cost_currency")
                or hidden.get("currency")
                or "USD"
            )
            return (
                parsed,
                str(currency or "").strip() or None,
                "_hidden_params.response_cost",
            )
    return None, None, None


def _dedupe_key(
    *,
    project_id: str,
    provider: str,
    model: str,
    response_payload: Any,
    response_id: str,
    run_id: str,
    source_key: str,
    usage_metadata: Mapping[str, Any],
) -> str:
    if response_id:
        identity = f"response:{provider}:{model}:{response_id}"
    elif response_payload:
        identity = (
            f"payload:{provider}:{model}:{source_key}:"
            f"{_canonical_json_digest(response_payload)}"
        )
    else:
        identity = (
            f"call:{provider}:{model}:{run_id}:{source_key}:"
            f"{_canonical_json_digest(usage_metadata)}"
        )
    return hashlib.sha256(f"{project_id}:{identity}".encode("utf-8")).hexdigest()


def build_usage_record(
    *,
    game_root: str | os.PathLike[str],
    task_mode: str,
    stage: str,
    provider: str,
    model: str,
    usage_metadata: Mapping[str, Any] | None,
    response_payload: Any = None,
    operation_id: str = "",
    run_id: str = "",
    manifest_id: str = "",
    thinking_level: str = "",
    execution_mode: str = "",
    source_key: str = "",
    source: Mapping[str, Any] | None = None,
    pricing_config: Mapping[str, Any] | None = None,
    estimated_cost: float | None = None,
    estimated_cost_currency: str = "",
    dedupe_key: str = "",
    recorded_at: str = "",
) -> dict[str, Any]:
    identity = project_identity(game_root)
    raw_usage = _json_safe_mapping(usage_metadata)
    normalized = normalize_usage_metadata(raw_usage)
    response_id = extract_response_id(response_payload)
    provider_name = str(provider or "").strip().lower() or "unknown"
    model_name = str(model or "").strip() or "unknown"
    run_identity = str(run_id or "").strip() or new_run_id("usage")
    key = str(dedupe_key or "").strip() or _dedupe_key(
        project_id=identity["project_id"],
        provider=provider_name,
        model=model_name,
        response_payload=response_payload,
        response_id=response_id,
        run_id=run_identity,
        source_key=str(source_key or ""),
        usage_metadata=raw_usage,
    )

    pricing = dict(pricing_config or {})
    computed_estimate = (
        _nonnegative_float(estimated_cost)
        if estimated_cost is not None
        else batch_cost_estimate.estimate_usage_cost(
            model_name,
            prompt_tokens=normalized["prompt_tokens"],
            output_tokens=normalized["billable_output_tokens"],
            pricing_config=pricing,
        )
    )
    estimate_basis = None
    if computed_estimate is not None:
        estimate_basis = (
            "caller_supplied" if estimated_cost is not None else "configured_pricing"
        )
    actual_cost, actual_currency, actual_source = extract_actual_cost(
        raw_usage, response_payload
    )
    estimate_currency = (
        str(estimated_cost_currency or pricing.get("currency") or "").strip() or None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "dedupe_key": key,
        "project": identity,
        "operation_id": str(operation_id or "").strip() or None,
        "run_id": run_identity,
        "manifest_id": str(manifest_id or "").strip() or None,
        "task_mode": normalize_task_mode(task_mode),
        "stage": str(stage or "").strip() or "unknown",
        "provider": provider_name,
        "model": model_name,
        "thinking_level": str(thinking_level or "").strip() or None,
        "execution_mode": str(execution_mode or "").strip() or None,
        "calls": 1,
        "prompt_tokens": normalized["prompt_tokens"],
        "completion_tokens": normalized["completion_tokens"],
        "total_tokens": normalized["total_tokens"],
        "thoughts_tokens": normalized["thoughts_tokens"],
        "cached_tokens": normalized["cached_tokens"],
        "total_tokens_derived": bool(normalized["total_tokens_derived"]),
        "provider_usage": raw_usage,
        "estimated_cost": computed_estimate,
        "estimated_cost_currency": estimate_currency if computed_estimate is not None else None,
        "estimated_cost_basis": estimate_basis,
        "actual_cost": actual_cost,
        "actual_cost_currency": actual_currency if actual_cost is not None else None,
        "actual_cost_source": actual_source,
        "response_id": response_id or None,
        "recorded_at": str(recorded_at or "").strip() or utc_now_iso(),
        "source": _json_safe_mapping(source),
    }


class UsageLedger:
    """Atomic JSON usage store bound to exactly one canonical game root."""

    def __init__(
        self,
        game_root: str | os.PathLike[str],
        *,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.project = project_identity(game_root)
        candidate_path = os.path.abspath(
            os.fspath(path) if path is not None else usage_ledger_path(game_root)
        )
        try:
            common_root = os.path.commonpath(
                [self.project["game_root"], candidate_path]
            )
        except ValueError as exc:
            raise UsageLedgerError(
                "Usage ledger path must stay inside the selected game_root"
            ) from exc
        if os.path.normcase(common_root) != os.path.normcase(self.project["game_root"]):
            raise UsageLedgerError(
                "Usage ledger path must stay inside the selected game_root"
            )
        self.path = candidate_path
        self.lock_path = f"{candidate_path}.lock"

    def _empty_payload(self) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "schema_version": SCHEMA_VERSION,
            "project": dict(self.project),
            "created_at": now,
            "updated_at": now,
            "records": [],
        }

    def load(self) -> dict[str, Any]:
        if not os.path.isfile(self.path):
            return self._empty_payload()
        try:
            with open(self.path, "r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UsageLedgerError(f"Could not read usage ledger {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise UsageLedgerError(f"Usage ledger root must be an object: {self.path}")
        try:
            schema_version = int(payload.get("schema_version") or 0)
        except (TypeError, ValueError) as exc:
            raise UsageLedgerError(
                f"Unsupported usage ledger schema_version in {self.path}: "
                f"{payload.get('schema_version')!r}"
            ) from exc
        if schema_version != SCHEMA_VERSION:
            raise UsageLedgerError(
                f"Unsupported usage ledger schema_version in {self.path}: "
                f"{payload.get('schema_version')!r}"
            )
        stored_project = payload.get("project")
        if not isinstance(stored_project, Mapping):
            raise UsageLedgerError(f"Usage ledger project identity is missing: {self.path}")
        if stored_project.get("project_id") != self.project["project_id"]:
            raise UsageLedgerError(
                "Usage ledger project identity does not match the selected game_root"
            )
        records = payload.get("records")
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise UsageLedgerError(f"Usage ledger records must be an array: {self.path}")
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        game_root = self.project["game_root"]
        if not os.path.isdir(game_root):
            raise UsageLedgerError(f"game_root does not exist: {game_root}")
        try:
            atomic_io.atomic_write_json(
                self.path,
                payload,
                ensure_ascii=False,
                indent=2,
            )
        except OSError as exc:
            raise UsageLedgerError(f"Could not write usage ledger {self.path}: {exc}") from exc

    def add_records(self, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        game_root = self.project["game_root"]
        if not os.path.isdir(game_root):
            raise UsageLedgerError(f"game_root does not exist: {game_root}")
        try:
            with atomic_io.exclusive_file_lock(self.lock_path):
                return self._add_records_unlocked(records)
        except OSError as exc:
            raise UsageLedgerError(
                f"Could not lock usage ledger {self.path}: {exc}"
            ) from exc

    def _add_records_unlocked(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload = self.load()
        stored = list(payload.get("records") or [])
        existing = {
            str(record.get("dedupe_key") or "")
            for record in stored
            if record.get("dedupe_key")
        }
        inserted = 0
        duplicates = 0
        for raw in records:
            record = _json_safe_mapping(raw)
            record_project = record.get("project")
            if not isinstance(record_project, Mapping):
                raise UsageLedgerError("Usage record project identity is missing")
            if record_project.get("project_id") != self.project["project_id"]:
                raise UsageLedgerError("Refusing to mix usage records from different projects")
            key = str(record.get("dedupe_key") or "").strip()
            if not key:
                raise UsageLedgerError("Usage record dedupe_key is missing")
            if key in existing:
                duplicates += 1
                continue
            existing.add(key)
            stored.append(record)
            inserted += 1
        if inserted:
            payload["records"] = stored
            payload["project"] = dict(self.project)
            payload["updated_at"] = utc_now_iso()
            self._write(payload)
        return {
            "ledger_path": self.path,
            "inserted_records": inserted,
            "duplicate_records": duplicates,
            "total_records": len(stored),
        }


def _manifest_task_and_stage(manifest: Mapping[str, Any]) -> tuple[str, str]:
    manifest_mode = str(manifest.get("mode") or "translation").strip().lower()
    task_mode = normalize_task_mode(manifest_mode)
    execution = str(
        manifest.get("execution_mode") or manifest.get("execution") or "batch"
    ).strip().lower()
    if manifest_mode == "final_review":
        stage = "final_review"
    else:
        stage = f"{execution}_{task_mode}"
    return task_mode, stage


def _manifest_identity(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    manifest_path = str(manifest.get("_manifest_path") or "").strip()
    parent_ref = str(
        manifest.get("retry_of_manifest")
        or manifest.get("split_from_manifest")
        or manifest_path
    ).strip()
    operation_seed = parent_ref or str(manifest.get("created_at") or "")
    operation_id = (
        "manifest-operation-"
        + hashlib.sha256(operation_seed.encode("utf-8")).hexdigest()[:20]
    )
    run_label = str(
        manifest.get("job_name")
        or manifest.get("result_file_name")
        or manifest.get("display_name")
        or ""
    ).strip()
    if not run_label:
        run_label = os.path.basename(os.path.dirname(manifest_path)) if manifest_path else ""
    if not run_label:
        run_label = new_run_id("manifest")
    manifest_id = (
        "manifest-" + hashlib.sha256(manifest_path.encode("utf-8")).hexdigest()[:20]
        if manifest_path
        else ""
    )
    return operation_id, run_label, manifest_id


def import_manifest_results(
    manifest: Mapping[str, Any],
    *,
    result_path: str | os.PathLike[str] | None = None,
    pricing_config: Mapping[str, Any] | None = None,
    ledger: UsageLedger | None = None,
) -> dict[str, Any]:
    """Offline-import usage metadata from a downloaded/sync result JSONL."""
    game_root = str(manifest.get("base_dir") or "").strip()
    if not game_root:
        raise UsageLedgerError("Manifest base_dir/game_root is missing")
    raw_result_path = (
        os.fspath(result_path)
        if result_path is not None
        else str(manifest.get("result_jsonl_path") or "")
    )
    if not raw_result_path:
        raise UsageLedgerError("Result JSONL path is missing")
    path = os.path.abspath(raw_result_path)
    if not os.path.isfile(path):
        raise UsageLedgerError(f"Result JSONL not found: {path}")

    task_mode, stage = _manifest_task_and_stage(manifest)
    operation_id, run_id, manifest_id = _manifest_identity(manifest)
    execution = str(
        manifest.get("execution_mode") or manifest.get("execution") or "batch"
    ).strip().lower()
    settings = manifest.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    thinking_level = str(settings.get("thinking_level") or "")
    default_provider = str(manifest.get("provider") or "").strip().lower()
    if not default_provider:
        default_provider = "gemini" if execution == "batch" else "unknown"
    default_model = str(
        manifest.get("model") or manifest.get("batch_model") or ""
    ).strip()
    effective_pricing = pricing_config if execution == "batch" else None

    candidates: list[dict[str, Any]] = []
    scanned_rows = 0
    skipped_rows = 0
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                scanned_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise UsageLedgerError(
                        f"Invalid result JSON at {path}:{row_index}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    skipped_rows += 1
                    continue
                row_key = str(row.get("key") or row.get("custom_id") or row_index)
                attempts = row.get("provider_response_attempts")
                if isinstance(attempts, list) and attempts:
                    appended_attempt = False
                    for attempt_index, attempt in enumerate(attempts, start=1):
                        if not isinstance(attempt, Mapping):
                            continue
                        attempt_kind = str(attempt.get("kind") or "request")
                        response_payload = attempt.get("response")
                        if not isinstance(response_payload, Mapping):
                            response_payload = {}
                        if attempt_kind == "first_pass" and not response_payload:
                            response_payload = row.get("response_payload")
                            if not isinstance(response_payload, Mapping):
                                response_payload = row.get("response")
                            if not isinstance(response_payload, Mapping):
                                response_payload = {}
                        usage = attempt.get("usage_metadata")
                        if attempt_kind == "first_pass" and not isinstance(
                            usage, Mapping
                        ):
                            usage = row.get("usage_metadata")
                        if not isinstance(usage, Mapping):
                            usage = extract_provider_usage(response_payload)
                        if not response_payload and not usage:
                            continue
                        raw_item_ids = attempt.get("item_ids")
                        item_ids = (
                            [str(item) for item in raw_item_ids]
                            if isinstance(raw_item_ids, list)
                            else []
                        )
                        attempt_key = f"{row_key}:attempt:{attempt_index}"
                        candidates.append(
                            build_usage_record(
                                game_root=game_root,
                                task_mode=task_mode,
                                stage=str(row.get("pipeline_stage") or stage),
                                provider=str(row.get("provider") or default_provider),
                                model=str(row.get("model") or default_model),
                                usage_metadata=usage,
                                response_payload=response_payload,
                                operation_id=operation_id,
                                run_id=run_id,
                                manifest_id=manifest_id,
                                thinking_level=thinking_level,
                                execution_mode=str(row.get("execution_mode") or execution),
                                source_key=attempt_key,
                                source={
                                    "kind": "manifest_result_attempt",
                                    "manifest_path": str(manifest.get("_manifest_path") or ""),
                                    "result_path": path,
                                    "row_key": row_key,
                                    "row_index": row_index,
                                    "attempt_index": attempt_index,
                                    "attempt_kind": attempt_kind,
                                    "item_ids": item_ids,
                                },
                                pricing_config=effective_pricing,
                            )
                        )
                        appended_attempt = True
                    if appended_attempt:
                        continue
                response_payload = row.get("response_payload")
                if not isinstance(response_payload, Mapping):
                    response_payload = row.get("response")
                if not isinstance(response_payload, Mapping):
                    response_payload = {}
                usage = row.get("usage_metadata")
                if not isinstance(usage, Mapping):
                    usage = extract_provider_usage(response_payload)
                if row.get("error") and not response_payload and not usage:
                    skipped_rows += 1
                    continue
                if not response_payload and not usage:
                    skipped_rows += 1
                    continue
                candidates.append(
                    build_usage_record(
                        game_root=game_root,
                        task_mode=task_mode,
                        stage=str(row.get("pipeline_stage") or stage),
                        provider=str(row.get("provider") or default_provider),
                        model=str(row.get("model") or default_model),
                        usage_metadata=usage,
                        response_payload=response_payload,
                        operation_id=operation_id,
                        run_id=run_id,
                        manifest_id=manifest_id,
                        thinking_level=thinking_level,
                        execution_mode=str(row.get("execution_mode") or execution),
                        source_key=row_key,
                        source={
                            "kind": "manifest_results",
                            "manifest_path": str(manifest.get("_manifest_path") or ""),
                            "result_path": path,
                            "row_key": row_key,
                            "row_index": row_index,
                        },
                        pricing_config=effective_pricing,
                    )
                )
    except OSError as exc:
        raise UsageLedgerError(f"Could not read result JSONL {path}: {exc}") from exc

    store = ledger or UsageLedger(game_root)
    write_summary = store.add_records(candidates)
    return {
        **write_summary,
        "game_root": canonical_game_root(game_root),
        "result_path": path,
        "scanned_rows": scanned_rows,
        "candidate_records": len(candidates),
        "skipped_rows": skipped_rows,
    }


def record_generation_usage(
    *,
    game_root: str | os.PathLike[str],
    task_mode: str,
    stage: str,
    provider: str,
    model: str,
    usage_metadata: Mapping[str, Any] | None,
    response_payload: Any = None,
    operation_id: str = "",
    run_id: str = "",
    thinking_level: str = "",
    execution_mode: str = "sync",
    source_key: str = "",
    source: Mapping[str, Any] | None = None,
    pricing_config: Mapping[str, Any] | None = None,
    ledger: UsageLedger | None = None,
) -> dict[str, Any]:
    record = build_usage_record(
        game_root=game_root,
        task_mode=task_mode,
        stage=stage,
        provider=provider,
        model=model,
        usage_metadata=usage_metadata,
        response_payload=response_payload,
        operation_id=operation_id,
        run_id=run_id,
        thinking_level=thinking_level,
        execution_mode=execution_mode,
        source_key=source_key,
        source=source,
        pricing_config=pricing_config,
    )
    store = ledger or UsageLedger(game_root)
    return store.add_records([record])


def normalize_group_by(group_by: Sequence[str] | str | None) -> tuple[str, ...]:
    raw = (
        [part.strip() for part in group_by.split(",")]
        if isinstance(group_by, str)
        else list(group_by or [])
    )
    if not raw:
        raw = ["task", "stage", "provider", "model"]
    fields: list[str] = []
    for value in raw:
        normalized = GROUP_FIELD_ALIASES.get(str(value).strip().lower())
        if not normalized:
            raise UsageLedgerError(f"Unsupported usage group field: {value}")
        if normalized not in fields:
            fields.append(normalized)
    return tuple(fields)


def _aggregate_cost(records: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    values: dict[str, float] = {}
    known = 0
    for record in records:
        cost = _nonnegative_float(record.get(f"{prefix}_cost"))
        if cost is None:
            continue
        currency = str(record.get(f"{prefix}_cost_currency") or "unknown")
        values[currency] = values.get(currency, 0.0) + cost
        known += 1
    return {
        "values": {
            currency: round(value, 8)
            for currency, value in sorted(values.items())
        },
        "known_records": known,
        "unknown_records": len(records) - known,
    }


def aggregate_usage_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "records": len(records),
        "calls": sum(
            int(record.get("calls") or 0)
            for record in records
            if _nonnegative_int(record.get("calls")) is not None
        ),
    }
    for field in TOKEN_FIELDS:
        values = [
            parsed
            for record in records
            if (parsed := _nonnegative_int(record.get(field))) is not None
        ]
        totals[field] = sum(values) if values else None
        totals[f"{field}_known_records"] = len(values)
        totals[f"{field}_unknown_records"] = len(records) - len(values)
    totals["estimated_cost"] = _aggregate_cost(records, "estimated")
    totals["actual_cost"] = _aggregate_cost(records, "actual")
    return totals


def _record_matches(record: Mapping[str, Any], field: str, expected: str) -> bool:
    actual = str(record.get(field) or "")
    if field in {"provider", "task_mode", "stage", "execution_mode"}:
        return actual.lower() == expected.lower()
    return actual == expected


def query_usage(
    game_root: str | os.PathLike[str],
    *,
    task: str = "",
    stage: str = "",
    provider: str = "",
    model: str = "",
    group_by: Sequence[str] | str | None = None,
    ledger: UsageLedger | None = None,
) -> dict[str, Any]:
    store = ledger or UsageLedger(game_root)
    payload = store.load()
    records: list[dict[str, Any]] = list(payload.get("records") or [])
    filters = {
        "task_mode": normalize_task_mode(task) if task else "",
        "stage": str(stage or "").strip(),
        "provider": str(provider or "").strip(),
        "model": str(model or "").strip(),
    }
    filtered = [
        record
        for record in records
        if all(
            not expected or _record_matches(record, field, expected)
            for field, expected in filters.items()
        )
    ]
    group_fields = normalize_group_by(group_by)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in filtered:
        key = tuple(str(record.get(field) or "unknown") for field in group_fields)
        buckets.setdefault(key, []).append(record)
    groups = []
    for key, rows in sorted(buckets.items(), key=lambda item: item[0]):
        group = {field: value for field, value in zip(group_fields, key)}
        group["totals"] = aggregate_usage_records(rows)
        groups.append(group)

    recent_run = None
    if filtered:
        latest = max(filtered, key=lambda row: str(row.get("recorded_at") or ""))
        latest_run_id = str(latest.get("run_id") or "")
        recent_rows = [
            row for row in filtered if str(row.get("run_id") or "") == latest_run_id
        ]
        recent_run = {
            "run_id": latest_run_id or None,
            "recorded_at": latest.get("recorded_at"),
            "task_modes": sorted(
                {str(row.get("task_mode") or "unknown") for row in recent_rows}
            ),
            "stages": sorted(
                {str(row.get("stage") or "unknown") for row in recent_rows}
            ),
            "providers": sorted(
                {str(row.get("provider") or "unknown") for row in recent_rows}
            ),
            "models": sorted(
                {str(row.get("model") or "unknown") for row in recent_rows}
            ),
            "totals": aggregate_usage_records(recent_rows),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "project": dict(store.project),
        "ledger_path": store.path,
        "filters": {key: value for key, value in filters.items() if value},
        "group_by": list(group_fields),
        "totals": aggregate_usage_records(filtered),
        "groups": groups,
        "recent_run": recent_run,
    }


def _format_token_metric(totals: Mapping[str, Any], field: str) -> str:
    value = totals.get(field)
    unknown = int(totals.get(f"{field}_unknown_records") or 0)
    text = "unknown" if value is None else f"{int(value):,}"
    if unknown:
        text += f" ({unknown} record(s) unknown)"
    return text


def _format_cost_metric(metric: Mapping[str, Any]) -> str:
    values = metric.get("values") if isinstance(metric, Mapping) else {}
    unknown = int(metric.get("unknown_records") or 0) if isinstance(metric, Mapping) else 0
    if isinstance(values, Mapping) and values:
        text = ", ".join(
            f"{float(value):.6f} {currency}"
            for currency, value in values.items()
        )
    else:
        text = "unknown"
    if unknown:
        text += f" ({unknown} record(s) unknown)"
    return text


def format_usage_report(report: Mapping[str, Any]) -> list[str]:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    project = report.get("project") if isinstance(report.get("project"), Mapping) else {}
    lines = [
        "Model usage ledger:",
        f"- Project: {project.get('game_root') or '(unknown)'}",
        f"- Ledger: {report.get('ledger_path') or '(unknown)'}",
        f"- Records / calls: {int(totals.get('records') or 0)} / {int(totals.get('calls') or 0)}",
        f"- Prompt tokens: {_format_token_metric(totals, 'prompt_tokens')}",
        f"- Completion tokens: {_format_token_metric(totals, 'completion_tokens')}",
        f"- Total tokens: {_format_token_metric(totals, 'total_tokens')}",
        f"- Thoughts tokens: {_format_token_metric(totals, 'thoughts_tokens')}",
        f"- Cached tokens: {_format_token_metric(totals, 'cached_tokens')}",
        (
            "- Estimated cost (configured pricing, not provider billing): "
            + _format_cost_metric(totals.get("estimated_cost") or {})
        ),
        (
            "- Actual cost (provider reported only): "
            + _format_cost_metric(totals.get("actual_cost") or {})
        ),
    ]
    groups = report.get("groups")
    if isinstance(groups, list) and groups:
        lines.append("Groups:")
        group_fields = report.get("group_by") or []
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            label = " / ".join(str(group.get(field) or "unknown") for field in group_fields)
            group_totals = group.get("totals") if isinstance(group.get("totals"), Mapping) else {}
            lines.append(
                f"- {label}: {int(group_totals.get('calls') or 0)} call(s), "
                f"{_format_token_metric(group_totals, 'total_tokens')} total tokens"
            )
    recent = report.get("recent_run")
    if isinstance(recent, Mapping):
        recent_totals = (
            recent.get("totals") if isinstance(recent.get("totals"), Mapping) else {}
        )
        lines.append(
            "- Recent run: "
            f"{recent.get('run_id') or '(unknown)'}; "
            f"{int(recent_totals.get('calls') or 0)} call(s); "
            f"{_format_token_metric(recent_totals, 'total_tokens')} total tokens"
        )
    return lines
