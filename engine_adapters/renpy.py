"""Read-only Ren'Py project discovery, inventory, and occurrence extraction.

P1 deliberately delegates the two legacy extraction views to
``translator_runtime`` and independently inventories every string-shaped
candidate.  The former preserves current task/identity semantics; the latter
makes parse failures and unsupported structures observable instead of letting
the legacy scanners' broad exception handlers become coverage claims.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import io
import os
from pathlib import Path
import tokenize
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import translation_core

from .contracts import (
    CONTENT_FINGERPRINT_SCHEMA_VERSION,
    ENGINE_ADAPTER_PROTOCOL_VERSION,
    VALIDATION_SCHEMA_VERSION,
    WRITEBACK_PLAN_SCHEMA_VERSION,
    Candidate,
    CandidateInventory,
    CoverageReportDraft,
    EngineCapabilities,
    InventoryPolicy,
    LocalizationMode,
    Occurrence,
    OpaqueLocator,
    ProjectDiscovery,
    ProjectDiscoveryRequest,
    RelocationResult,
    SourceDocument,
    ValidatedTranslation,
    ValidationResult,
    WritebackOperation,
    WritebackPlan,
)
from .coverage import (
    REVIEW_POLICIES,
    CoverageReport,
    build_coverage_report,
    classification_rules_digest,
    digest_json,
)
from .writeback import source_snapshot_fingerprint


ADAPTER_VERSION = "1.1.0"
LOCATOR_SCHEMA_VERSION = 1
# Same-file + same-source alone scores 125. Content-evidence matches must also
# clear this floor so bare unique-string hits without structural signals fail closed.
# Typical unique stale-block fallback scores 140+ (shared block_occurrence / speaker).
CONTENT_EVIDENCE_MIN_SCORE = 140


@dataclass(frozen=True)
class RenPyTranslationSnapshot:
    """One immutable discovery/inventory/extraction pass for workflow callers."""

    project: ProjectDiscovery
    inventory: CandidateInventory
    report: CoverageReport
    occurrences: tuple[Occurrence, ...]
    pending_tasks_by_file: Mapping[str, tuple[Mapping[str, Any], ...]]
    progress_by_file: Mapping[str, Mapping[str, int]]
    review_policy: str

    @property
    def pending_task_count(self) -> int:
        return sum(len(items) for items in self.pending_tasks_by_file.values())

    @property
    def recognized_unit_count(self) -> int:
        return len(self.occurrences)


def _normalize_rel_path(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Path escapes the localization root: {value}")
    return "/".join(parts)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _candidate_id(
    project: ProjectDiscovery,
    locator: OpaqueLocator,
) -> str:
    return "cand1:" + digest_json(
        {
            "engine": locator.engine,
            "project_snapshot_fingerprint": project.project_snapshot_fingerprint,
            "locator": locator.to_dict(),
        }
    )


def _candidate_scopes(classification: str, structure_kind: str) -> tuple[str, str]:
    if classification in {"translatable", "already_translated"}:
        analysis_scope = (
            "include" if structure_kind in {"dialogue_string", "narration_string"} else "exclude"
        )
        return "include", analysis_scope
    if classification == "explicitly_excluded":
        return "exclude", "exclude"
    return "unknown", "unknown"


def _bounded_excerpt(value: str, limit: int = 240) -> str:
    text = str(value or "").replace("\r", "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _normalize_fingerprint_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


class RenPyAdapter:
    """P2 Ren'Py adapter with read-only relocation and writeback planning.

    The adapter can read only the project/localization roots supplied to
    :meth:`discover_project`.  It returns declarative data only; common
    workflow code retains project, manifest, and atomic write authority.
    """

    protocol_version = ENGINE_ADAPTER_PROTOCOL_VERSION
    engine = "renpy"
    adapter_version = ADAPTER_VERSION
    locator_schema_version = LOCATOR_SCHEMA_VERSION

    def __init__(self, legacy_module: ModuleType | None = None):
        self._legacy_module = legacy_module

    def _legacy(self) -> ModuleType:
        if self._legacy_module is None:
            # Lazy import prevents translator_runtime -> adapter -> runtime cycles
            # and keeps importing this module free of provider/GUI dependencies.
            import translator_runtime

            self._legacy_module = translator_runtime
        return self._legacy_module

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.engine,
            adapter_version=self.adapter_version,
            supported_localization_modes=(LocalizationMode.HYBRID,),
            selected_localization_mode=LocalizationMode.HYBRID,
            source_inventory=True,
            native_catalog=True,
            relocation=True,
            declarative_writeback=("text_span_replace",),
            native_catalog_required_for_writeback=True,
        )

    def behavior_digest(self) -> str:
        return digest_json(
            {
                "engine_adapter_protocol_version": self.protocol_version,
                "engine": self.engine,
                "adapter_version": self.adapter_version,
                "locator_schema_version": self.locator_schema_version,
                "classification_rules_digest": classification_rules_digest(),
                "legacy_equivalence_contract": [
                    "collect_tasks_with_progress",
                    "scan_all_translation_units",
                ],
                "p2_relocation": True,
                "p2_validation": True,
                "p2_writeback_plan": True,
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "writeback_plan_schema_version": WRITEBACK_PLAN_SCHEMA_VERSION,
            }
        )

    def discover_project(
        self,
        request: ProjectDiscoveryRequest,
    ) -> ProjectDiscovery:
        localization_root = os.path.abspath(os.fspath(request.localization_root))
        project_root = os.path.abspath(
            os.fspath(request.project_root)
            if request.project_root
            else os.path.dirname(localization_root)
        )
        include_files = {
            _normalize_rel_path(value)
            for value in request.include_files
            if str(value or "").strip()
        }
        include_prefixes = tuple(
            sorted(
                {
                    _normalize_rel_path(value)
                    for value in request.include_prefixes
                    if str(value or "").strip()
                }
            )
        )

        documents: list[SourceDocument] = []
        if os.path.isdir(localization_root):
            localization_real = os.path.realpath(localization_root)
            for root, dir_names, file_names in os.walk(localization_root):
                dir_names.sort()
                for file_name in sorted(file_names):
                    if not file_name.endswith(".rpy"):
                        continue
                    file_path = os.path.abspath(os.path.join(root, file_name))
                    file_real = os.path.realpath(file_path)
                    try:
                        within_root = (
                            os.path.commonpath([localization_real, file_real]) == localization_real
                        )
                    except ValueError:
                        within_root = False
                    if not within_root:
                        raise ValueError(
                            f"Ren'Py source resolves outside localization root: {file_path}"
                        )

                    rel_path = _normalize_rel_path(os.path.relpath(file_path, localization_root))
                    if include_files or include_prefixes:
                        allowed = rel_path in include_files
                        if not allowed:
                            allowed = any(
                                rel_path.startswith(prefix) for prefix in include_prefixes
                            )
                        if not allowed:
                            continue

                    content = Path(file_path).read_bytes()
                    documents.append(
                        SourceDocument(
                            file_rel_path=rel_path,
                            file_path=file_path,
                            size=len(content),
                            sha256=_sha256_bytes(content),
                            content=content,
                        )
                    )

        documents.sort(key=lambda item: item.file_rel_path)
        source_payload = [document.manifest_entry() for document in documents]
        source_fingerprint = digest_json(source_payload)
        project_snapshot_fingerprint = digest_json(
            {
                "engine": self.engine,
                "localization_mode": LocalizationMode.HYBRID.value,
                "target_language": str(request.target_language or ""),
                "source_fingerprint": source_fingerprint,
            }
        )
        catalog_provenance = {
            "format": "renpy_tl_rpy",
            "target_language": str(request.target_language or ""),
            "path_set_digest": digest_json([document.file_rel_path for document in documents]),
            "generator": "",
            "engine_version": "",
            "generated_at": "",
            "generation_command_digest": "",
            "recorded_source_fingerprint": "",
            "live_source_fingerprint": source_fingerprint,
            "provenance_status": "inferred",
        }
        return ProjectDiscovery(
            engine=self.engine,
            adapter_version=self.adapter_version,
            project_root=project_root,
            localization_root=localization_root,
            target_language=str(request.target_language or ""),
            project_snapshot_fingerprint=project_snapshot_fingerprint,
            source_fingerprint=source_fingerprint,
            source_documents=tuple(documents),
            localization_mode=LocalizationMode.HYBRID,
            catalog_provenance=catalog_provenance,
        )

    def inventory_candidates(
        self,
        project: ProjectDiscovery,
        policy: InventoryPolicy,
    ) -> CandidateInventory:
        if project.engine != self.engine:
            raise ValueError(f"RenPyAdapter cannot inventory engine={project.engine!r}.")
        if policy.review_policy not in REVIEW_POLICIES:
            raise ValueError(f"Unsupported coverage review policy: {policy.review_policy}")
        candidates: list[Candidate] = []
        file_entries: list[Mapping[str, Any]] = []
        for document in project.source_documents:
            document_candidates, progress = self._inventory_document(
                project,
                document,
            )
            candidates.extend(document_candidates)
            file_entry = document.manifest_entry()
            file_entry.update(
                {
                    "candidate_count": len(document_candidates),
                    "pending_task_count": sum(
                        candidate.classification == "translatable"
                        for candidate in document_candidates
                    ),
                    "translated_count": int(progress.get("translated_count") or 0),
                    "parse_error_count": sum(
                        candidate.classification == "parse_error"
                        for candidate in document_candidates
                    ),
                }
            )
            file_entries.append(file_entry)

        return CandidateInventory(
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            project_snapshot_fingerprint=project.project_snapshot_fingerprint,
            candidates=tuple(candidates),
            files_scanned=tuple(file_entries),
        )

    def _inventory_document(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
    ) -> tuple[list[Candidate], Mapping[str, int]]:
        legacy = self._legacy()
        try:
            lines = document.lines()
        except UnicodeError as exc:
            candidate = self._parse_error_candidate(
                project,
                document,
                line_index=0,
                start_col=0,
                end_col=0,
                candidate_ordinal=1,
                block_name="_global",
                block_occurrence=1,
                ordinal=1,
                reason_code="renpy.tokenize_error",
                structure_kind="source_decode_error",
                excerpt=f"{type(exc).__name__}: {exc}",
            )
            return [candidate], {"translated_count": 0}

        scan_errors: list[tuple[str, Exception]] = []
        try:
            raw_tasks, progress = legacy.collect_tasks_with_progress(lines)
        except Exception as exc:  # fail visibly; never claim a complete inventory
            raw_tasks, progress = [], {"translated_count": 0}
            scan_errors.append(("collect_tasks_with_progress", exc))
        try:
            identity_mapping = legacy.scan_all_translation_units(
                lines,
                document.file_rel_path,
            )
        except Exception as exc:  # fail visibly; never claim a complete inventory
            identity_mapping = {}
            scan_errors.append(("scan_all_translation_units", exc))

        task_by_span: dict[tuple[int, int, int], dict[str, Any]] = {}
        for task in raw_tasks:
            current = dict(task)
            current["file_rel_path"] = document.file_rel_path
            current["file_path"] = document.file_path
            current["id"] = translation_core.build_identity_v2(
                document.file_rel_path,
                current.get("block_name", "_global"),
                current.get("block_index", 0),
                current.get("source_for_id") or current.get("text") or "",
                block_occurrence=current.get("block_occurrence", 1),
            )
            current["progress_entry"] = (
                f"task:{int(current.get('line') or 0)}:{int(current.get('start') or 0)}"
            )
            span = (
                int(current.get("line") or 0),
                int(current.get("start") or 0),
                int(current.get("end") or 0),
            )
            task_by_span[span] = current

        identity_by_span: dict[tuple[int, int, int], str] = {}
        for identity, value in identity_mapping.items():
            line_index, start_col, end_col, _text = value
            identity_by_span[(int(line_index), int(start_col), int(end_col))] = identity

        is_translation_file = any(line.lstrip().startswith("translate ") for line in lines)
        candidates: list[Candidate] = []
        paired_source_marker_lines: set[int] = set()
        character_display_spans: list[tuple[int, int, int, int]] = []
        valid_character_definition_lines: set[int] = set()
        speaker_names: dict[str, str] = {}
        block_occurrences: dict[str, int] = {}
        current_block = "_global"
        current_block_occurrence: int | None = None
        identity_ordinal = 0
        candidate_block_ordinal = 0
        candidate_ordinal = 0

        for stage, scan_exception in scan_errors:
            candidate_ordinal += 1
            candidates.append(
                self._parse_error_candidate(
                    project,
                    document,
                    line_index=0,
                    start_col=0,
                    end_col=0,
                    candidate_ordinal=candidate_ordinal,
                    block_name="_global",
                    block_occurrence=1,
                    ordinal=candidate_ordinal,
                    reason_code="renpy.ast_parse_error",
                    structure_kind="legacy_scan_error",
                    excerpt=(f"{stage}: {type(scan_exception).__name__}: {scan_exception}"),
                )
            )

        for line_index, line in enumerate(lines):
            definition_match = legacy.CHARACTER_DEFINE_RE.match(line)
            definition = None
            definition_failed = False
            if definition_match:
                try:
                    definition = legacy._parse_character_definition(
                        lines,
                        line_index,
                    )
                except Exception as exc:
                    definition_failed = True
                    candidate_ordinal += 1
                    candidate_block_ordinal += 1
                    candidates.append(
                        self._parse_error_candidate(
                            project,
                            document,
                            line_index=line_index,
                            start_col=0,
                            end_col=len(line.rstrip("\r\n")),
                            candidate_ordinal=candidate_ordinal,
                            block_name=current_block,
                            block_occurrence=self._next_block_occurrence(
                                block_occurrences,
                                current_block,
                                current_block_occurrence,
                            ),
                            ordinal=candidate_block_ordinal,
                            reason_code="renpy.ast_parse_error",
                            structure_kind="character_definition",
                            excerpt=f"{type(exc).__name__}: {exc}",
                        )
                    )
                if definition is None and not definition_failed:
                    candidate_ordinal += 1
                    candidate_block_ordinal += 1
                    candidates.append(
                        self._parse_error_candidate(
                            project,
                            document,
                            line_index=line_index,
                            start_col=0,
                            end_col=len(line.rstrip("\r\n")),
                            candidate_ordinal=candidate_ordinal,
                            block_name=current_block,
                            block_occurrence=self._next_block_occurrence(
                                block_occurrences,
                                current_block,
                                current_block_occurrence,
                            ),
                            ordinal=candidate_block_ordinal,
                            reason_code="renpy.ast_parse_error",
                            structure_kind="character_definition",
                            excerpt=line.strip(),
                        )
                    )
            if definition:
                definition_end = self._character_definition_end_line(
                    legacy,
                    lines,
                    line_index,
                )
                valid_character_definition_lines.update(range(line_index, definition_end + 1))
                speaker_id = str(definition.get("speaker_id") or "")
                speaker_name = str(definition.get("speaker_name") or "")
                if speaker_name:
                    speaker_names[speaker_id] = speaker_name
                else:
                    speaker_names.pop(speaker_id, None)
                character_display_spans.extend(definition.get("display_spans") or ())

            stripped = line.strip()
            if stripped.startswith("translate "):
                block_name = legacy._translate_block_name(line)
                if block_name:
                    current_block = block_name
                    current_block_occurrence = None
                    identity_ordinal = 0
                    candidate_block_ordinal = 0

            tokens: list[tokenize.TokenInfo] = []
            tokenize_error: Exception | None = None
            try:
                tokens.extend(tokenize.generate_tokens(io.StringIO(line).readline))
            except (IndentationError, SyntaxError, tokenize.TokenError) as exc:
                tokenize_error = exc
            if tokenize_error is not None and line_index not in valid_character_definition_lines:
                candidate_ordinal += 1
                candidate_block_ordinal += 1
                candidates.append(
                    self._parse_error_candidate(
                        project,
                        document,
                        line_index=line_index,
                        start_col=0,
                        end_col=len(line.rstrip("\r\n")),
                        candidate_ordinal=candidate_ordinal,
                        block_name=current_block,
                        block_occurrence=self._next_block_occurrence(
                            block_occurrences,
                            current_block,
                            current_block_occurrence,
                        ),
                        ordinal=candidate_block_ordinal,
                        reason_code="renpy.tokenize_error",
                        structure_kind="tokenize_region",
                        excerpt=(f"{type(tokenize_error).__name__}: {tokenize_error}"),
                    )
                )

            quote_errors = [
                token
                for token in tokens
                if token.type == tokenize.ERRORTOKEN and token.string in {'"', "'"}
            ]
            if quote_errors:
                token = quote_errors[0]
                candidate_ordinal += 1
                candidate_block_ordinal += 1
                candidates.append(
                    self._parse_error_candidate(
                        project,
                        document,
                        line_index=line_index,
                        start_col=token.start[1],
                        end_col=max(token.end[1], token.start[1] + 1),
                        candidate_ordinal=candidate_ordinal,
                        block_name=current_block,
                        block_occurrence=self._next_block_occurrence(
                            block_occurrences,
                            current_block,
                            current_block_occurrence,
                        ),
                        ordinal=candidate_block_ordinal,
                        reason_code="renpy.tokenize_error",
                        structure_kind="string_literal",
                        excerpt=line.strip(),
                    )
                )

            candidate_tokens: list[tuple[int, tokenize.TokenInfo]] = []
            fstring_start = getattr(tokenize, "FSTRING_START", -1)
            fstring_end = getattr(tokenize, "FSTRING_END", -1)
            token_index = 0
            while token_index < len(tokens):
                token = tokens[token_index]
                if token.type == tokenize.STRING:
                    candidate_tokens.append((token_index, token))
                elif token.type == fstring_start:
                    depth = 1
                    end_index = token_index + 1
                    while end_index < len(tokens):
                        if tokens[end_index].type == fstring_start:
                            depth += 1
                        elif tokens[end_index].type == fstring_end:
                            depth -= 1
                            if depth == 0:
                                break
                        end_index += 1
                    end_token = tokens[end_index] if end_index < len(tokens) else token
                    synthetic = tokenize.TokenInfo(
                        type=tokenize.STRING,
                        string=line[token.start[1] : end_token.end[1]],
                        start=token.start,
                        end=end_token.end,
                        line=line,
                    )
                    candidate_tokens.append((token_index, synthetic))
                    token_index = end_index
                token_index += 1

            for token_index, token in candidate_tokens:
                candidate_ordinal += 1
                candidate_block_ordinal += 1
                span = (line_index, token.start[1], token.end[1])
                identity = identity_by_span.get(span)
                legacy_item = task_by_span.get(span)
                if identity is not None or legacy_item is not None:
                    if current_block_occurrence is None:
                        current_block_occurrence = legacy._ensure_identity_block_occurrence(
                            block_occurrences,
                            current_block,
                            current_block_occurrence,
                        )
                    identity_ordinal += 1
                    ordinal = identity_ordinal
                else:
                    ordinal = candidate_block_ordinal
                block_occurrence = self._next_block_occurrence(
                    block_occurrences,
                    current_block,
                    current_block_occurrence,
                )

                marker = None
                if identity is not None or legacy_item is not None:
                    marker = self._source_marker_evidence(
                        legacy,
                        lines,
                        line_index,
                        is_translation_file=is_translation_file,
                    )
                if marker is not None:
                    paired_source_marker_lines.add(int(marker["line_index"]))

                candidate = self._candidate_for_token(
                    legacy=legacy,
                    project=project,
                    document=document,
                    lines=lines,
                    line_index=line_index,
                    line=line,
                    token=token,
                    token_index=token_index,
                    tokens=tokens,
                    is_translation_file=is_translation_file,
                    character_display_spans=character_display_spans,
                    speaker_names=speaker_names,
                    block_name=current_block,
                    block_occurrence=block_occurrence,
                    ordinal=ordinal,
                    candidate_ordinal=candidate_ordinal,
                    identity=identity,
                    legacy_item=legacy_item,
                    marker=marker,
                )
                candidates.append(candidate)

        if is_translation_file:
            marker_block = "_global"
            marker_block_occurrence = 1
            marker_block_occurrences: dict[str, int] = {}
            for line_index, line in enumerate(lines):
                raw_line = line.rstrip("\r\n")
                if line.strip().startswith("translate "):
                    parsed_block = legacy._translate_block_name(line)
                    if parsed_block:
                        marker_block = parsed_block
                        marker_block_occurrence = marker_block_occurrences.get(parsed_block, 0) + 1
                        marker_block_occurrences[parsed_block] = marker_block_occurrence
                comment_match = legacy.TL_COMMENT_SOURCE_RE.match(raw_line)
                old_match = legacy.TL_OLD_LINE_RE.match(raw_line)
                if comment_match:
                    if legacy.is_voice_comment_match(comment_match):
                        candidate_ordinal += 1
                        candidates.append(
                            self._marker_candidate(
                                project,
                                document,
                                line_index=line_index,
                                line=line,
                                candidate_ordinal=candidate_ordinal,
                                reason_code="renpy.voice_asset",
                                classification="explicitly_excluded",
                                structure_kind="voice_source_comment",
                                source_marker_kind="comment",
                                block_name=marker_block,
                                block_occurrence=marker_block_occurrence,
                            )
                        )
                    elif line_index not in paired_source_marker_lines:
                        candidate_ordinal += 1
                        candidates.append(
                            self._marker_candidate(
                                project,
                                document,
                                line_index=line_index,
                                line=line,
                                candidate_ordinal=candidate_ordinal,
                                reason_code="renpy.source_marker_unpaired",
                                classification="parse_error",
                                structure_kind="source_comment",
                                source_marker_kind="comment",
                                block_name=marker_block,
                                block_occurrence=marker_block_occurrence,
                            )
                        )
                elif self._looks_like_nonstandard_source_comment(raw_line):
                    candidate_ordinal += 1
                    candidates.append(
                        self._marker_candidate(
                            project,
                            document,
                            line_index=line_index,
                            line=line,
                            candidate_ordinal=candidate_ordinal,
                            reason_code="renpy.custom_statement_unsupported",
                            classification="unsupported",
                            structure_kind="nonstandard_source_comment",
                            source_marker_kind="comment",
                            block_name=marker_block,
                            block_occurrence=marker_block_occurrence,
                        )
                    )
                elif old_match and line_index not in paired_source_marker_lines:
                    # The old literal already has a token candidate. Replace its
                    # exclusion classification with the observable parse error.
                    for candidate in reversed(candidates):
                        locator = candidate.locator.locator
                        if (
                            locator.get("file_rel_path") == document.file_rel_path
                            and locator.get("line_hint") == line_index + 1
                            and candidate.structure_kind == "old_source_marker"
                        ):
                            candidate.classification = "parse_error"
                            merged_reasons: list[str] = []
                            for code in tuple(candidate.reason_codes or ()) + (
                                "renpy.source_marker_unpaired",
                            ):
                                if code and code not in merged_reasons:
                                    merged_reasons.append(code)
                            candidate.reason_codes = tuple(merged_reasons)
                            candidate.translation_scope = "unknown"
                            candidate.analysis_scope = "unknown"
                            break

        candidates.sort(
            key=lambda item: (
                int(item.locator.locator.get("line_hint") or 0),
                int(item.locator.locator.get("start_col_hint") or 0),
                int(item.locator.locator.get("candidate_ordinal") or 0),
            )
        )
        return candidates, progress

    @staticmethod
    def _next_block_occurrence(
        block_occurrences: Mapping[str, int],
        block_name: str,
        current_occurrence: int | None,
    ) -> int:
        if current_occurrence:
            return int(current_occurrence)
        return int(block_occurrences.get(block_name, 0)) + 1

    @staticmethod
    def _character_definition_end_line(
        legacy: ModuleType,
        lines: Sequence[str],
        start_index: int,
        max_lines: int = 80,
    ) -> int:
        match = legacy.CHARACTER_DEFINE_RE.match(lines[start_index])
        if not match:
            return start_index
        call_start_col = match.start("call")
        pieces: list[str] = []
        end_limit = min(len(lines), start_index + max_lines)
        for line_index in range(start_index, end_limit):
            pieces.append(
                lines[line_index][call_start_col:]
                if line_index == start_index
                else lines[line_index]
            )
            try:
                ast.parse("".join(pieces), mode="eval")
            except SyntaxError:
                continue
            return line_index
        return start_index

    def _candidate_for_token(
        self,
        *,
        legacy: ModuleType,
        project: ProjectDiscovery,
        document: SourceDocument,
        lines: Sequence[str],
        line_index: int,
        line: str,
        token: tokenize.TokenInfo,
        token_index: int,
        tokens: Sequence[tokenize.TokenInfo],
        is_translation_file: bool,
        character_display_spans: Sequence[tuple[int, int, int, int]],
        speaker_names: Mapping[str, str],
        block_name: str,
        block_occurrence: int,
        ordinal: int,
        candidate_ordinal: int,
        identity: str | None,
        legacy_item: Mapping[str, Any] | None,
        marker: Mapping[str, Any] | None,
    ) -> Candidate:
        prefix, quote = legacy.parse_string_literal_format(token.string)
        literal_prefix = token.string[: len(token.string) - len(token.string.lstrip("rRuUbBfF"))]
        is_dynamic = "f" in literal_prefix.lower()
        try:
            text_value = ast.literal_eval(token.string)
            literal_error: Exception | None = None
        except Exception as exc:
            text_value = ""
            literal_error = exc

        speaker_id = ""
        stripped = line.strip()
        if not (is_translation_file and stripped.startswith("new ")):
            speaker_id = legacy.infer_dialogue_speaker_id(
                line,
                token.start[1],
            )
        is_speaker_label = legacy._is_say_speaker_label_string_token(
            line,
            tokens,
            token_index,
        )
        if speaker_id or is_speaker_label:
            structure_kind = "dialogue_string"
            supported_reason = "renpy.dialogue_string"
        else:
            structure_kind = "narration_string"
            supported_reason = "renpy.narration_string"

        reasons: list[str]
        classification: str
        unit: translation_core.TranslationUnit | None = None
        final_legacy_item: Mapping[str, Any] | None = None

        if legacy_item is not None:
            classification = "translatable"
            reasons = [supported_reason]
            final_legacy_item = dict(legacy_item)
        elif identity is not None:
            classification = "already_translated"
            reasons = [supported_reason, "renpy.catalog.translation_present"]
        elif is_dynamic:
            classification = "unsupported"
            reasons = ["renpy.dynamic_string_expression"]
            structure_kind = "dynamic_string_expression"
        elif literal_error is not None or not isinstance(text_value, str):
            classification = "parse_error"
            reasons = ["renpy.ast_parse_error"]
            structure_kind = "string_literal"
        elif is_translation_file and legacy._is_keyword_argument_string_token(tokens, token_index):
            classification = "explicitly_excluded"
            reasons = ["renpy.keyword_argument"]
            structure_kind = "keyword_argument"
        elif legacy._is_character_display_token(
            line_index,
            token,
            character_display_spans,
        ):
            classification = "explicitly_excluded"
            reasons = ["renpy.character_display_definition"]
            structure_kind = "character_display_definition"
        elif stripped == "voice" or stripped.startswith("voice "):
            classification = "explicitly_excluded"
            reasons = ["renpy.voice_asset"]
            structure_kind = "voice_statement"
        elif is_translation_file and stripped.startswith("old "):
            if legacy.TL_OLD_LINE_RE.match(line.rstrip("\r\n")):
                classification = "explicitly_excluded"
                reasons = ["renpy.old_new_pair"]
                structure_kind = "old_source_marker"
            else:
                classification = "unsupported"
                reasons = ["renpy.custom_statement_unsupported"]
                structure_kind = "nonstandard_old_source_marker"
        elif is_translation_file and stripped.startswith("new "):
            classification = "parse_error"
            reasons = ["renpy.source_marker_unpaired"]
            structure_kind = "new_translation_without_source"
        elif self._looks_like_asset(legacy, str(text_value)):
            classification = "explicitly_excluded"
            reasons = ["renpy.asset_path"]
            structure_kind = "asset_literal"
        else:
            classification = "unknown"
            reasons = ["renpy.visibility_unknown"]
            structure_kind = "unknown_string_structure"

        source_marker_kind = "direct_source"
        source_text = str(text_value) if isinstance(text_value, str) else ""
        marker_line_number = 0
        if marker is not None:
            source_text = str(marker.get("text") or source_text)
            source_marker_kind = str(marker.get("kind") or source_marker_kind)
            marker_line_number = int(marker.get("line_index") or 0) + 1
            pair_reason = (
                "renpy.old_new_pair"
                if source_marker_kind == "old_new"
                else "renpy.translate_comment_pair"
            )
            if classification in {"translatable", "already_translated"}:
                reasons.append(pair_reason)
        elif (
            is_translation_file and stripped.startswith("new ") and classification == "translatable"
        ):
            reasons.append("renpy.catalog.missing_entry")

        if classification in {"translatable", "already_translated"}:
            item = dict(final_legacy_item or {})
            if not item:
                item = {
                    "id": identity or "",
                    "text": str(text_value),
                    "line": line_index,
                    "start": token.start[1],
                    "end": token.end[1],
                    "quote": quote,
                    "prefix": prefix,
                    "progress_entry": f"task:{line_index}:{token.start[1]}",
                    "block_name": block_name,
                    "block_index": ordinal,
                    "block_occurrence": block_occurrence,
                    "source_for_id": source_text,
                    "file_rel_path": document.file_rel_path,
                    "file_path": document.file_path,
                }
            unit_item = dict(item)
            unit_item["source"] = source_text
            if marker is not None:
                unit_item["current_translation"] = str(text_value)
            if speaker_id:
                unit_item["speaker_id"] = speaker_id
                unit_item["speaker"] = speaker_id
                if speaker_names.get(speaker_id):
                    unit_item["speaker_name"] = speaker_names[speaker_id]
            unit = translation_core.unit_from_sync_task(
                unit_item,
                file_rel_path=document.file_rel_path,
                file_path=document.file_path,
            )

        translation_scope, analysis_scope = _candidate_scopes(
            classification,
            structure_kind,
        )

        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "translate_block": block_name,
                "block_occurrence": block_occurrence,
                "ordinal": ordinal,
                "line_hint": line_index + 1,
                "start_col_hint": token.start[1],
                "end_col_hint": token.end[1],
                "source_marker_kind": source_marker_kind,
                "candidate_ordinal": candidate_ordinal,
            },
        )
        evidence: dict[str, Any] = {
            "literal": _bounded_excerpt(token.string),
            "speaker_id": speaker_id,
            "speaker_name": speaker_names.get(speaker_id, ""),
            "identity_v2": identity
            or (str(legacy_item.get("id") or "") if legacy_item is not None else ""),
            "context_before": (
                _bounded_excerpt(lines[line_index - 1].strip()) if line_index > 0 else ""
            ),
            "context_after": (
                _bounded_excerpt(lines[line_index + 1].strip())
                if line_index + 1 < len(lines)
                else ""
            ),
        }
        if marker is not None:
            evidence["source_marker_line"] = marker_line_number
            evidence["source_text"] = _bounded_excerpt(source_text)
        if literal_error is not None:
            evidence["parse_error"] = f"{type(literal_error).__name__}: {literal_error}"
        return Candidate(
            candidate_id=_candidate_id(project, locator),
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            locator=locator,
            raw_excerpt=_bounded_excerpt(line.strip()),
            structure_kind=structure_kind,
            classification=classification,
            reason_codes=tuple(dict.fromkeys(reasons)),
            translation_scope=translation_scope,
            analysis_scope=analysis_scope,
            catalog_link=(
                {
                    "format": "renpy_tl_rpy",
                    "file_rel_path": document.file_rel_path,
                    "source_marker_kind": source_marker_kind,
                }
                if marker is not None
                else None
            ),
            evidence=evidence,
            unit=unit,
            legacy_item=final_legacy_item,
        )

    @staticmethod
    def _looks_like_asset(legacy: ModuleType, text: str) -> bool:
        if legacy.CHARACTER_DISPLAY_ASSET_RE.match(text):
            return True
        return " " not in text and ("/" in text or "\\" in text)

    @staticmethod
    def _looks_like_nonstandard_source_comment(line: str) -> bool:
        stripped = str(line or "").strip()
        return stripped.startswith("#") and any(quote in stripped[1:] for quote in {'"', "'"})

    @staticmethod
    def _source_marker_evidence(
        legacy: ModuleType,
        lines: Sequence[str],
        line_index: int,
        *,
        is_translation_file: bool,
    ) -> Mapping[str, Any] | None:
        if not is_translation_file:
            return None
        for previous_index in range(line_index - 1, -1, -1):
            previous_line = lines[previous_index].strip()
            if not previous_line:
                continue
            comment_match = legacy.TL_COMMENT_SOURCE_RE.match(lines[previous_index].rstrip("\r\n"))
            if comment_match:
                if legacy.is_voice_comment_match(comment_match):
                    continue
                return {
                    "kind": "comment",
                    "line_index": previous_index,
                    "text": legacy.decode_string_literal_text(comment_match.group("text")),
                }
            old_match = legacy.TL_OLD_LINE_RE.match(lines[previous_index].rstrip("\r\n"))
            if old_match:
                return {
                    "kind": "old_new",
                    "line_index": previous_index,
                    "text": legacy.decode_string_literal_text(old_match.group("text")),
                }
            if legacy.is_voice_statement_line(previous_line):
                continue
            break
        return None

    def _parse_error_candidate(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        *,
        line_index: int,
        start_col: int,
        end_col: int,
        candidate_ordinal: int,
        block_name: str,
        block_occurrence: int,
        ordinal: int,
        reason_code: str,
        structure_kind: str,
        excerpt: str,
    ) -> Candidate:
        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "translate_block": block_name,
                "block_occurrence": block_occurrence,
                "ordinal": ordinal,
                "line_hint": line_index + 1,
                "start_col_hint": start_col,
                "end_col_hint": end_col,
                "source_marker_kind": "direct_source",
                "candidate_ordinal": candidate_ordinal,
                "error_region": structure_kind,
            },
        )
        return Candidate(
            candidate_id=_candidate_id(project, locator),
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            locator=locator,
            raw_excerpt=_bounded_excerpt(excerpt),
            structure_kind=structure_kind,
            classification="parse_error",
            reason_codes=(reason_code,),
            translation_scope="unknown",
            analysis_scope="unknown",
            evidence={},
        )

    def _marker_candidate(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        *,
        line_index: int,
        line: str,
        candidate_ordinal: int,
        reason_code: str,
        classification: str,
        structure_kind: str,
        source_marker_kind: str,
        block_name: str,
        block_occurrence: int,
    ) -> Candidate:
        quote_positions = [position for quote in {'"', "'"} if (position := line.find(quote)) >= 0]
        quote_start = min(quote_positions) if quote_positions else 0
        quote_character = line[quote_start] if quote_positions else ""
        quote_end = line.rfind(quote_character) + 1 if quote_character else 0
        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "translate_block": block_name,
                "block_occurrence": block_occurrence,
                "ordinal": candidate_ordinal,
                "line_hint": line_index + 1,
                "start_col_hint": max(0, quote_start),
                "end_col_hint": max(0, quote_end),
                "source_marker_kind": source_marker_kind,
                "candidate_ordinal": candidate_ordinal,
            },
        )
        translation_scope, analysis_scope = _candidate_scopes(
            classification,
            structure_kind,
        )
        return Candidate(
            candidate_id=_candidate_id(project, locator),
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            locator=locator,
            raw_excerpt=_bounded_excerpt(line.strip()),
            structure_kind=structure_kind,
            classification=classification,
            reason_codes=(reason_code,),
            translation_scope=translation_scope,
            analysis_scope=analysis_scope,
            evidence={},
        )

    def audit_extraction(
        self,
        project: ProjectDiscovery,
        inventory: CandidateInventory,
    ) -> CoverageReportDraft:
        source_changed = False
        for document in project.source_documents:
            try:
                live_digest = _sha256_bytes(Path(document.file_path).read_bytes())
            except OSError:
                source_changed = True
                continue
            if live_digest != document.sha256:
                source_changed = True

        provenance = dict(project.catalog_provenance)
        return CoverageReportDraft(
            source_fingerprint=project.source_fingerprint,
            reason_codes=("renpy.catalog.provenance_unknown",),
            catalog_provenance=provenance,
            catalog_freshness="unknown",
            source_changed_during_scan=source_changed,
        )

    def extract_occurrences(
        self,
        project: ProjectDiscovery,
        inventory: CandidateInventory,
        approved_candidate_ids: Sequence[str],
    ) -> Sequence[Occurrence]:
        candidates = inventory.by_id()
        documents = project.document_by_path()
        occurrences: list[Occurrence] = []
        seen: set[str] = set()
        for candidate_id in approved_candidate_ids:
            if candidate_id in seen:
                raise ValueError(f"Duplicate approved candidate_id: {candidate_id}")
            seen.add(candidate_id)
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ValueError(f"Unknown candidate_id: {candidate_id}")
            if candidate.classification not in {
                "translatable",
                "already_translated",
            }:
                raise ValueError(
                    "Candidate is not approved for occurrence extraction: "
                    f"{candidate_id} ({candidate.classification})"
                )
            if candidate.unit is None:
                raise ValueError(f"Candidate has no TranslationUnit: {candidate_id}")

            unit = candidate.unit
            document = documents.get(unit.file_rel_path)
            if document is None:
                raise ValueError(f"Candidate source document is missing: {unit.file_rel_path}")
            content_fingerprint = self._content_fingerprint(document, unit)
            occurrence_id = "occ1:" + digest_json(
                {
                    "engine": self.engine,
                    "project_snapshot_fingerprint": (project.project_snapshot_fingerprint),
                    "locator": candidate.locator.to_dict(),
                }
            )
            occurrences.append(
                Occurrence(
                    occurrence_id=occurrence_id,
                    engine=self.engine,
                    project_snapshot_fingerprint=(project.project_snapshot_fingerprint),
                    content_fingerprint=content_fingerprint,
                    candidate_id=candidate.candidate_id,
                    locator=candidate.locator,
                    unit=unit,
                )
            )
        return tuple(occurrences)

    @classmethod
    def _project_with_live_sources(
        cls,
        project: ProjectDiscovery,
        live_sources: Sequence[SourceDocument],
    ) -> ProjectDiscovery:
        documents = tuple(sorted(live_sources, key=lambda item: item.file_rel_path))
        source_fingerprint = source_snapshot_fingerprint(documents)
        project_snapshot_fingerprint = digest_json(
            {
                "engine": project.engine,
                "localization_mode": project.localization_mode.value,
                "target_language": project.target_language,
                "source_fingerprint": source_fingerprint,
            }
        )
        return replace(
            project,
            source_documents=documents,
            source_fingerprint=source_fingerprint,
            project_snapshot_fingerprint=project_snapshot_fingerprint,
        )

    @staticmethod
    def _content_fingerprint(document: SourceDocument, unit: translation_core.TranslationUnit) -> str:
        lines = document.lines()
        before = lines[unit.line - 1].strip() if unit.line > 0 else ""
        after = lines[unit.line + 1].strip() if unit.line + 1 < len(lines) else ""
        return digest_json(
            {
                "schema_version": CONTENT_FINGERPRINT_SCHEMA_VERSION,
                "source": _normalize_fingerprint_text(unit.source_text),
                "speaker_id": _normalize_fingerprint_text(unit.speaker_id),
                "speaker_name": _normalize_fingerprint_text(unit.speaker_name),
                "before": _normalize_fingerprint_text(before),
                "after": _normalize_fingerprint_text(after),
            }
        )

    @staticmethod
    def _token_counters(legacy: ModuleType, text: str) -> dict[str, Counter[str]]:
        return {
            "tag": Counter(legacy.RENPY_TAG_RE.findall(text or "")),
            "field": Counter(legacy.RENPY_FIELD_RE.findall(text or "")),
            "percent": Counter(legacy.PERCENT_FORMAT_TOKEN_RE.findall(text or "")),
        }

    @staticmethod
    def _counter_payload(counter: Counter[str]) -> dict[str, int]:
        return {key: counter[key] for key in sorted(counter)}

    @staticmethod
    def _render_literal(
        legacy: ModuleType,
        translated_text: str,
        prefix: str,
        quote: str,
    ) -> str:
        normalized = translated_text
        if getattr(legacy, "USE_TRANSLATION_MEMORY", False):
            normalized = legacy.apply_normalization(normalized)
        return legacy.quote_with(normalized, str(quote or '"'), prefix=prefix or "")

    @staticmethod
    def _literal_at_span(line: str, start: int, end: int) -> tuple[str, str] | None:
        try:
            tokens = tokenize.generate_tokens(io.StringIO(line).readline)
            for token in tokens:
                if token.type != tokenize.STRING:
                    continue
                if token.start[1] != start or token.end[1] != end:
                    continue
                value = ast.literal_eval(token.string)
                if isinstance(value, str):
                    return value, token.string
        except (IndentationError, SyntaxError, tokenize.TokenError, ValueError):
            return None
        return None

    @staticmethod
    def _relocation_score(original: Occurrence, candidate: Occurrence) -> int | None:
        original_unit = original.unit
        candidate_unit = candidate.unit
        if original_unit.file_rel_path != candidate_unit.file_rel_path:
            return None
        if _normalize_fingerprint_text(original_unit.source_text) != _normalize_fingerprint_text(
            candidate_unit.source_text
        ):
            return None
        score = 100
        original_locator = original.locator.locator
        candidate_locator = candidate.locator.locator
        if original_unit.file_rel_path == candidate_unit.file_rel_path:
            score += 25
        if original_unit.speaker_id == candidate_unit.speaker_id:
            score += 15
        if original_unit.speaker_name == candidate_unit.speaker_name:
            score += 5
        if original_locator.get("translate_block") == candidate_locator.get("translate_block"):
            score += 15
        if original_locator.get("block_occurrence") == candidate_locator.get("block_occurrence"):
            score += 10
        if original_locator.get("source_marker_kind") == candidate_locator.get("source_marker_kind"):
            score += 5
        if original_locator.get("ordinal") == candidate_locator.get("ordinal"):
            score += 5
        if original.content_fingerprint == candidate.content_fingerprint:
            score += 25
        return score

    def _validation_reason_codes(
        self,
        legacy: ModuleType,
        source_text: str,
        translated_text: str,
        message: str,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
        reason_codes: list[str] = []
        diagnostics: list[Mapping[str, Any]] = []
        if not str(translated_text or "").strip():
            reason_codes.append("common.translation.empty")
        missing_terms = getattr(legacy, "missing_preserved_terms", lambda *_: [])(
            source_text, translated_text
        )
        if missing_terms:
            reason_codes.append("common.preserve_term.missing")
            diagnostics.append({"code": "common.preserve_term.missing", "terms": list(missing_terms)})

        source_tokens = self._token_counters(legacy, source_text)
        translated_tokens = self._token_counters(legacy, translated_text)
        token_codes = {
            "tag": "renpy.tag.changed",
            "field": "renpy.field.changed",
            "percent": "renpy.percent_token.changed",
        }
        for kind, code in token_codes.items():
            missing = source_tokens[kind] - translated_tokens[kind]
            added = translated_tokens[kind] - source_tokens[kind]
            if missing or added:
                reason_codes.append(code)
                if kind == "tag":
                    if missing:
                        reason_codes.append("renpy.placeholder.missing")
                    if added:
                        reason_codes.append("renpy.placeholder.added")
                diagnostics.append(
                    {
                        "code": code,
                        "kind": kind,
                        "missing": self._counter_payload(missing),
                        "added": self._counter_payload(added),
                    }
                )
        if message == "No Chinese characters":
            reason_codes.append("common.target_language.missing")
        if not reason_codes and message and message != "OK":
            reason_codes.append("renpy.string_literal.unrenderable")
        if message and message != "OK":
            diagnostics.append({"message": message})
        return tuple(dict.fromkeys(reason_codes)), tuple(diagnostics)

    def relocate_occurrences(
        self,
        project: ProjectDiscovery,
        occurrences: Sequence[Occurrence],
        live_sources: Sequence[SourceDocument],
    ) -> RelocationResult:
        if project.engine != self.engine:
            raise ValueError(f"RenPyAdapter cannot relocate engine={project.engine!r}.")
        live_project = self._project_with_live_sources(project, live_sources)
        live_inventory = self.inventory_candidates(live_project, InventoryPolicy())
        approved_ids = [
            candidate.candidate_id
            for candidate in live_inventory.candidates
            if candidate.classification in {"translatable", "already_translated"}
        ]
        live_occurrences = tuple(
            self.extract_occurrences(live_project, live_inventory, approved_ids)
        )
        live_by_unit_id: dict[str, list[Occurrence]] = {}
        for candidate in live_occurrences:
            live_by_unit_id.setdefault(candidate.unit.id, []).append(candidate)

        relocated: list[Occurrence] = []
        unresolved: list[str] = []
        diagnostics: list[Mapping[str, Any]] = []
        used_live_ids: set[str] = set()
        for original in occurrences:
            if original.engine != self.engine:
                raise ValueError(f"RenPyAdapter cannot relocate engine={original.engine!r}.")
            exact_candidates = [
                candidate
                for candidate in live_by_unit_id.get(original.unit.id, [])
                if candidate.occurrence_id not in used_live_ids
            ]
            match = exact_candidates[0] if len(exact_candidates) == 1 else None
            match_kind = "identity_v2" if match is not None else "content_evidence"
            score = None
            if match is None:
                scored = []
                for candidate in live_occurrences:
                    if candidate.occurrence_id in used_live_ids:
                        continue
                    candidate_score = self._relocation_score(original, candidate)
                    if candidate_score is not None:
                        scored.append((candidate_score, candidate))
                scored.sort(key=lambda item: item[0], reverse=True)
                if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                    top_score, top_match = scored[0]
                    if top_score >= CONTENT_EVIDENCE_MIN_SCORE:
                        score, match = top_score, top_match
                    else:
                        diagnostics.append(
                            {
                                "occurrence_id": original.occurrence_id,
                                "reason_code": "common.locator.unresolved",
                                "status": "weak_content_evidence",
                                "score": top_score,
                                "min_score": CONTENT_EVIDENCE_MIN_SCORE,
                                "candidate_count": len(scored),
                            }
                        )
                elif scored:
                    diagnostics.append(
                        {
                            "occurrence_id": original.occurrence_id,
                            "reason_code": "common.locator.unresolved",
                            "status": "ambiguous",
                            "score": scored[0][0],
                            "candidate_count": len(scored),
                        }
                    )
            if match is None:
                unresolved.append(original.occurrence_id)
                if not any(item.get("occurrence_id") == original.occurrence_id for item in diagnostics):
                    diagnostics.append(
                        {
                            "occurrence_id": original.occurrence_id,
                            "reason_code": "common.locator.unresolved",
                            "status": "missing",
                        }
                    )
                continue
            used_live_ids.add(match.occurrence_id)
            updated_unit = replace(match.unit, id=original.unit.id, mode=original.unit.mode)
            relocated_occurrence_id = "occ1:" + digest_json(
                {
                    "engine": self.engine,
                    "project_snapshot_fingerprint": live_project.project_snapshot_fingerprint,
                    "locator": match.locator.to_dict(),
                }
            )
            relocated.append(
                replace(
                    match,
                    occurrence_id=relocated_occurrence_id,
                    project_snapshot_fingerprint=live_project.project_snapshot_fingerprint,
                    unit=updated_unit,
                )
            )
            diagnostics.append(
                {
                    "occurrence_id": original.occurrence_id,
                    "status": "relocated",
                    "match": match_kind,
                    "score": score,
                    "live_occurrence_id": relocated_occurrence_id,
                    "file_rel_path": updated_unit.file_rel_path,
                    "line": updated_unit.line,
                }
            )
        return RelocationResult(
            occurrences=tuple(relocated),
            unresolved_occurrence_ids=tuple(unresolved),
            diagnostics=tuple(diagnostics),
        )

    def validate_translation(
        self,
        occurrence: Occurrence,
        translated_text: str,
    ) -> ValidationResult:
        if occurrence.engine != self.engine:
            raise ValueError(f"RenPyAdapter cannot validate engine={occurrence.engine!r}.")
        legacy = self._legacy()
        source_text = occurrence.unit.source_text
        translated_text = str(translated_text or "")
        try:
            valid, message = legacy.validate_translation(source_text, translated_text)
        except Exception as exc:
            valid, message = False, f"Validation failed: {type(exc).__name__}: {exc}"
        reason_codes, diagnostics = self._validation_reason_codes(
            legacy, source_text, translated_text, str(message or "")
        )
        if valid:
            try:
                self._render_literal(
                    legacy, translated_text, occurrence.unit.prefix, occurrence.unit.quote
                )
            except Exception as exc:
                valid = False
                reason_codes = tuple(dict.fromkeys((*reason_codes, "renpy.string_literal.unrenderable")))
                diagnostics = (*diagnostics, {"code": "renpy.string_literal.unrenderable", "message": str(exc)})
        normalized = translated_text
        if getattr(legacy, "USE_TRANSLATION_MEMORY", False):
            normalized = legacy.apply_normalization(translated_text)
        source_constraints_digest = digest_json(
            {
                "source": source_text,
                "tokens": {
                    kind: self._counter_payload(counter)
                    for kind, counter in self._token_counters(legacy, source_text).items()
                },
            }
        )
        translation_digest = digest_json(
            {
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "translation": normalized,
            }
        )
        return ValidationResult(
            occurrence_id=occurrence.occurrence_id,
            engine=self.engine,
            status="pass" if valid else "block",
            reason_codes=tuple(reason_codes),
            diagnostics=tuple(diagnostics),
            source_constraints_digest=source_constraints_digest,
            translation_digest=translation_digest,
            normalized_translation=normalized if normalized != translated_text else None,
        )

    def build_writeback_plan(
        self,
        project: ProjectDiscovery,
        validated: Sequence[ValidatedTranslation],
        live_sources: Sequence[SourceDocument],
    ) -> WritebackPlan:
        if project.engine != self.engine:
            raise ValueError(f"RenPyAdapter cannot build a plan for engine={project.engine!r}.")
        documents = {document.file_rel_path: document for document in live_sources}
        live_source_fingerprint = source_snapshot_fingerprint(live_sources)
        operations: list[WritebackOperation] = []
        spans: list[tuple[str, int, int, int]] = []
        legacy = self._legacy()
        for item in validated:
            occurrence = item.occurrence
            if occurrence.engine != self.engine:
                raise ValueError(f"Unsupported occurrence engine: {occurrence.engine!r}")
            if item.validation.status != "pass":
                raise ValueError(
                    "Cannot build writeback plan for non-pass validation: "
                    + ",".join(item.validation.reason_codes)
                )
            unit = occurrence.unit
            rel_path = _normalize_rel_path(unit.file_rel_path)
            document = documents.get(rel_path)
            if document is None:
                raise ValueError(f"Writeback source document missing: {rel_path}")
            lines = document.lines()
            if unit.line < 0 or unit.line >= len(lines):
                raise ValueError(f"Writeback source line missing: {rel_path}:{unit.line}")
            if unit.start < 0 or unit.end > len(lines[unit.line]) or unit.start >= unit.end:
                raise ValueError(f"Writeback span invalid: {rel_path}:{unit.line}:{unit.start}-{unit.end}")
            raw_fragment = lines[unit.line][unit.start:unit.end]
            literal = self._literal_at_span(lines[unit.line], unit.start, unit.end)
            if literal is None or literal[0] != unit.text:
                raise ValueError(f"Writeback span/source mismatch: {rel_path}:{unit.line}")
            span = (rel_path, unit.line, unit.start, unit.end)
            if any(
                existing[0] == rel_path
                and existing[1] == unit.line
                and max(existing[2], unit.start) < min(existing[3], unit.end)
                for existing in spans
            ):
                raise ValueError(f"Overlapping writeback span: {rel_path}:{unit.line}")
            spans.append(span)
            replacement_fragment = self._render_literal(
                legacy, item.translated_text, unit.prefix, unit.quote
            )
            operation_payload = {
                "kind": "text_span_replace",
                "occurrence_id": occurrence.occurrence_id,
                "target_root": "localization_catalog",
                "target_rel_path": rel_path,
                "expected_file_sha256": document.sha256,
                "line": unit.line,
                "start_col": unit.start,
                "end_col": unit.end,
                "expected_fragment_sha256": _sha256_bytes(raw_fragment.encode("utf-8")),
                "expected_text_digest": _sha256_bytes(unit.text.encode("utf-8")),
                "replacement_fragment": replacement_fragment,
                "validation_digest": digest_json(item.validation.to_dict()),
            }
            operations.append(
                WritebackOperation(
                    operation_id="op1:" + digest_json(operation_payload),
                    **operation_payload,
                )
            )
        operations.sort(
            key=lambda operation: (
                operation.target_rel_path,
                operation.line,
                operation.start_col,
                operation.operation_id,
            )
        )
        project_identity_digest = digest_json(
            {
                "engine": project.engine,
                "target_language": project.target_language,
                "localization_mode": project.localization_mode.value,
                "localization_root": os.path.realpath(project.localization_root),
            }
        )
        coverage_digest = str(
            project.coverage_digest or project.catalog_provenance.get("coverage_digest") or ""
        )
        coverage_review_digest = str(
            project.coverage_review_digest
            or project.catalog_provenance.get("coverage_review_digest")
            or ""
        )
        plan_payload = {
            "writeback_plan_schema_version": WRITEBACK_PLAN_SCHEMA_VERSION,
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "project_identity_digest": project_identity_digest,
            "source_snapshot_fingerprint": live_source_fingerprint,
            "coverage_digest": coverage_digest,
            "coverage_review_digest": coverage_review_digest,
            "operations": [operation.to_dict() for operation in operations],
        }
        return WritebackPlan(
            engine=self.engine,
            adapter_version=self.adapter_version,
            project_identity_digest=project_identity_digest,
            source_snapshot_fingerprint=live_source_fingerprint,
            coverage_digest=coverage_digest,
            coverage_review_digest=coverage_review_digest,
            operations=tuple(operations),
            plan_digest=digest_json(plan_payload),
        )


def build_translation_snapshot(
    adapter: RenPyAdapter,
    request: ProjectDiscoveryRequest,
    policy: InventoryPolicy | None = None,
    *,
    include_occurrences: bool = True,
    include_task_payloads: bool = True,
) -> RenPyTranslationSnapshot:
    """Run the P1 read-only pipeline once for sync or Batch translation build.

    ``include_occurrences`` defaults to True for writeback/build consumers.
    Progress-only callers (environment check pending/translated counts) may set
    it to False: pending tasks and per-file progress still come from inventory,
    but occurrence extraction is skipped.

    ``include_task_payloads`` controls whether per-task payload dictionaries are
    materialized into ``pending_tasks_by_file``. Progress-only callers may set
    it to False to avoid copying every pending task; counts can be derived from
    the candidate inventory instead.
    """
    inventory_policy = policy or InventoryPolicy()
    project = adapter.discover_project(request)
    inventory = adapter.inventory_candidates(project, inventory_policy)
    draft = adapter.audit_extraction(project, inventory)
    report = build_coverage_report(
        project,
        inventory,
        draft,
        adapter_behavior_digest=adapter.behavior_digest(),
    )
    project = replace(project, coverage_digest=report.coverage_digest)
    if include_occurrences:
        approved_ids = [
            candidate.candidate_id
            for candidate in inventory.candidates
            if candidate.classification
            in {
                "translatable",
                "already_translated",
            }
        ]
        occurrences = tuple(
            adapter.extract_occurrences(
                project,
                inventory,
                approved_ids,
            )
        )
    else:
        occurrences = ()

    if include_task_payloads:
        pending_tasks: dict[str, list[Mapping[str, Any]]] = {
            document.file_rel_path: [] for document in project.source_documents
        }
        for candidate in inventory.candidates:
            if candidate.classification == "translatable" and candidate.legacy_item is not None:
                rel_path = str(candidate.locator.locator.get("file_rel_path") or "")
                pending_tasks.setdefault(rel_path, []).append(dict(candidate.legacy_item))
    else:
        pending_tasks = {}
    progress_by_file = {
        str(entry.get("file_rel_path") or ""): {
            "translated_count": int(entry.get("translated_count") or 0)
        }
        for entry in inventory.files_scanned
    }
    return RenPyTranslationSnapshot(
        project=project,
        inventory=inventory,
        report=report,
        occurrences=occurrences,
        pending_tasks_by_file={rel_path: tuple(tasks) for rel_path, tasks in pending_tasks.items()},
        progress_by_file=progress_by_file,
        review_policy=inventory_policy.review_policy,
    )


def iter_pending_tasks(
    snapshot: RenPyTranslationSnapshot,
) -> Iterable[Mapping[str, Any]]:
    for document in snapshot.project.source_documents:
        yield from snapshot.pending_tasks_by_file.get(
            document.file_rel_path,
            (),
        )
