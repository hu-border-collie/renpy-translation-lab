# -*- coding: utf-8 -*-
"""Read-only TyranoScript V600+ adapter for project discovery, inventory,
coverage audit, and occurrence extraction (#265 P5 / #399).

This module intentionally implements only the read side of the common
``EngineAdapter`` contract for now.  Writeback for Tyrano native
localization catalogs needs a new common operation kind and will be added in
a later PR; the adapter therefore fails closed for relocation, validation,
and writeback planning unless/until those operations are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import translation_core

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CONTENT_FINGERPRINT_SCHEMA_VERSION,
    ENGINE_ADAPTER_PROTOCOL_VERSION,
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
    WritebackPlan,
)
from .coverage import (
    REVIEW_POLICIES,
    digest_json,
)


ADAPTER_VERSION = "0.1.0"
LOCATOR_SCHEMA_VERSION = 1
DEFAULT_TAG_REGISTRY = {"glink": ("text",), "ptext": ("text",)}
SCENARIO_DIR = "data/scenario"
CATALOG_DIR = "data/others/lang"
CONFIG_REL_PATH = "data/system/Config.tjs"


@dataclass(frozen=True)
class TyranoNode:
    node_index: int
    line: int
    name: str
    pm: Mapping[str, Any]
    val: str = ""
    kind: str = "tag"
    in_iscript: bool = False


@dataclass(frozen=True)
class TyranoParseResult:
    nodes: tuple[TyranoNode, ...]
    comment_line_indexes: tuple[int, ...]
    parse_errors: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TyranoTranslationSnapshot:
    """One immutable discovery/inventory/extraction pass for P5 consumers."""

    project: ProjectDiscovery
    inventory: CandidateInventory
    report: object
    occurrences: tuple[Occurrence, ...]

    @property
    def source_document_count(self) -> int:
        return len(self.project.source_documents)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _bounded_excerpt(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _candidate_id(project: ProjectDiscovery, locator: OpaqueLocator) -> str:
    return "cand1:" + digest_json(
        {
            "engine": locator.engine,
            "project_snapshot_fingerprint": project.project_snapshot_fingerprint,
            "locator": locator.to_dict(),
        }
    )


def _occurrence_id(project: ProjectDiscovery, locator: OpaqueLocator) -> str:
    return "occ1:" + digest_json(
        {
            "engine": "tyrano",
            "project_snapshot_fingerprint": project.project_snapshot_fingerprint,
            "locator": locator.to_dict(),
        }
    )


def _normalize_rel_path(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Path escapes the project root: {value}")
    return "/".join(parts)


def _scenario_key(file_rel_path: str) -> str:
    rel = str(file_rel_path or "").replace("\\", "/")
    prefix = f"{SCENARIO_DIR}/"
    if rel.startswith(prefix):
        return rel[len(prefix):]
    return rel


def _scope_for(classification: str, structure_kind: str) -> tuple[str, str]:
    if classification in {"translatable", "already_translated"}:
        analysis_scope = "include" if structure_kind == "text" else "exclude"
        return "include", analysis_scope
    if classification == "explicitly_excluded":
        return "exclude", "exclude"
    return "unknown", "unknown"


def _read_keep_space_setting(project_root: Path) -> str:
    config_path = project_root / CONFIG_REL_PATH
    if not config_path.is_file():
        return "2"
    text = config_path.read_text(encoding="utf-8-sig", errors="replace")
    match = re.search(r";KeepSpaceInParameterValue\s*=\s*([123])\s*;", text)
    if not match:
        return "2"
    return match.group(1)


def parse_tyrano_scenario(
    lines: Sequence[str],
    *,
    keep_space: str = "2",
) -> TyranoParseResult:
    """Parse one .ks scenario with the official V600+ line/character scanner.

    The scanner intentionally mirrors ``kag.parser.parseScenario`` so the
    normalized source values equal the keys used by ``convertLang``.  Unlike
    the official parser, malformed quotes and unclosed inline tags are also
    surfaced in ``parse_errors`` instead of being silently compensated away.
    """
    nodes: list[TyranoNode] = []
    comment_lines: list[int] = []
    parse_errors: list[dict[str, Any]] = []
    flag_script = False
    in_block_comment = False

    def raise_node_error(
        line_index: int,
        reason_code: str,
        struct_kind: str,
        excerpt: str,
        *,
        extra_error: bool = False,
    ) -> None:
        parse_errors.append(
            {
                "line": line_index,
                "reason_code": reason_code,
                "structure_kind": struct_kind,
                "excerpt": _bounded_excerpt(excerpt),
            }
        )

    for line_index, raw_line in enumerate(lines):
        line = str(raw_line).replace("\r", "")
        line_str = line.strip()
        stripped = line_str

        # The official parser clears the iscript flag whenever a line contains
        # ``endscript``, even before handling comments.  Mirror that behaviour.
        if "endscript" in stripped:
            flag_script = False

        if in_block_comment:
            comment_lines.append(line_index)
            if stripped == "*/":
                in_block_comment = False
            continue
        if stripped == "/*":
            comment_lines.append(line_index)
            in_block_comment = True
            continue
        if stripped.startswith(";"):
            comment_lines.append(line_index)
            continue
        if stripped == "":
            continue

        first_char = stripped[0]
        if first_char == "*":
            label_tmp = stripped[1:].split("|")
            label_key = label_tmp[0].strip()
            label_val = label_tmp[1].strip() if len(label_tmp) > 1 else ""
            nodes.append(
                TyranoNode(
                    node_index=len(nodes),
                    line=line_index,
                    name="label",
                    pm={"line": line_index, "index": len(nodes), "label_name": label_key, "val": label_val},
                    val=label_val,
                    kind="label",
                )
            )
            continue

        if first_char == "#":
            tmp_line = stripped[1:].strip()
            if ":" in tmp_line:
                chara_name, chara_face = tmp_line.split(":", 1)
            else:
                chara_name, chara_face = tmp_line, ""
            nodes.append(
                TyranoNode(
                    node_index=len(nodes),
                    line=line_index,
                    name="chara_ptext",
                    pm={"name": chara_name.strip(), "face": chara_face.strip()},
                    val="",
                    kind="chara_ptext",
                )
            )
            continue

        scan_text: str
        if first_char == "@":
            scan_text = stripped[1:].strip()
            tag_str = scan_text
            node, tag_error = _make_tag_node(
                tag_str,
                line_index=line_index,
                node_index=len(nodes),
                keep_space=keep_space,
            )
            flag_script = _script_flag_after_node(flag_script, node.name)
            nodes.append(node)
            if tag_error:
                raise_node_error(
                    line_index,
                    tag_error["reason_code"],
                    tag_error["structure_kind"],
                    tag_error["excerpt"],
                )
            continue

        if first_char == "_":
            scan_text = stripped[1:]
        else:
            scan_text = stripped

        text = ""
        tag_buf = ""
        scanning_tag = False
        deep_kakko = 0
        start_quot = ""
        flag_escape = False
        tag_line_text = scan_text

        for char in scan_text:
            if flag_script:
                text += char
            elif scanning_tag:
                if char == "]":
                    if start_quot != "":
                        tag_buf += char
                    else:
                        deep_kakko -= 1
                        if deep_kakko == 0:
                            scanning_tag = False
                            node, tag_error = _make_tag_node(
                                tag_buf,
                                line_index=line_index,
                                node_index=len(nodes),
                                keep_space=keep_space,
                            )
                            flag_script = _script_flag_after_node(flag_script, node.name)
                            nodes.append(node)
                            if tag_error:
                                raise_node_error(
                                    line_index,
                                    tag_error["reason_code"],
                                    tag_error["structure_kind"],
                                    tag_error["excerpt"],
                                )
                            tag_buf = ""
                            start_quot = ""
                        else:
                            tag_buf += char
                elif char == "[":
                    if start_quot == "":
                        deep_kakko += 1
                    tag_buf += char
                elif char in {'"', "'", "`"}:
                    if start_quot == char:
                        start_quot = ""
                    elif start_quot == "":
                        start_quot = char
                    tag_buf += char
                else:
                    tag_buf += char
            elif flag_escape:
                text += char
                flag_escape = False
            elif char == "[":
                scanning_tag = True
                deep_kakko = 1
                if text != "":
                    nodes.append(
                        TyranoNode(
                            node_index=len(nodes),
                            line=line_index,
                            name="text",
                            pm={"val": text},
                            val=text,
                            kind="text",
                            in_iscript=flag_script,
                        )
                    )
                    text = ""
            elif char == "\\":
                flag_escape = True
            else:
                text += char

        if tag_buf:
            # Official parser's two silent compensation paths.  An unclosed
            # inline tag (no final ``]``) surfaces only after the official
            # compensation pass; quote defects are detected by
            # ``_parse_tag_parameters`` so escaped quotes such as ``It\'s``
            # are handled exactly like the official scanner.
            if not tag_buf.endswith("]"):
                raise_node_error(
                    line_index,
                    "tyrano.unclosed_inline_tag",
                    "inline_tag",
                    tag_line_text,
                )
            else:
                tag_buf = tag_buf[:-1]
            node, tag_error = _make_tag_node(
                tag_buf,
                line_index=line_index,
                node_index=len(nodes),
                keep_space=keep_space,
            )
            flag_script = _script_flag_after_node(flag_script, node.name)
            if tag_error:
                raise_node_error(
                    line_index,
                    tag_error["reason_code"],
                    tag_error["structure_kind"],
                    tag_error["excerpt"],
                )
            nodes.append(node)

        if text != "":
            nodes.append(
                TyranoNode(
                    node_index=len(nodes),
                    line=line_index,
                    name="text",
                    pm={"val": text},
                    val=text,
                    kind="text",
                    in_iscript=flag_script,
                )
            )

    return TyranoParseResult(
        nodes=tuple(nodes),
        comment_line_indexes=tuple(comment_lines),
        parse_errors=tuple(parse_errors),
    )


def _script_flag_after_node(flag_script: bool, node_name: str) -> bool:
    if node_name == "iscript":
        return True
    if node_name == "endscript":
        return False
    return flag_script


def _make_tag_node(
    tag_str: str,
    *,
    line_index: int,
    node_index: int,
    keep_space: str,
) -> tuple[TyranoNode, Mapping[str, Any] | None]:
    """Build one ``TyranoNode`` for a bracket tag string.

    Returns an optional error map.  The malformed line's tag is parsed anyway
    with the official scanner's normalization so the candidate inventory keeps
    a reviewable source value.
    """
    name, pm, error = _parse_tag_parameters(tag_str, keep_space=keep_space)
    if name == "iscript":
        pm = dict(pm)
    return (
        TyranoNode(
            node_index=node_index,
            line=line_index,
            name=name,
            pm=pm,
            val="",
            kind="tag",
        ),
        error,
    )


def _parse_tag_parameters(
    tag_str: str,
    *,
    keep_space: str,
) -> tuple[str, dict[str, str], Mapping[str, Any] | None]:
    """Parse tag name and ``key=value`` parameters like ``kag.parser.makeTag``.

    The official parser silently assigns empty strings to malformed tokens.
    The strict adapter returns an error map in addition to the normalized
    value when a quote is left open or a bare unquoted token follows another
    value.
    """
    trimmed = tag_str.strip()
    pm: dict[str, str] = {}
    error: dict[str, Any] | None = None
    index = 0
    length = len(trimmed)

    while index < length and trimmed[index] == " ":
        index += 1
    name_start = index
    while index < length and trimmed[index] != " ":
        index += 1
    name = trimmed[name_start:index]

    while index < length:
        while index < length and trimmed[index] == " ":
            index += 1
        if index >= length:
            break
        param_name_start = index
        while index < length and trimmed[index] not in {" ", "="}:
            index += 1
        if index >= length:
            pm[trimmed[param_name_start:index].strip()] = ""
            break
        param_name = trimmed[param_name_start:index].strip()
        if trimmed[index] == " ":
            # Skip blanks between parameter name and ``=`` (official scanner
            # stays in ``SCANNING_EQUAL`` until it sees ``=`` or a new token).
            lookahead = index
            while lookahead < length and trimmed[lookahead] == " ":
                lookahead += 1
            if lookahead < length and trimmed[lookahead] == "=":
                index = lookahead
                if param_name and param_name not in pm:
                    error = _set_error(
                        error,
                        "tyrano.unquoted_parameter_sequence",
                        "tag",
                        trimmed,
                    )
                param_name = ""
                continue
            elif lookahead < length:
                # Bare unquoted token following another parameter: official
                # parser creates an empty entry keyed by the bare token.
                if param_name and param_name not in pm:
                    error = _set_error(
                        error,
                        "tyrano.unquoted_parameter_sequence",
                        "tag",
                        trimmed,
                    )
                index = lookahead
                continue
            break

        # trimmed[index] == '='
        index += 1
        while index < length and trimmed[index] == " ":
            index += 1
        if index >= length:
            pm[param_name] = ""
            break

        if trimmed[index] in {'"', "'", "`"}:
            quote = trimmed[index]
            index += 1
            value_start = index
            value_chars: list[str] = []
            closed = False
            escaped = False
            while index < length:
                char = trimmed[index]
                index += 1
                if escaped:
                    value_chars.append(char)
                    escaped = False
                elif char == "\\":
                    # Official makeTag records the escaped character only;
                    # the backslash itself is not part of the catalog key.
                    escaped = True
                elif char == quote:
                    closed = True
                    break
                else:
                    value_chars.append(char)
            if not closed:
                error = _set_error(
                    error,
                    "tyrano.unterminated_quoted_parameter",
                    "tag",
                    trimmed,
                )
                # Official parser reads to end and then trims the partial value.
                raw_value = "".join(value_chars)
            else:
                raw_value = "".join(value_chars)
            pm[param_name] = _normalize_tag_value(raw_value, keep_space)
        else:
            value_start = index
            while index < length and trimmed[index] != " ":
                index += 1
            raw_value = trimmed[value_start:index]
            pm[param_name] = _normalize_tag_value(raw_value, keep_space)

    if name == "":
        error = _set_error(error, "tyrano.unclosed_inline_tag", "inline_tag", trimmed)
    return name, pm, error


def _set_error(
    current: Mapping[str, Any] | None,
    reason_code: str,
    structure_kind: str,
    excerpt: str,
) -> Mapping[str, Any]:
    if current is not None:
        return current
    return {
        "reason_code": reason_code,
        "structure_kind": structure_kind,
        "excerpt": _bounded_excerpt(excerpt),
    }


def _normalize_tag_value(raw_value: str, keep_space: str) -> str:
    value = raw_value
    if keep_space == "1":
        value = value.replace(" ", "")
    if keep_space != "3":
        value = value.strip()
    if value == "undefined":
        return ""
    return value


class TyranoAdapter:
    """Read-only TyranoScript V600+ adapter.

    Source inventory is a strict superset of the official parser: the same
    dialogue / tag / character candidates are produced, and malformed tags are
    inventoried as ``parse_error`` instead of being silently compensated.
    Catalog provenance is checked against the live language JSON during audit.
    """

    protocol_version = ENGINE_ADAPTER_PROTOCOL_VERSION
    engine = "tyrano"
    adapter_version = ADAPTER_VERSION
    locator_schema_version = LOCATOR_SCHEMA_VERSION

    def __init__(self) -> None:
        self._catalog_cache: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._catalog_cache_order: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # EngineAdapter discovery / inventory / audit / extract
    # ------------------------------------------------------------------

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.engine,
            adapter_version=self.adapter_version,
            supported_localization_modes=(LocalizationMode.HYBRID,),
            selected_localization_mode=LocalizationMode.HYBRID,
            source_inventory=True,
            native_catalog=True,
            relocation=False,
            declarative_writeback=(),
            native_catalog_required_for_writeback=True,
        )

    def behavior_digest(self) -> str:
        return digest_json(
            {
                "engine_adapter_protocol_version": self.protocol_version,
                "engine": self.engine,
                "adapter_version": self.adapter_version,
                "locator_schema_version": self.locator_schema_version,
                "read_only": True,
                "source_inventory": True,
                "native_catalog": True,
                "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                "content_fingerprint_schema_version": CONTENT_FINGERPRINT_SCHEMA_VERSION,
            }
        )

    def discover_project(
        self,
        request: ProjectDiscoveryRequest,
    ) -> ProjectDiscovery:
        if str(request.project_root or "").strip() == "":
            raise ValueError("Tyrano project discovery requires project_root.")
        project_root = os.path.abspath(os.fspath(request.project_root))
        target_language = str(request.target_language or "").strip()
        if not target_language:
            raise ValueError("Tyrano project discovery requires target_language.")

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

        project_root_path = Path(project_root)
        scenario_dir = project_root_path / SCENARIO_DIR
        if not scenario_dir.is_dir():
            raise ValueError(f"Tyrano scenario directory does not exist: {scenario_dir}")

        root_real = os.path.realpath(project_root)
        documents: list[SourceDocument] = []
        for dir_path, dir_names, file_names in os.walk(scenario_dir):
            dir_names.sort()
            for file_name in sorted(file_names):
                if not file_name.lower().endswith(".ks"):
                    continue
                file_path = os.path.abspath(os.path.join(dir_path, file_name))
                file_real = os.path.realpath(file_path)
                try:
                    within_root = os.path.commonpath([root_real, file_real]) == root_real
                except ValueError:
                    within_root = False
                if not within_root:
                    raise ValueError(f"Tyrano source resolves outside project root: {file_path}")
                rel_path = _normalize_rel_path(os.path.relpath(file_path, project_root))
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

        # Include the parser configuration so source_fingerprint changes when
        # KeepSpaceInParameterValue changes.  It is not a text candidate source,
        # but it is part of the source snapshot.
        config_path = project_root_path / CONFIG_REL_PATH
        if config_path.is_file():
            config_content = config_path.read_bytes()
            documents.append(
                SourceDocument(
                    file_rel_path="data/system/Config.tjs",
                    file_path=str(config_path),
                    size=len(config_content),
                    sha256=_sha256_bytes(config_content),
                    content=config_content,
                )
            )
        documents.sort(key=lambda item: item.file_rel_path)

        source_payload = [document.manifest_entry() for document in documents]
        source_fingerprint = digest_json(source_payload)
        project_snapshot_fingerprint = digest_json(
            {
                "engine": self.engine,
                "localization_mode": LocalizationMode.HYBRID.value,
                "target_language": target_language,
                "source_fingerprint": source_fingerprint,
            }
        )

        catalog_path = self._catalog_path(project_root_path, target_language)
        catalog_data: Mapping[str, Any] = {}
        catalog_sha256 = ""
        try:
            catalog_bytes = catalog_path.read_bytes()
            catalog_sha256 = _sha256_bytes(catalog_bytes)
            catalog_data = json.loads(catalog_bytes.decode("utf-8-sig"))
        except OSError:
            catalog_data = {}
        except json.JSONDecodeError:
            catalog_data = {}

        catalog_provenance = {
            "format": "tyrano_lang_json",
            "target_language": target_language,
            "catalog_rel_path": f"data/others/lang/{target_language}.json",
            "catalog_sha256": catalog_sha256,
            "catalog_file_exists": bool(catalog_path.is_file()),
            "recorded_source_fingerprint": "",
            "live_source_fingerprint": source_fingerprint,
            "provenance_status": "unknown",
            "generator": "",
            "engine_version": "",
            "generated_at": "",
        }
        self._cache_catalog(
            source_fingerprint,
            target_language,
            catalog_data,
        )
        return ProjectDiscovery(
            engine=self.engine,
            adapter_version=self.adapter_version,
            project_root=project_root,
            localization_root=str(catalog_path.parent),
            target_language=target_language,
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
            raise ValueError(f"TyranoAdapter cannot inventory engine={project.engine!r}.")
        if policy.review_policy not in REVIEW_POLICIES:
            raise ValueError(f"Unsupported coverage review policy: {policy.review_policy}")

        keep_space = _read_keep_space_setting(Path(project.project_root))
        catalog_data = self._load_catalog_data(project)

        candidates: list[Candidate] = []
        file_entries: list[Mapping[str, Any]] = []
        for document in project.source_documents:
            if not document.file_rel_path.endswith(".ks"):
                # Source file scans include Config.tjs for fingerprint stability;
                # it has no translatable candidates and should still be listed.
                file_entries.append(
                    {
                        **document.manifest_entry(),
                        "candidate_count": 0,
                        "pending_task_count": 0,
                        "translated_count": 0,
                        "parse_error_count": 0,
                    }
                )
                continue
            document_candidates = self._inventory_document(
                project,
                document,
                keep_space=keep_space,
                catalog_data=catalog_data,
            )
            candidates.extend(document_candidates)
            file_entries.append(
                {
                    **document.manifest_entry(),
                    "candidate_count": len(document_candidates),
                    "pending_task_count": sum(
                        candidate.classification == "translatable"
                        for candidate in document_candidates
                    ),
                    "translated_count": sum(
                        candidate.classification == "already_translated"
                        for candidate in document_candidates
                    ),
                    "parse_error_count": sum(
                        candidate.classification == "parse_error"
                        for candidate in document_candidates
                    ),
                }
            )

        return CandidateInventory(
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            project_snapshot_fingerprint=project.project_snapshot_fingerprint,
            candidates=tuple(candidates),
            files_scanned=tuple(file_entries),
        )

    # ------------------------------------------------------------------
    # Candidate inventory details
    # ------------------------------------------------------------------

    def _inventory_document(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        *,
        keep_space: str,
        catalog_data: Mapping[str, Any],
    ) -> list[Candidate]:
        lines = list(document.text().splitlines())
        parse_result = parse_tyrano_scenario(
            lines,
            keep_space=keep_space,
        )
        catalog = self._catalog_for_scenario(document, catalog_data)
        candidates: list[Candidate] = []
        line_errors: dict[int, list[str]] = {}
        for error in parse_result.parse_errors:
            line_index = int(error["line"])
            reasons = line_errors.setdefault(line_index, [])
            reason_code = str(error["reason_code"])
            if reason_code not in reasons:
                reasons.append(reason_code)

        for node in parse_result.nodes:
            candidates.append(
                self._candidate_for_node(
                    project,
                    document,
                    node,
                    line=node.line,
                    catalog=catalog,
                    catalog_data=catalog_data,
                    line_error_reasons=line_errors.get(node.line),
                )
            )

        for comment_line in parse_result.comment_line_indexes:
            candidates.append(
                self._comment_candidate(project, document, comment_line, lines[comment_line])
            )

        for line_index, reasons in line_errors.items():
            if any(
                int(candidate.locator.locator.get("line") or 0) == line_index
                and candidate.classification == "parse_error"
                for candidate in candidates
            ):
                continue
            raw_line = lines[line_index] if line_index < len(lines) else ""
            candidates.append(
                self._parse_error_candidate(
                    project,
                    document,
                    line_index=line_index,
                    excerpt=raw_line.strip(),
                    reason_code=reasons[0],
                )
            )

        candidates.sort(key=lambda candidate: (candidate.locator.locator["line"], str(candidate.locator.locator.get("node_index") or 0)))
        return candidates

    def _candidate_for_node(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        node: TyranoNode,
        *,
        line: int,
        catalog: Mapping[str, Any],
        catalog_data: Mapping[str, Any],
        line_error_reasons: Sequence[str] | None = None,
    ) -> Candidate:
        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "scenario": _scenario_key(document.file_rel_path),
                "line": line,
                "node_index": node.node_index,
                "kind": node.kind,
                "name": node.name,
            },
        )
        if line_error_reasons:
            normalized_value = ""
            if node.kind == "text":
                normalized_value = str(node.pm.get("val") or "")
            elif node.kind == "tag":
                normalized_value = _first_non_empty_pm(node.pm)
            reasons = list(line_error_reasons)
            if (
                node.kind == "tag"
                and node.name not in DEFAULT_TAG_REGISTRY
                and node.name not in {"iscript", "endscript"}
                and "tyrano.unregistered_macro_invocation" not in reasons
            ):
                reasons.append("tyrano.unregistered_macro_invocation")
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="parse_error_region",
                classification="parse_error",
                reason_codes=tuple(reasons),
                source_value=normalized_value,
                raw_excerpt=normalized_value,
                evidence={"parser_name": node.name},
            )

        if node.kind == "label":
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="label",
                classification="explicitly_excluded",
                reason_codes=("tyrano.engine_control_structure",),
                source_value=node.val,
                raw_excerpt=node.val,
                evidence={"parser_name": "label"},
            )
        if node.kind == "chara_ptext":
            chara_name = str(node.pm.get("name") or "")
            if chara_name.startswith("&"):
                return self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind="chara_ptext",
                    classification="unsupported",
                    reason_codes=("tyrano.dynamic_parameter_expression",),
                    source_value=chara_name,
                    raw_excerpt=chara_name,
                    evidence={"parser_name": "chara_ptext", "param_name": "name", "dynamic_expression": True},
                )
            target = catalog.get("charas", {}).get(chara_name)
            classification = _translation_classification(target)
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="chara_ptext",
                classification=classification,
                reason_codes=("tyrano.chara_ptext",),
                source_value=chara_name,
                raw_excerpt=chara_name,
                catalog_link=({"path": ["charas", chara_name], "translation": target} if classification == "already_translated" else None),
                evidence={"parser_name": "chara_ptext", "param_name": "name"},
                existing_translation=target,
            )

        if node.kind == "text":
            source_value = str(node.pm.get("val") or "")
            target = catalog.get("scenario", {}).get(source_value)
            if getattr(node, "in_iscript", False):
                return self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind="text",
                    classification="explicitly_excluded",
                    reason_codes=("tyrano.iscript_content",),
                    source_value=source_value,
                    raw_excerpt=source_value,
                    evidence={"parser_name": "text", "inside_iscript": True},
                )
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="text",
                classification=_translation_classification(target),
                reason_codes=("tyrano.text_node",),
                source_value=source_value,
                raw_excerpt=source_value,
                catalog_link=({"path": ["scenes", _scenario_key(document.file_rel_path), "scenario", source_value], "translation": target} if target else None),
                evidence={"parser_name": "text"},
                existing_translation=target,
            )

        # tag node
        tag_name = node.name
        tag_registry = catalog_data.get("tags", DEFAULT_TAG_REGISTRY)
        structural_tags = {
            "lang_set": ("tyrano.lang_set_control_tag", "explicitly_excluded"),
            "iscript": ("tyrano.iscript_boundary_tag", "explicitly_excluded"),
            "endscript": ("tyrano.iscript_boundary_tag", "explicitly_excluded"),
            "chara_new": ("tyrano.character_definition", "explicitly_excluded"),
            "nw": ("tyrano.engine_control_structure", "explicitly_excluded"),
            "l": ("tyrano.engine_control_structure", "explicitly_excluded"),
            "r": ("tyrano.engine_control_structure", "explicitly_excluded"),
            "p": ("tyrano.engine_control_structure", "explicitly_excluded"),
        }
        if tag_name in structural_tags:
            reason, classification = structural_tags[tag_name]
            source_value = ""
            if tag_name == "lang_set":
                source_value = str(node.pm.get("name") or "")
            elif tag_name == "chara_new":
                source_value = str(node.pm.get("jname") or "")
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="tag",
                classification=classification,
                reason_codes=(reason,),
                source_value=source_value,
                raw_excerpt=_bounded_excerpt("[" + tag_name + "]"),
                evidence={"parser_name": tag_name, "tag_name": tag_name},
            )

        registered_params = tag_registry.get(tag_name) or ()
        if tag_name not in tag_registry:
            # Known built-in tags that can carry player-visible text are
            # unsupported when their parameters are not registered; user-defined
            # macro invocations are unknown.
            if tag_name in {"ruby", "link", "button", "image", "bg", "text", "ptext", "glink"}:
                source_value = _first_non_empty_pm(node.pm)
                param_name = _first_non_empty_pm_key(node.pm)
                return self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind="tag",
                    classification="unsupported",
                    reason_codes=("tyrano.tag_parameter_not_registered",),
                    source_value=source_value,
                    raw_excerpt=source_value,
                    evidence={"parser_name": tag_name, "tag_name": tag_name, "param_name": param_name, "registered": False},
                )
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="tag",
                classification="unknown",
                reason_codes=("tyrano.unregistered_macro_invocation",),
                source_value=_first_non_empty_pm(node.pm),
                raw_excerpt=_first_non_empty_pm(node.pm),
                evidence={"parser_name": tag_name, "tag_name": tag_name, "registered": False},
            )

        matched = False
        for param_name in registered_params:
            if not isinstance(param_name, str):
                continue
            raw_value = str(node.pm.get(param_name) or "")
            if raw_value == "":
                continue
            matched = True
            if raw_value.startswith("&"):
                candidates = [
                    self._make_candidate(
                        project,
                        document,
                        locator,
                        line=line,
                        node_index=node.node_index,
                        structure_kind="tag",
                        classification="unsupported",
                        reason_codes=("tyrano.dynamic_parameter_expression",),
                        source_value=raw_value,
                        raw_excerpt=raw_value,
                        evidence={"parser_name": tag_name, "tag_name": tag_name, "param_name": param_name, "registered": True, "dynamic_expression": True},
                    )
                ]
                return candidates[0]
            target = catalog.get("tag", {}).get(tag_name, {}).get(param_name, {}).get(raw_value)
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="tag",
                classification=_translation_classification(target),
                reason_codes=("tyrano.registered_tag_parameter",),
                source_value=raw_value,
                raw_excerpt=raw_value,
                catalog_link=({"path": ["scenes", _scenario_key(document.file_rel_path), "tag", tag_name, param_name, raw_value], "translation": target} if target else None),
                evidence={"parser_name": tag_name, "tag_name": tag_name, "param_name": param_name, "registered": True},
                existing_translation=target,
            )
        if not matched:
            # Registered tag present but the source value was empty or missing;
            # keep the tag observable as an engine control/no-text candidate.
            return self._make_candidate(
                project,
                document,
                locator,
                line=line,
                node_index=node.node_index,
                structure_kind="tag",
                classification="explicitly_excluded",
                reason_codes=("tyrano.engine_control_structure",),
                source_value="",
                raw_excerpt="",
                evidence={"parser_name": tag_name, "tag_name": tag_name},
            )

    def _comment_candidate(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        line: int,
        raw_line: str,
    ) -> Candidate:
        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "scenario": _scenario_key(document.file_rel_path),
                "line": line,
                "node_index": None,
                "kind": "comment",
            },
        )
        return self._make_candidate(
            project,
            document,
            locator,
            line=line,
            node_index=None,
            structure_kind="comment",
            classification="explicitly_excluded",
            reason_codes=("tyrano.comment",),
            source_value=raw_line.strip(),
            raw_excerpt=raw_line.strip(),
            evidence={"parser_name": None},
        )

    def _parse_error_candidate(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        *,
        line_index: int,
        excerpt: str,
        reason_code: str,
    ) -> Candidate:
        locator = OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator={
                "file_rel_path": document.file_rel_path,
                "scenario": _scenario_key(document.file_rel_path),
                "line": line_index,
                "node_index": None,
                "kind": "parse_error_region",
                "reason_code": reason_code,
            },
        )
        return self._make_candidate(
            project,
            document,
            locator,
            line=line_index,
            node_index=None,
            structure_kind="parse_error_region",
            classification="parse_error",
            reason_codes=(reason_code,),
            source_value=excerpt,
            raw_excerpt=excerpt,
            evidence={"parser_name": None, "reason_code": reason_code},
        )

    def _make_candidate(
        self,
        project: ProjectDiscovery,
        document: SourceDocument,
        locator: OpaqueLocator,
        *,
        line: int,
        node_index: int | None,
        structure_kind: str,
        classification: str,
        reason_codes: Sequence[str],
        source_value: str,
        raw_excerpt: str,
        catalog_link: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
        existing_translation: str | None = None,
    ) -> Candidate:
        translation_scope, analysis_scope = _scope_for(classification, structure_kind)
        candidate = Candidate(
            candidate_id=_candidate_id(project, locator),
            engine=self.engine,
            adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint,
            locator=locator,
            raw_excerpt=_bounded_excerpt(raw_excerpt),
            structure_kind=structure_kind,
            classification=classification,
            reason_codes=tuple(reason_codes),
            translation_scope=translation_scope,
            analysis_scope=analysis_scope,
            catalog_link=dict(catalog_link) if catalog_link else None,
            evidence=dict(evidence or {}),
        )
        return candidate

    def _catalog_for_scenario(
        self,
        document: SourceDocument,
        catalog_data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        scenes = catalog_data.get("scenes") or {}
        scenario_key = _scenario_key(document.file_rel_path)
        scenario = scenes.get(scenario_key) or {}
        return {
            "scenario": scenario.get("scenario") or {},
            "tag": scenario.get("tag") or {},
            "charas": catalog_data.get("charas") or {},
        }

    # ------------------------------------------------------------------
    # Audit / extraction / unsupported write-side operations
    # ------------------------------------------------------------------

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

        catalog_data = self._load_catalog_data(project)
        catalog_path = Path(project.project_root) / "data/others/lang" / f"{project.target_language}.json"
        catalog_freshness = "unknown"
        report_reasons = ["tyrano.catalog.provenance_unknown"]
        if not catalog_path.is_file():
            catalog_freshness = "missing"
            report_reasons = ["tyrano.catalog.missing_file"]
        elif not catalog_data:
            catalog_freshness = "stale"
            report_reasons = ["tyrano.catalog.stale"]
        else:
            # The hand-written native catalog has no recorded source
            # fingerprint.  Treat it as provenance_unknown until a P5 sidecar
            # or imported Studio project file supplies one.
            catalog_freshness = "unknown"
            report_reasons = ["tyrano.catalog.provenance_unknown"]

        return CoverageReportDraft(
            source_fingerprint=project.source_fingerprint,
            reason_codes=tuple(report_reasons),
            catalog_provenance=dict(project.catalog_provenance),
            catalog_freshness=catalog_freshness,
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
            if candidate.classification not in {"translatable", "already_translated"}:
                raise ValueError(
                    "Candidate is not approved for occurrence extraction: "
                    f"{candidate_id} ({candidate.classification})"
                )
            locator = candidate.locator.locator
            line = int(locator.get("line") or 0)
            source_value = str(candidate.raw_excerpt or "")
            current_translation = ""
            if candidate.catalog_link:
                current_translation = str(candidate.catalog_link.get("translation") or "")
            unit = translation_core.TranslationUnit(
                id=translation_core.build_identity_v2(
                    str(locator.get("file_rel_path") or ""),
                    "tyrano",
                    line,
                    source_value,
                    block_occurrence=int(locator.get("node_index") or 0) + 1,
                ),
                mode=translation_core.MODE_TRANSLATION,
                text=source_value,
                source=source_value,
                current_translation=current_translation,
                file_rel_path=str(locator.get("file_rel_path") or ""),
                file_path=str(documents.get(str(locator.get("file_rel_path") or "")).file_path),
                line=line,
                line_number=line + 1,
                start=0,
                end=len(source_value),
                speaker_id="",
                speaker_name="",
            )
            content_fingerprint = digest_json(
                {
                    "engine": self.engine,
                    "locator": candidate.locator.to_dict(),
                    "source_value": source_value,
                }
            )
            occurrences.append(
                Occurrence(
                    occurrence_id=_occurrence_id(project, candidate.locator),
                    engine=self.engine,
                    project_snapshot_fingerprint=project.project_snapshot_fingerprint,
                    content_fingerprint=content_fingerprint,
                    candidate_id=candidate.candidate_id,
                    locator=candidate.locator,
                    unit=unit,
                )
            )
        return tuple(occurrences)

    # ------------------------------------------------------------------
    # Unsupported write-side operations fail closed
    # ------------------------------------------------------------------

    def relocate_occurrences(
        self,
        project: ProjectDiscovery,
        occurrences: Sequence[Occurrence],
        live_sources: Sequence[SourceDocument],
    ) -> RelocationResult:
        raise NotImplementedError(
            "TyranoAdapter relocation is not implemented yet; planning for P5 "
            "writeback remains unsupported in this read-only adapter version."
        )

    def validate_translation(
        self,
        occurrence: Occurrence,
        translated_text: str,
    ) -> ValidationResult:
        raise NotImplementedError(
            "TyranoAdapter validation is not implemented yet."
        )

    def build_writeback_plan(
        self,
        project: ProjectDiscovery,
        validated: Sequence[ValidatedTranslation],
        live_sources: Sequence[SourceDocument],
    ) -> WritebackPlan:
        raise NotImplementedError(
            "TyranoAdapter writeback planning is not implemented yet."
        )

    # ------------------------------------------------------------------
    # Catalog cache
    # ------------------------------------------------------------------

    @staticmethod
    def _catalog_path(project_root: Path, target_language: str) -> Path:
        return project_root / CATALOG_DIR / f"{target_language}.json"

    def _cache_catalog(
        self,
        source_fingerprint: str,
        target_language: str,
        data: Mapping[str, Any],
    ) -> None:
        key = (source_fingerprint, target_language)
        if key in self._catalog_cache:
            self._catalog_cache_order.remove(key)
        self._catalog_cache[key] = data
        self._catalog_cache_order.append(key)
        while len(self._catalog_cache_order) > 16:
            expired = self._catalog_cache_order.pop(0)
            self._catalog_cache.pop(expired, None)

    def _load_catalog_data(
        self,
        project: ProjectDiscovery,
    ) -> Mapping[str, Any]:
        key = (project.source_fingerprint, project.target_language)
        if key in self._catalog_cache:
            return self._catalog_cache[key]
        catalog_path = Path(project.catalog_provenance.get("catalog_rel_path") or "")
        if not catalog_path.is_absolute():
            catalog_path = Path(project.project_root) / catalog_path
        try:
            data = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self._cache_catalog(project.source_fingerprint, project.target_language, data)
        return data


def build_translation_snapshot(
    adapter: TyranoAdapter,
    request: ProjectDiscoveryRequest,
    policy: InventoryPolicy | None = None,
) -> TyranoTranslationSnapshot:
    """Run the P5 read-only pipeline once for fixtures and future consumers."""
    inventory_policy = policy or InventoryPolicy()
    project = adapter.discover_project(request)
    inventory = adapter.inventory_candidates(project, inventory_policy)
    from .coverage import build_coverage_report

    draft = adapter.audit_extraction(project, inventory)
    report = build_coverage_report(
        project,
        inventory,
        draft,
        adapter_behavior_digest=adapter.behavior_digest(),
    )
    project = replace(project, coverage_digest=report.coverage_digest)
    approved_ids = [
        candidate.candidate_id
        for candidate in inventory.candidates
        if candidate.classification in {"translatable", "already_translated"}
    ]
    occurrences = tuple(
        adapter.extract_occurrences(project, inventory, approved_ids)
    )
    return TyranoTranslationSnapshot(
        project=project,
        inventory=inventory,
        report=report,
        occurrences=occurrences,
    )


def _translation_classification(translation: str | None) -> str:
    if translation is not None and str(translation) != "":
        return "already_translated"
    return "translatable"


def _first_non_empty_pm(pm: Mapping[str, Any]) -> str:
    for value in pm.values():
        if str(value or "") != "":
            return str(value)
    return ""


def _first_non_empty_pm_key(pm: Mapping[str, Any]) -> str:
    for key, value in pm.items():
        if str(value or "") != "":
            return str(key)
    return ""


__all__ = [
    "ADAPTER_VERSION",
    "DEFAULT_TAG_REGISTRY",
    "TyranoAdapter",
    "TyranoParseResult",
    "TyranoNode",
    "TyranoTranslationSnapshot",
    "build_translation_snapshot",
    "parse_tyrano_scenario",
]
