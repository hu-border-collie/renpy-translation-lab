# -*- coding: utf-8 -*-
"""TyranoScript V600+ hybrid adapter (#265 P5 / #399).

The adapter inventories source ``.ks`` files but emits semantic writeback
operations only for existing rows in Tyrano's native language JSON. Direct
source-script mutation remains unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import translation_core

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
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
    digest_json,
)
from .writeback import WritebackPlanError, source_snapshot_fingerprint


ADAPTER_VERSION = "0.3.0"
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


def _validate_language_code(value: str) -> str:
    language = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", language):
        raise ValueError(f"Invalid Tyrano language code: {value!r}")
    return language


def _catalog_state_from_bytes(content: bytes) -> Mapping[str, Any]:
    """Decode one Tyrano language JSON without consulting the filesystem."""

    sha256 = _sha256_bytes(content)
    try:
        data = json.loads(content.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "data": {},
            "status": "invalid_json",
            "sha256": sha256,
            "content": content,
            "detail": type(exc).__name__,
        }
    if not isinstance(data, dict):
        return {
            "data": {},
            "status": "invalid_json",
            "sha256": sha256,
            "content": content,
            "detail": type(data).__name__,
        }
    return {
        "data": data,
        "status": "ok",
        "sha256": sha256,
        "content": content,
    }


def _keep_space_from_project(project: ProjectDiscovery) -> str:
    """Read parser whitespace semantics from the immutable discovery snapshot."""

    document = project.document_by_path().get(CONFIG_REL_PATH)
    if document is None:
        return "2"
    text = document.content.decode("utf-8-sig", errors="replace")
    match = re.search(r";?KeepSpaceInParameterValue\s*=\s*([123])\s*;?", text)
    if not match:
        return "2"
    return match.group(1)


def _is_ks_file(file_rel_path: str) -> bool:
    return str(file_rel_path or "").lower().endswith(".ks")


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
    match = re.search(r";?KeepSpaceInParameterValue\s*=\s*([123])\s*;?", text)
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
        official_compensated: bool = False,
    ) -> None:
        error = {
            "line": line_index,
            "reason_code": reason_code,
            "structure_kind": struct_kind,
            "excerpt": _bounded_excerpt(excerpt),
        }
        if official_compensated:
            error["official_compensated"] = True
        parse_errors.append(error)

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
            official_compensated = tag_buf.endswith("]")
            if not official_compensated:
                # An unclosed inline tag has no final ``]`` at all.
                raise_node_error(
                    line_index,
                    "tyrano.unclosed_inline_tag",
                    "inline_tag",
                    tag_line_text,
                )
            else:
                # Official parser strips the trailing ``]`` and re-emits the
                # tag through makeTag.  P5 flags the compensation explicitly.
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
                    official_compensated=official_compensated,
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
    """Parse ``kag.parser.makeTag``-style tag attributes.

    The official scanner is intentionally permissive; the strict adapter keeps
    normalized values but reports the two cases P5 must make visible:
    unclosed quoted attributes and bare unquoted tokens following a parameter.
    Space padding around ``=`` is legal and does not produce an error.
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
    name = trimmed[name_start:index].strip()

    while index < length:
        while index < length and trimmed[index] == " ":
            index += 1
        if index >= length:
            break

        param_name_start = index
        while index < length and trimmed[index] not in {" ", "="}:
            index += 1
        if index >= length:
            param_name = trimmed[param_name_start:index].strip()
            if param_name:
                pm.setdefault(param_name, "")
            break
        param_name = trimmed[param_name_start:index].strip()

        if trimmed[index] == " ":
            # Official ``SCANNING_EQUAL`` skips blanks until ``=`` or the next
            # token. Keep the current parameter name when ``=`` follows.
            lookahead = index
            while lookahead < length and trimmed[lookahead] == " ":
                lookahead += 1
            if lookahead >= length:
                if param_name:
                    pm.setdefault(param_name, "")
                break
            if trimmed[lookahead] == "=":
                index = lookahead
            else:
                if param_name:
                    pm.setdefault(param_name, "")
                    error = _set_error(
                        error,
                        "tyrano.unquoted_parameter_sequence",
                        "tag",
                        trimmed,
                    )
                index = lookahead
                continue

        # trimmed[index] == '='
        index += 1
        while index < length and trimmed[index] == " ":
            index += 1
        if index >= length:
            if param_name:
                pm.setdefault(param_name, "")
            break

        if trimmed[index] in {'"', "'", "`"}:
            quote = trimmed[index]
            index += 1
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
            if param_name:
                pm[param_name] = _normalize_tag_value(
                    "".join(value_chars),
                    keep_space,
                    quote,
                )
        else:
            value_start = index
            while index < length and trimmed[index] != " ":
                index += 1
            if param_name:
                pm[param_name] = _normalize_tag_value(
                    trimmed[value_start:index],
                    keep_space,
                    "",
                )

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


def _normalize_tag_value(raw_value: str, keep_space: str, quote: str) -> str:
    value = raw_value
    # Official makeTag only strips half-width spaces inside non-backtick
    # quoted values when KeepSpaceInParameterValue is 1.  Backtick-quoted
    # values preserve interior spaces.
    if keep_space == "1" and quote != "`":
        value = value.replace(" ", "")
    if keep_space != "3":
        value = value.strip()
    if value == "undefined":
        return ""
    return value


def _merged_tag_registry(catalog_data: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
    """Official built-in registrations are always active.

    TyranoStudio writes project registrations into the catalog's ``tags`` key;
    those may add or override parameters but must not silently remove the
    built-in ``glink`` / ``ptext`` defaults.
    """
    merged: dict[str, tuple[str, ...]] = {
        tag: tuple(params) for tag, params in DEFAULT_TAG_REGISTRY.items()
    }
    project_tags = catalog_data.get("tags") if isinstance(catalog_data, dict) else None
    if isinstance(project_tags, dict):
        for tag_name, params in project_tags.items():
            if isinstance(tag_name, str) and isinstance(params, list):
                normalized = tuple(str(item) for item in params if isinstance(item, str))
                if normalized:
                    merged[tag_name] = normalized
    return merged


class TyranoAdapter:
    """TyranoScript V600+ hybrid source/catalog adapter.

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
            relocation=True,
            declarative_writeback=("json_catalog_set",),
            native_catalog_required_for_writeback=True,
        )

    def behavior_digest(self) -> str:
        return digest_json(
            {
                "engine_adapter_protocol_version": self.protocol_version,
                "engine": self.engine,
                "adapter_version": self.adapter_version,
                "locator_schema_version": self.locator_schema_version,
                "read_only": False,
                "source_inventory": True,
                "native_catalog": True,
                "relocation": True,
                "json_catalog_writeback": True,
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
        target_language = _validate_language_code(request.target_language)
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
        catalog_dir = project_root_path / CATALOG_DIR
        catalog_path = self._catalog_path(project_root_path, target_language)
        catalog_states: dict[str, Mapping[str, Any]] = {}
        catalog_files: list[dict[str, Any]] = []
        invalid_catalog_files: list[str] = []
        language_names_by_casefold: dict[str, list[str]] = {}
        if catalog_dir.is_dir():
            for candidate_path in sorted(catalog_dir.iterdir(), key=lambda item: item.name):
                if not candidate_path.is_file() or candidate_path.suffix.lower() != ".json":
                    continue
                language = candidate_path.stem
                rel_path = _normalize_rel_path(
                    os.path.relpath(candidate_path, project_root_path)
                )
                content = candidate_path.read_bytes()
                state = _catalog_state_from_bytes(content)
                catalog_states[language] = state
                language_names_by_casefold.setdefault(language.casefold(), []).append(language)
                if not re.fullmatch(r"[A-Za-z0-9_-]+", language):
                    invalid_catalog_files.append(rel_path)
                catalog_files.append(
                    {
                        "language": language,
                        "rel_path": rel_path,
                        "sha256": str(state.get("sha256") or ""),
                        "status": str(state.get("status") or "unknown"),
                    }
                )
                documents.append(
                    SourceDocument(
                        file_rel_path=rel_path,
                        file_path=str(candidate_path),
                        size=len(content),
                        sha256=str(state.get("sha256") or ""),
                        content=content,
                    )
                )
        catalog_state = catalog_states.get(
            target_language,
            {"data": {}, "status": "missing", "sha256": "", "content": None},
        )
        language_collisions = [
            tuple(sorted(names))
            for names in language_names_by_casefold.values()
            if len(names) > 1
        ]
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

        catalog_provenance = {
            "format": "tyrano_lang_json",
            "target_language": target_language,
            "catalog_rel_path": "data/others/lang/" + target_language + ".json",
            "catalog_sha256": catalog_state["sha256"],
            "catalog_file_exists": catalog_state.get("status") != "missing",
            "catalog_status": catalog_state["status"],
            "catalog_files": tuple(catalog_files),
            "available_catalog_languages": tuple(sorted(catalog_states)),
            "invalid_catalog_files": tuple(sorted(invalid_catalog_files)),
            "language_code_collisions": tuple(sorted(language_collisions)),
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
            catalog_state,
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

        keep_space = _keep_space_from_project(project)
        catalog_data = self._load_catalog_data(project)

        candidates: list[Candidate] = []
        file_entries: list[Mapping[str, Any]] = []
        for document in project.source_documents:
            if not _is_ks_file(document.file_rel_path):
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
        line_official_compensation: set[int] = set()
        for error in parse_result.parse_errors:
            line_index = int(error["line"])
            reasons = line_errors.setdefault(line_index, [])
            reason_code = str(error["reason_code"])
            if reason_code not in reasons:
                reasons.append(reason_code)
            if error.get("official_compensated"):
                line_official_compensation.add(line_index)
        for line_index in line_official_compensation:
            reasons = line_errors.setdefault(line_index, [])
            if "tyrano.official_parser_compensated" not in reasons:
                reasons.append("tyrano.official_parser_compensated")

        for node in parse_result.nodes:
            candidates.extend(
                self._candidate_for_node(
                    project,
                    document,
                    node,
                    line=node.line,
                    catalog=catalog,
                    catalog_data=catalog_data,
                    line_error_reasons=line_errors.get(node.line),
                    line_has_official_compensation=(
                        "tyrano.official_parser_compensated" in line_errors.get(node.line, [])
                    ),
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

    def _candidate_locator(
        self,
        document: SourceDocument,
        *,
        line: int,
        node_index: int | None,
        kind: str,
        name: str | None,
        parser_value: str = "",
        tag_name: str = "",
        param_name: str = "",
    ) -> OpaqueLocator:
        locator: dict[str, Any] = {
            "file_rel_path": document.file_rel_path,
            "scenario": _scenario_key(document.file_rel_path),
            "line": line,
            "node_index": node_index,
            "kind": kind,
            "name": name or "",
            "parser_value": parser_value,
        }
        if tag_name:
            locator["tag_name"] = tag_name
        if param_name:
            locator["param_name"] = param_name
        return OpaqueLocator(
            engine=self.engine,
            locator_schema_version=self.locator_schema_version,
            locator=locator,
        )

    @staticmethod
    def _parse_error_source_value(
        node: TyranoNode,
        catalog_data: Mapping[str, Any],
    ) -> str:
        if node.kind == "text":
            return str(node.pm.get("val") or "")
        if node.kind == "tag":
            registered = _merged_tag_registry(catalog_data).get(node.name) or ()
            for param_name in registered:
                value = str(node.pm.get(param_name) or "")
                if value:
                    return value
        return ""

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
        line_has_official_compensation: bool = False,
    ) -> list[Candidate]:
        tag_registry = _merged_tag_registry(catalog_data)
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

        if line_error_reasons:
            source_value = self._parse_error_source_value(node, catalog_data)
            reasons = list(line_error_reasons)
            if line_has_official_compensation and "tyrano.official_parser_compensated" not in reasons:
                reasons.append("tyrano.official_parser_compensated")
            if (
                node.kind == "tag"
                and node.name not in DEFAULT_TAG_REGISTRY
                and node.name not in structural_tags
                and "tyrano.unregistered_macro_invocation" not in reasons
            ):
                reasons.append("tyrano.unregistered_macro_invocation")
            structure_kind = "text" if node.kind == "text" else "tag"
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=node.name,
                parser_value=source_value,
                tag_name=node.name if node.kind == "tag" else "",
                param_name="",
            )
            return [
                self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind=structure_kind,
                    classification="parse_error",
                    reason_codes=tuple(reasons),
                    source_value=source_value,
                    raw_excerpt=source_value or _first_non_empty_pm(node.pm),
                    evidence={"parser_name": node.name},
                )
            ]

        if node.kind == "label":
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=node.name,
                parser_value=node.val,
            )
            return [
                self._make_candidate(
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
            ]

        if node.kind == "chara_ptext":
            chara_name = str(node.pm.get("name") or "")
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=node.name,
                parser_value=chara_name,
            )
            if chara_name.startswith("&"):
                return [
                    self._make_candidate(
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
                ]
            target = catalog.get("charas", {}).get(chara_name)
            classification = _translation_classification(target)
            return [
                self._make_candidate(
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
            ]

        if node.kind == "text":
            source_value = str(node.pm.get("val") or "")
            target = catalog.get("scenario", {}).get(source_value)
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=node.name,
                parser_value=source_value,
            )
            if getattr(node, "in_iscript", False):
                return [
                    self._make_candidate(
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
                ]
            return [
                self._make_candidate(
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
            ]

        # tag node
        tag_name = node.name
        if tag_name in structural_tags:
            reason, classification = structural_tags[tag_name]
            source_value = ""
            reason_codes = [reason]
            evidence: dict[str, Any] = {
                "parser_name": tag_name,
                "tag_name": tag_name,
            }
            if tag_name == "lang_set":
                source_value = str(node.pm.get("name") or "")
                available_languages = set(
                    project.catalog_provenance.get("available_catalog_languages") or ()
                )
                if source_value.startswith("&"):
                    reason_codes.append("tyrano.lang_set.dynamic_expression")
                    evidence["dynamic_expression"] = True
                elif not re.fullmatch(r"[A-Za-z0-9_-]+", source_value):
                    reason_codes.append("tyrano.lang_set.language_code_invalid")
                else:
                    if source_value not in available_languages:
                        reason_codes.append("tyrano.lang_set.catalog_missing")
                    if source_value != project.target_language:
                        reason_codes.append("tyrano.lang_set.target_mismatch")
                evidence["target_language"] = project.target_language
                evidence["available_catalog_languages"] = tuple(
                    sorted(available_languages)
                )
            elif tag_name == "chara_new":
                source_value = str(node.pm.get("jname") or "")
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=tag_name,
                parser_value=source_value,
                tag_name=tag_name,
            )
            return [
                self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind="tag",
                    classification=classification,
                    reason_codes=tuple(reason_codes),
                    source_value=source_value,
                    raw_excerpt=source_value or _bounded_excerpt("[" + tag_name + "]"),
                    evidence=evidence,
                )
            ]

        registered_params = tag_registry.get(tag_name) or ()
        if tag_name not in tag_registry:
            source_value = _first_non_empty_pm(node.pm)
            param_name = _first_non_empty_pm_key(node.pm)
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=tag_name,
                parser_value=source_value,
                tag_name=tag_name,
                param_name=param_name,
            )
            if tag_name in {"ruby", "link", "button", "image", "bg", "text", "ptext", "glink"}:
                return [
                    self._make_candidate(
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
                ]
            return [
                self._make_candidate(
                    project,
                    document,
                    locator,
                    line=line,
                    node_index=node.node_index,
                    structure_kind="tag",
                    classification="unknown",
                    reason_codes=("tyrano.unregistered_macro_invocation",),
                    source_value=source_value,
                    raw_excerpt=source_value,
                    evidence={"parser_name": tag_name, "tag_name": tag_name, "registered": False},
                )
            ]

        candidates: list[Candidate] = []
        for param_name in registered_params:
            if not isinstance(param_name, str):
                continue
            raw_value = str(node.pm.get(param_name) or "")
            if raw_value == "":
                continue
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=tag_name,
                parser_value=raw_value,
                tag_name=tag_name,
                param_name=param_name,
            )
            if raw_value.startswith("&"):
                candidates.append(
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
                )
                continue
            target = catalog.get("tag", {}).get(tag_name, {}).get(param_name, {}).get(raw_value)
            candidates.append(
                self._make_candidate(
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
            )
        if not candidates:
            locator = self._candidate_locator(
                document,
                line=line,
                node_index=node.node_index,
                kind=node.kind,
                name=tag_name,
                tag_name=tag_name,
            )
            candidates.append(
                self._make_candidate(
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
            )
        return candidates

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
    # Audit / extraction
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

        states = self._catalog_states_from_documents(project)
        state = states.get(
            project.target_language,
            {"data": {}, "status": "missing", "sha256": "", "content": None},
        )
        catalog_data = state.get("data") or {}
        status = state.get("status", "unknown")
        if status == "missing":
            catalog_freshness = "missing"
            report_reasons = ["tyrano.catalog.missing_file"]
        elif status == "invalid_json":
            catalog_freshness = "stale"
            report_reasons = ["tyrano.catalog.invalid_json"]
        else:
            catalog_freshness = "unknown"
            report_reasons = []
            if not project.catalog_provenance.get("recorded_source_fingerprint"):
                report_reasons.append("tyrano.catalog.provenance_unknown")
        report_reasons.extend(self._catalog_inventory_audit_reasons(project))
        lang_set_reasons, referenced_languages, has_dynamic_language = (
            self._lang_set_audit(project, inventory, states)
        )
        report_reasons.extend(lang_set_reasons)

        languages_to_audit = {project.target_language, *referenced_languages}
        if has_dynamic_language:
            languages_to_audit.update(states)
        target_registry = _merged_tag_registry(catalog_data)
        for language in sorted(languages_to_audit):
            catalog_state = states.get(language)
            if catalog_state is None:
                continue
            if catalog_state.get("status") != "ok":
                report_reasons.append("tyrano.catalog.invalid_json")
                continue
            language_catalog = catalog_state.get("data") or {}
            report_reasons.extend(
                self._catalog_content_audit_reasons(
                    project,
                    inventory,
                    language_catalog,
                )
            )
            if language != project.target_language and (
                _merged_tag_registry(language_catalog) != target_registry
            ):
                report_reasons.append("tyrano.catalog.tag_registry_mismatch")

        return CoverageReportDraft(
            source_fingerprint=project.source_fingerprint,
            reason_codes=tuple(dict.fromkeys(report_reasons)),
            catalog_provenance=dict(project.catalog_provenance),
            catalog_freshness=catalog_freshness,
            source_changed_during_scan=source_changed,
        )

    @staticmethod
    def _catalog_states_from_documents(
        project: ProjectDiscovery,
    ) -> dict[str, Mapping[str, Any]]:
        prefix = f"{CATALOG_DIR}/"
        states: dict[str, Mapping[str, Any]] = {}
        for document in project.source_documents:
            rel_path = _normalize_rel_path(document.file_rel_path)
            if not rel_path.startswith(prefix):
                continue
            filename = rel_path[len(prefix) :]
            if "/" in filename or not filename.lower().endswith(".json"):
                continue
            language = filename[:-5]
            states[language] = _catalog_state_from_bytes(document.content)
        return states

    @staticmethod
    def _catalog_inventory_audit_reasons(project: ProjectDiscovery) -> list[str]:
        reasons: list[str] = []
        if project.catalog_provenance.get("invalid_catalog_files"):
            reasons.append("tyrano.catalog.language_code_invalid")
        if project.catalog_provenance.get("language_code_collisions"):
            reasons.append("tyrano.catalog.language_code_collision")
        return reasons

    @staticmethod
    def _lang_set_audit(
        project: ProjectDiscovery,
        inventory: CandidateInventory,
        catalog_states: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[str], set[str], bool]:
        reasons: list[str] = []
        referenced_languages: set[str] = set()
        has_dynamic = False
        for candidate in inventory.candidates:
            if "tyrano.lang_set_control_tag" not in candidate.reason_codes:
                continue
            language = str(candidate.locator.locator.get("parser_value") or "")
            if language.startswith("&"):
                has_dynamic = True
                reasons.append("tyrano.lang_set.dynamic_expression")
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", language):
                reasons.append("tyrano.lang_set.language_code_invalid")
                continue
            referenced_languages.add(language)
            if language not in catalog_states:
                reasons.append("tyrano.lang_set.catalog_missing")
            if language != project.target_language:
                reasons.append("tyrano.lang_set.target_mismatch")
        return reasons, referenced_languages, has_dynamic

    @staticmethod
    def _catalog_content_audit_reasons(
        project: ProjectDiscovery,
        inventory: CandidateInventory,
        catalog_data: Mapping[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        scenes = catalog_data.get("scenes")
        if scenes is None:
            return ["tyrano.catalog.missing_scenario"]
        if not isinstance(scenes, dict):
            return ["tyrano.catalog.invalid_json"]

        charas = catalog_data.get("charas", {})
        systems = catalog_data.get("systems", {})
        tags = catalog_data.get("tags", {})
        if not isinstance(charas, dict):
            reasons.append("tyrano.catalog.invalid_json")
            charas = {}
        if not isinstance(systems, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in getattr(systems, "items", lambda: ())()
        ):
            reasons.append("tyrano.catalog.invalid_json")
        if not isinstance(tags, dict):
            reasons.append("tyrano.catalog.invalid_json")
        else:
            for tag_name, params in tags.items():
                if (
                    not isinstance(tag_name, str)
                    or not isinstance(params, list)
                    or not params
                    or any(not isinstance(param, str) or not param for param in params)
                    or len(params) != len(set(params))
                ):
                    reasons.append("tyrano.catalog.invalid_json")

        source_scenario_keys = {
            _scenario_key(document.file_rel_path)
            for document in project.source_documents
            if _is_ks_file(document.file_rel_path)
        }
        catalog_scenario_keys = set(scenes)
        for scenario_key in source_scenario_keys:
            if scenario_key not in catalog_scenario_keys:
                reasons.append("tyrano.catalog.missing_scenario")
        for scenario_key in catalog_scenario_keys:
            if scenario_key not in source_scenario_keys:
                reasons.append("tyrano.catalog.stale")

        observed_text_values: dict[str, set[str]] = {}
        observed_tag_values: dict[tuple[str, str, str], set[str]] = {}
        observed_chara_values: set[str] = set()
        for candidate in inventory.candidates:
            if candidate.classification not in {"translatable", "already_translated"}:
                continue
            locator = candidate.locator.locator
            scenario_key = str(locator.get("scenario") or "")
            parser_value = str(locator.get("parser_value") or "")
            if candidate.structure_kind == "text":
                observed_text_values.setdefault(scenario_key, set()).add(parser_value)
            elif candidate.structure_kind == "chara_ptext":
                observed_chara_values.add(parser_value)
            elif candidate.structure_kind == "tag":
                tag_name = str(locator.get("tag_name") or "")
                param_name = str(locator.get("param_name") or "")
                if tag_name and param_name:
                    observed_tag_values.setdefault(
                        (scenario_key, tag_name, param_name), set()
                    ).add(parser_value)

        for scenario_key, scene in scenes.items():
            if not isinstance(scene, dict):
                reasons.append("tyrano.catalog.invalid_json")
                continue
            if "scenario" not in scene or "tag" not in scene:
                reasons.append("tyrano.catalog.invalid_json")
            scenario_rows = scene.get("scenario", {})
            tag_rows = scene.get("tag", {})
            if not isinstance(scenario_rows, dict) or not isinstance(tag_rows, dict):
                reasons.append("tyrano.catalog.invalid_json")
                continue
            observed_text = observed_text_values.get(scenario_key, set())
            for source_text, translation in scenario_rows.items():
                if source_text not in observed_text:
                    reasons.append("tyrano.catalog.stale")
                if not isinstance(source_text, str) or not isinstance(translation, str):
                    reasons.append("tyrano.catalog.invalid_json")
                elif translation == "":
                    reasons.append("tyrano.catalog.empty_translation")
            if any(source_text not in scenario_rows for source_text in observed_text):
                reasons.append("tyrano.catalog.missing_row")
            for tag_name, params in tag_rows.items():
                if not isinstance(params, dict):
                    reasons.append("tyrano.catalog.invalid_json")
                    continue
                for param_name, rows in params.items():
                    if not isinstance(rows, dict):
                        reasons.append("tyrano.catalog.invalid_json")
                        continue
                    observed_tag = observed_tag_values.get(
                        (scenario_key, tag_name, param_name), set()
                    )
                    for source_value, translation in rows.items():
                        if source_value not in observed_tag:
                            reasons.append("tyrano.catalog.stale")
                        if not isinstance(source_value, str) or not isinstance(translation, str):
                            reasons.append("tyrano.catalog.invalid_json")
                        elif translation == "":
                            reasons.append("tyrano.catalog.empty_translation")

        for (scenario_key, tag_name, param_name), source_values in observed_tag_values.items():
            scene = scenes.get(scenario_key)
            rows: Any = None
            if isinstance(scene, dict):
                tag_rows = scene.get("tag")
                if isinstance(tag_rows, dict):
                    tag = tag_rows.get(tag_name)
                    if isinstance(tag, dict):
                        rows = tag.get(param_name)
            if not isinstance(rows, dict) or any(
                source_value not in rows for source_value in source_values
            ):
                reasons.append("tyrano.catalog.missing_row")

        for source_value, translation in charas.items():
            if source_value not in observed_chara_values:
                reasons.append("tyrano.catalog.stale")
            if not isinstance(source_value, str) or not isinstance(translation, str):
                reasons.append("tyrano.catalog.invalid_json")
            elif translation == "":
                reasons.append("tyrano.catalog.empty_translation")
        if any(source_value not in charas for source_value in observed_chara_values):
            reasons.append("tyrano.catalog.missing_row")

        return reasons

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
            # Full parser-normalized value lives in the locator; ``raw_excerpt``
            # is only a bounded display string and must never seed identity.
            source_value = str(locator.get("parser_value") or "")
            if not source_value:
                raise ValueError(
                    f"Candidate has no parser_value for occurrence extraction: "
                    f"{candidate_id}"
                )
            document = documents.get(str(locator.get("file_rel_path") or ""))
            if document is None:
                raise ValueError(
                    f"Candidate source document is missing: {locator.get('file_rel_path')}"
                )
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
                file_path=document.file_path,
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
    # Validation and native catalog writeback
    # ------------------------------------------------------------------

    def _project_with_live_sources(
        self,
        project: ProjectDiscovery,
        live_sources: Sequence[SourceDocument],
    ) -> ProjectDiscovery:
        documents_by_path: dict[str, SourceDocument] = {}
        for document in live_sources:
            rel_path = _normalize_rel_path(document.file_rel_path)
            if rel_path in documents_by_path:
                raise ValueError(f"Duplicate Tyrano live source path: {rel_path}")
            documents_by_path[rel_path] = (
                document
                if document.file_rel_path == rel_path
                else replace(document, file_rel_path=rel_path)
            )
        documents = tuple(
            documents_by_path[rel_path] for rel_path in sorted(documents_by_path)
        )
        source_fingerprint = source_snapshot_fingerprint(documents)
        project_snapshot_fingerprint = digest_json(
            {
                "engine": self.engine,
                "localization_mode": project.localization_mode.value,
                "target_language": project.target_language,
                "source_fingerprint": source_fingerprint,
            }
        )

        catalog_rel_path = f"{CATALOG_DIR}/{project.target_language}.json"
        catalog_document = documents_by_path.get(catalog_rel_path)
        if catalog_document is None:
            catalog_state: Mapping[str, Any] = {
                "data": {},
                "status": "missing",
                "sha256": "",
                "content": None,
            }
        else:
            catalog_state = _catalog_state_from_bytes(catalog_document.content)

        catalog_files: list[dict[str, Any]] = []
        available_languages: list[str] = []
        invalid_catalog_files: list[str] = []
        names_by_casefold: dict[str, list[str]] = {}
        prefix = f"{CATALOG_DIR}/"
        for rel_path, document in documents_by_path.items():
            if not rel_path.startswith(prefix):
                continue
            filename = rel_path[len(prefix) :]
            if "/" in filename or not filename.lower().endswith(".json"):
                continue
            language = filename[:-5]
            state = _catalog_state_from_bytes(document.content)
            available_languages.append(language)
            names_by_casefold.setdefault(language.casefold(), []).append(language)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", language):
                invalid_catalog_files.append(rel_path)
            catalog_files.append(
                {
                    "language": language,
                    "rel_path": rel_path,
                    "sha256": str(state.get("sha256") or ""),
                    "status": str(state.get("status") or "unknown"),
                }
            )
        collisions = [
            tuple(sorted(names))
            for names in names_by_casefold.values()
            if len(names) > 1
        ]
        catalog_provenance = dict(project.catalog_provenance)
        catalog_provenance.update(
            {
                "catalog_sha256": str(catalog_state.get("sha256") or ""),
                "catalog_file_exists": catalog_document is not None,
                "catalog_status": str(catalog_state.get("status") or "unknown"),
                "catalog_files": tuple(sorted(catalog_files, key=lambda item: item["rel_path"])),
                "available_catalog_languages": tuple(sorted(available_languages)),
                "invalid_catalog_files": tuple(sorted(invalid_catalog_files)),
                "language_code_collisions": tuple(sorted(collisions)),
                "live_source_fingerprint": source_fingerprint,
            }
        )
        live_project = replace(
            project,
            project_snapshot_fingerprint=project_snapshot_fingerprint,
            source_fingerprint=source_fingerprint,
            source_documents=documents,
            catalog_provenance=catalog_provenance,
        )
        self._cache_catalog(
            source_fingerprint,
            project.target_language,
            catalog_state,
        )
        return live_project

    @staticmethod
    def _relocation_key(occurrence: Occurrence) -> tuple[str, ...]:
        locator = occurrence.locator.locator
        return (
            _normalize_rel_path(str(locator.get("file_rel_path") or "")),
            str(locator.get("kind") or ""),
            str(locator.get("name") or ""),
            str(locator.get("tag_name") or ""),
            str(locator.get("param_name") or ""),
            str(locator.get("parser_value") or ""),
        )

    def relocate_occurrences(
        self,
        project: ProjectDiscovery,
        occurrences: Sequence[Occurrence],
        live_sources: Sequence[SourceDocument],
    ) -> RelocationResult:
        if project.engine != self.engine:
            raise ValueError(f"TyranoAdapter cannot relocate engine={project.engine!r}.")
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
        live_by_semantic_key: dict[tuple[str, ...], list[Occurrence]] = {}
        for candidate in live_occurrences:
            live_by_unit_id.setdefault(candidate.unit.id, []).append(candidate)
            live_by_semantic_key.setdefault(self._relocation_key(candidate), []).append(candidate)

        relocated: list[Occurrence] = []
        unresolved: list[str] = []
        diagnostics: list[Mapping[str, Any]] = []
        used_live_ids: set[str] = set()
        for original in occurrences:
            if original.engine != self.engine or original.locator.engine != self.engine:
                raise ValueError(
                    f"TyranoAdapter cannot relocate engine={original.engine!r}."
                )
            exact = [
                candidate
                for candidate in live_by_unit_id.get(original.unit.id, ())
                if candidate.occurrence_id not in used_live_ids
                and self._relocation_key(candidate) == self._relocation_key(original)
            ]
            match = exact[0] if len(exact) == 1 else None
            match_kind = "identity_v2"
            candidates: Sequence[Occurrence] = exact
            if match is None:
                candidates = [
                    candidate
                    for candidate in live_by_semantic_key.get(
                        self._relocation_key(original), ()
                    )
                    if candidate.occurrence_id not in used_live_ids
                ]
                if len(candidates) == 1:
                    match = candidates[0]
                    match_kind = "semantic_locator"
            if match is None:
                unresolved.append(original.occurrence_id)
                diagnostics.append(
                    {
                        "occurrence_id": original.occurrence_id,
                        "reason_code": "common.locator.unresolved",
                        "status": "ambiguous" if candidates else "missing",
                        "candidate_count": len(candidates),
                    }
                )
                continue

            used_live_ids.add(match.occurrence_id)
            updated_unit = replace(
                match.unit,
                id=original.unit.id,
                mode=original.unit.mode,
            )
            relocated.append(replace(match, unit=updated_unit))
            diagnostics.append(
                {
                    "occurrence_id": original.occurrence_id,
                    "status": "relocated",
                    "match": match_kind,
                    "live_occurrence_id": match.occurrence_id,
                    "file_rel_path": updated_unit.file_rel_path,
                    "line": updated_unit.line,
                    "node_index": match.locator.locator.get("node_index"),
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
        if occurrence.engine != self.engine or occurrence.locator.engine != self.engine:
            raise ValueError(
                f"TyranoAdapter cannot validate engine={occurrence.engine!r}."
            )
        source_constraints_digest = digest_json(
            {
                "engine": self.engine,
                "content_fingerprint": occurrence.content_fingerprint,
                "locator": occurrence.locator.to_dict(),
                "source_text": occurrence.unit.text,
            }
        )
        reason_codes: list[str] = []
        diagnostics: list[Mapping[str, Any]] = []
        if not isinstance(translated_text, str) or translated_text == "":
            reason_codes.append("tyrano.translation.empty")
        elif any(unicodedata.category(character) == "Cc" for character in translated_text):
            reason_codes.append("tyrano.translation.control_character")
            diagnostics.append(
                {
                    "reason_code": "tyrano.translation.control_character",
                    "positions": [
                        index
                        for index, character in enumerate(translated_text)
                        if unicodedata.category(character) == "Cc"
                    ],
                }
            )
        translation_digest = digest_json(
            {
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "translation": translated_text,
            }
        )
        return ValidationResult(
            occurrence_id=occurrence.occurrence_id,
            engine=self.engine,
            status="block" if reason_codes else "pass",
            reason_codes=tuple(reason_codes),
            diagnostics=tuple(diagnostics),
            source_constraints_digest=source_constraints_digest,
            translation_digest=translation_digest,
        )

    @staticmethod
    def _catalog_json_path(occurrence: Occurrence) -> tuple[str, ...]:
        locator = occurrence.locator.locator
        source_value = str(locator.get("parser_value") or "")
        file_rel_path = _normalize_rel_path(str(locator.get("file_rel_path") or ""))
        scenario = str(locator.get("scenario") or "")
        if not source_value or source_value != occurrence.unit.text:
            raise WritebackPlanError(
                "tyrano.writeback.locator_invalid",
                f"Tyrano occurrence source does not match its locator: {occurrence.occurrence_id}",
            )
        if scenario != _scenario_key(file_rel_path):
            raise WritebackPlanError(
                "tyrano.writeback.locator_invalid",
                f"Tyrano occurrence scenario does not match its source path: {occurrence.occurrence_id}",
            )
        kind = str(locator.get("kind") or "")
        if kind == "text":
            return ("scenes", scenario, "scenario", source_value)
        if kind == "chara_ptext":
            return ("charas", source_value)
        if kind == "tag":
            tag_name = str(locator.get("tag_name") or "")
            param_name = str(locator.get("param_name") or "")
            if tag_name and param_name:
                return (
                    "scenes",
                    scenario,
                    "tag",
                    tag_name,
                    param_name,
                    source_value,
                )
        raise WritebackPlanError(
            "tyrano.writeback.locator_unsupported",
            f"Tyrano occurrence has no writable catalog locator: {occurrence.occurrence_id}",
        )

    @staticmethod
    def _catalog_value_at_path(
        catalog_data: Mapping[str, Any],
        json_path: tuple[str, ...],
    ) -> str:
        current: Any = catalog_data
        for part in json_path[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise WritebackPlanError(
                    "tyrano.writeback.catalog_path_missing",
                    "Tyrano catalog path is missing: " + "/".join(json_path),
                )
            current = current[part]
        leaf = json_path[-1]
        if not isinstance(current, dict) or leaf not in current:
            raise WritebackPlanError(
                "tyrano.writeback.catalog_row_missing",
                "Tyrano catalog row is missing: " + "/".join(json_path),
            )
        value = current[leaf]
        if not isinstance(value, str):
            raise WritebackPlanError(
                "tyrano.writeback.catalog_value_invalid",
                "Tyrano catalog row is not a string: " + "/".join(json_path),
            )
        return value

    def build_writeback_plan(
        self,
        project: ProjectDiscovery,
        validated: Sequence[ValidatedTranslation],
        live_sources: Sequence[SourceDocument],
    ) -> WritebackPlan:
        if project.engine != self.engine:
            raise ValueError(
                f"TyranoAdapter cannot build a plan for engine={project.engine!r}."
            )
        expected_documents = project.document_by_path()
        live_documents = {
            _normalize_rel_path(document.file_rel_path): document
            for document in live_sources
        }
        if len(live_documents) != len(live_sources):
            raise WritebackPlanError(
                "tyrano.writeback.source_snapshot_mismatch",
                "Tyrano writeback received duplicate live source paths.",
            )
        if set(live_documents) != set(expected_documents):
            raise WritebackPlanError(
                "tyrano.writeback.source_snapshot_mismatch",
                "Tyrano writeback requires the complete discovered source and catalog snapshot.",
            )
        for rel_path, expected in expected_documents.items():
            live = live_documents[rel_path]
            if live.sha256 != expected.sha256 or live.size != expected.size:
                raise WritebackPlanError(
                    "tyrano.writeback.source_snapshot_mismatch",
                    f"Tyrano writeback source changed after discovery: {rel_path}",
                )
        live_source_fingerprint = source_snapshot_fingerprint(tuple(live_documents.values()))
        if live_source_fingerprint != project.source_fingerprint:
            raise WritebackPlanError(
                "tyrano.writeback.source_snapshot_mismatch",
                "Tyrano writeback snapshot fingerprint changed after discovery.",
            )

        # Re-run the independent coverage audit over the exact immutable input
        # set used for this plan. Tyrano's runtime silently falls back to source
        # text for malformed or incomplete catalogs, so a blocked inventory may
        # never be allowed to reach semantic catalog rendering.
        from .coverage import build_coverage_report

        live_project = self._project_with_live_sources(project, live_sources)
        live_inventory = self.inventory_candidates(live_project, InventoryPolicy())
        coverage_draft = self.audit_extraction(live_project, live_inventory)
        coverage_report = build_coverage_report(
            live_project,
            live_inventory,
            coverage_draft,
            adapter_behavior_digest=self.behavior_digest(),
        )
        if coverage_report.coverage_status == "block":
            blocking_reasons = ", ".join(
                sorted(
                    code
                    for code, count in coverage_report.reason_counts.items()
                    if count
                )
            )
            raise WritebackPlanError(
                "tyrano.writeback.coverage_block",
                "Tyrano writeback coverage audit is blocked"
                + (f": {blocking_reasons}" if blocking_reasons else "."),
            )
        recorded_coverage_digest = str(
            project.coverage_digest
            or project.catalog_provenance.get("coverage_digest")
            or ""
        )
        if (
            recorded_coverage_digest
            and recorded_coverage_digest != coverage_report.coverage_digest
        ):
            raise WritebackPlanError(
                "tyrano.writeback.coverage_stale",
                "Tyrano writeback coverage digest no longer matches the live audit.",
            )

        catalog_rel_path = _normalize_rel_path(
            str(project.catalog_provenance.get("catalog_rel_path") or "")
        )
        catalog_document = live_documents.get(catalog_rel_path)
        if catalog_document is None:
            raise WritebackPlanError(
                "tyrano.writeback.catalog_missing",
                f"Tyrano catalog is missing from the live snapshot: {catalog_rel_path}",
            )
        if catalog_document.sha256 != str(
            project.catalog_provenance.get("catalog_sha256") or ""
        ):
            raise WritebackPlanError(
                "tyrano.writeback.catalog_snapshot_mismatch",
                f"Tyrano catalog changed after discovery: {catalog_rel_path}",
            )
        try:
            catalog_data = json.loads(catalog_document.text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WritebackPlanError(
                "tyrano.writeback.catalog_invalid_json",
                f"Tyrano catalog is not valid JSON: {type(exc).__name__}",
            ) from exc
        if not isinstance(catalog_data, dict):
            raise WritebackPlanError(
                "tyrano.writeback.catalog_invalid_json",
                "Tyrano catalog root must be an object.",
            )

        rows: dict[tuple[str, ...], tuple[ValidatedTranslation, str]] = {}
        for item in validated:
            occurrence = item.occurrence
            if occurrence.project_snapshot_fingerprint != project.project_snapshot_fingerprint:
                raise WritebackPlanError(
                    "tyrano.writeback.occurrence_snapshot_mismatch",
                    f"Tyrano occurrence is stale: {occurrence.occurrence_id}",
                )
            expected_content_fingerprint = digest_json(
                {
                    "engine": self.engine,
                    "locator": occurrence.locator.to_dict(),
                    "source_value": occurrence.unit.text,
                }
            )
            if occurrence.content_fingerprint != expected_content_fingerprint:
                raise WritebackPlanError(
                    "tyrano.writeback.occurrence_digest_mismatch",
                    f"Tyrano occurrence digest is invalid: {occurrence.occurrence_id}",
                )
            expected_validation = self.validate_translation(
                occurrence,
                item.translated_text,
            )
            if item.validation != expected_validation or expected_validation.status != "pass":
                raise WritebackPlanError(
                    "tyrano.writeback.validation_mismatch",
                    f"Tyrano validation is missing, stale, or blocking: {occurrence.occurrence_id}",
                )
            json_path = self._catalog_json_path(occurrence)
            current_value = self._catalog_value_at_path(catalog_data, json_path)
            previous = rows.get(json_path)
            if previous is not None and previous[0].translated_text != item.translated_text:
                raise WritebackPlanError(
                    "tyrano.writeback.catalog_translation_conflict",
                    "Tyrano occurrences mapped to one catalog row have conflicting translations: "
                    + "/".join(json_path),
                )
            if previous is None or occurrence.occurrence_id < previous[0].occurrence.occurrence_id:
                rows[json_path] = (item, current_value)

        operations: list[WritebackOperation] = []
        for json_path in sorted(rows):
            item, current_value = rows[json_path]
            operation_payload = {
                "kind": "json_catalog_set",
                "occurrence_id": item.occurrence.occurrence_id,
                "target_root": "localization_catalog",
                "target_rel_path": catalog_rel_path,
                "expected_file_sha256": catalog_document.sha256,
                "line": -1,
                "start_col": -1,
                "end_col": -1,
                "expected_fragment_sha256": _sha256_bytes(current_value.encode("utf-8")),
                "expected_text_digest": _sha256_bytes(
                    item.occurrence.unit.text.encode("utf-8")
                ),
                "replacement_fragment": item.translated_text,
                "validation_digest": digest_json(item.validation.to_dict()),
                "target_json_path": json_path,
            }
            operations.append(
                WritebackOperation(
                    operation_id="op1:" + digest_json(operation_payload),
                    **operation_payload,
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
        coverage_digest = coverage_report.coverage_digest
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
        state: Mapping[str, Any],
    ) -> None:
        key = (source_fingerprint, target_language)
        if key in self._catalog_cache:
            self._catalog_cache_order.remove(key)
        self._catalog_cache[key] = state
        self._catalog_cache_order.append(key)
        while len(self._catalog_cache_order) > 16:
            expired = self._catalog_cache_order.pop(0)
            self._catalog_cache.pop(expired, None)

    def _read_catalog_state(
        self,
        project_root: Path,
        target_language: str,
    ) -> Mapping[str, Any]:
        path = self._catalog_path(project_root, target_language)
        if not path.is_file():
            return {"data": {}, "status": "missing", "sha256": "", "content": None}
        try:
            catalog_bytes = path.read_bytes()
        except OSError as exc:
            return {
                "data": {},
                "status": "missing",
                "sha256": "",
                "content": None,
                "detail": type(exc).__name__,
            }
        return _catalog_state_from_bytes(catalog_bytes)

    def _load_catalog_state(
        self,
        project: ProjectDiscovery,
    ) -> Mapping[str, Any]:
        key = (project.source_fingerprint, project.target_language)
        if key in self._catalog_cache:
            return self._catalog_cache[key]
        state = self._read_catalog_state(
            Path(project.project_root),
            project.target_language,
        )
        self._cache_catalog(project.source_fingerprint, project.target_language, state)
        return state

    def _load_catalog_data(
        self,
        project: ProjectDiscovery,
    ) -> Mapping[str, Any]:
        return self._load_catalog_state(project).get("data", {})


def build_translation_snapshot(
    adapter: TyranoAdapter,
    request: ProjectDiscoveryRequest,
    policy: InventoryPolicy | None = None,
) -> TyranoTranslationSnapshot:
    """Build one immutable P5 discovery, coverage, and occurrence snapshot."""
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
