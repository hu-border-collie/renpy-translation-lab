"""Frozen-window MV narrative extraction and declarative source-copy plans.

This core API has no model calls or filesystem write authority. A support
window is explicit, content-addressed evidence supplied by the caller, not a
claim that an arbitrary MV core or plugin combination has been validated.
Product workflows, check/apply authorization and runtime acceptance are A3/A4.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence, TypeGuard

from quality_review.translation_core import MODE_TRANSLATION, TranslationUnit

from .contracts import (
    ENGINE_ADAPTER_PROTOCOL_VERSION,
    SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
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
from .coverage import REVIEW_POLICIES, build_coverage_report, digest_json, stable_json_dumps
from .rpgmaker_mv_controls import validate_message_position
from . import structure_rules
from .writeback import WritebackPlanError, source_snapshot_fingerprint, validate_writeback_plan


ADAPTER_VERSION = "0.1.0"
ENGINE = "rpgmaker_mv"
LOCATOR_SCHEMA_VERSION = 1
CORE_FILES = tuple(
    f"js/rpg_{name}.js"
    for name in ("core", "managers", "objects", "scenes", "sprites", "windows")
)
_MAP = re.compile(r"data/Map[0-9]+\.json")
_DEFERRED_CODES = frozenset({105, 405, 320, 324, 325, 355, 655, 356, 357})
_JSON_PATH = tuple[str | int, ...]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fail(code: str, detail: str) -> NoReturn:
    raise WritebackPlanError(f"rpgmaker_mv.{code}", detail)


@dataclass(frozen=True)
class MvSupportWindow:
    """Pin the six MV cores, plugin registry and every registered plugin.

    Hashes must come from independently reviewed material. The numeric popup
    subset is opt-in and requires an enabled GALV_MessageStyles entry. This
    object does not certify a runtime version or license any third-party files.
    """

    material_sha256: Mapping[str, str]
    message_styles: bool = False

    def __post_init__(self) -> None:
        materials = dict(self.material_sha256)
        if not set((*CORE_FILES, "js/plugins.js")).issubset(materials):
            raise ValueError("rpgmaker_mv.window.incomplete")
        for path, sha256 in materials.items():
            if path not in (*CORE_FILES, "js/plugins.js") and not re.fullmatch(
                r"js/plugins/[A-Za-z0-9_-]+\.js", path
            ):
                raise ValueError("rpgmaker_mv.window.path")
            if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError("rpgmaker_mv.window.digest")
        if type(self.message_styles) is not bool:
            raise ValueError("rpgmaker_mv.window.profile")
        object.__setattr__(self, "material_sha256", MappingProxyType(materials))

    @property
    def digest(self) -> str:
        return digest_json({
            "engine": ENGINE,
            "material_sha256": dict(self.material_sha256),
            "message_styles": self.message_styles,
        })


@dataclass(frozen=True)
class MvProjectDiscovery(ProjectDiscovery):
    """One immutable pair of source/target snapshots, with shared source identity."""

    target_documents: tuple[SourceDocument, ...] = ()
    support_window_digest: str = ""


@dataclass
class _Slot:
    path: _JSON_PATH
    text: str
    kind: str
    context: dict[str, Any] = field(default_factory=dict)


def _json(document: SourceDocument) -> Any:
    def members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("rpgmaker_mv.json.duplicate_key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("rpgmaker_mv.json.constant")

    data = json.loads(document.content.decode("utf-8-sig"),
                      object_pairs_hook=members, parse_constant=constant, parse_float=Decimal)
    pending = [data]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return data


def _at(data: Any, path: _JSON_PATH) -> Any:
    for part in path:
        if isinstance(data, dict) and type(part) is str:
            data = data[part]
        elif isinstance(data, list) and type(part) is int and part >= 0:
            data = data[part]
        else:
            raise ValueError("rpgmaker_mv.json.path")
    return data


def _walk(value: Any, path: _JSON_PATH = ()):
    """Independent raw traversal, deliberately unaware of page/message parsing."""
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, index))


def _raw_slots(data: Any) -> dict[_JSON_PATH, tuple[str, str]]:
    slots = {}
    for path, command in _walk(data):
        code, params = command.get("code"), command.get("parameters")
        if type(code) is not int or not isinstance(params, list):
            continue
        if code in (401, 402):
            index = 0 if code == 401 else 1
            if len(params) > index and isinstance(params[index], str):
                slots[(*path, "parameters", index)] = (
                    params[index], "message" if code == 401 else "mirror"
                )
        elif code == 102 and params and isinstance(params[0], list):
            for index, text in enumerate(params[0]):
                if isinstance(text, str):
                    slots[(*path, "parameters", 0, index)] = (text, "choice")
    return slots


def _integer(value: Any, minimum: int = 0, maximum: int | None = None) -> TypeGuard[int]:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _event_lists(data: Any, rel: str):
    if _MAP.fullmatch(rel):
        if not isinstance(data, dict) or not isinstance(data.get("events"), list):
            raise ValueError("map.events")
        for event_index, event in enumerate(data["events"]):
            if event is None:
                continue
            if not isinstance(event, dict) or not _integer(event.get("id"), 1):
                raise ValueError("map.event")
            if event["id"] != event_index or not isinstance(event.get("pages"), list):
                raise ValueError("map.event_identity")
            for page_index, page in enumerate(event["pages"]):
                if not isinstance(page, dict) or not isinstance(page.get("list"), list):
                    raise ValueError("map.page")
                yield (
                    ("events", event_index, "pages", page_index, "list"),
                    page["list"], {"event_id": event["id"], "page_index": page_index},
                )
    elif rel == "data/CommonEvents.json":
        if not isinstance(data, list):
            raise ValueError("common_events")
        for index, event in enumerate(data):
            if event is None:
                continue
            if (
                not isinstance(event, dict) or not _integer(event.get("id"), 1)
                or event["id"] != index or not isinstance(event.get("list"), list)
            ):
                raise ValueError("common_event")
            yield (index, "list"), event["list"], {"common_event_id": index}


def _scan(document: SourceDocument) -> tuple[list[_Slot], list[dict[str, Any]]]:
    slots: list[_Slot] = []
    issues: list[dict[str, Any]] = []
    frame: dict[str, Any] | None = None
    try:
        data = _json(document)
        for list_path, commands, owner in _event_lists(data, document.file_rel_path):
            message: dict[str, Any] | None = None
            choices: dict[int, dict[str, Any]] = {}
            for index, command in enumerate(commands):
                if not isinstance(command, dict):
                    raise ValueError("command.object")
                code, indent, params = (
                    command.get("code"), command.get("indent"), command.get("parameters")
                )
                if not _integer(code) or not _integer(indent) or not isinstance(params, list):
                    raise ValueError("command.type")
                command_path = (*list_path, index)
                context = {**owner, "command_index": index, "indent": indent}
                if code != 401:
                    message = None
                if choices and indent < max(choices):
                    raise ValueError("choice.unclosed")
                if code == 101:
                    if (
                        len(params) != 4 or not isinstance(params[0], str)
                        or not _integer(params[1], 0, 7)
                        or not _integer(params[2], 0, 2) or not _integer(params[3], 0, 2)
                        or index + 1 == len(commands)
                        or not isinstance(commands[index + 1], dict)
                        or commands[index + 1].get("code") != 401
                    ):
                        raise ValueError("message.header")
                    message = {"header_index": index, "settings": params,
                               "indent": indent, "line_index": 0}
                    block_lines = []
                    for following_index in range(index + 1, len(commands)):
                        following = commands[following_index]
                        if not isinstance(following, dict) or following.get("code") != 401:
                            break
                        following_params = following.get("parameters")
                        if not isinstance(following_params, list) or len(following_params) != 1:
                            raise ValueError("message.parameters")
                        block_lines.append(following_params[0])
                    message["lines"] = block_lines
                elif code == 401:
                    if (
                        message is None or message["indent"] != indent
                        or len(params) != 1 or not isinstance(params[0], str)
                    ):
                        raise ValueError("message.orphan_or_type")
                    slots.append(_Slot((*command_path, "parameters", 0), params[0], "message", {
                        **context, "message_header_index": message["header_index"],
                        "message_settings": message["settings"],
                        "message_line_index": message["line_index"],
                        "message_lines": message["lines"],
                    }))
                    message["line_index"] += 1
                elif code == 102:
                    if (
                        len(params) != 5 or not isinstance(params[0], list)
                        or not 1 <= len(params[0]) <= 6
                        or not all(isinstance(text, str) for text in params[0])
                        or not _integer(params[1], -2, len(params[0]) - 1)
                        or not _integer(params[2], -1, len(params[0]) - 1)
                        or not _integer(params[3], 0, 2) or not _integer(params[4], 0, 2)
                        or indent in choices
                    ):
                        raise ValueError("choice.parameters")
                    frame = {
                        "slots": [], "seen": set(), "cancel": params[1], "cancel_seen": False,
                    }
                    choices[indent] = frame
                    for choice_index, text in enumerate(params[0]):
                        slot = _Slot((*command_path, "parameters", 0, choice_index), text,
                                     "choice", {**context, "choice_index": choice_index,
                                                "choice_settings": params[1:],
                                                "mirror_paths": []})
                        slots.append(slot)
                        frame["slots"].append(slot)
                elif code == 402:
                    frame = choices.get(indent)
                    if (
                        frame is None or len(params) != 2 or not _integer(params[0])
                        or params[0] >= len(frame["slots"]) or params[0] in frame["seen"]
                    ):
                        raise ValueError("choice.mirror_index")
                    slot = frame["slots"][params[0]]
                    if params[1] != slot.text:
                        raise ValueError("choice.mirror_text")
                    mirror_path = (*command_path, "parameters", 1)
                    frame["seen"].add(params[0])
                    slot.context["mirror_paths"].append(list(mirror_path))
                    slots.append(_Slot(mirror_path, params[1], "mirror",
                                       {**context, "owner_path": list(slot.path)}))
                elif code == 403:
                    frame = choices.get(indent)
                    if (
                        frame is None or params or frame["cancel"] != -2
                        or frame["cancel_seen"]
                    ):
                        raise ValueError("choice.cancel")
                    frame["cancel_seen"] = True
                elif code == 404:
                    frame = choices.pop(indent, None)
                    if (
                        frame is None or params or len(frame["seen"]) != len(frame["slots"])
                        or (frame["cancel"] == -2) != frame["cancel_seen"]
                    ):
                        raise ValueError("choice.end")
                elif code in _DEFERRED_CODES:
                    # Source scripts, actor-name changes and scrolling/plugin
                    # text are visible gaps, not silently counted as translated.
                    slots.append(_Slot(command_path, "", "deferred",
                                       {**context, "event_code": code}))
            if choices:
                raise ValueError("choice.unclosed")
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
        issues.append({"reason": str(exc), "file_rel_path": document.file_rel_path})
    return slots, issues


def _documents(root: Path, window: MvSupportWindow) -> tuple[SourceDocument, ...]:
    paths = set(window.material_sha256)
    paths.update(path.relative_to(root).as_posix() for path in (root / "data").rglob("*.json"))
    if "data/CommonEvents.json" not in paths:
        _fail("discovery.incomplete", "Missing CommonEvents.json cannot be counted as zero coverage.")
    if not any(_MAP.fullmatch(path) for path in paths):
        _fail("discovery.no_maps", "No MV map data was found.")
    documents = []
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            _fail("discovery.path", f"Missing or escaped material: {relative}")
        content = path.read_bytes()
        if relative in window.material_sha256 and _sha(content) != window.material_sha256[relative]:
            _fail("window.mismatch", f"Core or plugin material changed: {relative}")
        documents.append(SourceDocument(relative, str(path), len(content), _sha(content), content))
    by_path = {doc.file_rel_path: doc for doc in documents}
    if (root / "js/rmmz_core.js").exists():
        _fail("window.engine", "MZ is not an MV support window.")
    if not re.search(r"""Utils\.RPGMAKER_NAME\s*=\s*(['"])MV\1""",
                     by_path["js/rpg_core.js"].text()):
        _fail("window.engine", "The frozen core does not declare MV.")
    registry = re.search(r"var\s+\$plugins\s*=\s*(\[.*\])\s*;?\s*$",
                         by_path["js/plugins.js"].text(), re.DOTALL)
    try:
        plugins = json.loads(registry[1]) if registry else None
        if not isinstance(plugins, list):
            raise ValueError("plugin registry")
        names = [plugin["name"] for plugin in plugins]
        if (
            len(set(names)) != len(names)
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                   for name in names)
            or any(type(plugin.get("status")) is not bool for plugin in plugins)
        ):
            raise ValueError("plugin entries")
        required = set((*CORE_FILES, "js/plugins.js"))
        required.update(f"js/plugins/{name}.js" for name in names)
        if required != set(window.material_sha256):
            raise ValueError("unbound plugins")
        popup_enabled = any(plugin["name"] == "GALV_MessageStyles" and plugin["status"]
                            for plugin in plugins)
        if popup_enabled != window.message_styles:
            raise ValueError("popup profile")
    except (ValueError, TypeError, KeyError) as exc:
        _fail("window.plugins", f"Plugin window or profile mismatch: {exc}")
    return tuple(documents)


def _verify_documents(documents: Sequence[SourceDocument]) -> None:
    if len({doc.file_rel_path for doc in documents}) != len(documents):
        _fail("snapshot.duplicate", "Duplicate material paths.")
    for doc in documents:
        if doc.size != len(doc.content) or doc.sha256 != _sha(doc.content):
            _fail("snapshot.metadata", f"Incorrect material metadata: {doc.file_rel_path}")


def _same_snapshot(expected: Sequence[SourceDocument], current: Sequence[SourceDocument]) -> bool:
    _verify_documents(expected)
    _verify_documents(current)
    return source_snapshot_fingerprint(expected) == source_snapshot_fingerprint(current)


def _same_structure(source: Any, target: Any, allowed: set[_JSON_PATH],
                    path: _JSON_PATH = ()) -> bool:
    if path in allowed and isinstance(source, str) and source.strip():
        return isinstance(target, str)
    if type(source) is not type(target):
        return False
    if isinstance(source, dict):
        return source.keys() == target.keys() and all(
            _same_structure(value, target[key], allowed, (*path, key))
            for key, value in source.items()
        )
    if isinstance(source, list):
        return len(source) == len(target) and all(
            _same_structure(left, right, allowed, (*path, index))
            for index, (left, right) in enumerate(zip(source, target))
        )
    return source == target


class RpgMakerMvAdapter:
    """Implement the shared Adapter protocol for an explicit MV narrative window.

    Discovery roots are content roots containing data/ and js/. localization_root
    names the independent target content root, not a catalog beneath the source.
    All methods are read-only or return in-memory declarative objects.
    """

    protocol_version = ENGINE_ADAPTER_PROTOCOL_VERSION
    engine = ENGINE
    adapter_version = ADAPTER_VERSION

    def __init__(self, support_window: MvSupportWindow):
        self.support_window = support_window

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.engine, adapter_version=self.adapter_version,
            supported_localization_modes=(LocalizationMode.SOURCE_EXTRACTION,),
            selected_localization_mode=LocalizationMode.SOURCE_EXTRACTION,
            relocation=True, declarative_writeback=("json_source_string_set",),
        )

    def behavior_digest(self) -> str:
        return digest_json({"engine": self.engine, "adapter_version": self.adapter_version,
                            "window": self.support_window.digest, "locator_schema_version": 1})

    def _identity(self, source_root: str, target_root: str, language: str, fingerprint: str) -> str:
        return digest_json({"engine": self.engine, "adapter_version": self.adapter_version,
                            "source_root": source_root, "target_root": target_root,
                            "target_language": language, "source_fingerprint": fingerprint,
                            "support_window": self.support_window.digest})

    def discover_project(self, request: ProjectDiscoveryRequest) -> MvProjectDiscovery:
        """Freeze separate source/current-target snapshots and exact runtime material."""
        if not request.project_root or not request.localization_root or not request.target_language:
            _fail("discovery.roots", "Explicit source, target and target language are required.")
        source_root, target_root = (Path(request.project_root).resolve(),
                                    Path(request.localization_root).resolve())
        if source_root.is_relative_to(target_root) or target_root.is_relative_to(source_root):
            _fail("discovery.overlap", "Source and target roots must be separate, non-nested copies.")
        if request.include_files or request.include_prefixes:
            _fail("discovery.filter", "Partial discovery cannot establish MV coverage.")
        sources, targets = (_documents(source_root, self.support_window),
                            _documents(target_root, self.support_window))
        target_by_path = {doc.file_rel_path: doc for doc in targets}
        if set(target_by_path) != {doc.file_rel_path for doc in sources}:
            _fail("target.structure", "Source and target material sets differ.")
        for source in sources:
            if not source.file_rel_path.endswith(".json"):
                continue
            target = target_by_path[source.file_rel_path]
            try:
                source_data, target_data = _json(source), _json(target)
            except (ValueError, UnicodeError, RecursionError):
                if source.content == target.content:
                    continue  # Inventory reports the invalid input as a blocker.
                _fail("target.structure", "Source and target JSON cannot be compared.")
            allowed = set(_raw_slots(source_data)) if self._narrative_file(source.file_rel_path) else set()
            if not _same_structure(source_data, target_data, allowed):
                _fail("target.structure", f"Non-narrative target structure changed: {source.file_rel_path}")
        fingerprint = source_snapshot_fingerprint(sources)
        return MvProjectDiscovery(
            engine=self.engine, adapter_version=self.adapter_version,
            project_root=str(source_root), localization_root=str(target_root),
            target_language=request.target_language,
            project_snapshot_fingerprint=self._identity(
                str(source_root), str(target_root), request.target_language, fingerprint),
            source_fingerprint=fingerprint, source_documents=sources,
            localization_mode=LocalizationMode.SOURCE_EXTRACTION,
            catalog_provenance={"support_window_digest": self.support_window.digest,
                                "target_snapshot_fingerprint": source_snapshot_fingerprint(targets),
                                "runtime_acceptance": "unverified"},
            target_documents=targets, support_window_digest=self.support_window.digest,
        )

    @staticmethod
    def _narrative_file(relative: str) -> bool:
        return bool(_MAP.fullmatch(relative)) or relative == "data/CommonEvents.json"

    def _verify_project(self, project: ProjectDiscovery) -> MvProjectDiscovery:
        if (
            not isinstance(project, MvProjectDiscovery) or project.engine != self.engine
            or project.adapter_version != self.adapter_version
            or project.support_window_digest != self.support_window.digest
            or project.localization_mode != LocalizationMode.SOURCE_EXTRACTION
        ):
            _fail("snapshot.project", "Wrong project, adapter or support window.")
        _verify_documents(project.source_documents)
        _verify_documents(project.target_documents)
        if (
            project.source_fingerprint != source_snapshot_fingerprint(project.source_documents)
            or project.project_snapshot_fingerprint != self._identity(
                project.project_root, project.localization_root, project.target_language,
                project.source_fingerprint)
            or project.catalog_provenance.get("target_snapshot_fingerprint")
            != source_snapshot_fingerprint(project.target_documents)
        ):
            _fail("snapshot.project", "Project snapshot identity is inconsistent.")
        return project

    def read_live_documents(self, project: ProjectDiscovery, *,
                            target: bool = False) -> tuple[SourceDocument, ...]:
        """Re-read a complete material set, including newly added data JSON files."""
        project = self._verify_project(project)
        root = Path(project.localization_root if target else project.project_root)
        return _documents(root, self.support_window)

    def _source_problem(self, slot: _Slot) -> str:
        try:
            validate_message_position(
                slot.text, message_styles=self.support_window.message_styles,
                first_line=slot.kind == "message" and slot.context["message_line_index"] == 0,
            )
        except ValueError as exc:
            return str(exc)
        return ""

    def _candidate(self, project: MvProjectDiscovery, document: SourceDocument,
                   slot: _Slot) -> Candidate:
        problem = self._source_problem(slot) if slot.kind in {"message", "choice"} else ""
        classification, reason = "translatable", f"rpgmaker_mv.{slot.kind}"
        if slot.kind == "mirror":
            classification, reason = "explicitly_excluded", "rpgmaker_mv.choice_mirror"
        elif slot.kind == "deferred":
            classification, reason = "unsupported", "rpgmaker_mv.deferred_text"
        elif not slot.text.strip():
            classification, reason = "explicitly_excluded", "rpgmaker_mv.empty_message"
        elif problem:
            classification, reason = "unknown", "rpgmaker_mv.unknown_control"
        locator = OpaqueLocator(self.engine, LOCATOR_SCHEMA_VERSION, {
            "file_rel_path": document.file_rel_path, "json_path": list(slot.path),
            "kind": slot.kind, "source_text": slot.text, **slot.context,
            "mv_message_styles": self.support_window.message_styles,
        })
        candidate_id = "cand1:" + digest_json({
            "snapshot": project.project_snapshot_fingerprint, "locator": locator.to_dict(),
        })
        scope = "include" if classification == "translatable" else (
            "exclude" if classification == "explicitly_excluded" else "unknown")
        return Candidate(
            candidate_id=candidate_id, engine=self.engine, adapter_version=self.adapter_version,
            source_fingerprint=project.source_fingerprint, locator=locator,
            raw_excerpt=slot.text[:240], structure_kind=f"mv_{slot.kind}",
            classification=classification, reason_codes=(reason,),
            translation_scope=scope, analysis_scope=scope,
            evidence={"control_problem": problem, "physical_position": list(slot.path)},
        )

    def inventory_candidates(self, project: ProjectDiscovery,
                             policy: InventoryPolicy) -> CandidateInventory:
        project = self._verify_project(project)
        if policy.review_policy not in REVIEW_POLICIES:
            raise ValueError("rpgmaker_mv.review_policy")
        candidates: list[Candidate] = []
        call_sites: dict[int, list[dict[str, Any]]] = {}
        for document in project.source_documents:
            if not self._narrative_file(document.file_rel_path):
                continue
            try:
                for path, command in _walk(_json(document)):
                    params = command.get("parameters")
                    if (command.get("code") == 117 and isinstance(params, list)
                            and len(params) == 1 and _integer(params[0], 1)):
                        call_sites.setdefault(params[0], []).append({
                            "file_rel_path": document.file_rel_path, "command_path": list(path),
                        })
            except (ValueError, UnicodeError, RecursionError):
                pass  # The document scan below produces the blocking candidate.
        for document in project.source_documents:
            rel = document.file_rel_path
            if self._narrative_file(rel):
                slots, issues = _scan(document)
                for slot in slots:
                    if "common_event_id" in slot.context:
                        slot.context["common_event_call_sites"] = call_sites.get(
                            slot.context["common_event_id"], [])
                candidates.extend(self._candidate(project, document, slot) for slot in slots)
                for issue in issues:
                    candidate = self._candidate(project, document, _Slot((), "", "error"))
                    candidate.classification = "parse_error"
                    candidate.reason_codes = ("rpgmaker_mv.invalid_event",)
                    candidate.translation_scope = candidate.analysis_scope = "unknown"
                    candidate.evidence = issue
                    candidates.append(candidate)
            elif rel.endswith(".json"):
                candidate = self._candidate(project, document, _Slot((), "", "outside_scope"))
                candidate.reason_codes = ("rpgmaker_mv.outside_narrative_scope",)
                try:
                    _json(document)
                except (ValueError, UnicodeError, RecursionError):
                    candidate.classification = "parse_error"
                    candidate.reason_codes = ("rpgmaker_mv.invalid_json",)
                    candidate.translation_scope = candidate.analysis_scope = "unknown"
                candidates.append(candidate)
        return CandidateInventory(
            self.engine, self.adapter_version, project.source_fingerprint,
            project.project_snapshot_fingerprint, tuple(candidates),
            tuple(doc.manifest_entry() for doc in project.source_documents),
        )

    def audit_extraction(self, project: ProjectDiscovery,
                         inventory: CandidateInventory) -> CoverageReportDraft:
        """Account for raw 401/102/402 leaves independently of extracted messages."""
        project = self._verify_project(project)
        expected: dict[tuple[str, _JSON_PATH], tuple[str, str]] = {}
        expected_classes: dict[tuple[str, _JSON_PATH], str] = {}
        reasons = []
        for document in project.source_documents:
            if not self._narrative_file(document.file_rel_path):
                continue
            try:
                data = _json(document)
                for path, row in _raw_slots(data).items():
                    key = (document.file_rel_path, path)
                    expected[key] = row
                    classification = "explicitly_excluded"
                    if row[0].strip() and row[1] != "mirror":
                        classification = "translatable"
                        try:
                            commands, index = _at(data, path[:-3]), path[-3]
                            first_line = (
                                row[1] == "message" and type(index) is int and index > 0
                                and isinstance(commands[index - 1], dict)
                                and commands[index - 1].get("code") == 101
                            )
                            validate_message_position(
                                row[0], message_styles=self.support_window.message_styles,
                                first_line=first_line,
                            )
                        except ValueError:
                            classification = "unknown"
                    expected_classes[key] = classification
            except (ValueError, UnicodeError, RecursionError):
                reasons.append("rpgmaker_mv.invalid_json")
        actual: Counter = Counter()
        for candidate in inventory.candidates:
            locator = candidate.locator.locator
            if locator.get("kind") not in {"message", "choice", "mirror"}:
                continue
            path = locator.get("json_path")
            if not isinstance(path, (list, tuple)) or any(
                not ((type(part) is int and part >= 0) or (type(part) is str and part))
                for part in path
            ):
                reasons.append("rpgmaker_mv.coverage.mismatch")
                continue
            key = (locator.get("file_rel_path"), tuple(path))
            actual[key] += 1
            original = expected.get(key)
            if original is None or original != (locator.get("source_text"), locator.get("kind")):
                reasons.append("rpgmaker_mv.coverage.mismatch")
            elif candidate.classification != expected_classes[key]:
                reasons.append("rpgmaker_mv.coverage.mismatch")
        if actual != Counter({key: 1 for key in expected}):
            reasons.append("rpgmaker_mv.coverage.mismatch")
        try:
            changed = not _same_snapshot(project.source_documents, self.read_live_documents(project))
            if not _same_snapshot(project.target_documents,
                                  self.read_live_documents(project, target=True)):
                reasons.append("rpgmaker_mv.snapshot.stale")
        except (ValueError, OSError):
            changed = True
        return CoverageReportDraft(
            project.source_fingerprint, tuple(sorted(set(reasons))),
            {**project.catalog_provenance, "scope": "401/102 narrative; 402 linked physical targets",
             "narrative_physical_positions": len(expected)},
            "current", changed,
        )

    def _report(self, project: ProjectDiscovery, inventory: CandidateInventory):
        return build_coverage_report(
            project, inventory, self.audit_extraction(project, inventory),
            adapter_behavior_digest=self.behavior_digest(),
        )

    def extract_occurrences(self, project: ProjectDiscovery, inventory: CandidateInventory,
                            approved_candidate_ids: Sequence[str]) -> Sequence[Occurrence]:
        project = self._verify_project(project)
        canonical = self.inventory_candidates(project, InventoryPolicy())
        if (
            inventory != canonical
            or [row.to_dict() for row in inventory.candidates]
            != [row.to_dict() for row in canonical.candidates]
            or self._report(project, inventory).coverage_status == "block"
        ):
            _fail("coverage.invalid", "Inventory is stale, altered or structurally blocked.")
        if len(set(approved_candidate_ids)) != len(approved_candidate_ids):
            _fail("occurrence.duplicate", "Duplicate selected candidate.")
        by_id = canonical.by_id()
        targets = {doc.file_rel_path: doc for doc in project.target_documents}
        parsed_targets = {relative: _json(document) for relative, document in targets.items()
                          if self._narrative_file(relative)}
        sources = project.document_by_path()
        occurrences = []
        for candidate_id in approved_candidate_ids:
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate.classification != "translatable":
                _fail("occurrence.candidate", "Candidate is unknown or not translatable.")
            locator = candidate.locator
            address = locator.locator
            relative = address["file_rel_path"]
            text = address["source_text"]
            unit = TranslationUnit(
                id="mv1:" + digest_json({"file": relative, "path": address["json_path"],
                                         "source": text, "window": self.support_window.digest}),
                mode=MODE_TRANSLATION, text=text, source=text,
                current_translation=_at(parsed_targets[relative], tuple(address["json_path"])),
                file_rel_path=relative, file_path=sources[relative].file_path,
                line=-1, line_number=0, start=0, end=len(text),
                metadata={"mv_message_styles": self.support_window.message_styles,
                          "mv_context": dict(address)},
            )
            occurrences.append(Occurrence(
                occurrence_id="occ1:" + digest_json({
                    "snapshot": project.project_snapshot_fingerprint, "locator": locator.to_dict()}),
                engine=self.engine, project_snapshot_fingerprint=project.project_snapshot_fingerprint,
                content_fingerprint=digest_json({"engine": self.engine,
                                                 "locator": locator.to_dict(), "source": text}),
                candidate_id=candidate_id, locator=locator, unit=unit,
            ))
        return tuple(occurrences)

    def relocate_occurrences(self, project: ProjectDiscovery, occurrences: Sequence[Occurrence],
                             live_sources: Sequence[SourceDocument]) -> RelocationResult:
        """Validate exact snapshot locations; never chase equal text after movement."""
        project = self._verify_project(project)
        if not _same_snapshot(project.source_documents, live_sources):
            return RelocationResult((), tuple(row.occurrence_id for row in occurrences),
                                    ({"reason_code": "rpgmaker_mv.snapshot.stale"},))
        inventory = self.inventory_candidates(project, InventoryPolicy())
        ids = [row.candidate_id for row in occurrences]
        try:
            fresh = self.extract_occurrences(project, inventory, ids)
        except ValueError:
            fresh = ()
        canonical = {row.occurrence_id: row for row in fresh}
        accepted, unresolved = [], []
        seen = set()
        for occurrence in occurrences:
            if occurrence.occurrence_id in seen or canonical.get(occurrence.occurrence_id) != occurrence:
                unresolved.append(occurrence.occurrence_id)
            else:
                accepted.append(occurrence)
            seen.add(occurrence.occurrence_id)
        return RelocationResult(tuple(accepted), tuple(unresolved))

    def validate_translation(self, occurrence: Occurrence, translated_text: str) -> ValidationResult:
        """Validate MV structure and UTF-8, preserving invalid input identity in blocked results."""
        if occurrence.engine != self.engine or occurrence.locator.engine != self.engine:
            raise ValueError("rpgmaker_mv.validation.engine")
        reasons = []
        serialized_translation = stable_json_dumps({
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "translation": translated_text,
        })
        try:
            translation_digest = _sha(serialized_translation.encode("utf-8"))
        except UnicodeEncodeError:
            # Preserve exact invalid code points for diagnostics only. They
            # remain blocked and are never repaired into writable translations.
            translation_digest = _sha(
                serialized_translation.encode("utf-8", errors="surrogatepass"))
            reasons.append("rpgmaker_mv.translation.encoding")
        address = occurrence.locator.locator
        if (
            not isinstance(translated_text, str) or not translated_text.strip()
            or not occurrence.unit.text.strip()
        ):
            reasons.append("rpgmaker_mv.translation.empty")
        elif not reasons:
            try:
                for text in (occurrence.unit.text, translated_text):
                    validate_message_position(
                        text, message_styles=self.support_window.message_styles,
                        first_line=address.get("kind") == "message"
                        and address.get("message_line_index") == 0,
                    )
                structure_rules.validate(
                    occurrence.unit.text, translated_text, self.engine,
                    mv_message_styles=self.support_window.message_styles,
                )
            except (ValueError, UnicodeError) as exc:
                reasons.append(str(exc) if not isinstance(exc, UnicodeError)
                               else "rpgmaker_mv.translation.encoding")
        return ValidationResult(
            occurrence_id=occurrence.occurrence_id, engine=self.engine,
            status="block" if reasons else "pass", reason_codes=tuple(reasons), diagnostics=(),
            source_constraints_digest=digest_json({
                "window": self.support_window.digest, "locator": occurrence.locator.to_dict(),
                "content": occurrence.content_fingerprint, "source": occurrence.unit.text}),
            translation_digest=translation_digest,
        )

    def build_writeback_plan(self, project: ProjectDiscovery, validated: Sequence[ValidatedTranslation],
                             live_sources: Sequence[SourceDocument], *,
                             live_targets: Sequence[SourceDocument] | None = None) -> WritebackPlan:
        """Build schema 2 operations against both exact snapshots, without writing.

        Source-only callers fail closed. Coverage/review bindings are mandatory;
        the downstream workflow must still authorize the latest check and apply.
        """
        project = self._verify_project(project)
        if live_targets is None:
            _fail("target.required", "MV plans require separate live target documents.")
        if not _same_snapshot(project.source_documents, live_sources):
            _fail("source.stale", "Frozen source changed.")
        if not _same_snapshot(project.target_documents, live_targets):
            _fail("target.stale", "Current target changed.")
        inventory = self.inventory_candidates(project, InventoryPolicy())
        report = self._report(project, inventory)
        if (
            report.coverage_status == "block" or not project.coverage_review_digest
            or project.coverage_digest != report.coverage_digest
        ):
            _fail("coverage.invalid", "Current non-blocking coverage and review bindings are required.")
        canonical = {
            row.occurrence_id: row for row in self.extract_occurrences(
                project, inventory, [item.occurrence.candidate_id for item in validated])
        }
        targets = {doc.file_rel_path: doc for doc in live_targets}
        parsed_targets = {rel: _json(doc) for rel, doc in targets.items() if rel.endswith(".json")}
        operations = []
        seen_occurrences, seen_targets = set(), set()
        for item in validated:
            occurrence = item.occurrence
            if canonical.get(occurrence.occurrence_id) != occurrence:
                _fail("occurrence.stale", "Occurrence does not match its canonical source and current target.")
            if occurrence.occurrence_id in seen_occurrences:
                _fail("occurrence.duplicate", "Duplicate translated occurrence.")
            seen_occurrences.add(occurrence.occurrence_id)
            validation = self.validate_translation(occurrence, item.translated_text)
            if validation != item.validation or validation.status != "pass":
                _fail("validation.stale", "Translation validation is blocking, stale or altered.")
            address = occurrence.locator.locator
            relative = address["file_rel_path"]
            paths = [address["json_path"], *address.get("mirror_paths", [])]
            for raw_path in paths:
                path = tuple(raw_path)
                if (relative, path) in seen_targets:
                    _fail("target.duplicate", "Multiple operations address the same physical target.")
                seen_targets.add((relative, path))
                current = _at(parsed_targets[relative], path)
                if not isinstance(current, str):
                    _fail("target.type", "Current target is not a string.")
                operation = WritebackOperation(
                    operation_id="", kind="json_source_string_set",
                    occurrence_id=occurrence.occurrence_id, target_root="source_copy",
                    target_rel_path=relative, expected_file_sha256=targets[relative].sha256,
                    line=-1, start_col=-1, end_col=-1,
                    expected_fragment_sha256=_sha(current.encode("utf-8")),
                    expected_text_digest=_sha(occurrence.unit.text.encode("utf-8")),
                    replacement_fragment=item.translated_text,
                    validation_digest=digest_json(validation.to_dict()), target_json_path=path,
                )
                payload = operation.to_dict()
                payload.pop("operation_id")
                operations.append(replace(operation, operation_id="op2:" + digest_json(payload)))
        plan = WritebackPlan(
            engine=self.engine, adapter_version=self.adapter_version,
            project_identity_digest=project.project_snapshot_fingerprint,
            source_snapshot_fingerprint=source_snapshot_fingerprint(live_sources),
            coverage_digest=project.coverage_digest, coverage_review_digest=project.coverage_review_digest,
            operations=tuple(operations), plan_digest="",
            writeback_plan_schema_version=SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION,
            target_snapshot_fingerprint=source_snapshot_fingerprint(live_targets),
        )
        payload = plan.to_dict()
        payload.pop("plan_digest")
        plan = replace(plan, plan_digest=digest_json(payload))
        validate_writeback_plan(plan, live_sources, live_targets=live_targets)
        return plan
