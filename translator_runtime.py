# -*- coding: utf-8 -*-
import os
import io
import ast
import json
import re
import time
import tokenize
import random
import sys
import glob
import pickle
import shutil
import subprocess
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# RLock so apply_runtime_config can nest inside locked_runtime_state /
# runtime_config_scope without deadlocking (GUI workers hold the outer lock).
_runtime_state_lock = threading.RLock()
_active_runtime_config: Optional["RuntimeConfig"] = None


@contextmanager
def locked_runtime_state():
    """Serialize temporary BASE_DIR / runtime-config overrides across workers."""
    with _runtime_state_lock:
        yield

from atomic_io import atomic_write_json, atomic_write_lines, file_sha256, sha256_text
import cli_contract
from engine_adapters.contracts import ProjectDiscoveryRequest, ValidatedTranslation
from engine_adapters.coverage import export_coverage_package
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
from engine_adapters.writeback import (
    WritebackPlanError,
    render_writeback_plan,
    source_snapshot_fingerprint,
)
from gemini_model_catalog import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_TRANSLATION_MODEL,
    catalog_extra_models,
    default_model_rotation_list,
    filter_gemini_generation_config,
    merge_model_lists,
    normalize_model_names,
)
from rag_memory import JsonRagStore, JsonSourceIndexStore, hash_text
import advanced_context
import embedding_runtime
from embedding_backend import EmbeddingBackendError, EmbeddingContractError
from rpa_safety import (
    DEFAULT_RPA_LIMITS,
    RpaExtractionBudget,
    copy_member,
    decode_and_validate_index,
    member_output_size,
    read_bounded_compressed_index,
)
import model_profile
import model_usage_ledger
import prompt_context
import story_memory
import sync_translation_preview
import translation_core
import translation_plan
import translation_quality
from sync_model_backend import (
    DEFAULT_SYNC_TIMEOUT_SECONDS,
    MAX_SYNC_TIMEOUT_SECONDS,
    MIN_SYNC_TIMEOUT_SECONDS,
    SyncGenerationRequest,
    normalize_sync_timeout_seconds,
    sync_error_detail,
    sync_error_summary,
    sync_recovery_decision,
)

# Configuration
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
# Tool layout is flat under the package root only — no parent-directory fallback
# for keys, logs, or a default game_root.
FLAT_CONFIG = os.path.join(TOOL_DIR, "api_keys.json")
TRANSLATOR_CONFIG = os.path.join(TOOL_DIR, "translator_config.json")
DEFAULT_GLOSSARY_FILE = os.path.join(TOOL_DIR, "glossary.json")
GLOSSARY_FILE = DEFAULT_GLOSSARY_FILE
ROOT_DIR = TOOL_DIR
DATA_DIR = TOOL_DIR
CONFIG_FILE = FLAT_CONFIG
ENV_GAME_ROOT = os.environ.get("GAME_ROOT") or os.environ.get("SA_GAME_ROOT")
# Forward-slash form matches normalize_tl_subdir() output on all platforms.
DEFAULT_TL_SUBDIR = "game/tl/schinese"
DEFAULT_PREP_LANGUAGE = "schinese"
DEFAULT_CONTEXT_STORAGE_LOCATION = "tool"
DEFAULT_CONTEXT_STORAGE_GAME_DIR_NAME = "translation_context"
# First entry is the rotation/fallback default (cost-efficient lite).
DEFAULT_MODELS = default_model_rotation_list()
DEFAULT_MAX_CHARS = translation_core.CANONICAL_CHUNK_MAX_CHARS
DEFAULT_MAX_ITEMS = translation_core.CANONICAL_CHUNK_MAX_ITEMS
DEFAULT_SYNC_MAX_OUTPUT_TOKENS = 24576
DEFAULT_SYNC_BACKEND = "gemini"
DEFAULT_SYNC_RAG_EMBEDDING_MODEL = DEFAULT_GEMINI_EMBEDDING_MODEL
DEFAULT_SYNC_RAG_EMBEDDING_BACKEND = embedding_runtime.BACKEND_GEMINI
DEFAULT_SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS = embedding_runtime.DEFAULT_TIMEOUT_SECONDS
DEFAULT_SYNC_SOURCE_INDEX_TOP_K = 4
DEFAULT_SYNC_SOURCE_INDEX_MIN_SIMILARITY = 0.72
DEFAULT_SYNC_SOURCE_INDEX_CHAR_LIMIT = 220
DEFAULT_SYNC_CONTEXT_BEFORE = 30
DEFAULT_SYNC_CONTEXT_AFTER = 10
DEFAULT_SYNC_MACRO_SETTING_FILE = "macro_setting.md"
# Rotation policy defaults: multi-key failover stays on; model hopping is off.
DEFAULT_API_KEY_ROTATION_ENABLED = True
DEFAULT_MODEL_ROTATION_ENABLED = False
DEFAULT_SYNC_RAG_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
DEFAULT_SYNC_RAG_DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
DEFAULT_SYNC_RAG_OUTPUT_DIMENSIONALITY = 768
DEFAULT_SYNC_RAG_TOP_K_HISTORY = 4
DEFAULT_SYNC_RAG_TOP_K_TERMS = 8
DEFAULT_SYNC_RAG_MIN_SIMILARITY = 0.72
DEFAULT_SYNC_RAG_SEGMENT_LINES = 4
DEFAULT_SYNC_RAG_HISTORY_CHAR_LIMIT = 220
DEFAULT_SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = translation_core.CANONICAL_STORY_CHAR_LIMIT
DEFAULT_SYNC_STORY_MEMORY_TOP_K_RELATIONS = 4
DEFAULT_SYNC_STORY_MEMORY_TOP_K_TERMS = 8

TL_SUBDIR = DEFAULT_TL_SUBDIR
# Empty until GAME_ROOT / translator_config.game_root is set — never invent parent of tool.
BASE_DIR = os.path.abspath(ENV_GAME_ROOT) if ENV_GAME_ROOT else ""
TL_DIR = os.path.abspath(os.path.join(BASE_DIR, TL_SUBDIR)) if BASE_DIR else ""
WORK_GAME_SUBDIR = "game"
WORK_GAME_DIR = (
    os.path.abspath(os.path.join(BASE_DIR, WORK_GAME_SUBDIR)) if BASE_DIR else ""
)
CONTEXT_STORAGE_LOCATION = DEFAULT_CONTEXT_STORAGE_LOCATION
CONTEXT_STORAGE_GAME_DIR_NAME = DEFAULT_CONTEXT_STORAGE_GAME_DIR_NAME
SOURCE_GAME_DIR = ""
PREP_ENABLED = True
PREP_UNPACK_RPA = True
PREP_GENERATE_TEMPLATE = True
PREP_REFRESH_EXISTING_TEMPLATE = True
PREP_LANGUAGE = DEFAULT_PREP_LANGUAGE
PREP_RENPY_SDK_DIR = ""
PREP_LAUNCHER_PY = ""
PREP_PYTHON_EXE = ""
PREP_UNPACK_COMMAND = None
PREP_TEMPLATE_COMMAND = None
# String prepare commands run with shell=True only when this opt-in is set.
PREP_ALLOW_SHELL_COMMANDS = False
LOG_DIR = os.path.join(ROOT_DIR, "logs")
FAILED_LOG = os.path.join(LOG_DIR, "translation_failures_v2.jsonl")
PROGRESS_LOG = os.path.join(LOG_DIR, "translation_progress_v2.json")
CONSOLE_LOG = os.path.join(LOG_DIR, "translation_console_output.log")
GENAI_MODULE = None

class DualLogger(object):
    """Duplicates stdout to a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Model definitions (Priority Order)
MODELS = list(DEFAULT_MODELS)

# Runtime State
CURRENT_KEY_INDEX = 0
CURRENT_MODEL_INDEX = 0
API_KEYS = []
API_KEY_ROTATION_ENABLED = DEFAULT_API_KEY_ROTATION_ENABLED
MODEL_ROTATION_ENABLED = DEFAULT_MODEL_ROTATION_ENABLED
# When model rotation is enabled and this list is non-empty, rotate only within it.
MODEL_ROTATION_MODELS: list[str] = []

PRESERVE_TERMS = [
    "???", "????", "?????", "[name]", "{sc=4}???{/sc}",
]

# Config Defaults
MAX_CHARS = DEFAULT_MAX_CHARS
MAX_ITEMS = DEFAULT_MAX_ITEMS
SYNC_MAX_OUTPUT_TOKENS = DEFAULT_SYNC_MAX_OUTPUT_TOKENS
SYNC_TIMEOUT_SECONDS = DEFAULT_SYNC_TIMEOUT_SECONDS
SYNC_BACKEND = DEFAULT_SYNC_BACKEND
SYNC_CONTEXT_BEFORE = DEFAULT_SYNC_CONTEXT_BEFORE
SYNC_CONTEXT_AFTER = DEFAULT_SYNC_CONTEXT_AFTER
SYNC_MACRO_SETTING_FILE = DEFAULT_SYNC_MACRO_SETTING_FILE
SYNC_MACRO_SETTING = ''
SYNC_MACRO_FINGERPRINT = ''
# Custom OpenAI-compatible LiteLLM providers parsed from sync.custom_litellm_providers.
CUSTOM_LITELLM_PROVIDERS: dict[str, object] = {}
MIN_DELAY = 1.0  # Reduced delay for SDK
MAX_DELAY = 3.0
BATCH_RETRIES = 3

FORCE_RETRANSLATE_ENGLISH = True
ALLOW_SINGLE_WORD_TRANSLATION = True
USE_TRANSLATION_MEMORY = True

# Optional RAG support for synchronous translation. Disabled by default so the
# sync script remains a lightweight repair/smoke-test path unless configured.
SYNC_RAG_ENABLED = False
SYNC_RAG_STORE_DIR = ""
SYNC_RAG_EMBEDDING_MODEL = DEFAULT_SYNC_RAG_EMBEDDING_MODEL
SYNC_RAG_EMBEDDING_BACKEND = DEFAULT_SYNC_RAG_EMBEDDING_BACKEND
SYNC_RAG_EMBEDDING_PROVIDER = ""
SYNC_RAG_EMBEDDING_ENDPOINT = ""
SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS = DEFAULT_SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
SYNC_RAG_EMBEDDING_API_KEY_ENV = ""
SYNC_RAG_EMBEDDING_LOAD_ERROR = ""
SYNC_RAG_QUERY_TASK_TYPE = DEFAULT_SYNC_RAG_QUERY_TASK_TYPE
SYNC_RAG_DOCUMENT_TASK_TYPE = DEFAULT_SYNC_RAG_DOCUMENT_TASK_TYPE
SYNC_RAG_OUTPUT_DIMENSIONALITY = DEFAULT_SYNC_RAG_OUTPUT_DIMENSIONALITY
SYNC_RAG_TOP_K_HISTORY = DEFAULT_SYNC_RAG_TOP_K_HISTORY
SYNC_RAG_TOP_K_TERMS = DEFAULT_SYNC_RAG_TOP_K_TERMS
SYNC_RAG_MIN_SIMILARITY = DEFAULT_SYNC_RAG_MIN_SIMILARITY
SYNC_RAG_SEGMENT_LINES = DEFAULT_SYNC_RAG_SEGMENT_LINES
SYNC_RAG_HISTORY_CHAR_LIMIT = DEFAULT_SYNC_RAG_HISTORY_CHAR_LIMIT
SYNC_RAG_UPDATE_ON_SUCCESS = True
SYNC_RAG_QUALITY_STATE = "sync_applied"
_SYNC_RAG_STORE = None
SYNC_SOURCE_INDEX_ENABLED = False
SYNC_SOURCE_INDEX_STORE_DIR = ""
SYNC_SOURCE_INDEX_TOP_K = DEFAULT_SYNC_SOURCE_INDEX_TOP_K
SYNC_SOURCE_INDEX_MIN_SIMILARITY = DEFAULT_SYNC_SOURCE_INDEX_MIN_SIMILARITY
SYNC_SOURCE_INDEX_CHAR_LIMIT = DEFAULT_SYNC_SOURCE_INDEX_CHAR_LIMIT
SYNC_SOURCE_INDEX_CHAR_BUDGET = advanced_context.DEFAULT_SOURCE_INDEX_CHAR_BUDGET
_SYNC_SOURCE_INDEX_STORE = None
SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = False

# Optional structured story memory for synchronous translation. Disabled by
# default to keep sync repair and smoke-test runs lightweight unless configured.
SYNC_STORY_MEMORY_ENABLED = False
SYNC_STORY_MEMORY_GRAPH_FILE = ""
SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = DEFAULT_SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS
SYNC_STORY_MEMORY_TOP_K_RELATIONS = DEFAULT_SYNC_STORY_MEMORY_TOP_K_RELATIONS
SYNC_STORY_MEMORY_TOP_K_TERMS = DEFAULT_SYNC_STORY_MEMORY_TOP_K_TERMS
SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = True
_SYNC_STORY_GRAPH = None
_SYNC_STORY_GRAPH_PATH = ""

# Optional allowlist to limit which files are processed (relative to TL_DIR).
INCLUDE_FILES = set()
INCLUDE_PREFIXES = set()

NORMALIZE_TRANSLATION_MAP = {}
PRESERVE_TERM_ALIASES = {
    "H.U.": ("H. U.", "Highwell University", "Highwell Uni"),
    "H. U.": ("H.U.", "Highwell University", "Highwell Uni"),
}

SPECIAL_ESCAPES = [
    ("\\", "\\\\"),
    ("\"", "\\\""),
    ("\a", "\\a"),
    ("\b", "\\b"),
    ("\f", "\\f"),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
    ("\v", "\\v"),
]

NON_TRANSLATABLE_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\\.", re.IGNORECASE),
]

BUILTIN_NON_TRANSLATABLE_EXACT = {
    "Esc",
    "Ctrl",
    "Shift",
    "Tab",
    "Enter",
    "Space",
    "Left",
    "Right",
    "Up",
    "Down",
    "Caps",
    "Page Up",
    "Page Down",
    "Home",
    "End",
    "Insert",
    "Delete",
    "Backspace",
    "DejaVu Sans",
    "Opendyslexic",
}
NON_TRANSLATABLE_EXACT = set(BUILTIN_NON_TRANSLATABLE_EXACT)

NON_TRANSLATABLE_TAG_ONLY = re.compile(r"^\{[^}]+\}$")
NON_TRANSLATABLE_SYMBOLS = re.compile(r"^[^A-Za-z0-9\u4e00-\u9fff]+$")
RENPY_TAG_RE = re.compile(r"\{[^}]*\}")
RENPY_FIELD_RE = re.compile(r"\[[^\]]+\]")
RENPY_FIELD_TOKEN_RE = re.compile(r"\[(?P<name>[^\]!:]+)(?:![^\]]*)?\]")
PERCENT_FORMAT_TOKEN_RE = re.compile(
    r"%(?:\([^)]+\))?[#0 +\-]*(?:\d+|\*)?(?:\.\d+|\.\*)?[hlL]?"
    r"[diouxXeEfFgGcrsa](?![A-Za-z])"
)
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+")
VOWEL_RE = re.compile(r"[aeiou]", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(.)\\1{2,}")
STUTTER_PATTERN = re.compile(r"\b\w-\w", re.IGNORECASE)
MULTI_DOT_PATTERN = re.compile(r"(\.{2,}|…{2,})")
# Matches sequences like "A B C" or "A. B. C." (single-letter tokens only)
LETTER_SEQUENCE_RE = re.compile(r"^(?:[A-Za-z]\.?)(?:\s+[A-Za-z]\.?)+$")
FILE_NAME_SIMPLE_RE = re.compile(r"^[\w.-]+\.\w+$", re.IGNORECASE)
PRESERVE_TERM_SOURCE_EXCLUSION_PATTERNS = {
    "Mark": [re.compile(r"\bMark my words\b", re.IGNORECASE)],
}
ROMAN_NUMERAL_LABEL_RE = re.compile(r"^(?:[+-][IVXLCDM]+|[IVXLCDM]{2,})$", re.IGNORECASE)
STRFTIME_FORMAT_RE = re.compile(r"^(?:%[A-Za-z]|[%:\s,./\-0-9])+$")
RENPY_IDENTIFIER_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:_name|_label|_id)$")
STRING_LITERAL_PREFIX_RE = re.compile(r"(?is)^(?P<prefix>[rubf]*)(?P<quote>'''|\"\"\"|'|\")")
TL_COMMENT_SOURCE_RE = re.compile(r'^\s*#\s*(?P<prefix>[^\"]*?)"(?P<text>.*)"\s*$')
TL_OLD_LINE_RE = re.compile(r'^\s*old\s+"(?P<text>.*)"\s*$')
TL_NEW_LINE_RE = re.compile(r'^\s*new\s+"(?P<text>.*)"\s*$')
CHARACTER_DEFINE_RE = re.compile(
    r"^\s*define\s+(?P<speaker>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<call>Character\s*\(.*)"
)
CHARACTER_DISPLAY_ASSET_RE = re.compile(
    r"^[\w./\\-]+\.(png|jpg|jpeg|bmp|gif|webp|ogg|mp3|wav|webm|mp4|avi|txt|json|rpy)$",
    re.IGNORECASE,
)
CHARACTER_DISPLAY_SYMBOLS_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
RENPY_NON_SPEAKER_NAMES = {
    "_",
    "call",
    "default",
    "define",
    "elif",
    "else",
    "extend",
    "hide",
    "if",
    "image",
    "init",
    "jump",
    "label",
    "menu",
    "old",
    "python",
    "renpy",
    "return",
    "scene",
    "screen",
    "set",
    "show",
    "text",
    "translate",
    "voice",
    "window",
    "with",
}
PRESERVE_TERMS_LOWER = {term.lower() for term in PRESERVE_TERMS}
FILE_EXTENSIONS = (
    "png", "jpg", "jpeg", "bmp", "gif", "webp", "txt", "pdf", "mp3", "wav", "ogg", "zip"
)
FILE_EXTENSION_PATTERN = "|".join(FILE_EXTENSIONS)
FILE_NAME_PATTERN = re.compile(rf"^[\w.-]+\.({FILE_EXTENSION_PATTERN})$", re.IGNORECASE)
EFFECT_MAX_LENGTH = 12


def initialize_runtime_logging():
    if isinstance(sys.stdout, DualLogger):
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    sys.stdout = DualLogger(CONSOLE_LOG)


def get_genai_module():
    global GENAI_MODULE
    if GENAI_MODULE is None:
        try:
            from google import genai as imported_genai
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: google-genai. Install with `pip install google-genai`."
            ) from exc
        GENAI_MODULE = imported_genai
    return GENAI_MODULE


def _normalize_rel_path(value):
    if not value:
        return ""
    value = str(value).replace("\\", "/").strip()
    value = value.lstrip("./")
    value = value.lstrip("/")
    return value


class InvalidTlSubdirError(ValueError):
    """Raised when ``tl_subdir`` is not a safe project-relative path."""


def normalize_tl_subdir(value):
    """Normalize and validate ``translator_config.json`` ``tl_subdir``.

    Accepts only relative paths that stay under the project root after join:
    absolute paths, drive-qualified paths, UNC paths, and any ``.`` / ``..``
    segment are rejected. Returns a forward-slash relative path.
    """
    if value is None:
        raise InvalidTlSubdirError("tl_subdir is missing")

    original = str(value).strip()
    if not original:
        raise InvalidTlSubdirError("tl_subdir is empty")

    text = original.replace("\\", "/")

    # Reject absolute / drive / UNC before stripping roots. On Windows,
    # os.path.isabs('/tmp/...') can be False, so also reject a leading '/'.
    if (
        os.path.isabs(original)
        or text.startswith("/")
        or re.match(r"^[A-Za-z]:", text)
    ):
        raise InvalidTlSubdirError(
            f"tl_subdir must be a relative path inside the game root, not absolute: {original!r}"
        )

    while text.startswith("./"):
        text = text[2:]
    if not text or text.startswith("/"):
        raise InvalidTlSubdirError(
            f"tl_subdir is empty or absolute after normalization: {original!r}"
        )

    parts = [part for part in text.split("/") if part]
    if not parts:
        raise InvalidTlSubdirError(
            f"tl_subdir is empty after normalization: {original!r}"
        )
    if any(part in {".", ".."} for part in parts):
        raise InvalidTlSubdirError(
            "tl_subdir must not contain '.' or '..' path segments "
            f"(got {original!r}). Use a path relative to game_root such as 'game/tl/schinese'."
        )
    return "/".join(parts)


def ensure_tl_dir_within_base(base_dir, tl_dir, *, tl_subdir=None):
    """Require the resolved TL directory to remain inside the project base dir."""
    if _path_contains_path(base_dir, tl_dir):
        return
    detail = f", tl_subdir={tl_subdir!r}" if tl_subdir is not None else ""
    raise InvalidTlSubdirError(
        "TL_DIR must remain inside BASE_DIR"
        f"{detail}: base_dir={base_dir!r}, tl_dir={tl_dir!r}"
    )


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _coerce_str_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    cleaned = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _is_placeholder_api_key(value):
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return True
    placeholder_markers = (
        "your-key",
        "your api key",
        "your-api-key",
        "your_gemini_api_key",
        "your-gemini-api-key",
        "paste-key",
        "paste-api-key",
        "replace-me",
    )
    return any(marker in text for marker in placeholder_markers)


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_positive_int(value, default):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_non_negative_int(value, default):
    """Coerce an integer that may be zero (0 disables the feature)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_non_empty_string(value, default):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_task_type(value, default):
    return embedding_runtime.persist_task_type(value, default)


def _normalize_context_storage_location(value):
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"game", "game_dir", "game_directory"}:
            return "game"
        if normalized in {"tool", "project", "repo", "repository", "internal"}:
            return "tool"
    return "tool"


def _normalize_context_storage_dir_name(value):
    if not isinstance(value, str):
        return "translation_context"
    stripped = value.strip()
    if os.path.isabs(stripped):
        return "translation_context"
    raw = stripped.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", raw) or raw.startswith("//"):
        return "translation_context"
    text = raw.strip("/")
    if not text:
        return "translation_context"
    parts = [part for part in text.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        return "translation_context"
    return "/".join(parts)


def load_context_storage_settings(config):
    global CONTEXT_STORAGE_LOCATION, CONTEXT_STORAGE_GAME_DIR_NAME
    storage = config.get("context_storage")
    if not isinstance(storage, dict):
        storage = {}
    location = storage.get("location", config.get("context_storage_location"))
    CONTEXT_STORAGE_LOCATION = _normalize_context_storage_location(location)
    dir_name = storage.get("game_dir_name", storage.get("directory_name", storage.get("directory")))
    CONTEXT_STORAGE_GAME_DIR_NAME = _normalize_context_storage_dir_name(dir_name)


def _resolve_path(base_dir, value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return _canonical_abs_path(text)
    # Relative paths need an explicit base; never fall back to process CWD.
    if not base_dir:
        return ""
    return _canonical_abs_path(os.path.join(base_dir, text))


def _canonical_path_within(base_dir, candidate):
    """Return True when *candidate* resolves inside *base_dir* (symlink-safe)."""
    if not base_dir or not candidate:
        return False
    try:
        base = os.path.realpath(os.path.abspath(base_dir))
        path = os.path.realpath(os.path.abspath(candidate))
    except OSError:
        return False
    base_norm = os.path.normcase(base).rstrip("\\/")
    path_norm = os.path.normcase(path)
    return path_norm == base_norm or path_norm.startswith(base_norm + os.sep)


def _resolve_preferred_path_from_bases(value, base_dirs):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return os.path.abspath(text)

    candidates = []
    for base_dir in base_dirs:
        if not base_dir:
            continue
        candidate = os.path.abspath(os.path.join(base_dir, text))
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


def _resolve_preferred_path(primary_base_dir, secondary_base_dir, value):
    return _resolve_preferred_path_from_bases(value, (primary_base_dir, secondary_base_dir))


def _is_filesystem_root(path):
    if not path:
        return False
    normalized = os.path.abspath(path)
    parent = os.path.dirname(normalized)
    return normalized == parent


def _parse_renpy_sdk_version(path):
    name = os.path.basename(os.path.abspath(path)).lower()
    match = re.search(r"renpy[-_](\d+(?:\.\d+)*)", name)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _renpy_sdk_sort_key(path):
    normalized = os.path.abspath(path)
    return (
        _parse_renpy_sdk_version(normalized),
        os.path.basename(normalized).lower(),
        normalized.lower(),
    )


def is_renpy_sdk_dir(path):
    """Return True when *path* looks like a Ren'Py SDK root (contains renpy.py)."""
    return bool(
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "renpy.py"))
    )


# Keep private alias used by older tests / internal call sites.
_is_renpy_sdk_dir = is_renpy_sdk_dir


def renpy_sdk_search_roots(
    *,
    game_root: str | None = None,
    tool_root: str | None = None,
    workspace_root: str | None = None,
    include_runtime_defaults: bool = True,
) -> list[str]:
    """Build de-duplicated directory list used when looking for a Ren'Py SDK.

    Roots are only *search bases* (each folder and its immediate children are
    scanned for ``renpy-*-sdk``). This is not the multi-project workspace
    contract; callers that want a specific workspace may pass
    ``workspace_root`` explicitly.
    """
    roots: list[str] = []

    def _add(path: str | None) -> None:
        if not path:
            return
        text = os.path.abspath(str(path))
        if text and text not in roots:
            roots.append(text)

    def _add_with_parents(path: str | None, *, parents: int = 0) -> None:
        if not path:
            return
        current = os.path.abspath(str(path))
        _add(current)
        for _ in range(max(0, parents)):
            parent = os.path.dirname(current)
            if not parent or parent == current:
                break
            _add(parent)
            current = parent

    if include_runtime_defaults:
        # Match historical prepare discovery: game work dir + parents, tool/root.
        if not _is_filesystem_root(BASE_DIR):
            _add_with_parents(BASE_DIR, parents=2)
        _add(ROOT_DIR)
        if ROOT_DIR and not _is_filesystem_root(ROOT_DIR):
            _add(os.path.dirname(ROOT_DIR))
        _add(TOOL_DIR)

    _add_with_parents(game_root, parents=2)
    _add(tool_root)
    if tool_root and not _is_filesystem_root(tool_root):
        _add(os.path.dirname(os.path.abspath(str(tool_root))))
    _add(workspace_root)

    return [path for path in roots if path and os.path.isdir(path)]


def discover_renpy_sdk_candidates(
    search_roots: list[str] | None = None,
    *,
    game_root: str | None = None,
    tool_root: str | None = None,
    workspace_root: str | None = None,
    include_runtime_defaults: bool = True,
) -> list[str]:
    """Return valid Ren'Py SDK directories under *search_roots*, newest first.

    When *search_roots* is omitted, roots are derived from runtime globals and
    any explicit ``game_root`` / ``tool_root`` / ``workspace_root`` arguments.
    """
    if search_roots is None:
        roots = renpy_sdk_search_roots(
            game_root=game_root,
            tool_root=tool_root,
            workspace_root=workspace_root,
            include_runtime_defaults=include_runtime_defaults,
        )
    else:
        roots = _dedupe_keep_order(
            [os.path.abspath(str(path)) for path in search_roots if path]
        )

    found: list[str] = []
    for base_dir in roots:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        if is_renpy_sdk_dir(base_dir):
            found.append(base_dir)
        for pattern in ("renpy-*-sdk", "renpy-*sdk", "renpy-sdk"):
            found.extend(glob.glob(os.path.join(base_dir, pattern)))

    return sorted(
        {
            os.path.abspath(candidate)
            for candidate in found
            if is_renpy_sdk_dir(candidate)
        },
        key=_renpy_sdk_sort_key,
        reverse=True,
    )


def discover_renpy_sdk_dir(
    search_roots: list[str] | None = None,
    *,
    game_root: str | None = None,
    tool_root: str | None = None,
    workspace_root: str | None = None,
    include_runtime_defaults: bool = True,
) -> str:
    """Return the best matching Ren'Py SDK directory, or empty string."""
    candidates = discover_renpy_sdk_candidates(
        search_roots,
        game_root=game_root,
        tool_root=tool_root,
        workspace_root=workspace_root,
        include_runtime_defaults=include_runtime_defaults,
    )
    return candidates[0] if candidates else ""


def _discover_renpy_sdk_dir():
    """Explicit-scan helper (GUI「查找 SDK」/ tests). Not used by prepare load.

    Prepare never calls this; empty ``renpy_sdk_dir`` stays empty until the user
    configures a path or runs interactive discovery.
    """
    if _is_filesystem_root(BASE_DIR):
        return ""
    return discover_renpy_sdk_dir(include_runtime_defaults=True)


def resolve_story_memory_graph_path(value):
    return _resolve_preferred_path_from_bases(value, (ROOT_DIR, BASE_DIR, TOOL_DIR))


class InvalidPrepareCommandError(ValueError):
    """Raised when a prepare command is malformed or shell mode is not allowed."""


def _coerce_command(value, *, field_name="command", allow_shell=False):
    """Normalize a prepare command to an argv list or (opt-in) shell string.

    Preferred form is a JSON/argv list such as ``["python", "unpack.py"]``.
    Plain strings are executed with ``shell=True`` only when *allow_shell* is
    true (``prepare.allow_shell_commands``).
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not allow_shell:
            raise InvalidPrepareCommandError(
                f"{field_name} is a shell string, but prepare.allow_shell_commands "
                "is not enabled. Prefer an argv list "
                '(example: ["python", "script.py", "{base_dir}"]) or set '
                "prepare.allow_shell_commands to true for trusted local configs only."
            )
        return text
    if isinstance(value, list):
        cmd = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, (str, int, float)):
                raise InvalidPrepareCommandError(
                    f"{field_name} argv entries must be strings "
                    f"(got {type(item).__name__})."
                )
            token = str(item).strip()
            if token:
                cmd.append(token)
        return cmd or None
    raise InvalidPrepareCommandError(
        f"{field_name} must be an argv list or (with allow_shell_commands) a string; "
        f"got {type(value).__name__}."
    )


def describe_prepare_command(command):
    """Return a single-line display string for logs and doctor reports."""
    if command is None:
        return ""
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        try:
            return subprocess.list2cmdline([str(part) for part in command])
        except Exception:
            return " ".join(str(part) for part in command)
    return str(command)


def prepare_command_uses_shell(command):
    return isinstance(command, str)


def refresh_derived_terms():
    global PRESERVE_TERMS_LOWER
    PRESERVE_TERMS_LOWER = {term.lower() for term in PRESERVE_TERMS if isinstance(term, str)}


def load_glossary():
    """Loads glossary terms from glossary.json (separate from API keys)."""
    global PRESERVE_TERMS, NON_TRANSLATABLE_EXACT, NORMALIZE_TRANSLATION_MAP

    if not os.path.exists(GLOSSARY_FILE):
        print(f"Glossary file not found: {GLOSSARY_FILE} (using defaults).")
        refresh_derived_terms()
        return

    try:
        with open(GLOSSARY_FILE, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle) or {}
    except Exception as e:
        print(f"Warning: Failed to load glossary file: {e}")
        refresh_derived_terms()
        return

    preserve_terms = _coerce_str_list(data.get("preserve_terms"))
    if preserve_terms:
        PRESERVE_TERMS = _dedupe_keep_order(preserve_terms)
        print(f"Loaded {len(PRESERVE_TERMS)} preserve terms from glossary.")
    elif "preserve_terms" in data:
        print("Warning: glossary.json preserve_terms is empty; using defaults.")

    non_translatable = _coerce_str_list(data.get("non_translatable_exact"))
    if non_translatable:
        NON_TRANSLATABLE_EXACT = set(BUILTIN_NON_TRANSLATABLE_EXACT)
        NON_TRANSLATABLE_EXACT.update(non_translatable)
        print(
            f"Loaded {len(non_translatable)} non-translatable exact terms "
            f"(+{len(BUILTIN_NON_TRANSLATABLE_EXACT)} built-in)."
        )
    elif "non_translatable_exact" in data:
        NON_TRANSLATABLE_EXACT = set(BUILTIN_NON_TRANSLATABLE_EXACT)
        print("Warning: glossary.json non_translatable_exact is empty; using built-in defaults.")

    normalize_map = data.get("normalize_map")
    if isinstance(normalize_map, dict) and normalize_map:
        NORMALIZE_TRANSLATION_MAP = {str(k): str(v) for k, v in normalize_map.items()}
        print(f"Loaded {len(NORMALIZE_TRANSLATION_MAP)} normalization rules.")
    elif "normalize_map" in data:
        print("glossary.json normalize_map is empty; no normalization rules loaded.")

    refresh_derived_terms()


def load_sync_rag_settings(config):
    global SYNC_RAG_ENABLED, SYNC_RAG_STORE_DIR, SYNC_RAG_EMBEDDING_MODEL
    global SYNC_RAG_EMBEDDING_BACKEND, SYNC_RAG_EMBEDDING_PROVIDER
    global SYNC_RAG_EMBEDDING_ENDPOINT, SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
    global SYNC_RAG_EMBEDDING_API_KEY_ENV, SYNC_RAG_EMBEDDING_LOAD_ERROR
    global SYNC_RAG_QUERY_TASK_TYPE, SYNC_RAG_DOCUMENT_TASK_TYPE
    global SYNC_RAG_OUTPUT_DIMENSIONALITY, SYNC_RAG_TOP_K_HISTORY
    global SYNC_RAG_TOP_K_TERMS, SYNC_RAG_MIN_SIMILARITY, SYNC_RAG_SEGMENT_LINES
    global SYNC_RAG_HISTORY_CHAR_LIMIT, SYNC_RAG_UPDATE_ON_SUCCESS, _SYNC_RAG_STORE

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}
    rag = sync.get("rag")
    if not isinstance(rag, dict):
        rag = {}

    SYNC_RAG_ENABLED = _coerce_bool(rag.get("enabled"), False)
    SYNC_RAG_EMBEDDING_MODEL = _coerce_non_empty_string(
        rag.get("embedding_model"),
        DEFAULT_SYNC_RAG_EMBEDDING_MODEL,
    )
    try:
        embedding_settings = embedding_runtime.parse_embedding_runtime_settings(
            rag,
            default_model=DEFAULT_SYNC_RAG_EMBEDDING_MODEL,
        )
        SYNC_RAG_EMBEDDING_LOAD_ERROR = ""
    except EmbeddingContractError as exc:
        if embedding_runtime.is_explicit_non_gemini_backend(rag):
            raise SystemExit(
                f"ERROR: invalid sync.rag embedding settings: {exc}"
            ) from exc
        print(
            f"Warning: invalid sync.rag embedding settings ({exc}); "
            "using Gemini embedding defaults."
        )
        SYNC_RAG_EMBEDDING_LOAD_ERROR = str(exc)
        embedding_settings = embedding_runtime.parse_embedding_runtime_settings(
            {"embedding_model": SYNC_RAG_EMBEDDING_MODEL},
            default_model=DEFAULT_SYNC_RAG_EMBEDDING_MODEL,
        )
    SYNC_RAG_EMBEDDING_BACKEND = embedding_settings.backend
    SYNC_RAG_EMBEDDING_PROVIDER = embedding_settings.provider
    SYNC_RAG_EMBEDDING_ENDPOINT = embedding_settings.endpoint
    SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS = embedding_settings.timeout_seconds
    SYNC_RAG_EMBEDDING_API_KEY_ENV = embedding_settings.api_key_env
    SYNC_RAG_EMBEDDING_MODEL = embedding_settings.model
    SYNC_RAG_OUTPUT_DIMENSIONALITY = embedding_settings.output_dimension
    SYNC_RAG_QUERY_TASK_TYPE = embedding_settings.native_query_task_type
    SYNC_RAG_DOCUMENT_TASK_TYPE = embedding_settings.native_document_task_type
    SYNC_RAG_TOP_K_HISTORY = _coerce_positive_int(rag.get("top_k_history"), 4)
    SYNC_RAG_TOP_K_TERMS = _coerce_positive_int(rag.get("top_k_terms"), 8)
    SYNC_RAG_MIN_SIMILARITY = _coerce_float(rag.get("min_similarity"), 0.72)
    SYNC_RAG_SEGMENT_LINES = _coerce_positive_int(rag.get("segment_lines"), 4)
    SYNC_RAG_HISTORY_CHAR_LIMIT = _coerce_positive_int(
        rag.get("history_char_limit"),
        220,
    )
    SYNC_RAG_UPDATE_ON_SUCCESS = _coerce_bool(rag.get("update_on_success"), True)

    store_dir = rag.get("store_dir")
    if store_dir:
        SYNC_RAG_STORE_DIR = _resolve_path(BASE_DIR, store_dir)
    else:
        SYNC_RAG_STORE_DIR = ""
    _SYNC_RAG_STORE = None


def load_sync_source_index_settings(config):
    global SYNC_SOURCE_INDEX_ENABLED, SYNC_SOURCE_INDEX_STORE_DIR
    global SYNC_SOURCE_INDEX_TOP_K, SYNC_SOURCE_INDEX_MIN_SIMILARITY
    global SYNC_SOURCE_INDEX_CHAR_LIMIT, SYNC_SOURCE_INDEX_CHAR_BUDGET
    global _SYNC_SOURCE_INDEX_STORE

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}
    source_index = sync.get("source_index")
    if not isinstance(source_index, dict):
        source_index = {}
    SYNC_SOURCE_INDEX_ENABLED = _coerce_bool(source_index.get("enabled"), False)
    SYNC_SOURCE_INDEX_TOP_K = _coerce_positive_int(
        source_index.get("top_k"),
        DEFAULT_SYNC_SOURCE_INDEX_TOP_K,
    )
    SYNC_SOURCE_INDEX_MIN_SIMILARITY = _coerce_float(
        source_index.get("min_similarity"),
        DEFAULT_SYNC_SOURCE_INDEX_MIN_SIMILARITY,
    )
    SYNC_SOURCE_INDEX_CHAR_LIMIT = _coerce_positive_int(
        source_index.get("char_limit"),
        DEFAULT_SYNC_SOURCE_INDEX_CHAR_LIMIT,
    )
    SYNC_SOURCE_INDEX_CHAR_BUDGET = advanced_context.DEFAULT_SOURCE_INDEX_CHAR_BUDGET
    store_dir = source_index.get("store_dir")
    if store_dir:
        SYNC_SOURCE_INDEX_STORE_DIR = _resolve_path(BASE_DIR, store_dir)
    else:
        SYNC_SOURCE_INDEX_STORE_DIR = ""
    _SYNC_SOURCE_INDEX_STORE = None


def load_sync_project_analysis_settings(config):
    global SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}
    project_analysis = sync.get("project_analysis")
    if not isinstance(project_analysis, dict):
        project_analysis = {}
    SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = _coerce_bool(
        project_analysis.get("inject_published_brief"),
        False,
    )


def load_sync_story_memory_settings(config):
    global SYNC_STORY_MEMORY_ENABLED, SYNC_STORY_MEMORY_GRAPH_FILE
    global SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS, SYNC_STORY_MEMORY_TOP_K_RELATIONS
    global SYNC_STORY_MEMORY_TOP_K_TERMS, SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY
    global _SYNC_STORY_GRAPH, _SYNC_STORY_GRAPH_PATH

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}
    story_config = sync.get("story_memory")
    if not isinstance(story_config, dict):
        story_config = {}

    SYNC_STORY_MEMORY_ENABLED = _coerce_bool(story_config.get("enabled"), False)
    SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = _coerce_positive_int(
        story_config.get("max_context_chars"),
        DEFAULT_SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
    )
    SYNC_STORY_MEMORY_TOP_K_RELATIONS = _coerce_positive_int(
        story_config.get("top_k_relations"),
        4,
    )
    SYNC_STORY_MEMORY_TOP_K_TERMS = _coerce_positive_int(
        story_config.get("top_k_terms"),
        8,
    )
    SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = _coerce_bool(
        story_config.get("include_scene_summary"),
        True,
    )

    graph_file = story_config.get("graph_file")
    if graph_file:
        SYNC_STORY_MEMORY_GRAPH_FILE = resolve_story_memory_graph_path(graph_file)
    elif SYNC_STORY_MEMORY_ENABLED:
        SYNC_STORY_MEMORY_GRAPH_FILE = get_default_story_memory_graph_path()
    else:
        SYNC_STORY_MEMORY_GRAPH_FILE = ""
    _SYNC_STORY_GRAPH = None
    _SYNC_STORY_GRAPH_PATH = ""


def load_rotation_settings(config):
    """Load API-key / model rotation policy from translator_config.rotation."""
    global API_KEY_ROTATION_ENABLED, MODEL_ROTATION_ENABLED, MODEL_ROTATION_MODELS

    section = config.get("rotation") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        section = {}

    api_key_cfg = section.get("api_key")
    if not isinstance(api_key_cfg, dict):
        api_key_cfg = {}
    model_cfg = section.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    previous_key = API_KEY_ROTATION_ENABLED
    previous_model = MODEL_ROTATION_ENABLED
    previous_pool = list(MODEL_ROTATION_MODELS)

    API_KEY_ROTATION_ENABLED = _coerce_bool(
        api_key_cfg.get("enabled"),
        DEFAULT_API_KEY_ROTATION_ENABLED,
    )
    MODEL_ROTATION_ENABLED = _coerce_bool(
        model_cfg.get("enabled"),
        DEFAULT_MODEL_ROTATION_ENABLED,
    )
    from gemini_model_catalog import filter_gemini_rotation_models

    raw_pool = normalize_model_names(model_cfg.get("models"))
    MODEL_ROTATION_MODELS = filter_gemini_rotation_models(
        raw_pool,
        translator_config=config if isinstance(config, dict) else None,
        reject_unknown=False,
    )
    dropped = [name for name in raw_pool if name not in set(MODEL_ROTATION_MODELS)]
    if dropped:
        print(
            "Warning: ignored unknown model rotation entries (not in Gemini catalog): "
            + ", ".join(dropped)
        )

    if API_KEY_ROTATION_ENABLED != previous_key:
        state = "enabled" if API_KEY_ROTATION_ENABLED else "disabled"
        print(f"API key rotation: {state}")
    if MODEL_ROTATION_ENABLED != previous_model or MODEL_ROTATION_MODELS != previous_pool:
        if MODEL_ROTATION_ENABLED:
            pool = ", ".join(MODEL_ROTATION_MODELS) if MODEL_ROTATION_MODELS else "(active model list)"
            print(f"Model rotation: enabled; pool={pool}")
        else:
            print("Model rotation: disabled")


def load_sync_translation_settings(config):
    """Load sync translation settings and project-bound Macro content.

    Context limits accept non-negative integers and default to 30/10 when
    omitted. Macro files must resolve under ``BASE_DIR``; missing, unreadable,
    or rejected files clear the Macro text and its fingerprint while keeping
    the configured file name for diagnostics.
    """
    global MAX_ITEMS, MAX_CHARS, SYNC_MAX_OUTPUT_TOKENS, SYNC_TIMEOUT_SECONDS
    global SYNC_BACKEND
    global CUSTOM_LITELLM_PROVIDERS
    global SYNC_CONTEXT_BEFORE, SYNC_CONTEXT_AFTER
    global SYNC_MACRO_SETTING_FILE, SYNC_MACRO_SETTING, SYNC_MACRO_FINGERPRINT

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}

    backend_name = str(sync.get("backend") or "gemini").strip().lower()
    if backend_name not in {"gemini", "litellm"}:
        raise model_profile.ModelRoutingConfigError(
            f"Unsupported sync backend: {backend_name}. Choose 'gemini' or 'litellm'."
        )
    SYNC_BACKEND = backend_name

    from litellm_provider_config import custom_provider_registry

    try:
        CUSTOM_LITELLM_PROVIDERS = custom_provider_registry(
            sync.get("custom_litellm_providers")
        )
    except ValueError as exc:
        # Same degrade-and-warn contract as the GUI: an invalid custom-provider
        # entry must not make the whole sync path crash at config load.
        CUSTOM_LITELLM_PROVIDERS = {}
        print(
            "Warning: 已忽略无效的 sync.custom_litellm_providers 配置"
            f"（{exc}）；请修正 translator_config.json 后重试。",  # noqa: RUF001
            flush=True,
        )

    load_rotation_settings(config)

    custom_models = sync.get("models")
    single_model = sync.get("model")
    if isinstance(custom_models, str):
        custom_models = [custom_models]
    if custom_models:
        # Explicit sync.models list still defines the active model list.
        replace_model_list(custom_models, "sync")
    elif backend_name == "litellm":
        # LiteLLM keeps a tight model list (provider-specific IDs).
        if isinstance(single_model, str) and single_model.strip():
            replace_model_list([single_model], "sync")
    else:
        selected = (
            single_model.strip()
            if isinstance(single_model, str) and single_model.strip()
            else DEFAULT_GEMINI_TRANSLATION_MODEL
        )
        if MODEL_ROTATION_ENABLED:
            # Enabled: selected model first, then explicit pool or catalog builtins.
            pool_extras = (
                MODEL_ROTATION_MODELS
                if MODEL_ROTATION_MODELS
                else merge_model_lists(
                    DEFAULT_MODELS,
                    catalog_extra_models(config, kind="translation"),
                )
            )
            replace_model_list(
                merge_model_lists([selected], pool_extras),
                "model-rotation",
            )
        else:
            # Default: pin to the configured model; no automatic hopping.
            replace_model_list([selected], "sync")

    previous_items = MAX_ITEMS
    previous_chars = MAX_CHARS
    previous_output_tokens = SYNC_MAX_OUTPUT_TOKENS
    previous_timeout = SYNC_TIMEOUT_SECONDS
    previous_context_before = SYNC_CONTEXT_BEFORE
    previous_context_after = SYNC_CONTEXT_AFTER
    previous_macro_file = SYNC_MACRO_SETTING_FILE

    MAX_ITEMS = _coerce_positive_int(sync.get("chunk_size"), MAX_ITEMS)
    MAX_CHARS = _coerce_positive_int(
        sync.get("max_source_chars", sync.get("target_chars")),
        MAX_CHARS,
    )
    SYNC_MAX_OUTPUT_TOKENS = _coerce_positive_int(
        sync.get("max_output_tokens"),
        SYNC_MAX_OUTPUT_TOKENS,
    )
    SYNC_TIMEOUT_SECONDS = normalize_sync_timeout_seconds(
        sync.get("timeout_seconds"),
        DEFAULT_SYNC_TIMEOUT_SECONDS,
    )
    SYNC_CONTEXT_BEFORE = _coerce_non_negative_int(
        sync.get("context_before"),
        DEFAULT_SYNC_CONTEXT_BEFORE,
    )
    SYNC_CONTEXT_AFTER = _coerce_non_negative_int(
        sync.get("context_after"),
        DEFAULT_SYNC_CONTEXT_AFTER,
    )

    macro_setting_file = sync.get("macro_setting_file")
    if isinstance(macro_setting_file, str) and macro_setting_file.strip():
        SYNC_MACRO_SETTING_FILE = macro_setting_file.strip()
    else:
        SYNC_MACRO_SETTING_FILE = DEFAULT_SYNC_MACRO_SETTING_FILE
    resolved_macro = _resolve_path(BASE_DIR, SYNC_MACRO_SETTING_FILE)
    macro_text = ''
    if resolved_macro and not _canonical_path_within(BASE_DIR, resolved_macro):
        print(
            f"Warning: Sync macro setting file {resolved_macro} is outside the "
            "project; ignoring it (macro_setting.md must live under game_root).",
            flush=True,
        )
        resolved_macro = ''
    if resolved_macro and os.path.isfile(resolved_macro):
        try:
            with open(resolved_macro, 'r', encoding='utf-8-sig') as handle:
                macro_text = handle.read()
        except OSError as exc:
            print(
                f"Warning: Failed to read sync macro setting file {resolved_macro}: {exc}",
                flush=True,
            )
    SYNC_MACRO_SETTING = macro_text.strip() if macro_text else ''
    SYNC_MACRO_FINGERPRINT = sha256_text(SYNC_MACRO_SETTING) if SYNC_MACRO_SETTING else ''

    if MAX_ITEMS != previous_items:
        print(f"Using sync chunk size: {MAX_ITEMS}")
    if MAX_CHARS != previous_chars:
        print(f"Using sync max source chars: {MAX_CHARS}")
    if SYNC_MAX_OUTPUT_TOKENS != previous_output_tokens:
        print(f"Using sync max output tokens: {SYNC_MAX_OUTPUT_TOKENS}")
    if SYNC_TIMEOUT_SECONDS != previous_timeout:
        print(f"Using sync request timeout: {SYNC_TIMEOUT_SECONDS} seconds")
    if SYNC_CONTEXT_BEFORE != previous_context_before or SYNC_CONTEXT_AFTER != previous_context_after:
        print(
            f"Using sync context window: before={SYNC_CONTEXT_BEFORE}, after={SYNC_CONTEXT_AFTER}",
            flush=True,
        )
    if SYNC_MACRO_SETTING_FILE != previous_macro_file:
        print(
            f"Using sync macro setting file: {SYNC_MACRO_SETTING_FILE}",
            flush=True,
        )


def coerce_normalized_rel_path_set(value):
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []

    normalized = set()
    for item in values:
        path = _normalize_rel_path(item)
        if path:
            normalized.add(path)
    return normalized


def replace_model_list(values, label):
    global MODELS, CURRENT_MODEL_INDEX

    models = []
    for value in values:
        model = str(value).strip()
        if model:
            models.append(model)
    if not models:
        return False

    MODELS = models
    CURRENT_MODEL_INDEX = 0
    print(f"Using {label} model list: {MODELS}")
    return True


def load_include_filters_from_config(config):
    global INCLUDE_FILES, INCLUDE_PREFIXES

    if "include_files" in config:
        INCLUDE_FILES = coerce_normalized_rel_path_set(config.get("include_files"))
        print(f"Using include_files allowlist ({len(INCLUDE_FILES)}).")

    if "include_prefixes" in config:
        INCLUDE_PREFIXES = coerce_normalized_rel_path_set(config.get("include_prefixes"))
        print(f"Using include_prefixes allowlist ({len(INCLUDE_PREFIXES)}).")


@dataclass
class RuntimeConfig:
    """Fresh per-project/job runtime configuration.

    Loaders always start from :func:`default_runtime_config` (or an explicit
    base config) and then apply validated overrides. Module-level globals remain
    as a CLI compatibility facade via :func:`apply_runtime_config`.
    """

    env_game_root: str = ""
    base_dir: str = ""
    tl_subdir: str = field(default_factory=lambda: DEFAULT_TL_SUBDIR)
    tl_dir: str = ""
    work_game_dir: str = ""
    source_game_dir: str = ""
    glossary_file: str = field(default_factory=lambda: DEFAULT_GLOSSARY_FILE)

    prep_enabled: bool = True
    prep_unpack_rpa: bool = True
    prep_generate_template: bool = True
    prep_refresh_existing_template: bool = True
    prep_language: str = DEFAULT_PREP_LANGUAGE
    prep_renpy_sdk_dir: str = ""
    prep_launcher_py: str = ""
    prep_python_exe: str = ""
    prep_unpack_command: Any = None
    prep_template_command: Any = None
    prep_allow_shell_commands: bool = False

    context_storage_location: str = DEFAULT_CONTEXT_STORAGE_LOCATION
    context_storage_game_dir_name: str = DEFAULT_CONTEXT_STORAGE_GAME_DIR_NAME

    api_keys: list = field(default_factory=list)
    models: list = field(default_factory=lambda: list(DEFAULT_MODELS))
    current_key_index: int = 0
    current_model_index: int = 0
    api_key_rotation_enabled: bool = DEFAULT_API_KEY_ROTATION_ENABLED
    model_rotation_enabled: bool = DEFAULT_MODEL_ROTATION_ENABLED
    model_rotation_models: list = field(default_factory=list)

    max_chars: int = DEFAULT_MAX_CHARS
    max_items: int = DEFAULT_MAX_ITEMS
    sync_max_output_tokens: int = DEFAULT_SYNC_MAX_OUTPUT_TOKENS
    sync_timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS
    sync_backend: str = DEFAULT_SYNC_BACKEND
    sync_context_before: int = DEFAULT_SYNC_CONTEXT_BEFORE
    sync_context_after: int = DEFAULT_SYNC_CONTEXT_AFTER
    sync_macro_setting_file: str = DEFAULT_SYNC_MACRO_SETTING_FILE
    sync_macro_setting: str = ""
    sync_macro_fingerprint: str = ""
    custom_litellm_providers: dict = field(default_factory=dict)

    include_files: set = field(default_factory=set)
    include_prefixes: set = field(default_factory=set)

    sync_rag_enabled: bool = False
    sync_rag_store_dir: str = ""
    sync_rag_embedding_model: str = DEFAULT_SYNC_RAG_EMBEDDING_MODEL
    sync_rag_embedding_backend: str = DEFAULT_SYNC_RAG_EMBEDDING_BACKEND
    sync_rag_embedding_provider: str = ""
    sync_rag_embedding_endpoint: str = ""
    sync_rag_embedding_timeout_seconds: float = DEFAULT_SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
    sync_rag_embedding_api_key_env: str = ""
    sync_rag_embedding_load_error: str = ""
    sync_rag_query_task_type: str = DEFAULT_SYNC_RAG_QUERY_TASK_TYPE
    sync_rag_document_task_type: str = DEFAULT_SYNC_RAG_DOCUMENT_TASK_TYPE
    sync_rag_output_dimensionality: int = DEFAULT_SYNC_RAG_OUTPUT_DIMENSIONALITY
    sync_rag_top_k_history: int = DEFAULT_SYNC_RAG_TOP_K_HISTORY
    sync_rag_top_k_terms: int = DEFAULT_SYNC_RAG_TOP_K_TERMS
    sync_rag_min_similarity: float = DEFAULT_SYNC_RAG_MIN_SIMILARITY
    sync_rag_segment_lines: int = DEFAULT_SYNC_RAG_SEGMENT_LINES
    sync_rag_history_char_limit: int = DEFAULT_SYNC_RAG_HISTORY_CHAR_LIMIT
    sync_rag_update_on_success: bool = True
    sync_source_index_enabled: bool = False
    sync_source_index_store_dir: str = ""
    sync_source_index_top_k: int = DEFAULT_SYNC_SOURCE_INDEX_TOP_K
    sync_source_index_min_similarity: float = DEFAULT_SYNC_SOURCE_INDEX_MIN_SIMILARITY
    sync_source_index_char_limit: int = DEFAULT_SYNC_SOURCE_INDEX_CHAR_LIMIT
    sync_project_analysis_inject_published_brief: bool = False

    sync_story_memory_enabled: bool = False
    sync_story_memory_graph_file: str = ""
    sync_story_memory_max_context_chars: int = DEFAULT_SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS
    sync_story_memory_top_k_relations: int = DEFAULT_SYNC_STORY_MEMORY_TOP_K_RELATIONS
    sync_story_memory_top_k_terms: int = DEFAULT_SYNC_STORY_MEMORY_TOP_K_TERMS
    sync_story_memory_include_scene_summary: bool = True

    def copy(self) -> "RuntimeConfig":
        """Return a deep-ish copy so nested containers are not shared."""
        return replace(
            self,
            api_keys=list(self.api_keys),
            models=list(self.models),
            model_rotation_models=list(self.model_rotation_models),
            custom_litellm_providers=dict(self.custom_litellm_providers or {}),
            include_files=set(self.include_files),
            include_prefixes=set(self.include_prefixes),
            prep_unpack_command=(
                list(self.prep_unpack_command)
                if isinstance(self.prep_unpack_command, list)
                else self.prep_unpack_command
            ),
            prep_template_command=(
                list(self.prep_template_command)
                if isinstance(self.prep_template_command, list)
                else self.prep_template_command
            ),
        )


# Alias used by the issue description / external docs.
ProjectContext = RuntimeConfig


def default_runtime_config() -> RuntimeConfig:
    """Build a fresh config object with code defaults (no file I/O)."""
    default_base = os.path.abspath(ENV_GAME_ROOT) if ENV_GAME_ROOT else ""
    tl_subdir = DEFAULT_TL_SUBDIR
    return RuntimeConfig(
        env_game_root=ENV_GAME_ROOT or "",
        base_dir=default_base,
        tl_subdir=tl_subdir,
        tl_dir=(
            os.path.abspath(os.path.join(default_base, tl_subdir)) if default_base else ""
        ),
        work_game_dir=(
            os.path.abspath(os.path.join(default_base, WORK_GAME_SUBDIR))
            if default_base
            else ""
        ),
        glossary_file=DEFAULT_GLOSSARY_FILE,
        models=list(DEFAULT_MODELS),
        max_chars=DEFAULT_MAX_CHARS,
        max_items=DEFAULT_MAX_ITEMS,
        sync_max_output_tokens=DEFAULT_SYNC_MAX_OUTPUT_TOKENS,
        sync_timeout_seconds=DEFAULT_SYNC_TIMEOUT_SECONDS,
        sync_backend=DEFAULT_SYNC_BACKEND,
        sync_context_before=DEFAULT_SYNC_CONTEXT_BEFORE,
        sync_context_after=DEFAULT_SYNC_CONTEXT_AFTER,
        sync_macro_setting_file=DEFAULT_SYNC_MACRO_SETTING_FILE,
        sync_macro_setting="",
        sync_macro_fingerprint="",
        custom_litellm_providers={},
        prep_language=DEFAULT_PREP_LANGUAGE,
        context_storage_location=DEFAULT_CONTEXT_STORAGE_LOCATION,
        context_storage_game_dir_name=DEFAULT_CONTEXT_STORAGE_GAME_DIR_NAME,
    )


def snapshot_runtime_config() -> RuntimeConfig:
    """Capture the current module-level globals into a RuntimeConfig object."""
    return RuntimeConfig(
        env_game_root=ENV_GAME_ROOT or "",
        base_dir=BASE_DIR,
        tl_subdir=TL_SUBDIR,
        tl_dir=TL_DIR,
        work_game_dir=WORK_GAME_DIR,
        source_game_dir=SOURCE_GAME_DIR,
        glossary_file=GLOSSARY_FILE,
        prep_enabled=PREP_ENABLED,
        prep_unpack_rpa=PREP_UNPACK_RPA,
        prep_generate_template=PREP_GENERATE_TEMPLATE,
        prep_refresh_existing_template=PREP_REFRESH_EXISTING_TEMPLATE,
        prep_language=PREP_LANGUAGE,
        prep_renpy_sdk_dir=PREP_RENPY_SDK_DIR,
        prep_launcher_py=PREP_LAUNCHER_PY,
        prep_python_exe=PREP_PYTHON_EXE,
        prep_unpack_command=(
            list(PREP_UNPACK_COMMAND)
            if isinstance(PREP_UNPACK_COMMAND, list)
            else PREP_UNPACK_COMMAND
        ),
        prep_template_command=(
            list(PREP_TEMPLATE_COMMAND)
            if isinstance(PREP_TEMPLATE_COMMAND, list)
            else PREP_TEMPLATE_COMMAND
        ),
        prep_allow_shell_commands=PREP_ALLOW_SHELL_COMMANDS,
        context_storage_location=CONTEXT_STORAGE_LOCATION,
        context_storage_game_dir_name=CONTEXT_STORAGE_GAME_DIR_NAME,
        api_keys=list(API_KEYS),
        models=list(MODELS),
        current_key_index=CURRENT_KEY_INDEX,
        current_model_index=CURRENT_MODEL_INDEX,
        api_key_rotation_enabled=API_KEY_ROTATION_ENABLED,
        model_rotation_enabled=MODEL_ROTATION_ENABLED,
        model_rotation_models=list(MODEL_ROTATION_MODELS),
        max_chars=MAX_CHARS,
        max_items=MAX_ITEMS,
        sync_max_output_tokens=SYNC_MAX_OUTPUT_TOKENS,
        sync_timeout_seconds=SYNC_TIMEOUT_SECONDS,
        sync_backend=SYNC_BACKEND,
        sync_context_before=SYNC_CONTEXT_BEFORE,
        sync_context_after=SYNC_CONTEXT_AFTER,
        sync_macro_setting_file=SYNC_MACRO_SETTING_FILE,
        sync_macro_setting=SYNC_MACRO_SETTING,
        sync_macro_fingerprint=SYNC_MACRO_FINGERPRINT,
        custom_litellm_providers=dict(CUSTOM_LITELLM_PROVIDERS),
        include_files=set(INCLUDE_FILES),
        include_prefixes=set(INCLUDE_PREFIXES),
        sync_rag_enabled=SYNC_RAG_ENABLED,
        sync_rag_store_dir=SYNC_RAG_STORE_DIR,
        sync_rag_embedding_model=SYNC_RAG_EMBEDDING_MODEL,
        sync_rag_embedding_backend=SYNC_RAG_EMBEDDING_BACKEND,
        sync_rag_embedding_provider=SYNC_RAG_EMBEDDING_PROVIDER,
        sync_rag_embedding_endpoint=SYNC_RAG_EMBEDDING_ENDPOINT,
        sync_rag_embedding_timeout_seconds=SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS,
        sync_rag_embedding_api_key_env=SYNC_RAG_EMBEDDING_API_KEY_ENV,
        sync_rag_embedding_load_error=SYNC_RAG_EMBEDDING_LOAD_ERROR,
        sync_rag_query_task_type=SYNC_RAG_QUERY_TASK_TYPE,
        sync_rag_document_task_type=SYNC_RAG_DOCUMENT_TASK_TYPE,
        sync_rag_output_dimensionality=SYNC_RAG_OUTPUT_DIMENSIONALITY,
        sync_rag_top_k_history=SYNC_RAG_TOP_K_HISTORY,
        sync_rag_top_k_terms=SYNC_RAG_TOP_K_TERMS,
        sync_rag_min_similarity=SYNC_RAG_MIN_SIMILARITY,
        sync_rag_segment_lines=SYNC_RAG_SEGMENT_LINES,
        sync_rag_history_char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
        sync_rag_update_on_success=SYNC_RAG_UPDATE_ON_SUCCESS,
        sync_source_index_enabled=SYNC_SOURCE_INDEX_ENABLED,
        sync_source_index_store_dir=SYNC_SOURCE_INDEX_STORE_DIR,
        sync_source_index_top_k=SYNC_SOURCE_INDEX_TOP_K,
        sync_source_index_min_similarity=SYNC_SOURCE_INDEX_MIN_SIMILARITY,
        sync_source_index_char_limit=SYNC_SOURCE_INDEX_CHAR_LIMIT,
        sync_project_analysis_inject_published_brief=SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF,
        sync_story_memory_enabled=SYNC_STORY_MEMORY_ENABLED,
        sync_story_memory_graph_file=SYNC_STORY_MEMORY_GRAPH_FILE,
        sync_story_memory_max_context_chars=SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
        sync_story_memory_top_k_relations=SYNC_STORY_MEMORY_TOP_K_RELATIONS,
        sync_story_memory_top_k_terms=SYNC_STORY_MEMORY_TOP_K_TERMS,
        sync_story_memory_include_scene_summary=SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
    )


def apply_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    """Publish a RuntimeConfig onto module-level globals (CLI compatibility)."""
    global ENV_GAME_ROOT, BASE_DIR, TL_SUBDIR, TL_DIR, WORK_GAME_DIR, SOURCE_GAME_DIR
    global GLOSSARY_FILE
    global PREP_ENABLED, PREP_UNPACK_RPA, PREP_GENERATE_TEMPLATE, PREP_REFRESH_EXISTING_TEMPLATE
    global PREP_LANGUAGE, PREP_RENPY_SDK_DIR, PREP_LAUNCHER_PY, PREP_PYTHON_EXE
    global PREP_UNPACK_COMMAND, PREP_TEMPLATE_COMMAND, PREP_ALLOW_SHELL_COMMANDS
    global CONTEXT_STORAGE_LOCATION, CONTEXT_STORAGE_GAME_DIR_NAME
    global API_KEYS, MODELS, CURRENT_KEY_INDEX, CURRENT_MODEL_INDEX
    global API_KEY_ROTATION_ENABLED, MODEL_ROTATION_ENABLED, MODEL_ROTATION_MODELS
    global MAX_CHARS, MAX_ITEMS, SYNC_MAX_OUTPUT_TOKENS, SYNC_TIMEOUT_SECONDS
    global SYNC_BACKEND
    global SYNC_CONTEXT_BEFORE, SYNC_CONTEXT_AFTER
    global SYNC_MACRO_SETTING_FILE, SYNC_MACRO_SETTING, SYNC_MACRO_FINGERPRINT
    global CUSTOM_LITELLM_PROVIDERS
    global INCLUDE_FILES, INCLUDE_PREFIXES
    global SYNC_RAG_ENABLED, SYNC_RAG_STORE_DIR, SYNC_RAG_EMBEDDING_MODEL
    global SYNC_RAG_EMBEDDING_BACKEND, SYNC_RAG_EMBEDDING_PROVIDER
    global SYNC_RAG_EMBEDDING_ENDPOINT, SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
    global SYNC_RAG_EMBEDDING_API_KEY_ENV, SYNC_RAG_EMBEDDING_LOAD_ERROR
    global SYNC_RAG_QUERY_TASK_TYPE, SYNC_RAG_DOCUMENT_TASK_TYPE
    global SYNC_RAG_OUTPUT_DIMENSIONALITY, SYNC_RAG_TOP_K_HISTORY, SYNC_RAG_TOP_K_TERMS
    global SYNC_RAG_MIN_SIMILARITY, SYNC_RAG_SEGMENT_LINES, SYNC_RAG_HISTORY_CHAR_LIMIT
    global SYNC_RAG_UPDATE_ON_SUCCESS, _SYNC_RAG_STORE
    global SYNC_SOURCE_INDEX_ENABLED, SYNC_SOURCE_INDEX_STORE_DIR
    global SYNC_SOURCE_INDEX_TOP_K, SYNC_SOURCE_INDEX_MIN_SIMILARITY
    global SYNC_SOURCE_INDEX_CHAR_LIMIT, SYNC_SOURCE_INDEX_CHAR_BUDGET
    global _SYNC_SOURCE_INDEX_STORE
    global SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
    global SYNC_STORY_MEMORY_ENABLED, SYNC_STORY_MEMORY_GRAPH_FILE
    global SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS, SYNC_STORY_MEMORY_TOP_K_RELATIONS
    global SYNC_STORY_MEMORY_TOP_K_TERMS, SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY
    global _SYNC_STORY_GRAPH, _SYNC_STORY_GRAPH_PATH, _active_runtime_config

    if not isinstance(config, RuntimeConfig):
        raise TypeError(f"Expected RuntimeConfig, got {type(config)!r}")

    applied = config.copy()
    with locked_runtime_state():
        ENV_GAME_ROOT = applied.env_game_root or None
        BASE_DIR = applied.base_dir
        TL_SUBDIR = applied.tl_subdir
        TL_DIR = applied.tl_dir
        WORK_GAME_DIR = applied.work_game_dir
        SOURCE_GAME_DIR = applied.source_game_dir
        GLOSSARY_FILE = applied.glossary_file or DEFAULT_GLOSSARY_FILE

        PREP_ENABLED = applied.prep_enabled
        PREP_UNPACK_RPA = applied.prep_unpack_rpa
        PREP_GENERATE_TEMPLATE = applied.prep_generate_template
        PREP_REFRESH_EXISTING_TEMPLATE = applied.prep_refresh_existing_template
        PREP_LANGUAGE = applied.prep_language or DEFAULT_PREP_LANGUAGE
        PREP_RENPY_SDK_DIR = applied.prep_renpy_sdk_dir
        PREP_LAUNCHER_PY = applied.prep_launcher_py
        PREP_PYTHON_EXE = applied.prep_python_exe
        PREP_UNPACK_COMMAND = applied.prep_unpack_command
        PREP_TEMPLATE_COMMAND = applied.prep_template_command
        PREP_ALLOW_SHELL_COMMANDS = applied.prep_allow_shell_commands

        CONTEXT_STORAGE_LOCATION = applied.context_storage_location
        CONTEXT_STORAGE_GAME_DIR_NAME = applied.context_storage_game_dir_name

        API_KEYS = list(applied.api_keys)
        MODELS = list(applied.models) if applied.models else list(DEFAULT_MODELS)
        CURRENT_KEY_INDEX = int(applied.current_key_index or 0)
        CURRENT_MODEL_INDEX = int(applied.current_model_index or 0)
        API_KEY_ROTATION_ENABLED = bool(applied.api_key_rotation_enabled)
        MODEL_ROTATION_ENABLED = bool(applied.model_rotation_enabled)
        MODEL_ROTATION_MODELS = list(applied.model_rotation_models or [])

        MAX_CHARS = int(applied.max_chars)
        MAX_ITEMS = int(applied.max_items)
        SYNC_MAX_OUTPUT_TOKENS = int(applied.sync_max_output_tokens)
        applied.sync_timeout_seconds = normalize_sync_timeout_seconds(
            applied.sync_timeout_seconds,
            DEFAULT_SYNC_TIMEOUT_SECONDS,
        )
        SYNC_TIMEOUT_SECONDS = applied.sync_timeout_seconds
        SYNC_BACKEND = applied.sync_backend or DEFAULT_SYNC_BACKEND
        SYNC_CONTEXT_BEFORE = applied.sync_context_before
        SYNC_CONTEXT_AFTER = applied.sync_context_after
        SYNC_MACRO_SETTING_FILE = applied.sync_macro_setting_file or DEFAULT_SYNC_MACRO_SETTING_FILE
        SYNC_MACRO_SETTING = applied.sync_macro_setting or ""
        SYNC_MACRO_FINGERPRINT = applied.sync_macro_fingerprint or ""
        CUSTOM_LITELLM_PROVIDERS = dict(applied.custom_litellm_providers or {})

        INCLUDE_FILES = set(applied.include_files)
        INCLUDE_PREFIXES = set(applied.include_prefixes)

        SYNC_RAG_ENABLED = applied.sync_rag_enabled
        SYNC_RAG_STORE_DIR = applied.sync_rag_store_dir
        SYNC_RAG_EMBEDDING_MODEL = applied.sync_rag_embedding_model
        SYNC_RAG_EMBEDDING_BACKEND = applied.sync_rag_embedding_backend
        SYNC_RAG_EMBEDDING_PROVIDER = applied.sync_rag_embedding_provider
        SYNC_RAG_EMBEDDING_ENDPOINT = applied.sync_rag_embedding_endpoint
        SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS = applied.sync_rag_embedding_timeout_seconds
        SYNC_RAG_EMBEDDING_API_KEY_ENV = applied.sync_rag_embedding_api_key_env
        SYNC_RAG_EMBEDDING_LOAD_ERROR = applied.sync_rag_embedding_load_error
        SYNC_RAG_QUERY_TASK_TYPE = applied.sync_rag_query_task_type
        SYNC_RAG_DOCUMENT_TASK_TYPE = applied.sync_rag_document_task_type
        SYNC_RAG_OUTPUT_DIMENSIONALITY = applied.sync_rag_output_dimensionality
        SYNC_RAG_TOP_K_HISTORY = applied.sync_rag_top_k_history
        SYNC_RAG_TOP_K_TERMS = applied.sync_rag_top_k_terms
        SYNC_RAG_MIN_SIMILARITY = applied.sync_rag_min_similarity
        SYNC_RAG_SEGMENT_LINES = applied.sync_rag_segment_lines
        SYNC_RAG_HISTORY_CHAR_LIMIT = applied.sync_rag_history_char_limit
        SYNC_RAG_UPDATE_ON_SUCCESS = applied.sync_rag_update_on_success
        _SYNC_RAG_STORE = None
        SYNC_SOURCE_INDEX_ENABLED = applied.sync_source_index_enabled
        SYNC_SOURCE_INDEX_STORE_DIR = applied.sync_source_index_store_dir
        SYNC_SOURCE_INDEX_TOP_K = applied.sync_source_index_top_k
        SYNC_SOURCE_INDEX_MIN_SIMILARITY = applied.sync_source_index_min_similarity
        SYNC_SOURCE_INDEX_CHAR_LIMIT = applied.sync_source_index_char_limit
        SYNC_SOURCE_INDEX_CHAR_BUDGET = advanced_context.DEFAULT_SOURCE_INDEX_CHAR_BUDGET
        _SYNC_SOURCE_INDEX_STORE = None
        SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = (
            applied.sync_project_analysis_inject_published_brief
        )

        SYNC_STORY_MEMORY_ENABLED = applied.sync_story_memory_enabled
        SYNC_STORY_MEMORY_GRAPH_FILE = applied.sync_story_memory_graph_file
        SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = applied.sync_story_memory_max_context_chars
        SYNC_STORY_MEMORY_TOP_K_RELATIONS = applied.sync_story_memory_top_k_relations
        SYNC_STORY_MEMORY_TOP_K_TERMS = applied.sync_story_memory_top_k_terms
        SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = applied.sync_story_memory_include_scene_summary
        _SYNC_STORY_GRAPH = None
        _SYNC_STORY_GRAPH_PATH = ""

        _active_runtime_config = applied.copy()
    return applied


def get_runtime_config() -> RuntimeConfig:
    """Return the last applied RuntimeConfig, or a snapshot of current globals."""
    if _active_runtime_config is not None:
        return _active_runtime_config.copy()
    return snapshot_runtime_config()


@contextmanager
def runtime_config_scope(
    config: Optional["RuntimeConfig"] = None,
    *,
    reload_translator_settings: bool = False,
    reload_runtime_config: bool = False,
    persist_corrected_game_root: bool = True,
    require_api_key: bool = False,
):
    """Temporarily publish a job-scoped RuntimeConfig, then restore the previous one.

    Long-lived hosts (GUI workers, registry scans) should use this so a job runs
    against a frozen configuration snapshot without permanently mutating process
    globals for other concurrent or subsequent work.

    Args:
        config: Explicit config to apply for the duration of the ``with`` block.
            When provided, disk loaders are not run unless a reload flag is also set.
        reload_translator_settings: When ``config`` is omitted, reload project
            settings from ``translator_config.json`` (and leave API keys as-is).
        reload_runtime_config: When ``config`` is omitted, call
            :func:`load_runtime_config` for a full defaults-first rebuild.
        persist_corrected_game_root: Forwarded to
            :func:`load_translator_settings` when reloading project settings.
            Readonly hosts such as doctor should pass ``False``.
        require_api_key: Forwarded to :func:`load_runtime_config` when reloading.
    """
    if reload_translator_settings and reload_runtime_config:
        raise ValueError(
            "reload_translator_settings and reload_runtime_config are mutually exclusive"
        )

    with locked_runtime_state():
        previous = snapshot_runtime_config()
        try:
            if config is not None:
                apply_runtime_config(config)
            elif reload_runtime_config:
                load_runtime_config(require_api_key=require_api_key)
            elif reload_translator_settings:
                load_translator_settings(
                    persist_corrected_game_root=persist_corrected_game_root
                )
            yield snapshot_runtime_config()
        finally:
            apply_runtime_config(previous)


def _read_json_object(path: str, *, label: str) -> dict:
    """Read a JSON object from disk. Missing file → {}; invalid JSON → {} with warning."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"Warning: Failed to load {label}: {exc}")
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        print(f"Warning: {label} root must be a JSON object; ignoring.")
        return {}
    return data


def _filter_api_keys(raw_keys) -> list:
    if not isinstance(raw_keys, (list, tuple)):
        return []
    return [
        key
        for key in raw_keys
        if isinstance(key, str) and key.strip() and not _is_placeholder_api_key(key)
    ]


def _api_keys_from_env() -> list:
    env_keys = [
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GEMINI_API_KEY_3"),
    ]
    return [key for key in env_keys if key]


def _normalize_model_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    models = []
    for value in values:
        model = str(value).strip()
        if model:
            models.append(model)
    return models


def _reset_project_settings_to_defaults():
    """Clear project/prepare globals that must not leak across reloads.

    Include filters and MAX_*/MODELS are intentionally left alone so the legacy
    ``load_config()`` → ``load_translator_settings()`` cascade still works.
    Call :func:`load_runtime_config` for a pure defaults-first rebuild.
    """
    global TL_SUBDIR, PREP_LANGUAGE, SOURCE_GAME_DIR, GLOSSARY_FILE
    global PREP_ENABLED, PREP_UNPACK_RPA, PREP_GENERATE_TEMPLATE, PREP_REFRESH_EXISTING_TEMPLATE
    global PREP_RENPY_SDK_DIR, PREP_LAUNCHER_PY, PREP_PYTHON_EXE
    global PREP_UNPACK_COMMAND, PREP_TEMPLATE_COMMAND, PREP_ALLOW_SHELL_COMMANDS
    global CONTEXT_STORAGE_LOCATION, CONTEXT_STORAGE_GAME_DIR_NAME
    global SYNC_BACKEND, SYNC_TIMEOUT_SECONDS
    global SYNC_RAG_ENABLED, SYNC_RAG_STORE_DIR, SYNC_RAG_EMBEDDING_MODEL
    global SYNC_RAG_EMBEDDING_BACKEND, SYNC_RAG_EMBEDDING_PROVIDER
    global SYNC_RAG_EMBEDDING_ENDPOINT, SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
    global SYNC_RAG_EMBEDDING_API_KEY_ENV, SYNC_RAG_EMBEDDING_LOAD_ERROR
    global SYNC_RAG_QUERY_TASK_TYPE, SYNC_RAG_DOCUMENT_TASK_TYPE
    global SYNC_RAG_OUTPUT_DIMENSIONALITY, SYNC_RAG_TOP_K_HISTORY, SYNC_RAG_TOP_K_TERMS
    global SYNC_RAG_MIN_SIMILARITY, SYNC_RAG_SEGMENT_LINES, SYNC_RAG_HISTORY_CHAR_LIMIT
    global SYNC_RAG_UPDATE_ON_SUCCESS, _SYNC_RAG_STORE
    global SYNC_SOURCE_INDEX_ENABLED, SYNC_SOURCE_INDEX_STORE_DIR
    global SYNC_SOURCE_INDEX_TOP_K, SYNC_SOURCE_INDEX_MIN_SIMILARITY
    global SYNC_SOURCE_INDEX_CHAR_LIMIT, SYNC_SOURCE_INDEX_CHAR_BUDGET
    global _SYNC_SOURCE_INDEX_STORE
    global SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
    global SYNC_STORY_MEMORY_ENABLED, SYNC_STORY_MEMORY_GRAPH_FILE
    global SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS, SYNC_STORY_MEMORY_TOP_K_RELATIONS
    global SYNC_STORY_MEMORY_TOP_K_TERMS, SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY
    global _SYNC_STORY_GRAPH, _SYNC_STORY_GRAPH_PATH
    global API_KEY_ROTATION_ENABLED, MODEL_ROTATION_ENABLED, MODEL_ROTATION_MODELS

    # Paths/language/prepare always recompute from defaults + current config.
    TL_SUBDIR = DEFAULT_TL_SUBDIR
    PREP_LANGUAGE = DEFAULT_PREP_LANGUAGE
    SOURCE_GAME_DIR = ""
    GLOSSARY_FILE = DEFAULT_GLOSSARY_FILE
    PREP_ENABLED = True
    PREP_UNPACK_RPA = True
    PREP_GENERATE_TEMPLATE = True
    PREP_REFRESH_EXISTING_TEMPLATE = True
    PREP_RENPY_SDK_DIR = ""
    PREP_LAUNCHER_PY = ""
    PREP_PYTHON_EXE = ""
    PREP_UNPACK_COMMAND = None
    PREP_TEMPLATE_COMMAND = None
    PREP_ALLOW_SHELL_COMMANDS = False
    CONTEXT_STORAGE_LOCATION = DEFAULT_CONTEXT_STORAGE_LOCATION
    CONTEXT_STORAGE_GAME_DIR_NAME = DEFAULT_CONTEXT_STORAGE_GAME_DIR_NAME

    # Rotation policy always recompute from translator_config.rotation defaults.
    API_KEY_ROTATION_ENABLED = DEFAULT_API_KEY_ROTATION_ENABLED
    MODEL_ROTATION_ENABLED = DEFAULT_MODEL_ROTATION_ENABLED
    MODEL_ROTATION_MODELS = []

    # RAG / story memory always recompute from translator_config.sync defaults.
    SYNC_BACKEND = DEFAULT_SYNC_BACKEND
    SYNC_TIMEOUT_SECONDS = DEFAULT_SYNC_TIMEOUT_SECONDS
    SYNC_RAG_ENABLED = False
    SYNC_RAG_STORE_DIR = ""
    SYNC_RAG_EMBEDDING_MODEL = DEFAULT_SYNC_RAG_EMBEDDING_MODEL
    SYNC_RAG_EMBEDDING_BACKEND = DEFAULT_SYNC_RAG_EMBEDDING_BACKEND
    SYNC_RAG_EMBEDDING_PROVIDER = ""
    SYNC_RAG_EMBEDDING_ENDPOINT = ""
    SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS = DEFAULT_SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
    SYNC_RAG_EMBEDDING_API_KEY_ENV = ""
    SYNC_RAG_EMBEDDING_LOAD_ERROR = ""
    SYNC_RAG_QUERY_TASK_TYPE = DEFAULT_SYNC_RAG_QUERY_TASK_TYPE
    SYNC_RAG_DOCUMENT_TASK_TYPE = DEFAULT_SYNC_RAG_DOCUMENT_TASK_TYPE
    SYNC_RAG_OUTPUT_DIMENSIONALITY = DEFAULT_SYNC_RAG_OUTPUT_DIMENSIONALITY
    SYNC_RAG_TOP_K_HISTORY = DEFAULT_SYNC_RAG_TOP_K_HISTORY
    SYNC_RAG_TOP_K_TERMS = DEFAULT_SYNC_RAG_TOP_K_TERMS
    SYNC_RAG_MIN_SIMILARITY = DEFAULT_SYNC_RAG_MIN_SIMILARITY
    SYNC_RAG_SEGMENT_LINES = DEFAULT_SYNC_RAG_SEGMENT_LINES
    SYNC_RAG_HISTORY_CHAR_LIMIT = DEFAULT_SYNC_RAG_HISTORY_CHAR_LIMIT
    SYNC_RAG_UPDATE_ON_SUCCESS = True
    _SYNC_RAG_STORE = None
    SYNC_SOURCE_INDEX_ENABLED = False
    SYNC_SOURCE_INDEX_STORE_DIR = ""
    SYNC_SOURCE_INDEX_TOP_K = DEFAULT_SYNC_SOURCE_INDEX_TOP_K
    SYNC_SOURCE_INDEX_MIN_SIMILARITY = DEFAULT_SYNC_SOURCE_INDEX_MIN_SIMILARITY
    SYNC_SOURCE_INDEX_CHAR_LIMIT = DEFAULT_SYNC_SOURCE_INDEX_CHAR_LIMIT
    SYNC_SOURCE_INDEX_CHAR_BUDGET = advanced_context.DEFAULT_SOURCE_INDEX_CHAR_BUDGET
    _SYNC_SOURCE_INDEX_STORE = None
    SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = False

    SYNC_STORY_MEMORY_ENABLED = False
    SYNC_STORY_MEMORY_GRAPH_FILE = ""
    SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = DEFAULT_SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS
    SYNC_STORY_MEMORY_TOP_K_RELATIONS = DEFAULT_SYNC_STORY_MEMORY_TOP_K_RELATIONS
    SYNC_STORY_MEMORY_TOP_K_TERMS = DEFAULT_SYNC_STORY_MEMORY_TOP_K_TERMS
    SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = True
    _SYNC_STORY_GRAPH = None
    _SYNC_STORY_GRAPH_PATH = ""


def load_translator_settings(*, persist_corrected_game_root: bool = True):
    """Loads per-game settings (game root, tl subdir) from translator_config.json or env.

    Always starts from code defaults for project/path/prepare fields so omitted
    keys cannot retain values from a previous load (issue #216).

    When *persist_corrected_game_root* is False (readonly commands such as
    ``project-analysis-status``), a corrected effective root is applied in
    memory only and ``translator_config.json`` is not rewritten.
    """
    global BASE_DIR, TL_DIR, TL_SUBDIR, ENV_GAME_ROOT, WORK_GAME_DIR, SOURCE_GAME_DIR, GLOSSARY_FILE
    global PREP_ENABLED, PREP_UNPACK_RPA, PREP_GENERATE_TEMPLATE, PREP_REFRESH_EXISTING_TEMPLATE, PREP_LANGUAGE
    global PREP_RENPY_SDK_DIR, PREP_LAUNCHER_PY, PREP_PYTHON_EXE, PREP_UNPACK_COMMAND, PREP_TEMPLATE_COMMAND
    global PREP_ALLOW_SHELL_COMMANDS
    global _active_runtime_config

    # Defaults first — then validated overrides from the current config file.
    _reset_project_settings_to_defaults()

    config = _read_json_object(TRANSLATOR_CONFIG, label="translator config")

    game_root = config.get("game_root")
    if isinstance(game_root, str) and game_root.strip():
        ENV_GAME_ROOT = game_root.strip()
    else:
        ENV_GAME_ROOT = os.environ.get("GAME_ROOT") or os.environ.get("SA_GAME_ROOT")

    if ENV_GAME_ROOT:
        original_root = _canonical_abs_path(ENV_GAME_ROOT)
        resolved_root = resolve_effective_game_root(original_root)
        if os.path.normcase(resolved_root) != os.path.normcase(original_root):
            ENV_GAME_ROOT = resolved_root
            should_persist = (
                persist_corrected_game_root
                and isinstance(game_root, str)
                and game_root.strip()
                and os.path.exists(TRANSLATOR_CONFIG)
            )
            if should_persist:
                try:
                    persist_game_root(resolved_root)
                except Exception as exc:
                    print(f"Warning: Failed to persist corrected game_root: {exc}")
                    _apply_game_root(resolved_root)
            else:
                _apply_game_root(resolved_root)
        else:
            BASE_DIR = original_root
    else:
        # No configured game_root: leave empty (do not invent tool parent as project).
        BASE_DIR = ""
        ENV_GAME_ROOT = ""

    load_context_storage_settings(config)

    from project_asset_paths import (
        resolve_configured_glossary_value,
        resolve_glossary_path,
    )

    glossary_file = resolve_configured_glossary_value(config)
    # Prefer current work over the tool directory for relative/bare names so
    # "glossary.json" cannot silently resolve to the install-tree default.
    resolved_glossary = resolve_glossary_path(
        glossary_file,
        game_root=BASE_DIR,
        tool_dir=TOOL_DIR,
    )
    GLOSSARY_FILE = resolved_glossary or DEFAULT_GLOSSARY_FILE

    tl_subdir = config.get("tl_subdir")
    try:
        # Always start from DEFAULT_TL_SUBDIR (not the previous load's value).
        candidate_subdir = DEFAULT_TL_SUBDIR
        if isinstance(tl_subdir, str) and tl_subdir.strip():
            candidate_subdir = normalize_tl_subdir(tl_subdir)
        else:
            candidate_subdir = normalize_tl_subdir(candidate_subdir)
        if BASE_DIR:
            candidate_tl_dir = _canonical_abs_path(
                os.path.join(BASE_DIR, candidate_subdir)
            )
            ensure_tl_dir_within_base(
                BASE_DIR,
                candidate_tl_dir,
                tl_subdir=candidate_subdir,
            )
        else:
            candidate_tl_dir = ""
    except InvalidTlSubdirError as exc:
        raise SystemExit(
            "ERROR: Invalid tl_subdir configuration. "
            "tl_subdir must be a relative path under game_root with no '..' segments "
            f"(example: 'game/tl/schinese'). Details: {exc}"
        ) from exc

    TL_SUBDIR = candidate_subdir
    TL_DIR = candidate_tl_dir
    WORK_GAME_DIR = (
        _canonical_abs_path(os.path.join(BASE_DIR, WORK_GAME_SUBDIR)) if BASE_DIR else ""
    )

    prepare = config.get("prepare")
    if not isinstance(prepare, dict):
        prepare = {}

    PREP_ENABLED = _coerce_bool(prepare.get("enabled"), True)
    PREP_UNPACK_RPA = _coerce_bool(prepare.get("unpack_rpa"), True)
    PREP_GENERATE_TEMPLATE = _coerce_bool(prepare.get("generate_template"), True)
    PREP_REFRESH_EXISTING_TEMPLATE = _coerce_bool(prepare.get("refresh_existing_template"), True)

    prep_language = prepare.get("language")
    if isinstance(prep_language, str) and prep_language.strip():
        PREP_LANGUAGE = prep_language.strip()
    else:
        # Omitted language must not retain a previous project's target language.
        PREP_LANGUAGE = DEFAULT_PREP_LANGUAGE

    # SDK path is explicit only: prepare.renpy_sdk_dir or RENPY_SDK_DIR.
    # Do not auto-scan nearby directories at load time; GUI「查找 SDK」is the
    # sole interactive discovery entry point.
    renpy_sdk_dir = prepare.get("renpy_sdk_dir")
    if not (isinstance(renpy_sdk_dir, str) and renpy_sdk_dir.strip()):
        renpy_sdk_dir = os.environ.get("RENPY_SDK_DIR")
    if isinstance(renpy_sdk_dir, str) and renpy_sdk_dir.strip():
        configured_sdk_raw = renpy_sdk_dir.strip()
        resolved_renpy_sdk_dir = _resolve_preferred_path_from_bases(
            configured_sdk_raw,
            (BASE_DIR, ROOT_DIR, TOOL_DIR),
        )
        if _is_renpy_sdk_dir(resolved_renpy_sdk_dir):
            PREP_RENPY_SDK_DIR = resolved_renpy_sdk_dir
        else:
            PREP_RENPY_SDK_DIR = ""
            shown = resolved_renpy_sdk_dir or configured_sdk_raw
            print(
                "Warning: configured Ren'Py SDK path is not a valid SDK "
                f"(missing renpy.py): {shown}. "
                "Ignoring it (no nearby auto-discovery). "
                "Fix prepare.renpy_sdk_dir / RENPY_SDK_DIR, or use GUI「查找 SDK」."
            )
    else:
        PREP_RENPY_SDK_DIR = ""

    source_game_dir = prepare.get("source_game_dir")
    if source_game_dir is not None:
        SOURCE_GAME_DIR = _resolve_path(BASE_DIR, source_game_dir)
    else:
        SOURCE_GAME_DIR = ""

    launcher_py = prepare.get("launcher_py")
    if launcher_py is not None:
        PREP_LAUNCHER_PY = _resolve_path(BASE_DIR, launcher_py)
    else:
        PREP_LAUNCHER_PY = ""

    python_exe = prepare.get("python_exe")
    if python_exe is not None:
        PREP_PYTHON_EXE = _resolve_path(BASE_DIR, python_exe)
    else:
        PREP_PYTHON_EXE = ""

    PREP_ALLOW_SHELL_COMMANDS = _coerce_bool(prepare.get("allow_shell_commands"), False)
    try:
        PREP_UNPACK_COMMAND = _coerce_command(
            prepare.get("unpack_command"),
            field_name="prepare.unpack_command",
            allow_shell=PREP_ALLOW_SHELL_COMMANDS,
        )
        PREP_TEMPLATE_COMMAND = _coerce_command(
            prepare.get("template_command"),
            field_name="prepare.template_command",
            allow_shell=PREP_ALLOW_SHELL_COMMANDS,
        )
    except InvalidPrepareCommandError as exc:
        raise SystemExit(
            "ERROR: Invalid prepare command configuration. "
            "translator_config.json is executable local configuration: custom prepare "
            "commands can run arbitrary processes. Prefer argv lists and enable "
            f"prepare.allow_shell_commands only for trusted shell strings. Details: {exc}"
        ) from exc
    from project_context_settings import apply_project_context_settings_to_config

    apply_project_context_settings_to_config(config, BASE_DIR)
    load_include_filters_from_config(config)
    load_sync_translation_settings(config)
    load_sync_rag_settings(config)
    load_sync_source_index_settings(config)
    load_sync_project_analysis_settings(config)
    load_sync_story_memory_settings(config)
    _active_runtime_config = snapshot_runtime_config()


def _require_gemini_api_key():
    if API_KEYS:
        return
    print("="*60)
    print("ERROR: No valid API keys found!")
    print("Please check api_keys.json or set GEMINI_API_KEY env vars.")
    print("="*60)
    raise SystemExit("No API keys available")


def load_config(*, require_api_key=True):
    """Loads API keys and settings from api_keys.json or environment.

    Always rebuilds API_KEYS from the current file + env so a previous load
    cannot leave stale credentials active (issue #216).
    """
    global API_KEYS, MODELS, MAX_CHARS, MAX_ITEMS, SYNC_MAX_OUTPUT_TOKENS
    global INCLUDE_FILES, INCLUDE_PREFIXES, CURRENT_KEY_INDEX, CURRENT_MODEL_INDEX
    global _active_runtime_config

    # Credentials always start empty and are rebuilt from the current sources.
    API_KEYS = []
    CURRENT_KEY_INDEX = 0

    # Try loading from JSON
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise TypeError("api_keys config root must be a JSON object")
                keys = config.get("api_keys", [])
                custom_models = config.get("models", [])
                single_model = config.get("model")
                batch_size = config.get("sync_chunk_size", config.get("batch_size"))
                max_chars = config.get("max_chars")
                sync_max_chars = config.get("sync_max_source_chars")
                if sync_max_chars is not None:
                    max_chars = sync_max_chars
                sync_max_output_tokens = config.get("sync_max_output_tokens")
                include_files = config.get("include_files")
                include_prefixes = config.get("include_prefixes")

                # Always assign (even when empty) so previous keys cannot stick.
                API_KEYS = _filter_api_keys(keys)
                if API_KEYS:
                    print(f"Loaded {len(API_KEYS)} API keys from config file.")

                if not custom_models and single_model:
                    custom_models = [single_model]
                elif isinstance(custom_models, str):
                    custom_models = [custom_models]

                if custom_models:
                    replace_model_list(custom_models, "custom")

                try:
                    if batch_size is not None:
                        batch_size = int(batch_size)
                        if batch_size > 0:
                            MAX_ITEMS = batch_size
                            print(f"Using custom batch size: {MAX_ITEMS}")
                except (TypeError, ValueError):
                    print("Warning: Invalid batch_size in config; using default.")

                try:
                    if max_chars is not None:
                        max_chars = int(max_chars)
                        if max_chars > 0:
                            MAX_CHARS = max_chars
                            print(f"Using custom max_chars: {MAX_CHARS}")
                except (TypeError, ValueError):
                    print("Warning: Invalid max_chars in config; using default.")

                try:
                    if sync_max_output_tokens is not None:
                        sync_max_output_tokens = int(sync_max_output_tokens)
                        if sync_max_output_tokens > 0:
                            SYNC_MAX_OUTPUT_TOKENS = sync_max_output_tokens
                            print(f"Using custom sync max output tokens: {SYNC_MAX_OUTPUT_TOKENS}")
                except (TypeError, ValueError):
                    print("Warning: Invalid sync_max_output_tokens in config; using default.")

                if "include_files" in config:
                    INCLUDE_FILES = coerce_normalized_rel_path_set(include_files)
                    print(f"Using include_files allowlist ({len(INCLUDE_FILES)}).")

                if "include_prefixes" in config:
                    INCLUDE_PREFIXES = coerce_normalized_rel_path_set(include_prefixes)
                    print(f"Using include_prefixes allowlist ({len(INCLUDE_PREFIXES)}).")

        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            # Failed parse must not keep previously loaded keys.
            API_KEYS = []
            CURRENT_KEY_INDEX = 0

    # Fallback to Environment Variables if no keys found
    if not API_KEYS:
        print("Checking environment variables for keys...")
        API_KEYS = _api_keys_from_env()

    if require_api_key:
        _require_gemini_api_key()

    _active_runtime_config = snapshot_runtime_config()


def load_runtime_config(*, require_api_key=True) -> RuntimeConfig:
    """Load a fresh RuntimeConfig from defaults + api_keys + translator_config.

    Preferred entry point for long-lived hosts (GUI workers, tests, embedded
    use). Compatibility wrappers :func:`load_config` and
    :func:`load_translator_settings` still exist for CLI callers.
    """
    # Start from pure defaults so omitted fields cannot leak across projects.
    base = default_runtime_config()
    apply_runtime_config(base)
    load_config(require_api_key=False)
    load_translator_settings()
    if require_api_key:
        _require_gemini_api_key()
    config = snapshot_runtime_config()
    apply_runtime_config(config)
    return config


SCRIPT_FILE_EXTENSIONS = {".rpy", ".rpym", ".rpyc", ".rpymc"}


def _has_files_with_extensions(base_dir, extensions):
    if not os.path.isdir(base_dir):
        return False
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                return True
    return False


def _is_translation_relpath(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    return bool(parts) and parts[0].lower() == "tl"


def _has_non_translation_files_with_extensions(base_dir, extensions):
    if not os.path.isdir(base_dir):
        return False
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() not in extensions:
                continue
            rel = os.path.relpath(os.path.join(root, file), base_dir)
            if _is_translation_relpath(rel):
                continue
            return True
    return False


def _list_rpa_files(game_dir):
    if not game_dir or not os.path.isdir(game_dir):
        return []
    archives = []
    for file in os.listdir(game_dir):
        path = os.path.join(game_dir, file)
        if os.path.isfile(path) and file.lower().endswith(".rpa"):
            archives.append(path)
    archives.sort(key=lambda p: os.path.basename(p).lower())
    return archives


def _canonical_abs_path(path):
    """Return a stable absolute path (long path on Windows, not 8.3 short names)."""
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    try:
        return str(Path(abs_path).resolve(strict=False))
    except OSError:
        return abs_path


canonical_abs_path = _canonical_abs_path


def _path_contains_path(container, contained):
    container_norm = _canonical_abs_path(container)
    contained_norm = _canonical_abs_path(contained)
    if not container_norm or not contained_norm:
        return False
    try:
        common = os.path.commonpath([container_norm, contained_norm])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(container_norm)


GAME_ROOT_REQUIRED_MESSAGE = (
    "game_root is not set. Configure translator_config.json game_root "
    "or environment variable GAME_ROOT / SA_GAME_ROOT before prepare, "
    "bootstrap-work, or other project path operations."
)


class MissingGameRootError(ValueError):
    """Raised when a project path is required but BASE_DIR / game_root is empty."""


def require_base_dir(base_dir=None) -> str:
    """Return a non-empty absolute game root, or raise MissingGameRootError.

    Never substitutes the process CWD when game_root is unset.
    """
    raw = base_dir if base_dir is not None else BASE_DIR
    if not (isinstance(raw, str) and str(raw).strip()):
        raise MissingGameRootError(GAME_ROOT_REQUIRED_MESSAGE)
    return _canonical_abs_path(str(raw).strip())


def resolve_project_root(base_dir=None):
    base = require_base_dir(base_dir)
    if os.path.basename(base).lower() in {"work", "original"}:
        return os.path.dirname(base)
    return base


def resolve_work_dir(base_dir=None):
    return _canonical_abs_path(os.path.join(resolve_project_root(base_dir), "work"))


def resolve_effective_game_root(game_root):
    """Prefer nested work/ when game_root points at a project-root layout."""
    normalized = _canonical_abs_path(game_root)
    if os.path.basename(normalized).lower() == "work":
        return normalized

    nested_work = os.path.join(normalized, "work")
    original_game = os.path.join(normalized, "original", "game")
    if os.path.isdir(nested_work) and os.path.isdir(original_game):
        return _canonical_abs_path(nested_work)
    return normalized


def resolve_original_game_dir(base_dir=None):
    if SOURCE_GAME_DIR and os.path.isdir(SOURCE_GAME_DIR):
        return _canonical_abs_path(SOURCE_GAME_DIR)

    try:
        root = resolve_project_root(base_dir)
    except MissingGameRootError:
        return ""
    candidate = os.path.join(root, "original", "game")
    if os.path.isdir(candidate):
        return _canonical_abs_path(candidate)
    return ""


def is_work_dir_empty(work_dir):
    if not os.path.isdir(work_dir):
        return True
    try:
        return len(os.listdir(work_dir)) == 0
    except OSError:
        return False


def work_dir_bootstrap_allowed(base_dir=None):
    try:
        work_dir = resolve_work_dir(base_dir)
    except MissingGameRootError as exc:
        return False, "", str(exc)
    if is_work_dir_empty(work_dir):
        return True, work_dir, ""
    return False, work_dir, "work directory already exists and is not empty"


WORK_BOOTSTRAP_COPY_PROGRESS_INTERVAL = 25


def _should_emit_bootstrap_copy_progress(files_copied, total_files):
    if total_files <= 0:
        return False
    if files_copied >= total_files:
        return True
    if files_copied == 1:
        return True
    return files_copied % WORK_BOOTSTRAP_COPY_PROGRESS_INTERVAL == 0


def _copy_game_directory(source_game_dir, target_game_dir):
    total_files = sum(len(files) for _, _, files in os.walk(source_game_dir))
    files_copied = 0
    if total_files:
        print(f"Work bootstrap copy progress: 0/{total_files} files.", flush=True)
    for root, _, files in os.walk(source_game_dir):
        rel = os.path.relpath(root, source_game_dir)
        dest_dir = target_game_dir if rel == "." else os.path.join(target_game_dir, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in files:
            src_path = os.path.join(root, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy2(src_path, dest_path)
            files_copied += 1
            if not _should_emit_bootstrap_copy_progress(files_copied, total_files):
                continue
            rel_file = os.path.relpath(src_path, source_game_dir).replace(os.sep, "/")
            print(
                f"Work bootstrap copy progress: {files_copied}/{total_files} files, file={rel_file}.",
                flush=True,
            )
    return files_copied


def _apply_game_root(work_dir):
    global BASE_DIR, TL_DIR, WORK_GAME_DIR, ENV_GAME_ROOT

    normalized = _canonical_abs_path(work_dir)
    ENV_GAME_ROOT = normalized
    BASE_DIR = normalized
    candidate_tl_dir = _canonical_abs_path(os.path.join(BASE_DIR, TL_SUBDIR))
    ensure_tl_dir_within_base(BASE_DIR, candidate_tl_dir, tl_subdir=TL_SUBDIR)
    TL_DIR = candidate_tl_dir
    WORK_GAME_DIR = _canonical_abs_path(os.path.join(BASE_DIR, WORK_GAME_SUBDIR))


def persist_game_root(work_dir):
    from project_asset_paths import sync_project_asset_paths_in_config

    normalized = _canonical_abs_path(work_dir)
    config = {}
    if os.path.exists(TRANSLATOR_CONFIG):
        try:
            with open(TRANSLATOR_CONFIG, "r", encoding="utf-8-sig") as handle:
                config = json.load(handle) or {}
        except Exception:
            config = {}

    config["game_root"] = normalized
    sync_project_asset_paths_in_config(config, normalized)
    with open(TRANSLATOR_CONFIG, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    _apply_game_root(normalized)
    return normalized


def bootstrap_work_from_original(*, save_game_root=False, refresh_runtime_paths=False, base_dir=None):
    try:
        base = require_base_dir(base_dir)
    except MissingGameRootError as exc:
        return {
            "status": "failed",
            "project_root": "",
            "work_dir": "",
            "source_game_dir": "",
            "files_copied": 0,
            "message": str(exc),
            "game_root_updated": False,
        }
    project_root = resolve_project_root(base)
    work_dir = resolve_work_dir(base)
    allowed, _, skip_reason = work_dir_bootstrap_allowed(base)
    source_game_dir = resolve_original_game_dir(base)

    if not allowed:
        return {
            "status": "skipped",
            "project_root": project_root,
            "work_dir": work_dir,
            "source_game_dir": source_game_dir,
            "files_copied": 0,
            "message": skip_reason,
            "game_root_updated": False,
        }

    if not source_game_dir:
        return {
            "status": "failed",
            "project_root": project_root,
            "work_dir": work_dir,
            "source_game_dir": "",
            "files_copied": 0,
            "message": (
                "original/game was not found; set prepare.source_game_dir or create work manually."
            ),
            "game_root_updated": False,
        }

    target_game_dir = os.path.join(work_dir, WORK_GAME_SUBDIR)
    if _path_contains_path(source_game_dir, target_game_dir):
        return {
            "status": "failed",
            "project_root": project_root,
            "work_dir": work_dir,
            "source_game_dir": source_game_dir,
            "files_copied": 0,
            "message": "source_game_dir must not contain work/game.",
            "game_root_updated": False,
        }
    staging_dir = os.path.join(work_dir, ".bootstrap_staging")
    try:
        os.makedirs(work_dir, exist_ok=True)
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        files_copied = _copy_game_directory(source_game_dir, staging_dir)
        if os.path.exists(target_game_dir):
            shutil.rmtree(target_game_dir)
        os.replace(staging_dir, target_game_dir)
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return {
            "status": "failed",
            "project_root": project_root,
            "work_dir": work_dir,
            "source_game_dir": source_game_dir,
            "files_copied": 0,
            "message": str(exc),
            "game_root_updated": False,
        }

    message = f"Copied {files_copied} files from original/game into work/game."
    game_root_updated = False
    if os.path.normcase(base) != os.path.normcase(work_dir):
        if save_game_root:
            try:
                persist_game_root(work_dir)
                game_root_updated = True
            except Exception as exc:
                _apply_game_root(work_dir)
                message = f"{message} Failed to update game_root: {exc}"
        elif refresh_runtime_paths:
            _apply_game_root(work_dir)

    return {
        "status": "created",
        "project_root": project_root,
        "work_dir": work_dir,
        "source_game_dir": source_game_dir,
        "files_copied": files_copied,
        "message": message,
        "game_root_updated": game_root_updated,
    }


def _guess_source_game_dir():
    candidates = []
    if SOURCE_GAME_DIR:
        candidates.append(SOURCE_GAME_DIR)
    candidates.append(WORK_GAME_DIR)

    original_game = resolve_original_game_dir()
    if original_game:
        candidates.append(original_game)

    seen = set()
    ordered = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = _canonical_abs_path(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    for candidate in ordered:
        if not os.path.isdir(candidate):
            continue
        if _has_non_translation_files_with_extensions(candidate, SCRIPT_FILE_EXTENSIONS):
            return candidate
        if _list_rpa_files(candidate):
            return candidate

    for candidate in ordered:
        if os.path.isdir(candidate):
            return candidate

    return WORK_GAME_DIR


def _copy_script_sources(source_game_dir, target_game_dir):
    if not source_game_dir or not target_game_dir:
        return 0
    if not os.path.isdir(source_game_dir):
        return 0
    if os.path.abspath(source_game_dir) == os.path.abspath(target_game_dir):
        return 0

    copied = 0
    for root, _, files in os.walk(source_game_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() not in SCRIPT_FILE_EXTENSIONS:
                continue

            src = os.path.join(root, file)
            rel = os.path.relpath(src, source_game_dir)
            dst = os.path.join(target_game_dir, rel)

            if os.path.exists(dst):
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    return copied


def _safe_archive_relpath(raw_name):
    if isinstance(raw_name, bytes):
        try:
            raw_name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            raw_name = raw_name.decode("latin-1", errors="replace")
    else:
        raw_name = str(raw_name)

    rel = raw_name.replace("\\", "/").strip().lstrip("/")
    parts = []
    for part in rel.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return ""
        parts.append(part)

    if not parts:
        return ""

    return os.path.join(*parts)


class _RestrictedRpaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Disallowed pickle global during RPA index load: {module}.{name}"
        )


def _load_pickle_blob(blob):
    return _RestrictedRpaUnpickler(io.BytesIO(blob), encoding="bytes").load()


def _read_rpa_index(archive_path, limits=DEFAULT_RPA_LIMITS):
    with open(archive_path, "rb") as infile:
        archive_size = os.fstat(infile.fileno()).st_size
        header = infile.read(40)

        if header.startswith(b"RPA-3.0 "):
            offset = int(header[8:24], 16)
            key = int(header[25:33], 16)
            payload = read_bounded_compressed_index(
                infile,
                offset,
                archive_size,
                limits,
            )
            return decode_and_validate_index(
                _load_pickle_blob(payload),
                archive_size,
                key=key,
                limits=limits,
            )

        if header.startswith(b"RPA-2.0 "):
            infile.seek(0)
            line = infile.read(24)
            offset = int(line[8:], 16)
            payload = read_bounded_compressed_index(
                infile,
                offset,
                archive_size,
                limits,
            )
            return decode_and_validate_index(
                _load_pickle_blob(payload),
                archive_size,
                limits=limits,
            )

    raise RuntimeError("Unsupported RPA format (expecting RPA-3.0 or RPA-2.0).")


def _extract_rpa_scripts(
    archive_path,
    target_game_dir,
    limits=DEFAULT_RPA_LIMITS,
    extraction_budget=None,
):
    index = _read_rpa_index(archive_path, limits=limits)
    target_root = os.path.abspath(target_game_dir)
    extracted = 0
    planned = []
    total_output = 0

    for raw_name, chunks in index.items():
        rel = _safe_archive_relpath(raw_name)
        if not rel:
            continue
        if os.path.splitext(rel)[1].lower() not in SCRIPT_FILE_EXTENSIONS:
            continue

        out_path = os.path.abspath(os.path.join(target_root, rel))
        try:
            if os.path.commonpath([target_root, out_path]) != target_root:
                continue
        except ValueError:
            continue

        if os.path.exists(out_path):
            continue

        total_output += member_output_size(chunks)
        planned.append((out_path, chunks))

    if extraction_budget is None:
        extraction_budget = RpaExtractionBudget(limits.max_total_extraction_bytes)
    extraction_budget.reserve(total_output)

    with open(archive_path, "rb") as source:
        for out_path, chunks in planned:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with open(out_path, "wb") as target:
                    copy_member(source, target, chunks, limits)
            except Exception:
                try:
                    os.remove(out_path)
                except FileNotFoundError:
                    pass
                raise
            extracted += 1

    return extracted


def _find_bundled_python(base_dir):
    if not base_dir:
        return ""

    if sys.platform.startswith("win"):
        patterns = [
            os.path.join(base_dir, "lib", "py3-windows-x86_64", "python.exe"),
            os.path.join(base_dir, "lib", "py3-windows-*", "python.exe"),
        ]
    elif sys.platform == "darwin":
        patterns = [
            os.path.join(base_dir, "lib", "py3-mac*", "python"),
            os.path.join(base_dir, "lib", "py3-macos*", "python"),
            os.path.join(base_dir, "lib", "py3-darwin*", "python"),
        ]
    else:
        patterns = [
            os.path.join(base_dir, "lib", "py3-linux*", "python"),
        ]

    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if os.path.isfile(candidate):
                return candidate

    return ""


def _resolve_prepare_python(runtime_root=""):
    if PREP_PYTHON_EXE and os.path.isfile(PREP_PYTHON_EXE):
        return PREP_PYTHON_EXE

    for base_dir in (runtime_root, PREP_RENPY_SDK_DIR, BASE_DIR):
        bundled = _find_bundled_python(base_dir)
        if bundled:
            return bundled

    return sys.executable


def _resolve_sdk_launcher(sdk_dir):
    if not sdk_dir:
        return ""
    launcher = os.path.join(sdk_dir, "renpy.py")
    if os.path.isfile(launcher):
        return launcher
    return ""


def _resolve_sdk_shell_launcher(sdk_dir):
    if not sdk_dir or sys.platform.startswith("win"):
        return ""
    launcher = os.path.join(sdk_dir, "renpy.sh")
    if os.path.isfile(launcher):
        return launcher
    return ""


def _is_sdk_launcher(launcher_py):
    return bool(launcher_py) and os.path.basename(launcher_py).lower() == "renpy.py"


def _prepare_launcher_root(launcher_py):
    if not launcher_py:
        return ""
    return os.path.dirname(os.path.abspath(launcher_py))


def _resolve_prepare_launcher():
    if PREP_LAUNCHER_PY and os.path.isfile(PREP_LAUNCHER_PY):
        return PREP_LAUNCHER_PY

    sdk_launcher = _resolve_sdk_launcher(PREP_RENPY_SDK_DIR)
    if sdk_launcher:
        return sdk_launcher

    # Only search the configured game root top-level — never process CWD.
    if not BASE_DIR:
        return ""

    py_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.py")))
    if not py_files:
        return ""

    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                head = handle.read(4096)
            if "renpy.bootstrap" in head:
                return path
        except Exception:
            continue

    exe_stems = {
        os.path.splitext(os.path.basename(path))[0].lower()
        for path in glob.glob(os.path.join(BASE_DIR, "*.exe"))
    }
    for path in py_files:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if stem in exe_stems:
            return path

    return ""


def _render_prepare_command(command, variables):
    def _fmt(token):
        try:
            return str(token).format(**variables)
        except KeyError as e:
            missing = str(e).strip("'")
            raise RuntimeError(f"Missing command placeholder: {missing}")

    if isinstance(command, list):
        rendered = []
        for token in command:
            rendered.append(_fmt(token))
        return rendered, False

    if isinstance(command, str):
        if not PREP_ALLOW_SHELL_COMMANDS:
            raise RuntimeError(
                "Shell string prepare commands require prepare.allow_shell_commands=true"
            )
        return _fmt(command), True

    raise RuntimeError(f"Unsupported prepare command type: {type(command).__name__}")


def _machine_subprocess_diagnostic_stream():
    """Return an OS-backed stderr stream for child diagnostics in machine mode."""

    if not cli_contract.machine_output_active():
        return None
    for stream in (sys.stderr, sys.__stderr__):
        if stream is None:
            continue
        try:
            stream.fileno()
        except (AttributeError, io.UnsupportedOperation, OSError):
            continue
        return stream
    return subprocess.DEVNULL


def _machine_subprocess_output_kwargs():
    stream = _machine_subprocess_diagnostic_stream()
    if stream is None:
        return {}
    return {"stdout": stream, "stderr": stream}


def _run_prepare_command(command, cwd, step_name):
    use_shell = prepare_command_uses_shell(command)
    if use_shell and not PREP_ALLOW_SHELL_COMMANDS:
        print(
            f"[Prepare] {step_name} refused: shell string commands require "
            "prepare.allow_shell_commands=true (trusted local config only)."
        )
        return False

    shown = describe_prepare_command(command)
    print(f"[Prepare] {step_name}")
    print(f"[Prepare]   cwd: {cwd}")
    print(f"[Prepare]   shell: {use_shell}")
    print(f"[Prepare]   command: {shown}")

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=use_shell,
            check=False,
            **_machine_subprocess_output_kwargs(),
        )
    except Exception as e:
        print(f"[Prepare] {step_name} failed to start: {e}")
        return False

    if result.returncode != 0:
        print(f"[Prepare] {step_name} failed with exit code {result.returncode}.")
        return False

    return True


def get_prepare_template_command_info(source_game_dir=""):
    if not source_game_dir:
        source_game_dir = _guess_source_game_dir()

    launcher_py = _resolve_prepare_launcher()
    python_exe = _resolve_prepare_python(_prepare_launcher_root(launcher_py))
    variables = {
        "python_exe": python_exe,
        "launcher_py": launcher_py,
        "language": PREP_LANGUAGE,
        "base_dir": BASE_DIR,
        "tl_dir": TL_DIR,
        "work_game_dir": WORK_GAME_DIR,
        "source_game_dir": source_game_dir,
    }

    if PREP_TEMPLATE_COMMAND:
        try:
            rendered, use_shell = _render_prepare_command(PREP_TEMPLATE_COMMAND, variables)
        except Exception as e:
            return {
                "available": False,
                "kind": "custom",
                "reason": str(e),
                "command": None,
                "cwd": BASE_DIR,
                "python_exe": python_exe,
                "launcher_py": launcher_py,
                "shell": False,
                "allow_shell_commands": PREP_ALLOW_SHELL_COMMANDS,
            }
        return {
            "available": True,
            "kind": "custom",
            "reason": "",
            "command": rendered,
            "cwd": BASE_DIR,
            "python_exe": python_exe,
            "launcher_py": launcher_py,
            "shell": use_shell,
            "allow_shell_commands": PREP_ALLOW_SHELL_COMMANDS,
        }

    if not launcher_py:
        return {
            "available": False,
            "kind": "auto",
            "reason": "Ren'Py SDK or game launcher was not found.",
            "command": None,
            "cwd": BASE_DIR,
            "python_exe": python_exe,
            "launcher_py": "",
            "shell": False,
            "allow_shell_commands": PREP_ALLOW_SHELL_COMMANDS,
        }

    if _is_sdk_launcher(launcher_py):
        sdk_dir = _prepare_launcher_root(launcher_py)
        shell_launcher = _resolve_sdk_shell_launcher(sdk_dir)
        if shell_launcher and not PREP_PYTHON_EXE:
            command = [shell_launcher, BASE_DIR, "translate", PREP_LANGUAGE]
            cwd = sdk_dir
            python_exe = ""
        else:
            command = [python_exe, launcher_py, BASE_DIR, "translate", PREP_LANGUAGE]
            cwd = BASE_DIR
        kind = "sdk"
    else:
        command = [python_exe, launcher_py, "translate", PREP_LANGUAGE]
        cwd = BASE_DIR
        kind = "game-launcher"

    return {
        "available": True,
        "kind": kind,
        "reason": "",
        "command": command,
        "cwd": cwd,
        "python_exe": python_exe,
        "launcher_py": launcher_py,
        "shell": False,
        "allow_shell_commands": PREP_ALLOW_SHELL_COMMANDS,
    }


def _run_unpack_command(command, archives, source_game_dir):
    command_text = " ".join(command) if isinstance(command, list) else str(command)
    per_archive = "{archive}" in command_text
    first_archive = archives[0] if archives else ""

    if per_archive:
        success = True
        for archive in archives:
            variables = {
                "archive": archive,
                "source_game_dir": source_game_dir,
                "work_game_dir": WORK_GAME_DIR,
                "base_dir": BASE_DIR,
                "tl_dir": TL_DIR,
                "language": PREP_LANGUAGE,
            }
            try:
                rendered, _ = _render_prepare_command(command, variables)
            except Exception as e:
                print(f"[Prepare] Custom RPA unpack command error: {e}")
                return False
            ok = _run_prepare_command(rendered, BASE_DIR, "Custom RPA unpack")
            success = success and ok
        return success

    variables = {
        "archive": first_archive,
        "source_game_dir": source_game_dir,
        "work_game_dir": WORK_GAME_DIR,
        "base_dir": BASE_DIR,
        "tl_dir": TL_DIR,
        "language": PREP_LANGUAGE,
    }
    try:
        rendered, _ = _render_prepare_command(command, variables)
    except Exception as e:
        print(f"[Prepare] Custom RPA unpack command error: {e}")
        return False
    return _run_prepare_command(rendered, BASE_DIR, "Custom RPA unpack")


def _has_translation_templates():
    return _has_files_with_extensions(TL_DIR, {".rpy"})


def _describe_template_unavailable(template_info):
    reason = template_info.get("reason") or "no command resolved"
    if template_info.get("kind") == "custom":
        return f"Custom template command error: {reason}"
    return reason


def run_prepare_steps():
    if not PREP_ENABLED:
        print("[Prepare] Disabled by translator_config.")
        return

    try:
        require_base_dir()
    except MissingGameRootError as exc:
        raise SystemExit(f"[Prepare] {exc}") from exc

    allowed, _, _ = work_dir_bootstrap_allowed()
    if allowed and resolve_original_game_dir():
        bootstrap_result = bootstrap_work_from_original(
            save_game_root=True,
            refresh_runtime_paths=True,
        )
        if bootstrap_result["status"] == "failed":
            raise SystemExit(
                f"[Prepare] Work bootstrap failed: {bootstrap_result['message']}"
            )
        if bootstrap_result["status"] == "created":
            print(f"[Prepare] Work bootstrap: {bootstrap_result['message']}")
            if bootstrap_result["game_root_updated"]:
                print(f"[Prepare] Updated game_root to: {bootstrap_result['work_dir']}")

    source_game_dir = _guess_source_game_dir()
    os.makedirs(WORK_GAME_DIR, exist_ok=True)
    print(f"[Prepare] Source game dir: {source_game_dir}")
    print(f"[Prepare] Work game dir: {WORK_GAME_DIR}")

    copied_scripts = _copy_script_sources(source_game_dir, WORK_GAME_DIR)
    if copied_scripts:
        print(f"[Prepare] Copied {copied_scripts} script files into work/game.")

    if PREP_UNPACK_RPA:
        has_scripts = _has_non_translation_files_with_extensions(WORK_GAME_DIR, SCRIPT_FILE_EXTENSIONS)
        if has_scripts:
            print("[Prepare] Script files already exist in work/game; skipping RPA unpack.")
        else:
            archives = _list_rpa_files(source_game_dir)
            if not archives:
                print("[Prepare] No .rpa files found; skip unpack.")
            elif PREP_UNPACK_COMMAND:
                _run_unpack_command(PREP_UNPACK_COMMAND, archives, source_game_dir)
            else:
                total_extracted = 0
                extraction_budget = RpaExtractionBudget(
                    DEFAULT_RPA_LIMITS.max_total_extraction_bytes
                )
                for archive in archives:
                    try:
                        extracted = _extract_rpa_scripts(
                            archive,
                            WORK_GAME_DIR,
                            extraction_budget=extraction_budget,
                        )
                        total_extracted += extracted
                        print(f"[Prepare] Extracted {extracted} script files from {os.path.basename(archive)}.")
                    except Exception as e:
                        print(f"[Prepare] Failed to unpack {os.path.basename(archive)}: {e}")
                print(f"[Prepare] Total extracted script files: {total_extracted}.")
    else:
        print("[Prepare] RPA unpack disabled.")

    if PREP_GENERATE_TEMPLATE:
        templates_exist = _has_translation_templates()
        if templates_exist and not PREP_REFRESH_EXISTING_TEMPLATE:
            print("[Prepare] Translation template already exists; skipping generation.")
        else:
            if templates_exist:
                print("[Prepare] Translation template exists; refreshing missing entries.")
            template_info = get_prepare_template_command_info(source_game_dir)
            if template_info["available"]:
                ok = _run_prepare_command(
                    template_info["command"],
                    template_info["cwd"],
                    "Generate tl template",
                )
                if not ok and not templates_exist:
                    raise SystemExit(
                        "[Prepare] Translation template generation failed and no TL files exist. "
                        "Install Ren'Py SDK, set RENPY_SDK_DIR, or prepare game/tl/<language> manually."
                    )
                if not ok and templates_exist:
                    print("[Prepare] Template refresh failed; continuing with existing TL files.")
            elif templates_exist:
                print(
                    f"[Prepare] {_describe_template_unavailable(template_info)}; "
                    "continuing with existing TL files."
                )
            else:
                reason = _describe_template_unavailable(template_info)
                raise SystemExit(
                    f"[Prepare] Cannot generate translation template: {reason}. "
                    "Install Ren'Py SDK and set RENPY_SDK_DIR or prepare.renpy_sdk_dir, "
                    "fix prepare.template_command, or create the TL template manually."
                )
    else:
        print("[Prepare] Template generation disabled.")

def get_current_api_key():
    return API_KEYS[CURRENT_KEY_INDEX]

def get_current_model():
    return MODELS[CURRENT_MODEL_INDEX]

def effective_model_rotation_pool() -> list[str]:
    """Runtime model sequence used for hopping (initialized by load_sync_translation_settings)."""
    return list(MODELS)

def api_key_rotation_attempts() -> int:
    """How many distinct key tries are allowed for the current rotation policy."""
    key_count = len(API_KEYS)
    if key_count <= 0:
        return 1
    if not API_KEY_ROTATION_ENABLED or key_count == 1:
        return 1
    return key_count


def rotate_api_key():
    """Advance to the next API key when rotation is enabled and multiple keys exist.

    Returns True only when the active key index actually changes.
    """
    global CURRENT_KEY_INDEX
    if not API_KEY_ROTATION_ENABLED:
        return False
    if len(API_KEYS) <= 1:
        return False
    previous = CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    if CURRENT_KEY_INDEX == previous:
        return False
    print(f"  ➜ Rotating to API Key #{CURRENT_KEY_INDEX + 1}")
    return True

def rotate_model():
    """Advance along the MODELS sequence built by load_sync_translation_settings.

    Does not re-apply MODEL_ROTATION_MODELS at runtime (that would drop the
    selected primary model or skip the first entry). Returns True only when the
    active model actually changes.
    """
    global CURRENT_MODEL_INDEX
    if not MODEL_ROTATION_ENABLED:
        return False
    if len(MODELS) <= 1:
        return False
    previous_index = CURRENT_MODEL_INDEX
    previous_model = MODELS[CURRENT_MODEL_INDEX] if MODELS else ""
    CURRENT_MODEL_INDEX = (CURRENT_MODEL_INDEX + 1) % len(MODELS)
    next_model = MODELS[CURRENT_MODEL_INDEX]
    if CURRENT_MODEL_INDEX == previous_index or next_model == previous_model:
        return False
    print(f"  ➜ Rotating to Model: {next_model}")
    return True

def configure_genai():
    """Ensures the google-genai library is available."""
    get_genai_module()


def create_genai_client(api_key=None):
    genai = get_genai_module()
    return genai.Client(api_key=api_key or get_current_api_key())


def _slugify(text):
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text or "").strip("-._")
    return text or "sync"


def _project_slug_from_base_dir(base_dir):
    """Derive a store slug from a project path without inventing CWD roots.

    Empty / unset base_dir yields ``unset`` so tool-local stores still work when
    game_root is not configured (e.g. batch retry metadata under logs/).
    """
    raw = base_dir if base_dir is not None else BASE_DIR
    if not (isinstance(raw, str) and str(raw).strip()):
        return "unset"
    base = _canonical_abs_path(str(raw).strip())
    if not base:
        return "unset"
    base_name = os.path.basename(base)
    if base_name.lower() in {"work", "original"}:
        parent = os.path.basename(os.path.dirname(base))
        return _slugify(parent or base_name)
    return _slugify(base_name)


def guess_project_slug():
    return _project_slug_from_base_dir(BASE_DIR)


def get_context_storage_location():
    return CONTEXT_STORAGE_LOCATION


def get_context_storage_root(base_dir=None):
    if CONTEXT_STORAGE_LOCATION == "game":
        try:
            project_root = resolve_project_root(base_dir)
        except MissingGameRootError:
            # Prefer tool logs over inventing a CWD project path.
            print(
                "Warning: context_storage.location is 'game' but game_root is unset; "
                "using tool logs root with an 'unset' project slug."
            )
            return LOG_DIR
        return _canonical_abs_path(
            os.path.join(project_root, CONTEXT_STORAGE_GAME_DIR_NAME)
        )
    return LOG_DIR


def get_default_context_store_dir(store_name, base_dir=None):
    root = get_context_storage_root(base_dir)
    if CONTEXT_STORAGE_LOCATION == "game":
        try:
            require_base_dir(base_dir)
            return os.path.join(root, store_name)
        except MissingGameRootError:
            # Match tool-mode isolation: .../store_name/unset under LOG_DIR.
            return os.path.join(LOG_DIR, store_name, "unset")
    slug = _project_slug_from_base_dir(base_dir if base_dir is not None else BASE_DIR)
    return os.path.join(root, store_name, slug)


def get_default_batch_rag_store_dir():
    return get_default_context_store_dir("rag_store")


def get_default_source_index_store_dir():
    return get_default_context_store_dir("source_index_store")


def get_default_project_analysis_store_dir(base_dir=None):
    return get_default_context_store_dir("project_analysis", base_dir)


def get_default_story_memory_graph_path():
    if CONTEXT_STORAGE_LOCATION == "game":
        return os.path.join(get_context_storage_root(), "story_memory", "story_graph.json")
    return os.path.join(LOG_DIR, "story_memory", "story_graph.json")


def get_default_sync_rag_store_dir():
    return get_default_context_store_dir("rag_store")


def current_sync_embedding_settings():
    """Return the active Sync embedding backend selection."""

    from dataclasses import replace

    settings = embedding_runtime.EmbeddingRuntimeSettings(
        backend=SYNC_RAG_EMBEDDING_BACKEND or embedding_runtime.BACKEND_GEMINI,
        provider=SYNC_RAG_EMBEDDING_PROVIDER
        or (
            embedding_runtime.DEFAULT_GEMINI_PROVIDER
            if (SYNC_RAG_EMBEDDING_BACKEND or embedding_runtime.BACKEND_GEMINI)
            == embedding_runtime.BACKEND_GEMINI
            else SYNC_RAG_EMBEDDING_PROVIDER
        ),
        model=SYNC_RAG_EMBEDDING_MODEL or DEFAULT_SYNC_RAG_EMBEDDING_MODEL,
        output_dimension=int(SYNC_RAG_OUTPUT_DIMENSIONALITY or 768),
        endpoint=SYNC_RAG_EMBEDDING_ENDPOINT or "",
        timeout_seconds=float(
            SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS
            or embedding_runtime.DEFAULT_TIMEOUT_SECONDS
        ),
        api_key_env=SYNC_RAG_EMBEDDING_API_KEY_ENV or "",
        native_query_task_type=SYNC_RAG_QUERY_TASK_TYPE,
        native_document_task_type=SYNC_RAG_DOCUMENT_TASK_TYPE,
    )
    if (
        settings.backend == embedding_runtime.BACKEND_OPENAI_COMPATIBLE
        and not settings.endpoint
    ):
        custom = (CUSTOM_LITELLM_PROVIDERS or {}).get(settings.provider)
        base_url = getattr(custom, "base_url", "") if custom is not None else ""
        if base_url:
            settings = replace(settings, endpoint=str(base_url))
    return settings


def _resolve_embedding_api_key(settings):
    if settings.backend != embedding_runtime.BACKEND_OPENAI_COMPATIBLE:
        return None
    env_name = settings.api_key_env
    if not env_name:
        custom = (CUSTOM_LITELLM_PROVIDERS or {}).get(settings.provider)
        env_name = getattr(custom, "api_key_env", "") if custom is not None else ""
    if not env_name and settings.provider == "openai":
        env_name = "OPENAI_API_KEY"
    value = embedding_runtime.resolve_api_key_from_env(env_name)
    return value or None


def _litellm_embedding_transport():
    try:
        import litellm
    except ImportError as exc:
        raise embedding_runtime.EmbeddingRuntimeError(
            "LiteLLM is not installed for the OpenAI-compatible embedding backend",
            reason="litellm_not_installed",
        ) from exc
    return litellm.embedding


def build_live_embedding_adapter(settings=None):
    """Build an adapter from the current runtime, injecting live transports."""

    settings = settings or current_sync_embedding_settings()
    if settings.backend == embedding_runtime.BACKEND_GEMINI:
        return embedding_runtime.build_embedding_adapter(
            settings,
            gemini_client=create_genai_client(),
        )
    return embedding_runtime.build_embedding_adapter(
        settings,
        openai_transport=_litellm_embedding_transport(),
        api_key=_resolve_embedding_api_key(settings),
    )


def get_sync_source_index_char_budget():
    return max(0, int(SYNC_SOURCE_INDEX_CHAR_BUDGET or 0))


def _attach_store_document_identity(store, settings=None, *, rebuild=False):
    settings = settings or current_sync_embedding_settings()
    return embedding_runtime.ensure_store_document_identity(
        store,
        settings.document_identity(),
        rebuild=rebuild,
    )


def get_sync_rag_store():
    global _SYNC_RAG_STORE, SYNC_RAG_STORE_DIR
    if not SYNC_RAG_ENABLED:
        return None
    if not SYNC_RAG_STORE_DIR:
        SYNC_RAG_STORE_DIR = get_default_sync_rag_store_dir()
    if (
        _SYNC_RAG_STORE is None
        or os.path.abspath(_SYNC_RAG_STORE.store_dir) != os.path.abspath(SYNC_RAG_STORE_DIR)
    ):
        _SYNC_RAG_STORE = JsonRagStore(SYNC_RAG_STORE_DIR)
        _SYNC_RAG_STORE.set_metadata(
            owner="gemini_translate.py",
            mode="sync",
            embedding_model=SYNC_RAG_EMBEDDING_MODEL,
            query_task_type=SYNC_RAG_QUERY_TASK_TYPE,
            document_task_type=SYNC_RAG_DOCUMENT_TASK_TYPE,
            output_dimensionality=SYNC_RAG_OUTPUT_DIMENSIONALITY,
        )
        _attach_store_document_identity(_SYNC_RAG_STORE)
    return _SYNC_RAG_STORE


def get_sync_source_index_store():
    global _SYNC_SOURCE_INDEX_STORE, SYNC_SOURCE_INDEX_STORE_DIR
    if not SYNC_SOURCE_INDEX_STORE_DIR:
        SYNC_SOURCE_INDEX_STORE_DIR = get_default_source_index_store_dir()
    if (
        _SYNC_SOURCE_INDEX_STORE is None
        or os.path.abspath(_SYNC_SOURCE_INDEX_STORE.store_dir)
        != os.path.abspath(SYNC_SOURCE_INDEX_STORE_DIR)
    ):
        _SYNC_SOURCE_INDEX_STORE = JsonSourceIndexStore(SYNC_SOURCE_INDEX_STORE_DIR)
        _attach_store_document_identity(_SYNC_SOURCE_INDEX_STORE)
    return _SYNC_SOURCE_INDEX_STORE


def compact_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def build_sync_rag_query_text(target_items):
    target_lines = [
        compact_text(item.get("text", ""))
        for item in target_items
        if compact_text(item.get("text", ""))
    ]
    if not target_lines:
        return ""
    return "Target:\n" + "\n".join(f"- {text}" for text in target_lines)


def embed_texts(contents, task_type):
    if not contents:
        return []
    settings = current_sync_embedding_settings()
    last_error = None
    key_attempts = api_key_rotation_attempts()
    attempts = (
        max(3, key_attempts * 2)
        if settings.backend == embedding_runtime.BACKEND_GEMINI
        else 1
    )
    for attempt in range(1, attempts + 1):
        try:
            adapter = build_live_embedding_adapter(settings)
            return embedding_runtime.embed_texts(
                adapter,
                contents,
                task_type,
                timeout_seconds=settings.timeout_seconds,
            )
        except EmbeddingBackendError as exc:
            last_error = exc
            if exc.retryable and attempt < attempts and settings.backend == embedding_runtime.BACKEND_GEMINI:
                rotate_api_key()
                continue
            raise
        except embedding_runtime.EmbeddingRuntimeError:
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding request failed without a captured exception.")


def embed_sync_query_text(query_text):
    query_text = compact_text(query_text)
    if not query_text:
        return []
    vectors = embed_texts([query_text], SYNC_RAG_QUERY_TASK_TYPE)
    return vectors[0] if vectors else []


def retrieve_sync_glossary_hits(target_items):
    """Return every lexical glossary hit for the current TARGET batch.

    Delegates to the shared implementation (issue #346, D2) with the runtime's
    configured glossary inputs. The list is deliberately not truncated by
    ``SYNC_RAG_TOP_K_TERMS``: that cap applies only to the RAG ``LOCKED TERMS``
    reference list, while the lexical injection required by issue #338 must
    always carry the batch's real hits (otherwise normalize hits could evict
    non-translatable terms and cause names to be mistranslated).
    """
    return translation_plan.retrieve_lexical_glossary_hits(
        target_items,
        normalize_map=NORMALIZE_TRANSLATION_MAP,
        preserve_terms=PRESERVE_TERMS,
        non_translatable_exact=NON_TRANSLATABLE_EXACT,
    )


def build_sync_local_context(tasks, start, end, before_limit, after_limit):
    """Build a file-bounded ContextWindow for one sync batch (issue #338).

    Delegates to the shared block-bounded window builder (issue #346, D1) so
    sync and batch consume the same local-context algorithm.

    The window is limited to the current file's pending task sequence (the
    caller passes one file at a time) and stops at translate-block boundaries
    so context cannot silently cross scenes when a block is identifiable.

    Returns ``(ContextWindow, diagnostics)`` where diagnostics records the
    applied limits, actual item/character counts, block bounding, and whether
    the budget truncated the context.
    """
    return translation_plan.build_local_context_window(tasks, start, end, before_limit, after_limit)


def format_sync_glossary_hits_block(hits, empty_label="(none)"):
    return prompt_context.format_glossary_hits_block(hits, empty_label)


def format_sync_history_hits_block(hits, empty_label="(none)"):
    return prompt_context.format_history_hits_block(
        hits,
        empty_label,
        char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
        include_source_text=False,
    )


def get_sync_story_graph():
    global _SYNC_STORY_GRAPH, _SYNC_STORY_GRAPH_PATH
    if not SYNC_STORY_MEMORY_ENABLED:
        return None
    graph_path = os.path.abspath(SYNC_STORY_MEMORY_GRAPH_FILE) if SYNC_STORY_MEMORY_GRAPH_FILE else ""
    if _SYNC_STORY_GRAPH is None or _SYNC_STORY_GRAPH_PATH != graph_path:
        _SYNC_STORY_GRAPH = story_memory.load_story_graph(graph_path)
        _SYNC_STORY_GRAPH_PATH = graph_path
    return _SYNC_STORY_GRAPH


def retrieve_sync_story_hits(target_items):
    if not SYNC_STORY_MEMORY_ENABLED:
        return None
    file_rel_path = ""
    for item in target_items or []:
        if isinstance(item, dict) and item.get("file_rel_path"):
            file_rel_path = item.get("file_rel_path")
            break
    return story_memory.retrieve_story_hits(
        get_sync_story_graph(),
        file_rel_path,
        target_items,
        top_k_relations=SYNC_STORY_MEMORY_TOP_K_RELATIONS,
        top_k_terms=SYNC_STORY_MEMORY_TOP_K_TERMS,
        include_scene_summary=SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
    )


def retrieve_sync_history_hits(target_items):
    if not SYNC_RAG_ENABLED:
        return [], {"enabled": False}
    store = get_sync_rag_store()
    if store is None or store.count_history() <= 0:
        return [], {"enabled": True, "reason": "empty_history_store"}

    query_text = advanced_context.build_source_only_query_text(target_items)
    if not query_text:
        return [], {"enabled": True, "reason": "empty_query"}

    settings = current_sync_embedding_settings()
    try:
        compatibility = embedding_runtime.store_compatibility_report(
            store,
            settings.query_identity(),
        )
        if not compatibility.compatible:
            return [], {
                "enabled": True,
                "reason": "rebuild_store",
                "action": compatibility.action,
                "embedding_compatibility": compatibility.to_dict(),
            }
        query_vector = embed_sync_query_text(query_text)
        return advanced_context.retrieve_history_hits_compatible(
            store,
            query_vector,
            settings.query_identity(),
            top_k=SYNC_RAG_TOP_K_HISTORY,
            min_similarity=SYNC_RAG_MIN_SIMILARITY,
            char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
            query_text=query_text,
        )
    except Exception as exc:
        diagnostics = embedding_runtime.public_error_diagnostics(exc)
        print(
            "Warning: Sync RAG history retrieval failed: "
            f"{diagnostics.get('error_category') or diagnostics.get('failure_reason')}"
        )
        return [], {"enabled": True, **diagnostics}


def retrieve_sync_source_hits(target_items):
    if not SYNC_SOURCE_INDEX_ENABLED:
        return [], {"enabled": False}
    query_text = advanced_context.build_source_only_query_text(target_items)
    char_budget = get_sync_source_index_char_budget()
    settings = current_sync_embedding_settings()
    embedding_provider = advanced_context.embedding_provider_diagnostics(settings)
    if not query_text:
        return [], {
            "enabled": True,
            "reason": "empty_query",
            "source_context_char_budget": char_budget,
            "embedding_provider": embedding_provider,
        }
    try:
        store = get_sync_source_index_store()
        if store is None or store.count_segments() <= 0:
            return [], {
                "enabled": True,
                "reason": "empty_source_store",
                "source_context_char_budget": char_budget,
                "store_dir": getattr(store, "store_dir", SYNC_SOURCE_INDEX_STORE_DIR or ""),
                "store_schema_version": (
                    (getattr(store, "metadata", {}) or {}).get("schema_version")
                    if store is not None
                    else None
                ),
                "embedding_provider": embedding_provider,
            }
        compatibility = embedding_runtime.store_compatibility_report(
            store,
            settings.query_identity(),
        )
        if not compatibility.compatible:
            return [], {
                "enabled": True,
                "reason": "rebuild_store",
                "action": compatibility.action,
                "embedding_compatibility": compatibility.to_dict(),
                "source_context_char_budget": char_budget,
                "store_dir": getattr(store, "store_dir", SYNC_SOURCE_INDEX_STORE_DIR or ""),
                "store_schema_version": (
                    (getattr(store, "metadata", {}) or {}).get("schema_version")
                    if store is not None
                    else None
                ),
                "embedding_provider": embedding_provider,
            }
        query_vector = embed_sync_query_text(query_text)
        hits, stats = advanced_context.retrieve_source_hits_compatible(
            store,
            query_vector,
            settings.query_identity(),
            top_k=SYNC_SOURCE_INDEX_TOP_K,
            min_similarity=SYNC_SOURCE_INDEX_MIN_SIMILARITY,
            char_limit=SYNC_SOURCE_INDEX_CHAR_LIMIT,
            query_text=query_text,
            embedding_model=SYNC_RAG_EMBEDDING_MODEL,
            embedding_task_type=SYNC_RAG_DOCUMENT_TASK_TYPE,
            embedding_dim=SYNC_RAG_OUTPUT_DIMENSIONALITY,
            char_budget=char_budget,
        )
        stats["embedding_provider"] = embedding_provider
        return hits, stats
    except Exception as exc:
        diagnostics = embedding_runtime.public_error_diagnostics(exc)
        print(
            "Warning: Sync source index retrieval failed: "
            f"{diagnostics.get('error_category') or diagnostics.get('failure_reason')}"
        )
        return [], {
            "enabled": True,
            "source_context_char_budget": char_budget,
            "embedding_provider": embedding_provider,
            **diagnostics,
        }


def load_sync_injectable_project_context(file_rel_path="", line_numbers=None):
    """Load a fresh published brief only when the Sync inject setting is on."""

    empty = {
        "text": "",
        "injectable": False,
        "reason": "injection_disabled",
        "status": {},
        "diagnostics": "",
        "labels": [],
        "routes": [],
        "local_diagnostics": "",
    }
    if not SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF:
        return empty
    from project_analysis import (
        compute_current_project_analysis_fingerprint,
        load_injectable_project_context,
    )

    current_fp = compute_current_project_analysis_fingerprint(BASE_DIR or None)
    if not current_fp:
        empty["reason"] = "source_fingerprint_unavailable"
        return empty
    normalized_lines = []
    for raw_value in line_numbers or []:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            normalized_lines.append(value)
    try:
        payload = load_injectable_project_context(
            base_dir=BASE_DIR or None,
            expected_source_fingerprint=current_fp,
            file_rel_path=file_rel_path,
            line_numbers=normalized_lines,
            enabled=True,
        )
    except Exception as exc:
        diagnostics = embedding_runtime.public_error_diagnostics(exc)
        print(
            "Warning: Sync published project analysis unavailable: "
            f"{diagnostics.get('failure_reason')}"
        )
        empty["reason"] = diagnostics.get("failure_reason") or "analysis_unavailable"
        return empty
    if not payload.get("injectable"):
        payload.setdefault("labels", [])
        payload.setdefault("routes", [])
        payload["text"] = ""
    return payload


def get_random_delay():
    return random.uniform(MIN_DELAY, MAX_DELAY)


def _normalize_string_prefix(prefix):
    if not prefix:
        return ""
    if any(ch.lower() in {"b", "f"} for ch in prefix):
        return prefix
    return "".join(ch for ch in prefix if ch.lower() != "r")


def parse_string_literal_format(token_string):
    match = STRING_LITERAL_PREFIX_RE.match(token_string or "")
    if not match:
        return "", '"'
    prefix = _normalize_string_prefix(match.group("prefix") or "")
    quote = match.group("quote") or '"'
    return prefix, quote


def quote_with(text, quote, prefix=""):
    escaped = text
    quote_char = (quote or '"')[0]
    for old, new in SPECIAL_ESCAPES:
        if old == quote_char:
            continue
        escaped = escaped.replace(old, new)
    escaped = escaped.replace(quote_char, "\\" + quote_char)
    return f"{prefix}{quote}{escaped}{quote}"


def contains_chinese(text):
    if not text:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

def _translated_has_renpy_field_for_term(translated, term):
    if not translated or not term or not term.isalpha():
        return False
    expected = term.lower()
    for match in RENPY_FIELD_TOKEN_RE.finditer(translated):
        field_name = match.group("name") or ""
        field_tokens = [token.lower() for token in WORD_TOKEN_RE.findall(field_name)]
        if field_tokens and field_tokens[0] == expected:
            return True
    return False


def _source_usage_excluded_for_preserve_term(original, term):
    for pattern in PRESERVE_TERM_SOURCE_EXCLUSION_PATTERNS.get(term, []):
        if pattern.search(original or ""):
            return True
    return False


def _translated_contains_preserve_alias(translated, term):
    for alias in PRESERVE_TERM_ALIASES.get(term, ()):
        if alias and alias in (translated or ""):
            return True
    return False


def missing_preserved_terms(original, translated):
    if not original or not translated:
        return []
    missing = []
    for term in PRESERVE_TERMS:
        if not term or term not in original:
            continue
        if _source_usage_excluded_for_preserve_term(original, term):
            continue
        if _translated_contains_preserve_alias(translated, term):
            continue
        if term.startswith("[") and term.endswith("]"):
            term_field = RENPY_FIELD_TOKEN_RE.fullmatch(term)
            if term_field:
                field_name = term_field.group("name")
                field_pattern = re.compile(rf"\[{re.escape(field_name)}(?:![^\]]*)?\]")
                if field_pattern.search(translated):
                    continue
        # Avoid false positives for short alphabetic fragments (e.g., "Mo" in "Moon").
        if term.isalpha() and len(term) <= 3:
            pattern = rf"(?<![A-Za-z0-9_']){re.escape(term)}(?![A-Za-z0-9_'])"
            if not re.search(pattern, original):
                continue
            if not re.search(pattern, translated) and not _translated_has_renpy_field_for_term(translated, term):
                missing.append(term)
            continue
        if term not in translated:
            missing.append(term)
    return missing


def is_non_translatable(text):
    if not text:
        return True
    stripped = text.strip()
    if text in PRESERVE_TERMS or text in NON_TRANSLATABLE_EXACT:
        return True
    if stripped in PRESERVE_TERMS or stripped in NON_TRANSLATABLE_EXACT:
        return True
    if NON_TRANSLATABLE_TAG_ONLY.match(stripped):
        return True
    if NON_TRANSLATABLE_SYMBOLS.match(stripped):
        return True
    if FILE_NAME_PATTERN.match(stripped) or FILE_NAME_SIMPLE_RE.match(stripped):
        return True
    if ROMAN_NUMERAL_LABEL_RE.match(stripped):
        return True
    if "%" in stripped and STRFTIME_FORMAT_RE.match(stripped):
        return True
    if RENPY_IDENTIFIER_LABEL_RE.match(stripped):
        return True
    if LETTER_SEQUENCE_RE.match(stripped):
        return True
    for pattern in NON_TRANSLATABLE_PATTERNS:
        if pattern.match(stripped):
            return True
    if is_name_hint(text):
        return True
    if is_sound_name_hint(text):
        return True
    if is_name_like(text):
        return True
    # Treat short effects/onomatopoeia as non-translatable
    if is_short_effect(text):
        return True

    return False



def is_english_like(text):
    if not text:
        return False
    if contains_chinese(text):
        return False
    if is_non_translatable(text):
        return False
    if any(ch.isalpha() for ch in text):
        return True
    return False


def is_name_hint(text):
    cleaned = RENPY_TAG_RE.sub("", text or "")
    cleaned = RENPY_FIELD_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False

    tokens = WORD_TOKEN_RE.findall(cleaned)
    if not tokens:
        return False

    if len(tokens) <= 3:
        lower_tokens = [token.lower() for token in tokens]
        if all(token in PRESERVE_TERMS_LOWER for token in lower_tokens):
            return True

    return False


def is_sound_name_hint(text):
    cleaned = RENPY_TAG_RE.sub("", text or "")
    cleaned = RENPY_FIELD_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False

    tokens = WORD_TOKEN_RE.findall(cleaned)
    if not tokens or len(tokens) > 3:
        return False

    lower_tokens = [token.lower() for token in tokens]
    if not any(token in PRESERVE_TERMS_LOWER for token in lower_tokens):
        return False

    for token in lower_tokens:
        if token in PRESERVE_TERMS_LOWER:
            continue
        if len(token) <= 4:
            continue
        if not VOWEL_RE.search(token):
            continue
        if REPEATED_CHAR_RE.search(token):
            continue
        return False

    return True


def is_name_like(text):
    cleaned = RENPY_TAG_RE.sub("", text or "")
    cleaned = RENPY_FIELD_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False
    if contains_chinese(cleaned):
        return False
    tokens = WORD_TOKEN_RE.findall(cleaned)
    if not tokens:
        return False
    lower_tokens = [token.lower() for token in tokens]
    return all(token in PRESERVE_TERMS_LOWER for token in lower_tokens)


def is_short_effect(text):
    if not text:
        return False
    cleaned = RENPY_TAG_RE.sub("", text)
    cleaned = RENPY_FIELD_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return True

    # Pure non-letter symbols or mostly punctuation
    if NON_TRANSLATABLE_SYMBOLS.match(cleaned):
        return True

    tokens = WORD_TOKEN_RE.findall(cleaned)
    if not tokens:
        return True

    # Stutters and dotted filler like "V.... S-s-ercap" (only if very short)
    if STUTTER_PATTERN.search(cleaned) or MULTI_DOT_PATTERN.search(cleaned):
        if len(tokens) <= 1 and len(cleaned) <= EFFECT_MAX_LENGTH:
            return True

    # Short single-word effects like "Grrrr" or "Hngh"
    if len(tokens) == 1:
        token = tokens[0]
        if token.lower() in PRESERVE_TERMS_LOWER:
            return True
        if REPEATED_CHAR_RE.search(token):
            return True
        if not VOWEL_RE.search(token) and len(token) <= 6:
            return True

    return False

def apply_normalization(text):
    if not text:
        return text
    for old, new in NORMALIZE_TRANSLATION_MAP.items():
        text = text.replace(old, new)
    return text


def _extract_word_tokens(text):
    cleaned = RENPY_TAG_RE.sub("", text or "")
    cleaned = RENPY_FIELD_RE.sub("", cleaned)
    return [token.lower() for token in WORD_TOKEN_RE.findall(cleaned)]


def _term_token_sequence_matches(tokens, terms):
    token_tuple = tuple(tokens)
    if not token_tuple:
        return False
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            continue
        term_tokens = tuple(_extract_word_tokens(term))
        if not term_tokens:
            continue
        if token_tuple == term_tokens:
            return True
        if len(token_tuple) % len(term_tokens) == 0:
            repeated = term_tokens * (len(token_tuple) // len(term_tokens))
            if token_tuple == repeated:
                return True
    return False


def allow_non_chinese_term_translation(original, translated, known_terms=None):
    if not original or not translated:
        return False
    if contains_chinese(translated):
        return False
    if is_non_translatable(original):
        return True

    original_tokens = _extract_word_tokens(original)
    translated_tokens = _extract_word_tokens(translated)
    if not original_tokens or not translated_tokens:
        return False
    if original_tokens != translated_tokens:
        return False

    allowed_terms = set(PRESERVE_TERMS_LOWER)
    known_term_strings = list(PRESERVE_TERMS)
    if known_terms:
        known_term_strings.extend(
            str(term).strip()
            for term in known_terms
            if isinstance(term, str) and str(term).strip()
        )
        allowed_terms.update(
            str(term).strip().lower()
            for term in known_terms
            if isinstance(term, str) and str(term).strip()
        )
    if not allowed_terms and not known_term_strings:
        return False
    if _term_token_sequence_matches(original_tokens, known_term_strings):
        return True
    return all(token in allowed_terms for token in original_tokens)


def validate_translation(original, translated):
    if not translated or not translated.strip():
        return False, "Empty translation"

    missing = missing_preserved_terms(original, translated)
    if missing:
        return False, f"Preserved terms missing: {', '.join(missing)}"

    original_tokens = Counter(
        RENPY_TAG_RE.findall(original)
        + RENPY_FIELD_RE.findall(original)
        + PERCENT_FORMAT_TOKEN_RE.findall(original)
    )
    translated_tokens = Counter(
        RENPY_TAG_RE.findall(translated)
        + RENPY_FIELD_RE.findall(translated)
        + PERCENT_FORMAT_TOKEN_RE.findall(translated)
    )
    if original_tokens != translated_tokens:
        missing_tokens = list((original_tokens - translated_tokens).elements())
        added_tokens = list((translated_tokens - original_tokens).elements())
        details = []
        if missing_tokens:
            details.append(f"missing {', '.join(missing_tokens)}")
        if added_tokens:
            details.append(f"added {', '.join(added_tokens)}")
        return False, f"Ren'Py placeholders/tags changed: {'; '.join(details)}"

    # If original is purely non-translatable, allow untouched output
    if is_non_translatable(original):
        return True, "OK"

    # If original was English and translated still has no Chinese, reject
    if is_english_like(original) and not contains_chinese(translated):
        if allow_non_chinese_term_translation(original, translated):
            return True, "OK"
        return False, "No Chinese characters"

    return True, "OK"

def load_progress():
    if not os.path.exists(PROGRESS_LOG):
        return {}
    try:
        with open(PROGRESS_LOG, "r", encoding="utf-8-sig") as handle:
            raw_progress = json.load(handle)
    except Exception:
        return {}
    if not isinstance(raw_progress, dict):
        return {}
    normalized = {}
    for filename, entries in raw_progress.items():
        normalized[str(filename)] = _normalize_progress_entries(entries)
    return normalized


def save_progress(progress):
    try:
        atomic_write_json(PROGRESS_LOG, progress, encoding="utf-8-sig", ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")


def _normalize_progress_entry(value):
    if isinstance(value, int):
        return f"line:{value}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.isdigit():
            return f"line:{int(text)}"
        return text
    return ""


def _normalize_progress_entries(values):
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    result = []
    seen = set()
    for value in values:
        normalized = _normalize_progress_entry(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _progress_line_entry(line_idx):
    return f"line:{int(line_idx)}"


def _progress_entry_for_task(task):
    return f"task:{int(task['line'])}:{int(task['start'])}"


def update_progress(filename, translated_lines):
    progress = load_progress()
    existing_entries = _normalize_progress_entries(progress.get(filename, []))
    new_entries = _normalize_progress_entries(translated_lines)
    progress[filename] = sorted(set(existing_entries + new_entries))
    save_progress(progress)


def _progress_key_for_path(file_path):
    try:
        rel_path = os.path.relpath(file_path, TL_DIR)
    except ValueError:
        rel_path = file_path
    normalized = _normalize_rel_path(rel_path)
    return normalized or os.path.basename(file_path)


def _upgrade_legacy_progress_keys(progress, file_paths):
    if not isinstance(progress, dict):
        return {}

    # Normalize separators / relative forms first so adapter file_rel_path keys
    # (always forward-slash) match stored progress keys from Windows paths.
    original_items = list(progress.items())
    progress.clear()
    separator_migrated = False
    for key, value in original_items:
        if not isinstance(key, str):
            continue
        norm_key = _normalize_rel_path(key) or key
        if norm_key != key:
            separator_migrated = True
        if norm_key in progress:
            merged_entries = set(_normalize_progress_entries(progress[norm_key]))
            merged_entries.update(_normalize_progress_entries(value))
            progress[norm_key] = sorted(merged_entries)
        else:
            progress[norm_key] = value

    basename_map = {}
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        basename_map.setdefault(basename, []).append(_progress_key_for_path(file_path))

    migrated = separator_migrated
    for basename, rel_paths in basename_map.items():
        legacy_lines = progress.get(basename)
        if legacy_lines is None:
            continue

        unique_rel_paths = _dedupe_keep_order(rel_paths)
        if len(unique_rel_paths) != 1:
            print(
                f"Warning: Ignoring ambiguous legacy progress key '{basename}' "
                f"({len(unique_rel_paths)} matching files)."
            )
            continue

        progress_key = unique_rel_paths[0]
        if progress_key == basename:
            # Basename already is the canonical key for a top-level TL file.
            continue
        merged_entries = set(_normalize_progress_entries(progress.get(progress_key, [])))
        merged_entries.update(_normalize_progress_entries(legacy_lines))
        progress[progress_key] = sorted(merged_entries)
        progress.pop(basename, None)
        migrated = True

    if migrated:
        save_progress(progress)
    return progress


def build_sync_adapter_writeback_plan(adapter, snapshot, file_rel_path, tasks, replacements):
    """Validate replacements and return a plan with its exact source tuple."""

    occurrences = tuple(
        occurrence
        for occurrence in snapshot.occurrences
        if occurrence.unit.file_rel_path == file_rel_path
    )
    occurrences_by_unit_id = {
        occurrence.unit.id: occurrence
        for occurrence in occurrences
        if occurrence.unit.id
    }
    tasks_by_span = {
        (
            int(task.get('line') or 0),
            int(task.get('start') or 0),
            int(task.get('end') or 0),
        ): task
        for task in tasks
    }
    validated = []
    used_occurrence_ids = set()

    for line, line_replacements in replacements.items():
        for replacement in line_replacements:
            start, end, translated, _prefix, _quote = replacement[:5]
            task = tasks_by_span.get((int(line), int(start), int(end)))
            occurrence = None
            if task is not None:
                occurrence = occurrences_by_unit_id.get(str(task.get('id') or ''))
            if occurrence is None:
                candidates = [
                    candidate
                    for candidate in occurrences
                    if candidate.unit.line == int(line)
                    and candidate.unit.start == int(start)
                    and candidate.unit.end == int(end)
                    and (
                        not task
                        or not task.get('text')
                        or candidate.unit.text == task.get('text')
                        or candidate.unit.source_text == task.get('text')
                    )
                ]
                if len(candidates) == 1:
                    occurrence = candidates[0]
                elif len(candidates) > 1:
                    raise WritebackPlanError(
                        'common.locator.unresolved',
                        f'Ambiguous sync occurrence at {file_rel_path}:{line}:{start}-{end}.',
                    )
            if occurrence is None:
                raise WritebackPlanError(
                    'common.locator.unresolved',
                    f'Sync occurrence could not be resolved at {file_rel_path}:{line}:{start}-{end}.',
                )
            if occurrence.occurrence_id in used_occurrence_ids:
                raise WritebackPlanError(
                    'common.writeback.span_overlap',
                    f'Duplicate sync occurrence at {file_rel_path}:{line}:{start}-{end}.',
                )
            used_occurrence_ids.add(occurrence.occurrence_id)
            validation = adapter.validate_translation(
                occurrence,
                str(translated or ''),
            )
            if validation.status != 'pass':
                codes = ','.join(validation.reason_codes) or 'adapter.validation.block'
                raise WritebackPlanError(
                    'adapter.validation.block',
                    f'Adapter validation blocked {file_rel_path}:{line}: {codes}.',
                )
            validated.append(
                ValidatedTranslation(
                    occurrence=occurrence,
                    translated_text=str(translated or ''),
                    validation=validation,
                )
            )

    live_sources = tuple(
        document
        for document in snapshot.project.source_documents
        if document.file_rel_path == file_rel_path
    )
    plan = adapter.build_writeback_plan(
        snapshot.project,
        tuple(validated),
        live_sources,
    )
    return plan, live_sources


def build_sync_adapter_preview(adapter, snapshot, file_rel_path, tasks, replacements):
    """Build and render one sync preview file without aborting sibling files."""

    try:
        plan, live_sources = build_sync_adapter_writeback_plan(
            adapter,
            snapshot,
            file_rel_path,
            tasks,
            replacements,
        )
        rendered = render_writeback_plan(plan, live_sources)
    except (KeyError, ValueError) as exc:
        return None, None, {
            "relative_path": file_rel_path,
            "reason_code": getattr(exc, "reason_code", "adapter.writeback.block"),
            "message": str(exc),
        }
    return plan, rendered, None


def render_replacement_lines(lines, replacements):
    """Return replacement output without modifying the caller's source lines."""
    rendered = list(lines)
    if not replacements:
        return rendered

    for line_idx, repls in replacements.items():
        if line_idx >= len(rendered):
            continue
        line = rendered[line_idx]
        for repl in sorted(repls, key=lambda x: x[0], reverse=True):
            if len(repl) == 4:
                start, end, translated, quote = repl
                prefix = ""
            else:
                start, end, translated, prefix, quote = repl[:5]
            if start < 0 or end > len(line) or start > end:
                continue
            normalized = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
            line = line[:start] + quote_with(normalized, quote, prefix=prefix) + line[end:]
        rendered[line_idx] = line
    return rendered


def commit_replacements(path, lines, replacements):
    """Apply replacements in memory, then atomically replace the target file.

    Writes to a same-directory temp file, fsyncs, and uses ``os.replace`` so a
    crash or I/O error cannot leave a truncated ``.rpy`` behind.
    """
    if not replacements:
        return

    rendered = render_replacement_lines(lines, replacements)
    lines[:] = rendered
    atomic_write_lines(path, rendered, encoding="utf-8")


def sync_rag_hash_key(text):
    return hash_text(text)


def extract_string_token_from_line(line):
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except Exception:
        return None

    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        try:
            text_value = ast.literal_eval(token.string)
        except Exception:
            continue
        if not isinstance(text_value, str):
            continue
        prefix, quote = parse_string_literal_format(token.string)
        return {
            "text": text_value,
            "start": token.start[1],
            "end": token.end[1],
            "prefix": prefix,
            "quote": quote,
        }
    return None


def decode_string_literal_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    try:
        value = ast.literal_eval('"' + raw_text + '"')
    except Exception:
        return raw_text
    return value if isinstance(value, str) else raw_text


def is_voice_comment_match(match):
    if not match:
        return False
    prefix = str(match.group("prefix") or "").strip()
    return prefix.split(None, 1)[0:1] == ["voice"]


def is_voice_statement_line(line):
    stripped = str(line or "").strip()
    return stripped == "voice" or stripped.startswith("voice ")


def next_translation_entry_target_index(lines, index):
    next_index = index + 1
    while next_index < len(lines):
        candidate = lines[next_index]
        if not candidate.strip() or is_voice_statement_line(candidate):
            next_index += 1
            continue
        break
    return next_index


def collect_translation_entries_from_lines(lines):
    entries = []
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip("\n")
        comment_match = TL_COMMENT_SOURCE_RE.match(raw_line)
        if comment_match:
            if is_voice_comment_match(comment_match):
                index += 1
                continue
            next_index = next_translation_entry_target_index(lines, index)
            if next_index < len(lines):
                candidate_line = lines[next_index].rstrip("\n")
                if not TL_OLD_LINE_RE.match(candidate_line):
                    token = extract_string_token_from_line(lines[next_index])
                else:
                    token = None
                if token:
                    entries.append(
                        {
                            "line_number": next_index + 1,
                            "source": decode_string_literal_text(comment_match.group("text")),
                            "translation": token["text"],
                            "start": token["start"],
                            "end": token["end"],
                            "prefix": token.get("prefix", ""),
                            "quote": token["quote"],
                        }
                    )
                    index = next_index
        else:
            old_match = TL_OLD_LINE_RE.match(raw_line)
            if old_match:
                next_index = index + 1
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                if next_index < len(lines) and TL_NEW_LINE_RE.match(lines[next_index].rstrip("\n")):
                    token = extract_string_token_from_line(lines[next_index])
                    if token:
                        entries.append(
                            {
                                "line_number": next_index + 1,
                                "source": decode_string_literal_text(old_match.group("text")),
                                "translation": token["text"],
                                "start": token["start"],
                                "end": token["end"],
                                "quote": token["quote"],
                            }
                        )
                        index = next_index
        index += 1

    for entry_index, entry in enumerate(entries):
        entry["entry_index"] = entry_index
    return entries


def should_index_sync_rag_entry(entry):
    source = compact_text(entry.get("source", ""))
    translation = compact_text(entry.get("translation", ""))
    if not source or not translation:
        return False
    if source == translation:
        return False
    return True


def build_sync_rag_record(file_rel_path, group, quality_state, record_scope="file_scan"):
    source_text = "\n".join(entry.get("source", "") for entry in group).strip()
    translated_text = "\n".join(entry.get("translation", "") for entry in group).strip()
    line_start = group[0]["line_number"]
    line_end = group[-1]["line_number"]
    combined_text = f"Source:\n{source_text}\n\nTranslation:\n{translated_text}"
    memory_id = sync_rag_hash_key(f"{file_rel_path}:{line_start}:{line_end}:{source_text}")
    return {
        "memory_id": memory_id,
        "file_rel_path": file_rel_path,
        "line_start": line_start,
        "line_end": line_end,
        "source_text": source_text,
        "translated_text": translated_text,
        "combined_text": combined_text,
        "quality_state": quality_state,
        "record_scope": record_scope,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checksum": hash_text(source_text),
        "translation_checksum": hash_text(translated_text),
    }


def collect_sync_rag_records_from_entries(file_rel_path, entries, quality_state, record_scope="file_scan"):
    records = []
    segment_size = max(1, SYNC_RAG_SEGMENT_LINES)
    usable_entries = [entry for entry in entries if should_index_sync_rag_entry(entry)]
    for start in range(0, len(usable_entries), segment_size):
        group = usable_entries[start:start + segment_size]
        if group:
            records.append(build_sync_rag_record(file_rel_path, group, quality_state, record_scope=record_scope))
    return records


def collect_sync_rag_records_for_file(file_path, quality_state=None):
    if quality_state is None:
        quality_state = SYNC_RAG_QUALITY_STATE
    if not file_path or not os.path.isfile(file_path):
        return []
    try:
        file_rel_path = _normalize_rel_path(os.path.relpath(file_path, TL_DIR))
    except ValueError:
        file_rel_path = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8-sig") as handle:
        entries = collect_translation_entries_from_lines(handle.readlines())

    return collect_sync_rag_records_from_entries(file_rel_path, entries, quality_state, record_scope="file_scan")


def collect_sync_rag_records_for_tasks(file_path, tasks, quality_state=None):
    if quality_state is None:
        quality_state = SYNC_RAG_QUALITY_STATE
    if not file_path or not tasks:
        return []
    try:
        file_rel_path = _normalize_rel_path(os.path.relpath(file_path, TL_DIR))
    except ValueError:
        file_rel_path = os.path.basename(file_path)

    entries = []
    for task in tasks:
        translated_text = task.get("translated_text")
        if not translated_text:
            continue
        entries.append(
            {
                "line_number": int(task["line"]) + 1,
                "source": task.get("text", ""),
                "translation": translated_text,
                "start": task.get("start", 0),
                "end": task.get("end", 0),
                "quote": task.get("quote", '"'),
            }
        )
    return collect_sync_rag_records_from_entries(file_rel_path, entries, quality_state, record_scope="task")


def embed_sync_history_records(records):
    embedded_records = []
    batch_size = 16
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        vectors = embed_texts([record["combined_text"] for record in batch], SYNC_RAG_DOCUMENT_TASK_TYPE)
        for record, vector in zip(batch, vectors):
            enriched = dict(record)
            enriched["embedding"] = vector
            enriched["embedding_model"] = SYNC_RAG_EMBEDDING_MODEL
            enriched["embedding_task_type"] = SYNC_RAG_DOCUMENT_TASK_TYPE
            enriched["embedding_dim"] = len(vector)
            embedded_records.append(enriched)
    return embedded_records


def upsert_sync_rag_records(store, records):
    pending_records = []
    for record in records:
        existing = store.get_history_record(record["memory_id"])
        if (
            existing
            and existing.get("source_checksum") == record["source_checksum"]
            and existing.get("translation_checksum") == record["translation_checksum"]
            and existing.get("embedding_model") == SYNC_RAG_EMBEDDING_MODEL
            and existing.get("embedding_task_type") == SYNC_RAG_DOCUMENT_TASK_TYPE
            and existing.get("embedding_dim") == SYNC_RAG_OUTPUT_DIMENSIONALITY
        ):
            continue
        pending_records.append(record)

    stats = {
        "pending": len(pending_records),
        "upserted": 0,
    }
    if not pending_records:
        return stats

    identity_status = _attach_store_document_identity(store)
    if not identity_status.get("ready"):
        stats["error"] = "rebuild_store"
        stats["embedding_compatibility"] = identity_status
        return stats

    embedded_records = embed_sync_history_records(pending_records)
    stats["upserted"] = store.upsert_history(embedded_records)
    return stats


def sync_rag_store_for_tasks(file_path, tasks, quality_state=None):
    if quality_state is None:
        quality_state = SYNC_RAG_QUALITY_STATE
    if not SYNC_RAG_ENABLED or not SYNC_RAG_UPDATE_ON_SUCCESS:
        return {"enabled": False}
    store = get_sync_rag_store()
    if store is None:
        return {"enabled": True, "error": "RAG store unavailable"}

    base_records = collect_sync_rag_records_for_tasks(file_path, tasks, quality_state=quality_state)
    stats = {
        "enabled": True,
        "store_dir": store.store_dir,
        "scanned": len(base_records),
        "pending": 0,
        "pruned": 0,
        "upserted": 0,
        "history_records_before": store.count_history(),
    }
    try:
        stats.update(upsert_sync_rag_records(store, base_records))
        stats["history_records_after"] = store.count_history()
    except Exception as exc:
        print(f"Warning: Failed to update sync RAG store: {exc}")
        stats["error"] = str(exc)
        stats["history_records_after"] = store.count_history()
    return stats


def sync_rag_store_for_file(file_path, quality_state=None):
    if quality_state is None:
        quality_state = SYNC_RAG_QUALITY_STATE
    if not SYNC_RAG_ENABLED or not SYNC_RAG_UPDATE_ON_SUCCESS:
        return {"enabled": False}
    store = get_sync_rag_store()
    if store is None:
        return {"enabled": True, "error": "RAG store unavailable"}

    base_records = collect_sync_rag_records_for_file(file_path, quality_state=quality_state)
    current_record_ids = {record["memory_id"] for record in base_records}
    try:
        file_rel_path = _normalize_rel_path(os.path.relpath(file_path, TL_DIR))
    except ValueError:
        file_rel_path = os.path.basename(file_path)
    obsolete_record_ids = [
        memory_id
        for memory_id in store.history_ids_for_file(file_rel_path, quality_state=quality_state)
        if memory_id not in current_record_ids
        and (store.get_history_record(memory_id) or {}).get("record_scope") == "file_scan"
    ]

    stats = {
        "enabled": True,
        "store_dir": store.store_dir,
        "scanned": len(base_records),
        "pending": 0,
        "pruned": 0,
        "upserted": 0,
        "history_records_before": store.count_history(),
    }

    try:
        upsert_stats = upsert_sync_rag_records(store, base_records)
        stats.update(upsert_stats)
        if obsolete_record_ids:
            stats["pruned"] = store.delete_history(obsolete_record_ids)
        stats["history_records_after"] = store.count_history()
    except Exception as exc:
        print(f"Warning: Failed to update sync RAG store: {exc}")
        stats["error"] = str(exc)
        stats["history_records_after"] = store.count_history()
    return stats


def maybe_update_sync_rag_store(file_path, tasks=None, full_file=False):
    if not SYNC_RAG_ENABLED or not SYNC_RAG_UPDATE_ON_SUCCESS:
        return
    if full_file:
        summary = sync_rag_store_for_file(file_path, quality_state=SYNC_RAG_QUALITY_STATE)
    else:
        summary = sync_rag_store_for_tasks(file_path, tasks or [], quality_state=SYNC_RAG_QUALITY_STATE)
    if summary.get("upserted"):
        print(f"  Sync RAG store updated: {summary.get('upserted', 0)} entries", flush=True)
    if summary.get("pruned"):
        print(f"  Sync RAG store pruned: {summary.get('pruned', 0)} obsolete entries", flush=True)
    elif summary.get("error"):
        print(f"  Warning: Sync RAG store update skipped: {summary['error']}", flush=True)


def log_failure(batch, error):
    try:
        with open(FAILED_LOG, "a", encoding="utf-8-sig") as handle:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in batch:
                handle.write(json.dumps({
                    "timestamp": timestamp,
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "error": str(error),
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  Warning: Could not log failure: {e}")


def build_prompt(
    items,
    glossary_hits=None,
    history_hits=None,
    story_hits=None,
    context_window=None,
    macro_setting=None,
    normalize_map=None,
    non_translatable_terms=None,
):
    """Build the sync translation prompt for one batch.

    ``context_window`` is a reference-only local context window bounded to the
    current file; ``macro_setting`` overrides the loaded ``SYNC_MACRO_SETTING``
    when explicitly provided, and falls back to the loaded value when omitted.
    ``normalize_map`` / ``non_translatable_terms`` are the current batch's
    lexical glossary hits and are injected independently of the RAG switch.
    """
    units = translation_core.units_from_items(
        items,
        translation_core.MODE_TRANSLATION,
    )
    context_bundle = translation_core.build_context_bundle(
        glossary_hits=glossary_hits or [],
        history_hits=history_hits or [],
        story_hits=story_hits,
    )
    return translation_core.build_sync_translation_prompt(
        units,
        PRESERVE_TERMS,
        context_bundle,
        history_char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
        story_char_limit=SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
        include_translation_memory=SYNC_RAG_ENABLED,
        context_window=context_window,
        macro_setting=macro_setting if macro_setting is not None else SYNC_MACRO_SETTING,
        normalize_map=normalize_map,
        non_translatable_terms=non_translatable_terms,
    )


def _sync_plan_context_policy():
    """Return the shared context policy while honoring legacy Sync overrides."""
    analysis_limit = translation_core.CANONICAL_ANALYSIS_CHAR_LIMIT
    if SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF:
        analysis_limit = 4000 + 800 + 1200
    return translation_plan.ContextPolicy(
        local_context_before=SYNC_CONTEXT_BEFORE,
        local_context_after=SYNC_CONTEXT_AFTER,
        history_char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
        source_index_char_limit=(
            get_sync_source_index_char_budget() if SYNC_SOURCE_INDEX_ENABLED else 0
        ),
        story_char_limit=SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
        analysis_char_limit=analysis_limit,
        include_source_text=True,
        include_translation_memory=SYNC_RAG_ENABLED,
        story_block_suffix='\n\n',
    )


def _sync_plan_generation_config():
    """Record D6 Sync generation differences in request identity."""
    return {
        'temperature': translation_plan.CANONICAL_TEMPERATURE,
        'max_output_tokens': SYNC_MAX_OUTPUT_TOKENS,
        'timeout': SYNC_TIMEOUT_SECONDS,
        'response_mime_type': 'application/json',
    }


def _sync_plan_story_graph_identity():
    """Return a portable, non-sensitive identity for the configured graph."""
    configured = str(SYNC_STORY_MEMORY_GRAPH_FILE or '').strip()
    if not configured:
        return {'scope': 'unset', 'path': '', 'sha256': ''}
    absolute = os.path.realpath(os.path.abspath(configured))
    project_root = os.path.realpath(os.path.abspath(BASE_DIR or os.curdir))
    try:
        relative = os.path.relpath(absolute, project_root)
    except ValueError:
        relative = ''
    inside_project = bool(
        relative
        and relative != os.pardir
        and not relative.startswith(os.pardir + os.sep)
        and not os.path.isabs(relative)
    )
    path_identity = (
        relative.replace('\\', '/')
        if inside_project
        else os.path.basename(absolute)
    )
    try:
        content_digest = file_sha256(absolute) if os.path.isfile(absolute) else ''
    except OSError:
        content_digest = ''
    return {
        'scope': 'project' if inside_project else 'external',
        'path': path_identity,
        'sha256': content_digest,
    }


def _sync_plan_config_snapshot():
    """Return the non-sensitive existing Sync settings that define a plan."""
    return {
        'target_language': PREP_LANGUAGE,
        'sync': {
            'backend': SYNC_BACKEND,
            'max_items': MAX_ITEMS,
            'max_source_chars': MAX_CHARS,
            'context_before': SYNC_CONTEXT_BEFORE,
            'context_after': SYNC_CONTEXT_AFTER,
            'rag_enabled': SYNC_RAG_ENABLED,
            'embedding': current_sync_embedding_settings().public_dict(),
            'source_index': {
                'enabled': SYNC_SOURCE_INDEX_ENABLED,
                'top_k': SYNC_SOURCE_INDEX_TOP_K,
                'min_similarity': SYNC_SOURCE_INDEX_MIN_SIMILARITY,
                'char_limit': SYNC_SOURCE_INDEX_CHAR_LIMIT,
            },
            'project_analysis': {
                'inject_published_brief': SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF,
            },
            'story_memory': {
                'enabled': SYNC_STORY_MEMORY_ENABLED,
                'graph_file': _sync_plan_story_graph_identity(),
                'max_context_chars': SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
                'top_k_relations': SYNC_STORY_MEMORY_TOP_K_RELATIONS,
                'top_k_terms': SYNC_STORY_MEMORY_TOP_K_TERMS,
                'include_scene_summary': (
                    SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY
                ),
            },
            'macro_fingerprint': SYNC_MACRO_FINGERPRINT,
        },
    }


def _sync_plan_source_identity(adapter_snapshot):
    project = adapter_snapshot.project
    documents = tuple(project.source_documents or ())
    return translation_plan.SourceIdentity(
        engine=str(project.engine or ''),
        adapter_version=str(project.adapter_version or ''),
        project_identity_digest=str(project.project_snapshot_fingerprint or ''),
        source_snapshot_fingerprint=(
            source_snapshot_fingerprint(documents) if documents else ''
        ),
        file_digests={
            str(document.file_rel_path or '').replace('\\', '/'): str(
                document.sha256 or ''
            )
            for document in documents
        },
    )


def current_sync_source_identity():
    """Rebuild the active project's read-only Sync source identity.

    Callers must load the current translator settings first.  This scan does
    not resolve a model route, require credentials, run prepare commands, or
    mutate project files, so offline check/apply can use it as a final
    writeback freshness predicate.
    """
    adapter = RenPyAdapter(legacy_module=sys.modules[__name__])
    snapshot = build_translation_snapshot(
        adapter,
        ProjectDiscoveryRequest(
            project_root=BASE_DIR,
            localization_root=TL_DIR,
            target_language=PREP_LANGUAGE,
            include_files=tuple(sorted(INCLUDE_FILES)),
            include_prefixes=tuple(sorted(INCLUDE_PREFIXES)),
        ),
    )
    return _sync_plan_source_identity(snapshot).to_dict()


def validate_sync_translation_plan_before_dispatch(plan_build):
    """Reject ordinary Sync source/adapter/plan drift before provider use."""
    if plan_build is None or plan_build.plan is None:
        raise RuntimeError('Sync TranslationPlan is missing before model dispatch.')
    plan_payload = plan_build.plan.to_dict()
    translation_plan.validate_plan_fingerprint(plan_payload)
    reasons = translation_plan.source_identity_differences(
        plan_payload.get('source_identity') or {},
        current_sync_source_identity(),
    )
    if reasons:
        raise RuntimeError(
            'Sync TranslationPlan is stale before model dispatch: '
            + ', '.join(reasons)
            + '. Regenerate the preview from the current project.'
        )
    summaries = list(plan_payload.get('request_summaries') or [])
    requests = list(plan_build.requests or [])
    if len(summaries) != len(requests):
        raise RuntimeError(
            'Sync TranslationPlan request summaries no longer match root requests.'
        )
    for index, (summary, request) in enumerate(zip(summaries, requests)):
        semantic_fingerprint, request_fingerprint = (
            translation_plan.recompute_request_fingerprints(request)
        )
        if (
            str(summary.get('request_id') or '') != request.request_id
            or str(summary.get('prompt_fingerprint') or '')
            != semantic_fingerprint
            or request.prompt_fingerprint != semantic_fingerprint
            or str(summary.get('request_fingerprint') or '')
            != request_fingerprint
            or request.request_fingerprint != request_fingerprint
        ):
            raise RuntimeError(
                'Sync TranslationPlan request binding is stale before model '
                f'dispatch: index={index}.'
            )
    return {
        'source': 'fresh',
        'adapter': 'fresh',
        'plan': 'fresh',
        'request_count': len(requests),
    }


def _render_sync_retrieval_reference_text(history_hits, story_hits, source_hits=None):
    """Render the shared retrieval layer without the lexical glossary block."""
    return advanced_context.render_retrieval_reference_text(
        history_hits,
        story_hits,
        source_hits,
        history_char_limit=SYNC_RAG_HISTORY_CHAR_LIMIT,
        story_char_limit=SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
        include_source_text=True,
    )


def build_sync_translation_plan(file_jobs, adapter_snapshot, routing_plan, *, run_id=''):
    """Build the ordinary Sync initial-translation plan and retrieval captures."""
    route = routing_plan.routes[model_profile.STAGE_TRANSLATION]
    profile = model_profile.profile_for_route(routing_plan, route)
    captures = []
    context_policy = _sync_plan_context_policy()
    preserve_terms = list(PRESERVE_TERMS)
    normalize_map = dict(NORMALIZE_TRANSLATION_MAP)
    non_translatable_exact = set(NON_TRANSLATABLE_EXACT)
    macro_setting = str(SYNC_MACRO_SETTING or '')
    retrieval_enabled = (
        SYNC_RAG_ENABLED
        or SYNC_STORY_MEMORY_ENABLED
        or SYNC_SOURCE_INDEX_ENABLED
        or SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
    )
    capabilities = routing_plan.capabilities.get(route.profile_id)
    budget = (
        capabilities.context_budget_tokens
        if capabilities is not None
        else None
    )
    if (
        file_jobs
        and budget is None
        and (MAX_ITEMS >= DEFAULT_MAX_ITEMS or MAX_CHARS >= DEFAULT_MAX_CHARS)
    ):
        print(
            'Warning: the selected model profile does not declare '
            'context_budget_tokens; Sync cannot preflight the current '
            f'{MAX_ITEMS}/{MAX_CHARS} chunk against provider context/output '
            'limits. If requests are rejected or truncated, lower '
            'translator_config.json sync.chunk_size and/or '
            'sync.max_source_chars.',
            flush=True,
        )

    def build_plan(retrieval_blocks_provider, analysis_blocks_provider=None):
        return translation_plan.build_translation_plan(
            file_jobs,
            execution_strategy=model_profile.ExecutionStrategy.SYNC.value,
            source_identity=_sync_plan_source_identity(adapter_snapshot),
            config_snapshot=_sync_plan_config_snapshot(),
            model_profile_snapshot=profile,
            run_id=run_id,
            chunk_policy=translation_plan.ChunkPolicy(
                max_items=MAX_ITEMS,
                max_chars=MAX_CHARS,
            ),
            context_policy=context_policy,
            preserve_terms=preserve_terms,
            normalize_map=normalize_map,
            non_translatable_exact=non_translatable_exact,
            macro_setting=macro_setting,
            retrieval_blocks_provider=retrieval_blocks_provider,
            analysis_blocks_provider=analysis_blocks_provider,
            generation_config=_sync_plan_generation_config(),
        )

    def validate_budget(plan_build, *, phase):
        if budget is None:
            return
        oversized = [
            request
            for request in plan_build.requests
            if int(request.capability_requirements.get('context_budget_tokens') or 0)
            > int(budget)
        ]
        if not oversized:
            return
        estimated = oversized[0].capability_requirements.get(
            'context_budget_tokens'
        )
        raise ValueError(
            'Sync TranslationPlan exceeds the selected model context budget: '
            f'request={oversized[0].request_id}, '
            f'estimated={estimated}, budget={budget}, '
            f'chunk_size={MAX_ITEMS}, max_source_chars={MAX_CHARS}, '
            f'phase={phase}. '
            'Lower translator_config.json sync.chunk_size and/or '
            'sync.max_source_chars, or select a model profile with a larger '
            'context budget.'
        )

    if file_jobs and retrieval_enabled and budget is not None:
        # This first pass uses the same fixed chunks, prompt, local context,
        # and char-upper-bound estimator without invoking a remote/local
        # retrieval provider. It catches obviously impossible combinations
        # before query embedding cost; the materialized pass below remains
        # authoritative for retrieved context near the budget boundary.
        preflight_build = build_plan(None)
        validate_budget(preflight_build, phase='pre-retrieval')

    if file_jobs and retrieval_enabled:
        print(
            'Preparing Sync TranslationPlan context: retrieval for all fixed '
            'chunks runs before the first model request.',
            flush=True,
        )

    def retrieval_provider(chunk_input):
        history_hits, rag_stats = (
            retrieve_sync_history_hits(chunk_input.target_items)
            if SYNC_RAG_ENABLED
            else ([], {'enabled': False})
        )
        source_hits, source_index_stats = (
            retrieve_sync_source_hits(chunk_input.target_items)
            if SYNC_SOURCE_INDEX_ENABLED
            else ([], {'enabled': False})
        )
        story_hits = (
            retrieve_sync_story_hits(chunk_input.target_items)
            if SYNC_STORY_MEMORY_ENABLED
            else None
        )
        text = translation_plan.normalize_context_provider_text(
            _render_sync_retrieval_reference_text(
                history_hits,
                story_hits,
                source_hits,
            )
        )
        if rag_stats.get('hit_count'):
            print(
                f"  Sync RAG memory hits: {rag_stats['hit_count']}",
                flush=True,
            )
        if source_index_stats.get('hit_count'):
            print(
                f"  Sync source index hits: {source_index_stats['hit_count']}",
                flush=True,
            )
        if source_index_stats.get('reason') == 'rebuild_store':
            print(
                '  Sync source index skipped: store identity requires rebuild.',
                flush=True,
            )
        if rag_stats.get('reason') == 'rebuild_store':
            print(
                '  Sync RAG skipped: store identity requires rebuild.',
                flush=True,
            )
        project_context = (
            load_sync_injectable_project_context(
                chunk_input.file_rel_path,
                [unit.display_line_number for unit in chunk_input.target_units],
            )
            if SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
            else {
                'text': '',
                'injectable': False,
                'reason': 'injection_disabled',
                'labels': [],
                'routes': [],
                'local_diagnostics': '',
            }
        )
        analysis_text = translation_plan.normalize_context_provider_text(
            advanced_context.render_analysis_reference_text(project_context)
        )
        captures.append({
            'file_rel_path': chunk_input.file_rel_path,
            'expected_ids': [unit.id for unit in chunk_input.target_units],
            'target_items': [dict(item) for item in chunk_input.target_items],
            'target_units': list(chunk_input.target_units),
            'context_window': chunk_input.context_window,
            'local_context_diagnostics': dict(
                chunk_input.local_context_diagnostics or {}
            ),
            'retrieval_blocks_text': text,
            'analysis_blocks_text': analysis_text,
            'rag_stats': dict(rag_stats or {}),
            'source_index_stats': dict(source_index_stats or {}),
            'project_analysis': advanced_context.analysis_skip_diagnostics(project_context),
            'history_hit_count': len(history_hits),
            'source_hit_count': len(source_hits),
            'story_memory_applied': story_memory.has_story_hits(story_hits),
            'context_policy': context_policy,
            'preserve_terms': preserve_terms,
            'normalize_map': normalize_map,
            'non_translatable_exact': non_translatable_exact,
            'macro_setting': macro_setting,
        })
        return {
            'text': text,
            'diagnostics': {
                'source_index': dict(source_index_stats or {}),
                **(
                    {
                        'embedding_provider': advanced_context.embedding_provider_diagnostics(
                            current_sync_embedding_settings()
                        )
                    }
                    if (SYNC_SOURCE_INDEX_ENABLED or SYNC_RAG_ENABLED)
                    else {}
                ),
            },
        }

    def analysis_provider(chunk_input):
        for capture in captures:
            if (
                capture.get('file_rel_path') == chunk_input.file_rel_path
                and list(capture.get('expected_ids') or [])
                == [unit.id for unit in chunk_input.target_units]
            ):
                return {
                    'text': capture.get('analysis_blocks_text') or '',
                    'diagnostics': {
                        'published_project_analysis': dict(
                            capture.get('project_analysis') or {}
                        ),
                    },
                }
        project_context = load_sync_injectable_project_context(
            chunk_input.file_rel_path,
            [unit.display_line_number for unit in chunk_input.target_units],
        )
        return {
            'text': translation_plan.normalize_context_provider_text(
                advanced_context.render_analysis_reference_text(project_context)
            ),
            'diagnostics': {
                'published_project_analysis': advanced_context.analysis_skip_diagnostics(
                    project_context
                ),
            },
        }

    plan_build = build_plan(retrieval_provider, analysis_provider)
    if len(captures) != len(plan_build.requests):
        raise RuntimeError(
            'Sync TranslationPlan retrieval capture count does not match its '
            f'requests: captures={len(captures)}, '
            f'requests={len(plan_build.requests)}.'
        )
    for index, (plan_chunk, request, capture) in enumerate(zip(
        plan_build.plan.chunks,
        plan_build.requests,
        captures,
    )):
        if (
            str(capture.get('file_rel_path') or '') != plan_chunk.file_rel_path
            or list(capture.get('expected_ids') or []) != request.expected_ids
        ):
            raise RuntimeError(
                'Sync TranslationPlan retrieval capture identity mismatch: '
                f'index={index}, request={request.request_id}.'
            )
    validate_budget(plan_build, phase='materialized-context')
    if retrieval_enabled:
        history_hit_count = sum(
            int((item.get('rag_stats') or {}).get('hit_count') or 0)
            for item in captures
        )
        source_hit_count = sum(
            int((item.get('source_index_stats') or {}).get('hit_count') or 0)
            for item in captures
        )
        analysis_chunk_count = sum(
            1
            for item in captures
            if (item.get('project_analysis') or {}).get('injectable')
        )
        story_chunk_count = sum(
            1 for item in captures if item.get('story_memory_applied')
        )
        print(
            'Sync TranslationPlan context frozen: '
            f'retrieval_chunks={len(captures)}, '
            f'history_hits={history_hit_count}, '
            f'source_hits={source_hit_count}, '
            f'story_chunks={story_chunk_count}, '
            f'analysis_chunks={analysis_chunk_count}.',
            flush=True,
        )
    return plan_build, captures


@dataclass
class SyncTranslationExecutionContext:
    """Current-project inputs needed by the durable Sync executor.

    The frozen :class:`translation_plan.PlanBuild` remains the semantic source
    of truth.  The additional maps only reconnect its stable item IDs to the
    live adapter and to #346's derived-request helper after a process restart.
    """

    plan_build: Any
    captures: list[dict[str, Any]]
    routing_plan: Any
    route: Any
    adapter: Any
    adapter_snapshot: Any
    pending_jobs: list[dict[str, Any]]
    items_by_id: dict[str, dict[str, Any]]
    occurrences_by_id: dict[str, Any]
    request_contexts: dict[str, dict[str, Any]]

    def item_resolver(self, _request, item_ids):
        resolved = []
        for item_id in item_ids:
            key = str(item_id)
            item = self.items_by_id.get(key)
            if item is None:
                raise ValueError(f'Sync TranslationPlan item is no longer available: {key}')
            resolved.append(dict(item))
        return resolved

    def _root_request_id(self, request):
        request_id = str((request or {}).get('request_id') or '')
        if request_id in self.request_contexts:
            return request_id
        candidates = [
            root_id
            for root_id in self.request_contexts
            if request_id.startswith(root_id + '--')
        ]
        if not candidates:
            raise ValueError(
                f'Sync derived request has no current root context: {request_id}'
            )
        return max(candidates, key=len)

    def context_resolver(self, request):
        capture = self.request_contexts[self._root_request_id(request)]
        return {
            'file_rel_path': capture.get('file_rel_path') or '',
            'file_path': capture.get('file_path') or '',
            'context_window': capture.get('context_window'),
            'local_context_diagnostics': dict(
                capture.get('local_context_diagnostics') or {}
            ),
            'context_policy': capture.get('context_policy'),
            'preserve_terms': list(capture.get('preserve_terms') or []),
            'normalize_map': dict(capture.get('normalize_map') or {}),
            'non_translatable_exact': list(
                capture.get('non_translatable_exact') or []
            ),
            'macro_setting': capture.get('macro_setting') or '',
            'retrieval_blocks_text': capture.get('retrieval_blocks_text') or '',
            'analysis_blocks_text': capture.get('analysis_blocks_text') or '',
        }

    def validate_translation(self, item, translated):
        occurrence = self.occurrences_by_id.get(str((item or {}).get('id') or ''))
        if occurrence is None:
            return False, 'common.locator.unresolved'
        validation = self.adapter.validate_translation(occurrence, str(translated or ''))
        if validation.status == 'pass':
            return True, 'OK'
        return False, ', '.join(validation.reason_codes) or 'adapter.validation.block'

    def validate_reused_translation(self, item_id, payload):
        translation = payload
        if isinstance(payload, dict):
            translation = payload.get('translation')
        item = self.items_by_id.get(str(item_id))
        if item is None or not isinstance(translation, str):
            return False
        valid, _reason = self.validate_translation(item, translation)
        return valid

    def durable_targets_payload(self, *, run_id=''):
        """Return the redacted, offline target shape consumed by check/preview."""
        chunks = []
        for index, request in enumerate(self.plan_build.requests):
            capture = self.captures[index]
            plan_chunk = self.plan_build.plan.chunks[index]
            chunks.append({
                'key': request.chunk_id,
                'request_id': request.request_id,
                'plan_id': request.plan_id,
                'chunk_id': request.chunk_id,
                'file_rel_path': plan_chunk.file_rel_path,
                'file_path': plan_chunk.file_path,
                'chunk_index': plan_chunk.chunk_index,
                'line_numbers': list(plan_chunk.line_numbers),
                'source_char_count': plan_chunk.source_char_count,
                'expected_ids': list(request.expected_ids),
                'prompt_fingerprint': request.prompt_fingerprint,
                'request_fingerprint': request.request_fingerprint,
                'items': [
                    translation_core.legacy_item_from_unit(
                        unit, translation_core.MODE_TRANSLATION
                    )
                    for unit in capture.get('target_units') or []
                ],
            })
        return {
            'schema_version': 1,
            'run_id': str(run_id or ''),
            'plan_id': self.plan_build.plan.plan_id,
            'plan_fingerprint': self.plan_build.plan.plan_fingerprint,
            'project_root': os.path.realpath(os.path.abspath(BASE_DIR)),
            'tl_dir': os.path.realpath(os.path.abspath(TL_DIR)),
            'target_language': PREP_LANGUAGE,
            'glossary_file': str(GLOSSARY_FILE or ''),
            'files': {
                str(job['file_rel_path']): {
                    'path': str(job['file_path']),
                    'task_count': len(job.get('tasks') or []),
                }
                for job in self.pending_jobs
            },
            'chunks': chunks,
        }


def prepare_sync_translation_execution_context(
    *,
    prepare=False,
    require_provider=True,
    persist_corrected_game_root=True,
    run_id='',
):
    """Build the ordinary Sync plan plus restart-safe production adapters.

    This is intentionally the same discovery, filtering, glossary, routing,
    retrieval and TranslationPlan path used by :func:`run_translation`.
    Provider calls are not made here.
    """
    try:
        load_config(require_api_key=False)
    except model_profile.ModelRoutingConfigError as exc:
        raise model_profile.routing_resolution_error(
            exc,
            stage=model_profile.STAGE_TRANSLATION,
        ) from exc
    load_translator_settings(
        persist_corrected_game_root=bool(persist_corrected_game_root)
    )
    if require_provider and SYNC_BACKEND == 'gemini':
        _require_gemini_api_key()
    load_glossary()
    routing_plan = freeze_translation_routing_plan()
    route = routing_plan.routes[model_profile.STAGE_TRANSLATION]
    if prepare:
        run_prepare_steps()
    elif PREP_ENABLED:
        print(
            'Prepare step skipped in durable Sync mode; use --prepare explicitly if needed.'
        )

    adapter = RenPyAdapter(legacy_module=sys.modules[__name__])
    adapter_snapshot = build_translation_snapshot(
        adapter,
        ProjectDiscoveryRequest(
            project_root=BASE_DIR,
            localization_root=TL_DIR,
            target_language=PREP_LANGUAGE,
            include_files=tuple(sorted(INCLUDE_FILES)),
            include_prefixes=tuple(sorted(INCLUDE_PREFIXES)),
        ),
    )
    occurrences_by_id = {
        occurrence.unit.id: occurrence
        for occurrence in adapter_snapshot.occurrences
        if occurrence.unit.id
    }
    global_progress = _upgrade_legacy_progress_keys(
        load_progress(),
        [document.file_path for document in adapter_snapshot.project.source_documents],
    )
    pending_jobs = []
    for document in adapter_snapshot.project.source_documents:
        progress_key = document.file_rel_path
        completed_entries = set(
            _normalize_progress_entries(global_progress.get(progress_key, []))
        )
        tasks = []
        for raw_task in adapter_snapshot.pending_tasks_by_file.get(progress_key, ()):
            task = dict(raw_task)
            if is_non_translatable(task['text']):
                continue
            progress_entry = task.get('progress_entry') or _progress_entry_for_task(task)
            if (
                progress_entry in completed_entries
                or _progress_line_entry(task['line']) in completed_entries
            ):
                if not FORCE_RETRANSLATE_ENGLISH or not is_english_like(task['text']):
                    continue
            task['progress_entry'] = _progress_entry_for_task(task)
            task['file_rel_path'] = progress_key
            task['file_path'] = document.file_path
            tasks.append(task)
        if tasks:
            pending_jobs.append({
                'file_rel_path': progress_key,
                'file_path': document.file_path,
                'tasks': tasks,
            })

    plan_build, captures = build_sync_translation_plan(
        pending_jobs,
        adapter_snapshot,
        routing_plan,
        run_id=run_id,
    )
    items_by_id = {}
    request_contexts = {}
    for index, request in enumerate(plan_build.requests):
        capture = captures[index]
        plan_chunk = plan_build.plan.chunks[index]
        capture['file_path'] = plan_chunk.file_path
        request_contexts[request.request_id] = capture
        for item in capture.get('target_items') or []:
            item_id = str(item.get('id') or '')
            if not item_id or item_id in items_by_id:
                raise RuntimeError(
                    f'Sync TranslationPlan has a missing or duplicate item ID: {item_id!r}'
                )
            items_by_id[item_id] = dict(item)
    return SyncTranslationExecutionContext(
        plan_build=plan_build,
        captures=captures,
        routing_plan=routing_plan,
        route=route,
        adapter=adapter,
        adapter_snapshot=adapter_snapshot,
        pending_jobs=pending_jobs,
        items_by_id=items_by_id,
        occurrences_by_id=occurrences_by_id,
        request_contexts=request_contexts,
    )

def get_nested(source, *candidates):
    for candidate in candidates:
        if source is None:
            continue
        if isinstance(source, dict) and candidate in source:
            return source.get(candidate)
        if hasattr(source, candidate):
            return getattr(source, candidate)
    return None


def serialize_unknown(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_unknown(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_unknown(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return serialize_unknown(method())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return serialize_unknown(vars(value))
    return str(value)


def extract_text_from_response_payload(response_payload):
    payload = response_payload
    if not isinstance(payload, dict):
        return ""

    nested_response = payload.get("response")
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(part["text"])
            if texts:
                return "".join(texts)

    text = payload.get("text")
    return text if isinstance(text, str) else ""


def extract_finish_reason(response_payload):
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get("response")
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("finishReason"):
                return str(candidate["finishReason"])
    return ""


def extract_prompt_feedback(response_payload):
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get("response")
    if isinstance(nested_response, dict):
        payload = nested_response

    prompt_feedback = payload.get("promptFeedback")
    return prompt_feedback if isinstance(prompt_feedback, dict) else {}


def build_response_json_schema(items):
    return translation_core.build_response_json_schema(
        items,
        mode=translation_core.MODE_TRANSLATION,
    )


def parse_json_payload(text):
    return translation_core.parse_model_response_json(text)


def normalize_result_items(payload):
    return translation_core.normalize_model_results(
        payload,
        mode=translation_core.MODE_TRANSLATION,
    )


def _routing_custom_providers(custom_providers):
    """Keep only real provider records for the resolver; mocks stay on the backend."""
    if not custom_providers:
        return None
    return {
        key: value
        for key, value in custom_providers.items()
        if getattr(value, "requires_key", None) is not None
    } or None


def freeze_translation_routing_plan(*, stage_overrides=None):
    """Snapshot and validate the translation route before task side effects."""
    current = get_current_model()
    overrides = dict(stage_overrides or {})
    overrides.setdefault(model_profile.STAGE_TRANSLATION, current)
    custom_providers = _routing_custom_providers(CUSTOM_LITELLM_PROVIDERS)
    try:
        plan = model_profile.resolve_routing_plan_from_runtime(
            sync_backend=SYNC_BACKEND,
            sync_model=current,
            sync_models=tuple(MODELS),
            custom_providers=custom_providers,
            execution=model_profile.ExecutionStrategy.SYNC,
            stage_overrides=overrides,
        )
    except (ValueError, TypeError) as exc:
        raise model_profile.routing_resolution_error(
            exc,
            stage=model_profile.STAGE_TRANSLATION,
        ) from exc
    try:
        from litellm_provider_config import load_provider_api_key

        def _has_key(provider_id):
            try:
                return bool(load_provider_api_key(provider_id))
            except Exception:
                return True
    except ImportError:
        _has_key = None
    issues = model_profile.validate_routing_plan(
        plan,
        stages={model_profile.STAGE_TRANSLATION},
        custom_providers=custom_providers,
        keyring_has_credential=_has_key,
    )
    if issues:
        raise model_profile.routing_validation_error(issues)
    return plan


def call_gemini_sdk(
    prompt,
    items,
    usage_run_id='',
    usage_buffer=None,
    usage_operation_id='',
    return_contract=False,
    route=None,
    plan=None,
    translation_request=None,
):
    """Calls the explicitly configured synchronous backend."""
    if plan is None or route is None:
        plan = freeze_translation_routing_plan()
        route = plan.routes[model_profile.STAGE_TRANSLATION]
    if not isinstance(route, model_profile.TaskRoute):
        raise TypeError(
            'call_gemini_sdk requires an explicit TaskRoute; '
            f'got {type(route).__name__}.'
        )
    profile = model_profile.profile_for_route(plan, route)
    model_name = profile.model
    if translation_request is not None and not isinstance(
        translation_request,
        translation_plan.TranslationRequest,
    ):
        raise TypeError('translation_request must be a TranslationRequest')
    generation_config = dict(
        translation_request.generation_config
        if translation_request is not None
        else _sync_plan_generation_config()
    )
    generation_config.update({
        "response_mime_type": "application/json",
        "response_json_schema": (
            translation_request.response_schema
            if translation_request is not None
            else build_response_json_schema(items)
        ),
    })
    if translation_request is not None:
        generation_config['system_instruction'] = (
            translation_request.system_instruction
        )
    generation_config = filter_gemini_generation_config(model_name, generation_config)

    backend = model_profile.build_sync_backend(
        profile,
        custom_providers=CUSTOM_LITELLM_PROVIDERS,
    )
    request = SyncGenerationRequest(
            model=model_name,
            contents=(
                translation_request.user_prompt
                if translation_request is not None
                else prompt
            ),
            config=generation_config,
        )
    result = backend.generate(request)
    output_diagnostics = model_usage_ledger.response_budget_diagnostics(
        response_text=getattr(result, 'response_text', ''),
        finish_reason=getattr(result, 'finish_reason', ''),
        usage_metadata=getattr(result, 'usage_metadata', None) or {},
        max_output_tokens=generation_config.get("max_output_tokens"),
    )

    if usage_run_id and BASE_DIR:
        item_ids = [str(item.get('id') or '') for item in items]
        try:
            record = model_usage_ledger.build_usage_record(
                game_root=BASE_DIR,
                task_mode='translation',
                stage='sync_translation',
                provider=result.provider,
                model=result.model,
                usage_metadata=getattr(result, 'usage_metadata', None) or {},
                response_payload=result.response_payload,
                operation_id=usage_operation_id or usage_run_id,
                run_id=usage_run_id,
                execution_mode=result.execution_mode,
                source_key='|'.join(item_ids),
                source={
                    'kind': 'sync_translation_response',
                    'item_ids': item_ids,
                    'request_id': (
                        translation_request.request_id
                        if translation_request is not None
                        else ''
                    ),
                },
                response_diagnostics=output_diagnostics,
                request_metadata=dict(
                    getattr(result, 'request_metadata', None) or {}
                ) | ({
                    'request_id': translation_request.request_id,
                    'plan_id': translation_request.plan_id,
                    'chunk_id': translation_request.chunk_id,
                    'prompt_fingerprint': translation_request.prompt_fingerprint,
                    'request_fingerprint': translation_request.request_fingerprint,
                } if translation_request is not None else {}),
            )
            if usage_buffer is not None:
                usage_buffer.append(record)
            else:
                model_usage_ledger.UsageLedger(BASE_DIR).add_records([record])
        except (OSError, ValueError, model_usage_ledger.UsageLedgerError) as exc:
            print(
                f'Warning: Model usage ledger record failed: {exc}',
                flush=True,
            )

    payload = None
    if result.parsed is not None:
        payload = serialize_unknown(result.parsed)
    elif result.response_text:
        payload = parse_json_payload(result.response_text)

    if payload is not None:
        report = translation_core.validate_model_response(
            payload,
            mode=translation_core.MODE_TRANSLATION,
            expected_units=items,
            allow_legacy=True,
        )
        if return_contract:
            return report
        if not report.items and report.expected_ids:
            reason = report.issues[0].reason_code if report.issues else 'response_contract_error'
            raise translation_core.ModelResponseContractError(
                reason,
                'Model response did not contain any valid requested translations.',
            )
        return report.items

    prompt_feedback = extract_prompt_feedback(result.response_payload)
    diagnostics = []
    if prompt_feedback:
        diagnostics.append(f"Prompt feedback: {prompt_feedback}")
    if result.finish_reason:
        diagnostics.append(f"Finish reason: {result.finish_reason}")
    detail = f" ({'; '.join(diagnostics)})" if diagnostics else ""
    reason_code = str(
        output_diagnostics.get('reason_code')
        or translation_core.CONTRACT_EMPTY_RESPONSE_TEXT
    )
    raise translation_core.ModelResponseContractError(
        reason_code,
        f"Invalid response from API. Missing structured text{detail}.",
    )

def process_batch(
    batch,
    replacements,
    usage_run_id='',
    usage_buffer=None,
    usage_operation_id='',
    translation_validator=None,
    context_window=None,
    return_contract=False,
    route=None,
    plan=None,
    translation_request=None,
):
    # Local glossary matches are lexical and must not depend on the RAG
    # switch: normalize_map / preserve / non-translatable hits always reach
    # the prompt for the current TARGET batch (issue #338).
    if translation_request is None:
        glossary_hits = retrieve_sync_glossary_hits(batch)
        normalize_map = {
            str(hit.get('source') or ''): str(hit.get('target') or '')
            for hit in glossary_hits
            if hit.get('kind') == 'normalize' and hit.get('source')
        }
        non_translatable_terms = [
            str(hit.get('source') or '')
            for hit in glossary_hits
            if hit.get('kind') == 'non_translatable' and hit.get('source')
        ]
        locked_glossary_hits = (
            glossary_hits[: max(1, SYNC_RAG_TOP_K_TERMS)] if SYNC_RAG_ENABLED else []
        )
        history_hits, rag_stats = (
            retrieve_sync_history_hits(batch) if SYNC_RAG_ENABLED else ([], {})
        )
        story_hits = (
            retrieve_sync_story_hits(batch) if SYNC_STORY_MEMORY_ENABLED else None
        )
        if rag_stats.get("hit_count"):
            print(f"  Sync RAG memory hits: {rag_stats['hit_count']}", flush=True)
        prompt = build_prompt(
            batch,
            glossary_hits=locked_glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
            context_window=context_window,
            normalize_map=normalize_map,
            non_translatable_terms=non_translatable_terms,
        )
    else:
        prompt = translation_request.user_prompt

    # Call API (SDK handles connection details)
    contract = call_gemini_sdk(
        prompt,
        batch,
        usage_run_id=usage_run_id,
        usage_buffer=usage_buffer,
        usage_operation_id=usage_operation_id,
        return_contract=True,
        route=route,
        plan=plan,
        translation_request=translation_request,
    )
    if isinstance(contract, list):
        # Compatibility for injected/test callables that still return the
        # historical normalized list instead of a contract report.
        contract = translation_core.validate_model_response(
            {'translations': contract},
            mode=translation_core.MODE_TRANSLATION,
            expected_units=batch,
            allow_legacy=True,
        )
    results = contract.items

    id_map = {_sync_contract_item_id(item.get("id")): item for item in batch}
    valid_progress_entries = []
    accepted_ids = []
    seen_result_ids = set()
    retry_ids = {
        normalized_id
        for item_id in contract.retry_ids
        if (normalized_id := _sync_contract_item_id(item_id))
    }

    for item in results:
        result_id = _sync_contract_item_id(item.get("id"))
        entry = id_map.get(result_id)
        if not entry:
            continue
        entry_id = _sync_contract_item_id(entry.get("id"))
        if entry_id in seen_result_ids:
            print(f"  Warning: Duplicate result id ignored: {entry_id}")
            continue
        seen_result_ids.add(entry_id)

        translated = item.get("translation", "")
        memory_translation = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
        if translation_validator is None:
            valid, msg = validate_translation(entry["text"], translated)
        else:
            valid, msg = translation_validator(entry, translated)

        if not valid:
            print(f"  Warning: Validation failed for {entry_id}: {msg}")
            retry_ids.add(entry_id)
            continue

        valid_progress_entries.append(entry["progress_entry"])
        accepted_ids.append(entry_id)
        entry["translated_text"] = memory_translation
        unit = translation_core.unit_from_sync_task(entry)
        action = translation_core.translation_writeback_action(unit, item)
        replacements.setdefault(action.line, []).append(
            translation_core.writeback_tuple(action, include_expected=False)
        )

    if not valid_progress_entries and not return_contract:
        raise RuntimeError("No valid translations in batch (all items rejected; consider expanding non-translatable rules or switching model)")

    # Calculate total chars to show valid data receipt without spoilers
    total_chars = sum(len(item.get("translation", "")) for item in results)
    print(f"  Translated {len(valid_progress_entries)}/{len(batch)} items. (Received {total_chars} chars of translation)", flush=True)
    if return_contract:
        return {
            'progress_entries': valid_progress_entries,
            'accepted_ids': accepted_ids,
            'retry_ids': [
                _sync_contract_item_id(item.get('id'))
                for item in batch
                if _sync_contract_item_id(item.get('id')) in retry_ids
            ],
            'contract': contract,
        }
    return valid_progress_entries


def new_sync_contract_diagnostics():
    """Return mutable per-preview response-contract counters."""
    return {
        'first_pass_expected': 0,
        'first_pass_valid': 0,
        'targeted_retry_requests': 0,
        'targeted_retry_items': 0,
        'split_retry_requests': 0,
        'reason_counts': {},
        'diagnostic_counts': {},
        'terminal_reason_counts': {},
        'retry_lineage': [],
        '_expected_ids': set(),
        '_valid_ids': set(),
    }


def _sync_contract_item_id(value):
    """Normalize runtime IDs to the string form used by response contracts."""
    return '' if value is None else str(value)


def _record_sync_contract_report(
    diagnostics,
    report,
    *,
    retry_kind,
    accepted_ids=None,
):
    if diagnostics is None:
        return
    accepted_ids = {
        normalized_id
        for item_id in (accepted_ids or ())
        if (normalized_id := _sync_contract_item_id(item_id))
    }
    if retry_kind == 'first_pass':
        diagnostics['first_pass_valid'] += len(accepted_ids)
    elif retry_kind == 'split_retry':
        diagnostics['split_retry_requests'] += 1
    # Contract-valid model output is not yet safe to preview. Only count IDs
    # accepted by the local/adapter validator as final valid results.
    diagnostics['_valid_ids'].update(accepted_ids)
    for reason_code, count in report.reason_counts().items():
        diagnostics['reason_counts'][reason_code] = (
            diagnostics['reason_counts'].get(reason_code, 0) + count
        )
    for reason_code, count in report.diagnostic_counts().items():
        diagnostics['diagnostic_counts'][reason_code] = (
            diagnostics['diagnostic_counts'].get(reason_code, 0) + count
        )


def _record_terminal_contract_report(diagnostics, report):
    if diagnostics is None:
        return
    reason_counts = report.reason_counts()
    if not reason_counts and not report.complete:
        reason_counts = {'response_contract_failure': 1}
    terminal_counts = diagnostics.setdefault('terminal_reason_counts', {})
    for reason_code, count in reason_counts.items():
        terminal_counts[reason_code] = terminal_counts.get(reason_code, 0) + count


def _record_unresolved_contract_items(failures, batch, reason_code, message):
    if failures is None:
        return
    for item in batch:
        failures.append({
            'relative_path': str(item.get('file_rel_path') or ''),
            'item_id': _sync_contract_item_id(item.get('id')),
            'reason_code': str(reason_code or 'response_contract_failure'),
            'message': str(message or 'Model response contract failed.'),
        })


def finalize_sync_contract_diagnostics(diagnostics):
    if diagnostics is None:
        return {}
    expected_ids = {
        normalized_id
        for item_id in (diagnostics.get('_expected_ids') or ())
        if (normalized_id := _sync_contract_item_id(item_id))
    }
    valid_ids = {
        normalized_id
        for item_id in (diagnostics.get('_valid_ids') or ())
        if (normalized_id := _sync_contract_item_id(item_id))
    }
    unresolved_ids = sorted(expected_ids - valid_ids)
    first_expected = int(diagnostics.get('first_pass_expected') or 0)
    first_valid = int(diagnostics.get('first_pass_valid') or 0)
    final_valid = len(expected_ids & valid_ids)
    return {
        'first_pass_expected': first_expected,
        'first_pass_valid': first_valid,
        'first_pass_completeness': (
            first_valid / first_expected if first_expected else 1.0
        ),
        'targeted_retry_requests': int(
            diagnostics.get('targeted_retry_requests') or 0
        ),
        'targeted_retry_items': int(diagnostics.get('targeted_retry_items') or 0),
        'split_retry_requests': int(diagnostics.get('split_retry_requests') or 0),
        'final_expected': len(expected_ids),
        'final_valid': final_valid,
        'final_completeness': final_valid / len(expected_ids) if expected_ids else 1.0,
        'unresolved_ids': unresolved_ids,
        'reason_counts': dict(diagnostics.get('reason_counts') or {}),
        'diagnostic_counts': dict(diagnostics.get('diagnostic_counts') or {}),
        'terminal_reason_counts': dict(
            diagnostics.get('terminal_reason_counts') or {}
        ),
        'retry_lineage': list(diagnostics.get('retry_lineage') or []),
    }


def print_sync_usage_summary(records):
    """Print the stable usage facts consumed by synchronous GUI reports."""
    totals = model_usage_ledger.aggregate_usage_records(list(records or ()))

    def token_value(field):
        if int(totals.get(f'{field}_known_records') or 0) <= 0:
            return 'unknown'
        return str(int(totals.get(field) or 0))

    diagnostics = totals.get('output_diagnostics')
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    for line in model_usage_ledger.format_sync_output_lines(
        completion=token_value('completion_tokens'),
        reasoning=token_value('reasoning_tokens'),
        text_output=token_value('text_output_tokens'),
        reasoning_budget_pressure=int(
            diagnostics.get('reasoning_budget_pressure_records') or 0
        ),
        truncated=int(diagnostics.get('truncated_records') or 0),
    ):
        print(line)


def derive_sync_retry_request(parent_request, batch, request_context, suffix, kind):
    """Build one D7 child request while leaving the initial plan immutable."""
    context = dict(request_context or {})
    return translation_plan.derive_translation_request(
        parent_request,
        batch,
        lineage_suffix=suffix,
        file_rel_path=str((batch[0] if batch else {}).get('file_rel_path') or ''),
        file_path=str((batch[0] if batch else {}).get('file_path') or ''),
        context_window=context.get('context_window'),
        local_context_diagnostics=context.get('local_context_diagnostics'),
        context_policy=context.get('context_policy') or _sync_plan_context_policy(),
        preserve_terms=context.get('preserve_terms', PRESERVE_TERMS),
        normalize_map=context.get('normalize_map', NORMALIZE_TRANSLATION_MAP),
        non_translatable_exact=context.get(
            'non_translatable_exact', NON_TRANSLATABLE_EXACT
        ),
        macro_setting=context.get('macro_setting', SYNC_MACRO_SETTING),
        retrieval_blocks_text=context.get('retrieval_blocks_text', ''),
        analysis_blocks_text=context.get('analysis_blocks_text', ''),
        retrieval_diagnostics=context.get('retrieval_diagnostics'),
        analysis_diagnostics=context.get('analysis_diagnostics'),
        lineage_kind=kind,
    )


def process_batch_with_retry(
    batch,
    replacements,
    retry_depth=0,
    usage_run_id='',
    usage_buffer=None,
    usage_operation_id='',
    translation_validator=None,
    contract_diagnostics=None,
    contract_failures=None,
    retry_kind='first_pass',
    context_window=None,
    route=None,
    plan=None,
    translation_request=None,
    request_context=None,
):
    if contract_diagnostics is not None and retry_kind == 'first_pass':
        original_ids = [
            normalized_id
            for item in batch
            if (normalized_id := _sync_contract_item_id(item.get('id')))
        ]
        contract_diagnostics['first_pass_expected'] += len(original_ids)
        contract_diagnostics['_expected_ids'].update(original_ids)
    if retry_depth >= 5:
        log_failure(batch, "Max retry depth reached")
        _record_unresolved_contract_items(
            contract_failures,
            batch,
            'max_retry_depth',
            'Model response could not be completed within the retry depth limit.',
        )
        return []

    error_str = "" # Initialize variable to be safe
    error_reason_code = ''
    split_reason_counts = {}
    error_detail = ''
    allow_split = True

    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            # Respect rate limits
            time.sleep(get_random_delay())

            outcome = process_batch(
                batch,
                replacements,
                translation_validator=translation_validator,
                usage_run_id=usage_run_id,
                usage_buffer=usage_buffer,
                usage_operation_id=usage_operation_id,
                context_window=context_window,
                return_contract=True,
                route=route,
                plan=plan,
                translation_request=translation_request,
            )
            _record_sync_contract_report(
                contract_diagnostics,
                outcome['contract'],
                retry_kind=retry_kind,
                accepted_ids=outcome['accepted_ids'],
            )
            successful = list(outcome['progress_entries'])
            retry_ids = {
                normalized_id
                for item_id in outcome['retry_ids']
                if (normalized_id := _sync_contract_item_id(item_id))
            }
            if not retry_ids:
                if not outcome['contract'].complete:
                    _record_terminal_contract_report(
                        contract_diagnostics,
                        outcome['contract'],
                    )
                return successful

            targeted_batch = [
                item
                for item in batch
                if _sync_contract_item_id(item.get('id')) in retry_ids
            ]
            if not targeted_batch:
                _record_terminal_contract_report(
                    contract_diagnostics,
                    outcome['contract'],
                )
                return successful
            if len(targeted_batch) == len(batch) and not successful:
                error_str = 'Model response made no progress for this batch.'
                error_reason_code = (
                    outcome['contract'].issues[0].reason_code
                    if outcome['contract'].issues
                    else 'validation_failed'
                )
                split_reason_counts = outcome['contract'].reason_counts()
                print(
                    '  ! No valid items accepted for this batch.',
                    flush=True,
                )
                break
            if contract_diagnostics is not None:
                contract_diagnostics['targeted_retry_requests'] = (
                    contract_diagnostics.get('targeted_retry_requests', 0) + 1
                )
                contract_diagnostics['targeted_retry_items'] = (
                    contract_diagnostics.get('targeted_retry_items', 0)
                    + len(targeted_batch)
                )
                lineage_entry = {
                    'kind': 'targeted',
                    'depth': retry_depth + 1,
                    'item_ids': [
                        _sync_contract_item_id(item.get('id'))
                        for item in targeted_batch
                    ],
                    'reason_counts': outcome['contract'].reason_counts(),
                }
                if translation_request is not None:
                    lineage_entry['parent_request_id'] = (
                        translation_request.request_id
                    )
                contract_diagnostics.setdefault('retry_lineage', []).append(
                    lineage_entry
                )
            print(
                f"  > Targeted retry for {len(targeted_batch)}/{len(batch)} items...",
                flush=True,
            )
            child_request = (
                derive_sync_retry_request(
                    translation_request,
                    targeted_batch,
                    request_context,
                    '--T',
                    'targeted',
                )
                if translation_request is not None
                else None
            )
            if contract_diagnostics is not None and child_request is not None:
                contract_diagnostics['retry_lineage'][-1]['request_id'] = (
                    child_request.request_id
                )
            retried = process_batch_with_retry(
                targeted_batch,
                replacements,
                retry_depth + 1,
                usage_run_id=usage_run_id,
                usage_buffer=usage_buffer,
                usage_operation_id=usage_operation_id,
                translation_validator=translation_validator,
                contract_diagnostics=contract_diagnostics,
                contract_failures=contract_failures,
                retry_kind='targeted_retry',
                # Targeted retries stay within the original batch's local
                # context window; reuse it so missing items keep the same
                # surrounding context as the first request.
                context_window=context_window,
                route=route,
                plan=plan,
                translation_request=child_request,
                request_context=request_context,
            )
            return successful + retried

        except Exception as e:
            recovery = sync_recovery_decision(e)
            is_structured_provider_failure = bool(
                getattr(e, 'category', '')
                or getattr(e, 'status_code', None) is not None
                or recovery.category != 'provider_error'
            )
            error_str = (
                sync_error_summary(e)
                if is_structured_provider_failure
                else str(e)
            )
            error_reason_code = str(getattr(e, 'reason_code', '') or '')
            if not error_reason_code and recovery.category != 'provider_error':
                error_reason_code = recovery.category
            if recovery.category == 'provider_error':
                # Unknown provider failures keep the original message in the
                # local failure log; the printed/manifest text stays safe.
                error_detail = sync_error_detail(e)
            split_reason_code = error_reason_code
            if not split_reason_code and 'Finish reason: 2' in error_str:
                split_reason_code = 'truncated_output'
            split_reason_counts = {
                split_reason_code or 'request_or_contract_failure': 1
            }
            print(f"  [Attempt {attempt}] Error: {error_str[:100]}...", flush=True)

            # Handle provider failures by structured recovery category.
            if recovery.category == 'rate_limit':
                print("  ! Rate limit hit.")
                if SYNC_BACKEND == 'gemini':
                    key_rotated = rotate_api_key()
                    if attempt > 1 or not key_rotated:
                        print("  ! Persistent rate limit. Switching model...")
                        if rotate_model():
                            continue
                    if key_rotated:
                        continue
                if attempt < BATCH_RETRIES:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                allow_split = False
                break
            if recovery.category in {'service_unavailable', 'timeout'}:
                if attempt < BATCH_RETRIES:
                    time.sleep(min(2 ** attempt, 5))
                    continue
                allow_split = False
                break
            if recovery.split_request or error_reason_code == 'truncated_output':
                print("  ! Invalid or truncated output. Splitting batch...")
                break
            # Authentication, unsupported capability, and missing dependency
            # errors are deterministic; repeating or splitting cannot help.
            if recovery.category in {
                'authentication',
                'unsupported_capability',
                'missing_dependency',
            }:
                allow_split = False
                break
            # Unknown provider errors may be caused by a single problematic
            # row (400 content policy, gateway hiccup). Splitting isolates the
            # failing row so healthy lines can still produce output; this
            # preserves the pre-#340 split rescue for unclassified failures.
            if recovery.category == 'provider_error':
                allow_split = True
                break

    # If batch failed after retries, try splitting
    if allow_split and len(batch) > 1:
        print("  > Splitting batch...", flush=True)
        mid = len(batch) // 2
        left_batch = batch[:mid]
        right_batch = batch[mid:]
        if contract_diagnostics is not None:
            left_request = (
                derive_sync_retry_request(
                    translation_request,
                    left_batch,
                    request_context,
                    '--L',
                    'split',
                )
                if translation_request is not None
                else None
            )
            right_request = (
                derive_sync_retry_request(
                    translation_request,
                    right_batch,
                    request_context,
                    '--R',
                    'split',
                )
                if translation_request is not None
                else None
            )
            lineage_entry = {
                'kind': 'split',
                'depth': retry_depth + 1,
                'item_ids': [
                    _sync_contract_item_id(item.get('id'))
                    for item in batch
                ],
                'child_item_ids': [
                    [
                        _sync_contract_item_id(item.get('id'))
                        for item in child_batch
                    ]
                    for child_batch in (left_batch, right_batch)
                ],
                'reason_counts': dict(
                    split_reason_counts
                    or {error_reason_code or 'request_or_contract_failure': 1}
                ),
            }
            if translation_request is not None:
                lineage_entry['parent_request_id'] = translation_request.request_id
                lineage_entry['child_request_ids'] = [
                    left_request.request_id,
                    right_request.request_id,
                ]
            contract_diagnostics.setdefault('retry_lineage', []).append(
                lineage_entry
            )
        else:
            left_request = (
                derive_sync_retry_request(
                    translation_request, left_batch, request_context, '--L', 'split'
                )
                if translation_request is not None else None
            )
            right_request = (
                derive_sync_retry_request(
                    translation_request, right_batch, request_context, '--R', 'split'
                )
                if translation_request is not None else None
            )
        r1 = process_batch_with_retry(
            left_batch, replacements, retry_depth + 1,
            usage_run_id=usage_run_id,
            usage_buffer=usage_buffer,
            usage_operation_id=usage_operation_id,
            translation_validator=translation_validator,
            contract_diagnostics=contract_diagnostics,
            contract_failures=contract_failures,
            retry_kind='split_retry',
            # D7 children retain the original plan chunk's surrounding context.
            route=route,
            plan=plan,
            translation_request=left_request,
            request_context=request_context,
        )
        r2 = process_batch_with_retry(
            right_batch, replacements, retry_depth + 1,
            usage_run_id=usage_run_id,
            usage_buffer=usage_buffer,
            usage_operation_id=usage_operation_id,
            translation_validator=translation_validator,
            contract_diagnostics=contract_diagnostics,
            contract_failures=contract_failures,
            retry_kind='split_retry',
            route=route,
            plan=plan,
            translation_request=right_request,
            request_context=request_context,
        )
        return r1 + r2

    log_message = f"Failed after retries: {error_str}"
    if error_detail and error_detail != error_str:
        log_message = f"{log_message} | original: {error_detail}"
    log_failure(batch, log_message)
    if contract_diagnostics is not None:
        terminal_code = error_reason_code or 'request_or_contract_failure'
        terminal_counts = contract_diagnostics.setdefault(
            'terminal_reason_counts', {}
        )
        terminal_counts[terminal_code] = terminal_counts.get(terminal_code, 0) + 1
    _record_unresolved_contract_items(
        contract_failures,
        batch,
        error_reason_code or 'request_or_contract_failure',
        error_str or 'Model request failed after retries.',
    )
    return []


def infer_dialogue_speaker_id(line, string_start_col):
    prefix = (line[:string_start_col] or "").strip()
    if not prefix:
        return ""
    prefix = prefix.rsplit(":", 1)[-1].strip()
    if not prefix or any(marker in prefix for marker in ("=", "(", ")", "[", "]", "{", "}")):
        return ""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(prefix).readline))
    except Exception:
        return ""
    for token in tokens:
        if token.type != tokenize.NAME:
            continue
        candidate = token.string.strip()
        if candidate and candidate.lower() not in RENPY_NON_SPEAKER_NAMES:
            return candidate
    return ""


def _character_display_value_node(expr):
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return expr
        return None
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "_"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        return _character_display_value_node(expr.args[0])
    return None


def _character_display_arg(call):
    if not isinstance(call, ast.Call):
        return None
    if not isinstance(call.func, ast.Name) or call.func.id != "Character":
        return None
    if call.args:
        return call.args[0]
    for keyword_arg in call.keywords:
        if keyword_arg.arg == "name":
            return keyword_arg.value
    return None


def normalize_character_display_name(text):
    text = " ".join(str(text).split()).strip()
    if not text:
        return ""
    if CHARACTER_DISPLAY_SYMBOLS_RE.match(text):
        return ""
    if CHARACTER_DISPLAY_ASSET_RE.match(text):
        return ""
    return text


def _literal_character_display_name(node):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return ""
    return normalize_character_display_name(node.value)


def _source_col_for_node(node, call_start_col):
    col = getattr(node, "col_offset", None)
    if col is None:
        return None
    if getattr(node, "lineno", 1) == 1:
        return col + call_start_col
    return col


def _source_end_col_for_node(node, call_start_col):
    col = getattr(node, "end_col_offset", None)
    if col is None:
        return None
    if getattr(node, "end_lineno", 1) == 1:
        return col + call_start_col
    return col


def _parse_character_definition(lines, start_idx, max_lines=80):
    match = CHARACTER_DEFINE_RE.match(lines[start_idx])
    if not match:
        return None

    call_start_col = match.start("call")
    pieces = []
    parsed_call = None
    end_limit = min(len(lines), start_idx + max_lines)
    for line_idx in range(start_idx, end_limit):
        if line_idx == start_idx:
            pieces.append(lines[line_idx][call_start_col:])
        else:
            pieces.append(lines[line_idx])
        try:
            parsed = ast.parse("".join(pieces), mode="eval")
        except SyntaxError:
            continue
        parsed_call = parsed.body
        break

    if parsed_call is None:
        return None

    display_arg = _character_display_arg(parsed_call)
    display_node = _character_display_value_node(display_arg)
    display_spans = []
    display_name = _literal_character_display_name(display_node)
    if display_node is not None:
        start_line = start_idx + getattr(display_node, "lineno", 1) - 1
        end_line = start_idx + getattr(display_node, "end_lineno", getattr(display_node, "lineno", 1)) - 1
        start_col = _source_col_for_node(display_node, call_start_col)
        end_col = _source_end_col_for_node(display_node, call_start_col)
        if start_col is not None and end_col is not None:
            display_spans.append((start_line, end_line, start_col, end_col))

    return {
        "speaker_id": match.group("speaker"),
        "speaker_name": display_name,
        "display_spans": display_spans,
    }


def _token_matches_span(line_idx, token, span):
    start_line, end_line, start_col, end_col = span
    token_start_line = line_idx
    token_end_line = line_idx + token.end[0] - token.start[0]
    token_start_col = token.start[1]
    token_end_col = token.end[1]

    if token_start_line < start_line or token_end_line > end_line:
        return False
    if token_start_line == start_line and token_start_col < start_col:
        return False
    if token_end_line == end_line and token_end_col > end_col:
        return False
    return True


def _is_character_display_token(line_idx, token, display_spans):
    return any(_token_matches_span(line_idx, token, span) for span in display_spans)


def find_source_text_for_translation_line(lines, idx):
    for prev_idx in range(idx - 1, -1, -1):
        prev_line = lines[prev_idx].strip()
        if not prev_line:
            continue

        comment_match = TL_COMMENT_SOURCE_RE.match(lines[prev_idx].rstrip("\n"))
        if comment_match:
            if is_voice_comment_match(comment_match):
                continue
            return decode_string_literal_text(comment_match.group("text"))

        old_match = TL_OLD_LINE_RE.match(lines[prev_idx].rstrip("\n"))
        if old_match:
            return decode_string_literal_text(old_match.group("text"))

        if is_voice_statement_line(prev_line):
            continue
        break
    return None


def _translate_block_name(line):
    match = re.match(r'^\s*translate\s+\S+\s+([^\s:]+)\s*:', line)
    return match.group(1) if match else None


def _previous_significant_token_index(tokens, start_index):
    for token_index in range(start_index - 1, -1, -1):
        token = tokens[token_index]
        if token.type in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            continue
        return token_index
    return None


def _is_keyword_argument_string_token(tokens, token_index):
    equal_index = _previous_significant_token_index(tokens, token_index)
    if equal_index is None or tokens[equal_index].string != "=":
        return False
    name_index = _previous_significant_token_index(tokens, equal_index)
    return name_index is not None and tokens[name_index].type == tokenize.NAME


def is_keyword_argument_string_span(line, start_col, end_col):
    try:
        start_col = int(start_col)
        end_col = int(end_col)
    except (TypeError, ValueError):
        return False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except Exception:
        return False
    for token_index, token in enumerate(tokens):
        if token.type != tokenize.STRING:
            continue
        if token.start[1] == start_col and token.end[1] == end_col:
            return _is_keyword_argument_string_token(tokens, token_index)
    return False


def _has_later_non_keyword_string(tokens, token_index):
    for later_index in range(token_index + 1, len(tokens)):
        later = tokens[later_index]
        if later.type != tokenize.STRING:
            continue
        if not _is_keyword_argument_string_token(tokens, later_index):
            return True
    return False


def _is_say_speaker_label_string_token(line, tokens, token_index):
    token = tokens[token_index]
    if token.type != tokenize.STRING:
        return False
    if _is_keyword_argument_string_token(tokens, token_index):
        return False
    if line[:token.start[1]].strip():
        return False
    return _has_later_non_keyword_string(tokens, token_index)


def is_say_speaker_label_string_span(line, start_col, end_col):
    try:
        start_col = int(start_col)
        end_col = int(end_col)
    except (TypeError, ValueError):
        return False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except Exception:
        return False
    for token_index, token in enumerate(tokens):
        if token.type != tokenize.STRING:
            continue
        if token.start[1] == start_col and token.end[1] == end_col:
            return _is_say_speaker_label_string_token(line, tokens, token_index)
    return False


def _is_translation_target_text(text_val):
    if not text_val or contains_chinese(text_val) or len(text_val) <= 1:
        return False
    if is_non_translatable(text_val):
        return False
    if (" " not in text_val) and ("/" in text_val or "\\" in text_val):
        return False
    return (
        " " in text_val
        or len(text_val) > 15
        or (text_val and text_val[0].isupper())
        or (ALLOW_SINGLE_WORD_TRANSLATION and is_english_like(text_val))
    )


def _ensure_identity_block_occurrence(block_occurrences, block_name, current_occurrence):
    if current_occurrence:
        return current_occurrence
    next_occurrence = block_occurrences.get(block_name, 0) + 1
    block_occurrences[block_name] = next_occurrence
    return next_occurrence


def scan_all_translation_units(lines, file_rel_path, mode=translation_core.MODE_TRANSLATION):
    mapping = {}
    is_translation_file = any(
        line.lstrip().startswith("translate ")
        for line in lines
    )
    speaker_names = {}
    character_display_spans = []

    current_block = "_global"
    current_block_occurrence = None
    block_occurrences = {}
    block_index = 0

    for idx, line in enumerate(lines):
        definition = _parse_character_definition(lines, idx)
        if definition:
            character_display_spans.extend(definition["display_spans"])

        sline = line.strip()
        if sline.startswith("translate "):
            block_name = _translate_block_name(line)
            if block_name:
                current_block = block_name
                current_block_occurrence = None
                block_index = 0

        if (
            not sline
            or sline.startswith("#")
            or sline.startswith("translate ")
            or sline == "voice"
            or sline.startswith("voice ")
            or (is_translation_file and sline.startswith("old "))
        ):
            continue

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
            for token_index, token in enumerate(tokens):
                if token.type != tokenize.STRING:
                    continue
                if is_translation_file and _is_keyword_argument_string_token(tokens, token_index):
                    continue
                if _is_character_display_token(idx, token, character_display_spans):
                    continue
                try:
                    text_val = ast.literal_eval(token.string)
                except Exception:
                    continue
                if not isinstance(text_val, str):
                    continue

                source_marker = find_source_text_for_translation_line(lines, idx) if is_translation_file else None
                source_for_id = source_marker if source_marker is not None else text_val
                if source_for_id is None:
                    source_for_id = text_val

                should_translate = _is_translation_target_text(text_val)
                identity_bearing = (is_translation_file and source_marker is not None) or should_translate
                if mode == translation_core.MODE_TRANSLATION and not identity_bearing:
                    continue
                if mode == translation_core.MODE_REVISION and is_translation_file and source_marker is None:
                    continue

                current_block_occurrence = _ensure_identity_block_occurrence(
                    block_occurrences,
                    current_block,
                    current_block_occurrence,
                )
                block_index += 1
                identity = translation_core.build_identity_v2(
                    file_rel_path,
                    current_block,
                    block_index,
                    source_for_id,
                    block_occurrence=current_block_occurrence,
                )
                mapping[identity] = (idx, token.start[1], token.end[1], text_val)
        except Exception:
            continue

    return mapping


def collect_tasks_with_progress(lines, skip_translated=True):
    # Logic to parse Ren'Py files
    # Note: caller handles filename lookup, this function just parses
    #
    # Returns (pending_tasks, progress) where progress includes:
    # - translated_count: identity-bearing targets that already contain Chinese
    #   (same heuristic as skip_translated / batch pending collection)

    tasks = []
    translated_count = 0
    # Detect Ren'Py translation files so we can protect `old` entries.
    is_translation_file = any(
        line.lstrip().startswith("translate ")
        for line in lines
    )
    speaker_names = {}
    character_display_spans = []

    current_block = "_global"
    current_block_occurrence = None
    block_occurrences = {}
    block_index = 0

    # Simple parser for Ren'Py strings
    for idx, line in enumerate(lines):
        definition = _parse_character_definition(lines, idx)
        if definition:
            speaker_id = definition["speaker_id"]
            speaker_name = definition["speaker_name"]
            if speaker_name:
                speaker_names[speaker_id] = speaker_name
            else:
                speaker_names.pop(speaker_id, None)
            character_display_spans.extend(definition["display_spans"])

        sline = line.strip()
        if sline.startswith("translate "):
            block_name = _translate_block_name(line)
            if block_name:
                current_block = block_name
                current_block_occurrence = None
                block_index = 0

        # In translation templates, `old` is a lookup key and must never be edited.
        if (
            not sline
            or sline.startswith("#")
            or sline.startswith("translate ")
            or sline == "voice"
            or sline.startswith("voice ")
            or (is_translation_file and sline.startswith("old "))
        ):
            continue

        # Very basic string extraction (robust enough for this task)
        # Look for dialogue lines: Character "Text" or "Text"
        # And strings: old "Text"

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
            for token_index, token in enumerate(tokens):
                if token.type != tokenize.STRING:
                    continue
                if is_translation_file and _is_keyword_argument_string_token(tokens, token_index):
                    continue
                if _is_character_display_token(idx, token, character_display_spans):
                    continue
                try:
                    text_val = ast.literal_eval(token.string)
                except Exception:
                    continue
                if not isinstance(text_val, str):
                    continue

                prefix, quote = parse_string_literal_format(token.string)

                # Simple heuristic: if it contains Chinese, it's already translated or source is CN
                # If it's pure ASCII/English, we want to translate it.
                source_marker = find_source_text_for_translation_line(lines, idx) if is_translation_file else None
                should_translate = _is_translation_target_text(text_val)
                identity_bearing = (is_translation_file and source_marker is not None) or should_translate
                if not identity_bearing:
                    continue

                source_for_id = source_marker if source_marker is not None else text_val
                if source_for_id is None:
                    source_for_id = text_val

                current_block_occurrence = _ensure_identity_block_occurrence(
                    block_occurrences,
                    current_block,
                    current_block_occurrence,
                )
                block_index += 1
                if not should_translate:
                    # Count finished units so doctor can show translated vs pending.
                    if (
                        skip_translated
                        and is_translation_file
                        and source_marker is not None
                        and contains_chinese(text_val)
                    ):
                        translated_count += 1
                    continue

                task_id = translation_core.build_identity_v2(
                    "",
                    current_block,
                    block_index,
                    source_for_id,
                    block_occurrence=current_block_occurrence,
                )
                task = {
                    "id": task_id,
                    "text": text_val,
                    "line": idx,
                    "start": token.start[1],
                    "end": token.end[1],
                    "quote": quote,
                    "prefix": prefix,
                    "progress_entry": f"task:{idx}:{token.start[1]}",
                    "block_name": current_block,
                    "block_index": block_index,
                    "block_occurrence": current_block_occurrence,
                    "source_for_id": source_for_id,
                }
                speaker_id = ""
                if not (is_translation_file and sline.startswith("new ")):
                    speaker_id = infer_dialogue_speaker_id(line, token.start[1])
                if speaker_id:
                    task["speaker_id"] = speaker_id
                    task["speaker"] = speaker_id
                    speaker_name = speaker_names.get(speaker_id)
                    if speaker_name:
                        task["speaker_name"] = speaker_name
                tasks.append(task)
        except Exception:
            continue

    return tasks, {"translated_count": translated_count}


def collect_tasks(lines, skip_translated=True):
    tasks, _progress = collect_tasks_with_progress(lines, skip_translated=skip_translated)
    return tasks

def apply_sync_translation_preview(manifest_path, *, allow_durable=False):
    """Apply a generated sync preview after validating every source file.

    Durable previews additionally require run-store freshness validation, so
    only the durable Batch entrypoint may opt into applying one here.
    """
    load_config(require_api_key=False)
    load_translator_settings()

    def record_progress(entry):
        update_progress(entry["relative_path"], entry.get("progress_entries") or [])
        translated_path = os.path.join(TL_DIR, *entry["relative_path"].split("/"))
        maybe_update_sync_rag_store(translated_path, full_file=True)

    try:
        manifest = sync_translation_preview.load_sync_preview(manifest_path)
    except ValueError as exc:
        raise SystemExit(f"Sync apply blocked: {exc}") from exc
    compatibility = manifest.get('_translation_plan_compatibility') or {}
    if compatibility.get('mode') == 'legacy':
        print(f"Warning: {compatibility.get('message')}", file=sys.stderr)
    if manifest.get('durable_check_binding') is not None and not allow_durable:
        raise SystemExit(
            'Sync apply blocked: Durable Sync previews must be applied with '
            '`gemini_translate_batch.py apply <RUN>` so the complete source '
            'snapshot is revalidated.'
        )
    prompt_context = manifest.get("prompt_context") or {}
    manifest_macro_fingerprint = str(prompt_context.get("macro_fingerprint") or "")
    # Legacy manifests without prompt_context keep applying; every manifest
    # recorded by the current preview flow carries a macro fingerprint, so any
    # difference (macro added, removed, or changed since preview) blocks
    # writeback.
    if "prompt_context" in manifest and manifest_macro_fingerprint != SYNC_MACRO_FINGERPRINT:
        raise SystemExit(
            "Sync apply blocked: the macro setting file changed since this "
            "preview was generated. Regenerate and review a new preview."
        )
    runtime_config = _read_json_object(TRANSLATOR_CONFIG, label="translator config")
    active_quality_policy = translation_quality.load_policy_from_config(
        runtime_config
    )
    try:
        manifest = sync_translation_preview.apply_sync_preview(
            manifest_path,
            active_project_root=BASE_DIR,
            active_tl_dir=TL_DIR,
            on_file_applied=record_progress,
            active_quality_policy=active_quality_policy,
            active_glossary_file=GLOSSARY_FILE,
        )
    except ValueError as exc:
        raise SystemExit(f"Sync apply blocked: {exc}") from exc

    summary = manifest.get("summary") or {}
    quality_gate = summary.get("quality_gate") or {}
    print(f"Sync apply manifest: {os.path.abspath(manifest_path)}")
    print(f"Applied files: {len(manifest.get('applied_files') or [])}")
    print(f"Applied translations: {int(summary.get('translated_items') or 0)}")
    print(
        "Sync apply quality gate: "
        f"{quality_gate.get('decision') or 'pass'}, "
        f"warnings={int(quality_gate.get('warning_count') or 0)}, "
        f"blockers={int(quality_gate.get('blocker_count') or 0)}"
    )
    print("Sync translation apply complete.")
    return manifest


def run_translation(*, prepare=False):
    """Translate into a reviewable preview package without changing project scripts."""
    try:
        load_config(require_api_key=False)
    except model_profile.ModelRoutingConfigError as exc:
        raise model_profile.routing_resolution_error(
            exc,
            stage=model_profile.STAGE_TRANSLATION,
        ) from exc
    load_translator_settings()
    if SYNC_BACKEND == "gemini":
        _require_gemini_api_key()
    load_glossary()
    print("=" * 60)
    print("Synchronous Translator Preview (Ren'Py)")
    print(f"Sync backend: {SYNC_BACKEND}")
    print(f"Models: {MODELS}")
    usage_run_id = model_usage_ledger.new_run_id('sync-translation')
    # One preview invocation is one operation; do not collapse all runs by project_id.
    usage_operation_id = usage_run_id
    usage_buffer: list = []
    if SYNC_BACKEND == "gemini":
        print(f"Gemini API Keys Loaded: {len(API_KEYS)}")
    else:
        print("Gemini API Key: not required for LiteLLM")
    print(f"Base dir: {BASE_DIR}")
    print(f"TL subdir: {TL_SUBDIR}")
    print(f"TL dir: {TL_DIR} (exists: {os.path.isdir(TL_DIR)})")
    print(f"Progress log: {PROGRESS_LOG}")
    print("=" * 60)
    routing_plan = freeze_translation_routing_plan()
    translation_route = routing_plan.routes[model_profile.STAGE_TRANSLATION]

    try:
        if prepare:
            run_prepare_steps()
        elif PREP_ENABLED:
            print("Prepare step skipped in preview mode; use --prepare explicitly if needed.")
        if not os.path.isdir(TL_DIR):
            print("WARNING: TL_DIR does not exist in preview mode.")

        adapter = RenPyAdapter(legacy_module=sys.modules[__name__])
        adapter_snapshot = build_translation_snapshot(
            adapter,
            ProjectDiscoveryRequest(
                project_root=BASE_DIR,
                localization_root=TL_DIR,
                target_language=PREP_LANGUAGE,
                include_files=tuple(sorted(INCLUDE_FILES)),
                include_prefixes=tuple(sorted(INCLUDE_PREFIXES)),
            ),
        )
        source_documents = adapter_snapshot.project.source_documents
        occurrences_by_unit_id = {
            occurrence.unit.id: occurrence
            for occurrence in adapter_snapshot.occurrences
            if occurrence.unit.id
        }

        def validate_sync_translation(entry, translated):
            occurrence = occurrences_by_unit_id.get(str(entry.get("id") or ""))
            if occurrence is None:
                return False, "common.locator.unresolved"
            validation = adapter.validate_translation(occurrence, translated)
            if validation.status == "pass":
                return True, "OK"
            reason = ", ".join(validation.reason_codes) or "adapter.validation.block"
            return False, reason

        files_to_process = [
            document.file_path
            for document in source_documents
        ]
        print(f"Found {len(files_to_process)} files.")

        global_progress = _upgrade_legacy_progress_keys(load_progress(), files_to_process)
        preview_files = []
        preview_failures = []
        contract_failures = []
        contract_diagnostics = new_sync_contract_diagnostics()
        attempted_file_contexts = []
        pending_jobs = []
        for document in source_documents:
            progress_key = document.file_rel_path
            completed_entries = set(
                _normalize_progress_entries(global_progress.get(progress_key, []))
            )
            tasks = []
            for raw_task in adapter_snapshot.pending_tasks_by_file.get(
                progress_key,
                (),
            ):
                task = dict(raw_task)
                if is_non_translatable(task["text"]):
                    continue
                progress_entry = (
                    task.get("progress_entry") or _progress_entry_for_task(task)
                )
                if (
                    progress_entry in completed_entries
                    or _progress_line_entry(task["line"]) in completed_entries
                ):
                    if (
                        not FORCE_RETRANSLATE_ENGLISH
                        or not is_english_like(task["text"])
                    ):
                        continue
                task["progress_entry"] = _progress_entry_for_task(task)
                task["file_rel_path"] = progress_key
                task["file_path"] = document.file_path
                tasks.append(task)
            if tasks:
                pending_jobs.append({
                    'file_rel_path': progress_key,
                    'file_path': document.file_path,
                    'tasks': tasks,
                })

        sync_plan_build, sync_plan_captures = build_sync_translation_plan(
            pending_jobs,
            adapter_snapshot,
            routing_plan,
            run_id=usage_run_id,
        )
        plan_records_by_file = {}
        for index, request in enumerate(sync_plan_build.requests):
            plan_chunk = sync_plan_build.plan.chunks[index]
            capture = sync_plan_captures[index]
            plan_records_by_file.setdefault(plan_chunk.file_rel_path, []).append(
                (plan_chunk, request, capture)
            )
        pending_jobs_by_file = {
            str(job.get('file_rel_path') or ''): job for job in pending_jobs
        }
        print(
            'Sync TranslationPlan: '
            f'id={sync_plan_build.plan.plan_id}, '
            f'fingerprint={sync_plan_build.plan.plan_fingerprint}, '
            f'requests={len(sync_plan_build.requests)}'
        )
        sync_plan_dispatch_diagnostics = (
            validate_sync_translation_plan_before_dispatch(sync_plan_build)
        )
        print(
            'Sync TranslationPlan freshness: '
            f"source={sync_plan_dispatch_diagnostics['source']}, "
            f"adapter={sync_plan_dispatch_diagnostics['adapter']}, "
            f"plan={sync_plan_dispatch_diagnostics['plan']}"
        )
        for document in source_documents:
            file_path = document.file_path
            filename = os.path.basename(file_path)
            progress_key = document.file_rel_path
            print(f"\nProcessing: {filename}")

            tasks = list(
                (pending_jobs_by_file.get(progress_key) or {}).get('tasks') or []
            )

            if not tasks:
                print("  No new lines to translate.")
                continue
            lines = document.lines()
            print(f"  Found {len(tasks)} lines to translate.")

            replacements = {}
            successful_entries = []
            file_context_batches = []
            task_by_id = {
                _sync_contract_item_id(task.get('id')): task for task in tasks
            }
            for plan_chunk, translation_request, request_context in (
                plan_records_by_file.get(progress_key) or []
            ):
                batch = [
                    task_by_id[item_id]
                    for item_id in translation_request.expected_ids
                    if item_id in task_by_id
                ]
                if len(batch) != len(translation_request.expected_ids):
                    raise RuntimeError(
                        'Sync TranslationPlan no longer matches pending tasks for '
                        f'{progress_key}: request={translation_request.request_id}.'
                    )
                context_stats = dict(plan_chunk.context_window_spec or {})
                context_stats.update({
                    'request_id': translation_request.request_id,
                    'chunk_id': translation_request.chunk_id,
                    'prompt_fingerprint': translation_request.prompt_fingerprint,
                    'request_fingerprint': translation_request.request_fingerprint,
                    'context_assembly': dict(
                        translation_request.context_assembly or {}
                    ),
                    'rag_stats': dict(request_context.get('rag_stats') or {}),
                    'source_index_stats': dict(
                        request_context.get('source_index_stats') or {}
                    ),
                    'project_analysis': dict(
                        request_context.get('project_analysis') or {}
                    ),
                })
                file_context_batches.append(context_stats)
                successful_entries.extend(process_batch_with_retry(
                    batch,
                    replacements,
                    usage_run_id=usage_run_id,
                    translation_validator=validate_sync_translation,
                    usage_buffer=usage_buffer,
                    usage_operation_id=usage_operation_id,
                    contract_diagnostics=contract_diagnostics,
                    contract_failures=contract_failures,
                    context_window=request_context.get('context_window'),
                    route=translation_route,
                    plan=routing_plan,
                    translation_request=translation_request,
                    request_context=request_context,
                ))
            # Persist attempted-batch diagnostics even when the file produced
            # no preview (all items rejected or adapter writeback blocked), so
            # the manifest can still explain the context construction.
            if file_context_batches:
                attempted_file_contexts.append({
                    "relative_path": progress_key,
                    "batches": file_context_batches,
                })

            if any(replacements.values()):
                normalized_entries = _normalize_progress_entries(successful_entries)
                accepted_entries = set(normalized_entries)
                quality_subjects = []
                for task in tasks:
                    task_progress = _normalize_progress_entry(
                        task.get("progress_entry")
                    )
                    if not task_progress or task_progress not in accepted_entries:
                        continue
                    translated_text = str(task.get("translated_text") or "")
                    if not translated_text:
                        task_line = int(task.get("line") or 0)
                        task_start = int(task.get("start") or 0)
                        for replacement in replacements.get(task_line, []) or []:
                            if (
                                isinstance(replacement, (tuple, list))
                                and len(replacement) >= 3
                                and int(replacement[0]) == task_start
                            ):
                                translated_text = str(replacement[2] or "")
                                break
                    quality_subjects.append(
                        {
                            "item_id": str(task.get("id") or ""),
                            "file_rel_path": progress_key,
                            "line": int(task.get("line") or 0),
                            "line_number": int(task.get("line") or 0) + 1,
                            "start": int(task.get("start") or 0),
                            "end": int(task.get("end") or 0),
                            "source": str(
                                task.get("source_for_id")
                                or task.get("text")
                                or ""
                            ),
                            "translation": translated_text,
                            "speaker_id": str(task.get("speaker_id") or ""),
                            "speaker_name": str(task.get("speaker_name") or ""),
                        }
                    )
                adapter_plan, rendered_by_file, preview_failure = build_sync_adapter_preview(
                    adapter,
                    adapter_snapshot,
                    progress_key,
                    tasks,
                    replacements,
                )
                if preview_failure is not None:
                    preview_failures.append(preview_failure)
                    print(f"  Warning: Preview skipped for {filename}: {preview_failure['message']}")
                    continue
                preview_lines = rendered_by_file[progress_key]
                source_text = "".join(lines)
                preview_text = "".join(preview_lines)
                # Preserve a leading UTF-8 BOM so source_sha256 (raw bytes) and
                # apply writeback stay consistent with the on-disk file.
                if document.content.startswith(b"\xef\xbb\xbf"):
                    if not source_text.startswith("\ufeff"):
                        source_text = "\ufeff" + source_text
                    if not preview_text.startswith("\ufeff"):
                        preview_text = "\ufeff" + preview_text
                preview_files.append(
                    {
                        "relative_path": progress_key,
                        "source_text": source_text,
                        "source_sha256": document.sha256,
                        "preview_text": preview_text,
                        "progress_entries": normalized_entries,
                        "translated_items": len(normalized_entries),
                        "writeback_plan": adapter_plan.to_dict(),
                        "quality_subjects": quality_subjects,
                        "prompt_context": {
                            "batches": file_context_batches,
                        },
                    }
                )
            print(f"  Previewed {filename}.")

        preview_failures.extend(contract_failures)
        finalized_contract = finalize_sync_contract_diagnostics(contract_diagnostics)
        context_batch_count = sum(
            len(file_entry.get("batches") or [])
            for file_entry in attempted_file_contexts
        )
        context_truncated_batches = sum(
            1
            for file_entry in attempted_file_contexts
            for batch_stats in file_entry.get("batches") or []
            if batch_stats.get("context_truncated")
        )
        macro_path = _resolve_path(BASE_DIR, SYNC_MACRO_SETTING_FILE)
        prompt_context = {
            "context_before": SYNC_CONTEXT_BEFORE,
            "context_after": SYNC_CONTEXT_AFTER,
            "macro_setting_file": SYNC_MACRO_SETTING_FILE,
            "macro_setting_path": macro_path,
            "macro_fingerprint": SYNC_MACRO_FINGERPRINT,
            "macro_applied": bool(SYNC_MACRO_SETTING),
            "batches": context_batch_count,
            "truncated_batches": context_truncated_batches,
            "files": attempted_file_contexts,
        }
        runtime_config = _read_json_object(
            TRANSLATOR_CONFIG,
            label="translator config",
        )
        quality_policy = translation_quality.load_policy_from_config(
            runtime_config
        )
        manifest_path, manifest = sync_translation_preview.create_sync_preview(
            log_dir=LOG_DIR,
            project_root=BASE_DIR,
            tl_dir=TL_DIR,
            files=preview_files,
            failures=preview_failures,
            contract_diagnostics=finalized_contract,
            prompt_context=prompt_context,
            quality_policy=quality_policy,
            glossary_file=GLOSSARY_FILE,
            translation_plan_payload=sync_plan_build.plan.to_dict(),
            request_ids=[
                request.request_id for request in sync_plan_build.requests
            ],
        )
        try:
            export_coverage_package(
                os.path.join(os.path.dirname(manifest_path), "coverage"),
                adapter_snapshot.project,
                adapter_snapshot.inventory,
                adapter_snapshot.report,
                review_policy=adapter_snapshot.review_policy,
            )
        except (OSError, ValueError) as exc:
            # Coverage is read-only P1 evidence; export failure must not block preview.
            print(f"WARNING: Coverage export skipped: {exc}")
        report_path = os.path.join(os.path.dirname(manifest_path), "preview.diff")
        summary = manifest.get("summary") or {}
        print(f"Sync preview manifest: {manifest_path}")
        print(f"Sync preview report: {report_path}")
        print(f"Preview files: {int(summary.get('files_changed') or 0)}")
        print(f"Preview translations: {int(summary.get('translated_items') or 0)}")
        quality_gate = summary.get("quality_gate") or {}
        quality_findings_path = manifest.get("last_quality_findings_path")
        if quality_findings_path:
            quality_findings_path = os.path.join(
                os.path.dirname(manifest_path),
                quality_findings_path,
            )
        print(
            "Sync quality gate: "
            f"{quality_gate.get('decision') or 'pass'}, "
            f"warnings={int(quality_gate.get('warning_count') or 0)}, "
            f"blockers={int(quality_gate.get('blocker_count') or 0)}"
        )
        if quality_findings_path:
            print(f"Sync quality findings: {quality_findings_path}")
        print(
            "Model contract completeness: "
            f"{finalized_contract.get('final_valid', 0)}/"
            f"{finalized_contract.get('final_expected', 0)}"
        )
        print(
            f"Sync local context: before={SYNC_CONTEXT_BEFORE}, after={SYNC_CONTEXT_AFTER}, "
            f"batches={context_batch_count}, truncated={context_truncated_batches}",
            flush=True,
        )
        print(
            f"Sync macro setting: file={SYNC_MACRO_SETTING_FILE}, "
            f"applied={bool(SYNC_MACRO_SETTING)}, "
            f"fingerprint={SYNC_MACRO_FINGERPRINT or '(none)'}",
            flush=True,
        )
        print(
            "Targeted retries: "
            f"{finalized_contract.get('targeted_retry_requests', 0)} requests / "
            f"{finalized_contract.get('targeted_retry_items', 0)} items"
        )
        print(
            "Unresolved contract items: "
            f"{len(finalized_contract.get('unresolved_ids') or [])}"
        )
        print_sync_usage_summary(usage_buffer)
        contract_partial = bool(
            finalized_contract.get('unresolved_ids')
            or finalized_contract.get('terminal_reason_counts')
        )
        if preview_failures:
            print(f"Preview failures: {len(preview_failures)}")
        if preview_failures or contract_partial:
            print("Preview status: partial")
        else:
            print("Preview status: safe")
        print("No project scripts were modified. Review the diff, then run with --apply MANIFEST.")
        return manifest_path
    finally:
        if usage_buffer and BASE_DIR:
            try:
                model_usage_ledger.UsageLedger(BASE_DIR).add_records(usage_buffer)
            except (OSError, ValueError, model_usage_ledger.UsageLedgerError) as exc:
                print(
                    f'Warning: Model usage ledger flush failed: {exc}',
                    flush=True,
                )
