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
import zlib
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_runtime_state_lock = threading.Lock()


@contextmanager
def locked_runtime_state():
    """Serialize temporary BASE_DIR overrides across worker threads."""
    with _runtime_state_lock:
        yield

from atomic_io import atomic_write_json, atomic_write_lines
from rag_memory import JsonRagStore, hash_text, truncate_text
import prompt_context
import story_memory
import translation_core
from sync_model_backend import GeminiSyncBackend, SyncGenerationRequest

# Configuration
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
FLAT_CONFIG = os.path.join(TOOL_DIR, "api_keys.json")
TRANSLATOR_CONFIG = os.path.join(TOOL_DIR, "translator_config.json")
DEFAULT_GLOSSARY_FILE = os.path.join(TOOL_DIR, "glossary.json")
GLOSSARY_FILE = DEFAULT_GLOSSARY_FILE
if os.path.isfile(FLAT_CONFIG):
    ROOT_DIR = TOOL_DIR
    DATA_DIR = TOOL_DIR
    CONFIG_FILE = FLAT_CONFIG
else:
    ROOT_DIR = os.path.abspath(os.path.join(TOOL_DIR, ".."))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    CONFIG_FILE = os.path.join(DATA_DIR, "api_keys.json")
ENV_GAME_ROOT = os.environ.get("GAME_ROOT") or os.environ.get("SA_GAME_ROOT")
DEFAULT_TL_SUBDIR = os.path.join("game", "tl", "schinese")
TL_SUBDIR = DEFAULT_TL_SUBDIR
BASE_DIR = os.path.abspath(ENV_GAME_ROOT) if ENV_GAME_ROOT else os.path.abspath(os.path.join(ROOT_DIR, ".."))
TL_DIR = os.path.abspath(os.path.join(BASE_DIR, TL_SUBDIR))
WORK_GAME_SUBDIR = "game"
WORK_GAME_DIR = os.path.abspath(os.path.join(BASE_DIR, WORK_GAME_SUBDIR))
CONTEXT_STORAGE_LOCATION = "tool"
CONTEXT_STORAGE_GAME_DIR_NAME = "translation_context"
SOURCE_GAME_DIR = ""
PREP_ENABLED = True
PREP_UNPACK_RPA = True
PREP_GENERATE_TEMPLATE = True
PREP_REFRESH_EXISTING_TEMPLATE = True
PREP_LANGUAGE = "schinese"
PREP_RENPY_SDK_DIR = ""
PREP_LAUNCHER_PY = ""
PREP_PYTHON_EXE = ""
PREP_UNPACK_COMMAND = None
PREP_TEMPLATE_COMMAND = None
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
MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro"
]

# Runtime State
CURRENT_KEY_INDEX = 0
CURRENT_MODEL_INDEX = 0
API_KEYS = []

PRESERVE_TERMS = [
    "???", "????", "?????", "[name]", "{sc=4}???{/sc}",
]

# Config Defaults
MAX_CHARS = 12000
MAX_ITEMS = 40
SYNC_MAX_OUTPUT_TOKENS = 24576
SYNC_BACKEND = "gemini"
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
SYNC_RAG_EMBEDDING_MODEL = "gemini-embedding-001"
SYNC_RAG_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
SYNC_RAG_DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
SYNC_RAG_OUTPUT_DIMENSIONALITY = 768
SYNC_RAG_TOP_K_HISTORY = 4
SYNC_RAG_TOP_K_TERMS = 8
SYNC_RAG_MIN_SIMILARITY = 0.72
SYNC_RAG_SEGMENT_LINES = 4
SYNC_RAG_HISTORY_CHAR_LIMIT = 220
SYNC_RAG_UPDATE_ON_SUCCESS = True
SYNC_RAG_QUALITY_STATE = "sync_applied"
_SYNC_RAG_STORE = None

# Optional structured story memory for synchronous translation. Disabled by
# default to keep sync repair and smoke-test runs lightweight unless configured.
SYNC_STORY_MEMORY_ENABLED = False
SYNC_STORY_MEMORY_GRAPH_FILE = ""
SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = 800
SYNC_STORY_MEMORY_TOP_K_RELATIONS = 4
SYNC_STORY_MEMORY_TOP_K_TERMS = 8
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
    allowed = {
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    }
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in allowed:
            return normalized
    return default


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
    return _canonical_abs_path(os.path.join(base_dir, text))


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


def _is_renpy_sdk_dir(path):
    return bool(
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "renpy.py"))
    )


def _discover_renpy_sdk_dir():
    if _is_filesystem_root(BASE_DIR):
        return ""

    base_dirs = _dedupe_keep_order(
        [
            BASE_DIR,
            os.path.dirname(BASE_DIR),
            os.path.dirname(os.path.dirname(BASE_DIR)),
            ROOT_DIR,
            os.path.dirname(ROOT_DIR),
            TOOL_DIR,
        ]
    )
    candidates = []
    for base_dir in base_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        if os.path.isfile(os.path.join(base_dir, "renpy.py")):
            candidates.append(base_dir)
        for pattern in ("renpy-*-sdk", "renpy-*sdk", "renpy-sdk"):
            candidates.extend(glob.glob(os.path.join(base_dir, pattern)))

    candidates = sorted(
        {
            os.path.abspath(candidate)
            for candidate in candidates
            if _is_renpy_sdk_dir(candidate)
        },
        key=_renpy_sdk_sort_key,
        reverse=True,
    )
    return candidates[0] if candidates else ""


def resolve_story_memory_graph_path(value):
    return _resolve_preferred_path_from_bases(value, (ROOT_DIR, BASE_DIR, TOOL_DIR))


def _coerce_command(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, list):
        cmd = []
        for item in value:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                cmd.append(token)
        return cmd or None
    return None


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
        "gemini-embedding-001",
    )
    SYNC_RAG_QUERY_TASK_TYPE = _normalize_task_type(
        rag.get("query_task_type"),
        "RETRIEVAL_QUERY",
    )
    SYNC_RAG_DOCUMENT_TASK_TYPE = _normalize_task_type(
        rag.get("document_task_type"),
        "RETRIEVAL_DOCUMENT",
    )
    SYNC_RAG_OUTPUT_DIMENSIONALITY = _coerce_positive_int(
        rag.get("output_dimensionality"),
        768,
    )
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
        800,
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


def load_sync_translation_settings(config):
    global MAX_ITEMS, MAX_CHARS, SYNC_MAX_OUTPUT_TOKENS, SYNC_BACKEND

    sync = config.get("sync")
    if not isinstance(sync, dict):
        sync = {}

    backend_name = str(sync.get("backend") or "gemini").strip().lower()
    if backend_name not in {"gemini", "litellm"}:
        raise ValueError(
            f"Unsupported sync backend: {backend_name}. Choose 'gemini' or 'litellm'."
        )
    SYNC_BACKEND = backend_name

    custom_models = sync.get("models")
    single_model = sync.get("model")
    if not custom_models and single_model:
        custom_models = [single_model]
    elif isinstance(custom_models, str):
        custom_models = [custom_models]
    if custom_models:
        replace_model_list(custom_models, "sync")

    previous_items = MAX_ITEMS
    previous_chars = MAX_CHARS
    previous_output_tokens = SYNC_MAX_OUTPUT_TOKENS

    MAX_ITEMS = _coerce_positive_int(sync.get("chunk_size"), MAX_ITEMS)
    MAX_CHARS = _coerce_positive_int(
        sync.get("max_source_chars", sync.get("target_chars")),
        MAX_CHARS,
    )
    SYNC_MAX_OUTPUT_TOKENS = _coerce_positive_int(
        sync.get("max_output_tokens"),
        SYNC_MAX_OUTPUT_TOKENS,
    )

    if MAX_ITEMS != previous_items:
        print(f"Using sync chunk size: {MAX_ITEMS}")
    if MAX_CHARS != previous_chars:
        print(f"Using sync max source chars: {MAX_CHARS}")
    if SYNC_MAX_OUTPUT_TOKENS != previous_output_tokens:
        print(f"Using sync max output tokens: {SYNC_MAX_OUTPUT_TOKENS}")


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


def load_translator_settings():
    """Loads per-game settings (game root, tl subdir) from translator_config.json or env."""
    global BASE_DIR, TL_DIR, TL_SUBDIR, ENV_GAME_ROOT, WORK_GAME_DIR, SOURCE_GAME_DIR, GLOSSARY_FILE
    global PREP_ENABLED, PREP_UNPACK_RPA, PREP_GENERATE_TEMPLATE, PREP_REFRESH_EXISTING_TEMPLATE, PREP_LANGUAGE
    global PREP_RENPY_SDK_DIR, PREP_LAUNCHER_PY, PREP_PYTHON_EXE, PREP_UNPACK_COMMAND, PREP_TEMPLATE_COMMAND

    config = {}
    if os.path.exists(TRANSLATOR_CONFIG):
        try:
            with open(TRANSLATOR_CONFIG, "r", encoding="utf-8-sig") as handle:
                config = json.load(handle) or {}
        except Exception as e:
            print(f"Warning: Failed to load translator config: {e}")

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
            if isinstance(game_root, str) and game_root.strip() and os.path.exists(TRANSLATOR_CONFIG):
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
        BASE_DIR = _canonical_abs_path(os.path.join(ROOT_DIR, ".."))

    load_context_storage_settings(config)

    glossary_file = config.get("glossary_file")
    if glossary_file is None:
        glossary_file = config.get("glossary_path")
    if glossary_file is not None:
        resolved_glossary = _resolve_preferred_path(TOOL_DIR, BASE_DIR, glossary_file)
        GLOSSARY_FILE = resolved_glossary or DEFAULT_GLOSSARY_FILE
    else:
        GLOSSARY_FILE = DEFAULT_GLOSSARY_FILE

    tl_subdir = config.get("tl_subdir")
    try:
        candidate_subdir = TL_SUBDIR
        if isinstance(tl_subdir, str) and tl_subdir.strip():
            candidate_subdir = normalize_tl_subdir(tl_subdir)
        else:
            # Keep the active/default value, but re-validate so a previously
            # accepted path still cannot escape after game_root changes.
            candidate_subdir = normalize_tl_subdir(candidate_subdir)
        candidate_tl_dir = _canonical_abs_path(os.path.join(BASE_DIR, candidate_subdir))
        ensure_tl_dir_within_base(
            BASE_DIR,
            candidate_tl_dir,
            tl_subdir=candidate_subdir,
        )
    except InvalidTlSubdirError as exc:
        raise SystemExit(
            "ERROR: Invalid tl_subdir configuration. "
            "tl_subdir must be a relative path under game_root with no '..' segments "
            f"(example: 'game/tl/schinese'). Details: {exc}"
        ) from exc

    TL_SUBDIR = candidate_subdir
    TL_DIR = candidate_tl_dir
    WORK_GAME_DIR = _canonical_abs_path(os.path.join(BASE_DIR, WORK_GAME_SUBDIR))

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

    renpy_sdk_dir = prepare.get("renpy_sdk_dir")
    if not (isinstance(renpy_sdk_dir, str) and renpy_sdk_dir.strip()):
        renpy_sdk_dir = os.environ.get("RENPY_SDK_DIR")
    resolved_renpy_sdk_dir = _resolve_preferred_path_from_bases(
        renpy_sdk_dir,
        (BASE_DIR, ROOT_DIR, TOOL_DIR),
    )
    if _is_renpy_sdk_dir(resolved_renpy_sdk_dir):
        PREP_RENPY_SDK_DIR = resolved_renpy_sdk_dir
    else:
        PREP_RENPY_SDK_DIR = _discover_renpy_sdk_dir()

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

    PREP_UNPACK_COMMAND = _coerce_command(prepare.get("unpack_command"))
    PREP_TEMPLATE_COMMAND = _coerce_command(prepare.get("template_command"))
    load_include_filters_from_config(config)
    load_sync_translation_settings(config)
    load_sync_rag_settings(config)
    load_sync_story_memory_settings(config)


def _require_gemini_api_key():
    if API_KEYS:
        return
    print("="*60)
    print("ERROR: No valid API keys found!")
    print("Please check api_keys.json or set GEMINI_API_KEY env vars.")
    print("="*60)
    raise SystemExit("No API keys available")


def load_config(*, require_api_key=True):
    """Loads API keys and settings from api_keys.json or environment."""
    global API_KEYS, MODELS, MAX_CHARS, MAX_ITEMS, SYNC_MAX_OUTPUT_TOKENS
    global INCLUDE_FILES, INCLUDE_PREFIXES
    
    # Try loading from JSON
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
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
                
                if keys:
                    API_KEYS = [
                        k for k in keys
                        if isinstance(k, str) and k.strip() and not _is_placeholder_api_key(k)
                    ]
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

    # Fallback to Environment Variables if no keys found
    if not API_KEYS:
        print("Checking environment variables for keys...")
        env_keys = [
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GEMINI_API_KEY_2"),
            os.environ.get("GEMINI_API_KEY_3"),
        ]
        API_KEYS = [k for k in env_keys if k]

    if require_api_key:
        _require_gemini_api_key()


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


def resolve_project_root(base_dir=None):
    base = _canonical_abs_path(base_dir or BASE_DIR)
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

    root = resolve_project_root(base_dir)
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
    work_dir = resolve_work_dir(base_dir)
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
    base = _canonical_abs_path(base_dir or BASE_DIR)
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


def _validate_rpa_index(raw_index):
    if not isinstance(raw_index, dict):
        raise RuntimeError("Invalid RPA index: expected a dictionary.")
    return raw_index


def _read_rpa_index(archive_path):
    with open(archive_path, "rb") as infile:
        header = infile.read(40)

        if header.startswith(b"RPA-3.0 "):
            offset = int(header[8:24], 16)
            key = int(header[25:33], 16)
            infile.seek(offset)
            raw_index = _validate_rpa_index(_load_pickle_blob(zlib.decompress(infile.read())))

            index = {}
            for name, chunks in raw_index.items():
                decoded_chunks = []
                for chunk in chunks:
                    if len(chunk) == 2:
                        start = b""
                        chunk_offset, chunk_len = chunk
                    else:
                        chunk_offset, chunk_len, start = chunk
                        if start is None:
                            start = b""
                        elif not isinstance(start, bytes):
                            start = str(start).encode("latin-1", errors="ignore")

                    decoded_chunks.append((int(chunk_offset) ^ key, int(chunk_len) ^ key, start))
                index[name] = decoded_chunks

            return index

        if header.startswith(b"RPA-2.0 "):
            infile.seek(0)
            line = infile.read(24)
            offset = int(line[8:], 16)
            infile.seek(offset)
            raw_index = _validate_rpa_index(_load_pickle_blob(zlib.decompress(infile.read())))

            index = {}
            for name, chunks in raw_index.items():
                decoded_chunks = []
                for chunk in chunks:
                    chunk_offset, chunk_len = chunk[:2]
                    start = b""
                    if len(chunk) >= 3:
                        start = chunk[2] or b""
                        if not isinstance(start, bytes):
                            start = str(start).encode("latin-1", errors="ignore")
                    decoded_chunks.append((int(chunk_offset), int(chunk_len), start))
                index[name] = decoded_chunks

            return index

    raise RuntimeError("Unsupported RPA format (expecting RPA-3.0 or RPA-2.0).")


def _extract_rpa_scripts(archive_path, target_game_dir):
    index = _read_rpa_index(archive_path)
    target_root = os.path.abspath(target_game_dir)
    extracted = 0

    with open(archive_path, "rb") as source:
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

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as target:
                for chunk_offset, chunk_len, start in chunks:
                    if start:
                        target.write(start)
                    source.seek(chunk_offset)
                    target.write(source.read(chunk_len))
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

    rendered = _fmt(command)
    return rendered, True


def _run_prepare_command(command, cwd, step_name):
    use_shell = isinstance(command, str)
    shown = command if use_shell else " ".join(command)
    print(f"[Prepare] {step_name}: {shown}")

    try:
        result = subprocess.run(command, cwd=cwd, shell=use_shell, check=False)
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
            rendered, _ = _render_prepare_command(PREP_TEMPLATE_COMMAND, variables)
        except Exception as e:
            return {
                "available": False,
                "kind": "custom",
                "reason": str(e),
                "command": None,
                "cwd": BASE_DIR,
                "python_exe": python_exe,
                "launcher_py": launcher_py,
            }
        return {
            "available": True,
            "kind": "custom",
            "reason": "",
            "command": rendered,
            "cwd": BASE_DIR,
            "python_exe": python_exe,
            "launcher_py": launcher_py,
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
                for archive in archives:
                    try:
                        extracted = _extract_rpa_scripts(archive, WORK_GAME_DIR)
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

def rotate_api_key():
    global CURRENT_KEY_INDEX
    if len(API_KEYS) > 1:
        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
        print(f"  ➜ Rotating to API Key #{CURRENT_KEY_INDEX + 1}")
        return True
    return False

def rotate_model():
    global CURRENT_MODEL_INDEX
    if len(MODELS) > 1:
        CURRENT_MODEL_INDEX = (CURRENT_MODEL_INDEX + 1) % len(MODELS)
        print(f"  ➜ Rotating to Model: {MODELS[CURRENT_MODEL_INDEX]}")
        return True
    return False

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
    base_name = os.path.basename(os.path.abspath(base_dir))
    if base_name.lower() == "work":
        parent = os.path.basename(os.path.dirname(os.path.abspath(base_dir)))
        return _slugify(parent or base_name)
    project_root = resolve_project_root(base_dir)
    return _slugify(os.path.basename(os.path.normpath(project_root)))


def guess_project_slug():
    return _project_slug_from_base_dir(BASE_DIR)


def get_context_storage_location():
    return CONTEXT_STORAGE_LOCATION


def get_context_storage_root(base_dir=None):
    if CONTEXT_STORAGE_LOCATION == "game":
        return _canonical_abs_path(os.path.join(resolve_project_root(base_dir), CONTEXT_STORAGE_GAME_DIR_NAME))
    return LOG_DIR


def get_default_context_store_dir(store_name, base_dir=None):
    root = get_context_storage_root(base_dir)
    if CONTEXT_STORAGE_LOCATION == "game":
        return os.path.join(root, store_name)
    slug = _project_slug_from_base_dir(base_dir or BASE_DIR)
    return os.path.join(root, store_name, slug)


def get_default_batch_rag_store_dir():
    return get_default_context_store_dir("rag_store")


def get_default_source_index_store_dir():
    return get_default_context_store_dir("source_index_store")


def get_default_story_memory_graph_path():
    if CONTEXT_STORAGE_LOCATION == "game":
        return os.path.join(get_context_storage_root(), "story_memory", "story_graph.json")
    return os.path.join(LOG_DIR, "story_memory", "story_graph.json")


def get_default_sync_rag_store_dir():
    return get_default_context_store_dir("rag_store")


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
    return _SYNC_RAG_STORE


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
    genai = get_genai_module()
    client = create_genai_client()
    response = client.models.embed_content(
        model=SYNC_RAG_EMBEDDING_MODEL,
        contents=contents,
        config=genai.types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=SYNC_RAG_OUTPUT_DIMENSIONALITY,
        ),
    )
    embeddings = getattr(response, "embeddings", None) or []
    values = [list(getattr(item, "values", None) or []) for item in embeddings]
    if len(values) != len(contents):
        raise RuntimeError(f"Embedding count mismatch: expected {len(contents)}, got {len(values)}")
    return values


def embed_sync_query_text(query_text):
    query_text = compact_text(query_text)
    if not query_text:
        return []
    vectors = embed_texts([query_text], SYNC_RAG_QUERY_TASK_TYPE)
    return vectors[0] if vectors else []


def retrieve_sync_glossary_hits(target_items):
    if not SYNC_RAG_ENABLED:
        return []
    combined_text = "\n".join(item.get("text", "") for item in target_items if item.get("text"))
    if not combined_text:
        return []
    hits = []
    seen = set()
    for source, target in (NORMALIZE_TRANSLATION_MAP or {}).items():
        if source and source in combined_text and source not in seen:
            hits.append({"source": source, "target": target, "kind": "normalize"})
            seen.add(source)
    for term in PRESERVE_TERMS:
        if not isinstance(term, str) or not term.strip():
            continue
        if term in combined_text and term not in seen:
            hits.append({"source": term, "target": term, "kind": "preserve"})
            seen.add(term)
    return hits[:SYNC_RAG_TOP_K_TERMS]


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

    query_text = build_sync_rag_query_text(target_items)
    if not query_text:
        return [], {"enabled": True, "reason": "empty_query"}

    try:
        query_vector = embed_sync_query_text(query_text)
        matches = store.search_history(
            query_vector,
            top_k=SYNC_RAG_TOP_K_HISTORY,
            min_similarity=SYNC_RAG_MIN_SIMILARITY,
        )
    except Exception as exc:
        print(f"Warning: Sync RAG history retrieval failed: {exc}")
        return [], {"enabled": True, "error": str(exc)}

    hits = []
    for match in matches:
        hits.append(
            {
                "memory_id": match.get("memory_id", ""),
                "file_rel_path": match.get("file_rel_path", ""),
                "line_start": match.get("line_start", 0),
                "line_end": match.get("line_end", 0),
                "source_text": truncate_text(match.get("source_text", ""), SYNC_RAG_HISTORY_CHAR_LIMIT),
                "translated_text": truncate_text(match.get("translated_text", ""), SYNC_RAG_HISTORY_CHAR_LIMIT),
                "quality_state": match.get("quality_state", ""),
                "score": float(match.get("score", 0.0)),
            }
        )

    return hits, {
        "enabled": True,
        "query_text": truncate_text(query_text, 400),
        "hit_count": len(hits),
    }


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

    basename_map = {}
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        basename_map.setdefault(basename, []).append(_progress_key_for_path(file_path))

    migrated = False
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
        merged_entries = set(_normalize_progress_entries(progress.get(progress_key, [])))
        merged_entries.update(_normalize_progress_entries(legacy_lines))
        progress[progress_key] = sorted(merged_entries)
        progress.pop(basename, None)
        migrated = True

    if migrated:
        save_progress(progress)
    return progress


def commit_replacements(path, lines, replacements):
    """Apply replacements in memory, then atomically replace the target file.

    Writes to a same-directory temp file, fsyncs, and uses ``os.replace`` so a
    crash or I/O error cannot leave a truncated ``.rpy`` behind.
    """
    if not replacements:
        return

    for line_idx, repls in replacements.items():
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        # Sort replacements by start position descending to avoid index shifting
        for repl in sorted(repls, key=lambda x: x[0], reverse=True):
            if len(repl) == 4:
                start, end, translated, quote = repl
                prefix = ""
            else:
                start, end, translated, prefix, quote = repl[:5]
            # Safety check indices
            if start < 0 or end > len(line):
                continue
            normalized = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
            line = line[:start] + quote_with(normalized, quote, prefix=prefix) + line[end:]
        lines[line_idx] = line

    atomic_write_lines(path, lines, encoding="utf-8")


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


def build_prompt(items, glossary_hits=None, history_hits=None, story_hits=None):
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
    if not text:
        raise ValueError("Empty response text")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def normalize_result_items(payload):
    return translation_core.normalize_model_results(
        payload,
        mode=translation_core.MODE_TRANSLATION,
    )


def call_gemini_sdk(prompt, items):
    """Calls the explicitly configured synchronous backend."""
    model_name = get_current_model()
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": SYNC_MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_json_schema": build_response_json_schema(items),
    }

    if SYNC_BACKEND == "litellm":
        from litellm_sync_backend import LiteLLMSyncBackend

        result = LiteLLMSyncBackend().generate(SyncGenerationRequest(
            model=model_name,
            contents=prompt,
            config=generation_config,
        ))
    else:
        configure_genai()
        backend = GeminiSyncBackend(
            create_genai_client(),
            serialize_response=serialize_unknown,
            extract_text=extract_text_from_response_payload,
            extract_finish_reason=extract_finish_reason,
        )
        result = backend.generate(SyncGenerationRequest(
            model=model_name,
            contents=prompt,
            config=generation_config,
        ))

    if result.parsed is not None:
        return normalize_result_items(serialize_unknown(result.parsed))
    if result.response_text:
        return normalize_result_items(parse_json_payload(result.response_text))

    prompt_feedback = extract_prompt_feedback(result.response_payload)
    diagnostics = []
    if prompt_feedback:
        diagnostics.append(f"Prompt feedback: {prompt_feedback}")
    if result.finish_reason:
        diagnostics.append(f"Finish reason: {result.finish_reason}")
    detail = f" ({'; '.join(diagnostics)})" if diagnostics else ""
    raise ValueError(f"Invalid response from API. Missing structured text{detail}.")

def process_batch(batch, replacements):
    glossary_hits = retrieve_sync_glossary_hits(batch) if SYNC_RAG_ENABLED else []
    history_hits, rag_stats = retrieve_sync_history_hits(batch) if SYNC_RAG_ENABLED else ([], {})
    story_hits = retrieve_sync_story_hits(batch) if SYNC_STORY_MEMORY_ENABLED else None
    if rag_stats.get("hit_count"):
        print(f"  Sync RAG memory hits: {rag_stats['hit_count']}", flush=True)
    prompt = build_prompt(
        batch,
        glossary_hits=glossary_hits,
        history_hits=history_hits,
        story_hits=story_hits,
    )

    # Call API (SDK handles connection details)
    results = call_gemini_sdk(prompt, batch)

    if not isinstance(results, list):
        raise RuntimeError(f"API returned {type(results)} instead of list")

    id_map = {item["id"]: item for item in batch}
    valid_progress_entries = []
    seen_result_ids = set()

    for item in results:
        entry = id_map.get(item.get("id"))
        if not entry:
            continue
        if entry["id"] in seen_result_ids:
            print(f"  Warning: Duplicate result id ignored: {entry['id']}")
            continue
        seen_result_ids.add(entry["id"])

        translated = item.get("translation", "")
        memory_translation = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
        valid, msg = validate_translation(entry["text"], translated)

        if not valid:
            print(f"  Warning: Validation failed for {entry['id']}: {msg}")
            continue

        valid_progress_entries.append(entry["progress_entry"])
        entry["translated_text"] = memory_translation
        unit = translation_core.unit_from_sync_task(entry)
        action = translation_core.translation_writeback_action(unit, item)
        replacements.setdefault(action.line, []).append(
            translation_core.writeback_tuple(action, include_expected=False)
        )

    if not valid_progress_entries:
        raise RuntimeError("No valid translations in batch (all items rejected; consider expanding non-translatable rules or switching model)")

    # Calculate total chars to show valid data receipt without spoilers
    total_chars = sum(len(item.get("translation", "")) for item in results)
    print(f"  Translated {len(valid_progress_entries)}/{len(batch)} items. (Received {total_chars} chars of translation)", flush=True)
    return valid_progress_entries


def process_batch_with_retry(batch, replacements, retry_depth=0):
    if retry_depth >= 5:
        log_failure(batch, "Max retry depth reached")
        return []

    error_str = "" # Initialize variable to be safe

    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            # Respect rate limits
            time.sleep(get_random_delay())

            return process_batch(batch, replacements)

        except Exception as e:
            error_str = str(e)
            print(f"  [Attempt {attempt}] Error: {error_str[:100]}...", flush=True)

            # Handle Specific Errors

            # 1. 429 Resource Exhausted -> Rotate Key AND Model if needed
            if "429" in error_str or "ResourceExhausted" in error_str:
                print("  ! Rate limit hit.")

                # First try rotating key
                key_rotated = rotate_api_key()

                # If we have retried multiple times (meaning keys are likely all exhausted for this model)
                # OR we only have 1 key, switch the model.
                if attempt > 1 or not key_rotated:
                    print("  ! Persistent rate limit. Switching model...")
                    if rotate_model():
                        continue

                if key_rotated:
                    continue
                else:
                    # No more keys, wait longer
                    time.sleep(10)

            # 2. 404 Not Found -> Rotate Model (Invalid model name)
            elif "404" in error_str or "NotFound" in error_str:
                print("  ! Model not found.")
                if rotate_model():
                    continue

            # 3. 500/503 Server Errors -> Just wait
            elif "500" in error_str or "503" in error_str:
                time.sleep(5)

            # 4. Truncated/Finish Reason 2 -> Break loop to split batch immediately
            elif "Finish reason: 2" in error_str:
                print("  ! Output truncated. Splitting batch...")
                break # Break retry loop to trigger batch splitting

            # Generic retry backoff
            time.sleep(2 ** attempt)

    # If batch failed after retries, try splitting
    if len(batch) > 1:
        print("  > Splitting batch...", flush=True)
        mid = len(batch) // 2
        r1 = process_batch_with_retry(batch[:mid], replacements, retry_depth + 1)
        r2 = process_batch_with_retry(batch[mid:], replacements, retry_depth + 1)
        return r1 + r2

    log_failure(batch, f"Failed after retries: {error_str}")
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

def run_translation():
    load_config(require_api_key=False)
    load_translator_settings()
    if SYNC_BACKEND == "gemini":
        _require_gemini_api_key()
    load_glossary()
    print("="*60)
    print("Synchronous Translator (Ren'Py)")
    print(f"Sync backend: {SYNC_BACKEND}")
    print(f"Models: {MODELS}")
    if SYNC_BACKEND == "gemini":
        print(f"Gemini API Keys Loaded: {len(API_KEYS)}")
    else:
        print("Gemini API Key: not required for LiteLLM")
    print(f"Base dir: {BASE_DIR}")
    print(f"TL subdir: {TL_SUBDIR}")
    print(f"TL dir: {TL_DIR} (exists: {os.path.isdir(TL_DIR)})")
    print(f"Progress log: {PROGRESS_LOG}")
    print(f"Translator config: {TRANSLATOR_CONFIG} (exists: {os.path.isfile(TRANSLATOR_CONFIG)})")
    print(f"Glossary: {GLOSSARY_FILE} (exists: {os.path.isfile(GLOSSARY_FILE)})")
    print(f"Prepare enabled: {PREP_ENABLED}")
    print(f"Prepare source game dir: {SOURCE_GAME_DIR or '(auto)'}")
    print(f"Prepare language: {PREP_LANGUAGE}")
    print(f"Prepare generate template: {PREP_GENERATE_TEMPLATE}")
    print(f"Prepare refresh existing template: {PREP_REFRESH_EXISTING_TEMPLATE}")
    print(f"Prepare Ren'Py SDK dir: {PREP_RENPY_SDK_DIR or '(not configured)'}")
    print("="*60)

    run_prepare_steps()
    if not os.path.isdir(TL_DIR):
        print("WARNING: TL_DIR does not exist after prepare step.")

    # Walk directory
    files_to_process = []
    for root, _, files in os.walk(TL_DIR):
        for file in files:
            if file.endswith(".rpy"):
                file_path = os.path.join(root, file)
                rel = _normalize_rel_path(os.path.relpath(file_path, TL_DIR))
                if INCLUDE_FILES or INCLUDE_PREFIXES:
                    allowed = False
                    if INCLUDE_FILES and rel in INCLUDE_FILES:
                        allowed = True
                    if not allowed and INCLUDE_PREFIXES:
                        for prefix in INCLUDE_PREFIXES:
                            if rel.startswith(prefix):
                                allowed = True
                                break
                    if not allowed:
                        continue
                files_to_process.append(file_path)

    print(f"Found {len(files_to_process)} files.")

    # Load global progress
    global_progress = _upgrade_legacy_progress_keys(load_progress(), files_to_process)

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        progress_key = _progress_key_for_path(file_path)
        print(f"\nProcessing: {filename}")

        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        # Collect tasks
        raw_tasks = collect_tasks(lines)
        completed_entries = set(_normalize_progress_entries(global_progress.get(progress_key, [])))

        tasks = []
        for task in raw_tasks:
            if is_non_translatable(task["text"]):
                continue
            progress_entry = task.get("progress_entry") or _progress_entry_for_task(task)
            if progress_entry in completed_entries or _progress_line_entry(task["line"]) in completed_entries:
                if not FORCE_RETRANSLATE_ENGLISH:
                    continue
                if not is_english_like(task["text"]):
                    continue
            tasks.append(task)

        if not tasks:
            print("  No new lines to translate.")
            continue

        print(f"  Found {len(tasks)} lines to translate.")

        # Process in batches
        replacements = {}
        batch = []
        current_batch_chars = 0
        sync_rag_needs_file_refresh = False

        for task in tasks:
            # Update ID to be unique per file and string literal
            task["id"] = translation_core.build_identity_v2(
                progress_key,
                task.get("block_name", "_global"),
                task.get("block_index", 0),
                task.get("source_for_id") or task["text"],
                block_occurrence=task.get("block_occurrence", 1),
            )
            task["progress_entry"] = _progress_entry_for_task(task)
            task["file_rel_path"] = progress_key

            task_len = len(task["text"])

            if len(batch) >= MAX_ITEMS or (current_batch_chars + task_len > MAX_CHARS):
                successful_entries = process_batch_with_retry(batch, replacements)
                if successful_entries:
                    commit_replacements(file_path, lines, replacements)
                    update_progress(progress_key, successful_entries)
                    maybe_update_sync_rag_store(file_path, tasks=batch)
                    sync_rag_needs_file_refresh = True
                    completed_entries.update(_normalize_progress_entries(successful_entries))
                    global_progress[progress_key] = sorted(completed_entries)
                    replacements = {}

                batch = []
                current_batch_chars = 0

            batch.append(task)
            current_batch_chars += task_len

        # Final batch
        if batch:
            successful_entries = process_batch_with_retry(batch, replacements)
            if successful_entries:
                commit_replacements(file_path, lines, replacements)
                update_progress(progress_key, successful_entries)
                maybe_update_sync_rag_store(file_path, tasks=batch)
                sync_rag_needs_file_refresh = True
                completed_entries.update(_normalize_progress_entries(successful_entries))
                global_progress[progress_key] = sorted(completed_entries)

        if sync_rag_needs_file_refresh:
            maybe_update_sync_rag_store(file_path, full_file=True)

        print(f"  Done with {filename}.")
