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
from datetime import datetime

from rag_memory import JsonRagStore, hash_text, truncate_text

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
SOURCE_GAME_DIR = ""
PREP_ENABLED = True
PREP_UNPACK_RPA = True
PREP_GENERATE_TEMPLATE = True
PREP_LANGUAGE = "schinese"
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
MAX_CHARS = 3000  # Reduced slightly to be safer
MAX_ITEMS = 20    # Back to a safer batch size (50 was too aggressive)
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

# Optional allowlist to limit which files are processed (relative to TL_DIR).
INCLUDE_FILES = set()
INCLUDE_PREFIXES = set()

NORMALIZE_TRANSLATION_MAP = {}

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

NON_TRANSLATABLE_EXACT = {
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
    "DejaVu Sans",
    "Opendyslexic",
}

NON_TRANSLATABLE_TAG_ONLY = re.compile(r"^\{[^}]+\}$")
NON_TRANSLATABLE_SYMBOLS = re.compile(r"^[^A-Za-z0-9\u4e00-\u9fff]+$")
RENPLY_TAG_RE = re.compile(r"\{[^}]*\}")
RENPLY_FIELD_RE = re.compile(r"\[[^\]]+\]")
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+")
VOWEL_RE = re.compile(r"[aeiou]", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(.)\\1{2,}")
STUTTER_PATTERN = re.compile(r"\b\w-\w", re.IGNORECASE)
MULTI_DOT_PATTERN = re.compile(r"(\.{2,}|…{2,})")
# Matches sequences like "A B C" or "A. B. C." (single-letter tokens only)
LETTER_SEQUENCE_RE = re.compile(r"^(?:[A-Za-z]\.?)(?:\s+[A-Za-z]\.?)+$")
FILE_NAME_SIMPLE_RE = re.compile(r"^[\w.-]+\.\w+$", re.IGNORECASE)
STRING_LITERAL_PREFIX_RE = re.compile(r"(?is)^(?P<prefix>[rubf]*)(?P<quote>'''|\"\"\"|'|\")")
TL_COMMENT_SOURCE_RE = re.compile(r'^\s*#\s*(?P<prefix>[^\"]*?)"(?P<text>.*)"\s*$')
TL_OLD_LINE_RE = re.compile(r'^\s*old\s+"(?P<text>.*)"\s*$')
TL_NEW_LINE_RE = re.compile(r'^\s*new\s+"(?P<text>.*)"\s*$')
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


def _resolve_path(base_dir, value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return os.path.abspath(text)
    return os.path.abspath(os.path.join(base_dir, text))


def _resolve_preferred_path(primary_base_dir, secondary_base_dir, value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return os.path.abspath(text)

    candidates = []
    for base_dir in (primary_base_dir, secondary_base_dir):
        if not base_dir:
            continue
        candidate = os.path.abspath(os.path.join(base_dir, text))
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


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
        NON_TRANSLATABLE_EXACT = set(non_translatable)
        print(f"Loaded {len(NON_TRANSLATABLE_EXACT)} non-translatable exact terms.")
    elif "non_translatable_exact" in data:
        print("Warning: glossary.json non_translatable_exact is empty; using defaults.")

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


def load_translator_settings():
    """Loads per-game settings (game root, tl subdir) from translator_config.json or env."""
    global BASE_DIR, TL_DIR, TL_SUBDIR, ENV_GAME_ROOT, WORK_GAME_DIR, SOURCE_GAME_DIR, GLOSSARY_FILE
    global PREP_ENABLED, PREP_UNPACK_RPA, PREP_GENERATE_TEMPLATE, PREP_LANGUAGE
    global PREP_LAUNCHER_PY, PREP_PYTHON_EXE, PREP_UNPACK_COMMAND, PREP_TEMPLATE_COMMAND

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
        BASE_DIR = os.path.abspath(ENV_GAME_ROOT)
    else:
        BASE_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

    glossary_file = config.get("glossary_file")
    if glossary_file is None:
        glossary_file = config.get("glossary_path")
    if glossary_file is not None:
        resolved_glossary = _resolve_preferred_path(TOOL_DIR, BASE_DIR, glossary_file)
        GLOSSARY_FILE = resolved_glossary or DEFAULT_GLOSSARY_FILE
    else:
        GLOSSARY_FILE = DEFAULT_GLOSSARY_FILE

    tl_subdir = config.get("tl_subdir")
    if isinstance(tl_subdir, str) and tl_subdir.strip():
        TL_SUBDIR = _normalize_rel_path(tl_subdir)

    TL_DIR = os.path.abspath(os.path.join(BASE_DIR, TL_SUBDIR))
    WORK_GAME_DIR = os.path.abspath(os.path.join(BASE_DIR, WORK_GAME_SUBDIR))

    prepare = config.get("prepare")
    if not isinstance(prepare, dict):
        prepare = {}

    PREP_ENABLED = _coerce_bool(prepare.get("enabled"), True)
    PREP_UNPACK_RPA = _coerce_bool(prepare.get("unpack_rpa"), True)
    PREP_GENERATE_TEMPLATE = _coerce_bool(prepare.get("generate_template"), True)

    prep_language = prepare.get("language")
    if isinstance(prep_language, str) and prep_language.strip():
        PREP_LANGUAGE = prep_language.strip()

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
    load_sync_rag_settings(config)


def load_config():
    """Loads API keys and settings from api_keys.json or environment."""
    global API_KEYS, MODELS, MAX_CHARS, MAX_ITEMS, INCLUDE_FILES, INCLUDE_PREFIXES
    
    # Try loading from JSON
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
                keys = config.get("api_keys", [])
                custom_models = config.get("models", [])
                single_model = config.get("model")
                batch_size = config.get("batch_size")
                max_chars = config.get("max_chars")
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
                    MODELS = [str(m) for m in custom_models if m]
                    print(f"Using custom model list: {MODELS}")

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

                if include_files:
                    if isinstance(include_files, str):
                        include_files = [include_files]
                    INCLUDE_FILES = {_normalize_rel_path(p) for p in include_files if _normalize_rel_path(p)}
                    print(f"Using include_files allowlist ({len(INCLUDE_FILES)}).")

                if include_prefixes:
                    if isinstance(include_prefixes, str):
                        include_prefixes = [include_prefixes]
                    INCLUDE_PREFIXES = {_normalize_rel_path(p) for p in include_prefixes if _normalize_rel_path(p)}
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

    if not API_KEYS:
        print("="*60)
        print("ERROR: No valid API keys found!")
        print("Please check api_keys.json or set GEMINI_API_KEY env vars.")
        print("="*60)
        raise SystemExit("No API keys available")


SCRIPT_FILE_EXTENSIONS = {".rpy", ".rpym", ".rpyc", ".rpymc"}


def _has_files_with_extensions(base_dir, extensions):
    if not os.path.isdir(base_dir):
        return False
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
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


def _guess_source_game_dir():
    candidates = []
    if SOURCE_GAME_DIR:
        candidates.append(SOURCE_GAME_DIR)
    candidates.append(WORK_GAME_DIR)

    if os.path.basename(BASE_DIR).lower() == "work":
        candidates.append(os.path.abspath(os.path.join(BASE_DIR, "..", "original", "game")))

    seen = set()
    ordered = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    for candidate in ordered:
        if not os.path.isdir(candidate):
            continue
        if _has_files_with_extensions(candidate, SCRIPT_FILE_EXTENSIONS):
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


def _load_pickle_blob(blob):
    return pickle.loads(blob, encoding="bytes")


def _read_rpa_index(archive_path):
    with open(archive_path, "rb") as infile:
        header = infile.read(40)

        if header.startswith(b"RPA-3.0 "):
            offset = int(header[8:24], 16)
            key = int(header[25:33], 16)
            infile.seek(offset)
            raw_index = _load_pickle_blob(zlib.decompress(infile.read()))

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
            raw_index = _load_pickle_blob(zlib.decompress(infile.read()))

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


def _resolve_prepare_python():
    if PREP_PYTHON_EXE and os.path.isfile(PREP_PYTHON_EXE):
        return PREP_PYTHON_EXE

    bundled = os.path.join(BASE_DIR, "lib", "py3-windows-x86_64", "python.exe")
    if os.path.isfile(bundled):
        return bundled

    for candidate in glob.glob(os.path.join(BASE_DIR, "lib", "py*", "python.exe")):
        if os.path.isfile(candidate):
            return candidate

    return sys.executable


def _resolve_prepare_launcher():
    if PREP_LAUNCHER_PY and os.path.isfile(PREP_LAUNCHER_PY):
        return PREP_LAUNCHER_PY

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

    return py_files[0]


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


def run_prepare_steps():
    if not PREP_ENABLED:
        print("[Prepare] Disabled by translator_config.")
        return

    source_game_dir = _guess_source_game_dir()
    os.makedirs(WORK_GAME_DIR, exist_ok=True)
    print(f"[Prepare] Source game dir: {source_game_dir}")
    print(f"[Prepare] Work game dir: {WORK_GAME_DIR}")

    copied_scripts = _copy_script_sources(source_game_dir, WORK_GAME_DIR)
    if copied_scripts:
        print(f"[Prepare] Copied {copied_scripts} script files into work/game.")

    if PREP_UNPACK_RPA:
        has_scripts = _has_files_with_extensions(WORK_GAME_DIR, SCRIPT_FILE_EXTENSIONS)
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
        if _has_translation_templates():
            print("[Prepare] Translation template already exists; skipping generation.")
        else:
            launcher_py = _resolve_prepare_launcher()
            python_exe = _resolve_prepare_python()
            if PREP_TEMPLATE_COMMAND:
                variables = {
                    "python_exe": python_exe,
                    "launcher_py": launcher_py,
                    "language": PREP_LANGUAGE,
                    "base_dir": BASE_DIR,
                    "tl_dir": TL_DIR,
                    "work_game_dir": WORK_GAME_DIR,
                    "source_game_dir": source_game_dir,
                }
                try:
                    rendered, _ = _render_prepare_command(PREP_TEMPLATE_COMMAND, variables)
                    _run_prepare_command(rendered, BASE_DIR, "Generate tl template")
                except Exception as e:
                    print(f"[Prepare] Template command error: {e}")
            elif launcher_py:
                command = [python_exe, launcher_py, "translate", PREP_LANGUAGE]
                _run_prepare_command(command, BASE_DIR, "Generate tl template")
            else:
                print("[Prepare] Could not locate <Game>.py launcher; skip template generation.")
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


def guess_project_slug():
    base_name = os.path.basename(os.path.abspath(BASE_DIR))
    if base_name.lower() == "work":
        parent = os.path.basename(os.path.dirname(os.path.abspath(BASE_DIR)))
        return _slugify(parent or base_name)
    return _slugify(base_name)


def get_default_sync_rag_store_dir():
    return os.path.join(LOG_DIR, "rag_store", guess_project_slug())


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
    if not hits:
        return empty_label
    lines = []
    for hit in hits:
        source = hit.get("source", "")
        target = hit.get("target", "")
        if not source:
            continue
        if source == target:
            lines.append(f"- Keep unchanged: {source}")
        else:
            lines.append(f"- {source} -> {target}")
    return "\n".join(lines) if lines else empty_label


def format_sync_history_hits_block(hits, empty_label="(none)"):
    if not hits:
        return empty_label
    lines = []
    for hit in hits:
        file_rel_path = hit.get("file_rel_path", "")
        line_start = hit.get("line_start", "")
        line_end = hit.get("line_end", "")
        score = hit.get("score", 0.0)
        quality = hit.get("quality_state", "")
        translated_text = hit.get("translated_text", "") or hit.get("source_text", "")
        translated_text = truncate_text(translated_text, SYNC_RAG_HISTORY_CHAR_LIMIT)
        lines.append(
            f"- [{file_rel_path}:{line_start}-{line_end} score={score:.3f} quality={quality}] {translated_text}"
        )
    return "\n".join(lines) if lines else empty_label


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
    for old, new in SPECIAL_ESCAPES:
        escaped = escaped.replace(old, new)
    quote_char = (quote or '"')[0]
    escaped = escaped.replace(quote_char, "\\" + quote_char)
    return f"{prefix}{quote}{escaped}{quote}"


def contains_chinese(text):
    if not text:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

def missing_preserved_terms(original, translated):
    if not original or not translated:
        return []
    missing = []
    for term in PRESERVE_TERMS:
        if not term or term not in original:
            continue
        # Avoid false positives for short alphabetic fragments (e.g., "Mo" in "Moon").
        if term.isalpha() and len(term) <= 3:
            pattern = rf"\b{re.escape(term)}\b"
            if not re.search(pattern, original):
                continue
            if not re.search(pattern, translated):
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
    cleaned = RENPLY_TAG_RE.sub("", text or "")
    cleaned = RENPLY_FIELD_RE.sub("", cleaned)
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
    cleaned = RENPLY_TAG_RE.sub("", text or "")
    cleaned = RENPLY_FIELD_RE.sub("", cleaned)
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
    cleaned = RENPLY_TAG_RE.sub("", text or "")
    cleaned = RENPLY_FIELD_RE.sub("", cleaned)
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
    cleaned = RENPLY_TAG_RE.sub("", text)
    cleaned = RENPLY_FIELD_RE.sub("", cleaned)
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
    cleaned = RENPLY_TAG_RE.sub("", text or "")
    cleaned = RENPLY_FIELD_RE.sub("", cleaned)
    return [token.lower() for token in WORD_TOKEN_RE.findall(cleaned)]


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
    if known_terms:
        allowed_terms.update(
            str(term).strip().lower()
            for term in known_terms
            if isinstance(term, str) and str(term).strip()
        )
    if not allowed_terms:
        return False
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
        with open(PROGRESS_LOG, "w", encoding="utf-8-sig") as handle:
            json.dump(progress, handle, ensure_ascii=False, indent=2)
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
    """Writes the replacements to the file."""
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
                start, end, translated, prefix, quote = repl
            # Safety check indices
            if start < 0 or end > len(line):
                continue
            normalized = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
            line = line[:start] + quote_with(normalized, quote, prefix=prefix) + line[end:]
        lines[line_idx] = line

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def sync_rag_hash_key(text):
    return hash_text(text)[:10]


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


def collect_translation_entries_from_lines(lines):
    entries = []
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip("\n")
        comment_match = TL_COMMENT_SOURCE_RE.match(raw_line)
        if comment_match:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines):
                token = extract_string_token_from_line(lines[next_index])
                if token:
                    entries.append(
                        {
                            "line_number": next_index + 1,
                            "source": comment_match.group("text"),
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
                                "source": old_match.group("text"),
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


def build_sync_rag_record(file_rel_path, group, quality_state):
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
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checksum": hash_text(source_text),
        "translation_checksum": hash_text(translated_text),
    }


def collect_sync_rag_records_for_file(file_path, quality_state=SYNC_RAG_QUALITY_STATE):
    if not file_path or not os.path.isfile(file_path):
        return []
    try:
        file_rel_path = _normalize_rel_path(os.path.relpath(file_path, TL_DIR))
    except ValueError:
        file_rel_path = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8-sig") as handle:
        entries = collect_translation_entries_from_lines(handle.readlines())

    records = []
    segment_size = max(1, SYNC_RAG_SEGMENT_LINES)
    usable_entries = [entry for entry in entries if should_index_sync_rag_entry(entry)]
    for start in range(0, len(usable_entries), segment_size):
        group = usable_entries[start:start + segment_size]
        if group:
            records.append(build_sync_rag_record(file_rel_path, group, quality_state))
    return records


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


def sync_rag_store_for_file(file_path, quality_state=SYNC_RAG_QUALITY_STATE):
    if not SYNC_RAG_ENABLED or not SYNC_RAG_UPDATE_ON_SUCCESS:
        return {"enabled": False}
    store = get_sync_rag_store()
    if store is None:
        return {"enabled": True, "error": "RAG store unavailable"}

    base_records = collect_sync_rag_records_for_file(file_path, quality_state=quality_state)
    pending_records = []
    for record in base_records:
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
        "enabled": True,
        "store_dir": store.store_dir,
        "scanned": len(base_records),
        "pending": len(pending_records),
        "upserted": 0,
        "history_records_before": store.count_history(),
    }
    if not pending_records:
        stats["history_records_after"] = store.count_history()
        return stats

    try:
        embedded_records = embed_sync_history_records(pending_records)
        stats["upserted"] = store.upsert_history(embedded_records)
        stats["history_records_after"] = store.count_history()
    except Exception as exc:
        print(f"Warning: Failed to update sync RAG store: {exc}")
        stats["error"] = str(exc)
        stats["history_records_after"] = store.count_history()
    return stats


def maybe_update_sync_rag_store(file_path):
    if not SYNC_RAG_ENABLED or not SYNC_RAG_UPDATE_ON_SUCCESS:
        return
    summary = sync_rag_store_for_file(file_path, quality_state=SYNC_RAG_QUALITY_STATE)
    if summary.get("upserted"):
        print(f"  Sync RAG store updated: {summary.get('upserted', 0)} entries", flush=True)
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

def build_prompt(items, glossary_hits=None, history_hits=None):
    glossary = ", ".join(PRESERVE_TERMS)
    payload = json.dumps(
        [{"id": item["id"], "text": item["text"]} for item in items],
        ensure_ascii=False,
    )
    reference_blocks = ""
    if SYNC_RAG_ENABLED:
        glossary_hits = glossary_hits or []
        history_hits = history_hits or []
        reference_blocks = (
            "\nReference blocks:\n"
            f"LOCKED TERMS:\n{format_sync_glossary_hits_block(glossary_hits, '(none)')}\n\n"
            f"RETRIEVED MEMORY:\n{format_sync_history_hits_block(history_hits, '(none)')}\n"
            "Use retrieved memory only as style and terminology reference; ignore it when unrelated.\n"
        )
    return (
        "You are translating a Ren'Py visual novel into Simplified Chinese (zh-CN).\n"
        "Rules:\n"
        f"1. Preserve these terms exactly (do not translate): {glossary}\n"
        "1.1 Keep all person names in English; do not translate names.\n"
        "2. Preserve Ren'Py tags like {i}, {/i}, {color=...}, [name], %s.\n"
        "3. Output plain Chinese text. No markdown, no Pinyin, no explanations.\n"
        "4. Return ONLY a JSON array matching the requested id/translation structure.\n"
        f"{reference_blocks}"
        f"Input JSON:\n{payload}"
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
    target_ids = [item["id"] for item in items]
    return {
        "type": "array",
        "minItems": len(items),
        "maxItems": len(items),
        "items": {
            "type": "object",
            "required": ["id", "translation"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "enum": target_ids},
                "translation": {"type": "string"},
            },
        },
    }


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
    data = payload
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            data = data["items"]
        elif isinstance(data.get("translations"), list):
            data = data["translations"]

    if not isinstance(data, list):
        raise ValueError(f"Response JSON is not a list: {type(data)}")

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        translation = item.get("translation")
        if item_id is None or translation is None:
            continue
        normalized.append({"id": str(item_id), "translation": str(translation)})
    return normalized


def call_gemini_sdk(prompt, items):
    """Calls Gemini using the current google-genai SDK."""
    configure_genai()
    model_name = get_current_model()
    
    try:
        client = create_genai_client()
        
        # Generation config
        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_json_schema": build_response_json_schema(items),
        }

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config,
        )

        parsed = get_nested(response, "parsed")
        if parsed is not None:
            return normalize_result_items(serialize_unknown(parsed))

        response_payload = serialize_unknown(response)
        response_text = extract_text_from_response_payload(response_payload)
        if response_text:
            return normalize_result_items(parse_json_payload(response_text))

        prompt_feedback = extract_prompt_feedback(response_payload)
        finish_reason = extract_finish_reason(response_payload)
        diagnostics = []
        if prompt_feedback:
            diagnostics.append(f"Prompt feedback: {prompt_feedback}")
        if finish_reason:
            diagnostics.append(f"Finish reason: {finish_reason}")
        detail = f" ({'; '.join(diagnostics)})" if diagnostics else ""
        raise ValueError(f"Invalid response from API. Missing structured text{detail}.")

    except Exception as e:
        # Re-raise to be handled by retry logic
        raise e

def process_batch(batch, replacements):
    glossary_hits = retrieve_sync_glossary_hits(batch) if SYNC_RAG_ENABLED else []
    history_hits, rag_stats = retrieve_sync_history_hits(batch) if SYNC_RAG_ENABLED else ([], {})
    if rag_stats.get("hit_count"):
        print(f"  Sync RAG memory hits: {rag_stats['hit_count']}", flush=True)
    prompt = build_prompt(batch, glossary_hits=glossary_hits, history_hits=history_hits)

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
        valid, msg = validate_translation(entry["text"], translated)

        if not valid:
            print(f"  Warning: Validation failed for {entry['id']}: {msg}")
            continue

        valid_progress_entries.append(entry["progress_entry"])
        line_idx = entry["line"]
        replacements.setdefault(line_idx, []).append(
            (entry["start"], entry["end"], translated, entry.get("prefix", ""), entry["quote"])
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

def collect_tasks(lines, skip_translated=True):
    # Logic to parse Ren'Py files
    # Note: caller handles filename lookup, this function just parses

    tasks = []
    # Detect Ren'Py translation files so we can protect `old` entries.
    is_translation_file = any(
        line.lstrip().startswith("translate ")
        for line in lines
    )

    # Simple parser for Ren'Py strings
    for idx, line in enumerate(lines):
        sline = line.strip()
        # In translation templates, `old` is a lookup key and must never be edited.
        if (
            not sline
            or sline.startswith("#")
            or sline.startswith("translate ")
            or (is_translation_file and sline.startswith("old "))
        ):
            continue

        # Very basic string extraction (robust enough for this task)
        # Look for dialogue lines: Character "Text" or "Text"
        # And strings: old "Text"

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
            for token in tokens:
                if token.type != tokenize.STRING:
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
                if text_val and not contains_chinese(text_val) and len(text_val) > 1:
                    if is_non_translatable(text_val):
                        continue
                    # Skip if it's likely a bare file path (no spaces)
                    if (" " not in text_val) and ("/" in text_val or "\\" in text_val):
                        continue

                    if (" " in text_val or len(text_val) > 15 or (text_val and text_val[0].isupper())
                            or (ALLOW_SINGLE_WORD_TRANSLATION and is_english_like(text_val))):
                        tasks.append({
                            "id": f"line_{idx}_{token.start[1]}", # Temp ID, updated later
                            "text": text_val,
                            "line": idx,
                            "start": token.start[1],
                            "end": token.end[1],
                            "quote": quote,
                            "prefix": prefix,
                            "progress_entry": f"task:{idx}:{token.start[1]}",
                        })
        except Exception:
            continue

    return tasks

def run_translation():
    load_config()
    load_translator_settings()
    load_glossary()
    print("="*60)
    print("Gemini Translator (Ren'Py)")
    print(f"Models: {MODELS}")
    print(f"API Keys Loaded: {len(API_KEYS)}")
    print(f"Base dir: {BASE_DIR}")
    print(f"TL dir: {TL_DIR} (exists: {os.path.isdir(TL_DIR)})")
    print(f"Progress log: {PROGRESS_LOG}")
    print(f"Translator config: {TRANSLATOR_CONFIG} (exists: {os.path.isfile(TRANSLATOR_CONFIG)})")
    print(f"Glossary: {GLOSSARY_FILE} (exists: {os.path.isfile(GLOSSARY_FILE)})")
    print(f"Prepare enabled: {PREP_ENABLED}")
    print(f"Prepare source game dir: {SOURCE_GAME_DIR or '(auto)'}")
    print(f"Prepare language: {PREP_LANGUAGE}")
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

        for task in tasks:
            # Update ID to be unique per file and string literal
            task["id"] = f"{progress_key}:{task['line']}:{task['start']}"
            task["progress_entry"] = _progress_entry_for_task(task)

            task_len = len(task["text"])

            if len(batch) >= MAX_ITEMS or (current_batch_chars + task_len > MAX_CHARS):
                successful_entries = process_batch_with_retry(batch, replacements)
                if successful_entries:
                    commit_replacements(file_path, lines, replacements)
                    update_progress(progress_key, successful_entries)
                    maybe_update_sync_rag_store(file_path)
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
                maybe_update_sync_rag_store(file_path)
                completed_entries.update(_normalize_progress_entries(successful_entries))
                global_progress[progress_key] = sorted(completed_entries)

        print(f"  Done with {filename}.")
