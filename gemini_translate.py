# -*- coding: utf-8 -*-
import argparse
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
from typing import Any, cast

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
            import google.generativeai as imported_genai
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: google-generativeai. Install with `pip install -r requirements.txt`."
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
    """Configures the genai library with the current key."""
    genai = get_genai_module()
    cast(Any, genai).configure(api_key=get_current_api_key())


def create_model(model_name: str):
    genai = get_genai_module()
    return cast(Any, genai).GenerativeModel(model_name)

def get_random_delay():
    return random.uniform(MIN_DELAY, MAX_DELAY)

def quote_with(text, quote):
    escaped = text
    for old, new in SPECIAL_ESCAPES:
        escaped = escaped.replace(old, new)
    escaped = escaped.replace(quote, "\\\\" + quote)
    return f"{quote}{escaped}{quote}"

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
            return json.load(handle)
    except Exception:
        return {}

def save_progress(progress):
    try:
        with open(PROGRESS_LOG, "w", encoding="utf-8-sig") as handle:
            json.dump(progress, handle, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def update_progress(filename, translated_lines):
    progress = load_progress()
    if filename not in progress:
        progress[filename] = []
    progress[filename].extend(translated_lines)
    progress[filename] = list(set(progress[filename]))
    save_progress(progress)

def commit_replacements(path, lines, replacements):
    """Writes the replacements to the file."""
    if not replacements:
        return
    
    for line_idx, repls in replacements.items():
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        # Sort replacements by start position descending to avoid index shifting
        for start, end, translated, quote in sorted(repls, key=lambda x: x[0], reverse=True):
            # Safety check indices
            if start < 0 or end > len(line):
                continue
            normalized = apply_normalization(translated) if USE_TRANSLATION_MEMORY else translated
            line = line[:start] + quote_with(normalized, quote) + line[end:]
        lines[line_idx] = line
        
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)

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

def build_prompt(items):
    glossary = ", ".join(PRESERVE_TERMS)
    payload = json.dumps(
        [{"id": item["id"], "text": item["text"]} for item in items],
        ensure_ascii=False,
    )
    return (
        "You are translating a Ren'Py visual novel into Simplified Chinese (zh-CN).\n"
        "Rules:\n"
        f"1. Preserve these terms exactly (do not translate): {glossary}\n"
        "1.1 Keep all person names in English; do not translate names.\n"
        "2. Preserve Ren'Py tags like {i}, {/i}, {color=...}, [name], %s.\n"
        "3. Output plain Chinese text. No markdown, no Pinyin, no explanations.\n"
        "4. Return ONLY a JSON array of objects with 'id' and 'translation' keys.\n"
        "5. Example Input: [{\"id\":\"1\", \"text\":\"Hello\"}]\n"
        "6. Example Output: [{\"id\":\"1\", \"translation\":\"你好\"}]\n"
        f"Input JSON:\n{payload}"
    )

def call_gemini_sdk(prompt):
    """Calls Gemini using the official SDK."""
    configure_genai()
    model_name = get_current_model()
    
    try:
        model = create_model(model_name)
        
        # Generation config
        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 8192, # Increased for JSON safety
            "response_mime_type": "application/json" # Force JSON output if supported by model
        }
        
        # Disable all safety filters
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=cast(Any, generation_config),
            safety_settings=safety_settings
        )
        
        # Handle cases where response is blocked or empty
        try:
            return json.loads(response.text)
        except Exception:
            # Check if it was blocked
            if response.prompt_feedback:
                print(f"  [API Feedback] Prompt blocked: {response.prompt_feedback}")
            elif response.candidates and response.candidates[0].finish_reason != 1: # 1 is STOP
                print(f"  [API Feedback] Finish reason: {response.candidates[0].finish_reason}")
                # print(f"  Safety ratings: {response.candidates[0].safety_ratings}")
            
            raise ValueError(f"Invalid response from API. Text access failed.")

    except Exception as e:
        # Re-raise to be handled by retry logic
        raise e

def process_batch(batch, replacements):
    prompt = build_prompt(batch)
    
    # Call API (SDK handles connection details)
    results = call_gemini_sdk(prompt)
    
    if not isinstance(results, list):
        raise RuntimeError(f"API returned {type(results)} instead of list")
    
    id_map = {item["id"]: item for item in batch}
    valid_count = 0
    
    for item in results:
        entry = id_map.get(item.get("id"))
        if not entry:
            continue
            
        translated = item.get("translation", "")
        valid, msg = validate_translation(entry["text"], translated)
        
        if not valid:
            print(f"  Warning: Validation failed for {entry['id']}: {msg}")
            continue
            
        valid_count += 1
        line_idx = entry["line"]
        replacements.setdefault(line_idx, []).append(
            (entry["start"], entry["end"], translated, entry["quote"])
        )
        
    if valid_count == 0:
        raise RuntimeError("No valid translations in batch (all items rejected; consider expanding non-translatable rules or switching model)")
        
    # Calculate total chars to show valid data receipt without spoilers
    total_chars = sum(len(item.get("translation", "")) for item in results)
    print(f"  Translated {valid_count}/{len(batch)} items. (Received {total_chars} chars of translation)", flush=True)
    return True

def process_batch_with_retry(batch, replacements, retry_depth=0):
    if retry_depth >= 5:
        log_failure(batch, "Max retry depth reached")
        return False

    error_str = "" # Initialize variable to be safe
    
    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            # Respect rate limits
            time.sleep(get_random_delay())
            
            process_batch(batch, replacements)
            return True
            
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
        return r1 and r2
        
    log_failure(batch, f"Failed after retries: {error_str}")
    return False

def collect_tasks(lines, skip_translated=True):
    # Logic to parse Ren'Py files
    progress = load_progress()
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
        
        # Check for dialogue
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
            for i, token in enumerate(tokens):
                if token.type == tokenize.STRING:
                    try:
                        text_val = ast.literal_eval(token.string)
                    except:
                        continue
                        
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
                                "id": f"line_{idx}", # Temp ID, updated later
                                "text": text_val,
                                "line": idx,
                                "start": token.start[1],
                                "end": token.end[1],
                                "quote": token.string[0]
                            })
        except:
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
    global_progress = load_progress()

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        # Check if fully done (optional optimization, skip for now to be safe)
        
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            
        # Collect tasks
        # We need to filter tasks that are already in global_progress[filename]
        raw_tasks = collect_tasks(lines)
        completed_lines = set(global_progress.get(filename, []))
        
        tasks = []
        for task in raw_tasks:
            if is_non_translatable(task["text"]):
                continue
            if task["line"] in completed_lines:
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
        translated_line_indices = []
        
        batch = []
        current_batch_chars = 0
        
        for task in tasks:
            # Update ID to be unique per file
            task["id"] = f"{filename}:{task['line']}"
            
            task_len = len(task["text"])
            
            if len(batch) >= MAX_ITEMS or (current_batch_chars + task_len > MAX_CHARS):
                # Process
                if process_batch_with_retry(batch, replacements):
                    # Success
                    batch_indices = [t["line"] for t in batch]
                    translated_line_indices.extend(batch_indices)
                    # Commit to file every batch to be safe
                    commit_replacements(file_path, lines, replacements)
                    update_progress(filename, batch_indices)
                    replacements = {} # Clear after commit
                
                batch = []
                current_batch_chars = 0
            
            batch.append(task)
            current_batch_chars += task_len
            
        # Final batch
        if batch:
            if process_batch_with_retry(batch, replacements):
                batch_indices = [t["line"] for t in batch]
                translated_line_indices.extend(batch_indices)
                commit_replacements(file_path, lines, replacements)
                update_progress(filename, batch_indices)

        print(f"  Done with {filename}.")


def build_arg_parser():
    return argparse.ArgumentParser(
        description="Synchronous translator for Ren'Py tl files using the Gemini SDK."
    )


def main(argv=None):
    parser = build_arg_parser()
    parser.parse_args(argv)
    initialize_runtime_logging()
    run_translation()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
