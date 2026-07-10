# -*- coding: utf-8 -*-
import argparse
import ast
import copy
import hashlib
import io
import json
import os
import re
import sys
import time
import tokenize
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rag_memory import JsonRagStore, JsonSourceIndexStore, JsonSourceIndexStoreLockError, hash_text, truncate_text
import batch_cost_estimate
import batch_non_chinese_rules
import batch_submit_recovery
import doctor_recommendations as doctor_rec
import keyword_glossary_merge
import prompt_context
import translation_ab_experiment
import story_memory
import translation_core
import translator_runtime as runtime

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

legacy = runtime

LOG_DIR = legacy.LOG_DIR
FAILED_LOG = os.path.join(LOG_DIR, 'translation_failures_batch.jsonl')
PROGRESS_LOG = os.path.join(LOG_DIR, 'translation_progress_batch.json')
CONSOLE_LOG = os.path.join(LOG_DIR, 'translation_batch_console_output.log')
BATCH_JOBS_DIR = os.path.join(LOG_DIR, 'batch_jobs')
LATEST_MANIFEST_FILE = os.path.join(BATCH_JOBS_DIR, 'latest_manifest.txt')
REPAIR_RUNS_DIR = os.path.join(LOG_DIR, 'repair_runs')
SYNC_RUNS_DIR = os.path.join(LOG_DIR, 'sync_runs')


class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def ensure_batch_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BATCH_JOBS_DIR, exist_ok=True)
    os.makedirs(REPAIR_RUNS_DIR, exist_ok=True)
    os.makedirs(SYNC_RUNS_DIR, exist_ok=True)


def initialize_batch_logging():
    if isinstance(sys.stdout, DualLogger):
        return
    ensure_batch_dirs()
    try:
        sys.stdout = DualLogger(CONSOLE_LOG)
    except OSError as exc:
        print(f'Warning: Could not open console log {CONSOLE_LOG}: {exc}')

BATCH_MODEL = 'gemini-3.1-flash-lite'
BATCH_TARGET_SIZE = 60
BATCH_TARGET_CHARS = 18000
BATCH_RETRY_TARGET_SIZE = 8
BATCH_RETRY_TARGET_CHARS = 4000
BATCH_CONTEXT_BEFORE = 30
BATCH_CONTEXT_AFTER = 10
BATCH_MAX_OUTPUT_TOKENS = 32768
BATCH_TEMPERATURE = 0.2
BATCH_THINKING_LEVEL = 'minimal'
BATCH_SAFETY_SETTINGS = []
BATCH_DISPLAY_NAME_PREFIX = 'renpy-translate'
BATCH_SPLIT_RECOMMEND_CHUNKS = 400
BATCH_SPLIT_RECOMMEND_ITEMS = 12000
BATCH_MACRO_SETTING = ''
BATCH_NON_CHINESE_RULES = batch_non_chinese_rules.normalize_non_chinese_rules(None)
MANIFEST_MODE_TRANSLATION = 'translation'
MANIFEST_MODE_KEYWORD_EXTRACTION = 'keyword_extraction'
MANIFEST_MODE_REVISION = 'revision'
CHECK_CONTRACT_VERSION = 1
CHECK_SAFETY_SAFE = 'safe'
CHECK_SAFETY_WARN = 'warn'
CHECK_SAFETY_BLOCK = 'block'
KEYWORD_DISPLAY_NAME_PREFIX = 'renpy-keywords'
KEYWORD_CHUNK_SIZE = 40
KEYWORD_MAX_CANDIDATES_PER_CHUNK = 12
REVISION_DISPLAY_NAME_PREFIX = 'renpy-revise'
REVISION_CHUNK_SIZE = 6

CHECK_WARN_REASON_CODES = {
    'partial_result_items',
    'response_missing_item_id',
    'schema_or_item_mismatch',
    'validation_failed',
    'missing_chunk_rows',
}
CHECK_BLOCK_REASON_CODES = {
    'invalid_result_jsonl_row',
    'unknown_chunk_key',
    'row_error',
    'missing_response_text',
    'failed_to_parse_model_json',
    'truncated_output',
    'duplicate_result_id',
    'source_line_missing',
    'source_text_mismatch',
    'missing_manifest_file',
    'target_file_missing',
    'target_file_path_escaped',
    'v2_relocation_missing',
}

RAG_ENABLED = False
RAG_STORE_DIR = ''
RAG_EMBEDDING_MODEL = 'gemini-embedding-001'
RAG_QUERY_TASK_TYPE = 'RETRIEVAL_QUERY'
RAG_DOCUMENT_TASK_TYPE = 'RETRIEVAL_DOCUMENT'
RAG_OUTPUT_DIMENSIONALITY = 768
RAG_TOP_K_HISTORY = 4
RAG_TOP_K_TERMS = 8
RAG_MIN_SIMILARITY = 0.72
RAG_SEGMENT_LINES = 4
RAG_BOOTSTRAP_ON_BUILD = True
RAG_HISTORY_CHAR_LIMIT = 220
_RAG_STORE = None
_RAG_PRESERVED_TERMS_CACHE = None
_RAG_PRESERVED_TERMS_CACHE_KEY = None

SOURCE_INDEX_ENABLED = False
SOURCE_INDEX_STORE_DIR = ''
_SOURCE_INDEX_STORE = None
SOURCE_INDEX_SCHEMA_VERSION = 1
SOURCE_INDEX_TOP_K = 4
SOURCE_INDEX_MIN_SIMILARITY = 0.72
SOURCE_INDEX_CHAR_LIMIT = 220

STORY_MEMORY_ENABLED = False
STORY_MEMORY_GRAPH_FILE = ''
STORY_MEMORY_MAX_CONTEXT_CHARS = 1200
STORY_MEMORY_TOP_K_RELATIONS = 6
STORY_MEMORY_TOP_K_TERMS = 12
STORY_MEMORY_INCLUDE_SCENE_SUMMARY = True
_STORY_GRAPH = None
_STORY_GRAPH_PATH = ''


def load_json_file(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8-sig') as handle:
            return json.load(handle) or {}
    except Exception as exc:
        print(f'Warning: Failed to load JSON {path}: {exc}')
        return {}


def coerce_positive_int(value, default):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'yes', 'on'}:
            return True
        if lowered in {'0', 'false', 'no', 'off'}:
            return False
    return default


def coerce_non_empty_string(value, default):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def coerce_thinking_level(value, default):
    if value is None or value is False:
        return ''
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {'none', 'off', 'disabled', 'false', '0'}:
            return ''
        return stripped
    return default


def format_thinking_level_for_display():
    return BATCH_THINKING_LEVEL or '(not sent)'


def read_text_file(path):
    if not path or not os.path.isfile(path):
        return ''
    with open(path, 'r', encoding='utf-8-sig') as handle:
        return handle.read().strip()


def normalize_task_type(value, default):
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned:
            return cleaned
    return default


def normalize_batch_safety_settings(value):
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        normalized = cleaned.lower().replace('-', '_')
        if normalized in {'relaxed_adult', 'adult', 'sexually_explicit_block_none'}:
            return [
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                }
            ]
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    settings = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        category = str(entry.get('category') or entry.get('harm_category') or '').strip().upper()
        threshold = str(entry.get('threshold') or entry.get('block_threshold') or '').strip().upper()
        if not category or not threshold:
            continue
        if not category.startswith('HARM_CATEGORY_'):
            category = f'HARM_CATEGORY_{category}'
        if threshold in {'NONE', 'NO_BLOCK', 'BLOCKNONE'}:
            threshold = 'BLOCK_NONE'
        settings.append({'category': category, 'threshold': threshold})
    return settings


def load_batch_settings():
    global BATCH_MODEL, BATCH_TARGET_SIZE, BATCH_CONTEXT_BEFORE, BATCH_CONTEXT_AFTER
    global BATCH_TARGET_CHARS, BATCH_RETRY_TARGET_SIZE, BATCH_RETRY_TARGET_CHARS
    global BATCH_MAX_OUTPUT_TOKENS, BATCH_TEMPERATURE, BATCH_THINKING_LEVEL
    global BATCH_SAFETY_SETTINGS, BATCH_DISPLAY_NAME_PREFIX, BATCH_MACRO_SETTING
    global KEYWORD_DISPLAY_NAME_PREFIX, KEYWORD_CHUNK_SIZE, KEYWORD_MAX_CANDIDATES_PER_CHUNK
    global REVISION_DISPLAY_NAME_PREFIX, REVISION_CHUNK_SIZE
    global RAG_ENABLED, RAG_STORE_DIR, RAG_EMBEDDING_MODEL, RAG_QUERY_TASK_TYPE
    global RAG_DOCUMENT_TASK_TYPE, RAG_OUTPUT_DIMENSIONALITY, RAG_TOP_K_HISTORY
    global RAG_TOP_K_TERMS, RAG_MIN_SIMILARITY, RAG_SEGMENT_LINES
    global RAG_BOOTSTRAP_ON_BUILD, RAG_HISTORY_CHAR_LIMIT, _RAG_STORE
    global SOURCE_INDEX_ENABLED, SOURCE_INDEX_STORE_DIR, _SOURCE_INDEX_STORE
    global SOURCE_INDEX_TOP_K, SOURCE_INDEX_MIN_SIMILARITY, SOURCE_INDEX_CHAR_LIMIT
    global STORY_MEMORY_ENABLED, STORY_MEMORY_GRAPH_FILE, STORY_MEMORY_MAX_CONTEXT_CHARS
    global STORY_MEMORY_TOP_K_RELATIONS, STORY_MEMORY_TOP_K_TERMS
    global STORY_MEMORY_INCLUDE_SCENE_SUMMARY, _STORY_GRAPH, _STORY_GRAPH_PATH
    global BATCH_NON_CHINESE_RULES

    config = load_json_file(legacy.CONFIG_FILE)
    translator_config = load_json_file(legacy.TRANSLATOR_CONFIG)

    batch_model = config.get('batch_model')
    if isinstance(batch_model, str) and batch_model.strip():
        BATCH_MODEL = batch_model.strip()

    BATCH_TARGET_SIZE = coerce_positive_int(
        config.get('batch_target_size', config.get('batch_size')),
        BATCH_TARGET_SIZE,
    )
    BATCH_TARGET_CHARS = coerce_positive_int(
        config.get('batch_target_chars', config.get('batch_max_source_chars')),
        BATCH_TARGET_CHARS,
    )
    BATCH_CONTEXT_BEFORE = coerce_positive_int(config.get('batch_context_before'), BATCH_CONTEXT_BEFORE)
    BATCH_CONTEXT_AFTER = coerce_positive_int(config.get('batch_context_after'), BATCH_CONTEXT_AFTER)
    BATCH_MAX_OUTPUT_TOKENS = coerce_positive_int(
        config.get('batch_max_output_tokens'),
        BATCH_MAX_OUTPUT_TOKENS,
    )
    if 'batch_thinking_level' in config:
        BATCH_THINKING_LEVEL = coerce_thinking_level(
            config.get('batch_thinking_level'),
            BATCH_THINKING_LEVEL,
        )
    if 'batch_safety_settings' in config:
        BATCH_SAFETY_SETTINGS = normalize_batch_safety_settings(config.get('batch_safety_settings'))

    display_name_prefix = config.get('batch_display_name_prefix')
    if isinstance(display_name_prefix, str) and display_name_prefix.strip():
        BATCH_DISPLAY_NAME_PREFIX = display_name_prefix.strip()

    macro_setting = config.get('batch_macro_setting')
    if isinstance(macro_setting, str) and macro_setting.strip():
        BATCH_MACRO_SETTING = macro_setting.strip()

    batch = translator_config.get('batch')
    if not isinstance(batch, dict):
        batch = {}

    BATCH_NON_CHINESE_RULES = batch_non_chinese_rules.load_non_chinese_rules(translator_config)

    model_name = batch.get('model')
    if isinstance(model_name, str) and model_name.strip():
        BATCH_MODEL = model_name.strip()

    display_name_prefix = batch.get('display_name_prefix')
    if isinstance(display_name_prefix, str) and display_name_prefix.strip():
        BATCH_DISPLAY_NAME_PREFIX = display_name_prefix.strip()

    BATCH_TARGET_SIZE = coerce_positive_int(batch.get('chunk_size'), BATCH_TARGET_SIZE)
    BATCH_TARGET_CHARS = coerce_positive_int(
        batch.get('max_source_chars', batch.get('target_chars')),
        BATCH_TARGET_CHARS,
    )
    BATCH_CONTEXT_BEFORE = coerce_positive_int(batch.get('context_before'), BATCH_CONTEXT_BEFORE)
    BATCH_CONTEXT_AFTER = coerce_positive_int(batch.get('context_after'), BATCH_CONTEXT_AFTER)
    BATCH_RETRY_TARGET_SIZE = coerce_positive_int(batch.get('retry_chunk_size'), BATCH_RETRY_TARGET_SIZE)
    BATCH_RETRY_TARGET_CHARS = coerce_positive_int(
        batch.get('retry_max_source_chars', batch.get('retry_target_chars')),
        BATCH_RETRY_TARGET_CHARS,
    )
    BATCH_MAX_OUTPUT_TOKENS = coerce_positive_int(
        batch.get('max_output_tokens'),
        BATCH_MAX_OUTPUT_TOKENS,
    )
    BATCH_TEMPERATURE = coerce_float(batch.get('temperature'), BATCH_TEMPERATURE)
    if 'thinking_level' in batch:
        BATCH_THINKING_LEVEL = coerce_thinking_level(
            batch.get('thinking_level'),
            BATCH_THINKING_LEVEL,
        )
    if 'safety_settings' in batch:
        BATCH_SAFETY_SETTINGS = normalize_batch_safety_settings(batch.get('safety_settings'))

    keyword_config = batch.get('keyword_extraction')
    if not isinstance(keyword_config, dict):
        keyword_config = {}
    KEYWORD_CHUNK_SIZE = coerce_positive_int(
        keyword_config.get('chunk_size'),
        KEYWORD_CHUNK_SIZE,
    )
    KEYWORD_MAX_CANDIDATES_PER_CHUNK = coerce_positive_int(
        keyword_config.get('max_candidates_per_chunk'),
        KEYWORD_MAX_CANDIDATES_PER_CHUNK,
    )
    display_name_prefix = keyword_config.get('display_name_prefix')
    if isinstance(display_name_prefix, str) and display_name_prefix.strip():
        KEYWORD_DISPLAY_NAME_PREFIX = display_name_prefix.strip()

    revision_config = batch.get('revision')
    if not isinstance(revision_config, dict):
        revision_config = {}
    REVISION_CHUNK_SIZE = coerce_positive_int(
        revision_config.get('chunk_size'),
        REVISION_CHUNK_SIZE,
    )
    revision_display_name_prefix = revision_config.get('display_name_prefix')
    if isinstance(revision_display_name_prefix, str) and revision_display_name_prefix.strip():
        REVISION_DISPLAY_NAME_PREFIX = revision_display_name_prefix.strip()

    macro_setting_file = batch.get('macro_setting_file')
    if macro_setting_file:
        resolved_path = legacy._resolve_path(legacy.BASE_DIR, macro_setting_file)
        macro_text = read_text_file(resolved_path)
        if macro_text:
            BATCH_MACRO_SETTING = macro_text

    macro_setting = batch.get('macro_setting')
    if isinstance(macro_setting, str) and macro_setting.strip():
        BATCH_MACRO_SETTING = macro_setting.strip()

    rag = batch.get('rag')
    if not isinstance(rag, dict):
        rag = {}

    RAG_ENABLED = coerce_bool(rag.get('enabled'), RAG_ENABLED)
    RAG_EMBEDDING_MODEL = coerce_non_empty_string(rag.get('embedding_model'), RAG_EMBEDDING_MODEL)
    RAG_QUERY_TASK_TYPE = normalize_task_type(rag.get('query_task_type'), RAG_QUERY_TASK_TYPE)
    RAG_DOCUMENT_TASK_TYPE = normalize_task_type(rag.get('document_task_type'), RAG_DOCUMENT_TASK_TYPE)
    RAG_OUTPUT_DIMENSIONALITY = coerce_positive_int(
        rag.get('output_dimensionality'),
        RAG_OUTPUT_DIMENSIONALITY,
    )
    RAG_TOP_K_HISTORY = coerce_positive_int(rag.get('top_k_history'), RAG_TOP_K_HISTORY)
    RAG_TOP_K_TERMS = coerce_positive_int(rag.get('top_k_terms'), RAG_TOP_K_TERMS)
    RAG_MIN_SIMILARITY = coerce_float(rag.get('min_similarity'), RAG_MIN_SIMILARITY)
    RAG_SEGMENT_LINES = coerce_positive_int(rag.get('segment_lines'), RAG_SEGMENT_LINES)
    RAG_BOOTSTRAP_ON_BUILD = coerce_bool(rag.get('bootstrap_on_build'), RAG_BOOTSTRAP_ON_BUILD)
    RAG_HISTORY_CHAR_LIMIT = coerce_positive_int(rag.get('history_char_limit'), RAG_HISTORY_CHAR_LIMIT)

    store_dir = rag.get('store_dir')
    if store_dir:
        RAG_STORE_DIR = legacy._resolve_path(legacy.BASE_DIR, store_dir)
    else:
        RAG_STORE_DIR = ''

    _RAG_STORE = None

    source_index_config = batch.get('source_index')
    if not isinstance(source_index_config, dict):
        source_index_config = {}

    SOURCE_INDEX_ENABLED = coerce_bool(source_index_config.get('enabled'), SOURCE_INDEX_ENABLED)
    SOURCE_INDEX_TOP_K = coerce_positive_int(source_index_config.get('top_k'), SOURCE_INDEX_TOP_K)
    SOURCE_INDEX_MIN_SIMILARITY = coerce_float(source_index_config.get('min_similarity'), SOURCE_INDEX_MIN_SIMILARITY)
    SOURCE_INDEX_CHAR_LIMIT = coerce_positive_int(source_index_config.get('char_limit'), SOURCE_INDEX_CHAR_LIMIT)
    source_index_store_dir = source_index_config.get('store_dir')
    if source_index_store_dir:
        SOURCE_INDEX_STORE_DIR = legacy._resolve_path(legacy.BASE_DIR, source_index_store_dir)
    else:
        SOURCE_INDEX_STORE_DIR = ''

    _SOURCE_INDEX_STORE = None

    story_config = batch.get('story_memory')
    if not isinstance(story_config, dict):
        story_config = {}

    STORY_MEMORY_ENABLED = coerce_bool(story_config.get('enabled'), False)
    STORY_MEMORY_MAX_CONTEXT_CHARS = coerce_positive_int(
        story_config.get('max_context_chars'),
        1200,
    )
    STORY_MEMORY_TOP_K_RELATIONS = coerce_positive_int(
        story_config.get('top_k_relations'),
        6,
    )
    STORY_MEMORY_TOP_K_TERMS = coerce_positive_int(
        story_config.get('top_k_terms'),
        12,
    )
    STORY_MEMORY_INCLUDE_SCENE_SUMMARY = coerce_bool(
        story_config.get('include_scene_summary'),
        True,
    )
    graph_file = story_config.get('graph_file')
    if graph_file:
        STORY_MEMORY_GRAPH_FILE = legacy.resolve_story_memory_graph_path(graph_file)
    elif STORY_MEMORY_ENABLED:
        STORY_MEMORY_GRAPH_FILE = legacy.get_default_story_memory_graph_path()
    else:
        STORY_MEMORY_GRAPH_FILE = ''
    _STORY_GRAPH = None
    _STORY_GRAPH_PATH = ''

def load_progress():
    if not os.path.exists(PROGRESS_LOG):
        return {}
    try:
        with open(PROGRESS_LOG, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_progress(progress):
    ensure_batch_dirs()
    with open(PROGRESS_LOG, 'w', encoding='utf-8') as handle:
        json.dump(progress, handle, ensure_ascii=False, indent=2)


def update_progress(file_key, translated_lines):
    progress = load_progress()
    progress.setdefault(file_key, [])
    progress[file_key].extend(translated_lines)
    progress[file_key] = sorted(set(progress[file_key]))
    save_progress(progress)


def ensure_batch_sdk():
    if genai is None or genai_types is None:
        raise SystemExit('google-genai is not installed. Run: pip install google-genai')


def normalize_api_key_index(value):
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    api_keys = getattr(legacy, 'API_KEYS', []) or []
    if 0 <= index < len(api_keys):
        return index
    return None


def create_batch_client(api_key_index=None):
    ensure_batch_sdk()
    if api_key_index is None:
        api_key = legacy.get_current_api_key()
    else:
        index = normalize_api_key_index(api_key_index)
        if index is None:
            raise SystemExit(f'Invalid API key index: {api_key_index}')
        api_key = legacy.API_KEYS[index]
    return genai.Client(api_key=api_key)


def is_quota_error(exc):
    status_code = getattr(exc, 'status_code', None)
    if status_code == 429:
        return True
    text = str(exc)
    return '429' in text or 'RESOURCE_EXHAUSTED' in text


def is_not_found_error(exc):
    status_code = getattr(exc, 'status_code', None)
    if status_code == 404:
        return True
    text = str(exc)
    return '404' in text or 'NOT_FOUND' in text


def is_unavailable_error(exc):
    status_code = getattr(exc, 'status_code', None)
    if status_code == 503:
        return True
    text = str(exc)
    retryable_markers = (
        '503',
        'UNAVAILABLE',
        'UNEXPECTED_EOF_WHILE_READING',
        'EOF occurred in violation of protocol',
        'ConnectError',
        'ReadError',
        'ConnectTimeout',
        'ReadTimeout',
        'RemoteProtocolError',
    )
    return any(marker in text for marker in retryable_markers)


def allow_non_chinese_repair_translation(original, translated):
    if legacy.allow_non_chinese_term_translation(
        original,
        translated,
        known_terms=collect_shared_rag_preserved_terms(),
    ):
        return True
    if not original or not translated or original == translated:
        return False
    if legacy.contains_chinese(translated):
        return False
    stripped = original.strip()
    if stripped.startswith('{#') or '%' in stripped:
        return True
    return False


def iter_manifest_api_key_indices(manifest):
    api_keys = getattr(legacy, 'API_KEYS', []) or []
    preferred = []
    for key in ('submitted_api_key_index', 'last_status_api_key_index'):
        index = normalize_api_key_index(manifest.get(key))
        if index is not None and index not in preferred:
            preferred.append(index)
    for index in range(len(api_keys)):
        if index not in preferred:
            preferred.append(index)
    return preferred


def fetch_batch_job_for_manifest(manifest):
    if not manifest.get('job_name'):
        raise SystemExit('Manifest does not have a job_name yet.')

    last_error = None
    for api_key_index in iter_manifest_api_key_indices(manifest):
        client = create_batch_client(api_key_index=api_key_index)
        try:
            batch_job = client.batches.get(name=manifest['job_name'])
            manifest['submitted_api_key_index'] = api_key_index
            manifest['submitted_api_key_number'] = api_key_index + 1
            manifest['last_status_api_key_index'] = api_key_index
            return client, batch_job
        except Exception as exc:
            last_error = exc
            if is_not_found_error(exc):
                continue
            raise

    if last_error is not None and is_not_found_error(last_error):
        raise SystemExit(
            'Batch job not found under any configured API key/project. '
            'It may belong to a different project, or the job may no longer exist.'
        )
    if last_error is not None:
        raise last_error
    raise SystemExit('No API keys available to query batch job.')


def slugify(text):
    text = re.sub(r'[^A-Za-z0-9._-]+', '-', text or '').strip('-._')
    return text or 'batch'


def guess_project_slug():
    base_name = os.path.basename(os.path.abspath(legacy.BASE_DIR))
    if base_name.lower() == 'work':
        parent = os.path.basename(os.path.dirname(os.path.abspath(legacy.BASE_DIR)))
        return slugify(parent or base_name)
    return slugify(base_name)


def hash_key(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]


def get_default_rag_store_dir():
    return legacy.get_default_batch_rag_store_dir()


def get_default_source_index_store_dir():
    return legacy.get_default_source_index_store_dir()


def get_source_index_char_budget():
    return max(0, int(SOURCE_INDEX_TOP_K or 0)) * max(0, int(SOURCE_INDEX_CHAR_LIMIT or 0))


def get_source_index_store(update_metadata=True):
    global _SOURCE_INDEX_STORE, SOURCE_INDEX_STORE_DIR
    if not SOURCE_INDEX_STORE_DIR:
        SOURCE_INDEX_STORE_DIR = get_default_source_index_store_dir()
    if _SOURCE_INDEX_STORE is None or os.path.abspath(_SOURCE_INDEX_STORE.store_dir) != os.path.abspath(SOURCE_INDEX_STORE_DIR):
        _SOURCE_INDEX_STORE = JsonSourceIndexStore(SOURCE_INDEX_STORE_DIR)
    if update_metadata:
        _SOURCE_INDEX_STORE.set_metadata(
            schema_version=SOURCE_INDEX_SCHEMA_VERSION,
            project_slug=guess_project_slug(),
            embedding_model=RAG_EMBEDDING_MODEL,
            document_task_type=RAG_DOCUMENT_TASK_TYPE,
            output_dimensionality=RAG_OUTPUT_DIMENSIONALITY,
        )
    return _SOURCE_INDEX_STORE


def get_rag_store():
    global _RAG_STORE, RAG_STORE_DIR
    if not RAG_ENABLED:
        return None
    if not RAG_STORE_DIR:
        RAG_STORE_DIR = get_default_rag_store_dir()
    if _RAG_STORE is None or os.path.abspath(_RAG_STORE.store_dir) != os.path.abspath(RAG_STORE_DIR):
        _RAG_STORE = JsonRagStore(RAG_STORE_DIR)
        _RAG_STORE.set_metadata(
            project_slug=guess_project_slug(),
            embedding_model=RAG_EMBEDDING_MODEL,
            query_task_type=RAG_QUERY_TASK_TYPE,
            document_task_type=RAG_DOCUMENT_TASK_TYPE,
            output_dimensionality=RAG_OUTPUT_DIMENSIONALITY,
        )
    return _RAG_STORE


def extract_word_tokens(text):
    return legacy._extract_word_tokens(text)


def collect_shared_rag_preserved_terms():
    global _RAG_PRESERVED_TERMS_CACHE, _RAG_PRESERVED_TERMS_CACHE_KEY

    store = get_rag_store()
    cache_key = os.path.abspath(store.store_dir) if store is not None else ''
    if _RAG_PRESERVED_TERMS_CACHE is not None and _RAG_PRESERVED_TERMS_CACHE_KEY == cache_key:
        return set(_RAG_PRESERVED_TERMS_CACHE)

    terms = set(getattr(legacy, 'PRESERVE_TERMS_LOWER', set()) or [])
    if store is not None:
        store.load()
        for record in store.history.values():
            source_tokens = set(extract_word_tokens(record.get('source_text', '')))
            translated_tokens = set(extract_word_tokens(record.get('translated_text', '')))
            terms.update(source_tokens & translated_tokens)

    _RAG_PRESERVED_TERMS_CACHE = tuple(sorted(terms))
    _RAG_PRESERVED_TERMS_CACHE_KEY = cache_key
    return set(terms)


def collect_chunk_known_terms(chunk):
    terms = collect_shared_rag_preserved_terms()
    for hit in chunk.get('glossary_hits') or []:
        for value in (hit.get('source', ''), hit.get('target', '')):
            if value:
                terms.add(value)
            terms.update(extract_word_tokens(value))
    for hit in chunk.get('history_hits') or []:
        source_tokens = set(extract_word_tokens(hit.get('source_text', '')))
        translated_tokens = set(extract_word_tokens(hit.get('translated_text', '')))
        terms.update(source_tokens & translated_tokens)
    return terms


def _manifest_tl_base_dir(manifest):
    base_dir = manifest.get('tl_dir') if isinstance(manifest, dict) else ''
    if isinstance(base_dir, str) and base_dir.strip():
        return base_dir.strip()
    return legacy.TL_DIR


def _manifest_file_path_for_chunk(manifest, chunk):
    if not isinstance(chunk, dict):
        return ''
    file_key = chunk.get('file_rel_path') or chunk.get('file') or ''
    if not isinstance(file_key, str) or not file_key.strip():
        return ''
    try:
        return resolve_path_under_dir(
            _manifest_tl_base_dir(manifest),
            file_key,
            f'manifest file key {file_key}',
        )
    except SystemExit:
        return ''


def _item_source_line_number(item):
    if not isinstance(item, dict):
        return 0
    for field in ('line_number', 'target_line_number'):
        try:
            value = int(item.get(field) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    try:
        line_index = int(item.get('line') or 0)
    except (TypeError, ValueError):
        return 0
    return line_index + 1 if line_index >= 0 else 0


class NonChineseFileReadCache:
    """Per-call cache for TL/source reads inside non-Chinese validation helpers."""

    def __init__(self):
        self._lines_by_path = {}
        self._line_by_path_and_number = {}

    def read_line(self, path, line_number):
        if not path or line_number <= 0:
            return ''
        cache_key = (path, line_number)
        if cache_key in self._line_by_path_and_number:
            return self._line_by_path_and_number[cache_key]
        lines = self.read_lines(path)
        if line_number <= len(lines):
            line = lines[line_number - 1]
        else:
            line = ''
        self._line_by_path_and_number[cache_key] = line
        return line

    def read_lines(self, path):
        if not path:
            return []
        if path not in self._lines_by_path:
            try:
                with open(path, 'r', encoding='utf-8-sig') as handle:
                    lines = handle.readlines()
            except OSError:
                lines = []
            self._lines_by_path[path] = lines
            for index, line in enumerate(lines, 1):
                self._line_by_path_and_number.setdefault((path, index), line)
        return self._lines_by_path[path]

    @staticmethod
    def _read_line_uncached(path, line_number):
        if not path or line_number <= 0:
            return ''
        try:
            with open(path, 'r', encoding='utf-8-sig') as handle:
                for current_number, line in enumerate(handle, 1):
                    if current_number == line_number:
                        return line
        except OSError:
            return ''
        return ''


def _read_line_at(path, line_number, *, file_read_cache=None):
    if file_read_cache is not None:
        return file_read_cache.read_line(path, line_number)
    return NonChineseFileReadCache._read_line_uncached(path, line_number)


def _manifest_tl_line_for_item(manifest, chunk, item, *, file_read_cache=None):
    return _read_line_at(
        _manifest_file_path_for_chunk(manifest, chunk),
        _item_source_line_number(item),
        file_read_cache=file_read_cache,
    )


def is_manifest_keyword_argument_item(manifest, chunk, item, *, file_read_cache=None):
    if not isinstance(item, dict):
        return False
    line = _manifest_tl_line_for_item(manifest, chunk, item, file_read_cache=file_read_cache)
    if not line:
        return False
    return legacy.is_keyword_argument_string_span(line, item.get('start'), item.get('end'))


def is_manifest_say_speaker_label_item(manifest, chunk, item, *, file_read_cache=None):
    if not isinstance(item, dict):
        return False
    line = _manifest_tl_line_for_item(manifest, chunk, item, file_read_cache=file_read_cache)
    if not line:
        return False
    return legacy.is_say_speaker_label_string_span(line, item.get('start'), item.get('end'))


def is_manifest_old_new_static_label_item(manifest, chunk, item, *, file_read_cache=None):
    if not isinstance(item, dict):
        return False
    line = _manifest_tl_line_for_item(manifest, chunk, item, file_read_cache=file_read_cache)
    if not line:
        return False
    stripped = line.lstrip()
    return stripped.startswith('old ') or stripped.startswith('new ')

GAME_LINE_COMMENT_RE = re.compile(r'^\s*#\s+(.+?):(\d+)\s*$')
OLD_NEW_LINE_RE = re.compile(r'^\s*(?:old|new)\s+"(?P<text>.*)"\s*$')
STATIC_NAME_PUNCT_TRANSLATION = str.maketrans({
    '。': '.',
    '，': ',',
    '、': ',',
    '！': '!',
    '？': '?',
    '：': ':',
    '；': ';',
})


def normalize_static_name_or_credit_text(text):
    return compact_text((text or '').translate(STATIC_NAME_PUNCT_TRANSLATION))


def static_name_or_credit_text_matches(original, translated):
    return normalize_static_name_or_credit_text(original) == normalize_static_name_or_credit_text(translated)


NON_CHINESE_TOKEN_PUNCT_TRANSLATION = str.maketrans({
    '。': '.',
    '，': ',',
    '、': ',',
    '！': '!',
    '？': '?',
    '：': ':',
    '；': ';',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
})


def normalize_non_chinese_token_text(text):
    cleaned = legacy.RENPY_TAG_RE.sub('', text or '')
    cleaned = legacy.RENPY_FIELD_RE.sub('', cleaned)
    cleaned = cleaned.translate(NON_CHINESE_TOKEN_PUNCT_TRANSLATION).strip()
    cleaned = cleaned.strip('"\'')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'[.!?,:;]+$', '', cleaned).strip()
    return cleaned


def looks_like_preserved_or_acronym_text(text):
    cleaned = normalize_non_chinese_token_text(text)
    if not cleaned or legacy.contains_chinese(cleaned):
        return False
    tokens = legacy.WORD_TOKEN_RE.findall(cleaned)
    if not tokens or len(tokens) > 4:
        return False
    for token in tokens:
        if token.lower() in legacy.PRESERVE_TERMS_LOWER:
            continue
        if token.isupper() and 1 <= len(token) <= 8:
            continue
        return False
    return True


def matching_preserved_or_acronym_non_chinese_text(original, translated):
    if legacy.contains_chinese(translated or ''):
        return False
    original_norm = normalize_non_chinese_token_text(original)
    translated_norm = normalize_non_chinese_token_text(translated)
    if not original_norm or original_norm.lower() != translated_norm.lower():
        return False
    return looks_like_preserved_or_acronym_text(original_norm)


def looks_like_static_name_or_credit_text(text):
    cleaned = legacy.RENPY_TAG_RE.sub('', text or '')
    cleaned = legacy.RENPY_FIELD_RE.sub('', cleaned)
    cleaned = cleaned.strip()
    if not cleaned or legacy.contains_chinese(cleaned):
        return False
    if ':' in cleaned or any(mark in cleaned for mark in '!?！？'):
        return False

    tokens = legacy.WORD_TOKEN_RE.findall(cleaned)
    if not tokens:
        return True
    if len(tokens) > 12:
        return False
    if legacy.is_non_translatable(cleaned):
        return True

    allowed_particles = {'a', 'an', 'and', 'de', 'del', 'der', 'of', 'the', 'van', 'von'}
    for token in tokens:
        if token.lower() in allowed_particles:
            continue
        if token.isupper() or token[:1].isupper():
            continue
        return False
    return True


def _manifest_base_dir(manifest):
    base_dir = manifest.get('base_dir') if isinstance(manifest, dict) else ''
    if isinstance(base_dir, str) and base_dir.strip():
        return base_dir.strip()
    return legacy.BASE_DIR


def _read_source_line_for_tl_item(manifest, chunk, item, *, file_read_cache=None):
    tl_path = _manifest_file_path_for_chunk(manifest, chunk)
    line_number = _item_source_line_number(item)
    if not tl_path or line_number <= 0:
        return ''
    if file_read_cache is not None:
        lines = file_read_cache.read_lines(tl_path)
    else:
        try:
            with open(tl_path, 'r', encoding='utf-8-sig') as handle:
                lines = handle.readlines()
        except OSError:
            return ''
    start_index = max(0, line_number - 6)
    end_index = min(len(lines), line_number)
    for index in range(end_index - 1, start_index - 1, -1):
        match = GAME_LINE_COMMENT_RE.match(lines[index])
        if not match:
            continue
        source_rel_path, source_line_number = match.groups()
        try:
            source_path = resolve_path_under_dir(
                _manifest_base_dir(manifest),
                source_rel_path,
                f'game source line {source_rel_path}',
            )
            return _read_line_at(
                source_path,
                int(source_line_number),
                file_read_cache=file_read_cache,
            )
        except (SystemExit, ValueError):
            return ''
    return ''


def is_manifest_player_name_comparison_item(manifest, chunk, original, item, *, file_read_cache=None):
    if not isinstance(item, dict):
        return False
    line = _read_source_line_for_tl_item(
        manifest,
        chunk,
        item,
        file_read_cache=file_read_cache,
    )
    if not line:
        return False
    if not re.search(r'\b(?:Main|main_nm|yourname|persistent\.MainEP)\b\s*==\s*_\(', line):
        return False
    compared_name_match = re.search(r'_\(\s*["\']([^"\']+)["\']\s*\)', line)
    if not compared_name_match:
        return False
    compared_name = compared_name_match.group(1)
    if compact_text(original) != compact_text(compared_name):
        return False
    return looks_like_static_name_or_credit_text(original)


def is_manifest_static_non_chinese_item(
    manifest,
    chunk,
    original,
    translated,
    item=None,
    *,
    file_read_cache=None,
):
    if not static_name_or_credit_text_matches(original, translated):
        return False
    if legacy.contains_chinese(translated or ''):
        return False

    rel_path = str((chunk or {}).get('file_rel_path') or '').replace('\\', '/').lower()
    rel_name = os.path.basename(rel_path)
    rules = batch_non_chinese_rules.effective_non_chinese_rules(
        manifest,
        runtime_rules=BATCH_NON_CHINESE_RULES,
    )

    if batch_non_chinese_rules.rel_path_matches(
        rel_path,
        rel_name,
        rules.get('static_name_credit_unconditional_rel_paths', []),
    ):
        return True

    if batch_non_chinese_rules.rel_path_matches(
        rel_path,
        rel_name,
        rules.get('static_name_credit_rel_paths', []),
    ):
        return looks_like_static_name_or_credit_text(original)

    if batch_non_chinese_rules.rel_path_matches(
        rel_path,
        rel_name,
        rules.get('charselect_rel_paths', []),
    ):
        if any(mark in (original or '') for mark in ':!?！？'):
            return False
        return looks_like_static_name_or_credit_text(original)

    if is_manifest_old_new_static_label_item(
        manifest,
        chunk,
        item,
        file_read_cache=file_read_cache,
    ):
        line = _manifest_tl_line_for_item(
            manifest,
            chunk,
            item,
            file_read_cache=file_read_cache,
        )
        label_match = OLD_NEW_LINE_RE.match(line or '')
        if not label_match:
            return False
        if compact_text(original) != compact_text(label_match.group('text')):
            return False
        return looks_like_static_name_or_credit_text(original)

    if (
        batch_non_chinese_rules.rel_path_matches(
            rel_path,
            rel_name,
            rules.get('player_name_comparison_rel_paths', []),
        )
        and is_manifest_player_name_comparison_item(
            manifest,
            chunk,
            original,
            item,
            file_read_cache=file_read_cache,
        )
    ):
        return True

    if (
        batch_non_chinese_rules.rel_path_has_suffix(rel_path, rules.get('define_rel_path_suffixes', []))
        or batch_non_chinese_rules.rel_path_has_prefix(rel_path, rules.get('define_rel_path_prefixes', []))
    ):
        cleaned = legacy.RENPY_TAG_RE.sub('', original or '').strip()
        if legacy.is_non_translatable(cleaned):
            return True
        if any(mark in cleaned for mark in '.!?。！？'):
            return False
        tokens = legacy.WORD_TOKEN_RE.findall(cleaned)
        return 1 <= len(tokens) <= 3

    return False


def allow_non_chinese_batch_translation(manifest, chunk, original, translated, item=None):
    file_read_cache = NonChineseFileReadCache()
    unchanged = (original or '').strip() == (translated or '').strip()
    if (
        (unchanged and (
            is_manifest_keyword_argument_item(
                manifest,
                chunk,
                item,
                file_read_cache=file_read_cache,
            )
            or is_manifest_say_speaker_label_item(
                manifest,
                chunk,
                item,
                file_read_cache=file_read_cache,
            )
        ))
        or is_manifest_static_non_chinese_item(
            manifest,
            chunk,
            original,
            translated,
            item,
            file_read_cache=file_read_cache,
        )
        or matching_preserved_or_acronym_non_chinese_text(original, translated)
    ):
        return True
    return legacy.allow_non_chinese_term_translation(
        original,
        translated,
        known_terms=collect_chunk_known_terms(chunk),
    )

def compact_text(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def item_text(item):
    if isinstance(item, dict):
        return item.get('text') or item.get('source') or ''
    return item


def compact_item_texts(items):
    compacted = []
    for item in items or []:
        text = compact_text(item_text(item))
        if text:
            compacted.append(text)
    return compacted


def build_rag_query_text(target_items, context_past):
    parts = []
    local_past = compact_item_texts(context_past[-2:])
    target_lines = compact_item_texts(target_items)
    if local_past:
        parts.append('Context before:\n' + '\n'.join(f'- {text}' for text in local_past))
    if target_lines:
        parts.append('Target:\n' + '\n'.join(f'- {text}' for text in target_lines))
    return '\n\n'.join(parts)


def embed_texts(contents, task_type):
    if not contents:
        return []
    api_key_count = len(getattr(legacy, 'API_KEYS', []) or [])
    attempts = max(3, api_key_count * 2)
    last_error = None
    for attempt in range(1, attempts + 1):
        client = create_batch_client()
        try:
            response = client.models.embed_content(
                model=RAG_EMBEDDING_MODEL,
                contents=contents,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=RAG_OUTPUT_DIMENSIONALITY,
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            retryable = is_quota_error(exc) or is_unavailable_error(exc)
            if retryable and attempt < attempts:
                rotated = legacy.rotate_api_key()
                label = 'quota' if is_quota_error(exc) else 'service unavailable'
                key_action = 'next API key' if rotated else 'same API key'
                print(f'Embedding request hit {label}. Retrying with {key_action} ({attempt}/{attempts})...')
                time.sleep(min(attempt, 2))
                continue
            raise
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError('Embedding request failed without a captured exception.')
    embeddings = getattr(response, 'embeddings', None) or []
    values = [list(getattr(item, 'values', None) or []) for item in embeddings]
    if len(values) != len(contents):
        raise RuntimeError(f'Embedding count mismatch: expected {len(contents)}, got {len(values)}')
    return values


def embed_query_text(query_text):
    query_text = compact_text(query_text)
    if not query_text:
        return []
    vectors = embed_texts([query_text], RAG_QUERY_TASK_TYPE)
    return vectors[0] if vectors else []


def retrieve_glossary_hits(target_items):
    if not RAG_ENABLED:
        return []
    combined_text = '\n'.join(item.get('text', '') for item in target_items if item.get('text'))
    if not combined_text:
        return []
    hits = []
    seen = set()
    for source, target in (legacy.NORMALIZE_TRANSLATION_MAP or {}).items():
        if source and source in combined_text and source not in seen:
            hits.append({'source': source, 'target': target, 'kind': 'normalize'})
            seen.add(source)
    for term in legacy.PRESERVE_TERMS:
        if not isinstance(term, str) or not term.strip():
            continue
        if term in combined_text and term not in seen:
            hits.append({'source': term, 'target': term, 'kind': 'preserve'})
            seen.add(term)
    return hits[:RAG_TOP_K_TERMS]


def format_glossary_hits_block(hits, empty_label='(none)'):
    return prompt_context.format_glossary_hits_block(hits, empty_label)


def format_history_hits_block(hits, empty_label='(none)'):
    return prompt_context.format_history_hits_block(
        hits,
        empty_label,
        char_limit=RAG_HISTORY_CHAR_LIMIT,
        include_source_text=True,
    )


def retrieve_history_hits(target_items, context_past):
    if not RAG_ENABLED:
        return [], {'enabled': False}
    store = get_rag_store()
    if store is None or store.count_history() <= 0:
        return [], {'enabled': True, 'reason': 'empty_history_store'}

    query_text = build_rag_query_text(target_items, context_past)
    if not query_text:
        return [], {'enabled': True, 'reason': 'empty_query'}

    try:
        query_vector = embed_query_text(query_text)
        matches = store.search_history(
            query_vector,
            top_k=RAG_TOP_K_HISTORY,
            min_similarity=RAG_MIN_SIMILARITY,
        )
    except Exception as exc:
        print(f'Warning: RAG history retrieval failed: {exc}')
        return [], {'enabled': True, 'error': str(exc)}

    hits = []
    for match in matches:
        hits.append(
            {
                'memory_id': match.get('memory_id', ''),
                'file_rel_path': match.get('file_rel_path', ''),
                'line_start': match.get('line_start', 0),
                'line_end': match.get('line_end', 0),
                'source_text': truncate_text(match.get('source_text', ''), RAG_HISTORY_CHAR_LIMIT),
                'translated_text': truncate_text(match.get('translated_text', ''), RAG_HISTORY_CHAR_LIMIT),
                'quality_state': match.get('quality_state', ''),
                'score': float(match.get('score', 0.0)),
            }
        )

    return hits, {
        'enabled': True,
        'query_text': truncate_text(query_text, 400),
        'hit_count': len(hits),
    }


def retrieve_source_hits(target_items, context_past):
    if not SOURCE_INDEX_ENABLED:
        return [], {'enabled': False}

    query_text = build_rag_query_text(target_items, context_past)
    if not query_text:
        return [], {
            'enabled': True,
            'reason': 'empty_query',
            'source_context_char_budget': get_source_index_char_budget(),
        }

    try:
        store = get_source_index_store(update_metadata=False)
        if store is None or store.count_segments() <= 0:
            return [], {
                'enabled': True,
                'reason': 'empty_source_store',
                'source_context_char_budget': get_source_index_char_budget(),
                'store_dir': getattr(store, 'store_dir', SOURCE_INDEX_STORE_DIR or ''),
                'store_schema_version': (getattr(store, 'metadata', {}) or {}).get('schema_version') if store else None,
            }
        query_vector = embed_query_text(query_text)
        matches, search_diagnostics = store.search_segments(
            query_vector,
            top_k=SOURCE_INDEX_TOP_K,
            min_similarity=SOURCE_INDEX_MIN_SIMILARITY,
            embedding_model=RAG_EMBEDDING_MODEL,
            embedding_task_type=RAG_DOCUMENT_TASK_TYPE,
            embedding_dim=RAG_OUTPUT_DIMENSIONALITY,
            return_diagnostics=True,
        )
    except Exception as exc:
        print(f'Warning: Source index retrieval failed: {exc}')
        return [], {
            'enabled': True,
            'error': str(exc),
            'failure_reason': 'retrieval_error',
            'source_context_char_budget': get_source_index_char_budget(),
        }

    hits = []
    truncated_count = 0
    source_context_chars = 0
    for match in matches:
        source_text = match.get('source_text', '')
        truncated_source_text = truncate_text(source_text, SOURCE_INDEX_CHAR_LIMIT)
        was_truncated = isinstance(source_text, str) and truncated_source_text != source_text
        if was_truncated:
            truncated_count += 1
        source_context_chars += len(truncated_source_text)
        hit = {
            'source_id': match.get('source_id', ''),
            'file_rel_path': match.get('file_rel_path', ''),
            'line_start': match.get('line_start', 0),
            'line_end': match.get('line_end', 0),
            'source_text': truncated_source_text,
            'source_text_truncated': was_truncated,
            'score': float(match.get('score', 0.0)),
        }
        hits.append(hit)

    return hits, {
        'enabled': True,
        'query_text': truncate_text(query_text, 400),
        'query_char_count': len(query_text),
        'hit_count': len(hits),
        'matched_count': search_diagnostics.get('matched_before_top_k', len(matches)),
        'filtered_count': search_diagnostics.get('metadata_filtered_count', 0),
        'stale_hits_skipped': search_diagnostics.get('metadata_filtered_count', 0),
        'below_similarity_count': search_diagnostics.get('below_similarity_count', 0),
        'truncated_count': truncated_count,
        'source_context_chars': source_context_chars,
        'source_context_char_budget': get_source_index_char_budget(),
        'store_dir': getattr(store, 'store_dir', SOURCE_INDEX_STORE_DIR or ''),
        'store_schema_version': (getattr(store, 'metadata', {}) or {}).get('schema_version'),
        'search_diagnostics': search_diagnostics,
    }


def get_story_graph():
    global _STORY_GRAPH, _STORY_GRAPH_PATH
    if not STORY_MEMORY_ENABLED:
        return None
    graph_path = os.path.abspath(STORY_MEMORY_GRAPH_FILE) if STORY_MEMORY_GRAPH_FILE else ''
    if _STORY_GRAPH is None or _STORY_GRAPH_PATH != graph_path:
        _STORY_GRAPH = story_memory.load_story_graph(graph_path)
        _STORY_GRAPH_PATH = graph_path
    return _STORY_GRAPH


def retrieve_batch_story_hits(file_rel_path, target_items, context_past, context_future):
    if not STORY_MEMORY_ENABLED:
        return None
    return story_memory.retrieve_story_hits(
        get_story_graph(),
        file_rel_path,
        target_items,
        context_past=context_past,
        context_future=context_future,
        top_k_relations=STORY_MEMORY_TOP_K_RELATIONS,
        top_k_terms=STORY_MEMORY_TOP_K_TERMS,
        include_scene_summary=STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
    )



def manifest_path_for_target(target):
    if target:
        candidate = os.path.abspath(target)
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, 'manifest.json')
        if os.path.isfile(candidate):
            return candidate
        raise SystemExit(f'Manifest not found: {target}')

    if os.path.isfile(LATEST_MANIFEST_FILE):
        with open(LATEST_MANIFEST_FILE, 'r', encoding='utf-8') as handle:
            candidate = handle.read().strip()
        if candidate and os.path.isfile(candidate):
            return candidate

    manifests = []
    for root, _, files in os.walk(BATCH_JOBS_DIR):
        if 'manifest.json' in files:
            manifests.append(os.path.join(root, 'manifest.json'))
    if not manifests:
        raise SystemExit('No batch manifest found.')
    manifests.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return manifests[0]


def remember_latest_manifest(manifest_path):
    ensure_batch_dirs()
    with open(LATEST_MANIFEST_FILE, 'w', encoding='utf-8') as handle:
        handle.write(manifest_path)


def load_manifest(target=None):
    manifest_path = manifest_path_for_target(target)
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    manifest['_manifest_path'] = manifest_path
    manifest['_package_dir'] = os.path.dirname(manifest_path)
    return manifest


def manifest_mode(manifest):
    mode = manifest.get('mode', MANIFEST_MODE_TRANSLATION)
    return mode if isinstance(mode, str) and mode.strip() else MANIFEST_MODE_TRANSLATION


def require_manifest_mode(manifest, expected_mode, command_name):
    current_mode = manifest_mode(manifest)
    if current_mode != expected_mode:
        raise SystemExit(
            f'{command_name} only supports {expected_mode} manifests; '
            f'this manifest is {current_mode}.'
        )


def _canonical_abs_path(path):
    abs_path = os.path.abspath(path)
    try:
        return str(Path(abs_path).resolve(strict=False))
    except OSError:
        return abs_path


def _normalized_abs_path(path):
    return os.path.normcase(_canonical_abs_path(path))


def path_is_within_dir(base_dir, candidate):
    base = _normalized_abs_path(base_dir)
    target = _normalized_abs_path(candidate)
    try:
        return os.path.commonpath([base, target]) == base
    except ValueError:
        return False


def normalize_safe_rel_path(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'Unsafe {field_name}: empty path.')
    text = value.strip().replace('\\', '/')
    if os.path.isabs(text) or re.match(r'^[A-Za-z]:', text):
        raise SystemExit(f'Unsafe {field_name}: absolute paths are not allowed here.')
    parts = []
    for part in text.split('/'):
        if not part or part == '.':
            continue
        if part == '..':
            raise SystemExit(f'Unsafe {field_name}: parent directory segments are not allowed.')
        parts.append(part)
    if not parts:
        raise SystemExit(f'Unsafe {field_name}: empty path.')
    return os.path.join(*parts)


def resolve_path_under_dir(base_dir, value, field_name):
    if not base_dir:
        raise SystemExit(f'Unsafe {field_name}: base directory is missing.')
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'Unsafe {field_name}: empty path.')
    raw = value.strip()
    if os.path.isabs(raw):
        candidate = _canonical_abs_path(raw)
    else:
        candidate = _canonical_abs_path(os.path.join(base_dir, normalize_safe_rel_path(raw, field_name)))
    if not path_is_within_dir(base_dir, candidate):
        raise SystemExit(f'Unsafe {field_name}: {value} escapes {base_dir}.')
    return candidate


def resolve_manifest_result_path(manifest):
    package_dir = manifest.get('_package_dir')
    result_path = manifest.get('result_jsonl_path')
    if result_path:
        return resolve_path_under_dir(package_dir, result_path, 'result_jsonl_path')
    return os.path.join(package_dir, 'results.jsonl')


def resolve_manifest_file_path(manifest, file_key, file_info):
    path_value = file_info.get('path') if isinstance(file_info, dict) else ''
    if path_value:
        return resolve_path_under_dir(legacy.TL_DIR, path_value, f'manifest file path for {file_key}')
    return resolve_path_under_dir(legacy.TL_DIR, file_key, f'manifest file key {file_key}')


def save_manifest(manifest, update_latest=True):
    manifest_path = manifest['_manifest_path']
    data = dict(manifest)
    data.pop('_manifest_path', None)
    data.pop('_package_dir', None)
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    if update_latest:
        remember_latest_manifest(manifest_path)


def safe_nonnegative_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def manifest_summary_counts(manifest):
    summary = manifest.get('summary') if isinstance(manifest.get('summary'), dict) else {}
    chunk_count = safe_nonnegative_int(summary.get('chunk_count'))
    item_count = safe_nonnegative_int(summary.get('item_count'))
    chunks = manifest.get('chunks')
    if isinstance(chunks, list):
        if not chunk_count:
            chunk_count = len(chunks)
        if not item_count:
            for chunk in chunks:
                if isinstance(chunk, dict):
                    items = chunk.get('items')
                    if isinstance(items, list):
                        item_count += len(items)
    return chunk_count, item_count


def manifest_exceeds_split_recommendation(manifest):
    chunk_count, item_count = manifest_summary_counts(manifest)
    return (
        chunk_count > BATCH_SPLIT_RECOMMEND_CHUNKS
        or item_count > BATCH_SPLIT_RECOMMEND_ITEMS
    )


def quote_command_arg(value):
    text = str(value or '')
    if text == '' or re.search(r'\s|["&]', text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def split_display_name_prefix(manifest):
    display_name = manifest.get('display_name')
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    package_dir = manifest.get('_package_dir')
    if isinstance(package_dir, str) and package_dir.strip():
        return os.path.basename(package_dir.rstrip('\\/')) or BATCH_DISPLAY_NAME_PREFIX
    return BATCH_DISPLAY_NAME_PREFIX


def build_split_recommendation(manifest):
    if not manifest_exceeds_split_recommendation(manifest):
        return {}
    chunk_count, item_count = manifest_summary_counts(manifest)
    manifest_path = manifest.get('_manifest_path') or ''
    prefix = split_display_name_prefix(manifest)
    command = ' '.join([
        'python',
        'gemini_translate_batch.py',
        'split',
        quote_command_arg(manifest_path),
        '--max-chunks',
        str(BATCH_SPLIT_RECOMMEND_CHUNKS),
        '--max-items',
        str(BATCH_SPLIT_RECOMMEND_ITEMS),
        '--display-name-prefix',
        quote_command_arg(prefix),
    ])
    return {
        'reason': 'quota_or_resource_exhausted',
        'chunk_count': chunk_count,
        'item_count': item_count,
        'max_chunks': BATCH_SPLIT_RECOMMEND_CHUNKS,
        'max_items': BATCH_SPLIT_RECOMMEND_ITEMS,
        'command': command,
    }


def attach_submit_split_recommendation(manifest):
    recommendation = build_split_recommendation(manifest)
    if recommendation:
        manifest['split_recommended'] = True
        manifest['last_submit_quota_recommendation'] = recommendation
    else:
        manifest.pop('split_recommended', None)
        manifest.pop('last_submit_quota_recommendation', None)
    return recommendation


def _clear_submit_failure_metadata(manifest):
    manifest['last_submit_error'] = ''
    manifest.pop('last_submit_error_type', None)
    manifest.pop('split_recommended', None)
    manifest.pop('last_submit_quota_recommendation', None)


def print_submit_split_recommendation(recommendation):
    print('Quota/resource limit hit during batch submit.')
    if not recommendation:
        print('Wait for quota reset or retry with another API key before submitting again.')
        return
    print(
        f"Package size: {recommendation['chunk_count']} chunks, "
        f"{recommendation['item_count']} items."
    )
    print(f"Suggested split command: {recommendation['command']}")
    print('After splitting, continue from the first split manifest.')


def load_json_object_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def next_split_manifest_path(manifest):
    split_index = safe_nonnegative_int(manifest.get('split_index'))
    split_total = safe_nonnegative_int(manifest.get('split_total'))
    if not split_index or not split_total or split_index >= split_total:
        return ''

    current_path = manifest.get('_manifest_path')
    current_abs = _canonical_abs_path(current_path) if isinstance(current_path, str) and current_path else ''
    parent_path = manifest.get('split_from_manifest')
    children = []
    if isinstance(parent_path, str) and parent_path.strip() and os.path.isfile(parent_path):
        parent_manifest = load_json_object_file(parent_path)
        raw_children = parent_manifest.get('split_children')
        if isinstance(raw_children, list):
            children = [child for child in raw_children if isinstance(child, str) and child.strip()]

    candidate = ''
    if children:
        normalized_current = _normalized_abs_path(current_abs) if current_abs else ''
        for position, child in enumerate(children):
            if _normalized_abs_path(child) == normalized_current:
                if position + 1 < len(children):
                    candidate = children[position + 1]
                break
        if not candidate and split_index < len(children):
            candidate = children[split_index]

    if not candidate:
        package_dir = manifest.get('_package_dir')
        if not isinstance(package_dir, str) or not package_dir.strip():
            package_dir = os.path.dirname(current_abs)
        if package_dir:
            split_root = os.path.dirname(package_dir)
            candidate = os.path.join(
                split_root,
                f'part{split_index + 1:02d}_of_{split_total:02d}',
                'manifest.json',
            )

    if not candidate:
        return ''
    candidate = _canonical_abs_path(candidate)
    return candidate if os.path.isfile(candidate) else ''


def mark_next_split_after_apply(manifest):
    next_manifest = next_split_manifest_path(manifest)
    if next_manifest:
        manifest['next_split_manifest_path'] = next_manifest
        manifest['next_split_ready_at'] = datetime.now().isoformat(timespec='seconds')
    return next_manifest


def print_next_split_after_apply(next_manifest):
    if not next_manifest:
        return
    print(f'Next split manifest: {next_manifest}')
    print('Latest manifest set to next split package.')
    print('Run continue/status from the GUI to submit or monitor the next split package.')

def stable_json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def stable_json_sha256(value):
    return hashlib.sha256(stable_json_dumps(value).encode('utf-8')).hexdigest()


def file_content_fingerprint(path):
    digest = hashlib.sha256()
    row_count = 0
    size = 0
    with open(path, 'rb') as handle:
        for raw_line in handle:
            size += len(raw_line)
            digest.update(raw_line)
            if raw_line.strip():
                row_count += 1
    return {
        'path': os.path.abspath(path),
        'size': size,
        'sha256': digest.hexdigest(),
        'row_count': row_count,
    }


def manifest_target_shape(manifest):
    chunks = []
    item_count = 0
    for chunk in manifest.get('chunks') or []:
        items = []
        for item in chunk.get('items') or []:
            source_text = item.get('source', item.get('text', ''))
            items.append(
                {
                    'id': item.get('id', ''),
                    'file_rel_path': item.get('file_rel_path', chunk.get('file_rel_path', '')),
                    'line': item.get('line', item.get('line_number')),
                    'start': item.get('start'),
                    'end': item.get('end'),
                    'source_checksum': hash_text(source_text),
                }
            )
        item_count += len(items)
        chunks.append(
            {
                'key': chunk.get('key', ''),
                'file_rel_path': chunk.get('file_rel_path', ''),
                'chunk_index': chunk.get('chunk_index'),
                'item_count': len(items),
                'items': items,
            }
        )
    return {
        'chunk_count': len(chunks),
        'item_count': item_count,
        'chunk_keys': [chunk['key'] for chunk in chunks],
        'digest': stable_json_sha256(chunks),
    }


def build_check_fingerprint(manifest):
    result_path = resolve_manifest_result_path(manifest)
    package_dir = manifest.get('_package_dir', '')
    payload = {
        'check_contract_version': CHECK_CONTRACT_VERSION,
        'manifest_path': os.path.abspath(manifest.get('_manifest_path', '')) if manifest.get('_manifest_path') else '',
        'package_id': os.path.basename(os.path.abspath(package_dir)) if package_dir else '',
        'mode': manifest_mode(manifest),
        'manifest_version': manifest.get('manifest_version', 1),
        'core_schema_version': manifest.get('core_schema_version', 1),
        'batch_model': manifest.get('batch_model', ''),
        'settings': manifest.get('settings') or {},
        'result': file_content_fingerprint(result_path),
        'target_shape': manifest_target_shape(manifest),
    }
    payload['fingerprint_sha256'] = stable_json_sha256(payload)
    return payload


def check_fingerprint_id(fingerprint):
    if not isinstance(fingerprint, dict):
        return ''
    return fingerprint.get('fingerprint_sha256') or stable_json_sha256(fingerprint)


def safety_status_for_reason(reason_code):
    if reason_code in CHECK_BLOCK_REASON_CODES:
        return CHECK_SAFETY_BLOCK
    if reason_code in CHECK_WARN_REASON_CODES:
        return CHECK_SAFETY_WARN
    return CHECK_SAFETY_BLOCK


def summarize_check_safety(summary):
    reason_counts = summary.get('reason_counts') or {}
    warn_reasons = {}
    block_reasons = {}
    for reason_code, count in sorted(reason_counts.items()):
        if not count:
            continue
        status = safety_status_for_reason(reason_code)
        if status == CHECK_SAFETY_WARN:
            warn_reasons[reason_code] = count
        else:
            block_reasons[reason_code] = count

    if summary.get('failure_items', 0) and not warn_reasons and not block_reasons:
        block_reasons['unclassified_failure'] = summary.get('failure_items', 0)

    warn_count = sum(warn_reasons.values())
    block_count = sum(block_reasons.values())
    if block_count:
        level = CHECK_SAFETY_BLOCK
    elif warn_count:
        level = CHECK_SAFETY_WARN
    else:
        level = CHECK_SAFETY_SAFE

    return {
        'level': level,
        'counts': {
            'safe': summary.get('valid_items', 0),
            'warn': warn_count,
            'block': block_count,
        },
        'reasons': {
            CHECK_SAFETY_WARN: warn_reasons,
            CHECK_SAFETY_BLOCK: block_reasons,
        },
    }


def attach_check_contract(manifest, summary):
    safety = summarize_check_safety(summary)
    summary['check_contract_version'] = CHECK_CONTRACT_VERSION
    summary['check_fingerprint'] = build_check_fingerprint(manifest)
    summary['safety_level'] = safety['level']
    summary['safety_counts'] = safety['counts']
    summary['safety_reasons'] = safety['reasons']
    return summary


def infer_failure_reason_code(entry):
    reason_code = entry.get('reason_code')
    if isinstance(reason_code, str) and reason_code:
        return reason_code
    error = str(entry.get('error') or '').lower()
    if 'invalid result jsonl row' in error:
        return 'invalid_result_jsonl_row'
    if 'unknown chunk key' in error:
        return 'unknown_chunk_key'
    if 'missing text in response payload' in error:
        return 'missing_response_text'
    if 'failed to parse model json' in error:
        return 'failed_to_parse_model_json'
    if 'response missing item id' in error:
        return 'response_missing_item_id'
    if 'validation failed' in error:
        return 'validation_failed'
    if 'no result row found' in error:
        return 'missing_chunk_rows'
    if 'source line missing' in error:
        return 'source_line_missing'
    if 'source text mismatch' in error:
        return 'source_text_mismatch'
    if 'manifest file entry missing' in error:
        return 'missing_manifest_file'
    if 'target file missing' in error:
        return 'target_file_missing'
    if 'v2 relocation missing' in error:
        return 'v2_relocation_missing'
    if 'escapes' in error:
        return 'target_file_path_escaped'
    return 'unclassified_failure'


def annotate_failure_entries(entries):
    for entry in entries:
        reason_code = infer_failure_reason_code(entry)
        entry.setdefault('reason_code', reason_code)
        entry.setdefault('status', safety_status_for_reason(reason_code))
        if entry.get('id') and not entry.get('item_id'):
            entry['item_id'] = entry['id']
        if entry.get('text') and not entry.get('source_checksum'):
            entry['source_checksum'] = hash_text(entry.get('text', ''))
    return entries


def write_json_report(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl_report(path, entries):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


def write_check_failure_report(manifest, failure_entries):
    path = os.path.join(manifest.get('_package_dir', ''), 'check_failures.jsonl')
    write_jsonl_report(path, annotate_failure_entries(failure_entries))
    return path


def write_apply_failure_report(manifest, reason_code, message, summary=None, failure_entries=None, current_fingerprint=None):
    failure_entries = annotate_failure_entries(failure_entries or [])
    failures_path = os.path.join(manifest.get('_package_dir', ''), 'failures.jsonl')
    report_path = os.path.join(manifest.get('_package_dir', ''), 'apply_failure_report.json')
    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'status': CHECK_SAFETY_BLOCK,
        'reason_code': reason_code,
        'error': message,
        'manifest_path': manifest.get('_manifest_path', ''),
        'last_check_at': manifest.get('last_check_at', ''),
        'last_check_safety_level': (manifest.get('last_check_summary') or {}).get('safety_level', ''),
        'current_check_fingerprint': current_fingerprint or {},
        'summary': summary or {},
        'failure_count': len(failure_entries),
        'failures_path': failures_path if failure_entries else '',
    }
    write_json_report(report_path, payload)
    manifest['last_apply_failure_report_path'] = report_path
    return report_path


def fail_apply_preflight(manifest, reason_code, message, current_fingerprint=None):
    report_path = write_apply_failure_report(
        manifest,
        reason_code,
        message,
        current_fingerprint=current_fingerprint,
    )
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    raise SystemExit(f'{message} Report: {report_path}')


def require_safe_check_for_apply(manifest):
    last_summary = manifest.get('last_check_summary')
    if not isinstance(last_summary, dict):
        fail_apply_preflight(
            manifest,
            'missing_check',
            'Manifest has no valid check summary. Run check before apply.',
        )
    if last_summary.get('check_contract_version') != CHECK_CONTRACT_VERSION:
        fail_apply_preflight(
            manifest,
            'stale_check_contract',
            'Manifest check summary was produced by an older check contract. Run check again before apply.',
        )

    checked_fingerprint = last_summary.get('check_fingerprint')
    current_fingerprint = build_check_fingerprint(manifest)
    if check_fingerprint_id(checked_fingerprint) != check_fingerprint_id(current_fingerprint):
        fail_apply_preflight(
            manifest,
            'stale_check_fingerprint',
            'Manifest or results changed after the last check. Run check again before apply.',
            current_fingerprint=current_fingerprint,
        )

    safety_level = last_summary.get('safety_level')
    if safety_level != CHECK_SAFETY_SAFE:
        reason_code = 'unsafe_check_status'
        message = (
            f'Last check safety status is {safety_level or "unknown"}, not safe. '
            'Repair the results or run check again before apply.'
        )
        fail_apply_preflight(manifest, reason_code, message, current_fingerprint=current_fingerprint)




def load_request_rows(manifest):
    input_jsonl_path = manifest.get('input_jsonl_path')
    if not input_jsonl_path or not os.path.isfile(input_jsonl_path):
        raise SystemExit(f"Input JSONL not found: {input_jsonl_path}")
    rows = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_files_for_chunks(chunks):
    files = {}
    for chunk in chunks:
        rel_path = chunk['file_rel_path']
        if rel_path not in files:
            files[rel_path] = {
                'path': chunk['file_path'],
                'task_count': 0,
            }
        files[rel_path]['task_count'] += len(chunk.get('items', []))
    return files


def create_batch_package_dir(package_name):
    base_dir = os.path.join(BATCH_JOBS_DIR, package_name)
    candidates = [base_dir]
    candidates.extend(f'{base_dir}_{index:02d}' for index in range(1, 1000))
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f'Could not create unique batch package directory for {package_name}.')


def copy_split_context_metadata(source_manifest, part_manifest, part_chunks):
    if source_manifest.get('keyword_settings'):
        part_manifest['keyword_settings'] = dict(source_manifest.get('keyword_settings') or {})
    if source_manifest.get('revision_settings'):
        part_manifest['revision_settings'] = dict(source_manifest.get('revision_settings') or {})

    for key in ('rag_enabled', 'rag_store_path', 'rag_settings'):
        if key in source_manifest:
            part_manifest[key] = source_manifest[key]

    if source_manifest.get('rag_enabled'):
        source_rag_summary = source_manifest.get('rag_summary') or {}
        part_manifest['rag_summary'] = summarize_batch_rag(
            part_chunks,
            dict(source_rag_summary.get('prepare') or {}),
        )

    for key in (
        'story_memory_enabled',
        'story_memory_graph_file',
        'story_memory_settings',
    ):
        if key in source_manifest:
            part_manifest[key] = source_manifest[key]

    if source_manifest.get('story_memory_enabled'):
        story_settings = source_manifest.get('story_memory_settings') or {}
        part_manifest['story_memory_summary'] = summarize_batch_story_memory(
            part_chunks,
            graph_file=source_manifest.get('story_memory_graph_file', ''),
            max_context_chars=story_settings.get('max_context_chars'),
        )

    for key in (
        'source_index_enabled',
        'source_index_store_path',
        'source_index_settings',
    ):
        if key in source_manifest:
            part_manifest[key] = source_manifest[key]

    if source_manifest.get('source_index_enabled'):
        part_manifest['source_index_summary'] = summarize_batch_source_index(part_chunks)


def split_chunks_and_lines(chunks, request_lines, max_chunks=0, max_items=0):
    groups = []
    current_chunks = []
    current_lines = []
    current_item_count = 0

    for chunk, line in zip(chunks, request_lines):
        chunk_item_count = len(chunk.get('items', []))
        should_flush = False

        if current_chunks:
            if max_chunks and len(current_chunks) >= max_chunks:
                should_flush = True
            elif max_items and current_item_count + chunk_item_count > max_items:
                should_flush = True

        if should_flush:
            groups.append((current_chunks, current_lines))
            current_chunks = []
            current_lines = []
            current_item_count = 0

        current_chunks.append(chunk)
        current_lines.append(line)
        current_item_count += chunk_item_count

    if current_chunks:
        groups.append((current_chunks, current_lines))

    return groups

def get_state_name(state):
    if state is None:
        return ''
    name = getattr(state, 'name', None)
    return name or str(state)


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
    for method_name in ('model_dump', 'dict'):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return serialize_unknown(method())
            except Exception:
                pass
    if hasattr(value, '__dict__'):
        return serialize_unknown(vars(value))
    return str(value)


def extract_batch_stats(batch_job):
    stats = get_nested(batch_job, 'batch_stats', 'batchStats')
    if not stats:
        return {}
    result = {}
    for key in ('request_count', 'successful_request_count', 'failed_request_count', 'pending_request_count'):
        camel = ''.join(part.capitalize() if idx else part for idx, part in enumerate(key.split('_')))
        value = get_nested(stats, key, camel)
        if value is not None:
            result[key] = value
    return result


def write_status_snapshot(manifest, batch_job):
    snapshot_path = os.path.join(manifest['_package_dir'], 'last_status_snapshot.json')
    payload = {
        'checked_at': datetime.now().isoformat(timespec='seconds'),
        'job_state': get_state_name(getattr(batch_job, 'state', None)),
        'job_error': serialize_unknown(get_nested(batch_job, 'error')),
        'batch_stats': extract_batch_stats(batch_job),
        'job': serialize_unknown(batch_job),
    }
    with open(snapshot_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    manifest['last_status_snapshot_path'] = snapshot_path

def collect_files_to_process():
    files_to_process = []
    for root, _, files in os.walk(legacy.TL_DIR):
        for file_name in files:
            if not file_name.endswith('.rpy'):
                continue
            file_path = os.path.join(root, file_name)
            rel_path = legacy._normalize_rel_path(os.path.relpath(file_path, legacy.TL_DIR))
            if legacy.INCLUDE_FILES or legacy.INCLUDE_PREFIXES:
                allowed = False
                if legacy.INCLUDE_FILES and rel_path in legacy.INCLUDE_FILES:
                    allowed = True
                if not allowed and legacy.INCLUDE_PREFIXES:
                    for prefix in legacy.INCLUDE_PREFIXES:
                        if rel_path.startswith(prefix):
                            allowed = True
                            break
                if not allowed:
                    continue
            files_to_process.append((rel_path, file_path))
    files_to_process.sort(key=lambda item: item[0])
    return files_to_process


def collect_pending_file_jobs():
    jobs = []

    for rel_path, file_path in collect_files_to_process():
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()

        raw_tasks = legacy.collect_tasks(lines)
        pending = []

        for task in raw_tasks:
            if legacy.is_non_translatable(task['text']):
                continue
            current = dict(task)
            current['file_rel_path'] = rel_path
            current['file_path'] = file_path
            current['id'] = translation_core.build_identity_v2(
                rel_path,
                current.get('block_name', '_global'),
                current.get('block_index', 0),
                current.get('source_for_id') or current['text'],
                block_occurrence=current.get('block_occurrence', 1),
            )
            pending.append(current)

        if pending:
            jobs.append(
                {
                    'file_rel_path': rel_path,
                    'file_path': file_path,
                    'task_count': len(pending),
                    'tasks': pending,
                }
            )

    return jobs


def format_context_block(lines, empty_label):
    return translation_core.format_context_block(lines, empty_label)


def build_system_instruction():
    return translation_core.build_translation_system_instruction(
        legacy.PRESERVE_TERMS,
        macro_setting=BATCH_MACRO_SETTING,
    )


def build_user_prompt(
    context_past,
    target_items,
    context_future,
    glossary_hits=None,
    history_hits=None,
    story_hits=None,
    source_hits=None,
):
    return translation_core.build_translation_user_prompt(
        translation_core.ContextWindow(context_past, context_future),
        target_items,
        translation_core.build_context_bundle(
            glossary_hits=glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
            source_hits=source_hits,
        ),
        history_char_limit=RAG_HISTORY_CHAR_LIMIT,
        story_char_limit=STORY_MEMORY_MAX_CONTEXT_CHARS,
        include_translation_memory=True,
        include_source_text=True,
    )



def build_response_json_schema(target_items):
    return translation_core.build_response_json_schema(
        target_items,
        mode=translation_core.MODE_TRANSLATION,
    )


def build_generation_config(target_items):
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_response_json_schema(target_items),
    }
    if BATCH_THINKING_LEVEL and BATCH_MODEL.startswith('gemini-3'):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return config


def build_batch_request(chunk):
    request = {
        'system_instruction': {'parts': [{'text': build_system_instruction()}]},
        'contents': [
            {
                'role': 'user',
                'parts': [
                    {
                        'text': build_user_prompt(
                            chunk['context_past'],
                            chunk['items'],
                            chunk['context_future'],
                            glossary_hits=chunk.get('glossary_hits') or [],
                            history_hits=chunk.get('history_hits') or [],
                            story_hits=chunk.get('story_hits') if 'story_hits' in chunk else None,
                            source_hits=chunk.get('source_hits') or [],
                        )
                    }
                ],
            }
        ],
        'generation_config': build_generation_config(chunk['items']),
    }
    if BATCH_SAFETY_SETTINGS:
        request['safety_settings'] = BATCH_SAFETY_SETTINGS
    return {
        'key': chunk['key'],
        'request': request,
    }

def task_text_char_count(task):
    text = task.get('text', '') if isinstance(task, dict) else ''
    return len(text) if isinstance(text, str) else len(str(text))


def iter_translation_chunk_ranges(tasks):
    total = len(tasks)
    start = 0
    while start < total:
        end = start
        current_chars = 0
        while end < total and (end - start) < BATCH_TARGET_SIZE:
            item_chars = task_text_char_count(tasks[end])
            if end > start and current_chars + item_chars > BATCH_TARGET_CHARS:
                break
            current_chars += item_chars
            end += 1
        if end == start:
            end = start + 1
        yield start, end
        start = end


def count_translation_chunks(file_jobs):
    total_chunks = 0
    for job in file_jobs:
        total_chunks += sum(1 for _ in iter_translation_chunk_ranges(job.get('tasks') or []))
    return total_chunks


def build_chunks(file_jobs):
    chunks = []
    total_chunks = count_translation_chunks(file_jobs)
    processed_chunks = 0
    if SOURCE_INDEX_ENABLED and total_chunks:
        print(f'Source index retrieval for build: {total_chunks} chunks to query.')
        sys.stdout.flush()
    for job in file_jobs:
        tasks = job['tasks']
        total = len(tasks)
        for chunk_number, (start, end) in enumerate(iter_translation_chunk_ranges(tasks), start=1):
            target_items = tasks[start:end]
            target_units = translation_core.units_from_items(
                target_items,
                translation_core.MODE_TRANSLATION,
                file_rel_path=job['file_rel_path'],
                file_path=job['file_path'],
            )
            context_past = tasks[max(0, start - BATCH_CONTEXT_BEFORE):start]
            context_future = tasks[end:min(total, end + BATCH_CONTEXT_AFTER)]
            glossary_hits = retrieve_glossary_hits(target_items) if RAG_ENABLED else []
            history_hits, rag_stats = retrieve_history_hits(target_items, context_past) if RAG_ENABLED else ([], {})
            if SOURCE_INDEX_ENABLED:
                print(
                    'Source index retrieval progress: '
                    f'{processed_chunks + 1}/{total_chunks} chunks, '
                    f'file={job["file_rel_path"]}, chunk={chunk_number}.'
                )
                sys.stdout.flush()
            source_hits, source_index_stats = retrieve_source_hits(target_items, context_past) if SOURCE_INDEX_ENABLED else ([], {})
            processed_chunks += 1
            story_hits = retrieve_batch_story_hits(
                job['file_rel_path'],
                target_items,
                context_past,
                context_future,
            ) if STORY_MEMORY_ENABLED else None
            chunk_key = f"{hash_key(job['file_rel_path'])}-{chunk_number:05d}"
            chunk = {
                'key': chunk_key,
                'mode': MANIFEST_MODE_TRANSLATION,
                'file_rel_path': job['file_rel_path'],
                'file_path': job['file_path'],
                'chunk_index': chunk_number,
                'line_numbers': [unit.line for unit in target_units],
                'source_char_count': sum(task_text_char_count(item) for item in target_items),
                'context_past': context_past,
                'context_future': context_future,
                'glossary_hits': glossary_hits,
                'history_hits': history_hits,
                'rag_stats': rag_stats,
                'source_hits': source_hits,
                'source_index_stats': source_index_stats,
                'items': [
                    translation_core.legacy_item_from_unit(unit, translation_core.MODE_TRANSLATION)
                    for unit in target_units
                ],
            }
            if STORY_MEMORY_ENABLED and story_memory.has_story_hits(story_hits):
                chunk['story_hits'] = story_hits
            chunks.append(chunk)
    if SOURCE_INDEX_ENABLED and total_chunks:
        print(f'Source index retrieval complete: {processed_chunks}/{total_chunks} chunks queried.')
        sys.stdout.flush()
    return chunks


def summarize_batch_rag(chunks, prepare_summary):
    chunk_count = len(chunks)
    chunks_with_history_hits = sum(1 for chunk in chunks if chunk.get('history_hits'))
    history_hit_count = sum(len(chunk.get('history_hits') or []) for chunk in chunks)
    return {
        'prepare': prepare_summary,
        'chunks_with_glossary_hits': sum(1 for chunk in chunks if chunk.get('glossary_hits')),
        'chunks_with_history_hits': chunks_with_history_hits,
        'history_hit_count': history_hit_count,
        'history_hit_rate': (chunks_with_history_hits / chunk_count) if chunk_count else 0.0,
        'history_retrieval_errors': sum(
            1 for chunk in chunks
            if (chunk.get('rag_stats') or {}).get('error')
        ),
    }


def summarize_batch_source_index(chunks):
    chunk_count = len(chunks)
    chunks_with_source_hits = sum(1 for chunk in chunks if chunk.get('source_hits'))
    source_hit_count = sum(len(chunk.get('source_hits') or []) for chunk in chunks)
    stats_list = [chunk.get('source_index_stats') or {} for chunk in chunks]
    source_retrieval_errors = sum(1 for stats in stats_list if stats.get('error'))
    failure_reasons = {}
    for stats in stats_list:
        reason = stats.get('failure_reason') or stats.get('reason')
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    store_schema_versions = sorted(
        {
            stats.get('store_schema_version')
            for stats in stats_list
            if stats.get('store_schema_version') is not None
        },
        key=str,
    )
    return {
        'enabled': SOURCE_INDEX_ENABLED,
        'store_dir': SOURCE_INDEX_STORE_DIR or get_default_source_index_store_dir(),
        'schema_version': SOURCE_INDEX_SCHEMA_VERSION,
        'store_schema_versions': store_schema_versions,
        'per_chunk_char_budget': get_source_index_char_budget(),
        'chunks_with_source_hits': chunks_with_source_hits,
        'source_hit_count': source_hit_count,
        'source_hit_rate': (chunks_with_source_hits / chunk_count) if chunk_count else 0.0,
        'source_retrieval_errors': source_retrieval_errors,
        'source_retrieval_failure_reasons': failure_reasons,
        'source_context_truncation_count': sum(int(stats.get('truncated_count') or 0) for stats in stats_list),
        'source_context_char_count': sum(int(stats.get('source_context_chars') or 0) for stats in stats_list),
        'source_context_char_budget': sum(int(stats.get('source_context_char_budget') or 0) for stats in stats_list),
        'source_filtered_count': sum(int(stats.get('filtered_count') or 0) for stats in stats_list),
        'stale_hits_skipped': sum(int(stats.get('stale_hits_skipped') or 0) for stats in stats_list),
        'below_similarity_count': sum(int(stats.get('below_similarity_count') or 0) for stats in stats_list),
    }


def summarize_batch_story_memory(chunks, graph_file=None, max_context_chars=None):
    chunk_count = len(chunks)
    hit_counts = {key: 0 for key in story_memory.STORY_HIT_CATEGORIES}
    chunks_with_story_hits = 0
    truncated_story_blocks = 0
    formatted_char_count = 0
    requested_limit = max_context_chars if max_context_chars is not None else STORY_MEMORY_MAX_CONTEXT_CHARS
    try:
        context_limit = max(1, int(requested_limit or 1))
    except (TypeError, ValueError):
        try:
            context_limit = max(1, int(STORY_MEMORY_MAX_CONTEXT_CHARS or 1))
        except (TypeError, ValueError):
            context_limit = 1

    for chunk in chunks:
        story_hits = chunk.get('story_hits')
        if not story_memory.has_story_hits(story_hits):
            continue
        chunks_with_story_hits += 1
        chunk_counts = story_memory.story_hit_counts(story_hits)
        for key in hit_counts:
            hit_counts[key] += chunk_counts.get(key, 0)
        formatted_block = story_memory.format_story_hits_block(story_hits, context_limit)
        over_limit_probe = story_memory.format_story_hits_block(story_hits, context_limit + 1)
        formatted_char_count += len(formatted_block)
        if len(over_limit_probe) > context_limit:
            truncated_story_blocks += 1

    return {
        'graph_file': STORY_MEMORY_GRAPH_FILE if graph_file is None else graph_file,
        'chunks_with_story_hits': chunks_with_story_hits,
        'story_hit_rate': (chunks_with_story_hits / chunk_count) if chunk_count else 0.0,
        'hit_counts': hit_counts,
        'total_hit_count': sum(hit_counts.values()),
        'truncated_story_blocks': truncated_story_blocks,
        'formatted_char_count': formatted_char_count,
    }



def get_batch_risk_warnings():
    warnings_list = []
    if BATCH_TARGET_SIZE > 80:
        warnings_list.append(f'chunk_size={BATCH_TARGET_SIZE} is aggressive for Gemini 3 Flash structured output.')
    if BATCH_CONTEXT_BEFORE > 40 or BATCH_CONTEXT_AFTER > 20:
        warnings_list.append(
            f'context_before/context_after ({BATCH_CONTEXT_BEFORE}/{BATCH_CONTEXT_AFTER}) may inflate prompt tokens.'
        )
    if BATCH_MAX_OUTPUT_TOKENS < 2048:
        warnings_list.append(f'max_output_tokens={BATCH_MAX_OUTPUT_TOKENS} is likely too low for JSON batch output.')
    if BATCH_MODEL.startswith('gemini-3') and BATCH_THINKING_LEVEL and BATCH_THINKING_LEVEL.lower() != 'minimal':
        warnings_list.append(
            f'thinking_level={BATCH_THINKING_LEVEL} may waste output budget on reasoning tokens.'
        )
    return warnings_list


def create_batch_package(display_name_override='', skip_prepare=False):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_pending_file_jobs()
    if not file_jobs:
        print('No pending lines to translate.')
        return None

    rag_prepare_summary = prepare_rag_store(file_jobs)

    chunks = build_chunks(file_jobs)
    if not chunks:
        print('No chunks built.')
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    package_name = f'{timestamp}_{guess_project_slug()}'
    package_dir = os.path.join(BATCH_JOBS_DIR, package_name)
    os.makedirs(package_dir, exist_ok=True)

    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'{BATCH_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'

    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    with open(input_jsonl_path, 'w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(build_batch_request(chunk), ensure_ascii=False) + '\n')

    build_warnings = get_batch_risk_warnings()

    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': MANIFEST_MODE_TRANSLATION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': {
            'target_size': BATCH_TARGET_SIZE,
            'target_chars': BATCH_TARGET_CHARS,
            'context_before': BATCH_CONTEXT_BEFORE,
            'context_after': BATCH_CONTEXT_AFTER,
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        'build_warnings': build_warnings,
        'rag_enabled': RAG_ENABLED,
        'rag_store_path': RAG_STORE_DIR if RAG_ENABLED else '',
        'rag_settings': {
            'embedding_model': RAG_EMBEDDING_MODEL,
            'query_task_type': RAG_QUERY_TASK_TYPE,
            'document_task_type': RAG_DOCUMENT_TASK_TYPE,
            'output_dimensionality': RAG_OUTPUT_DIMENSIONALITY,
            'top_k_history': RAG_TOP_K_HISTORY,
            'top_k_terms': RAG_TOP_K_TERMS,
            'min_similarity': RAG_MIN_SIMILARITY,
            'segment_lines': RAG_SEGMENT_LINES,
            'bootstrap_on_build': RAG_BOOTSTRAP_ON_BUILD,
        } if RAG_ENABLED else {},
        'rag_summary': summarize_batch_rag(chunks, rag_prepare_summary) if RAG_ENABLED else {},
        'source_index_enabled': SOURCE_INDEX_ENABLED,
        'source_index_store_path': (SOURCE_INDEX_STORE_DIR or get_default_source_index_store_dir()) if SOURCE_INDEX_ENABLED else '',
        'source_index_settings': {
            'schema_version': SOURCE_INDEX_SCHEMA_VERSION,
            'top_k': SOURCE_INDEX_TOP_K,
            'min_similarity': SOURCE_INDEX_MIN_SIMILARITY,
            'char_limit': SOURCE_INDEX_CHAR_LIMIT,
            'char_budget_per_chunk': get_source_index_char_budget(),
        } if SOURCE_INDEX_ENABLED else {},
        'source_index_summary': summarize_batch_source_index(chunks) if SOURCE_INDEX_ENABLED else {},
        'story_memory_enabled': STORY_MEMORY_ENABLED,
        'story_memory_graph_file': STORY_MEMORY_GRAPH_FILE if STORY_MEMORY_ENABLED else '',
        'story_memory_settings': {
            'max_context_chars': STORY_MEMORY_MAX_CONTEXT_CHARS,
            'top_k_relations': STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_terms': STORY_MEMORY_TOP_K_TERMS,
            'include_scene_summary': STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
        } if STORY_MEMORY_ENABLED else {},
        'story_memory_summary': summarize_batch_story_memory(chunks) if STORY_MEMORY_ENABLED else {},
        'summary': {
            'file_count': len(file_jobs),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk['items']) for chunk in chunks),
        },
        'files': {
            job['file_rel_path']: {
                'path': job['file_path'],
                'task_count': job['task_count'],
            }
            for job in file_jobs
        },
        'chunks': chunks,
    }

    manifest_path = os.path.join(package_dir, 'manifest.json')
    manifest['_manifest_path'] = manifest_path
    try:
        cost_estimate = batch_cost_estimate.attach_cost_estimate_to_manifest(
            manifest,
            translator_config=load_json_file(legacy.TRANSLATOR_CONFIG),
        )
    except Exception as exc:
        cost_estimate = None
        build_warnings.append(f'Cost estimate unavailable: {exc}')
        manifest['build_warnings'] = build_warnings

    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    remember_latest_manifest(manifest_path)

    print(f'Created batch package: {package_dir}')
    print(f"TL subdir: {manifest['tl_subdir']}")
    print(f"Target language: {manifest['target_language']}")
    print(f"Pending files: {manifest['summary']['file_count']}")
    print(f"Chunks: {manifest['summary']['chunk_count']}")
    print(f"Items: {manifest['summary']['item_count']}")
    if cost_estimate:
        for line in batch_cost_estimate.format_cost_estimate_lines(cost_estimate):
            print(line)
    if build_warnings:
        print('Warnings:')
        for warning_text in build_warnings:
            print(f'- {warning_text}')
    return manifest_path




def should_include_revision_entry(entry):
    source = compact_text(entry.get('source', ''))
    current_translation = compact_text(entry.get('translation', ''))
    return bool(source and current_translation)


def collect_revision_file_jobs():
    jobs = []
    for rel_path, file_path in collect_files_to_process():
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            entries = collect_translation_entries_from_lines(handle.readlines(), file_rel_path=rel_path)

        items = []
        for entry in entries:
            if not should_include_revision_entry(entry):
                continue
            try:
                line_number = int(entry.get('line_number'))
            except (TypeError, ValueError):
                line_number = 0
            line_index = max(0, line_number - 1)
            start = int(entry.get('start', 0) or 0)
            item = {
                'id': (
                    entry.get('identity_v2')
                    or f"{rel_path}:{line_index}:{start}:revision:{entry.get('entry_index', len(items))}"
                ),
                'text': entry.get('source', ''),
                'source': entry.get('source', ''),
                'current_translation': entry.get('translation', ''),
                'file_rel_path': rel_path,
                'line': line_index,
                'line_number': line_number,
                'start': start,
                'end': int(entry.get('end', 0) or 0),
                'prefix': entry.get('prefix', ''),
                'quote': entry.get('quote', '"'),
            }
            speaker_id = entry.get('speaker_id') or entry.get('speaker')
            if speaker_id:
                item['speaker_id'] = speaker_id
            items.append(item)

        if items:
            jobs.append(
                {
                    'file_rel_path': rel_path,
                    'file_path': file_path,
                    'task_count': len(items),
                    'items': items,
                }
            )
    return jobs


def format_revision_context_block(items, empty_label):
    return translation_core.format_revision_context_block(items, empty_label)


def build_revision_chunks(file_jobs, chunk_size=None):
    chunk_size = max(1, int(chunk_size or REVISION_CHUNK_SIZE))
    chunks = []
    for job in file_jobs:
        items = job['items']
        total = len(items)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            target_items = items[start:end]
            target_units = translation_core.units_from_items(
                target_items,
                translation_core.MODE_REVISION,
                file_rel_path=job['file_rel_path'],
                file_path=job['file_path'],
            )
            context_past_items = items[max(0, start - BATCH_CONTEXT_BEFORE):start]
            context_future_items = items[end:min(total, end + BATCH_CONTEXT_AFTER)]
            glossary_hits = retrieve_glossary_hits(target_items) if RAG_ENABLED else []
            history_hits, rag_stats = retrieve_history_hits(
                target_items,
                [item.get('source', '') for item in context_past_items],
            ) if RAG_ENABLED else ([], {})
            story_hits = retrieve_batch_story_hits(
                job['file_rel_path'],
                target_items,
                [item.get('source', '') for item in context_past_items],
                [item.get('source', '') for item in context_future_items],
            ) if STORY_MEMORY_ENABLED else None
            chunk_number = start // chunk_size + 1
            chunk = {
                'key': f"rv-{hash_key(job['file_rel_path'])}-{chunk_number:05d}",
                'mode': MANIFEST_MODE_REVISION,
                'file_rel_path': job['file_rel_path'],
                'file_path': job['file_path'],
                'chunk_index': chunk_number,
                'line_numbers': [item.get('line_number', 0) for item in target_items],
                'context_past': [
                    {
                        'source': item.get('source', ''),
                        'current_translation': item.get('current_translation', ''),
                    }
                    for item in context_past_items
                ],
                'context_future': [
                    {
                        'source': item.get('source', ''),
                        'current_translation': item.get('current_translation', ''),
                    }
                    for item in context_future_items
                ],
                'glossary_hits': glossary_hits,
                'history_hits': history_hits,
                'rag_stats': rag_stats,
                'items': [
                    translation_core.legacy_item_from_unit(unit, translation_core.MODE_REVISION)
                    for unit in target_units
                ],
            }
            if STORY_MEMORY_ENABLED and story_memory.has_story_hits(story_hits):
                chunk['story_hits'] = story_hits
            chunks.append(chunk)
    return chunks


def build_revision_system_instruction():
    return translation_core.build_revision_system_instruction(
        legacy.PRESERVE_TERMS,
        macro_setting=BATCH_MACRO_SETTING,
    )


def build_revision_user_prompt(chunk):
    return translation_core.build_revision_user_prompt(
        translation_core.ContextWindow(
            chunk.get('context_past') or [],
            chunk.get('context_future') or [],
        ),
        translation_core.units_from_items(
            chunk['items'],
            translation_core.MODE_REVISION,
            file_rel_path=chunk.get('file_rel_path', ''),
            file_path=chunk.get('file_path', ''),
        ),
        translation_core.build_context_bundle(
            glossary_hits=chunk.get('glossary_hits') or [],
            history_hits=chunk.get('history_hits') or [],
            story_hits=chunk.get('story_hits'),
            rag_stats=chunk.get('rag_stats') or {},
        ),
        history_char_limit=RAG_HISTORY_CHAR_LIMIT,
        story_char_limit=STORY_MEMORY_MAX_CONTEXT_CHARS,
        include_source_text=True,
    )


def build_revision_response_json_schema(target_items):
    return translation_core.build_response_json_schema(
        target_items,
        mode=translation_core.MODE_REVISION,
    )


def build_revision_generation_config(target_items):
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_revision_response_json_schema(target_items),
    }
    if BATCH_THINKING_LEVEL and BATCH_MODEL.startswith('gemini-3'):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return config


def build_revision_request(chunk):
    return {
        'key': chunk['key'],
        'request': {
            'system_instruction': {'parts': [{'text': build_revision_system_instruction()}]},
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': build_revision_user_prompt(chunk)}],
                }
            ],
            'generation_config': build_revision_generation_config(chunk['items']),
        },
    }


def create_revision_package(display_name_override='', skip_prepare=False, chunk_size=None):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_revision_file_jobs()
    if not file_jobs:
        print('No revision source lines found.')
        return None

    chunk_size = max(1, int(chunk_size or REVISION_CHUNK_SIZE))
    rag_prepare_summary = prepare_rag_store(file_jobs)
    chunks = build_revision_chunks(file_jobs, chunk_size=chunk_size)
    if not chunks:
        print('No revision chunks built.')
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_name = f'{timestamp}_{guess_project_slug()}_revisions'
    package_dir = create_batch_package_dir(package_name)

    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'{REVISION_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'

    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    with open(input_jsonl_path, 'w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(build_revision_request(chunk), ensure_ascii=False) + '\n')

    build_warnings = get_batch_risk_warnings()
    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': MANIFEST_MODE_REVISION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': {
            'revision_chunk_size': chunk_size,
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        'revision_settings': {
            'chunk_size': chunk_size,
        },
        'summary': {
            'file_count': len(file_jobs),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk['items']) for chunk in chunks),
        },
        'files': {
            job['file_rel_path']: {
                'path': job['file_path'],
                'task_count': job['task_count'],
            }
            for job in file_jobs
        },
        'chunks': chunks,
        'build_warnings': build_warnings,
    }

    if RAG_ENABLED:
        manifest['rag_enabled'] = True
        manifest['rag_store_path'] = RAG_STORE_DIR or get_default_rag_store_dir()
        manifest['rag_settings'] = {
            'top_k_history': RAG_TOP_K_HISTORY,
            'top_k_terms': RAG_TOP_K_TERMS,
            'min_similarity': RAG_MIN_SIMILARITY,
            'segment_lines': RAG_SEGMENT_LINES,
        }
        manifest['rag_summary'] = summarize_batch_rag(chunks, rag_prepare_summary)
    if STORY_MEMORY_ENABLED:
        manifest['story_memory_enabled'] = True
        manifest['story_memory_graph_file'] = STORY_MEMORY_GRAPH_FILE
        manifest['story_memory_settings'] = {
            'top_k_terms': STORY_MEMORY_TOP_K_TERMS,
            'top_k_characters': STORY_MEMORY_TOP_K_CHARACTERS,
            'top_k_relations': STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_scenes': STORY_MEMORY_TOP_K_SCENES,
            'max_context_chars': STORY_MEMORY_MAX_CONTEXT_CHARS,
        }
        manifest['story_memory_summary'] = summarize_batch_story_memory(chunks)

    manifest_path = os.path.join(package_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    remember_latest_manifest(manifest_path)

    print(f'Created revision package: {package_dir}')
    print(f"Source files: {manifest['summary']['file_count']}")
    print(f"Chunks: {manifest['summary']['chunk_count']}")
    print(f"Revision items: {manifest['summary']['item_count']}")
    print('Mode: revision')
    if build_warnings:
        print('Warnings:')
        for warning_text in build_warnings:
            print(f'- {warning_text}')
    return manifest_path


def should_include_keyword_source(text):
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalnum() or '\u4e00' <= ch <= '\u9fff' for ch in stripped)


def keyword_source_line_number(entry):
    for key in ('source_line_number', 'line_number'):
        try:
            value = int(entry.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def collect_keyword_file_jobs():
    jobs = []
    for rel_path, file_path in collect_files_to_process():
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            entries = collect_repair_entries_from_lines(handle.readlines())

        items = []
        for entry in entries:
            source_text = str(entry.get('source') or '').strip()
            if not should_include_keyword_source(source_text):
                continue
            line_number = keyword_source_line_number(entry)
            item = {
                'id': f"{rel_path}:{line_number}:keyword:{entry.get('entry_index', len(items))}",
                'text': source_text,
                'file_rel_path': rel_path,
                'line_number': line_number,
                'translation_line_number': entry.get('line_number', 0),
            }
            speaker_id = entry.get('speaker_id') or entry.get('speaker')
            if speaker_id:
                item['speaker_id'] = speaker_id
            items.append(item)

        if items:
            jobs.append(
                {
                    'file_rel_path': rel_path,
                    'file_path': file_path,
                    'task_count': len(items),
                    'items': items,
                }
            )
    return jobs


def build_keyword_chunks(file_jobs, chunk_size=None):
    chunk_size = max(1, int(chunk_size or KEYWORD_CHUNK_SIZE))
    chunks = []
    for job in file_jobs:
        items = job['items']
        for start in range(0, len(items), chunk_size):
            target_items = items[start:start + chunk_size]
            target_units = translation_core.units_from_items(
                target_items,
                translation_core.MODE_KEYWORD_EXTRACTION,
                file_rel_path=job['file_rel_path'],
                file_path=job['file_path'],
            )
            chunk_number = start // chunk_size + 1
            chunks.append(
                {
                    'key': f"kw-{hash_key(job['file_rel_path'])}-{chunk_number:05d}",
                    'mode': MANIFEST_MODE_KEYWORD_EXTRACTION,
                    'file_rel_path': job['file_rel_path'],
                    'file_path': job['file_path'],
                    'chunk_index': chunk_number,
                    'line_numbers': [unit.display_line_number for unit in target_units],
                    'items': [
                        translation_core.legacy_item_from_unit(
                            unit,
                            translation_core.MODE_KEYWORD_EXTRACTION,
                        )
                        for unit in target_units
                    ],
                }
            )
    return chunks


def format_keyword_glossary_block():
    return translation_core.build_keyword_glossary_block(
        legacy.PRESERVE_TERMS,
        legacy.NORMALIZE_TRANSLATION_MAP,
        getattr(legacy, 'NON_TRANSLATABLE_EXACT', set()),
    )


def build_keyword_system_instruction(max_candidates_per_chunk=None):
    return translation_core.build_keyword_system_instruction(
        legacy.PRESERVE_TERMS,
        legacy.NORMALIZE_TRANSLATION_MAP,
        getattr(legacy, 'NON_TRANSLATABLE_EXACT', set()),
        macro_setting=BATCH_MACRO_SETTING,
        max_candidates_per_chunk=max_candidates_per_chunk or KEYWORD_MAX_CANDIDATES_PER_CHUNK,
    )


def build_keyword_user_prompt(target_items):
    return translation_core.build_keyword_user_prompt(target_items)


def build_keyword_response_json_schema(max_candidates_per_chunk=None):
    return translation_core.build_response_json_schema(
        mode=translation_core.MODE_KEYWORD_EXTRACTION,
        max_candidates_per_chunk=max_candidates_per_chunk or KEYWORD_MAX_CANDIDATES_PER_CHUNK,
    )


def build_keyword_generation_config(max_candidates_per_chunk=None):
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_keyword_response_json_schema(max_candidates_per_chunk),
    }
    if BATCH_THINKING_LEVEL and BATCH_MODEL.startswith('gemini-3'):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return config


def build_keyword_request(chunk, max_candidates_per_chunk=None):
    return {
        'key': chunk['key'],
        'request': {
            'system_instruction': {
                'parts': [{'text': build_keyword_system_instruction(max_candidates_per_chunk)}],
            },
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': build_keyword_user_prompt(chunk['items'])}],
                }
            ],
            'generation_config': build_keyword_generation_config(max_candidates_per_chunk),
        },
    }


def create_keyword_package(display_name_override='', skip_prepare=True, chunk_size=None, max_candidates_per_chunk=None):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_keyword_file_jobs()
    if not file_jobs:
        print('No keyword source lines found.')
        return None

    chunk_size = max(1, int(chunk_size or KEYWORD_CHUNK_SIZE))
    max_candidates = max(1, int(max_candidates_per_chunk or KEYWORD_MAX_CANDIDATES_PER_CHUNK))
    chunks = build_keyword_chunks(file_jobs, chunk_size=chunk_size)
    if not chunks:
        print('No keyword chunks built.')
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_name = f'{timestamp}_{guess_project_slug()}_keywords'
    package_dir = create_batch_package_dir(package_name)

    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'{KEYWORD_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'

    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    with open(input_jsonl_path, 'w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(build_keyword_request(chunk, max_candidates), ensure_ascii=False) + '\n')

    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': MANIFEST_MODE_KEYWORD_EXTRACTION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': {
            'keyword_chunk_size': chunk_size,
            'keyword_max_candidates_per_chunk': max_candidates,
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        'keyword_settings': {
            'chunk_size': chunk_size,
            'max_candidates_per_chunk': max_candidates,
        },
        'summary': {
            'file_count': len(file_jobs),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk['items']) for chunk in chunks),
        },
        'files': {
            job['file_rel_path']: {
                'path': job['file_path'],
                'task_count': job['task_count'],
            }
            for job in file_jobs
        },
        'chunks': chunks,
    }

    manifest_path = os.path.join(package_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    remember_latest_manifest(manifest_path)

    print(f'Created keyword package: {package_dir}')
    print(f"Source files: {manifest['summary']['file_count']}")
    print(f"Chunks: {manifest['summary']['chunk_count']}")
    print(f"Source lines: {manifest['summary']['item_count']}")
    print('Mode: keyword_extraction')
    return manifest_path


def split_manifest(target=None, max_chunks=600, max_items=0, display_name_prefix=''):
    manifest = load_manifest(target)
    chunks = manifest.get('chunks') or []
    if not chunks:
        raise SystemExit('Manifest does not contain any chunks to split.')

    input_jsonl_path = manifest.get('input_jsonl_path')
    if not input_jsonl_path or not os.path.isfile(input_jsonl_path):
        raise SystemExit(f'Input JSONL not found: {input_jsonl_path}')

    if max_chunks <= 0 and max_items <= 0:
        raise SystemExit('At least one of --max-chunks or --max-items must be greater than 0.')

    with open(input_jsonl_path, 'r', encoding='utf-8') as handle:
        request_lines = handle.readlines()

    if len(request_lines) != len(chunks):
        raise SystemExit(
            f'Chunk count mismatch between manifest ({len(chunks)}) and requests.jsonl ({len(request_lines)}).'
        )

    for index, (chunk, raw_line) in enumerate(zip(chunks, request_lines), start=1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f'Invalid JSONL row #{index}: {exc}') from exc
        if row.get('key') != chunk.get('key'):
            raise SystemExit(
                f"Chunk key mismatch at row #{index}: manifest={chunk.get('key')} jsonl={row.get('key')}"
            )

    grouped = split_chunks_and_lines(
        chunks,
        request_lines,
        max_chunks=max_chunks,
        max_items=max_items,
    )
    if len(grouped) <= 1:
        print('Split not needed; current package already fits the requested limits.')
        return [manifest['_manifest_path']]

    source_package_dir = manifest['_package_dir']
    split_root = os.path.join(source_package_dir, 'split_parts')
    os.makedirs(split_root, exist_ok=True)

    total_parts = len(grouped)
    now = datetime.now().isoformat(timespec='seconds')
    created_manifests = []
    source_display_name = manifest.get('display_name') or os.path.basename(source_package_dir)
    part_name_prefix = display_name_prefix.strip() if display_name_prefix else source_display_name

    for index, (part_chunks, part_lines) in enumerate(grouped, start=1):
        part_dir = os.path.join(split_root, f'part{index:02d}_of_{total_parts:02d}')
        os.makedirs(part_dir, exist_ok=True)

        part_input_jsonl_path = os.path.join(part_dir, 'requests.jsonl')
        with open(part_input_jsonl_path, 'w', encoding='utf-8') as handle:
            handle.writelines(part_lines)

        part_files = summarize_files_for_chunks(part_chunks)
        part_manifest = {
            'version': manifest.get('version', 1),
            'core_schema_version': manifest.get(
                'core_schema_version',
                translation_core.CORE_SCHEMA_VERSION,
            ),
            'mode': manifest_mode(manifest),
            'created_at': now,
            'display_name': f'{part_name_prefix}-part{index:02d}',
            'batch_model': manifest.get('batch_model', BATCH_MODEL),
            'base_dir': manifest.get('base_dir', legacy.BASE_DIR),
            'tl_dir': manifest.get('tl_dir', legacy.TL_DIR),
            **_manifest_target_language_fields(manifest),
            **batch_non_chinese_rules.manifest_non_chinese_rules_fields(manifest),
            'input_jsonl_path': part_input_jsonl_path,
            'result_jsonl_path': '',
            'job_name': '',
            'job_state': 'LOCAL_ONLY',
            'uploaded_file_name': '',
            'result_file_name': '',
            'settings': dict(manifest.get('settings') or {}),
            'summary': {
                'file_count': len(part_files),
                'chunk_count': len(part_chunks),
                'item_count': sum(len(chunk.get('items', [])) for chunk in part_chunks),
            },
            'files': part_files,
            'chunks': part_chunks,
            'split_from_manifest': manifest['_manifest_path'],
            'split_from_package': source_package_dir,
            'split_index': index,
            'split_total': total_parts,
            'split_limits': {
                'max_chunks': max_chunks,
                'max_items': max_items,
            },
        }
        copy_split_context_metadata(manifest, part_manifest, part_chunks)

        part_manifest_path = os.path.join(part_dir, 'manifest.json')
        with open(part_manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(part_manifest, handle, ensure_ascii=False, indent=2)

        canonical_manifest_path = _canonical_abs_path(part_manifest_path)
        created_manifests.append(canonical_manifest_path)
        remember_latest_manifest(canonical_manifest_path)

        print(f'Created split package: {part_dir}')
        print(f"Chunks: {part_manifest['summary']['chunk_count']}")
        print(f"Items: {part_manifest['summary']['item_count']}")

    manifest['split_children'] = created_manifests
    manifest['split_generated_at'] = now
    manifest['job_state'] = 'LOCAL_SPLIT_SOURCE'
    save_manifest(manifest, update_latest=False)
    remember_latest_manifest(created_manifests[0])

    print(f'Source manifest updated: {manifest["_manifest_path"]}')
    print(f'Latest manifest set to first split package: {created_manifests[0]}')
    return created_manifests


def current_batch_settings_snapshot():
    return {
        'target_size': BATCH_TARGET_SIZE,
        'target_chars': BATCH_TARGET_CHARS,
        'retry_target_size': BATCH_RETRY_TARGET_SIZE,
        'retry_target_chars': BATCH_RETRY_TARGET_CHARS,
        'context_before': BATCH_CONTEXT_BEFORE,
        'context_after': BATCH_CONTEXT_AFTER,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'temperature': BATCH_TEMPERATURE,
        'thinking_level': BATCH_THINKING_LEVEL,
        'safety_settings': BATCH_SAFETY_SETTINGS,
    }


def create_unique_child_dir(root_dir, name):
    os.makedirs(root_dir, exist_ok=True)
    base_dir = os.path.join(root_dir, name)
    candidates = [base_dir]
    candidates.extend(f'{base_dir}_{index:02d}' for index in range(1, 1000))
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f'Could not create unique directory for {name}.')


def retry_root_for_manifest(manifest):
    package_dir = os.path.normpath(manifest['_package_dir'])
    parts = package_dir.split(os.sep)
    if 'retry_parts' not in parts:
        return os.path.join(package_dir, 'retry_parts')
    retry_index = parts.index('retry_parts')
    split_package_dir = os.sep.join(parts[:retry_index])
    if not split_package_dir:
        return os.path.join(package_dir, 'retry_parts')
    return os.path.join(split_package_dir, 'retry_parts')


def retry_chunk_limits():
    return (
        max(1, min(BATCH_TARGET_SIZE, BATCH_RETRY_TARGET_SIZE)),
        max(1, min(BATCH_TARGET_CHARS, BATCH_RETRY_TARGET_CHARS)),
    )


def iter_retry_item_ranges(items):
    max_items, max_chars = retry_chunk_limits()
    total = len(items)
    start = 0
    while start < total:
        end = start
        current_chars = 0
        while end < total and (end - start) < max_items:
            item_chars = task_text_char_count(items[end])
            if end > start and current_chars + item_chars > max_chars:
                break
            current_chars += item_chars
            end += 1
        if end == start:
            end = start + 1
        yield start, end
        start = end


def build_retry_subchunk(chunk, start, end, sub_index):
    items = chunk.get('items') or []
    subchunk = copy.deepcopy(chunk)
    subitems = copy.deepcopy(items[start:end])
    parent_key = str(chunk.get('key') or '')
    subchunk['key'] = f'{parent_key}-retry-{sub_index:03d}'
    subchunk['retry_parent_key'] = parent_key
    subchunk['retry_item_start'] = start
    subchunk['retry_item_end'] = end
    subchunk['retry_item_ids'] = [item.get('id') for item in subitems]
    subchunk['items'] = subitems
    subchunk['line_numbers'] = [item.get('line') for item in subitems if item.get('line') is not None]
    subchunk['source_char_count'] = sum(task_text_char_count(item) for item in subitems)

    context_past = copy.deepcopy(chunk.get('context_past') or [])
    context_future = copy.deepcopy(chunk.get('context_future') or [])
    if BATCH_CONTEXT_BEFORE:
        context_past = (context_past + copy.deepcopy(items[max(0, start - BATCH_CONTEXT_BEFORE):start]))[-BATCH_CONTEXT_BEFORE:]
    if BATCH_CONTEXT_AFTER:
        context_future = (copy.deepcopy(items[end:min(len(items), end + BATCH_CONTEXT_AFTER)]) + context_future)[:BATCH_CONTEXT_AFTER]
    subchunk['context_past'] = context_past
    subchunk['context_future'] = context_future
    return subchunk


def split_retry_chunk(chunk):
    items = chunk.get('items') or []
    if not items:
        return [copy.deepcopy(chunk)]

    ranges = list(iter_retry_item_ranges(items))
    if len(ranges) <= 1:
        return [copy.deepcopy(chunk)]
    return [
        build_retry_subchunk(chunk, start, end, index)
        for index, (start, end) in enumerate(ranges, start=1)
    ]


def build_retry_chunks_for_keys(manifest, retry_keys):
    retry_key_set = set(retry_keys)
    retry_chunks = []
    for chunk in manifest.get('chunks') or []:
        if chunk.get('key') in retry_key_set:
            retry_chunks.extend(split_retry_chunk(chunk))
    return retry_chunks


def chunk_item_target_shapes(chunk, items=None):
    shapes = []
    for item in items if items is not None else (chunk.get('items') or []):
        source_text = item.get('source', item.get('text', ''))
        shapes.append(
            {
                'id': item.get('id', ''),
                'file_rel_path': item.get('file_rel_path', chunk.get('file_rel_path', '')),
                'line': item.get('line', item.get('line_number')),
                'start': item.get('start'),
                'end': item.get('end'),
                'source_checksum': hash_text(source_text),
            }
        )
    return shapes


def retry_subchunk_matches_parent(parent_chunk, retry_chunk):
    parent_shapes = {
        shape['id']: shape
        for shape in chunk_item_target_shapes(parent_chunk)
        if shape.get('id')
    }
    for shape in chunk_item_target_shapes(retry_chunk):
        if parent_shapes.get(shape.get('id')) != shape:
            return False
    return True


def chunk_target_signature(chunk):
    return stable_json_sha256(
        {
            'key': chunk.get('key', ''),
            'file_rel_path': chunk.get('file_rel_path', ''),
            'chunk_index': chunk.get('chunk_index'),
            'items': chunk_item_target_shapes(chunk),
        }
    )


def collect_result_integrity_issue_keys(manifest):
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit('Result JSONL not found. Run download first.')

    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    processed_keys = set()
    issue_keys = set()
    reason_counts = {}

    with open(result_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bump_counter(reason_counts, 'invalid_result_jsonl_row')
                continue

            key = row.get('key')
            if not key or key not in chunk_map:
                bump_counter(reason_counts, 'unknown_chunk_key')
                continue

            processed_keys.add(key)
            chunk = chunk_map[key]
            chunk_items = chunk.get('items') or []
            item_ids = {item.get('id') for item in chunk_items}
            response_payload = row.get('response') or {}
            finish_reason = extract_finish_reason(response_payload)

            if row.get('error'):
                issue_keys.add(key)
                bump_counter(reason_counts, 'row_error')
                continue

            response_text = extract_text_from_response_payload(response_payload)
            if not response_text:
                issue_keys.add(key)
                bump_counter(reason_counts, 'missing_response_text')
                continue

            try:
                payload = parse_json_payload(response_text)
                result_items = normalize_result_items(payload)
            except Exception:
                issue_keys.add(key)
                bump_counter(reason_counts, 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'failed_to_parse_model_json')
                continue

            seen_ids = set()
            if len(result_items) < len(chunk_items):
                issue_keys.add(key)
                bump_counter(reason_counts, 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_result_items')

            for result_item in result_items:
                result_id = result_item.get('id')
                if result_id not in item_ids:
                    issue_keys.add(key)
                    bump_counter(reason_counts, 'schema_or_item_mismatch')
                    continue
                if result_id in seen_ids:
                    issue_keys.add(key)
                    bump_counter(reason_counts, 'duplicate_result_id')
                    continue
                seen_ids.add(result_id)

            missing_ids = item_ids - seen_ids
            if missing_ids:
                issue_keys.add(key)
                bump_counter(reason_counts, 'response_missing_item_id', len(missing_ids))

    missing_keys = set(chunk_map.keys()) - processed_keys
    if missing_keys:
        issue_keys.update(missing_keys)
        bump_counter(reason_counts, 'missing_chunk_rows', len(missing_keys))

    return issue_keys, reason_counts


def collect_retry_chunk_keys(manifest):
    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    _replacements, _translated, failure_entries, summary = collect_result_actions(
        manifest,
        validate_sources=True,
    )
    retry_keys = set()
    for entry in failure_entries:
        key = entry.get('key')
        if key in chunk_map:
            retry_keys.add(key)

    integrity_keys, integrity_reason_counts = collect_result_integrity_issue_keys(manifest)
    retry_keys.update(key for key in integrity_keys if key in chunk_map)

    reason_counts = dict(summary.get('reason_counts') or {})
    for reason_code, count in integrity_reason_counts.items():
        reason_counts.setdefault(reason_code, count)

    ordered_keys = [chunk['key'] for chunk in manifest.get('chunks', []) if chunk.get('key') in retry_keys]
    return ordered_keys, failure_entries, summary, reason_counts


def build_retry_package(target=None, display_name_override=''):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'build-retry')
    retry_keys, failure_entries, summary, reason_counts = collect_retry_chunk_keys(manifest)
    if not retry_keys:
        print('No retry chunks needed.')
        return None


    retry_chunks = build_retry_chunks_for_keys(manifest, retry_keys)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    retry_root = retry_root_for_manifest(manifest)
    retry_dir = create_unique_child_dir(retry_root, f'{timestamp}_retry')

    input_jsonl_path = os.path.join(retry_dir, 'requests.jsonl')
    request_rows = [build_batch_request(chunk) for chunk in retry_chunks]
    write_jsonl_file(input_jsonl_path, request_rows)

    source_display_name = manifest.get('display_name') or os.path.basename(manifest['_package_dir'])
    display_name = display_name_override.strip() if display_name_override else f'{source_display_name}-retry-{timestamp}'
    retry_files = summarize_files_for_chunks(retry_chunks)
    retry_manifest = {
        'version': manifest.get('version', 2),
        'manifest_version': manifest.get('manifest_version', 2),
        'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
        'mode': MANIFEST_MODE_TRANSLATION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': manifest.get('base_dir', legacy.BASE_DIR),
        'tl_dir': manifest.get('tl_dir', legacy.TL_DIR),
        **_manifest_target_language_fields(manifest),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(manifest),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': current_batch_settings_snapshot(),
        'summary': {
            'file_count': len(retry_files),
            'chunk_count': len(retry_chunks),
            'source_chunk_count': len(retry_keys),
            'item_count': sum(len(chunk.get('items') or []) for chunk in retry_chunks),
        },
        'files': retry_files,
        'chunks': retry_chunks,
        'retry_of_manifest': manifest['_manifest_path'],
        'retry_of_package': manifest['_package_dir'],
        'retry_source_result_jsonl_path': resolve_manifest_result_path(manifest),
        'retry_source_check_report_path': manifest.get('last_check_report_path', ''),
        'retry_reason_counts': reason_counts,
        'retry_failed_item_count': len(failure_entries),
        'retry_chunk_keys': retry_keys,
    }
    copy_split_context_metadata(manifest, retry_manifest, retry_chunks)

    retry_manifest_path = os.path.join(retry_dir, 'manifest.json')
    with open(retry_manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(retry_manifest, handle, ensure_ascii=False, indent=2)

    manifest.setdefault('retry_children', []).append(retry_manifest_path)
    manifest['last_retry_manifest_path'] = retry_manifest_path
    manifest['last_retry_generated_at'] = datetime.now().isoformat(timespec='seconds')
    save_manifest(manifest, update_latest=False)
    remember_latest_manifest(retry_manifest_path)

    print(f'Created retry package: {retry_dir}')
    print(f"Retry source chunks: {retry_manifest['summary']['source_chunk_count']}")
    print(f"Retry request chunks: {retry_manifest['summary']['chunk_count']}")
    print(f"Retry items: {retry_manifest['summary']['item_count']}")
    print(f"Failure items considered: {len(failure_entries)}")
    print(f'Manifest: {retry_manifest_path}')
    return retry_manifest_path


def load_result_rows_by_key(manifest, label):
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit(f'{label} result JSONL not found: {result_path}')
    rows = []
    rows_by_key = {}
    with open(result_path, 'r', encoding='utf-8') as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'Invalid {label} result JSONL row #{index}: {exc}') from exc
            key = row.get('key')
            if not key:
                raise SystemExit(f'Missing key in {label} result JSONL row #{index}.')
            if key in rows_by_key:
                raise SystemExit(f'Duplicate key in {label} result JSONL: {key}')
            rows.append(row)
            rows_by_key[key] = row
    return rows, rows_by_key, result_path


def result_items_from_row(row, label, allow_empty=False):
    response_payload = row.get('response', {}) if isinstance(row, dict) else {}
    response_text = extract_text_from_response_payload(response_payload)
    if not response_text:
        if allow_empty:
            return []
        raise SystemExit(f'Missing text in {label} result row: {row.get("key", "") if isinstance(row, dict) else ""}')
    try:
        return normalize_result_items(parse_json_payload(response_text))
    except Exception as exc:
        if allow_empty:
            return []
        raise SystemExit(f'Failed to parse {label} result row JSON: {exc}') from exc


def response_payload_with_text(response_payload, response_text):
    payload = copy.deepcopy(response_payload) if isinstance(response_payload, dict) else {}
    target = payload.get('response') if isinstance(payload.get('response'), dict) else payload
    candidates = target.get('candidates')
    if not isinstance(candidates, list) or not candidates:
        candidates = [{}]
        target['candidates'] = candidates
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    candidates[0] = candidate
    content = candidate.get('content') if isinstance(candidate.get('content'), dict) else {}
    content['parts'] = [{'text': response_text}]
    content.setdefault('role', 'model')
    candidate['content'] = content
    candidate.setdefault('finishReason', 'STOP')
    return payload


def compact_result_items_for_response(result_items):
    compacted = []
    for item in result_items:
        item_id = item.get('id')
        if not item_id:
            continue
        compacted.append({'id': item_id, 'translation': item.get('translation', '')})
    return compacted


def merge_parent_row_with_retry_item_rows(parent_row, parent_chunk, retry_chunks, retry_rows_by_key):
    merged_by_id = {}
    if parent_row:
        for item in result_items_from_row(parent_row, 'parent', allow_empty=True):
            if item.get('id'):
                merged_by_id[item['id']] = item

    replaced_ids = set()
    for retry_chunk in retry_chunks:
        retry_key = retry_chunk.get('key')
        retry_row = retry_rows_by_key.get(retry_key)
        if not retry_row:
            raise SystemExit(f'Retry result is missing row for partial chunk: {retry_key}')
        allowed_ids = {item.get('id') for item in retry_chunk.get('items') or []}
        for item in result_items_from_row(retry_row, 'retry'):
            item_id = item.get('id')
            if item_id in allowed_ids:
                merged_by_id[item_id] = item
                replaced_ids.add(item_id)

    ordered_items = []
    for target_item in parent_chunk.get('items') or []:
        item_id = target_item.get('id')
        if item_id in merged_by_id:
            ordered_items.append(merged_by_id[item_id])

    merged_row = copy.deepcopy(parent_row) if isinstance(parent_row, dict) else {}
    merged_row['key'] = parent_chunk.get('key')
    merged_row.pop('error', None)
    merged_text = json.dumps(compact_result_items_for_response(ordered_items), ensure_ascii=False, indent=2)
    merged_row['response'] = response_payload_with_text(merged_row.get('response', {}), merged_text)
    return merged_row, len(replaced_ids)


def assert_retry_manifest_matches_parent(parent_manifest, retry_manifest):
    retry_of_manifest = retry_manifest.get('retry_of_manifest')
    if retry_of_manifest and _normalized_abs_path(retry_of_manifest) != _normalized_abs_path(parent_manifest['_manifest_path']):
        raise SystemExit(
            'Retry manifest was generated for a different parent manifest: '
            f'{retry_of_manifest}'
        )

    parent_chunks = {chunk['key']: chunk for chunk in parent_manifest.get('chunks') or []}
    retry_chunks = retry_manifest.get('chunks') or []
    if not retry_chunks:
        raise SystemExit('Retry manifest has no chunks.')

    for chunk in retry_chunks:
        key = chunk.get('key')
        parent_key = chunk.get('retry_parent_key') or key
        parent_chunk = parent_chunks.get(parent_key)
        if not parent_chunk:
            raise SystemExit(f'Retry chunk is not present in parent manifest: {key}')
        if chunk.get('retry_parent_key'):
            if not retry_subchunk_matches_parent(parent_chunk, chunk):
                raise SystemExit(f'Retry chunk target shape differs from parent manifest: {key}')
        elif chunk_target_signature(chunk) != chunk_target_signature(parent_chunk):
            raise SystemExit(f'Retry chunk target shape differs from parent manifest: {key}')

def merge_retry_results(parent_target, retry_target):
    parent_manifest = load_manifest(parent_target)
    retry_manifest = load_manifest(retry_target)
    require_manifest_mode(parent_manifest, MANIFEST_MODE_TRANSLATION, 'merge-retry')
    require_manifest_mode(retry_manifest, MANIFEST_MODE_TRANSLATION, 'merge-retry')
    assert_retry_manifest_matches_parent(parent_manifest, retry_manifest)

    retry_chunks = retry_manifest.get('chunks') or []
    retry_keys = [chunk['key'] for chunk in retry_chunks]
    retry_key_set = set(retry_keys)
    parent_chunks = {chunk['key']: chunk for chunk in parent_manifest.get('chunks') or []}
    parent_rows, parent_rows_by_key, parent_result_path = load_result_rows_by_key(parent_manifest, 'parent')
    retry_rows, retry_rows_by_key, retry_result_path = load_result_rows_by_key(retry_manifest, 'retry')

    unknown_retry_rows = set(retry_rows_by_key) - retry_key_set
    if unknown_retry_rows:
        raise SystemExit(f'Retry result contains rows outside retry chunks: {sorted(unknown_retry_rows)[:5]}')

    missing_retry_rows = retry_key_set - set(retry_rows_by_key)
    if missing_retry_rows:
        raise SystemExit(f'Retry result is missing rows for chunks: {sorted(missing_retry_rows)[:5]}')

    direct_retry_keys = []
    partial_chunks_by_parent = {}
    for chunk in retry_chunks:
        parent_key = chunk.get('retry_parent_key')
        if parent_key:
            partial_chunks_by_parent.setdefault(parent_key, []).append(chunk)
        else:
            direct_retry_keys.append(chunk.get('key'))
    direct_retry_key_set = set(direct_retry_keys)

    merged_rows = []
    replaced_keys = set()
    replaced_item_count = 0
    for row in parent_rows:
        key = row.get('key')
        if key in direct_retry_key_set:
            merged_rows.append(retry_rows_by_key[key])
            replaced_keys.add(key)
            retry_chunk = next((chunk for chunk in retry_chunks if chunk.get('key') == key), {})
            replaced_item_count += len(retry_chunk.get('items') or [])
        elif key in partial_chunks_by_parent:
            parent_chunk = parent_chunks.get(key)
            merged_row, item_count = merge_parent_row_with_retry_item_rows(
                row,
                parent_chunk,
                partial_chunks_by_parent[key],
                retry_rows_by_key,
            )
            merged_rows.append(merged_row)
            replaced_keys.add(key)
            replaced_item_count += item_count
        else:
            merged_rows.append(row)

    for key in direct_retry_keys:
        if key not in parent_rows_by_key:
            merged_rows.append(retry_rows_by_key[key])
            replaced_keys.add(key)
            retry_chunk = next((chunk for chunk in retry_chunks if chunk.get('key') == key), {})
            replaced_item_count += len(retry_chunk.get('items') or [])

    for parent_key, partial_chunks in partial_chunks_by_parent.items():
        if parent_key in parent_rows_by_key:
            continue
        parent_chunk = parent_chunks.get(parent_key)
        merged_row, item_count = merge_parent_row_with_retry_item_rows(
            {},
            parent_chunk,
            partial_chunks,
            retry_rows_by_key,
        )
        merged_rows.append(merged_row)
        replaced_keys.add(parent_key)
        replaced_item_count += item_count

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_name = f'results.merged_{timestamp}.jsonl'
    merged_path = os.path.join(parent_manifest['_package_dir'], merged_name)
    write_jsonl_file(merged_path, merged_rows)

    parent_manifest['result_jsonl_path'] = merged_name
    parent_manifest['job_state'] = 'RESULTS_MERGED'
    parent_manifest.setdefault('retry_merge_history', []).append(
        {
            'merged_at': datetime.now().isoformat(timespec='seconds'),
            'retry_manifest': retry_manifest['_manifest_path'],
            'retry_result_jsonl_path': retry_result_path,
            'previous_result_jsonl_path': parent_result_path,
            'merged_result_jsonl_path': merged_path,
            'replaced_chunks': len(replaced_keys),
            'replaced_items': replaced_item_count,
        }
    )
    parent_manifest['last_retry_merged_manifest_path'] = retry_manifest['_manifest_path']
    parent_manifest['last_retry_merged_at'] = datetime.now().isoformat(timespec='seconds')
    for key in (
        'last_check_at',
        'last_check_summary',
        'last_check_report_path',
    ):
        parent_manifest.pop(key, None)
    save_manifest(parent_manifest, update_latest=True)

    print(f'Merged retry results into: {parent_manifest["_manifest_path"]}')
    print(f'Previous result JSONL: {parent_result_path}')
    print(f'Retry result JSONL: {retry_result_path}')
    print(f'Merged result JSONL: {merged_path}')
    print(f'Replaced chunks: {len(replaced_keys)}')
    print('Run check on the parent manifest before apply.')
    return parent_manifest['_manifest_path']

def ensure_manifest_cost_estimate(manifest):
    estimate = manifest.get('cost_estimate')
    if isinstance(estimate, dict) and estimate.get('estimated_cost_max') is not None:
        return estimate
    try:
        return batch_cost_estimate.attach_cost_estimate_to_manifest(
            manifest,
            translator_config=load_json_file(legacy.TRANSLATOR_CONFIG),
        )
    except FileNotFoundError as exc:
        jsonl_path = manifest.get('input_jsonl_path') or ''
        raise SystemExit(
            f'Batch input JSONL not found: {jsonl_path or exc}'
        ) from exc
    except json.JSONDecodeError as exc:
        jsonl_path = manifest.get('input_jsonl_path') or ''
        raise SystemExit(
            f'Batch input JSONL is not valid JSON: {jsonl_path} ({exc})'
        ) from exc


def _manifest_package_dir(manifest):
    package_dir = manifest.get('_package_dir')
    if package_dir:
        return package_dir
    manifest_path = manifest.get('_manifest_path')
    if manifest_path:
        return os.path.dirname(manifest_path)
    raise SystemExit('Manifest package directory is missing.')


def _raise_uncertain_submit_blocked(uncertain_state):
    message = uncertain_state.get('message') or batch_submit_recovery.BLOCKED_MESSAGE_PREFIX
    recovery_hint = uncertain_state.get('recovery_hint') or batch_submit_recovery.RECOVER_HINT
    raise SystemExit(f'{message}\n{recovery_hint}')


def recover_submit_manifest(target=None, verify_remote=True):
    manifest = load_manifest(target)
    if manifest.get('job_name'):
        print(f"Manifest already submitted: {manifest['job_name']}")
        print(f"Manifest: {manifest['_manifest_path']}")
        return manifest['_manifest_path']

    package_dir = _manifest_package_dir(manifest)
    entries = batch_submit_recovery.read_submit_journal_entries(package_dir)
    pending_job = batch_submit_recovery.find_uncommitted_job_created(entries, manifest)
    if pending_job is None:
        uncertain_state = batch_submit_recovery.get_uncertain_submit_state(
            manifest,
            package_dir=package_dir,
        )
        if uncertain_state and uncertain_state.get('kind') == 'upload_pending_job_create':
            for hint in batch_submit_recovery.format_uncertain_submit_hints(uncertain_state):
                print(hint)
            raise SystemExit(
                'No recoverable remote job found. Re-run submit with --resume to continue job creation.'
            )
        raise SystemExit('No recoverable submit state found for this manifest.')

    job_name = pending_job.get('job_name', '')
    if verify_remote:
        client = create_batch_client()
        try:
            batch_job = client.batches.get(name=job_name)
            remote_state = get_state_name(getattr(batch_job, 'state', None))
            if remote_state:
                pending_job['job_state'] = remote_state
            print(f'Verified remote batch job: {job_name}')
            if remote_state:
                print(f'Remote state: {remote_state}')
        except Exception as exc:
            print(f'Warning: Could not verify remote job {job_name}: {exc}')

    batch_submit_recovery.apply_recovered_job_to_manifest(
        manifest,
        pending_job,
        package_dir=package_dir,
        submitted_api_key_index=getattr(legacy, 'CURRENT_KEY_INDEX', 0),
    )
    save_manifest(manifest)
    print(f'Recovered batch job: {manifest["job_name"]}')
    print(f"Manifest: {manifest['_manifest_path']}")
    return manifest['_manifest_path']


def submit_manifest(
    target=None,
    display_name_override='',
    model_override='',
    max_cost=None,
    force_resubmit=False,
    resume_upload=False,
):
    manifest = load_manifest(target) if target else None
    if manifest is None:
        manifest_path = create_batch_package(display_name_override=display_name_override)
        if not manifest_path:
            return None
        manifest = load_manifest(manifest_path)

    if manifest.get('job_name'):
        raise SystemExit(f"Manifest already submitted: {manifest['job_name']}")

    package_dir = _manifest_package_dir(manifest)
    uncertain_state = batch_submit_recovery.get_uncertain_submit_state(
        manifest,
        package_dir=package_dir,
    )
    if uncertain_state:
        if uncertain_state.get('kind') == 'job_created_uncommitted':
            _raise_uncertain_submit_blocked(uncertain_state)
        if uncertain_state.get('kind') == 'upload_pending_job_create':
            if resume_upload:
                current_checksum = batch_submit_recovery.compute_request_checksum(manifest)
                saved_checksum = manifest.get('request_checksum')
                if saved_checksum and saved_checksum != current_checksum:
                    raise SystemExit(
                        'Submit blocked: input JSONL changed since upload. '
                        'Re-run submit with --force to start over.'
                    )
            elif force_resubmit:
                batch_submit_recovery.clear_incomplete_submit_state(manifest)
                uncertain_state = None
            else:
                _raise_uncertain_submit_blocked(uncertain_state)

    if display_name_override:
        manifest['display_name'] = display_name_override.strip()
    if model_override:
        manifest['batch_model'] = model_override.strip()

    if max_cost is not None:
        cost_estimate = ensure_manifest_cost_estimate(manifest)
        for line in batch_cost_estimate.format_cost_estimate_lines(cost_estimate):
            print(line)
        if batch_cost_estimate.cost_estimate_exceeds_max(cost_estimate, max_cost):
            currency = cost_estimate.get('currency') or 'USD'
            raise SystemExit(
                'Submit blocked by --max-cost: '
                f"estimated max {cost_estimate.get('estimated_cost_max', 0):.4f} {currency} "
                f'exceeds limit {float(max_cost):.4f} {currency}.'
            )

    resume_existing_upload = (
        resume_upload
        and uncertain_state is not None
        and uncertain_state.get('kind') == 'upload_pending_job_create'
        and manifest.get('uploaded_file_name')
    )
    if not resume_existing_upload:
        batch_submit_recovery.begin_submit_attempt(manifest, package_dir=package_dir)
        save_manifest(manifest)

    attempts = max(1, len(getattr(legacy, 'API_KEYS', [])))
    last_error = None

    for attempt in range(1, attempts + 1):
        client = create_batch_client()
        uploaded_file_name = ''
        try:
            if resume_existing_upload:
                uploaded_file_name = manifest['uploaded_file_name']
                print(f"Reusing uploaded JSONL: {uploaded_file_name}")
            else:
                print(f"Uploading JSONL: {manifest['input_jsonl_path']}")
                uploaded_file = client.files.upload(
                    file=manifest['input_jsonl_path'],
                    config=genai_types.UploadFileConfig(
                        display_name=manifest['display_name'],
                        mime_type='jsonl',
                    ),
                )
                uploaded_file_name = getattr(uploaded_file, 'name', '')
                batch_submit_recovery.record_upload_completed(
                    manifest,
                    package_dir=package_dir,
                    uploaded_file_name=uploaded_file_name,
                )
                _clear_submit_failure_metadata(manifest)
                save_manifest(manifest)
                print(f'Uploaded file: {uploaded_file_name}')

            print(f"Creating batch job with model: {manifest['batch_model']}")
            batch_job = client.batches.create(
                model=manifest['batch_model'],
                src=uploaded_file_name,
                config={'display_name': manifest['display_name']},
            )

            job_name = getattr(batch_job, 'name', '')
            job_state = get_state_name(getattr(batch_job, 'state', None))
            batch_submit_recovery.record_job_created(
                manifest,
                package_dir=package_dir,
                job_name=job_name,
                job_state=job_state,
                uploaded_file_name=uploaded_file_name,
            )

            manifest['job_name'] = job_name
            manifest['job_state'] = job_state
            manifest['submitted_at'] = datetime.now().isoformat(timespec='seconds')
            manifest['last_status_checked_at'] = manifest['submitted_at']
            manifest['submitted_api_key_index'] = getattr(legacy, 'CURRENT_KEY_INDEX', 0)
            manifest['submitted_api_key_number'] = manifest['submitted_api_key_index'] + 1
            manifest['last_status_api_key_index'] = manifest['submitted_api_key_index']
            _clear_submit_failure_metadata(manifest)
            save_manifest(manifest)
            batch_submit_recovery.record_manifest_committed(manifest, package_dir=package_dir)
            save_manifest(manifest)

            print(f"Batch job created: {manifest['job_name']}")
            print(f"Manifest: {manifest['_manifest_path']}")
            return manifest['_manifest_path']
        except Exception as exc:
            last_error = exc
            quota_error = is_quota_error(exc)
            manifest['last_submit_error'] = str(exc)
            manifest['last_submit_error_type'] = (
                'quota_or_resource_exhausted' if quota_error else 'submit_error'
            )
            manifest['job_state'] = 'SUBMIT_FAILED'
            recommendation = attach_submit_split_recommendation(manifest) if quota_error else {}
            if not quota_error:
                manifest.pop('split_recommended', None)
                manifest.pop('last_submit_quota_recommendation', None)
            if uploaded_file_name:
                manifest['uploaded_file_name'] = uploaded_file_name
                manifest.setdefault('uploaded_file_names', [])
                if uploaded_file_name not in manifest['uploaded_file_names']:
                    manifest['uploaded_file_names'].append(uploaded_file_name)
                if manifest.get('submit_state') != batch_submit_recovery.SUBMIT_STATE_JOB_CREATED:
                    manifest['submit_state'] = batch_submit_recovery.SUBMIT_STATE_UPLOADED
            save_manifest(manifest)

            if quota_error and attempt < attempts and legacy.rotate_api_key():
                print(f'Quota hit during batch submit. Retrying with next API key ({attempt}/{attempts})...')
                resume_existing_upload = False
                continue
            if quota_error:
                print_submit_split_recommendation(recommendation)
            raise

    if last_error is not None:
        raise last_error
    return None


def refresh_manifest_status(manifest):
    client, batch_job = fetch_batch_job_for_manifest(manifest)

    manifest['job_state'] = get_state_name(getattr(batch_job, 'state', None))
    manifest['last_status_checked_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['batch_stats'] = extract_batch_stats(batch_job)
    manifest['job_error'] = serialize_unknown(get_nested(batch_job, 'error'))
    write_status_snapshot(manifest, batch_job)

    dest = get_nested(batch_job, 'dest')
    if dest:
        result_file_name = get_nested(dest, 'file_name', 'fileName')
        if result_file_name:
            manifest['result_file_name'] = result_file_name

    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    return manifest


def show_status(target=None):
    manifest = load_manifest(target)
    uncertain_state = batch_submit_recovery.get_uncertain_submit_state(manifest)
    if uncertain_state:
        print('Submit recovery required before re-submitting this package.')
        for hint in batch_submit_recovery.format_uncertain_submit_hints(uncertain_state):
            print(hint)
    if manifest.get('job_name'):
        manifest = refresh_manifest_status(manifest)
    print(f"Manifest: {manifest['_manifest_path']}")
    print(f"Job: {manifest.get('job_name')}")
    print(f"State: {manifest.get('job_state')}")
    stats = manifest.get('batch_stats') or {}
    if stats:
        print(
            'Stats: '
            f"total={stats.get('request_count', '?')} "
            f"ok={stats.get('successful_request_count', '?')} "
            f"failed={stats.get('failed_request_count', '?')} "
            f"pending={stats.get('pending_request_count', '?')}"
        )
    if manifest.get('result_file_name'):
        print(f"Result file: {manifest['result_file_name']}")
    if manifest.get('job_error'):
        print(f"Error: {manifest['job_error']}")
    elif manifest.get('job_state') == 'JOB_STATE_FAILED':
        snapshot_path = manifest.get('last_status_snapshot_path')
        if snapshot_path:
            print('Error: API returned JOB_STATE_FAILED but no explicit job_error field.')
            print(f'Status snapshot: {snapshot_path}')
    return manifest


def decode_downloaded_content(downloaded):
    if isinstance(downloaded, bytes):
        return downloaded.decode('utf-8')
    if hasattr(downloaded, 'decode'):
        return downloaded.decode('utf-8')
    if hasattr(downloaded, 'text'):
        return downloaded.text
    return str(downloaded)


def download_results(target=None, force=False):
    manifest = load_manifest(target)
    manifest = refresh_manifest_status(manifest)
    state = manifest.get('job_state')
    if state != 'JOB_STATE_SUCCEEDED':
        raise SystemExit(f'Batch job is not succeeded yet: {state}')

    result_path = resolve_manifest_result_path(manifest)
    if os.path.isfile(result_path) and not force:
        print(f'Result file already exists: {result_path}')
        return result_path

    result_file_name = manifest.get('result_file_name')
    if not result_file_name:
        raise SystemExit('Result file name is missing from manifest/job metadata.')

    client = create_batch_client(api_key_index=manifest.get('submitted_api_key_index'))
    print(f'Downloading result file: {result_file_name}')
    downloaded = client.files.download(file=result_file_name)
    text = decode_downloaded_content(downloaded)

    with open(result_path, 'w', encoding='utf-8') as handle:
        handle.write(text)

    manifest['result_jsonl_path'] = result_path
    manifest['downloaded_at'] = datetime.now().isoformat(timespec='seconds')
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print(f'Saved results to: {result_path}')
    return result_path

def extract_text_from_response_payload(response_payload):
    payload = response_payload
    if not isinstance(payload, dict):
        return ''

    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get('candidates')
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get('content') if isinstance(candidate, dict) else None
            parts = content.get('parts') if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get('text'):
                    texts.append(part['text'])
            if texts:
                return ''.join(texts)

    text = payload.get('text')
    return text if isinstance(text, str) else ''


def extract_finish_reason(response_payload):
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get('candidates')
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get('finishReason'):
                return str(candidate['finishReason'])
    return ''


def extract_usage_metadata(response_payload):
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response
    usage = payload.get('usageMetadata')
    return usage if isinstance(usage, dict) else {}


def summarize_usage_metadata(usage_metadata):
    if not isinstance(usage_metadata, dict):
        return {}
    summary = {}
    for key in ('promptTokenCount', 'thoughtsTokenCount', 'candidatesTokenCount', 'totalTokenCount'):
        value = usage_metadata.get(key)
        if value is not None:
            summary[key] = value
    return summary


def bump_counter(bucket, name, amount=1):
    bucket[name] = bucket.get(name, 0) + amount


def salvage_partial_json_array(text):
    start = text.find('[')
    if start < 0:
        return []

    decoder = json.JSONDecoder()
    index = start + 1
    items = []
    while index < len(text):
        while index < len(text) and text[index] in ' \r\n\t,':
            index += 1
        if index >= len(text):
            break
        if text[index] == ']':
            return items
        try:
            item, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        items.append(item)
    return items


def parse_json_payload(text):
    if not text:
        raise ValueError('Empty response text')

    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        embedded_payloads = []
        for start, char in enumerate(cleaned):
            if char not in '[{':
                continue
            try:
                payload, end = decoder.raw_decode(cleaned[start:])
                embedded_payloads.append((start + end, -start, payload))
            except json.JSONDecodeError:
                continue
        if embedded_payloads:
            _end, neg_start, payload = max(embedded_payloads, key=lambda item: (item[0], item[1]))
            start = -neg_start
            previous_index = start - 1
            while previous_index >= 0 and cleaned[previous_index].isspace():
                previous_index -= 1
            salvaged = salvage_partial_json_array(cleaned)
            if (
                salvaged
                and previous_index >= 0
                and cleaned[previous_index] in '[,'
            ):
                return salvaged
            return payload

        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        salvaged = salvage_partial_json_array(cleaned)
        if salvaged:
            return salvaged
        raise


def normalize_result_items(payload):
    return translation_core.normalize_model_results(
        payload,
        mode=translation_core.MODE_TRANSLATION,
    )


KEYWORD_CATEGORIES = translation_core.KEYWORD_CATEGORIES


def coerce_keyword_confidence(value):
    return translation_core.coerce_keyword_confidence(value)


def normalize_keyword_candidates(payload):
    return translation_core.normalize_model_results(
        payload,
        mode=translation_core.MODE_KEYWORD_EXTRACTION,
    )


def normalize_keyword_summary(payload):
    if not isinstance(payload, dict):
        return {'chunk_summary': '', 'summary_evidence_item_ids': []}

    summary_text = ''
    for key in ('chunk_summary', 'plot_summary', 'scene_summary', 'summary'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            summary_text = value
            break

    raw_source_item_ids = payload.get('summary_evidence_item_ids')
    if not isinstance(raw_source_item_ids, list):
        raw_source_item_ids = payload.get('summary_source_item_ids')
    if not isinstance(raw_source_item_ids, list):
        raw_source_item_ids = []

    return {
        'chunk_summary': compact_text(str(summary_text or '')),
        'summary_evidence_item_ids': [
            str(value) for value in raw_source_item_ids if str(value).strip()
        ],
    }


def normalize_keyword_extraction_payload(payload):
    return {
        'candidates': normalize_keyword_candidates(payload),
        **normalize_keyword_summary(payload),
    }


def keyword_candidate_key(candidate):
    return (
        compact_text(candidate.get('source', '')).lower(),
        compact_text(candidate.get('suggested_target', '')).lower(),
        compact_text(candidate.get('category', '')).lower(),
    )


def merge_keyword_candidate(existing, incoming):
    existing['confidence'] = max(existing.get('confidence', 0.0), incoming.get('confidence', 0.0))
    for field in ('source_files', 'source_item_ids', 'source_lines'):
        values = list(existing.get(field) or [])
        for value in incoming.get(field) or []:
            if value not in values:
                values.append(value)
        existing[field] = values
    evidence_values = list(existing.get('evidence_items') or [])
    evidence = incoming.get('evidence')
    if evidence and evidence not in evidence_values:
        evidence_values.append(evidence)
    existing['evidence_items'] = evidence_values
    if evidence_values:
        existing['evidence'] = ' / '.join(evidence_values[:3])
    existing['occurrences'] = int(existing.get('occurrences', 1)) + int(incoming.get('occurrences', 1))
    return existing


def resolve_keyword_export_path(manifest, value, default_name, field_name):
    package_dir = manifest.get('_package_dir')
    if value:
        return resolve_path_under_dir(package_dir, value, field_name)
    return os.path.join(package_dir, default_name)


def validate_keyword_export_paths(manifest, *output_paths):
    normalized_outputs = [_normalized_abs_path(path) for path in output_paths if path]
    if len(normalized_outputs) != len(set(normalized_outputs)):
        raise SystemExit('Keyword export outputs must be different files.')

    reserved_paths = {
        os.path.join(manifest.get('_package_dir', ''), 'manifest.json'),
        os.path.join(manifest.get('_package_dir', ''), 'requests.jsonl'),
        os.path.join(manifest.get('_package_dir', ''), 'results.jsonl'),
        os.path.join(manifest.get('_package_dir', ''), 'failures.jsonl'),
    }
    for manifest_key in ('_manifest_path', 'input_jsonl_path', 'result_jsonl_path'):
        value = manifest.get(manifest_key)
        if value:
            reserved_paths.add(value)
    normalized_reserved = {_normalized_abs_path(path) for path in reserved_paths if path}
    for output_path in output_paths:
        if _normalized_abs_path(output_path) in normalized_reserved:
            raise SystemExit(f'Keyword export output would overwrite reserved package file: {output_path}')


def match_keyword_items_by_ids(source_item_ids, chunk):
    requested_ids = {str(value) for value in source_item_ids or [] if str(value).strip()}
    if not requested_ids:
        return []

    matched = []
    for item in chunk.get('items') or []:
        if str(item.get('id') or '') in requested_ids:
            matched.append(item)
    return matched


def match_keyword_candidate_items(candidate, chunk):
    items = chunk.get('items') or []
    requested_ids = {str(value) for value in candidate.get('source_item_ids') or [] if str(value).strip()}
    evidence = compact_text(candidate.get('evidence', '')).lower()
    source = compact_text(candidate.get('source', '')).lower()
    matched = []

    for item in items:
        item_id = str(item.get('id') or '')
        item_text = compact_text(item.get('text', '')).lower()
        if item_id and item_id in requested_ids:
            matched.append(item)
            continue
        if item_id and evidence and item_id.lower() in evidence:
            matched.append(item)
            continue
        if source and item_text and source in item_text:
            matched.append(item)

    deduped = []
    seen = set()
    for item in matched:
        item_id = item.get('id')
        key = item_id or (item.get('line_number'), item.get('text'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def markdown_escape_cell(value):
    return compact_text(str(value or '')).replace('|', '\\|').replace('\n', ' ')


def write_keyword_markdown(path, candidates, summary):
    lines = [
        '# Keyword Candidates',
        '',
        f"- Candidate count: {len(candidates)}",
        f"- Parsed chunks: {summary.get('parsed_chunks', 0)}/{summary.get('expected_chunks', summary.get('result_rows', 0))}",
        f"- Missing chunk rows: {summary.get('missing_chunk_rows', 0)}",
        f"- Ambiguous provenance candidates: {summary.get('ambiguous_provenance_candidates', 0)}",
        '',
        '| Source | Suggested target | Category | Confidence | Evidence | Files |',
        '| --- | --- | --- | ---: | --- | --- |',
    ]
    for candidate in candidates:
        files = ', '.join(candidate.get('source_files') or [])
        lines.append(
            '| '
            + ' | '.join(
                [
                    markdown_escape_cell(candidate.get('source')),
                    markdown_escape_cell(candidate.get('suggested_target')),
                    markdown_escape_cell(candidate.get('category')),
                    f"{candidate.get('confidence', 0.0):.2f}",
                    markdown_escape_cell(candidate.get('evidence')),
                    markdown_escape_cell(files),
                ]
            )
            + ' |'
        )
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def write_keyword_summary_markdown(path, summaries, summary):
    lines = [
        '# Keyword Chunk Summaries',
        '',
        f"- Summary count: {len(summaries)}",
        f"- Parsed chunks: {summary.get('parsed_chunks', 0)}/{summary.get('expected_chunks', summary.get('result_rows', 0))}",
        f"- Ambiguous summary provenance chunks: {summary.get('ambiguous_summary_chunks', 0)}",
        '',
        '| File | Chunk lines | Evidence lines | Summary | Evidence item ids |',
        '| --- | ---: | ---: | --- | --- |',
    ]
    for item in summaries:
        lines.append(
            '| '
            + ' | '.join(
                [
                    markdown_escape_cell(item.get('file_rel_path')),
                    markdown_escape_cell(', '.join(str(value) for value in item.get('line_numbers') or [])),
                    markdown_escape_cell(', '.join(str(value) for value in item.get('source_lines') or [])),
                    markdown_escape_cell(item.get('chunk_summary')),
                    markdown_escape_cell(', '.join(item.get('summary_evidence_item_ids') or [])),
                ]
            )
            + ' |'
        )
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def export_keyword_candidates(
    target=None,
    output_jsonl='',
    output_markdown='',
    output_summary_jsonl='',
    output_summary_markdown='',
):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_KEYWORD_EXTRACTION, 'export-keywords')
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit('Result JSONL not found. Run download first.')

    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    processed_keys = set()
    merged_candidates = {}
    summary = {
        'expected_chunks': len(chunk_map),
        'result_rows': 0,
        'processed_chunks': 0,
        'parsed_chunks': 0,
        'candidate_count_raw': 0,
        'candidate_count_deduped': 0,
        'chunk_row_errors': 0,
        'unknown_chunk_keys': 0,
        'missing_response_chunks': 0,
        'missing_chunk_rows': 0,
        'ambiguous_provenance_candidates': 0,
        'chunk_summary_count': 0,
        'ambiguous_summary_chunks': 0,
        'parse_errors': 0,
        'reason_counts': {},
    }
    chunk_summaries = []

    with open(result_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            summary['result_rows'] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'invalid_result_jsonl_row')
                continue
            key = row.get('key')
            chunk = chunk_map.get(key)
            if not chunk:
                summary['unknown_chunk_keys'] += 1
                bump_counter(summary['reason_counts'], 'unknown_chunk_key')
                continue
            processed_keys.add(key)
            if row.get('error'):
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'row_error')
                continue
            response_text = extract_text_from_response_payload(row.get('response', {}))
            if not response_text:
                summary['parse_errors'] += 1
                summary['missing_response_chunks'] += 1
                bump_counter(summary['reason_counts'], 'missing_response_text')
                continue
            try:
                keyword_payload = normalize_keyword_extraction_payload(parse_json_payload(response_text))
                candidates = keyword_payload['candidates']
            except Exception:
                summary['parse_errors'] += 1
                bump_counter(summary['reason_counts'], 'failed_to_parse_keyword_json')
                continue

            summary['parsed_chunks'] += 1
            chunk_summary = keyword_payload.get('chunk_summary', '')
            if chunk_summary:
                matched_summary_items = match_keyword_items_by_ids(
                    keyword_payload.get('summary_evidence_item_ids'),
                    chunk,
                )
                if not matched_summary_items:
                    summary['ambiguous_summary_chunks'] += 1
                    bump_counter(summary['reason_counts'], 'ambiguous_summary_provenance')
                summary['chunk_summary_count'] += 1
                chunk_summaries.append(
                    {
                        'key': key,
                        'file_rel_path': chunk.get('file_rel_path', ''),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'line_numbers': chunk.get('line_numbers') or [],
                        'chunk_summary': chunk_summary,
                        'summary_evidence_item_ids': [
                            item.get('id') for item in matched_summary_items if item.get('id')
                        ],
                        'source_lines': sorted(
                            {item.get('line_number', 0) for item in matched_summary_items if item.get('line_number')}
                        ),
                    }
                )
            summary['candidate_count_raw'] += len(candidates)
            for candidate in candidates:
                matched_items = match_keyword_candidate_items(candidate, chunk)
                if not matched_items:
                    summary['ambiguous_provenance_candidates'] += 1
                    bump_counter(summary['reason_counts'], 'ambiguous_candidate_provenance')
                enriched = dict(candidate)
                enriched['source_files'] = [chunk.get('file_rel_path', '')] if chunk.get('file_rel_path') else []
                enriched['source_lines'] = sorted(
                    {item.get('line_number', 0) for item in matched_items if item.get('line_number')}
                )
                enriched['source_item_ids'] = [
                    item.get('id') for item in matched_items if item.get('id')
                ]
                enriched['evidence_items'] = [candidate['evidence']] if candidate.get('evidence') else []
                enriched['occurrences'] = 1
                key_tuple = keyword_candidate_key(enriched)
                if key_tuple in merged_candidates:
                    merge_keyword_candidate(merged_candidates[key_tuple], enriched)
                else:
                    merged_candidates[key_tuple] = enriched

    missing_keys = set(chunk_map.keys()) - processed_keys
    if missing_keys:
        summary['missing_chunk_rows'] = len(missing_keys)
        bump_counter(summary['reason_counts'], 'missing_chunk_rows', len(missing_keys))
    summary['processed_chunks'] = len(processed_keys)

    candidates = sorted(
        merged_candidates.values(),
        key=lambda item: (-item.get('confidence', 0.0), item.get('category', ''), item.get('source', '').lower()),
    )
    summary['candidate_count_deduped'] = len(candidates)

    jsonl_path = resolve_keyword_export_path(manifest, output_jsonl, 'keyword_candidates.jsonl', 'keyword JSONL output')
    markdown_path = resolve_keyword_export_path(manifest, output_markdown, 'keyword_candidates.md', 'keyword Markdown output')
    summary_jsonl_path = resolve_keyword_export_path(
        manifest,
        output_summary_jsonl,
        'keyword_chunk_summaries.jsonl',
        'keyword summary JSONL output',
    )
    summary_markdown_path = resolve_keyword_export_path(
        manifest,
        output_summary_markdown,
        'keyword_chunk_summaries.md',
        'keyword summary Markdown output',
    )
    validate_keyword_export_paths(manifest, jsonl_path, markdown_path, summary_jsonl_path, summary_markdown_path)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_markdown_path), exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf-8') as handle:
        for candidate in candidates:
            serializable = dict(candidate)
            serializable.pop('evidence_items', None)
            handle.write(json.dumps(serializable, ensure_ascii=False) + '\n')
    with open(summary_jsonl_path, 'w', encoding='utf-8') as handle:
        for item in chunk_summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    write_keyword_markdown(markdown_path, candidates, summary)
    write_keyword_summary_markdown(summary_markdown_path, chunk_summaries, summary)

    manifest['keyword_exported_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['keyword_export'] = {
        'jsonl_path': jsonl_path,
        'markdown_path': markdown_path,
        'summary_jsonl_path': summary_jsonl_path,
        'summary_markdown_path': summary_markdown_path,
        'summary': summary,
    }
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print(f'Keyword candidates: {summary["candidate_count_deduped"]} deduped from {summary["candidate_count_raw"]} raw')
    print(f'Chunk summaries: {summary["chunk_summary_count"]}')
    print(f'JSONL: {jsonl_path}')
    print(f'Markdown: {markdown_path}')
    print(f'Summary JSONL: {summary_jsonl_path}')
    print(f'Summary Markdown: {summary_markdown_path}')
    if summary.get('reason_counts'):
        print('Warnings:')
        for name in sorted(summary['reason_counts']):
            print(f"- {name}: {summary['reason_counts'][name]}")
    return manifest['keyword_export']


def append_failure_entries(entries, package_dir=''):
    if not entries:
        return

    entries = annotate_failure_entries(entries)
    ensure_batch_dirs()
    paths = [FAILED_LOG]
    if package_dir:
        paths.append(os.path.join(package_dir, 'failures.jsonl'))

    for path in paths:
        try:
            with open(path, 'a', encoding='utf-8') as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as exc:
            print(f'Warning: Could not write failure log {path}: {exc}')


def extract_string_token_text_at(line, start, end):
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except Exception:
        return None
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        if token.start[1] != start or token.end[1] != end:
            continue
        try:
            text_value = ast.literal_eval(token.string)
        except Exception:
            return None
        if not isinstance(text_value, str):
            return None
        return text_value
    return None


def unpack_replacement_for_validation(replacement):
    start, end, translated, prefix, quote = replacement[:5]
    source_text = replacement[5] if len(replacement) > 5 else ''
    item_id = replacement[6] if len(replacement) > 6 else ''
    chunk_key = replacement[7] if len(replacement) > 7 else ''
    return start, end, translated, prefix, quote, source_text, item_id, chunk_key


def make_failure_entry(manifest, error, file_rel_path='', item_id='', line=None, text='', **extra):
    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'package': manifest.get('_package_dir', ''),
        'error': error,
    }
    if file_rel_path:
        entry['file_rel_path'] = file_rel_path
    if item_id:
        entry['id'] = item_id
    if line is not None:
        entry['line'] = line
    if text:
        entry['text'] = text
    entry.update(extra)
    return entry


def validate_replacements_for_lines(manifest, file_key, replacements_by_line, lines, summary):
    validated_replacements = {}
    validated_lines = set()
    failure_entries = []
    skipped_items = 0
    source_mismatch_items = 0

    for line_idx, repls in replacements_by_line.items():
        for repl in repls:
            start, end, translated, _prefix, _quote, source_text, item_id, chunk_key = unpack_replacement_for_validation(repl)
            if line_idx < 0 or line_idx >= len(lines):
                skipped_items += 1
                bump_counter(summary['reason_counts'], 'source_line_missing')
                failure_entries.append(make_failure_entry(
                    manifest,
                    'Source line missing during source validation',
                    file_rel_path=file_key,
                    item_id=item_id,
                    line=line_idx,
                    text=source_text,
                    key=chunk_key,
                    start=start,
                    end=end,
                ))
                continue

            current_text = extract_string_token_text_at(lines[line_idx], start, end)
            if current_text != source_text:
                already_applied_text = current_text
                if already_applied_text is None:
                    current_token = extract_string_token_from_line(lines[line_idx])
                    if current_token:
                        already_applied_text = current_token.get('text')
                if already_applied_text in translated_text_variants(translated):
                    summary['already_applied_items'] = summary.get('already_applied_items', 0) + 1
                    validated_lines.add(line_idx)
                    continue
                skipped_items += 1
                source_mismatch_items += 1
                bump_counter(summary['reason_counts'], 'source_text_mismatch')
                failure_entries.append(make_failure_entry(
                    manifest,
                    'Source text mismatch during source validation',
                    file_rel_path=file_key,
                    item_id=item_id,
                    line=line_idx,
                    text=source_text,
                    key=chunk_key,
                    start=start,
                    end=end,
                    current_text=current_text if current_text is not None else '',
                ))
                continue

            validated_replacements.setdefault(line_idx, []).append(repl)
            validated_lines.add(line_idx)

    return validated_replacements, validated_lines, failure_entries, skipped_items, source_mismatch_items


def validate_result_replacements(manifest, replacements_by_file, summary):
    validated_replacements = {}
    validated_lines_by_file = {}
    failure_entries = []
    skipped_items = 0
    source_mismatch_items = 0
    candidate_items = summary.get('valid_items', 0)
    files_info = manifest.get('files') or {}

    for file_key, replacements_by_line in replacements_by_file.items():
        file_info = files_info.get(file_key)
        if not file_info:
            for line_idx, repls in replacements_by_line.items():
                for repl in repls:
                    start, end, _translated, _prefix, _quote, source_text, item_id, chunk_key = unpack_replacement_for_validation(repl)
                    skipped_items += 1
                    bump_counter(summary['reason_counts'], 'missing_manifest_file')
                    failure_entries.append(make_failure_entry(
                        manifest,
                        'Manifest file entry missing for result item',
                        file_rel_path=file_key,
                        item_id=item_id,
                        line=line_idx,
                        text=source_text,
                        key=chunk_key,
                        start=start,
                        end=end,
                    ))
            continue

        file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        if not os.path.isfile(file_path):
            for line_idx, repls in replacements_by_line.items():
                for repl in repls:
                    start, end, _translated, _prefix, _quote, source_text, item_id, chunk_key = unpack_replacement_for_validation(repl)
                    skipped_items += 1
                    bump_counter(summary['reason_counts'], 'target_file_missing')
                    failure_entries.append(make_failure_entry(
                        manifest,
                        'Target file missing during source validation',
                        file_rel_path=file_key,
                        item_id=item_id,
                        line=line_idx,
                        text=source_text,
                        key=chunk_key,
                        start=start,
                        end=end,
                        file=file_path,
                    ))
            continue

        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()

        file_replacements, file_lines, file_failures, file_skipped, file_mismatches = validate_replacements_for_lines(
            manifest,
            file_key,
            replacements_by_line,
            lines,
            summary,
        )
        if file_replacements or file_lines:
            validated_replacements[file_key] = file_replacements
            validated_lines_by_file[file_key] = file_lines
        failure_entries.extend(file_failures)
        skipped_items += file_skipped
        source_mismatch_items += file_mismatches

    pending_files = len(validated_replacements)
    pending_lines = sum(len(lines) for lines in validated_lines_by_file.values())
    summary['candidate_valid_items'] = candidate_items
    summary['valid_items'] = candidate_items - skipped_items
    summary['source_mismatch_items'] = source_mismatch_items
    summary['skipped_items'] = skipped_items
    summary['pending_files'] = pending_files
    summary['pending_lines'] = pending_lines
    return validated_replacements, validated_lines_by_file, failure_entries


def summarize_pending_replacements(replacements_by_file, translated_lines_by_file, summary):
    summary.setdefault('candidate_valid_items', summary.get('valid_items', 0))
    summary.setdefault('source_mismatch_items', 0)
    summary.setdefault('skipped_items', 0)
    summary['pending_files'] = len(replacements_by_file)
    summary['pending_lines'] = sum(len(lines) for lines in translated_lines_by_file.values())


def coerce_revision_should_update(value):
    return translation_core.coerce_revision_should_update(value)


def normalize_revision_items(payload):
    return translation_core.normalize_model_results(
        payload,
        mode=translation_core.MODE_REVISION,
    )


def make_revision_preview_entry(target_item, result_item, status, error=''):
    return {
        'id': target_item.get('id', ''),
        'file_rel_path': target_item.get('file_rel_path', ''),
        'line': target_item.get('line_number', target_item.get('line', 0)),
        'source': target_item.get('source', target_item.get('text', '')),
        'current_translation': target_item.get('current_translation', ''),
        'revised_translation': result_item.get('revised_translation', ''),
        'should_update': result_item.get('should_update', False),
        'reason': result_item.get('reason', ''),
        'status': status,
        'error': error,
    }


def reconcile_revision_preview_entries(preview_entries, validation_failures):
    failures_by_item = {}
    for failure in validation_failures:
        item_id = failure.get('item_id') or failure.get('id')
        if item_id:
            failures_by_item[str(item_id)] = failure
    if not failures_by_item:
        return preview_entries

    reconciled = []
    for entry in preview_entries:
        failure = failures_by_item.get(str(entry.get('id') or ''))
        if entry.get('status') != 'pending' or not failure:
            reconciled.append(entry)
            continue
        error = str(failure.get('error') or 'Source validation skipped this revision.')
        status = 'source_mismatch' if 'Source text mismatch' in error else 'skipped'
        updated = dict(entry)
        updated['status'] = status
        updated['error'] = error
        if failure.get('current_text') is not None:
            updated['current_text'] = failure.get('current_text')
        reconciled.append(updated)
    return reconciled


def is_v2_manifest(manifest):
    return manifest.get('manifest_version', 1) == 2 or manifest.get('version', 1) == 2


def relocate_v2_chunk_items(manifest, chunk, scanned_units_by_file, mode):
    if not is_v2_manifest(manifest):
        return []
    file_key = chunk['file_rel_path']
    if file_key not in scanned_units_by_file:
        file_path = manifest.get('files', {}).get(file_key, {}).get('path')
        if not file_path or not os.path.exists(file_path):
            file_path = os.path.join(legacy.TL_DIR, file_key)
        file_lines = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                file_lines = f.readlines()
        scanned_units_by_file[file_key] = legacy.scan_all_translation_units(
            file_lines,
            file_key,
            mode=mode,
        )

    scanned_map = scanned_units_by_file.get(file_key, {})
    missing_items = []
    for item in chunk.get('items') or []:
        scanned = scanned_map.get(item.get('id'))
        if not scanned:
            missing_items.append(item)
            continue
        scanned_line, scanned_start, scanned_end, _scanned_source = scanned
        item['line'] = scanned_line
        item['line_number'] = scanned_line + 1
        item['start'] = scanned_start
        item['end'] = scanned_end
    return missing_items


def record_v2_relocation_failures(manifest, chunk, missing_items, summary, failure_entries, key=''):
    if not missing_items:
        return set()
    bump_counter(summary['reason_counts'], 'v2_relocation_missing', len(missing_items))
    missing_ids = set()
    for item in missing_items:
        item_id = str(item.get('id') or '')
        if item_id:
            missing_ids.add(item_id)
        failure_entries.append(make_failure_entry(
            manifest,
            'V2 relocation missing for result item',
            file_rel_path=chunk.get('file_rel_path', ''),
            item_id=item_id,
            line=item.get('line'),
            text=item.get('source', item.get('text', '')),
            key=key or chunk.get('key', ''),
            reason_code='v2_relocation_missing',
        ))
    return missing_ids


def translated_text_variants(translated):
    variants = {translated}
    if getattr(legacy, 'USE_TRANSLATION_MEMORY', False):
        variants.add(legacy.apply_normalization(translated))
    return variants


def filter_non_translatable_noop_relocation_missing(missing_items, result_items):
    """Drop relocation misses that only preserve non-translatable source text."""
    if not missing_items:
        return []
    result_by_id = {
        str(item.get('id') or ''): item.get('translation', '')
        for item in result_items or []
        if isinstance(item, dict)
    }
    remaining = []
    for item in missing_items:
        source = item.get('text') or item.get('source') or ''
        item_id = str(item.get('id') or '')
        translated = result_by_id.get(item_id, '')
        if (
            legacy.is_non_translatable(source)
            and (translated or '').strip() == (source or '').strip()
        ):
            continue
        remaining.append(item)
    return remaining


def filter_already_applied_relocation_missing(manifest, chunk, missing_items, result_items, summary):
    if not missing_items:
        return []
    result_by_id = {
        str(item.get('id') or ''): item.get('translation', '')
        for item in result_items or []
        if isinstance(item, dict)
    }
    file_key = chunk.get('file_rel_path', '')
    file_info = manifest.get('files', {}).get(file_key)
    if not file_info:
        return missing_items
    file_path = resolve_manifest_file_path(manifest, file_key, file_info)
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
    except Exception:
        return missing_items

    remaining = []
    for item in missing_items:
        item_id = str(item.get('id') or '')
        translated = result_by_id.get(item_id, '')
        line_idx = item.get('line')
        if not translated or not isinstance(line_idx, int) or line_idx < 0 or line_idx >= len(lines):
            remaining.append(item)
            continue
        current_token = extract_string_token_from_line(lines[line_idx])
        current_text = current_token.get('text') if current_token else None
        if current_text in translated_text_variants(translated):
            summary['already_applied_items'] = summary.get('already_applied_items', 0) + 1
            continue
        remaining.append(item)
    return remaining


def collect_revision_actions(manifest, validate_sources=False):
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit('Result JSONL not found. Run download first.')

    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    replacements_by_file = {}
    revised_lines_by_file = {}
    scanned_units_by_file = {}
    processed_keys = set()
    failure_entries = []
    preview_entries = []
    summary = {
        'expected_chunks': len(chunk_map),
        'result_rows': 0,
        'expected_items': sum(len(chunk['items']) for chunk in chunk_map.values()),
        'parsed_items': 0,
        'valid_items': 0,
        'revision_candidate_items': 0,
        'unchanged_items': 0,
        'chunk_row_errors': 0,
        'missing_response_chunks': 0,
        'partial_chunks': 0,
        'max_tokens_chunks': 0,
        'reason_counts': {},
    }

    with open(result_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            summary['result_rows'] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'invalid_result_jsonl_row')
                failure_entries.append(make_failure_entry(manifest, f'Invalid result JSONL row: {exc}', text=line[:500]))
                continue

            key = row.get('key')
            if not key or key not in chunk_map:
                bump_counter(summary['reason_counts'], 'unknown_chunk_key')
                failure_entries.append(make_failure_entry(manifest, 'Unknown chunk key in result file', key=key))
                continue

            processed_keys.add(key)
            chunk = chunk_map[key]
            chunk_items = chunk['items']
            relocation_missing = relocate_v2_chunk_items(
                manifest,
                chunk,
                scanned_units_by_file,
                translation_core.MODE_REVISION,
            )
            relocation_missing_ids = record_v2_relocation_failures(
                manifest,
                chunk,
                relocation_missing,
                summary,
                failure_entries,
                key=key,
            )
            active_chunk_items = [
                item for item in chunk_items
                if str(item.get('id') or '') not in relocation_missing_ids
            ]
            if relocation_missing_ids and not active_chunk_items:
                continue
            item_map = {item['id']: item for item in active_chunk_items}
            response_payload = row.get('response', {})
            finish_reason = extract_finish_reason(response_payload)
            usage_metadata = summarize_usage_metadata(extract_usage_metadata(response_payload))
            if finish_reason == 'MAX_TOKENS':
                summary['max_tokens_chunks'] += 1

            if row.get('error'):
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'row_error')
                for item in active_chunk_items:
                    failure_entries.append(make_failure_entry(
                        manifest,
                        serialize_unknown(row.get('error')),
                        file_rel_path=chunk['file_rel_path'],
                        item_id=item['id'],
                        line=item['line'],
                        text=item.get('source', item.get('text', '')),
                        key=key,
                        finish_reason=finish_reason,
                        usage_metadata=usage_metadata,
                    ))
                continue

            response_text = extract_text_from_response_payload(response_payload)
            if not response_text:
                summary['missing_response_chunks'] += 1
                bump_counter(summary['reason_counts'], 'missing_response_text')
                for item in active_chunk_items:
                    failure_entries.append(make_failure_entry(
                        manifest,
                        'Missing text in response payload',
                        file_rel_path=chunk['file_rel_path'],
                        item_id=item['id'],
                        line=item['line'],
                        text=item.get('source', item.get('text', '')),
                        key=key,
                        finish_reason=finish_reason,
                        usage_metadata=usage_metadata,
                    ))
                continue

            try:
                payload = parse_json_payload(response_text)
                result_items = normalize_revision_items(payload)
            except Exception as exc:
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'failed_to_parse_revision_json'
                bump_counter(summary['reason_counts'], reason_name)
                for item in active_chunk_items:
                    failure_entries.append(make_failure_entry(
                        manifest,
                        f'Failed to parse revision JSON: {exc}',
                        file_rel_path=chunk['file_rel_path'],
                        item_id=item['id'],
                        line=item['line'],
                        text=item.get('source', item.get('text', '')),
                        key=key,
                        response_preview=response_text[:500],
                        finish_reason=finish_reason,
                        usage_metadata=usage_metadata,
                    ))
                continue

            if len(result_items) < len(active_chunk_items):
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_revision_items'
                bump_counter(summary['reason_counts'], reason_name)

            seen_ids = set()
            for result_item in result_items:
                result_id = result_item['id']
                if result_id in relocation_missing_ids:
                    continue
                target_item = item_map.get(result_id)
                if not target_item:
                    bump_counter(summary['reason_counts'], 'schema_or_item_mismatch')
                    continue
                if result_id in seen_ids:
                    bump_counter(summary['reason_counts'], 'duplicate_result_id')
                    continue
                seen_ids.add(result_id)
                summary['parsed_items'] += 1

                target_unit = translation_core.unit_from_manifest_item(
                    target_item,
                    mode=translation_core.MODE_REVISION,
                    chunk=chunk,
                )
                current_translation = target_unit.current_translation
                revised_translation = result_item.get('revised_translation', '')
                if not revised_translation and not result_item.get('should_update'):
                    revised_translation = current_translation
                    result_item['revised_translation'] = revised_translation
                should_update = result_item.get('should_update') and compact_text(revised_translation) != compact_text(current_translation)
                if not should_update:
                    summary['unchanged_items'] += 1
                    preview_entries.append(make_revision_preview_entry(target_item, result_item, 'unchanged'))
                    continue

                source_text = target_unit.source_text
                valid, reason = legacy.validate_translation(source_text, revised_translation)
                if not valid and reason == 'No Chinese characters' and allow_non_chinese_batch_translation(
                    manifest,
                    chunk,
                    source_text,
                    revised_translation,
                    item=target_item,
                ):
                    valid = True
                    reason = 'OK'
                if not valid:
                    bump_counter(summary['reason_counts'], 'validation_failed')
                    failure_entries.append(make_failure_entry(
                        manifest,
                        f'Validation failed: {reason}',
                        file_rel_path=chunk['file_rel_path'],
                        item_id=target_item['id'],
                        line=target_item['line'],
                        text=source_text,
                        key=key,
                        translation=revised_translation,
                        finish_reason=finish_reason,
                        usage_metadata=usage_metadata,
                    ))
                    preview_entries.append(make_revision_preview_entry(target_item, result_item, 'invalid', reason))
                    continue

                summary['valid_items'] += 1
                summary['revision_candidate_items'] += 1
                preview_entries.append(make_revision_preview_entry(target_item, result_item, 'pending'))
                action = translation_core.revision_writeback_action(
                    target_unit,
                    result_item,
                    chunk_key=key,
                )
                file_key = chunk['file_rel_path']
                replacements_by_file.setdefault(file_key, {}).setdefault(target_item['line'], []).append(
                    translation_core.writeback_tuple(action, include_expected=True)
                )
                revised_lines_by_file.setdefault(file_key, set()).add(target_item['line'])

            missing_ids = set(item_map.keys()) - seen_ids
            if missing_ids:
                bump_counter(summary['reason_counts'], 'response_missing_item_id', len(missing_ids))
            for missing_id in sorted(missing_ids):
                item = item_map[missing_id]
                failure_entries.append(make_failure_entry(
                    manifest,
                    'Response missing item id',
                    file_rel_path=chunk['file_rel_path'],
                    item_id=item['id'],
                    line=item['line'],
                    text=item.get('source', item.get('text', '')),
                    key=key,
                    finish_reason=finish_reason,
                    usage_metadata=usage_metadata,
                ))

    missing_keys = set(chunk_map.keys()) - processed_keys
    if missing_keys:
        bump_counter(summary['reason_counts'], 'missing_chunk_rows', len(missing_keys))
    for key in sorted(missing_keys):
        chunk = chunk_map[key]
        relocation_missing = relocate_v2_chunk_items(
            manifest,
            chunk,
            scanned_units_by_file,
            translation_core.MODE_REVISION,
        )
        relocation_missing_ids = record_v2_relocation_failures(
            manifest,
            chunk,
            relocation_missing,
            summary,
            failure_entries,
            key=key,
        )
        for item in chunk['items']:
            if str(item.get('id') or '') in relocation_missing_ids:
                continue
            failure_entries.append(make_failure_entry(
                manifest,
                'No result row found for chunk',
                file_rel_path=chunk['file_rel_path'],
                item_id=item['id'],
                line=item['line'],
                text=item.get('source', item.get('text', '')),
                key=key,
            ))

    summary['failure_items'] = len(failure_entries)
    summary['processed_chunks'] = len(processed_keys)
    if validate_sources:
        replacements_by_file, revised_lines_by_file, validation_failures = validate_result_replacements(
            manifest,
            replacements_by_file,
            summary,
        )
        failure_entries.extend(validation_failures)
        preview_entries = reconcile_revision_preview_entries(preview_entries, validation_failures)
        summary['failure_items'] = len(failure_entries)
    else:
        summarize_pending_replacements(replacements_by_file, revised_lines_by_file, summary)
    return replacements_by_file, revised_lines_by_file, failure_entries, summary, preview_entries


def print_revision_summary(summary):
    print(f"Expected chunks: {summary['expected_chunks']}")
    print(f"Result rows: {summary['result_rows']}")
    print(f"Processed chunks: {summary['processed_chunks']}")
    print(f"Expected items: {summary['expected_items']}")
    print(f"Parsed items: {summary.get('parsed_items', 0)}")
    if 'candidate_valid_items' in summary:
        print(f"Candidate revision items: {summary['candidate_valid_items']}")
    else:
        print(f"Candidate revision items: {summary.get('revision_candidate_items', 0)}")
    print(f"Recoverable revision items: {summary['valid_items']}")
    print(f"Unchanged items: {summary.get('unchanged_items', 0)}")
    print(f"Pending files: {summary.get('pending_files', 0)}")
    print(f"Pending lines: {summary.get('pending_lines', 0)}")
    print(f"Skipped items: {summary.get('skipped_items', 0)}")
    print(f"Source mismatches: {summary.get('source_mismatch_items', 0)}")
    print(f"Failure items: {summary['failure_items']}")
    print(f"Chunk row errors: {summary['chunk_row_errors']}")
    print(f"Missing-response chunks: {summary['missing_response_chunks']}")
    print(f"Partial/truncated chunks: {summary['partial_chunks']}")
    print(f"MAX_TOKENS chunks: {summary['max_tokens_chunks']}")
    if summary.get('reason_counts'):
        print('Failure categories:')
        for name in sorted(summary['reason_counts']):
            print(f"- {name}: {summary['reason_counts'][name]}")


def resolve_revision_output_path(manifest, value, default_name, field_name):
    package_dir = manifest.get('_package_dir')
    if value:
        return resolve_path_under_dir(package_dir, value, field_name)
    return os.path.join(package_dir, default_name)


def validate_revision_output_paths(manifest, jsonl_path, markdown_path):
    normalized_jsonl = _normalized_abs_path(jsonl_path)
    normalized_markdown = _normalized_abs_path(markdown_path)
    if normalized_jsonl == normalized_markdown:
        raise SystemExit('Revision preview JSONL and Markdown outputs must be different files.')

    reserved_paths = {
        os.path.join(manifest.get('_package_dir', ''), 'manifest.json'),
        os.path.join(manifest.get('_package_dir', ''), 'requests.jsonl'),
        os.path.join(manifest.get('_package_dir', ''), 'results.jsonl'),
        os.path.join(manifest.get('_package_dir', ''), 'failures.jsonl'),
    }
    for manifest_key in ('_manifest_path', 'input_jsonl_path', 'result_jsonl_path'):
        value = manifest.get(manifest_key)
        if value:
            reserved_paths.add(value)
    normalized_reserved = {_normalized_abs_path(path) for path in reserved_paths if path}
    for output_path in (jsonl_path, markdown_path):
        if _normalized_abs_path(output_path) in normalized_reserved:
            raise SystemExit(f'Revision preview output would overwrite reserved package file: {output_path}')


def write_revision_markdown(path, entries, summary):
    lines = [
        '# Revision Preview',
        '',
        f"- Pending revisions: {summary.get('valid_items', 0)}",
        f"- Unchanged items: {summary.get('unchanged_items', 0)}",
        f"- Failure items: {summary.get('failure_items', 0)}",
        '',
        '| Status | Source | Current | Revised | Reason | File | Line |',
        '| --- | --- | --- | --- | --- | --- | ---: |',
    ]
    for entry in entries:
        lines.append(
            '| '
            + ' | '.join(
                [
                    markdown_escape_cell(entry.get('status')),
                    markdown_escape_cell(entry.get('source')),
                    markdown_escape_cell(entry.get('current_translation')),
                    markdown_escape_cell(entry.get('revised_translation')),
                    markdown_escape_cell(entry.get('reason') or entry.get('error')),
                    markdown_escape_cell(entry.get('file_rel_path')),
                    markdown_escape_cell(entry.get('line')),
                ]
            )
            + ' |'
        )
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def preview_revisions(target=None, output_jsonl='', output_markdown=''):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_REVISION, 'preview-revisions')
    _replacements, _lines, _failure_entries, summary, preview_entries = collect_revision_actions(
        manifest,
        validate_sources=True,
    )
    jsonl_path = resolve_revision_output_path(manifest, output_jsonl, 'revision_preview.jsonl', 'revision JSONL output')
    markdown_path = resolve_revision_output_path(manifest, output_markdown, 'revision_preview.md', 'revision Markdown output')
    validate_revision_output_paths(manifest, jsonl_path, markdown_path)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf-8') as handle:
        for entry in preview_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    write_revision_markdown(markdown_path, preview_entries, summary)

    manifest['last_revision_preview_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['last_revision_preview'] = {
        'jsonl_path': jsonl_path,
        'markdown_path': markdown_path,
        'summary': summary,
    }
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    print_revision_summary(summary)
    print(f'Preview JSONL: {jsonl_path}')
    print(f'Preview Markdown: {markdown_path}')
    return manifest


def collect_result_actions(manifest, validate_sources=False):
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit('Result JSONL not found. Run download first.')

    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    replacements_by_file = {}
    translated_lines_by_file = {}
    scanned_units_by_file = {}
    processed_keys = set()
    failure_entries = []
    summary = {
        'expected_chunks': len(chunk_map),
        'result_rows': 0,
        'expected_items': sum(len(chunk['items']) for chunk in chunk_map.values()),
        'valid_items': 0,
        'chunk_row_errors': 0,
        'missing_response_chunks': 0,
        'partial_chunks': 0,
        'max_tokens_chunks': 0,
        'reason_counts': {},
    }

    with open(result_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            summary['result_rows'] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'invalid_result_jsonl_row')
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'package': manifest['_package_dir'],
                        'error': f'Invalid result JSONL row: {exc}',
                        'raw': line[:500],
                    }
                )
                continue

            key = row.get('key')
            if not key or key not in chunk_map:
                bump_counter(summary['reason_counts'], 'unknown_chunk_key')
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'package': manifest['_package_dir'],
                        'error': 'Unknown chunk key in result file',
                        'key': key,
                    }
                )
                continue

            processed_keys.add(key)
            chunk = chunk_map[key]
            chunk_items = chunk['items']
            relocation_missing = relocate_v2_chunk_items(
                manifest,
                chunk,
                scanned_units_by_file,
                translation_core.MODE_TRANSLATION,
            )
            response_payload = row.get('response', {})
            finish_reason = extract_finish_reason(response_payload)
            usage_metadata = summarize_usage_metadata(extract_usage_metadata(response_payload))
            if finish_reason == 'MAX_TOKENS':
                summary['max_tokens_chunks'] += 1

            if row.get('error'):
                relocation_missing_ids = record_v2_relocation_failures(
                    manifest,
                    chunk,
                    relocation_missing,
                    summary,
                    failure_entries,
                    key=key,
                )
                active_chunk_items = [
                    item for item in chunk_items
                    if str(item.get('id') or '') not in relocation_missing_ids
                ]
                if relocation_missing_ids and not active_chunk_items:
                    continue
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'row_error')
                for item in active_chunk_items:
                    failure_entries.append(
                        {
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'package': manifest['_package_dir'],
                            'key': key,
                            'file_rel_path': chunk['file_rel_path'],
                            'id': item['id'],
                            'line': item['line'],
                            'text': item['text'],
                            'error': serialize_unknown(row.get('error')),
                            'finish_reason': finish_reason,
                            'usage_metadata': usage_metadata,
                        }
                    )
                continue

            response_text = extract_text_from_response_payload(response_payload)
            if not response_text:
                relocation_missing_ids = record_v2_relocation_failures(
                    manifest,
                    chunk,
                    relocation_missing,
                    summary,
                    failure_entries,
                    key=key,
                )
                active_chunk_items = [
                    item for item in chunk_items
                    if str(item.get('id') or '') not in relocation_missing_ids
                ]
                if relocation_missing_ids and not active_chunk_items:
                    continue
                summary['missing_response_chunks'] += 1
                bump_counter(summary['reason_counts'], 'missing_response_text')
                for item in active_chunk_items:
                    failure_entries.append(
                        {
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'package': manifest['_package_dir'],
                            'key': key,
                            'file_rel_path': chunk['file_rel_path'],
                            'id': item['id'],
                            'line': item['line'],
                            'text': item['text'],
                            'error': 'Missing text in response payload',
                            'finish_reason': finish_reason,
                            'usage_metadata': usage_metadata,
                        }
                    )
                continue

            try:
                payload = parse_json_payload(response_text)
                result_items = normalize_result_items(payload)
            except Exception as exc:
                relocation_missing_ids = record_v2_relocation_failures(
                    manifest,
                    chunk,
                    relocation_missing,
                    summary,
                    failure_entries,
                    key=key,
                )
                active_chunk_items = [
                    item for item in chunk_items
                    if str(item.get('id') or '') not in relocation_missing_ids
                ]
                if relocation_missing_ids and not active_chunk_items:
                    continue
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'failed_to_parse_model_json'
                bump_counter(summary['reason_counts'], reason_name)
                for item in active_chunk_items:
                    failure_entries.append(
                        {
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'package': manifest['_package_dir'],
                            'key': key,
                            'file_rel_path': chunk['file_rel_path'],
                            'id': item['id'],
                            'line': item['line'],
                            'text': item['text'],
                            'error': f'Failed to parse model JSON: {exc}',
                            'response_preview': response_text[:500],
                            'finish_reason': finish_reason,
                            'usage_metadata': usage_metadata,
                        }
                    )
                continue

            relocation_missing = filter_already_applied_relocation_missing(
                manifest,
                chunk,
                relocation_missing,
                result_items,
                summary,
            )
            relocation_missing = filter_non_translatable_noop_relocation_missing(
                relocation_missing,
                result_items,
            )
            relocation_missing_ids = record_v2_relocation_failures(
                manifest,
                chunk,
                relocation_missing,
                summary,
                failure_entries,
                key=key,
            )
            active_chunk_items = [
                item for item in chunk_items
                if str(item.get('id') or '') not in relocation_missing_ids
            ]
            if relocation_missing_ids and not active_chunk_items:
                continue
            item_map = {item['id']: item for item in active_chunk_items}

            if len(result_items) < len(active_chunk_items):
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_result_items'
                bump_counter(summary['reason_counts'], reason_name)

            seen_ids = set()
            for result_item in result_items:
                result_id = result_item['id']
                if result_id in relocation_missing_ids:
                    continue
                target_item = item_map.get(result_id)
                if not target_item:
                    bump_counter(summary['reason_counts'], 'schema_or_item_mismatch')
                    continue
                if result_id in seen_ids:
                    bump_counter(summary['reason_counts'], 'duplicate_result_id')
                    continue
                seen_ids.add(result_id)

                target_unit = translation_core.unit_from_manifest_item(
                    target_item,
                    mode=translation_core.MODE_TRANSLATION,
                    chunk=chunk,
                )
                valid, reason = legacy.validate_translation(target_unit.text, result_item['translation'])
                if not valid and reason == 'No Chinese characters' and allow_non_chinese_batch_translation(
                    manifest,
                    chunk,
                    target_unit.text,
                    result_item['translation'],
                    item=target_item,
                ):
                    valid = True
                    reason = 'OK'
                if not valid:
                    bump_counter(summary['reason_counts'], 'validation_failed')
                    failure_entries.append(
                        {
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'package': manifest['_package_dir'],
                            'key': key,
                            'file_rel_path': chunk['file_rel_path'],
                            'id': target_item['id'],
                            'line': target_item['line'],
                            'text': target_unit.text,
                            'error': f'Validation failed: {reason}',
                            'translation': result_item['translation'],
                            'finish_reason': finish_reason,
                            'usage_metadata': usage_metadata,
                        }
                    )
                    continue

                summary['valid_items'] += 1
                action = translation_core.translation_writeback_action(
                    target_unit,
                    result_item,
                    chunk_key=key,
                )
                file_key = chunk['file_rel_path']
                replacements_by_file.setdefault(file_key, {}).setdefault(target_item['line'], []).append(
                    translation_core.writeback_tuple(action, include_expected=True)
                )
                translated_lines_by_file.setdefault(file_key, set()).add(target_item['line'])

            missing_ids = set(item_map.keys()) - seen_ids
            if missing_ids:
                bump_counter(summary['reason_counts'], 'response_missing_item_id', len(missing_ids))
            for missing_id in sorted(missing_ids):
                item = item_map[missing_id]
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'package': manifest['_package_dir'],
                        'key': key,
                        'file_rel_path': chunk['file_rel_path'],
                        'id': item['id'],
                        'line': item['line'],
                        'text': item['text'],
                        'error': 'Response missing item id',
                        'finish_reason': finish_reason,
                        'usage_metadata': usage_metadata,
                    }
                )

    missing_keys = set(chunk_map.keys()) - processed_keys
    if missing_keys:
        bump_counter(summary['reason_counts'], 'missing_chunk_rows', len(missing_keys))
    for key in sorted(missing_keys):
        chunk = chunk_map[key]
        relocation_missing = relocate_v2_chunk_items(
            manifest,
            chunk,
            scanned_units_by_file,
            translation_core.MODE_TRANSLATION,
        )
        relocation_missing_ids = record_v2_relocation_failures(
            manifest,
            chunk,
            relocation_missing,
            summary,
            failure_entries,
            key=key,
        )
        for item in chunk['items']:
            if str(item.get('id') or '') in relocation_missing_ids:
                continue
            failure_entries.append(
                {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'package': manifest['_package_dir'],
                    'key': key,
                    'file_rel_path': chunk['file_rel_path'],
                    'id': item['id'],
                    'line': item['line'],
                    'text': item['text'],
                    'error': 'No result row found for chunk',
                }
            )

    summary['failure_items'] = len(failure_entries)
    summary['processed_chunks'] = len(processed_keys)
    if validate_sources:
        replacements_by_file, translated_lines_by_file, validation_failures = validate_result_replacements(
            manifest,
            replacements_by_file,
            summary,
        )
        failure_entries.extend(validation_failures)
        summary['failure_items'] = len(failure_entries)
    else:
        summarize_pending_replacements(replacements_by_file, translated_lines_by_file, summary)
    return replacements_by_file, translated_lines_by_file, failure_entries, summary


def print_check_summary(summary):
    print(f"Expected chunks: {summary['expected_chunks']}")
    print(f"Result rows: {summary['result_rows']}")
    print(f"Processed chunks: {summary['processed_chunks']}")
    print(f"Expected items: {summary['expected_items']}")
    if 'candidate_valid_items' in summary:
        print(f"Candidate valid items: {summary['candidate_valid_items']}")
    print(f"Recoverable valid items: {summary['valid_items']}")
    print(f"Pending files: {summary.get('pending_files', 0)}")
    print(f"Pending lines: {summary.get('pending_lines', 0)}")
    print(f"Skipped items: {summary.get('skipped_items', 0)}")
    print(f"Source mismatches: {summary.get('source_mismatch_items', 0)}")
    print(f"Failure items: {summary['failure_items']}")
    print(f"Chunk row errors: {summary['chunk_row_errors']}")
    print(f"Missing-response chunks: {summary['missing_response_chunks']}")
    print(f"Partial/truncated chunks: {summary['partial_chunks']}")
    print(f"MAX_TOKENS chunks: {summary['max_tokens_chunks']}")
    if summary.get('reason_counts'):
        print('Failure categories:')
        for name in sorted(summary['reason_counts']):
            print(f"- {name}: {summary['reason_counts'][name]}")
    if summary.get('safety_level'):
        print(f"Safety status: {summary['safety_level']}")
        safety_reasons = summary.get('safety_reasons') or {}
        for status in (CHECK_SAFETY_WARN, CHECK_SAFETY_BLOCK):
            reasons = safety_reasons.get(status) or {}
            if reasons:
                print(f"{status.capitalize()} reasons:")
                for name in sorted(reasons):
                    print(f"- {name}: {reasons[name]}")


def probe_requests(target=None, limit=3, offset=0, api_key_index=None):
    manifest = load_manifest(target)
    rows = load_request_rows(manifest)
    if offset < 0:
        offset = 0
    if limit <= 0:
        raise SystemExit('--limit must be greater than 0.')
    sample = rows[offset:offset + limit]
    if not sample:
        raise SystemExit('No request rows available for the requested probe range.')

    client = create_batch_client(api_key_index=api_key_index)
    summary = {
        'sample_count': len(sample),
        'parse_ok': 0,
        'full_item_match': 0,
        'max_tokens': 0,
        'missing_text': 0,
        'request_errors': 0,
    }
    probe_results = []

    for index, row in enumerate(sample, start=1):
        key = row.get('key', f'probe-{index}')
        request_payload = row.get('request') or {}
        config = dict(request_payload.get('generation_config') or {})
        system_instruction = request_payload.get('system_instruction')
        if system_instruction:
            config['system_instruction'] = system_instruction
        safety_settings = request_payload.get('safety_settings')
        if safety_settings:
            config['safety_settings'] = safety_settings

        expected_items = len(((manifest.get('chunks') or []) and next((chunk['items'] for chunk in manifest['chunks'] if chunk['key'] == key), [])) or [])
        parse_ok = False
        parsed_items = 0
        parse_error = ''
        finish_reason = ''
        usage_metadata = {}
        response_text = ''
        try:
            response = client.models.generate_content(
                model=manifest.get('batch_model') or BATCH_MODEL,
                contents=request_payload.get('contents') or [],
                config=config,
            )
            response_payload = serialize_unknown(response)
            finish_reason = extract_finish_reason(response_payload)
            usage_metadata = summarize_usage_metadata(extract_usage_metadata(response_payload))
            response_text = extract_text_from_response_payload(response_payload)
        except Exception as exc:
            summary['request_errors'] += 1
            parse_error = str(exc)
        if response_text:
            try:
                payload = parse_json_payload(response_text)
                result_items = normalize_result_items(payload)
                parsed_items = len(result_items)
                parse_ok = True
            except Exception as exc:
                parse_error = str(exc)
        else:
            summary['missing_text'] += 1
            if not parse_error:
                parse_error = 'Missing text in response payload'

        if finish_reason == 'MAX_TOKENS':
            summary['max_tokens'] += 1
        if parse_ok:
            summary['parse_ok'] += 1
        if parse_ok and parsed_items == expected_items:
            summary['full_item_match'] += 1

        probe_row = {
            'index': index,
            'key': key,
            'finish_reason': finish_reason,
            'usage_metadata': usage_metadata,
            'expected_items': expected_items,
            'parsed_items': parsed_items,
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'response_preview': response_text[:500] if response_text else '',
        }
        probe_results.append(probe_row)
        print(f"[{index}/{len(sample)}] {key}")
        print(f"  finish_reason: {finish_reason or '(none)'}")
        print(f"  usage: {usage_metadata or {}}")
        print(f"  parsed_items: {parsed_items}/{expected_items}")
        print(f"  parse_ok: {parse_ok}")
        if parse_error:
            print(f"  parse_error: {parse_error}")

    summary_path = os.path.join(manifest['_package_dir'], 'probe_summary.json')
    results_path = os.path.join(manifest['_package_dir'], 'probe_results.jsonl')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(results_path, 'w', encoding='utf-8') as handle:
        for row in probe_results:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('Probe summary:')
    print(f"- sample_count: {summary['sample_count']}")
    print(f"- parse_ok: {summary['parse_ok']}")
    print(f"- full_item_match: {summary['full_item_match']}")
    print(f"- max_tokens: {summary['max_tokens']}")
    print(f"- missing_text: {summary['missing_text']}")
    print(f"- request_errors: {summary['request_errors']}")
    print(f"- summary_file: {summary_path}")
    print(f"- results_file: {results_path}")
    return summary


def check_results(target=None):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'check')
    _replacements, _translated, failure_entries, summary = collect_result_actions(manifest, validate_sources=True)
    attach_check_contract(manifest, summary)
    check_report_path = write_check_failure_report(manifest, failure_entries)
    manifest['last_check_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['last_check_summary'] = summary
    manifest['last_check_report_path'] = check_report_path
    manifest.pop('last_apply_failure_report_path', None)
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    print(f"Manifest: {manifest['_manifest_path']}")
    print_check_summary(summary)
    print(f"Check failure report: {check_report_path}")
    return manifest


def apply_results(target=None, force=False):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'apply')
    if manifest.get('applied_at') and not force:
        raise SystemExit('Manifest was already applied. Re-run apply with --force to bypass this guard; source validation still applies.')
    require_safe_check_for_apply(manifest)

    replacements_by_file, translated_lines_by_file, failure_entries, summary = collect_result_actions(
        manifest,
        validate_sources=True,
    )
    attach_check_contract(manifest, summary)
    if summary.get('safety_level') != CHECK_SAFETY_SAFE:
        append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])
        report_path = write_apply_failure_report(
            manifest,
            'unsafe_apply_recheck',
            f'Apply recheck status is {summary.get("safety_level")}, not safe. No files were written.',
            summary=summary,
            failure_entries=failure_entries,
            current_fingerprint=summary.get('check_fingerprint'),
        )
        save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
        raise SystemExit(f'Apply refused because current results are not safe. Report: {report_path}')

    applied_files = 0
    applied_lines = 0
    revalidated_replacements_by_file = {}
    revalidated_line_numbers_by_file = {}
    revalidated_file_paths = {}
    revalidated_file_lines = {}
    rag_jobs = []
    file_keys = set(replacements_by_file) | set(translated_lines_by_file)
    for file_key in file_keys:
        replacements = replacements_by_file.get(file_key, {})
        file_info = manifest['files'].get(file_key)
        if not file_info:
            continue
        file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
        replacements, line_numbers_set, revalidation_failures, revalidated_skipped, revalidated_mismatches = validate_replacements_for_lines(
            manifest,
            file_key,
            replacements,
            lines,
            summary,
        )
        line_numbers_set.update(translated_lines_by_file.get(file_key, set()))
        if revalidated_skipped:
            summary['valid_items'] = max(0, summary['valid_items'] - revalidated_skipped)
            summary['skipped_items'] = summary.get('skipped_items', 0) + revalidated_skipped
            summary['source_mismatch_items'] = summary.get('source_mismatch_items', 0) + revalidated_mismatches
            failure_entries.extend(revalidation_failures)
            summary['failure_items'] = len(failure_entries)
        if not replacements and not line_numbers_set:
            continue
        revalidated_replacements_by_file[file_key] = replacements
        revalidated_line_numbers_by_file[file_key] = set(line_numbers_set)
        revalidated_file_paths[file_key] = file_path
        revalidated_file_lines[file_key] = lines

    attach_check_contract(manifest, summary)
    if summary.get('safety_level') != CHECK_SAFETY_SAFE:
        append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])
        report_path = write_apply_failure_report(
            manifest,
            'unsafe_apply_revalidation',
            f'Apply source revalidation status is {summary.get("safety_level")}, not safe. No files were written.',
            summary=summary,
            failure_entries=failure_entries,
            current_fingerprint=summary.get('check_fingerprint'),
        )
        save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
        raise SystemExit(f'Apply refused because source revalidation is not safe. Report: {report_path}')

    for file_key, replacements in revalidated_replacements_by_file.items():
        file_path = revalidated_file_paths[file_key]
        lines = revalidated_file_lines[file_key]
        if replacements:
            legacy.commit_replacements(file_path, lines, replacements)
        line_numbers = sorted(revalidated_line_numbers_by_file[file_key])
        update_progress(file_key, line_numbers)
        applied_files += 1
        applied_lines += len(line_numbers)
        if line_numbers:
            rag_jobs.append({'file_rel_path': file_key, 'file_path': file_path})

    summary['pending_files'] = applied_files
    summary['pending_lines'] = applied_lines

    rag_apply_summary = {}
    if RAG_ENABLED and rag_jobs:
        rag_apply_summary = sync_rag_store_for_jobs(rag_jobs, quality_state='batch_applied')

    manifest['applied_at'] = datetime.now().isoformat(timespec='seconds')
    manifest.pop('last_apply_failure_report_path', None)
    manifest['apply_summary'] = {
        'applied_files': applied_files,
        'applied_lines': applied_lines,
        'candidate_items': summary.get('candidate_valid_items', summary['valid_items']),
        'recoverable_items': summary['valid_items'],
        'skipped_items': summary.get('skipped_items', 0),
        'source_mismatch_items': summary.get('source_mismatch_items', 0),
        'failure_count': len(failure_entries),
        'rag': rag_apply_summary,
    }
    next_split_manifest = mark_next_split_after_apply(manifest)
    should_update_latest = manifest.get('execution') != 'sync'
    save_manifest(manifest, update_latest=should_update_latest and not next_split_manifest)
    if next_split_manifest and should_update_latest:
        remember_latest_manifest(next_split_manifest)

    print_check_summary(summary)
    print(f'Applied files: {applied_files}')
    print(f'Applied lines: {applied_lines}')
    print(f'Failures logged: {len(failure_entries)}')
    if rag_apply_summary:
        print(f"RAG store updated: {rag_apply_summary.get('upserted', 0)} entries")
    print_next_split_after_apply(next_split_manifest)
    if failure_entries:
        print(f"Failure log: {os.path.join(manifest['_package_dir'], 'failures.jsonl')}")
    return manifest


def apply_revisions(target=None, force=False):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_REVISION, 'apply-revisions')
    if manifest.get('revision_applied_at') and not force:
        raise SystemExit('Revision manifest was already applied. Re-run apply-revisions with --force to bypass this guard; source validation still applies.')

    replacements_by_file, _revised_lines_by_file, failure_entries, summary, preview_entries = collect_revision_actions(
        manifest,
        validate_sources=True,
    )

    applied_files = 0
    applied_lines = 0
    final_pending_files = 0
    final_pending_lines = 0
    rag_jobs = []
    for file_key, replacements in replacements_by_file.items():
        file_info = manifest['files'].get(file_key)
        if not file_info:
            continue
        file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
        replacements, line_numbers_set, revalidation_failures, revalidated_skipped, revalidated_mismatches = validate_replacements_for_lines(
            manifest,
            file_key,
            replacements,
            lines,
            summary,
        )
        if revalidated_skipped:
            summary['valid_items'] = max(0, summary['valid_items'] - revalidated_skipped)
            summary['skipped_items'] = summary.get('skipped_items', 0) + revalidated_skipped
            summary['source_mismatch_items'] = summary.get('source_mismatch_items', 0) + revalidated_mismatches
            failure_entries.extend(revalidation_failures)
            summary['failure_items'] = len(failure_entries)
        if not replacements and not line_numbers_set:
            continue
        if replacements:
            legacy.commit_replacements(file_path, lines, replacements)
        line_numbers = sorted(line_numbers_set)
        update_progress(file_key, line_numbers)
        applied_files += 1
        applied_lines += len(line_numbers)
        final_pending_files += 1
        final_pending_lines += len(line_numbers)
        if line_numbers:
            rag_jobs.append({'file_rel_path': file_key, 'file_path': file_path})

    summary['pending_files'] = final_pending_files
    summary['pending_lines'] = final_pending_lines
    append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])

    rag_apply_summary = {}
    if RAG_ENABLED and rag_jobs:
        rag_apply_summary = sync_rag_store_for_jobs(rag_jobs, quality_state='revision_applied')

    manifest['revision_applied_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['revision_apply_summary'] = {
        'applied_files': applied_files,
        'applied_lines': applied_lines,
        'candidate_items': summary.get('candidate_valid_items', summary['valid_items']),
        'recoverable_items': summary['valid_items'],
        'unchanged_items': summary.get('unchanged_items', 0),
        'skipped_items': summary.get('skipped_items', 0),
        'source_mismatch_items': summary.get('source_mismatch_items', 0),
        'failure_count': len(failure_entries),
        'rag': rag_apply_summary,
    }
    manifest['last_revision_apply_summary'] = summary
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print_revision_summary(summary)
    print(f'Applied files: {applied_files}')
    print(f'Applied lines: {applied_lines}')
    print(f'Failures logged: {len(failure_entries)}')
    if rag_apply_summary:
        print(f"RAG store updated: {rag_apply_summary.get('upserted', 0)} entries")
    if failure_entries:
        print(f"Failure log: {os.path.join(manifest['_package_dir'], 'failures.jsonl')}")
    return manifest


REPAIR_LINE_COMMENT_RE = re.compile(r'^\s*#\s*(?P<prefix>[^\"]*?)"(?P<text>.*)"\s*$')
REPAIR_OLD_LINE_RE = re.compile(r'^\s*old\s+"(?P<text>.*)"\s*$')
REPAIR_NEW_LINE_RE = re.compile(r'^\s*new\s+"(?P<text>.*)"\s*$')


def is_voice_comment_match(match):
    if not match:
        return False
    prefix = str(match.group('prefix') or '').strip()
    return prefix.split(None, 1)[0:1] == ['voice']


def is_voice_statement_line(line):
    stripped = str(line or '').strip()
    return stripped == 'voice' or stripped.startswith('voice ')


def next_translation_entry_target_index(lines, index):
    next_index = index + 1
    while next_index < len(lines):
        candidate = lines[next_index]
        if not candidate.strip() or is_voice_statement_line(candidate):
            next_index += 1
            continue
        break
    return next_index


def write_jsonl_file(path, entries):
    with open(path, 'w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


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
        prefix, quote = legacy.parse_string_literal_format(token.string)
        return {
            'text': text_value,
            'start': token.start[1],
            'end': token.end[1],
            'prefix': prefix,
            'quote': quote,
        }
    return None


def infer_repair_speaker_id(prefix='', line='', string_start_col=None):
    if line and string_start_col is not None:
        stripped = line.lstrip()
        if not stripped.startswith(('old ', 'new ')):
            speaker_id = legacy.infer_dialogue_speaker_id(line, string_start_col)
            if speaker_id:
                return speaker_id

    prefix = str(prefix or '')
    if not prefix.strip():
        return ''
    return legacy.infer_dialogue_speaker_id(f'{prefix}"x"', len(prefix))


def build_identity_v2_by_span(lines, file_rel_path):
    if not file_rel_path:
        return {}
    try:
        units = legacy.scan_all_translation_units(
            lines,
            file_rel_path,
            mode=translation_core.MODE_REVISION,
        )
    except Exception:
        return {}
    return {
        (line_idx + 1, start, end): unit_id
        for unit_id, (line_idx, start, end, _text) in units.items()
    }


def attach_identity_v2(entry, identity_v2_by_span):
    identity = identity_v2_by_span.get(
        (
            entry.get('line_number'),
            entry.get('start'),
            entry.get('end'),
        )
    )
    if identity:
        entry['identity_v2'] = identity
    return entry


def collect_translation_entries_from_lines(lines, file_rel_path=''):
    entries = []
    identity_v2_by_span = build_identity_v2_by_span(lines, file_rel_path)
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip('\n')
        comment_match = REPAIR_LINE_COMMENT_RE.match(raw_line)
        if comment_match:
            if is_voice_comment_match(comment_match):
                index += 1
                continue
            next_index = next_translation_entry_target_index(lines, index)
            if next_index < len(lines):
                token = extract_string_token_from_line(lines[next_index])
                if token:
                    speaker_id = infer_repair_speaker_id(
                        comment_match.group('prefix'),
                        lines[next_index],
                        token['start'],
                    )
                    entry = {
                        'line_number': next_index + 1,
                        'source_line_number': index + 1,
                        'source': comment_match.group('text'),
                        'translation': token['text'],
                        'start': token['start'],
                        'end': token['end'],
                        'prefix': token.get('prefix', ''),
                        'quote': token['quote'],
                    }
                    if speaker_id:
                        entry['speaker_id'] = speaker_id
                        entry['speaker'] = speaker_id
                    entries.append(attach_identity_v2(entry, identity_v2_by_span))
            index = next_index
        else:
            old_match = REPAIR_OLD_LINE_RE.match(raw_line)
            if old_match:
                next_index = index + 1
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                if next_index < len(lines) and REPAIR_NEW_LINE_RE.match(lines[next_index].rstrip('\n')):
                    token = extract_string_token_from_line(lines[next_index])
                    if token:
                        entry = {
                            'line_number': next_index + 1,
                            'source_line_number': index + 1,
                            'source': old_match.group('text'),
                            'translation': token['text'],
                            'start': token['start'],
                            'end': token['end'],
                            'quote': token['quote'],
                        }
                        entries.append(attach_identity_v2(entry, identity_v2_by_span))
                index = next_index
        index += 1

    for entry_index, entry in enumerate(entries):
        entry['entry_index'] = entry_index
    return entries


def collect_repair_entries_from_lines(lines):
    entries = collect_translation_entries_from_lines(lines)
    seen_spans = {
        (entry.get('line_number'), entry.get('start'), entry.get('end'))
        for entry in entries
    }

    for task in legacy.collect_tasks(lines):
        span = (int(task['line']) + 1, task.get('start'), task.get('end'))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        entries.append(
            {
                'line_number': span[0],
                'source_line_number': span[0],
                'source': task.get('text', ''),
                'translation': task.get('text', ''),
                'start': task.get('start', 0),
                'end': task.get('end', 0),
                'prefix': task.get('prefix', ''),
                'quote': task.get('quote', '"'),
                'speaker_id': task.get('speaker_id', ''),
                'speaker': task.get('speaker', ''),
            }
        )

    entries.sort(key=lambda entry: (entry.get('line_number', 0), entry.get('start', 0), entry.get('end', 0)))
    for entry_index, entry in enumerate(entries):
        entry['entry_index'] = entry_index
    return entries


def parse_repair_start_hint(item):
    for key in ('start', 'column', 'col'):
        try:
            if item.get(key) is not None:
                return int(item.get(key))
        except (TypeError, ValueError):
            pass

    raw_id = item.get('id')
    if not raw_id:
        return None
    numeric_suffix = []
    for part in reversed(str(raw_id).split(':')):
        try:
            numeric_suffix.append(int(part))
        except (TypeError, ValueError):
            break
    numeric_suffix.reverse()
    if len(numeric_suffix) < 2:
        return None

    try:
        item_line = int(item.get('line'))
    except (TypeError, ValueError):
        return None

    candidates = []
    if len(numeric_suffix) >= 3:
        candidates.append((numeric_suffix[-3], numeric_suffix[-2]))
    candidates.append((numeric_suffix[-2], numeric_suffix[-1]))

    for line_hint, start in candidates:
        if line_hint == item_line or line_hint + 1 == item_line:
            return start
    return None


def find_repair_entry_for_item(item, candidates):
    if not candidates:
        return None
    source = item.get('source', '')
    start_hint = parse_repair_start_hint(item)

    if start_hint is not None:
        for candidate in candidates:
            if candidate.get('start') == start_hint and candidate.get('source') == source:
                return candidate
        for candidate in candidates:
            if candidate.get('start') == start_hint:
                return candidate

    for candidate in candidates:
        if candidate.get('source') == source or candidate.get('translation') == source:
            return candidate

    return candidates[0] if len(candidates) == 1 else None


def should_index_rag_entry(entry):
    source = compact_text(entry.get('source', ''))
    translation = compact_text(entry.get('translation', ''))
    if not source or not translation:
        return False
    if source == translation:
        return False
    return True


def build_rag_record(file_rel_path, group, quality_state):
    source_text = '\n'.join(entry.get('source', '') for entry in group).strip()
    translated_text = '\n'.join(entry.get('translation', '') for entry in group).strip()
    line_start = group[0]['line_number']
    line_end = group[-1]['line_number']
    combined_text = f"Source:\n{source_text}\n\nTranslation:\n{translated_text}"
    memory_id = hash_key(f"{file_rel_path}:{line_start}:{line_end}:{source_text}")
    return {
        'memory_id': memory_id,
        'file_rel_path': file_rel_path,
        'line_start': line_start,
        'line_end': line_end,
        'source_text': source_text,
        'translated_text': translated_text,
        'combined_text': combined_text,
        'quality_state': quality_state,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'source_checksum': hash_text(source_text),
        'translation_checksum': hash_text(translated_text),
    }


def collect_rag_seed_records_for_jobs(file_jobs, quality_state='seed'):
    records = []
    segment_size = max(1, RAG_SEGMENT_LINES)
    for job in file_jobs:
        file_rel_path = job.get('file_rel_path')
        file_path = job.get('file_path')
        if not file_rel_path or not file_path or not os.path.isfile(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            entries = collect_translation_entries_from_lines(handle.readlines())
        usable_entries = [entry for entry in entries if should_index_rag_entry(entry)]
        for start in range(0, len(usable_entries), segment_size):
            group = usable_entries[start:start + segment_size]
            if group:
                records.append(build_rag_record(file_rel_path, group, quality_state))
    return records


def build_source_segment(file_rel_path, group):
    source_text = '\n'.join(entry.get('source', '') for entry in group).strip()
    line_start = group[0]['line_number']
    line_end = group[-1]['line_number']
    source_id = hash_key(f"{file_rel_path}:{line_start}:{line_end}")
    source_checksum = hash_text(source_text)
    now = datetime.now().isoformat(timespec='seconds')
    return {
        'source_id': source_id,
        'file_rel_path': file_rel_path,
        'line_start': line_start,
        'line_end': line_end,
        'line_span': [line_start, line_end],
        'source_text': source_text,
        'source_checksum': source_checksum,
        'embedding': [],
        'embedding_metadata': {},
        'created_at': now,
        'updated_at': now,
    }


def collect_source_segments_for_jobs(file_jobs):
    records = []
    segment_size = max(1, RAG_SEGMENT_LINES)
    for job in file_jobs:
        file_rel_path = job.get('file_rel_path')
        file_path = job.get('file_path')
        if not file_rel_path or not file_path or not os.path.isfile(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            entries = collect_translation_entries_from_lines(handle.readlines(), file_rel_path=file_rel_path)
        usable_entries = []
        for entry in entries:
            src = (entry.get('source') or '').strip()
            if src:
                usable_entries.append(entry)
        for start in range(0, len(usable_entries), segment_size):
            group = usable_entries[start:start + segment_size]
            if group:
                records.append(build_source_segment(file_rel_path, group))
    return records


def coerce_external_seed_text(row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def coerce_external_seed_line(value, default=None):
    try:
        line_number = int(value)
    except (TypeError, ValueError):
        return default
    return line_number if line_number > 0 else default


def hash_file_contents(path):
    digest = hashlib.sha1()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:10]


def external_seed_source_name(seed_path):
    return f'external/{hash_file_contents(seed_path)}'


def build_external_rag_seed_record(row, source_name, row_number, quality_state='external_seed'):
    if not isinstance(row, dict):
        return None

    source_text = coerce_external_seed_text(row, ('source_text', 'source'))
    translated_text = coerce_external_seed_text(row, ('translated_text', 'translation', 'target'))
    if not should_index_rag_entry({'source': source_text, 'translation': translated_text}):
        return None

    file_rel_path = row.get('file_rel_path') or row.get('file') or source_name
    if not isinstance(file_rel_path, str) or not file_rel_path.strip():
        file_rel_path = source_name
    file_rel_path = legacy._normalize_rel_path(file_rel_path.strip())

    line_start = coerce_external_seed_line(row.get('line_start'))
    if line_start is None:
        line_start = coerce_external_seed_line(row.get('line'), row_number)
    line_end = coerce_external_seed_line(row.get('line_end'), line_start)
    if line_end < line_start:
        line_end = line_start

    memory_id = row.get('memory_id')
    if not isinstance(memory_id, str) or not memory_id.strip():
        memory_id = hash_key(f'external:{file_rel_path}:{line_start}:{line_end}:{source_text}')
    else:
        memory_id = memory_id.strip()

    combined_text = f"Source:\n{source_text}\n\nTranslation:\n{translated_text}"
    return {
        'memory_id': memory_id,
        'file_rel_path': file_rel_path,
        'line_start': line_start,
        'line_end': line_end,
        'source_text': source_text,
        'translated_text': translated_text,
        'combined_text': combined_text,
        'quality_state': quality_state,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'source_checksum': hash_text(source_text),
        'translation_checksum': hash_text(translated_text),
    }


def load_external_rag_seed_records(seed_jsonl_paths, quality_state='external_seed'):
    records = []
    invalid_json = 0
    filtered = 0
    paths = [path for path in (seed_jsonl_paths or []) if path]
    for seed_path in paths:
        if not os.path.isfile(seed_path):
            raise SystemExit(f'External RAG seed JSONL not found: {seed_path}')
        source_name = external_seed_source_name(seed_path)
        with open(seed_path, 'r', encoding='utf-8-sig') as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                record = build_external_rag_seed_record(row, source_name, row_number, quality_state=quality_state)
                if record is None:
                    filtered += 1
                    continue
                records.append(record)
    return records, {
        'external_seed_files': len(paths),
        'external_seed_records': len(records),
        'external_seed_invalid_json': invalid_json,
        'external_seed_filtered': filtered,
        'external_seed_skipped': invalid_json + filtered,
    }


def embed_history_records(records, *, progress_offset=0, progress_total=None):
    embedded_records = []
    batch_size = 16
    total = len(records) if progress_total is None else progress_total
    if total:
        completed = min(max(progress_offset, 0), total)
        print(f'RAG update progress: {completed}/{total} records.', flush=True)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        vectors = embed_texts([record['source_text'] for record in batch], RAG_DOCUMENT_TASK_TYPE)
        for record, vector in zip(batch, vectors):
            enriched = dict(record)
            enriched['embedding'] = vector
            enriched['embedding_model'] = RAG_EMBEDDING_MODEL
            enriched['embedding_task_type'] = RAG_DOCUMENT_TASK_TYPE
            enriched['embedding_dim'] = len(vector)
            enriched['embedding_text_kind'] = 'source_text'
            enriched['embedding_text_checksum'] = hash_text(record.get('source_text', ''))
            embedded_records.append(enriched)
        if total:
            completed = min(max(progress_offset + len(embedded_records), 0), total)
            print(f'RAG update progress: {completed}/{total} records.', flush=True)
    return embedded_records


def has_current_source_embedding(existing, record):
    return (
        existing
        and existing.get('source_checksum') == record['source_checksum']
        and existing.get('embedding_model') == RAG_EMBEDDING_MODEL
        and existing.get('embedding_task_type') == RAG_DOCUMENT_TASK_TYPE
        and existing.get('embedding_dim') == RAG_OUTPUT_DIMENSIONALITY
        and existing.get('embedding_text_kind') == 'source_text'
        and existing.get('embedding_text_checksum') == hash_text(record.get('source_text', ''))
        and isinstance(existing.get('embedding'), list)
        and bool(existing.get('embedding'))
    )


def reuse_existing_source_embedding(record, existing):
    enriched = dict(record)
    for key in (
        'embedding',
        'embedding_model',
        'embedding_task_type',
        'embedding_dim',
        'embedding_text_kind',
        'embedding_text_checksum',
    ):
        enriched[key] = existing.get(key)
    return enriched


def all_rag_file_jobs():
    return [
        {'file_rel_path': rel_path, 'file_path': file_path}
        for rel_path, file_path in collect_files_to_process()
    ]


def sync_rag_store_for_jobs(
    file_jobs,
    quality_state='seed',
    scan_all_files=False,
    extra_records=None,
    extra_summary=None,
):
    if not RAG_ENABLED:
        return {'enabled': False}
    store = get_rag_store()
    if store is None:
        return {'enabled': True, 'error': 'RAG store unavailable'}

    scan_jobs = all_rag_file_jobs() if scan_all_files else file_jobs
    base_records = collect_rag_seed_records_for_jobs(scan_jobs, quality_state=quality_state)
    base_records.extend(extra_records or [])
    records_to_embed = []
    records_with_reused_embedding = []
    for record in base_records:
        existing = store.get_history_record(record['memory_id'])
        if not existing:
            store.load()
            for hist_rec in store.history.values():
                if hist_rec.get('source_checksum') == record['source_checksum']:
                    existing = hist_rec
                    break
        if has_current_source_embedding(existing, record):
            if (existing.get('translation_checksum') == record['translation_checksum']
                    and existing.get('memory_id') == record['memory_id']):
                continue
            records_with_reused_embedding.append(reuse_existing_source_embedding(record, existing))
        else:
            records_to_embed.append(record)
    pending_records = records_with_reused_embedding + records_to_embed

    stats = {
        'enabled': True,
        'store_dir': store.store_dir,
        'scan_scope': 'all_files' if scan_all_files else 'pending_files',
        'files_scanned': len(scan_jobs),
        'scanned': len(base_records),
        'pending': len(pending_records),
        'embedding_pending': len(records_to_embed),
        'reused_embeddings': len(records_with_reused_embedding),
        'embedded': 0,
        'upserted': 0,
        'history_records_before': store.count_history(),
    }
    stats.update(extra_summary or {})
    print(
        f'RAG scan progress: {len(base_records)} records scanned from '
        f'{len(scan_jobs)} files, {len(pending_records)} pending.',
        flush=True,
    )
    if not pending_records:
        stats['history_records_after'] = store.count_history()
        return stats

    try:
        embedded_records = embed_history_records(
            records_to_embed,
            progress_offset=len(records_with_reused_embedding),
            progress_total=len(pending_records),
        )
        stats['embedded'] = len(embedded_records)
        stats['upserted'] = store.upsert_history(records_with_reused_embedding + embedded_records)
        stats['history_records_after'] = store.count_history()
    except Exception as exc:
        print(f'Warning: Failed to update RAG store: {exc}')
        stats['error'] = str(exc)
        stats['history_records_after'] = store.count_history()
    return stats


def prepare_rag_store(file_jobs):
    if not RAG_ENABLED:
        return {'enabled': False}
    store = get_rag_store()
    summary = {
        'enabled': True,
        'store_dir': store.store_dir if store else '',
        'history_records_before': store.count_history() if store else 0,
        'bootstrap_on_build': RAG_BOOTSTRAP_ON_BUILD,
    }
    if RAG_BOOTSTRAP_ON_BUILD:
        summary.update(sync_rag_store_for_jobs(file_jobs, quality_state='seed', scan_all_files=True))
    return summary


def print_rag_bootstrap_summary(summary):
    if not summary.get('enabled'):
        print('RAG is disabled. Enable batch.rag.enabled=true before bootstrapping.')
        return

    print('RAG bootstrap summary:')
    for key in (
        'store_dir',
        'scan_scope',
        'files_scanned',
        'scanned',
        'external_seed_files',
        'external_seed_records',
        'external_seed_invalid_json',
        'external_seed_filtered',
        'external_seed_skipped',
        'pending',
        'embedding_pending',
        'reused_embeddings',
        'embedded',
        'upserted',
        'history_records_before',
        'history_records_after',
    ):
        if key in summary:
            print(f'- {key}: {summary[key]}')
    if summary.get('error'):
        print(f"- error: {summary['error']}")


def embed_source_segments(records):
    embedded_records = []
    batch_size = 16
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        vectors = embed_texts([record['source_text'] for record in batch], RAG_DOCUMENT_TASK_TYPE)
        for record, vector in zip(batch, vectors):
            enriched = dict(record)
            enriched['embedding'] = vector
            enriched['embedding_metadata'] = {
                'embedding_model': RAG_EMBEDDING_MODEL,
                'embedding_task_type': RAG_DOCUMENT_TASK_TYPE,
                'embedding_dim': len(vector),
                'embedding_text_checksum': hash_text(record.get('source_text', '')),
            }
            enriched['updated_at'] = datetime.now().isoformat(timespec='seconds')
            embedded_records.append(enriched)
    return embedded_records


def source_segment_has_current_embedding(existing, record):
    if not existing or existing.get('source_checksum') != record.get('source_checksum'):
        return False
    embedding = existing.get('embedding')
    if not isinstance(embedding, list) or not embedding:
        return False
    if len(embedding) != RAG_OUTPUT_DIMENSIONALITY:
        return False
    metadata = existing.get('embedding_metadata') or {}
    return (
        metadata.get('embedding_model') == RAG_EMBEDDING_MODEL
        and metadata.get('embedding_task_type') == RAG_DOCUMENT_TASK_TYPE
        and metadata.get('embedding_dim') == RAG_OUTPUT_DIMENSIONALITY
        and metadata.get('embedding_text_checksum') == hash_text(record.get('source_text', ''))
    )


def print_source_index_bootstrap_summary(summary):
    print('Source Index bootstrap final summary:')
    for key in (
        'store_dir',
        'files_scanned',
        'scanned',
        'history_records_before',
        'reused_embeddings',
        'embedding_pending',
        'embedded',
        'upserted',
        'stale_count',
        'prune_enabled',
        'pruned',
        'history_records_after',
    ):
        if key in summary:
            print(f'- {key}: {summary[key]}')
    if summary.get('error'):
        print(f"- error: {summary['error']}")


def bootstrap_source_index(skip_prepare=False, prune=True):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    store = get_source_index_store()
    store.load()

    scan_jobs = all_rag_file_jobs()
    scanned_segments = collect_source_segments_for_jobs(scan_jobs)
    store.set_metadata(last_scanned_total=len(scanned_segments))

    stored_before = store.count_segments()
    scanned_ids = {seg['source_id'] for seg in scanned_segments}

    records_to_embed = []
    records_with_reused_embedding = []

    for record in scanned_segments:
        existing = store.get_segment(record['source_id'])
        if not existing:
            for seg in store.segments.values():
                if source_segment_has_current_embedding(seg, record):
                    existing = seg
                    break

        if source_segment_has_current_embedding(existing, record):
            enriched = dict(record)
            enriched['embedding'] = existing['embedding']
            enriched['embedding_metadata'] = existing['embedding_metadata']
            if 'created_at' in existing:
                enriched['created_at'] = existing['created_at']
            records_with_reused_embedding.append(enriched)
        else:
            records_to_embed.append(record)

    stale_segments = []
    for source_id, seg in store.segments.items():
        if source_id not in scanned_ids:
            stale_segments.append(seg)

    stale_count = len(stale_segments)
    stale_details = [
        {
            'source_id': seg['source_id'],
            'file_rel_path': seg['file_rel_path'],
            'line_start': seg['line_start'],
            'line_end': seg['line_end'],
        }
        for seg in stale_segments
    ]

    print("=" * 60)
    print("Source Index Sync Stats (Pre-run):")
    print(f"- Store directory: {store.store_dir}")
    print(f"- Files scanned: {len(scan_jobs)}")
    print(f"- Total segments scanned from files: {len(scanned_segments)}")
    print(f"- Total segments stored previously: {stored_before}")
    print(f"- Unchanged segments (reusing embeddings): {len(records_with_reused_embedding)}")
    print(f"- New/updated segments (need embeddings): {len(records_to_embed)}")
    print(f"- Stale segments in database: {stale_count}")
    if stale_count > 0:
        print("  Stale segments details:")
        for item in stale_details:
            print(f"    * {item['file_rel_path']}:{item['line_start']}-{item['line_end']} (ID: {item['source_id']})")
    print("=" * 60)
    sys.stdout.flush()

    summary = {
        'enabled': True,
        'store_dir': store.store_dir,
        'files_scanned': len(scan_jobs),
        'scanned': len(scanned_segments),
        'history_records_before': stored_before,
        'reused_embeddings': len(records_with_reused_embedding),
        'embedding_pending': len(records_to_embed),
        'stale_count': stale_count,
        'prune_enabled': prune,
        'embedded': 0,
        'upserted': 0,
        'pruned': 0,
    }

    if not records_to_embed and (not stale_segments or not prune):
        summary['history_records_after'] = store.count_segments()
        print("No new embeddings required, and no stale segments to prune.")
        return summary

    try:
        if records_with_reused_embedding:
            reused_upserted = store.upsert_segments(records_with_reused_embedding)
            summary['upserted'] += reused_upserted
            print(f"Reused embeddings written: {reused_upserted}.")
            sys.stdout.flush()

        if records_to_embed:
            print(f"Generating embeddings for {len(records_to_embed)} segments...")
            sys.stdout.flush()
            batch_size = 16
            for start in range(0, len(records_to_embed), batch_size):
                batch = records_to_embed[start:start + batch_size]
                embedded_records = embed_source_segments(batch)
                summary['embedded'] += len(embedded_records)
                summary['upserted'] += store.upsert_segments(embedded_records)
                processed = min(start + len(batch), len(records_to_embed))
                print(
                    "Source index embedding progress: "
                    f"{processed}/{len(records_to_embed)} scanned, "
                    f"{summary['embedded']} embedded, "
                    f"{store.count_segments()} stored."
                )
                sys.stdout.flush()

        if stale_segments and prune:
            print(f"Pruning {stale_count} stale segments...")
            sys.stdout.flush()
            prune_count = store.delete_segments([seg['source_id'] for seg in stale_segments])
            summary['pruned'] = prune_count

        summary['history_records_after'] = store.count_segments()
        print(f"Sync complete. Stored segments count is now: {summary['history_records_after']}.")
    except Exception as exc:
        print(f'Warning: Failed to update Source Index store: {exc}')
        summary['error'] = str(exc)
        summary['history_records_after'] = store.count_segments()

    return summary


def bootstrap_rag_store(skip_prepare=False, seed_jsonl_paths=None):
    if not RAG_ENABLED:
        summary = {'enabled': False}
        print_rag_bootstrap_summary(summary)
        return summary

    seed_jsonl_paths = [path for path in (seed_jsonl_paths or []) if path]
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR) and not seed_jsonl_paths:
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    external_records, external_summary = load_external_rag_seed_records(seed_jsonl_paths)
    summary = sync_rag_store_for_jobs(
        [],
        quality_state='seed',
        scan_all_files=True,
        extra_records=external_records,
        extra_summary=external_summary,
    )
    print_rag_bootstrap_summary(summary)
    return summary



def entry_context_text(entry):
    translated = entry.get('translation', '')
    if legacy.contains_chinese(translated):
        return translated
    return entry.get('source', '')


def load_repair_report_items(report_path):
    if not report_path or not os.path.isfile(report_path):
        raise SystemExit(f'Repair report not found: {report_path}')

    items = []
    seen = set()
    with open(report_path, 'r', encoding='utf-8-sig') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'Invalid repair report JSONL row: {exc}') from exc

            batch_style = False
            file_rel_path = ''
            file_path = row.get('file')
            if isinstance(file_path, str) and file_path.strip():
                file_path = resolve_path_under_dir(legacy.TL_DIR, file_path, 'repair file')
            else:
                file_rel_path = row.get('file_rel_path')
                if isinstance(file_rel_path, str) and file_rel_path.strip():
                    file_path = resolve_path_under_dir(legacy.TL_DIR, file_rel_path, 'repair file_rel_path')
                    batch_style = True
                else:
                    file_path = ''

            line_number = row.get('line')
            try:
                line_number = int(line_number)
            except (TypeError, ValueError):
                continue

            source_text = row.get('source')
            if source_text is None:
                source_text = row.get('text')
                if source_text is not None:
                    batch_style = True
            if source_text is None:
                continue

            if batch_style:
                line_number += 1

            if not file_path:
                continue

            normalized_row = dict(row)
            normalized_row['file'] = file_path
            normalized_row['file_rel_path'] = file_rel_path_for_repair(file_path, file_rel_path)
            normalized_row['line'] = line_number
            normalized_row['source'] = source_text

            dedupe_key = (
                file_path,
                line_number,
                str(source_text),
                str(row.get('id') or ''),
                str(row.get('start')),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            items.append(normalized_row)

    items.sort(key=lambda item: (item.get('game', ''), item['file'], item['line']))
    return items


def file_rel_path_for_repair(file_path, preferred=''):
    preferred = str(preferred or '').strip()
    if preferred:
        return legacy._normalize_rel_path(preferred)
    try:
        return legacy._normalize_rel_path(os.path.relpath(file_path, legacy.TL_DIR))
    except Exception:
        return legacy._normalize_rel_path(os.path.basename(file_path))


def build_repair_jobs(report_items, batch_size=2, context_before=2, context_after=2):
    jobs = []
    unresolved = []
    items_by_file = {}
    for item in report_items:
        items_by_file.setdefault(item['file'], []).append(item)

    for file_path in sorted(items_by_file):
        file_items = sorted(items_by_file[file_path], key=lambda item: item['line'])
        file_rel_path = file_rel_path_for_repair(
            file_path,
            file_items[0].get('file_rel_path') if file_items else '',
        )
        with open(file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
        entries = collect_repair_entries_from_lines(lines)
        line_map = {}
        for entry in entries:
            line_map.setdefault(entry['line_number'], []).append(entry)

        targets = []
        for item in file_items:
            entry = find_repair_entry_for_item(item, line_map.get(item['line'], []))
            if not entry:
                unresolved.append(
                    {
                        'file': file_path,
                        'line': item.get('line'),
                        'source': item.get('source', ''),
                        'error': 'Could not locate target line in current tl file',
                    }
                )
                continue
            target = dict(item)
            target['id'] = f"{file_path}:{entry['line_number']}:{entry['start']}:{entry['end']}"
            target['text'] = item['source']
            target['start'] = entry['start']
            target['end'] = entry['end']
            target['prefix'] = entry.get('prefix', '')
            target['quote'] = entry['quote']
            target['entry_index'] = entry['entry_index']
            target['file_rel_path'] = file_rel_path
            target['speaker_id'] = target.get('speaker_id') or entry.get('speaker_id', '')
            target['speaker'] = target.get('speaker') or entry.get('speaker', '')
            targets.append(target)

        if not targets:
            continue

        current_group = []
        previous_index = None
        for target in targets:
            current_index = target['entry_index']
            if (
                current_group
                and (
                    len(current_group) >= batch_size
                    or previous_index is None
                    or current_index != previous_index + 1
                )
            ):
                jobs.append(_build_repair_job(file_rel_path, file_path, entries, current_group, context_before, context_after))
                current_group = []
            current_group.append(target)
            previous_index = current_index
        if current_group:
            jobs.append(_build_repair_job(file_rel_path, file_path, entries, current_group, context_before, context_after))

    return jobs, unresolved


def _build_repair_job(file_rel_path, file_path, entries, target_group, context_before, context_after):
    first_index = target_group[0]['entry_index']
    last_index = target_group[-1]['entry_index']
    context_past = [
        entry_context_text(entry)
        for entry in entries[max(0, first_index - context_before):first_index]
    ]
    context_future = [
        entry_context_text(entry)
        for entry in entries[last_index + 1:last_index + 1 + context_after]
    ]
    job = {
        'key': hashlib.sha1(f"repair:{file_path}:{target_group[0]['line']}:{target_group[-1]['line']}".encode('utf-8')).hexdigest()[:12],
        'file_rel_path': file_rel_path,
        'file_path': file_path,
        'context_past': context_past,
        'context_future': context_future,
        'items': [
            {
                'id': target['id'],
                'text': target['text'],
                'line': target['line'],
                'line_number': target['line'],
                'start': target['start'],
                'end': target['end'],
                'prefix': target.get('prefix', ''),
                'quote': target['quote'],
                'file_rel_path': target.get('file_rel_path', file_rel_path),
                'speaker_id': target.get('speaker_id', ''),
                'speaker': target.get('speaker', ''),
            }
            for target in target_group
        ],
    }
    story_hits = retrieve_batch_story_hits(
        file_rel_path,
        job['items'],
        context_past,
        context_future,
    ) if STORY_MEMORY_ENABLED else None
    if STORY_MEMORY_ENABLED and story_memory.has_story_hits(story_hits):
        job['story_hits'] = story_hits
    return job


def build_repair_request(job):
    instruction = (
        build_system_instruction()
        + '\nSome targets may be short interjections, short UI text, or short reactions. Translate them naturally in context.'
    )
    request = {
        'system_instruction': {'parts': [{'text': instruction}]},
        'contents': [
            {
                'role': 'user',
                'parts': [
                    {
                        'text': build_user_prompt(
                            job['context_past'],
                            job['items'],
                            job['context_future'],
                            story_hits=job.get('story_hits') if 'story_hits' in job else None,
                            source_hits=job.get('source_hits') or [],
                        )
                    }
                ],
            }
        ],
        'generation_config': build_generation_config(job['items']),
    }
    if BATCH_SAFETY_SETTINGS:
        request['safety_settings'] = BATCH_SAFETY_SETTINGS
    return {
        'key': job['key'],
        'request': request,
    }

def run_sync_request(request_payload, model_name, api_key_index=None):
    attempts = 1 if api_key_index is not None else max(1, len(getattr(legacy, 'API_KEYS', [])))
    last_error = None

    for attempt in range(1, attempts + 1):
        client = create_batch_client(api_key_index=api_key_index)
        try:
            config = dict(request_payload.get('generation_config') or {})
            system_instruction = request_payload.get('system_instruction')
            if system_instruction:
                config['system_instruction'] = system_instruction
            safety_settings = request_payload.get('safety_settings')
            if safety_settings:
                config['safety_settings'] = safety_settings
            response = client.models.generate_content(
                model=model_name,
                contents=request_payload.get('contents') or [],
                config=config,
            )
            response_payload = serialize_unknown(response)
            return {
                'response_payload': response_payload,
                'response_text': extract_text_from_response_payload(response_payload),
                'finish_reason': extract_finish_reason(response_payload),
                'usage_metadata': summarize_usage_metadata(extract_usage_metadata(response_payload)),
            }
        except Exception as exc:
            last_error = exc
            retryable = is_quota_error(exc) or is_unavailable_error(exc)
            if api_key_index is None and retryable and attempt < attempts and legacy.rotate_api_key():
                label = 'quota' if is_quota_error(exc) else 'service unavailable'
                print(f'Sync request hit {label}. Retrying with next API key ({attempt}/{attempts})...')
                time.sleep(min(attempt, 2))
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError('Sync request failed without a captured exception.')


def create_sync_package_dir(package_name):
    ensure_batch_dirs()
    base_dir = os.path.join(SYNC_RUNS_DIR, package_name)
    candidates = [base_dir]
    candidates.extend(f'{base_dir}_{index:02d}' for index in range(1, 1000))
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f'Could not create unique sync run directory for {package_name}.')


def select_chunk_window(chunks, limit=0, offset=0):
    if offset < 0:
        offset = 0
    if limit and limit > 0:
        return chunks[offset:offset + limit]
    return chunks[offset:]


def write_request_rows(path, request_rows):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in request_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_manifest_file(package_dir, manifest, update_latest=True):
    manifest_path = os.path.join(package_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if update_latest:
        remember_latest_manifest(manifest_path)
    return manifest_path


def execute_sync_request_rows(manifest_path, request_rows, api_key_index=None):
    manifest = load_manifest(manifest_path)
    result_path = resolve_manifest_result_path(manifest)
    summary = {
        'request_count': len(request_rows),
        'successful_request_count': 0,
        'failed_request_count': 0,
        'max_tokens_count': 0,
        'missing_text_count': 0,
        'reason_counts': {},
    }
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as handle:
        for index, row in enumerate(request_rows, start=1):
            key = row.get('key', f'sync-{index}')
            print(f'[{index}/{len(request_rows)}] {key}')
            result_row = {'key': key}
            try:
                result = run_sync_request(
                    row.get('request') or {},
                    manifest.get('batch_model') or BATCH_MODEL,
                    api_key_index=api_key_index,
                )
                result_row['response'] = result.get('response_payload') or {}
                result_row['finish_reason'] = result.get('finish_reason', '')
                result_row['usage_metadata'] = result.get('usage_metadata') or {}
                summary['successful_request_count'] += 1
                if result.get('finish_reason') == 'MAX_TOKENS':
                    summary['max_tokens_count'] += 1
                    bump_counter(summary['reason_counts'], 'max_tokens')
                if not result.get('response_text'):
                    summary['missing_text_count'] += 1
                    bump_counter(summary['reason_counts'], 'missing_response_text')
                print(f"  finish_reason: {result.get('finish_reason') or '(none)'}")
            except Exception as exc:
                summary['failed_request_count'] += 1
                bump_counter(summary['reason_counts'], 'request_error')
                result_row['error'] = str(exc)
                print(f'  error: {str(exc)[:160]}')
            handle.write(json.dumps(result_row, ensure_ascii=False) + '\n')

    manifest['sync_completed_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['job_state'] = 'SYNC_COMPLETED' if summary['failed_request_count'] == 0 else 'SYNC_PARTIAL'
    manifest['sync_summary'] = summary
    manifest['result_jsonl_path'] = result_path
    save_manifest(manifest, update_latest=False)
    return manifest


def make_sync_manifest(
    *,
    package_dir,
    mode,
    display_name,
    chunks,
    request_rows,
    settings,
    extra_fields=None,
):
    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    result_jsonl_path = os.path.join(package_dir, 'results.jsonl')
    write_request_rows(input_jsonl_path, request_rows)
    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': mode,
        'execution': 'sync',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': result_jsonl_path,
        'job_name': '',
        'job_state': 'SYNC_LOCAL',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': settings,
        'summary': {
            'file_count': len(summarize_files_for_chunks(chunks)),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk.get('items') or []) for chunk in chunks),
        },
        'files': summarize_files_for_chunks(chunks),
        'chunks': chunks,
        'build_warnings': get_batch_risk_warnings(),
    }
    if extra_fields:
        manifest.update(extra_fields)
    return write_manifest_file(package_dir, manifest, update_latest=manifest.get('execution') != 'sync')


def sync_keyword_candidates(
    display_name_override='',
    skip_prepare=True,
    chunk_size=None,
    max_candidates_per_chunk=None,
    limit=0,
    offset=0,
    output_jsonl='',
    output_markdown='',
    output_summary_jsonl='',
    output_summary_markdown='',
    api_key_index=None,
):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_keyword_file_jobs()
    if not file_jobs:
        print('No keyword source lines found.')
        return None

    chunk_size = max(1, int(chunk_size or KEYWORD_CHUNK_SIZE))
    max_candidates = max(1, int(max_candidates_per_chunk or KEYWORD_MAX_CANDIDATES_PER_CHUNK))
    chunks = select_chunk_window(build_keyword_chunks(file_jobs, chunk_size=chunk_size), limit=limit, offset=offset)
    if not chunks:
        raise SystemExit('No keyword chunks available for the requested range.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_dir = create_sync_package_dir(f'{timestamp}_{guess_project_slug()}_sync_keywords')
    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'sync-{KEYWORD_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'
    request_rows = [build_keyword_request(chunk, max_candidates) for chunk in chunks]
    manifest_path = make_sync_manifest(
        package_dir=package_dir,
        mode=MANIFEST_MODE_KEYWORD_EXTRACTION,
        display_name=display_name,
        chunks=chunks,
        request_rows=request_rows,
        settings={
            'keyword_chunk_size': chunk_size,
            'max_candidates_per_chunk': max_candidates,
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        extra_fields={
            'keyword_settings': {
                'chunk_size': chunk_size,
                'max_candidates_per_chunk': max_candidates,
            },
        },
    )
    manifest = execute_sync_request_rows(manifest_path, request_rows, api_key_index=api_key_index)
    print(f"Sync keyword run: {manifest['_package_dir']}")
    return export_keyword_candidates(
        target=manifest['_manifest_path'],
        output_jsonl=output_jsonl,
        output_markdown=output_markdown,
        output_summary_jsonl=output_summary_jsonl,
        output_summary_markdown=output_summary_markdown,
    )


def sync_revisions(
    display_name_override='',
    skip_prepare=False,
    chunk_size=None,
    limit=0,
    offset=0,
    output_jsonl='',
    output_markdown='',
    apply=False,
    force=False,
    api_key_index=None,
):
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_revision_file_jobs()
    if not file_jobs:
        print('No revision source lines found.')
        return None

    chunk_size = max(1, int(chunk_size or REVISION_CHUNK_SIZE))
    rag_prepare_summary = prepare_rag_store(file_jobs)
    chunks = select_chunk_window(build_revision_chunks(file_jobs, chunk_size=chunk_size), limit=limit, offset=offset)
    if not chunks:
        raise SystemExit('No revision chunks available for the requested range.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_dir = create_sync_package_dir(f'{timestamp}_{guess_project_slug()}_sync_revisions')
    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'sync-{REVISION_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'
    request_rows = [build_revision_request(chunk) for chunk in chunks]
    extra_fields = {
        'revision_settings': {
            'chunk_size': chunk_size,
        },
    }
    if RAG_ENABLED:
        extra_fields['rag_enabled'] = True
        extra_fields['rag_store_path'] = RAG_STORE_DIR or get_default_rag_store_dir()
        extra_fields['rag_settings'] = {
            'top_k_history': RAG_TOP_K_HISTORY,
            'top_k_terms': RAG_TOP_K_TERMS,
            'min_similarity': RAG_MIN_SIMILARITY,
            'segment_lines': RAG_SEGMENT_LINES,
        }
        extra_fields['rag_summary'] = summarize_batch_rag(chunks, rag_prepare_summary)
    if STORY_MEMORY_ENABLED:
        extra_fields['story_memory_enabled'] = True
        extra_fields['story_memory_graph_file'] = STORY_MEMORY_GRAPH_FILE
        extra_fields['story_memory_settings'] = {
            'top_k_terms': STORY_MEMORY_TOP_K_TERMS,
            'top_k_characters': STORY_MEMORY_TOP_K_CHARACTERS,
            'top_k_relations': STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_scenes': STORY_MEMORY_TOP_K_SCENES,
            'max_context_chars': STORY_MEMORY_MAX_CONTEXT_CHARS,
        }
        extra_fields['story_memory_summary'] = summarize_batch_story_memory(chunks)
    manifest_path = make_sync_manifest(
        package_dir=package_dir,
        mode=MANIFEST_MODE_REVISION,
        display_name=display_name,
        chunks=chunks,
        request_rows=request_rows,
        settings={
            'revision_chunk_size': chunk_size,
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        extra_fields=extra_fields,
    )
    manifest = execute_sync_request_rows(manifest_path, request_rows, api_key_index=api_key_index)
    print(f"Sync revision run: {manifest['_package_dir']}")
    preview_manifest = preview_revisions(
        target=manifest['_manifest_path'],
        output_jsonl=output_jsonl,
        output_markdown=output_markdown,
    )
    if apply:
        return apply_revisions(preview_manifest['_manifest_path'], force=force)
    return preview_manifest


def print_repair_summary(summary):
    print(f"Requested items: {summary['requested_items']}")
    print(f"Repair jobs: {summary['job_count']}")
    print(f"Applied items: {summary['applied_items']}")
    print(f"Applied files: {summary['applied_files']}")
    print(f"Failure items: {summary['failure_items']}")
    print(f"Request errors: {summary['request_errors']}")
    print(f"Parse errors: {summary['parse_errors']}")
    print(f"Validation failures: {summary['validation_failures']}")
    print(f"Missing item ids: {summary['missing_item_ids']}")
    print(f"Unresolved items: {summary['unresolved_items']}")
    if summary.get('story_memory_enabled'):
        story_summary = summary.get('story_memory_summary') or {}
        print(
            'Story Memory repair hits: '
            f"{story_summary.get('chunks_with_story_hits', 0)}/{summary['job_count']} jobs"
        )
    if summary.get('reason_counts'):
        print('Failure categories:')
        for name in sorted(summary['reason_counts']):
            print(f"- {name}: {summary['reason_counts'][name]}")


def repair_remaining_items(report_path, limit=0, offset=0, batch_size=2, context_before=2, context_after=2, api_key_index=None):
    report_items = load_repair_report_items(report_path)
    if offset < 0:
        offset = 0
    if limit and limit > 0:
        report_items = report_items[offset:offset + limit]
    else:
        report_items = report_items[offset:]
    if not report_items:
        raise SystemExit('No repair items available for the requested range.')

    jobs, unresolved = build_repair_jobs(
        report_items,
        batch_size=batch_size,
        context_before=context_before,
        context_after=context_after,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_stem = os.path.splitext(os.path.basename(report_path))[0]
    run_dir = os.path.join(REPAIR_RUNS_DIR, f'{timestamp}_{report_stem}')
    os.makedirs(run_dir, exist_ok=True)

    request_log_path = os.path.join(run_dir, 'repair_requests.jsonl')
    result_log_path = os.path.join(run_dir, 'repair_results.jsonl')
    failure_log_path = os.path.join(run_dir, 'repair_failures.jsonl')
    summary_path = os.path.join(run_dir, 'repair_summary.json')

    replacements_by_file = {}
    result_entries = []
    failure_entries = []
    reason_counts = {}
    summary = {
        'report_path': report_path,
        'run_dir': run_dir,
        'requested_items': len(report_items),
        'job_count': len(jobs),
        'applied_items': 0,
        'applied_files': 0,
        'failure_items': 0,
        'request_errors': 0,
        'parse_errors': 0,
        'validation_failures': 0,
        'missing_item_ids': 0,
        'unresolved_items': len(unresolved),
        'story_memory_enabled': STORY_MEMORY_ENABLED,
        'story_memory_graph_file': STORY_MEMORY_GRAPH_FILE if STORY_MEMORY_ENABLED else '',
        'story_memory_summary': summarize_batch_story_memory(jobs) if STORY_MEMORY_ENABLED else {},
        'reason_counts': reason_counts,
    }

    for row in unresolved:
        bump_counter(reason_counts, 'unresolved_line')
        failure_entries.append(
            {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'error': row['error'],
                'file': row['file'],
                'line': row.get('line'),
                'source': row.get('source', ''),
            }
        )

    request_rows = [build_repair_request(job) for job in jobs]
    write_jsonl_file(request_log_path, request_rows)

    for index, (job, request_row) in enumerate(zip(jobs, request_rows), start=1):
        finish_reason = ''
        usage_metadata = {}
        response_text = ''
        parse_error = ''
        result_items = []
        parse_ok = False
        try:
            response_data = run_sync_request(
                request_row['request'],
                model_name=BATCH_MODEL,
                api_key_index=api_key_index,
            )
            finish_reason = response_data['finish_reason']
            usage_metadata = response_data['usage_metadata']
            response_text = response_data['response_text']
        except Exception as exc:
            summary['request_errors'] += 1
            bump_counter(reason_counts, 'request_error')
            parse_error = str(exc)
            for item in job['items']:
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'file': job['file_path'],
                        'line': item['line'],
                        'source': item['text'],
                        'id': item['id'],
                        'error': parse_error,
                    }
                )
            result_entries.append(
                {
                    'index': index,
                    'key': job['key'],
                    'file': job['file_path'],
                    'expected_items': len(job['items']),
                    'parsed_items': 0,
                    'parse_ok': False,
                    'parse_error': parse_error,
                    'finish_reason': finish_reason,
                    'usage_metadata': usage_metadata,
                    'response_preview': '',
                }
            )
            continue

        if response_text:
            try:
                payload = parse_json_payload(response_text)
                result_items = normalize_result_items(payload)
                parse_ok = True
            except Exception as exc:
                parse_error = str(exc)
                summary['parse_errors'] += 1
                bump_counter(reason_counts, 'parse_error')
        else:
            parse_error = 'Missing text in response payload'
            summary['parse_errors'] += 1
            bump_counter(reason_counts, 'missing_response_text')

        if not parse_ok:
            for item in job['items']:
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'file': job['file_path'],
                        'line': item['line'],
                        'source': item['text'],
                        'id': item['id'],
                        'error': parse_error,
                        'finish_reason': finish_reason,
                        'usage_metadata': usage_metadata,
                        'response_preview': response_text[:500],
                    }
                )
            result_entries.append(
                {
                    'index': index,
                    'key': job['key'],
                    'file': job['file_path'],
                    'expected_items': len(job['items']),
                    'parsed_items': 0,
                    'parse_ok': False,
                    'parse_error': parse_error,
                    'finish_reason': finish_reason,
                    'usage_metadata': usage_metadata,
                    'response_preview': response_text[:500],
                }
            )
            continue

        item_map = {item['id']: item for item in job['items']}
        seen_ids = set()
        for result_item in result_items:
            target_item = item_map.get(result_item['id'])
            if not target_item:
                bump_counter(reason_counts, 'schema_or_item_mismatch')
                continue
            seen_ids.add(result_item['id'])
            valid, reason = legacy.validate_translation(target_item['text'], result_item['translation'])
            if not valid and reason == 'No Chinese characters' and allow_non_chinese_repair_translation(
                target_item['text'], result_item['translation']
            ):
                valid = True
            if not valid:
                summary['validation_failures'] += 1
                bump_counter(reason_counts, 'validation_failed')
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'file': job['file_path'],
                        'line': target_item['line'],
                        'source': target_item['text'],
                        'translation': result_item['translation'],
                        'id': target_item['id'],
                        'error': f'Validation failed: {reason}',
                        'finish_reason': finish_reason,
                        'usage_metadata': usage_metadata,
                    }
                )
                continue

            replacements_by_file.setdefault(job['file_path'], {}).setdefault(target_item['line'] - 1, []).append(
                (
                    target_item['start'],
                    target_item['end'],
                    result_item['translation'],
                    target_item.get('prefix', ''),
                    target_item['quote'],
                )
            )
            summary['applied_items'] += 1

        missing_ids = set(item_map.keys()) - seen_ids
        if missing_ids:
            summary['missing_item_ids'] += len(missing_ids)
            bump_counter(reason_counts, 'response_missing_item_id', len(missing_ids))
        for missing_id in sorted(missing_ids):
            item = item_map[missing_id]
            failure_entries.append(
                {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'file': job['file_path'],
                    'line': item['line'],
                    'source': item['text'],
                    'id': item['id'],
                    'error': 'Response missing item id',
                    'finish_reason': finish_reason,
                    'usage_metadata': usage_metadata,
                }
            )

        result_entries.append(
            {
                'index': index,
                'key': job['key'],
                'file': job['file_path'],
                'expected_items': len(job['items']),
                'parsed_items': len(result_items),
                'parse_ok': True,
                'parse_error': '',
                'finish_reason': finish_reason,
                'usage_metadata': usage_metadata,
                'response_preview': response_text[:500],
            }
        )

    for file_path, replacements in replacements_by_file.items():
        safe_file_path = resolve_path_under_dir(legacy.TL_DIR, file_path, 'repair writeback file')
        with open(safe_file_path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
        legacy.commit_replacements(safe_file_path, lines, replacements)
        summary['applied_files'] += 1

    summary['failure_items'] = len(failure_entries)
    write_jsonl_file(result_log_path, result_entries)
    write_jsonl_file(failure_log_path, failure_entries)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f'Repair report: {report_path}')
    print(f'Repair run dir: {run_dir}')
    print_repair_summary(summary)
    print(f'Repair results: {result_log_path}')
    print(f'Repair failures: {failure_log_path}')
    return summary


DOCTOR_TRANSLATE_BLOCK_RE = re.compile(r'^translate\s+\S+\s+(?!strings\b)(?!python\b)\S+\s*:')
DOCTOR_STRING_BLOCK_RE = re.compile(r'^translate\s+\S+\s+strings\s*:')


def _stringify_command(command):
    if command is None:
        return ''
    if isinstance(command, list):
        return ' '.join(str(part) for part in command)
    return str(command)


def collect_tl_doctor_counts():
    counts = {
        'rpy_files': 0,
        'translate_blocks': 0,
        'string_sections': 0,
        'old_lines': 0,
        'new_lines': 0,
        'commented_original_lines': 0,
    }
    if not os.path.isdir(legacy.TL_DIR):
        return counts

    for root, _, files in os.walk(legacy.TL_DIR):
        for file_name in files:
            if not file_name.endswith('.rpy'):
                continue
            counts['rpy_files'] += 1
            path = os.path.join(root, file_name)
            try:
                with open(path, 'r', encoding='utf-8-sig') as handle:
                    lines = handle.readlines()
            except Exception:
                continue

            for line in lines:
                if DOCTOR_TRANSLATE_BLOCK_RE.match(line):
                    counts['translate_blocks'] += 1
                if DOCTOR_STRING_BLOCK_RE.match(line):
                    counts['string_sections'] += 1
                if legacy.TL_OLD_LINE_RE.match(line):
                    counts['old_lines'] += 1
                if legacy.TL_NEW_LINE_RE.match(line):
                    counts['new_lines'] += 1
                if legacy.TL_COMMENT_SOURCE_RE.match(line):
                    counts['commented_original_lines'] += 1

    return counts


def collect_doctor_layout_context(report):
    base_dir = os.path.abspath(report.get('base_dir', ''))
    work_dir = report.get('work_dir', '') or legacy.resolve_work_dir(base_dir)
    work_exists = os.path.isdir(work_dir)
    work_empty = (not work_exists) or legacy.is_work_dir_empty(work_dir)
    work_game_dir = os.path.join(work_dir, legacy.WORK_GAME_SUBDIR) if work_dir else ''
    return {
        'is_work_root': os.path.basename(base_dir).lower() == 'work',
        'work_dir': work_dir,
        'work_exists': work_exists,
        'work_empty': work_empty,
        'work_game_exists': os.path.isdir(work_game_dir),
        'has_tl': int(report.get('counts', {}).get('rpy_files', 0)) > 0,
        'has_original': bool(report.get('original_game_dir')),
    }


def assess_doctor_layout_status(report, context=None):
    ctx = context or collect_doctor_layout_context(report)
    is_work_root = ctx['is_work_root']
    has_tl = ctx['has_tl']
    has_original = ctx['has_original']
    work_exists = ctx['work_exists']

    if not is_work_root:
        if work_exists or has_original or has_tl:
            return 'switch_to_work'
        return 'failed'

    if has_tl:
        return 'ready'
    if has_original or report.get('can_generate_template') or ctx.get('work_game_exists'):
        return 'attention'
    return 'failed'


def _doctor_pending_task_count(report):
    try:
        return int(report.get('pending_task_count') or 0)
    except (TypeError, ValueError):
        return 0


def _doctor_pending_baseline(report):
    counts = report.get('counts') or {}
    baseline = int(counts.get('commented_original_lines') or 0)
    if baseline <= 0:
        baseline = int(counts.get('translate_blocks') or 0)
    return baseline


def _doctor_pending_is_minor(report):
    """True when remaining pending lines are negligible for a mostly-finished project."""
    pending = _doctor_pending_task_count(report)
    if pending <= 0:
        return True
    baseline = _doctor_pending_baseline(report)
    if baseline <= 0:
        return pending < 50
    return pending < 50 or (pending / baseline) < 0.01


def _doctor_should_recommend_enabling_rag(report):
    pending = _doctor_pending_task_count(report)
    if pending <= 0 or not _doctor_has_existing_translations(report):
        return False
    if _doctor_pending_is_minor(report):
        return False
    baseline = _doctor_pending_baseline(report)
    if pending >= 150:
        return True
    if baseline > 0 and (pending / baseline) >= 0.01:
        return True
    return pending >= 50


def _doctor_source_index_needs_bootstrap(source_index):
    if not source_index.get('enabled'):
        return ''
    segments = int(source_index.get('source_segments') or 0)
    expected = int(source_index.get('expected_segments') or 0)
    if not source_index.get('store_exists') or segments <= 0:
        return 'missing'
    if expected > 0 and segments < expected:
        return 'incomplete'
    return ''


def _doctor_rag_needs_bootstrap(rag):
    if not rag.get('enabled'):
        return False
    if not rag.get('store_exists'):
        return True
    return int(rag.get('history_records') or 0) <= 0


def _doctor_has_existing_translations(report):
    counts = report.get('counts') or {}
    if int(counts.get('old_lines') or 0) > 0:
        return True

    translate_blocks = int(counts.get('translate_blocks') or 0)
    pending = _doctor_pending_task_count(report)
    if translate_blocks > 0 and pending > 0 and pending < translate_blocks:
        return True
    return False


def collect_doctor_workflow_state(report):
    """Return a normal workflow state separately from actionable recommendations."""
    if report.get('layout_status') != 'ready':
        return ''
    has_tl = int((report.get('counts') or {}).get('rpy_files') or 0) > 0
    if not has_tl:
        return ''

    pending = _doctor_pending_task_count(report)
    if pending <= 0:
        return doctor_rec.NO_PENDING_LINES

    has_existing_translations = _doctor_has_existing_translations(report)
    if has_existing_translations and _doctor_pending_is_minor(report):
        return doctor_rec.SUBSTANTIALLY_COMPLETE
    if has_existing_translations:
        return doctor_rec.START_INCREMENTAL_BATCH
    return doctor_rec.START_PENDING_BATCH


def collect_doctor_recommendations(report):
    recommendations = []

    layout_status = report.get('layout_status', '')
    if layout_status == 'switch_to_work':
        work_dir = report.get('work_dir', '')
        if work_dir:
            recommendations.append(
                doctor_rec.make_doctor_recommendation(
                    doctor_rec.SWITCH_TO_WORK,
                    work_dir=work_dir,
                )
            )
        work_missing_or_empty = not report.get('work_exists') or report.get('work_empty')
        if work_missing_or_empty and report.get('work_bootstrap_allowed') and report.get('original_game_dir'):
            recommendations.append(
                doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_WORK)
            )
        return recommendations

    mode = report.get('mode', '')
    has_tl = report.get('counts', {}).get('rpy_files', 0) > 0
    pending = _doctor_pending_task_count(report)
    has_existing_translations = _doctor_has_existing_translations(report)

    if report.get('work_bootstrap_allowed') and report.get('original_game_dir'):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_WORK)
        )
        return recommendations
    elif not has_tl and report.get('prepare_enabled') and report.get('can_generate_template'):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.GENERATE_TEMPLATE)
        )
        return recommendations
    elif not has_tl and report.get('prepare_enabled') and mode == 'blocked_missing_template':
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.INSTALL_SDK_GENERATE_TEMPLATE)
        )
        return recommendations
    elif not has_tl and not report.get('prepare_enabled'):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.ENABLE_PREPARE)
        )
        return recommendations

    context_status = report.get('context_status') or {}
    source_index = context_status.get('source_index') or {}
    rag = context_status.get('rag') or {}

    source_index_status = _doctor_source_index_needs_bootstrap(source_index)
    if source_index_status == 'missing':
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_SOURCE_INDEX)
        )
        return recommendations
    if source_index_status == 'incomplete':
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE)
        )
        return recommendations

    if _doctor_rag_needs_bootstrap(rag):
        if rag.get('bootstrap_on_build'):
            recommendations.append(
                doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD)
            )
        else:
            recommendations.append(
                doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG)
            )
        return recommendations

    if (
        not rag.get('enabled')
        and has_existing_translations
        and pending > 0
        and _doctor_should_recommend_enabling_rag(report)
    ):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.ENABLE_RAG_FOR_CONSISTENCY)
        )
        return recommendations

    if (
        not source_index.get('enabled')
        and pending > 0
        and has_tl
        and not has_existing_translations
    ):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT)
        )
        return recommendations

    return recommendations


def _store_dir_has_context_files(store_dir, file_names):
    if not store_dir:
        return False
    if not os.path.isdir(store_dir):
        return False
    for file_name in file_names:
        if os.path.isfile(os.path.join(store_dir, file_name)):
            return True
    return False


def _load_store_count(store, count_method_name):
    try:
        count_method = getattr(store, count_method_name)
        return count_method(), '', getattr(store, 'metadata', {}) or {}
    except Exception as exc:
        return 0, str(exc), {}


def _resolve_source_index_expected_segments(store, metadata):
    expected_raw = metadata.get('last_scanned_total')
    try:
        expected = int(expected_raw)
    except (TypeError, ValueError):
        expected = 0
    if expected > 0:
        return expected, ''

    if not os.path.isdir(legacy.TL_DIR):
        return 0, ''

    try:
        scanned = len(collect_source_segments_for_jobs(all_rag_file_jobs()))
    except Exception as exc:
        return 0, str(exc)

    if scanned > 0:
        try:
            store.set_metadata(last_scanned_total=scanned)
        except Exception as exc:
            return scanned, str(exc)
    return scanned, ''


def collect_doctor_context_status():
    rag_store_dir = RAG_STORE_DIR or get_default_rag_store_dir()
    source_index_store_dir = SOURCE_INDEX_STORE_DIR or get_default_source_index_store_dir()

    rag_status = {
        'enabled': RAG_ENABLED,
        'store_dir': rag_store_dir if RAG_ENABLED else '',
        'store_exists': False,
        'history_records': 0,
        'bootstrap_on_build': RAG_BOOTSTRAP_ON_BUILD,
        'updated_at': '',
        'error': '',
    }
    if RAG_ENABLED:
        rag_exists = _store_dir_has_context_files(rag_store_dir, ('history.jsonl', 'metadata.json'))
        rag_status['store_exists'] = rag_exists
        if rag_exists:
            store = JsonRagStore(rag_store_dir)
            count, error, metadata = _load_store_count(store, 'count_history')
            rag_status['history_records'] = count
            rag_status['updated_at'] = metadata.get('updated_at', '')
            rag_status['error'] = error

    source_index_status = {
        'enabled': SOURCE_INDEX_ENABLED,
        'store_dir': source_index_store_dir if SOURCE_INDEX_ENABLED else '',
        'store_exists': False,
        'source_segments': 0,
        'schema_version': '',
        'updated_at': '',
        'error': '',
    }
    if SOURCE_INDEX_ENABLED:
        source_exists = _store_dir_has_context_files(
            source_index_store_dir,
            ('source_segments.jsonl', 'source_metadata.json'),
        )
        source_index_status['store_exists'] = source_exists
        if source_exists:
            store = JsonSourceIndexStore(source_index_store_dir)
            count, error, metadata = _load_store_count(store, 'count_segments')
            source_index_status['source_segments'] = count
            schema_version = metadata.get('schema_version', '')
            source_index_status['schema_version'] = schema_version if schema_version is not None else ''
            expected_segments, expected_error = _resolve_source_index_expected_segments(store, metadata)
            if expected_segments > 0:
                source_index_status['expected_segments'] = expected_segments
            source_index_status['updated_at'] = metadata.get('updated_at', '')
            combined_error = ' | '.join(
                part for part in (error, expected_error) if part
            )
            source_index_status['error'] = combined_error

    return {
        'rag': rag_status,
        'source_index': source_index_status,
    }

def _read_translator_config_object():
    if not os.path.exists(legacy.TRANSLATOR_CONFIG):
        return {}
    try:
        with open(legacy.TRANSLATOR_CONFIG, 'r', encoding='utf-8-sig') as handle:
            data = json.load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_translation_match_key(text):
    return re.sub(r'\s+', ' ', str(text or '').strip()).casefold()


def _normalize_translation_value(text):
    return re.sub(r'\s+', ' ', str(text or '').strip())


def _load_glossary_normalize_map(glossary_path):
    if not glossary_path or not os.path.isfile(glossary_path):
        return {}
    try:
        with open(glossary_path, 'r', encoding='utf-8-sig') as handle:
            data = json.load(handle) or {}
    except Exception:
        return {}
    normalize_map = data.get('normalize_map')
    if not isinstance(normalize_map, dict):
        return {}
    loaded = {}
    for source, target in normalize_map.items():
        key = _normalize_translation_match_key(source)
        value = _normalize_translation_value(target)
        if key and value:
            loaded[key] = value
    return loaded


def _collect_story_graph_translation_entries(graph):
    entries = []
    if not isinstance(graph, dict):
        return entries

    for term in graph.get('terms') or []:
        if not isinstance(term, dict):
            continue
        target = _normalize_translation_value(term.get('target') or term.get('translation'))
        if not target:
            continue
        source = _normalize_translation_value(term.get('source') or term.get('term'))
        if source:
            entries.append(('story_graph.terms', source, target))
        for alias in term.get('aliases') or []:
            alias_text = _normalize_translation_value(alias)
            if alias_text:
                entries.append(('story_graph.terms', alias_text, target))

    characters = graph.get('characters') or {}
    if isinstance(characters, dict):
        char_items = characters.items()
    elif isinstance(characters, list):
        char_items = []
        for item in characters:
            if isinstance(item, dict):
                char_id = item.get('id') or item.get('key') or item.get('name')
                char_items.append((char_id, item))
    else:
        char_items = []

    for char_id, raw_data in char_items:
        if not isinstance(raw_data, dict):
            continue
        target = _normalize_translation_value(
            raw_data.get('zh_name') or raw_data.get('target') or ''
        )
        if not target:
            continue
        label = _normalize_translation_value(char_id) or _normalize_translation_value(raw_data.get('name'))
        source_keys = [
            _normalize_translation_value(raw_data.get('name')),
            _normalize_translation_value(char_id),
        ]
        source_keys.extend(
            _normalize_translation_value(alias)
            for alias in (raw_data.get('aliases') or [])
        )
        for source in source_keys:
            if source:
                entries.append((f'story_graph.characters.{label or char_id}', source, target))
    return entries


def _resolve_doctor_story_graph_path():
    config = _read_translator_config_object()
    batch = config.get('batch') if isinstance(config.get('batch'), dict) else {}
    story_cfg = batch.get('story_memory') if isinstance(batch.get('story_memory'), dict) else {}
    graph_file = story_cfg.get('graph_file') or ''
    if isinstance(graph_file, str) and graph_file.strip():
        return legacy.resolve_story_memory_graph_path(graph_file.strip())
    return legacy.get_default_story_memory_graph_path()


def collect_glossary_story_graph_conflicts(glossary_path='', story_graph_path=''):
    glossary_map = _load_glossary_normalize_map(glossary_path)
    if not glossary_map:
        return []
    if not story_graph_path or not os.path.isfile(story_graph_path):
        return []

    graph = story_memory.load_story_graph(story_graph_path)
    conflicts = []
    seen = set()
    for source_label, source_text, story_target in _collect_story_graph_translation_entries(graph):
        glossary_target = glossary_map.get(_normalize_translation_match_key(source_text))
        if not glossary_target:
            continue
        if _normalize_translation_value(glossary_target) == _normalize_translation_value(story_target):
            continue
        dedupe_key = (
            _normalize_translation_match_key(source_text),
            glossary_target,
            story_target,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        conflicts.append(
            f'Translation conflict for "{source_text}": glossary.json -> "{glossary_target}", '
            f'{source_label} -> "{story_target}".'
        )
    return conflicts


def collect_doctor_project_assets_status(base_dir):
    from project_asset_paths import expected_project_asset_paths, paths_match_project

    config = _read_translator_config_object()
    expected = expected_project_asset_paths(base_dir)

    glossary_configured = config.get('glossary_file') or config.get('glossary_path') or ''
    batch = config.get('batch') if isinstance(config.get('batch'), dict) else {}
    macro_configured = batch.get('macro_setting_file') or ''

    glossary_resolved = (
        legacy._resolve_preferred_path(legacy.TOOL_DIR, base_dir, glossary_configured)
        if glossary_configured
        else expected['glossary_file']
    )
    macro_resolved = (
        legacy._resolve_preferred_path(base_dir, base_dir, macro_configured)
        if macro_configured
        else expected['macro_setting_file']
    )

    return {
        'glossary_file': glossary_resolved or expected['glossary_file'],
        'glossary_exists': os.path.isfile(glossary_resolved or expected['glossary_file']),
        'glossary_matches_project': paths_match_project(
            glossary_resolved or glossary_configured,
            expected['glossary_file'],
        ),
        'macro_setting_file': macro_resolved or expected['macro_setting_file'],
        'macro_exists': os.path.isfile(macro_resolved or expected['macro_setting_file']),
        'macro_matches_project': paths_match_project(
            macro_resolved or macro_configured,
            expected['macro_setting_file'],
        ),
        'expected_glossary_file': expected['glossary_file'],
        'expected_macro_setting_file': expected['macro_setting_file'],
    }


def collect_doctor_project_assets_warnings(project_assets):
    warnings = []
    if not project_assets:
        return warnings

    glossary_file = project_assets.get('glossary_file') or ''
    macro_file = project_assets.get('macro_setting_file') or ''

    if not project_assets.get('glossary_matches_project'):
        expected = project_assets.get('expected_glossary_file') or ''
        warnings.append(
            f'glossary_file does not match current project; expected {expected}, '
            f'configured {glossary_file or "(not set)"}.'
        )
    elif not project_assets.get('glossary_exists'):
        expected = project_assets.get('expected_glossary_file') or glossary_file
        warnings.append(
            f'glossary.json not found for current project ({expected}); '
            'batch translation will fall back to default preserve terms.'
        )

    if not project_assets.get('macro_matches_project'):
        expected = project_assets.get('expected_macro_setting_file') or ''
        warnings.append(
            f'macro_setting_file does not match current project; expected {expected}, '
            f'configured {macro_file or "(not set)"}.'
        )
    elif not project_assets.get('macro_exists'):
        expected = project_assets.get('expected_macro_setting_file') or macro_file
        warnings.append(
            f'macro_setting.md not found for current project ({expected}); '
            'batch translation will run without project style guidance.'
        )

    return warnings


def collect_doctor_report():
    source_game_dir = legacy._guess_source_game_dir()
    template_info = legacy.get_prepare_template_command_info(source_game_dir)
    counts = collect_tl_doctor_counts()
    tl_exists = os.path.isdir(legacy.TL_DIR)
    has_tl_files = counts['rpy_files'] > 0
    can_generate_template = bool(template_info.get('available'))
    original_game_dir = legacy.resolve_original_game_dir()
    work_bootstrap_allowed, work_dir, _ = legacy.work_dir_bootstrap_allowed()

    warnings = []

    legacy_manifests = []
    if os.path.isdir(BATCH_JOBS_DIR):
        for name in os.listdir(BATCH_JOBS_DIR):
            sub_dir = os.path.join(BATCH_JOBS_DIR, name)
            if not os.path.isdir(sub_dir):
                continue
            manifest_path = os.path.join(sub_dir, 'manifest.json')
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        m_data = json.load(f)
                    version = m_data.get('manifest_version', m_data.get('version', 1))
                    if version < 2:
                        legacy_manifests.append(name)
                except Exception:
                    pass
    if legacy_manifests:
        warnings.append(
            f"Found {len(legacy_manifests)} legacy manifest(s) (v1) in batch jobs "
            f"(e.g., {legacy_manifests[0]}). They will run in compatibility fallback mode."
        )

    if RAG_ENABLED:
        try:
            store = get_rag_store()
            if store:
                store.load()
                if store.history:
                    has_legacy_keys = False
                    for key in store.history.keys():
                        if len(key.split(':')) != 4:
                            has_legacy_keys = True
                            break
                    if has_legacy_keys:
                        warnings.append(
                            "RAG store contains legacy ID format keys. They will be seamlessly "
                            "migrated on the next successful writeback (checksum fallback enabled)."
                        )
        except Exception:
            pass

    if counts['old_lines'] != counts['new_lines']:
        warnings.append('old/new line counts differ; string translation blocks may be malformed.')
    if counts['translate_blocks'] and not counts['commented_original_lines']:
        warnings.append(
            'Dialogue translation blocks do not include source comments; revision/RAG source pairing may be limited.'
        )
    template_reason = template_info.get('reason', '')
    if not can_generate_template and template_info.get('kind') == 'custom':
        warnings.append(f'Custom template command cannot be rendered: {template_reason or "unknown error"}.')
    if not can_generate_template and not has_tl_files:
        if template_info.get('kind') == 'custom':
            warnings.append('No TL files and custom template command is unavailable; template generation is required.')
        else:
            warnings.append('No TL files and no Ren\'Py SDK/game launcher found; template generation is required.')
    elif not can_generate_template and has_tl_files:
        if template_info.get('kind') == 'custom':
            warnings.append('Custom template command is unavailable; existing TL files can still be processed.')
        else:
            warnings.append('Ren\'Py SDK/game launcher not found; existing TL files can still be processed.')

    if has_tl_files:
        mode = 'existing_tl_only'
    elif can_generate_template:
        mode = 'can_generate_template'
    else:
        mode = 'blocked_missing_template'

    pending_task_count = 0
    pending_file_count = 0
    if has_tl_files:
        try:
            file_jobs = collect_pending_file_jobs()
            pending_file_count = len(file_jobs)
            pending_task_count = sum(job['task_count'] for job in file_jobs)
        except Exception as exc:
            print(f'Warning: Could not compute pending translation counts: {exc}')

    context_status = collect_doctor_context_status()
    project_assets = collect_doctor_project_assets_status(legacy.BASE_DIR)
    warnings.extend(collect_doctor_project_assets_warnings(project_assets))
    glossary_path = project_assets.get('glossary_file') or ''
    if project_assets.get('glossary_exists'):
        warnings.extend(
            collect_glossary_story_graph_conflicts(
                glossary_path=glossary_path,
                story_graph_path=_resolve_doctor_story_graph_path(),
            )
        )

    report = {
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        'tl_subdir': legacy.TL_SUBDIR,
        'language': legacy.PREP_LANGUAGE,
        'source_game_dir': source_game_dir,
        'original_game_dir': original_game_dir,
        'work_dir': work_dir,
        'work_bootstrap_allowed': work_bootstrap_allowed,
        'prepare_enabled': legacy.PREP_ENABLED,
        'generate_template': legacy.PREP_GENERATE_TEMPLATE,
        'refresh_existing_template': legacy.PREP_REFRESH_EXISTING_TEMPLATE,
        'renpy_sdk_dir': legacy.PREP_RENPY_SDK_DIR,
        'tl_exists': tl_exists,
        'can_generate_template': can_generate_template,
        'template_command_kind': template_info.get('kind', ''),
        'template_command': _stringify_command(template_info.get('command')),
        'template_reason': template_info.get('reason', ''),
        'python_exe': template_info.get('python_exe', ''),
        'launcher_py': template_info.get('launcher_py', ''),
        'mode': mode,
        'counts': counts,
        'pending_task_count': pending_task_count,
        'pending_file_count': pending_file_count,
        'context_status': context_status,
        'project_assets': project_assets,
        'warnings': warnings,
    }
    layout_context = collect_doctor_layout_context(report)
    report.update(layout_context)
    report['layout_status'] = assess_doctor_layout_status(report, layout_context)
    finalize_doctor_actionable_signals(report)
    return report


def finalize_doctor_actionable_signals(report):
    """Attach recommendations + workflow_state with required-prep gate.

    Recommendations are computed first so required prep (e.g. bootstrap_rag)
    can suppress readiness-flavored workflow codes and avoid CLI dual-signal.
    """
    report['recommendations'] = collect_doctor_recommendations(report)
    workflow_state = collect_doctor_workflow_state(report)
    if workflow_state and doctor_rec.recommendations_block_workflow_state(
        report['recommendations']
    ):
        workflow_state = ''
    report['workflow_state'] = workflow_state
    return report


def print_doctor_report(report):
    counts = report['counts']
    context_status = report.get('context_status') or {}
    rag_context = context_status.get('rag') or {}
    source_index_context = context_status.get('source_index') or {}
    print('Doctor report:')
    print(f"- Base dir: {report['base_dir']}")
    print(f"- TL dir: {report['tl_dir']} (exists: {report['tl_exists']})")
    print(f"- TL subdir: {report.get('tl_subdir') or ''}")
    print(f"- Language: {report['language']}")
    print(
        f"- Prepare: enabled={report['prepare_enabled']}, "
        f"generate_template={report['generate_template']}, "
        f"refresh_existing_template={report['refresh_existing_template']}"
    )
    print(f"- Ren'Py SDK dir: {report['renpy_sdk_dir'] or '(not configured)'}")
    print(f"- Launcher: {report['launcher_py'] or '(not found)'}")
    print(f"- Python: {report['python_exe'] or '(not resolved)'}")
    if report['can_generate_template']:
        print(f"- Template generation: available ({report['template_command_kind']})")
        print(f"- Template command: {report['template_command']}")
    else:
        print(f"- Template generation: unavailable ({report['template_reason'] or 'no command resolved'})")
    print(f"- Mode: {report['mode']}")
    print(f"- Is work root: {report.get('is_work_root', False)}")
    print(
        f"- Work dir: {report.get('work_dir', '')} "
        f"(exists: {report.get('work_exists', False)}, empty: {report.get('work_empty', True)})"
    )
    print(
        f"- Original game dir: {report.get('original_game_dir') or '(not found)'}"
    )
    print(f"- Layout status: {report.get('layout_status', '')}")
    workflow_state = str(report.get('workflow_state') or '').strip()
    if workflow_state:
        print(f"- Workflow state: {workflow_state}")
    print(
        '- TL scan: '
        f"rpy_files={counts['rpy_files']}, "
        f"translate_blocks={counts['translate_blocks']}, "
        f"string_sections={counts['string_sections']}, "
        f"old_lines={counts['old_lines']}, "
        f"new_lines={counts['new_lines']}, "
        f"commented_original_lines={counts['commented_original_lines']}"
    )
    if report['tl_exists'] and counts['rpy_files'] > 0:
        print(
            '- Pending translation: '
            f"task_count={report['pending_task_count']}, "
            f"file_count={report['pending_file_count']}"
        )
        if report['pending_task_count'] > 0:
            print(
                '  Note: counts English strings without Han characters; may include '
                'preserved names, patron lists, or punctuation-only updates. '
                'This does not indicate missed batch writeback.'
            )
    print(
        '- RAG context: '
        f"enabled={rag_context.get('enabled', False)}, "
        f"store_dir={rag_context.get('store_dir') or ''}, "
        f"store_exists={rag_context.get('store_exists', False)}, "
        f"history_records={rag_context.get('history_records', 0)}, "
        f"bootstrap_on_build={rag_context.get('bootstrap_on_build', False)}, "
        f"updated_at={rag_context.get('updated_at') or ''}, "
        f"error={rag_context.get('error') or ''}"
    )
    print(
        '- Source index context: '
        f"enabled={source_index_context.get('enabled', False)}, "
        f"store_dir={source_index_context.get('store_dir') or ''}, "
        f"store_exists={source_index_context.get('store_exists', False)}, "
        f"source_segments={source_index_context.get('source_segments', 0)}, "
        f"expected_segments={source_index_context.get('expected_segments', 0)}, "
        f"schema_version={source_index_context.get('schema_version') or ''}, "
        f"updated_at={source_index_context.get('updated_at') or ''}, "
        f"error={source_index_context.get('error') or ''}"
    )
    project_assets = report.get('project_assets') or {}
    print(
        '- Project assets: '
        f"glossary_exists={project_assets.get('glossary_exists', False)}, "
        f"glossary_matches_project={project_assets.get('glossary_matches_project', False)}, "
        f"glossary_file={project_assets.get('glossary_file') or ''}, "
        f"macro_exists={project_assets.get('macro_exists', False)}, "
        f"macro_matches_project={project_assets.get('macro_matches_project', False)}, "
        f"macro_setting_file={project_assets.get('macro_setting_file') or ''}"
    )
    if report['warnings']:
        print('Warnings:')
        for warning in report['warnings']:
            print(f'- {warning}')
    if report.get('recommendations'):
        print('Recommendations:')
        for recommendation in report['recommendations']:
            rec = doctor_rec.normalize_doctor_recommendation(recommendation)
            print(f'- {doctor_rec.format_doctor_recommendation_cli_line(rec)}')


def print_work_bootstrap_summary(result):
    print('Work bootstrap summary:')
    print(f"- status: {result.get('status', '')}")
    print(f"- project_root: {result.get('project_root', '')}")
    print(f"- work_dir: {result.get('work_dir', '')}")
    print(f"- source_game_dir: {result.get('source_game_dir', '')}")
    print(f"- files_copied: {result.get('files_copied', 0)}")
    print(f"- game_root_updated: {result.get('game_root_updated', False)}")
    print(f"- message: {result.get('message', '')}")


def run_bootstrap_work(*, save_game_root=True, refresh_runtime_paths=True):
    result = legacy.bootstrap_work_from_original(
        save_game_root=save_game_root,
        refresh_runtime_paths=refresh_runtime_paths,
    )
    print_work_bootstrap_summary(result)
    if result.get('status') == 'failed':
        raise SystemExit(f"[Bootstrap] {result.get('message', 'work bootstrap failed')}")
    return result


def print_template_generation_summary(result):
    print('Template generation summary:')
    print(f"- status: {result.get('status', '')}")
    print(f"- tl_subdir: {result.get('tl_subdir', '')}")
    print(f"- tl_dir: {result.get('tl_dir', '')}")
    print(f"- tl_exists: {result.get('tl_exists', False)}")
    print(f"- rpy_files: {result.get('rpy_files', 0)}")
    print(f"- language: {result.get('language', '')}")
    print(f"- message: {result.get('message', '')}")


def _build_template_generation_result(status, message, counts=None):
    if counts is None:
        counts = collect_tl_doctor_counts()
    return {
        'status': status,
        'tl_subdir': legacy.TL_SUBDIR,
        'tl_dir': legacy.TL_DIR,
        'tl_exists': os.path.isdir(legacy.TL_DIR),
        'rpy_files': counts['rpy_files'],
        'language': legacy.PREP_LANGUAGE,
        'message': message,
    }


def _raise_generate_template_failure(result):
    print_template_generation_summary(result)
    raise SystemExit(f"[GenerateTemplate] {result['message']}")


def run_generate_template():
    if not legacy.PREP_ENABLED:
        _raise_generate_template_failure(
            _build_template_generation_result(
                'failed',
                'prepare is disabled in translator_config.json',
            )
        )

    if not legacy.PREP_GENERATE_TEMPLATE:
        _raise_generate_template_failure(
            _build_template_generation_result(
                'failed',
                'prepare.generate_template is disabled in translator_config.json',
            )
        )

    try:
        legacy.run_prepare_steps()
    except SystemExit as exc:
        message = str(exc.args[0]) if exc.args else 'Template generation failed during prepare.'
        _raise_generate_template_failure(
            _build_template_generation_result('failed', message)
        )

    counts = collect_tl_doctor_counts()
    rpy_files = counts['rpy_files']
    if rpy_files > 0:
        status = 'ready'
        message = f'Translation template ready with {rpy_files} TL file(s).'
    else:
        status = 'failed'
        message = 'Template generation finished but no TL files were found.'

    result = _build_template_generation_result(status, message, counts=counts)
    print_template_generation_summary(result)
    if status != 'ready':
        raise SystemExit(f"[GenerateTemplate] {message}")
    return result


def _manifest_target_language_fields(source_manifest=None):
    fields = {
        'tl_subdir': legacy.TL_SUBDIR,
        'target_language': legacy.PREP_LANGUAGE,
    }
    if not isinstance(source_manifest, dict):
        return fields
    tl_subdir = source_manifest.get('tl_subdir')
    if isinstance(tl_subdir, str) and tl_subdir.strip():
        fields['tl_subdir'] = tl_subdir.strip()
    target_language = source_manifest.get('target_language')
    if isinstance(target_language, str) and target_language.strip():
        fields['target_language'] = target_language.strip()
    return fields


def print_banner():
    print('=' * 60)
    print('Gemini Batch Translator (Ren\'Py)')
    print(f'Base dir: {legacy.BASE_DIR}')
    print(f'TL subdir: {legacy.TL_SUBDIR}')
    print(f'TL dir: {legacy.TL_DIR} (exists: {os.path.isdir(legacy.TL_DIR)})')
    print(f'Target language: {legacy.PREP_LANGUAGE}')
    print(f'Batch jobs dir: {BATCH_JOBS_DIR}')
    print(f'Translator config: {legacy.TRANSLATOR_CONFIG} (exists: {os.path.isfile(legacy.TRANSLATOR_CONFIG)})')
    print(f'Glossary: {legacy.GLOSSARY_FILE} (exists: {os.path.isfile(legacy.GLOSSARY_FILE)})')
    print(f'Batch model: {BATCH_MODEL}')
    print(
        f'Chunk settings: target={BATCH_TARGET_SIZE}, '
        f'target_chars={BATCH_TARGET_CHARS}, '
        f'context_before={BATCH_CONTEXT_BEFORE}, context_after={BATCH_CONTEXT_AFTER}'
    )
    print(f'Max output tokens: {BATCH_MAX_OUTPUT_TOKENS}')
    print(f'Thinking level: {format_thinking_level_for_display()}')
    print(
        f'Prepare: enabled={legacy.PREP_ENABLED}, '
        f'generate_template={legacy.PREP_GENERATE_TEMPLATE}, '
        f'refresh_existing_template={legacy.PREP_REFRESH_EXISTING_TEMPLATE}'
    )
    print('=' * 60)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Batch translator for Ren\'Py tl files using Gemini Batch API.'
    )
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('doctor', help='Inspect prepare, SDK, and TL template compatibility without writing files.')

    bootstrap_work_parser = subparsers.add_parser(
        'bootstrap-work',
        help='Create work/ from original/game when work is missing or empty (no TL generation).',
    )
    bootstrap_work_parser.add_argument(
        '--no-update-game-root',
        action='store_true',
        help='Do not update translator_config.json game_root to work/ after bootstrap.',
    )

    subparsers.add_parser(
        'generate-template',
        help='Run prepare steps only to generate or refresh tl/<language> templates.',
    )

    build_parser = subparsers.add_parser('build', help='Build local batch package and JSONL only.')
    build_parser.add_argument('--display-name', default='', help='Override Batch display name.')
    build_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before collecting tasks.',
    )

    keyword_build_parser = subparsers.add_parser(
        'build-keywords',
        help='Build a keyword extraction batch package without changing translation files.',
    )
    keyword_build_parser.add_argument('--display-name', default='', help='Override Batch display name.')
    keyword_build_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Compatibility no-op; keyword builds skip prepare by default.',
    )
    keyword_build_parser.add_argument(
        '--prepare',
        action='store_true',
        help='Run auto prepare steps before collecting keyword sources.',
    )
    keyword_build_parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Source line count per keyword extraction chunk. Defaults to batch.keyword_extraction.chunk_size.',
    )
    keyword_build_parser.add_argument(
        '--max-candidates-per-chunk',
        type=int,
        default=0,
        help='Maximum keyword candidates requested from each chunk.',
    )

    revision_build_parser = subparsers.add_parser(
        'build-revisions',
        help='Build a revision batch package for existing old/new TL translations.',
    )
    revision_build_parser.add_argument('--display-name', default='', help='Override Batch display name.')
    revision_build_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before collecting revision sources.',
    )
    revision_build_parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Old/new pair count per revision chunk. Defaults to batch.revision.chunk_size.',
    )

    bootstrap_rag_parser = subparsers.add_parser(
        'bootstrap-rag',
        help='Prebuild or refresh the Batch RAG history store from all allowed TL files.',
    )
    bootstrap_rag_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before scanning TL files.',
    )
    bootstrap_rag_parser.add_argument(
        '--seed-jsonl',
        action='append',
        default=None,
        help='Import external parallel corpus JSONL rows as additional RAG seed records. Can be repeated.',
    )

    bootstrap_source_index_parser = subparsers.add_parser(
        'bootstrap-source-index',
        help='Prebuild or refresh the Batch source-only index store from all allowed TL files.',
    )
    bootstrap_source_index_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before scanning TL files.',
    )
    bootstrap_source_index_parser.add_argument(
        '--no-prune',
        action='store_true',
        help='Do not prune stale segments from the index store after indexing.',
    )

    submit_parser = subparsers.add_parser('submit', help='Create and submit a batch job.')
    submit_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Existing manifest path or package dir. If omitted, build a new package first.',
    )
    submit_parser.add_argument('--display-name', default='', help='Override Batch display name.')
    submit_parser.add_argument('--model', default='', help='Override batch model.')
    submit_parser.add_argument(
        '--max-cost',
        type=float,
        default=None,
        help='Reject submit when estimated max cost exceeds this value (same currency as batch.pricing).',
    )
    submit_parser.add_argument(
        '--force',
        action='store_true',
        help='Start a fresh submit attempt after an incomplete upload-only state.',
    )
    submit_parser.add_argument(
        '--resume',
        action='store_true',
        help='Continue job creation using a previously uploaded input file.',
    )

    recover_submit_parser = subparsers.add_parser(
        'recover-submit',
        help='Recover a batch job from submit journal when manifest was not updated.',
    )
    recover_submit_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    recover_submit_parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip remote job verification before writing manifest.',
    )

    estimate_cost_parser = subparsers.add_parser(
        'estimate-cost',
        help='Estimate token usage and cost for an existing batch package.',
    )
    estimate_cost_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )

    status_parser = subparsers.add_parser('status', help='Refresh and show batch job status.')
    status_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )

    check_parser = subparsers.add_parser('check', help='Dry-run parse downloaded results and summarize recoverable items.')
    check_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )

    probe_parser = subparsers.add_parser('probe', help='Run a small synchronous smoke test with normal generate_content calls.')
    probe_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    probe_parser.add_argument('--limit', type=int, default=3, help='How many request rows to probe.')
    probe_parser.add_argument('--offset', type=int, default=0, help='Start offset within requests.jsonl.')
    probe_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')

    download_parser = subparsers.add_parser('download', help='Download batch results for a succeeded job.')
    download_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    download_parser.add_argument('--force', action='store_true', help='Overwrite local results.jsonl.')

    apply_parser = subparsers.add_parser('apply', help='Apply downloaded results back into tl files.')
    apply_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    apply_parser.add_argument(
        '--force',
        action='store_true',
        help='Bypass the applied_at guard; source validation still applies.',
    )

    keyword_export_parser = subparsers.add_parser(
        'export-keywords',
        help='Export keyword extraction batch results to JSONL and Markdown review files.',
    )
    keyword_export_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Keyword manifest path or package dir. Defaults to latest package.',
    )
    keyword_export_parser.add_argument('--jsonl', default='', help='Relative output JSONL path inside the package.')
    keyword_export_parser.add_argument('--markdown', default='', help='Relative output Markdown path inside the package.')
    keyword_export_parser.add_argument(
        '--summary-jsonl',
        default='',
        help='Relative chunk summary JSONL path inside the package.',
    )
    keyword_export_parser.add_argument(
        '--summary-markdown',
        default='',
        help='Relative chunk summary Markdown path inside the package.',
    )

    merge_keywords_parser = subparsers.add_parser(
        'merge-keywords-to-glossary',
        help='Review keyword_candidates.jsonl entries and append accepted ones to glossary.json.',
    )
    merge_keywords_parser.add_argument(
        'target',
        help='keyword_candidates.jsonl path, keyword package dir, or manifest.json.',
    )
    merge_keywords_parser.add_argument(
        '--glossary',
        default='',
        help='Glossary JSON path. Defaults to translator_config glossary_file.',
    )
    merge_keywords_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview accepted/skipped entries without writing glossary.json.',
    )
    merge_keywords_parser.add_argument(
        '--preview',
        action='store_true',
        help='Alias for --dry-run.',
    )
    merge_keywords_parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Skip candidates below this confidence threshold.',
    )
    merge_keywords_parser.add_argument(
        '--accept-confidence',
        type=float,
        default=None,
        help='Auto-accept candidates at or above this confidence without prompting.',
    )
    merge_keywords_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing glossary entries with conflicting targets.',
    )
    merge_keywords_parser.add_argument(
        '--yes',
        action='store_true',
        help='Accept all non-skipped candidates without interactive prompts.',
    )
    merge_keywords_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating a timestamped glossary backup before writing.',
    )

    compare_variants_parser = subparsers.add_parser(
        'compare-variants',
        help='Run a synchronous translation A/B experiment from a batch manifest without writing game files.',
    )
    compare_variants_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Translation manifest path or package dir. Defaults to latest package.',
    )
    compare_variants_parser.add_argument(
        '--variants-file',
        required=True,
        help='JSON file describing experiment variants and config overrides.',
    )
    compare_variants_parser.add_argument('--limit', type=int, default=3, help='Number of manifest chunks to sample.')
    compare_variants_parser.add_argument('--offset', type=int, default=0, help='Chunk offset within the manifest.')
    compare_variants_parser.add_argument(
        '--output-dir',
        default='',
        help='Directory for ab_report.md and ab_results.jsonl. Defaults to logs/experiments/<timestamp>_ab/.',
    )
    compare_variants_parser.add_argument('--model', default='', help='Optional model override for all variants.')
    compare_variants_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')
    compare_variants_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build per-variant prompts and reports without calling the API.',
    )

    revision_preview_parser = subparsers.add_parser(
        'preview-revisions',
        help='Dry-run downloaded revision results and export JSONL/Markdown preview reports.',
    )
    revision_preview_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Revision manifest path or package dir. Defaults to latest package.',
    )
    revision_preview_parser.add_argument('--jsonl', default='', help='Relative output JSONL path inside the package.')
    revision_preview_parser.add_argument('--markdown', default='', help='Relative output Markdown path inside the package.')

    revision_apply_parser = subparsers.add_parser(
        'apply-revisions',
        help='Apply validated revision results back into existing TL new lines.',
    )
    revision_apply_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Revision manifest path or package dir. Defaults to latest package.',
    )
    revision_apply_parser.add_argument(
        '--force',
        action='store_true',
        help='Bypass the revision_applied_at guard; source validation still applies.',
    )

    sync_keyword_parser = subparsers.add_parser(
        'sync-keywords',
        help='Synchronously extract keyword candidates and export JSONL/Markdown reports.',
    )
    sync_keyword_parser.add_argument('--display-name', default='', help='Override sync run display name.')
    sync_keyword_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Compatibility no-op; sync keyword runs skip prepare by default.',
    )
    sync_keyword_parser.add_argument(
        '--prepare',
        action='store_true',
        help='Run auto prepare steps before collecting keyword sources.',
    )
    sync_keyword_parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Source line count per sync keyword request. Defaults to batch.keyword_extraction.chunk_size.',
    )
    sync_keyword_parser.add_argument(
        '--max-candidates-per-chunk',
        type=int,
        default=0,
        help='Maximum keyword candidates requested from each sync request.',
    )
    sync_keyword_parser.add_argument('--limit', type=int, default=0, help='Maximum request chunks to run. Set 0 for all.')
    sync_keyword_parser.add_argument('--offset', type=int, default=0, help='Start offset within built keyword chunks.')
    sync_keyword_parser.add_argument('--jsonl', default='', help='Relative output JSONL path inside the sync run dir.')
    sync_keyword_parser.add_argument('--markdown', default='', help='Relative output Markdown path inside the sync run dir.')
    sync_keyword_parser.add_argument(
        '--summary-jsonl',
        default='',
        help='Relative chunk summary JSONL path inside the sync run dir.',
    )
    sync_keyword_parser.add_argument(
        '--summary-markdown',
        default='',
        help='Relative chunk summary Markdown path inside the sync run dir.',
    )
    sync_keyword_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')

    sync_revision_parser = subparsers.add_parser(
        'sync-revisions',
        help='Synchronously revise existing old/new TL translations, preview by default, and optionally apply.',
    )
    sync_revision_parser.add_argument('--display-name', default='', help='Override sync run display name.')
    sync_revision_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before collecting revision sources.',
    )
    sync_revision_parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Old/new pair count per sync revision request. Defaults to batch.revision.chunk_size.',
    )
    sync_revision_parser.add_argument('--limit', type=int, default=0, help='Maximum request chunks to run. Set 0 for all.')
    sync_revision_parser.add_argument('--offset', type=int, default=0, help='Start offset within built revision chunks.')
    sync_revision_parser.add_argument('--jsonl', default='', help='Relative preview JSONL path inside the sync run dir.')
    sync_revision_parser.add_argument('--markdown', default='', help='Relative preview Markdown path inside the sync run dir.')
    sync_revision_parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply validated revisions after writing the preview report.',
    )
    sync_revision_parser.add_argument(
        '--force',
        action='store_true',
        help='When used with --apply, bypass the revision_applied_at guard; source validation still applies.',
    )
    sync_revision_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')

    split_parser = subparsers.add_parser('split', help='Split an existing batch package into smaller local packages.')
    split_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    split_parser.add_argument(
        '--max-chunks',
        type=int,
        default=600,
        help='Maximum chunk count per split package. Set 0 to disable this limit.',
    )
    split_parser.add_argument(
        '--max-items',
        type=int,
        default=0,
        help='Maximum item count per split package. Set 0 to disable this limit.',
    )
    split_parser.add_argument(
        '--display-name-prefix',
        default='',
        help='Override display-name prefix for generated child packages.',
    )

    retry_parser = subparsers.add_parser(
        'build-retry',
        help='Build a local retry package for unsafe translation chunks in a checked batch package.',
    )
    retry_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Parent translation manifest path or package dir. Defaults to latest package.',
    )
    retry_parser.add_argument('--display-name', default='', help='Override retry Batch display name.')

    merge_retry_parser = subparsers.add_parser(
        'merge-retry',
        help='Merge downloaded retry package results back into the parent translation package.',
    )
    merge_retry_parser.add_argument('parent', help='Parent translation manifest path or package dir.')
    merge_retry_parser.add_argument('retry', help='Retry translation manifest path or package dir.')

    repair_parser = subparsers.add_parser('repair', help='Synchronously repair specific remaining untranslated items from a JSONL report.')
    repair_parser.add_argument('report', help='JSONL report path, typically remaining_need_translate_*.jsonl')
    repair_parser.add_argument('--limit', type=int, default=0, help='Optional maximum number of report items to process.')
    repair_parser.add_argument('--offset', type=int, default=0, help='Optional starting offset within the report.')
    repair_parser.add_argument('--batch-size', type=int, default=2, help='How many adjacent items to repair per synchronous request.')
    repair_parser.add_argument('--context-before', type=int, default=2, help='How many prior nearby entries to include as context.')
    repair_parser.add_argument('--context-after', type=int, default=2, help='How many following nearby entries to include as context.')
    repair_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command is None:
        parser.print_help()
        return

    if command == 'doctor':
        legacy.load_translator_settings()
        legacy.load_glossary()
        load_batch_settings()
        print_banner()
        print_doctor_report(collect_doctor_report())
        return

    if command == 'bootstrap-work':
        legacy.load_translator_settings()
        legacy.load_glossary()
        load_batch_settings()
        print_banner()
        update_game_root = not args.no_update_game_root
        run_bootstrap_work(
            save_game_root=update_game_root,
            refresh_runtime_paths=update_game_root,
        )
        return

    if command == 'generate-template':
        legacy.load_translator_settings()
        legacy.load_glossary()
        load_batch_settings()
        print_banner()
        run_generate_template()
        return

    initialize_batch_logging()
    legacy.load_config()
    legacy.load_translator_settings()
    legacy.load_glossary()
    load_batch_settings()
    print_banner()

    if command == 'build':
        create_batch_package(
            display_name_override=args.display_name,
            skip_prepare=args.skip_prepare,
        )
        return

    if command == 'build-keywords':
        create_keyword_package(
            display_name_override=args.display_name,
            skip_prepare=(not args.prepare) or args.skip_prepare,
            chunk_size=args.chunk_size,
            max_candidates_per_chunk=args.max_candidates_per_chunk,
        )
        return

    if command == 'build-revisions':
        create_revision_package(
            display_name_override=args.display_name,
            skip_prepare=args.skip_prepare,
            chunk_size=args.chunk_size,
        )
        return

    if command == 'bootstrap-rag':
        bootstrap_rag_store(skip_prepare=args.skip_prepare, seed_jsonl_paths=args.seed_jsonl)
        return

    if command == 'bootstrap-source-index':
        summary = bootstrap_source_index(skip_prepare=args.skip_prepare, prune=(not args.no_prune))
        print_source_index_bootstrap_summary(summary)
        return

    if command == 'estimate-cost':
        manifest = load_manifest(args.target or None)
        estimate = ensure_manifest_cost_estimate(manifest)
        for line in batch_cost_estimate.format_cost_estimate_lines(estimate):
            print(line)
        return

    if command == 'submit':
        submit_manifest(
            target=args.target or None,
            display_name_override=args.display_name,
            model_override=args.model,
            max_cost=args.max_cost,
            force_resubmit=args.force,
            resume_upload=args.resume,
        )
        return

    if command == 'recover-submit':
        recover_submit_manifest(
            target=args.target or None,
            verify_remote=not args.no_verify,
        )
        return

    if command == 'status':
        show_status(args.target or None)
        return

    if command == 'check':
        check_results(args.target or None)
        return

    if command == 'probe':
        probe_requests(
            target=args.target or None,
            limit=args.limit,
            offset=args.offset,
            api_key_index=args.api_key_index,
        )
        return

    if command == 'download':
        download_results(args.target or None, force=args.force)
        return

    if command == 'apply':
        apply_results(args.target or None, force=args.force)
        return

    if command == 'export-keywords':
        export_keyword_candidates(
            target=args.target or None,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
            output_summary_jsonl=args.summary_jsonl,
            output_summary_markdown=args.summary_markdown,
        )
        return

    if command == 'merge-keywords-to-glossary':
        candidates_path = keyword_glossary_merge.resolve_keyword_candidates_path(args.target)
        glossary_path = args.glossary.strip() if args.glossary else legacy.GLOSSARY_FILE
        dry_run = args.dry_run or args.preview
        keyword_glossary_merge.merge_keywords_to_glossary(
            candidates_path,
            glossary_path,
            dry_run=dry_run,
            min_confidence=max(0.0, float(args.min_confidence or 0.0)),
            accept_confidence=args.accept_confidence,
            overwrite=args.overwrite,
            interactive=not args.yes and not dry_run,
            backup=not args.no_backup,
        )
        return

    if command == 'compare-variants':
        manifest = load_manifest(args.target or None)
        variants = translation_ab_experiment.load_variants_file(args.variants_file)
        summary = translation_ab_experiment.run_translation_ab_experiment(
            manifest,
            variants,
            limit=args.limit,
            offset=args.offset,
            output_dir=args.output_dir.strip(),
            model_override=args.model,
            api_key_index=args.api_key_index,
            dry_run=args.dry_run,
        )
        print('Translation A/B experiment:')
        print(f"- output_dir: {summary['output_dir']}")
        print(f"- chunks: {summary['chunk_count']}")
        print(f"- variants: {summary['variant_count']}")
        print(f"- dry_run: {summary['dry_run']}")
        print(f"- report: {summary['report_path']}")
        print(f"- results: {summary['results_path']}")
        return

    if command == 'preview-revisions':
        preview_revisions(
            target=args.target or None,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
        )
        return

    if command == 'apply-revisions':
        apply_revisions(args.target or None, force=args.force)
        return

    if command == 'sync-keywords':
        sync_keyword_candidates(
            display_name_override=args.display_name,
            skip_prepare=(not args.prepare) or args.skip_prepare,
            chunk_size=args.chunk_size,
            max_candidates_per_chunk=args.max_candidates_per_chunk,
            limit=args.limit,
            offset=args.offset,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
            output_summary_jsonl=args.summary_jsonl,
            output_summary_markdown=args.summary_markdown,
            api_key_index=args.api_key_index,
        )
        return

    if command == 'sync-revisions':
        sync_revisions(
            display_name_override=args.display_name,
            skip_prepare=args.skip_prepare,
            chunk_size=args.chunk_size,
            limit=args.limit,
            offset=args.offset,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
            apply=args.apply,
            force=args.force,
            api_key_index=args.api_key_index,
        )
        return

    if command == 'split':
        split_manifest(
            target=args.target or None,
            max_chunks=args.max_chunks,
            max_items=args.max_items,
            display_name_prefix=args.display_name_prefix,
        )
        return

    if command == 'build-retry':
        build_retry_package(
            target=args.target or None,
            display_name_override=args.display_name,
        )
        return

    if command == 'merge-retry':
        merge_retry_results(args.parent, args.retry)
        return

    if command == 'repair':
        repair_remaining_items(
            report_path=args.report,
            limit=args.limit,
            offset=args.offset,
            batch_size=args.batch_size,
            context_before=args.context_before,
            context_after=args.context_after,
            api_key_index=args.api_key_index,
        )
        return

    parser.print_help()


if __name__ == '__main__':
    main()
