# -*- coding: utf-8 -*-
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import tokenize
from dataclasses import asdict, replace
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from atomic_io import (
    atomic_write,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_many_lines,
    atomic_write_text,
    file_sha256,
    result_artifact_is_complete,
    recover_atomic_write_transaction,
    sha256_text,
)
from rag_memory import JsonRagStore, JsonSourceIndexStore, JsonSourceIndexStoreLockError, hash_text, truncate_text
import batch_cost_estimate
import batch_non_chinese_rules
import batch_submit_recovery
import cli_contract
import cli_discovery
import doctor_recommendations as doctor_rec
from engine_adapters.contracts import (
    Occurrence,
    OpaqueLocator,
    ProjectDiscoveryRequest,
    SourceDocument,
    ValidatedTranslation,
)
from engine_adapters.coverage import export_coverage_package, load_review_record
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
import engine_adapters.reuse as engine_reuse
import engine_adapters.versioning as engine_versioning
from engine_adapters.writeback import (
    WritebackPlanError,
    render_writeback_plan,
    source_snapshot_fingerprint,
)
import keyword_glossary_merge
import keyword_history
import model_profile
import model_usage_ledger
import prompt_context
import revision_corpus
import revision_proposals
import revision_selection
import translation_ab_experiment
import story_memory
import sync_translation_preview
import translation_core
import translation_plan
import translation_quality
import translator_runtime as runtime
from gemini_model_catalog import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_TRANSLATION_MODEL,
    filter_gemini_generation_config,
    is_gemini_3_model,
)
from project_version import __version__
from sync_model_backend import (
    DEFAULT_SYNC_RETRY_ATTEMPTS,
    DEFAULT_SYNC_TIMEOUT_SECONDS,
    SyncGenerationRequest,
    normalize_sync_timeout_seconds,
    sync_recovery_decision,
    sync_error_category,
    sync_error_summary,
)

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
PROJECT_SNAPSHOTS_DIR = os.path.join(LOG_DIR, 'project_snapshots')
PROJECT_RECONCILIATIONS_DIR = os.path.join(LOG_DIR, 'project_reconciliations')
PROJECT_TRANSLATION_RECORDS_DIR = os.path.join(LOG_DIR, 'translation_records')
PROJECT_REUSE_DIR = os.path.join(LOG_DIR, 'translation_reuse')
SYNC_BACKEND = 'gemini'
SYNC_MODEL = ''
SYNC_TIMEOUT_SECONDS = DEFAULT_SYNC_TIMEOUT_SECONDS
DURABLE_SYNC_COMMANDS = frozenset(
    {'sync-start', 'sync-resume', 'sync-status', 'sync-cancel', 'sync-derive'}
)
MACHINE_OUTPUT_COMMANDS = frozenset(
    {
        'doctor',
        'build',
        'submit',
        'status',
        'download',
        'check',
        'apply',
        'apply-revisions',
        'export-revision-corpus',
        'import-revision-proposals',
        'confirm-revision-proposals',
        'export-project-snapshot',
        'reconcile-project-snapshots',
        'build-translation-records',
        'build-reuse-candidates',
        'import-reuse-decisions',
        'export-reuse-results',
        'quality-ack',
        'quality-unack',
        'build-revisions',
        'preview-revisions',
        'sync-revisions',
        'build-keywords',
        'export-keywords',
        'sync-keywords',
        'merge-keywords-to-glossary',
        'final-review-build',
        'final-review-status',
        'final-review-export',
        'final-review-resume',
        'final-review-ingest-results',
        'final-review-create-revisions',
        *DURABLE_SYNC_COMMANDS,
    }
)
EXPLICIT_TARGET_COMMANDS = frozenset(
    {
        'submit',
        'status',
        'download',
        'check',
        'apply',
        'quality-ack',
        'quality-unack',
        'preview-revisions',
        'export-keywords',
        'final-review-status',
        'final-review-export',
        'final-review-resume',
        'final-review-ingest-results',
        'final-review-create-revisions',
    }
)
# Local-only batch commands must not be blocked by API-key preflight.
OFFLINE_BATCH_COMMANDS = frozenset(
    {
        'check',
        'apply',
        'estimate-cost',
        'preview-revisions',
        'apply-revisions',
        'export-revision-corpus',
        'import-revision-proposals',
        'confirm-revision-proposals',
        'export-project-snapshot',
        'reconcile-project-snapshots',
        'build-translation-records',
        'build-reuse-candidates',
        'import-reuse-decisions',
        'export-reuse-results',
        'quality-ack',
        'quality-unack',
        'split',
        'build-retry',
        'merge-retry',
        'export-keywords',
        'merge-keywords-to-glossary',
    }
)

REVISION_PREVIEW_CONTRACT_VERSION = 1
REVISION_APPLY_STATES = frozenset({'applied', 'no_op', 'blocked', 'partial'})


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
    os.makedirs(PROJECT_SNAPSHOTS_DIR, exist_ok=True)
    os.makedirs(PROJECT_RECONCILIATIONS_DIR, exist_ok=True)
    os.makedirs(PROJECT_TRANSLATION_RECORDS_DIR, exist_ok=True)
    os.makedirs(PROJECT_REUSE_DIR, exist_ok=True)


def initialize_batch_logging():
    if isinstance(sys.stdout, DualLogger):
        return
    ensure_batch_dirs()
    try:
        sys.stdout = DualLogger(CONSOLE_LOG)
    except OSError as exc:
        print(f'Warning: Could not open console log {CONSOLE_LOG}: {exc}')

BATCH_MODEL = DEFAULT_GEMINI_TRANSLATION_MODEL
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
BATCH_QUALITY_POLICY = translation_quality.normalize_policy(None)
MANIFEST_MODE_TRANSLATION = 'translation'
MANIFEST_MODE_KEYWORD_EXTRACTION = 'keyword_extraction'
MANIFEST_MODE_REVISION = 'revision'
MANIFEST_MODE_FINAL_REVIEW = 'final_review'
CHECK_CONTRACT_VERSION = 3
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
    'response_missing_expected_id',
    'result_missing_field',
    'result_invalid_field_type',
    'result_empty_translation',
    'result_unknown_id',
    'result_unknown_source_id',
    'result_duplicate_id',
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
    'adapter_writeback_block',
    'empty_response_text',
    'invalid_json',
    'response_envelope_missing',
    'response_items_not_array',
    'result_item_not_object',
    'result_missing_id',
}

RAG_ENABLED = False
RAG_STORE_DIR = ''
RAG_EMBEDDING_MODEL = DEFAULT_GEMINI_EMBEDDING_MODEL
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

PROJECT_ANALYSIS_ENABLED = False
PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = False
PROJECT_ANALYSIS_STORE_DIR = ''
PROJECT_ANALYSIS_MAX_BRIEF_CHARS = 4000
PROJECT_ANALYSIS_MODEL = ''
PROJECT_ANALYSIS_THINKING_LEVEL = ''
PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS = 800
PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS = 1200
PROJECT_ANALYSIS_MAX_INPUT_CHARS = 12000
PROJECT_ANALYSIS_MAX_OUTPUT_TOKENS = 2048
_PROJECT_BRIEF_CACHE = None
_PROJECT_BRIEF_CACHE_KEY = None

FINAL_REVIEW_ENABLED = True
FINAL_REVIEW_REQUIRE_ZERO_PENDING = True
FINAL_REVIEW_CHUNK_SIZE = 16
FINAL_REVIEW_PROMPT_SCHEMA_VERSION = 'final-review-v1'
FINAL_REVIEW_MODEL = ''
FINAL_REVIEW_DISPLAY_NAME_PREFIX = 'renpy-final-review'


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


def load_batch_settings(*, tolerate_routing_errors=False):
    """Load persisted settings, normalizing sync timeouts to the shared contract.

    ``sync.timeout_seconds`` is a per-model-request limit in seconds. Missing or
    invalid values default to 120 seconds, and effective values stay within 5-600.
    """
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
    global PROJECT_ANALYSIS_ENABLED, PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
    global PROJECT_ANALYSIS_STORE_DIR, PROJECT_ANALYSIS_MAX_BRIEF_CHARS
    global PROJECT_ANALYSIS_MODEL, PROJECT_ANALYSIS_THINKING_LEVEL
    global PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS, PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS
    global PROJECT_ANALYSIS_MAX_INPUT_CHARS, PROJECT_ANALYSIS_MAX_OUTPUT_TOKENS
    global _PROJECT_BRIEF_CACHE, _PROJECT_BRIEF_CACHE_KEY
    global FINAL_REVIEW_ENABLED, FINAL_REVIEW_REQUIRE_ZERO_PENDING, FINAL_REVIEW_CHUNK_SIZE
    global FINAL_REVIEW_PROMPT_SCHEMA_VERSION, FINAL_REVIEW_MODEL
    global FINAL_REVIEW_DISPLAY_NAME_PREFIX
    global BATCH_NON_CHINESE_RULES, BATCH_QUALITY_POLICY, SYNC_BACKEND, SYNC_MODEL, SYNC_TIMEOUT_SECONDS

    config = load_json_file(legacy.CONFIG_FILE)
    translator_config = load_json_file(legacy.TRANSLATOR_CONFIG)
    # Per-project RAG / source-index flags (work/project_context_settings.json).
    try:
        from project_context_settings import apply_project_context_settings_to_config

        apply_project_context_settings_to_config(translator_config, legacy.BASE_DIR)
    except Exception as exc:
        print(f'Warning: Failed to apply project context settings: {exc}')

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
    BATCH_QUALITY_POLICY = translation_quality.load_policy_from_config(translator_config)

    sync = translator_config.get('sync')
    if not isinstance(sync, dict):
        sync = {}
    backend_name = str(sync.get('backend') or 'gemini').strip().lower()
    if backend_name not in {'gemini', 'litellm'}:
        if not tolerate_routing_errors:
            exc = model_profile.ModelRoutingConfigError(
                f"Unsupported sync backend: {backend_name}. "
                "Choose 'gemini' or 'litellm'."
            )
            raise model_profile.routing_resolution_error(exc) from exc
    SYNC_BACKEND = backend_name
    sync_model = sync.get('model')
    if isinstance(sync_model, str) and sync_model.strip():
        SYNC_MODEL = sync_model.strip()
    else:
        SYNC_MODEL = ''
    SYNC_TIMEOUT_SECONDS = normalize_sync_timeout_seconds(
        sync.get('timeout_seconds'),
        DEFAULT_SYNC_TIMEOUT_SECONDS,
    )
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

    project_analysis_config = batch.get('project_analysis')
    if not isinstance(project_analysis_config, dict):
        project_analysis_config = {}
    PROJECT_ANALYSIS_ENABLED = coerce_bool(
        project_analysis_config.get('enabled'),
        PROJECT_ANALYSIS_ENABLED,
    )
    PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = coerce_bool(
        project_analysis_config.get('inject_published_brief'),
        PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF,
    )
    PROJECT_ANALYSIS_MAX_BRIEF_CHARS = coerce_positive_int(
        project_analysis_config.get('max_brief_chars'),
        PROJECT_ANALYSIS_MAX_BRIEF_CHARS,
    )
    PROJECT_ANALYSIS_MODEL = coerce_non_empty_string(
        project_analysis_config.get('model'),
        PROJECT_ANALYSIS_MODEL,
    )
    if 'thinking_level' in project_analysis_config:
        PROJECT_ANALYSIS_THINKING_LEVEL = coerce_thinking_level(
            project_analysis_config.get('thinking_level'),
            PROJECT_ANALYSIS_THINKING_LEVEL,
        )
    PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS = coerce_positive_int(
        project_analysis_config.get('max_label_summary_chars'),
        PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS,
    )
    PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS = coerce_positive_int(
        project_analysis_config.get('max_route_summary_chars'),
        PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS,
    )
    PROJECT_ANALYSIS_MAX_INPUT_CHARS = coerce_positive_int(
        project_analysis_config.get('max_input_chars_per_request'),
        PROJECT_ANALYSIS_MAX_INPUT_CHARS,
    )
    PROJECT_ANALYSIS_MAX_OUTPUT_TOKENS = coerce_positive_int(
        project_analysis_config.get('max_output_tokens'),
        PROJECT_ANALYSIS_MAX_OUTPUT_TOKENS,
    )
    pa_store = project_analysis_config.get('store_dir')
    if pa_store:
        PROJECT_ANALYSIS_STORE_DIR = legacy._resolve_path(legacy.BASE_DIR, pa_store)
    else:
        PROJECT_ANALYSIS_STORE_DIR = ''
    _PROJECT_BRIEF_CACHE = None
    _PROJECT_BRIEF_CACHE_KEY = None

    final_review_config = batch.get('final_review')
    if not isinstance(final_review_config, dict):
        final_review_config = {}
    FINAL_REVIEW_ENABLED = coerce_bool(
        final_review_config.get('enabled'),
        FINAL_REVIEW_ENABLED,
    )
    FINAL_REVIEW_REQUIRE_ZERO_PENDING = coerce_bool(
        final_review_config.get('require_zero_pending'),
        FINAL_REVIEW_REQUIRE_ZERO_PENDING,
    )
    FINAL_REVIEW_CHUNK_SIZE = coerce_positive_int(
        final_review_config.get('chunk_size'),
        FINAL_REVIEW_CHUNK_SIZE,
    )
    FINAL_REVIEW_PROMPT_SCHEMA_VERSION = coerce_non_empty_string(
        final_review_config.get('prompt_schema_version'),
        FINAL_REVIEW_PROMPT_SCHEMA_VERSION,
    )
    FINAL_REVIEW_MODEL = coerce_non_empty_string(
        final_review_config.get('model'),
        FINAL_REVIEW_MODEL,
    )
    FINAL_REVIEW_DISPLAY_NAME_PREFIX = coerce_non_empty_string(
        final_review_config.get('display_name_prefix'),
        FINAL_REVIEW_DISPLAY_NAME_PREFIX,
    )


def compute_current_project_analysis_fingerprint(base_dir=None, store_dir=None):
    """Recompute structure fingerprint from scripts under build-time roots when known.

    Prefers ``project_identity.script_roots`` persisted by
    ``build_structure_drafts`` so custom ``--script-root`` builds stay injectable.
    Falls back to default game/work/original discovery under *base_dir*.

    When a project moves or is copied, relative graph/root paths are rebased onto
    the current *base_dir*. Legacy absolute paths that were inside the stored
    project base are rebased by the same relative offset; external roots stay put.
    """
    from project_analysis import resolve_project_analysis_store
    from project_analysis_routes import digest_script_paths, discover_script_files

    base = base_dir if base_dir is not None else (legacy.BASE_DIR or None)
    roots = []
    graph_base = base or ''
    resolved_store = store_dir if store_dir is not None else (PROJECT_ANALYSIS_STORE_DIR or None)
    try:
        store = resolve_project_analysis_store(resolved_store, base_dir=base)
        manifest = store.load_manifest() or {}
        identity = manifest.get('project_identity') if isinstance(manifest, dict) else {}
        if isinstance(identity, dict):
            stored_identity_base = str(identity.get('base_dir') or '').strip()
            stored_base = str(identity.get('graph_base') or stored_identity_base).strip()
            if base and stored_base and os.path.isabs(stored_base) and stored_identity_base:
                try:
                    relative_graph_base = os.path.relpath(stored_base, stored_identity_base)
                except ValueError:
                    relative_graph_base = ''
                if relative_graph_base and not relative_graph_base.startswith('..'):
                    stored_base = os.path.join(base, relative_graph_base)
            if stored_base:
                if not os.path.isabs(stored_base) and base:
                    graph_base = os.path.abspath(os.path.join(base, stored_base))
                else:
                    graph_base = stored_base
            elif base:
                graph_base = base
            stored_roots = identity.get('script_roots') or []
            if isinstance(stored_roots, list):
                for raw_root in stored_roots:
                    root = str(raw_root or '').strip()
                    if not root:
                        continue
                    if not os.path.isabs(root):
                        relocation_base = base or stored_base
                        if relocation_base:
                            root = os.path.join(relocation_base, root)
                    elif base and stored_identity_base:
                        try:
                            relative_root = os.path.relpath(root, stored_identity_base)
                        except ValueError:
                            relative_root = ''
                        if relative_root and not relative_root.startswith('..'):
                            root = os.path.join(base, relative_root)
                    roots.append(root)
    except Exception:
        roots = []

    if not roots:
        if not base:
            return ''
        for rel in ('game', os.path.join('work', 'game'), os.path.join('original', 'game')):
            candidate = os.path.join(base, rel)
            if os.path.isdir(candidate):
                roots.append(candidate)
        if not roots:
            roots.append(base)
        graph_base = base

    paths = discover_script_files(roots)
    if not paths:
        return ''
    return digest_script_paths(paths, base_dir=graph_base or base)


def load_injectable_project_context_for_prompts(file_rel_path='', line_numbers=None):
    """Load cached global artifacts and select target-local context in memory."""
    global _PROJECT_BRIEF_CACHE, _PROJECT_BRIEF_CACHE_KEY
    empty = {'text': '', 'diagnostics': '', 'labels': [], 'routes': [], 'local_diagnostics': ''}
    if not PROJECT_ANALYSIS_ENABLED or not PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF:
        return empty
    from project_analysis import (
        KIND_LABEL,
        load_injectable_project_context,
        resolve_project_analysis_store,
        select_project_local_context,
    )

    store_dir = PROJECT_ANALYSIS_STORE_DIR or None
    base_dir = legacy.BASE_DIR or None
    current_fp = compute_current_project_analysis_fingerprint(base_dir, store_dir=store_dir)
    if not current_fp:
        return empty
    normalized_lines = []
    for raw_value in line_numbers or []:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            normalized_lines.append(value)
    cache_key = (
        bool(PROJECT_ANALYSIS_ENABLED), bool(PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF),
        store_dir or '', int(PROJECT_ANALYSIS_MAX_BRIEF_CHARS),
        int(PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS),
        int(PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS), str(base_dir or ''), current_fp,
    )
    source = (
        _PROJECT_BRIEF_CACHE
        if _PROJECT_BRIEF_CACHE is not None and _PROJECT_BRIEF_CACHE_KEY == cache_key
        else None
    )
    if source is None:
        try:
            payload = load_injectable_project_context(
                store_dir=store_dir, base_dir=base_dir,
                expected_source_fingerprint=current_fp,
                max_brief_chars=PROJECT_ANALYSIS_MAX_BRIEF_CHARS,
                enabled=True,
            )
            label_records = []
            route_records = []
            if payload.get('injectable'):
                store = resolve_project_analysis_store(store_dir, base_dir=base_dir)
                label_records = store.load_summaries(KIND_LABEL)
                route_records = store.load_routes()
        except Exception as exc:
            print(f'Warning: project analysis local context unavailable: {exc}', file=sys.stderr)
            return empty
        source = {
            'result': {
                'text': str(payload.get('text') or '') if payload.get('injectable') else '',
                'diagnostics': str(payload.get('diagnostics') or '') if payload.get('injectable') else '',
                'labels': [],
                'routes': [],
                'local_diagnostics': '',
            },
            'label_records': label_records,
            'route_records': route_records,
        }
        _PROJECT_BRIEF_CACHE = source
        _PROJECT_BRIEF_CACHE_KEY = cache_key
    result = dict(source['result'])
    if result['text'] and str(file_rel_path or '').strip():
        result.update(
            select_project_local_context(
                source['label_records'],
                source['route_records'],
                file_rel_path=file_rel_path,
                line_numbers=tuple(sorted(set(normalized_lines))),
                max_label_chars=PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS,
                max_route_chars=PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS,
            )
        )
    return result


def load_injectable_project_brief_for_prompts():
    """Compatibility wrapper returning only the published global brief."""
    payload = load_injectable_project_context_for_prompts()
    return payload['text'], payload['diagnostics']


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
    atomic_write_json(PROGRESS_LOG, progress, ensure_ascii=False, indent=2)


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
    # Use runtime helper so empty BASE_DIR yields "unset", never CWD basename.
    return legacy.guess_project_slug()


def hash_key(text):
    return translation_core.file_hash_key(text)


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


def _canonical_manifest_dir(value, field_name):
    if not isinstance(value, str) or not value.strip():
        return ''
    raw = value.strip()
    if not os.path.isabs(raw):
        raise cli_contract.MachineContractError(
            f'Manifest {field_name} must be an absolute path: {raw}',
            code_name='INVALID_MANIFEST_PATH',
            suggested_action='rebuild_or_repair_manifest',
            details={'field': field_name, 'path': raw},
        )
    return _canonical_abs_path(raw)


def _infer_legacy_manifest_tl_dir(manifest):
    candidates = []
    files_info = manifest.get('files') if isinstance(manifest, dict) else {}
    if not isinstance(files_info, dict):
        return ''
    for file_key, file_info in files_info.items():
        path_value = file_info.get('path') if isinstance(file_info, dict) else ''
        if not isinstance(path_value, str) or not path_value.strip() or not os.path.isabs(path_value):
            continue
        rel_path = normalize_safe_rel_path(file_key, f'manifest file key {file_key}')
        candidate = _canonical_abs_path(path_value)
        for _part in Path(rel_path).parts:
            candidate = os.path.dirname(candidate)
        resolved = resolve_path_under_dir(candidate, rel_path, f'manifest file key {file_key}')
        if _normalized_abs_path(resolved) != _normalized_abs_path(path_value):
            raise cli_contract.MachineContractError(
                (
                    f'Unsafe legacy manifest file path for {file_key}: '
                    'path escapes the inferred translation directory or does not match its file key.'
                ),
                code_name='INVALID_MANIFEST_PATH',
                suggested_action='rebuild_or_repair_manifest',
                details={'field': f'files.{file_key}.path', 'path': path_value},
            )
        candidates.append(_canonical_abs_path(candidate))
    if not candidates:
        return ''
    first = candidates[0]
    if any(_normalized_abs_path(candidate) != _normalized_abs_path(first) for candidate in candidates[1:]):
        raise cli_contract.MachineContractError(
            'Legacy manifest file paths do not share one translation directory.',
            code_name='INVALID_MANIFEST_PATH',
            suggested_action='rebuild_or_repair_manifest',
            details={'field': 'files'},
        )
    return first


def manifest_project_identity(manifest):
    if not isinstance(manifest, dict):
        raise cli_contract.MachineContractError(
            'Manifest project identity is missing; rebuild the batch package.',
            code_name='MANIFEST_PROJECT_IDENTITY_MISSING',
            suggested_action='rebuild_batch_package',
        )
    manifest_tl_dir = _canonical_manifest_dir(manifest.get('tl_dir'), 'tl_dir')
    identity_source = 'manifest'
    if not manifest_tl_dir:
        manifest_tl_dir = _infer_legacy_manifest_tl_dir(manifest)
        identity_source = 'legacy_file_paths'
    if not manifest_tl_dir:
        raise cli_contract.MachineContractError(
            (
                'Manifest project identity is missing and cannot be inferred from absolute file paths; '
                'rebuild the batch package before check/apply.'
            ),
            code_name='MANIFEST_PROJECT_IDENTITY_MISSING',
            suggested_action='rebuild_batch_package',
        )
    return {
        'base_dir': _canonical_manifest_dir(manifest.get('base_dir'), 'base_dir'),
        'tl_dir': manifest_tl_dir,
        'source': identity_source,
    }


def require_manifest_project_match(manifest, command_name):
    identity = manifest_project_identity(manifest)
    active_tl_dir = _canonical_abs_path(legacy.TL_DIR)
    if _normalized_abs_path(identity['tl_dir']) != _normalized_abs_path(active_tl_dir):
        raise cli_contract.MachineContractError(
            (
                f'{command_name} refused: manifest project does not match the active project '
                f'(manifest tl_dir={identity["tl_dir"]}, active tl_dir={active_tl_dir}).'
            ),
            code_name='MANIFEST_PROJECT_MISMATCH',
            suggested_action='select_matching_project_or_manifest',
            details={
                'command': command_name,
                'manifest_tl_dir': identity['tl_dir'],
                'active_tl_dir': active_tl_dir,
            },
        )
    if identity['base_dir']:
        active_base_dir = _canonical_abs_path(legacy.BASE_DIR)
        if _normalized_abs_path(identity['base_dir']) != _normalized_abs_path(active_base_dir):
            raise cli_contract.MachineContractError(
                (
                    f'{command_name} refused: manifest project does not match the active project '
                    f'(manifest base_dir={identity["base_dir"]}, active base_dir={active_base_dir}).'
                ),
                code_name='MANIFEST_PROJECT_MISMATCH',
                suggested_action='select_matching_project_or_manifest',
                details={
                    'command': command_name,
                    'manifest_base_dir': identity['base_dir'],
                    'active_base_dir': active_base_dir,
                },
            )
    return identity


def _manifest_tl_base_dir(manifest):
    return manifest_project_identity(manifest)['tl_dir']


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


def _adapter_target_language_policy_allows(
    manifest,
    chunk,
    item,
    original,
    translated,
    reason='',
    reason_codes=(),
):
    """Allow only the legacy target-language failure through Batch policy."""

    normalized_codes = {
        str(code or '').strip()
        for code in reason_codes or ()
        if str(code or '').strip()
    }
    target_language_only = (
        normalized_codes == {'common.target_language.missing'}
        if normalized_codes
        else reason == 'No Chinese characters'
    )
    return target_language_only and allow_non_chinese_batch_translation(
        manifest, chunk or {}, original, translated, item=item
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
    key_attempts = (
        legacy.api_key_rotation_attempts()
        if hasattr(legacy, 'api_key_rotation_attempts')
        else max(1, api_key_count)
    )
    attempts = max(3, key_attempts * 2)
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
    """Return lexical glossary hits for a chunk, never gated on RAG (issue #346, D2).

    The shared implementation covers ``normalize_map`` / ``preserve_terms`` /
    ``non_translatable_exact`` and deliberately truncates nothing: RAG top-k
    applies only to the RAG ``LOCKED TERMS`` reference list, while the lexical
    injection required by issue #338 must always carry the chunk's real hits.
    """
    return translation_plan.retrieve_lexical_glossary_hits(
        target_items,
        normalize_map=legacy.NORMALIZE_TRANSLATION_MAP,
        preserve_terms=legacy.PRESERVE_TERMS,
        non_translatable_exact=getattr(legacy, 'NON_TRANSLATABLE_EXACT', set()),
    )


def retrieve_revision_glossary_hits(target_items):
    """Return the legacy RAG-gated, top-k-capped glossary hits for revision.

    Revision packages keep their pre-#346 contract: no RAG means no LOCKED
    TERMS, and RAG top-k still caps the reference list. Only the translation
    plan path (issue #346 P2/D2) deliberately uses the unbounded lexical
    helper above.
    """
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
        raise cli_contract.MachineContractError(
            f'Manifest not found: {target}',
            code_name='MANIFEST_NOT_FOUND',
            suggested_action='pass_existing_manifest_path',
            details={'target': str(target), 'resolved_path': candidate},
        )

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
        raise cli_contract.MachineContractError(
            'No batch manifest found.',
            code_name='MANIFEST_NOT_FOUND',
            suggested_action='build_batch_package',
            details={'search_directory': os.path.abspath(BATCH_JOBS_DIR)},
        )
    manifests.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return manifests[0]


def remember_latest_manifest(manifest_path):
    ensure_batch_dirs()
    atomic_write_text(LATEST_MANIFEST_FILE, str(manifest_path))


def load_manifest(target=None):
    """Load and validate a JSON manifest or raise a structured contract error.

    Raises:
        cli_contract.MachineContractError: If the manifest cannot be read as
            UTF-8 JSON or its root value is not a JSON object.
    """

    manifest_path = manifest_path_for_target(target)
    try:
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise cli_contract.MachineContractError(
            (
                f'Manifest is not valid JSON: {manifest_path} '
                f'(line {exc.lineno}, column {exc.colno}).'
            ),
            code_name='INVALID_MANIFEST_JSON',
            suggested_action='rebuild_or_repair_manifest',
            details={
                'manifest_path': manifest_path,
                'line': exc.lineno,
                'column': exc.colno,
            },
        ) from exc
    except UnicodeDecodeError as exc:
        raise cli_contract.MachineContractError(
            f'Manifest is not valid UTF-8: {manifest_path}.',
            code_name='INVALID_MANIFEST_ENCODING',
            suggested_action='rebuild_or_repair_manifest',
            details={'manifest_path': manifest_path},
        ) from exc
    except OSError as exc:
        raise cli_contract.MachineContractError(
            f'Manifest could not be read: {manifest_path} ({exc}).',
            code_name='MANIFEST_UNREADABLE',
            suggested_action='inspect_manifest_permissions',
            details={'manifest_path': manifest_path},
        ) from exc
    if not isinstance(manifest, dict):
        raise cli_contract.MachineContractError(
            f'Manifest root must be a JSON object: {manifest_path}.',
            code_name='INVALID_MANIFEST_SHAPE',
            suggested_action='rebuild_or_repair_manifest',
            details={
                'manifest_path': manifest_path,
                'actual_type': type(manifest).__name__,
            },
        )
    manifest['_manifest_path'] = manifest_path
    manifest['_package_dir'] = os.path.dirname(manifest_path)
    return manifest


def manifest_mode(manifest):
    mode = manifest.get('mode', MANIFEST_MODE_TRANSLATION)
    return mode if isinstance(mode, str) and mode.strip() else MANIFEST_MODE_TRANSLATION


def require_manifest_mode(manifest, expected_mode, command_name):
    current_mode = manifest_mode(manifest)
    if current_mode != expected_mode:
        raise cli_contract.MachineContractError(
            (
                f'{command_name} only supports {expected_mode} manifests; '
                f'this manifest is {current_mode}.'
            ),
            code_name='MANIFEST_MODE_MISMATCH',
            suggested_action='use_matching_workflow',
            details={
                'command': command_name,
                'expected_mode': expected_mode,
                'actual_mode': current_mode,
            },
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
    tl_dir = _manifest_tl_base_dir(manifest)
    path_value = file_info.get('path') if isinstance(file_info, dict) else ''
    if path_value:
        return resolve_path_under_dir(tl_dir, path_value, f'manifest file path for {file_key}')
    return resolve_path_under_dir(tl_dir, file_key, f'manifest file key {file_key}')


def save_manifest(manifest, update_latest=True):
    manifest_path = manifest['_manifest_path']
    data = dict(manifest)
    data.pop('_manifest_path', None)
    data.pop('_package_dir', None)
    atomic_write_json(manifest_path, data, ensure_ascii=False, indent=2)
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
    project_identity = manifest_project_identity(manifest)
    payload = {
        'check_contract_version': CHECK_CONTRACT_VERSION,
        'manifest_path': os.path.abspath(manifest.get('_manifest_path', '')) if manifest.get('_manifest_path') else '',
        'package_id': os.path.basename(os.path.abspath(package_dir)) if package_dir else '',
        'mode': manifest_mode(manifest),
        'manifest_version': manifest.get('manifest_version', 1),
        'core_schema_version': manifest.get('core_schema_version', 1),
        'batch_model': manifest.get('batch_model', ''),
        'settings': manifest.get('settings') or {},
        'quality_policy': BATCH_QUALITY_POLICY,
        'project': project_identity,
        'result': file_content_fingerprint(result_path),
        'target_shape': manifest_target_shape(manifest),
    }
    if isinstance(manifest.get('durable_sync_source'), dict):
        payload['durable_sync_source'] = dict(manifest['durable_sync_source'])
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


def summarize_writeback_gate(safety, quality_gate):
    structural_blocker_count = safety.get('counts', {}).get(CHECK_SAFETY_BLOCK, 0)
    # Legacy structural warnings (partial rows, missing ids, ...) remain part of
    # the #39 writeback safety contract.  They are not quality warnings.
    structural_blocker_count += safety.get('counts', {}).get(CHECK_SAFETY_WARN, 0)
    quality_blocker_count = int((quality_gate or {}).get('blocker_count') or 0)
    blocker_count = structural_blocker_count + quality_blocker_count
    can_apply = blocker_count == 0 and safety.get('level') != CHECK_SAFETY_BLOCK
    if safety.get('level') == CHECK_SAFETY_BLOCK:
        can_apply = False
    return {
        'decision': (
            translation_quality.GATE_ALLOW if can_apply else translation_quality.GATE_DENY
        ),
        'can_apply': can_apply,
        'blocker_count': blocker_count,
        'structural_blocker_count': structural_blocker_count,
        'quality_blocker_count': quality_blocker_count,
    }


def attach_check_contract(manifest, summary, quality_findings=None):
    safety = summarize_check_safety(summary)
    if quality_findings is not None:
        pruned_ids = translation_quality.prune_acknowledged_finding_ids(
            manifest.get('quality_acknowledged_finding_ids'),
            quality_findings,
        )
        manifest['quality_acknowledged_finding_ids'] = pruned_ids
        quality_gate = translation_quality.summarize_quality_gate(
            quality_findings,
            acknowledged_ids=pruned_ids,
        )
    else:
        # Apply revalidation does not recompute quality findings; the preflight
        # already proved the last check is fresh.  Reuse the persisted quality
        # gate so configured blockers remain visible and still block writeback.
        last_summary = manifest.get('last_check_summary')
        persisted_quality_gate = (
            last_summary.get('quality_gate')
            if isinstance(last_summary, dict)
            else None
        )
        if isinstance(persisted_quality_gate, dict):
            quality_gate = dict(persisted_quality_gate)
            quality_gate.setdefault('has_warnings', False)
            quality_gate.setdefault('acknowledged_count', 0)
        else:
            quality_gate = translation_quality.summarize_quality_gate(
                [],
                acknowledged_ids=manifest.get('quality_acknowledged_finding_ids') or [],
            )
    writeback_gate = summarize_writeback_gate(safety, quality_gate)
    can_apply = bool(writeback_gate.get('can_apply'))
    check_status = translation_quality.overall_check_status(writeback_gate, quality_gate)

    summary['check_contract_version'] = CHECK_CONTRACT_VERSION
    summary['check_fingerprint'] = build_check_fingerprint(manifest)
    # ``safety_level`` remains the legacy *structural* safety label.  Quality
    # blockers are expressed through writeback_gate / quality_gate so the GUI
    # can explain the actual reason instead of misreporting a source problem.
    summary['safety_level'] = safety['level']
    summary['safety_counts'] = safety['counts']
    summary['safety_reasons'] = safety['reasons']
    summary['check_status'] = check_status
    summary['can_apply'] = can_apply
    summary['has_warnings'] = bool(quality_gate.get('has_warnings'))
    summary['writeback_gate'] = writeback_gate
    summary['quality_gate'] = quality_gate
    summary['quality_finding_schema_version'] = (
        translation_quality.QUALITY_FINDING_SCHEMA_VERSION
    )
    summary['quality_rule_schema_version'] = translation_quality.QUALITY_RULE_SCHEMA_VERSION
    summary['quality_policy_digest'] = translation_quality.policy_digest(BATCH_QUALITY_POLICY)
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
    atomic_write_json(path, payload, ensure_ascii=False, indent=2)


def write_jsonl_report(path, entries):
    atomic_write_jsonl(path, entries, ensure_ascii=False)


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
    blocked = reason_code == 'unsafe_check_status'
    raise cli_contract.MachineContractError(
        f'{message} Report: {report_path}',
        code_name={
            'missing_check': 'APPLY_CHECK_REQUIRED',
            'stale_check_contract': 'STALE_CHECK_CONTRACT',
            'stale_check_fingerprint': 'STALE_CHECK_FINGERPRINT',
            'unsafe_check_status': 'UNSAFE_CHECK_STATUS',
        }.get(reason_code, 'APPLY_PREFLIGHT_FAILED'),
        suggested_action='repair_results_or_run_check' if blocked else 'run_check_again',
        semantic_exit_code=(
            cli_contract.EXIT_BLOCKED if blocked else cli_contract.EXIT_INVALID_STATE
        ),
        details={
            'reason_code': reason_code,
            'report_path': report_path,
            'current_fingerprint': current_fingerprint or {},
        },
    )


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

    writeback_gate = last_summary.get('writeback_gate')
    if not isinstance(writeback_gate, dict):
        fail_apply_preflight(
            manifest,
            'missing_writeback_gate',
            'Manifest check summary has no writeback gate. Run check again before apply.',
            current_fingerprint=current_fingerprint,
        )
    if writeback_gate.get('decision') != translation_quality.GATE_ALLOW:
        reason_code = 'unsafe_check_status'
        safety_level = last_summary.get('safety_level')
        quality_gate = last_summary.get('quality_gate') or {}
        message = (
            f'Last check writeback gate is {writeback_gate.get("decision") or "unknown"}, not safe to apply. '
            f'(safety={safety_level or "unknown"}, '
            f'quality={quality_gate.get("decision") or "pass"}). '
            'Repair the results or run check again before apply.'
        )
        fail_apply_preflight(manifest, reason_code, message, current_fingerprint=current_fingerprint)




def load_request_rows(manifest):
    input_jsonl_path = manifest.get('input_jsonl_path')
    if not input_jsonl_path or not os.path.isfile(input_jsonl_path):
        raise cli_contract.MachineContractError(
            f"Input JSONL not found: {input_jsonl_path}",
            code_name='BATCH_INPUT_NOT_FOUND',
            suggested_action='rebuild_batch_package',
            details={'input_jsonl_path': input_jsonl_path or ''},
        )
    rows = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise cli_contract.MachineContractError(
                    (
                        f'Batch input JSONL is not valid JSON: {input_jsonl_path} '
                        f'(line {line_number}, column {exc.colno}).'
                    ),
                    code_name='INVALID_BATCH_INPUT_JSON',
                    suggested_action='rebuild_batch_package',
                    details={
                        'input_jsonl_path': input_jsonl_path,
                        'line': line_number,
                        'column': exc.colno,
                    },
                ) from exc
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
    if source_manifest.get('model_routing'):
        part_manifest['model_routing'] = copy.deepcopy(source_manifest.get('model_routing'))
    if source_manifest.get('translation_plan'):
        part_manifest['translation_plan'] = copy.deepcopy(source_manifest.get('translation_plan'))
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
    atomic_write_json(snapshot_path, payload, ensure_ascii=False, indent=2)
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


class TranslationFileJobs(list):
    """Legacy-compatible job list carrying its read-only coverage snapshot."""

    def __init__(self, values=(), *, coverage_snapshot=None, adapter_snapshot=None):
        super().__init__(values)
        self.coverage_snapshot = coverage_snapshot
        self.adapter_snapshot = adapter_snapshot


def collect_pending_file_jobs(
    *,
    include_complete_files=False,
    include_occurrences=True,
    include_task_payloads=True,
):
    """Collect per-file pending translation tasks.

    By default only files with at least one pending task are returned (batch build).
    Pass ``include_complete_files=True`` for doctor progress so fully-translated
    files still contribute to ``translated_count``.

    Progress-only callers (environment check) should pass
    ``include_occurrences=False`` and usually ``include_task_payloads=False``:
    pending/translated counts stay identical to the full build path, but the
    expensive occurrence extraction and large task list copies are skipped.
    """
    adapter_snapshot = build_translation_snapshot(
        RenPyAdapter(legacy_module=legacy),
        ProjectDiscoveryRequest(
            project_root=legacy.BASE_DIR,
            localization_root=legacy.TL_DIR,
            target_language=legacy.PREP_LANGUAGE,
            include_files=tuple(sorted(legacy.INCLUDE_FILES)),
            include_prefixes=tuple(sorted(legacy.INCLUDE_PREFIXES)),
        ),
        include_occurrences=include_occurrences,
        include_task_payloads=include_task_payloads,
    )
    jobs = TranslationFileJobs(
        coverage_snapshot=adapter_snapshot,
        adapter_snapshot=adapter_snapshot,
    )

    for document in adapter_snapshot.project.source_documents:
        rel_path = document.file_rel_path
        file_path = document.file_path
        if include_task_payloads:
            pending = [
                dict(task)
                for task in adapter_snapshot.pending_tasks_by_file.get(rel_path, ())
                if not legacy.is_non_translatable(task['text'])
            ]
            task_count = len(pending)
        else:
            task_count = sum(
                1
                for candidate in adapter_snapshot.inventory.candidates
                if candidate.classification == "translatable"
                and candidate.legacy_item is not None
                and str(candidate.locator.locator.get("file_rel_path") or "") == rel_path
                and not legacy.is_non_translatable(candidate.legacy_item["text"])
            )
            pending = []
        progress = adapter_snapshot.progress_by_file.get(rel_path, {})
        translated_count = int(progress.get('translated_count') or 0)
        if task_count or (include_complete_files and translated_count):
            jobs.append(
                {
                    'file_rel_path': rel_path,
                    'file_path': file_path,
                    'task_count': task_count,
                    'translated_count': translated_count,
                    'tasks': pending,
                }
            )

    return jobs


def summarize_translation_progress(file_jobs):
    """Aggregate pending vs already-translated task counts from file jobs."""
    pending_task_count = 0
    translated_task_count = 0
    pending_file_count = 0
    for job in file_jobs or []:
        task_count = int(job.get('task_count') or 0)
        translated_count = int(job.get('translated_count') or 0)
        pending_task_count += task_count
        translated_task_count += translated_count
        if task_count > 0:
            pending_file_count += 1
    return {
        'pending_task_count': pending_task_count,
        'translated_task_count': translated_task_count,
        'total_task_count': pending_task_count + translated_task_count,
        'pending_file_count': pending_file_count,
    }


def collect_doctor_translation_progress():
    """Doctor-only progress summary: same counts as full pending jobs, cheaper path.

    Uses the same inventory/classification and ``is_non_translatable`` filter as
    :func:`collect_pending_file_jobs`, but skips occurrence extraction and does
    not materialize per-task payloads.
    """
    file_jobs = collect_pending_file_jobs(
        include_complete_files=True,
        include_occurrences=False,
        include_task_payloads=False,
    )
    return summarize_translation_progress(file_jobs)


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
    file_rel_path='',
):
    target_units = translation_core.units_from_items(target_items)
    target_file = file_rel_path or next(
        (unit.file_rel_path for unit in target_units if unit.file_rel_path),
        '',
    )
    project_context = load_injectable_project_context_for_prompts(
        target_file,
        [unit.display_line_number for unit in target_units],
    )
    return translation_core.build_translation_user_prompt(
        translation_core.ContextWindow(context_past, context_future),
        target_items,
        translation_core.build_context_bundle(
            glossary_hits=glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
            source_hits=source_hits,
            project_brief_text=project_context['text'],
            project_brief_diagnostics=project_context['diagnostics'],
            project_local_labels=project_context['labels'],
            project_local_routes=project_context['routes'],
            project_local_diagnostics=project_context['local_diagnostics'],
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


def build_generation_config(target_items, model=None):
    effective_model = str(model or BATCH_MODEL or '')
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_response_json_schema(target_items),
    }
    if BATCH_THINKING_LEVEL and is_gemini_3_model(effective_model):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return filter_gemini_generation_config(effective_model, config)


def _chunk_has_plan_request(chunk):
    """True when a chunk carries a P1 TranslationRequest rendering (P2 plan path)."""
    return bool(
        isinstance(chunk, dict)
        and chunk.get('request_id')
        and chunk.get('system_instruction')
        and chunk.get('user_prompt')
    )


def build_batch_request(chunk, model=None):
    """Wrap a chunk as a Gemini Batch request row.

    Chunks built by the issue #346 P2 plan path already carry the rendered
    ``system_instruction`` / ``user_prompt`` / ``response_schema`` and audit
    fingerprints; this function only adds the Batch envelope and keeps the
    request row auditable. Legacy chunks and retry subchunks still use the
    legacy prompt builders.
    """
    if _chunk_has_plan_request(chunk):
        generation_config = build_generation_config(chunk['items'], model=model)
        response_schema = chunk.get('response_schema')
        if isinstance(response_schema, dict) and response_schema:
            generation_config['response_json_schema'] = response_schema
        request = {
            'system_instruction': {
                'parts': [{'text': chunk['system_instruction']}],
            },
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': chunk['user_prompt']}],
                }
            ],
            'generation_config': generation_config,
        }
        if BATCH_SAFETY_SETTINGS:
            request['safety_settings'] = BATCH_SAFETY_SETTINGS
        row = {
            'key': chunk['key'],
            'request': request,
        }
        for field in (
            'request_id',
            'plan_id',
            'chunk_id',
            'prompt_fingerprint',
            'request_fingerprint',
        ):
            if chunk.get(field):
                row[field] = chunk[field]
        return row

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
                            file_rel_path=chunk.get('file_rel_path') or '',
                        )
                    }
                ],
            }
        ],
        'generation_config': build_generation_config(chunk['items'], model=model),
    }
    if BATCH_SAFETY_SETTINGS:
        request['safety_settings'] = BATCH_SAFETY_SETTINGS
    return {
        'key': chunk['key'],
        'request': request,
    }

def task_text_char_count(task):
    return translation_core.translation_text_char_count(task)


def iter_translation_chunk_ranges(tasks):
    yield from translation_core.iter_translation_chunk_ranges(
        tasks,
        max_items=BATCH_TARGET_SIZE,
        max_chars=BATCH_TARGET_CHARS,
    )


def count_translation_chunks(file_jobs):
    total_chunks = 0
    for job in file_jobs:
        total_chunks += sum(1 for _ in iter_translation_chunk_ranges(job.get('tasks') or []))
    return total_chunks


def _batch_plan_context_policy():
    """Return the D1/D5 context policy used by Batch plan requests."""
    return translation_plan.ContextPolicy(
        local_context_before=BATCH_CONTEXT_BEFORE,
        local_context_after=BATCH_CONTEXT_AFTER,
        # The retrieval layer backstop is history+story in the shared
        # policy. Batch can additionally inject Source Index reference text
        # (top_k * char_limit), so add that budget to the backstop;
        # section-level limits are already applied by the providers.
        history_char_limit=RAG_HISTORY_CHAR_LIMIT + get_source_index_char_budget(),
        story_char_limit=STORY_MEMORY_MAX_CONTEXT_CHARS,
        # Project analysis injects both the global brief and local
        # label/route summaries; keep the backstop large enough for all
        # three sections instead of only max_brief_chars.
        analysis_char_limit=(
            PROJECT_ANALYSIS_MAX_BRIEF_CHARS
            + PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS
            + PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS
        ),
        include_source_text=True,
        include_translation_memory=True,
        story_block_suffix='\n\n',
    )


def _batch_plan_generation_config():
    """D6 generation metadata shared by plan requests (response schema is added
    by :func:`build_batch_request` per chunk).
    """
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
    }
    if BATCH_THINKING_LEVEL and is_gemini_3_model(BATCH_MODEL):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return config


def _batch_plan_source_identity(file_jobs):
    """Build the P1 source identity from the adapter snapshot when available."""
    snapshot = getattr(file_jobs, 'adapter_snapshot', None)
    if snapshot is None:
        return translation_plan.SourceIdentity()
    project = snapshot.project
    documents = tuple(project.source_documents or ())
    return translation_plan.SourceIdentity(
        engine=str(project.engine or ''),
        adapter_version=str(project.adapter_version or ''),
        project_identity_digest=str(project.project_snapshot_fingerprint or ''),
        source_snapshot_fingerprint=source_snapshot_fingerprint(documents) if documents else '',
        file_digests={
            str(document.file_rel_path or ''): str(document.sha256 or '')
            for document in documents
        },
    )


def _batch_plan_config_snapshot():
    """Non-sensitive config snapshot used for plan identity."""
    return {
        'target_language': getattr(legacy, 'PREP_LANGUAGE', ''),
        'source_language': getattr(legacy, 'PREP_SOURCE_LANGUAGE', ''),
        'batch': current_batch_settings_snapshot(),
        'rag_enabled': RAG_ENABLED,
        'source_index_enabled': SOURCE_INDEX_ENABLED,
        'story_memory_enabled': STORY_MEMORY_ENABLED,
        'project_analysis_enabled': PROJECT_ANALYSIS_ENABLED,
        'macro_setting': BATCH_MACRO_SETTING,
    }


def _render_batch_retrieval_reference_text(history_hits, story_hits, source_hits):
    """Render the retrieval layer without the lexical glossary block.

    Lexical glossary is injected by the canonical prompt's project layer
    (issue #346, D2); the retrieval layer carries only RAG history, source
    index, and story memory reference text.
    """
    blocks = []
    if history_hits:
        blocks.append(
            'RETRIEVED MEMORY:\n'
            f'{format_history_hits_block(history_hits)}\n\n'
        )
    if source_hits:
        blocks.append(
            'RELATED PROJECT CONTEXT:\n'
            f'{prompt_context.format_source_hits_block(source_hits)}\n\n'
        )
    if story_memory.has_story_hits(story_hits):
        blocks.append(
            'STORY MEMORY:\n'
            f"{story_memory.format_story_hits_block(story_hits, STORY_MEMORY_MAX_CONTEXT_CHARS)}\n\n"
        )
    return ''.join(blocks)


def _render_batch_analysis_reference_text(project_context):
    """Render the Published Project Analysis layer from existing batch context."""
    project_context = project_context or {}
    blocks = []
    brief = str(project_context.get('text') or '').strip()
    if brief:
        diagnostics = str(project_context.get('diagnostics') or '')
        blocks.append(
            'PROJECT BRIEF:\n'
            f'{prompt_context.format_project_brief_block(brief, diagnostics=diagnostics)}\n\n'
        )
    local_context = prompt_context.format_project_local_context_block(
        project_context.get('labels') or [],
        project_context.get('routes') or [],
        str(project_context.get('local_diagnostics') or ''),
    )
    if local_context.strip():
        blocks.append(f'PROJECT LOCAL CONTEXT:\n{local_context}\n\n')
    return ''.join(blocks)


def _build_retry_subchunk_plan_request(parent_chunk, subchunk):
    """Build a canonical, lineage-bearing TranslationRequest for a retry subchunk.

    This keeps issue #346 P2's contract for split/item-scoped retries: the
    subchunk still sends the canonical system/user prompt (D3/D5), the local
    glossary still follows D2, and the request id is a deterministic child of
    the parent plan id and subchunk key (D7-style lineage). Legacy chunks
    without plan fields still use the legacy retry prompt path.
    """
    file_rel_path = str(subchunk.get('file_rel_path') or '')
    file_path = str(subchunk.get('file_path') or '')
    target_items = list(subchunk.get('items') or [])
    target_units = translation_core.units_from_items(
        target_items,
        translation_core.MODE_TRANSLATION,
        file_rel_path=file_rel_path,
        file_path=file_path,
    )
    expected_ids = [unit.id for unit in target_units]
    context_window = translation_core.ContextWindow(
        subchunk.get('context_past') or [],
        subchunk.get('context_future') or [],
    )
    lexical_hits = retrieve_glossary_hits(target_items)
    context_past = list(subchunk.get('context_past') or [])
    context_future = list(subchunk.get('context_future') or [])
    history_hits, _rag_stats = (
        retrieve_history_hits(target_items, context_past)
        if RAG_ENABLED
        else ([], {'enabled': False})
    )
    source_hits, _source_index_stats = (
        retrieve_source_hits(target_items, context_past)
        if SOURCE_INDEX_ENABLED
        else ([], {})
    )
    story_hits = (
        retrieve_batch_story_hits(
            file_rel_path,
            target_items,
            context_past,
            context_future,
        )
        if STORY_MEMORY_ENABLED
        else None
    )
    retrieval_text = _render_batch_retrieval_reference_text(history_hits, story_hits, source_hits)
    project_context = load_injectable_project_context_for_prompts(
        file_rel_path,
        [unit.display_line_number for unit in target_units],
    )
    analysis_text = _render_batch_analysis_reference_text(project_context)
    chunk_input = translation_plan.ChunkContextInput(
        file_rel_path=file_rel_path,
        target_items=target_items,
        target_units=target_units,
        context_window=context_window,
        local_context_diagnostics={
            'retry_subchunk': True,
            'parent_key': str(parent_chunk.get('key') or ''),
        },
        macro_setting=BATCH_MACRO_SETTING,
        lexical_glossary_hits=lexical_hits,
        retrieval_blocks_text=retrieval_text,
        analysis_blocks_text=analysis_text,
    )
    assembly = translation_plan.assemble_context_layers(chunk_input, _batch_plan_context_policy())
    reference_blocks_text = '\n\n'.join(
        layer.text.rstrip('\n')
        for layer in assembly.layers
        if layer.layer
        in (translation_plan.CONTEXT_LAYER_RETRIEVAL, translation_plan.CONTEXT_LAYER_ANALYSIS)
        and layer.text
    )
    system_instruction = translation_core.build_canonical_translation_system_instruction(
        legacy.PRESERVE_TERMS,
        macro_setting=BATCH_MACRO_SETTING,
    )
    user_prompt = translation_core.build_canonical_translation_user_prompt(
        context_window,
        target_units,
        reference_blocks_text=reference_blocks_text,
        lexical_glossary_text=translation_plan.render_lexical_glossary_text(lexical_hits),
    )
    plan_id = str(parent_chunk.get('plan_id') or '')
    if not plan_id:
        plan_id = translation_plan.short_fingerprint(
            translation_plan.canonical_json({'retry_of': str(parent_chunk.get('key') or '')})
        )
    chunk_id = str(subchunk.get('key') or '')
    request = translation_plan.TranslationRequest(
        request_id=translation_plan.build_request_id(plan_id, chunk_id, expected_ids),
        plan_id=plan_id,
        chunk_id=chunk_id,
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        response_schema=translation_core.build_response_json_schema(
            target_units,
            mode=translation_core.MODE_TRANSLATION,
        ),
        expected_ids=expected_ids,
        capability_requirements={
            'structured_output': True,
            'context_budget_tokens': translation_plan.estimate_context_tokens(
                system_instruction,
                user_prompt,
            ),
            'estimate_method': translation_plan.CONTEXT_TOKEN_ESTIMATE_METHOD,
        },
        generation_config=translation_plan.redact_sensitive(_batch_plan_generation_config()),
        transport_metadata=translation_plan.redact_sensitive(
            {
                'batch_key': chunk_id,
                'retry_parent_key': str(parent_chunk.get('key') or ''),
                'retry_item_start': subchunk.get('retry_item_start'),
                'retry_item_end': subchunk.get('retry_item_end'),
                'retry_item_ids': list(subchunk.get('retry_item_ids') or []),
            }
        ),
        context_assembly=assembly.to_dict(),
    )
    request.prompt_fingerprint = translation_plan.short_fingerprint(
        translation_plan.canonical_json(request.semantic_payload())
    )
    request.request_fingerprint = translation_plan.short_fingerprint(
        translation_plan.canonical_json(request.audit_payload())
    )
    return request

def _build_batch_translation_plan(file_jobs, routing_plan=None):
    """Build the issue #346 P2 plan for Gemini Batch and its legacy chunks.

    Returns ``{'plan_build': PlanBuild, 'chunks': [legacy chunk dicts]}``.
    The legacy chunk dicts keep the pre-#346 shape consumed by submit/probe/
    split/repair, and additionally carry the rendered ``TranslationRequest``
    fields so :func:`build_batch_request` only wraps them in a Batch envelope.
    """
    total_chunks = count_translation_chunks(file_jobs)
    if SOURCE_INDEX_ENABLED and total_chunks:
        print(f'Source index retrieval for build: {total_chunks} chunks to query.')
        sys.stdout.flush()

    captures = []
    per_file_chunk_numbers = {}

    def retrieval_provider(chunk_input):
        target_items = chunk_input.target_items
        file_rel_path = str(chunk_input.file_rel_path or '')
        before = list(chunk_input.context_window.before or [])
        after = list(chunk_input.context_window.after or [])
        history_hits, rag_stats = (
            retrieve_history_hits(target_items, before) if RAG_ENABLED else ([], {'enabled': False})
        )
        if SOURCE_INDEX_ENABLED:
            chunk_number = per_file_chunk_numbers.get(file_rel_path, 0) + 1
            per_file_chunk_numbers[file_rel_path] = chunk_number
            print(
                'Source index retrieval progress: '
                f'{len(captures) + 1}/{total_chunks} chunks, '
                f'file={file_rel_path}, chunk={chunk_number}.'
            )
            sys.stdout.flush()
        source_hits, source_index_stats = (
            retrieve_source_hits(target_items, before) if SOURCE_INDEX_ENABLED else ([], {})
        )
        story_hits = (
            retrieve_batch_story_hits(
                file_rel_path,
                target_items,
                before,
                after,
            )
            if STORY_MEMORY_ENABLED
            else None
        )
        captures.append(
            {
                'file_rel_path': file_rel_path,
                'target_items': target_items,
                'target_units': chunk_input.target_units,
                'context_past': before,
                'context_future': after,
                'glossary_hits': retrieve_glossary_hits(target_items),
                'history_hits': history_hits,
                'rag_stats': rag_stats,
                'source_hits': source_hits,
                'source_index_stats': source_index_stats,
                'story_hits': story_hits,
            }
        )
        return _render_batch_retrieval_reference_text(history_hits, story_hits, source_hits)

    def analysis_provider(chunk_input):
        target_units = chunk_input.target_units
        project_context = load_injectable_project_context_for_prompts(
            str(chunk_input.file_rel_path or ''),
            [unit.display_line_number for unit in target_units],
        )
        return _render_batch_analysis_reference_text(project_context)

    if routing_plan is None:
        model_profile_snapshot = None
    else:
        translation_route = routing_plan.routes.get(model_profile.STAGE_TRANSLATION)
        model_profile_snapshot = (
            routing_plan.profiles.get(translation_route.profile_id)
            if translation_route is not None
            else None
        )

    batch_generation_config = _batch_plan_generation_config()
    plan_build = translation_plan.build_translation_plan(
        file_jobs,
        execution_strategy=model_profile.ExecutionStrategy.GEMINI_BATCH.value,
        source_identity=_batch_plan_source_identity(file_jobs),
        config_snapshot=_batch_plan_config_snapshot(),
        model_profile_snapshot=model_profile_snapshot,
        run_id='',
        artifacts=None,
        chunk_policy=translation_plan.ChunkPolicy(
            max_items=BATCH_TARGET_SIZE,
            max_chars=BATCH_TARGET_CHARS,
        ),
        context_policy=_batch_plan_context_policy(),
        preserve_terms=legacy.PRESERVE_TERMS,
        normalize_map=legacy.NORMALIZE_TRANSLATION_MAP,
        non_translatable_exact=getattr(legacy, 'NON_TRANSLATABLE_EXACT', set()),
        macro_setting=BATCH_MACRO_SETTING,
        retrieval_blocks_provider=retrieval_provider,
        analysis_blocks_provider=analysis_provider,
        generation_config=batch_generation_config,
    )

    chunks = []
    plan_chunks = list(plan_build.plan.chunks)
    requests = list(plan_build.requests)
    for index, request in enumerate(requests):
        capture = captures[index] if index < len(captures) else {}
        plan_chunk = plan_chunks[index] if index < len(plan_chunks) else None
        chunk = {
            'key': request.chunk_id,
            'mode': MANIFEST_MODE_TRANSLATION,
            'file_rel_path': plan_chunk.file_rel_path if plan_chunk else capture.get('file_rel_path', ''),
            'file_path': plan_chunk.file_path if plan_chunk else '',
            'chunk_index': plan_chunk.chunk_index if plan_chunk else index + 1,
            'line_numbers': list(plan_chunk.line_numbers) if plan_chunk else [],
            'source_char_count': plan_chunk.source_char_count if plan_chunk else 0,
            'context_past': capture.get('context_past', []),
            'context_future': capture.get('context_future', []),
            'glossary_hits': capture.get('glossary_hits', []),
            'history_hits': capture.get('history_hits', []),
            'rag_stats': capture.get('rag_stats', {}),
            'source_hits': capture.get('source_hits', []),
            'source_index_stats': capture.get('source_index_stats', {}),
            'items': [
                translation_core.legacy_item_from_unit(unit, translation_core.MODE_TRANSLATION)
                for unit in capture.get('target_units', [])
            ],
            'request_id': request.request_id,
            'plan_id': request.plan_id,
            'chunk_id': request.chunk_id,
            'system_instruction': request.system_instruction,
            'user_prompt': request.user_prompt,
            'response_schema': request.response_schema,
            'expected_ids': request.expected_ids,
            'capability_requirements': request.capability_requirements,
            'generation_config': request.generation_config,
            'transport_metadata': request.transport_metadata,
            'context_assembly': request.context_assembly,
            'prompt_fingerprint': request.prompt_fingerprint,
            'request_fingerprint': request.request_fingerprint,
        }
        if STORY_MEMORY_ENABLED and story_memory.has_story_hits(capture.get('story_hits')):
            chunk['story_hits'] = capture['story_hits']
        chunks.append(chunk)

    if SOURCE_INDEX_ENABLED and total_chunks:
        print(
            f'Source index retrieval complete: {len(captures)}/{total_chunks} chunks queried.'
        )
        sys.stdout.flush()
    return {'plan_build': plan_build, 'chunks': chunks}


def build_chunks(file_jobs):
    """Build legacy-compatible chunks via the shared TranslationPlan (issue #346 P2)."""
    return _build_batch_translation_plan(file_jobs).get('chunks', [])


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
    if is_gemini_3_model(BATCH_MODEL) and BATCH_THINKING_LEVEL and BATCH_THINKING_LEVEL.lower() != 'minimal':
        warnings_list.append(
            f'thinking_level={BATCH_THINKING_LEVEL} may waste output budget on reasoning tokens.'
        )
    return warnings_list


def create_batch_package(display_name_override='', skip_prepare=False):
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.GEMINI_BATCH,
        required_stages={model_profile.STAGE_TRANSLATION},
    )
    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    file_jobs = collect_pending_file_jobs()
    if not file_jobs:
        print('No pending lines to translate.')
        return None

    rag_prepare_summary = prepare_rag_store(file_jobs)

    batch_plan_payload = _build_batch_translation_plan(file_jobs, routing_plan=routing_plan)
    chunks = batch_plan_payload.get('chunks', [])
    if not chunks:
        print('No chunks built.')
        return None
    translation_plan_manifest = batch_plan_payload.get('plan_build')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    package_name = f'{timestamp}_{guess_project_slug()}'
    package_dir = os.path.join(BATCH_JOBS_DIR, package_name)
    os.makedirs(package_dir, exist_ok=True)
    coverage_export_warnings = []
    coverage_snapshot = getattr(file_jobs, 'coverage_snapshot', None)
    if coverage_snapshot is not None:
        try:
            export_coverage_package(
                os.path.join(package_dir, 'coverage'),
                coverage_snapshot.project,
                coverage_snapshot.inventory,
                coverage_snapshot.report,
                review_policy=coverage_snapshot.review_policy,
            )
        except (OSError, ValueError) as exc:
            # Coverage is read-only P1 evidence; do not abort package creation.
            coverage_export_warnings.append(f'Coverage export skipped: {exc}')

    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        display_name = f'{BATCH_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'

    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    with open(input_jsonl_path, 'w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(build_batch_request(chunk), ensure_ascii=False) + '\n')

    build_warnings = get_batch_risk_warnings()
    build_warnings.extend(coverage_export_warnings)

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
        **translation_quality.manifest_quality_policy_fields(runtime_policy=BATCH_QUALITY_POLICY),
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'translation_plan': (
            translation_plan_manifest.plan.to_dict()
            if translation_plan_manifest is not None
            else {}
        ),
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
    attach_model_routing(
        manifest,
        routing_plan,
    )

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


def collect_revision_file_jobs(file_paths=None, include_empty_files=False):
    """Collect revision old/new jobs, optionally over an explicit file set.

    ``file_paths`` must be an iterable of ``(rel_path, abs_path)`` pairs (for
    example the output of ``collect_files_to_process()``). Passing the same set
    that produced the source digests removes the new-file race window between
    digest collection and scanning. ``include_empty_files`` keeps jobs for
    scanned files without recognized entries so coverage summaries can account
    for every in-scope file instead of silently dropping empty ones.
    """
    jobs = []
    if file_paths is None:
        file_paths = collect_files_to_process()
    for rel_path, file_path in file_paths:
        with open(file_path, 'rb') as handle:
            raw = handle.read()
        source_digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-8-sig')
        # Mirror the legacy text-mode read (universal newlines): CRLF/CR are
        # normalized to LF and only LF splits lines, so parsing is identical
        # for every consumer (build-revisions, final-review handoff, export).
        lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        entries = collect_translation_entries_from_lines(lines, file_rel_path=rel_path)

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

        if items or include_empty_files:
            jobs.append(
                {
                    'file_rel_path': rel_path,
                    'file_path': file_path,
                    'source_digest': source_digest,
                    'line_count': len(lines),
                    'task_count': len(items),
                    'items': items,
                }
            )
    return jobs


def run_revision_corpus_export(output_dir=None):
    """Export the read-only revision polishing corpus for the active project.

    Files are scanned once with the existing revision scanner; per-file source
    digests are computed before and after the scan so a corpus produced while
    sources changed is explicitly flagged instead of silently mixed.
    """
    file_paths = list(collect_files_to_process())
    file_path_map = {rel_path: file_path for rel_path, file_path in file_paths}
    digests_before = revision_corpus.collect_file_digests(file_path_map)
    file_jobs = collect_revision_file_jobs(
        file_paths=file_paths,
        include_empty_files=True,
    )
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
        os.makedirs(target_dir, exist_ok=True)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        target_dir = create_batch_package_dir(
            f'{stamp}_{guess_project_slug()}_revision_corpus'
        )
    digests_after = revision_corpus.collect_file_digests(file_path_map)
    scanned_digests = {
        job['file_rel_path']: job['source_digest'] for job in file_jobs
    }
    file_line_counts = {
        job['file_rel_path']: job.get('line_count') or 0 for job in file_jobs
    }
    manifest = revision_corpus.export_revision_corpus(
        target_dir,
        file_jobs,
        project_slug=guess_project_slug(),
        game_root=legacy.BASE_DIR,
        tl_dir=legacy.TL_DIR,
        tl_subdir=legacy.TL_SUBDIR,
        include_files=sorted(legacy.INCLUDE_FILES),
        include_prefixes=sorted(legacy.INCLUDE_PREFIXES),
        source_digests_before=digests_before,
        source_digests_after=digests_after,
        source_digests_scanned=scanned_digests,
        file_line_counts=file_line_counts,
    )
    print(f'Exported revision corpus: {target_dir}')
    print(
        f"Files: {manifest['scope']['file_count']}, "
        f"items: {manifest['scope']['item_count']}"
    )
    print(f"JSONL: {manifest['paths']['jsonl']}")
    print(f"Markdown: {manifest['paths']['markdown']}")
    if manifest.get('source', {}).get('source_changed_during_scan'):
        print(
            'Warning: source files changed during the scan; '
            'rerun export for a consistent snapshot.'
        )
    return manifest


def _versioning_artifact_component(value):
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value or '').strip())
    return normalized.strip('._-')[:80].strip('._-') or 'version'


def _create_versioning_output_dir(root_dir, name):
    os.makedirs(root_dir, exist_ok=True)
    return create_unique_child_dir(root_dir, name)


def run_project_snapshot_export(
    *,
    version_id,
    version_label='',
    source_revision='',
    output_dir=None,
    coverage_review_path=None,
):
    """Export one source-only P3 project snapshot for the active project."""
    adapter_snapshot = build_translation_snapshot(
        RenPyAdapter(legacy_module=legacy),
        ProjectDiscoveryRequest(
            project_root=legacy.BASE_DIR,
            localization_root=legacy.TL_DIR,
            target_language=legacy.PREP_LANGUAGE,
            include_files=tuple(sorted(legacy.INCLUDE_FILES)),
            include_prefixes=tuple(sorted(legacy.INCLUDE_PREFIXES)),
        ),
    )
    review_record = None
    if coverage_review_path and str(coverage_review_path).strip():
        review_record = load_review_record(str(coverage_review_path).strip())
    game_version = engine_versioning.GameVersion(
        version_id=str(version_id).strip(),
        label=str(version_label or ''),
        source_revision=str(source_revision or ''),
    )
    snapshot = engine_versioning.build_project_snapshot(
        adapter_snapshot,
        game_version,
        coverage_review=review_record,
    )
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
        os.makedirs(target_dir, exist_ok=True)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = (
            f'{stamp}_{guess_project_slug()}_'
            f'{_versioning_artifact_component(game_version.version_id)}'
        )
        target_dir = _create_versioning_output_dir(PROJECT_SNAPSHOTS_DIR, name)
    paths = engine_versioning.export_project_snapshot(snapshot, target_dir)
    result = {
        'kind': engine_versioning.PROJECT_SNAPSHOT_KIND,
        'snapshot_digest': snapshot.snapshot_digest,
        'version_id': snapshot.game_version.version_id,
        'engine': snapshot.engine,
        'occurrence_count': len(snapshot.occurrences),
        'coverage': snapshot.coverage.to_dict(),
        'paths': {
            'output_dir': paths.package_dir,
            'snapshot': paths.snapshot_path,
            'occurrences': paths.occurrences_path,
        },
    }
    print(f'Exported project snapshot: {paths.package_dir}')
    print(f'Game version: {snapshot.game_version.version_id}')
    print(f'Occurrences: {len(snapshot.occurrences)}')
    print(f'Snapshot digest: {snapshot.snapshot_digest}')
    print(f'Manifest: {paths.snapshot_path}')
    return result


def run_project_snapshot_reconciliation(base_path, target_path, *, output_dir=None):
    """Compare two saved snapshots without reading or writing live game files."""
    base = engine_versioning.load_project_snapshot(base_path)
    target = engine_versioning.load_project_snapshot(target_path)
    report = engine_versioning.reconcile_project_snapshots(base, target)
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
        os.makedirs(target_dir, exist_ok=True)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_part = _versioning_artifact_component(base.game_version.version_id)
        target_part = _versioning_artifact_component(target.game_version.version_id)
        target_dir = _create_versioning_output_dir(
            PROJECT_RECONCILIATIONS_DIR,
            f'{stamp}_{base_part}_to_{target_part}',
        )
    paths = engine_versioning.export_reconciliation_report(report, target_dir)
    result = {
        'kind': engine_versioning.RECONCILIATION_KIND,
        'status': report.status,
        'reconciliation_digest': report.reconciliation_digest,
        'base_version_id': report.base_version_id,
        'target_version_id': report.target_version_id,
        'summary': dict(report.summary),
        'coverage_changes': dict(report.coverage_changes),
        'paths': {
            'output_dir': paths.package_dir,
            'report': paths.report_path,
            'items': paths.items_path,
        },
    }
    print(f'Exported reconciliation report: {paths.package_dir}')
    print(f'Versions: {report.base_version_id} -> {report.target_version_id}')
    print(
        'Matches: '
        f"{report.summary.get('matched', 0)}, "
        f"ambiguous: {report.summary.get('ambiguous', 0)}, "
        f"added: {report.summary.get('added', 0)}, "
        f"deleted: {report.summary.get('deleted', 0)}"
    )
    print(f'Report: {paths.report_path}')
    return result


def run_translation_records_export(
    snapshot_path,
    manifest_target,
    *,
    origin='model_initial',
    previous_records_path='',
    output_dir=None,
):
    """Freeze one Batch package's validated translations as P4 records."""
    snapshot = engine_versioning.load_project_snapshot(snapshot_path)
    manifest = load_manifest(manifest_target)
    require_manifest_mode(
        manifest,
        MANIFEST_MODE_TRANSLATION,
        'build-translation-records',
    )
    _rows, rows_by_key, result_path = load_result_rows_by_key(
        manifest,
        'translation records source',
    )
    manifest_identity = {
        'manifest_path': _canonical_abs_path(manifest['_manifest_path']),
        'result_path': _canonical_abs_path(result_path),
        'project_identity': manifest_project_identity(manifest),
    }
    previous_records = None
    if previous_records_path and str(previous_records_path).strip():
        previous_records = engine_reuse.load_translation_records(
            str(previous_records_path).strip()
        )
        if previous_records.version_id != snapshot.game_version.version_id:
            raise SystemExit(
                'Previous translation records version does not match the '
                'snapshot version.'
            )
        if previous_records.snapshot_digest != snapshot.snapshot_digest:
            raise SystemExit(
                'Previous translation records do not match this snapshot '
                'digest; revision history cannot be chained across versions.'
            )
    previous_by_unit = {
        record.unit_id: record
        for record in (previous_records.records if previous_records else ())
    }
    inputs = []
    for chunk in manifest.get('chunks') or []:
        chunk_key = str(chunk.get('key') or '')
        row = rows_by_key.get(chunk_key)
        if row is None:
            raise SystemExit(f'Result JSONL is missing chunk: {chunk_key}')
        chunk_items = chunk.get('items') or []
        items = result_items_from_row(
            row,
            'translation records source',
            chunk_items,
        )
        items_by_id = {str(item.get('id') or ''): item for item in items}
        for unit in chunk_items:
            unit_id = str(unit.get('id') or '')
            item = items_by_id.get(unit_id)
            if item is None:
                raise SystemExit(
                    f'Chunk {chunk_key} result is missing unit: {unit_id}'
                )
            translation = str(item.get('translation') or '')
            if not translation.strip():
                raise SystemExit(f'Empty translation for unit: {unit_id}')
            previous_record = previous_by_unit.get(unit_id)
            revision_history = (
                engine_reuse.derive_revision_history(
                    previous_record,
                    new_translation=translation,
                    new_origin=origin,
                )
                if previous_record is not None
                else ()
            )
            inputs.append(
                engine_reuse.TranslationInput(
                    unit_id=unit_id,
                    translation_text=translation,
                    source_text=str(unit.get('source', unit.get('text', '')) or ''),
                    origin=origin,
                    revision_history=revision_history,
                    chunk_key=chunk_key,
                    row_key=str(row.get('key') or chunk_key),
                    extra={'manifest_identity': manifest_identity},
                )
            )
    record_set = engine_reuse.build_translation_records(snapshot, inputs)
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        version_part = _versioning_artifact_component(
            snapshot.game_version.version_id
        )
        target_dir = _create_versioning_output_dir(
            PROJECT_TRANSLATION_RECORDS_DIR,
            f'{stamp}_{guess_project_slug()}_{version_part}',
        )
    paths = engine_reuse.export_translation_records(record_set, target_dir)
    result = {
        'kind': engine_reuse.TRANSLATION_RECORDS_KIND,
        'version_id': record_set.version_id,
        'snapshot_digest': record_set.snapshot_digest,
        'record_count': len(record_set.records),
        'record_set_digest': record_set.record_set_digest,
        'target_language': record_set.target_language,
        'paths': {
            'output_dir': paths.package_dir,
            'manifest': paths.manifest_path,
            'records': paths.records_path,
        },
    }
    print(f'Exported translation records: {paths.package_dir}')
    print(f'Game version: {record_set.version_id}')
    print(f'Records: {len(record_set.records)}')
    print(f'Record set digest: {record_set.record_set_digest}')
    print(f'Manifest: {paths.manifest_path}')
    return result


def _load_reuse_live_inputs(
    *,
    base_snapshot_path='',
    target_snapshot_path='',
    reconciliation_path='',
    base_records_path='',
    recorded_paths=None,
):
    recorded = dict(recorded_paths or {})

    def _resolve(explicit, key, label):
        value = str(explicit or '').strip() or str(recorded.get(key) or '')
        if not value:
            raise SystemExit(
                f'Missing {label} path; pass it explicitly or rebuild the '
                'reuse package with recorded input paths.'
            )
        return value

    return {
        'base_snapshot': engine_versioning.load_project_snapshot(
            _resolve(base_snapshot_path, 'base_snapshot', 'base snapshot')
        ),
        'target_snapshot': engine_versioning.load_project_snapshot(
            _resolve(target_snapshot_path, 'target_snapshot', 'target snapshot')
        ),
        'reconciliation': engine_versioning.load_reconciliation_report(
            _resolve(reconciliation_path, 'reconciliation', 'reconciliation')
        ),
        'base_records': engine_reuse.load_translation_records(
            _resolve(base_records_path, 'base_records', 'base records')
        ),
    }


def run_reuse_candidates_build(
    base_snapshot_path,
    target_snapshot_path,
    reconciliation_path,
    base_records_path,
    *,
    output_dir=None,
):
    """Derive reviewable P4 reuse candidates from saved P3 artifacts."""
    base = engine_versioning.load_project_snapshot(base_snapshot_path)
    target = engine_versioning.load_project_snapshot(target_snapshot_path)
    reconciliation = engine_versioning.load_reconciliation_report(
        reconciliation_path
    )
    base_records = engine_reuse.load_translation_records(base_records_path)
    candidate_set = engine_reuse.build_reuse_candidates(
        reconciliation,
        base,
        target,
        base_records,
    )
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_part = _versioning_artifact_component(base.game_version.version_id)
        target_part = _versioning_artifact_component(target.game_version.version_id)
        target_dir = _create_versioning_output_dir(
            PROJECT_REUSE_DIR,
            f'{stamp}_{base_part}_to_{target_part}_candidates',
        )
    input_paths = {
        'base_snapshot': os.path.abspath(base_snapshot_path),
        'target_snapshot': os.path.abspath(target_snapshot_path),
        'reconciliation': os.path.abspath(reconciliation_path),
        'base_records': os.path.abspath(base_records_path),
    }
    paths = engine_reuse.export_reuse_candidates(
        candidate_set,
        target_dir,
        target_snapshot=target,
        input_paths=input_paths,
    )
    result = {
        'kind': engine_reuse.REUSE_CANDIDATES_KIND,
        'status': candidate_set.status,
        'candidate_count': len(candidate_set.candidates),
        'candidate_set_digest': candidate_set.candidate_set_digest,
        'reconciliation_digest': candidate_set.reconciliation_digest,
        'base_version_id': candidate_set.base_version_id,
        'target_version_id': candidate_set.target_version_id,
        'summary': dict(candidate_set.summary),
        'paths': {
            'output_dir': paths.package_dir,
            'report': paths.report_path,
            'candidates': paths.candidates_path,
            'review': paths.review_path,
            'decisions_template': paths.decisions_template_path,
        },
    }
    print(f'Exported reuse candidates: {paths.package_dir}')
    print(
        f"Versions: {candidate_set.base_version_id} -> "
        f"{candidate_set.target_version_id}"
    )
    print(
        'Candidates: '
        f"{candidate_set.summary.get('class_exact_reuse', 0)} exact, "
        f"{candidate_set.summary.get('class_moved_reuse', 0)} moved, "
        f"{candidate_set.summary.get('class_source_modified_reference', 0)} "
        'source-modified, '
        f"{candidate_set.summary.get('class_ambiguous', 0)} ambiguous"
    )
    print(f'Review sheet: {paths.review_path}')
    print(f'Decisions template: {paths.decisions_template_path}')
    return result


def run_reuse_decisions_import(
    reuse_path,
    decisions_path,
    *,
    base_snapshot_path='',
    target_snapshot_path='',
    reconciliation_path='',
    base_records_path='',
    output_dir=None,
):
    """Apply reviewer decisions and export an audited candidate package."""
    candidate_set = engine_reuse.load_reuse_candidates(reuse_path)
    decisions = engine_reuse.load_reuse_decisions(decisions_path)
    recorded_paths = {}
    reuse_report_path = Path(reuse_path)
    if reuse_report_path.is_dir():
        reuse_report_path = reuse_report_path / engine_reuse.DEFAULT_REUSE_REPORT_FILENAME
    if reuse_report_path.is_file():
        try:
            with open(reuse_report_path, 'r', encoding='utf-8') as handle:
                recorded_paths = dict(
                    json.loads(handle.read()).get('input_paths') or {}
                )
        except (OSError, json.JSONDecodeError):
            recorded_paths = {}
    live = _load_reuse_live_inputs(
        base_snapshot_path=base_snapshot_path,
        target_snapshot_path=target_snapshot_path,
        reconciliation_path=reconciliation_path,
        base_records_path=base_records_path,
        recorded_paths=recorded_paths,
    )
    updated = engine_reuse.apply_reuse_decisions(
        candidate_set,
        decisions,
        reconciliation=live['reconciliation'],
        base_snapshot=live['base_snapshot'],
        target_snapshot=live['target_snapshot'],
        base_records=live['base_records'],
    )
    if output_dir and str(output_dir).strip():
        target_dir = os.path.abspath(str(output_dir).strip())
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_part = _versioning_artifact_component(updated.base_version_id)
        target_part = _versioning_artifact_component(updated.target_version_id)
        target_dir = _create_versioning_output_dir(
            PROJECT_REUSE_DIR,
            f'{stamp}_{base_part}_to_{target_part}_decisions',
        )
    paths = engine_reuse.export_reuse_candidates(
        updated,
        target_dir,
        target_snapshot=live['target_snapshot'],
        input_paths=recorded_paths,
    )
    result = {
        'kind': engine_reuse.REUSE_CANDIDATES_KIND,
        'status': updated.status,
        'candidate_count': len(updated.candidates),
        'candidate_set_digest': updated.candidate_set_digest,
        'decisions_applied': len(decisions),
        'lineage_decisions': len(updated.lineage_decisions),
        'summary': dict(updated.summary),
        'paths': {
            'output_dir': paths.package_dir,
            'report': paths.report_path,
            'candidates': paths.candidates_path,
            'review': paths.review_path,
            'decisions_template': paths.decisions_template_path,
        },
    }
    print(f'Applied {len(decisions)} reuse decisions: {paths.package_dir}')
    print(
        'Accepted: '
        f"{updated.summary.get('status_accepted', 0)}, "
        f"rejected: {updated.summary.get('status_rejected', 0)}, "
        f"pending: {updated.summary.get('status_pending', 0)}"
    )
    print(f'Report: {paths.report_path}')
    return result


def _load_reuse_recorded_paths(reuse_path):
    report_path = Path(reuse_path)
    if report_path.is_dir():
        report_path = report_path / engine_reuse.DEFAULT_REUSE_REPORT_FILENAME
    if not report_path.is_file():
        return {}
    try:
        with open(report_path, 'r', encoding='utf-8') as handle:
            payload = json.loads(handle.read())
        paths = payload.get('input_paths')
        return dict(paths) if isinstance(paths, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def run_reuse_results_export(
    reuse_path,
    manifest_target,
    *,
    base_snapshot_path='',
    target_snapshot_path='',
    reconciliation_path='',
    base_records_path='',
):
    """Merge accepted, fresh reuse translations into a Batch package.

    The output is a canonical results JSONL plus manifest bookkeeping only;
    every game-file write still has to pass the existing ``check -> apply``
    gates, exactly like merged retry results.
    """
    candidate_set = engine_reuse.load_reuse_candidates(reuse_path)
    recorded_paths = _load_reuse_recorded_paths(reuse_path)
    live = _load_reuse_live_inputs(
        base_snapshot_path=base_snapshot_path,
        target_snapshot_path=target_snapshot_path,
        reconciliation_path=reconciliation_path,
        base_records_path=base_records_path,
        recorded_paths=recorded_paths,
    )
    prefill = engine_reuse.collect_reuse_prefill(
        candidate_set,
        reconciliation=live['reconciliation'],
        base_snapshot=live['base_snapshot'],
        target_snapshot=live['target_snapshot'],
        base_records=live['base_records'],
    )
    if not prefill:
        raise SystemExit(
            'No accepted direct-reuse translations to export. '
            'Accept candidates with import-reuse-decisions first.'
        )
    prefill_by_unit = {}
    for entry in prefill:
        prefill_by_unit[entry.target_unit_id] = entry

    manifest = load_manifest(manifest_target)
    require_manifest_mode(
        manifest,
        MANIFEST_MODE_TRANSLATION,
        'export-reuse-results',
    )
    parent_result_path = resolve_manifest_result_path(manifest)
    parent_rows_by_key = {}
    if os.path.isfile(parent_result_path):
        _rows, parent_rows_by_key, _path = load_result_rows_by_key(
            manifest,
            'parent results',
        )

    rows = []
    reused_count = 0
    parent_kept_count = 0
    used_units = set()
    missing_units = []
    for chunk in manifest.get('chunks') or []:
        chunk_key = str(chunk.get('key') or '')
        chunk_items = chunk.get('items') or []
        parent_row = parent_rows_by_key.get(chunk_key)
        parent_items_by_id = {}
        if parent_row is not None:
            parent_items = result_items_from_row(
                parent_row,
                'parent results',
                chunk_items,
                allow_empty=True,
            )
            parent_items_by_id = {
                str(item.get('id') or ''): item for item in parent_items
            }
        row_items = []
        chunk_entries = []
        for unit in chunk_items:
            unit_id = str(unit.get('id') or '')
            entry = prefill_by_unit.get(unit_id)
            if entry is not None:
                unit_source = str(unit.get('source', unit.get('text', '')) or '')
                if hash_text(unit_source) != hash_text(entry.source_text):
                    raise SystemExit(
                        'Batch item source no longer matches the reuse target '
                        f'snapshot: {unit_id}'
                    )
                row_items.append(
                    {'id': unit_id, 'translation': entry.translation_text}
                )
                chunk_entries.append(entry)
                used_units.add(unit_id)
                continue
            parent_item = parent_items_by_id.get(unit_id)
            if parent_item is not None and str(
                parent_item.get('translation') or ''
            ).strip():
                row_items.append(
                    {
                        'id': unit_id,
                        'translation': str(parent_item.get('translation') or ''),
                    }
                )
                continue
            missing_units.append(unit_id)
        if not chunk_entries:
            if parent_row is not None:
                rows.append(parent_row)
            elif missing_units:
                pass
            continue
        reused_count += len(chunk_entries)
        parent_kept_count += len(row_items) - len(chunk_entries)
        rows.append(
            canonical_translation_result_row(
                {
                    'key': chunk_key,
                    'normalized_response': {'translations': row_items},
                    'reuse_provenance': {
                        'candidate_ids': [
                            entry.candidate_id for entry in chunk_entries
                        ],
                        'candidate_digests': [
                            entry.candidate_digest for entry in chunk_entries
                        ],
                        'base_version_id': candidate_set.base_version_id,
                        'target_version_id': candidate_set.target_version_id,
                    },
                },
                chunk,
            )
        )
    if missing_units:
        preview = ', '.join(missing_units[:5])
        raise SystemExit(
            f'Batch package still has {len(missing_units)} uncovered units '
            f'(first: {preview}). Translate them through the normal flow or '
            'reject their reuse candidates before exporting.'
        )
    unused_prefill = sorted(set(prefill_by_unit) - used_units)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_name = f'results.reuse_{timestamp}.jsonl'
    merged_path = os.path.join(manifest['_package_dir'], merged_name)
    write_jsonl_file(merged_path, rows)
    had_parent_results = os.path.isfile(parent_result_path)
    manifest['result_jsonl_path'] = merged_name
    manifest.pop('result_jsonl_sha256', None)
    manifest['job_state'] = 'RESULTS_MERGED'
    manifest.setdefault('reuse_export_history', []).append(
        {
            'exported_at': datetime.now().isoformat(timespec='seconds'),
            'reuse_report': _canonical_abs_path(
                str(Path(reuse_path))
            ),
            'candidate_set_digest': candidate_set.candidate_set_digest,
            'base_version_id': candidate_set.base_version_id,
            'target_version_id': candidate_set.target_version_id,
            'previous_result_jsonl_path': (
                _canonical_abs_path(parent_result_path)
                if had_parent_results
                else ''
            ),
            'merged_result_jsonl_path': _canonical_abs_path(merged_path),
            'reused_items': reused_count,
            'parent_items_kept': parent_kept_count,
            'chunk_count': len(rows),
            'unused_prefill_units': len(unused_prefill),
        }
    )
    for key in ('last_check_at', 'last_check_summary', 'last_check_report_path'):
        manifest.pop(key, None)
    save_manifest(manifest, update_latest=True)

    result = {
        'kind': 'translation_reuse_results',
        'manifest_path': manifest['_manifest_path'],
        'result_jsonl_path': merged_path,
        'reused_items': reused_count,
        'parent_items_kept': parent_kept_count,
        'chunk_count': len(rows),
        'unused_prefill_units': len(unused_prefill),
        'candidate_set_digest': candidate_set.candidate_set_digest,
    }
    print(f'Reuse results exported: {merged_path}')
    print(f'Reused items: {reused_count}, parent items kept: {parent_kept_count}')
    if unused_prefill:
        print(
            f'Accepted reuse items not present in this package: '
            f'{len(unused_prefill)}'
        )
    print(f'Manifest: {manifest["_manifest_path"]}')
    print('Run check on this manifest before apply.')
    return result


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
            glossary_hits = retrieve_revision_glossary_hits(target_items)
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
    project_context = load_injectable_project_context_for_prompts(
        chunk.get('file_rel_path') or '',
        chunk.get('line_numbers') or [
            item.get('line_number') or (int(item.get('line') or 0) + 1)
            for item in (chunk.get('items') or [])
        ],
    )
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
            project_brief_text=project_context['text'],
            project_brief_diagnostics=project_context['diagnostics'],
            project_local_labels=project_context['labels'],
            project_local_routes=project_context['routes'],
            project_local_diagnostics=project_context['local_diagnostics'],
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


def build_revision_generation_config(target_items, model=None):
    effective_model = str(model or BATCH_MODEL or '')
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_revision_response_json_schema(target_items),
    }
    if BATCH_THINKING_LEVEL and is_gemini_3_model(effective_model):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return filter_gemini_generation_config(effective_model, config)


def build_revision_request(chunk, model=None):
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
            'generation_config': build_revision_generation_config(
                chunk['items'],
                model=model,
            ),
        },
    }


def create_revision_package(display_name_override='', skip_prepare=False, chunk_size=None):
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.GEMINI_BATCH,
        required_stages={model_profile.STAGE_REVISION},
    )
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
        **translation_quality.manifest_quality_policy_fields(runtime_policy=BATCH_QUALITY_POLICY),
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
            'max_context_chars': STORY_MEMORY_MAX_CONTEXT_CHARS,
            'top_k_relations': STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_terms': STORY_MEMORY_TOP_K_TERMS,
            'include_scene_summary': STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
        }
        manifest['story_memory_summary'] = summarize_batch_story_memory(chunks)

    attach_model_routing(
        manifest,
        routing_plan,
    )

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


def _proposal_import_report_markdown(report):
    lines = [
        '# Revision Proposal Import',
        '',
        f"- Status: {report.get('status') or 'unknown'}",
        f"- Input rows: {report.get('input_count', 0)}",
        f"- Requested selected rows: {report.get('requested_selected_count', 0)}",
        f"- Validated selected rows: {report.get('selected_count', 0)}",
        f"- Candidate rows: {report.get('candidate_count', 0)}",
        '',
    ]
    diagnostics = list(report.get('diagnostics') or [])
    if diagnostics:
        lines.extend(['## Diagnostics', ''])
        for item in diagnostics:
            lines.append(
                f"- `{item.get('code') or 'UNKNOWN'}` row {item.get('row') or '-'}: "
                f"{item.get('message') or ''}"
            )
    return '\n'.join(lines) + '\n'


def _write_proposal_import_report(package_dir, report):
    json_path = os.path.join(package_dir, 'proposal_import_report.json')
    markdown_path = os.path.join(package_dir, 'proposal_import_report.md')
    atomic_write_json(json_path, report, ensure_ascii=False, indent=2)
    atomic_write_text(markdown_path, _proposal_import_report_markdown(report))
    return {'import_report': json_path, 'import_report_markdown': markdown_path}


def _proposal_status_from_preview_summary(summary):
    """Map the latest revision preview summary to proposal eligibility state."""
    failures = int((summary or {}).get('failure_items') or 0)
    candidates = int((summary or {}).get('valid_items') or 0)
    unchanged = int((summary or {}).get('unchanged_items') or 0)
    if failures and candidates:
        return 'partial'
    if failures:
        return 'blocked'
    if candidates:
        return 'previewed'
    if unchanged:
        return 'no_op'
    return 'blocked'


def _collect_revision_proposal_live_context():
    """Collect the live revision scan and both source snapshot boundaries."""

    file_paths = list(collect_files_to_process())
    file_path_map = {rel_path: file_path for rel_path, file_path in file_paths}
    digests_before = revision_corpus.collect_file_digests(file_path_map)
    live_jobs = collect_revision_file_jobs(
        file_paths=file_paths,
        include_empty_files=True,
    )
    digests_after = revision_corpus.collect_file_digests(file_path_map)
    live_snapshot_digest = revision_corpus.aggregate_digest(digests_before)
    live_items = {
        str(item.get('id') or ''): item
        for job in live_jobs
        for item in (job.get('items') or [])
    }
    return {
        'file_paths': file_paths,
        'file_path_map': file_path_map,
        'digests_before': digests_before,
        'digests_after': digests_after,
        'live_snapshot_digest': live_snapshot_digest,
        'live_items': live_items,
        'live_jobs': live_jobs,
    }


def _revision_proposal_project_identity():
    return revision_selection.project_identity_from_paths(
        game_root=legacy.BASE_DIR,
        tl_dir=legacy.TL_DIR,
    )


def _proposal_import_diagnostics_from_candidates(candidates):
    diagnostics = []
    seen = set()
    for candidate in candidates or []:
        row = int(candidate.get('row') or 0)
        occurrence_id = str(candidate.get('identity_v2') or '')
        codes = list(candidate.get('diagnostic_codes') or [])
        messages = list(candidate.get('diagnostic_messages') or [])
        for index, code in enumerate(codes):
            key = (row, occurrence_id, str(code))
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append({
                'code': str(code),
                'message': str(messages[index] if index < len(messages) else ''),
                'row': row,
                'occurrence_id': occurrence_id,
            })
    return diagnostics


def _write_selection_stage_report(package_dir, report):
    json_path = os.path.join(package_dir, 'staged_selection_report.json')
    markdown_path = os.path.join(package_dir, 'staged_selection_report.md')
    atomic_write_json(json_path, report, ensure_ascii=False, indent=2)
    lines = [
        '# Revision Proposal Staged Selection',
        '',
        f"- Status: {report.get('status') or 'unknown'}",
        f"- Input rows: {report.get('input_count', 0)}",
        f"- Valid candidates: {report.get('valid_count', 0)}",
        f"- Initially selected: {report.get('selected_count', 0)}",
        f"- Unselected: {report.get('unselected_count', 0)}",
        f"- Invalid: {report.get('invalid_count', 0)}",
        f"- Stale: {report.get('stale_count', 0)}",
        f"- Conflict: {report.get('conflict_count', 0)}",
        f"- No-op: {report.get('no_op_count', 0)}",
        '',
    ]
    diagnostics = list(report.get('diagnostics') or [])
    if diagnostics:
        lines.extend(['## Diagnostics', ''])
        for item in diagnostics:
            lines.append(
                f"- `{item.get('code') or 'UNKNOWN'}` row {item.get('row') or '-'}: "
                f"{item.get('message') or ''}"
            )
    atomic_write_text(markdown_path, '\n'.join(lines) + '\n')
    return {
        'staged_selection_report': json_path,
        'staged_selection_report_markdown': markdown_path,
    }


def _write_selection_confirmation_report(package_dir, report):
    """Write a refusal/outcome report without mutating the immutable stage audit."""

    json_path = os.path.join(package_dir, 'selection_confirmation_report.json')
    markdown_path = os.path.join(package_dir, 'selection_confirmation_report.md')
    atomic_write_json(json_path, report, ensure_ascii=False, indent=2)
    lines = [
        '# Revision Proposal Selection Confirmation',
        '',
        f"- Status: {report.get('status') or 'unknown'}",
        f"- Staged selection: {report.get('staged_selection_path') or ''}",
        f"- Selection request: {report.get('selection_path') or ''}",
        '',
    ]
    diagnostics = list(report.get('diagnostics') or [])
    if diagnostics:
        lines.extend(['## Diagnostics', ''])
        for item in diagnostics:
            lines.append(
                f"- `{item.get('code') or 'UNKNOWN'}` row {item.get('row') or '-'}: "
                f"{item.get('message') or ''}"
            )
    atomic_write_text(markdown_path, '\n'.join(lines) + '\n')
    return {
        'selection_confirmation_report': json_path,
        'selection_confirmation_report_markdown': markdown_path,
    }


def _stage_revision_proposals(
    *,
    proposal_path,
    resolved_corpus_manifest,
    corpus_manifest,
    rows,
    live_context,
    operation_identity='',
):
    """Persist a candidate session without creating a revision preview."""

    package_dir = create_batch_package_dir(
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{guess_project_slug()}_revision_proposals_stage"
    )
    proposal_sha256 = _sha256_file(proposal_path)
    corpus_manifest_sha256 = (
        _sha256_file(resolved_corpus_manifest)
        if resolved_corpus_manifest and os.path.isfile(resolved_corpus_manifest)
        else ''
    )
    extra_diagnostics = []
    if live_context['digests_before'] != live_context['digests_after']:
        extra_diagnostics.append({
            'code': 'LIVE_SOURCE_CHANGED_DURING_IMPORT',
            'message': 'live project files changed while proposals were being imported',
            'row': 0,
            'occurrence_id': '',
        })
    project_identity = _revision_proposal_project_identity()
    stage = revision_selection.build_staged_selection(
        rows=rows,
        live_items=live_context['live_items'],
        live_snapshot_digest=live_context['live_snapshot_digest'],
        project_identity=project_identity,
        proposal_path=proposal_path,
        proposal_sha256=proposal_sha256,
        corpus_manifest_path=resolved_corpus_manifest,
        corpus_manifest_sha256=corpus_manifest_sha256,
        corpus_manifest=corpus_manifest,
        source_file_digests=live_context['digests_before'],
        operation_id=operation_identity,
        extra_diagnostics=extra_diagnostics,
    )
    stage_path = os.path.join(package_dir, 'staged_selection.json')
    revision_selection.write_staged_selection(stage_path, stage)
    summary = dict(stage.get('summary') or {})
    session = dict(stage.get('session') or {})
    session_status = str(session.get('session_status') or '').strip()
    if session_status not in {'ready', 'stale', 'no_valid_candidates'}:
        session_status = (
            'stale'
            if summary.get('session_stale')
            else 'ready'
            if summary.get('selectable_count')
            else 'no_valid_candidates'
        )
    diagnostics = _proposal_import_diagnostics_from_candidates(
        stage.get('candidates') or []
    )
    report = {
        'schema_version': revision_proposals.IMPORT_REPORT_SCHEMA_VERSION,
        'kind': 'revision_proposal_import',
        'status': 'staged',
        'session_status': session_status,
        'proposal_path': proposal_path,
        'corpus_manifest_path': resolved_corpus_manifest,
        'operation_identity': (stage.get('session') or {}).get('operation_identity') or '',
        'staged_selection_digest': stage.get('staged_selection_digest') or '',
        'live_snapshot_digest': live_context['live_snapshot_digest'],
        'input_count': len(rows),
        'requested_selected_count': sum(row.get('selected') is True for row in rows),
        'candidate_count': len(stage.get('candidates') or []),
        'valid_count': int(summary.get('valid_count') or 0),
        'selectable_count': int(summary.get('selectable_count') or 0),
        'selected_count': int(summary.get('selected_count') or 0),
        'unselected_count': int(summary.get('unselected_count') or 0),
        'no_op_count': int(summary.get('no_op_count') or 0),
        'invalid_count': int(summary.get('invalid_count') or 0),
        'stale_count': int(summary.get('stale_count') or 0),
        'conflict_count': int(summary.get('conflict_count') or 0),
        'diagnostics': diagnostics,
        'candidates': list(stage.get('candidates') or []),
        'suggested_action': (
            're_export_corpus_and_regenerate_proposals'
            if session_status == 'stale'
            else 'select_valid_candidates'
            if summary.get('selectable_count')
            else 'fix_proposal_diagnostics'
        ),
    }
    report_paths = _write_selection_stage_report(package_dir, report)
    return {
        **report,
        'stage': stage,
        'paths': {
            'output_dir': package_dir,
            'staged_selection': stage_path,
            **report_paths,
        },
    }


def _build_revision_proposal_preview_package(
    *,
    proposal_path,
    resolved_corpus_manifest,
    live_snapshot_digest,
    live_jobs,
    validation,
    package_dir,
    report,
    artifacts,
    proposal_state_extra=None,
):
    """Build and preview a standard revision package for confirmed candidates."""

    selected_by_identity = {
        str(row['identity_v2']): row for row in validation.proposals
    }
    filtered_jobs = []
    for job in live_jobs:
        items = [
            item for item in (job.get('items') or [])
            if str(item.get('id') or '') in selected_by_identity
        ]
        if items:
            filtered_jobs.append({**job, 'items': items, 'task_count': len(items)})
    chunks = build_revision_chunks(filtered_jobs, chunk_size=REVISION_CHUNK_SIZE)
    requests_path = os.path.join(package_dir, 'requests.jsonl')
    results_path = os.path.join(package_dir, 'results.jsonl')
    atomic_write_text(requests_path, '')
    result_rows = []
    for chunk in chunks:
        results = []
        for item in chunk['items']:
            proposal = selected_by_identity[str(item.get('id') or '')]
            proposed = str(proposal.get('proposed_translation') or '').strip()
            results.append({
                'id': item['id'],
                'should_update': compact_text(proposed) != compact_text(item.get('current_translation') or ''),
                'revised_translation': proposed,
                'reason': str(proposal.get('reason') or '').strip() or 'Imported revision proposal',
            })
        result_rows.append({
            'key': chunk['key'],
            'response': {'candidates': [{
                'content': {'parts': [{'text': json.dumps(results, ensure_ascii=False)}]},
                'finishReason': 'STOP',
            }]},
        })
    atomic_write_jsonl(results_path, result_rows, ensure_ascii=False)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    extra_state = dict(proposal_state_extra or {})
    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': MANIFEST_MODE_REVISION,
        'execution': 'proposal_import',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': f'{REVISION_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{stamp}',
        'batch_model': '',
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        **translation_quality.manifest_quality_policy_fields(
            runtime_policy=BATCH_QUALITY_POLICY
        ),
        'input_jsonl_path': requests_path,
        'result_jsonl_path': results_path,
        'job_name': '',
        'job_state': 'LOCAL_CANDIDATES',
        'submit_disabled': True,
        'settings': {'revision_chunk_size': REVISION_CHUNK_SIZE},
        'revision_settings': {
            'chunk_size': REVISION_CHUNK_SIZE,
            'candidate_source': 'revision_proposals',
        },
        'summary': {
            'file_count': len(filtered_jobs),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk['items']) for chunk in chunks),
            'proposal_count': len(validation.proposals),
        },
        'files': {
            job['file_rel_path']: {
                'path': job['file_path'],
                'task_count': job['task_count'],
            }
            for job in filtered_jobs
        },
        'chunks': chunks,
        'proposal_import': {
            'schema_version': revision_proposals.PROPOSAL_SCHEMA_VERSION,
            'status': 'imported',
            'history': ['imported'],
            'writeback_eligible': False,
            'proposal_path': proposal_path,
            'proposal_sha256': _sha256_file(proposal_path),
            'corpus_manifest_path': resolved_corpus_manifest,
            'corpus_snapshot_digest': live_snapshot_digest,
            'report_path': artifacts['import_report'],
            **extra_state,
        },
    }
    manifest_path = os.path.join(package_dir, 'manifest.json')
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    previewed = preview_revisions(manifest_path, update_latest=False)
    preview_summary = dict((previewed.get('last_revision_preview') or {}).get('summary') or {})
    final_status = _proposal_status_from_preview_summary(preview_summary)
    report.update({
        'status': final_status,
        'preview_summary': preview_summary,
        'suggested_action': (
            'run_apply_revisions' if final_status == 'previewed'
            else 'no_writeback_needed' if final_status == 'no_op'
            else 'fix_preview_diagnostics_and_reimport'
        ),
    })
    _write_proposal_import_report(package_dir, report)
    save_manifest(
        previewed,
        update_latest=final_status in {'previewed', 'no_op'},
    )
    return {
        **report,
        'manifest': previewed,
        'paths': {
            'output_dir': package_dir,
            'manifest': manifest_path,
            'revision_preview_jsonl': previewed['last_revision_preview']['jsonl_path'],
            'revision_preview_markdown': previewed['last_revision_preview']['markdown_path'],
            **artifacts,
        },
    }


def import_revision_proposals(
    proposal_path,
    *,
    corpus_manifest_path='',
    stage=False,
    operation_identity='',
):
    """Import structured proposals into the existing revision preview gate.

    The command is local-only and never writes ``.rpy``.  Invalid structural or
    stale input produces an auditable report but no revision manifest.  Valid
    candidates are encoded as ordinary revision results and immediately
    previewed by ``preview_revisions``.
    """
    proposal_path = os.path.abspath(str(proposal_path or '').strip())
    if not proposal_path or not os.path.isfile(proposal_path):
        raise cli_contract.MachineContractError(
            f'Proposal JSONL not found: {proposal_path or "(missing)"}',
            code_name='PROPOSAL_FILE_NOT_FOUND',
            suggested_action='pass_existing_proposal_jsonl',
            semantic_exit_code=cli_contract.EXIT_USAGE,
        )
    try:
        rows = revision_proposals.load_jsonl(proposal_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise cli_contract.MachineContractError(
            f'Invalid proposal JSONL: {exc}',
            code_name='PROPOSAL_JSONL_INVALID',
            suggested_action='fix_proposal_jsonl',
            semantic_exit_code=cli_contract.EXIT_INVALID_STATE,
        ) from exc
    if not rows:
        raise cli_contract.MachineContractError(
            'Proposal JSONL contains no proposal rows.',
            code_name='NO_PROPOSAL_ROWS',
            suggested_action='provide_non_empty_proposal_jsonl',
            semantic_exit_code=cli_contract.EXIT_USAGE,
        )
    resolved_corpus_manifest = revision_proposals.find_corpus_manifest(
        proposal_path,
        corpus_manifest_path,
    )
    corpus_manifest = None
    if resolved_corpus_manifest:
        try:
            corpus_manifest = revision_proposals.load_corpus_manifest(
                resolved_corpus_manifest
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise cli_contract.MachineContractError(
                f'Invalid revision corpus manifest: {exc}',
                code_name='CORPUS_MANIFEST_INVALID',
                suggested_action='pass_matching_revision_corpus_manifest',
                semantic_exit_code=cli_contract.EXIT_INVALID_STATE,
            ) from exc

    live_context = _collect_revision_proposal_live_context()
    live_jobs = live_context['live_jobs']
    live_items = live_context['live_items']
    live_snapshot_digest = live_context['live_snapshot_digest']
    validation = revision_proposals.validate(
        rows,
        live_items,
        live_snapshot_digest=live_snapshot_digest,
        live_project_identity=_revision_proposal_project_identity(),
        corpus_manifest=corpus_manifest,
    )
    diagnostics = [dict(item) for item in validation.diagnostics]
    if live_context['digests_before'] != live_context['digests_after']:
        diagnostics.append({
            'code': 'LIVE_SOURCE_CHANGED_DURING_IMPORT',
            'message': 'live project files changed while proposals were being imported',
            'row': 0,
            'occurrence_id': '',
        })
    if stage:
        return _stage_revision_proposals(
            proposal_path=proposal_path,
            resolved_corpus_manifest=resolved_corpus_manifest,
            corpus_manifest=corpus_manifest,
            rows=rows,
            live_context=live_context,
            operation_identity=operation_identity,
        )
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_dir = create_batch_package_dir(
        f'{stamp}_{guess_project_slug()}_revision_proposals'
    )
    status = validation.status
    if revision_proposals.diagnostics_are_stale(diagnostics):
        status = 'stale'
    elif diagnostics:
        status = 'blocked'
    report = {
        'schema_version': revision_proposals.IMPORT_REPORT_SCHEMA_VERSION,
        'kind': 'revision_proposal_import',
        'status': status,
        'proposal_path': proposal_path,
        'corpus_manifest_path': resolved_corpus_manifest,
        'live_snapshot_digest': live_snapshot_digest,
        'input_count': validation.input_count,
        'requested_selected_count': validation.requested_selected_count,
        'selected_count': validation.selected_count,
        'candidate_count': len(validation.proposals),
        'diagnostics': diagnostics,
        'suggested_action': (
            're_export_corpus_and_regenerate_proposals'
            if status == 'stale'
            else 'fix_proposal_diagnostics'
            if status == 'blocked'
            else 'no_writeback_needed'
            if status == 'no_op'
            else 'inspect_revision_preview'
        ),
    }
    artifacts = _write_proposal_import_report(package_dir, report)
    if diagnostics or not validation.proposals:
        print(f'Revision proposal import status: {status}')
        print(f'Import report: {artifacts["import_report"]}')
        return {**report, 'paths': {'output_dir': package_dir, **artifacts}}

    result = _build_revision_proposal_preview_package(
        proposal_path=proposal_path,
        resolved_corpus_manifest=resolved_corpus_manifest,
        live_snapshot_digest=live_snapshot_digest,
        live_jobs=live_jobs,
        validation=validation,
        package_dir=package_dir,
        report=report,
        artifacts=artifacts,
    )
    print(f"Revision proposal import status: {result['status']}")
    print(f"Manifest: {result['paths']['manifest']}")
    return result


def _staged_selection_confirmation_outcome(
    *,
    stage_path,
    selection_path,
    status,
    suggested_action,
    diagnostics,
):
    """Return a machine-contract-friendly confirmation refusal."""

    package_dir = create_batch_package_dir(
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{guess_project_slug()}_revision_proposals_selection_confirmation"
    )
    report = {
        'schema_version': revision_proposals.IMPORT_REPORT_SCHEMA_VERSION,
        'kind': 'revision_proposal_selection_confirmation',
        'status': status,
        'output_dir': package_dir,
        'staged_selection_path': os.path.abspath(stage_path),
        'selection_path': os.path.abspath(selection_path),
        'diagnostics': [dict(item) for item in diagnostics or []],
        'suggested_action': suggested_action,
    }
    report_paths = _write_selection_confirmation_report(package_dir, report)
    return {
        **report,
        'paths': {
            'output_dir': package_dir,
            'staged_selection': os.path.abspath(stage_path),
            'selection': os.path.abspath(selection_path),
            **report_paths,
        },
    }


def _same_revision_project_identity(expected, actual):
    return revision_selection.project_identities_match(expected, actual)


def confirm_revision_proposals(staged_selection_path, selection_path):
    """Confirm a staged selection and hand it to the existing preview gate."""

    stage_path = os.path.abspath(str(staged_selection_path or '').strip())
    request_path = os.path.abspath(str(selection_path or '').strip())
    try:
        stage = revision_selection.load_staged_selection(stage_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise cli_contract.MachineContractError(
            f'Invalid staged selection: {exc}',
            code_name='STAGED_SELECTION_INVALID',
            suggested_action='reimport_revision_proposals_for_new_stage',
            semantic_exit_code=cli_contract.EXIT_INVALID_STATE,
        ) from exc
    try:
        request = revision_selection.load_selection_request(request_path)
        selected_ids = revision_selection.validate_selection_request(stage, request)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise cli_contract.MachineContractError(
            f'Invalid revision proposal selection: {exc}',
            code_name='SELECTION_REQUEST_INVALID',
            suggested_action='confirm_current_staged_candidates',
            semantic_exit_code=cli_contract.EXIT_INVALID_STATE,
        ) from exc

    session = dict(stage.get('session') or {})
    proposal_path = os.path.abspath(str(session.get('proposal_path') or ''))
    resolved_corpus_manifest = os.path.abspath(
        str(session.get('corpus_manifest_path') or '')
    ) if str(session.get('corpus_manifest_path') or '').strip() else ''
    diagnostics = []
    if not proposal_path or not os.path.isfile(proposal_path):
        diagnostics.append({
            'code': 'PROPOSAL_FILE_STALE',
            'message': 'the original proposal JSONL is missing',
            'row': 0,
            'occurrence_id': '',
        })
    elif _sha256_file(proposal_path) != str(session.get('proposal_sha256') or ''):
        diagnostics.append({
            'code': 'PROPOSAL_FILE_STALE',
            'message': 'the original proposal JSONL changed after staging',
            'row': 0,
            'occurrence_id': '',
        })
    if resolved_corpus_manifest:
        if not os.path.isfile(resolved_corpus_manifest):
            diagnostics.append({
                'code': 'CORPUS_MANIFEST_STALE',
                'message': 'the companion corpus manifest is missing',
                'row': 0,
                'occurrence_id': '',
            })
        elif _sha256_file(resolved_corpus_manifest) != str(
            session.get('corpus_manifest_sha256') or ''
        ):
            diagnostics.append({
                'code': 'CORPUS_MANIFEST_STALE',
                'message': 'the companion corpus manifest changed after staging',
                'row': 0,
                'occurrence_id': '',
            })
    current_project = _revision_proposal_project_identity()
    if not _same_revision_project_identity(
        session.get('project_identity'),
        current_project,
    ):
        diagnostics.append({
            'code': 'PROJECT_IDENTITY_STALE',
            'message': 'the current project does not match the staged selection',
            'row': 0,
            'occurrence_id': '',
        })

    live_context = _collect_revision_proposal_live_context()
    if live_context['digests_before'] != live_context['digests_after']:
        diagnostics.append({
            'code': 'LIVE_SOURCE_CHANGED_DURING_IMPORT',
            'message': 'live project files changed while the selection was being confirmed',
            'row': 0,
            'occurrence_id': '',
        })
    if dict(session.get('source_file_digests') or {}) != live_context['digests_before']:
        diagnostics.append({
            'code': 'SOURCE_SNAPSHOT_STALE',
            'message': 'source files changed after the staged selection was created',
            'row': 0,
            'occurrence_id': '',
        })
    if str(session.get('live_snapshot_digest') or '') != live_context['live_snapshot_digest']:
        diagnostics.append({
            'code': 'CORPUS_SNAPSHOT_STALE',
            'message': 'the live project snapshot does not match the staged selection',
            'row': 0,
            'occurrence_id': '',
        })
    if diagnostics:
        return _staged_selection_confirmation_outcome(
            stage_path=stage_path,
            selection_path=request_path,
            status='stale',
            suggested_action='reimport_revision_proposals_for_new_stage',
            diagnostics=diagnostics,
        )

    corpus_manifest = None
    if resolved_corpus_manifest:
        try:
            corpus_manifest = revision_proposals.load_corpus_manifest(
                resolved_corpus_manifest
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise cli_contract.MachineContractError(
                f'Invalid revision corpus manifest: {exc}',
                code_name='CORPUS_MANIFEST_INVALID',
                suggested_action='pass_matching_revision_corpus_manifest',
                semantic_exit_code=cli_contract.EXIT_INVALID_STATE,
            ) from exc

    candidate_map = {
        str(candidate.get('identity_v2') or ''): candidate
        for candidate in stage.get('candidates') or []
    }
    selected_rows = []
    for identity in selected_ids:
        candidate = candidate_map[identity]
        row = dict(candidate.get('proposal') or {})
        row['_row_number'] = int(candidate.get('row') or 0)
        row['selected'] = True
        disposition = str(row.get('disposition') or '').strip().lower()
        if disposition not in revision_proposals.SELECTED_DISPOSITIONS:
            row['disposition'] = 'selected'
        selected_rows.append(row)

    if not selected_rows:
        return _staged_selection_confirmation_outcome(
            stage_path=stage_path,
            selection_path=request_path,
            status='no_op',
            suggested_action='select_valid_candidates',
            diagnostics=[],
        )

    validation = revision_proposals.validate(
        selected_rows,
        live_context['live_items'],
        live_snapshot_digest=live_context['live_snapshot_digest'],
        live_project_identity=current_project,
        corpus_manifest=corpus_manifest,
    )
    if validation.diagnostics or not validation.proposals:
        return _staged_selection_confirmation_outcome(
            stage_path=stage_path,
            selection_path=request_path,
            status=(
                'stale'
                if revision_proposals.diagnostics_are_stale(validation.diagnostics)
                else 'blocked'
            ),
            suggested_action='reimport_revision_proposals_for_new_stage',
            diagnostics=validation.diagnostics,
        )

    package_dir = create_batch_package_dir(
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{guess_project_slug()}_revision_proposals_confirmed"
    )
    report = {
        'schema_version': revision_proposals.IMPORT_REPORT_SCHEMA_VERSION,
        'kind': 'revision_proposal_import',
        'status': 'confirmed',
        'proposal_path': proposal_path,
        'corpus_manifest_path': resolved_corpus_manifest,
        'live_snapshot_digest': live_context['live_snapshot_digest'],
        'input_count': len(stage.get('candidates') or []),
        'requested_selected_count': len(selected_ids),
        'selected_count': len(validation.proposals),
        'candidate_count': len(stage.get('candidates') or []),
        'selected_identity_v2': selected_ids,
        'staged_selection_digest': stage.get('staged_selection_digest') or '',
        'selection_digest': request.get('selection_digest') or '',
        'operation_identity': session.get('operation_identity') or '',
        'diagnostics': [],
        'suggested_action': 'inspect_revision_preview',
    }
    artifacts = _write_proposal_import_report(package_dir, report)
    result = _build_revision_proposal_preview_package(
        proposal_path=proposal_path,
        resolved_corpus_manifest=resolved_corpus_manifest,
        live_snapshot_digest=live_context['live_snapshot_digest'],
        live_jobs=live_context['live_jobs'],
        validation=validation,
        package_dir=package_dir,
        report=report,
        artifacts=artifacts,
        proposal_state_extra={
            'operation_identity': session.get('operation_identity') or '',
            'staged_selection_path': stage_path,
            'staged_selection_digest': stage.get('staged_selection_digest') or '',
            'selection_path': request_path,
            'selection_digest': request.get('selection_digest') or '',
            'selection_sha256': _sha256_file(request_path),
            'selected_identity_v2': selected_ids,
        },
    )
    print(f"Revision proposal selection status: {result['status']}")
    print(f"Manifest: {result['paths']['manifest']}")
    return result


def _flatten_revision_items(file_jobs):
    items = []
    for job in file_jobs or []:
        for item in job.get('items') or []:
            current = dict(item)
            current.setdefault('file_rel_path', job.get('file_rel_path', ''))
            current.setdefault('file_path', job.get('file_path', ''))
            items.append(current)
    return items


def _pending_file_rows_for_final_review(file_jobs):
    rows = []
    for job in file_jobs or []:
        count = int(job.get('task_count') or 0)
        if count <= 0:
            continue
        rows.append(
            {
                'file_rel_path': job.get('file_rel_path') or '',
                'pending_task_count': count,
            }
        )
    return rows


def _load_final_review_glossary_terms(glossary_path, limit=200):
    """Lightweight glossary term list for final-review prompt injection."""
    path = str(glossary_path or '').strip()
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8-sig') as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    pairs = []
    seen = set()
    normalize_map = data.get('normalize_map') or {}
    if isinstance(normalize_map, dict):
        for source, target in normalize_map.items():
            key = str(source or '').strip()
            if not key or key in seen:
                continue
            pairs.append({'source': key, 'target': str(target or key).strip() or key})
            seen.add(key)
            if len(pairs) >= limit:
                return pairs
    for term in data.get('preserve_terms') or []:
        key = str(term or '').strip()
        if not key or key in seen:
            continue
        pairs.append({'source': key, 'target': key})
        seen.add(key)
        if len(pairs) >= limit:
            break
    return pairs


def collect_project_analysis_optional_inputs(store_dir=None, base_dir=None):
    """Collect bounded, non-blocking context layers with explicit provenance."""
    from project_analysis import resolve_project_analysis_store

    store = resolve_project_analysis_store(store_dir, base_dir=base_dir)
    labels = store.load_summaries("label")
    query_text = truncate_text(
        "\n".join(str(record.get("summary") or "") for record in labels if record.get("summary")),
        4000,
    )
    first_file = next(
        (
            str(source_file)
            for record in labels
            for source_file in (record.get("source_files") or [])
            if str(source_file or "").strip()
        ),
        "",
    )
    inputs = {}

    glossary_path = getattr(legacy, "GLOSSARY_FILE", "") or ""
    glossary_terms = _load_final_review_glossary_terms(glossary_path, limit=120)
    if glossary_terms:
        inputs["glossary"] = {
            "content": prompt_context.format_glossary_hits_block(glossary_terms),
            "provenance": {
                "kind": "glossary",
                "artifact": os.path.basename(glossary_path),
                "term_count": len(glossary_terms),
            },
        }

    macro_text = truncate_text(BATCH_MACRO_SETTING, 4000).strip()
    if macro_text:
        inputs["macro_setting"] = {
            "content": macro_text,
            "provenance": {
                "kind": "macro_setting",
                "artifact": "batch.macro_setting",
                "char_count": len(macro_text),
            },
        }

    if SOURCE_INDEX_ENABLED and query_text:
        source_hits, source_stats = retrieve_source_hits([{"text": query_text}], [])
        if source_hits:
            inputs["source_index"] = {
                "content": prompt_context.format_source_hits_block(source_hits),
                "provenance": {
                    "kind": "source_index",
                    "store": os.path.basename(str(source_stats.get("store_dir") or SOURCE_INDEX_STORE_DIR)),
                    "hit_count": len(source_hits),
                    "source_ids": [str(hit.get("source_id") or "") for hit in source_hits],
                },
            }

    if STORY_MEMORY_ENABLED and query_text:
        try:
            story_hits = retrieve_batch_story_hits(
                first_file,
                [{"text": query_text}],
                [],
                [],
            )
            if story_memory.has_story_hits(story_hits):
                story_text = story_memory.format_story_hits_block(
                    story_hits,
                    min(4000, STORY_MEMORY_MAX_CONTEXT_CHARS),
                )
                inputs["story_memory"] = {
                    "content": story_text,
                    "provenance": {
                        "kind": "story_memory",
                        "artifact": os.path.basename(
                            STORY_MEMORY_GRAPH_FILE or "story_graph.json"
                        ),
                        "query_file": first_file,
                    },
                }
        except Exception as exc:
            print(
                f"Warning: optional Story Memory input unavailable: {exc}",
                file=sys.stderr,
            )
    return inputs

def _collect_final_review_context_snapshot(translation_items):
    """Build the frozen context snapshot for a final-review campaign."""
    import final_review as fr

    include_filters = []
    raw_include = getattr(legacy, 'INCLUDE_FILTERS', None) or getattr(
        legacy, 'FILE_INCLUDE_FILTERS', None
    )
    if isinstance(raw_include, (list, tuple)):
        include_filters = [str(x) for x in raw_include if str(x).strip()]

    pa_status = ''
    pa_fp = ''
    pa_version = None
    pa_brief_text = ''
    pa_lineage = None
    if PROJECT_ANALYSIS_ENABLED and PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF:
        try:
            from project_analysis import (
                collect_project_analysis_status,
                load_injectable_project_brief,
            )

            pa_fp = compute_current_project_analysis_fingerprint(
                legacy.BASE_DIR, store_dir=PROJECT_ANALYSIS_STORE_DIR or None
            )
            status_payload = collect_project_analysis_status(
                store_dir=PROJECT_ANALYSIS_STORE_DIR or None,
                base_dir=legacy.BASE_DIR or None,
                expected_source_fingerprint=pa_fp or '',
            )
            pa_status = str(status_payload.get('brief_status') or '')
            # Prefer lineage/version from the brief artifact — not bare schema_version.
            brief_entry = (status_payload.get('artifacts') or {}).get('project_brief') or {}
            pa_lineage = brief_entry.get('lineage') if isinstance(brief_entry, dict) else None
            if isinstance(pa_lineage, dict):
                pa_version = (
                    pa_lineage.get('generated_at')
                    or pa_lineage.get('prompt_schema_version')
                    or status_payload.get('schema_version')
                )
            else:
                pa_version = status_payload.get('schema_version')
            # Hash the *actual* injectable text (already max_chars truncated).
            brief_payload = load_injectable_project_brief(
                store_dir=PROJECT_ANALYSIS_STORE_DIR or None,
                base_dir=legacy.BASE_DIR or None,
                expected_source_fingerprint=pa_fp or '',
                max_chars=PROJECT_ANALYSIS_MAX_BRIEF_CHARS,
                enabled=True,
            )
            if brief_payload.get('injectable') and brief_payload.get('text'):
                pa_brief_text = str(brief_payload.get('text') or '')
            else:
                # Not injectable → do not pin fingerprint/brief into the digest.
                pa_fp = ''
                pa_brief_text = ''
                pa_lineage = None
        except Exception:
            pa_status = 'error'
            pa_fp = ''
            pa_brief_text = ''
            pa_lineage = None

    macro_path = ''
    for candidate in (
        getattr(legacy, 'MACRO_SETTING_FILE', ''),
        getattr(legacy, 'MACRO_SETTING_PATH', ''),
    ):
        if candidate and os.path.isfile(str(candidate)):
            macro_path = str(candidate)
            break

    source_index_path = ''
    if SOURCE_INDEX_ENABLED:
        source_index_path = SOURCE_INDEX_STORE_DIR or get_default_source_index_store_dir()

    glossary_path = getattr(legacy, 'GLOSSARY_FILE', '') or ''
    macro_text = BATCH_MACRO_SETTING or ''
    snapshot = fr.build_context_snapshot(
        translation_items=translation_items,
        glossary_path=glossary_path,
        glossary_enabled=True,
        macro_setting_text=macro_text,
        macro_setting_path=macro_path or None,
        story_memory_enabled=bool(STORY_MEMORY_ENABLED),
        story_memory_graph_path=STORY_MEMORY_GRAPH_FILE or None,
        source_index_enabled=bool(SOURCE_INDEX_ENABLED),
        source_index_store_path=source_index_path or None,
        project_analysis_enabled=bool(PROJECT_ANALYSIS_ENABLED),
        project_analysis_inject=bool(PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF),
        project_analysis_status=pa_status,
        project_analysis_fingerprint=pa_fp,
        project_analysis_version=pa_version,
        project_analysis_store_path=PROJECT_ANALYSIS_STORE_DIR or None,
        project_analysis_brief_text=pa_brief_text,
        project_analysis_lineage=pa_lineage,
        include_filters=include_filters,
        base_dir=legacy.BASE_DIR or '',
        tl_dir=legacy.TL_DIR or '',
    )
    # Frozen text actually injected into review prompts (digest-aligned).
    snapshot['prompt_context'] = {
        'macro_setting': str(macro_text)[:8000],
        'project_analysis_brief': str(pa_brief_text or '')[:8000],
        'glossary_terms': _load_final_review_glossary_terms(glossary_path),
    }
    return snapshot


def create_final_review_package(
    display_name_override='',
    skip_prepare=False,
    chunk_size=None,
    allow_pending=False,
    require_zero_pending=None,
):
    """Build a report-only final-review campaign package (no LLM, no .rpy writes)."""
    import final_review as fr

    if not FINAL_REVIEW_ENABLED:
        raise SystemExit(
            'Final review is disabled in config (batch.final_review.enabled=false).'
        )

    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.GEMINI_BATCH,
        required_stages={model_profile.STAGE_FINAL_REVIEW},
    )

    if not skip_prepare:
        legacy.run_prepare_steps()
    if not os.path.isdir(legacy.TL_DIR):
        raise SystemExit(f'TL dir does not exist: {legacy.TL_DIR}')

    pending_jobs = collect_pending_file_jobs(include_complete_files=False)
    pending_progress = summarize_translation_progress(pending_jobs)
    pending_count = int(pending_progress.get('pending_task_count') or 0)

    file_jobs = collect_revision_file_jobs()
    translation_items = _flatten_revision_items(file_jobs)

    enforce_zero = (
        FINAL_REVIEW_REQUIRE_ZERO_PENDING
        if require_zero_pending is None
        else bool(require_zero_pending)
    )
    readiness = fr.evaluate_readiness(
        pending_task_count=pending_count,
        pending_files=_pending_file_rows_for_final_review(pending_jobs),
        review_item_count=len(translation_items),
        require_zero_pending=enforce_zero,
        allow_pending=bool(allow_pending),
    )
    try:
        fr.require_readiness(readiness)
    except fr.FinalReviewReadinessError as exc:
        print('Final review readiness check failed:')
        for reason in exc.reasons:
            print(f'- {reason}')
        raise SystemExit(1) from exc

    chunk_size = max(1, int(chunk_size or FINAL_REVIEW_CHUNK_SIZE or fr.DEFAULT_CHUNK_SIZE))
    model = FINAL_REVIEW_MODEL or BATCH_MODEL or ''
    prompt_schema = FINAL_REVIEW_PROMPT_SCHEMA_VERSION or fr.PROMPT_SCHEMA_VERSION

    snapshot = _collect_final_review_context_snapshot(translation_items)
    units = fr.build_review_units(
        translation_items,
        chunk_size=chunk_size,
        context_digest=snapshot.get('context_digest') or '',
        snapshot_digest=snapshot.get('snapshot_digest') or '',
        model=model,
        prompt_schema_version=prompt_schema,
    )
    if not units:
        print('No final-review units built (no translated items in scope).')
        return None

    package_name = fr.suggest_package_name(guess_project_slug())
    package_dir = create_batch_package_dir(package_name)

    display_name = display_name_override.strip() if display_name_override else ''
    if not display_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        display_name = f'{FINAL_REVIEW_DISPLAY_NAME_PREFIX}-{guess_project_slug()}-{timestamp}'

    import final_review_llm as fr_llm

    requests_path = os.path.join(package_dir, fr.REQUESTS_JSONL_FILENAME)
    request_count = fr_llm.write_requests_jsonl(
        requests_path,
        units,
        temperature=BATCH_TEMPERATURE,
        max_output_tokens=BATCH_MAX_OUTPUT_TOKENS,
        thinking_level=BATCH_THINKING_LEVEL,
        model=model,
        safety_settings=BATCH_SAFETY_SETTINGS or None,
        shared_context=snapshot.get('prompt_context') or {},
    )

    manifest = fr.build_campaign_manifest(
        package_dir=package_dir,
        display_name=display_name,
        snapshot=snapshot,
        units=units,
        readiness=readiness,
        base_dir=legacy.BASE_DIR or '',
        tl_dir=legacy.TL_DIR or '',
        model=model,
        prompt_schema_version=prompt_schema,
        chunk_size=chunk_size,
        batch_model=BATCH_MODEL,
        settings={
            'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
            'temperature': BATCH_TEMPERATURE,
            'thinking_level': BATCH_THINKING_LEVEL,
        },
        extra={
            **_manifest_target_language_fields(),
            'build_warnings': get_batch_risk_warnings(),
            'input_jsonl_path': requests_path,
            'request_count': request_count,
            'model_routing': routing_plan.to_manifest_dict(),
        },
    )
    # batch_cost_estimate uses summary.chunk_count for max output tokens.
    manifest.setdefault('summary', {})
    manifest['summary']['chunk_count'] = request_count
    manifest['summary']['request_count'] = request_count
    paths = fr.write_campaign_package(
        package_dir,
        manifest=manifest,
        snapshot=snapshot,
        units=units,
        findings=[],
        write_report=True,
    )
    remember_latest_manifest(paths['manifest'])

    print(f'Created final-review campaign: {package_dir}')
    print(f"Units: {manifest['summary']['unit_count']}")
    print(f"Items: {manifest['summary']['item_count']}")
    print(f'Requests: {request_count} → {requests_path}')
    print(f"Context digest: {str(snapshot.get('context_digest') or '')[:16]}…")
    print(f"Snapshot digest: {str(snapshot.get('snapshot_digest') or '')[:16]}…")
    print('Mode: final_review (report-only; no autofix)')
    print(
        'Next: submit → status → download → final-review-ingest-results '
        '(or final-review-resume after partial completion)'
    )
    if readiness.reasons:
        print('Notes:')
        for reason in readiness.reasons:
            print(f'- {reason}')
    return paths['manifest']


def run_final_review_status(target=None, as_json=False):
    import final_review as fr

    package_target = manifest_path_for_target(target)
    try:
        status = fr.collect_campaign_status(package_target)
    except fr.FinalReviewError as exc:
        raise SystemExit(f'Final review status error: {exc}') from exc
    if as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(fr.format_status_text(status))
    return status


def run_final_review_export(target=None, output_jsonl='', output_markdown=''):
    import final_review as fr

    package_target = manifest_path_for_target(target)
    try:
        result = fr.export_findings(
            package_target,
            output_jsonl=output_jsonl,
            output_markdown=output_markdown,
        )
    except fr.FinalReviewError as exc:
        raise SystemExit(f'Final review export error: {exc}') from exc
    print(f"Findings JSONL: {result['jsonl_path']}")
    print(f"Report Markdown: {result['markdown_path']}")
    print(f"Finding count: {result['finding_count']}")
    print(f"Campaign status: {result['status'].get('status')}")
    return result


def run_final_review_resume(target=None, force=False):
    """Rebuild requests for pending/stale/failed units; skip unchanged done units.

    Recomputes live shared context (glossary/macro/PA…) so digest-based skip /
    stale detection matches the current workspace, not only the frozen build
    snapshot. Request prompts also inject the live prompt_context block.
    """
    import final_review as fr
    import final_review_llm as fr_llm

    package_target = manifest_path_for_target(target)
    package = fr.load_campaign_package(package_target)
    package_dir = package['paths']['package_dir']
    frozen_snapshot = dict(package.get('snapshot') or {})
    frozen_context = str(
        frozen_snapshot.get('context_digest') or package['manifest'].get('context_digest') or ''
    )
    frozen_prompt_context = dict(frozen_snapshot.get('prompt_context') or {})

    items = []
    for unit in package.get('units') or []:
        if isinstance(unit, dict):
            items.extend(unit.get('items') or [])
    try:
        live_snapshot = _collect_final_review_context_snapshot(items)
        live_context = str(live_snapshot.get('context_digest') or '') or frozen_context
        live_prompt_context = dict(live_snapshot.get('prompt_context') or {}) or frozen_prompt_context
    except Exception as exc:
        print(f'Warning: live context refresh failed ({exc}); using frozen campaign snapshot.')
        live_context = frozen_context
        live_prompt_context = frozen_prompt_context

    try:
        result = fr_llm.prepare_resume_requests(
            package_dir,
            force=bool(force),
            live_context_digest=live_context,
            shared_context=live_prompt_context,
            temperature=BATCH_TEMPERATURE,
            max_output_tokens=BATCH_MAX_OUTPUT_TOKENS,
            thinking_level=BATCH_THINKING_LEVEL,
            model=FINAL_REVIEW_MODEL or BATCH_MODEL or '',
            safety_settings=BATCH_SAFETY_SETTINGS or None,
        )
    except fr.FinalReviewError as exc:
        raise SystemExit(f'Final review resume error: {exc}') from exc

    remember_latest_manifest(result['paths']['manifest'])
    print(f"Resume package: {package_dir}")
    print(f"Units to run: {result['run_count']}")
    print(f"Units skipped (done+same digest): {result['skip_count']}")
    print(f"Force: {bool(force)}")
    if live_context and frozen_context and live_context != frozen_context:
        print('Live shared context differs from frozen build snapshot; stale units re-queued.')
    print(f"Requests JSONL: {os.path.join(package_dir, fr.REQUESTS_JSONL_FILENAME)}")
    if result['run_count']:
        print(
            'Next: submit → status → download → final-review-ingest-results '
            '(do not reuse pre-resume results.jsonl)'
        )
    else:
        print('No units to run; campaign is up to date for current digests.')
    return result


def run_final_review_ingest_results(target=None, result_path='', allow_stale_results=False):
    """Parse downloaded Batch/sync results into findings (report-only)."""
    import final_review as fr
    import final_review_llm as fr_llm

    package_target = manifest_path_for_target(target)
    package = fr.load_campaign_package(package_target)
    package_dir = package['paths']['package_dir']
    try:
        result = fr_llm.ingest_results_into_package(
            package_dir,
            result_path=result_path or '',
            provider=str(SYNC_BACKEND or 'gemini'),
            model=FINAL_REVIEW_MODEL or BATCH_MODEL or '',
            extract_text=extract_text_from_response_payload,
            allow_stale_results=bool(allow_stale_results),
        )
    except fr.FinalReviewError as exc:
        raise SystemExit(f'Final review ingest error: {exc}') from exc

    remember_latest_manifest(result['paths']['manifest'])
    summary = result.get('summary') or {}
    print(f"Ingested results for: {package_dir}")
    print(f"Result rows: {summary.get('result_rows', 0)}")
    print(f"Done units: {summary.get('done_units', 0)}")
    print(f"Failed units: {summary.get('failed_units', 0)}")
    print(f"Findings: {summary.get('finding_count', 0)}")
    print(f"Campaign status: {(result.get('status') or {}).get('status')}")
    print('Report-only: no .rpy writes. Select findings before creating revision candidates.')
    return result


def run_final_review_create_revisions(target=None, finding_ids=None):
    import final_review_revision
    import sys

    return final_review_revision.create_revision_package(sys.modules[__name__], target, finding_ids)


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


def build_keyword_generation_config(max_candidates_per_chunk=None, model=None):
    effective_model = str(model or BATCH_MODEL or '')
    config = {
        'temperature': BATCH_TEMPERATURE,
        'max_output_tokens': BATCH_MAX_OUTPUT_TOKENS,
        'response_mime_type': 'application/json',
        'response_json_schema': build_keyword_response_json_schema(max_candidates_per_chunk),
    }
    if BATCH_THINKING_LEVEL and is_gemini_3_model(effective_model):
        config['thinking_config'] = {
            'thinking_level': BATCH_THINKING_LEVEL.upper(),
        }
    return filter_gemini_generation_config(effective_model, config)


def build_keyword_request(chunk, max_candidates_per_chunk=None, model=None):
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
            'generation_config': build_keyword_generation_config(
                max_candidates_per_chunk,
                model=model,
            ),
        },
    }


def create_keyword_package(display_name_override='', skip_prepare=True, chunk_size=None, max_candidates_per_chunk=None):
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.GEMINI_BATCH,
        required_stages={model_profile.STAGE_KEYWORD},
    )
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
        **translation_quality.manifest_quality_policy_fields(runtime_policy=BATCH_QUALITY_POLICY),
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
    attach_model_routing(
        manifest,
        routing_plan,
    )

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
            **translation_quality.manifest_quality_policy_fields(
            manifest,
            runtime_policy=BATCH_QUALITY_POLICY,
        ),
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
    if _chunk_has_plan_request(chunk):
        # The parent was built through TranslationPlan. Rebuild a derived
        # canonical request for the subchunk instead of falling back to the
        # legacy builders; the derived request uses the parent plan_id and
        # carries deterministic retry lineage in transport metadata.
        plan_request = _build_retry_subchunk_plan_request(chunk, subchunk)
        subchunk.update(
            {
                'request_id': plan_request.request_id,
                'plan_id': plan_request.plan_id,
                'chunk_id': plan_request.chunk_id,
                'system_instruction': plan_request.system_instruction,
                'user_prompt': plan_request.user_prompt,
                'response_schema': plan_request.response_schema,
                'expected_ids': plan_request.expected_ids,
                'capability_requirements': plan_request.capability_requirements,
                'generation_config': plan_request.generation_config,
                'transport_metadata': plan_request.transport_metadata,
                'context_assembly': plan_request.context_assembly,
                'prompt_fingerprint': plan_request.prompt_fingerprint,
                'request_fingerprint': plan_request.request_fingerprint,
            }
        )
    else:
        # Old manifests have no plan fields; their retry subchunks keep using
        # the legacy builders via build_batch_request.
        for plan_field in (
            'request_id',
            'plan_id',
            'chunk_id',
            'system_instruction',
            'user_prompt',
            'response_schema',
            'expected_ids',
            'capability_requirements',
            'generation_config',
            'transport_metadata',
            'context_assembly',
            'prompt_fingerprint',
            'request_fingerprint',
        ):
            subchunk.pop(plan_field, None)
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


def build_retry_chunks_for_keys(manifest, retry_keys, retry_item_ids_by_key=None):
    retry_key_set = set(retry_keys)
    retry_item_ids_by_key = retry_item_ids_by_key or {}
    retry_chunks = []
    for chunk in manifest.get('chunks') or []:
        key = chunk.get('key')
        if key not in retry_key_set:
            continue
        requested_ids = {
            str(item_id)
            for item_id in retry_item_ids_by_key.get(key, ())
            if item_id
        }
        items = chunk.get('items') or []
        available_ids = {
            str(item.get('id') or '')
            for item in items
            if str(item.get('id') or '')
        }
        retry_items = [
            copy.deepcopy(item)
            for item in items
            if str(item.get('id') or '') in requested_ids
        ]
        if (
            requested_ids
            and requested_ids <= available_ids
            and retry_items
            and len(retry_items) < len(items)
        ):
            scoped_chunk = copy.deepcopy(chunk)
            scoped_chunk['items'] = retry_items
            retry_chunks.extend(
                build_retry_subchunk(scoped_chunk, start, end, index)
                for index, (start, end) in enumerate(
                    iter_retry_item_ranges(retry_items),
                    start=1,
                )
            )
            continue
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
            response_payload = row.get('response') or {}
            finish_reason = extract_finish_reason(response_payload)

            if row.get('error'):
                issue_keys.add(key)
                bump_counter(reason_counts, 'row_error')
                continue

            response_text = extract_text_from_response_payload(response_payload)
            if not response_text and not isinstance(row.get('normalized_response'), dict):
                issue_keys.add(key)
                bump_counter(
                    reason_counts,
                    translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
                )
                continue

            try:
                payload = result_row_contract_payload(row)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_TRANSLATION,
                    chunk_items,
                )
                current_reason_counts = contract.reason_counts()
                for reason_code, count in current_reason_counts.items():
                    bump_counter(reason_counts, reason_code, count)
                persisted_reason_deltas = persisted_contract_reason_deltas(
                    row,
                    contract,
                )
                for reason_code, count in persisted_reason_deltas.items():
                    bump_counter(reason_counts, reason_code, count)
            except Exception as exc:
                issue_keys.add(key)
                bump_counter(
                    reason_counts,
                    'truncated_output'
                    if finish_reason == 'MAX_TOKENS'
                    else contract_error_reason(exc, 'failed_to_parse_model_json'),
                )
                continue

            # Envelope-level issues such as an extra unknown ID may not map to
            # any requested retry ID. Retry the whole chunk in that case so a
            # warn result never becomes impossible to repair.
            if contract.issues or persisted_reason_deltas:
                issue_keys.add(key)
                if contract.retry_ids:
                    bump_counter(
                        reason_counts,
                        'truncated_output'
                        if finish_reason == 'MAX_TOKENS'
                        else 'partial_result_items',
                    )

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


    retry_key_set = set(retry_keys)
    retry_item_ids_by_key = {}
    for entry in failure_entries:
        key = entry.get('key')
        item_id = entry.get('id')
        if key in retry_key_set and item_id:
            retry_item_ids_by_key.setdefault(key, set()).add(str(item_id))
    retry_chunks = build_retry_chunks_for_keys(
        manifest,
        retry_keys,
        retry_item_ids_by_key,
    )
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
        **translation_quality.manifest_quality_policy_fields(
            manifest,
            runtime_policy=BATCH_QUALITY_POLICY,
        ),
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


def result_items_from_row(row, label, expected_items, allow_empty=False):
    """Return validated translation items from one persisted result row.

    ``normalized_response`` takes precedence over the raw provider payload.
    ``expected_items`` defines the required validation scope; ``allow_empty``
    permits an empty result, while the default requires at least one valid item.
    """
    try:
        normalized = row.get('normalized_response') if isinstance(row, dict) else None
        if isinstance(normalized, dict):
            payload = normalized
        else:
            response_payload = row.get('response', {}) if isinstance(row, dict) else {}
            response_text = extract_text_from_response_payload(response_payload)
            if not response_text:
                if allow_empty:
                    return []
                raise ValueError('missing response text')
            payload = parse_json_payload(response_text)
        contract = validate_result_contract(
            payload,
            translation_core.MODE_TRANSLATION,
            expected_items,
        )
        if not contract.items and not allow_empty:
            reasons = ', '.join(sorted(contract.reason_counts())) or 'no valid items'
            raise ValueError(f'translation response contract failed: {reasons}')
        return contract.items
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


def canonical_translation_result_row(row, chunk):
    """Attach an authoritative named envelope while preserving provider output."""
    canonical = copy.deepcopy(row) if isinstance(row, dict) else {}
    persisted_diagnostics = canonical.get('contract_diagnostics')
    normalized = canonical.get('normalized_response')
    if isinstance(normalized, dict):
        contract = validate_result_contract(
            normalized,
            translation_core.MODE_TRANSLATION,
            chunk.get('items') or [],
        )
        canonical['normalized_response'] = contract.to_envelope()
        canonical['contract_diagnostics'] = merge_terminal_contract_diagnostics(
            contract,
            [persisted_diagnostics],
        )
        canonical.setdefault('response_semantics', {
            'response': 'provider_payload',
            'normalized_response': 'final_merged_contract',
        })
        return canonical
    response_text = extract_text_from_response_payload(canonical.get('response', {}))
    if not response_text:
        return canonical
    try:
        contract = validate_result_contract(
            parse_json_payload(response_text),
            translation_core.MODE_TRANSLATION,
            chunk.get('items') or [],
        )
    except Exception:
        return canonical
    canonical['normalized_response'] = contract.to_envelope()
    canonical['contract_diagnostics'] = merge_terminal_contract_diagnostics(
        contract,
        [persisted_diagnostics],
    )
    canonical['response_semantics'] = {
        'response': 'provider_payload',
        'normalized_response': 'final_merged_contract',
    }
    return canonical


def merge_parent_row_with_retry_item_rows(parent_row, parent_chunk, retry_chunks, retry_rows_by_key):
    merged_by_id = {}
    terminal_retry_diagnostics = []
    parent_item_ids = {
        str(item.get('id') or '')
        for item in parent_chunk.get('items') or []
        if str(item.get('id') or '')
    }
    if parent_row:
        for item in result_items_from_row(
            parent_row,
            'parent',
            parent_chunk.get('items') or [],
            allow_empty=True,
        ):
            if item.get('id'):
                merged_by_id[item['id']] = item

    replaced_ids = set()
    for retry_chunk in retry_chunks:
        retry_key = retry_chunk.get('key')
        retry_row = retry_rows_by_key.get(retry_key)
        if not retry_row:
            raise SystemExit(f'Retry result is missing row for partial chunk: {retry_key}')
        allowed_ids = {
            str(item.get('id') or '')
            for item in retry_chunk.get('items') or []
            if str(item.get('id') or '')
        }
        for item in result_items_from_row(
            retry_row,
            'retry',
            retry_chunk.get('items') or [],
            allow_empty=True,
        ):
            item_id = item.get('id')
            if item_id in allowed_ids:
                merged_by_id[item_id] = item
                replaced_ids.add(item_id)
        retry_contract = validate_result_contract(
            result_row_contract_payload(retry_row),
            translation_core.MODE_TRANSLATION,
            retry_chunk.get('items') or [],
        )
        terminal_retry_diagnostics.append(
            merge_terminal_contract_diagnostics(
                retry_contract,
                [retry_row.get('contract_diagnostics')],
                ignored_unknown_ids=parent_item_ids - allowed_ids,
            )
        )

    ordered_items = []
    for target_item in parent_chunk.get('items') or []:
        item_id = target_item.get('id')
        if item_id in merged_by_id:
            ordered_items.append(merged_by_id[item_id])

    merged_row = copy.deepcopy(parent_row) if isinstance(parent_row, dict) else {}
    merged_row['key'] = parent_chunk.get('key')
    merged_row.pop('error', None)
    merged_payload = {
        'translations': compact_result_items_for_response(ordered_items),
    }
    merged_contract = validate_result_contract(
        merged_payload,
        translation_core.MODE_TRANSLATION,
        parent_chunk.get('items') or [],
    )
    merged_row['normalized_response'] = merged_contract.to_envelope()
    merged_row['contract_diagnostics'] = merge_terminal_contract_diagnostics(
        merged_contract,
        terminal_retry_diagnostics,
    )
    merged_row['response_semantics'] = {
        'response': 'first_pass_provider_payload',
        'normalized_response': 'final_merged_contract',
    }
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

    seen_retry_keys = set()
    for chunk in retry_chunks:
        key = chunk.get('key')
        if not key:
            raise SystemExit('Retry manifest contains a chunk without a key.')
        if key in seen_retry_keys:
            raise SystemExit(f'Retry manifest contains duplicate chunk key: {key}')
        seen_retry_keys.add(key)
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
    retry_chunks_by_key = {chunk['key']: chunk for chunk in retry_chunks}
    retry_keys = list(retry_chunks_by_key)
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

    import_manifest_usage_best_effort(parent_manifest)
    import_manifest_usage_best_effort(retry_manifest, result_path=retry_result_path)

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
            retry_chunk = retry_chunks_by_key[key]
            merged_rows.append(canonical_translation_result_row(
                retry_rows_by_key[key], retry_chunk
            ))
            replaced_keys.add(key)
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
            retry_chunk = retry_chunks_by_key[key]
            merged_rows.append(canonical_translation_result_row(
                retry_rows_by_key[key], retry_chunk
            ))
            replaced_keys.add(key)
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
        raise cli_contract.MachineContractError(
            f'Batch input JSONL not found: {jsonl_path or exc}',
            code_name='BATCH_INPUT_NOT_FOUND',
            suggested_action='rebuild_batch_package',
            details={'input_jsonl_path': jsonl_path},
        ) from exc
    except json.JSONDecodeError as exc:
        jsonl_path = manifest.get('input_jsonl_path') or ''
        raise cli_contract.MachineContractError(
            f'Batch input JSONL is not valid JSON: {jsonl_path} ({exc})',
            code_name='INVALID_BATCH_INPUT_JSON',
            suggested_action='rebuild_batch_package',
            details={
                'input_jsonl_path': jsonl_path,
                'line': exc.lineno,
                'column': exc.colno,
            },
        ) from exc


def _manifest_package_dir(manifest):
    package_dir = manifest.get('_package_dir')
    if package_dir:
        return package_dir
    manifest_path = manifest.get('_manifest_path')
    if manifest_path:
        return os.path.dirname(manifest_path)
    raise cli_contract.MachineContractError(
        'Manifest package directory is missing.',
        code_name='MANIFEST_PACKAGE_DIR_MISSING',
        suggested_action='rebuild_or_repair_manifest',
    )


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

    if manifest.get('submit_disabled'):
        raise SystemExit(
            'Submit disabled for this manifest: it contains local revision candidates '
            'and must use preview-revisions/apply-revisions.'
        )
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

    submit_stage = model_profile.stage_for_manifest_mode(manifest_mode(manifest))
    submit_plan = resolve_manifest_routing_plan(
        manifest,
        execution=model_profile.ExecutionStrategy.GEMINI_BATCH,
    )
    if model_override:
        submit_plan = model_profile.override_gemini_batch_stage(
            submit_plan,
            submit_stage,
            model_override.strip(),
            custom_providers=_runtime_custom_providers(),
        )
        attach_model_routing(manifest, submit_plan)
    require_valid_routing_plan(submit_plan, {submit_stage})

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

    attempts = (
        legacy.api_key_rotation_attempts()
        if hasattr(legacy, 'api_key_rotation_attempts')
        else max(1, len(getattr(legacy, 'API_KEYS', []) or []))
    )
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
    expected_sha = manifest.get('result_jsonl_sha256')
    if os.path.isfile(result_path) and not force:
        if result_artifact_is_complete(result_path, expected_sha):
            import_manifest_usage_best_effort(manifest)
            print(f'Result file already exists: {result_path}')
            return result_path
        print(
            f'Result file looks incomplete or corrupt (will re-download): {result_path}'
        )

    result_file_name = manifest.get('result_file_name')
    if not result_file_name:
        raise SystemExit('Result file name is missing from manifest/job metadata.')

    client = create_batch_client(api_key_index=manifest.get('submitted_api_key_index'))
    print(f'Downloading result file: {result_file_name}')
    downloaded = client.files.download(file=result_file_name)
    text = decode_downloaded_content(downloaded)
    if not isinstance(text, str):
        text = str(text)
    content_sha = sha256_text(text)

    # Atomic replace so a crash cannot leave a truncated results.jsonl that
    # later download runs would skip without --force.
    atomic_write_text(result_path, text)
    atomic_write_text(f'{result_path}.sha256', content_sha + '\n')

    manifest['result_jsonl_path'] = result_path
    manifest['result_jsonl_sha256'] = content_sha
    manifest['downloaded_at'] = datetime.now().isoformat(timespec='seconds')
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    import_manifest_usage_best_effort(manifest)
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
    return dict(usage_metadata)


def import_manifest_usage(manifest, result_path=None):
    """Offline-import real provider result sources for one manifest lineage.

    A merged retry file can contain locally synthesized rows. For that state,
    import the original parent result plus each retry result instead of treating
    the merged JSONL as another provider call.
    """
    pricing_config = batch_cost_estimate.load_pricing_config(
        _read_translator_config_object()
    )
    sources = []
    if result_path:
        sources.append((manifest, result_path))
    else:
        merge_history = manifest.get('retry_merge_history')
        if isinstance(merge_history, list) and merge_history:
            first = merge_history[0] if isinstance(merge_history[0], dict) else {}
            parent_result = first.get('previous_result_jsonl_path')
            if parent_result:
                sources.append((manifest, parent_result))
            for entry in merge_history:
                if not isinstance(entry, dict):
                    continue
                retry_manifest_path = entry.get('retry_manifest')
                if not retry_manifest_path:
                    continue
                retry_manifest = load_manifest(retry_manifest_path)
                retry_result = (
                    entry.get('retry_result_jsonl_path')
                    or resolve_manifest_result_path(retry_manifest)
                )
                sources.append((retry_manifest, retry_result))
        if not sources:
            sources.append((manifest, resolve_manifest_result_path(manifest)))

    summaries = [
        model_usage_ledger.import_manifest_results(
            source_manifest,
            result_path=source_result,
            pricing_config=pricing_config,
        )
        for source_manifest, source_result in sources
    ]
    last = summaries[-1]
    combined = {
        **last,
        'result_path': last.get('result_path') if len(summaries) == 1 else '',
        'result_paths': [item.get('result_path') for item in summaries],
    }
    for key in (
        'scanned_rows', 'candidate_records', 'skipped_rows',
        'inserted_records', 'duplicate_records',
    ):
        combined[key] = sum(int(item.get(key) or 0) for item in summaries)
    return combined


def import_manifest_usage_best_effort(manifest, result_path=None):
    """Record auxiliary usage without changing the translation/result workflow."""
    try:
        return import_manifest_usage(manifest, result_path=result_path)
    except (
        OSError,
        ValueError,
        model_usage_ledger.UsageLedgerError,
    ) as exc:
        print(f'Warning: Model usage ledger import failed: {exc}')
        return {
            'inserted_records': 0,
            'duplicate_records': 0,
            'error': str(exc),
        }


def record_generation_usage_best_effort(
    *,
    task_mode,
    stage,
    result,
    operation_id,
    run_id,
    source_key,
    thinking_level='',
    source=None,
    pricing_config=None,
):
    """Record one successful synchronous response without failing its caller."""
    if not legacy.BASE_DIR:
        return {
            'inserted_records': 0,
            'duplicate_records': 0,
            'error': 'game_root is unset',
        }
    try:
        return model_usage_ledger.record_generation_usage(
            game_root=legacy.BASE_DIR,
            task_mode=task_mode,
            stage=stage,
            provider=str(result.get('provider') or SYNC_BACKEND or 'unknown'),
            model=str(result.get('model') or SYNC_MODEL or BATCH_MODEL or 'unknown'),
            usage_metadata=result.get('usage_metadata') or {},
            response_payload=result.get('response_payload') or {},
            operation_id=operation_id,
            run_id=run_id,
            thinking_level=thinking_level,
            execution_mode=str(result.get('execution_mode') or 'sync'),
            source_key=source_key,
            source=source or {},
            response_diagnostics=result.get('output_diagnostics') or {},
            request_metadata=result.get('request_metadata') or {},
            pricing_config=pricing_config,
        )
    except (OSError, ValueError, model_usage_ledger.UsageLedgerError) as exc:
        print(f'Warning: Model usage ledger record failed: {exc}')
        return {
            'inserted_records': 0,
            'duplicate_records': 0,
            'error': str(exc),
        }


def bump_counter(bucket, name, amount=1):
    bucket[name] = bucket.get(name, 0) + amount


def contract_error_reason(exc, fallback):
    """Return a stable contract reason for parse and validation failures."""
    if isinstance(exc, json.JSONDecodeError):
        return translation_core.CONTRACT_INVALID_JSON
    return str(getattr(exc, 'reason_code', '') or fallback)


def record_contract_reasons(summary, report):
    bucket = summary.setdefault('reason_counts', {})
    for reason_code, count in report.reason_counts().items():
        bump_counter(bucket, reason_code, count)
    diagnostic_bucket = summary.setdefault('diagnostic_counts', {})
    for reason_code, count in report.diagnostic_counts().items():
        bump_counter(diagnostic_bucket, reason_code, count)


def sync_output_diagnostics(result, request_payload=None):
    """Return safe output-budget diagnostics for one synchronous response."""
    diagnostics = result.get('output_diagnostics')
    if isinstance(diagnostics, dict) and diagnostics:
        return model_usage_ledger.normalize_response_diagnostics(diagnostics)
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    generation_config = request_payload.get('generation_config')
    generation_config = generation_config if isinstance(generation_config, dict) else {}
    return model_usage_ledger.response_budget_diagnostics(
        response_text=result.get('response_text') or '',
        finish_reason=result.get('finish_reason') or '',
        usage_metadata=result.get('usage_metadata') or {},
        max_output_tokens=generation_config.get('max_output_tokens'),
    )


def record_sync_output_summary(summary, diagnostics):
    """Add one safe response diagnostic to a sync manifest summary."""
    diagnostics = model_usage_ledger.normalize_response_diagnostics(diagnostics)
    for field in ('completion_tokens', 'reasoning_tokens', 'text_output_tokens'):
        value = diagnostics.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            summary[field] = int(summary.get(field) or 0) + value
            known_key = f'{field}_known_requests'
            summary[known_key] = int(summary.get(known_key) or 0) + 1
    summary['reasoning_budget_pressure_count'] = int(
        summary.get('reasoning_budget_pressure_count') or 0
    ) + int(diagnostics.get('reasoning_budget_pressure') is True)
    summary['truncated_output_count'] = int(
        summary.get('truncated_output_count') or 0
    ) + int(diagnostics.get('truncated') is True)
    reason_code = str(diagnostics.get('reason_code') or '').strip()
    if reason_code:
        bump_counter(summary.setdefault('output_reason_counts', {}), reason_code)


def _sync_token_summary_value(summary, field):
    """Format an aggregate only when at least one provider reported it."""
    if int(summary.get(f'{field}_known_requests') or 0) <= 0:
        return 'unknown'
    return str(int(summary.get(field) or 0))


def print_sync_output_summary(summary):
    """Print the stable CLI token/diagnostic contract consumed by the GUI."""
    for line in model_usage_ledger.format_sync_output_lines(
        completion=_sync_token_summary_value(summary, 'completion_tokens'),
        reasoning=_sync_token_summary_value(summary, 'reasoning_tokens'),
        text_output=_sync_token_summary_value(summary, 'text_output_tokens'),
        reasoning_budget_pressure=int(
            summary.get('reasoning_budget_pressure_count') or 0
        ),
        truncated=int(summary.get('truncated_output_count') or 0),
    ):
        print(line)


def contract_diagnostics_counts(diagnostics, field_name):
    if not isinstance(diagnostics, dict):
        return {}
    raw_counts = diagnostics.get(field_name)
    if not isinstance(raw_counts, dict):
        return {}

    counts = {}
    for reason_code, raw_count in raw_counts.items():
        if not reason_code or isinstance(raw_count, bool):
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(reason_code)] = count
    return counts


def persisted_contract_reason_counts(row):
    """Return unresolved final issues erased by response normalization."""
    diagnostics = row.get('contract_diagnostics')
    if not isinstance(diagnostics, dict) or diagnostics.get('complete') is not False:
        return {}
    return contract_diagnostics_counts(diagnostics, 'reason_counts')


def persisted_contract_reason_deltas(row, report):
    """Return persisted issues erased by the canonical normalized envelope."""
    current_counts = report.reason_counts()
    return {
        reason_code: count - current_counts.get(reason_code, 0)
        for reason_code, count in persisted_contract_reason_counts(row).items()
        if count > current_counts.get(reason_code, 0)
    }


def record_result_row_contract_reasons(summary, row, report):
    """Record current validation plus non-reconstructable persisted issues."""
    record_contract_reasons(summary, report)
    deltas = persisted_contract_reason_deltas(row, report)
    bucket = summary.setdefault('reason_counts', {})
    for reason_code, count in deltas.items():
        bump_counter(bucket, reason_code, count)
    return deltas


def persisted_contract_issue_entries(row, reason_deltas):
    """Expand persisted issue deltas into failure-report entries."""
    diagnostics = row.get('contract_diagnostics')
    issues = diagnostics.get('issues') if isinstance(diagnostics, dict) else None
    remaining = dict(reason_deltas or {})
    entries = []
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            reason_code = str(issue.get('reason_code') or '')
            if remaining.get(reason_code, 0) <= 0:
                continue
            entries.append(dict(issue))
            remaining[reason_code] -= 1
    for reason_code, count in remaining.items():
        for _index in range(max(0, count)):
            entries.append({
                'reason_code': reason_code,
                'message': f'Persisted model contract issue: {reason_code}',
            })
    return entries


def result_row_contract_failure_entries(
    manifest,
    chunk,
    row,
    report,
    item_map,
    reason_deltas=None,
    finish_reason='',
    usage_metadata=None,
    ignored_item_ids=None,
):
    """Return auditable failures for current and persisted contract issues."""
    ignored_item_ids = {
        str(item_id)
        for item_id in (ignored_item_ids or ())
        if str(item_id)
    }
    issue_entries = [
        issue.to_dict()
        for issue in report.issues
        if issue.reason_code != translation_core.CONTRACT_MISSING_EXPECTED_ID
    ]
    issue_entries.extend(
        issue
        for issue in persisted_contract_issue_entries(row, reason_deltas)
        if issue.get('reason_code')
        != translation_core.CONTRACT_MISSING_EXPECTED_ID
    )

    failures = []
    for issue in issue_entries:
        issue_id = str(issue.get('id') or '')
        if issue_id in ignored_item_ids:
            continue
        target_item = item_map.get(issue_id)
        extra = {
            'reason_code': str(
                issue.get('reason_code') or 'response_contract_error'
            ),
            'finish_reason': finish_reason,
            'usage_metadata': usage_metadata or {},
        }
        for field in ('result_index', 'field'):
            if field in issue:
                extra[field] = issue[field]
        failures.append(make_failure_entry(
            manifest,
            str(
                issue.get('message')
                or f'Model contract issue: {extra["reason_code"]}'
            ),
            file_rel_path=chunk.get('file_rel_path', ''),
            item_id=issue_id,
            line=target_item.get('line') if target_item else None,
            text=(
                target_item.get('source') or target_item.get('text', '')
                if target_item
                else ''
            ),
            key=chunk.get('key', ''),
            **extra,
        ))
    return failures


def merge_terminal_contract_diagnostics(
    report,
    terminal_diagnostics,
    ignored_unknown_ids=None,
):
    """Preserve terminal issues that canonical envelopes cannot reconstruct."""
    merged = report.to_diagnostics()
    ignored_unknown_ids = {
        str(item_id)
        for item_id in (ignored_unknown_ids or ())
        if str(item_id)
    }

    def remove_ignored_unknown_issues(diagnostics):
        if not ignored_unknown_ids or not isinstance(diagnostics, dict):
            return diagnostics
        issues = diagnostics.get('issues')
        if not isinstance(issues, list):
            return diagnostics
        retained = []
        removed = 0
        for issue in issues:
            if (
                isinstance(issue, dict)
                and str(issue.get('reason_code') or '')
                == translation_core.CONTRACT_UNKNOWN_ID
                and str(issue.get('id') or '') in ignored_unknown_ids
            ):
                removed += 1
                continue
            retained.append(issue)
        if not removed:
            return diagnostics
        filtered = copy.deepcopy(diagnostics)
        filtered['issues'] = retained
        counts = dict(filtered.get('reason_counts') or {})
        validated_counts = contract_diagnostics_counts(
            filtered,
            'reason_counts',
        )
        remaining = max(
            0,
            validated_counts.get(translation_core.CONTRACT_UNKNOWN_ID, 0)
            - removed,
        )
        if remaining:
            counts[translation_core.CONTRACT_UNKNOWN_ID] = remaining
        else:
            counts.pop(translation_core.CONTRACT_UNKNOWN_ID, None)
        filtered['reason_counts'] = counts
        if not counts and not filtered.get('retry_ids'):
            filtered['complete'] = True
        return filtered

    merged = remove_ignored_unknown_issues(merged)
    source_reason_counts = {}
    source_diagnostic_counts = {}
    source_issues = []
    source_diagnostics_entries = []
    for diagnostics in terminal_diagnostics or []:
        if not isinstance(diagnostics, dict):
            continue
        diagnostics = remove_ignored_unknown_issues(diagnostics)
        for reason_code, count in contract_diagnostics_counts(
            diagnostics,
            'reason_counts',
        ).items():
            bump_counter(source_reason_counts, reason_code, count)
        for reason_code, count in contract_diagnostics_counts(
            diagnostics,
            'diagnostic_counts',
        ).items():
            bump_counter(source_diagnostic_counts, reason_code, count)
        if isinstance(diagnostics.get('issues'), list):
            source_issues.extend(
                dict(issue)
                for issue in diagnostics['issues']
                if isinstance(issue, dict)
            )
        if isinstance(diagnostics.get('diagnostics'), list):
            source_diagnostics_entries.extend(
                dict(item)
                for item in diagnostics['diagnostics']
                if isinstance(item, dict)
            )

    def merge_entries(count_field, entry_field, source_counts, source_entries):
        current_counts = merged.setdefault(count_field, {})
        current_entries = merged.setdefault(entry_field, [])
        added = 0
        for reason_code, source_count in source_counts.items():
            current_count = current_counts.get(reason_code, 0)
            target_count = max(current_count, source_count)
            delta = target_count - current_count
            if not delta:
                continue
            current_counts[reason_code] = target_count
            matching = [
                dict(entry)
                for entry in source_entries
                if str(entry.get('reason_code') or '') == reason_code
            ]
            existing = {
                stable_json_sha256(entry)
                for entry in current_entries
                if isinstance(entry, dict)
            }
            additions = []
            for entry in matching:
                identity = stable_json_sha256(entry)
                if identity in existing:
                    continue
                existing.add(identity)
                additions.append(entry)
                if len(additions) == delta:
                    break
            current_entries.extend(additions)
            for _index in range(max(0, delta - len(additions))):
                current_entries.append({'reason_code': reason_code})
            added += delta
        return added

    added_issues = merge_entries(
        'reason_counts',
        'issues',
        source_reason_counts,
        source_issues,
    )
    merge_entries(
        'diagnostic_counts',
        'diagnostics',
        source_diagnostic_counts,
        source_diagnostics_entries,
    )
    if added_issues:
        merged['complete'] = False
    return merged


def validate_result_contract(payload, mode, expected_items):
    return translation_core.validate_model_response(
        payload,
        mode=mode,
        expected_units=expected_items,
        allow_legacy=True,
    )


def result_row_contract_payload(row):
    """Return the authoritative contract payload from a persisted result row.

    Sync rows retain ``response`` as the raw first-pass provider payload for
    compatibility and usage accounting. After a targeted retry,
    ``normalized_response`` is the final merged contract and therefore takes
    precedence for validation, preview, export, and apply consumers.
    """
    normalized = row.get('normalized_response')
    if isinstance(normalized, dict):
        return normalized
    response_text = extract_text_from_response_payload(row.get('response', {}))
    return parse_json_payload(response_text)


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
    return translation_core.parse_model_response_json(text)


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
        os.path.join(manifest.get('_package_dir', ''), 'quality_findings.jsonl'),
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


def _keyword_history_file_paths(manifest):
    """Resolve the keyword package's recorded source files for history scan."""

    file_paths = []
    base_dir = str(manifest.get('base_dir') or '').strip()
    package_dir = str(manifest.get('_package_dir') or '').strip()
    for rel_path, info in sorted((manifest.get('files') or {}).items()):
        if not isinstance(info, dict):
            continue
        raw_path = str(info.get('path') or '').strip()
        if not raw_path:
            continue
        file_path = raw_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(base_dir or package_dir, file_path)
        file_paths.append((str(rel_path), os.path.abspath(file_path)))
    return file_paths


def collect_keyword_history_corpus(manifest):
    """Build a read-only revision-corpus projection for keyword evidence.

    The scan reuses ``collect_revision_file_jobs`` and
    ``revision_corpus.build_corpus_items``.  It is intentionally derived at
    export time and is never persisted as a second canonical translation
    store.  Digests are collected before scanning and after corpus construction
    so a source change during either phase fails closed.  A missing source file
    fails closed so candidates receive an ``unavailable`` evidence record rather
    than an apparently safe match.
    """

    file_paths = _keyword_history_file_paths(manifest)
    if not file_paths:
        return {
            'items': [],
            'status': 'unavailable',
            'reason': 'history_scan_unavailable',
            'file_count': 0,
            'source_changed_during_scan': False,
            'diagnostics': ['keyword manifest has no recorded source files'],
        }

    file_path_map = {rel_path: file_path for rel_path, file_path in file_paths}
    missing = sorted(rel_path for rel_path, file_path in file_paths if not os.path.isfile(file_path))
    if missing:
        return {
            'items': [],
            'status': 'unavailable',
            'reason': 'history_scan_unavailable',
            'file_count': len(file_paths),
            'source_changed_during_scan': False,
            'diagnostics': [f'missing source file: {rel_path}' for rel_path in missing],
        }

    try:
        digests_before = revision_corpus.collect_file_digests(file_path_map)
        file_jobs = collect_revision_file_jobs(
            file_paths=file_paths,
            include_empty_files=True,
        )
        items, diagnostics = revision_corpus.build_corpus_items(file_jobs)
        digests_after = revision_corpus.collect_file_digests(file_path_map)
        # The revision scanner also recognizes untranslated comment/source
        # pairs.  Ordinary changed translations are historical evidence;
        # unchanged rows are kept as review-only preserve_evidence so
        # preserve-term candidates are not silently auto-accepted.
        items = [
            item for item in items
            if keyword_history.is_history_evidence_row(item)
        ]
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            'items': [],
            'status': 'unavailable',
            'reason': 'history_scan_unavailable',
            'file_count': len(file_paths),
            'source_changed_during_scan': False,
            'diagnostics': [str(exc)],
        }

    return {
        'items': items,
        'status': 'ready',
        'reason': '',
        'file_count': len(file_paths),
        'source_changed_during_scan': digests_before != digests_after,
        'diagnostics': diagnostics,
    }


def _keyword_history_summary(candidates, history_scan):
    status_counts = {
        keyword_history.STATUS_CONSISTENT: 0,
        keyword_history.STATUS_CONFLICT: 0,
        keyword_history.STATUS_AMBIGUOUS: 0,
        keyword_history.STATUS_PRESERVE_EVIDENCE: 0,
        keyword_history.STATUS_UNMATCHED: 0,
        keyword_history.STATUS_UNAVAILABLE: 0,
    }
    for candidate in candidates:
        evidence = candidate.get('history_evidence') or {}
        status = str(evidence.get('status') or keyword_history.STATUS_UNAVAILABLE)
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        'schema_version': keyword_history.HISTORY_EVIDENCE_SCHEMA_VERSION,
        'scan_status': history_scan.get('status') or 'unavailable',
        'file_count': int(history_scan.get('file_count') or 0),
        'occurrence_count': len(history_scan.get('items') or []),
        'source_changed_during_scan': bool(history_scan.get('source_changed_during_scan')),
        'diagnostic_count': len(history_scan.get('diagnostics') or []),
        'candidate_status_counts': status_counts,
        'diagnostics': list(history_scan.get('diagnostics') or []),
    }


def format_keyword_history_markdown(evidence):
    """Render one compact history-evidence cell for the Markdown report."""

    evidence = evidence if isinstance(evidence, dict) else {}
    status = str(evidence.get('status') or keyword_history.STATUS_UNAVAILABLE)
    first = evidence.get('first_occurrence') or {}
    if first:
        location = f"{first.get('file_rel_path') or '?'}:L{first.get('line_number') or 0}"
        identity = first.get('identity_v2') or first.get('occurrence_id') or '?'
        translation = str(first.get('current_translation') or '(空)')
        text = f"{location} [{identity}] → {translation} [{status}]"
    else:
        text = f"无首次 occurrence [{status}]"
    reasons = evidence.get('conflict_reasons') or []
    if reasons:
        text += '；' + '；'.join(str(reason) for reason in reasons)
    return text


def write_keyword_markdown(path, candidates, summary):
    lines = [
        '# Keyword Candidates',
        '',
        f"- Candidate count: {len(candidates)}",
        f"- Parsed chunks: {summary.get('parsed_chunks', 0)}/{summary.get('expected_chunks', summary.get('result_rows', 0))}",
        f"- Missing chunk rows: {summary.get('missing_chunk_rows', 0)}",
        f"- Ambiguous provenance candidates: {summary.get('ambiguous_provenance_candidates', 0)}",
        f"- Historical evidence: {summary.get('history_candidate_status_counts', {})}",
        '',
        '| Source | Suggested target | Category | Confidence | Evidence | Files | First historical occurrence / current translation |',
        '| --- | --- | --- | ---: | --- | --- | --- |',
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
                    markdown_escape_cell(
                        format_keyword_history_markdown(candidate.get('history_evidence'))
                    ),
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
            if not response_text and not isinstance(row.get('normalized_response'), dict):
                summary['parse_errors'] += 1
                summary['missing_response_chunks'] += 1
                bump_counter(
                    summary['reason_counts'],
                    translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
                )
                continue
            try:
                payload = result_row_contract_payload(row)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_KEYWORD_EXTRACTION,
                    chunk.get('items') or [],
                )
                record_contract_reasons(summary, contract)
                keyword_payload = contract.to_envelope()
                candidates = contract.items
            except Exception as exc:
                summary['parse_errors'] += 1
                bump_counter(
                    summary['reason_counts'],
                    contract_error_reason(exc, 'failed_to_parse_keyword_json'),
                )
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

    history_scan = collect_keyword_history_corpus(manifest)
    candidates = keyword_history.attach_keyword_history_evidence(
        candidates,
        history_scan.get('items') or [],
        source_changed_during_scan=bool(history_scan.get('source_changed_during_scan')),
        unavailable_reason=str(history_scan.get('reason') or ''),
    )
    history_summary = _keyword_history_summary(candidates, history_scan)
    summary['candidate_count_deduped'] = len(candidates)
    summary['history_scan_status'] = history_summary['scan_status']
    summary['history_occurrence_count'] = history_summary['occurrence_count']
    summary['history_file_count'] = history_summary['file_count']
    summary['history_source_changed_during_scan'] = history_summary['source_changed_during_scan']
    summary['history_candidate_status_counts'] = history_summary['candidate_status_counts']
    summary['history_diagnostic_count'] = history_summary['diagnostic_count']
    if history_summary['diagnostics']:
        summary['history_diagnostics'] = history_summary['diagnostics']
        bump_counter(summary['reason_counts'], 'history_scan_diagnostic', len(history_summary['diagnostics']))
    for status, count in history_summary['candidate_status_counts'].items():
        if status != keyword_history.STATUS_CONSISTENT and count:
            bump_counter(summary['reason_counts'], f'history_{status}', count)

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
        'history_evidence': history_summary,
    }
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print(f'Keyword candidates: {summary["candidate_count_deduped"]} deduped from {summary["candidate_count_raw"]} raw')
    print(f'Chunk summaries: {summary["chunk_summary_count"]}')
    print(f'JSONL: {jsonl_path}')
    print(f'Markdown: {markdown_path}')
    print(f'Summary JSONL: {summary_jsonl_path}')
    print(f'Summary Markdown: {summary_markdown_path}')
    print(
        'Historical evidence: '
        f"{summary['history_occurrence_count']} occurrences, "
        f"statuses={summary['history_candidate_status_counts']}"
    )
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


def _adapter_request_for_manifest(manifest, file_keys):
    identity = {}
    try:
        identity = manifest_project_identity(manifest)
    except (Exception, SystemExit):
        identity = {}

    tl_dir = str(identity.get('tl_dir') or getattr(legacy, 'TL_DIR', '') or '')
    if not tl_dir:
        for file_key in file_keys:
            file_info = (manifest.get('files') or {}).get(file_key) or {}
            path_value = file_info.get('path') if isinstance(file_info, dict) else ''
            if not path_value or not os.path.isabs(path_value):
                continue
            try:
                rel_path = normalize_safe_rel_path(file_key, f'manifest file key {file_key}')
                candidate = _canonical_abs_path(path_value)
                for _part in Path(rel_path).parts:
                    candidate = os.path.dirname(candidate)
                tl_dir = candidate
                break
            except (Exception, SystemExit):
                continue
    if not tl_dir:
        raise WritebackPlanError(
            'common.writeback.project_mismatch',
            "Cannot determine the live Ren'Py localization directory for writeback.",
        )

    project_root = str(
        identity.get('base_dir')
        or getattr(legacy, 'BASE_DIR', '')
        or os.path.dirname(tl_dir)
    )
    return ProjectDiscoveryRequest(
        project_root=project_root,
        localization_root=tl_dir,
        target_language=str(
            manifest.get('target_language')
            or manifest.get('language')
            or getattr(legacy, 'PREP_LANGUAGE', '')
            or ''
        ),
        include_files=tuple(sorted(str(value) for value in file_keys)),
    )


def _source_document_from_path(file_key, file_path):
    content = Path(file_path).read_bytes()
    return SourceDocument(
        file_rel_path=str(file_key),
        file_path=str(file_path),
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def _adapter_render_key(file_key):
    try:
        normalized = normalize_safe_rel_path(
            str(file_key),
            f'adapter writeback file key {file_key}',
        )
    except SystemExit as exc:
        raise WritebackPlanError(
            'common.writeback.path_escape',
            str(exc),
        ) from exc
    return normalized.replace('\\', '/')


def _normalize_adapter_rendered_files(rendered_by_file):
    normalized = {}
    for file_key, rendered_lines in rendered_by_file.items():
        normalized_key = _adapter_render_key(file_key)
        if normalized_key in normalized:
            raise WritebackPlanError(
                'common.writeback.plan_invalid',
                f'Duplicate rendered adapter target: {normalized_key}.',
            )
        normalized[normalized_key] = rendered_lines
    return normalized


def _require_adapter_render_targets(replacements_by_file, rendered_by_file):
    expected = {
        _adapter_render_key(file_key)
        for file_key, replacements_by_line in replacements_by_file.items()
        if any(replacements_by_line.values())
    }
    actual = set(rendered_by_file)
    if actual == expected:
        return
    missing = ', '.join(sorted(expected - actual)) or 'none'
    unexpected = ', '.join(sorted(actual - expected)) or 'none'
    raise WritebackPlanError(
        'common.writeback.target_missing',
        'Adapter rendered targets do not match validated replacements '
        f'(missing: {missing}; unexpected: {unexpected}).',
    )


def _find_adapter_occurrence(occurrences, file_key, line, start, end, expected_text, item_id):
    normalized_file_key = _adapter_render_key(file_key)

    def matches_validated_replacement(occurrence):
        unit = occurrence.unit
        return (
            unit.file_rel_path == normalized_file_key
            and unit.line == line
            and unit.start == start
            and unit.end == end
            and expected_text
            in {
                unit.text,
                unit.source_text,
                unit.current_translation,
            }
        )

    if item_id:
        by_id = [
            occurrence
            for occurrence in occurrences
            if occurrence.unit.id == item_id
            and occurrence.unit.file_rel_path == normalized_file_key
        ]
        if len(by_id) == 1 and matches_validated_replacement(by_id[0]):
            return by_id[0]

    positioned = [
        occurrence
        for occurrence in occurrences
        if matches_validated_replacement(occurrence)
    ]
    if len(positioned) == 1:
        return positioned[0]
    if len(positioned) > 1:
        raise WritebackPlanError(
            'common.locator.unresolved',
            f'Ambiguous adapter occurrence at {normalized_file_key}:{line}:{start}-{end}.',
        )
    raise WritebackPlanError(
        'common.locator.unresolved',
        'Adapter occurrence could not be resolved at '
        f'{normalized_file_key}:{line}:{start}-{end}.',
    )


def _build_adapter_writeback_plan(manifest, replacements_by_file, live_sources=None):
    file_keys = tuple(
        sorted(
            file_key
            for file_key, replacements_by_line in replacements_by_file.items()
            if any(replacements_by_line.values())
        )
    )
    if not file_keys:
        return None, None

    request = _adapter_request_for_manifest(manifest, file_keys)
    adapter = RenPyAdapter(legacy_module=legacy)
    snapshot = build_translation_snapshot(adapter, request)
    validated = []
    used_occurrence_ids = set()
    occurrences = tuple(snapshot.occurrences)
    chunks_by_key = {
        str(chunk.get('key') or ''): chunk
        for chunk in manifest.get('chunks', [])
    }
    if live_sources is not None:
        authoritative_sources = tuple(
            sorted(live_sources, key=lambda document: document.file_rel_path)
        )
        if source_snapshot_fingerprint(authoritative_sources) != source_snapshot_fingerprint(
            snapshot.project.source_documents
        ):
            raise WritebackPlanError(
                'common.writeback.source_snapshot_mismatch',
                'Live source changed between apply validation and adapter discovery.',
            )
        snapshot = replace(
            snapshot,
            project=replace(snapshot.project, source_documents=authoritative_sources),
        )

    for file_key in file_keys:
        replacements_by_line = replacements_by_file.get(file_key) or {}
        for line, replacements in replacements_by_line.items():
            for replacement in replacements:
                start, end, translated, _prefix, _quote, expected_text, item_id, chunk_key = (
                    unpack_replacement_for_validation(replacement)
                )
                occurrence = _find_adapter_occurrence(
                    occurrences,
                    file_key,
                    int(line),
                    int(start),
                    int(end),
                    str(expected_text or ''),
                    str(item_id or ''),
                )
                if occurrence.occurrence_id in used_occurrence_ids:
                    raise WritebackPlanError(
                        'common.writeback.span_overlap',
                        f'Duplicate adapter occurrence in writeback: {file_key}:{line}:{start}-{end}.',
                    )
                used_occurrence_ids.add(occurrence.occurrence_id)
                validation = adapter.validate_translation(
                    occurrence,
                    str(translated or ''),
                )
                if validation.status != 'pass':
                    policy_chunk = chunks_by_key.get(str(chunk_key or ''))
                    if (
                        policy_chunk is None
                        or policy_chunk.get('file_rel_path') != file_key
                    ):
                        policy_chunk = next(
                            (
                                chunk
                                for chunk in manifest.get('chunks', [])
                                if chunk.get('file_rel_path') == file_key
                            ),
                            {'file_rel_path': file_key, 'items': []},
                        )
                    policy_item = next(
                        (
                            candidate
                            for candidate in policy_chunk.get('items', [])
                            if str(candidate.get('id') or '') == str(item_id or '')
                        ),
                        None,
                    )
                    original_text = (
                        occurrence.unit.source_text
                        if manifest_mode(manifest) == MANIFEST_MODE_REVISION
                        else occurrence.unit.text
                    )
                    if _adapter_target_language_policy_allows(
                        manifest,
                        policy_chunk,
                        policy_item,
                        original_text,
                        str(translated or ''),
                        reason_codes=validation.reason_codes,
                    ):
                        validation = replace(validation, status='pass')
                    else:
                        codes = ','.join(validation.reason_codes) or 'adapter.validation.block'
                        raise WritebackPlanError(
                            'adapter.validation.block',
                            f'Adapter validation blocked {file_key}:{line}: {codes}.',
                        )
                validated.append(
                    ValidatedTranslation(
                        occurrence=occurrence,
                        translated_text=str(translated or ''),
                        validation=validation,
                    )
                )

    plan = adapter.build_writeback_plan(
        snapshot.project,
        tuple(validated),
        snapshot.project.source_documents,
    )
    return plan, snapshot


def _validate_adapter_writeback_plan(
    manifest,
    replacements_by_file,
    summary,
    failure_entries,
    live_sources=None,
):
    if not any(
        any(replacements_by_line.values())
        for replacements_by_line in replacements_by_file.values()
    ):
        summary['adapter_writeback_status'] = 'empty'
        summary['adapter_writeback_operations'] = 0
        return None, None
    try:
        plan, snapshot = _build_adapter_writeback_plan(
            manifest,
            replacements_by_file,
            live_sources=live_sources,
        )
        rendered_by_file = _normalize_adapter_rendered_files(
            render_writeback_plan(plan, snapshot.project.source_documents)
        )
        _require_adapter_render_targets(replacements_by_file, rendered_by_file)
    except (ValueError, OSError) as exc:
        reason_code = getattr(exc, 'reason_code', '') or 'adapter_writeback_block'
        bump_counter(summary['reason_counts'], 'adapter_writeback_block')
        failure_entries.append(
            make_failure_entry(
                manifest,
                f'Adapter writeback plan blocked: {exc}',
                reason_code='adapter_writeback_block',
                adapter_reason_code=reason_code,
            )
        )
        summary['adapter_writeback_status'] = 'block'
        summary['adapter_writeback_plan_digest'] = ''
        summary['adapter_writeback_operations'] = 0
        return None, None

    summary['adapter_writeback_status'] = 'pass'
    summary['adapter_writeback_plan_digest'] = plan.plan_digest
    summary['adapter_writeback_operations'] = len(plan.operations)
    return plan, snapshot


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


def _manifest_occurrence(project, chunk, item, mode, item_index):
    file_key = str(chunk.get('file_rel_path') or '')
    unit = translation_core.unit_from_manifest_item(item, mode=mode, chunk=chunk)
    block_name = str(item.get('block_name') or '_global')
    block_occurrence = int(item.get('block_occurrence') or 1)
    ordinal = int(item.get('block_index') or item.get('ordinal') or 0)
    item_id = str(item.get('id') or '')
    identity_prefix = file_key.replace('\\', '/') + ':'
    if item_id.startswith(identity_prefix):
        try:
            block_token, ordinal_text, _source_hash = item_id[len(identity_prefix):].rsplit(':', 2)
            if '#' in block_token:
                parsed_block, parsed_occurrence = block_token.rsplit('#', 1)
                block_name = parsed_block or block_name
                block_occurrence = max(1, int(parsed_occurrence))
            else:
                block_name = block_token or block_name
            ordinal = int(ordinal_text)
        except (TypeError, ValueError):
            pass
    locator = OpaqueLocator(
        engine='renpy',
        locator_schema_version=1,
        locator={
            'file_rel_path': file_key,
            'translate_block': block_name,
            'block_occurrence': block_occurrence,
            'ordinal': ordinal,
            'line_hint': int(item.get('line_number') or int(item.get('line') or 0) + 1),
            'start_col_hint': int(item.get('start') or 0),
            'end_col_hint': int(item.get('end') or 0),
            'source_marker_kind': str(item.get('source_marker_kind') or 'direct_source'),
            'candidate_ordinal': int(item.get('candidate_ordinal') or item_index + 1),
        },
    )
    return Occurrence(
        occurrence_id=f'manifest:{file_key}:{item_id or item_index}',
        engine='renpy',
        project_snapshot_fingerprint=project.project_snapshot_fingerprint,
        content_fingerprint=str(item.get('content_fingerprint') or ''),
        candidate_id=f'manifest:{item_id or item_index}',
        locator=locator,
        unit=unit,
    )


def relocate_v2_chunk_items(manifest, chunk, scanned_units_by_file, mode):
    if not is_v2_manifest(manifest):
        return []
    file_key = chunk['file_rel_path']
    if file_key not in scanned_units_by_file:
        adapter = RenPyAdapter(legacy_module=legacy)
        project = adapter.discover_project(_adapter_request_for_manifest(manifest, (file_key,)))
        scanned_units_by_file[file_key] = {
            'adapter': adapter,
            'project': project,
        }

    context = scanned_units_by_file[file_key]
    adapter = context['adapter']
    project = context['project']
    items = list(chunk.get('items') or [])
    originals = tuple(
        _manifest_occurrence(project, chunk, item, mode, item_index)
        for item_index, item in enumerate(items)
    )
    relocation = adapter.relocate_occurrences(
        project,
        originals,
        project.source_documents,
    )
    relocated_by_id = {
        occurrence.unit.id: occurrence
        for occurrence in relocation.occurrences
    }
    missing_items = []
    context['relocated_by_id'] = relocated_by_id
    for item in items:
        relocated = relocated_by_id.get(str(item.get('id') or ''))
        if relocated is None:
            missing_items.append(item)
            continue
        unit = relocated.unit
        item['line'] = unit.line
        item['line_number'] = unit.line + 1
        item['start'] = unit.start
        item['end'] = unit.end
        item['prefix'] = unit.prefix
        item['quote'] = unit.quote
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
        failure_entries.append(
            make_failure_entry(
                manifest,
                'V2 relocation missing for result item',
                file_rel_path=chunk.get('file_rel_path', ''),
                item_id=item_id,
                line=item.get('line'),
                text=item.get('source', item.get('text', '')),
                key=key or chunk.get('key', ''),
                reason_code='v2_relocation_missing',
            )
        )
    return missing_ids


def validate_batch_item_translation(
    scanned_units_by_file,
    file_key,
    item,
    source_text,
    translated_text,
):
    """Validate a relocated v2 item through its engine adapter."""

    context = scanned_units_by_file.get(file_key) or {}
    occurrence = (context.get('relocated_by_id') or {}).get(str(item.get('id') or ''))
    adapter = context.get('adapter')
    if occurrence is None or adapter is None:
        valid, reason = legacy.validate_translation(source_text, translated_text)
        return valid, reason, (), ()
    validation = adapter.validate_translation(occurrence, translated_text)
    if validation.status == 'pass':
        return True, 'OK', validation.reason_codes, validation.diagnostics
    reason = ', '.join(validation.reason_codes) or 'adapter.validation.block'
    return False, reason, validation.reason_codes, validation.diagnostics


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
            item_map = {
                str(item.get('id') or ''): item
                for item in active_chunk_items
            }
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
            if not response_text and not isinstance(row.get('normalized_response'), dict):
                summary['missing_response_chunks'] += 1
                bump_counter(
                    summary['reason_counts'],
                    translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
                )
                for item in active_chunk_items:
                    failure_entries.append(make_failure_entry(
                        manifest,
                        'Missing text in response payload',
                        reason_code=translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
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
                payload = result_row_contract_payload(row)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_REVISION,
                    active_chunk_items,
                )
                persisted_reason_deltas = record_result_row_contract_reasons(
                    summary,
                    row,
                    contract,
                )
                result_items = contract.items
            except Exception as exc:
                summary['partial_chunks'] += 1
                reason_name = (
                    'truncated_output'
                    if finish_reason == 'MAX_TOKENS'
                    else contract_error_reason(exc, 'failed_to_parse_revision_json')
                )
                bump_counter(summary['reason_counts'], reason_name)
                for item in active_chunk_items:
                    failure_entries.append(make_failure_entry(
                        manifest,
                        f'Failed to parse revision JSON: {exc}',
                        reason_code=reason_name,
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

            if contract.retry_ids or contract.issues or persisted_reason_deltas:
                summary['partial_chunks'] += 1
            if contract.retry_ids:
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_revision_items'
                bump_counter(summary['reason_counts'], reason_name)

            contract_failures = result_row_contract_failure_entries(
                manifest,
                chunk,
                row,
                contract,
                item_map,
                persisted_reason_deltas,
                finish_reason,
                usage_metadata,
            )
            failure_entries.extend(contract_failures)
            contract_failure_ids = {
                str(failure.get('id') or '')
                for failure in contract_failures
                if str(failure.get('id') or '') in item_map
            }

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
                valid, reason, adapter_reason_codes, adapter_diagnostics = (
                    validate_batch_item_translation(
                        scanned_units_by_file,
                        chunk['file_rel_path'],
                        target_item,
                        source_text,
                        revised_translation,
                    )
                )
                if not valid and _adapter_target_language_policy_allows(
                    manifest,
                    chunk,
                    target_item,
                    source_text,
                    revised_translation,
                    reason=reason,
                    reason_codes=adapter_reason_codes,
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
                        adapter_reason_codes=list(adapter_reason_codes),
                        adapter_diagnostics=list(adapter_diagnostics),
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

            missing_ids = (
                set(item_map.keys()) - seen_ids - contract_failure_ids
            )
            for missing_id in sorted(missing_ids):
                item = item_map[missing_id]
                failure_entries.append(make_failure_entry(
                    manifest,
                    'Response missing expected id',
                    file_rel_path=chunk['file_rel_path'],
                    item_id=item['id'],
                    line=item['line'],
                    text=item.get('source', item.get('text', '')),
                    key=key,
                    finish_reason=finish_reason,
                    usage_metadata=usage_metadata,
                    reason_code='response_missing_expected_id',
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
        _plan, _snapshot = _validate_adapter_writeback_plan(
            manifest,
            replacements_by_file,
            summary,
            failure_entries,
        )
        if summary.get('adapter_writeback_status') == 'block':
            replacements_by_file.clear()
            revised_lines_by_file.clear()
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
    if 'quality_gate' in summary:
        quality_gate = summary.get('quality_gate') or {}
        print(
            f"Quality gate: {quality_gate.get('decision', 'unknown')} "
            f"(warnings={quality_gate.get('warning_count', 0)}, "
            f"blockers={quality_gate.get('blocker_count', 0)})"
        )
        writeback_gate = summary.get('writeback_gate') or {}
        print(
            f"Revision writeback gate: {writeback_gate.get('decision', 'unknown')}"
        )
    if summary.get('quality_findings_path'):
        print(f"Quality findings: {summary['quality_findings_path']}")


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
        os.path.join(manifest.get('_package_dir', ''), 'quality_findings.jsonl'),
    }
    for manifest_key in ('_manifest_path', 'input_jsonl_path', 'result_jsonl_path'):
        value = manifest.get(manifest_key)
        if value:
            reserved_paths.add(value)
    normalized_reserved = {_normalized_abs_path(path) for path in reserved_paths if path}
    for output_path in (jsonl_path, markdown_path):
        if _normalized_abs_path(output_path) in normalized_reserved:
            raise SystemExit(f'Revision preview output would overwrite reserved package file: {output_path}')


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def _revision_manifest_identity(manifest):
    """Stable fingerprint of the revision manifest content bound by preview."""
    keys = (
        'mode', 'manifest_version', 'version', 'core_schema_version', 'display_name',
        'job_name', 'created_at', 'execution', 'batch_model', 'model',
        'base_dir', 'tl_dir', 'target_language', 'language',
        'input_jsonl_path', 'result_jsonl_path',
        'settings', 'revision_settings',
        'summary', 'files', 'chunks', 'final_review_source',
    )
    payload = {key: manifest.get(key) for key in keys}
    # Policy drift is enforced by ``_revision_quality_staleness``, not here.
    # Preserve the v1 fingerprint for pre-proposal revision/final-review
    # packages.  Proposal manifests bind their immutable import provenance and
    # eligibility state without adding a null field to every legacy payload.
    if isinstance(manifest.get('proposal_import'), dict):
        payload['proposal_import'] = manifest['proposal_import']
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()


def _revision_source_snapshots(manifest):
    """Per-file SHA-256 of every source file named by the manifest.

    Missing files are recorded as ``None`` so a file appearing or disappearing
    between preview and apply is detected as a snapshot change.
    """
    snapshots = {}
    for file_key, file_info in (manifest.get('files') or {}).items():
        try:
            file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        except SystemExit:
            snapshots[str(file_key)] = None
            continue
        snapshots[str(file_key)] = (
            _sha256_file(file_path) if os.path.isfile(file_path) else None
        )
    return snapshots


def _mark_revision_apply_blocked(manifest, reason, message):
    """Persist a blocked apply outcome before refusing the command."""
    now = datetime.now().isoformat(timespec='seconds')
    manifest['revision_apply_state'] = 'blocked'
    manifest['revision_apply_checked_at'] = now
    manifest['revision_apply_blocked_reason'] = reason
    manifest['revision_apply_message'] = message
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    print(f'Revision apply state: blocked')
    print(f'Revision apply reason: {reason}')
    raise SystemExit(f'Revision apply refused: {message}')


def _require_valid_revision_preview(manifest):
    """Require a valid, matching preview before any revision writeback.

    ``--force`` deliberately does not bypass preview staleness, project identity
    or source snapshot checks: it only bypasses the already-applied guard.
    """
    proposal_import = manifest.get('proposal_import')
    if isinstance(proposal_import, dict) and (
        not proposal_import.get('writeback_eligible')
        or proposal_import.get('status') not in {'previewed', 'no_op'}
    ):
        _mark_revision_apply_blocked(
            manifest,
            'proposal_import_not_eligible',
            'proposal import is blocked, partial, stale, or not previewed; re-import valid proposals.',
        )
    if isinstance(proposal_import, dict):
        selection_path = str(proposal_import.get('selection_path') or '').strip()
        selection_sha256 = str(proposal_import.get('selection_sha256') or '').strip()
        if selection_path and selection_sha256:
            if not os.path.isfile(selection_path) or _sha256_file(selection_path) != selection_sha256:
                _mark_revision_apply_blocked(
                    manifest,
                    'selection_changed',
                    'revision proposal selection changed since preview; run confirm-revision-proposals again.',
                )
    preview = manifest.get('last_revision_preview')
    if (
        not isinstance(preview, dict)
        or preview.get('schema_version') != REVISION_PREVIEW_CONTRACT_VERSION
    ):
        _mark_revision_apply_blocked(
            manifest,
            'missing_preview',
            'run preview-revisions before apply-revisions.',
        )

    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        _mark_revision_apply_blocked(
            manifest,
            'results_missing',
            'result JSONL no longer exists; run preview-revisions again.',
        )
    if _sha256_file(result_path) != preview.get('results_sha256'):
        _mark_revision_apply_blocked(
            manifest,
            'results_changed',
            'result JSONL changed since preview; run preview-revisions again.',
        )
    if _revision_manifest_identity(manifest) != preview.get('manifest_identity'):
        _mark_revision_apply_blocked(
            manifest,
            'manifest_changed',
            'manifest changed since preview; run preview-revisions again.',
        )

    try:
        current_project = manifest_project_identity(manifest)
    except Exception as exc:
        _mark_revision_apply_blocked(
            manifest,
            'project_unknown',
            f'cannot resolve project identity: {exc}',
        )
    current_identity = {
        'base_dir': _normalized_abs_path(str(current_project.get('base_dir') or '')),
        'tl_dir': _normalized_abs_path(str(current_project.get('tl_dir') or '')),
        'source': str(current_project.get('source') or ''),
    }
    expected_project = preview.get('project_identity')
    if not isinstance(expected_project, dict):
        _mark_revision_apply_blocked(
            manifest,
            'project_changed',
            'project identity changed since preview; run preview-revisions again.',
        )
    expected_identity = {
        'base_dir': _normalized_abs_path(str(expected_project.get('base_dir') or '')),
        'tl_dir': _normalized_abs_path(str(expected_project.get('tl_dir') or '')),
        'source': str(expected_project.get('source') or ''),
    }
    if current_identity != expected_identity:
        _mark_revision_apply_blocked(
            manifest,
            'project_changed',
            'project identity changed since preview; run preview-revisions again.',
        )

    current_snapshots = _revision_source_snapshots(manifest)
    expected_snapshots = preview.get('source_snapshots')
    if not isinstance(expected_snapshots, dict) or current_snapshots != expected_snapshots:
        _mark_revision_apply_blocked(
            manifest,
            'source_changed',
            'source files changed since preview; run preview-revisions again.',
        )

    quality_staleness = _revision_quality_staleness(manifest, preview)
    if quality_staleness is not None:
        reason, message = quality_staleness
        _mark_revision_apply_blocked(manifest, reason, message)
    return preview


def _revision_quality_staleness(manifest, preview):
    """Return ``(reason, message)`` when revision quality evidence is stale."""

    expected_rule_version = preview.get('quality_rule_schema_version')
    if expected_rule_version is not None and expected_rule_version != (
        translation_quality.QUALITY_RULE_SCHEMA_VERSION
    ):
        return (
            'quality_rules_changed',
            'quality rules changed since revision preview; run preview-revisions again.',
        )
    expected_runtime_policy_digest = preview.get('quality_policy_runtime_digest')
    if expected_runtime_policy_digest and expected_runtime_policy_digest != (
        translation_quality.policy_digest(BATCH_QUALITY_POLICY)
    ):
        return (
            'quality_policy_changed',
            'quality policy changed since revision preview; run preview-revisions again.',
        )
    expected_manifest_policy_digest = preview.get('quality_policy_digest')
    if expected_manifest_policy_digest and expected_manifest_policy_digest != (
        translation_quality.policy_digest(
            translation_quality.effective_policy(manifest)
        )
    ):
        return (
            'quality_policy_changed',
            'manifest quality policy changed since revision preview; run preview-revisions again.',
        )
    quality_findings_path = preview.get('quality_findings_path')
    if quality_findings_path:
        if not os.path.isfile(quality_findings_path):
            return (
                'quality_findings_missing',
                'quality findings report no longer exists; run preview-revisions again.',
            )
        if _sha256_file(quality_findings_path) != preview.get(
            'quality_findings_sha256'
        ):
            return (
                'quality_findings_changed',
                'quality findings changed since revision preview; run preview-revisions again.',
            )
    writeback_gate = preview.get('writeback_gate')
    if isinstance(writeback_gate, dict) and writeback_gate.get(
        'decision'
    ) != translation_quality.GATE_ALLOW:
        return (
            'revision_writeback_gate_denied',
            'revision preview is not allowed to write back; resolve quality blockers or structural blocks and run preview-revisions again.',
        )
    return None


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


def preview_revisions(
    target=None,
    output_jsonl='',
    output_markdown='',
    *,
    update_latest=None,
):
    """Build a revision preview and optionally control the latest pointer."""
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_REVISION, 'preview-revisions')
    replacements_by_file, _lines, _failure_entries, summary, preview_entries = collect_revision_actions(
        manifest,
        validate_sources=True,
    )
    _quality_findings, quality_report_path = run_revision_quality_check(
        manifest,
        summary,
        replacements_by_file,
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

    now = datetime.now().isoformat(timespec='seconds')
    manifest['last_revision_preview_at'] = now
    manifest['last_revision_quality_findings_path'] = quality_report_path
    manifest['last_revision_quality_findings_sha256'] = _sha256_file(
        quality_report_path
    )
    manifest['last_revision_preview'] = {
        'schema_version': REVISION_PREVIEW_CONTRACT_VERSION,
        'generated_at': now,
        'jsonl_path': jsonl_path,
        'markdown_path': markdown_path,
        'results_path': _canonical_abs_path(resolve_manifest_result_path(manifest)),
        'results_sha256': _sha256_file(resolve_manifest_result_path(manifest)),
        'manifest_identity': _revision_manifest_identity(manifest),
        'project_identity': manifest_project_identity(manifest),
        'source_snapshots': _revision_source_snapshots(manifest),
        'quality_findings_path': quality_report_path,
        'quality_findings_sha256': _sha256_file(quality_report_path),
        'quality_findings_count': summary.get('quality_findings_count', 0),
        'quality_findings_digest': summary.get('quality_findings_digest'),
        'quality_gate': summary.get('quality_gate'),
        'writeback_gate': summary.get('writeback_gate'),
        'check_status': summary.get('check_status'),
        'quality_finding_schema_version': summary.get(
            'quality_finding_schema_version'
        ),
        'quality_rule_schema_version': summary.get('quality_rule_schema_version'),
        'quality_policy_digest': summary.get('quality_policy_digest'),
        'quality_policy_runtime_digest': summary.get(
            'quality_policy_runtime_digest'
        ),
        'summary': summary,
    }
    proposal_state = manifest.get('proposal_import')
    if isinstance(proposal_state, dict):
        proposal_state = dict(proposal_state)
        proposal_status = _proposal_status_from_preview_summary(summary)
        history = list(proposal_state.get('history') or [])
        if not history or history[-1] != proposal_status:
            history.append(proposal_status)
        proposal_state['status'] = proposal_status
        proposal_state['history'] = history
        proposal_state['writeback_eligible'] = proposal_status in {
            'previewed',
            'no_op',
        }
        manifest['proposal_import'] = proposal_state
        manifest['last_revision_preview']['manifest_identity'] = (
            _revision_manifest_identity(manifest)
        )
    # A fresh preview invalidates any prior blocked/no_op/partial terminal state:
    # the user may have fixed the blocker and expects the writeback gate to reopen.
    # A prior real writeback is preserved in revision_apply_history.
    for stale_key in (
        'revision_apply_state',
        'revision_apply_checked_at',
        'revision_apply_blocked_reason',
        'revision_apply_message',
    ):
        manifest.pop(stale_key, None)
    if manifest.get('revision_applied_at'):
        history = list(manifest.get('revision_apply_history') or [])
        history.append(
            {
                'applied_at': manifest['revision_applied_at'],
                'summary': dict(manifest.get('revision_apply_summary') or {}),
            }
        )
        manifest['revision_apply_history'] = history
    manifest.pop('revision_applied_at', None)
    manifest.pop('revision_apply_summary', None)
    manifest.pop('last_revision_apply_summary', None)
    should_update_latest = (
        manifest.get('execution') != 'sync'
        if update_latest is None
        else bool(update_latest)
    )
    save_manifest(manifest, update_latest=should_update_latest)
    if manifest.get('final_review_source'):
        import final_review as fr
        import final_review_revision

        final_review_revision.sync_linked_findings(manifest, fr.REVISION_STATE_PREVIEWED)
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
            if not response_text and not isinstance(row.get('normalized_response'), dict):
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
                bump_counter(
                    summary['reason_counts'],
                    translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
                )
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
                            'reason_code': translation_core.CONTRACT_EMPTY_RESPONSE_TEXT,
                            'finish_reason': finish_reason,
                            'usage_metadata': usage_metadata,
                        }
                    )
                continue

            try:
                payload = result_row_contract_payload(row)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_TRANSLATION,
                    chunk_items,
                )
                persisted_reason_deltas = record_result_row_contract_reasons(
                    summary,
                    row,
                    contract,
                )
                result_items = contract.items
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
                reason_name = (
                    'truncated_output'
                    if finish_reason == 'MAX_TOKENS'
                    else contract_error_reason(exc, 'failed_to_parse_model_json')
                )
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
                            'reason_code': reason_name,
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
            item_map = {
                str(item.get('id') or ''): item
                for item in active_chunk_items
            }

            active_retry_ids = set(contract.retry_ids) - relocation_missing_ids
            if active_retry_ids or contract.issues or persisted_reason_deltas:
                summary['partial_chunks'] += 1
            if active_retry_ids:
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_result_items'
                bump_counter(summary['reason_counts'], reason_name)

            contract_failures = result_row_contract_failure_entries(
                manifest,
                chunk,
                row,
                contract,
                item_map,
                persisted_reason_deltas,
                finish_reason,
                usage_metadata,
                ignored_item_ids=relocation_missing_ids,
            )
            failure_entries.extend(contract_failures)
            contract_failure_ids = {
                str(failure.get('id') or '')
                for failure in contract_failures
                if str(failure.get('id') or '') in item_map
            }

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
                valid, reason, adapter_reason_codes, adapter_diagnostics = (
                    validate_batch_item_translation(
                        scanned_units_by_file,
                        chunk['file_rel_path'],
                        target_item,
                        target_unit.text,
                        result_item['translation'],
                    )
                )
                if not valid and _adapter_target_language_policy_allows(
                    manifest,
                    chunk,
                    target_item,
                    target_unit.text,
                    result_item['translation'],
                    reason=reason,
                    reason_codes=adapter_reason_codes,
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
                            'adapter_reason_codes': list(adapter_reason_codes),
                            'adapter_diagnostics': list(adapter_diagnostics),
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

            missing_ids = (
                set(item_map.keys()) - seen_ids - contract_failure_ids
            )
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
                        'error': 'Response missing expected id',
                        'reason_code': 'response_missing_expected_id',
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
        _plan, _snapshot = _validate_adapter_writeback_plan(
            manifest,
            replacements_by_file,
            summary,
            failure_entries,
        )
        if summary.get('adapter_writeback_status') == 'block':
            replacements_by_file.clear()
            translated_lines_by_file.clear()
        summary['failure_items'] = len(failure_entries)
    else:
        summarize_pending_replacements(replacements_by_file, translated_lines_by_file, summary)
    return replacements_by_file, translated_lines_by_file, failure_entries, summary


def collect_quality_subjects(manifest, replacements_by_file, stats=None):
    """Build quality-check subjects from structurally validated replacements.

    Quality rules only inspect items that already passed translation contract
    validation and source validation.  Structural failures continue to be
    reported through the existing check failure report and writeback gate.

    When *stats* is provided it receives collection counters so callers can
    surface silently unmapped actions instead of skipping them invisibly.
    """

    item_index = {}
    counters = {
        'quality_action_items': 0,
        'quality_subject_items': 0,
        'quality_unmatched_items': 0,
    }
    for chunk in manifest.get('chunks') or []:
        if not isinstance(chunk, dict):
            continue
        file_rel_path = str(chunk.get('file_rel_path') or '')
        for item in chunk.get('items') or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('id') or '')
            item_index[(file_rel_path, item_id)] = (chunk, item)
            item_index[(str(chunk.get('key') or ''), item_id)] = (chunk, item)

    subjects = []
    for file_key, replacements_by_line in replacements_by_file.items():
        for line_index, actions in replacements_by_line.items():
            for action in actions or []:
                counters['quality_action_items'] += 1
                if not isinstance(action, (tuple, list)) or len(action) < 6:
                    counters['quality_unmatched_items'] += 1
                    continue
                replacement = str(action[2] or '')
                expected_text = str(action[5] or '') if len(action) > 5 else ''
                item_id = str(action[6] or '') if len(action) > 6 else ''
                chunk_key = str(action[7] or '') if len(action) > 7 else ''
                chunk, item = item_index.get(
                    (str(file_key), item_id),
                    item_index.get((chunk_key, item_id), (None, None)),
                )
                if chunk is None or item is None:
                    counters['quality_unmatched_items'] += 1
                    continue
                try:
                    unit = translation_core.unit_from_manifest_item(
                        item,
                        mode=translation_core.MODE_TRANSLATION,
                        chunk=chunk,
                    )
                    if unit is None:
                        raise ValueError('unit_from_manifest_item returned None')
                    subject = {
                        'item_id': item_id,
                        'file_rel_path': str(file_key),
                        'line': unit.line,
                        'line_number': unit.display_line_number,
                        'start': unit.start,
                        'end': unit.end,
                        'source': expected_text or unit.text,
                        'translation': replacement,
                        'speaker_id': unit.speaker_id,
                        'speaker_name': unit.speaker_name,
                    }
                except (AttributeError, TypeError, ValueError):
                    # Quality inspection is additive; a malformed item must not
                    # take down the structural check workflow.  Count it and
                    # let the existing failure contract own diagnostics.
                    counters['quality_unmatched_items'] += 1
                    continue
                subjects.append(subject)
    counters['quality_subject_items'] = len(subjects)
    if isinstance(stats, dict):
        stats.update(counters)
    return subjects


def write_quality_findings(manifest, findings):
    path = os.path.join(manifest.get('_package_dir', ''), 'quality_findings.jsonl')
    atomic_write_jsonl(
        path,
        [translation_quality.normalize_finding(finding) for finding in findings],
        ensure_ascii=False,
    )
    return path


def collect_revision_quality_subjects(manifest, replacements_by_file, stats=None):
    """Build quality subjects from structurally validated revision actions.

    Revision quality inspection uses the same stable identity fields as Batch
    check: item ID, file, line, source, and the text that is about to be
    written back.  Quality findings never replace the structural writeback
    contract; they are attached only to actions that already passed it.
    """

    item_index = {}
    counters = {
        'quality_action_items': 0,
        'quality_subject_items': 0,
        'quality_unmatched_items': 0,
    }
    for chunk in manifest.get('chunks') or []:
        if not isinstance(chunk, dict):
            continue
        file_rel_path = str(chunk.get('file_rel_path') or '')
        for item in chunk.get('items') or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('id') or '')
            item_index[(file_rel_path, item_id)] = (chunk, item)
            item_index[(str(chunk.get('key') or ''), item_id)] = (chunk, item)

    subjects = []
    for file_key, replacements_by_line in replacements_by_file.items():
        for line_index, actions in replacements_by_line.items():
            for action in actions or []:
                counters['quality_action_items'] += 1
                if not isinstance(action, (tuple, list)) or len(action) < 6:
                    counters['quality_unmatched_items'] += 1
                    continue
                replacement = str(action[2] or '')
                item_id = str(action[6] or '') if len(action) > 6 else ''
                chunk_key = str(action[7] or '') if len(action) > 7 else ''
                chunk, item = item_index.get(
                    (str(file_key), item_id),
                    item_index.get((chunk_key, item_id), (None, None)),
                )
                if chunk is None or item is None:
                    counters['quality_unmatched_items'] += 1
                    continue
                try:
                    unit = translation_core.unit_from_manifest_item(
                        item,
                        mode=translation_core.MODE_REVISION,
                        chunk=chunk,
                    )
                    if unit is None:
                        raise ValueError('unit_from_manifest_item returned None')
                    subject = {
                        'item_id': item_id,
                        'file_rel_path': str(file_key),
                        'line': unit.line,
                        'line_number': unit.display_line_number,
                        'start': unit.start,
                        'end': unit.end,
                        'source': unit.source_text,
                        'translation': replacement,
                        'speaker_id': unit.speaker_id,
                        'speaker_name': unit.speaker_name,
                    }
                except (AttributeError, TypeError, ValueError):
                    counters['quality_unmatched_items'] += 1
                    continue
                subjects.append(subject)
    counters['quality_subject_items'] = len(subjects)
    if isinstance(stats, dict):
        stats.update(counters)
    return subjects


def run_revision_quality_check(
    manifest,
    summary,
    replacements_by_file,
    *,
    apply_stage=False,
):
    """Run shared mechanical rules on revision writeback candidates.

    Like Batch ``check``, revision consumes the current runtime policy
    (``BATCH_QUALITY_POLICY``) and then refreshes the manifest snapshot so the
    persisted policy matches the findings just produced.  A policy change after
    preview is detected through the separately persisted runtime digest.
    """

    collection_stats = {}
    quality_subjects = collect_revision_quality_subjects(
        manifest,
        replacements_by_file,
        stats=collection_stats,
    )
    glossary_path = str(
        manifest.get('glossary_file')
        or os.environ.get('GLOSSARY_FILE')
        or getattr(legacy, 'GLOSSARY_FILE', '')
        or ''
    )
    quality_glossary_map = {}
    quality_glossary_base = ''
    for base_dir in (
        str(manifest.get('_package_dir') or ''),
        str(manifest.get('base_dir') or ''),
    ):
        if not base_dir:
            continue
        candidate = translation_quality.load_glossary_map(
            glossary_path,
            base_dir=base_dir,
        )
        if candidate:
            quality_glossary_map = candidate
            quality_glossary_base = base_dir
            break
    if not quality_glossary_map and glossary_path:
        quality_glossary_map = translation_quality.load_glossary_map(glossary_path)

    policy = translation_quality.normalize_policy(BATCH_QUALITY_POLICY)
    manifest['quality_policy'] = policy
    quality_findings = translation_quality.check_quality(
        quality_subjects,
        policy=policy,
        glossary_map=quality_glossary_map,
    )
    if int(collection_stats.get('quality_unmatched_items') or 0) > 0:
        quality_findings.append(
            translation_quality.make_unmatched_quality_subject_finding(
                collection_stats
            )
        )
    if apply_stage:
        quality_report_path = os.path.join(
            manifest.get('_package_dir', ''),
            'quality_findings.apply.jsonl',
        )
        atomic_write_jsonl(
            quality_report_path,
            [
                translation_quality.normalize_finding(finding)
                for finding in quality_findings
            ],
            ensure_ascii=False,
        )
    else:
        quality_report_path = write_quality_findings(
            manifest,
            quality_findings,
        )
    quality_gate = translation_quality.summarize_quality_gate(
        quality_findings,
        acknowledged_ids=manifest.get('quality_acknowledged_finding_ids') or [],
    )
    quality_reason_counts = {}
    for finding in quality_findings:
        bump_counter(
            quality_reason_counts,
            finding.get('reason_code') or 'quality.unknown',
        )

    summary.update(collection_stats)
    summary['quality_glossary_path'] = glossary_path
    summary['quality_glossary_base'] = quality_glossary_base
    summary['quality_glossary_entries'] = len(quality_glossary_map)
    summary['quality_glossary_loaded'] = bool(
        not glossary_path or quality_glossary_map
    )
    summary['quality_findings_count'] = len(quality_findings)
    summary['quality_reason_counts'] = quality_reason_counts
    summary['quality_findings_path'] = quality_report_path
    summary['quality_findings_sha256'] = _sha256_file(quality_report_path)
    summary['quality_findings_digest'] = translation_quality.findings_digest(
        quality_findings
    )
    summary['quality_gate'] = quality_gate
    summary['quality_finding_schema_version'] = (
        translation_quality.QUALITY_FINDING_SCHEMA_VERSION
    )
    summary['quality_rule_schema_version'] = (
        translation_quality.QUALITY_RULE_SCHEMA_VERSION
    )
    summary['quality_policy_digest'] = translation_quality.policy_digest(policy)
    summary['quality_policy_runtime_digest'] = translation_quality.policy_digest(
        BATCH_QUALITY_POLICY
    )

    summarize_revision_writeback_gate(summary)
    summary['has_warnings'] = bool(quality_gate.get('has_warnings'))
    summary['check_status'] = translation_quality.overall_check_status(
        summary['writeback_gate'],
        quality_gate,
    )
    return quality_findings, quality_report_path


def summarize_revision_writeback_gate(summary):
    """Compute revision writeback gate from structural and quality blockers."""

    quality_gate = summary.get('quality_gate') or {}
    structural_block = summary.get('adapter_writeback_status') == 'block'
    quality_blocker_count = int(quality_gate.get('blocker_count') or 0)
    can_apply = (not structural_block) and quality_blocker_count == 0
    summary['writeback_gate'] = {
        'decision': (
            translation_quality.GATE_ALLOW
            if can_apply
            else translation_quality.GATE_DENY
        ),
        'can_apply': can_apply,
        'blocker_count': quality_blocker_count + (1 if structural_block else 0),
        'structural_blocker_count': 1 if structural_block else 0,
        'quality_blocker_count': quality_blocker_count,
    }
    return summary['writeback_gate']


def resolve_quality_findings_path(manifest):
    return translation_quality.resolve_quality_findings_path(
        manifest,
        package_dir=str((manifest or {}).get('_package_dir') or ''),
        manifest_path=str((manifest or {}).get('_manifest_path') or ''),
    )


def read_quality_findings(manifest):
    path = resolve_quality_findings_path(manifest)
    if not path or not os.path.exists(path):
        raise cli_contract.MachineContractError(
            f'Quality findings report is not available: {path or "missing path"}.',
            code_name='QUALITY_FINDINGS_UNAVAILABLE',
            suggested_action='rerun_check',
            details={'quality_findings_path': path},
        )
    findings = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise cli_contract.MachineContractError(
                        (
                            f'Quality findings report contains invalid JSON at line '
                            f'{line_number}: {exc}.'
                        ),
                        code_name='INVALID_QUALITY_FINDINGS_JSON',
                        suggested_action='rerun_check',
                        details={'quality_findings_path': path, 'line': line_number},
                    ) from exc
                if not isinstance(payload, dict):
                    raise cli_contract.MachineContractError(
                        (
                            f'Quality findings report line {line_number} is not a '
                            f'JSON object.'
                        ),
                        code_name='INVALID_QUALITY_FINDINGS_JSON',
                        suggested_action='rerun_check',
                        details={'quality_findings_path': path, 'line': line_number},
                    )
                findings.append(translation_quality.normalize_finding(payload))
    except OSError as exc:
        raise cli_contract.MachineContractError(
            f'Quality findings report could not be read: {path} ({exc}).',
            code_name='QUALITY_FINDINGS_UNAVAILABLE',
            suggested_action='rerun_check',
            details={'quality_findings_path': path},
        ) from exc
    return findings


def quality_acknowledge_command(
    target=None,
    *,
    finding_ids=(),
    all_findings=False,
    unack=False,
):
    """Execute quality-ack / quality-unack and return its structured result."""

    command_name = 'quality-unack' if unack else 'quality-ack'
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, command_name)
    require_manifest_project_match(manifest, command_name)
    findings = read_quality_findings(manifest)
    old_ids = {
        str((finding_id or '')).strip()
        for finding_id in manifest.get('quality_acknowledged_finding_ids') or []
        if str((finding_id or '')).strip()
    }
    old_gate = translation_quality.summarize_quality_gate(
        findings,
        acknowledged_ids=old_ids,
    )
    previous_acknowledged_finding_ids = sorted(old_ids)
    applied = translation_quality.apply_manifest_quality_acknowledgement(
        manifest,
        findings,
        finding_ids=finding_ids,
        all_findings=all_findings,
        unack=unack,
    )
    manifest = applied['manifest']
    selected_ids = applied['selected_ids'] if (finding_ids or all_findings) else set()
    unmatched = applied['unmatched'] if (finding_ids or all_findings) else []
    new_gate = applied['quality_gate']
    new_ids = {
        str((finding_id or '')).strip()
        for finding_id in manifest.get('quality_acknowledged_finding_ids') or []
        if str((finding_id or '')).strip()
    }
    if new_ids != old_ids:
        save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    return {
        'manifest': manifest,
        'findings': findings,
        'old_gate': old_gate,
        'new_gate': new_gate,
        'selected_ids': selected_ids,
        'unmatched': unmatched,
        'previous_acknowledged_finding_ids': previous_acknowledged_finding_ids,
        'acknowledged_finding_ids': list(
            manifest.get('quality_acknowledged_finding_ids') or []
        ),
    }


def print_quality_acknowledgement_summary(manifest, findings, gate, unmatched):
    print(f"Manifest: {manifest['_manifest_path']}")
    print(f"Quality findings report: {resolve_quality_findings_path(manifest)}")
    print(f"Quality gate: {gate.get('decision', 'unknown')}")
    print(f"Quality warnings: {gate.get('warning_count', 0)}")
    print(f"Quality blockers: {gate.get('blocker_count', 0)}")
    print(f"Acknowledged warnings: {gate.get('acknowledged_count', 0)}")
    unacknowledged_warnings = max(
        0,
        int(gate.get('warning_count') or 0) - int(gate.get('acknowledged_count') or 0),
    )
    print(f"Unacknowledged warnings: {unacknowledged_warnings}")
    acknowledged_ids = {
        str((finding_id or '')).strip()
        for finding_id in manifest.get('quality_acknowledged_finding_ids') or []
    }
    shown = 0
    for finding in findings:
        if finding.get('disposition') != translation_quality.DISPOSITION_WARNING:
            continue
        finding_id = str((finding.get('finding_id') or '')).strip()
        if finding_id in acknowledged_ids:
            continue
        shown += 1
        file_text = finding.get('file') or ''
        line = finding.get('line')
        location = f"{file_text}:{line}" if file_text else finding_id
        print(
            f"- [{severity_label_for_quality(finding.get('severity'))}] "
            f"{finding.get('reason_code')} {location}"
        )
    if not shown and unacknowledged_warnings == 0:
        print("No unacknowledged warnings.")
    for finding_id in unmatched:
        print(f"Ignored unknown finding ID: {finding_id}")


def severity_label_for_quality(severity):
    text = str(severity or '').strip().lower()
    if text == 'high':
        return '高'
    if text == 'medium':
        return '中'
    if text == 'low':
        return '低'
    return text or '未知'


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

    writeback_gate = summary.get('writeback_gate')
    if isinstance(writeback_gate, dict):
        print(f"Writeback gate: {writeback_gate.get('decision', 'unknown')}")
        print(f"Writeback blockers: {writeback_gate.get('blocker_count', 0)}")

    quality_gate = summary.get('quality_gate')
    if isinstance(quality_gate, dict):
        print(f"Quality gate: {quality_gate.get('decision', 'unknown')}")
        if summary.get('quality_glossary_path'):
            print(f"Quality glossary entries: {summary.get('quality_glossary_entries', 0)}")
            if not summary.get('quality_glossary_loaded'):
                print(f"Quality glossary not loaded: {summary['quality_glossary_path']}")
        if summary.get('quality_subject_items') is not None:
            print(f"Quality subjects: {summary.get('quality_subject_items', 0)}")
        if summary.get('quality_unmatched_items'):
            print(f"Quality unmatched items: {summary['quality_unmatched_items']}")
        if summary.get('quality_coverage_complete') is False:
            print("Quality coverage: incomplete")
        print(f"Quality warnings: {quality_gate.get('warning_count', 0)}")
        print(f"Quality blockers: {quality_gate.get('blocker_count', 0)}")
        print(f"Acknowledged warnings: {quality_gate.get('acknowledged_count', 0)}")
        quality_reason_counts = summary.get('quality_reason_counts') or {}
        if quality_reason_counts:
            print('Quality categories:')
            for name in sorted(quality_reason_counts):
                print(f"- {name}: {quality_reason_counts[name]}")
    if summary.get('check_status'):
        print(f"Check status: {summary['check_status']}")


def probe_requests(target=None, limit=3, offset=0, api_key_index=None):
    """Probe only request rows bound to non-empty manifest translation chunks."""
    manifest = load_manifest(target)
    rows = load_request_rows(manifest)
    if offset < 0:
        offset = 0
    if limit <= 0:
        raise SystemExit('--limit must be greater than 0.')
    sample = rows[offset:offset + limit]
    if not sample:
        raise SystemExit('No request rows available for the requested probe range.')

    chunks_by_key = {
        str(chunk.get('key') or ''): chunk
        for chunk in manifest.get('chunks') or []
        if isinstance(chunk, dict) and str(chunk.get('key') or '')
    }
    sample_chunks = []
    for index, row in enumerate(sample, start=offset + 1):
        key = str(row.get('key') or '') if isinstance(row, dict) else ''
        chunk = chunks_by_key.get(key)
        if chunk is None:
            raise cli_contract.MachineContractError(
                f'Probe request row #{index} has no matching manifest chunk: {key or "(missing)"}',
                code_name='PROBE_REQUEST_CHUNK_MISSING',
                suggested_action='rebuild_batch_package',
                details={'row': index, 'key': key},
            )
        items = chunk.get('items')
        if not isinstance(items, list) or not items:
            raise cli_contract.MachineContractError(
                f'Probe request row #{index} references an empty manifest chunk: {key}',
                code_name='PROBE_REQUEST_CHUNK_EMPTY',
                suggested_action='rebuild_batch_package',
                details={'row': index, 'key': key},
            )
        sample_chunks.append(chunk)

    usage_run_id = model_usage_ledger.new_run_id('probe')
    usage_operation_id = (
        'probe-manifest-'
        + hashlib.sha256(
            str(manifest.get('_manifest_path') or '').encode('utf-8')
        ).hexdigest()[:20]
    )
    routing_plan = resolve_manifest_routing_plan(manifest)
    probe_route = route_for_manifest(routing_plan, manifest)
    require_valid_routing_plan(routing_plan, {probe_route.stage})
    summary = {
        'sample_count': len(sample),
        'parse_ok': 0,
        'full_item_match': 0,
        'max_tokens': 0,
        'missing_text': 0,
        'request_errors': 0,
    }
    probe_results = []

    for index, (row, chunk) in enumerate(zip(sample, sample_chunks), start=1):
        key = str(row.get('key') or '')
        request_payload = row.get('request') or {}
        model_name = route_model(routing_plan, probe_route)
        config = filter_gemini_generation_config(
            model_name,
            request_payload.get('generation_config') or {},
        )
        config['timeout'] = SYNC_TIMEOUT_SECONDS
        system_instruction = request_payload.get('system_instruction')
        if system_instruction:
            config['system_instruction'] = system_instruction
        safety_settings = request_payload.get('safety_settings')
        if safety_settings:
            config['safety_settings'] = safety_settings

        chunk_items = chunk['items']
        expected_items = len(chunk_items)
        parse_ok = False
        parsed_items = 0
        parse_error = ''
        finish_reason = ''
        usage_metadata = {}
        response_text = ''
        result = None
        try:
            probe_payload = dict(request_payload)
            probe_payload['generation_config'] = config
            raw = run_sync_request(
                probe_payload,
                probe_route,
                plan=routing_plan,
                api_key_index=api_key_index,
            )
            finish_reason = raw.get('finish_reason') or ''
            usage_metadata = dict(raw.get('usage_metadata') or {})
            response_text = raw.get('response_text') or ''
            result = raw
        except Exception as exc:
            summary['request_errors'] += 1
            parse_error = str(exc)
        else:
            record_generation_usage_best_effort(
                task_mode='analysis',
                stage='probe',
                result=result if isinstance(result, dict) else _sync_result_to_dict(result),
                operation_id=usage_operation_id,
                run_id=usage_run_id,
                source_key=str(key),
                thinking_level=str((manifest.get('settings') or {}).get('thinking_level') or ''),
                source={
                    'kind': 'probe_response',
                    'manifest_path': str(manifest.get('_manifest_path') or ''),
                    'row_key': str(key),
                    'sample_index': index,
                },
            )
        if response_text:
            try:
                payload = parse_json_payload(response_text)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_TRANSLATION,
                    chunk_items,
                )
                parsed_items = len(contract.items)
                parse_ok = contract.complete
                if contract.issues:
                    parse_error = ', '.join(sorted(contract.reason_counts()))
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


def _resolve_durable_sync_store(target):
    """Return a durable store for an explicit RUN selector/path, else ``None``."""
    from sync_run_contracts import validate_run_id
    from sync_run_store import SyncRunStore

    raw = str(target or '').strip()
    if not raw:
        return None
    if validate_run_id(raw):
        return SyncRunStore(_durable_sync_root_dir(), raw)
    candidate = Path(raw).resolve()
    if candidate.is_file() and candidate.name == 'state.sqlite3':
        candidate = candidate.parent
    if not candidate.is_dir() or not validate_run_id(candidate.name):
        return None
    root = Path(_durable_sync_root_dir()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise cli_contract.MachineContractError(
            'Durable Sync run path must stay inside the active project log directory.',
            code_name='SYNC_RUN_PATH_OUTSIDE_PROJECT',
            suggested_action='pass_active_project_run_id',
        )
    return SyncRunStore(root, candidate.name)


def _verified_store_artifact(store, kind):
    row = store.get_artifact(kind=kind)
    if row is None:
        raise cli_contract.MachineContractError(
            f'Durable Sync artifact is missing: {kind}.',
            code_name='SYNC_RUN_ARTIFACT_MISSING',
            suggested_action='resume_run',
            details={'run_id': store.run_id, 'artifact': kind},
        )
    path = store.resolve_artifact_path(row['relative_path'])
    if not path.is_file() or file_sha256(path) != str(row['sha256']):
        raise cli_contract.MachineContractError(
            f'Durable Sync artifact failed hash validation: {kind}.',
            code_name='SYNC_RUN_ARTIFACT_STALE',
            suggested_action='resume_run',
            details={'run_id': store.run_id, 'artifact': kind},
        )
    return {'path': str(path), 'sha256': str(row['sha256'])}


def _build_durable_sync_check_manifest(store):
    from sync_result_export import export_run_artifacts
    from sync_run_contracts import RunStatus

    run = store.get_run()
    if RunStatus(str(run['status'])) not in {
        RunStatus.CANCELLED,
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_ERRORS,
        RunStatus.FAILED,
    }:
        raise cli_contract.MachineContractError(
            'Durable Sync check requires a terminal run.',
            code_name='SYNC_RUN_NOT_TERMINAL',
            suggested_action='resume_run',
            details={'run_id': store.run_id, 'run_status': run['status']},
        )
    violations = store.verify_integrity()
    if violations:
        raise cli_contract.MachineContractError(
            'Durable Sync run failed integrity verification.',
            code_name='SYNC_RUN_STORAGE_ERROR',
            suggested_action='inspect_durable_sync_run',
            details={'run_id': store.run_id, 'violations': violations[:20]},
        )
    export_run_artifacts(store)
    targets_artifact = _verified_store_artifact(store, 'targets_json')
    results_artifact = _verified_store_artifact(store, 'results_jsonl')
    run_manifest_artifact = _verified_store_artifact(store, 'run_manifest_json')
    try:
        targets = json.loads(Path(targets_artifact['path']).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise cli_contract.MachineContractError(
            f'Durable Sync targets artifact is unreadable: {exc}.',
            code_name='SYNC_RUN_STORAGE_ERROR',
            suggested_action='inspect_durable_sync_run',
        ) from exc
    plan = store.get_plan()['plan']
    if (
        str(targets.get('run_id') or '') != store.run_id
        or str(targets.get('plan_id') or '') != str(plan.get('plan_id') or '')
        or str(targets.get('plan_fingerprint') or '')
        != str(plan.get('plan_fingerprint') or '')
    ):
        raise cli_contract.MachineContractError(
            'Durable Sync targets artifact does not match the frozen run.',
            code_name='SYNC_RUN_STORAGE_ERROR',
            suggested_action='inspect_durable_sync_run',
        )
    chunks = []
    for raw_chunk in targets.get('chunks') or []:
        chunk = dict(raw_chunk)
        chunk['mode'] = MANIFEST_MODE_TRANSLATION
        chunks.append(chunk)
    profile = dict(plan.get('model_profile_snapshot') or {})
    manifest_path = store.run_dir / 'check_manifest.json'
    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': MANIFEST_MODE_TRANSLATION,
        'execution': 'sync',
        'durable_sync': True,
        'created_at': str(run.get('created_at') or ''),
        'display_name': store.run_id,
        'batch_model': str(profile.get('model') or profile.get('model_name') or ''),
        'base_dir': str(targets.get('project_root') or ''),
        'tl_dir': str(targets.get('tl_dir') or ''),
        'tl_subdir': legacy.TL_SUBDIR,
        'target_language': str(targets.get('target_language') or legacy.PREP_LANGUAGE),
        'glossary_file': str(targets.get('glossary_file') or ''),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        **translation_quality.manifest_quality_policy_fields(
            runtime_policy=BATCH_QUALITY_POLICY
        ),
        'input_jsonl_path': str(
            (_verified_store_artifact(store, 'requests_jsonl'))['path']
        ),
        'result_jsonl_path': results_artifact['path'],
        'job_state': str(run['status']),
        'translation_plan': plan,
        'settings': {
            'target_size': int((plan.get('chunk_policy') or {}).get('max_items') or 0),
            'target_chars': int((plan.get('chunk_policy') or {}).get('max_chars') or 0),
        },
        'summary': {
            'file_count': len(targets.get('files') or {}),
            'chunk_count': len(chunks),
            'item_count': sum(len(chunk.get('items') or []) for chunk in chunks),
        },
        'files': dict(targets.get('files') or {}),
        'chunks': chunks,
        'durable_sync_source': {
            'run_id': store.run_id,
            'run_status': str(run['status']),
            'plan_id': str(plan.get('plan_id') or ''),
            'plan_fingerprint': str(plan.get('plan_fingerprint') or ''),
            'run_manifest_sha256': run_manifest_artifact['sha256'],
            'results_sha256': results_artifact['sha256'],
            'targets_sha256': targets_artifact['sha256'],
        },
        '_manifest_path': str(manifest_path),
        '_package_dir': str(store.run_dir),
    }
    atomic_write_json(
        manifest_path,
        {key: value for key, value in manifest.items() if not key.startswith('_')},
        ensure_ascii=False,
        indent=2,
    )
    return load_manifest(str(manifest_path))


def _durable_sync_preview_files(manifest, replacements_by_file, translated_by_file):
    quality_subjects = collect_quality_subjects(manifest, replacements_by_file)
    quality_by_file = {}
    for subject in quality_subjects:
        quality_by_file.setdefault(str(subject.get('file_rel_path') or ''), []).append(subject)
    preview_files = []
    for file_key, replacements in replacements_by_file.items():
        plan, snapshot = _build_adapter_writeback_plan(
            manifest,
            {file_key: replacements},
        )
        rendered = _normalize_adapter_rendered_files(
            render_writeback_plan(plan, snapshot.project.source_documents)
        )
        _require_adapter_render_targets({file_key: replacements}, rendered)
        document = next(
            item
            for item in snapshot.project.source_documents
            if _adapter_render_key(item.file_rel_path) == _adapter_render_key(file_key)
        )
        source_text = ''.join(document.lines())
        preview_text = ''.join(rendered[_adapter_render_key(file_key)])
        if document.content.startswith(b'\xef\xbb\xbf'):
            if not source_text.startswith('\ufeff'):
                source_text = '\ufeff' + source_text
            if not preview_text.startswith('\ufeff'):
                preview_text = '\ufeff' + preview_text
        progress_entries = sorted(translated_by_file.get(file_key) or [])
        preview_files.append({
            'relative_path': file_key,
            'source_text': source_text,
            'source_sha256': document.sha256,
            'preview_text': preview_text,
            'progress_entries': progress_entries,
            'translated_items': len(progress_entries),
            'writeback_plan': plan.to_dict(),
            'quality_subjects': quality_by_file.get(file_key, []),
        })
    return preview_files


def _create_checked_durable_sync_preview(store, manifest):
    summary = dict(manifest.get('last_check_summary') or {})
    gate = dict(summary.get('writeback_gate') or {})
    if gate.get('decision') != translation_quality.GATE_ALLOW:
        return ''
    replacements, translated, failures, recheck = collect_result_actions(
        manifest,
        validate_sources=True,
    )
    attach_check_contract(manifest, recheck)
    if failures or (recheck.get('writeback_gate') or {}).get(
        'decision'
    ) != translation_quality.GATE_ALLOW:
        raise cli_contract.MachineContractError(
            'Durable Sync preview revalidation no longer allows writeback.',
            code_name='STALE_CHECK',
            suggested_action='run_check_again',
        )
    preview_files = _durable_sync_preview_files(manifest, replacements, translated)
    bindings = {
        'run_manifest': _verified_store_artifact(store, 'run_manifest_json'),
        'results': _verified_store_artifact(store, 'results_jsonl'),
        'targets': _verified_store_artifact(store, 'targets_json'),
        'check_manifest': {
            'path': str(manifest['_manifest_path']),
            'sha256': file_sha256(manifest['_manifest_path']),
        },
        'check_fingerprint': dict(summary.get('check_fingerprint') or {}),
        'writeback_gate': gate,
    }
    plan = store.get_plan()['plan']
    preview_path, _preview = sync_translation_preview.create_sync_preview(
        log_dir=store.run_dir,
        project_root=manifest['base_dir'],
        tl_dir=manifest['tl_dir'],
        files=preview_files,
        failures=(),
        contract_diagnostics={
            'final_expected': int(summary.get('expected_items') or 0),
            'final_valid': int(summary.get('valid_items') or 0),
            'unresolved_ids': [],
        },
        prompt_context={
            'macro_fingerprint': str(getattr(legacy, 'SYNC_MACRO_FINGERPRINT', '') or '')
        },
        quality_policy=BATCH_QUALITY_POLICY,
        glossary_file=manifest.get('glossary_file') or '',
        translation_plan_payload=plan,
        request_ids=[
            str(row['request_id'])
            for row in store.list_requests()
            if row.get('parent_request_id') is None
        ],
        durable_check_binding=bindings,
    )
    relative = Path(preview_path).resolve().relative_to(store.run_dir.resolve())
    store.put_artifact(
        kind='preview_manifest',
        relative_path=str(relative),
        sha256_digest=file_sha256(preview_path),
        schema_version=sync_translation_preview.VERSION,
    )
    print(f'Durable Sync preview: {preview_path}')
    return preview_path


def check_durable_sync_results(store):
    manifest = _build_durable_sync_check_manifest(store)
    checked = check_results(manifest['_manifest_path'])
    _create_checked_durable_sync_preview(store, checked)
    return checked


def apply_durable_sync_results(store):
    violations = store.verify_integrity()
    if violations:
        raise cli_contract.MachineContractError(
            'Durable Sync run failed integrity verification before apply.',
            code_name='SYNC_RUN_STORAGE_ERROR',
            suggested_action='inspect_durable_sync_run',
            details={'run_id': store.run_id, 'violations': violations[:20]},
        )
    preview = _verified_store_artifact(store, 'preview_manifest')
    applied = legacy.apply_sync_translation_preview(preview['path'])
    relative = Path(preview['path']).resolve().relative_to(store.run_dir.resolve())
    store.put_artifact(
        kind='preview_manifest',
        relative_path=str(relative),
        sha256_digest=file_sha256(preview['path']),
        schema_version=sync_translation_preview.VERSION,
    )
    return applied


def check_results(target=None):
    durable_store = _resolve_durable_sync_store(target)
    if durable_store is not None:
        return check_durable_sync_results(durable_store)
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'check')
    require_manifest_project_match(manifest, 'check')
    replacements_by_file, _translated, failure_entries, summary = collect_result_actions(
        manifest,
        validate_sources=True,
    )
    quality_collection_stats = {}
    quality_subjects = collect_quality_subjects(
        manifest,
        replacements_by_file,
        stats=quality_collection_stats,
    )
    glossary_path = str(
        manifest.get('glossary_file')
        or os.environ.get('GLOSSARY_FILE')
        or getattr(legacy, 'GLOSSARY_FILE', '')
        or ''
    )
    quality_glossary_map = {}
    quality_glossary_base = ''
    for base_dir in (
        str(manifest.get('_package_dir') or ''),
        str(manifest.get('base_dir') or ''),
    ):
        if not base_dir:
            continue
        candidate = translation_quality.load_glossary_map(
            glossary_path,
            base_dir=base_dir,
        )
        if candidate:
            quality_glossary_map = candidate
            quality_glossary_base = base_dir
            break
    if not quality_glossary_map and glossary_path:
        quality_glossary_map = translation_quality.load_glossary_map(glossary_path)
    summary['quality_glossary_path'] = glossary_path
    summary['quality_glossary_base'] = quality_glossary_base
    summary['quality_glossary_entries'] = len(quality_glossary_map)
    summary['quality_glossary_loaded'] = bool(
        not glossary_path or quality_glossary_map
    )
    quality_findings = translation_quality.check_quality(
        quality_subjects,
        manifest=manifest,
        policy=BATCH_QUALITY_POLICY,
        glossary_map=quality_glossary_map,
    )
    quality_coverage_complete = not int(
        quality_collection_stats.get('quality_unmatched_items') or 0
    )
    summary['quality_coverage_complete'] = quality_coverage_complete
    if not quality_coverage_complete:
        quality_findings.append(
            translation_quality.make_unmatched_quality_subject_finding(
                quality_collection_stats
            )
        )
    quality_report_path = write_quality_findings(manifest, quality_findings)
    quality_reason_counts = {}
    for finding in quality_findings:
        bump_counter(
            quality_reason_counts,
            finding.get('reason_code') or 'quality.unknown',
        )
    summary['quality_findings_count'] = len(quality_findings)
    summary.update(quality_collection_stats)
    summary['quality_policy_source'] = 'runtime'
    summary['quality_policy_runtime_digest'] = translation_quality.policy_digest(
        BATCH_QUALITY_POLICY
    )
    manifest_policy = manifest.get('quality_policy')
    if isinstance(manifest_policy, dict):
        summary['quality_policy_manifest_digest'] = translation_quality.policy_digest(
            manifest_policy
        )
    summary['quality_reason_counts'] = quality_reason_counts
    summary['quality_findings_path'] = quality_report_path
    attach_check_contract(manifest, summary, quality_findings=quality_findings)
    check_report_path = write_check_failure_report(manifest, failure_entries)
    manifest['last_check_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['last_check_summary'] = summary
    manifest['last_check_report_path'] = check_report_path
    manifest['last_quality_findings_path'] = quality_report_path
    # Keep the persisted policy snapshot in sync with the policy that produced
    # these findings; split/retry packages and GUI readers consume the snapshot.
    manifest['quality_policy'] = translation_quality.normalize_policy(BATCH_QUALITY_POLICY)
    manifest.pop('last_apply_failure_report_path', None)
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    print(f"Manifest: {manifest['_manifest_path']}")
    print_check_summary(summary)
    print(f"Check failure report: {check_report_path}")
    print(f"Quality findings report: {quality_report_path}")
    return manifest


def apply_results(target=None, force=False):
    durable_store = _resolve_durable_sync_store(target)
    if durable_store is not None:
        # Durable Sync always applies through its bound preview.  ``force``
        # intentionally cannot bypass check/source/artifact predicates.
        return apply_durable_sync_results(durable_store)
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'apply')
    if manifest.get('applied_at') and not force:
        raise SystemExit('Manifest was already applied. Re-run apply with --force to bypass this guard; source validation still applies.')
    require_manifest_project_match(manifest, 'apply')

    transaction_path = os.path.join(
        manifest['_package_dir'],
        '.apply_writeback_transaction.json',
    )
    recover_atomic_write_transaction(transaction_path)
    require_safe_check_for_apply(manifest)

    replacements_by_file, translated_lines_by_file, failure_entries, summary = collect_result_actions(
        manifest,
        validate_sources=True,
    )
    attach_check_contract(manifest, summary)
    writeback_gate = summary.get('writeback_gate') or {}
    if writeback_gate.get('decision') != translation_quality.GATE_ALLOW:
        append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])
        report_path = write_apply_failure_report(
            manifest,
            'unsafe_apply_recheck',
            f'Apply recheck writeback gate is {writeback_gate.get("decision") or "unknown"}, not allow. No files were written.',
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
    rag_jobs = []
    revalidated_source_documents = {}
    file_keys = set(replacements_by_file) | set(translated_lines_by_file)
    for file_key in file_keys:
        replacements = replacements_by_file.get(file_key, {})
        file_info = manifest['files'].get(file_key)
        if not file_info:
            continue
        file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        source_document = _source_document_from_path(file_key, file_path)
        lines = source_document.lines()
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
        if replacements:
            revalidated_replacements_by_file[file_key] = replacements
            revalidated_source_documents[file_key] = source_document
        revalidated_line_numbers_by_file[file_key] = set(line_numbers_set)
        revalidated_file_paths[file_key] = file_path

    adapter_plan, adapter_snapshot = _validate_adapter_writeback_plan(
        manifest,
        revalidated_replacements_by_file,
        summary,
        failure_entries,
        live_sources=tuple(
            revalidated_source_documents[file_key]
            for file_key in revalidated_replacements_by_file
            if file_key in revalidated_source_documents
        ),
    )
    attach_check_contract(manifest, summary)
    writeback_gate = summary.get('writeback_gate') or {}
    if writeback_gate.get('decision') != translation_quality.GATE_ALLOW:
        append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])
        report_path = write_apply_failure_report(
            manifest,
            'unsafe_apply_revalidation',
            f'Apply source revalidation writeback gate is {writeback_gate.get("decision") or "unknown"}, not allow. No files were written.',
            summary=summary,
            failure_entries=failure_entries,
            current_fingerprint=summary.get('check_fingerprint'),
        )
        save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
        raise SystemExit(f'Apply refused because source revalidation is not safe. Report: {report_path}')

    writeback_files = []
    if adapter_plan is not None and adapter_snapshot is not None:
        rendered_by_file = _normalize_adapter_rendered_files(
            render_writeback_plan(
                adapter_plan,
                adapter_snapshot.project.source_documents,
            )
        )
        _require_adapter_render_targets(
            revalidated_replacements_by_file,
            rendered_by_file,
        )
        for file_key in revalidated_replacements_by_file:
            if not revalidated_replacements_by_file[file_key]:
                continue
            rendered_lines = rendered_by_file[_adapter_render_key(file_key)]
            writeback_files.append(
                (revalidated_file_paths[file_key], rendered_lines)
            )
    if writeback_files:
        atomic_write_many_lines(
            writeback_files,
            journal_path=transaction_path,
            encoding='utf-8',
        )

    for file_key, line_numbers_set in revalidated_line_numbers_by_file.items():
        file_path = revalidated_file_paths[file_key]
        line_numbers = sorted(line_numbers_set)
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
        raise SystemExit(
            'Revision manifest was already applied. Run preview-revisions again to '
            'refresh the writeback gate; --force does not bypass source snapshot checks.'
        )

    require_manifest_project_match(manifest, 'apply-revisions')
    _require_valid_revision_preview(manifest)
    if manifest.get('final_review_source'):
        import final_review as fr
        import final_review_revision

        final_review_revision.sync_linked_findings(manifest, fr.REVISION_STATE_PREVIEWED)
    transaction_path = os.path.join(
        manifest['_package_dir'],
        '.revision_writeback_transaction.json',
    )
    recover_atomic_write_transaction(transaction_path)

    replacements_by_file, _revised_lines_by_file, failure_entries, summary, preview_entries = collect_revision_actions(
        manifest,
        validate_sources=True,
    )

    revalidated_replacements_by_file = {}
    revalidated_file_paths = {}
    revalidated_source_documents = {}
    for file_key, replacements in replacements_by_file.items():
        file_info = manifest['files'].get(file_key)
        if not file_info:
            continue
        file_path = resolve_manifest_file_path(manifest, file_key, file_info)
        source_document = _source_document_from_path(file_key, file_path)
        lines = source_document.lines()
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
            revalidated_replacements_by_file[file_key] = replacements
            revalidated_file_paths[file_key] = file_path
            revalidated_source_documents[file_key] = source_document

    _quality_findings, quality_report_path = run_revision_quality_check(
        manifest,
        summary,
        revalidated_replacements_by_file,
        apply_stage=True,
    )
    # The preview-bound quality_findings.jsonl stays immutable; apply-time
    # findings live in quality_findings.apply.jsonl and are attached only to
    # the apply summary so a later apply re-check cannot see the preview
    # artifact as tampered.
    manifest['last_revision_apply_summary'] = summary

    # Validate the structural writeback plan before terminating on quality
    # blockers so both structural and quality diagnostics are persisted in the
    # blocked apply summary.
    adapter_plan, adapter_snapshot = _validate_adapter_writeback_plan(
        manifest,
        revalidated_replacements_by_file,
        summary,
        failure_entries,
        live_sources=tuple(
            revalidated_source_documents[file_key]
            for file_key in revalidated_replacements_by_file
            if file_key in revalidated_source_documents
        ),
    )
    structural_block = summary.get('adapter_writeback_status') == 'block'
    if structural_block:
        summary['pending_files'] = 0
        summary['pending_lines'] = 0

    quality_gate = summary.get('quality_gate') or {}
    quality_blocker_count = int(quality_gate.get('blocker_count') or 0)
    if quality_blocker_count > 0 or structural_block:
        summarize_revision_writeback_gate(summary)
        summary['check_status'] = translation_quality.overall_check_status(
            summary['writeback_gate'],
            quality_gate,
        )
        append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])
        if quality_blocker_count > 0 and structural_block:
            reason = 'quality_and_structural_blockers_present'
            message = (
                'configured quality blocker rules matched revision candidates and '
                'the adapter writeback plan is not safe. No files were written.'
            )
        elif quality_blocker_count > 0:
            reason = 'quality_blockers_present'
            message = (
                'configured quality blocker rules matched revision candidates. '
                'No files were written.'
            )
        else:
            reason = 'adapter_writeback_block'
            message = 'the adapter writeback plan is not safe. No files were written.'
        _mark_revision_apply_blocked(manifest, reason, message)

    writeback_files = []
    applied_file_keys = set()
    if adapter_plan is not None and adapter_snapshot is not None:
        rendered_by_file = _normalize_adapter_rendered_files(
            render_writeback_plan(
                adapter_plan,
                adapter_snapshot.project.source_documents,
            )
        )
        _require_adapter_render_targets(
            revalidated_replacements_by_file,
            rendered_by_file,
        )
        for file_key in revalidated_replacements_by_file:
            rendered_lines = rendered_by_file[_adapter_render_key(file_key)]
            writeback_files.append((revalidated_file_paths[file_key], rendered_lines))
            applied_file_keys.add(file_key)

    if writeback_files:
        atomic_write_many_lines(
            writeback_files,
            journal_path=transaction_path,
            encoding='utf-8',
        )

    applied_files = len(applied_file_keys)
    applied_lines = sum(
        len(replacements_by_line)
        for file_key, replacements_by_line in revalidated_replacements_by_file.items()
        if file_key in applied_file_keys
    )
    rag_jobs = []
    for file_key, replacements_by_line in revalidated_replacements_by_file.items():
        if file_key not in applied_file_keys:
            continue
        line_numbers = sorted(replacements_by_line.keys())
        update_progress(file_key, line_numbers)
        if line_numbers:
            rag_jobs.append(
                {
                    'file_rel_path': file_key,
                    'file_path': revalidated_file_paths[file_key],
                }
            )

    summary['pending_files'] = applied_files
    summary['pending_lines'] = applied_lines
    append_failure_entries(failure_entries, package_dir=manifest['_package_dir'])

    rag_apply_summary = {}
    if RAG_ENABLED and rag_jobs:
        rag_apply_summary = sync_rag_store_for_jobs(rag_jobs, quality_state='revision_applied')

    has_blocking_outcome = (
        summary.get('skipped_items', 0) > 0
        or summary.get('source_mismatch_items', 0) > 0
        or len(failure_entries) > 0
    )
    if applied_lines == 0 and has_blocking_outcome:
        apply_state = 'blocked'
        manifest['revision_apply_blocked_reason'] = 'all_items_blocked'
        manifest['revision_apply_message'] = (
            'No revisions could be written back because every candidate was skipped, '
            'source-mismatched, or failed validation.'
        )
    elif applied_lines == 0:
        apply_state = 'no_op'
    elif has_blocking_outcome:
        apply_state = 'partial'
    else:
        apply_state = 'applied'

    now = datetime.now().isoformat(timespec='seconds')
    manifest['revision_apply_state'] = apply_state
    manifest['revision_apply_checked_at'] = now
    manifest['revision_apply_summary'] = {
        'state': apply_state,
        'applied_files': applied_files,
        'applied_lines': applied_lines,
        'candidate_items': summary.get('candidate_valid_items', summary['valid_items']),
        'recoverable_items': summary['valid_items'],
        'unchanged_items': summary.get('unchanged_items', 0),
        'already_applied_items': summary.get('already_applied_items', 0),
        'skipped_items': summary.get('skipped_items', 0),
        'source_mismatch_items': summary.get('source_mismatch_items', 0),
        'failure_count': len(failure_entries),
        'quality_gate': summary.get('quality_gate'),
        'writeback_gate': summary.get('writeback_gate'),
        'quality_findings_count': summary.get('quality_findings_count', 0),
        'quality_findings_path': 'quality_findings.apply.jsonl',
        'quality_findings_sha256': summary.get('quality_findings_sha256'),
        'rag': rag_apply_summary,
    }
    manifest['last_revision_apply_summary'] = summary
    if apply_state in ('applied', 'partial'):
        manifest['revision_applied_at'] = now
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    if manifest.get('final_review_source'):
        import final_review as fr
        import final_review_revision

        if apply_state in ('applied', 'partial'):
            applied_lines_by_file = {
                file_key: set(replacements_by_line.keys())
                for file_key, replacements_by_line
                in revalidated_replacements_by_file.items()
                if file_key in applied_file_keys
            }
            applied_item_ids = {
                str(item.get('id') or '')
                for chunk in manifest.get('chunks', [])
                for item in chunk.get('items', [])
                if item.get('line')
                in applied_lines_by_file.get(chunk.get('file_rel_path'), set())
            }
            final_review_revision.sync_linked_findings(
                manifest,
                fr.REVISION_STATE_APPLIED,
                identity_ids=applied_item_ids,
            )

    print_revision_summary(summary)
    print(f'Revision apply state: {apply_state}')
    if apply_state == 'blocked':
        print(f'Revision apply reason: {manifest.get("revision_apply_blocked_reason") or ""}')
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


def build_repair_request(job, model=None):
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
                            file_rel_path=job.get('file_rel_path') or '',
                        )
                    }
                ],
            }
        ],
        'generation_config': build_generation_config(job['items'], model=model),
    }
    if BATCH_SAFETY_SETTINGS:
        request['safety_settings'] = BATCH_SAFETY_SETTINGS
    return {
        'key': job['key'],
        'request': request,
    }

def _sync_result_to_dict(result):
    response = {
        'response_payload': result.response_payload,
        'response_text': result.response_text,
        'finish_reason': result.finish_reason,
        'usage_metadata': dict(result.usage_metadata),
        'provider': result.provider,
        'model': result.model,
        'execution_mode': result.execution_mode,
    }
    request_metadata = dict(getattr(result, 'request_metadata', None) or {})
    if request_metadata:
        response['request_metadata'] = request_metadata
    output_diagnostics = dict(getattr(result, 'output_diagnostics', None) or {})
    if output_diagnostics:
        response['output_diagnostics'] = output_diagnostics
    return response


def _run_sync_backend_with_retry(
    backend,
    request,
    *,
    attempts=DEFAULT_SYNC_RETRY_ATTEMPTS,
):
    """Retry only transient structured categories on the same backend."""
    limit = max(1, int(attempts or 1))
    for attempt in range(1, limit + 1):
        try:
            return backend.generate(request)
        except Exception as exc:
            decision = sync_recovery_decision(exc)
            if not decision.retry_same_request or attempt >= limit:
                raise
            print(
                f'Sync request {decision.category}; retrying '
                f'({attempt}/{limit})...',
            )
            if decision.backoff:
                time.sleep(min(attempt, 2))
    raise RuntimeError('Sync request failed without a captured exception.')


def _routing_config_origins():
    path = str(getattr(legacy, 'TRANSLATOR_CONFIG', '') or '')
    if not path:
        return ()
    view = {
        'sync': {
            'backend': SYNC_BACKEND,
            'model': SYNC_MODEL,
        },
        'batch': {
            'model': BATCH_MODEL,
            'project_analysis': {'model': PROJECT_ANALYSIS_MODEL},
            'final_review': {'model': FINAL_REVIEW_MODEL},
        },
    }
    fingerprint = 'sha256:' + hashlib.sha256(
        json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return (
        model_profile.ConfigOrigin(
            kind='translator_config',
            path=path,
            fingerprint=fingerprint,
        ),
    )


def _runtime_custom_providers():
    return {
        key: value
        for key, value in (getattr(legacy, 'CUSTOM_LITELLM_PROVIDERS', None) or {}).items()
        if getattr(value, 'requires_key', None) is not None
    } or None


def _runtime_keyring_has_credential(provider_id):
    """Best-effort keyring probe for routing preflight.

    An unavailable credential store is not evidence that a slot is empty; the
    request backend will surface that infrastructure error with its own stable
    category. Only a successful empty read becomes a missing-reference issue.
    """
    try:
        from litellm_provider_config import load_provider_api_key

        return bool(load_provider_api_key(provider_id))
    except Exception:
        return True


def require_valid_routing_plan(plan, required_stages):
    """Raise the stable machine refusal for active-stage routing problems."""
    issues = model_profile.validate_routing_plan(
        plan,
        stages=required_stages,
        custom_providers=_runtime_custom_providers(),
        keyring_has_credential=_runtime_keyring_has_credential,
    )
    if issues:
        raise model_profile.routing_validation_error(issues)
    return plan


def freeze_runtime_routing_plan(
    *,
    execution=model_profile.ExecutionStrategy.SYNC,
    stage_overrides=None,
    created_at='',
    required_stages=None,
):
    """Snapshot routing from loaded globals and optionally fail fast.

    ``required_stages`` scopes validation to the routes this task will really
    execute. Callers must invoke this before creating task artifacts.
    """
    custom_providers = _runtime_custom_providers()
    try:
        plan = model_profile.resolve_routing_plan_from_runtime(
            sync_backend=SYNC_BACKEND,
            sync_model=SYNC_MODEL,
            batch_model=BATCH_MODEL,
            project_analysis_model=PROJECT_ANALYSIS_MODEL,
            final_review_model=FINAL_REVIEW_MODEL,
            sync_models=tuple(getattr(legacy, 'MODELS', ()) or ()),
            custom_providers=custom_providers,
            execution=execution,
            stage_overrides=stage_overrides,
            created_at=created_at,
            config_origins=_routing_config_origins(),
        )
    except (ValueError, TypeError) as exc:
        stages = tuple(sorted(str(item) for item in (required_stages or ())))
        stage = stages[0] if stages else ''
        raise model_profile.routing_resolution_error(exc, stage=stage) from exc
    if required_stages is not None:
        require_valid_routing_plan(plan, required_stages)
    return plan


def routing_plan_from_manifest(manifest):
    payload = (manifest or {}).get('model_routing')
    if not isinstance(payload, dict) or not payload:
        return None
    return model_profile.ModelRoutingPlan.from_manifest_dict(payload)


def attach_model_routing(manifest, plan):
    """Write the frozen plan snapshot onto a manifest dict."""
    manifest['model_routing'] = plan.to_manifest_dict()
    return plan


def _legacy_manifest_recorded_models(manifest):
    """Return ``(model, batch_model, provider)`` persisted on a pre-snapshot run."""
    payload = manifest or {}
    return (
        str(payload.get('model') or '').strip(),
        str(payload.get('batch_model') or '').strip(),
        str(payload.get('provider') or '').strip(),
    )


def resolve_manifest_routing_plan(manifest, *, execution=None, stage_overrides=None):
    """Return the frozen plan for a run, preferring the manifest snapshot.

    Old manifests without ``model_routing`` keep the model recorded on that
    manifest (``model`` / ``batch_model`` / ``provider``). Live ``SYNC_MODEL``
    / ``BATCH_MODEL`` are used only when those fields are also missing.
    """
    plan = routing_plan_from_manifest(manifest)
    if plan is not None:
        return plan
    if execution is None:
        execution = (
            model_profile.ExecutionStrategy.SYNC
            if str((manifest or {}).get('execution') or '') == 'sync'
            else model_profile.ExecutionStrategy.GEMINI_BATCH
        )
    persisted_model, persisted_batch, persisted_provider = (
        _legacy_manifest_recorded_models(manifest)
    )
    recorded = persisted_model or persisted_batch
    merged_overrides = dict(stage_overrides or {})
    if not recorded:
        return freeze_runtime_routing_plan(
            execution=execution,
            stage_overrides=merged_overrides or None,
        )
    stage = model_profile.stage_for_manifest_mode(manifest_mode(manifest))
    merged_overrides.setdefault(stage, recorded)
    return model_profile.resolve_routing_plan_from_runtime(
        sync_backend=persisted_provider or SYNC_BACKEND,
        sync_model=persisted_model,
        batch_model=persisted_batch or persisted_model,
        project_analysis_model=PROJECT_ANALYSIS_MODEL,
        final_review_model=FINAL_REVIEW_MODEL,
        sync_models=tuple(getattr(legacy, 'MODELS', ()) or ()),
        custom_providers=_runtime_custom_providers(),
        execution=execution,
        stage_overrides=merged_overrides or None,
        config_origins=_routing_config_origins(),
    )


def route_for_manifest(plan, manifest):
    stage = model_profile.stage_for_manifest_mode(manifest_mode(manifest))
    try:
        return plan.routes[stage]
    except KeyError as exc:
        raise ValueError(
            f'Frozen routing plan has no route for stage {stage}.'
        ) from exc


def route_model(plan, route):
    return model_profile.profile_for_route(plan, route).model


def _require_task_route(route):
    if not isinstance(route, model_profile.TaskRoute):
        raise TypeError(
            'run_sync_request requires an explicit TaskRoute; '
            f'got {type(route).__name__}.'
        )
    return route


def build_project_analysis_sync_runner(plan, route):
    """Return the shipped project-analysis generate callback.

    Freezes the TaskRoute/plan at construction time so retries and later
    config mutations cannot switch profiles.
    """
    from sync_model_backend import SYNC_EXECUTION_MODE, SyncGenerationResult

    route = _require_task_route(route)

    def _generate(request):
        payload = {
            'contents': request.contents,
            'generation_config': dict(request.config or {}),
        }
        system = (request.config or {}).get('system_instruction')
        if system:
            payload['system_instruction'] = system
        raw = run_sync_request(payload, route, plan=plan)
        return SyncGenerationResult(
            provider=str(raw.get('provider') or SYNC_BACKEND or 'gemini'),
            model=str(raw.get('model') or route_model(plan, route)),
            execution_mode=str(raw.get('execution_mode') or SYNC_EXECUTION_MODE),
            response_payload=raw.get('response_payload') or raw,
            response_text=str(raw.get('response_text') or ''),
            finish_reason=str(raw.get('finish_reason') or ''),
            usage_metadata=dict(raw.get('usage_metadata') or {}),
            output_diagnostics=dict(raw.get('output_diagnostics') or {}),
            request_metadata=dict(raw.get('request_metadata') or {}),
        )

    return _generate


def run_sync_request(
    request_payload,
    route,
    plan=None,
    *,
    api_key_index=None,
    retry_attempts=None,
    timeout_seconds=None,
):
    """Execute one sync request using a frozen TaskRoute.

    The model comes from ``plan.profiles[route.profile_id]``. ``SYNC_MODEL``
    is never consulted here; callers must freeze a :class:`ModelRoutingPlan`
    at run start and pass that snapshot.
    """
    route = _require_task_route(route)
    if plan is None:
        raise TypeError(
            'run_sync_request requires the frozen ModelRoutingPlan from run start.'
        )
    profile = model_profile.profile_for_route(plan, route)
    effective_model = profile.model
    config = dict(request_payload.get('generation_config') or {})
    config['timeout'] = normalize_sync_timeout_seconds(
        SYNC_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    system_instruction = request_payload.get('system_instruction')
    if system_instruction:
        config['system_instruction'] = system_instruction
    safety_settings = request_payload.get('safety_settings')
    if safety_settings:
        config['safety_settings'] = safety_settings
    config = filter_gemini_generation_config(effective_model, config)

    if profile.adapter == model_profile.ADAPTER_LITELLM:
        if api_key_index is not None:
            raise SystemExit('--api-key-index is only supported by the Gemini sync backend.')
        backend = model_profile.build_sync_backend(
            profile,
            custom_providers=legacy.CUSTOM_LITELLM_PROVIDERS,
        )
        request = SyncGenerationRequest(
            model=effective_model,
            contents=request_payload.get('contents') or [],
            config=config,
        )
        result = _run_sync_backend_with_retry(
            backend,
            request,
            attempts=(
                DEFAULT_SYNC_RETRY_ATTEMPTS
                if retry_attempts is None
                else retry_attempts
            ),
        )
        response = _sync_result_to_dict(result)
        response['output_diagnostics'] = model_usage_ledger.response_budget_diagnostics(
            response_text=result.response_text,
            finish_reason=result.finish_reason,
            usage_metadata=result.usage_metadata,
            max_output_tokens=config.get('max_output_tokens'),
        )
        return response

    key_attempts = (
        legacy.api_key_rotation_attempts()
        if api_key_index is None and hasattr(legacy, 'api_key_rotation_attempts')
        else 1
    )
    attempts = (
        max(DEFAULT_SYNC_RETRY_ATTEMPTS, key_attempts)
        if retry_attempts is None
        else max(1, int(retry_attempts))
    )
    last_error = None

    for attempt in range(1, attempts + 1):
        client = create_batch_client(api_key_index=api_key_index)
        try:
            backend = model_profile.build_sync_backend(
                profile,
                client=client,
                serialize_response=serialize_unknown,
                extract_text=extract_text_from_response_payload,
                extract_finish_reason=extract_finish_reason,
                extract_usage=lambda payload: summarize_usage_metadata(
                    extract_usage_metadata(payload)
                ),
            )
            result = backend.generate(SyncGenerationRequest(
                model=effective_model,
                contents=request_payload.get('contents') or [],
                config=config,
            ))
            response = _sync_result_to_dict(result)
            response['output_diagnostics'] = model_usage_ledger.response_budget_diagnostics(
                response_text=result.response_text,
                finish_reason=result.finish_reason,
                usage_metadata=result.usage_metadata,
                max_output_tokens=config.get('max_output_tokens'),
            )
            return response

        except Exception as exc:
            last_error = exc
            decision = sync_recovery_decision(exc)
            if decision.retry_same_request and attempt < attempts:
                rotated = bool(
                    decision.rotate_credentials
                    and api_key_index is None
                    and legacy.rotate_api_key()
                )
                label = decision.category.replace('_', ' ')
                key_action = 'next API key' if rotated else 'same API key'
                print(
                    f'Sync request hit {label}. Retrying with {key_action} '
                    f'({attempt}/{attempts})...'
                )
                if decision.backoff:
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


def _contract_mode_for_manifest(manifest):
    mode = manifest_mode(manifest)
    if mode == MANIFEST_MODE_REVISION:
        return translation_core.MODE_REVISION
    if mode == MANIFEST_MODE_KEYWORD_EXTRACTION:
        return translation_core.MODE_KEYWORD_EXTRACTION
    return translation_core.MODE_TRANSLATION


def _effective_sync_model(manifest, plan=None):
    """Return the frozen model for this sync manifest.

    After a run starts, live ``SYNC_MODEL`` is ignored. Old manifests without
    a ``model_routing`` snapshot fall back to the persisted model fields.
    """
    resolved = plan or routing_plan_from_manifest(manifest)
    if resolved is not None:
        return route_model(resolved, route_for_manifest(resolved, manifest))
    return str(manifest.get('model') or manifest.get('batch_model') or BATCH_MODEL or '')


def _targeted_sync_request_row(manifest, chunk, item_ids, *, model=None):
    selected_ids = {str(item_id) for item_id in item_ids if str(item_id)}
    targeted = dict(chunk)
    targeted['items'] = [
        item for item in chunk.get('items') or []
        if str(item.get('id') or '') in selected_ids
    ]
    targeted['key'] = f"{chunk.get('key', 'sync')}-targeted-001"
    effective_model = str(model or _effective_sync_model(manifest))
    mode = _contract_mode_for_manifest(manifest)
    if mode == translation_core.MODE_REVISION:
        return build_revision_request(targeted, model=effective_model), targeted
    if mode == translation_core.MODE_KEYWORD_EXTRACTION:
        max_candidates = int(
            (manifest.get('keyword_settings') or {}).get('max_candidates_per_chunk')
            or KEYWORD_MAX_CANDIDATES_PER_CHUNK
        )
        return build_keyword_request(
            targeted,
            max_candidates,
            model=effective_model,
        ), targeted
    return build_batch_request(targeted, model=effective_model), targeted


def _contract_from_sync_result(result, chunk, mode):
    payload = parse_json_payload(result.get('response_text') or '')
    return validate_result_contract(payload, mode, chunk.get('items') or [])


def _merge_sync_contract_reports(first, retry, chunk, mode):
    if mode == translation_core.MODE_KEYWORD_EXTRACTION:
        retry_made_progress = bool(retry.items)
        merged_items = []
        seen = set()
        for item in [*first.items, *retry.items]:
            identity = (
                compact_text(item.get('source', '')).casefold(),
                compact_text(item.get('suggested_target', '')).casefold(),
                compact_text(item.get('category', '')).casefold(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            merged_items.append(item)
        first_summary = str(first.metadata.get('chunk_summary') or '')
        retry_summary = str(retry.metadata.get('chunk_summary') or '')
        merged_payload = {
            translation_core.MODEL_RESPONSE_ENVELOPE_KEYS[mode]: merged_items,
            'chunk_summary': retry_summary or first_summary,
            'summary_evidence_item_ids': list(dict.fromkeys([
                *list(first.metadata.get('summary_evidence_item_ids') or []),
                *list(retry.metadata.get('summary_evidence_item_ids') or []),
            ])),
        }
        merged = validate_result_contract(
            merged_payload,
            mode,
            chunk.get('items') or [],
        )
        merged.diagnostics = list(dict.fromkeys([
            *merged.diagnostics,
            *first.diagnostics,
            *retry.diagnostics,
        ]))
        terminal_reports = [retry] if retry_made_progress else [first, retry]
        merged.issues = list(dict.fromkeys([
            *merged.issues,
            *(issue for report in terminal_reports for issue in report.issues),
        ]))
        if merged.issues:
            merged.retry_ids = [
                str(item.get('id') or '')
                for item in chunk.get('items') or []
                if str(item.get('id') or '')
            ]
        return merged
    merged_by_id = {
        str(item.get('id') or ''): item
        for item in first.items
        if str(item.get('id') or '')
    }
    retryable_ids = {str(item_id) for item_id in first.retry_ids}
    merged_by_id.update({
        str(item.get('id') or ''): item
        for item in retry.items
        if str(item.get('id') or '') in retryable_ids
    })
    envelope_key = translation_core.MODEL_RESPONSE_ENVELOPE_KEYS[mode]
    merged_payload = {
        envelope_key: [
            merged_by_id[str(item.get('id') or '')]
            for item in chunk.get('items') or []
            if str(item.get('id') or '') in merged_by_id
        ]
    }
    merged = validate_result_contract(
        merged_payload,
        mode,
        chunk.get('items') or [],
    )
    retryable_ids = {str(item_id) for item_id in first.retry_ids}
    first_terminal_issues = [
        issue
        for issue in first.issues
        if str(issue.item_id or '') not in retryable_ids
    ]
    merged.issues = list(dict.fromkeys([
        *merged.issues,
        *first_terminal_issues,
        *retry.issues,
    ]))
    merged.diagnostics = list(dict.fromkeys([
        *merged.diagnostics,
        *first.diagnostics,
        *retry.diagnostics,
    ]))
    return merged


def write_request_rows(path, request_rows):
    atomic_write_jsonl(path, request_rows, ensure_ascii=False)


def write_manifest_file(package_dir, manifest, update_latest=True):
    manifest_path = os.path.join(package_dir, 'manifest.json')
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    if update_latest:
        remember_latest_manifest(manifest_path)
    return manifest_path


def execute_sync_request_rows(manifest_path, request_rows, api_key_index=None, *, routing_plan=None):
    """Execute one complete, unique request row for every manifest chunk.

    Full coverage is validated before any provider call so the rewritten result
    JSONL and the manifest's terminal sync status always describe the same run.
    The routing plan is frozen before the first request; later config changes
    and targeted retries reuse that snapshot.
    """
    manifest = load_manifest(manifest_path)
    result_path = resolve_manifest_result_path(manifest)
    plan = routing_plan or resolve_manifest_routing_plan(
        manifest,
        execution=model_profile.ExecutionStrategy.SYNC,
    )
    route = route_for_manifest(plan, manifest)
    require_valid_routing_plan(plan, {route.stage})
    effective_model = route_model(plan, route)
    if not manifest.get('model_routing'):
        attach_model_routing(manifest, plan)
    manifest_chunks = list(manifest.get('chunks') or [])
    chunk_map = {chunk.get('key'): chunk for chunk in manifest_chunks}
    contract_mode = _contract_mode_for_manifest(manifest)
    keyword_contract = contract_mode == translation_core.MODE_KEYWORD_EXTRACTION
    requested_chunks = []
    requested_keys = []
    for index, row in enumerate(request_rows, start=1):
        key = str(row.get('key') or '') if isinstance(row, dict) else ''
        chunk = chunk_map.get(key)
        if chunk is None:
            raise cli_contract.MachineContractError(
                f'Sync request row #{index} has no matching manifest chunk: '
                f'{key or "(missing)"}',
                code_name='SYNC_REQUEST_CHUNK_MISSING',
                suggested_action='rebuild_sync_package',
                details={'row': index, 'key': key},
            )
        if key in requested_keys:
            raise cli_contract.MachineContractError(
                f'Sync request rows contain a duplicate manifest chunk: {key}',
                code_name='SYNC_REQUEST_CHUNK_DUPLICATE',
                suggested_action='rebuild_sync_package',
                details={'row': index, 'key': key},
            )
        items = chunk.get('items')
        if not isinstance(items, list) or not items:
            raise cli_contract.MachineContractError(
                f'Sync request row #{index} references an empty manifest chunk: {key}',
                code_name='SYNC_REQUEST_CHUNK_EMPTY',
                suggested_action='rebuild_sync_package',
                details={'row': index, 'key': key},
            )
        requested_keys.append(key)
        requested_chunks.append(chunk)
    manifest_keys = [
        str(chunk.get('key') or '')
        for chunk in manifest_chunks
    ]
    missing_keys = [key for key in manifest_keys if key not in requested_keys]
    if missing_keys:
        raise cli_contract.MachineContractError(
            'Sync request rows do not cover every manifest chunk: '
            + ', '.join(missing_keys),
            code_name='SYNC_REQUEST_CHUNK_INCOMPLETE',
            suggested_action='rebuild_sync_package',
            details={'missing_keys': missing_keys},
        )
    summary = {
        'request_count': len(request_rows),
        'successful_request_count': 0,
        'failed_request_count': 0,
        'max_tokens_count': 0,
        'missing_text_count': 0,
        'contract_partial_requests': 0,
        'targeted_retry_requests': 0,
        'targeted_retry_items': 0,
        'completion_tokens': 0,
        'completion_tokens_known_requests': 0,
        'reasoning_tokens': 0,
        'reasoning_tokens_known_requests': 0,
        'text_output_tokens': 0,
        'text_output_tokens_known_requests': 0,
        'reasoning_budget_pressure_count': 0,
        'truncated_output_count': 0,
        'output_reason_counts': {},
        'error_category_counts': {},
        'reason_counts': {},
    }
    if keyword_contract:
        summary.update({
            'contract_expected_chunks': len(request_rows),
            'contract_first_pass_complete_chunks': 0,
            'contract_final_complete_chunks': 0,
        })
    else:
        summary.update({
            'contract_expected_items': sum(
                len(chunk.get('items') or []) for chunk in requested_chunks
            ),
            'contract_first_pass_valid_items': 0,
            'contract_final_valid_items': 0,
        })
    result_rows = []
    for index, row in enumerate(request_rows, start=1):
        key = row.get('key', f'sync-{index}')
        chunk = requested_chunks[index - 1]
        print(f'[{index}/{len(request_rows)}] {key}')
        result_row = {'key': key}
        try:
            result = run_sync_request(
                row.get('request') or {},
                route,
                plan=plan,
                api_key_index=api_key_index,
            )
            result_row['response'] = result.get('response_payload') or {}
            result_row['response_semantics'] = {
                'response': 'first_pass_provider_payload',
                'normalized_response': 'final_merged_contract',
            }
            result_row['finish_reason'] = result.get('finish_reason', '')
            result_row['usage_metadata'] = result.get('usage_metadata') or {}
            result_row['output_diagnostics'] = sync_output_diagnostics(
                result,
                row.get('request') or {},
            )
            record_sync_output_summary(summary, result_row['output_diagnostics'])
            if result.get('request_metadata'):
                result_row['request_metadata'] = (
                    model_usage_ledger.normalize_request_metadata(
                        result.get('request_metadata') or {}
                    )
                )
            result_row['provider'] = result.get('provider') or SYNC_BACKEND
            result_row['model'] = result.get('model') or effective_model
            result_row['execution_mode'] = result.get('execution_mode') or 'sync'
            summary['successful_request_count'] += 1
            result_row['provider_response_attempts'] = [{
                'kind': 'first_pass',
                'finish_reason': result.get('finish_reason', ''),
                'usage_metadata': result.get('usage_metadata') or {},
                'output_diagnostics': result_row['output_diagnostics'],
                'request_metadata': result_row.get('request_metadata') or {},
            }]
            if result.get('finish_reason') == 'MAX_TOKENS':
                summary['max_tokens_count'] += 1
                bump_counter(summary['reason_counts'], 'max_tokens')
            if not result.get('response_text'):
                summary['missing_text_count'] += 1
            print(f"  finish_reason: {result.get('finish_reason') or '(none)'}")

            first_contract = None
            first_error = None
            try:
                first_contract = _contract_from_sync_result(
                    result,
                    chunk,
                    contract_mode,
                )
                if keyword_contract:
                    summary['contract_first_pass_complete_chunks'] += int(
                        first_contract.complete
                    )
                else:
                    summary['contract_first_pass_valid_items'] += len(
                        first_contract.valid_ids
                    )
                record_contract_reasons(summary, first_contract)
                result_row['provider_response_attempts'][0][
                    'contract_diagnostics'
                ] = first_contract.to_diagnostics()
            except Exception as exc:
                first_error = exc
                bump_counter(
                    summary['reason_counts'],
                    contract_error_reason(exc, 'response_contract_error'),
                )

            retry_ids = (
                list(first_contract.retry_ids)
                if first_contract is not None
                else [str(item.get('id') or '') for item in chunk.get('items') or []]
            )
            if contract_mode == translation_core.MODE_KEYWORD_EXTRACTION and (
                first_error is not None
                or (first_contract is not None and first_contract.issues)
            ):
                retry_ids = [
                    str(item.get('id') or '') for item in chunk.get('items') or []
                ]

            final_contract = first_contract
            if retry_ids:
                retry_row, retry_chunk = _targeted_sync_request_row(
                    manifest,
                    chunk,
                    retry_ids,
                    model=effective_model,
                )
                summary['targeted_retry_requests'] += 1
                summary['targeted_retry_items'] += len(retry_chunk.get('items') or [])
                print(
                    f"  targeted retry: {len(retry_chunk.get('items') or [])} items"
                )
                try:
                    retry_result = run_sync_request(
                        retry_row.get('request') or {},
                        route,
                        plan=plan,
                        api_key_index=api_key_index,
                    )
                    result_row['provider_response_attempts'].append({
                        'kind': 'targeted_retry',
                        'item_ids': [
                            item.get('id') for item in retry_chunk.get('items') or []
                        ],
                        'response': retry_result.get('response_payload') or {},
                        'finish_reason': retry_result.get('finish_reason', ''),
                        'usage_metadata': retry_result.get('usage_metadata') or {},
                        'output_diagnostics': sync_output_diagnostics(
                            retry_result,
                            retry_row.get('request') or {},
                        ),
                        'request_metadata': (
                            model_usage_ledger.normalize_request_metadata(
                                retry_result.get('request_metadata') or {}
                            )
                        ),
                    })
                    record_sync_output_summary(
                        summary,
                        result_row['provider_response_attempts'][-1][
                            'output_diagnostics'
                        ],
                    )
                    retry_contract = _contract_from_sync_result(
                        retry_result,
                        retry_chunk,
                        contract_mode,
                    )
                    result_row['provider_response_attempts'][-1][
                        'contract_diagnostics'
                    ] = retry_contract.to_diagnostics()
                    record_contract_reasons(summary, retry_contract)
                    final_contract = (
                        retry_contract
                        if first_contract is None
                        else _merge_sync_contract_reports(
                            first_contract,
                            retry_contract,
                            chunk,
                            contract_mode,
                        )
                    )
                except Exception as exc:
                    error_category = sync_error_category(exc)
                    result_row['provider_response_attempts'].append({
                        'kind': 'targeted_retry',
                        'item_ids': [
                            item.get('id') for item in retry_chunk.get('items') or []
                        ],
                        'error_category': error_category,
                        'error': sync_error_summary(exc),
                        'request_metadata': (
                            model_usage_ledger.normalize_request_metadata(
                                getattr(exc, 'request_metadata', None) or {}
                            )
                        ),
                    })
                    bump_counter(
                        summary['error_category_counts'],
                        error_category,
                    )
                    bump_counter(
                        summary['reason_counts'],
                        contract_error_reason(
                            exc,
                            'targeted_retry_contract_error',
                        ),
                    )

            if final_contract is not None:
                result_row['normalized_response'] = final_contract.to_envelope()
                diagnostics = final_contract.to_diagnostics()
                diagnostics['first_pass_valid_count'] = (
                    len(first_contract.valid_ids) if first_contract is not None else 0
                )
                diagnostics['targeted_retry_count'] = 1 if retry_ids else 0
                result_row['contract_diagnostics'] = diagnostics
                if keyword_contract:
                    summary['contract_final_complete_chunks'] += int(
                        final_contract.complete
                    )
                else:
                    summary['contract_final_valid_items'] += len(
                        final_contract.valid_ids
                    )
                if not final_contract.complete:
                    summary['contract_partial_requests'] += 1
            else:
                summary['contract_partial_requests'] += 1
                result_row['contract_diagnostics'] = {
                    'mode': contract_mode,
                    'complete': False,
                    'expected_count': len(chunk.get('items') or []),
                    'valid_count': 0,
                    'retry_ids': retry_ids,
                    'reason_counts': {
                        contract_error_reason(
                            first_error,
                            'response_contract_error',
                        ): 1
                    },
                }
        except Exception as exc:
            error_category = sync_error_category(exc)
            summary['failed_request_count'] += 1
            bump_counter(summary['reason_counts'], error_category)
            bump_counter(summary['error_category_counts'], error_category)
            result_row['error_category'] = error_category
            result_row['error'] = sync_error_summary(exc)
            request_metadata = model_usage_ledger.normalize_request_metadata(
                getattr(exc, 'request_metadata', None) or {}
            )
            if request_metadata:
                result_row['request_metadata'] = request_metadata
            print(f"  error: {result_row['error']}")
        result_rows.append(result_row)

    atomic_write_jsonl(result_path, result_rows, ensure_ascii=False)
    content_sha = file_sha256(result_path)
    atomic_write_text(f'{result_path}.sha256', content_sha + '\n')

    manifest['sync_completed_at'] = datetime.now().isoformat(timespec='seconds')
    if keyword_contract:
        expected = summary['contract_expected_chunks']
        summary['contract_first_pass_chunk_completeness'] = (
            summary['contract_first_pass_complete_chunks'] / expected
            if expected else 1.0
        )
        summary['contract_final_chunk_completeness'] = (
            summary['contract_final_complete_chunks'] / expected
            if expected else 1.0
        )
    else:
        expected = summary['contract_expected_items']
        summary['contract_first_pass_completeness'] = (
            summary['contract_first_pass_valid_items'] / expected
            if expected else 1.0
        )
        summary['contract_final_completeness'] = (
            summary['contract_final_valid_items'] / expected
            if expected else 1.0
        )
    manifest['job_state'] = (
        'SYNC_COMPLETED'
        if summary['failed_request_count'] == 0
        and summary['contract_partial_requests'] == 0
        else 'SYNC_PARTIAL'
    )
    manifest['sync_summary'] = summary
    manifest['result_jsonl_path'] = result_path
    manifest['result_jsonl_sha256'] = content_sha
    save_manifest(manifest, update_latest=False)
    if keyword_contract:
        print(
            'Model contract chunk completeness: '
            f"{summary['contract_final_complete_chunks']}/"
            f"{summary['contract_expected_chunks']}"
        )
    else:
        unresolved_items = max(
            0,
            summary['contract_expected_items'] - summary['contract_final_valid_items'],
        )
        print(
            'Model contract completeness: '
            f"{summary['contract_final_valid_items']}/"
            f"{summary['contract_expected_items']}"
        )
    print(
        'Targeted retries: '
        f"{summary['targeted_retry_requests']} requests / "
        f"{summary['targeted_retry_items']} items"
    )
    if not keyword_contract:
        print(f'Unresolved contract items: {unresolved_items}')
    print(f"Contract partial requests: {summary['contract_partial_requests']}")
    print_sync_output_summary(summary)
    import_manifest_usage_best_effort(manifest)
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
    routing_plan=None,
):
    settings = dict(settings or {})
    settings.setdefault('timeout_seconds', SYNC_TIMEOUT_SECONDS)
    input_jsonl_path = os.path.join(package_dir, 'requests.jsonl')
    result_jsonl_path = os.path.join(package_dir, 'results.jsonl')
    write_request_rows(input_jsonl_path, request_rows)
    plan = routing_plan or freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.SYNC,
    )
    route = plan.routes[model_profile.stage_for_manifest_mode(mode)]
    profile = model_profile.profile_for_route(plan, route)
    manifest = {
        'version': 2,
        'manifest_version': 2,
        'core_schema_version': 2,
        'mode': mode,
        'execution': 'sync',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': profile.model,
        'provider': SYNC_BACKEND,
        'model': profile.model,
        'execution_mode': 'sync',
        'model_routing': plan.to_manifest_dict(),
        'model_response_contract': {
            'version': 1,
            'mode': mode,
            'envelope_key': translation_core.MODEL_RESPONSE_ENVELOPE_KEYS.get(mode, ''),
            'legacy_bare_array_readable': True,
        },
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        **_manifest_target_language_fields(),
        **batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        **translation_quality.manifest_quality_policy_fields(runtime_policy=BATCH_QUALITY_POLICY),
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
    if 'model_routing' not in manifest or not manifest.get('model_routing'):
        attach_model_routing(manifest, plan)
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
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.SYNC,
        required_stages={model_profile.STAGE_KEYWORD},
    )
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
    effective_model = route_model(
        routing_plan,
        routing_plan.routes[model_profile.STAGE_KEYWORD],
    )
    request_rows = [
        build_keyword_request(chunk, max_candidates, model=effective_model)
        for chunk in chunks
    ]
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
        routing_plan=routing_plan,
    )
    manifest = execute_sync_request_rows(
        manifest_path,
        request_rows,
        api_key_index=api_key_index,
        routing_plan=routing_plan,
    )
    print(f"Sync keyword run: {manifest['_package_dir']}")
    export = export_keyword_candidates(
        target=manifest['_manifest_path'],
        output_jsonl=output_jsonl,
        output_markdown=output_markdown,
        output_summary_jsonl=output_summary_jsonl,
        output_summary_markdown=output_summary_markdown,
    )
    payload = dict(export or {})
    payload['manifest_path'] = manifest['_manifest_path']
    return payload


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
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.SYNC,
        required_stages={model_profile.STAGE_REVISION},
    )
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
    effective_model = route_model(
        routing_plan,
        routing_plan.routes[model_profile.STAGE_REVISION],
    )
    request_rows = [
        build_revision_request(chunk, model=effective_model)
        for chunk in chunks
    ]
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
            'max_context_chars': STORY_MEMORY_MAX_CONTEXT_CHARS,
            'top_k_relations': STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_terms': STORY_MEMORY_TOP_K_TERMS,
            'include_scene_summary': STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
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
        routing_plan=routing_plan,
    )
    manifest = execute_sync_request_rows(
        manifest_path,
        request_rows,
        api_key_index=api_key_index,
        routing_plan=routing_plan,
    )
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
    print_sync_output_summary(summary)
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
    routing_plan = freeze_runtime_routing_plan(
        execution=model_profile.ExecutionStrategy.SYNC,
        required_stages={model_profile.STAGE_TRANSLATION},
    )
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

    usage_run_id = f'repair-{os.path.basename(run_dir)}'
    usage_operation_id = (
        'repair-report-'
        + hashlib.sha256(os.path.abspath(report_path).encode('utf-8')).hexdigest()[:20]
    )
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
        'completion_tokens': 0,
        'completion_tokens_known_requests': 0,
        'reasoning_tokens': 0,
        'reasoning_tokens_known_requests': 0,
        'text_output_tokens': 0,
        'text_output_tokens_known_requests': 0,
        'reasoning_budget_pressure_count': 0,
        'truncated_output_count': 0,
        'output_reason_counts': {},
        'error_category_counts': {},
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

    repair_route = routing_plan.routes[model_profile.STAGE_TRANSLATION]
    effective_model = route_model(routing_plan, repair_route)
    request_rows = [build_repair_request(job, model=effective_model) for job in jobs]
    write_jsonl_file(request_log_path, request_rows)

    for index, (job, request_row) in enumerate(zip(jobs, request_rows), start=1):
        finish_reason = ''
        usage_metadata = {}
        response_text = ''
        parse_error = ''
        output_diagnostics = {}
        request_metadata = {}
        result_items = []
        parse_ok = False
        try:
            response_data = run_sync_request(
                request_row['request'],
                repair_route,
                plan=routing_plan,
                api_key_index=api_key_index,
            )
            finish_reason = response_data['finish_reason']
            usage_metadata = response_data['usage_metadata']
            response_text = response_data['response_text']
            output_diagnostics = sync_output_diagnostics(
                response_data,
                request_row.get('request') or {},
            )
            request_metadata = model_usage_ledger.normalize_request_metadata(
                response_data.get('request_metadata') or {}
            )
            record_sync_output_summary(summary, output_diagnostics)
        except Exception as exc:
            error_category = sync_error_category(exc)
            summary['request_errors'] += 1
            bump_counter(reason_counts, error_category)
            bump_counter(summary['error_category_counts'], error_category)
            parse_error = sync_error_summary(exc)
            request_metadata = model_usage_ledger.normalize_request_metadata(
                getattr(exc, 'request_metadata', None) or {}
            )
            for item in job['items']:
                failure_entries.append(
                    {
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                        'file': job['file_path'],
                        'line': item['line'],
                        'source': item['text'],
                        'id': item['id'],
                        'error': parse_error,
                        'error_category': error_category,
                        'request_metadata': request_metadata,
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
                    'error_category': error_category,
                    'finish_reason': finish_reason,
                    'usage_metadata': usage_metadata,
                    'request_metadata': request_metadata,
                    'response_preview': '',
                }
            )
            continue

        record_generation_usage_best_effort(
            task_mode='repair',
            stage='repair',
            result=response_data,
            operation_id=usage_operation_id,
            run_id=usage_run_id,
            source_key=str(job.get('key') or index),
            thinking_level=BATCH_THINKING_LEVEL,
            source={
                'kind': 'repair_response',
                'report_path': os.path.abspath(report_path),
                'run_dir': os.path.abspath(run_dir),
                'job_key': str(job.get('key') or ''),
                'request_index': index,
            },
        )

        if response_text:
            try:
                payload = parse_json_payload(response_text)
                contract = validate_result_contract(
                    payload,
                    translation_core.MODE_TRANSLATION,
                    job['items'],
                )
                record_contract_reasons(summary, contract)
                result_items = contract.items
                parse_ok = bool(result_items) or not job['items']
            except Exception as exc:
                parse_error = str(exc)
                summary['parse_errors'] += 1
                bump_counter(
                    reason_counts,
                    contract_error_reason(exc, 'parse_error'),
                )
        else:
            reason_code = str(
                output_diagnostics.get('reason_code')
                or translation_core.CONTRACT_EMPTY_RESPONSE_TEXT
            )
            parse_error = f'Missing text in response payload [{reason_code}]'
            summary['parse_errors'] += 1
            bump_counter(reason_counts, reason_code)

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
                        'output_diagnostics': output_diagnostics,
                        'request_metadata': request_metadata,
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
                    'output_diagnostics': output_diagnostics,
                    'request_metadata': request_metadata,
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
        for missing_id in sorted(missing_ids):
            item = item_map[missing_id]
            failure_entries.append(
                {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'file': job['file_path'],
                    'line': item['line'],
                    'source': item['text'],
                    'id': item['id'],
                    'error': 'Response missing expected id',
                    'reason_code': 'response_missing_expected_id',
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
                'output_diagnostics': output_diagnostics,
                'request_metadata': request_metadata,
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
    return legacy.describe_prepare_command(command)


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
    """True when the project already has real Chinese translation progress.

    Blank Ren'Py templates still contain ``old``/``new`` string pairs, so
    ``old_lines`` alone must not count as historical translations. That signal
    is used for RAG tips and incremental vs first-pass workflow state; both
    require actual translated targets (Han characters), not template structure.
    """
    # Full doctor reports always include this field (0 when nothing is translated).
    if 'translated_task_count' in report:
        return int(report.get('translated_task_count') or 0) > 0

    # Hand-built / older report shapes without an explicit Chinese progress
    # field: do not guess from counts. Template structure (old_lines) is
    # unreliable, and ``pending < translate_blocks`` also holds when many rows
    # are filtered as non-translatable, so a wrong signal would suggest RAG /
    # incremental state for a first-pass project. Treat as first-pass.
    return False


def collect_doctor_workflow_state(report):
    """Return a normal workflow state separately from actionable recommendations."""
    if report.get('layout_status') != 'ready':
        return ''
    if report.get('mode') == 'blocked_invalid_tl_subdir':
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

    if report.get('mode') == 'blocked_invalid_tl_subdir':
        # Path escape is a hard config error; surface it via warnings only.
        return recommendations

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

    # Past layout/template blockers: collect required prep and optional tips together
    # (required first, then optional). Do not early-return between context tips so
    # e.g. "bootstrap source index" and "enable RAG" can both appear.
    context_status = report.get('context_status') or {}
    source_index = context_status.get('source_index') or {}
    rag = context_status.get('rag') or {}
    project_analysis = context_status.get('project_analysis') or {}

    source_index_status = _doctor_source_index_needs_bootstrap(source_index)
    if source_index_status == 'missing':
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_SOURCE_INDEX)
        )
    elif source_index_status == 'incomplete':
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE)
        )

    if _doctor_rag_needs_bootstrap(rag):
        if rag.get('bootstrap_on_build'):
            recommendations.append(
                doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD)
            )
        else:
            recommendations.append(
                doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG)
            )
    elif (
        not rag.get('enabled')
        and has_existing_translations
        and pending > 0
        and _doctor_should_recommend_enabling_rag(report)
    ):
        recommendations.append(
            doctor_rec.make_doctor_recommendation(doctor_rec.ENABLE_RAG_FOR_CONSISTENCY)
        )

    if project_analysis.get('enabled'):
        if not project_analysis.get('model'):
            recommendations.append(
                doctor_rec.make_doctor_recommendation(
                    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_MODEL
                )
            )
        if int(project_analysis.get('api_key_count') or 0) <= 0:
            recommendations.append(
                doctor_rec.make_doctor_recommendation(
                    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_API
                )
            )
        analysis_status = str(project_analysis.get('overall_status') or 'missing')
        if analysis_status == 'missing':
            recommendations.append(
                doctor_rec.make_doctor_recommendation(
                    doctor_rec.BUILD_PROJECT_ANALYSIS
                )
            )
        elif analysis_status == 'stale':
            recommendations.append(
                doctor_rec.make_doctor_recommendation(
                    doctor_rec.REFRESH_PROJECT_ANALYSIS
                )
            )
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

    try:
        from project_analysis import collect_project_analysis_status

        project_analysis_status = collect_project_analysis_status()
        project_analysis_status.update(
            {
                'enabled': bool(PROJECT_ANALYSIS_ENABLED),
                'inject_published_brief': bool(PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF),
                'model': PROJECT_ANALYSIS_MODEL or BATCH_MODEL or SYNC_MODEL or '',
                'api_key_count': len(getattr(legacy, 'API_KEYS', []) or []),
            }
        )
    except Exception as exc:
        project_analysis_status = {
            'store_dir': '',
            'store_exists': False,
            'overall_status': 'failed',
            'error': str(exc),
        }

    return {
        'rag': rag_status,
        'source_index': source_index_status,
        'project_analysis': project_analysis_status,
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
    from project_asset_paths import (
        expected_project_asset_paths,
        paths_match_project,
        resolve_configured_glossary_value,
        resolve_glossary_path,
        resolve_macro_setting_path,
    )

    config = _read_translator_config_object()
    expected = expected_project_asset_paths(base_dir)

    glossary_configured = resolve_configured_glossary_value(config)
    batch = config.get('batch') if isinstance(config.get('batch'), dict) else {}
    macro_configured = batch.get('macro_setting_file') or ''

    glossary_resolved = resolve_glossary_path(
        glossary_configured,
        game_root=base_dir,
        tool_dir=legacy.TOOL_DIR,
    ) or expected['glossary_file']
    macro_resolved = resolve_macro_setting_path(
        macro_configured,
        game_root=base_dir,
        tool_dir=legacy.TOOL_DIR,
    ) or expected['macro_setting_file']

    return {
        'glossary_file': glossary_resolved,
        'glossary_exists': os.path.isfile(glossary_resolved),
        'glossary_matches_project': paths_match_project(
            glossary_resolved,
            expected['glossary_file'],
        ),
        'macro_setting_file': macro_resolved,
        'macro_exists': os.path.isfile(macro_resolved),
        'macro_matches_project': paths_match_project(
            macro_resolved,
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


def collect_doctor_model_routing_status():
    """Return read-only routing snapshots and machine-decidable issues.

    ``status`` is ``ok`` or ``attention``. It never uses ``blocked`` and does
    not change the doctor command exit or workflow gate.
    """
    custom_providers = _runtime_custom_providers()
    plans = {}
    issues = []
    seen = set()
    for strategy in (
        model_profile.ExecutionStrategy.SYNC,
        model_profile.ExecutionStrategy.GEMINI_BATCH,
    ):
        try:
            plan = model_profile.resolve_routing_plan_from_runtime(
                sync_backend=SYNC_BACKEND,
                sync_model=SYNC_MODEL,
                batch_model=BATCH_MODEL,
                project_analysis_model=PROJECT_ANALYSIS_MODEL,
                final_review_model=FINAL_REVIEW_MODEL,
                sync_models=tuple(getattr(legacy, 'MODELS', ()) or ()),
                custom_providers=custom_providers,
                execution=strategy,
                config_origins=_routing_config_origins(),
            )
        except (ValueError, TypeError) as exc:
            payload = model_profile.routing_resolution_issue(exc).to_manifest_dict()
            payload['execution_strategy'] = strategy.value
            issues.append(payload)
            continue
        plans[strategy.value] = plan.to_manifest_dict()
        for issue in model_profile.validate_routing_plan(
            plan,
            custom_providers=custom_providers,
            keyring_has_credential=_runtime_keyring_has_credential,
        ):
            payload = issue.to_manifest_dict()
            payload['execution_strategy'] = strategy.value
            key = (
                payload['code'], payload['stage'], payload['profile_id'],
                tuple(payload['missing_capabilities']), strategy.value,
            )
            if key not in seen:
                seen.add(key)
                issues.append(payload)
    return {
        'status': 'attention' if issues else 'ok',
        'issues': issues,
        'plans': plans,
    }


def collect_doctor_report():
    source_game_dir = legacy._guess_source_game_dir()
    template_info = legacy.get_prepare_template_command_info(source_game_dir)
    can_generate_template = bool(template_info.get('available'))
    original_game_dir = legacy.resolve_original_game_dir()
    work_bootstrap_allowed, work_dir, _ = legacy.work_dir_bootstrap_allowed()

    warnings = []
    tl_path_invalid = False
    try:
        legacy.ensure_tl_dir_within_base(
            legacy.BASE_DIR,
            legacy.TL_DIR,
            tl_subdir=legacy.TL_SUBDIR,
        )
        legacy.normalize_tl_subdir(legacy.TL_SUBDIR)
    except Exception as exc:
        # Catch InvalidTlSubdirError and any unexpected path errors so doctor
        # still returns a structured report instead of crashing mid-scan.
        tl_path_invalid = True
        warnings.append(
            f'Invalid tl_subdir / TL_DIR boundary: {exc}. '
            "Fix translator_config.json tl_subdir so it is a relative path under "
            "game_root with no '..' segments (example: 'game/tl/schinese'). "
            'Build/apply must not proceed until this is fixed.'
        )

    # Do not walk TL_DIR when it may escape the project root.
    if tl_path_invalid:
        counts = {
            'rpy_files': 0,
            'translate_blocks': 0,
            'string_sections': 0,
            'old_lines': 0,
            'new_lines': 0,
            'commented_original_lines': 0,
        }
        tl_exists = False
        has_tl_files = False
    else:
        counts = collect_tl_doctor_counts()
        tl_exists = os.path.isdir(legacy.TL_DIR)
        has_tl_files = counts['rpy_files'] > 0

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

    # Custom prepare commands are a trusted local execution boundary.
    allow_shell = bool(getattr(legacy, 'PREP_ALLOW_SHELL_COMMANDS', False))
    shell_command_fields = []
    if legacy.prepare_command_uses_shell(getattr(legacy, 'PREP_UNPACK_COMMAND', None)):
        shell_command_fields.append('prepare.unpack_command')
    if legacy.prepare_command_uses_shell(getattr(legacy, 'PREP_TEMPLATE_COMMAND', None)):
        shell_command_fields.append('prepare.template_command')
    if allow_shell:
        warnings.append(
            'HIGH RISK: prepare.allow_shell_commands is enabled. '
            'translator_config.json is executable local configuration and shell-string '
            'prepare commands can run arbitrary system commands during prepare. '
            'Prefer argv lists and disable shell mode unless you fully trust this config.'
        )
    if shell_command_fields:
        warnings.append(
            'HIGH RISK: shell-string prepare command(s) configured: '
            + ', '.join(shell_command_fields)
            + '. Resolved commands run under game_root with shell=True.'
        )

    if tl_path_invalid:
        mode = 'blocked_invalid_tl_subdir'
    elif has_tl_files:
        mode = 'existing_tl_only'
    elif can_generate_template:
        mode = 'can_generate_template'
    else:
        mode = 'blocked_missing_template'

    pending_task_count = 0
    pending_file_count = 0
    translated_task_count = 0
    total_task_count = 0
    # Avoid walking a TL tree that may sit outside the project root.
    # Progress counts use the same inventory filter as batch build, without the
    # occurrence-extraction work that build/writeback needs.
    if has_tl_files and not tl_path_invalid:
        try:
            progress = collect_doctor_translation_progress()
            pending_file_count = progress['pending_file_count']
            pending_task_count = progress['pending_task_count']
            translated_task_count = progress['translated_task_count']
            total_task_count = progress['total_task_count']
        except Exception as exc:
            print(f'Warning: Could not compute pending translation counts: {exc}')

    context_status = collect_doctor_context_status()
    project_assets = collect_doctor_project_assets_status(legacy.BASE_DIR)
    warnings.extend(collect_doctor_project_assets_warnings(project_assets))
    model_routing_status = collect_doctor_model_routing_status()
    for issue in model_routing_status['issues']:
        warnings.append(
            'Model routing preflight '
            f"[{issue['code']}] ({issue['execution_strategy']}/"
            f"{issue['stage'] or 'profile'}): {issue['message']}"
        )
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
        'template_command_cwd': template_info.get('cwd', ''),
        'template_command_shell': bool(template_info.get('shell')),
        'template_reason': template_info.get('reason', ''),
        'python_exe': template_info.get('python_exe', ''),
        'launcher_py': template_info.get('launcher_py', ''),
        'allow_shell_commands': allow_shell,
        'shell_prepare_command_fields': list(shell_command_fields),
        'mode': mode,
        'counts': counts,
        'pending_task_count': pending_task_count,
        'pending_file_count': pending_file_count,
        'translated_task_count': translated_task_count,
        'total_task_count': total_task_count,
        'context_status': context_status,
        'project_assets': project_assets,
        'model_routing': model_routing_status,
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
    project_analysis_context = context_status.get('project_analysis') or {}
    print('Doctor report:')
    print(f"- Base dir: {report['base_dir']}")
    print(f"- TL dir: {report['tl_dir']} (exists: {report['tl_exists']})")
    print(f"- TL subdir: {report.get('tl_subdir') or ''}")
    print(f"- Language: {report['language']}")
    routing_status = report.get('model_routing') or {}
    routing_issues = list(routing_status.get('issues') or [])
    print(
        f"- Model routing: {routing_status.get('status') or 'unknown'} "
        f"({len(routing_issues)} issue(s))"
    )
    for issue in routing_issues:
        print(
            "  - "
            f"[{issue.get('code')}] {issue.get('execution_strategy')}/"
            f"{issue.get('stage') or 'profile'}: {issue.get('message')}"
        )
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
        if report.get('template_command_cwd'):
            print(f"- Template cwd: {report['template_command_cwd']}")
        if report.get('template_command_shell'):
            print('- Template shell: True (HIGH RISK)')
    else:
        print(f"- Template generation: unavailable ({report['template_reason'] or 'no command resolved'})")
    if report.get('allow_shell_commands'):
        print('- Allow shell prepare commands: True (HIGH RISK; trusted local config only)')
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
            f"file_count={report['pending_file_count']}, "
            f"translated_count={report.get('translated_task_count', 0)}, "
            f"total_count={report.get('total_task_count', 0)}"
        )
        if report['pending_task_count'] > 0:
            print(
                '  Note: pending counts English strings without Han characters; may include '
                'preserved names, patron lists, or punctuation-only updates. '
                'translated_count counts targets that already contain Chinese. '
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
    print(
        '- Project analysis: '
        f"overall={project_analysis_context.get('overall_status') or 'missing'}, "
        f"store_dir={project_analysis_context.get('store_dir') or ''}, "
        f"store_exists={project_analysis_context.get('store_exists', False)}, "
        f"injectable={project_analysis_context.get('injectable', False)}, "
        f"chunks={project_analysis_context.get('chunk_count', 0)}, "
        f"labels={project_analysis_context.get('label_count', 0)}, "
        f"routes={project_analysis_context.get('route_count', 0)}, "
        f"brief={project_analysis_context.get('brief_status') or 'missing'}, "
        f"updated_at={project_analysis_context.get('updated_at') or ''}, "
        f"error={project_analysis_context.get('error') or ''}"
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


def add_json_shaping_arguments(command_parser):
    command_parser.add_argument(
        '--compact',
        action='store_true',
        help='Emit JSON without indentation or extra spaces.',
    )
    command_parser.add_argument(
        '--fields',
        action='append',
        nargs='+',
        default=[],
        metavar='PATH',
        help=(
            'Project JSON output to dot-separated field paths. May be repeated; '
            'comma-separated paths are also accepted.'
        ),
    )
    command_parser.add_argument(
        '--output-file',
        default='',
        metavar='PATH',
        help='Atomically write the final JSON document to PATH instead of stdout.',
    )
    return command_parser


def add_machine_output_argument(command_parser):
    command_parser.add_argument(
        '--output',
        choices=('text', 'json'),
        default='text',
        help='Output human-readable text (default) or one machine-readable JSON document.',
    )
    command_parser.add_argument(
        '--strict-exit-codes',
        action='store_true',
        help=(
            'With --output json, return semantic exit codes for needs-action, '
            'blocked, invalid-state, and retryable outcomes.'
        ),
    )
    command_parser.add_argument(
        '--non-interactive',
        action='store_true',
        help=(
            'Guarantee no stdin prompts; manifest-consuming commands also require '
            'an explicit target.'
        ),
    )
    command_parser.add_argument(
        '--require-explicit-target',
        action='store_true',
        help=(
            'Reject implicit latest-manifest or submit-build fallback for commands '
            'that consume a manifest.'
        ),
    )
    return add_json_shaping_arguments(command_parser)


def add_durable_sync_output_arguments(command_parser):
    """Add the schema-v1 machine contract plus #347's local ``--json`` alias."""
    add_machine_output_argument(command_parser)
    command_parser.add_argument(
        '--json',
        dest='output',
        action='store_const',
        const='json',
        help='Alias for --output json on durable Sync commands.',
    )
    return command_parser


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Batch translator for Ren\'Py tl files using Gemini Batch API.'
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True)

    doctor_parser = subparsers.add_parser('doctor', help='Inspect prepare, SDK, and TL template compatibility without writing files.')
    add_machine_output_argument(doctor_parser)

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
    add_machine_output_argument(build_parser)
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
    add_machine_output_argument(keyword_build_parser)

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
    add_machine_output_argument(revision_build_parser)

    export_revision_corpus_parser = subparsers.add_parser(
        'export-revision-corpus',
        help=(
            'Export a read-only revision polishing corpus '
            '(JSONL + Markdown + manifest) without modifying game files.'
        ),
    )
    export_revision_corpus_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the corpus into DIR instead of a fresh timestamped Batch jobs subdirectory.',
    )
    add_machine_output_argument(export_revision_corpus_parser)

    import_revision_proposals_parser = subparsers.add_parser(
        'import-revision-proposals',
        help=(
            'Validate structured human/Agent proposal JSONL against the live project, '
            'build a local revision package, and run preview-revisions. Never writes .rpy.'
        ),
    )
    import_revision_proposals_parser.add_argument(
        'proposal',
        help=(
            'Proposal JSONL path (schema_version=1, identity_v2, project identity, '
            'snapshots, provenance).'
        ),
    )
    import_revision_proposals_parser.add_argument(
        '--corpus-manifest',
        default='',
        help=(
            'Companion revision_corpus_manifest.json. Defaults to the proposal file directory '
            'when present.'
        ),
    )
    import_revision_proposals_parser.add_argument(
        '--stage',
        action='store_true',
        help=(
            'Only validate and persist a staged-selection artifact; do not generate '
            'a revision preview until confirm-revision-proposals is called.'
        ),
    )
    import_revision_proposals_parser.add_argument(
        '--operation-identity',
        default='',
        help='Optional caller-owned operation identity to bind to a staged session.',
    )
    add_machine_output_argument(import_revision_proposals_parser)

    confirm_revision_proposals_parser = subparsers.add_parser(
        'confirm-revision-proposals',
        help=(
            'Confirm an immutable staged proposal selection and generate the existing '
            'revision preview; never writes .rpy.'
        ),
    )
    confirm_revision_proposals_parser.add_argument(
        'staged_selection',
        help='staged_selection.json produced by import-revision-proposals --stage.',
    )
    confirm_revision_proposals_parser.add_argument(
        '--selection-file',
        required=True,
        help='Explicit revision_proposal_selection JSON confirmation artifact.',
    )
    add_machine_output_argument(confirm_revision_proposals_parser)

    export_project_snapshot_parser = subparsers.add_parser(
        'export-project-snapshot',
        help=(
            'Export a source-only project/game-version snapshot '
            '(JSON + JSONL) without modifying game files.'
        ),
    )
    export_project_snapshot_parser.add_argument(
        '--version-id',
        required=True,
        help='Stable project-supplied game version ID (for example 1.4.0 or build-20260809).',
    )
    export_project_snapshot_parser.add_argument(
        '--version-label',
        default='',
        help='Optional human-readable game version label.',
    )
    export_project_snapshot_parser.add_argument(
        '--source-revision',
        default='',
        help='Optional source revision/build identifier stored as provenance.',
    )
    export_project_snapshot_parser.add_argument(
        '--coverage-review',
        default='',
        metavar='FILE',
        help=(
            'Optional completed coverage review JSON. '
            'Without it the snapshot freezes a pending review record.'
        ),
    )
    export_project_snapshot_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the snapshot into DIR instead of logs/project_snapshots/.',
    )
    add_machine_output_argument(export_project_snapshot_parser)

    reconcile_project_snapshots_parser = subparsers.add_parser(
        'reconcile-project-snapshots',
        help=(
            'Compare two saved project snapshots and export a read-only '
            'reconciliation report.'
        ),
    )
    reconcile_project_snapshots_parser.add_argument(
        'base',
        help='Base snapshot directory or project_snapshot.json path.',
    )
    reconcile_project_snapshots_parser.add_argument(
        'target',
        help='Target snapshot directory or project_snapshot.json path.',
    )
    reconcile_project_snapshots_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the report into DIR instead of logs/project_reconciliations/.',
    )
    add_machine_output_argument(reconcile_project_snapshots_parser)

    build_translation_records_parser = subparsers.add_parser(
        'build-translation-records',
        help=(
            'Freeze one Batch package\'s validated translations as a '
            'versioned P4 translation-records artifact.'
        ),
    )
    build_translation_records_parser.add_argument(
        'snapshot',
        help='Base snapshot directory or project_snapshot.json path.',
    )
    build_translation_records_parser.add_argument(
        'manifest',
        help='Batch package manifest (translation mode) with downloaded results.',
    )
    build_translation_records_parser.add_argument(
        '--origin',
        default='model_initial',
        choices=sorted(engine_reuse.RECORD_ORIGINS),
        help='Translation provenance recorded for every exported row.',
    )
    build_translation_records_parser.add_argument(
        '--previous-records',
        default='',
        metavar='PATH',
        help=(
            'Previous translation-records package for the same snapshot; '
            'carries record-level revision history into the new export.'
        ),
    )
    build_translation_records_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the records into DIR instead of logs/translation_records/.',
    )
    add_machine_output_argument(build_translation_records_parser)

    build_reuse_candidates_parser = subparsers.add_parser(
        'build-reuse-candidates',
        help=(
            'Derive reviewable translation-reuse candidates from saved P3 '
            'snapshots, a reconciliation report, and base translation records.'
        ),
    )
    build_reuse_candidates_parser.add_argument(
        'base_snapshot',
        help='Base (old) snapshot directory or project_snapshot.json path.',
    )
    build_reuse_candidates_parser.add_argument(
        'target_snapshot',
        help='Target (new) snapshot directory or project_snapshot.json path.',
    )
    build_reuse_candidates_parser.add_argument(
        'reconciliation',
        help='Reconciliation report directory or reconciliation_report.json path.',
    )
    build_reuse_candidates_parser.add_argument(
        'base_records',
        help='Base translation-records directory or manifest path.',
    )
    build_reuse_candidates_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the package into DIR instead of logs/translation_reuse/.',
    )
    add_machine_output_argument(build_reuse_candidates_parser)

    import_reuse_decisions_parser = subparsers.add_parser(
        'import-reuse-decisions',
        help=(
            'Apply human/agent reuse decisions with provenance and export an '
            'audited candidate package.'
        ),
    )
    import_reuse_decisions_parser.add_argument(
        'reuse',
        help='Reuse candidates package directory or reuse_report.json path.',
    )
    import_reuse_decisions_parser.add_argument(
        'decisions',
        help='Reuse decisions JSONL file.',
    )
    import_reuse_decisions_parser.add_argument(
        '--base-snapshot',
        default='',
        metavar='PATH',
        help='Override the recorded base snapshot path.',
    )
    import_reuse_decisions_parser.add_argument(
        '--target-snapshot',
        default='',
        metavar='PATH',
        help='Override the recorded target snapshot path.',
    )
    import_reuse_decisions_parser.add_argument(
        '--reconciliation',
        default='',
        metavar='PATH',
        help='Override the recorded reconciliation report path.',
    )
    import_reuse_decisions_parser.add_argument(
        '--base-records',
        default='',
        metavar='PATH',
        help='Override the recorded base translation-records path.',
    )
    import_reuse_decisions_parser.add_argument(
        '--output-dir',
        default='',
        metavar='DIR',
        help='Write the updated package into DIR instead of logs/translation_reuse/.',
    )
    add_machine_output_argument(import_reuse_decisions_parser)

    export_reuse_results_parser = subparsers.add_parser(
        'export-reuse-results',
        help=(
            'Merge accepted, fresh reuse translations into a Batch package as '
            'canonical results; game-file writes still require check -> apply.'
        ),
    )
    export_reuse_results_parser.add_argument(
        'reuse',
        help='Reuse candidates package directory or reuse_report.json path.',
    )
    export_reuse_results_parser.add_argument(
        'manifest',
        help='Target-version Batch package manifest (translation mode).',
    )
    export_reuse_results_parser.add_argument(
        '--base-snapshot',
        default='',
        metavar='PATH',
        help='Override the recorded base snapshot path.',
    )
    export_reuse_results_parser.add_argument(
        '--target-snapshot',
        default='',
        metavar='PATH',
        help='Override the recorded target snapshot path.',
    )
    export_reuse_results_parser.add_argument(
        '--reconciliation',
        default='',
        metavar='PATH',
        help='Override the recorded reconciliation report path.',
    )
    export_reuse_results_parser.add_argument(
        '--base-records',
        default='',
        metavar='PATH',
        help='Override the recorded base translation-records path.',
    )
    add_machine_output_argument(export_reuse_results_parser)

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

    final_review_build_parser = subparsers.add_parser(
        'final-review-build',
        help=(
            'Build a report-only final-review campaign package: readiness gate, '
            'frozen context/translation snapshot digests, and review units. '
            'Does not call the model or write .rpy files.'
        ),
    )
    final_review_build_parser.add_argument(
        '--display-name',
        default='',
        help='Override campaign display name.',
    )
    final_review_build_parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip auto prepare steps before collecting review sources.',
    )
    final_review_build_parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Items per review unit. Defaults to batch.final_review.chunk_size.',
    )
    final_review_build_parser.add_argument(
        '--allow-pending',
        action='store_true',
        help=(
            'Allow building even when pending translations remain in scope '
            '(not recommended; results may be incomplete).'
        ),
    )
    add_machine_output_argument(final_review_build_parser)

    final_review_status_parser = subparsers.add_parser(
        'final-review-status',
        help='Show final-review campaign progress and unit status counts.',
    )
    final_review_status_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Campaign package dir or manifest path. Defaults to latest package.',
    )
    final_review_status_parser.add_argument(
        '--json',
        action='store_true',
        help='Print unversioned JSON status (legacy); prefer --output json for the versioned contract.',
    )
    add_machine_output_argument(final_review_status_parser)

    final_review_export_parser = subparsers.add_parser(
        'final-review-export',
        help='Export final-review findings JSONL and Markdown report (report-only).',
    )
    final_review_export_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Campaign package dir or manifest path. Defaults to latest package.',
    )
    final_review_export_parser.add_argument(
        '--jsonl',
        default='',
        help='Output findings JSONL path (default: package findings.jsonl).',
    )
    final_review_export_parser.add_argument(
        '--markdown',
        default='',
        help='Output report Markdown path (default: package report.md).',
    )
    add_machine_output_argument(final_review_export_parser)

    final_review_resume_parser = subparsers.add_parser(
        'final-review-resume',
        help=(
            'Rebuild Batch requests for pending/stale/failed review units; '
            'skip done units whose input_digest is unchanged against live shared context. '
            'Pass --force to re-run all. Invalidates prior results.jsonl so download cannot reuse it.'
        ),
    )
    final_review_resume_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Campaign package dir or manifest path. Defaults to latest package.',
    )
    final_review_resume_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-queue all units, including done units with matching digests.',
    )
    add_machine_output_argument(final_review_resume_parser)

    final_review_ingest_parser = subparsers.add_parser(
        'final-review-ingest-results',
        help=(
            'Parse downloaded final-review result JSONL into findings and unit status '
            '(report-only; does not write .rpy).'
        ),
    )
    final_review_ingest_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Campaign package dir or manifest path. Defaults to latest package.',
    )
    final_review_ingest_parser.add_argument(
        '--result',
        default='',
        help=(
            'Result JSONL path. Defaults to a post-resume download path '
            '(manifest result_jsonl_path). Pre-resume package results.jsonl is rejected '
            'unless --allow-stale-results is set.'
        ),
    )
    final_review_ingest_parser.add_argument(
        '--allow-stale-results',
        action='store_true',
        help=(
            'Allow ingesting package results.jsonl even when it may predate the last '
            'resume (escape hatch / re-parse). Prefer a fresh download or --result.'
        ),
    )
    add_machine_output_argument(final_review_ingest_parser)

    final_review_revisions_parser = subparsers.add_parser(
        'final-review-create-revisions',
        help=(
            'Convert explicitly selected final-review findings into a normal revision '
            'package and run preview-revisions. Never writes .rpy files.'
        ),
    )
    final_review_revisions_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Final-review campaign package or manifest. Defaults to latest package.',
    )
    final_review_revisions_parser.add_argument(
        '--finding-id',
        action='append',
        default=[],
        help=(
            'Finding id to convert (repeatable). Without this option, uses findings '
            'already marked selection_state=selected.'
        ),
    )
    add_machine_output_argument(final_review_revisions_parser)
    project_analysis_status_parser = subparsers.add_parser(
        'project-analysis-status',
        help=(
            'Inspect Project Analysis store status (missing/draft/published/stale) '
            'without generating or injecting analysis context.'
        ),
    )
    project_analysis_status_parser.add_argument(
        '--store-dir',
        default='',
        help='Override project analysis store directory. Defaults to context_storage path.',
    )
    project_analysis_status_parser.add_argument(
        '--source-fingerprint',
        default='',
        help=(
            'Optional expected source fingerprint. When set, published artifacts with a '
            'mismatched lineage fingerprint are reported as stale.'
        ),
    )
    project_analysis_status_parser.add_argument(
        '--json',
        action='store_true',
        help='Print machine-readable JSON status instead of the human-readable report.',
    )

    pa_ingest = subparsers.add_parser(
        'project-analysis-ingest-keywords',
        help='Import keyword_chunk_summaries.jsonl into Project Analysis chunk drafts.',
    )
    pa_ingest.add_argument(
        '--summary-jsonl',
        required=True,
        help='Path to keyword_chunk_summaries.jsonl from keyword export.',
    )
    pa_ingest.add_argument('--store-dir', default='', help='Override analysis store directory.')

    pa_build = subparsers.add_parser(
        'project-analysis-build-structure',
        help=(
            'Parse Ren\'Py labels/jumps into draft label/route summaries and project brief '
            '(no LLM). Uses existing chunk drafts when present.'
        ),
    )
    pa_build.add_argument('--store-dir', default='', help='Override analysis store directory.')
    pa_build.add_argument(
        '--script-root',
        action='append',
        default=None,
        help='Script root to scan for .rpy files. Repeatable. Defaults under game_root.',
    )
    pa_build.add_argument(
        '--entry-label',
        action='append',
        default=None,
        help='Optional route entry label. Repeatable.',
    )

    pa_generate = subparsers.add_parser(
        'project-analysis-generate',
        help=(
            'LLM map-reduce: refine label → route → project brief drafts. '
            'Requires structure drafts from project-analysis-build-structure. '
            'Never writes glossary/story_graph/.rpy; publish remains manual.'
        ),
    )
    pa_generate.add_argument('--store-dir', default='', help='Override analysis store directory.')
    pa_generate.add_argument(
        '--model',
        default='',
        help='Override batch.project_analysis.model (else falls back to batch/sync model).',
    )
    pa_generate.add_argument(
        '--force',
        action='store_true',
        help='Re-generate even when drafts already have matching LLM lineage.',
    )

    pa_inspect = subparsers.add_parser(
        'project-analysis-inspect',
        help='Inspect draft/published analysis artifacts (JSON).',
    )
    pa_inspect.add_argument('--store-dir', default='', help='Override analysis store directory.')
    pa_inspect.add_argument(
        '--kind',
        default='status',
        choices=['status', 'labels', 'routes', 'chunks', 'brief'],
        help='What to print (default: status).',
    )

    pa_diff = subparsers.add_parser(
        'project-analysis-diff',
        help='Compare draft vs published project brief.',
    )
    pa_diff.add_argument('--store-dir', default='', help='Override analysis store directory.')

    pa_publish = subparsers.add_parser(
        'project-analysis-publish',
        help='Publish draft project brief for optional prompt injection.',
    )
    pa_publish.add_argument('--store-dir', default='', help='Override analysis store directory.')
    pa_publish.add_argument(
        '--force',
        action='store_true',
        help='Replace published brief or force-publish stale with --source-fingerprint.',
    )
    pa_publish.add_argument(
        '--source-fingerprint',
        default='',
        help='Current structure fingerprint (required with --force for stale/missing lineage).',
    )

    pa_unpublish = subparsers.add_parser(
        'project-analysis-unpublish',
        help='Unpublish project brief so it is no longer injectable.',
    )
    pa_unpublish.add_argument('--store-dir', default='', help='Override analysis store directory.')

    sync_start_parser = subparsers.add_parser(
        'sync-start',
        help='Start a durable synchronous translation run from the current project.',
    )
    add_durable_sync_output_arguments(sync_start_parser)
    sync_start_parser.add_argument(
        '--client-token',
        default='',
        metavar='TOKEN',
        help='Optional caller-owned idempotency token; blank always creates a new run.',
    )

    sync_resume_parser = subparsers.add_parser(
        'sync-resume',
        help='Resume one explicit durable synchronous translation run.',
    )
    add_durable_sync_output_arguments(sync_resume_parser)
    sync_resume_parser.add_argument('run', metavar='RUN', help='Durable Sync run ID.')

    sync_status_parser = subparsers.add_parser(
        'sync-status',
        help='Inspect one explicit durable Sync run or the latest valid durable run.',
    )
    add_durable_sync_output_arguments(sync_status_parser)
    sync_status_parser.add_argument(
        'run', nargs='?', default='', metavar='RUN', help='Durable Sync run ID.'
    )
    sync_status_parser.add_argument(
        '--latest',
        action='store_true',
        help='Select the uniquely latest valid durable run in the active project log.',
    )

    sync_cancel_parser = subparsers.add_parser(
        'sync-cancel',
        help='Commit cancellation intent for one explicit durable Sync run.',
    )
    add_durable_sync_output_arguments(sync_cancel_parser)
    sync_cancel_parser.add_argument('run', metavar='RUN', help='Durable Sync run ID.')

    sync_derive_parser = subparsers.add_parser(
        'sync-derive',
        help='Create a new run from one terminal durable Sync run and the current plan.',
    )
    add_durable_sync_output_arguments(sync_derive_parser)
    sync_derive_parser.add_argument('run', metavar='RUN', help='Source durable Sync run ID.')
    unknown_mode = sync_derive_parser.add_mutually_exclusive_group()
    unknown_mode.add_argument(
        '--retry-unknown',
        action='store_true',
        help='Retry outcome-unknown items in the derived run.',
    )
    unknown_mode.add_argument(
        '--exclude-unknown',
        action='store_true',
        help='Require the current scoped plan to omit all outcome-unknown items.',
    )
    sync_derive_parser.add_argument(
        '--ack-duplicate-billing-risk',
        action='store_true',
        help='Acknowledge duplicate execution/billing risk required by --retry-unknown.',
    )

    submit_parser = subparsers.add_parser('submit', help='Create and submit a batch job.')
    add_machine_output_argument(submit_parser)
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

    usage_import_parser = subparsers.add_parser(
        'usage-import',
        help='Offline-import provider usage from an existing results.jsonl into the project ledger.',
    )
    usage_import_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    usage_import_parser.add_argument(
        '--json',
        action='store_true',
        help='Print a machine-readable JSON import summary.',
    )

    usage_report_parser = subparsers.add_parser(
        'usage-report',
        help='Report actual recorded model usage for the current project.',
    )
    usage_report_parser.add_argument('--task', default='', help='Filter by task mode.')
    usage_report_parser.add_argument('--stage', default='', help='Filter by pipeline stage.')
    usage_report_parser.add_argument('--provider', default='', help='Filter by provider.')
    usage_report_parser.add_argument('--model', default='', help='Filter by model.')
    usage_report_parser.add_argument(
        '--group-by',
        default='task,stage,provider,model',
        help='Comma-separated grouping fields: task, stage, provider, model, run, operation, execution.',
    )
    usage_report_parser.add_argument(
        '--json',
        action='store_true',
        help='Print a machine-readable JSON report.',
    )

    status_parser = subparsers.add_parser('status', help='Refresh and show batch job status.')
    add_machine_output_argument(status_parser)
    status_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )

    check_parser = subparsers.add_parser('check', help='Dry-run parse downloaded results and summarize recoverable items.')
    add_machine_output_argument(check_parser)
    check_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )

    for ack_command in ('quality-ack', 'quality-unack'):
        action_text = (
            'Acknowledge' if ack_command == 'quality-ack' else 'Revert acknowledged'
        )
        ack_parser = subparsers.add_parser(
            ack_command,
            help=(
                f'{action_text} warning finding IDs in the manifest without '
                f'touching quality_findings.jsonl.'
            ),
        )
        add_machine_output_argument(ack_parser)
        ack_parser.add_argument(
            'target',
            nargs='?',
            default='',
            help='Manifest path or package dir. Defaults to latest package.',
        )
        selection = ack_parser.add_mutually_exclusive_group()
        selection.add_argument(
            '--finding',
            dest='finding_ids',
            action='append',
            default=[],
            metavar='ID',
            help=(
                'Finding ID to select. Repeat to select multiple warning findings.'
            ),
        )
        selection.add_argument(
            '--all',
            dest='all_findings',
            action='store_true',
            help=(
                'Select all current warning findings.'
                if ack_command == 'quality-ack'
                else 'Revert all acknowledged warning findings.'
            ),
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
    add_machine_output_argument(download_parser)
    download_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Manifest path or package dir. Defaults to latest package.',
    )
    download_parser.add_argument('--force', action='store_true', help='Overwrite local results.jsonl.')

    apply_parser = subparsers.add_parser('apply', help='Apply downloaded results back into tl files.')
    add_machine_output_argument(apply_parser)
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
    add_machine_output_argument(keyword_export_parser)

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
        help=(
            'Auto-accept candidates at or above this confidence without prompting; '
            'history conflicts, missing evidence, or evidence that no longer matches '
            'the candidate still require review unless --yes is used.'
        ),
    )
    merge_keywords_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing glossary entries with conflicting targets.',
    )
    merge_keywords_parser.add_argument(
        '--yes',
        action='store_true',
        help=(
            'Accept all non-skipped candidates without interactive prompts; explicitly '
            'override history-evidence review.'
        ),
    )
    merge_keywords_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating a timestamped glossary backup before writing.',
    )
    add_machine_output_argument(merge_keywords_parser)

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
    add_machine_output_argument(revision_preview_parser)

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
        help=(
            'Bypass the revision_applied_at guard without refreshing preview; '
            'preview and source snapshot validation still apply.'
        ),
    )
    add_machine_output_argument(revision_apply_parser)

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
    add_machine_output_argument(sync_keyword_parser)

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
        help=(
            'When used with --apply, bypass the revision_applied_at guard without '
            'refreshing preview; preview and source snapshot validation still apply.'
        ),
    )
    sync_revision_parser.add_argument('--api-key-index', type=int, default=None, help='Optional API key index override.')
    add_machine_output_argument(sync_revision_parser)

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
    capabilities_parser = subparsers.add_parser(
        'capabilities',
        help='Print machine-readable CLI capabilities and the command index.',
    )
    add_json_shaping_arguments(capabilities_parser)
    schema_parser = subparsers.add_parser(
        'schema',
        help='Print the machine-readable argparse schema for one command.',
    )
    schema_parser.add_argument(
        'schema_command',
        choices=sorted(subparsers.choices),
        help='Command whose current argparse schema should be returned.',
    )
    add_json_shaping_arguments(schema_parser)


    return parser


def validate_machine_invocation(args):
    """Enforce opt-in deterministic invocation rules before workflow setup."""

    command = str(getattr(args, 'command', '') or '')
    require_target = bool(
        getattr(args, 'non_interactive', False)
        or getattr(args, 'require_explicit_target', False)
    )
    target = str(getattr(args, 'target', '') or '').strip()
    if require_target and command in EXPLICIT_TARGET_COMMANDS and not target:
        raise cli_contract.MachineContractError(
            f'{command} requires an explicit manifest path or package directory.',
            code_name='EXPLICIT_TARGET_REQUIRED',
            suggested_action='pass_manifest_path',
            details={'required_argument': 'target'},
        )


def _durable_sync_root_dir():
    return os.path.join(str(getattr(legacy, 'LOG_DIR', LOG_DIR)), 'sync_runs')


def _durable_sync_store_only_service(*, deliver_usage=False):
    from sync_run_service import SyncRunService

    return SyncRunService(
        _durable_sync_root_dir(),
        freshness_reporter=lambda _store: {
            'resume_allowed': None,
            'source': 'not_checked',
            'profile': 'not_checked',
            'config': 'not_checked',
            'reasons': ['freshness_rechecked_before_dispatch'],
        },
        game_root=(legacy.BASE_DIR if deliver_usage and legacy.BASE_DIR else None),
    )


def _durable_sync_production_service(*, require_provider):
    from sync_run_service import build_production_sync_run_service

    context = legacy.prepare_sync_translation_execution_context(
        require_provider=require_provider,
        persist_corrected_game_root=True,
    )
    load_batch_settings()
    pricing = batch_cost_estimate.load_pricing_config(
        _read_translator_config_object()
    )
    service = build_production_sync_run_service(
        _durable_sync_root_dir(),
        context,
        game_root=legacy.BASE_DIR or None,
        pricing_config=pricing,
    )
    return service, context


def _print_durable_sync_snapshot(snapshot):
    progress = dict((snapshot or {}).get('progress') or {})
    items = dict(progress.get('items') or {})
    requests = dict(progress.get('requests') or {})
    print(f"Durable Sync run: {snapshot.get('run_id') or '(unknown)'}")
    print(f"Status: {snapshot.get('run_status') or 'unknown'}")
    print(
        'Items: '
        f"accepted={int(items.get('accepted') or 0)}, "
        f"unresolved={int(items.get('unresolved') or 0)}, "
        f"expected={int(items.get('expected') or 0)}"
    )
    print(
        'Requests: '
        f"total={int(requests.get('total') or 0)}, "
        f"pending={int(requests.get('pending') or 0)}, "
        f"in_flight={int(requests.get('in_flight') or 0)}"
    )
    print(f"Next action: {snapshot.get('next_action') or 'none'}")
    run_dir = ((snapshot.get('artifacts') or {}).get('run_dir') or '')
    if run_dir:
        print(f'Run directory: {run_dir}')


def run_durable_sync_command(args):
    """Dispatch #347's five durable commands through the pure service API."""
    from sync_run_contracts import RunStatus
    from sync_run_service import SyncRunService
    from sync_run_store import SyncRunStore

    command = str(args.command or '')
    root_dir = _durable_sync_root_dir()
    if command == 'sync-status':
        run_id = str(getattr(args, 'run', '') or '').strip()
        latest = bool(getattr(args, 'latest', False))
        if bool(run_id) == latest:
            raise cli_contract.MachineContractError(
                'sync-status requires exactly one of RUN or --latest.',
                code_name='INVALID_RUN_SELECTOR',
                suggested_action='pass_run_or_latest',
                details={'semantic_exit_code': cli_contract.EXIT_INVALID_STATE},
            )
        snapshot = _durable_sync_store_only_service().status(
            run_id or None,
            latest=latest,
        )
    elif command == 'sync-start':
        service, context = _durable_sync_production_service(require_provider=True)
        if not context.plan_build.requests:
            raise cli_contract.MachineContractError(
                'No pending translations are available for durable Sync.',
                code_name='SYNC_RUN_NO_WORK',
                suggested_action='inspect_project_scope',
                details={'semantic_exit_code': cli_contract.EXIT_INVALID_STATE},
            )
        snapshot = service.start(
            context.plan_build,
            client_token=getattr(args, 'client_token', '') or None,
        )
    elif command == 'sync-resume':
        run_id = str(args.run)
        store = SyncRunStore(root_dir, run_id)
        status = RunStatus(str(store.get_run()['status']))
        if status in {RunStatus.CANCELLED, RunStatus.COMPLETED,
                      RunStatus.COMPLETED_WITH_ERRORS, RunStatus.FAILED}:
            legacy.load_translator_settings(persist_corrected_game_root=False)
            service = _durable_sync_store_only_service(deliver_usage=True)
        else:
            service, _context = _durable_sync_production_service(
                require_provider=True
            )
        snapshot = service.resume(run_id)
    elif command == 'sync-cancel':
        class _NoDispatchBackend:
            def send(self, *_args, **_kwargs):
                raise RuntimeError('cancel closeout must not dispatch')

            def cancel(self, *, attempt):
                return False

        service = SyncRunService(
            root_dir,
            backend_factory=lambda _store: _NoDispatchBackend(),
            derived_builder_factory=lambda _store: (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError('cancel closeout must not derive requests')
                )
            ),
        )
        snapshot = service.cancel(str(args.run))
    elif command == 'sync-derive':
        service, context = _durable_sync_production_service(require_provider=True)
        snapshot = service.derive(
            str(args.run),
            context.plan_build,
            retry_unknown=bool(getattr(args, 'retry_unknown', False)),
            ack_duplicate_billing_risk=bool(
                getattr(args, 'ack_duplicate_billing_risk', False)
            ),
            exclude_unknown=bool(getattr(args, 'exclude_unknown', False)),
        )
    else:
        raise SystemExit(f'Unknown durable Sync command: {command}')
    _print_durable_sync_snapshot(snapshot)
    return snapshot


def dispatch_command(parser, args):
    command = args.command
    if command is None:
        parser.print_help()
        return
    if command == 'capabilities':
        payload = cli_discovery.capabilities(
            parser,
            cli_version=__version__,
            machine_output_commands=MACHINE_OUTPUT_COMMANDS,
            explicit_target_commands=EXPLICIT_TARGET_COMMANDS,
            result_schema_version=cli_contract.CLI_SCHEMA_VERSION,
        )
        return _write_machine_document(
            payload,
            args,
            command_completed=True,
            workflow_started=True,
        )

    if command == 'schema':
        payload = cli_discovery.command_schema(parser, args.schema_command)
        return _write_machine_document(
            payload,
            args,
            command_completed=True,
            workflow_started=True,
        )


    if command in MACHINE_OUTPUT_COMMANDS:
        validate_machine_invocation(args)

    if command in DURABLE_SYNC_COMMANDS:
        return run_durable_sync_command(args)

    if command == 'doctor':
        # doctor is read-only: never persist auto-corrected game_root.
        legacy.load_translator_settings(persist_corrected_game_root=False)
        legacy.load_glossary()
        load_batch_settings(tolerate_routing_errors=True)
        print_banner()
        report = collect_doctor_report()
        print_doctor_report(report)
        return report

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

    if command in {
        'final-review-build',
        'final-review-status',
        'final-review-export',
        'final-review-resume',
        'final-review-ingest-results',
        'final-review-create-revisions',
    }:
        legacy.load_translator_settings(persist_corrected_game_root=False)
        legacy.load_glossary()
        load_batch_settings()
        if command == 'final-review-build':
            print_banner()
            return (
                create_final_review_package(
                    display_name_override=getattr(args, 'display_name', '') or '',
                    skip_prepare=bool(getattr(args, 'skip_prepare', False)),
                    chunk_size=getattr(args, 'chunk_size', 0) or None,
                    allow_pending=bool(getattr(args, 'allow_pending', False)),
                )
                or ''
            )
        if command == 'final-review-status':
            if not getattr(args, 'json', False):
                print_banner()
            return run_final_review_status(
                getattr(args, 'target', '') or None,
                as_json=bool(getattr(args, 'json', False)),
            )
        if command == 'final-review-export':
            print_banner()
            return run_final_review_export(
                getattr(args, 'target', '') or None,
                output_jsonl=getattr(args, 'jsonl', '') or '',
                output_markdown=getattr(args, 'markdown', '') or '',
            )
        if command == 'final-review-resume':
            print_banner()
            return run_final_review_resume(
                getattr(args, 'target', '') or None,
                force=bool(getattr(args, 'force', False)),
            )
        if command == 'final-review-ingest-results':
            print_banner()
            return run_final_review_ingest_results(
                getattr(args, 'target', '') or None,
                result_path=getattr(args, 'result', '') or '',
                allow_stale_results=bool(getattr(args, 'allow_stale_results', False)),
            )

        if command == 'final-review-create-revisions':
            print_banner()
            return run_final_review_create_revisions(
                getattr(args, 'target', '') or None,
                finding_ids=getattr(args, 'finding_id', []) or [],
            )
    if command in {
        'project-analysis-status',
        'project-analysis-ingest-keywords',
        'project-analysis-build-structure',
        'project-analysis-generate',
        'project-analysis-inspect',
        'project-analysis-diff',
        'project-analysis-publish',
        'project-analysis-unpublish',
    }:
        from project_analysis import (
            ProjectAnalysisError,
            collect_project_analysis_status,
            format_brief_diff,
            print_status,
            publish_project_brief,
            resolve_project_analysis_store,
            unpublish_project_brief,
        )
        from project_analysis_generate import (
            build_structure_drafts,
            ingest_keyword_summaries,
        )
        from project_analysis_llm import run_mapreduce_drafts

        store_dir = getattr(args, 'store_dir', None) or None
        as_json = bool(getattr(args, 'json', False))
        source_fp = getattr(args, 'source_fingerprint', '') or ''

        def _load_settings_quiet():
            if not store_dir or command == 'project-analysis-generate':
                # generate needs API keys from load_config() for run_sync_request.
                if command == 'project-analysis-generate':
                    try:
                        legacy.load_config()
                    except model_profile.ModelRoutingConfigError as exc:
                        raise model_profile.routing_resolution_error(
                            exc,
                            stage=model_profile.STAGE_PROJECT_ANALYSIS,
                        ) from exc
                legacy.load_translator_settings(persist_corrected_game_root=False)
                load_batch_settings()

        def _run_pa():
            _load_settings_quiet()
            if command == 'project-analysis-status':
                return collect_project_analysis_status(
                    store_dir=store_dir,
                    expected_source_fingerprint=source_fp,
                )
            if command == 'project-analysis-ingest-keywords':
                return ingest_keyword_summaries(
                    args.summary_jsonl,
                    store_dir=store_dir,
                    base_dir=legacy.BASE_DIR or None,
                )
            if command == 'project-analysis-build-structure':
                return build_structure_drafts(
                    store_dir=store_dir,
                    base_dir=legacy.BASE_DIR or None,
                    script_roots=args.script_root or None,
                    entry_labels=args.entry_label or None,
                )
            if command == 'project-analysis-generate':
                cli_model = str(getattr(args, 'model', '') or '').strip()
                analysis_overrides = (
                    {model_profile.STAGE_PROJECT_ANALYSIS: cli_model}
                    if cli_model
                    else None
                )
                analysis_plan = freeze_runtime_routing_plan(
                    execution=model_profile.ExecutionStrategy.SYNC,
                    stage_overrides=analysis_overrides,
                    required_stages={model_profile.STAGE_PROJECT_ANALYSIS},
                )
                analysis_route = analysis_plan.routes[model_profile.STAGE_PROJECT_ANALYSIS]
                model = route_model(analysis_plan, analysis_route)
                analysis_usage_run_id = model_usage_ledger.new_run_id('project-analysis')
                analysis_usage_operation_id = (
                    'project-analysis-'
                    + hashlib.sha256(str(legacy.BASE_DIR or '').encode('utf-8')).hexdigest()[:20]
                )
                _generate = build_project_analysis_sync_runner(
                    analysis_plan,
                    analysis_route,
                )

                pricing_config = batch_cost_estimate.load_pricing_config(
                    _read_translator_config_object()
                )
                model_rates = (
                    batch_cost_estimate.resolve_model_pricing(model, pricing_config) or {}
                )

                def _record_project_analysis_usage(event):
                    result = event.get('result')
                    if result is None:
                        return
                    usage_result = _sync_result_to_dict(result)
                    usage_result['output_diagnostics'] = (
                        event.get('output_diagnostics') or {}
                    )
                    usage_result['request_metadata'] = (
                        event.get('request_metadata') or {}
                    )
                    record_generation_usage_best_effort(
                        task_mode='analysis',
                        stage=str(event.get('stage') or 'project_analysis'),
                        result=usage_result,
                        operation_id=analysis_usage_operation_id,
                        run_id=analysis_usage_run_id,
                        source_key=str(event.get('artifact_id') or ''),
                        thinking_level=PROJECT_ANALYSIS_THINKING_LEVEL,
                        source={
                            'kind': 'project_analysis_response',
                            'store_dir': str(store_dir or ''),
                            'artifact_id': str(event.get('artifact_id') or ''),
                            'stage': str(event.get('stage') or ''),
                        },
                        pricing_config=pricing_config,
                    )

                def _project_analysis_progress(event):
                    print(
                        "PROJECT_ANALYSIS_PROGRESS "
                        + json.dumps(dict(event), ensure_ascii=False, sort_keys=True),
                        file=sys.stderr,
                        flush=True,
                    )
                return run_mapreduce_drafts(
                    store_dir=store_dir,
                    base_dir=legacy.BASE_DIR or None,
                    generate=_generate,
                    config={
                        'model': model,
                        'thinking_level': PROJECT_ANALYSIS_THINKING_LEVEL,
                        'timeout_seconds': SYNC_TIMEOUT_SECONDS,
                        'max_label_summary_chars': PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS,
                        'max_route_summary_chars': PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS,
                        'max_brief_chars': PROJECT_ANALYSIS_MAX_BRIEF_CHARS,
                        'max_input_chars_per_request': PROJECT_ANALYSIS_MAX_INPUT_CHARS,
                        'max_output_tokens': PROJECT_ANALYSIS_MAX_OUTPUT_TOKENS,
                    },
                    force=bool(getattr(args, 'force', False)),
                    provider=SYNC_BACKEND or 'gemini',
                    model=model,
                    progress=_project_analysis_progress,
                    usage_recorder=_record_project_analysis_usage,
                    pricing={
                        "currency": pricing_config.get("currency") or "USD",
                        "input_per_million": model_rates.get("input_per_million") or 0.0,
                        "output_per_million": model_rates.get("output_per_million") or 0.0,
                    },
                    analysis_inputs=collect_project_analysis_optional_inputs(
                        store_dir=store_dir,
                        base_dir=legacy.BASE_DIR or None,
                    ),
                )
            if command == 'project-analysis-inspect':
                store = resolve_project_analysis_store(
                    store_dir, base_dir=legacy.BASE_DIR or None
                )
                kind = getattr(args, 'kind', 'status') or 'status'
                if kind == 'status':
                    return store.collect_status()
                if kind == 'chunks':
                    return {'chunks': store.load_summaries('chunk')}
                if kind == 'labels':
                    return {'labels': store.load_summaries('label')}
                if kind == 'routes':
                    return {'routes': store.load_routes()}
                if kind == 'brief':
                    return {
                        'draft': store.load_brief_text(published=False),
                        'published': store.load_brief_text(published=True),
                    }
                return store.collect_status()
            if command == 'project-analysis-diff':
                return format_brief_diff(store_dir, base_dir=legacy.BASE_DIR or None)
            if command == 'project-analysis-publish':
                return publish_project_brief(
                    store_dir,
                    base_dir=legacy.BASE_DIR or None,
                    force=bool(getattr(args, 'force', False)),
                    current_source_fingerprint=getattr(args, 'source_fingerprint', '')
                    or '',
                )
            if command == 'project-analysis-unpublish':
                return unpublish_project_brief(
                    store_dir, base_dir=legacy.BASE_DIR or None
                )
            raise SystemExit(f'Unknown project-analysis command: {command}')

        try:
            if command == 'project-analysis-status' and not as_json:
                result = _run_pa()
                print_banner()
                print_status(result)
                return
            # JSON / machine actions: keep stdout clean when possible.
            quiet = command != 'project-analysis-status' or as_json
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = _run_pa()
            else:
                result = _run_pa()
            if command == 'project-analysis-status' and as_json:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            elif command != 'project-analysis-status':
                if command in {
                    'project-analysis-inspect',
                    'project-analysis-diff',
                    'project-analysis-ingest-keywords',
                    'project-analysis-build-structure',
                    'project-analysis-generate',
                    'project-analysis-publish',
                    'project-analysis-unpublish',
                }:
                    # Drop full graph from build output for readability unless tiny.
                    if (
                        command == 'project-analysis-build-structure'
                        and isinstance(result, dict)
                        and 'graph' in result
                    ):
                        result = dict(result)
                        result['graph'] = {
                            'label_count': len((result['graph'] or {}).get('labels') or {}),
                            'route_count': len((result['graph'] or {}).get('routes') or []),
                            'unresolved_edges': len(
                                (result['graph'] or {}).get('unresolved_edges') or []
                            ),
                        }
                    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        except ProjectAnalysisError as exc:
            raise SystemExit(f'Project analysis error: {exc}') from exc

    if command == 'export-project-snapshot':
        # Read-only scan: do not persist an auto-corrected game_root and do not
        # require provider configuration or API keys.
        legacy.load_translator_settings(persist_corrected_game_root=False)
        load_batch_settings()
        try:
            return run_project_snapshot_export(
                version_id=getattr(args, 'version_id', ''),
                version_label=getattr(args, 'version_label', '') or '',
                source_revision=getattr(args, 'source_revision', '') or '',
                output_dir=getattr(args, 'output_dir', '') or None,
                coverage_review_path=getattr(args, 'coverage_review', '') or None,
            )
        except ValueError as exc:
            raise SystemExit(f'Project snapshot error: {exc}') from exc

    if command == 'reconcile-project-snapshots':
        # Saved-artifact comparison is fully offline and does not inspect the
        # active project or load translator configuration.
        try:
            return run_project_snapshot_reconciliation(
                getattr(args, 'base', ''),
                getattr(args, 'target', ''),
                output_dir=getattr(args, 'output_dir', '') or None,
            )
        except ValueError as exc:
            raise SystemExit(f'Project reconciliation error: {exc}') from exc

    if command == 'build-translation-records':
        # Offline artifact export from a saved snapshot plus one Batch package.
        try:
            return run_translation_records_export(
                getattr(args, 'snapshot', ''),
                getattr(args, 'manifest', ''),
                origin=getattr(args, 'origin', 'model_initial'),
                previous_records_path=getattr(args, 'previous_records', '') or '',
                output_dir=getattr(args, 'output_dir', '') or None,
            )
        except ValueError as exc:
            raise SystemExit(f'Translation records error: {exc}') from exc

    if command == 'build-reuse-candidates':
        # Offline derivation from saved P3/P4 artifacts only.
        try:
            return run_reuse_candidates_build(
                getattr(args, 'base_snapshot', ''),
                getattr(args, 'target_snapshot', ''),
                getattr(args, 'reconciliation', ''),
                getattr(args, 'base_records', ''),
                output_dir=getattr(args, 'output_dir', '') or None,
            )
        except ValueError as exc:
            raise SystemExit(f'Reuse candidates error: {exc}') from exc

    if command == 'import-reuse-decisions':
        # Offline decision import; freshness is validated against live inputs.
        try:
            return run_reuse_decisions_import(
                getattr(args, 'reuse', ''),
                getattr(args, 'decisions', ''),
                base_snapshot_path=getattr(args, 'base_snapshot', '') or '',
                target_snapshot_path=getattr(args, 'target_snapshot', '') or '',
                reconciliation_path=getattr(args, 'reconciliation', '') or '',
                base_records_path=getattr(args, 'base_records', '') or '',
                output_dir=getattr(args, 'output_dir', '') or None,
            )
        except ValueError as exc:
            raise SystemExit(f'Reuse decisions error: {exc}') from exc

    if command == 'export-reuse-results':
        # Writes only the canonical results JSONL plus manifest bookkeeping;
        # game-file writes still go through check -> apply.
        initialize_batch_logging()
        try:
            return run_reuse_results_export(
                getattr(args, 'reuse', ''),
                getattr(args, 'manifest', ''),
                base_snapshot_path=getattr(args, 'base_snapshot', '') or '',
                target_snapshot_path=getattr(args, 'target_snapshot', '') or '',
                reconciliation_path=getattr(args, 'reconciliation', '') or '',
                base_records_path=getattr(args, 'base_records', '') or '',
            )
        except ValueError as exc:
            raise SystemExit(f'Reuse results error: {exc}') from exc

    initialize_batch_logging()
    if command == 'export-revision-corpus':
        # Read-only export: avoid the common load path below, which can
        # persist a corrected game_root into translator_config.json.
        legacy.load_translator_settings(persist_corrected_game_root=False)
        legacy.load_glossary()
        load_batch_settings()
        return run_revision_corpus_export(
            getattr(args, 'output_dir', '') or None,
        )

    if command == 'import-revision-proposals':
        # Local candidate conversion and preview only: no provider/API setup,
        # prepare command, or game-file write is allowed here.
        legacy.load_translator_settings(persist_corrected_game_root=False)
        legacy.load_glossary()
        load_batch_settings()
        return import_revision_proposals(
            args.proposal,
            corpus_manifest_path=args.corpus_manifest,
            stage=bool(getattr(args, 'stage', False)),
            operation_identity=getattr(args, 'operation_identity', '') or '',
        )

    if command == 'confirm-revision-proposals':
        # Confirmation is still local-only.  The core rechecks the staged
        # session, current project/source snapshots, and the ordinary preview
        # gates before any apply command can become eligible.
        legacy.load_translator_settings(persist_corrected_game_root=False)
        legacy.load_glossary()
        load_batch_settings()
        return confirm_revision_proposals(
            args.staged_selection,
            args.selection_file,
        )

    if command in {'usage-import', 'usage-report'}:
        as_json = bool(getattr(args, 'json', False))

        def _run_usage_command():
            legacy.load_config(require_api_key=False)
            legacy.load_translator_settings(persist_corrected_game_root=False)
            load_batch_settings()
            if command == 'usage-import':
                manifest = load_manifest(getattr(args, 'target', '') or None)
                result = import_manifest_usage(manifest)
                result['report'] = model_usage_ledger.query_usage(result['game_root'])
                return result
            if not legacy.BASE_DIR:
                raise model_usage_ledger.UsageLedgerError(
                    'game_root is required for usage-report'
                )
            return model_usage_ledger.query_usage(
                legacy.BASE_DIR,
                task=getattr(args, 'task', '') or '',
                stage=getattr(args, 'stage', '') or '',
                provider=getattr(args, 'provider', '') or '',
                model=getattr(args, 'model', '') or '',
                group_by=getattr(args, 'group_by', '') or '',
            )

        try:
            if as_json:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = _run_usage_command()
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return result
            result = _run_usage_command()
            if command == 'usage-import':
                print('Model usage import:')
                result_paths = result.get('result_paths') or [result.get('result_path')]
                for result_path in result_paths:
                    if result_path:
                        print(f"- result: {result_path}")
                print(f"- scanned rows: {int(result.get('scanned_rows') or 0)}")
                print(f"- inserted records: {int(result.get('inserted_records') or 0)}")
                print(f"- duplicate records: {int(result.get('duplicate_records') or 0)}")
                print(f"- ledger: {result.get('ledger_path') or ''}")
            else:
                for line in model_usage_ledger.format_usage_report(result):
                    print(line)
            return result
        except model_usage_ledger.UsageLedgerError as exc:
            raise SystemExit(f'Model usage ledger error: {exc}') from exc

    require_api_key = command not in OFFLINE_BATCH_COMMANDS
    try:
        legacy.load_config(require_api_key=require_api_key)
    except model_profile.ModelRoutingConfigError as exc:
        raise model_profile.routing_resolution_error(exc) from exc
    legacy.load_translator_settings()
    legacy.load_glossary()
    load_batch_settings()
    print_banner()

    if command == 'build':
        return create_batch_package(
            display_name_override=args.display_name,
            skip_prepare=args.skip_prepare,
        )

    if command == 'build-keywords':
        return (
            create_keyword_package(
                display_name_override=args.display_name,
                skip_prepare=(not args.prepare) or args.skip_prepare,
                chunk_size=args.chunk_size,
                max_candidates_per_chunk=args.max_candidates_per_chunk,
            )
            or ''
        )

    if command == 'build-revisions':
        return (
            create_revision_package(
                display_name_override=args.display_name,
                skip_prepare=args.skip_prepare,
                chunk_size=args.chunk_size,
            )
            or ''
        )

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
        return submit_manifest(
            target=args.target or None,
            display_name_override=args.display_name,
            model_override=args.model,
            max_cost=args.max_cost,
            force_resubmit=args.force,
            resume_upload=args.resume,
        )

    if command == 'recover-submit':
        recover_submit_manifest(
            target=args.target or None,
            verify_remote=not args.no_verify,
        )
        return

    if command == 'status':
        return show_status(args.target or None)

    if command == 'check':
        return check_results(args.target or None)

    if command in {'quality-ack', 'quality-unack'}:
        result = quality_acknowledge_command(
            args.target or None,
            finding_ids=tuple(getattr(args, 'finding_ids', []) or []),
            all_findings=bool(getattr(args, 'all_findings', False)),
            unack=command == 'quality-unack',
        )
        manifest = result['manifest']
        print_quality_acknowledgement_summary(
            manifest,
            result['findings'],
            result['new_gate'],
            result['unmatched'],
        )
        return result

    if command == 'probe':
        probe_requests(
            target=args.target or None,
            limit=args.limit,
            offset=args.offset,
            api_key_index=args.api_key_index,
        )
        return

    if command == 'download':
        return download_results(args.target or None, force=args.force)

    if command == 'apply':
        return apply_results(args.target or None, force=args.force)

    if command == 'export-keywords':
        return export_keyword_candidates(
            target=args.target or None,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
            output_summary_jsonl=args.summary_jsonl,
            output_summary_markdown=args.summary_markdown,
        )

    if command == 'merge-keywords-to-glossary':
        candidates_path = keyword_glossary_merge.resolve_keyword_candidates_path(args.target)
        glossary_path = args.glossary.strip() if args.glossary else legacy.GLOSSARY_FILE
        dry_run = args.dry_run or args.preview
        interactive = not args.yes and not dry_run
        if interactive and (
            cli_contract.machine_output_active()
            or getattr(args, 'non_interactive', False)
        ):
            message = (
                'merge-keywords-to-glossary prompts for review; '
                'pass --yes or --dry-run to run non-interactively.'
            )
            if cli_contract.machine_output_active():
                raise cli_contract.MachineContractError(
                    message,
                    code_name='INTERACTIVE_REVIEW_UNSUPPORTED',
                    suggested_action='pass_yes_or_dry_run',
                    details={'command': 'merge-keywords-to-glossary'},
                )
            raise SystemExit(message)
        summary = keyword_glossary_merge.merge_keywords_to_glossary(
            candidates_path,
            glossary_path,
            dry_run=dry_run,
            min_confidence=max(0.0, float(args.min_confidence or 0.0)),
            accept_confidence=args.accept_confidence,
            overwrite=args.overwrite,
            interactive=interactive,
            backup=not args.no_backup,
            allow_history_review=bool(args.yes),
        )
        return (
            asdict(summary)
            if isinstance(summary, keyword_glossary_merge.MergeSummary)
            else summary
        )

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
        print_sync_output_summary(summary.get('output_summary') or {})
        return

    if command == 'preview-revisions':
        return preview_revisions(
            target=args.target or None,
            output_jsonl=args.jsonl,
            output_markdown=args.markdown,
        )

    if command == 'apply-revisions':
        apply_revisions(args.target or None, force=args.force)
        return

    if command == 'sync-keywords':
        return sync_keyword_candidates(
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

    if command == 'sync-revisions':
        return sync_revisions(
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


def _nonempty_artifacts(**paths):
    return {
        name: value
        for name, value in paths.items()
        if value not in (None, '')
    }


def _machine_manifest_summary(manifest):
    return {
        'mode': manifest.get('mode', ''),
        'job_name': manifest.get('job_name', ''),
        'job_state': manifest.get('job_state', ''),
        'summary': dict(manifest.get('summary') or {}),
        'batch_stats': dict(manifest.get('batch_stats') or {}),
        'last_status_checked_at': manifest.get('last_status_checked_at', ''),
        'downloaded_at': manifest.get('downloaded_at', ''),
        'applied_at': manifest.get('applied_at', ''),
    }


def _load_machine_manifest(command, value, args):
    if isinstance(value, dict):
        return value
    builder_commands = {
        'build',
        'submit',
        'build-revisions',
        'build-keywords',
        'final-review-build',
    }
    if command in builder_commands and value:
        return load_manifest(value)
    return load_manifest(getattr(args, 'target', '') or None)


def build_machine_success_envelope(command, value, args):
    """Translate existing command return values into the versioned CLI contract."""

    if command in DURABLE_SYNC_COMMANDS:
        snapshot = dict(value or {})
        raw_artifacts = dict(snapshot.pop('artifacts', {}) or {})
        artifacts = _nonempty_artifacts(
            run_dir=raw_artifacts.get('run_dir'),
            state_db=raw_artifacts.get('state_db'),
            plan=raw_artifacts.get('plan_json'),
            requests=raw_artifacts.get('requests_jsonl'),
            manifest=raw_artifacts.get('run_manifest_json'),
            results=raw_artifacts.get('results_jsonl'),
            result_sha256=raw_artifacts.get('results_sha256'),
            events=raw_artifacts.get('events_jsonl'),
        )
        return cli_contract.success_envelope(
            command,
            status=str(snapshot.get('run_status') or 'unknown'),
            result=snapshot,
            artifacts=artifacts,
        )

    if (
        command == 'apply'
        and isinstance(value, dict)
        and isinstance(value.get('durable_check_binding'), dict)
    ):
        manifest = dict(value)
        last_result = str(manifest.get('last_apply_result') or 'applied')
        return cli_contract.success_envelope(
            command,
            status=last_result,
            result={
                'apply': {
                    'state': str(manifest.get('state') or ''),
                    'last_apply_result': last_result,
                    'applied_files': list(manifest.get('applied_files') or []),
                    'summary': dict(manifest.get('summary') or {}),
                },
            },
            artifacts=_nonempty_artifacts(
                manifest=manifest.get('_manifest_path'),
                quality_findings=manifest.get('last_quality_findings_path'),
            ),
        )

    if command == 'doctor':
        report = dict(value or {})
        recommendations = list(report.get('recommendations') or [])
        blocked = doctor_rec.recommendations_block_workflow_state(recommendations)
        status = (
            'blocked'
            if blocked
            else report.get('workflow_state') or report.get('mode') or 'ready'
        )
        return cli_contract.success_envelope(
            command,
            status=status,
            result=report,
            warnings=report.get('warnings') or [],
        )

    if command in {'build', 'submit'} and not value:
        return cli_contract.success_envelope(
            command,
            status='no_work',
            result={'reason': 'no_pending_translation_work'},
        )

    if (
        command
        in {
            'build-revisions',
            'build-keywords',
            'final-review-build',
            'sync-revisions',
            'sync-keywords',
        }
        and not value
    ):
        return cli_contract.success_envelope(
            command,
            status='no_work',
            result={'reason': 'no_source_items'},
        )

    if command in {'quality-ack', 'quality-unack'}:
        payload = dict(value or {})
        manifest = payload.get('manifest')
        summary = payload.get('summary') or {}
        quality_gate = payload.get('new_gate') or summary.get('quality_gate') or {}
        previous_gate = payload.get('old_gate') or {}
        selected_finding_ids = sorted(payload.get('selected_ids') or [])
        unmatched_finding_ids = sorted(payload.get('unmatched') or [])
        acknowledged_finding_ids = sorted(
            payload.get('acknowledged_finding_ids') or []
        )
        previous_acknowledged_finding_ids = sorted(
            payload.get('previous_acknowledged_finding_ids') or []
        )
        result = {
            'manifest_path': (
                manifest.get('_manifest_path')
                if isinstance(manifest, dict)
                else ''
            ),
            'quality_gate': dict(quality_gate),
            'previous_quality_gate': dict(previous_gate),
            'acknowledged_finding_ids': acknowledged_finding_ids,
            'selected_finding_ids': selected_finding_ids,
            'unmatched_finding_ids': unmatched_finding_ids,
        }
        requested = bool(
            getattr(args, 'finding_ids', None)
            or getattr(args, 'all_findings', False)
        )
        if previous_acknowledged_finding_ids != acknowledged_finding_ids:
            status = 'updated'
        elif requested:
            status = 'no_work'
        else:
            status = 'listed'
        return cli_contract.success_envelope(
            command,
            status=status,
            result=result,
            artifacts=_nonempty_artifacts(
                manifest=result['manifest_path'],
                quality_findings=(
                    resolve_quality_findings_path(manifest)
                    if isinstance(manifest, dict)
                    else ''
                ),
            ),
        )

    if command in {'import-revision-proposals', 'confirm-revision-proposals'}:
        imported = dict(value or {})
        paths = dict(imported.get('paths') or {})
        result = {
            'input_count': int(imported.get('input_count') or 0),
            'requested_selected_count': int(
                imported.get('requested_selected_count') or 0
            ),
            'selected_count': int(imported.get('selected_count') or 0),
            'candidate_count': int(imported.get('candidate_count') or 0),
            'valid_count': int(imported.get('valid_count') or 0),
            'selectable_count': int(imported.get('selectable_count') or 0),
            'unselected_count': int(imported.get('unselected_count') or 0),
            'invalid_count': int(imported.get('invalid_count') or 0),
            'stale_count': int(imported.get('stale_count') or 0),
            'conflict_count': int(imported.get('conflict_count') or 0),
            'no_op_count': int(imported.get('no_op_count') or 0),
            'diagnostics': list(imported.get('diagnostics') or []),
            'preview_summary': dict(imported.get('preview_summary') or {}),
            'suggested_action': imported.get('suggested_action') or '',
            'session_status': imported.get('session_status') or '',
            'operation_identity': imported.get('operation_identity') or '',
            'staged_selection_digest': imported.get('staged_selection_digest') or '',
            'selection_digest': imported.get('selection_digest') or '',
            'selected_identity_v2': list(imported.get('selected_identity_v2') or []),
        }
        if command == 'import-revision-proposals' and (
            getattr(args, 'stage', False) or imported.get('stage')
        ):
            result['candidates'] = list(imported.get('candidates') or [])
        return cli_contract.success_envelope(
            command,
            status=str(imported.get('status') or 'blocked'),
            result=result,
            artifacts={
                'manifest': paths.get('manifest') or '',
                'staged_selection': paths.get('staged_selection') or '',
                'selection': paths.get('selection') or '',
                'import_report': paths.get('import_report') or '',
                'import_report_markdown': paths.get('import_report_markdown') or '',
                'staged_selection_report': paths.get('staged_selection_report') or '',
                'staged_selection_report_markdown': paths.get('staged_selection_report_markdown') or '',
                'selection_confirmation_output_dir': paths.get('output_dir') or '',
                'selection_confirmation_report': paths.get('selection_confirmation_report') or '',
                'selection_confirmation_report_markdown': paths.get('selection_confirmation_report_markdown') or '',
                'revision_preview_jsonl': paths.get('revision_preview_jsonl') or '',
                'revision_preview_markdown': paths.get('revision_preview_markdown') or '',
            },
        )

    manifest = _load_machine_manifest(command, value, args)
    result = _machine_manifest_summary(manifest)
    artifacts = _nonempty_artifacts(
        manifest=manifest.get('_manifest_path'),
        input_jsonl=manifest.get('input_jsonl_path'),
        result_jsonl=manifest.get('result_jsonl_path'),
        result_sha256=(
            f"{manifest.get('result_jsonl_path')}.sha256"
            if manifest.get('result_jsonl_sha256') and manifest.get('result_jsonl_path')
            else ''
        ),
        status_snapshot=manifest.get('last_status_snapshot_path'),
        check_report=manifest.get('last_check_report_path'),
        quality_findings=manifest.get('last_quality_findings_path'),
        apply_failure_report=manifest.get('last_apply_failure_report_path'),
    )
    warnings = list(manifest.get('build_warnings') or [])
    status = 'completed'

    if command in {'build', 'build-revisions', 'build-keywords', 'final-review-build'}:
        status = str(manifest.get('job_state') or 'LOCAL_ONLY')
        if command == 'build':
            result['cost_estimate'] = dict(manifest.get('cost_estimate') or {})
    elif command in {'submit', 'status'}:
        status = str(manifest.get('job_state') or 'unknown')
        if manifest.get('job_error'):
            result['job_error'] = manifest.get('job_error')
    elif command == 'download':
        status = 'downloaded'
        result['result_jsonl_sha256'] = manifest.get('result_jsonl_sha256', '')
    elif command == 'check':
        check_summary = dict(manifest.get('last_check_summary') or {})
        result['check'] = check_summary
        status = str(
            check_summary.get('check_status')
            or check_summary.get('safety_level')
            or 'unknown'
        )
    elif command == 'apply':
        result['apply'] = dict(manifest.get('apply_summary') or {})
        result['apply']['next_split_manifest'] = manifest.get('next_split_manifest_path', '')
        status = 'applied' if manifest.get('applied_at') else 'completed'
    elif command == 'apply-revisions' or (
        command == 'sync-revisions' and getattr(args, 'apply', False)
    ):
        result['revision_apply'] = dict(manifest.get('revision_apply_summary') or {})
        result['revision_apply_state'] = manifest.get('revision_apply_state') or ''
        if manifest.get('revision_apply_blocked_reason'):
            result['revision_apply_blocked_reason'] = manifest.get(
                'revision_apply_blocked_reason'
            )
        status = str(
            manifest.get('revision_apply_state')
            or ('applied' if manifest.get('revision_applied_at') else 'completed')
        )
    elif command in {'preview-revisions', 'final-review-create-revisions'} or (
        command == 'sync-revisions' and not getattr(args, 'apply', False)
    ):
        manifest = (
            value
            if isinstance(value, dict)
            else _load_machine_manifest(command, value, args)
        )
        preview = dict(manifest.get('last_revision_preview') or {})
        result = {
            'manifest_path': manifest.get('_manifest_path') or '',
            'preview_jsonl_path': preview.get('jsonl_path') or '',
            'preview_markdown_path': preview.get('markdown_path') or '',
            'check_status': preview.get('check_status') or '',
            'writeback_gate': dict(preview.get('writeback_gate') or {}),
            'quality_gate': dict(preview.get('quality_gate') or {}),
            'quality_findings_count': int(preview.get('quality_findings_count') or 0),
            'summary': dict(preview.get('summary') or {}),
        }
        if command == 'final-review-create-revisions':
            result['final_review_source'] = dict(
                manifest.get('final_review_source') or {}
            )
        return cli_contract.success_envelope(
            command,
            status=str(preview.get('check_status') or 'completed'),
            result=result,
            artifacts=_nonempty_artifacts(
                manifest=result['manifest_path'],
                revision_preview_jsonl=result['preview_jsonl_path'],
                revision_preview_markdown=result['preview_markdown_path'],
                quality_findings=preview.get('quality_findings_path') or '',
            ),
        )
    elif command in {'export-keywords', 'sync-keywords'}:
        export = dict(value or {})
        if command == 'export-keywords':
            keyword_manifest = load_manifest(getattr(args, 'target', '') or None)
            manifest_path = keyword_manifest.get('_manifest_path') or ''
        else:
            manifest_path = str(export.get('manifest_path') or '')
        result = {
            'manifest_path': manifest_path,
            'jsonl_path': export.get('jsonl_path') or '',
            'markdown_path': export.get('markdown_path') or '',
            'summary_jsonl_path': export.get('summary_jsonl_path') or '',
            'summary_markdown_path': export.get('summary_markdown_path') or '',
            'summary': dict(export.get('summary') or {}),
            'history_evidence': dict(export.get('history_evidence') or {}),
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts=_nonempty_artifacts(
                manifest=result['manifest_path'],
                keyword_candidates=result['jsonl_path'],
                keyword_candidates_markdown=result['markdown_path'],
                keyword_chunk_summaries=result['summary_jsonl_path'],
                keyword_chunk_summaries_markdown=result['summary_markdown_path'],
            ),
        )
    elif command == 'merge-keywords-to-glossary':
        summary = dict(value or {})
        dry_run = bool(summary.get('dry_run'))
        wrote_glossary = bool(summary.get('wrote_glossary'))
        result = {
            'candidates_path': summary.get('candidates_path') or '',
            'glossary_path': summary.get('glossary_path') or '',
            'candidates_read': int(summary.get('candidates_read') or 0),
            'accepted': int(summary.get('accepted') or 0),
            'overwritten': int(summary.get('overwritten') or 0),
            'skipped_duplicate': int(summary.get('skipped_duplicate') or 0),
            'skipped_low_confidence': int(summary.get('skipped_low_confidence') or 0),
            'skipped_empty': int(summary.get('skipped_empty') or 0),
            'skipped_user': int(summary.get('skipped_user') or 0),
            'dry_run': dry_run,
            'wrote_glossary': wrote_glossary,
        }
        status = 'previewed' if dry_run else ('merged' if wrote_glossary else 'no_work')
        return cli_contract.success_envelope(
            command,
            status=status,
            result=result,
            artifacts=_nonempty_artifacts(
                glossary=result['glossary_path'],
                glossary_backup=summary.get('backup_path') or '',
                keyword_candidates=result['candidates_path'],
            ),
        )
    elif command == 'final-review-status':
        campaign = dict(value or {})
        return cli_contract.success_envelope(
            command,
            status=str(campaign.get('status') or 'unknown'),
            result=campaign,
            artifacts=_nonempty_artifacts(
                manifest=campaign.get('manifest_path') or '',
            ),
        )
    elif command == 'final-review-export':
        export = dict(value or {})
        campaign = dict(export.get('status') or {})
        result = {
            'jsonl_path': export.get('jsonl_path') or '',
            'markdown_path': export.get('markdown_path') or '',
            'finding_count': int(export.get('finding_count') or 0),
            'campaign_status': campaign.get('status') or '',
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts=_nonempty_artifacts(
                findings_jsonl=result['jsonl_path'],
                findings_markdown=result['markdown_path'],
            ),
        )
    elif command == 'final-review-resume':
        resume = dict(value or {})
        paths = dict(resume.get('paths') or {})
        campaign = dict(resume.get('status') or {})
        run_count = int(resume.get('run_count') or 0)
        result = {
            'manifest_path': paths.get('manifest') or '',
            'package_dir': paths.get('package_dir') or '',
            'run_count': run_count,
            'skip_count': int(resume.get('skip_count') or 0),
            'to_run_unit_ids': list(resume.get('to_run_unit_ids') or []),
            'force': bool(getattr(args, 'force', False)),
            'campaign_status': campaign.get('status') or '',
        }
        return cli_contract.success_envelope(
            command,
            status='rebuilt' if run_count else 'no_work',
            result=result,
            artifacts=_nonempty_artifacts(
                manifest=result['manifest_path'],
                review_units=paths.get('review_units') or '',
                campaign_report=paths.get('report') or '',
            ),
        )
    elif command == 'final-review-ingest-results':
        ingest = dict(value or {})
        paths = dict(ingest.get('paths') or {})
        campaign = dict(ingest.get('status') or {})
        result = {
            'manifest_path': paths.get('manifest') or '',
            'package_dir': paths.get('package_dir') or '',
            'summary': dict(ingest.get('summary') or {}),
            'campaign_status': campaign.get('status') or '',
        }
        return cli_contract.success_envelope(
            command,
            status=str(campaign.get('status') or 'completed'),
            result=result,
            artifacts=_nonempty_artifacts(
                manifest=result['manifest_path'],
                findings_jsonl=paths.get('findings') or '',
                quality_findings=paths.get('quality_findings') or '',
                campaign_report=paths.get('report') or '',
            ),
        )
    elif command == 'export-project-snapshot':
        snapshot = dict(value or {})
        paths = dict(snapshot.get('paths') or {})
        coverage = dict(snapshot.get('coverage') or {})
        result = {
            'version_id': snapshot.get('version_id') or '',
            'engine': snapshot.get('engine') or '',
            'snapshot_digest': snapshot.get('snapshot_digest') or '',
            'occurrence_count': int(snapshot.get('occurrence_count') or 0),
            'coverage_status': coverage.get('coverage_status') or '',
            'review_status': coverage.get('review_status') or '',
            'review_policy_satisfied': bool(
                coverage.get('review_policy_satisfied')
            ),
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts={
                'project_snapshot': paths.get('snapshot') or '',
                'unit_occurrences': paths.get('occurrences') or '',
            },
        )
    elif command == 'reconcile-project-snapshots':
        reconciliation = dict(value or {})
        paths = dict(reconciliation.get('paths') or {})
        result = {
            'base_version_id': reconciliation.get('base_version_id') or '',
            'target_version_id': reconciliation.get('target_version_id') or '',
            'reconciliation_digest': (
                reconciliation.get('reconciliation_digest') or ''
            ),
            'summary': dict(reconciliation.get('summary') or {}),
            'coverage_changes': dict(
                reconciliation.get('coverage_changes') or {}
            ),
        }
        return cli_contract.success_envelope(
            command,
            status=str(reconciliation.get('status') or 'completed'),
            result=result,
            artifacts={
                'reconciliation_report': paths.get('report') or '',
                'reconciliation_items': paths.get('items') or '',
            },
        )
    elif command == 'build-translation-records':
        records = dict(value or {})
        paths = dict(records.get('paths') or {})
        result = {
            'version_id': records.get('version_id') or '',
            'snapshot_digest': records.get('snapshot_digest') or '',
            'record_count': int(records.get('record_count') or 0),
            'record_set_digest': records.get('record_set_digest') or '',
            'target_language': records.get('target_language') or '',
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts={
                'translation_records_manifest': paths.get('manifest') or '',
                'translation_records': paths.get('records') or '',
            },
        )
    elif command == 'build-reuse-candidates':
        candidates = dict(value or {})
        paths = dict(candidates.get('paths') or {})
        result = {
            'base_version_id': candidates.get('base_version_id') or '',
            'target_version_id': candidates.get('target_version_id') or '',
            'candidate_count': int(candidates.get('candidate_count') or 0),
            'candidate_set_digest': candidates.get('candidate_set_digest') or '',
            'reconciliation_digest': (
                candidates.get('reconciliation_digest') or ''
            ),
            'summary': dict(candidates.get('summary') or {}),
        }
        return cli_contract.success_envelope(
            command,
            status=str(candidates.get('status') or 'completed'),
            result=result,
            artifacts={
                'reuse_report': paths.get('report') or '',
                'reuse_candidates': paths.get('candidates') or '',
                'reuse_review': paths.get('review') or '',
                'reuse_decisions_template': (
                    paths.get('decisions_template') or ''
                ),
            },
        )
    elif command == 'import-reuse-decisions':
        decisions = dict(value or {})
        paths = dict(decisions.get('paths') or {})
        result = {
            'candidate_count': int(decisions.get('candidate_count') or 0),
            'candidate_set_digest': decisions.get('candidate_set_digest') or '',
            'decisions_applied': int(decisions.get('decisions_applied') or 0),
            'lineage_decisions': int(decisions.get('lineage_decisions') or 0),
            'summary': dict(decisions.get('summary') or {}),
        }
        return cli_contract.success_envelope(
            command,
            status=str(decisions.get('status') or 'completed'),
            result=result,
            artifacts={
                'reuse_report': paths.get('report') or '',
                'reuse_candidates': paths.get('candidates') or '',
                'reuse_review': paths.get('review') or '',
            },
        )
    elif command == 'export-reuse-results':
        reuse_results = dict(value or {})
        result = {
            'manifest_path': reuse_results.get('manifest_path') or '',
            'result_jsonl_path': reuse_results.get('result_jsonl_path') or '',
            'reused_items': int(reuse_results.get('reused_items') or 0),
            'parent_items_kept': int(reuse_results.get('parent_items_kept') or 0),
            'chunk_count': int(reuse_results.get('chunk_count') or 0),
            'unused_prefill_units': int(
                reuse_results.get('unused_prefill_units') or 0
            ),
            'candidate_set_digest': (
                reuse_results.get('candidate_set_digest') or ''
            ),
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts={
                'manifest': reuse_results.get('manifest_path') or '',
                'reuse_result_jsonl': (
                    reuse_results.get('result_jsonl_path') or ''
                ),
            },
        )
    elif command == 'export-revision-corpus':
        corpus = dict(value or {})
        paths = dict(corpus.get('paths') or {})
        scope = dict(corpus.get('scope') or {})
        source = dict(corpus.get('source') or {})
        result = {
            'output_dir': paths.get('output_dir') or '',
            'corpus_jsonl': paths.get('jsonl') or '',
            'corpus_markdown': paths.get('markdown') or '',
            'corpus_manifest': paths.get('manifest') or '',
            'file_count': scope.get('file_count') or 0,
            'item_count': scope.get('item_count') or 0,
            'source_changed_during_scan': bool(
                source.get('source_changed_during_scan')
            ),
        }
        return cli_contract.success_envelope(
            command,
            status='completed',
            result=result,
            artifacts={
                'corpus_manifest': paths.get('manifest') or '',
                'corpus_jsonl': paths.get('jsonl') or '',
                'corpus_markdown': paths.get('markdown') or '',
            },
        )
    return cli_contract.success_envelope(
        command,
        status=status,
        result=result,
        artifacts=artifacts,
        warnings=warnings,
    )


def _system_exit_message(exc):
    if isinstance(exc.code, str) and exc.code.strip():
        return exc.code.strip()
    if exc.code not in (None, 0):
        return f'Command exited with status {exc.code}.'
    return 'Command stopped before producing a result.'


def _machine_field_paths(args):
    paths = []
    for group in getattr(args, 'fields', []) or []:
        values = group if isinstance(group, (list, tuple)) else [group]
        for value in values:
            raw_items = str(value).split(',')
            if any(not item.strip() for item in raw_items):
                raise ValueError(f'Invalid field path list: {value!r}')
            for item in raw_items:
                path = item.strip()
                cli_contract.field_path_parts(path)
                paths.append(path)
    return list(dict.fromkeys(paths))


def _write_json_payload(document, args, *, record_output_artifact=False):
    output_file = str(getattr(args, 'output_file', '') or '').strip()
    payload = dict(document)
    if output_file and record_output_artifact:
        payload['artifacts'] = dict(payload.get('artifacts') or {})
        payload['artifacts']['output_file'] = os.path.abspath(output_file)
    field_paths = _machine_field_paths(args)
    if field_paths:
        payload = cli_contract.project_fields(payload, field_paths)

    compact = bool(getattr(args, 'compact', False))
    if output_file:
        atomic_write(
            output_file,
            lambda stream: cli_contract.write_json_envelope(
                payload,
                stream,
                compact=compact,
            ),
        )
        return
    cli_contract.write_json_envelope(payload, sys.stdout, compact=compact)


def _is_output_file_failure(document):
    error = document.get('error') if isinstance(document, dict) else None
    return isinstance(error, dict) and error.get('code') == 'OUTPUT_FILE_WRITE_FAILED'


def _output_file_failure_envelope(
    args,
    exc,
    *,
    workflow_started,
    command_completed,
    original_document=None,
):
    output_file = str(getattr(args, 'output_file', '') or '').strip()
    details = {
        'output_file': os.path.abspath(output_file) if output_file else '',
        'exception_type': exc.__class__.__name__,
        'reason': str(exc),
        'command_completed': bool(command_completed),
        'semantic_exit_code': cli_contract.EXIT_INVALID_STATE,
        'workflow_started': bool(workflow_started),
    }
    if isinstance(original_document, dict):
        if isinstance(original_document.get('status'), str):
            details['original_status'] = original_document['status']
        if isinstance(original_document.get('ok'), bool):
            details['original_ok'] = original_document['ok']
        original_error = original_document.get('error')
        if isinstance(original_error, dict) and original_error.get('code'):
            details['original_error_code'] = str(original_error['code'])
    return cli_contract.error_envelope(
        str(getattr(args, 'command', '') or ''),
        code='OUTPUT_FILE_WRITE_FAILED',
        message=f'Could not write JSON output file: {details["output_file"] or output_file}.',
        suggested_action='choose_writable_output_path',
        details=details,
    )


def _write_output_file_failure(
    args,
    exc,
    *,
    workflow_started,
    command_completed,
    original_document=None,
):
    envelope = _output_file_failure_envelope(
        args,
        exc,
        command_completed=command_completed,
        original_document=original_document,
        workflow_started=workflow_started,
    )
    print(
        f'Failed to write --output-file; returning structured error on stdout: {exc}',
        file=sys.stderr,
    )
    fallback_args = copy.copy(args)
    fallback_args.output_file = ''
    fallback_args.fields = []
    _write_json_payload(envelope, fallback_args)
    return envelope


def _write_machine_document(
    document,
    args,
    *,
    record_output_artifact=False,
    command_completed=False,
    workflow_started=False,
):
    """Write a document, falling back to stdout only for output-file failures."""

    try:
        _write_json_payload(
            document,
            args,
            record_output_artifact=record_output_artifact,
        )
    except (OSError, ValueError) as exc:
        if not str(getattr(args, 'output_file', '') or '').strip():
            raise
        return _write_output_file_failure(
            args,
            exc,
            command_completed=command_completed,
            original_document=document,
            workflow_started=workflow_started,
        )
    return document


def _candidate_manifest_paths_from_args(args):
    """Collect explicit or implied manifest paths for --output-file conflict checks."""

    candidates = []
    seen = set()

    def add_candidate(value):
        if not isinstance(value, str) or not value.strip():
            return
        raw = value.strip()
        abs_path = os.path.abspath(raw)
        if os.path.isdir(abs_path):
            abs_path = os.path.join(abs_path, 'manifest.json')
        key = _normalized_abs_path(abs_path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(abs_path)

    for attr in ('target', 'parent', 'retry'):
        add_candidate(getattr(args, attr, None))

    if not candidates and os.path.isfile(LATEST_MANIFEST_FILE):
        try:
            with open(LATEST_MANIFEST_FILE, 'r', encoding='utf-8') as handle:
                latest = handle.read().strip()
        except OSError:
            latest = ''
        add_candidate(latest)

    return candidates


def _collect_manifest_protected_paths(manifest_path):
    """Return task inputs and writeback targets associated with one manifest."""

    protected = [manifest_path]
    package_dir = os.path.dirname(manifest_path)
    try:
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return protected
    if not isinstance(manifest, dict):
        return protected

    manifest = dict(manifest)
    manifest['_manifest_path'] = manifest_path
    manifest['_package_dir'] = package_dir

    def add_path(value):
        if not isinstance(value, str) or not value.strip():
            return
        raw = value.strip()
        try:
            if os.path.isabs(raw):
                protected.append(_canonical_abs_path(raw))
            else:
                protected.append(
                    resolve_path_under_dir(package_dir, raw, 'protected output path')
                )
        except SystemExit:
            protected.append(os.path.abspath(os.path.join(package_dir, raw)))

    try:
        result_path = resolve_manifest_result_path(manifest)
    except SystemExit:
        result_path = os.path.join(package_dir, 'results.jsonl')
    protected.append(result_path)
    protected.append(f'{result_path}.sha256')

    for key in (
        'input_jsonl_path',
        'last_check_report_path',
        'last_status_snapshot_path',
        'last_apply_failure_report_path',
        'next_split_manifest_path',
        'last_retry_manifest_path',
    ):
        add_path(manifest.get(key))

    values = manifest.get('retry_children')
    if isinstance(values, list):
        for item in values:
            add_path(item)

    last_preview = manifest.get('last_revision_preview')
    if isinstance(last_preview, dict):
        add_path(last_preview.get('jsonl_path'))
        add_path(last_preview.get('markdown_path'))

    files = manifest.get('files')
    if isinstance(files, dict):
        for file_key, file_info in files.items():
            path_value = ''
            if isinstance(file_info, dict):
                path_value = file_info.get('path') or ''
            if isinstance(path_value, str) and path_value.strip() and os.path.isabs(path_value):
                add_path(path_value)
                continue
            try:
                protected.append(
                    resolve_manifest_file_path(
                        manifest,
                        file_key,
                        file_info if isinstance(file_info, dict) else {},
                    )
                )
            except (SystemExit, cli_contract.MachineContractError, OSError, TypeError, ValueError):
                continue

    return protected


def _collect_output_file_protected_paths(args):
    """Collect paths that --output-file must never overwrite."""

    protected = []
    seen = set()

    def add_path(value):
        if not isinstance(value, str) or not value.strip():
            return
        try:
            canonical = _canonical_abs_path(value)
            key = _normalized_abs_path(canonical)
        except (OSError, TypeError, ValueError):
            return
        if key in seen:
            return
        seen.add(key)
        protected.append(canonical)

    for attr in (
        'target',
        'base',
        'parent',
        'retry',
        'report',
        'coverage_review',
        'output_dir',
        'jsonl',
        'markdown',
        'summary_jsonl',
        'summary_markdown',
        'variants_file',
        'proposal',
        'corpus_manifest',
    ):
        value = getattr(args, attr, None)
        if not isinstance(value, str) or not value.strip():
            continue
        raw = value.strip()
        abs_path = os.path.abspath(raw)
        add_path(abs_path)
        if os.path.isdir(abs_path):
            add_path(os.path.join(abs_path, 'manifest.json'))

    command = str(getattr(args, 'command', '') or '')
    output_dir = str(getattr(args, 'output_dir', '') or '').strip()
    if output_dir and command == 'export-project-snapshot':
        add_path(os.path.join(output_dir, engine_versioning.DEFAULT_SNAPSHOT_FILENAME))
        add_path(
            os.path.join(
                output_dir,
                engine_versioning.DEFAULT_OCCURRENCES_FILENAME,
            )
        )
    if output_dir and command == 'reconcile-project-snapshots':
        add_path(
            os.path.join(
                output_dir,
                engine_versioning.DEFAULT_RECONCILIATION_FILENAME,
            )
        )
        add_path(
            os.path.join(
                output_dir,
                engine_versioning.DEFAULT_RECONCILIATION_ITEMS_FILENAME,
            )
        )
    if output_dir and command == 'build-translation-records':
        add_path(
            os.path.join(
                output_dir,
                engine_reuse.DEFAULT_RECORDS_MANIFEST_FILENAME,
            )
        )
        add_path(
            os.path.join(output_dir, engine_reuse.DEFAULT_RECORDS_FILENAME)
        )
    if output_dir and command in {'build-reuse-candidates', 'import-reuse-decisions'}:
        add_path(
            os.path.join(
                output_dir,
                engine_reuse.DEFAULT_REUSE_REPORT_FILENAME,
            )
        )
        add_path(
            os.path.join(output_dir, engine_reuse.DEFAULT_CANDIDATES_FILENAME)
        )

    # merge-keywords-to-glossary falls back to the active glossary file.
    glossary_value = getattr(args, 'glossary', None)
    if isinstance(glossary_value, str) and glossary_value.strip():
        add_path(os.path.abspath(glossary_value.strip()))
    elif str(getattr(args, 'command', '') or '') == 'merge-keywords-to-glossary':
        default_glossary = getattr(legacy, 'GLOSSARY_FILE', '') or ''
        if default_glossary:
            add_path(os.path.abspath(default_glossary))

    for manifest_path in _candidate_manifest_paths_from_args(args):
        add_path(manifest_path)
        if os.path.isfile(manifest_path):
            for path in _collect_manifest_protected_paths(manifest_path):
                add_path(path)

    return protected


def _find_output_file_path_conflict(args, output_target):
    """Return conflict details when --output-file collides with a task path."""

    output_key = _normalized_abs_path(output_target)
    for protected in _collect_output_file_protected_paths(args):
        if _normalized_abs_path(protected) == output_key:
            return {
                'output_file': _canonical_abs_path(output_target),
                'conflict_path': protected,
                'command': str(getattr(args, 'command', '') or ''),
            }
    return None


def _preflight_output_file(args):
    """Verify that an output target is safe and writable before workflow side effects."""

    output_file = str(getattr(args, 'output_file', '') or '').strip()
    if not output_file:
        return
    target = os.path.abspath(output_file)
    if os.path.isdir(target):
        raise IsADirectoryError(f'Output path is a directory: {target}')
    conflict = _find_output_file_path_conflict(args, target)
    if conflict:
        raise cli_contract.MachineContractError(
            (
                f'--output-file collides with a task input or writeback path: '
                f'{conflict["output_file"]} == {conflict["conflict_path"]}. '
                'Choose an independent report path.'
            ),
            code_name='OUTPUT_FILE_PATH_CONFLICT',
            suggested_action='choose_independent_output_file',
            details=conflict,
            semantic_exit_code=cli_contract.EXIT_USAGE,
        )
    directory = os.path.dirname(target) or os.curdir
    os.makedirs(directory, exist_ok=True)
    fd, probe_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(target)}.',
        suffix='.probe.tmp',
        dir=directory,
    )
    os.close(fd)
    os.unlink(probe_path)


def _output_file_failure_exit_code(args):
    if getattr(args, 'strict_exit_codes', False):
        return cli_contract.EXIT_INVALID_STATE
    return 1


def _write_machine_envelope(envelope, args, *, command_completed):
    return _write_machine_document(
        envelope,
        args,
        record_output_artifact=True,
        command_completed=command_completed,
        workflow_started=True,
    )


def _write_machine_usage_error(args, *, code, message, suggested_action):
    """Write a usage error without reapplying invalid output projection."""

    command = str(getattr(args, 'command', '') or '')
    safe_args = copy.copy(args)
    safe_args.fields = []
    envelope = cli_contract.error_envelope(
        command,
        code=code,
        message=message,
        suggested_action=suggested_action,
        details={'semantic_exit_code': cli_contract.EXIT_USAGE},
    )
    emitted = _write_machine_document(
        envelope,
        safe_args,
        record_output_artifact=command in MACHINE_OUTPUT_COMMANDS,
        command_completed=False,
    )
    if _is_output_file_failure(emitted):
        return _output_file_failure_exit_code(args)
    return cli_contract.EXIT_USAGE


def run_machine_command(parser, args):
    """Run one supported command with clean JSON stdout and text diagnostics."""

    command = str(args.command or '')
    try:
        with (
            cli_contract.machine_output_context(),
            contextlib.redirect_stdout(sys.stderr),
        ):
            value = dispatch_command(parser, args)
            envelope = build_machine_success_envelope(command, value, args)
    except SystemExit as exc:
        message = _system_exit_message(exc)
        legacy_exit_code = exc.code if isinstance(exc.code, int) and exc.code else 1
        if isinstance(exc, cli_contract.MachineContractError):
            details = dict(exc.details)
            details.update(
                {
                    'exit_code': legacy_exit_code,
                    'semantic_exit_code': exc.semantic_exit_code,
                }
            )
            envelope = cli_contract.error_envelope(
                command,
                code=exc.code_name,
                message=message,
                retryable=exc.retryable,
                suggested_action=exc.suggested_action,
                details=details,
            )
        elif args.strict_exit_codes:
            classification = cli_contract.classify_error(
                message,
                exception_type='SystemExit',
            )
            envelope = cli_contract.error_envelope(
                command,
                code=classification['code'],
                message=message,
                retryable=classification['retryable'],
                suggested_action=classification['suggested_action'],
                details={
                    'exit_code': legacy_exit_code,
                    'semantic_exit_code': classification['exit_code'],
                },
            )
        else:
            envelope = cli_contract.error_envelope(
                command,
                code='COMMAND_REFUSED',
                message=message,
                details={'exit_code': legacy_exit_code},
            )
        emitted = _write_machine_envelope(envelope, args, command_completed=False)
        if _is_output_file_failure(emitted):
            return _output_file_failure_exit_code(args)
        if args.strict_exit_codes:
            return cli_contract.strict_exit_code(emitted)
        return legacy_exit_code
    except Exception as exc:
        from sync_run_contracts import ErrorCode, SyncRunError

        if isinstance(exc, SyncRunError):
            invalid_codes = {
                ErrorCode.SYNC_RUN_NOT_FOUND,
                ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH,
                ErrorCode.SYNC_RUN_CLIENT_TOKEN_CONFLICT,
                ErrorCode.SYNC_RUN_SCHEMA_UNSUPPORTED,
                ErrorCode.SYNC_RUN_OUTCOME_UNKNOWN,
            }
            if exc.code is ErrorCode.SYNC_RUN_BUSY:
                semantic_exit = cli_contract.EXIT_RETRYABLE
            elif exc.code in invalid_codes:
                semantic_exit = cli_contract.EXIT_INVALID_STATE
            else:
                semantic_exit = cli_contract.EXIT_BLOCKED
            envelope = cli_contract.error_envelope(
                command,
                code=exc.code.value,
                message=str(exc),
                retryable=exc.retryable,
                suggested_action=(
                    'retry_later'
                    if exc.retryable
                    else 'inspect_durable_sync_run'
                ),
                details={
                    **dict(exc.safe_details),
                    'semantic_exit_code': semantic_exit,
                },
            )
            emitted = _write_machine_envelope(
                envelope, args, command_completed=False
            )
            if _is_output_file_failure(emitted):
                return _output_file_failure_exit_code(args)
            if args.strict_exit_codes:
                return cli_contract.strict_exit_code(emitted)
            return 1
        traceback.print_exc(file=sys.stderr)
        message = str(exc) or exc.__class__.__name__
        exception_type = exc.__class__.__name__
        if args.strict_exit_codes:
            classification = cli_contract.classify_error(
                message,
                exception_type=exception_type,
            )
            envelope = cli_contract.error_envelope(
                command,
                code=classification['code'],
                message=message,
                retryable=classification['retryable'],
                suggested_action=classification['suggested_action'],
                details={
                    'exception_type': exception_type,
                    'semantic_exit_code': classification['exit_code'],
                },
            )
        else:
            envelope = cli_contract.error_envelope(
                command,
                code='INTERNAL_ERROR',
                message=message,
                details={'exception_type': exception_type},
            )
        emitted = _write_machine_envelope(envelope, args, command_completed=False)
        if _is_output_file_failure(emitted):
            return _output_file_failure_exit_code(args)
        if args.strict_exit_codes:
            return cli_contract.strict_exit_code(emitted)
        return 1

    emitted = _write_machine_envelope(envelope, args, command_completed=True)
    if _is_output_file_failure(emitted):
        return _output_file_failure_exit_code(args)
    if args.strict_exit_codes:
        return cli_contract.strict_exit_code(emitted)
    return 0


def _machine_option_tokens(argv):
    """Return raw CLI tokens before the ``--`` positional-argument sentinel."""

    tokens = []
    for value in argv:
        token = str(value)
        if token == '--':
            break
        tokens.append(token)
    return tokens


def _machine_output_requested(argv):
    """Return whether raw CLI arguments explicitly request JSON output."""

    tokens = _machine_option_tokens(argv)
    if '--json' in tokens and any(token in DURABLE_SYNC_COMMANDS for token in tokens):
        return True
    for index, token in enumerate(tokens):
        if token == '--output=json':
            return True
        if (
            token == '--output'
            and index + 1 < len(tokens)
            and tokens[index + 1] == 'json'
        ):
            return True
    return False


def _parser_command_choices(parser):
    """Return current root parser subcommand names without duplicating them."""

    for action in getattr(parser, '_actions', []):
        if getattr(action, 'dest', '') != 'command':
            continue
        choices = getattr(action, 'choices', None)
        if isinstance(choices, dict):
            return set(choices)
    return set()


def _infer_machine_parse_command(parser, argv):
    """Identify the requested command when argparse failed before a Namespace exists."""

    choices = _parser_command_choices(parser)
    for value in _machine_option_tokens(argv):
        if value in choices:
            return value
    return 'cli'


def _argparse_error_message(diagnostics):
    """Extract argparse's concise error line for the machine envelope."""

    for line in reversed(str(diagnostics or '').splitlines()):
        marker = 'error:'
        marker_index = line.lower().find(marker)
        if marker_index >= 0:
            message = line[marker_index + len(marker):].strip()
            if message:
                return message
    return 'Invalid command-line arguments.'


def _machine_parse_error_args(parser, argv):
    """Build the minimal output options available before argparse succeeds."""

    tokens = set(_machine_option_tokens(argv))
    return argparse.Namespace(
        command=_infer_machine_parse_command(parser, argv),
        output='json',
        strict_exit_codes='--strict-exit-codes' in tokens,
        compact='--compact' in tokens,
        fields=[],
        # Parser failure happens before output-file safety checks. Always use
        # stdout so an untrusted or incomplete argument list cannot overwrite
        # a path while reporting its own syntax error.
        output_file='',
    )


def _write_machine_parse_error(parser, argv, exc, diagnostics):
    """Emit a schema-v1 usage envelope after a machine-mode parse failure."""

    if diagnostics:
        sys.stderr.write(diagnostics)
        sys.stderr.flush()

    args = _machine_parse_error_args(parser, argv)
    exit_code = exc.code if isinstance(exc.code, int) else cli_contract.EXIT_USAGE
    envelope = cli_contract.error_envelope(
        args.command,
        code='ARGUMENT_PARSE_ERROR',
        message=_argparse_error_message(diagnostics),
        suggested_action='fix_command_arguments',
        details={
            'parser': parser.prog,
            'exit_code': exit_code,
            'semantic_exit_code': cli_contract.EXIT_USAGE,
            'workflow_started': False,
            'command_completed': False,
        },
    )
    _write_machine_document(
        envelope,
        args,
        record_output_artifact=False,
        command_completed=False,
        workflow_started=False,
    )
    return cli_contract.EXIT_USAGE


def main(argv=None):
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    machine_output_requested = _machine_output_requested(raw_argv)
    parser_diagnostics = io.StringIO()
    try:
        if machine_output_requested:
            with contextlib.redirect_stderr(parser_diagnostics):
                args = parser.parse_args(raw_argv)
        else:
            args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        if not machine_output_requested:
            raise
        if exc.code in (None, 0):
            diagnostics = parser_diagnostics.getvalue()
            if diagnostics:
                sys.stderr.write(diagnostics)
                sys.stderr.flush()
            raise
        return _write_machine_parse_error(
            parser,
            raw_argv,
            exc,
            parser_diagnostics.getvalue(),
        )
    output = getattr(args, 'output', 'text')
    json_only_options = []
    if getattr(args, 'strict_exit_codes', False):
        json_only_options.append('--strict-exit-codes')
    if getattr(args, 'compact', False):
        json_only_options.append('--compact')
    if getattr(args, 'fields', None):
        json_only_options.append('--fields')
    if getattr(args, 'output_file', ''):
        json_only_options.append('--output-file')
    if (
        json_only_options
        and output != 'json'
        and args.command not in {'capabilities', 'schema'}
    ):
        parser.error(f"{', '.join(json_only_options)} requires --output json")
    if getattr(args, 'fields', None):
        try:
            _machine_field_paths(args)
        except ValueError as exc:
            if output == 'json' or args.command in {'capabilities', 'schema'}:
                return _write_machine_usage_error(
                    args,
                    code='INVALID_FIELD_PATH',
                    message=str(exc),
                    suggested_action='fix_field_path',
                )
            parser.error(str(exc))
    if output == 'json' and args.command not in MACHINE_OUTPUT_COMMANDS:
        cli_contract.write_json_envelope(
            cli_contract.error_envelope(
                str(args.command or ''),
                code='OUTPUT_NOT_SUPPORTED',
                message='JSON output is not supported for this command.',
            ),
            sys.stdout,
        )
        return 2
    if getattr(args, 'output_file', ''):
        try:
            _preflight_output_file(args)
        except cli_contract.MachineContractError as exc:
            # Never write the error envelope to the conflicting --output-file path.
            safe_args = copy.copy(args)
            safe_args.output_file = ''
            if output == 'json' or args.command in {'capabilities', 'schema'}:
                envelope = cli_contract.error_envelope(
                    str(args.command or ''),
                    code=exc.code_name,
                    message=str(exc),
                    retryable=exc.retryable,
                    suggested_action=exc.suggested_action,
                    details={
                        **dict(exc.details),
                        'command_completed': False,
                        'workflow_started': False,
                        'semantic_exit_code': exc.semantic_exit_code,
                    },
                )
                _write_machine_document(
                    envelope,
                    safe_args,
                    record_output_artifact=False,
                    command_completed=False,
                    workflow_started=False,
                )
                if getattr(args, 'strict_exit_codes', False):
                    return exc.semantic_exit_code
                return (
                    cli_contract.EXIT_USAGE
                    if exc.semantic_exit_code == cli_contract.EXIT_USAGE
                    else 1
                )
            raise SystemExit(str(exc)) from exc
        except (OSError, ValueError) as exc:
            _write_output_file_failure(
                args,
                exc,
                command_completed=False,
                workflow_started=False,
            )
            return _output_file_failure_exit_code(args)
    if output == 'json':
        return run_machine_command(parser, args)
    try:
        result = dispatch_command(parser, args)
    except Exception as exc:
        from sync_run_contracts import SyncRunError

        if isinstance(exc, SyncRunError):
            raise SystemExit(f'{exc.code.value}: {exc}') from exc
        raise
    if args.command in {'capabilities', 'schema'} and _is_output_file_failure(result):
        return _output_file_failure_exit_code(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
