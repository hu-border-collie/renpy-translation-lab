# -*- coding: utf-8 -*-
import argparse
import ast
import hashlib
import io
import json
import os
import re
import sys
import time
import tokenize
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rag_memory import JsonRagStore, hash_text, truncate_text
import prompt_context
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
    sys.stdout = DualLogger(CONSOLE_LOG)

BATCH_MODEL = 'gemini-3.1-flash-lite'
BATCH_TARGET_SIZE = 4
BATCH_CONTEXT_BEFORE = 8
BATCH_CONTEXT_AFTER = 4
BATCH_MAX_OUTPUT_TOKENS = 4096
BATCH_TEMPERATURE = 0.2
BATCH_THINKING_LEVEL = 'minimal'
BATCH_DISPLAY_NAME_PREFIX = 'renpy-translate'
BATCH_MACRO_SETTING = ''
MANIFEST_MODE_TRANSLATION = 'translation'
MANIFEST_MODE_KEYWORD_EXTRACTION = 'keyword_extraction'
MANIFEST_MODE_REVISION = 'revision'
KEYWORD_DISPLAY_NAME_PREFIX = 'renpy-keywords'
KEYWORD_CHUNK_SIZE = 40
KEYWORD_MAX_CANDIDATES_PER_CHUNK = 12
REVISION_DISPLAY_NAME_PREFIX = 'renpy-revise'
REVISION_CHUNK_SIZE = 6

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


def load_batch_settings():
    global BATCH_MODEL, BATCH_TARGET_SIZE, BATCH_CONTEXT_BEFORE, BATCH_CONTEXT_AFTER
    global BATCH_MAX_OUTPUT_TOKENS, BATCH_TEMPERATURE, BATCH_THINKING_LEVEL
    global BATCH_DISPLAY_NAME_PREFIX, BATCH_MACRO_SETTING
    global KEYWORD_DISPLAY_NAME_PREFIX, KEYWORD_CHUNK_SIZE, KEYWORD_MAX_CANDIDATES_PER_CHUNK
    global REVISION_DISPLAY_NAME_PREFIX, REVISION_CHUNK_SIZE
    global RAG_ENABLED, RAG_STORE_DIR, RAG_EMBEDDING_MODEL, RAG_QUERY_TASK_TYPE
    global RAG_DOCUMENT_TASK_TYPE, RAG_OUTPUT_DIMENSIONALITY, RAG_TOP_K_HISTORY
    global RAG_TOP_K_TERMS, RAG_MIN_SIMILARITY, RAG_SEGMENT_LINES
    global RAG_BOOTSTRAP_ON_BUILD, RAG_HISTORY_CHAR_LIMIT, _RAG_STORE
    global STORY_MEMORY_ENABLED, STORY_MEMORY_GRAPH_FILE, STORY_MEMORY_MAX_CONTEXT_CHARS
    global STORY_MEMORY_TOP_K_RELATIONS, STORY_MEMORY_TOP_K_TERMS
    global STORY_MEMORY_INCLUDE_SCENE_SUMMARY, _STORY_GRAPH, _STORY_GRAPH_PATH

    config = load_json_file(legacy.CONFIG_FILE)
    translator_config = load_json_file(legacy.TRANSLATOR_CONFIG)

    batch_model = config.get('batch_model')
    if isinstance(batch_model, str) and batch_model.strip():
        BATCH_MODEL = batch_model.strip()

    BATCH_TARGET_SIZE = coerce_positive_int(
        config.get('batch_target_size', config.get('batch_size')),
        BATCH_TARGET_SIZE,
    )
    BATCH_CONTEXT_BEFORE = coerce_positive_int(config.get('batch_context_before'), BATCH_CONTEXT_BEFORE)
    BATCH_CONTEXT_AFTER = coerce_positive_int(config.get('batch_context_after'), BATCH_CONTEXT_AFTER)
    BATCH_MAX_OUTPUT_TOKENS = coerce_positive_int(
        config.get('batch_max_output_tokens'),
        BATCH_MAX_OUTPUT_TOKENS,
    )
    BATCH_THINKING_LEVEL = coerce_non_empty_string(
        config.get('batch_thinking_level'),
        BATCH_THINKING_LEVEL,
    )

    display_name_prefix = config.get('batch_display_name_prefix')
    if isinstance(display_name_prefix, str) and display_name_prefix.strip():
        BATCH_DISPLAY_NAME_PREFIX = display_name_prefix.strip()

    macro_setting = config.get('batch_macro_setting')
    if isinstance(macro_setting, str) and macro_setting.strip():
        BATCH_MACRO_SETTING = macro_setting.strip()

    batch = translator_config.get('batch')
    if not isinstance(batch, dict):
        batch = {}

    model_name = batch.get('model')
    if isinstance(model_name, str) and model_name.strip():
        BATCH_MODEL = model_name.strip()

    display_name_prefix = batch.get('display_name_prefix')
    if isinstance(display_name_prefix, str) and display_name_prefix.strip():
        BATCH_DISPLAY_NAME_PREFIX = display_name_prefix.strip()

    BATCH_TARGET_SIZE = coerce_positive_int(batch.get('chunk_size'), BATCH_TARGET_SIZE)
    BATCH_CONTEXT_BEFORE = coerce_positive_int(batch.get('context_before'), BATCH_CONTEXT_BEFORE)
    BATCH_CONTEXT_AFTER = coerce_positive_int(batch.get('context_after'), BATCH_CONTEXT_AFTER)
    BATCH_MAX_OUTPUT_TOKENS = coerce_positive_int(
        batch.get('max_output_tokens'),
        BATCH_MAX_OUTPUT_TOKENS,
    )
    BATCH_TEMPERATURE = coerce_float(batch.get('temperature'), BATCH_TEMPERATURE)
    BATCH_THINKING_LEVEL = coerce_non_empty_string(
        batch.get('thinking_level'),
        BATCH_THINKING_LEVEL,
    )

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
    return '503' in text or 'UNAVAILABLE' in text


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
    return os.path.join(LOG_DIR, 'rag_store', guess_project_slug())


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
        terms.update(extract_word_tokens(hit.get('source', '')))
        terms.update(extract_word_tokens(hit.get('target', '')))
    for hit in chunk.get('history_hits') or []:
        source_tokens = set(extract_word_tokens(hit.get('source_text', '')))
        translated_tokens = set(extract_word_tokens(hit.get('translated_text', '')))
        terms.update(source_tokens & translated_tokens)
    return terms


def allow_non_chinese_batch_translation(manifest, chunk, original, translated):
    if not manifest.get('rag_enabled'):
        return False
    return legacy.allow_non_chinese_term_translation(
        original,
        translated,
        known_terms=collect_chunk_known_terms(chunk),
    )


def compact_text(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def build_rag_query_text(target_items, context_past):
    parts = []
    local_past = [compact_text(text) for text in context_past[-2:] if compact_text(text)]
    target_lines = [compact_text(item.get('text', '')) for item in target_items if compact_text(item.get('text', ''))]
    if local_past:
        parts.append('Context before:\n' + '\n'.join(f'- {text}' for text in local_past))
    if target_lines:
        parts.append('Target:\n' + '\n'.join(f'- {text}' for text in target_lines))
    return '\n\n'.join(parts)


def embed_texts(contents, task_type):
    if not contents:
        return []
    client = create_batch_client()
    response = client.models.embed_content(
        model=RAG_EMBEDDING_MODEL,
        contents=contents,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=RAG_OUTPUT_DIMENSIONALITY,
        ),
    )
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


def _normalized_abs_path(path):
    return os.path.normcase(os.path.abspath(path))


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
        candidate = os.path.abspath(raw)
    else:
        candidate = os.path.abspath(os.path.join(base_dir, normalize_safe_rel_path(raw, field_name)))
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
            current['id'] = f"{rel_path}:{current['line']}:{current['start']}"
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
):
    return translation_core.build_translation_user_prompt(
        translation_core.ContextWindow(context_past, context_future),
        translation_core.units_from_items(target_items, translation_core.MODE_TRANSLATION),
        translation_core.build_context_bundle(
            glossary_hits=glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
        ),
        history_char_limit=RAG_HISTORY_CHAR_LIMIT,
        story_char_limit=STORY_MEMORY_MAX_CONTEXT_CHARS,
        include_translation_memory=True,
        include_source_text=True,
    )



def build_response_json_schema(target_items):
    return translation_core.build_response_json_schema(
        translation_core.units_from_items(target_items, translation_core.MODE_TRANSLATION),
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
    return {
        'key': chunk['key'],
        'request': {
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
                            )
                        }
                    ],
                }
            ],
            'generation_config': build_generation_config(chunk['items']),
        },
    }



def build_chunks(file_jobs):
    chunks = []
    for job in file_jobs:
        tasks = job['tasks']
        total = len(tasks)
        for start in range(0, total, BATCH_TARGET_SIZE):
            end = min(start + BATCH_TARGET_SIZE, total)
            target_items = tasks[start:end]
            target_units = translation_core.units_from_items(
                target_items,
                translation_core.MODE_TRANSLATION,
                file_rel_path=job['file_rel_path'],
                file_path=job['file_path'],
            )
            context_past = [item['text'] for item in tasks[max(0, start - BATCH_CONTEXT_BEFORE):start]]
            context_future = [item['text'] for item in tasks[end:min(total, end + BATCH_CONTEXT_AFTER)]]
            glossary_hits = retrieve_glossary_hits(target_items) if RAG_ENABLED else []
            history_hits, rag_stats = retrieve_history_hits(target_items, context_past) if RAG_ENABLED else ([], {})
            story_hits = retrieve_batch_story_hits(
                job['file_rel_path'],
                target_items,
                context_past,
                context_future,
            ) if STORY_MEMORY_ENABLED else None
            chunk_number = start // BATCH_TARGET_SIZE + 1
            chunk_key = f"{hash_key(job['file_rel_path'])}-{chunk_number:05d}"
            chunk = {
                'key': chunk_key,
                'mode': MANIFEST_MODE_TRANSLATION,
                'file_rel_path': job['file_rel_path'],
                'file_path': job['file_path'],
                'chunk_index': chunk_number,
                'line_numbers': [unit.line for unit in target_units],
                'context_past': context_past,
                'context_future': context_future,
                'glossary_hits': glossary_hits,
                'history_hits': history_hits,
                'rag_stats': rag_stats,
                'items': [
                    translation_core.legacy_item_from_unit(unit, translation_core.MODE_TRANSLATION)
                    for unit in target_units
                ],
            }
            if STORY_MEMORY_ENABLED and story_memory.has_story_hits(story_hits):
                chunk['story_hits'] = story_hits
            chunks.append(chunk)
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
    if BATCH_TARGET_SIZE > 4:
        warnings_list.append(f'chunk_size={BATCH_TARGET_SIZE} is aggressive for Gemini 3 Flash structured output.')
    if BATCH_CONTEXT_BEFORE > 12 or BATCH_CONTEXT_AFTER > 6:
        warnings_list.append(
            f'context_before/context_after ({BATCH_CONTEXT_BEFORE}/{BATCH_CONTEXT_AFTER}) may inflate prompt tokens.'
        )
    if BATCH_MAX_OUTPUT_TOKENS < 2048:
        warnings_list.append(f'max_output_tokens={BATCH_MAX_OUTPUT_TOKENS} is likely too low for JSON batch output.')
    if BATCH_MODEL.startswith('gemini-3') and (BATCH_THINKING_LEVEL or '').lower() != 'minimal':
        warnings_list.append(
            f'thinking_level={BATCH_THINKING_LEVEL or "(default)"} may waste output budget on reasoning tokens.'
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
        'version': 1,
        'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
        'mode': MANIFEST_MODE_TRANSLATION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
        'input_jsonl_path': input_jsonl_path,
        'result_jsonl_path': '',
        'job_name': '',
        'job_state': 'LOCAL_ONLY',
        'uploaded_file_name': '',
        'result_file_name': '',
        'settings': {
            'target_size': BATCH_TARGET_SIZE,
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
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    remember_latest_manifest(manifest_path)

    print(f'Created batch package: {package_dir}')
    print(f"Pending files: {manifest['summary']['file_count']}")
    print(f"Chunks: {manifest['summary']['chunk_count']}")
    print(f"Items: {manifest['summary']['item_count']}")
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
            entries = collect_translation_entries_from_lines(handle.readlines())

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
                'id': f"{rel_path}:{line_index}:{start}:revision:{entry.get('entry_index', len(items))}",
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
        translation_core.units_from_items(target_items, translation_core.MODE_REVISION),
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
        'version': 1,
        'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
        'mode': MANIFEST_MODE_REVISION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
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
    return translation_core.build_keyword_user_prompt(
        translation_core.units_from_items(
            target_items,
            translation_core.MODE_KEYWORD_EXTRACTION,
        )
    )


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
        'version': 1,
        'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
        'mode': MANIFEST_MODE_KEYWORD_EXTRACTION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
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

        created_manifests.append(part_manifest_path)
        remember_latest_manifest(part_manifest_path)

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

def submit_manifest(target=None, display_name_override='', model_override=''):
    manifest = load_manifest(target) if target else None
    if manifest is None:
        manifest_path = create_batch_package(display_name_override=display_name_override)
        if not manifest_path:
            return None
        manifest = load_manifest(manifest_path)

    if manifest.get('job_name'):
        raise SystemExit(f"Manifest already submitted: {manifest['job_name']}")

    if display_name_override:
        manifest['display_name'] = display_name_override.strip()
    if model_override:
        manifest['batch_model'] = model_override.strip()

    attempts = max(1, len(getattr(legacy, 'API_KEYS', [])))
    last_error = None

    for attempt in range(1, attempts + 1):
        client = create_batch_client()
        uploaded_file = None
        try:
            print(f"Uploading JSONL: {manifest['input_jsonl_path']}")
            uploaded_file = client.files.upload(
                file=manifest['input_jsonl_path'],
                config=genai_types.UploadFileConfig(
                    display_name=manifest['display_name'],
                    mime_type='jsonl',
                ),
            )
            manifest['uploaded_file_name'] = getattr(uploaded_file, 'name', '')
            manifest.setdefault('uploaded_file_names', [])
            if manifest['uploaded_file_name'] and manifest['uploaded_file_name'] not in manifest['uploaded_file_names']:
                manifest['uploaded_file_names'].append(manifest['uploaded_file_name'])
            manifest['last_submit_error'] = ''
            save_manifest(manifest)
            print(f'Uploaded file: {uploaded_file.name}')

            print(f"Creating batch job with model: {manifest['batch_model']}")
            batch_job = client.batches.create(
                model=manifest['batch_model'],
                src=uploaded_file.name,
                config={'display_name': manifest['display_name']},
            )

            manifest['job_name'] = getattr(batch_job, 'name', '')
            manifest['job_state'] = get_state_name(getattr(batch_job, 'state', None))
            manifest['submitted_at'] = datetime.now().isoformat(timespec='seconds')
            manifest['last_status_checked_at'] = manifest['submitted_at']
            manifest['submitted_api_key_index'] = getattr(legacy, 'CURRENT_KEY_INDEX', 0)
            manifest['submitted_api_key_number'] = manifest['submitted_api_key_index'] + 1
            manifest['last_status_api_key_index'] = manifest['submitted_api_key_index']
            manifest['last_submit_error'] = ''
            save_manifest(manifest)

            print(f"Batch job created: {manifest['job_name']}")
            print(f"Manifest: {manifest['_manifest_path']}")
            return manifest['_manifest_path']
        except Exception as exc:
            last_error = exc
            manifest['last_submit_error'] = str(exc)
            manifest['job_state'] = 'SUBMIT_FAILED'
            if uploaded_file is not None:
                manifest['uploaded_file_name'] = getattr(uploaded_file, 'name', '')
                manifest.setdefault('uploaded_file_names', [])
                if manifest['uploaded_file_name'] and manifest['uploaded_file_name'] not in manifest['uploaded_file_names']:
                    manifest['uploaded_file_names'].append(manifest['uploaded_file_name'])
            save_manifest(manifest)

            if is_quota_error(exc) and attempt < attempts and legacy.rotate_api_key():
                print(f'Quota hit during batch submit. Retrying with next API key ({attempt}/{attempts})...')
                continue
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


def validate_keyword_export_paths(manifest, jsonl_path, markdown_path):
    normalized_jsonl = _normalized_abs_path(jsonl_path)
    normalized_markdown = _normalized_abs_path(markdown_path)
    if normalized_jsonl == normalized_markdown:
        raise SystemExit('Keyword export JSONL and Markdown outputs must be different files.')

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
            raise SystemExit(f'Keyword export output would overwrite reserved package file: {output_path}')


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


def export_keyword_candidates(target=None, output_jsonl='', output_markdown=''):
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
        'parse_errors': 0,
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
                candidates = normalize_keyword_candidates(parse_json_payload(response_text))
            except Exception:
                summary['parse_errors'] += 1
                bump_counter(summary['reason_counts'], 'failed_to_parse_keyword_json')
                continue

            summary['parsed_chunks'] += 1
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
    validate_keyword_export_paths(manifest, jsonl_path, markdown_path)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf-8') as handle:
        for candidate in candidates:
            serializable = dict(candidate)
            serializable.pop('evidence_items', None)
            handle.write(json.dumps(serializable, ensure_ascii=False) + '\n')
    write_keyword_markdown(markdown_path, candidates, summary)

    manifest['keyword_exported_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['keyword_export'] = {
        'jsonl_path': jsonl_path,
        'markdown_path': markdown_path,
        'summary': summary,
    }
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print(f'Keyword candidates: {summary["candidate_count_deduped"]} deduped from {summary["candidate_count_raw"]} raw')
    print(f'JSONL: {jsonl_path}')
    print(f'Markdown: {markdown_path}')
    if summary.get('reason_counts'):
        print('Warnings:')
        for name in sorted(summary['reason_counts']):
            print(f"- {name}: {summary['reason_counts'][name]}")
    return manifest['keyword_export']


def append_failure_entries(entries, package_dir=''):
    if not entries:
        return

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
            start, end, _translated, _prefix, _quote, source_text, item_id, chunk_key = unpack_replacement_for_validation(repl)
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
        if file_replacements:
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


def collect_revision_actions(manifest, validate_sources=False):
    result_path = resolve_manifest_result_path(manifest)
    if not os.path.isfile(result_path):
        raise SystemExit('Result JSONL not found. Run download first.')

    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks', [])}
    replacements_by_file = {}
    revised_lines_by_file = {}
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
            item_map = {item['id']: item for item in chunk_items}
            response_payload = row.get('response', {})
            finish_reason = extract_finish_reason(response_payload)
            usage_metadata = summarize_usage_metadata(extract_usage_metadata(response_payload))
            if finish_reason == 'MAX_TOKENS':
                summary['max_tokens_chunks'] += 1

            if row.get('error'):
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'row_error')
                for item in chunk_items:
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
                for item in chunk_items:
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
                for item in chunk_items:
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

            if len(result_items) < len(chunk_items):
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_revision_items'
                bump_counter(summary['reason_counts'], reason_name)

            seen_ids = set()
            for result_item in result_items:
                result_id = result_item['id']
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
        for item in chunk['items']:
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
            item_map = {item['id']: item for item in chunk_items}
            response_payload = row.get('response', {})
            finish_reason = extract_finish_reason(response_payload)
            usage_metadata = summarize_usage_metadata(extract_usage_metadata(response_payload))
            if finish_reason == 'MAX_TOKENS':
                summary['max_tokens_chunks'] += 1

            if row.get('error'):
                summary['chunk_row_errors'] += 1
                bump_counter(summary['reason_counts'], 'row_error')
                for item in chunk_items:
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
                summary['missing_response_chunks'] += 1
                bump_counter(summary['reason_counts'], 'missing_response_text')
                for item in chunk_items:
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
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'failed_to_parse_model_json'
                bump_counter(summary['reason_counts'], reason_name)
                for item in chunk_items:
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

            if len(result_items) < len(chunk_items):
                summary['partial_chunks'] += 1
                reason_name = 'truncated_output' if finish_reason == 'MAX_TOKENS' else 'partial_result_items'
                bump_counter(summary['reason_counts'], reason_name)

            seen_ids = set()
            for result_item in result_items:
                result_id = result_item['id']
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
        for item in chunk['items']:
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
    _replacements, _translated, _failures, summary = collect_result_actions(manifest, validate_sources=True)
    manifest['last_check_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['last_check_summary'] = summary
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')
    print(f"Manifest: {manifest['_manifest_path']}")
    print_check_summary(summary)
    return manifest


def apply_results(target=None, force=False):
    manifest = load_manifest(target)
    require_manifest_mode(manifest, MANIFEST_MODE_TRANSLATION, 'apply')
    if manifest.get('applied_at') and not force:
        raise SystemExit('Manifest was already applied. Re-run apply with --force to bypass this guard; source validation still applies.')

    replacements_by_file, _translated_lines_by_file, failure_entries, summary = collect_result_actions(
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
        if not replacements:
            continue
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
        rag_apply_summary = sync_rag_store_for_jobs(rag_jobs, quality_state='batch_applied')

    manifest['applied_at'] = datetime.now().isoformat(timespec='seconds')
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
    save_manifest(manifest, update_latest=manifest.get('execution') != 'sync')

    print_check_summary(summary)
    print(f'Applied files: {applied_files}')
    print(f'Applied lines: {applied_lines}')
    print(f'Failures logged: {len(failure_entries)}')
    if rag_apply_summary:
        print(f"RAG store updated: {rag_apply_summary.get('upserted', 0)} entries")
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
        if not replacements:
            continue
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


def collect_translation_entries_from_lines(lines):
    entries = []
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip('\n')
        comment_match = REPAIR_LINE_COMMENT_RE.match(raw_line)
        if comment_match:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
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
                    entries.append(
                        entry
                    )
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
                        entries.append(
                            {
                                'line_number': next_index + 1,
                                'source_line_number': index + 1,
                                'source': old_match.group('text'),
                                'translation': token['text'],
                                'start': token['start'],
                                'end': token['end'],
                                'quote': token['quote'],
                            }
                        )
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


def embed_history_records(records):
    embedded_records = []
    batch_size = 16
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
        if has_current_source_embedding(existing, record):
            if existing.get('translation_checksum') == record['translation_checksum']:
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
    if not pending_records:
        stats['history_records_after'] = store.count_history()
        return stats

    try:
        embedded_records = embed_history_records(records_to_embed)
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
    return {
        'key': job['key'],
        'request': {
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
                            )
                        }
                    ],
                }
            ],
            'generation_config': build_generation_config(job['items']),
        },
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
        'version': 1,
        'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
        'mode': mode,
        'execution': 'sync',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'display_name': display_name,
        'batch_model': BATCH_MODEL,
        'base_dir': legacy.BASE_DIR,
        'tl_dir': legacy.TL_DIR,
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


def print_banner():
    print('=' * 60)
    print('Gemini Batch Translator (Ren\'Py)')
    print(f'Base dir: {legacy.BASE_DIR}')
    print(f'TL dir: {legacy.TL_DIR} (exists: {os.path.isdir(legacy.TL_DIR)})')
    print(f'Batch jobs dir: {BATCH_JOBS_DIR}')
    print(f'Translator config: {legacy.TRANSLATOR_CONFIG} (exists: {os.path.isfile(legacy.TRANSLATOR_CONFIG)})')
    print(f'Glossary: {legacy.GLOSSARY_FILE} (exists: {os.path.isfile(legacy.GLOSSARY_FILE)})')
    print(f'Batch model: {BATCH_MODEL}')
    print(
        f'Chunk settings: target={BATCH_TARGET_SIZE}, '
        f'context_before={BATCH_CONTEXT_BEFORE}, context_after={BATCH_CONTEXT_AFTER}'
    )
    print(f'Max output tokens: {BATCH_MAX_OUTPUT_TOKENS}')
    print(f'Thinking level: {BATCH_THINKING_LEVEL or "(default)"}')
    print(f'Prepare enabled: {legacy.PREP_ENABLED}')
    print('=' * 60)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Batch translator for Ren\'Py tl files using Gemini Batch API.'
    )
    subparsers = parser.add_subparsers(dest='command')

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

    submit_parser = subparsers.add_parser('submit', help='Create and submit a batch job.')
    submit_parser.add_argument(
        'target',
        nargs='?',
        default='',
        help='Existing manifest path or package dir. If omitted, build a new package first.',
    )
    submit_parser.add_argument('--display-name', default='', help='Override Batch display name.')
    submit_parser.add_argument('--model', default='', help='Override batch model.')

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

    if command == 'submit':
        submit_manifest(
            target=args.target or None,
            display_name_override=args.display_name,
            model_override=args.model,
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
        )
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
