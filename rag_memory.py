# -*- coding: utf-8 -*-
import hashlib
import json
import math
import os
from datetime import datetime


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def hash_text(text):
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()


def truncate_text(text, limit=220):
    if not isinstance(text, str):
        return ''
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[:limit - 3] + '...'


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_num = float(left_value)
        right_num = float(right_value)
        dot += left_num * right_num
        left_norm += left_num * left_num
        right_norm += right_num * right_num
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class JsonRagStore(object):
    def __init__(self, store_dir):
        self.store_dir = os.path.abspath(store_dir)
        self.metadata_path = os.path.join(self.store_dir, 'metadata.json')
        self.history_path = os.path.join(self.store_dir, 'history.jsonl')
        self._loaded = False
        self.metadata = {}
        self.history = {}

    def load(self):
        if self._loaded:
            return
        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata = self._load_json_file(self.metadata_path)
        self.history = {}
        if os.path.isfile(self.history_path):
            with open(self.history_path, 'r', encoding='utf-8-sig') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    memory_id = record.get('memory_id')
                    if memory_id:
                        self.history[memory_id] = record
        self._loaded = True

    def _load_json_file(self, path):
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8-sig') as handle:
                data = json.load(handle)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def count_history(self):
        self.load()
        return len(self.history)

    def get_history_record(self, memory_id):
        self.load()
        return self.history.get(memory_id)

    def set_metadata(self, **updates):
        self.load()
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        for key, value in updates.items():
            self.metadata[key] = value
        self.metadata['updated_at'] = now_iso()
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = self.metadata['updated_at']
        self._write_metadata()

    def _write_metadata(self):
        os.makedirs(self.store_dir, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)

    def _write_history(self):
        os.makedirs(self.store_dir, exist_ok=True)
        with open(self.history_path, 'w', encoding='utf-8') as handle:
            for memory_id in sorted(self.history):
                handle.write(json.dumps(self.history[memory_id], ensure_ascii=False) + '\n')

    def upsert_history(self, records):
        self.load()
        changed = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            memory_id = record.get('memory_id')
            embedding = record.get('embedding')
            if not memory_id or not isinstance(embedding, list) or not embedding:
                continue
            existing = self.history.get(memory_id)
            if existing == record:
                continue
            self.history[memory_id] = record
            changed += 1
        if changed:
            self._write_history()
            self.set_metadata(history_count=len(self.history))
        return changed

    def search_history(self, query_vector, top_k=4, min_similarity=0.72):
        self.load()
        results = []
        for record in self.history.values():
            vector = record.get('embedding')
            if not isinstance(vector, list) or not vector:
                continue
            score = cosine_similarity(query_vector, vector)
            if score < min_similarity:
                continue
            result = dict(record)
            result['score'] = score
            results.append(result)
        results.sort(key=self._sort_key, reverse=True)
        if top_k > 0:
            return results[:top_k]
        return results

    def _sort_key(self, record):
        quality_state = record.get('quality_state') or ''
        quality_rank = {
            'manual_polished': 3,
            'seed': 2,
            'batch_applied': 1,
            'sync_applied': 1,
        }.get(quality_state, 0)
        return (float(record.get('score') or 0.0), quality_rank, record.get('created_at') or '')
