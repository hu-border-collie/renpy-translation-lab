# -*- coding: utf-8 -*-
from contextlib import contextmanager
import hashlib
import json
import math
import os
import socket
from datetime import datetime


LOCK_STALE_AFTER_SECONDS = 60 * 60


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


class JsonRagStoreLockError(RuntimeError):
    pass


class JsonRagStore(object):
    def __init__(self, store_dir):
        self.store_dir = os.path.abspath(store_dir)
        self.metadata_path = os.path.join(self.store_dir, 'metadata.json')
        self.history_path = os.path.join(self.store_dir, 'history.jsonl')
        self.lock_path = os.path.join(self.store_dir, '.rag_store.lock')
        self._loaded = False
        self._disk_snapshot = None
        self.metadata = {}
        self.history = {}
        self.file_index = {}

    def _warn(self, message):
        print(f'Warning: {message}')

    def load(self):
        if self._loaded:
            return
        self._load_from_disk()

    def _load_from_disk(self):
        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata = self._load_json_file(self.metadata_path)
        self.history = {}
        self.file_index = {}
        if os.path.isfile(self.history_path):
            with open(self.history_path, 'r', encoding='utf-8-sig') as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._warn(f'Skipping invalid RAG history row {self.history_path}:{line_number}: {exc}')
                        continue
                    if not isinstance(record, dict):
                        self._warn(f'Skipping non-object RAG history row {self.history_path}:{line_number}')
                        continue
                    memory_id = record.get('memory_id')
                    if memory_id:
                        self.history[memory_id] = record
                        self._index_record(memory_id, record)
        self._loaded = True
        self._disk_snapshot = self._disk_version()

    def _file_version(self, path):
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _disk_version(self):
        return (
            self._file_version(self.metadata_path),
            self._file_version(self.history_path),
        )

    def _refresh_from_disk_if_changed(self):
        if not self._loaded or self._disk_snapshot != self._disk_version():
            self._load_from_disk()

    def _load_json_file(self, path):
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8-sig') as handle:
                data = json.load(handle)
        except Exception as exc:
            self._warn(f'Failed to load RAG metadata {path}: {exc}')
            return {}
        if not isinstance(data, dict):
            self._warn(f'Ignoring non-object RAG metadata {path}')
            return {}
        return data

    def count_history(self):
        self.load()
        return len(self.history)

    def get_history_record(self, memory_id):
        self.load()
        return self.history.get(memory_id)

    def set_metadata(self, **updates):
        with self._locked('set_metadata') as lock_info:
            self._refresh_from_disk_if_changed()
            self._update_metadata_unlocked('set_metadata', updates, lock_info)

    def _write_metadata(self):
        def write(handle):
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)

        self._atomic_write(self.metadata_path, write)

    def _write_history(self):
        def write(handle):
            for memory_id in sorted(self.history):
                handle.write(json.dumps(self.history[memory_id], ensure_ascii=False) + '\n')

        self._atomic_write(self.history_path, write)

    def _atomic_write(self, path, writer):
        os.makedirs(self.store_dir, exist_ok=True)
        tmp_path = f'{path}.tmp.{os.getpid()}.{id(self)}'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            self._disk_snapshot = self._disk_version()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    self._warn(f'Failed to remove temporary RAG store file {tmp_path}: {exc}')

    def _lock_owner(self, operation):
        return {
            'operation': operation,
            'owner': socket.gethostname(),
            'pid': os.getpid(),
            'created_at': now_iso(),
        }

    def _format_lock_owner(self, data):
        if not isinstance(data, dict):
            return 'unknown owner'
        parts = []
        operation = data.get('operation')
        owner = data.get('owner')
        pid = data.get('pid')
        created_at = data.get('created_at')
        if operation:
            parts.append(f'operation={operation}')
        if owner:
            parts.append(f'owner={owner}')
        if pid:
            parts.append(f'pid={pid}')
        if created_at:
            parts.append(f'created_at={created_at}')
        return ', '.join(parts) if parts else 'unknown owner'

    def _read_lock_owner(self):
        try:
            with open(self.lock_path, 'r', encoding='utf-8-sig') as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _lock_age_seconds(self, data):
        if not isinstance(data, dict):
            return None
        created_at = data.get('created_at')
        if not created_at:
            return None
        try:
            created_at = datetime.fromisoformat(str(created_at))
        except (TypeError, ValueError):
            return None
        return (datetime.now() - created_at).total_seconds()

    def _is_lock_owner_alive(self, pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return None
        if os.name == 'nt':
            return self._is_windows_process_alive(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    def _is_windows_process_alive(self, pid):
        import ctypes

        process_query_limited_information = 0x1000
        error_invalid_parameter = 87
        still_active = 259
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            if ctypes.get_last_error() == error_invalid_parameter:
                return False
            return True
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    def _is_stale_lock(self, data):
        if not isinstance(data, dict):
            return False
        local_owner = data.get('owner') == socket.gethostname()
        if local_owner and data.get('pid') is not None:
            is_alive = self._is_lock_owner_alive(data.get('pid'))
            if is_alive is False:
                return True
            if is_alive is True:
                return False
        age_seconds = self._lock_age_seconds(data)
        return age_seconds is not None and age_seconds >= LOCK_STALE_AFTER_SECONDS

    def _recover_stale_lock(self, existing):
        if not self._is_stale_lock(existing):
            return False
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            return True
        except OSError as exc:
            self._warn(f'Failed to remove stale RAG store lock {self.lock_path}: {exc}')
            return False
        self._warn(f'Recovered stale RAG store lock {self.lock_path} ({self._format_lock_owner(existing)})')
        return True

    def _open_lock_file(self):
        return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

    @contextmanager
    def _locked(self, operation):
        os.makedirs(self.store_dir, exist_ok=True)
        lock_info = self._lock_owner(operation)
        try:
            fd = self._open_lock_file()
        except FileExistsError:
            existing = self._read_lock_owner()
            if not self._recover_stale_lock(existing):
                raise JsonRagStoreLockError(
                    f'RAG store is locked at {self.lock_path} ({self._format_lock_owner(existing)}).'
                )
            try:
                fd = self._open_lock_file()
            except FileExistsError:
                existing = self._read_lock_owner()
                raise JsonRagStoreLockError(
                    f'RAG store is locked at {self.lock_path} ({self._format_lock_owner(existing)}).'
                )

        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(lock_info, handle, ensure_ascii=False)
            yield lock_info
        finally:
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._warn(f'Failed to remove RAG store lock {self.lock_path}: {exc}')

    def _update_metadata_unlocked(self, operation, updates, lock_info):
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        for key, value in updates.items():
            self.metadata[key] = value
        updated_at = now_iso()
        self.metadata['updated_at'] = updated_at
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = updated_at
        self.metadata['last_write'] = {
            'operation': operation,
            'owner': lock_info.get('owner'),
            'pid': lock_info.get('pid'),
            'lock_created_at': lock_info.get('created_at'),
            'updated_at': updated_at,
        }
        self._write_metadata()

    def _index_record(self, memory_id, record):
        file_rel_path = record.get('file_rel_path')
        if not file_rel_path:
            return
        self.file_index.setdefault(file_rel_path, set()).add(memory_id)

    def _unindex_record(self, memory_id, record):
        file_rel_path = record.get('file_rel_path')
        if not file_rel_path:
            return
        bucket = self.file_index.get(file_rel_path)
        if not bucket:
            return
        bucket.discard(memory_id)
        if not bucket:
            self.file_index.pop(file_rel_path, None)

    def upsert_history(self, records):
        with self._locked('upsert_history') as lock_info:
            self._refresh_from_disk_if_changed()
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
                if existing:
                    self._unindex_record(memory_id, existing)
                self.history[memory_id] = record
                self._index_record(memory_id, record)
                changed += 1
            if changed:
                self._write_history()
                self._update_metadata_unlocked(
                    'upsert_history',
                    {'history_count': len(self.history)},
                    lock_info,
                )
        return changed

    def delete_history(self, memory_ids):
        with self._locked('delete_history') as lock_info:
            self._refresh_from_disk_if_changed()
            changed = 0
            for memory_id in memory_ids:
                existing = self.history.pop(memory_id, None)
                if not existing:
                    continue
                self._unindex_record(memory_id, existing)
                changed += 1
            if changed:
                self._write_history()
                self._update_metadata_unlocked(
                    'delete_history',
                    {'history_count': len(self.history)},
                    lock_info,
                )
        return changed

    def history_ids_for_file(self, file_rel_path, quality_state=None):
        self.load()
        memory_ids = list(self.file_index.get(file_rel_path, set()))
        if quality_state is None:
            return memory_ids
        return [
            memory_id for memory_id in memory_ids
            if self.history.get(memory_id, {}).get('quality_state') == quality_state
        ]

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
