"""Durable embedding batches, separate from a published retrieval store."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from foundation.atomic_io import atomic_write_jsonl


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class EmbeddingCheckpoint:
    """Append checksummed batches while the caller holds its store write lock.

    The binding pins project inputs and embedding identity. Each entry also pins
    its complete current record. A damaged or unfinished batch discards only the
    suffix after the last verified batch; uncommitted vectors are never trusted.
    This journal is not a retrieval store and cannot publish readiness.
    """

    def __init__(self, path: Path, binding: str, dimension: int):
        self.path = path
        self.dimension = dimension
        self.header = {'schema_version': 1, 'binding': binding, 'dimension': dimension}
        self.entries = {}
        self._load()

    def _valid_entry(self, entry):
        if not isinstance(entry, dict):
            return False
        vector = entry.get('embedding')
        return (isinstance(entry.get('id'), str) and bool(entry['id'])
            and isinstance(entry.get('fingerprint'), str) and bool(entry['fingerprint'])
            and isinstance(vector, list) and len(vector) == self.dimension
            and all(type(value) in (int, float) and math.isfinite(value) for value in vector))

    def _load(self):
        if not self.path.exists():
            atomic_write_jsonl(self.path, [self.header])
            return
        with self.path.open('rb') as handle:
            header_line = handle.readline()
            try:
                header = json.loads(header_line) if header_line.endswith(b'\n') else None
            except (ValueError, UnicodeError):
                header = None
            if header == self.header:
                valid_end = handle.tell()
                for line in handle:
                    try:
                        payload = json.loads(line)
                        entries = payload['entries']
                        valid = (line.endswith(b'\n') and isinstance(entries, list) and bool(entries)
                            and all(self._valid_entry(entry) for entry in entries)
                            and payload['sha256'] == _digest(entries))
                    except (ValueError, UnicodeError, KeyError, TypeError, OverflowError):
                        valid = False
                    if not valid:
                        break
                    self.entries.update((entry['id'], entry) for entry in entries)
                    valid_end = handle.tell()
            else:
                valid_end = None
        if valid_end is None:
            atomic_write_jsonl(self.path, [self.header])
        elif self.path.stat().st_size != valid_end:
            with self.path.open('r+b') as handle:
                handle.truncate(valid_end)
                handle.flush()
                os.fsync(handle.fileno())

    def get(self, record_id: str, fingerprint: str):
        """Return only a vector matching the complete current record."""
        entry = self.entries.get(record_id)
        return entry['embedding'] if entry and entry['fingerprint'] == fingerprint else None

    def save(self, entries: list[dict]) -> None:
        """Flush batches before acknowledging progress or requesting more work."""
        if not entries:
            return
        if not all(self._valid_entry(entry) for entry in entries):
            raise ValueError('Invalid embedding checkpoint batch.')
        with self.path.open('ab') as handle:
            for start in range(0, len(entries), 16):
                batch = entries[start:start + 16]
                payload = {'entries': batch, 'sha256': _digest(batch)}
                handle.write((json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        self.entries.update((entry['id'], entry) for entry in entries)

    def clear(self) -> None:
        """Remove the journal only after the completed store has been published."""
        self.path.unlink(missing_ok=True)
