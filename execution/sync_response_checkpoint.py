"""Local checkpoints for frozen synchronous provider requests."""
from __future__ import annotations

import json
from pathlib import Path

from engine_adapters.coverage import digest_json
from foundation.atomic_io import atomic_write_json


class SyncResponseCheckpointError(ValueError):
    """A checkpoint cannot safely be read, reused or persisted."""


class SyncRequestUnknownError(SyncResponseCheckpointError):
    """An interrupted provider dispatch has no durable response."""


def read_record(path: Path) -> dict:
    try:
        envelope = json.loads(path.read_text(encoding='utf-8'))
        payload = envelope['payload']
        if not isinstance(payload, dict) or envelope['sha256'] != digest_json(payload):
            raise ValueError('checksum mismatch')
        return payload
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise SyncResponseCheckpointError(f'Invalid sync checkpoint: {path.name}.') from exc


def write_record(path: Path, payload: dict) -> None:
    try:
        atomic_write_json(path, {'payload': payload, 'sha256': digest_json(payload)},
                          ensure_ascii=False, indent=2)
    except (OSError, ValueError, TypeError) as exc:
        raise SyncResponseCheckpointError(
            'Could not save sync checkpoint; execution stopped to preserve recovery.'
        ) from exc


def _read_attempt(path: Path) -> dict:
    record = read_record(path)
    key, phase = record.get('key'), record.get('phase')
    if (not isinstance(key, str) or not key
            or not isinstance(phase, str) or not phase
            or record.get('state') not in {'inflight', 'returned'}
            or path.stem != digest_json([key, phase])
            or (record['state'] == 'returned' and not isinstance(record.get('response'), dict))):
        raise SyncResponseCheckpointError('Sync attempt identity is invalid.')
    return record


def unknown_sync_requests(package_dir: str | Path, task: str) -> list[str]:
    """Return in-flight requests for a read-only recovery confirmation."""
    attempts = Path(package_dir) / f'{task}_checkpoint' / 'attempts'
    return [record['key'] for path in sorted(attempts.glob('*.json'))
            if (record := _read_attempt(path))['state'] == 'inflight']


class SyncResponseCheckpoint:
    """Persist dispatch and response under the caller's exclusive package lock.

    Raw responses, including usage and responses that fail validation, are
    independent of formal results. Replaying an attempt uses its original
    request digest and does not make a new provider call.
    """

    def __init__(self, package_dir: str | Path, binding: str, *, task: str,
                 retry_unknown=False, require_binding=False, keep_dispatch_markers=False):
        self.root = Path(package_dir) / f'{task}_checkpoint'
        self.binding = binding
        self.retry_unknown = retry_unknown
        self.reused_attempts = 0
        self.attempts: dict[str, dict] = {}
        self.returned_first_pass: set[str] = set()
        self.dispatch_markers = self.root / 'dispatches' if keep_dispatch_markers else None
        attempts_dir = self.root / 'attempts'
        try:
            attempts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SyncResponseCheckpointError('Could not create sync checkpoint.') from exc
        header = self.root / 'binding.json'
        expected = {'version': 1, 'binding': binding}
        paths = sorted(attempts_dir.glob('*.json'))
        if header.exists():
            if read_record(header) != expected:
                raise SyncResponseCheckpointError('Sync checkpoint inputs changed; start a new run.')
        else:
            if require_binding or paths:
                raise SyncResponseCheckpointError('Sync checkpoint binding is missing.')
            write_record(header, expected)
        for path in paths:
            record = _read_attempt(path)
            if record.get('binding') != binding:
                raise SyncResponseCheckpointError('Sync attempt identity is invalid.')
            self.attempts[path.stem] = record
            if record['phase'] == 'first_pass' and record['state'] == 'returned':
                self.returned_first_pass.add(record['key'])
        if self.dispatch_markers is not None:
            self._validate_dispatch_markers(self.dispatch_markers)

    def _validate_dispatch_markers(self, directory: Path) -> None:
        """Treat a lost response file for a known dispatch as a damaged checkpoint."""
        marked = set()
        for path in directory.glob('*.json'):
            marker = _read_attempt(path)
            attempt = self.attempts.get(path.stem)
            if attempt is None:
                raise SyncResponseCheckpointError('Sync checkpoint is missing a saved attempt.')
            if marker['state'] != 'inflight' or any(
                marker.get(field) != attempt.get(field)
                for field in ('binding', 'key', 'phase', 'request_sha256')
            ):
                raise SyncResponseCheckpointError('Sync dispatch marker does not match its saved attempt.')
            marked.add(path.stem)
        if any(record['state'] == 'returned' and identity not in marked
               for identity, record in self.attempts.items()):
            raise SyncResponseCheckpointError('Sync checkpoint is missing a dispatch marker.')

    def require_known_attempts(self) -> None:
        if any(row['state'] == 'inflight' for row in self.attempts.values()) and not self.retry_unknown:
            raise SyncRequestUnknownError(
                'A sync request was interrupted without a saved response. '
                'Use --resume <MANIFEST> --retry-unknown only after reviewing possible duplicate requests.'
            )

    def call(self, key: str, phase: str, request: dict, dispatch, *, before_dispatch=None) -> dict:
        """Save the dispatch marker and returned data before business validation."""
        identity = digest_json([key, phase])
        request_sha = digest_json(request)
        previous = self.attempts.get(identity)
        if previous is not None:
            if previous.get('request_sha256') != request_sha:
                raise SyncResponseCheckpointError('Frozen sync request changed; start a new run.')
            if previous['state'] == 'returned':
                self.reused_attempts += 1
                return previous['response']
            if not self.retry_unknown:
                raise SyncRequestUnknownError('Sync request outcome is unknown; review before retrying.')
        if before_dispatch is not None:
            before_dispatch()
        record = {'binding': self.binding, 'key': key, 'phase': phase,
                  'request_sha256': request_sha, 'state': 'inflight'}
        path = self.root / 'attempts' / f'{identity}.json'
        write_record(path, record)
        self.attempts[identity] = record
        if self.dispatch_markers is not None:
            marker = self.dispatch_markers / path.name
            if not marker.is_file():
                write_record(marker, record)
        response = dispatch()
        # Save only the fields needed for canonical results and usage.
        response = {name: response[name] for name in (
            'response_payload', 'response_text', 'finish_reason', 'usage_metadata',
            'request_metadata', 'output_diagnostics', 'provider', 'model', 'execution_mode',
        ) if name in response}
        record = {**record, 'state': 'returned', 'response': response}
        write_record(path, record)
        self.attempts[identity] = record
        if phase == 'first_pass':
            self.returned_first_pass.add(key)
        return response
