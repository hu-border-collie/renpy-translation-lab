"""Durable responses for synchronous keyword requests, independent of reports.

An attempt is marked in flight before dispatch and its response is atomically
saved before validation or a targeted retry. Interrupted attempts with no saved
response require explicit acknowledgement; they are never silently resent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine_adapters.coverage import digest_json
from foundation.atomic_io import atomic_write_json


class KeywordCheckpointError(ValueError):
    """A keyword checkpoint cannot safely be read, reused or persisted."""


class KeywordRequestUnknownError(KeywordCheckpointError):
    """An interrupted provider attempt has no durable response."""


@dataclass(frozen=True)
class KeywordResumePlan:
    """Separate durable valid results from this invocation's outstanding work."""

    restored: dict[str, dict]
    pending: list[dict]
    failed: int
    incomplete: int


def checkpoint_binding(manifest: dict, request_rows: list) -> str:
    """Bind cached responses to frozen sources, requests, routes and settings."""
    fields = (
        'mode', 'execution', 'base_dir', 'tl_dir', 'model', 'model_routing',
        'settings', 'keyword_settings', 'mv_preprocessing_identity',
        'keyword_source_files', 'target_language', 'generation_target', 'chunks',
    )
    return digest_json({
        'manifest': {key: manifest.get(key) for key in fields},
        'requests': request_rows,
    })


def _read(path: Path) -> dict:
    try:
        envelope = json.loads(path.read_text(encoding='utf-8'))
        payload = envelope['payload']
        if not isinstance(payload, dict) or envelope['sha256'] != digest_json(payload):
            raise ValueError('checksum mismatch')
        return payload
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise KeywordCheckpointError(f'Invalid keyword checkpoint: {path.name}.') from exc


def _write(path: Path, payload: dict) -> None:
    try:
        atomic_write_json(path, {'payload': payload, 'sha256': digest_json(payload)},
                          ensure_ascii=False, indent=2)
    except (OSError, ValueError, TypeError) as exc:
        raise KeywordCheckpointError(
            'Could not save keyword response; execution stopped to preserve the checkpoint.'
        ) from exc


def unknown_keyword_requests(package_dir: str | Path) -> list[str]:
    """Read unresolved dispatches for CLI/GUI recovery acknowledgement."""
    attempts = Path(package_dir) / 'keyword_checkpoint' / 'attempts'
    return [record['key'] for path in sorted(attempts.glob('*.json'))
            if (record := _read(path)).get('state') == 'inflight']


class KeywordCheckpoint:
    """Use under the package's exclusive lock for the entire execution."""

    def __init__(self, package_dir: str | Path, binding: str, *, retry_unknown=False):
        self.root = Path(package_dir) / 'keyword_checkpoint'
        self.binding = binding
        self.retry_unknown = retry_unknown
        self.reused_attempts = 0
        self.attempts: dict[str, dict] = {}
        self.saved_keys: set[str] = set()
        self.completed_results: dict[str, dict] = {}
        self.generations: dict[str, int] = {}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / 'attempts').mkdir(exist_ok=True)
            (self.root / 'completed').mkdir(exist_ok=True)
        except OSError as exc:
            raise KeywordCheckpointError('Could not create keyword checkpoint.') from exc
        header = self.root / 'binding.json'
        expected = {'version': 1, 'binding': binding}
        if header.exists():
            if _read(header) != expected:
                raise KeywordCheckpointError('Keyword checkpoint inputs changed; start a new run.')
        else:
            if list((self.root / 'attempts').glob('*.json')):
                raise KeywordCheckpointError('Keyword checkpoint binding is missing.')
            _write(header, expected)
        for path in sorted((self.root / 'attempts').glob('*.json')):
            record = _read(path)
            if (record.get('binding') != binding
                    or record.get('state') not in {'inflight', 'returned'}
                    or path.stem != digest_json([record.get('key'), record.get('phase')])
                    or (record['state'] == 'returned' and not isinstance(record.get('response'), dict))):
                raise KeywordCheckpointError('Keyword attempt identity is invalid.')
            self.attempts[path.stem] = record
        for path in sorted((self.root / 'repairs').glob('*.json')):
            record = _read(path)
            generation = record.get('generation')
            if (record.get('binding') != binding
                    or not isinstance(record.get('key'), str)
                    or path.stem != digest_json(record['key'])
                    or type(generation) is not int or generation < 1):
                raise KeywordCheckpointError('Keyword repair identity is invalid.')
            self.generations[record['key']] = generation
        for record in self.attempts.values():
            phase = record['phase']
            if phase in {'first_pass', 'targeted_retry'}:
                continue
            prefix, _, stage = str(phase).partition(':')
            try:
                generation = int(prefix.removeprefix('repair-'))
            except ValueError:
                generation = 0
            if (stage not in {'first_pass', 'targeted_retry'} or generation < 1
                    or phase != self._phase(stage, generation)
                    or generation > self.generations.get(record['key'], 0)):
                raise KeywordCheckpointError('Keyword repair attempt identity is invalid.')
        for path in sorted((self.root / 'completed').glob('*.json')):
            record = _read(path)
            if record.get('binding') != binding or path.stem != digest_json(record.get('key')):
                raise KeywordCheckpointError('Keyword result identity is invalid.')
            result = record.get('result')
            if not isinstance(result, dict) or result.get('key') != record['key']:
                raise KeywordCheckpointError('Keyword result row is invalid.')
            generation = result.get('keyword_checkpoint_generation', 0)
            if (type(generation) is not int or generation < 0
                    or generation > self.generations.get(record['key'], 0)):
                raise KeywordCheckpointError('Keyword result generation is invalid.')
            phases = ['first_pass']
            if (result.get('contract_diagnostics') or {}).get('targeted_retry_count'):
                phases.append('targeted_retry')
            if any(digest_json([record['key'], self._phase(phase, generation)])
                   not in self.attempts for phase in phases):
                raise KeywordCheckpointError('A saved keyword response is missing; refusing to resend.')
            if (result.get('contract_diagnostics') or {}).get('complete') is True and any(
                    self.attempts[digest_json([record['key'], self._phase(phase, generation)])]['state']
                    != 'returned' for phase in phases):
                raise KeywordCheckpointError('A valid keyword result has no saved response.')
            self.saved_keys.add(record['key'])
            self.completed_results[record['key']] = result
        self.resumed_chunks = len(self.saved_keys)
        if any(row['state'] == 'inflight' for row in self.attempts.values()) and not retry_unknown:
            raise KeywordRequestUnknownError(
                'A keyword request was interrupted without a saved response. '
                'Use --resume <MANIFEST> --retry-unknown only after reviewing possible duplicate requests.'
            )

    @staticmethod
    def _phase(phase: str, generation: int) -> str:
        return f'repair-{generation}:{phase}' if generation else phase

    def plan(self, request_rows: list[dict]) -> KeywordResumePlan:
        """Restore only complete results; failures and unfinished work stay queued."""
        restored: dict[str, dict] = {}
        pending: list[dict] = []
        failed = incomplete = 0
        for row in request_rows:
            result = self.completed_results.get(row['key'])
            if result is not None and not result.get('error') and not result.get('error_category'):
                if (result.get('contract_diagnostics') or {}).get('complete') is True:
                    restored[row['key']] = result
                    continue
                incomplete += 1
            elif result is not None:
                failed += 1
            pending.append(row)
        return KeywordResumePlan(restored, pending, failed, incomplete)

    def begin_repair(self, key: str) -> None:
        """Allow one new repair round per explicit resume, preserving every response.

        Persist the round before dispatch. A restart resumes an unfinished round
        instead of creating another request after its response was already saved.
        Unknown attempts stay in the same round and retain the acknowledgement gate.
        """
        result = self.completed_results.get(key)
        if result is None or result.get('error') or result.get('error_category'):
            return
        generation = self.generations.get(key, 0)
        if generation > result.get('keyword_checkpoint_generation', 0):
            return
        if any(record['key'] == key and record['state'] == 'inflight'
               for record in self.attempts.values()):
            return
        generation += 1
        _write(self.root / 'repairs' / f'{digest_json(key)}.json', {
            'binding': self.binding, 'key': key, 'generation': generation,
        })
        self.generations[key] = generation

    def restore(self, key: str) -> dict:
        """Return a validated completed row without rewriting its checkpoint."""
        result = self.completed_results[key]
        generation = result.get('keyword_checkpoint_generation', 0)
        phases = ['first_pass']
        if (result.get('contract_diagnostics') or {}).get('targeted_retry_count'):
            phases.append('targeted_retry')
        for phase in phases:
            record = self.attempts[digest_json([key, self._phase(phase, generation)])]
            if record['state'] == 'returned':
                self.reused_attempts += 1
        return result

    def call(self, key: str, phase: str, request: dict, dispatch, *, before_dispatch=None) -> dict:
        """Reuse a saved response or persist both sides of one provider call."""
        phase = self._phase(phase, self.generations.get(key, 0))
        identity = digest_json([key, phase])
        request_sha = digest_json(request)
        previous = self.attempts.get(identity)
        if previous is not None:
            if previous.get('request_sha256') != request_sha:
                raise KeywordCheckpointError('Frozen keyword request changed; start a new run.')
            if previous['state'] == 'returned':
                self.reused_attempts += 1
                return previous['response']
            if not self.retry_unknown:
                raise KeywordRequestUnknownError('Keyword request outcome is unknown; review before retrying.')
        if before_dispatch is not None:
            before_dispatch()
        record = {'binding': self.binding, 'key': key, 'phase': phase,
                  'request_sha256': request_sha, 'state': 'inflight'}
        path = self.root / 'attempts' / f'{identity}.json'
        _write(path, record)
        self.attempts[identity] = record
        response = dispatch()
        # Keep the same response fields used by canonical result generation;
        # provider clients, credentials and other transient objects stay out.
        response = {name: response[name] for name in (
            'response_payload', 'response_text', 'finish_reason', 'usage_metadata',
            'request_metadata', 'provider', 'model', 'execution_mode',
        ) if name in response}
        record = {**record, 'state': 'returned', 'response': response}
        _write(path, record)
        self.attempts[identity] = record
        return response

    def complete(self, key: str, result: dict) -> None:
        """Save the validated row before advancing the displayed saved count."""
        if generation := self.generations.get(key, 0):
            result['keyword_checkpoint_generation'] = generation
        _write(self.root / 'completed' / f'{digest_json(key)}.json',
               {'binding': self.binding, 'key': key, 'result': result})
        self.saved_keys.add(key)
        self.completed_results[key] = result
