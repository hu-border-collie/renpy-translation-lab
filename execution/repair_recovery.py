"""Local checkpoints and transactional publication for Ren'Py repair runs."""
from __future__ import annotations

from pathlib import Path

from engine_adapters.coverage import digest_json
from execution.sync_response_checkpoint import SyncResponseCheckpoint, SyncResponseCheckpointError, read_record, write_record
from foundation import atomic_io


class RepairRun:
    """Bind one locked repair directory to its report, requests and target bytes."""

    def __init__(self, directory, identity, files, *, resume):
        self.directory = Path(directory).resolve()
        self._header = self.directory / 'repair_run.json'
        self._journal = self.directory / '.repair_writeback_transaction.json'
        self.results = self.directory / 'repair_results.jsonl'
        self.failures = self.directory / 'repair_failures.jsonl'
        self.summary = self.directory / 'repair_summary.json'
        self._identity = identity
        self._resume = resume
        paths = sorted(set(files))
        if resume:
            self._record = read_record(self._header)
            sources = self._record.get('sources')
            if (self._record.get('version') != 1 or self._record.get('identity') != identity
                    or not isinstance(sources, dict) or sorted(sources) != paths):
                raise SyncResponseCheckpointError('Repair report, options or project changed since this run started.')
            self.sources = sources
            self._recover_publication()
        else:
            self.sources = {path: atomic_io.file_sha256(path) for path in paths}
            self._record = {
                'version': 1, 'identity': identity, 'sources': self.sources,
            }

    def _recover_publication(self):
        journal = atomic_io.read_atomic_write_transaction_journal(
            self._journal, expected_transaction_kind='repair',
        )
        if journal is None:
            return
        artifacts = {str(self.results), str(self.failures), str(self.summary)}
        entries = journal['entries']
        targets = {entry['target'] for entry in entries}
        expected_binding = digest_json(self._record)
        if (journal.get('metadata') != {'repair_binding': expected_binding}
                or len(targets) != len(entries)
                or not artifacts <= targets or not targets <= artifacts | self.sources.keys()
                or any(not self._matches_publication_entry(entry) for entry in entries)):
            raise SyncResponseCheckpointError('Repair writeback journal does not match this run.')
        atomic_io.recover_atomic_write_transaction(
            self._journal, expected_transaction_kind='repair', verify_targets=True,
        )

    def _matches_publication_entry(self, entry):
        target = Path(entry['target'])
        expected_preimage = self.sources.get(entry['target'])
        if (target.resolve() != target or entry.get('target_preimage_sha256') != expected_preimage
                or entry['existed'] != (expected_preimage is not None)
                or not entry.get('staged_sha256')):
            return False
        for name, suffix in (('staged_path', '.txn.tmp'), ('backup_path', '.txn.bak')):
            if name == 'backup_path' and not entry['existed']:
                return not entry[name]
            path = Path(entry[name])
            if (path.parent != target.parent or path.is_symlink()
                    or not path.name.startswith(f'.{target.name}.') or not path.name.endswith(suffix)):
                return False
        return True

    def require_source_snapshot(self):
        for path, expected in self.sources.items():
            if atomic_io.file_sha256(path) != expected:
                raise SyncResponseCheckpointError('A repair target changed since this run started.')

    def bind_requests(self, request_rows, *, retry_unknown):
        """Verify the entire request plan before any saved response or send."""
        request_digest = digest_json(request_rows)
        if self._resume:
            if self._record.get('requests_digest') != request_digest:
                raise SyncResponseCheckpointError('Repair requests or model configuration changed.')
        else:
            self._record['requests_digest'] = request_digest
            write_record(self._header, self._record)
        checkpoint = SyncResponseCheckpoint(
            self.directory, digest_json(self._record), task='repair',
            retry_unknown=retry_unknown, require_binding=self._resume, keep_dispatch_markers=True,
        )
        expected = {
            digest_json([f"{index}:{row['key']}", 'first_pass']): digest_json(row['request'])
            for index, row in enumerate(request_rows, start=1)
        }
        if any(key not in expected or expected[key] != record.get('request_sha256')
               for key, record in checkpoint.attempts.items()):
            raise SyncResponseCheckpointError('A saved repair attempt does not match the frozen requests.')
        if not retry_unknown and self.has_unknown(checkpoint):
            raise SyncResponseCheckpointError(
                'A repair request was dispatched without a saved response. '
                'Use --resume <RUN_DIR> --retry-unknown only after accepting possible duplicate charges.'
            )
        return checkpoint

    @staticmethod
    def has_unknown(checkpoint):
        return any(row['state'] == 'inflight' for row in checkpoint.attempts.values())

    def publish(self, writes, result_lines, failure_lines, summary_lines):
        """Commit the guarded target files and result reports as one transaction."""
        if any(path not in self.sources for path, _ in writes):
            raise SyncResponseCheckpointError('A repair write target is outside the frozen snapshot.')
        self.require_source_snapshot()
        artifacts = [
            (self.results, result_lines), (self.failures, failure_lines), (self.summary, summary_lines),
        ]
        atomic_io.atomic_write_many_lines(
            [*writes, *artifacts], journal_path=self._journal,
            transaction_kind='repair', metadata={'repair_binding': digest_json(self._record)},
            expected_preimages={**self.sources, **{path: None for path, _ in artifacts}},
        )
