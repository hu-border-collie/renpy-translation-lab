"""Recover validated Ren'Py query embeddings before a durable run exists."""
from __future__ import annotations

import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from context_engine import embedding_runtime
from context_engine.embedding_backend import (
    EmbeddingBatchRequest, EmbeddingBatchResult, validate_embedding_result,
)
from engine_adapters.coverage import digest_json
from execution.sync_response_checkpoint import SyncResponseCheckpoint, SyncResponseCheckpointError
from execution.sync_run_contracts import ErrorCode, SyncRunError, client_token_digest
from foundation.atomic_io import AtomicFileLockTimeoutError, exclusive_file_lock


class SyncPlanQueryCheckpoint:
    """Hold a per-preparation lock until the frozen plan is durably published.

    Project, material, scope and model drift select a different checkpoint.
    Only request digests, validated vectors and safe usage are saved locally.
    """

    def __init__(self, locks: ExitStack, *, client_token='', retry_token=''):
        self._locks = locks
        self._client_token_digest = client_token_digest(client_token)
        self._retry_token = retry_token
        self._checkpoint = None
        self._reused: set[str] = set()
        self._new: set[str] = set()

    def bind(self, run_root, *, project_root, tl_dir, plan_digest, settings, build_adapter):
        """Return a query function bound to the local plan and embedding identity."""
        identity = settings.query_identity()
        binding = digest_json({
            'version': 1, 'plan_digest': plan_digest, 'client_token_digest': self._client_token_digest,
            'project_root': os.path.normcase(os.path.realpath(project_root)),
            'tl_dir': os.path.normcase(os.path.realpath(tl_dir)),
            'embedding_identity': identity.to_dict(),
            'native_task_type': settings.native_query_task_type,
        })
        directory = Path(run_root) / '.plan-preparation' / binding
        try:
            self._locks.enter_context(exclusive_file_lock(directory / 'prepare.lock', timeout=0.1))
            checkpoint = SyncResponseCheckpoint(directory, binding, task='query', keep_dispatch_markers=True)
        except AtomicFileLockTimeoutError as exc:
            raise SyncRunError(ErrorCode.SYNC_RUN_BUSY, 'Another process is preparing this Sync plan.') from exc
        except (OSError, SyncResponseCheckpointError) as exc:
            raise self._storage_error() from exc
        unknown = [row for row in checkpoint.attempts.values() if row['state'] == 'inflight']
        token = digest_json([binding, sorted(
            (row['key'], row.get('dispatch_nonce', '')) for row in unknown
        )]) if unknown else ''
        if self._retry_token and self._retry_token != token:
            raise self._stale_confirmation()
        if unknown and not self._retry_token:
            raise SyncRunError(
                ErrorCode.SYNC_PLAN_QUERY_OUTCOME_UNKNOWN,
                'A query embedding was dispatched without a saved response. '
                f'Repeat sync-start with --retry-unknown-embedding {token} only after accepting possible duplicate charges.',
                safe_details={'confirmation_token': token, 'unknown_queries': len(unknown)},
            )
        checkpoint.retry_unknown = bool(unknown and self._retry_token)
        self._checkpoint = checkpoint

        def embed(query_text):
            request = EmbeddingBatchRequest(
                identity=identity, inputs=(query_text,), timeout_seconds=settings.timeout_seconds,
            )
            key = request.fingerprint
            was_saved = key in checkpoint.returned_first_pass
            adapter = None

            def provider_error(exc):
                return SyncRunError(
                    ErrorCode.SYNC_PLAN_QUERY_FAILED,
                    'Query embedding failed during Sync plan preparation. Saved queries were retained.',
                    safe_details={
                        **embedding_runtime.public_error_diagnostics(exc),
                        'saved_queries': len(checkpoint.returned_first_pass),
                    },
                )

            def prepare_dispatch():
                nonlocal adapter
                try:
                    adapter = build_adapter()
                except Exception as exc:
                    raise provider_error(exc) from exc

            def dispatch():
                try:
                    result = validate_embedding_result(request, adapter.embed(request))
                except Exception as exc:
                    raise provider_error(exc) from exc
                return {'response_payload': {'vector': list(result.vectors[0])},
                        'usage_metadata': result.usage.to_dict()}

            try:
                response = checkpoint.call(
                    key, 'first_pass', request.to_dict(), dispatch, before_dispatch=prepare_dispatch,
                )
                vector = EmbeddingBatchResult(
                    identity=identity, request_fingerprint=key,
                    vectors=(response['response_payload']['vector'],),
                ).vectors[0]
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise self._storage_error() from exc
            if not was_saved:
                self._new.add(key)
            elif key not in self._new:
                self._reused.add(key)
            return list(vector)

        return embed

    def require_confirmation_used(self):
        if self._retry_token and (
            self._checkpoint is None
            or any(row['state'] == 'inflight' for row in self._checkpoint.attempts.values())
        ):
            raise self._stale_confirmation()

    def summary(self):
        if self._checkpoint is None:
            return {}
        return {'saved_queries': len(self._checkpoint.returned_first_pass),
                'reused_queries': len(self._reused), 'new_queries': len(self._new)}

    def publish(self):
        """Retire the checkpoint after the run and its target artifact exist."""
        try:
            if self._checkpoint is not None:
                root = self._checkpoint.root
                retirement = root.with_name(f'.published-{uuid4().hex}')
                os.replace(root, retirement)
                if retirement.resolve().parent == root.parent.resolve() and not retirement.is_symlink():
                    shutil.rmtree(retirement, ignore_errors=True)
        finally:
            self._locks.close()

    @staticmethod
    def _storage_error():
        return SyncRunError(ErrorCode.SYNC_RUN_STORAGE_ERROR, 'Could not validate or save the Sync query checkpoint.')

    @staticmethod
    def _stale_confirmation():
        return SyncRunError(
            ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH,
            'The confirmed query checkpoint has changed. Start again without retry confirmation.',
        )
