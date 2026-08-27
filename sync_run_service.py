# -*- coding: utf-8 -*-
"""Pure service boundary for durable Sync start/resume/status/cancel (#347 P4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from atomic_io import atomic_write_text, file_sha256
from durable_sync_executor import (
    DurableSyncExecutor,
    ProviderFailure,
    ProviderOutcome,
)
from sync_result_export import deliver_usage_outbox, export_run_artifacts
from sync_retry_policy import ExecutorPolicy
from sync_retry_policy import missing_lineage_suffix
import sync_run_contracts as contracts
from sync_run_contracts import (
    ErrorCategory,
    ErrorCode,
    RunStatus,
    SyncRunError,
    build_run_id,
    canonical_json,
    normalize_client_token,
    sha256_hex,
)
from sync_run_store import SyncRunStore


BackendFactory = Callable[[SyncRunStore], Any]
DerivedBuilderFactory = Callable[[SyncRunStore], Callable]
ReservationFactory = Callable[[SyncRunStore], Callable[[Mapping[str, Any]], Mapping[str, Any]]]
FreshnessReporter = Callable[[SyncRunStore], Mapping[str, Any]]
ReuseValidator = Callable[[str, Any], bool]
RunArtifactProvider = Callable[[SyncRunStore], Sequence[Mapping[str, Any]]]


def _plan_build_payload(plan_build) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = getattr(plan_build, 'plan', None)
    requests = getattr(plan_build, 'requests', None)
    if plan is None and isinstance(plan_build, Mapping):
        plan = plan_build.get('plan')
        requests = plan_build.get('requests')
    if hasattr(plan, 'to_dict'):
        plan = plan.to_dict()
    request_payloads = [
        request.to_dict() if hasattr(request, 'to_dict') else dict(request)
        for request in (requests or [])
    ]
    if not isinstance(plan, Mapping) or requests is None:
        raise ValueError('plan_build requires a plan and request collection')
    return dict(plan), request_payloads


def _profile_digest(plan: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(plan.get('model_profile_snapshot') or {}))


def _resume_fingerprint(plan: Mapping[str, Any]) -> str:
    payload = {
        'plan_id': plan.get('plan_id'),
        'plan_fingerprint': plan.get('plan_fingerprint'),
        'source_identity': plan.get('source_identity') or {},
        'config_fingerprint': plan.get('config_fingerprint') or '',
        'model_profile_snapshot': plan.get('model_profile_snapshot') or {},
    }
    return sha256_hex(canonical_json(payload))


def default_freshness_report(_store: SyncRunStore) -> dict[str, Any]:
    return {
        'resume_allowed': True,
        'source': 'fresh',
        'profile': 'fresh',
        'config': 'fresh',
        'reasons': [],
    }


def find_latest_run(root_dir: str | Path) -> str:
    """Select the latest valid durable DB, ignoring legacy preview folders."""
    root = Path(root_dir)
    candidates: list[tuple[str, str]] = []
    if root.is_dir():
        for entry in root.iterdir():
            if not entry.is_dir() or not contracts.validate_run_id(entry.name):
                continue
            if not (entry / 'state.sqlite3').is_file():
                continue
            try:
                store = SyncRunStore(root, entry.name)
                run = store.get_run()
            except (OSError, ValueError, SyncRunError):
                continue
            if str(run.get('run_id') or '') != entry.name:
                continue
            candidates.append((str(run.get('created_at') or ''), entry.name))
    if not candidates:
        raise SyncRunError(
            ErrorCode.SYNC_RUN_NOT_FOUND,
            'no durable Sync run was found',
            safe_details={'root_dir': str(root)},
        )
    candidates.sort(reverse=True)
    latest_time = candidates[0][0]
    latest = [run_id for created_at, run_id in candidates if created_at == latest_time]
    if len(latest) != 1:
        raise SyncRunError(
            ErrorCode.SYNC_RUN_NOT_FOUND,
            'latest durable Sync run is ambiguous',
            safe_details={'candidate_count': len(latest)},
        )
    return latest[0]


class SyncRunService:
    """GUI/CLI-neutral facade; all methods return JSON-safe snapshots."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        backend_factory: BackendFactory | None = None,
        derived_builder_factory: DerivedBuilderFactory | None = None,
        reservation_factory: ReservationFactory | None = None,
        freshness_reporter: FreshnessReporter | None = None,
        game_root: str | Path | None = None,
        pricing_config: Mapping[str, Any] | None = None,
        reuse_validator: ReuseValidator | None = None,
        run_artifact_provider: RunArtifactProvider | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.backend_factory = backend_factory
        self.derived_builder_factory = derived_builder_factory
        self.reservation_factory = reservation_factory
        self.freshness_reporter = freshness_reporter or default_freshness_report
        self.game_root = None if game_root is None else Path(game_root)
        self.pricing_config = dict(pricing_config or {})
        self.reuse_validator = reuse_validator or (lambda _item_id, _translation: False)
        self.run_artifact_provider = run_artifact_provider

    def start(
        self,
        plan_build,
        *,
        policy: ExecutorPolicy | Mapping[str, Any] | None = None,
        client_token: str | None = None,
        wait_for_backoff: bool = True,
    ) -> dict[str, Any]:
        plan, requests = _plan_build_payload(plan_build)
        if not requests:
            raise ValueError('durable Sync start has no pending translation requests')
        frozen_policy = (
            policy if isinstance(policy, ExecutorPolicy) else ExecutorPolicy.from_mapping(policy)
        )
        self._preflight_cost_reservations(requests, frozen_policy)
        normalized_token = normalize_client_token(client_token)
        run_id = build_run_id(normalized_token)
        store, created = SyncRunStore.bootstrap(
            self.root_dir,
            run_id,
            plan=plan,
            requests=requests,
            client_token=normalized_token,
            executor_policy=frozen_policy.to_dict(),
            run_meta={
                'profile_digest': _profile_digest(plan),
                'config_digest': str(plan.get('config_fingerprint') or ''),
                'resume_compatibility_fingerprint': _resume_fingerprint(plan),
            },
        )
        self._ensure_run_artifacts(store)
        before = int(store.get_run()['revision'])
        snapshot = self._execute(store, wait_for_backoff=wait_for_backoff)
        snapshot['changed'] = bool(created or int(snapshot['revision']) != before)
        return snapshot

    def resume(
        self,
        run_id: str,
        *,
        policy_overrides: Mapping[str, Any] | None = None,
        wait_for_backoff: bool = True,
    ) -> dict[str, Any]:
        store = SyncRunStore(self.root_dir, run_id)
        if policy_overrides:
            stored = ExecutorPolicy.from_mapping(
                json.loads(store.get_run()['policy_json'] or '{}')
            )
            merged = stored.to_dict()
            merged.update(dict(policy_overrides))
            if ExecutorPolicy.from_mapping(merged) != stored:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'resume policy overrides cannot change a frozen run',
                    safe_details={'run_id': run_id},
                )
        before = int(store.get_run()['revision'])
        run_status = RunStatus(str(store.get_run()['status']))
        if run_status in contracts.RUN_TERMINAL_STATES:
            snapshot = self._postprocess(store)
        else:
            snapshot = self._execute(store, wait_for_backoff=wait_for_backoff)
        snapshot['changed'] = int(snapshot['revision']) != before
        return snapshot

    def status(self, run_id: str | None = None, *, latest: bool = False) -> dict[str, Any]:
        if bool(run_id) == bool(latest):
            raise ValueError('status requires exactly one of run_id or latest=True')
        selected = find_latest_run(self.root_dir) if latest else str(run_id)
        store = SyncRunStore(self.root_dir, selected)
        snapshot = self._snapshot(store)
        snapshot['changed'] = False
        return snapshot

    def cancel(self, run_id: str, *, reason: str = 'user') -> dict[str, Any]:
        store = SyncRunStore(self.root_dir, run_id)
        changed = store.cancel_intent(reason=reason)
        try:
            run_status = RunStatus(str(store.get_run()['status']))
            if run_status is RunStatus.CANCEL_REQUESTED:
                snapshot = self._execute(store, wait_for_backoff=False)
            else:
                snapshot = self._postprocess(store)
        except SyncRunError as exc:
            if exc.code is not ErrorCode.SYNC_RUN_BUSY:
                raise
            snapshot = self._snapshot(store)
        snapshot['changed'] = changed
        return snapshot

    def derive(
        self,
        run_id: str,
        current_plan_build,
        *,
        policy: ExecutorPolicy | Mapping[str, Any] | None = None,
        reuse_policy: Mapping[str, Any] | None = None,
        retry_unknown: bool = False,
        ack_duplicate_billing_risk: bool = False,
        exclude_unknown: bool = False,
        wait_for_backoff: bool = True,
    ) -> dict[str, Any]:
        """Create and execute a new run without mutating ``run_id``.

        Reuse is deliberately strict: the complete immutable root request
        payload must match, the source winner must be authoritative, and the
        current adapter validator must still accept the translation.
        """
        if retry_unknown and exclude_unknown:
            raise ValueError('retry_unknown and exclude_unknown are mutually exclusive')
        if retry_unknown and not ack_duplicate_billing_risk:
            raise ValueError(
                'retry_unknown requires ack_duplicate_billing_risk=True'
            )
        if ack_duplicate_billing_risk and not retry_unknown:
            raise ValueError(
                'ack_duplicate_billing_risk is only valid with retry_unknown'
            )
        source = SyncRunStore(self.root_dir, run_id)
        violations = source.verify_integrity()
        if violations:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'source durable run failed integrity verification',
                safe_details={'run_id': run_id, 'violations': violations[:20]},
            )
        if RunStatus(str(source.get_run()['status'])) not in contracts.RUN_TERMINAL_STATES:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'derive requires a terminal source run',
                safe_details={'run_id': run_id},
            )
        plan, requests = _plan_build_payload(current_plan_build)
        if not requests:
            raise ValueError('durable Sync derive has no pending translation requests')
        unknown_ids = self._unknown_item_ids(source)
        current_ids = {
            str(item_id)
            for request in requests
            for item_id in request.get('expected_ids') or []
        }
        if unknown_ids and not retry_unknown and not exclude_unknown:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_OUTCOME_UNKNOWN,
                'source run contains outcome-unknown items',
                safe_details={
                    'run_id': run_id,
                    'unknown_count': len(unknown_ids),
                },
            )
        if exclude_unknown and current_ids & unknown_ids:
            raise ValueError(
                'exclude_unknown requires a current scoped plan that omits every unknown ID'
            )

        frozen_policy = (
            policy if isinstance(policy, ExecutorPolicy) else ExecutorPolicy.from_mapping(policy)
        )
        self._preflight_cost_reservations(requests, frozen_policy)
        derived_run_id = build_run_id()
        derivation = {
            'source_run_id': run_id,
            'reuse_policy': dict(reuse_policy or {}),
            'retry_unknown': bool(retry_unknown),
            'duplicate_billing_risk_acknowledged': bool(
                ack_duplicate_billing_risk
            ),
            'excluded_unknown_ids': (
                sorted(unknown_ids) if exclude_unknown else []
            ),
        }
        target, _created = SyncRunStore.bootstrap(
            self.root_dir,
            derived_run_id,
            plan=plan,
            requests=requests,
            executor_policy=frozen_policy.to_dict(),
            run_meta={
                'profile_digest': _profile_digest(plan),
                'config_digest': str(plan.get('config_fingerprint') or ''),
                'resume_compatibility_fingerprint': _resume_fingerprint(plan),
                'derived_from_run_id': run_id,
                'derivation': derivation,
            },
        )
        self._ensure_run_artifacts(target)
        self._seed_reusable_results(
            source,
            target,
            requests=requests,
            retry_unknown_ids=unknown_ids if retry_unknown else set(),
            policy=frozen_policy,
        )
        snapshot = self._execute(target, wait_for_backoff=wait_for_backoff)
        snapshot['changed'] = True
        return snapshot

    @staticmethod
    def _unknown_item_ids(store: SyncRunStore) -> set[str]:
        unknown: set[str] = set()
        with store._conn() as conn:
            rows = conn.execute(
                'SELECT expected_ids_json FROM requests WHERE run_id = ? AND status = ?',
                (store.run_id, contracts.RequestStatus.OUTCOME_UNKNOWN.value),
            ).fetchall()
            for row in rows:
                unknown.update(json.loads(row['expected_ids_json'] or '[]'))
        return unknown

    def _seed_reusable_results(
        self,
        source: SyncRunStore,
        target: SyncRunStore,
        *,
        requests: Sequence[Mapping[str, Any]],
        retry_unknown_ids: set[str],
        policy: ExecutorPolicy,
    ) -> None:
        if self.derived_builder_factory is None:
            return
        owner = f'derive-seed-{target.run_id[-16:]}'
        target.acquire_lease(owner_token=owner)
        try:
            child_builder = self.derived_builder_factory(target)
            with source._conn() as conn:
                source_roots = {
                    str(row['request_id']): row
                    for row in conn.execute(
                        'SELECT * FROM requests WHERE run_id = ? '
                        'AND parent_request_id IS NULL',
                        (source.run_id,),
                    ).fetchall()
                }
                for current in requests:
                    request_id = str(current['request_id'])
                    source_root = source_roots.get(request_id)
                    if source_root is None:
                        continue
                    if str(source_root['payload_json']) != canonical_json(dict(current)):
                        continue
                    expected_ids = list(current.get('expected_ids') or [])
                    winners = conn.execute(
                        'SELECT item_results.* FROM item_results JOIN requests '
                        'ON requests.run_id = item_results.run_id '
                        'AND requests.request_id = item_results.winner_request_id '
                        'WHERE item_results.run_id = ? AND requests.root_request_id = ?',
                        (source.run_id, request_id),
                    ).fetchall()
                    reusable = {}
                    for winner in winners:
                        item_id = str(winner['item_id'])
                        if item_id not in expected_ids or item_id in retry_unknown_ids:
                            continue
                        attempt_id = winner['winner_attempt_id']
                        if attempt_id:
                            attempt = conn.execute(
                                'SELECT status FROM attempts WHERE attempt_id = ?',
                                (attempt_id,),
                            ).fetchone()
                            if attempt is None or str(attempt['status']) != contracts.AttemptStatus.SUCCEEDED.value:
                                continue
                        elif not winner['reused_from_run_id']:
                            continue
                        translation = json.loads(winner['translation_payload_json'])
                        if not self.reuse_validator(item_id, translation):
                            continue
                        reusable[item_id] = {
                            'translation': translation,
                            'translation_digest': winner['translation_digest'],
                            'validation_diagnostics': json.loads(
                                winner['validation_diagnostics_json'] or '{}'
                            ),
                            'source_attempt_id': (
                                attempt_id or winner['reused_from_attempt_id'] or ''
                            ),
                        }
                    if not reusable:
                        continue
                    missing = [item_id for item_id in expected_ids if item_id not in reusable]
                    children = []
                    if missing:
                        reason = target.lineage_budget_reason(
                            request_id=request_id,
                            child_count=1,
                            policy=policy.to_dict(),
                        )
                        if reason:
                            continue
                        children = [
                            child_builder(
                                current,
                                missing,
                                missing_lineage_suffix(missing),
                                contracts.LineageKind.MISSING_IDS,
                            )
                        ]
                    target.seed_reused_results(
                        request_id=request_id,
                        owner_token=owner,
                        source_run_id=source.run_id,
                        reused_items=reusable,
                        derived_requests=children,
                    )
        finally:
            target.release_lease(owner_token=owner)

    def _preflight_cost_reservations(
        self,
        requests: Sequence[Mapping[str, Any]],
        policy: ExecutorPolicy,
    ) -> None:
        if policy.max_estimated_cost is None:
            return
        if self.reservation_factory is None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUDGET_EXHAUSTED,
                'a hard cost cap requires trustworthy per-request pricing',
                safe_details={'reason_code': contracts.REASON_RUN_BUDGET_EXHAUSTED_COST},
            )
        # The real reservation callable is store-bound and cannot exist before
        # T0.  A factory may expose a pure ``preflight`` method for this phase.
        preflight = getattr(self.reservation_factory, 'preflight', None)
        if not callable(preflight):
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUDGET_EXHAUSTED,
                'cost reservation provider does not support preflight pricing',
                safe_details={'reason_code': contracts.REASON_RUN_BUDGET_EXHAUSTED_COST},
            )
        estimates = [dict(preflight(request) or {}) for request in requests]
        if any(
            SyncRunStore._numeric_cost(
                reservation, 'estimated_cost', 'cost_upper_bound', 'cost'
            ) is None
            for reservation in estimates
        ):
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUDGET_EXHAUSTED,
                'cost preflight returned an unknown estimate',
                safe_details={'reason_code': contracts.REASON_RUN_BUDGET_EXHAUSTED_COST},
            )

    def _execute(self, store: SyncRunStore, *, wait_for_backoff: bool) -> dict[str, Any]:
        self._ensure_run_artifacts(store)
        if RunStatus(str(store.get_run()['status'])) in contracts.RUN_TERMINAL_STATES:
            return self._postprocess(store)
        if self.backend_factory is None or self.derived_builder_factory is None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'durable Sync execution dependencies are not configured',
                safe_details={'run_id': store.run_id},
            )
        report = dict(self.freshness_reporter(store) or {})
        if not bool(report.get('resume_allowed', False)):
            raise SyncRunError(
                ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH,
                'durable Sync freshness check refused model dispatch',
                safe_details={
                    'run_id': store.run_id,
                    'reasons': list(report.get('reasons') or []),
                },
            )
        reservation = (
            self.reservation_factory(store)
            if self.reservation_factory is not None
            else None
        )
        plan = store.get_plan()['plan']
        profile = dict(plan.get('model_profile_snapshot') or {})
        executor = DurableSyncExecutor(
            store,
            self.backend_factory(store),
            derived_request_builder=self.derived_builder_factory(store),
            reservation_provider=reservation,
            freshness_check=lambda _store: bool(
                self.freshness_reporter(_store).get('resume_allowed', False)
            ),
            provider=str(profile.get('provider') or ''),
            model=str(profile.get('model') or profile.get('model_name') or ''),
            profile_digest=_profile_digest(plan),
        )
        executor.run(wait_for_backoff=wait_for_backoff)
        return self._postprocess(store)

    def _ensure_run_artifacts(self, store: SyncRunStore) -> None:
        if self.run_artifact_provider is None:
            return
        for artifact in self.run_artifact_provider(store) or ():
            kind = str(artifact.get('kind') or '').strip()
            relative_path = str(artifact.get('relative_path') or '').strip()
            content = artifact.get('content')
            if not kind or not relative_path or not isinstance(content, str):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'run artifact provider returned an invalid artifact',
                    safe_details={'run_id': store.run_id},
                )
            path = store.resolve_artifact_path(relative_path)
            atomic_write_text(path, content, newline='\n')
            store.put_artifact(
                kind=kind,
                relative_path=relative_path,
                sha256_digest=file_sha256(path),
                schema_version=int(artifact.get('schema_version') or 1),
            )

    def _postprocess(self, store: SyncRunStore) -> dict[str, Any]:
        status = RunStatus(str(store.get_run()['status']))
        if status in contracts.RUN_TERMINAL_STATES:
            export_run_artifacts(store)
        if self.game_root is not None:
            deliver_usage_outbox(
                store,
                game_root=self.game_root,
                pricing_config=self.pricing_config,
            )
        return self._snapshot(store)

    def _snapshot(self, store: SyncRunStore) -> dict[str, Any]:
        snapshot = store.build_snapshot()
        snapshot['freshness'] = dict(self.freshness_reporter(store) or {})
        snapshot['artifacts'] = {
            'run_dir': str(store.run_dir),
            'state_db': str(store.db_path),
            **{
                row['kind']: str(store.run_dir / row['relative_path'])
                for row in self._artifact_rows(store)
            },
        }
        return snapshot

    @staticmethod
    def _artifact_rows(store: SyncRunStore) -> list[dict[str, Any]]:
        with store._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM artifacts WHERE run_id = ? ORDER BY kind',
                (store.run_id,),
            ).fetchall()
            return [dict(row) for row in rows]


_BACKEND_CATEGORY_MAP = {
    'authentication': ErrorCategory.AUTHENTICATION,
    'rate_limit': ErrorCategory.RATE_LIMIT,
    'service_unavailable': ErrorCategory.PROVIDER_SERVER,
    'timeout': ErrorCategory.TIMEOUT,
    'invalid_response': ErrorCategory.INVALID_STRUCTURED_RESPONSE,
    'unsupported_capability': ErrorCategory.UNSUPPORTED_CAPABILITY,
    'missing_dependency': ErrorCategory.MISSING_DEPENDENCY,
    'provider_error': ErrorCategory.UNKNOWN_PROVIDER,
}


class ProductionSyncBackendAdapter:
    """Convert the shipped one-call Sync backend into executor receipts.

    ``generate_once`` must perform exactly one Provider call.  #347 owns every
    retry outside this adapter.
    """

    def __init__(
        self,
        generate_once: Callable[[Mapping[str, Any], float], Mapping[str, Any]],
        item_resolver: Callable[[Mapping[str, Any], Sequence[str]], Sequence[Mapping]],
        *,
        translation_validator: Callable[[Mapping, str], tuple[bool, str]] | None = None,
    ):
        self.generate_once = generate_once
        self.item_resolver = item_resolver
        self.translation_validator = translation_validator

    def send(self, request, *, attempt, timeout_seconds):
        import sync_model_backend
        import translation_core

        try:
            result = dict(self.generate_once(request, timeout_seconds) or {})
            parsed = result.get('parsed')
            if parsed is None:
                text = str(result.get('response_text') or '')
                parsed = json.loads(text) if text else None
            expected_ids = list(request.get('expected_ids') or [])
            target_items = list(self.item_resolver(request, expected_ids))
            actual_ids = [str(item.get('id') or '') for item in target_items]
            if actual_ids != expected_ids:
                raise ValueError('production item resolver changed request identity')
            report = translation_core.validate_model_response(
                parsed,
                mode=translation_core.MODE_TRANSLATION,
                expected_units=target_items,
                allow_legacy=True,
            )
            accepted = {}
            for item in report.items:
                item_id = str(item.get('id') or '')
                translation = str(item.get('translation') or '')
                if not translation.strip():
                    continue
                source = next(target for target in target_items if str(target.get('id')) == item_id)
                if self.translation_validator is not None:
                    valid, _reason = self.translation_validator(source, translation)
                    if not valid:
                        continue
                accepted[item_id] = {'translation': translation}
            if not accepted:
                reason = (
                    report.issues[0].reason_code
                    if report.issues
                    else 'response.zero_valid_items'
                )
                raise ProviderFailure(
                    ErrorCategory.INVALID_STRUCTURED_RESPONSE,
                    reason,
                    usage_metadata=result.get('usage_metadata') or {},
                )
            return ProviderOutcome(
                accepted_items=accepted,
                response_payload=result.get('response_payload') or {},
                normalized_payload=report.to_envelope(),
                contract_diagnostics=report.to_diagnostics(),
                usage_metadata=result.get('usage_metadata') or {},
            )
        except ProviderFailure:
            raise
        except Exception as exc:
            category = sync_model_backend.sync_error_category(exc)
            raise ProviderFailure(
                _BACKEND_CATEGORY_MAP.get(category, ErrorCategory.UNKNOWN_PROVIDER),
                f'provider.{category}',
                safe_details={'exception_type': type(exc).__name__},
                retry_after_seconds=getattr(exc, 'retry_after_seconds', None),
            ) from exc

    def cancel(self, *, attempt):
        return False


def build_production_backend_adapter(
    routing_plan,
    route,
    item_resolver: Callable[[Mapping[str, Any], Sequence[str]], Sequence[Mapping]],
    *,
    translation_validator: Callable[[Mapping, str], tuple[bool, str]] | None = None,
) -> ProductionSyncBackendAdapter:
    """Bind the current shipped routing/backend stack to one-call attempts."""
    import gemini_translate_batch

    def generate_once(request: Mapping[str, Any], timeout_seconds: float):
        return gemini_translate_batch.run_sync_request(
            {
                'contents': request.get('user_prompt') or '',
                'system_instruction': request.get('system_instruction') or '',
                'generation_config': {
                    **dict(request.get('generation_config') or {}),
                    'response_mime_type': 'application/json',
                    'response_json_schema': dict(request.get('response_schema') or {}),
                },
            },
            route,
            plan=routing_plan,
            # A durable attempt must correspond to exactly one Provider
            # invocation. Hidden same-attempt key rotation would make crash
            # recovery and outcome_unknown billing semantics unauditable.
            retry_attempts=1,
            allow_credential_rotation=False,
            timeout_seconds=timeout_seconds,
        )

    return ProductionSyncBackendAdapter(
        generate_once,
        item_resolver,
        translation_validator=translation_validator,
    )


def build_production_sync_run_service(
    root_dir: str | Path,
    execution_context,
    *,
    game_root: str | Path | None = None,
    pricing_config: Mapping[str, Any] | None = None,
    reservation_factory: ReservationFactory | None = None,
) -> SyncRunService:
    """Bind a current-project runtime context to the pure service facade.

    Freshness is recomputed from the same #346 plan and root request payloads
    that a new run would freeze.  Derived request rows are deliberately not
    compared to the root plan: they are deterministic descendants whose
    payload hashes are already guarded by :class:`SyncRunStore`.
    """
    current_plan, current_requests = _plan_build_payload(
        execution_context.plan_build
    )
    current_request_payloads = {
        str(request.get('request_id') or ''): canonical_json(request)
        for request in current_requests
    }

    def freshness_reporter(store: SyncRunStore) -> dict[str, Any]:
        stored_plan = store.get_plan()['plan']
        source_fresh = canonical_json(
            stored_plan.get('source_identity') or {}
        ) == canonical_json(current_plan.get('source_identity') or {})
        profile_fresh = canonical_json(
            stored_plan.get('model_profile_snapshot') or {}
        ) == canonical_json(current_plan.get('model_profile_snapshot') or {})
        config_fresh = str(stored_plan.get('config_fingerprint') or '') == str(
            current_plan.get('config_fingerprint') or ''
        )
        plan_fresh = str(stored_plan.get('plan_fingerprint') or '') == str(
            current_plan.get('plan_fingerprint') or ''
        )
        stored_roots = {
            str(row['request_id']): str(row['payload_json'])
            for row in store.list_requests()
            if row.get('parent_request_id') is None
        }
        requests_fresh = stored_roots == current_request_payloads
        reasons = []
        if not source_fresh:
            reasons.append('source_snapshot_changed')
        if not profile_fresh:
            reasons.append('model_profile_changed')
        if not config_fresh:
            reasons.append('sync_config_changed')
        if not plan_fresh:
            reasons.append('translation_plan_changed')
        if not requests_fresh:
            reasons.append('root_requests_changed')
        return {
            'resume_allowed': not reasons,
            'source': 'fresh' if source_fresh else 'stale',
            'profile': 'fresh' if profile_fresh else 'stale',
            'config': 'fresh' if config_fresh else 'stale',
            'reasons': reasons,
        }

    def backend_factory(_store: SyncRunStore):
        return build_production_backend_adapter(
            execution_context.routing_plan,
            execution_context.route,
            execution_context.item_resolver,
            translation_validator=execution_context.validate_translation,
        )

    def derived_builder_factory(_store: SyncRunStore):
        from durable_sync_executor import TranslationPlanDerivedRequestBuilder

        return TranslationPlanDerivedRequestBuilder(
            execution_context.item_resolver,
            context_resolver=execution_context.context_resolver,
        )

    def run_artifact_provider(store: SyncRunStore):
        payload = execution_context.durable_targets_payload(run_id=store.run_id)
        return [{
            'kind': 'targets_json',
            'relative_path': 'targets.json',
            'content': canonical_json(payload) + '\n',
            'schema_version': 1,
        }]

    return SyncRunService(
        root_dir,
        backend_factory=backend_factory,
        derived_builder_factory=derived_builder_factory,
        reservation_factory=reservation_factory,
        freshness_reporter=freshness_reporter,
        game_root=game_root,
        pricing_config=pricing_config,
        reuse_validator=execution_context.validate_reused_translation,
        run_artifact_provider=run_artifact_provider,
    )
