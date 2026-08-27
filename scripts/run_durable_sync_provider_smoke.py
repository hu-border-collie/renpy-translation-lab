# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""Opt-in one-call Provider smoke across an abrupt durable-Sync interruption.

The child persists T2 dispatch intent, performs exactly one bounded contract
request through the shipped Provider adapter, writes only a safe success/fail
marker, and exits via ``os._exit`` before T3.  The parent then proves resume
marks the attempt outcome-unknown and never issues a duplicate Provider call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_io import atomic_write_json
from durable_sync_executor import DurableSyncExecutor
from litellm_provider_config import CustomLiteLLMProvider
from sync_retry_policy import ExecutorPolicy
from sync_run_contracts import RequestStatus, RunStatus, build_run_id
from sync_run_store import SyncRunStore
from scripts import run_provider_contract_smoke as contract_smoke


ACK_TEXT = 'I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST'
CHILD_SUCCESS_EXIT = 86
CUSTOM_PROVIDER_ID = 'durable_smoke_custom'
CUSTOM_KEY_ENV = 'DURABLE_SYNC_CUSTOM_API_KEY'


def _plan():
    return {
        'schema_version': 1,
        'plan_id': 'plan-durable-smoke',
        'plan_fingerprint': 'durable-smoke-v1',
        'source_identity': {'source_snapshot_fingerprint': 'smoke-source'},
        'config_fingerprint': 'smoke-config',
        'model_profile_snapshot': {'provider': 'smoke', 'model': 'smoke'},
        'execution_strategy': 'sync',
        'chunk_policy': {'max_items': 1},
        'context_policy': {},
        'chunks': [],
        'request_summaries': [],
        'artifacts': {},
    }


def _request():
    return {
        'request_id': 'req-1',
        'plan_id': 'plan-durable-smoke',
        'chunk_id': 'chunk-1',
        'system_instruction': 'Return the requested translation JSON only.',
        'user_prompt': 'Translate smoke-1.',
        'response_schema': {},
        'expected_ids': ['smoke-1'],
        'capability_requirements': {'structured_output': True},
        'generation_config': {
            'max_output_tokens': contract_smoke.MAX_OUTPUT_TOKENS,
            'timeout': contract_smoke.REQUEST_TIMEOUT_SECONDS,
        },
        'transport_metadata': {'smoke': True},
        'context_assembly': {},
        'prompt_fingerprint': 'smoke-prompt',
        'request_fingerprint': 'smoke-request',
    }


def _custom_spec(model: str) -> contract_smoke.ProviderSpec:
    return contract_smoke.ProviderSpec(
        'custom',
        'litellm',
        f'{CUSTOM_PROVIDER_ID}/{model}',
        CUSTOM_KEY_ENV,
    )


def _resolve_spec(args) -> contract_smoke.ProviderSpec:
    if args.provider_class == 'gemini':
        return contract_smoke.PROVIDER_BY_NAME['gemini']
    if args.provider_class == 'litellm':
        spec = contract_smoke.PROVIDER_BY_NAME.get(args.litellm_provider)
        if spec is None or spec.backend != 'litellm':
            raise ValueError('litellm provider must name a shipped built-in Provider')
        return spec
    if not args.custom_base_url or not args.custom_model:
        raise ValueError('custom smoke requires --custom-base-url and --custom-model')
    return _custom_spec(args.custom_model)


def _create_backend(spec, api_key, *, custom_base_url=''):
    if spec.name != 'custom':
        return contract_smoke.create_backend(spec, api_key)
    from litellm_sync_backend import LiteLLMSyncBackend

    custom = CustomLiteLLMProvider(
        id=CUSTOM_PROVIDER_ID,
        label='Durable smoke custom Provider',
        base_url=str(custom_base_url),
        models_url='',
        api_key_env=CUSTOM_KEY_ENV,
        requires_key=True,
    )
    return LiteLLMSyncBackend(
        api_key=api_key,
        custom_providers={CUSTOM_PROVIDER_ID: custom},
    )


def _worker(args) -> int:
    marker = Path(args._worker_marker)
    spec = contract_smoke.ProviderSpec(
        args._worker_name,
        args._worker_backend,
        args._worker_model,
        args._worker_secret_env,
    )
    api_key = contract_smoke.api_key_for(spec)
    try:
        store = SyncRunStore(args._worker_root, args._worker_run)
        owner = 'provider-smoke-child'
        store.acquire_lease(owner_token=owner, ttl_seconds=0.1)
        attempt_id = store.prepare_attempt(request_id='req-1', owner_token=owner)
        store.dispatch_attempt(attempt_id=attempt_id, owner_token=owner)
        backend = _create_backend(
            spec,
            api_key,
            custom_base_url=args._worker_custom_base_url,
        )
        result = backend.generate(contract_smoke.contract_request(spec))
        contract_smoke.validate_result(spec, result)
        atomic_write_json(
            marker,
            {
                'status': 'provider_succeeded_before_t3',
                'attempt_id': attempt_id,
                'provider': spec.name,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        atomic_write_json(
            marker,
            {
                'status': 'provider_failed',
                'provider': spec.name,
                'category': contract_smoke.classify_error(exc),
            },
            ensure_ascii=False,
            indent=2,
        )
        return 1
    os._exit(CHILD_SUCCESS_EXIT)


class _NeverDispatchBackend:
    calls = 0

    def send(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError('resume attempted a duplicate Provider request')

    def cancel(self, *, attempt):
        return False


def _unused_child_builder(*_args, **_kwargs):
    raise AssertionError('single-item unknown recovery must not derive a child')


def run_interruption_smoke(args) -> int:
    if args.acknowledge_billable_request != ACK_TEXT:
        print(
            f'REFUSE pass --acknowledge-billable-request {ACK_TEXT}',
            file=sys.stderr,
        )
        return 2
    try:
        spec = _resolve_spec(args)
    except ValueError as exc:
        print(f'REFUSE {exc}', file=sys.stderr)
        return 2
    api_key = contract_smoke.api_key_for(spec)
    if not api_key:
        print(f'SKIP provider={spec.name} missing_secret={spec.secret_environment}')
        return 0

    with tempfile.TemporaryDirectory(prefix='durable-sync-provider-smoke-') as tmp:
        root = Path(tmp) / 'runs'
        run_id = build_run_id()
        policy = ExecutorPolicy()
        store, _created = SyncRunStore.bootstrap(
            root,
            run_id,
            plan=_plan(),
            requests=[_request()],
            executor_policy=policy.to_dict(),
        )
        marker = Path(tmp) / 'safe_marker.json'
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--_worker-root', str(root),
            '--_worker-run', run_id,
            '--_worker-marker', str(marker),
            '--_worker-name', spec.name,
            '--_worker-backend', spec.backend,
            '--_worker-model', spec.model,
            '--_worker-secret-env', spec.secret_environment,
        ]
        if args.custom_base_url:
            command.extend(['--_worker-custom-base-url', args.custom_base_url])
        child = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=contract_smoke.REQUEST_TIMEOUT_SECONDS + 15,
        )
        try:
            marker_payload = json.loads(marker.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            marker_payload = {'status': 'missing_safe_marker'}
        if child.returncode != CHILD_SUCCESS_EXIT or marker_payload.get('status') != (
            'provider_succeeded_before_t3'
        ):
            print(
                f"FAIL provider={spec.name} category={marker_payload.get('category') or 'smoke_worker'}",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.15)
        backend = _NeverDispatchBackend()
        snapshot = DurableSyncExecutor(
            store,
            backend,
            derived_request_builder=_unused_child_builder,
            policy=policy,
        ).run()
        request = store.get_request('req-1')
        if (
            snapshot.get('run_status') != RunStatus.FAILED.value
            or request.get('status') != RequestStatus.OUTCOME_UNKNOWN.value
            or backend.calls
        ):
            print(f'FAIL provider={spec.name} category=unsafe_resume', file=sys.stderr)
            return 1
    print(
        f'PASS provider={spec.name} interruption=after_provider_before_t3 '
        'requests=1 duplicate_requests=0 outcome=unknown'
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--provider-class',
        choices=('gemini', 'litellm', 'custom'),
        default='gemini',
    )
    parser.add_argument('--litellm-provider', default='deepseek')
    parser.add_argument('--custom-base-url', default='')
    parser.add_argument('--custom-model', default='')
    parser.add_argument('--acknowledge-billable-request', default='')
    parser.add_argument('--_worker-root', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-run', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-marker', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-name', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-backend', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-model', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-secret-env', default='', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-custom-base-url', default='', help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args._worker_root:
        return _worker(args)
    return run_interruption_smoke(args)


if __name__ == '__main__':
    raise SystemExit(main())
