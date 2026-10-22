"""Offline research harness for #520; no credentials, HTTP, or writeback.

Freeze an original TranslationPlan, construct the proposed Batch envelope, and
inspect a locally supplied receipt. This is not an ExecutionStrategy or an import
path for production manifests. Real receipts belong in an ignored directory.
The default v1 reproduces the first dated run, including its name-policy conflict.
The explicit DeepSeek v2 fixes that conflict without rewriting the first run.
Neither small fixture establishes full-game translation quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_review import translation_plan as tp  # noqa: E402 - repository path bootstrap
from quality_review import translation_core  # noqa: E402
from quality_review import structure_protection  # noqa: E402
from providers.openai_compatible_sync_backend import OpenAICompatibleSyncBackend  # noqa: E402
from providers.sync_model_backend import SyncGenerationRequest  # noqa: E402

VERSION = 1
MODEL = 'openai/gpt-4.1-mini'
PROVIDER = {'only': ['openai'], 'allow_fallbacks': False}
ENDPOINT = '/v1/chat/completions'
FIXTURE = ROOT / 'tests/fixtures/openrouter_batch_spike/source.json'
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}
REMOTE_STATES = TERMINAL | {'validating', 'in_progress', 'finalizing', 'cancelling'}


def digest(value):
    return hashlib.sha256(tp.canonical_json(value).encode('utf-8')).hexdigest()


def chat_body(request, *, model=MODEL):
    """Use the existing Sync serializer without resolving or sending credentials."""
    backend = OpenAICompatibleSyncBackend(
        provider='openrouter', base_url='https://openrouter.ai/api/v1',
        credential_ref={'kind': 'none'}, structured_output_mode='strict_json_schema',
    )
    body, _ = backend._build_payload(SyncGenerationRequest(
        model=model, contents=request.user_prompt,
        config={**request.generation_config,
                'system_instruction': request.system_instruction,
                'response_json_schema': request.response_schema},
    ))
    # Batch inherits its model, and does not stream. Do not append :batch.
    body.pop('model')
    body.pop('stream')
    if request.generation_config.get('reasoning') is not None:
        # Explicit research transport extension: the production Sync serializer
        # does not currently forward OpenRouter's reasoning controls.
        body['reasoning'] = deepcopy(request.generation_config['reasoning'])
    return body


def freeze_fixture(fixture=None, *, experiment='gpt41-v1'):
    """Freeze the existing Sync plan as a semantic baseline for a candidate transport."""
    source = fixture or json.loads(FIXTURE.read_text(encoding='utf-8'))
    if experiment not in {'gpt41-v1', 'deepseek41-v2'}:
        raise ValueError('UNKNOWN_RESEARCH_EXPERIMENT')
    modern = experiment == 'deepseek41-v2'
    model = 'deepseek/deepseek-v4.1-flash' if modern else MODEL
    provider = {'only': ['deepinfra/fp8'], 'allow_fallbacks': False} if modern else PROVIDER
    generation = ({'temperature': 0.2, 'max_output_tokens': 1024, 'reasoning': {'enabled': False}} if modern
                  else {'temperature': 0.2, 'max_output_tokens': 1024})
    macro = ('Original harbor vignette. Translate into Simplified Chinese. '
             'Keep Mira and Oren in English. Keep restrained conversational voices.' if modern
             else 'Original harbor vignette. Translate into Simplified Chinese. '
                  'Mira = 米拉; Oren = 奥伦. Keep restrained conversational voices.')
    profile = {
        'id': 'issue520-openrouter', 'adapter': 'openai_compatible',
        'provider': 'openrouter', 'model': model,
        'base_url': 'https://openrouter.ai/api/v1',
        'credential_ref': {'kind': 'keyring', 'name': 'openrouter'},
        'params': {**generation, 'structured_output_mode': 'strict_json_schema'},
    }
    build = tp.build_translation_plan(
        source['file_jobs'], execution_strategy='sync',
        source_identity={
            'engine': 'renpy', 'adapter_version': 'research-fixture-v1',
            'project_identity_digest': digest('issue520-original'),
            'source_snapshot_fingerprint': digest(source['file_jobs']),
            'file_digests': {j['file_rel_path']: digest(j) for j in source['file_jobs']},
        },
        model_profile_snapshot=profile,
        config_snapshot={'source_language': 'en', 'target_language': 'zh-CN'},
        chunk_policy=tp.ChunkPolicy(max_items=3),
        macro_setting=macro, generation_config=generation,
    )
    mappings, requests = [], []
    for request in build.requests:
        body = chat_body(request, model=model)
        custom_id = 'i520-' + digest({
            'plan': build.plan.plan_fingerprint, 'request': request.to_dict(),
        })[:40]
        requests.append({'custom_id': custom_id, 'body': body})
        mappings.append({
            'custom_id': custom_id, 'request': request.to_dict(),
            'wire_body_sha256': digest(body),
        })
    if len({r['custom_id'] for r in requests}) != len(requests):
        raise ValueError('CUSTOM_ID_COLLISION')
    # Insertion order matters on the wire: requests MUST be last.
    envelope = {'endpoint': ENDPOINT, 'model': model, 'provider': deepcopy(provider),
                'completion_window': '24h', 'requests': requests}
    frozen = {'contract_version': VERSION, 'purpose': 'research-only-no-writeback',
              'source': source, 'plan': build.plan.to_dict(), 'mapping': mappings,
              'envelope': envelope}
    if modern:
        frozen['experiment'] = experiment
    frozen['frozen_sha256'] = digest(frozen)
    return frozen


def validate_frozen(frozen):
    """Reject stale source, prompt, schema, identity, or wire data before association."""
    payload = deepcopy(frozen)
    recorded = payload.pop('frozen_sha256', '')
    if payload.get('contract_version') != VERSION or digest(payload) != recorded:
        raise ValueError('FROZEN_DIGEST_MISMATCH')
    tp.validate_plan_fingerprint(frozen['plan'])
    # Rebuild from the public source rather than trusting re-hashed stale fields.
    if freeze_fixture(frozen['source'], experiment=frozen.get('experiment', 'gpt41-v1')) != frozen:
        raise ValueError('FROZEN_SOURCE_OR_REQUEST_MISMATCH')
    return frozen['frozen_sha256']


def serialize_envelope(frozen):
    validate_frozen(frozen)
    e = frozen['envelope']
    ordered = {k: e[k] for k in ('endpoint', 'model', 'provider', 'completion_window', 'requests')}
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def submission_recovery(record):
    """Return a read-only recovery instruction; an attempted POST is never retried.

    This small research rule does not claim server-side idempotency. In particular,
    a missing ID after dispatch does not establish that no paid work was created.
    """
    state = record.get('state')
    batch_id = record.get('batch_id')
    if state == 'prepared':
        return 'review_budget_before_first_submit'
    if batch_id and state in {'accepted', 'dispatched_outcome_unknown'}:
        return 'poll_known_batch'
    if state == 'dispatched_outcome_unknown':
        return 'reconcile_list_or_support_keep_reservation'
    if state == 'rejected' and record.get('http_status') in {400, 401, 402, 403, 404, 422}:
        return 'stop_rejected_attempt'
    return 'attention_no_submit'


def map_receipt(frozen, receipt, *, batch_id):
    """Associate untrusted offline results; candidates never imply check=allow.

    Fatal envelope/ID corruption quarantines the whole receipt. Ordinary item
    failures remain per-request failures, so successful siblings can be inspected.
    Semantic quality and structure/source freshness remain the existing checker's
    responsibility in any future production integration.
    """
    validate_frozen(frozen)
    if not batch_id or receipt.get('id') != batch_id:
        raise ValueError('BATCH_ID_MISMATCH')
    modern = frozen.get('experiment') == 'deepseek41-v2'
    model = frozen['envelope']['model']
    provider_name = 'DeepInfra' if modern else 'OpenAI'
    allowed_models = ({model, 'deepseek/deepseek-v4.1-flash-20260910',
                       'deepseek-ai/DeepSeek-V4.1-Flash', 'DeepSeek-V4.1-Flash'} if modern
                      else {MODEL, 'gpt-4.1-mini-2025-04-14', 'openai/gpt-4.1-mini-2025-04-14'})
    if (receipt.get('model') not in allowed_models or receipt.get('provider') != provider_name
            or receipt.get('endpoint') != ENDPOINT):
        raise ValueError('ROUTE_MISMATCH')
    status = receipt.get('status')
    if status not in REMOTE_STATES:
        raise ValueError('UNKNOWN_REMOTE_STATE')
    output = {'contract_version': VERSION, 'frozen_sha256': frozen['frozen_sha256'],
              'batch_id': batch_id, 'remote_status': status,
              'receipt_sha256': digest(receipt), 'candidates': [], 'failures': [],
              'usage': deepcopy(receipt.get('usage')), 'writeback_authorized': False}
    rows = receipt.get('results')
    if status != 'completed':
        if rows is not None:
            raise ValueError('UNEXPECTED_NONCOMPLETED_RESULTS')
        output['local_state'] = 'terminal_no_results' if status in TERMINAL else 'pending'
        return output
    if not isinstance(rows, list):
        raise ValueError('COMPLETED_RESULTS_MISSING')
    expected = {m['custom_id']: m for m in frozen['mapping']}
    source_items = {t['id']: t for job in frozen['source']['file_jobs'] for t in job['tasks']}
    if any(not isinstance(r, dict) or not isinstance(r.get('custom_id'), str) for r in rows):
        raise ValueError('INVALID_RESULT_ROW')
    counts = Counter(r['custom_id'] for r in rows)
    if any(n != 1 for n in counts.values()) or set(counts) - set(expected):
        raise ValueError('DUPLICATE_OR_UNKNOWN_CUSTOM_ID')
    rc = receipt.get('request_counts') or {}
    successes = sum(r.get('response') is not None for r in rows)
    if (rc.get('total') != len(expected) or rc.get('completed') != successes
            or rc.get('failed') != len(rows) - successes):
        raise ValueError('REQUEST_COUNTS_MISMATCH')
    for row in rows:
        cid = row['custom_id']
        mapping = expected[cid]
        request = mapping['request']
        response, error = row.get('response'), row.get('error')
        if (response is None) == (error is None):
            raise ValueError('RESPONSE_ERROR_EXCLUSIVITY')
        failure = None
        if error is not None:
            failure = 'REMOTE_ITEM_ERROR'
        elif response.get('status_code') != 200:
            failure = 'REMOTE_HTTP_ERROR'
        else:
            body = response.get('body') or {}
            if body.get('model') not in allowed_models:
                raise ValueError('RESULT_MODEL_MISMATCH')
            if body.get('provider') not in {None, provider_name}:
                raise ValueError('RESULT_PROVIDER_MISMATCH')
            choices = body.get('choices') or []
            choice = choices[0] if len(choices) == 1 else {}
            message = choice.get('message') or {}
            if choice.get('finish_reason') != 'stop' or message.get('refusal'):
                failure = 'INCOMPLETE_OR_REFUSED'
            else:
                try:
                    parsed = json.loads(message.get('content', ''))
                    units = [source_items[i] for i in request['expected_ids']]
                    report = translation_core.validate_model_response(
                        parsed, expected_units=units, allow_legacy=False,
                    )
                    report = structure_protection.validate_report(report, request, units)
                    if not report.complete:
                        raise ValueError
                    items = report.to_envelope()['translations']
                except (ValueError, KeyError, TypeError):
                    failure = 'INVALID_CONTENT_OR_ITEM_IDS'
                if failure is None:
                    output['candidates'].append({
                        'custom_id': cid, 'chunk_id': request['chunk_id'],
                        'request_id': request['request_id'],
                        'request_fingerprint': request['request_fingerprint'],
                        'prompt_fingerprint': request['prompt_fingerprint'],
                        'result_sha256': digest(row), 'generation_id': body.get('id'),
                        'translations': items,
                    })
        if failure:
            output['failures'].append({'custom_id': cid, 'category': failure})
    for cid in sorted(set(expected) - set(counts)):
        output['failures'].append({'custom_id': cid, 'category': 'MISSING_CUSTOM_ID'})
    output['local_state'] = 'candidates_with_errors' if output['failures'] else 'candidates'
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='Offline: freeze the original public fixture.')
    freeze.add_argument('--output-dir', type=Path, required=True)
    freeze.add_argument('--experiment', choices=['gpt41-v1', 'deepseek41-v2'], default='gpt41-v1')
    inspect = sub.add_parser('inspect', help='Offline: inspect a private local receipt; no import/apply.')
    inspect.add_argument('--frozen', type=Path, required=True)
    inspect.add_argument('--receipt', type=Path, required=True)
    inspect.add_argument('--batch-id', required=True)
    inspect.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        frozen = freeze_fixture(experiment=args.experiment)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'frozen.json').write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (args.output_dir / 'submit.json').write_bytes(serialize_envelope(frozen))
        print(json.dumps({'frozen_sha256': frozen['frozen_sha256'],
                          'requests': len(frozen['mapping']),
                          'wire_bytes': len(serialize_envelope(frozen))}))
    else:
        result = map_receipt(json.loads(args.frozen.read_text(encoding='utf-8')),
                             json.loads(args.receipt.read_text(encoding='utf-8')),
                             batch_id=args.batch_id)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'local_state': result['local_state'],
                          'candidates': len(result['candidates']), 'failures': len(result['failures']),
                          'writeback_authorized': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
