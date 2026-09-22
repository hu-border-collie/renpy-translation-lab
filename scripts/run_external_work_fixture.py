"""Run the original #508 fixture through actual CLI dispatch with networking disabled.

Default: serial canned candidates + candidate revision + check/preview/apply.
--prepare-only exports material for independent host agents. --resume consumes
their separate --submission files through the very same contract.
This is engineering evidence, not a real chapter or semantic-quality benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SERIAL = [
    '灯笼还是温的。', '这灯笼还热着呢。', '[name]，别跟着那道蓝光走。',
    '哦，{i}真棒{/i}。又是一扇锁着的门。', '等待信号', '回到灯笼旁',
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Isolated fixture run directory.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--submission', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
        tl = root / 'project' / 'game' / 'tl' / 'schinese'
        tl.mkdir(parents=True)
        (tl / 'chapter.rpy').write_bytes((REPO / 'tests/fixtures/external_work/chapter.rpy').read_bytes())
        write_json(root / 'config.json', {
            'game_root': str(root / 'project'), 'tl_subdir': 'game/tl/schinese',
            'glossary_file': '', 'prepare': {'enabled': False, 'language': 'schinese'},
            'batch': {'rag': {'enabled': False}, 'source_index': {'enabled': False},
                      'project_analysis': {'enabled': False}},
        })
    sys.path.insert(0, str(REPO))
    import gemini_translate_batch as batch
    import external_translation_work as work

    batch.legacy.TRANSLATOR_CONFIG = str(root / 'config.json')
    batch.legacy.API_KEYS = []
    steps: list[dict[str, Any]] = []
    start = time.perf_counter()

    def cli(*command: str) -> dict[str, Any]:
        output, diagnostics = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            code = batch.main([*command, '--output', 'json', '--non-interactive', '--strict-exit-codes'])
        result = json.loads(output.getvalue())
        steps.append({'command': command[0], 'exit_code': code, 'status': result['status']})
        if code not in (0, 3) or not result['ok']:
            raise RuntimeError(f'{command[0]} failed: {result}')
        return result['result']

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('Project model/network invocation is forbidden in this fixture run.')

    with contextlib.ExitStack() as guards:
        guards.enter_context(mock.patch.object(socket.socket, 'connect', forbidden))
        for module, names in (
            (batch, ('create_client', 'run_sync_request', 'prepare_rag_store', 'embed_source_segments')),
            (batch.legacy, ('create_client',)),
        ):
            for name in names:
                if hasattr(module, name):
                    guards.enter_context(mock.patch.object(module, name, forbidden))
        if not args.resume:
            cli('work-export', '--output-dir', str(root / 'work'))
        manifest = str(root / 'work' / 'manifest.json')
        package = json.loads((root / 'work' / 'work.json').read_text(encoding='utf-8'))
        if args.prepare_only:
            write_json(root / 'assignment.json', {
                'work_path': str(root / 'work' / 'work.json'),
                'scope_a': [row['occurrence_id'] for row in package['items'][:3]],
                'scope_b': [row['occurrence_id'] for row in package['items'][3:]],
                'review_status': 'unreviewed', 'host_model': 'unknown', 'host_usage': 'unknown',
            })
            print(json.dumps({'status': 'prepared', 'assignment': str(root / 'assignment.json')}, ensure_ascii=False))
            return 0
        submissions = list(args.submission)
        if not submissions:
            for name, indexes in (('serial-a', range(3)), ('serial-b', range(3, 6))):
                submission = work.submission_template(package, submission_id=name,
                                                     producer={'type': 'human', 'name': 'canned original fixture'})
                submission['items'] = [
                    {'occurrence_id': package['items'][i]['occurrence_id'],
                     'snapshot_digest': package['items'][i]['snapshot_digest'],
                     'expected_candidate_digest': '', 'translation': SERIAL[i],
                     'reason': '原创 fixture 工程验证', 'review': {'status': 'unreviewed'}}
                    for i in indexes
                ]
                path = root / f'{name}.json'
                write_json(path, submission)
                submissions.append(path)
        for path in submissions:
            cli('work-submit', manifest, str(path.resolve()))
            cli('work-status', manifest)
        receipt = cli('work-submit', manifest, str(submissions[0].resolve()))
        assert receipt['generation'] == 1, 'Replay must return original receipt, not a new generation.'
        if not args.submission:
            item = cli('work-read', manifest, '--limit', '1')['items'][0]
            revision = work.submission_template(package, submission_id='serial-revision',
                                               producer={'type': 'human', 'name': 'canned fixture revision'})
            revision['items'] = [{
                'occurrence_id': item['occurrence_id'], 'snapshot_digest': item['snapshot_digest'],
                'expected_candidate_digest': item['candidate']['candidate_digest'],
                'translation': '灯笼还带着余温。', 'reason': '候选订正验证',
                'review': {'status': 'reviewed', 'reviewer': 'fixture coordinator'},
            }]
            path = root / 'revision.json'
            write_json(path, revision)
            cli('work-submit', manifest, str(path))
        cli('check', manifest)
        check_summary = batch.load_manifest(manifest)['last_check_summary']
        assert check_summary['writeback_gate']['decision'] == 'allow'
        cli('work-preview', manifest)
        cli('work-apply', manifest)
        cli('work-apply', manifest)
        status = cli('work-status', manifest)
        target = root / 'project' / 'game' / 'tl' / 'schinese' / 'chapter.rpy'
        text = target.read_text(encoding='utf-8')
        assert '    a ""' not in text and '    b ""' not in text and '    new ""' not in text
        assert status['remaining_ids'] == [] and status['writeback'] == 'applied'
        assert '[name]' in text and '{i}' in text and '{/i}' in text
        result = {
            'kind': 'original_fixture_engineering_run', 'schema_version': 1,
            'route': 'independent_submissions' if args.submission else 'serial_canned_fixture',
            'project_network': 'disabled_and_guarded', 'project_model_calls': 'forbidden',
            'host_model': 'unknown', 'host_usage': 'unknown', 'host_cost': 'unknown',
            'cli_elapsed_seconds': round(time.perf_counter() - start, 3), 'steps': steps,
            'received_count': status['received_count'], 'remaining_count': len(status['remaining_ids']),
            'writeback': status['writeback'], 'writeback_gate': check_summary['writeback_gate'],
            'quality_gate': check_summary['quality_gate'],
            'semantic_quality': 'not_independently_accepted', 'in_game': 'not_tested',
            'real_chapter_experiment': 'pending_authorized_sample_and_budget',
        }
        write_json(root / 'run_report.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
