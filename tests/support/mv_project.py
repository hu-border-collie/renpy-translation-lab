"""Original MV material for product tests, never an executable game."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from engine_adapters.coverage import build_review_template, default_coverage_review_path
from engine_adapters.rpgmaker_mv import CORE_FILES
from execution.mv_sync import build_snapshot


def make_mv_project(root: Path):
    root = root.resolve()
    fixture = Path(__file__).resolve().parents[1] / 'fixtures' / 'rpgmaker_mv_narrative'
    source, target = root / 'source', root / 'target'
    shutil.copytree(fixture / 'source', source)
    materials = {rel: f'// Original test marker: {rel}\n'.encode() for rel in CORE_FILES}
    materials['js/rpg_core.js'] += b"Utils.RPGMAKER_NAME = 'MV';\n"
    materials['js/plugins.js'] = b'var $plugins = [{"name":"GALV_MessageStyles","status":true,"parameters":{}}];\n'
    materials['js/plugins/GALV_MessageStyles.js'] = b'// Original inert plugin marker.\n'
    for rel, content in materials.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    shutil.copytree(source, target)
    window = root / 'approved-window.json'
    window.write_text(json.dumps({'schema_version': 1, 'message_styles': True,
        'material_sha256': {rel: hashlib.sha256(content).hexdigest() for rel, content in materials.items()}}), encoding='utf-8')
    config = {'engine': 'rpgmaker_mv', 'game_root': str(target),
              'rpgmaker_mv': {'source_root': str(source), 'support_window_file': str(window)},
              'prepare': {'enabled': False},
              'sync': {'backend': 'litellm', 'model': 'openai/test-model',
                       'rag_enabled': False, 'source_index': {'enabled': False},
                       'story_memory': {'enabled': False},
                       'project_analysis': {'inject_published_brief': False}}}
    config_file = root / 'translator_config.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')
    snapshot = build_snapshot(config, config_file, require_review=False)
    review = build_review_template(snapshot.report)
    review.update(status='agent_reviewed', confirmed_at='2026-10-25T00:00:00Z',
                  reviewer={'type': 'agent', 'id': 'original-fixture-test-review', 'tool': 'unittest'})
    review_file = Path(default_coverage_review_path(target))
    review_file.parent.mkdir(parents=True)
    review_file.write_text(json.dumps(review), encoding='utf-8')
    return config_file, config, json.loads((fixture / 'positions.json').read_text(encoding='utf-8'))


def write_mv_scope(path: Path, source: Path, positions):
    payload = {'schema_version': 1, 'source_files': {}, 'occurrences': []}
    for row in positions:
        relative = row['file']
        payload['source_files'][relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        payload['occurrences'].append({'file_rel_path': relative, 'json_path': row['path'],
            'source_sha256': hashlib.sha256(row['source'].encode('utf-8')).hexdigest()})
    path.write_text(json.dumps(payload), encoding='utf-8')
    return payload
