"""Exact MV occurrence scoping without weakening full-material coverage."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from engine_adapters.mv_scope import read_occurrence_scope
from engine_adapters.project_config import read_mv_project_config, mv_material_revision
from execution.mv_sync import build_snapshot, material_identity, live_material_identity, require_material_fresh
from tests.support.mv_project import make_mv_project, write_mv_scope


class MvOccurrenceScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.file, self.config, self.positions = make_mv_project(self.root)
        self.scope_file = self.root / 'occurrence-scope.json'
        self.rows = self.positions[:3]
        self.payload = write_mv_scope(self.scope_file, self.root / 'source', self.rows)
        self.config['rpgmaker_mv']['occurrence_scope_file'] = self.scope_file.name

    def test_only_exact_positions_are_selected_and_mirrors_stay_linked(self):
        snapshot = build_snapshot(self.config, self.file)
        self.assertEqual(len(snapshot.occurrences), 14)
        self.assertEqual(snapshot.pending_task_count, 3)
        self.assertEqual({tuple(row.locator.locator['json_path']) for row in snapshot.selected_occurrences},
                         {tuple(row['path']) for row in self.rows})
        choice = next(row for row in snapshot.selected_occurrences if row.locator.locator.get('mirror_paths'))
        self.assertEqual(choice.locator.locator['mirror_paths'], self.rows[2]['mirrors'])
        self.assertEqual(material_identity(snapshot)['occurrence_scope_digest'], read_occurrence_scope(self.scope_file)[0])
        self.assertEqual(read_mv_project_config(self.config, self.file).occurrence_scope_file, self.scope_file)
        unscoped = copy.deepcopy(self.config)
        unscoped['rpgmaker_mv'].pop('occurrence_scope_file')
        self.assertNotIn('occurrence_scope_digest', material_identity(build_snapshot(unscoped, self.file)))

    def test_scope_drift_invalidates_gui_revision_and_dispatch_identity(self):
        snapshot = build_snapshot(self.config, self.file)
        before = material_identity(snapshot)
        revision = mv_material_revision(self.config, self.file)
        self.scope_file.write_bytes(self.scope_file.read_bytes() + b' ')
        self.assertNotEqual(revision, mv_material_revision(self.config, self.file))
        current = live_material_identity(snapshot, self.config, self.file)
        with self.assertRaisesRegex(ValueError, 'occurrence_scope_digest'):
            require_material_fresh(before, current)

    def test_invalid_scope_shapes_and_duplicate_json_keys_are_rejected(self):
        for mutate in (
            lambda p: p.update(schema_version=True), lambda p: p.update(schema_version=2),
            lambda p: p.update(occurrences=[]),
            lambda p: p['occurrences'][0]['json_path'].__setitem__(1, True),
            lambda p: p['occurrences'][0]['json_path'].__setitem__(1, -1),
            lambda p: p['occurrences'].append(copy.deepcopy(p['occurrences'][0])),
            lambda p: p['source_files'].update({'unused.json': '0' * 64}),
            lambda p: p['occurrences'][0].update(source_sha256='bad'),
        ):
            payload = copy.deepcopy(self.payload)
            mutate(payload)
            self.scope_file.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_occurrence_scope(self.scope_file)
        self.scope_file.write_text(json.dumps(self.payload).replace('"schema_version": 1',
            '"schema_version": 1, "schema_version": 1'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_occurrence_scope(self.scope_file)

    def test_stale_sources_mirror_paths_and_file_filter_conflicts_are_rejected(self):
        for mutate in (
            lambda p: p['source_files'].update({self.rows[0]['file']: '0' * 64}),
            lambda p: p['occurrences'][0].update(source_sha256='0' * 64),
            lambda p: p['occurrences'][0].update(json_path=['missing']),
            lambda p: p['occurrences'][2].update(json_path=self.rows[2]['mirrors'][0]),
        ):
            payload = copy.deepcopy(self.payload)
            mutate(payload)
            self.scope_file.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                build_snapshot(self.config, self.file)
        self.scope_file.write_text(json.dumps(self.payload), encoding='utf-8')
        self.config['include_files'] = ['data/Map002.json']
        with self.assertRaisesRegex(ValueError, 'conflict'):
            build_snapshot(self.config, self.file)

    def test_scope_does_not_hide_unknown_controls_in_other_files(self):
        path = self.root / 'source/data/Map002.json'
        path.write_bytes(path.read_bytes().replace(b'Same words.', b'\\\\Unknown[1]', 1))
        (self.root / 'target/data/Map002.json').write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'coverage is blocked'):
            build_snapshot(self.config, self.file, require_review=False)


if __name__ == '__main__':
    unittest.main()
