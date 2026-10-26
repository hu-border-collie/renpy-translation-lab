"""Explicit engine configuration, audited windows and GUI invalidation evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from engine_adapters.project_config import (project_engine, read_mv_project_config,
                                            read_project_configuration, require_engine_operation, mv_material_revision)
from tests.support.mv_project import make_mv_project


class ProjectConfigTests(unittest.TestCase):
    def test_default_and_invalid_engine_are_distinct(self):
        self.assertEqual(project_engine({}), 'renpy')
        for value in ('', 'rpgmaker_mz', None, False, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project_engine({'engine': value})

    def test_routes_refuse_unsupported_mv_operations(self):
        for operation in ('build', 'work-export', 'generate-template', 'prepare', 'legacy-sync',
                          'build-keywords', 'preview-revisions', 'project-analysis-build-structure'):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'durable Sync'):
                require_engine_operation({'engine': 'rpgmaker_mv'}, operation)
        for operation in ('sync-start', 'sync-status', 'sync-resume', 'sync-derive',
                          'sync-cancel', 'check', 'apply', 'doctor', 'coverage-status',
                          'sync-keywords', 'export-keywords', 'merge-keywords-to-glossary',
                          'bootstrap-rag', 'bootstrap-source-index'):
            require_engine_operation({'engine': 'rpgmaker_mv'}, operation)

    def test_malformed_configuration_is_not_interpreted_as_default_renpy(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'translator_config.json'
            for content in ('[]', '{"engine":', '{"engine":"renpy","engine":"rpgmaker_mv"}',
                            '{"engine":"rpgmaker_mv","metadata":NaN}'):
                with self.subTest(content=content), self.assertRaises(ValueError):
                    file.write_text(content, encoding='utf-8')
                    read_project_configuration(file)

    def test_relative_roots_window_schema_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file, config, _positions = make_mv_project(root)
            config['game_root'] = 'target'
            config['rpgmaker_mv'] = {'source_root': 'source', 'support_window_file': 'approved-window.json'}
            settings = read_mv_project_config(config, file)
            self.assertEqual(settings.target_root, root / 'target')
            self.assertEqual(settings.source_root, root / 'source')
            config['rpgmaker_mv']['reviewed_history_file'] = 'reviewed.jsonl'
            self.assertEqual(read_mv_project_config(config, file).reviewed_history_file, root / 'reviewed.jsonl')
            config['rpgmaker_mv']['reviewed_history_file'] = False
            with self.assertRaisesRegex(ValueError, 'reviewed_history_file'):
                read_mv_project_config(config, file)
            config['rpgmaker_mv']['reviewed_history_file'] = ''
            revision = mv_material_revision(config, file)
            path = root / 'source/data/Map001.json'
            path.write_bytes(path.read_bytes() + b' ')
            self.assertNotEqual(revision, mv_material_revision(config, file))
            original = json.loads((root / 'approved-window.json').read_text())
            for key, value in (('schema_version', True), ('message_styles', 'yes'), ('material_sha256', [])):
                with self.subTest(key=key):
                    changed = copy.deepcopy(original)
                    changed[key] = value
                    (root / 'approved-window.json').write_text(json.dumps(changed), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'schema'):
                        read_mv_project_config(config, file)


if __name__ == '__main__':
    unittest.main()
