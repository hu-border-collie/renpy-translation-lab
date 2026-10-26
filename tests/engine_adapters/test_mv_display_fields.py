"""Original display positions, explicit roles, exact bytes and refusal paths."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from engine_adapters import InventoryPolicy, ValidatedTranslation, render_writeback_plan
from engine_adapters.project_config import read_mv_project_config
from engine_adapters.rpgmaker_mv import RpgMakerMvAdapter
from engine_adapters.rpgmaker_mv_fields import display_role
from execution.mv_sync import build_snapshot, material_identity, require_material_fresh
from quality_review import structure_protection
from tests.support.mv_project import add_mv_display_material, make_mv_project, review_mv_project, write_mv_scope


FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/rpgmaker_mv_display'


class MvDisplayFieldTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.file, self.config, _ = make_mv_project(self.root)
        self.positions = add_mv_display_material(self.root)
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms', 'map_display_name']

    def snapshot(self):
        return review_mv_project(self.config, self.file)

    def submitted(self, snapshot):
        positions = {(row['file'], tuple(row['path'])): row for row in self.positions}
        validated = []
        for row in snapshot.occurrences:
            key = row.unit.file_rel_path, tuple(row.locator.locator['json_path'])
            if key in positions:
                self.assertEqual(row.unit.source, positions[key]['source'])
                translated = positions[key]['translation']
                validated.append(ValidatedTranslation(row, translated, snapshot.adapter.validate_translation(row, translated)))
        self.assertEqual(len(validated), 11)
        return validated

    def test_default_is_narrative_and_each_opt_in_is_independent(self):
        for fields, expected in (([], 14), (['system_terms'], 22), (['map_display_name'], 17),
                                  (['system_terms', 'map_display_name'], 25)):
            self.config['rpgmaker_mv']['display_fields'] = fields
            with self.subTest(fields=fields):
                snapshot = self.snapshot()
                self.assertEqual(snapshot.pending_task_count, expected)
                self.assertFalse(snapshot.report.invariant_errors)
        self.config['rpgmaker_mv'].pop('display_fields')
        self.assertEqual(self.snapshot().pending_task_count, 14)

    def test_fixed_positions_independent_audit_and_handwritten_output(self):
        snapshot = self.snapshot()
        self.assertEqual(snapshot.report.catalog_provenance['narrative_physical_positions'], 21)
        self.assertEqual(snapshot.report.catalog_provenance['display_physical_positions'], 13)
        submitted = self.submitted(snapshot)
        plan = snapshot.adapter.build_writeback_plan(snapshot.project, submitted,
            snapshot.project.source_documents, live_targets=snapshot.project.target_documents)
        rendered = render_writeback_plan(plan, snapshot.project.source_documents,
                                         live_targets=snapshot.project.target_documents)
        self.assertEqual(len(plan.operations), 11)
        self.assertEqual(plan.adapter_version, '0.3.0')
        self.assertEqual(plan.writeback_plan_schema_version, 2)
        for relative, lines in rendered.items():
            if relative in ('data/Map001.json', 'data/Map002.json'):
                row = next(row for row in self.positions if row['file'] == relative)
                expected = (self.root / 'target' / relative).read_bytes().replace(
                    json.dumps(row['source']).encode(), json.dumps(row['translation'], ensure_ascii=False).encode())
            else:
                expected = (FIXTURE / 'expected' / relative).read_bytes()
            self.assertEqual(''.join(lines).encode('utf-8'), expected)
        display = next(row for row in snapshot.inventory.candidates if row.structure_kind == 'mv_system_message')
        altered = replace(snapshot.inventory, candidates=tuple(row for row in snapshot.inventory.candidates if row != display))
        self.assertIn('rpgmaker_mv.coverage.mismatch', snapshot.adapter.audit_extraction(snapshot.project, altered).reason_codes)

    def test_formats_move_but_spelling_count_control_order_and_role_are_bound(self):
        snapshot = self.snapshot()
        submitted = self.submitted(snapshot)
        template = next(item for item in submitted if item.occurrence.unit.source.startswith('Load '))
        row = template.occurrence
        self.assertEqual(template.validation.status, 'pass')
        for text in ('读取%1。', '读取%1／%2／%3。', '读取%1／%2\\v[2]。',
                     '读取%01／%2\\V[2]。', '读取%1／%2\\pop[0]。', '读取%1\n%2\\V[2]。'):
            with self.subTest(text=text):
                self.assertEqual(snapshot.adapter.validate_translation(row, text).status, 'block')
        repeated = next(item.occurrence for item in submitted if item.occurrence.unit.source.startswith('%1 loses'))
        self.assertEqual(snapshot.adapter.validate_translation(repeated, '%1损失%2。').status, 'block')
        views, maps = structure_protection.model_units([row.unit], engine='rpgmaker_mv', request_id='original-request')
        self.assertNotIn('%1', views[0].text)
        mapping = maps['items'][row.unit.id]
        self.assertEqual(mapping['mv_text_role'], 'system_message')
        protected_translation = template.translated_text
        for entry in mapping['entries']:
            protected_translation = protected_translation.replace(entry['value'], entry['marker'], 1)
        restored = structure_protection.restore(protected_translation, mapping, source=row.unit.text,
            engine='rpgmaker_mv', request_id='original-request', item_id=row.unit.id,
            mv_message_styles=True, mv_text_role='system_message')
        self.assertEqual(restored, template.translated_text)
        with self.assertRaisesRegex(ValueError, 'mapping_mismatch'):
            structure_protection.restore(protected_translation, mapping, source=row.unit.text,
                engine='rpgmaker_mv', request_id='original-request', item_id=row.unit.id, mv_message_styles=True)

    def test_target_bom_crlf_and_existing_translation_preserve_independent_source(self):
        target = self.root / 'target/data/System.json'
        raw = b'\xef\xbb\xbf' + target.read_bytes().replace(b'\n', b'\r\n').replace(
            b'"Rank"', '"已有阶位"'.encode())
        target.write_bytes(raw)
        snapshot = self.snapshot()
        row = next(row for row in snapshot.occurrences if row.unit.source == 'Rank')
        self.assertEqual(row.unit.current_translation, '已有阶位')
        self.assertEqual(snapshot.pending_task_count, 24)
        plan = snapshot.adapter.build_writeback_plan(snapshot.project, self.submitted(snapshot),
            snapshot.project.source_documents, live_targets=snapshot.project.target_documents)
        rendered = render_writeback_plan(plan, snapshot.project.source_documents,
                                         live_targets=snapshot.project.target_documents)
        expected = b'\xef\xbb\xbf' + (FIXTURE / 'expected/data/System.json').read_bytes().replace(b'\n', b'\r\n')
        self.assertEqual(''.join(rendered['data/System.json']).encode(), expected)
        self.assertIn(b'"Rank"', (self.root / 'source/data/System.json').read_bytes())

    def test_precise_scope_accepts_enabled_leaves_and_rejects_reserved_or_disabled(self):
        scope = self.root / 'display-scope.json'
        rows = [self.positions[0], self.positions[-1]]
        write_mv_scope(scope, self.root / 'source', rows)
        self.config['rpgmaker_mv']['occurrence_scope_file'] = str(scope)
        self.assertEqual(self.snapshot().pending_task_count, 2)
        self.config['rpgmaker_mv']['display_fields'] = []
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            self.snapshot()
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms']
        for path in (['terms', 'commands', 20], ['terms', 'messages', 'pluginLabel'], ['gameTitle']):
            source = json.loads((self.root / 'source/data/System.json').read_bytes())
            value = source
            for component in path:
                value = value[component]
            write_mv_scope(scope, self.root / 'source', [{'file': 'data/System.json', 'path': path, 'source': value}])
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unsupported'):
                self.snapshot()
        self.assertEqual(display_role('data/System.json', ('terms', 'basic', True), ('system_terms',)), '')

    def test_non_target_structure_and_unsupported_fields_stay_immutable(self):
        for relative, old, new in (
            ('data/System.json', b'Original fixture title', b'Changed title'),
            ('data/System.json', b'Reserved twenty', b'Changed reserve'),
            ('data/System.json', b'Preserve unknown member', b'Changed plugin member'),
            ('data/Map003.json', b'Preserve map note', b'Changed note'),
            ('data/Actors.json', b'name', b'label'),
        ):
            path = self.root / 'target' / relative
            original = path.read_bytes()
            self.assertIn(old, original)
            path.write_bytes(original.replace(old, new))
            with self.subTest(relative=relative, old=old), self.assertRaisesRegex(ValueError, 'structure changed'):
                build_snapshot(self.config, self.file, require_review=False)
            path.write_bytes(original)

    def test_malformed_selected_fields_and_unknown_controls_block_full_audit(self):
        path = self.root / 'source/data/System.json'
        original = path.read_bytes()
        for old, new in ((b'"Rank"', b'123'), (b'"Rank"', b'"\\\\Unknown[1]"'),
                         (b'"Rank"', b'"\\\\pop[0]Label"'), (b'"basic":[', b'"basic":{"invalid":')):
            raw = original.replace(old, new, 1)
            for name in ('source', 'target'):
                (self.root / name / 'data/System.json').write_bytes(raw)
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'coverage is blocked'):
                build_snapshot(self.config, self.file, require_review=False)

    def test_selection_change_invalidates_review_snapshot_and_old_material_version(self):
        snapshot = self.snapshot()
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms']
        with self.assertRaisesRegex(ValueError, 'review'):
            build_snapshot(self.config, self.file)
        narrowed = RpgMakerMvAdapter(snapshot.adapter.support_window, display_fields=['system_terms'])
        with self.assertRaisesRegex(ValueError, 'Wrong project'):
            narrowed.inventory_candidates(snapshot.project, InventoryPolicy())
        current = material_identity(snapshot)
        old = {**current, 'adapter_version': '0.2.0'}
        old.pop('display_fields')
        old.pop('adapter_behavior_digest')
        with self.assertRaisesRegex(ValueError, 'adapter_version'):
            require_material_fresh(old, current)

    def test_config_rejects_malformed_unknown_and_duplicate_opt_ins(self):
        for value in (None, '', {}, True, ['database'], ['system_terms', 'system_terms'], [False]):
            self.config['rpgmaker_mv']['display_fields'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'display_fields'):
                read_mv_project_config(self.config, self.file)


if __name__ == '__main__':
    unittest.main()
