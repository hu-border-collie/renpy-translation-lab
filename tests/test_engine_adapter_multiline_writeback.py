"""Cross-line Ren'Py writeback plan tests (issue #471)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest, ValidatedTranslation
from engine_adapters.coverage import digest_json
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
from engine_adapters.writeback import WritebackPlanError, render_writeback_plan
from sync_translation_preview import _deserialize_writeback_plan


class RenPyMultilineWritebackTests(unittest.TestCase):
    def snapshot_for(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        target = tl_dir / "script.rpy"
        target.write_text(text, encoding="utf-8")
        request = ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(tl_dir),
            target_language="schinese",
        )
        adapter = RenPyAdapter(legacy_module=runtime)
        snapshot = build_translation_snapshot(adapter, request)
        return root, tl_dir, target, adapter, request, snapshot

    @staticmethod
    def occurrence_for(snapshot, text: str):
        return next(
            occurrence for occurrence in snapshot.occurrences if occurrence.unit.text == text
        )

    def plan_for(self, source: str, source_text: str, translated_text: str):
        _root, _tl_dir, _target, adapter, _request, snapshot = self.snapshot_for(source)
        occurrence = self.occurrence_for(snapshot, source_text)
        validation = adapter.validate_translation(occurrence, translated_text)
        self.assertEqual(validation.status, "pass", validation.reason_codes)
        plan = adapter.build_writeback_plan(
            snapshot.project,
            (ValidatedTranslation(occurrence, translated_text, validation),),
            snapshot.project.source_documents,
        )
        return adapter, snapshot, plan

    def test_triple_quoted_plan_renders_multiline_operation(self):
        source = 'e """Hello\nworld."""\n'
        _adapter, snapshot, plan = self.plan_for(source, "Hello\nworld.", "你好\n世界")

        operation = plan.operations[0]
        self.assertEqual(operation.kind, "multiline_text_span_replace")
        self.assertEqual(operation.line, 0)
        self.assertEqual(operation.end_line, 1)
        self.assertEqual(operation.start_col, 2)
        self.assertEqual(operation.end_col, 9)
        self.assertEqual(operation.expected_text_digest, hashlib.sha256(b"Hello\nworld.").hexdigest())
        self.assertEqual(ast_literal(operation.replacement_fragment), "你好\n世界")

        rendered = render_writeback_plan(plan, snapshot.project.source_documents)
        rendered_text = "".join(rendered["script.rpy"]).replace("\r\n", "\n")
        self.assertEqual(rendered_text, "e " + operation.replacement_fragment + "\n")

    def test_backslash_continued_plan_renders_valid_single_line_literal(self):
        source = 'e "Hello \\\nworld."\n'
        _adapter, snapshot, plan = self.plan_for(source, "Hello world.", "你好世界")

        operation = plan.operations[0]
        self.assertEqual(operation.kind, "multiline_text_span_replace")
        self.assertEqual(operation.end_line, 1)
        self.assertEqual(ast_literal(operation.replacement_fragment), "你好世界")

        rendered = render_writeback_plan(plan, snapshot.project.source_documents)
        self.assertEqual(
            "".join(rendered["script.rpy"]).replace("\r\n", "\n"),
            'e "你好世界"\n',
        )

    def test_plan_round_trips_through_sync_preview_deserializer(self):
        source = 'e """Hello\nworld."""\n'
        _adapter, snapshot, plan = self.plan_for(source, "Hello\nworld.", "你好\n世界")

        parsed = _deserialize_writeback_plan(plan.to_dict())
        self.assertEqual(parsed.operations[0].kind, "multiline_text_span_replace")
        self.assertEqual(parsed.operations[0].end_line, 1)
        rendered = render_writeback_plan(parsed, snapshot.project.source_documents)
        self.assertEqual(
            "".join(rendered["script.rpy"]).replace("\r\n", "\n"),
            "e " + plan.operations[0].replacement_fragment + "\n",
        )

    def _tampered_plan(self, plan, index: int, **changes):
        operation = replace(plan.operations[index], **changes)
        operation_payload = operation.to_dict()
        operation_payload.pop("operation_id")
        operation = replace(operation, operation_id="op1:" + digest_json(operation_payload))
        operations = list(plan.operations)
        operations[index] = operation
        tampered = replace(plan, operations=tuple(operations))
        plan_payload = tampered.to_dict()
        plan_payload.pop("plan_digest")
        return replace(tampered, plan_digest=digest_json(plan_payload))

    def test_common_consumer_rejects_multiline_end_line_out_of_range(self):
        source = 'e """Hello\nworld."""\n'
        _adapter, snapshot, plan = self.plan_for(source, "Hello\nworld.", "你好\n世界")
        tampered = self._tampered_plan(plan, 0, end_line=99)

        with self.assertRaises(WritebackPlanError) as context:
            render_writeback_plan(tampered, snapshot.project.source_documents)
        self.assertEqual(context.exception.reason_code, "common.writeback.span_invalid")

    def test_common_consumer_rejects_multiline_fragment_drift(self):
        source = 'e """Hello\nworld."""\n'
        _adapter, snapshot, plan = self.plan_for(source, "Hello\nworld.", "你好\n世界")
        tampered = self._tampered_plan(plan, 0, expected_fragment_sha256="0" * 64)

        with self.assertRaises(WritebackPlanError) as context:
            render_writeback_plan(tampered, snapshot.project.source_documents)
        self.assertEqual(context.exception.reason_code, "common.writeback.span_mismatch")

    def test_common_consumer_rejects_overlapping_multiline_spans(self):
        source = (
            'e """Alpha\n'
            'beta."""\n'
            'e """Gamma\n'
            'delta."""\n'
        )
        _root, _tl_dir, _target, adapter, _request, snapshot = self.snapshot_for(source)
        occurrences = list(snapshot.occurrences)
        self.assertEqual(len(occurrences), 2)
        validated = tuple(
            ValidatedTranslation(
                occurrence,
                "一\n二" if index == 0 else "三\n四",
                adapter.validate_translation(occurrence, "一\n二" if index == 0 else "三\n四"),
            )
            for index, occurrence in enumerate(occurrences)
        )
        plan = adapter.build_writeback_plan(
            snapshot.project,
            validated,
            snapshot.project.source_documents,
        )
        first = plan.operations[0]
        tampered = self._tampered_plan(
            plan,
            1,
            line=first.line,
            start_col=first.start_col,
            end_line=first.end_line,
            end_col=first.end_col,
            expected_fragment_sha256=first.expected_fragment_sha256,
        )

        with self.assertRaises(WritebackPlanError) as context:
            render_writeback_plan(tampered, snapshot.project.source_documents)
        self.assertEqual(context.exception.reason_code, "common.writeback.span_overlap")


def ast_literal(fragment: str) -> str:
    import ast

    value = ast.literal_eval(fragment)
    if not isinstance(value, str):
        raise AssertionError(f"not a string literal: {fragment!r}")
    return value


if __name__ == "__main__":
    unittest.main()
