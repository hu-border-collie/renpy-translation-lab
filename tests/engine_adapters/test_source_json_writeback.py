"""Array-aware source-copy rendering and legacy plan compatibility (P1)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from engine_adapters.contracts import (
    SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION,
    SourceDocument,
    WritebackOperation,
    WritebackPlan,
)
from engine_adapters.coverage import digest_json
from engine_adapters.writeback import (
    WritebackPlanError,
    render_writeback_plan,
    source_snapshot_fingerprint,
    validate_writeback_plan,
)
from execution.sync_translation_preview import _deserialize_writeback_plan
from foundation import atomic_io


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "source_json_writeback"


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document(content, *, relative="Map001.json", side="source"):
    return SourceDocument(
        file_rel_path=relative,
        file_path=f"{side}/{relative}",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def value_at(doc, path):
    value = json.loads(doc.content.decode("utf-8-sig"))
    for part in path:
        value = value[part]
    return value


def sign_operation(operation, **changes):
    operation = replace(operation, **changes)
    payload = operation.to_dict()
    payload.pop("operation_id")
    prefix = "op2:" if operation.kind == "json_source_string_set" else "op1:"
    return replace(operation, operation_id=prefix + digest_json(payload))


def operation_for(source, target, path, translation, identifier="sample"):
    return sign_operation(WritebackOperation(
        operation_id="",
        kind="json_source_string_set",
        occurrence_id=f"fixture:{identifier}",
        target_root="source_copy",
        target_rel_path=target.file_rel_path,
        expected_file_sha256=target.sha256,
        line=-1,
        start_col=-1,
        end_col=-1,
        expected_fragment_sha256=text_sha(value_at(target, path)),
        expected_text_digest=text_sha(value_at(source, path)),
        replacement_fragment=translation,
        validation_digest="original-fixture-fixed-text",
        target_json_path=tuple(path),
    ))


def sign_plan(plan, **changes):
    plan = replace(plan, **changes)
    payload = plan.to_dict()
    payload.pop("plan_digest")
    return replace(plan, plan_digest=digest_json(payload))


def plan_for(sources, targets, operations):
    return sign_plan(WritebackPlan(
        engine="original-json-fixture",
        adapter_version="p1-contract-only",
        project_identity_digest="original-fixture-project",
        source_snapshot_fingerprint=source_snapshot_fingerprint(sources),
        coverage_digest="original-fixture-position-list",
        coverage_review_digest="fixed-literal-review",
        operations=tuple(operations),
        plan_digest="",
        writeback_plan_schema_version=SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION,
        target_snapshot_fingerprint=source_snapshot_fingerprint(targets),
    ))


class SourceJsonWritebackTests(unittest.TestCase):
    def setUp(self):
        self.raw = (FIXTURE / "source" / "Map001.json").read_bytes()
        self.expected = (FIXTURE / "expected" / "Map001.json").read_bytes()
        self.positions = json.loads((FIXTURE / "positions.json").read_text(encoding="utf-8"))

    def fixture_plan(self, *, source_bytes=None, target_bytes=None):
        source = document(self.raw if source_bytes is None else source_bytes)
        target = document(source.content if target_bytes is None else target_bytes, side="target")
        operations = []
        for row in self.positions:
            self.assertEqual(value_at(source, row["path"]), row["source"])
            operations.append(operation_for(source, target, row["path"], row["translation"], row["id"]))
        return source, target, plan_for((source,), (target,), operations)

    def rendered(self, plan, source, target):
        return "".join(render_writeback_plan(plan, (source,), live_targets=(target,))[target.file_rel_path]).encode("utf-8")

    def assert_rejected(self, plan, source, target, reason):
        with self.assertRaises(WritebackPlanError) as error:
            self.rendered(plan, source, target)
        self.assertEqual(error.exception.reason_code, reason)

    def test_fixture_writes_five_independent_positions_and_preserves_every_other_byte(self):
        source, target, plan = self.fixture_plan()
        self.assertEqual(len(validate_writeback_plan(plan, (source,), live_targets=(target,))), 5)
        result = self.rendered(plan, source, target)
        self.assertEqual(result, self.expected)
        self.assertEqual(source.content, self.raw)
        self.assertEqual(target.content, self.raw)
        self.assertEqual(len(plan.operations), 5)
        self.assertEqual(len({op.operation_id for op in plan.operations}), 5)
        self.assertIn(b'1.234567890123456789, "scale":1e+02', result)
        translated = document(result)
        for row in self.positions:
            self.assertEqual(value_at(translated, row["path"]), row["translation"])
        self.assertEqual(value_at(translated, ("events", 1, "pages", 0, "list", 3, "parameters", 0)), "")

    def test_bom_newline_and_missing_final_newline_survive(self):
        base = self.raw.replace(b"\r\n", b"\n")
        expected = self.expected.replace(b"\r\n", b"\n")
        for bom in (b"", b"\xef\xbb\xbf"):
            for newline in (b"\n", b"\r\n", b"\r"):
                for final in (True, False):
                    with self.subTest(bom=bool(bom), newline=newline, final=final):
                        src = base if final else base.rstrip(b"\n")
                        dst = expected if final else expected.rstrip(b"\n")
                        src = bom + src.replace(b"\n", newline)
                        dst = bom + dst.replace(b"\n", newline)
                        source, target, plan = self.fixture_plan(source_bytes=src)
                        self.assertEqual(self.rendered(plan, source, target), dst)

    def test_unchanged_decoded_string_keeps_original_escape_spelling(self):
        raw = b' { "text" : "\\u0048ello\\nworld", "number":1e+02 }'
        source, target = document(raw), document(raw, side="target")
        op = operation_for(source, target, ("text",), "Hello\nworld")
        plan = plan_for((source,), (target,), (op,))
        self.assertEqual(self.rendered(plan, source, target), raw)

    def test_nested_escaped_keys_and_string_contents_use_value_token_spans(self):
        raw = b'{"quo\\\"te":["a\\\\b",{"\\u4e2d":"old \\\"quote\\\""}],"n":1e+02}'
        source, target = document(raw), document(raw, side="target")
        translation = '中文 "引号" \\路径\n第二行'
        op = operation_for(source, target, ('quo"te', 1, "中"), translation)
        result = self.rendered(plan_for((source,), (target,), (op,)), source, target)
        self.assertEqual(value_at(document(result), ('quo"te', 1, "中")), translation)
        self.assertIn(b'"a\\\\b"', result)
        self.assertTrue(result.endswith(b',"n":1e+02}'))

    def test_root_array_is_supported_and_numeric_object_key_stays_a_string(self):
        for raw, path in ((b'[null,{"text":"old"}]', (1, "text")),
                          (b'{"0":{"text":"old"}}', ("0", "text"))):
            with self.subTest(path=path):
                source, target = document(raw), document(raw, side="target")
                op = operation_for(source, target, path, "新值")
                result = self.rendered(plan_for((source,), (target,), (op,)), source, target)
                self.assertEqual(value_at(document(result), path), "新值")

    def test_source_and_current_translation_are_separate_snapshots(self):
        source, target, plan = self.fixture_plan(target_bytes=self.expected)
        old = plan.operations[0]
        self.assertNotEqual(old.expected_text_digest, old.expected_fragment_sha256)
        op = sign_operation(old, replacement_fragment=r"\pop[1]这是第二次订正。")
        plan = sign_plan(plan, operations=(op, *plan.operations[1:]))
        result = self.rendered(plan, source, target)
        self.assertEqual(value_at(document(result), op.target_json_path), op.replacement_fragment)
        self.assertEqual(source.content, self.raw)
        wrong_source = sign_operation(op, expected_text_digest=old.expected_fragment_sha256)
        self.assert_rejected(sign_plan(plan, operations=(wrong_source,)), source, target,
                             "common.writeback.source_value_mismatch")

    def test_persisted_schema2_roundtrip_preserves_integer_paths_and_both_snapshots(self):
        source, target, plan = self.fixture_plan()
        payload = json.loads(json.dumps(plan.to_dict(), ensure_ascii=False))
        restored = _deserialize_writeback_plan(payload)
        self.assertEqual(restored.to_dict(), payload)
        self.assertIs(type(restored.operations[0].target_json_path[1]), int)
        self.assertEqual(restored.target_snapshot_fingerprint, plan.target_snapshot_fingerprint)
        self.assertEqual(self.rendered(restored, source, target), self.expected)

    def test_schema2_cannot_use_legacy_single_snapshot_rendering(self):
        source, _target, plan = self.fixture_plan()
        with self.assertRaises(WritebackPlanError) as error:
            render_writeback_plan(plan, (source,))
        self.assertEqual(error.exception.reason_code, "common.writeback.target_snapshot_required")

    def test_stale_source_or_target_rejects_even_if_only_unselected_data_changed(self):
        source, target, plan = self.fixture_plan()
        changed = self.raw.replace(b'"x":3', b'"x":4')
        self.assert_rejected(plan, document(changed), target, "common.writeback.source_snapshot_mismatch")
        self.assert_rejected(plan, source, document(changed, side="target"), "common.writeback.target_snapshot_mismatch")

    def test_file_and_value_guards_survive_rebinding_outer_snapshot(self):
        source, target, plan = self.fixture_plan()
        changed = document(self.raw.replace(b"lantern is lit", b"lantern is dark"), side="target")
        rebound = sign_plan(plan, target_snapshot_fingerprint=source_snapshot_fingerprint((changed,)))
        self.assert_rejected(rebound, source, changed, "common.writeback.source_snapshot_mismatch")
        op = sign_operation(plan.operations[0], expected_file_sha256=changed.sha256)
        self.assert_rejected(sign_plan(rebound, operations=(op,)), source, changed,
                             "common.writeback.target_value_mismatch")
        source_op = sign_operation(plan.operations[0], expected_text_digest=text_sha("not the source"))
        self.assert_rejected(sign_plan(plan, operations=(source_op,)), source, target,
                             "common.writeback.source_value_mismatch")

    def test_document_metadata_cannot_hide_changed_content(self):
        source, target, plan = self.fixture_plan()
        changed = replace(target, content=target.content.replace(b'"x":3', b'"x":4'))
        self.assert_rejected(plan, source, changed, "common.writeback.source_snapshot_mismatch")

    def test_missing_frozen_source_and_duplicate_document_are_rejected(self):
        source, target, plan = self.fixture_plan()
        missing = sign_plan(plan, source_snapshot_fingerprint=source_snapshot_fingerprint(()))
        with self.assertRaises(WritebackPlanError) as error:
            render_writeback_plan(missing, (), live_targets=(target,))
        self.assertEqual(error.exception.reason_code, "common.writeback.source_missing")
        duplicate = sign_plan(plan, target_snapshot_fingerprint=source_snapshot_fingerprint((target, target)))
        with self.assertRaises(WritebackPlanError) as error:
            render_writeback_plan(duplicate, (source,), live_targets=(target, target))
        self.assertEqual(error.exception.reason_code, "common.writeback.plan_invalid")

    def test_malformed_path_components_are_rejected_without_coercion(self):
        source, target, plan = self.fixture_plan()
        for part in (True, False, -1, 1.0, None, ""):
            with self.subTest(part=part):
                path = ("events", part, *plan.operations[0].target_json_path[2:])
                op = sign_operation(plan.operations[0], target_json_path=path)
                self.assert_rejected(sign_plan(plan, operations=(op,)), source, target, "common.writeback.plan_invalid")
                payload = plan.to_dict()
                payload["operations"][0]["target_json_path"] = list(path)
                with self.assertRaises(ValueError):
                    _deserialize_writeback_plan(payload)

    def test_missing_wrongly_typed_and_nonstring_leaves_are_rejected(self):
        source, target, plan = self.fixture_plan()
        for path, reason in (
            (("events", "1"), "source_json_path_missing"),
            (("events", 500), "source_json_path_missing"),
            (("missing",), "source_json_path_missing"),
            (("events", 0, "text"), "source_json_path_missing"),
            (("events", 0), "source_json_value_invalid"),
            (("events", 1, "x"), "source_json_value_invalid"),
            (("events", 1, "pages"), "source_json_value_invalid"),
            (("opaque", "enabled"), "source_json_value_invalid"),
        ):
            with self.subTest(path=path):
                op = sign_operation(plan.operations[0], target_json_path=path)
                self.assert_rejected(sign_plan(plan, operations=(op,)), source, target, f"common.writeback.{reason}")

    def test_empty_source_stays_structural_and_empty_target_needs_explicit_binding(self):
        source, target, plan = self.fixture_plan()
        path = ("events", 1, "pages", 0, "list", 3, "parameters", 0)
        op = operation_for(source, target, path, "should not add text")
        self.assert_rejected(sign_plan(plan, operations=(op,)), source, target,
                             "common.writeback.source_json_value_invalid")
        src = document(b'{"text":"original"}')
        dst = document(b'{"text":""}', side="target")
        op = operation_for(src, dst, ("text",), "译文")
        result = self.rendered(plan_for((src,), (dst,), (op,)), src, dst)
        self.assertEqual(value_at(document(result), ("text",)), "译文")

    def test_blank_and_nonstring_replacements_are_rejected(self):
        source, target, plan = self.fixture_plan()
        for value in ("", " \n\t", None, 5, False):
            with self.subTest(value=value):
                op = sign_operation(plan.operations[0], replacement_fragment=value)
                self.assert_rejected(sign_plan(plan, operations=(op,)), source, target,
                                     "common.writeback.replacement_invalid")

    def test_duplicate_targets_noncopy_roots_and_text_coordinates_are_rejected(self):
        source, target, plan = self.fixture_plan()
        op = sign_operation(plan.operations[0], replacement_fragment="another translation")
        self.assert_rejected(sign_plan(plan, operations=(*plan.operations, op)), source, target,
                             "common.writeback.target_duplicate")
        for change, reason in (({"target_root":"localization_catalog"}, "target_root_unsupported"),
                               ({"target_rel_path":"../Map001.json"}, "path_escape"),
                               ({"line":0}, "plan_invalid"), ({"end_line":0}, "plan_invalid")):
            with self.subTest(change=change):
                op = sign_operation(plan.operations[0], **change)
                self.assert_rejected(sign_plan(plan, operations=(op,)), source, target, f"common.writeback.{reason}")

    def test_plan_and_operation_digest_tampering_are_rejected(self):
        source, target, plan = self.fixture_plan()
        self.assert_rejected(replace(plan, coverage_digest="edited"), source, target,
                             "common.writeback.plan_digest_mismatch")
        op = replace(plan.operations[0], replacement_fragment="edited")
        self.assert_rejected(sign_plan(plan, operations=(op,)), source, target,
                             "common.writeback.plan_digest_mismatch")

    def test_invalid_duplicate_member_and_non_json_constants_are_rejected(self):
        source, target, plan = self.fixture_plan()
        bad_values = (b'{broken', self.raw.rstrip(b"\r\n")[:-1],
                      self.raw.replace(b'"displayName" :', b'"extra":1,"extra":2,"displayName" :'),
                      self.raw.replace(b'"ratio":1.234567890123456789', b'"ratio":NaN'))
        for raw in bad_values:
            for side in ("source", "target"):
                with self.subTest(raw=raw[:25], side=side):
                    changed = document(raw, side=side)
                    sources, targets = ((changed,), (target,)) if side == "source" else ((source,), (changed,))
                    ops = plan.operations if side == "source" else tuple(sign_operation(op, expected_file_sha256=changed.sha256) for op in plan.operations)
                    rebound = plan_for(sources, targets, ops)
                    self.assert_rejected(rebound, sources[0], targets[0], "common.writeback.source_json_invalid")

    def test_unknown_or_noninteger_plan_versions_and_missing_target_binding_are_rejected(self):
        source, target, plan = self.fixture_plan()
        for schema in (0, 3, True, 2.0, "2"):
            with self.subTest(schema=schema):
                payload = plan.to_dict()
                payload["writeback_plan_schema_version"] = schema
                with self.assertRaises(ValueError):
                    _deserialize_writeback_plan(payload)
                self.assert_rejected(replace(plan, writeback_plan_schema_version=schema), source, target,
                                     "common.writeback.schema_unsupported")
        for binding in ("", None, 2):
            payload = plan.to_dict()
            payload["target_snapshot_fingerprint"] = binding
            with self.assertRaises(ValueError):
                _deserialize_writeback_plan(payload)

    def test_legacy_catalog_payload_digest_and_rendering_remain_unchanged(self):
        doc = document(b'{"messages":{"hello":"Hello"}}', relative="catalog.json")
        op = sign_operation(WritebackOperation(
            operation_id="", kind="json_catalog_set", occurrence_id="legacy:hello",
            target_root="localization_catalog", target_rel_path=doc.file_rel_path,
            expected_file_sha256=doc.sha256, line=-1, start_col=-1, end_col=-1,
            expected_fragment_sha256=text_sha("Hello"), expected_text_digest=text_sha("Hello"),
            replacement_fragment="你好", validation_digest="legacy-validation",
            target_json_path=("messages", "hello"),
        ))
        plan = sign_plan(WritebackPlan(
            engine="tyrano", adapter_version="0.2.0", project_identity_digest="legacy-project",
            source_snapshot_fingerprint=source_snapshot_fingerprint((doc,)),
            coverage_digest="legacy-coverage", coverage_review_digest="legacy-review",
            operations=(op,), plan_digest="",
        ))
        payload = plan.to_dict()
        self.assertEqual(payload["writeback_plan_schema_version"], 1)
        self.assertNotIn("target_snapshot_fingerprint", payload)
        restored = _deserialize_writeback_plan(json.loads(json.dumps(payload)))
        self.assertEqual(restored.to_dict(), payload)
        for legacy_version in ("1", 1.0, True):
            legacy_payload = {**payload, "writeback_plan_schema_version":legacy_version}
            self.assertEqual(_deserialize_writeback_plan(legacy_payload).to_dict(), payload)
        self.assertEqual(json.loads("".join(render_writeback_plan(restored, (doc,))[doc.file_rel_path])),
                         {"messages":{"hello":"你好"}})
        malformed = plan.to_dict()
        malformed["operations"][0]["target_json_path"] = ["messages", 0]
        with self.assertRaises(ValueError):
            _deserialize_writeback_plan(malformed)
        v2 = plan_for((doc,), (doc,), (op,))
        self.assert_rejected(v2, doc, doc, "common.writeback.operation_unsupported")

    def test_schema2_operation_cannot_be_downgraded_to_schema1(self):
        source, target, plan = self.fixture_plan()
        legacy = sign_plan(plan, writeback_plan_schema_version=1, target_snapshot_fingerprint="")
        with self.assertRaises(WritebackPlanError) as error:
            render_writeback_plan(legacy, (source,))
        self.assertEqual(error.exception.reason_code, "common.writeback.operation_unsupported")

    def transaction_setup(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source_root, target_root = root / "original", root / "copy"
        source_root.mkdir()
        target_root.mkdir()
        meta = b' {"caption":"Second lamp", "n":1e+02}\r\n'
        sources, targets = [], []
        for relative, raw in (("Map001.json", self.raw), ("meta.json", meta)):
            source_path, target_path = source_root / relative, target_root / relative
            source_path.write_bytes(raw)
            target_path.write_bytes(raw)
            sources.append(replace(document(raw, relative=relative), file_path=str(source_path)))
            targets.append(replace(document(raw, relative=relative, side="target"), file_path=str(target_path)))
        operations = [operation_for(sources[0], targets[0], row["path"], row["translation"], row["id"]) for row in self.positions]
        operations.append(operation_for(sources[1], targets[1], ("caption",), "另一盏灯", "second-file"))
        plan = _deserialize_writeback_plan(json.loads(json.dumps(plan_for(sources, targets, operations).to_dict())))
        rendered = render_writeback_plan(plan, sources, live_targets=targets)
        writes = [(target_root / relative, lines) for relative, lines in rendered.items()]
        expected_preimages = {Path(doc.file_path):doc.sha256 for doc in targets}
        return source_root, target_root, sources, targets, writes, expected_preimages

    def test_persisted_plan_commits_only_to_copy_through_existing_atomic_writer(self):
        source_root, target_root, _sources, _targets, writes, guards = self.transaction_setup()
        journal = target_root / "transaction.json"
        atomic_io.atomic_write_many_lines(writes, journal_path=journal, expected_preimages=guards)
        self.assertEqual((target_root / "Map001.json").read_bytes(), self.expected)
        self.assertEqual((source_root / "Map001.json").read_bytes(), self.raw)
        self.assertEqual(
            json.loads((target_root / "meta.json").read_text(encoding="utf-8")),
            {"caption": "另一盏灯", "n": 100},
        )
        self.assertFalse(journal.exists())

    def test_late_target_edit_is_rejected_by_existing_atomic_preimage_guard(self):
        _source_root, target_root, _sources, _targets, writes, guards = self.transaction_setup()
        target = target_root / "Map001.json"
        edited = self.raw.replace(b'"x":3', b'"x":4')
        target.write_bytes(edited)
        with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
            atomic_io.atomic_write_many_lines(writes, journal_path=target_root / "transaction.json", expected_preimages=guards)
        self.assertEqual(target.read_bytes(), edited)

    def test_failed_second_replace_restores_first_file_bytes_and_allows_retry(self):
        source_root, target_root, _sources, targets, writes, guards = self.transaction_setup()
        second = target_root / "meta.json"
        journal = target_root / "transaction.json"
        real_replace = os.replace
        failed = False

        def fail_second(source, destination):
            nonlocal failed
            if not failed and Path(destination) == second and str(source).endswith(".txn.tmp"):
                failed = True
                raise OSError("fixture second replace failure")
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "fixture second replace failure"):
                atomic_io.atomic_write_many_lines(writes, journal_path=journal, expected_preimages=guards)
        for doc in targets:
            self.assertEqual(Path(doc.file_path).read_bytes(), doc.content)
        self.assertEqual((source_root / "Map001.json").read_bytes(), self.raw)
        self.assertFalse(journal.exists())
        atomic_io.atomic_write_many_lines(writes, journal_path=journal, expected_preimages=guards)
        self.assertEqual((target_root / "Map001.json").read_bytes(), self.expected)

    def test_crash_journal_recovers_original_bytes_without_retranslating(self):
        source_root, target_root, _sources, targets, writes, guards = self.transaction_setup()
        second = target_root / "meta.json"
        journal = target_root / "transaction.json"
        real_replace = os.replace

        def crash_second(source, destination):
            if Path(destination) == second and str(source).endswith(".txn.tmp"):
                raise SystemExit("simulated process exit")
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, "replace", side_effect=crash_second):
            with self.assertRaises(SystemExit):
                atomic_io.atomic_write_many_lines(writes, journal_path=journal, expected_preimages=guards)
        self.assertTrue(journal.exists())
        self.assertEqual((target_root / "Map001.json").read_bytes(), self.expected)
        self.assertTrue(atomic_io.recover_atomic_write_transaction(journal, expected_transaction_kind="apply"))
        for doc in targets:
            self.assertEqual(Path(doc.file_path).read_bytes(), doc.content)
        self.assertEqual((source_root / "Map001.json").read_bytes(), self.raw)


if __name__ == "__main__":
    unittest.main()
