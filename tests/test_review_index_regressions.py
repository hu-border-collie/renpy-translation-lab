"""Production-bound regressions for the #427 S1 review contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import review_index as ri
import revision_corpus
import translation_quality
from tests.test_review_index import _write_corpus, _write_findings


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_decision(entry, **changes):
    return {
        "occurrence_id": entry["occurrence_id"],
        "project_identity_digest": entry["project"]["identity_digest"],
        "lifecycle": "resolved",
        "reviewer": {"type": "human", "name": "regression-reviewer"},
        "binding": dict(entry["binding"]),
        "decided_at": "2026-09-21T00:00:00+00:00",
        **changes,
    }


def package_bytes(package):
    return {path.name: path.read_bytes() for path in package.iterdir() if path.is_file()}


class DecisionLogIsolationTests(unittest.TestCase):
    def test_other_project_default_log_blocks_rebuild_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_a = _write_corpus(root / "a", project_slug="project-a")
            corpus_b = _write_corpus(root / "b", project_slug="project-b")
            package = root / "index"
            ri.build_review_index(corpus_a, output_dir=package)
            incoming = root / "incoming.jsonl"
            write_rows(incoming, [make_decision(ri.load_review_index(package)[1][0])])
            ri.import_decisions_into_index(package, incoming)
            before = package_bytes(package)
            with self.assertRaises(ri.ReviewIndexError) as error:
                ri.build_review_index(corpus_b, output_dir=package)
            self.assertEqual(error.exception.code, ri.DIAGNOSTIC_DECISION_PROJECT_MISMATCH)
            self.assertEqual(package_bytes(package), before)

    def test_existing_foreign_log_blocks_import_including_orphans(self):
        for external, orphan in ((False, False), (False, True), (True, True)):
            with (
                self.subTest(external=external, orphan=orphan),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                package = root / "index"
                ri.build_review_index(_write_corpus(root), output_dir=package)
                entry = ri.load_review_index(package)[1][0]
                incoming = root / "incoming.jsonl"
                write_rows(incoming, [make_decision(entry)])
                log = (
                    root / "external.jsonl"
                    if external
                    else package / ri.REVIEW_DECISIONS_JSONL_NAME
                )
                foreign = make_decision(
                    entry,
                    project_identity_digest="another-project",
                    occurrence_id="foreign-orphan" if orphan else entry["occurrence_id"],
                )
                write_rows(log, [foreign])
                if external:
                    manifest_path = package / ri.REVIEW_INDEX_MANIFEST_NAME
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["inputs"]["decisions"]["path"] = str(log)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                before = package_bytes(package)
                log_before = log.read_bytes()
                with self.assertRaises(ri.ReviewIndexError) as error:
                    ri.import_decisions_into_index(package, incoming)
                self.assertEqual(error.exception.code, ri.DIAGNOSTIC_DECISION_PROJECT_MISMATCH)
                self.assertEqual(package_bytes(package), before)
                self.assertEqual(log.read_bytes(), log_before)

    def test_explicit_decisions_cannot_hide_foreign_package_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            entry = ri.load_review_index(package)[1][0]
            external = root / "valid.jsonl"
            write_rows(external, [make_decision(entry)])
            write_rows(
                package / ri.REVIEW_DECISIONS_JSONL_NAME,
                [
                    make_decision(entry, project_identity_digest="foreign-project"),
                ],
            )
            before = package_bytes(package)
            with self.assertRaises(ri.ReviewIndexError):
                ri.build_review_index(corpus, decisions_path=external, output_dir=package)
            self.assertEqual(package_bytes(package), before)


class FindingBindingTests(unittest.TestCase):
    def test_old_id_only_binding_needs_recheck_only_when_findings_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = ri.load_corpus_bundle(_write_corpus(root))
            findings = ri.load_quality_findings(_write_findings(root))["findings"]
            entries, _ = ri.build_index_entries(
                corpus["rows"], corpus["manifest"], findings=findings
            )
            decisions = []
            for entry in entries:
                binding = dict(entry["binding"])
                binding["evidence_digest"] = ri._digest_payload(
                    {
                        "source_digest": binding["source_digest"],
                        "target_digest": binding["target_digest"],
                        "context_digest": binding["context_digest"],
                        "finding_ids": entry["quality_finding_ids"],
                    }
                )
                binding["entry_id"] = ri._digest_payload(
                    {
                        "project_id": entry["project"]["identity_digest"],
                        "occurrence_id": entry["occurrence_id"],
                        "file_rel_path": entry["file_rel_path"],
                        "line": entry["locator"]["line"],
                        **{
                            key: binding[key]
                            for key in (
                                "source_digest",
                                "target_digest",
                                "context_digest",
                                "evidence_digest",
                            )
                        },
                    }
                )[:24]
                decisions.append(ri.normalize_decision(make_decision(entry, binding=binding)))
            applied, diagnostics = ri.apply_decisions(entries, decisions)
            self.assertEqual(diagnostics, [])
            for entry in applied:
                expected = "needs_recheck" if entry["quality_findings"] else "resolved"
                self.assertEqual(entry["review"]["lifecycle"], expected)
                self.assertEqual(len(entry["review"]["history"]), 1)

    def test_same_id_semantic_changes_require_recheck_and_preserve_history(self):
        changes = {
            "disposition": "blocker",
            "severity": "high",
            "rule_version": "2",
            "suggestion": "Use the updated terminology",
            "evidence": "new evidence with the same finding id",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            findings = _write_findings(root)
            original = ri.load_jsonl(findings)
            package = root / "index"
            ri.build_review_index(corpus, quality_findings_path=findings, output_dir=package)
            entry = ri.load_review_index(package)[1][0]
            incoming = root / "incoming.jsonl"
            write_rows(incoming, [make_decision(entry)])
            ri.import_decisions_into_index(package, incoming)
            log = package / ri.REVIEW_DECISIONS_JSONL_NAME
            before = log.read_bytes()
            for field, value in changes.items():
                with self.subTest(field=field):
                    rows = copy.deepcopy(original)
                    rows[0][field] = value
                    write_rows(findings, rows)
                    ri.build_review_index(
                        corpus, quality_findings_path=findings, output_dir=package
                    )
                    review = ri.load_review_index(package)[1][0]["review"]
                    self.assertEqual(review["lifecycle"], "needs_recheck")
                    self.assertIn("evidence_digest", review["changed_bindings"])
                    self.assertEqual(review["previous_lifecycle"], "resolved")
                    self.assertEqual(len(review["history"]), 1)
                    self.assertEqual(log.read_bytes(), before)
            write_rows(findings, original)
            ri.build_review_index(corpus, quality_findings_path=findings, output_dir=package)
            self.assertEqual(ri.load_review_index(package)[1][0]["review"]["lifecycle"], "resolved")

    def test_equivalent_evidence_order_and_volatile_metadata_keep_binding(self):
        manifest = {"project": {"slug": "demo", "tl_subdir": "schinese"}}
        rows = [
            {
                "occurrence_id": "one",
                "source": "Hello",
                "current_translation": "你好",
                "file_rel_path": "script.rpy",
                "locator": {"line": 2},
            }
        ]
        finding = translation_quality.normalize_finding(
            {
                "item_id": "one",
                "file": "script.rpy",
                "line": 2,
                "reason_code": "quality.typography.ascii_ellipsis",
                "evidence": '{"left": 1, "right": {"a": 2, "b": 3}}',
            }
        )
        first = ri.build_index_entries(rows, manifest, findings=[finding])[0][0]
        changed = {
            **finding,
            "evidence": '{"right":{"b":3,"a":2},"left":1}',
            "generated_at": "2099-01-01",
            "request_id": "unrelated-run",
        }
        second = ri.build_index_entries(rows, manifest, findings=[changed])[0][0]
        self.assertEqual(first["binding"], second["binding"])

    def test_real_policy_promotion_preserves_id_but_invalidates_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, _ = production_record_fixture(root)
            bundle = ri.load_corpus_bundle(corpus)
            row = bundle["rows"][0]
            subject = {
                "item_id": row["occurrence_id"],
                "file_rel_path": row["file_rel_path"],
                "line_number": row["locator"]["line_number"],
                "source": row["source"],
                "translation": row["current_translation"],
            }
            warning = translation_quality.check_subject(subject)
            blocker = translation_quality.check_subject(
                subject, policy={"rules": {"ascii_ellipsis": "blocker"}}
            )
            self.assertEqual(
                [item["finding_id"] for item in warning], [item["finding_id"] for item in blocker]
            )
            original, _ = ri.build_index_entries(
                bundle["rows"], bundle["manifest"], findings=warning
            )
            changed, _ = ri.build_index_entries(
                bundle["rows"], bundle["manifest"], findings=blocker
            )
            applied, _ = ri.apply_decisions(
                changed, [ri.normalize_decision(make_decision(original[0]))]
            )
            self.assertEqual(applied[0]["review"]["lifecycle"], "needs_recheck")


def production_record_fixture(root):
    """Use the actual scanner, snapshot and record/corpus exporters."""
    import gemini_translate_batch as batch
    import translator_runtime as runtime
    from engine_adapters import (
        GameVersion,
        ProjectDiscoveryRequest,
        RenPyAdapter,
        build_project_snapshot,
        build_translation_snapshot,
    )
    from engine_adapters.reuse import (
        TranslationInput,
        build_translation_records,
        export_translation_records,
    )

    tl = root / "game" / "tl" / "schinese"
    tl.mkdir(parents=True)
    script = tl / "script.rpy"
    script.write_text(
        'translate schinese start:\n    old "Hello world"\n    new "你好..."\n'
        '\n    old "Hello world"\n    new "你好..."\n',
        encoding="utf-8",
    )
    scan = build_translation_snapshot(
        RenPyAdapter(legacy_module=runtime),
        ProjectDiscoveryRequest(
            project_root=str(root), localization_root=str(tl), target_language="schinese"
        ),
    )
    snapshot = build_project_snapshot(scan, GameVersion(version_id="v1"))
    record_set = build_translation_records(
        snapshot,
        [
            TranslationInput(
                unit_id=item.unit.id,
                source_text=item.unit.source_text,
                translation_text=item.unit.current_translation,
                origin="human_confirmed",
            )
            for item in scan.occurrences
        ],
    )
    export_translation_records(record_set, root / "records")
    jobs = batch.collect_revision_file_jobs(file_paths=[("script.rpy", str(script))])
    digests = {"script.rpy": hashlib.sha256(script.read_bytes()).hexdigest()}
    corpus = root / "corpus"
    revision_corpus.export_revision_corpus(
        str(corpus),
        jobs,
        project_slug="demo",
        game_root=str(root),
        tl_dir=str(tl),
        tl_subdir="schinese",
        source_digests_before=digests,
        source_digests_after=digests,
        source_digests_scanned=digests,
    )
    return corpus, root / "records" / "translation_records.jsonl"


class TranslationRecordBindingTests(unittest.TestCase):
    @staticmethod
    def recreate_record(record, **changes):
        from engine_adapters.reuse import TranslationRecord

        fields = {
            key: copy.deepcopy(record[key])
            for key in (
                "version_id",
                "snapshot_digest",
                "occurrence_id",
                "unit_id",
                "source_text",
                "translation_text",
                "target_language",
                "origin",
                "provenance",
                "status",
                "revision_history",
            )
        }
        return TranslationRecord.create(**{**fields, **changes}).to_dict()

    def test_real_record_round_trip_keeps_distinct_occurrences_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, records = production_record_fixture(root)
            ri.build_review_index(
                corpus, translation_records_path=records, output_dir=root / "index"
            )
            manifest, entries = ri.load_review_index(root / "index")
            self.assertEqual(len(entries), 2)
            self.assertEqual(manifest["inputs"]["translation_records"]["count"], 2)
            self.assertTrue(all(item["translation_record"] is not None for item in entries))
            attached = [entry["translation_record"] for entry in entries]
            self.assertEqual(len({item["record_id"] for item in attached}), 2)
            for entry, record in zip(entries, attached):
                self.assertEqual(record["unit_id"], entry["identity_v2"])
                self.assertTrue(record["occurrence_id"].startswith("occ1:"))
                self.assertEqual(record["version_id"], "v1")
                self.assertTrue(record["snapshot_digest"])
                self.assertTrue(record["provenance"])
            self.assertEqual(manifest["diagnostics"], [])

    def test_missing_stale_and_incompatible_record_evidence_is_diagnosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, records_path = production_record_fixture(root)
            bundle = ri.load_corpus_bundle(corpus)
            original = ri.load_jsonl(records_path)[0]
            missing = copy.deepcopy(original["provenance"])
            missing.pop("source_binding")
            stale = copy.deepcopy(original["provenance"])
            stale["source_binding"]["source_digest"] = "different-project-or-version"
            wrong_line = {**original["provenance"], "line_number": 99}
            cases = [
                ({"provenance": missing}, "missing_source_binding"),
                ({"provenance": stale}, "source_snapshot_mismatch"),
                ({"provenance": wrong_line}, "locator_mismatch"),
                ({"source_text": "Different original"}, "source_mismatch"),
                ({"translation_text": "旧译文"}, "target_mismatch"),
                ({"target_language": "japanese"}, "target_language_mismatch"),
                ({"status": "superseded"}, "record_not_active"),
                ({"unit_id": "unknown-unit"}, "identity_not_found"),
            ]
            for changes, reason in cases:
                with self.subTest(reason=reason):
                    row = self.recreate_record(original, **changes)
                    # These are valid exported records, not malformed JSON stubs.
                    from engine_adapters.reuse import TranslationRecord

                    TranslationRecord.from_dict(row)
                    entries, diagnostics = ri.build_index_entries(
                        bundle["rows"], bundle["manifest"], records=[row]
                    )
                    self.assertTrue(all(entry["translation_record"] is None for entry in entries))
                    self.assertEqual(diagnostics[0]["code"], ri.DIAGNOSTIC_RECORD_UNMATCHED)
                    self.assertEqual(diagnostics[0]["reason"], reason)
            invalid = {**original, "translation_text": "tampered without new digest"}
            entries, diagnostics = ri.build_index_entries(
                bundle["rows"], bundle["manifest"], records=[invalid, {}]
            )
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(all(item["reason"] == "invalid_record" for item in diagnostics))

    def test_competing_versions_are_ambiguous_but_exact_duplicates_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, records_path = production_record_fixture(root)
            bundle = ri.load_corpus_bundle(corpus)
            original = ri.load_jsonl(records_path)[0]
            other_version = self.recreate_record(
                original, version_id="v2", snapshot_digest="other-version-snapshot"
            )
            for candidates in ([original, other_version], [other_version, original]):
                entries, diagnostics = ri.build_index_entries(
                    bundle["rows"], bundle["manifest"], records=candidates
                )
                self.assertTrue(all(entry["translation_record"] is None for entry in entries))
                self.assertEqual(diagnostics[0]["code"], ri.DIAGNOSTIC_RECORD_AMBIGUOUS)
                self.assertEqual(diagnostics[0]["reason"], "multiple_records")
            entries, diagnostics = ri.build_index_entries(
                bundle["rows"], bundle["manifest"], records=[original, original]
            )
            self.assertEqual(diagnostics, [])
            self.assertEqual(sum(entry["translation_record"] is not None for entry in entries), 1)

    def test_corpus_from_a_changed_snapshot_does_not_borrow_matching_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, records_path = production_record_fixture(root)
            bundle = ri.load_corpus_bundle(corpus)
            changed_manifest = copy.deepcopy(bundle["manifest"])
            changed_manifest["source"]["file_digests"]["script.rpy"] = "a" * 64
            changed_manifest["source"]["snapshot_digest"] = revision_corpus.aggregate_digest(
                changed_manifest["source"]["file_digests"]
            )
            entries, diagnostics = ri.build_index_entries(
                bundle["rows"], changed_manifest, records=ri.load_jsonl(records_path)
            )
            self.assertTrue(all(entry["translation_record"] is None for entry in entries))
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(
                all(item["reason"] == "source_snapshot_mismatch" for item in diagnostics)
            )


if __name__ == "__main__":
    unittest.main()
