"""Offline tests for the #427 rebuildable review index and decisions."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import review_index as ri
import revision_corpus
import translation_quality


def _write_corpus(
    root: Path,
    *,
    current_translation: str = "你好",
    game_root: Path | None = None,
    project_slug: str = "demo",
) -> Path:
    target = root / "corpus"
    file_jobs = [
        {
            "file_rel_path": "tl/chapter01.rpy",
            "items": [
                {
                    "identity_v2": "occ-1",
                    "source": "Hello",
                    "current_translation": current_translation,
                    "line": 10,
                    "line_number": 10,
                    "start": 1,
                    "end": 10,
                },
                {
                    "identity_v2": "occ-2",
                    "source": "Hello",
                    "current_translation": current_translation,
                    "line": 20,
                    "line_number": 20,
                    "start": 11,
                    "end": 20,
                },
            ],
        },
        {
            "file_rel_path": "tl/chapter02.rpy",
            "items": [
                {
                    "identity_v2": "occ-3",
                    "source": "Gate",
                    "current_translation": "门",
                    "line": 5,
                    "line_number": 5,
                    "start": 1,
                    "end": 5,
                }
            ],
        },
    ]
    revision_corpus.export_revision_corpus(
        str(target),
        file_jobs,
        project_slug=project_slug,
        game_root=str(game_root or root / "game"),
        tl_dir=str((game_root or root / "game") / "tl"),
        tl_subdir="schinese",
        source_digests_before={
            "tl/chapter01.rpy": "a" * 64,
            "tl/chapter02.rpy": "b" * 64,
        },
        source_digests_after={
            "tl/chapter01.rpy": "a" * 64,
            "tl/chapter02.rpy": "b" * 64,
        },
        source_digests_scanned={
            "tl/chapter01.rpy": "a" * 64,
            "tl/chapter02.rpy": "b" * 64,
        },
    )
    return target


def _write_findings(root: Path) -> Path:
    path = root / "quality_findings.jsonl"
    rows = [
        translation_quality.normalize_finding(
            {
                "item_id": "occ-1",
                "file": "tl/chapter01.rpy",
                "line": 10,
                "reason_code": "glossary_term_not_applied",
                "severity": "medium",
                "disposition": "warning",
                "evidence": "term missing",
                "suggestion": "apply glossary term",
            }
        ),
        translation_quality.normalize_finding(
            {
                "item_id": "occ-missing",
                "file": "tl/chapter99.rpy",
                "line": 1,
                "reason_code": "cjk_latin_spacing",
                "severity": "low",
                "disposition": "warning",
                "evidence": "unmatched",
                "suggestion": "",
            }
        ),
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


class BuildIndexTests(unittest.TestCase):
    def test_build_attaches_findings_and_reports_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            findings = _write_findings(root)
            manifest = ri.build_review_index(
                corpus,
                quality_findings_path=findings,
                output_dir=root / "index",
            )
            entries = ri.load_jsonl(root / "index" / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(manifest["scope"]["entry_count"], 3)
        self.assertEqual(manifest["scope"]["matched_finding_count"], 1)
        self.assertEqual(manifest["scope"]["unmatched_finding_count"], 1)
        by_occurrence = {entry["occurrence_id"]: entry for entry in entries}
        self.assertEqual(len(by_occurrence), 3)
        self.assertEqual(
            by_occurrence["occ-1"]["quality_finding_ids"],
            [
                translation_quality.normalize_finding(
                    {
                        "item_id": "occ-1",
                        "file": "tl/chapter01.rpy",
                        "line": 10,
                        "reason_code": "glossary_term_not_applied",
                        "severity": "medium",
                        "disposition": "warning",
                        "evidence": "term missing",
                        "suggestion": "apply glossary term",
                    }
                )["finding_id"]
            ],
        )
        self.assertEqual(by_occurrence["occ-2"]["issue_count"], 0)
        self.assertTrue(
            any(
                item["code"] == ri.DIAGNOSTIC_FINDING_UNMATCHED
                for item in manifest["diagnostics"]
            )
        )
        self.assertTrue(all(entry["review"]["lifecycle"] == "open" for entry in entries))

    def test_explicit_missing_decisions_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)

            with self.assertRaises(ri.ReviewIndexError) as captured:
                ri.build_review_index(
                    corpus,
                    decisions_path=root / "missing_decisions.jsonl",
                    output_dir=root / "index",
                )

        self.assertEqual(
            captured.exception.code,
            "REVIEW_INDEX_INPUT_MISSING",
        )

    def test_load_index_without_paths_jsonl_uses_single_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            manifest = json.loads(
                (package / ri.REVIEW_INDEX_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            manifest.pop("paths", None)
            fallback = root / "fallback"
            fallback.mkdir()
            (fallback / ri.REVIEW_INDEX_MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            (fallback / ri.REVIEW_INDEX_JSONL_NAME).write_text(
                (package / ri.REVIEW_INDEX_JSONL_NAME).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with contextlib.chdir(root):
                loaded_manifest, entries = ri.load_review_index("fallback")

        self.assertEqual(loaded_manifest["kind"], "review_index")
        self.assertEqual(len(entries), 3)

    def test_package_internal_decisions_path_is_persisted_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            (root / "index").mkdir()
            (root / "index" / "review_decisions.jsonl").write_text(
                "", encoding="utf-8"
            )
            with contextlib.chdir(root):
                manifest = ri.build_review_index(
                    "corpus",
                    decisions_path="index/review_decisions.jsonl",
                    output_dir="index",
                )

        self.assertFalse(
            os.path.isabs(manifest["inputs"]["decisions"]["path"])
        )
        self.assertEqual(
            manifest["inputs"]["decisions"]["path"],
            "review_decisions.jsonl",
        )

    def test_missing_recorded_decisions_falls_back_to_package_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            manifest_path = package / ri.REVIEW_INDEX_MANIFEST_NAME
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing_external = root / "missing_decisions.jsonl"
            payload["inputs"]["decisions"] = {
                "path": str(missing_external),
                "digest": "",
                "count": 0,
            }
            payload["paths"]["decisions"] = str(missing_external)
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
            incoming = root / "incoming.jsonl"
            incoming.write_text(
                json.dumps(
                    {
                        "occurrence_id": "occ-1",
                        "project_identity_digest": target["project"]["identity_digest"],
                        "lifecycle": "resolved",
                        "reviewer": {"type": "human", "name": "reviewer-a"},
                        "binding": dict(target["binding"]),
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            ri.import_decisions_into_index(manifest_path, incoming)
            local_exists = (package / ri.REVIEW_DECISIONS_JSONL_NAME).is_file()

        self.assertTrue(local_exists)
        self.assertFalse(missing_external.exists())

    def test_output_dir_reuse_does_not_borrow_other_project_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_a = _write_corpus(root / "a", project_slug="project-a")
            package = root / "index"
            ri.build_review_index(corpus_a, output_dir=package)
            entries_a = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(
                entry for entry in entries_a if entry["occurrence_id"] == "occ-1"
            )
            decisions_file = root / "project_a_decisions.jsonl"
            decisions_file.write_text(
                json.dumps(
                    {
                        "occurrence_id": "occ-1",
                        "project_identity_digest": target["project"]["identity_digest"],
                        "lifecycle": "resolved",
                        "reviewer": {"type": "human", "name": "reviewer-a"},
                        "binding": dict(target["binding"]),
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ri.build_review_index(
                corpus_a,
                decisions_path=decisions_file,
                output_dir=package,
            )

            corpus_b = _write_corpus(root / "b", project_slug="project-b")
            rebuilt = ri.build_review_index(corpus_b, output_dir=package)
            entries_b = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(rebuilt["inputs"]["decisions"]["path"], "")
        self.assertTrue(
            any(
                item.get("code") == ri.DIAGNOSTIC_INDEX_INPUT_PROJECT_MISMATCH
                for item in rebuilt["diagnostics"]
            )
        )
        self.assertTrue(
            all(
                entry["review"]["lifecycle"] == "open"
                for entry in entries_b
            )
        )

    def test_rebuild_is_deterministic_and_preserves_duplicate_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            first = ri.build_review_index(corpus, output_dir=root / "index-a")
            second = ri.build_review_index(corpus, output_dir=root / "index-b")
            entries_a = ri.load_jsonl(root / "index-a" / ri.REVIEW_INDEX_JSONL_NAME)
            entries_b = ri.load_jsonl(root / "index-b" / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(first["scope"]["entry_count"], 3)
        self.assertEqual(
            [entry["entry_id"] for entry in entries_a],
            [entry["entry_id"] for entry in entries_b],
        )
        self.assertEqual(
            [entry["occurrence_id"] for entry in entries_a],
            ["occ-1", "occ-2", "occ-3"],
        )


class DecisionTests(unittest.TestCase):
    def test_project_identity_is_portable_across_game_root_move(self) -> None:
        first = ri.project_identity(
            {"project": {"slug": "demo", "tl_subdir": "schinese", "game_root": "/a"}}
        )
        second = ri.project_identity(
            {"project": {"slug": "demo", "tl_subdir": "schinese", "game_root": "/b"}}
        )
        other_project = ri.project_identity(
            {"project": {"slug": "other", "tl_subdir": "schinese", "game_root": "/a"}}
        )

        self.assertEqual(first["identity_digest"], second["identity_digest"])
        self.assertNotEqual(first["identity_digest"], other_project["identity_digest"])
        self.assertNotIn("/a", json.dumps(first, ensure_ascii=False))

    def test_ignored_decision_is_lifecycle_only(self) -> None:
        entry = {
            "occurrence_id": "occ-1",
            "quality_findings": [{"finding_id": "f1"}],
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "review": {},
        }
        decision = {
            "decision_id": "d1",
            "occurrence_id": "occ-1",
            "lifecycle": "ignored",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "note": "looked at it",
            "decided_at": "2026-09-20T00:00:00+00:00",
            "binding": dict(entry["binding"]),
        }

        applied, _ = ri.apply_decisions([entry], [decision])

        self.assertEqual(applied[0]["review"]["lifecycle"], "ignored")
        self.assertEqual(applied[0]["quality_findings"], [{"finding_id": "f1"}])
        self.assertNotIn("authorization", applied[0])

    def test_import_resolved_decision_updates_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest = ri.build_review_index(corpus, output_dir=root / "index")
            index_path = root / "index" / ri.REVIEW_INDEX_MANIFEST_NAME
            entries = ri.load_jsonl(root / "index" / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
            decision = {
                "occurrence_id": "occ-1",
                "project_identity_digest": target["project"]["identity_digest"],
                "lifecycle": "resolved",
                "reviewer": {"type": "human", "name": "reviewer-a"},
                "binding": dict(target["binding"]),
                "note": "checked",
            }
            decisions_file = root / "import.jsonl"
            decisions_file.write_text(
                json.dumps(decision, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            result = ri.import_decisions_into_index(index_path, decisions_file)
            refreshed = ri.load_jsonl(root / "index" / ri.REVIEW_INDEX_JSONL_NAME)
            markdown = (
                root / "index" / ri.REVIEW_INDEX_MARKDOWN_NAME
            ).read_text(encoding="utf-8")
            manifest_payload = json.loads(
                (root / "index" / ri.REVIEW_INDEX_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["merge"]["imported_count"], 1)
        refreshed_target = next(
            entry for entry in refreshed if entry["occurrence_id"] == "occ-1"
        )
        self.assertEqual(refreshed_target["review"]["lifecycle"], "resolved")
        self.assertEqual(
            refreshed_target["review"]["reviewer"]["name"], "reviewer-a"
        )
        self.assertFalse(refreshed_target["review"]["needs_recheck"])
        self.assertIn("[resolved]", markdown)
        self.assertEqual(manifest["scope"]["entry_count"], 3)
        recorded_decisions = manifest_payload["inputs"]["decisions"]["path"]
        self.assertFalse(os.path.isabs(recorded_decisions))
        self.assertEqual(recorded_decisions, "review_decisions.jsonl")
        self.assertEqual(
            manifest_payload["paths"]["decisions"],
            recorded_decisions,
        )

    def test_binding_change_derives_needs_recheck(self) -> None:
        entry = {
            "occurrence_id": "occ-1",
            "binding": {
                "entry_id": "old-entry",
                "snapshot_digest": "old-snapshot",
                "source_digest": "old-source",
                "target_digest": "old-target",
                "context_digest": "old-context",
                "evidence_digest": "old-evidence",
            },
            "review": {},
        }
        decision = {
            "decision_id": "d1",
            "occurrence_id": "occ-1",
            "lifecycle": "resolved",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "note": "",
            "decided_at": "2026-09-20T00:00:00+00:00",
            "binding": {
                "entry_id": "old-entry",
                "snapshot_digest": "old-snapshot",
                "source_digest": "old-source",
                "target_digest": "changed-target",
                "context_digest": "old-context",
                "evidence_digest": "old-evidence",
            },
        }
        changed_entry = {
            **entry,
            "binding": {**entry["binding"], "target_digest": "new-target"},
        }

        applied, diagnostics = ri.apply_decisions([changed_entry], [decision])

        self.assertEqual(applied[0]["review"]["lifecycle"], "needs_recheck")
        self.assertTrue(applied[0]["review"]["needs_recheck"])
        self.assertEqual(applied[0]["review"]["changed_bindings"], ["target_digest"])
        self.assertEqual(applied[0]["review"]["previous_lifecycle"], "resolved")
        self.assertEqual(diagnostics, [])

    def test_rebuild_with_changed_target_derives_needs_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_a = _write_corpus(root / "a")
            index_a = root / "index-a"
            ri.build_review_index(corpus_a, output_dir=index_a)
            entries_a = ri.load_jsonl(index_a / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries_a if entry["occurrence_id"] == "occ-1")
            decision = {
                "occurrence_id": "occ-1",
                "project_identity_digest": target["project"]["identity_digest"],
                "lifecycle": "resolved",
                "reviewer": {"type": "human", "name": "reviewer-a"},
                "binding": dict(target["binding"]),
                "note": "resolved against old target",
            }
            decisions_file = root / "decisions.jsonl"
            decisions_file.write_text(
                json.dumps(decision, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            corpus_b = _write_corpus(
                root / "b",
                current_translation="您好",
                game_root=root / "a" / "game",
            )
            index_b = root / "index-b"
            manifest_b = ri.build_review_index(
                corpus_b,
                decisions_path=decisions_file,
                output_dir=index_b,
            )
            entries_b = ri.load_jsonl(index_b / ri.REVIEW_INDEX_JSONL_NAME)
            decision_line_count = decisions_file.read_text(
                encoding="utf-8"
            ).count("\n")

        refreshed = next(
            entry for entry in entries_b if entry["occurrence_id"] == "occ-1"
        )
        self.assertEqual(refreshed["review"]["lifecycle"], "needs_recheck")
        self.assertIn("target_digest", refreshed["review"]["changed_bindings"])
        self.assertEqual(refreshed["review"]["previous_lifecycle"], "resolved")
        self.assertEqual(manifest_b["scope"]["needs_recheck_count"], 1)
        self.assertEqual(decision_line_count, 1)

    def test_matching_decision_clears_stale_previous_lifecycle(self) -> None:
        entry = {
            "occurrence_id": "occ-1",
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "review": {
                "lifecycle": "needs_recheck",
                "needs_recheck": True,
                "changed_bindings": ["target_digest"],
                "previous_lifecycle": "resolved",
            },
        }
        decision = {
            "decision_id": "d1",
            "occurrence_id": "occ-1",
            "lifecycle": "resolved",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "note": "",
            "decided_at": "2026-09-20T00:00:00+00:00",
            "binding": dict(entry["binding"]),
        }

        applied, _ = ri.apply_decisions([entry], [decision])

        self.assertEqual(applied[0]["review"]["lifecycle"], "resolved")
        self.assertFalse(applied[0]["review"]["needs_recheck"])
        self.assertEqual(applied[0]["review"]["changed_bindings"], [])
        self.assertEqual(applied[0]["review"]["previous_lifecycle"], "")

    def test_decision_id_must_match_content(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "decision_id": "tampered",
                    "occurrence_id": "occ-1",
                    "project_identity_digest": "project-a",
                    "lifecycle": "resolved",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_template_import_is_idempotent_without_decided_at(self) -> None:
        raw = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "lifecycle": "resolved",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "",
        }

        first = ri.normalize_decision(dict(raw))
        second = ri.normalize_decision(dict(raw))
        merged, summary = ri.merge_decisions([], [first, second])

        self.assertEqual(first["decision_id"], second["decision_id"])
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertEqual(len(merged), 1)

    def test_import_project_mismatch_reports_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
            decision = {
                "occurrence_id": "occ-1",
                "project_identity_digest": "other-project",
                "lifecycle": "resolved",
                "reviewer": {"type": "human", "name": "reviewer-a"},
                "binding": dict(target["binding"]),
                "note": "",
            }
            incoming = root / "incoming.jsonl"
            incoming.write_text(
                json.dumps(decision, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ri.ReviewIndexError) as captured:
                ri.import_decisions_into_index(package, incoming)
            decisions_file_exists = (
                package / ri.REVIEW_DECISIONS_JSONL_NAME
            ).exists()

        self.assertEqual(
            captured.exception.code,
            ri.DIAGNOSTIC_DECISION_PROJECT_MISMATCH,
        )
        self.assertFalse(decisions_file_exists)

    def test_manual_mismatched_decision_reports_blocked_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
            decision = {
                "occurrence_id": "occ-1",
                "project_identity_digest": "other-project",
                "lifecycle": "resolved",
                "reviewer": {"type": "human", "name": "reviewer-a"},
                "binding": dict(target["binding"]),
                "note": "",
            }
            decisions_file = root / "manual_decisions.jsonl"
            decisions_file.write_text(
                json.dumps(decision, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            manifest = ri.build_review_index(
                corpus,
                decisions_path=decisions_file,
                output_dir=root / "index-manual",
            )
            refreshed = ri.load_jsonl(root / "index-manual" / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(manifest["scope"]["project_mismatch_count"], 1)
        self.assertEqual(
            next(
                entry for entry in refreshed if entry["occurrence_id"] == "occ-1"
            )["review"]["lifecycle"],
            "open",
        )

    def test_jsonl_round_trip_keeps_unicode_line_separators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            note = "line one\u2028line two\u0085line three"
            decision = ri.normalize_decision(
                {
                    "occurrence_id": "occ-1",
                    "project_identity_digest": "project-a",
                    "lifecycle": "resolved",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "entry",
                        "snapshot_digest": "snapshot",
                        "source_digest": "source",
                        "target_digest": "target",
                        "context_digest": "context",
                        "evidence_digest": "evidence",
                    },
                    "note": note,
                }
            )
            ri.atomic_write_jsonl(path, [decision], ensure_ascii=False)
            rows = ri.load_jsonl(path, label="decisions")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["note"], note)

    def test_copied_package_reads_and_writes_its_own_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target_one = next(
                entry for entry in entries if entry["occurrence_id"] == "occ-1"
            )
            incoming_one = root / "one.jsonl"
            incoming_one.write_text(
                json.dumps(
                    {
                        "occurrence_id": "occ-1",
                        "project_identity_digest": target_one["project"]["identity_digest"],
                        "lifecycle": "resolved",
                        "reviewer": {"type": "human", "name": "reviewer-a"},
                        "binding": dict(target_one["binding"]),
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ri.import_decisions_into_index(package, incoming_one)
            copied = root / "copy"
            shutil.copytree(package, copied)
            original_decisions = (
                package / ri.REVIEW_DECISIONS_JSONL_NAME
            ).read_text(encoding="utf-8")
            copied_status = ri.review_index_status(
                *ri.load_review_index(copied)
            )
            copied_entries = ri.load_jsonl(copied / ri.REVIEW_INDEX_JSONL_NAME)
            target_two = next(
                entry for entry in copied_entries if entry["occurrence_id"] == "occ-2"
            )
            incoming_two = root / "two.jsonl"
            incoming_two.write_text(
                json.dumps(
                    {
                        "occurrence_id": "occ-2",
                        "project_identity_digest": target_two["project"]["identity_digest"],
                        "lifecycle": "ignored",
                        "reviewer": {"type": "human", "name": "reviewer-b"},
                        "binding": dict(target_two["binding"]),
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ri.import_decisions_into_index(copied, incoming_two)
            copied_decisions = (
                copied / ri.REVIEW_DECISIONS_JSONL_NAME
            ).read_text(encoding="utf-8")
            original_after = (
                package / ri.REVIEW_DECISIONS_JSONL_NAME
            ).read_text(encoding="utf-8")

        self.assertEqual(copied_status["entry_count"], 3)
        self.assertEqual(original_decisions, original_after)
        self.assertEqual(original_after.count("\n"), 1)
        self.assertEqual(copied_decisions.count("\n"), 2)

    def test_partial_template_import_skips_todo_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            template_path = root / "template.jsonl"
            ri.export_decisions(package, output_file=template_path)
            rows = ri.load_jsonl(template_path, label="template")
            rows[0]["lifecycle"] = "resolved"
            rows[0]["reviewer"] = {"type": "human", "name": "reviewer-a"}
            edited = root / "edited.jsonl"
            edited.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False) for row in rows
                )
                + "\n",
                encoding="utf-8",
            )

            result = ri.import_decisions_into_index(package, edited)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(result["merge"]["imported_count"], 1)
        self.assertGreater(result["merge"]["skipped_template_count"], 0)
        by_occurrence = {entry["occurrence_id"]: entry for entry in entries}
        self.assertEqual(by_occurrence["occ-1"]["review"]["lifecycle"], "resolved")
        self.assertEqual(by_occurrence["occ-2"]["review"]["lifecycle"], "open")

    def test_inferred_timestamp_does_not_replay_older_action(self) -> None:
        base = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "same note",
        }
        ignored = ri.normalize_decision({**base, "lifecycle": "ignored"})
        resolved = ri.normalize_decision(
            {
                **base,
                "lifecycle": "resolved",
                "decided_at": "2099-01-01T00:00:00+00:00",
            }
        )
        replay = ri.normalize_decision({**base, "lifecycle": "ignored"})

        merged, summary = ri.merge_decisions([ignored, resolved], [replay])

        self.assertEqual(summary["stale_count"], 1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[-1]["lifecycle"], "resolved")

    def test_timestamp_iso_variants_compare_by_instant(self) -> None:
        base = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "",
        }
        resolved = ri.normalize_decision(
            {
                **base,
                "lifecycle": "resolved",
                "decided_at": "2026-09-20T10:00:00Z",
            }
        )
        ignored = ri.normalize_decision(
            {
                **base,
                "lifecycle": "ignored",
                "decided_at": "2026-09-20T10:00:00+00:00",
            }
        )

        merged, summary = ri.merge_decisions([resolved], [ignored])

        self.assertEqual(summary["stale_count"], 0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[-1]["lifecycle"], "ignored")

    def test_invalid_decided_at_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "occurrence_id": "occ-1",
                    "project_identity_digest": "project-a",
                    "lifecycle": "resolved",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                    "decided_at": "not-a-timestamp",
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_reimporting_older_file_does_not_revert_newer_decision(self) -> None:
        base = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "",
        }
        ignored = ri.normalize_decision(
            {
                **base,
                "lifecycle": "ignored",
                "decided_at": "2026-09-20T00:00:00+00:00",
            }
        )
        resolved = ri.normalize_decision(
            {
                **base,
                "lifecycle": "resolved",
                "decided_at": "2026-09-20T01:00:00+00:00",
            }
        )

        merged, summary = ri.merge_decisions([ignored, resolved], [ignored])

        self.assertEqual(summary["stale_count"], 1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[-1]["lifecycle"], "resolved")

    def test_bare_corpus_jsonl_without_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "revision_corpus.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "occurrence_id": "occ-1",
                        "file_rel_path": "tl/ch1.rpy",
                        "source": "Hello",
                        "current_translation": "你好",
                        "locator": {"line": 1},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ri.ReviewIndexError) as captured:
                ri.build_review_index(jsonl, output_dir=root / "index")

        self.assertEqual(captured.exception.code, "REVIEW_INDEX_INPUT_MISSING")

    def test_repeated_multi_action_import_is_idempotent(self) -> None:
        base = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "",
        }
        ignored = ri.normalize_decision({**base, "lifecycle": "ignored"})
        resolved = ri.normalize_decision({**base, "lifecycle": "resolved"})

        merged, summary = ri.merge_decisions([ignored, resolved], [ignored, resolved])

        self.assertEqual(summary["duplicate_count"], 2)
        self.assertEqual(len(merged), 2)

    def test_corpus_without_project_identity_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.build_index_entries(
                [
                    {
                        "occurrence_id": "occ-1",
                        "file_rel_path": "tl/ch1.rpy",
                        "source": "Hello",
                        "current_translation": "你好",
                        "locator": {"line": 1},
                    }
                ],
                {},
            )

        self.assertEqual(captured.exception.code, "REVIEW_INDEX_INPUT_INVALID")

    def test_repeating_older_decision_is_a_new_reversion_action(self) -> None:
        base = {
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-a",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "note": "same note",
        }
        ignored = ri.normalize_decision(
            {
                **base,
                "lifecycle": "ignored",
                "decided_at": "2026-09-20T00:00:00+00:00",
            }
        )
        resolved = ri.normalize_decision(
            {
                **base,
                "lifecycle": "resolved",
                "note": "resolved note",
                "decided_at": "2026-09-20T01:00:00+00:00",
            }
        )
        reverted = ri.normalize_decision(
            {
                **base,
                "lifecycle": "ignored",
                "decided_at": "2026-09-20T02:00:00+00:00",
            }
        )

        merged, summary = ri.merge_decisions([ignored, resolved], [reverted])

        self.assertEqual(summary["duplicate_count"], 0)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[-1]["lifecycle"], "ignored")

    def test_unsupported_decision_schema_version_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "schema_version": 99,
                    "occurrence_id": "occ-1",
                    "project_identity_digest": "project-a",
                    "lifecycle": "resolved",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_template_placeholder_reviewer_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "occurrence_id": "occ-1",
                    "project_identity_digest": "project-a",
                    "lifecycle": "open",
                    "reviewer": {"type": "human", "name": "TODO"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_rebuild_reuses_recorded_quality_findings_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            findings = _write_findings(root)
            package = root / "index"
            ri.build_review_index(
                corpus,
                quality_findings_path=findings,
                output_dir=package,
            )

            rebuilt = ri.build_review_index(corpus, output_dir=package)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(
            Path(rebuilt["inputs"]["quality_findings"]["path"]).resolve(),
            findings.resolve(),
        )
        target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
        self.assertTrue(target["quality_finding_ids"])

    def test_rebuild_reuses_recorded_external_decisions_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            package = root / "index"
            ri.build_review_index(corpus, output_dir=package)
            entries = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)
            target = next(entry for entry in entries if entry["occurrence_id"] == "occ-1")
            external_decisions = root / "external_decisions.jsonl"
            external_decisions.write_text(
                json.dumps(
                    {
                        "occurrence_id": "occ-1",
                        "project_identity_digest": target["project"]["identity_digest"],
                        "lifecycle": "resolved",
                        "reviewer": {"type": "human", "name": "reviewer-a"},
                        "binding": dict(target["binding"]),
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            ri.build_review_index(
                corpus,
                decisions_path=external_decisions,
                output_dir=package,
            )
            rebuilt = ri.build_review_index(corpus, output_dir=package)
            refreshed = ri.load_jsonl(package / ri.REVIEW_INDEX_JSONL_NAME)

        self.assertEqual(
            Path(rebuilt["inputs"]["decisions"]["path"]).resolve(),
            external_decisions.resolve(),
        )
        refreshed_target = next(
            entry for entry in refreshed if entry["occurrence_id"] == "occ-1"
        )
        self.assertEqual(refreshed_target["review"]["lifecycle"], "resolved")

    def test_decision_without_project_identity_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "occurrence_id": "occ-1",
                    "lifecycle": "resolved",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_cross_project_decision_is_not_applied(self) -> None:
        entry = {
            "occurrence_id": "occ-1",
            "project": {"identity_digest": "project-a"},
            "binding": {
                "entry_id": "entry",
                "snapshot_digest": "snapshot",
                "source_digest": "source",
                "target_digest": "target",
                "context_digest": "context",
                "evidence_digest": "evidence",
            },
            "review": {},
        }
        decision = {
            "decision_id": "d1",
            "occurrence_id": "occ-1",
            "project_identity_digest": "project-b",
            "lifecycle": "resolved",
            "reviewer": {"type": "human", "name": "reviewer-a"},
            "note": "",
            "decided_at": "2026-09-20T00:00:00+00:00",
            "binding": dict(entry["binding"]),
        }

        applied, diagnostics = ri.apply_decisions([entry], [decision])

        self.assertEqual(applied[0]["review"]["lifecycle"], "open")
        self.assertTrue(
            any(
                item["code"] == ri.DIAGNOSTIC_DECISION_PROJECT_MISMATCH
                for item in diagnostics
            )
        )

    def test_invalid_decision_is_rejected(self) -> None:
        with self.assertRaises(ri.ReviewIndexError) as captured:
            ri.normalize_decision(
                {
                    "occurrence_id": "occ-1",
                    "lifecycle": "needs_recheck",
                    "reviewer": {"type": "human", "name": "reviewer-a"},
                    "binding": {
                        "entry_id": "x",
                        "snapshot_digest": "x",
                        "source_digest": "x",
                        "target_digest": "x",
                        "context_digest": "x",
                        "evidence_digest": "x",
                    },
                }
            )

        self.assertEqual(captured.exception.code, "REVIEW_DECISION_INVALID")

    def test_export_without_decisions_returns_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            ri.build_review_index(corpus, output_dir=root / "index")
            output_file = root / "template.jsonl"

            result = ri.export_decisions(
                root / "index" / ri.REVIEW_INDEX_MANIFEST_NAME,
                output_file=output_file,
            )
            rows = ri.load_jsonl(output_file, label="template")

        self.assertEqual(result["mode"], "template")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["lifecycle"] == "open" for row in rows))
        self.assertTrue(all(row["binding"]["entry_id"] for row in rows))


class CliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parents[1] / "gemini_translate_batch.py"),
                *args,
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )

    def test_build_status_export_import_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            findings = _write_findings(root)
            index_dir = root / "index"

            built = self._run(
                "review-index-build",
                "--corpus",
                str(corpus),
                "--quality-findings",
                str(findings),
                "--output-dir",
                str(index_dir),
                "--output",
                "json",
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            built_payload = json.loads(built.stdout)
            self.assertEqual(built_payload["command"], "review-index-build")
            self.assertEqual(built_payload["result"]["scope"]["entry_count"], 3)

            status = self._run(
                "review-index-status",
                "--index",
                str(index_dir),
                "--output",
                "json",
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            status_payload = json.loads(status.stdout)
            self.assertEqual(status_payload["result"]["entry_count"], 3)

            template_path = root / "template.jsonl"
            exported = self._run(
                "review-decisions-export",
                "--index",
                str(index_dir),
                "--file",
                str(template_path),
                "--output",
                "json",
            )
            self.assertEqual(exported.returncode, 0, exported.stderr)
            exported_payload = json.loads(exported.stdout)
            self.assertEqual(exported_payload["result"]["mode"], "template")
            rows = ri.load_jsonl(template_path, label="template")
            rows[0]["lifecycle"] = "resolved"
            rows[0]["reviewer"] = {"type": "human", "name": "cli-reviewer"}
            import_path = root / "import.jsonl"
            import_path.write_text(
                json.dumps(rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            imported = self._run(
                "review-decisions-import",
                "--index",
                str(index_dir),
                "--file",
                str(import_path),
                "--output",
                "json",
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            imported_payload = json.loads(imported.stdout)
            self.assertEqual(imported_payload["result"]["merge"]["imported_count"], 1)
            self.assertEqual(
                imported_payload["result"]["scope"]["lifecycle_counts"]["resolved"],
                1,
            )

            changed_corpus = _write_corpus(
                root / "changed",
                current_translation="您好",
                game_root=root / "game",
            )
            rebuilt = self._run(
                "review-index-build",
                "--corpus",
                str(changed_corpus),
                "--decisions",
                str(index_dir / ri.REVIEW_DECISIONS_JSONL_NAME),
                "--output-dir",
                str(root / "index-changed"),
                "--output",
                "json",
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            rebuilt_payload = json.loads(rebuilt.stdout)
            self.assertEqual(rebuilt_payload["status"], "needs_recheck")
            self.assertEqual(
                rebuilt_payload["result"]["scope"]["needs_recheck_count"],
                1,
            )

            rebuilt_status = self._run(
                "review-index-status",
                "--index",
                str(root / "index-changed"),
                "--output",
                "json",
            )
            self.assertEqual(rebuilt_status.returncode, 0, rebuilt_status.stderr)
            self.assertEqual(
                json.loads(rebuilt_status.stdout)["status"],
                "needs_recheck",
            )


if __name__ == "__main__":
    unittest.main()
