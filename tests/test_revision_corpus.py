"""Read-only revision corpus export tests (#320 / Epic #318 P1)."""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
import revision_corpus


FIXTURE = Path(__file__).parent / "fixtures" / "golden_revision_minimal" / "tl"


class RevisionCorpusExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tl_dir = self.root / "game" / "tl" / "schinese"
        self.tl_dir.mkdir(parents=True)
        shutil.copytree(FIXTURE, self.tl_dir, dirs_exist_ok=True)
        self.old = {
            "base": batch.legacy.BASE_DIR,
            "tl": batch.legacy.TL_DIR,
            "files": set(batch.legacy.INCLUDE_FILES),
            "prefixes": set(batch.legacy.INCLUDE_PREFIXES),
            "log": batch.LOG_DIR,
            "jobs": batch.BATCH_JOBS_DIR,
            "latest": batch.LATEST_MANIFEST_FILE,
            "progress": batch.PROGRESS_LOG,
            "rag": batch.RAG_ENABLED,
            "story": batch.STORY_MEMORY_ENABLED,
        }
        log_dir = self.root / "logs"
        batch.legacy.BASE_DIR = str(self.root)
        batch.legacy.TL_DIR = str(self.tl_dir)
        batch.legacy.INCLUDE_FILES = set()
        batch.legacy.INCLUDE_PREFIXES = set()
        batch.LOG_DIR = str(log_dir)
        batch.BATCH_JOBS_DIR = str(log_dir / "batch_jobs")
        batch.LATEST_MANIFEST_FILE = str(log_dir / "batch_jobs" / "latest_manifest.txt")
        batch.PROGRESS_LOG = str(log_dir / "translation_progress_batch.json")
        batch.RAG_ENABLED = False
        batch.STORY_MEMORY_ENABLED = False

    def tearDown(self):
        batch.legacy.BASE_DIR = self.old["base"]
        batch.legacy.TL_DIR = self.old["tl"]
        batch.legacy.INCLUDE_FILES = self.old["files"]
        batch.legacy.INCLUDE_PREFIXES = self.old["prefixes"]
        batch.LOG_DIR = self.old["log"]
        batch.BATCH_JOBS_DIR = self.old["jobs"]
        batch.LATEST_MANIFEST_FILE = self.old["latest"]
        batch.PROGRESS_LOG = self.old["progress"]
        batch.RAG_ENABLED = self.old["rag"]
        batch.STORY_MEMORY_ENABLED = self.old["story"]
        self.temp.cleanup()

    def _write_tl(self, rel_path: str, text: str) -> Path:
        target = self.tl_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def _export(self, output_dir: Path, **kwargs):
        jobs = batch.collect_revision_file_jobs()
        defaults = {
            "project_slug": "demo",
            "game_root": str(self.root),
            "tl_dir": str(self.tl_dir),
            "tl_subdir": "game/tl/schinese",
        }
        defaults.update(kwargs)
        return revision_corpus.export_revision_corpus(str(output_dir), jobs, **defaults)

    def test_export_writes_jsonl_markdown_and_manifest(self):
        out = self.root / "out"
        manifest = self._export(out)

        jsonl_path = out / "revision_corpus.jsonl"
        md_path = out / "revision_corpus.md"
        manifest_path = out / "revision_corpus_manifest.json"
        self.assertTrue(jsonl_path.is_file())
        self.assertTrue(md_path.is_file())
        self.assertTrue(manifest_path.is_file())

        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(manifest["scope"]["item_count"], len(rows))
        self.assertEqual(manifest["scope"]["file_count"], 1)
        md = md_path.read_text(encoding="utf-8")
        self.assertEqual(md.count("- L"), len(rows))
        for row in rows:
            self.assertEqual(
                row["schema_version"],
                revision_corpus.REVISION_CORPUS_SCHEMA_VERSION,
            )
            self.assertTrue(row["occurrence_id"])
            self.assertEqual(row["file_rel_path"], "chapter01/revisions.rpy")
            self.assertGreaterEqual(row["locator"]["line_number"], 1)
            self.assertGreaterEqual(row["locator"]["ordinal"], 1)
            self.assertTrue(row["snapshot_digest"])

    def test_duplicate_source_keeps_distinct_occurrences(self):
        self._write_tl(
            "chapter02/dups.rpy",
            'translate schinese chapter02_terms:\n'
            '    old "Repeat Me"\n'
            '    new "重复我"\n'
            '\n'
            '    old "Repeat Me"\n'
            '    new "重复我二号"\n',
        )
        rows, _ = revision_corpus.build_corpus_items(
            batch.collect_revision_file_jobs()
        )
        repeats = [row for row in rows if row["source"] == "Repeat Me"]
        self.assertEqual(len(repeats), 2)
        self.assertNotEqual(repeats[0]["occurrence_id"], repeats[1]["occurrence_id"])

    def test_comment_translations_are_exported(self):
        self._write_tl(
            "chapter03/comments.rpy",
            '# 艾琳 "这扇门通往虚空。"\n"这扇门通向虚空。"\n',
        )
        rows, _ = revision_corpus.build_corpus_items(
            batch.collect_revision_file_jobs()
        )
        comment = [row for row in rows if row["source"] == "这扇门通往虚空。"]
        self.assertEqual(len(comment), 1)
        self.assertEqual(comment[0]["current_translation"], "这扇门通向虚空。")
        self.assertEqual(comment[0]["file_rel_path"], "chapter03/comments.rpy")

    def test_export_is_deterministic(self):
        out1 = self.root / "out1"
        out2 = self.root / "out2"
        self._export(out1)
        self._export(out2)
        self.assertEqual(
            (out1 / "revision_corpus.jsonl").read_bytes(),
            (out2 / "revision_corpus.jsonl").read_bytes(),
        )
        self.assertEqual(
            (out1 / "revision_corpus.md").read_bytes(),
            (out2 / "revision_corpus.md").read_bytes(),
        )

    def test_manifest_contains_project_and_snapshot_identity(self):
        file_paths = {
            rel_path: file_path
            for rel_path, file_path in batch.collect_files_to_process()
        }
        digests = revision_corpus.collect_file_digests(file_paths)
        manifest = self._export(
            self.root / "out",
            include_files=["a.rpy"],
            include_prefixes=["chapter"],
            source_digests_before=digests,
            source_digests_after=digests,
        )
        self.assertEqual(manifest["kind"], "revision_corpus")
        self.assertEqual(manifest["project"]["slug"], "demo")
        self.assertEqual(manifest["project"]["game_root"], str(self.root))
        self.assertEqual(manifest["project"]["tl_dir"], str(self.tl_dir))
        self.assertEqual(manifest["project"]["tl_subdir"], "game/tl/schinese")
        self.assertEqual(manifest["project"]["include_files"], ["a.rpy"])
        self.assertEqual(manifest["project"]["include_prefixes"], ["chapter"])
        self.assertEqual(
            manifest["source"]["snapshot_digest"],
            revision_corpus.aggregate_digest(digests),
        )
        self.assertEqual(
            manifest["source"]["file_digests"],
            dict(sorted(digests.items())),
        )
        self.assertFalse(manifest["source"]["source_changed_during_scan"])
        self.assertEqual(
            manifest["scanner"]["engine"],
            revision_corpus.SCANNER_ENGINE,
        )

    def test_source_changed_during_scan_is_flagged(self):
        jobs = []
        changed = revision_corpus.export_revision_corpus(
            str(self.root / "out1"),
            jobs,
            project_slug="demo",
            game_root="",
            tl_dir="",
            tl_subdir="",
            source_digests_before={"a.rpy": "digest-1"},
            source_digests_after={"a.rpy": "digest-2"},
        )
        self.assertTrue(changed["source"]["source_changed_during_scan"])

        stable = revision_corpus.export_revision_corpus(
            str(self.root / "out2"),
            jobs,
            project_slug="demo",
            game_root="",
            tl_dir="",
            tl_subdir="",
            source_digests_before={"a.rpy": "digest-1"},
            source_digests_after={"a.rpy": "digest-1"},
        )
        self.assertFalse(stable["source"]["source_changed_during_scan"])

    def test_scanned_files_without_digest_are_flagged(self):
        jobs = [
            {
                "file_rel_path": "chapter01/revisions.rpy",
                "items": [
                    {
                        "id": "id-1",
                        "source": "S",
                        "current_translation": "T",
                        "line_number": 1,
                    }
                ],
            }
        ]
        manifest = revision_corpus.export_revision_corpus(
            str(self.root / "out"),
            jobs,
            project_slug="demo",
            game_root="",
            tl_dir="",
            tl_subdir="",
            source_digests_before={},
            source_digests_after={},
        )
        self.assertTrue(manifest["source"]["source_changed_during_scan"])
        self.assertEqual(
            manifest["source"]["scanned_files_missing_digest"],
            ["chapter01/revisions.rpy"],
        )

    def test_scanned_digest_mismatch_is_flagged(self):
        jobs = [
            {
                "file_rel_path": "a.rpy",
                "items": [
                    {
                        "id": "id-1",
                        "source": "S",
                        "current_translation": "T",
                        "line_number": 1,
                    }
                ],
            }
        ]
        manifest = revision_corpus.export_revision_corpus(
            str(self.root / "out"),
            jobs,
            project_slug="demo",
            game_root="",
            tl_dir="",
            tl_subdir="",
            source_digests_before={"a.rpy": "digest-1"},
            source_digests_after={"a.rpy": "digest-1"},
            source_digests_scanned={"a.rpy": "digest-mid-scan"},
        )
        self.assertTrue(manifest["source"]["source_changed_during_scan"])
        self.assertEqual(
            manifest["source"]["scanned_files_digest_mismatch"],
            ["a.rpy"],
        )

        stable = revision_corpus.export_revision_corpus(
            str(self.root / "out2"),
            jobs,
            project_slug="demo",
            game_root="",
            tl_dir="",
            tl_subdir="",
            source_digests_before={"a.rpy": "digest-1"},
            source_digests_after={"a.rpy": "digest-1"},
            source_digests_scanned={"a.rpy": "digest-1"},
        )
        self.assertFalse(stable["source"]["source_changed_during_scan"])
        self.assertEqual(
            stable["source"]["scanned_files_digest_mismatch"],
            [],
        )

    def test_revision_jobs_record_scanned_source_digest(self):
        jobs = batch.collect_revision_file_jobs()
        self.assertTrue(jobs)
        self.assertTrue(all(job.get("source_digest") for job in jobs))
        first = jobs[0]
        expected = revision_corpus.collect_file_digests(
            {first["file_rel_path"]: first["file_path"]}
        )[first["file_rel_path"]]
        self.assertEqual(first["source_digest"], expected)

    def test_locator_non_numeric_produces_diagnostic(self):
        jobs = [
            {
                "file_rel_path": "a.rpy",
                "items": [
                    {
                        "id": "id-1",
                        "source": "S",
                        "current_translation": "T",
                        "line": "oops",
                        "line_number": 3,
                        "start": 1,
                        "end": "x",
                    }
                ],
            }
        ]
        rows, diagnostics = revision_corpus.build_corpus_items(jobs)
        self.assertEqual(rows[0]["locator"]["line"], 0)
        self.assertEqual(rows[0]["locator"]["end"], 0)
        self.assertEqual(rows[0]["locator"]["line_number"], 3)
        self.assertEqual(
            {entry["field"] for entry in diagnostics},
            {"line", "end"},
        )
        self.assertTrue(
            all(entry["code"] == "LOCATOR_NON_NUMERIC" for entry in diagnostics)
        )

    def test_context_links_adjacent_items(self):
        jobs = [
            {
                "file_rel_path": "a.rpy",
                "items": [
                    {
                        "id": "id-1",
                        "source": "First",
                        "current_translation": "第一",
                        "line_number": 1,
                    },
                    {
                        "id": "id-2",
                        "source": "Second",
                        "current_translation": "第二",
                        "line_number": 2,
                    },
                ],
            }
        ]
        rows, _ = revision_corpus.build_corpus_items(jobs)
        self.assertIsNone(rows[0]["context"]["previous"])
        self.assertEqual(
            rows[0]["context"]["next"],
            {"source": "Second", "current_translation": "第二"},
        )
        self.assertEqual(
            rows[1]["context"]["previous"],
            {"source": "First", "current_translation": "第一"},
        )
        self.assertIsNone(rows[1]["context"]["next"])

    def test_build_corpus_items_sorts_files_explicitly(self):
        jobs = [
            {
                "file_rel_path": "z.rpy",
                "items": [
                    {
                        "id": "z-1",
                        "source": "Z",
                        "current_translation": "Z译",
                    }
                ],
            },
            {
                "file_rel_path": "a.rpy",
                "items": [
                    {
                        "id": "a-1",
                        "source": "A",
                        "current_translation": "A译",
                    }
                ],
            },
        ]
        rows, _ = revision_corpus.build_corpus_items(jobs)
        self.assertEqual(
            [row["file_rel_path"] for row in rows],
            ["a.rpy", "z.rpy"],
        )

    def test_build_corpus_items_preserves_speaker_and_locator_fields(self):
        jobs = [
            {
                "file_rel_path": "a.rpy",
                "items": [
                    {
                        "id": "fallback-id",
                        "identity_v2": "id-1",
                        "source": "Hello",
                        "current_translation": "你好",
                        "line_number": 3,
                        "line": 2,
                        "start": 4,
                        "end": 11,
                        "speaker_id": "alice",
                    }
                ],
            }
        ]
        rows, _ = revision_corpus.build_corpus_items(jobs)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["identity_v2"], "id-1")
        self.assertEqual(row["occurrence_id"], "id-1")
        self.assertEqual(row["speaker_id"], "alice")
        self.assertEqual(row["locator"]["line_number"], 3)
        self.assertEqual(row["locator"]["ordinal"], 1)
        self.assertEqual(row["display_line"], 3)
        self.assertEqual(row["source"], "Hello")
        self.assertEqual(row["current_translation"], "你好")

    def test_cli_registers_offline_machine_command(self):
        self.assertIn("export-revision-corpus", batch.MACHINE_OUTPUT_COMMANDS)
        self.assertIn("export-revision-corpus", batch.OFFLINE_BATCH_COMMANDS)
        parser = batch.build_arg_parser()
        args = parser.parse_args(
            ["export-revision-corpus", "--output-dir", "x", "--output", "json"]
        )
        self.assertEqual(args.command, "export-revision-corpus")
        self.assertEqual(args.output_dir, "x")
        self.assertEqual(args.output, "json")

    def test_cli_machine_envelope_reports_artifacts(self):
        manifest_value = {
            "paths": {
                "output_dir": "C:/out",
                "jsonl": "C:/out/revision_corpus.jsonl",
                "markdown": "C:/out/revision_corpus.md",
                "manifest": "C:/out/revision_corpus_manifest.json",
            },
            "scope": {"file_count": 2, "item_count": 5},
            "source": {"source_changed_during_scan": False},
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest_value),
            redirect_stdout(stdout),
        ):
            exit_code = batch.main(["export-revision-corpus", "--output", "json"])

        self.assertEqual(exit_code, 0)
        envelope = json.loads(stdout.getvalue())
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["command"], "export-revision-corpus")
        self.assertEqual(envelope["status"], "completed")
        self.assertEqual(envelope["result"]["item_count"], 5)
        self.assertEqual(envelope["result"]["file_count"], 2)
        self.assertEqual(
            envelope["artifacts"]["corpus_jsonl"],
            "C:/out/revision_corpus.jsonl",
        )
        self.assertEqual(
            envelope["artifacts"]["corpus_manifest"],
            "C:/out/revision_corpus_manifest.json",
        )

    def test_end_to_end_cli_export(self):
        stdout = io.StringIO()
        with (
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "initialize_batch_logging"),
            redirect_stdout(stdout),
        ):
            exit_code = batch.main(["export-revision-corpus", "--output", "json"])

        self.assertEqual(exit_code, 0)
        envelope = json.loads(stdout.getvalue())
        self.assertTrue(envelope["ok"])
        jsonl_path = envelope["result"]["corpus_jsonl"]
        self.assertTrue(os.path.isfile(jsonl_path))
        with open(jsonl_path, encoding="utf-8") as handle:
            rows = [
                json.loads(line) for line in handle if line.strip()
            ]
        self.assertEqual(envelope["result"]["item_count"], len(rows))
        self.assertGreaterEqual(len(rows), 3)
        manifest_path = envelope["result"]["corpus_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            persisted = json.load(handle)
        self.assertEqual(persisted["kind"], "revision_corpus")


if __name__ == "__main__":
    unittest.main()
