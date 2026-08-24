import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
import revision_corpus
import revision_proposals as proposals
import revision_selection


class RevisionProposalContractTests(unittest.TestCase):
    def setUp(self):
        self.live = {
            "occ-1": {
                "id": "occ-1",
                "file_rel_path": "chapter.rpy",
                "source": "Hello {name}",
                "current_translation": "你好 {name}",
            }
        }
        self.corpus_digest = "a" * 64

    def row(self, **updates):
        row = {
            "schema_version": 1,
            "occurrence_id": "occ-1",
            "identity_v2": "occ-1",
            "file_rel_path": "chapter.rpy",
            "source": "Hello {name}",
            "current_translation": "你好 {name}",
            "proposed_translation": "您好，{name}",
            "reason": "语气更自然",
            "selected": True,
            "disposition": "accepted",
            "producer": {"type": "human"},
            "project_identity": {"tl_dir": "C:/demo/game/tl/schinese"},
            "snapshot_digest": revision_corpus.item_snapshot_digest(
                "Hello {name}", "你好 {name}"
            ),
            "corpus_snapshot_digest": self.corpus_digest,
        }
        row.update(updates)
        return row

    def test_valid_human_proposal_is_imported(self):
        result = proposals.validate(
            [self.row()], self.live, live_snapshot_digest=self.corpus_digest
        )
        self.assertEqual(result.status, "imported")
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.diagnostics, ())

    def test_duplicate_and_conflicting_occurrences_are_blocked(self):
        duplicate = proposals.validate(
            [self.row(), self.row()], self.live, live_snapshot_digest=self.corpus_digest
        )
        self.assertIn("DUPLICATE_OCCURRENCE_ID", {x["code"] for x in duplicate.diagnostics})
        conflict = proposals.validate(
            [self.row(), self.row(proposed_translation="另一版")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertIn("CONFLICTING_PROPOSAL", {x["code"] for x in conflict.diagnostics})

    def test_unknown_and_stale_current_translation_are_rejected(self):
        unknown = proposals.validate(
            [self.row(occurrence_id="missing", identity_v2="missing")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertIn("UNKNOWN_OCCURRENCE_ID", {x["code"] for x in unknown.diagnostics})
        stale = proposals.validate(
            [self.row(current_translation="旧译文")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(stale.status, "stale")
        self.assertIn("CURRENT_TRANSLATION_STALE", {x["code"] for x in stale.diagnostics})

    def test_occurrence_and_identity_v2_must_agree(self):
        result = proposals.validate(
            [self.row(identity_v2="different")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertIn("IDENTITY_MISMATCH", {x["code"] for x in result.diagnostics})

    def test_reason_is_required_for_selected_proposals(self):
        result = proposals.validate(
            [self.row(reason="  ")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.selected_count, 0)
        self.assertIn("MISSING_REASON", {x["code"] for x in result.diagnostics})

    def test_proposed_translation_and_reason_must_be_strings(self):
        result = proposals.validate(
            [self.row(proposed_translation=123, reason=["not", "text"])],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.selected_count, 0)
        self.assertIn(
            "INVALID_PROPOSED_TRANSLATION_TYPE",
            {x["code"] for x in result.diagnostics},
        )
        self.assertIn("INVALID_REASON_TYPE", {x["code"] for x in result.diagnostics})

    def test_source_and_current_translation_must_be_strings(self):
        result = proposals.validate(
            [self.row(source=123, current_translation=["你好 {name}"])],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.selected_count, 0)
        self.assertIn("INVALID_SOURCE_TYPE", {x["code"] for x in result.diagnostics})
        self.assertIn(
            "INVALID_CURRENT_TRANSLATION_TYPE",
            {x["code"] for x in result.diagnostics},
        )

    def test_stringified_source_and_current_translation_are_not_accepted(self):
        live = {
            "occ-1": {
                "id": "occ-1",
                "file_rel_path": "chapter.rpy",
                "source": "123",
                "current_translation": "456",
            }
        }
        row = self.row(
            source=123,
            current_translation=456,
            snapshot_digest=revision_corpus.item_snapshot_digest("123", "456"),
        )
        result = proposals.validate(
            [row], live, live_snapshot_digest=self.corpus_digest
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.selected_count, 0)
        self.assertIn("INVALID_SOURCE_TYPE", {x["code"] for x in result.diagnostics})
        self.assertIn(
            "INVALID_CURRENT_TRANSLATION_TYPE",
            {x["code"] for x in result.diagnostics},
        )

    def test_conflicting_item_snapshot_fields_are_blocked(self):
        result = proposals.validate(
            [self.row(item_snapshot_digest="b" * 64)],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.selected_count, 0)
        self.assertIn(
            "ITEM_SNAPSHOT_CONFLICT",
            {x["code"] for x in result.diagnostics},
        )

    def test_file_path_separator_style_is_normalized(self):
        self.live["occ-1"]["file_rel_path"] = "chapter/scene.rpy"
        result = proposals.validate(
            [self.row(file_rel_path=r"chapter\scene.rpy")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "imported")

    def test_shared_project_identity_normalizes_paths_for_cli_and_gui(self):
        project = revision_selection.project_identity_from_paths(
            game_root=r"C:\Demo\work\..\work",
            tl_dir=r"C:\Demo\work\game\tl\schinese\.",
        )
        equivalent = revision_selection.project_identity_from_paths(
            game_root=r"c:\demo\work",
            tl_dir=r"c:\demo\work\game\tl\schinese",
        )
        self.assertEqual(project, equivalent)
        self.assertTrue(revision_selection.project_identities_match(project, equivalent))
        self.assertEqual(
            revision_selection.operation_identity(
                project_identity=project,
                proposal_path=r"C:\Demo\work\proposals.jsonl",
                proposal_sha256="a" * 64,
                corpus_manifest_path=r"C:\Demo\work\corpus.json",
                corpus_manifest_sha256="b" * 64,
            ),
            revision_selection.operation_identity(
                project_identity=equivalent,
                proposal_path=r"c:\demo\work\.\proposals.jsonl",
                proposal_sha256="a" * 64,
                corpus_manifest_path=r"c:\demo\work\corpus.json",
                corpus_manifest_sha256="b" * 64,
            ),
        )

    def test_corpus_snapshot_mismatch_is_stale(self):
        result = proposals.validate(
            [self.row(corpus_snapshot_digest="b" * 64)],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "stale")
        self.assertIn("CORPUS_SNAPSHOT_STALE", {x["code"] for x in result.diagnostics})

    def test_inconsistent_corpus_export_is_stale(self):
        result = proposals.validate(
            [self.row()],
            self.live,
            live_snapshot_digest=self.corpus_digest,
            corpus_manifest={
                "project": {"tl_dir": "C:/demo/game/tl/schinese"},
                "source": {
                    "snapshot_digest": self.corpus_digest,
                    "source_changed_during_scan": True,
                }
            },
        )
        self.assertEqual(result.status, "stale")
        self.assertIn(
            "CORPUS_SNAPSHOT_INCONSISTENT",
            {x["code"] for x in result.diagnostics},
        )

    def test_legacy_revision_identity_keeps_pre_proposal_fingerprint(self):
        manifest = {
            "mode": "revision",
            "manifest_version": 2,
            "version": 2,
            "core_schema_version": 2,
            "display_name": "legacy",
            "summary": {"item_count": 1},
            "files": {"chapter.rpy": {"task_count": 1}},
            "chunks": [{"key": "rv-1"}],
        }
        legacy_keys = (
            "mode", "manifest_version", "version", "core_schema_version",
            "display_name", "job_name", "created_at", "execution",
            "batch_model", "model", "base_dir", "tl_dir",
            "target_language", "language", "input_jsonl_path",
            "result_jsonl_path", "settings", "revision_settings", "summary",
            "files", "chunks", "final_review_source",
        )
        payload = {key: manifest.get(key) for key in legacy_keys}
        expected = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(batch._revision_manifest_identity(manifest), expected)
        self.assertEqual(
            batch._revision_manifest_identity({**manifest, "proposal_import": None}),
            expected,
        )

    def test_proposal_import_state_is_bound_by_manifest_identity(self):
        manifest = {
            "mode": "revision",
            "proposal_import": {
                "status": "previewed",
                "writeback_eligible": True,
            },
        }
        previewed = batch._revision_manifest_identity(manifest)
        blocked = batch._revision_manifest_identity(
            {
                **manifest,
                "proposal_import": {
                    "status": "blocked",
                    "writeback_eligible": False,
                },
            }
        )
        self.assertNotEqual(previewed, blocked)

    def test_unselected_rows_produce_no_op(self):
        result = proposals.validate(
            [self.row(selected=False, disposition="rejected")],
            self.live,
            live_snapshot_digest=self.corpus_digest,
        )
        self.assertEqual(result.status, "no_op")
        self.assertFalse(result.proposals)

    def test_project_identity_is_required_without_corpus_manifest(self):
        result = proposals.validate(
            [self.row(project_identity=None)],
            self.live,
            live_snapshot_digest=self.corpus_digest,
            live_project_identity={"tl_dir": "C:/demo/game/tl/schinese"},
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("MISSING_PROJECT_IDENTITY", {x["code"] for x in result.diagnostics})

    def test_cross_project_proposal_is_stale_without_corpus_manifest(self):
        result = proposals.validate(
            [self.row(project_identity={"tl_dir": "C:/other/game/tl/schinese"})],
            self.live,
            live_snapshot_digest=self.corpus_digest,
            live_project_identity={"tl_dir": "C:/demo/game/tl/schinese"},
        )
        self.assertEqual(result.status, "stale")
        self.assertEqual(result.requested_selected_count, 1)
        self.assertEqual(result.selected_count, 0)
        self.assertIn("PROJECT_IDENTITY_STALE", {x["code"] for x in result.diagnostics})

    def test_matching_corpus_manifest_supplies_project_identity(self):
        result = proposals.validate(
            [self.row(project_identity=None)],
            self.live,
            live_snapshot_digest=self.corpus_digest,
            live_project_identity={"tl_dir": "C:/demo/game/tl/schinese"},
            corpus_manifest={
                "project": {"tl_dir": "C:/demo/game/tl/schinese"},
                "source": {"snapshot_digest": self.corpus_digest},
            },
        )
        self.assertEqual(result.status, "imported")
        self.assertEqual(result.selected_count, 1)

    def test_jsonl_loader_reports_bad_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "proposal.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self.row(), ensure_ascii=False) + "\n{bad\n")
            with self.assertRaisesRegex(ValueError, "row 2"):
                proposals.load_jsonl(path)


class RevisionProposalImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tl_dir = self.root / "game" / "tl" / "schinese"
        self.tl_dir.mkdir(parents=True)
        self.rpy = self.tl_dir / "chapter.rpy"
        self.rpy.write_text(
            'translate schinese demo:\n    old "Hello {name}"\n    new "你好 {name}"\n',
            encoding="utf-8",
        )
        self.old = {
            "base": batch.legacy.BASE_DIR,
            "tl": batch.legacy.TL_DIR,
            "files": set(batch.legacy.INCLUDE_FILES),
            "prefixes": set(batch.legacy.INCLUDE_PREFIXES),
            "jobs": batch.BATCH_JOBS_DIR,
            "latest": batch.LATEST_MANIFEST_FILE,
            "rag": batch.RAG_ENABLED,
            "story": batch.STORY_MEMORY_ENABLED,
        }
        batch.legacy.BASE_DIR = str(self.root)
        batch.legacy.TL_DIR = str(self.tl_dir)
        batch.legacy.INCLUDE_FILES = set()
        batch.legacy.INCLUDE_PREFIXES = set()
        batch.BATCH_JOBS_DIR = str(self.root / "logs" / "batch_jobs")
        batch.LATEST_MANIFEST_FILE = str(Path(batch.BATCH_JOBS_DIR) / "latest_manifest.txt")
        batch.RAG_ENABLED = False
        batch.STORY_MEMORY_ENABLED = False

    def tearDown(self):
        batch.legacy.BASE_DIR = self.old["base"]
        batch.legacy.TL_DIR = self.old["tl"]
        batch.legacy.INCLUDE_FILES = self.old["files"]
        batch.legacy.INCLUDE_PREFIXES = self.old["prefixes"]
        batch.BATCH_JOBS_DIR = self.old["jobs"]
        batch.LATEST_MANIFEST_FILE = self.old["latest"]
        batch.RAG_ENABLED = self.old["rag"]
        batch.STORY_MEMORY_ENABLED = self.old["story"]
        self.temp.cleanup()

    def _proposal(self, proposed="您好，{name}"):
        item = batch.collect_revision_file_jobs()[0]["items"][0]
        paths = dict(batch.collect_files_to_process())
        corpus_digest = revision_corpus.aggregate_digest(
            revision_corpus.collect_file_digests(paths)
        )
        return {
            "schema_version": 1,
            "occurrence_id": item["id"],
            "identity_v2": item["id"],
            "file_rel_path": item["file_rel_path"],
            "source": item["source"],
            "current_translation": item["current_translation"],
            "proposed_translation": proposed,
            "reason": "test",
            "selected": True,
            "disposition": "accepted",
            "producer": {"type": "agent", "tool": "unit-test"},
            "project_identity": {"tl_dir": str(self.tl_dir)},
            "snapshot_digest": revision_corpus.item_snapshot_digest(
                item["source"], item["current_translation"]
            ),
            "corpus_snapshot_digest": corpus_digest,
        }

    def _write_proposal(self, row):
        path = self.root / "proposal.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def test_valid_proposal_builds_standard_preview_without_writing_rpy(self):
        before = self.rpy.read_bytes()
        result = batch.import_revision_proposals(str(self._write_proposal(self._proposal())))
        self.assertEqual(result["status"], "previewed")
        self.assertEqual(self.rpy.read_bytes(), before)
        self.assertTrue(Path(result["paths"]["manifest"]).is_file())
        self.assertTrue(Path(result["paths"]["revision_preview_jsonl"]).is_file())
        manifest = result["manifest"]
        self.assertEqual(manifest["mode"], batch.MANIFEST_MODE_REVISION)
        self.assertTrue(manifest["proposal_import"]["writeback_eligible"])
        self.assertEqual(
            Path(batch.LATEST_MANIFEST_FILE).read_text(encoding="utf-8"),
            result["paths"]["manifest"],
        )

    def test_empty_proposal_file_is_rejected_before_artifacts_are_created(self):
        proposal_path = self.root / "empty.jsonl"
        proposal_path.write_text("\n", encoding="utf-8")
        before = self.rpy.read_bytes()
        with self.assertRaises(batch.cli_contract.MachineContractError) as caught:
            batch.import_revision_proposals(str(proposal_path))
        self.assertEqual(caught.exception.code_name, "NO_PROPOSAL_ROWS")
        self.assertEqual(self.rpy.read_bytes(), before)
        self.assertFalse(Path(batch.BATCH_JOBS_DIR).exists())

    def test_corpus_manifest_without_snapshot_digest_is_rejected(self):
        corpus_path = self.root / "missing-snapshot-corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "kind": "revision_corpus",
                    "schema_version": revision_corpus.REVISION_CORPUS_SCHEMA_VERSION,
                    "project": {"tl_dir": str(self.tl_dir)},
                    "source": {"source_changed_during_scan": False},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(batch.cli_contract.MachineContractError) as caught:
            batch.import_revision_proposals(
                str(self._write_proposal(self._proposal())),
                corpus_manifest_path=str(corpus_path),
            )
        self.assertEqual(caught.exception.code_name, "CORPUS_MANIFEST_INVALID")
        self.assertFalse(Path(batch.BATCH_JOBS_DIR).exists())

    def test_malformed_corpus_manifest_has_stable_machine_error(self):
        corpus_path = self.root / "bad-corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "kind": "revision_corpus",
                    "schema_version": revision_corpus.REVISION_CORPUS_SCHEMA_VERSION,
                    "project": {"tl_dir": str(self.tl_dir)},
                    "source": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(batch.cli_contract.MachineContractError) as caught:
            batch.import_revision_proposals(
                str(self._write_proposal(self._proposal())),
                corpus_manifest_path=str(corpus_path),
            )
        self.assertEqual(caught.exception.code_name, "CORPUS_MANIFEST_INVALID")
        self.assertFalse(Path(batch.BATCH_JOBS_DIR).exists())

    def test_unselected_proposals_report_no_writeback_needed(self):
        row = self._proposal()
        row.update(selected=False, disposition="rejected")
        result = batch.import_revision_proposals(str(self._write_proposal(row)))
        self.assertEqual(result["status"], "no_op")
        self.assertEqual(result["suggested_action"], "no_writeback_needed")
        self.assertNotIn("manifest", result["paths"])

    def test_broken_interpolation_token_is_blocked_and_never_writes(self):
        before = self.rpy.read_bytes()
        latest_path = Path(batch.LATEST_MANIFEST_FILE)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        previous_manifest = self.root / "previous" / "manifest.json"
        latest_path.write_text(str(previous_manifest), encoding="utf-8")
        result = batch.import_revision_proposals(
            str(self._write_proposal(self._proposal(proposed="您好")))
        )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.rpy.read_bytes(), before)
        self.assertEqual(latest_path.read_text(encoding="utf-8"), str(previous_manifest))
        self.assertFalse(result["manifest"]["proposal_import"]["writeback_eligible"])
        with self.assertRaisesRegex(SystemExit, "proposal import"):
            batch.apply_revisions(result["paths"]["manifest"], force=True)
        self.assertEqual(self.rpy.read_bytes(), before)

    def test_repreview_recomputes_proposal_writeback_eligibility(self):
        result = batch.import_revision_proposals(
            str(self._write_proposal(self._proposal()))
        )
        manifest_path = result["paths"]["manifest"]
        self.rpy.write_text(
            'translate schinese demo:\n'
            '    old "Hello {name}"\n'
            '    new "源文件已变化 {name}"\n',
            encoding="utf-8",
        )

        repreviewed = batch.preview_revisions(manifest_path)

        self.assertEqual(repreviewed["proposal_import"]["status"], "blocked")
        self.assertFalse(repreviewed["proposal_import"]["writeback_eligible"])
        self.assertEqual(
            repreviewed["proposal_import"]["history"],
            ["imported", "previewed", "blocked"],
        )
        with self.assertRaisesRegex(SystemExit, "proposal import"):
            batch.apply_revisions(manifest_path, force=True)

    def test_duplicate_source_applies_only_selected_occurrence(self):
        self.rpy.write_text(
            'translate schinese demo:\n'
            '    old "Repeat"\n'
            '    new "第一处"\n\n'
            '    old "Repeat"\n'
            '    new "第二处"\n',
            encoding="utf-8",
        )
        jobs = batch.collect_revision_file_jobs()
        target = jobs[0]["items"][1]
        paths = dict(batch.collect_files_to_process())
        corpus_digest = revision_corpus.aggregate_digest(
            revision_corpus.collect_file_digests(paths)
        )
        row = {
            "schema_version": 1,
            "occurrence_id": target["id"],
            "identity_v2": target["id"],
            "file_rel_path": target["file_rel_path"],
            "source": target["source"],
            "current_translation": target["current_translation"],
            "proposed_translation": "只改第二处",
            "reason": "target occurrence",
            "selected": True,
            "disposition": "accepted",
            "producer": {"type": "human"},
            "project_identity": {"tl_dir": str(self.tl_dir)},
            "snapshot_digest": revision_corpus.item_snapshot_digest(
                target["source"], target["current_translation"]
            ),
            "corpus_snapshot_digest": corpus_digest,
        }
        result = batch.import_revision_proposals(str(self._write_proposal(row)))
        batch.apply_revisions(result["paths"]["manifest"])
        written = self.rpy.read_text(encoding="utf-8")
        self.assertIn('new "第一处"', written)
        self.assertIn('new "只改第二处"', written)

    def test_stage_selection_persists_candidates_without_preview_or_writeback(self):
        before = self.rpy.read_bytes()
        result = batch.import_revision_proposals(
            str(self._write_proposal(self._proposal())),
            stage=True,
        )

        self.assertEqual(result["status"], "staged")
        self.assertNotIn("manifest", result["paths"])
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["selectable_count"], 1)
        self.assertTrue(Path(result["paths"]["staged_selection"]).is_file())
        self.assertEqual(self.rpy.read_bytes(), before)

        stage = revision_selection.load_staged_selection(
            result["paths"]["staged_selection"]
        )
        self.assertEqual(
            [candidate["identity_v2"] for candidate in stage["candidates"]],
            [self._proposal()["identity_v2"]],
        )
        self.assertEqual(
            len(revision_selection.filter_candidates(stage, valid_only=True)),
            1,
        )

    def test_stage_missing_identity_is_reported_as_invalid_candidate(self):
        row = self._proposal()
        row.update(occurrence_id="", identity_v2="")
        result = batch.import_revision_proposals(
            str(self._write_proposal(row)),
            stage=True,
        )

        self.assertEqual(result["status"], "staged")
        self.assertEqual(result["invalid_count"], 1)
        self.assertEqual(result["selectable_count"], 0)
        stage = revision_selection.load_staged_selection(
            result["paths"]["staged_selection"]
        )
        self.assertEqual(stage["candidates"][0]["identity_v2"], "")
        self.assertEqual(
            stage["candidates"][0]["status"],
            revision_selection.STATUS_INVALID,
        )

    def test_stage_whitespace_translation_change_remains_selectable(self):
        self.rpy.write_text(
            'translate schinese demo:\n'
            '    old "Hello {name}"\n'
            '    new "你好世界"\n',
            encoding="utf-8",
        )
        result = batch.import_revision_proposals(
            str(self._write_proposal(self._proposal(proposed="你好 世界"))),
            stage=True,
        )

        self.assertEqual(result["selectable_count"], 1)
        self.assertEqual(result["no_op_count"], 0)

    def test_stage_row_level_stale_keeps_ready_session_for_other_candidates(self):
        live = {
            "valid-id": {
                "id": "valid-id",
                "file_rel_path": "chapter.rpy",
                "source": "Hello {name}",
                "current_translation": "你好 {name}",
            },
            "occ-2": {
                "id": "occ-2",
                "file_rel_path": "other.rpy",
                "source": "Other",
                "current_translation": "当前译文",
            },
        }
        valid = self._proposal()
        live[valid["occurrence_id"]] = live.pop("valid-id")
        live[valid["occurrence_id"]]["id"] = valid["occurrence_id"]
        stale = dict(valid)
        stale.update(
            occurrence_id="occ-2",
            identity_v2="occ-2",
            file_rel_path="other.rpy",
            source="Other",
            current_translation="旧译文",
            proposed_translation="新译文",
            snapshot_digest=revision_corpus.item_snapshot_digest("Other", "旧译文"),
        )
        stage = revision_selection.build_staged_selection(
            rows=[valid, stale],
            live_items=live,
            live_snapshot_digest=valid["corpus_snapshot_digest"],
            project_identity={"game_root": self.root, "tl_dir": self.tl_dir},
            proposal_path="C:/demo/proposals.jsonl",
            proposal_sha256="b" * 64,
            operation_id="operation-1",
        )

        self.assertEqual(stage["session"]["session_status"], "ready")
        self.assertEqual(stage["summary"]["selectable_count"], 1)
        self.assertEqual(stage["summary"]["stale_count"], 1)

    def test_confirmed_stage_selection_generates_preview_and_binds_selection(self):
        proposal_path = self._write_proposal(self._proposal())
        expected_identity = revision_selection.operation_identity(
            project_identity=revision_selection.project_identity_from_paths(
                game_root=self.root,
                tl_dir=self.tl_dir,
            ),
            proposal_path=str(proposal_path),
            proposal_sha256=revision_selection.file_sha256(str(proposal_path)),
        )
        staged = batch.import_revision_proposals(
            str(proposal_path),
            stage=True,
            operation_identity=expected_identity,
        )
        stage = revision_selection.load_staged_selection(
            staged["paths"]["staged_selection"]
        )
        identity = stage["candidates"][0]["identity_v2"]
        request = revision_selection.make_selection_request(stage, [identity])
        selection_path = self.root / "selection.json"
        revision_selection.write_selection_request(str(selection_path), request)

        result = batch.confirm_revision_proposals(
            staged["paths"]["staged_selection"],
            str(selection_path),
        )

        self.assertEqual(result["status"], "previewed")
        self.assertTrue(Path(result["paths"]["manifest"]).is_file())
        proposal_state = result["manifest"]["proposal_import"]
        self.assertEqual(proposal_state["selected_identity_v2"], [identity])
        self.assertEqual(
            proposal_state["staged_selection_digest"],
            stage["staged_selection_digest"],
        )
        self.assertEqual(proposal_state["operation_identity"], expected_identity)
        self.assertTrue(proposal_state["selection_sha256"])
        self.assertEqual(self.rpy.read_text(encoding="utf-8").count("您好"), 0)
        selection_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "selection changed"):
            batch.apply_revisions(result["paths"]["manifest"], force=True)

    def test_confirmed_stage_selection_is_stale_after_source_change(self):
        proposal_path = self._write_proposal(self._proposal())
        staged = batch.import_revision_proposals(str(proposal_path), stage=True)
        stage_report = Path(staged["paths"]["staged_selection_report"])
        original_stage_report = stage_report.read_bytes()
        stage = revision_selection.load_staged_selection(
            staged["paths"]["staged_selection"]
        )
        request = revision_selection.make_selection_request(
            stage,
            [stage["candidates"][0]["identity_v2"]],
        )
        selection_path = self.root / "selection.json"
        revision_selection.write_selection_request(str(selection_path), request)
        self.rpy.write_text(
            'translate schinese demo:\n'
            '    old "Hello {name}"\n'
            '    new "源文件发生变化 {name}"\n',
            encoding="utf-8",
        )

        result = batch.confirm_revision_proposals(
            staged["paths"]["staged_selection"],
            str(selection_path),
        )

        self.assertEqual(result["status"], "stale")
        self.assertNotIn("manifest", result["paths"])
        self.assertTrue(Path(result["paths"]["selection_confirmation_report"]).is_file())
        self.assertEqual(stage_report.read_bytes(), original_stage_report)

    def test_confirmed_stage_selection_is_stale_after_project_switch(self):
        proposal_path = self._write_proposal(self._proposal())
        staged = batch.import_revision_proposals(str(proposal_path), stage=True)
        stage = revision_selection.load_staged_selection(
            staged["paths"]["staged_selection"]
        )
        request = revision_selection.make_selection_request(
            stage,
            [stage["candidates"][0]["identity_v2"]],
        )
        selection_path = self.root / "selection.json"
        revision_selection.write_selection_request(str(selection_path), request)
        batch.legacy.BASE_DIR = str(self.root / "other-game")

        result = batch.confirm_revision_proposals(
            staged["paths"]["staged_selection"],
            str(selection_path),
        )

        self.assertEqual(result["status"], "stale")
        self.assertTrue(
            any(
                item["code"] == "PROJECT_IDENTITY_STALE"
                for item in result["diagnostics"]
            )
        )

    def test_stage_duplicate_identity_marks_all_occurrences_as_conflict(self):
        row = self._proposal()
        staged = batch.import_revision_proposals(
            str(self._write_proposal(row)),
            stage=True,
        )
        duplicate_path = self.root / "duplicate.jsonl"
        duplicate_path.write_text(
            json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        duplicate = batch.import_revision_proposals(str(duplicate_path), stage=True)

        self.assertEqual(duplicate["conflict_count"], 2)
        self.assertEqual(duplicate["selectable_count"], 0)
        self.assertEqual(
            {candidate["status"] for candidate in duplicate["candidates"]},
            {revision_selection.STATUS_CONFLICT},
        )
        self.assertEqual(staged["status"], "staged")


if __name__ == "__main__":
    unittest.main()
