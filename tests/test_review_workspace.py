"""Real index/decision/proposal chain for ordinary item review (#427 S2)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import review_index
import review_workspace
import revision_proposals
import gemini_translate_batch as batch

from tests.test_review_index import _write_corpus, _write_findings


class ReviewWorkspaceTests(unittest.TestCase):
    def test_plain_item_draft_exports_full_proposal_with_distinct_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            self.assertEqual(len(entries), 3)
            self.assertFalse(entries[1]["has_issues"])
            index_path = manifest["_manifest_path"]
            proposed = "完整译文" * 200
            review_workspace.save_draft(
                index_path, manifest, entries, "occ-2", proposed, "逐句核对"
            )
            path = review_workspace.export_proposals(
                index_path, manifest, entries, ["occ-2"]
            )
            rows = revision_proposals.load_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["occurrence_id"], "occ-2")
            self.assertEqual(rows[0]["proposed_translation"], proposed)
            self.assertEqual(rows[0]["source"], "Hello")
            self.assertEqual(rows[0]["selected"], True)

    def test_decision_restores_and_changed_evidence_requires_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            index_path = manifest["_manifest_path"]
            review_workspace.save_decision(
                index_path, entries, "occ-1", "ignored", "reviewer", "已核对"
            )
            _, restored = review_index.load_review_index(index_path)
            self.assertEqual(restored[0]["review"]["lifecycle"], "ignored")
            review_workspace.save_draft(
                index_path, manifest, restored, "occ-1", "您好", "语气"
            )
            _write_corpus(root, current_translation="你好，朋友")
            changed, changed_entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            self.assertEqual(changed_entries[0]["review"]["lifecycle"], "needs_recheck")
            draft = review_workspace.load_drafts(index_path, changed)["occ-1"]
            self.assertFalse(review_workspace.draft_is_current(draft, changed_entries[0]))
            with self.assertRaisesRegex(ValueError, "证据已变化"):
                review_workspace.export_proposals(
                    index_path, changed, changed_entries, ["occ-1"]
                )

    def test_cross_project_and_stale_corpus_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            with self.assertRaisesRegex(ValueError, "项目不一致"):
                review_workspace.open_workspace(
                    str(corpus), expected_game_root=str(root / "other")
                )
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            index_path = manifest["_manifest_path"]
            review_workspace.save_draft(
                index_path, manifest, entries, "occ-1", "你好啊", "风格"
            )
            corpus_jsonl = corpus / "revision_corpus.jsonl"
            corpus_jsonl.write_text(corpus_jsonl.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "语料内容已变化"):
                review_workspace.export_proposals(
                    index_path, manifest, entries, ["occ-1"]
                )

    def test_filter_and_page_scope(self) -> None:
        rows = [
            {"occurrence_id": str(i), "source": "same", "current_translation": "译文",
             "file_rel_path": "a", "speaker_id": "A", "quality_findings": [],
             "review": {"lifecycle": "open"}}
            for i in range(110)
        ]
        rows[55]["quality_findings"] = [{"severity": "high"}]
        self.assertEqual(review_workspace.filter_page(rows, page=1)[0], 110)
        self.assertEqual(len(review_workspace.filter_page(rows, page=1)[1]), 50)
        self.assertEqual(len(review_workspace.filter_page(rows, page=2)[1]), 10)
        total, page = review_workspace.filter_page(
            rows, findings="yes", severity="high", speaker="a", query="same"
        )
        self.assertEqual((total, page[0]["occurrence_id"]), (1, "55"))

    def test_existing_index_keeps_attached_finding_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            findings = _write_findings(root)
            index_dir = root / "index"
            review_index.build_review_index(
                corpus, quality_findings_path=findings, output_dir=index_dir
            )
            _manifest, entries = review_workspace.open_workspace(
                str(index_dir / "review_index_manifest.json"),
                expected_game_root=str(root / "game"),
            )
            self.assertEqual(len(entries[0]["quality_findings"]), 1)
            self.assertEqual(entries[1]["quality_findings"], [])

    def test_foreign_decision_and_duplicate_selection_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_a = _write_corpus(root / "a", project_slug="a")
            corpus_b = _write_corpus(root / "b", project_slug="b")
            manifest_a, entries_a = review_workspace.open_workspace(
                str(corpus_a), expected_game_root=str(root / "a" / "game")
            )
            manifest_b, _entries_b = review_workspace.open_workspace(
                str(corpus_b), expected_game_root=str(root / "b" / "game")
            )
            with self.assertRaisesRegex(ValueError, "其他项目"):
                review_workspace.save_decision(
                    manifest_b["_manifest_path"], entries_a, "occ-1", "resolved", "reviewer", ""
                )
            with self.assertRaisesRegex(ValueError, "不重复"):
                review_workspace.export_proposals(
                    manifest_a["_manifest_path"], manifest_a, entries_a,
                    ["occ-1", "occ-1"],
                )

    def test_generated_plain_item_uses_real_revision_import_preview(self) -> None:
        from tests.test_revision_proposals import RevisionProposalImportTests

        fixture = RevisionProposalImportTests()
        fixture.setUp()
        try:
            corpus = fixture.root / "corpus"
            batch.run_revision_corpus_export(output_dir=str(corpus))
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(fixture.root),
                expected_tl_dir=str(fixture.tl_dir),
            )
            self.assertEqual(len(entries), 1)
            self.assertFalse(entries[0]["has_issues"])
            index_path = manifest["_manifest_path"]
            review_workspace.save_draft(
                index_path, manifest, entries, entries[0]["occurrence_id"],
                "您好，{name}", "语气更自然",
            )
            proposal_path = review_workspace.export_proposals(
                index_path, manifest, entries, [entries[0]["occurrence_id"]]
            )
            original = fixture.rpy.read_bytes()
            result = batch.import_revision_proposals(
                proposal_path,
                corpus_manifest_path=str(corpus / "revision_corpus_manifest.json"),
            )
            self.assertEqual(result["status"], "previewed")
            self.assertTrue(result["manifest"]["proposal_import"]["writeback_eligible"])
            self.assertEqual(fixture.rpy.read_bytes(), original)
        finally:
            fixture.tearDown()
