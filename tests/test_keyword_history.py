import json
import tempfile
import unittest
from pathlib import Path

import keyword_history


class KeywordHistoryEvidenceTests(unittest.TestCase):
    def _row(
        self,
        identity,
        file_rel_path,
        line_number,
        source,
        current_translation,
        start=0,
    ):
        return {
            "identity_v2": identity,
            "occurrence_id": identity,
            "file_rel_path": file_rel_path,
            "display_line": line_number,
            "locator": {"line_number": line_number, "start": start},
            "source": source,
            "current_translation": current_translation,
        }

    def test_first_occurrence_is_stable_across_input_order_and_keeps_identity(self):
        rows = [
            self._row(
                "chapter02/dialogue:2",
                "chapter02/dialogue.rpy",
                2,
                "Noah waved.",
                "诺亚挥了挥手。",
            ),
            self._row(
                "chapter01/intro:8",
                "chapter01/intro.rpy",
                8,
                "Noah, come here.",
                "诺亚，过来。",
            ),
            self._row(
                "chapter03/dialogue:4",
                "chapter03/dialogue.rpy",
                4,
                "Noah, wait!",
                "诺亚啊，等等！",
            ),
        ]
        candidate = {
            "source": "Noah",
            "suggested_target": "诺亚",
            "category": "character",
        }

        first = keyword_history.build_keyword_history_evidence(candidate, rows)
        shuffled = keyword_history.build_keyword_history_evidence(candidate, list(reversed(rows)))

        self.assertEqual(first, shuffled)
        self.assertEqual(first["status"], keyword_history.STATUS_AMBIGUOUS)
        self.assertEqual(first["first_occurrence"]["identity_v2"], "chapter01/intro:8")
        self.assertEqual(first["first_occurrence"]["current_translation"], "诺亚，过来。")
        self.assertIn("multiple_historical_translations", first["conflict_codes"])

    def test_plural_and_case_variants_are_human_review_evidence(self):
        rows = [
            self._row("id1", "a.rpy", 1, "The LIGHTS are on.", "灯亮着。"),
            self._row("id2", "b.rpy", 2, "a LIGHT spell", "一个光系法术"),
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "light", "suggested_target": "光", "category": "term"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_AMBIGUOUS)
        self.assertIn("case_variant", evidence["conflict_codes"])
        self.assertIn("plural_variant", evidence["conflict_codes"])

    def test_interpolation_only_match_is_not_history_evidence(self):
        rows = [
            self._row(
                "id1",
                "a.rpy",
                3,
                "Hello {name}!",
                "你好，[name]！",
            ),
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "name", "suggested_target": "名字", "category": "term"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_UNMATCHED)
        self.assertIsNone(evidence["first_occurrence"])
        self.assertIn("only_interpolation_match", evidence["conflict_codes"])

    def test_word_boundaries_avoid_substring_false_positive(self):
        rows = [self._row("id1", "a.rpy", 1, "cart arrived", "购物车到了")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "art", "suggested_target": "艺术", "category": "concept"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_UNMATCHED)
        self.assertEqual(evidence["match_count"], 0)

    def test_direct_first_translation_conflict_is_explained(self):
        rows = [self._row("id1", "a.rpy", 1, "Void Gate", "虚空门")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "Void Gate", "suggested_target": "星门", "category": "place"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_CONFLICT)
        self.assertIn("candidate_target_conflict", evidence["conflict_codes"])
        self.assertEqual(evidence["first_occurrence"]["current_translation"], "虚空门")

    def test_chinese_substring_alignment_is_human_review_only(self):
        rows = [
            self._row(
                "id1",
                "a.rpy",
                1,
                "cart arrived",
                "购物车到了",
            )
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "cart", "suggested_target": "车", "category": "item"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_AMBIGUOUS)
        self.assertTrue(evidence["review_required"])
        self.assertIn("translation_alignment_unknown", evidence["conflict_codes"])

    def test_exact_chinese_translation_remains_safe_alignment(self):
        rows = [self._row("id1", "a.rpy", 1, "cart", "车")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "cart", "suggested_target": "车", "category": "item"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_CONSISTENT)
        self.assertFalse(evidence["review_required"])
        self.assertEqual(evidence["conflict_codes"], [])

    def test_load_corpus_items_accepts_manifest_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_path = root / "revision_corpus.jsonl"
            corpus_path.write_text(
                json.dumps({"identity_v2": "id1"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "revision_corpus_manifest.json"
            manifest_path.write_text(
                json.dumps({"paths": {"jsonl": "revision_corpus.jsonl"}}, ensure_ascii=False),
                encoding="utf-8",
            )

            rows = keyword_history.load_corpus_items(str(manifest_path))

        self.assertEqual(rows, [{"identity_v2": "id1"}])

    def test_mixed_interpolation_match_is_human_review_evidence(self):
        rows = [
            self._row(
                "id1",
                "a.rpy",
                3,
                "Hello {name}! My name is Nova.",
                "名字",
            )
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "name", "suggested_target": "名字", "category": "term"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_AMBIGUOUS)
        self.assertTrue(evidence["review_required"])
        self.assertIn("interpolation_match_ignored", evidence["conflict_codes"])
        self.assertEqual(
            evidence["occurrences"][0]["matched_text"],
            "name",
        )

        match_result = keyword_history.match_keyword_in_source(
            rows[0]["source"],
            "name",
        )
        self.assertFalse(match_result["interpolation_only"])
        self.assertTrue(match_result["interpolation_match_ignored"])

    def test_attach_batch_collector_matches_individual_builds(self):
        rows = [
            self._row("id1", "a.rpy", 1, "Void Gate", "虚空门"),
            self._row(
                "id2",
                "b.rpy",
                2,
                "The Crystal Key gleams.",
                "水晶钥匙闪闪发光。",
            ),
            self._row(
                "id3",
                "c.rpy",
                3,
                "Hello {name}! My name is Nova.",
                "名字",
            ),
            self._row(
                "id4",
                "d.rpy",
                4,
                "Noah’s compass is ready.",
                "诺亚的罗盘准备好了。",
            ),
            self._row("id5", "e.rpy", 5, "AR", "AR"),
        ]
        candidates = [
            {"source": "Void Gate", "suggested_target": "虚空门"},
            {"source": "Crystal Key", "suggested_target": "水晶钥匙"},
            {"source": "name", "suggested_target": "名字"},
            {"source": "Noah", "suggested_target": "诺亚"},
            {"source": "AR", "suggested_target": "AR"},
        ]

        enriched = keyword_history.attach_keyword_history_evidence(
            candidates,
            rows,
        )

        for candidate, row in zip(candidates, enriched):
            with self.subTest(source=candidate["source"]):
                self.assertEqual(
                    row["history_evidence"],
                    keyword_history.build_keyword_history_evidence(candidate, rows),
                )

    def test_consistent_evidence_validator_rejects_internally_mismatched_payload(self):
        rows = [self._row("id1", "a.rpy", 1, "Void Gate", "虚空门")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "Void Gate", "suggested_target": "虚空门"},
            rows,
        )
        self.assertTrue(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

        changed_first = dict(evidence["first_occurrence"])
        changed_first["line_number"] = 99
        evidence["first_occurrence"] = changed_first
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "Void Gate", "suggested_target": "虚空门"},
            rows,
        )
        evidence["translations"] = ["另一译法"]
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

    def test_consistent_validator_rejects_multiple_reported_translations(self):
        rows = [
            self._row("id1", "a.rpy", 1, "Void Gate", "虚空门"),
            self._row("id2", "b.rpy", 2, "Void Gate", "星门"),
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "Void Gate", "suggested_target": "虚空门"},
            rows,
        )
        self.assertEqual(evidence["status"], keyword_history.STATUS_AMBIGUOUS)

        # Hand-edited payload that claims consistent while carrying two
        # historical translations must fail closed.
        evidence["status"] = keyword_history.STATUS_CONSISTENT
        evidence["review_required"] = False
        evidence["translations"] = ["虚空门", "星门"]
        evidence["conflict_codes"] = []
        evidence["conflict_reasons"] = []
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

    def test_consistent_validator_uses_export_alignment_rule_for_non_chinese_target(self):
        rows = [
            self._row(
                "id1",
                "a.rpy",
                1,
                "The STAR guild arrived.",
                "STAR 的公会到了。",
            )
        ]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "STAR", "suggested_target": "STAR"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_CONSISTENT)
        self.assertTrue(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

        evidence["candidate_target"] = "STARLIGHT"
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

    def test_consistent_validator_rejects_target_mismatching_unique_translation(self):
        rows = [self._row("id1", "a.rpy", 1, "Void Gate", "虚空门")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "Void Gate", "suggested_target": "虚空门"},
            rows,
        )
        self.assertTrue(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

        evidence["candidate_target"] = "星门"
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

    def test_preserve_row_is_review_evidence_for_preserve_candidate(self):
        rows = [self._row("id1", "a.rpy", 1, "AR", "AR")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "AR", "suggested_target": "AR", "category": "term"},
            rows,
        )

        self.assertEqual(
            evidence["status"],
            keyword_history.STATUS_PRESERVE_EVIDENCE,
        )
        self.assertTrue(evidence["review_required"])
        self.assertIn("preserve_evidence", evidence["conflict_codes"])
        self.assertIn(
            "保留不译",
            evidence["conflict_reasons"][0],
        )
        self.assertFalse(
            keyword_history.is_complete_consistent_history_evidence(evidence)
        )

    def test_preserve_row_with_translated_candidate_is_a_conflict(self):
        rows = [self._row("id1", "a.rpy", 1, "AR", "AR")]
        evidence = keyword_history.build_keyword_history_evidence(
            {"source": "AR", "suggested_target": "增强现实", "category": "term"},
            rows,
        )

        self.assertEqual(evidence["status"], keyword_history.STATUS_CONFLICT)
        self.assertIn("candidate_target_conflict", evidence["conflict_codes"])

    def test_fixture_covers_possessive_and_vocative_forms(self):
        fixture_path = Path(__file__).parent / "fixtures" / "keyword_history_forms.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                evidence = keyword_history.build_keyword_history_evidence(
                    case["candidate"],
                    case["rows"],
                )
                expected = case["expected"]
                self.assertEqual(evidence["status"], expected["status"])
                self.assertEqual(
                    evidence["occurrences"][0]["match_kind"],
                    expected["match_kind"],
                )
                reason_code = expected["reason_code"]
                if reason_code:
                    self.assertIn(reason_code, evidence["conflict_codes"])
                else:
                    self.assertEqual(evidence["conflict_codes"], [])


if __name__ == "__main__":
    unittest.main()
