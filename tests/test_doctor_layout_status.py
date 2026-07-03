"""Matrix coverage for doctor layout status and recommendations."""
from __future__ import annotations

import unittest
from unittest import mock

import gemini_translate_batch as batch_mod


def _layout_report(
    *,
    base_dir: str,
    work_dir: str = "",
    rpy_files: int = 0,
    original_game_dir: str = "",
    can_generate_template: bool = False,
    work_bootstrap_allowed: bool = False,
    prepare_enabled: bool = True,
    layout_status: str = "",
) -> dict:
    if not work_dir:
        work_dir = base_dir if base_dir.endswith("/work") or base_dir.endswith("\\work") else f"{base_dir}/work"
    report = {
        "base_dir": base_dir,
        "work_dir": work_dir,
        "counts": {"rpy_files": rpy_files},
        "original_game_dir": original_game_dir,
        "can_generate_template": can_generate_template,
        "work_bootstrap_allowed": work_bootstrap_allowed,
        "prepare_enabled": prepare_enabled,
        "mode": "blocked_missing_template",
    }
    if layout_status:
        report["layout_status"] = layout_status
    return report


class DoctorLayoutStatusMatrixTests(unittest.TestCase):
    def test_project_root_without_work_original_or_tl_is_failed(self):
        report = _layout_report(base_dir="C:/Games/Example")
        with (
            mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=True),
            mock.patch("os.path.isdir", return_value=False),
        ):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "failed")

    def test_project_root_with_empty_work_dir_is_switch_to_work(self):
        report = _layout_report(base_dir="C:/Games/Example")

        def isdir(path):
            return str(path).replace("\\", "/").endswith("/work")

        with (
            mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=True),
            mock.patch("os.path.isdir", side_effect=isdir),
        ):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "switch_to_work")

    def test_project_root_with_original_is_switch_to_work(self):
        report = _layout_report(
            base_dir="C:/Games/Example",
            original_game_dir="C:/Games/Example/original/game",
        )
        with (
            mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=True),
            mock.patch("os.path.isdir", return_value=False),
        ):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "switch_to_work")

    def test_project_root_with_tl_is_switch_to_work(self):
        report = _layout_report(base_dir="C:/Games/Example", rpy_files=2)
        with (
            mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=True),
            mock.patch("os.path.isdir", return_value=False),
        ):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "switch_to_work")

    def test_work_root_with_tl_is_ready(self):
        report = _layout_report(base_dir="C:/Games/Example/work", rpy_files=1)
        self.assertEqual(batch_mod.assess_doctor_layout_status(report), "ready")

    def test_work_root_with_game_content_but_no_tl_is_attention(self):
        report = _layout_report(base_dir="C:/Games/Example/work")

        def isdir(path):
            normalized = str(path).replace("\\", "/")
            return normalized.endswith("/work") or normalized.endswith("/work/game")

        with mock.patch("os.path.isdir", side_effect=isdir):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "attention")

    def test_work_root_with_template_generation_is_attention(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            can_generate_template=True,
        )
        with mock.patch("os.path.isdir", return_value=True):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "attention")

    def test_work_root_with_original_is_attention(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            original_game_dir="C:/Games/Example/original/game",
        )
        with mock.patch("os.path.isdir", return_value=True):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "attention")

    def test_work_root_empty_without_paths_is_failed(self):
        report = _layout_report(base_dir="C:/Games/Example/work")

        def isdir(path):
            normalized = str(path).replace("\\", "/")
            return normalized.endswith("/work")

        with (
            mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=True),
            mock.patch("os.path.isdir", side_effect=isdir),
        ):
            self.assertEqual(batch_mod.assess_doctor_layout_status(report), "failed")


class DoctorRecommendationMatrixTests(unittest.TestCase):
    def test_switch_to_work_only_recommends_switch_and_bootstrap(self):
        report = _layout_report(
            base_dir="C:/Games/Example",
            work_dir="C:/Games/Example/work",
            original_game_dir="C:/Games/Example/original/game",
            can_generate_template=True,
            work_bootstrap_allowed=True,
            prepare_enabled=True,
            layout_status="switch_to_work",
        )
        recommendations = batch_mod.collect_doctor_recommendations(report)
        joined = " ".join(recommendations)

        self.assertIn("switch to C:/Games/Example/work", joined)
        self.assertIn("bootstrap-work", joined)
        self.assertNotIn("gemini_translate_batch.py build", joined)
        self.assertNotIn("Ren'Py SDK", joined)

    def test_switch_to_work_without_original_only_recommends_switch(self):
        report = _layout_report(
            base_dir="C:/Games/Example",
            work_dir="C:/Games/Example/work",
            can_generate_template=True,
            layout_status="switch_to_work",
        )
        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("switch to C:/Games/Example/work", recommendations[0])

    def test_switch_to_work_with_existing_work_does_not_recommend_bootstrap(self):
        report = _layout_report(
            base_dir="C:/Games/Example/original",
            work_dir="C:/Games/Example/work",
            original_game_dir="C:/Games/Example/original/game",
            can_generate_template=True,
            work_bootstrap_allowed=False,
            layout_status="switch_to_work",
        )
        report["work_exists"] = True
        report["work_empty"] = False
        recommendations = batch_mod.collect_doctor_recommendations(report)
        joined = " ".join(recommendations)

        self.assertIn("switch to C:/Games/Example/work", joined)
        self.assertNotIn("bootstrap-work", joined)

    def test_work_root_without_tl_recommends_build_when_template_available(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            can_generate_template=True,
            prepare_enabled=True,
            layout_status="attention",
        )
        recommendations = batch_mod.collect_doctor_recommendations(report)
        joined = " ".join(recommendations)

        self.assertIn("gemini_translate_batch.py build", joined)
        self.assertNotIn("switch to", joined)

    def test_resolve_source_index_expected_segments_scans_when_metadata_missing(self):
        with (
            mock.patch.object(batch_mod, 'collect_source_segments_for_jobs', return_value=[{}, {}, {}]),
            mock.patch.object(batch_mod.legacy, 'TL_DIR', 'C:/Games/Example/work/game/tl/schinese'),
            mock.patch.object(batch_mod.os.path, 'isdir', return_value=True),
        ):
            store = mock.Mock()
            expected, error = batch_mod._resolve_source_index_expected_segments(store, {})

        self.assertEqual(expected, 3)
        self.assertEqual(error, '')
        store.set_metadata.assert_called_once_with(last_scanned_total=3)

    def test_incomplete_source_index_blocks_batch_translation_recommendation(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=3,
            layout_status="ready",
        )
        report["pending_task_count"] = 8500
        report["context_status"] = {
            "source_index": {
                "enabled": True,
                "store_exists": True,
                "source_segments": 4200,
                "expected_segments": 12000,
            }
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("incomplete", recommendations[0])
        self.assertNotIn("Pending translation lines", recommendations[0])

    def test_empty_rag_blocks_batch_translation_when_enabled(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=20,
            layout_status="ready",
        )
        report["pending_task_count"] = 240
        report["counts"] = {
            "rpy_files": 20,
            "translate_blocks": 12000,
            "old_lines": 120,
            "new_lines": 120,
        }
        report["context_status"] = {
            "rag": {
                "enabled": True,
                "store_exists": False,
                "history_records": 0,
                "bootstrap_on_build": False,
            },
            "source_index": {
                "enabled": True,
                "store_exists": True,
                "source_segments": 2979,
                "expected_segments": 2979,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("bootstrap-rag", recommendations[0])
        self.assertNotIn("Pending translation lines", recommendations[0])

    def test_incremental_translation_uses_incremental_recommendation(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=20,
            layout_status="ready",
        )
        report["pending_task_count"] = 240
        report["counts"] = {
            "rpy_files": 20,
            "translate_blocks": 12000,
            "old_lines": 120,
            "new_lines": 120,
        }
        report["context_status"] = {
            "rag": {
                "enabled": True,
                "store_exists": True,
                "history_records": 5200,
            },
            "source_index": {
                "enabled": True,
                "store_exists": True,
                "source_segments": 2979,
                "expected_segments": 2979,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("Incremental translation", recommendations[0])

    def test_mostly_complete_project_without_rag_does_not_require_bootstrap(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=48,
            layout_status="ready",
        )
        report["pending_task_count"] = 45
        report["counts"] = {
            "rpy_files": 48,
            "translate_blocks": 120000,
            "old_lines": 800,
            "new_lines": 800,
            "commented_original_lines": 85000,
        }
        report["context_status"] = {
            "rag": {"enabled": False},
            "source_index": {
                "enabled": True,
                "store_exists": True,
                "source_segments": 12000,
                "expected_segments": 12000,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("substantially complete", recommendations[0])
        self.assertNotIn("RAG disabled", recommendations[0])

    def test_existing_translations_without_rag_recommends_enable_rag(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=20,
            layout_status="ready",
        )
        report["pending_task_count"] = 240
        report["counts"] = {
            "rpy_files": 20,
            "translate_blocks": 12000,
            "old_lines": 120,
            "new_lines": 120,
        }
        report["context_status"] = {
            "rag": {"enabled": False},
            "source_index": {
                "enabled": True,
                "store_exists": True,
                "source_segments": 2979,
                "expected_segments": 2979,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("RAG disabled", recommendations[0])


if __name__ == "__main__":
    unittest.main()