"""Matrix coverage for doctor layout status and recommendations."""
from __future__ import annotations

import unittest
from unittest import mock

import doctor_recommendations as doctor_rec
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
        codes = doctor_rec.doctor_recommendation_codes(recommendations)

        self.assertIn(doctor_rec.SWITCH_TO_WORK, codes)
        self.assertIn(doctor_rec.BOOTSTRAP_WORK, codes)
        self.assertNotIn(doctor_rec.GENERATE_TEMPLATE, codes)
        self.assertNotIn(doctor_rec.INSTALL_SDK_GENERATE_TEMPLATE, codes)

    def test_switch_to_work_without_original_only_recommends_switch(self):
        report = _layout_report(
            base_dir="C:/Games/Example",
            work_dir="C:/Games/Example/work",
            can_generate_template=True,
            layout_status="switch_to_work",
        )
        recommendations = batch_mod.collect_doctor_recommendations(report)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["code"], doctor_rec.SWITCH_TO_WORK)
        self.assertEqual(
            recommendations[0]["params"]["work_dir"],
            "C:/Games/Example/work",
        )

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
        codes = doctor_rec.doctor_recommendation_codes(recommendations)

        self.assertIn(doctor_rec.SWITCH_TO_WORK, codes)
        self.assertNotIn(doctor_rec.BOOTSTRAP_WORK, codes)

    def test_work_root_without_tl_recommends_generate_template_when_available(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            can_generate_template=True,
            prepare_enabled=True,
            layout_status="attention",
        )
        recommendations = batch_mod.collect_doctor_recommendations(report)
        codes = doctor_rec.doctor_recommendation_codes(recommendations)

        self.assertEqual(codes, [doctor_rec.GENERATE_TEMPLATE])
        self.assertNotIn(doctor_rec.SWITCH_TO_WORK, codes)

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
        self.assertEqual(
            recommendations[0]["code"],
            doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE,
        )
        self.assertNotIn(doctor_rec.START_PENDING_BATCH, doctor_rec.doctor_recommendation_codes(recommendations))

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

        # Production gate used by collect_doctor_report(): required prep clears workflow_state.
        finalized = batch_mod.finalize_doctor_actionable_signals(report)

        self.assertEqual(len(finalized["recommendations"]), 1)
        self.assertEqual(finalized["recommendations"][0]["code"], doctor_rec.BOOTSTRAP_RAG)
        self.assertNotIn(
            doctor_rec.START_PENDING_BATCH,
            doctor_rec.doctor_recommendation_codes(finalized["recommendations"]),
        )
        # Without the gate this scenario would emit start_incremental_batch.
        self.assertEqual(
            batch_mod.collect_doctor_workflow_state(report),
            doctor_rec.START_INCREMENTAL_BATCH,
        )
        self.assertEqual(finalized["workflow_state"], "")

    def test_ready_incremental_translation_has_no_recommendation(self):
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

        self.assertEqual(recommendations, [])
        self.assertFalse(
            set(doctor_rec.doctor_recommendation_codes(recommendations))
            & doctor_rec.WORKFLOW_STATE_CODES
        )
        self.assertEqual(
            batch_mod.collect_doctor_workflow_state(report),
            doctor_rec.START_INCREMENTAL_BATCH,
        )

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

        self.assertEqual(recommendations, [])
        self.assertEqual(
            batch_mod.collect_doctor_workflow_state(report),
            doctor_rec.SUBSTANTIALLY_COMPLETE,
        )
        self.assertNotIn(doctor_rec.ENABLE_RAG_FOR_CONSISTENCY, doctor_rec.doctor_recommendation_codes(recommendations))

    def test_ready_project_without_pending_lines_has_no_recommendation(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=20,
            layout_status="ready",
        )
        report["pending_task_count"] = 0
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

        self.assertEqual(recommendations, [])
        self.assertEqual(
            batch_mod.collect_doctor_workflow_state(report),
            doctor_rec.NO_PENDING_LINES,
        )

    def test_ready_new_translation_has_pending_workflow_state(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            rpy_files=20,
            layout_status="ready",
        )
        report["pending_task_count"] = 240
        report["counts"] = {
            "rpy_files": 20,
            "translate_blocks": 240,
            "old_lines": 0,
        }

        self.assertEqual(
            batch_mod.collect_doctor_workflow_state(report),
            doctor_rec.START_PENDING_BATCH,
        )

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
        self.assertEqual(recommendations[0]["code"], doctor_rec.ENABLE_RAG_FOR_CONSISTENCY)
        # Optional tip must not suppress workflow_state.
        self.assertFalse(doctor_rec.recommendations_block_workflow_state(recommendations))

    def test_required_source_index_and_optional_enable_rag_coexist(self):
        """Required prep and optional tips are listed together (required first)."""
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
                "store_exists": False,
                "source_segments": 0,
                "expected_segments": 3000,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)
        codes = doctor_rec.doctor_recommendation_codes(recommendations)

        self.assertEqual(
            codes,
            [
                doctor_rec.BOOTSTRAP_SOURCE_INDEX,
                doctor_rec.ENABLE_RAG_FOR_CONSISTENCY,
            ],
        )
        # Required prep still clears workflow_state; optional tip alone would not.
        finalized = batch_mod.finalize_doctor_actionable_signals(report)
        self.assertEqual(finalized["workflow_state"], "")
        self.assertTrue(doctor_rec.recommendations_block_workflow_state(recommendations))

    def test_required_source_index_and_required_rag_bootstrap_coexist(self):
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
                "store_exists": False,
                "source_segments": 0,
                "expected_segments": 3000,
            },
        }

        codes = doctor_rec.doctor_recommendation_codes(
            batch_mod.collect_doctor_recommendations(report)
        )
        self.assertEqual(
            codes,
            [
                doctor_rec.BOOTSTRAP_SOURCE_INDEX,
                doctor_rec.BOOTSTRAP_RAG,
            ],
        )

    def test_recommendations_never_emit_workflow_state_codes(self):
        """Phase markers live only in workflow_state, not in the recommendation list."""
        cases = [
            # required source index
            {
                "pending_task_count": 100,
                "context_status": {
                    "source_index": {
                        "enabled": True,
                        "store_exists": False,
                        "source_segments": 0,
                        "expected_segments": 100,
                    }
                },
            },
            # ready incremental (no recs)
            {
                "pending_task_count": 240,
                "counts": {
                    "rpy_files": 20,
                    "translate_blocks": 12000,
                    "old_lines": 120,
                    "new_lines": 120,
                },
                "context_status": {
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
                },
            },
            # no pending
            {
                "pending_task_count": 0,
                "context_status": {
                    "rag": {"enabled": False},
                    "source_index": {
                        "enabled": True,
                        "store_exists": True,
                        "source_segments": 100,
                        "expected_segments": 100,
                    },
                },
            },
        ]
        for case in cases:
            report = _layout_report(
                base_dir="C:/Games/Example/work",
                rpy_files=20,
                layout_status="ready",
            )
            report.update(case)
            codes = set(
                doctor_rec.doctor_recommendation_codes(
                    batch_mod.collect_doctor_recommendations(report)
                )
            )
            self.assertFalse(
                codes & doctor_rec.WORKFLOW_STATE_CODES,
                msg=f"workflow codes leaked into recommendations: {codes & doctor_rec.WORKFLOW_STATE_CODES}",
            )

    def test_blocking_recommendation_suppresses_low_priority_recommendations(self):
        report = _layout_report(
            base_dir="C:/Games/Example/work",
            original_game_dir="C:/Games/Example/original/game",
            work_bootstrap_allowed=True,
            layout_status="attention",
        )
        # 配置已启用但未就绪的 RAG 和原文索引，如果在未抑制状态下会触发建议
        report["context_status"] = {
            "rag": {
                "enabled": True,
                "store_exists": False,
                "history_records": 0,
                "bootstrap_on_build": False,
            },
            "source_index": {
                "enabled": True,
                "store_exists": False,
                "source_segments": 0,
                "expected_segments": 100,
            },
        }

        recommendations = batch_mod.collect_doctor_recommendations(report)
        codes = doctor_rec.doctor_recommendation_codes(recommendations)

        # 应当且仅应当推荐 BOOTSTRAP_WORK
        self.assertIn(doctor_rec.BOOTSTRAP_WORK, codes)
        # 低优先级的 RAG 与原文索引预建必须被抑制
        self.assertNotIn(doctor_rec.BOOTSTRAP_RAG, codes)
        self.assertNotIn(doctor_rec.BOOTSTRAP_SOURCE_INDEX, codes)


if __name__ == "__main__":
    unittest.main()
