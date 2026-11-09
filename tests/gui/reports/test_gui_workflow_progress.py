import unittest

from gui_qt.reports.workflow_progress import (
    create_workflow_progress_state,
    update_workflow_progress_from_line,
)


class GuiWorkflowProgressTests(unittest.TestCase):
    def test_keyword_pending_queue_excludes_restored_results_and_saved_errors(self):
        state = create_workflow_progress_state('sync_keywords')
        state = update_workflow_progress_from_line(
            'Sync keyword queue: total=768; reused=655; pending=113; failed=112; incomplete=1.', state)
        self.assertEqual((state.current, state.total), (0, 113))
        self.assertIn('已复用有效结果：655', '\n'.join(state.facts))
        self.assertIn('任务总量：768', '\n'.join(state.facts))
        state = update_workflow_progress_from_line(
            'Sync keyword saved: 768/768 requests; resumed=768.', state)
        self.assertEqual((state.current, state.total), (0, 113))
        self.assertIn('已保存处理记录：768/768', state.facts)
        for line in ('[1/113] failed-key', 'Sync keyword phase: waiting failed-key',
                     '  targeted retry: 40 items',
                     'Sync keyword saved: 768/768 requests; resumed=768.'):
            state = update_workflow_progress_from_line(line, state)
            self.assertEqual(state.current, 0)
        line = 'Sync keyword progress: 1/113 pending blocks processed.'
        state = update_workflow_progress_from_line(line, state)
        self.assertEqual((state.current, state.total), (1, 113))
        self.assertIn('本轮待补已处理 1/113', state.label)
        self.assertFalse(any('failed-key' in fact for fact in state.facts))
        self.assertEqual(update_workflow_progress_from_line(line, state), state)
        self.assertEqual(update_workflow_progress_from_line(
            'Sync keyword progress: 1/768 pending blocks processed.', state), state)

    def test_keyword_fully_restored_queue_has_no_zero_over_zero_percentage(self):
        state = create_workflow_progress_state('sync_keywords')
        state = update_workflow_progress_from_line(
            'Sync keyword queue: total=768; reused=768; pending=0; failed=0; incomplete=0.', state)
        state = update_workflow_progress_from_line(
            'Sync keyword saved: 768/768 requests; resumed=768.', state)
        self.assertEqual(state.total, 0)
        self.assertTrue(state.indeterminate)
        self.assertIn('正在生成报告', state.label)
        self.assertNotIn('0/0', state.label)

    def test_keywords_show_scan_then_saved_count_without_counting_dispatch_as_done(self):
        state = create_workflow_progress_state("sync_keywords")
        self.assertTrue(state.visible)
        self.assertTrue(state.indeterminate)
        state = update_workflow_progress_from_line(
            "Sync keyword saved: 3/10 requests; resumed=3.", state)
        self.assertEqual((state.current, state.total), (0, 10))
        self.assertIn("载入历史处理记录：3", state.facts)
        state = update_workflow_progress_from_line("[4/10] fixture-key", state)
        self.assertEqual(state.current, 0)
        state = update_workflow_progress_from_line(
            "Sync keyword phase: waiting fixture-key", state)
        self.assertIn("正在等待关键词服务响应：fixture-key", state.facts)
        state = update_workflow_progress_from_line(
            "Sync keyword saved: 4/10 requests; resumed=3.", state)
        self.assertEqual(state.current, 4)
        self.assertFalse(state.indeterminate)

    def test_keyword_resume_full_record_count_does_not_fill_current_run_progress(self):
        state = create_workflow_progress_state("sync_keywords")
        state = update_workflow_progress_from_line(
            "Sync keyword saved: 768/768 requests; resumed=768.", state)
        self.assertEqual((state.current, state.total), (0, 768))
        self.assertIn("本轮已处理 0/768", state.label)
        self.assertIn("已保存处理记录：768/768", state.facts)

        for index in range(1, 78):
            state = update_workflow_progress_from_line(f"[{index}/768] cached-{index}", state)
            self.assertEqual(state.current, index - 1)
            state = update_workflow_progress_from_line(
                "Sync keyword saved: 768/768 requests; resumed=768.", state)
            self.assertEqual(state.current, index)

        state = update_workflow_progress_from_line("[78/768] retry-key", state)
        state = update_workflow_progress_from_line("Sync keyword phase: waiting retry-key", state)
        self.assertEqual(state.current, 77)
        self.assertIn("正在等待关键词服务响应：retry-key", state.facts)
        state = update_workflow_progress_from_line("  finish_reason: stop", state)
        self.assertEqual(state.current, 77)
        state = update_workflow_progress_from_line(
            "Sync keyword saved: 768/768 requests; resumed=768.", state)
        self.assertEqual(state.current, 78)
        self.assertIn("本轮已处理 78/768", state.label)
        self.assertFalse(any("retry-key" in fact for fact in state.facts))

    def test_keyword_failed_row_advances_processing_only_after_saved(self):
        state = create_workflow_progress_state("sync_keywords")
        for line in ("Sync keyword saved: 2/2 requests; resumed=2.",
                     "[1/2] failed-key", "  error: service temporarily unavailable"):
            state = update_workflow_progress_from_line(line, state)
        self.assertEqual(state.current, 0)
        line = "Sync keyword saved: 2/2 requests; resumed=2."
        state = update_workflow_progress_from_line(line, state)
        self.assertEqual(state.current, 1)
        self.assertIn("本轮已处理 1/2", state.label)
        self.assertEqual(update_workflow_progress_from_line(line, state).current, 1)

    def test_source_index_build_progress_tracks_chunk_lookup(self):
        state = create_workflow_progress_state("source_index_build")

        state = update_workflow_progress_from_line(
            "Source index retrieval for build: 3 chunks to query.",
            state,
        )
        self.assertTrue(state.visible)
        self.assertEqual(state.current, 0)
        self.assertEqual(state.total, 3)
        self.assertEqual(state.label, "检索原文索引 0/3")

        state = update_workflow_progress_from_line(
            "Source index retrieval progress: 2/3 chunks, file=script.v1.rpy, chunk=chunk-02.",
            state,
        )
        self.assertEqual(state.current, 2)
        self.assertEqual(state.total, 3)
        self.assertEqual(state.facts, ("当前文件：script.v1.rpy", "当前分块：chunk-02"))

        state = update_workflow_progress_from_line(
            "Source index retrieval complete: 3/3 chunks queried.",
            state,
        )
        self.assertEqual(state.current, 3)
        self.assertEqual(state.label, "检索原文索引 3/3")

    def test_source_index_build_progress_adds_eta_with_zero_start_time(self):
        state = create_workflow_progress_state("source_index_build")

        state = update_workflow_progress_from_line(
            "Source index retrieval for build: 10 chunks to query.",
            state,
            now=0.0,
        )
        state = update_workflow_progress_from_line(
            "Source index retrieval progress: 5/10 chunks, file=script.rpy, chunk=5.",
            state,
            now=10.0,
        )

        self.assertEqual(state.remaining_seconds, 10)
        self.assertIn("约剩 10 秒", state.label)

    def test_sync_request_progress_tracks_request_rows(self):
        state = create_workflow_progress_state("sync_requests")

        state = update_workflow_progress_from_line("[2/5] keyword-script-02", state)

        self.assertTrue(state.visible)
        self.assertEqual(state.current, 2)
        self.assertEqual(state.total, 5)
        self.assertEqual(state.label, "请求 2/5")
        self.assertEqual(state.facts, ("当前请求：keyword-script-02",))

    def test_sync_translation_progress_accumulates_batch_successes(self):
        state = create_workflow_progress_state("sync_translation")

        for line in (
            "Found 2 files.",
            "Processing: script.v1.rpy",
            "Found 5 lines to translate.",
            "Translated 2/3 items. (Received 42 chars of translation)",
            "Translated 3/3 items. (Received 84 chars of translation)",
        ):
            state = update_workflow_progress_from_line(line, state)

        self.assertEqual(state.current, 5)
        self.assertEqual(state.total, 5)
        self.assertEqual(state.label, "script.v1.rpy 5/5")
        self.assertEqual(state.facts, ("文件：1/2",))

        state = update_workflow_progress_from_line("Done with script.v1.rpy.", state)
        self.assertEqual(state.current, 1)
        self.assertEqual(state.total, 2)
        self.assertEqual(state.label, "文件 1/2")

    def test_sync_translation_progress_advances_files_without_new_lines(self):
        state = create_workflow_progress_state("sync_translation")

        for line in (
            "Found 2 files.",
            "Processing: old.rpy",
            "No new lines to translate.",
        ):
            state = update_workflow_progress_from_line(line, state)

        self.assertEqual(state.current, 1)
        self.assertEqual(state.total, 2)
        self.assertEqual(state.label, "文件 1/2")
        self.assertEqual(state.facts, ("无新增内容：old.rpy",))

    def test_rag_progress_tracks_scan_and_update_counts(self):
        state = create_workflow_progress_state("rag_bootstrap")

        state = update_workflow_progress_from_line(
            "RAG scan progress: 12 records scanned from 4 files, 5 pending.",
            state,
        )
        self.assertTrue(state.visible)
        self.assertEqual(state.current, 0)
        self.assertEqual(state.total, 5)
        self.assertEqual(state.label, "记忆库待写入 0/5")
        self.assertEqual(state.facts, ("扫描记录：12", "扫描文件：4"))

        state = update_workflow_progress_from_line(
            "RAG update progress: 3/5 records.",
            state,
        )
        self.assertEqual(state.current, 3)
        self.assertEqual(state.total, 5)
        self.assertEqual(state.label, "更新记忆库 3/5")

    def test_rag_progress_handles_no_pending_records(self):
        state = create_workflow_progress_state("rag_bootstrap")

        state = update_workflow_progress_from_line(
            "RAG scan progress: 8 records scanned from 2 files, 0 pending.",
            state,
        )

        self.assertTrue(state.visible)
        self.assertEqual(state.current, 8)
        self.assertEqual(state.total, 8)
        self.assertEqual(state.label, "记忆库无需更新")

    def test_work_bootstrap_progress_keeps_dotted_paths(self):
        state = create_workflow_progress_state("work_bootstrap")
        self.assertTrue(state.visible)
        self.assertTrue(state.indeterminate)
        self.assertEqual(state.label, "正在统计文件…")


        state = update_workflow_progress_from_line(
            "Work bootstrap copy progress: 0/7 files.",
            state,
        )
        self.assertEqual(state.current, 0)
        self.assertEqual(state.total, 7)
        self.assertEqual(state.label, "复制文件 0/7")
        self.assertFalse(state.indeterminate)

        state = update_workflow_progress_from_line(
            "Work bootstrap copy progress: 4/7 files, file=game/script.v1.rpy.",
            state,
        )
        self.assertEqual(state.current, 4)
        self.assertEqual(state.total, 7)
        self.assertEqual(state.label, "复制文件 4/7")
        self.assertEqual(state.facts, ("当前文件：game/script.v1.rpy",))


if __name__ == "__main__":
    unittest.main()
