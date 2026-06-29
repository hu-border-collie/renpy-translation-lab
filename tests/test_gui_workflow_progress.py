import unittest

from gui_qt.workflow_progress import (
    create_workflow_progress_state,
    update_workflow_progress_from_line,
)


class GuiWorkflowProgressTests(unittest.TestCase):
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
