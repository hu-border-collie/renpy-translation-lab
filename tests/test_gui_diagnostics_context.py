import unittest

from gui_qt.diagnostics_context import (
    build_cli_commands,
    build_diagnostics_context,
    collect_existing_report_paths,
    format_cli_command,
    idle_diagnostics_context,
    join_directory_file,
    manifest_for_preview,
    quote_cli_arg,
    resolve_package_dir,
    sync_diagnostics_context,
)


class GuiDiagnosticsContextTests(unittest.TestCase):
    def test_quote_cli_arg_quotes_paths_with_spaces(self):
        self.assertEqual(quote_cli_arg("C:\\Games\\My Game\\manifest.json"), '"C:\\Games\\My Game\\manifest.json"')
        self.assertEqual(quote_cli_arg("doctor"), "doctor")

    def test_format_cli_command_uses_argument_list_style(self):
        command = format_cli_command(
            "python",
            "C:\\tool\\gemini_translate_batch.py",
            ["status", "C:\\jobs\\manifest.json"],
        )
        self.assertIn("python", command)
        self.assertIn("gemini_translate_batch.py", command)
        self.assertIn("status", command)

    def test_resolve_package_dir_from_manifest_path(self):
        package = resolve_package_dir(r"C:\logs\batch_jobs\job1\manifest.json")
        self.assertEqual(package, r"C:\logs\batch_jobs\job1")

    def test_resolve_package_dir_from_posix_manifest_path(self):
        package = resolve_package_dir("/tmp/jobs/job1/manifest.json")
        self.assertEqual(package, "/tmp/jobs/job1")

    def test_manifest_for_preview_omits_large_sections(self):
        preview = manifest_for_preview(
            {
                "mode": "translation",
                "files": {"a.rpy": {}},
                "chunks": [{"id": 1}],
            }
        )
        self.assertNotIn("files", preview)
        self.assertNotIn("chunks", preview)
        self.assertIn("_preview_note", preview)

    def test_collect_existing_report_paths_only_returns_existing_files(self):
        package_dir = r"C:\logs\batch_jobs\job1"
        paths = {
            join_directory_file(package_dir, "check_failures.jsonl"): True,
            join_directory_file(package_dir, "results.jsonl"): False,
        }

        entries = collect_existing_report_paths(
            package_dir,
            {"last_check_report_path": r"C:\logs\custom_report.json"},
            path_exists=lambda path: paths.get(path, False),
        )
        labels = [entry.label for entry in entries]
        self.assertIn("检查失败明细", labels)
        self.assertNotIn("Batch 结果", labels)

    def test_build_cli_commands_includes_submit_when_job_missing(self):
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\jobs\manifest.json",
            manifest={"job_name": ""},
        )
        labels = [command.label for command in commands]
        self.assertIn("提交 Batch 任务", labels)
        self.assertIn("查询任务状态", labels)

    def test_build_cli_commands_omits_apply_after_applied(self):
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\jobs\manifest.json",
            manifest={"job_name": "jobs/abc", "applied_at": "2026-01-01T00:00:00"},
        )
        labels = [command.label for command in commands]
        self.assertNotIn("写回翻译（仅可写回）", labels)

    def test_warn_translation_commands_include_remediation_paths(self):
        manifest_path = r"C:\logs\batch_jobs\job1\manifest.json"
        retry_path = r"C:\logs\batch_jobs\job1\retry_parts\retry1\manifest.json"
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=manifest_path,
            manifest={
                "mode": "translation",
                "job_name": "batches/parent",
                "last_check_summary": {"safety_level": "warn"},
                "last_check_report_path": r"C:\logs\batch_jobs\job1\check_failures.jsonl",
                "last_retry_manifest_path": retry_path,
            },
        )

        by_label = {command.label: command.command for command in commands}
        self.assertNotIn("生成 retry 包", by_label)
        self.assertIn("提交 retry 任务", by_label)
        self.assertIn("合并 retry 结果", by_label)
        self.assertIn("重新检查翻译结果", by_label)
        self.assertNotIn("写回翻译（仅可写回）", by_label)
        self.assertNotIn("同步修复失败项", by_label)
        self.assertNotIn("生成订正包", by_label)
        self.assertNotIn("预览订正结果", by_label)
        self.assertIn(retry_path, by_label["提交 retry 任务"])
        self.assertIn(retry_path, by_label["合并 retry 结果"])

    def test_warn_without_existing_retry_includes_build_retry(self):
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\logs\batch_jobs\job1\manifest.json",
            manifest={
                "mode": "translation",
                "job_name": "batches/parent",
                "last_check_summary": {"safety_level": "warn"},
            },
        )

        by_label = {command.label: command.command for command in commands}
        self.assertIn("生成 retry 包", by_label)
        self.assertIn("build-retry", by_label["生成 retry 包"])
        self.assertIn("RETRY_MANIFEST_PATH", by_label["提交 retry 任务"])

    def test_warn_retry_placeholder_is_shell_safe(self):
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\logs\batch_jobs\job1\manifest.json",
            manifest={
                "mode": "translation",
                "job_name": "batches/parent",
                "last_check_summary": {"safety_level": "warn"},
            },
        )

        command_text = "\n".join(command.command for command in commands)
        self.assertIn("RETRY_MANIFEST_PATH", command_text)
        self.assertNotIn("<retry-manifest.json>", command_text)

    def test_retry_manifest_commands_merge_back_to_parent(self):
        retry_path = r"C:\logs\batch_jobs\job1\retry_parts\retry1\manifest.json"
        parent_path = r"C:\logs\batch_jobs\job1\manifest.json"
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=retry_path,
            manifest={
                "mode": "translation",
                "job_name": "batches/retry",
                "retry_of_manifest": parent_path,
            },
        )

        by_label = {command.label: command.command for command in commands}
        self.assertIn("合并 retry 结果", by_label)
        self.assertIn("重新检查父任务", by_label)
        self.assertNotIn("写回翻译（仅可写回）", by_label)
        self.assertIn(parent_path, by_label["合并 retry 结果"])
        self.assertIn(retry_path, by_label["合并 retry 结果"])

    def test_revision_manifest_commands_use_revision_apply_flow(self):
        manifest_path = r"C:\logs\batch_jobs\rev1\manifest.json"
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=manifest_path,
            manifest={"mode": "revision", "job_name": "batches/rev"},
        )

        by_label = {command.label: command.command for command in commands}
        self.assertIn("预览订正结果", by_label)
        self.assertIn("应用订正（预览确认后）", by_label)
        self.assertNotIn("检查翻译结果", by_label)
        self.assertNotIn("写回翻译（仅可写回）", by_label)
        self.assertIn("preview-revisions", by_label["预览订正结果"])
        self.assertIn("apply-revisions", by_label["应用订正（预览确认后）"])

    def test_applied_revision_manifest_omits_revision_apply_command(self):
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\logs\batch_jobs\rev1\manifest.json",
            manifest={
                "mode": "revision",
                "job_name": "batches/rev",
                "revision_applied_at": "2026-06-25T20:00:00",
            },
        )

        labels = [command.label for command in commands]
        self.assertIn("预览订正结果", labels)
        self.assertNotIn("应用订正（预览确认后）", labels)

    def test_keyword_manifest_commands_use_export_keywords_flow(self):
        manifest_path = r"C:\logs\batch_jobs\kw1\manifest.json"
        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=manifest_path,
            manifest={"mode": "keyword_extraction", "job_name": "batches/kw"},
        )

        by_label = {command.label: command.command for command in commands}
        self.assertIn("查询关键词状态", by_label)
        self.assertIn("导出关键词报告", by_label)
        self.assertNotIn("检查翻译结果", by_label)
        self.assertNotIn("写回翻译（仅可写回）", by_label)
        self.assertIn("export-keywords", by_label["导出关键词报告"])
        self.assertIn(manifest_path, by_label["导出关键词报告"])

    def test_build_diagnostics_context_idle_without_manifest(self):
        context = build_diagnostics_context(
            latest_manifest_path=None,
            manifest=None,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
        )
        self.assertEqual(context.status, idle_diagnostics_context().status)
        self.assertEqual(context.commands, [])

    def test_build_diagnostics_context_idle_when_latest_path_suppressed(self):
        context = build_diagnostics_context(
            latest_manifest_path=None,
            manifest=None,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=lambda _path: True,
        )
        self.assertEqual(context.status, "idle")
        self.assertEqual(context.heading, idle_diagnostics_context().heading)

    def test_build_diagnostics_context_warns_when_latest_path_without_manifest(self):
        context = build_diagnostics_context(
            latest_manifest_path=r"C:\logs\batch_jobs\job1\manifest.json",
            manifest=None,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=lambda _path: True,
        )
        self.assertEqual(context.status, "warning")
        self.assertIn("无法读取任务清单", context.heading)

    def test_sync_diagnostics_context_shows_sync_command(self):
        context = sync_diagnostics_context(
            sync_script_path=r"C:\tools\gemini_translate.py",
            python_exe=r"C:\Python\python.exe",
        )

        self.assertEqual(context.status, "ready")
        self.assertIn("同步翻译上下文", context.heading)
        self.assertEqual(len(context.commands), 1)
        self.assertIn("gemini_translate.py", context.commands[0].command)
        self.assertEqual(context.manifest_json_preview, "")

    def test_build_diagnostics_context_ready_with_manifest(self):
        manifest_path = r"C:\logs\batch_jobs\job1\manifest.json"
        package_dir = r"C:\logs\batch_jobs\job1"
        manifest = {
            "_manifest_path": manifest_path,
            "mode": "translation",
            "job_name": "batches/123",
            "job_state": "JOB_STATE_SUCCEEDED",
            "last_check_summary": {"safety_level": "safe"},
            "display_name": "demo-batch",
        }

        def path_exists(path: str) -> bool:
            return path in {
                manifest_path,
                join_directory_file(package_dir, "requests.jsonl"),
                join_directory_file(r"C:\logs\batch_jobs", "latest_manifest.txt"),
            }

        context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=path_exists,
        )
        self.assertEqual(context.status, "ready")
        self.assertTrue(any("任务清单：" in fact for fact in context.facts))
        self.assertTrue(any("最近检查：可写回" in fact for fact in context.facts))
        self.assertTrue(any(entry.label == "Batch 请求" for entry in context.paths))
        self.assertTrue(context.commands)
        self.assertIn('"mode": "translation"', context.manifest_json_preview)

    def test_build_diagnostics_context_warns_when_latest_differs(self):
        manifest_path = r"C:\logs\batch_jobs\job1\manifest.json"
        latest_path = r"C:\logs\batch_jobs\job2\manifest.json"
        context = build_diagnostics_context(
            latest_manifest_path=latest_path,
            manifest={"_manifest_path": manifest_path, "mode": "translation"},
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=lambda _path: True,
        )
        self.assertEqual(context.status, "warning")

    def test_collect_existing_report_paths_deduplicates_equivalent_windows_paths(self):
        package_dir = r"C:\logs\batch_jobs\job1"

        entries = collect_existing_report_paths(
            package_dir,
            {"last_check_report_path": r"c:/logs/batch_jobs/job1/check_failures.jsonl"},
            path_exists=lambda path: path.lower().endswith("check_failures.jsonl"),
        )
        duplicate_labels = [
            entry.label
            for entry in entries
            if entry.label in {"检查失败明细", "最近检查报告"}
        ]
        self.assertEqual(len(duplicate_labels), 1)

    def test_build_diagnostics_context_ready_when_latest_paths_equivalent(self):
        manifest_path = r"C:\logs\batch_jobs\job1\manifest.json"
        context = build_diagnostics_context(
            latest_manifest_path=r"c:/logs/batch_jobs/job1/manifest.json",
            manifest={"_manifest_path": manifest_path, "mode": "translation"},
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=lambda _path: True,
        )
        self.assertEqual(context.status, "ready")


if __name__ == "__main__":
    unittest.main()