import unittest

from gui_qt.diagnostics_context import (
    build_cli_commands,
    build_diagnostics_context,
    collect_existing_report_paths,
    format_cli_command,
    format_manifest_json_preview,
    idle_diagnostics_context,
    manifest_for_preview,
    quote_cli_arg,
    resolve_package_dir,
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
            f"{package_dir}\\check_failures.jsonl": True,
            f"{package_dir}\\results.jsonl": False,
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
        self.assertNotIn("写回翻译（仅 safe）", labels)

    def test_build_diagnostics_context_idle_without_manifest(self):
        context = build_diagnostics_context(
            latest_manifest_path=None,
            manifest=None,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
        )
        self.assertEqual(context.status, idle_diagnostics_context().status)
        self.assertEqual(context.commands, [])

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
                f"{package_dir}\\requests.jsonl",
                r"C:\logs\batch_jobs\latest_manifest.txt",
            }

        context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path="gemini_translate_batch.py",
            logs_dir=r"C:\logs\batch_jobs",
            path_exists=path_exists,
        )
        self.assertEqual(context.status, "ready")
        self.assertTrue(any("Manifest" in fact for fact in context.facts))
        self.assertTrue(any(entry.label == "Batch 请求" for entry in context.paths))
        self.assertTrue(context.commands)
        self.assertIn('"mode": "translation"', format_manifest_json_preview(manifest))

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


if __name__ == "__main__":
    unittest.main()