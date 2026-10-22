import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_quality_gates as gates


class QualityGateConfigTests(unittest.TestCase):
    def test_tool_versions_match_requirements_dev(self):
        text = gates.DEV_REQUIREMENTS.read_text(encoding="utf-8")
        for name, version in gates.TOOL_VERSIONS.items():
            self.assertIn(f"{name}=={version}", text)

    def test_pyproject_and_exceptions_files_exist(self):
        self.assertTrue((gates.REPO_ROOT / "pyproject.toml").is_file())
        self.assertTrue(gates.EXCEPTIONS_PATH.is_file())
        for relative in gates.AUDIT_LOCKS:
            self.assertTrue((gates.REPO_ROOT / relative).is_file(), relative)

    def test_exceptions_file_is_valid_and_has_required_fields(self):
        exceptions = gates.load_audit_exceptions()
        self.assertGreaterEqual(len(exceptions), 1)
        for item in exceptions:
            for key in ("id", "package", "reason", "review_by"):
                self.assertIsInstance(item[key], str)
                self.assertTrue(str(item[key]).strip())

    def test_exception_ids_are_unique(self):
        ids = gates.exception_ids()
        self.assertEqual(len(ids), len(set(ids)))

    def test_load_audit_exceptions_rejects_incomplete_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "exceptions.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "exceptions": [
                            {
                                "id": "CVE-TEST",
                                "package": "example",
                                "reason": "",
                                "review_by": "2026-08-01",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(gates.QualityGateError):
                gates.load_audit_exceptions(path)

    def test_policy_command_exits_zero(self):
        self.assertEqual(gates.main(["policy"]), 0)

    def test_blocking_runner_fails_when_any_step_fails(self):
        with mock.patch.object(gates, "run_ruff_scoped", return_value=1), mock.patch.object(
            gates, "run_ruff_critical", return_value=0
        ), mock.patch.object(gates, "run_mypy_scoped", return_value=0), mock.patch.object(
            gates, "run_pip_audit", return_value=0
        ), mock.patch("builtins.print"):
            self.assertEqual(gates.run_blocking(), 1)

    def test_pip_audit_passes_ignore_flags_from_exceptions(self):
        captured: list[list[str]] = []

        def fake_run(command, *, cwd=None):  # noqa: ANN001
            captured.append(list(command))
            return 0

        with mock.patch.object(gates, "_run", side_effect=fake_run), mock.patch.object(
            gates, "exception_ids", return_value=["PYSEC-TEST-1", "PYSEC-TEST-2"]
        ):
            code = gates.run_pip_audit(locks=gates.AUDIT_LOCKS[:1])
        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        command = captured[0]
        self.assertIn("pip_audit", command)
        self.assertIn("--disable-pip", command)
        self.assertIn("--ignore-vuln", command)
        self.assertIn("PYSEC-TEST-1", command)
        self.assertIn("PYSEC-TEST-2", command)


if __name__ == "__main__":
    unittest.main()
