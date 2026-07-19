import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import run_renpy_integration as integration


class RenPyIntegrationRunnerTests(unittest.TestCase):
    def test_missing_sdk_launcher_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = Path(tmp) / "sdk"
            sdk.mkdir()

            with self.assertRaisesRegex(
                integration.RenPyIntegrationError,
                "missing the POSIX launcher",
            ):
                integration.resolve_launcher(sdk, "linux")

    def test_runner_generates_template_then_lints_and_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdk = root / "sdk"
            sdk.mkdir()
            (sdk / "renpy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            actions = []

            def fake_runner(command, **kwargs):
                project_dir = Path(command[3])
                action = command[4]
                actions.append((action, command[5:]))
                if action == "translate":
                    target = project_dir / "game" / "tl" / "schinese" / "script.rpy"
                    target.parent.mkdir(parents=True)
                    target.write_text(
                        "translate schinese start_fixture:\n"
                        "    # Eileen \"Hello from the integration fixture.\"\n"
                        "    Eileen \"\"\n",
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(command, 0, "ok")

            integration.run_integration(
                sdk,
                platform_name="linux",
                command_runner=fake_runner,
                temp_parent=root / "temp",
            )

            self.assertEqual(
                actions,
                [
                    ("translate", ["schinese"]),
                    ("lint", ["--error-code"]),
                    ("test", ["smoke", "--report-detailed"]),
                ],
            )

    def test_runner_stops_when_template_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdk = root / "sdk"
            sdk.mkdir()
            (sdk / "renpy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            actions = []

            def fake_runner(command, **kwargs):
                actions.append(command[4])
                return subprocess.CompletedProcess(command, 0, "ok")

            with self.assertRaisesRegex(
                integration.RenPyIntegrationError,
                "did not generate any translation scripts",
            ):
                integration.run_integration(
                    sdk,
                    platform_name="linux",
                    command_runner=fake_runner,
                    temp_parent=root / "temp",
                )

            self.assertEqual(actions, ["translate"])

    def test_scheduled_workflow_pins_official_sdk_and_checksum(self):
        workflow = (
            integration.REPO_ROOT / ".github" / "workflows" / "renpy-integration.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('RENPY_VERSION: "8.5.2"', workflow)
        version_reference = chr(36) + "{RENPY_VERSION}"
        self.assertIn(
            f"https://www.renpy.org/dl/{version_reference}/"
            f"renpy-{version_reference}-sdk.tar.bz2",
            workflow,
        )
        self.assertIn(
            "cf9ed145e5b32521a4b2caddb4cd3073c64259ac51e1f7aab94a8a8ff72b55c4",
            workflow,
        )
        self.assertIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)


if __name__ == "__main__":
    unittest.main()
