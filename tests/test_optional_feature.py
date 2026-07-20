"""Tests for optional feature status probing and CLI install messages."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from optional_feature import (
    FeatureInstallState,
    PackageRequirement,
    ensure_relation_analyzer_dependencies,
    format_shell_command,
    hash_checked_install_command,
    missing_feature_cli_message,
    probe_feature,
    relation_analyzer_feature,
)


class OptionalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feature = relation_analyzer_feature()

    def test_relation_analyzer_feature_reads_pinned_packages_and_scipy(self) -> None:
        names = {package.distribution for package in self.feature.packages}
        self.assertEqual(
            names,
            {"numpy", "matplotlib", "scikit-learn", "pillow", "scipy"},
        )
        self.assertTrue(self.feature.lock_path().is_file())
        scipy = next(pkg for pkg in self.feature.packages if pkg.distribution == "scipy")
        self.assertEqual(scipy.version, "")
        self.assertEqual(scipy.import_names, ("scipy",))

    def test_probe_not_installed_when_all_missing(self) -> None:
        with mock.patch.object(
            PackageRequirement,
            "is_present",
            return_value=False,
        ), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            return_value=False,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            return_value="",
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.NOT_INSTALLED)
        self.assertEqual(status.action_label, "安装并启用")
        self.assertEqual(len(status.missing), len(self.feature.packages))

    def test_probe_partially_installed(self) -> None:
        present = {"numpy", "pillow"}

        def _present(self: PackageRequirement) -> bool:
            return self.distribution in present

        def _version(self: PackageRequirement) -> str:
            return "1.0.0" if self.distribution in present else ""

        with mock.patch.object(
            PackageRequirement,
            "is_present",
            _present,
        ), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            _present,
        ), mock.patch.object(
            PackageRequirement,
            "import_names_available",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            _version,
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.PARTIALLY_INSTALLED)
        self.assertEqual(status.action_label, "修复安装")
        self.assertTrue(any("matplotlib" in item for item in status.missing))
        self.assertTrue(any("scikit-learn" in item for item in status.missing))
        self.assertTrue(any("scipy" in item for item in status.missing))

    def test_probe_partial_when_metadata_present_but_import_missing(self) -> None:
        def _is_present(self: PackageRequirement) -> bool:
            return self.distribution != "scipy"

        def _metadata(self: PackageRequirement) -> bool:
            return True

        def _imports(self: PackageRequirement) -> bool:
            return self.distribution != "scipy"

        def _version(self: PackageRequirement) -> str:
            if self.distribution == "scipy":
                return "1.17.1"
            return self.version or "1.0.0"

        with mock.patch.object(PackageRequirement, "is_present", _is_present), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            _metadata,
        ), mock.patch.object(
            PackageRequirement,
            "import_names_available",
            _imports,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            _version,
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.PARTIALLY_INSTALLED)
        self.assertEqual(status.action_label, "修复安装")
        self.assertTrue(any("scipy" in item and "模块不可用" in item for item in status.missing))

    def test_probe_installed_and_update_available(self) -> None:
        def _version(self: PackageRequirement) -> str:
            if self.distribution == "numpy":
                return "0.0.1"
            if not self.version:
                return "1.0.0"
            return self.version

        with mock.patch.object(
            PackageRequirement,
            "is_present",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "import_names_available",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            _version,
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.UPDATE_AVAILABLE)
        self.assertEqual(status.action_label, "更新")
        self.assertTrue(any(item.startswith("numpy ") for item in status.outdated))
        # Presence-only scipy must not create a pin mismatch entry.
        self.assertFalse(any(item.startswith("scipy ") for item in status.outdated))

        with mock.patch.object(
            PackageRequirement,
            "is_present",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "import_names_available",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            lambda self: self.version or "1.0.0",
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.INSTALLED)
        self.assertEqual(status.action_label, "已启用")

    def test_probe_installing_and_failed_overlay(self) -> None:
        installing = probe_feature(self.feature, installing=True)
        self.assertEqual(installing.state, FeatureInstallState.INSTALLING)
        self.assertEqual(installing.action_label, "正在安装…")

        with mock.patch.object(
            PackageRequirement,
            "is_present",
            return_value=False,
        ), mock.patch.object(
            PackageRequirement,
            "metadata_present",
            return_value=False,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            return_value="",
        ):
            failed = probe_feature(self.feature, last_failed=True)
        self.assertEqual(failed.state, FeatureInstallState.FAILED)
        self.assertIn("失败", failed.message)

    def test_import_names_available_uses_find_spec_not_import(self) -> None:
        package = PackageRequirement("numpy", "2.4.2", import_names=("numpy",))
        fake_spec = object()
        with mock.patch(
            "optional_feature.importlib.util.find_spec",
            return_value=fake_spec,
        ) as find_spec:
            self.assertTrue(package.import_names_available())
        find_spec.assert_called_once_with("numpy")

        with mock.patch(
            "optional_feature.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertFalse(package.import_names_available())

        with mock.patch(
            "optional_feature.importlib.util.find_spec",
            side_effect=ValueError("broken"),
        ):
            self.assertFalse(package.import_names_available())

    def test_hash_checked_install_command_uses_absolute_paths_and_quotes(self) -> None:
        python = r"C:\Program Files\Python311\python.exe"
        command = hash_checked_install_command(
            self.feature,
            python_executable=python,
            repo_root=Path(__file__).resolve().parents[1],
        )
        self.assertIn("--require-hashes", command)
        lock = self.feature.lock_path().resolve()
        self.assertIn(str(lock), command)
        # Relative lock path alone must not be the only reference.
        self.assertNotEqual(
            command,
            f"{python} -m pip install --require-hashes -r requirements-lock/py311-relation-analyzer.txt",
        )
        # Spaces in the interpreter path must be quoted on Windows.
        self.assertIn(format_shell_command([python]), command)

    def test_missing_feature_cli_message_is_actionable_with_absolute_paths(self) -> None:
        python = r"C:\Program Files\Python311\python.exe"
        repo_root = Path(__file__).resolve().parents[1]
        message = missing_feature_cli_message(
            self.feature,
            python_executable=python,
            repo_root=repo_root,
            detail="缺少：numpy。",
        )
        self.assertIn("关系分析器", message)
        self.assertIn("缺少：numpy。", message)
        self.assertIn("--require-hashes", message)
        self.assertIn(str(self.feature.lock_path(repo_root)), message)
        self.assertIn(str(self.feature.requirements_path(repo_root)), message)
        self.assertNotIn("Traceback", message)

    def test_ensure_relation_analyzer_dependencies_exits_without_traceback_chain(
        self,
    ) -> None:
        with mock.patch(
            "optional_feature.probe_feature",
            return_value=mock.Mock(
                state=FeatureInstallState.NOT_INSTALLED,
                missing=("numpy",),
            ),
        ):
            with self.assertRaises(SystemExit) as raised:
                ensure_relation_analyzer_dependencies(
                    python_executable=r"C:\Program Files\Python311\python.exe",
                )
        message = str(raised.exception)
        self.assertIn("pip install", message)
        self.assertIn(str(self.feature.lock_path().resolve()), message)
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
