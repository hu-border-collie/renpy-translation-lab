"""Tests for optional feature status probing and CLI install messages."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from optional_feature import (
    FeatureInstallState,
    PackageRequirement,
    ensure_relation_analyzer_dependencies,
    hash_checked_install_command,
    missing_feature_cli_message,
    probe_feature,
    relation_analyzer_feature,
)


class OptionalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feature = relation_analyzer_feature()

    def test_relation_analyzer_feature_reads_pinned_packages(self) -> None:
        names = {package.distribution for package in self.feature.packages}
        self.assertEqual(names, {"numpy", "matplotlib", "scikit-learn", "pillow"})
        self.assertTrue(self.feature.lock_path().is_file())

    def test_probe_not_installed_when_all_missing(self) -> None:
        with mock.patch.object(
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

        with mock.patch.object(PackageRequirement, "metadata_present", _present), mock.patch.object(
            PackageRequirement,
            "installed_version",
            _version,
        ):
            status = probe_feature(self.feature)
        self.assertEqual(status.state, FeatureInstallState.PARTIALLY_INSTALLED)
        self.assertEqual(status.action_label, "修复安装")
        self.assertIn("matplotlib", status.missing)
        self.assertIn("scikit-learn", status.missing)

    def test_probe_installed_and_update_available(self) -> None:
        def _version(self: PackageRequirement) -> str:
            if self.distribution == "numpy":
                return "0.0.1"
            return self.version

        with mock.patch.object(
            PackageRequirement,
            "metadata_present",
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

        with mock.patch.object(
            PackageRequirement,
            "metadata_present",
            return_value=True,
        ), mock.patch.object(
            PackageRequirement,
            "installed_version",
            lambda self: self.version,
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

    def test_hash_checked_install_command_uses_lock(self) -> None:
        command = hash_checked_install_command(self.feature, python_executable="python")
        self.assertIn("--require-hashes", command)
        self.assertIn("requirements-lock/py311-relation-analyzer.txt", command)

    def test_missing_feature_cli_message_is_actionable(self) -> None:
        message = missing_feature_cli_message(
            self.feature,
            python_executable="python",
            detail="缺少：numpy。",
        )
        self.assertIn("关系分析器", message)
        self.assertIn("缺少：numpy。", message)
        self.assertIn("--require-hashes", message)
        self.assertIn("requirements-relation-analyzer.txt", message)
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
                ensure_relation_analyzer_dependencies(python_executable="python")
        message = str(raised.exception)
        self.assertIn("pip install", message)
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
