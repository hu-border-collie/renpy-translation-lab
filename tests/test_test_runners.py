"""Smoke tests for CLI/GUI unittest discovery helpers."""
from __future__ import annotations

import pathlib
import sys
import unittest

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from run_cli_tests import build_suite as build_cli_suite
from run_gui_tests import build_suite as build_gui_suite


def _iter_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def _module_names(test_ids: set[str]) -> set[str]:
    return {test_id.split(".", 1)[0] for test_id in test_ids}


class TestDiscoveryRunners(unittest.TestCase):
    def test_cli_suite_excludes_gui_modules(self):
        names = {case.id() for case in _iter_cases(build_cli_suite())}
        self.assertTrue(names)
        self.assertFalse(
            any(module.startswith("test_gui_") for module in _module_names(names))
        )

    def test_gui_suite_only_includes_gui_modules(self):
        names = {case.id() for case in _iter_cases(build_gui_suite())}
        self.assertTrue(names)
        self.assertTrue(
            all(module.startswith("test_gui_") for module in _module_names(names))
        )


if __name__ == "__main__":
    unittest.main()