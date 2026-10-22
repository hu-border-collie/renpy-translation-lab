"""Discover CLI/core unittest modules, excluding GUI and settings-widget tests."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support.runner_support import (
    ensure_tests_on_path,
    is_gui_test_module,
    iter_test_modules,
    parse_runner_args,
    run_discovered_suite,
    test_module_name,
)


def build_suite() -> unittest.TestSuite:
    _, directory = ensure_tests_on_path()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in iter_test_modules(directory):
        if is_gui_test_module(path):
            continue
        suite.addTests(loader.loadTestsFromName(test_module_name(path, directory)))
    return suite


def main(argv: list[str] | None = None) -> int:
    args = parse_runner_args(__doc__ or "", argv)
    return run_discovered_suite(build_suite(), quiet=args.quiet, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
