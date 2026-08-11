"""Shared helpers for CLI/GUI unittest discovery entrypoints."""
from __future__ import annotations

import argparse
import pathlib
import sys
import unittest


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def tests_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def ensure_tests_on_path() -> tuple[pathlib.Path, pathlib.Path]:
    root = repo_root()
    directory = tests_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return root, directory


def parse_runner_args(
    description: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args(argv)


def run_discovered_suite(
    suite: unittest.TestSuite,
    *,
    quiet: bool = False,
    verbose: bool = False,
    resultclass=None,
) -> int:
    """Run a suite and return its process exit status.

    When provided, ``resultclass`` is passed to ``unittest.TextTestRunner``
    and must be a compatible ``unittest.TextTestResult`` implementation.
    """
    verbosity = 2 if verbose else (1 if quiet else 2)
    runner_options = {"verbosity": verbosity}
    if resultclass is not None:
        runner_options["resultclass"] = resultclass
    runner = unittest.TextTestRunner(**runner_options)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1
