"""Discover unittest modules for CLI-only CI (excludes ``test_gui_*``)."""
from __future__ import annotations

import argparse
import pathlib
import sys
import unittest


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def tests_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def build_suite() -> unittest.TestSuite:
    root = repo_root()
    directory = tests_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(directory.glob("test_*.py")):
        if path.name.startswith("test_gui_"):
            continue
        suite.addTests(loader.loadTestsFromName(path.stem))
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)

    verbosity = 2 if args.verbose else (1 if args.quiet else 2)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(build_suite())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())