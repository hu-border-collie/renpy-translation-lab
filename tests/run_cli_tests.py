"""Discover unittest modules for CLI-only CI (excludes ``test_gui_*``)."""
from __future__ import annotations

import unittest

from test_runner_common import ensure_tests_on_path, parse_runner_args, run_discovered_suite


def build_suite() -> unittest.TestSuite:
    _, directory = ensure_tests_on_path()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(directory.glob("test_*.py")):
        if path.name.startswith("test_gui_"):
            continue
        suite.addTests(loader.loadTestsFromName(path.stem))
    return suite


def main(argv: list[str] | None = None) -> int:
    args = parse_runner_args(__doc__ or "", argv)
    return run_discovered_suite(build_suite(), quiet=args.quiet, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())