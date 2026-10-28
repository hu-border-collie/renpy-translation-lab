"""Discover GUI and settings-widget unittest modules for guarded GUI CI."""
from __future__ import annotations

import os
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
            suite.addTests(loader.loadTestsFromName(test_module_name(path, directory)))
    return suite


def main(
    argv: list[str] | None = None,
    *,
    shutdown_runtime: bool = True,
) -> int:
    """Run the GUI suite; ``shutdown_runtime=False`` is hard-exit mode.

    In hard-exit mode the caller is expected to call :func:`_terminate_process`
    immediately, so no Qt pool teardown is performed at all.
    """
    args = parse_runner_args(__doc__ or "", argv)
    ensure_tests_on_path()
    from tests.support.gui_test_support import (
        guarded_gui_test_environment,
        guarded_test_result_class,
        shutdown_gui_test_runtime,
    )

    guard = None
    with guarded_gui_test_environment(process_events=shutdown_runtime) as guard:
        exit_code = run_discovered_suite(
            build_suite(),
            quiet=args.quiet,
            verbose=args.verbose,
            resultclass=guarded_test_result_class(guard),
        )
    unexpected_dialogs = bool(guard and guard.rejected_dialogs)
    if shutdown_runtime:
        runtime_stopped = shutdown_gui_test_runtime()
    else:
        # Script entrypoint mode hard-exits immediately after ``main`` returns.
        # Skip even the pool-only Qt teardown here: offscreen Linux runners can
        # segfault after all tests passed while QThreadPool shuts down.
        runtime_stopped = True
    return 1 if unexpected_dialogs or not runtime_stopped else exit_code


def _terminate_process(exit_code: int) -> None:
    """Exit after flushing output without running unstable Qt finalizers."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes

        # os._exit can still enter Windows DLL shutdown and fault in Qt after
        # a successful suite. End only this test process without DLL callbacks;
        # the completed suite's status remains the process exit code.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess.restype = ctypes.c_int
        if not kernel32.TerminateProcess(kernel32.GetCurrentProcess(), int(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return  # Terminating the calling process does not return on Windows.
    os._exit(int(exit_code))


if __name__ == "__main__":
    # Skip Qt finalization before the hard exit; the suite and modal guard
    # already determined the status, including any test or unexpected dialog.
    _terminate_process(main(shutdown_runtime=False))
