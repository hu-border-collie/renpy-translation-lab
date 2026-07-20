"""Run staged lint, type-check, and dependency-audit quality gates.

Blocking scopes are narrow by design (issue #230). Expand only after measuring
a new baseline and documenting debt — see docs/ci.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCEPTIONS_PATH = REPO_ROOT / "quality" / "pip-audit-exceptions.json"
DEV_REQUIREMENTS = REPO_ROOT / "requirements-dev.txt"

# Narrow blocking lint path: maintainable scripts with low debt.
RUFF_SCOPED_PATHS = ("scripts",)

# Repo-wide correctness rules only (pyflakes-critical). Broader style rules stay
# scoped until a measured full-repo baseline is accepted.
RUFF_CRITICAL_SELECT = "E9,F63,F7,F82"

# Narrow blocking type-check path. follow_imports=skip is set in pyproject.toml.
MYPY_SCOPED_PATHS = ("scripts",)

# Product lock profiles audited for known vulnerabilities.
AUDIT_LOCKS = (
    "requirements-lock/py311-cli.txt",
    "requirements-lock/py311-gui.txt",
    "requirements-lock/py311-linux-litellm.txt",
    "requirements-lock/py311-windows-litellm.txt",
)

# Pinned tools from requirements-dev.txt (documented for --print-policy).
TOOL_VERSIONS = {
    "ruff": "0.12.5",
    "mypy": "1.17.0",
    "pip-audit": "2.9.0",
}


class QualityGateError(RuntimeError):
    """Raised when a blocking quality gate fails."""


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> int:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(list(command), cwd=str(cwd), check=False)
    return int(completed.returncode)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def load_audit_exceptions(path: Path = EXCEPTIONS_PATH) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise QualityGateError(f"unsupported exceptions file format: {path}")
    exceptions = payload.get("exceptions")
    if not isinstance(exceptions, list):
        raise QualityGateError(f"exceptions must be a list: {path}")
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(exceptions):
        if not isinstance(item, dict):
            raise QualityGateError(f"exception[{index}] must be an object")
        vuln_id = item.get("id")
        package = item.get("package")
        reason = item.get("reason")
        review_by = item.get("review_by")
        required = (vuln_id, package, reason, review_by)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise QualityGateError(
                f"exception[{index}] requires non-empty string "
                "id, package, reason, review_by"
            )
        normalized.append(item)
    return normalized


def exception_ids(path: Path = EXCEPTIONS_PATH) -> list[str]:
    return [str(item["id"]) for item in load_audit_exceptions(path)]


def run_ruff_scoped() -> int:
    """Blocking: default ruff rules on the narrow scripts/ scope."""
    return _run(
        _python_module(
            "ruff",
            "check",
            *RUFF_SCOPED_PATHS,
            "--config",
            str(REPO_ROOT / "pyproject.toml"),
        )
    )


def run_ruff_critical() -> int:
    """Blocking: critical correctness rules across the repository."""
    return _run(
        _python_module(
            "ruff",
            "check",
            ".",
            "--select",
            RUFF_CRITICAL_SELECT,
            "--config",
            str(REPO_ROOT / "pyproject.toml"),
        )
    )


def run_mypy_scoped() -> int:
    """Blocking: mypy on the narrow scripts/ scope."""
    return _run(
        _python_module(
            "mypy",
            *MYPY_SCOPED_PATHS,
            "--config-file",
            str(REPO_ROOT / "pyproject.toml"),
        )
    )


def run_pip_audit(*, locks: Sequence[str] = AUDIT_LOCKS) -> int:
    """Blocking: pip-audit every product lock, applying the exception policy."""
    ignored = exception_ids()
    ignore_flags: list[str] = []
    for vuln_id in ignored:
        ignore_flags.extend(["--ignore-vuln", vuln_id])

    failures = 0
    for relative in locks:
        lock_path = REPO_ROOT / relative
        if not lock_path.is_file():
            print(f"error: missing lock file {relative}", file=sys.stderr)
            failures += 1
            continue
        code = _run(
            _python_module(
                "pip_audit",
                "--disable-pip",
                "--progress-spinner",
                "off",
                "-r",
                str(lock_path),
                *ignore_flags,
            )
        )
        if code != 0:
            failures += 1
    return 1 if failures else 0


def run_ruff_advisory_repo() -> int:
    """Advisory: full-repo default ruff (does not block merge by itself)."""
    # Intentionally uses a smaller, less noisy select than the full pyproject
    # default so the advisory signal stays readable while debt is paid down.
    return _run(
        _python_module(
            "ruff",
            "check",
            ".",
            "--select",
            "E,F",
            "--statistics",
            "--config",
            str(REPO_ROOT / "pyproject.toml"),
        )
    )


def print_policy() -> None:
    print("Quality gate policy (issue #230)")
    print()
    print("Pinned tools (requirements-dev.txt):")
    for name, version in TOOL_VERSIONS.items():
        print(f"  - {name}=={version}")
    print()
    print("Blocking (PR / main):")
    print(f"  - ruff scoped: paths={list(RUFF_SCOPED_PATHS)} rules=pyproject default")
    print(f"  - ruff critical: paths=. select={RUFF_CRITICAL_SELECT}")
    print(f"  - mypy scoped: paths={list(MYPY_SCOPED_PATHS)} follow_imports=skip")
    print(f"  - pip-audit: locks={list(AUDIT_LOCKS)}")
    print(f"    exceptions: {EXCEPTIONS_PATH.relative_to(REPO_ROOT)}")
    print()
    print("Advisory (documented / optional local):")
    print("  - ruff full-repo E,F statistics (existing debt; expand blocking later)")
    print()
    print("Exception policy:")
    print("  - Every ignored vulnerability needs id, package, reason, review_by")
    print("  - Prefer upgrading the pin in a dedicated dependency PR over long-lived ignores")
    print("  - Do not add application cleanup into the same PR as a gate expansion")


def run_blocking() -> int:
    steps = (
        ("ruff-scoped", run_ruff_scoped),
        ("ruff-critical", run_ruff_critical),
        ("mypy-scoped", run_mypy_scoped),
        ("pip-audit", run_pip_audit),
    )
    failed: list[str] = []
    for name, func in steps:
        print(f"==> {name}", flush=True)
        code = func()
        if code != 0:
            print(f"FAIL {name} (exit {code})", flush=True)
            failed.append(name)
        else:
            print(f"OK   {name}", flush=True)
    if failed:
        print("Blocking quality gates failed:", ", ".join(failed), file=sys.stderr)
        return 1
    print("All blocking quality gates passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "all",
            "lint",
            "lint-critical",
            "typecheck",
            "audit",
            "advisory-lint",
            "policy",
        ),
        help="Gate command to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Ensure repository root imports resolve when invoked as a file path.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    args = build_parser().parse_args(argv)
    if args.command == "policy":
        print_policy()
        return 0
    if args.command == "lint":
        return run_ruff_scoped()
    if args.command == "lint-critical":
        return run_ruff_critical()
    if args.command == "typecheck":
        return run_mypy_scoped()
    if args.command == "audit":
        return run_pip_audit()
    if args.command == "advisory-lint":
        return run_ruff_advisory_repo()
    if args.command == "all":
        return run_blocking()
    raise QualityGateError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualityGateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
