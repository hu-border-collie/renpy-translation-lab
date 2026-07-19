"""Generate or verify hashed Python 3.11 dependency locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = REPO_ROOT / "requirements-lock"
MANIFEST_PATH = LOCK_DIR / "manifest.json"
UV_VERSION = "0.11.29"
PYTHON_VERSION = "3.11"
PROFILES: dict[str, tuple[str, ...]] = {
    "cli": ("requirements.txt",),
    "gui": ("requirements.txt", "requirements-gui.txt"),
    "litellm": ("requirements.txt", "requirements-litellm.txt"),
}
PLATFORMS = {
    "windows": "windows",
    "linux": "x86_64-manylinux_2_34",
}
OWNED_SOURCES = (
    ".gitattributes",
    "requirements.txt",
    "requirements-core.txt",
    "requirements-genai.txt",
    "requirements-gui.txt",
    "requirements-litellm.txt",
    "relation_analyzer/requirements.txt",
    "relation_analyzer/requirements-semantic.txt",
    "scripts/compile_dependency_locks.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_relative_path(platform: str, profile: str) -> str:
    return f"requirements-lock/py311-{platform}-{profile}.txt"


def expected_locks() -> tuple[str, ...]:
    return tuple(
        lock_relative_path(platform, profile)
        for platform in PLATFORMS
        for profile in PROFILES
    )


def manifest_payload(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator": {"name": "uv", "version": UV_VERSION},
        "python_version": PYTHON_VERSION,
        "sources": {
            relative: sha256_file(repo_root / relative)
            for relative in OWNED_SOURCES
        },
        "locks": {
            relative: sha256_file(repo_root / relative)
            for relative in expected_locks()
        },
    }


def verify_manifest(
    repo_root: Path = REPO_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> list[str]:
    if not manifest_path.is_file():
        return [f"missing lock manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid lock manifest: {exc}"]
    return verify_manifest_payload(manifest, repo_root)


def verify_manifest_payload(
    manifest: object,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest root must be an object"]
    errors: list[str] = []
    expected_metadata = {
        "schema_version": 1,
        "generator": {"name": "uv", "version": UV_VERSION},
        "python_version": PYTHON_VERSION,
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest {key} is stale")

    expected_groups = {
        "sources": OWNED_SOURCES,
        "locks": expected_locks(),
    }
    for group, expected_paths in expected_groups.items():
        recorded = manifest.get(group)
        if not isinstance(recorded, dict):
            errors.append(f"manifest {group} must be an object")
            continue
        if set(recorded) != set(expected_paths):
            errors.append(f"manifest {group} path set is stale")
        for relative in expected_paths:
            path = repo_root / relative
            if not path.is_file():
                errors.append(f"missing {group[:-1]} file: {relative}")
                continue
            actual = sha256_file(path)
            if recorded.get(relative) != actual:
                errors.append(f"stale or manually edited {group[:-1]}: {relative}")
    return errors


def verify_uv(uv_command: str) -> None:
    completed = subprocess.run(
        [uv_command, "--version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not completed.stdout.startswith(f"uv {UV_VERSION} "):
        raise RuntimeError(
            f"lock generation requires uv {UV_VERSION}; got {completed.stdout.strip()}"
        )


def generate_locks(uv_command: str, *, upgrade: bool = False) -> None:
    verify_uv(uv_command)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for platform, uv_platform in PLATFORMS.items():
        for profile, inputs in PROFILES.items():
            output = REPO_ROOT / lock_relative_path(platform, profile)
            command = [
                uv_command,
                "pip",
                "compile",
                *inputs,
                "--python-version",
                PYTHON_VERSION,
                "--python-platform",
                uv_platform,
                "--generate-hashes",
                "--no-annotate",
                "--no-header",
                "--only-binary",
                ":all:",
                "--output-file",
                str(output),
                "--quiet",
            ]
            if upgrade:
                command.append("--upgrade")
            print(f"Generating {output.relative_to(REPO_ROOT)}")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if "--hash=sha256:" not in output.read_text(encoding="utf-8"):
                raise RuntimeError(f"generated lock has no hashes: {output}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed source and lock hashes without network access",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="allow transitive upgrades while regenerating all locks",
    )
    parser.add_argument("--uv", default="uv", help="uv executable path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        errors = verify_manifest()
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            print("Regenerate locks with scripts/compile_dependency_locks.py.")
            return 1
        print("Dependency lock manifest is current.")
        return 0
    generate_locks(args.uv, upgrade=args.upgrade)
    print("Dependency locks and manifest updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
