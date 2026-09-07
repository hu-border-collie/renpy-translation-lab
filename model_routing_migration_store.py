"""Explicit staged migration/rollback transactions using the common config store."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config_store import config_write_lock, replace_config_bytes, write_private_artifact
from model_routing_migration import preview_migration


def fingerprint(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_config_bytes(data: bytes) -> dict[str, Any]:
    """Strictly parse an object; reject duplicate keys and non-JSON constants."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON number")

    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Configuration must contain valid UTF-8 JSON without duplicate fields") from exc
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def _file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Configuration must be a regular existing file, not a symlink")
    return path.absolute()


def _preview_bytes(original: bytes):
    config = parse_config_bytes(original)
    if "model_routing" not in config:
        batch = config.get("batch")
        if not isinstance(batch, dict) or not isinstance(batch.get("model"), str) or not batch["model"].strip():
            # File migration deliberately does not load/execute a game's config.
            # Its batch_model fallback cannot be inferred from this file alone.
            raise ValueError("File migration requires explicit batch.model; external project defaults are not loaded")
    return preview_migration(config)


def preview_config_file(path: Path) -> dict[str, Any]:
    """Return credential-safe metadata and a fingerprint without writing anything."""
    original = _file(path).read_bytes()
    report = _preview_bytes(original).public_report()
    report["source_fingerprint"] = fingerprint(original)
    return report


def migrate_config_file(path: Path, *, expected_fingerprint: str) -> dict[str, Any]:
    """Stage v1 explicitly; preserve old runtime fields and an exact-byte backup.

    The report is durable before replacement, so an interruption after commit
    still leaves rollback evidence. Failed precommit writes never replace source.
    """
    path = _file(path)
    with config_write_lock(path):
        original = path.read_bytes()
        if fingerprint(original) != expected_fingerprint:
            raise ValueError("Configuration changed since preview")
        preview = _preview_bytes(original)
        report = preview.public_report()
        report["source_fingerprint"] = fingerprint(original)
        if preview.status == "already_current":
            return report
        newline = "\r\n" if b"\r\n" in original else "\n"
        text = json.dumps(preview.config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        migrated = text.replace("\n", newline).encode("utf-8")
        if original.startswith(b"\xef\xbb\xbf"):
            migrated = b"\xef\xbb\xbf" + migrated
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
        backup = path.with_name(path.name + ".migration-" + stamp + ".bak")
        report_path = backup.with_suffix(".json")
        report.update({
            "report_schema": 1, "config_name": path.name,
            "backup_name": backup.name, "target_fingerprint": fingerprint(migrated),
            "backup_path": str(backup), "report_path": str(report_path),
            "status": "staged",
        })
        write_private_artifact(backup, original, permissions_from=path)
        write_private_artifact(report_path, json.dumps(report, indent=2).encode("utf-8"), permissions_from=path)
        replace_config_bytes(path, migrated, expected=original)
        return report


def rollback_config_file(path: Path, *, report_path: Path) -> dict[str, Any]:
    """Restore exact backup bytes only when source, report and backup still match."""
    path = _file(path)
    report_path = _file(report_path)
    if report_path.parent.resolve() != path.parent.resolve():
        raise ValueError("Migration report must be beside the configuration")
    report = parse_config_bytes(report_path.read_bytes())
    name = report.get("backup_name")
    if (type(report.get("report_schema")) is not int or report["report_schema"] != 1
            or report.get("status") != "staged" or report.get("config_name") != path.name
            or not isinstance(name, str) or Path(name).name != name
            or "/" in name or "\\" in name
            or not name.startswith(path.name + ".migration-") or not name.endswith(".bak")):
        raise ValueError("Invalid migration report")
    backup = _file(path.parent / name)
    original = backup.read_bytes()
    if fingerprint(original) != report.get("source_fingerprint"):
        raise ValueError("Migration backup fingerprint mismatch")
    # A report cannot designate arbitrary bytes as a rollback configuration.
    restored = parse_config_bytes(original)
    if "model_routing" in restored:
        raise ValueError("Rollback source must be a legacy configuration")
    with config_write_lock(path):
        current = path.read_bytes()
        if fingerprint(current) != report.get("target_fingerprint"):
            raise ValueError("Configuration changed since migration; rollback refused")
        replace_config_bytes(path, original, expected=current)
    return {"status": "rolled_back", "source_fingerprint": fingerprint(original)}
