"""Human-readable typed JSON positions without inventing script line numbers."""
import json


def json_source_location(row):
    """Return an MV typed path label, or None for the existing line-based path."""
    locator = row.get('source_locator') or row.get('locator') or {}
    address = locator.get('locator', locator) if isinstance(locator, dict) else {}
    if not isinstance(address, dict):
        return None
    parts = row.get('json_path', address.get('json_path'))
    relative = row.get('file_rel_path') or address.get('file_rel_path')
    if (not relative or not isinstance(parts, (list, tuple)) or not parts
            or any(not ((type(part) is int and part >= 0) or (type(part) is str and part)) for part in parts)):
        return None
    return f"{relative}::{json.dumps(list(parts), ensure_ascii=False, separators=(',', ':'))}"
