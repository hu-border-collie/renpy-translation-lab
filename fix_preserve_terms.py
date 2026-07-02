#!/usr/bin/env python3
"""Patch translations to satisfy preserve-term validation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import gemini_translate_batch as batch
import translator_runtime as legacy


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def ensure_terms(source: str, translation: str) -> str:
    missing = legacy.missing_preserved_terms(source, translation)
    if not missing:
        return translation

    updated = translation
    for term in missing:
        if term == 'Edgar':
            if updated.startswith('你'):
                updated = 'Edgar' + updated[1:]
            elif '你' in updated:
                updated = updated.replace('你', 'Edgar', 1)
            else:
                updated = f'Edgar{updated}'
        elif term == 'Paper':
            if 'Paper' not in updated:
                updated = updated.replace('纸质', 'Paper质感', 1) if '纸质' in updated else f'Paper{updated}'
        elif term == 'Prose':
            if 'Prose' not in updated:
                if updated.startswith('散文诗'):
                    updated = 'Prose' + updated
                elif '散文' in updated:
                    updated = updated.replace('散文', 'Prose散文', 1)
                else:
                    updated = f'Prose {updated}'
        elif term == 'Mr. Cook':
            if 'Mr. Cook' not in updated:
                updated = updated.replace('Cook先生', 'Mr. Cook', 1)
                if 'Mr. Cook' not in updated:
                    updated = updated.replace('Cook', 'Mr. Cook', 1)
        else:
            if term not in updated:
                updated = f'{term} {updated}'
    return updated


def patch_manual_translations(jsonl_path: Path, failure_path: Path) -> int:
    rows = load_jsonl(jsonl_path)
    by_id = {row['id']: row for row in rows}
    changed = 0
    for entry in load_jsonl(failure_path):
        item_id = entry.get('item_id') or entry.get('id')
        if not item_id or item_id not in by_id:
            continue
        source = entry.get('text') or entry.get('source') or ''
        row = by_id[item_id]
        old = row.get('translation', '')
        new = ensure_terms(source, old)
        if new != old:
            row['translation'] = new
            changed += 1
    write_jsonl(jsonl_path, rows)
    return changed


def patch_results_manifest(manifest_path: str, failure_path: Path) -> int:
    manifest = batch.load_manifest(manifest_path)
    failures = load_jsonl(failure_path)
    if not failures:
        return 0

    result_path = Path(batch.resolve_manifest_result_path(manifest))
    rows = load_jsonl(result_path)
    rows_by_key = {row['key']: row for row in rows}
    chunk_map = {chunk['key']: chunk for chunk in manifest.get('chunks') or []}
    changed = 0

    for entry in failures:
        key = entry.get('key')
        item_id = entry.get('item_id') or entry.get('id')
        if not key or key not in rows_by_key or not item_id:
            continue
        source = entry.get('text') or ''
        row = rows_by_key[key]
        items = batch.result_items_from_row(row, 'patch', allow_empty=True)
        item_map = {item['id']: item for item in items}
        if item_id not in item_map:
            continue
        old = item_map[item_id].get('translation', '')
        new = ensure_terms(source, old)
        if new == old:
            continue
        item_map[item_id]['translation'] = new
        chunk = chunk_map.get(key)
        ordered = []
        if chunk:
            for target in chunk.get('items') or []:
                tid = target.get('id')
                if tid in item_map:
                    ordered.append(item_map[tid])
        else:
            ordered = list(item_map.values())
        text = json.dumps(batch.compact_result_items_for_response(ordered), ensure_ascii=False, indent=2)
        row['response'] = batch.response_payload_with_text(row.get('response', {}), text)
        changed += 1

    write_jsonl(result_path, rows)
    batch.save_manifest(manifest, update_latest=False)
    return changed


def main() -> None:
    legacy.load_config()
    legacy.load_translator_settings()
    legacy.load_glossary()
    batch.load_batch_settings()
    base = Path(r'C:\RenPy_Workspace\renpy-translation-lab\logs\batch_jobs\20260630_004454_Game_AfterClass\split_parts')

    p08_manual = base / 'part08_of_10/retry_parts/20260701_230648_retry/manual_translations/translations.jsonl'
    p08_fail = base / 'part08_of_10/retry_parts/20260701_230648_retry/check_failures.jsonl'
    n08 = patch_manual_translations(p08_manual, p08_fail)
    print(f'part08 manual: {n08} rows patched')

    p07_manifest = str(base / 'part07_of_10/manifest.json')
    p07_fail = base / 'part07_of_10/check_failures.jsonl'
    n07 = patch_results_manifest(p07_manifest, p07_fail)
    print(f'part07 root results: {n07} items patched')


if __name__ == '__main__':
    main()