#!/usr/bin/env python3
"""Import manual_translations/translations.jsonl into Batch results.jsonl and merge up."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import gemini_translate_batch as batch


def load_translations(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def enrich_translations_from_manifest(manifest: dict, translations: list[dict]) -> list[dict]:
    item_index: dict[str, dict] = {}
    for chunk in manifest.get('chunks') or []:
        chunk_key = chunk.get('key', '')
        parent_key = chunk.get('retry_parent_key') or chunk_key
        file_rel_path = chunk.get('file_rel_path', '')
        for item in chunk.get('items') or []:
            item_id = item.get('id')
            if not item_id:
                continue
            item_index[item_id] = {
                'chunk_key': chunk_key,
                'parent_key': parent_key,
                'file_rel_path': file_rel_path,
                'line': item.get('line'),
                'source': item.get('text', item.get('source', '')),
            }

    enriched = []
    for row in translations:
        merged = dict(row)
        meta = item_index.get(merged.get('id', ''))
        if meta:
            for key, value in meta.items():
                merged.setdefault(key, value)
        if not merged.get('chunk_key'):
            raise SystemExit(f"Missing chunk_key in manual translation row: {merged.get('id')}")
        enriched.append(merged)
    return enriched


def build_results_rows(manifest: dict, translations: list[dict]) -> list[dict]:
    translations = enrich_translations_from_manifest(manifest, translations)
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for row in translations:
        chunk_key = row.get('chunk_key')
        if not chunk_key:
            raise SystemExit(f"Missing chunk_key in manual translation row: {row.get('id')}")
        by_chunk[chunk_key].append(row)

    result_rows = []
    for chunk in manifest.get('chunks') or []:
        chunk_key = chunk.get('key')
        if not chunk_key:
            continue
        chunk_translations = by_chunk.get(chunk_key)
        if not chunk_translations:
            raise SystemExit(f"Missing manual translations for chunk: {chunk_key}")

        item_ids = [item.get('id') for item in chunk.get('items') or []]
        translation_by_id = {row['id']: row['translation'] for row in chunk_translations}
        missing = [item_id for item_id in item_ids if item_id not in translation_by_id]
        if missing:
            raise SystemExit(f"Chunk {chunk_key} missing translation ids: {missing[:3]}")

        result_items = [
            {'id': item_id, 'translation': translation_by_id[item_id]}
            for item_id in item_ids
        ]
        response_text = json.dumps(
            batch.compact_result_items_for_response(result_items),
            ensure_ascii=False,
            indent=2,
        )
        result_rows.append(
            {
                'key': chunk_key,
                'response': batch.response_payload_with_text({}, response_text),
            }
        )
    return result_rows


def import_manual_translations(manifest_path: str | Path) -> Path:
    manifest = batch.load_manifest(str(manifest_path))
    package_dir = Path(manifest['_package_dir'])
    manual_dir = package_dir / 'manual_translations'
    translations_path = manual_dir / 'translations.jsonl'
    if not translations_path.is_file():
        raise SystemExit(f'Manual translations not found: {translations_path}')

    translations = load_translations(translations_path)
    result_rows = build_results_rows(manifest, translations)
    results_path = package_dir / 'results.jsonl'
    batch.write_jsonl_file(str(results_path), result_rows)

    manifest['result_jsonl_path'] = 'results.jsonl'
    manifest['job_state'] = 'JOB_STATE_SUCCEEDED'
    batch.save_manifest(manifest, update_latest=False)
    print(f'Imported {len(result_rows)} result rows -> {results_path}')
    return results_path


def retry_merge_chain(leaf_manifest_path: str | Path) -> list[tuple[str, str]]:
    chain = []
    current = Path(leaf_manifest_path)
    visited = {str(current)}
    while True:
        manifest = batch.load_manifest(str(current))
        parent = manifest.get('retry_of_manifest')
        if not parent:
            break
        parent_text = str(parent)
        if parent_text in visited:
            raise SystemExit(f'Detected cyclic retry_of_manifest chain at {parent_text}')
        visited.add(parent_text)
        chain.append((parent_text, str(current)))
        current = Path(parent)
    return chain


def run_batch_command(command: str, *args: str) -> None:
    cmd = [sys.executable, str(Path(__file__).with_name('gemini_translate_batch.py')), command, *args]
    print('>', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def check_safety(manifest_path: str | Path) -> str:
    manifest = batch.load_manifest(str(manifest_path))
    summary = manifest.get('last_check_summary') or {}
    return str(summary.get('safety_level') or '')


def process_package(manifest_path: str | Path, apply_root: bool = False) -> dict:
    manifest_path = str(manifest_path)
    import_manual_translations(manifest_path)
    run_batch_command('check', manifest_path)
    safety = check_safety(manifest_path)
    if safety != batch.CHECK_SAFETY_SAFE:
        return {
            'manifest': manifest_path,
            'status': 'check_failed_leaf',
            'safety': safety,
        }

    chain = retry_merge_chain(manifest_path)
    root_manifest = chain[-1][0] if chain else manifest_path
    for parent, retry in chain:
        run_batch_command('merge-retry', parent, retry)
        run_batch_command('check', parent)

    root_safety = check_safety(root_manifest)
    result = {
        'manifest': manifest_path,
        'root_manifest': root_manifest,
        'merge_steps': len(chain),
        'status': 'merged',
        'root_safety': root_safety,
    }
    if apply_root and root_safety == batch.CHECK_SAFETY_SAFE:
        run_batch_command('apply', root_manifest)
        result['status'] = 'applied'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', help='Manual retry manifest paths')
    parser.add_argument('--import-only', action='store_true', help='Only build results.jsonl')
    parser.add_argument('--apply', action='store_true', help='Apply root split part when safe')
    args = parser.parse_args()

    for manifest_path in args.manifests:
        if args.import_only:
            import_manual_translations(manifest_path)
            continue
        result = process_package(manifest_path, apply_root=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        root_safety = result.get('root_safety')
        if root_safety == batch.CHECK_SAFETY_SAFE:
            continue
        if root_safety == 'warn':
            print(f'WARNING: root check is warn for {manifest_path}; apply may be refused until warnings are fixed.')
            continue
        if not args.import_only:
            raise SystemExit(f"Root check not safe for {manifest_path}: {result}")


if __name__ == '__main__':
    main()