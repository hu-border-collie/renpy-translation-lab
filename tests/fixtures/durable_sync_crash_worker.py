# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""Abrupt-exit helper for durable Sync recovery tests (never imports tests)."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sync_run_store import SyncRunStore


def main(argv) -> int:
    root_dir, run_id, mode = argv
    store = SyncRunStore(root_dir, run_id)
    owner = 'abrupt-child-owner'
    store.acquire_lease(owner_token=owner, ttl_seconds=0.05)
    attempt_id = store.prepare_attempt(request_id='req-1', owner_token=owner)
    if mode == 'prepared':
        os._exit(91)
    store.dispatch_attempt(attempt_id=attempt_id, owner_token=owner)
    if mode == 'dispatched':
        os._exit(92)
    if mode == 'committed':
        store.record_success(
            attempt_id=attempt_id,
            owner_token=owner,
            accepted_items={'item-1': {'translation': '一'}},
            normalized_payload={
                'translations': [{'id': 'item-1', 'translation': '一'}]
            },
            usage_metadata={'total_tokens': 1},
        )
        os._exit(93)
    raise ValueError(f'unknown crash mode: {mode}')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
