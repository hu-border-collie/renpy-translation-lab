# -*- coding: utf-8 -*-
"""Offline tests for Project Analysis LLM map-reduce (#254 PR B)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import project_analysis as pa
import project_analysis_generate as gen
import project_analysis_llm as llm
from sync_model_backend import SyncGenerationRequest, SyncGenerationResult, SYNC_EXECUTION_MODE

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "project_analysis_routes_minimal"
SCRIPT = FIXTURE_DIR / "script.rpy"
KEYWORDS = FIXTURE_DIR / "keyword_chunk_summaries.jsonl"


class FakeBackend:
    provider = "fake"

    def __init__(self) -> None:
        self.calls: list[SyncGenerationRequest] = []

    def generate(self, request: SyncGenerationRequest) -> SyncGenerationResult:
        self.calls.append(request)
        system = str((request.config or {}).get("system_instruction") or "")
        user = ""
        contents = request.contents or []
        if contents and isinstance(contents[0], dict):
            parts = contents[0].get("parts") or []
            if parts and isinstance(parts[0], dict):
                user = str(parts[0].get("text") or "")
        if "Label id:" in user:
            text = "LLM-LABEL: refined label summary for testing."
        elif "Route id:" in user:
            text = "LLM-ROUTE: refined route summary for testing."
        else:
            text = "# LLM Brief\n\nRefined project brief for testing."
        return SyncGenerationResult(
            provider=self.provider,
            model=request.model,
            execution_mode=SYNC_EXECUTION_MODE,
            response_payload={"text": text},
            response_text=text,
        )


class MapReduceTests(unittest.TestCase):
    def test_mapreduce_refines_labels_routes_brief(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "pa")
            gen.ingest_keyword_summaries(str(KEYWORDS), store_dir=store_dir)
            gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            backend = FakeBackend()
            result = llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "fake-model"},
                force=True,
            )
            self.assertGreaterEqual(result["labels_refined"], 1)
            self.assertGreaterEqual(result["routes_refined"], 1)
            self.assertGreater(len(backend.calls), 0)
            self.assertEqual(result["prompt_schema_version"], llm.PROMPT_SCHEMA_VERSION)

            store = pa.ProjectAnalysisStore(store_dir)
            labels = store.load_summaries(pa.KIND_LABEL)
            self.assertTrue(any("LLM-LABEL" in (r.get("summary") or "") for r in labels))
            routes = store.load_routes()
            self.assertTrue(any("LLM-ROUTE" in (r.get("summary") or "") for r in routes))
            brief = store.load_brief_text(published=False)
            self.assertIn("LLM Brief", brief)
            # Evidence IDs preserved on at least one refined label when present.
            with_ids = [r for r in labels if r.get("evidence_item_ids")]
            self.assertTrue(with_ids)

    def test_skip_when_already_refined(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "pa")
            gen.ingest_keyword_summaries(str(KEYWORDS), store_dir=store_dir)
            gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            backend = FakeBackend()
            first = llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "fake-model"},
                force=True,
                provider="fake",
            )
            calls_after_first = len(backend.calls)
            second = llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "fake-model"},
                force=False,
                provider="fake",
            )
            self.assertGreaterEqual(first["labels_refined"], 1)
            self.assertEqual(second["labels_refined"], 0)
            self.assertGreaterEqual(second["labels_skipped"], 1)
            # Only brief may still call if skip logic differs; labels/routes skip.
            self.assertLessEqual(len(backend.calls) - calls_after_first, 1)

    def test_model_switch_invalidates_label_route_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "pa")
            gen.ingest_keyword_summaries(str(KEYWORDS), store_dir=store_dir)
            gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            backend = FakeBackend()
            llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "model-a", "thinking_level": ""},
                force=True,
                provider="fake",
                model="model-a",
            )
            calls_mid = len(backend.calls)
            result = llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "model-b", "thinking_level": "high"},
                force=False,
                provider="fake",
                model="model-b",
            )
            self.assertGreaterEqual(result["labels_refined"], 1)
            self.assertGreater(len(backend.calls), calls_mid)
            store = pa.ProjectAnalysisStore(store_dir)
            labels = store.load_summaries(pa.KIND_LABEL)
            self.assertTrue(
                any(
                    (r.get("lineage") or {}).get("model") == "model-b"
                    for r in labels
                )
            )

    def test_structure_rebuild_preserves_unaffected_llm_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "pa")
            gen.ingest_keyword_summaries(str(KEYWORDS), store_dir=store_dir)
            gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            backend = FakeBackend()
            llm.run_mapreduce_drafts(
                store_dir=store_dir,
                backend=backend,
                config={"model": "fake-model"},
                force=True,
                provider="fake",
            )
            store = pa.ProjectAnalysisStore(store_dir)
            before = {r["id"]: r.get("summary") for r in store.load_summaries(pa.KIND_LABEL)}
            # Rebuild structure without script content change → preserve LLM text.
            stats = gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            self.assertGreaterEqual(stats["label_merge"]["preserved"], 1)
            after = {r["id"]: r.get("summary") for r in store.load_summaries(pa.KIND_LABEL)}
            for lid, summary in before.items():
                if summary and "LLM-LABEL" in str(summary):
                    self.assertEqual(after.get(lid), summary)

    def test_requires_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(pa.ProjectAnalysisError):
                llm.run_mapreduce_drafts(
                    store_dir=tmp,
                    backend=FakeBackend(),
                    config={"model": "fake-model"},
                )


if __name__ == "__main__":
    unittest.main()
