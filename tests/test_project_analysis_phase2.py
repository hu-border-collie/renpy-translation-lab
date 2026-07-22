# -*- coding: utf-8 -*-
"""Phase 2 (#254) route structure, generate, publish, and injection gates."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import project_analysis as pa
import project_analysis_generate as gen
import project_analysis_routes as routes
import prompt_context
import translation_core

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "project_analysis_routes_minimal"
SCRIPT = FIXTURE_DIR / "script.rpy"
KEYWORDS = FIXTURE_DIR / "keyword_chunk_summaries.jsonl"


class RouteParseTests(unittest.TestCase):
    def test_fixture_has_two_routes_shared_label_and_unresolved(self):
        graph = routes.build_route_graph(
            [str(SCRIPT)],
            base_dir=str(FIXTURE_DIR),
            entry_labels=["start"],
        )
        self.assertIn("hub", graph.labels)
        self.assertIn("shared_end", graph.labels)
        self.assertGreaterEqual(len(graph.unresolved_edges), 1)
        self.assertTrue(any(e.unresolved for e in graph.unresolved_edges))

        # At least path_a and path_b appear in some route.
        joined = " ".join(
            "->".join(r.get("label_ids") or []) for r in graph.routes
        )
        self.assertIn("path_a", joined)
        self.assertIn("path_b", joined)
        # shared_end is shared across branches
        shared_hits = [
            r for r in graph.routes if "shared_end" in (r.get("label_ids") or [])
        ]
        self.assertGreaterEqual(len(shared_hits), 1)
        # unresolved flag on path_b route
        self.assertTrue(
            any(r.get("unresolved") for r in graph.routes),
            msg=f"expected unresolved route in {graph.routes!r}",
        )


class GenerateAndPublishTests(unittest.TestCase):
    def test_ingest_build_publish_inject_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "project_analysis")
            ingest = gen.ingest_keyword_summaries(
                str(KEYWORDS), store_dir=store_dir
            )
            self.assertEqual(ingest["chunks_written"], 3)

            built = gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=str(FIXTURE_DIR),
                script_roots=[str(FIXTURE_DIR)],
                entry_labels=["start"],
            )
            self.assertGreaterEqual(built["labels"], 4)
            self.assertGreaterEqual(built["routes"], 1)
            self.assertGreaterEqual(built["unresolved_edges"], 1)

            store = pa.ProjectAnalysisStore(store_dir)
            draft = store.load_brief_text(published=False)
            self.assertIn("Project Analysis Brief", draft)
            self.assertFalse(store.load_brief_text(published=True).strip())

            # Draft must not inject.
            blocked = pa.load_injectable_project_brief(
                store_dir=store_dir, enabled=True
            )
            self.assertFalse(blocked["injectable"])
            self.assertEqual(blocked["text"], "")

            pub = pa.publish_project_brief(store_dir=store_dir)
            self.assertTrue(pub["changed"])
            self.assertEqual(pub["status"], pa.STATUS_PUBLISHED)

            allowed = pa.load_injectable_project_brief(
                store_dir=store_dir, enabled=True
            )
            self.assertTrue(allowed["injectable"])
            self.assertIn("Project Analysis Brief", allowed["text"])

            # Stale when expected fingerprint mismatches.
            stale = pa.load_injectable_project_brief(
                store_dir=store_dir,
                enabled=True,
                expected_source_fingerprint="not-the-real-fp",
            )
            self.assertFalse(stale["injectable"])

            unpub = pa.unpublish_project_brief(store_dir=store_dir)
            self.assertIn(unpub["status"], {pa.STATUS_DRAFT, pa.STATUS_MISSING})
            after = pa.load_injectable_project_brief(
                store_dir=store_dir, enabled=True
            )
            self.assertFalse(after["injectable"])

    def test_publish_empty_draft_fails_without_clobbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("# published keep me\n", published=True)
            with self.assertRaises(pa.ProjectAnalysisError):
                pa.publish_project_brief(store_dir=tmp)
            self.assertIn("keep me", store.load_brief_text(published=True))

    def test_local_invalidation_does_not_touch_other_route(self):
        chunks = [
            {
                "id": "chunk-a",
                "kind": "chunk",
                "status": "published",
                "source_files": ["a.rpy"],
                "evidence_item_ids": ["item-a"],
                "upstream_artifact_ids": [],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "A",
            },
            {
                "id": "chunk-b",
                "kind": "chunk",
                "status": "published",
                "source_files": ["b.rpy"],
                "evidence_item_ids": ["item-b"],
                "upstream_artifact_ids": [],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "B",
            },
        ]
        labels = [
            {
                "id": "label:a",
                "kind": "label",
                "status": "published",
                "label_id": "a",
                "source_files": [],
                "evidence_item_ids": [],
                "upstream_artifact_ids": ["chunk-a"],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "A",
            },
            {
                "id": "label:b",
                "kind": "label",
                "status": "published",
                "label_id": "b",
                "source_files": [],
                "evidence_item_ids": [],
                "upstream_artifact_ids": ["chunk-b"],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "B",
            },
        ]
        route_recs = [
            {
                "id": "route-a",
                "kind": "route",
                "status": "published",
                "route_id": "route-a",
                "source_files": [],
                "evidence_item_ids": [],
                "upstream_artifact_ids": ["label:a"],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "A",
            },
            {
                "id": "route-b",
                "kind": "route",
                "status": "published",
                "route_id": "route-b",
                "source_files": [],
                "evidence_item_ids": [],
                "upstream_artifact_ids": ["label:b"],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "B",
            },
        ]
        plan = pa.plan_invalidation(
            chunks=chunks,
            labels=labels,
            routes=route_recs,
            project_brief={
                "id": "project_brief",
                "kind": "project_brief",
                "status": "published",
                "source_files": [],
                "evidence_item_ids": [],
                "upstream_artifact_ids": ["route-a", "route-b"],
                "lineage": pa.empty_lineage(source_fingerprint="fp"),
                "summary": "brief",
            },
            changed_item_ids=["item-a"],
        )
        self.assertIn("route-a", plan.stale_artifact_ids)
        self.assertNotIn("route-b", plan.stale_artifact_ids)
        self.assertTrue(plan.brief_stale)


class PromptInjectionTests(unittest.TestCase):
    def test_reference_blocks_include_brief_only_when_text_present(self):
        without = prompt_context.build_reference_blocks(
            include_translation_memory=False,
            project_brief_text="",
        )
        self.assertNotIn("PROJECT BRIEF", without)

        with_brief = prompt_context.build_reference_blocks(
            include_translation_memory=False,
            project_brief_text="Hello route world",
            project_brief_diagnostics="schema=1 status=published",
        )
        self.assertIn("PROJECT BRIEF", with_brief)
        self.assertIn("Hello route world", with_brief)

    def test_context_bundle_passes_brief_through(self):
        bundle = translation_core.build_context_bundle(
            project_brief_text="Brief body",
            project_brief_diagnostics="fp=abc",
        )
        text = translation_core.build_reference_blocks(
            bundle, include_translation_memory=False
        )
        self.assertIn("PROJECT BRIEF", text)
        self.assertIn("Brief body", text)


class CliPhase2Tests(unittest.TestCase):
    def test_cli_ingest_publish_unpublish(self):
        import gemini_translate_batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            store_dir = os.path.join(tmp, "pa")
            with mock.patch("builtins.print"):
                batch_mod.main(
                    [
                        "project-analysis-ingest-keywords",
                        "--summary-jsonl",
                        str(KEYWORDS),
                        "--store-dir",
                        store_dir,
                    ]
                )
                batch_mod.main(
                    [
                        "project-analysis-build-structure",
                        "--store-dir",
                        store_dir,
                        "--script-root",
                        str(FIXTURE_DIR),
                        "--entry-label",
                        "start",
                    ]
                )
            # Publish prints JSON
            printed = []

            def capture(*args, **kwargs):
                if args:
                    printed.append(str(args[0]))

            with mock.patch("builtins.print", side_effect=capture):
                batch_mod.main(
                    ["project-analysis-publish", "--store-dir", store_dir]
                )
            payload = json.loads(printed[-1])
            self.assertEqual(payload["status"], "published")

            allowed = pa.load_injectable_project_brief(store_dir=store_dir, enabled=True)
            self.assertTrue(allowed["injectable"])


if __name__ == "__main__":
    unittest.main()
