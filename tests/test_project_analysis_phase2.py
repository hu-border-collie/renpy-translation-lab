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

    def test_safe_relpath_cross_drive_does_not_raise(self):
        # Simulate Windows cross-drive: function must not raise ValueError.
        path = str(SCRIPT)
        other_base = "D:\\not\\this\\drive" if os.name == "nt" else "/mnt/other"
        rel = routes.safe_relpath(path, other_base)
        self.assertTrue(rel)
        self.assertNotIn("..", rel.split("/"))

    def test_call_is_not_exclusive_branch(self):
        text = (
            "label start:\n"
            "    call helper\n"
            "    jump ending\n"
            "\n"
            "label helper:\n"
            "    \"side\"\n"
            "    return\n"
            "\n"
            "label ending:\n"
            "    \"done\"\n"
            "    return\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "call_flow.rpy"
            script.write_text(text, encoding="utf-8")
            graph = routes.build_route_graph(
                [str(script)], base_dir=tmp, entry_labels=["start"]
            )
            paths = ["->".join(r.get("label_ids") or []) for r in graph.routes]
            # Must not invent mutually exclusive start→helper vs start→ending only.
            self.assertTrue(
                any(p == "start->ending" or p.startswith("start->ending") for p in paths)
                or any("ending" in p and "start" in p for p in paths),
                msg=paths,
            )
            # helper should not be a sole exclusive sibling route without ending
            exclusive_helper = [
                p for p in paths if p == "start->helper" or p.endswith("->helper")
            ]
            self.assertEqual(exclusive_helper, [], msg=paths)

    def test_script_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.rpy"
            script.write_text("label a:\n    jump b\n\nlabel b:\n    return\n", encoding="utf-8")
            fp1 = routes.digest_script_paths([str(script)], base_dir=tmp)
            script.write_text(
                "label a:\n    jump b\n\nlabel b:\n    \"changed\"\n    return\n",
                encoding="utf-8",
            )
            fp2 = routes.digest_script_paths([str(script)], base_dir=tmp)
            self.assertNotEqual(fp1, fp2)

    def test_call_expression_not_parsed_as_named_call(self):
        text = "label a:\n    call expression some_flag\n    call helper\n"
        labels, edges = routes.parse_rpy_labels_and_edges(text, file_rel_path="x.rpy")
        kinds = {(e.kind, e.target_label, e.unresolved) for e in edges}
        self.assertIn(("unresolved", "", True), kinds)
        self.assertIn(("call", "helper", False), kinds)
        self.assertNotIn(("call", "expression", False), kinds)

    def test_persisted_script_roots_used_for_fingerprint(self):
        import gemini_translate_batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom_scripts"
            custom.mkdir()
            (custom / "only.rpy").write_text(
                "label start:\n    jump end\n\nlabel end:\n    return\n",
                encoding="utf-8",
            )
            store_dir = os.path.join(tmp, "pa")
            # Default game layout under tmp is empty; structure uses custom root.
            built = gen.build_structure_drafts(
                store_dir=store_dir,
                base_dir=tmp,
                script_roots=[str(custom)],
                entry_labels=["start"],
            )
            self.assertEqual(built["source_fingerprint"], routes.digest_script_paths(
                [str(custom / "only.rpy")], base_dir=tmp
            ))
            # Runtime fingerprint must reuse stored roots, not only default layout.
            fp = batch_mod.compute_current_project_analysis_fingerprint(
                tmp, store_dir=store_dir
            )
            self.assertEqual(fp, built["source_fingerprint"])


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
            structure_fp = built["source_fingerprint"]
            self.assertTrue(structure_fp)

            store = pa.ProjectAnalysisStore(store_dir)
            labels = store.load_summaries(pa.KIND_LABEL)
            label_text = "\n".join(str(r.get("summary") or "") for r in labels)
            # Keyword spans must land on path_a / path_b / hub (not only unassigned).
            self.assertIn("Path A", label_text)
            self.assertIn("Path B", label_text)
            self.assertIn("hub", label_text.lower())

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
                store_dir=store_dir,
                enabled=True,
                expected_source_fingerprint=structure_fp,
            )
            self.assertTrue(allowed["injectable"])
            self.assertIn("Project Analysis Brief", allowed["text"])

            # Script edit changes fingerprint → injection blocked (use temp copy).
            with tempfile.TemporaryDirectory() as tmp2:
                script_copy = Path(tmp2) / "script.rpy"
                script_copy.write_text(
                    (FIXTURE_DIR / "script.rpy").read_text(encoding="utf-8")
                    + "\n# edited\n",
                    encoding="utf-8",
                )
                new_fp = routes.digest_script_paths(
                    [str(script_copy)], base_dir=tmp2
                )
                self.assertNotEqual(new_fp, structure_fp)
                stale = pa.load_injectable_project_brief(
                    store_dir=store_dir,
                    enabled=True,
                    expected_source_fingerprint=new_fp,
                )
                self.assertFalse(stale["injectable"])

            unpub = pa.unpublish_project_brief(store_dir=store_dir)
            self.assertIn(unpub["status"], {pa.STATUS_DRAFT, pa.STATUS_MISSING})
            after = pa.load_injectable_project_brief(
                store_dir=store_dir,
                enabled=True,
                expected_source_fingerprint=structure_fp,
            )
            self.assertFalse(after["injectable"])

    def test_publish_rejects_stale_without_force_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("# draft body\n", published=False)
            manifest = pa.empty_manifest(store_dir=tmp)
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF] = {
                "status": pa.STATUS_STALE,
                "draft_present": True,
                "published_present": False,
                "lineage": pa.empty_lineage(source_fingerprint="old-fp"),
                "id": "project_brief",
            }
            store.save_manifest(manifest)
            with self.assertRaises(pa.ProjectAnalysisError):
                pa.publish_project_brief(store_dir=tmp)
            # force without current fingerprint still fails
            with self.assertRaises(pa.ProjectAnalysisError):
                pa.publish_project_brief(store_dir=tmp, force=True)
            # force + current fingerprint allowed
            out = pa.publish_project_brief(
                store_dir=tmp,
                force=True,
                current_source_fingerprint="new-fp",
            )
            self.assertEqual(out["status"], pa.STATUS_PUBLISHED)

    def test_publish_empty_draft_fails_without_clobbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("# published keep me\n", published=True)
            with self.assertRaises(pa.ProjectAnalysisError):
                pa.publish_project_brief(store_dir=tmp)
            self.assertIn("keep me", store.load_brief_text(published=True))

    def test_publish_rejects_missing_lineage_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("# draft only\n", published=False)
            manifest = pa.empty_manifest(store_dir=tmp)
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF] = {
                "status": pa.STATUS_DRAFT,
                "draft_present": True,
                "published_present": False,
                "lineage": pa.empty_lineage(),
                "id": "project_brief",
            }
            store.save_manifest(manifest)
            with self.assertRaisesRegex(
                pa.ProjectAnalysisError, "source_fingerprint"
            ):
                pa.publish_project_brief(store_dir=tmp)

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
