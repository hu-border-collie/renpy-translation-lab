# -*- coding: utf-8 -*-
"""Cross-module regression coverage for the advanced-context work in #341.

These tests exercise local integration contracts only.  Real provider calls and
translation-quality A/B evaluation remain external acceptance activities.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import advanced_context
import embedding_backend
import embedding_runtime
import project_analysis as pa
from rag_memory import JsonSourceIndexStore
import translation_plan


class Issue341IntegrationTests(unittest.TestCase):
    def test_fresh_published_analysis_flows_into_translation_plan(self):
        """A fresh published brief reaches the prompt and plan diagnostics."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            game_dir = tmp_path / "game"
            tl_dir = game_dir / "tl" / "schinese"
            tl_dir.mkdir(parents=True)

            source_rpy = tl_dir / "script.rpy"
            source_rpy.write_text(
                'translate schinese start_123:\n'
                '    # game/script.rpy:10\n'
                '    old "Elena smiles softly as the void gate closes."\n'
                '    new ""\n',
                encoding="utf-8",
            )

            import project_analysis_generate as gen
            fixture_dir = Path(__file__).resolve().parent / "fixtures" / "project_analysis_routes_minimal"
            keywords_file = fixture_dir / "keyword_chunk_summaries.jsonl"
            store_dir = tmp_path / "project_analysis"

            gen.ingest_keyword_summaries(str(keywords_file), store_dir=str(store_dir))
            built = gen.build_structure_drafts(
                store_dir=str(store_dir),
                base_dir=str(fixture_dir),
                script_roots=[str(fixture_dir)],
                entry_labels=["start"],
            )
            pa.publish_project_brief(store_dir=str(store_dir))

            store = pa.ProjectAnalysisStore(store_dir)
            target = next(
                record
                for record in store.load_summaries(pa.KIND_LABEL)
                if record.get("source_files") and record.get("line_span")
            )
            source_fp = built["source_fingerprint"]

            loaded_context = pa.load_injectable_project_context(
                store_dir=str(store_dir),
                base_dir=str(fixture_dir),
                expected_source_fingerprint=source_fp,
                file_rel_path=target["source_files"][0],
                line_numbers=[target["line_span"][0]],
                enabled=True,
            )

            self.assertTrue(loaded_context["injectable"])
            analysis_diagnostics = advanced_context.analysis_skip_diagnostics(loaded_context)
            self.assertTrue(analysis_diagnostics["injectable"])
            self.assertEqual(analysis_diagnostics["brief_status"], "published")
            self.assertEqual(analysis_diagnostics["source_fingerprint"], source_fp)
            self.assertGreater(analysis_diagnostics["injected_chars"], 0)

            analysis_text = advanced_context.render_analysis_reference_text(loaded_context)
            self.assertIn("PROJECT BRIEF:", analysis_text)
            self.assertIn("PROJECT LOCAL CONTEXT:", analysis_text)

            file_jobs = [
                {
                    "file_rel_path": "script.rpy",
                    "file_path": str(source_rpy),
                    "tasks": [
                        {
                            "id": "script.rpy:start_123:10:a1b2c3d4",
                            "text": "Elena smiles softly as the void gate closes.",
                            "line": 10,
                        }
                    ],
                }
            ]

            build = translation_plan.build_translation_plan(
                file_jobs,
                execution_strategy=translation_plan.STRATEGY_SYNC,
                source_identity=translation_plan.SourceIdentity(
                    engine="renpy",
                    adapter_version="1.1.0",
                    file_digests={"script.rpy": "mock-digest"},
                ),
                analysis_blocks_provider=lambda chunk_input: {
                    "text": analysis_text,
                    "diagnostics": {
                        "published_project_analysis": analysis_diagnostics,
                    },
                },
            )

            plan = build.plan
            requests = build.requests
            self.assertEqual(len(requests), 1)
            req = requests[0]
            self.assertIn("PROJECT BRIEF:", req.user_prompt)
            self.assertTrue(plan.plan_fingerprint)
            self.assertEqual(len(plan.request_summaries), 1)
            summary = plan.request_summaries[0]
            analysis_diag_layer = next(
                layer
                for layer in summary["context_diagnostics"]["layers"]
                if layer["layer"] == translation_plan.CONTEXT_LAYER_ANALYSIS
            )
            pa_diag = analysis_diag_layer["diagnostics"]["provider"]["published_project_analysis"]
            self.assertTrue(pa_diag["injectable"])
            self.assertEqual(pa_diag["brief_status"], "published")
            self.assertEqual(pa_diag["source_fingerprint"], source_fp)

    def test_openai_compatible_embedding_identity_guards_source_retrieval(self):
        """An injected non-Gemini adapter preserves identity through retrieval."""
        transport_calls = []

        def mock_transport(**kwargs):
            transport_calls.append(kwargs)
            inputs = kwargs.get("input", [])
            dim = kwargs.get("dimensions", 1536)
            return {
                "data": [
                    {"index": idx, "embedding": [0.05 * (idx + 1)] * dim}
                    for idx in range(len(inputs))
                ],
                "usage": {
                    "prompt_tokens": len(inputs) * 10,
                    "total_tokens": len(inputs) * 10,
                },
            }

        settings_dict = {
            "embedding_backend": "openai_compatible",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_endpoint": "https://api.openai.com/v1",
            "embedding_timeout_seconds": 20.0,
            "output_dimensionality": 1536,
            "embedding_api_key_env": "OPENAI_API_KEY",
        }

        settings = embedding_runtime.parse_embedding_runtime_settings(settings_dict)
        adapter = embedding_runtime.build_embedding_adapter(
            settings,
            openai_transport=mock_transport,
            api_key="mock-key",
        )

        doc_identity = adapter.identity(embedding_backend.EmbeddingTaskType.DOCUMENT)
        query_identity = adapter.identity(embedding_backend.EmbeddingTaskType.QUERY)

        self.assertEqual(adapter.backend, "openai_compatible")
        self.assertTrue(doc_identity.provider.startswith("openai@sha256:"))
        self.assertEqual(doc_identity.model, "text-embedding-3-small")
        self.assertEqual(doc_identity.task_type, embedding_backend.EmbeddingTaskType.DOCUMENT)
        self.assertEqual(doc_identity.output_dimension, 1536)
        self.assertTrue(doc_identity.fingerprint)

        # Batch embed execution
        texts = ["Text segment A", "Text segment B"]
        vectors = embedding_runtime.embed_texts(
            adapter,
            texts,
            embedding_backend.EmbeddingTaskType.DOCUMENT,
            timeout_seconds=20.0,
        )
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 1536)
        self.assertEqual(len(vectors[1]), 1536)
        self.assertEqual(transport_calls[0]["dimensions"], 1536)
        self.assertEqual(transport_calls[0]["model"], "text-embedding-3-small")

        # Source index store persistence and compatibility verification
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSourceIndexStore(tmp)
            report = embedding_runtime.ensure_store_document_identity(store, doc_identity)
            self.assertTrue(report["compatible"])
            self.assertEqual(report["action"], "none")

            store.upsert_segments([
                {
                    "source_id": "s-1",
                    "file_rel_path": "game/script.rpy",
                    "line_start": 1,
                    "line_end": 5,
                    "source_text": texts[0],
                    "embedding": vectors[0],
                }
            ])

            hits, diag = advanced_context.retrieve_source_hits_compatible(
                store,
                [0.05] * 1536,
                query_identity,
                top_k=1,
                min_similarity=0.0,
                char_limit=100,
                query_text="Text segment",
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["source_id"], "s-1")
            self.assertTrue(diag["embedding_compatibility"]["compatible"])

            # Mismatched model query
            mismatched_settings = embedding_runtime.parse_embedding_runtime_settings({
                "embedding_backend": "openai_compatible",
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-large",
                "embedding_endpoint": "https://api.openai.com/v1",
                "output_dimensionality": 1536,
            })
            mismatched_hits, mismatched_diag = advanced_context.retrieve_source_hits_compatible(
                store,
                [0.05] * 1536,
                mismatched_settings.query_identity(),
                top_k=1,
                min_similarity=0.0,
                char_limit=100,
                query_text="Text segment",
            )
            self.assertEqual(len(mismatched_hits), 0)
            self.assertFalse(mismatched_diag["embedding_compatibility"]["compatible"])
            self.assertEqual(mismatched_diag["embedding_compatibility"]["action"], "rebuild_store")
            self.assertIn("model_mismatch", mismatched_diag["embedding_compatibility"]["codes"])

if __name__ == "__main__":
    unittest.main()
