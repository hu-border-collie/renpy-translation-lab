import json
import math
import tempfile
import unittest
from types import SimpleNamespace

from embedding_adapters import (
    GeminiEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
    build_provider_identity,
)
from embedding_backend import (
    EmbeddingBackendError,
    EmbeddingBatchRequest,
    EmbeddingContractError,
    EmbeddingErrorCategory,
    EmbeddingTaskType,
)
from rag_memory import (
    EmbeddingStoreIdentityError,
    JsonRagStore,
    JsonSourceIndexStore,
)


class _GeminiModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _StatusError(Exception):
    def __init__(self, status_code, secret):
        self.status_code = status_code
        super().__init__(secret)


class EmbeddingAdapterTests(unittest.TestCase):
    @staticmethod
    def request(adapter, task=EmbeddingTaskType.DOCUMENT, inputs=("alpha", "beta")):
        return EmbeddingBatchRequest(
            identity=adapter.identity(task),
            inputs=inputs,
            timeout_seconds=12.5,
            metadata={"purpose": "offline-test"},
        )

    def test_gemini_maps_document_task_timeout_usage_and_batch(self):
        response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[1, 0],
                    statistics=SimpleNamespace(token_count=3, truncated=False),
                ),
                SimpleNamespace(
                    values=[0, 1],
                    statistics=SimpleNamespace(token_count=4, truncated=True),
                ),
            ],
            metadata=SimpleNamespace(billable_character_count=11),
        )
        models = _GeminiModels(response=response)
        configs = []
        adapter = GeminiEmbeddingAdapter(
            client=SimpleNamespace(models=models),
            model="gemini-embedding-001",
            output_dimension=2,
            config_factory=lambda **kwargs: configs.append(kwargs) or kwargs,
        )

        result = adapter.embed(self.request(adapter))

        self.assertEqual(result.vectors, ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(result.usage.input_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(result.usage.metadata["billable_character_count"], 11)
        self.assertEqual(result.usage.metadata["truncated_input_count"], 1)
        self.assertEqual(configs[0]["task_type"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(configs[0]["timeout_seconds"], 12.5)
        self.assertEqual(models.calls[0]["contents"], ["alpha", "beta"])

    def test_gemini_maps_query_task(self):
        models = _GeminiModels(
            response=SimpleNamespace(embeddings=[SimpleNamespace(values=[1, 0])])
        )
        configs = []
        adapter = GeminiEmbeddingAdapter(
            client=SimpleNamespace(models=models),
            model="gemini-embedding-001",
            output_dimension=2,
            config_factory=lambda **kwargs: configs.append(kwargs) or kwargs,
        )

        adapter.embed(self.request(adapter, EmbeddingTaskType.QUERY, ("query",)))

        self.assertEqual(configs[0]["task_type"], "RETRIEVAL_QUERY")

    def test_openai_compatible_reorders_batch_and_forwards_timeout(self):
        calls = []

        def transport(**kwargs):
            calls.append(kwargs)
            return {
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }

        adapter = OpenAICompatibleEmbeddingAdapter(
            transport=transport,
            model="openai/text-embedding-3-small",
            output_dimension=2,
            provider="openai",
            endpoint="https://api.example.test/v1/",
            api_key="fake-key-value",
            request_headers={"Authorization": "Bearer super-secret"},
        )

        result = adapter.embed(self.request(adapter))

        self.assertEqual(result.vectors, ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(calls[0]["timeout"], 12.5)
        self.assertEqual(calls[0]["dimensions"], 2)
        self.assertEqual(calls[0]["api_key"], "fake-key-value")

    def test_count_dimension_and_non_finite_drift_are_invalid_response(self):
        bad_payloads = (
            {"data": [{"index": 0, "embedding": [1, 0]}]},
            {
                "data": [
                    {"index": 0, "embedding": [1, 0, 0]},
                    {"index": 1, "embedding": [0, 1, 0]},
                ]
            },
            {
                "data": [
                    {"index": 0, "embedding": [math.nan, 0]},
                    {"index": 1, "embedding": [0, 1]},
                ]
            },
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                adapter = OpenAICompatibleEmbeddingAdapter(
                    transport=lambda **_kwargs: payload,
                    model="embed-model",
                    output_dimension=2,
                    provider="local",
                )
                with self.assertRaises(EmbeddingBackendError) as raised:
                    adapter.embed(self.request(adapter))
                self.assertEqual(raised.exception.category, EmbeddingErrorCategory.INVALID_RESPONSE)

    def test_timeout_and_provider_error_are_classified_and_redacted(self):
        secrets = (
            (TimeoutError("https://secret.test/?api_key=oops"), EmbeddingErrorCategory.TIMEOUT),
            (
                _StatusError(401, "Authorization: Bearer hidden"),
                EmbeddingErrorCategory.AUTHENTICATION,
            ),
            (_StatusError(429, "provider raw body secret"), EmbeddingErrorCategory.RATE_LIMIT),
        )
        for error, category in secrets:
            with self.subTest(category=category):
                adapter = OpenAICompatibleEmbeddingAdapter(
                    transport=lambda **_kwargs: (_ for _ in ()).throw(error),
                    model="embed-model",
                    output_dimension=2,
                    provider="local",
                )
                with self.assertRaises(EmbeddingBackendError) as raised:
                    adapter.embed(self.request(adapter))
                public = str(raised.exception)
                self.assertEqual(raised.exception.category, category)
                self.assertNotIn("secret", public.lower())
                self.assertNotIn("authorization", public.lower())
                self.assertNotIn("http", public.lower())

    def test_provider_identity_is_stable_endpoint_specific_and_credential_free(self):
        first = build_provider_identity(
            backend="openai_compatible",
            provider="custom",
            endpoint="HTTPS://Example.Test:443/v1/",
            default_endpoint="unused",
            configuration={"deployment": "blue", "routing": {"region": "east"}},
        )
        second = build_provider_identity(
            backend="openai_compatible",
            provider="custom",
            endpoint="https://example.test/v1",
            default_endpoint="unused",
            configuration={"routing": {"region": "east"}, "deployment": "blue"},
        )
        changed = build_provider_identity(
            backend="openai_compatible",
            provider="custom",
            endpoint="https://example.test/v2",
            default_endpoint="unused",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotIn("example.test", first)

        with self.assertRaises(EmbeddingContractError):
            build_provider_identity(
                backend="openai_compatible",
                provider="custom",
                endpoint="https://example.test/v1?token=secret",
                default_endpoint="unused",
            )
        with self.assertRaises(EmbeddingContractError):
            build_provider_identity(
                backend="openai_compatible",
                provider="custom",
                default_endpoint="default",
                configuration={"Authorization": "Bearer hidden"},
            )
        with self.assertRaises(EmbeddingContractError):
            build_provider_identity(
                backend="openai_compatible",
                provider="custom",
                default_endpoint="default",
                configuration={"routing": {"fallback": "https://example.test/v1?signature=hidden"}},
            )

    def test_adapter_identity_does_not_include_key_header_or_endpoint(self):
        adapter = OpenAICompatibleEmbeddingAdapter(
            transport=lambda **_kwargs: {"data": []},
            model="embed-model",
            output_dimension=2,
            provider="custom",
            endpoint="https://private.example.test/v1",
            api_key="fake-key-value",
            request_headers={"Authorization": "Bearer hidden"},
        )
        serialized = json.dumps(adapter.identity(EmbeddingTaskType.DOCUMENT).to_dict())
        self.assertNotIn("hidden", serialized)
        self.assertNotIn("private.example.test", serialized)
        self.assertNotIn("Authorization", serialized)


class EmbeddingStoreIdentityTests(unittest.TestCase):
    @staticmethod
    def adapter(provider="local", dimension=2):
        return OpenAICompatibleEmbeddingAdapter(
            transport=lambda **_kwargs: {"data": []},
            model="embed-model",
            output_dimension=dimension,
            provider=provider,
        )

    def test_rag_store_persists_identity_and_allows_compatible_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRagStore(tmp)
            adapter = self.adapter()
            store.set_embedding_identity(adapter.identity(EmbeddingTaskType.DOCUMENT))
            store.upsert_history(
                [{"memory_id": "m1", "embedding": [1.0, 0.0], "source_text": "alpha"}]
            )

            results, diagnostics = store.search_history_compatible(
                [1.0, 0.0],
                adapter.identity(EmbeddingTaskType.QUERY),
                min_similarity=0.0,
            )

            self.assertEqual([item["memory_id"] for item in results], ["m1"])
            self.assertTrue(diagnostics["embedding_compatibility"]["compatible"])
            reloaded = JsonRagStore(tmp)
            reloaded.load()
            self.assertEqual(
                reloaded.metadata["embedding_identity"]["backend"],
                "openai_compatible",
            )

    def test_source_store_mismatch_fails_closed_with_field_rebuild_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSourceIndexStore(tmp)
            document_adapter = self.adapter(provider="alpha")
            query_adapter = self.adapter(provider="beta", dimension=3)
            store.set_embedding_identity(document_adapter.identity(EmbeddingTaskType.DOCUMENT))
            store.upsert_segments(
                [{"source_id": "s1", "embedding": [1.0, 0.0], "source_text": "alpha"}]
            )

            results, diagnostics = store.search_segments_compatible(
                [1.0, 0.0, 0.0],
                query_adapter.identity(EmbeddingTaskType.QUERY),
                min_similarity=0.0,
            )

            report = diagnostics["embedding_compatibility"]
            self.assertEqual(results, [])
            self.assertFalse(report["compatible"])
            self.assertEqual(report["action"], "rebuild_store")
            self.assertEqual(
                report["codes"],
                ["provider_mismatch", "dimension_mismatch"],
            )
            self.assertEqual(
                [item["field"] for item in report["mismatches"]],
                ["provider", "output_dimension"],
            )

    def test_missing_and_invalid_identity_fail_closed_without_echoing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRagStore(tmp)
            query_identity = self.adapter().identity(EmbeddingTaskType.QUERY)
            _results, missing = store.search_history_compatible([1, 0], query_identity)
            self.assertEqual(
                missing["embedding_compatibility"]["codes"],
                ["store_identity_missing"],
            )
            store.set_metadata(embedding_identity={"api_key": "do-not-echo"})
            _results, invalid = store.search_history_compatible([1, 0], query_identity)
            serialized = json.dumps(invalid)
            self.assertIn("store_identity_invalid", serialized)
            self.assertNotIn("do-not-echo", serialized)
            self.assertEqual(invalid["embedding_compatibility"]["action"], "rebuild_store")

    def test_nonempty_store_identity_cannot_be_silently_attached_or_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRagStore(tmp)
            store.upsert_history(
                [{"memory_id": "legacy", "embedding": [1.0, 0.0], "source_text": "alpha"}]
            )
            with self.assertRaises(EmbeddingStoreIdentityError) as missing:
                store.set_embedding_identity(self.adapter().identity(EmbeddingTaskType.DOCUMENT))
            self.assertEqual(missing.exception.action, "rebuild_store")

        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSourceIndexStore(tmp)
            first = self.adapter(provider="alpha")
            store.set_embedding_identity(first.identity(EmbeddingTaskType.DOCUMENT))
            store.upsert_segments(
                [{"source_id": "s1", "embedding": [1.0, 0.0], "source_text": "alpha"}]
            )
            store.set_embedding_identity(first.identity(EmbeddingTaskType.DOCUMENT))
            with self.assertRaises(EmbeddingStoreIdentityError):
                store.set_embedding_identity(
                    self.adapter(provider="beta").identity(EmbeddingTaskType.DOCUMENT)
                )


if __name__ == "__main__":
    unittest.main()
