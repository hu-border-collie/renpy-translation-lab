import json
import importlib.util
import io
import math
import os
from contextlib import redirect_stderr, redirect_stdout
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class _GoogleAPIError(Exception):
    def __init__(self, code, status, secret):
        self.code = code
        self.status = status
        super().__init__(secret)


class UnsupportedParamsError(_StatusError):
    """Injected LiteLLM local-validation error; no optional SDK needed."""


class EmbeddingAdapterTests(unittest.TestCase):
    @staticmethod
    def request(adapter, task=EmbeddingTaskType.DOCUMENT, inputs=("alpha", "beta")):
        return EmbeddingBatchRequest(
            identity=adapter.identity(task),
            inputs=inputs,
            timeout_seconds=12.5,
            metadata={"purpose": "offline-test"},
        )

    def test_openai_compatible_explicitly_requests_numeric_vectors(self):
        for transport_kind in ('litellm', 'openai_client'):
            with self.subTest(transport_kind=transport_kind):
                def transport(**kwargs):
                    if kwargs.get('encoding_format') != 'float':
                        raise _StatusError(400, 'encoding_format must be float or base64')
                    return {'data': [
                        {'index': 0, 'embedding': [1.0, 0.0]},
                        {'index': 1, 'embedding': [0.0, 1.0]},
                    ]}

                adapter = OpenAICompatibleEmbeddingAdapter(
                    transport=transport, transport_kind=transport_kind,
                    model='openai/text-embedding-3-small', output_dimension=2,
                    provider='openrouter', endpoint='https://openrouter.ai/api/v1',
                )
                result = adapter.embed(self.request(adapter))
                self.assertEqual(result.vectors, ((1.0, 0.0), (0.0, 1.0)))

    def test_litellm_omits_only_rejected_encoding_once(self):
        calls = []

        def transport(**kwargs):
            calls.append(kwargs)
            if 'encoding_format' in kwargs:
                raise UnsupportedParamsError(400, "Setting {'encoding_format': 'float'} is not supported")
            return {'data': [{'index': 0, 'embedding': [1.0, 0.0]}]}

        adapter = OpenAICompatibleEmbeddingAdapter(
            transport=transport, model='ollama/sample-embedding', output_dimension=2,
            provider='ollama', endpoint='http://localhost:11434',
            api_key='fake-key', request_headers={'X-Test': 'unchanged'},
        )
        request = self.request(adapter, inputs=('original minimal fixture',))
        with mock.patch.dict('sys.modules', {'litellm': SimpleNamespace(UnsupportedParamsError=UnsupportedParamsError)}):
            result = adapter.embed(request)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['encoding_format'], 'float')
        self.assertEqual(calls[1], {k: v for k, v in calls[0].items() if k != 'encoding_format'})
        self.assertEqual(calls[1]['dimensions'], 2)
        self.assertEqual(result.identity, request.identity)
        self.assertEqual(result.request_fingerprint, request.fingerprint)
        self.assertEqual(result.vectors, ((1.0, 0.0),))

    def test_encoding_compatibility_does_not_retry_other_failures(self):
        cases = (
            ('litellm', UnsupportedParamsError(400, "Setting {'dimensions': 2} is not supported")),
            ('openai_client', UnsupportedParamsError(400, "Setting {'encoding_format': 'float'} is not supported")),
            ('litellm', _StatusError(400, "encoding_format rejected by server")),
            ('litellm', _StatusError(401, "encoding_format authentication secret")),
            ('litellm', TimeoutError('encoding_format timeout secret')),
        )
        for kind, error in cases:
            with self.subTest(kind=kind, error=type(error).__name__):
                transport = mock.Mock(side_effect=error)
                adapter = OpenAICompatibleEmbeddingAdapter(
                    transport=transport, transport_kind=kind, model='sample-embedding',
                    output_dimension=2, provider='local', endpoint='http://localhost:11434',
                )
                with (
                    mock.patch.dict('sys.modules', {'litellm': SimpleNamespace(UnsupportedParamsError=UnsupportedParamsError)}),
                    self.assertRaises(EmbeddingBackendError) as raised,
                ):
                    adapter.embed(self.request(adapter))
                self.assertEqual(transport.call_count, 1)
                self.assertNotIn('secret', str(raised.exception))

    def test_encoding_compatibility_is_bounded_and_keeps_vector_validation(self):
        error = UnsupportedParamsError(400, "Setting {'encoding_format': 'float'} is not supported")
        for second in (error, {'data': [{'index': 0, 'embedding': [1.0, 0.0, 0.0]}]}):
            with self.subTest(second=type(second).__name__):
                transport = mock.Mock(side_effect=[error, second])
                adapter = OpenAICompatibleEmbeddingAdapter(
                    transport=transport, model='sample-embedding', output_dimension=2,
                    provider='local', endpoint='http://localhost:11434',
                )
                with (
                    mock.patch.dict('sys.modules', {'litellm': SimpleNamespace(UnsupportedParamsError=UnsupportedParamsError)}),
                    self.assertRaises(EmbeddingBackendError) as raised,
                ):
                    adapter.embed(self.request(adapter, inputs=('original minimal fixture',)))
                self.assertEqual(transport.call_count, 2)
                expected = (EmbeddingErrorCategory.INVALID_REQUEST if isinstance(second, Exception)
                            else EmbeddingErrorCategory.INVALID_RESPONSE)
                self.assertEqual(raised.exception.category, expected)

    @unittest.skipUnless(importlib.util.find_spec('litellm'), 'optional LiteLLM SDK unavailable')
    def test_real_litellm_ollama_validation_reaches_mock_http(self):
        import httpx

        calls = []

        def send(_client, request, *_args, **_kwargs):
            self.assertIn(request.url.host, {'localhost', '127.0.0.1'})
            calls.append(request.url.path.lstrip('/'))
            return httpx.Response(200, request=request, json={
                'embedding': [1.0, 0.0], 'embeddings': [[1.0, 0.0]],
                'model_info': {'sample.embedding_length': 2}, 'prompt_eval_count': 1,
            })

        async def async_send(client, request, *args, **kwargs):
            return send(client, request, *args, **kwargs)

        with (
            mock.patch.dict(os.environ, {'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False'}),
            mock.patch.object(httpx.Client, 'send', send),
            mock.patch.object(httpx.AsyncClient, 'send', async_send),
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
        ):
            import litellm

            adapter = OpenAICompatibleEmbeddingAdapter(
                transport=litellm.embedding, model='ollama/sample-embedding', output_dimension=2,
                provider='ollama', endpoint='http://localhost:11434',
            )
            result = adapter.embed(self.request(adapter, inputs=('original minimal fixture',)))

        self.assertEqual(result.vectors, ((1.0, 0.0),))
        self.assertEqual(sum(path in {'api/embed', 'api/embeddings'} for path in calls), 1, calls)

    def test_gemini_maps_document_task_timeout_usage_and_batch(self):
        response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[1, 0],
                    statistics=SimpleNamespace(token_count=3.0, truncated=False),
                ),
                SimpleNamespace(
                    values=[0, 1],
                    statistics=SimpleNamespace(token_count=4.0, truncated=True),
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

    def test_gemini_non_integral_usage_is_not_reported(self):
        models = _GeminiModels(
            response=SimpleNamespace(
                embeddings=[
                    SimpleNamespace(
                        values=[1, 0],
                        statistics=SimpleNamespace(token_count=1.5, truncated=False),
                    )
                ]
            )
        )
        adapter = GeminiEmbeddingAdapter(
            client=SimpleNamespace(models=models),
            model="gemini-embedding-001",
            output_dimension=2,
            config_factory=lambda **kwargs: kwargs,
        )

        result = adapter.embed(self.request(adapter, inputs=("alpha",)))

        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.total_tokens)

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
        self.assertEqual(calls[0]["api_base"], "https://api.example.test/v1")
        self.assertEqual(calls[0]["api_key"], "fake-key-value")

    def test_openai_official_default_is_pinned_and_matches_explicit_endpoint(self):
        calls = []

        def transport(**kwargs):
            calls.append(kwargs)
            return {"data": [{"index": 0, "embedding": [1, 0]}]}

        default_adapter = OpenAICompatibleEmbeddingAdapter(
            transport=transport,
            model="openai/text-embedding-3-small",
            output_dimension=2,
            provider="openai",
        )
        explicit_adapter = OpenAICompatibleEmbeddingAdapter(
            transport=transport,
            model="openai/text-embedding-3-small",
            output_dimension=2,
            provider="openai",
            endpoint="https://api.openai.com:443/v1/",
        )

        default_adapter.embed(self.request(default_adapter, inputs=("alpha",)))

        self.assertEqual(calls[0]["api_base"], "https://api.openai.com/v1")
        self.assertEqual(
            default_adapter.identity(EmbeddingTaskType.DOCUMENT),
            explicit_adapter.identity(EmbeddingTaskType.DOCUMENT),
        )

    def test_custom_and_implicit_endpoints_are_fail_closed(self):
        def transport(**_kwargs):
            return {"data": []}

        transport.base_url = "https://transport-private.example.test/v1"

        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_BASE": "https://environment-private.example.test/v1"},
        ):
            with self.assertRaises(EmbeddingContractError) as environment_error:
                OpenAICompatibleEmbeddingAdapter(
                    transport=transport,
                    model="embed-model",
                    output_dimension=2,
                    provider="local",
                )
        with self.assertRaises(EmbeddingContractError) as transport_error:
            OpenAICompatibleEmbeddingAdapter(
                transport=transport,
                transport_kind="openai_client",
                model="embed-model",
                output_dimension=2,
                provider="openai",
            )

        for public in (str(environment_error.exception), str(transport_error.exception)):
            self.assertIn("explicit embedding endpoint", public)
            self.assertNotIn("private.example.test", public)

        custom = OpenAICompatibleEmbeddingAdapter(
            transport=transport,
            model="embed-model",
            output_dimension=2,
            provider="local",
            endpoint="https://explicit.example.test/v1",
        )
        self.assertNotIn(
            "explicit.example.test",
            custom.identity(EmbeddingTaskType.DOCUMENT).provider,
        )

    def test_openai_client_rejects_ignored_auth_overrides(self):
        cases = (
            {"api_key": "fake-client-key"},
            {"request_headers": {"X-Test-Authorization": "fake-header-value"}},
        )
        for overrides in cases:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaises(EmbeddingContractError) as raised:
                    OpenAICompatibleEmbeddingAdapter(
                        transport=lambda **_kwargs: {"data": []},
                        transport_kind="openai_client",
                        model="embed-model",
                        output_dimension=2,
                        provider="openai",
                        endpoint="https://client.example.test/v1",
                        **overrides,
                    )
                public = str(raised.exception)
                self.assertIn("configured on the OpenAI client transport", public)
                self.assertNotIn("fake-client-key", public)
                self.assertNotIn("fake-header-value", public)

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
                    endpoint="https://local.example.test/v1",
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
                    endpoint="https://local.example.test/v1",
                )
                with self.assertRaises(EmbeddingBackendError) as raised:
                    adapter.embed(self.request(adapter))
                public = str(raised.exception)
                self.assertEqual(raised.exception.category, category)
                self.assertNotIn("secret", public.lower())
                self.assertNotIn("authorization", public.lower())
                self.assertNotIn("http", public.lower())

    def test_google_api_error_code_is_classified_and_redacted(self):
        cases = (
            (401, "UNAUTHENTICATED", EmbeddingErrorCategory.AUTHENTICATION, False),
            (429, "RESOURCE_EXHAUSTED", EmbeddingErrorCategory.RATE_LIMIT, True),
            (503, "UNAVAILABLE", EmbeddingErrorCategory.UNAVAILABLE, True),
        )
        for code, status, category, retryable in cases:
            with self.subTest(code=code):
                adapter = GeminiEmbeddingAdapter(
                    client=SimpleNamespace(
                        models=_GeminiModels(
                            error=_GoogleAPIError(
                                code,
                                status,
                                "https://provider-private.example.test/?key=hidden",
                            )
                        )
                    ),
                    model="gemini-embedding-001",
                    output_dimension=2,
                    config_factory=lambda **kwargs: kwargs,
                )
                with self.assertRaises(EmbeddingBackendError) as raised:
                    adapter.embed(self.request(adapter))
                self.assertEqual(raised.exception.category, category)
                self.assertEqual(raised.exception.retryable, retryable)
                self.assertNotIn("private.example.test", str(raised.exception))

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
            endpoint=f"https://{provider}.example.test/v1",
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

    def test_endpoint_identity_mismatch_fails_store_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            official = OpenAICompatibleEmbeddingAdapter(
                transport=lambda **_kwargs: {"data": []},
                model="embed-model",
                output_dimension=2,
                provider="openai",
            )
            custom = OpenAICompatibleEmbeddingAdapter(
                transport=lambda **_kwargs: {"data": []},
                model="embed-model",
                output_dimension=2,
                provider="openai",
                endpoint="https://custom.example.test/v1",
            )
            store = JsonRagStore(tmp)
            store.set_embedding_identity(official.identity(EmbeddingTaskType.DOCUMENT))
            store.upsert_history(
                [{"memory_id": "m1", "embedding": [1.0, 0.0], "source_text": "alpha"}]
            )

            results, diagnostics = store.search_history_compatible(
                [1.0, 0.0],
                custom.identity(EmbeddingTaskType.QUERY),
                min_similarity=0.0,
            )

            self.assertEqual(results, [])
            self.assertEqual(
                diagnostics["embedding_compatibility"]["codes"],
                ["provider_mismatch"],
            )
            self.assertEqual(
                diagnostics["embedding_compatibility"]["action"],
                "rebuild_store",
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
            Path(tmp, "metadata.json").write_text(
                json.dumps({"embedding_identity": {"api_key": "do-not-echo"}}),
                encoding="utf-8",
            )
            reloaded = JsonRagStore(tmp)
            _results, invalid = reloaded.search_history_compatible([1, 0], query_identity)
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

    def test_text_only_legacy_stores_can_adopt_identity_safely(self):
        identity = self.adapter().identity(EmbeddingTaskType.DOCUMENT)
        cases = (
            (JsonRagStore, "history.jsonl", {"memory_id": "text-only", "source_text": "alpha"}),
            (
                JsonSourceIndexStore,
                "source_segments.jsonl",
                {"source_id": "text-only", "source_text": "alpha", "embedding": []},
            ),
        )
        for store_type, filename, record in cases:
            with self.subTest(store_type=store_type.__name__), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, filename).write_text(json.dumps(record) + "\n", encoding="utf-8")
                store = store_type(tmp)

                store.set_embedding_identity(identity)

                self.assertTrue(
                    store.embedding_compatibility(
                        self.adapter().identity(EmbeddingTaskType.QUERY)
                    ).compatible
                )

    def test_set_metadata_cannot_bypass_identity_guard(self):
        identity = self.adapter().identity(EmbeddingTaskType.DOCUMENT)
        cases = (
            (
                JsonRagStore,
                lambda store: store.upsert_history([{"memory_id": "m1", "embedding": [1.0, 0.0]}]),
            ),
            (
                JsonSourceIndexStore,
                lambda store: store.upsert_segments([{"source_id": "s1", "embedding": [1.0, 0.0]}]),
            ),
        )
        for store_type, add_vector in cases:
            with self.subTest(store_type=store_type.__name__), tempfile.TemporaryDirectory() as tmp:
                store = store_type(tmp)
                add_vector(store)

                with self.assertRaises(EmbeddingContractError):
                    store.set_metadata(embedding_identity=identity.to_dict())

                report = store.embedding_compatibility(
                    self.adapter().identity(EmbeddingTaskType.QUERY)
                )
                self.assertEqual(report.codes, ("store_identity_missing",))
                self.assertEqual(report.action, "rebuild_store")


if __name__ == "__main__":
    unittest.main()
