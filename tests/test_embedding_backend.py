# -*- coding: utf-8 -*-
"""Pure unit tests for the provider-neutral embedding core (issue #341)."""

import json
import math
import unittest

import embedding_backend as embedding


def identity(task=embedding.EmbeddingTaskType.DOCUMENT, **overrides):
    values = dict(
        backend='google_genai',
        provider='gemini',
        model='gemini-embedding-001',
        task_type=task,
        output_dimension=768,
    )
    values.update(overrides)
    return embedding.EmbeddingIdentity(**values)


class IdentityTests(unittest.TestCase):
    def test_identity_serialization_and_fingerprint_are_deterministic(self):
        first = identity()
        second = embedding.EmbeddingIdentity.from_dict(
            json.loads(json.dumps(first.to_dict(), ensure_ascii=False))
        )
        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertRegex(first.fingerprint, r'^sha256:[0-9a-f]{64}$')
        self.assertEqual(
            embedding.canonical_json(first.to_dict()),
            embedding.canonical_json(second.to_dict()),
        )

    def test_fingerprint_covers_every_vector_space_field(self):
        baseline = identity()
        variants = (
            identity(backend='openai_compatible'),
            identity(provider='custom-openai'),
            identity(model='text-embedding-3-small'),
            identity(task=embedding.EmbeddingTaskType.QUERY),
            identity(output_dimension=1536),
        )
        for variant in variants:
            self.assertNotEqual(baseline.fingerprint, variant.fingerprint)

    def test_from_dict_rejects_tampered_fingerprint(self):
        payload = identity().to_dict()
        payload['model'] = 'different-model'
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'fingerprint'):
            embedding.EmbeddingIdentity.from_dict(payload)

    def test_rejects_empty_model_and_invalid_dimension(self):
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'model'):
            identity(model='  ')
        for value in (0, -1, True, 1.5, '768'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(embedding.EmbeddingContractError, 'output_dimension'):
                    identity(output_dimension=value)


class RequestResultTests(unittest.TestCase):
    def request(self, **overrides):
        values = dict(
            identity=identity(),
            inputs=('first document', 'second document'),
            timeout_seconds=15,
            metadata={'trace_id': 'local-1', 'attempt': 1},
        )
        values.update(overrides)
        return embedding.EmbeddingBatchRequest(**values)

    def result(self, request, **overrides):
        values = dict(
            identity=request.identity,
            request_fingerprint=request.fingerprint,
            vectors=tuple((0.0,) * 768 for _ in request.inputs),
            usage=embedding.EmbeddingUsage(input_tokens=4, total_tokens=4),
            metadata={'provider_request_id': 'request-1'},
        )
        values.update(overrides)
        return embedding.EmbeddingBatchResult(**values)

    def test_valid_batch_round_trip_and_usage_metadata(self):
        request = self.request()
        result = self.result(request)
        self.assertIs(embedding.validate_embedding_result(request, result), result)
        self.assertEqual(result.usage.to_dict()['input_tokens'], 4)
        self.assertEqual(len(result.to_dict()['vectors']), 2)

    def test_request_fingerprint_is_stable_across_metadata_key_order(self):
        first = self.request(metadata={'b': 2, 'a': {'z': 1, 'x': 0}})
        second = self.request(metadata={'a': {'x': 0, 'z': 1}, 'b': 2})
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_rejects_empty_input_and_invalid_timeout(self):
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'inputs must not be empty'):
            self.request(inputs=())
        with self.assertRaisesRegex(embedding.EmbeddingContractError, r'inputs\[0\]'):
            self.request(inputs=(' ',))
        for value in (0, -1, math.inf, math.nan, True, '10'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(embedding.EmbeddingContractError, 'timeout_seconds'):
                    self.request(timeout_seconds=value)

    def test_rejects_credential_shaped_metadata_recursively(self):
        unsafe_values = (
            {'api_key': 'secret'},
            {'headers': {'Authorization': 'Bearer secret'}},
            {'nested': [{'session_token': 'secret'}]},
            {'credential_ref': {'name': 'ENV_NAME'}},
        )
        for metadata in unsafe_values:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(embedding.EmbeddingContractError, 'credential-shaped'):
                    self.request(metadata=metadata)

    def test_allows_token_counts_but_rejects_non_finite_metadata(self):
        request = self.request(metadata={'usage_tokens': 10, 'prompt_tokens': 5})
        self.assertEqual(request.metadata['usage_tokens'], 10)
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'NaN or Infinity'):
            self.request(metadata={'score': math.nan})

    def test_metadata_must_be_an_object_at_each_contract_boundary(self):
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'metadata must be an object'):
            self.request(metadata=['not', 'an', 'object'])
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'usage.metadata must be an object'):
            embedding.EmbeddingUsage(metadata='not-an-object')
        request = self.request(inputs=('one',))
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'metadata must be an object'):
            self.result(request, metadata=('not', 'an', 'object'))

    def test_rejects_vector_dimension_and_non_finite_values(self):
        request = self.request(inputs=('one',))
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'dimension mismatch'):
            self.result(request, vectors=((0.0, 1.0),))
        bad_vector = [0.0] * 768
        bad_vector[-1] = math.inf
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'must be finite'):
            self.result(request, vectors=(bad_vector,))

    def test_rejects_result_count_request_binding_and_identity_mismatch(self):
        request = self.request()
        one_vector = ((0.0,) * 768,)
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'count mismatch'):
            embedding.validate_embedding_result(
                request,
                self.result(request, vectors=one_vector),
            )
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'request_fingerprint'):
            embedding.validate_embedding_result(
                request,
                self.result(request, request_fingerprint='sha256:wrong'),
            )
        other_identity = identity(provider='other-provider')
        with self.assertRaisesRegex(embedding.EmbeddingContractError, 'identity'):
            embedding.validate_embedding_result(
                request,
                self.result(request, identity=other_identity),
            )


class CompatibilityTests(unittest.TestCase):
    def test_document_store_and_matching_query_are_compatible(self):
        report = embedding.check_store_query_compatibility(
            identity(embedding.EmbeddingTaskType.DOCUMENT),
            identity(embedding.EmbeddingTaskType.QUERY),
        )
        self.assertTrue(report.compatible)
        self.assertEqual(report.codes, ('compatible',))
        self.assertEqual(report.action, 'none')

    def test_every_identity_mismatch_returns_rebuild_diagnostic(self):
        store = identity(embedding.EmbeddingTaskType.QUERY, backend='stored-backend')
        query = identity(
            embedding.EmbeddingTaskType.DOCUMENT,
            backend='query-backend',
            provider='other-provider',
            model='other-model',
            output_dimension=1536,
        )
        report = embedding.check_store_query_compatibility(store, query)
        self.assertFalse(report.compatible)
        self.assertEqual(report.action, 'rebuild_store')
        self.assertEqual(
            report.codes,
            (
                'store_task_not_document',
                'query_task_not_query',
                'backend_mismatch',
                'provider_mismatch',
                'model_mismatch',
                'dimension_mismatch',
            ),
        )
        self.assertIn('do not compare', report.message)
        self.assertIn('Rebuild the store', report.message)
        self.assertEqual(
            [item['field'] for item in report.to_dict()['mismatches']],
            ['task_type', 'task_type', 'backend', 'provider', 'model', 'output_dimension'],
        )

    def test_model_mismatch_alone_is_explicit(self):
        report = embedding.check_store_query_compatibility(
            identity(),
            identity(embedding.EmbeddingTaskType.QUERY, model='new-model'),
        )
        self.assertEqual(report.codes, ('model_mismatch',))


class ErrorAndProtocolTests(unittest.TestCase):
    def test_backend_error_exposes_stable_category(self):
        error = embedding.EmbeddingBackendError(
            embedding.EmbeddingErrorCategory.RATE_LIMIT,
            'quota exceeded',
            retryable=True,
            provider_code='429',
        )
        self.assertEqual(error.category.value, 'rate_limit')
        self.assertTrue(error.retryable)
        self.assertEqual(error.provider_code, '429')

    def test_runtime_protocol_accepts_minimal_backend(self):
        class FakeBackend:
            def embed(self, request):
                return request

        self.assertIsInstance(FakeBackend(), embedding.EmbeddingBackend)


if __name__ == '__main__':
    unittest.main()
