"""Behavior and import-boundary coverage for the shared sync executor."""
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import model_profile as mp
import sync_request
from sync_model_backend import SyncBackendError, SyncGenerationResult
from sync_run_service import build_production_backend_adapter


class SyncRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        self.dependencies = sync_request.SyncRequestRuntime(
            timeout_seconds=45,
            custom_providers={'example': object()},
            create_client=mock.Mock(return_value=self.client),
            credential_attempts=mock.Mock(return_value=4),
            rotate_credentials=mock.Mock(return_value=True),
        )

    def plan(self, adapter='gemini'):
        plan = mp.resolve_routing_plan({
            'sync': {'backend': 'litellm' if adapter != 'gemini' else 'gemini',
                     'model': 'openai/frozen' if adapter != 'gemini' else 'gemini-frozen'},
        })
        route = plan.routes['translation']
        if adapter == mp.ADAPTER_OPENAI_COMPATIBLE:
            profile = replace(plan.profiles[route.profile_id], adapter=adapter,
                              base_url='https://example.com/v1')
            plan = replace(plan, profiles={**plan.profiles, route.profile_id: profile})
        return plan, route

    def test_gemini_response_usage_and_timeout_use_actual_adapter(self):
        plan, route = self.plan()
        payload = {
            'candidates': [{'content': {'parts': [{'text': 'hello'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 4, 'candidatesTokenCount': 2},
        }
        self.client.models.generate_content.return_value = payload
        result = sync_request.run_sync_request(
            {'contents': 'source', 'system_instruction': 'translate',
             'safety_settings': [{'category': 'test'}]},
            route, plan, runtime=self.dependencies, timeout_seconds=31,
        )
        call = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(call['model'], 'gemini-frozen')
        self.assertEqual(call['config']['http_options']['timeout'], 31000)
        self.assertEqual(call['config']['system_instruction'], 'translate')
        self.assertEqual(call['config']['safety_settings'], [{'category': 'test'}])
        self.assertEqual(result['response_payload'], payload)
        self.assertEqual(result['response_text'], 'hello')
        self.assertEqual(result['finish_reason'], 'STOP')
        self.assertEqual(result['usage_metadata'], payload['usageMetadata'])
        self.assertEqual(result['provider'], 'gemini')
        self.assertIn('output_diagnostics', result)

    def test_non_gemini_routes_preserve_provider_model_and_metadata(self):
        for adapter in (mp.ADAPTER_LITELLM, mp.ADAPTER_OPENAI_COMPATIBLE):
            with self.subTest(adapter=adapter):
                plan, route = self.plan(adapter)
                result = SyncGenerationResult(
                    provider='actual-provider', model='actual-model', execution_mode='sync',
                    response_payload={'choices': []}, response_text='[]', finish_reason='stop',
                    usage_metadata={'total_tokens': 9}, request_metadata={'request_id': 'r-1'},
                )
                backend = mock.Mock()
                backend.generate.return_value = result
                with mock.patch.object(mp, 'build_sync_backend', return_value=backend) as build:
                    response = sync_request.run_sync_request(
                        {'contents': 'source'}, route, plan, runtime=self.dependencies,
                    )
                self.assertEqual(response['provider'], 'actual-provider')
                self.assertEqual(response['model'], 'actual-model')
                self.assertEqual(response['usage_metadata'], {'total_tokens': 9})
                self.assertEqual(response['request_metadata'], {'request_id': 'r-1'})
                request = backend.generate.call_args.args[0]
                self.assertEqual(request.model, 'openai/frozen')
                self.assertEqual(request.config['timeout'], 45)
                self.assertEqual('custom_providers' in build.call_args.kwargs,
                                 adapter == mp.ADAPTER_LITELLM)
        self.dependencies.create_client.assert_not_called()

    def test_durable_attempt_never_retries_or_rotates_any_adapter(self):
        for adapter in ('gemini', mp.ADAPTER_LITELLM, mp.ADAPTER_OPENAI_COMPATIBLE):
            for category in ('authentication', 'rate_limit', 'timeout', 'service_unavailable'):
                with self.subTest(adapter=adapter, category=category):
                    plan, route = self.plan(adapter)
                    backend = mock.Mock()
                    error = SyncBackendError(category, request_metadata={'provider': 'actual'})
                    backend.generate.side_effect = error
                    durable = build_production_backend_adapter(
                        plan, route, lambda _request, _ids: [], request_runtime=self.dependencies,
                    )
                    with mock.patch.object(mp, 'build_sync_backend', return_value=backend):
                        with self.assertRaises(SyncBackendError) as raised:
                            durable.generate_once({'user_prompt': 'source'}, 29)
                    self.assertIs(raised.exception, error)
                    backend.generate.assert_called_once()
                    request = backend.generate.call_args.args[0]
                    self.assertEqual(request.config['timeout'], 29)
        self.dependencies.credential_attempts.assert_not_called()
        self.dependencies.rotate_credentials.assert_not_called()

    def test_gemini_retry_uses_frozen_route_and_existing_rotation_budget(self):
        plan, route = self.plan()
        backend = mock.Mock()
        backend.generate.side_effect = [SyncBackendError('rate_limit')] * 3 + [
            SyncGenerationResult('gemini', 'gemini-frozen', 'sync', {}, response_text='ok')]
        with mock.patch.object(mp, 'build_sync_backend', return_value=backend), \
                mock.patch.object(sync_request.time, 'sleep') as sleep:
            result = sync_request.run_sync_request({}, route, plan, runtime=self.dependencies)
        self.assertEqual(result['response_text'], 'ok')
        self.assertEqual(backend.generate.call_count, 4)
        self.assertEqual(self.dependencies.rotate_credentials.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2, 2])
        self.assertEqual({c.args[0].model for c in backend.generate.call_args_list}, {'gemini-frozen'})

    def test_explicit_key_index_suppresses_rotation(self):
        plan, route = self.plan()
        backend = mock.Mock()
        backend.generate.side_effect = SyncBackendError('rate_limit')
        with mock.patch.object(mp, 'build_sync_backend', return_value=backend), \
                mock.patch.object(sync_request.time, 'sleep'):
            with self.assertRaises(SyncBackendError):
                sync_request.run_sync_request({}, route, plan, runtime=self.dependencies,
                                              api_key_index=1, retry_attempts=2)
        self.assertEqual(backend.generate.call_count, 2)
        self.dependencies.create_client.assert_called_with(api_key_index=1)
        self.dependencies.rotate_credentials.assert_not_called()
        self.dependencies.credential_attempts.assert_not_called()

    def test_runtime_client_uses_existing_credential_store(self):
        import translator_runtime as runtime
        with mock.patch.object(runtime, 'API_KEYS', ['first', 'second']), \
                mock.patch.object(runtime, 'create_genai_client') as client:
            sync_request.create_runtime_client(api_key_index=1)
            client.assert_called_once_with(api_key='second')
            for invalid in (-1, 2, 'invalid'):
                with self.subTest(index=invalid), self.assertRaises(SystemExit):
                    sync_request.create_runtime_client(api_key_index=invalid)
            self.assertEqual(client.call_count, 1)

    def test_shared_and_service_execution_do_not_import_cli(self):
        code = '''
import importlib.abc
import sys
import unittest
class NoCLI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'gemini_translate_batch', 'gemini_translate'}:
            raise AssertionError('CLI dependency: ' + fullname)
sys.meta_path.insert(0, NoCLI())
import sync_request
sync_request.runtime_dependencies()
from tests.test_sync_request import SyncRequestTests
suite = unittest.TestSuite(SyncRequestTests(name) for name in (
    'test_gemini_response_usage_and_timeout_use_actual_adapter',
    'test_non_gemini_routes_preserve_provider_model_and_metadata',
    'test_durable_attempt_never_retries_or_rotates_any_adapter',
))
result = unittest.TextTestRunner().run(suite)
assert result.wasSuccessful()
assert 'gemini_translate_batch' not in sys.modules
'''
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
