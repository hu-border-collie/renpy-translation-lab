"""Production adapter regressions for short-lived durable request bindings."""

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gemini_translate_batch as batch
import model_profile
import sync_request
import translator_runtime as runtime
from litellm_provider_config import CustomLiteLLMProvider, ProviderApiKeyStore
from litellm_sync_backend import LiteLLMBackendError
from openai_compatible_sync_backend import HTTPResponse
from sync_run_contracts import build_run_id
from sync_run_service import build_production_sync_run_service
from sync_run_store import SyncRunStore
from tests.test_sync_run_service import plan_build


class DurableRequestIsolationTests(unittest.TestCase):
    def _context(self, plan):
        return SimpleNamespace(
            plan_build=plan_build(), routing_plan=plan,
            route=plan.routes['translation'],
            item_resolver=lambda _request, ids: [
                {'id': item_id, 'text': 'hello'} for item_id in ids
            ],
            validate_translation=lambda _item, _text: (True, 'OK'),
            context_resolver=lambda _request: {},
            validate_reused_translation=lambda _item_id, _payload: True,
            durable_targets_payload=lambda **_kwargs: {},
        )

    def _service(self, root, *, model=None, plan=None, request_runtime=None):
        if plan is None:
            plan = model_profile.resolve_routing_plan({
                'sync': {'backend': 'gemini', 'model': model},
            })
        return build_production_sync_run_service(
            root, self._context(plan), request_runtime=request_runtime,
        )

    def test_default_and_explicit_service_keep_gemini_key_after_other_project_load(self):
        calls = []

        class Client:
            def __init__(self, api_key):
                self.models = SimpleNamespace(generate_content=self.generate_content)
                self.api_key = api_key

            def generate_content(self, **kwargs):
                calls.append((self.api_key, kwargs['model'],
                              kwargs['config']['http_options']['timeout']))
                return {
                    'candidates': [{'content': {'parts': [{'text': 'ok'}]},
                                    'finishReason': 'STOP'}],
                    'usageMetadata': {'totalTokenCount': 2},
                }

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runtime, 'get_genai_module',
                                  return_value=SimpleNamespace(Client=Client)), \
                mock.patch.object(batch, 'ensure_batch_sdk'), \
                mock.patch.object(batch, 'genai', SimpleNamespace(Client=Client)):
            with mock.patch.object(runtime, 'API_KEYS', ['a-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 41):
                default_a = self._service(Path(tmp) / 'a', model='gemini-a')
                explicit_a = self._service(
                    Path(tmp) / 'a-explicit', model='gemini-a-explicit',
                    request_runtime=sync_request.runtime_dependencies(
                        timeout_seconds=41, create_client=batch.create_batch_client,
                    ),
                )
            with mock.patch.object(runtime, 'API_KEYS', ['b-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 62):
                default_b = self._service(Path(tmp) / 'b', model='gemini-b')
                for service, timeout in (
                    (default_a, 41), (explicit_a, 41),
                    (default_b, 62), (default_a, 41), (explicit_a, 41),
                ):
                    service.backend_factory(None).generate_once(
                        {'user_prompt': 'hello'}, timeout,
                    )
        self.assertEqual(calls, [
            ('a-secret', 'gemini-a', 41000),
            ('a-secret', 'gemini-a-explicit', 41000),
            ('b-secret', 'gemini-b', 62000),
            ('a-secret', 'gemini-a', 41000),
            ('a-secret', 'gemini-a-explicit', 41000),
        ])

    def test_start_freezes_bound_timeout_in_policy_and_actual_request(self):
        calls = []

        class Client:
            def __init__(self, api_key):
                self.api_key = api_key
                self.models = SimpleNamespace(generate_content=self.generate_content)

            def generate_content(self, **kwargs):
                calls.append((self.api_key, kwargs['model'],
                              kwargs['config']['http_options']['timeout']))
                return {
                    'candidates': [{'content': {'parts': [{
                        'text': '{"translations":[{"id":"item-1","translation":"一"}]}'
                    }]}, 'finishReason': 'STOP'}],
                    'usageMetadata': {'totalTokenCount': 2},
                }

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runtime, 'get_genai_module',
                                  return_value=SimpleNamespace(Client=Client)):
            with mock.patch.object(runtime, 'API_KEYS', ['a-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 41):
                service_a = self._service(Path(tmp) / 'a', model='gemini-a')
            with mock.patch.object(runtime, 'API_KEYS', ['b-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 62):
                service_b = self._service(Path(tmp) / 'b', model='gemini-b')
                snapshots = [service.start(plan_build()) for service in
                             (service_a, service_b, service_a)]
                service_a.start(plan_build(), policy={'attempt_timeout_seconds': 23})
                derived = service_a.derive(snapshots[0]['run_id'], plan_build())
                stored_policy = json.loads(SyncRunStore(
                    Path(tmp) / 'a', snapshots[0]['run_id'],
                ).get_run()['policy_json'])
                derived_policy = json.loads(SyncRunStore(
                    Path(tmp) / 'a', derived['run_id'],
                ).get_run()['policy_json'])
        self.assertEqual(calls, [
            ('a-secret', 'gemini-a', 41000),
            ('b-secret', 'gemini-b', 62000),
            ('a-secret', 'gemini-a', 41000),
            ('a-secret', 'gemini-a', 23000),
        ])
        self.assertEqual(stored_policy['attempt_timeout_seconds'], 41)
        self.assertEqual(derived_policy['attempt_timeout_seconds'], 41)

    def test_litellm_services_keep_custom_endpoint_and_use_one_provider_call(self):
        calls = []
        provider_a = CustomLiteLLMProvider(
            'customa', 'A', 'https://a.test/v1', '', requires_key=False,
        )
        provider_b = replace(provider_a, label='B', base_url='https://b.test/v1')

        def completion(**kwargs):
            calls.append((kwargs['api_base'], kwargs['model'], kwargs['timeout']))
            return {'choices': [{'message': {'content': 'ok'},
                                 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 3}}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch('litellm.completion', side_effect=completion):
            with mock.patch.object(runtime, 'CUSTOM_LITELLM_PROVIDERS',
                                   {'customa': provider_a}):
                plan_a = model_profile.resolve_routing_plan(
                    {'sync': {'backend': 'litellm', 'model': 'customa/a'}},
                    custom_providers={'customa': provider_a},
                )
                service_a = self._service(Path(tmp) / 'a', plan=plan_a)
            with mock.patch.object(runtime, 'CUSTOM_LITELLM_PROVIDERS',
                                   {'customa': provider_b}):
                plan_b = model_profile.resolve_routing_plan(
                    {'sync': {'backend': 'litellm', 'model': 'customa/b'}},
                    custom_providers={'customa': provider_b},
                )
                service_b = self._service(Path(tmp) / 'b', plan=plan_b)
                for service, timeout in ((service_a, 41), (service_b, 62),
                                         (service_a, 41)):
                    service.backend_factory(None).generate_once(
                        {'user_prompt': 'hello'}, timeout,
                    )
        self.assertEqual(calls, [
            ('https://a.test/v1', 'openai/a', 41),
            ('https://b.test/v1', 'openai/b', 62),
            ('https://a.test/v1', 'openai/a', 41),
        ])

    def test_resume_rebinds_current_key_but_uses_stored_timeout(self):
        calls = []

        class Client:
            def __init__(self, api_key):
                self.api_key = api_key
                self.models = SimpleNamespace(generate_content=self.generate_content)

            def generate_content(self, **kwargs):
                calls.append((self.api_key,
                              kwargs['config']['http_options']['timeout']))
                return {
                    'candidates': [{'content': {'parts': [{
                        'text': '{"translations":[{"id":"item-1","translation":"一"}]}'
                    }]}, 'finishReason': 'STOP'}],
                }

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runtime, 'get_genai_module',
                                  return_value=SimpleNamespace(Client=Client)):
            root = Path(tmp) / 'runs'
            with mock.patch.object(runtime, 'API_KEYS', ['old-secret']), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 41):
                original = self._service(root, model='gemini-resume')
                payload = plan_build()
                store, _created = SyncRunStore.bootstrap(
                    root, build_run_id(), plan=payload['plan'],
                    requests=payload['requests'],
                    executor_policy=original.default_policy.to_dict(),
                )
                original._ensure_run_artifacts(store)
            with mock.patch.object(runtime, 'API_KEYS', ['renewed-secret']), \
                    mock.patch.object(runtime, 'SYNC_TIMEOUT_SECONDS', 62):
                resumed_service = self._service(root, model='gemini-resume')
                resumed_service.resume(store.run_id)
        self.assertEqual(calls, [('renewed-secret', 41000)])

    def test_durable_cli_assembly_binds_batch_factory_key_before_project_switch(self):
        calls = []

        class Client:
            def __init__(self, api_key):
                self.models = SimpleNamespace(generate_content=self.generate_content)
                self.api_key = api_key

            def generate_content(self, **kwargs):
                calls.append((self.api_key, kwargs['model']))
                return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

        plan = model_profile.resolve_routing_plan({
            'sync': {'backend': 'gemini', 'model': 'gemini-cli-a'},
        })
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(batch.legacy, 'prepare_sync_translation_execution_context',
                                  return_value=self._context(plan)), \
                mock.patch.object(batch, 'load_batch_settings'), \
                mock.patch.object(batch, '_read_translator_config_object',
                                  return_value={}), \
                mock.patch.object(batch.batch_cost_estimate, 'load_pricing_config',
                                  return_value={}), \
                mock.patch.object(batch, '_durable_sync_root_dir', return_value=tmp), \
                mock.patch.object(batch, 'ensure_batch_sdk'), \
                mock.patch.object(batch, 'genai', SimpleNamespace(Client=Client)):
            with mock.patch.object(runtime, 'API_KEYS', ['cli-a-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0), \
                    mock.patch.object(batch, 'SYNC_TIMEOUT_SECONDS', 41):
                service, _context = batch._durable_sync_production_service(
                    require_provider=True,
                )
            with mock.patch.object(runtime, 'API_KEYS', ['cli-b-secret']), \
                    mock.patch.object(runtime, 'CURRENT_KEY_INDEX', 0):
                service.backend_factory(None).generate_once(
                    {'user_prompt': 'hello'}, 41,
                )
        self.assertEqual(calls, [('cli-a-secret', 'gemini-cli-a')])

    def test_litellm_durable_rate_limit_never_rotates_inside_attempt(self):
        provider = CustomLiteLLMProvider(
            'customa', 'A', 'https://a.test/v1', '', requires_key=True,
        )
        plan = model_profile.resolve_routing_plan(
            {'sync': {'backend': 'litellm', 'model': 'customa/a'}},
            custom_providers={'customa': provider},
        )
        store = ProviderApiKeyStore(keys=('first-secret', 'second-secret'), active_index=0)
        calls = []

        class RateLimitError(RuntimeError):
            category = 'rate_limit'

        def completion(**kwargs):
            calls.append(kwargs['api_key'])
            raise RateLimitError('limited')

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch('litellm.completion', side_effect=completion), \
                mock.patch('litellm_provider_config.load_provider_api_key',
                           return_value='first-secret'), \
                mock.patch('litellm_provider_config.load_provider_key_store',
                           return_value=store), \
                mock.patch.object(runtime, 'CUSTOM_LITELLM_PROVIDERS',
                                  {'customa': provider}):
            service = self._service(Path(tmp), plan=plan)
            with self.assertRaises(LiteLLMBackendError) as raised:
                service.backend_factory(None).generate_once(
                    {'user_prompt': 'hello'}, 41,
                )
        self.assertEqual(raised.exception.category, 'rate_limit')
        self.assertEqual(calls, ['first-secret'])
        self.assertNotIn('first-secret', repr(raised.exception.request_metadata))

    def test_direct_adapter_keeps_frozen_model_endpoint_and_timeout(self):
        calls = []

        def transport(request, timeout):
            calls.append((request.full_url, json.loads(request.data)['model'], timeout))
            return HTTPResponse(
                status=200, headers={'Content-Type': 'application/json'},
                body=json.dumps({
                    'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 2},
                }).encode(),
            )

        def direct_plan(model, url):
            plan = model_profile.resolve_routing_plan({
                'sync': {'backend': 'litellm', 'model': model},
            })
            route = plan.routes['translation']
            profile = replace(plan.profiles[route.profile_id],
                              adapter=model_profile.ADAPTER_OPENAI_COMPATIBLE,
                              base_url=url,
                              credential_ref=model_profile.CredentialRef('none', ''))
            return replace(plan, profiles={**plan.profiles, route.profile_id: profile})

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch('openai_compatible_sync_backend._default_transport',
                           side_effect=transport):
            service_a = self._service(Path(tmp) / 'a',
                                      plan=direct_plan('openai/a', 'https://a.test/v1'))
            service_b = self._service(Path(tmp) / 'b',
                                      plan=direct_plan('openai/b', 'https://b.test/v1'))
            for service, timeout in ((service_a, 41), (service_b, 62),
                                     (service_a, 41)):
                service.backend_factory(None).generate_once(
                    {'user_prompt': 'hello'}, timeout,
                )
        self.assertEqual(calls, [
            ('https://a.test/v1/chat/completions', 'openai/a', 41),
            ('https://b.test/v1/chat/completions', 'openai/b', 62),
            ('https://a.test/v1/chat/completions', 'openai/a', 41),
        ])

    def test_runtime_binding_copies_nested_config_and_hides_credentials(self):
        nested = {'customa': {'options': {'values': [1]}}}
        with mock.patch.object(runtime, 'API_KEYS', ['private-secret']), \
                mock.patch.object(runtime, 'CUSTOM_LITELLM_PROVIDERS', nested):
            binding = sync_request.runtime_dependencies()
        nested['customa']['options']['values'].append(2)
        self.assertEqual(binding.custom_providers['customa']['options']['values'], [1])
        self.assertNotIn('private-secret', repr(binding))


if __name__ == '__main__':
    unittest.main()
