"""Exception diagnostics retain useful facts without provider/private text."""
import json
import unittest

from context_engine.embedding_backend import EmbeddingBackendError, EmbeddingErrorCategory
from context_engine.embedding_diagnostics import capture_exception_diagnostics
from context_engine.embedding_runtime import public_error_diagnostics


class EmbeddingDiagnosticsTests(unittest.TestCase):
    def test_cause_and_stack_coordinates_survive_without_messages_paths_or_locals(self):
        namespace = {'private_text': 'private-dialogue-and-key'}
        script = ('def request():\n'
                  '    try:\n'
                  '        raise OSError(54, private_text)\n'
                  '    except OSError as cause:\n'
                  '        raise RuntimeError(private_text) from cause\n'
                  'request()\n')
        try:
            exec(compile(script, 'C:/private-user/game-secret/transport.py', 'exec'), namespace)
        except RuntimeError as exc:
            facts = capture_exception_diagnostics(exc)
        self.assertEqual(facts['exception_type'], 'RuntimeError')
        self.assertEqual(facts['exception_chain'], 'RuntimeError -> OSError')
        self.assertIn('transport.py:request:', facts['traceback_location'])
        serialized = json.dumps(facts)
        for private in ('private-dialogue', 'private-user', 'game-secret', 'private_text'):
            self.assertNotIn(private, serialized)

    def test_public_boundary_rejects_arbitrary_payload_fields_and_invalid_facts(self):
        error = EmbeddingBackendError(EmbeddingErrorCategory.PROVIDER_ERROR, diagnostics={
            'exception_type': 'https://private.test/?key=secret',
            'exception_chain': 'Error\nAuthorization: secret',
            'http_status': True, 'provider_status': 'secret',
            'traceback_location': 'C:/private/game.py:send:1',
            'response_body': 'secret', 'request_headers': {'Authorization': 'secret'},
        })
        facts = public_error_diagnostics(error)
        self.assertEqual(dict(error.diagnostics), {})
        self.assertNotIn('secret', json.dumps(facts))
        self.assertNotIn('private', json.dumps(facts))

    def test_diagnostic_capture_handles_broken_status_properties_and_cyclic_causes(self):
        class BrokenError(Exception):
            @property
            def status_code(self):
                raise ValueError('private-status-accessor')

        error = BrokenError('private-provider-message')
        error.__cause__ = error
        facts = capture_exception_diagnostics(error)
        self.assertEqual(facts['exception_chain'], 'BrokenError')
        self.assertNotIn('http_status', facts)
        self.assertNotIn('private', json.dumps(facts))

    def test_local_disk_error_is_not_reported_as_provider_failure(self):
        facts = public_error_diagnostics(PermissionError(13, 'private-file-name'))
        self.assertEqual(facts['error_category'], 'local_error')
        self.assertEqual(facts['exception_type'], 'PermissionError')
        self.assertEqual(facts['os_errno'], 13)
        self.assertNotIn('private-file-name', json.dumps(facts))

    def test_unwrapped_timeout_and_connection_error_are_not_mislabelled_as_disk_errors(self):
        for error, category in ((TimeoutError('private'), 'timeout'),
                                (ConnectionResetError('private'), 'unavailable')):
            with self.subTest(category=category):
                facts = public_error_diagnostics(error)
                self.assertEqual(facts['error_category'], category)
                self.assertTrue(facts['retryable'])
                self.assertNotIn('private', json.dumps(facts))
