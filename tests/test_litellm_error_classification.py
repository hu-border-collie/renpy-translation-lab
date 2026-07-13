import sys
import types
import unittest
from unittest import mock

from litellm_sync_backend import _error_category


class LiteLLMErrorClassificationTests(unittest.TestCase):
    def test_prefers_litellm_typed_exceptions(self):
        class RateLimitError(Exception):
            pass

        class ServiceUnavailableError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        fake_litellm = types.SimpleNamespace(
            RateLimitError=RateLimitError,
            ServiceUnavailableError=ServiceUnavailableError,
            AuthenticationError=AuthenticationError,
        )
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            self.assertEqual(_error_category(RateLimitError()), "rate_limit")
            self.assertEqual(
                _error_category(ServiceUnavailableError()), "service_unavailable"
            )
            self.assertEqual(_error_category(AuthenticationError()), "authentication")


if __name__ == "__main__":
    unittest.main()
