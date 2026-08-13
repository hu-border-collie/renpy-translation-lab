import unittest

from sync_model_backend import sync_error_category


class LiteLLMErrorClassificationTests(unittest.TestCase):
    def test_litellm_typed_exception_names_classify_via_shared_category(self):
        """LiteLLM typed errors are covered by the shared type-name fallback."""

        class RateLimitError(Exception):
            pass

        class ServiceUnavailableError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        self.assertEqual(
            sync_error_category(RateLimitError()),
            "rate_limit",
        )
        self.assertEqual(
            sync_error_category(ServiceUnavailableError()),
            "service_unavailable",
        )
        self.assertEqual(
            sync_error_category(AuthenticationError()),
            "authentication",
        )


if __name__ == "__main__":
    unittest.main()
