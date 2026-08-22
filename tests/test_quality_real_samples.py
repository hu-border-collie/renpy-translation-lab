"""Offline regression corpus tests for issue #364 real-sample calibration.

The corpus lives in ``tests/fixtures/quality_real_samples/samples.json`` and
contains one privacy-safe positive and one allowlist/legitimate negative sample
per first-version mechanical reason code.  These tests are the in-repo half of
the calibration baseline: they run without any private game text and prove both
that known positives still report and that known negatives stay silent after
allowlist / disposition tuning.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import translation_quality as quality

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "quality_real_samples"
    / "samples.json"
)


def load_corpus() -> tuple[list[dict], dict]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AssertionError("quality_real_samples/samples.json must contain a non-empty cases list")
    if not all(isinstance(case, dict) for case in cases):
        raise AssertionError("every entry in cases must be an object")
    return cases, payload


def findings_for_sample(sample: dict) -> list[dict]:
    """Run one corpus sample through the shared mechanical quality checker."""
    subject = sample.get("subject")
    if not isinstance(subject, dict):
        raise AssertionError("each sample must contain a subject object")
    policy = quality.normalize_policy(sample.get("policy"))
    glossary_map = sample.get("glossary_map") or {}
    if not isinstance(glossary_map, dict):
        raise AssertionError("glossary_map must be an object when present")
    return quality.check_subject(
        subject,
        policy=policy,
        glossary_map=dict(glossary_map),
    )


def reason_codes_for_sample(sample: dict) -> set[str]:
    return {finding["reason_code"] for finding in findings_for_sample(sample)}


class QualityRealSampleCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases, cls.payload = load_corpus()

    def test_corpus_schema_and_reason_codes_are_valid(self) -> None:
        self.assertEqual(self.payload.get("schema_version"), 1)
        reason_codes = [case.get("reason_code") for case in self.cases]
        self.assertTrue(all(reason_codes), "each case needs a reason_code")
        self.assertTrue(
            all(code in quality.ALL_REASON_CODES for code in reason_codes),
            "corpus contains an unknown reason_code",
        )
        for case in self.cases:
            with self.subTest(reason_code=case.get("reason_code")):
                self.assertEqual(
                    quality.REASON_TO_RULE_KEY.get(case["reason_code"]),
                    case.get("rule_id"),
                    "rule_id must match translation_quality.REASON_TO_RULE_KEY",
                )
                for side in ("positive", "negative"):
                    sample = case.get(side)
                    self.assertIsInstance(sample, dict)
                    self.assertIsInstance(sample.get("subject"), dict)

    def test_every_first_version_reason_code_has_a_sample_pair(self) -> None:
        covered = {case["reason_code"] for case in self.cases}

        self.assertEqual(covered, set(quality.ALL_REASON_CODES))

    def test_every_positive_sample_reports_its_reason_code(self) -> None:
        for case in self.cases:
            with self.subTest(reason_code=case["reason_code"]):
                self.assertIn(
                    case["reason_code"],
                    reason_codes_for_sample(case["positive"]),
                )

    def test_every_negative_sample_stays_silent_for_its_reason_code(self) -> None:
        for case in self.cases:
            with self.subTest(reason_code=case["reason_code"]):
                self.assertNotIn(
                    case["reason_code"],
                    reason_codes_for_sample(case["negative"]),
                )


if __name__ == "__main__":
    unittest.main()
