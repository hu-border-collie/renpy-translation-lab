"""Shared preflight cost / coverage / quality display (#488 S2)."""
from __future__ import annotations

import unittest

import preflight_display


def known_cost(**overrides):
    payload = {
        "status": "known",
        "model": "gemini-3.5-flash",
        "strategy": "sync",
        "currency": "USD",
        "estimated_cost_min": 0.0123,
        "estimated_cost_max": 0.0456,
        "pricing_source": "defaults",
        "excluded": ["provider 排队与重试", "未计入的实际输出"],
    }
    payload.update(overrides)
    return payload


def ready_coverage(**overrides):
    payload = {
        "status": "ready",
        "completion": "confirmed",
        "classification_counts": {"translatable": 3, "unknown": 1},
        "unknown_count": 1,
        "parse_error_count": 0,
    }
    payload.update(overrides)
    return payload


def available_quality(**overrides):
    payload = {
        "status": "available",
        "finding_count": 2,
        "severity_counts": {"high": 1, "medium": 1, "low": 0, "info": 0},
    }
    payload.update(overrides)
    return payload


class PreflightDisplayTests(unittest.TestCase):
    def test_known_cost_shows_model_range_and_source(self) -> None:
        line = preflight_display.format_preflight_cost_line(known_cost())
        self.assertIn("gemini-3.5-flash", line)
        self.assertIn("sync", line)
        self.assertIn("0.0123", line)
        self.assertIn("0.0456", line)
        self.assertIn("USD", line)
        self.assertIn("defaults", line)
        self.assertIn("未计入的实际输出", line)
        self.assertNotIn("无法估算", line)
        self.assertNotIn("免费", line)

    def test_unknown_cost_explains_reason_without_zero_or_free(self) -> None:
        line = preflight_display.format_preflight_cost_line(
            {"status": "unknown", "reason": "pricing_unavailable"}
        )
        self.assertIn("无法估算", line)
        self.assertIn("没有可用价格表", line)
        self.assertNotIn("免费", line)
        self.assertNotIn("0.0", line)
        self.assertNotIn("¥0", line)
        self.assertNotIn("$0", line)

    def test_missing_cost_is_unknown_not_zero(self) -> None:
        line = preflight_display.format_preflight_cost_line(None)
        self.assertIn("无法估算", line)
        self.assertIn("摘要缺失", line)
        self.assertNotIn("免费", line)
        self.assertNotIn("0.0", line)

    def test_known_cost_without_amounts_does_not_invent_zero(self) -> None:
        line = preflight_display.format_preflight_cost_line(
            known_cost(estimated_cost_min=None, estimated_cost_max=None)
        )
        self.assertIn("无法估算", line)
        self.assertNotIn("免费", line)

    def test_coverage_shows_status_counts_and_unknowns(self) -> None:
        line = preflight_display.format_preflight_coverage_line(ready_coverage())
        self.assertIn("ready", line)
        self.assertIn("confirmed", line)
        self.assertIn("translatable=3", line)
        self.assertIn("unknown 1", line)
        self.assertIn("parse_error 0", line)
        self.assertNotIn("覆盖完成", line)

    def test_unknown_coverage_does_not_claim_completion(self) -> None:
        line = preflight_display.format_preflight_coverage_line(
            {"status": "unknown", "completion": "unknown"}
        )
        self.assertIn("扫描证据缺失", line)
        self.assertIn("不能报告为覆盖完成", line)

    def test_missing_coverage_does_not_claim_completion(self) -> None:
        line = preflight_display.format_preflight_coverage_line({})
        self.assertIn("不能报告为覆盖完成", line)

    def test_available_quality_shows_finding_counts_not_pass(self) -> None:
        line = preflight_display.format_preflight_quality_line(available_quality())
        self.assertIn("finding 2", line)
        self.assertIn("high=1", line)
        self.assertIn("medium=1", line)
        self.assertIn("check/apply", line)
        self.assertNotIn("质量通过", line)

    def test_stale_quality_is_explicit(self) -> None:
        line = preflight_display.format_preflight_quality_line(
            {"status": "stale", "reason": "plan_mismatch"}
        )
        self.assertIn("过期", line)
        self.assertIn("不属于当前翻译计划", line)
        self.assertNotIn("finding", line)
        self.assertNotIn("质量通过", line)

    def test_not_available_quality_is_explicit(self) -> None:
        line = preflight_display.format_preflight_quality_line(
            {
                "status": "not_available",
                "reason": "quality_source_not_supported_for_strategy",
            }
        )
        self.assertIn("没有可验证的已有质量报告", line)
        self.assertNotIn("finding", line)
        self.assertNotIn("质量通过", line)

    def test_unknown_quality_is_explicit(self) -> None:
        line = preflight_display.format_preflight_quality_line(
            {"status": "unknown", "reason": "report_unreadable"}
        )
        self.assertIn("无法读取或解析", line)
        self.assertNotIn("质量通过", line)

    def test_missing_payload_keys_do_not_crash_or_claim_success(self) -> None:
        fields = preflight_display.format_preflight_summary_fields({})
        self.assertIn("无法估算", fields["cost"])
        self.assertIn("不能报告为覆盖完成", fields["coverage"])
        self.assertIn("不能当作当前质量结论", fields["quality"])
        joined = "\n".join(preflight_display.format_preflight_summary_lines(None))
        self.assertNotIn("质量通过", joined)
        self.assertNotIn("免费", joined)
