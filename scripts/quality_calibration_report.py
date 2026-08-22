"""Build an offline Markdown calibration report from ``quality_findings.jsonl``.

This is an issue #364 development/calibration helper (CLI-only, scripts
scope).  It turns a findings report produced by ``check``, sync preview,
revision preview, or final review into a reproducible baseline showing, per
reason code: finding count, file distribution, and a bounded set of
file/line/source/translation samples for manual misreport / miss annotation.

Usage::

    python scripts/quality_calibration_report.py path/to/quality_findings.jsonl
    python scripts/quality_calibration_report.py path/to/quality_findings.jsonl \\
        --sample-limit 10 -o docs/plans/quality_calibration_baseline.md

Rows are loaded with ``translation_quality.load_findings(strict=True)`` so a
malformed report fails loudly instead of silently skewing the baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLE_LIMIT = 5
UNLOCATED_FILE = "(unlocated)"


def _quality_module() -> Any:
    """Import the shared quality module after repository path bootstrap."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import translation_quality

    return translation_quality


class CalibrationReportError(RuntimeError):
    """Raised when a calibration report input cannot be loaded safely."""


def _file_label(finding: Mapping[str, Any]) -> str:
    value = str(finding.get("file") or "").strip()
    return value or UNLOCATED_FILE


def _md_cell(value: Any) -> str:
    """Escape one Markdown table cell while preserving readable samples."""
    text = str(value or "")
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def load_report_findings(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load and contract-validate a findings report for calibration use."""

    raw_path = os.fspath(path)
    if not raw_path or not os.path.isfile(raw_path):
        raise CalibrationReportError(
            f"quality findings file not found: {raw_path or '<empty path>'}"
        )
    try:
        return _quality_module().load_findings(raw_path, strict=True)
    except (OSError, ValueError) as exc:
        raise CalibrationReportError(
            f"could not load quality findings report: {raw_path}: {exc}"
        ) from exc


def reason_counters(
    findings: list[dict[str, Any]],
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    """Return global reason counts and per-reason file counts."""

    reason_counts: Counter[str] = Counter()
    file_counts: dict[str, Counter[str]] = {}
    for finding in findings:
        reason_code = str(finding.get("reason_code") or "quality.unknown").strip()
        reason_counts[reason_code] += 1
        file_counts.setdefault(reason_code, Counter())[_file_label(finding)] += 1
    return reason_counts, file_counts


def ordered_reason_codes(reason_counts: Counter[str]) -> list[str]:
    """Order codes by findings descending, then by code for stable output."""

    return sorted(reason_counts, key=lambda code: (-reason_counts[code], code))


def samples_for_reason(
    findings: list[dict[str, Any]],
    reason_code: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return deterministic, bounded samples for one reason code."""

    matching = [
        finding
        for finding in findings
        if str(finding.get("reason_code") or "") == reason_code
    ]
    matching.sort(
        key=lambda finding: (
            _file_label(finding),
            int(finding.get("line") or 0),
            str(finding.get("item_id") or ""),
        )
    )
    return matching[: max(limit, 0)]


def render_report(
    findings: list[dict[str, Any]],
    *,
    source_path: str,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    generated_at: str | None = None,
) -> str:
    """Render the calibration baseline as Markdown."""

    quality = _quality_module()
    reason_counts, file_counts = reason_counters(findings)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = [
        "# Quality calibration report",
        "",
        f"- Source: `{_md_cell(source_path)}`",
        f"- Generated: {generated_at}",
        f"- Findings: {len(findings)}",
        f"- Reason codes: {len(reason_counts)}",
        f"- Files: {len({_file_label(finding) for finding in findings})}",
        f"- Samples per reason code: {max(sample_limit, 0)}",
        "",
        "## Summary",
        "",
        "| rank | reason_code | rule_id | findings | files |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, reason_code in enumerate(ordered_reason_codes(reason_counts), start=1):
        lines.append(
            "| {rank} | `{code}` | `{rule}` | {count} | {files} |".format(
                rank=rank,
                code=_md_cell(reason_code),
                rule=_md_cell(quality.REASON_TO_RULE_KEY.get(reason_code, reason_code)),
                count=reason_counts[reason_code],
                files=len(file_counts.get(reason_code, {})),
            )
        )

    for reason_code in ordered_reason_codes(reason_counts):
        code_counts = file_counts.get(reason_code, Counter())
        samples = samples_for_reason(
            findings,
            reason_code,
            limit=sample_limit,
        )
        lines.extend(
            [
                "",
                f"## {_md_cell(reason_code)}",
                "",
                f"- Rule: `{quality.REASON_TO_RULE_KEY.get(reason_code, reason_code)}`",
                f"- Findings: {reason_counts[reason_code]}",
                f"- Files: {len(code_counts)}",
                "",
                "### File distribution",
                "",
                "| file | findings |",
                "|---|---:|",
            ]
        )
        for file_label, count in code_counts.most_common():
            lines.append(f"| {_md_cell(file_label)} | {count} |")
        if samples:
            lines.extend(
                [
                    "",
                    "### Samples",
                    "",
                    "| file | line | source | translation |",
                    "|---|---:|---|---|",
                ]
            )
            for finding in samples:
                lines.append(
                    "| {file} | {line} | {source} | {translation} |".format(
                        file=_md_cell(_file_label(finding)),
                        line=int(finding.get("line") or 0),
                        source=_md_cell(finding.get("source")),
                        translation=_md_cell(finding.get("translation")),
                    )
                )
        else:
            lines.extend(["", "### Samples", "", "_No samples for this reason code._"])

    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "findings",
        help="Path to a quality_findings.jsonl report produced by check/sync/revision/final review",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write the Markdown report to this path instead of stdout",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"Maximum sample rows per reason code (default: {DEFAULT_SAMPLE_LIMIT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_limit < 0:
        parser.error("--sample-limit must be >= 0")
    try:
        findings = load_report_findings(args.findings)
        markdown = render_report(
            findings,
            source_path=args.findings,
            sample_limit=args.sample_limit,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"Calibration report written to: {output_path}")
        else:
            sys.stdout.write(markdown)
        return 0
    except CalibrationReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
