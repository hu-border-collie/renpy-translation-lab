"""Export quality findings as a self-contained, offline HTML review report."""

from __future__ import annotations

import html
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atomic_io import atomic_write_text
import translation_quality


REPORT_SCHEMA_VERSION = 1
DEFAULT_REPORT_FILENAME = "quality_report.html"

SEVERITY_LABELS = {
    "info": "提示",
    "low": "低",
    "medium": "中",
    "high": "高",
}

REASON_LABELS = {
    translation_quality.REASON_WAIT_TAG_INSIDE_CJK: "等待标签插入中文词内",
    translation_quality.REASON_UNCLOSED_DELIMITERS: "未闭合或破损的括号",
    translation_quality.REASON_ENGLISH_SUFFIX_ADJACENT: "中文与英文形态词尾粘连",
    translation_quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE: "可疑英文残留",
    translation_quality.REASON_CJK_LATIN_SPACING: "CJK/拉丁字符间距",
    translation_quality.REASON_HALFWIDTH_PUNCTUATION: "半角标点或异常引号",
    translation_quality.REASON_ASCII_ELLIPSIS: "ASCII 省略号",
    translation_quality.REASON_GLOSSARY_TERM_NOT_APPLIED: "glossary 译法未满足",
    translation_quality.REASON_SPEAKER_LABEL_UNTRANSLATED: "说话人标签未翻译",
    translation_quality.REASON_INTERJECTION_UNTRANSLATED: "短感叹词/拟声词未翻译",
    translation_quality.REASON_KNOWN_GARBLED_PHRASE: "已知错乱词",
    translation_quality.REASON_UNMATCHED_QUALITY_SUBJECT: "质量采集无法匹配",
    translation_quality.FINAL_REVIEW_REASON_OMISSION: "最终审校：漏译",
    translation_quality.FINAL_REVIEW_REASON_MISTRANSLATION: "最终审校：误译",
    translation_quality.FINAL_REVIEW_REASON_ADDITION: "最终审校：多余内容",
    translation_quality.FINAL_REVIEW_REASON_FORMAT: "最终审校：格式问题",
    translation_quality.FINAL_REVIEW_REASON_TERMINOLOGY: "最终审校：术语问题",
    translation_quality.FINAL_REVIEW_REASON_ADDRESS: "最终审校：称呼问题",
    translation_quality.FINAL_REVIEW_REASON_STYLE_DRIFT: "最终审校：文风漂移",
    translation_quality.FINAL_REVIEW_REASON_NEEDS_CONFIRMATION: "最终审校：待确认",
}


class QualityReportExportError(ValueError):
    """Raised when a quality report cannot be read or exported safely."""


def reason_label(reason_code: str) -> str:
    return REASON_LABELS.get(reason_code, reason_code or "未知规则")


def severity_label(severity: str) -> str:
    return SEVERITY_LABELS.get(severity, severity or "未知")


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _package_dir(manifest: Mapping[str, Any], manifest_path: str) -> str:
    explicit = str(manifest.get("_package_dir") or "").strip()
    if explicit:
        return os.path.abspath(explicit)
    path = str(manifest_path or manifest.get("_manifest_path") or "").strip()
    return os.path.dirname(os.path.abspath(path)) if path else ""


def resolve_report_source(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str = "",
) -> str:
    """Resolve the currently authoritative quality-findings JSONL path."""

    return translation_quality.resolve_quality_findings_path(
        manifest,
        package_dir=_package_dir(manifest, manifest_path),
        manifest_path=str(manifest_path or manifest.get("_manifest_path") or ""),
    )


def load_quality_findings(path: str) -> list[dict[str, Any]]:
    """Load and normalize every finding from a UTF-8 JSONL report."""

    source = Path(path)
    if not path or not source.is_file():
        raise QualityReportExportError(
            "未找到质量检查报告；请先运行 check，或确认 manifest 引用的报告仍然存在。"
        )
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise QualityReportExportError(f"无法读取质量检查报告：{source}") from exc

    findings: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise QualityReportExportError(
                f"质量检查报告第 {line_number} 行不是有效 JSON。"
            ) from exc
        if not isinstance(value, dict):
            raise QualityReportExportError(
                f"质量检查报告第 {line_number} 行不是 JSON 对象。"
            )
        findings.append(translation_quality.normalize_finding(value))
    return findings


def _location(finding: Mapping[str, Any]) -> str:
    parts = []
    if finding.get("file"):
        parts.append(str(finding["file"]))
    if finding.get("line") not in (None, ""):
        parts.append(f"第 {finding['line']} 行")
    if finding.get("item_id"):
        parts.append(f"ID {finding['item_id']}")
    return " / ".join(parts) or "位置未知"


def _finding_card(finding: Mapping[str, Any], acknowledged: set[str]) -> str:
    finding_id = str(finding.get("finding_id") or "")
    reason = str(finding.get("reason_code") or "")
    severity = str(finding.get("severity") or "medium").lower()
    disposition = str(finding.get("disposition") or "warning").lower()
    state = "acknowledged" if finding_id in acknowledged else "open"
    state_label = "已确认" if state == "acknowledged" else "待处理"
    searchable = " ".join(
        str(finding.get(field) or "")
        for field in (
            "reason_code",
            "file",
            "item_id",
            "source",
            "translation",
            "evidence",
            "suggestion",
        )
    ).casefold()

    detail_rows = []
    for label, field in (
        ("原文", "source"),
        ("译文", "translation"),
        ("证据", "evidence"),
        ("建议", "suggestion"),
    ):
        value = finding.get(field)
        if value not in (None, ""):
            detail_rows.append(
                f'<div class="detail"><dt>{label}</dt><dd>{_escape(value)}</dd></div>'
            )

    return (
        '<article class="finding" '
        f'data-reason="{_escape(reason)}" data-file="{_escape(finding.get("file"))}" '
        f'data-severity="{_escape(severity)}" data-state="{state}" '
        f'data-search="{_escape(searchable)}">'
        '<div class="finding-head">'
        f'<span class="severity severity-{_escape(severity)}">{_escape(severity_label(severity))}</span>'
        f'<span class="disposition">{_escape(disposition)}</span>'
        f'<span class="state state-{state}">{state_label}</span>'
        f'<h3>{_escape(reason_label(reason))}</h3>'
        f'<code>{_escape(reason)}</code>'
        '</div>'
        f'<p class="location">{_escape(_location(finding))}</p>'
        f'<dl>{"".join(detail_rows)}</dl>'
        '</article>'
    )


def _summary_rows(counter: Counter[str], *, labeler=None) -> str:
    if not counter:
        return '<p class="empty">暂无数据</p>'
    peak = max(counter.values())
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        label = labeler(key) if labeler else key or "未知"
        width = max(4, round(count / peak * 100))
        rows.append(
            '<div class="bar-row">'
            f'<span title="{_escape(key)}">{_escape(label)}</span>'
            f'<div class="bar-track"><i style="width:{width}%"></i></div>'
            f'<strong>{count}</strong>'
            '</div>'
        )
    return "".join(rows)


def render_quality_report_html(
    findings: list[Mapping[str, Any]],
    *,
    acknowledged_finding_ids: set[str] | None = None,
    title: str = "译文质量体检报告",
    source_name: str = "quality_findings.jsonl",
) -> str:
    """Render a portable HTML report; all finding content is HTML-escaped."""

    acknowledged = set(acknowledged_finding_ids or set())
    reason_counts = Counter(str(item.get("reason_code") or "") for item in findings)
    file_counts = Counter(str(item.get("file") or "") for item in findings)
    warning_count = sum(
        1 for item in findings if item.get("disposition") != "blocker"
    )
    blocker_count = sum(
        1 for item in findings if item.get("disposition") == "blocker"
    )
    acknowledged_count = sum(
        1
        for item in findings
        if str(item.get("finding_id") or "") in acknowledged
    )
    reason_options = "".join(
        f'<option value="{_escape(code)}">{_escape(reason_label(code))}（{count}）</option>'
        for code, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    cards = "".join(_finding_card(item, acknowledged) for item in findings)
    if not cards:
        cards = '<p class="empty result-empty">报告中没有质量报警。</p>'

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
  <title>{_escape(title)}</title>
  <style>
    :root{{--ink:#e8eaf2;--muted:#9fa6b8;--panel:#161924;--line:#30364a;--accent:#7c9cff;--warn:#f1b75b;--danger:#ff6f7d;--ok:#6dd6a8}}
    *{{box-sizing:border-box}} body{{margin:0;background:#0d0f17;color:var(--ink);font:15px/1.6 Inter,"Segoe UI","Microsoft YaHei",sans-serif}}
    main{{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:48px 0 72px}}
    header{{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;margin-bottom:28px;animation:arrive .45s ease-out both}} h1{{font-size:clamp(30px,5vw,56px);line-height:1.05;margin:0 0 10px;letter-spacing:-.03em}} .eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase}} .source{{color:var(--muted);margin:0}} .safety-note{{max-width:420px;border-left:3px solid var(--warn);padding:10px 14px;color:#d8dbe5;background:#171821}}
    .metrics{{display:grid;grid-template-columns:repeat(4,1fr);margin:24px 0;border-block:1px solid var(--line);animation:arrive .45s .06s ease-out both}} .metric{{padding:18px}} .metric+.metric{{border-left:1px solid var(--line)}} .metric strong{{display:block;font-size:30px;line-height:1.1}} .metric span{{color:var(--muted)}}
    .summaries{{display:grid;grid-template-columns:1fr 1fr;gap:36px;margin-bottom:28px;animation:arrive .45s .12s ease-out both}} .panel{{padding-top:18px;border-top:1px solid var(--line)}} h2{{font-size:18px;margin:0 0 14px}} .bar-row{{display:grid;grid-template-columns:minmax(110px,1fr) minmax(80px,1.6fr) 36px;gap:10px;align-items:center;margin:9px 0}} .bar-row>span{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .bar-track{{height:7px;background:#292e40;border-radius:20px;overflow:hidden}} .bar-track i{{display:block;height:100%;background:var(--accent);transform-origin:left;animation:grow .5s .18s ease-out both}}
    .controls{{position:sticky;top:10px;z-index:3;padding:14px;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:10px;margin:18px 0;border:1px solid var(--line);border-radius:12px;background:rgba(22,25,36,.96);backdrop-filter:blur(10px)}} input,select{{width:100%;color:var(--ink);background:#0f121b;border:1px solid var(--line);border-radius:9px;padding:10px 12px;font:inherit;transition:border-color .16s ease}} input:focus,select:focus{{border-color:var(--accent);outline:2px solid transparent}} .count{{color:var(--muted);margin:10px 2px}}
    .findings{{border-bottom:1px solid var(--line)}} .finding{{padding:20px 2px;border-top:1px solid var(--line);animation:arrive .32s ease-out both}} .finding[hidden]{{display:none}} .finding-head{{display:grid;grid-template-columns:auto auto auto 1fr;gap:8px;align-items:center}} .finding h3{{font-size:17px;margin:0}} .finding code{{grid-column:4;color:var(--muted);font-size:12px}} .severity,.disposition,.state{{border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:12px;white-space:nowrap}} .severity-high{{color:var(--danger)}} .severity-medium{{color:var(--warn)}} .state-acknowledged{{color:var(--ok)}} .state-open{{color:var(--warn)}} .location{{color:var(--muted);margin:10px 0}} dl{{margin:0}} .detail{{display:grid;grid-template-columns:50px 1fr;gap:10px;border-top:1px solid #272c3d;padding:9px 0}} dt{{color:var(--muted)}} dd{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}} .empty{{color:var(--muted)}} .result-empty[hidden]{{display:none}}
    @keyframes arrive{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}} @keyframes grow{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
    @media(max-width:760px){{header{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr 1fr}}.summaries{{grid-template-columns:1fr}}.controls{{grid-template-columns:1fr 1fr;position:static}}.finding-head{{grid-template-columns:auto auto auto}}.finding h3,.finding code{{grid-column:1/-1}}}}
    @media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}} @media print{{body{{background:white;color:#111}}main{{width:100%;padding:0}}.controls{{display:none}}.metric,.panel,.finding{{background:white;border-color:#ccc;break-inside:avoid}}.source,.location,.finding code,.metric span{{color:#555}}}}
  </style>
</head>
<body>
<main>
  <header><div><div class="eyebrow">Ren'Py Translation Lab</div><h1>{_escape(title)}</h1><p class="source">来源：{_escape(source_name)} · 报告 schema v{REPORT_SCHEMA_VERSION}</p></div><div class="safety-note">可写回不等于可交付。此报告汇总机械质量报警与最终审校 finding，不会修改译文或写回状态。</div></header>
  <section class="metrics" aria-label="质量概览"><div class="metric"><strong>{len(findings)}</strong><span>全部报警</span></div><div class="metric"><strong>{warning_count}</strong><span>warning</span></div><div class="metric"><strong>{blocker_count}</strong><span>blocker</span></div><div class="metric"><strong>{acknowledged_count}</strong><span>已确认</span></div></section>
  <section class="summaries"><div class="panel"><h2>按规则</h2>{_summary_rows(reason_counts, labeler=reason_label)}</div><div class="panel"><h2>按文件</h2>{_summary_rows(file_counts)}</div></section>
  <section class="controls" aria-label="筛选报警"><input id="search" type="search" placeholder="搜索文件、原文、译文、证据……"><select id="reason"><option value="">全部规则</option>{reason_options}</select><select id="severity"><option value="">全部严重程度</option><option value="high">高</option><option value="medium">中</option><option value="low">低</option><option value="info">提示</option></select><select id="state"><option value="">全部状态</option><option value="open">待处理</option><option value="acknowledged">已确认</option></select></section>
  <p class="count" id="count">显示 {len(findings)} / {len(findings)} 条</p>
  <section class="findings" id="findings">{cards}</section>
</main>
<script>
(()=>{{const q=id=>document.getElementById(id), cards=[...document.querySelectorAll('.finding')], empty=document.querySelector('.result-empty'); function apply(){{const needle=q('search').value.trim().toLocaleLowerCase(),reason=q('reason').value,severity=q('severity').value,state=q('state').value;let visible=0;cards.forEach(card=>{{const show=(!needle||card.dataset.search.includes(needle))&&(!reason||card.dataset.reason===reason)&&(!severity||card.dataset.severity===severity)&&(!state||card.dataset.state===state);card.hidden=!show;if(show)visible++;}});q('count').textContent=`显示 ${{visible}} / ${{cards.length}} 条`;if(empty)empty.hidden=visible!==0;}} ['search','reason','severity','state'].forEach(id=>q(id).addEventListener(id==='search'?'input':'change',apply));}})();
</script>
</body>
</html>
'''


def export_quality_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str = "",
    output_path: str = "",
) -> dict[str, Any]:
    """Read the current findings and atomically write one standalone report."""

    source_path = resolve_report_source(manifest, manifest_path=manifest_path)
    findings = load_quality_findings(source_path)
    acknowledged = {
        str(value).strip()
        for value in manifest.get("quality_acknowledged_finding_ids") or []
        if str(value).strip()
    }
    resolved_output = str(output_path or "").strip()
    if resolved_output:
        resolved_output = os.path.abspath(resolved_output)
    else:
        resolved_output = os.path.join(
            os.path.dirname(os.path.abspath(source_path)),
            DEFAULT_REPORT_FILENAME,
        )
    if os.path.isdir(resolved_output):
        raise QualityReportExportError(f"HTML 输出路径不能是目录：{resolved_output}")
    protected_paths = {
        os.path.normcase(os.path.abspath(path))
        for path in (
            source_path,
            str(manifest_path or manifest.get("_manifest_path") or ""),
        )
        if str(path or "").strip()
    }
    if os.path.normcase(resolved_output) in protected_paths:
        raise QualityReportExportError(
            "HTML 输出路径不能覆盖 manifest 或 quality_findings.jsonl。"
        )

    title_hint = str(
        manifest.get("display_name")
        or manifest.get("job_display_name")
        or "译文质量体检报告"
    ).strip()
    title = f"{title_hint} · 质量体检" if title_hint != "译文质量体检报告" else title_hint
    document = render_quality_report_html(
        findings,
        acknowledged_finding_ids=acknowledged,
        title=title,
        source_name=os.path.basename(source_path),
    )
    try:
        os.makedirs(os.path.dirname(resolved_output) or os.curdir, exist_ok=True)
        atomic_write_text(resolved_output, document)
    except OSError as exc:
        raise QualityReportExportError(f"无法写入 HTML 报告：{resolved_output}") from exc

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_path": str(manifest_path or manifest.get("_manifest_path") or ""),
        "source_path": source_path,
        "output_path": resolved_output,
        "finding_count": len(findings),
        "warning_count": sum(
            1 for item in findings if item.get("disposition") != "blocker"
        ),
        "blocker_count": sum(
            1 for item in findings if item.get("disposition") == "blocker"
        ),
        "acknowledged_count": sum(
            1
            for item in findings
            if str(item.get("finding_id") or "") in acknowledged
        ),
    }
