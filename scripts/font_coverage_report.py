#!/usr/bin/env python3
"""Read-only Ren'Py font glyph coverage report (#487 spike).

The script never starts Ren'Py, imports game code or writes into the scanned
project.  It accepts explicit text/TL inputs and emits a JSON or Markdown
report for manual review.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atomic_io import atomic_write_text  # noqa: E402
import font_coverage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读扫描 Ren'Py game 目录中的静态 font 引用与字体 cmap，"
            "报告 checked / missing / unknown 字形覆盖。"
        )
    )
    parser.add_argument(
        "--game-root",
        required=True,
        help="Ren'Py game 目录（包含 .rpy 与字体资源；不会被修改）。",
    )
    parser.add_argument(
        "--tl-dir",
        default="",
        help="可选：tl/<language> 目录，best-effort 提取 new/dialogue 译文样本。",
    )
    parser.add_argument(
        "--text-file",
        default="",
        help="可选：每行一个待检查文本样本。",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="可选：显式文本样本（可重复）。",
    )
    parser.add_argument(
        "--max-missing-chars",
        type=int,
        default=font_coverage.DEFAULT_MAX_MISSING_CHARS,
        help="报告中每个字体最多列出的缺字数量。",
    )
    parser.add_argument(
        "--output",
        choices=("markdown", "json"),
        default="markdown",
        help="输出格式（默认 markdown）。",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="可选：原子写入指定文件；省略时写 stdout。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    game_root = Path(args.game_root)
    if not game_root.is_dir():
        print(f"game-root 不是目录：{game_root}", file=sys.stderr)
        return 2
    tl_dir = Path(args.tl_dir) if args.tl_dir else None
    if tl_dir is not None and not tl_dir.is_dir():
        print(f"tl-dir 不是目录：{tl_dir}", file=sys.stderr)
        return 2
    text_file = Path(args.text_file) if args.text_file else None
    if text_file is not None and not text_file.is_file():
        print(f"text-file 不存在：{text_file}", file=sys.stderr)
        return 2
    try:
        report = font_coverage.analyze_font_coverage(
            game_root,
            tl_dir=tl_dir,
            text_file=text_file,
            text_values=args.text,
            max_missing_chars=args.max_missing_chars,
        )
    except font_coverage.FontCoverageError as exc:
        print(f"font coverage error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    rendered = (
        font_coverage.report_to_json(report)
        if args.output == "json"
        else font_coverage.format_report_markdown(report)
    )
    if args.output_file:
        atomic_write_text(os.fspath(args.output_file), rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
