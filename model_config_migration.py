"""Developer-only, explicit P1 migration CLI; production activation is P2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_routing_migration_store import (
    migrate_config_file, preview_config_file, rollback_config_file,
)


def refusal_details(exc: Exception) -> tuple[str, str]:
    """Classify known refusal messages without reflecting their input values."""
    message = str(exc)
    for prefix, reason, action in (
        ("Configuration changed since preview", "stale_preview", "重新预览配置，再使用新的源指纹"),
        ("Configuration changed since migration", "edited_after_migration", "保留当前编辑，人工核对备份；禁止直接覆盖"),
        ("Configuration is locked", "config_locked", "检查活动写入进程；确认无写入者后再处理遗留锁"),
        ("Embedded credential", "embedded_credentials", "移除配置中的密钥值，改用凭据引用"),
        ("Ambiguous legacy", "ambiguous_legacy_models", "核对实际模型；明确 sync.model 并与 sync.models 首项保持一致"),
        ("Implicit rotation", "implicit_rotation_pool", "确认并在配置副本中冻结完整 sync.models 轮换列表"),
        ("File migration requires", "missing_explicit_batch_model", "核对项目默认模型并明确设置 batch.model"),
        ("Migration backup fingerprint", "backup_changed", "寻找未修改的原始备份和匹配报告"),
        ("Invalid migration report", "invalid_report", "使用该配置原始迁移生成的报告"),
        ("Invalid model_routing", "invalid_v1", "检查新 section 的 schema、引用与字段类型；不会回退旧配置"),
    ):
        if message.startswith(prefix):
            return reason, action
    if isinstance(exc, OSError):
        return "file_io_failed", "检查路径和文件权限；原配置未提交时可重新预览后重试"
    return "invalid_config", "检查 JSON 格式、Provider 配置、模型与轮换列表；参阅迁移文档"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="模型配置离线迁移（P1 暂存工具；生产 reader/GUI 激活属于 P2/P3）",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("preview", "只读检查并返回源指纹"),
        ("migrate", "备份并暂存 v1；保留旧字段，暂不改变生产路由"),
        ("rollback", "校验指纹并恢复原配置字节"),
    ):
        command = sub.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--config", type=Path, required=True, help="显式指定配置副本路径")
        command.add_argument("--json", action="store_true", help="输出稳定 JSON 状态")
        if name == "migrate":
            command.add_argument("--expected-fingerprint", required=True, help="preview 返回的 source_fingerprint")
            command.add_argument("--stage-only", action="store_true", required=True,
                                 help="确认本阶段只暂存 schema，不激活生产路由")
        elif name == "rollback":
            command.add_argument("--report", type=Path, required=True, help="迁移生成的报告路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "preview":
            report = preview_config_file(args.config)
        elif args.action == "migrate":
            report = migrate_config_file(args.config, expected_fingerprint=args.expected_fingerprint)
        else:
            report = rollback_config_file(args.config, report_path=args.report)
    except (ValueError, OSError, TypeError) as exc:
        # Provider parsers and OS exceptions can include endpoints/private paths.
        # Closed error metadata avoids leaking those values through --json/logs.
        reason, next_action = refusal_details(exc)
        report = {"status": "refused", "error_code": "MODEL_CONFIG_MIGRATION_REFUSED",
                  "error_type": type(exc).__name__,
                  "reason": reason, "next_action": next_action}
        print(json.dumps(report, ensure_ascii=False))
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
