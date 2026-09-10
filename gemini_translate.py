# -*- coding: utf-8 -*-
import argparse
import sys

from project_version import __version__
import translator_runtime as runtime
from translator_runtime import *  # noqa: F401,F403


DEPRECATION_NOTICE = (
    "[DEPRECATED] \u540c\u6b65\u517c\u5bb9\u5165\u53e3\u4e0d\u518d\u6301\u4e45\u5316\u53ef\u6062\u590d run\u3002\n"
    "  \u63a8\u8350\u4f7f\u7528\uff1apython gemini_translate_batch.py sync-start --profile <PROFILE_ID>\n"
    "  \u7136\u540e check <RUN> / apply <RUN>\uff1b\u547d\u4ee4\u6620\u5c04\u89c1 docs/sync_workflow.md\u3002"
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronous translator for Ren'Py tl files. The default command creates "
            "a reviewable preview and never modifies project scripts. "
            "Deprecated: this compatibility entry does not persist a resumable run; "
            "prefer gemini_translate_batch.py sync-start/check/apply."
        ),
        epilog=(
            "\u8fc1\u79fb\u6620\u5c04\uff1agemini_translate.py -> gemini_translate_batch.py sync-start\uff1b"
            "gemini_translate.py --apply M -> gemini_translate_batch.py check <RUN> \u540e apply <RUN>\u3002"
        ),
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument(
        '--apply',
        metavar='MANIFEST',
        help='Apply a previously generated sync preview after source revalidation.',
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Run configured prepare steps before generating the preview.',
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.apply and args.prepare:
        parser.error('--prepare cannot be combined with --apply')
    runtime.initialize_runtime_logging()
    print(DEPRECATION_NOTICE, file=sys.stderr)
    if args.apply:
        runtime.apply_sync_translation_preview(args.apply)
    else:
        runtime.run_translation(prepare=args.prepare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
