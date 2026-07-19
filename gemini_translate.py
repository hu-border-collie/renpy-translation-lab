# -*- coding: utf-8 -*-
import argparse

from project_version import __version__
import translator_runtime as runtime
from translator_runtime import *  # noqa: F401,F403


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronous translator for Ren'Py tl files. The default command creates "
            "a reviewable preview and never modifies project scripts."
        )
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
    if args.apply:
        runtime.apply_sync_translation_preview(args.apply)
    else:
        runtime.run_translation(prepare=args.prepare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
