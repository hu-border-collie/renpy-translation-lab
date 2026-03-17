# -*- coding: utf-8 -*-
import argparse

import translator_runtime as runtime
from translator_runtime import *  # noqa: F401,F403


def build_arg_parser():
    return argparse.ArgumentParser(
        description="Synchronous translator for Ren'Py tl files using the google-genai SDK."
    )


def main(argv=None):
    parser = build_arg_parser()
    parser.parse_args(argv)
    runtime.initialize_runtime_logging()
    runtime.run_translation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
