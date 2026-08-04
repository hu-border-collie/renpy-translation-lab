"""Explicit package entrypoint for ``python -m relation_analyzer``.

Prefer ``python -m relation_analyzer.cli`` or the root ``extract_relations.py``
shim for documented usage. This module exists so package-level ``-m`` does not
emit a RuntimeWarning / missing-__main__ failure while still routing to the CLI.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
