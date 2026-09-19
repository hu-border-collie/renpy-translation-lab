init python:
    import os
    import pathlib
    marker = os.environ.get("FONT_COVERAGE_CANARY_PATH")
    if marker:
        pathlib.Path(marker).write_text("must not run", encoding="utf-8")
