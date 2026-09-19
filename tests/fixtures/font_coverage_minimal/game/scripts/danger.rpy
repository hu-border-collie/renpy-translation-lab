init python:
    import pathlib
    pathlib.Path("FONT_COVERAGE_SIDE_EFFECT.txt").write_text("must not run", encoding="utf-8")
