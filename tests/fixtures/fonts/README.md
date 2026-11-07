# Keyword review header font fixture

`noto_sans_cjk_sc_bold_header_subset.ttf` is a 12 KB subset of Noto Sans CJK SC Bold
(face 2 of `NotoSansCJK-Bold.ttc`). It contains only the characters used by
`KEYWORD_MERGE_TABLE_HEADERS`, so the theme probe can render those Chinese header
labels on CI images that ship no CJK font.

Upstream: Google Noto fonts (https://github.com/notofonts/noto-cjk), SIL Open Font
License 1.1. The license text already vendored for the font-coverage fixtures is
`tests/fixtures/font_coverage_minimal/game/fonts/LICENSE-OFL-1.1.txt`.
