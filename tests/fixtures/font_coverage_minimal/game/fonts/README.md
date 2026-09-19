# Font coverage spike fixtures (#487)

These fonts are redistributable test fixtures for the read-only font coverage spike.
They were generated with `fontTools` 4.46.0 `subset` from Debian/Ubuntu Noto packages:

- `test_cjk_subset.ttf`: subset of `NotoSansCJK-Regular.ttc` face 2
  (`Noto Sans CJK SC`), covering the characters in `texts/cjk_covered.txt`
  and `tl/schinese/strings.rpy`.
  SHA-256: `13ccf7d416e7980875b7bcc4c49d9786e7bd93b3fd8b390d4512bea99428f545`
- `test_latin_subset.ttf`: subset of `NotoMono-Regular.ttf`, covering
  `texts/ascii.txt` but intentionally missing CJK glyphs.
  SHA-256: `3bef2e666f791d9f3f34847504c72098770a8117c6b0a441c0774d48b1d13e9b`

Upstream: Google Noto fonts (<https://github.com/notofonts/noto-fonts>),
licensed under the SIL Open Font License 1.1; see `LICENSE-OFL-1.1.txt`.

`corrupt.ttf` is project-generated invalid bytes used only for the
missing/corrupt-font negative case; it is not a font.

No private game fonts, scripts, or maps are committed.
