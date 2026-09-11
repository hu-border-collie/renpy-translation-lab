# Ren'Py coverage block attribution fixture

Original synthetic fixture for issue
[#426](https://github.com/hu-border-collie/renpy-translation-lab/issues/426).

- **Source**: written for this repository by the project maintainers.
- **License**: MIT, same as the repository root `LICENSE`. It may be
  redistributed with this repository and in downstream test packages.
- **Content boundary**: all dialogue, speaker labels, comments, and paths are
  invented. No private game code, maps, assets, dialogue, or project identity
  is copied here.

`game/tl/schinese/attribution_samples.rpy` intentionally mixes:

- cross-line extraction cases (multi-line triple-quoted and backslash-continued
  strings, extracted as multiline candidates since #460);
- structures that currently produce `parse_error` or `unsupported` inventory
  candidates (dynamic f-strings, non-standard `old` markers, orphan `old`
  rows, and trailing quoted comments);
- an `unknown` single-quoted dialogue span;
- legitimate exclusions (`voice`, asset paths);
- negative controls that must stay extractable: normal paired dialogue, a
  translated speaker label, an empty target that is still pending, a dialogue
  line following excluded asset statements, and visible dialogue containing an
  asset-path-looking substring that must not be over-excluded.

The fixture is consumed read-only by
`scripts/coverage_block_attribution.py` and
`tests/test_coverage_attribution.py`. It is not a translation workflow
fixture and must not be written back to.
