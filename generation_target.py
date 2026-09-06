"""Generation-target contract: catalog language vs model output language.

Issue #412 contract 1: this tool currently generates Simplified Chinese only.
``tl_subdir`` / ``prepare.language`` name the Ren'Py TL catalog directory; they
do not change the model target. Custom catalog directory names remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHINESE = 'schinese'
DEFAULT_GENERATION_TARGET = SCHINESE
SUPPORTED_GENERATION_TARGETS = frozenset({SCHINESE})
GENERATION_TARGET_PROMPT_NAME = 'Simplified Chinese'
UNSUPPORTED_CODE = 'generation_target.unsupported'
CATALOG_NOT_GENERATION_WARNING_PREFIX = (
    'Catalog language only names the Ren\'Py TL directory; '
    'model generation remains Simplified Chinese'
)

# Accepted spellings that still mean the supported generation target.
_SCHINESE_ALIASES = frozenset({
    'schinese',
    'zh-cn',
    'zh_cn',
    'zh-hans',
    'zh_hans',
    'simplified chinese',
    'simplified_chinese',
})

# Well-known Ren'Py language directory names that users often confuse with
# the model generation target. Custom catalog slugs are not listed here so
# they stay silent-compatible.
NON_CHINESE_CATALOG_HINTS = frozenset({
    'japanese',
    'korean',
    'tchinese',
    'english',
    'french',
    'spanish',
    'german',
    'russian',
    'italian',
    'portuguese',
    'polish',
    'turkish',
    'arabic',
    'thai',
    'vietnamese',
    'indonesian',
    'malay',
    'hindi',
    'ukrainian',
    'czech',
    'dutch',
    'swedish',
    'finnish',
    'danish',
    'norwegian',
    'greek',
    'hungarian',
    'romanian',
    'bulgarian',
})


class GenerationTargetError(ValueError):
    """Raised when a configured generation target is outside the support contract."""

    def __init__(self, message, *, code=UNSUPPORTED_CODE, requested=''):
        super().__init__(message)
        self.code = str(code or UNSUPPORTED_CODE)
        self.requested = str(requested or '')


@dataclass(frozen=True)
class GenerationTargetResolution:
    requested: str
    canonical: str
    supported: bool
    explicit: bool

    def to_dict(self):
        return {
            'requested': self.requested,
            'canonical': self.canonical,
            'supported': self.supported,
            'explicit': self.explicit,
            'prompt_name': GENERATION_TARGET_PROMPT_NAME if self.supported else '',
        }


def canonicalize_generation_target(value) -> str:
    """Return the canonical generation-target id, or a stripped lowercase token."""
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_GENERATION_TARGET
    token = value.strip().lower().replace(' ', '_')
    if token in _SCHINESE_ALIASES:
        return SCHINESE
    return token


def resolve_generation_target(value) -> GenerationTargetResolution:
    """Resolve a configured generation target (omitted/blank → schinese)."""
    explicit = isinstance(value, str) and bool(value.strip())
    requested = value.strip() if explicit else DEFAULT_GENERATION_TARGET
    canonical = canonicalize_generation_target(value)
    return GenerationTargetResolution(
        requested=requested,
        canonical=canonical,
        supported=canonical in SUPPORTED_GENERATION_TARGETS,
        explicit=explicit,
    )


def resolve_generation_target_from_config(config) -> GenerationTargetResolution:
    """Read optional ``generation.target_language`` from translator_config."""
    mapping = config if isinstance(config, dict) else {}
    generation = mapping.get('generation')
    raw = None
    if isinstance(generation, dict):
        raw = generation.get('target_language')
    return resolve_generation_target(raw)


def unsupported_message(resolution: GenerationTargetResolution) -> str:
    requested = resolution.requested or resolution.canonical
    return (
        f"ERROR: [{UNSUPPORTED_CODE}] generation.target_language={requested!r} "
        'is not supported. This tool currently generates Simplified Chinese only '
        f'(generation.target_language={SCHINESE!r}). '
        'tl_subdir / prepare.language only select the Ren\'Py catalog directory; '
        'they do not change the model target.'
    )


def require_supported(resolution: GenerationTargetResolution) -> str:
    """Return the canonical target or raise :class:`GenerationTargetError`."""
    if not resolution.supported:
        raise GenerationTargetError(
            unsupported_message(resolution),
            requested=resolution.requested or resolution.canonical,
        )
    return resolution.canonical


def catalog_language_hint(prep_language='', tl_subdir='') -> str:
    """Return a catalog token that looks like a non-Chinese generation target."""
    candidates = []
    if isinstance(prep_language, str) and prep_language.strip():
        candidates.append(prep_language.strip())
    if isinstance(tl_subdir, str) and tl_subdir.strip():
        segment = tl_subdir.replace('\\', '/').rstrip('/').split('/')[-1]
        if segment:
            candidates.append(segment)
    for raw in candidates:
        token = raw.strip().lower()
        if not token:
            continue
        if canonicalize_generation_target(token) == SCHINESE:
            continue
        if token in NON_CHINESE_CATALOG_HINTS:
            return raw.strip()
    return ''


def catalog_not_generation_warning(catalog_language) -> str:
    return (
        f"{CATALOG_NOT_GENERATION_WARNING_PREFIX} "
        f"(catalog={catalog_language!s})."
    )
