"""Bounded exception facts without messages, request data, paths or locals."""
from __future__ import annotations

from collections.abc import Mapping
import re


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,79}\Z')
_LOCATION = re.compile(r'[A-Za-z0-9_.-]{1,96}\.pyw?:[A-Za-z_][A-Za-z0-9_]{0,79}:[0-9]{1,7}\Z')
_PROVIDER_STATUSES = frozenset({
    'CANCELLED', 'UNKNOWN', 'INVALID_ARGUMENT', 'DEADLINE_EXCEEDED', 'NOT_FOUND',
    'ALREADY_EXISTS', 'PERMISSION_DENIED', 'RESOURCE_EXHAUSTED', 'FAILED_PRECONDITION',
    'ABORTED', 'OUT_OF_RANGE', 'UNIMPLEMENTED', 'INTERNAL', 'UNAVAILABLE',
    'DATA_LOSS', 'UNAUTHENTICATED',
})
EXCEPTION_DIAGNOSTIC_FIELDS = (
    'exception_type', 'exception_chain', 'http_status', 'provider_status',
    'traceback_location', 'os_errno', 'os_winerror', 'attempt_count',
)
BOOTSTRAP_DIAGNOSTIC_FIELDS = (
    'failure_reason', 'error_category', 'retryable', 'failure_stage',
    *EXCEPTION_DIAGNOSTIC_FIELDS, 'batch_start', 'batch_end', 'batch_total',
    'phase_elapsed_seconds', 'elapsed_seconds',
    'diagnostic_persistence_failed', 'diagnostic_write_exception',
)


def _attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def exception_http_status(exc: BaseException) -> int | None:
    """Read only numeric HTTP status attributes, never response text or headers."""
    for owner in (exc, _attribute(exc, 'response')):
        for name in ('status_code', 'code', 'status', 'http_status'):
            value = _attribute(owner, name)
            if not isinstance(value, int):
                value = _attribute(value, 'value')
            if type(value) is int and 100 <= value <= 599:
                return value
    return None


def safe_exception_diagnostics(values: Mapping[str, object]) -> dict[str, object]:
    """Revalidate the allowlist when facts cross a public or persistence boundary."""
    result: dict[str, object] = {}
    for key in ('exception_type', 'exception_chain', 'traceback_location'):
        value = values.get(key)
        if not isinstance(value, str):
            continue
        parts = value.split(' -> ' if key == 'exception_chain' else ' > ')
        pattern = _LOCATION if key == 'traceback_location' else _IDENTIFIER
        limit = 1 if key == 'exception_type' else 8
        if 1 <= len(parts) <= limit and all(pattern.fullmatch(part) for part in parts):
            result[key] = value
    status = values.get('http_status')
    if type(status) is int and 100 <= status <= 599:
        result['http_status'] = status
    provider_status = values.get('provider_status')
    if isinstance(provider_status, str) and provider_status in _PROVIDER_STATUSES:
        result['provider_status'] = provider_status
    for key in ('os_errno', 'os_winerror', 'attempt_count'):
        value = values.get(key)
        if type(value) is int and 0 < value <= 1_000_000:
            result[key] = value
    return result


def capture_exception_diagnostics(exc: BaseException) -> dict[str, object]:
    """Capture type/cause names and frame coordinates before wrapping an error.

    No exception string, source line, absolute path, local variable, response
    body or header is inspected. Cause cycles and long stacks are bounded.
    """
    result: dict[str, object] = {'exception_type': type(exc).__name__}
    status = exception_http_status(exc)
    if status is not None:
        result['http_status'] = status
    result['provider_status'] = _attribute(exc, 'status')
    result['os_errno'] = _attribute(exc, 'errno')
    result['os_winerror'] = _attribute(exc, 'winerror')
    chain, locations, seen = [], [], set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(type(current).__name__)
        frames = []
        trace = current.__traceback__
        while trace is not None:
            code = trace.tb_frame.f_code
            filename = code.co_filename.replace('\\', '/').rsplit('/', 1)[-1]
            location = f'{filename}:{code.co_name}:{trace.tb_lineno}'
            if _LOCATION.fullmatch(location):
                frames.append(location)
            trace = trace.tb_next
        locations.extend(frames[-8:])
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    result['exception_chain'] = ' -> '.join(chain)
    if locations:
        result['traceback_location'] = ' > '.join(locations[-8:])
    return safe_exception_diagnostics(result)
