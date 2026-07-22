# -*- coding: utf-8 -*-
"""Token and cost estimation for Batch translation manifests."""
from __future__ import annotations

import json
import os
from datetime import datetime

DEFAULT_PRICING = {
    'version': 1,
    'currency': 'USD',
    'chars_per_input_token': 4.0,
    # Batch API rates (≈50% of standard paid tier). See:
    # https://ai.google.dev/gemini-api/docs/pricing
    'models': {
        'gemini-3.6-flash': {
            'input_per_million': 0.75,
            'output_per_million': 3.75,
        },
        'gemini-3.5-flash': {
            'input_per_million': 0.75,
            'output_per_million': 4.50,
        },
        'gemini-3.5-flash-lite': {
            'input_per_million': 0.15,
            'output_per_million': 1.25,
        },
        'gemini-3.1-flash-lite': {
            'input_per_million': 0.125,
            'output_per_million': 0.75,
        },
        'gemini-3-flash-preview': {
            'input_per_million': 0.50,
            'output_per_million': 3.00,
        },
    },
}


def _coerce_positive_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_pricing_config(translator_config=None):
    pricing = dict(DEFAULT_PRICING)
    pricing['models'] = dict(DEFAULT_PRICING.get('models') or {})

    if not isinstance(translator_config, dict):
        return pricing

    batch = translator_config.get('batch')
    if not isinstance(batch, dict):
        return pricing

    configured = batch.get('pricing')
    if not isinstance(configured, dict):
        return pricing

    if configured.get('version') is not None:
        pricing['version'] = configured.get('version')
    if isinstance(configured.get('currency'), str) and configured['currency'].strip():
        pricing['currency'] = configured['currency'].strip()
    pricing['chars_per_input_token'] = _coerce_positive_float(
        configured.get('chars_per_input_token'),
        pricing['chars_per_input_token'],
    )

    model_table = configured.get('models')
    if isinstance(model_table, dict):
        merged = dict(pricing['models'])
        for model_name, rates in model_table.items():
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            if not isinstance(rates, dict):
                continue
            merged[model_name.strip()] = {
                'input_per_million': _coerce_positive_float(
                    rates.get('input_per_million'),
                    (merged.get(model_name.strip()) or {}).get('input_per_million', 0.0),
                ),
                'output_per_million': _coerce_positive_float(
                    rates.get('output_per_million'),
                    (merged.get(model_name.strip()) or {}).get('output_per_million', 0.0),
                ),
            }
        pricing['models'] = merged

    return pricing


def resolve_model_pricing(model_name, pricing_config):
    models = pricing_config.get('models') if isinstance(pricing_config, dict) else {}
    if not isinstance(models, dict):
        return None

    normalized = str(model_name or '').strip()
    if normalized in models:
        return models[normalized]

    best_match = ''
    best_rates = None
    for candidate, rates in models.items():
        if not isinstance(candidate, str) or not candidate:
            continue
        if normalized.startswith(candidate) and len(candidate) > len(best_match):
            best_match = candidate
            best_rates = rates
    return best_rates


def iter_request_text_parts(request_payload):
    request = request_payload.get('request') if isinstance(request_payload, dict) else None
    if not isinstance(request, dict):
        request = request_payload if isinstance(request_payload, dict) else {}

    system_instruction = request.get('system_instruction') or {}
    for part in system_instruction.get('parts') or []:
        if isinstance(part, dict):
            text = part.get('text')
            if isinstance(text, str) and text:
                yield text

    for content in request.get('contents') or []:
        if not isinstance(content, dict):
            continue
        for part in content.get('parts') or []:
            if isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str) and text:
                    yield text


def estimate_jsonl_input_chars(jsonl_path):
    total_chars = 0
    request_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            request_count += 1
            total_chars += sum(len(text) for text in iter_request_text_parts(payload))
    return request_count, total_chars


def estimate_manifest_tokens(manifest, pricing_config=None):
    pricing_config = pricing_config or DEFAULT_PRICING
    jsonl_path = manifest.get('input_jsonl_path') or ''
    if not jsonl_path or not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f'Batch input JSONL not found: {jsonl_path or "(missing)"}')

    request_count, input_chars = estimate_jsonl_input_chars(jsonl_path)
    chars_per_token = _coerce_positive_float(
        pricing_config.get('chars_per_input_token'),
        DEFAULT_PRICING['chars_per_input_token'],
    )
    estimated_input_tokens = int((input_chars / chars_per_token) + 0.999999)

    settings = manifest.get('settings') if isinstance(manifest.get('settings'), dict) else {}
    max_output_tokens = int(settings.get('max_output_tokens') or 0)

    summary = manifest.get('summary') if isinstance(manifest.get('summary'), dict) else {}
    chunk_count = int(summary.get('chunk_count') or 0)
    if chunk_count <= 0 and isinstance(manifest.get('chunks'), list):
        chunk_count = len(manifest['chunks'])

    estimated_output_tokens = max(0, chunk_count * max_output_tokens)

    return {
        'request_count': request_count,
        'chunk_count': chunk_count,
        'input_chars': input_chars,
        'estimated_input_tokens': estimated_input_tokens,
        'estimated_output_tokens_max': estimated_output_tokens,
        'chars_per_input_token': chars_per_token,
        'max_output_tokens_per_chunk': max_output_tokens,
    }


def estimate_manifest_cost(manifest, pricing_config=None, translator_config=None):
    pricing_config = pricing_config or load_pricing_config(translator_config)
    token_summary = estimate_manifest_tokens(manifest, pricing_config)
    model_name = str(manifest.get('batch_model') or '').strip()
    model_rates = resolve_model_pricing(model_name, pricing_config) or {}

    input_rate = _coerce_positive_float(model_rates.get('input_per_million'), 0.0)
    output_rate = _coerce_positive_float(model_rates.get('output_per_million'), 0.0)

    input_tokens = token_summary['estimated_input_tokens']
    output_tokens = token_summary['estimated_output_tokens_max']
    estimated_cost_min = (input_tokens * input_rate) / 1_000_000
    estimated_cost_max = estimated_cost_min + (output_tokens * output_rate) / 1_000_000

    return {
        **token_summary,
        'model': model_name,
        'pricing_version': pricing_config.get('version'),
        'currency': pricing_config.get('currency') or 'USD',
        'input_per_million': input_rate,
        'output_per_million': output_rate,
        'estimated_cost_min': round(estimated_cost_min, 4),
        'estimated_cost_max': round(estimated_cost_max, 4),
        'estimated_at': datetime.now().isoformat(timespec='seconds'),
    }


def attach_cost_estimate_to_manifest(manifest, pricing_config=None, translator_config=None):
    estimate = estimate_manifest_cost(
        manifest,
        pricing_config=pricing_config,
        translator_config=translator_config,
    )
    manifest['cost_estimate'] = estimate
    return estimate


def cost_estimate_exceeds_max(estimate, max_cost):
    try:
        limit = float(max_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid max-cost value: {max_cost!r}') from exc
    if limit < 0:
        raise ValueError('max-cost must be non-negative.')
    return float(estimate.get('estimated_cost_max') or 0.0) > limit


def format_cost_estimate_lines(estimate):
    currency = estimate.get('currency') or 'USD'
    lines = [
        'Cost estimate:',
        f"- Model: {estimate.get('model') or '(unknown)'}",
        f"- Requests: {estimate.get('request_count', 0)}",
        f"- Estimated input tokens: {estimate.get('estimated_input_tokens', 0)}",
        f"- Estimated output tokens (max): {estimate.get('estimated_output_tokens_max', 0)}",
        (
            f"- Estimated cost: {estimate.get('estimated_cost_min', 0):.4f} "
            f"to {estimate.get('estimated_cost_max', 0):.4f} {currency}"
        ),
        f"- Pricing table version: {estimate.get('pricing_version', '')}",
    ]
    if not estimate.get('input_per_million') and not estimate.get('output_per_million'):
        lines.append('- Note: model pricing is not configured; cost range may be 0.')
    return lines