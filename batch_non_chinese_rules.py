# -*- coding: utf-8 -*-
"""Configurable file-path allowlists for batch non-Chinese validation."""
from __future__ import annotations

import copy

DEFAULT_NON_CHINESE_RULES = {
    'static_name_credit_rel_paths': [
        'screens_menu_about.rpy',
        'screens_menu_gallery_bg.rpy',
        'screens_patronlistitem.rpy',
    ],
    'static_name_credit_unconditional_rel_paths': [
        'screens_patronlistitem.rpy',
    ],
    'charselect_rel_paths': [
        'screens_charselect.rpy',
    ],
    'player_name_comparison_rel_paths': [
        'script.rpy',
    ],
    'define_rel_path_suffixes': [
        'script_define.rpy',
    ],
    'define_rel_path_prefixes': [
        'script_characters',
    ],
}

_RULE_LIST_KEYS = tuple(DEFAULT_NON_CHINESE_RULES.keys())


def _normalize_path_entry(value: object) -> str:
    if not isinstance(value, str):
        return ''
    return value.strip().replace('\\', '/').lower()


def _normalize_path_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        entry = _normalize_path_entry(raw)
        if not entry or entry in seen:
            continue
        seen.add(entry)
        normalized.append(entry)
    return normalized


def _merge_rule_lists(default_values: list[str], configured_values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for entry in list(default_values) + list(configured_values):
        if not entry or entry in seen:
            continue
        seen.add(entry)
        merged.append(entry)
    return merged


def normalize_non_chinese_rules(configured: object | None) -> dict[str, list[str]]:
    rules = {
        key: list(DEFAULT_NON_CHINESE_RULES[key])
        for key in _RULE_LIST_KEYS
    }
    if not isinstance(configured, dict):
        return rules

    for key in _RULE_LIST_KEYS:
        if key not in configured:
            continue
        rules[key] = _normalize_path_list(configured.get(key))

    extra_static = _normalize_path_list(configured.get('extra_static_name_credit_rel_paths'))
    if extra_static:
        rules['static_name_credit_rel_paths'] = _merge_rule_lists(
            rules['static_name_credit_rel_paths'],
            extra_static,
        )

    return rules


def load_non_chinese_rules(translator_config: dict | None) -> dict[str, list[str]]:
    if not isinstance(translator_config, dict):
        return normalize_non_chinese_rules(None)
    batch = translator_config.get('batch')
    if not isinstance(batch, dict):
        return normalize_non_chinese_rules(None)
    configured = batch.get('non_chinese_validation')
    if configured is None:
        configured = batch.get('non_chinese_rules')
    return normalize_non_chinese_rules(configured)


def effective_non_chinese_rules(
    manifest: dict | None,
    *,
    runtime_rules: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    if isinstance(manifest, dict):
        manifest_rules = manifest.get('non_chinese_rules')
        if isinstance(manifest_rules, dict):
            return normalize_non_chinese_rules(manifest_rules)
    if isinstance(runtime_rules, dict):
        return normalize_non_chinese_rules(runtime_rules)
    return normalize_non_chinese_rules(None)


def manifest_non_chinese_rules_fields(source_manifest: dict | None = None) -> dict[str, dict[str, list[str]]]:
    rules = effective_non_chinese_rules(source_manifest)
    return {'non_chinese_rules': copy.deepcopy(rules)}


def rel_path_matches(rel_path: str, rel_name: str, candidates: list[str]) -> bool:
    normalized_path = _normalize_path_entry(rel_path)
    normalized_name = _normalize_path_entry(rel_name)
    for candidate in candidates:
        if not candidate:
            continue
        if normalized_name == candidate or normalized_path == candidate:
            return True
        if normalized_path.endswith('/' + candidate):
            return True
    return False


def rel_path_has_suffix(rel_path: str, suffixes: list[str]) -> bool:
    normalized_path = _normalize_path_entry(rel_path)
    for suffix in suffixes:
        if normalized_path.endswith(suffix):
            return True
    return False


def rel_path_has_prefix(rel_path: str, prefixes: list[str]) -> bool:
    normalized_path = _normalize_path_entry(rel_path)
    for prefix in prefixes:
        if normalized_path.startswith(prefix):
            return True
    return False