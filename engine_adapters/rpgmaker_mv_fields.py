"""Explicit MV display roles; never infer translatability from arbitrary JSON.

The paths follow the core TextManager consumers, not a game's current strings.
Unknown members, command reserves (20/23), database names and MapInfos remain
outside this policy. Optional occurrence scopes can further narrow these roles.
"""
from __future__ import annotations

import re
from typing import Any


DISPLAY_FIELDS = ("system_terms", "map_display_name")
DISPLAY_ROLES = frozenset({"system_term", "system_message", "map_display_name"})
FIELD_POLICY_VERSION = 1
TERM_INDICES = {"basic": tuple(range(10)), "params": tuple(range(8)),
                "commands": (*range(20), 21, 22, 24, 25)}
TERM_MESSAGES = frozenset("""
alwaysDash commandRemember bgmVolume bgsVolume meVolume seVolume possession
expTotal expNext saveMessage loadMessage file partyName emerge preemptive
surprise escapeStart escapeFailure victory defeat obtainExp obtainGold
obtainItem levelUp obtainSkill useItem criticalToEnemy criticalToActor
actorDamage actorRecovery actorGain actorLoss actorDrain actorNoDamage actorNoHit
enemyDamage enemyRecovery enemyGain enemyLoss enemyDrain enemyNoDamage enemyNoHit
evasion magicEvasion magicReflection counterAttack substitute buffAdd debuffAdd
buffRemove actionFailure
""".split())
_MAP = re.compile(r"data/Map[0-9]+\.json")


def parse_display_fields(value: Any) -> tuple[str, ...]:
    """Reject unknown or malformed opt-ins instead of silently widening scope."""
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) or item not in DISPLAY_FIELDS for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("rpgmaker_mv.display_fields must list unique system_terms / map_display_name values.")
    return tuple(sorted(value))


def display_role(relative: str, path: tuple, fields: tuple[str, ...]) -> str:
    """Return a role only for an exact typed path in an explicitly enabled group."""
    if "map_display_name" in fields and _MAP.fullmatch(relative) and path == ("displayName",):
        return "map_display_name"
    if "system_terms" not in fields or relative != "data/System.json" or len(path) != 3 or path[0] != "terms":
        return ""
    if path[1] in TERM_INDICES and type(path[2]) is int and path[2] in TERM_INDICES[path[1]]:
        return "system_term"
    if path[1] == "messages" and type(path[2]) is str and path[2] in TERM_MESSAGES:
        return "system_message"
    return ""


def display_slots(data: Any, relative: str, fields: tuple[str, ...]) -> list[tuple[tuple, str, str]]:
    """Read existing string leaves, rejecting malformed selected containers.

    Missing optional members and null leaves stay untouched. Present non-string
    values at approved positions are malformed input, never zero pending work.
    """
    rows = []
    if "map_display_name" in fields and _MAP.fullmatch(relative):
        if not isinstance(data, dict):
            raise ValueError("rpgmaker_mv.display.map_object")
        if "displayName" in data:
            if not isinstance(data["displayName"], str):
                raise ValueError("rpgmaker_mv.display.string")
            rows.append((("displayName",), data["displayName"], "map_display_name"))
    if "system_terms" in fields and relative == "data/System.json":
        if not isinstance(data, dict) or not isinstance(data.get("terms"), dict):
            raise ValueError("rpgmaker_mv.display.terms_object")
        terms = data["terms"]
        for group, indices in TERM_INDICES.items():
            if group not in terms:
                continue
            values = terms[group]
            if not isinstance(values, list):
                raise ValueError("rpgmaker_mv.display.terms_array")
            for index in indices:
                if index < len(values) and values[index] is not None:
                    if not isinstance(values[index], str):
                        raise ValueError("rpgmaker_mv.display.string")
                    rows.append((("terms", group, index), values[index], "system_term"))
        if "messages" in terms:
            if not isinstance(terms["messages"], dict):
                raise ValueError("rpgmaker_mv.display.messages_object")
            for key in sorted(TERM_MESSAGES & terms["messages"].keys()):
                value = terms["messages"][key]
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("rpgmaker_mv.display.string")
                    rows.append((("terms", "messages", key), value, "system_message"))
    return rows
