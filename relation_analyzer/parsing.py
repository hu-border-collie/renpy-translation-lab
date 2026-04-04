from .common import *

def iter_input_files(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            [path for path in input_path.rglob("*") if path.suffix.lower() in {".rpy", ".txt"}],
            key=lambda item: str(item).lower(),
        )
        if files:
            return files
    raise SystemExit(f"❌ 找不到可读取的输入: {input_path}")

def split_text_paragraphs(content):
    paragraphs = []
    for block in re.split(r"\n\s*\n+", content):
        merged = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if len(merged) >= 6:
            paragraphs.append(merged)
    return paragraphs

def extract_first_string_token(source_line):
    try:
        for token in tokenize.generate_tokens(io.StringIO(source_line).readline):
            if token.type == tokenize.STRING:
                return token.string, token.start[1]
    except tokenize.TokenError:
        return None, None
    return None, None

def normalize_text(text):
    text = " ".join(str(text).split()).strip()
    if not text or len(text) < 2:
        return ""
    if ONLY_SYMBOLS_RE.match(text):
        return ""
    if ASSET_FILE_RE.match(text):
        return ""
    return text

def is_valid_say_attribute_token(token):
    return (
        bool(IDENTIFIER_RE.match(token))
        and token not in CONTROL_KEYWORDS
        and token not in TEXT_COMMANDS
    )

def parse_dialogue_line(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("old ") or stripped.startswith("new "):
        return None

    literal, start_col = extract_first_string_token(stripped)
    if not literal:
        return None

    leading = stripped[:start_col].strip()
    speaker = None
    command = None
    if leading:
        parts = leading.split()
        first = parts[0]
        if first in TEXT_COMMANDS:
            command = first
        elif first in CONTROL_KEYWORDS:
            return None
        elif IDENTIFIER_RE.match(first):
            if any(not is_valid_say_attribute_token(token) for token in parts[1:]):
                return None
            speaker = first
        else:
            return None

    try:
        text = normalize_text(ast.literal_eval(literal))
    except (SyntaxError, ValueError):
        return None

    if not text:
        return None

    return {"speaker": speaker, "text": text, "command": command}

def apply_extend_speaker(parsed, last_speaker):
    speaker = parsed["speaker"]
    if speaker is None and parsed.get("command") == "extend" and last_speaker:
        speaker = last_speaker
    return {
        "speaker": speaker,
        "speaker_name": resolve_speaker_name(speaker),
        "text": parsed["text"],
    }

def extract_units_from_translation_file(lines, file_path):
    units = []
    in_block = False
    in_strings_block = False
    last_speaker = None

    for line_no, line in enumerate(lines, start=1):
        match = TRANSLATE_BLOCK_RE.match(line)
        if match:
            in_block = True
            in_strings_block = match.group("label").strip() == "strings"
            last_speaker = None
            continue

        if not in_block or in_strings_block:
            continue

        parsed = parse_dialogue_line(line)
        if not parsed:
            continue
        normalized = apply_extend_speaker(parsed, last_speaker)
        if normalized["speaker"]:
            last_speaker = normalized["speaker"]

        units.append(
            {
                "source": str(file_path),
                "line_no": line_no,
                "speaker": normalized["speaker"],
                "speaker_name": normalized["speaker_name"],
                "text": normalized["text"],
            }
        )

    return units

def extract_units_from_raw_rpy(lines, file_path):
    units = []
    last_speaker = None
    for line_no, line in enumerate(lines, start=1):
        parsed = parse_dialogue_line(line)
        if not parsed:
            continue
        normalized = apply_extend_speaker(parsed, last_speaker)
        if normalized["speaker"]:
            last_speaker = normalized["speaker"]
        units.append(
            {
                "source": str(file_path),
                "line_no": line_no,
                "speaker": normalized["speaker"],
                "speaker_name": normalized["speaker_name"],
                "text": normalized["text"],
            }
        )
    return units

def extract_units_from_rpy(file_path):
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    if any(TRANSLATE_BLOCK_RE.match(line) for line in lines):
        units = extract_units_from_translation_file(lines, file_path)
        if units:
            return units
    return extract_units_from_raw_rpy(lines, file_path)

def load_text_units(input_path, context_window):
    units = []
    files = iter_input_files(input_path)

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            content = file_path.read_text(encoding="utf-8-sig")
            paragraphs = split_text_paragraphs(content)
            for index, paragraph in enumerate(paragraphs, start=1):
                units.append(
                    {
                        "source": str(file_path),
                        "line_no": index,
                        "speaker": None,
                        "speaker_name": None,
                        "text": paragraph,
                    }
                )
            continue

        if suffix == ".rpy":
            units.extend(extract_units_from_rpy(file_path))

    if not units:
        raise SystemExit("❌ 没有抽取到可分析的剧情文本。")

    return build_contextual_units(units, context_window)

def build_contextual_units(units, context_window):
    if context_window <= 0:
        return [{**unit, "context": format_unit(unit)} for unit in units]

    contextual_units = []
    current_group = []
    current_source = None

    def flush_group(group):
        for index, unit in enumerate(group):
            start = max(0, index - context_window)
            end = min(len(group), index + context_window + 1)
            context = "\n".join(format_unit(item) for item in group[start:end])
            contextual_units.append({**unit, "context": context})

    for unit in units:
        if current_source is None:
            current_source = unit["source"]
        if unit["source"] != current_source:
            flush_group(current_group)
            current_group = []
            current_source = unit["source"]
        current_group.append(unit)

    if current_group:
        flush_group(current_group)

    return contextual_units

def format_unit(unit):
    if unit["speaker_name"]:
        return f"{unit['speaker_name']}: {unit['text']}"
    return unit["text"]

SPEAKER_STYLE_SUFFIX_GROUPS = {
    ('no', 'side'),
}
SPEAKER_STYLE_SUFFIX_TOKENS = {
    'alt', 'angry', 'annoyed', 'big', 'blank', 'blush', 'cg', 'curious', 'day', 'extra',
    'happy', 'mad', 'neutral', 'night', 'open', 'portrait', 'sad', 'shadow', 'shock',
    'shout', 'side', 'small', 'smile', 'smiling', 'sprite', 'thinking', 'tiny', 'upset',
    'whisper', 'worry', 'worried',
}


def guess_character_name_from_speaker(speaker):
    if not speaker:
        return None
    raw_value = str(speaker).strip()
    if not raw_value:
        return None

    parts = [part for part in raw_value.split('_') if part]
    if not parts:
        return raw_value

    lowered = [part.lower() for part in parts]
    changed = False
    while len(lowered) >= 2 and tuple(lowered[-2:]) in SPEAKER_STYLE_SUFFIX_GROUPS:
        lowered = lowered[:-2]
        parts = parts[:-2]
        changed = True
    while len(lowered) > 1 and lowered[-1] in SPEAKER_STYLE_SUFFIX_TOKENS:
        lowered = lowered[:-1]
        parts = parts[:-1]
        changed = True

    if not parts:
        parts = [raw_value]

    if len(parts) == 1:
        part = parts[0]
        if part.isascii() and part.replace('-', '').isalnum():
            return part.capitalize()
        return part

    if changed or all(part.isascii() for part in parts):
        return ' '.join(part.capitalize() for part in parts)
    return raw_value


def resolve_speaker_name(speaker):
    if not speaker:
        return None
    mapped = SPEAKER_TO_CHARACTER.get(speaker)
    if mapped:
        return mapped
    return guess_character_name_from_speaker(speaker)


def infer_characters_from_units(units, limit):
    if not limit or limit <= 0:
        return []
    counts = {}
    for unit in units:
        speaker_name = unit.get('speaker_name')
        if not speaker_name:
            continue
        normalized = speaker_name.strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return [name for name, _ in ordered[:limit]]


def normalize_character_aliases(characters):
    alias_map = {}
    for char in characters:
        aliases = [char]
        aliases.extend(CHARACTER_ALIASES.get(char, []))
        deduped = []
        seen = set()
        for alias in aliases:
            normalized = alias.strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
        alias_map[char] = deduped
    return alias_map

def compile_alias_patterns(alias):
    escaped = re.escape(alias)
    if ASCII_NAME_RE.match(alias):
        return [re.compile(rf"\b{escaped}\b", re.IGNORECASE)]
    if len(alias) == 1:
        return [
            re.compile(rf"(?<![A-Za-z0-9_\u4e00-\u9fff]){escaped}(?![A-Za-z0-9_\u4e00-\u9fff])"),
            re.compile(rf"(?<=[{SINGLE_CHAR_PRECEDERS}]){escaped}(?![A-Za-z0-9_\u4e00-\u9fff])"),
            re.compile(rf"(?<![A-Za-z0-9_\u4e00-\u9fff]){escaped}(?=[{SINGLE_CHAR_FOLLOWERS}])"),
        ]
    return [re.compile(escaped)]

def build_character_matchers(alias_map):
    return {
        char: [pattern for alias in aliases for pattern in compile_alias_patterns(alias)]
        for char, aliases in alias_map.items()
    }

def speaker_matches_character(unit, aliases):
    speaker_name = unit.get("speaker_name")
    if not speaker_name:
        return False
    speaker_name_lower = speaker_name.lower()
    for alias in aliases:
        if speaker_name_lower == alias.lower():
            return True
    return False

def text_mentions_character(text, patterns):
    return any(pattern.search(text) for pattern in patterns)

def collect_character_texts(units, characters):
    alias_map = normalize_character_aliases(characters)
    matchers = build_character_matchers(alias_map)
    char_texts = {char: [] for char in characters}
    seen_texts = {char: set() for char in characters}

    for char in characters:
        if any(len(alias) == 1 for alias in alias_map[char]):
            print(f"⚠️ [{char}] 含单字别名，已启用严格匹配规则；若误差仍大，请补充 CHARACTER_ALIASES。")

    for unit in units:
        context = unit["context"]
        for char in characters:
            if not speaker_matches_character(unit, alias_map[char]) and not text_mentions_character(context, matchers[char]):
                continue
            if context in seen_texts[char]:
                continue
            char_texts[char].append(context)
            seen_texts[char].add(context)

    return char_texts

