from .common import Path, json, re
from .parsing import normalize_character_aliases, speaker_matches_character
from .relations import pair_key


def character_id_from_name(name):
    text = str(name or '').strip().lower()
    text = re.sub(r'[^\w]+', '_', text, flags=re.UNICODE)
    text = text.strip('_')
    return text or 'character'


def _dedupe_preserve_order(values):
    result = []
    seen = set()
    for value in values:
        text = str(value or '').strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _source_root_dir(source_root):
    if not source_root:
        return None
    root = Path(source_root)
    if root.is_file():
        root = root.parent
    try:
        return root.resolve()
    except OSError:
        return root


def _source_file(value, source_root=None):
    text = str(value or '').strip()
    if not text:
        return ''
    root = _source_root_dir(source_root)
    if root:
        try:
            return Path(text).resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
    return text.replace('\\', '/')


def _source_files(values, source_root=None):
    normalized = []
    seen = set()
    for value in values or []:
        source = _source_file(value, source_root)
        if not source or source in seen:
            continue
        normalized.append(source)
        seen.add(source)
    return sorted(normalized)


def collect_speaker_seed_stats(units, source_root=None):
    speakers = {}
    for unit in units:
        speaker = str(unit.get('speaker') or '').strip()
        if not speaker:
            continue
        info = speakers.setdefault(
            speaker,
            {
                'speaker_id': speaker,
                'speaker_name_candidates': {},
                'count': 0,
                'source_files': set(),
                'line_numbers': [],
            },
        )
        info['count'] += 1
        source = _source_file(unit.get('source'), source_root)
        if source:
            info['source_files'].add(source)
        line_no = unit.get('line_no')
        if isinstance(line_no, int):
            info['line_numbers'].append(line_no)
        speaker_name = str(unit.get('speaker_name') or '').strip()
        if speaker_name:
            candidates = info['speaker_name_candidates']
            candidates[speaker_name] = candidates.get(speaker_name, 0) + 1

    normalized = {}
    for speaker, info in speakers.items():
        candidates = [
            {'name': name, 'count': count}
            for name, count in sorted(
                info['speaker_name_candidates'].items(),
                key=lambda item: (-item[1], item[0].lower()),
            )
        ]
        lines = info['line_numbers']
        normalized[speaker] = {
            'speaker_id': speaker,
            'speaker_name_candidates': candidates,
            'count': info['count'],
            'source_files': sorted(info['source_files']),
            'line_min': min(lines) if lines else None,
            'line_max': max(lines) if lines else None,
        }
    return normalized


def build_character_seed(units, characters, source_root=None):
    speaker_stats = collect_speaker_seed_stats(units, source_root)
    alias_map = normalize_character_aliases(characters)
    speakers_by_character = {character: [] for character in characters}
    seen_by_character = {character: set() for character in characters}

    for unit in units:
        speaker = str(unit.get('speaker') or '').strip()
        if not speaker or speaker not in speaker_stats:
            continue
        for character in characters:
            if not speaker_matches_character(unit, alias_map[character]):
                continue
            if speaker not in seen_by_character[character]:
                speakers_by_character[character].append(speaker)
                seen_by_character[character].add(speaker)
            break

    character_entries = {}
    used_ids = set()
    for character in characters:
        name = str(character or '').strip()
        if not name:
            continue
        base_id = character_id_from_name(name)
        char_id = base_id
        suffix = 2
        while char_id in used_ids:
            char_id = f'{base_id}_{suffix}'
            suffix += 1
        used_ids.add(char_id)

        speaker_ids = _dedupe_preserve_order(speakers_by_character.get(character, []))
        candidate_details = []
        source_files = set()
        speaker_count = 0
        for speaker_id in speaker_ids:
            stats = speaker_stats[speaker_id]
            speaker_count += stats['count']
            source_files.update(stats['source_files'])
            candidate_details.append(
                {
                    'speaker_id': speaker_id,
                    'speaker_name_candidates': stats['speaker_name_candidates'],
                    'count': stats['count'],
                    'source_files': stats['source_files'],
                    'line_min': stats['line_min'],
                    'line_max': stats['line_max'],
                }
            )

        entry = {
            'name': name,
            'speaker_ids': speaker_ids,
            'speaker_name_candidates': candidate_details,
            'seed_stats': {
                'speaker_count': speaker_count,
                'source_files': sorted(source_files),
                'needs_human_review': True,
            },
        }
        character_entries[char_id] = entry

    return character_entries


def _character_id_map(characters):
    mapping = {}
    used_ids = set()
    for character in characters:
        name = str(character or '').strip()
        if not name:
            continue
        base_id = character_id_from_name(name)
        char_id = base_id
        suffix = 2
        while char_id in used_ids:
            char_id = f'{base_id}_{suffix}'
            suffix += 1
        used_ids.add(char_id)
        mapping[name] = char_id
    return mapping


def _has_relation_evidence(row):
    raw_total = (
        float(row.get('co_scene_raw', 0.0))
        + float(row.get('dialogue_raw', 0.0))
        + float(row.get('mention_raw', 0.0))
    )
    return float(row.get('score', 0.0)) > 0.0 and raw_total > 0.0


def build_relation_seed(units, characters, relation_data, source_root=None):
    id_map = _character_id_map(characters)
    pair_source_files = relation_data.get('pair_source_files', {})
    relation_entries = []
    for row in relation_data.get('pair_rows', []):
        if not _has_relation_evidence(row):
            continue
        left_name = row['left']
        right_name = row['right']
        left_id = id_map.get(left_name, character_id_from_name(left_name))
        right_id = id_map.get(right_name, character_id_from_name(right_name))
        key = pair_key(left_name, right_name)
        relation_entries.append(
            {
                'left': left_id,
                'right': right_id,
                'type': 'candidate',
                'confidence': round(max(0.0, min(float(row.get('score', 0.0)), 1.0)), 4),
                'note': '候选关系；请人工确认具体语义类型后再用于正式 story_graph.json。',
                'seed_stats': {
                    'left_name': left_name,
                    'right_name': right_name,
                    'score': round(float(row.get('score', 0.0)), 6),
                    'co_scene_raw': float(row.get('co_scene_raw', 0.0)),
                    'dialogue_raw': float(row.get('dialogue_raw', 0.0)),
                    'mention_raw': float(row.get('mention_raw', 0.0)),
                    'dominant_component': row.get('dominant_component', ''),
                    'source_files': _source_files(pair_source_files.get(key, []), source_root),
                    'needs_human_review': True,
                },
            }
        )
    return relation_entries


def build_story_graph_seed(units, characters, relation_data, source_root=None):
    return {
        'schema_version': 1,
        'characters': build_character_seed(units, relation_data.get('characters', characters), source_root),
        'relations': build_relation_seed(units, relation_data.get('characters', characters), relation_data, source_root),
        'terms': [],
        'scenes': [],
    }


def write_story_graph_seed(output_path, units, characters, relation_data, source_root=None):
    seed = build_story_graph_seed(units, characters, relation_data, source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(seed, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f"🧩 已导出 Story Memory seed: {output_path}")
    return seed
