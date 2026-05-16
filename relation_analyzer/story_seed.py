from .common import Path, json, re
from .relations import collect_relation_units, iter_source_groups, pair_key


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
    by_name = {}
    for speaker, stats in speaker_stats.items():
        for candidate in stats['speaker_name_candidates']:
            key = candidate['name'].strip().lower()
            if key:
                by_name.setdefault(key, []).append(speaker)

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

        speaker_ids = _dedupe_preserve_order(by_name.get(name.lower(), []))
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


def collect_pair_source_files(units, characters, segment_size=1, source_root=None):
    relation_units, _, _, _ = collect_relation_units(units, characters)
    pair_sources = {}
    active = set(characters)
    segment_size = max(1, int(segment_size or 1))

    for group in iter_source_groups(relation_units):
        source = _source_file(group[0].get('source'), source_root) if group else ''

        for start in range(0, len(group), segment_size):
            segment = group[start:start + segment_size]
            present = []
            seen = set()
            for unit in segment:
                for char in unit.get('participants', []):
                    if char in active and char not in seen:
                        present.append(char)
                        seen.add(char)
            for left_index in range(len(present)):
                for right_index in range(left_index + 1, len(present)):
                    pair_sources.setdefault(pair_key(present[left_index], present[right_index]), set()).add(source)

        spoken = [unit for unit in group if unit.get('speaker_character') in active]
        for previous, current in zip(spoken, spoken[1:]):
            left = previous.get('speaker_character')
            right = current.get('speaker_character')
            if left and right and left != right:
                pair_sources.setdefault(pair_key(left, right), set()).add(source)

        for unit in group:
            speaker_character = unit.get('speaker_character') if unit.get('speaker_character') in active else None
            mentioned = [char for char in unit.get('mentioned_characters', []) if char in active]
            if speaker_character:
                for char in mentioned:
                    if char != speaker_character:
                        pair_sources.setdefault(pair_key(speaker_character, char), set()).add(source)
            if len(mentioned) >= 2:
                for left_index in range(len(mentioned)):
                    for right_index in range(left_index + 1, len(mentioned)):
                        pair_sources.setdefault(pair_key(mentioned[left_index], mentioned[right_index]), set()).add(source)

    return {
        key: sorted(value for value in sources if value)
        for key, sources in pair_sources.items()
    }


def _has_relation_evidence(row):
    raw_total = (
        float(row.get('co_scene_raw', 0.0))
        + float(row.get('dialogue_raw', 0.0))
        + float(row.get('mention_raw', 0.0))
    )
    return float(row.get('score', 0.0)) > 0.0 and raw_total > 0.0


def build_relation_seed(units, characters, relation_data, source_root=None):
    id_map = _character_id_map(characters)
    source_files = collect_pair_source_files(
        units,
        characters,
        relation_data.get('segment_size', 1),
        source_root,
    )
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
                    'source_files': source_files.get(key, []),
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
