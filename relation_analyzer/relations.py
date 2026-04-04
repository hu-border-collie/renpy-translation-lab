import csv

from .common import *
from .parsing import normalize_character_aliases, build_character_matchers, speaker_matches_character, text_mentions_character

def pair_key(left, right):
    return (left, right) if left <= right else (right, left)

def iter_source_groups(units):
    current_group = []
    current_source = None

    for unit in units:
        source = unit['source']
        if current_source is None:
            current_source = source
        if source != current_source:
            if current_group:
                yield current_group
            current_group = []
            current_source = source
        current_group.append(unit)

    if current_group:
        yield current_group

def collect_relation_units(units, characters):
    alias_map = normalize_character_aliases(characters)
    matchers = build_character_matchers(alias_map)
    relation_units = []
    participation_counts = {char: 0 for char in characters}
    speaker_counts = {char: 0 for char in characters}
    mention_presence_counts = {char: 0 for char in characters}

    for char in characters:
        if any(len(alias) == 1 for alias in alias_map[char]):
            print(f"⚠️ [{char}] 含单字别名，关系模式仍会沿用严格匹配规则；若误差较大，请继续补充 CHARACTER_ALIASES。")

    for unit in units:
        speaker_character = None
        for char in characters:
            if speaker_matches_character(unit, alias_map[char]):
                speaker_character = char
                break

        mentioned_characters = []
        text_value = unit.get('text', '')
        for char in characters:
            if char == speaker_character:
                continue
            if text_mentions_character(text_value, matchers[char]):
                mentioned_characters.append(char)

        participants = []
        if speaker_character:
            participants.append(speaker_character)
        for char in mentioned_characters:
            if char not in participants:
                participants.append(char)

        for char in participants:
            participation_counts[char] += 1
            mention_presence_counts[char] += 1
        if speaker_character:
            speaker_counts[speaker_character] += 1

        relation_units.append({
            **unit,
            'speaker_character': speaker_character,
            'mentioned_characters': mentioned_characters,
            'participants': participants,
        })

    return relation_units, participation_counts, speaker_counts, mention_presence_counts

def build_density_matrix(chars, pair_counts, presence_counts):
    np = load_numpy()
    matrix = np.zeros((len(chars), len(chars)), dtype=float)
    index_map = {char: index for index, char in enumerate(chars)}

    for (left, right), raw_value in pair_counts.items():
        if left not in index_map or right not in index_map:
            continue
        denominator = (max(1, presence_counts.get(left, 0)) * max(1, presence_counts.get(right, 0))) ** 0.5
        score = float(raw_value) / denominator if denominator else 0.0
        left_index = index_map[left]
        right_index = index_map[right]
        matrix[left_index][right_index] = score
        matrix[right_index][left_index] = score

    np.fill_diagonal(matrix, 1.0)
    return matrix

def scale_off_diagonal(matrix):
    np = load_numpy()
    scaled = np.array(matrix, dtype=float, copy=True)
    if len(scaled) <= 1:
        np.fill_diagonal(scaled, 1.0)
        return scaled

    off_diagonal = [float(scaled[row][col]) for row in range(len(scaled)) for col in range(row + 1, len(scaled))]
    max_value = max(off_diagonal) if off_diagonal else 0.0
    if max_value > 0:
        for row in range(len(scaled)):
            for col in range(row + 1, len(scaled)):
                value = float(scaled[row][col]) / max_value
                scaled[row][col] = value
                scaled[col][row] = value
    else:
        for row in range(len(scaled)):
            for col in range(row + 1, len(scaled)):
                scaled[row][col] = 0.0
                scaled[col][row] = 0.0

    np.fill_diagonal(scaled, 1.0)
    return scaled

def build_relation_pair_rows(total_matrix, chars, component_matrices, component_raw_counts, component_labels):
    rows = []
    for left in range(len(chars)):
        for right in range(left + 1, len(chars)):
            left_name = chars[left]
            right_name = chars[right]
            key = pair_key(left_name, right_name)
            component_scores = {name: float(matrix[left][right]) for name, matrix in component_matrices.items()}
            dominant_component_key = max(component_scores, key=component_scores.get)
            rows.append({
                'left': left_name,
                'right': right_name,
                'score': float(total_matrix[left][right]),
                'co_scene': component_scores['co_scene'],
                'dialogue': component_scores['dialogue'],
                'mention': component_scores['mention'],
                'co_scene_raw': float(component_raw_counts['co_scene'].get(key, 0.0)),
                'dialogue_raw': float(component_raw_counts['dialogue'].get(key, 0.0)),
                'mention_raw': float(component_raw_counts['mention'].get(key, 0.0)),
                'dominant_component_key': dominant_component_key,
                'dominant_component': component_labels[dominant_component_key],
                'left_index': left,
                'right_index': right,
            })
    rows.sort(key=lambda item: item['score'], reverse=True)
    return rows

def compute_relation_data(units, characters, segment_size):
    np = load_numpy()
    relation_units, participation_counts, speaker_counts, mention_presence_counts = collect_relation_units(units, characters)
    active_characters = [char for char in characters if participation_counts.get(char, 0) > 0]

    for char in characters:
        if char not in active_characters:
            print(f"⚠️ 未发现 [{char}] 的有效出场或提及，已从关系图中跳过。")

    if len(active_characters) < 2:
        return {
            'characters': active_characters,
            'presence_counts': {char: participation_counts.get(char, 0) for char in active_characters},
            'pair_rows': [],
            'total_matrix': np.zeros((len(active_characters), len(active_characters)), dtype=float),
        }

    active_set = set(active_characters)
    scene_presence_counts = {char: 0 for char in active_characters}
    co_scene_raw = {}
    dialogue_raw = {}
    mention_raw = {}
    segment_size = max(1, int(segment_size))

    for group in iter_source_groups(relation_units):
        for start in range(0, len(group), segment_size):
            segment = group[start:start + segment_size]
            present = []
            seen = set()
            for unit in segment:
                for char in unit['participants']:
                    if char in active_set and char not in seen:
                        present.append(char)
                        seen.add(char)
            if not present:
                continue
            for char in present:
                scene_presence_counts[char] += 1
            for left in range(len(present)):
                for right in range(left + 1, len(present)):
                    key = pair_key(present[left], present[right])
                    co_scene_raw[key] = co_scene_raw.get(key, 0.0) + 1.0

        spoken = [unit for unit in group if unit['speaker_character'] in active_set]
        for previous, current in zip(spoken, spoken[1:]):
            left = previous['speaker_character']
            right = current['speaker_character']
            if left == right:
                continue
            key = pair_key(left, right)
            dialogue_raw[key] = dialogue_raw.get(key, 0.0) + 1.0

        for unit in group:
            speaker_character = unit['speaker_character'] if unit['speaker_character'] in active_set else None
            mentioned = [char for char in unit['mentioned_characters'] if char in active_set]
            if speaker_character:
                for char in mentioned:
                    if char == speaker_character:
                        continue
                    key = pair_key(speaker_character, char)
                    mention_raw[key] = mention_raw.get(key, 0.0) + 1.0
            if len(mentioned) >= 2:
                bonus = 0.35 if speaker_character else 0.55
                for left in range(len(mentioned)):
                    for right in range(left + 1, len(mentioned)):
                        key = pair_key(mentioned[left], mentioned[right])
                        mention_raw[key] = mention_raw.get(key, 0.0) + bonus

    component_labels = {
        'co_scene': '共场景',
        'dialogue': '对话往来',
        'mention': '相互提及',
    }
    component_colors = {
        'co_scene': '#4e79a7',
        'dialogue': '#f28e2b',
        'mention': '#59a14f',
    }
    component_weights = {
        'co_scene': 0.45,
        'dialogue': 0.35,
        'mention': 0.20,
    }

    component_raw_counts = {
        'co_scene': co_scene_raw,
        'dialogue': dialogue_raw,
        'mention': mention_raw,
    }
    component_density_matrices = {
        'co_scene': build_density_matrix(active_characters, co_scene_raw, scene_presence_counts),
        'dialogue': build_density_matrix(active_characters, dialogue_raw, speaker_counts),
        'mention': build_density_matrix(active_characters, mention_raw, mention_presence_counts),
    }
    component_scaled_matrices = {
        name: scale_off_diagonal(matrix)
        for name, matrix in component_density_matrices.items()
    }

    total_matrix = np.zeros((len(active_characters), len(active_characters)), dtype=float)
    for component_name, weight in component_weights.items():
        total_matrix += component_scaled_matrices[component_name] * weight
    np.fill_diagonal(total_matrix, 1.0)

    pair_rows = build_relation_pair_rows(
        total_matrix,
        active_characters,
        component_scaled_matrices,
        component_raw_counts,
        component_labels,
    )

    return {
        'characters': active_characters,
        'presence_counts': {char: participation_counts.get(char, 0) for char in active_characters},
        'speaker_counts': {char: speaker_counts.get(char, 0) for char in active_characters},
        'scene_presence_counts': {char: scene_presence_counts.get(char, 0) for char in active_characters},
        'component_labels': component_labels,
        'component_colors': component_colors,
        'component_weights': component_weights,
        'component_raw_counts': component_raw_counts,
        'component_density_matrices': component_density_matrices,
        'component_matrices': component_scaled_matrices,
        'total_matrix': total_matrix,
        'pair_rows': pair_rows,
    }

def select_relation_edges(sim_matrix, chars):
    np = load_numpy()
    pair_scores = []
    for left in range(len(chars)):
        for right in range(left + 1, len(chars)):
            score = float(sim_matrix[left][right])
            if score > 0:
                pair_scores.append((score, left, right))

    if not pair_scores:
        return []

    pair_scores.sort(reverse=True, key=lambda item: item[0])
    raw_scores = np.array([score for score, _, _ in pair_scores], dtype=float)
    adaptive_threshold = max(0.18, float(np.quantile(raw_scores, 0.65)))
    max_edges = min(max(len(chars) - 1, 3), len(pair_scores))
    selected = [item for item in pair_scores if item[0] >= adaptive_threshold][:max_edges]
    if len(selected) < min(3, len(pair_scores)):
        selected = pair_scores[:min(3, len(pair_scores))]
    return selected

def compute_force_layout(sim_matrix, iterations=280, seed=42):
    np = load_numpy()
    count = len(sim_matrix)
    if count == 0:
        return np.zeros((0, 2), dtype=float)
    if count == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if count == 2:
        return np.array([[-0.65, 0.0], [0.65, 0.0]], dtype=float)

    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    rng = np.random.default_rng(seed)
    positions = positions + rng.normal(0.0, 0.045, size=positions.shape)

    positive = np.clip(np.array(sim_matrix, dtype=float), 0.0, None)
    off_diagonal = [float(positive[row][col]) for row in range(count) for col in range(row + 1, count)]
    max_weight = max(off_diagonal) if off_diagonal else 1.0
    if max_weight <= 0:
        max_weight = 1.0
    normalized = positive / max_weight

    for _ in range(iterations):
        displacement = np.zeros_like(positions)
        for left in range(count):
            for right in range(left + 1, count):
                delta = positions[right] - positions[left]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-4:
                    delta = rng.normal(0.0, 0.01, size=2)
                    distance = float(np.linalg.norm(delta)) or 1e-4
                direction = delta / distance

                repulsion = 0.015 / max(distance * distance, 0.02)
                displacement[left] -= direction * repulsion
                displacement[right] += direction * repulsion

                weight = float(normalized[left][right])
                if weight > 0:
                    desired_distance = 1.30 - weight * 0.82
                    attraction = (distance - desired_distance) * (0.035 + weight * 0.085)
                    displacement[left] += direction * attraction
                    displacement[right] -= direction * attraction

        positions += displacement * 0.22
        positions -= positions.mean(axis=0, keepdims=True)
        radius = float(np.max(np.linalg.norm(positions, axis=1)))
        if radius > 1.45:
            positions /= radius / 1.45

    return positions

def write_relation_csv(csv_output, relation_data):
    pair_rows = relation_data['pair_rows']
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_output, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'left',
            'right',
            'total_score',
            'co_scene_score',
            'dialogue_score',
            'mention_score',
            'co_scene_raw',
            'dialogue_raw',
            'mention_raw',
            'dominant_component',
        ])
        for row in pair_rows:
            writer.writerow([
                row['left'],
                row['right'],
                f"{row['score']:.6f}",
                f"{row['co_scene']:.6f}",
                f"{row['dialogue']:.6f}",
                f"{row['mention']:.6f}",
                f"{row['co_scene_raw']:.3f}",
                f"{row['dialogue_raw']:.3f}",
                f"{row['mention_raw']:.3f}",
                row['dominant_component'],
            ])
    print(f"📝 已导出关系明细: {csv_output}")

