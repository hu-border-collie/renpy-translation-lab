from .common import *
from .relations import compute_force_layout, pair_key, select_relation_edges, write_relation_csv

def select_similarity_edges(sim_matrix, chars):
    np = load_numpy()
    pair_scores = []
    for left in range(len(chars)):
        for right in range(left + 1, len(chars)):
            pair_scores.append((float(sim_matrix[left][right]), left, right))

    if not pair_scores:
        return []

    pair_scores.sort(reverse=True, key=lambda item: item[0])
    raw_scores = np.array([score for score, _, _ in pair_scores], dtype=float)
    adaptive_threshold = max(0.55, float(np.quantile(raw_scores, 0.7)))
    max_edges = min(max(len(chars) - 1, 3), len(pair_scores))
    selected = [item for item in pair_scores if item[0] >= adaptive_threshold][:max_edges]
    if len(selected) < min(3, len(pair_scores)):
        selected = pair_scores[:min(3, len(pair_scores))]
    return selected

def place_labels(vectors_2d, labels):
    np = load_numpy()
    x_values = vectors_2d[:, 0]
    y_values = vectors_2d[:, 1]
    x_range = float(x_values.max() - x_values.min()) if len(x_values) > 1 else 1.0
    y_range = float(y_values.max() - y_values.min()) if len(y_values) > 1 else 1.0
    x_range = x_range or 1.0
    y_range = y_range or 1.0

    margin_x = max(x_range * 0.18, 0.04)
    margin_y = max(y_range * 0.18, 0.04)
    label_step_x = max(x_range * 0.065, 0.015)
    label_step_y = max(y_range * 0.065, 0.015)

    x_min = float(x_values.min()) - margin_x
    x_max = float(x_values.max()) + margin_x
    y_min = float(y_values.min()) - margin_y
    y_max = float(y_values.max()) + margin_y

    placements = [None] * len(labels)
    boxes = []
    candidate_directions = [
        (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
        (1.2, 0.2), (-1.2, 0.2), (1.2, -0.2), (-1.2, -0.2),
        (0.2, 1.2), (-0.2, 1.2), (0.2, -1.2), (-0.2, -1.2),
        (0.0, 1.3), (0.0, -1.3), (1.3, 0.0), (-1.3, 0.0),
    ]
    radius_scales = (1.0, 1.45, 1.9, 2.35, 2.8)

    center_x = float(np.mean(x_values))
    center_y = float(np.mean(y_values))
    order = sorted(
        range(len(labels)),
        key=lambda index: (vectors_2d[index, 0] - center_x) ** 2 + (vectors_2d[index, 1] - center_y) ** 2,
    )

    for index in order:
        char = labels[index]
        point_x = float(vectors_2d[index, 0])
        point_y = float(vectors_2d[index, 1])
        label_width = max(x_range * (0.03 * max(len(char), 4)), 0.028)
        label_height = max(y_range * 0.095, 0.022)
        best_candidate = None
        best_score = None

        for radius in radius_scales:
            for dx, dy in candidate_directions:
                label_x = point_x + dx * label_step_x * radius
                label_y = point_y + dy * label_step_y * radius

                if dx > 0.2:
                    x0, x1 = label_x, label_x + label_width
                    ha = 'left'
                elif dx < -0.2:
                    x0, x1 = label_x - label_width, label_x
                    ha = 'right'
                else:
                    x0, x1 = label_x - label_width / 2, label_x + label_width / 2
                    ha = 'center'

                if dy > 0.2:
                    y0, y1 = label_y, label_y + label_height
                    va = 'bottom'
                elif dy < -0.2:
                    y0, y1 = label_y - label_height, label_y
                    va = 'top'
                else:
                    y0, y1 = label_y - label_height / 2, label_y + label_height / 2
                    va = 'center'

                if x0 < x_min or x1 > x_max or y0 < y_min or y1 > y_max:
                    continue

                overlap_penalty = 0.0
                for box in boxes:
                    overlap_x = max(0.0, min(x1, box[1]) - max(x0, box[0]))
                    overlap_y = max(0.0, min(y1, box[3]) - max(y0, box[2]))
                    overlap_penalty += overlap_x * overlap_y

                distance_penalty = ((label_x - point_x) / label_step_x) ** 2 + ((label_y - point_y) / label_step_y) ** 2
                score = overlap_penalty * 1000 + distance_penalty

                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = {
                        'x': label_x,
                        'y': label_y,
                        'ha': ha,
                        'va': va,
                        'box': (x0, x1, y0, y1),
                    }

        if best_candidate is None:
            fallback_x = min(max(point_x + label_step_x, x_min + label_width / 2), x_max - label_width / 2)
            fallback_y = min(max(point_y + label_step_y, y_min + label_height / 2), y_max - label_height / 2)
            best_candidate = {
                'x': fallback_x,
                'y': fallback_y,
                'ha': 'left',
                'va': 'bottom',
                'box': (fallback_x, fallback_x + label_width, fallback_y, fallback_y + label_height),
            }

        boxes.append(best_candidate['box'])
        placements[index] = best_candidate

    return placements, (x_min, x_max, y_min, y_max)

def build_pair_rows(sim_matrix, chars):
    rows = []
    for left in range(len(chars)):
        for right in range(left + 1, len(chars)):
            rows.append({
                'left': chars[left],
                'right': chars[right],
                'score': float(sim_matrix[left][right]),
                'left_index': left,
                'right_index': right,
            })
    rows.sort(key=lambda item: item['score'], reverse=True)
    return rows

def project_vectors_2d(matrix, PCA):
    np = load_numpy()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('project_vectors_2d 需要二维矩阵输入。')

    sample_count, feature_count = matrix.shape
    if sample_count == 0:
        return np.zeros((0, 2), dtype=float)

    n_components = min(2, sample_count, feature_count)
    if n_components <= 0:
        return np.zeros((sample_count, 2), dtype=float)

    vectors_2d = PCA(n_components=n_components).fit_transform(matrix)
    if vectors_2d.shape[1] < 2:
        padding = np.zeros((sample_count, 2 - vectors_2d.shape[1]), dtype=float)
        vectors_2d = np.hstack([vectors_2d, padding])
    return vectors_2d

def analyze_and_plot(char_vectors, char_texts, output_path, portraits=None):
    np = load_numpy()
    plt, PCA, cosine_similarity, OffsetImage, AnnotationBbox = load_plot_libs()
    chars = list(char_vectors.keys())
    portraits = portraits or {}
    matrix = np.array([char_vectors[char] for char in chars], dtype=float)
    counts = {char: len(char_texts.get(char, [])) for char in chars}

    raw_sim_matrix = cosine_similarity(matrix)
    centered_matrix = matrix - matrix.mean(axis=0, keepdims=True)
    relative_sim_matrix = cosine_similarity(centered_matrix)
    raw_pair_rows = build_pair_rows(raw_sim_matrix, chars)
    relative_pair_rows = build_pair_rows(relative_sim_matrix, chars)

    print("\nRaw Similarity Matrix (Cosine Similarity):")
    print("-" * 50)
    name_width = max(8, max(len(char) for char in chars) + 2)
    print(f"{'':<{name_width}}", end="")
    for char in chars:
        print(f"{char:<{name_width}}", end="")
    print()
    for row_index, char in enumerate(chars):
        print(f"{char:<{name_width}}", end="")
        for col_index in range(len(chars)):
            print(f"{raw_sim_matrix[row_index][col_index]:<{name_width}.3f}", end="")
        print()
    print("-" * 50)

    print("\nRelative Similarity Matrix (Centered Cosine Similarity):")
    print("-" * 50)
    print(f"{'':<{name_width}}", end="")
    for char in chars:
        print(f"{char:<{name_width}}", end="")
    print()
    for row_index, char in enumerate(chars):
        print(f"{char:<{name_width}}", end="")
        for col_index in range(len(chars)):
            print(f"{relative_sim_matrix[row_index][col_index]:<{name_width}.3f}", end="")
        print()
    print("-" * 50)

    print("\nTop Raw Similar Pairs:")
    for index, row in enumerate(raw_pair_rows[:min(8, len(raw_pair_rows))], start=1):
        print(f"{index}. {row['left']} <-> {row['right']}: {row['score']:.3f}")

    print("\nTop Relative Similar Pairs:")
    for index, row in enumerate(relative_pair_rows[:min(8, len(relative_pair_rows))], start=1):
        print(f"{index}. {row['left']} <-> {row['right']}: {row['score']:.3f}")

    raw_vectors_2d = project_vectors_2d(matrix, PCA)
    relative_vectors_2d = project_vectors_2d(centered_matrix, PCA)
    raw_edges = select_similarity_edges(raw_sim_matrix, chars)
    relative_edges = select_similarity_edges(relative_sim_matrix, chars)
    map_labels = [f"{char}\n{counts[char]}" for char in chars]
    raw_placements, raw_bounds = place_labels(raw_vectors_2d, map_labels)
    relative_placements, relative_bounds = place_labels(relative_vectors_2d, map_labels)

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.22, wspace=0.18)
    ax_raw_heatmap = fig.add_subplot(grid[0, 0])
    ax_raw_map = fig.add_subplot(grid[0, 1])
    ax_relative_heatmap = fig.add_subplot(grid[1, 0])
    ax_relative_map = fig.add_subplot(grid[1, 1])

    heatmap_labels = [f"{char}\n({counts[char]})" for char in chars]

    def draw_heatmap(ax, sim_matrix, title, vmin, vmax):
        heatmap = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(chars)))
        ax.set_yticks(range(len(chars)))
        ax.set_xticklabels(heatmap_labels, rotation=35, ha='right', fontsize=10)
        ax.set_yticklabels(heatmap_labels, fontsize=10)
        ax.set_title(title, fontsize=13, pad=10)
        midpoint = (vmin + vmax) / 2
        for row_index in range(len(chars)):
            for col_index in range(len(chars)):
                value = float(sim_matrix[row_index][col_index])
                text_color = 'white' if abs(value - midpoint) > (vmax - vmin) * 0.28 else '#1f2933'
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha='center',
                    va='center',
                    fontsize=9,
                    color=text_color,
                    weight='bold' if row_index == col_index else 'normal',
                )
        colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label('Cosine similarity')

    def draw_map(ax, vectors_2d, sim_matrix, pair_rows, edges, placements, bounds, title, relative=False):
        x_min, x_max, y_min, y_max = bounds
        ax.set_axisbelow(True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.grid(True, linestyle='--', alpha=0.25)

        count_values = np.array([counts[char] for char in chars], dtype=float)
        count_min = float(count_values.min()) if len(count_values) else 0.0
        count_max = float(count_values.max()) if len(count_values) else 1.0
        count_span = (count_max - count_min) or 1.0
        node_sizes = 420 + ((count_values - count_min) / count_span) * 1200
        avatar_zoom = 0.17 + ((count_values - count_min) / count_span) * 0.08
        axis_scale = max(x_max - x_min, y_max - y_min)
        label_offset = max(axis_scale * 0.018, 0.008)

        for score, left, right in edges:
            x0, y0 = vectors_2d[left]
            x1, y1 = vectors_2d[right]
            if relative:
                intensity = min(1.0, max(0.0, (score + 0.2) / 0.9))
                line_alpha = 0.18 + intensity * 0.65
                line_width = 1.2 + max(0.0, score) * 3.8
                line_color = '#c0392b' if score < 0 else '#2471a3'
            else:
                line_alpha = min(0.85, max(0.25, (score - 0.40) / 0.50))
                line_width = 1.4 + max(0.0, score - 0.45) * 4.0
                line_color = '#4b6584' if score >= 0.75 else '#95a5a6'
            ax.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, alpha=line_alpha, zorder=1)

            delta_x = float(x1 - x0)
            delta_y = float(y1 - y0)
            norm = (delta_x ** 2 + delta_y ** 2) ** 0.5 or 1.0
            normal_x = -delta_y / norm
            normal_y = delta_x / norm
            mid_x = (x0 + x1) / 2 + normal_x * label_offset
            mid_y = (y0 + y1) / 2 + normal_y * label_offset
            ax.text(
                mid_x,
                mid_y,
                f'{score:.2f}',
                fontsize=9,
                color='#2f3640',
                ha='center',
                va='center',
                bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88},
                zorder=2,
            )

        ax.scatter(
            vectors_2d[:, 0],
            vectors_2d[:, 1],
            c='#d6eaf8',
            s=node_sizes,
            alpha=0.65,
            edgecolors='#5dade2',
            linewidths=1.2,
            zorder=3,
        )

        for index, char in enumerate(chars):
            point_x, point_y = vectors_2d[index]
            portrait = portraits.get(char)
            if portrait is not None:
                image_box = OffsetImage(portrait, zoom=float(avatar_zoom[index]))
                annotation_box = AnnotationBbox(
                    image_box,
                    (point_x, point_y),
                    frameon=True,
                    pad=0.16,
                    bboxprops={'edgecolor': '#1b2631', 'facecolor': 'white', 'lw': 1.2},
                    zorder=4,
                )
                ax.add_artist(annotation_box)
            else:
                ax.scatter(
                    [point_x],
                    [point_y],
                    c='#2e86de',
                    s=float(node_sizes[index]) * 0.42,
                    alpha=0.95,
                    edgecolors='#1b2631',
                    linewidths=1.1,
                    zorder=4,
                )

        for index, label in enumerate(map_labels):
            point_x, point_y = vectors_2d[index]
            placement = placements[index]
            ax.annotate(
                label,
                xy=(point_x, point_y),
                xytext=(placement['x'], placement['y']),
                textcoords='data',
                fontsize=11,
                weight='bold',
                ha=placement['ha'],
                va=placement['va'],
                bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#dfe6e9', 'alpha': 0.96},
                arrowprops={'arrowstyle': '-', 'color': '#95a5a6', 'lw': 1.0, 'alpha': 0.85},
                zorder=5,
            )

        summary_title = '\u76f8\u4f3c\u7ec4\u5408 Top 5' if not relative else '\u76f8\u5bf9\u63a5\u8fd1\u7ec4\u5408 Top 5'
        summary_lines = [summary_title]
        for index, row in enumerate(pair_rows[:min(5, len(pair_rows))], start=1):
            summary_lines.append(f"{index}. {row['left']} - {row['right']}  {row['score']:.2f}")
        summary_lines.append('')
        summary_lines.append('\u8fde\u7ebf = \u9ad8\u76f8\u4f3c\u5ea6')
        summary_lines.append('\u5916\u5708 = \u6587\u672c\u91cf')
        summary_lines.append('\u5934\u50cf = \u89d2\u8272\u5f62\u8c61')
        ax.text(
            0.02,
            0.02,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=10,
            color='#2f3640',
            bbox={'boxstyle': 'round,pad=0.4', 'facecolor': 'white', 'edgecolor': '#ced6e0', 'alpha': 0.95},
            zorder=6,
        )

    draw_heatmap(
        ax_raw_heatmap,
        raw_sim_matrix,
        '\u539f\u59cb\u76f8\u4f3c\u5ea6\u70ed\u529b\u56fe',
        vmin=min(-0.2, float(raw_sim_matrix.min())),
        vmax=1.0,
    )
    draw_heatmap(
        ax_relative_heatmap,
        relative_sim_matrix,
        '\u76f8\u5bf9\u76f8\u4f3c\u5ea6\u70ed\u529b\u56fe\uff08\u53bb\u516c\u5171\u5747\u503c\uff09',
        vmin=-1.0,
        vmax=1.0,
    )
    draw_map(
        ax_raw_map,
        raw_vectors_2d,
        raw_sim_matrix,
        raw_pair_rows,
        raw_edges,
        raw_placements,
        raw_bounds,
        '\u539f\u59cb\u8bed\u4e49\u5173\u7cfb\u56fe',
        relative=False,
    )
    draw_map(
        ax_relative_map,
        relative_vectors_2d,
        relative_sim_matrix,
        relative_pair_rows,
        relative_edges,
        relative_placements,
        relative_bounds,
        '\u76f8\u5bf9\u8bed\u4e49\u5173\u7cfb\u56fe\uff08\u53bb\u516c\u5171\u5747\u503c\uff09',
        relative=True,
    )

    fig.suptitle('\u5267\u672c\u89d2\u8272\u8bed\u4e49\u5173\u7cfb\u603b\u89c8', fontsize=18, y=0.985)
    plt.figtext(
        0.5,
        0.012,
        '\u4e0a\u6392\u770b\u539f\u59cb\u76f8\u4f3c\u5ea6\uff0c\u4e0b\u6392\u770b\u53bb\u6389\u6574\u4f53\u8bed\u6599\u516c\u5171\u6210\u5206\u540e\u7684\u76f8\u5bf9\u5dee\u5f02\u3002',
        ha='center',
        fontsize=10,
        color='#576574',
    )
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.94, wspace=0.18, hspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {output_path}")

def analyze_and_plot_relation(relation_data, output_path, csv_output, portraits=None):
    np = load_numpy()
    plt, _, _, OffsetImage, AnnotationBbox = load_plot_libs()
    chars = relation_data['characters']
    portraits = portraits or {}
    counts = relation_data['presence_counts']
    total_matrix = relation_data['total_matrix']
    pair_rows = relation_data['pair_rows']
    component_labels = relation_data['component_labels']
    component_colors = relation_data['component_colors']
    component_weights = relation_data['component_weights']

    print("\nRelation Matrix (Weighted Score):")
    print("-" * 50)
    name_width = max(8, max(len(char) for char in chars) + 2)
    print(f"{'':<{name_width}}", end="")
    for char in chars:
        print(f"{char:<{name_width}}", end="")
    print()
    for row_index, char in enumerate(chars):
        print(f"{char:<{name_width}}", end="")
        for col_index in range(len(chars)):
            print(f"{total_matrix[row_index][col_index]:<{name_width}.3f}", end="")
        print()
    print("-" * 50)

    print("\nTop Related Pairs:")
    for index, row in enumerate(pair_rows[:min(10, len(pair_rows))], start=1):
        print(
            f"{index}. {row['left']} <-> {row['right']}: 总分 {row['score']:.3f} | 共场景 {row['co_scene']:.3f} | 对话 {row['dialogue']:.3f} | 提及 {row['mention']:.3f}"
        )

    write_relation_csv(csv_output, relation_data)

    vectors_2d = compute_force_layout(total_matrix)
    edges = select_relation_edges(total_matrix, chars)
    map_labels = [f"{char}\n{counts[char]}" for char in chars]
    placements, bounds = place_labels(vectors_2d, map_labels)
    pair_lookup = {pair_key(row['left'], row['right']): row for row in pair_rows}

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(17, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.16)
    ax_heatmap = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1])

    heatmap = ax_heatmap.imshow(total_matrix, cmap='YlOrRd', vmin=0.0, vmax=1.0)
    heatmap_labels = [f"{char}\n({counts[char]})" for char in chars]
    ax_heatmap.set_xticks(range(len(chars)))
    ax_heatmap.set_yticks(range(len(chars)))
    ax_heatmap.set_xticklabels(heatmap_labels, rotation=35, ha='right', fontsize=10)
    ax_heatmap.set_yticklabels(heatmap_labels, fontsize=10)
    ax_heatmap.set_title('人物关系热力图', fontsize=14, pad=10)
    for row_index in range(len(chars)):
        for col_index in range(len(chars)):
            value = float(total_matrix[row_index][col_index])
            text_color = 'white' if value >= 0.62 else '#1f2933'
            ax_heatmap.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha='center',
                va='center',
                fontsize=9,
                color=text_color,
                weight='bold' if row_index == col_index else 'normal',
            )
    colorbar = fig.colorbar(heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04)
    colorbar.set_label('关系强度')

    x_min, x_max, y_min, y_max = bounds
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.grid(False)
    ax_map.set_title('人物关系网络图', fontsize=14, pad=10)

    count_values = np.array([counts[char] for char in chars], dtype=float)
    count_min = float(count_values.min()) if len(count_values) else 0.0
    count_max = float(count_values.max()) if len(count_values) else 1.0
    count_span = (count_max - count_min) or 1.0
    node_sizes = 420 + ((count_values - count_min) / count_span) * 1200
    avatar_zoom = 0.17 + ((count_values - count_min) / count_span) * 0.08
    axis_scale = max(x_max - x_min, y_max - y_min)
    label_offset = max(axis_scale * 0.018, 0.008)

    for score, left_index, right_index in sorted(edges, key=lambda item: item[0]):
        left_name = chars[left_index]
        right_name = chars[right_index]
        row = pair_lookup[pair_key(left_name, right_name)]
        line_color = component_colors[row['dominant_component_key']]
        x0, y0 = vectors_2d[left_index]
        x1, y1 = vectors_2d[right_index]
        line_alpha = 0.22 + score * 0.62
        line_width = 1.4 + score * 4.4
        ax_map.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, alpha=line_alpha, zorder=1)

        delta_x = float(x1 - x0)
        delta_y = float(y1 - y0)
        norm = (delta_x ** 2 + delta_y ** 2) ** 0.5 or 1.0
        normal_x = -delta_y / norm
        normal_y = delta_x / norm
        mid_x = (x0 + x1) / 2 + normal_x * label_offset
        mid_y = (y0 + y1) / 2 + normal_y * label_offset
        ax_map.text(
            mid_x,
            mid_y,
            f"{score:.2f}",
            fontsize=9,
            color='#2f3640',
            ha='center',
            va='center',
            bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9},
            zorder=2,
        )

    ax_map.scatter(
        vectors_2d[:, 0],
        vectors_2d[:, 1],
        c='#d6eaf8',
        s=node_sizes,
        alpha=0.68,
        edgecolors='#5dade2',
        linewidths=1.25,
        zorder=3,
    )

    for index, char in enumerate(chars):
        point_x, point_y = vectors_2d[index]
        portrait = portraits.get(char)
        if portrait is not None:
            image_box = OffsetImage(portrait, zoom=float(avatar_zoom[index]))
            annotation_box = AnnotationBbox(
                image_box,
                (point_x, point_y),
                frameon=True,
                pad=0.16,
                bboxprops={'edgecolor': '#1b2631', 'facecolor': 'white', 'lw': 1.2},
                zorder=4,
            )
            ax_map.add_artist(annotation_box)
        else:
            ax_map.scatter(
                [point_x],
                [point_y],
                c='#2e86de',
                s=float(node_sizes[index]) * 0.42,
                alpha=0.95,
                edgecolors='#1b2631',
                linewidths=1.1,
                zorder=4,
            )

    for index, label in enumerate(map_labels):
        point_x, point_y = vectors_2d[index]
        placement = placements[index]
        ax_map.annotate(
            label,
            xy=(point_x, point_y),
            xytext=(placement['x'], placement['y']),
            textcoords='data',
            fontsize=11,
            weight='bold',
            ha=placement['ha'],
            va=placement['va'],
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#dfe6e9', 'alpha': 0.96},
            arrowprops={'arrowstyle': '-', 'color': '#95a5a6', 'lw': 1.0, 'alpha': 0.85},
            zorder=5,
        )

    summary_lines = ['关系 Top 5']
    for index, row in enumerate(pair_rows[:min(5, len(pair_rows))], start=1):
        summary_lines.append(f"{index}. {row['left']} - {row['right']}  {row['score']:.2f}")
    summary_lines.append('')
    summary_lines.append(f"蓝线 = {component_labels['co_scene']}")
    summary_lines.append(f"橙线 = {component_labels['dialogue']}")
    summary_lines.append(f"绿线 = {component_labels['mention']}")
    summary_lines.append('外圈 = 参与文本量')
    summary_lines.append('头像 = 角色形象')
    ax_map.text(
        0.02,
        0.02,
        '\n'.join(summary_lines),
        transform=ax_map.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        color='#2f3640',
        bbox={'boxstyle': 'round,pad=0.4', 'facecolor': 'white', 'edgecolor': '#ced6e0', 'alpha': 0.95},
        zorder=6,
    )

    fig.suptitle('剧本人物关系总览', fontsize=18, y=0.975)
    formula = (
        f"总分 = {component_weights['co_scene']:.2f}×{component_labels['co_scene']} + {component_weights['dialogue']:.2f}×{component_labels['dialogue']} + {component_weights['mention']:.2f}×{component_labels['mention']}（各分项已按角色活跃度归一化）"
    )
    plt.figtext(0.5, 0.02, formula, ha='center', fontsize=10, color='#576574')
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.93, wspace=0.16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {output_path}")

