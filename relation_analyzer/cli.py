from .common import *
from .parsing import load_text_units, collect_character_texts, infer_characters_from_units
from .semantic import extract_character_vectors
from .relations import compute_relation_data
from .plotting import analyze_and_plot, analyze_and_plot_relation
from .story_seed import write_story_graph_seed


def parse_args():
    parser = argparse.ArgumentParser(description='提取剧情文本的人物关系或语义关系并生成图表。')
    parser.add_argument('input', nargs='?', default=str(INPUT_FILE), help='输入文件或目录，支持 .txt / .rpy')
    parser.add_argument('--output', default=str(OUTPUT_FILE), help='输出图片路径')
    parser.add_argument('--characters', help='逗号分隔的角色名列表，例如：Spencer,Ian,Andrew')
    parser.add_argument('--auto-characters', type=int, default=AUTO_CHARACTER_LIMIT, help='未显式提供角色名时，自动选取出场最多的说话人数量；0 表示禁用自动推断')
    parser.add_argument('--portraits', choices=('auto', 'off'), default='auto', help='是否自动从 archive.rpa 读取头像')
    parser.add_argument('--mode', choices=('relation', 'semantic'), default='relation', help='分析模式：relation 为人物关系图，semantic 为语义相似图')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='单次 embedding 批量大小（semantic 模式使用）')
    parser.add_argument('--context-window', type=int, default=CONTEXT_WINDOW, help='每个片段向前向后拼接的上下文行数')
    parser.add_argument('--model', default=EMBEDDING_MODEL, help='embedding 模型名（semantic 模式使用）')
    parser.add_argument('--output-dimensionality', type=int, default=OUTPUT_DIMENSIONALITY, help='embedding 输出维度（semantic 模式使用）')
    parser.add_argument('--max-texts-per-character', type=int, default=MAX_TEXTS_PER_CHARACTER, help='每个角色最多用于 embedding 的片段数，0 表示不限制（semantic 模式使用）')
    parser.add_argument('--cache-dir', default=str(EMBEDDING_CACHE_DIR), help='embedding 缓存目录（semantic 模式使用）')
    parser.add_argument('--relation-window-size', type=int, default=12, help='relation 模式下划分局部剧情段的窗口大小')
    parser.add_argument('--csv-output', help='relation 模式导出 CSV 路径，默认与图片同目录同名后缀 _relations.csv')
    parser.add_argument('--story-seed-output', help='relation 模式额外导出 story_graph.seed.json 候选数据，供人工确认后维护 story_graph.json')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    cache_dir = resolve_path(args.cache_dir)
    requested_characters = [item.strip() for item in (args.characters.split(',') if args.characters else TARGET_CHARACTERS) if item.strip()]

    if args.context_window < 0:
        raise SystemExit('❌ context-window 不能小于 0。')
    if args.auto_characters < 0:
        raise SystemExit('❌ auto-characters 不能小于 0。')
    if args.mode == 'semantic' and args.batch_size <= 0:
        raise SystemExit('❌ batch-size 必须大于 0。')
    if args.mode == 'semantic' and args.max_texts_per_character < 0:
        raise SystemExit('❌ max-texts-per-character 不能小于 0。')
    if args.mode == 'relation' and args.relation_window_size <= 0:
        raise SystemExit('❌ relation-window-size 必须大于 0。')

    units = load_text_units(input_path, args.context_window)
    print(f"📚 成功读取输入，共抽取到 {len(units)} 个有效剧情片段。")

    characters = list(requested_characters)
    if not characters and args.auto_characters > 0:
        characters = infer_characters_from_units(units, args.auto_characters)
        if characters:
            print(f"🔎 未显式提供角色列表，已自动选取出场最多的 {len(characters)} 个说话人: {', '.join(characters)}")

    if len(characters) < 2:
        raise SystemExit('❌ 至少需要提供 2 个角色名；也可以不传 --characters，让脚本自动推断主要说话人。')

    portraits = {}
    if args.portraits == 'auto':
        portraits = resolve_character_portraits(input_path, characters)

    if args.mode == 'relation':
        relation_data = compute_relation_data(units, characters, args.relation_window_size)
        if len(relation_data['characters']) > 1:
            if args.story_seed_output:
                write_story_graph_seed(resolve_path(args.story_seed_output), units, characters, relation_data, source_root=input_path)
            active_portraits = {char: portraits.get(char) for char in relation_data['characters']} if portraits else {}
            csv_output = resolve_path(args.csv_output) if args.csv_output else output_path.with_name(f"{output_path.stem}_relations.csv")
            analyze_and_plot_relation(relation_data, output_path, csv_output, active_portraits)
        else:
            raise SystemExit('❌ 提取到的有效角色少于 2 个，无法计算关系。')
        return

    char_texts = collect_character_texts(units, characters)
    vectors = extract_character_vectors(
        char_texts,
        batch_size=args.batch_size,
        model_name=args.model,
        output_dimensionality=args.output_dimensionality,
        max_texts_per_character=args.max_texts_per_character,
        cache_dir=cache_dir,
    )

    if len(vectors) > 1:
        active_portraits = {char: portraits.get(char) for char in vectors.keys()} if portraits else {}
        analyze_and_plot(vectors, char_texts, output_path, active_portraits)
    else:
        raise SystemExit('❌ 提取到的有效角色少于 2 个，无法计算关系。')
