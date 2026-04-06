from .common import *

def embed_texts(texts, batch_size, model_name, output_dimensionality, cache_dir):
    np = load_numpy()
    types = None
    client = None
    all_embeddings = []
    config_kwargs = {"output_dimensionality": output_dimensionality}
    task_type = get_task_type_for_model(model_name)
    if task_type:
        config_kwargs["task_type"] = task_type
    cache_bucket = get_embedding_cache_bucket(cache_dir, model_name, output_dimensionality, task_type)
    cache_hits = 0
    api_calls = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = [None] * len(batch)
        missing_texts = []
        missing_indices = []

        for index, text_item in enumerate(batch):
            cached = load_cached_embedding(cache_bucket, text_item)
            if cached is not None:
                batch_embeddings[index] = cached
                cache_hits += 1
            else:
                missing_texts.append(text_item)
                missing_indices.append(index)

        if not missing_texts:
            all_embeddings.extend(batch_embeddings)
            continue

        if types is None:
            _, types = load_embedding_libs()
        if client is None:
            client = get_client()

        attempt = 0
        while True:
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=missing_texts,
                    config=types.EmbedContentConfig(**config_kwargs),
                )
                api_calls += 1
                embeddings = list(getattr(response, "embeddings", []) or [])
                if len(embeddings) != len(missing_texts):
                    raise RuntimeError(f"\u672c\u6279\u8fd4\u56de\u5411\u91cf\u6570\u91cf\u5f02\u5e38: \u671f\u671b {len(missing_texts)}\uff0c\u5b9e\u9645 {len(embeddings)}")
                for missing_index, embedding in zip(missing_indices, embeddings):
                    vector = np.asarray(embedding.values, dtype=float)
                    batch_embeddings[missing_index] = vector
                    save_cached_embedding(cache_bucket, batch[missing_index], vector)
                break
            except Exception as exc:
                if is_auth_error(exc):
                    expired_source = get_api_key_source()
                    if rotate_api_key():
                        client = get_client()
                        print(f"\u26a0\ufe0f \u5f53\u524d Gemini API key \u5df2\u5931\u6548\uff0c\u5df2\u5207\u6362\u5230\u4e0b\u4e00\u4e2a key: {get_api_key_source()} (\u4e0a\u4e00\u628a: {expired_source})")
                        continue
                    raise SystemExit(
                        "\u274c \u6240\u6709\u53ef\u7528\u7684 Gemini API key \u90fd\u65e0\u6548\u6216\u5df2\u8fc7\u671f\u3002"
                        f"\u6700\u540e\u4e00\u6b21\u5931\u8d25\u6765\u6e90: {expired_source}\u3002"
                        "\u8bf7\u66f4\u65b0\u4ed3\u5e93\u6839\u76ee\u5f55\u4e0b\u7684 api_keys.json\uff08\u53ef\u53c2\u8003 api_keys.example.json\uff09\uff0c"
                        "\u6216\u4fee\u6b63\u73af\u5883\u53d8\u91cf GEMINI_API_KEY\u3002"
                    ) from exc
                attempt += 1
                wait_seconds = get_retry_delay_seconds(exc) if is_rate_limit_error(exc) else None
                if attempt >= API_RETRIES:
                    raise RuntimeError(f"Embedding \u8bf7\u6c42\u8fde\u7eed\u5931\u8d25\uff0c\u6700\u540e\u4e00\u6b21\u9519\u8bef: {exc}") from exc
                if wait_seconds is None:
                    wait_seconds = attempt * 2
                wait_seconds = max(wait_seconds, 1.0)
                print(f"\u26a0\ufe0f Embedding \u8bf7\u6c42\u5931\u8d25\uff0c\u7b2c {attempt} \u6b21\u91cd\u8bd5\u524d\u7b49\u5f85 {wait_seconds:.1f} \u79d2: {exc}")
                time.sleep(wait_seconds)

        if any(item is None for item in batch_embeddings):
            raise RuntimeError("\u5b58\u5728\u672a\u586b\u5145\u7684 embedding \u7ed3\u679c\uff0c\u7f13\u5b58\u6216\u8bf7\u6c42\u6d41\u7a0b\u5f02\u5e38\u3002")
        all_embeddings.extend(batch_embeddings)

    if not all_embeddings:
        raise RuntimeError("\u6ca1\u6709\u62ff\u5230\u4efb\u4f55 embedding \u7ed3\u679c\u3002")

    if cache_hits:
        print(f"\U0001f4be Embedding \u7f13\u5b58\u547d\u4e2d {cache_hits} \u6761\uff0c\u5b9e\u9645 API \u8c03\u7528 {api_calls} \u6b21\u3002")

    return np.vstack(all_embeddings)

def extract_character_vectors(char_texts, batch_size, model_name, output_dimensionality, max_texts_per_character, cache_dir):
    char_vectors = {}
    print("\U0001f680 \u6b63\u5728\u8c03\u7528 Gemini Embedding \u63d0\u53d6\u89d2\u8272\u8bed\u4e49\u5411\u91cf...")

    for char, texts in char_texts.items():
        if not texts:
            print(f"\u26a0\ufe0f \u5267\u672c\u4e2d\u672a\u53d1\u73b0\u5173\u4e8e [{char}] \u7684\u6709\u6548\u5267\u60c5\u7247\u6bb5\uff0c\u8df3\u8fc7\u3002")
            continue

        selected_texts = sample_texts_evenly(texts, max_texts_per_character)
        if len(selected_texts) != len(texts):
            print(f"\u2702\ufe0f [{char}] \u5df2\u4ece {len(texts)} \u6bb5\u5267\u60c5\u4e0a\u4e0b\u6587\u4e2d\u5747\u5300\u91c7\u6837 {len(selected_texts)} \u6bb5\u3002")
        print(f"\U0001f4e6 \u6b63\u5728\u5411\u91cf\u5316 [{char}] \u7684 {len(selected_texts)} \u6bb5\u5267\u60c5\u4e0a\u4e0b\u6587...")
        embeddings = embed_texts(selected_texts, batch_size, model_name, output_dimensionality, cache_dir)
        char_vectors[char] = embeddings.mean(axis=0)

    return char_vectors

