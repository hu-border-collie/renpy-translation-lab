import argparse
import ast
import hashlib
import io
import json
import os
import pickle
import re
import sys
import time
import tokenize
import zlib
from pathlib import Path

# ====== 路径 ======
MODULE_DIR = Path(__file__).resolve().parent
TOOL_ROOT = MODULE_DIR.parent
CONFIG_SEARCH_ROOTS = [TOOL_ROOT, MODULE_DIR]


def _find_local_config(filename):
    for root in CONFIG_SEARCH_ROOTS:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return TOOL_ROOT / filename


INPUT_FILE = TOOL_ROOT / "script.txt"
OUTPUT_FILE = TOOL_ROOT / "character_matrix.png"
API_KEY_CONFIG = _find_local_config("api_keys.json")
TARGET_CHARACTERS = []
AUTO_CHARACTER_LIMIT = 6
CHARACTER_ALIASES = {}
SPEAKER_TO_CHARACTER = {}
BATCH_SIZE = 100
CONTEXT_WINDOW = 1
EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
API_RETRIES = 3
MAX_TEXTS_PER_CHARACTER = 0
EMBEDDING_CACHE_DIR = TOOL_ROOT / ".cache" / "extract_relations_embeddings"
PORTRAIT_CANDIDATES = {}
PORTRAIT_PENALTY_HINTS = ("nude", "battle_damaged", "pumpkin_gore", "gun", "smoking", "night", "blood", "gore")

TRANSLATE_BLOCK_RE = re.compile(r"^\s*translate\s+\w+\s+(?P<label>[^:]+)\s*:\s*$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ASCII_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ ]*$")
ASSET_FILE_RE = re.compile(
    r"^[\w./\\-]+\.(png|jpg|jpeg|bmp|gif|webp|ogg|mp3|wav|webm|mp4|avi|txt|json|rpy)$",
    re.IGNORECASE,
)
ONLY_SYMBOLS_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
CONTROL_KEYWORDS = {
    "call", "camera", "hide", "if", "elif", "else", "image", "init", "jump",
    "label", "menu", "pass", "pause", "play", "python", "queue", "return",
    "scene", "screen", "show", "stop", "style", "transform", "voice", "window",
    "while", "for", "with", "default", "define", "layeredimage",
}
TEXT_COMMANDS = {"centered", "extend", "narrator"}
SINGLE_CHAR_PRECEDERS = "和与跟对向给找叫帮替把被让朝等爱恨像为比同陪靠从离到往"
SINGLE_CHAR_FOLLOWERS = "说问道想看听笑哭叫答喊写拿抱推拉打找跟在向给被让来去走跑站坐等爱恨帮指摸望瞪"

_CLIENT = None
_NUMPY = None
_PLOT_LIBS = None
_EMBEDDING_LIBS = None
_IMAGE_LIBS = None
_API_KEYS = None
_API_KEY_INDEX = 0
_ARCHIVE_INDEX_CACHE = {}
_PORTRAIT_CACHE = {}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def is_placeholder_api_key(value):
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return True
    placeholder_markers = (
        "your-key",
        "your api key",
        "your-api-key",
        "your_gemini_api_key",
        "your-gemini-api-key",
        "paste-key",
        "paste-api-key",
        "replace-me",
    )
    return any(marker in text for marker in placeholder_markers)

def load_api_keys_from_config():
    if not API_KEY_CONFIG.exists():
        return []

    try:
        config = json.loads(API_KEY_CONFIG.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"⚠️ 读取 api_keys.json 失败，将回退到环境变量: {exc}")
        return []

    keys = config.get("api_keys", []) if isinstance(config, dict) else []
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return []

    valid_keys = []
    for index, key in enumerate(keys, start=1):
        if isinstance(key, str) and key.strip() and not is_placeholder_api_key(key):
            valid_keys.append((key.strip(), f"{API_KEY_CONFIG}#{index}"))
    return valid_keys

def load_api_keys_from_environment():
    valid_keys = []
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name)
        if value and value.strip():
            valid_keys.append((value.strip(), env_name))
    return valid_keys

def load_api_keys():
    global _API_KEYS, _API_KEY_INDEX
    if _API_KEYS is not None:
        return _API_KEYS

    config_keys = load_api_keys_from_config()
    if config_keys:
        _API_KEYS = config_keys
        _API_KEY_INDEX = 0
        return _API_KEYS

    env_keys = load_api_keys_from_environment()
    if env_keys:
        _API_KEYS = env_keys
        _API_KEY_INDEX = 0
        return _API_KEYS

    raise SystemExit("❌ 未找到可用 API key。请在仓库根目录的 api_keys.json 中配置 api_keys，或设置 GEMINI_API_KEY。")

def get_api_key():
    return load_api_keys()[_API_KEY_INDEX][0]

def get_api_key_source():
    return load_api_keys()[_API_KEY_INDEX][1]

def rotate_api_key():
    global _API_KEY_INDEX, _CLIENT
    api_keys = load_api_keys()
    if _API_KEY_INDEX + 1 >= len(api_keys):
        return False
    _API_KEY_INDEX += 1
    _CLIENT = None
    return True

def is_auth_error(exc):
    text = str(exc).lower()
    markers = (
        "api key expired",
        "api key invalid",
        "api_key_invalid",
        "invalid api key",
        "please renew the api key",
        "unauthenticated",
        "permission_denied",
    )
    return any(marker in text for marker in markers)

def is_rate_limit_error(exc):
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota exceeded" in text or "429" in text

def get_task_type_for_model(model_name):
    normalized_model = model_name.split("/")[-1].lower()
    if normalized_model == "gemini-embedding-001":
        return "SEMANTIC_SIMILARITY"
    return None

def get_retry_delay_seconds(exc):
    text = str(exc)
    delays = []
    patterns = (
        r"Please retry in ([0-9]+(?:\.[0-9]+)?)(ms|s)",
        r"'retryDelay': '([0-9]+(?:\.[0-9]+)?)(ms|s)'",
    )
    for pattern in patterns:
        for value_text, unit in re.findall(pattern, text, flags=re.IGNORECASE):
            value = float(value_text)
            if unit.lower() == "ms":
                value /= 1000.0
            delays.append(value)
    if not delays:
        return None
    return max(delays)

def sample_texts_evenly(texts, max_texts):
    if not max_texts or max_texts <= 0 or len(texts) <= max_texts:
        return list(texts)
    if max_texts == 1:
        return [texts[len(texts) // 2]]
    last_index = len(texts) - 1
    indices = [int(i * last_index / (max_texts - 1)) for i in range(max_texts)]
    return [texts[index] for index in indices]

def get_embedding_cache_bucket(cache_dir, model_name, output_dimensionality, task_type):
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.split("/")[-1])
    safe_task = (task_type or "default").lower()
    bucket = cache_dir / safe_model / f"{output_dimensionality}d_{safe_task}"
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket

def get_embedding_cache_path(cache_bucket, text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return cache_bucket / f"{digest}.npy"

def load_cached_embedding(cache_bucket, text):
    np = load_numpy()
    cache_path = get_embedding_cache_path(cache_bucket, text)
    if not cache_path.exists():
        return None
    try:
        return np.load(cache_path, allow_pickle=False)
    except Exception:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None

def save_cached_embedding(cache_bucket, text, vector):
    np = load_numpy()
    cache_path = get_embedding_cache_path(cache_bucket, text)
    if cache_path.exists():
        return
    temp_path = cache_path.with_suffix(".tmp")
    with open(temp_path, "wb") as handle:
        np.save(handle, np.asarray(vector, dtype=np.float32))
    temp_path.replace(cache_path)

def load_numpy():
    global _NUMPY
    if _NUMPY is None:
        try:
            import numpy as imported_numpy
        except ImportError as exc:
            raise SystemExit("❌ 缺少依赖 numpy，请先安装：pip install numpy") from exc
        _NUMPY = imported_numpy
    return _NUMPY

def load_image_libs():
    global _IMAGE_LIBS
    if _IMAGE_LIBS is None:
        try:
            from PIL import Image as imported_image
        except ImportError as exc:
            raise SystemExit("\u274c \u7f3a\u5c11\u4f9d\u8d56 pillow\uff0c\u8bf7\u5148\u5b89\u88c5\uff1apip install pillow") from exc
        _IMAGE_LIBS = imported_image
    return _IMAGE_LIBS

class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Disallowed pickle global during RPA index load: {module}.{name}"
        )

def load_pickle_blob(blob):
    return _RestrictedUnpickler(io.BytesIO(blob)).load()

def read_rpa_index(archive_path):
    archive_path = str(archive_path)
    cached = _ARCHIVE_INDEX_CACHE.get(archive_path)
    if cached is not None:
        return cached

    with open(archive_path, 'rb') as infile:
        header = infile.read(40)

        if header.startswith(b'RPA-3.0 '):
            offset = int(header[8:24], 16)
            key = int(header[25:33], 16)
            infile.seek(offset)
            raw_index = load_pickle_blob(zlib.decompress(infile.read()))
            index = {}
            for name, chunks in raw_index.items():
                decoded_chunks = []
                for chunk in chunks:
                    if len(chunk) == 2:
                        start = b''
                        chunk_offset, chunk_len = chunk
                    else:
                        chunk_offset, chunk_len, start = chunk
                        if start is None:
                            start = b''
                        elif not isinstance(start, bytes):
                            start = str(start).encode('latin-1', errors='ignore')
                    decoded_chunks.append((int(chunk_offset) ^ key, int(chunk_len) ^ key, start))
                index[str(name)] = decoded_chunks
            _ARCHIVE_INDEX_CACHE[archive_path] = index
            return index

        if header.startswith(b'RPA-2.0 '):
            infile.seek(0)
            line = infile.read(24)
            offset = int(line[8:], 16)
            infile.seek(offset)
            raw_index = load_pickle_blob(zlib.decompress(infile.read()))
            index = {}
            for name, chunks in raw_index.items():
                decoded_chunks = []
                for chunk in chunks:
                    chunk_offset, chunk_len = chunk[:2]
                    start = b''
                    if len(chunk) >= 3:
                        start = chunk[2] or b''
                        if not isinstance(start, bytes):
                            start = str(start).encode('latin-1', errors='ignore')
                    decoded_chunks.append((int(chunk_offset), int(chunk_len), start))
                index[str(name)] = decoded_chunks
            _ARCHIVE_INDEX_CACHE[archive_path] = index
            return index

    raise RuntimeError('Unsupported RPA format (expecting RPA-3.0 or RPA-2.0).')

def read_rpa_member(archive_path, member_name):
    index = read_rpa_index(archive_path)
    chunks = index.get(member_name)
    if not chunks:
        return None

    data = bytearray()
    with open(archive_path, 'rb') as source:
        for chunk_offset, chunk_len, start in chunks:
            if start:
                data.extend(start)
            source.seek(chunk_offset)
            data.extend(source.read(chunk_len))
    return bytes(data)

def find_archive_for_input(input_path):
    path = Path(input_path)
    cursor = path.parent if path.is_file() else path
    candidates = []

    for current in [cursor, *cursor.parents]:
        if current.name.lower() == 'work':
            project_dir = current.parent
            candidates.extend(sorted(project_dir.glob('build/*/game/archive.rpa')))
        candidates.extend(sorted(current.glob('archive.rpa')))
        candidates.extend(sorted(current.glob('game/archive.rpa')))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        return resolved
    return None

def choose_portrait_member(archive_path, char):
    index = read_rpa_index(archive_path)
    preferred = PORTRAIT_CANDIDATES.get(char, [])
    for member_name in preferred:
        if member_name in index:
            return member_name

    normalized_char = char.lower()
    matched = []
    for member_name in index:
        lower = member_name.lower().replace('\\', '/')
        if not lower.endswith(('.webp', '.png', '.jpg', '.jpeg')):
            continue
        if normalized_char == 'spencer':
            ok = '/side spencer/' in lower
        else:
            ok = f'/{normalized_char}/' in lower
        if ok:
            matched.append(member_name)

    if not matched:
        return None

    def portrait_score(name):
        lower = name.lower()
        score = 0
        if 'neutral' in lower:
            score -= 6
        if 'blank' in lower:
            score -= 5
        if 'smiling' in lower:
            score -= 4
        if 'curious' in lower:
            score -= 3
        if any(hint in lower for hint in PORTRAIT_PENALTY_HINTS):
            score += 8
        score += lower.count('day')
        score += len(lower) / 1000
        return score

    matched.sort(key=portrait_score)
    return matched[0]

def make_portrait_array(image_bytes):
    np = load_numpy()
    Image = load_image_libs()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert('RGBA')
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        if image.height > image.width * 1.15:
            image = image.crop((0, 0, image.width, max(1, int(image.height * 0.62))))
        side = max(image.width, image.height)
        canvas = Image.new('RGBA', (side, side), (255, 255, 255, 0))
        offset_x = (side - image.width) // 2
        offset_y = (side - image.height) // 2
        canvas.paste(image, (offset_x, offset_y), image)
        canvas = canvas.resize((160, 160))
        return np.asarray(canvas)

def resolve_character_portraits(input_path, characters):
    archive_path = find_archive_for_input(input_path)
    if not archive_path:
        return {}

    portraits = {}
    archive_key = str(archive_path)
    for char in characters:
        cache_key = (archive_key, char)
        if cache_key in _PORTRAIT_CACHE:
            if _PORTRAIT_CACHE[cache_key] is not None:
                portraits[char] = _PORTRAIT_CACHE[cache_key]
            continue

        member_name = choose_portrait_member(archive_path, char)
        if not member_name:
            _PORTRAIT_CACHE[cache_key] = None
            continue

        try:
            image_bytes = read_rpa_member(archive_path, member_name)
            if not image_bytes:
                _PORTRAIT_CACHE[cache_key] = None
                continue
            portrait_array = make_portrait_array(image_bytes)
        except Exception as exc:
            print(f"\u26a0\ufe0f \u8bfb\u53d6\u89d2\u8272\u5934\u50cf\u5931\u8d25 [{char}]: {exc}")
            _PORTRAIT_CACHE[cache_key] = None
            continue

        _PORTRAIT_CACHE[cache_key] = portrait_array
        portraits[char] = portrait_array

    if portraits:
        print(f"\u5df2\u4ece\u8d44\u6e90\u5305\u8bfb\u53d6 {len(portraits)} \u4e2a\u89d2\u8272\u5934\u50cf\u3002")
    return portraits

def load_plot_libs():
    global _PLOT_LIBS
    if _PLOT_LIBS is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as imported_pyplot
            from matplotlib.offsetbox import AnnotationBbox as imported_annotation_bbox
            from matplotlib.offsetbox import OffsetImage as imported_offset_image
        except ImportError as exc:
            raise SystemExit("\u274c \u7f3a\u5c11\u4f9d\u8d56 matplotlib\uff0c\u8bf7\u5148\u5b89\u88c5\uff1apip install matplotlib") from exc

        try:
            from sklearn.decomposition import PCA as imported_pca
            from sklearn.metrics.pairwise import cosine_similarity as imported_cosine_similarity
        except ImportError as exc:
            raise SystemExit("\u274c \u7f3a\u5c11\u4f9d\u8d56 scikit-learn\uff0c\u8bf7\u5148\u5b89\u88c5\uff1apip install scikit-learn") from exc

        _PLOT_LIBS = (
            imported_pyplot,
            imported_pca,
            imported_cosine_similarity,
            imported_offset_image,
            imported_annotation_bbox,
        )
    return _PLOT_LIBS

def load_embedding_libs():
    global _EMBEDDING_LIBS
    if _EMBEDDING_LIBS is None:
        try:
            from google import genai as imported_genai
            from google.genai import types as imported_types
        except ImportError as exc:
            raise SystemExit("❌ 缺少依赖 google-genai，请先安装：pip install google-genai") from exc
        _EMBEDDING_LIBS = (imported_genai, imported_types)
    return _EMBEDDING_LIBS

def get_client():
    global _CLIENT
    if _CLIENT is None:
        genai, _ = load_embedding_libs()
        _CLIENT = genai.Client(api_key=get_api_key())
        print(f"🔑 当前使用的 Gemini API key 来源: {get_api_key_source()}")
    return _CLIENT

def resolve_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = TOOL_ROOT / path
    return path.resolve()

