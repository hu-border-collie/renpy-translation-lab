# relation_analyzer

`relation_analyzer/` 是 [extract_relations.py](../extract_relations.py) 的内部实现目录。

模块划分：
- `common.py`
  - 共享常量、依赖加载、API key 读取、embedding 缓存和头像资源读取
- `parsing.py`
  - `.rpy / .txt` 文本解析、说话人归一化、角色自动推断和语义模式文本收集
- `relations.py`
  - `relation` 模式的人物关系统计、打分和 CSV 数据生成
- `semantic.py`
  - `semantic` 模式的 embedding 调用、重试、缓存和角色向量生成
- `plotting.py`
  - 热力图、关系网络图和语义图绘制
- `cli.py`
  - 命令行参数入口

## 使用说明

对外入口仍然是仓库根目录下的 `extract_relations.py`。常见用法：
- 不传 `--characters` 时，会自动选择出场最多的说话人
- `--mode relation` 会输出人物关系热力图、网络图和 `*_relations.csv`
- `--mode semantic` 会输出语义相似度热力图和网络图
- 如果不想读取立绘头像，可以加 `--portraits off`
- `relation` 模式不需要 Gemini API
- `semantic` 模式需要有效的 Gemini API key 和 `google-genai`

## 依赖

基础依赖：
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pillow`

仅 `semantic` 模式额外需要：
- `google-genai`

安装时直接使用仓库根目录的 [requirements.txt](../requirements.txt) 即可。
