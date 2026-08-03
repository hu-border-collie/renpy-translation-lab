"""extract_relations 的内部实现包。

CLI 入口请使用 ``python -m relation_analyzer.cli`` 或 ``relation_analyzer.cli:main``。
本包初始化故意不导入 CLI，避免 ``python -m relation_analyzer.cli`` 触发
``sys.modules`` RuntimeWarning 并丢失帮助输出。
"""
