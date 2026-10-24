# MV 叙事核心的原创离线材料

本目录的 JSON、对白、参数名和预期译文均为仓库原创，按仓库许可证提供；不含商业
游戏、引擎、插件、字体、存档或素材。测试生成的六个 core 和插件文件仅是原创惰性
标记字节，用来验证摘要 / profile 窗口；它们不构成可运行的 MV / GALV 实现。

- [source/data/Map001.json](source/data/Map001.json)：消息续行、popup / 大小写颜色码、
  变量 / 姓名、多事件页、空消息、独立 403 取消及公共事件调用。
- [source/data/Map002.json](source/data/Map002.json)：另一语境的相同原文、禁用取消、
  映射选项，以及明确未支持的脚本 / 插件命令。
- [source/data/CommonEvents.json](source/data/CommonEvents.json)：公共事件物理文本，
  被两个调用点引用；不按运行调用次数重复记译文或写回。
- [source/data/Actors.json](source/data/Actors.json)：A2 叙事窗口之外的数据库文本与资源名。
- [positions.json](positions.json)：手选 14 个非空 occurrence、6 个独立镜像目标，以及
  固定原文 / 中文。两个 “Same words.” 按不同位置采用不同译文。
- [expected/data/Map001.json](expected/data/Map001.json)、
  [expected/data/Map002.json](expected/data/Map002.json)和
  [expected/data/CommonEvents.json](expected/data/CommonEvents.json)：手写预期字节，
  只更改上述 20 处；保留空消息、特殊数值拼写、空白、分支、资源及所有其他字段。
  预期文件不能由被测 Adapter / renderer 重建。

[测试模块](../../engine_adapters/test_rpgmaker_mv.py)另以原创固定命令扩充嵌套选项、
非法事件、错误关联、未知控制码、漂移、伪造 / 重复、事务故障和恢复用例，并确认
现有 TranslationPlan 仍拒绝 MV 产品入口。该材料验证共享核心，不能代替真实游戏
清单、完整翻译覆盖、原生显示 / 存读档、插件运行或独立语义审阅。
