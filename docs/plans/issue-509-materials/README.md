# #509 原创 MV 1.5.1 工程数据覆盖层

这里是《The Night Signal》的原创事件/地图数据，适用仓库 [MIT 许可](../../../LICENSE)。
它需要用户自有的合法 **RPG Maker MV 1.5.1** 环境组装，不能单独启动；不包含编辑器、core、NW.js、默认数据库、字体、图像或音频。不要拿 MZ 或其他 MV 版本组装后继承本轮版本结论。

本轮已在官方日文 Windows 试用编辑器 About 及实际 core 中分别确认 1.5.1，使用 New Project 创建工程，再关闭工程并写入原创事件数据。修正版已完成 TPP 8.8.9 / 实际 RMMVJS parser 2.19 的真实 GUI 导入及无修改/限定译文导出，部分原生 NW.js 和 HTML5 运行有证据；前插命令后的 context 迁移出现一处译文丢失，生产 bridge 仍暂缓。具体覆盖与未测项见[实验报告](../issue-509-translatorpp-roundtrip-2026-10-21.md)。

## 内容与分母

- [Map001.json](Map001.json)：一个 17×13 地图、两个事件各两页、三个选项和取消分支；数值与剧情均用于本轮构造测试。
- [CommonEvents.json](CommonEvents.json)：一个经 117 调用的公共事件，变量 1 加一后显示两条消息。
- [occurrences.json](occurrences.json)：**实际工程参数路径**，由 TPP 导入前的独立扫描取得；19 条 401、3 个 102 数组元素、3 个 402 标签镜像，共 25 个参数位置。主要文本出现位置为 22 个、唯一原文为 19 个。
- [storyboard.json](storyboard.json)：制作材料与固定构造译文，包含两列冲突、null/空串的原始测试意图；不是 TPP 工程或实际导出结果。实测右列候选生效、空串回退原文；GUI Delete 仍保存空串，之后在未使用列输入再撤销，取得非空原文行的 null。独立 null 工程重开、保存、实际导出均已验证，具体行值和限定结论见报告 §4.5。

网格行数、parser context 和上述参数位置不能互换。102 与 402 的同名文本分别列明，不用去重行数当 occurrence；地图显示名、数据库、脚本和其他命令文本不在本清单的 coverage 范围。

两份游戏 JSON 在 `.gitattributes` 中单独设置 `-text`，保留实测原件的 CRLF 字节；`occurrences.json` 中的文件 SHA-256 按这些原始字节计算。Git blob 及 LF/CRLF 两种 checkout 配置均须与原件同 hash，不以 JSON 内容相同代替字节核对。

## 从合法新工程复现

1. 在 MV 1.5.1 中新建一个独立工程，确认其 `Game.rpgproject` 为 `RPGMV 1.5.1`、六个 `rpg_*.js` 与本轮冻结窗口一致。运行环境和素材由合法安装自行提供。
2. 关闭这个新工程，保留最初生成的文件/hash；只在本轮新副本中，将 `data/Map001.json` 和 `data/CommonEvents.json` 换为这里的两份原创文件。
3. 用编辑器设置：标题 `The Night Signal`，初始队伍仅 actor 1，actor 1 名为 `Mira`，起点地图 1、坐标 (8,6)。switch 1 命名 `Gate page two`，variable 1 命名 `Bell count`；事件会在进入第一页时把变量置为 7。标题背景图和标题 BGM 名称留空。本轮保留日文模板其他默认菜单用语和素材。
4. 作为整文件未选守卫，将 item 1 名称设为 `Unselected bronze key`、说明设为 `UNSELECTED_RESOURCE_509_KEEP`，冻结后在 TPP 中不选择该数据库文件。
5. 清空插件列表。本轮 `plugins.js` 使用编辑器生成的两行注释头，随后为 `var $plugins =` 和空数组；编辑器必须能重新打开并显示空插件管理器。不要用失去生成头的自定义注释替换文件。
6. 重新打开工程，核对地图上的两个事件及页面条件；再关闭，冻结全文件树/hash 和独立 occurrence 清单。只从该原件创建 TPP 输入和运行副本。

本目录不提供通用项目写入器或翻译 writeback。这里的组装是原创夹具制作，不是本项目 `check -> apply` 集成；TPP 仍须经其正常 GUI 创建工程、选列/context、导出到新游戏副本，再独立比较。

## 运行路线

起点正上方为 event 1，右边为 event 2。标题菜单选择新游戏。

- 面向上方并确认：依次显示同文、两行引号/路径、`Count: 7; guide: Mira.` 和颜色/图标/等待消息。
- `Look at the chart` 显示图表分支；`Call the bell` 显示敲铃消息，调用公共事件，显示另一处同文和 `Bell total: 8.`。
- `Leave` 或取消会显示各自消息，并打开 switch 1；再次与 event 1 交互应进入第二页，同文后为四行消息。
- 与右边 event 2 交互：第一页显示同文和抽屉消息，开启自开关 A；下一次交互为第二页的 `The latch clicks twice.`。

原生 NW.js 和 HTML5 浏览器运行结果要分开报告。浏览器按键须保持正常按键时长，让引擎能在更新帧中读取；不能通过调用引擎内部函数跳过页面/分支验证。

## 已保留的构造失败

首次空插件文件没有保留编辑器认可的生成头，复开被拒绝；修正后复开成功。第二次 HTML5 检查发现字面路径只在 JSON 层转义，原生消息显示成 `C:Gate.`；修正版在 MV 消息层也双写反斜杠，已实际显示 `C:\Trial\Gate`。旧材料、失败数据和对应 hash 留本地忽略目录，未抹掉失败，也不把这些构造错误归为 TPP 缺陷。

当前文件来自修正版 v3 清单；之后若再改任一文本、命令或配置，应产生新基线，不复用旧检查或跨版本 context 结论。
