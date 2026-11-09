# 应用图标

`app-icon.png` 是 GUI 与仓库首页共用的应用图标：蓝色圆角底、展开书页与对话气泡。
保留透明边缘，浅色和深色主题共用，不跟随按钮图标的主题着色。

- 原始尺寸：1254 × 1254，RGBA PNG；Qt 按窗口、侧栏和屏幕缩放比例渲染。
- 使用位置：`QApplication` 默认窗口图标、主窗口图标、左侧品牌区、仓库首页。
- 来源：2026-10-29 使用 Codex 内置 imagegen 生成；未使用 CLI/API fallback。
- 原始生成图直接复制入仓库；不依赖用户目录中的生成缓存，也不需要额外运行时依赖。
- Windows 源码启动会在创建 Qt UI 前设置进程级应用身份，不修改注册表或桌面快捷方式。
- `app-icon.ico` 从同一 PNG 通过 Qt 图像编码器导出，包含 16、24、32、48、64、128、256 像素的透明图标，供 Windows 快捷方式使用。
- Windows 快捷方式应引用资源管理器可直接读取的固定绝对路径，避免引用应用私有缓存或临时目录中的图标；更新后应在资源管理器中确认实际显示。

## 生成提示词

```text
Use case: logo-brand.
Asset type: the actual square desktop application icon for Ren'Py Translation Lab, a visual-novel translation workbench. Produce a single finished icon, not a presentation or contact sheet.
Primary request: a memorable, calm, carefully designed symbol combining an open book and a dialogue bubble. The two broad pages of the open book should themselves suggest two conversational panels, representing the source story and its translation. This must read as ONE coherent compact symbol, not a collection of separate objects.
Style: exceptionally clean vector-like geometric app icon with soft, deliberate corners, confident filled shapes and restrained depth. Refined desktop utility identity, friendly to fiction readers. Small-size legibility is paramount.
Composition: front view, centered large white and pale-cyan book/dialogue symbol inside a cobalt-blue rounded-square tile. A small speech tail is integrated into the lower outline of the book. Clear central negative-space seam. Use broad bold silhouettes, no tiny internal lines. The tile occupies about 88% of the square canvas, leaving equal transparent margins.
Palette: match the existing application's blue interface: cobalt #2563EB and deep blue #1D4ED8, white pages with a pale icy-blue second page. Very subtle tonal variation, crisp silhouette. Maintain contrast on light and dark desktops.
Background: genuinely transparent outside the rounded-square tile, preserving clean alpha.
Constraints: no letters, no words, no typography, no watermark, no extra pictograms, no sparkles, no orbit, no robot, no magnifying glass, no game-controller, no 3D perspective, no photographic texture, no external cast shadow. Do not reproduce another app's logo. Square 1024px icon.
```
