"""Download optional GUI fonts from their upstream publishers."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gui_qt.font_helpers import (  # noqa: E402
    MONO_FONT_FILENAME,
    UI_FONT_FILENAME,
    user_fonts_dir,
)


HARMONYOS_URL = (
    "https://developer.huawei.com/Enexport/sites/default/images/download/next/"
    "HarmonyOS-Sans.rar"
)
HARMONYOS_ARCHIVE_SHA256 = "510274fbc12e80abe641d7b0d9bd4d2bb4fec111b7b710122364a7723fe12bd7"
HARMONYOS_MEMBER = "HarmonyOS-Sans/HarmonyOS_SansSC/HarmonyOS_SansSC_Regular.ttf"
HARMONYOS_FONT_SHA256 = "984cf609545acee8ef060780fb70fc3099b058c0553416331b6e863fdf7c26fa"

LXGW_VERSION = "v1.522"
LXGW_URL = (
    f"https://github.com/lxgw/LxgwWenkaiGB/releases/download/{LXGW_VERSION}/"
    "LXGWWenKaiMonoGB-Regular.ttf"
)
LXGW_FONT_SHA256 = "fb82a0d6b9c0a1a3c83ad303eab1cc998e6a52c1028027fc3527455bdadb4ecb"


class FontInstallError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise FontInstallError(
            f"{label} SHA-256 校验失败：期望 {expected}，实际 {actual}"
        )


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "renpy-translation-lab/font-installer"})
    try:
        with urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        raise FontInstallError(f"下载失败：{url}\n{exc}") from exc


def _archive_tool() -> str:
    for name in ("bsdtar", "tar"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise FontInstallError("找不到 tar 或 bsdtar，无法解压华为官方 RAR 字体包。")


def _extract_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temp_path.open("wb") as output:
            completed = subprocess.run(
                [_archive_tool(), "-xOf", str(archive), member],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise FontInstallError(f"无法从华为字体包解压 {member}：{detail}")
        _verify(temp_path, HARMONYOS_FONT_SHA256, "HarmonyOS Sans SC 字体")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def install_fonts(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="renpy-fonts-") as temp_dir:
        temp_root = Path(temp_dir)

        harmony_archive = temp_root / "HarmonyOS-Sans.rar"
        print("正在从华为官方来源下载 HarmonyOS Sans...")
        _download(HARMONYOS_URL, harmony_archive)
        _verify(harmony_archive, HARMONYOS_ARCHIVE_SHA256, "HarmonyOS Sans 官方包")
        harmony_target = destination / UI_FONT_FILENAME
        _extract_member(harmony_archive, HARMONYOS_MEMBER, harmony_target)
        installed.append(harmony_target)

        lxgw_temp = temp_root / MONO_FONT_FILENAME
        print(f"正在从霞鹜文楷官方 Release 下载 {LXGW_VERSION}...")
        _download(LXGW_URL, lxgw_temp)
        _verify(lxgw_temp, LXGW_FONT_SHA256, "LXGW WenKai Mono GB 字体")
        lxgw_target = destination / MONO_FONT_FILENAME
        lxgw_stage = lxgw_target.with_suffix(lxgw_target.suffix + ".tmp")
        try:
            shutil.copyfile(lxgw_temp, lxgw_stage)
            _verify(lxgw_stage, LXGW_FONT_SHA256, "LXGW WenKai Mono GB 字体")
            os.replace(lxgw_stage, lxgw_target)
        finally:
            lxgw_stage.unlink(missing_ok=True)
        installed.append(lxgw_target)
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从字体发布者的官方来源下载并校验可选 GUI 字体。"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=user_fonts_dir(),
        help="字体安装目录（默认：当前用户缓存目录）。",
    )
    args = parser.parse_args(argv)
    try:
        installed = install_fonts(args.destination.expanduser().resolve())
    except FontInstallError as exc:
        print(f"字体安装失败：{exc}", file=sys.stderr)
        return 1
    print("字体安装完成：")
    for path in installed:
        print(f"- {path}")
    print("请重启 GUI 以加载字体。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
