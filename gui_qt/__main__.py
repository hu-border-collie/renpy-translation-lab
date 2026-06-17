"""Entry point for the optional GUI.

Run with:
    python -m gui_qt

If PySide6 is missing, prints a clear message and exits without traceback.
"""
import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "尚未安装 PySide6。\n"
            "图形界面是可选组件，可以用下面的命令安装：\n"
            "    pip install -r requirements-gui.txt\n"
            "然后重新运行：python -m gui_qt"
        )
        return 1

    # Defer heavy imports until we know the dependency exists
    from .app import run_app

    return run_app(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
