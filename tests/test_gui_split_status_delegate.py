import unittest

try:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QStandardItem, QStandardItemModel
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QStyleOptionViewItem, QTableWidget, QTableWidgetItem

    from gui_qt.split_status_delegate import (
        SPLIT_ACTION_DATA_ROLE,
        SplitStatusActionDelegate,
        read_split_action_payload,
    )
    from gui_qt.split_status_table_helpers import split_action_item_payload
    from gui_qt.app import MainWindow
    from tests import gui_test_support
except ImportError as exc:
    SplitStatusActionDelegate = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(SplitStatusActionDelegate is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiSplitStatusDelegateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_read_split_action_payload_from_model_index(self):
        model = QStandardItemModel(1, 1)
        item = QStandardItem("")
        item.setData(
            split_action_item_payload(
                selectable=True,
                manifest_path=r"C:\pkg\part02\manifest.json",
                part_label="part02/03",
            ),
            SPLIT_ACTION_DATA_ROLE,
        )
        model.setItem(0, 0, item)
        payload = read_split_action_payload(model.index(0, 0))
        self.assertEqual(payload["manifest_path"], r"C:\pkg\part02\manifest.json")
        self.assertEqual(payload["part_label"], "part02/03")

    def test_delegate_emits_select_requested_for_action_cells(self):
        table = QTableWidget(1, 6)
        delegate = SplitStatusActionDelegate(table)
        selected: list[str] = []
        delegate.select_requested.connect(selected.append)
        table.setItemDelegateForColumn(5, delegate)

        item = QTableWidgetItem("")
        item.setData(
            SPLIT_ACTION_DATA_ROLE,
            split_action_item_payload(
                selectable=True,
                manifest_path=r"C:\pkg\part02\manifest.json",
                part_label="part02/03",
            ),
        )
        table.setItem(0, 5, item)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        index = table.model().index(0, 5)
        option = QStyleOptionViewItem()
        option.rect = table.visualRect(index)
        option.palette = table.palette()
        option.font = table.font()
        center = option.rect.center()

        release_event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(center),
            QPointF(center),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        delegate._pressed_index = index
        handled = delegate.editorEvent(
            release_event,
            table.model(),
            option,
            index,
        )

        self.assertTrue(handled)
        self.assertEqual(selected, [r"C:\pkg\part02\manifest.json"])

    def test_keyboard_activate_emits_select_requested(self):
        table = QTableWidget(1, 1)
        delegate = SplitStatusActionDelegate(table)
        selected: list[str] = []
        delegate.select_requested.connect(selected.append)

        item = QTableWidgetItem("")
        item.setData(
            SPLIT_ACTION_DATA_ROLE,
            split_action_item_payload(
                selectable=True,
                manifest_path=r"C:\pkg\part02\manifest.json",
                part_label="part02/03",
            ),
        )
        table.setItem(0, 0, item)

        self.assertTrue(delegate.keyboard_activate(table.model().index(0, 0)))
        self.assertEqual(selected, [r"C:\pkg\part02\manifest.json"])

    def test_keyboard_activate_ignores_cells_without_action(self):
        table = QTableWidget(1, 1)
        delegate = SplitStatusActionDelegate(table)
        selected: list[str] = []
        delegate.select_requested.connect(selected.append)

        table.setItem(0, 0, QTableWidgetItem("plain"))
        self.assertFalse(delegate.keyboard_activate(table.model().index(0, 0)))
        self.assertEqual(selected, [])

    def test_main_window_routes_enter_key_to_split_select(self):
        window = MainWindow()
        try:
            invoked: list[str] = []
            window._select_split_manifest = lambda path: invoked.append(path)
            table = window.split_status_table
            item = QTableWidgetItem("")
            item.setData(
                SPLIT_ACTION_DATA_ROLE,
                split_action_item_payload(
                    selectable=True,
                    manifest_path=r"C:\pkg\part02\manifest.json",
                    part_label="part02/03",
                ),
            )
            table.setRowCount(1)
            table.setItem(0, 0, item)
            table.setCurrentCell(0, 0)

            QTest.keyClick(table, Qt.Key.Key_Return)
            self.assertEqual(invoked, [r"C:\pkg\part02\manifest.json"])

            invoked.clear()
            table.setItem(0, 0, QTableWidgetItem("plain"))
            QTest.keyClick(table, Qt.Key.Key_Space)
            self.assertEqual(invoked, [])
        finally:
            gui_test_support.close_main_window(window)
            window.deleteLater()


if __name__ == "__main__":
    unittest.main()
