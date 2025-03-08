import numpy as np

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Slot, Signal


class ColorList(QTableWidget):

    remove_color = Signal(int)

    def __init__(self, parent=None):
        QTableWidget.__init__(self, 0, 4, parent)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setHorizontalHeaderLabels(["ID", "notes", "Lab mean", "RGB mean"])

    @Slot()
    def addColor(self, identifier, lab_mean, lab_stdev, rgb_mean):
        self.insertRow( self.rowCount() )

        bg_color = QColor(*[int(x) for x in rgb_mean])
        text_color = QColor(0,0,0) if np.mean(rgb_mean) > 126 else QColor(255,255,255)

        row_content = [
            str(identifier),
            str("    "),
            " ".join(['{:.2f}'.format(x) for x in lab_mean]),
            " ".join(['{:.2f}'.format(x) for x in rgb_mean])
        ]

        for i,s in enumerate(row_content):
            item = QTableWidgetItem(s)
            if not i == 1: item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            item.setBackground(bg_color)
            item.setForeground(text_color)
            self.setItem( self.rowCount()-1, i, item )

    @Slot()
    def removeSelectedRow(self):
        selected_items = self.selectedItems()
        rows = [item.row() for item in selected_items]
        unique_rows = list(set(rows))
        unique_rows.sort(reverse=True)

        for row in unique_rows:
            index = int(self.item(row, 0).text())
            self.removeRow(row)
            self.remove_color.emit(index)
