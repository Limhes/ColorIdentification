# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLayout, QPushButton,
    QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(800, 600)
        self.verticalLayout = QVBoxLayout(Widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.frame = QFrame(Widget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.frame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.spnNumColors = QSpinBox(self.frame)
        self.spnNumColors.setObjectName(u"spnNumColors")
        self.spnNumColors.setMinimum(1)
        self.spnNumColors.setMaximum(5)

        self.gridLayout.addWidget(self.spnNumColors, 0, 2, 1, 1)

        self.btnAnalyzeColors = QPushButton(self.frame)
        self.btnAnalyzeColors.setObjectName(u"btnAnalyzeColors")

        self.gridLayout.addWidget(self.btnAnalyzeColors, 0, 3, 1, 1)

        self.btnLoadFile = QPushButton(self.frame)
        self.btnLoadFile.setObjectName(u"btnLoadFile")

        self.gridLayout.addWidget(self.btnLoadFile, 0, 0, 1, 1)

        self.btnFindStamps = QPushButton(self.frame)
        self.btnFindStamps.setObjectName(u"btnFindStamps")

        self.gridLayout.addWidget(self.btnFindStamps, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.frame)

        self.frame_2 = QFrame(Widget)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.lblImageDisplay = QLabel(self.frame_2)
        self.lblImageDisplay.setObjectName(u"lblImageDisplay")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lblImageDisplay.sizePolicy().hasHeightForWidth())
        self.lblImageDisplay.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.lblImageDisplay)

        self.tblColors = QTableWidget(self.frame_2)
        self.tblColors.setObjectName(u"tblColors")
        sizePolicy1.setHeightForWidth(self.tblColors.sizePolicy().hasHeightForWidth())
        self.tblColors.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.tblColors)


        self.verticalLayout.addWidget(self.frame_2)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.btnAnalyzeColors.setText(QCoreApplication.translate("Widget", u"Analyze colors", None))
        self.btnLoadFile.setText(QCoreApplication.translate("Widget", u"Load file", None))
        self.btnFindStamps.setText(QCoreApplication.translate("Widget", u"Detect stamps", None))
        self.lblImageDisplay.setText("")
    # retranslateUi

