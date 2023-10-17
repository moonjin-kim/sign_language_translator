import sys
import urllib.request
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5.QtCore import QCoreApplication
from TranslationWindow import TranslationWindow

form_class = uic.loadUiType("./ui2/home.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.tutorial.clicked.connect(self.loadImageFromFile)
        self.start_Translation.clicked.connect(self.start_Translation_window)
        self.close_btn.clicked.connect(QCoreApplication.instance().quit)

    def loadImageFromFile(self) :
        #QPixmap 객체 생성 후 이미지 파일을 이용하여 QPixmap에 사진 데이터 Load하고, Label을 이용하여 화면에 표시
        img = cv2.imread("test.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)
        qImg = QtGui.QImage(img.data, 640, 480, 640*480, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.today_word.setPixmap(pixmap)
        self.today_word.resize(pixmap.width(), pixmap.height())
        self.today_word.show()

    def start_Translation_window(self) :
        self.hide()
        self.second = TranslationWindow()
        self.second.exec()
        self.show()
        
    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                    QMessageBox.Yes|QMessageBox.No)

        if re == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()  


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_() 