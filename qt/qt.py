import cv2
import sys
import time
import subprocess
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont,QTextCursor, QTextImageFormat, QPixmap,QIcon
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, \
    QMessageBox, QStyleFactory, QMainWindow,QStackedWidget
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem,QWidget,QTreeView

class FaceRecognitionUI(QDialog):
    def __init__(self):
        super().__init__()  # 调用 QDialog 的构造函数以完成初始化


        # 创建字体初始化
        font_base = QFont("楷体", 16, QFont.Bold)
        # font.setItalic(True) #斜体
        palette = self.palette() #创建一个新的调色板对象palette
        #palette.setColor(self.backgroundRole(), Qt.red)
        self.setPalette(palette)

        #背景
        background_pixmap = QPixmap('R.jpg')
        background_palette = QPalette()
        background_palette.setBrush(QPalette.Window, QBrush(background_pixmap))
        # 将QPalette对象设置为窗口的调色板
        self.setPalette(background_palette)
        # 启用调色板
        self.setAutoFillBackground(True)


        # logo
        button = QPushButton()
        icon = QIcon()
        pixmap = QPixmap("logo.jpg")
        icon.addPixmap(pixmap)
        button.setIcon(icon)

        #
        self.setWindowTitle("欢迎进入商汤智能系统")
        self.setFont(font_base)
        #self.setStyleSheet("background-color: #92d0d0;")
        self.resize(800, 600)

        #图片
        self.image_label_logo = QLabel(self) #QLabel控件
        self.image_label_logo.setAlignment(Qt.AlignCenter)
        pixmap_logo = QPixmap('logo.jpg')
        pixmap_logo = pixmap_logo.scaled(pixmap_logo.width()/2, pixmap_logo.height()/2)
        self.image_label_logo.setPixmap(pixmap_logo)
        self.image_label_logo.setGeometry(0, 0, pixmap_logo.width(), pixmap_logo.height())

        #字
        info_label = QLabel("SBDT团队", self)
        font_team = QFont("楷体", 14, QFont.Normal)
        info_label.setStyleSheet("color: #ffffff;")
        info_label.setFont(font_team)
        info_label.move(700, 560)

        # 创建信息栏
        info_label = QLabel("设备启动核验", self)
        font_team = QFont("黑躰",40, QFont.Bold)
        info_label.setStyleSheet("color: #202929;")
        info_label.setFont(font_team)
        info_label.move(250, 30)

        # 创建信息栏
        info_label = QLabel("识别中，请将脸部对准摄像头", self)
        info_label.move(250,500)
        info_label.setStyleSheet("color: #f5f5f5;")
        info_label.setStyleSheet("background-color: rgba(192, 192, 192, 128);")

        enter_button_style = """
                               QPushButton
                               {
                               color:white;
                               background-color:rgb(14 , 150 , 254);
                               border-radius:5px;
                               }

                               QPushButton:hover
                               {
                               color:white;
                               background-color:rgb(44 , 137 , 255);
                               }

                               QPushButton:pressed
                               {
                               color:white;
                               background-color:rgb(14 , 135 , 228);
                               padding-left:3px;
                               padding-top:3px;
                               }
                               """

        #enter_button = QPushButton("进入", self)
        #enter_button.move(600, 500)
        #self.enter_button.setEnabled(False)

        #enter_button.clicked.connect(self.open_page2)

        # 创建定时器
        timer = QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(50)

        # 创建摄像头读取器
        self.camera = cv2.VideoCapture(0)
    def update_image(self):
            # 读取摄像头中的图像
            ret, image = self.camera.read()
            print(image.shape)
            if not ret:
                return

            # 将OpenCV图像转换为Qt图像
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将Qt图像转换为Qt图元并显示
            self.image_label_view = QLabel(self)
            self.image_label_view.move(150, 100)  # 先将控件移动到指定位置
            pixmap = QPixmap.fromImage(qt_image)
            large_pixmap = 500
            new_pixmap = pixmap.scaled(large_pixmap, int(pixmap.height() * large_pixmap / pixmap.width()))
            self.image_label_view.setPixmap(new_pixmap)
            self.image_label_view.show()


    def on_register_button_clicked(self):
            # 停止读取摄像头
            #self.camera.release()

            # 显示注册对话框
            register_dialog = RegisterDialog(self)
            register_dialog.exec_()

            # 重新打开摄像头
            self.camera = cv2.VideoCapture(0)


    def open_page2(self):
        self.window2 = RegisterDialog(self)
        self.window2.show()
        # 将当前窗口隐藏
        self.hide()
        # 等待新窗口关闭
        self.window2.exec_()
        # 删除新窗口并显示当前窗口
        self.window2.deleteLater()
        self.show()

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("欢迎进入商汤智能系统")
        self.resize(800, 600)

        # 背景
        background_pixmap = QPixmap('R.jpg')
        background_palette = QPalette()
        background_palette.setBrush(QPalette.Window, QBrush(background_pixmap))
        # 将QPalette对象设置为窗口的调色板
        self.setPalette(background_palette)
        # 启用调色板
        self.setAutoFillBackground(True)

        # 图片
        self.image_label_logo = QLabel(self)  # QLabel控件
        self.image_label_logo.setAlignment(Qt.AlignCenter)
        pixmap_logo = QPixmap('logo.jpg')
        pixmap_logo = pixmap_logo.scaled(pixmap_logo.width() / 2, pixmap_logo.height() / 2)
        self.image_label_logo.setPixmap(pixmap_logo)
        self.image_label_logo.setGeometry(0, 0, pixmap_logo.width(), pixmap_logo.height())

        # 创建信息栏
        info_label = QLabel("员工违规检测", self)
        font_team = QFont("楷体", 40, QFont.Bold)
        info_label.setStyleSheet("color: #202929;")
        info_label.setFont(font_team)
        info_label.move(300, 30)

        # 创建按钮
        exit_button_style = """
                        QPushButton
                        {
                        color:white;
                        background-color:rgb(14 , 150 , 254);
                        border-radius:5px;
                        }

                        QPushButton:hover
                        {
                        color:white;
                        background-color:rgb(44 , 137 , 255);
                        }

                        QPushButton:pressed
                        {
                        color:white;
                        background-color:rgb(14 , 135 , 228);
                        padding-left:3px;
                        padding-top:3px;
                        }
                        """
        self.exit_button = QPushButton("退出", self)
        self.exit_button.move(600,500)
        self.exit_button.clicked.connect(self.close)



        '''
        self.tableWidget = QTableWidget(self)  # 添加self作为父部件
        self.tableWidget.move(50, 100)  # 这里可以改变位置
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["时间",  "示例"])
        self.tableWidget.setShowGrid(True)  #分割线
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 100)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        def add_row(self, data):
            row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_count)
            for column, item in enumerate(data):
                self.tableWidget.setItem(row_count, column, QTableWidgetItem(str(item)))
                '''
        self.browser=QTextBrowser(self)
        self.browser.move(50, 100)
        self.browser.resize(300, 300)
        #self.browser.setStyleSheet('background: transparent;border - width: 0;border - style: outset')
        self.browser.setStyleSheet('background-color:rgba(255,255,255,.5)')
        self.browser.setPlaceholderText("程序启动至今无违规记录")
        #palette.setColor(QPalette::Normal, QPalette::PlaceholderText, Qt::red);
        '''
        self.connect(self.lineedit,SIGNAL("returnPressed()"),self.updateUI)
        def updateUI (self):
            try :
                datetime = QtCore.QDateTime.currentDateTime() 
            except:self.browser.append(
                "<font color=red>%s is invalid!</font>"% text 
            )'''



        def closeEvent(self, event):
            self.parent.camera.release()
            event.accept()


        #self.exit_button.setStyleSheet(exit_button_style)


        # 创建定时器
        timer = QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(50)

        # 创建摄像头读取器
        self.camera = cv2.VideoCapture(0)

    def update_image(self):
        # 读取摄像头中的图像
        ret, image = self.camera.read()
        if not ret:
            return

        # 将OpenCV图像转换为Qt图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # 将Qt图像转换为Qt图元并显示
        self.image_label_view = QLabel(self)
        self.image_label_view.move(200, 100)  # 先将控件移动到指定位置
        pixmap = QPixmap.fromImage(qt_image)
        large_pixmap = 300
        new_pixmap = pixmap.scaled(large_pixmap, int(pixmap.height() * large_pixmap / pixmap.width()))
        self.image_label_view.move(430, 130)
        self.image_label_view.setPixmap(new_pixmap)
        self.image_label_view.show()



    def on_register_button_clicked(self):
        # 停止读取摄像头
        self.camera.release()

        # 显示注册对话框
        register_dialog = RegisterDialog(self)
        register_dialog.exec_()

        # 重新打开摄像头
        self.camera = cv2.VideoCapture(0)


'''

        self.do_button = QPushButton("操作", self)
        self.do_button.clicked.connect(self.on_register_button_clicked)
        self.do_button.move(670, 500)

        def on_register_button_clicked(self):
            datetime = QtCore.QDateTime.currentDateTime()
            row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_count)
            self.tableWidget.setItem(row_count, 0, QtWidgets.QTableWidgetItem(datetime.toString("yyyy.MM.dd hh:mm")))

'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #QApplication.setQuitOnLastWindowClosed(False)
    style = QStyleFactory.create("Fusion")
    app.setStyle(style)
    face_recognition_ui = FaceRecognitionUI()
    face_recognition_ui.show()

    sys.exit(app.exec())