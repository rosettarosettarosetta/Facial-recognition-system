import subprocess
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QFont, QPixmap, QIcon
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton
from Face_recognition import FaceRecognition
from IrregularitiesWindow import IrregularitiesWindow
# from IrregularitiesWindowCopy import IrregularitiesWindow
from real_gesture import *


class FaceRecognitionUI(QDialog):
    def __init__(self):
        super().__init__()  # 调用 QDialog 的构造函数以完成初始化

        # 创建字体初始化
        font_base = QFont("楷体", 16, QFont.Bold)
        # font.setItalic(True) #斜体
        palette = self.palette()  # 创建一个新的调色板对象palette
        # palette.setColor(self.backgroundRole(), Qt.red)
        self.setPalette(palette)

        # 背景
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
        # self.setStyleSheet("background-color: #92d0d0;")
        self.setFixedSize(800, 600)

        # 图片
        self.image_label_logo = QLabel(self)  # QLabel控件
        self.image_label_logo.setAlignment(Qt.AlignCenter)
        pixmap_logo = QPixmap('logo.jpg')
        pixmap_logo = pixmap_logo.scaled(pixmap_logo.width() / 2, pixmap_logo.height() / 2)
        self.image_label_logo.setPixmap(pixmap_logo)
        self.image_label_logo.setGeometry(0, 0, pixmap_logo.width(), pixmap_logo.height())

        # 字
        info_label = QLabel("SBDT团队", self)
        font_team = QFont("楷体", 14, QFont.Normal)
        info_label.setStyleSheet("color: #ffffff;")
        info_label.setFont(font_team)
        info_label.move(700, 560)

        # 创建信息栏
        info_label = QLabel("设备启动核验", self)
        font_team = QFont("黑躰", 40, QFont.Bold)
        info_label.setStyleSheet("color: #202929;")
        info_label.setFont(font_team)
        info_label.move(250, 30)

        # 创建信息栏
        self.info_label_l = QLabel("识别中，请将脸部对准摄像头", self)
        self.info_label_l.move(250, 500)
        self.info_label_l.setStyleSheet("color: #f5f5f5;")
        self.info_label_l.setStyleSheet("background-color: rgba(192, 192, 192, 128);")

        self.i = 0
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
        self.fr = FaceRecognition()
        self.camera = cv2.VideoCapture(0)
        self.output = None
        self.process = subprocess.Popen(['python', 'Face_recognition.py'], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE)
        self.open_camera_and_start_timer()
        self.img = None
        # 启动另一个Python进程，并在其中执行相应的Python脚本




    def open_camera_and_start_timer(self):
        # 创建摄像头读取器
        # 创建定时器
        self.Stimer = QTimer(self)
        self.Stimer.timeout.connect(self.onTimer)
        self.Stimer.start(100)  # 设置定时器的时间间隔为1秒

    def onTimer(self):
        ret, image = self.camera.read()
        self.img = image
        if not ret:
            return

        self.output = self.fr.detect(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.output == 'gty' or self.output == 'sc':  # 人脸
            id = self.output
            if id == 'gty':
                id = '耿天羽'
            if id == 'sc':
                id = '沈辰'
            self.change_label_text( id + "识别成功! 设备启动中..")
            self.i += 1
            if self.i > 3:

                self.Stimer.stop()
                self.output = None
                image = None
                self.open_page2()
        else:
            self.i = 0
            self.change_label_text("识别中，请将脸部对准摄像头\n")
        #            self.output = None
        #           image = None
        #            self.open_page2()

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

    def open_page2(self):
        self.camera.release()
        print('release  sucess')
        self.window2 = IrregularitiesWindow(self)
        self.window2.show()
        # 将当前窗口隐藏
        self.hide()
        # 等待新窗口关闭
        self.window2.exec()
        # 删除新窗口并显示当前窗口
        self.window2.deleteLater()
        self.camera = cv2.VideoCapture(0)
        self.show()

    def closeEvent(self, event):
        self.parent.camera.release()
        event.accept()

    def change_label_text(self, new_text):
        self.info_label_l.setText(new_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # QApplication.setQuitOnLastWindowClosed(False)
    style = QStyleFactory.create("Fusion")
    app.setStyle(style)
    face_recognition_ui = FaceRecognitionUI()
    face_recognition_ui.show()

    sys.exit(app.exec())
