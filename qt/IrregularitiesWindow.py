from PyQt5.QtCore import QCoreApplication
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QFont, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton
from real_gesture import *
from PySide6.QtCore import QThread, Signal


class IrregularitiesWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("欢迎进入商汤智能系统")
        self.setFixedSize(800, 600)

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
        self.exit_button.move(600, 500)
        self.exit_button.clicked.connect(self.close)


        self.browser = QTextBrowser(self)
        self.browser.move(50, 100)
        self.browser.resize(300, 300)
        # self.browser.setStyleSheet('background: transparent;border - width: 0;border - style: outset')
        #self.browser.setStyleSheet('background-color:rgba(255,255,255,.5)')
        self.browser.setStyleSheet('background-color: rgba(255, 255, 255, .5); color: black;')
        self.browser.setPlaceholderText("程序启动至今无违规记录")

        # palette.setColor(QPalette::Normal, QPalette::PlaceholderText, Qt::red);

        self.i = 0
        self.real_gesture = RealGesture()
        self.camera = cv2.VideoCapture(0)
        self.output = None

        #self.process = subprocess.Popen(['python', 'real_gesture.py'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        # 创建定时器
        self.open_camera_and_start_timer()
        self.img = None



    def open_camera_and_start_timer(self):
        self.Stimer = QTimer(self)
        self.Stimer.timeout.connect(self.update_image)
        self.Stimer.start(100)  # 设置定时器的时间间隔为1秒

    def update_image(self):
        ret, image = self.camera.read()
        if not ret:
            print('no image')
            return

        self.output = self.real_gesture.gesture(image)  # 将此行移到这里
        self.browser.append(self.output)  # 使用此行将输出添加到文本浏览器中

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

    def __del__(self):
        self.camera.release()

    def closeEvent(self, event):
        self.camera.release()

        # 停止定时器
        self.parent().Stimer.stop()

        # 关闭应用程序
        QCoreApplication.instance().quit()

        event.accept()

