import time
import cv2
from STESDK import FaceDetector

# 使用OpenCV的VideoCapture打开摄像头获取图片，设置视频源为0，设置分辨率为700
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# 创建人脸检测对象
face_detector = FaceDetector()

total = 0

while True:
    # 从摄像头读取一帧图片
    ret, frame = cap.read()
    # 复制当前帧的图像，以便在其上绘制矩形，避免影响原图frame对象
    img = frame.copy()

    # 使用人脸检测器检测人脸，并返回人脸的矩形坐标，返回的坐标为左上角和右下角的坐标
    rects = face_detector.detect(img)

    # 如果检测到正好一个人脸
    if len(rects) == 1:
        # 在图像上绘制一个矩形框，标识出人脸的位置
        face = cv2.rectangle(frame, (rects[0][0], rects[0][1]), (rects[0][2], rects[0][3]), (0, 0, 255), 1)

        # 计算矩形框的长和宽
        lx = rects[0][3] - rects[0][1]
        ly = rects[0][2] - rects[0][0]
        print(lx, ly)

        # 如果矩形框的长和宽都大于140，避免离摄像头太远，检测不清晰的情况
        if lx > 140 and ly > 140:
            total += 1
            print(total)

        # 显示绘制了矩形框的图像
        cv2.imshow('result', face)
        cv2.waitKey(30)

    else:
        cv2.imshow('result', img)
        cv2.waitKey(30)

    # 如果total达到60
    # if total == 60:
    #     # 保存当前图像，并使用当前时间作为文件名
    #     filename = time.strftime("./data/image/%Y-%m-%d-%H_%M_%S", time.localtime()) + '.jpg'
    #     cv2.imwrite(filename, img)
    #     break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

# 等待一段时间，确保资源得到释放
time.sleep(5)
