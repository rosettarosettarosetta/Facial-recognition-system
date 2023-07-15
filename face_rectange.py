import cv2
from STESDK import FaceDetector

face_detector = FaceDetector()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # 人脸检测
        rect = face_detector.detect(frame)

        # 在帧上渲染人脸矩形框
        rect_frame = face_detector.render(frame, rect)

        # 显示帧
        cv2.imshow("face rect", rect_frame)

        # 等待按键
        k = cv2.waitKey(30)
        # 如果按下 'q' 键，退出循环
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    else:
        print("frame is empty")

# 释放摄像头资源
cap.release()
