import cv2
from STESDK import Segmenter
from STESDK import BodyDetector

body_detector = BodyDetector()
body_segmenter = Segmenter()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()

    # 进行人体检测
    rects = body_detector.detect(frame)

    if len(rects) > 0:
        # 获取第一个检测到的人体框
        rect = rects[0]

        # 进行人体分割
        seg_frame = body_segmenter.segment(frame, rect)

        # 显示分割后的帧
        cv2.imshow("seg.jpg", seg_frame)
    else:
        # 如果没有检测到人体，显示原始帧
        cv2.imshow("seg.jpg", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和销毁窗口
cap.release()
cv2.destroyAllWindows()
