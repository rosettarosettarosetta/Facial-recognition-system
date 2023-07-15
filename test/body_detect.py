import cv2
from STESDK import BodyDetector, BodyAligner

# 创建人体检测器和关键点检测器
body_detector = BodyDetector()
body_aligner = BodyAligner()#人体关键点

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # 人体位置检测
    rects = body_detector.detect(frame)

    # 对每个检测到的人体进行关键点检测
    for rect in rects:
        points = body_aligner.align(frame, rect)

        # 把关键点标记在图片上
        frame = body_aligner.render(frame, points)

    # 获取图片大小
    height, width, _ = frame.shape

    # 调整图片大小以适应显示器分辨率
    resized_frame = cv2.resize(frame, (width // 2, height // 2))

    # 显示图片
    cv2.imshow("body keypoints", resized_frame)

    # 按下 "q" 键退出循环
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
