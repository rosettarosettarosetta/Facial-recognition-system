import time
import cv2
from STESDK import BodyDetector, BodyAligner

# 创建人体检测器和关键点检测器
body_detector = BodyDetector()
body_aligner = BodyAligner()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # 人体位置检测
        rect = body_detector.detect(frame)

        # 如果检测到人体
        if rect:
            # 关键点检测
            points = body_aligner.align(frame, rect[0])

            # 获取两个新的关键点
            y = (points[0][1] + points[1][1]) / 2
            x2 = (points[1][0] + points[3][0]) / 2
            x1 = (points[1][0] + points[2][0]) / 2
            new_points = [(int(x1), int(y)), (int(x2), int(y))]

            # 把关键点标记在图片上
            points_frame = body_aligner.render(frame, points)

            # 添加新的关键点
            for point in new_points:
                x, y = point
                cv2.circle(points_frame, (x, y), 5, (0, 255, 0), -1)

            # 在新关键点上添加数字标签
            for i, point in enumerate(new_points):
                x, y = point
                cv2.putText(points_frame, f"New {i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("body keypoints", points_frame)

            time.sleep(0.2)

            # 输出新关键点的坐标
            for i, point in enumerate(new_points):
                x, y = point
                print(f"New point {i}: ({x}, {y})")

        k = cv2.waitKey(30)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    else:
        print("frame is empty")
