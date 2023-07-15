import cv2
import STESDK
from STESDK import HandDetector, HandAligner, HandTracker

hand_detector = HandDetector()
hand_aligner = HandAligner()
hand_tracker = HandTracker()

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if ret:
        rects = hand_tracker.track(frame)
        rect_frame = hand_detector.render(frame, rects)
        STESDK.imshow("hand keypoints with average", rect_frame)

        # 如果检测到手
        if rects:
            # 获取关键点
            points = hand_aligner.align(frame, rects[0])

            # 在帧上渲染关键点
            points_frame = hand_aligner.render(frame, points)

            # 显示关键点帧
            # cv2.imshow("hand keypoints", points_frame)

            # 计算关键点的平均位置
            if len(points) > 0:  # 使用 len(points) 来检查列表是否为空
                avg_x = sum(point[0] for point in points) / len(points)
                avg_y = sum(point[1] for point in points) / len(points)
                avg_point = (int(avg_x), int(avg_y))

                # 在帧上绘制平均位置点
                cv2.circle(points_frame, avg_point, 5, (0, 255, 0), -1)
                STESDK.imshow("hand keypoints with average", points_frame)

                # 输出平均位置
                print(f"Average position of keypoints: {avg_point}")



        k = cv2.waitKey(30)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    else:
        print("frame is empty")


