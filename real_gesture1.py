import cv2
import STESDK
import numpy as np
from STESDK import FaceDetector, HandDetector, HandAligner, HandTracker

def is_phone_call_gesture(face_rect, hand_avg_point, threshold):
    if not face_rect:
        return False
    face_left, face_top, face_width, face_height = face_rect
    face_right = face_left + face_width
    face_bottom = face_top + face_height

    return (face_left - threshold <= hand_avg_point[0] <= face_right + threshold) and (face_top - threshold <= hand_avg_point[1] <= face_bottom + threshold)

face_detector = FaceDetector()
hand_detector = HandDetector()
hand_aligner = HandAligner()
hand_tracker = HandTracker()

cap = cv2.VideoCapture(0)
threshold = 100  # 设置距离阈值，根据实际情况调整这个值

while True:
    ret, frame = cap.read()
    if ret:
        # 人脸检测
        face_rects = face_detector.detect(frame)
        if face_rects:
            face_rect = face_rects[0]
            face_frame = face_detector.render(frame, [face_rect])
        else:
            face_frame = frame

        # 手部检测
        hand_rects = hand_tracker.track(frame)
        if hand_rects:
            hand_rect = hand_rects[0]
            hand_points = hand_aligner.align(frame, hand_rect)

            if len(hand_points) > 0:
                avg_x = sum(point[0] for point in hand_points) / len(hand_points)
                avg_y = sum(point[1] for point in hand_points) / len(hand_points)
                hand_avg_point = (int(avg_x), int(avg_y))

                cv2.circle(face_frame, hand_avg_point, 5, (0, 255, 0), -1)

                if is_phone_call_gesture(face_rect, hand_avg_point, threshold):
                    print("Phone call gesture detected!")
                else:
                    print("No phone call gesture detected.")
            else:
                hand_avg_point = None
        else:
            hand_avg_point = None

        cv2.imshow("phone call detection", face_frame)
        k = cv2.waitKey(30)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    else:
        print("frame is empty")

cap.release()
