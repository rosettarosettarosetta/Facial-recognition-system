import STESDK
import cv2
from STESDK import HandTracker, HandDetector

hand_tracker = HandTracker()
hand_detector = HandDetector()
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    if _:
        rects = hand_tracker.track(frame)
        rect_frame = hand_detector.render(frame,rects)
        STESDK.imshow("hand rect",rect_frame)
        k = cv2.waitKey(30)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
        print("hand rect is; ", rects)
    else:
        print("frame is empty")
