# 获得手指关键点
import STESDK
import cv2
from STESDK import HandDetector, HandAligner

hand_detector = HandDetector()
hand_aligner = HandAligner()
frame = cv2.imread(r"C:\Users\86195\Desktop\contest-phone\data\hand\gty\frame_30.jpg")
print("frame size is: ", frame.shape)
rect = hand_detector.detect(frame)
points = hand_aligner.align(frame,rect[0])
points_frame = hand_aligner.render(frame,points)
# cv2.imshow("face points",points_frame)

rect_frame = hand_detector.render(frame,rect)
STESDK.imshow("face rect",rect_frame)

k = cv2.waitKey(0)
if k == ord("q"):
 cv2.destroyAllWindows()