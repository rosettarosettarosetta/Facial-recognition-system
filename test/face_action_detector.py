import cv2
from STESDK import FaceActionDetector

live_action = FaceActionDetector(1)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    _,frame = cap.read()
    if _:
        res = live_action.detect(frame)
        cv2.putText(frame,str(res),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow("hand rect",frame)
        k = cv2.waitKey(30)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
