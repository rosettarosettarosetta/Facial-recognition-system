import cv2
from STESDK import HandAction

hand_action = HandAction()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # action = hand_action.fetchAction(frame)
        action = hand_action.fetchMovement(frame)
        cv2.putText(frame, str(action), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
