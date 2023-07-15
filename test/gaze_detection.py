import cv2
from STESDK import Gaze
import time

eye = Gaze()
cap = cv2.VideoCapture(0)

blink_count = 0
frame_count = 0
blink_threshold = 0.5
time_limit = 10
start_time = time.time()
show_tired_message = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blink_prob = eye.gazeimg(frame)

    if blink_prob[0] < blink_threshold:
        frame_count += 1
    else:
        if frame_count >= 3:
            blink_count += 1
            frame_count = 0
        else:
            frame_count = 0

    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        if blink_count >= 3:
            show_tired_message = True
        else:
            show_tired_message = False
        start_time = time.time()
        blink_count = 0

    if show_tired_message:
        cv2.putText(frame, "Tired, need resting", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Blink count: {blink_count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
