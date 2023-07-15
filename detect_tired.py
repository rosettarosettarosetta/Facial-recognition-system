import cv2
import time
from STESDK import Gaze, AngleEstimatorCNN, FaceDetector, FaceAligner, FaceAttribute

eye = Gaze()
angle_estimator = AngleEstimatorCNN()
face_detector = FaceDetector()
face_aligner = FaceAligner()
face_attribute = FaceAttribute()

cap = cv2.VideoCapture(0)

blink_count = 0
frame_count = 0
blink_threshold = 0.5
time_limit = 10
start_time = time.time()
show_tired_message = False
continuous_closed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blink detection
    blink_prob = eye.gazeimg(frame)
    if blink_prob[0] < blink_threshold:
        frame_count += 1
    else:
        if frame_count >= 3:
            blink_count += 1
            frame_count = 0
        else:
            frame_count = 0

    # Head pose estimation
    rects = face_detector.detect(frame)
    if len(rects) > 0:
        key_points = face_aligner.align(frame, rects[0])
        pose = angle_estimator.estimate(key_points)

    # Fatigue detection based on eye status
    attribute = face_attribute.fetchAttribute(frame)
    eye_status = attribute['eye']
    if eye_status == 'close':
        if continuous_closed:
            count += 1
        else:
            continuous_closed = True
            count = 1
    else:
        continuous_closed = False
        count = 0

    # Conditions for displaying tired warning message
    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        if (pose[1] > 20 and count > 10) or (pose[1] > 20 and blink_count >= 3):
            show_tired_message = True
        else:
            show_tired_message = False
        start_time = time.time()

    if show_tired_message:
        cv2.putText(frame, "Tired, warning!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
