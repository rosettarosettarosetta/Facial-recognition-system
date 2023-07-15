import cv2
from STESDK import AngleEstimatorCNN, FaceDetector, FaceAligner

angle_estimator = AngleEstimatorCNN()
face_detector = FaceDetector()
face_aligner = FaceAligner()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rects = face_detector.detect(frame)

    if len(rects) > 0:
        key_points = face_aligner.align(frame, rects[0])
        pose = angle_estimator.estimate(key_points)
        render_frame = angle_estimator.render(frame, rects[0], pose)
    else:
        render_frame = frame

    print(pose[1])

    if pose[1] > 20:
        cv2.putText(render_frame, "Tired!Warning!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", render_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()