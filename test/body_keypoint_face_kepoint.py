import cv2
from STESDK import FaceDetector, FaceAligner, BodyDetector, BodyAligner

face_detector = FaceDetector()
face_aligner = FaceAligner()
body_detector = BodyDetector()
body_aligner = BodyAligner()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 人脸检测和关键点检测
    face_rects = face_detector.detect(frame)
    if len(face_rects) > 0:
        face_key_points = face_aligner.align(frame, face_rects[0])
        face_frame = face_aligner.render(frame, face_key_points)
    else:
        face_frame = frame

    # 人体检测和关键点检测
    body_rect = body_detector.detect(frame)
    if len(body_rect) > 0:
        body_key_points = body_aligner.align(frame, body_rect[0])
        body_frame = body_aligner.render(face_frame, body_key_points)
    else:
        body_frame = face_frame

    # 当人脸关键点检测不到，但可以检测到人体时，显示 "Tired, Warning!"
    if len(face_rects) == 0 and len(body_rect) > 0:
        cv2.putText(body_frame, "Tired, Warning!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", body_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
