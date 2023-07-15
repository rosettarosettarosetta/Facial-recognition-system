import cv2
from STESDK import FaceDetector, FaceAligner

# 初始化人脸检测器和对齐器
face_detector = FaceDetector()
face_aligner = FaceAligner()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()

    # 进行人脸检测
    rects = face_detector.detect(frame)

    if len(rects) > 0:
        # 获取第一个检测到的人脸框
        rect = rects[0]

        # 进行人脸对齐
        key_points = face_aligner.align(frame, rect)

        # 裁剪、旋转、缩放脸部图像
        face_img = face_aligner.crop_face(frame, key_points)

        # 保存修剪后的图像
        # cv2.imwrite('output.jpg', face_img)

        print("face rect is; ", rect)
        print("face keypoints is; ", (key_points))
        print("face keypoint length is: ", len(key_points))

        # 在图像上绘制人脸框和关键点
        rect_frame = face_detector.render(frame, rects)
        points_frame = face_aligner.render(rect_frame, key_points)

        cv2.imshow("Face", points_frame)
    else:
        cv2.imshow("Face", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和销毁窗口
cap.release()
cv2.destroyAllWindows()


