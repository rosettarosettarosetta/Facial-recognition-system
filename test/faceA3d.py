# FaceA3d 的算法主要是基于视频流的，依赖于每帧之间的差异来检测运动的特征，所以在静态的照片中可能不太有效。
import time
import numpy as np
import cv2
from STESDK import FaceA3d

face3d_aligner = FaceA3d()
cap = cv2.VideoCapture(r"C:\Users\86195\Desktop\contest-phone\gty_face.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video stream")
        break

    try:
        res = face3d_aligner.align(frame)

        # 把关键点标记在图片上
        rendered_frame = face3d_aligner.render(frame,res)
        cv2.imshow("3dfacepoint",rendered_frame)
        cv2.imwrite("rendered_frame.jpg", rendered_frame)
        print(face3d_aligner.render(frame,res))

        k = cv2.waitKey(1)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.2)
