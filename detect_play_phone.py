import time
import cv2
import STESDK
from STESDK import AngleEstimatorCNN, FaceDetector, FaceAligner, HandDetector, HandAligner, HandTracker
from yolov7.detect_phone import load_yolo_model, yolo_detect_frame

import os
import sys

# 将项目根目录添加到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

weights_path = 'yolov7/weights/result_050702.pt'
img_size = 640
device = '0'  # 使用GPU设备编号，如果使用CPU，请设置为 'cpu'

# 初始化手部检测器、对齐器和跟踪器
hand_detector = HandDetector()
hand_aligner = HandAligner()
hand_tracker = HandTracker()

# 在while循环外部初始化计数器和时间戳
start_time_no_hand = time.time()
no_hand_duration = 0

# 初始化YOLOv7模型
model, device, half = load_yolo_model(weights_path, img_size, device)

angle_estimator = AngleEstimatorCNN()
face_detector = FaceDetector()
face_aligner = FaceAligner()

conf_thres = 0.60  # 检测的置信度阈值，您可以根据需要调整
iou_thres = 0.45  # NMS的IoU阈值，您可以根据需要调整
augment = False  # 是否进行数据增强
classes = None  # 检测的类别，设置为None表示检测所有类别
agnostic_nms = False  # 如果为True，则在NMS中对类别进行归一化处理

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

        # 进行姿态估计
        pose = angle_estimator.estimate(key_points)

        # 跟踪手部并渲染边框
        hand_rect = hand_tracker.track(frame)

        # 如果检测到手
        if hand_rect:
            no_hand_duration = 0  # 重置没有检测到手的持续时间
            if pose[1] > 15 and yolo_detect_frame(model, device, half, frame, img_size, conf_thres, iou_thres, augment,
                                                  classes, agnostic_nms):
                print("玩手机")
                cv2.putText(frame, "Playing Phone!Warning!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            render_frame = angle_estimator.render(frame, rect, pose)
            STESDK.imshow("Face Monitoring", render_frame)
        else:
            # 计算没有检测到手的持续时间
            no_hand_duration += time.time() - start_time_no_hand
            start_time_no_hand = time.time()  # 更新开始时间
            if pose[1] > 15:
                if no_hand_duration > 30:  # 如果连续没有检测到手的时间超过5秒
                    cv2.putText(frame, "Playing Phone!Warning!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Put your hands on the desk!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            STESDK.imshow("Face Monitoring", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和销毁窗口
cap.release()
cv2.destroyAllWindows()
