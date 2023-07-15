import cv2
import STESDK
import numpy as np
from STESDK import HandDetector, HandAligner, HandTracker, BodyDetector, BodyAligner
from yolov7.detect_phone import load_yolo_model, yolo_detect_frame,rec_phone
import os
import sys

class RealGesture( ):
    def __init__(self):
        # 将项目根目录添加到sys.path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        self.frame_counter = 0


        # 初始化YOLOv7模型
        self.weights_path = '../yolov7/weights/result_050702.pt'
        self.img_size = 640
        self.device = '0'  # 使用GPU设备编号，如果使用CPU，请设置为 'cpu'
        self.model, self.device, self.half = load_yolo_model(self.weights_path, self.img_size, self.device)


        # 初始化手部检测器、对齐器和跟踪器
        self.hand_detector = HandDetector()
        self.hand_aligner = HandAligner()
        self.hand_tracker = HandTracker()

        # 初始化人体检测器和对齐器
        self.body_detector = BodyDetector()
        self.body_aligner = BodyAligner()

    def is_phone_call_gesture(self,face_points, hand_avg_point, threshold):
        # 计算手部关键点平均位置与人脸关键点的距离
        distances = [np.sqrt((point[0] - hand_avg_point[0]) ** 2 + (point[1] - hand_avg_point[1]) ** 2) for point in
                     face_points]
        # 如果距离小于阈值，则判断为打电话姿态
        return any(distance < threshold for distance in distances)


    def gesture(self,frame):

        if frame is None:
            print("Received an empty frame")
            return

        # print('3')
        # self.model, self.device, self.half = load_yolo_model(self.weights_path, self.device)#


        img = cv2.resize(frame, (640, 480))

        # 打开摄像头
        #cap = cv2.VideoCapture(0)

        # 设置分辨率
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 设置距离阈值，您可以根据实际情况调整这个值
        threshold = 150

        # 检测间隔（以帧为单位）
        detection_interval = 5

        # frame_counter = 0

        conf_thres = 0.50  # 检测的置信度阈值，您可以根据需要调整
        iou_thres = 0.45  # NMS的IoU阈值，您可以根据需要调整
        augment = False  # 是否进行数据增强
        classes = None  # 检测的类别，设置为None表示检测所有类别
        agnostic_nms = False  # 如果为True，则在NMS中对类别进行归一化处理

        frame = img  # 曾经是cap.read  #ret, frame = img
        ret = 1  # 增加
        if ret:
            self.frame_counter += 1

            if self.frame_counter:
                # 跟踪手部并渲染边框
                rects = self.hand_tracker.track(frame)

                # 如果检测到手
                if rects:
                    # 获取手部关键点
                    hand_points = self.hand_aligner.align(frame, rects[0])

                    # 计算关键点的平均位置
                    if len(hand_points) > 0:
                        avg_x = sum(point[0] for point in hand_points) / len(hand_points)
                        avg_y = sum(point[1] for point in hand_points) / len(hand_points)
                        hand_avg_point = (int(avg_x), int(avg_y))

                        # 在帧上绘制平均位置点
                        cv2.circle(frame, hand_avg_point, 5, (0, 255, 0), -1)

                        # 输出平均位置
                        # print(f"Average position of hand keypoints: {hand_avg_point}")

                        # 人体位置检测
                        body_rect = self.body_detector.detect(frame)

                        # 如果检测到人体
                        if body_rect:
                            # 关键点检测
                            body_points = self.body_aligner.align(frame, body_rect[0])

                            # 获取两个新的关键点（人脸关键点的两个位置信息）
                            y = (body_points[0][1] + body_points[1][1]) / 2
                            x2 = (body_points[1][0] + body_points[3][0]) / 2
                            x1 = (body_points[1][0] + body_points[2][0]) / 2
                            new_points = [(int(x1), int(y)), (int(x2), int(y))]

                            # 判断是否为打电话姿态
                            if self.is_phone_call_gesture(new_points, hand_avg_point, threshold) and yolo_detect_frame(
                                    self.model, self.device, self.half, frame, self.img_size, conf_thres,
                                    iou_thres, augment, classes,
                                    agnostic_nms):
                                detection_time, label, x1, x2, y1, y2 = rec_phone(self.model, self.device, self.half,
                                                                                  frame,
                                                                                  self.img_size, conf_thres, iou_thres,
                                                                                  augment, classes, agnostic_nms)
                                return f"帧数：{self.frame_counter},推理时延: {detection_time:.2f}ms, 标签: ({label}), 预测框坐标: ({x1:.2f}), ({x2:.2f}) , ({y1:.2f}), {y2:.2f}),安全设备：商汤科技"
                            else:
                                detection_time = rec_phone(self.model, self.device, self.half, frame, self.img_size,
                                                           conf_thres, iou_thres, augment, classes,
                                                           agnostic_nms)
                                return f"帧数：{self.frame_counter}，推理时延: {detection_time:}ms"
                else:
                    detection_time = rec_phone(self.model, self.device, self.half, frame, self.img_size,
                                               conf_thres, iou_thres, augment, classes,
                                               agnostic_nms)
                    return "没有检测到手，请确保手出现在摄像头视野内\n" \
                           f"帧数：{self.frame_counter}，推理时延: {detection_time:}ms"


