import csv
import time
import cv2
import numpy as np
from STESDK import FaceDetector, FaceTracker, LiveVerification, FaceAligner, FaceExtractor


class FaceRecognition:
    def __init__(self):
        csv_path = r'C:\Users\86195\Desktop\contest-phone\data\csv\dataset.csv'
        self.known_faces = self.load_known_faces(csv_path)

        print('loaded known-face model...')
        self.face_detector = FaceDetector()  # 0.7
        self.face_aligner = FaceAligner()  # 0.3
        self.face_extractor = FaceExtractor()  # 0.3

        # 创建人脸检测对象
        self.face_tracker = FaceTracker()
        # 活体检测
        self.live_verify = LiveVerification()
        self.rects = None
        self.frame_count = 0

    def extract_feature(self, img, face_detector, face_aligner, face_extractor):
        # 步骤一：人脸检测
        self.rect = face_detector.detect(img)
        if len(self.rect) > 0:
            # 步骤二：获取人脸关键点
            keypoints = face_aligner.align(img, self.rect[0])
            # 步骤三：根据人脸关键点裁剪人脸位置图像
            crop_frame = face_aligner.crop_face(img, keypoints)
            # 步骤四：人脸特征提取
            feature = face_extractor.extract(crop_frame)
            return feature
        else:
            print('No face detected!')

    def load_known_faces(self, csv_path):
        known_faces = {}
        with open(csv_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='|')
            for row in csv_reader:
                name = row[1]
                feature_vector = [float(x) for x in row[2][1:-1].split(',')]
                known_faces[name] = feature_vector
        return known_faces

    def cosine_similarity(self, a, b):
        """
        计算余弦相似度
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def detect(self, frame):
        start_time = time.time()
        img = frame.copy()

        # 每隔 10 帧进行一次人脸检测
        if self.frame_count % 10 == 0:
            # 使用人脸检测器检测人脸，并返回人脸的矩形坐标，返回的坐标为左上角和右下角的坐标
            self.rects = self.face_detector.detect(img)
        # 更新帧计数器
        self.frame_count += 1

        # 如果检测到正好一个人脸
        if len(self.rects) >= 1:
            # 在图像上绘制一个矩形框，标识出人脸的位置
            # face = cv2.rectangle(frame, (self.rects[0][0], self.rects[0][1]), (self.rects[0][2], self.rects[0][3]), (0, 0, 255), 1)

            # 进行活体检测
            res = self.live_verify.verify(frame)
            print('res[0] : ' + str(res[0]))
            # res是一个列表，仅有一个数，res[0]是非活体的概率
            if res[0] < 0.5:
                text = "Live Face Detected"
                print(text)

                # 提取当前帧的人脸特征
                current_feature = self.extract_feature(frame, self.face_detector, self.face_aligner, self.face_extractor)

                if current_feature is not None:
                    # 用于保存最佳匹配结果
                    best_match = None
                    best_similarity = -1

                    # 遍历已知人脸特征，与当前人脸特征进行相似度计算
                    for name, known_feature in self.known_faces.items():
                        similarity = self.cosine_similarity(current_feature, known_feature)
                        print('similarity' + str(similarity))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name

                    # 如果相似度大于阈值，显示匹配到的人名
                    if best_similarity > 0.5:
                        text = f"{best_match}"
                    else:
                        text = "No Match Found"


            else:
                text = "NoFace"
                color = (0, 0, 255)  # 红色
            return text

        else:
            text = 'None'  # 要显示的文本
            return text





