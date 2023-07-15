import csv

import numpy as np
import torch
import cv2
from STESDK import FaceDetector, FaceAligner, FaceExtractor


# 定义一个人脸特征提取函数
def extract_feature(img):
    # 创建人脸检测、人脸对齐和人脸特征提取对象
    face_detector = FaceDetector()
    face_aligner = FaceAligner()
    face_extractor = FaceExtractor()
    # 步骤一：人脸检测
    rect = face_detector.detect(img)
    if len(rect) > 0:
        # 步骤二：获取人脸关键点
        keypoints = face_aligner.align(img, rect[0])
        # 步骤三：根据人脸关键点裁剪人脸位置图像
        crop_frame = face_aligner.crop_face(img, keypoints)
        # 步骤四：人脸特征提取
        feature = face_extractor.extract(crop_frame)
        return feature
    else:
        print('No face detected!')

def load_known_faces(csv_path):
    known_faces = {}
    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in csv_reader:
            name = row[1]
            feature_vector = [float(x) for x in row[2][1:-1].split(',')]
            known_faces[name] = feature_vector
    return known_faces

def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 定义主函数
def main():
    # 加载已知人脸特征
    csv_path = r'C:\Users\86195\Desktop\contest-phone\data\csv\val.process'
    known_faces = load_known_faces(csv_path)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 循环处理摄像头的每一帧
    while True:
        # 从摄像头读取一帧
        ret, frame = cap.read()

        # 如果无法读取帧，则退出循环
        if not ret:
            print("Unable to capture video")
            break

        # 提取当前帧的人脸特征
        current_feature = extract_feature(frame)

        if current_feature is not None:
            # 用于保存最佳匹配结果
            best_match = None
            best_similarity = -1

            # 遍历已知人脸特征，与当前人脸特征进行相似度计算
            for name, known_feature in known_faces.items():
                similarity = cosine_similarity(current_feature, known_feature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name

            # 如果相似度大于阈值，显示匹配到的人名
            if best_similarity > 0.5:
                print(f"Matched: {best_match}")
                cv2.putText(frame, f"name: {best_match}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),2)
            else:
                cv2.putText(frame, "Unknown", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 在窗口中显示当前帧
        cv2.imshow('frame', frame)

        # 如果用户按下了 'q' 键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
