import csv
import os
import time
import cv2
from STESDK import FaceDetector, FaceTracker, LiveVerification, FaceAligner, FaceExtractor


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

# 使用OpenCV的VideoCapture打开摄像头获取图片，设置视频源为0，设置分辨率为700
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# 创建人脸检测对象
face_detector = FaceDetector()
face_tracker = FaceTracker()

# 活体检测
live_verify = LiveVerification()

count = 0
total = 0

while True:
    # 从摄像头读取一帧图片
    ret, frame = cap.read()
    # 复制当前帧的图像，以便在其上绘制矩形，避免影响原图frame对象
    img = frame.copy()

    # 使用人脸检测器检测人脸，并返回人脸的矩形坐标，返回的坐标为左上角和右下角的坐标
    # rects = face_detector.detect(img)
    rects = face_tracker.track(img)

    # 如果检测到正好一个人脸
    if len(rects) == 1:

        # 在图像上绘制一个矩形框，标识出人脸的位置
        face = cv2.rectangle(frame, (rects[0][0], rects[0][1]), (rects[0][2], rects[0][3]), (0, 0, 255), 1)

        # 进行活体检测
        res = live_verify.verify(frame)

        print(res[0])

        # res是一个列表，仅有一个数，res[0]是非活体的概率
        if res[0] < 0.4 :
            text = "Live Face Detected"
            color = (0, 255, 0)  # 绿色

            count += 1

            # 读取人脸函数
            image = frame.copy()

            # 提取人脸特征
            feature = extract_feature(image)

            # 设置数据库位置并保存图像
            image_folder_path = '../data/image_database/sc'
            image_name = f"sc{count}.jpg"
            person_id = "sc"

            # 保存图片到 ./data/image_database 文件夹
            cv2.imwrite(os.path.join(image_folder_path, image_name), image)

            # 指定数据库文件路径
            # 将图片名称，人名ID、人脸特征保存到数据库中data/dataset.process
            database_path = '../data/csv/dataset.csv'
            datafile = os.path.exists(database_path)

            if datafile is not True:
                # 如果数据库文件不存在，则创建数据库文件，并写入表头
                with open(database_path, 'w', newline="") as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['image_name', 'person_id', 'feature'])
                    spamwriter.writerow([image_name] + [person_id] + [str(feature)])
            else:
                # 如果数据库文件存在，则直接写入数据
                with open(database_path, 'a', newline="") as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([image_name] + [person_id] + [str(feature)])
            print('Database updated successfully!')

        else:
            text = "No Live Face Detected"
            color = (0, 0, 255)  # 红色

        org = (100, 100)  # 文本的左下角坐标
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        fontScale = 1  # 字体大小
        thickness = 2  # 字体粗细
        image = cv2.putText(face, text, org, font, fontScale, color, thickness)  # 在图像上绘制文本
        cv2.imshow('result', image)  # 显示图像

    else:
        image = img.copy()  # 复制图像
        text = 'No Face Detected!Please move closer to the camera. '  # 要显示的文本
        org = (100, 100)  # 文本的左下角坐标
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        fontScale = 1  # 字体大小
        color = (0, 0, 255)  # 字体颜色，红色
        thickness = 2  # 字体粗细
        image = cv2.putText(image, text, org, font, fontScale, color, thickness)  # 在图像上绘制文本
        cv2.imshow('result', image)  # 显示图像

    key = cv2.waitKey(30)  # 将按键值存储在变量key中
    if key & 0xFF == ord('q'):
        break
