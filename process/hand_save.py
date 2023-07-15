import csv
import os
from STESDK import FaceDetector, FaceAligner, FaceExtractor
import cv2


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


# 读取人脸函数
image = cv2.imread('face_vista/gty.jpg')

# 提取人脸特征
feature = extract_feature(image)

# 设置数据库位置并保存图像
image_folder_path = '../data/image_database'
image_name = "gty.jpg"  # 文件名改为人脸检测图像对应的名字
person_id = "gty"

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
