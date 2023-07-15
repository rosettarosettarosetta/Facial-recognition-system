import cv2
import time
from STESDK import FaceAttribute

# 初始化人脸属性检测器
face_attribute = FaceAttribute()

# 打开摄像头
cap = cv2.VideoCapture(0)

count = 0
start_time = time.time()

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()

    # 进行人脸属性检测
    attribute = face_attribute.fetchAttribute(frame)

    # print(attribute)
    # 情绪
    print(attribute['eye'])
    #
    # if attribute['mouth'] == 'open':
        #count += 1

    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print("张嘴次数:", count)

# 释放摄像头和销毁窗口
cap.release()
cv2.destroyAllWindows()
