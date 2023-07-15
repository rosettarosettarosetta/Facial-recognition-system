import cv2
import time
from STESDK import FaceAttribute

# 初始化人脸属性检测器
face_attribute = FaceAttribute()

# 打开摄像头
cap = cv2.VideoCapture(0)

count = 0
start_time = time.time()
continuous_closed = False

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()

    # 进行人脸属性检测
    attribute = face_attribute.fetchAttribute(frame)

    # 获取眼睛状态
    eye_status = attribute['eye']

    # 检测闭眼
    if eye_status == 'close':
        if continuous_closed:
            count += 1
        else:
            continuous_closed = True
            count = 1
    else:
        continuous_closed = False
        count = 0

    # 判断闭眼次数是否超过阈值
    if count > 3:
        # 在图像上显示 "Tired, need resting"
        cv2.putText(frame, "Tired, need resting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和销毁窗口
cap.release()
cv2.destroyAllWindows()
