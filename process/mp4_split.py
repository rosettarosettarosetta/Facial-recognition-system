import cv2
import os

# 创建保存图片的目录
save_dir = '../data/hand/gty/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取视频文件
cap = cv2.VideoCapture('./data/hand/gty_single_hand.mp4')

# 视频的总帧数和帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 计算每一秒的帧数
sec_per_frame = 1 / fps

# 获取每30帧对应的时间点
time_points = [sec_per_frame * i * 30 for i in range(1, 31)]

# 循环读取每一帧并保存
for i in range(frame_count):
    ret, frame = cap.read()

    # 如果读取失败，则跳过
    if not ret:
        continue

    # 计算当前帧所在的时间点
    current_time = i * sec_per_frame

    # 如果当前时间点是需要保存的时间点之一，则保存当前帧
    if current_time in time_points:
        filename = f"{save_dir}frame_{int(current_time / sec_per_frame)}.jpg"
        cv2.imwrite(filename, frame)

    # 如果已经保存了30张图片，则退出循环
    if len(time_points) == 0:
        break

# 释放视频对象
cap.release()
