import cv2
from STESDK import BodyDetector, BodyAligner

# 创建人体检测器和关键点检测器
body_detector = BodyDetector()
body_aligner = BodyAligner()

# 读取图片
frame = cv2.imread(r"C:\Users\86195\Desktop\contest-phone\gty_body.jpg")
print("frame size is: ", frame.shape)

# 人体位置检测
rect = body_detector.detect(frame)

# 关键点检测
points = body_aligner.align(frame, rect[0])

# 获取两个新的关键点
y = (points[0][1] + points[1][1]) / 2
x2 = (points[1][0] + points[3][0]) / 2
x1 = (points[1][0] + points[2][0]) / 2
new_points = [(int(x1), int(y)),(int(x2), int(y))]


# 把关键点标记在图片上
points_frame = body_aligner.render(frame, points)

# 添加新的关键点
for point in new_points:
    x, y = point
    cv2.circle(points_frame, (x, y), 5, (0, 255, 0), -1)

# 获取图片大小
height, width, _ = points_frame.shape

# 调整图片大小以适应显示器分辨率
resized_frame = cv2.resize(points_frame, (width // 5, height // 5))

# 显示图片并绘制关键点数字
for i, point in enumerate(points):
    x, y = point
    x, y = int(x // 5), int(y // 5)  # 缩放关键点坐标
    cv2.putText(resized_frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

for i, point in enumerate(new_points):
    x, y = int(point[0] // 5), int(point[1] // 5)  # 缩放新关键点坐标
    cv2.putText(resized_frame, f"New {i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("body keypoints", resized_frame)

# 获取每个关键点的坐标
for i, point in enumerate(points):
    x, y = point
    print(f"Keypoint {i}: ({x}, {y})")

# 输出新关键点的坐标
for i, point in enumerate(new_points):
    x, y = point
    print(f"New point {i}: ({x}, {y})")

# 获得人体检测人框
rect_frame = body_detector.render(frame, rect)
# 调整图片大小以适应显示器分辨率
rect_frame = cv2.resize(rect_frame, (width // 5, height // 5))
# 显示图片
cv2.imshow("body rect", rect_frame)

# 等待用户关闭窗口
k = cv2.waitKey(0)
if k == ord("q"):
    cv2.destroyAllWindows()
