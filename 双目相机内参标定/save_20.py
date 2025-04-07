'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-27 00:08:06
Description: 保存20张图片用于标定
Copyright (c) 2025 by ${wds2dxh}, All Rights Reserved. 
'''

import cv2
import os


cap = cv2.VideoCapture(0)
save_path = "save_20"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# 因为是双目摄像头，所以需要设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# 创建一个窗口，用于显示双目图像
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

i = 0 # 定义一个变量，用于保存图片的编号
while True:
    ret, frame = cap.read()
    # 分割双目图像
    right_frame = frame[:, :1280, :]
    left_frame = frame[:, 1280:, :]

    # 在窗口中显示双目图像
    cv2.imshow("left_frame", left_frame)
    cv2.imshow("right_frame", right_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"{save_path}/left_{i}.jpg", left_frame)  # f表示格式化字符串，{}表示占位符
        cv2.imwrite(f"{save_path}/right_{i}.jpg", right_frame)  # f表示格式化字符串，{}表示占位符
        print(f"save {i}.jpg")  
        i += 1
        if i == 20:
            break
cap.release()
cv2.destroyAllWindows()