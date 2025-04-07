'''
Author: wds-Ubuntu22-cqu wdsnpshy@163.com
Date: 2025-04-07 20:44:54
Description: 获取双目摄像头的图像
邮箱：wdsnpshy@163.com 
Copyright (c) 2025 by ${wds-Ubuntu22-cqu}, All Rights Reserved. 
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    # 分割双目图像
    right_frame = frame[:, :1280, :]
    left_frame = frame[:, 1280:, :]

    # 显示双目图像
    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
