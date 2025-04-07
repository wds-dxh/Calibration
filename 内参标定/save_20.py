'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-27 00:08:06
Description: 保存20张图片用于标定
Copyright (c) 2025 by ${wds2dxh}, All Rights Reserved. 
'''

import cv2
import os


cap = cv2.VideoCapture(1)
save_path = "save_20"
if not os.path.exists(save_path):
    os.mkdir(save_path)

i = 19 # 定义一个变量，用于保存图片的编号
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"{save_path}/{i}.jpg", frame)  # f表示格式化字符串，{}表示占位符
        print(f"save {i}.jpg")  
        i += 1
        if i == 23:
            break
cap.release()
cv2.destroyAllWindows()