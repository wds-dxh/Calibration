'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-26 15:27:59
Description: 
Copyright (c) 2025 by ${wds2dxh}, All Rights Reserved. 
'''
import cv2


cap = cv2.VideoCapture(1)
pattern_size = (9,6)

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(frame, pattern_size)
    # print(ret)
    # print(corners)
    # break

    # break
    cv2.drawChessboardCorners(frame, pattern_size, corners, ret)        # 画出角点
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
