'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-04-07 21:10:00
Description: 测试双目相机标定结果，进行图像校正并显示
Copyright (c) 2025 by wds2dxh, All Rights Reserved. 
'''
import numpy as np
import cv2
import os

def main():
    # 加载标定结果
    if not os.path.exists("stereo_calibration_result.npz"):
        print("错误：找不到标定结果文件 stereo_calibration_result.npz")
        print("请先运行 stereo_calibrate.py 进行双目标定")
        return
    
    calibration_data = np.load("stereo_calibration_result.npz")
    
    # 提取标定参数
    camera_matrix_left = calibration_data["camera_matrix_left"]
    dist_coeffs_left = calibration_data["dist_coeffs_left"]
    camera_matrix_right = calibration_data["camera_matrix_right"]
    dist_coeffs_right = calibration_data["dist_coeffs_right"]
    R1 = calibration_data["R1"]
    R2 = calibration_data["R2"]
    P1 = calibration_data["P1"]
    P2 = calibration_data["P2"]
    Q = calibration_data["Q"]
    map_left_x = calibration_data["map_left_x"]
    map_left_y = calibration_data["map_left_y"]
    map_right_x = calibration_data["map_right_x"]
    map_right_y = calibration_data["map_right_y"]
    
    print("标定参数加载成功！")
    
    # 打开相机
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("错误：无法打开相机")
        return
    
    # 创建窗口
    cv2.namedWindow('原始双目图像', cv2.WINDOW_NORMAL)
    cv2.namedWindow('校正后的双目图像', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取图像")
            break
        
        # 分割双目图像
        right_frame = frame[:, :1280, :]
        left_frame = frame[:, 1280:, :]
        
        # 进行图像校正s
        left_rectified = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)
        
        # 绘制水平线以便于观察校正效果
        height, width = left_rectified.shape[:2]
        for i in range(0, height, 30):
            cv2.line(left_rectified, (0, i), (width, i), (0, 255, 0), 1)
            cv2.line(right_rectified, (0, i), (width, i), (0, 255, 0), 1)
        
        # 合并原始图像和校正后的图像进行显示
        original_stereo = np.hstack((left_frame, right_frame))
        rectified_stereo = np.hstack((left_rectified, right_rectified))
        
        # 显示图像
        cv2.imshow('原始双目图像', original_stereo)
        cv2.imshow('校正后的双目图像', rectified_stereo)
        
        # 按下 ESC 键退出
        key = cv2.waitKey(1)
        if key == 27:  # ESC键
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print("程序已退出")

if __name__ == "__main__":
    main() 