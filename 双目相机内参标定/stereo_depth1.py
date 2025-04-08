'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-04-07 21:20:00
Description: 使用双目相机标定结果计算视差和深度图
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
    map_left_x = calibration_data["map_left_x"]
    map_left_y = calibration_data["map_left_y"]
    map_right_x = calibration_data["map_right_x"]
    map_right_y = calibration_data["map_right_y"]
    Q = calibration_data["Q"]
    
    print("标定参数加载成功！")
    
    # 创建立体匹配器
    # SGBM参数设置
    window_size = 12
    min_disp = 0
    num_disp = 7  # 必须是16的倍数
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,  # 最小视差
        numDisparities=num_disp,  # 视差范围
        blockSize=window_size,  # 匹配窗口大小
        P1=8 * 3 * window_size**2,  # 惩罚参数
        P2=32 * 3 * window_size**2,  # 惩罚参数
        disp12MaxDiff=1,  # 视差一致性检查
        uniquenessRatio=10,  # 唯一性约束
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 打开相机
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("错误：无法打开相机")
        return
    
    # 创建窗口
    cv2.namedWindow('校正后的双目图像', cv2.WINDOW_NORMAL)
    cv2.namedWindow('视差图', cv2.WINDOW_NORMAL)
    cv2.namedWindow('深度图', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取图像")
            break
        
        # 分割双目图像
        right_frame = frame[:, :1280, :]
        left_frame = frame[:, 1280:, :]
        
        # 进行图像校正
        left_rectified = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)
        
        # 转换为灰度图
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        
        # 计算视差图
        disparity = stereo.compute(gray_left, gray_right)
        
        # 归一化视差图以便显示
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        # 计算3D点云和深度图
        disparity_float = disparity.astype(np.float32) / 16.0
        depth = cv2.reprojectImageTo3D(disparity_float, Q)
        
        # 提取深度值进行显示
        depth_map = depth[:, :, 2]
        
        # 过滤无效深度值
        mask = disparity_float > min_disp
        depth_map = np.where(mask, depth_map, 0)
        
        # 限制深度范围（例如：0.5米到5米）
        valid_depth_mask = (depth_map > 0.5) & (depth_map < 5.0)
        valid_depth = np.where(valid_depth_mask, depth_map, 0)
        
        # 归一化深度图以便显示
        if valid_depth.max() > valid_depth.min():
            depth_normalized = cv2.normalize(valid_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            depth_normalized = np.zeros_like(valid_depth, dtype=np.uint8)
            
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_RAINBOW)
        
        # 合并校正后的图像进行显示
        rectified_stereo = np.hstack((left_rectified, right_rectified))
        
        # 显示图像
        cv2.imshow('校正后的双目图像', rectified_stereo)
        cv2.imshow('视差图', disparity_color)
        cv2.imshow('深度图', depth_color)
        
        # 按下 ESC 键退出，按下 's' 键保存当前帧
        key = cv2.waitKey(1)
        if key == 27:  # ESC键
            break
        elif key == ord('s'):
            cv2.imwrite('left_rectified.jpg', left_rectified)
            cv2.imwrite('right_rectified.jpg', right_rectified)
            cv2.imwrite('disparity.jpg', disparity_color)
            cv2.imwrite('depth.jpg', depth_color)
            print("已保存当前帧图像")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print("程序已退出")

if __name__ == "__main__":
    main() 