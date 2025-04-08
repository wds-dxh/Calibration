'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-04-07 21:20:00
Description: 使用双目相机标定结果计算视差和深度图，支持直接读取图片和动态调整参数
Copyright (c) 2025 by wds2dxh, All Rights Reserved. 
'''
import numpy as np
import cv2
import os
import argparse

def process_stereo_images(left_img_path, right_img_path, calibration_file="stereo_calibration_result.npz", 
                         window_size=5, num_disp=64):
    """
    处理双目图像，计算视差和深度图
    
    参数:
        left_img_path: 左图像路径
        right_img_path: 右图像路径
        calibration_file: 标定结果文件
        window_size: 初始窗口大小
        num_disp: 初始视差数
    """
    # 加载标定结果
    if not os.path.exists(calibration_file):
        print(f"错误：找不到标定结果文件 {calibration_file}")
        print("请先运行 stereo_calibrate.py 进行双目标定")
        return
    
    calibration_data = np.load(calibration_file)
    
    # 提取标定参数
    map_left_x = calibration_data["map_left_x"]
    map_left_y = calibration_data["map_left_y"]
    map_right_x = calibration_data["map_right_x"]
    map_right_y = calibration_data["map_right_y"]
    Q = calibration_data["Q"]
    
    # 计算基线距离（以毫米为单位）
    T = calibration_data["T"]
    baseline = abs(T[0][0]) / 1000.0  # 转换为米
    
    # 获取相机焦距（以像素为单位）
    camera_matrix_left = calibration_data["camera_matrix_left"]
    focal_length = camera_matrix_left[0, 0]  # fx
    
    print("标定参数加载成功！")
    print(f"基线距离: {baseline*100:.2f} 厘米")
    print(f"相机焦距: {focal_length:.2f} 像素")
    
    # 读取图片
    left_frame = cv2.imread(left_img_path)
    right_frame = cv2.imread(right_img_path)
    
    if left_frame is None or right_frame is None:
        print(f"错误：无法读取图像 {left_img_path} 或 {right_img_path}")
        return
    
    # 创建窗口
    cv2.namedWindow('校正后的双目图像', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2222', cv2.WINDOW_NORMAL)
    cv2.namedWindow('深度图', cv2.WINDOW_NORMAL)
    cv2.namedWindow('1111', cv2.WINDOW_NORMAL)  
    cv2.namedWindow('原始视差', cv2.WINDOW_NORMAL)
    
    # 调整窗口大小
    cv2.resizeWindow('校正后的双目图像', 960, 360)
    cv2.resizeWindow('2222', 480, 360)
    cv2.resizeWindow('深度图', 480, 360)
    cv2.resizeWindow('1111', 600, 300)
    cv2.resizeWindow('原始视差', 480, 360)
    
    # 初始化参数
    if window_size % 2 == 0:
        window_size += 1  # 确保为奇数
    if window_size < 5:
        window_size = 5
    
    # 确保 num_disp 是16的倍数
    num_disp = (num_disp // 16) * 16
    if num_disp < 16:
        num_disp = 16
    
    # 创建参数控制轨迹条
    cv2.createTrackbar('Window Size', '1111', window_size, 25, lambda x: None)
    cv2.createTrackbar('Num Disparities/16', '1111', num_disp//16, 16, lambda x: None)
    cv2.createTrackbar('PreFilterCap', '1111', 31, 63, lambda x: None)
    cv2.createTrackbar('UniquenessRatio', '1111', 10, 30, lambda x: None)
    cv2.createTrackbar('Min Depth (cm)', '1111', 50, 500, lambda x: None)
    cv2.createTrackbar('Max Depth (cm)', '1111', 500, 1000, lambda x: None)
    
    # 创建鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 确保点击在视差图内并且视差值有效
            if disparity is not None and 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
                disp_value = disparity[y, x] / 16.0  # SGBM视差值需要除以16
                if disp_value > 0:
                    # 计算深度(单位:米)
                    depth = (baseline * focal_length) / disp_value
                    # 提取3D坐标
                    point_3d = depth_map[y, x]
                    
                    print(f"\n点击位置 ({x}, {y}):")
                    print(f"视差值: {disp_value:.2f} 像素")
                    print(f"深度: {depth*100:.2f} 厘米")
                    print(f"3D坐标 (x,y,z): ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}) 米")
                else:
                    print(f"\n点击位置 ({x}, {y}): 无效视差值")
    
    # 设置鼠标回调函数
    cv2.setMouseCallback('2222', mouse_callback)
    
    # 存储上一次的参数值，以检测变化
    last_window_size = window_size
    last_num_disp = num_disp
    last_pre_filter_cap = 31
    last_uniqueness_ratio = 10
    last_min_depth = 50
    last_max_depth = 500
    
    # 初始化视差对象
    stereo = None
    disparity = None
    depth_map = None
    
    while True:
        # 获取当前参数值
        current_window_size = cv2.getTrackbarPos('Window Size', '1111')
        current_num_disp = cv2.getTrackbarPos('Num Disparities/16', '1111') * 16
        current_pre_filter_cap = cv2.getTrackbarPos('PreFilterCap', '1111')
        current_uniqueness_ratio = cv2.getTrackbarPos('UniquenessRatio', '1111')
        current_min_depth = cv2.getTrackbarPos('Min Depth (cm)', '1111')
        current_max_depth = cv2.getTrackbarPos('Max Depth (cm)', '1111')
        
        # 确保参数有效
        if current_window_size < 5:
            current_window_size = 5
        if current_window_size % 2 == 0:
            current_window_size += 1
        if current_num_disp < 16:
            current_num_disp = 16
        
        # 检查参数是否有变化
        params_changed = (current_window_size != last_window_size or 
                          current_num_disp != last_num_disp or
                          current_pre_filter_cap != last_pre_filter_cap or
                          current_uniqueness_ratio != last_uniqueness_ratio or
                          current_min_depth != last_min_depth or
                          current_max_depth != last_max_depth)
        
        # 更新上一次的参数值
        last_window_size = current_window_size
        last_num_disp = current_num_disp
        last_pre_filter_cap = current_pre_filter_cap
        last_uniqueness_ratio = current_uniqueness_ratio
        last_min_depth = current_min_depth
        last_max_depth = current_max_depth
        
        # 进行图像校正
        left_rectified = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)
        
        # 转换为灰度图
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        
        # 如果参数发生变化或者是第一次运行，更新立体匹配器
        if stereo is None or params_changed:
            # 创建SGBM立体匹配器
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=current_num_disp,
                blockSize=current_window_size,
                P1=8 * 3 * current_window_size**2,
                P2=32 * 3 * current_window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=current_uniqueness_ratio,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=current_pre_filter_cap,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            
            # 计算视差图
            disparity = stereo.compute(gray_left, gray_right)
            
            # 计算3D点云和深度图
            disparity_float = disparity.astype(np.float32) / 16.0
            depth_map = cv2.reprojectImageTo3D(disparity_float, Q)
            
            # 归一化视差图以便显示
            disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
            
            # 提取深度值进行显示
            depth_values = depth_map[:, :, 2]
            
            # 过滤无效深度值
            mask = disparity_float > 0
            depth_values = np.where(mask, depth_values, 0)
            
            # 限制深度范围
            min_depth_m = current_min_depth / 100.0  # 厘米转米 (默认0.5米)
            max_depth_m = current_max_depth / 100.0  # 厘米转米 (默认5米)
            
            # 确保min_depth小于max_depth
            if min_depth_m >= max_depth_m:
                min_depth_m = 0.1
                max_depth_m = 10.0
            
            valid_depth_mask = (depth_values > min_depth_m) & (depth_values < max_depth_m)
            valid_depth = np.where(valid_depth_mask, depth_values, 0)
            
            # 归一化深度图以便显示
            if valid_depth.max() > valid_depth.min():
                depth_normalized = cv2.normalize(valid_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                depth_normalized = np.zeros_like(valid_depth, dtype=np.uint8)
                
            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_RAINBOW)
            
            # 显示参数信息
            param_info = (f"Window: {current_window_size}, Disp: {current_num_disp}, "
                          f"PreCap: {current_pre_filter_cap}, Uniq: {current_uniqueness_ratio}, "
                          f"Depth: {current_min_depth}-{current_max_depth}cm")
            cv2.putText(disparity_color, param_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 在深度图上显示测量信息
            depth_info = f"Click to measure distance. Baseline: {baseline*100:.1f}cm"
            cv2.putText(depth_color, depth_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 增加深度范围的调试输出
            if valid_depth.size > 0:
                print(f"深度范围: 最小={valid_depth.min():.2f}米, 最大={valid_depth.max():.2f}米")
                print(f"视差范围: 最小={disparity[disparity>0].min()/16:.2f}, 最大={disparity.max()/16:.2f}")
            else:
                print("警告: 未检测到有效深度")
        
        # 绘制水平线帮助观察校正效果
        rectified_with_lines = np.copy(np.hstack((left_rectified, right_rectified)))
        h, w = rectified_with_lines.shape[:2]
        for i in range(0, h, 30):
            cv2.line(rectified_with_lines, (0, i), (w, i), (0, 255, 0), 1)
        
        # 显示图像
        cv2.imshow('校正后的双目图像', rectified_with_lines)
        cv2.imshow('2222', disparity_color)
        cv2.imshow('深度图', depth_color)
        
        # 在计算完视差后添加一个原始视差图窗口
        cv2.namedWindow('原始视差', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('原始视差', 480, 360)

        # 将视差归一化显示(两种方式都试试)
        # 方式1：直接归一化
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 方式2：手动缩放和裁剪
        disp_scaled = np.clip(disparity / 16.0, 0, 255).astype(np.uint8)

        # 显示两个视差图以便比较
        cv2.imshow('原始视差', disp_vis)
        
        # 按下 ESC 键退出，按下 's' 键保存当前帧
        key = cv2.waitKey(100)
        if key == 27:  # ESC键
            break
        elif key == ord('s'):
            # 保存当前结果
            timestamp = cv2.getTickCount()
            cv2.imwrite(f'left_rectified_{timestamp}.jpg', left_rectified)
            cv2.imwrite(f'right_rectified_{timestamp}.jpg', right_rectified)
            cv2.imwrite(f'disparity_{timestamp}.jpg', disparity_color)
            cv2.imwrite(f'depth_{timestamp}.jpg', depth_color)
            print(f"已保存当前帧图像，时间戳: {timestamp}")
    
    # 释放资源
    cv2.destroyAllWindows()
    
    print("程序已退出")

def main():
    parser = argparse.ArgumentParser(description='使用双目相机标定结果计算视差和深度图')
    parser.add_argument('--left', type=str, required=True, help='左相机图像路径')
    parser.add_argument('--right', type=str, required=True, help='右相机图像路径')
    parser.add_argument('--calibration', type=str, default='stereo_calibration_result.npz', help='标定结果文件路径')
    parser.add_argument('--window_size', type=int, default=5, help='匹配窗口大小')
    parser.add_argument('--num_disp', type=int, default=64, help='视差数量，必须是16的倍数')
    
    args = parser.parse_args()
    
    process_stereo_images(args.left, args.right, args.calibration, args.window_size, args.num_disp)

if __name__ == "__main__":
    main() 