'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-27 00:08:06
Description: 保存图片用于标定，改进版支持更多配置选项
Copyright (c) 2025 by wds2dxh, All Rights Reserved. 
'''

import cv2
import os
import argparse
import numpy as np

def save_calibration_images(device=0, width=2560, height=720, output_dir="save_20", max_images=20):
    """
    从双目相机捕获并保存校准图像
    
    参数:
        device: 相机设备号
        width: 帧宽度
        height: 帧高度
        output_dir: 输出目录
        max_images: 最大图像数量
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 初始化相机
    cap = cv2.VideoCapture(device)
    
    # 设置相机参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("错误：无法打开相机")
        return
    
    # 创建窗口
    cv2.namedWindow("left_frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("right_frame", cv2.WINDOW_NORMAL)
    
    # 调整窗口大小
    cv2.resizeWindow("left_frame", 640, 480)
    cv2.resizeWindow("right_frame", 640, 480)
    
    image_count = 0
    
    print("按's'键保存图像，按'q'键退出")
    print(f"将保存图像到: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取图像")
            break
            
        # 分割双目图像
        frame_width = frame.shape[1]
        right_frame = frame[:, :frame_width//2, :]
        left_frame = frame[:, frame_width//2:, :]
        
        # 显示图像
        cv2.imshow("left_frame", left_frame)
        cv2.imshow("right_frame", right_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break
        elif key == ord('s'):
            # 保存当前帧
            left_filename = f"{output_dir}/left_{image_count}.jpg"
            right_filename = f"{output_dir}/right_{image_count}.jpg"
            
            cv2.imwrite(left_filename, left_frame)
            cv2.imwrite(right_filename, right_frame)
            
            print(f"已保存图像对 {image_count+1}/{max_images}: {left_filename}, {right_filename}")
            
            image_count += 1
            if image_count >= max_images:
                print(f"已达到最大图像数量 ({max_images})")
                break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"总共保存了 {image_count} 对图像")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='保存双目相机图像用于标定')
    parser.add_argument('--device', type=int, default=0, help='相机设备号')
    parser.add_argument('--width', type=int, default=2560, help='帧宽度')
    parser.add_argument('--height', type=int, default=720, help='帧高度')
    parser.add_argument('--output', type=str, default='save_20', help='输出目录')
    parser.add_argument('--max-images', type=int, default=20, help='要保存的图像对数量')
    
    args = parser.parse_args()
    
    save_calibration_images(
        device=args.device,
        width=args.width,
        height=args.height,
        output_dir=args.output,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main()