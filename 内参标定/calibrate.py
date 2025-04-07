'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-27 00:19:59
Description: 
Copyright (c) 2025 by ${wds2dxh}, All Rights Reserved. 
'''
import numpy as np
import cv2
import os
import sys

# 共有七处修改点

# 1.初始化参数
board_cols = 9  # 修改点1：棋盘格列数。请根据自己的标定棋盘格列数进行修改
board_rows = 6  # 修改点2：棋盘格行数。请根据自己的标定棋盘格行数进行修改
square_size = 20.0 # 修改点3：棋盘格每个小格的实际大小，单位：毫米。请根据实际棋盘格大小进行修改

# 创建一个 3D 点数组 object_points，用于存储每个角点的世界坐标（实际物理世界中的位置）。
object_points = np.zeros((board_cols * board_rows, 3), np.float32)  # 创建一个 (cols * rows) x 3 的零矩阵
object_points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)  # 将棋盘格的 2D 坐标赋值给前两列
object_points *= square_size  # 将坐标转换为实际物理世界中的尺寸

# print(object_points)  # 输出物理世界坐标系下的角点位置

# 初始化存储所有图像和物体坐标点的列表
object_points_all = []  # 存储所有图像的物体坐标（世界坐标系）
image_points_left = []  # 存储左相机图像中的角点坐标
image_points_right = []  # 存储右相机图像中的角点坐标（如果是双目标定）

# 2.读取标定图片
# 修改点4：指定用于标定的图像数量
image_nums = 22  # 这里假设有22张标定图像，你可以根据实际情况修改
for i in range(0, image_nums):
    # 添加错误处理和文件存在检查
    image_path = "./save_20/{}.jpg".format(i)
    if not os.path.exists(image_path):
        print(f"警告：找不到图片 {image_path}")
        continue
        
    img_left = cv2.imread(image_path)
    if img_left is None:
        print(f"错误：无法读取图片 {image_path}")
        continue
    
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    
    # 添加图像尺寸自动获取
    h, w = gray_left.shape[:2]
    
    # 添加更详细的角点检测信息
    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, (board_cols, board_rows), None
    )   
    if ret_left:
        print(f"成功找到第 {i} 张图片的角点")
        object_points_all.append(object_points)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners2_left = cv2.cornerSubPix(
            gray_left, corners_left, (11, 11), (-1, -1), criteria
        )
        image_points_left.append(corners2_left)

        # 绘制角点并保存结果图像
        img_with_corners = img_left.copy()
        cv2.drawChessboardCorners(
            img_with_corners, (board_cols, board_rows), corners2_left, ret_left
        )
        cv2.imshow("检测到的角点", img_with_corners)
        cv2.waitKey(500)  # 显示0.5秒
    else:
        print(f"警告：无法在第 {i} 张图片中找到角点")

# 添加检查是否有足够的标定图片
if len(image_points_left) < 3:
    print("错误：成功检测到角点的图片数量不足，至少需要3张图片才能进行标定")
    cv2.destroyAllWindows()
    exit(1)

print(f"\n共成功处理 {len(image_points_left)} 张图片")

# 5.相机标定
# 使用 OpenCV 的 calibrateCamera 函数进行相机标定，获取相机矩阵、畸变系数等
calibration_success, camera_matrix_left, distortion_coeffs_left, rotation_vectors_left, translation_vectors_left = cv2.calibrateCamera(
   object_points_all, image_points_left, (w, h), None, None
)

# 打印标定结果
print("calibration_success", calibration_success)  # 标定是否成功（布尔值）
print("camera_matrix_left:", camera_matrix_left)  # 左相机的相机矩阵
print("distortion_coeffs_left", distortion_coeffs_left)  # 左相机的畸变系数
