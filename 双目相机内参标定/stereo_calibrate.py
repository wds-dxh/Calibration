'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-04-07 21:00:00
Description: 双目相机标定程序，使用stereoCalibrate并保存为npz文件
Copyright (c) 2025 by wds2dxh, All Rights Reserved. 
'''
import numpy as np
import cv2
import os
import glob

# 初始化参数
board_cols = 9  # 棋盘格列数
board_rows = 6  # 棋盘格行数
square_size = 21.8  # 棋盘格每个小格的实际大小，单位：毫米

# 创建一个 3D 点数组，用于存储每个角点的世界坐标
object_points = np.zeros((board_cols * board_rows, 3), np.float32)
object_points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
object_points *= square_size
 
# 初始化存储所有图像和物体坐标点的列表
object_points_all = []  # 存储所有图像的物体坐标（世界坐标系）
image_points_left = []  # 存储左相机图像中的角点坐标
image_points_right = []  # 存储右相机图像中的角点坐标

# 图像目录
image_dir = "./save_20"
if not os.path.exists(image_dir):
    print(f"错误：目录 {image_dir} 不存在")
    exit(1)

# 获取图像文件
left_images = sorted(glob.glob(f"{image_dir}/left_*.jpg"))
right_images = sorted(glob.glob(f"{image_dir}/right_*.jpg"))

if len(left_images) == 0 or len(right_images) == 0:
    print("错误：未找到图像文件")
    exit(1)

if len(left_images) != len(right_images):
    print(f"警告：左相机图像数量({len(left_images)})与右相机图像数量({len(right_images)})不一致")

print(f"找到 {len(left_images)} 对立体图像")

# 创建显示窗口
cv2.namedWindow('左相机角点', cv2.WINDOW_NORMAL)
cv2.namedWindow('右相机角点', cv2.WINDOW_NORMAL)

# 处理每一对图像
valid_pairs = 0
for i in range(len(left_images)):
    left_img_path = left_images[i]
    right_img_path = right_images[i]
    
    # 读取图像
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    
    if img_left is None or img_right is None:
        print(f"警告：无法读取图像对 {i}")
        continue
    
    # 转换为灰度图
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # 获取图像尺寸
    h_left, w_left = gray_left.shape[:2]
    h_right, w_right = gray_right.shape[:2]
    
    # 寻找棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, (board_cols, board_rows), None
    )
    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right, (board_cols, board_rows), None
    )
    
    # 如果两幅图像都找到角点
    if ret_left and ret_right:
        valid_pairs += 1
        print(f"成功找到第 {i} 对图像的角点")
        
        # 添加物体点
        object_points_all.append(object_points)
        
        # 角点精细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        corners2_left = cv2.cornerSubPix(
            gray_left, corners_left, (11, 11), (-1, -1), criteria
        )
        corners2_right = cv2.cornerSubPix(
            gray_right, corners_right, (11, 11), (-1, -1), criteria
        )
        
        # 存储图像点
        image_points_left.append(corners2_left)
        image_points_right.append(corners2_right)
        
        # 绘制角点
        img_left_corners = img_left.copy()
        img_right_corners = img_right.copy()
        
        cv2.drawChessboardCorners(
            img_left_corners, (board_cols, board_rows), corners2_left, ret_left
        )
        cv2.drawChessboardCorners(
            img_right_corners, (board_cols, board_rows), corners2_right, ret_right
        )
        
        # 显示角点
        cv2.imshow('左相机角点', img_left_corners)
        cv2.imshow('右相机角点', img_right_corners)
        
        # 等待按键
        key = cv2.waitKey(0)
        if key == 27:  # ESC键退出
            break
    else:
        if not ret_left:
            print(f"警告：无法在左相机图像 {left_img_path} 中找到角点")
        if not ret_right:
            print(f"警告：无法在右相机图像 {right_img_path} 中找到角点")

# 检查是否有足够的标定图片
if valid_pairs < 3:
    print("错误：成功检测到角点的图像对数量不足，至少需要3对图像才能进行标定")
    cv2.destroyAllWindows()
    exit(1)

print(f"\n共成功处理 {valid_pairs} 对图像")

# 先分别对左右相机进行单目标定
print("\n正在进行左相机单目标定...")
ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    object_points_all, image_points_left, (w_left, h_left), None, None
)

print("\n正在进行右相机单目标定...")
ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    object_points_all, image_points_right, (w_right, h_right), None, None
)

# 设置为无畸变
# camera_matrix_left = np.eye(3)
# dist_coeffs_left = np.zeros((4, 1))
# camera_matrix_right = np.eye(3)
# dist_coeffs_right = np.zeros((4, 1))

print("\n正在进行双目标定...")
# 双目标定，计算两个相机之间的位置关系
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-5) # 迭代终止条件
stereocalib_flags = cv2.CALIB_FIX_INTRINSIC # 固定内参

ret_stereo, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
    object_points_all,  # 物体点坐标
    image_points_left,  # 左相机图像点坐标
    image_points_right, # 右相机图像点坐标

    camera_matrix_left, # 左相机内参矩阵
    dist_coeffs_left,   # 左相机畸变系数
    camera_matrix_right, # 右相机内参矩阵
    dist_coeffs_right,  # 右相机畸变系数

    (w_left, h_left),   # 图像尺寸
    criteria=stereocalib_criteria, # 迭代终止条件
    flags=stereocalib_flags   # 标定标志
)

# 计算立体校正参数（用于双目矫正）
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    (w_left, h_left), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.9
)

# 计算立体校正映射
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    camera_matrix_left, dist_coeffs_left, R1, P1, (w_left, h_left), cv2.CV_32FC1
)

map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, (w_right, h_right), cv2.CV_32FC1
)

# 保存双目标定结果
np.savez("stereo_calibration_result.npz",
    camera_matrix_left=camera_matrix_left,  # 左相机内参矩阵
    dist_coeffs_left=dist_coeffs_left,      # 左相机畸变系数
    camera_matrix_right=camera_matrix_right, # 右相机内参矩阵
    dist_coeffs_right=dist_coeffs_right,     # 右相机畸变系数
    R=R,                                    # 旋转矩阵 R
    T=T,                                    # 平移向量 T
    E=E,                                    # 本质矩阵 E
    F=F,                                    # 基础矩阵 F

    R1=R1,                                  # 立体校正旋转矩阵 R1
    R2=R2,                                  # 立体校正旋转矩阵 R2
    P1=P1,                                  # 立体校正投影矩阵 P1
    P2=P2,                                  # 立体校正投影矩阵 P2
    Q=Q,                                    # 视差到深度映射矩阵 Q
    map_left_x=map_left_x,                  # 左相机映射矩阵 map_left_x
    map_left_y=map_left_y,                  # 左相机映射矩阵 map_left_y
    map_right_x=map_right_x,                # 右相机映射矩阵 map_right_x
    map_right_y=map_right_y,                # 右相机映射矩阵 map_right_y
    stereo_calibration_error=ret_stereo      # 立体标定误差 stereo_calibration_error
)

# 打印标定结果
print("\n双目标定完成！")
print(f"立体标定误差: {ret_stereo}")
print(f"旋转矩阵 R:\n{R}")
print(f"平移向量 T:\n{T}")
print(f"本质矩阵 E:\n{E}")
print(f"基础矩阵 F:\n{F}")

print("\n标定结果已保存到 stereo_calibration_result.npz")

cv2.destroyAllWindows() 