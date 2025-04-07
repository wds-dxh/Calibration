'''
Author: wds2dxh wdsnpshy@163.com
Date: 2025-02-27 00:47:36
Description: 使用标定好的相机参数对图像进行校正
Copyright (c) 2025 by ${wds2dxh}, All Rights Reserved. 
'''

import cv2
import numpy as np

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    使用相机参数对图像进行校正
    
    参数:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    返回:
        undistorted_img: 校正后的图像
    """
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 获取最优相机矩阵，作用是获取一个最佳的相机矩阵，使得图像校正后的图像尽可能多的保留原始图像的信息（比如可能会裁剪掉一些图像的边缘信息）
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(     
        camera_matrix, dist_coeffs, (w,h), 1, (w,h)
    )
    
    # 校正图像
    undistorted_img = cv2.undistort(
        image, 
        camera_matrix, 
        dist_coeffs, 
        None, 
        new_camera_matrix
    )
    
    # 裁剪图像
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    return undistorted_img

# 使用示例
image_path = "./save_20/0.jpg"
img = cv2.imread(image_path)

#加载标定结果
# calibration_result = np.load("calibration_result.npz")
# camera_matrix = calibration_result["camera_matrix_left"]
# dist_coeffs = calibration_result["distortion_coeffs_left"]



# 使用之前获得的相机参数
camera_matrix = np.array([[5.85944013e+03, 0.00000000e+00, 3.12206258e+02], # 修改点1：相机内参矩阵
                         [0.00000000e+00, 6.83849170e+03, 2.41970339e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([[-7.30909746e+00, -4.36531650e+02, -2.21391554e-02, # 修改点2：畸变系数
                        2.20096496e-01, -1.66286367e+00]])

# 进行图像校正
undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.stereoCalibrate