import cv2
import numpy as np

def project_3d_to_2d(point_3d, camera_matrix, dist_coeffs):
    """将3D点投影到2D图像平面"""
    point = np.array([point_3d], dtype=np.float32)
    projected, _ = cv2.projectPoints(
        point,
        np.zeros(3),  # rvec
        np.zeros(3),  # tvec
        camera_matrix,
        dist_coeffs
    )
    return projected[0][0]  # 返回(u, v)