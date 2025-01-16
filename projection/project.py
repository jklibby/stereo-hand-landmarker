from typing import *
import pathlib
import numpy as np

stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")

def load_calibration():
    intrinsic_dir = pathlib.Path("camera_intrinsics")
    intrinsics = list()
    for intrinsic_file in sorted(list(intrinsic_dir.glob("camera_calibration_*.npz"))):
        intrinsics.append(np.load(intrinsic_file))
    return intrinsics



def project3d(p1: np.ndarray, p2: np.ndarray, dir: str, world_scaling:int=1):
    intrinsics = load_calibration()
    K1 = intrinsics[0]["calibration_mtx"]
    K2 = intrinsics[1]["calibration_mtx"]
    R, T = stereo_calibration['R'], stereo_calibration['T']
    
    # projects from camera 2 to camera 1
    extrinsic = np.hstack([R, T])
    projection_matrix_1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
    projection_matrix_2 = np.dot(K2, extrinsic)
    
    # pcd = [triangulate(projection_matrix_1, projection_matrix_2, p1[i, :], p2[i, :]) for i in range(p1.shape[0])]
    # numpy_pcd = np.array(pcd)
    numpy_pcd = vect_triangulate(projection_matrix_1, projection_matrix_2, p1, p2)
    return numpy_pcd


import numpy as np

def vect_triangulate(P1, P2, p1, p2):
    """
    Vectorized triangulation for multiple correspondences.

    Parameters
    ----------
    P1 : ndarray of shape (3,4)
        Projection matrix for camera 1.
    P2 : ndarray of shape (3,4)
        Projection matrix for camera 2.
    p1 : ndarray of shape (N, 2)
        Array of matched points in image 1.
    p2 : ndarray of shape (N, 2)
        Array of matched points in image 2.

    Returns
    -------
    points_3d : ndarray of shape (N, 3)
        Triangulated 3D points.
    """
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    A = np.stack([
        y1[:, None]*P1[2,:] - P1[1,:],
        P1[0,:] - x1[:, None]*P1[2,:],
        y2[:, None]*P2[2,:] - P2[1,:],
        P2[0,:] - x2[:, None]*P2[2,:]
    ], axis=1)  # shape (N,4,4)

    B = np.einsum('nij,njk->nik', A.transpose(0,2,1), A)

    N = B.shape[0]
    points_3d = np.zeros((N, 3), dtype=float)

    for i in range(N):
        _, _, Vh = np.linalg.svd(B[i], full_matrices=False)
        points_3d[i] = Vh[3,0:3] / Vh[3,3]

    return points_3d


def triangulate(P1, P2, point1, point2):
    point1 = point1.reshape(-1)
    point2 = point2.reshape(-1)
    A = np.array([point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ])
    A = A.reshape((4,4))
    
    B = A.T @ A
    U, s, Vh = np.linalg.svd(B, full_matrices = False)

    point_3d = Vh[3,0:3]/Vh[3,3]
    return point_3d

