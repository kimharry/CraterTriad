import numpy as np
import pdb
from config import R_MOON

def sort_clockwise(craters):
    p1, p2, p3 = craters[0]['pos'], craters[1]['pos'], craters[2]['pos']
    
    if len(p1) == 2:
        # 2D case
        p1 = np.append(p1, 0)
        p2 = np.append(p2, 0)
        p3 = np.append(p3, 0)

    centroid = (p1 + p2 + p3) / 3
    normal = np.cross(p2 - p1, p3 - p1)

    if np.dot(normal, centroid) > 0:
        return [craters[2], craters[1], craters[0]]

    return craters

def get_center_vector(lat, lon):
    return R_MOON * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def get_ENU_to_Moon_matrix(p_M):
    """
        Returns the transformation matrix from local ENU coordinate to Moon-centered coordinate (T^E_M)
    """
    k = np.array([0, 0, 1])

    u = p_M / np.linalg.norm(p_M)
    e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
    n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

    return np.column_stack([e, n, u])

def get_2d_conic_matrix(theta, a, b, x_c=0, y_c=0):
    """
    Convert crater parameters to 2D conic matrix.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    a2 = a * a
    b2 = b * b
    
    A = a2 * sin_t**2 + b2 * cos_t**2
    B = 2 * (b2 - a2) * cos_t * sin_t
    C = a2 * cos_t**2 + b2 * sin_t**2
    D = -2 * A * x_c - B * y_c
    F = -B * x_c - 2 * C * y_c
    G = A * x_c**2 + B * x_c * y_c + C * y_c**2 - a2 * b2
    
    M = np.array([
        [A,   B/2, D/2],
        [B/2, C,   F/2],
        [D/2, F/2, G  ]
    ])
    
    return M

def normalize_matrix(M):
    det = np.linalg.det(M)
    scale = np.sign(det) * (np.abs(det) ** (1/3))
    return M / scale

def get_adjugate(M):
    return np.linalg.inv(M) * np.linalg.det(M)

def get_disk_quadric(T_E_M, p_M, conic_matrix_2d):
    """
        T_E_M: Transformation matrix from ENU to Moon-centered coordinates (3x3)
        p_M: Crater center position in Moon-centered coordinates (3x1)
        conic_matrix_2d: 2D conic matrix representing the crater shape
    """
    C_star = get_adjugate(conic_matrix_2d)
    S = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    H_M = np.hstack([T_E_M @ S, p_M.reshape(3, 1)])
    k = np.array([0, 0, 1])

    term1 = np.vstack([H_M, k])
    Q_star = term1 @ C_star @ term1.T

    return Q_star

def calculate_invariants(A1, A2, A3):
    pass

def EPS(craters, r):
    combs = []
    n = len(craters)
    for dj in range(1, n-1):
        for dk in range(1, n-dj):
            for ii in range(r):
                for i in range(ii, n-dj-dk, r):
                    j = i + dj
                    k = j + dk
                    combs.append([craters[i], craters[j], craters[k]])
    return combs