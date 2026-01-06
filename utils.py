import numpy as np
import pdb
from config import R_MOON, K

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

def get_adjugate(M):
    return np.linalg.pinv(M) * np.linalg.det(M)

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

def normalize_vector(v):
    return v / np.linalg.norm(v)

def calculate_invariants(A, c):
    """
        A: list of 3x3 matrices representing conic sections
        c: list of crater centers
    """

    def skew_symmetric(v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    
    l = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        A_i, A_j = normalize_vector(A[i]), normalize_vector(A[j])
        c_i, c_j = c[i], c[j]
        
        eigs = np.linalg.eigvals(A_j @ np.linalg.pinv(-A_i))

        z_exists = False
        for eig in eigs:
            # pdb.set_trace()
            B_ij = eig * A_i + A_j
            B_ij_star = get_adjugate(B_ij)
            
            # k: index of the largest magnitude diagonal of B_ij_star
            k = np.argmax(np.abs(np.diag(B_ij_star)))

            if B_ij_star[k, k] < 0:
                z = -B_ij_star[:, k] / np.sqrt(-B_ij_star[k, k])
                z = z.real
                z_exists = True
                break
        
        if not z_exists:
            # print("Error: no valid z found")
            return 0
        
        D = B_ij + skew_symmetric(z.flatten())
        # m, n: indices of the largest entry ||D_mn|| in D
        m, n = np.unravel_index(np.argmax(np.abs(D)), D.shape)

        h = np.array([D[m, 0], D[m, 1], D[m, 2]]).reshape(3, 1)
        g = np.array([D[0, n], D[1, n], D[2, n]]).reshape(3, 1)

        if h.T @ c_i * h.T @ c_j < 0:
            l.append(h)
        elif g.T @ c_i * g.T @ c_j < 0:
            l.append(g)
        else:
            # print("Error: not enough elements in l")
            return 1
    
    A_star = [get_adjugate(Ai) for Ai in A]
    l_ij = l[0]
    l_ik = l[1]
    l_jk = l[2]
    
    J1 = np.arccosh(np.linalg.norm(l_ij.T @ A_star[0] @ l_ik) / np.sqrt((l_ij.T @ A_star[0] @ l_ij) * (l_ik.T @ A_star[0] @ l_ik))).item() # ij, ik
    J2 = np.arccosh(np.linalg.norm(l_ij.T @ A_star[1] @ l_jk) / np.sqrt((l_ij.T @ A_star[1] @ l_ij) * (l_jk.T @ A_star[1] @ l_jk))).item() # ij, jk
    J3 = np.arccosh(np.linalg.norm(l_ik.T @ A_star[2] @ l_jk) / np.sqrt((l_ik.T @ A_star[2] @ l_ik) * (l_jk.T @ A_star[2] @ l_jk))).item() # ik, jk
    # pdb.set_trace()
    return [J1, J2, J3]


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

def proj_db2img(T_M_C, r_M, Q_star):
    """
        T_M_C: Moon to Camera transformation matrix
        r_M: 3D position vector of camera in Moon frame
        Q_star: Disk Quadric from database

        Return: projected conic matrix
    """
    P_M_C = K @ T_M_C @ np.hstack([np.eye(3), -r_M.reshape(3, 1)])
    A_proj = P_M_C @ get_adjugate(Q_star) @ P_M_C.T
    return A_proj

def conic_to_yY(A):
    A_uu = A[:2, :2]
    A_u1 = A[:2, 2]
    A_11 = A[2, 2]

    y = -np.linalg.pinv(A_uu) @ A_u1
    
    mu = A_u1.T @ np.linalg.pinv(A_uu) @ A_u1 - A_11
    Y = (1.0 / mu) * A_uu
    
    return y, Y

def d_GA(y_i, y_j, Y_i, Y_j):
    coeff = 4 * np.sqrt(np.linalg.det(Y_i) * np.linalg.det(Y_j)) / np.linalg.det(Y_i + Y_j)
    exp = np.exp(-0.5 * (y_i - y_j).T @ Y_i @ np.linalg.pinv(Y_i + Y_j) @ Y_j @ (y_i - y_j))
    return np.arccos(coeff * exp)

def variance(a, b, sigma_img):
    return 0.85**2 / (a * b) * sigma_img**2