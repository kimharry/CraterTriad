import numpy as np
import pdb

from torch import fill
from config import R_MOON, K, ALTITUDE
from matplotlib import pyplot as plt

def sort_clockwise(craters):
    p1, p2, p3 = craters[0]['pos'], craters[1]['pos'], craters[2]['pos']
    
    if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) < 0:
        return [craters[2], craters[1], craters[0]]

    return craters

def get_center_vector(lat, lon, r=R_MOON):
    return r * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def get_ENU_to_Moon_matrix(p_M):
    """
        Returns the transformation matrix from local ENU coordinate to Moon-centered coordinate (T^E_M)
    """
    k = np.array([0, 0, 1])

    u = p_M / np.linalg.norm(p_M)
    e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
    n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

    return np.column_stack([e, n, u])

def get_TMC(lat, lon):
    p_M = get_center_vector(np.radians(lat), np.radians(lon), r=R_MOON + ALTITUDE)

    T_M_C = get_ENU_to_Moon_matrix(p_M)
    T_M_C[:, 2] = T_M_C[:, 2] * -1
    return T_M_C.T

def get_adjugate(M):
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]

    adj = np.array([
        [m11*m22 - m12*m21, -(m01*m22 - m02*m21), m01*m12 - m02*m11],
        [-(m10*m22 - m12*m20), m00*m22 - m02*m20, -(m00*m12 - m02*m10)],
        [m10*m21 - m11*m20, -(m00*m21 - m01*m20), m00*m11 - m01*m10]
    ])
    
    return adj

def normalize_vector(v):
    return v / np.linalg.norm(v)

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
    
    return normalize_vector(M)

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
    return normalize_vector(Q_star)

def calculate_invariants(A, c, a):
    """
        A: list of 3x3 matrices representing conic sections
        c: list of crater centers
    """

    def skew_symmetric(v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    l = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        A_i, A_j = A[i], A[j]
        c_i, c_j = c[i], c[j]
        
        eigs = np.linalg.eigvals(A_j @ np.linalg.inv(-A_i))

        valid_line_found = False
        for eig in eigs:
            if abs(eig.imag) > 1e-10:
                continue

            B_ij = eig * A_i + A_j
            B_ij_star = get_adjugate(B_ij)
            
            # k: index of the largest magnitude diagonal of B_ij_star
            k = np.argmax(np.abs(np.diag(B_ij_star)))

            if B_ij_star[k, k] < 0:
                z = -B_ij_star[:, k] / np.sqrt(-B_ij_star[k, k])

                D = B_ij + skew_symmetric(z.flatten())
                # m, n: indices of the largest entry ||D_mn|| in D
                m, n = np.unravel_index(np.argmax(np.abs(D)), D.shape)

                g = normalize_vector(D[:, n].reshape(3, 1))
                h = normalize_vector(D[m, :].reshape(3, 1))

                epsilon = 1e-6
                valid_g_i = np.dot(g.flatten(), c_i.flatten())
                valid_g_j = np.dot(g.flatten(), c_j.flatten())
                valid_h_i = np.dot(h.flatten(), c_i.flatten())
                valid_h_j = np.dot(h.flatten(), c_j.flatten())

                if valid_g_i * valid_g_j < 0 and abs(valid_g_i) > epsilon and abs(valid_g_j) > epsilon:
                    l.append(h)
                    valid_line_found = True
                    break
                elif valid_h_i * valid_h_j < 0 and abs(valid_h_i) > epsilon and abs(valid_h_j) > epsilon:
                    l.append(g)
                    valid_line_found = True
                    break
        
        if not valid_line_found:
            # pdb.set_trace()
            return None
    
    if len(l) < 3:
        # pdb.set_trace()
        return None
    
    A_star = [get_adjugate(Ai) for Ai in A]
    l_ij = l[0]
    l_ik = l[1]
    l_jk = l[2]

    # draw A_i, A_j, g, h
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(plt.Circle((c[0][0], c[0][1]), a[0], color='r', fill=False, label='A_i')) # A_i (ellipse center)
    ax.add_patch(plt.Circle((c[1][0], c[1][1]), a[1], color='g', fill=False, label='A_j')) # A_j (ellipse center)
    ax.add_patch(plt.Circle((c[2][0], c[2][1]), a[2], color='b', fill=False, label='A_k')) # A_k (ellipse center)

    x_lim = [min(c[0][0]-a[0], c[1][0]-a[1], c[2][0]-a[2]) - 1, max(c[0][0]+a[0], c[1][0]+a[1], c[2][0]+a[2]) + 1]
    y_lim = [min(c[0][1]-a[0], c[1][1]-a[1], c[2][1]-a[2]) - 1, max(c[0][1]+a[0], c[1][1]+a[1], c[2][1]+a[2]) + 1]

    l1_vec = l_ij.flatten()
    l2_vec = l_ik.flatten()
    l3_vec = l_jk.flatten()

    l1_x_vals = np.array(x_lim)
    l1_y_vals = (-l1_vec[0] * l1_x_vals - l1_vec[2]) / l1_vec[1]
    ax.plot(l1_x_vals, l1_y_vals, 'r', label='l1') # g (line vector)
    
    l2_x_vals = np.array(x_lim)
    l2_y_vals = (-l2_vec[0] * l2_x_vals - l2_vec[2]) / l2_vec[1]
    ax.plot(l2_x_vals, l2_y_vals, 'g', label='l2') # h (line vector)

    l3_x_vals = np.array(x_lim)
    l3_y_vals = (-l3_vec[0] * l3_x_vals - l3_vec[2]) / l3_vec[1]
    ax.plot(l3_x_vals, l3_y_vals, 'b', label='l3') # h (line vector)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.show()

    # pdb.set_trace()
    plt.close()

    term1 = (l_ij.T @ A_star[0] @ l_ij) * (l_ik.T @ A_star[0] @ l_ik)
    term2 = (l_ij.T @ A_star[1] @ l_ij) * (l_jk.T @ A_star[1] @ l_jk)
    term3 = (l_ik.T @ A_star[2] @ l_ik) * (l_jk.T @ A_star[2] @ l_jk)
    if term1 < 0 or term2 < 0 or term3 < 0:
        # pdb.set_trace()
        return None

    J1_val = np.linalg.norm(l_ij.T @ A_star[0] @ l_ik) / np.sqrt(term1)
    J2_val = np.linalg.norm(l_ij.T @ A_star[1] @ l_jk) / np.sqrt(term2)
    J3_val = np.linalg.norm(l_ik.T @ A_star[2] @ l_jk) / np.sqrt(term3)
    
    if J1_val == np.nan or J2_val == np.nan or J3_val == np.nan:
        return None
    if J1_val < 0.999 or J2_val < 0.999 or J3_val < 0.999:
        return [J1_val.item(), J2_val.item(), J3_val.item(), False]
    
    one_cnt = 0
    if 0.999 <= J1_val <= 1.0:
        J1_val = 1.0
        one_cnt += 1
    if 0.999 <= J2_val <= 1.0:
        J2_val = 1.0
        one_cnt += 1
    if 0.999 <= J3_val <= 1.0:
        J3_val = 1.0
        one_cnt += 1

    J1 = np.arccosh(J1_val).item()
    J2 = np.arccosh(J2_val).item()
    J3 = np.arccosh(J3_val).item()

    return [J1, J2, J3, one_cnt]


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

def proj_db2img(T_M_C, r_M, Q_star, K=K):
    """
        T_M_C: Moon to Camera transformation matrix
        r_M: 3D position vector of camera in Moon frame
        Q_star: Disk Quadric from database

        Return: projected 2d conic matrix, center of conic in image plane
    """
    P_M_C = K @ T_M_C.T @ np.hstack([np.eye(3), -r_M.reshape(3, 1)])
    A_dual = P_M_C @ Q_star @ P_M_C.T

    try:
        A_proj = np.linalg.inv(A_dual)
    except:
        return None
    
    if np.trace(A_proj[:2, :2]) < 1e-10:
        A_proj = -A_proj

    c_homo = A_dual[:, 2]

    if abs(c_homo[2]) > 1e-10:
        center = c_homo / c_homo[2]
    else:
        center = np.array([0, 0, 1])

    val_at_center = center.T @ A_proj @ center
    breakpoint()
    if val_at_center >= 0:
        return None

    return A_proj, center

def conic_to_yY(A):
    A_uu = A[:2, :2]
    A_u1 = A[:2, 2]
    A_11 = A[2, 2]

    y = -np.linalg.inv(A_uu) @ A_u1
    
    mu = A_u1.T @ np.linalg.inv(A_uu) @ A_u1 - A_11
    Y = (1.0 / mu) * A_uu
    
    return y, Y

def d_GA(y_i, y_j, Y_i, Y_j):
    coeff = 4 * np.sqrt(np.linalg.det(Y_i) * np.linalg.det(Y_j)) / np.linalg.det(Y_i + Y_j)
    exp = np.exp(-0.5 * (y_i - y_j).T @ Y_i @ np.linalg.inv(Y_i + Y_j) @ Y_j @ (y_i - y_j))
    return np.arccos(coeff * exp)

def variance(a, b, sigma_img):
    return 0.85**2 / (a * b) * sigma_img**2