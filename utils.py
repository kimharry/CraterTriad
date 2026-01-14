import numpy as np
from config import R_MOON, K, ALTITUDE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    p_M = get_center_vector(lat, lon, r=R_MOON + ALTITUDE)

    T_E_M = get_ENU_to_Moon_matrix(p_M)
    e = T_E_M[:, 0]
    n = T_E_M[:, 1] * -1
    u = T_E_M[:, 2] * -1

    T_M_C = np.column_stack([e, n, u])

    return T_M_C

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

def get_conic_locus_matrix(theta, a, b, x_c=0, y_c=0):
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

    det_M = np.linalg.det(M)
    if abs(det_M) < 1e-10:
        return None
    
    # For ellipse, det should be negative
    # Normalize: det(M) = -1
    scale = abs(det_M) ** (1/3)
    M = M / scale

    return M

def get_disk_quadric(T_E_M, p_M, C):
    """
        T_E_M: Transformation matrix from ENU to Moon-centered coordinates (3x3)
        p_M: Crater center position in Moon-centered coordinates (3x1)
        C: 2D conic locus matrix representing the crater shape
    """
    C_star = get_adjugate(C)
    S = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    H_M = np.hstack([T_E_M @ S, p_M.reshape(3, 1)])
    k = np.array([0, 0, 1])

    term1 = np.vstack([H_M, k])
    Q_star = term1 @ C_star @ term1.T
    
    Q_star = 0.5 * (Q_star + Q_star.T)
    return Q_star

def calculate_invariants(As, A_stars):
    """
        As: list of conic locus matrices
        A_stars: list of conic envelope matrices

        Return: list of invariants
    """

    def skew_symmetric(v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def visualize(ellipses, l_12, l_23, l_31):
        # draw A_i, A_j, g, h
        centers = [ellipse['center'] for ellipse in ellipses]
        majors = [ellipse['major'] for ellipse in ellipses]
        minors = [ellipse['minor'] for ellipse in ellipses]
        angles = [ellipse['angle'] for ellipse in ellipses]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_patch(Ellipse(centers[0], 2*majors[0], 2*minors[0], angle=angles[0], color='r', fill=False, label='A_i'))
        ax.add_patch(Ellipse(centers[1], 2*majors[1], 2*minors[1], angle=angles[1], color='g', fill=False, label='A_j'))
        ax.add_patch(Ellipse(centers[2], 2*majors[2], 2*minors[2], angle=angles[2], color='b', fill=False, label='A_k'))

        # x_lim = [min(ellipses[0][0][0]-ellipses[0][1], ellipses[1][0][0]-ellipses[1][1], ellipses[2][0][0]-ellipses[2][1]) - 1, \
        #          max(ellipses[0][0][0]+ellipses[0][1], ellipses[1][0][0]+ellipses[1][1], ellipses[2][0][0]+ellipses[2][1]) + 1]
        # y_lim = [min(ellipses[0][0][1]-ellipses[0][1], ellipses[1][0][1]-ellipses[1][1], ellipses[2][0][1]-ellipses[2][1]) - 1, \
        #          max(ellipses[0][0][1]+ellipses[0][1], ellipses[1][0][1]+ellipses[1][1], ellipses[2][0][1]+ellipses[2][1]) + 1]

        x_lim = [0, 1000]
        y_lim = [1000, 0]

        l12_vec = l_12.flatten()
        l23_vec = l_23.flatten()
        l31_vec = l_31.flatten()

        l12_x_vals = np.array(x_lim)
        l12_y_vals = (-l12_vec[0] * l12_x_vals - l12_vec[2]) / l12_vec[1]
        ax.plot(l12_x_vals, l12_y_vals, 'r', label='l12')
        
        l23_x_vals = np.array(x_lim)
        l23_y_vals = (-l23_vec[0] * l23_x_vals - l23_vec[2]) / l23_vec[1]
        ax.plot(l23_x_vals, l23_y_vals, 'g', label='l23')

        l31_x_vals = np.array(x_lim)
        l31_y_vals = (-l31_vec[0] * l31_x_vals - l31_vec[2]) / l31_vec[1]
        ax.plot(l31_x_vals, l31_y_vals, 'b', label='l31')

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True)
        ax.legend()
        # breakpoint()
        plt.show()


    ellipses = [get_ellipse_params(A) for A in As]
    l = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        A_i, A_j = As[i], As[j]
        c_i = np.array([ellipses[i]['center'][0], ellipses[i]['center'][1], 1.0])
        c_j = np.array([ellipses[j]['center'][0], ellipses[j]['center'][1], 1.0])
        
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

                g = normalize_vector(D[:, n]).reshape(3, 1)
                h = normalize_vector(D[m, :]).reshape(3, 1)

                epsilon = 1e-6
                valid_g_i = np.dot(g.flatten(), c_i)
                valid_g_j = np.dot(g.flatten(), c_j)
                valid_h_i = np.dot(h.flatten(), c_i)
                valid_h_j = np.dot(h.flatten(), c_j)

                # breakpoint()
                # visualize(ellipses, g, h)
                if valid_g_i * valid_g_j < 0 and abs(valid_g_i) > epsilon and abs(valid_g_j) > epsilon:
                    l.append(g)
                    valid_line_found = True
                    break
                elif valid_h_i * valid_h_j < 0 and abs(valid_h_i) > epsilon and abs(valid_h_j) > epsilon:
                    l.append(h)
                    valid_line_found = True
                    break
        
        if not valid_line_found:
            return None
    
    if len(l) < 3:
        return None
    
    l_12 = l[0]
    l_23 = l[1]
    l_31 = l[2]

    visualize(ellipses, l_12, l_23, l_31)

    term1 = (l_12.T @ A_stars[0] @ l_12).item() * (l_31.T @ A_stars[0] @ l_31).item()
    term2 = (l_12.T @ A_stars[1] @ l_12).item() * (l_23.T @ A_stars[1] @ l_23).item()
    term3 = (l_23.T @ A_stars[2] @ l_23).item() * (l_31.T @ A_stars[2] @ l_31).item()
    if term1 < 0 or term2 < 0 or term3 < 0:
        return None

    J1_val = np.abs(l_12.T @ A_stars[0] @ l_31).item() / np.sqrt(term1)
    J2_val = np.abs(l_12.T @ A_stars[1] @ l_23).item() / np.sqrt(term2)
    J3_val = np.abs(l_23.T @ A_stars[2] @ l_31).item() / np.sqrt(term3)
    
    if np.isnan(J1_val) or np.isnan(J2_val) or np.isnan(J3_val):
        return None
    if J1_val < 0.999 or J2_val < 0.999 or J3_val < 0.999:
        return [J1_val, J2_val, J3_val, False]
    
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

    J1 = np.arccosh(J1_val)
    J2 = np.arccosh(J2_val)
    J3 = np.arccosh(J3_val)

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
        T_M_C: Camera to Moon transformation matrix
        r_M: 3D position vector of camera in Moon frame
        Q_star: Disk Quadric from database

        Return: projected 2d conic envelope matrix, conic locus matrix
    """
    P_M_C = K @ T_M_C.T @ np.hstack([np.eye(3), -r_M.reshape(3, 1)])
    A_star = P_M_C @ Q_star @ P_M_C.T
    A_star = 0.5 * (A_star + A_star.T)

    A = get_adjugate(A_star)
    
    det_A = np.linalg.det(A)
    if det_A > 0:
        A = -A
        det_A = -det_A
    
    scale = np.abs(det_A) ** (1/3)
    if scale > 1e-10:
        A /= scale
    
    A = 0.5 * (A + A.T)

    # breakpoint()
    
    return A_star, A

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

def get_ellipse_params(A):
    a = A[0, 0]
    b = 2 * A[0, 1]
    c = A[1, 1]
    d = 2 * A[0, 2]
    f = 2 * A[1, 2]
    g = A[2, 2]

    det = a*c - (b/2)**2
    x0 = (b * f/2 - c * d) / (2 * det)
    y0 = (b * d/2 - a * f) / (2 * det)
    
    Y = np.array([
        [a, b/2],
        [b/2, c]
    ])

    eigvals, eigvecs = np.linalg.eig(Y)
    
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    f_prime = g + 0.5 * (d*x0 + f*y0)

    if eigvals[0] < 0 or eigvals[1] < 0 or f_prime > 0:
        return None
    
    a = np.sqrt(-f_prime / eigvals[0])
    b = np.sqrt(-f_prime / eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    if angle < 0:
        angle += 180

    return {'center': (x0, y0), 'major': a, 'minor': b, 'angle': angle}