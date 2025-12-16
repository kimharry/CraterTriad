import numpy as np
import pdb

R_MOON = 1737.4

def get_enu_vecs(center_vec):
    """
    Get e, n, u vectors from center vector of triad
    """
    u = center_vec / np.linalg.norm(center_vec)
    
    k = np.array([0, 0, 1])
    
    # pdb.set_trace()
    
    e = np.cross(k, u)
    e = e / np.linalg.norm(e)
        
    n = np.cross(u, e)
    n = n / np.linalg.norm(n)
    
    return e, n, u

def project_crater_to_plane(crater):
    """
    Project 3D crater to 2D plane
    """
    lat = np.radians(crater['lat'])
    lon = np.radians(crater['lon'])
    center_vec = R_MOON * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    
    E, N, U = get_enu_vecs(center_vec)
    
    rel_pos = crater['pos'] - center_vec
    
    x_local = np.dot(rel_pos, E) # East 성분
    y_local = np.dot(rel_pos, N) # North 성분
    
    return x_local, y_local

def get_conic_matrix(crater):
    """
    Convert crater parameters to 3x3 Conic Matrix A.
    """
    x, y = project_crater_to_plane(crater)

    theta = np.radians(crater['theta'])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    a2 = crater['a']**2
    b2 = crater['b']**2
    
    A = a2 * sin_t**2 + b2 * cos_t**2
    B = 2 * (b2 - a2) * cos_t * sin_t
    C = a2 * cos_t**2 + b2 * sin_t**2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * x**2 + B * x * y + C * y**2 - a2 * b2
    
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

def calculate_invariants(A1, A2, A3):
    A1 = normalize_matrix(A1)
    A2 = normalize_matrix(A2)
    A3 = normalize_matrix(A3)
    
    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    A3_inv = np.linalg.inv(A3)
    
    I12 = np.trace(A1_inv @ A2)
    I21 = np.trace(A2_inv @ A1)
    
    I23 = np.trace(A2_inv @ A3)
    I32 = np.trace(A3_inv @ A2)
    
    I31 = np.trace(A3_inv @ A1)
    I13 = np.trace(A1_inv @ A3)
    
    Term1 = get_adjugate(A2 + A3)
    Term2 = get_adjugate(A2 - A3)
    I123 = np.trace((Term1 - Term2) @ A1)
    
    return [I12, I23, I31, I21, I32, I13, I123]