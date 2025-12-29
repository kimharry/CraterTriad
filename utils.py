import numpy as np

R_MOON = 1737.4
LAT_AVG = 45.05
LON_AVG = 193.8

def sort_clockwise(craters):
    coords = np.array([[c['pos'][0], c['pos'][1]] for c in craters])
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return [craters[i] for i in sorted_indices]

def lonlat_to_local_2d(lon, lat):
    """
    Project lonlat to local 2D plane
    """
    
    x_local = (lon - LON_AVG) * np.cos(lat)
    y_local = lat - LAT_AVG
    
    return x_local, y_local

def get_conic_matrix(theta, a, b, x_c=0, y_c=0):
    """
    Convert crater parameters to 3x3 Conic Matrix A.
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
    
    I123 = np.trace((get_adjugate(A2 + A3) - get_adjugate(A2 - A3)) @ A1)
    
    return [I12, I21, I23, I32, I31, I13, I123]

def get_center_vector(lat, lon):
    return R_MOON * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def get_ENU_to_Moon_matrix(lat, lon):
    """
        Returns the transformation matrix from local ENU coordinate to Moon-centered coordinate
    """
    k = np.array([0, 0, 1])
    p = get_center_vector(lat, lon)

    u = p / np.linalg.norm(p)
    e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
    n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

    return np.array([e, n, u])

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