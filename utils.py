from math import sqrt
import numpy as np
import pdb

R_MOON = 1737.4
LAT_AVG = 45.05
LON_AVG = 193.8 

def lonlat_to_local_2d(lon, lat):
    """
    Project lonlat to local 2D plane
    """
    
    x_local = (lon - LON_AVG) * np.cos(lat)
    y_local = lat - LAT_AVG
    
    return x_local, y_local

def get_conic_matrix(crater, for_index=False):
    """
    Convert crater parameters to 3x3 Conic Matrix A.
    """
    if for_index:
        x_c, y_c = lonlat_to_local_2d(crater['lon'], crater['lat'])
        theta = np.radians(crater['theta'])
    else:
        x_c, y_c = crater['x_c'], crater['y_c']
        theta = crater['theta']

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    a2 = crater['a']**2
    b2 = crater['b']**2
    
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

    term1 = I12 + I23 + I31
    term2 = (2 * (I12**3 + I23**3 + I31**3) + 12*I12*I23*I31 - 3*(I12**2 * (I23 + I31) + I23**2 * (I31 + I12) + I31**2 * (I12 + I23))) \
            / (I12**2 + I23**2 + I31**2 - (I12*I23 + I23*I31 + I31*I12))
    term3 = -3 * sqrt(3) * (I12 - I23)*(I23 - I31)*(I31 - I12) \
            / (I12**2 + I23**2 + I31**2 - (I12*I23 + I23*I31 + I31*I12))
    term4 = I21 + I32 + I13
    term5 = 1.5 * (I12*I21 + I23*I32 + I31*I13) - 0.5 * (I12 + I23 + I31) * (I21 + I32 + I13)
    term6 = sqrt(3) / 2 * ((I12*I13 + I23*I21 + I31*I32) - (I12*I32 + I23*I13 + I31*I21))
    term7 = I123
    
    return [term1, term2, term3, term4, term5, term6, term7]