import numpy as np

def get_conic_matrix(x, y, a, b, theta_deg):
    """
    Convert crater parameters to 3x3 Conic Matrix A.
    """
    theta = np.radians(theta_deg)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    a2 = a**2
    b2 = b**2
    
    # Calculate coefficients based on paper's Eq. (10)~(15)
    A = a2 * sin_t**2 + b2 * cos_t**2
    B = 2 * (b2 - a2) * cos_t * sin_t
    C = a2 * cos_t**2 + b2 * sin_t**2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * x**2 + B * x * y + C * y**2 - a2 * b2
    
    # 3x3 matrix
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
    # 1. Normalize (|A| = 1)
    A1 = normalize_matrix(A1)
    A2 = normalize_matrix(A2)
    A3 = normalize_matrix(A3)
    
    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    A3_inv = np.linalg.inv(A3)
    
    # 2. Pairwise Invariants
    # Iij = Trace[Ai_inv * Aj]
    I12 = np.trace(A1_inv @ A2)
    I21 = np.trace(A2_inv @ A1)
    
    I23 = np.trace(A2_inv @ A3)
    I32 = np.trace(A3_inv @ A2)
    
    I31 = np.trace(A3_inv @ A1)
    I13 = np.trace(A1_inv @ A3)
    
    # 3. Triple Invariant
    # Iijk = Trace{ [(Aj + Ak)* - (Aj - Ak)*] @ Ai }
    Term1 = get_adjugate(A2 + A3)
    Term2 = get_adjugate(A2 - A3)
    I123 = np.trace((Term1 - Term2) @ A1)
    
    return [I12, I23, I31, I21, I32, I13, I123]