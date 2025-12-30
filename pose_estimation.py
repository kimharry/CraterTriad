import pickle
import numpy as np
from utils import sort_clockwise, get_conic_matrix, EPS, calculate_invariants, get_center_vector
import pdb

K = np.array([
    [2218.77, 0.0, 550.0],
    [0.0, 2218.77, 550.0],
    [0.0, 0.0, 1.0]
])

def identify_craters(descriptors):
    with open('data/index.pkl', 'rb') as f:
        index = pickle.load(f)
    
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    matches = []
    k = 3
    min_dist = 1e10
    for index_key in index.keys():
        distance = np.linalg.norm(np.array(index_key) - np.array(descriptors))
        if distance < min_dist:
            min_dist = distance
            ids = index[index_key]
            matches.append([craters[id] for id in ids])
            if len(matches) > k:
                matches.pop(0)
    print(min_dist)
    return matches

def estimate_pose(detect, match, T_M_C):
    A1, A2, A3 = detect
    C1 = match[0]['conic_matrix']
    C2 = match[1]['conic_matrix']
    C3 = match[2]['conic_matrix']

    p_M1 = get_center_vector(match[0]['lat'], match[0]['lon'])
    p_M2 = get_center_vector(match[1]['lat'], match[1]['lon'])
    p_M3 = get_center_vector(match[2]['lat'], match[2]['lon'])

    k = np.array([0, 0, 1]).T
    
    B1 = T_M_C.T @ K.T @ A1 @ K @ T_M_C
    B2 = T_M_C.T @ K.T @ A2 @ K @ T_M_C
    B3 = T_M_C.T @ K.T @ A3 @ K @ T_M_C
    
    # 3 x 2 selective matrix
    # [I_2x2 
    #  0_1x2]
    S = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])

    temp1 = (S.T @ C1 @ S).flatten()
    temp2 = (S.T @ C2 @ S).flatten()
    temp3 = (S.T @ C3 @ S).flatten()
    
    T_E1_M = match[0]['T_E_M']
    T_E2_M = match[1]['T_E_M']
    T_E3_M = match[2]['T_E_M']

    s_hat_1 = (temp1 @ (S.T @ T_E1_M.T @ B1 @ T_E1_M @ S).flatten()) / (temp1.T @ temp1)
    s_hat_2 = (temp2 @ (S.T @ T_E2_M.T @ B2 @ T_E2_M @ S).flatten()) / (temp2.T @ temp2)
    s_hat_3 = (temp3 @ (S.T @ T_E3_M.T @ B3 @ T_E3_M @ S).flatten()) / (temp3.T @ temp3)

    H = np.vstack([
        S.T @ T_E1_M.T @ B1,
        S.T @ T_E2_M.T @ B2,
        S.T @ T_E3_M.T @ B3
    ])
    y = np.vstack([
        S.T @ T_E1_M.T @ B1 @ p_M1 - s_hat_1 * S.T @ C1 @ k,
        S.T @ T_E2_M.T @ B2 @ p_M2 - s_hat_2 * S.T @ C2 @ k,
        S.T @ T_E3_M.T @ B3 @ p_M3 - s_hat_3 * S.T @ C3 @ k
    ]).reshape(-1, 1)
    
    r_M = np.linalg.pinv(H.T @ H) @ H.T @ y
    return r_M

def proj_db2img(T_M_C, r_M, p_M, a, b, theta, T_E_M):
    """
        T_M_C: Moon to Camera transformation matrix
        r_M: 3D position vector of camera in Moon frame
        p_M: Crater center in Moon frame
        a, b: Semi-major and semi-minor axes (km)
        theta: Rotation angle of crater (radians)
        T_E_M: Crater's local ENU frame transformation matrix

        Return: y, Y parameters for the projected crater
    """

    t_C = T_M_C @ (p_M.reshape(-1, 1) - r_M) # camera to crater vector
    R_E_C = T_M_C @ T_E_M # rotation of crater local frame to camera frame

    H = K @ np.column_stack([R_E_C[:, 0], R_E_C[:, 1], t_C])
    H_inv = np.linalg.pinv(H)
    R_2D = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])

    C_local = R_2D @ np.diag([1/a**2, 1/b**2, -1]) @ R_2D.T

    # pdb.set_trace()
    A_proj = H_inv.T @ C_local @ H_inv
    A_proj /= np.linalg.norm(A_proj)

    A_uu = A_proj[:2, :2]
    A_u1 = A_proj[:2, 2]
    A_11 = A_proj[2, 2]

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

def validate_pose(match, comb, r_M, T_M_C, sigma_img=1.0):
    """
        match: Identified crater triad match
        comb: Current combination of craters
        r_M: Estimated camera position
        T_M_C: Moon to Camera transformation matrix
        
        return: True if pose is valid, False otherwise
    """

    if r_M is None:
        return False
    if np.isnan(r_M).any() or np.isinf(r_M).any():
        return False

    for i in range(len(match)):
        y_i, Y_i = proj_db2img(T_M_C, r_M, get_center_vector(match[i]['lat'], match[i]['lon']), \
            match[i]['a'], match[i]['b'], match[i]['theta'], match[i]['T_E_M'])
        
        y_j = np.array(comb[i]['pos']).T
        R = np.array([[np.cos(comb[i]['theta']), -np.sin(comb[i]['theta'])],
                      [np.sin(comb[i]['theta']),  np.cos(comb[i]['theta'])]])
        Y_j = R @ np.diag([1/comb[i]['a'], 1/comb[i]['b']])

        ga = d_GA(y_i, y_j, Y_i, Y_j)
        var = variance(comb[i]['a'], comb[i]['b'], sigma_img)

        print("Chi-square:", ga**2 / var)
        if ga**2 / var > 13.277: # 99% confidence
            return False

    return True

def main(detections, T_M_C):
    """
        detections: includes 'pos(x, y)', 'a', 'b', 'theta'
        T_M_C: Moon to Camera Transformation Matrix (T_C_M: Camera to Moon Transformation Matrix, T_C_M = T_M_C.T)
    """
    if len(detections) < 3:
        return None

    combs = EPS(detections, 3)

    for comb in combs:
        comb = sort_clockwise(comb)

        # overlap check
        if np.linalg.norm(comb[0]['pos'] - comb[1]['pos']) < comb[0]['a'] + comb[1]['a'] or \
           np.linalg.norm(comb[1]['pos'] - comb[2]['pos']) < comb[1]['a'] + comb[2]['a'] or \
           np.linalg.norm(comb[2]['pos'] - comb[0]['pos']) < comb[2]['a'] + comb[0]['a']:
            continue

        A1 = get_conic_matrix(comb[0]['theta'], comb[0]['a'], comb[0]['b'])
        A2 = get_conic_matrix(comb[1]['theta'], comb[1]['a'], comb[1]['b'])
        A3 = get_conic_matrix(comb[2]['theta'], comb[2]['a'], comb[2]['b'])
        
        for _ in range(3):
            descriptors = calculate_invariants(A1, A2, A3)
            matches = identify_craters(descriptors)
            if len(matches) == 0:
                # print("No matches found")
                break

            for match in matches:
                r_M = estimate_pose([A1, A2, A3], match, T_M_C)
                # print("Estimated pose:", r_M)
                if validate_pose(match, comb, r_M, T_M_C):
                    return r_M
            
            # print("Not a valid pose \n")
            A1, A2, A3 = A2, A3, A1

    # print("No valid pose found")
    return None

if __name__ == "__main__":
    # testing
    detections = [
        {'pos': np.array([195.5, 182.5]), 'a': 164.125, 'b': 161.130, 'theta': 0.0}, # lat: 45.91146, lon: 193.59334
        {'pos': np.array([835.5, 415.5]), 'a': 40.5, 'b': 38.5, 'theta': 0.0}, # lat: 45.72357, lon: 194.31649
        {'pos': np.array([978.3, 459.63]), 'a': 46.06, 'b': 45.83, 'theta': np.radians(22.76)}, # lat: 45.68981, lon: 194.47532
        {'pos': np.array([372.48, 677.23]), 'a': 101.835, 'b': 89.0, 'theta': np.radians(97.94)} # lat: 45.51962, lon: 193.78878
    ]
    
    T_M_C = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])

    r_M = main(detections, T_M_C)
    print("Estimated pose:", r_M)