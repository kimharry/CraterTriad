import pickle
import numpy as np
from utils import sort_clockwise, get_2d_conic_matrix, EPS, calculate_invariants, get_center_vector, proj_db2img, conic_to_yY, d_GA, variance
from config import T_M_C, K
import pdb

def identify_craters(descriptors):
    with open('data/index.pkl', 'rb') as f:
        index = pickle.load(f)
    
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    invar_matches = {}
    k = 5
    for index_key in index.keys():
        invar_matches[index_key] = np.linalg.norm(np.array(index_key) - np.array(descriptors))
        
    # Sort by distance and return top k
    sorted_invar_matches = dict(sorted(invar_matches.items(), key=lambda x: x[1]))
    crater_matches = []
    for i in range(k):
        invar = list(sorted_invar_matches.keys())[i]
        ids = index[invar]
        crater_matches.append([craters[ids[0]], craters[ids[1]], craters[ids[2]]])

    return crater_matches

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

def validate_pose(match, detect_triad, r_M, T_M_C, sigma_img=1.0):
    """
        match: Identified crater triad match
        detect_triad: Current triad of craters (sorted)
        r_M: Estimated camera position
        T_M_C: Moon to Camera transformation matrix
        
        return: True if pose is valid, False otherwise
    """

    if r_M is None:
        return False
    if np.isnan(r_M).any() or np.isinf(r_M).any():
        return False

    for i in range(len(match)):
        A_proj = proj_db2img(T_M_C, r_M, match[i]['Q_star'])
        y_i, Y_i = conic_to_yY(A_proj)
        
        y_j = np.array(detect_triad[i]['pos']).T
        R = np.array([[np.cos(detect_triad[i]['theta']), -np.sin(detect_triad[i]['theta'])],
                      [np.sin(detect_triad[i]['theta']),  np.cos(detect_triad[i]['theta'])]])
        Y_j = R @ np.diag([1/detect_triad[i]['a'], 1/detect_triad[i]['b']])

        ga = d_GA(y_i, y_j, Y_i, Y_j)
        var = variance(match[i]['a'], match[i]['b'], sigma_img)

        chi_square = ga**2 / var
        print(f"Crater {i}: Chi-square = {chi_square}")
        if chi_square > 13.277 or np.isnan(chi_square) or np.isinf(chi_square): # 99% confidence
            print()
            return False

    print()
    return True

def main(detections, T_M_C):
    """
        detections: includes 'pos(x, y)', 'a', 'b', 'theta'
        T_M_C: Moon to Camera Transformation Matrix (T_C_M: Camera to Moon Transformation Matrix, T_C_M = T_M_C.T)
    """
    if len(detections) < 3:
        return None

    combs = EPS(detections, 3)
    # combs = combs[::-1]

    for comb in combs:
        detect_triad = sort_clockwise(comb)

        # overlap check
        if np.linalg.norm(detect_triad[0]['pos'] - detect_triad[1]['pos']) < detect_triad[0]['a'] + detect_triad[1]['a'] or \
           np.linalg.norm(detect_triad[1]['pos'] - detect_triad[2]['pos']) < detect_triad[1]['a'] + detect_triad[2]['a'] or \
           np.linalg.norm(detect_triad[2]['pos'] - detect_triad[0]['pos']) < detect_triad[2]['a'] + detect_triad[0]['a']:
            continue

        A1 = get_2d_conic_matrix(detect_triad[0]['theta'], detect_triad[0]['a'], detect_triad[0]['b'])
        A2 = get_2d_conic_matrix(detect_triad[1]['theta'], detect_triad[1]['a'], detect_triad[1]['b'])
        A3 = get_2d_conic_matrix(detect_triad[2]['theta'], detect_triad[2]['a'], detect_triad[2]['b'])

        pos1 = np.linalg.inv(K) @ np.hstack([detect_triad[0]['pos'], 1])
        pos2 = np.linalg.inv(K) @ np.hstack([detect_triad[1]['pos'], 1])
        pos3 = np.linalg.inv(K) @ np.hstack([detect_triad[2]['pos'], 1])
        
        for _ in range(3):
            # pdb.set_trace()
            descriptors = calculate_invariants([A1, A2, A3], [pos1, pos2, pos3])
            if descriptors == None:
                print("Invariants calculation failed")
                continue
            matches = identify_craters(descriptors)
            if len(matches) == 0:
                # print("No matches found")
                break
            
            # pdb.set_trace()
            for match in matches:
                r_M = estimate_pose([A1, A2, A3], match, T_M_C)
                # print("Estimated pose:", r_M)
                if validate_pose(match, detect_triad, r_M, T_M_C):
                    return r_M
            
            # print("Not a valid pose \n")
            A1, A2, A3 = A2, A3, A1
            pos1, pos2, pos3 = pos2, pos3, pos1

    print("No valid pose found")
    return None

if __name__ == "__main__":
    # testing
    # detections = [
    #     {'pos': np.array([195.5, 182.5]), 'a': 164.125, 'b': 161.130, 'theta': 0.0}, # lat: 45.91146, lon: 193.59334
    #     {'pos': np.array([835.5, 415.5]), 'a': 40.5, 'b': 38.5, 'theta': 0.0}, # lat: 45.72357, lon: 194.31649
    #     {'pos': np.array([978.3, 459.63]), 'a': 46.06, 'b': 45.83, 'theta': np.radians(22.76)}, # lat: 45.68981, lon: 194.47532
    #     {'pos': np.array([372.48, 677.23]), 'a': 101.835, 'b': 89.0, 'theta': np.radians(97.94)} # lat: 45.51962, lon: 193.78878
    # ]
    
    detections = [
        {'pos': np.array([181.0, 171.0]), 'a': 147.0, 'b': 138.985, 'theta': 0.0}, # lat: 45.9009, lon: 193.598 -> 02-1-149335
        {'pos': np.array([752.88, 377.22]), 'a': 37.495, 'b': 35.225, 'theta': np.radians(105.32)}, # lat: 45.7187, lon: 194.319 -> 02-1-149307
        {'pos': np.array([880.84, 421.95]), 'a': 38.125, 'b': 33.485, 'theta': np.radians(5.77)}, # lat: 45.6808, lon: 194.482 -> 02-1-149305
        {'pos': np.array([342.33, 615.2]), 'a': 89.645, 'b': 79.955, 'theta': np.radians(72.89)} # lat: 45.5168, lon: 193.796 -> 02-1-144582
    ]

    r_M = main(detections, T_M_C)
    print("Estimated pose:", r_M)