import pickle
import numpy as np
from utils import *
from config import K

def identify_craters(descriptors):
    with open('data/index.pkl', 'rb') as f:
        index = pickle.load(f)
    
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    invar_matches = {}
    k = 5
    for index_key in index.keys():
        invar_matches[index_key] = np.linalg.norm(np.array(index_key) - descriptors)
        
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
    C1 = match[0]['conic_locus']
    C2 = match[1]['conic_locus']
    C3 = match[2]['conic_locus']

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

def validate_pose(detect, match, r_M, T_M_C, sigma_img=1.0):
    """
        detect: Current triad of craters (sorted)
        match: Identified crater triad match
        r_M: Estimated camera position
        T_M_C: Moon to Camera transformation matrix
        sigma_img: Image noise standard deviation
        
        return: True if pose is valid, False otherwise
    """

    if r_M is None:
        return False
    if np.isnan(r_M).any() or np.isinf(r_M).any():
        return False

    chi_squares = []
    for k in range(len(match)):
        _, A_proj = proj_db2img(T_M_C, r_M, match[k]['Q_star'])
        y_i, Y_i = conic_to_yY(A_proj)
        
        y_j = np.array(detect[k]['pos']).reshape(2, 1)
        a_j = detect[k]['a']
        b_j = detect[k]['b']
        theta_j = detect[k]['theta']
        Y_j = np.array([[np.cos(theta_j), -np.sin(theta_j)], [np.sin(theta_j), np.cos(theta_j)]]) @ \
              np.array([[1/a_j**2, 0], [0, 1/b_j**2]]) @ \
              np.array([[np.cos(theta_j), np.sin(theta_j)], [-np.sin(theta_j), np.cos(theta_j)]])

        ga = d_GA(y_i, y_j, Y_i, Y_j)
        var = variance(match[k]['a'], match[k]['b'], sigma_img)

        chi_square = ga**2 / var
        if chi_square > 13.277 or np.isnan(chi_square) or np.isinf(chi_square): # 99% confidence
            return False, None
        chi_squares.append(chi_square)

    return True, np.mean(chi_squares)

def main(detections, T_M_C):
    """
        detections: includes 'pos(x, y)', 'a', 'b', 'theta'
        T_M_C: Moon to Camera Transformation Matrix (T_C_M: Camera to Moon Transformation Matrix, T_C_M = T_M_C.T)
    """
    if len(detections) < 3:
        return None

    combs = EPS(detections, 3)
    # combs = combs[::-1]

    estimated_poses = []
    for comb in combs:
        detect = sort_clockwise(comb)

        # overlap check
        if np.linalg.norm(detect[0]['pos'] - detect[1]['pos']) < detect[0]['a'] + detect[1]['a'] or \
           np.linalg.norm(detect[1]['pos'] - detect[2]['pos']) < detect[1]['a'] + detect[2]['a'] or \
           np.linalg.norm(detect[2]['pos'] - detect[0]['pos']) < detect[2]['a'] + detect[0]['a']:
            continue

        A1 = get_conic_locus(detect[0]['theta'], detect[0]['a'], detect[0]['b'], x_c=detect[0]['pos'][0], y_c=detect[0]['pos'][1])
        A2 = get_conic_locus(detect[1]['theta'], detect[1]['a'], detect[1]['b'], x_c=detect[1]['pos'][0], y_c=detect[1]['pos'][1])
        A3 = get_conic_locus(detect[2]['theta'], detect[2]['a'], detect[2]['b'], x_c=detect[2]['pos'][0], y_c=detect[2]['pos'][1])

        A1_star = get_adjugate(A1)
        A2_star = get_adjugate(A2)
        A3_star = get_adjugate(A3)
        
        descriptors = calculate_invariants([A1, A2, A3], [A1_star, A2_star, A3_star])
        if descriptors is None:
            print("Invariants calculation failed: None")
            continue
        elif descriptors[-1] == -1:
            print("Invariants calculation failed: -1")
            continue

        descriptors = np.array(descriptors[:-1])

        best_mean_chi_square = float('inf')
        best_match = None
        for _ in range(3):
            matches = identify_craters(descriptors)
            if len(matches) == 0:
                print("No matches found")
                break
            
            for match in matches:
                r_M = estimate_pose([A1, A2, A3], match, T_M_C)
                is_valid, mean_chi_square = validate_pose(detect, match, r_M, T_M_C)
                if is_valid and mean_chi_square < best_mean_chi_square:
                    best_mean_chi_square = mean_chi_square
                    best_match = r_M
            
            descriptors = np.roll(descriptors, 1)
        
        if best_match is not None:
            estimated_poses.append(best_match)

    if len(estimated_poses) == 0:
        print("No valid pose found")
        return None
    
    return np.mean(estimated_poses, axis=0)

if __name__ == "__main__":
    detections = [
        {'pos': np.array([181.0, 171.0]), 'a': 147.0, 'b': 138.985, 'theta': 0.0}, # lat: 45.9009, lon: 193.598 -> 02-1-149335
        {'pos': np.array([752.88, 377.22]), 'a': 37.495, 'b': 35.225, 'theta': np.radians(105.32)}, # lat: 45.7187, lon: 194.319 -> 02-1-149307
        {'pos': np.array([880.84, 421.95]), 'a': 38.125, 'b': 33.485, 'theta': np.radians(5.77)}, # lat: 45.6808, lon: 194.482 -> 02-1-149305
        {'pos': np.array([342.33, 615.2]), 'a': 89.645, 'b': 79.955, 'theta': np.radians(72.89)} # lat: 45.5168, lon: 193.796 -> 02-1-144582
    ]

    lat_gt = np.radians(45.6141) 
    lon_gt = np.radians(193.99626)
    T_M_C = get_TMC(lat_gt, lon_gt)
    r_M = main(detections, T_M_C)
    print("\nFinal estimated pose:", r_M)

    r_M_gt = np.array([-1214.613115, -302.75296, 1278.901654]).reshape(3, 1)
    err_dist = np.linalg.norm(r_M - r_M_gt)
    print(f"Error distance: {err_dist} km")