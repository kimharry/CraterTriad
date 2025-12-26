import pickle
import numpy as np
from utils import sort_clockwise, get_conic_matrix, get_center_vector, get_transformation_matrix
from itertools import combinations

R_MOON = 1737.4

def identify_craters(detections):

    with open('data/index.pkl', 'rb') as f:
        index = pickle.load(f)
    
    matches = []
    nn = 3
    min_dist = 1e10
    for index_key in index.keys():
        distance = (np.array(index_key) - np.array(descriptors)).norm()
        if distance < min_dist:
            min_dist = distance
            matches.append(index[index_key])
            if len(matches) > nn:
                matches.pop(0)
    return matches

def estimate_pose(detections, K, T_M_C):
    """
        detections: includes 'pos(x, y)', 'a', 'b', 'theta'
        K: Camera Intrinsic Matrix
        T_M_C: Moon to Camera Transformation Matrix (T_C_M: Camera to Moon Transformation Matrix, T_C_M = T_M_C.T)
    """
    
    for comb in combinations(detections, 3):
        comb = sort_clockwise(comb)
        matches = identify_craters(comb)
        if len(matches) == 0:
            continue
        
        