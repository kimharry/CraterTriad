# lat: 45.9009, lon: 193.598 -> 02-1-149335 | {'pos': np.array([195.5, 182.5]), 'a': 164.125, 'b': 161.130, 'theta': 0.0}
# lat: 45.7187, lon: 194.319 -> 02-1-149307 | {'pos': np.array([835.5, 415.5]), 'a': 40.5, 'b': 38.5, 'theta': 0.0}
# lat: 45.6808, lon: 194.482 -> 02-1-149305 | {'pos': np.array([978.3, 459.63]), 'a': 46.06, 'b': 45.83, 'theta': np.radians(22.76)}
# lat: 45.5168, lon: 193.796 -> 02-1-144582 | {'pos': np.array([372.48, 677.23]), 'a': 101.835, 'b': 89.0, 'theta': np.radians(97.94)}

import pickle
import numpy as np
from utils import get_conic_matrix,calculate_invariants
from itertools import combinations
from pose_estimation import estimate_pose, validate_pose

with open('data/filtered_craters_local.pkl', 'rb') as f:
    craters = pickle.load(f)

with open('data/index.pkl', 'rb') as f:
    index = pickle.load(f)

ids = ['02-1-149335', '02-1-149307', '02-1-149305', '02-1-144582']

detections = {}
detections['02-1-149335'] = {'pos': np.array([195.5, 182.5]), 'a': 164.125, 'b': 161.130, 'theta': 0.0}
detections['02-1-149307'] = {'pos': np.array([835.5, 415.5]), 'a': 40.5, 'b': 38.5, 'theta': 0.0}
detections['02-1-149305'] = {'pos': np.array([978.3, 459.63]), 'a': 46.06, 'b': 45.83, 'theta': np.radians(22.76)}
detections['02-1-144582'] = {'pos': np.array([372.48, 677.23]), 'a': 101.835, 'b': 89.0, 'theta': np.radians(97.94)}

for comb in combinations(ids, 3):
    C1, C2, C3 = [craters[id]['conic_matrix'] for id in comb]
    invar_GT = np.array(calculate_invariants(C1, C2, C3))

    detection = [detections[id] for id in comb]

    A1 = get_conic_matrix(detection[0]['theta'], detection[0]['a'], detection[0]['b'])
    A2 = get_conic_matrix(detection[1]['theta'], detection[1]['a'], detection[1]['b'])
    A3 = get_conic_matrix(detection[2]['theta'], detection[2]['a'], detection[2]['b'])
    invar_est = np.array(calculate_invariants(A1, A2, A3))

    # print(invar_GT)
    # print(invar_est)
    # print(np.linalg.norm(invar_GT - invar_est))
    # print("\n")

    T_M_C = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    r_M = estimate_pose([A1, A2, A3], [craters[id] for id in comb], T_M_C)
    print(r_M)
    print(validate_pose([craters[id] for id in comb], detection, r_M, T_M_C))
    print("\n")