# lat: 45.9009, lon: 193.598 -> 02-1-149335 | {'pos': np.array([181.0, 171.0]), 'a': 147.0, 'b': 138.985, 'theta': 0.0}
# lat: 45.7187, lon: 194.319 -> 02-1-149307 | {'pos': np.array([752.88, 377.22]), 'a': 37.495, 'b': 35.225, 'theta': np.radians(105.32)}
# lat: 45.6808, lon: 194.482 -> 02-1-149305 | {'pos': np.array([880.84, 421.95]), 'a': 38.125, 'b': 33.485, 'theta': np.radians(5.77)}
# lat: 45.5168, lon: 193.796 -> 02-1-144582 | {'pos': np.array([342.33, 615.2]), 'a': 89.645, 'b': 79.955, 'theta': np.radians(72.89)}

import pickle
import numpy as np
from utils import get_conic_matrix, calculate_invariants, sort_clockwise
from itertools import combinations, permutations
from pose_estimation import estimate_pose, validate_pose

with open('data/filtered_craters_local.pkl', 'rb') as f:
    craters = pickle.load(f)

with open('data/index.pkl', 'rb') as f:
    index = pickle.load(f)

ids = ['02-1-149335', '02-1-149307', '02-1-149305', '02-1-144582']

detections = {}
detections['02-1-149335'] = {'id': '02-1-149335', 'pos': np.array([181.0, 171.0]), 'a': 147.0, 'b': 138.985, 'theta': 0.0}
detections['02-1-149307'] = {'id': '02-1-149307', 'pos': np.array([752.88, 377.22]), 'a': 37.495, 'b': 35.225, 'theta': np.radians(105.32)}
detections['02-1-149305'] = {'id': '02-1-149305', 'pos': np.array([880.84, 421.95]), 'a': 38.125, 'b': 33.485, 'theta': np.radians(5.77)}
detections['02-1-144582'] = {'id': '02-1-144582', 'pos': np.array([342.33, 615.2]), 'a': 89.645, 'b': 79.955, 'theta': np.radians(72.89)}

for comb in combinations(ids, 3):
    crater1, crater2, crater3 = [craters[id] for id in comb]
    crater1, crater2, crater3 = sort_clockwise([crater1, crater2, crater3])
    invar_GT = None
    for invars, id_set in index.items():
        if set(id_set) == {crater1['id'], crater2['id'], crater3['id']}:
            invar_GT = invars
            break

    if invar_GT is None:
        print(f"No GT invariants found for {comb}")
        continue

    detection = [detections[id] for id in comb]
    
    for perm in permutations(range(3)):
        detection_perm = [detection[i] for i in perm]
        A1 = get_conic_matrix(detection_perm[0]['theta'], detection_perm[0]['a'], detection_perm[0]['b'])
        A2 = get_conic_matrix(detection_perm[1]['theta'], detection_perm[1]['a'], detection_perm[1]['b'])
        A3 = get_conic_matrix(detection_perm[2]['theta'], detection_perm[2]['a'], detection_perm[2]['b'])
        invar_est = np.array(calculate_invariants(A1, A2, A3))

        # find nearest neighbor's id
        min_dist = 1e10
        nearest_key = None
        for index_key in index.keys():
            distance = np.linalg.norm(np.array(index_key) - invar_est)
            if distance < min_dist:
                min_dist = distance
                nearest_key = index_key
        
        ids = index[nearest_key]
        print(f"Ground truth IDs: {[d['id'] for d in detection]}")
        print(f"Nearest neighbor IDs: {ids}")
        print(f"Min distance: {min_dist}")
        print(f"Error: {np.linalg.norm(invar_GT - invar_est)}")
        print()

        # detection[0], detection[1], detection[2] = detection[1], detection[2], detection[0]
        

    print("-" * 50)
    print()

    # T_M_C = np.array([
    #     [0.0, 0.0, 1.0],
    #     [0.0, 1.0, 0.0],
    #     [-1.0, 0.0, 0.0]
    # ])
    # r_M = estimate_pose([A1, A2, A3], [craters[id] for id in comb], T_M_C)
    # print(r_M)
    # print(validate_pose([craters[id] for id in comb], detection, r_M, T_M_C))
    # print("\n")