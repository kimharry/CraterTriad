import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle
from utils import sort_clockwise, calculate_invariants, proj_db2img, get_center_vector, get_TMC
from config import SWATH, R_MOON, ALTITUDE
import pdb

def build_index():
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    print(f"Loaded Craters: {len(craters)}")
    print(f"Swath: {SWATH}")
    
    index = {}
    swath_cnt = 0
    overlap_cnt = 0
    less_than_1 = []
    err_cnt = 0
    for comb in tqdm(combinations(craters, 3), total=len(list(combinations(craters, 3))), desc="Building index"):
        c1 = craters[comb[0]]
        c2 = craters[comb[1]]
        c3 = craters[comb[2]]
        
        d12 = np.linalg.norm(c1['pos'] - c2['pos'])
        d23 = np.linalg.norm(c2['pos'] - c3['pos'])
        d31 = np.linalg.norm(c3['pos'] - c1['pos'])

        # Swath check
        max_d = SWATH * np.sqrt(2)
        if (d12 > max_d) or (d23 > max_d) or (d31 > max_d):
            swath_cnt += 1
            continue
        
        # Overlap check
        if (d12 < c1['b'] + c2['b']) or \
           (d23 < c2['b'] + c3['b']) or \
           (d31 < c3['b'] + c1['b']):
            overlap_cnt += 1
            continue
            
        c1, c2, c3 = sort_clockwise([c1, c2, c3])
        center_lat = (c1['lat'] + c2['lat'] + c3['lat']) / 3
        center_lon = (c1['lon'] + c2['lon'] + c3['lon']) / 3
        r_M = get_center_vector(center_lat, center_lon, r=R_MOON + ALTITUDE)
        T_M_C = get_TMC(center_lat, center_lon)

        A, c = [], []
        for crater in [c1, c2, c3]:
            conic, center = proj_db2img(T_M_C, r_M, crater['Q_star'], np.eye(3))
            A.append(conic)
            c.append(center)

        # pdb.set_trace()
        descriptor = calculate_invariants(A, c)

        if descriptor is None:
            err_cnt += 1
        elif descriptor[-1] == False:
            if descriptor[0] < 1:
                less_than_1.append(descriptor[0])
            if descriptor[1] < 1:
                less_than_1.append(descriptor[1])
            if descriptor[2] < 1:
                less_than_1.append(descriptor[2])
        else:
            index[tuple(descriptor)] = [c1['id'], c2['id'], c3['id']]

    print(f"Valid Triads: {len(index)}")
    print(f"Swath Filtered: {swath_cnt}")
    print(f"Overlap Filtered: {overlap_cnt}")
    print(f"Invariants Calculation Failed: {err_cnt}")
    print(f"Less than 1: {len(less_than_1)}")
    pdb.set_trace()
    with open('data/index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print("Index saved to: index.pkl")

if __name__ == "__main__":
    build_index()