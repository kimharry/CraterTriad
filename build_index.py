import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle
from utils import sort_clockwise, calculate_invariants, proj_db2img
from config import SWATH, T_M_C

def build_index():
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    print(f"Loaded Craters: {len(craters)}")
    print(f"Swath: {SWATH}")
    
    index = {}
    swath_cnt = 0
    overlap_cnt = 0
    err_codes = []
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
        A = [proj_db2img(T_M_C, c1['pos'], c1['Q_star']), proj_db2img(T_M_C, c2['pos'], c2['Q_star']), proj_db2img(T_M_C, c3['pos'], c3['Q_star'])]
        descriptor = calculate_invariants(A)
        if type(descriptor) is not int:
            index[tuple(descriptor)] = [c1['id'], c2['id'], c3['id']]
        
        else:
            err_codes.append(descriptor)

    print(f"Valid Triads: {len(index)}")
    print(f"Swath Filtered: {swath_cnt}")
    print(f"Overlap Filtered: {overlap_cnt}")

    print(f"No Valid z found: {len(err_codes) - sum(err_codes)}")
    print(f"h/g errors: {sum(err_codes)}")
    
    with open('data/index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print("Index saved to: index.pkl")

if __name__ == "__main__":
    build_index()