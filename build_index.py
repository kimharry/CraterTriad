import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle
from utils import sort_clockwise, calculate_invariants
from config import SWATH

def build_index():
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    print(f"Loaded Craters: {len(craters)}")
    
    index = {}
    
    for comb in tqdm(combinations(craters, 3), desc="Index generation", total=len(list(combinations(craters, 3)))):
        c1 = craters[comb[0]]
        c2 = craters[comb[1]]
        c3 = craters[comb[2]]
        
        d12 = np.linalg.norm(c1['pos'] - c2['pos'])
        d23 = np.linalg.norm(c2['pos'] - c3['pos'])
        d31 = np.linalg.norm(c3['pos'] - c1['pos'])

        # Swath check
        max_d = SWATH * np.sqrt(2)
        if (d12 > max_d) or (d23 > max_d) or (d31 > max_d):
            continue
        
        # Overlap check
        if (d12 < c1['b'] + c2['b']) or \
           (d23 < c2['b'] + c3['b']) or \
           (d31 < c3['b'] + c1['b']):
            continue
            
        c1, c2, c3 = sort_clockwise([c1, c2, c3])
        
        descriptor = calculate_invariants(c1['conic_matrix'], c2['conic_matrix'], c3['conic_matrix'])
        index[tuple(descriptor)] = [c1['id'], c2['id'], c3['id']]

    print(f"Valid Triads: {len(index)}")
    
    with open('data/index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print("Index saved to: index.pkl")

if __name__ == "__main__":
    build_index()