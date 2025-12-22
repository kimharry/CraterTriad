import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle

# SWATH = 1.88 # Altitude: 3.5km
# SWATH = 8.04 # Altitude: 15km
SWATH = 16.08 # Altitude: 30km

def sort_clockwise(craters):
    coords = np.array([[c['pos'][0], c['pos'][1]] for c in craters])
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return [craters[i] for i in sorted_indices]

def main():
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)

    print(f"Loaded Craters: {len(craters)}")
    
    triads = []
    
    for comb in tqdm(combinations(craters, 3), desc="Triad generation"):
        c1, c2, c3 = comb
        
        d12 = np.linalg.norm(c1['pos'] - c2['pos'])
        d23 = np.linalg.norm(c2['pos'] - c3['pos'])
        d31 = np.linalg.norm(c3['pos'] - c1['pos'])

        # Swath check
        max_d = SWATH * np.sqrt(2)
        if (d12 > max_d) or (d23 > max_d) or (d31 > max_d):
            continue
        
        # Overlap check
        if (d12 < c1['a'] + c2['a']) or \
           (d23 < c2['a'] + c3['a']) or \
           (d31 < c3['a'] + c1['a']):
            continue
            
        sorted_comb = sort_clockwise([c1, c2, c3])
        
        triads.append({
            'id1': sorted_comb[0]['id'],
            'id2': sorted_comb[1]['id'],
            'id3': sorted_comb[2]['id'],
            'geoms': sorted_comb 
        })

    print(f"Valid Triads: {len(triads)}")
    
    with open('data/triads.pkl', 'wb') as f:
        pickle.dump(triads, f)
    print("Saved intermediate data: triads.pkl")

if __name__ == "__main__":
    main()