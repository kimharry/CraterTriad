import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle

R_MOON = 1737.4
# SWATH = 1.88 # Altitude: 3.5km
SWATH = 8.04 # Altitude: 15km

def lla_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = R_MOON * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_MOON * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_MOON * np.sin(lat_rad)
    
    return np.array([x, y, z])

def sort_clockwise(craters):
    coords = np.array([[c['pos'][0], c['pos'][1]] for c in craters])
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return [craters[i] for i in sorted_indices]

def main():
    df = pd.read_csv('data/filtered_craters_local.csv')
    
    craters = []
    for idx, row in df.iterrows():
        pos = lla_to_cartesian(row['LAT_ELLI_IMG'], row['LON_ELLI_IMG'])
        
        a = row['DIAM_ELLI_MAJOR_IMG'] / 2
        b = row['DIAM_ELLI_MINOR_IMG'] / 2
        theta = row['DIAM_ELLI_ANGLE_IMG']
        
        craters.append({
            'id': row['CRATER_ID'],
            'lat': row['LAT_ELLI_IMG'],
            'lon': row['LON_ELLI_IMG'],
            'pos': pos,
            'a': a,
            'b': b,
            'theta': theta
        })

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
    
    with open('data/triads_data5.pkl', 'wb') as f:
        pickle.dump(triads, f)
    print("Saved intermediate data: triads_data5.pkl")

if __name__ == "__main__":
    main()