import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle

R_MOON = 1737.4

def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """Equirectangular Projection"""
    x = (lon - origin_lon) * (np.pi / 180) * R_MOON * np.cos(np.radians(origin_lat))
    y = (lat - origin_lat) * (np.pi / 180) * R_MOON
    return x, y

def sort_clockwise(craters):
    coords = np.array([[c['x'], c['y']] for c in craters])
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return [craters[i] for i in sorted_indices]

def main():
    df = pd.read_csv('data/filtered_craters_local3.csv')
    
    # Set projection origin
    origin_lat = df['LAT_ELLI_IMG'].mean()
    origin_lon = df['LON_ELLI_IMG'].mean()
    
    craters = []
    for idx, row in df.iterrows():
        x, y = latlon_to_xy(row['LAT_ELLI_IMG'], row['LON_ELLI_IMG'], origin_lat, origin_lon)
        
        a = row['DIAM_ELLI_MAJOR_IMG'] / 2
        b = row['DIAM_ELLI_MINOR_IMG'] / 2
        theta = row['DIAM_ELLI_ANGLE_IMG']
        
        craters.append({
            'id': row['CRATER_ID'],
            'x': x,
            'y': y,
            'a': a,
            'b': b,
            'theta': theta
        })

    print(f"Loaded Craters: {len(craters)}")
    
    triads = []
    
    for comb in tqdm(combinations(craters, 3), desc="Triad generation"):
        c1, c2, c3 = comb
        
        # Overlap check
        d12 = np.hypot(c1['x']-c2['x'], c1['y']-c2['y'])
        d23 = np.hypot(c2['x']-c3['x'], c2['y']-c3['y'])
        d31 = np.hypot(c3['x']-c1['x'], c3['y']-c1['y'])
        
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
    
    with open('data/triads_data.pkl', 'wb') as f:
        pickle.dump(triads, f)
    print("Saved intermediate data: triads_data.pkl")

if __name__ == "__main__":
    main()