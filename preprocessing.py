import pandas as pd
import matplotlib.pyplot as plt
from utils import get_center_vector, get_conic_locus, get_ENU_to_Moon_matrix, get_disk_quadric
import numpy as np
import pickle
from config import MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, MAX_AXIS_KM, MIN_ARC, MIN_PTS

def filter_craters(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Raw Data Loaded: {len(df):,}")
    except FileNotFoundError:
        print("[Error] File Not Found")
        return

    df = df.dropna(subset=['LAT_ELLI_IMG', 'LON_ELLI_IMG', 'DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG', 'DIAM_ELLI_ANGLE_IMG'])

    filtered_df = df[
        # location filter
        (df['LAT_ELLI_IMG'] >= MIN_LAT) & (df['LAT_ELLI_IMG'] <= MAX_LAT) &
        (df['LON_ELLI_IMG'] >= MIN_LON) & (df['LON_ELLI_IMG'] <= MAX_LON) &
        
        # size filter (major diameter)
        (df['DIAM_ELLI_MAJOR_IMG'] < MAX_AXIS_KM) &
        
        # quality filter
        (df['ARC_IMG'] >= MIN_ARC) & 
        (df['PTS_RIM_IMG'] >= MIN_PTS)
    ].copy()

    filtered_df.reset_index(drop=True, inplace=True)

    craters = {}
    for _, row in filtered_df.iterrows():
        lat = np.radians(row['LAT_ELLI_IMG'])
        lon = np.radians(row['LON_ELLI_IMG'])
        a = row['DIAM_ELLI_MAJOR_IMG'] / 2
        b = row['DIAM_ELLI_MINOR_IMG'] / 2
        theta = np.radians(row['DIAM_ELLI_ANGLE_IMG'])
        
        p_M = get_center_vector(lat, lon)
        T_E_M = get_ENU_to_Moon_matrix(p_M)
        conic_locus = get_conic_locus(theta, a, b)
        disk_quadric = get_disk_quadric(T_E_M, p_M, conic_locus)
        
        craters[row['CRATER_ID']] = {
            'id': row['CRATER_ID'],
            'lat': lat,
            'lon': lon,
            'pos': p_M,
            'a': a,
            'b': b,
            'theta': theta,
            'T_E_M': T_E_M,
            'conic_locus': conic_locus,
            'Q_star': disk_quadric
        }

    print("\nFiltered Data:")
    print(f" - Range: Lat {MIN_LAT}~{MAX_LAT}, Lon {MIN_LON}~{MAX_LON}")
    print(f" - Size: < {MAX_AXIS_KM} km")
    print(f" - Extracted Craters: {len(filtered_df):,}")
    
    # histogram
    # filtered_df['DIAM_ELLI_MAJOR_IMG'].hist(bins=50)
    # plt.title("Diameter Histogram")
    # plt.xlabel("Diameter (km)")
    # plt.ylabel("Frequency")
    # plt.show()
    
    with open(output_path, 'wb') as f:
        pickle.dump(craters, f)
    print(f"\nFiltered data saved to: {output_path}")

if __name__ == "__main__":
    filter_craters('data/lunar_crater_database_robbins_2018.csv', 'data/filtered_craters_local.pkl')