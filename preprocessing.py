import pandas as pd
import matplotlib.pyplot as plt
from utils import lonlat_to_local_2d
import numpy as np
import pickle

def filter_craters(file_path, output_path):

    try:
        df = pd.read_csv(file_path)
        print(f"[Info] 원본 데이터 로드 완료: 총 {len(df):,}개")
    except FileNotFoundError:
        print("[Error] 파일을 찾을 수 없습니다.")
        return
    
    # 100km x 100km
    MIN_LAT = 43.4
    MAX_LAT = 46.7
    MIN_LON = 191.4
    MAX_LON = 196.2

    # size limit (semi-major axis)
    MAX_AXIS_KM = 50.0

    # quality limit
    # ARC_IMG: 림 보존율
    # PTS_RIM_IMG: 피팅에 사용된 점의 개수 (최소 5개 이상은 되어야 타원 피팅 가능)
    # DIAM_ELLI_ELLIP_IMG: 타원율
    MIN_ARC = 0.8
    MIN_PTS = 5

    df = df.dropna(subset=['LAT_ELLI_IMG', 'LON_ELLI_IMG', 'DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG', 'DIAM_ELLI_ANGLE_IMG'])

    filtered_df = df[
        # location filter
        (df['LAT_ELLI_IMG'] >= MIN_LAT) & (df['LAT_ELLI_IMG'] <= MAX_LAT) &
        (df['LON_ELLI_IMG'] >= MIN_LON) & (df['LON_ELLI_IMG'] <= MAX_LON) &
        
        # size filter (semi-major axis)
        (df['DIAM_ELLI_MAJOR_IMG'] < MAX_AXIS_KM) &
        
        # quality filter
        (df['ARC_IMG'] >= MIN_ARC) & 
        (df['PTS_RIM_IMG'] >= MIN_PTS)
    ].copy()

    filtered_df.reset_index(drop=True, inplace=True)

    craters = []
    for idx, row in filtered_df.iterrows():
        x, y = lonlat_to_local_2d(row['LAT_ELLI_IMG'], row['LON_ELLI_IMG'])
        
        a = row['DIAM_ELLI_MAJOR_IMG'] / 2
        b = row['DIAM_ELLI_MINOR_IMG'] / 2
        theta = row['DIAM_ELLI_ANGLE_IMG']
        
        craters.append({
            'id': row['CRATER_ID'],
            'lat': row['LAT_ELLI_IMG'],
            'lon': row['LON_ELLI_IMG'],
            'pos': np.array([x, y]),
            'a': a,
            'b': b,
            'theta': theta
        })


    print("\nFiltered Data:")
    print(f" - Range: Lat {MIN_LAT}~{MAX_LAT}, Lon {MIN_LON}~{MAX_LON}")
    print(f" - Size: < {MAX_AXIS_KM} km")
    print(f" - Extracted Craters: {len(filtered_df):,}개")
    
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