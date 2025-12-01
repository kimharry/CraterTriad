import pandas as pd

def filter_craters(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {len(df):,} craters")
    except FileNotFoundError:
        print("File Not Found. Please check the path.")
        return
    
    MIN_LAT = 40.0
    MAX_LAT = 43.3
    MIN_LON = 196.7
    MAX_LON = 200.0

    MIN_DIAM_KM = 1.0
    MAX_DIAM_KM = 40.0

    # MIN_ARC_VALIDITY = 0.9
    # MAX_ELLIPTICITY = 1.2

    col_lat = 'LAT_CIRC_IMG'
    col_lon = 'LON_CIRC_IMG'
    col_diam = 'DIAM_CIRC_IMG'
    # col_arc = 'ARC_IMG'
    # col_ellip = 'ELLIPTICITY_ROBUST'

    filtered_df = df[
        (df[col_lat] >= MIN_LAT) & (df[col_lat] <= MAX_LAT) &
        (df[col_lon] >= MIN_LON) & (df[col_lon] <= MAX_LON) &
        (df[col_diam] >= MIN_DIAM_KM) & (df[col_diam] < MAX_DIAM_KM)
        # (df[col_arc] >= MIN_ARC_VALIDITY) & 
        # (df[col_ellip] <= MAX_ELLIPTICITY)
    ].copy()
    
    print("\nResult:")
    print(f" - Spatial Range: Lat {MIN_LAT}~{MAX_LAT}, Lon {MIN_LON}~{MAX_LON}")
    print(f" - Scale Range: {MIN_DIAM_KM}km ~ {MAX_DIAM_KM}km")
    print(f" - Num. Craters: {len(filtered_df):,}개")
    
    filtered_df.reset_index(drop=True, inplace=True)
    
    cols_to_save = ['CRATER_ID', col_lat, col_lon, col_diam, 'ANGLE_CIRC_IMG']
    available_cols = [c for c in cols_to_save if c in filtered_df.columns]
    
    filtered_df[available_cols].to_csv(output_path, index=False)
    print(f"\nFiltered data saved to: {output_path}")

    return filtered_df

if __name__ == "__main__":
    path = 'data/lunar_crater_database_robbins_2018.csv'
    result = filter_craters(path, 'data/filtered_craters_local.csv')