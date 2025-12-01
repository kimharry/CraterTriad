import pandas as pd
from scipy.spatial import KDTree
import pickle

def main():
    print("Loading data...")
    df = pd.read_csv('data/crater_index_db.csv')
    
    feature_cols = [c for c in df.columns if c.startswith('inv_')]
    features = df[feature_cols].values
    
    crater_ids = df[['id1', 'id2', 'id3']].values
    
    print("Building KD-Tree...")
    tree = KDTree(features)
    
    index_data = {
        'tree': tree,
        'ids': crater_ids
    }
    
    with open('data/crater_kdtree.pkl', 'wb') as f:
        pickle.dump(index_data, f)
        
    print("Index file saved: crater_kdtree.pkl")
    print("Now you can use this file to perform real-time matching.")

if __name__ == "__main__":
    main()