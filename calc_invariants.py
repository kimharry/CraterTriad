import pickle
import pandas as pd
from tqdm import tqdm
from utils import get_conic_matrix, calculate_invariants


def main():
    with open('data/triads.pkl', 'rb') as f:
        triads = pickle.load(f)
    
    index = []
    
    print("Calculating invariants...")
    for t in tqdm(triads):
        c1, c2, c3 = t['geoms']
        
        A1 = get_conic_matrix(c1, for_index=True)
        A2 = get_conic_matrix(c2, for_index=True)
        A3 = get_conic_matrix(c3, for_index=True)

        index.append([t['id1'], t['id2'], t['id3']]+calculate_invariants(A1, A2, A3))
    
    df = pd.DataFrame(index, columns=['id1', 'id2', 'id3', 'desc_0', 'desc_1', 'desc_2', 'desc_3', 'desc_4', 'desc_5', 'desc_6'])
    df.to_csv('data/crater_index_db.csv', index=False)
    print(f"Index DB saved: crater_index_db.csv ({len(df)} entries)")

if __name__ == "__main__":
    main()