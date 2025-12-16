import pickle
import pandas as pd
from tqdm import tqdm
from utils import get_conic_matrix, calculate_invariants


def main():
    with open('data/triads_data5.pkl', 'rb') as f:
        triads = pickle.load(f)
    
    descriptors = []
    ids = []
    
    print("Calculating invariants...")
    for t in tqdm(triads):
        c1, c2, c3 = t['geoms']
        
        A1 = get_conic_matrix(c1)
        A2 = get_conic_matrix(c2)
        A3 = get_conic_matrix(c3)
        
        # Calculate 7 invariants
        invs = calculate_invariants(A1, A2, A3)
        
        # Make sorted descriptor
        desc = sorted(invs)
        
        descriptors.append(desc)
        ids.append([t['id1'], t['id2'], t['id3']])
        
    df_desc = pd.DataFrame(descriptors, columns=[f'inv_{i}' for i in range(7)])
    df_ids = pd.DataFrame(ids, columns=['id1', 'id2', 'id3'])
    
    final_df = pd.concat([df_ids, df_desc], axis=1)
    final_df.to_csv('data/crater_index_db5.csv', index=False)
    print(f"Index DB saved: crater_index_db5.csv ({len(final_df)} entries)")

if __name__ == "__main__":
    main()