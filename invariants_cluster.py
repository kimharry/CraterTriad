import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb

R_MOON = 1737.4

def get_center_vector(crater):
    lat = crater['lat']
    lon = crater['lon']
    
    return R_MOON * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def main():
    with open('data/triads_data6.pkl', 'rb') as f:
        triads = pickle.load(f)

    index_map = pd.read_csv('data/crater_index_db6.csv')
    
    centroids = []
    invs = []
    for t in tqdm(triads):
        coords = np.array([[c['pos'][0], c['pos'][1]] for c in t['geoms']])
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)
        # pdb.set_trace()
        inv = index_map[(index_map['id1'] == t['id1']) & (index_map['id2'] == t['id2']) & (index_map['id3'] == t['id3'])]\
                    [['inv_0', 'inv_1', 'inv_2', 'inv_3', 'inv_4', 'inv_5', 'inv_6']].iloc[0].to_list()
        
        invs.append(inv)

    centroids = np.array(centroids)
    invs = np.array(invs)
    invs_tsne = TSNE(n_components=3).fit_transform(invs)

    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.scatter(invs_tsne[:, 0], invs_tsne[:, 1], color='b')
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(invs_tsne[:, 0], invs_tsne[:, 1], invs_tsne[:, 2], c=centroids[:, 0], cmap='viridis', s=1)
    plt.show()
    # plt.scatter(centroids[:][0], centroids[:][1], color='r')
    # plt.subplot()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(invs_tsne[:, 0], invs_tsne[:, 1], invs_tsne[:, 2], c=centroids[:, 1], cmap='viridis', s=1)
    plt.show()


if __name__ == '__main__':
    main()