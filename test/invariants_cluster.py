import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb

def main():
    with open('data/triads.pkl', 'rb') as f:
        triads = pickle.load(f)

    index_map = pd.read_csv('data/crater_index_db.csv')
    
    centroids = []
    descs = []
    for t in tqdm(triads):
        coords = np.array([[c['pos'][0], c['pos'][1]] for c in t['geoms']])
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)
        # pdb.set_trace()
        desc = index_map[(index_map['id1'] == t['id1']) & (index_map['id2'] == t['id2']) & (index_map['id3'] == t['id3'])]\
                    [['desc_0', 'desc_1', 'desc_2', 'desc_3', 'desc_4', 'desc_5', 'desc_6']].iloc[0].to_list()
        
        descs.append(desc)

    centroids = np.array(centroids)
    descs = np.array(descs)
    descs_tsne = TSNE(n_components=3).fit_transform(descs)

    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.scatter(invs_tsne[:, 0], invs_tsne[:, 1], color='b')
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(descs_tsne[:, 0], descs_tsne[:, 1], descs_tsne[:, 2], c=centroids[:, 0], cmap='viridis', s=1)
    plt.show()
    # plt.scatter(centroids[:][0], centroids[:][1], color='r')
    # plt.subplot()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(descs_tsne[:, 0], descs_tsne[:, 1], descs_tsne[:, 2], c=centroids[:, 1], cmap='viridis', s=1)
    plt.show()


if __name__ == '__main__':
    main()