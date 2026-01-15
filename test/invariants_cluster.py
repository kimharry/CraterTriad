import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open('data/filtered_craters_local.pkl', 'rb') as f:
        craters = pickle.load(f)
    with open('data/index.pkl', 'rb') as f:
        index = pickle.load(f)
    
    centroids = []
    descs = []
    for desc, ids in index.items():
        coords = np.array([craters[id]['pos'] for id in ids])
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)

        descs.append(desc)

    centroids = np.array(centroids)

    centroids = centroids / np.linalg.norm(centroids, axis=0, keepdims=True)
    descs = np.array(descs)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(descs[:, 0], descs[:, 1], descs[:, 2], c=centroids[:, 0], cmap='viridis', s=1)
    # plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(descs[:, 0], descs[:, 1], descs[:, 2], c=centroids[:, 1], cmap='viridis', s=1)
    # plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(descs[:, 0], descs[:, 1], descs[:, 2], c=centroids[:, 2], cmap='viridis', s=1)
    # plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].hist(descs[:, 0], bins=50, color='r', alpha=0.5, label='Invariant 1')
    ax[1].hist(descs[:, 1], bins=50, color='g', alpha=0.5, label='Invariant 2')
    ax[2].hist(descs[:, 2], bins=50, color='b', alpha=0.5, label='Invariant 3')

    ax[0].set_ylim(0, 31000)
    ax[0].set_xlabel('Invariant 1')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].set_xlabel('Invariant 2')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    ax[1].set_ylim(0, 30000)
    ax[2].set_xlabel('Invariant 3')
    ax[2].set_ylabel('Frequency')
    ax[2].legend()
    ax[2].set_ylim(0, 30000)
    plt.show()

if __name__ == '__main__':
    main()