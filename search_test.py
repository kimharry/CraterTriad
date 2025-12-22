import numpy as np
import pandas as pd
import random
import tqdm

index_map = pd.read_csv('data/crater_index_db6.csv')
index = index_map.values

sim_cnts = []
for gt in tqdm.tqdm(index):
    cnt = 0
    for triad in index:
        dist = np.linalg.norm(gt[3:] - triad[3:])
        if dist < 0.1:
            if dist == 0:
                if cnt != 0:
                    sim_cnts.append(cnt)
                break
            else:
                cnt += 1


print(sim_cnts)