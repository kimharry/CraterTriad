import random
import numpy as np
import pandas as pd
import tqdm

index_map = pd.read_csv('data/crater_index_db6.csv')
index = index_map.values
gts = index[:1000]
random.shuffle(index)

corrects = 0
wrongs = 0
for gt in tqdm.tqdm(gts):
    for triad in index:
        # add noise to triad
        inv = triad[3:] + np.random.normal(0, 0.1, 7)
        
        dist = np.linalg.norm(gt[3:] - inv)
        if dist < 0.1:
            if gt[0] == triad[0] and gt[1] == triad[1] and gt[2] == triad[2]:
                corrects += 1
            else:
                wrongs += 1
            break


print(corrects, wrongs)