import random
import numpy as np
import pandas as pd
import tqdm

index_map = pd.read_csv('data/crater_index_db.csv')
index = index_map.values
samples = index[:1000]
random.shuffle(index)

# add noise to samples
for i in range(len(samples)):
    samples[i][3:] += np.random.normal(0, 10, 7)

corrects = 0
wrongs = 0
for sample in tqdm.tqdm(samples):
    for triad in index:
        dist = np.linalg.norm(triad[3:] - sample[3:])
        if dist < 1:
            if sample[0] == triad[0] and sample[1] == triad[1] and sample[2] == triad[2]:
                corrects += 1
            else:
                wrongs += 1
            break


print(corrects, wrongs)