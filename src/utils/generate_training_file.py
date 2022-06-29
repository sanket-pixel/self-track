import os.path
import random

import pandas as pd
import numpy as np

os.chdir(os.path.join("..", ".."))
data_dir = os.path.join("data","MOT17_tracknet_training")

tracks = sorted(os.listdir(data_dir))[:-1]

max_tracks = 10
min_tracks = 5
minimum_samples = 10
max_samples = 128
total_batches = 1000
all_patches = []
for _ in range(total_batches):
    t = np.random.randint(min_tracks, max_tracks)
    track_ids = random.sample(tracks,t)
    patch_list = []
    extra_patch_list = []
    for tid in track_ids:
        patches = pd.read_csv(os.path.join("data", "MOT17_tracknet_training", tid, "all_tracks.txt"))
        selected_patches = patches.sample(minimum_samples)
        extra_patches = patches.sample(50)
        patch_list.append(selected_patches)
        extra_patch_list.append(extra_patches)

    selected_patch_all = pd.concat(patch_list)
    extra_patch_all = pd.concat(extra_patch_list)
    remaining_patches = extra_patch_all.sample(max_samples - len(selected_patch_all))
    total_patches = pd.concat([selected_patch_all,remaining_patches]).sample(frac=1)
    all_patches.append(total_patches)

all_patches = pd.concat(all_patches)
all_patches.to_csv("data/training_data.txt",index=False)

