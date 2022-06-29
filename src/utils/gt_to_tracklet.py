import pandas as pd
import os
import cv2
import numpy as np
from pathlib import Path

d = '/home/group-cvg/cvg-students/sshah/d_m_r'

MOT_folder = 'data/MOT17/'

for seq in os.listdir(os.path.join(d, MOT_folder,"psuedo_gt")):
    print(seq)
    track_path = os.path.join(d, MOT_folder, "psuedo_gt",seq, "tracks.txt")
    tracks = pd.read_csv(track_path, header=None)
    tracks[1] = tracks[1].astype(int)
    frame_list = sorted(os.listdir(os.path.join(d, MOT_folder,"train", seq, "img1")))
    frame_list_img = []
    for f in frame_list:
        frame_list_img.append(cv2.imread(os.path.join(d, MOT_folder, "train", seq, "img1", f)))
    img_h, img_w,_ = frame_list_img[0].shape
    for t in tracks[1].unique():
        print(t)
        track_dir = os.path.join("data", "tracklet_store", f"psuedo_gt_{seq}", str(t))
        patch_dir = os.path.join(track_dir, "patches")
        Path(os.path.join(d, patch_dir)).mkdir(exist_ok=True, parents=True)
        tracklet = tracks[tracks[1] == t]
        tracklet = tracklet[tracklet[6] > 0.1]
        for i, det in enumerate(tracklet.values):
            frame_id = int(det[0]) - 1
            x, y, w, h = det[[2, 3, 4, 5]].astype(int)
            if x <0 :
                x = 0
            if x >= img_w:
                x=img_w
            if y < 0:
                y=0
            if y>=img_h:
                y=img_h
            frame = frame_list_img[frame_id]
            # frame = cv2.imread(os.path.join(os.path.join(d, MOT_folder, seq, "img1", frame_list[frame_id])))
            patch = frame[y:y + h, x:x + w]
            patch_path = os.path.join(d, patch_dir, '{i}.png'.format(i=i))
            try:
                cv2.imwrite(patch_path, patch)
            except:
                print("here")
        bbox_path = os.path.join(d, track_dir, "bboxes.csv")
        tracklet = tracklet[[0, 1, 2, 3, 4, 5, 6]]
        tracklet[0] = tracklet[0].astype(int)
        tracklet = tracklet.rename({0:'t',1:"id",2:"x",3:"y",4:"w",5:"h",6:"c"},axis=1)
        tracklet.to_csv(bbox_path, index=False, header=True)
