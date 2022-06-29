import pandas as pd
import numpy as np
import os

dmr_dir = "/home/group-cvg/cvg-students/sshah/d_m_r"
data_dir = os.path.join('data')
tracklet_store_dir = os.path.join(data_dir,'tracklet_store')

sequences = os.listdir(os.path.join(dmr_dir,tracklet_store_dir))

total_tracks = 0
track_seq_list = []
for seq in sequences:
    track_list = []
    print(seq)
    if "MOT" in seq:
        pass
    else:
        seq_path = os.path.join(tracklet_store_dir, seq)
        tracks = sorted(os.listdir(os.path.join(dmr_dir, seq_path)))[:-2]
        for t in tracks:
            bboxes = pd.read_csv(os.path.join(dmr_dir, seq_path, t, "bboxes.csv"))
            bboxes['img_name'] = bboxes['t'].astype(int).astype(str) + '.png'
            bboxes['img_path'] = seq_path + "/" + t + "/" + "patches" + "/" + bboxes['img_name']
            bboxes = bboxes[bboxes['img_path'].apply(lambda x: os.path.exists(os.path.join(dmr_dir, x)))]
            track_list.append(bboxes)
        try:
            all_tracks_seq = pd.concat(track_list)
            all_tracks_seq['id'] = all_tracks_seq['id'] + total_tracks
            track_seq_list.append(all_tracks_seq)
            total_tracks = all_tracks_seq['id'].max() + 1
            print(total_tracks)
        except:
            pass

    all_tracks = pd.concat(track_seq_list)
    all_tracks.to_csv(os.path.join(dmr_dir, "data","reid_model","psuedo_gt","train","more_data_tracklets.csv"), index=False)

all_tracks = pd.read_csv(os.path.join(dmr_dir, "data","reid_model","psuedo_gt","train","more_data_tracklets.csv"))
all_tracks = all_tracks[~all_tracks['img_path'].str.contains("Charades").astype(bool)]
all_tracks=all_tracks.dropna()
tracklet_len = all_tracks.groupby(by='id').count()['t'].reset_index()
tracklet_len.rename({"t":"tracklet_length"},axis=1,inplace=True)
tracklet_len = tracklet_len.set_index(['id'])
all_tracks = all_tracks.join(tracklet_len, on='id')
all_tracks = all_tracks[all_tracks["tracklet_length"] > 50]
all_tracks.to_csv(os.path.join(dmr_dir,"data","reid_model","psuedo_gt","train","more_data_refined.csv"),index=False)