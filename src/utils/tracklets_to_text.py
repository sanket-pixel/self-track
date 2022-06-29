import pandas as pd
import os
import random
dmr_dir = '/home/group-cvg/cvg-students/sshah/d_m_r'

tracklet_store_path = os.path.join("data","tracklet_store")
# tracklet_sequences = sorted(os.listdir(os.path.join(dmr_dir, tracklet_store_path)))[:7]
MOT_train_seq = ['MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                       'MOT17-11-FRCNN', 'MOT17-13-FRCNN', 'MOT17-02-FRCNN',
                       'MOT17-10-FRCNN']
total_tracks = 0
track_seq_list = []
for seq in MOT_train_seq:
    print(seq)
    seq = f'psuedo_gt_{seq}'
    track_list = []
    seq_path = os.path.join(tracklet_store_path, seq)
    tracks = sorted(os.listdir(os.path.join(dmr_dir, seq_path)))[:-1]
    for t in tracks:
        track_path = os.path.join(seq_path,t,"patches")
        img_list = sorted(os.listdir(os.path.join(dmr_dir, track_path)))
        img_df = pd.DataFrame(img_list,columns=["img_name"])
        try:
            img_df['img_id'] = img_df['img_name'].str.split('.png', 1, expand=True)[0].astype(int)
        except:
            continue
        img_df['img_path'] = track_path +"/"+ img_df['img_name']
        bboxes = pd.read_csv(os.path.join(dmr_dir, seq_path, t, "bboxes.csv"))
        # bboxes = bboxes.rename({1: "id"}, axis=1)
        bboxes = bboxes.join(img_df.set_index(['img_id']))
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
all_tracks.to_csv(os.path.join(dmr_dir, "data","reid_model","psuedo_gt","train","MOT17_tracklets.csv"),index=False)

tracklet_training_text_file = os.path.join(dmr_dir, "data","reid_model","psuedo_gt","train","MOT17_tracklets.csv")
all_tracks = pd.read_csv(tracklet_training_text_file)
# all_tracks = all_tracks[~all_tracks['img_path'].str.contains("Charades").astype(bool)]
all_tracks=all_tracks.dropna()
tracklet_len = all_tracks.groupby(by='id').count()['t'].reset_index()
tracklet_len.rename({"t":"tracklet_length"},axis=1,inplace=True)
tracklet_len = tracklet_len.set_index(['id'])
all_tracks = all_tracks.join(tracklet_len, on='id')
all_tracks = all_tracks[all_tracks["tracklet_length"] > 50]
all_tracks.to_csv(os.path.join(dmr_dir, "data","reid_model","psuedo_gt","train","MOT17_tracklet_training_refined.csv"),index=False)

# tracklet_training_MOT17 = pd.read_csv(
#     os.path.join(dmr_dir, "data", "reid_model", "psuedo_gt", "train", "MOT17_refined.csv"))
# tracklet_training_extra = pd.read_csv(
#     os.path.join(dmr_dir, "data", "reid_model", "psuedo_gt", "train", "more_data_refined.csv"))
#
# all_ids = list(tracklet_training_extra.id.unique())
# # random.shuffle(all_ids)
# ratios = [25,50,75,100]
# for r in ratios:
#     print(r)
#     selected_ids = all_ids[: int(len(all_ids)*(r/100.0))]
#     selected_extra_data = tracklet_training_extra[tracklet_training_extra['id'].isin(selected_ids)]
#     max_id_extra = selected_extra_data.id.unique().max() + 1
#     tracklet_training_MOT17['id'] = tracklet_training_MOT17['id'] + max_id_extra
#     combined_tracklets = pd.concat([tracklet_training_MOT17, tracklet_training_extra], axis=0)
#     combined_tracklets.to_csv(os.path.join(dmr_dir,"data", "reid_model", "psuedo_gt", "train", f"MOT17_more_data_{r}.csv"), index=False)
