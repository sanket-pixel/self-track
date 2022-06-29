import pandas as pd
import os
import numpy as np
from pathlib import Path
pd.options.mode.chained_assignment = None  # default='warn'


dmr_path = '/home/group-cvg/cvg-students/sshah/d_m_r'
MOT17_folder = 'data/MOT17/psuedo_gt'
MOT20_folder = 'data/MOT20/train'





def create_tracks(sequences,track_path,MOT_folder):
    gt_tracklets = []
    max_id = 0
    for seq in sequences:
        print(seq)
        MOT_path = os.path.join(dmr_path, MOT_folder)
        gt_tracks = pd.read_csv(os.path.join(MOT_path, seq, 'tracks.txt'), header=None, names = ['t','id','x','y','w','h','c','d','e'])
        gt_tracks = gt_tracks[gt_tracks['c'] >0]
        gt_tracks['new_id'] = gt_tracks['id'] + max_id
        gt_tracks['seq'] = seq
        max_id = gt_tracks['new_id'].max()
        gt_tracklets.append(gt_tracks)
    gt_all = pd.concat(gt_tracklets)
    gt_all.to_csv(os.path.join(dmr_path,track_path),index=False)

def create_batches(track_path, offset,seq_length,batch_path):
    MOT17_gt = pd.read_csv(os.path.join(dmr_path, track_path))
    samples = []
    b_idx = 0
    for id in MOT17_gt['new_id'].unique():
        gt_id = MOT17_gt[MOT17_gt['new_id']==id]
        gt_id = gt_id.reset_index()
        start_id = np.arange(len(gt_id), step=offset)
        end_id = start_id + seq_length
        for s_idx, e_idx in tuple(zip(start_id, end_id)):
            if e_idx < len(gt_id):
                sample = gt_id[s_idx:e_idx]
                if sample['t'].diff(1).max() == 1.0:
                    sample['b_idx'] = b_idx
                    b_idx +=1
                    print(b_idx)
                    samples.append(sample)
    sample_all = pd.concat(samples)
    sample_all.to_csv(os.path.join(dmr_path,batch_path),index=False)


def create_random_sized_batches(track_path, offset,seq_length,batch_path):
    MOT17_gt = pd.read_csv(os.path.join(dmr_path, track_path))
    samples = []
    b_idx = 0
    for id in MOT17_gt['new_id'].unique():
        gt_id = MOT17_gt[MOT17_gt['new_id']==id]
        gt_id = gt_id.reset_index()
        start_id = np.arange(len(gt_id), step=offset)
        end_id = start_id + np.random.randint(5, seq_length, start_id.shape[0])
        for s_idx, e_idx in tuple(zip(start_id, end_id)):
            if e_idx < len(gt_id):
                sample = gt_id[s_idx:e_idx]
                if sample['t'].diff(1).max() == 1.0:
                    sample['b_idx'] = b_idx
                    b_idx +=1
                    print(b_idx)
                    samples.append(sample)
    sample_all = pd.concat(samples)
    sample_all.to_csv(os.path.join(dmr_path,batch_path),index=False)


def main():
    offset = 1

    seq_length = 100

    # training batch creation
    train_path = os.path.join(dmr_path, "data", "motion_model", "psuedo_gt", "train","seq_{}".format(str(seq_length)))
    Path(train_path).mkdir(parents=True,exist_ok=True)
    train_seq = ['MOT17-04-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', 'MOT17-09-FRCNN', 'MOT17-05-FRCNN','MOT17-10-FRCNN', 'MOT17-02-FRCNN']
    create_tracks(train_seq, track_path= os.path.join(train_path,"tracks.txt"),MOT_folder=MOT17_folder)
    create_batches(track_path=os.path.join(train_path,"tracks.txt"),offset=offset, seq_length=seq_length,batch_path=os.path.join(train_path,"batches.txt"))

    # c(track_path=os.path.join(train_path,"tracks.txt"),offset=offset, seq_length=seq_length,batch_path=os.path.join(train_path,"batches.txt"))

    # validation batch creation
    # validation_path = os.path.join(dmr_path, "data", "motion_model", "gt", "validation","seq_{}".format(str(seq_length)))
    # Path(validation_path).mkdir(parents=True,exist_ok=True)
    # validation_seq = ['MOT20-01', 'MOT20-02', 'MOT20-03','MOT20-05']
    # create_tracks(validation_seq, track_path= os.path.join(validation_path,"tracks.txt"),MOT_folder=MOT20_folder)
    # create_batches(track_path=os.path.join(validation_path,"tracks.txt"),offset=offset, seq_length=seq_length,batch_path=os.path.join(validation_path,"batches.txt"))

if __name__ == '__main__':
    main()