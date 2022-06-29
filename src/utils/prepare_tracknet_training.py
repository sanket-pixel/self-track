import os
import pandas as pd
import cv2
import random
import torch
from pathlib import Path
import configparser

os.chdir(os.path.join("..", ".."))


def get_sequences():
    sequences_files = sorted(os.listdir(os.path.join("data", "MOT17_split")))
    sequences = []
    for seq in sequences_files:
        sequences.append(seq.split(".csv")[0])
    return sequences


"""
The method prepares the dataloader text file
for tracknet. Reads all tracklets from all sequences and 
filters them before storing them.
"""


def prepare_tracknet_training_txt():
    seq_path = os.path.join("results", "bytetrack", "psuedo_gt", "sequences")
    track_filename = "tracks.txt"
    # get sequence list
    train_sequences = ['MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                       'MOT17-11-FRCNN', 'MOT17-13-FRCNN', 'MOT17-02-FRCNN',
                       'MOT17-10-FRCNN']
    track_list = []
    total_tracks = 0
    for i, seq in enumerate(train_sequences):
        # read seq h,w
        seq_info_path = os.path.join("data", "MOT17", "train", seq, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(seq_info_path)
        imWidth = config.getint("Sequence", "imWidth")
        imHeight = config.getint("Sequence", "imHeight")
        # read tracks for this sequence from results folder
        track_path = os.path.join(seq_path, seq, track_filename)
        track_df = pd.read_csv(track_path, header=None)
        # add column with sequence name
        track_df[9] = seq
        # add column with frame h and w
        track_df[10] = imWidth
        track_df[11] = imHeight
        # add to total tracks
        track_df[1] += total_tracks
        total_tracks = track_df[1].max() + 1
        # append to track list
        track_list.append(track_df)
    # concat all tracks from all sequences
    all_tracks = pd.concat(track_list).reset_index()
    # drop -1 -1 columns
    all_tracks.drop(["index", 7, 8], axis=1, inplace=True)
    # give meaningful names to columns
    all_tracks.rename(columns={0: "frame_id", 1: "track_id", 2: "x",
                               3: "y", 4: "w", 5: "h", 6: "conf", 9: "seq_id"}, inplace=True)
    # remove tracks with negative bboxes and all bboxes outside frame
    all_tracks = all_tracks[(all_tracks["x"] > 0) & (all_tracks["y"] > 0)]
    all_tracks = all_tracks[(all_tracks["x"] <= all_tracks[10]) & (all_tracks["y"] <= all_tracks[11])]
    # start storing training dataset
    dataset_filename = "MOT17_tracknet_training"
    dataset_path = os.path.join("data", dataset_filename)
    # get tracklet length of each track id
    tracklet_length = all_tracks.groupby(["track_id"]).count()["frame_id"].reset_index()
    tracklet_length.rename({"frame_id": "tracklet_length"}, axis=1, inplace=True)
    tracklet_length = tracklet_length.set_index(['track_id'])
    all_tracks = all_tracks.join(tracklet_length, on='track_id')
    # filter out all tracklets with length more than 100
    all_tracks = all_tracks[all_tracks["tracklet_length"] > 50]
    all_tracks = all_tracks[all_tracks['conf'] > 0.5]
    # get new track ids from 0 to len(track_ids)1
    unique_track_ids = all_tracks["track_id"].unique()
    track_id_idx = unique_track_ids.argsort()
    track_id_df = pd.DataFrame.from_dict(dict(zip(unique_track_ids, track_id_idx)), "index", columns=["s_track_id"])
    all_tracks = all_tracks.join(track_id_df, on='track_id')
    all_track_dataset_path = os.path.join(dataset_path, "MOT17_tracknet_training.txt")
    all_tracks.to_csv(all_track_dataset_path, index=False)
    print("Total tracks IDs generated : {t_id}".format(t_id=all_tracks['s_track_id'].max()))
    print("Total samples generated : {t_id}".format(t_id=len(all_tracks)))
    print("done")

def combine_train_val_txt():
    dataset_filename = "MOT17_tracknet_training"
    # dataset_filename = "MOT17_mini_tracknet_split"
    dataset_path = os.path.join("data", dataset_filename)
    train_comb_t = []
    val_comb_t = []
    for t in os.listdir(dataset_path):
        try:
            train_t = pd.read_csv(os.path.join(dataset_path, t, "train_tracks.txt"))
            train_comb_t.append(train_t)
            val_t = pd.read_csv(os.path.join(dataset_path, t, "val_tracks.txt"))
            val_comb_t.append(val_t)
        except:
            pass
    train_tracks = pd.concat(train_comb_t)
    val_tracks = pd.concat(val_comb_t)
    train_tracks.to_csv(os.path.join(dataset_path, "train_tracks.txt"), index=False)
    val_tracks.to_csv(os.path.join(dataset_path, "val_tracks.txt"), index=False)


def split_training_val():
    dataset_filename = "MOT17_tracknet_training"
    track_data = pd.read_csv(os.path.join("data", dataset_filename, "{data}.txt".format(data=dataset_filename)))
    track_data_train = track_data[track_data['s_track_id'] <=320]
    track_data_val = track_data[track_data['s_track_id'] > 320]
    track_train_path = os.path.join("data",dataset_filename, "MOT17_tracknet_training_train.txt")
    track_data_train.to_csv(track_train_path, index=False)
    track_val_path = os.path.join("data", dataset_filename, "MOT17_tracknet_training_val.txt")
    track_data_val.to_csv(track_val_path, index=False)


# prepare_tracknet_training_txt()
split_training_val()
# combine_train_val_txt()
