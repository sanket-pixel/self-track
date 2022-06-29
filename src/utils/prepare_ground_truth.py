from loguru import logger
import pandas as pd
import os
import configparser
from MOT_utils import get_sequences

os.chdir(os.path.join("..", ".."))

# read config file
config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "tracking.config"))
# dataloader params
MOT_folder = config.get("DataLoader", "MOT_folder")
split_path = config.get("DataLoader", "split_path")
detector = config.get("DataLoader", "detector")


"""
The method creates text files for train and validation by splitting 
the videos in two equal halves.
"""
def train_val_split(seq):
    gt_path = os.path.join(MOT_folder, seq, "gt/gt.txt")
    gt_df = pd.read_csv(gt_path, header=None)
    total_frames = len(gt_df[0].unique())
    train_half_idx = int(total_frames / 2)
    train_file = "gt_train_half.txt"
    train_path = os.path.join(MOT_folder,seq,"gt",train_file)
    gt_train_df = gt_df[gt_df[0] <= train_half_idx]
    gt_train_df = gt_train_df.sort_values(0)
    gt_train_df.to_csv(train_path,index=False)

    val_file = "gt_val_half.txt"
    gt_val_df = gt_df[gt_df[0] > train_half_idx]
    val_path = os.path.join(MOT_folder,seq,"gt",val_file)
    gt_val_df = gt_val_df.sort_values(0)
    gt_val_df.to_csv(val_path,index=False)


sequences = get_sequences(MOT_folder)
for seq in sequences:
    logger.info("Splitting for sequences : {seq}".format(seq=seq))
    train_val_split(seq)