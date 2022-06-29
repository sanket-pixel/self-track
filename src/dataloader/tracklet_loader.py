import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from src.utils.configuration import Configuration
import numpy as np
import cv2
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

# data loader config
motion_config = Configuration.load_json(os.path.join("src", "configs", "motion.json"))
training_data_path = motion_config.DataLoader.training_data_path
validation_data_path = motion_config.DataLoader.validation_data_path
warp_path = motion_config.DataLoader.warp_path
use_warp = motion_config.DataLoader.use_warp
vel_only = motion_config.DataLoader.vel_only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tracklet(Dataset):
    def __init__(self, mode="train", N=-1, dmr_path='.'):
        self.data_path = os.path.join(dmr_path, training_data_path)
        self.data_df = pd.read_csv(self.data_path)
        self.num_samples = self.data_df['b_idx'].max()
        self.dmr_path = dmr_path
        self.N = N
        self.warp_path = warp_path

    def __len__(self):
        return int(self.num_samples*0.05)

    def __getitem__(self, idx):
        track_data = self.data_df[self.data_df['b_idx'] == idx]
        # calculate center
        track_data['cx'] = track_data['x'] + (track_data['w'] / 2.0)
        track_data['cy'] = track_data['y'] + (track_data['h'] / 2.0)
        seq = track_data['seq'].unique()[0]
        # if warp mode is on, extract warp matrix
        if use_warp:
            warp_list_path = os.path.join(self.dmr_path, self.warp_path, f'warp_matrix_{seq}.pickle')
            with open(warp_list_path, 'rb') as handle:
                warp_arr = np.array(pickle.load(handle))
            warp_batch = warp_arr[track_data['t'].values.astype(int)-1][2:]
            warp_batch = torch.from_numpy(warp_batch).flatten(1, 2)

        # calculate velocities
        track_data[['vel_cx', 'vel_cy', 'vel_w', 'vel_h']] = track_data[['cx', 'cy', 'w', 'h']].diff(1)
        # if only velocities are needed, filter them out of features
        if vel_only:
            track_data = track_data[['vel_cx', 'vel_cy', 'vel_w', 'vel_h']]
        else:
            track_data = track_data[['cx', 'cy', 'w', 'h', 'vel_cx', 'vel_cy', 'vel_w', 'vel_h']]
            # track_data = track_data[['cx', 'cy', 'w', 'h']]
        # take first N-1 frames as input and the Nth frame as target
        input_features = track_data[1:-1]
        target_features = track_data[-1:]
        if use_warp:
            return torch.Tensor(input_features.values), warp_batch, torch.Tensor(target_features.values)
        else:
            return torch.Tensor(input_features.values), torch.Tensor(target_features.values)


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch])
    ## pad
    input = [torch.Tensor(t[0]) for t in batch]
    if use_warp:
        warp = torch.cat([t[1].unsqueeze(0) for t in batch]).permute(1,0,2)
        target = torch.concat([torch.Tensor(t[2]) for t in batch]).unsqueeze(0)
        input = torch.nn.utils.rnn.pad_sequence(input, padding_value=-100.0)
        ## compute mask
        mask = (input != -100.0)
        return input, warp, lengths, mask, target
    else:
        target = torch.concat([torch.Tensor(t[1]) for t in batch]).unsqueeze(0)
        input = torch.nn.utils.rnn.pad_sequence(input, padding_value=-100.0)
        ## compute mask
        mask = (input != -100.0)
        return input, lengths, mask, target
