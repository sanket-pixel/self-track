import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import torch
from src.utils.configuration import Configuration

reid_config = Configuration.load_json(os.path.join("src", "configs", "reid.json"))
data_path_train = reid_config.DataLoader.training_data_path
data_path_val = reid_config.DataLoader.validation_data_path
MOT_folder = reid_config.DataLoader.MOT_folder
t_img_h = reid_config.Training.t_img_h
t_img_w = reid_config.Training.t_img_w
det_conf_filter = reid_config.DataLoader.det_conf_filter

def get_transform(mode="train"):
    if mode=="auto_augment":
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.AutoAugment(),
                                        transforms.Resize(size=(t_img_h,t_img_w),
                                                          interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    elif mode=="no_augment":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(t_img_h,t_img_w),  interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


def get_sequences(MOT_folder):
    sequences = sorted(os.listdir(MOT_folder))
    frame_dict = {}
    for s in sequences:
        try:
            frames= sorted(os.listdir(os.path.join(MOT_folder, s, "img1")))
            frame_dict[s] = frames
        except:
            pass
    return frame_dict


def to_np(t):
    return t.permute(1, 2, 0).cpu().numpy()


def get_patch(frame, bbox):
    patch = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return patch


def get_num_classes(data_df):
    return len(data_df.id.unique())


class TrackInstance(Dataset):

    def __init__(self, mode="train", transform_mode="auto_augment", dmr_path = "."):
        self.mode = mode
        self.transform = get_transform(transform_mode)
        self.sequences = get_sequences(os.path.join(dmr_path,MOT_folder))
        if mode == "train":
            self.data_path = os.path.join(dmr_path,data_path_train)
            self.data_df = pd.read_csv(self.data_path)
            self.data_df = self.data_df[self.data_df['c'] > det_conf_filter]

        elif mode=="val":
            self.data_path = os.path.join(dmr_path,data_path_val)
            self.data_df = pd.read_csv(self.data_path)
            self.data_df = self.data_df.drop(['6','7'], axis=1)

        self.num_classes = get_num_classes(self.data_df)
        self.classes = np.sort(self.data_df.id.unique())
        self.dmr_path = dmr_path

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        track_instance_info = self.data_df.iloc[idx].values
        patch_path = os.path.join(self.dmr_path, track_instance_info[8])
        patch = cv2.imread(patch_path)
        track_id = track_instance_info[1]
        patch = self.transform(patch)
        return patch, track_id

    def get_weights(self):
        class_sample_count = np.array(
            [len(np.where(self.data_df.id == t)[0]) for t in self.classes])
        weight = 1. / class_sample_count
        target = self.data_df['id'].values
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        return samples_weigth
