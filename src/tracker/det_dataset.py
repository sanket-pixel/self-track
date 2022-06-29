from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import torch
import configparser

def get_transform(img_h,img_w):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(img_h, img_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_patch(box, img):
    x1,y1,x2,y2 = box
    patch = img[y1:y2,x1:x2]
    return patch

class DetectionDataset(Dataset):
    def __init__(self, boxes, frame, t_img_h,t_img_w):
        self.transform = get_transform(t_img_h,t_img_w)
        self.boxes = boxes
        self.frame = frame


    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        box =  self.boxes[idx][:4].astype(int)
        patch = self.transform(get_patch(box, self.frame))
        return patch

