import os
from torchvision.ops import nms
import torch
import numpy as np
from torchvision.transforms import Normalize, Resize, Compose
from src.models.reid import metric_learning
import cv2
import pickle
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sequences(tracking_path):
    sequence_ids = os.listdir(tracking_path)
    sequence_list = []
    for seq_id in sequence_ids:
        if "FRCNN" in seq_id:
            sequence_list.append(seq_id)
    return sorted(sequence_list)


def perform_nms(det_df, nms_thres):
    tensor_df = torch.from_numpy(det_df.values)
    bbox = tensor_df[:, 2:6].clone()
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    scores = tensor_df[:, 6]
    nms_idx = nms(bbox, scores, nms_thres)
    return det_df.reset_index().iloc[nms_idx]


def convert_wh_to_corners(bbox):
    try:
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    except:
        print(bbox)
    return bbox

def convert_corners_to_wh(bbox):
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
    return bbox

def find_set_difference(t1, t2):
    combined = np.concatenate((t1, t2))
    uniques, counts = np.unique(combined, return_counts=True)
    difference = uniques[counts == 1]
    return difference


def get_bbox_image(frame_bbox, frame, idx):
    b = frame_bbox[idx]
    p = frame[b[1]:b[3], b[0]:b[2]]
    return p




def load_pretext_model(pretext_model_path, embedding_dim=2048,projection_dim=128):
    pretext_model = torch.nn.DataParallel(metric_learning.ReIDModel(embedding_dim=embedding_dim,projection_dim=projection_dim))
    checkpoint = torch.load(pretext_model_path, map_location=torch.device(device))
    pretext_model.load_state_dict(checkpoint['net'])
    pretext_model.to(device)  # send to device
    pretext_model.eval()  # set eval mode on
    return pretext_model

def get_transforms():
    resize = Resize(256)
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    return Compose([resize, normalize])

def get_embedding_bbox(detection_list, pretext_model):
    embedding_list = []
    # iterate over all detections and extract embeddings
    # start = timeit.default_timer()
    for i, detection in enumerate(detection_list):
        # make detection to float
        detection = (detection.unsqueeze(0) / 255.0).to(device)
        # get transform for normalize, resize
        transform = get_transforms()
        if detection.shape[-1] > 0 and detection.shape[-2] > 0:
            # apply transform to detection
            detection = transform(detection)
            # extract embedding
            with torch.no_grad():
                detection_embedding = pretext_model(detection)
            embedding_list.append(detection_embedding.cpu().numpy())
    detection_embeddings = np.stack(embedding_list).squeeze(1)
    return detection_embeddings


def show_bbox(frame_bbox, frame, idx):
    b = frame_bbox[idx].astype(int)
    f = frame.permute(1, 2, 0).numpy()
    p = f[b[1]:b[3], b[0]:b[2]]
    cv2.imshow("bbox", p)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def load_embedding_bbox(seq_id, det_mode,mode, tracking_path):
    cache_folder = os.path.join(tracking_path, seq_id, "cache", det_mode, mode)
    with open(os.path.join(cache_folder, 'embedding.pickle'), 'rb') as f:
        embedding = pickle.load(f)
    return embedding

def save_embeddings_bbox(embedding, seq_id,det_mode,mode, tracking_path):
    cache_folder = os.path.join(tracking_path, seq_id, "cache", det_mode, mode)
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cache_folder, 'embedding.pickle'), 'wb') as f:
        pickle.dump(embedding, f)