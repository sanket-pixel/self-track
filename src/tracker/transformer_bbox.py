import numpy as np
import os
import cv2
import configparser
from src.models.motion.tbmot import MotionModel
import torch
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_age = 20
min_box_area = 100
'''Motion Model'''

config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "tracking.config"))
max_age = config.getint("TrackerParams", "max_age")

arch = config.get("MotionModel", "arch")
motion_cp_path = config.get("MotionModel", "motion_cp_path")
feature_size = config.getint("MotionModel", "feature_size")
num_layers = config.getint("MotionModel", "num_layers")
nhead = config.getint("MotionModel", "nhead")
bbox_dim = config.getint("MotionModel", "bbox_dim")
warp_dim = config.getint("MotionModel", "warp_dim")
N = config.getint("MotionModel", "N")
warp_path = config.get("MotionModel", "warp_path")
dmr_path = config.get("TrackerInfo", "dmr_path")
use_warp = config.getboolean("MotionModel", "use_warp")
vel_only = config.getboolean("MotionModel", "vel_only")
train_mode = config.getboolean("TrackerInfo", "train_mode")
dataset = config.get("DataLoader","dataset")

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class NNTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """

    def __init__(self, boxes, embeddings, scores, frame_id, model=None,seq=None):
        # initialize id, time since update hitrs, hit streak, etc
        self.count = boxes.shape[0]
        self.history = np.array([convert_bbox_to_z(boxes)])
        self.match_flag = np.ones((self.count, 1)).astype(np.bool)
        self.added_flag = np.ones((self.count, 1)).astype(np.bool)
        self.time_since_update = np.zeros(self.count)
        self.id = np.arange(self.count)
        self.frame_id = np.array([frame_id] * self.count)
        self.seq = seq
        self.hits = np.zeros(self.count)
        self.hit_streak = np.zeros(self.count)
        self.age = np.zeros(self.count)
        if frame_id == 1:
            self.status = np.ones(self.count)
        else:
            self.status = np.zeros(self.count)
        self.embeddings = embeddings
        self.scores = scores

        self.use_warp = use_warp
        if train_mode:
            self.motion_model = model
        else:
            # load model and checkpoint
            self.motion_model = MotionModel(arch=arch, feature_size=feature_size, num_layers=num_layers, nhead=nhead,
                                            bbox_dim=bbox_dim,
                                            warp_dim=warp_dim, use_warp=use_warp).to(device)
            checkpoint = torch.load(os.path.join(dmr_path, motion_cp_path), map_location=torch.device(device))
            self.motion_model.load_state_dict(checkpoint['net'])
            self.motion_model.to(device)  # send to device
            self.motion_model.eval()  # set eval mode on
        if self.use_warp:
            with open(os.path.join(dmr_path, warp_path, 'warp_matrix_{seq}.pickle'.format(seq=self.seq)),
                      'rb') as handle:
                self.warp_arr = np.array(pickle.load(handle))
            if frame_id >= self.warp_arr.shape[0]:
                self.warp_history = np.array([self.warp_arr[0].flatten()])
            else:
                self.warp_history = np.array([self.warp_arr[frame_id].flatten()])

    def update(self, match_mask, track_detection_bbox_map, track_detection_embedding_map, scores, frame_id, beta=0.5):
        self.time_since_update[match_mask] = 0
        self.hits[match_mask] += 1
        self.hit_streak[match_mask] += 1
        self.embeddings[match_mask] = beta * self.embeddings[match_mask] + (1 - beta) * track_detection_embedding_map[
            match_mask]
        self.status[match_mask] = 1
        track_det = np.expand_dims(convert_bbox_to_z(track_detection_bbox_map[:, :4]), 0)
        self.history[-1][match_mask] = track_det[0][match_mask]
        self.match_flag = np.concatenate([self.match_flag, match_mask.reshape(-1, 1)], 1)
        self.added_flag = np.concatenate([self.added_flag, np.ones((match_mask.shape[0], 1)).astype(bool)], 1)
        if self.use_warp:
            if frame_id >= self.warp_arr.shape[0]:
                self.warp_history = np.vstack([self.warp_history, self.warp_arr[0].flatten()])
            else:
                self.warp_history = np.vstack([self.warp_history, self.warp_arr[frame_id].flatten()])
        if self.history.shape[0] > N:
            self.history = self.history[1:N + 1, :, :]
            self.match_flag = self.match_flag[:, 1:N + 1]
            self.added_flag = self.added_flag[:, 1:N + 1]
            if self.use_warp:
                self.warp_history = self.warp_history[1:N + 1, :]
        self.scores[match_mask] = track_detection_bbox_map[match_mask, -1]
        self.frame_id[match_mask] = frame_id

    def predict(self, img=None):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.history.shape[0] == 1:
            if vel_only:
                tensor_z = torch.zeros(self.history.shape).to(device)
            else:
                tensor_z = torch.zeros(self.history.shape[0],self.history.shape[1],self.history.shape[2]*2).to(device)
            added_padding_mask = torch.Tensor(self.added_flag).cuda()
            if self.use_warp:
                warp_matrix = torch.from_numpy(self.warp_history).to(device)
                warp_matrix = warp_matrix.flatten().view(-1, 1, 6)
        else:

            bbox_diff = np.diff(self.history, 1, 0)
            bbox_diff = np.nan_to_num(bbox_diff)
            tensor_z = torch.Tensor(bbox_diff).to(device)
            added_padding_mask = torch.Tensor(self.added_flag[:, 1:]).cuda()
            if self.use_warp:
                warp_matrix = torch.from_numpy(self.warp_history[1:]).to(device)
                warp_matrix = warp_matrix.flatten().view(-1, 1, 6)
        with torch.no_grad():
            if self.use_warp:
                predicted_z = self.motion_model(tensor_z, input_warp=warp_matrix, added_padding_mask=added_padding_mask)
            else:
                predicted_z = self.motion_model(tensor_z, added_padding_mask=added_padding_mask)
        if vel_only:
            predicted_z = predicted_z.cpu().numpy() + self.history[-1]
            self.history = np.concatenate([self.history, predicted_z])
            predicted_bbox = convert_x_to_bbox(predicted_z.squeeze())
        else:
            predicted_z = predicted_z[:,:,:4].cpu().numpy()
            self.history = np.concatenate([self.history, predicted_z])
            predicted_bbox = convert_x_to_bbox(predicted_z.squeeze())
        self.age += 1
        self.status[self.time_since_update > max_age] = 3
        self.status[self.time_since_update <= max_age] = 2
        self.hit_streak[self.time_since_update > 0] = 0
        self.time_since_update += 1
        return predicted_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        boxes = convert_x_to_bbox(self.history[-1])
        boxes_wh = convert_corners_to_wh(boxes)
        tracks = np.concatenate([np.expand_dims(self.frame_id, 1), np.expand_dims(self.id, 1), boxes_wh], 1)
        area_flag = (tracks[:, 4] * tracks[:, 5]) > min_box_area
        vertical_flag = tracks[:, 4] / tracks[:, 5] < 1.6
        active_flag = self.status == 1
        valid_idx = area_flag & vertical_flag & active_flag
        valid_tracks = tracks[valid_idx]
        return valid_tracks

    def add_tracks(self, new_detections, new_detection_embeddings, frame_id):
        num_new_detections = new_detections.shape[0]
        self.count = self.count + num_new_detections
        buffer_size = self.history.shape[0]
        new_history = np.ones((buffer_size, num_new_detections, 4)) * np.nan
        new_match_flag = np.zeros((num_new_detections, buffer_size)).astype(np.bool)
        new_added_flag = np.zeros((num_new_detections, buffer_size)).astype(np.bool)
        new_history[-1:, :, :] = convert_bbox_to_z(new_detections)
        new_match_flag[:, -1] = True
        new_added_flag[:, -1] = True
        self.history = np.concatenate([self.history, new_history], 1)
        self.match_flag = np.concatenate([self.match_flag, new_match_flag], 0)
        self.added_flag = np.concatenate([self.added_flag, new_added_flag], 0)
        time_since_update_new = np.zeros(num_new_detections)
        frame_id_new = np.ones(num_new_detections) * frame_id
        hits_new = np.zeros(num_new_detections)
        hits_streak_new = np.zeros(num_new_detections)
        age_new = np.zeros(num_new_detections)
        status_new = np.zeros(num_new_detections)
        embeddings_new = new_detection_embeddings

        score_new = new_detections[:, -1]
        self.time_since_update = np.concatenate([self.time_since_update, time_since_update_new])
        self.id = np.arange(self.count)
        self.frame_id = np.concatenate([self.frame_id, frame_id_new])
        self.hits = np.concatenate([self.hits, hits_new])
        self.hit_streak = np.concatenate([self.hit_streak, hits_streak_new])
        self.age = np.concatenate([self.age, age_new])
        self.status = np.concatenate([self.status, status_new])
        self.scores = np.concatenate([self.scores, score_new])
        self.embeddings = np.concatenate([self.embeddings, embeddings_new])


def convert_bbox_to_z(bbox):
    """
      Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,w,h]
      """

    w = (bbox[:, 2] - bbox[:, 0])
    h = (bbox[:, 3] - bbox[:, 1])
    x = (bbox[:, 0]) + (w / 2)
    y = (bbox[:, 1]) + (h / 2)

    return np.array([x, y, w, h]).T


def convert_x_to_bbox(x):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    x, y, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.array([x1, y1, x2, y2]).T


def convert_corners_to_wh(boxes):
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes
