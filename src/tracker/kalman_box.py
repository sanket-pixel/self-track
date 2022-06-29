import cv2
import configparser
import os
from .kalman import VectorizedKalmanFilter
import numpy as np


min_box_area = 100
np.seterr(divide='ignore', invalid='ignore')
class KalmanTracker(object):

    def __init__(self, boxes, embeddings, scores, frame_id):

        self.count = boxes.shape[0]
        self.kf = VectorizedKalmanFilter(dim_x=7, dim_z=4, num_states=self.count)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[:, 4:, 4:] *= 1000.
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:, :4] = convert_bbox_to_z(boxes)

        self.time_since_update = np.zeros(self.count)
        self.id = np.arange(self.count)
        self.frame_id = np.array([frame_id] * self.count)
        self.hits = np.zeros(self.count)
        self.hit_streak = np.zeros(self.count)
        self.age = np.zeros(self.count)
        if frame_id == 1:
            self.status = np.ones(self.count)
        else:
            self.status = np.zeros(self.count)
        self.embeddings = embeddings
        self.scores = scores

    def update(self, match_mask, track_detection_bbox_map, track_detection_embedding_map=None, score=-1, frame_id=-1, beta=0.5):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update[match_mask] = 0
        self.hits[match_mask] += 1
        self.hit_streak[match_mask] += 1
        self.embeddings[match_mask] = beta * self.embeddings[match_mask] + (1 - beta) * track_detection_embedding_map[
            match_mask]
        self.status[match_mask] = 1
        x = self.kf.x.copy()
        P = self.kf.P.copy()
        self.kf.update(convert_bbox_to_z(track_detection_bbox_map))
        self.kf.P[~match_mask] = P[~match_mask]
        self.kf.x[~match_mask] = x[~match_mask]
        self.scores[match_mask] = track_detection_bbox_map[match_mask,-1]
        self.frame_id[match_mask] = frame_id

    def predict(self, img=None, max_age=20):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if (self.kf.x[:, 6] + self.kf.x[:, 2]) <= 0:
        #     self.kf.x[:, 6] *= 0.0
        self.kf.x[(self.kf.x[:, 6] + self.kf.x[:, 2]) <= 0, 6] = 0.0
        self.kf.predict()
        self.age += 1
        self.status[self.time_since_update > max_age] = 3
        self.status[self.time_since_update <= max_age] = 2
        self.hit_streak[self.time_since_update > 0] = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x, self.scores)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        boxes = convert_x_to_bbox(self.kf.x, self.scores)
        boxes_wh = convert_corners_to_wh(boxes)
        tracks = np.concatenate([np.expand_dims(self.frame_id,1), np.expand_dims(self.id, 1), boxes_wh], 1)
        area_flag = (tracks[:, 4] * tracks[:, 5]) > min_box_area
        vertical_flag = tracks[:, 4] / tracks[:, 5] < 1.6
        active_flag = self.status==1
        valid_idx = area_flag & vertical_flag & active_flag
        valid_tracks = tracks[valid_idx]
        return valid_tracks

    def add_tracks(self, new_detections, new_detection_embeddings, frame_id):
        num_new_detections = new_detections.shape[0]
        dim_x = self.kf.x.shape[1]

        self.count = self.count+ num_new_detections

        P_new = np.expand_dims(np.eye(dim_x), 0).repeat(num_new_detections, 0)
        P_new[:, 4:, 4:] *= 1000.
        P_new *= 10
        self.kf.P = np.concatenate([self.kf.P, P_new])

        x_new = np.zeros((num_new_detections, dim_x))
        x_new[:, :4] = convert_bbox_to_z(new_detections)
        self.kf.x = np.concatenate([self.kf.x, x_new])

        time_since_update_new = np.zeros(num_new_detections)
        frame_id_new = np.ones(num_new_detections) * frame_id
        hits_new = np.zeros(num_new_detections)
        hits_streak_new = np.zeros(num_new_detections)
        age_new = np.zeros(num_new_detections)
        status_new = np.zeros(num_new_detections)
        embeddings_new = new_detection_embeddings

        score_new = new_detections[:,-1]
        self.time_since_update = np.concatenate([self.time_since_update, time_since_update_new])
        self.id = np.arange(self.count)
        self.frame_id = np.concatenate([self.frame_id, frame_id_new])
        self.hits = np.concatenate([self.hits, hits_new])
        self.hit_streak = np.concatenate([self.hit_streak, hits_streak_new])
        self.age = np.concatenate([self.age, age_new])
        self.status = np.concatenate([self.status, status_new])
        self.scores = np.concatenate([self.scores, score_new])
        self.embeddings = np.concatenate([self.embeddings,embeddings_new])




def convert_bbox_to_z(boxes):
    """
      Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
      """
    boxes = np.expand_dims(boxes,2)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x = boxes[:, 0] + w / 2.
    y = boxes[:, 1] + h / 2.
    s = w * h  # scale is just area
    #noinspection
    r = w / h
    r[r != r] = 0
    return np.concatenate([x, y, s, r], 1)


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    x = np.expand_dims(x,2)
    score = np.expand_dims(score,1)
    w = np.sqrt(x[:,2] * x[:,3])
    h = x[:,2] / w
    return np.concatenate([x[:,0] - w / 2., x[:,1] - h / 2., x[:,0] + w / 2., x[:,1] + h / 2., score],1)


def convert_wh_to_corners(boxes):
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes

def convert_corners_to_wh(boxes):
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes

