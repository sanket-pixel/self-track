import numpy as np
import configparser
import os
import motmetrics as mm
# from numba import jit
from torch.nn.functional import cosine_similarity, normalize
from scipy.optimize import linear_sum_assignment as linear_assignment
from .tn_utils import *
from .kalman import VectorizedKalmanFilter
from .kalman_box import KalmanTracker
from .transformer_bbox import NNTracker
from torchvision.ops import box_iou

# read config file
config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "tracking.config"))

# Detector
det_mode = config.get("Detector", "det_mode")
nms_thres = config.getfloat("Detector", "nms_thres")
nms_flag = config.getboolean("Detector", "nms_flag")
high_thres = config.getfloat("Detector", "high_thres")
low_thres = config.getfloat("Detector", "low_thres")
track_init_thres = config.getfloat("Detector", "track_init_thres")
exp_file = config.get("Detector", "exp_file")
detector_model_file = config.get("Detector", "chkpt_file")
half = config.getboolean("Detector", "half")

# MotionModel
motion_cp_path = config.get("MotionModel", "motion_cp_path")
motion_model_name = config.get("MotionModel", "motion_model")
# file, obj = motion_model_name.split('.')
# motion_model = getattr(globals()[file], obj)

# ReID
reid_model_path = config.get("ReID", "reid_model_path")
embedding_dim = config.getint("ReID", "embedding_dim")
n = config.getboolean("ReID", "normalize_embeddings")

# TrackerInfo
tracking_filename = config.get("TrackerInfo", "tracking_filename")
experiment_name = config.get("TrackerInfo", "experiment_name")
# TrackerParams
max_age = config.getint("TrackerParams", "max_age")
min_hits = config.getint("TrackerParams", "min_hits")
beta = config.getfloat("TrackerParams", "beta")
iou_thres = config.getfloat("TrackerParams", "iou_thres")
min_box_area = config.getfloat("TrackerParams", "min_box_area")
iou_thres_2 = config.getfloat("TrackerParams", "iou_thres_2")
embedding_distance_type = config.get("TrackerParams", "embedding_distance_type")
bbox_distance_type = config.get("TrackerParams", "bbox_distance_type")
visual = config.getboolean("TrackerParams", "visual")


# @jit
def iou(bb_test, bb_gt):
    """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def initialize_tracks(detections, scores, detection_embeddings, frame_id, model=None,seq=None):
    init_flag = detections[:, -1] > track_init_thres
    detections = detections[init_flag]
    scores = scores[init_flag]
    detection_embeddings = detection_embeddings[init_flag]
    if motion_model_name == "kalman":
        vec_tracks = KalmanTracker(detections,
                                   detection_embeddings, scores, frame_id)
    else:
        vec_tracks = NNTracker(detections, detection_embeddings, scores, frame_id,model=model,seq=seq)
    return vec_tracks


def vec_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def embedding_distance_matrix(track_embeddings, detection_embedding):
    track_embeddings = torch.from_numpy(track_embeddings)
    detection_embedding = torch.from_numpy(detection_embedding)
    D_cos = 1 - cosine_similarity(track_embeddings.unsqueeze(1),
                                  detection_embedding, -1)
    D_cos = D_cos.numpy()
    # D_cos = D_cos/D_cos. max(1).reshape(-1,1)
    return D_cos


def embedding_cosine_similarity_matrix(track_embeddings, detection_embeddings, n=True):
    if n:
        track_embeddings = normalize(torch.from_numpy(track_embeddings))
        detection_embeddings = normalize(torch.from_numpy(detection_embeddings))
    else:
        track_embeddings = torch.from_numpy(track_embeddings)
        detection_embeddings = torch.from_numpy(detection_embeddings)
    d_cos = cosine_similarity(track_embeddings.unsqueeze(1),
                              detection_embeddings, -1)
    return d_cos


def embedding_lp_similarity_matrix(track_embeddings, detection_embeddings, n=True):
    if n:
        track_embeddings = normalize(torch.from_numpy(track_embeddings))
        detection_embeddings = normalize(torch.from_numpy(detection_embeddings))
    else:
        track_embeddings = torch.from_numpy(track_embeddings)
        detection_embeddings = torch.from_numpy(detection_embeddings)
    # return (1 - torch.cdist(track_embeddings, detection_embeddings))
    return 1 - torch.cdist(track_embeddings, detection_embeddings)


def visibility_matrix(track_bbox, detection_bbox):
    d_bbox = np.zeros((len(track_bbox), len(detection_bbox)), dtype=np.float32)
    for t, trk in enumerate(track_bbox):
        for d, det in enumerate(detection_bbox):
            d_bbox[t, d] = 1 - iou(trk, det)
    return d_bbox


# TODO parallelize this
def iou_matrix(track_bbox, detection_bbox):
    d_bbox = np.zeros((len(track_bbox), len(detection_bbox)), dtype=np.float32)
    for t, trk in enumerate(track_bbox):
        for d, det in enumerate(detection_bbox):
            d_bbox[t, d] = iou(trk, det)
    return d_bbox


def bbox_distance_matrix(track_bbox, detection_bbox):
    bbox_difference = np.expand_dims(track_bbox, 1) - detection_bbox
    track_hw = convert_corners_to_wh(track_bbox)[:, 2:]
    perimeter_track = 2 * track_hw.sum(1)
    d_bbox = np.linalg.norm(bbox_difference, axis=2) / (2 * np.expand_dims(perimeter_track, 1))
    return d_bbox


def associate_detections_to_tracks(detections, scores, tracks, detection_embeddings):
    track_match_mask = np.zeros((tracks.count)).astype(bool)
    detection_match_mask = np.zeros((detections.shape[0])).astype(bool)
    track_detection_map = np.ones((tracks.count, 5)) * np.nan
    track_detection_embedding_map = np.zeros((tracks.count, embedding_dim))
    """ Step 1 : Split detections into low and high score groups """
    D_high_idx = scores > high_thres
    D_low_idx = (scores >= low_thres) & (scores < high_thres)
    """ Step 2: Predict new locations of Tracks """
    predicted_tracks = tracks.predict(max_age=max_age)
    """ Step 3: Get cost matrix for (tracks, detections) using IOU and visual embeddings """
    D_iou = vec_iou(predicted_tracks[:, :4], detections[:, :-1])
    # D_iou = box_iou(torch.Tensor(predicted_tracks[:, :-1]), torch.Tensor(detections[:, :-1])).numpy()
    # get cosine similarity between track and detections embeddings
    if visual:
        D_cos = globals()[embedding_distance_type](tracks.embeddings, detection_embeddings).numpy()
        # D_total = D_cos
        # D_total = D_cos * D_iou
        D_total = (D_cos + D_iou)/2
    else:
        D_total = D_iou
    # make out dead detections
    D_total[tracks.status == 3, :] = 0
    D_total = D_total
    ''' Step 4: Perform first match between D_high and predicted tracks '''
    # mask out low conf detections
    D_total_high = D_total.copy()
    D_total_high[:, ~D_high_idx] = 0
    # perform first match on high confidence detections and tracks
    first_matches = linear_assignment(-D_total_high)
    first_match_costs = D_total_high[first_matches]
    # find accept matches
    first_match_accept_mask = first_match_costs >= iou_thres
    track_match_mask[first_matches[0][first_match_accept_mask]] = True
    detection_match_mask[first_matches[1][first_match_accept_mask]] = True
    track_detection_map[first_matches[0][first_match_accept_mask]] = detections[
        first_matches[1][first_match_accept_mask]]
    track_detection_embedding_map[first_matches[0][first_match_accept_mask]] = detection_embeddings[
        first_matches[1][first_match_accept_mask]]
    ''' Step 5: Perform second match '''
    D_iou_low = D_iou.copy()
    # mask out high confidence detections
    D_iou_low[:, ~D_low_idx] = 0
    # mask out already matched tracks
    D_iou_low[track_match_mask] = 0
    # perform matching between low confidence detections and remaining tracks
    second_matches = linear_assignment(-D_iou_low)
    second_match_costs = D_iou_low[second_matches]
    # find accept second matches
    second_match_accept_mask = second_match_costs >= iou_thres_2
    track_match_mask[second_matches[0][second_match_accept_mask]] = True
    detection_match_mask[second_matches[1][second_match_accept_mask]] = True
    track_detection_map[second_matches[0][second_match_accept_mask]] = detections[
        second_matches[1][second_match_accept_mask]]
    track_detection_embedding_map[second_matches[0][second_match_accept_mask]] = detection_embeddings[
        second_matches[1][second_match_accept_mask]]
    return track_match_mask, detection_match_mask, track_detection_map, track_detection_embedding_map
