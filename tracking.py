import logging
import os
import shutil

import numpy as np
from loguru import logger
import pandas as pd
import torch
import sys
from pathlib import Path
from yolox.exp import get_exp
from yolox.utils import fuse_model
from src.utils.preprocess_yolo_input import preproc
import glob
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from src.tracker import det_dataset
from argparse import ArgumentParser

use_cuda = torch.cuda.is_available()
torch.set_printoptions(precision=3, sci_mode=False)
from yolox.utils import postprocess
from src.tracker.tn_da import *
from src.metrics.hota import get_hota_mota, get_hota_mota_all
# read config file
config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "tracking.config"))

# dataloader params
MOT_folder = config.get("DataLoader", "MOT_folder")
detection_folder = config.get("DataLoader", "detection_folder")
image_path = config.get("DataLoader", "image_path")
detector = config.get("DataLoader", "detector")
ini_file_path = config.get("DataLoader", "ini_file_path")
dataset = config.get("DataLoader", "dataset")

# detector params
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
warp_path = config.get("MotionModel", "warp_path")
N = config.get("MotionModel", "N")

# ReID
reid_model_path = config.get("ReID", "reid_model_path")
embedding_dim = config.getint("ReID", "embedding_dim")
project_dim = config.getint("ReID", "project_dim")
num_cpu = os.cpu_count()
t_img_h = config.getint("ReID","t_img_h")
t_img_w =  config.getint("ReID","t_img_w")
save_embedding = config.getboolean("ReID","save_embedding")
use_saved_embedding = config.getboolean("ReID","use_saved_embedding")
# TrackerInfo
tracking_filename = config.get("TrackerInfo", "tracking_filename")
experiment_name = config.get("TrackerInfo", "experiment_name")
save_imgs = config.getboolean("TrackerInfo", "save_imgs")
save_video = config.getboolean("TrackerInfo", "save_video")

# TrackerParams
max_age = config.getint("TrackerParams", "max_age")
min_hits = config.getint("TrackerParams", "min_hits")
beta = config.getfloat("TrackerParams", "beta")
iou_thres = config.getfloat("TrackerParams", "iou_thres")
iou_thres_2 = config.getfloat("TrackerParams", "iou_thres_2")
min_box_area = config.getfloat("TrackerParams", "min_box_area")
visual = config.getboolean("TrackerParams", "visual")


def load_detector_model(exp_file, detector_model_file, half, dmr_dir):
    # get exp
    exp_file = os.path.join("exps", exp_file)
    exp = get_exp(exp_file)
    # get model
    model = exp.get_model()
    if use_cuda:
        model.cuda(0)
    model.eval()
    # load model state dict
    logger.info("loading checkpoint ..")
    print(os.listdir(dmr_dir))
    detector_model_path = os.path.join(dmr_dir, "models", detector_model_file)
    if use_cuda:
        loc = "cuda:0"
    else:
        loc = "cpu"
    ckpt = torch.load(detector_model_path, map_location=loc)
    model.load_state_dict(ckpt["model"])
    logger.info("checkpoint loading done ..")
    # fuse model
    logger.info("fusing model ..")
    model = fuse_model(model)
    if half:
        model = model.half()
    return exp, model


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def get_transforms():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(t_img_h, t_img_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_image_list(seq, dmr_dir):
    frame_folder = os.path.join(dmr_dir, MOT_folder, seq, image_path)
    image_list = sorted(os.listdir(frame_folder))
    return frame_folder, image_list


def get_frame(frame_folder, img_name, test_size, tensor_type):
    image_file_name = os.path.join(frame_folder, img_name)
    imgs = cv2.imread(image_file_name)
    img_og = np.copy(imgs)
    img_h, img_w = imgs.shape[0], imgs.shape[1]
    # preprocess frame
    imgs = torch.from_numpy(preproc(imgs, test_size)).unsqueeze(0)
    imgs = imgs.type(tensor_type)
    return imgs, img_og, img_h, img_w


def post_process_detections(exp, outputs, img_h, img_w):
    outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
    detections = outputs[:, :4]
    scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))
    detections /= scale
    scores = outputs[:, 4] * outputs[:, 5]
    return detections, scores


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(120, 40), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def process_det(det, img_w, img_h):
    proc_det = det.copy()
    valid = True
    if det[0] >= img_w or det[1] >= img_h:
        valid = False
    if det[0] < 0:
        proc_det[0] = 0
    if det[1] < 0:
        proc_det[1] = 0
    if det[2] > img_w:
        proc_det[2] = img_w
    if det[3] > img_h:
        proc_det[3] = img_h

    return proc_det, valid


def make_results_folder(sequences, dmr_dir):
    results_folder = os.path.join(dmr_dir, "results", tracking_filename, experiment_name)
    eval_folder = os.path.join(results_folder, "evaluation")
    Path(eval_folder).mkdir(parents=True, exist_ok=True)
    seq_folder = os.path.join(results_folder, "sequences")
    Path(seq_folder).mkdir(parents=True, exist_ok=True)
    seq_folder = os.path.join(results_folder, "detections")
    Path(seq_folder).mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        s_folder = os.path.join(results_folder, "sequences", seq)
        d_folder = os.path.join(results_folder, "detections", seq)
        Path(s_folder).mkdir(parents=True, exist_ok=True)
        Path(d_folder).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(os.path.join(results_folder, "configs")):
        shutil.copytree(os.path.join("src", "configs"), os.path.join(results_folder, "configs"))


def make_video(seq, predicted_tracks_df, img_list, img_name_list, dmr_dir):
    results_folder = os.path.join(dmr_dir, "results", tracking_filename, experiment_name)
    logger.info("Saving video for {seq}".format(seq=seq))
    s_folder = os.path.join(results_folder, "sequences", seq)
    video_path = os.path.join(s_folder, "video.mp4")
    ini_path = os.path.join(dmr_dir, MOT_folder, seq, ini_file_path)
    config = configparser.ConfigParser()
    config.read(ini_path)
    h = config.getint("Sequence", "imHeight")
    w = config.getint("Sequence", "imWidth")
    frameRate = config.getint("Sequence", "frameRate")
    total_tracks = int(predicted_tracks_df[1].max())
    colors = torch.randint(0, 255, (total_tracks + 1, 3))
    fontScale = 1
    thickness = 3
    lineType = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    writer = cv2.VideoWriter(video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=1,
                             frameSize=(w, h))
    for j, img_n in enumerate(img_list):
        try:
            img = cv2.imread(img_n)
            img_id = int(img_name_list[j].split(".")[0])
            frame_track = predicted_tracks_df[predicted_tracks_df[0] == img_id]
            track_id = frame_track[1].values.astype(int)
            tracker_bbox = frame_track[[2, 3, 4, 5]].values.astype(int)
            for i, bbox in enumerate(tracker_bbox):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              color=tuple(colors[track_id[i]].tolist()), thickness=2)
                bottomLeftCornerOfText = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[2] / 2))
                fontColor = tuple(colors[track_id[i]].tolist())
                cv2.putText(img, str(track_id[i]), bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                            lineType)
        except:
            pass
        cv2.putText(img, str(j + 1), (100, 100), font, 2, (200, 200, 200), thickness, lineType)
        writer.write(img)
    writer.release()


def to_np(t):
    return t.permute(1, 2, 0).cpu().numpy()


class Filter(object):
    def __init__(self, img_og=None, img_w=None, img_h=None):
        self.img_og = img_og
        self.img_h = img_h
        self.img_w = img_w

    def filter_detection(self, data):
        '''
        det, d, img_w, img_h, img_og, scores, valid_detections,
                         valid_scores, detection_img_list
        '''
        det, score = data
        proc_det, valid = process_det(det, self.img_w, self.img_h)
        if valid:
            patch = self.img_og[proc_det[1]:proc_det[3], proc_det[0]:proc_det[2]]
        else:
            patch = None
        return patch, det, score


def get_sequences(tracking_path="", dataset=""):
    sequence_ids = os.listdir(tracking_path)
    sequence_list = []
    for seq_id in sequence_ids:
        if dataset == "MOT17":
            if "FRCNN" in seq_id:
                sequence_list.append(seq_id)
        elif dataset == "MOT20" or dataset == "DanceTrack":
            sequence_list.append(seq_id)

    return sorted(sequence_list)


def load_model(dmr_dir):
    if use_cuda:
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    else:
        tensor_type = torch.HalfTensor if half else torch.FloatTensor
    exp, model = load_detector_model(exp_file, detector_model_file, half, dmr_dir)

    return tensor_type, exp, model


def get_detections(frame_folder, img_name, tensor_type, exp, model):
    frame_real_id = int(img_name.split(".")[0])

    imgs, img_og, img_h, img_w = get_frame(frame_folder, img_name,
                                           exp.test_size, tensor_type)
    outputs = model(imgs)
    detections, scores = post_process_detections(exp, outputs, img_h, img_w)
    detections = detections.int()
    detections = detections[scores >= 0.1].cpu().numpy()
    scores = scores[scores >= 0.1].cpu().numpy()
    detections_for_save = np.zeros((detections.shape[0], detections.shape[1] + 6))
    detections_for_save[:, 2:6] = detections
    detections_for_save[:, 0] = frame_real_id
    detections_for_save[:, 6] = scores
    detections_for_save[:, 7] = -1
    detections_for_save[:, 8] = -1
    detections_for_save[:, 9] = -1
    detections_for_save[:, 1] = -1
    detections_for_save[:, 4] = detections_for_save[:, 4] - detections_for_save[:, 2]
    detections_for_save[:, 5] = detections_for_save[:, 5] - detections_for_save[:, 3]
    return detections, detections_for_save


def generate_tracks(dmr_dir):
    logging.disable(logging.WARNING)
    sequences = get_sequences(os.path.join(dmr_dir, MOT_folder), dataset)

    logger.info(f"Inference for {experiment_name}")

    logger.info("making results folder")
    make_results_folder(sequences, dmr_dir)
    results_folder = os.path.join(dmr_dir, "results", tracking_filename, experiment_name)
    track_folder_path = os.path.join(results_folder, f"{dataset}-train", "validation", "data")
    Path(track_folder_path).mkdir(exist_ok=True, parents=True)
    if det_mode == "online":
        tensor_type, exp, model = load_model(dmr_dir)
    else:
        tensor_type = None
        exp = None
        model = None
    logger.info("loading pretext model")
    if visual:
        pretext_model = load_pretext_model(os.path.join(dmr_dir, reid_model_path), embedding_dim, project_dim)
        if use_cuda:
            pretext_model = pretext_model.to(device)


    logger.info("iterating over frames")
    for seq in sorted(sequences, reverse=False):

        ini_path = os.path.join(dmr_dir, MOT_folder, seq, ini_file_path)
        c = configparser.ConfigParser()
        c.read(ini_path)
        img_h = c.getint("Sequence", "imHeight")
        img_w = c.getint("Sequence", "imWidth")
        if det_mode != "online":
            det_path = os.path.join(dmr_dir, detection_folder, seq, "det", "det.txt")
            detection_df = pd.read_csv(det_path, header=None)
            detection_df[4] = detection_df[2] + detection_df[4]
            detection_df[5] = detection_df[3] + detection_df[5]
        logger.info("Performing tracking for {seq}".format(seq=seq))
        frame_id = 0
        predicted_tracks = []
        detection_list = []
        detection_embedding_for_seq = []
        if use_saved_embedding:
            embedding_path = os.path.join(dmr_dir,"results","Re-Identification/loss_compare/save_embeddings_triplet/saved_embeddings",f"{seq}.pickle")
            with open(embedding_path, 'rb') as f:
                seq_embedding_list = pickle.load(f)
        frame_folder, img_name_list = get_image_list(seq, dmr_dir)
        for i, img_name in enumerate(tqdm(img_name_list)):
            if i==300:
                print("stop")
            frame_id += 1
            with torch.no_grad():
                # get output detections
                if det_mode == "online":
                    detections, detections_for_save = get_detections(frame_folder, img_name, tensor_type, exp, model)
                    detection_list.append(detections_for_save)
                    scores = detections_for_save[:, 6]
                else:
                    d_df = detection_df[detection_df[0] == (i + 1)]
                    detections = d_df[[2, 3, 4, 5]].values
                    scores = d_df[6].values

                # remove detection outside frame
                detections_with_score = np.zeros((detections.shape[0], detections.shape[1] + 1))
                detections_with_score[:, :4] = detections
                detections_with_score[:, 4] = scores
                detections = detections_with_score.copy()
                detections = detections[(detections[:, 0] <= img_w) & (detections[:, 1] <= img_h)]

                # clip detections outside frame within the frame
                detection_processed = detections.copy()
                detection_processed[detections[:, 0] < 0, 0] = 0
                detection_processed[detections[:, 1] < 0, 1] = 0
                detection_processed[detections[:, 2] > img_w, 2] = img_w
                detection_processed[detections[:, 3] > img_h, 3] = img_h
                # for MOT20 make clipped detections as og detections
                if dataset == "MOT20":
                    detections = detection_processed.copy()

                detection_embeddings = []
                if visual:
                    if not use_saved_embedding:
                        img_og = cv2.imread(os.path.join(frame_folder, img_name))
                        det_data = det_dataset.DetectionDataset(detections, img_og, t_img_h, t_img_w)
                        det_dataloader = DataLoader(det_data, batch_size=200, shuffle=False, num_workers=num_cpu)
                        for i, batch in enumerate(det_dataloader):
                            embedding = pretext_model(batch.cuda(), mode="eval")
                            detection_embeddings.append(embedding)
                        detection_embeddings = torch.cat(detection_embeddings).cpu().numpy()
                        if save_embedding:
                            detection_embedding_for_seq.append(detection_embeddings)
                    else:
                        detection_embeddings = seq_embedding_list[i]

                else:
                    detection_embeddings = np.ones((detections.shape[0], embedding_dim)) * np.nan

                # for first frame make all detections as new tracks
                if frame_id == 1:
                    # initialize new tracks
                    tracks = initialize_tracks(detections, scores, detection_embeddings, frame_id)

                    # get states of all initialized tracks
                    ret = tracks.get_state()

                if detections.shape[0] > 0 and frame_id > 1:
                    # associate detections to tracks
                    track_match_mask, detection_match_mask, track_detection_bbox_map, track_detection_embedding_map = \
                        associate_detections_to_tracks(detections, scores, tracks,
                                                       detection_embeddings)

                    # update matched tracks
                    tracks.update(track_match_mask, track_detection_bbox_map,
                                  track_detection_embedding_map, scores, frame_id)

                    # for unmatched detections, initialize new tracks
                    add_track_flag = (detections[:, -1] > track_init_thres) & (~detection_match_mask)
                    if detections[add_track_flag].shape[0] > 0:
                        tracks.add_tracks(detections[add_track_flag], detection_embeddings[add_track_flag], frame_id)
                    ret = tracks.get_state()

            if len(ret) > 0:
                predicted_tracks.append(ret)
        if det_mode == "online":
            all_detections = pd.DataFrame(np.concatenate(detection_list))
            det_dir_path = os.path.join(dmr_dir, detection_folder, seq, "det")
            Path(det_dir_path).mkdir(parents=True, exist_ok=True)
            det_path = os.path.join(det_dir_path, 'det.txt')
            all_detections.to_csv(det_path, index=False, header=False)
        s_folder = os.path.join(results_folder, "sequences", seq)
        track_path = os.path.join(s_folder, "tracks.txt")
        predicted_tracks_all = pd.DataFrame(np.concatenate(predicted_tracks))
        predicted_tracks_all[7] = -1
        predicted_tracks_all[8] = -1
        predicted_tracks_all.to_csv(track_path, header=False, index=False)

        if save_embedding:
            embedding_path = os.path.join(results_folder, "saved_embeddings", f'{seq}.pickle')
            Path(os.path.join(results_folder, "saved_embeddings")).mkdir(parents=True, exist_ok=True)
            with open(embedding_path, 'wb') as f:
                pickle.dump(detection_embedding_for_seq, f)
        t_path = os.path.join(track_folder_path, '{seq}.txt'.format(seq=seq))
        predicted_tracks_all.to_csv(t_path, header=False, index=False)

    eval_df = get_hota_mota_all(dmr_dir, results_folder)
    eval_df.to_csv(os.path.join(results_folder, "evaluation", "hota.csv"))



def track_for_model(dmr_dir, motion_model, reid_model, model_dir, num_seq):
    print("Performing tracking for length {N}".format(N=N))
    logging.disable(logging.WARNING)
    sequences = get_sequences(os.path.join(dmr_dir, MOT_folder), dataset)
    track_folder_path = os.path.join(dmr_dir, model_dir, "MOT20-val", "validation", "data")
    Path(track_folder_path).mkdir(exist_ok=True, parents=True)
    for seq in sorted(sequences, reverse=False)[:num_seq]:
        ini_path = os.path.join(dmr_dir, MOT_folder, seq, ini_file_path)
        c = configparser.ConfigParser()
        c.read(ini_path)
        img_h = c.getint("Sequence", "imHeight")
        img_w = c.getint("Sequence", "imWidth")
        if det_mode != "online":
            det_path = os.path.join(dmr_dir, detection_folder, seq, "det", "det.txt")
            detection_df = pd.read_csv(det_path, header=None)
            detection_df[4] = detection_df[2] + detection_df[4]
            detection_df[5] = detection_df[3] + detection_df[5]
        logger.info("Performing tracking for {seq}".format(seq=seq))
        frame_id = 0
        predicted_tracks = []
        detection_list = []
        frame_folder, img_name_list = get_image_list(seq, dmr_dir)
        for i, img_name in enumerate(tqdm(img_name_list)):
            frame_id += 1
            with torch.no_grad():
                # get output detections
                if det_mode == "online":
                    detections, detections_for_save = get_detections(frame_folder, img_name)
                    detection_list.append(detections_for_save)
                else:
                    d_df = detection_df[detection_df[0] == (i + 1)]
                    detections = d_df[[2, 3, 4, 5]].values
                    scores = d_df[6].values

                # remove detection outside frame
                detections_with_score = np.zeros((detections.shape[0], detections.shape[1] + 1))
                detections_with_score[:, :4] = detections
                detections_with_score[:, 4] = scores
                detections = detections_with_score.copy()
                detections = detections[(detections[:, 0] <= img_w) & (detections[:, 1] <= img_h)]

                # clip detections outside frame within the frame
                detection_processed = detections.copy()
                detection_processed[detections[:, 0] < 0, 0] = 0
                detection_processed[detections[:, 1] < 0, 1] = 0
                detection_processed[detections[:, 2] > img_w, 2] = img_w
                detection_processed[detections[:, 3] > img_h, 3] = img_h
                # for MOT20 make clipped detections as og detections
                if dataset == "MOT20":
                    detections = detection_processed.copy()

                detection_embeddings = []
                if visual:
                    img_og = cv2.imread(os.path.join(frame_folder, img_name))
                    det_data = det_dataset.DetectionDataset(detections, img_og, t_img_h,t_img_w)
                    det_dataloader = DataLoader(det_data, batch_size=200, shuffle=False, num_workers=os.cpu_count(),pin_memory=True)
                    for i, batch in enumerate(det_dataloader):
                        embedding = reid_model(batch.cuda(), mode="eval")
                        detection_embeddings.append(embedding)
                    detection_embeddings = torch.cat(detection_embeddings).cpu().numpy()
                else:
                    detection_embeddings = np.ones((detections.shape[0], embedding_dim)) * np.nan

                # for first frame make all detections as new tracks
                if frame_id == 1:
                    # initialize new tracks
                    tracks = initialize_tracks(detections, scores, detection_embeddings, frame_id, model=motion_model,seq=seq)

                    # get states of all initialized tracks
                    ret = tracks.get_state()

                if detections.shape[0] > 0 and frame_id > 1:
                    # associate detections to tracks
                    track_match_mask, detection_match_mask, track_detection_bbox_map, track_detection_embedding_map = \
                        associate_detections_to_tracks(detections, scores, tracks,
                                                       detection_embeddings)

                    # update matched tracks
                    tracks.update(track_match_mask, track_detection_bbox_map,
                                  track_detection_embedding_map, scores, frame_id)

                    # for unmatched detections, initialize new tracks
                    add_track_flag = (detections[:, -1] > track_init_thres) & (~detection_match_mask)
                    if detections[add_track_flag].shape[0] > 0:
                        tracks.add_tracks(detections[add_track_flag], detection_embeddings[add_track_flag], frame_id)
                    ret = tracks.get_state()

            if len(ret) > 0:
                predicted_tracks.append(ret)
        predicted_tracks_all = pd.DataFrame(np.concatenate(predicted_tracks))
        predicted_tracks_all[7] = -1
        predicted_tracks_all[8] = -1
        predicted_tracks_all.to_csv(os.path.join(track_folder_path, '{seq}.txt'.format(seq=seq)), header=False,
                                    index=False)

    hota = get_hota_mota(dmr_dir, model_dir)
    print("IDF1", hota['IDF1'])

    return hota


def save_videos(dmr_dir):
    sequences = get_sequences(os.path.join(dmr_dir, MOT_folder), dataset)
    for seq in sorted(sequences, reverse=False):
        results_folder = os.path.join(dmr_dir, "results", tracking_filename, experiment_name)
        s_folder = os.path.join(results_folder, "sequences", seq)
        track_path = os.path.join(s_folder, "tracks.txt")
        predicted_df = pd.read_csv(track_path, header=None)
        frame_folder, img_name_list = get_image_list(seq, dmr_dir)
        img_list = []
        for i, img_name in enumerate(img_name_list):
            image_file_name = os.path.join(frame_folder, img_name)
            # imgs = cv2.imread(image_file_name)
            img_list.append(image_file_name)
        make_video(seq, predicted_df, img_list, img_name_list, dmr_dir)


def main(track=True, video=True, dmt_dir=""):
    if track:
        generate_tracks(dmt_dir)
    if video:
        save_videos(dmt_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dmr_dir", help="path where model is stored")
    args = parser.parse_args()
    main(track=True, video=False, dmt_dir=args.dmr_dir)
