import logging
from loguru import logger
import pandas as pd
import torch
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from yolox.exp import get_exp
from yolox.utils import fuse_model
from preprocess_yolo_input import preproc
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

import time
from yolox.utils import postprocess
from src.tracker.byte_track.bt_da import *

# read config file
config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "tracking.config"))
# dataloader params
MOT_folder = config.get("DataLoader", "MOT_folder")
image_path = config.get("DataLoader", "image_path")
detector = config.get("DataLoader", "detector")
ini_file_path = config.get("DataLoader", "ini_file_path")

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

# TrackerInfo
pretext_model_path = config.get("TrackerInfo", "pretext_model_path")
embedding_size = config.getint("TrackerInfo", "embedding_size")
tracking_filename = config.get("TrackerInfo", "tracking_filename")
experiment_name = config.get("TrackerInfo", "experiment_name")
start=config.getint("TrackerInfo", "start")
end = config.getint("TrackerInfo", "end")
# TrackerParams
max_age = config.getint("TrackerParams", "max_age")
min_hits = config.getint("TrackerParams", "min_hits")
beta = config.getfloat("TrackerParams", "beta")
iou_thres = config.getfloat("TrackerParams", "iou_thres")
iou_thres_2 = config.getfloat("TrackerParams", "iou_thres_2")
min_box_area = config.getfloat("TrackerParams", "min_box_area")
visual = config.getboolean("TrackerParams", "visual")
save_video = config.getboolean("TrackerInfo", "save_video")
online_det = config.getboolean("TrackerInfo", "online_det")


def load__detector_model(exp_file, detector_model_file, half):
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
    detector_model_path = os.path.join("models", detector_model_file)
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


def get_image_list(seq_path):
    img_name_list = sorted(os.listdir(seq_path))
    return seq_path, img_name_list


def to_np(t):
    return t.permute(1, 2, 0).cpu().numpy()


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
    detections, scores = [], []
    if not outputs is None:
        detections = outputs[:, :4]
        scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))
        detections /= scale
        scores = outputs[:, 4] * outputs[:, 5]
        scores = scores.cpu().numpy()

    return detections, scores


def make_results_folder(sequence_path):
    sequence_path = sequence_path.replace("/", '_')
    results_folder = os.path.join("data", "tracklet_store", sequence_path)
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    return results_folder


def make_video_and_patches(seq, predicted_tracks_df, img_list, img_name_list):
    results_folder = os.path.join("data", "tracklet_store", seq)
    logger.info("Saving video for {seq}".format(seq=seq))
    video_path = os.path.join(results_folder, "video.mp4")
    h = img_list[0].shape[0]
    w = img_list[0].shape[1]
    frameRate = 12
    total_tracks = int(predicted_tracks_df[1].max())
    frame_ids = predicted_tracks_df[0].unique().astype(int)
    track_patch_dict = {}
    for t in predicted_tracks_df[1].unique().astype(int):
        track_patch_dict[t] = {"patches": [], "bboxes": [], "ts": [], "conf": []}
    for i, fid in enumerate(frame_ids):
        frame_tracks = predicted_tracks_df[predicted_tracks_df[0] == fid]
        frame = img_list[int(frame_ids[int(i)] - 1)]
        for track in frame_tracks.iterrows():
            track_id = int(track[1][1])
            ts = int(track[1][0])
            bbox = list(track[1][2:6].astype(int)) + list([ts])
            if bbox[0] < 0:
                bbox[0] = 0
            elif bbox[1] < 0:
                bbox[1] = 0
            elif bbox[0] + bbox[2] > w:
                bbox[2] = w - bbox[0]
            elif bbox[1] + bbox[3] > h:
                bbox[3] = h - bbox[1]

            patch = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            bbox = list(track[1])
            track_patch_dict[track_id]["patches"].append(patch)
            track_patch_dict[track_id]['bboxes'].append(bbox)
            track_patch_dict[track_id]['ts'].append(ts)

    for k, v in track_patch_dict.items():
        try:
            track_folder = os.path.join(results_folder, str(k))
            patch_folder = os.path.join(track_folder, "patches")
            Path(patch_folder).mkdir(parents=True, exist_ok=True)
            bbox_df = pd.DataFrame(np.array(v['bboxes']), columns=["t", "id", "x", "y", "w", "h", "c"])
            bbox_df.to_csv(os.path.join(track_folder, "bboxes.csv"), index=False)
            for i, p in enumerate(v["patches"]):
                cv2.imwrite(os.path.join(patch_folder, "{ts}.png".format(ts=v["ts"][i])), p)
        except:
            pass

    try:
        colors = torch.randint(0, 255, (total_tracks + 1, 3))
        fontScale = 1
        thickness = 3
        lineType = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        writer = cv2.VideoWriter(video_path, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=12,
                                 frameSize=(w, h))
        for j, img in enumerate(img_list):
            try:
                # img_id = int(img_name_list[j].split(".")[0])
                frame_track = predicted_tracks_df[predicted_tracks_df[0] == j+1]
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

    except:
        pass


def generate_tracks(sequence_path):
    logging.disable(logging.WARNING)
    if use_cuda:
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    else:
        tensor_type = torch.HalfTensor if half else torch.FloatTensor
    logger.info("loading model")
    # if online_det:
    exp, model = load__detector_model(exp_file, detector_model_file, half)
    inference_time = 0
    track_time = 0
    # iterate over frames
    logger.info("iterating over frames")
    logger.info("Performing tracking for {seq}".format(seq=sequence_path))
    frame_id = 0
    KalmanBoxTracker.count = 0
    predicted_tracks = []
    img_list = []
    detection_list = []
    ret = []
    tracks = []
    frame_folder, img_name_list = get_image_list(sequence_path)
    for i, img_name in enumerate(tqdm(img_name_list)):
        frame_id += 1
        detection_img_list = []
        detection_embeddings = []
        with torch.no_grad():
            imgs, img_og, img_h, img_w = get_frame(frame_folder, img_name,
                                                   exp.test_size, tensor_type)
            img_list.append(img_og)
            start = time.time()
            # get output detections
            outputs = model(imgs)
            detections, scores = post_process_detections(exp, outputs, img_h, img_w)
            if len(detections)>0:
                detections = detections.int().cpu().numpy()
                d = pd.DataFrame(detections)
                d[4] = frame_id
                d[5] = scores
                detection_list.append(d)

                for det in detections:
                    # add embedding calc here
                    detection_img_list.append(img_og[det[1]:det[3], det[0]:det[2]])
                    if visual:
                        detection_embedding = None
                    else:
                        detection_embedding = None
                    detection_embeddings.append(detection_embedding)
                # for first frame make all detections as new tracks
                if frame_id == 1:
                    # initialize new tracks
                    tracks = initialize_tracks(detections, scores, detection_img_list, detection_embeddings, frame_id)
                    # get states of all initialized tracks
                    ret, tracks = get_track_states(tracks, frame_id)

                if detections.shape[0] > 0 and frame_id > 1:
                    # associate detections to tracks
                    matched_tracks, matched_detections, \
                    unmatched_tracks, unmatched_detections = \
                        associate_detections_to_tracks(detections, scores, tracks, detection_embeddings)
                    # update matched tracks
                    tracks = update_matched_tracks(tracks, detections, matched_tracks, matched_detections, img_og,
                                                   detection_embeddings, scores)
                    # delete dead tracks
                    tracks = delete_dead_tracks(tracks)
                    # for unmatched detections, initialize new tracks
                    for idx in unmatched_detections:
                        score = scores[idx]
                        if score > track_init_thres:
                            bbox_img = get_bbox_image(detections, img_og, idx)
                            new_track = KalmanBoxTracker(detections[idx], bbox_img, detection_embeddings[idx], score,
                                                         frame_id)
                            tracks.append(new_track)
                    ret, tracks = get_track_states(tracks, frame_id)
        if len(ret) > 0:
            predicted_tracks.append(np.concatenate(ret))

    if len(predicted_tracks)>0:
        results_folder = make_results_folder(sequence_path)
        track_path = os.path.join(results_folder, "tracks.txt")
        predicted_tracks_all = np.concatenate(predicted_tracks)
        predicted_tracks_df = pd.DataFrame(predicted_tracks_all)
        predicted_tracks_df.to_csv(track_path, header=False, index=False)
        sequence_path = sequence_path.replace("/","_")
        make_video_and_patches(sequence_path, predicted_tracks_df, img_list, img_name_list)


def main():
    seq_file_path = os.path.join("datastore", "TAO", "train", "seq_file_path.txt")
    print("Start : ", start)
    print("End : ", end)
    seq_list = pd.read_csv(seq_file_path).values.flatten()[start:end]
    for file in seq_list:
        if file.replace("/","_") not in os.listdir(os.path.join("data", "tracklet_store")):
            logger.info("Generating tracks for {file}".format(file=file))
            generate_tracks(file)


if __name__ == '__main__':
    main()
