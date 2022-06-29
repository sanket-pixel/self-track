from multiprocessing import freeze_support
import sys
import os
import argparse
from TrackEval import trackeval
import pandas as pd

freeze_support()


def get_hota_mota_all(dmr_dir, track_path):
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['TRACKERS_FOLDER'] = os.path.join(dmr_dir, track_path)
    default_dataset_config['SPLIT_TO_EVAL'] = 'train'

    default_dataset_config["SEQ_INFO"] = None
    default_dataset_config["BENCHMARK"] = 'MOT20'
    default_dataset_config["DO_PREPROC"] = False
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['NUM_PARALLEL_CORES'] = 16
    config['USE_PARALLEL'] = True
    config['PRINT_RESULTS'] = False
    config['PRINT_ONLY_COMBINED'] = False
    config['PRINT_CONFIG'] = False

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    trackers_to_eval = ['validation']
    tracker_eval = {}
    for tracker in trackers_to_eval:
        sequence_list = list(output_res['MotChallenge2DBox'][tracker].keys())
        for sequence in sequence_list:
            tracker_eval[sequence] = {}
            hota = output_res['MotChallenge2DBox'][tracker][sequence]['pedestrian']['HOTA']
            mota = output_res['MotChallenge2DBox'][tracker][sequence]['pedestrian']['CLEAR']
            identity = output_res['MotChallenge2DBox'][tracker][sequence]['pedestrian']['Identity']
            h = pd.DataFrame(hota)[['HOTA', 'DetA', 'AssA', 'LocA', 'RHOTA']].mean().to_dict()
            m = pd.Series(mota)[['MOTA', 'MOTP', 'CLR_FP', 'CLR_FN']].to_dict()
            i = pd.Series(identity)[['IDF1', 'IDP', 'IDR']].to_dict()
            h.update(m)
            h.update(i)
            tracker_eval[sequence] = h

    eval_df = pd.DataFrame(tracker_eval).T.round(3)

    eval_df = eval_df[['HOTA', 'DetA', 'AssA','MOTA', 'MOTP', 'CLR_FP', 'CLR_FN', 'IDF1', 'IDR','IDP']]

    return eval_df


def get_hota_mota(dmr_dir, track_path):
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['TRACKERS_FOLDER'] = os.path.join(dmr_dir, track_path)
    default_dataset_config['SPLIT_TO_EVAL'] = 'val'

    default_dataset_config["SEQ_INFO"] = None
    default_dataset_config["BENCHMARK"] = 'MOT20'
    default_dataset_config["DO_PREPROC"] = False
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['NUM_PARALLEL_CORES'] = 16
    config['USE_PARALLEL'] = True
    config['PRINT_RESULTS'] = False
    config['PRINT_ONLY_COMBINED'] = False
    config['PRINT_CONFIG'] = False

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    trackers_to_eval = ['validation']
    tracker_eval = {}
    for tracker in trackers_to_eval:
        hota = output_res['MotChallenge2DBox'][tracker]['COMBINED_SEQ']['pedestrian']['HOTA']
        mota = output_res['MotChallenge2DBox'][tracker]['COMBINED_SEQ']['pedestrian']['CLEAR']
        identity = output_res['MotChallenge2DBox'][tracker]['COMBINED_SEQ']['pedestrian']['Identity']
        h = pd.DataFrame(hota)[['HOTA', 'DetA', 'AssA', 'LocA', 'RHOTA']].mean().to_dict()
        m = pd.Series(mota)[['MOTA', 'MOTP', 'CLR_FP', 'CLR_FN']].to_dict()
        i = pd.Series(identity)[['IDF1', 'IDP', 'IDR']].to_dict()
        h.update(m)
        h.update(i)
        tracker_eval[tracker] = h

    eval_df = pd.DataFrame(tracker_eval).T.round(3)
    eval_hota = eval_df[['HOTA', 'DetA', 'AssA', 'LocA', 'RHOTA']]
    eval_mota = eval_df[['MOTA', 'MOTP', 'CLR_FP', 'CLR_FP', 'IDF1', 'IDR']]

    mota = eval_mota.to_dict('records')[0]
    hota = eval_hota.to_dict('records')[0]

    hota.update(mota)
    return hota
