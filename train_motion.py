# import statements
import torch
from loguru import logger
import numpy as np
from tqdm import tqdm
import time
import os
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime
from argparse import ArgumentParser

# local imports
from src.dataloader.tracklet_loader import Tracklet, collate_fn_padd
from src.models.motion.tbmot import MotionModel
from src.utils.configuration import Configuration
from src.models.utils import save_checkpoint, load_checkpoint
from tracking import track_for_model

def train_eval(dmr_dir):
    # set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # load config files
    motion_config = Configuration.load_json(os.path.join("src", "configs", "motion.json"))
    # data loader config
    use_warp = motion_config.DataLoader.use_warp
    vel_only = motion_config.DataLoader.vel_only
    # model config
    arch = motion_config.Model.arch
    nhead = motion_config.Model.nhead
    feature_size = motion_config.Model.feature_size
    num_layers = motion_config.Model.num_layers
    bbox_dim = motion_config.Model.bbox_dim
    warp_dim = motion_config.Model.warp_dim
    N = motion_config.Model.N

    # Training Config
    batch_size = motion_config.Training.batch_size
    num_workers = os.cpu_count()
    epochs = motion_config.Training.epochs
    learning_rate = motion_config.Training.learning_rate
    init_epoch = motion_config.Training.init_epoch
    resume = motion_config.Training.resume
    cp_path = motion_config.Training.cp_path
    path_to_best = motion_config.Training.path_to_best
    val_frequency = motion_config.Training.val_frequency
    model_dir_name = motion_config.Training.model_dir_name
    experiment_name = motion_config.Training.experiment_name

    # experiment folder
    d = datetime.datetime.now()
    timestamp = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + "_" + str(d.minute) + "_" + str(
        d.second)
    model_dir = os.path.join("models", model_dir_name, experiment_name, timestamp)
    exp_folder = os.path.join(dmr_dir, model_dir)
    exp_folder_no_time = os.path.join(dmr_dir, "models", model_dir_name, experiment_name)
    if resume:
        cp_dir = cp_path.split("/")[0]
        exp_folder = os.path.join(dmr_dir, "models", model_dir_name, cp_dir)
    Path(exp_folder).mkdir(exist_ok=True, parents=True)
    # copy config file in the experiment folder
    if not os.path.exists(os.path.join(exp_folder, "configs")):
        shutil.copytree(os.path.join("src", "configs"), os.path.join(exp_folder, "configs"))


    # configure logger
    sink = os.path.join(exp_folder, "training_log.log")
    logger.add(sink=sink)
    logger.info("Config Loaded")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}.")

    # tracklet_dataset
    logger.info(f"Preparing training and validation dataset..")
    train_dataset = Tracklet(mode="train", N=N, dmr_path=dmr_dir)
    # tracklet dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=False, shuffle=False,collate_fn=collate_fn_padd)
    # instantiate network (which has been imported from *networks.py*)
    logger.info(f"Initializing model..")
    trans_bbox = MotionModel(arch=arch, feature_size=feature_size, num_layers=num_layers, nhead=nhead, bbox_dim=bbox_dim, warp_dim=warp_dim,use_warp=use_warp)

    # loss function
    criterion = torch.nn.SmoothL1Loss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        trans_bbox = trans_bbox.cuda()
        criterion = criterion.cuda()

    # optimizer
    logger.info(f"Initializing Optimizer..")
    # combine backbone and fully connected
    params = list(trans_bbox.parameters())
    optimizer = torch.optim.Adam(params=params, lr=learning_rate)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    if resume:
        logger.info(f"Loading saved checkpoint..")
        if use_cuda:
            map_location = "cuda:0"
        else:
            map_location = "cpu"
        # custom method for loading last checkpoint
        ckpt = load_checkpoint(exp_folder_no_time, cp_path, map_location=map_location,load_best=True)
        trans_bbox.load_state_dict(ckpt['net'])
        init_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])
        logger.info("Latest checkpoint loaded.")

    # tensorboard
    tensorboard_path = os.path.join(exp_folder, "tensorboard_logs")
    writer = SummaryWriter(tensorboard_path)

    # now we start the main loop
    n_iter = start_n_iter
    best_val_accuracy = 0
    total_i = 0
    is_best = False
    logger.info(f"Training begins ..")

    for epoch in range(init_epoch, epochs):
        # eval using initial weights
        # hota = track_for_model(dmr_dir, trans_bbox.eval(), model_dir)
        # writer.add_scalar('HOTA', hota['HOTA'], epoch)
        # writer.add_scalar('MOTA', hota['MOTA'], epoch)
        # writer.add_scalar('IDF1', hota['IDF1'], epoch)
        # if hota['HOTA'] > best_val_accuracy:
        #     best_val_accuracy = hota['HOTA']
        #     is_best = True

        # set models to train mode
        trans_bbox.train()
        ckpt_folder = os.path.join(exp_folder, "checkpoints")
        Path(ckpt_folder).mkdir(parents=True, exist_ok=True)
        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader))
        start_time = time.time()
        # for loop going through dataset
        training_loss = 0
        for i, batch in pbar:
            # data preparation
            if use_warp:
                input_bbox, warp_matrices,seq_lengths, mask, target_bbox = batch
                warp_matrices = warp_matrices.cuda()
                input_bbox = input_bbox.cuda()
                target_bbox = target_bbox.cuda()
                predicted_bbox = trans_bbox(input_bbox, warp_matrices,added_padding_mask=mask.all(2).T)
            else:
                input_bbox, seq_lengths, mask, target_bbox = batch
                input_bbox = input_bbox.cuda()
                target_bbox = target_bbox.cuda()
                mask = mask.cuda()
                predicted_bbox = trans_bbox(input_bbox ,added_padding_mask=mask.all(2).T)
            prepare_time = start_time - time.time()
            # forward and backward pass
            l1_loss = criterion(predicted_bbox, target_bbox)
            training_loss += l1_loss.item()
            mean_loss = training_loss / (i+1)
            if mean_loss < 90:
                for g in optimizer.param_groups:
                    g['lr'] = 0.01
            optimizer.zero_grad()
            l1_loss.backward()
            optimizer.step()
            # udpate tensorboardX
            total_i += 1
            writer.add_scalar('Training Loss',mean_loss , i + 1)
            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {mean_loss:.4f},  epoch: {epoch}/{epochs}')
            start_time = time.time()

        # start hota on MOT20
        cpkt = {
            'net': trans_bbox.state_dict(),
            'epoch': epoch,
            'n_iter': n_iter,
            'optim': optimizer.state_dict(),
            'val_stats': 0
        }
        save_path = os.path.join(ckpt_folder, 'model.ckpt')
        save_checkpoint(cpkt, save_path, is_best, best_ckpt_path=path_to_best)
        is_best = False

    print("training_done")



def main(args):
    train_eval(args.dmr_dir)





if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dmr_dir", help="path where model is stored")
    args = parser.parse_args()
    main(args)