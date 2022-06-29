"""
This file contains the training logic for
tracknet.
"""
# import statements
import time
from pathlib import Path
import torch
import numpy as np
import os
import shutil
from loguru import logger
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from pytorch_metric_learning import distances, losses, miners, reducers, samplers
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter
from src.dataloader.patch_loader import TrackInstance
from src.models.reid.metric_learning import ReIDModel
from src.utils.configuration import Configuration
from src.models.utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from matplotlib import cm
from matplotlib import pyplot as plt
import datetime
from argparse import ArgumentParser
from tracking import track_for_model

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def get_knn_accuracy(embedding_list, label_list):
    all_embeddings = torch.cat(embedding_list).cpu()
    all_track_ids = torch.cat(label_list).cpu()
    dist = torch.cdist(all_embeddings, all_embeddings)
    knn = dist.topk(200, largest=False)
    knn_label = all_track_ids[knn.indices]
    track_predictions = knn_label.mode(dim=1)[0]
    accuracy = accuracy_score(all_track_ids, track_predictions)
    return accuracy


def plot_embeddings(embedding_list, label_list):
    all_embeddings = torch.cat(embedding_list).cpu()
    all_track_ids = torch.cat(label_list).cpu()
    vis_categories = all_track_ids.unique()[:20]
    cat_indices = []
    for cat in vis_categories:
        cat_indices.append(torch.where(all_track_ids == cat)[0])
    cat_indices = torch.concat(cat_indices)
    vis_embeddings = all_embeddings[cat_indices]
    vis_label = all_track_ids[cat_indices]
    tsne = TSNE(2, verbose=1)
    embedding2d = tsne.fit_transform(vis_embeddings)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = cm.get_cmap('tab20')
    for i, lab in enumerate(vis_categories):
        indices = vis_label == lab
        ax.scatter(embedding2d[indices, 0], embedding2d[indices, 1], c=np.array(cmap(i)).reshape(1, 4),
                   label=lab,
                   alpha=0.5)
    return fig


def train_eval(dmr_dir):
    # load config files
    reid_config = Configuration.load_json(os.path.join("src", "configs", "reid.json"))

    # data loader config
    training_data_path = reid_config.DataLoader.training_data_path
    det_conf_filter = reid_config.DataLoader.det_conf_filter
    # metric learning config
    distance = reid_config.MetricLearning.distance
    loss = reid_config.MetricLearning.loss
    miner = reid_config.MetricLearning.miner
    reducer = reid_config.MetricLearning.reducer
    sampler = reid_config.MetricLearning.sampler

    # training config
    resume = reid_config.Training.resume
    batch_size = reid_config.Training.batch_size
    num_workers = os.cpu_count()
    val_frequency = reid_config.Training.val_frequency
    epochs = reid_config.Training.epochs
    learning_rate = reid_config.Training.learning_rate
    embedding_dim = reid_config.Training.embedding_dim
    projection_dim = reid_config.Training.projection_dim
    t_img_h = reid_config.Training.t_img_h
    t_img_w = reid_config.Training.t_img_w
    model_dir_name = reid_config.Training.model_dir_name
    experiment_name = reid_config.Training.experiment_name
    init_epoch = reid_config.Training.init_epoch
    cp_path = reid_config.Training.cp_path
    path_to_best = reid_config.Training.path_to_best

    # experiment folder
    d = datetime.datetime.now()
    timestamp = d.strftime("%m_%d_%Y_%H_%M")
    exp_folder = os.path.join(dmr_dir, "models", model_dir_name, experiment_name, timestamp)
    model_dir = os.path.join("models", model_dir_name, experiment_name, timestamp)
    exp_folder_no_time = os.path.join(dmr_dir, "models", model_dir_name, experiment_name)
    if resume:
        exp_folder = os.path.join(dmr_dir, "models", model_dir_name, experiment_name, cp_path)
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

    logger.info(f"Training with  {experiment_name}")

    # reID dataset
    logger.info(f"Preparing training and validation dataset..")
    train_dataset = TrackInstance(mode="train", transform_mode="auto_augment", dmr_path=dmr_dir)
    val_dataset = TrackInstance(mode="val", transform_mode="no_augment", dmr_path=dmr_dir)

    # reID dataloader
    if sampler['name'] == "MPerClassSampler":
        data_df = pd.read_csv(os.path.join(dmr_dir, training_data_path))
        # confidence values filter
        data_df = data_df[data_df['c'] > det_conf_filter]
        labels = data_df['id'].values
        sampler['params']['labels'] = labels
        sampler['params']['length_before_new_iter'] = len(train_dataset)
        sampler['batch_size'] = batch_size

    train_sampler = getattr(samplers, sampler['name'])(**sampler['params'])

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=True, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True, shuffle=False)

    # initialize model
    logger.info(f"Initializing model..")
    reid_model = ReIDModel(embedding_dim=embedding_dim, projection_dim=projection_dim)
    reid_model = torch.nn.DataParallel(reid_model)

    # loss
    logger.info(f"Initializing Loss..")
    if loss['name'] == "CrossEntropyLoss":
        criterion = getattr(torch.nn, loss['name'])(**loss['params'])
    else:
        distance = getattr(distances, distance['name'])(**distance['params'])
        reducer = getattr(reducers, reducer['name'])(**reducer['params'])
        if miner['name'] == "BatchEasyHardMiner":
            miner['params']['pos_strategy'] = miners.BatchHardMiner.EASY
            miner['params']['neg_strategy'] = miners.BatchHardMiner.SEMIHARD
        miner = getattr(miners, miner['name'])(**miner['params'])
        loss['params']['distance'] = distance
        loss['params']['reducer'] = reducer
        criterion = getattr(losses, loss['name'])(**loss['params'])

    # if GPU available move model and loss to GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        reid_model = reid_model.cuda()
        criterion = criterion.cuda()

    # optimizer
    logger.info(f"Initializing Optimizer..")
    params = list(reid_model.parameters())
    optimizer = torch.optim.Adam(params=params, lr=learning_rate)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    if resume:
        logger.info(f"Loading saved checkpoint..")
        if use_cuda:
            map_location = "cuda:0"
        else:
            map_location = "cpu"
        ckpt = load_checkpoint(exp_folder_no_time, cp_path, map_location=map_location)
        reid_model.load_state_dict(ckpt['net'])
        init_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])
        logger.info("Latest checkpoint loaded.")

    # tensorboard
    tensorboard_path = os.path.join(exp_folder, "tensorboard_logs")
    writer = SummaryWriter(tensorboard_path)

    # now we start the main loop

    # first we evaluate on validation set to get accuracy
    # and embedding plot before training
    n_iter = start_n_iter
    best_val_accuracy = 0
    total_i = 0
    is_best = False
    color = list(np.random.choice(range(200), size=3))
    logger.info(f"Training begins ..")
    for epoch in range(init_epoch, epochs):
        # evaluate on validation dataset
        pbar = tqdm(enumerate(val_dataloader),
                    total=len(val_dataloader))
        # for loop going through dataset
        with torch.no_grad():
            reid_model.eval()
            embedding_list = []
            label_list = []
            for i, batch in pbar:
                # data preparation
                patch, track_id = batch
                if use_cuda:
                    patch = patch.cuda()
                    track_id = track_id.cuda()
                # forward and backward pass
                embedding = reid_model(patch, mode="eval")
                embedding_list.append(embedding)
                label_list.append(track_id)

        accuracy = get_knn_accuracy(embedding_list, label_list)
        if accuracy > best_val_accuracy:
            logger.info(f'Found new best at epoch ' + str(epoch))
            is_best = True
            best_val_accuracy = accuracy
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
        # plot validation embeddings
        if epoch % val_frequency == 0:
            fig = plot_embeddings(embedding_list, label_list)
            writer.add_figure("Validation Embedding", fig, epoch)
        # # eval HOTA
        # if epoch % 10 == 0:
        #     hota = track_for_model(dmr_dir, motion_model=None, reid_model=reid_model.eval(), model_dir=model_dir, num_seq=1)
        #     writer.add_scalar('HOTA', hota['HOTA'], epoch)
        #     writer.add_scalar('MOTA', hota['MOTA'], epoch)
        #     writer.add_scalar('IDF1', hota['IDF1'], epoch)
        #     if hota['HOTA'] > best_val_accuracy:
        #         logger.info(f'Found new best at epoch ' + str(epoch))
        #         best_val_accuracy = hota['HOTA']
        #         is_best = True
        # set models to train mode
        reid_model.train()
        ckpt_folder = os.path.join(exp_folder, "checkpoints")
        Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

        cpkt = {
            'net': reid_model.state_dict(),
            'epoch': epoch,
            'n_iter': n_iter,
            'optim': optimizer.state_dict(),
            'val_stats': accuracy
        }
        save_path = os.path.join(ckpt_folder, 'model.ckpt')
        save_checkpoint(cpkt, save_path, is_best, best_ckpt_path=path_to_best)
        is_best = False

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader))
        start_time = time.time()
        # for loop going through dataset
        running_loss = 0
        for i, batch in pbar:
            # data preparation
            patch, track_id = batch
            if use_cuda:
                patch = patch.cuda()
                track_id = track_id.cuda()
            prepare_time = start_time - time.time()
            # forward and backward pass
            if loss['name'] == "CrossEntropyLoss":
                embedding = reid_model(patch, mode="eval")
                c_loss = criterion(embedding, track_id)
            else:
                embedding = reid_model(patch, mode="train")
                miner_output = miner(embedding, track_id)
                c_loss = criterion(embedding, track_id, miner_output)
            running_loss += c_loss.item()
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

            # udpate tensorboardX
            total_i += 1
            writer.add_scalar('Training Loss', running_loss / (i + 1), total_i)

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {running_loss / (i + 1):.4f},  epoch: {epoch}/{epochs}')
            start_time = time.time()



    print("training_done")


def main(args):
    train_eval(args.dmr_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dmr_dir", help="path where model is stored")
    args = parser.parse_args()
    main(args)
