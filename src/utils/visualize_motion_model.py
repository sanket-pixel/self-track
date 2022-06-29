from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
from tqdm import tqdm

from src.dataloader.tracklet_loader import Tracklet
from src.models.motion.tbmot import MotionModel
from src.utils.configuration import Configuration
from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

motion_config = Configuration.load_json(os.path.join("src", "configs", "motion.json"))

# data loader config
training_data_path = motion_config.DataLoader.training_data_path

# transformer config
nhead = motion_config.Transformer.nhead
feature_size = motion_config.Transformer.feature_size
linear_feature_size = motion_config.Transformer.linear_feature_size
num_layers = motion_config.Transformer.num_layers
N = motion_config.Transformer.N
dropout = motion_config.Transformer.dropout
layer_norm_eps = motion_config.Transformer.layer_norm_eps

# Training Config
batch_size = motion_config.Training.batch_size
num_workers = motion_config.Training.num_workers
epochs = motion_config.Training.epochs
learning_rate = motion_config.Training.learning_rate
init_epoch = motion_config.Training.init_epoch
resume = motion_config.Training.resume
cp_path = motion_config.Training.cp_path
path_to_best = motion_config.Training.path_to_best
val_frequency = motion_config.Training.val_frequency
model_dir_name = motion_config.Training.model_dir_name
experiment_name = motion_config.Training.experiment_name


dmr_dir = '/home/group-cvg/cvg-students/sshah/d_m_r'

val_dataset = Tracklet(mode="visualize",N=N, dmr_path=dmr_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=False, shuffle=True)
exp_folder = os.path.join(dmr_dir, "models", model_dir_name, experiment_name, cp_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_bbox = MotionModel(feature_size=feature_size,num_layers=num_layers,nhead=nhead,bbox_dim=4)
# ckpt = load_checkpoint(ckpt_dir_or_file=exp_folder,cp_path="checkpoints",load_best=True,map_location=device)
# trans_bbox.load_state_dict(ckpt['net'])
criterion = torch.nn.SmoothL1Loss()

use_cuda = torch.cuda.is_available()
if use_cuda:
    trans_bbox = trans_bbox.cuda()
    criterion = criterion.cuda()

def rect_coord(bbox,color='r',linestyle='--'):
    x, y, w, h = bbox
    x1 = x - w / 2
    y1 = y - h / 2
    patch = patches.Rectangle((x1,y1), w, h, edgecolor=color,facecolor='none',linestyle=linestyle)
    return patch



tensorboard_path = os.path.join(exp_folder, "tensorboard_logs")
writer = SummaryWriter(tensorboard_path)
validation_loss = []
# start validation
pbar = tqdm(enumerate(val_dataloader),
            total=len(val_dataloader),colour="BLUE")
with torch.no_grad():
    trans_bbox.eval()
    for i, batch in pbar:
        # data preparation
        input_bbox, target_bbox, frame = batch
        frame = frame.squeeze().numpy()
        if use_cuda:
            input_bbox = input_bbox.cuda().permute(1, 0, 2)
            target_bbox = target_bbox.cuda().permute(1, 0, 2)
        # forward and backward pass
        predicted_bbox = trans_bbox(input_bbox[:,:,-4:])
        l1_loss = criterion(predicted_bbox, target_bbox[:,:,-4:])
        validation_loss.append(l1_loss.item())
        fig, ax = plt.subplots()
        ax.imshow(frame)
        gt_point = target_bbox[0, 0, :4].squeeze().cpu().numpy()
        gt_patch = rect_coord(gt_point,'g',linestyle='--')
        ax.add_patch(gt_patch)
        previous_point = input_bbox[-1:, 0, :4].squeeze().cpu().numpy()
        # prev_patch = rect_coord(previous_point,'r')
        # ax.add_patch(prev_patch)
        predicted_diff = predicted_bbox[0, 0, -4:].squeeze().cpu().numpy()
        predicted_point = previous_point + predicted_diff
        predicted_patch = rect_coord(predicted_point, 'b',linestyle='-.')
        ax.add_patch(predicted_patch)
        # plt.scatter(gt_point[0], gt_point[1], s=2, c='red', marker='o')
        # plt.scatter(predicted_point[0], predicted_point[1], s=2, c='green', marker='o')
        # plt.scatter(previous_point[0], previous_point[1], s=5, c='blue', marker='x')
        # plt.show()
        if i==20:
            print(sum(validation_loss))
            plt.show()
            break


    avg_loss = np.mean(validation_loss)
    print(avg_loss)




