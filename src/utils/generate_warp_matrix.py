import pandas as pd
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

def get_warp_matrix(img_t1,img_t2):
    im1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY)
    sz = img_t1.shape
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 200
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
    return warp_matrix

def main(dmr_dir):
    MOT_path = "data/MOT20/train/"
    MOT_seq = os.path.join(dmr_dir,MOT_path)
    sequences = ['MOT20-02']
    for seq in sequences:
        warp_matrix_list = []
        print(" -- Generate warp matrix for {}".format(seq))
        frame_dir = os.path.join(MOT_seq,seq,"img1")
        # store first warp matrix as identity
        frame_list = sorted(os.listdir(frame_dir))
        for i, frames in tqdm(enumerate(frame_list), total=len(frame_list)):
            if i==0:
                warp_matrix_list.append(np.eye(2, 3, dtype=np.float32))
                continue
            frame_1_path = os.path.join(frame_dir,frame_list[i-1])
            frame_2_path = os.path.join(frame_dir,frame_list[i])
            frame_t1 = cv2.imread(frame_1_path)
            frame_t2 = cv2.imread(frame_2_path)
            warp_matrix = get_warp_matrix(frame_t1, frame_t2)
            warp_matrix_list.append(warp_matrix)
        print(len(warp_matrix_list))

        with open(os.path.join(dmr_dir,"data","motion_model","warp_matrix","MOT20","warp_matrix_{}.pickle".format(seq)), 'wb') as handle:
            pickle.dump(warp_matrix_list, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dmr_dir", help="path where model is stored")
    args = parser.parse_args()
    main(args)