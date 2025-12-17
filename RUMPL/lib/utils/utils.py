# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------
#
# Portions of this file are:
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import random
import torch

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        os.makedirs(root_output_dir, exist_ok=True)
        # root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def create_logger_sweep(cfg, cfg_name, exp_name=None, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    if exp_name is not None:
        exp_name = exp_name + '_' + time_str
        final_output_dir = root_output_dir / dataset / model / exp_name
    else:
        final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    if exp_name is not None:
        log_file = '{}_{}_{}.log'.format(exp_name, time_str, phase)
    else:
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))
        
        if 'random_seed_state' in checkpoint:
            print('=> random seed state loaded')
            random.setstate(checkpoint['random_seed_state'])
            np.random.set_state(checkpoint['numpy_random_seed_state'])
            torch.set_rng_state(checkpoint['torch_random_seed_state'])
        
        if 'wandb_id' in checkpoint:
            print('=> wandb id: {}'.format(checkpoint['wandb_id']))
            return start_epoch, model, optimizer, checkpoint['wandb_id']

        return start_epoch, model, optimizer

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))



def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()

MMPOSE2H36M = {
    1: 12,  # rhip
    2: 14,  # rkne
    3: 16,  # rank
    4: 11,  # lhip
    5: 13,  # lkne
    6: 15,  # lank
    9: 0,   # nose
    11: 5,  # lsho
    12: 7,  # lelb
    13: 9,  # lwri
    14: 6,  # rsho
    15: 8,  # relb
    16: 10, # rwri
}

def mmpose2h36m(mmpose, scores):
    h36m = np.zeros((17, 2))
    conf = np.zeros((17, 1))
    for i in range(17):
        try:
            h36m[i] = mmpose[MMPOSE2H36M[i]]
            conf[i] = scores[MMPOSE2H36M[i]]
        except KeyError:
            h36m[i] = np.array([0, 0])
            conf[i] = 0
    
    # calculate head
    head = mmpose[0:5].mean(axis=0)
    h36m[10] = head
    conf[10] = scores[0:5].mean(axis=0)
    
    # calculate neck
    neck = mmpose[3:7].mean(axis=0)
    h36m[8] = neck
    conf[8] = scores[3:7].mean(axis=0)
    
    # calculate root
    root = mmpose[11:13].mean(axis=0)
    h36m[0] = root
    conf[0] = scores[11:13].mean(axis=0)
    
    # calculate belly
    belly = np.mean([neck, root], axis=0)
    h36m[7] = belly
    conf[7] = np.mean([conf[8], conf[0]], axis=0)
    
    return h36m, conf

# from mmpose.apis import MMPoseInferencer
# pose2d_model = 'td-hm_hrnet-w32_8xb64-210e_coco-384x288'
# comp_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# mmpose_inferencer_util = MMPoseInferencer(pose2d_model, device=comp_device)

def run_mmpose(image, mmpose_inferencer, convert_to_h36m=True, return_coco=False, device='cuda', img_id=0):
    # result_mmpose_generator = mmpose_inferencer_util(image, show=False, device=device, verbose=False) #, show_progress=False)
    # result_mmpose = next(result_mmpose_generator)
    
    # do nasty workaround to work with gpu
    # temp_dir_image = '/globalscratch/users/a/b/abolfazl/PPT_files/temp_imgs/temp_img_{}.png'.format(img_id)
    # temp_dir_out = '/globalscratch/users/a/b/abolfazl/PPT_files/temp_outs'
    
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    temp_dir_image = '/scratch/abolfazl/{}/temp_imgs/'.format(slurm_job_id)
    temp_dir_out = '/scratch/abolfazl/{}/temp_outs/'.format(slurm_job_id)
    
    os.makedirs(temp_dir_image, exist_ok=True)
    os.makedirs(temp_dir_out, exist_ok=True)
    
    image_name = 'temp_img_{}.png'.format(img_id)
    # dump image to temp file as png
    im = Image.fromarray(image)
    im.save(os.path.join(temp_dir_image, image_name))
    model='td-hm_hrnet-w32_8xb64-210e_coco-384x288'
    
    # run script to get predictions
    os.system('python ~/mmpose/demo/inferencer_demo.py {} --pose2d {} --pred-out-dir {}'.format(os.path.join(temp_dir_image, image_name),
                                                                                                model, temp_dir_out))
    
    json_mmpose = os.path.join(temp_dir_out, 'temp_img_{}.json'.format(img_id))
    elapsed = time.time()
    while not os.path.exists(json_mmpose):
        time.sleep(.001)
        elapsed = time.time() - elapsed
        if elapsed > 100:
            print('Timeout! No predictions found.')
            raise Exception('Timeout! No predictions found.')
        
    # load predictions
    with open(json_mmpose, 'r') as f:
        mmpose_data = json.load(f)
    
    if convert_to_h36m == True:
        joints_2d, joints_2d_conf = mmpose2h36m(np.array(mmpose_data[0]['keypoints']), np.array(mmpose_data[0]['keypoint_scores']))
    elif convert_to_h36m == False:
        joints_2d = np.array(mmpose_data[0]['keypoints'])
        joints_2d_conf = np.array(mmpose_data[0]['keypoint_scores']).reshape((-1,1))
        
    # remove temp files
    os.remove(os.path.join(temp_dir_image, image_name))
    os.remove(json_mmpose)
    return joints_2d, joints_2d_conf
    
    # predictions_mmpose = result_mmpose['predictions'][0][0] # first image - first prediction
    # keypoints = np.array(predictions_mmpose['keypoints'])
    # keypoint_scores = np.array(predictions_mmpose['keypoint_scores'])
    # if convert_to_h36m:
    #     h36m_points_mmpose, h36m_scores_mmpose = mmpose2h36m(keypoints, keypoint_scores)
    #     if return_coco:
    #         return h36m_points_mmpose, h36m_scores_mmpose, keypoints, keypoint_scores.reshape(-1, 1)
    #     return h36m_points_mmpose, h36m_scores_mmpose
    # else:
    #     return keypoints, keypoint_scores.reshape(-1, 1)

def amass_vertices_to_joints(vertices, h36m_jregressor, R, t, K, lower_body_reversed=True):
    # joints = torch.einsum('bik,ji->bjk', [vertices, h36m_jregressor])
    joints = h36m_jregressor @ vertices
    joints = joints[None]
    joints_copy = joints.copy()
    if lower_body_reversed:
        # to account for the mistake in the regressor (lower body reversed)
        joints[:, 1:4] = joints_copy[:, 4:7]
        joints[:, 4:7] = joints_copy[:, 1:4]
    joints_cam = world_to_cam(joints, R, t)
    points_image = cam_to_image(joints_cam, K)
    return points_image, joints

def OKS(gt, pred, vis, sigmas):
    bbox_area = (gt[:, 0].max() - gt[:, 0].min()) * (gt[:, 1].max() - gt[:, 1].min())
    varss = (np.array(sigmas) * 2) ** 2
    
    d_2 = ((pred - gt) ** 2).sum(1)

    e = np.exp(- (d_2 / varss / (bbox_area + np.spacing(1)) / 2))
    
    e[vis == 0] = np.nan
    oks = np.nanmean(e)
    return oks