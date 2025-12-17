
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
import h5py
import pickle
import json
import argparse
import numpy as np
import copy
import collections
from scipy.spatial.transform import Rotation as R

import _init_paths
from core.inference import get_max_preds
from core.config import config
from core.config import update_config
from utils.utils import create_logger
from multiviews.pictorial import rpsm
from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
from multiviews.triangulate import triangulate_poses
from core.config import get_model_name
from core.evaluate import calc_mpjpe, print_per_kp, calc_distance_per_dim
import dataset
import models
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Pose Estimation')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    parser.add_argument(
        '--mmpose-as', type=str, default='pred', choices=['GT', 'pred'],
        help='whether to use mmpose output as GT or prediction'
    )
    parser.add_argument(
        '--eval-comments', help='eval comments', type=str, default=None
    )
    parser.add_argument(
        '--test-mmpose-confs-th', help='test mmpose confs th', type=float, default=None
    )
    parser.add_argument(
        '--test-mmpose-type', help='test mmpose type', type=str, default=None
    )

    parser.add_argument(
        '--test-add-noise-to-camera-calib', help='test add noise to camera calib', action='store_true', default=None
    )

    parser.add_argument(
        '--noise-rot-deg', help='noise rot degree', type=float, default=None
    )

    parser.add_argument(
        '--noise-trans-std', help='noise trans std', type=float, default=None
    )
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length

def create_datum_dict_for_amass(test_dataset, camera_id, idx):
    view_finder = test_dataset.views_in_amass.index(camera_id)
    datum = {
        'image': 'dummy',
        'joints_2d': test_dataset.db_2d['joints_2d_mmpose'][idx][view_finder] if test_dataset.use_mmpose else test_dataset.db_2d['joints_2d_amass'][idx][view_finder],
        'joints_2d_conf': test_dataset.db_2d['confs_2d_mmpose'][idx][view_finder] if test_dataset.use_mmpose else np.ones_like(test_dataset.db_2d['confs_2d_mmpose'][idx][view_finder]),
        'joints_3d_conf': np.ones_like(test_dataset.db[idx]),
    }
    pose_3d = copy.deepcopy(test_dataset.db[idx])
    if test_dataset.mix_smart_3d_amass_with_triangulated_mmpose:
        pose_3d_tri = copy.deepcopy(test_dataset.db_2d['triangulated_3d_mmpose'][idx])
        error_triang = np.sqrt(np.sum((pose_3d[test_dataset.keypoints_to_mix_amass_with_3d_triangulated_mmpose] - pose_3d_tri[test_dataset.keypoints_to_mix_amass_with_3d_triangulated_mmpose]) ** 2, axis=1))
        error_tring_all = np.zeros((17), dtype=bool)
        error_tring_all[test_dataset.keypoints_to_mix_amass_with_3d_triangulated_mmpose] = error_triang < test_dataset.epipolar_error_acceptance_threshold
        pose_3d[error_tring_all] = pose_3d_tri[error_tring_all]
    datum['joints_3d'] = pose_3d * 100 # convert to cm
    
    amass_2d_joints_vis = np.ones((datum['joints_2d'].shape[0], 3)) # (17, 3)
    amass_2d_joints_vis[datum['joints_2d'][..., 0] > test_dataset.image_size[0]] = 0
    amass_2d_joints_vis[datum['joints_2d'][..., 0] < 0] = 0
    amass_2d_joints_vis[datum['joints_2d'][..., 1] > test_dataset.image_size[1]] = 0
    amass_2d_joints_vis[datum['joints_2d'][..., 1] < 0] = 0
    datum['joints_vis'] = amass_2d_joints_vis
    
    camera_setup_to_use = test_dataset.db_2d['camera_setup_used'][idx]
    camera = test_dataset.cmu_cameras[camera_id][camera_setup_to_use].copy()
    datum['camera'] = camera
    return datum

def add_noise_to_rt(R_mat, t_vec, rot_noise_deg=1.0, trans_noise_std=0.01):
        """
        Add Gaussian noise to rotation matrix (via random axis-angle perturbation)
        and to translation vector.
        
        Args:
            R_mat (np.ndarray): 3x3 rotation matrix
            t_vec (np.ndarray): 3-element translation vector
            rot_noise_deg (float): std deviation of rotational noise in degrees
            trans_noise_std (float): std deviation of translation noise
            
        Returns:
            R_noisy (np.ndarray): Noisy 3x3 rotation matrix
            t_noisy (np.ndarray): Noisy 3-element translation vector
        """
        # Generate small random rotation (noise)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle_rad = np.deg2rad(np.random.normal(0, rot_noise_deg))
        rot_noise = R.from_rotvec(axis * angle_rad).as_matrix()
        
        # Apply noise to original rotation
        R_noisy = rot_noise @ R_mat  # You can also use R_mat @ rot_noise depending on convention
        
        # Add noise to translation vector
        t_noise = np.random.normal(0, trans_noise_std, size=3).reshape(*t_vec.shape)
        t_noisy = t_vec + t_noise

        return R_noisy, t_noisy

def main():
    time_tri = AverageMeter()
    
    args = parse_args()
    if args.test_mmpose_type is not None:
        config.DATASET.TEST_MMPOSE_TYPE = args.test_mmpose_type

    if args.test_add_noise_to_camera_calib is not None:
        config.DATASET.TEST_ADD_NOISE_TO_CAMERA_CALIB = args.test_add_noise_to_camera_calib
    if args.noise_rot_deg is not None:
        config.DATASET.NOISE_ROT_DEG = args.noise_rot_deg
    if args.noise_trans_std is not None:
        config.DATASET.NOISE_TRANS_STD = args.noise_trans_std
        
    if args.mmpose_as == 'GT':
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'test3d-tri-mmpose-GT')
    elif args.mmpose_as == 'pred':
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'test3d-tri-mmpose-pred')
    
    # if args.mmpose_as == 'GT':
    #     config.DATASET.USE_MMPOSE_VAL = True
    run_on_all_cameras = config.DATASET.TEST_ON_ALL_CAMERAS
    if config.DATASET.TEST_DATASET == 'multiview_cmu_panoptic':
        all_views= list(range(31))
        all_views.remove(20)
        all_views = all_views if config.DATASET.ALL_VIEWS_CMU is None else config.DATASET.ALL_VIEWS_CMU
        
    elif config.DATASET.TEST_DATASET == 'multiview_rich':
        all_views = list(range(8))
    elif config.DATASET.TEST_DATASET == 'multiview_openmplposer':
        all_views = list(range(1, 4))
    else:
        all_views = list(range(1, 5))
    n_views_train_test_all = config.DATASET.N_VIEWS_TRAIN_TEST_ALL

    prediction_path = os.path.join(final_output_dir,
                                   config.TEST.HEATMAP_LOCATION_FILE)
    if args.mmpose_as == 'pred':
        ########### load mmpose val dataset as pred ############
        config.DATASET.USE_MMPOSE_VAL = False
        if config.DATASET.TEST_MULTI_PERSON:
            config.DATASET.USE_MMPOSE_VAL = True
        test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
            config, config.DATASET.TEST_SUBSET, False)
        config.DATASET.USE_MMPOSE_VAL = True
        test_dataset_with_mmpose = eval('dataset.' + config.DATASET.TEST_DATASET)(
            config, config.DATASET.TEST_SUBSET, False)
    elif args.mmpose_as == 'GT':
        ########### load mmpose val dataset as GT ############
        config.DATASET.USE_MMPOSE_VAL = True
        test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
            config, config.DATASET.TEST_SUBSET, False)
        all_locations = h5py.File(prediction_path)['locations']     # (#sample, 17, 2)
        all_heatmaps = h5py.File(prediction_path)['heatmaps']       # (#sample, 17, 64, 64)

    cnt = 0
    if config.DATASET.TEST_DATASET == 'multiview_amass_cmu_panoptic_mpl':
        grouping = [[i]*len(config.DATASET.TEST_VIEWS) for i in range(len(test_dataset))] # there is no need for grouping in amass datasets because it's already grouped
        grouping_with_mmpose = [[i]*len(config.DATASET.TEST_VIEWS) for i in range(len(test_dataset_with_mmpose))] # there is no need for grouping in amass datasets because it's already grouped
    else:
        grouping = test_dataset.grouping
        grouping_with_mmpose = test_dataset_with_mmpose.grouping
        
    assert len(grouping) == len(grouping_with_mmpose), 'grouping size not match! grouping: {} vs grouping_with_mmpose: {}'.format(len(grouping), len(grouping_with_mmpose))
    mpjpes = []
    mpjpe_score_output = {}
    pose3d_output = {}
    
    actual_joints = test_dataset.actual_joints
    distance_mm_per_keypoint = {kp: [] for _, kp in actual_joints.items()}
    all_3d_gts = []
    all_3d_preds = []
    all_confs = []
    fnames = []

    not_consider_kp = None
    if config.NOT_CONSIDER_SOME_KP_IN_EVAL is not None:
        not_consider_kp = config.NOT_CONSIDER_SOME_KP_IN_EVAL
        kps_dict = {v: k for k, v in actual_joints.items()}
        not_consider_kp = [kps_dict[kp] for kp in not_consider_kp]
        
        
    for items, items_with_mmpose in zip(grouping, grouping_with_mmpose):      # all 4 views for one subject
        heatmaps = []
        locations = []
        poses = []          # save same 3D world coordinate 4 times
        poses_confs = []   # save same 3D world coordinate confidence 4 times
        cameras = []
        confs = []
        locations_gt = []
        confs_gt = []
        # fnames.append([])
        if run_on_all_cameras and config.DATASET.TEST_DATASET not in ['multiview_rich', 'multiview_cmu_panoptic', 'multiview_h36m']:
            views_chosen = np.random.choice(all_views, n_views_train_test_all, replace=False)
            items_chosen = [items[all_views.index(v)] for v in views_chosen]
            items_with_mmpose_chosen = [items_with_mmpose[all_views.index(v)] for v in views_chosen] 
        else:
            items_chosen = items
            items_with_mmpose_chosen = items_with_mmpose

        # get information of all views
        for cam_iter, idx in enumerate(items_chosen):   #
            idx_with_mmpose = items_with_mmpose_chosen[cam_iter]
            if config.DATASET.TEST_DATASET.startswith('multiview_amass_'):
                camera_id = config.DATASET.TEST_VIEWS[cam_iter]
                datum = create_datum_dict_for_amass(test_dataset, camera_id, idx)
            else:
                datum = test_dataset.db[idx]
            camera = datum['camera']        # dict: {R, T, fx, fy, cx, cy}
            if 'fx' not in camera:
                camera['fx'] = camera['K'][0, 0]
                camera['fy'] = camera['K'][1, 1]
                camera['cx'] = camera['K'][0, 2]
                camera['cy'] = camera['K'][1, 2]
            if 'k' not in camera:
                camera['k'] = [camera['distCoef'][0], camera['distCoef'][1], camera['distCoef'][4]]
                camera['p'] = [camera['distCoef'][2], camera['distCoef'][3]]

            if config.DATASET.TEST_ADD_NOISE_TO_CAMERA_CALIB:
                camera['R'], camera['T'] = add_noise_to_rt(camera['R'], camera['T'], rot_noise_deg=config.DATASET.NOISE_ROT_DEG, trans_noise_std=config.DATASET.NOISE_TRANS_STD)
                camera['t'] = -camera['R'] @ camera['T']
            cameras.append(camera)
            # fnames[-1].append(datum['image'])

            if args.mmpose_as == 'pred':
                if config.DATASET.TEST_DATASET == 'multiview_cmu_panoptic':
                    poses.append(datum['joints_3d'])
                    poses_confs.append(datum['joints_3d_conf'])
                elif config.DATASET.TEST_DATASET == 'multiview_amass_cmu_panoptic_mpl':
                    poses.append(datum['joints_3d'])
                elif config.DATASET.TEST_DATASET == 'multiview_rich':
                    # if config.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST:
                    #     pose_3d_tri = copy.deepcopy(datum['triangulated_3d_mmpose'][idx])
                    poses.append(datum['joints_3d'])
                elif config.DATASET.TEST_DATASET.startswith('multiview_openmplposer'):
                    poses.append(datum['joints_3d'])
                else:
                    poses.append(
                        camera_to_world_frame(datum['joints_3d_camera'], camera['R'],
                                            camera['T']))       # pose in 3D world coordinate (17, 3)
                
                if config.DATASET.TEST_DATASET == 'multiview_amass_cmu_panoptic_mpl':
                    datum_pred = create_datum_dict_for_amass(test_dataset_with_mmpose, camera_id, idx_with_mmpose)
                else:
                    datum_pred = test_dataset_with_mmpose.db[idx_with_mmpose]
                assert datum['image'] == datum_pred['image']
                locations.append(datum_pred['joints_2d'])
                confs.append(datum_pred['joints_2d_conf'])
            elif args.mmpose_as == 'GT':
                locations_gt.append(datum['joints_2d'])
                confs_gt.append(datum['joints_2d_conf'])
                locations.append(all_locations[cnt])
                heatmaps.append(all_heatmaps[cnt])
            cnt += 1

        # s_11_act_16_subact_01_ca_04/s_11_act_16_subact_01_ca_04_000090.jpg
        keypoint_vis = datum['joints_vis']  # (20, 3)
        u2a = test_dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        u = np.array(list(a2u.values()))
        keypoint_vis = keypoint_vis[u]          # (17, 3)

        locations = np.array(locations)[:, :, :2]               # (#view, 17, 2) in original scale
        if args.mmpose_as == 'pred':
            confs = np.array(confs)[:, :, :1]                       # (#view, 17, 1)
            # TODO put zero confidence when the keypoint is not visible
        elif args.mmpose_as == 'GT':
            heatmaps = np.array(heatmaps)                           # (#view, 17, 64, 64)
            _, confs = get_max_preds(heatmaps)                      # (#view, 17, 1)
            locations_gt = np.array(locations_gt)[:, :, :2]               # (#view, 17, 2) in original scale
            confs_gt = np.array(confs_gt)[:, :, :1]                       # (#view, 17, 1)
            poses = triangulate_poses(cameras, locations_gt, confs_gt.squeeze())

        start = time.time()
        prediction = triangulate_poses(cameras, locations, confs.squeeze())      # list, element: (17, 3)
        time_tri.update(time.time() - start)
        
        

        if config.DATASET.TEST_DATASET == 'multiview_cmu_panoptic':
            if config.DATASET.TEST_MULTI_PERSON:
                poses[-1][(poses_confs[-1] <= 0).reshape(-1), :] = np.nan
                pjpe, mpjpe = calc_mpjpe(prediction[-1][None], poses[-1][None], mode='absolute')

            else:
                poses[0][(poses_confs[0] <= 0).reshape(-1), :] = np.nan
            # pjpe = np.sqrt(np.nansum((prediction[0] * keypoint_vis - poses[0] * keypoint_vis)**2, axis=1))
            # mpjpe = np.nanmean(pjpe)
            
                pjpe, mpjpe = calc_mpjpe(prediction[0][None], poses[0][None], mode='absolute')
            mpjpes.append(mpjpe)
        elif config.DATASET.TEST_DATASET == 'multiview_rich' or config.DATASET.TEST_DATASET.startswith('multiview_openmplposer'):
            pjpe = np.sqrt(np.sum((prediction[0] - poses[0])**2, axis=1))
            mpjpe = np.mean(pjpe)
            mpjpes.append(mpjpe)
        else:
            pjpe = np.sqrt(np.sum((prediction[0] * keypoint_vis - poses[0] * keypoint_vis)**2, axis=1))
            mpjpe = np.mean(pjpe)
            mpjpes.append(mpjpe)
        # mpjpe = np.mean(np.sqrt(np.sum((prediction[0] * keypoint_vis - poses[0] * keypoint_vis)**2, axis=1)))
        # mpjpes.append(mpjpe)
        
        if config.DATASET.TEST_MULTI_PERSON:
            all_3d_gts.append(poses[-1])
            all_3d_preds.append(prediction[-1])
        else:
            all_3d_gts.append(poses[0])
            all_3d_preds.append(prediction[0])
        # if args.test_mmpose_confs_th is not None:
        confs_ = np.array(confs).min(axis=0)
        all_confs.append(confs_)
        
        if config.MPJPE_PER_KEYPOINT:            
            for i, kp in actual_joints.items():
                distance_mm_per_keypoint[kp].append(pjpe[i])
        
        

        # print(mpjpe)
        if mpjpe > 150:
            print('Wrong MPJPE !!! ', datum['image'])

        # ================== save MPJPE score ==================
        if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            seq, frame = datum['image'].split('/')[1].split('_ca_')      # s_11_act_16_subact_01, 04_000090.jpg
            cams = ''
            for i in items_chosen:
                cams += '_{:02d}'.format(test_dataset.db[i]['camera_id'])
            frame_name = seq + '_' + frame[2:-4] + cams
            # frame_name = seq + frame[2:-4]
            mpjpe_score_output[frame_name] = mpjpe

            # ================== save 3D pose ================
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()
        
        elif config.DATASET.TRAIN_DATASET == 'multiview_cmu_panoptic':
            seq = datum['image'].split('/')[0]  # 171026_pose1
            fname = datum['image'].split('_')[-1].replace('.jpg', '')  # 000000000000
            cams = ''
            for i in items_chosen:
                cams += '_{:02d}'.format(test_dataset.db[i]['camera_id'])
            frame_name = seq + '_' + fname + cams
            # frame_name = seq + '_' + fname
            
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()
        
        elif config.DATASET.TRAIN_DATASET == 'multiview_rich':
            seq = datum['image'].split('/')[0]  # ParkingLot1_004_pushup1
            fname = datum['image'].split('/')[-1].split('_')[0] # 00017
            cams = ''
            for i in items_chosen:
                cams += '_{:02d}'.format(test_dataset.db[i]['camera_id'])
            frame_name = seq + '_' + fname + cams
            
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()
        elif config.DATASET.TEST_DATASET.startswith('multiview_amass_rumpl') and valid_camera_ids != []:
                fnames = [fname + valid_camera_id for fname, valid_camera_id in zip(fnames, valid_camera_ids)]
                
        elif config.DATASET.TEST_DATASET.startswith('multiview_openmplposer'):
            fname = datum['image']
            cams = ''
            for i in items_chosen:
                cams += '_{:02d}'.format(test_dataset.db[i]['camera_id'])
            frame_name = fname + cams

            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()
        elif config.DATASET.TEST_DATASET == 'multiview_amass_cmu_panoptic_mpl':
            frame_name = str(idx)
            
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()
            
        fnames.append(frame_name)

    # logger.info('Triangulation Time: sum (avg) {time_tri.sum:.3f}s ({time_tri.avg:.3f}s)'.format(time_tri=time_tri))
    msg = 'Triangulation Time: sum (avg) {time_tri.sum:.3f}s ({time_tri.avg:.3f}s)\t' \
            'Speed {speed:.1f} samples/s'.format(
                time_tri=time_tri,
                speed=len(all_3d_gts) / time_tri.sum,
                )
    logger.info(msg)
    # write speed to a file
    with open(os.path.join(final_output_dir, 'speed.txt'), 'w') as f:
        f.write(str(len(all_3d_gts) / time_tri.sum))
    logger.info('Triangulation MPJPE {}'.format(np.mean(mpjpes)))
    
    all_3d_gts = np.array(all_3d_gts)
    all_3d_preds = np.array(all_3d_preds)
    all_confs = np.array(all_confs)
    all_confs_dump = all_confs.copy()
    if args.test_mmpose_confs_th is not None:    
        all_confs = all_confs.mean(axis=1).squeeze()
        all_3d_gts = all_3d_gts[all_confs > args.test_mmpose_confs_th]
        all_3d_preds = all_3d_preds[all_confs > args.test_mmpose_confs_th]
        all_confs_dump = all_confs_dump[all_confs > args.test_mmpose_confs_th]
        logger.info('MMPOSE confs threshold applied: {}'.format(config.DATASET.TEST_MMPOSE_CONFS_TH))
    
    logger.info('Num samples: {}'.format(len(all_3d_preds)))
        
    pjpe_abs, mpjpe_abs = calc_mpjpe(all_3d_preds, all_3d_gts, mode='absolute', not_consider_kp=not_consider_kp)
    pjpe_rel, mpjpe_rel = calc_mpjpe(all_3d_preds, all_3d_gts, mode='relative', not_consider_kp=not_consider_kp)
    distance_per_dim_per_kp, distance_per_dim = calc_distance_per_dim(all_3d_preds, all_3d_gts) 
    if not_consider_kp is not None:
        logger.info('Warning: MPJPE is calculated not considering keypoints: {}'.format(not_consider_kp))
    logger.info('Absolute MPJPE: {}'.format(mpjpe_abs))
    to_log = print_per_kp(pjpe_abs, actual_joints.values())
    logger.info('Absolute MPJPE per keypoint: {}'.format(to_log))
    
    logger.info('Distance per dimension: {}'.format(distance_per_dim))
    to_log = print_per_kp(distance_per_dim_per_kp, actual_joints.values())
    logger.info('Distance per dimension per keypoint: {}'.format(to_log))
    
    logger.info('Relative MPJPE: {}'.format(mpjpe_rel))
    to_log = print_per_kp(pjpe_rel, actual_joints.values())
    logger.info('Relative MPJPE per keypoint: {}'.format(to_log))
    
    ### save mpjpe to pkl file
    name_values = collections.OrderedDict()
    joint_names = actual_joints
    for i in range(len(joint_names)):
        name_values[joint_names[i]] = pjpe_abs[i]
    test_name = config.DATASET.TEST_DATASET
    # use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
    state = 'mmpose_as_{}'.format(args.mmpose_as)
    eval_comments = args.eval_comments if args.eval_comments is not None else ''
    if eval_comments != '':
        test_name += '_{}'.format(eval_comments)
    absolute_or_relative = 'absolute'
    pkl_file = os.path.join(final_output_dir, 'mpjpe_{}_{}_{}.pkl'.format(test_name, state, absolute_or_relative))    
    with open(pkl_file, 'wb') as f:
        pickle.dump(name_values, f)
        
    name_values = collections.OrderedDict()
    joint_names = actual_joints
    for i in range(len(joint_names)):
        name_values[joint_names[i]] = pjpe_rel[i]
    test_name = config.DATASET.TEST_DATASET
    # use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
    state = 'mmpose_as_{}'.format(args.mmpose_as)
    eval_comments = args.eval_comments if args.eval_comments is not None else ''
    if eval_comments != '':
        test_name += '_{}'.format(eval_comments)
    absolute_or_relative = 'relative'
    pkl_file = os.path.join(final_output_dir, 'mpjpe_{}_{}_{}.pkl'.format(test_name, state, absolute_or_relative))    
    with open(pkl_file, 'wb') as f:
        pickle.dump(name_values, f)
    ##############################
    
    ### save 3D pose to pkl file    
    save_file = config.TEST.PRED_GT_LOCATION_FILE
    test_name = config.DATASET.TEST_DATASET
    state = 'mmpose_as_{}'.format(args.mmpose_as)
    # use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
    eval_comments = args.eval_comments if args.eval_comments is not None else ''
    if eval_comments != '':
        test_name += '_{}'.format(eval_comments)
    file_name = os.path.join(final_output_dir, '{}_{}_{}.pkl'.format(save_file.replace('.pkl', ''), test_name, state))
    with open(file_name, 'wb') as f:
        pickle.dump({'pred': all_3d_preds, 'gt': all_3d_gts}, f)
        
    if len(fnames) > 0:
        # fnames = [fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[-1].replace('.jpg','') for fname in fnames]
        # to_dump = {}
        # for fname, pred_dump, gt_dump in zip(fnames, all_3d_preds, all_3d_gts):
        #     to_dump[fname] = {'gt': gt_dump, 'pred': pred_dump}
        to_dump = {
            'fnames': fnames,
            'pred': all_3d_preds,
            'gt': all_3d_gts,
            'confs_2d': all_confs_dump,
        }
        file_name = os.path.join(final_output_dir, '{}_{}_{}_dict.pkl'.format(save_file.replace('.pkl', ''), test_name, state))
        with open(file_name, 'wb') as f:
            pickle.dump(to_dump, f)
    
    
    if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
        json.dump(mpjpe_score_output, open(os.path.join(final_output_dir, 'mpjpe_score.json'), 'w'), indent=4, sort_keys=True)
        json.dump(pose3d_output, open(os.path.join(final_output_dir, 'output_3d_joint.json'), 'w'), indent=4, sort_keys=True)

        # MPJPE on Each Action Sequence
        action_map = test_dataset.index_to_action_names()
        avg = []
        avg_pose_seq = {}
        for k in action_map:
            avg_pose_seq[k] = []

        for frame in mpjpe_score_output:
            # frame name: s_09_act_02_subact_01_000001
            act = frame[9:11]
            act = int(act)
            avg_pose_seq[act].append(mpjpe_score_output[frame])

            avg.append(mpjpe_score_output[frame])

        pose_seq_out_str = "\n"
        to_dump_per_action = {}
        for k in action_map:
            res = avg_pose_seq[k]
            mpjpe = sum(res) / len(res)
            pose_seq_out_str = pose_seq_out_str + action_map[k] + '\t{}\n'.format(mpjpe)
            to_dump_per_action[action_map[k]] = mpjpe

        logger.info('MPJPE on each Action: ' + pose_seq_out_str)
        absolute_or_relative = 'absolute'
        pkl_file = os.path.join(final_output_dir, 'mpjpe_perAction_{}_{}_{}.pkl'.format(test_name, state, absolute_or_relative))    
        with open(pkl_file, 'wb') as f:
            pickle.dump(to_dump_per_action, f)

        print(sum(avg) / len(avg))
    
    elif config.DATASET.TRAIN_DATASET == 'multiview_cmu_panoptic':
        json.dump(mpjpe_score_output, open(os.path.join(final_output_dir, 'mpjpe_score.json'), 'w'), indent=4, sort_keys=True)
        json.dump(pose3d_output, open(os.path.join(final_output_dir, 'output_3d_joint.json'), 'w'), indent=4, sort_keys=True)
        
    # # if config.DATASET.TRAIN_DATASET == 'multiview_h36m' and config.MPJPE_PER_KEYPOINT:
    # if config.MPJPE_PER_KEYPOINT:
    #     # actual_joints = test_dataset.actual_joints
    #     # distance_mm_per_keypoint = {kp: [] for _, kp in actual_joints.items()}
    #     # for frame_name, v in pose3d_output.items():
    #     #     pred = np.array(v['pred'])
    #     #     gt = np.array(v['GT'])
    #     #     distance = np.sqrt(np.sum((pred - gt)**2, axis=1))
    #     #     for i, kp in actual_joints.items():
    #     #         distance_mm_per_keypoint[kp].append(distance[i])
    #     distance_mm_per_keypoint = {k: np.mean(v) for k, v in distance_mm_per_keypoint.items()}
    #     str_per_kp = "\n"
    #     for k, v in distance_mm_per_keypoint.items():
    #         str_per_kp = str_per_kp + k + '\t{}\n'.format(v)
    #     logger.info('MPJPE per keypoint: {}'.format(str_per_kp))
    #     print(distance_mm_per_keypoint)
        
    # # relative pose error
    # actual_joints = test_dataset.actual_joints
    # # distance_rel_per_keypoint = {kp: [] for _, kp in actual_joints.items()}
    # distance_rel_per_keypoint = np.zeros((len(actual_joints)))
    # for frame_name, v in pose3d_output.items():
    #     pred = np.array(v['pred'])
    #     gt = np.array(v['GT'])
    #     pred_on_gt = pred - pred[0] + gt[0]
    #     dist = np.sqrt(np.nansum((pred_on_gt - gt)**2, axis=1))
    #     distance_rel_per_keypoint += dist
    # distance_rel_per_keypoint /= len(pose3d_output)
    # distance_rel_per_keypoint = {k: v for k, v in zip(actual_joints.values(), distance_rel_per_keypoint)}
    # str_per_kp = "\n"
    # for k, v in distance_rel_per_keypoint.items():
    #     str_per_kp = str_per_kp + k + '\t{}\n'.format(v)
    # logger.info('Relative Pose Error per keypoint: {}'.format(str_per_kp))
    # print(distance_rel_per_keypoint)
            

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count            
        

if __name__ == '__main__':
    main()

