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

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts
from utils.utils_amass import rotate_pose
from utils.calib import cam_to_world, cam_to_image, world_to_cam
import collections
import logging
import pickle
import json
import glob 
import xml.etree.ElementTree as ET
from itertools import combinations


logger = logging.getLogger(__name__)

downsample = 16

# set random seed
np.random.seed(0)

COCO2H36M = {
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

class JointsDataset_RUMPL(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset
        self.views = cfg.DATASET.TRAIN_VIEWS if is_train else cfg.DATASET.TEST_VIEWS
        if self.views is None:
            self.views = list(range(1, 5))
        self.n_views = len(self.views)
        
        # work with random number of views
        if cfg.DATASET.TRAIN_RANDOM_NUM_VIEWS:
            self.max_random_n_views = cfg.DATASET.MAX_NUM_VIEWS
        else:
            self.max_random_n_views = None
            
        self.test_multi_person = cfg.DATASET.TEST_MULTI_PERSON if not is_train else False
        
        self.run_on_all_cameras = cfg.DATASET.TRAIN_ON_ALL_CAMERAS if is_train else cfg.DATASET.TEST_ON_ALL_CAMERAS
        self.run_on_sphere = cfg.DATASET.TRAIN_ON_SPHERE if is_train else cfg.DATASET.TEST_ON_SPHERE
        self.not_train_on_test_views = cfg.DATASET.NOT_TRAIN_ON_TEST_VIEWS
        
        self.sphere_views = cfg.DATASET.TRAIN_SPHERE_VIEWS if is_train else cfg.DATASET.TEST_SPHERE_VIEWS
        
        # self.sphere_main_cameras = cfg.DATASET.SPHERE_MAIN_CAMERAS
        # if self.run_on_all_cameras:
        #     self.n_views = cfg.DATASET.N_VIEWS_TRAIN_TEST_ALL
        
        self.amass_with_random_cameras = cfg.DATASET.TRAIN_AMASS_WITH_RANDOM_CAMERAS if is_train else cfg.DATASET.TEST_AMASS_WITH_RANDOM_CAMERAS
        
        if self.run_on_sphere and self.amass_with_random_cameras:
            raise ValueError('run_on_sphere and amass_with_random_cameras cannot be both True')
        
        self.run_on_specific_camera_setups = cfg.DATASET.TRAIN_ON_SPECIFIC_CAMERA_SETUPS
        self.pick_random_cameras_from_specific_setups = cfg.DATASET.PICK_RANDOM_CAMERAS_FROM_SPECIFIC_SETUPS
            
        self.all_views_cmu = list(range(31))
        self.all_views_cmu.remove(20)
        self.all_views_cmu = self.all_views_cmu if cfg.DATASET.ALL_VIEWS_CMU is None else cfg.DATASET.ALL_VIEWS_CMU

        self.all_views_openmplposer = list(range(1, 4))
        
        self.all_views_rich = list(range(8))
        
        self.views_in_amass = [3, 6, 12, 13, 23]
        
        self.min_angle_diff = cfg.DATASET.MIN_ANGLE_DIFF
        self.min_oks = cfg.DATASET.MIN_OKS
        
        self.use_mmpose = False
        if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
            self.use_mmpose = True
        elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_VAL) and not is_train:
            self.use_mmpose = True
            
        self.mmpose_type = cfg.DATASET.TRAIN_MMPOSE_TYPE if is_train else cfg.DATASET.TEST_MMPOSE_TYPE
        self.depth_type = cfg.DATASET.TRAIN_DEPTH_TYPE if is_train else cfg.DATASET.TEST_DEPTH_TYPE
        
        self.joints_vis_from_mmpose = cfg.DATASET.JOINTS_VIS_FROM_MMPOSE
        
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        
        self.h36m_old_datasets = cfg.DATASET.TRAIN_USE_H36M_OLD_DATASETS if is_train else cfg.DATASET.TEST_USE_H36M_OLD_DATASETS
        self.h36m_dataset_name = cfg.DATASET.TRAIN_H36M_DATASET_NAME if is_train else cfg.DATASET.TEST_H36M_DATASET_NAME
        if is_train:
            self.filter_groupings = cfg.DATASET.FILTER_GROUPINGS and cfg.DATASET.TRAIN_FILTER_GROUPINGS
        else:
            self.filter_groupings = cfg.DATASET.FILTER_GROUPINGS and cfg.DATASET.TEST_FILTER_GROUPINGS

        
        self.kp_visiblity_th = cfg.DATASET.KP_VISIBILITY_TH
        self.zero_tokens_for_missing_joints = cfg.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS
        
            
        self.use_3d_triangulated_mmpose = cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.USE_3D_TRIANGULATED_MMPOSE_TEST
        self.mix_3d_amass_with_triangulated_mmpose = cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST
        self.mix_smart_3d_amass_with_triangulated_mmpose = cfg.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST
        self.epipolar_error_acceptance_threshold = cfg.DATASET.EPIPOLAR_ERROR_ACCEPTANCE_THRESHOLD
        self.keypoints_to_mix_amass_with_3d_triangulated_mmpose = cfg.DATASET.KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE
        self.filter_cmu_wrong_cases = cfg.DATASET.TRAIN_FILTER_CMU_WRONG_CASES if is_train else cfg.DATASET.TEST_FILTER_CMU_WRONG_CASES
            
        self.mix_gt_with_mmpose = cfg.DATASET.MIX_GT_WITH_MMPOSE_WHEN_USE_MMPOSE
        self.mix_amass_with_mmpose = cfg.DATASET.MIX_AMASS_WITH_MMPOSE_WHEN_USE_MMPOSE or self.mix_gt_with_mmpose
        self.keypoints_to_mix = cfg.DATASET.KEYPOINTS_TO_MIX
        
        self.target_normalized_3d = cfg.DATASET.TARGET_NORMALIZED_3D
        
        self.room_min_x = cfg.DATASET.ROOM_MIN_X
        self.room_max_x = cfg.DATASET.ROOM_MAX_X
        self.room_min_y = cfg.DATASET.ROOM_MIN_Y
        self.room_max_y = cfg.DATASET.ROOM_MAX_Y
        self.room_min_z = cfg.DATASET.ROOM_MIN_Z
        self.room_max_z = cfg.DATASET.ROOM_MAX_Z
        self.room_center = cfg.DATASET.ROOM_CENTER
        
        self.root = cfg.DATASET.ROOT
        self.root_2 = cfg.DATASET.ROOT_2DATSET
        self.root_3 = cfg.DATASET.ROOT_3DATSET
        if cfg.DATASET.ROOT_TRAIN is not None and is_train:
            self.root = cfg.DATASET.ROOT_TRAIN
        if cfg.DATASET.ROOT_TEST is not None and not is_train:
            self.root = cfg.DATASET.ROOT_TEST
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        if cfg.NETWORK.IMAGE_SIZE_TRAIN is not None and is_train:
            self.image_size = cfg.NETWORK.IMAGE_SIZE_TRAIN
        if cfg.NETWORK.IMAGE_SIZE_TEST is not None and not is_train:
            self.image_size = cfg.NETWORK.IMAGE_SIZE_TEST
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        self.transform = transform
        self.db = []

        self.APPLY_NOISE = cfg.DATASET.APPLY_NOISE if is_train else cfg.DATASET.APPLY_NOISE_TEST
        self.NOISE_LEVEL = cfg.DATASET.NOISE_LEVEL
        
        self.APPLY_NOISE_MISSING = cfg.DATASET.APPLY_NOISE_MISSING
        self.APPLY_NOISE_MISSING_TEST = cfg.DATASET.APPLY_NOISE_MISSING_TEST
        self.MISSING_LEVEL = float(cfg.DATASET.MISSING_LEVEL)
        
        # old version
        self.apply_noise_cameras = cfg.DATASET.APPLY_NOISE_CAMERAS
        self.R_noise_value = cfg.DATASET.R_NOISE_VALUE
        self.t_noise_value = cfg.DATASET.T_NOISE_VALUE

        # noise camera new
        self.add_noise_to_camera_calib = cfg.DATASET.TRAIN_ADD_NOISE_TO_CAMERA_CALIB if is_train else cfg.DATASET.TEST_ADD_NOISE_TO_CAMERA_CALIB
        self.noise_rot_deg = cfg.DATASET.NOISE_ROT_DEG
        self.noise_trans_std = cfg.DATASET.NOISE_TRANS_STD
        
        self.APPLY_SMART_PSEUDO_TRAINING = cfg.TRAIN.SMART_PSEUDO_TRAINING if is_train else False
        self.epipolar_error_threshold = cfg.TRAIN.EPIPOLAR_ERROR_THRESHOLD
        self.downsample = cfg.DOWNSAMPLE
        
        self.centeralize_root_first = cfg.DATASET.CENTERALIZE_ROOT_FIRST
        self.no_augmentation_3d = cfg.DATASET.NO_AUGMENTATION_3D
        
        self.output_in_meter = cfg.DATASET.OUTPUT_IN_METER
        self.no_augmentation = cfg.DATASET.NO_AUGMENTATION
        self.clip_joints = cfg.DATASET.CLIP_JOINTS
        self.inputs_normalized = cfg.DATASET.INPUTS_NORMALIZED
        self.normalize_cameras = cfg.DATASET.NORMALIZE_CAMERAS
        self.normalize_room = cfg.DATASET.NORMALIZE_ROOM
        self.normalize_room_firstly = cfg.DATASET.NORMALIZE_ROOM_FIRSTLY
        self.flip = cfg.DATASET.FLIP_3D
        self.rotate = cfg.DATASET.ROTATE_3D
        
        self.use_grid = True if cfg.DATASET.USE_GRID else False
        self.bug_test = cfg.DATASET.BUG_TEST_3D_EMB
        self.switch_x_z = cfg.DATASET.SWITCH_X_Z
        self.switch_x_y = cfg.DATASET.SWITCH_X_Y
        self.switch_y_z = cfg.DATASET.SWITCH_Y_Z
        self.switch_z_x_y = cfg.DATASET.SWITCH_Z_X_Y
        self.switch_y_z_x = cfg.DATASET.SWITCH_Y_Z_X
        self.amass_data_no_axis_swap = cfg.DATASET.AMASS_DATA_NO_AXIS_SWAP
        self.use_t = cfg.DATASET.USE_T
        
        self.amass_val_located = cfg.DATASET.AMASS_VAL_LOCATED
        
        self.CMU_KEYPOINT_STANDARD = cfg.DATASET.CMU_KEYPOINT_STANDARD
        
        self.cmu_calib = cfg.DATASET.CMU_CALIB
        self.cmu_calibs_train = cfg.DATASET.TRAIN_CMU_CALIB
        self.cmu_calibs_val = cfg.DATASET.TEST_CMU_CALIB
        self.dome_calib_file_train = cfg.DATASET.TRAIN_DOME_CALIB_FILE
        self.dome_calib_file_val = cfg.DATASET.TEST_DOME_CALIB_FILE
        self.h36m_calib_actors = cfg.DATASET.TRAIN_H36M_CALIB_ACTORS if is_train else cfg.DATASET.TEST_H36M_CALIB_ACTORS
        
        self.use_helper_cameras = cfg.DATASET.USE_HELPER_CAMERAS
        self.views_helper = cfg.DATASET.TRAIN_VIEWS_HELPER
        if self.views_helper is not None:
            self.n_views_helper = len(self.views_helper)
        self.root_views_helper = cfg.DATASET.ROOT_VIEWS_HELPER
        
        self.train_n_samples = cfg.DATASET.TRAIN_N_SAMPLES
        self.test_n_samples = cfg.DATASET.TEST_N_SAMPLES
        
        self.penalize_confidence = cfg.DATASET.PENALIZE_CONFIDENCE
        self.penalize_confidence_a = cfg.DATASET.PENALIZE_CONFIDENCE_A
        self.penalize_confidence_b = cfg.DATASET.PENALIZE_CONFIDENCE_B
        self.penalize_factor = cfg.DATASET.PENALIZE_CONFIDENCE_FACTOR
        self.only_keep_inside_room = cfg.DATASET.ONLY_KEEP_INSIDE_ROOM
        self.only_keep_if_in_calibs_actors = cfg.DATASET.ONLY_KEEP_IF_IN_CALIBS_ACTORS
        
        self.train_on_all_amass = cfg.DATASET.TRAIN_ON_ALL_AMASS
        
        self.place_person_in_center = cfg.DATASET.TRAIN_PLACE_PERSON_IN_CENTER if is_train else cfg.DATASET.TEST_PLACE_PERSON_IN_CENTER
        self.bring_amass_root_to_room_center = cfg.DATASET.BRING_AMASS_ROOT_TO_ROOM_CENTER
        
        self.intrinsic_to_meters = cfg.DATASET.INTRINSIC_TO_METERS_IF_OUTPUT_IN_METER
        
        self.use_h36m_cameras_on_cmu = cfg.DATASET.USE_H36M_CAMERAS_ON_CMU
        self.use_cmu_cameras_on_cmu = cfg.DATASET.USE_CMU_CAMERAS_ON_CMU
        self.use_cmu_cameras_on_h36m = cfg.DATASET.USE_CMU_CAMERAS_ON_H36M
        
        self.axis_yz_swap_for_3d = cfg.DATASET.AXIS_YZ_SWAP_FOR_3D
        self.negate_y_or_z_for_3d = cfg.DATASET.NEGATE_Y_OR_Z_FOR_3D
        if self.negate_y_or_z_for_3d == 'z':
            self.neg = 2
        elif self.negate_y_or_z_for_3d == 'y':
            self.neg = 1
        else:
            raise 'negate_y_or_z_for_3d should be either y or z'
        
        self.shift_room = cfg.DATASET.SHIFT_ROOM
        if is_train:
            if type(cfg.DATASET.SHIFT_ROOM_TRAIN_VALUE) == str:
                self.shift_room_value = cfg.DATASET.SHIFT_ROOM_TRAIN_VALUE
            else:
                self.shift_room_value = np.array(cfg.DATASET.SHIFT_ROOM_TRAIN_VALUE)
        else:
            if type(cfg.DATASET.SHIFT_ROOM_TEST_VALUE) == str:
                self.shift_room_value = cfg.DATASET.SHIFT_ROOM_TEST_VALUE
            else:
                self.shift_room_value = np.array(cfg.DATASET.SHIFT_ROOM_TEST_VALUE)
        # self.shift_room_value = np.array(cfg.DATASET.SHIFT_ROOM_TRAIN_VALUE) if is_train else np.array(cfg.DATASET.SHIFT_ROOM_TEST_VALUE)
        
        # if self.shift_room and self.normalize_room:
        #     raise ValueError('shift_room and normalize_room cannot be both True')
        
        self.intersection_ray_with = cfg.DATASET.INTERSECTION_RAY_WITH
        self.ray_as_intersection_with_donut = cfg.DATASET.RAY_AS_INTERSECTION_WITH_DONUT
        
        self.ground_z = cfg.DATASET.GROUND_Z
        
        # self.direction_as_angles = cfg.DATASET.DIRECTION_AS_ANGLES
        
        self.kp_visiblity_th = cfg.DATASET.KP_VISIBILITY_TH
        self.zero_tokens_for_missing_joints = cfg.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS
        
        ### master cameras ###
        self.n_master_cameras = cfg.DATASET.N_MASTER_CAMERAS
        
        #### sine encoding rays ####
        self.apply_sine_encoding_on_rays = cfg.DATASET.APPLY_SINE_ENCODING_ON_RAYS
        self.sine_d_model = cfg.DATASET.SINE_D_MODEL
        
        self.concat_cam_centers_to_rays = cfg.NETWORK.CONCAT_CAM_CENTER_TO_INPUTS
        self.concat_cam_axis_to_rays = cfg.NETWORK.CONCAT_CAM_AXIS_TO_RAYS
        if self.concat_cam_axis_to_rays and self.concat_cam_centers_to_rays:
            raise ValueError('concat_cam_centers_to_rays and concat_cam_axis_to_rays cannot be both True')
        
        #### use depth ####
        self.use_depth = cfg.NETWORK.CONCAT_DEPTH_AS_INPUT
        self.use_depth_with_noise_on_distance = cfg.DATASET.USE_DEPTH_WITH_NOISE_ON_DISTANCE
        if cfg.DATASET.TRAIN_USE_DEPTH_WITH_NOISE_ON_DISTANCE is not None and is_train:
            self.use_depth_with_noise_on_distance = cfg.DATASET.TRAIN_USE_DEPTH_WITH_NOISE_ON_DISTANCE
        if cfg.DATASET.TEST_USE_DEPTH_WITH_NOISE_ON_DISTANCE is not None and not is_train:
            self.use_depth_with_noise_on_distance = cfg.DATASET.TEST_USE_DEPTH_WITH_NOISE_ON_DISTANCE
        self.use_gt_depth = cfg.DATASET.TRAIN_USE_GT_DEPTH if is_train else cfg.DATASET.TEST_USE_GT_DEPTH
        if self.use_gt_depth and self.use_depth_with_noise_on_distance:
            raise ValueError('use_gt_depth and use_depth_with_noise_on_distance cannot be both True')
        self.mu_depth_noise = cfg.DATASET.MU_DEPTH_NOISE
        self.sigma_depth_noise = cfg.DATASET.SIGMA_DEPTH_NOISE

        # openmplposer
        self.openmplposer_calibs_train = cfg.DATASET.OPENMPLPOSER_CALIBS_TRAIN
        self.openmplposer_calibs_val = cfg.DATASET.OPENMPLPOSER_CALIBS_VAL
        
        self.use_amass_old_datasets = cfg.DATASET.TRAIN_USE_AMASS_OLD_DATASETS if is_train else cfg.DATASET.TEST_USE_AMASS_OLD_DATASETS
        self.use_amass_new_datasets_with_old_way = cfg.DATASET.TRAIN_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY if is_train else cfg.DATASET.TEST_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY
        self.flip_lower_body_kp = cfg.DATASET.FLIP_LOWER_BODY_KP_TEST if not is_train else False
        self.num_joints = 17
        union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax',
            9: 'neck',
            10: 'upper neck',
            11: 'nose',
            12: 'head',
            13: 'head top',
            14: 'lsho',
            15: 'lelb',
            16: 'lwri',
            17: 'rsho',
            18: 'relb',
            19: 'rwri'
        }

        self.union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }
        
        self.close_joints = {
            'rhip': 'lhip',
            'lkne': 'rkne',
            'rank': 'lank',
            'rsho': 'lsho',
            'relb': 'lelb',
            'nose': 'head',
        }
        
        self.COCO_PERSON_SIGMAS = [
            0.026,  # nose
            0.025,  # eyes
            0.025,  # eyes
            0.035,  # ears
            0.035,  # ears
            0.079,  # shoulders
            0.079,  # shoulders
            0.072,  # elbows
            0.072,  # elbows
            0.062,  # wrists
            0.062,  # wrists
            0.107,  # hips
            0.107,  # hips
            0.087,  # knees
            0.087,  # knees
            0.089,  # ankles
            0.089,  # ankles
        ]
        
        # self.amass_normalization_const = {
        #     'means': [-0.40025026059636765, -0.9625251337422954, -0.5074277076365897],
        #     'vars': [0.24167735219571976, 0.20952544156186956, 0.3594625811074876],
        # }
        
        self.actual_joints = {}
        self.u2a_mapping = {}
        
        
        self.val_keypoint_standard = cfg.DATASET.VAL_KEYPOINT_STANDARD

        # grid coordinate. For

        _y, _x = torch.meshgrid(torch.arange(self.image_size[0] // self.downsample),
                                torch.arange(self.image_size[1] // self.downsample))
        grid = torch.stack([_x, _y], dim=-1)  # Tensor, size:(32, 32, 2) val: 0-32
        grid = grid * self.downsample + self.downsample / 2.0 - 0.5  # Tensor, size:(32, 32, 2), val: 0-256
        self.grid = grid.view(-1, 2)  # Tensor, size:(hw, 2), val: 0-256

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping
    
    def compatible_cams(self, cameras):
        for k,cam in cameras.items():    
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['k'] = np.array([cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]])
            cam['p'] = np.array([cam['distCoef'][2], cam['distCoef'][3]])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3,1))
            cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
            cam['fx'] = cam['K'][0,0]
            cam['fy'] = cam['K'][1,1]
            cam['cx'] = cam['K'][0,2]
            cam['cy'] = cam['K'][1,2]
        return cameras

    def load_amass_new(self, anno_file):
        with open(anno_file, 'rb') as f:
            db = pickle.load(f)
        if 'triangulated_3d_mmpose' in db.keys():
            if db['triangulated_3d_mmpose'] is None:
                del db['triangulated_3d_mmpose']
        if self.use_mmpose:
            joints_2d_mmpose = db['joints_2d_mmpose']
            nan_indices = np.argwhere(np.isnan(joints_2d_mmpose))
            to_remove = np.ones((joints_2d_mmpose.shape[0],), dtype=bool)
            to_remove[nan_indices[:, 0]] = False
            if not to_remove.all():    
                for k in db.keys():
                    if k == "camera_parameters_all":
                        db[k] = [db[k][i] for i in range(len(db[k])) if to_remove[i]]
                    db[k] = db[k][to_remove]
                
        db_2d = {}
        for k in db.keys():
            if k != 'joints_3d':
                db_2d[k] = db[k]
        if self.mix_smart_3d_amass_with_triangulated_mmpose:
            db = db['joints_3d']
        elif self.mix_3d_amass_with_triangulated_mmpose:
            db = db['joints_3d']
            db[:, self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] = db_2d['triangulated_3d_mmpose'][:, self.keypoints_to_mix_amass_with_3d_triangulated_mmpose]
        elif self.use_3d_triangulated_mmpose:
            db = db_2d['triangulated_3d_mmpose']
        else:
            db = db['joints_3d']
        
        try:
            views_used = db_2d['views_used']
            self.views_in_amass = list(views_used[0])
        except:
            pass
        
        if self.run_on_sphere and self.sphere_views == []:
            n_heights = db_2d['camera_parameters_all'][0][-1]['category_height'] + 1
            n_angles = db_2d['camera_parameters_all'][0][-1]['category_angle'] + 1
            camera_combinations = []
            for i in range(n_heights):
                for j in range(n_angles):
                    if j == 0:
                        continue
                    camera_combinations.append([i * n_angles, i * n_angles + j])
            n_camera_combinations = len(camera_combinations)
            n_poses = len(db)
            
            db_2d['joints_3d'] = db.copy()
            db = np.arange(len(db))
            db = np.repeat(db, n_camera_combinations)
            db = [[d, c] for d, c in zip(db, camera_combinations * n_poses)]
        return db, db_2d
    
    def load_all_cameras_h36m(self, calib_file, actors):
        """
        loads all cameras from the calibration file and returns a dictionary
        containing the camera parameters for each camera
        each item has a list showing different camera setups (actors)
        """
        with open(calib_file, 'rb') as f:
            camera_data = pickle.load(f)
        cams = range(1, 5)
        cameras = {cam_id:[] for cam_id in cams}
        for cam_setup in actors:
            for cam_id in cams:
                camera = camera_data[(cam_setup, cam_id)]
                camera_dict = {}
                camera_dict['camera_setup'] = cam_setup
                camera_dict['camera_id'] = cam_id
                camera_dict['R'] = camera[0]
                camera_dict['T'] = camera[1]
                camera_dict['t'] = - np.linalg.inv(camera_dict['R'].T) @ camera_dict['T']
                camera_dict['fx'] = camera[2][0].squeeze()
                camera_dict['fy'] = camera[2][1].squeeze()
                camera_dict['cx'] = camera[3][0].squeeze()
                camera_dict['cy'] = camera[3][1].squeeze()
                camera_dict['k'] = camera[4]
                camera_dict['p'] = camera[5]
                camera_dict['K'] = np.array([[camera_dict['fx'], 0, camera_dict['cx']], 
                                             [0, camera_dict['fy'], camera_dict['cy']], 
                                            [0, 0, 1]])
                cameras[cam_id].append(camera_dict)
        return cameras
    
    def load_all_cameras_cmu(self, cmu_calibs):
        cams = range(31)
        cameras_all_calibs = {cam_id: [] for cam_id in cams}
        for cam_setup in cmu_calibs:
            calib_file = osp.join(self.root_3, cam_setup, 'calibration_{}.json'.format(cam_setup))
            with open(calib_file, 'r') as f:
                calibration_cat = json.load(f)
            cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd'}
            cameras = self.compatible_cams(cameras)
            for cam_id, cam in cameras.items():
                cameras_all_calibs[cam_id].append(cam)
        return cameras_all_calibs
    
    def load_all_cameras_openmplposer(self, calibs):

        def parse_matrix(node):
            rows = int(node.find('rows').text)
            cols = int(node.find('cols').text)
            data_text = node.find('data').text.strip().replace('\n', ' ')
            data = list(map(float, data_text.split()))
            return np.array(data).reshape((rows, cols))
        
        cams = range(1, 4)
        cameras_all_calibs = {cam_id: [] for cam_id in cams}
        xml_files = glob.glob(osp.join(calibs, '*.xml'))
        distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic

        for xml_file in xml_files:

            # Parse the XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            K = parse_matrix(root.find('Intrinsics'))
            P = parse_matrix(root.find('CameraMatrix'))

            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            R = P[:, :3]
            T = P[:, 3:]

            R = R.T

            F = np.diag([-1, 1, 1])
            R = R.copy()
            R = F @ R
            k = np.array([distCoef[0], distCoef[1], distCoef[4]])
            p = np.array([distCoef[2], distCoef[3]])

            camera_dict = {
                'camera_setup': 0,
                'camera_id': int(xml_file.split('/')[-1].split('_')[-1].replace('.xml', '')),
                'K': np.array(K),
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k': k,
                'p': p,
                'R': np.array(R),
                'T': np.array(T),
                't': -np.linalg.inv(R.T) @ T,
            }
            cameras_all_calibs[camera_dict['camera_id']].append(camera_dict)
        return cameras_all_calibs
    
    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError
    
    def filter_db(self, db):
        if self.use_h36m_cameras_on_cmu:
            views = [3, 6, 12, 13, 23]
            views = views[:self.n_views]
        else:
            views = self.views
        db_filtered = []
        for i in db:
            if i['camera_id'] not in views:
                continue
            else:
                db_filtered.append(i)
        return db_filtered

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for items in self.grouping:
            # for item in items:
            item = items[0]
            gt.append(self.db[item]['joints_3d'][su, :])       # (17, 3) in original scale
        gt = np.array(gt)           # (num_sample, 17, 3) 
        pred = pred[:, su, :]      # (num_sample, 17, 3) 
        # if self.relative_evaluation:
        #     gt = gt - gt[:, 0:1, :]    # (num_sample, 17, 3)
        #     pred = pred - pred[:, 0:1, :]  # (num_sample, 17, 3)
        distance_mm_per_keypoint = {kp: [] for _, kp in self.actual_joints.items()}
        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        
        for i, kp in self.actual_joints.items():
            distance_mm_per_keypoint[kp] = distance[:, i]
        
        distance_mm_per_keypoint = {k: np.mean(v) for k, v in distance_mm_per_keypoint.items()}
        str_per_kp = "\n"
        for k, v in distance_mm_per_keypoint.items():
            str_per_kp = str_per_kp + k + '\t{}\n'.format(v)
        logger.info('3D MPJPE per keypoint: {}'.format(str_per_kp))
        # detected = (distance <= headsize * threshold)

        # joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])
        mpjpe = np.mean(distance, axis=0)

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = mpjpe[i]
        return name_values, np.mean(distance)

    def __len__(self,):
        return len(self.db)

    def get_rays(self, idx, camera_id=None, camera_setup_to_use=None, shift_room_tri=None):
        db_rec = copy.deepcopy(self.db[idx])

        # ==================================== Image ====================================
        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        if db_rec['source'] == 'cmu_panoptic':
            image_file = osp.join(self.root, image_dir,
                              db_rec['image'])
        elif db_rec['source'] == 'rich':
            image_file = osp.join(self.root, 'images', 
                                  'train' if self.is_train else 'val',
                                  db_rec['image'])
        elif db_rec['source'] == 'openmplposer':
            image_file = 'NA'
        else:    
            image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                                db_rec['image'])

        # ==================================== Label ====================================
        joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        if self.flip_lower_body_kp:
            joints_copy = joints.copy()
            joints[4:7] = joints_copy[1:4]
            joints[1:4] = joints_copy[4:7]

        if 'joints_2d_conf' in db_rec and self.joints_vis_from_mmpose:
            joints_vis = db_rec['joints_2d_conf'].copy()        # (17, 3)   0,0,0 or 1,1,1
            if self.flip_lower_body_kp:
                joints_vis_copy = joints_vis.copy()
                joints_vis[4:7] = joints_vis_copy[1:4]
                joints_vis[1:4] = joints_vis_copy[4:7]
        else:
            joints_vis = np.ones((17, 1))
        

        center = np.array(db_rec['center']).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(db_rec['scale']).copy()        # (2, )     (s1, s2)
        rotation = 0

        # ==================================== Camera  ====================================
        if self.use_h36m_cameras_on_cmu:
            camera = self.h36m_cameras[camera_id][camera_setup_to_use].copy()
            if self.output_in_meter:
                camera['T'] = camera['T'] / 1000
                camera['t'] = camera['t'] / 1000
                
        elif self.use_cmu_cameras_on_h36m or self.use_cmu_cameras_on_cmu:
            camera = self.cmu_cameras[camera_id][camera_setup_to_use].copy()
            if self.output_in_meter:
                camera['T'] = camera['T'] / 100
                camera['t'] = camera['t'] / 100
        else:
            camera = db_rec['camera'].copy()  
            
        image_size = self.image_size.copy()
        if db_rec['source'] == 'rich' and camera['cx'] < camera['cy']:
            image_size = [image_size[1], image_size[0]]
        # ========================= 3D Target ===========================
        if db_rec['source'] == 'cmu_panoptic':
            joints_3d = db_rec['joints_3d'].copy()      # (17, 3) in world coordinate
            if self.flip_lower_body_kp:
                joints_3d_copy = joints_3d.copy()
                joints_3d[4:7] = joints_3d_copy[1:4]
                joints_3d[1:4] = joints_3d_copy[4:7]
            if self.place_person_in_center: # it is for debug purpose
                if self.use_h36m_cameras_on_cmu:
                    raise ValueError('Not implemented')
                joints_3d[:, 0] = joints_3d[:, 0] - joints_3d[0, 0]
                joints_3d[:, 2] = joints_3d[:, 2] - joints_3d[0, 2]
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.output_in_meter:
                joints_3d = joints_3d / 100
            if self.use_h36m_cameras_on_cmu:
                # swap y and z
                joints_3d_copy = joints_3d.copy()
                joints_3d[:, 1] = joints_3d[:, 2]   # swap y and z
                joints_3d[:, 2] = -joints_3d_copy[:, 1]  # swap y and z
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.use_cmu_cameras_on_cmu:
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
        elif db_rec['source'] == 'rich' or db_rec['source'] == 'openmplposer':
            joints_3d = db_rec['joints_3d'].copy()      # (17, 3) in world coordinate
        elif db_rec['source'] == 'h36m':
            # Attention: the world joints of h36m are defected --> use camera coords and transform them to world
            joints_3d_camera = db_rec['joints_3d_camera'].copy()      # (17, 3) in camera coordinate
            joints_3d = cam_to_world(joints_3d_camera[None], camera['R'], camera['t']).squeeze()  # (17, 3) in world coordinate
            if self.flip_lower_body_kp:
                joints_3d_copy = joints_3d.copy()
                joints_3d[4:7] = joints_3d_copy[1:4]
                joints_3d[1:4] = joints_3d_copy[4:7]
            # bring the center of the world coordinate to the center of the room
            if self.place_person_in_center: # it is for debug purpose
                joints_3d[:, 0] = joints_3d[:, 0] - joints_3d[0, 0]
                joints_3d[:, 1] = joints_3d[:, 1] - joints_3d[0, 1]
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.output_in_meter:
                joints_3d = joints_3d / 1000
                
            if self.use_cmu_cameras_on_h36m:
                joints_3d_copy = joints_3d.copy()
                joints_3d[:, 1] = -joints_3d[:, 2]   # swap y and z
                joints_3d[:, 2] = joints_3d_copy[:, 1]  # swap y and z
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            # joints_3d[:, 0] = joints_3d[:, 0] - self.room_center[0]
            # joints_3d[:, 2] = joints_3d[:, 2] - self.room_center[2]
            
        
        # compute the person's height
        persons_height = np.max(joints[:, 1]) - np.min(joints[:, 1])
        
                    
        if self.output_in_meter:
            if db_rec['source'] == 'cmu_panoptic':
                camera['T'] = camera['T'] / 100
                camera['t'] = camera['t'] / 100
                if shift_room_tri is not None:
                    shift_room_tri = shift_room_tri / 100
            elif db_rec['source'] == 'rich' or db_rec['source'] == 'openmplposer':
                pass
            else:
                camera['T'] = camera['T'] / 1000
                camera['t'] = camera['t'] / 1000
                if shift_room_tri is not None:
                    shift_room_tri = shift_room_tri / 1000

        if self.add_noise_to_camera_calib:
            camera['R'], camera['T'] = self.add_noise_to_rt(camera['R'], camera['T'], rot_noise_deg=self.noise_rot_deg, trans_noise_std=self.noise_trans_std)
            camera['t'] = -camera['R'] @ camera['T']
                
        if self.shift_room:
            if type(self.shift_room_value) == str:
                if self.shift_room_value in ['to_avg_pose', 'to_conf_kp']:
                    joints_3d = joints_3d + shift_room_tri
                    camera['T'] = camera['T'] + shift_room_tri.reshape(3, 1)
                    camera['t'] = -camera['R'] @ camera['T']
            else:
                joints_3d = joints_3d + self.shift_room_value
                camera['T'] = camera['T'] + self.shift_room_value.reshape(3, 1)
                camera['t'] = -camera['R'] @ camera['T']
            # camera['t'] = camera['t'] + self.shift_room_value.reshape(3, 1)
        
        if self.normalize_room and self.normalize_room_firstly:
            room_x_scale = self.room_max_x - self.room_min_x
            room_y_scale = self.room_max_y - self.room_min_y
            room_z_scale = self.room_max_z - self.room_min_z
            if db_rec['source'] == 'cmu_panoptic':
                room_center = np.array([(self.room_max_x + self.room_min_x) / 2, -(self.room_max_z + self.room_min_z) / 2, (self.room_max_y + self.room_min_y) / 2])
            elif db_rec['source'] == 'rich':
                room_center = np.array([(self.room_max_x + self.room_min_x) / 2, -(self.room_max_z + self.room_min_z) / 2, (self.room_max_y + self.room_min_y) / 2])
            else:
                room_center = np.array([(self.room_max_x + self.room_min_x) / 2, (self.room_max_y + self.room_min_y) / 2, (self.room_max_z + self.room_min_z) / 2])
                
            joints_3d = self.normalize_pose3d_coordinates(joints_3d, room_center, room_x_scale, room_y_scale, room_z_scale)
            camera['T'] = self.normalize_pose3d_coordinates(camera['T'], room_center, room_x_scale, room_y_scale, room_z_scale)
            camera['t'] = self.normalize_pose3d_coordinates(camera['t'], room_center, room_x_scale, room_y_scale, room_z_scale)
            
        joints_org = joints.copy()
        noise_vis = np.zeros((17, 2))
        noise_penalize_conf = np.ones((17, 1))
        # ========================== 2D joints scalar ====================================
        if self.APPLY_NOISE:
            noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
            noise_vis = noise.copy()
            joints = joints + noise
            # penalize the confidence of the noisy joints
            # penalize_conf = np.exp(-np.abs(noise.sum(axis=1)) / 2)
            if self.penalize_confidence == 'exp_error':
                a = self.penalize_confidence_a
                b = self.penalize_confidence_b
                noise_value = np.sqrt((noise ** 2).sum(axis=1))
                penalize_conf = a * np.exp(- b * noise_value)
            elif self.penalize_confidence == 'linear':
                a = self.penalize_confidence_a
                b = self.penalize_confidence_b
                noise_value = np.sqrt((noise ** 2).sum(axis=1))
                penalize_conf = a * noise_value + b
            elif self.penalize_confidence == 'exp_sqrt':
                penalize_conf = np.exp(-np.sqrt((noise ** 2).sum(axis=1)) / 2)
            else:
                penalize_conf = np.ones((17,))
            noise_penalize_conf = penalize_conf.copy()
            joints_vis = joints_vis * penalize_conf[:, None]
            
        if self.inputs_normalized and self.normalize_cameras:
            cam_center_image = np.array([camera['cx'], camera['cy']])      # (2, )     (cx, cy)  in original image scale
            cam_center_image = self.normalize_screen_coordinates(cam_center_image, image_size[0], image_size[1])
            camera['cx'] = cam_center_image[0]
            camera['cy'] = cam_center_image[1]
            focal_length = np.array([camera['fx'], camera['fy']])      # (2, )     (fx, fy)  in original image scale
            focal_length = focal_length / image_size[0] * 2
            camera['fx'] = focal_length[0]
            camera['fy'] = focal_length[1]
            
            R = camera['R'].copy()
            K = np.array([
                [focal_length[0], 0, cam_center_image[0]],
                [0, focal_length[1], cam_center_image[1]],
                [0, 0, 1.],
            ])
        else:
            R = camera['R'].copy()
            K = np.array([
                [float(camera['fx']), 0, float(camera['cx'])],
                [0, float(camera['fy']), float(camera['cy'])],
                [0, 0, 1.],
            ])
        if self.use_t:
            T = camera['t'].copy()
        else:
            T = camera['T'].copy()
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()
        if self.use_t:
            cam_center = torch.Tensor(T.T)    # Tensor, (1, 3) camera center in world coordinate
        else:
            cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate
        # cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate

        # fix the system error of camera
        # distCoeffs = np.array(
        #     [float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        # data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
        # joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        # center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()

        # ==================================== Preprocess ====================================
        # augmentation factor
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
        if db_rec['source'] == 'cmu_panoptic' and image_size[0] == 256:
            scale = scale * 4.0     # the images are hd, so we need to scale them down
            
        # affine transformation matrix
        trans = get_affine_transform(center, scale, rotation, image_size)              # (2, 3)
        trans_inv = get_affine_transform(center, scale, rotation, image_size, inv=1)   # (2, 3)
        
        if self.no_augmentation:
            trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            trans_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
        KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

        # image_size = self.image_size
        # if db_rec['source'] == 'rich' and camera['cx'] < camera['cy']:
        #     image_size = (image_size[1], image_size[0])
        if self.clip_joints:
            if not self.no_augmentation:
                for i in range(self.num_joints):
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= image_size[0] or
                            joints[i, 1] >= image_size[1]):
                        joints_vis[i, :] = 0
            # penalize confidence if kp goes out of image
            joints_vis[:, 0] = np.where(0 < joints[:, 0], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 0] < image_size[0] - 1, joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(0 < joints[:, 1], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 1] < image_size[1] - 1, joints_vis[:, 0], 0)
            joints[:, 0] = np.clip(joints[:, 0], 0, image_size[0] - 1)
            joints[:, 1] = np.clip(joints[:, 1], 0, image_size[1] - 1)
        else:
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    if not self.no_augmentation:
                        joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= image_size[0] or
                            joints[i, 1] >= image_size[1]):
                        joints_vis[i, :] = 0
                        joints[i, :] = 0        # to avoid further errors when downscaling
                else:
                    joints[i, :] = 0            # to avoid further errors when downscaling

        # # ========================== heatmap ==========================
        # target, target_weight = self.generate_target(joints, joints_vis)

        # target = torch.from_numpy(target)                   # (17, 64, 64) heatmap
        # target_weight = torch.from_numpy(target_weight)
        
        if self.APPLY_NOISE_MISSING and self.is_train:
            missing = np.random.uniform(0, 1, joints_vis.shape[0])
            mask = np.ones_like(joints_vis)
            mask[missing < self.MISSING_LEVEL] = 0
            joints_vis = joints_vis * mask
            joints = joints * mask[:, 0:1]
        

        if db_rec['source'] == 'cmu_panoptic':
            joints_3d_conf = db_rec['joints_3d_conf'].copy()
            if self.flip_lower_body_kp:
                joints_3d_conf_copy = joints_3d_conf.copy()
                joints_3d_conf[4:7] = joints_3d_conf_copy[1:4]
                joints_3d_conf[1:4] = joints_3d_conf_copy[4:7]
        else:
            joints_3d_conf = np.ones_like(joints_3d[:, 0])
        joints_3d_conf[joints_3d_conf < 0] = 0
        
        # joints_3d = torch.from_numpy(joints_3d).float()
        joints_3d_conf = torch.from_numpy(joints_3d_conf).float()
        
        # ========================== 3D ray vectors ====================================
        # (256/down * 256/down, 3)
        if self.inputs_normalized:
            joints = self.normalize_screen_coordinates(joints, image_size[0], image_size[1])
            joints_org = self.normalize_screen_coordinates(joints_org, image_size[0], image_size[1])
            noise_vis = self.normalize_screen_coordinates(noise_vis, image_size[0], image_size[1])
            
        joints_ds = joints / self.downsample
        coords_ray = self.create_3d_ray_coords(camera, trans_inv, joints_ds=joints_ds, concat_cam_center=self.concat_cam_centers_to_rays, concat_cam_axis=self.concat_cam_axis_to_rays)
        # coords_ray = coords_ray.reshape(int(np.sqrt(coords_ray.shape[0])), int(np.sqrt(coords_ray.shape[0])), 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)

        # generate direction vectors and intersection points with x=0, y=0, z=0 planes
        direction_vectors, intersection_points = self.generate_direction_vectors_and_intersection_points(coords_ray, cam_center)
        # if self.use_mmpose:
        #     print('Using mmpose for direction vectors and intersection points')
        direction_vectors = direction_vectors.astype(np.float32)
        intersection_points = intersection_points.astype(np.float32)
        
        # ========================== Depth ====================================
        depth_vals = None
        if self.use_depth:
            if self.use_depth_with_noise_on_distance or self.use_gt_depth:
                distance_ = np.sum((joints_3d - camera['T'].copy().reshape(1, 3)) ** 2, axis=1) ** 0.5
                if not self.use_gt_depth:
                    distance_ = distance_ + np.random.normal(self.mu_depth_noise, self.sigma_depth_noise, distance_.shape)
                depth_vals = distance_.reshape(-1, 1)
            else:
                depth_vals = np.array(db_rec['depth']).reshape(-1, 1).copy()
            
        image = db_rec['image']
        if db_rec['source'] == 'openmplposer':
            protocol = db_rec['image'].split('/')[-4]
            dataset = db_rec['image'].split('/')[-3]
            video = db_rec['image'].split('/')[-1].replace('.pkl', '')
            frame = db_rec['frame_id']
            image = f'{protocol}_{dataset}_{video}_{frame}'
            
        # ==========================  Meta Info ==========================
        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],   # (17, 2) in origin image (1000, 1000)
            'joints_2d_transformed': joints,    # (17, 2) in input image (256, 256)
            'joints_vis': joints_vis,
            'joints_3d_conf': joints_3d_conf, # (17, 1) confidence of 3d GT joints
            'source': db_rec['source'],
            
            'cam_center': cam_center,   # (1, 3) in world coordinate
            'rays': coords_ray,         # (256/down * 256/down, 3)  in world coordinate
            'KRT': KRT,                 # (3, 4) for augmented image
            'K': K,
            'RT': Rt,

            'img-path': image_file,                      # str
            'subject_id': db_rec['subject'],             # string 11
            'cam_id': db_rec['camera_id'],                # string 01
            'image': image,  
            'joints_2d_org': joints_org,     # (17, 2) in origin image (1000, 1000) without added noise
            'noise_vis': noise_vis,
            # 'noise_penalize_conf': noise_penalize_conf,      
            
            # 'direction_vectors': direction_vectors,   # (17, 3) in world coordinate
            # 'intersection_points': intersection_points,   # (17, 3) in world coordinate     
            # 'direction_angles': direction_angles,   # (17, 2) in world coordinate
            # 'persons_height': persons_height,
            
        }
        K_ = np.array([K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]])
        return direction_vectors, intersection_points, joints_vis, joints_3d, meta, joints, K_, Rt, depth_vals

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight
    
    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return (X/w)*2 - [1, h/w]
    
    def normalize_pose3d_coordinates(self, X, center, x_width, y_width, z_width): 
        assert X.shape[-1] == 3 or X.shape[0] == 3
        w = max(x_width, y_width, z_width)
        if X.shape[-1] == 3:
            return (X - center) / w
        elif X.shape[0] == 3 and len(X.shape) == 2:
            return (X - center.reshape(3, 1)) / w
        else:
            assert 'Wrong input shape'
            
    def generate_combinations(self, input_list, n):
        return list(combinations(input_list, n))
        

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1     # 13
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def create_3d_ray_coords(self, camera, trans_inv, joints_ds, concat_cam_center=False, concat_cam_axis=False):
        multiplier = 1.0                        # avoid numerical instability
        if self.downsample != 1 and self.use_grid:
            grid = self.grid.clone()                # Tensor,   (hw, 2), val in 0-256
            grid = grid.reshape(self.image_size[1] // self.downsample, self.image_size[0] // self.downsample, 2)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 2)
            # transform to original image R.T.dot(x.T) + T
            coords = affine_transform_pts(grid.numpy(), trans_inv)  # array, size: (hw, 2), val: 0-1000
        else:
            coords = joints_ds.copy()

        if np.isscalar(camera['fx']):
            coords[:, 0] = (coords[:, 0] - camera['cx']) / camera['fx'] * multiplier
            coords[:, 1] = (coords[:, 1] - camera['cy']) / camera['fy'] * multiplier
        elif camera['fx'].shape == ():
            coords[:, 0] = (coords[:, 0] - camera['cx']) / camera['fx'] * multiplier
            coords[:, 1] = (coords[:, 1] - camera['cy']) / camera['fy'] * multiplier
        else:
            coords[:, 0] = (coords[:, 0] - camera['cx'][0]) / camera['fx'][0] * multiplier      # array
            coords[:, 1] = (coords[:, 1] - camera['cy'][0]) / camera['fy'][0] * multiplier
        

        # (hw, 3) 3D points in cam coord
        coords_cam = np.concatenate((coords,
                                     multiplier * np.ones((coords.shape[0], 1))), axis=1)   # array
        
        if concat_cam_center:
            coords_cam = np.concatenate((coords_cam, np.zeros((1, 3))), axis=0)
        elif concat_cam_axis:
            coords_cam = np.concatenate((coords_cam, np.array([[0, 0, 1]])), axis=0)

        if self.use_t:
            coords_world = (camera['R'].T @ coords_cam.T + camera['t']).T  # (hw, 3)    in world coordinate    array
        else:
            coords_world = (camera['R'].T @ coords_cam.T + camera['T']).T  # (hw, 3)    in world coordinate    array
        coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
        if self.bug_test:
            coords_world[:, :] = 0
            
        # if self.apply_sine_encoding_on_rays:
        #     coords_world = self.assign_positional_encoding_to_rays(coords_world, self.sine_d_model)
        return coords_world
    
    def compute_sine_cosine_encoding(self, coord, d_model):
        """
        Compute sine-cosine positional encoding for a single coordinate.

        Args:
        coord (float): The coordinate value (x, y, or z).
        d_model (int): The dimension of the model.

        Returns:
        numpy.ndarray: A positional encoding vector of size d_model.
        """
        encoding = np.zeros(d_model)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        encoding[0::2] = np.sin(coord * div_term)
        encoding[1::2] = np.cos(coord * div_term)
        return encoding

    def assign_positional_encoding_to_rays(self, rays, d_model):
        """
        Assign positional encodings to 3D rays without shifting coordinates.

        Args:
        rays (numpy.ndarray): The 3D rays of shape (num_rays, 3), where each ray is represented by (x, y, z) coordinates.
        d_model (int): The dimension of the model.

        Returns:
        numpy.ndarray: The rays with positional encodings of shape (num_rays, d_model).
        """
        num_rays = rays.shape[0]
        encoded_rays = np.zeros((num_rays, d_model))
        
        for i, ray in enumerate(rays):
            x, y, z = ray
            x_encoding = self.compute_sine_cosine_encoding(x, d_model // 3)
            y_encoding = self.compute_sine_cosine_encoding(y, d_model // 3)
            z_encoding = self.compute_sine_cosine_encoding(z, d_model // 3)
            encoded_rays[i] = np.concatenate((x_encoding, y_encoding, z_encoding))
        
        return encoded_rays
    
    def generate_direction_vectors_and_intersection_points(self, coords_world, cam_center):
        coords_world = np.array(coords_world)
        cam_center = np.array(cam_center)
        
        # repeat cam_center with the same number of coords_world
        cam_center = np.repeat(cam_center, len(coords_world), axis=0)
        
        if self.intersection_ray_with == 'Ground':
            direction_vectors, intersections_x, intersections_y, intersections_z, closest_intersections, intersections_donut, intersections_given_z = self.line_properties_batch(coords_world, cam_center, given_z=self.ground_z)
            intersections = intersections_given_z
        else:
            direction_vectors, intersections_x, intersections_y, intersections_z, closest_intersections, intersections_donut = self.line_properties_batch(coords_world, cam_center)
            if self.intersection_ray_with in ['Closest', 'closest']:
                intersections = closest_intersections
            elif self.intersection_ray_with in ['x', 'X']:
                intersections = intersections_x
            elif self.intersection_ray_with in ['y', 'Y']:
                intersections = intersections_y
            elif self.intersection_ray_with in ['z', 'Z']:
                intersections = intersections_z
            elif self.intersection_ray_with in ['all', 'All']:
                intersections = np.concatenate((intersections_x, intersections_y, intersections_z), axis=1)
            elif self.intersection_ray_with in ['Cam', 'cam', 'Camera', 'camera', 'M']:
                intersections = cam_center.copy()
            else:
                raise ValueError('Invalid intersection_ray_with value. Must be one of "closest", "x", "y", or "z".')
        
        if self.ray_as_intersection_with_donut:
            intersections = intersections_donut
        
        direction_vectors = np.array(direction_vectors)
        intersections = np.array(intersections)
        
        return direction_vectors, intersections
    
    def line_properties_batch(self, points1, points2, given_z=None):
        """ Calculate direction vectors, intersection points with x = 0, y = 0, z = 0, and 
            closest intersection points to (0,0,0) for a batch of line segments.

        Args:
            points1: (batch_size, 3) array of starting points
            points2: (batch_size, 3) array of ending points

        Returns:
            _type_: _description_
        """
        # Ensure the inputs are numpy arrays for vectorized operations
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        # Calculate direction vectors
        direction_vectors = points2 - points1
        
        # Initialize lists to hold intersection points
        intersections_x = []
        intersections_y = []
        intersections_z = []
        intersections_given_z = []
        closest_intersections = []
        
        for i in range(len(points1)):
            x1, y1, z1 = points1[i]
            x2, y2, z2 = points2[i]
            
            # Calculate intersection with x = 0
            if x2 != x1:
                t_x = -x1 / (x2 - x1)
                y_x = y1 + t_x * (y2 - y1)
                z_x = z1 + t_x * (z2 - z1)
                intersections_x.append((0, y_x, z_x))
            else:
                intersections_x.append((0, 100_000, 100_000)) # a large number to avoid numerical instability
            
            # Calculate intersection with y = 0
            if y2 != y1:
                t_y = -y1 / (y2 - y1)
                x_y = x1 + t_y * (x2 - x1)
                z_y = z1 + t_y * (z2 - z1)
                intersections_y.append((x_y, 0, z_y))
            else:
                intersections_y.append((100_000, 0, 100_000)) # a large number to avoid numerical instability
            
            # Calculate intersection with z = 0
            if z2 != z1:
                t_z = -z1 / (z2 - z1)
                x_z = x1 + t_z * (x2 - x1)
                y_z = y1 + t_z * (y2 - y1)
                intersections_z.append((x_z, y_z, 0))
            else:
                intersections_z.append((100_000, 100_000, 0)) # a large number to avoid numerical instability
                
            # Intersection with user-defined z
            if given_z is not None and z2 != z1:
                t_given_z = (given_z - z1) / (z2 - z1)
                x_given_z = x1 + t_given_z * (x2 - x1)
                y_given_z = y1 + t_given_z * (y2 - y1)
                intersections_given_z.append((x_given_z, y_given_z, given_z))
            else:
                intersections_given_z.append((100_000, 100_000, given_z))  # Large value for stability

            
            # Determine closest intersection to (0,0,0)
            distances = []
            if intersections_x[-1] is not None:
                distances.append((np.linalg.norm(intersections_x[-1]), intersections_x[-1]))
            if intersections_y[-1] is not None:
                distances.append((np.linalg.norm(intersections_y[-1]), intersections_y[-1]))
            if intersections_z[-1] is not None:
                distances.append((np.linalg.norm(intersections_z[-1]), intersections_z[-1]))
            
            if distances:
                closest_intersections.append(min(distances)[1])
            else:
                closest_intersections.append(None)
        
        if self.ray_as_intersection_with_donut:
            # find intersections with spheres of radius 0.5 and 1 centered at (0,0,0)
            lines = np.stack([points1, points2], axis=1)
            intersections_s10 = self.batch_line_sphere_intersection(lines, np.array([0, 0, 0]), 10)
            intersections_s20 = self.batch_line_sphere_intersection(lines, np.array([0, 0, 0]), 20)
            intersections_donut = np.concatenate((intersections_s10, intersections_s20), axis=1)
        else:
            intersections_donut = None
            
        if given_z is not None:
            return direction_vectors, intersections_x, intersections_y, intersections_z, closest_intersections, intersections_donut, intersections_given_z
        
        return direction_vectors, intersections_x, intersections_y, intersections_z, closest_intersections, intersections_donut
    
    def batch_line_sphere_intersection(self, lines, center, radius):
        """
        Find the intersection points of multiple infinite lines and a sphere.
        
        :param lines: Array of shape (N, 2, 3) where N is the number of lines,
                    each line is defined by two 3D points
        :param center: Center of the sphere (numpy array of shape (3,))
        :param radius: Radius of the sphere
        :return: List of lists, each inner list contains the intersection points for a line
        """
        # Ensure inputs are numpy arrays
        lines = np.asarray(lines)
        center = np.asarray(center)
        
        # Calculate direction vectors for all lines
        directions = lines[:, 1] - lines[:, 0]
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        # Vector from sphere center to first point of each line
        f = lines[:, 0] - center
        
        # Calculate quadratic equation coefficients for all lines
        a = np.sum(directions**2, axis=1)
        b = 2 * np.sum(f * directions, axis=1)
        c = np.sum(f**2, axis=1) - radius**2
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        # Initialize list to store intersections
        all_intersections = []
        
        for i in range(len(lines)):
            if discriminant[i] < 0:
                all_intersections.append([])  # No intersection
            else:
                t1 = (-b[i] + np.sqrt(discriminant[i])) / (2*a[i])
                t2 = (-b[i] - np.sqrt(discriminant[i])) / (2*a[i])
                
                intersection1 = lines[i, 0] + t1 * directions[i]
                intersection2 = lines[i, 0] + t2 * directions[i]
                
                # choose the intersection with z > 0
                if intersection1[2] > 0:
                    all_intersections.append(intersection1)
                elif intersection2[2] > 0:
                    all_intersections.append(intersection2)
                else:
                    all_intersections.append([])
                
        all_intersections = np.array(all_intersections)
        return all_intersections
    
    # Function to convert a normal vector to spherical coordinates (angles)
    def vector_to_angles(self, vectors):
        # Extract vector components

        # Compute azimuth (phi)
        azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Compute elevation (theta)
        elevation = np.arctan2(np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2), vectors[:, 2])

        # # Convert from radians to degrees
        # azimuth_deg = np.degrees(azimuth)
        # elevation_deg = np.degrees(elevation)

        return np.stack([azimuth, elevation], axis=1)
        # return azimuth_deg, elevation_deg
        
    def create_master_cameras(self, 
                                n_cameras=5, 
                                room_center=np.array([0, 0, 0]),
                                room_min_x=None,
                                room_max_x=None,
                                room_min_y=None,
                                room_max_y=None,
                                room_min_z=None,
                                room_max_z=None,
                                flip_yz=False,
    ):
        # Create a list of cameras
        cameras = []
        # create a circle of cameras around the room
        angles = np.linspace(0, 2*np.pi, n_cameras, endpoint=False)
        radius = max(room_max_y - room_min_y, room_max_x - room_min_x) / 2 + 2
        z = 2
        for angle in angles:
            # Create a camera at x and y given by the angle with z = 2 and radius = max( [room_max_x - room_min_x] / 2, [room_max_y - room_min_y] / 2)
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            T = np.array([x, y, z])
            forward_ = room_center - T
            forward_ = forward_ / np.linalg.norm(forward_)
            
            # Define an up vector (we assume Y-axis as up)
            up_ = np.array([0, 0, -1])

            # Calculate the right vector (cross product of up and forward)
            right_ = np.cross(up_, forward_)
            right_ = right_ / np.linalg.norm(right_)
            
            up_ = np.cross(forward_, right_)
            
            R = np.array([right_, up_, forward_])
            T = T.reshape(3, 1)
            t = - R @ T
            
            if flip_yz:
                R_ = R.copy()
                R[1] = -R_[2]
                R[2] = R_[1]
                T_ = T.copy()
                T[1] = -T_[2]
                T[2] = T_[1]
            # fx and fy and cx and cy are the same for all cameras (taken from cmu hd cams 14)
            cameras.append({
                'R': R,
                'T': T,
                't': t,
                'cx': 940.0,
                'cy': 565.0,
                'fx': 1650.0,
                'fy': 1650.0,
            })
            cameras[-1]['K'] = np.array([
                [float(cameras[-1]['fx']), 0, float(cameras[-1]['cx'])],
                [0, float(cameras[-1]['fy']), float(cameras[-1]['cy'])],
                [0, 0, 1.],
            ])
            
        return cameras
    
    def random_camera_in_room(self, camera_location_limit, room_size, camera_location_outside_room=True, camera_dist_from_person=0, person_location=None, room_center=None):
        
        # Center of the room
        if room_center is None:
            center = np.array([np.random.uniform(-0.5, 0.5), 
                            np.random.uniform(-0.5, 0.5), 
                            np.random.uniform(0, 1)])
        else:
            center = np.array([np.random.uniform(room_center[0]-0.5, room_center[0]+0.5), 
                            np.random.uniform(room_center[1]-0.5, room_center[1]+0.5), 
                            np.random.uniform(room_center[2], room_center[2]+1)])
        
        person_location = np.array([0, 0, 0]) if person_location is None else np.array(person_location)
        
        # Random camera position (T) inside the room
        # Ensure it's not placed at the exact center
        if camera_dist_from_person > 0:
            distance = 0
            while distance < camera_dist_from_person:
                T = np.array([np.random.uniform(camera_location_limit[0], camera_location_limit[1]), 
                            np.random.uniform(camera_location_limit[2], camera_location_limit[3]),
                            np.random.uniform(camera_location_limit[4], camera_location_limit[5])])
                distance = np.linalg.norm(T - person_location)
        elif not camera_location_outside_room:
            T = np.array([np.random.uniform(camera_location_limit[0], camera_location_limit[1]), 
                        np.random.uniform(camera_location_limit[2], camera_location_limit[3]),
                        np.random.uniform(camera_location_limit[4], camera_location_limit[5])])
        else:
            x1 = np.random.uniform(camera_location_limit[0], room_size[0])
            x2 = np.random.uniform(room_size[1], camera_location_limit[1])
            y1 = np.random.uniform(camera_location_limit[2], room_size[2])
            y2 = np.random.uniform(room_size[3], camera_location_limit[3])
            z = np.random.uniform(camera_location_limit[4], camera_location_limit[5])
            
            x_choice = np.random.randint(0, 2)
            y_choice = np.random.randint(0, 2)
            
            T = np.array([x1 if x_choice == 0 else x2, 
                        y1 if y_choice == 0 else y2,
                        z])
        
        if np.allclose(T, center):
            T[0] += 0.1
        
        # Define the forward vector (from camera to the center)
        forward = center - T
        forward /= np.linalg.norm(forward)
        
        # Define an up vector (we assume Y-axis as up)
        up = np.array([0, 0, -1])

        # Calculate the right vector (cross product of up and forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)

        # Recalculate the up vector to ensure orthogonality
        up = np.cross(forward, right)

        # Construct the rotation matrix R
        R = np.array([right, up, forward])
        T = T.reshape(3, 1)

        t = - R @ T

        return {
            'R': R,
            'T': T,
            't': t,
        }
        # return R, T, t
        
        
    def coco_to_h36m(self, coco, scores):
        h36m = np.zeros_like(coco)
        conf = np.zeros_like(scores)
        for i in range(len(coco)):
            try:
                h36m[i] = coco[COCO2H36M[i]]
                conf[i] = scores[COCO2H36M[i]]
            except KeyError:
                h36m[i] = np.zeros_like(coco[i])
                conf[i] = 0
        
        # calculate head
        head = coco[0:5].mean(axis=0)
        h36m[10] = head
        conf[10] = scores[0:5].mean(axis=0)
        
        # calculate neck
        neck = coco[3:7].mean(axis=0)
        h36m[8] = neck
        conf[8] = scores[3:7].mean(axis=0)
        
        # calculate root
        root = coco[11:13].mean(axis=0)
        h36m[0] = root
        conf[0] = scores[11:13].mean(axis=0)
        
        # calculate belly
        belly = np.mean([neck, root], axis=0)
        h36m[7] = belly
        conf[7] = np.mean([conf[8], conf[0]], axis=0)
        
        return h36m, conf
    
    def h36m_to_coco(self, h36m, scores):
        coco = np.zeros_like(h36m)
        conf = np.zeros_like(scores)
        H36M2COCO = {k: v for v, k in COCO2H36M.items()}
        for i in range(len(h36m)):
            try:
                coco[i] = h36m[H36M2COCO[i]]
                conf[i] = scores[H36M2COCO[i]]
            except KeyError:
                coco[i] = np.zeros_like(h36m[i])
                conf[i] = 0
                
        # head -> reye, leye, rear, lear
        coco[1:5] = h36m[10]
        conf[1:5] = scores[10]
        
        
        return coco, conf
            
        
    
    def add_noise_to_rt(self, R_mat, t_vec, rot_noise_deg=1.0, trans_noise_std=0.01):
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
