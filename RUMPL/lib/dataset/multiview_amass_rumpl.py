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

import os.path as osp
import numpy as np
import pickle
import json
import collections
import random
import torch
import copy
import cv2
from utils.calib import smart_pseudo_remove_weight
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts
from utils.calib import *
from utils.utils import *
from utils.utils_amass import rotate_pose
from multiviews.triangulate import triangulate_poses

from dataset.joints_dataset_rumpl import JointsDataset_RUMPL
import logging
logger = logging.getLogger(__name__)

downsample = 16

# set random seed
# np.random.seed(1)
# random.seed(1)

class MultiView_AMASS_RUMPL(JointsDataset_RUMPL):

    """
    The purpose is to train on H36M 3D data based on the camera system from CMU Panoptic.
    the 3D data is taken from h36M, and the 2D is created based on camera parameters from CMU Panoptic.
    """
    def __init__(self, cfg, image_set, is_train, transform=None, is_mmpose=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = 17
        
        if self.CMU_KEYPOINT_STANDARD == 'h36m':
            self.actual_joints = {
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
        elif self.CMU_KEYPOINT_STANDARD == 'coco':
            self.actual_joints = {
                0: 'nose',
                1: 'leye',
                2: 'reye',
                3: 'lear',
                4: 'rear',
                5: 'lsho',
                6: 'rsho',
                7: 'lelb',
                8: 'relb',
                9: 'lwri',
                10: 'rwri',
                11: 'lhip',
                12: 'rhip',
                13: 'lkne',
                14: 'rkne',
                15: 'lank',
                16: 'rank'
            }
            self.union_joints = self.actual_joints
        
        self.joints_right = [1, 2, 3, 14, 15, 16]
        self.joints_left = [4, 5, 6, 11, 12, 13]

        self.val_on_train = cfg.DATASET.VAL_ON_TRAIN
        self.is_train = is_train
        
        self.use_mmpose = False
        if is_mmpose:
            self.use_mmpose = True
        else:
            if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
                self.use_mmpose = True
            elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_TEST) and not is_train:
                self.use_mmpose = True
        
        
        self.inputs_normalized = cfg.DATASET.INPUTS_NORMALIZED
        
        self.no_augmentation = cfg.DATASET.NO_AUGMENTATION
        self.output_in_meter = cfg.DATASET.OUTPUT_IN_METER
        self.clip_joints = cfg.DATASET.CLIP_JOINTS
            
        self.camera_manual_order = cfg.DATASET.CAMERA_MANUAL_ORDER
        print('camera_manual_order',self.camera_manual_order)
        
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        self.amass_dataset_type = cfg.DATASET.AMASS_DATASET_TYPE

        # self.amass_with_openmplposer_cameras = cfg.DATASET.AMASS_WITH_OPENMPLPOSER_CAMERAS

        # if self.amass_with_openmplposer_cameras and self.amass_with_random_cameras:
        #     raise ValueError('amass_with_openmplposer_cameras and amass_with_random_cameras cannot be both True')
        
        if self.amass_with_random_cameras or self.run_on_sphere:
            self.dome_cameras = {}
            self.n_all_cameras = 0
            self.all_camera_ids = []
            self.n_camera_setups = 0

        ###### might be not needed
        # elif self.amass_with_openmplposer_cameras:
        #     openmplposer_calibs = self.openmplposer_calibs_train if is_train else self.openmplposer_calibs_val
        #     self.openmplposer_cameras = self.load_all_cameras_openmplposer(openmplposer_calibs)
        #     if self.run_on_all_cameras:
        #         pass
        #     else:
        #         self.openmplposer_cameras = {view: v for view, v in self.openmplposer_cameras.items() if view in self.views}
            
        #     self.n_all_cameras = len(self.openmplposer_cameras)    
        #     self.all_camera_ids = list(self.openmplposer_cameras.keys())
        #     self.n_camera_setups = len(self.openmplposer_cameras[self.all_camera_ids[0]])
            
        else:
            dome_calib_file = self.dome_calib_file_train if is_train else self.dome_calib_file_val
            self.dome_cameras = self.load_all_cameras(dome_calib_file)
            if self.run_on_all_cameras or (self.run_on_specific_camera_setups is not None and self.run_on_specific_camera_setups):
                pass
            else:
                
                self.dome_cameras = {view: v for view, v in self.dome_cameras.items() if view in self.views}
                
            self.n_all_cameras = len(self.dome_cameras)    
            self.all_camera_ids = list(self.dome_cameras.keys())
                
            self.n_camera_setups = len(self.dome_cameras[self.all_camera_ids[0]])
        
        if self.amass_dataset_type is not None:
            anno_file = osp.join(self.root_2, self.amass_dataset_type, 'amass_mmpose_joints_{}.pkl'.format(image_set))
        else:
            anno_file = osp.join(self.root_2, 
                                'amass_mmpose_joints_{}.pkl'.format(image_set))
        self.db, self.db_2d = self.load_amass_new(anno_file)
        
        if self.amass_with_random_cameras or self.run_on_sphere:
            assert 'camera_parameters_all' in self.db_2d.keys()
            self.dome_cameras = self.db_2d['camera_parameters_all']
            self.all_camera_ids = list(range(len(self.dome_cameras[0])))
            self.how_many_random_cameras_to_use = cfg.DATASET.TRAIN_HOW_MANY_RANDOM_CAMERAS_TO_USE if is_train else cfg.DATASET.TEST_HOW_MANY_RANDOM_CAMERAS_TO_USE
            if self.how_many_random_cameras_to_use is not None:
                self.all_camera_ids = self.all_camera_ids[:self.how_many_random_cameras_to_use] if self.how_many_random_cameras_to_use < len(self.all_camera_ids) else self.all_camera_ids
        
        if cfg.DATASET.TRAIN_N_DOWN_SAMPLE is not None and is_train:
            self.db = self.db[::cfg.DATASET.TRAIN_N_DOWN_SAMPLE]
        if cfg.DATASET.TEST_N_DOWN_SAMPLE is not None and not is_train:
            self.db = self.db[::cfg.DATASET.TEST_N_DOWN_SAMPLE]
                
        if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
            self.db = self.db[:cfg.DATASET.TRAIN_N_SAMPLES]
        if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
            self.db = self.db[:cfg.DATASET.TEST_N_SAMPLES]
        for k in self.db_2d.keys():
            if cfg.DATASET.TRAIN_N_DOWN_SAMPLE is not None and is_train:
                self.db_2d[k] = self.db_2d[k][::cfg.DATASET.TRAIN_N_DOWN_SAMPLE]
            if cfg.DATASET.TEST_N_DOWN_SAMPLE is not None and not is_train:
                self.db_2d[k] = self.db_2d[k][::cfg.DATASET.TEST_N_DOWN_SAMPLE]
            if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
                self.db_2d[k] = self.db_2d[k][:cfg.DATASET.TRAIN_N_SAMPLES]
            if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
                self.db_2d[k] = self.db_2d[k][:cfg.DATASET.TEST_N_SAMPLES]
        logger.info('=> {} load {} samples'.format(image_set, len(self.db)))

        self.u2a_mapping = super().get_mapping()
        # logger.info('=> {} num samples: {}'.format(image_set, len(self.db)))
        # super().do_mapping()
        
    def locate_person_in_the_room_val(self, db):
        if not self.output_in_meter:
            room_max_x = self.room_max_x / 100
            room_min_x = self.room_min_x / 100
            room_max_y = self.room_max_y / 100
            room_min_y = self.room_min_y / 100
        else:
            room_max_x = self.room_max_x
            room_min_x = self.room_min_x
            room_max_y = self.room_max_y
            room_min_y = self.room_min_y
        db[:, :, 0] = db[:, :, 0] - db[:, 0:1, 0]  # subtract root joint
        db[:, :, 1] = db[:, :, 1] - db[:, 0:1, 1]  # subtract root joint
        
        if self.rotate:
            rotation = np.random.rand(len(db)) * 360
            db = rotate_pose(db, rotation, axis='z')
                
        if self.no_augmentation_3d:
            location_3d = np.zeros((db.shape[0], 3))
        else:
            location_3d = np.concatenate([np.random.rand(db.shape[0], 1) * (room_max_x - room_min_x) + room_min_x, np.random.rand(db.shape[0], 1) * (room_max_y - room_min_y) + room_min_y, np.zeros((db.shape[0], 1))], axis=1)
        db += location_3d[:, None, :]
        return db
    
    def index_to_action_names(self):
        return None
    
    
    def load_all_cameras(self, calib_file):
        distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic
        with open(calib_file, 'r') as f:
            camera_data = json.load(f)
        cams = range(1, len(camera_data) + 1)
        cameras = {cam_id:[] for cam_id in cams}
        for i in cams:
            camera = camera_data[i - 1]
            cam_id = int(camera['name'].split('_')[1])
            camera_dict = {}
            camera_dict['camera_setup'] = 'dome'
            camera_dict['camera_id'] = cam_id
            camera_dict['R'] = np.array(camera['R'])
            camera_dict['t'] = np.array(camera['t']).reshape(3, 1)
            camera_dict['T'] = np.array(camera['T']).reshape(3, 1)
            camera_dict['K'] = np.array(camera['K'])
            camera_dict['fx'] = camera_dict['K'][0,0]
            camera_dict['fy'] = camera_dict['K'][1,1]
            camera_dict['cx'] = camera_dict['K'][0,2]
            camera_dict['cy'] = camera_dict['K'][1,2]
            camera_dict['k'] = np.array([distCoef[0], distCoef[1], distCoef[4]])
            camera_dict['p'] = np.array([distCoef[2], distCoef[3]])
            cameras[cam_id].append(camera_dict)
            
        return cameras

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)    
        CAM_IX = {x: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = CAM_IX[db[i]['camera_id']]
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(self.views)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)
                

        return filtered_grouping

    
    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        idx_image = idx
        if self.run_on_sphere and self.sphere_views == []:
            camera_ids = self.db[idx][1]
            idx = self.db[idx][0]   # for sphere dataset
            pose_3d = copy.deepcopy(self.db_2d['joints_3d'][idx])
        elif self.run_on_sphere and self.sphere_views != []:
            camera_ids = self.sphere_views
            pose_3d = copy.deepcopy(self.db[idx])
        else:
            pose_3d = copy.deepcopy(self.db[idx])
        camera_setup_to_use = self.db_2d['camera_setup_used'][idx]
        
        n_views = self.n_views
        if self.max_random_n_views is not None and self.is_train:
            # n_views = np.random.randint(2, self.max_random_n_views + 1)
            n_views = self.max_random_n_views
        
        if self.mix_smart_3d_amass_with_triangulated_mmpose:
            try:
                pose_3d_tri = copy.deepcopy(self.db_2d['triangulated_3d_mmpose'][idx])
            except:
                cameras_tri = []
                for camera_setup in range(self.n_camera_setups):
                    cameras_tri.append([])
                    for view in self.views_in_amass:
                        cameras_tri[-1].append(self.dome_cameras[view][camera_setup])
                joints_2d_mmpose_all = copy.deepcopy(self.db_2d['joints_2d_mmpose'][idx])
                confs_2d_mmpose_all = copy.deepcopy(self.db_2d['confs_2d_mmpose'][idx])
                triangulated_3d_mmpose = triangulate_poses(cameras_tri[camera_setup_to_use], 
                                                       joints_2d_mmpose_all, 
                                                       confs_2d_mmpose_all.squeeze(), 
                                                       conf_threshold=.85)
                pose_3d_tri = triangulated_3d_mmpose[0]
            error_triang = np.sqrt(np.sum((pose_3d[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] - pose_3d_tri[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose]) ** 2, axis=1))
            error_tring_all = np.zeros((17), dtype=bool)
            error_tring_all[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] = error_triang < self.epipolar_error_acceptance_threshold
            pose_3d[error_tring_all] = pose_3d_tri[error_tring_all]
        joints_3d_org = copy.deepcopy(pose_3d)
        if self.centeralize_root_first and self.is_train:     # only x and y
            # if not self.use_amass_old_datasets:
            #     pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
            #     pose_3d[:, 2] = pose_3d[:, 2] - pose_3d[0, 2]  # subtract root joint
            # else:
            pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
            pose_3d[:, 1] = pose_3d[:, 1] - pose_3d[0, 1]  # subtract root joint
        
        # if self.amass_data_no_axis_swap:
        if self.normalize_room:
            room_x_scale = self.room_max_x - self.room_min_x
            room_y_scale = self.room_max_y - self.room_min_y
            room_z_scale = self.room_max_z - self.room_min_z
            room_center = np.array([(self.room_max_x + self.room_min_x) / 2, (self.room_max_y + self.room_min_y) / 2, (self.room_max_z + self.room_min_z) / 2])
        
        if not self.output_in_meter:
            pose_3d = pose_3d * 100           # (17, 3)   in world coordinate. *100 to convert to cm
            # generate 2 random numbers between -100 and 100 for x and z and add them to all joints
        if self.rotate and self.is_train:
            rotation = np.random.rand(1) * 360
            # if self.amass_data_no_axis_swap:
            pose_3d = rotate_pose(pose_3d[None], rotation, axis='z')[0]
            # else:
            #     pose_3d = rotate_pose(pose_3d[None], rotation, axis='y')[0]
        if self.no_augmentation_3d or not self.is_train:
            augmentation_3d = np.zeros((1, 3))
        else:
            if self.flip:     # flipping the pose is wrong here because the face cannot be flipped (#TODO in amass preprocess)
                flipping = np.random.rand(1)
                if flipping > 0.5:
                    pose_3d_copy = pose_3d.copy()
                    pose_3d[self.joints_right] = pose_3d_copy[self.joints_left]
                    pose_3d[self.joints_left] = pose_3d_copy[self.joints_right]
                    
            
            # if self.amass_data_no_axis_swap:
            augmentation_3d = np.concatenate([np.random.rand(1, 1) * (self.room_max_x - self.room_min_x) + self.room_min_x, np.random.rand(1, 1) * (self.room_max_y - self.room_min_y) + self.room_min_y, np.zeros((1, 1))], axis=1)
            # else:    
            #     augmentation_3d = np.concatenate([np.random.rand(1, 1) * (self.room_max_x - self.room_min_x) + self.room_min_x, np.zeros((1, 1)), np.random.rand(1, 1) * (self.room_max_y - self.room_min_y) + self.room_min_y], axis=1)
            pose_3d = pose_3d + augmentation_3d
        
        if self.amass_with_random_cameras:
            camera_ids = None
            if self.min_angle_diff > 0:
                angle_diff = 0
                angle_counter = 0
                while angle_diff < self.min_angle_diff and angle_counter < 100:
                    camera_ids = np.random.choice(self.all_camera_ids, n_views, replace=False)
                    angles = [self.project_and_calculate_angle(self.db_2d['camera_parameters_all'][idx][x]['T']) for x in camera_ids]
                    angle_diff = np.max(angles) - np.min(angles)
                    angle_counter += 1
                if angle_counter == 100:
                    logger.info('could not find a good camera setup for the {}th sample'.format(idx))
            
                
            if self.min_oks > 0:
                oks = np.zeros((n_views,))
                oks_counter = 0
                while (oks < self.min_oks).any() and oks_counter < 100:
                    camera_ids = np.random.choice(self.all_camera_ids, n_views, replace=False)
                    visiblities = []
                    for camera_id in camera_ids:
                        image_size = self.image_size.copy()
                        if self.db_2d['camera_parameters_all'][idx][camera_id]['cx'] < self.db_2d['camera_parameters_all'][idx][camera_id]['cy']:
                            image_size = [image_size[1], image_size[0]]
                        vis = np.ones(self.num_joints)
                        vis[self.db_2d['joints_2d_amass'][idx][camera_id][:, 0] < 0] = 0
                        vis[self.db_2d['joints_2d_amass'][idx][camera_id][:, 0] > image_size[0]] = 0
                        vis[self.db_2d['joints_2d_amass'][idx][camera_id][:, 1] < 0] = 0
                        vis[self.db_2d['joints_2d_amass'][idx][camera_id][:, 1] > image_size[1]] = 0
                        
                        visiblities.append(vis)
                    visiblities = np.array(visiblities)
                    oks = np.array([OKS(self.db_2d['joints_2d_amass'][idx][x], self.db_2d['joints_2d_mmpose'][idx][x], visiblities[x], self.COCO_PERSON_SIGMAS) for x in range(n_views)])
                    oks_counter += 1
                if oks_counter == 100:
                    logger.info('could not find a good camera setup for the {}th sample'.format(idx))
            
            if camera_ids is None:
                if n_views == len(self.all_camera_ids):
                    camera_ids = self.all_camera_ids
                else:
                    camera_ids = np.random.choice(self.all_camera_ids, n_views, replace=False)
        elif self.run_on_sphere:
            pass   # camera_ids are already set
        elif self.run_on_specific_camera_setups is not None and self.run_on_specific_camera_setups:
            if self.pick_random_cameras_from_specific_setups:
                random_picks = [np.random.randint(0, len(self.run_on_specific_camera_setups)) for _ in range(n_views)]
                camera_ids = [self.run_on_specific_camera_setups[x][i] for i, x in enumerate(random_picks)]
            else:
                camera_ids = self.run_on_specific_camera_setups[np.random.randint(0, len(self.run_on_specific_camera_setups))]
        elif self.run_on_all_cameras:
            if self.not_train_on_test_views and self.is_train:
                self.all_camera_ids = [x for x in self.all_camera_ids if x not in self.views]
            camera_ids = np.random.choice(self.all_camera_ids, n_views, replace=False)
        else:
            camera_ids = np.array(self.views)
            
        """
            1. get rays from the 3D pose
            2. find points on the rays that are closest to the middle point (corresponding to the real 3D point)
            3. get the middle point
            4. pass the rays and the middle point to the network shape: (n_joints, n_views or 1, 3)
        """
        
        shift_room_tri = None
        if self.shift_room:
            if type(self.shift_room_value) == str:
                if self.shift_room_value in ['to_avg_pose', 'to_conf_kp']:
                    cameras_tri = [self.db_2d['camera_parameters_all'][idx][x].copy() for x in camera_ids]
                    for cam in cameras_tri:
                        if 'distCoef' not in cam.keys():
                            distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic
                            cam['distCoef'] = distCoef
                        cam['k'] = [cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]]
                        cam['p'] = [cam['distCoef'][2], cam['distCoef'][3]]
                    loc_2d = np.array([self.db_2d['camera_parameters_all'][idx][x] for x in camera_ids])
                    if self.use_mmpose:
                        loc_2d = np.array([self.db_2d['joints_2d_mmpose'][idx][x] for x in camera_ids])
                        conf_2d = np.array([self.db_2d['confs_2d_mmpose'][idx][x] for x in camera_ids])
                    else:
                        loc_2d = np.array([self.db_2d['joints_2d_amass'][idx][x] for x in camera_ids])
                        conf_2d = np.ones((len(camera_ids), self.num_joints))
                    # conf_mmpose_2d = np.array([self.db[x]['joints_2d_conf'] for x in items])
                    joints_3d_tri = triangulate_poses(cameras_tri, loc_2d, conf_2d)
                    confs = np.min(conf_2d, axis=0).reshape(-1)
                    if self.shift_room_value == 'to_avg_pose':
                        shift_room_tri = - np.average(joints_3d_tri.squeeze(), weights=confs, axis=0).reshape(-1)
                    elif self.shift_room_value == 'to_conf_kp':
                        shift_room_tri = - joints_3d_tri[0, np.argmax(confs)]
        
        directions, intersections, joints_2d_confs, joints_2ds, camera_params, depths = [], [], [], [], [], []
        for camera_id in camera_ids:
            direction, intersection, joints_2d_conf, joints_3d, joints_2d, K, Rt, depth_vals = self.get_rays(pose_3d, camera_id, joints_3d_org, camera_setup_to_use, idx=idx, shift_room_tri=shift_room_tri)
            directions.append(direction)
            intersections.append(intersection)
            joints_2d_confs.append(joints_2d_conf)
            joints_2ds.append(joints_2d)
            if self.axis_yz_swap_for_3d:
                if self.neg == 1:
                    P = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    # P = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                else:
                    raise ValueError('neg should be 1 or 2')
                # R = P @ Rt[:, :3] @ P.T
                # R = P @ Rt[:, :3]
                R = Rt[:, :3] @ P.T
                t = Rt[:, 3:]
                # T = self.db_2d['camera_parameters_all'][idx][camera_id]['T'].copy()
                # T_ = P @ T
                # t = -R @ T_
                Rt = np.concatenate([R, t], axis=1)
            camera_params.append(np.concatenate([K.flatten(), Rt.flatten()]))
            
            depths.append(depth_vals.reshape(-1, 1)) if depth_vals is not None else depths.append(None)
            
        directions = np.array(directions)           # (n_views, 17, 3)
        intersections = np.array(intersections)     # (n_views, 17, 3)
        joints_2d_confs = np.array(joints_2d_confs)     # (n_views, 17, 1)
        joints_2ds = np.array(joints_2ds)     # (n_views, 17, 2)
        camera_params = np.array(camera_params)     # (n_views, 12)
        # repeat the camera_params for each joint
        camera_params = np.repeat(camera_params[:, None, :], self.num_joints, axis=1)   # (n_views, 17, 12)
        depths = np.array(depths) if depths[0] is not None else None
        joints_2ds = np.concatenate([joints_2ds, camera_params], axis=2)   # (n_views, 17, 14)
        
        directions = np.transpose(directions, (1, 0, 2))   # (17, n_views, 3)
        intersections = np.transpose(intersections, (1, 0, 2))   # (17, n_views, 3)
        joints_2d_confs_ = np.transpose(joints_2d_confs, (1, 0, 2))   # (17, n_views, 1)
        joints_2ds = np.transpose(joints_2ds, (1, 0, 2))   # (17, n_views, 14)
        depths = np.transpose(depths, (1, 0, 2)) if depths is not None else None
        
        # do logical and on joints_2d_confs on axis 1
        joints_2d_confs = np.where(joints_2d_confs_ > self.kp_visiblity_th, 1, 0)
        joints_2d_confs = np.all(joints_2d_confs, axis=1, keepdims=True)   # (17, 1, 1)
        
        closest_points_all = []
        for direction, intersection in zip(directions, intersections):
            closest_points = closest_points_on_n_skew_lines(intersection, direction)    # (n_views, 3)
            closest_points_all.append(closest_points)
        closest_points_all = np.array(closest_points_all)   # (17, n_views, 3)
        
        middle_points = closest_points_all.mean(axis=1, keepdims=True)   # (17, 1, 3)
        middle_points *= joints_2d_confs                                 # (17, 1, 3)
        
        
        if self.axis_yz_swap_for_3d:
            middle_points = middle_points[:, :, [0, 2, 1]]
            closest_points_all = closest_points_all[:, :, [0, 2, 1]]
            joints_3d = joints_3d[:, [0, 2, 1]]
            directions = directions[:, :, [0, 2, 1]]
            intersections = intersections[:, :, [0, 2, 1]]
            
            middle_points[:, :, self.neg] = -middle_points[:, :, self.neg]
            closest_points_all[:, :, self.neg] = -closest_points_all[:, :, self.neg]
            joints_3d[:, self.neg] = -joints_3d[:, self.neg]
            directions[:, :, self.neg] = -directions[:, :, self.neg]                
            intersections[:, :, self.neg] = -intersections[:, :, self.neg]
        
        if self.normalize_room:
            if self.axis_yz_swap_for_3d:
                room_center = room_center[[0, 2, 1]]
                room_center[self.neg] = -room_center[self.neg]
            middle_points = self.normalize_pose3d_coordinates(middle_points, room_center, room_x_scale, room_y_scale, room_z_scale)
            closest_points_all = self.normalize_pose3d_coordinates(closest_points_all, room_center, room_x_scale, room_y_scale, room_z_scale)
            joints_3d = self.normalize_pose3d_coordinates(joints_3d, room_center, room_x_scale, room_y_scale, room_z_scale)
            directions = self.normalize_pose3d_coordinates(directions, room_center, room_x_scale, room_y_scale, room_z_scale)
            intersections = self.normalize_pose3d_coordinates(intersections, room_center, room_x_scale, room_y_scale, room_z_scale)
            
        joints_3d = torch.from_numpy(joints_3d).float()
        middle_points = torch.from_numpy(middle_points).float()
        closest_points_all = np.concatenate([closest_points_all, joints_2d_confs_], axis=2)  # (17, n_views, 4)
        closest_points_all = torch.from_numpy(closest_points_all).float()
        rays = np.concatenate([directions, intersections, joints_2d_confs_], axis=2)   # (17, n_views, 6)
        if self.use_depth:
            rays = np.concatenate([rays, depths], axis=2)
        joints_2ds = np.concatenate([joints_2ds, joints_2d_confs_], axis=2)
        joints_2ds = torch.from_numpy(joints_2ds).float()
        # depths = torch.from_numpy(depths).float() if depths is not None else None
        # camera_params = torch.from_numpy(camera_params).float()
        if self.zero_tokens_for_missing_joints:
            rays *= joints_2d_confs
            closest_points_all *= joints_2d_confs
            joints_2ds *= joints_2d_confs
            # camera_params *= joints_2d_confs
        rays = torch.from_numpy(rays).float()
        fname_camera_ids = '_'.join([str(x) for x in camera_ids])
        fname_camera_ids = '_{}'.format(fname_camera_ids)

        meta = {
            'image': 'amass_{}'.format(idx_image),
            'fname_camera_ids': fname_camera_ids,
        }
            
        return middle_points, closest_points_all, joints_3d, rays, meta, joints_2ds
        
    
    def get_rays(self, joints_3d, camera_id, joints_3d_org=None, camera_setup_to_use=0, idx=None, shift_room_tri=None):

        # ==================================== Label ====================================
        
        if self.amass_with_random_cameras or self.run_on_sphere:
            camera = self.db_2d['camera_parameters_all'][idx][camera_id].copy()
        else:
            camera = self.dome_cameras[camera_id][camera_setup_to_use].copy()
            
        if self.add_noise_to_camera_calib:
            camera['R'], camera['T'] = self.add_noise_to_rt(camera['R'], camera['T'], rot_noise_deg=self.noise_rot_deg, trans_noise_std=self.noise_trans_std)
            camera['t'] = -camera['R'] @ camera['T']

        image_size = self.image_size.copy()
        if camera['cx'] < camera['cy']:
            image_size = [image_size[1], image_size[0]]
            
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
            
        if self.normalize_room and self.normalize_room_firstly:
            room_x_scale = self.room_max_x - self.room_min_x
            room_y_scale = self.room_max_y - self.room_min_y
            room_z_scale = self.room_max_z - self.room_min_z
            room_center = np.array([(self.room_max_x + self.room_min_x) / 2, (self.room_max_y + self.room_min_y) / 2, (self.room_max_z + self.room_min_z) / 2])
            joints_3d = self.normalize_pose3d_coordinates(joints_3d, room_center, room_x_scale, room_y_scale, room_z_scale)
            camera['T'] = self.normalize_pose3d_coordinates(camera['T'], room_center, room_x_scale, room_y_scale, room_z_scale)
            camera['t'] = self.normalize_pose3d_coordinates(camera['t'], room_center, room_x_scale, room_y_scale, room_z_scale)
            
        if self.apply_noise_cameras and self.is_train:
            noise = np.random.normal(0, 1, camera['T'].shape) * self.t_noise_value
            camera['t'] = camera['t'] + noise
            noise = np.random.normal(0, 1, camera['R'].shape) * self.R_noise_value
            camera['R'] = camera['R'] + noise
            camera['T'] = -camera['R'].T @ camera['t'].squeeze()
            
        if self.use_amass_old_datasets or self.use_amass_new_datasets_with_old_way:    
            joints_3d_cam = world_to_cam(joints_3d[None,:,:], camera['R'], camera['t'])  # (17, 3) in camera coordinate
            joints_2d = cam_to_image(joints_3d_cam, camera['K'])[0]        # (17, 2) in original image scale (1000, 1000)
            
        else:
            if self.amass_with_random_cameras or self.run_on_sphere:
                view_finder = camera_id
            else:
                view_finder = self.views_in_amass.index(camera_id)
            if self.use_mmpose:
                joints_2d = copy.deepcopy(self.db_2d['joints_2d_mmpose'][idx][view_finder])
                if self.mix_amass_with_mmpose:
                    joints_2d[self.keypoints_to_mix] = copy.deepcopy(self.db_2d['joints_2d_amass'][idx][view_finder][self.keypoints_to_mix])
                if np.isnan(joints_2d).any():
                    joints_2d = np.zeros_like(joints_2d)
            else:
                joints_2d = copy.deepcopy(self.db_2d['joints_2d_amass'][idx][view_finder])
        
        if self.target_normalized_3d:
            joints_3d = joints_3d_org
        
        # compute the person's height
        persons_height = np.max(joints_2d[:, 1]) - np.min(joints_2d[:, 1])
        
        joints = joints_2d.copy()
        # joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        # print('joints', joints)
        joints_org = joints.copy()
        joints_vis = np.ones((17, 1))        # (17, 3)   0,0,0 or 1,1,1
        if not self.use_amass_old_datasets and self.use_mmpose and self.joints_vis_from_mmpose:
            joints_vis = copy.deepcopy(self.db_2d['confs_2d_mmpose'][idx][view_finder])
            if self.mix_amass_with_mmpose:
                joints_vis[self.keypoints_to_mix] = 1
        elif self.use_amass_new_datasets_with_old_way and self.use_mmpose:
            raise 'disable use_mmpose with use_amass_new_datasets_with_old_way'
        noise_vis = np.zeros((17, 2))
        noise_penalize_conf = np.ones((17, 1))
        if self.APPLY_NOISE and self.is_train:
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
            # joints_vis = joints_vis * np.repeat(penalize_conf[:, None], 3, axis=1)
            

        center = np.array(np.random.rand(2,) * 250 + 250).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(np.random.rand(2,) + 2).copy()        # (2, )     (s1, s2) random number between 2 and 3
        rotation = 0
        
        if self.no_augmentation:
            center = np.array([image_size[0]/2, image_size[1]/2]).copy()      # (2, )     (cx, cy)  in original image scale
            scale = np.array([1, 1]).copy()        # (2, )     (s1, s2) random number between 2 and 3
            rotation = 0

        # ==================================== Camera  ====================================
        # camera matrix
        # normalize the camera matrix too
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

        if image_size[0] == 256:
            scale = scale * 4.0     # the images are hd, so we need to scale them down
        # scale = scale * 2.0     # the images are hd, so we need to scale them down
            
        # affine transformation matrix
        trans = get_affine_transform(center, scale, rotation, image_size)              # (2, 3)
        trans_inv = get_affine_transform(center, scale, rotation, image_size, inv=1)   # (2, 3)
        
        if self.no_augmentation:
            trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            trans_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        

        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
        # cropK = K
        KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

        if self.clip_joints:
            if not self.no_augmentation:
                for i in range(self.num_joints):
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= image_size[0] or
                            joints[i, 1] >= image_size[1]):
                        joints_vis[i, :] = 0
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
            
        if (self.APPLY_NOISE_MISSING and self.is_train) or (self.APPLY_NOISE_MISSING_TEST and not self.is_train):
            missing = np.random.uniform(0, 1, joints_vis.shape[0])
            mask = np.ones_like(joints_vis)
            mask[missing < self.MISSING_LEVEL] = 0
            joints_vis = joints_vis * mask
            joints = joints * mask[:, 0:1]

        # ========================== 3D ray vectors ====================================
        # (256/down * 256/down, 3)
        if self.inputs_normalized:
            joints = self.normalize_screen_coordinates(joints, image_size[0], image_size[1])
            joints_org = self.normalize_screen_coordinates(joints_org, image_size[0], image_size[1])
            noise_vis = self.normalize_screen_coordinates(noise_vis, image_size[0], image_size[1])
            
        joints_ds = joints / self.downsample
        coords_ray = self.create_3d_ray_coords(camera, trans_inv, joints_ds, concat_cam_center=self.concat_cam_centers_to_rays, concat_cam_axis=self.concat_cam_axis_to_rays)
        # coords_ray = coords_ray.reshape(int(np.sqrt(coords_ray.shape[0])), int(np.sqrt(coords_ray.shape[0])), 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)
        # coords_ray = coords_ray.reshape(image_size[0] // self.downsample, image_size[1] // self.downsample, 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)
        
        # generate direction vectors and intersection points with x=0, y=0, z=0 planes
        direction_vectors, intersection_points = self.generate_direction_vectors_and_intersection_points(coords_ray, cam_center)
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
                depth_maps = self.db_2d['body_depth_all_mmpose'][idx][view_finder].copy() if self.use_mmpose else self.db_2d['body_depth_all_gt'][idx][view_finder].copy()
                # depth_maps = self.db_2d['body_depth_all_mmpose'][idx].copy() if self.use_mmpose else self.db_2d['body_depth_all_gt'][idx].copy()
                depth_vals = depth_maps[:, 2, 2]
                if not self.output_in_meter:
                    raise NotImplementedError('output_in_meter should be True')
            
        
        K_ = np.array([K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]])
        return direction_vectors, intersection_points, joints_vis, joints_3d, joints, K_, Rt, depth_vals
    
    
    
    def project_and_calculate_angle(self, point):
        # The input point is assumed to be a 3D point (x, y, z)
        x, y, z = point
        
        # Calculate the angle between the projected point and the x-axis
        # The angle is given by atan2(y, x), which computes the angle in radians
        angle_radians = np.arctan2(y, x)
        
        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees
    

    def __len__(self):
        return len(self.db)

    def get_key_str(self, datum):
        return 's_{:02}_pose_{}_imgid_{:08}'.format(
            datum['subject'], datum['pose_id'],
            datum['image_id'])
        
    def get_key_str_h36m(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for item in self.db:
            gt.append(item[su, :])       # (17, 3) in original scale
        gt = np.array(gt) * 100         # (num_sample, 17, 3)  *100 to convert to cm
        gt = gt - gt[:, 0:1]
        pred = pred - pred[:, 0:1]
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