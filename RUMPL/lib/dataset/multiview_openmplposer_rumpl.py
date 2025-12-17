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
import collections
import random
import logging
import copy
import torch
from utils.calib import *

from dataset.joints_dataset_rumpl import JointsDataset_RUMPL
from multiviews.triangulate import triangulate_poses

logger = logging.getLogger(__name__)

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiView_OpenMPLPoser_RUMPL(JointsDataset_RUMPL):

    def __init__(self, cfg, image_set, is_train, transform=None, is_mmpose=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = 17
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
        self.val_on_train = cfg.DATASET.VAL_ON_TRAIN
        self.is_train = is_train
            
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        self.use_mmpose = False
        if is_mmpose:
            self.use_mmpose = True
        else:
            if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
                self.use_mmpose = True
            elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_TEST) and not is_train:
                self.use_mmpose = True
        
        self.openmplposer_dataset_name = cfg.DATASET.TRAIN_OPENMPLPOSER_DATASET_NAME if is_train else cfg.DATASET.TEST_OPENMPLPOSER_DATASET_NAME
    
        dataset_folder_name = 'datasets_mmpose' if self.use_mmpose else 'datasets' 
        dataset_folder_name = 'datasets_mmpose_depth' if self.use_depth else dataset_folder_name
        dataset_folder_name_2 = self.openmplposer_dataset_name + '_' + self.mmpose_type if self.use_mmpose else self.openmplposer_dataset_name
        dataset_folder_name_2 = dataset_folder_name_2 + '_' + self.depth_type if self.use_depth else dataset_folder_name_2
        dataset_path = osp.join(self.root, 'MPL_data', dataset_folder_name, dataset_folder_name_2)
        # dataset_folder_name_2 = dataset_folder_name_2.replace('annot', 'annot_mmpose') if self.use_mmpose else dataset_folder_name_2
        if self.val_on_train:
            anno_file = osp.join(dataset_path, 'openmplposer_train.pkl')
            
        elif cfg.DATASET.CROP:
            anno_file = osp.join(dataset_path, 'openmplposer_{}.pkl'.format(image_set))
        else:
            raise ValueError('Not implemented yet')

        self.db = self.load_db(anno_file)
        if not self.run_on_all_cameras:
            self.db = self.filter_db(self.db)   # remove images that are with other cameras

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
            self.grouping = self.grouping[:cfg.DATASET.TRAIN_N_SAMPLES]
        if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
            self.grouping = self.grouping[:cfg.DATASET.TEST_N_SAMPLES]
        self.group_size = len(self.grouping)
        logger.info('=> {} num samples: {}'.format(image_set, self.group_size))
        
    def add_noise_to_camera(self, camera):
        # add noise to camera params
        camera['R'] = camera['R'] + np.random.normal(0, 1.0, camera['R'].shape) * self.R_noise_value
        camera['T'] = camera['T'] + np.random.normal(0, 1.0, camera['T'].shape) * self.T_noise_value
        return camera
        
    def add_noise_to_cameras(self):
        # add noise to cameras
        for db_rec in self.db:
            db_rec['camera'] = self.add_noise_to_camera(db_rec['camera'])

    def index_to_action_names(self):
        return None

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
    
    

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        if self.run_on_all_cameras:
            views = self.all_views_openmplposer
            CAM_IX = {x: i for i, x in enumerate(views)}
        else:
            views = self.views
            CAM_IX = {x: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            if db[i]['camera_id'] not in views:
                continue
            camera_id = CAM_IX[db[i]['camera_id']]
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(views)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        if self.run_on_all_cameras:
            view_combinations = self.generate_combinations(views, self.n_views)
            for _, v in grouping.items():
                for view_comb in view_combinations:
                    if np.all(np.array([v[CAM_IX[view]] for view in view_comb]) != -1):
                        filtered_grouping.append([v[CAM_IX[view]] for view in view_comb])
        else:
            for _, v in grouping.items():
                if np.all(np.array(v) != -1):
                    filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        # items = self.grouping[idx]
        items = copy.deepcopy(self.grouping[idx])
        # if self.run_on_all_cameras:
        #     views = np.random.choice(self.all_views_cmu, self.n_views, replace=False)
        #     items = [items[self.all_views_cmu.index(v)] for v in views]
        if self.normalize_room:
            room_x_scale = self.room_max_x - self.room_min_x
            room_y_scale = self.room_max_y - self.room_min_y
            room_z_scale = self.room_max_z - self.room_min_z
            
            room_center = np.array([(self.room_max_x + self.room_min_x) / 2, (self.room_max_y + self.room_min_y) / 2, (self.room_max_z + self.room_min_z) / 2])
                
        
        shift_room_tri = None
        if self.shift_room:
            if type(self.shift_room_value) == str:
                if self.shift_room_value in ['to_avg_pose', 'to_conf_kp']:
                    cameras_tri = [self.db[x]['camera'].copy() for x in items]
                    for cam in cameras_tri:
                        cam['k'] = [cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]]
                        cam['p'] = [cam['distCoef'][2], cam['distCoef'][3]]
                    loc_2d = np.array([self.db[x]['joints_2d'] for x in items])
                    if self.use_mmpose:
                        conf_2d = np.array([self.db[x]['joints_2d_conf'] for x in items])
                    else:
                        conf_2d = np.ones((len(items), self.num_joints))
                    # conf_mmpose_2d = np.array([self.db[x]['joints_2d_conf'] for x in items])
                    joints_3d_tri = triangulate_poses(cameras_tri, loc_2d, conf_2d)
                    confs = np.min(conf_2d, axis=0).reshape(-1)
                    if self.shift_room_value == 'to_avg_pose':
                        shift_room_tri = - np.average(joints_3d_tri.squeeze(), weights=confs, axis=0).reshape(-1)
                    elif self.shift_room_value == 'to_conf_kp':
                        shift_room_tri = - joints_3d_tri[0, np.argmax(confs)]
                
        # directions, intersections, joints_2d_confs = [], [], []
        directions, intersections, joints_2d_confs, joints_2ds, camera_params, depths = [], [], [], [], [], []
        
        fname_camera_ids = ''
        for ix, item in enumerate(items):
            direction, intersection, joints_2d_conf, joints_3d, meta, joints_2d, K, Rt, depth_vals = self.get_rays(item, shift_room_tri=shift_room_tri)
            directions.append(direction)
            intersections.append(intersection)
            joints_2d_confs.append(joints_2d_conf)
            joints_2ds.append(joints_2d)
            camera_params.append(np.concatenate([K.flatten(), Rt.flatten()]))
            depths.append(depth_vals.reshape(-1, 1)) if depth_vals is not None else depths.append(None)

            fname_camera_ids += '_{:02d}'.format(meta['cam_id'])
        meta['fname_camera_ids'] = fname_camera_ids
            
            
        directions = np.array(directions)           # (n_views, 17, 3)
        intersections = np.array(intersections)     # (n_views, 17, 3)
        joints_2d_confs = np.array(joints_2d_confs)     # (n_views, 17, 1)
        joints_2ds = np.array(joints_2ds)     # (n_views, 17, 2)
        camera_params = np.array(camera_params)     # (n_views, 12)
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
        
        if self.normalize_room:
            middle_points = self.normalize_pose3d_coordinates(middle_points, room_center, room_x_scale, room_y_scale, room_z_scale)
            closest_points_all = self.normalize_pose3d_coordinates(closest_points_all, room_center, room_x_scale, room_y_scale, room_z_scale)
            joints_3d = self.normalize_pose3d_coordinates(joints_3d, room_center, room_x_scale, room_y_scale, room_z_scale)
            directions = self.normalize_pose3d_coordinates(directions, room_center, room_x_scale, room_y_scale, room_z_scale)
            intersections = self.normalize_pose3d_coordinates(intersections, room_center, room_x_scale, room_y_scale, room_z_scale)
            
            meta['room_scaled'] = True
            meta['room_x_scale'] = max(room_x_scale, room_y_scale, room_z_scale)
            meta['room_y_scale'] = max(room_x_scale, room_y_scale, room_z_scale)
            meta['room_z_scale'] = max(room_x_scale, room_y_scale, room_z_scale)
            meta['room_center'] = room_center
            meta['room_scaled_equal'] = True
            
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
        
        # fname = meta['image']
        # if fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[-1].replace('.jpg','') == '171204_pose5_00022704':
        #     print('ix:', idx)
        
        if self.shift_room:
            if type(self.shift_room_value) == str:
                if self.shift_room_value in ['to_avg_pose', 'to_conf_kp']:
                    meta['shift_room_tri'] = shift_room_tri
            
        return middle_points, closest_points_all, joints_3d, rays, meta, joints_2ds
    

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 'cat_{}_vid_{}_frmid_{:08}'.format(
            datum['subject'],
            datum['image'].split('/')[-1].replace('.pkl', ''),
            datum['image_id'])
        
