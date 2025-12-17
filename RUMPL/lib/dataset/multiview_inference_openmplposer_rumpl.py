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
from utils.calib import *


from torch.utils.data import Dataset
from dataset.joints_dataset_rumpl import JointsDataset_RUMPL
import logging
import json
import glob
import os
import xml.etree.ElementTree as ET
import torch
import re

logger = logging.getLogger(__name__)

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiViewInference_OpenMPLPoser_RUMPL(JointsDataset_RUMPL):

    def __init__(self, cfg, transform=None):
        super().__init__(cfg, None, False, transform)
        self.num_joints = 17
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
        self.keypoints_inverse_dict = {v: k for k, v in self.actual_joints.items()}
        
        self.cameras_path = cfg.DATASET.INFERENCE_CAMERAS_PATH
        self.data_dir = cfg.DATASET.INFERENCE_DATA_DIR
        self.openmplposer_cameras = self.load_all_cameras(self.cameras_path)

        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.use_t = cfg.DATASET.USE_T
        self.clip_joints = cfg.DATASET.CLIP_JOINTS

        self.kp_visiblity_th = cfg.DATASET.KP_VISIBILITY_TH
        self.zero_tokens_for_missing_joints = cfg.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS

        self.inputs_normalized = cfg.DATASET.INPUTS_NORMALIZED

        # self.filename_list = glob.glob(f'{self.data_dir}')

        # logger.info('=> load {} files from {}'.format(len(self.filename_list), self.data_dir))


        with open(self.data_dir, 'rb') as f:
            data = pickle.load(f)
        self.db = data['yolo_keypoints']

        for k, v in self.db.items():
            if k == 'yolo_version':
                continue
            self.db[k] = v.numpy()

        logger.info('=> load {} frames from {}'.format(len(self.db['vcam0']), self.data_dir))


    def load_all_cameras(self, calibs):

        def parse_matrix(node):
            rows = int(node.find('rows').text)
            cols = int(node.find('cols').text)
            data_text = node.find('data').text.strip().replace('\n', ' ')
            data = list(map(float, data_text.split()))
            return np.array(data).reshape((rows, cols))
        
        cams = range(1, 4)
        cameras_all_calibs = {cam_id: [] for cam_id in cams}
        xml_files = glob.glob(osp.join(calibs, '*.xml'))
        # Ensure sorted list of camera_files
        def get_cam_id(path):
            filename = os.path.basename(path)
            _match = re.search(r'Camera_(\d+)\.xml', filename)
            return int(_match.group(1)) if _match else float('inf')

        xml_files.sort(key=get_cam_id) 
        distCoef =  [0, 0, 0, 0, 0]

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

            R = P[:3, :3]
            T = P[:3, 3:]

            # R = R.T

            # F = np.diag([-1, 1, 1])
            # R = R.copy()
            # R = F @ R

            F = np.diag([1, -1, -1])
            R = F @ R.T  # Adjust the rotation matrix

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

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        idx_image = idx


        camera_ids = self.openmplposer_cameras.keys()

        directions, intersections, joints_2d_confs, joints_2ds, camera_params = [], [], [], [], []

        for camera_id in camera_ids:
            direction, intersection, joints_2d_conf, joints_3d, meta, joints_2d, K, Rt = self.get_rays(idx, camera_id)
            directions.append(direction)
            intersections.append(intersection)
            joints_2d_confs.append(joints_2d_conf)
            joints_2ds.append(joints_2d)
            camera_params.append(np.concatenate([K.flatten(), Rt.flatten()]))
            
        directions = np.array(directions)           # (n_views, 17, 3)
        intersections = np.array(intersections)     # (n_views, 17, 3)
        joints_2d_confs = np.array(joints_2d_confs)     # (n_views, 17, 1)
        joints_2ds = np.array(joints_2ds)     # (n_views, 17, 2)
        camera_params = np.array(camera_params)     # (n_views, 12)
        camera_params = np.repeat(camera_params[:, None, :], self.num_joints, axis=1)   # (n_views, 17, 12)
        joints_2ds = np.concatenate([joints_2ds, camera_params], axis=2)   # (n_views, 17, 14)
        
        directions = np.transpose(directions, (1, 0, 2))   # (17, n_views, 3)
        intersections = np.transpose(intersections, (1, 0, 2))   # (17, n_views, 3)
        joints_2d_confs_ = np.transpose(joints_2d_confs, (1, 0, 2))   # (17, n_views, 1)
        joints_2ds = np.transpose(joints_2ds, (1, 0, 2))   # (17, n_views, 14)
        
        
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
        

        middle_points = torch.from_numpy(middle_points).float()
        closest_points_all = np.concatenate([closest_points_all, joints_2d_confs_], axis=2)  # (17, n_views, 4)        
        closest_points_all = torch.from_numpy(closest_points_all).float()
        rays = np.concatenate([directions, intersections, joints_2d_confs_], axis=2)   # (17, n_views, 6)
        joints_2ds = np.concatenate([joints_2ds, joints_2d_confs_], axis=2)
        joints_2ds = torch.from_numpy(joints_2ds).float()
        
        # camera_params = torch.from_numpy(camera_params).float()
        if self.zero_tokens_for_missing_joints:
            rays *= joints_2d_confs
            closest_points_all *= joints_2d_confs
            joints_2ds *= joints_2d_confs
            # camera_params *= joints_2d_confs
        rays = torch.from_numpy(rays).float()
        

        return middle_points, closest_points_all, [], rays, meta, joints_2ds




    def get_rays(self, idx, camera_id):

        camera = self.openmplposer_cameras[camera_id][0].copy()

        joints = self.db['vcam{}'.format(camera_id - 1)][idx].copy()  # (17, 2)
        joints_vis = self.db['confidences'][idx, :, camera_id - 1].copy().reshape(-1, 1)  # (17, 1)

        R = camera['R']
        T = camera['T']
        t = camera['t']
        K = camera['K']

        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()
        cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate

        if self.clip_joints:
            joints_vis[:, 0] = np.where(0 < joints[:, 0], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 0] < self.image_size[0] - 1, joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(0 < joints[:, 1], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 1] < self.image_size[1] - 1, joints_vis[:, 0], 0)
            joints[:, 0] = np.clip(joints[:, 0], 0, self.image_size[0] - 1)
            joints[:, 1] = np.clip(joints[:, 1], 0, self.image_size[1] - 1)
        else:
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= self.image_size[0] or
                            joints[i, 1] >= self.image_size[1]):
                        joints_vis[i, :] = 0
                        joints[i, :] = 0        # to avoid further errors when downscaling
                else:
                    joints[i, :] = 0            # to avoid further errors when downscaling

        if self.inputs_normalized:
            joints = self.normalize_screen_coordinates(joints, self.image_size[0], self.image_size[1])

        coords_ray = self.create_3d_ray_coords(camera, joints)
        # joints_vis = np.ones((self.num_joints, 1), dtype=np.float32)  # all visible

        direction_vectors, intersection_points = self.generate_direction_vectors_and_intersection_points(coords_ray, cam_center)
        direction_vectors = direction_vectors.astype(np.float32)
        intersection_points = intersection_points.astype(np.float32)


        meta = {
            'camera_id': camera_id,
            'frame_id': idx,
            'rays': coords_ray,  # (hw, 3)
            'cam_center': cam_center,  # (3,)
        }
        K_ = np.array([K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]])
        return direction_vectors, intersection_points, joints_vis, None, meta, joints, K_, Rt


    def create_3d_ray_coords(self, camera, joints_ds):
        multiplier = 1.0                        # avoid numerical instability

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
        # coords_cam = - coords_cam.copy()  # in openmplposer, we need to flip the x-axis

        if self.use_t:
            coords_world = (camera['R'].T @ coords_cam.T + camera['t']).T  # (hw, 3)    in world coordinate    array
        else:
            coords_world = (camera['R'].T @ coords_cam.T + camera['T']).T  # (hw, 3)    in world coordinate    array
        coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
        return coords_world

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return (X/w)*2 - [1, h/w]
    
    def __len__(self):
        return len(self.db['vcam0'])

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])
        
    def isdamaged(self, db_rec):
        # from https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py#L174
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if db_rec['subject'] == 9:
            if db_rec['action'] != 5 or db_rec['subaction'] != 2:
                if db_rec['action'] != 10 or db_rec['subaction'] != 2:
                    if db_rec['action'] != 13 or db_rec['subaction'] != 1:
                        return False
        else:
            return False
        return True