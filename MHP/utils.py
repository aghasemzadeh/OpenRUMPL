
import os
from os import path as osp
import numpy as np
import torch
from tqdm import tqdm
import pickle
import trimesh
import argparse
import glob 
import torch
import json
import copy
from scipy.spatial.transform import Rotation as R
import cv2
import xml.etree.ElementTree as ET
import re


def cam_to_image(point_cam, K):
    # takes batches of points and converts them to image 2D coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    
    K = np.array(K)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    M = np.hstack([K, np.zeros((3, 1))])
    M = np.stack([M]*point_cam.shape[1])                      # create M for 17 points
    M = np.stack([M]*p.shape[0])             # create M for all matchings
    
    point_img = M @ p
    point_img_squeezed = np.squeeze(point_img, axis=-1)
    point_img = point_img_squeezed[:,:,:] / point_img[:,:,2]  # convert back to cartesian (check that x_homo[1] > 0)
    point_img = point_img[:,:,:2]
    return point_img

def world_to_cam(point_world, R, T):
    # takes batches of points and converts them to world coordinates
    # point_world: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_world.shape[0], point_world.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_world, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E = np.stack([E]*point_world.shape[1])                      # create E for 17 points
    E = np.stack([E]*p.shape[0])             # create E for all matchings
    point_cam = E @ p
    point_cam_squeezed = np.squeeze(point_cam, axis=-1)
    point_cam = point_cam_squeezed[:,:,:] / point_cam[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_cam = point_cam[:,:,:3]
    return point_cam

def cam_to_world(point_cam, R, T):
    # takes batches of points and converts them to world coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, 3, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E_inv = np.linalg.inv(E)
    E_inv = np.stack([E_inv]*point_cam.shape[1])                      # create E_inv for 3 points
    E_inv = np.stack([E_inv]*p.shape[0])             # create E_inv for all matchings
    point_world = E_inv @ p
    point_world_squeezed = np.squeeze(point_world, axis=-1)
    point_world = point_world_squeezed[:,:,:] / point_world[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_world = point_world[:,:,:3]
    return point_world

# load camera calib info
def load_all_cameras_h36m(calib_file, actors):
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
            camera_dict['T'] = camera[1] / 1000
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

def compatible_cams(cameras):
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['k'] = np.array([cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]])
        cam['p'] = np.array([cam['distCoef'][2], cam['distCoef'][3]])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1)) / 100
        cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
        cam['fx'] = cam['K'][0,0]
        cam['fy'] = cam['K'][1,1]
        cam['cx'] = cam['K'][0,2]
        cam['cy'] = cam['K'][1,2]
    return cameras

def load_all_cameras_cmu(calib_root, cmu_calibs):
    """
    loads all cameras from the calibration file and returns a dictionary
    containing the camera parameters for each camera
    each item has a list showing different camera setups (actors)
    """
    cams = range(31)
    cameras_all_calibs = {cam_id: [] for cam_id in cams}
    for cam_setup in cmu_calibs:
        calib_file = osp.join(calib_root, 'calibration_{}.json'.format(cam_setup))
        with open(calib_file, 'r') as f:
            calibration_cat = json.load(f)
        cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd'}
        cameras = compatible_cams(cameras)
        for cam_id, cam in cameras.items():
            cameras_all_calibs[cam_id].append(cam)
    return cameras_all_calibs

def load_all_cameras_rich(calib_file, rich_scenes):
    with open(calib_file, 'r') as f:
        camera_data = json.load(f)
    cameras_all_calibs = {cam_id: [] for cam_id in range(0, 8)}
    for scene in rich_scenes:
        for cam_id in range(0, 8):
            try:
                camera = camera_data[scene][str(cam_id)]
            except KeyError:
                cameras_all_calibs[cam_id].append({})
                continue
                
            camera_dict = {}
            camera_dict['camera_setup'] = scene
            camera_dict['camera_id'] = cam_id
            camera_dict['R'] = np.array(camera['R'])
            camera_dict['t'] = np.array(camera['t']).reshape(3, 1)
            camera_dict['T'] = np.array(camera['T']).reshape(3, 1)
            camera_dict['K'] = np.array(camera['K'])
            camera_dict['fx'] = camera_dict['K'][0,0]
            camera_dict['fy'] = camera_dict['K'][1,1]
            camera_dict['cx'] = camera_dict['K'][0,2]
            camera_dict['cy'] = camera_dict['K'][1,2]
            camera_dict['k'] = np.array([camera['distCoef'][0], camera['distCoef'][1], camera['distCoef'][4]])
            camera_dict['p'] = np.array([camera['distCoef'][2], camera['distCoef'][3]])
            cameras_all_calibs[cam_id].append(camera_dict)
    return cameras_all_calibs

def get_rotation_matrix(rotation):
    # Convert Euler angles to rotation matrix
    r = R.from_euler('xyz', rotation)
    return r.as_matrix()

def get_translation_matrix(position):
    # Convert position to translation vector
    return np.array(position).reshape(3, 1)

def adjust_rotation_matrix(R):
    # Swap Y and Z axes
    swap_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    adjusted_R = swap_matrix @ R
    return adjusted_R

def adjust_translation_vector(t):
    adjusted_t = np.array([t[0], t[2], -t[1]])
    return adjusted_t


def load_all_cameras_dome(calib_file):
    distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic
    with open(calib_file, 'r') as f:
        camera_data = json.load(f)
    # cams = range(1, len(camera_data.keys()) + 1)
    cams = range(1, len(camera_data) + 1)
    cameras = {cam_id:[] for cam_id in cams}
    for i in cams:
        camera = camera_data[i - 1]
        # camera = camera_data['Camera_{}'.format(i)]
        # cam_id = i
        cam_id = int(camera['name'].split('_')[1])
        camera_dict = {}
        camera_dict['camera_setup'] = 'dome'
        camera_dict['camera_id'] = cam_id
        # camera_dict['R'] = get_rotation_matrix(camera['rotation'])
        # camera_dict['T'] = get_translation_matrix(camera['position'])
        # camera_dict['t'] = - np.linalg.inv(camera_dict['R'].T) @ camera_dict['T']
        # camera_dict['t'] = get_translation_matrix(camera['position'])
        # camera_dict['T'] = - camera_dict['R'].T @ camera_dict['t']
        camera_dict['R'] = np.array(camera['R'])
        # mat = np.array([[1, 0, 0], 
        #     [0, -1, 0], 
        #     [0, 0, -1]])
        # camera_dict['R'] = camera_dict['R'] @ mat
        camera_dict['t'] = np.array(camera['t']).reshape(3, 1)
        # Flip the extrinsic parameters vertically
        # flip_matrix = np.array([
        #     [-1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ])

        # camera_dict['R'] = camera_dict['R'] @ flip_matrix
        # camera_dict['t'][0] = - camera_dict['t'][0]
        # camera_dict['t'] = flip_matrix @ camera_dict['t']
        # camera_dict['t'][1] = -camera_dict['t'][1]
        # camera_dict['t'][2] = -camera_dict['t'][2]
        camera_dict['T'] = np.array(camera['T']).reshape(3, 1)
        # camera_dict['T'] = np.array(camera['t']).reshape(3, 1)
        # camera_dict['t'] = np.array(camera['t']).reshape(3, 1)
        # camera_dict['T'] = - camera_dict['R'].T @ camera_dict['t']
        # camera_dict['t'] = np.array(camera['t']).reshape(3, 1)
        # camera_dict['T'] = - camera_dict['R'].T @ camera_dict['t']
        # camera_dict['T'] = np.array(camera['t']).reshape(3, 1)
        # camera_dict['T'] = np.array(camera['t']).reshape(3, 1)
        # camera_dict['t'] = - np.linalg.inv(camera_dict['R'].T) @ camera_dict['T']
        # Adjust R and t
        # camera_dict['R'] = adjust_rotation_matrix(camera_dict['R'])
        # camera_dict['T'] = adjust_translation_vector(camera_dict['T'])
        # camera_dict['t'] = - np.linalg.inv(camera_dict['R'].T) @ camera_dict['T']
        # camera_dict['t'] = - np.linalg.inv(camera_dict['R']) @ camera_dict['T']
        # camera_dict['t'] = - camera_dict['R'] @ camera_dict['T']
        camera_dict['K'] = np.array(camera['K'])
        camera_dict['fx'] = camera_dict['K'][0,0]
        camera_dict['fy'] = camera_dict['K'][1,1]
        camera_dict['cx'] = camera_dict['K'][0,2]
        camera_dict['cy'] = camera_dict['K'][1,2]
        camera_dict['k'] = np.array([distCoef[0], distCoef[1], distCoef[4]])
        camera_dict['p'] = np.array([distCoef[2], distCoef[3]])
        cameras[cam_id].append(camera_dict)
        
    return cameras

# def load_all_20_cameras():
#     calib_file = 'cameras_20.json'
#     with open(calib_file, 'r') as f:
#         calibration_cat = json.load(f)
#     cameras = {i:cam for i, cam in enumerate(calibration_cat['cameras'])}
#     for cam_id, cam in cameras.items():
#         cam[]
def extract_from_xml(file_path):
    # Parse the XML file using ElementTree and get the root element
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the 'Intrinsics' element in the XML and extract the 'data' text
    intrinsics = root.find('Intrinsics')
    data_text = intrinsics.find('data').text
    # Split the text into lines and convert each line into a list of floats
    K = np.array([list(map(float, line.split())) for line in data_text.strip().split('\n')])

    # Find the 'CameraMatrix' element in the XML and extract the 'data' text
    extrinsics = root.find('CameraMatrix')
    data_text = extrinsics.find('data').text

    # Split the text into lines and convert each line into a list of floats
    camera_pose = np.array([list(map(float, line.split())) for line in data_text.strip().split('\n')])

    #Imae size
    image_width = int(root.find('image_width').text)
    image_height = int(root.find('image_height').text)
    image_size = (image_width, image_height)
    
    return K, camera_pose, image_size

def load_all_cameras_openmplposer(calib_root):
    """
    loads all cameras from the root directory and returns a dictionary
    containing the camera parameters for each camera is in a xml file
    """
    def parse_matrix(node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data_text = node.find('data').text.strip().replace('\n', ' ')
        data = list(map(float, data_text.split()))
        return np.array(data).reshape((rows, cols))
    
    cams = range(1, 4)
    distCoef =  [0,0,0,0,0]  # default distortion coefficients, can be changed later
    cameras_all_calibs = {cam_id: [] for cam_id in cams}
    xml_files = glob.glob(osp.join(calib_root, '*.xml'))
    # Ensure sorted list of camera_files
    def get_cam_id(path):
        filename = os.path.basename(path)
        _match = re.search(r'Camera_(\d+)\.xml', filename)
        return int(_match.group(1)) if _match else float('inf')

    xml_files.sort(key=get_cam_id) 
    for xml_file in xml_files:

        # Parse the XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        K = parse_matrix(root.find('Intrinsics'))
        P = parse_matrix(root.find('CameraMatrix'))

        # K = np.linalg.inv(K)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        R = P[:3, :3]
        T = P[:3, 3:]

        # R = R.T

        # F = np.diag([-1, 1, 1])
        # R = R.copy()
        # # # t = t.copy()
        # R = F @ R
        # # t = -np.linalg.inv(R.T) @ T
        # # t = F @ t
        # # T = -R.T @ t

        F = np.diag([1, -1, -1])
        R = F @ R.T  # Adjust the rotation matrix

        # R = R.T


        K_, camera_pose, _ = extract_from_xml(xml_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))

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
            'K_': np.array(K_),
            'camera_pose': camera_pose,
            'distCoef': distCoef,
        }
        cameras_all_calibs[camera_dict['camera_id']].append(camera_dict)

    return cameras_all_calibs

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

def calculate_mpjpe(pred, gt):
    if len(pred.shape) == 2:
        pjpe = np.sqrt(((pred - gt) ** 2).sum(axis=1))
        return pjpe.mean()
    else:
        pjpe = np.sqrt(((pred - gt) ** 2).sum(axis=2))
        return pjpe.mean(axis=1)
    
def calculate_scale_factor_1d(source_points, dest_points):
    distances_source = np.abs(np.diff(source_points, axis=0))
    distances_dest = np.abs(np.diff(dest_points, axis=0))

    scale_factor = distances_dest.mean() / distances_source.mean()
    return scale_factor.item()

def run_mmpose(image, mmpose_inferencer, convert_to_h36m=True, return_coco=False):
    result_mmpose_generator = mmpose_inferencer(image, show=False) #, device='cuda:0') #, show_progress=False)
    result_mmpose = next(result_mmpose_generator)
    if len(result_mmpose['predictions'][0]) == 0:
        keypoints = np.zeros((17, 2), dtype=float)
        keypoint_scores = np.zeros((1, 17), dtype=float)
    else:
        predictions_mmpose = result_mmpose['predictions'][0][0] # first image - first prediction
        keypoints = np.array(predictions_mmpose['keypoints'])
        # keypoints = predictions_mmpose['keypoints'].cpu().numpy()
        keypoint_scores = np.array(predictions_mmpose['keypoint_scores'])
        # keypoint_scores = predictions_mmpose['keypoint_scores'].cpu().numpy()
    del result_mmpose_generator
    if convert_to_h36m:
        h36m_points_mmpose, h36m_scores_mmpose = mmpose2h36m(keypoints, keypoint_scores)
        if return_coco:
            return h36m_points_mmpose, h36m_scores_mmpose, keypoints, keypoint_scores.reshape(-1, 1)
        return h36m_points_mmpose, h36m_scores_mmpose
    else:
        return keypoints, keypoint_scores.reshape(-1, 1)
    
def run_yolo(image, yolo_model):
    results = yolo_model.predict(image, device=0, verbose=False, stream=False)
    
    res = results[0]
    if len(res) > 0:
        keypoints = res[0].keypoints.data.cpu().numpy()[0, :, :2] 
        keypoint_scores = res[0].keypoints.conf.cpu().numpy()
    else:
        keypoints = np.zeros((17, 2))
        keypoint_scores = np.zeros((17, 1))
        print('Warning! No keypoints detected by YOLO')

    return keypoints, keypoint_scores.reshape(-1, 1)

def amass_vertices_to_joints(vertices, h36m_jregressor, R=None, t=None, K=None, lower_body_reversed=True):
    joints = torch.einsum('bik,ji->bjk', [vertices, h36m_jregressor])
    joints_copy = joints.clone()
    if lower_body_reversed:
        # to account for the mistake in the regressor (lower body reversed)
        joints[:, 1:4] = joints_copy[:, 4:7]
        joints[:, 4:7] = joints_copy[:, 1:4]
        
    if R is None:
        return None, joints.cpu().numpy()
    joints_cam = world_to_cam(joints, R, t)
    points_image = cam_to_image(joints_cam, K)
    return points_image, joints.cpu().numpy()

def find_most_aligned_joints(points_image_amass, keypoints_most_aligned_with_h36m, imw, imh):
    # find the most aligned joints
    most_aligned_joints = []
    for joint in keypoints_most_aligned_with_h36m:
        joint_amass = points_image_amass[joint]
        if joint_amass[0] < 0 or joint_amass[0] > imw or joint_amass[1] < 0 or joint_amass[1] > imh:
            continue
        most_aligned_joints.append(joint)
    if len(most_aligned_joints) == 1:
        if most_aligned_joints[0] == keypoints_most_aligned_with_h36m[0]:
            most_aligned_joints.append(keypoints_most_aligned_with_h36m[1])
        else:
            most_aligned_joints.append(keypoints_most_aligned_with_h36m[0])
    return most_aligned_joints

def filter_wrong_mmpose_detections(h36m_points_mmpose, h36m_scores_mmpose, points_image_amass, imw, imh):
    keypoints_outside = np.logical_or(np.logical_or(points_image_amass[:, 0] < 0, points_image_amass[:, 0] > imw),
                                        np.logical_or(points_image_amass[:, 1] < 0, points_image_amass[:, 1] > imh))
    h36m_scores_mmpose[keypoints_outside] = 0
    return h36m_scores_mmpose

def fit_skeleton_to_amass(skeleton, amass_joints_2d, most_reliable_joints, mmpose_confs, imw, imh, apply_joint_clip, fit_using_most_aligned=False):
    if most_reliable_joints == []:
        return np.zeros((17, 2)), np.zeros((17, 1))
    scale_factor_x = calculate_scale_factor_1d(skeleton[most_reliable_joints, 0], amass_joints_2d[most_reliable_joints, 0])
    scale_factor_y = calculate_scale_factor_1d(skeleton[most_reliable_joints, 1], amass_joints_2d[most_reliable_joints, 1])
    scale_factor = np.array([scale_factor_x, scale_factor_y])
    if fit_using_most_aligned:
        len_most_reliable_joints = len(most_reliable_joints)
        all_possible = (np.stack([skeleton] * len_most_reliable_joints) - skeleton[most_reliable_joints, None]) * scale_factor + amass_joints_2d[most_reliable_joints, None]
        amass_joints_2d_ = np.stack([amass_joints_2d] * len_most_reliable_joints)
    else:
        all_possible = (np.stack([skeleton] * 17) - skeleton[range(17), None]) * scale_factor + amass_joints_2d[range(17), None]
        amass_joints_2d_ = np.stack([amass_joints_2d] * 17)
    all_mpjpes = calculate_mpjpe(all_possible, amass_joints_2d_)
    best_kp = all_mpjpes.argmin()
    best_match_joints = all_possible[best_kp]
    # clip best match joints to image size 
    to_zero = np.logical_or(np.logical_or(best_match_joints[:, 0] < 0, best_match_joints[:, 0] > imw), np.logical_or(best_match_joints[:, 1] < 0, best_match_joints[:, 1] > imh))
    mmpose_confs_ = mmpose_confs.copy()
    if apply_joint_clip:
        mmpose_confs_[to_zero] = 0
        best_match_joints[:, 0] = np.clip(best_match_joints[:, 0], 0, imw - 1)
        best_match_joints[:, 1] = np.clip(best_match_joints[:, 1], 0, imh - 1)
    return best_match_joints, mmpose_confs_

def locate_mesh_in_room(vertices, room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, room_max_z, h36m_or_cmu, rotate=False, root_loc=None):
    if root_loc is None:
        root_loc = vertices[0, :]
    if h36m_or_cmu == 'triangulate_3d':
        vertices[:] = vertices[:] - vertices[0, :]
        # vertices_copy = vertices.clone()
        # vertices[:, 1] = -vertices[:, 2]   # swap y and z
        # vertices[:, 2] = vertices_copy[:, 1]  # swap y and z
        return vertices
    augmentation_3d = torch.cat([torch.rand(1, 1) * (room_max_x - room_min_x) + room_min_x, torch.rand(1, 1) * (room_max_y - room_min_y) + room_min_y, torch.rand(1, 1) * (room_max_z - room_min_z) + room_min_z], dim=1)
    if room_min_z != 0 and room_max_z != 0:
        vertices = vertices - root_loc
        # vertices = vertices - vertices[0, :]
        pass
    else:
        vertices[:, 0] = vertices[:, 0] - root_loc[0]
        vertices[:, 1] = vertices[:, 1] - root_loc[1]
        # vertices[:, 0] = vertices[:, 0] - vertices[0, 0]
        # vertices[:, 1] = vertices[:, 1] - vertices[0, 1]
    if rotate:
        rotation = np.random.rand(1) * 360
        vertices = rotate_pose(vertices[None], rotation, axis='z').squeeze()
        # vertices = vertices.type(torch.float32)
    vertices_augmented = vertices + augmentation_3d
    vertices_augmented = vertices_augmented.type(torch.float32)
    if h36m_or_cmu == 'cmu': # or h36m_or_cmu == 'dome':
        vertices_augmented_copy = vertices_augmented.clone()
        vertices_augmented[:, 1] = -vertices_augmented[:, 2]   # swap y and z
        vertices_augmented[:, 2] = vertices_augmented_copy[:, 1]  # swap y and z
    return vertices_augmented


def rotate_pose(poses, angles_degrees, axis='y'):
    # Convert angles to radians
    angles_radians = np.radians(angles_degrees)

    # Number of poses in the batch
    b = poses.shape[0]
    rotation_matrix = np.zeros((3, 3, b))
    # Initialize rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix[0, 0, :] = 1
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[1, 2, :] = -np.sin(angles_radians)
        rotation_matrix[2, 1, :] = np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'y':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 2, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = 1
        rotation_matrix[2, 0, :] = -np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'z':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 1, :] = -np.sin(angles_radians)
        rotation_matrix[1, 0, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[2, 2, :] = 1
    else:
        raise ValueError("Invalid axis. Please choose 'x', 'y', or 'z'.")
    # Rotate the poses
    rotated_poses = np.matmul(poses, rotation_matrix.transpose((2,1,0)))

    return rotated_poses


def add_camera_between(cameras, cam_id_new, cam_id, cam_id_2, avg_on_axis=[0, 1, 2]):
    cameras[cam_id_new] = copy.deepcopy(cameras[cam_id])
    for i, _ in enumerate(cameras[cam_id_new]):
        cameras[cam_id_new][i]['camera_id'] = cam_id_new
        for j in avg_on_axis:    
            cameras[cam_id_new][i]['T'][j] = (cameras[cam_id][i]['T'][j] + cameras[cam_id_2][i]['T'][j]) / 2
        cameras[cam_id_new][i]['R'] = (cameras[cam_id][i]['R'] + cameras[cam_id_2][i]['R']) / 2
        cameras[cam_id_new][i]['t'] = -np.array(np.dot(np.linalg.inv(cameras[cam_id_new][i]['R'].T), cameras[cam_id_new][i]['T']))

    return cameras


def random_camera_in_room(camera_location_limit, room_size, camera_location_outside_room, camera_dist_from_person=0, person_location=None, room_center=None):
    # Room dimensions
    
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

    return R, T, t

def get_sphere_points_angular(radius=6, z_start=1, z_end=5, angle_step=5, end_angle=180):
    """
    Calculate points on a sphere using angular steps.
    
    Args:
        radius (float): Radius of the sphere
        z_start (float): Starting z coordinate
        z_end (float): Ending z coordinate
        angle_step (float): Angular step in degrees
        
    Returns:
        dict: Dictionary with z-levels as keys and lists of (x,y,z) coordinates as values
    """
    # Convert angle step to radians
    angle_step_rad = np.radians(angle_step)
    
    # Calculate theta (polar angle) range for given z values
    theta_start = np.arccos(z_end/radius)  # angle from positive z-axis
    theta_end = np.arccos(z_start/radius)
    
    # Generate theta values
    theta_values = np.arange(theta_start, theta_end + angle_step_rad, angle_step_rad)
    
    # Generate phi values (azimuthal angle in x-y plane)
    end_angle_rad = np.radians(end_angle)
    phi_values = np.arange(0, end_angle_rad, angle_step_rad)
    
    # Initialize result dictionary
    result = {}
    
    # Generate points
    for theta in theta_values:
        # Calculate z for this theta
        z = radius * np.cos(theta)
        
        # Round z to 3 decimal places
        z_rounded = round(z, 3)
        
        # Skip if z is outside our range
        if z_rounded < z_start or z_rounded > z_end:
            continue
            
        # Initialize points list for this z level
        if z_rounded not in result:
            result[z_rounded] = []
        
        # Calculate points around this latitude
        for phi in phi_values:
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            
            # Round coordinates to 3 decimal places
            point = [round(x, 3), round(y, 3)]
            result[z_rounded].append(point)

    return result

def get_cameras_on_sphere(radius=6, z_start=1, z_end=5, angle_step=5, end_angle=180, person_location=None, room_center=None):
    """
    Calculate camera positions on a sphere using angular steps.
    
    Args:
        radius (float): Radius of the sphere
        z_start (float): Starting z coordinate
        z_end (float): Ending z coordinate
        angle_step (float): Angular step in degrees
        
    Returns:
        dict: Dictionary camera parameters for each camera
    """
    # Center of the room
    if room_center is None:
        center = np.array([0, 0, 0])
    else:
        center = np.array(room_center)
        
    person_location = np.array([0, 0, 0]) if person_location is None else np.array(person_location)
    
        
    locations_dict = get_sphere_points_angular(radius, z_start, z_end, angle_step, end_angle)
    
    cameras = []
    for i, (height, locations) in enumerate(locations_dict.items()):
        for j, loc in enumerate(locations):
            x = loc[0]
            y = loc[1]
            z = height
            T = np.array([x, y, z])
            
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
            
            cameras.append(
                {
                    'R': R,
                    'T': T,
                    't': t,
                    'category_height': i,
                    'category_angle': j,
                }
            )
            
    return cameras
    
    