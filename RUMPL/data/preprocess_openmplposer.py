import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse
import glob
import xml.etree.ElementTree as ET

# import torch

# for detecting the bbox of humans
# from mmdet.apis import DetInferencer


img_size = (1280, 720)

CAMERA_FINDER = {
    0: 'camera_2',
    1: 'camera_3',
    2: 'camera_1',
}

joints_coco = {
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

# mmdet for bbox detection
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    
    x = np.asarray(R*X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x

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

def compatible_cams(cameras):
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))
        cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
        cam['fx'] = cam['K'][0,0]
        cam['fy'] = cam['K'][1,1]
        cam['cx'] = cam['K'][0,2]
        cam['cy'] = cam['K'][1,2]
    return cameras
    

def process_video(args):
    video_info = args['video_info']
    cameras = args['cameras']
    category = args['category']
    preprocess_mmpose = args['preprocess_mmpose']
    with open(video_info, 'rb') as f:
        try:
            data_dict = pickle.load(f)['pose_estimation_keypoints']
        except:
            data_dict = pickle.load(f)['yolo_keypoints']
    
    dataset = []
    for cam_id in cameras.keys():
        # cam_id = int(cam.replace('vcam', ''))

        # if cam_id not in cameras:
        #     print('camera {} not found in cameras'.format(cam_id))
        #     continue
        camera = cameras[cam_id]
        # for frame_id, joints_2d in enumerate(data_dict[f'vcam{cam_id}']):
        for frame_id in range(len(data_dict[f'vcam{cam_id-1}'])):
            # print('Processing frame {} for camera {}'.format(frame_id, cam_id))
            joints_3d = np.array(data_dict['ground_truth'][frame_id])

            K = np.matrix(camera['K'])
            distCoef = np.array(camera['distCoef'])
            camera['k'] = np.array([distCoef[0], distCoef[1], distCoef[4]])
            camera['p'] = np.array([distCoef[2], distCoef[3]])
            R = np.matrix(camera['R'])
            t = np.array(camera['t']).reshape((3,1))

            if preprocess_mmpose == 'mmpose':
                joints_2d = np.array(data_dict[f'vcam{cam_id-1}'][frame_id])[:, :2]  # (17, 2)
            elif preprocess_mmpose == 'preprocess':
                joints_2d = cam_to_image(world_to_cam(joints_3d[None], R, t), K)[0]


            invis = (joints_2d[:,0] < 0 ) + (joints_2d[:,0] > img_size[0]) + (joints_2d[:,1] < 0) + (joints_2d[:,1] > img_size[1])
            vis = 1 - invis
            vis = np.repeat(vis.reshape(-1,1), 3, axis=1)   # for compatibility with h36m
            max_x = np.max(joints_2d[vis[:,0] == True, 0], initial=0)
            min_x = np.min(joints_2d[vis[:,0] == True, 0], initial=0)
            max_y = np.max(joints_2d[vis[:,0] == True, 1], initial=0)
            min_y = np.min(joints_2d[vis[:,0] == True, 1], initial=0)
            box = np.array([min_x, min_y, max_x, max_y])
            
            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
            scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)

            data = {
                'image': video_info,
                'joints_2d': joints_2d,
                'joints_vis': vis,
                'joints_3d': joints_3d,
                # 'joints_3d_conf': skel[:,3].reshape((-1,1)),
                'source': 'openmplposer',
                'subject': category,
                'pose_id': 0,  # pose_id is not used in openmplposer
                'frame_id': frame_id,
                'image_id': frame_id,
                'camera_id': cam_id,
                'center': center,
                'scale': scale,
                'box': box,
                'camera': camera,
            }
            if preprocess_mmpose == 'mmpose':
                joints_2d_conf = np.array(data_dict['confidences'][frame_id][:, cam_id-1])  # (17,)
                data['joints_2d_conf'] = joints_2d_conf.reshape((-1, 1))  # (17, 1)
            dataset.append(data)
    return dataset


def parse_matrix(node):
    rows = int(node.find('rows').text)
    cols = int(node.find('cols').text)
    data_text = node.find('data').text.strip().replace('\n', ' ')
    data = list(map(float, data_text.split()))
    return np.array(data).reshape((rows, cols))
   
def create_dataset(camera_path,
                   cams,
                   running_modes, 
                   pkl_dir, 
                   dir_openmplposer,
                   train_or_test,
                   pkl_dir_mmpose,
                   mmpose_output_path, 
                   keypoints_standard,
                   dont_discard_less_than_5_visible_joints,
                   protocols):

    xml_files = glob.glob(os.path.join(camera_path, '*.xml'))
    cameras = {}
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
        # F = np.diag([-1, 1, 1])
        # R = F @ R.T

        F = np.diag([1, -1, -1])
        R = F @ R.T  # Adjust the rotation matrix
        distCoef =  [0, 0, 0, 0, 0]
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
            'distCoef': np.array(distCoef),
            'R': np.array(R),
            'T': np.array(T),
            't': -np.linalg.inv(R.T) @ T,
        }
        cameras[camera_dict['camera_id']] = camera_dict

    for mode in running_modes:    
            
        video_list = []
        for protocol in protocols:
            cats = os.listdir(os.path.join(dir_openmplposer, protocol))
            for cat in tqdm(cats, desc='Fetching {} frames'.format(train_or_test)):
                for video in os.listdir(os.path.join(dir_openmplposer, protocol, cat, train_or_test)):
                    # if len(video_list) >= 10:
                    #     break
                    if video.endswith('.pkl'):
                        video_list.append({
                            'video_info': os.path.join(dir_openmplposer, protocol, cat, train_or_test, video),
                            'cameras': cameras,
                            'category': cat,
                            'preprocess_mmpose': mode,
                        })
            
        print('Total frames to process: {}'.format(len(video_list)))
        
        dataset_ = []
        pool = Pool()
        # for result in tqdm(pool.imap(process_video, video_list), total=len(video_list), desc='Processing {} frames'.format(train_or_test)):
        #     # if result != {}:
        #     dataset_.extend(result)

        for video in tqdm(video_list, desc='Processing {} frames'.format(train_or_test)):
            result = process_video(video)
            dataset_.extend(result)
        
        if mode == 'preprocess':
            with open(os.path.join(pkl_dir, 'openmplposer_{}.pkl'.format(train_or_test)), 'wb') as f:
                pickle.dump(dataset_, f)
            print('Saved {} data in {}'.format(train_or_test, os.path.join(pkl_dir, 'openmplposer_{}.pkl'.format(train_or_test))))
        elif mode == 'mmpose':
            with open(os.path.join(pkl_dir_mmpose, 'openmplposer_{}.pkl'.format(train_or_test)), 'wb') as f:
                pickle.dump(dataset_, f)
            print('Saved mmpose data in {}'.format(os.path.join(pkl_dir_mmpose, 'openmplposer_{}.pkl'.format(train_or_test))))
        
    return 'done'
        
    
def parse_args():
    parser = argparse.ArgumentParser(description='prepare openmplposer dataset')
    parser.add_argument('openmplposer_dir', help='directory of openmplposer')
    parser.add_argument('dataset', help='name of dataset')   # eg annot_same_scene_different_cams, annot_different_scene_same_cams_2test_pose')
    parser.add_argument('--running-modes', nargs='+', default=['preprocess', 'filter', 'mmpose'])
    parser.add_argument('--keypoints-standard', default='h36m', help='standard for 2d keypoints', choices=['h36m', 'coco'])
    parser.add_argument('--run-for-sets', nargs='+', default=['train', 'test'])
    # parser.add_argument('--skip-step-train', type=int, default=2)
    # parser.add_argument('--skip-step-test', type=int, default=6)
    # parser.add_argument('--write-kept-keys', action='store_true')
    # parser.add_argument('--kept-keys-path', default='./filtered_grouping_keys.txt')
    # parser.add_argument('--filter-from-file', action='store_true')
    parser.add_argument('--protocols', nargs='+', default=['yolov8n-pose_protocol_1'], help='list of protocols to run')
    parser.add_argument('--mmpose-dataset-name', default='mmpose')
    parser.add_argument('--mmpose-output-path', default='./mmpose_outputs')
    parser.add_argument('--dont-discard-less-than-5-visible-joints', action='store_true')
    parser.add_argument('--camera-path', default='./cameras', help='path to the camera xml files')
    parser.add_argument('--train-cams', nargs='+', default=[1, 2, 3], type=int, help='list of cameras for training')
    parser.add_argument('--test-cams', nargs='+', default=[1, 2, 3], type=int, help='list of cameras for test')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()
    dir_openmplposer = args.openmplposer_dir
    dataset_name = args.dataset
    running_modes = args.running_modes
    run_for_sets = args.run_for_sets
    # not_run_for_train = args.not_run_for_train
    # not_run_for_test = args.not_run_for_test
    train_cams = args.train_cams
    test_cams = args.test_cams
    # only_run_filter = args.only_run_filter
    # filter_data = args.filter_data
    # train_skip_step = args.skip_step_train
    # test_skip_step = args.skip_step_test
    # write_kept_keys = args.write_kept_keys
    # kept_keys_path = args.kept_keys_path
    mmpose_dataset_name = args.mmpose_dataset_name
    mmpose_output_path = args.mmpose_output_path
    # filter_from_file = args.filter_from_file
    dont_discard_less_than_5_visible_joints = args.dont_discard_less_than_5_visible_joints
    protocols = args.protocols
    
    keypoints_standard = args.keypoints_standard
    pkl_dir_mmpose = None
    camera_path = args.camera_path
    
    print('running on dataset: {}'.format(dataset_name))
    print('keypoints standard: {}'.format(keypoints_standard))
    
    pkl_dir = os.path.join(dir_openmplposer, 'MPL_data', 'datasets', dataset_name)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir, exist_ok=True)
    # pkl_dir_filtered = None
    # if 'filter' in running_modes or 'mmpose' in running_modes:
    #     pkl_dir_filtered = pkl_dir + '_filtered_{}_{}'.format(train_skip_step, test_skip_step)
    #     if not os.path.exists(pkl_dir_filtered):
    #         os.makedirs(pkl_dir_filtered, exist_ok=True)
            
    if 'mmpose' in running_modes:
        pkl_dir_base = os.path.basename(pkl_dir)
        pkl_dir_mmpose = os.path.join(dir_openmplposer, 'MPL_data', 'datasets_mmpose', '{}_{}'.format(pkl_dir_base, mmpose_dataset_name))
        if not os.path.exists(pkl_dir_mmpose):
            os.makedirs(pkl_dir_mmpose, exist_ok=True)
    
    if 'train' in run_for_sets:    
        create_dataset(
            camera_path=camera_path,
            cams=train_cams,
            running_modes=running_modes,
            pkl_dir=pkl_dir,
            dir_openmplposer=dir_openmplposer,
            train_or_test='train',
            pkl_dir_mmpose=pkl_dir_mmpose,
            mmpose_output_path=mmpose_output_path,
            keypoints_standard=keypoints_standard,
            dont_discard_less_than_5_visible_joints=dont_discard_less_than_5_visible_joints,
            protocols=protocols
        )
            
    if 'test' in run_for_sets:
        create_dataset(
            camera_path=camera_path,
            cams=test_cams,
            running_modes=running_modes,
            pkl_dir=pkl_dir,
            dir_openmplposer=dir_openmplposer,
            train_or_test='test',
            pkl_dir_mmpose=pkl_dir_mmpose,
            mmpose_output_path=mmpose_output_path,
            keypoints_standard=keypoints_standard,
            dont_discard_less_than_5_visible_joints=dont_discard_less_than_5_visible_joints,
            protocols=protocols
        )
    
   