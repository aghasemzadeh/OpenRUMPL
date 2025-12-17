import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse
import torch
# from multiviews.triangulate import triangulate_poses

val_categories = [
    "LectureHall_003_wipingchairs1",
    "LectureHall_018_wipingtable1",
    "LectureHall_020_wipingchairs1",
    "ParkingLot1_002_burpee1",
    "ParkingLot1_002_burpee2",
    "ParkingLot1_002_stretching2",
    "ParkingLot1_004_eating2",
    "ParkingLot1_004_phonetalk2",
    "ParkingLot1_004_pushup1",
    "ParkingLot1_004_stretching2",
    "ParkingLot1_004_takingphotos2",
    "ParkingLot1_005_burpeejump1",
    "ParkingLot1_005_overfence2",
    "ParkingLot1_005_pushup1",
    "ParkingLot1_007_burpee2",
    "ParkingLot2_008_burpeejump2",
    "ParkingLot2_008_eating2",
    "ParkingLot2_014_eating2",
    "ParkingLot2_014_overfence1",
    "ParkingLot2_016_phonetalk5",
    "Pavallion_000_yoga1",
    "Pavallion_002_sidebalancerun",
    "Pavallion_002_yoga1",
    "Pavallion_003_yoga1",
    "Pavallion_006_yoga1",
    "Pavallion_013_sidebalancerun",
    "Pavallion_018_sidebalancerun",
    "Pavallion_018_yoga1",
]

img_size = (3008, 4112)

joints_cmu = {
    0: 'neck',
    1: 'nose',
    2: 'root',
    3: 'lsho',
    4: 'lelb',
    5: 'lwri',
    6: 'lhip',
    7: 'lkne',
    8: 'lank',
    9: 'rsho',
    10: 'relb',
    11: 'rwri',
    12: 'rhip',
    13: 'rkne',
    14: 'rank',
    15: 'leye',
    16: 'lear',
    17: 'reye',
    18: 'rear'
}
joints_cmu_inverse_dict = {v: k for k, v in joints_cmu.items()}

joints_h36m = {
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    # 7: 'belly',
    8: 'neck',
    9: 'nose',
    # 10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
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


def extract_cam_param_xml(xml_path='', dtype=np.float32):
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = np.array([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = np.array([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = np.array([cam_center], dtype=dtype)

    k1 = np.array([distortion_vec[0]], dtype=dtype)
    k2 = np.array([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2


def cmu_to_h36m(pose_cmu):
    pose_h36m = np.zeros((17, 4))
    for i in range(17):
        if i == 7 or i == 10:
            continue
        pose_h36m[i] = pose_cmu[joints_cmu_inverse_dict[joints_h36m[i]]]
    pose_cmu[pose_cmu[:, 3] == -1, 3] = np.finfo(float).eps   # change the confidence of missing joints to 0
    head = np.average(pose_cmu[15:19,:3], axis=0, weights=pose_cmu[15:19,3]).reshape((1,3))
    head_conf = np.average(pose_cmu[15:19,3], weights=pose_cmu[15:19,3].astype(bool)).reshape((1,1))
    head = np.hstack([head, head_conf])
    belly = np.average(np.array([pose_cmu[0, :3], pose_cmu[2, :3]]), axis=0, weights=np.array([pose_cmu[0, 3], pose_cmu[2, 3]])).reshape((1,3))
    belly_conf = np.average(np.array([pose_cmu[0, 3], pose_cmu[2, 3]]), weights=np.array([pose_cmu[0, 3], pose_cmu[2, 3]]).astype(bool)).reshape((1,1))
    belly = np.hstack([belly, belly_conf])
    pose_h36m[7] = belly
    pose_h36m[10] = head
    return pose_h36m

def cmu_to_coco(pose_cmu):
    pose_coco = np.zeros((17, 4))
    pose_cmu[pose_cmu[:, 3] == -1, 3] = np.finfo(float).eps   # change the confidence of missing joints to 0
    for i in range(17):
        pose_coco[i] = pose_cmu[joints_cmu_inverse_dict[joints_coco[i]]]
    return pose_coco

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

def error_2d(pose_gt, pose_pred, vis):
    error = np.linalg.norm(pose_gt - pose_pred, axis=1)
    error = error * vis[:,0]
    error = error[error > 0]
    return np.mean(error)

def preprocess_rich_mmpose(args):
    data = args[0]
    dir_rich_mmpose = args[1]
    keypoints_standard = args[2]
    dataset = copy.deepcopy(data)
    json_mmpose = os.path.join(dir_rich_mmpose, 
                                data['image'].split('/')[-3],   # scene
                                data['image'].split('/')[-2],   # camera
                                data['image'].split('/')[-1].replace('.jpeg', '.json'))
    try:
        with open(json_mmpose, 'r') as f:
            mmpose_data = json.load(f)
    except FileNotFoundError:
        print('mmpose file not found: {}'.format(json_mmpose))
        return []
    
    errors_2d_ = []
    for mmpose_person in mmpose_data:
        joints_mmpose = np.array(mmpose_person['keypoints'])
        errors_2d_.append(error_2d(dataset['joints_2d'], joints_mmpose, dataset['joints_vis'][:, 0].reshape(-1,1)))
    
    errors_2d_ = np.array(errors_2d_)
    best_ix = np.argmin(errors_2d_)
    if keypoints_standard == 'h36m':
        joints_2d, joints_2d_conf = mmpose2h36m(np.array(mmpose_data[best_ix]['keypoints']), np.array(mmpose_data[best_ix]['keypoint_scores']))
    elif keypoints_standard == 'coco':
        joints_2d = np.array(mmpose_data[best_ix]['keypoints'])
        joints_2d_conf = np.array(mmpose_data[best_ix]['keypoint_scores']).reshape((-1,1))
    dataset['mmpose_2d'] = True
    dataset['joints_2d'] = joints_2d
    dataset['joints_2d_conf'] = joints_2d_conf
    dataset['keypoint_standard'] = keypoints_standard
    
    return dataset

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
    

def process_frame(args):
    frame_info = args['frame_info']
    annot_info = args['annot_info']
    vertices = args['vertices']
    jregressor = args['jregressor']
    calib_dir = args['calib_dir']
    keypoints_standard = args['keypoints_standard']
    dont_discard_less_than_5_visible_joints = args.get('dont_discard_less_than_5_visible_joints', True)
    # print('dont_discard_less_than_5_visible_joints: {}'.format(dont_discard_less_than_5_visible_joints))
    # with open(annot_info, 'rb') as f:
    #     # bdata = pickle.load(f)
    #     vertices = pickle.load(f)['vertices'].detach().cpu()
        # del bdata
    # vertices = bdata['vertices'].detach().cpu()
    # return {}
    joints = torch.einsum('bik,ji->bjk', [vertices, jregressor]).detach().cpu().numpy()
    # print('frame_info: {}'.format(frame_info))
    SCENE_NAME = frame_info.split('_')[0]
    CAMERA_ID = int(frame_info.split('/')[-1].split('_')[1].split('.')[0])
    calib_path = os.path.join(calib_dir, SCENE_NAME, 'calibration', f'{CAMERA_ID:03d}.xml')
    dtype = np.float32
    focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2 \
                    = extract_cam_param_xml(xml_path=calib_path, dtype=dtype)
    
    K = np.array([[focal_length_x, 0, center[0, 0]], [0, focal_length_y, center[0, 1]], [0, 0, 1]])
    R = rotation
    T = cam_center.reshape((3,1))
    t = translation.reshape((3,1))
    distCoef = np.array([0, 0, 0, 0, 0])
    fx = focal_length_x
    fy = focal_length_y
    cx = center[0, 0]
    cy = center[0, 1]
    
    camera = {
        'K': K,
        'distCoef': distCoef,
        'R': R,
        't': t,
        'T': T,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
    }
    
    # print('joints shape: {}'.format(joints.shape))
    joints_2d = cam_to_image(world_to_cam(joints, R, t), K)[0]
    # print('joints_2d shape: {}'.format(joints_2d.shape))
    joints_3d = joints[0, :, :3]
    joints_2d = joints_2d.reshape(-1, 2)
    invis = (joints_2d[:,0] < 0 ) + (joints_2d[:,0] > img_size[0]) + (joints_2d[:,1] < 0) + (joints_2d[:,1] > img_size[1])
    vis = 1 - invis
    
    # if number of visible joints is less than 5, discard
    if np.sum(vis) < 5 and not dont_discard_less_than_5_visible_joints:
        print('not enough visible keypoints: {}'.format(np.sum(vis)))
        return {}
    vis = np.repeat(vis.reshape(-1,1), 3, axis=1)   # for compatibility with h36m
    
    # skel_camera = world_to_cam(skel[None, :, 0:3], R, t)[0]
    # box = _infer_box(skel_camera, camera, 2)
    
    max_x = np.max(joints_2d[vis[:,0] == True, 0], initial=0)
    min_x = np.min(joints_2d[vis[:,0] == True, 0], initial=0)
    max_y = np.max(joints_2d[vis[:,0] == True, 1], initial=0)
    min_y = np.min(joints_2d[vis[:,0] == True, 1], initial=0)
    box = np.array([min_x, min_y, max_x, max_y])
    
    
    center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
    
    dataset = {
        'image': frame_info,
        'joints_2d': joints_2d,
        'joints_vis': vis,
        'joints_3d': joints_3d,
        'source': 'rich',
        'subject': 0,
        'pose_id': frame_info.split('/')[0],
        'image_id': frame_info.split('/')[-1].replace('.jpeg', ''),
        'camera_id': CAMERA_ID,
        'center': center,
        'scale': scale,
        'box': box,
        'camera': camera,
    }
    return dataset

def get_key_str(datum):
    return 's_{}_frameid_{}'.format(
        datum['pose_id'],
        datum['image_id'].split('_')[0])
        
def filter_group(data, views, skip_step):
    grouping = {}
    nitems = len(data)
    CAM_IX = {x: i for i, x in enumerate(views)}
    # print('views: {}'.format(views))
    # print('CAM_IX: {}'.format(CAM_IX))
    for i in range(nitems):
        keystr = get_key_str(data[i])
        # print('keystr: {}'.format(keystr))
        camera_id = CAM_IX[data[i]['camera_id']]
        if keystr not in grouping:
            grouping[keystr] = [-1] * len(views)
        grouping[keystr][camera_id] = i
        
    # print('grouping: {}'.format(grouping))
    # print('grouping: {}'.format(len(grouping)))
    
    # filtered_grouping = []
    # for _, v in grouping.items():
    #     if np.all(np.array(v) != -1):
    #         filtered_grouping.append(v)
    
    filtered_grouping = [v for _, v in grouping.items()]
    
    filtered_grouping = filtered_grouping[::skip_step]
    return filtered_grouping

def check_if_file_in_keys(args):
    data = args[0]
    keys = args[1]
    # keys = remaining_keys
    image = data['image']
    for j in keys:
        if j in image:
            return True
    return False

def get_avg_depth(depth, x, y, image_size):
    x, y = round(x), round(y)
    x_min, x_max = max(0, x - 2), min(image_size[1] - 1, x + 2)
    y_min, y_max = max(0, y - 2), min(image_size[0] - 1, y + 2)
    
    return np.mean(depth[y_min:y_max+1, x_min:x_max+1])

def process_depth(args):
    data = args[0]
    dir_cmu_mmpose_depth = args[1]
    depth_file = os.path.join(dir_cmu_mmpose_depth, data['image'].replace('.jpeg', '.pkl'))
    try:
        # depth_vals = np.load(depth_file)
        with open(depth_file, 'rb') as f:
            depth_vals = pickle.load(f)
        camera = data['camera']
        focal_length_real = (camera['fx'] + camera['fy']) / 2
        depth_real = (depth_vals['focallength_px'] / focal_length_real) * depth_vals['depth']
    except FileNotFoundError:
        print('depth file not found: {}'.format(depth_file))
        return []
    image_size = depth_real.shape
    joints_2d = data['joints_2d'].copy()
    # joints_2d[:, 0] = np.clip(joints_2d[:, 0], 0, image_size[1] - 1)
    # joints_2d[:, 1] = np.clip(joints_2d[:, 1], 0, image_size[0] - 1)
    
    # depth = [depth_real[round(joints_2d[i][1]), round(joints_2d[i][0])] for i in range(len(joints_2d))]

    joints_2d[:, 0] = np.clip(joints_2d[:, 0], 2, image_size[1] - 3)
    joints_2d[:, 1] = np.clip(joints_2d[:, 1], 2, image_size[0] - 3)
    # Compute the average depth for each joint
    depth = [get_avg_depth(depth_real, joints_2d[i][0], joints_2d[i][1], image_size) for i in range(len(joints_2d))]
    data_ = copy.deepcopy(data)
    data_['depth'] = depth
    return data_
    
   
def create_dataset(running_modes, 
                   skip_step, 
                   pkl_dir, 
                   pkl_dir_filtered, 
                   dir_rich, 
                   train_or_val, 
                   write_kept_keys, 
                   kept_keys_path, 
                   pkl_dir_filtered_mmpose, 
                   mmpose_output_path, 
                   keypoints_standard, 
                   filter_from_file, 
                   dont_discard_less_than_5_visible_joints,
                   regresor_path=None,
                   calib_dir='data/scan_calibration',
                   pkl_dir_filtered_mmpose_depth=None,
                   depth_output_path=None):
    dir_rich_img = os.path.join(dir_rich, 'images', 'val' if train_or_val=='validation' else 'train')
    dir_rich_annot = os.path.join(dir_rich, 'annotations', 'val_body_smplh' if train_or_val=='validation' else 'train_body_smplh')
    cams = list(range(0, 8))
    jregressor = torch.Tensor(np.load(regresor_path))
    # print('jregressor loaded from {}'.format(regresor_path))
    # print('running on {}'.format(train_or_val))
    # print('running modes: {}'.format(running_modes))
    
    if 'preprocess' in running_modes:
        
        cats = os.listdir(dir_rich_img)
        cats_2 = os.listdir(dir_rich_annot)
        assert set(cats) == set(cats_2)
        # print('cams: {}'.format(cams))
        
        frame_list = []
        for cat in tqdm(cats, desc='Fetching {} frames'.format(train_or_val)):
            # if len(frame_list) >= 5:
            #     break
            for cam in cams:
                try:
                    for frame in tqdm(os.listdir(os.path.join(dir_rich_img, cat, 'cam_{0:02d}'.format(cam))), desc='Fetching in {} cam {}'.format(cat, cam)):
                        frame_ix = frame.split('_')[0]
                        scene_ix = cat.split('_')[1]
                        annot_file = os.path.join(dir_rich_annot, cat, frame_ix, '{}.pkl'.format(scene_ix))
                        if frame.endswith('.jpeg') and os.path.isfile(annot_file):
                            # print('found')
                            f = open(annot_file, 'rb')
                            vertices = pickle.load(f)['vertices'].detach().cpu().clone()
                            f.close()
                            frame_list.append({
                                'frame_info': os.path.join(cat, 'cam_{0:02d}'.format(cam), frame),
                                'annot_info': annot_file,
                                'vertices': vertices,
                                'jregressor': jregressor,
                                'keypoints_standard': keypoints_standard,
                                'calib_dir': calib_dir,
                                'dont_discard_less_than_5_visible_joints': dont_discard_less_than_5_visible_joints
                                })
                            # del vertices
                except Exception as e:
                    print(e)
                    continue
                except FileNotFoundError:
                    continue
        # frame_list.sort(key=lambda x: x['frame_info'])
        print('Total frames to process: {}'.format(len(frame_list)))
        
        dataset_ = []
        # dataset_.append(process_frame(frame_list[0]))
        # dataset_.append(process_frame(frame_list[1]))
        for result in tqdm(map(process_frame, frame_list), total=len(frame_list), desc='Processing {} frames'.format(train_or_val)):
            if result != {}:
                dataset_.append(result)
        # raise
        # pool = Pool()
        # for result in tqdm(pool.imap(process_frame, frame_list), total=len(frame_list), desc='Processing {} frames'.format(train_or_val)):
        #     if result != {}:
        #         dataset_.append(result)
        
        with open(os.path.join(pkl_dir, 'rich_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_, f)
        print('saved data in {}'.format(os.path.join(pkl_dir, 'rich_{}.pkl'.format(train_or_val))))
        
        
    # --------- filter the data based on the requested frames
    if 'filter' in running_modes:
        if 'preprocess' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir, 'rich_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir, 'rich_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_ = pickle.load(f)
                
        if filter_from_file:
            with open(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val)), 'r') as f:
                keys = f.readlines()
            keys = [x.strip() for x in keys]
            args = [(x, keys) for x in dataset_]
            remaining_idx = []
            pool = Pool()
            for result in tqdm(pool.imap(check_if_file_in_keys, args), total=len(dataset_), desc='filtering {} data'.format(train_or_val)):
                if result:
                    remaining_idx.append(True)
                else:
                    remaining_idx.append(False)
                    
            dataset_filtered = [x for x, y in zip(dataset_, remaining_idx) if y]
        else:
            remaining_idx = filter_group(dataset_, cams, skip_step)
            
            remaining_keys = []
            dataset_filtered = []
            if write_kept_keys:
                f = open(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val)), 'w')
            
            for k in tqdm(remaining_idx, desc='filtering {} data'.format(train_or_val)):
                for i in k:
                    if i == -1:
                        continue
                    if write_kept_keys:
                        f.write(dataset_[i]['image'])   
                    remaining_keys.append(dataset_[i]['image'])
                    dataset_filtered.append(dataset_[i])
                    if write_kept_keys:
                        f.write('\n')
            if write_kept_keys:
                f.close()
            
        print('\n filter keys stored in {}'.format(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val))))
                
        print('{} dataset filtered: {}, org {} data: {}'.format(train_or_val, train_or_val, len(dataset_filtered), len(dataset_)))
        with open(os.path.join(pkl_dir_filtered, 'rich_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered, f)
        
    # --------- replace 2D joints with mmpose detections
    if 'mmpose' in running_modes:
        if 'filter' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir_filtered, 'rich_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir_filtered, 'rich_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_filtered = pickle.load(f)
        
        mmpose_output_path = os.path.join(mmpose_output_path, 'val' if train_or_val=='validation' else 'train')
        dataset_filtered_mmpose = []
        mmpose_output_path_list = [mmpose_output_path] * len(dataset_filtered)
        keypoints_standard_list = [keypoints_standard] * len(dataset_filtered)
        pool = Pool()
        for result in tqdm(pool.imap(preprocess_rich_mmpose, zip(dataset_filtered, mmpose_output_path_list, keypoints_standard_list)), total=len(dataset_filtered), desc='preprocessing {} mmpose data'.format(train_or_val)):
            dataset_filtered_mmpose.append(result)
        
        dataset_filtered_mmpose = [x for x in dataset_filtered_mmpose if len(x) > 0]
        print('{} data mmpose: {}, org {} data: {}'.format(train_or_val, len(dataset_filtered_mmpose), train_or_val, len(dataset_filtered)))
        with open(os.path.join(pkl_dir_filtered_mmpose, 'rich_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered_mmpose, f)
        print('saved mmpose data in {}'.format(os.path.join(pkl_dir_filtered_mmpose, 'rich_{}.pkl'.format(train_or_val))))
        
    # --------- add depth information
    if 'depth' in running_modes:
        if 'mmpose' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir_filtered_mmpose, 'rich_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir_filtered_mmpose, 'rich_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_filtered_mmpose = pickle.load(f)
                
        depth_output_path = os.path.join(depth_output_path, 'val' if train_or_val=='validation' else 'train')
        dataset_filtered_mmpose_depth = []
        depth_output_path_list = [depth_output_path] * len(dataset_filtered_mmpose)
        
        pool = Pool()
        for result in tqdm(pool.imap(process_depth, zip(dataset_filtered_mmpose, depth_output_path_list)), total=len(dataset_filtered_mmpose), desc='preprocessing {} depth data'.format(train_or_val)):
            dataset_filtered_mmpose_depth.append(result)
            
        dataset_filtered_mmpose_depth = [x for x in dataset_filtered_mmpose_depth if len(x) > 0]
        print('{} data depth: {}, org {} data: {}'.format(train_or_val, len(dataset_filtered_mmpose_depth), train_or_val, len(dataset_filtered_mmpose)))
        with open(os.path.join(pkl_dir_filtered_mmpose_depth, 'rich_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered_mmpose_depth, f)
        print('saved depth data in {}'.format(os.path.join(pkl_dir_filtered_mmpose_depth, 'rich_{}.pkl'.format(train_or_val))))
        
    return 'done'
        
    
def parse_args():
    parser = argparse.ArgumentParser(description='prepare rich dataset')
    parser.add_argument('rich_dir', help='directory of the rich dataset')
    parser.add_argument('dataset', help='name of dataset')   
    parser.add_argument('--running-modes', nargs='+', default=['preprocess', 'filter', 'mmpose'])
    parser.add_argument('--keypoints-standard', default='coco', help='standard for 2d keypoints', choices=['h36m', 'coco'])
    parser.add_argument('--run-for-sets', nargs='+', default=['train', 'validation'])
    parser.add_argument('--skip-step-train', type=int, default=2)
    parser.add_argument('--skip-step-val', type=int, default=6)
    parser.add_argument('--write-kept-keys', action='store_true')
    parser.add_argument('--kept-keys-path', default='./filtered_grouping_keys.txt')
    parser.add_argument('--filter-from-file', action='store_true')
    parser.add_argument('--mmpose-dataset-name', default='mmpose')
    parser.add_argument('--mmpose-output-path', default='./mmpose_outputs')
    parser.add_argument('--dont-discard-less-than-5-visible-joints', action='store_true')
    parser.add_argument('--regressor-path', default='/globalscratch/users/a/b/abolfazl/amass_data/support_data/regressors/J_regressor_coco.npy')
    parser.add_argument('--calib-dir', default='/globalscratch/users/a/b/abolfazl/RICH/rich_toolkit/data/scan_calibration')
    parser.add_argument('--depth-dataset-name', default='depth') 
    parser.add_argument('--depth-output-path', default='./depths')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()
    dir_rich = args.rich_dir
    dataset_name = args.dataset
    running_modes = args.running_modes
    run_for_sets = args.run_for_sets
    
    train_skip_step = args.skip_step_train
    val_skip_step = args.skip_step_val
    write_kept_keys = args.write_kept_keys
    kept_keys_path = args.kept_keys_path
    mmpose_dataset_name = args.mmpose_dataset_name
    mmpose_output_path = args.mmpose_output_path
    depth_dataset_name = args.depth_dataset_name
    depth_output_path = args.depth_output_path
    filter_from_file = args.filter_from_file
    dont_discard_less_than_5_visible_joints = args.dont_discard_less_than_5_visible_joints
    
    regresor_path = args.regressor_path
    calib_dir = args.calib_dir
    
    keypoints_standard = args.keypoints_standard
    pkl_dir_filtered_mmpose = None
    
    print('running on dataset: {}'.format(dataset_name))
    print('keypoints standard: {}'.format(keypoints_standard))
    
    pkl_dir = os.path.join(dir_rich, 'PPT_data', 'datasets', dataset_name)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir, exist_ok=True)
    pkl_dir_filtered = None
    if 'filter' in running_modes or 'mmpose' in running_modes or 'depth' in running_modes:
        pkl_dir_filtered = pkl_dir + '_filtered_{}_{}'.format(train_skip_step, val_skip_step)
        if not os.path.exists(pkl_dir_filtered):
            os.makedirs(pkl_dir_filtered, exist_ok=True)
            
    if 'mmpose' in running_modes or 'depth' in running_modes:
        pkl_dir_filtered_base = os.path.basename(pkl_dir_filtered)
        pkl_dir_filtered_mmpose = os.path.join(dir_rich, 'PPT_data', 'datasets_mmpose', '{}_{}'.format(pkl_dir_filtered_base, mmpose_dataset_name))
        if not os.path.exists(pkl_dir_filtered_mmpose):
            os.makedirs(pkl_dir_filtered_mmpose, exist_ok=True)
            
    if 'depth' in running_modes:
        pkl_dir_filtered_mmpose_base = os.path.basename(pkl_dir_filtered_mmpose)
        pkl_dir_filtered_mmpose_depth = os.path.join(dir_rich, 'PPT_data', 'datasets_mmpose_depth', '{}_{}'.format(pkl_dir_filtered_mmpose_base, depth_dataset_name))
        if not os.path.exists(pkl_dir_filtered_mmpose_depth):
            os.makedirs(pkl_dir_filtered_mmpose_depth, exist_ok=True)
    
    if 'train' in run_for_sets:    
        create_dataset(running_modes, 
                       train_skip_step, 
                       pkl_dir, 
                       pkl_dir_filtered, 
                       dir_rich, 
                       'train', 
                       write_kept_keys, 
                       kept_keys_path, 
                       pkl_dir_filtered_mmpose, 
                       mmpose_output_path, 
                       keypoints_standard, 
                       filter_from_file, 
                       dont_discard_less_than_5_visible_joints,
                       regresor_path=regresor_path,
                       calib_dir=calib_dir,
                       pkl_dir_filtered_mmpose_depth=pkl_dir_filtered_mmpose_depth,
                       depth_output_path=depth_output_path
                       )
    if 'validation' in run_for_sets:
        create_dataset(running_modes, 
                       val_skip_step, 
                       pkl_dir, 
                       pkl_dir_filtered, 
                       dir_rich, 
                       'validation', 
                       write_kept_keys, 
                       kept_keys_path, 
                       pkl_dir_filtered_mmpose, 
                       mmpose_output_path, 
                       keypoints_standard, 
                       filter_from_file, 
                       dont_discard_less_than_5_visible_joints,
                       regresor_path=regresor_path,
                       calib_dir=calib_dir,
                       pkl_dir_filtered_mmpose_depth=pkl_dir_filtered_mmpose_depth,
                       depth_output_path=depth_output_path
                       )
    
   