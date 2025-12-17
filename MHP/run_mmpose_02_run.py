import os
from os import path as osp
import numpy as np
import torch
from tqdm import tqdm
import pickle
import trimesh
import argparse
import glob 
from body_visualizer.tools.vis_tools import colors, imagearray2file
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.tools.omni_tools import log2file, makepath
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
# from amass.data.prepare_data import prepare_amass
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mmpose.apis import MMPoseInferencer
from utils import *
from multiviews.triangulate import triangulate_poses
import ultralytics
from ultralytics import YOLO
import imageio
import time
import depth_pro

# support_dir = '/data/Git_Repo/TransMoCap/support_data'
# support_dir = '/globalscratch/users/a/b/abolfazl/amass_data/support_data'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

amass_splits = {
    'validation': ['CMU'],
    'test': ['CMU'],
    'train': ['Eyes_Japan_Dataset',
             'ACCAD',
             'DFaust_67',
             'BMLhandball',
             'BioMotionLab_NTroje',
             'SFU',
             'Transitions_mocap',
             'TCD_handMocap',
             'TotalCapture',
             'KIT',
             'MPI_HDM05',
             'HumanEva',
             'MPI_mosh',
             'BMLmovi',
             'SOMA',
             'MPI_Limits',
             'WEIZMANN',
             'EKUT',
             'SSM_synced',
             'GRAB',
             'DanceDB',
             'HUMAN4D',
             'CNRS'
             ]
}

# get amass train split from input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--dataset-split-number', default=0, type=int, help='Dataset split')
    parser.add_argument('--support-dir', default='amass_data/support_data', type=str, help='Support directory of amass')
    parser.add_argument('--amass-data-dir', default='amass_data/amass_data_poses', type=str, help='PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz')
    parser.add_argument('--work-dir', default='amass_data/prepared_data/', type=str, help='Work directory to save the data')
    parser.add_argument('--train-datasets', default=None, nargs='+', type=str, help='Train datasets')
    parser.add_argument('--valid-datasets', default=None, nargs='+', type=str, help='Valid datasets')
    parser.add_argument('--exp', default='amass', type=str, help='Experiment name')
    parser.add_argument('--extra-name', default=None, type=str, help='Extra name for the experiment')
    parser.add_argument('--use-cams-from', default='h36m', type=str, help='Use cameras from',
                        choices=['h36m', 'cmu', 'openmplposer', 'dome', 'rich'])
    parser.add_argument('--calib-file-h36m', default=None, type=str, help='Camera calibration file')
    parser.add_argument('--calib-root-cmu', default=None, type=str, help='Camera calibration file')
    parser.add_argument('--calib-path-openmplposer', default=None, type=str, help='Camera calibration path')
    parser.add_argument('--actors-h36m', default=[1, 5, 6, 7, 8, 9, 11], nargs='+', type=int, help='Actors to use')
    parser.add_argument('--calibs-cmu', default=['171204_pose5', '171204_pose6'], nargs='+', type=str, help='Calibrations to use')
    parser.add_argument('--calib-file-dome', default=None, type=str, help='Camera calibration file')
    parser.add_argument('--views-cmu', default=[3, 6, 12, 13, 23], nargs='+', type=int, help='Views to use')
    parser.add_argument('--calib-file-rich', default=None, type=str, help='Camera calibration file')
    parser.add_argument('--scenes-rich', default=['BBQ', 'Gym', 'LectureHall', 'ParkingLot1', 'ParkingLot2', 'Pavallion'], nargs='+', type=str, help='Scenes to use')
    parser.add_argument('--room-size', default=[-1.0, 1.0, -1.0, 1.0, 0.0, 0.0], nargs='+', type=float, help='Room size [min_x, max_x, min_y, max_y, min_z, max_z]')
    parser.add_argument('--operation-on', default=['train', 'validation'], nargs='+', type=str, help='Operations on')
    parser.add_argument('--image-width', default=1000, type=int, help='Image width')
    parser.add_argument('--image-height', default=1000, type=int, help='Image height')
    parser.add_argument('--apply-rotation', default=False, action='store_true', help='Apply rotation')
    parser.add_argument('--apply-joint-clip', default=False, action='store_true', help='Apply joint clip')
    parser.add_argument('--n-frames', default=-1, type=int, help='Number of frames to use')
    parser.add_argument('--fit-mmpose-to-amass', default=False, action='store_true', help='Fit skeleton to AMASS')
    parser.add_argument('--fit-using-most-aligned', default=False, action='store_true', help='Fit using all joints or only the most aligned ones')
    parser.add_argument('--regressor', default='h36m', type=str, help='Regressor to use', choices=['h36m', 'coco', 'both'])
    parser.add_argument('--triangulate', default=False, action='store_true', help='Triangulate the mmpose keypoints')
    parser.add_argument('--triangulate-th', default=0.85, type=float, help='Triangulation threshold')
    parser.add_argument('--pose2d-model', default='human', type=str, help='2D pose model to use for mmpose')
    parser.add_argument('--views-dome', default=None, nargs='+', type=int, help='Views to use')
    parser.add_argument('--run-on-every-nth-frame', default=-1, type=int, help='Run on every nth frame')
    parser.add_argument('--save-temp-checkpoints', default=False, action='store_true', help='Save temporary checkpoints')
    parser.add_argument('--run-on-random-cameras', default=False, action='store_true', help='Run on random cameras')
    parser.add_argument('--camera-location-limit', default=[-4.0, 4.0, -4.0, 4.0, 1.0, 4.0], nargs='+', type=float, help='Camera location limit')
    parser.add_argument('--n-cameras-per-person', default=10, type=int, help='Number of cameras per person')
    parser.add_argument('--camera-location-outside-room', default=False, action='store_true', help='Camera location outside room (range of room size)')
    parser.add_argument('--camera-dist-from-person', default=0, type=float, help='Camera distance from person. if 0, this feature is disabled')
    parser.add_argument('--dont-filter-outside-image-when-triangulating', default=False, action='store_true', help='Dont Filter joints outside image when triangulating (useful for rich dataset)')
    parser.add_argument('--room-center', default=None, nargs='+', type=float, help='Room center [x, y, z]')
    parser.add_argument('--look-at-person', default=False, action='store_true', help='Look at person')
    parser.add_argument('--sphere-camera', default=False, action='store_true', help='Use sphere to place cameras for validation purposes')
    parser.add_argument('--sphere-radius', default=6.0, type=float, help='Sphere radius')
    parser.add_argument('--sphere-z-start', default=1.0, type=float, help='Sphere z start')
    parser.add_argument('--sphere-z-end', default=4.0, type=float, help='Sphere z end')
    parser.add_argument('--sphere-angle-step', default=10, type=int, help='Sphere angle step')
    parser.add_argument('--sphere-end-angle', default=180, type=int, help='Sphere end angle')
    parser.add_argument('--save-images', default=False, action='store_true', help='Save images')
    parser.add_argument('--image-save-dir', default='/globalscratch/users/a/b/abolfazl/amass/', type=str, help='Image save directory')
    parser.add_argument('--save-depth-maps', default=False, action='store_true', help='Save depth maps')
    return parser.parse_args()


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas=16):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        
        data =  {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]
        data['trans'] = data['trans']

        return data


actual_joints = {
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

keypoints_most_aligned_with_h36m = [
    2,  # rkne
    3,  # rank
    5,  # lkne
    9,  # nose
    11, # lsho
    12, # lelb
    13, # lwri
    14, # rsho
    15, # relb
    16, # rwri
]

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


def save_dataset(
                dataset_split_number=0,
                subset=None,
                work_dir=None,
                bm=None,
                faces=None,
                J_regressor=None,
                regressor='h36m',
                num_betas=None,
                cameras=None,
                views=None,
                n_camera_setups=1,
                imw=1000,
                imh=1000,
                room_min_x=-1.0,
                room_max_x=1.0,
                room_min_y=-1.0,
                room_max_y=1.0,
                room_min_z=0.0,
                room_max_z=0.0,
                calibs=[9, 11],
                h36m_or_cmu='h36m',
                use_cams_from='h36m',
                apply_rotation=False,
                apply_joint_clip=False,
                number_of_frames=-1,
                fit_using_most_aligned=False,
                fit_mmpose_to_amass=False,
                extra_name=None,
                triangulate=False,
                triangulate_th=0.85,
                pose2d_model='human',
                run_on_every_nth_frame=-1,
                save_temp_checkpoints=False,
                run_on_random_cameras=False,
                camera_location_limit=[-4.0, 4.0, -4.0, 4.0, 1.0, 4.0],
                n_cameras_per_person=10,
                camera_location_outside_room=False,
                camera_dist_from_person=0,
                filter_outside_image_when_triangulating=True,
                room_center=None,
                look_at_person=False,
                sphere_camera=False,
                sphere_radius=None,
                sphere_z_start=None,
                sphere_z_end=None,
                sphere_angle_step=None,
                sphere_end_angle=None,
                save_images=False,
                image_save_dir='./images',
                save_depth_maps=False,
                ):
    calibs_used = '_'.join([str(a) for a in calibs])
    if fit_mmpose_to_amass:
        file_path = work_dir + '/stage_V/' + subset + '/split_{}_{}_amass_mmpose_joints_{}_{}_{}_calibs_{}_{}_fit_mmpose_to_amass_{}_{}_{}_{}_{}_{}.pkl'.format(
            dataset_split_number,
            extra_name if extra_name is not None else '',
            subset, 
            h36m_or_cmu, 
            regressor,
            calibs_used, 
            'rotated' if apply_rotation else 'not_rotated', 
            'joint_clip' if apply_joint_clip else 'no_joint_clip', 
            'fit_using_most_aligned' if fit_using_most_aligned else 'fit_using_all',
            'triangulated' if triangulate else 'not_triangulated',
            'random_cameras_{}'.format(n_cameras_per_person) if run_on_random_cameras else '',
            'run_every_{}'.format(run_on_every_nth_frame) if run_on_every_nth_frame != -1 else '',
            'numfram_{}'.format(number_of_frames) if number_of_frames != -1 else '',
            )
    else:
        file_path = work_dir + '/stage_V/' + subset + '/split_{}_{}_amass_mmpose_joints_{}_{}_{}_calibs_{}_no_fit_{}_{}_{}_{}_{}.pkl'.format(
            dataset_split_number,
            extra_name if extra_name is not None else '',
            subset, 
            h36m_or_cmu, 
            regressor,
            calibs_used, 
            'rotated' if apply_rotation else 'not_rotated', 
            'triangulated' if triangulate else 'not_triangulated',
            'run_every_{}'.format(run_on_every_nth_frame) if run_on_every_nth_frame != -1 else '',
            'random_cameras_{}'.format(n_cameras_per_person) if run_on_random_cameras else '',
            'numfram_{}'.format(number_of_frames) if number_of_frames != -1 else '',
            )
    if os.path.exists(file_path):
        print('File already exists:', file_path)
        return
    highest_id_done = -1
    if save_temp_checkpoints:
        temp_dir = file_path.replace('.pkl', '_temp_files')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        list_of_done_files = os.listdir(temp_dir)
        list_of_done_files = [f for f in list_of_done_files if f.endswith('.pkl')]
        ids_done_files = [int(f.split('_')[-1].replace('.pkl', '')) for f in list_of_done_files]
        highest_id_done = max(ids_done_files) if len(ids_done_files) > 0 else -1
        
    split_dir = os.path.join(work_dir, 'stage_III', subset)
    
    

    ds = AMASS_DS(dataset_dir=split_dir, num_betas=num_betas)
    ds = torch.load(os.path.join(work_dir, 'stage_IV', subset, 'amass_ds_{}.pt'.format(dataset_split_number)))
    print('Train split has %d datapoints.'%len(ds))

    batch_size = 1
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=5)
    # prepare dataset
    joints_3d_all = []
    joints_2d_mmpose_all = []
    confs_2d_mmpose_all = []
    joints_2d_amass_all = []
    confs_2d_mmpose_tri_all = []
    triangulated_3d_mmpose_all = []
    camera_setup_used = []
    views_all = []
    joints_3d_h36m_all = []
    joints_2d_mmpose_h36m_all = []
    confs_2d_mmpose_h36m_all = []
    joints_2d_amass_h36m_all = []
    camera_parameters_all = []
    body_depth_all_gt = []
    body_depth_all_mmpose = []
    image_name_all = []
    
    if save_temp_checkpoints and highest_id_done != -1:
        print('Continuing from index:', highest_id_done)
        with open(os.path.join(temp_dir, 'temp_{}.pkl'.format(highest_id_done)), 'rb') as f:
            temp_data = pickle.load(f)
            joints_3d_all = temp_data['joints_3d']
            joints_2d_mmpose_all = temp_data['joints_2d_mmpose']
            confs_2d_mmpose_all = temp_data['confs_2d_mmpose']
            joints_2d_amass_all = temp_data['joints_2d_amass']
            triangulated_3d_mmpose_all = temp_data['triangulated_3d_mmpose']
            joints_3d_h36m_all = temp_data['joints_3d_h36m']
            joints_2d_mmpose_h36m_all = temp_data['joints_2d_mmpose_h36m']
            confs_2d_mmpose_h36m_all = temp_data['confs_2d_mmpose_h36m']
            joints_2d_amass_h36m_all = temp_data['joints_2d_amass_h36m']
            camera_setup_used = temp_data['camera_setup_used']
            views_all = temp_data['views_used']
            camera_parameters_all = temp_data['camera_parameters_all']
            body_depth_all_gt = temp_data['body_depth_all_gt']
            body_depth_all_mmpose = temp_data['body_depth_all_mmpose']
            image_name_all = temp_data['image_name_all']
            # random.setstate(temp_data['random_seed_state'])
            np.random.set_state(temp_data['numpy_random_seed_state'])
            torch.set_rng_state(temp_data['torch_random_seed_state'])
        f.close()
        
    if save_depth_maps:
        # model_depth, transform_depth = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
        model_depth, transform_depth = depth_pro.create_model_and_transforms(device=comp_device)
        model_depth.eval()
        
    #### cameras is initially a dict of lists (view)(camera_setup) --> change it to list of lists (camera_setup)(view)
    cameras_tri = []
    if triangulate and not run_on_random_cameras and not sphere_camera:
        for camera_setup in range(n_camera_setups):
            cameras_tri.append([])
            for i, view in enumerate(views):
                cameras_tri[-1].append(cameras[view][camera_setup])
    if regressor == 'both':
        jregressor = [torch.Tensor(np.load(jr)) for jr in J_regressor]
    else:
        jregressor = torch.Tensor(np.load(J_regressor))
    if not run_on_random_cameras and not sphere_camera:
        if fit_mmpose_to_amass:
            mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
            mv.set_cam_trans(trans=[0, 0, 0])
        else:
            mesh_viewers = []
            for camera_setup in range(n_camera_setups):
                mesh_viewers.append({})
                for view in views:
                    fx = cameras[view][camera_setup]['fx']
                    fy = cameras[view][camera_setup]['fy']
                    cx = cameras[view][camera_setup]['cx']
                    cy = cameras[view][camera_setup]['cy']
                    mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                    # mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
                    mv.set_cam_trans(trans=[0, 0, 0])
                    if h36m_or_cmu == 'openmplposer':
                        camera_pose = cameras[view][camera_setup]['camera_pose']
                        K_ = cameras[view][camera_setup]['K_']
                        # mv.updateCam(camera_pose, K_)
                        mv.camera_node.camera.fx = K_[0, 0]
                        mv.camera_node.camera.fy = K_[1, 1]
                        mv.camera_node.camera.cx = K_[0, 2]
                        mv.camera_node.camera.cy = K_[1, 2]
                        mv.scene.set_pose(mv.camera_node, pose=camera_pose)
                    mesh_viewers[-1][view] = {
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy,
                        'mv': mv,
                    }
        # pass
    
    if 'yolo' in pose2d_model:
        yolo_inferencer = YOLO(f'{pose2d_model}.pt')
        yolo_inferencer.to(0)
    else:
        # mmpose_inferencer = MMPoseInferencer('human', device='cuda:0')
        mmpose_inferencer = MMPoseInferencer(pose2d_model, device='cuda:0')

    #### deal with random cameras
    if run_on_random_cameras:
        views_original_dataset = copy.deepcopy(views)
        views = range(1, n_cameras_per_person+1)
        
        mesh_viewers = []
        for camera_setup in range(n_camera_setups):
            mesh_viewers.append({})
            for view in views:
                while True:     # make sure the camera is not empty
                    view_ = np.random.choice(views_original_dataset)
                    if 'fx' in cameras[view_][camera_setup]:
                        break
                # view_ = np.random.choice(views_original_dataset)
                fx = cameras[view_][camera_setup]['fx']
                fy = cameras[view_][camera_setup]['fy']
                cx = cameras[view_][camera_setup]['cx']
                cy = cameras[view_][camera_setup]['cy']
                if cx > cy:
                    mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                else:
                    mv = MeshViewer(width=imh, height=imw, use_offscreen=True, fx=fy, fy=fx, cx=cx, cy=cy)
                mv.set_cam_trans(trans=[0, 0, 0])
                mesh_viewers[-1][view] = {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'mv': mv,
                }
    elif sphere_camera:
        views_original_dataset = copy.deepcopy(views)
        camera_setup = 0
        view_count = 0
        while True:     # make sure the camera is not empty
            view_ = views_original_dataset[view_count]
            if 'fx' in cameras[view_][camera_setup]:
                break
            view_count += 1
            if view_count == len(views_original_dataset):
                raise ValueError('No camera found')
            
        cameras_sphere = get_cameras_on_sphere(radius=sphere_radius, z_start=sphere_z_start, z_end=sphere_z_end, angle_step=sphere_angle_step, end_angle=sphere_end_angle)
        views = range(0, len(cameras_sphere))
        
        
        mesh_viewers = []
        for camera_setup in range(1):
            mesh_viewers.append({})
            for view in views:
                fx = cameras[view_][camera_setup]['fx']
                fy = cameras[view_][camera_setup]['fy']
                cx = cameras[view_][camera_setup]['cx']
                cy = cameras[view_][camera_setup]['cy']
                if cx > cy:
                    mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                else:
                    mv = MeshViewer(width=imh, height=imw, use_offscreen=True, fx=fy, fy=fx, cx=cx, cy=cy)
                mv.set_cam_trans(trans=[0, 0, 0])
                mesh_viewers[-1][view] = {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'mv': mv,
                }
    for ix, bdata in tqdm(enumerate(dataloader), total=len(dataloader), desc='Processing dataset {} {} split {}'.format(subset, 
                                                                                                                        extra_name if extra_name is not None else '',
                                                                                                                        dataset_split_number)):
        if number_of_frames != -1 and ix > number_of_frames:
            break
        
        if run_on_every_nth_frame != -1 and ix % run_on_every_nth_frame != 0:
            continue
        
        if ix <= highest_id_done:
            print('Skipping index:', ix)
            continue
        # if ix < 10:
        #     continue
        # elif ix > 10:
        #     break
        body_v = bm.forward(**{k:v.to(comp_device) for k,v in bdata.items() if k in ['pose_body', 'pose_hand', 'betas','root_orient', 'trans']}).v
        # body_v = bm.forward(**{k:v.to(comp_device) for k,v in bdata.items() if k in ['pose_body', 'pose_hand', 'betas','root_orient']}).v
        vertices = torch.Tensor(c2c(body_v))[0]
        if regressor == 'h36m':
            _, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, lower_body_reversed=True)
            root_loc = joints_3d[0, 0]
        elif regressor == 'coco':
            _, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, lower_body_reversed=False)
            root_loc = joints_3d[0, 11:13].mean(0)
        elif regressor == 'both':
            _, joints_3d = amass_vertices_to_joints(vertices[None], jregressor[1],lower_body_reversed=False)
            root_loc = joints_3d[0, 11:13].mean(0)
            
        vertices = locate_mesh_in_room(vertices, room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, room_max_z, h36m_or_cmu=h36m_or_cmu, rotate=apply_rotation, root_loc=root_loc)
        # if vertices[0, 0] > room_max_x or vertices[0, 0] < room_min_x or vertices[0, 1] > room_max_y or vertices[0, 1] < room_min_y or vertices[0, 2] > room_max_z or vertices[0, 2] < room_min_z:
        #     print('Person is outside room:', ix)
        #     continue
        if run_on_random_cameras and use_cams_from != 'rich' or sphere_camera or use_cams_from == 'openmplposer':
            camera_setup_to_use = 0
        else:
            camera_setup_to_use = np.random.randint(0, n_camera_setups)
        camera_setup_used.append(camera_setup_to_use)
        joints_2d_mmpose_all.append([])
        confs_2d_mmpose_all.append([])
        joints_2d_amass_all.append([])
        confs_2d_mmpose_tri_all.append([])
        joints_2d_mmpose_h36m_all.append([])
        confs_2d_mmpose_h36m_all.append([])
        joints_2d_amass_h36m_all.append([])
        views_all.append([])
        camera_parameters_all.append([])
        body_depth_all_gt.append([])
        body_depth_all_mmpose.append([])
        if run_on_random_cameras or sphere_camera:
            cameras_tri = [[]]
        for view_ix, view in enumerate(views):
            if run_on_random_cameras:
                room_size = [room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, room_max_z]
                if look_at_person:
                    room_center[0] = vertices[0,0].item()
                    room_center[1] = vertices[0,1].item()
                R, T, t = random_camera_in_room(camera_location_limit, room_size, camera_location_outside_room, camera_dist_from_person=camera_dist_from_person, person_location=vertices[0], room_center=room_center)
                # view_ = np.random.randint(min(views_original_dataset), max(views_original_dataset)+1)
                # view_ = np.random.choice(views_original_dataset)
                # K = cameras[view_][camera_setup_to_use]['K']
                # fx = cameras[view_][camera_setup_to_use]['fx']
                # fy = cameras[view_][camera_setup_to_use]['fy']
                # cx = cameras[view_][camera_setup_to_use]['cx']
                # cy = cameras[view_][camera_setup_to_use]['cy']
                distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic
                k = np.array([distCoef[0], distCoef[1], distCoef[4]])
                p = np.array([distCoef[2], distCoef[3]])
                fx = mesh_viewers[camera_setup_to_use][view]['fx']
                fy = mesh_viewers[camera_setup_to_use][view]['fy']
                cx = mesh_viewers[camera_setup_to_use][view]['cx']
                cy = mesh_viewers[camera_setup_to_use][view]['cy']
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                mv = mesh_viewers[camera_setup_to_use][view]['mv']
                camera_parameters_all[-1].append({'R': R, 'T': T, 't': t, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p})
                cameras_tri[-1].append({'R': R, 'T': T, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p})
                # if not fit_mmpose_to_amass:
                #     mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                #     mv.set_cam_trans(trans=[0, 0, 0])
                #     # mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
                #     # mv.set_cam_trans(trans=[0, 0, 0])
                
            elif sphere_camera:
                R = cameras_sphere[view]['R']
                T = cameras_sphere[view]['T']
                t = cameras_sphere[view]['t']
                distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic
                k = np.array([distCoef[0], distCoef[1], distCoef[4]])
                p = np.array([distCoef[2], distCoef[3]])
                fx = mesh_viewers[camera_setup_to_use][view]['fx']
                fy = mesh_viewers[camera_setup_to_use][view]['fy']
                cx = mesh_viewers[camera_setup_to_use][view]['cx']
                cy = mesh_viewers[camera_setup_to_use][view]['cy']
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                mv = mesh_viewers[camera_setup_to_use][view]['mv']
                camera_parameters_all[-1].append({'R': R, 'T': T, 't': t, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p,
                                                  'category_height': cameras_sphere[view]['category_height'],
                                                  'category_angle': cameras_sphere[view]['category_angle'],
                                                    })
                cameras_tri[-1].append({'R': R, 'T': T, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p})

            else:
                R = cameras[view][camera_setup_to_use]['R']
                t = cameras[view][camera_setup_to_use]['t']
                T = cameras[view][camera_setup_to_use]['T']
                k = cameras[view][camera_setup_to_use]['k']
                p = cameras[view][camera_setup_to_use]['p']
                K = cameras[view][camera_setup_to_use]['K']
                fx = cameras[view][camera_setup_to_use]['fx']
                fy = cameras[view][camera_setup_to_use]['fy']
                cx = cameras[view][camera_setup_to_use]['cx']
                cy = cameras[view][camera_setup_to_use]['cy']
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                mv = mesh_viewers[camera_setup_to_use][view]['mv']
                camera_parameters_all[-1].append({'R': R, 'T': T, 't': t, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p
                                                    })
                # cameras_tri[-1].append({'R': R, 'T': T, 'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k': k, 'p': p})
                if not fit_mmpose_to_amass:
                    # mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                    # mv.set_cam_trans(trans=[0, 0, 0])
                    # mv = mesh_viewers[camera_setup_to_use][view_ix]
                    mv = mesh_viewers[camera_setup_to_use][view]['mv']
            Rt=np.eye(4)
            ##################
            Rt[:3, :3] = -R
            Rt[:3, 3] = -t.T
            Rt[0, :] = -Rt[0, :]        # flip x axis
            vertices_transformed = np.dot(vertices, Rt[:3, :3].T) + Rt[:3, 3]
            if h36m_or_cmu == 'openmplposer':
                vertices_transformed = vertices
            body_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
            mv.set_static_meshes([body_mesh])
            # try:
            body_image = mv.render(render_wireframe=False)
            # save the image for depth map estimation
            if save_images:
                imageio.imwrite(os.path.join(image_save_dir, 'images', 'amass_{}_{}.png'.format(ix, view)), body_image)
                image_name_all.append('amass_{}_{}.png'.format(ix, view))
            
            # except:
            #     print('Error in rendering image for index {} and view {}'.format(ix, view))
            #     continue
            # h36m_points_mmpose, h36m_scores_mmpose, coco_points_mmpose, coco_scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=)
            if 'yolo' in pose2d_model:
                if regressor == 'h36m':
                    raise NotImplementedError('YOLO does not support H36M regressor yet.')
                elif regressor == 'coco':
                    points_mmpose, scores_mmpose = run_yolo(body_image, yolo_inferencer)
                    points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, R, t, K, lower_body_reversed=False)
            else:
                if regressor == 'h36m':
                    points_mmpose, scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=True)
                    points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, R, t, K, lower_body_reversed=True)
                elif regressor == 'coco':
                    points_mmpose, scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=False)
                    points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, R, t, K, lower_body_reversed=False)
                elif regressor == 'both':
                    points_mmpose_h36m, scores_mmpose_h36m, points_mmpose, scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=True, return_coco=True)
                    points_image_amass_h36m, joints_3d_h36m = amass_vertices_to_joints(vertices[None], jregressor[0], R, t, K, lower_body_reversed=True)
                    points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor[1], R, t, K, lower_body_reversed=False)
                    
                    points_image_amass_h36m = points_image_amass_h36m[0]
                    joints_3d_h36m = joints_3d_h36m[0]
                
            points_image_amass = points_image_amass[0]
            joints_3d = joints_3d[0]
            # if joints_3d[11:13, 0].mean() > room_max_x or joints_3d[11:13, 0].mean() < room_min_x or joints_3d[11:13, 1].mean() > room_max_y or joints_3d[11:13, 1].mean() < room_min_y or joints_3d[11:13, 2].mean() > room_max_z or joints_3d[11:13, 2].mean() < room_min_z:
            #     print('Person is outside room:', ix)
            #     continue
            if triangulate:
                if filter_outside_image_when_triangulating:
                    scores_mmpose_tri = filter_wrong_mmpose_detections(points_mmpose, scores_mmpose, points_image_amass, imw, imh)    # to filter joints outside image
                else:
                    scores_mmpose_tri = scores_mmpose
            if fit_mmpose_to_amass:
                keypoints_most_aligned = find_most_aligned_joints(points_image_amass, keypoints_most_aligned_with_h36m, imw, imh)
                aligned_mmpose_joints, mmpose_confs = fit_skeleton_to_amass(points_mmpose, points_image_amass, keypoints_most_aligned, scores_mmpose, imw, imh, apply_joint_clip, fit_using_most_aligned=fit_using_most_aligned)
                joints_2d_mmpose_all[-1].append(aligned_mmpose_joints)
                confs_2d_mmpose_all[-1].append(mmpose_confs)
                if regressor == 'both':
                    raise ValueError('Not implemented')
                # joints_2d_amass_all[-1].append(points_image_amass)
            else:
                joints_2d_mmpose_all[-1].append(points_mmpose)
                confs_2d_mmpose_all[-1].append(scores_mmpose)
                if regressor == 'both':
                    joints_2d_mmpose_h36m_all[-1].append(points_mmpose_h36m)
                    confs_2d_mmpose_h36m_all[-1].append(scores_mmpose_h36m)
                    joints_2d_amass_h36m_all[-1].append(points_image_amass_h36m)
                    
            if save_depth_maps:   # for this part, you need to run ml-depth-pro script at the same time
                # # run a timer
                # start = time.time()
                # while True:
                #     try:
                #         body_depth = np.load(os.path.join(image_save_dir, 'depths', 'amass_{}_{}.npy'.format(ix, view)))
                #         break
                #     except:
                #         time.sleep(.1)
                #         if time.time() - start > 100:
                #             print('Timeout for depth map: ix: {}, view: {}'.format(ix, view))
                #             body_depth = np.zeros((imh, imw))
                #             break
                # body_depth_all.append(body_depth)
                
                body_image_transformed = transform_depth(body_image.copy())
                prediction_depth = model_depth.infer(body_image_transformed, f_px=None)
                depth_map = prediction_depth["depth"]  # Depth in meters
                depth_map = depth_map.cpu()
                image_width = body_image.shape[1]
                image_height = body_image.shape[0]
                
                depth_map_gt = np.zeros((points_image_amass.shape[0], 5, 5))
                joints_2d_gt = np.round(points_image_amass).astype(int)
                joints_2d_gt[:, 0] = np.clip(joints_2d_gt[:, 0], 2, image_width-3)
                joints_2d_gt[:, 1] = np.clip(joints_2d_gt[:, 1], 2, image_height-3)
                for i in range(5):
                    for j in range(5):
                        depth_map_gt[:, i, j] = depth_map[joints_2d_gt[:, 1]+i-2, joints_2d_gt[:, 0]+j-2]
                body_depth_all_gt[-1].append(depth_map_gt)
                
                depth_map_mmpose = np.zeros((points_mmpose.shape[0], 5, 5))
                joints_2d_mmpose = np.round(points_mmpose).astype(int)
                joints_2d_mmpose[:, 0] = np.clip(joints_2d_mmpose[:, 0], 2, image_width-3)
                joints_2d_mmpose[:, 1] = np.clip(joints_2d_mmpose[:, 1], 2, image_height-3)
                for i in range(5):
                    for j in range(5):
                        depth_map_mmpose[:, i, j] = depth_map[joints_2d_mmpose[:, 1]+i-2, joints_2d_mmpose[:, 0]+j-2]
                body_depth_all_mmpose[-1].append(depth_map_mmpose)
                    
            
            joints_2d_amass_all[-1].append(points_image_amass)
            if regressor == 'both':
                joints_2d_amass_h36m_all[-1].append(points_image_amass_h36m)
                
            if triangulate:
                confs_2d_mmpose_tri_all[-1].append(scores_mmpose_tri)
                
            views_all[-1].append(view)
            
        joints_3d_all.append(joints_3d)
        if regressor == 'both':
            joints_3d_h36m_all.append(joints_3d_h36m)
            
        if triangulate:
            if run_on_random_cameras or sphere_camera:
                camera_setup_to_use_for_tri = 0
            else:
                camera_setup_to_use_for_tri = camera_setup_to_use
            triangulated_3d_mmpose = triangulate_poses(cameras_tri[camera_setup_to_use_for_tri],
                                                       np.array(joints_2d_mmpose_all[-1]), 
                                                    #    np.array(joints_2d_amass_all[-1]), 
                                                       np.array(confs_2d_mmpose_tri_all[-1]).squeeze(), 
                                                    #    np.ones_like(np.array(confs_2d_mmpose_tri_all[-1]).squeeze()), 
                                                       conf_threshold=triangulate_th)
            triangulated_3d_mmpose_all.append(triangulated_3d_mmpose[0])
            # print('Triangulated 3D:', triangulated_3d_mmpose[0])
            
        if save_temp_checkpoints and ix % 50 == 0:
            to_save = {
                'joints_3d': joints_3d_all,
                'joints_2d_mmpose': joints_2d_mmpose_all,
                'confs_2d_mmpose': confs_2d_mmpose_all,
                'joints_2d_amass': joints_2d_amass_all,
                'triangulated_3d_mmpose': triangulated_3d_mmpose_all,
                'camera_setup_used': camera_setup_used,
                'views_used': views_all,
                'joints_3d_h36m': joints_3d_h36m_all,
                'joints_2d_mmpose_h36m': joints_2d_mmpose_h36m_all,
                'confs_2d_mmpose_h36m': confs_2d_mmpose_h36m_all,
                'joints_2d_amass_h36m': joints_2d_amass_h36m_all,
                'camera_parameters_all': camera_parameters_all,
                'body_depth_all_gt': body_depth_all_gt,
                'body_depth_all_mmpose': body_depth_all_mmpose,
                'image_name_all': image_name_all,
                # 'random_seed_state': random.getstate(),
                'numpy_random_seed_state': np.random.get_state(),
                'torch_random_seed_state': torch.get_rng_state(),
            }
            pickle.dump(to_save, open(os.path.join(temp_dir, 'temp_{}.pkl'.format(ix)), 'wb'))
        
    joints_3d_all_np = np.array(joints_3d_all)
    joints_2d_mmpose_all_np = np.array(joints_2d_mmpose_all)
    confs_2d_mmpose_all_np = np.array(confs_2d_mmpose_all)
    joints_2d_amass_all_np = np.array(joints_2d_amass_all)
    joints_3d_h36m_all_np = np.array(joints_3d_h36m_all)
    joints_2d_mmpose_h36m_all_np = np.array(joints_2d_mmpose_h36m_all)
    confs_2d_mmpose_h36m_all_np = np.array(confs_2d_mmpose_h36m_all)
    joints_2d_amass_h36m_all_np = np.array(joints_2d_amass_h36m_all)
    if triangulate:
        triangulated_3d_mmpose_all_np = np.array(triangulated_3d_mmpose_all)
    else:
        triangulated_3d_mmpose_all_np = None
    camera_setup_used = np.array(camera_setup_used)
    views_all = np.array(views_all)
    
    with open(file_path, 'wb') as f:
        pickle.dump({
            'joints_3d': joints_3d_all_np,
            'joints_2d_mmpose': joints_2d_mmpose_all_np,
            'confs_2d_mmpose': confs_2d_mmpose_all_np,
            'joints_2d_amass': joints_2d_amass_all_np,
            'triangulated_3d_mmpose': triangulated_3d_mmpose_all_np,
            'camera_setup_used': camera_setup_used,
            'views_used': views_all,
            'joints_3d_h36m': joints_3d_h36m_all_np,
            'joints_2d_mmpose_h36m': joints_2d_mmpose_h36m_all_np,
            'confs_2d_mmpose_h36m': confs_2d_mmpose_h36m_all_np,
            'joints_2d_amass_h36m': joints_2d_amass_h36m_all_np,
            'camera_parameters_all': camera_parameters_all,
            'body_depth_all_gt': body_depth_all_gt,
            'body_depth_all_mmpose': body_depth_all_mmpose,
            'image_name_all': image_name_all,
        }, f)
    print('Saved dataset to:', file_path)
    
    # remove temp files
    if save_temp_checkpoints:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

def main():
    args = parse_args()
    seed = args.dataset_split_number
    seed_torch(seed)
    if args.train_datasets is not None:
        amass_splits['train'] = args.train_datasets
    if args.valid_datasets is not None:
        amass_splits['validation'] = args.valid_datasets
    expr_code = args.exp

    print('Train split has %d datasets.'%len(amass_splits['train']))
    print('Train datasets:', amass_splits['train'])


    msg = ''' Initial use of standard AMASS dataset preparation pipeline '''

    # amass_dir =  '/data/amass_data_poses/' #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
    # amass_dir =  '/globalscratch/users/a/b/abolfazl/amass_data_poses' #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
    amass_dir = args.amass_data_dir
    
    if args.regressor == 'h36m':
        J_regressor = amass_dir + '/J_regressor_h36m.npy'
    elif args.regressor == 'coco':
        J_regressor = amass_dir + '/J_regressor_coco.npy'
    elif args.regressor == 'both':
        J_regressor = [
            amass_dir + '/J_regressor_h36m.npy',
            amass_dir + '/J_regressor_coco.npy'
        ]
    else:
        raise ValueError('Unknown regressor: {}'.format(args.regressor))
    # work_dir = '/data/Git_Repo/amass/support_data/prepared_data/{}/'.format(expr_code)
    # work_dir = '/globalscratch/users/a/b/abolfazl/amass_data/support_data/prepared_data/{}/'.format(expr_code)

    work_dir = os.path.join(args.work_dir, expr_code)


    logger = log2file(makepath(work_dir, '%s.log' % (expr_code), isfile=True))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)

    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['validation'])))

    logger('Train split has %d datasets.'%len(amass_splits['train']))
    logger('Train datasets: {}'.format(amass_splits['train']))

    # if osp.join(work_dir, 'stage_III') not in glob.glob(osp.join(work_dir, '*')):
    #     prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)

    bm_fname = osp.join(args.support_dir, 'body_models/smplh/male/model.npz')

    num_betas = 16 # number of body parameters

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(comp_device)
    faces = c2c(bm.f)
    # print('faces:', faces.shape)

    if args.use_cams_from == 'h36m':
        if args.calib_file_h36m is None:
            raise ValueError('Please provide the camera calibration file for H36M')
        cameras = load_all_cameras_h36m(args.calib_file_h36m, actors=args.actors_h36m)
        logger('Loaded cameras from H36M')
        n_all_cameras = len(cameras) 
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])  
        views = range(1, n_all_cameras+1)
        calibs = args.actors_h36m
    elif args.use_cams_from == 'cmu':
        if args.calib_root_cmu is None:
            raise ValueError('Please provide the camera calibration root for CMU')
        cameras = load_all_cameras_cmu(args.calib_root_cmu, cmu_calibs=args.calibs_cmu)
        logger('Loaded cameras from CMU')
        n_all_cameras = len(cameras) 
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])
        views = args.views_cmu
        calibs = args.calibs_cmu
    elif args.use_cams_from == 'dome':
        cameras = load_all_cameras_dome(args.calib_file_dome)
        logger('Loaded cameras from DOME')
        views = args.views_dome
        if views is None:
            views = range(1, len(cameras)+1)
        cameras = {k: v for k, v in cameras.items() if k in views}
        n_all_cameras = len(cameras)
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])
        calibs = 'dome'
    elif args.use_cams_from == 'rich':
        cameras = load_all_cameras_rich(args.calib_file_rich, rich_scenes=args.scenes_rich)
        logger('Loaded cameras from RICH')
        n_all_cameras = len(cameras)
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])
        views = range(0, n_all_cameras)
        calibs = args.scenes_rich
        
    elif args.use_cams_from == 'openmplposer':
        if args.calib_path_openmplposer is None:
            raise ValueError('Please provide the camera calibration root for OpenMPLPoser')
        cameras = load_all_cameras_openmplposer(args.calib_path_openmplposer)
        logger('Loaded cameras from OpenMPLPoser')
        n_all_cameras = len(cameras) 
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])
        views = range(1, n_all_cameras+1)
        calibs = ['openmplposer']
    
    print('Views:', views)
    
    if osp.join(work_dir, 'stage_V') not in glob.glob(osp.join(work_dir, '*')):
            os.makedirs(osp.join(work_dir, 'stage_V'))
    
    for subset in args.operation_on:
        if osp.join(work_dir, 'stage_V', subset) not in glob.glob(osp.join(work_dir, 'stage_V', '*')):
            os.makedirs(osp.join(work_dir, 'stage_V', subset))
        
    if args.run_on_random_cameras and args.sphere_camera:
        raise ValueError('Cannot run on random cameras and sphere cameras at the same time')
    
    if args.image_save_dir is not None:
        image_save_dir = os.path.join(args.image_save_dir, 'images_saved', '{}'.format(args.extra_name))
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(os.path.join(image_save_dir, 'images'), exist_ok=True)
        

    h36m_or_cmu = args.use_cams_from if not args.run_on_random_cameras and not args.sphere_camera else 'dome'
    for subset in args.operation_on:
        save_dataset(
                    dataset_split_number=args.dataset_split_number,
                    subset=subset,
                    work_dir=work_dir,
                    bm=bm,
                    faces=faces,
                    J_regressor=J_regressor,
                    regressor=args.regressor,
                    num_betas=num_betas,
                    cameras=cameras,
                    views=views,
                    n_camera_setups=n_camera_setups,
                    imw=args.image_width,
                    imh=args.image_height,
                    room_min_x=args.room_size[0],
                    room_max_x=args.room_size[1],
                    room_min_y=args.room_size[2],
                    room_max_y=args.room_size[3],
                    room_min_z=args.room_size[4],
                    room_max_z=args.room_size[5],
                    h36m_or_cmu=h36m_or_cmu,
                    use_cams_from=args.use_cams_from,
                    calibs=calibs,
                    apply_rotation=args.apply_rotation,
                    apply_joint_clip=args.apply_joint_clip,
                    number_of_frames=args.n_frames,
                    fit_using_most_aligned=args.fit_using_most_aligned,
                    fit_mmpose_to_amass=args.fit_mmpose_to_amass,
                    extra_name=args.extra_name,
                    triangulate=args.triangulate,
                    triangulate_th=args.triangulate_th,
                    pose2d_model=args.pose2d_model,
                    run_on_every_nth_frame=args.run_on_every_nth_frame,
                    save_temp_checkpoints=args.save_temp_checkpoints,
                    run_on_random_cameras=args.run_on_random_cameras,
                    camera_location_limit=args.camera_location_limit,
                    n_cameras_per_person=args.n_cameras_per_person,
                    camera_location_outside_room=args.camera_location_outside_room,
                    camera_dist_from_person=args.camera_dist_from_person,
                    filter_outside_image_when_triangulating=not args.dont_filter_outside_image_when_triangulating,
                    room_center=args.room_center,
                    look_at_person=args.look_at_person,
                    sphere_camera=args.sphere_camera,
                    sphere_radius=args.sphere_radius,
                    sphere_z_start=args.sphere_z_start,
                    sphere_z_end=args.sphere_z_end,
                    sphere_angle_step=args.sphere_angle_step,
                    sphere_end_angle=args.sphere_end_angle,
                    save_images=args.save_images,
                    image_save_dir=image_save_dir,
                    save_depth_maps=args.save_depth_maps,
                    )
    
if __name__ == '__main__':
    main()