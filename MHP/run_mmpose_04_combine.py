import os
from os import path as osp
import numpy as np
import torch
import pickle
import argparse
import glob 
from human_body_prior.tools.omni_tools import log2file, makepath
from utils import *


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
    parser.add_argument('--exp', default='amass', type=str, help='Experiment name')
    parser.add_argument('--extra-name', default='', type=str, help='Extra name for the experiment')
    parser.add_argument('--n-splits', default=100, type=int, help='Number of splits')
    parser.add_argument('--operation-on', default=['train', 'validation'], nargs='+', type=str, help='Operations on')
    parser.add_argument('--work-dir', default='amass_data/prepared_data/', type=str, help='Work directory to save the data')

    return parser.parse_args()


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


def main():
    seed_torch(0)
    args = parse_args()
    # if args.train_datasets is not None:
    #     amass_splits['train'] = args.train_datasets
    expr_code = args.exp

    # print('Train split has %d datasets.'%len(amass_splits['train']))
    # print('Train datasets:', amass_splits['train'])


    msg = ''' Initial use of standard AMASS dataset preparation pipeline '''


    work_dir = os.path.join(args.work_dir, expr_code)


    logger = log2file(makepath(work_dir, '%s.log' % (expr_code), isfile=True))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)
    
    for subset in args.operation_on:
        # check for files in stage V
        list_files = glob.glob(os.path.join(work_dir, 'stage_V', subset, '*.pkl'))
        list_files.sort()
        list_files = [file for file in list_files if args.extra_name in file]
        list_splits = [int(file.split('/')[-1].split('_')[1]) for file in list_files]
        assert len(list_files) == args.n_splits - 1, 'Number of files in stage V is not {}: {}\t files missing:{}'.format(args.n_splits - 1,len(list_files), list(set(list(range(args.n_splits - 1))) - set(list_splits)))
        joints_3d_all = []
        joints_2d_mmpose_all = []
        confs_2d_mmpose_all = []
        joints_2d_amass_all = []
        triangulated_3d_mmpose_all = []
        camera_setup_used = []
        camera_parameters_all = []
        body_depth_all_gt = []
        body_depth_all_mmpose = []
        views_used = []
        for file in list_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                print('Loaded:', file)
                joints_3d_all.append(data['joints_3d'])
                joints_2d_mmpose_all.append(data['joints_2d_mmpose'])
                confs_2d_mmpose_all.append(data['confs_2d_mmpose'])
                joints_2d_amass_all.append(data['joints_2d_amass'])
                camera_setup_used.append(data['camera_setup_used'])
                camera_parameters_all += data['camera_parameters_all']
                if len(data['body_depth_all_gt']) == len(data['joints_2d_amass']):
                    body_depth_all_gt.append(data['body_depth_all_gt'])
                    body_depth_all_mmpose.append(data['body_depth_all_mmpose'])
                else:
                    body_depth_gt = np.array(data['body_depth_all_gt']).reshape(data['joints_2d_amass'].shape[0],data['joints_2d_amass'].shape[1], data['body_depth_all_gt'][0].shape[0], data['body_depth_all_gt'][0].shape[1], data['body_depth_all_gt'][0].shape[2])
                    body_depth_mmpose = np.array(data['body_depth_all_mmpose']).reshape(data['joints_2d_amass'].shape[0],data['joints_2d_amass'].shape[1], data['body_depth_all_mmpose'][0].shape[0], data['body_depth_all_mmpose'][0].shape[1], data['body_depth_all_mmpose'][0].shape[2])
                    body_depth_all_gt.append(body_depth_gt)
                    body_depth_all_mmpose.append(body_depth_mmpose)
                
                try:
                    triangulated_3d_mmpose_all.append(data['triangulated_3d_mmpose'])
                except:
                    pass
                try:
                    views_used.append(data['views_used'])
                except:
                    pass
                
        joints_3d_all_np = np.concatenate(joints_3d_all, axis=0)
        joints_2d_amass_all_np = np.concatenate(joints_2d_amass_all, axis=0)
        joints_2d_mmpose_all_np = np.concatenate(joints_2d_mmpose_all, axis=0)
        confs_2d_mmpose_all_np = np.concatenate(confs_2d_mmpose_all, axis=0)
        camera_setup_used = np.concatenate(camera_setup_used, axis=0)
        body_depth_all_gt_np = np.concatenate(body_depth_all_gt, axis=0)
        body_depth_all_mmpose_np = np.concatenate(body_depth_all_mmpose, axis=0)
        
        if len(triangulated_3d_mmpose_all) > 0 and triangulated_3d_mmpose_all[0] is not None:
            triangulated_3d_mmpose_all_np = np.concatenate(triangulated_3d_mmpose_all, axis=0)
        else:
            triangulated_3d_mmpose_all_np = None
        if len(views_used) > 0:
            views_used = np.concatenate(views_used, axis=0)
        else:
            len_views = joints_2d_mmpose_all_np.shape[1]
            if len_views == 29:
                print('Warning: Using 29 views by default')
                views_used = list(range(31))
                views_used.remove(20)
                views_used.remove(21)
                views_used = np.array(views_used)
                views_used = np.stack(joints_2d_mmpose_all_np.shape[0] * [views_used])
            elif len_views == 30:
                print('Warning: Using 30 views by default')
                views_used = list(range(31))
                views_used.remove(20)
                views_used = np.array(views_used)
                views_used = np.stack(joints_2d_mmpose_all_np.shape[0] * [views_used])
            else:
                raise ValueError('Unknown number of views:', len_views)
        
        dataset_name = list_files[0].split('/')[-1].replace('split_0', '').replace('.pkl', '')
        # print('joints_3d_all_np:', joints_3d_all_np.shape)
        
        file_path = osp.join(work_dir, 'datasets', dataset_name)
        os.makedirs(file_path, exist_ok=True)
        file_path = '{}/amass_mmpose_joints_{}.pkl'.format(file_path, subset)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'joints_3d': joints_3d_all_np,
                'joints_2d_mmpose': joints_2d_mmpose_all_np,
                'confs_2d_mmpose': confs_2d_mmpose_all_np,
                'joints_2d_amass': joints_2d_amass_all_np,
                'triangulated_3d_mmpose': triangulated_3d_mmpose_all_np,
                'camera_setup_used': camera_setup_used,
                'views_used': views_used,
                'camera_parameters_all': camera_parameters_all,
                'body_depth_all_gt': body_depth_all_gt_np,
                'body_depth_all_mmpose': body_depth_all_mmpose_np,
            }, f)
        print('Saved dataset to:', file_path)
    
    
            

    
if __name__ == '__main__':
    main()