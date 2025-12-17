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
import yaml

import numpy as np
from easydict import EasyDict as edict

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    
config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.MODEL = 'multiview_transpose'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100
config.WANDB = True
config.WANDB_LOG_IMG = False
config.LOG_IMAGE_FREQ = 1000
config.LOG_WANDB_DIR = 'wandb'
config.MPJPE_PER_KEYPOINT = True
config.TRI_2D_GT = False
config.TARGET_COORDS = False
config.DOWNSAMPLE = 16
config.SEED = 0
config.VALIDATE_ON_TWO_DATASETS = False
config.NOT_CONSIDER_SOME_KP_IN_EVAL = None
config.SLURM_JOB_ID = None

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.NAME = 'pose_hrnet'
config.NETWORK.INIT_WEIGHTS = True
config.NETWORK.PRETRAINED = ''
config.NETWORK.NUM_JOINTS = 17
config.NETWORK.TAG_PER_JOINT = True
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.NETWORK.IMAGE_SIZE_TRAIN = None
config.NETWORK.IMAGE_SIZE_TEST = None
config.NETWORK.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
config.NETWORK.PATCH_SIZE = [64, 64]
config.NETWORK.SIGMA = 2
config.NETWORK.HIDDEN_HEATMAP_DIM = -1
config.NETWORK.TRANSFORMER_DEPTH = 2
config.NETWORK.TRANSFORMER_HEADS = 2
config.NETWORK.TRANSFORMER_MLP_RATIO = 2
config.NETWORK.POS_EMBEDDING_TYPE = 'learnable' 
config.NETWORK.DIM = 2
config.NETWORK.MULTI_TRANSFORMER_DEPTH = [12, 12]
config.NETWORK.MULTI_TRANSFORMER_HEADS = [16, 16]
config.NETWORK.MULTI_DIM = [48, 48]
config.NETWORK.INIT = False
config.NETWORK.NUM_BRANCHES = 1
config.NETWORK.BASEconfigHANNEL = 32
config.NETWORK.EXTRA = edict()
config.NETWORK.NO_VISUAL_TOKEN_FUSION = False
config.NETWORK.POSE_3D_EMB_LEARNABLE = False

config.NETWORK.POSEFORMER_DROP_RATE = 0
config.NETWORK.POSEFORMER_ATTN_DROP_RATE = 0
config.NETWORK.POSEFORMER_DROP_PATH_RATE = 0.1
config.NETWORK.POSEFORMER_ADD_CONFIDENCE_INPUT = False
config.NETWORK.POSEFORMER_MULT_CONFIDENCE_EMB = False
config.NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB = False
config.NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD = False
# config.NETWORK.CONFIDENCE_IN_RAYS = False
config.NETWORK.POSEFORMER_LINEAR_WEIGHTED_MEAN = False
config.NETWORK.POSEFORMER_ADD_3D_POS_ENCODING_IN_SPATIAL = False
config.NETWORK.POSEFORMER_INPUT_RAYS_AS_TOKEN = False
config.NETWORK.POSEFORMER_ADD_3D_POS_ENCODING_TO_RAYS = False
config.NETWORK.POSEFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT = False
config.NETWORK.POSEFORMER_MULTIPLE_SPATIAL_BLOCKS = False
config.NETWORK.POSEFORMER_NO_TRANSFORMER_SPT = False
config.NETWORK.POSEFORMER_NO_TRANSFORMER_FPT = False
config.NETWORK.POSEFORMER_CONFIDENCE_IN_FPT = False
config.NETWORK.POSEFORMER_OUTPUT_HEAD_DEEP = False
config.NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD = False
config.NETWORK.POSEFORMER_OUTPUT_HEAD_HIDDEN_DIM = 1024

config.NETWORK.POSEFORMER_FPT_BLOCKS_VIEW_KEYPOINT_TOKENS = False
config.NETWORK.POSEFORMER_RAY_TOKENS_AS_ONLY_INPUT = False

config.NETWORK.POSEFORMER_LEARNED_QUERY_FPT = False

config.NETWORK.CONCAT_CAM_CENTER_TO_INPUTS = False
config.NETWORK.CONCAT_CAM_AXIS_TO_RAYS = False

config.NETWORK.EMBED_PERSON_HEIGHT = False


config.NETWORK.INIT_WEIGHTS_FROM = 'scratch'

config.NETWORK.MODEL_SIMPLE_DEEPER = False
config.NETWORK.MODEL_SIMPLE_KADKHODA = False
config.NETWORK.SIMPLE_MLP_HIDDEN_SIZE = 1024

config.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS = False
config.NETWORK.APPLY_VIEW_FUSION = False

config.NETWORK.ADD_VIEW_ENCODING = True

config.NETWORK.POSE_3D_FUSER_CONCAT_DIRECTION_INTERSECTION_FIRST = False

config.NETWORK.FEED_CAMERA_CALIBRATION = False
config.NETWORK.FEED_ONLY_2D = False

config.NETWORK.CONCAT_DEPTH_AS_INPUT = False

config.NETWORK.NOT_USE_INTERSECTION_FEATURES = False
# common params for NETWORK
# config.NETWORK = edict()
# config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
# config.NETWORK.NUM_JOINTS = 20
# config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
# config.NETWORK.IMAGE_SIZE = np.array([320, 320])
# config.NETWORK.SIGMA = 2
# config.NETWORK.TARGET_TYPE = 'gaussian'
# config.NETWORK.AGGRE = True

# # Transformer
# config.NETWORK.DIM_MODEL = 256
# config.NETWORK.DIM_FEEDFORWARD = 1024
# config.NETWORK.ENCODER_LAYERS = 3
# config.NETWORK.N_HEAD = 8
# # 2D positional encoding
# config.NETWORK.POS_EMBEDDING = 'sine'
# config.NETWORK.ATTENTION_ACTIVATION = 'relu'

# config.NETWORK.INIT_WEIGHTS = True

# # TransFusion
# config.NETWORK.FUSION = True            # Fuse 2 views or not
# config.NETWORK.POS_EMB_3D = 'geometry'      # 3D position embedding type: none, learnable, geometry
# config.NETWORK.REG_HEAD = True
# config.NETWORK.GAMMA = 10

# # pose_resnet related params
# config.NETWORK.EXTRA = edict()
# config.NETWORK.EXTRA.NUM_LAYERS = 50
# config.NETWORK.EXTRA.DECONV_WITH_BIAS = False
# config.NETWORK.EXTRA.FINAL_CONV_KERNEL = 1
# config.NETWORK.EXTRA.NUM_DECONV_FILTERS = 1

config.LOSS = edict()
config.LOSS.TYPE = 'MPJPE'
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.WEIGHT_AXIS = None
config.LOSS.WEIGHT_ON_VISIBLIITY = False

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = '../data/h36m/'
config.DATASET.ROOT_2DATSET = '../data/h36m/'
config.DATASET.ROOT_3DATSET = '../data/h36m/'
config.DATASET.ROOT_TRAIN = None
config.DATASET.ROOT_TEST = None
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multi_view_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 0
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True
config.DATASET.WITH_DAMAGE = True
config.DATASET.APPLY_NOISE = False
config.DATASET.APPLY_NOISE_TEST = False
config.DATASET.NOISE_LEVEL = 1.0
config.DATASET.APPLY_NOISE_CAMERAS = False
config.DATASET.R_NOISE_VALUE = 0.0
config.DATASET.T_NOISE_VALUE = 0.0
config.DATASET.APPLY_NOISE_MISSING = False
config.DATASET.APPLY_NOISE_MISSING_TEST = False
config.DATASET.MISSING_LEVEL = 0.0          # between 0 and 1. 0.0 means no missing
config.DATASET.USE_MMPOSE_TRAIN = False
config.DATASET.USE_MMPOSE_VAL = False
config.DATASET.USE_MMPOSE_TEST = False
config.DATASET.USE_3D_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.USE_3D_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.EPIPOLAR_ERROR_ACCEPTANCE_THRESHOLD = [0.06, 0.06, 0.09, 0.09]   # in meters! corresponds to KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE
config.DATASET.KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE = [5, 6, 11, 12]

config.DATASET.MIX_AMASS_WITH_MMPOSE_WHEN_USE_MMPOSE = False
config.DATASET.MIX_GT_WITH_MMPOSE_WHEN_USE_MMPOSE = False       # if true, it will mix the gt with mmpose when use mmpose (keypoints_to_mix)
config.DATASET.KEYPOINTS_TO_MIX = [0, 1, 4]

config.DATASET.CAMERA_MANUAL_ORDER = False
config.DATASET.VAL_ON_TRAIN = False
config.DATASET.DATASET_TYPE = 'annot_different_scene_same_cams'   # annot_different_scene_same_cams, annot_same_scene_different_cams, annot_different_scene_different_cams
config.DATASET.TARGET_NORMALIZED_3D = False
config.DATASET.INPUTS_NORMALIZED = False
config.DATASET.NORMALIZE_CAMERAS = False
config.DATASET.NORMALIZE_ROOM = False
config.DATASET.NORMALIZE_ROOM_FIRSTLY = False
config.DATASET.USE_T = False
config.DATASET.NO_AUGMENTATION = False
config.DATASET.NO_AUGMENTATION_3D = False
config.DATASET.OUTPUT_IN_METER = False
config.DATASET.CLIP_JOINTS = False

config.DATASET.TEST_MMPOSE_CONFS_TH = None

config.DATASET.TRAIN_VIEWS = None
config.DATASET.TEST_VIEWS = None
config.DATASET.ROOM_MIN_X = -100
config.DATASET.ROOM_MAX_X = 100
config.DATASET.ROOM_MIN_Y = -100
config.DATASET.ROOM_MAX_Y = 100
config.DATASET.ROOM_MIN_Z = 0
config.DATASET.ROOM_MAX_Z = 200
config.DATASET.ROOM_CENTER = [0, 0, 0]
config.DATASET.CENTERALIZE_ROOT_FIRST = False
config.DATASET.FLIP_3D = False
config.DATASET.ROTATE_3D = False
config.DATASET.USE_GRID = True
config.DATASET.BUG_TEST_3D_EMB = False
config.DATASET.TRAIN_ON_ALL_CAMERAS = False
config.DATASET.TEST_ON_ALL_CAMERAS = False
config.DATASET.TRAIN_ON_SPHERE = False
config.DATASET.TEST_ON_SPHERE = False

config.DATASET.TRAIN_SPHERE_VIEWS = []
config.DATASET.TEST_SPHERE_VIEWS = []

config.DATASET.SPHERE_MAIN_CAMERAS = None

config.DATASET.NOT_TRAIN_ON_TEST_VIEWS = False
config.DATASET.TRAIN_ON_SPECIFIC_CAMERA_SETUPS = None
config.DATASET.PICK_RANDOM_CAMERAS_FROM_SPECIFIC_SETUPS = False

config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 2
config.DATASET.AMASS_VAL_LOCATED = False
config.DATASET.AMASS_DATASET_TYPE = None
config.DATASET.AMASS_SUPPORT_DIR = ''

config.DATASET.TRAIN_N_SAMPLES = None
config.DATASET.TEST_N_SAMPLES = None
config.DATASET.TRAIN_N_DOWN_SAMPLE = None
config.DATASET.TEST_N_DOWN_SAMPLE = None
config.DATASET.TRAIN_N_DOWN_SAMAPLE = None  # typo dummy for backward compatibility
config.DATASET.TRAIN_HOW_MANY_RANDOM_CAMERAS_TO_USE = None
config.DATASET.TEST_HOW_MANY_RANDOM_CAMERAS_TO_USE = None
config.DATASET.SWITCH_X_Z = False
config.DATASET.SWITCH_X_Y = False
config.DATASET.SWITCH_Y_Z = False
config.DATASET.SWITCH_Z_X_Y = False
config.DATASET.SWITCH_Y_Z_X = False
config.DATASET.AMASS_DATA_NO_AXIS_SWAP = False
config.DATASET.CMU_CALIB = None
config.DATASET.TRAIN_CMU_CALIB = ['171204_pose4']
config.DATASET.TEST_CMU_CALIB = ['171204_pose4']
config.DATASET.TRAIN_H36M_CALIB_ACTORS = [9]
config.DATASET.TEST_H36M_CALIB_ACTORS = [9]

config.DATASET.PENALIZE_CONFIDENCE = 'no'  # exp_error, linear, exp_sqrt
config.DATASET.PENALIZE_CONFIDENCE_FACTOR = 1.0
config.DATASET.PENALIZE_CONFIDENCE_A = 0.83206918
config.DATASET.PENALIZE_CONFIDENCE_B = 0.00850464

config.DATASET.TRAIN_CMU_FILTERED_NEW = True
config.DATASET.TEST_CMU_FILTERED_NEW = False

config.DATASET.TRAIN_USE_CMU_OLD_DATASETS = True
config.DATASET.TEST_USE_CMU_OLD_DATASETS = True
config.DATASET.TRAIN_CMU_DATASET_NAME = 'annot_standard_7train_2val_filtered_5_64'
config.DATASET.TEST_CMU_DATASET_NAME = 'annot_standard_7train_2val_filtered_5_64'
config.DATASET.TRAIN_MMPOSE_TYPE = 'mmpose_rtm_coco'
config.DATASET.TEST_MMPOSE_TYPE = 'mmpose_rtm_coco'
config.DATASET.TRAIN_USE_AMASS_OLD_DATASETS = True
config.DATASET.TEST_USE_AMASS_OLD_DATASETS = True

config.DATASET.TRAIN_DEPTH_TYPE = 'depth_apple'
config.DATASET.TEST_DEPTH_TYPE = 'depth_apple'
config.DATASET.USE_DEPTH_WITH_NOISE_ON_DISTANCE = False
config.DATASET.TRAIN_USE_DEPTH_WITH_NOISE_ON_DISTANCE = None
config.DATASET.TEST_USE_DEPTH_WITH_NOISE_ON_DISTANCE = None
config.DATASET.TRAIN_USE_GT_DEPTH = False
config.DATASET.TEST_USE_GT_DEPTH = False
config.DATASET.MU_DEPTH_NOISE = 0
config.DATASET.SIGMA_DEPTH_NOISE = 0.5

config.DATASET.JOINTS_VIS_FROM_MMPOSE = True

config.DATASET.TRAIN_USE_H36M_OLD_DATASETS = True
config.DATASET.TEST_USE_H36M_OLD_DATASETS = True
config.DATASET.TRAIN_H36M_DATASET_NAME = 'annot_filtered_5_64'
config.DATASET.TEST_H36M_DATASET_NAME = 'annot_filtered_5_64'
config.DATASET.FILTER_GROUPINGS = True
config.DATASET.TRAIN_FILTER_GROUPINGS = True
config.DATASET.TEST_FILTER_GROUPINGS = True

config.DATASET.TRAIN_FILTER_CMU_WRONG_CASES = False
config.DATASET.TEST_FILTER_CMU_WRONG_CASES = False

config.DATASET.TRAIN_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY = False
config.DATASET.TEST_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY = False

config.DATASET.ONLY_KEEP_INSIDE_ROOM = False
config.DATASET.ONLY_KEEP_IF_IN_CALIBS_ACTORS = False

config.DATASET.TRAIN_ON_ALL_AMASS = False

config.DATASET.TRAIN_PLACE_PERSON_IN_CENTER = False
config.DATASET.TEST_PLACE_PERSON_IN_CENTER = False

config.DATASET.BRING_AMASS_ROOT_TO_ROOM_CENTER = False
config.DATASET.INTRINSIC_TO_METERS_IF_OUTPUT_IN_METER = False

config.DATASET.USE_HELPER_CAMERAS = False
config.DATASET.TRAIN_VIEWS_HELPER = None
config.DATASET.ROOT_VIEWS_HELPER = None     # path to cmu panoptic dataset

config.DATASET.USE_H36M_CAMERAS_ON_CMU = False
config.DATASET.USE_CMU_CAMERAS_ON_CMU = False
config.DATASET.USE_CMU_CAMERAS_ON_H36M = False

config.DATASET.FLIP_LOWER_BODY_KP_TEST = False

config.DATASET.REDUCE_CONFIDENCE_CLOSE_JOINTS = False

config.DATASET.APPLY_SINE_ENCODING_ON_RAYS = False
config.DATASET.SINE_D_MODEL = 60

config.DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF = False
config.DATASET.SINE_L_NERF = 10

config.DATASET.DIRECTION_AND_INTERSECTION_AS_RAYS = False
config.DATASET.INTERSECTION_RAY_WITH = 'Closest' # closest plane among x, y, z. choises: closest, X, Y, Z, All, cam


config.DATASET.RAY_AS_INTERSECTION_WITH_DONUT = False

config.DATASET.DIRECTION_AS_ANGLES = False

config.DATASET.CMU_KEYPOINT_STANDARD = 'h36m' # h36m, coco

config.DATASET.AXIS_YZ_SWAP_FOR_3D = False
config.DATASET.NEGATE_Y_OR_Z_FOR_3D = 'y'   # y (cmu and rich)
config.DATASET.TRAIN_DOME_CALIB_FILE = None
config.DATASET.TEST_DOME_CALIB_FILE = None

config.DATASET.TRAIN_AMASS_WITH_RANDOM_CAMERAS = False
config.DATASET.TEST_AMASS_WITH_RANDOM_CAMERAS = False

config.DATASET.MIN_ANGLE_DIFF = 0   # for random camera selection of amass
config.DATASET.MIN_OKS = 0  # for random camera selection of amass


config.DATASET.ALL_VIEWS_CMU = None

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 0

# master camera
config.DATASET.N_MASTER_CAMERAS = 5

config.DATASET.KP_VISIBILITY_TH = 0.1

config.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS = False

# rich
config.DATASET.TRAIN_RICH_DATASET_NAME = 'annot_smplh_filtered_5_64'
config.DATASET.TEST_RICH_DATASET_NAME = 'annot_smplh_filtered_5_64'

config.DATASET.SHIFT_ROOM = False
config.DATASET.SHIFT_ROOM_TRAIN_VALUE = [0, 0, 0]
config.DATASET.SHIFT_ROOM_TEST_VALUE = [0, 0, 0]

# openmplposer
config.DATASET.TRAIN_OPENMPLPOSER_DATASET_NAME = 'annot_yolov8n-pose_protocol_1'
config.DATASET.TEST_OPENMPLPOSER_DATASET_NAME = 'annot_yolov8n-pose_protocol_1'

# work with random number of views
config.DATASET.TRAIN_RANDOM_NUM_VIEWS = False
config.DATASET.MAX_NUM_VIEWS = 5
config.DATASET.MIN_NUM_VIEWS = 2

# random camera creation
config.DATASET.RANDOM_CAMERA_LOCATION_LIMIT = [0, 0, 0, 0, 0, 0]
config.DATASET.RANDOM_CAMERA_DIST_FROM_PERSON = 2

config.DATASET.VAL_KEYPOINT_STANDARD = None

config.DATASET.GROUND_Z = 0


config.DATASET.OPENMPLPOSER_CALIBS_TRAIN = ''
config.DATASET.OPENMPLPOSER_CALIBS_VAL = ''
# dataset inference
config.DATASET.INFERENCE_DATA_DIR = ''
config.DATASET.INFERENCE_CAMERAS_PATH = ''

config.DATASET.KP_VISIBILITY_TH = 0.1
config.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS = False

config.DATASET.TEST_MULTI_PERSON = False
# noise camera new
config.DATASET.TRAIN_ADD_NOISE_TO_CAMERA_CALIB = False
config.DATASET.TEST_ADD_NOISE_TO_CAMERA_CALIB = False
config.DATASET.NOISE_ROT_DEG = 10.0
config.DATASET.NOISE_TRANS_STD = 0.1

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

config.TRAIN.DO_FUSION = True
config.TRAIN.SMART_PSEUDO_TRAINING = False
config.TRAIN.EPIPOLAR_ERROR_THRESHOLD = 5

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = ''
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.HEATMAP_LOCATION_FILE = 'predicted_heatmaps.h5'
config.TEST.PRED_GT_LOCATION_FILE = 'preds_gt.pkl'
config.TEST.DO_FUSION = True
config.TEST.EVAL_COMMENTS = ''

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True

# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.PAIRWISE_FILE = ''
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = 2000
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file, 'r') as f:
        exp_config = edict(yaml.load(f, Loader=Loader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=999)
    # deconv_suffix = ''.join(
    #     'd{}'.format(num_filters)
    #     for num_filters in cfg.NETWORK.EXTRA.NUM_DECONV_FILTERS)
    deconv_suffix = 'd1'
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])

