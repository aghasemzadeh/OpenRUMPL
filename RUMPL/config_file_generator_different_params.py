import yaml
import os
import shutil
from lib.core.config import config
from lib.core.config import update_config
from tqdm import tqdm 
import copy
import pandas as pd

from itertools import combinations

def generate_combinations(input_list, n):
    return list(combinations(input_list, n))

''' 
                ######### Default params #########
    name:   cmu_0_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0
    
    
    'NETWORK.POSEFORMER_ADD_CONFIDENCE_INPUT': False
    'NETWORK.POSEFORMER_MULT_CONFIDENCE_EMB': False
    'NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB': False
    'NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD': True
    'NETWORK.POSEFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT': False
    'NETWORK.POSEFORMER_INPUT_RAYS_AS_TOKEN': True
    'NETWORK.POSE_3D_EMB_LEARNABLE': True
    'NETWORK.POSEFORMER_ADD_3D_POS_ENCODING_TO_RAYS': True
    
    
    'NETWORK.POSEFORMER_MULTIPLE_SPATIAL_BLOCKS': True
    'NETWORK.POSEFORMER_NO_TRANSFORMER_SPT': False
    'NETWORK.POSEFORMER_NO_TRANSFORMER_FPT': False
    
    'NETWORK.POSEFORMER_CONFIDENCE_IN_FPT': False
    
    'NETWORK.POSEFORMER_OUTPUT_HEAD_DEEP': False
    '''


def make_different_parameters(configs_ro_run):
    different_params = []
    for i in range(len(configs_ro_run)):
        different_params.append({
            'name': [],
            'modifications': {}
        })
        ####### FPT
        try:
            if configs_ro_run.loc[i]['FPT'] == 'No':
                different_params[i]['name'].append(('_FPT_', '_NoFPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_NO_TRANSFORMER_FPT'] = True
            elif configs_ro_run.loc[i]['FPT'] == 'Yes':
                different_params[i]['name'].append(('_FPT_', '_FPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_NO_TRANSFORMER_FPT'] = False
            else:
                raise
        except KeyError:
            pass
        ####### SPT
        try:
            if configs_ro_run.loc[i]['SPT'] == 'No':
                different_params[i]['name'].append(('_MultiSPT_', '_NoSPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_NO_TRANSFORMER_SPT'] = True
                different_params[i]['modifications']['NETWORK.POSEFORMER_MULTIPLE_SPATIAL_BLOCKS'] = False
            elif configs_ro_run.loc[i]['SPT'] == 'Single':
                different_params[i]['name'].append(('_MultiSPT_', '_SingleSPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_MULTIPLE_SPATIAL_BLOCKS'] = False
            elif configs_ro_run.loc[i]['SPT'] == 'Multi':
                different_params[i]['name'].append(('_MultiSPT_', '_MultiSPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_MULTIPLE_SPATIAL_BLOCKS'] = True
            else:
                raise
        except KeyError:
            pass
        ####### Raytoken
        try:
            if configs_ro_run.loc[i]['Raytoken'] == False:
                different_params[i]['name'].append(('_Raytoken_', '_RaytokenNo_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_INPUT_RAYS_AS_TOKEN'] = False
            elif configs_ro_run.loc[i]['Raytoken'] == True:
                different_params[i]['name'].append(('_Raytoken_', '_Raytoken_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_INPUT_RAYS_AS_TOKEN'] = True
            else:
                raise
        except KeyError:
            pass
        ####### Learn3dEnc
        try:
            if configs_ro_run.loc[i]['Learn3dEnc'] == False:
                different_params[i]['name'].append(('_Learn3dEnc_', '_Learn3dEncNo_'))
                different_params[i]['modifications']['NETWORK.POSE_3D_EMB_LEARNABLE'] = False   
            elif configs_ro_run.loc[i]['Learn3dEnc'] == True:
                different_params[i]['name'].append(('_Learn3dEnc_', '_Learn3dEnc_'))
                different_params[i]['modifications']['NETWORK.POSE_3D_EMB_LEARNABLE'] = True
            else:
                raise
        except KeyError:
            pass
        ####### Add3dEncRays
        try:
            if configs_ro_run.loc[i]['Add3dEncRays'] == False:
                different_params[i]['name'].append(('_Add3dEncRays_', '_Add3dEncRaysNo_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_ADD_3D_POS_ENCODING_TO_RAYS'] = False
            elif configs_ro_run.loc[i]['Add3dEncRays'] == True:
                different_params[i]['name'].append(('_Add3dEncRays_', '_Add3dEncRays_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_ADD_3D_POS_ENCODING_TO_RAYS'] = True
            else:
                raise
        except KeyError:
            pass
        ####### Conf
        try:
            if configs_ro_run.loc[i]['Conf'] == 'No':
                different_params[i]['name'].append(('_Conf3rd_', '_Confno_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
            elif configs_ro_run.loc[i]['Conf'] == 'Add':
                different_params[i]['name'].append(('_Conf3rd_', '_ConfAdd_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_ADD_CONFIDENCE_INPUT'] = True
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
            elif configs_ro_run.loc[i]['Conf'] == 'Mult':
                different_params[i]['name'].append(('_Conf3rd_', '_ConfMult_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_MULT_CONFIDENCE_EMB'] = True
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
            elif configs_ro_run.loc[i]['Conf'] == 'Concat':
                different_params[i]['name'].append(('_Conf3rd_', '_ConfConcat_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB'] = True
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
            elif configs_ro_run.loc[i]['Conf'] == 'Weight':
                different_params[i]['name'].append(('_Conf3rd_', '_ConfWeight_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT'] = True
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
            elif configs_ro_run.loc[i]['Conf'] == '3rd':
                different_params[i]['name'].append(('_Conf3rd_', '_Conf3rd_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_INPUT_AS_THIRD'] = True
            else:
                raise
        except KeyError:
            pass
        ####### ConfFPT
        try:
            if configs_ro_run.loc[i]['ConfFPT'] == False:
                different_params[i]['name'].append(('_ConfFPTNo_', '_ConfFPTNo_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_IN_FPT'] = False
            elif configs_ro_run.loc[i]['ConfFPT'] == True:
                different_params[i]['name'].append(('_ConfFPTNo_', '_ConfFPT_'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_CONFIDENCE_IN_FPT'] = True
            else:
                raise
        except KeyError:
            pass
        ###### RegHead
        try:
            if configs_ro_run.loc[i]['RegHead'] == 'Shallow':
                different_params[i]['name'].append(('_RHShallow', '_RHShallow'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_OUTPUT_HEAD_DEEP'] = False
            elif configs_ro_run.loc[i]['RegHead'] == 'Deep':
                different_params[i]['name'].append(('_RHShallow', '_RHDeep'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_OUTPUT_HEAD_DEEP'] = True
            elif configs_ro_run.loc[i]['RegHead'] == 'Kadkhod':
                different_params[i]['name'].append(('_RHShallow', '_RHKadkhod'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_OUTPUT_HEAD_DEEP'] = False
                different_params[i]['modifications']['NETWORK.POSEFORMER_OUTPUT_HEAD_KADKHOD'] = True
                different_params[i]['modifications']['LOSS.TYPE'] = 'MPJPE_KADKHODA'
            else:
                print('RegHead: ' + configs_ro_run.loc[i]['RegHead'])
                raise 
        except KeyError:
            pass
        ###### DirIntersect
        try:
            if configs_ro_run.loc[i]['DirIntersect'] == True:
                different_params[i]['name'].append(('_DirIntersect', '_DirIntersect'))
                different_params[i]['modifications']['DATASET.DIRECTION_AND_INTERSECTION_AS_RAYS'] = True
            else:
                different_params[i]['name'].append(('_DirIntersect', '_DirIntersectNo'))
                different_params[i]['modifications']['DATASET.DIRECTION_AND_INTERSECTION_AS_RAYS'] = False
        except KeyError:
            pass
        ###### FPTAllTokens
        try:
            if configs_ro_run.loc[i]['FPTAllTokens'] == True:
                different_params[i]['name'].append(('_FPTAllTokens', '_FPTAllTokens'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_FPT_BLOCKS_VIEW_KEYPOINT_TOKENS'] = True
            else:
                different_params[i]['name'].append(('_FPTAllTokens', '_FPTAllTokensNo'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_FPT_BLOCKS_VIEW_KEYPOINT_TOKENS'] = False
        except KeyError:
            pass
        ###### RaysOnly
        try:
            if configs_ro_run.loc[i]['RaysOnly'] == True:
                different_params[i]['name'].append(('_RaysOnly', '_RaysOnly'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_RAY_TOKENS_AS_ONLY_INPUT'] = True
            else:
                different_params[i]['name'].append(('_RaysOnly', '_RaysOnlyNo'))
                different_params[i]['modifications']['NETWORK.POSEFORMER_RAY_TOKENS_AS_ONLY_INPUT'] = False
        except KeyError:
            pass
        ###### RaySineEnc
        try:
            if configs_ro_run.loc[i]['RaySineEnc'] == 'Normal' or configs_ro_run.loc[i]['RaySineEnc'] == True:
                different_params[i]['name'].append(('_RaySineEnc', '_RaySineEnc'))
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS'] = True
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF'] = False
                # different_params[i]['modifications']['NETWORK.DIM'] = 64
            elif configs_ro_run.loc[i]['RaySineEnc'] == 'Nerf':
                different_params[i]['name'].append(('_RaySineEnc', '_RaySineEncNerf'))
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS'] = False
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF'] = True
                different_params[i]['modifications']['DATASET.SINE_L_NERF'] = 10
                # different_params[i]['modifications']['NETWORK.DIM'] = 64
            else:
                different_params[i]['name'].append(('_RaySineEnc', '_RaySineEncNo'))
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS'] = False
                different_params[i]['modifications']['DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF'] = False
        except KeyError:
            pass
        ###### TrainAll
        try:
            if configs_ro_run.loc[i]['TrainAllCams'] == 'Yes':
                different_params[i]['name'].append(('_TrainAllCamsNo', '_TrainAllCams'))
                different_params[i]['name'].append(('_TACNo', '_TAC'))
                different_params[i]['modifications']['DATASET.TRAIN_ON_ALL_CAMERAS'] = True
            elif configs_ro_run.loc[i]['TrainAllCams'] == 'No':
                different_params[i]['name'].append(('_TrainAllCamsNo', '_TrainAllCamsNo'))
                different_params[i]['name'].append(('_TACNo', '_TACNo'))
                different_params[i]['modifications']['DATASET.TRAIN_ON_ALL_CAMERAS'] = False
            elif configs_ro_run.loc[i]['TrainAllCams'] == 'ButTest':
                different_params[i]['name'].append(('_TrainAllCamsNo', '_TrainAllCamsButTest'))
                different_params[i]['name'].append(('_TACNo', '_TAButTest'))
                different_params[i]['modifications']['DATASET.TRAIN_ON_ALL_CAMERAS'] = True
                different_params[i]['modifications']['DATASET.NOT_TRAIN_ON_TEST_VIEWS'] = True
            else:
                raise
        except KeyError:
            pass
        ###### Intersection with
        try:
            if configs_ro_run.loc[i]['IntersectWith'] == 'Closest':
                different_params[i]['name'].append(('_IntersectC', '_IntersectC'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'Closest'
            elif configs_ro_run.loc[i]['IntersectWith'] == 'X':
                different_params[i]['name'].append(('_IntersectC', '_IntersectX'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'X'
            elif configs_ro_run.loc[i]['IntersectWith'] == 'Y':
                different_params[i]['name'].append(('_IntersectC', '_IntersectY'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'Y'
            elif configs_ro_run.loc[i]['IntersectWith'] == 'Z':
                different_params[i]['name'].append(('_IntersectC', '_IntersectZ'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'Z'   
            elif configs_ro_run.loc[i]['IntersectWith'] == 'All':
                different_params[i]['name'].append(('_IntersectC', '_IntersectAll'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'All'
            elif configs_ro_run.loc[i]['IntersectWith'] == 'Camera':
                different_params[i]['name'].append(('_IntersectC', '_IntersectM'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'Camera'
            elif configs_ro_run.loc[i]['IntersectWith'] == 'Ground':
                different_params[i]['name'].append(('_IntersectC', '_IntersectG'))
                different_params[i]['modifications']['DATASET.INTERSECTION_RAY_WITH'] = 'Ground'
            else:
                raise
        except KeyError:
            pass
        ###### number of samples
        try:
            ndownsample = configs_ro_run.loc[i]['NDownSample']
            different_params[i]['name'].append(('_NDownSample01', '_NDownSample{:02d}'.format(ndownsample)))
            different_params[i]['modifications']['DATASET.TRAIN_N_DOWN_SAMAPLE'] = ndownsample
        except KeyError:
            pass
        ##### Train on Specific Camera Setups
        try:
            specific_camera_setups = configs_ro_run.loc[i]['TrainSpecCS']
            if isinstance(specific_camera_setups, list):
                different_params[i]['name'].append(('_TrainSpecCSNo', '_TrainSpecCS'))
                different_params[i]['name'].append(('_TSCSNo', '_TSCS'))
                different_params[i]['modifications']['DATASET.TRAIN_ON_SPECIFIC_CAMERA_SETUPS'] = specific_camera_setups
            else:
                different_params[i]['name'].append(('_TrainSpecCSNo', '_TrainSpecCSNo'))
                different_params[i]['name'].append(('_TSCSNo', '_TSCSNo'))
                different_params[i]['modifications']['DATASET.TRAIN_ON_SPECIFIC_CAMERA_SETUPS'] = False
        except KeyError:
            pass
        ##### Train on Specific Camera Setups Random
        try:
            if configs_ro_run.loc[i]['TrainSpecRandom'] == True:
                different_params[i]['name'].append(('_TrainSpecRandomNo', '_TrainSpecRandom'))
                different_params[i]['name'].append(('_TSRNo', '_TSR'))
                different_params[i]['modifications']['DATASET.PICK_RANDOM_CAMERAS_FROM_SPECIFIC_SETUPS'] = True
            else:
                different_params[i]['name'].append(('_TrainSpecRandomNo', '_TrainSpecRandomNo'))
                different_params[i]['name'].append(('_TSRNo', '_TSRNo'))
                different_params[i]['modifications']['DATASET.PICK_RANDOM_CAMERAS_FROM_SPECIFIC_SETUPS'] = False
        except KeyError:
            pass
        ##### Loss Type
        try:
            if configs_ro_run.loc[i]['Loss'] == 'MPJPE':
                different_params[i]['name'].append(('_LossMPJPE', '_LossMPJPE'))
                different_params[i]['modifications']['LOSS.TYPE'] = 'MPJPE'
            elif configs_ro_run.loc[i]['Loss'] == 'PlaneProjection':
                different_params[i]['name'].append(('_LossMPJPE', '_LossPP'))
                different_params[i]['modifications']['LOSS.TYPE'] = 'PlaneProjectionLoss'
            else:
                raise
        except KeyError:
            pass
        ##### Missing Noise
        try:
            if configs_ro_run.loc[i]['MissingNoise'] == 'No':
                different_params[i]['name'].append(('_MissNo', '_MissNo'))
                different_params[i]['modifications']['DATASET.APPLY_NOISE_MISSING'] = False
                different_params[i]['modifications']['DATASET.MISSING_LEVEL'] = float(str(0.0))
            elif isinstance(configs_ro_run.loc[i]['MissingNoise'], float):
                different_params[i]['name'].append(('_MissNo', '_Miss{:2.0f}'.format(configs_ro_run.loc[i]['MissingNoise'] * 100)))
                different_params[i]['modifications']['DATASET.APPLY_NOISE_MISSING'] = True
                different_params[i]['modifications']['DATASET.MISSING_LEVEL'] = float(str(configs_ro_run.loc[i]['MissingNoise']))
            else:
                raise
        except KeyError:
            pass
        ##### ZeroTokens
        try:
            if configs_ro_run.loc[i]['ZeroTokens'] == False:
                different_params[i]['name'].append(('_ZrTknsNo', '_ZrTknsNo'))
                different_params[i]['modifications']['DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS'] = False
            elif configs_ro_run.loc[i]['ZeroTokens'] == True:
                different_params[i]['name'].append(('_ZrTknsNo', '_ZrTkns'))
                different_params[i]['modifications']['DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS'] = True
            else:
                raise
        except KeyError:
            pass
        ##### FuserMode
        try:
            if configs_ro_run.loc[i]['FuserMode'] == 'Rays':
                different_params[i]['name'].append(('_FuserRays', '_FuserRays'))
                different_params[i]['modifications']['NETWORK.APPLY_VIEW_FUSION'] = True
            elif configs_ro_run.loc[i]['FuserMode'] == 'Middle':
                different_params[i]['name'].append(('_FuserRays', '_FuserMid'))
                different_params[i]['modifications']['NETWORK.APPLY_VIEW_FUSION'] = False
                different_params[i]['modifications']['NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS'] = True
            elif configs_ro_run.loc[i]['FuserMode'] == 'Closest':
                different_params[i]['name'].append(('_FuserRays', '_FuserClos'))
                different_params[i]['modifications']['NETWORK.APPLY_VIEW_FUSION'] = False
                different_params[i]['modifications']['NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS'] = False
            elif configs_ro_run.loc[i]['FuserMode'] == 'CamCal':
                different_params[i]['name'].append(('_FuserRays', '_FuserCamCal'))
                different_params[i]['modifications']['NETWORK.FEED_CAMERA_CALIBRATION'] = True
                different_params[i]['modifications']['NETWORK.FEED_ONLY_2D'] = False
                different_params[i]['modifications']['NETWORK.APPLY_VIEW_FUSION'] = True
                different_params[i]['modifications']['NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS'] = False
            elif configs_ro_run.loc[i]['FuserMode'] == '2D':
                different_params[i]['name'].append(('_FuserRays', '_Fuser2D'))
                different_params[i]['modifications']['NETWORK.FEED_CAMERA_CALIBRATION'] = False
                different_params[i]['modifications']['NETWORK.FEED_ONLY_2D'] = True
                different_params[i]['modifications']['NETWORK.APPLY_VIEW_FUSION'] = True
                different_params[i]['modifications']['NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS'] = False
                
            else:
                raise
        except KeyError:
            pass
        ##### RandomNumViews
        try:
            if configs_ro_run.loc[i]['RandomNumViews'] == 0:
                different_params[i]['name'].append(('_RNV2', '_RNV0'))
                different_params[i]['modifications']['DATASET.TRAIN_RANDOM_NUM_VIEWS'] = False
                # different_params[i]['modifications']['DATASET.MAX_NUM_VIEWS'] = False
            else:
                random_num_views = int(configs_ro_run.loc[i]['RandomNumViews'])
                different_params[i]['name'].append(('_RNV2', '_RNV{}'.format(random_num_views)))
                different_params[i]['modifications']['DATASET.TRAIN_RANDOM_NUM_VIEWS'] = True
                different_params[i]['modifications']['DATASET.MAX_NUM_VIEWS'] = random_num_views
        except KeyError:
            pass
        
    return different_params
     

###### AMASS
config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass/rumpl_0_amass_yolo_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

# config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_random/rumpl_100_amass_yolo_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser/rumpl_200_amass_yolo_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_3_openmplposer_aligned/rumpl_300_amass_yolo_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_4_openmplposer_aligned_yolo11/rumpl_400_amass_yolo11_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_5_openmplposer_aligned_rtmos/rumpl_500_amass_rtmos_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_6_openmplposer_aligned_yolo8_p3/rumpl_600_amass_yolo_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

# config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_7_openmplposer_aligned_yolo11_p3/rumpl_700_amass_yolo11_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'

# config_example = '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_8_openmplposer_aligned_rtmos_p3/rumpl_800_amass_rtmos_Conf3rd_2views_Seed0_RaySineEnc_IntersectC_MissNo_ZrTknsNo_FuserRays_RNV2.yaml'
list_configurations_to_run = [
 
    
    ['Concat',  'No', 'Camera',    0.2,    False,  'Rays',     0],
    ['No',      'No', 'Camera',    0.2,    False,  'Rays',     0],
    ['Concat',  'No', 'Camera',    'No',    False,  'Rays',     0],



    # ['Concat',  'No', 'Camera',    0.2,    False,  'Rays',     3],
    # ['No',      'No', 'Camera',    0.2,    False,  'Rays',     3],
    # ['Concat',  'No', 'Camera',    'No',    False,  'Rays',     3],

    
    
]


views = [1, 2, 3]

n_views_list = list(range(1, len(views)+1))
n_views_list = [3]

run_on_predefined_view_combinations = False
predefined_view_combinations = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],   # backward
    [9, 10, 11, 12], # forward
    [13, 14, 15, 16], # right
    [17, 18, 19, 20], # left
    [21, 22, 23, 24], # up
    [25, 26, 27, 28], # down
]

change_test_views = True
number_jobs_per_sbatch = 1
# views = [1, 11, 21, 31]
# n_views_list = [2]

# views = list(range(31))
# views.remove(20)
# views.remove(21)
# n_views_list = [2]

# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# seeds = [0, 1, 2]
seeds = [0]
# seeds = [1, 2]
##############################################################

# columns = ['FPT', 'SPT', 'Raytoken', 'Learn3dEnc', 'Add3dEncRays', 'Conf', 'ConfFPT']
columns = []
if '_FPT' in config_example or '_FPTNo' in config_example:
    columns.append('FPT')
if '_MultiSPT' in config_example or '_SingleSPT' in config_example or '_NoSPT' in config_example:
    columns.append('SPT')
if '_RayToken' in config_example:
    columns.append('RayToken')
if '_Learn3DEnc' in config_example:
    columns.append('Learn3dEnc')
if '_Add3DEncRays' in config_example:
    columns.append('Add3dEncRays')
if '_Conf' in config_example:
    columns.append('Conf')
if '_ConfFPT' in config_example:
    columns.append('ConfFPT')
if '_RH' in config_example:
    columns.append('RegHead')
if '_DirIntersect' in config_example:
    columns.append('DirIntersect')
if '_FPTAllTokens' in config_example:
    columns.append('FPTAllTokens')
if '_RaysOnly' in config_example:
    columns.append('RaysOnly')
if '_RaySineEnc' in config_example:
    columns.append('RaySineEnc')
if '_TrainAllCamsNo' in config_example or '_TACNo' in config_example:
# len(list_configurations_to_run[0]) > len(columns):
    columns.append('TrainAllCams')
    # assert len(list_configurations_to_run[0]) == len(columns), 'add the new column to the columns list'
if '_IntersectC' in config_example:
    columns.append('IntersectWith')
if '_NDownSample01' in config_example:
    columns.append('NDownSample')
if '_TrainSpecCSNo' in config_example or '_TSCSNo' in config_example:
    columns.append('TrainSpecCS')
if '_TrainSpecRandomNo' in config_example or '_TSRNo' in config_example:
    columns.append('TrainSpecRandom')
if '_LossMPJPE' in config_example:
    columns.append('Loss')
if '_MissNo' in config_example:
    columns.append('MissingNoise')
if '_ZrTknsNo' in config_example:
    columns.append('ZeroTokens')
if '_FuserRays' in config_example:
    columns.append('FuserMode')
if '_RNV2' in config_example:
    columns.append('RandomNumViews')
    
configurations_to_run = pd.DataFrame(columns=columns)

for i in range(len(list_configurations_to_run)):
    configurations_to_run.loc[len(configurations_to_run)] = list_configurations_to_run[i]

different_params = make_different_parameters(configurations_to_run)

# print('different_params', different_params)

bash_script = [
    '#!/bin/bash',
    '#SBATCH --job-name=mpl',
    # '#SBATCH -c 4',
    '#SBATCH -c 16',
    '#SBATCH -p gpu',
    '#SBATCH --gres=gpu:1',
    # '#SBATCH --gres=gpu:TeslaA10:1',
    '#SBATCH --time=2-00:00:00',
    '#SBATCH --mem=150G',
    '#SBATCH --qos=preemptible',
    # '#SBATCH -w=mb-icg101,mb-icg102',
    '#SBATCH -x mb-cas001',
    # '#SBATCH -x mb-cas001,mb-mil101,mb-mil102,mb-mil110,mb-rom103,mb-rom102',
    'echo "Job on $HOSTNAME"',
]

dir_path = config_example.split('/')[:-1]
dir_path = '/'.join(dir_path)


conf_id_main = int(config_example.split('/')[-1].split('_')[1])
list_configs = os.listdir(dir_path)
list_configs = [x for x in list_configs if x.endswith('.yaml')]
conf_id_highest = 0
for conf in list_configs:
    try:
        conf_id = int(conf.split('_')[1])
    except:
        pass
    if conf_id > conf_id_highest:
        conf_id_highest = conf_id

# find the working directory
folds = os.listdir('sbatch_runs')
folds = [x for x in folds if 'sbatch_runs' in x]
sbatch_folder_id = 1
if len(folds) > 0:
    folds = [int(x.split('_')[-1]) for x in folds]
    sbatch_folder_id = max(folds) + 1
    
os.makedirs('sbatch_runs/sbatch_runs_{}'.format(sbatch_folder_id), exist_ok=True)

# open the config example
with open(config_example, 'r') as stream:
    # config_dict_org = yaml.safe_load(stream)
    config_dict_org = yaml.full_load(stream)

if 'pose_former' in config_example:
    to_run = 'python run/pose2d/train_pose_former.py --cfg {} --gpus 0 &'.format(config_example)
elif 'pose_3d_fuser' in config_example:
    to_run = 'python run/pose2d/train_pose_3d_fuser.py --cfg {} --gpus 0 &'.format(config_example)
elif 'rumpl_amass' in config_example:
    to_run = 'python run/train_rumpl.py --cfg {} &'.format(config_example)
else:
    raise

counter = 1
sbatch_run_counter = 1
start_id = conf_id_highest
# start_id = 1172

for j in tqdm(range(len(seeds))):
    if run_on_predefined_view_combinations:
        n_views_list = [0]
    for i in n_views_list:
        # if i > 1:
        #     break
        if run_on_predefined_view_combinations:
            combinations_list = predefined_view_combinations
        else:
            combinations_list = generate_combinations(views, i)
        for views_comb in combinations_list:
            n_views = len(views_comb)
            for ch in different_params:
                config_dict = copy.deepcopy(config_dict_org)
                new_config_id = counter + start_id
                
                prefix = config_example.split('/')[-1].split('_')[0]
                new_config_path = config_example.replace('{}_{}'.format(prefix, conf_id_main), '{}_{}'.format(prefix, new_config_id))
                # new_config_path = config_example.replace('hm_{}'.format(conf_id), 'hm_{}'.format(new_config_id))
                views_comb_str = ''
                for v in views_comb:
                    views_comb_str += 'V{}'.format(v)
                new_config_path = new_config_path.replace('2views', '{}views{}'.format(n_views, views_comb_str))
                new_config_path = new_config_path.replace('Seed0', 'Seed{}'.format(seeds[j]))
                config_dict['DATASET']['TRAIN_VIEWS'] = list(views_comb)
                if change_test_views:
                    config_dict['DATASET']['TEST_VIEWS'] = list(views_comb)
                config_dict['SEED'] = seeds[j]
                for name_changes in ch['name']:
                    new_config_path = new_config_path.replace(name_changes[0], name_changes[1])
                for key, value in ch['modifications'].items():
                    # print(key, value)
                    key_list = key.split('.')
                    if len(key_list) == 1:
                        config_dict[key_list[0]] = value
                    elif len(key_list) == 2:
                        config_dict[key_list[0]][key_list[1]] = value
                    else:
                        raise
                    
                # save the new config
                with open(new_config_path, 'w') as file:
                    documents = yaml.dump(config_dict, file)
                
                if 'pose_former' in config_example:
                    to_run = 'python run/pose2d/train_pose_former.py --cfg {} --gpus 0 &'.format(new_config_path)
                elif 'pose_3d_fuser' in config_example:
                    to_run = 'python run/pose2d/train_pose_3d_fuser.py --cfg {} --gpus 0 &'.format(new_config_path)
                elif 'rumpl_amass' in config_example:
                    to_run = 'python run/train_rumpl.py --cfg {} &'.format(new_config_path)
                else:
                    raise
                sbatched = False
                
                if number_jobs_per_sbatch == 1:
                    sbatch_txt = copy.deepcopy(bash_script)
                    if i == 5:
                        sbatch_txt[6] = '#SBATCH --mem=70G'
                    elif i == 4:
                        sbatch_txt[6] = '#SBATCH --mem=70G'
                    elif i == 3:
                        sbatch_txt[6] = '#SBATCH --mem=80G'
                    else:
                        sbatch_txt[6] = '#SBATCH --mem=80G'
                        
                    sbatch_txt.append(to_run)
                    sbatch_txt.append('wait')
                    with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:
                        for line in sbatch_txt:
                            file.write(line + '\n')
                    sbatch_run_counter += 1
                
                if number_jobs_per_sbatch == 2:
                    if counter % 2 == 1:
                        sbatch_txt = copy.deepcopy(bash_script)
                        sbatch_txt[6] = '#SBATCH --mem=80G'
                        
                        # if i == 5:
                        #     sbatch_txt[6] = '#SBATCH --mem=30G'
                        # elif i == 4:
                        #     sbatch_txt[6] = '#SBATCH --mem=25G'
                        # elif i == 3:
                        #     sbatch_txt[6] = '#SBATCH --mem=20G'
                        # elif i == 2:
                        #     sbatch_txt[6] = '#SBATCH --mem=15G'
                        # else:
                        #     sbatch_txt[6] = '#SBATCH --mem=10G'
                        sbatch_txt.append(to_run)
                    else:
                        sbatch_txt.append(to_run)
                        sbatch_txt.append('wait')
                        with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:
                            for line in sbatch_txt:
                                file.write(line + '\n')
                        sbatch_txt = []
                        sbatched = True
                        sbatch_run_counter += 1
                        
                    if not sbatched and j == len(seeds)-1 and i == n_views_list[-1] and views_comb == combinations_list[-1] and ch == different_params[-1]:
                        sbatch_txt.append('wait')
                        with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:
                            for line in sbatch_txt:
                                file.write(line + '\n')
                                
                        sbatch_run_counter += 1
                        
                
                
                if number_jobs_per_sbatch == 3:
                    if counter % 3 == 1:
                        sbatch_txt = copy.deepcopy(bash_script)
                        # sbatch_txt[6] = '#SBATCH --mem=150G'
                        if i == 5:
                            # sbatch_txt[6] = '#SBATCH --mem=45G'
                            # sbatch_txt[6] = '#SBATCH --mem=120G'
                            sbatch_txt[6] = '#SBATCH --mem=160G'
                        elif i == 4:
                            # sbatch_txt[6] = '#SBATCH --mem=40G'
                            sbatch_txt[6] = '#SBATCH --mem=100G'
                        elif i == 3:
                            # sbatch_txt[6] = '#SBATCH --mem=35G'
                            sbatch_txt[6] = '#SBATCH --mem=100G'
                        elif i == 2:
                            sbatch_txt[6] = '#SBATCH --mem=60G'
                            # sbatch_txt[6] = '#SBATCH --mem=90G'
                            # sbatch_txt[6] = '#SBATCH --mem=160G'
                        else:
                            # sbatch_txt[6] = '#SBATCH --mem=25G'
                            sbatch_txt[6] = '#SBATCH --mem=80G'
                        sbatch_txt.append(to_run)
                    elif counter % 3 == 2:
                        sbatch_txt.append(to_run)
                    else:
                        sbatch_txt.append(to_run)
                        sbatch_txt.append('wait')
                        # with open('sbatch_runs/sbatch_{}.sh'.format(counter//3), 'w') as file:
                        with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:   
                            for line in sbatch_txt:
                                file.write(line + '\n')
                        sbatch_txt = []
                        sbatched = True
                        sbatch_run_counter += 1
                        
                    
                    if not sbatched and j == len(seeds)-1 and i == n_views_list[-1] and views_comb == combinations_list[-1] and ch == different_params[-1]:
                        sbatch_txt.append('wait')
                        with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:
                            for line in sbatch_txt:
                                file.write(line + '\n')
                                
                        sbatch_run_counter += 1
                    
                # print(to_run)
                counter += 1
            
# run sbatch files
# for i in range(1, counter//3):
#     os.system('sbatch sbatch_runs/sbatch_{}.sh'.format(i))

for i in range(1, sbatch_run_counter):
    # if i == 500:
    #     break
    print('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))
    os.system('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))

# for i in range(1, counter//2):
#     # if i == 500:
#     #     break
#     print('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))
#     os.system('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))

# for i in range(1, counter):
#     os.system('sbatch sbatch_runs/sbatch_{}.sh'.format(i))

# import os
# for i in range(1, 10):
#     print('sbatch sbatch_runs/sbatch_runs_220/sbatch_{}.sh'.format(i))
#     os.system('sbatch sbatch_runs/sbatch_runs_220/sbatch_{}.sh'.format(i))
    