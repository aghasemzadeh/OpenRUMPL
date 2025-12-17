#!/bin/bash

#SBATCH --job-name=mmpose
#SBATCH -c 60
#SBATCH -p gpu
##SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

################ how to process CMU Panoptic dataset ################
# 1. run preprocess_openmplposer.py to generate the .pkl files for train and validation
# 2. run preprocess_openmplposer_filtered.py to filter the unncessary frames
# 3. run preprocess_openmplposer_mmpose.py to generate the .pkl files with mmpose 2D keypoints

# run eval_cmu_panoptic_mmpose.py to evaluate the mmpose 2D keypoints
####################################################################



### 5 cams in coco
openmplposer_dir=/globalscratch/users/a/b/abolfazl/OpenMPLPoser_files/data/keypoints_new     # path to the OpenMPLPoser directory containing the keypoints
dataset=annot_antoine   # name of the dataset given
camera_path=/home/ucl/elen/abolfazl/OpenMPLPoser/data/virtual_cameras/protocol_1
mmpose_dataset_name=mmpose_coco                           # suffix given to the dataset containing mmpose 2D keypoints
# mmpose_output_path=     # path to the directory where the mmpose 2D keypoints are saved
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot.txt                      # path to the file containing the filtered frames
# model=td-hm_hrnet-w32_8xb64-210e_coco-384x288                   # name of the mmpose model used for 2D keypoint estimation
# protocols=rtmo-s_8xb32-600e_coco-640x640_protocol_3
protocols=yolov8n-pose_protocol_1

# python preprocess_openmplposer.py $openmplposer_dir $dataset --running-modes preprocess filter --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name  --keypoints-standard coco 
# txt_file=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_train.txt
# sh run_mmpose_on_cmu_fromTXT.sh $txt_file $model $mmpose_output_path $openmplposer_dir
# python preprocess_openmplposer.py $openmplposer_dir $dataset --running-modes mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name  --keypoints-standard coco 

python preprocess_openmplposer.py $openmplposer_dir ${dataset}_${protocols} --running-modes preprocess --run-for-sets test --mmpose-dataset-name $mmpose_dataset_name  --keypoints-standard coco --camera-path $camera_path --protocols $protocols
python preprocess_openmplposer.py $openmplposer_dir ${dataset}_${protocols} --running-modes mmpose --run-for-sets test --mmpose-dataset-name $mmpose_dataset_name  --keypoints-standard coco --camera-path $camera_path --protocols $protocols


####################################################################

echo "Done"