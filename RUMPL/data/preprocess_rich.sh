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
# 1. run preprocess_cmu_panoptic.py to generate the .pkl files for train and validation
# 2. run preprocess_cmu_panoptic_filtered.py to filter the unncessary frames
# 3. run preprocess_cmu_panoptic_mmpose.py to generate the .pkl files with mmpose 2D keypoints

# run eval_cmu_panoptic_mmpose.py to evaluate the mmpose 2D keypoints
####################################################################


rich_dir=
dataset=annot_smplh
mmpose_dataset_name=mmpose_hrnet_coco
mmpose_output_path=
kept_keys_path=./filtered_keys_rich.txt
python preprocess_rich.py $rich_dir $dataset --running-modes preprocess filter mmpose --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco --dont-discard-less-than-5-visible-joints &
wait


echo "Done"