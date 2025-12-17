#!/bin/bash

#SBATCH --job-name=amass_mmpose
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:TeslaA100_80:1
#SBATCH --time=10:00:00
#SBATCH --mem=10G
##SBATCH -w mb-icg102
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

support_dir=$1
work_dir=$2
amass_data_dir=$3
calib_path=$4
split_number=$5


# MMPOSE HRNET
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose --extra-name hrnet --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir 
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose --extra-name hrnet --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on validation --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir 


# YOLO
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose --extra-name yolo --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose --extra-name yolo --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on validation --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose 

# YOLO training set aligned with openmplposer

# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name yolo --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose \
#     --train-datasets Eyes_Japan_Dataset \
#              ACCAD \
#              DFaust_67 \
#              BMLhandball \
#              SFU \
#              Transitions_mocap \
#              TCD_handMocap \
#              TotalCapture \
#              KIT \
#              HumanEva \
#              MPI_mosh \
#              BMLmovi \
#              SOMA \
#              MPI_Limits \
#              WEIZMANN \
#              EKUT \
#              SSM_synced \
#              GRAB \
#              DanceDB \
#              HUMAN4D \
#              CNRS

### YOLO 11

# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name yolo11 --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolo11n-pose \
#     --train-datasets Eyes_Japan_Dataset \
#              ACCAD \
#              DFaust_67 \
#              BMLhandball \
#              SFU \
#              Transitions_mocap \
#              TCD_handMocap \
#              TotalCapture \
#              KIT \
#              HumanEva \
#              MPI_mosh \
#              BMLmovi \
#              SOMA \
#              MPI_Limits \
#              WEIZMANN \
#              EKUT \
#              SSM_synced \
#              GRAB \
#              DanceDB \
#              HUMAN4D \
#              CNRS

# MMPOSE HRNET training set aligned with openmplposer

# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name hrnet --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 \
#     --train-datasets Eyes_Japan_Dataset \
#              ACCAD \
#              DFaust_67 \
#              BMLhandball \
#              SFU \
#              Transitions_mocap \
#              TCD_handMocap \
#              TotalCapture \
#              KIT \
#              HumanEva \
#              MPI_mosh \
#              BMLmovi \
#              SOMA \
#              MPI_Limits \
#              WEIZMANN \
#              EKUT \
#              SSM_synced \
#              GRAB \
#              DanceDB \
#              HUMAN4D \
#              CNRS

## RTM POSE
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name rtmos --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model rtmo-s_8xb32-600e_coco-640x640 \
#     --train-datasets Eyes_Japan_Dataset \
#              ACCAD \
#              DFaust_67 \
#              BMLhandball \
#              SFU \
#              Transitions_mocap \
#              TCD_handMocap \
#              TotalCapture \
#              KIT \
#              HumanEva \
#              MPI_mosh \
#              BMLmovi \
#              SOMA \
#              MPI_Limits \
#              WEIZMANN \
#              EKUT \
#              SSM_synced \
#              GRAB \
#              DanceDB \
#              HUMAN4D \
#              CNRS



###### YOLO 8 protocol 3

# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose --extra-name yolo8_protocol3 --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -4.18 4.28 -4.1 4.38 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose 

# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name yolo8_protocol3 --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -4.18 4.28 -4.1 4.38 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose \
#     --train-datasets MPI_HDM05 \
#              BioMotionLab_NTroje \
#              CMU \
#              ACCAD \
#              BMLmovi \
#              EKUT \
#              Eyes_Japan_Dataset \
#              KIT \
#              MPI_Limits \
#              MPI_mosh \
#              SFU \
#              TotalCapture 


#### YOLO 11 protocol 3
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name yolo11_protocol3 --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -4.18 4.28 -4.1 4.38 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolo11n-pose \
#     --train-datasets MPI_HDM05 \
#              BioMotionLab_NTroje \
#              CMU \
#              ACCAD \
#              BMLmovi \
#              EKUT \
#              Eyes_Japan_Dataset \
#              KIT \
#              MPI_Limits \
#              MPI_mosh \
#              SFU \
#              TotalCapture 


#### RTMO-s protocol 3
python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp openmplposer_aligend --extra-name rtmos_protocol3 --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -4.18 4.28 -4.1 4.38 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model rtmo-s_8xb32-600e_coco-640x640 \
    --train-datasets MPI_HDM05 \
             BioMotionLab_NTroje \
             CMU \
             ACCAD \
             BMLmovi \
             EKUT \
             Eyes_Japan_Dataset \
             KIT \
             MPI_Limits \
             MPI_mosh \
             SFU \
             TotalCapture 

# YOLO random cameras
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose_split_250 --extra-name yolo_random --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose --save-temp-checkpoints --run-on-random-cameras --n-cameras-per-person 20 --camera-location-limit -4 4 -4 4 2.7 3.3 --dont-filter-outside-image-when-triangulating --camera-dist-from-person 2 --look-at-person --room-center 0 0 0
# python run_mmpose_02_run.py --dataset-split-number $split_number --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir  --exp all_with_mmpose_split_250 --extra-name yolo_random --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on validation --image-width 1280 --image-height 720 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model yolov8n-pose --save-temp-checkpoints --run-on-random-cameras --n-cameras-per-person 20 --camera-location-limit -4 4 -4 4 2.7 3.3 --dont-filter-outside-image-when-triangulating --camera-dist-from-person 2 --look-at-person --room-center 0 0 0

echo "All done"
