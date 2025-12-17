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
rich_dir=$4
split_number=$5


python run_mmpose_02_run.py --dataset-split-number $split_number --exp all_with_mmpose_split_250 --extra-name random_20_rich_small_box_hips_cam_dist_2 --use-cams-from rich --calib-file-rich $calib_file_rich  --room-size -0.6 0.4 -0.4 0.4 -0.5 0.5 --operation-on train --image-width 4112 --image-height 3008 --apply-rotation --regressor coco --triangulate --triangulate-th 0.95 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 --save-temp-checkpoints --run-on-random-cameras --n-cameras-per-person 20 --camera-location-limit -6.8 6.7 -6.8 7.7 -1.3 3.7 --dont-filter-outside-image-when-triangulating --camera-dist-from-person 2 --look-at-person --room-center 0 0 -0.5 \
    --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir


echo "All done"
