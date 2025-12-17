#!/bin/bash

#SBATCH --job-name=amass_mmpose
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:TeslaA100_80:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
##SBATCH -w mb-icg102
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

support_dir=/globalscratch/users/a/b/abolfazl/amass_data/support_data   # path to the support data in amass directory
work_dir=/globalscratch/users/a/b/abolfazl/amass_data/support_data/prepared_data   # path to the support data in amass directory
amass_data_dir=/globalscratch/users/a/b/abolfazl/amass_data_poses #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
# calib_path=/globalscratch/users/a/b/abolfazl/OpenMPLPoser_files/cameras # path to the OpenMPLPoser cameras
calib_path=/home/ucl/elen/abolfazl/OpenMPLPoser/data/virtual_cameras/protocol_3 # path to the OpenMPLPoser cameras
### for preparing the amass dataset (only need to run once for both cmu and h36m)
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on train
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on validation

# python run_mmpose_01_create_dataset.py --exp openmplposer_aligend --n-splits 100 --work-dir $work_dir --amass-data-dir $amass_data_dir --operation-on train --train-datasets Eyes_Japan_Dataset \
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

### to run MHP in parallel (if you dont touch the split number in the last step, no need to change these)
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 0
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 1
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 2
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 3
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 4
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 5
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 6
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 7
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 8
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 9
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 10
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 11
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 12
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 13
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 14
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 15
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 16
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 17
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 18
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 19
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 20
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 21
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 22
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 23
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 24
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 25
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 26
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 27
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 28
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 29
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 30
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 31
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 32
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 33
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 34
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 35
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 36
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 37
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 38
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 39
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 40
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 41
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 42
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 43
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 44
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 45
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 46
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 47
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 48
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 49
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 50
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 51
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 52
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 53
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 54
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 55
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 56
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 57
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 58
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 59
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 60
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 61
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 62
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 63
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 64
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 65
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 66
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 67
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 68
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 69
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 70
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 71
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 72
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 73
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 74
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 75
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 76
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 77
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 78
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 79
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 80
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 81
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 82
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 83
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 84
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 85
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 86
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 87
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 88
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 89
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 90
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 91
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 92
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 93
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 94
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 95
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 96
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 97
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 98
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 99

# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 100
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 101
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 102
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 103
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 104
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 105
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 106
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 107
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 108
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 109
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 110
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 111
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 112
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 113
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 114
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 115
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 116
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 117
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 118
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 119
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 120
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 121
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 122
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 123
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 124
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 125
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 126
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 127
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 128
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 129
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 130
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 131
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 132
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 133
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 134
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 135
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 136
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 137
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 138
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 139
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 140
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 141
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 142
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 143
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 144
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 145
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 146
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 147
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 148
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 149
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 150
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 151
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 152
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 153
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 154
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 155
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 156
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 157
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 158
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 159
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 160
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 161
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 162
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 163
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 164
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 165
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 166
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 167
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 168
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 169
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 170
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 171
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 172
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 173
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 174
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 175
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 176
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 177
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 178
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 179
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 180
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 181
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 182
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 183
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 184
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 185
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 186
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 187
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 188
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 189
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 190
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 191
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 192
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 193
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 194
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 195
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 196
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 197
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 198
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 199
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 200
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 201
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 202
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 203
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 204
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 205
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 206
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 207
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 208
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 209
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 210
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 211
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 212
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 213
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 214
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 215
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 216
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 217
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 218
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 219
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 220
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 221
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 222
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 223
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 224
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 225
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 226
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 227
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 228
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 229
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 230
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 231
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 232
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 233
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 234
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 235
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 236
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 237
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 238
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 239
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 240
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 241
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 242
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 243
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 244
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 245
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 246
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 247
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 248




# to combine all the results
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on train
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on validation

# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name yolo --operation-on train
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name yolo --operation-on validation


# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name yolo --operation-on train
# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name hrnet --operation-on train

# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name yolo11 --operation-on train
# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name rtmos --operation-on train

python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name yolo8_protocol3 --operation-on train
# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name yolo11_protocol3 --operation-on train
# python run_mmpose_04_combine.py --exp openmplposer_aligend --work-dir $work_dir --extra-name rtmos_protocol3 --operation-on train


# python run_mmpose_04_combine.py --exp all_with_mmpose_split_250 --work-dir $work_dir --extra-name yolo_random --operation-on train --n-splits 250
# python run_mmpose_04_combine.py --exp all_with_mmpose_split_250 --work-dir $work_dir --extra-name yolo_random --operation-on validation --n-splits 250

echo "All done"
