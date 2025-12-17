#!/bin/bash
#SBATCH --job-name=rumpl
#SBATCH -c 16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --qos=preemptible
#SBATCH -x mb-cas001
echo "Job on $HOSTNAME"



# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_4_openmplposer_aligned_yolo11/rumpl_401_amass_yolo11_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val --test-openmplposer-dataset-name annot_yolo11n-pose_protocol_1
# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_4_openmplposer_aligned_yolo11/rumpl_401_amass_yolo11_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val --test-openmplposer-dataset-name annot_yolo11n-pose_protocol_2

# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_5_openmplposer_aligned_rtmos/rumpl_501_amass_rtmos_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val --test-openmplposer-dataset-name annot_rtmo-s_8xb32-600e_coco-640x640_protocol_1
# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_5_openmplposer_aligned_rtmos/rumpl_501_amass_rtmos_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val --test-openmplposer-dataset-name annot_rtmo-s_8xb32-600e_coco-640x640_protocol_2


# python run/valid_rumpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_3_openmplposer_aligned_yolo8/rumpl_301_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# python run/valid_rumpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_4_openmplposer_aligned_yolo11/rumpl_401_amass_yolo11_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# python run/valid_rumpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_poser_5_openmplposer_aligned_rtmos/rumpl_501_amass_rtmos_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_6_openmplposer_aligned_yolo8_p3/rumpl_601_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_7_openmplposer_aligned_yolo11_p3/rumpl_701_amass_yolo11_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# python run/valid_rumpl.py --cfg configs/openmplposer/rumpl_amass_poser_8_openmplposer_aligned_rtmos_p3/rumpl_801_amass_rtmos_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml --use-mmpose-val &
# wait


# python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_yolo8_p3.yaml
# python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_yolo‍‍11.yaml
# python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_yolo‍‍11_p3.yaml
# python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_rtmos.yaml
# python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_rtmos_p3.yaml


python run/triangulate.py --cfg configs/openmplposer/rumpl_amass_poser/rumpl_triangulation_yolo_antoine.yaml

echo "Job finished on $HOSTNAME at $(date)"