# 🚀 Welcome to RUMPL

**RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting**

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2512.15488)

Hi there! 👋 Welcome to the official repository for **RUMPL**. We are excited to have you here. This framework is designed to help you with 3D pose estimation and motion learning using the power of your dataset.

Below is everything you need to get set up and running smoothly. Let's dive in! 🏊‍♂️

---
## 🛠️ Installation

Getting started is easy. Just follow these steps to prepare your environment.

### 1. Install Python Dependencies
First things first, let's grab the necessary python packages:

```bash
pip install -r requirements.txt
```

### 2. External Tools
RUMPL relies on two awesome external frameworks. You'll need to install them separately:

* **🏃 AMASS Framework:** Required for running MHP.
  [👉 Install AMASS here](https://github.com/nghorbani/amass)
* **📸 MMPose:** Required for off-the-shelf 2D pose estimation.
  [👉 Install MMPose here](https://github.com/open-mmlab/mmpose)

---

## 📦 Dataset Setup

You will need the **AMASS dataset** for training (plus camera calibrations from your test dataset). RUMPL has been tested on **CMU**, **Human3.6M**, and **RICH**.

Here is how to prep each one:

### 🏛️ CMU Panoptic
1. Download the images and annotations using the [Panoptic Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox).
2. Once downloaded, correct the paths in our script and run:

    ```bash
    cd RUMPL/data
    sh preprocess_cmu_panoptic_all_cams.sh
    ```

### 🧍 Human3.6M (H3.6M)
1. Prepare images and annotations using the [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox).
2. Update your paths and run:

    ```bash
    cd RUMPL/data
    sh preprocess_h36m.sh
    ```

### 💰 RICH Dataset
1. Download and prepare data from the [RICH Website](https://rich.is.tue.mpg.de/).
2. Update your paths and run:

    ```bash
    cd RUMPL/data
    sh preprocess_rich.sh
    ```

---

## 🏃 Generating 3D Data (MHP)

To generate the 3D dataset for RUMPL, we use the MHP module.

1. **Download SMPL Data:** Head over to the [AMASS Website](https://amass.is.tue.mpg.de/index.html) and download the pose SMPL data (SMPL+H is supported).
2. **Run the Generation:** We have prepared scripts for each dataset. Make sure you have the [amass framework](https://github.com/nghorbani/amass) installed and your paths configured.

```bash
    cd MHP
    sh run_mmpose_00_cmu.sh      # For CMU
    sh run_mmpose_00_h36m.sh     # For Human3.6M
    sh run_mmpose_00_rich.sh     # For RICH
    ```

    > **💡 Pro Tip:** Take a peek inside these scripts before running them! There are some parallel processing commands that are commented out—you might want to enable them to speed things up.
```
---

## 🙌 Acknowledgements

We stand on the shoulders of giants! 🌍 Big thanks to the authors of these amazing projects:

* **[PPT](https://github.com/HowieMa/PPT/tree/main)**
* **[PoseFormer](https://github.com/zczcwh/PoseFormer/tree/main)**
* **[H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox)** (Microsoft)

---

## ⚖️ License & Copyright

**Copyright © 2024 UCLouvain.**
*Author: Seyed Abolfazl Ghasemzadeh*

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

This means you are free to use, modify, and distribute this software, but if you run it as a network service (SaaS), you must share your source code.

See the [LICENSE](LICENSE) file for the full legal text.

---

**Happy Coding! 💻✨**
If you have questions, run into bugs, or just want to say hi, feel free to **[open an issue](../../issues)**. We're here to help!

---

## 📝 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{ghasemzadeh2025rumpl,
  title={RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting},
  author={Ghasemzadeh, Seyed Abolfazl and Alahi, Alexandre and De Vleeschouwer, Christophe},
  journal={arXiv preprint arXiv:2512.15488},
  year={2025}
}
