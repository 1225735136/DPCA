# DPCA
The official implementation of "DPCA: Depth Prior Correction for All-Day Monocular Depth Estimation".
# DAV2Correction Teacher → DAV2Small Student

This repository implements a monocular depth estimation method based on the md4allDD baseline:

1. Train a DAV2Correction network as the teacher.
2. Distill the corrected teacher into a DAV2Small metric-depth student.

The nuScenes and Oxford RobotCar configurations are included. Training datasets, pseudo-depth maps, pretrained weights, checkpoints, and experiment outputs are not included.

## Method

The correction teacher takes an RGB image together with offline DAV2 metric-depth and relative-depth pseudo-labels. It learns a per-pixel depth correction using photometric reconstruction, odometry, and smoothness losses.

The DAV2Small student takes RGB images only. During training, it learns from the frozen correction teacher using the md4allDD distillation pipeline. At inference and evaluation time, the student does not need the teacher or pseudo-depth maps.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt

The DAV2 metric implementation used by the student and pseudo-depth generator is included in generate-pseudo/. Download the required pretrained weights separately and update their paths in the configuration files.

Prepare the dataset and pseudo-depth maps separately. The correction teacher requires both metric and relative pseudo-depth maps. Update dataset, pseudo-depth, translation-image, and checkpoint paths in the relevant YAML files before training.

## Train on nuScenes

First, train the correction teacher:

python train.py --config config/train_dav2_correction_nuscenes.yaml

After training, select a teacher checkpoint using daytime validation. Set its path in config/train_md4allDD_dav2_small_nuscenes.yaml under LOAD.DAYTIME_TRANSLATION_TEACHER_PATH.

Then train the DAV2Small student:

python train.py --config config/train_md4allDD_dav2_small_nuscenes.yaml

Evaluate the student on the nuScenes validation set:

python evaluation/evaluate_depth.py \
  --config config/eval_md4allDD_dav2_small_nuscenes_val.yaml

Student evaluation uses RGB images and LiDAR ground truth for metrics. It does not load pseudo-depth maps or run the correction teacher.

## Train on Oxford RobotCar

Train the correction teacher:

python train.py --config config/train_dav2_correction_robotcar.yaml

Set the selected teacher checkpoint path in config/train_md4allDD_dav2_robotcar.yaml, then train the DAV2Small student:

python train.py --config config/train_md4allDD_dav2_small_robotcar.yaml

## Generate Metric Pseudo-Depth

The DAV2 metric pseudo-depth generator and usage instructions are in generate-pseudo/ (generate-pseudo/README.md). Pretrained weights and generated pseudo-depth maps must be obtained or generated separately.
