# md4allDD + DAV2 correction teacher + DAV2 Small student

This repository contains the code for the following training pipeline:

1. Start from the md4allDD self-supervised baseline.
2. Train a DAV2 correction network as the daytime teacher, using metric and relative DAV2 pseudo-depth, photometric loss, and odometry loss.
3. Distill the corrected teacher into a DAV2 Small metric student with the md4allDD training pipeline.

The checked-in configs are for Oxford RobotCar. Dataset files, pseudo-depth maps, pretrained weights, checkpoints, logs, and evaluation outputs are kept outside the repository.

## Setup

Install dependencies from `requirements.txt`. The DAV2 metric implementation used by the student and pseudo-label generator is included under [generate-pseudo/README.md](generate-pseudo/README.md), with its upstream license. Obtain the DAV2 Small metric pretrained checkpoint separately. Set the dataset and checkpoint paths in the two training configs before running. The correction teacher config expects RobotCar data, split/filter files, and both metric and relative pseudo-depth maps. The student config expects the trained correction-teacher checkpoint and the RobotCar day-to-night translated images.

## Train

Train the correction teacher:

```bash
python train.py --config config/train_dav2_correction_robotcar.yaml
```

After training, copy the selected teacher checkpoint to the path configured by `LOAD.DAYTIME_TRANSLATION_TEACHER_PATH` in `config/train_md4allDD_dav2_robotcar.yaml`. Then train the DAV2 Small student:

```bash
python train.py --config config/train_md4allDD_dav2_small_robotcar.yaml
```

The student config inherits the md4allDD distillation settings from `train_md4allDD_dav2_robotcar.yaml` and selects `dav2_small` as its depth model.

For the same DAV2Correction teacher → DAV2Small student method on nuScenes, see [the nuScenes guide](docs/dav2_correction_dav2_small_nuscenes.md) and the matching train/evaluation configs.

## Generate metric pseudo-depth

See [generate-pseudo/README.md](generate-pseudo/README.md) for the DAV2 metric pseudo-label generator and RobotCar command.

## Repository contents

Core md4all training, data, loss, evaluation, correction-teacher, and student code are retained. Large local datasets and generated artifacts are excluded by `.gitignore`.
