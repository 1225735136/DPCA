# nuScenes: DAV2Correction teacher → DAV2Small student

This is the nuScenes configuration for the method in this repository:

1. Train a DAV2Correction teacher from daytime RGB, offline DAV2 metric and relative pseudo-depth, photometric reconstruction, and odometry loss.
2. Distill the correction teacher into an RGB-only DAV2Small student with the md4allDD supervised-distillation pipeline.

The configs use nuScenes `trainval`, 320×576 input, four-scale AbsRel distillation, and the weather augmentation paths used by this method. LiDAR depth is for validation only; it is not a student training target.

## Paths to configure

In `config/train_dav2_correction_nuscenes.yaml`, set:

- `DATASET.DATAROOT` to the md4all nuScenes root (the code appends `trainval`).
- `DATASET.PSEUDO_DEPTH.METRIC_PATH` and `RELATIVE_PATH` to the generated DAV2 label directories.

In `config/train_md4allDD_dav2_small_nuscenes.yaml`, set:

- `LOAD.DAY_NIGHT_TRANSLATION_PATH` and `DAY_CLEAR_DAY_RAIN_TRANSLATION_PATH` to the translated image folders.
- `MODEL.DEPTH.DAV2.PRETRAINED_PATH` to the DAV2 metric ViT-S checkpoint.

## Run

Train the correction teacher:

```bash
python train.py --config config/train_dav2_correction_nuscenes.yaml
```

Copy the teacher checkpoint selected by daytime validation to
`log/checkpoints/dav2_correction_nuscenes/teacher.ckpt`, the path used by the student config. Then train the DAV2Small student:

```bash
python train.py --config config/train_md4allDD_dav2_small_nuscenes.yaml
```

Evaluate the trained student on nuScenes validation. Override the checkpoint path with the selected best student checkpoint if needed:

```bash
python evaluation/evaluate_depth.py --config config/eval_md4allDD_dav2_small_nuscenes_val.yaml
```

Student validation and evaluation run without loading pseudo-depth maps or evaluating the teacher. Pseudo-depth maps are required for the correction-teacher training phase and for teacher targets during student training.
