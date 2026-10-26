# DAV2 metric pseudo-depth generator

This folder contains the project-specific generator and the Depth Anything V2 metric model code it imports. It writes one uint16 PNG and one `.txt` scale file per input image, plus a `manifest.json`. The generated labels belong in an external dataset directory and are not part of this code repository.

## Requirements

Install compatible versions of PyTorch, torchvision, OpenCV, NumPy, and tqdm. Download a Depth Anything V2 metric checkpoint separately; checkpoints are not included here. The upstream Depth Anything V2 license is included as `DEPTH_ANYTHING_V2_LICENSE`.

## RobotCar

Set the dataset root and checkpoint paths, then run:

```bash
export ROBOTCAR_DATA_ROOT=/path/to/robotcar-md4all
export ROBOTCAR_PSEUDO_OUTPUT=/path/to/robotcar-md4all/pseudo_labels_depthanything_v2_metric_png
export DAV2_METRIC_CHECKPOINT=/path/to/depth_anything_v2_metric_vkitti_vitl.pth
bash generate-pseudo/generate_robotcar_metric_pseudo_labels.sh
```

For a short preview run, pass generator options through the wrapper:

```bash
bash generate-pseudo/generate_robotcar_metric_pseudo_labels.sh --limit 10 --preview-count 2
```

The Python script also supports `--dataset nuscenes`, a dry run, a processing limit, and preview selection. Run `python generate-pseudo/generate_metric_pseudo_labels.py --help` for all options.

Depth labels use fixed-scale quantization: `depth_m = uint16_png / 65535 * png_scale_m`. The manifest records the format, model, checkpoint metadata, image geometry, and scale.
