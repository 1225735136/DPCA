#!/usr/bin/env python3
"""Generate Depth Anything V2 metric-depth pseudo labels for md4allDD.

The output mirrors the source image tree.  For example,
``samples/CAM_FRONT/foo.jpg`` becomes ``<output>/samples/CAM_FRONT/foo.jpg.png``
and ``foo.jpg.txt``.  PNG stores a uint16 metric depth quantized with the
per-file scale written in the adjacent text file:

    depth_m = uint16_png / 65535 * float(scale_txt)

The default scale is 80 m, the valid range of the outdoor Virtual KITTI model.
Unlike the earlier V1 generator, this scale is fixed rather than batch
dependent, so a pixel value has the same meaning in every label.

No GPU work happens until this file is run (``--dry-run`` only enumerates
images).  It deliberately performs one-image inference: DepthAnythingV2's
``infer_image`` keeps the original image geometry and avoids a large batch's
VRAM requirement.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
# There are relative- and metric-depth packages with the same module name.
# Put metric_depth first so this script always imports the metric model.
sys.path.insert(0, str(SCRIPT_DIR / "metric_depth"))


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dataset_images(dataset: str, data_root: Path) -> List[Path]:
    """Return only camera RGB images, never previous labels or translations."""
    if dataset == "nuscenes":
        roots = [data_root / "samples" / camera for camera in (
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        )]
    else:
        roots = sorted(data_root.glob("20??-??-??-??-??-??/stereo/left"))

    images = []
    for root in roots:
        if root.is_dir():
            images.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images)


def default_condition(dataset: str, relative_path: Path) -> Optional[str]:
    """Small representative day/night selection matching the local md4all sets."""
    name = relative_path.name
    parts = relative_path.parts
    if dataset == "nuscenes":
        # n008 is a daytime log and n015 is a nighttime log in this dataset.
        if name.startswith("n008-"):
            return "day"
        if name.startswith("n015-"):
            return "night"
    else:
        # The standard md4all RobotCar split uses these two recordings.
        if "2014-12-09-13-21-02" in parts:
            return "day"
        if "2014-12-16-18-44-24" in parts:
            return "night"
    return None


def load_preview_selection(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if not isinstance(selection, dict) or not all(value in {"day", "night"} for value in selection.values()):
        raise ValueError("--preview-selection must be a JSON object mapping relative paths to 'day' or 'night'.")
    return selection


def write_manifest(args: argparse.Namespace, output_root: Path) -> None:
    checkpoint = Path(args.checkpoint)
    manifest = {
        "generator": "depth_anything_v2_metric",
        "format": "metric_depth_uint16_png_fixed_scale_v1",
        "representation": "metric_depth",
        "units": "meters",
        "quantization": "depth_m / png_scale_m * 65535",
        "png_scale_m": args.png_scale,
        "geometry": "original_uncropped_image",
        "dataset": args.dataset,
        "data_root": str(Path(args.data_root).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns if checkpoint.is_file() else None,
        "encoder": args.encoder,
        "model_max_depth_m": args.max_depth,
        "input_size": args.input_size,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def save_label(depth_m: np.ndarray, png_path: Path, scale_m: float, compression: int) -> None:
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=scale_m, neginf=0.0)
    quantized = np.rint(np.clip(depth_m, 0.0, scale_m) / scale_m * 65535.0).astype(np.uint16)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png_path), quantized, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
        raise IOError(f"Failed to write {png_path}")
    png_path.with_suffix(".txt").write_text(f"{scale_m:.8g}\n", encoding="utf-8")


def save_preview(raw_bgr: np.ndarray, depth_m: np.ndarray, path: Path, scale_m: float) -> None:
    visual = np.rint(np.clip(depth_m, 0.0, scale_m) / scale_m * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
    separator = np.full((raw_bgr.shape[0], 24, 3), 255, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.hconcat([raw_bgr, separator, colored])):
        raise IOError(f"Failed to write preview {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "robotcar"))
    parser.add_argument("--data-root", required=True, help="nuScenes trainval root or RobotCar md4all root")
    parser.add_argument("--output-root", required=True, help="New pseudo-label root; must not be inside an image root")
    parser.add_argument("--checkpoint", required=True, help="Depth Anything V2 metric VKITTI checkpoint")
    parser.add_argument("--encoder", choices=MODEL_CONFIGS, default="vitl")
    parser.add_argument("--max-depth", type=float, default=80.0, help="VKITTI model's metric output ceiling (m)")
    parser.add_argument("--png-scale", type=float, default=80.0, help="Meters represented by uint16 value 65535")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="cuda", help="e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--compression", type=int, choices=range(10), default=1)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate labels that already have both PNG and TXT")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many images (useful for a smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="List the planned count and exit without loading the model")
    parser.add_argument("--preview-count", type=int, default=0, help="Save up to this many previews for each of day and night")
    parser.add_argument("--preview-selection", type=Path, help="Optional JSON: {relative_image_path: 'day'|'night'}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.png_scale <= 0 or args.max_depth <= 0 or args.input_size <= 0:
        raise ValueError("--png-scale, --max-depth, and --input-size must be positive.")
    data_root, output_root = Path(args.data_root), Path(args.output_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    images = dataset_images(args.dataset, data_root)
    if args.limit is not None:
        images = images[:args.limit]
    print(f"Found {len(images)} {args.dataset} camera images under {data_root}")
    if args.dry_run:
        return

    # Keep imports here so --help and --dry-run neither require the inference
    # environment nor initialize CUDA while another training job is running.
    global cv2, torch, DepthAnythingV2
    import cv2
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu to explicitly run on CPU.")

    selection = load_preview_selection(args.preview_selection)
    write_manifest(args, output_root)
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], "max_depth": args.max_depth})
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(args.device).eval()
    preview_counts = {"day": 0, "night": 0}

    for image_path in tqdm(images, desc="Generating metric pseudo labels"):
        relative = image_path.relative_to(data_root)
        label_path = output_root / (str(relative) + ".png")
        scale_path = label_path.with_suffix(".txt")
        if not args.overwrite and label_path.is_file() and scale_path.is_file():
            continue
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            tqdm.write(f"Skipping unreadable image: {image_path}")
            continue
        with torch.inference_mode():
            depth_m = model.infer_image(raw_bgr, args.input_size)
        save_label(depth_m, label_path, args.png_scale, args.compression)

        condition = selection.get(str(relative), default_condition(args.dataset, relative))
        if args.preview_count > 0 and condition in preview_counts and preview_counts[condition] < args.preview_count:
            preview_path = output_root / "previews" / condition / (str(relative) + ".color.png")
            save_preview(raw_bgr, depth_m, preview_path, args.png_scale)
            preview_counts[condition] += 1


if __name__ == "__main__":
    main()
