"""Generate slice-sweep videos for CT volumes in images/."""
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

FPS = 30
# CT windowing: clip to soft-tissue window by default
WIN_MIN = -1000
WIN_MAX = 1000
AXES = [
    ("axial",    2),  # sweep along z
    ("coronal",  1),  # sweep along y
    ("sagittal", 0),  # sweep along x
]


def to_uint8(slice_2d, win_min=WIN_MIN, win_max=WIN_MAX):
    clipped = np.clip(slice_2d, win_min, win_max)
    scaled = (clipped - win_min) / (win_max - win_min) * 255.0
    return scaled.astype(np.uint8)


def write_video(vol, out_path, axis_idx, fps=FPS):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Video export requires pip install opencv-python-headless") from exc
    n_slices = vol.shape[axis_idx]
    # get a sample slice to determine output frame size
    sample = np.take(vol, 0, axis=axis_idx)
    h, w = sample.shape
    # ensure even dimensions for H.264
    h_out = h + (h % 2)
    w_out = w + (w % 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w_out, h_out), isColor=False)
    for i in range(n_slices):
        sl = np.take(vol, i, axis=axis_idx)
        frame = to_uint8(sl)
        if frame.shape != (h_out, w_out):
            frame = cv2.copyMakeBorder(frame, 0, h_out - h, 0, w_out - w,
                                       cv2.BORDER_CONSTANT, value=0)
        writer.write(frame)
    writer.release()
    print(f"  wrote {out_path} ({n_slices} frames)")


def process_volume(nii_path, output_dir):
    img = nib.load(nii_path)
    vol = img.get_fdata(dtype=np.float32)
    # derive output prefix from directory + filename
    parts = nii_path.replace("\\", "/").split("/")
    dataset = parts[-2]   # e.g. baseline
    sample = os.path.splitext(os.path.splitext(parts[-1])[0])[0]  # strip .nii.gz
    print(f"Processing {dataset}/{sample}  shape={vol.shape}")
    for axis_name, axis_idx in AXES:
        out_name = f"{dataset}__{sample}__{axis_name}.mp4"
        out_path = os.path.join(output_dir, out_name)
        write_video(vol, out_path, axis_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.rglob("*.nii.gz"))
    if not paths:
        raise SystemExit("No .nii.gz files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        process_volume(str(path), args.output_dir)


if __name__ == "__main__":
    main()
