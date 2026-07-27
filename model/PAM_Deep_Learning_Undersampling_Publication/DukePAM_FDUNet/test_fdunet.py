#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-image tiled evaluation for FD-Unet on Duke PAM.

Important:
- Images are read with cv2.IMREAD_UNCHANGED to preserve the original
  bit depth and intensity range.
- uint8 images are normalized by 255.
- uint16 images are normalized by 65535.
- Grayscale PAM images are copied to three channels only for RGB saving.
- FD-Unet remains a single-channel model.
- Metrics are computed in the original grayscale domain.
- Complete images are reconstructed by overlapping tiled inference before
  PSNR, SSIM, and RMSE calculation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from FD_UNet import getModel
from train_fdunet import (
    build_pairs,
    tiled_predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image tiled evaluation of FD-Unet on Duke PAM."
    )

    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4, 8],
        required=True,
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Default: "
            "checkpoints/FDUNet_DukePAM_x{scale}/best_weights.h5"
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
    )

    parser.add_argument(
        "--save-limit",
        type=int,
        default=0,
        help="Maximum number of comparison images to save; 0 saves all.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tile_size != 128:
        raise ValueError(
            "FD-Unet was trained with 128x128 patches. "
            "Use --tile-size 128."
        )

    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < tile_size."
        )


def normalize_raw_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an OpenCV image to float32 [0, 1] according to its dtype.

    This avoids converting uint16 PAM PNG files directly to uint8 and
    clipping most high-intensity pixels.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0

    elif np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float32) / maximum

    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

        minimum = float(np.nanmin(image))
        maximum = float(np.nanmax(image))

        if minimum < 0.0 or maximum > 1.0:
            if maximum > minimum:
                image = (
                    image - minimum
                ) / (
                    maximum - minimum
                )
            else:
                image = np.zeros_like(
                    image,
                    dtype=np.float32,
                )

    else:
        raise TypeError(
            f"Unsupported image dtype: {image.dtype}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


def read_rgb(path: Path) -> np.ndarray:
    """
    Read image while preserving its original bit depth.

    Returns:
        H x W x 3 RGB float32 image in [0, 1].
    """
    raw = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if raw is None:
        raise FileNotFoundError(
            f"Failed to read image: {path}"
        )

    raw = normalize_raw_image(raw)

    if raw.ndim == 2:
        rgb = np.repeat(
            raw[..., None],
            3,
            axis=2,
        )

    elif raw.ndim == 3:
        channels = raw.shape[2]

        if channels == 1:
            rgb = np.repeat(
                raw,
                3,
                axis=2,
            )

        elif channels == 3:
            rgb = cv2.cvtColor(
                raw,
                cv2.COLOR_BGR2RGB,
            )

        elif channels == 4:
            rgb = cv2.cvtColor(
                raw,
                cv2.COLOR_BGRA2RGB,
            )

        else:
            raise ValueError(
                f"Unsupported channel count {channels}: {path}"
            )

    else:
        raise ValueError(
            f"Unsupported image shape {raw.shape}: {path}"
        )

    return np.ascontiguousarray(
        rgb,
        dtype=np.float32,
    )


def rgb_to_gray(
    image_rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert RGB image to H x W grayscale float32.

    For grayscale PAM copied to RGB, this preserves the original intensity.
    """
    if (
        image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
    ):
        raise ValueError(
            f"Expected HxWx3 RGB image, got {image_rgb.shape}."
        )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return np.ascontiguousarray(
        gray,
        dtype=np.float32,
    )


def gray_to_model_input(
    image_gray: np.ndarray,
) -> np.ndarray:
    """
    Convert H x W grayscale image to H x W x 1 for FD-Unet.
    """
    image_gray = np.asarray(
        image_gray,
        dtype=np.float32,
    )

    image_gray = np.squeeze(image_gray)

    if image_gray.ndim != 2:
        raise ValueError(
            f"Expected HxW grayscale image, got {image_gray.shape}."
        )

    return np.ascontiguousarray(
        image_gray[..., None],
        dtype=np.float32,
    )


def gray_to_rgb(
    image_gray: np.ndarray,
) -> np.ndarray:
    """
    Copy one grayscale channel to RGB for saving.
    """
    image_gray = np.asarray(
        image_gray,
        dtype=np.float32,
    )

    image_gray = np.squeeze(image_gray)

    if image_gray.ndim != 2:
        raise ValueError(
            f"Expected HxW grayscale image, got {image_gray.shape}."
        )

    return np.repeat(
        image_gray[..., None],
        3,
        axis=2,
    )


def resize_rgb_to_match(
    image_rgb: np.ndarray,
    target_shape: Sequence[int],
) -> np.ndarray:
    target_height = int(target_shape[0])
    target_width = int(target_shape[1])

    if image_rgb.shape[:2] == (
        target_height,
        target_width,
    ):
        return image_rgb

    resized = cv2.resize(
        image_rgb,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    return np.clip(
        resized,
        0.0,
        1.0,
    ).astype(np.float32)


def save_rgb(
    image_rgb: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save float RGB image as an 8-bit RGB PNG.

    No inversion, pseudocolor, or contrast stretching is applied.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_uint8 = np.clip(
        image_rgb * 255.0,
        0.0,
        255.0,
    ).round().astype(np.uint8)

    Image.fromarray(
        image_uint8,
    ).save(output_path)


def crop_metric_border(
    image: np.ndarray,
    border: int,
) -> np.ndarray:
    if border <= 0:
        return image

    height, width = image.shape[:2]

    if (
        height <= 2 * border
        or width <= 2 * border
    ):
        raise ValueError(
            f"Image shape {image.shape} is too small "
            f"for crop border {border}."
        )

    return image[
        border:-border,
        border:-border,
    ]


def calculate_gray_metrics(
    target_gray: np.ndarray,
    image_gray: np.ndarray,
    border: int,
) -> Tuple[float, float, float]:
    """
    Calculate RMSE, PSNR, and SSIM in the grayscale domain.
    """
    target_cropped = crop_metric_border(
        target_gray,
        border,
    )

    image_cropped = crop_metric_border(
        image_gray,
        border,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    image_cropped
                    - target_cropped
                )
                ** 2
            )
        )
    )

    psnr = float(
        peak_signal_noise_ratio(
            target_cropped,
            image_cropped,
            data_range=1.0,
        )
    )

    ssim = float(
        structural_similarity(
            target_cropped,
            image_cropped,
            data_range=1.0,
        )
    )

    return rmse, psnr, ssim


def create_comparison_image(
    input_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
    target_rgb: np.ndarray,
    input_psnr: float,
    input_ssim: float,
    output_psnr: float,
    output_ssim: float,
    output_path: Path,
) -> None:
    panels = [
        np.clip(
            input_rgb * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8),

        np.clip(
            prediction_rgb * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8),

        np.clip(
            target_rgb * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8),
    ]

    height, width = panels[0].shape[:2]
    label_height = 54

    canvas = Image.new(
        "RGB",
        (
            width * 3,
            height + label_height,
        ),
        color=(0, 0, 0),
    )

    for panel_index, panel in enumerate(panels):
        canvas.paste(
            Image.fromarray(panel),
            (
                panel_index * width,
                label_height,
            ),
        )

    drawing = ImageDraw.Draw(canvas)

    drawing.text(
        (8, 8),
        (
            f"Input  PSNR={input_psnr:.4f}  "
            f"SSIM={input_ssim:.4f}"
        ),
        fill=(255, 255, 255),
    )

    drawing.text(
        (width + 8, 8),
        (
            f"FD-Unet  PSNR={output_psnr:.4f}  "
            f"SSIM={output_ssim:.4f}"
        ),
        fill=(255, 255, 255),
    )

    drawing.text(
        (width * 2 + 8, 8),
        "HR / Ground Truth",
        fill=(255, 255, 255),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)


def print_image_information(
    label: str,
    path: Path,
    image_rgb: np.ndarray,
) -> None:
    """
    Print the first image's raw dtype and normalized range.
    """
    raw = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    print(
        f"{label} path: {path}"
    )
    print(
        f"{label} raw dtype: {raw.dtype}"
    )
    print(
        f"{label} raw shape: {raw.shape}"
    )
    print(
        f"{label} raw min/max: "
        f"{raw.min()} / {raw.max()}"
    )
    print(
        f"{label} normalized min/max: "
        f"{image_rgb.min():.8f} / "
        f"{image_rgb.max():.8f}"
    )
    print(
        f"{label} normalized mean: "
        f"{image_rgb.mean():.8f}"
    )


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    weights_path = (
        args.weights
        if args.weights is not None
        else (
            Path("checkpoints")
            / f"FDUNet_DukePAM_x{args.scale}"
            / "best_weights.h5"
        )
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}"
        )

    output_root = (
        Path("results")
        / f"FDUNet_DukePAM_x{args.scale}"
        / args.split
    )

    prediction_dir = output_root / "pred"
    input_dir = output_root / "bicubic_lr"
    target_dir = output_root / "hr"
    comparison_dir = output_root / "comparison"

    for folder in (
        prediction_dir,
        input_dir,
        target_dir,
        comparison_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    model = getModel(
        input_shape=(128, 128, 1),
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal",
    )

    model.load_weights(
        str(weights_path)
    )

    pairs = build_pairs(
        args.split,
        args.scale,
    )

    rows = []

    input_rmse_values = []
    input_psnr_values = []
    input_ssim_values = []

    output_rmse_values = []
    output_psnr_values = []
    output_ssim_values = []

    for index, pair in enumerate(
        pairs,
        start=1,
    ):
        lr_path, hr_path = pair

        lr_path = Path(lr_path)
        hr_path = Path(hr_path)

        input_rgb = read_rgb(
            lr_path
        )

        target_rgb = read_rgb(
            hr_path
        )

        input_rgb = resize_rgb_to_match(
            input_rgb,
            target_rgb.shape,
        )

        if index == 1:
            print()
            print_image_information(
                "LR",
                lr_path,
                input_rgb,
            )
            print_image_information(
                "HR",
                hr_path,
                target_rgb,
            )
            print()

        input_gray = rgb_to_gray(
            input_rgb
        )

        target_gray = rgb_to_gray(
            target_rgb
        )

        model_input = gray_to_model_input(
            input_gray
        )

        prediction = tiled_predict(
            model,
            model_input,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        prediction = np.squeeze(
            prediction
        )

        if prediction.ndim != 2:
            raise ValueError(
                f"Unexpected FD-Unet output shape: {prediction.shape}"
            )

        prediction_gray = np.clip(
            prediction,
            0.0,
            1.0,
        )

        prediction_rgb = gray_to_rgb(
            prediction_gray
        )

        input_rmse, input_psnr, input_ssim = (
            calculate_gray_metrics(
                target_gray=target_gray,
                image_gray=input_gray,
                border=args.scale,
            )
        )

        output_rmse, output_psnr, output_ssim = (
            calculate_gray_metrics(
                target_gray=target_gray,
                image_gray=prediction_gray,
                border=args.scale,
            )
        )

        input_rmse_values.append(
            input_rmse
        )
        input_psnr_values.append(
            input_psnr
        )
        input_ssim_values.append(
            input_ssim
        )

        output_rmse_values.append(
            output_rmse
        )
        output_psnr_values.append(
            output_psnr
        )
        output_ssim_values.append(
            output_ssim
        )

        name = hr_path.stem
        filename = f"{name}.png"

        save_rgb(
            input_rgb,
            input_dir / filename,
        )

        save_rgb(
            prediction_rgb,
            prediction_dir / filename,
        )

        save_rgb(
            target_rgb,
            target_dir / filename,
        )

        if (
            args.save_limit == 0
            or index <= args.save_limit
        ):
            create_comparison_image(
                input_rgb=input_rgb,
                prediction_rgb=prediction_rgb,
                target_rgb=target_rgb,
                input_psnr=input_psnr,
                input_ssim=input_ssim,
                output_psnr=output_psnr,
                output_ssim=output_ssim,
                output_path=(
                    comparison_dir
                    / f"{name}_comparison.png"
                ),
            )

        rows.append(
            {
                "index": index,
                "name": name,
                "lr_path": str(lr_path),
                "hr_path": str(hr_path),
                "height": target_gray.shape[0],
                "width": target_gray.shape[1],
                "input_rmse": input_rmse,
                "input_psnr": input_psnr,
                "input_ssim": input_ssim,
                "output_rmse": output_rmse,
                "output_psnr": output_psnr,
                "output_ssim": output_ssim,
                "psnr_gain": (
                    output_psnr
                    - input_psnr
                ),
                "ssim_gain": (
                    output_ssim
                    - input_ssim
                ),
            }
        )

        print(
            f"[{index:02d}/{len(pairs):02d}] "
            f"{name}: "
            f"Input PSNR={input_psnr:.4f}, "
            f"SSIM={input_ssim:.4f} | "
            f"FD-Unet PSNR={output_psnr:.4f}, "
            f"SSIM={output_ssim:.4f}"
        )

    if not rows:
        raise RuntimeError(
            "No images were tested."
        )

    mean_input_rmse = float(
        np.mean(input_rmse_values)
    )

    mean_input_psnr = float(
        np.mean(input_psnr_values)
    )

    mean_input_ssim = float(
        np.mean(input_ssim_values)
    )

    mean_output_rmse = float(
        np.mean(output_rmse_values)
    )

    mean_output_psnr = float(
        np.mean(output_psnr_values)
    )

    mean_output_ssim = float(
        np.mean(output_ssim_values)
    )

    csv_path = output_root / "metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    summary_path = output_root / "summary.txt"

    summary_lines = [
        f"scale: x{args.scale}",
        f"split: {args.split}",
        f"weights: {weights_path}",
        f"number of images: {len(rows)}",
        "image reader: cv2.IMREAD_UNCHANGED",
        "metric domain: grayscale",
        "saved image mode: RGB",
        "evaluation: full-image tiled inference",
        f"crop border: {args.scale}",
        f"tile size: {args.tile_size}",
        f"overlap: {args.overlap}",
        f"mean input RMSE: {mean_input_rmse:.6f}",
        f"mean input PSNR: {mean_input_psnr:.6f}",
        f"mean input SSIM: {mean_input_ssim:.6f}",
        f"mean output RMSE: {mean_output_rmse:.6f}",
        f"mean output PSNR: {mean_output_psnr:.6f}",
        f"mean output SSIM: {mean_output_ssim:.6f}",
        (
            "mean PSNR gain: "
            f"{mean_output_psnr - mean_input_psnr:.6f}"
        ),
        (
            "mean SSIM gain: "
            f"{mean_output_ssim - mean_input_ssim:.6f}"
        ),
    ]

    summary_path.write_text(
        "\n".join(
            summary_lines
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("Testing completed.")

    print(
        "Input baseline: "
        f"RMSE={mean_input_rmse:.6f}, "
        f"PSNR={mean_input_psnr:.6f}, "
        f"SSIM={mean_input_ssim:.6f}"
    )

    print(
        "FD-Unet output: "
        f"RMSE={mean_output_rmse:.6f}, "
        f"PSNR={mean_output_psnr:.6f}, "
        f"SSIM={mean_output_ssim:.6f}"
    )

    print(
        f"Results: {output_root}"
    )

    print(
        f"Metrics: {csv_path}"
    )

    print(
        f"Visual comparisons: {comparison_dir}"
    )


if __name__ == "__main__":
    main()