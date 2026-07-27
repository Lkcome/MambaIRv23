#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-image tiled evaluation for UPAMNet on Duke PAM.

Important:
- Images are read with cv2.IMREAD_UNCHANGED to preserve their original
  bit depth and intensity range.
- uint8 images are normalized by 255.
- uint16 images are normalized by 65535.
- Grayscale PAM images are copied to three channels for RGB saving.
- UPAMNet remains a single-channel model.
- Metrics are computed in the grayscale domain.
- Prediction is reconstructed as a complete image before evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)
from tqdm import tqdm

from train_upamnet import (
    DEFAULT_DATA_ROOT,
    build_pairs,
    create_model,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image tiled test of UPAMNet on Duke PAM."
    )

    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        choices=[2, 4, 8],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=4,
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
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments"),
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--save-limit",
        type=int,
        default=0,
        help="Maximum number of comparison images to save; 0 saves all.",
    )

    # Must match training.
    parser.add_argument(
        "--inner-channel",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--norm-groups",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--res-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tile_size != 128:
        raise ValueError(
            "UPAMNet was trained with 128x128 patches. "
            "Use --tile-size 128."
        )

    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < tile_size."
        )

    if args.tile_batch_size < 1:
        raise ValueError(
            "tile-batch-size must be at least 1."
        )


def normalize_raw_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an OpenCV image to float32 [0, 1] according to its dtype.

    This avoids PIL converting high-bit-depth PNG data directly to 8-bit
    and clipping most pixels to white.
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
    Read an image while preserving its original bit depth.

    Returns:
        H x W x 3 RGB float32 array in [0, 1].
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


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to one-channel float32 data.

    For replicated grayscale PAM images, all channels are equal and this
    operation returns the same original intensity.
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


def gray_to_rgb(image_gray: np.ndarray) -> np.ndarray:
    """
    Copy one grayscale channel into R, G, and B.
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
    Save float RGB image as an ordinary 8-bit RGB PNG.

    No inversion, colormap, or contrast stretching is used.
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
        mode="RGB",
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
    Compute metrics in the original grayscale domain.
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


def sliding_positions(
    length: int,
    tile_size: int,
    stride: int,
) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def make_blending_window(
    tile_size: int,
) -> np.ndarray:
    """
    Positive Hanning window for smooth tile fusion.
    """
    one_dimensional = np.hanning(
        tile_size,
    ).astype(np.float32)

    window = np.outer(
        one_dimensional,
        one_dimensional,
    )

    window = np.maximum(
        window,
        1e-3,
    )

    return window.astype(np.float32)


def pad_for_tiling(
    image: np.ndarray,
    tile_size: int,
) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape

    pad_bottom = max(
        0,
        tile_size - height,
    )

    pad_right = max(
        0,
        tile_size - width,
    )

    if (
        pad_bottom == 0
        and pad_right == 0
    ):
        return image, height, width

    padded = np.pad(
        image,
        (
            (0, pad_bottom),
            (0, pad_right),
        ),
        mode=(
            "reflect"
            if min(height, width) > 1
            else "edge"
        ),
    )

    return padded, height, width


@torch.no_grad()
def tiled_inference(
    model: torch.nn.Module,
    image_gray: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
) -> np.ndarray:
    padded, original_height, original_width = pad_for_tiling(
        image_gray,
        tile_size,
    )

    height, width = padded.shape
    stride = tile_size - overlap

    y_positions = sliding_positions(
        height,
        tile_size,
        stride,
    )

    x_positions = sliding_positions(
        width,
        tile_size,
        stride,
    )

    blending_window = make_blending_window(
        tile_size,
    )

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    pending_tiles: List[np.ndarray] = []
    pending_locations: List[Tuple[int, int]] = []

    def flush_tiles() -> None:
        if not pending_tiles:
            return

        tile_batch = np.stack(
            pending_tiles,
            axis=0,
        )[:, None, :, :]

        input_tensor = torch.from_numpy(
            tile_batch,
        ).float().to(device)

        residual_tensor = model(
            input_tensor
        )

        prediction_tensor = torch.clamp(
            input_tensor + residual_tensor,
            min=0.0,
            max=1.0,
        )

        predictions = (
            prediction_tensor[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        for prediction, location in zip(
            predictions,
            pending_locations,
        ):
            top, left = location

            prediction_sum[
                top:top + tile_size,
                left:left + tile_size,
            ] += (
                prediction
                * blending_window
            )

            weight_sum[
                top:top + tile_size,
                left:left + tile_size,
            ] += blending_window

        pending_tiles.clear()
        pending_locations.clear()

    for top in y_positions:
        for left in x_positions:
            tile = padded[
                top:top + tile_size,
                left:left + tile_size,
            ]

            pending_tiles.append(
                np.ascontiguousarray(
                    tile,
                    dtype=np.float32,
                )
            )

            pending_locations.append(
                (top, left)
            )

            if (
                len(pending_tiles)
                == tile_batch_size
            ):
                flush_tiles()

    flush_tiles()

    prediction = prediction_sum / np.maximum(
        weight_sum,
        1e-8,
    )

    prediction = prediction[
        :original_height,
        :original_width,
    ]

    return np.clip(
        prediction,
        0.0,
        1.0,
    ).astype(np.float32)


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
            Image.fromarray(
                panel,
                mode="RGB",
            ),
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
            f"UPAMNet  PSNR={output_psnr:.4f}  "
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


def mean_metric(
    rows: Sequence[Dict[str, object]],
    key: str,
) -> float:
    return float(
        np.mean(
            [
                float(row[key])
                for row in rows
            ]
        )
    )


def print_image_information(
    label: str,
    path: Path,
    image_rgb: np.ndarray,
) -> None:
    """
    Print the first image's source information so that bit-depth and
    normalization problems are visible directly in the log.
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
    set_random_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        f"cuda:{args.gpu}"
    )

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = (
            args.experiment_root
            / f"UPAMNet_DukePAM_x{args.scale}"
            / "checkpoints"
            / "best_perf.tar"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    validation_pairs = build_pairs(
        data_root=args.data_root,
        split="valid",
        scale=args.scale,
    )

    model = create_model(
        args
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    result_dir = (
        args.result_root
        / f"UPAMNet_DukePAM_x{args.scale}"
        / "valid_full"
    )

    input_dir = result_dir / "input"
    prediction_dir = result_dir / "pred"
    target_dir = result_dir / "hr"
    comparison_dir = result_dir / "comparison"

    for folder in (
        input_dir,
        prediction_dir,
        target_dir,
        comparison_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    metric_rows: List[Dict[str, object]] = []

    for image_index, pair in enumerate(
        tqdm(
            validation_pairs,
            desc=f"Full-image testing x{args.scale}",
        ),
        start=1,
    ):
        lr_path, hr_path = pair

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

        if image_index == 1:
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

        prediction_gray = tiled_inference(
            model=model,
            image_gray=input_gray,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_batch_size=args.tile_batch_size,
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
            or image_index <= args.save_limit
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

        metric_rows.append(
            {
                "index": image_index,
                "name": name,
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
            f"[{image_index:02d}/{len(validation_pairs):02d}] "
            f"{name}: "
            f"Input PSNR={input_psnr:.4f}, "
            f"SSIM={input_ssim:.4f} | "
            f"UPAMNet PSNR={output_psnr:.4f}, "
            f"SSIM={output_ssim:.4f}"
        )

    if not metric_rows:
        raise RuntimeError(
            "No validation images were tested."
        )

    summary_values = {
        "mean_input_rmse": mean_metric(
            metric_rows,
            "input_rmse",
        ),
        "mean_input_psnr": mean_metric(
            metric_rows,
            "input_psnr",
        ),
        "mean_input_ssim": mean_metric(
            metric_rows,
            "input_ssim",
        ),
        "mean_output_rmse": mean_metric(
            metric_rows,
            "output_rmse",
        ),
        "mean_output_psnr": mean_metric(
            metric_rows,
            "output_psnr",
        ),
        "mean_output_ssim": mean_metric(
            metric_rows,
            "output_ssim",
        ),
        "mean_psnr_gain": mean_metric(
            metric_rows,
            "psnr_gain",
        ),
        "mean_ssim_gain": mean_metric(
            metric_rows,
            "ssim_gain",
        ),
    }

    metrics_path = result_dir / "metrics.csv"

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=list(
                metric_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            metric_rows
        )

    summary_path = result_dir / "summary.txt"

    summary_lines = [
        f"scale: x{args.scale}",
        f"checkpoint: {checkpoint_path}",
        f"number of images: {len(metric_rows)}",
        "image reader: cv2.IMREAD_UNCHANGED",
        "metric domain: grayscale",
        "saved image mode: RGB",
        "evaluation: full-image tiled inference",
        f"crop border: {args.scale}",
        f"tile size: {args.tile_size}",
        f"overlap: {args.overlap}",
        f"tile batch size: {args.tile_batch_size}",
        (
            "mean input RMSE: "
            f"{summary_values['mean_input_rmse']:.6f}"
        ),
        (
            "mean input PSNR: "
            f"{summary_values['mean_input_psnr']:.6f}"
        ),
        (
            "mean input SSIM: "
            f"{summary_values['mean_input_ssim']:.6f}"
        ),
        (
            "mean output RMSE: "
            f"{summary_values['mean_output_rmse']:.6f}"
        ),
        (
            "mean output PSNR: "
            f"{summary_values['mean_output_psnr']:.6f}"
        ),
        (
            "mean output SSIM: "
            f"{summary_values['mean_output_ssim']:.6f}"
        ),
        (
            "mean PSNR gain: "
            f"{summary_values['mean_psnr_gain']:.6f}"
        ),
        (
            "mean SSIM gain: "
            f"{summary_values['mean_ssim_gain']:.6f}"
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
        f"RMSE={summary_values['mean_input_rmse']:.6f}, "
        f"PSNR={summary_values['mean_input_psnr']:.6f}, "
        f"SSIM={summary_values['mean_input_ssim']:.6f}"
    )

    print(
        "UPAMNet output: "
        f"RMSE={summary_values['mean_output_rmse']:.6f}, "
        f"PSNR={summary_values['mean_output_psnr']:.6f}, "
        f"SSIM={summary_values['mean_output_ssim']:.6f}"
    )

    print(f"Results: {result_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()