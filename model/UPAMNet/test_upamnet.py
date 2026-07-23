#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-image tiled evaluation for UPAMNet on the Duke PAM validation set.

Key changes from the original test script:
- evaluates every complete validation image instead of one 128x128 center crop;
- uses overlapping 128x128 tiled inference with weighted fusion;
- reports both input baseline and restored-image RMSE/PSNR/SSIM;
- saves full-resolution input, prediction, HR, and comparison images;
- records per-image metrics and the checkpoint used.
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from train_upamnet import (
    DEFAULT_DATA_ROOT,
    build_pairs,
    create_model,
    read_grayscale,
    resize_to_match,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image tiled test of UPAMNet on Duke PAM."
    )

    parser.add_argument("--scale", type=int, required=True, choices=[2, 4, 8])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=4,
        help="Number of 128x128 tiles inferred together.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size. Keep 128 to match UPAMNet training.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap between adjacent tiles.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--save-limit",
        type=int,
        default=0,
        help="Maximum number of comparison images to save; 0 saves all.",
    )

    # Must match training.
    parser.add_argument("--inner-channel", type=int, default=16)
    parser.add_argument("--norm-groups", type=int, default=16)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tile_size != 128:
        raise ValueError(
            "UPAMNet was trained with image_size=128; use --tile-size 128."
        )
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")
    if args.tile_batch_size < 1:
        raise ValueError("tile-batch-size must be at least 1.")


def save_grayscale(image_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(output_path)


def metric_triplet(target: np.ndarray, image: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((image - target) ** 2)))
    psnr = float(peak_signal_noise_ratio(target, image, data_range=1.0))
    ssim = float(structural_similarity(target, image, data_range=1.0))
    return rmse, psnr, ssim


def sliding_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def make_blending_window(tile_size: int) -> np.ndarray:
    """Smooth positive window; nonzero borders avoid uncovered edge pixels."""
    one_dimensional = np.hanning(tile_size).astype(np.float32)
    window = np.outer(one_dimensional, one_dimensional)
    window = np.maximum(window, 1e-3)
    return window.astype(np.float32)


def pad_for_tiling(image: np.ndarray, tile_size: int) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)

    if pad_bottom == 0 and pad_right == 0:
        return image, height, width

    padded = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right)),
        mode="reflect" if min(height, width) > 1 else "edge",
    )
    return padded, height, width


@torch.no_grad()
def tiled_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
) -> np.ndarray:
    padded, original_height, original_width = pad_for_tiling(image, tile_size)
    height, width = padded.shape
    stride = tile_size - overlap

    y_positions = sliding_positions(height, tile_size, stride)
    x_positions = sliding_positions(width, tile_size, stride)

    blending_window = make_blending_window(tile_size)
    prediction_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    pending_tiles: List[np.ndarray] = []
    pending_locations: List[Tuple[int, int]] = []

    def flush_tiles() -> None:
        if not pending_tiles:
            return

        tile_batch = np.stack(pending_tiles, axis=0)[:, None, :, :]
        input_tensor = torch.from_numpy(tile_batch).float().to(device)

        prediction_tensor = torch.clamp(
            input_tensor + model(input_tensor),
            min=0.0,
            max=1.0,
        )
        predictions = prediction_tensor[:, 0].cpu().numpy()

        for prediction, (top, left) in zip(predictions, pending_locations):
            prediction_sum[
                top : top + tile_size,
                left : left + tile_size,
            ] += prediction * blending_window
            weight_sum[
                top : top + tile_size,
                left : left + tile_size,
            ] += blending_window

        pending_tiles.clear()
        pending_locations.clear()

    for top in y_positions:
        for left in x_positions:
            tile = padded[
                top : top + tile_size,
                left : left + tile_size,
            ]
            pending_tiles.append(np.ascontiguousarray(tile))
            pending_locations.append((top, left))

            if len(pending_tiles) == tile_batch_size:
                flush_tiles()

    flush_tiles()

    prediction = prediction_sum / np.maximum(weight_sum, 1e-8)
    return prediction[:original_height, :original_width]


def create_comparison_image(
    input_image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    input_psnr: float,
    input_ssim: float,
    output_psnr: float,
    output_ssim: float,
    output_path: Path,
) -> None:
    panels = [
        np.clip(input_image * 255.0, 0, 255).astype(np.uint8),
        np.clip(prediction * 255.0, 0, 255).astype(np.uint8),
        np.clip(target * 255.0, 0, 255).astype(np.uint8),
    ]

    height, width = panels[0].shape
    label_height = 54
    canvas = Image.new("L", (width * 3, height + label_height), color=0)

    for panel_index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (panel_index * width, label_height))

    drawing = ImageDraw.Draw(canvas)
    drawing.text(
        (8, 8),
        f"Input  PSNR={input_psnr:.4f}  SSIM={input_ssim:.4f}",
        fill=255,
    )
    drawing.text(
        (width + 8, 8),
        f"UPAMNet  PSNR={output_psnr:.4f}  SSIM={output_ssim:.4f}",
        fill=255,
    )
    drawing.text((width * 2 + 8, 8), "HR / Ground Truth", fill=255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    set_random_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device(f"cuda:{args.gpu}")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            args.experiment_root
            / f"UPAMNet_DukePAM_x{args.scale}"
            / "checkpoints"
            / "best_perf.tar"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    validation_pairs = build_pairs(
        data_root=args.data_root,
        split="valid",
        scale=args.scale,
    )

    model = create_model(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    for folder in (input_dir, prediction_dir, target_dir, comparison_dir):
        folder.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for image_index, (lr_path, hr_path) in enumerate(
            tqdm(validation_pairs, desc=f"Full-image testing x{args.scale}"),
            start=1,
        ):
            input_image = read_grayscale(lr_path)
            target = read_grayscale(hr_path)
            input_image = resize_to_match(input_image, target.shape)

            prediction = tiled_inference(
                model=model,
                image=input_image,
                device=device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                tile_batch_size=args.tile_batch_size,
            )

            input_rmse, input_psnr, input_ssim = metric_triplet(target, input_image)
            output_rmse, output_psnr, output_ssim = metric_triplet(target, prediction)

            name = hr_path.stem
            filename = f"{name}.png"

            save_grayscale(input_image, input_dir / filename)
            save_grayscale(prediction, prediction_dir / filename)
            save_grayscale(target, target_dir / filename)

            if args.save_limit == 0 or image_index <= args.save_limit:
                create_comparison_image(
                    input_image=input_image,
                    prediction=prediction,
                    target=target,
                    input_psnr=input_psnr,
                    input_ssim=input_ssim,
                    output_psnr=output_psnr,
                    output_ssim=output_ssim,
                    output_path=comparison_dir / f"{name}_comparison.png",
                )

            metric_rows.append(
                {
                    "index": image_index,
                    "name": name,
                    "height": target.shape[0],
                    "width": target.shape[1],
                    "input_rmse": input_rmse,
                    "input_psnr": input_psnr,
                    "input_ssim": input_ssim,
                    "output_rmse": output_rmse,
                    "output_psnr": output_psnr,
                    "output_ssim": output_ssim,
                    "psnr_gain": output_psnr - input_psnr,
                    "ssim_gain": output_ssim - input_ssim,
                }
            )

    if not metric_rows:
        raise RuntimeError("No validation images were tested.")

    summary_values = {
        "mean_input_rmse": mean_metric(metric_rows, "input_rmse"),
        "mean_input_psnr": mean_metric(metric_rows, "input_psnr"),
        "mean_input_ssim": mean_metric(metric_rows, "input_ssim"),
        "mean_output_rmse": mean_metric(metric_rows, "output_rmse"),
        "mean_output_psnr": mean_metric(metric_rows, "output_psnr"),
        "mean_output_ssim": mean_metric(metric_rows, "output_ssim"),
        "mean_psnr_gain": mean_metric(metric_rows, "psnr_gain"),
        "mean_ssim_gain": mean_metric(metric_rows, "ssim_gain"),
    }

    metrics_path = result_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_path = result_dir / "summary.txt"
    summary_lines = [
        f"scale: x{args.scale}",
        f"checkpoint: {checkpoint_path}",
        f"number of images: {len(metric_rows)}",
        f"evaluation: full-image tiled inference",
        f"tile size: {args.tile_size}",
        f"overlap: {args.overlap}",
        f"tile batch size: {args.tile_batch_size}",
        f"mean input RMSE: {summary_values['mean_input_rmse']:.6f}",
        f"mean input PSNR: {summary_values['mean_input_psnr']:.6f}",
        f"mean input SSIM: {summary_values['mean_input_ssim']:.6f}",
        f"mean output RMSE: {summary_values['mean_output_rmse']:.6f}",
        f"mean output PSNR: {summary_values['mean_output_psnr']:.6f}",
        f"mean output SSIM: {summary_values['mean_output_ssim']:.6f}",
        f"mean PSNR gain: {summary_values['mean_psnr_gain']:.6f}",
        f"mean SSIM gain: {summary_values['mean_ssim_gain']:.6f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

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
    print(
        "Mean gain: "
        f"PSNR={summary_values['mean_psnr_gain']:.6f}, "
        f"SSIM={summary_values['mean_ssim_gain']:.6f}"
    )
    print(f"Results: {result_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()