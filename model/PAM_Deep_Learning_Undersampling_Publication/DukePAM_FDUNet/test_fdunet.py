import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from FD_UNet import getModel
from train_fdunet import (
    DATA_ROOT,
    build_pairs,
    calculate_full_psnr,
    calculate_full_ssim,
    tiled_predict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[2, 4, 8], required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="默认使用 checkpoints/FDUNet_DukePAM_x{scale}/best_weights.h5",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--save-limit", type=int, default=0,
                        help="0 表示保存全部可视化结果")
    return parser.parse_args()


def read_gray(path):
    with Image.open(path) as image:
        image = image.convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[..., None]


def resize_lr_to_hr(lr, hr_shape):
    hr_height, hr_width = hr_shape[:2]

    if lr.shape[:2] == (hr_height, hr_width):
        return lr

    image = Image.fromarray(
        np.clip(lr[..., 0] * 255.0, 0, 255).astype(np.uint8)
    )
    image = image.resize(
        (hr_width, hr_height),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(image, dtype=np.float32)[..., None] / 255.0


def to_uint8(image):
    return np.clip(image[..., 0] * 255.0, 0, 255).astype(np.uint8)


def save_gray(image, path):
    Image.fromarray(to_uint8(image), mode="L").save(path)


def make_comparison(lr, pred, hr, psnr, ssim, output_path):
    lr_u8 = to_uint8(lr)
    pred_u8 = to_uint8(pred)
    hr_u8 = to_uint8(hr)

    height, width = hr_u8.shape
    label_height = 48

    canvas = Image.new(
        "L",
        (width * 3, height + label_height),
        color=0,
    )

    canvas.paste(Image.fromarray(lr_u8), (0, label_height))
    canvas.paste(Image.fromarray(pred_u8), (width, label_height))
    canvas.paste(Image.fromarray(hr_u8), (width * 2, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Bicubic LR", fill=255)
    draw.text(
        (width + 10, 10),
        f"FD-Unet  PSNR={psnr:.4f}  SSIM={ssim:.4f}",
        fill=255,
    )
    draw.text((width * 2 + 10, 10), "HR / Ground Truth", fill=255)

    canvas.save(output_path)


def main():
    args = parse_args()

    weights_path = (
        Path(args.weights)
        if args.weights
        else Path("checkpoints")
        / f"FDUNet_DukePAM_x{args.scale}"
        / "best_weights.h5"
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_root = (
        Path("results")
        / f"FDUNet_DukePAM_x{args.scale}"
        / args.split
    )
    pred_dir = output_root / "pred"
    lr_dir = output_root / "bicubic_lr"
    hr_dir = output_root / "hr"
    comparison_dir = output_root / "comparison"

    for folder in [pred_dir, lr_dir, hr_dir, comparison_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    model = getModel(
        input_shape=(128, 128, 1),
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal",
    )
    model.load_weights(str(weights_path))

    pairs = build_pairs(args.split, args.scale)

    rows = []
    psnr_values = []
    ssim_values = []

    for index, (lr_path, hr_path) in enumerate(pairs, start=1):
        lr = read_gray(lr_path)
        hr = read_gray(hr_path)
        lr_up = resize_lr_to_hr(lr, hr.shape)

        pred = tiled_predict(
            model,
            lr_up,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        pred = np.clip(pred, 0.0, 1.0)

        psnr = calculate_full_psnr(
            hr,
            pred,
            border=args.scale,
        )
        ssim = calculate_full_ssim(
            hr,
            pred,
            border=args.scale,
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        name = Path(hr_path).stem

        save_gray(pred, pred_dir / f"{name}.png")
        save_gray(lr_up, lr_dir / f"{name}.png")
        save_gray(hr, hr_dir / f"{name}.png")

        if args.save_limit == 0 or index <= args.save_limit:
            make_comparison(
                lr_up,
                pred,
                hr,
                psnr,
                ssim,
                comparison_dir / f"{name}_comparison.png",
            )

        rows.append(
            {
                "index": index,
                "name": name,
                "lr_path": lr_path,
                "hr_path": hr_path,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

        print(
            f"[{index:02d}/{len(pairs):02d}] "
            f"{name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}"
        )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    csv_path = output_root / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "name",
                "lr_path",
                "hr_path",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_root / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"scale: x{args.scale}",
                f"split: {args.split}",
                f"weights: {weights_path}",
                f"number of images: {len(rows)}",
                f"mean PSNR: {mean_psnr:.6f}",
                f"mean SSIM: {mean_ssim:.6f}",
                f"crop border: {args.scale}",
                f"tile size: {args.tile_size}",
                f"overlap: {args.overlap}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nTesting completed.")
    print(f"Mean PSNR: {mean_psnr:.6f}")
    print(f"Mean SSIM: {mean_ssim:.6f}")
    print(f"Results: {output_root}")
    print(f"Metrics: {csv_path}")
    print(f"Visual comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()