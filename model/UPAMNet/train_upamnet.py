#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train UPAMNet on the Duke PAM dataset for x2, x4, and x8 restoration.

This script keeps the main settings of the official UPAMNet implementation:
- residual learning: target residual = HR - input
- percept_patch_prior loss
- AdamW optimizer
- initial learning rate = 1e-3
- weight decay = 1e-4
- 20 epochs
- StepLR updated per iteration
- best checkpoint selected by minimum validation RMSE

Duke PAM adaptation:
- batch size defaults to 4
- LR images are bicubic-resized only when LR and HR sizes differ
- prior masks are generated from HR images:
    0 = background
    1 = vessel/object
    2 = edge
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loss import percept_patch_prior
from model import UPAMNet


DEFAULT_DATA_ROOT = Path(
    "/data1/like/MambaIRv21/datasets/Duke_PAM_datasets_xyraw"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UPAMNet on Duke PAM."
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
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--repeats-per-image",
        type=int,
        default=10,
        help=(
            "Number of augmented appearances of each 128x128 image "
            "per epoch."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments"),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a numbered checkpoint every N epochs; 0 disables it.",
    )

    # Official model defaults from main.py.
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

    # Official percept_patch_prior loss weights.
    parser.add_argument(
        "--w-per",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--w-back-per",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--w-object-per",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--w-edge-per",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    images = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images found in: {folder}")

    return images


def canonical_name(path: Path, scale: int) -> str:
    stem = path.stem

    suffixes = {
        2: ("_pam22", "_x2", "_s2"),
        4: ("_pam44", "_x4", "_s4"),
        8: ("_pam88", "_x8", "_s8"),
    }[scale]

    lower_stem = stem.lower()

    for suffix in suffixes:
        if lower_stem.endswith(suffix):
            return stem[: -len(suffix)].lower()

    return lower_stem


def resolve_split_directories(
    data_root: Path,
    split: str,
    scale: int,
) -> Tuple[Path, Path]:
    candidates = [
        (
            data_root / split / f"LR_S{scale}",
            data_root / split / f"HR_x{scale}",
        ),
        (
            data_root / split / f"LR_x{scale}",
            data_root / split / f"HR_x{scale}",
        ),
        (
            data_root / split / "LR" / f"X{scale}",
            data_root / split / "HR",
        ),
    ]

    for lr_dir, hr_dir in candidates:
        if lr_dir.exists() and hr_dir.exists():
            return lr_dir, hr_dir

    attempted = "\n".join(
        f"LR: {lr_dir}\nHR: {hr_dir}"
        for lr_dir, hr_dir in candidates
    )

    raise FileNotFoundError(
        f"Could not find {split} x{scale} LR/HR folders.\n{attempted}"
    )


def build_pairs(
    data_root: Path,
    split: str,
    scale: int,
) -> List[Tuple[Path, Path]]:
    lr_dir, hr_dir = resolve_split_directories(
        data_root=data_root,
        split=split,
        scale=scale,
    )

    lr_images = list_images(lr_dir)
    hr_images = list_images(hr_dir)

    hr_map: Dict[str, Path] = {}

    for hr_path in hr_images:
        name = canonical_name(hr_path, scale)

        if name in hr_map:
            raise RuntimeError(
                f"Duplicate HR pair key '{name}': "
                f"{hr_map[name]} and {hr_path}"
            )

        hr_map[name] = hr_path

    pairs: List[Tuple[Path, Path]] = []
    missing_hr: List[Path] = []

    for lr_path in lr_images:
        name = canonical_name(lr_path, scale)
        hr_path = hr_map.get(name)

        if hr_path is None:
            missing_hr.append(lr_path)
        else:
            pairs.append((lr_path, hr_path))

    if missing_hr:
        examples = "\n".join(
            str(path)
            for path in missing_hr[:10]
        )

        raise RuntimeError(
            f"{len(missing_hr)} LR images do not have matching HR images. "
            f"Examples:\n{examples}"
        )

    print(
        f"{split} x{scale}: "
        f"LR={len(lr_images)}, "
        f"HR={len(hr_images)}, "
        f"pairs={len(pairs)}"
    )

    return pairs


def read_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        array = np.asarray(
            image,
            dtype=np.float32,
        )

    return array / 255.0


def resize_to_match(
    image: np.ndarray,
    target_shape: Sequence[int],
) -> np.ndarray:
    target_height = int(target_shape[0])
    target_width = int(target_shape[1])

    if image.shape == (target_height, target_width):
        return image

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )

def paired_random_crop(
    lr: np.ndarray,
    hr: np.ndarray,
    patch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = hr.shape

    if height < patch_size or width < patch_size:
        target_height = max(height, patch_size)
        target_width = max(width, patch_size)

        lr = cv2.resize(
            lr,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

        hr = cv2.resize(
            hr,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

        height, width = hr.shape

    top = random.randint(
        0,
        height - patch_size,
    )

    left = random.randint(
        0,
        width - patch_size,
    )

    lr_patch = lr[
        top : top + patch_size,
        left : left + patch_size,
    ]

    hr_patch = hr[
        top : top + patch_size,
        left : left + patch_size,
    ]

    return (
        np.ascontiguousarray(lr_patch),
        np.ascontiguousarray(hr_patch),
    )

def apply_synchronized_augmentation(
    lr: np.ndarray,
    hr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        lr = np.fliplr(lr)
        hr = np.fliplr(hr)

    if random.random() < 0.5:
        lr = np.flipud(lr)
        hr = np.flipud(hr)

    rotation_count = random.randint(0, 3)

    if rotation_count:
        lr = np.rot90(lr, rotation_count)
        hr = np.rot90(hr, rotation_count)

    return (
        np.ascontiguousarray(lr),
        np.ascontiguousarray(hr),
    )


def generate_prior_mask(hr: np.ndarray) -> np.ndarray:
    """
    Generate the three-class mask required by percept_patch_prior.

    Class definitions used by the official loss:
        0 = background
        1 = object / vessel
        2 = edge

    The official repository does not provide its PA_dataset or original
    mask-generation code. This is the Duke PAM adaptation.
    """
    hr_uint8 = np.clip(
        hr * 255.0,
        0,
        255,
    ).astype(np.uint8)

    _, foreground = cv2.threshold(
        hr_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )

    edges = cv2.Canny(
        hr_uint8,
        threshold1=30,
        threshold2=90,
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    mask = np.zeros_like(
        hr_uint8,
        dtype=np.float32,
    )

    mask[foreground > 0] = 1.0
    mask[edges > 0] = 2.0

    return mask


class DukePAMDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        training: bool,
        repeats_per_image: int = 1,
    ) -> None:
        self.pairs = list(pairs)
        self.training = training

        if training:
            self.repeats_per_image = repeats_per_image
        else:
            self.repeats_per_image = 1

    def __len__(self) -> int:
        return len(self.pairs) * self.repeats_per_image

    def __getitem__(self, index: int) -> Dict[str, object]:
        lr_path, hr_path = self.pairs[
            index % len(self.pairs)
        ]

        lr = read_grayscale(lr_path)
        hr = read_grayscale(hr_path)

        lr = resize_to_match(
            image=lr,
            target_shape=hr.shape,
        )

        if self.training:
            lr, hr = paired_random_crop(
                lr=lr,
                hr=hr,
                patch_size=128,
            )

            lr, hr = apply_synchronized_augmentation(
                lr=lr,
                hr=hr,
            )
        else:
            lr, hr = center_crop_or_resize(
                lr=lr,
                hr=hr,
                patch_size=128,
            )

        prior_mask = generate_prior_mask(hr)

        return {
            "input": torch.from_numpy(
                lr[None, ...]
            ).float(),
            "gt": torch.from_numpy(
                hr[None, ...]
            ).float(),
            "mask": torch.from_numpy(
                prior_mask[None, ...]
            ).float(),
            "name": hr_path.stem,
        }


def center_crop_or_resize(
    lr: np.ndarray,
    hr: np.ndarray,
    patch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = hr.shape

    if height < patch_size or width < patch_size:
        lr = cv2.resize(
            lr,
            (patch_size, patch_size),
            interpolation=cv2.INTER_CUBIC,
        )

        hr = cv2.resize(
            hr,
            (patch_size, patch_size),
            interpolation=cv2.INTER_CUBIC,
        )

        return (
            np.ascontiguousarray(lr),
            np.ascontiguousarray(hr),
        )

    top = (height - patch_size) // 2
    left = (width - patch_size) // 2

    lr_patch = lr[
        top : top + patch_size,
        left : left + patch_size,
    ]

    hr_patch = hr[
        top : top + patch_size,
        left : left + patch_size,
    ]

    return (
        np.ascontiguousarray(lr_patch),
        np.ascontiguousarray(hr_patch),
    )

def create_model(args: argparse.Namespace) -> UPAMNet:
    return UPAMNet(
        in_channel=1,
        out_channel=1,
        inner_channel=args.inner_channel,
        norm_groups=args.norm_groups,
        channel_mults=(1, 2, 4, 8),
        attn_res=[16, 32, 64],
        res_blocks=args.res_blocks,
        dropout=args.dropout,
        image_size=128,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    rmse_values: List[float] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    for batch in tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    ):
        input_image = batch["input"].to(
            device,
            non_blocking=True,
        )

        target = batch["gt"].to(
            device,
            non_blocking=True,
        )

        prediction = torch.clamp(
            input_image + model(input_image),
            min=0.0,
            max=1.0,
        )

        prediction_numpy = prediction[:, 0].cpu().numpy()
        target_numpy = target[:, 0].cpu().numpy()

        for predicted_image, target_image in zip(
            prediction_numpy,
            target_numpy,
        ):
            rmse = float(
                np.sqrt(
                    np.mean(
                        (predicted_image - target_image) ** 2
                    )
                )
            )

            psnr = float(
                peak_signal_noise_ratio(
                    target_image,
                    predicted_image,
                    data_range=1.0,
                )
            )

            ssim = float(
                structural_similarity(
                    target_image,
                    predicted_image,
                    data_range=1.0,
                )
            )

            rmse_values.append(rmse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

    return {
        "rmse": float(np.mean(rmse_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    epoch: int,
    best_rmse: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "best_rmse": best_rmse,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in vars(args).items()
            },
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device(f"cuda:{args.gpu}")

    experiment_dir = (
        args.output_root
        / f"UPAMNet_DukePAM_x{args.scale}"
    )

    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }

    with open(
        experiment_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config,
            config_file,
            indent=2,
            ensure_ascii=False,
        )

    train_pairs = build_pairs(
        data_root=args.data_root,
        split="train",
        scale=args.scale,
    )

    validation_pairs = build_pairs(
        data_root=args.data_root,
        split="valid",
        scale=args.scale,
    )

    train_dataset = DukePAMDataset(
        pairs=train_pairs,
        training=True,
        repeats_per_image=args.repeats_per_image,
    )

    validation_dataset = DukePAMDataset(
        pairs=validation_pairs,
        training=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = create_model(args).to(device)

    loss_function = percept_patch_prior(
        args=args
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler_step_size = max(
        1,
        int(
            args.epochs
            * 0.4
            * len(train_dataloader)
        ),
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=0.1,
    )

    start_epoch = 0
    best_rmse = math.inf

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: "
                f"{args.resume}"
            )

        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(
            checkpoint.get("epoch", -1)
        ) + 1

        best_rmse = float(
            checkpoint.get(
                "best_rmse",
                math.inf,
            )
        )

        print(
            f"Resumed from {args.resume}; "
            f"next epoch={start_epoch + 1}"
        )

    metrics_path = log_dir / "metrics.csv"
    create_csv_header = not metrics_path.exists()

    metrics_file = open(
        metrics_path,
        "a",
        newline="",
        encoding="utf-8",
    )

    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "mse_loss",
            "background_perceptual_loss",
            "object_perceptual_loss",
            "edge_perceptual_loss",
            "validation_rmse",
            "validation_psnr",
            "validation_ssim",
            "learning_rate",
        ],
    )

    if create_csv_header:
        metrics_writer.writeheader()

    print(f"Device: {device}")
    print(
        f"Train samples per epoch: "
        f"{len(train_dataset)}"
    )
    print(
        f"Train iterations per epoch: "
        f"{len(train_dataloader)}"
    )
    print(
        f"Scheduler step size: "
        f"{scheduler_step_size} iterations"
    )
    print(
        f"Model parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    try:
        for epoch in range(
            start_epoch,
            args.epochs,
        ):
            model.train()

            accumulated = {
                "total": 0.0,
                "mse": 0.0,
                "background": 0.0,
                "object": 0.0,
                "edge": 0.0,
            }

            processed_samples = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=(
                    f"Epoch "
                    f"{epoch + 1}/{args.epochs}"
                ),
            )

            for batch in progress_bar:
                input_image = batch["input"].to(
                    device,
                    non_blocking=True,
                )

                target = batch["gt"].to(
                    device,
                    non_blocking=True,
                )

                prior_mask = batch["mask"].to(
                    device,
                    non_blocking=True,
                )

                predicted_residual = model(
                    input_image
                )

                target_residual = (
                    target - input_image
                )

                loss_terms = loss_function(
                    pred=predicted_residual,
                    gt=target_residual,
                    epoch=epoch,
                    mask=prior_mask,
                )

                total_loss = (
                    loss_terms["mse_loss"]
                    + args.w_per
                    * (
                        args.w_back_per
                        * loss_terms[
                            "per_back_loss"
                        ]
                        + args.w_object_per
                        * loss_terms[
                            "per_object_loss"
                        ]
                        + args.w_edge_per
                        * loss_terms[
                            "per_edge_loss"
                        ]
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                batch_size = input_image.shape[0]
                processed_samples += batch_size

                accumulated["total"] += (
                    float(total_loss.item())
                    * batch_size
                )

                accumulated["mse"] += (
                    float(
                        loss_terms[
                            "mse_loss"
                        ].item()
                    )
                    * batch_size
                )

                accumulated["background"] += (
                    float(
                        loss_terms[
                            "per_back_loss"
                        ].item()
                    )
                    * batch_size
                )

                accumulated["object"] += (
                    float(
                        loss_terms[
                            "per_object_loss"
                        ].item()
                    )
                    * batch_size
                )

                accumulated["edge"] += (
                    float(
                        loss_terms[
                            "per_edge_loss"
                        ].item()
                    )
                    * batch_size
                )

                average_train_loss = (
                    accumulated["total"] / processed_samples
                )

                progress_bar.set_postfix(
                    loss=f"{average_train_loss:.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            validation_metrics = validate(
                model=model,
                dataloader=validation_dataloader,
                device=device,
            )

            average_losses = {
                key: value / processed_samples
                for key, value in accumulated.items()
            }

            learning_rate = optimizer.param_groups[0][
                "lr"
            ]

            print(
                f"Epoch {epoch + 1}: "
                f"loss={average_losses['total']:.6f}, "
                f"RMSE={validation_metrics['rmse']:.6f}, "
                f"PSNR={validation_metrics['psnr']:.4f}, "
                f"SSIM={validation_metrics['ssim']:.4f}, "
                f"lr={learning_rate:.2e}"
            )

            if (
                validation_metrics["rmse"]
                < best_rmse
            ):
                best_rmse = validation_metrics[
                    "rmse"
                ]

                save_checkpoint(
                    checkpoint_path=(
                        checkpoint_dir
                        / "best_perf.tar"
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_rmse=best_rmse,
                    args=args,
                )

                print(
                    "Saved best checkpoint: "
                    f"RMSE={best_rmse:.6f}"
                )

            save_checkpoint(
                checkpoint_path=(
                    checkpoint_dir
                    / "last.tar"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_rmse=best_rmse,
                args=args,
            )

            if (
                args.save_every > 0
                and (epoch + 1)
                % args.save_every
                == 0
            ):
                save_checkpoint(
                    checkpoint_path=(
                        checkpoint_dir
                        / (
                            "checkpoint_"
                            f"{epoch + 1:03d}.tar"
                        )
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_rmse=best_rmse,
                    args=args,
                )

            metrics_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": average_losses[
                        "total"
                    ],
                    "mse_loss": average_losses[
                        "mse"
                    ],
                    "background_perceptual_loss": (
                        average_losses[
                            "background"
                        ]
                    ),
                    "object_perceptual_loss": (
                        average_losses["object"]
                    ),
                    "edge_perceptual_loss": (
                        average_losses["edge"]
                    ),
                    "validation_rmse": (
                        validation_metrics[
                            "rmse"
                        ]
                    ),
                    "validation_psnr": (
                        validation_metrics[
                            "psnr"
                        ]
                    ),
                    "validation_ssim": (
                        validation_metrics[
                            "ssim"
                        ]
                    ),
                    "learning_rate": learning_rate,
                }
            )

            metrics_file.flush()

    finally:
        metrics_file.close()

    print("Training completed.")
    print(
        "Best checkpoint: "
        f"{checkpoint_dir / 'best_perf.tar'}"
    )


if __name__ == "__main__":
    main()