import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from FD_UNet import getModel


DATA_ROOT = Path(
    "/data1/like/MambaIRv21/datasets/Duke_PAM_datasets_xyraw"
)

LR_SUFFIX = {
    2: "_pam22",
    4: "_pam44",
    8: "_pam88",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[2, 4, 8], required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1489)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=119)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPU devices:", gpus)


def image_files(folder):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def normalized_key(path, scale, is_lr):
    stem = path.stem

    if is_lr:
        suffix = LR_SUFFIX[scale]
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


def build_pairs(split, scale):
    lr_dir = DATA_ROOT / split / f"LR_S{scale}"
    hr_dir = DATA_ROOT / split / f"HR_x{scale}"

    lr_files = image_files(lr_dir)
    hr_files = image_files(hr_dir)

    lr_map = {
        normalized_key(path, scale, True): path
        for path in lr_files
    }

    hr_map = {
        normalized_key(path, scale, False): path
        for path in hr_files
    }

    common = sorted(set(lr_map) & set(hr_map))

    if not common:
        raise RuntimeError(
            f"No paired images found:\nLR={lr_dir}\nHR={hr_dir}"
        )

    missing_hr = sorted(set(lr_map) - set(hr_map))
    missing_lr = sorted(set(hr_map) - set(lr_map))

    if missing_hr:
        print(f"Warning: {len(missing_hr)} LR images have no HR pair.")

    if missing_lr:
        print(f"Warning: {len(missing_lr)} HR images have no LR pair.")

    pairs = [(str(lr_map[key]), str(hr_map[key])) for key in common]

    print(
        f"{split} x{scale}: "
        f"LR={len(lr_files)}, HR={len(hr_files)}, pairs={len(pairs)}"
    )

    return pairs


def decode_path(path_tensor):
    return path_tensor.numpy().decode("utf-8")


def load_pair_python(lr_path_tensor, hr_path_tensor):
    lr_path = decode_path(lr_path_tensor)
    hr_path = decode_path(hr_path_tensor)

    with Image.open(lr_path) as image:
        lr = image.convert("L")

    with Image.open(hr_path) as image:
        hr = image.convert("L")

    hr_width, hr_height = hr.size

    # FD-Unet 输入输出尺寸一致，所以先把低分辨率图像 bicubic 上采样到 HR。
    if lr.size != hr.size:
        lr = lr.resize(
            (hr_width, hr_height),
            resample=Image.Resampling.BICUBIC,
        )

    lr = np.asarray(lr, dtype=np.float32) / 255.0
    hr = np.asarray(hr, dtype=np.float32) / 255.0

    lr = np.expand_dims(lr, axis=-1)
    hr = np.expand_dims(hr, axis=-1)

    return lr, hr


def load_pair(lr_path, hr_path):
    lr, hr = tf.py_function(
        func=load_pair_python,
        inp=[lr_path, hr_path],
        Tout=[tf.float32, tf.float32],
    )

    # 原始 PAM 图像尺寸不固定，这里只固定通道数。
    lr.set_shape([None, None, 1])
    hr.set_shape([None, None, 1])

    return lr, hr


PATCH_SIZE = 128


def random_aligned_crop(lr, hr):
    """从 LR/HR 的相同位置随机裁剪 128×128 patch。"""
    pair = tf.concat([lr, hr], axis=-1)

    shape = tf.shape(pair)
    height = shape[0]
    width = shape[1]

    tf.debugging.assert_greater_equal(
        height,
        PATCH_SIZE,
        message="Image height is smaller than patch size.",
    )
    tf.debugging.assert_greater_equal(
        width,
        PATCH_SIZE,
        message="Image width is smaller than patch size.",
    )

    pair = tf.image.random_crop(
        pair,
        size=[PATCH_SIZE, PATCH_SIZE, 2],
    )

    lr_patch = pair[..., :1]
    hr_patch = pair[..., 1:]

    lr_patch.set_shape([PATCH_SIZE, PATCH_SIZE, 1])
    hr_patch.set_shape([PATCH_SIZE, PATCH_SIZE, 1])

    return lr_patch, hr_patch





def augment(lr, hr):
    pair = tf.concat([lr, hr], axis=-1)

    pair = tf.image.random_flip_left_right(pair)
    pair = tf.image.random_flip_up_down(pair)

    rotations = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    pair = tf.image.rot90(pair, rotations)

    return pair[..., :1], pair[..., 1:]


def create_train_dataset(pairs, batch_size, workers):
    lr_paths = [pair[0] for pair in pairs]
    hr_paths = [pair[1] for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))

    dataset = dataset.shuffle(
        buffer_size=min(len(pairs), 4096),
        reshuffle_each_iteration=True,
    )

    dataset = dataset.map(
        load_pair,
        num_parallel_calls=workers,
    )

    dataset = dataset.map(
        random_aligned_crop,
        num_parallel_calls=workers,
    )

    dataset = dataset.map(
        augment,
        num_parallel_calls=workers,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=True,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def psnr_metric(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def make_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def tiled_predict(model, image, tile_size=128, overlap=32):
    """
    对完整图像滑窗推理并重新拼接。
    滑窗只解决显存问题，最终评价区域仍然是完整图像。
    """
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap 必须满足 0 <= overlap < tile_size")

    original_height, original_width = image.shape[:2]

    pad_height = max(tile_size - original_height, 0)
    pad_width = max(tile_size - original_width, 0)

    if pad_height > 0 or pad_width > 0:
        image = np.pad(
            image,
            (
                (0, pad_height),
                (0, pad_width),
                (0, 0),
            ),
            mode="reflect",
        )

    height, width = image.shape[:2]
    stride = tile_size - overlap

    y_positions = make_positions(height, tile_size, stride)
    x_positions = make_positions(width, tile_size, stride)

    prediction_sum = np.zeros_like(image, dtype=np.float32)
    prediction_count = np.zeros_like(image, dtype=np.float32)

    for y in y_positions:
        for x in x_positions:
            tile = image[
                y:y + tile_size,
                x:x + tile_size,
                :
            ]

            prediction = model(
                tile[None, ...],
                training=False,
            ).numpy()[0]

            prediction = np.clip(prediction, 0.0, 1.0)

            prediction_sum[
                y:y + tile_size,
                x:x + tile_size,
                :
            ] += prediction

            prediction_count[
                y:y + tile_size,
                x:x + tile_size,
                :
            ] += 1.0

    prediction = prediction_sum / np.maximum(
        prediction_count,
        1e-8,
    )

    return prediction[
        :original_height,
        :original_width,
        :
    ]


def crop_metric_border(image, border):
    if border <= 0:
        return image

    return image[
        border:-border,
        border:-border,
        :
    ]


def calculate_full_psnr(gt, prediction, border):
    gt = crop_metric_border(gt, border)
    prediction = crop_metric_border(prediction, border)

    mse = np.mean(
        (
            gt.astype(np.float64)
            - prediction.astype(np.float64)
        ) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)


def calculate_full_ssim(gt, prediction, border):
    gt = crop_metric_border(gt, border)
    prediction = crop_metric_border(prediction, border)

    gt = tf.convert_to_tensor(
        gt[None, ...],
        dtype=tf.float32,
    )

    prediction = tf.convert_to_tensor(
        prediction[None, ...],
        dtype=tf.float32,
    )

    return float(
        tf.image.ssim(
            gt,
            prediction,
            max_val=1.0,
        ).numpy()[0]
    )

def ssim_metric(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


class FullImageValidationCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        valid_pairs,
        scale,
        checkpoint_dir,
        tile_size=128,
        overlap=32,
        val_interval=119,
    ):
        super().__init__()

        self.valid_pairs = valid_pairs
        self.scale = scale
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.val_interval = val_interval
        self.best_psnr = -float("inf")

    def read_gray_image(self, path):
        with Image.open(path) as image:
            image = image.convert("L")
            array = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

        return array[..., None]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        current_epoch = epoch + 1
        total_epochs = int(self.params.get("epochs", current_epoch))

        # 保留字段，确保 CSVLogger 从第 1 个 epoch 起就有验证列。
        logs["val_psnr_metric"] = np.nan
        logs["val_ssim_metric"] = np.nan

        should_validate = (
            current_epoch % self.val_interval == 0
            or current_epoch == total_epochs
        )

        if not should_validate:
            print(
                f"Epoch {current_epoch}: 跳过整图验证；"
                f"下次验证间隔为每 {self.val_interval} epochs。"
            )
            return

        psnr_values = []
        ssim_values = []

        print("\n开始整图验证……")

        for index, (lr_path, hr_path) in enumerate(
            self.valid_pairs,
            start=1,
        ):
            lr = self.read_gray_image(lr_path)
            hr = self.read_gray_image(hr_path)

            hr_height, hr_width = hr.shape[:2]

            # FD-Unet 输入输出同尺寸，因此先将 LR 放大到 HR 尺寸。
            if lr.shape[:2] != hr.shape[:2]:
                lr_image = Image.fromarray(
                    np.clip(
                        lr[..., 0] * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                )

                lr_image = lr_image.resize(
                    (hr_width, hr_height),
                    resample=Image.Resampling.BICUBIC,
                )

                lr = (
                    np.asarray(
                        lr_image,
                        dtype=np.float32,
                    )[..., None]
                    / 255.0
                )

            prediction = tiled_predict(
                self.model,
                lr,
                tile_size=self.tile_size,
                overlap=self.overlap,
            )

            psnr = calculate_full_psnr(
                hr,
                prediction,
                border=self.scale,
            )

            ssim = calculate_full_ssim(
                hr,
                prediction,
                border=self.scale,
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            print(
                f"[{index:02d}/{len(self.valid_pairs):02d}] "
                f"{Path(hr_path).name}: "
                f"PSNR={psnr:.4f}, "
                f"SSIM={ssim:.4f}"
            )

        mean_psnr = float(np.mean(psnr_values))
        mean_ssim = float(np.mean(ssim_values))

        logs["val_psnr_metric"] = mean_psnr
        logs["val_ssim_metric"] = mean_ssim

        latest_path = (
            self.checkpoint_dir
            / "latest_weights.h5"
        )
        self.model.save_weights(str(latest_path))

        if mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr

            best_path = (
                self.checkpoint_dir
                / "best_weights.h5"
            )
            self.model.save_weights(str(best_path))

            print(
                f"整图 PSNR 提升至 {mean_psnr:.4f}，"
                f"已保存最佳模型：{best_path}"
            )

        print(
            f"Epoch {epoch + 1} 整图验证："
            f"PSNR={mean_psnr:.4f}, "
            f"SSIM={mean_ssim:.4f}"
        )


def main():
    args = parse_args()

    set_seed(args.seed)
    configure_gpu()

    train_pairs = build_pairs("train", args.scale)
    valid_pairs = build_pairs("valid", args.scale)

    train_dataset = create_train_dataset(
        train_pairs,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    experiment_name = f"FDUNet_DukePAM_x{args.scale}"
    checkpoint_dir = Path("checkpoints") / experiment_name
    log_dir = Path("logs") / experiment_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = getModel(
        input_shape=(128, 128, 1),
        filters=args.filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[psnr_metric, ssim_metric],
    )

    model.summary()

    full_image_validation = FullImageValidationCallback(
        valid_pairs=valid_pairs,
        scale=args.scale,
        checkpoint_dir=checkpoint_dir,
        tile_size=128,
        overlap=32,
        val_interval=args.val_interval,
    )

    # 按 MambaIRv23 的 250k iteration 训练设置近似换算：
    # batch_size=2 时每个 epoch 为 floor(337/2)=168 iterations。
    # 125k、200k、225k、237.5k iterations 对应约
    # 745、1191、1340、1414 epochs。
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: (
            args.learning_rate
            * (0.5 ** sum(
                epoch + 1 >= milestone
                for milestone in [745, 1191, 1340, 1414]
            ))
        ),
        verbose=1,
    )

    callbacks = [
        # 该回调负责周期性整图验证并保存 best_weights.h5。
        full_image_validation,

        # 每个 epoch 保存一次最新权重，便于训练中断后恢复。
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "latest_weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),

        tf.keras.callbacks.CSVLogger(
            str(log_dir / "training_log.csv"),
            append=True,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir / "tensorboard")
        ),

        lr_scheduler,
    ]

    config = vars(args).copy()
    config["data_root"] = str(DATA_ROOT)
    config["train_pairs"] = len(train_pairs)
    config["valid_pairs"] = len(valid_pairs)
    config["validation_mode"] = "full_image_tiled_inference"
    config["tile_size"] = 128
    config["tile_overlap"] = 32
    config["crop_border"] = args.scale
    config["val_interval"] = args.val_interval
    config["target_iterations"] = 250000
    config["approx_iterations"] = args.epochs * (len(train_pairs) // args.batch_size)

    with open(
        checkpoint_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Training completed.")
    print(
        "Best weights:",
        checkpoint_dir / "best_weights.h5",
    )


if __name__ == "__main__":
    main()