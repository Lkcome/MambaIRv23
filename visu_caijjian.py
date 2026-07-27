from pathlib import Path
import cv2
import numpy as np


# ==================================================
# 1. 路径
# ==================================================
#input_path = Path("/data1/like/MambaIRv23/model/EDSR-PyTorch/experiment/RCAN_Duke_x4_test/results-DukePAM/reslt_OR_2 (22)_index0_x4_SR.png")
#input_path = Path("/data1/like/MambaIRv23/model/HAT/results/HAT_SRx4/visualization/DuKe photoacoustic microscopy datasets/reslt_OR_2 (22)_index0_pam44_HAT_SRx4.png")
# input_path = Path(
#     "/data1/like/MambaIRv21/datasets/"
#     "Duke_PAM_datasets_xyraw/valid/HR_x4/"
#     "reslt_OR_2 (22)_index0.png"
# )

input_path = Path("/data1/like/MambaIRv23/datasets/reslt_OR_2 (22)_index0_pam44_test_MaIR_DukePAM_x4.png")



output_dir = Path("/data1/like/MambaIRv23/datasets/visual/MaIR")
# output_dir = Path("/data1/like/MambaIRv23/datasets/visual")
output_dir.mkdir(parents=True, exist_ok=True)

marked_path = output_dir / f"{input_path.stem}_marked.png"
crop_path = output_dir / f"{input_path.stem}_crop.png"


# ==================================================
# 2. 红框和裁剪区域
# x = 420:820
# y = 180:580
# ==================================================
left = 520
top = 1020
right = 720
bottom = 1220

line_width = 6


# ==================================================
# 3. 按原始位深读取
# IMREAD_UNCHANGED：不转8位、不归一化、不改变通道
# ==================================================
image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

if image is None:
    raise FileNotFoundError(f"无法读取图像：{input_path}")

height, width = image.shape[:2]

print(f"图像尺寸：{width} × {height}")
print(f"数据类型：{image.dtype}")
print(f"图像形状：{image.shape}")
print(f"像素最小值：{image.min()}")
print(f"像素最大值：{image.max()}")


# ==================================================
# 4. 检查坐标
# ==================================================
if not (0 <= left < right <= width):
    raise ValueError(
        f"x 坐标错误，图像宽度为 {width}，"
        f"当前 left={left}, right={right}"
    )

if not (0 <= top < bottom <= height):
    raise ValueError(
        f"y 坐标错误，图像高度为 {height}，"
        f"当前 top={top}, bottom={bottom}"
    )


# ==================================================
# 5. 直接裁剪原始像素
# 不缩放、不插值、不归一化
# copy() 避免后续绘制影响裁剪区域
# ==================================================
crop_image = image[top:bottom, left:right].copy()

success = cv2.imwrite(
    str(crop_path),
    crop_image,
    [cv2.IMWRITE_PNG_COMPRESSION, 0]
)

if not success:
    raise RuntimeError(f"局部图保存失败：{crop_path}")


# ==================================================
# 6. 制作带红框的完整图
# 仅 marked_image 是三通道副本
# 原图和 crop_image 均不改变
# ==================================================
if image.ndim == 2:
    # 灰度图复制为三通道，位深和像素值保持不变
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

elif image.ndim == 3 and image.shape[2] == 3:
    marked_image = image.copy()

elif image.ndim == 3 and image.shape[2] == 4:
    marked_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

else:
    raise ValueError(f"不支持的图像格式：shape={image.shape}")


# 根据原图位深设置红色最大值
if marked_image.dtype == np.uint16:
    red_color = (0, 0, 65535)  # OpenCV采用BGR顺序
elif marked_image.dtype == np.uint8:
    red_color = (0, 0, 255)
else:
    raise TypeError(
        f"当前图像数据类型为 {marked_image.dtype}，"
        "代码仅处理 uint8 或 uint16 PNG"
    )


cv2.rectangle(
    marked_image,
    (left, top),
    (right - 1, bottom - 1),
    red_color,
    thickness=line_width
)

success = cv2.imwrite(
    str(marked_path),
    marked_image,
    [cv2.IMWRITE_PNG_COMPRESSION, 0]
)

if not success:
    raise RuntimeError(f"红框图保存失败：{marked_path}")


print("\n处理完成：")
print(f"原图未修改：{input_path}")
print(f"带红框完整图：{marked_path}")
print(f"原始像素局部图：{crop_path}")
print(f"局部图尺寸：{right - left} × {bottom - top}")