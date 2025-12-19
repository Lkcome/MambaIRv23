import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 输入输出路径
input_folder = 'datasets/yemai'
output_folder = 'datasets/yemai_map'
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .npy 文件
npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

for file_name in tqdm(npy_files, desc='处理文件'):
    file_path = os.path.join(input_folder, file_name)
    data = np.load(file_path)

    # 检查数据维度
    if data.ndim != 3:
        raise ValueError(f"{file_name} 的维度不是 3D，实际为 {data.shape}")

    # 执行最大振幅投影（MAP）
    projection = np.max(np.abs(data), axis=2)  # 对深度方向求绝对值最大值

    # 归一化到 [0, 255]
    projection_norm = ((projection - projection.min()) / (projection.max() - projection.min()) * 255).astype(np.uint8)

    # 保存 PNG
    image = Image.fromarray(projection_norm)
    base_name = os.path.splitext(file_name)[0]
    image.save(os.path.join(output_folder, f"{base_name}_map.png"))
