# basicsr/archs/scan_strategies_advanced.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 基础工具
# ----------------------------
def _linear_index(y, x, W):
    return y * W + x

def _clip_coords(coords, H, W):
    return [(y, x) for (y, x) in coords if 0 <= y < H and 0 <= x < W]


class GuidanceHead(nn.Module):
    """
    从特征图中同时预测方向场 θ 和退化图 D
    输入:  x: (B,1,H,W) 或 (B,C,H,W)
    输出:  theta: (B,H,W) \in (-pi, pi)
           D    : (B,H,W) \in (0,1)
    """
    def __init__(self, in_ch=1, mid_ch=16):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(mid_ch, 1, 1)
        self.deg_head   = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)         # (B,1,H,W)
        f = self.feat(x)              # (B,mid,H,W)
        theta = torch.tanh(self.theta_head(f)) * math.pi   # (-pi,pi)
        D = torch.sigmoid(self.deg_head(f))                # (0,1)
        return theta.squeeze(1), D.squeeze(1)              # (B,H,W), (B,H,W)


class ScanWeightHead(nn.Module):
    """
    利用 (theta, D) 生成一张 soft 权重图 w(y,x) \in (0,1)
    可用于 CSH 或 RACS
    """
    def __init__(self, in_ch=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, theta, D):
        """
        theta, D: (B,H,W)
        return:  w: (B,H,W)
        """
        # 示例特征: [D, cos(theta)]，也可以换成 sin 等别的组合
        feat = torch.stack([D, torch.cos(theta)], dim=1)   # (B,2,H,W)
        w = torch.sigmoid(self.conv(feat))                 # (B,1,H,W)
        return w.squeeze(1)



# ----------------------------
# 引导信号：方向场 θ 与 退化图 D
# ----------------------------
@torch.no_grad()
def sobel_orientation(feat, H, W):
    """
    feat: (B,1,H,W) 或 (B,C,H,W)；输出 theta ∈ [-pi, pi], shape=(B,H,W)
    实现：通道均值 -> Sobel -> atan2
    """
    if feat.dim() == 4 and feat.size(1) != 1:
        x = feat.mean(dim=1, keepdim=True)   # (B,1,H,W)
    else:
        x = feat if feat.dim() == 4 else feat.view(feat.size(0), 1, H, W)

    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    theta = torch.atan2(gy, gx).squeeze(1)   # (B,H,W)
    return theta

class DegradationHead(nn.Module):
    """
    轻量退化图 D(y,x) 预测头：3×3 Conv -> ReLU -> 1×1 Conv -> Sigmoid
    输入可为 (B,1,H,W) 或 (B,C,H,W)；输出 (B,H,W)
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 1, 1, 1, 0)

    def forward(self, x):
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        h = F.relu(self.conv1(x))
        d = torch.sigmoid(self.conv2(h))     # (B,1,H,W)
        return d.squeeze(1)                  # (B,H,W)

# ----------------------------
# Hilbert 生成（方块内次序）
# ----------------------------
def _hilbert_curve_coords(n):
    """返回 2^k × 2^k 网格的 Hilbert 次序 (y,x) 列表。n 必须是 2 的幂。"""
    assert (n & (n - 1)) == 0 and n > 0, "n must be power-of-two"
    def rot(n_, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n_ - 1 - x
                y = n_ - 1 - y
            x, y = y, x
        return x, y

    coords = []
    for d in range(n * n):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        coords.append((y, x))
    return coords

def _tile_hilbert_indices(H, W, block=8):
    """
    将 Hilbert 次序平铺到 H×W（块大小 block），返回线性索引序列（长度≈H*W）。
    """
    pow2 = 1 << (block - 1).bit_length()
    hilb = _hilbert_curve_coords(pow2)

    order = []
    for by in range(0, H, block):
        for bx in range(0, W, block):
            for (iy, ix) in hilb:
                if pow2 != block:
                    ix = int(ix * (block - 1) / (pow2 - 1))
                    iy = int(iy * (block - 1) / (pow2 - 1))
                y, x = by + iy, bx + ix
                if 0 <= y < H and 0 <= x < W:
                    order.append(_linear_index(y, x, W))
    return order

# ----------------------------
# 1) CSH：曲线条带 + Hilbert 子块 + 坏处加密
# ----------------------------
@torch.no_grad()
def build_csh_indices(H, W, theta, D, stripe=8, kappa=1.0, block_base=8, device="cpu"):
    """
    theta: (B,H,W), D: (B,H,W)；输出 idx (H*W,)
    条带内 S 形，行偏移随 θ 微调；子块采用 Hilbert；坏处区域使用更小子块（加密）。
    """
    theta = theta.mean(dim=0)   # (H,W)
    D = D.mean(dim=0)           # (H,W)

    # 行偏移（沿 θ 的 cos 分量，限制幅度）
    offsets = (torch.cos(theta) * kappa).clamp(-stripe//2, stripe//2)
    offsets = offsets.mean(dim=1)            # (H,)
    offsets = offsets.round().to(torch.long)

    def hilbert_block_coords(n):
        pow2 = 1 << (n - 1).bit_length()
        hb = _hilbert_curve_coords(pow2)
        if pow2 == n:
            return hb
        out=[]
        for (iy, ix) in hb:
            ix = int(ix * (n - 1) / (pow2 - 1))
            iy = int(iy * (n - 1) / (pow2 - 1))
            out.append((iy, ix))
        return out

    def local_block(y0, y1):
        d_local = D[y0:y1].mean()
        return block_base if d_local < 0.5 else max(4, block_base // 2)

    order=[]
    for sy in range(0, H, stripe):
        rows = list(range(sy, min(sy + stripe, H)))
        blk = local_block(sy, min(sy + stripe, H))
        hb = hilbert_block_coords(blk)

        for bx in range(0, W, blk):
            sub_cols = []
            for (iy, ix) in hb:
                x = bx + ix
                if x < W:
                    sub_cols.append(x)
            for rid, y in enumerate(rows):
                off = int(offsets[y].item())
                cols = sub_cols.copy()
                cols = cols[off % W:] + cols[:off % W]
                if (y - sy) % 2 == 1:
                    cols = list(reversed(cols))
                for x in cols:
                    order.append(_linear_index(y, x, W))

    seen = set()
    out = []
    for v in order:
        if v not in seen:
            seen.add(v); out.append(v)
    if len(out) < H * W:
        rest = [i for i in range(H * W) if i not in seen]
        out.extend(rest)

    return torch.tensor(out[:H*W], device=device, dtype=torch.long)

# ----------------------------
# 2) RACS：修复感知的跨尺度交错
# ----------------------------
@torch.no_grad()
def build_racs_indices(H, W, theta, D, ds=2, interleave=(3, 1), steps=3, device="cpu"):
    """
    RACS（真·低分辨率 cross-scale）：
      1) 在低分辨率网格 (Hs, Ws) 上构建路径：按低分 D 优先选起点，并沿低分 θ 的方向做平滑游走；
      2) 将每个低分网格 cell 映射回高分辨率 ds×ds patch，并在 patch 内做局部蛇形扫描；
      3) 输出长度恰为 H*W 的排列索引（每个像素出现一次），可直接用于重排/neighbor。
    注：interleave 保留为兼容参数，此版本不再做“高分蛇形 hi 与 lo 交错”，因为 RACS 本身就是辅助路径。
    """

    # ------- 0) 准备高分 D / θ -------
    dmap = D.mean(dim=0)      # (H, W)
    tmap = theta.mean(dim=0)  # (H, W)

    # ------- 1) 构建低分网格 (Hs, Ws) 并做低分聚合 -------
    Hs = (H + ds - 1) // ds
    Ws = (W + ds - 1) // ds

    d_low = torch.zeros((Hs, Ws), device=dmap.device, dtype=dmap.dtype)
    ux_low = torch.zeros((Hs, Ws), device=dmap.device, dtype=dmap.dtype)
    uy_low = torch.zeros((Hs, Ws), device=dmap.device, dtype=dmap.dtype)

    for gy in range(Hs):
        y0 = gy * ds
        y1 = min(y0 + ds, H)
        for gx in range(Ws):
            x0 = gx * ds
            x1 = min(x0 + ds, W)

            patch_d = dmap[y0:y1, x0:x1]
            d_low[gy, gx] = patch_d.mean()

            # 方向场用 unit-vector 平均，避免角度 wrap-around
            patch_t = tmap[y0:y1, x0:x1]
            ux = torch.cos(patch_t).mean()
            uy = torch.sin(patch_t).mean()
            n = torch.sqrt(ux * ux + uy * uy) + 1e-8
            ux_low[gy, gx] = ux / n
            uy_low[gy, gx] = uy / n

    # ------- 2) 低分锚点（退化优先） -------
    anchors = [(gy, gx) for gy in range(Hs) for gx in range(Ws)]
    anchors.sort(key=lambda p: float(d_low[p[0], p[1]]), reverse=True)

    # ------- 3) 在低分网格上沿方向游走（cross-scale path） -------
    # 8 邻域方向
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def best_step(gy, gx):
        vx = float(ux_low[gy, gx])
        vy = float(uy_low[gy, gx])
        best = (gy, gx)
        best_score = -1e9
        for dy, dx in dirs:
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < Hs and 0 <= nx < Ws:
                # 与方向一致性（dot） + 轻微偏向更坏区域（可选）
                dot = vx * dx + vy * dy
                score = dot + 0.15 * float(d_low[ny, nx])
                if score > best_score:
                    best_score = score
                    best = (ny, nx)
        return best

    cell_path = []
    used_cell = set()

    # 先从坏处优先的 anchors 出发，每个起点走 steps 步
    for (gy0, gx0) in anchors:
        if (gy0, gx0) in used_cell:
            continue
        gy, gx = gy0, gx0
        for _ in range(max(1, int(steps))):
            if (gy, gx) not in used_cell:
                used_cell.add((gy, gx))
                cell_path.append((gy, gx))

            ny, nx = best_step(gy, gx)
            # 如果原地不动或下一步已访问过太多，就停止这条游走
            if (ny, nx) == (gy, gx):
                break
            gy, gx = ny, nx

    # 把没覆盖到的低分 cell 按退化优先补齐，保证最终覆盖所有像素
    for (gy, gx) in anchors:
        if (gy, gx) not in used_cell:
            used_cell.add((gy, gx))
            cell_path.append((gy, gx))

    # ------- 4) 低分 cell → 高分 patch 展开（patch 内蛇形扫描） -------
    out = []
    used_pix = set()

    for (gy, gx) in cell_path:
        y0 = gy * ds
        x0 = gx * ds
        y1 = min(y0 + ds, H)
        x1 = min(x0 + ds, W)

        for yy in range(y0, y1):
            xs = list(range(x0, x1))
            if ((yy - y0) % 2) == 1:
                xs = list(reversed(xs))
            for xx in xs:
                idx = _linear_index(yy, xx, W)
                if idx not in used_pix:
                    used_pix.add(idx)
                    out.append(idx)

    # ------- 5) 兜底补齐（理论上不会缺，但保证鲁棒） -------
    if len(out) < H * W:
        for yy in range(H):
            for xx in range(W):
                idx = _linear_index(yy, xx, W)
                if idx not in used_pix:
                    out.append(idx)
                    used_pix.add(idx)

    assert len(out) == H * W, f"RACS produced {len(out)} != {H*W}"
    return torch.tensor(out, device=device, dtype=torch.long)



    # ======================================================
# 调试辅助模块（仅日志打印，不影响计算图）
# ======================================================
# import time

# _last_debug = {'hw': None, 'count': 0, 'start': time.time()}

# def debug_scan_stats(H, W, theta, D, idx1, idx2, fusion="channel", fuse=None):
#     """
#     用于在训练时打印双扫描机制状态。
#     每1000个batch自动打印一次：
#       - 图像尺寸 (H,W)
#       - 平均方向角度 theta 均值
#       - 平均退化强度 D 均值
#       - CSH/RACS 索引差异度
#       - 当前融合模式（channel / temporal）
#     """
#     global _last_debug
#     _last_debug['count'] += 1
#     if _last_debug['count'] % 1000 != 0:
#         return  # 1000次打印一次

#     delta_t = time.time() - _last_debug['start']
#     _last_debug['start'] = time.time()
#     _last_debug['hw'] = (H, W)

#     diff_ratio = (idx1 != idx2).float().mean().item()
#     fuse_str = fuse if (fusion == "temporal" and fuse) else "1:1 (channel)"

#     print(f"[DEBUG][DualScan] step={_last_debug['count']}, size=({H}×{W}), "
#           f"θ_mean={theta.mean():.3f}, D_mean={D.mean():.3f}, "
#           f"CSH≠RACS比率={diff_ratio:.3f}, Δt={delta_t:.2f}s, "
#           f"fusion={fusion}, fuse={fuse_str}", flush=True)

# import time
# import torch
# import torch.distributed as dist

# _last_debug = {'count': 0, 'start': time.time()}

# def debug_scan_stats(H, W, theta, D, idx1, idx2, fusion="channel", fuse=None):
#     global _last_debug
#     _last_debug['count'] += 1
#     if _last_debug['count'] % 1000 != 0:
#         return

#     # 只在 rank0 打印，避免 DDP 多卡刷屏
#     if dist.is_available() and dist.is_initialized():
#         if dist.get_rank() != 0:
#             return

#     with torch.no_grad():
#         delta_t = time.time() - _last_debug['start']
#         _last_debug['start'] = time.time()

#         # 全部 detach，避免任何图引用
#         theta_m = theta.detach().mean().item()
#         D_m = D.detach().mean().item()
#         diff_ratio = (idx1.detach() != idx2.detach()).float().mean().item()

#         # 可选：更有用（建议加）
#         theta_s = theta.detach().std().item()
#         D_s = D.detach().std().item()

#         fuse_str = fuse if (fusion == "temporal" and fuse) else "1:1 (channel)"
#         print(
#             f"[DEBUG][DualScan] step={_last_debug['count']}, size=({H}×{W}), "
#             f"θ_mean={theta_m:.3f}, θ_std={theta_s:.3f}, "
#             f"D_mean={D_m:.3f}, D_std={D_s:.3f}, "
#             f"CSH≠RACS比率={diff_ratio:.3f}, Δt={delta_t:.2f}s, "
#             f"fusion={fusion}, fuse={fuse_str}",
#             flush=True
#         )

# ======================================================
# Debug helper (logging only, does not affect autograd)
# ======================================================
# import time
# import torch.distributed as dist

# _last_debug = {"count": 0, "start": time.time()}

# def debug_scan_stats(
#     H, W, theta, D, idx1, idx2,
#     fusion="channel", fuse=None,
#     rebuild=None,
#     every=1000,
# ):
#     """
#     每 every 次调用打印一次（注意：这里的“调用次数”是“模块调用次数”，不是 iter）
#     打印：
#       - theta/D 的均值与标准差（看 guidance 是否空间塌缩）
#       - idx1/idx2 的差异比例
#       - idx 的短签名（看 idx 是否一直固定）
#       - rebuild 是否发生（看 cache 是否在起作用）
#     """
#     global _last_debug
#     _last_debug["count"] += 1
#     if every is not None and every > 0 and (_last_debug["count"] % every != 0):
#         return

#     # 只在 rank0 打印（单卡时 dist 未初始化，不影响）
#     if dist.is_available() and dist.is_initialized():
#         if dist.get_rank() != 0:
#             return

#     with torch.no_grad():
#         delta_t = time.time() - _last_debug["start"]
#         _last_debug["start"] = time.time()

#         theta_det = theta.detach()
#         D_det = D.detach()
#         idx1_det = idx1.detach()
#         idx2_det = idx2.detach()

#         theta_m = theta_det.mean().item()
#         theta_s = theta_det.std().item()
#         D_m = D_det.mean().item()
#         D_s = D_det.std().item()

#         diff_ratio = (idx1_det != idx2_det).float().mean().item()

#         # idx 的短签名：如果一直不变，说明 idx 固定（很可能被 cache 住）
#         sig1 = int(idx1_det[:128].sum().item())
#         sig2 = int(idx2_det[:128].sum().item())

#         fuse_str = fuse if (fusion == "temporal" and fuse) else "1:1 (channel)"
#         print(
#             f"[DEBUG][DualScan] call={_last_debug['count']}, size=({H}×{W}), "
#             f"θ_mean={theta_m:.3f}, θ_std={theta_s:.3f}, "
#             f"D_mean={D_m:.3f}, D_std={D_s:.3f}, "
#             f"CSH≠RACS={diff_ratio:.3f}, rebuild={rebuild}, "
#             f"idxsig=({sig1},{sig2}), Δt={delta_t:.2f}s, "
#             f"fusion={fusion}, fuse={fuse_str}",
#             flush=True
#         )

