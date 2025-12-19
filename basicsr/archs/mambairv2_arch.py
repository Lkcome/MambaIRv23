import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
# from .scan_strategies_advanced import (
#     sobel_orientation, DegradationHead,
#     build_csh_indices, build_racs_indices
# )
from .scan_strategies_advanced import (
    build_csh_indices,
    build_racs_indices,
    GuidanceHead,
    ScanWeightHead,
)




def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.,**kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        # === 新增：双扫描（CSH+RACS）选项、引导头与缓存 ===
        # self.dual_scan = kwargs.get("dual_scan", True)
        # self.scan_opts = kwargs.get("scan_opts", {
        #     "stripe": 8, "kappa": 1.0, "block_base": 8,   # CSH
        #     "ds": 2, "interleave": (3, 1), "steps": 3     # RACS
        # })
        # self.use_guidance = kwargs.get("use_guidance", True)

        # if self.use_guidance:
        #     self.deg_head = DegradationHead(in_ch=1)
        
        self.dual_scan = kwargs.get("dual_scan", True)
        self.scan_opts = kwargs.get("scan_opts", {
            "stripe": 8, "kappa": 1.0, "block_base": 8,   # CSH
            "ds": 2, "interleave": (3, 1), "steps": 3,    # RACS
            # --- 建议默认开启：真正“动态双扫描 + 时域插入式融合” ---
            "dynamic_idx": True,                          # True: 每次 forward 都根据 θ/D 重新生成扫描索引
            "fusion": "temporal",                         # "channel" | "temporal"
            "fuse": (3, 1),                               # temporal 时的插入节奏：每 3 个 CSH 插 1 个 RACS
            # 可选：降低开销（默认200次更新一次）
            "idx_update_interval": 200,
        })

        self.use_guidance = kwargs.get("use_guidance", True)
        
        if self.use_guidance:
            # 统一的可学习引导头: 输出 theta, D
            self.guidance_head = GuidanceHead(in_ch=1)
            # 分别给 CSH / RACS 一套 soft 权重
            self.csh_weight_head  = ScanWeightHead(in_ch=2)
            self.racs_weight_head = ScanWeightHead(in_ch=2)
        
        # --- scan index cache ---
        self.register_buffer("_idx1", None, persistent=False)
        self.register_buffer("_idx2", None, persistent=False)
        self.register_buffer("_rev1", None, persistent=False)
        self.register_buffer("_rev2", None, persistent=False)
        self.register_buffer("_cache_hw", torch.tensor([-1, -1], dtype=torch.long), persistent=False)

        # --- forward counter for interval update ---
        self.register_buffer("_fwd_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.idx_update_interval = int(self.scan_opts.get("idx_update_interval", 200))

        # --- temporal fused index cache (optional) ---
        self.register_buffer("_idx_fused", None, persistent=False)
        self.register_buffer("_inv_fused", None, persistent=False)
        self.register_buffer("_src_fused", None, persistent=False)

        # --- fusion config ---
        self.mix_proj = None
        self.fusion = str(self.scan_opts.get("fusion", "channel")).lower()
        self.fuse_steps = tuple(self.scan_opts.get("fuse", []))
        if self.fusion not in {"channel", "temporal"}:
            self.fusion = "channel"




    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        #1.生成Promot路由相关
        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        #2.生成语义排序索引（仅用于fallback）
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        #3.投影与特征准备
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        # x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c
        x = x.view(B, cc, -1).permute(0, 2, 1)  # (B,N,C)

        # === 新增：
        # 4.CSH + RACS 双扫描互补（保留 ASE 路由与 prompt 不变） ===
        # 说明：
        # - 使用共享引导：方向场 theta、退化图 D
        # - 生成两套索引：idx1=CSH、idx2=RACS（与(H,W)绑定缓存）
        # - 重排得到 x_csh / x_racs -> 通道拼接 -> 线性降回 C -> 送入 SelectiveScan
        # - 逆映射使用 CSH 的逆索引



        # if self.dual_scan:
        #     # ---- 4.1 引导信号生成 ----
        #     if self.use_guidance:
        #         x_img = x.mean(dim=-1).view(B, 1, H, W)     # (B,1,H,W)
        #         theta = sobel_orientation(x_img, H, W)      # (B,H,W)
        #         D = self.deg_head(x_img)                    # (B,H,W)
        #     else:
        #         theta = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
        #         D = torch.zeros(B, H, W, device=x.device, dtype=x.dtype) + 0.5

        #     # ---- 4.2 缓存索引 (避免重复生成) ----
        #     if (self._cache_hw[0].item() != H) or (self._cache_hw[1].item() != W) or (self._idx1 is None):
        #         idx1 = build_csh_indices(
        #             H, W, theta=theta, D=D,
        #             stripe=self.scan_opts.get("stripe", 8),
        #             kappa=self.scan_opts.get("kappa", 1.0),
        #             block_base=self.scan_opts.get("block_base", 8),
        #             device=x.device
        #         )
        #         idx2 = build_racs_indices(
        #             H, W, theta=theta, D=D,
        #             ds=self.scan_opts.get("ds", 2),
        #             interleave=self.scan_opts.get("interleave", (3, 1)),
        #             steps=self.scan_opts.get("steps", 3),
        #             device=x.device
        #         )
        #         # === 新增：调试打印 ===
        #         #from basicsr.archs.scan_strategies_advanced import debug_scan_stats
        #         #debug_scan_stats(H, W, theta, D, idx1, idx2,fusion=self.fusion, fuse=self.fuse_steps)

        #         rev1 = torch.argsort(idx1)
        #         rev2 = torch.argsort(idx2)
        #         self._idx1, self._idx2 = idx1[None, :], idx2[None, :]
        #         self._rev1, self._rev2 = rev1[None, :], rev2[None, :]
        #         self._cache_hw[...] = torch.tensor([H, W], device=x.device)


        if self.dual_scan:
            # === 1) 可学习 θ, D ===
            if self.use_guidance:
                # 从 token 还原成伪图像 (和之前一样，用平均通道)
                x_img = x.mean(dim=-1).view(B, 1, H, W)     # (B,1,H,W)
                theta, D = self.guidance_head(x_img)        # (B,H,W), (B,H,W)
            else:
                theta = torch.zeros(B, H, W, device=x.device)
                D      = torch.full((B, H, W), 0.5, device=x.device)
        
            # 未调试打印原版
            # === 2) 构造索引（动态：每 idx_update_interval 次更新一次；索引本身仍不可微） ===
            dynamic_idx = bool(self.scan_opts.get("dynamic_idx", True))
            idx_update_interval = int(self.scan_opts.get("idx_update_interval", 200))

            # forward 计数
            self._fwd_count += 1

            # 是否需要重建
            rebuild = (self._idx1 is None) or (int(self._cache_hw[0].item()) != H) or (int(self._cache_hw[1].item()) != W)
            if dynamic_idx and (idx_update_interval > 1):
                rebuild = rebuild or (int(self._fwd_count.item()) % idx_update_interval == 0)
            elif dynamic_idx and (idx_update_interval <= 1):
                rebuild = True

            if rebuild:
                # ★ 在 CPU 上生成离散索引，避免 GPU 张量在 Python 循环里频繁 .item() 同步
                theta_idx = theta.detach().to("cpu")
                D_idx     = D.detach().to("cpu")

                idx1 = build_csh_indices(
                    H, W,
                    theta_idx, D_idx,
                    stripe=self.scan_opts.get("stripe", 8),
                    kappa=self.scan_opts.get("kappa", 1.0),
                    block_base=self.scan_opts.get("block_base", 8),
                    device="cpu",
                )
                idx2 = build_racs_indices(
                    H, W,
                    theta_idx, D_idx,
                    ds=self.scan_opts.get("ds", 2),
                    interleave=self.scan_opts.get("interleave", (3, 1)),
                    steps=self.scan_opts.get("steps", 3),
                    device="cpu",
                )

                rev1 = torch.argsort(idx1)
                rev2 = torch.argsort(idx2)

                self._idx1 = idx1[None, :].to(x.device)
                self._idx2 = idx2[None, :].to(x.device)
                self._rev1 = rev1[None, :].to(x.device)
                self._rev2 = rev2[None, :].to(x.device)
                self._cache_hw[...] = torch.tensor([H, W], device=x.device, dtype=torch.long)
            # ===== [ADD] build temporal-fused index cache only when needed =====
            use_temporal = (self.fusion == "temporal") and (len(self.fuse_steps) == 2)
            if use_temporal:
                need_fuse = rebuild or (self._idx_fused is None) or (self._inv_fused is None) or (self._src_fused is None)
                if need_fuse:
                    m, n2 = int(self.fuse_steps[0]), int(self.fuse_steps[1])

                    # fused idx 只需要基于 idx1/idx2（与 B 无关），放 CPU 拼一次即可
                    idx_csh_1d = self._idx1[0].detach().cpu()
                    idx_rac_1d = self._idx2[0].detach().cpu()
                    N = idx_csh_1d.numel()

                    out = []
                    src = []
                    used = torch.zeros(N, dtype=torch.bool)  # CPU

                    i = j = 0
                    while i < N or j < N:
                        c = 0
                        while i < N and c < m:
                            v = int(idx_csh_1d[i].item()); i += 1
                            if not used[v]:
                                used[v] = True
                                out.append(v); src.append(0); c += 1

                        c = 0
                        while j < N and c < n2:
                            v = int(idx_rac_1d[j].item()); j += 1
                            if not used[v]:
                                used[v] = True
                                out.append(v); src.append(1); c += 1

                        if i >= N and j >= N:
                            break

                    if len(out) < N:
                        for v in range(N):
                            if not used[v]:
                                out.append(v); src.append(0)

                    idx_fused_cpu = torch.tensor(out, dtype=torch.long)  # (N,)
                    src_cpu = torch.tensor(src, dtype=torch.long)        # (N,)

                    # inv_fused：像素id -> fused序列位置（CPU 上一次性算好）
                    inv_cpu = torch.empty_like(idx_fused_cpu)
                    inv_cpu[idx_fused_cpu] = torch.arange(N, dtype=torch.long)

                    # 缓存到 GPU（后续每个 iter 直接复用）
                    self._idx_fused = idx_fused_cpu[None, :].to(x.device)  # (1,N)
                    self._inv_fused = inv_cpu[None, :].to(x.device)        # (1,N)
                    self._src_fused = src_cpu.to(x.device)                 # (N,)
            # ===== [ADD END] =====



            # # === 2) 构造索引用 detach 版本 (保持索引不可微) ===
            # theta_idx = theta.detach()
            # D_idx     = D.detach()

            # # 关键：need_rebuild 必须在 build 之前计算，才能真实反映 cache 行为
            # need_rebuild = (int(self._cache_hw[0].item()) != H) or (int(self._cache_hw[1].item()) != W) or (self._idx1 is None)

            # if need_rebuild:
            #     idx1 = build_csh_indices(
            #         H, W,
            #         theta_idx, D_idx,
            #         stripe=self.scan_opts.get("stripe", 8),
            #         kappa=self.scan_opts.get("kappa", 1.0),
            #         block_base=self.scan_opts.get("block_base", 8),
            #     )   # (N,)
            #     idx2 = build_racs_indices(
            #         H, W,
            #         theta_idx, D_idx,
            #         ds=self.scan_opts.get("ds", 2),
            #         interleave=self.scan_opts.get("interleave", (3, 1)),
            #         steps=self.scan_opts.get("steps", 3),
            #     )   # (N,)

            #     rev1 = torch.argsort(idx1)
            #     rev2 = torch.argsort(idx2)

            #     self._idx1 = idx1[None, :].to(x.device)   # (1,N)
            #     self._idx2 = idx2[None, :].to(x.device)
            #     self._rev1 = rev1[None, :].to(x.device)
            #     self._rev2 = rev2[None, :].to(x.device)
            #     self._cache_hw[...] = torch.tensor([H, W], device=x.device, dtype=torch.long)

            # # debug：每次 forward 都调用，但内部每 every 次才打印一次
            # from basicsr.archs.scan_strategies_advanced import debug_scan_stats
            # debug_scan_stats(
            #     H, W, theta, D,
            #     self._idx1[0], self._idx2[0],
            #     fusion=self.fusion, fuse=self.fuse_steps,
            #     rebuild=need_rebuild,
            #     every=1000,
            # )



            # # ---- 4.3 序列重排 ----
            # x_csh  = semantic_neighbor(x, self._idx1.repeat(B, 1))   # (B, n, C)
            # x_racs = semantic_neighbor(x, self._idx2.repeat(B, 1))   # (B, n, C)

            # === 3) 计算两路的 soft 权重图 (B,H,W) ===
            if self.use_guidance:
                w_csh_2d  = self.csh_weight_head(theta, D)   # (B,H,W)
                w_racs_2d = self.racs_weight_head(theta, D)  # (B,H,W)
            else:
                w_csh_2d  = torch.ones(B, H, W, device=x.device)
                w_racs_2d = torch.ones(B, H, W, device=x.device)
        
            # 展平成 (B,N,1)，以便和 token 对齐
            w_csh_flat  = w_csh_2d.view(B, -1, 1)   # (B,N,1)
            w_racs_flat = w_racs_2d.view(B, -1, 1)  # (B,N,1)
        
            # === 4) 根据两种索引，把权重也 gather 成序列顺序 ===
            idx1_b = self._idx1.repeat(B, 1)  # (B,N)
            idx2_b = self._idx2.repeat(B, 1)
        
            w_csh_seq  = semantic_neighbor(w_csh_flat,  idx1_b)  # (B,N,1)
            w_racs_seq = semantic_neighbor(w_racs_flat, idx2_b)  # (B,N,1)
        
            # === 5) 先构造两路序列，再乘权重 ===
            x_csh  = semantic_neighbor(x, idx1_b)   # (B,N,C)
            x_racs = semantic_neighbor(x, idx2_b)   # (B,N,C)
        
            x_csh  = x_csh  * w_csh_seq             # 逐点加权 (可微)
            x_racs = x_racs * w_racs_seq

            

            # ---- 4.4 融合（可切换：channel / temporal） ----
            if (self.fusion == "temporal") and (len(self.fuse_steps) == 2):
                # ★ 时域交错融合（真正使用两路加权后的序列 x_csh / x_racs）
                # ===== [REPLACE] use cached fused indices =====
                idx_fused_1d = self._idx_fused[0]                 # (N,) on GPU
                src_1d = self._src_fused                          # (N,) on GPU
                idx_fused = self._idx_fused.expand(B, -1)         # (B,N) view
                inv_fused = self._inv_fused.expand(B, -1)         # (B,N) view
                # ===== [REPLACE END] =====
                N = idx_fused_1d.numel()


                # 用 “像素 id → 各自序列位置” 的逆映射，把 token 从 x_csh/x_racs 取出来再融合
                pos_csh = self._rev1[0].index_select(0, idx_fused_1d)  # (N,)
                pos_rac = self._rev2[0].index_select(0, idx_fused_1d)  # (N,)

                gather_csh = pos_csh.view(1, N, 1).expand(B, N, cc)
                gather_rac = pos_rac.view(1, N, 1).expand(B, N, cc)

                x_csh_fused = torch.gather(x_csh, dim=1, index=gather_csh)
                x_rac_fused = torch.gather(x_racs, dim=1, index=gather_rac)

                mask = (src_1d.view(1, N, 1).expand(B, N, cc) == 1)
                semantic_x = torch.where(mask, x_rac_fused, x_csh_fused)  # (B,N,C)

                y = self.selectiveScan(semantic_x, prompt)
                y = self.out_proj(self.out_norm(y))
                x = semantic_neighbor(y, inv_fused)  # (B,N,C)

            
            else:
                # ★ 通道融合（现行：1:1 拼通道 → Linear）
                if self.mix_proj is None:
                    self.mix_proj = torch.nn.Linear(cc * 2, cc).to(x.device)
                semantic_x = self.mix_proj(torch.cat([x_csh, x_racs], dim=-1))  # (B,N,C)
                #Selective Scan (保留 ASE 逻辑)
                y = self.selectiveScan(semantic_x, prompt)
                y = self.out_proj(self.out_norm(y))
                # 用 CSH 的逆索引回折（通道融合下，序列顺序与 CSH 对齐）
                x = semantic_neighbor(y, self._rev1.repeat(B, 1))


        else:
            # === 4*. 回退逻辑：原 SGN 扫描 ===
            semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
            y = self.selectiveScan(semantic_x, prompt)
            y = self.out_proj(self.out_norm(y))
            x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold
        # === 新增结束 ===

        #5.输出
        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 **kwargs,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = ASSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
            **kwargs #新增:透传配置参数
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # part1: Window-MHSA
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x


class BasicBlock(nn.Module):
    """ A basic ASSB for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        num_tokens (int): Token number for each token dictionary.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                AttentiveLayer(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth - 1,
                    **kwargs,#继续透传
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class ASSB(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 **kwargs,
                 ):
        super(ASSB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            **kwargs,#继续透传
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class MambaIRv2(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 d_state=8,
                 depths=(6, 6, 6, 6,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ASSB(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                **kwargs,# 把 YAML 中 dual_scan / use_guidance / scan_opts 传到底层
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        #coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x



if __name__ == '__main__':
    upscale = 4
    model = MambaIRv2(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect').cuda()

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)

    # Test
    _input = torch.randn([2, 3, 64, 64]).cuda()
    output = model(_input).cuda()
    print(output.shape)

