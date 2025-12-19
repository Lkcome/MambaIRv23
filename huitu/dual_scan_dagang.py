# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, FancyArrowPatch

# plt.rcParams["font.family"] = "DejaVu Sans"  # 论文可以换 Times New Roman 等

# fig, axes = plt.subplots(2, 2, figsize=(6, 6))
# axes = axes.flatten()

# for ax in axes:
#     ax.set_aspect("equal")
#     ax.axis("off")

# def draw_grid(ax, x0=0, y0=0, w=4, h=4, nx=4, ny=4, fc="#fff9e6"):
#     ax.add_patch(Rectangle((x0, y0), w, h,
#                            facecolor=fc, edgecolor="black", linewidth=1.0))
#     for i in range(1, ny):
#         ax.plot([x0, x0+w], [y0+i*h/ny, y0+i*h/ny],
#                 color="#bfbfbf", linewidth=0.6)
#     for j in range(1, nx):
#         ax.plot([x0+j*w/nx, x0+j*w/nx], [y0, y0+h],
#                 color="#bfbfbf", linewidth=0.6)

# def arrow(ax, p1, p2, **kw):
#     arr = FancyArrowPatch(p1, p2,
#                           arrowstyle="-|>", mutation_scale=10,
#                           linewidth=1.0, **kw)
#     ax.add_patch(arr)

# # ---------------- (a) Z-scan ----------------
# ax = axes[0]
# ax.set_title("(a) Z-scan", fontsize=10, pad=4)

# draw_grid(ax)

# # 简单画一个 Z 形路径
# pts = [(0.3, 3.7), (3.7, 3.7), (0.3, 0.3), (3.7, 0.3)]
# for i in range(len(pts)-1):
#     ax.plot([pts[i][0], pts[i+1][0]],
#             [pts[i][1], pts[i+1][1]],
#             color="black", linewidth=1.2)
# arrow(ax, pts[-2], pts[-1], color="black")

# # ---------------- (b) 单一路径 S-scan ----------------
# ax = axes[1]
# ax.set_title("(b) S-scan", fontsize=10, pad=4)

# draw_grid(ax)

# # 蛇形/S型扫描：每行反向
# path = []
# nx, ny = 4, 4
# x0, y0, w, h = 0, 0, 4, 4
# for row in range(ny):
#     y = y0 + (ny-row-0.5)*h/ny
#     if row % 2 == 0:
#         xs = np.linspace(x0+0.5*w/nx, x0+w-0.5*w/nx, nx)
#     else:
#         xs = np.linspace(x0+w-0.5*w/nx, x0+0.5*w/nx, nx)
#     for x in xs:
#         path.append((x, y))
# px, py = zip(*path)
# ax.plot(px, py, color="black", linewidth=1.2)
# arrow(ax, path[-2], path[-1], color="black")

# # ---------------- (c) Local window + Z-scan ----------------
# ax = axes[2]
# ax.set_title("(c) Window + Z-scan", fontsize=10, pad=4)

# # 四个 window
# win_w, win_h = 2, 2
# colors = ["#e6f2ff", "#e6ffe6", "#fff0e6", "#f5e6ff"]
# idx = 0
# for i in range(2):
#     for j in range(2):
#         x0, y0 = j*win_w, i*win_h
#         draw_grid(ax, x0, y0, win_w, win_h, nx=2, ny=2, fc=colors[idx])
#         # 每个 window 里面画一个迷你 Z
#         p1 = (x0+0.2, y0+win_h-0.2)
#         p2 = (x0+win_w-0.2, y0+win_h-0.2)
#         p3 = (x0+0.2, y0+0.2)
#         p4 = (x0+win_w-0.2, y0+0.2)
#         mini = [p1, p2, p3, p4]
#         for k in range(len(mini)-1):
#             ax.plot([mini[k][0], mini[k+1][0]],
#                     [mini[k][1], mini[k+1][1]],
#                     color="black", linewidth=0.9)
#         arrow(ax, mini[-2], mini[-1], color="black")
#         idx += 1

# # ---------------- (d) Ours: CSH + RACS ----------------
# ax = axes[3]
# ax.set_title("(d) Ours: CSH + RACS", fontsize=10, pad=4)

# # --------- 背景竖直条带（CSH 的条带）---------
# W, H = 4, 3
# stripe_w = W / 3.0
# colors = ["#e6f2ff", "#fff5e6", "#e6ffe6"]
# for k in range(3):
#     x0 = k * stripe_w
#     ax.add_patch(Rectangle((x0, 0), stripe_w, H,
#                            facecolor=colors[k], edgecolor="none", alpha=0.9))

# # 外边框
# ax.add_patch(Rectangle((0, 0), W, H,
#                        fill=False, edgecolor="black", linewidth=1.0))

# # --------- CSH 主路径：在条带中间弯曲穿行 ---------
# xs = np.linspace(0.3, W-0.3, 200)
# ys = 1.5 + 0.9 * np.sin(2*np.pi*xs / W)   # 中心在 1.5，高度约 3
# ax.plot(xs, ys, color="black", linewidth=1.4)
# # 终点箭头
# ax.arrow(xs[-2], ys[-2],
#          xs[-1]-xs[-2], ys[-1]-ys[-2],
#          head_width=0.12, head_length=0.25,
#          fc="black", ec="black", length_includes_head=True)

# ax.text(0.15, H-0.15, "CSH\ncurved stripe",
#         fontsize=8, ha="left", va="top")

# # --------- RACS：上方一条低分路径 + 向 CSH 插入的箭头 ---------
# # 低分路径（2x 或 3x 节点），表示 low-res scan
# low_y = H + 0.5
# low_xs = np.linspace(0.5, W-0.5, 4)
# low_ys = low_y + 0.2*np.sin(np.linspace(0, np.pi, 4))
# ax.plot(low_xs, low_ys, color="#d62728", linewidth=1.0, linestyle="--")
# for x, y in zip(low_xs, low_ys):
#     ax.scatter(x, y, s=10, color="#d62728")

# ax.text(W/2, low_y+0.45, "RACS low-res path", fontsize=8,
#         ha="center", va="bottom", color="#d62728")

# # 从 low-res path “插入”到 CSH 曲线上的若干点
# # 选几个对应的索引
# idxs = [30, 90, 150]
# for li, ci in zip([0, 1, 3], idxs):
#     sx, sy = low_xs[li], low_ys[li]
#     tx, ty = xs[ci], ys[ci]
#     arr = FancyArrowPatch((sx, sy), (tx, ty),
#                           arrowstyle="-|>", mutation_scale=10,
#                           linewidth=1.0, linestyle="dashed",
#                           color="#d62728")
#     ax.add_patch(arr)

# ax.text(W-0.05, 0.2,
#         "low-res tokens\ninjected into\nCSH sequence",
#         fontsize=8, ha="right", va="bottom", color="#d62728")

# ax.set_xlim(-0.2, W+0.2)
# ax.set_ylim(-0.3, H+1.0)

# plt.tight_layout()
# plt.savefig("dual_scan_dagang.png", dpi=300, bbox_inches="tight")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "DejaVu Sans"  # 论文可改成 Times New Roman 等

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flatten()

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

def draw_grid(ax, x0=0, y0=0, w=4, h=4, nx=4, ny=4, fc="#fff9e6"):
    ax.add_patch(Rectangle((x0, y0), w, h,
                           facecolor=fc, edgecolor="black", linewidth=1.0))
    for i in range(1, ny):
        ax.plot([x0, x0+w], [y0+i*h/ny, y0+i*h/ny],
                color="#bfbfbf", linewidth=0.6)
    for j in range(1, nx):
        ax.plot([x0+j*w/nx, x0+j*w/nx], [y0, y0+h],
                color="#bfbfbf", linewidth=0.6)

def add_arrow(ax, p1, p2, **kw):
    arr = FancyArrowPatch(p1, p2,
                          arrowstyle="-|>", mutation_scale=10,
                          linewidth=1.0, **kw)
    ax.add_patch(arr)

# ---------------- (a) Z-scan ----------------
ax = axes[0]
ax.set_title("(a) Z-scan", fontsize=10, pad=4)
draw_grid(ax)

pts = [(0.3, 3.7), (3.7, 3.7), (0.3, 0.3), (3.7, 0.3)]
for i in range(len(pts)-1):
    ax.plot([pts[i][0], pts[i+1][0]],
            [pts[i][1], pts[i+1][1]],
            color="black", linewidth=1.2)
add_arrow(ax, pts[-2], pts[-1], color="black")

# ---------------- (b) S-scan ----------------
ax = axes[1]
ax.set_title("(b) S-scan", fontsize=10, pad=4)
draw_grid(ax)

path = []
nx, ny = 4, 4
x0, y0, w, h = 0, 0, 4, 4
for row in range(ny):
    y = y0 + (ny-row-0.5)*h/ny
    if row % 2 == 0:
        xs = np.linspace(x0+0.5*w/nx, x0+w-0.5*w/nx, nx)
    else:
        xs = np.linspace(x0+w-0.5*w/nx, x0+0.5*w/nx, nx)
    for x in xs:
        path.append((x, y))
px, py = zip(*path)
ax.plot(px, py, color="black", linewidth=1.2)
add_arrow(ax, path[-2], path[-1], color="black")

# ---------------- (c) Window + Z-scan ----------------
ax = axes[2]
ax.set_title("(c) Window + Z-scan", fontsize=10, pad=4)

win_w, win_h = 2, 2
colors_win = ["#e6f2ff", "#e6ffe6", "#fff0e6", "#f5e6ff"]
idx = 0
for i in range(2):
    for j in range(2):
        x0, y0 = j*win_w, i*win_h
        draw_grid(ax, x0, y0, win_w, win_h, nx=2, ny=2,
                  fc=colors_win[idx])
        p1 = (x0+0.25, y0+win_h-0.25)
        p2 = (x0+win_w-0.25, y0+win_h-0.25)
        p3 = (x0+0.25, y0+0.25)
        p4 = (x0+win_w-0.25, y0+0.25)
        mini = [p1, p2, p3, p4]
        for k in range(len(mini)-1):
            ax.plot([mini[k][0], mini[k+1][0]],
                    [mini[k][1], mini[k+1][1]],
                    color="black", linewidth=0.9)
        add_arrow(ax, mini[-2], mini[-1], color="black")
        idx += 1

# ---------------- (d) Ours: CSH + RACS ----------------
ax = axes[3]
ax.set_title("(d) Ours: CSH + RACS", fontsize=10, pad=4)

# 竖直条带 + 网格
W, H = 4, 4
stripe_w = W / 3.0
stripe_colors = ["#e6f2ff", "#fff5e6", "#e6ffe6"]
for k in range(3):
    x0 = k * stripe_w
    ax.add_patch(Rectangle((x0, 0), stripe_w, H,
                           facecolor=stripe_colors[k],
                           edgecolor="none", alpha=0.9))
draw_grid(ax, 0, 0, W, H, nx=4, ny=4, fc="none")  # 叠加网格框
ax.add_patch(Rectangle((0, 0), W, H,
                       fill=False, edgecolor="black", linewidth=1.0))

# ---- CSH 主路径：条带内 & 跨条带的 S 形 ----
c_path = []
for col_block in range(3):          # 三条条带
    xs_block = np.linspace(col_block*stripe_w + stripe_w/4,
                           col_block*stripe_w + 3*stripe_w/4, 4)
    # 奇偶条带交替向下 / 向上走，表示“within & across stripes”
    if col_block % 2 == 0:
        ys_block = np.linspace(H-0.5, 0.5, 4)
    else:
        ys_block = np.linspace(0.5, H-0.5, 4)
    for x, y in zip(xs_block, ys_block):
        c_path.append((x, y))

cpx, cpy = zip(*c_path)
ax.plot(cpx, cpy, color="black", linewidth=1.4)
add_arrow(ax, c_path[-2], c_path[-1], color="black")

ax.text(0.15, H-0.15, "CSH\n(curved stripes)", fontsize=8,
        ha="left", va="top")

# ---- RACS 路径：粗粒度红色虚线 + 与 CSH 的交点 ----
r_nodes = [(0.7, 3.3), (1.3, 2.1), (2.4, 1.2), (3.3, 0.7)]
rpx, rpy = zip(*r_nodes)
ax.plot(rpx, rpy, color="#d62728", linewidth=1.1, linestyle="--")
for x, y in r_nodes:
    ax.add_patch(Circle((x, y), 0.08, color="#d62728"))

# 把 RACS 的几个节点“对齐”到最近的 CSH 路径点，用灰色细线表示交错位置
for (rx, ry) in r_nodes:
    # 找距当前 RACS 节点最近的 CSH 点
    d2 = (np.array(cpx) - rx)**2 + (np.array(cpy) - ry)**2
    j = int(np.argmin(d2))
    cx, cy = cpx[j], cpy[j]
    ax.plot([rx, cx], [ry, cy],
            color="#d62728", linewidth=0.8, linestyle="dotted")

ax.text(W-0.05, 0.2,
        "RACS (red dashed):\n"
        "low-res path\ninterleaved at\njoint nodes",
        fontsize=7.5, ha="right", va="bottom", color="#d62728")

ax.set_xlim(-0.2, W+0.2)
ax.set_ylim(-0.2, H+0.2)

plt.tight_layout()
plt.savefig("dual_scan_dagang.png", dpi=300, bbox_inches="tight")
