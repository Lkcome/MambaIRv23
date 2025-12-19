import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyArrow

plt.rcParams["font.family"] = "DejaVu Sans"  # 需要中文可改为 SimHei 等

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ax_thetaD, ax_csh, ax_racs = axes
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# ============================================================
# 1. θ & D 可视化
# ============================================================
ax_thetaD.set_title(r"(a) Guidance signals: direction field $\theta$ and degradation map $D$",
                    fontsize=11, pad=8)

# 网格大小
H, W = 9, 9
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

# 伪造一个方向场：中间偏斜，边缘水平/竖直
theta = np.arctan2(Y - H/2, X - W/2)  # [-pi,pi]
U = np.cos(theta)
V = np.sin(theta)

# 画 D：中心更“坏”——热力图
D = np.exp(-((X - W/2)**2 + (Y - H/2)**2) / (2*(W/4)**2))
im = ax_thetaD.imshow(D, cmap="Reds", origin="lower")
fig.colorbar(im, ax=ax_thetaD, fraction=0.046, pad=0.02)
ax_thetaD.text(-0.5, H + 0.3, r"$D$: higher = more degraded",
               fontsize=8, ha="left", va="bottom")

# 画 θ：在较稀疏的点上画箭头
step = 2
ax_thetaD.quiver(X[::step, ::step], Y[::step, ::step],
                 U[::step, ::step], V[::step, ::step],
                 color="black", scale=30, width=0.005)
ax_thetaD.text(-0.5, -1.0, r"$\theta$: local dominant direction",
               fontsize=8, ha="left", va="top")

# 边框
ax_thetaD.add_patch(Rectangle((-0.5, -0.5), W, H,
                              fill=False, linewidth=1.2, edgecolor="black"))

# ============================================================
# 2. CSH：弯曲条带 + Hilbert 子块
# ============================================================
ax_csh.set_title("(b) CSH: curved stripes + Hilbert sub-blocks",
                 fontsize=11, pad=8)

# 大矩形代表特征图
Wc, Hc = 10, 6
ax_csh.add_patch(Rectangle((0, 0), Wc, Hc,
                           fill=False, linewidth=1.2, edgecolor="black"))

# 画几条弯曲条带（用多段折线近似曲线）
num_stripes = 3
colors = ["#e6f2ff", "#fef0d9", "#e5f5e0"]
stripe_height = Hc / num_stripes

for i in range(num_stripes):
    y0 = i * stripe_height
    # 填充条带背景
    ax_csh.add_patch(Rectangle((0, y0), Wc, stripe_height,
                               facecolor=colors[i], edgecolor="none", alpha=0.7))

# 在中间条带画一条“弯曲轨迹”
xs = np.linspace(0.5, Wc - 0.5, 40)
ys = Hc/2 + 0.7*np.sin(2*np.pi*xs/Wc)
ax_csh.plot(xs, ys, color="black", linewidth=1.5)
ax_csh.arrow(xs[-2], ys[-2],
             xs[-1]-xs[-2], ys[-1]-ys[-2],
             head_width=0.18, head_length=0.4, fc="black", ec="black",
             length_includes_head=True)

ax_csh.text(0.2, Hc-0.4, "curved stripe\naligned with θ",
            fontsize=8, ha="left", va="top")

# 在条带内部放一个 Hilbert 子块示意
sub_x, sub_y = 2.0, Hc/2 - stripe_height/2 + 0.2
sub_w, sub_h = 3.0, stripe_height - 0.4
ax_csh.add_patch(Rectangle((sub_x, sub_y), sub_w, sub_h,
                           fill=False, linewidth=1.0, edgecolor="black"))
ax_csh.text(sub_x+sub_w/2, sub_y+sub_h+0.1, "Hilbert sub-block",
            fontsize=8, ha="center", va="bottom")

# 子块内部画 4x2 小格 + Hilbert 路径
nx, ny = 4, 2
dx, dy = sub_w/nx, sub_h/ny
for i in range(ny):
    for j in range(nx):
        ax_csh.add_patch(Rectangle((sub_x + j*dx, sub_y + i*dy),
                                   dx, dy, fill=False, edgecolor="#888888", linewidth=0.5))
# 简化版 Hilbert path
path_pts = [
    (sub_x+0.5*dx, sub_y+0.5*dy),
    (sub_x+0.5*dx, sub_y+1.5*dy),
    (sub_x+1.5*dx, sub_y+1.5*dy),
    (sub_x+1.5*dx, sub_y+0.5*dy),
    (sub_x+2.5*dx, sub_y+0.5*dy),
    (sub_x+2.5*dx, sub_y+1.5*dy),
    (sub_x+3.5*dx, sub_y+1.5*dy),
    (sub_x+3.5*dx, sub_y+0.5*dy),
]
px, py = zip(*path_pts)
ax_csh.plot(px, py, color="black", linewidth=1.0)

# 在“坏处加密”区域增加更多小格子
bad_region = Rectangle((6.5, Hc/2 - stripe_height/2 + 0.2),
                       3.0, stripe_height-0.4,
                       fill=False, edgecolor="red", linewidth=1.0, linestyle="--")
ax_csh.add_patch(bad_region)
ax_csh.text(6.5, bad_region.get_y()+bad_region.get_height()+0.1,
            "denser blocks\nin degraded region", fontsize=8,
            ha="left", va="bottom", color="red")

# 在坏处区域画更密的竖线/横线
nx2, ny2 = 6, 2
dx2 = bad_region.get_width()/nx2
dy2 = bad_region.get_height()/ny2
for i in range(ny2+1):
    ax_csh.plot([bad_region.get_x(), bad_region.get_x()+bad_region.get_width()],
                [bad_region.get_y()+i*dy2]*2, color="#bbbbbb", linewidth=0.4)
for j in range(nx2+1):
    ax_csh.plot([bad_region.get_x()+j*dx2]*2,
                [bad_region.get_y(), bad_region.get_y()+bad_region.get_height()],
                color="#bbbbbb", linewidth=0.4)

ax_csh.set_xlim(-0.5, Wc+0.5)
ax_csh.set_ylim(-0.5, Hc+0.5)

# ============================================================
# 3. RACS：跨尺度交错
# ============================================================
ax_racs.set_title("(c) RACS: repair-aware cross-scale interleaving",
                  fontsize=11, pad=8)

# 上方：低分辨率网格 (2x2)
low_x, low_y = 0.5, 3.5
low_w, low_h = 3.0, 3.0
ax_racs.add_patch(Rectangle((low_x, low_y), low_w, low_h,
                            fill=False, linewidth=1.0, edgecolor="black"))
ax_racs.text(low_x+low_w/2, low_y+low_h+0.2,
             "low-res grid (guided by D)", fontsize=8,
             ha="center", va="bottom")

for i in range(3):
    ax_racs.plot([low_x, low_x+low_w],
                 [low_y+i*low_h/2, low_y+i*low_h/2],
                 color="#999999", linewidth=0.8)
    ax_racs.plot([low_x+i*low_w/2, low_x+i*low_w/2],
                 [low_y, low_y+low_h],
                 color="#999999", linewidth=0.8)

# 用深色表示“坏处优先”格子
ax_racs.add_patch(Rectangle((low_x+low_w/2, low_y+low_h/2),
                            low_w/2, low_h/2,
                            facecolor="#fb6a4a", alpha=0.7, edgecolor="black"))
ax_racs.text(low_x+3*low_w/4, low_y+3*low_h/4,
             "bad", fontsize=7, ha="center", va="center", color="white")

# 下方：高分辨率网格 (4x4) + 蛇形路径
high_x, high_y = 0.5, 0.0
high_w, high_h = 4.0, 3.0
ax_racs.add_patch(Rectangle((high_x, high_y), high_w, high_h,
                            fill=False, linewidth=1.0, edgecolor="black"))
ax_racs.text(high_x+high_w/2, high_y+high_h+0.2,
             "full-res snake path", fontsize=8,
             ha="center", va="bottom")

for i in range(5):
    ax_racs.plot([high_x, high_x+high_w],
                 [high_y+i*high_h/4, high_y+i*high_h/4],
                 color="#cccccc", linewidth=0.6)
    ax_racs.plot([high_x+i*high_w/4, high_x+i*high_w/4],
                 [high_y, high_y+high_h],
                 color="#cccccc", linewidth=0.6)

# 绘制蛇形扫描路径
pts = []
for row in range(4):
    ys = high_y + (row+0.5)*high_h/4
    if row % 2 == 0:
        xs = np.linspace(high_x+0.5*high_w/4, high_x+high_w-0.5*high_w/4, 4)
    else:
        xs = np.linspace(high_x+high_w-0.5*high_w/4, high_x+0.5*high_w/4, 4)
    for x0 in xs:
        pts.append((x0, ys))
px, py = zip(*pts)
ax_racs.plot(px, py, color="black", linewidth=1.0)
ax_racs.arrow(px[-2], py[-2],
              px[-1]-px[-2], py[-1]-py[-2],
              head_width=0.12, head_length=0.25,
              fc="black", ec="black", length_includes_head=True)

# 从低分“坏处”格子连到若干高分位置，表示跨尺度插入
src = (low_x+3*low_w/4, low_y+3*low_h/4)
targets = [
    (high_x+3.5*high_w/4, high_y+3.5*high_h/4),
    (high_x+2.5*high_w/4, high_y+2.5*high_h/4),
]
for tx, ty in targets:
    arr = FancyArrowPatch(src, (tx, ty),
                          arrowstyle="-|>", mutation_scale=10,
                          linewidth=1.0, linestyle="dashed",
                          color="#fb6a4a")
    ax_racs.add_patch(arr)

ax_racs.text(high_x+high_w+0.2, (low_y+high_y+high_h)/2,
             "low-res path\ninjected into\nfull-res scan",
             fontsize=8, ha="left", va="center")

ax_racs.set_xlim(-0.5, 7.0)
ax_racs.set_ylim(-0.5, 7.0)

plt.tight_layout()
plt.savefig("dual_scan_keshihua.png", dpi=300, bbox_inches="tight")
