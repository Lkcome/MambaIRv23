import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"  # 按需改字体

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax_csh, ax_shift = axes
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# ============================================================
# (b-1) CSH: curved stripes + Hilbert sub-blocks
# ============================================================
ax_csh.set_title("(b-1) CSH: curved stripes + Hilbert sub-blocks",
                 fontsize=11, pad=8)

Wc, Hc = 10, 6

# 3 条水平条带背景（像你第一版那样）
stripe_colors = ["#e6f2ff", "#fff5e6", "#e6ffe6"]
stripe_h = Hc / 3
for i in range(3):
    ax_csh.add_patch(Rectangle((0, i*stripe_h), Wc, stripe_h,
                               facecolor=stripe_colors[i],
                               edgecolor="none", alpha=0.9))

# 外边框
ax_csh.add_patch(Rectangle((0, 0), Wc, Hc,
                           fill=False, linewidth=1.2, edgecolor="black"))

# 弯曲扫描路径（中心在中间条带）
xs = np.linspace(0.5, Wc-0.5, 200)
ys = Hc/2 + 0.8*np.sin(2*np.pi*xs/Wc)
ax_csh.plot(xs, ys, color="black", linewidth=2.0)
ax_csh.arrow(xs[-3], ys[-3],
             xs[-1]-xs[-3], ys[-1]-ys[-3],
             head_width=0.22, head_length=0.45,
             fc="black", ec="black", length_includes_head=True)

ax_csh.text(0.4, ys[0]+0.9,
            "curved stripe\naligned with $\\theta$",
            fontsize=9, ha="left", va="bottom")

# ---- Hilbert sub-block（贴在中间）----
sub_w, sub_h = 2.6, 1.6
sub_x = 2.2
# 取曲线在 x≈3 的 y 作为中心
mid_idx = np.argmin(np.abs(xs - (sub_x + sub_w/2)))
sub_y = ys[mid_idx] - sub_h/2

ax_csh.add_patch(Rectangle((sub_x, sub_y), sub_w, sub_h,
                           facecolor="white", edgecolor="black", linewidth=1.0))
ax_csh.text(sub_x+sub_w/2, sub_y+sub_h+0.1,
            "Hilbert sub-block", fontsize=9,
            ha="center", va="bottom")

# 子块网格 + Hilbert path
nx, ny = 4, 2
dx, dy = sub_w/nx, sub_h/ny
for i in range(ny+1):
    ax_csh.plot([sub_x, sub_x+sub_w],
                [sub_y+i*dy, sub_y+i*dy],
                color="#b0b0b0", linewidth=0.6)
for j in range(nx+1):
    ax_csh.plot([sub_x+j*dx, sub_x+j*dx],
                [sub_y, sub_y+sub_h],
                color="#b0b0b0", linewidth=0.6)

hilbert_pts = [
    (sub_x+0.5*dx, sub_y+0.5*dy),
    (sub_x+0.5*dx, sub_y+1.5*dy),
    (sub_x+1.5*dx, sub_y+1.5*dy),
    (sub_x+1.5*dx, sub_y+0.5*dy),
    (sub_x+2.5*dx, sub_y+0.5*dy),
    (sub_x+2.5*dx, sub_y+1.5*dy),
    (sub_x+3.5*dx, sub_y+1.5*dy),
    (sub_x+3.5*dx, sub_y+0.5*dy),
]
hx, hy = zip(*hilbert_pts)
ax_csh.plot(hx, hy, color="black", linewidth=1.0)

# ---- “坏处加密”区域：尾部更密的网格 ----
bad_w, bad_h = 2.4, stripe_h*0.9
bad_x = Wc - bad_w - 0.6
bad_idx = np.argmin(np.abs(xs - (bad_x + bad_w/2)))
bad_y = ys[bad_idx] - bad_h/2

ax_csh.add_patch(Rectangle((bad_x, bad_y), bad_w, bad_h,
                           fill=False, edgecolor="red",
                           linewidth=1.0, linestyle="--"))
ax_csh.text(bad_x+bad_w/2, bad_y+bad_h+0.1,
            "denser blocks\nin degraded region",
            fontsize=9, ha="center", va="bottom", color="red")

nx2, ny2 = 6, 2
dx2, dy2 = bad_w/nx2, bad_h/ny2
for i in range(ny2+1):
    ax_csh.plot([bad_x, bad_x+bad_w],
                [bad_y+i*dy2, bad_y+i*dy2],
                color="#d0d0d0", linewidth=0.5)
for j in range(nx2+1):
    ax_csh.plot([bad_x+j*dx2, bad_x+j*dx2],
                [bad_y, bad_y+bad_h],
                color="#d0d0d0", linewidth=0.5)

ax_csh.set_xlim(-0.3, Wc+0.3)
ax_csh.set_ylim(-0.3, Hc+0.3)

# ============================================================
# (b-2) Shift-stripe across Module i / i+1
# ============================================================
ax_shift.set_title("(b-2) Shift-stripe across depth (Module $i$ and $i{+}1$)",
                   fontsize=11, pad=8)

Wm, Hm = 6, 5     # 每个模块框大小
gap_y = 0.8       # 两个模块之间的竖向间距

# Module i 框
mod1_y = gap_y + Hm
ax_shift.add_patch(Rectangle((0, mod1_y), Wm, Hm,
                             fill=False, linewidth=1.2, edgecolor="black"))
ax_shift.text(Wm/2, mod1_y+Hm+0.2, "Module $i$",
              fontsize=9, ha="center", va="bottom")

# Module i+1 框
mod2_y = 0
ax_shift.add_patch(Rectangle((0, mod2_y), Wm, Hm,
                             fill=False, linewidth=1.2, edgecolor="black"))
ax_shift.text(Wm/2, mod2_y+Hm+0.2, "Module $i{+}1$",
              fontsize=9, ha="center", va="bottom")

# 在两个模块内分别画条带（第二个模块整体向右平移半条带宽）
num_stripes = 3
stripe_h2 = Hm / num_stripes
shift_x = Wm / (num_stripes*2)   # 水平平移量

colors2 = ["#e6f2ff", "#fff5e6", "#e6ffe6"]

# Module i：不平移
for k in range(num_stripes):
    y0 = mod1_y + k*stripe_h2
    ax_shift.add_patch(Rectangle((0, y0), Wm, stripe_h2,
                                 facecolor=colors2[k], edgecolor="none", alpha=0.9))

# Module i+1：整体向右 shift
for k in range(num_stripes):
    y0 = mod2_y + k*stripe_h2
    # 左边留一点空白，模拟“移位后覆盖另一部分区域”
    ax_shift.add_patch(Rectangle((shift_x, y0), Wm, stripe_h2,
                                 facecolor=colors2[k], edgecolor="none", alpha=0.9))

# 画两条示意性的 S-shaped path，显示条带内/条带间都连续
def draw_s_path(y_base, color="black"):
    xs_ = np.linspace(0.4, Wm-0.4, 80)
    ys_ = y_base + Hm/2 + 0.7*np.sin(2*np.pi*xs_/Wm)
    ax_shift.plot(xs_, ys_, color=color, linewidth=1.4)

draw_s_path(mod1_y, color="black")
draw_s_path(mod2_y, color="black")

# 模块之间的箭头，说明“shift-stripe across depth”
arr = FancyArrowPatch((Wm+0.3, mod1_y+Hm/2),
                      (Wm+0.3, mod2_y+Hm/2),
                      arrowstyle="-|>", mutation_scale=14,
                      linewidth=1.4, color="black")
ax_shift.add_patch(arr)
ax_shift.text(Wm+0.45, (mod1_y+Hm/2 + mod2_y+Hm/2)/2,
              "shifted stripes\nenhance inter-stripe\ninteraction",
              fontsize=8, ha="left", va="center")

ax_shift.set_xlim(-0.3, Wm+2.5)
ax_shift.set_ylim(-0.3, 2*Hm+gap_y+0.7)

plt.tight_layout()
plt.savefig("csh.png", dpi=300, bbox_inches="tight")
