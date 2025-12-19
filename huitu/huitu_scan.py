import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "DejaVu Sans"   # 按需改成 Times New Roman / 中文字体

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 7)
ax.axis("off")

# --------- 小工具函数 ---------
def box(x, y, w, h, text, fontsize=10,
        fc="#fdfdfd", ec="black", lw=1.2, alpha=1.0):
    rect = Rectangle((x - w/2, y - h/2), w, h,
                     linewidth=lw, edgecolor=ec,
                     facecolor=fc, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return rect

def arrow(x1, y1, x2, y2, text="", fontsize=9,
          style="-|>", connectionstyle="arc3",
          color="black", ls="-", offset_text=(0, 0.18)):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=12,
                          linewidth=1.2, linestyle=ls,
                          connectionstyle=connectionstyle, color=color)
    ax.add_patch(arr)
    if text:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(xm + offset_text[0], ym + offset_text[1],
                text, ha="center", va="bottom", fontsize=fontsize)
    return arr

def draw_grid(x, y, size=1.4, n=3):
    colors = ["#fee0d2", "#deebf7", "#e5f5e0", "#fdd0a2"]
    cell = size / n
    start_x = x - size/2
    start_y = y - size/2
    k = 1
    for i in range(n):
        for j in range(n):
            c = colors[(i + j) % len(colors)]
            rect = Rectangle((start_x + j*cell, start_y + i*cell),
                             cell, cell, facecolor=c,
                             edgecolor="black", linewidth=0.7)
            ax.add_patch(rect)
            ax.text(start_x + j*cell + cell/2,
                    start_y + i*cell + cell/2,
                    str(k), ha="center", va="center", fontsize=8)
            k += 1

# --------- 标题 & 外框 ---------
ax.text(7, 6.4, "Dual-Scan Guided Neighboring (CSH + RACS)",
        ha="center", va="center", fontsize=16, fontweight="bold")

outer = Rectangle((-0.5, 0.2), 15.0, 6.0,
                  linewidth=1.2, edgecolor="#b0b0b0",
                  facecolor="#f7fcff", zorder=-1)
ax.add_patch(outer)

# --------- 左侧：输入特征 ----------
draw_grid(x=1.0, y=3.0, size=1.6, n=3)
ax.text(1.0, 1.7, "Input feature map",
        ha="center", va="center", fontsize=9)

# --------- 上方：引导信号 θ, D ----------
theta_box = box(x=3.3, y=5.3, w=2.6, h=0.9,
                text=r"Direction field $\theta$",
                fontsize=10, fc="#e6f2ff")
D_box = box(x=6.4, y=5.3, w=2.6, h=0.9,
            text=r"Degradation map $D$",
            fontsize=10, fc="#fff3d6")

ax.text(4.8, 4.7, "shared guidance",
        ha="center", va="center", fontsize=9, color="gray")

# --------- 中部：CSH / RACS 两个子块 ----------
csh_box = box(x=5.0, y=4.0, w=4.0, h=1.6,
              text="CSH stripe +\nHilbert sub-blocks",
              fontsize=10, fc="#f5f3ff")
ax.text(5.0, 5.1, "CSH-unfold",
        ha="center", va="center", fontsize=9)

racs_box = box(x=5.0, y=2.0, w=4.0, h=1.6,
               text="RACS cross-scale\n(low-res ↔ full-res)",
               fontsize=10, fc="#f2fff2")
ax.text(5.0, 3.1, "RACS-unfold",
        ha="center", va="center", fontsize=9)

# 输入 -> CSH / RACS
arrow(1.8, 3.2, 3.0, 4.0)
arrow(1.8, 2.8, 3.0, 2.0)

# θ, D -> CSH/RACS：虚线箭头
arrow(3.3, 4.9, 3.9, 4.4, ls="dashed", color="gray")
arrow(6.4, 4.9, 6.1, 4.4, ls="dashed", color="gray")

arrow(3.3, 4.9, 3.9, 2.6, ls="dashed", color="gray")
arrow(6.4, 4.9, 6.1, 2.6, ls="dashed", color="gray")

# --------- 融合节点 ----------
fusion = Circle((7.5, 3.0), radius=0.3,
                edgecolor="black", facecolor="#f0f0f0")
ax.add_patch(fusion)
ax.text(7.5, 3.0, "⊕", ha="center", va="center", fontsize=13)
ax.text(7.5, 1.8, "Fusion\n(channel / temporal)",
        ha="center", va="center", fontsize=9)

arrow(7.0, 4.0, 7.3, 3.25)  # CSH -> fusion
arrow(7.0, 2.0, 7.3, 2.75)  # RACS -> fusion

# --------- ASE / Mamba 核心 ----------
ase_box = box(x=9.5, y=3.0, w=1.4, h=3.0,
              text="ASE /\nMamba", fontsize=11)
arrow(7.8, 3.0, 8.8, 3.0,
      text="dual-scan sequence", fontsize=9,
      offset_text=(0, 0.25))

# --------- Dual-Scan-fold & 输出 ----------
arrow(10.2, 3.0, 11.4, 3.0,
      text="Dual-Scan-fold", fontsize=9,
      offset_text=(0, 0.25))

draw_grid(x=12.7, y=3.0, size=1.6, n=3)
ax.text(12.7, 1.7, "Output feature map",
        ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("dual_scan_structure.png", dpi=300, bbox_inches="tight")
