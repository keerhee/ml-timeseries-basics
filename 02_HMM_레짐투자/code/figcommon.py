import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 22, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#94A3B8", "axes.labelcolor": "#1E293B",
    "xtick.color": "#475569", "ytick.color": "#475569", "axes.unicode_minus": False, "axes.titlesize": 22})
INK, BLUE, NAVY, GRN, RED, AMB, GRY, LG = "#1E293B", "#2563EB", "#1E4E8C", "#2E7D4F", "#C0392B", "#B7791F", "#64748B", "#CBD5E1"
CM = LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#BFDBFE", BLUE])
def save(fig, name):
    fig.savefig("fig/" + name, dpi=150, facecolor="white"); plt.close(fig)
def heat(ax, A, labels, title, fs=22):
    ax.imshow(A, cmap=CM, vmin=0, vmax=1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=fs, color="white" if A[i, j] > .6 else INK)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title); [s.set_visible(False) for s in ax.spines.values()]
