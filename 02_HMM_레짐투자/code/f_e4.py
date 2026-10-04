from figcommon import *
import json
B = json.load(open("data/e4_bic.json"))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8)); fig.subplots_adjust(left=.1, right=.98, bottom=.22, top=.86, wspace=.35)
for ax, tk in zip(axs, ["SPY", "TSLA"]):
    rows = B[tk]; K = [r["K"] for r in rows]; b = np.array([r["bic"] for r in rows]); b = b - b.min()
    ax.plot(K, b, "-o", color=BLUE, lw=3, ms=10); k0 = int(np.argmin(b)); ax.plot(K[k0], 0, "o", ms=18, mfc="none", mec=RED, mew=3)
    ax.set_title(f"{tk} 일간수익률 — BIC 최소 K={K[k0]}"); ax.set_xlabel("상태 수 K"); ax.set_ylabel("ΔBIC"); ax.set_xticks(K)
    ax.set_ylim(-0.08 * b.max(), b.max() * 1.1)
save(fig, "e4_bic.png")
