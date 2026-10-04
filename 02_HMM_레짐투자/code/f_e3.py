from figcommon import *
import pandas as pd, json
C = pd.read_pickle("data/e3_curves.pkl"); ST = json.load(open("data/e3_stats.json"))
S = json.load(open("data/e3_signals.json")); idx = pd.to_datetime(S["index"])
W = (1 + C).cumprod()
# 1) 누적 + 낙폭
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
fig.subplots_adjust(left=.09, right=.98, bottom=.1, top=.97, hspace=.1)
for k, c, n in [("bh", GRY, "매수보유"), ("hmm", BLUE, "HMM 0/1"), ("jm50", GRN, "점프모형 λ=50")]:
    a1.plot(W.index, W[k], color=c, lw=2.4, label=n)
    a2.plot(W.index, (W[k] / W[k].cummax() - 1) * 100, color=c, lw=1.8)
a1.set_yscale("log"); a1.set_ylabel("누적 (로그)"); a1.legend(frameon=False, fontsize=18, ncol=3, loc="upper left")
a2.set_ylabel("낙폭 (%)")
save(fig, "e3_equity.png")
# 2) 레짐 음영
spy = pd.read_csv("data/SPY.csv", index_col=0, parse_dates=True).iloc[:, 0]
fig, axs = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True); fig.subplots_adjust(left=.07, right=.98, bottom=.1, top=.93, hspace=.25)
for ax, k, n in [(axs[0], "hmm", f"HMM — 연 {ST['hmm']['sw']:.1f}회 전환"), (axs[1], "jm50", f"점프모형 λ=50 — 연 {ST['jm50']['sw']:.1f}회 전환")]:
    s = pd.Series(S[k], index=idx).loc["2006-03-23":]; p = spy.reindex(s.index)
    ax.plot(p.index, p, color=INK, lw=1.2); ax.set_yscale("log"); ax.set_yticks([])
    bear = (s == 0).values; y0, y1 = p.min() * .9, p.max() * 1.1
    ax.fill_between(s.index, y0, y1, where=bear, color=RED, alpha=.18, lw=0); ax.set_ylim(y0, y1)
    ax.set_title(n, loc="left", fontsize=20)
save(fig, "e3_shading.png")
# 3) λ 대 전환횟수·샤프
ks = ["jm0", "jm10", "jm50", "jm100"]; lam = [0, 10, 50, 100]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8)); fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.86, wspace=.3)
a1.bar([str(l) for l in lam], [ST[k]["sw"] for k in ks], color=BLUE); a1.axhline(ST["hmm"]["sw"], color=RED, ls="--", lw=2)
a1.text(3.45, ST["hmm"]["sw"] + .4, "HMM", color=RED, ha="right", fontsize=18); a1.set_title("연간 레짐 전환 횟수"); a1.set_xlabel("점프 벌점 λ")
a2.bar([str(l) for l in lam], [ST[k]["sr"] for k in ks], color=NAVY); a2.axhline(ST["bh"]["sr"], color=GRY, ls="--", lw=2)
a2.set_ylim(0, 0.72); a2.text(-0.4, ST["bh"]["sr"] + .03, "매수보유", color=GRY, ha="left", fontsize=18); a2.set_title("샤프비율 (비용·지연 반영)"); a2.set_xlabel("점프 벌점 λ")
for a, key in [(a1, "sw"), (a2, "sr")]:
    for i, k in enumerate(ks): a.text(i, ST[k][key], f"{ST[k][key]:.2f}" if key == "sr" else f"{ST[k][key]:.1f}", ha="center", va="bottom", fontsize=18)
save(fig, "e3_lambda.png")
