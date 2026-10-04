from figcommon import *
import pandas as pd, json
C = pd.read_pickle("data/e2_curves.pkl"); S = json.load(open("data/e2_stats.json"))
band = np.load("data/e2_randpath.npy"); srs = np.load("data/e2_randsr.npy")
W = (1 + C).cumprod()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]}); fig.subplots_adjust(left=.07, right=.98, bottom=.19, top=.9, wspace=.25)
a1.fill_between(W.index, band[0], band[2], color=LG, alpha=.7, lw=0, label="무작위 10종목 5–95%")
a1.plot(W.index, W["ew"], color=GRY, lw=2.2, label="동일가중 40종목")
a1.plot(W.index, W["hmm_net"], color=BLUE, lw=2.6, label="HMM 상위 10 (비용 차감)")
a1.legend(frameon=False, fontsize=17, loc="upper left"); a1.set_title("누적 성과 (2016.1–2026.9)"); a1.set_ylabel("배")
a2.hist(srs, bins=40, color=LG, edgecolor="white"); 
for v, c, n, yy, ha, dx in [(S["stats"]["hard_gross"]["sr"], BLUE, "HMM 총", .92, "left", .01), (S["stats"]["hard_net"]["sr"], NAVY, "HMM 순", .92, "right", -.01), (S["stats"]["ew"]["sr"], GRY, "동일가중", .72, "left", .01)]:
    a2.axvline(v, color=c, lw=3); a2.text(v + dx, a2.get_ylim()[1] * yy, n, color=c, fontsize=17, fontweight="bold", ha=ha)
a2.set_title("무작위 1,000회 샤프 분포"); a2.set_yticks([]); a2.set_xlabel("연율 샤프비율")
save(fig, "e2_perf.png")
# 편입 빈도 vs 변동성
import warnings; warnings.filterwarnings("ignore")
px = pd.read_csv("data/univ40.csv", index_col=0, parse_dates=True); ret = px.pct_change().iloc[1:]
H = pd.read_pickle("data/e2_hard.pkl"); idx = C.index
rng = np.random.default_rng(0); sc = H.loc[idx] + rng.normal(0, 1e-9, H.loc[idx].shape)
w = (sc.rank(axis=1, ascending=False) <= 10).astype(float)
fr = w.mean().sort_values(ascending=False); vol = ret.loc[idx].std() * np.sqrt(252) * 100
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]}); fig.subplots_adjust(left=.07, right=.98, bottom=.2, top=.9, wspace=.3)
top = fr.head(6); a1.bar(top.index, top.values * 100, color=BLUE); a1.set_title("편입 일수 비율 상위 6종목 (%)")
for i, v in enumerate(top.values): a1.text(i, v * 100 + 1, f"{v*100:.0f}", ha="center", fontsize=17)
a1.tick_params(axis="x", labelsize=16); a1.set_ylim(0, 85)
a2.scatter(vol[fr.index], fr.values * 100, s=80, color=NAVY)
for t, dx in [("NVDA", -10), ("TSLA", -10), ("META", -10)]: a2.annotate(t, (vol[t], fr[t] * 100), xytext=(0, -30), textcoords="offset points", fontsize=16, ha="center")
a2.set_ylim(-3, 82); a2.set_xlim(15, 64)
a2.set_xlabel("연율 변동성 (%)"); a2.set_ylabel("편입 비율 (%)"); a2.set_title(f"변동성과 편입 비율 (상관 {np.corrcoef(vol[fr.index], fr)[0,1]:.2f})")
save(fig, "e2_picks.png")
# 검정력
sa = S["a_sd"]; edge = np.linspace(0.0001, 0.003, 300); N = (2 * sa / edge) ** 2
fig, ax = plt.subplots(figsize=(12, 3.9)); fig.subplots_adjust(left=.1, right=.98, bottom=.22, top=.92)
ax.plot(edge * 100, N, color=BLUE, lw=3); ax.set_yscale("log"); ax.set_xlabel("하루 평균 초과수익 (%)"); ax.set_ylabel("필요 거래일 N")
for e, c, n in [(0.0018, AMB, "영상의 하루 차이 0.18%"), (S["a_mean"], RED, f"10년 재현 실현값 {S['a_mean']*100:.3f}%")]:
    n_ = (2 * sa / e) ** 2; ax.plot(e * 100, n_, "o", ms=13, color=c); ax.text(e * 100 + .006, n_ * 1.25, f"{n}  →  {n_:,.0f}일", color=c, fontsize=18, fontweight="bold")
ax.axhline(1, color=GRY, ls=":", lw=1.5); ax.text(0.29, 1.3, "영상의 표본: 1일", color=GRY, ha="right", fontsize=17)
ax.set_title(f"t ≥ 2에 필요한 표본 (초과수익 일간 표준편차 {sa*100:.2f}% 가정)")
save(fig, "e2_power.png")
