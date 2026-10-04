from figcommon import *
import json
E = json.load(open("data/e1.json")); F = E["full"]
mu, sd, A = np.array(F["mu"]), np.array(F["sd"]), np.array(F["A"])
o = np.argsort(sd); mu, sd, A = mu[o], sd[o], A[np.ix_(o, o)]   # 저변동 → 고변동
lab = ["저변동", "중변동", "고변동"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
fig.subplots_adjust(left=.1, right=.98, bottom=.2, top=.88, wspace=.35)
heat(a1, A, ["저", "중", "고"], "전이행렬 (오늘 행 → 내일 열)")
from scipy.stats import norm
x = np.linspace(-.25, .25, 800); cols = [GRN, AMB, RED]
for k in range(3):
    a2.plot(x * 100, norm.pdf(x, mu[k], sd[k]), color=cols[k], lw=3, label=f"{lab[k]}  μ={mu[k]*100:+.2f}%  σ={sd[k]*100:.1f}%")
a2.set_xlim(-20, 20); a2.set_xlabel("일간 로그수익률 (%)"); a2.set_yticks([]); a2.legend(frameon=False, fontsize=18, loc="upper right")
a2.set_title("상태별 수익률 분포")
save(fig, "e1_tsla_full.png")
dur = 1 / (1 - np.diag(A)); print("dur", dur.round(1), "mu", (mu*100).round(3), "sd", (sd*100).round(2))
w, v = np.linalg.eig(A.T); p = np.real(v[:, np.argmax(np.real(w))]); p /= p.sum(); print("pi", p.round(3), "eig", np.round(w, 3))
# 1년 창 초기값 민감도
Y = sorted(E["y1"], key=lambda z: -z["ll"])
pick = [Y[0], Y[1], Y[2], Y[9]]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.9)); fig.subplots_adjust(left=.05, right=.99, bottom=.08, top=.8, wspace=.35)
for ax, z in zip(axs, pick):
    heat(ax, np.array(z["A"]), ["H", "N", "S"], f"로그우도 {z['ll']:.1f}", fs=19)
fig.suptitle("같은 250일 자료, 초기값만 다른 EM 추정 결과 (평균 순으로 H·N·S 정렬)", fontsize=20, color=INK)
save(fig, "e1_tsla_1y.png")
print([round(z["ll"], 1) for z in Y])
