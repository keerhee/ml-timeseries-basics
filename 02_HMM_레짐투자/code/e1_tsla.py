import numpy as np, pandas as pd, warnings, json
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore")
px = pd.read_csv("data/TSLA.csv", index_col=0, parse_dates=True).iloc[:,0]
r = np.log(px).diff().dropna()
def fit(x, seed, K=3):
    m = GaussianHMM(K, covariance_type="diag", n_iter=1000, tol=1e-6, random_state=seed).fit(x.reshape(-1,1))
    o = np.argsort(m.means_.ravel())[::-1]  # H(최대 평균) → S
    return m.score(x.reshape(-1,1)), m.means_.ravel()[o], np.sqrt(m.covars_.ravel()[o]), m.transmat_[np.ix_(o,o)]
out = {}
# 전체 표본
res = [fit(r.values, s) for s in range(10)]
best = max(res, key=lambda z: z[0])
print("FULL", r.index[0].date(), r.index[-1].date(), len(r))
print("mu", (best[1]*100).round(3), "sd", (best[2]*100).round(2)); print(best[3].round(3))
out["full"] = dict(ll=best[0], mu=best[1].tolist(), sd=best[2].tolist(), A=best[3].tolist(), n=len(r))
# 최근 1년(250일) 창 — 초기값 20개
x = r.values[-250:]
res = [fit(x, s) for s in range(20)]
lls = np.array([z[0] for z in res])
print("1Y lls", np.round(np.sort(lls)[::-1], 1))
maxdiag = [z[3][np.arange(3), np.arange(3)].round(2) for z in res]
rowmax = [tuple(np.argmax(z[3], 1)) for z in res]
from collections import Counter; print(Counter(rowmax))
for z in sorted(res, key=lambda z: -z[0])[:4]:
    print(round(z[0],1), (z[1]*100).round(2), (z[2]*100).round(2)); print(z[3].round(2))
out["y1"] = [dict(ll=z[0], mu=z[1].tolist(), sd=z[2].tolist(), A=z[3].tolist()) for z in res]
json.dump(out, open("data/e1.json","w"))
