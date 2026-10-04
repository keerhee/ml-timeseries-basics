# SPY 0/1 레짐 전략: HMM vs 통계적 점프모형(JM) vs 매수보유 — 워크포워드, 1일 지연, 편도 10bp
import numpy as np, pandas as pd, warnings, json, logging, sys
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
spy = pd.read_csv("data/SPY.csv", index_col=0, parse_dates=True).iloc[:, 0]
irx = pd.read_csv("data/IRX.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(spy.index).ffill()
ret = spy.pct_change().dropna(); rf = (irx.reindex(ret.index) / 100 / 252).fillna(0)
ex = ret - rf
def ewm_dd(x, hl): return np.sqrt((np.minimum(x, 0) ** 2).ewm(halflife=hl).mean())
feat = pd.DataFrame({"dd10": np.log(ewm_dd(ex, 10)),
                     "so20": ex.ewm(halflife=20).mean() / ewm_dd(ex, 20),
                     "so60": ex.ewm(halflife=60).mean() / ewm_dd(ex, 60)}).iloc[60:]
ex = ex.loc[feat.index]; ret = ret.loc[feat.index]; rf = rf.loc[feat.index]
T = len(ex); WMAX, WMIN, STEP = 3000, 1500, 126
t_start = WMIN

def dp(L, lam):
    T, K = L.shape; V = np.empty((T, K)); B = np.zeros((T, K), int); V[0] = L[0]
    for t in range(1, T):
        p = V[t - 1]; m = p.min(); j = p.argmin()
        for k in range(K):
            stay = p[k]; sw = m + lam if j != k else np.partition(p, 1)[1] + lam
            if stay <= sw: V[t, k] = L[t, k] + stay; B[t, k] = k
            else: V[t, k] = L[t, k] + sw; B[t, k] = j if j != k else np.argsort(p)[1]
    s = np.empty(T, int); s[-1] = V[-1].argmin()
    for t in range(T - 1, 0, -1): s[t - 1] = B[t, s[t]]
    return s, V[-1]

def jm_fit(X, lam, K=2, n_init=6, seed=0):
    rng = np.random.default_rng(seed); best = None
    for it in range(n_init):
        th = X[rng.choice(len(X), K, replace=False)]
        prev = None
        for _ in range(30):
            L = 0.5 * ((X[:, None, :] - th[None]) ** 2).sum(-1)
            s, _ = dp(L, lam)
            if prev is not None and (s == prev).all(): break
            prev = s
            th = np.array([X[s == k].mean(0) if (s == k).any() else th[k] for k in range(K)])
        L = 0.5 * ((X[:, None, :] - th[None]) ** 2).sum(-1)
        obj = L[np.arange(len(X)), s].sum() + lam * (np.diff(s) != 0).sum()
        if best is None or obj < best[0]: best = (obj, th, s)
    return best[1], best[2]

def run_jm(lam):
    sig = np.full(T, np.nan)
    for t0 in range(t_start, T, STEP):
        lo = max(0, t0 - WMAX); F = feat.values[lo:t0]; mu, sd = F.mean(0), F.std(0)
        X = (F - mu) / sd; th, s = jm_fit(X, lam)
        # 강세 = 학습구간 평균 초과수익이 큰 상태
        e = ex.values[lo:t0]; bull = np.argmax([e[s == k].mean() if (s == k).any() else -9 for k in range(2)])
        # 온라인 추론: 학습구간 전방 가치함수를 이어서 갱신
        L = 0.5 * ((X[:, None, :] - th[None]) ** 2).sum(-1)
        V = L[0].copy()
        for t in range(1, len(X)):
            V = L[t] + np.minimum(V, V[::-1] + lam)
        for t in range(t0, min(t0 + STEP, T)):
            x = (feat.values[t] - mu) / sd; l = 0.5 * ((x - th) ** 2).sum(1)
            V = l + np.minimum(V, V[::-1] + lam); sig[t] = 1.0 if V.argmin() == bull else 0.0
    return sig

def run_hmm():
    sig = np.full(T, np.nan); pb = np.full(T, np.nan)
    for t0 in range(t_start, T, STEP):
        lo = max(0, t0 - WMAX); x = ex.values[lo:t0].reshape(-1, 1)
        best = None
        for s in range(3):
            m = GaussianHMM(2, "diag", n_iter=200, random_state=s).fit(x); ll = m.score(x)
            if best is None or ll > best[0]: best = (ll, m)
        m = best[1]; mu = m.means_.ravel(); sd = np.sqrt(m.covars_.ravel()); A = m.transmat_
        bull = np.argmin(sd); a = m.predict_proba(x)[-1]
        for t in range(t0, min(t0 + STEP, T)):
            a = (a @ A) * norm.pdf(ex.values[t], mu, sd); a /= a.sum()
            pb[t] = a[bull]; sig[t] = 1.0 if a[bull] > 0.5 else 0.0
    return sig, pb

def perf(sig, cost=0.001, delay=1):
    s = pd.Series(sig, index=ex.index)
    pos = s.shift(1 + delay)          # t일 종가 신호 → t+1+delay일 수익에 적용
    ok = pos.notna(); pos = pos[ok]
    r = pos * ret[ok] + (1 - pos) * rf[ok] - cost * pos.diff().abs().fillna(0)
    return r, pos

if __name__ == "__main__":
    out = {}
    hs, pb = run_hmm(); out["hmm"] = hs.tolist(); out["pbull"] = pb.tolist()
    for lam in [0.0, 10.0, 50.0, 100.0]:
        out[f"jm{int(lam)}"] = run_jm(lam).tolist(); print("jm", lam, flush=True)
    out["index"] = [str(d.date()) for d in ex.index]
    json.dump(out, open("data/e3_signals.json", "w")); print("ok")
