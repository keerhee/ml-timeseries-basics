# 영상 전략의 워크포워드 재현: 40개 대형주, 3-상태 HMM, 월 1회 재적합, 매일 전방 필터
import numpy as np, pandas as pd, warnings, json, logging
from hmmlearn.hmm import GaussianHMM
from multiprocessing import Pool
from scipy.stats import norm
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
px = pd.read_csv("data/univ40.csv", index_col=0, parse_dates=True)
R = np.log(px).diff().iloc[1:]
W, STEP, K = 750, 21, 3
start = R.index.get_loc(R.loc["2016-01-01":].index[0])

def run(tk):
    r = R[tk].values; T = len(r)
    hard = np.full(T, np.nan); soft = np.full(T, np.nan)
    for t0 in range(start, T, STEP):
        x = r[t0 - W:t0].reshape(-1, 1)
        m = GaussianHMM(K, "diag", n_iter=100, tol=1e-4, init_params="", params="stmc")
        m.startprob_ = np.ones(K) / K; m.transmat_ = np.full((K, K), .05) + np.eye(K) * (1 - .05 * K)
        m.means_ = np.quantile(x, [.1, .5, .9]).reshape(-1, 1); m.covars_ = np.full((K, 1), x.var())
        m.fit(x); mu = m.means_.ravel(); sd = np.sqrt(m.covars_.ravel()); A = m.transmat_
        a = m.predict_proba(x)[-1]          # 창 끝의 확률 = 과거 정보만 사용한 필터 확률
        for t in range(t0, min(t0 + STEP, T)):
            # t일 종가 이후: r[t] 관측 반영(전방 필터) → t+1 예측
            a = (a @ A) * norm.pdf(r[t], mu, sd); a = a / a.sum()
            s_now = np.argmax(a); s_next = np.argmax(A[s_now])
            hard[t] = mu[s_next]; soft[t] = (a @ A) @ mu
    return tk, hard, soft

if __name__ == "__main__":
    res = []
    for i, tk in enumerate(R.columns):
        res.append(run(tk)); print(i, tk, flush=True)
    H = pd.DataFrame({k: h for k, h, s in res}, index=R.index); S = pd.DataFrame({k: s for k, h, s in res}, index=R.index)
    H.to_pickle("data/e2_hard.pkl"); S.to_pickle("data/e2_soft.pkl"); print("done")
