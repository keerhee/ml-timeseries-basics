import numpy as np, pandas as pd, json
px = pd.read_csv("data/univ40.csv", index_col=0, parse_dates=True)
ret = px.pct_change().iloc[1:]
H = pd.read_pickle("data/e2_hard.pkl"); S = pd.read_pickle("data/e2_soft.pkl")
rng = np.random.default_rng(0)
def book(score, delay=0, n=10, tie=True):
    sc = score + (rng.normal(0, 1e-9, score.shape) if tie else 0)
    rk = sc.rank(axis=1, ascending=False); w = (rk <= n).astype(float) / n
    w = w.where(score.notna().all(axis=1), np.nan, axis=0).dropna()
    fwd = ret.shift(-(1 + delay)).loc[w.index]; ok = fwd.notna().all(axis=1)
    w, fwd = w[ok], fwd[ok]; g = (w * fwd).sum(1); to = w.diff().abs().sum(1).fillna(0)
    return g, to, w
def st(g, to=None, c=0.0005):
    net = g - (c * to if to is not None else 0); a = net.mean() * 252; v = net.std() * np.sqrt(252)
    wv = (1 + net).cumprod(); return dict(ret=a, vol=v, sr=a / v, mdd=(wv / wv.cummax() - 1).min(), to=(to.mean() if to is not None else 0))
gH, toH, wH = book(H); gS, toS, _ = book(S); gH1, toH1, _ = book(H, delay=1)
idx = gH.index; ew = ret.shift(-1).loc[idx].mean(1)
out = {"hard_gross": st(gH), "hard_net": st(gH, toH), "soft_net": st(gS.loc[idx], toS.loc[idx]),
       "hard_delay_net": st(gH1, toH1), "ew": st(ew)}
for k, v in out.items(): print(k, {a: round(b, 4) for a, b in v.items()})
print(idx[0].date(), idx[-1].date(), len(idx))
# 무작위 top-10: 1000 경로 (매일 새로 추첨)
F = ret.shift(-1).loc[idx].values; T, N = F.shape; sims = np.empty((1000, T))
for i in range(1000):
    pick = np.argsort(rng.random((T, N)), 1)[:, :10]; sims[i] = F[np.arange(T)[:, None], pick].mean(1)
rs = sims.mean(1) * 252; vs = sims.std(1) * np.sqrt(252); srs = rs / vs
print("random ann ret pct", np.percentile(rs, [5, 50, 95]).round(4), "SR", np.percentile(srs, [5, 50, 95]).round(3))
pct = (srs < out["hard_gross"]["sr"]).mean(); print("HMM gross SR percentile vs random", pct)
# 일간 초과 a = HMM - EW
a = gH - ew; ab, sa = a.mean(), a.std(); t = ab / (sa / np.sqrt(len(a)))
print("daily active mean %", ab * 100, "sd %", sa * 100, "t", t)
Nreq = (2 * sa / 0.0018) ** 2; print("N for 0.18%/day edge:", Nreq)
# 하루 단위 HMM > 무작위 비율
print("days HMM beats EW:", (a > 0).mean())
# 연도별
yr = pd.DataFrame({"HMM": gH - 0.0005 * toH, "EW": ew}).groupby(gH.index.year).apply(lambda d: (1 + d).prod() - 1)
print(yr.round(3))
# 신호 분포: 예측 상태의 평균이 양수인 종목 비율
print("share positive hard score", (H.loc[idx] > 0).mean().mean())
json.dump(dict(stats=out, rand_sr=np.percentile(srs, [5, 50, 95]).tolist(), rand_ret=np.percentile(rs, [5, 50, 95]).tolist(), pct=float(pct),
               a_mean=float(ab), a_sd=float(sa), t=float(t), Nreq=float(Nreq), beat=float((a > 0).mean()),
               yr=yr.reset_index().rename(columns={"index": "year"}).to_dict("list")), open("data/e2_stats.json", "w"), indent=1, default=float)
pd.DataFrame({"hmm_net": gH - 0.0005 * toH, "hmm_gross": gH, "ew": ew}).to_pickle("data/e2_curves.pkl")
np.save("data/e2_randsr.npy", srs); np.save("data/e2_randpath.npy", np.percentile(np.cumprod(1 + sims, 1), [5, 50, 95], axis=0))
