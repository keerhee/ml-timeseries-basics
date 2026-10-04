import numpy as np, pandas as pd, json
import importlib.util, sys
spec = importlib.util.spec_from_file_location("e3", "e3_regime.py"); e3 = importlib.util.module_from_spec(spec); sys.modules["e3"] = e3
src = open("e3_regime.py").read().split('if __name__')[0]; exec(src, e3.__dict__)
S = json.load(open("data/e3_signals.json")); idx = pd.to_datetime(S["index"])
def stats(r, pos=None):
    r = r.dropna(); ann = (1 + r).prod() ** (252 / len(r)) - 1; vol = r.std() * np.sqrt(252)
    exr = r - e3.rf.loc[r.index]; sr = exr.mean() / exr.std() * np.sqrt(252)
    w = (1 + r).cumprod(); mdd = (w / w.cummax() - 1).min()
    d = dict(ret=ann, vol=vol, sr=sr, mdd=mdd)
    if pos is not None:
        ch = pos.diff().abs().fillna(0); yrs = len(pos) / 252
        d.update(sw=ch.sum() / yrs, inv=pos.mean())
    return d
res = {}; curves = {}
ref = None
for k in ["hmm", "jm0", "jm10", "jm50", "jm100"]:
    r, pos = e3.perf(np.array(S[k], dtype=float)); res[k] = stats(r, pos); curves[k] = r
    ref = r.index if ref is None else ref.intersection(r.index)
bh = e3.ret.loc[ref]; res["bh"] = stats(bh); curves["bh"] = bh
# 동일 기간으로 재계산
for k in list(curves):
    r = curves[k].loc[ref]
    if k != "bh":
        _, pos = e3.perf(np.array(S[k], dtype=float)); pos = pos.loc[ref]; res[k] = stats(r, pos)
    else: res[k] = stats(r)
print(ref[0].date(), ref[-1].date(), len(ref))
df = pd.DataFrame(res).T; print(df.round(3))
# 무비용 비교
for k in ["hmm", "jm50"]:
    r, pos = e3.perf(np.array(S[k], dtype=float), cost=0); print(k, "nocost", {a: round(b, 3) for a, b in stats(r.loc[ref], pos.loc[ref]).items()})
    r, pos = e3.perf(np.array(S[k], dtype=float), delay=0); print(k, "nodelay", {a: round(b, 3) for a, b in stats(r.loc[ref], pos.loc[ref]).items()})
pd.DataFrame({k: v.loc[ref] for k, v in curves.items()}).to_pickle("data/e3_curves.pkl")
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in res.items()}, open("data/e3_stats.json", "w"), indent=1)
pb = np.array(S["pbull"], dtype=float)
r, pos = e3.perf(pb); st = stats(r.loc[ref], pos.loc[ref]); print("hmm_soft", {a: round(b, 3) for a, b in st.items()})
res["hmm_soft"] = st; curves["hmm_soft"] = r
# 히스테리시스: 0.3/0.7 밴드
h = np.full(len(pb), np.nan); cur = 1.0
for i, p in enumerate(pb):
    if np.isnan(p): continue
    if p > 0.7: cur = 1.0
    elif p < 0.3: cur = 0.0
    h[i] = cur
r, pos = e3.perf(h); st = stats(r.loc[ref], pos.loc[ref]); print("hmm_band", {a: round(b, 3) for a, b in st.items()})
res["hmm_band"] = st; curves["hmm_band"] = r
pd.DataFrame({k: v.loc[ref] for k, v in curves.items()}).to_pickle("data/e3_curves.pkl")
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in res.items()}, open("data/e3_stats.json", "w"), indent=1)
# 연도별: 2008, 2020, 2022 구간
C = pd.DataFrame({k: v.loc[ref] for k, v in curves.items()})
for a, b in [("2007-10-09", "2009-03-09"), ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")]:
    print(a, b, ((1 + C.loc[a:b]).prod() - 1).round(3)[["bh", "hmm", "jm50", "hmm_band"]].to_dict())
