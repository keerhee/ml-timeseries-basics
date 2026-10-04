import numpy as np, pandas as pd, warnings, logging, json
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
out = {}
for tk in ["SPY", "TSLA"]:
    px = pd.read_csv(f"data/{tk}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    x = np.log(px).diff().dropna().values.reshape(-1, 1); n = len(x); rows = []
    for K in range(1, 6):
        ll = max(GaussianHMM(K, "diag", n_iter=300, random_state=s).fit(x).score(x) for s in range(3))
        p = (K - 1) + K * (K - 1) + 2 * K
        rows.append(dict(K=K, p=p, ll=ll, bic=-2 * ll + p * np.log(n), aic=-2 * ll + 2 * p))
    out[tk] = rows; print(tk, n); [print(r) for r in rows]
json.dump(out, open("data/e4_bic.json", "w"))
