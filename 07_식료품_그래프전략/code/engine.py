import pandas as pd, numpy as np, networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
U=['KR','ACI','SFM','TGT','WMT','COST','NGVC','BJ','GO','WMK','IMKTA','VLGEA']
px=pd.read_csv('data/prices.csv',index_col=0,parse_dates=True)
irx=pd.read_csv('data/irx.csv',index_col=0,parse_dates=True).squeeze()
R=px.pct_change().iloc[1:]; RF=(irx.reindex(R.index).ffill()/100/252)
START='2019-01-01'; COST=0.0010
idx=R.index; t0=idx.searchsorted(pd.Timestamp(START))
def avail(win): return [u for u in U if win[u].notna().all()]
def rebal_dates(freq):
    if freq=='W': return list(range(t0,len(idx),5))
    m=pd.Series(idx[t0:]).dt.to_period('M'); first=m.ne(m.shift()).values
    return [t0+i for i in np.where(first)[0]]
def comps_graph(C,names,tau):
    G=nx.Graph(); G.add_nodes_from(names)
    G.add_edges_from((a,b) for i,a in enumerate(names) for b in names[i+1:] if C.loc[a,b]>tau)
    return [list(c) for c in nx.connected_components(G)]
def comps_hclust(C,names,K):
    D=np.sqrt(np.clip(0.5*(1-C.values),0,None)); np.fill_diagonal(D,0)
    Z=linkage(squareform(D,checks=False),'average'); lab=fcluster(Z,K,'maxclust')
    return [[n for n,l in zip(names,lab) if l==k] for k in set(lab)]
# ---- weight rules -------------------------------------------------
def w_orig(t):
    win=R.iloc[t-21:t]; nm=avail(win); w=win[nm]
    sr=w.sub(RF.iloc[t-21:t],axis=0).mean()/w.std()
    p={}
    for c in comps_graph(w.corr(),nm,0.5):
        b=max(c,key=lambda x:sr[x])
        if sr[b]>=0.25: p[b]=sr[b]
    return pd.Series(p)/sum(p.values()) if p else pd.Series({'SPY':1.0})
def score_mom(t,nm):
    win=R.iloc[t-252:t-21][nm]; ex=win.sub(RF.iloc[t-252:t-21],axis=0)
    sr=ex.mean()/win.std()*np.sqrt(252)            # 12-1 annualized Sharpe
    return 0.5*sr+0.5*sr.mean()                      # shrink toward cross-section mean
def w_s1(t):   # better signal, monthly, same graph grouping, SR-proportional
    nm=avail(R.iloc[t-252:t]); sc=score_mom(t,nm); C=R.iloc[t-126:t][nm].corr()
    p={}
    for c in comps_graph(C,nm,0.5):
        b=max(c,key=lambda x:sc[x])
        if sc[b]>0: p[b]=sc[b]
    return pd.Series(p)/sum(p.values()) if p else pd.Series({'SPY':1.0})
def w_s2(t):   # + hierarchical clusters K=4 + inverse-vol weights
    nm=avail(R.iloc[t-252:t]); sc=score_mom(t,nm); C=R.iloc[t-126:t][nm].corr()
    vol=R.iloc[t-63:t][nm].std(); p={}
    for c in comps_hclust(C,nm,4):
        b=max(c,key=lambda x:sc[x])
        if sc[b]>0: p[b]=1/vol[b]
    return pd.Series(p)/sum(p.values()) if p else pd.Series({'SPY':1.0})
def w_s3(t):   # core-satellite 70% SPY + 30% S2
    s=w_s2(t)*0.3; s['SPY']=s.get('SPY',0)+0.7; return s
def w_ew(t):
    nm=avail(R.iloc[t-21:t]); return pd.Series(1/len(nm),index=nm)
def w_spy(t): return pd.Series({'SPY':1.0})
def w_xlp(t): return pd.Series({'XLP':1.0})
def run(rule,freq,cost=COST):
    rd=rebal_dates(freq)+[len(idx)]; out=[]; wprev=pd.Series(dtype=float); log=[]
    for a,b in zip(rd[:-1],rd[1:]):
        w=rule(a); allk=w.index.union(wprev.index)
        to=(w.reindex(allk,fill_value=0)-wprev.reindex(allk,fill_value=0)).abs().sum()
        seg=R.iloc[a:b][w.index].fillna(0)
        val=(1+seg).cumprod()*w; port=val.sum(axis=1); dr=port.pct_change(); dr.iloc[0]=port.iloc[0]-1
        dr.iloc[0]-=to*cost; out.append(dr)
        wprev=val.iloc[-1]/val.iloc[-1].sum(); log.append((idx[a],w,to))
    return pd.concat(out),log
