import engine as E; from engine import *
spy,_=run(w_spy,'M'); rf=RF.reindex(spy.index)
def m(r):
    y=len(r)/252; ann=(1+r).prod()**(1/y)-1; sh=(r-rf).mean()/r.std()*np.sqrt(252); w=(1+r).cumprod(); return round(ann*100,1),round(sh,2),round((w/w.cummax()-1).min()*100,1)
rows=[]
# cost sensitivity original
for c in [0,0.0005,0.0010,0.0020]:
    r,_=run(w_orig,'W',cost=c); rows.append(('원 전략 비용 %dbp'%(c*1e4),)+m(r))
for c in [0,0.0020]:
    r,_=run(w_s3,'M',cost=c); rows.append(('③ 비용 %dbp'%(c*1e4),)+m(r))
# K sensitivity for S2
orig=E.comps_hclust
for K in [3,5,6]:
    E.comps_hclust=lambda C,nm,k,K=K: orig(C,nm,K)
    r,_=run(E.w_s2,'M'); rows.append((f'② 군집 K={K}',)+m(r))
E.comps_hclust=orig
# core weight
for cw in [0.5,0.8]:
    f=lambda t,cw=cw: (lambda s:(s.__setitem__('SPY',s.get('SPY',0)+cw),s)[1])(E.w_s2(t)*(1-cw))
    r,_=run(f,'M'); rows.append((f'코어 {int(cw*100)}/{int((1-cw)*100)}',)+m(r))
# shrink lambda
def mk(lam):
    def sc(t,nm):
        win=R.iloc[t-252:t-21][nm]; sr=win.sub(RF.iloc[t-252:t-21],axis=0).mean()/win.std()*np.sqrt(252); return lam*sr+(1-lam)*sr.mean()
    return sc
o=E.score_mom
for lam in [1.0,0.25]:
    E.score_mom=mk(lam); r,_=run(E.w_s2,'M'); rows.append((f'② 수축 λ={lam}',)+m(r))
E.score_mom=o
rows.append(('SPY',)+m(spy))
for x in rows: print(x)
