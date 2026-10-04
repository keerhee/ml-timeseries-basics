from engine import *; import pickle
S={'원 전략(주간)':(w_orig,'W'),'① 신호 교체':(w_s1,'M'),'② +군집·위험가중':(w_s2,'M'),'③ 코어-위성 70/30':(w_s3,'M'),
   '균등 식료품 바스켓':(w_ew,'M'),'S&P 500 (SPY)':(w_spy,'M'),'필수소비재 ETF (XLP)':(w_xlp,'M')}
rets={};logs={}
for k,(f,fr) in S.items(): rets[k],logs[k]=run(f,fr)
D=pd.DataFrame(rets); spy=D['S&P 500 (SPY)']; rf=RF.reindex(D.index)
def met(r):
    y=len(r)/252; cum=(1+r).prod()-1; ann=(1+cum)**(1/y)-1; vol=r.std()*np.sqrt(252)
    sh=(r-rf).mean()/r.std()*np.sqrt(252); ex=r-spy; te=ex.std()*np.sqrt(252); ir=ex.mean()*252/te if te>0 else np.nan
    wealth=(1+r).cumprod(); mdd=(wealth/wealth.cummax()-1).min()
    m=(1+r).resample('ME').prod()-(1+spy).resample('ME').prod()
    return dict(ann=ann*100,vol=vol*100,sharpe=sh,exc=(ann-((1+(1+spy).prod()-1)**(1/y)-1))*100,te=te*100,ir=ir,t_ir=ir*np.sqrt(y) if te>0 else np.nan,mdd=mdd*100,win=(m>0).mean()*100)
M=pd.DataFrame({k:met(D[k]) for k in D}).T
TO={k:np.mean([x[2] for x in logs[k]])*(52 if S[k][1]=='W' else 12)*100 for k in logs}
M['turnover']=pd.Series(TO)
pd.set_option('display.width',200); print(M.round(2))
Y=pd.DataFrame({k:(1+D[k]).groupby(D.index.year).prod()-1 for k in D})*100; print(Y.round(1))
D.to_csv('data/daily_returns.csv'); M.to_csv('data/metrics.csv'); Y.to_csv('data/yearly.csv'); pickle.dump(logs,open('data/logs.pkl','wb'))
