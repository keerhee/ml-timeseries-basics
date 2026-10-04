import pandas as pd, numpy as np, matplotlib, pickle
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NotoSansCJK-Regular' in f or 'NotoSansCJK-Bold' in f: fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':13,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#94A3B8','axes.labelcolor':'#1E293B',
 'xtick.color':'#475569','ytick.color':'#475569','savefig.dpi':200,'axes.titleweight':'bold','axes.titlesize':14})
INK='#1E293B';BLUE='#2563EB';DB='#1E4E8C';GRN='#2E7D4F';RED='#C0392B';AMB='#B7791F';GR='#94A3B8'
def save(n): plt.tight_layout(); plt.savefig(f'fig/{n}.png',facecolor='white'); plt.close()
D=pd.read_csv('data/daily_returns.csv',index_col=0,parse_dates=True)
M=pd.read_csv('data/metrics.csv',index_col=0); Y=pd.read_csv('data/yearly.csv',index_col=0)
K=list(D.columns); short={K[0]:'원 전략',K[1]:'① 신호 교체',K[2]:'② +군집·\n위험가중',K[3]:'③ 코어-위성\n70/30',K[4]:'균등\n바스켓',K[5]:'S&P 500',K[6]:'XLP'}
col={K[0]:RED,K[1]:'#93C5FD',K[2]:BLUE,K[3]:DB,K[4]:GR,K[5]:INK,K[6]:'#CBD5E1'}
# 1 sharpe ladder
fig,ax=plt.subplots(figsize=(11,4.4)); x=np.arange(7)
b=ax.bar(x,M.sharpe,color=[col[k] for k in K],width=.62)
for i,v in enumerate(M.sharpe): ax.text(i,v+.02,f'{v:.2f}',ha='center',fontsize=12.5,fontweight='bold')
ax.axhline(M.sharpe[K[5]],color=INK,ls='--',lw=1); ax.set_xticks(x); ax.set_xticklabels([short[k] for k in K],fontsize=11.5)
ax.set_ylabel('샤프 지수 (연율)'); ax.set_ylim(0,1.0); ax.set_title('2019.1–2026.9, 거래비용 편도 10bp 반영')
save('fig_ladder')
# 2 wealth log
fig,ax=plt.subplots(figsize=(11,4.4))
for k,ls,lw in [(K[0],'-',1.8),(K[2],'-',1.8),(K[3],'-',2.6),(K[5],'--',2.2),(K[4],':',2.0)]:
    ax.plot(D.index,1000*(1+D[k]).cumprod(),color=col[k],ls=ls,lw=lw,label=short[k].replace('\n',''))
ax.set_yscale('log'); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([500,1000,2000,3000]); ax.set_yticklabels(['500','1,000','2,000','3,000'])
ax.set_ylabel('1,000달러의 가치 ($, 로그)'); ax.legend(frameon=False,ncol=5,loc='upper left',fontsize=11)
save('fig_wealth')
# 3 drawdown
fig,ax=plt.subplots(figsize=(11,3.9))
for k in [K[0],K[5],K[3]]:
    w=(1+D[k]).cumprod(); dd=(w/w.cummax()-1)*100; ax.plot(D.index,dd,color=col[k],lw=1.8 if k!=K[3] else 2.4,label=f"{short[k].replace(chr(10),'')} (최대 {dd.min():.0f}%)")
ax.set_ylabel('고점 대비 낙폭 (%)'); ax.legend(frameon=False,loc='lower left',fontsize=11.5)
save('fig_drawdown')
# 4 risk-return
fig,ax=plt.subplots(figsize=(10,4.6))
for k in K:
    ax.scatter(M.vol[k],M.ann[k],s=220,color=col[k],zorder=3,edgecolor='white')
    ax.annotate(short[k].replace('\n',''),(M.vol[k],M.ann[k]),xytext=((-14,8) if k==K[3] else (9,-4)),textcoords='offset points',fontsize=11.5,color=INK,ha=('right' if k==K[3] else 'left'))
rf=2.2
xs=np.linspace(0,32,10); ax.plot(xs,rf+M.sharpe[K[5]]*xs,color=INK,lw=.8,ls=':'); ax.text(21.5,rf+M.sharpe[K[5]]*21.5+0.3,'S&P 500과 같은 샤프 지수',fontsize=10.5,color='#475569',rotation=0)
ax.set_xlim(12,33); ax.set_ylim(-1,21); ax.set_xlabel('연율 변동성 (%)'); ax.set_ylabel('연율 수익률 (%)')
save('fig_riskret')
# 5 turnover + cost
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.0),gridspec_kw={'width_ratios':[1.25,1]})
kk=K[:5]; a1.barh([short[k].replace('\n','') for k in kk][::-1],M.turnover[kk][::-1],color=[col[k] for k in kk][::-1])
for i,v in enumerate(M.turnover[kk][::-1]): a1.text(v+60,i,f'{v:,.0f}%',va='center',fontsize=11)
a1.set_xlabel('연간 회전율 (%)'); a1.set_xlim(0,6600); a1.set_title('얼마나 자주 사고파나',fontsize=13)
c=[0,5,10,20]; s=[0.28,0.19,0.10,-0.08]; a2.plot(c,s,'o-',color=RED,lw=2.4,label='원 전략'); a2.plot([0,20],[0.85,0.82],'s-',color=DB,lw=2.4,label='③ 코어-위성')
a2.axhline(0,color=GR,lw=.8); a2.set_xlabel('편도 거래비용 (bp)'); a2.set_ylabel('샤프 지수'); a2.legend(frameon=False); a2.set_title('비용이 샤프를 깎는 속도',fontsize=13)
save('fig_turnover')
# 6 yearly excess heatmap
E=Y.sub(Y[K[5]],axis=0)[[K[0],K[1],K[2],K[3],K[4]]]
fig,ax=plt.subplots(figsize=(11,3.8)); vmax=40
im=ax.imshow(E.T.values,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
ax.set_xticks(range(len(E.index))); ax.set_xticklabels([str(i) if i!=2026 else '2026\n(~9월)' for i in E.index])
ax.set_yticks(range(5)); ax.set_yticklabels([short[k].replace('\n','') for k in E.columns])
for i in range(5):
    for j in range(len(E.index)):
        v=E.T.values[i,j]; ax.text(j,i,f'{v:+.0f}',ha='center',va='center',fontsize=11.5,color='white' if abs(v)>22 else INK)
ax.set_title('연도별 S&P 500 대비 초과수익 (%p)'); ax.spines[:].set_visible(False)
save('fig_yearly')
# 7 sensitivity
rows=[('③ 코어 70/30 (기준)',0.84),('코어 50/50',0.80),('코어 80/20',0.83),('③ 비용 0bp',0.85),('③ 비용 20bp',0.82),
      ('② 군집 K=3',0.51),('② 군집 K=4 (기준)',0.58),('② 군집 K=5',0.51),('② 군집 K=6',0.48),('② 수축 없음',0.62),('② 수축 λ=0.25',0.58)]
fig,ax=plt.subplots(figsize=(10,4.6))
ys=np.arange(len(rows))[::-1]
for y,(n,v) in zip(ys,rows): ax.plot([0,v],[y,y],color='#CBD5E1',lw=2); ax.scatter(v,y,s=120,color=DB if n.startswith(('③','코어')) else BLUE,zorder=3); ax.text(v+.015,y,f'{v:.2f}',va='center',fontsize=11)
ax.axvline(0.79,color=INK,ls='--',lw=1.2); ax.text(0.79,len(rows)-0.4,' S&P 500 0.79',fontsize=11)
ax.axvline(0.10,color=RED,ls=':',lw=1.2); ax.text(0.10,len(rows)-0.4,' 원 전략 0.10',fontsize=11,color=RED)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows],fontsize=11.5); ax.set_xlim(0,1.0); ax.set_xlabel('샤프 지수 (2019.1–2026.9)')
save('fig_sens')
# 8 IC
ic1=pd.read_csv('data/ic1.csv',index_col=0).squeeze(); ic2=pd.read_csv('data/ic2.csv',index_col=0).squeeze()
fig,ax=plt.subplots(figsize=(10,3.9))
bins=np.linspace(-1,1,21)
ax.hist(ic1,bins=bins,alpha=.55,color=RED,label=f'1개월 샤프 → 다음 주 (평균 {ic1.mean():+.3f}, {len(ic1)}회)',density=True)
ax.hist(ic2,bins=bins,alpha=.55,color=BLUE,label=f'12-1 샤프 → 다음 달 (평균 {ic2.mean():+.3f}, {len(ic2)}회)',density=True)
ax.axvline(0,color=INK,lw=1); ax.set_xlabel('순위 상관계수 (IC): 신호 순위와 다음 기간 수익률 순위'); ax.set_ylabel('밀도'); ax.legend(frameon=False,fontsize=11)
save('fig_ic')
# 9 dendrogram example
from engine import R,avail,idx,t0
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.spatial.distance import squareform
t=idx.searchsorted(pd.Timestamp('2024-03-01')); nm=avail(R.iloc[t-252:t]); C=R.iloc[t-126:t][nm].corr()
Dm=np.sqrt(np.clip(0.5*(1-C.values),0,None)); np.fill_diagonal(Dm,0); Z=linkage(squareform(Dm,checks=False),'average')
fig,ax=plt.subplots(figsize=(10,4.2))
cut=Z[-4,2]+1e-6 if False else (Z[-4,2]+Z[-3,2])/2
from scipy.cluster.hierarchy import set_link_color_palette; set_link_color_palette([BLUE,GRN,AMB,DB]); dendrogram(Z,labels=nm,ax=ax,color_threshold=cut,above_threshold_color=GR,leaf_font_size=12)
ax.axhline(cut,color=RED,ls='--',lw=1.3); ax.text(ax.get_xlim()[0]+2,cut+0.012,'여기서 잘라 4개 군집',ha='left',color=RED,fontsize=11.5,bbox=dict(fc='white',ec='none',pad=1))
ax.set_ylabel('상관 거리'); ax.set_title('2024년 3월 기준 (직전 6개월 상관)')
save('fig_dendro')
