import json, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,'axes.linewidth':0.6})
R=json.load(open('results.json'))
lu=pd.read_csv('lda_user.csv'); lf=pd.read_csv('lda_full.csv'); ku=pd.read_csv('km_user.csv'); kf=pd.read_csv('km_full.csv')
fig,ax=plt.subplots(2,1,figsize=(3.5,4.6))
ax[0].errorbar(lu.k,lu.cv_mean,yerr=lu.cv_sd,color='C0',marker='o',ms=3,lw=1,capsize=2,label='C$_V$ (patient)')
ax[0].errorbar(lf.k,lf.cv_mean,yerr=lf.cv_sd,color='C0',marker='o',ms=3,lw=1,capsize=2,ls='--',label='C$_V$ (full)')
ax[0].plot(lu.k,lu.nmi_mean,color='C2',marker='v',ms=3,lw=1,label='NMI (patient)')
ax[0].plot(lf.k,lf.nmi_mean,color='C2',marker='v',ms=3,lw=1,ls='--',label='NMI (full)')
ax[0].set_xlabel('Number of topics K'); ax[0].set_ylabel('Score'); ax[0].set_title('(a) LDA'); ax[0].legend(fontsize=6,frameon=False,ncol=2,loc='lower right'); ax[0].set_xticks(lu.k)
for d,ls,n in [(ku,'-','patient'),(kf,'--','full')]:
    ax[1].plot(d.k,d.sil,ls=ls,color='C0',marker='o',ms=3,lw=1,label=f'Silhouette ({n})')
    ax[1].errorbar(d.k,d.ari,yerr=d.ari_sd,ls=ls,color='C1',marker='^',ms=3,lw=1,capsize=2,label=f'ARI ({n})')
    ax[1].plot(d.k,d.nmi,ls=ls,color='C2',marker='v',ms=3,lw=1,label=f'NMI ({n})')
ax[1].set_xlabel('Number of clusters K'); ax[1].set_title('(b) K-Means'); ax[1].legend(fontsize=5.5,frameon=False,ncol=3,loc='upper center'); ax[1].set_ylim(0,1.28); ax[1].set_ylabel('Score'); ax[1].set_xticks(ku.k)
for a in ax: a.grid(alpha=.25,lw=.4); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_ksweep.png',dpi=300)

# co-occurrence heatmap: lift among top 16 symptoms
df=pd.read_csv('symptoms_extracted.csv'); df['symptoms']=df.symptoms.apply(eval)
te=TransactionEncoder(); S=pd.DataFrame(te.fit(df.symptoms).transform(df.symptoms),columns=te.columns_)
top=S.mean().sort_values(ascending=False).index[:16]
p=S[top].mean().values; J=(S[top].T.astype(int)@S[top].astype(int)).values/len(S)
L=J/np.outer(p,p); np.fill_diagonal(L,np.nan)
fig,a=plt.subplots(figsize=(3.5,3.2))
im=a.imshow(np.log2(np.where(L>0,L,np.nan)),cmap='RdBu_r',vmin=-4,vmax=4)
lab=[t.replace('_',' ') for t in top]
a.set_xticks(range(16)); a.set_xticklabels(lab,rotation=90,fontsize=6); a.set_yticks(range(16)); a.set_yticklabels(lab,fontsize=6)
cb=plt.colorbar(im,fraction=.046,pad=.03); cb.set_label('log$_2$(lift); blank = never co-occur',fontsize=6); cb.ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig('fig_lift.png',dpi=300)

# rule stats
fi=apriori(S,min_support=0.02,use_colnames=True,max_len=3); ru=association_rules(fi,metric='confidence',min_threshold=0.5); ru=ru[ru.lift>1.5]
y=df.disease.values
nd=ru.apply(lambda r: len(set(y[S[list(r.antecedents|r.consequents)].all(axis=1).values])),axis=1)
# dominant-disease share
dom=ru.apply(lambda r: pd.Series(y[S[list(r.antecedents|r.consequents)].all(axis=1).values]).value_counts(normalize=True).iloc[0],axis=1)
print('rules',len(ru),'single-disease',(nd==1).sum(),'dominant>=0.8',(dom>=0.8).sum(),'max conf',ru.confidence.max(),'n conf>=.85',(ru.confidence>=.85).sum())
# pairwise rules only
pr=ru[(ru.antecedents.str.len()==1)&(ru.consequents.str.len()==1)]
print('pairwise rules',len(pr))
S2=S.copy(); print('fatigue diseases', S.groupby(y)['fatigue'].mean().sort_values(ascending=False).head(10).round(2).to_dict())
