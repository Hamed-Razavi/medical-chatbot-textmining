import pandas as pd, numpy as np, matplotlib, ast
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8})
df=pd.read_csv('symptoms_extracted_v2.csv'); df['s']=df.symptoms_v2.apply(ast.literal_eval)
te=TransactionEncoder(); S=pd.DataFrame(te.fit(df.s).transform(df.s),columns=te.columns_)
top=S.mean().sort_values(ascending=False).index[:16]
p=S[top].mean().values; J=(S[top].T.astype(int)@S[top].astype(int)).values/len(S)
L=J/np.outer(p,p); np.fill_diagonal(L,np.nan)
fig,a=plt.subplots(figsize=(3.5,3.7))
im=a.imshow(np.log2(np.where(L>0,L,np.nan)),cmap='RdBu_r',vmin=-4,vmax=4)
lab=[t.replace('_',' ') for t in top]
a.set_xticks(range(16)); a.set_xticklabels(lab,rotation=90,fontsize=7.5); a.set_yticks(range(16)); a.set_yticklabels(lab,fontsize=7.5)
cb=plt.colorbar(im,fraction=.046,pad=.03); cb.set_label('log$_2$(lift); blank = never co-occur',fontsize=7.5); cb.ax.tick_params(labelsize=7.5)
plt.tight_layout(); plt.savefig('fig_lift_v2.png',dpi=400,bbox_inches='tight',pad_inches=0.03)
