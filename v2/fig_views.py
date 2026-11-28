import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8.5,'axes.linewidth':0.6})
A=json.load(open('results_partA.json')); D=json.load(open('results_partD.json'))
basic=lambda p,c:[2*p-c[1],2*p-c[0]]
views=['patient','patient_masked','bot','bot_masked','full','full_masked','full_masked_lenmatched']
lab=['Patient','Patient, masked','Bot','Bot, masked','Full','Full, masked','Full, masked,\nlength-matched']
fig,ax=plt.subplots(2,1,figsize=(3.5,4.6),sharex=True)
x=np.arange(len(views)); w=0.38
for j,(m,c) in enumerate([('ari','#4C72B0'),('ami','#DD8452')]):
    mean=[D['point'][v][m] for v in views]; lo=[basic(D['point'][v][m],D['ci'][v][m])[0] for v in views]; hi=[basic(D['point'][v][m],D['ci'][v][m])[1] for v in views]
    ax[0].bar(x+(j-0.5)*w,mean,w,color=c,label=m.upper(),yerr=[np.subtract(mean,lo),np.subtract(hi,mean)],capsize=1.5,error_kw={'lw':0.6})
ax[0].set_ylabel('K-means (K = 24)'); ax[0].set_ylim(0,1.05); ax[0].legend(frameon=False,fontsize=7.5,ncol=2,loc='upper left'); ax[0].set_title('(a) K-means agreement with labels (95% CI)',fontsize=8)
f1=[A['supervised'][v]['f1'][0] for v in views]; sd=[A['supervised'][v]['f1'][1] for v in views]
ax[1].bar(x,f1,0.6,color='#55A868',yerr=sd,capsize=1.5,error_kw={'lw':0.6})
ax[1].set_ylabel('Macro-F1'); ax[1].set_ylim(0,1.05); ax[1].set_title('(b) Logistic regression (mean ± SD)',fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab,fontsize=7.5,rotation=35,ha='right',rotation_mode='anchor')
for a in ax: a.grid(axis='y',alpha=.25,lw=.4); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_views.png',dpi=400)
