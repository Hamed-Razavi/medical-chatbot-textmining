import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7,'axes.linewidth':0.6})
C=json.load(open('results_partC.json')); A=json.load(open('results_partA.json'))
fig,ax=plt.subplots(2,1,figsize=(3.5,4.6))
for v,ls in [('patient','-'),('full','--')]:
    r=C[v]; k=[x['k'] for x in r]
    ax[0].errorbar(k,[x['cv'] for x in r],yerr=[x['cv_sd'] for x in r],color='C0',ls=ls,marker='o',ms=3,lw=1,capsize=2,label=f'C$_V$ ({v})')
    ax[0].errorbar(k,[x['npmi'] for x in r],yerr=[x['npmi_sd'] for x in r],color='C3',ls=ls,marker='s',ms=3,lw=1,capsize=2,label=f'NPMI ({v})')
    ax[0].errorbar(k,[x['ami'] for x in r],yerr=[x['ami_sd'] for x in r],color='C2',ls=ls,marker='v',ms=3,lw=1,capsize=2,label=f'AMI ({v})')
    s=A['kmeans'][v]['sweep']; k2=[x['k'] for x in s]
    ax[1].errorbar(k2,[x['sil'] for x in s],yerr=[x['sil_sd'] for x in s],color='C0',ls=ls,marker='o',ms=3,lw=1,capsize=2,label=f'Silhouette ({v})')
    ax[1].errorbar(k2,[x['ari'] for x in s],yerr=[x['ari_sd'] for x in s],color='C1',ls=ls,marker='^',ms=3,lw=1,capsize=2,label=f'ARI ({v})')
    ax[1].errorbar(k2,[x['ami'] for x in s],yerr=[x['ami_sd'] for x in s],color='C2',ls=ls,marker='v',ms=3,lw=1,capsize=2,label=f'AMI ({v})')
ax[0].axhline(0,color='grey',lw=.4)
ax[0].set_xlabel('Number of topics K'); ax[0].set_ylabel('Score'); ax[0].set_title('(a) LDA',fontsize=7); ax[0].set_xticks([5,10,15,20,24,30]); ax[0].set_ylim(-0.2,1.15)
ax[0].legend(fontsize=5.3,frameon=False,ncol=3,loc='upper center')
ax[1].set_xlabel('Number of clusters K'); ax[1].set_ylabel('Score'); ax[1].set_title('(b) TF-IDF K-means',fontsize=7); ax[1].set_xticks([5,10,15,20,24,30,40]); ax[1].set_ylim(0,1.25)
ax[1].legend(fontsize=5.3,frameon=False,ncol=3,loc='upper center')
for a in ax: a.grid(alpha=.25,lw=.4); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_ksweep_v2.png',dpi=400)
