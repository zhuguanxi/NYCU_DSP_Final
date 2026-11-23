"""Render publication-size figures from public summaries only; no model execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ML=Path(__file__).resolve().parents[1]
REPORT=ML/'reports';FIG=ML/'figures'

def read(name): return json.loads((REPORT/name).read_text(encoding='utf-8'))

def pipeline():
    fig,ax=plt.subplots(figsize=(6.8,4.9));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,label,color='#eef3f6'):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',facecolor=color,edgecolor='#657d8b',lw=.8))
        ax.text(x,y,label,ha='center',va='center',fontsize=8.5,linespacing=1.4)
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526b79',lw=1))
    box(2.4,9.45,3.5,.65,'P12 A/B development only');box(7.2,9.45,3.6,.65,'P12 C / eICU evaluation')
    box(5,8.15,8.7,.85,'Fixed cleaning, units, time window and feature schema\nFit imputation / encoding / scaling on training data only')
    arrow((2.4,9.1),(4,8.65));arrow((7.2,9.1),(6.3,8.65))
    box(1.8,6.65,2.7,.9,'435 tabular columns\nFull / Elastic');box(5,6.65,2.7,.9,'+140 time summaries\nGCS routing: P26');box(8.2,6.65,2.7,.9,'20 x 48 hourly series\nCNN: fixed 8 epochs')
    for x in (1.8,5,8.2): arrow((x,7.65),(x,7.2))
    box(1.8,5.1,2.7,.85,'GCS-only augmentation\n60% fixed on A/B');arrow((1.8,6.15),(1.8,5.6))
    box(4.5,3.9,4.4,.9,'P28: 50% / 50% log-odds mixture\nP26 + GCS-only Elastic','#e6f2ee')
    arrow((1.8,4.6),(3.3,4.45));arrow((5,6.1),(4.8,4.45))
    box(6.3,2.35,5.3,.9,'P33: 75% P28 + 25% CNN probabilities','#fff0df')
    arrow((5,3.35),(5.8,2.9));arrow((8.2,6.1),(8.2,2.9))
    box(5,0.65,8.8,1.05,'Freeze every model before scoring; keep all five seeds\nC: clean / full channels / random hours / contiguous blocks\neICU: P12-only transfer; grouped adaptation reported separately')
    arrow((6.3,1.8),(6.3,1.25))
    fig.tight_layout(pad=.25);fig.savefig(FIG/'final_report_pipeline.png',dpi=220);plt.close(fig)

def seed_chart(p60):
    models=('frozen_full','p28','p33','kedgn_ab');names=('Full','P28','P33','KEDGN A+B')
    colors=('#74818b','#159589','#e67f32','#8665ad')
    fig,ax=plt.subplots(figsize=(6.8,3.25));x=np.arange(3)
    for i,(m,n,c) in enumerate(zip(models,names,colors)):
        rows=[p60['summary'][k][m]['metrics']['ap'] for k in ('clean','gcs_full','six_full')]
        ax.bar(x+(i-1.5)*.19,[r['mean'] for r in rows],.18,yerr=[r['sd'] for r in rows],capsize=2,color=c,label=n,error_kw={'lw':.8})
    ax.set_xticks(x,['Clean','GCS absent','Six channels absent']);ax.set_ylabel('Mean AP (5 individual seeds)')
    ax.set_ylim(0,.70);ax.spines[['top','right']].set_visible(False);ax.legend(ncol=4,loc='upper center',frameon=False,fontsize=8)
    ax.tick_params(labelsize=9);ax.yaxis.label.set_size(9)
    fig.tight_layout(pad=.7);fig.savefig(FIG/'final_report_seed_comparison.png',dpi=220);plt.close(fig)

def curves(p60,p61,target):
    models=('frozen_full','p28','p33','kedgn_ab');names=('Full','P28','P33','KEDGN A+B');colors=('#74818b','#159589','#e67f32','#8665ad')
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.95),sharey=True)
    for ax,design in zip(axes,('random','block')):
        for model,name,color in zip(models,names,colors):
            values=[p60['summary']['clean'][model]['metrics']['ap']['mean']]
            spread=[p60['summary']['clean'][model]['metrics']['ap']['sd']]
            for length in (12,24,36):
                row=p61['curves'][f'{target}_{design}_{length}'][model]
                values.append(row['mean']);spread.append(row['training_seed_sd'])
            values.append(p60['summary'][f'{target}_full'][model]['metrics']['ap']['mean'])
            spread.append(p60['summary'][f'{target}_full'][model]['metrics']['ap']['sd'])
            x=np.arange(5)*25;v=np.asarray(values);s=np.asarray(spread)
            ax.plot(x,v,'o-',color=color,label=name,ms=3,lw=1.2);ax.fill_between(x,v-s,v+s,color=color,alpha=.12)
        ax.set_title(('GCS' if target=='gcs' else 'Six channels')+' / '+design,fontsize=9)
        ax.set(xticks=[0,25,50,75,100],ylim=(.34,.63),xlabel='Removed clock hours (%)')
        ax.tick_params(labelsize=8);ax.xaxis.label.set_size(8);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean individual-seed AP',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.90),pad=.6);fig.savefig(FIG/f'final_report_missingness_{target}.png',dpi=220);plt.close(fig)

if __name__=='__main__':
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.unicode_minus':False})
    p60=read('P60_five_seed_stability_summary.json');p61=read('P61_progressive_block_summary.json')
    pipeline();seed_chart(p60)
    for target in ('gcs','six'): curves(p60,p61,target)
    print('Rendered four report-size figures from public aggregate summaries only.')
