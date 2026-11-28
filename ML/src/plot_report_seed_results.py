"""Scientific figure from published cohort aggregates only."""
import os
from pathlib import Path
from report_seed_common import ML,read
os.environ.setdefault('MPLCONFIGDIR',str(ML/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 data=read(ML/'reports/P63_report_five_seed_summary.json')
 assert data['complete'] and data['sample_sd_ddof']==1
 rows={(r['cohort'],r['model']):r for r in data['mean_sd']}
 models=('frozen_full','r1_paper','kedgn_ab','p22','gcs_only_full','p26','p28','ma_gcs60_ap')
 labels=('XGBoost Full','R1 XGBoost','KEDGN A+B','P22 時間窗','P11 GCS-only','P26 分流','P28 混合','MA-AP＋GCS60')
 plt.rcParams.update({'font.sans-serif':['Microsoft JhengHei','DejaVu Sans'],'axes.unicode_minus':False,
  'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
 colors=['#334155','#64748b','#7c3aed','#0891b2','#16a34a','#0d9488','#2563eb','#ea580c']
 fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
 for ax,(case,title) in zip(axes.ravel(),(('clean','Set C：未遮蔽'),('gcs_full','Set C：GCS 全遮'),
   ('six_full','Set C：六欄全遮'),('eicu_hourly','eICU Demo：整點'))):
  y=np.arange(len(models));means=[rows[case,m]['ap_mean'] for m in models];sd=[rows[case,m]['ap_sd'] for m in models]
  for i,(a,b,c) in enumerate(zip(means,sd,colors)):
   ax.errorbar(a,i,xerr=b,fmt='o',color=c,capsize=3,markersize=6,linewidth=1.6)
  ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Average Precision (AP)')
  ax.set_title(title,loc='left',fontweight='bold');ax.set_axisbelow(True);ax.grid(axis='x',color='#e2e8f0')
  lo=min(a-b for a,b in zip(means,sd));hi=max(a+b for a,b in zip(means,sd));margin=max(.012,(hi-lo)*.12)
  ax.set_xlim(max(0,lo-margin),min(1,hi+margin))
 fig.suptitle('報告用模型：五種子 AP 平均 ± 樣本標準差\nSD 描述訓練種子變異；不是患者信賴區間',fontsize=16,fontweight='bold')
 path=ML/'figures/P63_report_five_seed_AP.png';fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
 print('Saved aggregate-only five-seed AP figure:',path,flush=True)

if __name__=='__main__':main()
