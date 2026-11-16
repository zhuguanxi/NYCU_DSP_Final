"""Interpret cached fold models using development validation records only."""
from pathlib import Path
import os
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[2]/'ML/.mplcache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import json,itertools,time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,roc_curve,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import X,Y,y,FD,ART,REP,FIG,FOLDS,CAT,coefficient_table,frame_md

FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':130,'savefig.dpi':160})
best=json.loads((ART/'best_models.json').read_text());m=best['selected_boost']

def plot_ablations():
 for filename,by,title,out in [('family_folds.csv','family','Feature families','family_AP.png'),('selection_folds.csv','strategy','Feature selection','selection_AP.png')]:
  a=pd.read_csv(ART/filename);models=a.model.unique();fig,ax=plt.subplots(figsize=(8,4))
  levels=sorted(a[by].unique()) if by=='family' else ['Full','Quality','Elastic','Top10','Top20']
  for j,model in enumerate(models):
   z=a[a.model==model].groupby(by).AP.agg(['mean','std']).reindex(levels)
   xx=np.arange(len(levels))+(j-(len(models)-1)/2)*.32
   ax.bar(xx,z['mean'],width=.3,yerr=z['std'],capsize=3,label=model)
  ax.set_xticks(range(len(levels)),['F'+str(i) if by=='family' else i for i in levels]);ax.set_ylabel('Average precision (development CV)');ax.set_title(title);ax.legend();fig.tight_layout();fig.savefig(FIG/out);plt.close(fig)
 a=pd.read_csv(ART/'parameter_comparison.csv');fig,axs=plt.subplots(2,2,figsize=(11,7))
 for ax,model in zip(axs.flat,a.model.unique()):
  z=a[a.model==model];ax.barh(range(len(z)),z.AP);ax.set_yticks(range(len(z)),[str(json.loads(p)) for p in z.params],fontsize=8);ax.set_title(model);ax.set_xlabel('Mean AP (3-fold)')
 fig.tight_layout();fig.savefig(FIG/'parameter_AP.png');plt.close(fig)

def data_relations():
 representative={}
 for g in FD.group.unique():
  cols=FD[FD.group==g].index
  if g.startswith('static_'):
   if g not in CAT:representative[g]=g
  else:
   for stat in ['mean','last','ever']:
    c=f'{g}__{stat}'
    if c in cols:representative[g]=c;break
 z=pd.DataFrame({g:X[c] for g,c in representative.items()})
 corr=z.corr(method='spearman',min_periods=100);counts=z.notna().astype(np.int64).T.dot(z.notna().astype(np.int64))
 corr.to_csv(ART/'clinical_spearman.csv');counts.to_csv(ART/'clinical_pairwise_n.csv')
 pairs=[]
 for i,g in enumerate(z.columns):
  for h in z.columns[i+1:]:
   rho=corr.loc[g,h]
   if pd.notna(rho):pairs.append(dict(variable1=g,variable2=h,rho=rho,abs_rho=abs(rho),observed_n=int(counts.loc[g,h])))
 pd.DataFrame(pairs).sort_values('abs_rho',ascending=False).to_csv(ART/'clinical_correlated_pairs.csv',index=False)
 effects=[]
 for c in FD.index:
  if c in CAT:continue
  a=X.loc[y==0,c].dropna();b=X.loc[y==1,c].dropna()
  rb=float(2*mannwhitneyu(b,a,alternative='two-sided').statistic/(len(a)*len(b))-1) if len(a)>1 and len(b)>1 else np.nan
  effects.append(dict(feature=c,variable=FD.loc[c,'group'],statistic=FD.loc[c,'statistic'],alive_n=len(a),death_n=len(b),alive_median=a.median(),death_median=b.median(),rank_biserial=rb,alive_missing=1-len(a)/(y==0).sum(),death_missing=1-len(b)/(y==1).sum()))
 pd.DataFrame(effects).to_csv(ART/'outcome_associations.csv',index=False)
 summaries=[]
 for cat in ['static_ICUType','static_Gender']:
  z2=pd.DataFrame({'group':X[cat].fillna(-1),'y':y})
  for g,a in z2.groupby('group'):summaries.append(dict(feature=cat,category=g,n=len(a),deaths=int(a.y.sum()),mortality_rate=a.y.mean()))
 pd.DataFrame(summaries).to_csv(ART/'categorical_outcome_rates.csv',index=False)
 return corr,counts,representative

def importance(corr):
 out=ART/'group_permutation.csv';rows=pd.read_csv(out).to_dict('records') if out.exists() else []
 coefs=[];sample_ids=[]
 # Additional highly correlated clinical clusters (not all single-feature clusters).
 groups=corr.columns.tolist();edges=[(a,b) for a,b in itertools.combinations(groups,2) if pd.notna(corr.loc[a,b]) and abs(corr.loc[a,b])>=.9]
 clusters=[]
 for a,b in edges:
  overlapping=[c for c in clusters if a in c or b in c]
  new={a,b}
  for c in overlapping:new|=c;clusters.remove(c)
  clusters.append(new)
 (ART/'correlated_clinical_clusters.json').write_text(json.dumps([sorted(c) for c in clusters],indent=2))
 for fold,(_,va) in enumerate(FOLDS):
  pipe,info,cols=joblib.load(ART/'runs'/best[m]['key']/f'fold{fold}.joblib')
  sel,_=train_test_split(va,train_size=min(1500,len(va)-1),stratify=y.iloc[va],random_state=42+fold)
  z=X.iloc[sel][cols].copy();yy=y.iloc[sel];base=average_precision_score(yy,pipe.predict_proba(z)[:,1])
  sample_ids.extend(dict(RecordID=int(X.index[i]),fold=fold) for i in sel)
  gmap={g:[c for c in cols if FD.loc[c,'group']==g] for g in FD.loc[cols].group.unique()}
  for cluster in clusters:
   name='CLUSTER:'+'+'.join(sorted(cluster));gmap[name]=[c for c in cols if FD.loc[c,'group'] in cluster]
  for ix,(g,cc) in enumerate(gmap.items()):
   if any(r['fold']==fold and r['variable']==g for r in rows):continue
   rng=np.random.default_rng(9000+1000*fold+ix);drops=[]
   for repeat in range(5):
    zz=z.copy();order=rng.permutation(len(z));zz.loc[:,cc]=z.iloc[order][cc].to_numpy()
    drops.append(base-average_precision_score(yy,pipe.predict_proba(zz)[:,1]))
   rows.append(dict(variable=g,fold=fold,base_sample_AP=base,mean_AP_drop=float(np.mean(drops)),permutation_SD=float(np.std(drops,ddof=1)),n=len(z),deaths=int(yy.sum()),columns=len(cc)))
   pd.DataFrame(rows).to_csv(out,index=False)
  lr,lri,lrc=joblib.load(ART/'runs'/best['LR']['key']/f'fold{fold}.joblib')
  c=coefficient_table(lr,lrc);c['fold']=fold;coefs.append(c)
  print(f'Explanation fold {fold}: {len(gmap)} groups on {len(sel)} validation records',flush=True)
 pd.DataFrame(sample_ids).to_csv(ART/'permutation_sample_ids.csv',index=False)
 coef=pd.concat(coefs);coef.to_csv(ART/'LR_coefficients_by_fold.csv',index=False)
 coef_summary=coef.groupby(['encoded_feature','feature','variable','statistic']).coefficient.agg(['mean','std','count']).reset_index()
 sign=coef.groupby('encoded_feature').coefficient.apply(lambda s:int(max((s>0).sum(),(s<0).sum())))
 coef_summary['same_sign_folds']=coef_summary.encoded_feature.map(sign);coef_summary.to_csv(ART/'LR_coefficient_stability.csv',index=False)
 imp=pd.DataFrame(rows).groupby('variable').agg(mean_AP_drop=('mean_AP_drop','mean'),fold_SD=('mean_AP_drop','std'),folds=('fold','nunique')).sort_values('mean_AP_drop',ascending=False)
 imp.to_csv(ART/'group_permutation_summary.csv')
 top=imp[~imp.index.str.startswith('CLUSTER:')].head(15).sort_values('mean_AP_drop')
 fig,ax=plt.subplots(figsize=(8,5));ax.barh(top.index,top.mean_AP_drop,xerr=top.fold_SD,capsize=3);ax.set_xlabel('AP decrease after grouped permutation (mean across folds)');ax.set_title(f'{m}: clinical variable dependence');fig.tight_layout();fig.savefig(FIG/'group_importance.png');plt.close(fig)
 return imp,coef_summary

def selection_stability():
 mapping=json.loads((ART/'selection_runs.json').read_text());sel=best['selected_strategy']
 records=[]
 for strategy in ['Quality','Elastic','Top10','Top20']:
  a=pd.read_csv(ART/'runs'/mapping[m+'_'+strategy]/'selected.csv')
  for (feature,var),z in a.groupby(['feature','variable']):records.append(dict(strategy=strategy,feature=feature,variable=var,selected_folds=z.fold.nunique(),selection_fraction=z.fold.nunique()/3))
 r=pd.DataFrame(records);r.to_csv(ART/'feature_selection_stability.csv',index=False)
 selected=pd.read_csv(ART/'runs'/mapping[m+'_'+sel]/'selected.csv') if sel!='Full' else pd.read_csv(ART/'runs'/best[m]['key']/'selected.csv')
 feats=selected.groupby(['feature','variable']).fold.nunique().reset_index(name='selected_folds');feats['selection_fraction']=feats.selected_folds/3;feats['family']=feats.feature.map(FD.family);feats['statistic']=feats.feature.map(FD.statistic);feats['development_missing_rate']=feats.feature.map(X.isna().mean())
 feats.to_csv(ART/'selected_features.csv',index=False)
 groups=selected.groupby('variable').fold.nunique().rename('selected_folds').to_frame();groups['stable_candidate']=groups.selected_folds>=2;groups['basis']=f'{sel}: training-only selector, 3-fold development';groups['for_dl']='candidate only; validate full versus reduced channels in future DL'
 imp=pd.read_csv(ART/'group_permutation_summary.csv').set_index('variable');groups=groups.join(imp[['mean_AP_drop','fold_SD']],how='left');groups.to_csv(ART/'candidate_variables_for_dl.csv')
 return feats,groups

def main():
 plot_ablations();corr,counts,reps=data_relations();imp,coef=importance(corr);feats,groups=selection_stability()
 topgroups=[g for g in imp.index if g in corr.index][:14]
 c=corr.loc[topgroups,topgroups];fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(c)),c.columns,rotation=75,ha='right');ax.set_yticks(range(len(c)),c.index);fig.colorbar(im,ax=ax,label='Spearman rho (observed pairs)');ax.set_title('Clinical relationships: representative summaries');fig.tight_layout();fig.savefig(FIG/'clinical_correlation.png');plt.close(fig)
 # SHAP uses an existing fold model, never an extra fit or final C data.
 import shap
 pipe,info,cols=joblib.load(ART/'runs'/best[m]['key']/'fold0.joblib');va=FOLDS[0][1]
 sample,_=train_test_split(va,train_size=500,stratify=y.iloc[va],random_state=142)
 raw=X.iloc[sample][cols];z=pipe['pre'].transform(raw);names=pipe['pre'].get_feature_names_out()
 explainer=shap.TreeExplainer(pipe['model'],feature_perturbation='tree_path_dependent',model_output='raw')
 values=explainer.shap_values(z)
 if isinstance(values,list):values=values[1]
 values=np.asarray(values)
 if values.ndim==3:values=values[:,:,1]
 pd.DataFrame(values,index=raw.index,columns=names).to_parquet(ART/'shap_validation_sample.parquet')
 pd.DataFrame({'RecordID':raw.index}).to_csv(ART/'shap_sample_ids.csv',index=False)
 shap.summary_plot(values,z,feature_names=names,show=False,max_display=15);plt.title(f'{m}: fold 0 validation SHAP (raw log-odds)');plt.tight_layout();plt.savefig(FIG/'SHAP_summary.png');plt.close()
 meanabs=np.abs(values).mean(axis=0);top=np.argsort(-meanabs)
 selected=[]
 for i in top:
  f=names[i]
  if f in raw.columns and FD.loc[f,'statistic'] not in ['missing','count','std_defined','slope_defined'] and f not in CAT:
   if FD.loc[f,'group'] not in [FD.loc[names[j],'group'] for j in selected]:selected.append(i)
  if len(selected)==3:break
 dep=[]
 fig,axs=plt.subplots(1,len(selected),figsize=(12,4),squeeze=False)
 for ax,i in zip(axs.flat,selected):
  f=names[i];observed=raw[f].notna();ax.scatter(raw.loc[observed,f],values[observed.to_numpy(),i],s=12,alpha=.55);ax.set_xlabel(f);ax.set_ylabel('SHAP (log-odds)');ax.set_title(f'n observed={observed.sum()}');dep.append(f)
 fig.tight_layout();fig.savefig(FIG/'SHAP_dependence.png');plt.close(fig)
 pred=pd.read_csv(ART/'runs'/best[m]['key']/'oof.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 fpr,tpr,_=roc_curve(pred.y,pred.score);pr,re,_=precision_recall_curve(pred.y,pred.score);axs[0].plot(fpr,tpr);axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='FPR',ylabel='Recall',title='Development OOF ROC');axs[1].step(re,pr,where='post');axs[1].axhline(pred.y.mean(),linestyle='--',color='gray');axs[1].set(xlabel='Recall',ylabel='Precision',title='Development OOF PR');fig.tight_layout();fig.savefig(FIG/'OOF_ROC_PR.png');plt.close(fig)
 from sklearn.calibration import calibration_curve
 probtrue,probpred=calibration_curve(pred.y,pred.score,n_bins=10,strategy='quantile');fig,ax=plt.subplots(figsize=(5,4));ax.plot(probpred,probtrue,'o-');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed mortality fraction',title='Development OOF calibration (no recalibration)');fig.tight_layout();fig.savefig(FIG/'OOF_calibration.png');plt.close(fig)
 joint=[];fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,(g,h) in zip(axs,[('BUN','Creatinine'),('NIDiasABP','NIMAP'),('static_Age','BUN')]):
  v=X[[reps[g],reps[h]]].dropna();a=v.sample(min(2000,len(v)),random_state=242)
  ax.scatter(a.iloc[:,0],a.iloc[:,1],s=8,alpha=.25);ax.set_xlabel(reps[g]);ax.set_ylabel(reps[h]);ax.set_xlim(v.iloc[:,0].quantile(.01),v.iloc[:,0].quantile(.99));ax.set_ylim(v.iloc[:,1].quantile(.01),v.iloc[:,1].quantile(.99));ax.set_title(f'rho={corr.loc[g,h]:.2f}; n={len(v)}')
  joint.append(dict(variable1=g,variable2=h,rho=corr.loc[g,h],effective_n=len(v),plotted_n=len(a),display_limits='p01-p99; observations not removed from statistics'))
 fig.tight_layout();fig.savefig(FIG/'joint_relationships.png');plt.close(fig);pd.DataFrame(joint).to_csv(ART/'joint_relationships.csv',index=False)
 effects=pd.read_csv(ART/'outcome_associations.csv');bio=effects[(effects.statistic.isin(['mean','last','admission']))&(~effects.variable.isin(CAT))].copy();bio['abs_effect']=bio.rank_biserial.abs();bio=bio.sort_values('abs_effect',ascending=False).head(10)
 pairs=pd.read_csv(ART/'clinical_correlated_pairs.csv').head(8)
 report=f'''# P5：臨床變數關係、重要性與穩定性報告

日期：2026-10-13。僅使用A/B；C未評估。重要性屬於模型依賴，無因果宣稱。所有探索結果是開發假說。

## 驗證折分組置亂重要性

模型 **{m}**；每折固定分層子樣本1500筆，每組5次置亂。同一變數全部摘要、missing/count一起置亂；另測試 |rho|>=0.90 的跨變數相關簇。數值為AP下降，不是死亡機率增幅。折間SD不是信賴區間。

{frame_md(imp.reset_index().head(15))}

![分組重要性](../figures/group_importance.png)

## 數值關係與資料覆蓋

使用未補值的代表摘要，至少100個共同有效樣本。高度相關變數可能互相替代；所有成對n和rho都已保存。相關簇整體置亂可輔助理解單項的重要性被分攤。

{frame_md(pairs[['variable1','variable2','rho','observed_n']])}

![相關性](../figures/clinical_correlation.png)

## 與死亡的描述性關聯

下表是有效觀測上的分布差異，正的rank-biserial表示死亡組較高，不表示因果。沒有p-value特徵篩選或多重比較顯著性宣稱；缺失率和有效樣本數必須一起看。

{frame_md(bio[['feature','alive_n','death_n','alive_median','death_median','rank_biserial']])}

## 非線性與共同變化

既有fold0模型，在500筆固定驗證樣本作Tree SHAP。使用tree-path-dependent、raw輸出，SHAP值單位為log-odds，而非風險百分點。尚未作重新校準。

![SHAP](../figures/SHAP_summary.png)
![關係散點](../figures/SHAP_dependence.png)

散點選擇 {', '.join(dep)}，只畫實際有值的觀測，缺失補值對預測的作用仍包含在模型中。另以F0–F2重訓消融與LR方向一致性表輔助解讀。這些圖不能決定臨床門檻或證明一個變數改變另一個變數。

![共同變化](../figures/joint_relationships.png)

共同變化圖使用真實未補值的成對觀測；為顯示縮放採p01–p99範圍，不刪除統計資料，每圖最多抽取2000點。有效n及抽樣規則在joint_relationships.csv。數值與觀測模式的差異主要以F0–F2的實際重訓消融與LR不同statistic係數解讀；未作數值／missing旗標分開置亂，以免打破其資料一致性。

## 候選清單

精簡策略 **{best['selected_strategy']}** 的三折欄位聯集 {len(feats)} 欄；這是穩定性候選，不等於三折或最終模型都用同一份欄位。臨床群組 {len(groups)} 個，至少兩折入選 {int(groups.stable_candidate.sum())} 個。詳見 [欄位清單](../artifacts/selected_features.csv)、[深度學習候選通道](../artifacts/candidate_variables_for_dl.csv)。低排名變數仍保留於原始序列；驗證重要性未回灌同折selector。

方法參考：[Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)、[SHAP因果解讀限制](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html)。
'''
 (REP/'P5_變數關係與解釋.md').write_text(report,encoding='utf-8')
 print('P5 reports and plots completed',flush=True)

if __name__=='__main__':main()

