"""Save deployable development pipelines, audit results, and write the study index."""
from pathlib import Path
import json,hashlib,platform,sys
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,roc_auc_score
from study import X,y,Y,FD,ART,REP,FIG,fitted,select_cols,basic_cols,frame_md,DATA_SHA

ROOT=Path(__file__).resolve().parents[2]
NAMES={'static_Age':'年齡','static_Gender':'性別','static_Height':'入ICU身高','static_Weight':'入ICU體重','static_ICUType':'ICU類型','Albumin':'白蛋白','ALP':'鹼性磷酸酶','ALT':'ALT','AST':'AST','Bilirubin':'膽紅素','BUN':'血中尿素氮','Cholesterol':'膽固醇','Creatinine':'肌酸酐','DiasABP':'侵入式舒張壓','FiO2':'吸入氧氣比例','GCS':'昏迷指數','Glucose':'葡萄糖','HCO3':'碳酸氫鹽','HCT':'血比容','HR':'心率','K':'鉀','Lactate':'乳酸','Mg':'鎂','MAP':'侵入式平均血壓','MechVent':'機械通氣狀態','Na':'鈉','NIDiasABP':'非侵入式舒張壓','NIMAP':'非侵入式平均血壓','NISysABP':'非侵入式收縮壓','PaCO2':'動脈二氧化碳分壓','PaO2':'動脈氧分壓','pH':'動脈pH','Platelets':'血小板','RespRate':'呼吸率','SaO2':'血氧飽和度','SysABP':'侵入式收縮壓','Temp':'體溫','TroponinI':'肌鈣蛋白I','TroponinT':'肌鈣蛋白T','Urine':'每次記錄尿量','WBC':'白血球','Weight':'體重時間序列'}

def audit():
 assert len(X)==8000 and X.index.is_unique and set(Y.source_set)=={'A','B'}
 assert set(FD.index)==set(X.columns)
 a=pd.read_csv(ART/'folds3.csv');assert a.RecordID.nunique()==8000 and len(a)==8000
 assert set(a.RecordID)==set(X.index) and set(a.fold3)=={0,1,2}
 best=json.loads((ART/'best_models.json').read_text())
 keys={p.name for p in (ART/'runs').iterdir() if (p/'config.json').exists() and json.loads((p/'config.json').read_text()).get('data_sha')==DATA_SHA}
 tested=[]
 for key in keys:
  p=ART/'runs'/key;config=json.loads((p/'config.json').read_text());assert config['data_sha']==DATA_SHA
  expected=X.index[X.static_Age>=18] if config['adult'] else X.index
  o=pd.read_csv(p/'oof.csv');assert len(o)==len(expected) and o.RecordID.is_unique and set(o.RecordID)==set(expected)
  assert np.array_equal(o.y.to_numpy(),Y.loc[o.RecordID,'In_hospital_death'].to_numpy())
  assert np.isfinite(o.score).all() and o.score.between(0,1).all()
  r=pd.read_csv(p/'results.csv');assert len(r)==config['nfold'] and r.converged.all()
  for k,z in o.groupby('fold'):
   assert np.isclose(average_precision_score(z.y,z.score),r.loc[r.fold==k,'AP'].iloc[0])
   assert np.isclose(roc_auc_score(z.y,z.score),r.loc[r.fold==k,'AUROC'].iloc[0])
  tested.append(key)
 # Independently validate sample summary / typing.
 from prepare import DATA,clean
 raw=pd.read_csv(DATA/'set-a/132539.txt');v=raw[raw.Parameter=='HR'].Value
 assert np.isclose(X.loc[132539,'HR__mean'],v[v>=0].mean())
 assert clean('HR',0)[1]=='valid' and clean('Gender',2)[1]=='invalid_code'
 for var in ['HR','Lactate','TroponinI','TroponinT']:
  assert np.array_equal(X[f'{var}__missing'],(X[f'{var}__count']==0).astype(int))
 assert (X['TroponinI__count']>0).any() and (X['TroponinT__count']>0).any()
 preparation=json.loads((ART/'preparation_manifest.json').read_text())
 h=hashlib.sha256((DATA/'set-c.tar.gz').read_bytes()).hexdigest();assert h==preparation['C_sha256']
 result={'passed':True,'corrected_data_sha256':DATA_SHA,'OOF_configs_validated':len(tested),'record_n':8000,'fold_ids_unique':True,'troponin_columns_validated':True,'C_original_archive_unchanged':True,'C_predictions_created':False,'all_current_fit_converged':all(not json.loads(l)['convergence_warnings'] for l in (ART/'fit_ledger.jsonl').read_text().splitlines() if json.loads(l).get('feature_data_sha256')==DATA_SHA)}
 (ART/'study_validation.json').write_text(json.dumps(result,indent=2));return result

def save_pipelines():
 best=json.loads((ART/'best_models.json').read_text());m=best['selected_boost'];s=best['selected_strategy'];folder=ART/'final_models';folder.mkdir(exist_ok=True)
 metadata={}
 for strategy in dict.fromkeys(['Full',s]):
  target=folder/(strategy+'.joblib')
  if target.exists():
   pipe,info,cols=joblib.load(target)
   assert info.get('feature_data_sha256')==DATA_SHA, 'Saved final model uses a different feature table'
  else:
   train=np.arange(len(X));cols=select_cols(train,2,strategy,'allAB')
   pipe,info=fitted(train,cols,m,best[m]['params'],'final_development_refit');joblib.dump((pipe,info,cols),target)
  metadata[strategy]={'model':m,'columns':cols,'parameters':best[m]['params'],'trained_on':'A+B (8000 encounters)','test_C_evaluated':False,'data_sha256':DATA_SHA}
  names=pipe['pre'].get_feature_names_out();pd.DataFrame({'encoded_column':names}).to_csv(folder/(strategy+'_encoded_columns.csv'),index=False)
  before=pipe.predict_proba(X.iloc[:10][cols])[:,1];loaded=joblib.load(target)[0];assert np.allclose(before,loaded.predict_proba(X.iloc[:10][cols])[:,1])
 (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
 return metadata

def enrich():
 fd=FD.copy();fd.index.name='feature';fd['clinical_name_zh']=fd.group.map(NAMES);fd.to_csv(ART/'feature_dictionary.csv',encoding='utf-8-sig')
 for f in ['group_permutation_summary.csv','candidate_variables_for_dl.csv']:
  z=pd.read_csv(ART/f);z['clinical_name_zh']=z.variable.map(NAMES).fillna(z.variable);z.to_csv(ART/f,index=False,encoding='utf-8-sig')
 params=pd.read_csv(ART/'parameter_comparison.csv');txt='\n\n## 配置影響\n\n'+frame_md(params[['model','params','AP','AUROC','fit_s','converged']])+'\n\nXGBoost與LightGBM的最高平均AP差異很小，按預定AP規則選取XGBoost，不宣稱統計顯著優越。正則化／樹深變化只在本次候選範圍內解讀。\n'
 p=REP/'P2_有限調參.md';p.write_text(p.read_text(encoding='utf-8').split('\n\n## 配置影響')[0]+txt,encoding='utf-8')
 for name in ['P3_特徵家族','P4_特徵篩選']:
  a=pd.read_csv(ART/('family_folds.csv' if name.startswith('P3') else 'selection_folds.csv'))
  extra='\n\n## 對照與解讀\n\n'
  if name.startswith('P3'):
   t=a.groupby(['model','family']).AP.mean().unstack()
   for model,row in t.iterrows():extra+=f'- {model}：F1−F0={row[1]-row[0]:+.4f}，F2−F1={row[2]-row[1]:+.4f}。\n'
   extra+='新增摘要並非對每個模型都有一致收益；沒有把折間差異當成顯著性證明。\n\n![特徵家族](../figures/family_AP.png)\n'
  else:
   t=a.groupby(['model','strategy']).agg(AP=('AP','mean'),AP_delta=('AP_delta_to_full','mean'),n=('input_n','mean')).reset_index();extra+=frame_md(t)+'\n\nTop10／20指動態臨床變數，還有預先保留的靜態欄位；每折選擇不同，應與selected.csv及穩定性清單一起看。\n\n![精簡比較](../figures/selection_AP.png)\n'
  p=REP/(name+'.md');p.write_text(p.read_text(encoding='utf-8').split('\n\n## 對照與解讀')[0]+extra,encoding='utf-8')

def write_index(meta,checks):
 best=json.loads((ART/'best_models.json').read_text());m=best['selected_boost'];s=best['selected_strategy']
 full=pd.read_csv(ART/'runs'/best[m]['key']/'results.csv');mapping=json.loads((ART/'selection_runs.json').read_text());red=pd.read_csv(ART/'runs'/mapping[m+'_'+s]/'results.csv')
 led=[json.loads(l) for l in (ART/'fit_ledger.jsonl').read_text().splitlines()];old=[l for l in led if not l.get('feature_data_sha256')];current=[l for l in led if l.get('feature_data_sha256')==DATA_SHA]
 imp=pd.read_csv(ART/'group_permutation_summary.csv');top=imp[~imp.variable.str.startswith('CLUSTER:')].head(10)
 pairs=pd.read_csv(ART/'clinical_correlated_pairs.csv').head(5)
 conf=pd.read_csv(ART/'confirmation_folds.csv');ct=conf.groupby(['strategy','nfold','adult']).agg(AP=('AP','mean'),AP_SD=('AP','std'),AUROC=('AUROC','mean'),n=('input_n','mean')).reset_index()
 folds=pd.read_csv(ART/'selection_folds.csv');selected=folds[(folds.model==m)&(folds.strategy==s)]
 fulln=len(meta['Full']['columns']);redn=len(meta[s]['columns'])
 report=f'''# ICU 機器學習探索：階段報告索引與總結

執行日期：2026-10-13。核准計畫中的P0–P6已執行；全部模型分數來自A/B開發交叉驗證，C未作模型评估。深度學習與外部增補尚未執行。

## 核心結果

- 開發資料8,000筆，死亡1,122筆（14.025%）。三組原始特徵數150／292／435。
- 選定 **{m}** 作後續表格基準。完整F2三折AP **{full.AP.mean():.4f} ± {full.AP.std():.4f}**、AUROC **{full.AUROC.mean():.4f}**。
- 三折精簡策略 **{s}**：AP **{red.AP.mean():.4f} ± {red.AP.std():.4f}**；相對完整平均差 **{red.AP.mean()-full.AP.mean():+.4f}**。此策略按開發結果選取，分數仍可能偏樂觀。
- 最後以全部A/B重新fit的完整／精簡模型，輸入分別 **{fulln}／{redn}** 欄。這和三折平均欄數可能不同；最終C表現尚未知。
- 對增補時間摘要與觀測模式的收益，LR與boosting不完全一致，支持繼續保留共同設定下的消融。

## 各階段報告

| 階段 | 報告 | 主要產出 |
|---|---|---|
| P0 | [資料與特徵重建](reports/P0_資料與特徵重建.md) | 原始序列、規則、字典、對齊與質量稽核 |
| P1 | [基準模型](reports/P1_基準模型.md) | Dummy與四種方法的三折性能／成本 |
| P2 | [有限調參](reports/P2_有限調參.md) | 18組候選配置，最高分與完整配置影響 |
| P3 | [特徵家族](reports/P3_特徵家族.md) | 基本摘要、變化、觀測模式的消融 |
| P4 | [特徵篩選](reports/P4_特徵篩選.md) | 品質／Elastic Net／Top10／Top20比較 |
| P5 | [變數關係與解釋](reports/P5_變數關係與解釋.md) | 相關、有效n、分組置亂、LR與SHAP |
| P6 | [穩定性與敏感性](reports/P6_穩定性補充.md) | 同一開發集五折、成人限定三折 |

## 五折及成人敏感性

{frame_md(ct)}

五折／成人限定屬於開發集穩定性補充。成人分析重新切分，不能與原三折做嚴格成對推論；沒有作患者身分層級隔離。不能用平均±SD宣稱95%信賴區間或臨床有效性。

## 哪些變數值得後續研究

下面是完整boosting模型在驗證折的group permutation依賴排名；數值是AP下降，並非死亡風險增加幅度，也不是因果重要性。

{frame_md(top[['variable','clinical_name_zh','mean_AP_drop','fold_SD']])}

高相關且可替代的變數，單項置亂可能低估依賴；已保存相關簇聯合置亂結果。另有數值摘要與missing/count的不同意義，需一起對照LR係數與原始分布。

![重要性](figures/group_importance.png)

## 銜接深度學習

- [候選通道](artifacts/candidate_variables_for_dl.csv) 與 [候選欄位](artifacts/selected_features.csv) 是訓練內selector的跨折穩定性記錄，**不能直接視為深度模型的最佳輸入**。
- 原始序列 [observations_AB.parquet](artifacts/observations_AB.parquet) 仍包含所有原始變數、值、時間與清理旗標。Weight admission與動態序列有重疊，資料字典已揭露。
- 後續應比較深度模型完整／精簡通道，保留原始時間、mask與可得靜態資訊，不把手工summary等同於時間序列。
- 已保存共同RecordID切分與完整／精簡表格pipelines。深度模型也只在A/B選方案，最後凍結後再共同評估C。
- 時間／缺失／變數關係的探索只能提供設計假說，不證明attention／graph一定優於本輪表格模型。

## 稽核與實際成本

- 總計 **{len(led)}** 次fit，其中正式修正版 **{len(current)}** 次，初版欄名除錯 **{len(old)}** 次不納入結果。
- 正式版本fit耗時合計 **{sum(l['fit_s'] for l in current):.1f}秒**；這不包含特徵提取、相關性、SHAP／置亂等成本。實際牆鐘時間不可用此合計代替。
- 最高採樣RSS **{max(l['peak_rss_mb'] for l in current):.1f}MB**；所有正式版本fit收斂警告 **{sum(bool(l['convergence_warnings']) for l in current)}** 個。
- 初版官方縮寫TropI/TropT未對到原始TroponinI/TroponinT，已修正重建並重跑；[除錯紀錄](artifacts/superseded_debug_fits.json) 保留，沒有混入最終分數。
- [驗證結果](artifacts/study_validation.json)：唯一鍵、標籤對齊、OOF全覆蓋及AP重算、肌鈣蛋白欄位、模型保存讀回與C原壓縮檔雜湊均已檢查。

## 限制與下一步

未做外部驗證、統計顯著或臨床因果宣稱。保留非負極端值與部分生理零值，這些仍可能包含紀錄問題；資料清理與先前EDA不同。C先前已被EDA探索，後續須揭露此限制。

建議先審查候選清單及資料質量，再決定深度方法與外部增補來源。本輪已提供實際表格基準，可用於後續共同切分比較。
'''
 index_path=ROOT/'ML/README.md'
 # Keep the later C evaluation notice when regenerating development reports.
 if index_path.exists():
  previous=index_path.read_text(encoding='utf-8')
  if '<!-- C_EVALUATION_START -->' in previous and '<!-- C_EVALUATION_END -->' in previous:
   notice=previous.split('<!-- C_EVALUATION_START -->',1)[1].split('<!-- C_EVALUATION_END -->',1)[0]
   title,body=report.split('\n',1)
   report=title+'\n\n<!-- C_EVALUATION_START -->'+notice+'<!-- C_EVALUATION_END -->\n'+body
   report=report.replace('全部模型分數來自A/B開發交叉驗證，C未作模型评估','P0–P6的模型分數來自A/B開發交叉驗證，該階段未評估C').replace('最終C表現尚未知','P0–P6尚未評估C；後續結果見頂端P7連結')
 index_path.write_text(report,encoding='utf-8')
 env={'python':sys.version,'platform':platform.platform(),'CPU_count':__import__('os').cpu_count(),'threads_per_model':4,'data_sha':DATA_SHA,'total_fits':len(led),'formal_fits':len(current),'superseded_fits':len(old),'C_used_for_models':False,'selected_boost':m,'selected_strategy':s}
 (ART/'execution_manifest.json').write_text(json.dumps(env,indent=2))

if __name__=='__main__':
 meta=save_pipelines();checks=audit();enrich();write_index(meta,checks);print('Final pipelines, audit, and reports completed')
