"""Publish aggregate report tables; never publish patient arrays or model files."""
import csv
from pathlib import Path
import numpy as np
from report_seed_common import *
from freeze_report_seed_study import validate_frozen

REPORT=ML/'reports'
METRICS=('ap','auroc','brier','sensitivity','rho')
CASE_NAMES={'clean':'Set C 未遮蔽','gcs_full':'Set C GCS 全遮','six_full':'Set C 六欄全遮',
 'eicu_hourly':'Demo 整點','eicu_native':'Demo 原始間隔'}

def stats(values):
 a=np.asarray(values,dtype=float)
 if np.all(a==a[0]):return dict(mean=float(a[0]),sd=0.,minimum=float(a[0]),maximum=float(a[0]))
 return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()))

def privacy(value):
 if isinstance(value,dict):
  assert not set(value)&{'ids','y','predictions','probabilities','patientunitstayid','uniquepid','RecordID'}
  for v in value.values():privacy(v)
 elif isinstance(value,list):
  for v in value:privacy(v)
 elif isinstance(value,float):assert np.isfinite(value)

def csv_file(path,rows):
 with path.open('w',encoding='utf-8',newline='') as file:
  out=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');out.writeheader();out.writerows(rows)

def table(columns,rows):
 return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])

def main():
 frozen=validate_frozen();evaluation=read(ROOT/'evaluation/summary.json')
 assert evaluation['complete'] and evaluation['frozen_sha256']==sha(ROOT/'frozen.json')
 verify_sources(evaluation['source_sha256'])
 rows=evaluation['rows'];assert len(rows)==400
 clean={(r['model'],r['seed']):r['ap'] for r in rows if r['cohort']=='clean'}
 individual=[]
 for row in rows:
  individual.append({**row,'ap_drop_from_clean':clean[row['model'],row['seed']]-row['ap'] if not row['cohort'].startswith('eicu') else None})
 means=[];differences=[]
 lookup={(r['cohort'],r['model'],r['seed']):r for r in individual}
 for case in CASES:
  record=evaluation['cases'][case];path=ROOT/'evaluation'/f'{case}.npz'
  assert sha(path)==record['predictions_sha256'];verify_sources(record['input_sha256'])
  for name in MODELS:
   group=[r for r in individual if r['cohort']==case and r['model']==name]
   assert len(group)==5
   summary=dict(cohort=case,model=name,seed_count=5,stays=group[0]['stays'],deaths=group[0]['deaths'])
   for metric in (*METRICS,'ap_drop_from_clean'):
    values=[r[metric] for r in group]
    summary.update({metric+'_'+key:value for key,value in stats(values).items()} if all(v is not None for v in values) else
     {metric+'_'+key:None for key in ('mean','sd','minimum','maximum')})
   means.append(summary)
  pairs=(('p22','frozen_full'),('gcs_only_full','frozen_full'),('p26','p22'),('p28','p26'),
   ('r1_paper','frozen_full'),('r1_matched','frozen_full'),('ma_plain_ap','ma_plain_off'),
   ('ma_gcs60_ap','ma_gcs60_off'),('ma_gcs60_ap','ma_plain_ap'),
   ('ma_plain_rho95','ma_plain_off'),('ma_gcs60_rho95','ma_gcs60_off'))
  for after,before in pairs:
   for seed in SEEDS:
    a,b=lookup[case,after,seed],lookup[case,before,seed]
    differences.append(dict(cohort=case,after=after,before=before,seed=seed,
     **{'delta_'+m:a[m]-b[m] if a[m] is not None and b[m] is not None else None for m in (*METRICS,'ap_drop_from_clean')}))
 index=[]
 for name in MODELS:
  group=[r for r in frozen['records'] if r['model']==name]
  index.append(dict(model=name,label=NAMES[name],seeds=sorted(r['seed'] for r in group),weight_sets=len(group),
   new_final_fits=sum(r['new_fit'] for r in group),main_report_model=name in MAIN))
 public=dict(complete=True,date='2026-11-28',scope='16 fixed report predictors, five seeds each, five original primary cohorts',
  main_models=list(MAIN),inventory=index,individual_metrics=individual,mean_sd=means,
  metric_aggregation="arithmetic mean of five individual seed metrics",
  cross_seed_probability_averaging=False,main_metrics=["ap","auroc","brier","sensitivity"],sensitivity_threshold=0.5,
  reporting_standard_sha256=sha(REPORT/"P63_reporting_standard.md"),
  seed_index_differences=differences,new_formal_fits=32,new_ma_fits=24,new_r1_fits=8,
  numerical_training_diagnostic_fits=12,diagnostic_trees_per_fit=3,training_device='CPU; original fixed algorithms',
  new_gpu_training=False,selected_seed=None,parameters_reselected=False,test_selected=False,
  sample_sd_ddof=1,frozen_sha256=sha(ROOT/'frozen.json'),protocol_sha256=sha(ROOT/'protocol.json'),
  evaluation_summary_sha256=sha(ROOT/'evaluation/summary.json'),publisher_source_sha256=sha(Path(__file__)),
  caveats=['Five seeds are model randomness repeats, not five independent data samples',
   'Augmentation positions, input preprocessing and parameters remain fixed',
   'KEDGN uses seeds0-4; no paired-seed interpretation with seeds42-46',
   'Cross-seed probability averaging is excluded by the common report standard',
   'No new paired patient confidence intervals are attached to the seed means',
   'Set C and Demo were repeatedly viewed; external improvement is not independently established',
   'This stage does not repeat all P9 masks or P61 temporal-block evaluations'])
 privacy(public)
 write(REPORT/'P63_report_five_seed_summary.json',public)
 csv_file(REPORT/'P63_all_seed_results.csv',individual)
 csv_file(REPORT/'P63_mean_sd_results.csv',means)
 fields=['cohort','model','seed_count','stays','deaths',*[m+s for m in ('ap','auroc','brier','sensitivity') for s in ('_mean','_sd')]]
 csv_file(REPORT/'P63_main_model_metrics.csv',[{key:r[key] for key in fields} for r in means if r['model'] in MAIN])
 obsolete=REPORT/'P63_probability_ensembles.csv'
 if obsolete.exists():
  archived=ROOT/'publication_history/P63_probability_ensembles.csv';archived.parent.mkdir(parents=True,exist_ok=True)
  if archived.exists():assert sha(archived)==sha(obsolete)
  else:archived.write_bytes(obsolete.read_bytes())
  obsolete.unlink()
 csv_file(REPORT/'P63_seed_differences.csv',differences)
 from plot_report_seed_results import main as plot
 plot()
 mean={(r['cohort'],r['model']):r for r in means}
 fmt=lambda case,name:f'{mean[case,name]["ap_mean"]:.5f} ± {mean[case,name]["ap_sd"]:.5f}'
 detailed=[]
 for case in CASES:
  n,d=(782,91) if case.startswith('eicu') else (4000,585)
  detailed.extend([f'## {CASE_NAMES[case]}：六模型四指標（{n}次住院／{d}次死亡）','',
   table(['模型','AP','AUROC','Brier','敏感度 @0.5'],[[NAMES[name],*[f'{mean[case,name][m+"_mean"]:.5f} ± {mean[case,name][m+"_sd"]:.5f}' for m in ('ap','auroc','brier','sensitivity')]] for name in MAIN]),''])
 lines=['# P63：報告用固定五種子結果','',
  '**統一口徑：每個種子分別計算指標，再列五次平均 ± 樣本標準差（ddof=1）。** 不使用跨種子機率集成、不挑最好種子。所有 AP、AUROC、Brier 與敏感度均採同一規則；敏感度固定門檻0.5。完整規則見 [報告統一口徑](P63_reporting_standard.md)。','',
  'P28 本身的表格混合屬固定模型架構；每個種子各執行一套既定 P28，不額外平均不同種子版本的機率。','',
  '**KEDGN 為固定作者架構／參數的 A+B 適配版。** 與原作者流程有解析、病例保留及選 checkpoint 規則差異；原作者切分復現與本表 Set C 分開。完整核對見 [P64 實作稽核](P64_kedgn_implementation_audit.md)。','',
  '本輪補齊報告用六組主模型與必要控制組，總共 **16 個固定預測程序 × 五種子權重 × 五個主要情境**。全體權重先封存後評估，沒有重選參數／alpha／種子。主表為 **先各算一次指標，再取五次算術平均 ± 樣本標準差（ddof=1）**。標準差不是患者信賴區間。','',
  '新增 24 次 MA Full 與八次 R1 最終 A+B 擬合，共 **32 次正式擬合**；另有 12 次 A/B 三樹數值診斷，與正式結果分開。其餘重用既有權重，沒有重訓 KEDGN。新增訓練沿原 CPU 演算法；使用者允許 GPU，本輪沒有為裝置更換演算法。','',
  '![五種子 AP 平均與樣本標準差](../figures/P63_report_five_seed_AP.png)','',
  '## 六組主模型：AP 平均 ± 種子 SD','',
  table(['模型',*[CASE_NAMES[c] for c in CASES]],[[NAMES[n],*[fmt(c,n) for c in CASES]] for n in MAIN]),'',
  *detailed,
  '## 支持組與 MA 機制控制','',
  table(['模型',*[CASE_NAMES[c] for c in CASES]],[[NAMES[n],*[fmt(c,n) for c in CASES]] for n in MODELS if n not in MAIN]),'',
  '## MA 補值依賴 rho：平均 ± 種子 SD','',
  'rho 是至少一條實際樹路徑直接讀到補值的病例比例。每個模型內跨樹取聯集；它不是資料缺失率，也不涵蓋 missing/count 旗標的間接使用。表內單位為百分比；原 XGBoost／R1 等未量測，不能補成零。','',
  table(['模型',*[CASE_NAMES[c] for c in CASES]],[[NAMES[n],*[f'{mean[c,n]["rho_mean"]*100:.2f}% ± {mean[c,n]["rho_sd"]*100:.2f}%' for c in CASES]] for n,_,_,_ in MA_CONFIGS]),'',
  '## 種子與實際權重','',
  table(['模型','種子','權重套數','新增全 A+B 擬合'],[[NAMES[r['model']],','.join(map(str,r['seeds'])),r['weight_sets'],r['new_final_fits']] for r in index]),'',
  'P22 的五種子時間窗分支原已在 P60 訓練，本輪補其獨立評估。P26 與 P28 為可執行組合，重用相應種子的元件、固定分流與混合權重，不另擬合組合器。KEDGN 沿原0–4；其索引不等於其他模型的42–46。','',
  '已逐筆確認：Full／Elastic、GCS-only、R1 兩版及 MA 六程序，在五個情境中各自的五種子機率完全相同，SD=0 並非顯示位數造成。這些配置保留全行全欄訓練與固定增強位置；本輪未觀察到改變種子造成有效預測差異，不能推廣成 MA 永遠不受種子影響。沒有為製造變異新增抽樣，也不把相同結果當作五個獨立測試資料集。','',
  '## 報告使用與限制','',
  '主表的每一列都已有五套實際權重與五次預測。均值／SD 描述模型初始化或抽樣變異，沒有包含重新切分、重新選參、不同缺失增強位置、醫院間變異。既有 seed42 的患者 bootstrap 區間不能直接貼到這張五次平均表。','',
  'Set C 固定4,000次住院／585次死亡；Demo固定782／91，尿量沿已修正明確 mL 映射。Demo兩種取樣是同一批住院，不能當兩個獨立外部資料庫。AP 跨隊列大小亦受死亡比例影響。','',
  'R1 保留148欄四項摘要、原訓練有效量測均值補值與兩套事前固定樹配置。重建資料先核對舊 A/B 計數與補值；五情境的 seed42 逐筆預測須精確重現匯入快照。此處是共同隊列 baseline，沒有宣稱逐一重現原文全部模型與切分。','',
  '所有 MA 的 alpha 沿 P62 原 A/B 選擇固定，沒有每個種子重做搜尋。因此本輪擴充既有 AP／rho95 程序，不包含中間 alpha 的新測試或完整 P9/P61 缺失矩陣。','',
  'Set C／Demo 先前已反覆使用，這輪為固定配置的種子補強，不是新的獨立跨院確認；表內較高平均不能單憑種子 SD 當作患者層級顯著改善。','',
  '## 檔案與重跑','',
  '[報告統一口徑](P63_reporting_standard.md) · [原事前訓練規格](P63_report_five_seed_protocol.md) · [400列逐種子指標](P63_all_seed_results.csv) · [80列平均與SD](P63_mean_sd_results.csv) · [六模型四指標](P63_main_model_metrics.csv) · [同索引種子差值](P63_seed_differences.csv) · [完整彙總](P63_report_five_seed_summary.json)','',
  '本機權重索引為 `ML/artifacts/p63_report_five_seed/frozen.json`；原始資料、權重、快取、逐筆預測和正式期末報告均不公開。來源依賴原 P60/P62/KEDGN 的本機產物及使用者提供的 R1 快照。','',
  '從 DSP 根目錄，主環境使用 `ML/.venv/Scripts/python.exe`，MA 使用 `ML/artifacts/ma_gbt_runtime/.venv/Scripts/python.exe` 並令 PYTHONPATH 指向 `ML/artifacts/ma_gbt_runtime/upstream`。依序執行：','',
  '```text','主環境：ML/src/train_report_r1_seeds.py',
  'MA環境：ML/src/train_report_ma_seeds.py check',
  'MA環境：ML/src/train_report_ma_seeds.py train --workers 6',
  '主環境：ML/src/freeze_report_seed_study.py',
  'MA環境：ML/src/evaluate_report_ma_seeds.py',
  '主環境：ML/src/evaluate_report_seeds.py',
  '主環境：ML/src/publish_report_seed_results.py',
  '主環境：ML/src/verify_report_seed_results.py','```','',
  '已完成權重與快取會驗證來源後重用；原始來源或固定規格改變時拒絕沿用，應另設新輸出根目錄，不覆寫歷史成果。']
 (REPORT/'P63_report_five_seed_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
 print('Published individual-seed metrics and mean/sample-SD only; no cross-seed ensemble.',flush=True)

if __name__=='__main__':main()
