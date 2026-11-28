"""Evaluate all frozen P65 fits, then publish only per-seed metrics and sample SD."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv,json,statistics
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,recall_score
from train_xgboost_colsample import ML,ROOT,P63,SEEDS,MODELS,BASELINE,read,sha,write,relative,verify_sources,validate_frozen
from report_seed_common import CASES,case_paths,MAIN,NAMES
from r1_report_features import transform

REPORT=ML/'reports';METRICS=('ap','auroc','brier','sensitivity')
NEW_NAMES={'full_col90':'XGBoost Full＋colsample0.9','r1_paper_col90':'R1論文參數＋colsample0.9'}
LABELS={**NAMES,**NEW_NAMES,'kedgn_ab':'KEDGN A+B 適配版'}
MAIN_NEW=('full_col90','p22','ma_gcs60_ap','p28','kedgn_ab','r1_paper_col90')
CASE_NAMES={'clean':'Set C 未遮蔽','gcs_full':'Set C GCS 全遮','six_full':'Set C 六欄全遮',
    'eicu_hourly':'eICU Demo 整點','eicu_native':'eICU Demo 原始間隔'}
def score(y,p):return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
    brier=float(brier_score_loss(y,p)),sensitivity=float(recall_score(y,p>=.5)))
def stats(values):
    if all(v==values[0] for v in values):return float(values[0]),0.
    return float(statistics.mean(values)),float(statistics.stdev(values))
def csvfile(name,rows):
    with (REPORT/name).open('w',encoding='utf-8',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');out.writeheader();out.writerows(rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def main():
    frozen=validate_frozen();sources={relative(Path(__file__)):sha(Path(__file__))}
    old=read(REPORT/'P63_report_five_seed_summary.json')
    old_means={(r['cohort'],r['model']):r for r in old['mean_sd']}
    means_meta=read(P63/'training_inputs/r1_AB_metadata.json');r1meta=read(P63/'evaluation_inputs/r1_inputs.json')
    verify_sources(r1meta['files'])
    fitted={(r['model'],r['seed']):joblib.load(ML/r['file']) for r in frozen['records']}
    rows=[];mean_rows=[];comparisons=[];primary=[];cases={};variance=[]
    for case in CASES:
        meta=read(P63/f'evaluation/{case}.json');baseline_file=P63/f'evaluation/{case}.npz'
        assert sha(baseline_file)==meta['predictions_sha256']
        with np.load(baseline_file) as a:ids=a['ids'].copy();y=a['y'].copy();old_names=list(a['models']);old_p=a['predictions'].copy()
        n,d=(782,91) if case.startswith('eicu') else (4000,585)
        assert (len(y),int(y.sum()))==(n,d)
        features,_=case_paths(case);assert sha(features)==meta['input_sha256'][relative(features)]
        frame=pd.read_parquet(features);np.testing.assert_array_equal(frame.index,ids)
        rawfile=ML/r1meta['features'][case if case.startswith('eicu') else 'clean']
        raw=pd.read_parquet(rawfile);np.testing.assert_array_equal(raw.index,ids)
        masked=() if case=='clean' or case.startswith('eicu') else ('GCS',) if case=='gcs_full' else ('GCS','Lactate','MAP','Mg','Urine','MechVent')
        r1x=transform(raw.astype('float64'),means_meta['means'],masked)
        p=np.empty((2,5,len(y)))
        for j,name in enumerate(MODELS):
            for k,seed in enumerate(SEEDS):
                if name=='full_col90':
                    model,columns=fitted[name,seed];p[j,k]=model.predict_proba(frame.loc[:,columns])[:,1]
                else:p[j,k]=fitted[name,seed].predict_proba(r1x)[:,1]
                assert np.isfinite(p[j,k]).all() and ((p[j,k]>=0)&(p[j,k]<=1)).all()
                rows.append(dict(cohort=case,model=name,seed=seed,stays=len(y),deaths=int(y.sum()),**score(y,p[j,k])))
            group=[r for r in rows if r['cohort']==case and r['model']==name]
            row=dict(cohort=case,model=name,seed_count=5,stays=len(y),deaths=int(y.sum()))
            for metric in METRICS:row[metric+'_mean'],row[metric+'_sd']=stats([r[metric] for r in group])
            mean_rows.append(row)
            reference=old_means[case,BASELINE[name]]
            comparisons.append(dict(cohort=case,model=name,reference=BASELINE[name],
                **{key:value for m in METRICS for key,value in [(m+'_reference_mean',reference[m+'_mean']),
                    (m+'_reference_sd',reference[m+'_sd']),(m+'_new_mean',row[m+'_mean']),(m+'_new_sd',row[m+'_sd']),
                    (m+'_delta_mean',row[m+'_mean']-reference[m+'_mean'])]}))
            variance.append(dict(cohort=case,model=name,unique_probability_vectors=len({v.tobytes() for v in p[j]}),
                maximum_probability_difference_from_seed42=float(np.max(np.abs(p[j]-p[j,0])))))
        destination=ROOT/'predictions'/f'{case}.npz';destination.parent.mkdir(exist_ok=True)
        np.savez_compressed(destination,ids=ids,y=y,models=np.asarray(MODELS),seeds=np.asarray(SEEDS),probabilities=p)
        cases[case]=dict(file=relative(destination),sha256=sha(destination),input_sha256={relative(x):sha(x) for x in (baseline_file,features,rawfile)})
        lookup={(r['cohort'],r['model']):r for r in mean_rows}
        for name in MAIN_NEW:
            current=lookup[case,name] if name in MODELS else old_means[case,name]
            primary.append(dict(cohort=case,model=name,label=LABELS[name],seed_count=5,stays=n,deaths=d,
                **{m+s:current[m+s] for m in METRICS for s in ('_mean','_sd')}))
        print('Evaluated both col90 models x five seeds:',case,flush=True)
    validate_frozen();assert len(rows)==50 and len(mean_rows)==10 and len(primary)==30
    distinct={name:len({r['tree_structure_sha256'] for r in frozen['records'] if r['model']==name}) for name in MODELS}
    for case in CASES:
        with np.load(ML/cases[case]['file']) as a:
            for j,name in enumerate(MODELS):
                for k,seed in enumerate(SEEDS):
                    reference=next(r for r in rows if r['cohort']==case and r['model']==name and r['seed']==seed)
                    for metric,value in score(a['y'],a['probabilities'][j,k]).items():assert value==reference[metric]
    result=dict(complete=True,date='2026-11-28',fixed_colsample_bytree=.9,selected_on_tests=False,parameter_search=False,
        cross_seed_probability_averaging=False,sample_sd_ddof=1,sensitivity_threshold=.5,new_final_fits=10,
        main_models=list(MAIN_NEW),new_model_names=NEW_NAMES,individual_metrics=rows,mean_sd=mean_rows,
        original_comparisons=comparisons,main_six_metrics=primary,seed_probability_checks=variance,
        distinct_tree_structures=distinct,frozen_sha256=sha(ROOT/'frozen.json'),protocol_sha256=sha(ROOT/'protocol.json'),
        evaluator_source_sha256=sha(Path(__file__)),baseline_mean_csv_sha256=sha(REPORT/'P63_mean_sd_results.csv'))
    write(ROOT/'evaluation.json',dict(cases=cases,source_sha256=sources,complete=True,frozen_sha256=sha(ROOT/'frozen.json')))
    write(REPORT/'P65_colsample_summary.json',result)
    csvfile('P65_all_seed_results.csv',rows);csvfile('P65_mean_sd_results.csv',mean_rows)
    csvfile('P65_original_comparisons.csv',comparisons);csvfile('P65_main_six_metrics.csv',primary)
    main_lookup={(r['cohort'],r['model']):r for r in primary}
    detail=[]
    for case in CASES:
        detail.extend([f'## {CASE_NAMES[case]}','',table(['模型','AP','AUROC','Brier','敏感度 @0.5'],[
            [LABELS[name],*[f'{main_lookup[case,name][m+"_mean"]:.5f} ± {main_lookup[case,name][m+"_sd"]:.5f}' for m in METRICS]] for name in MAIN_NEW]),''])
    difference=table(['隊列','新版本','原版 AP','colsample0.9 AP','平均 ΔAP'],[
        [CASE_NAMES[r['cohort']],NEW_NAMES[r['model']],f'{r["ap_reference_mean"]:.5f} ± {r["ap_reference_sd"]:.5f}',
         f'{r["ap_new_mean"]:.5f} ± {r["ap_new_sd"]:.5f}',f'{r["ap_delta_mean"]:+.5f}'] for r in comparisons])
    lines=['# P65：Full／R1 的90%特徵抽樣與報告六模型','',
        '依使用者要求，Full與R1各新增固定colsample_bytree=0.9的五種子版本；其餘參數、原A+B病例、特徵schema與前處理沿原設定。這是新的訓練配置，原版留作對照。沒有依Set C／Demo挑比例、挑種子或挑較高的統計口徑。','',
        '每棵樹隨機抽90%的候選特徵，病例抽樣仍為100%。Full抽樣作用於編碼後的特徵；R1作用於148個摘要欄位。這不是將病人的量測改成缺失，也不是整個模型永久刪除固定10%欄位。','',
        '**報告全部數字：各種子先計算指標，再列五次平均±樣本SD（ddof1）；不做跨種子機率集成。** 敏感度固定0.5，Set C與Demo分開；兩個Demo版本為同一批病例。','',
        f'新增10次全A+B最終擬合，兩模型各五個種子42–46；實際不同樹結構套數為Full {distinct["full_col90"]}/5、R1 {distinct["r1_paper_col90"]}/5。CPU hist演算法保持不變，沒有為加速同時更換裝置。P22、P28、MA、KEDGN沿既有凍結結果，沒有重訓。','',
        'R1新版本應稱「論文參數＋特徵抽樣0.9」，它已增加原參數版沒有的隨機特徵抽樣，不能冒充逐項相同的論文配置。KEDGN沿A+B適配版；其流程限制見[P64稽核](P64_kedgn_implementation_audit.md)。','',
        '## 原版與新版本：完整AP變化','',difference,'',*detail,
        '## 範圍與限制','',
        '未改資料切分、前處理或增強位置，沒有重新搜尋參數。標準差反映本輪模型隨機性，不涵蓋重新切分／選參或跨院變異；不同方法的隨機性來源也不同。所有情境均完整列出，沒有因測試成績較低就改比例或捨棄種子。','',
        '[事前規格](P65_colsample_protocol.md) · [50列逐種子](P65_all_seed_results.csv) · [新版本平均／SD](P65_mean_sd_results.csv) · [原版差值](P65_original_comparisons.csv) · [六模型四指標](P65_main_six_metrics.csv) · [完整彙總](P65_colsample_summary.json) · [重要環境](../ENVIRONMENT.md)','',
        '從DSP根目錄重跑：','', '```powershell',
        '& ML/.venv/Scripts/python.exe ML/src/train_xgboost_colsample.py',
        '& ML/.venv/Scripts/python.exe ML/src/evaluate_xgboost_colsample.py','```','',
        '已完成權重會核對來源後保留。新權重、快取及逐筆預測位於Git排除的ML/artifacts/p65_xgboost_colsample_09；資料、模型及正式報告PDF不提交。']
    (REPORT/'P65_colsample_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'complete':True,'distinct_tree_structures':distinct,'comparisons':comparisons},indent=2),flush=True)

if __name__=='__main__':main()
