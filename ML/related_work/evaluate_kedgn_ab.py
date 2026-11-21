"""Compare frozen A+B KEDGN with P53 Set A KEDGN and the original controls."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
from compare_frozen_kedgn import ROOT as P53, CONTROLS, result_row
from evaluate_missingness_augmentation import paired_intervals
from evaluate_predictive_boosting import contrast_c
from prepare import OUT, sha
from stress_missingness_c import score_metrics
from train_predictive_boosting import verify_sources

NAMES={'set_a_kedgn':'Set A KEDGN（五種子平均）','frozen_full':'凍結 XGBoost Full',
       'frozen_elastic':'凍結 XGBoost Elastic','gcs_only_full':'GCS-only Full',
       'gcs_only_elastic':'GCS-only Elastic','p26':'P26 時間窗分流',
       'p28':'P28 分流混合','p33':'P33 逐小時序列混合'}
COHORTS={'set_c':'P12 Set C','eicu_hourly':'eICU Demo 整點','eicu_native':'eICU Demo 原始間隔'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(root):
    destination=root/'summary.json'
    if destination.exists(): raise FileExistsError(destination)
    inference=read(root/'inference.json')
    frozen=read(root/'frozen.json')
    protocol=read(root/'protocol.json')
    selection=read(root/'selection.json')
    if sha(root/'frozen.json')!=inference['frozen_sha256']: raise ValueError('Final freeze changed')
    if sha(root/'selection.json')!=frozen['selection_sha256']: raise ValueError('A/B selection changed')
    if sha(root/'protocol.json')!=frozen['protocol_sha256']: raise ValueError('Protocol changed')
    if selection['test_data_used'] or frozen['test_data_used']: raise ValueError('Test use during training')
    inputs=read(root/'inputs/manifest.json')
    verify_sources(inputs['source_sha256'])
    controls_manifest=read(P53/'control_sources.json')
    old_inference=read(P53/'inference.json')
    verify_sources(controls_manifest['source_sha256'])
    if sha(P53/'controls.npz')!=controls_manifest['controls_sha256']: raise ValueError('Old controls changed')
    if sha(P53/'inputs.json')!=inference['p53_inputs_sha256']: raise ValueError('Test inputs changed')
    for seed in range(5):
        receipt=frozen['seeds'][str(seed)]
        for file,key in [('weights.pt','weights_sha256'),('statistics.npz','statistics_sha256')]:
            if sha(root/f'final_seed_{seed}'/file)!=receipt[key]: raise ValueError('Final model source changed')
        if receipt['training_stays']!=8000 or receipt['epochs']!=selection['selected_epoch']:
            raise ValueError('Unexpected final fit')
    reference=np.load(P53/'controls.npz',allow_pickle=False)
    results={}
    baselines={}
    primary={}
    for name in COHORTS:
        path=root/f'{name}_predictions.npz'
        oldpath=P53/f'{name}_predictions.npz'
        if sha(path)!=inference['predictions'][name]: raise ValueError('A/B prediction changed')
        if sha(oldpath)!=old_inference['predictions'][name]['sha256']: raise ValueError('Set A prediction changed')
        with np.load(path,allow_pickle=False) as saved:
            ids,y,pred=saved['ids'],saved['y'],saved['probabilities']
        with np.load(oldpath,allow_pickle=False) as saved:
            np.testing.assert_array_equal(ids,saved['ids'])
            np.testing.assert_array_equal(y,saved['y'])
            oldmean=saved['probabilities'].mean(axis=0,dtype=np.float64)
        np.testing.assert_array_equal(ids,reference[f'{name}_ids'])
        np.testing.assert_array_equal(y,reference[f'{name}_y'])
        if pred.shape!=(5,len(y)): raise ValueError('Expected five final seeds')
        ensemble,result=result_row(y,pred)
        old={'set_a_kedgn':oldmean,**{tag:reference[f'{name}_{tag}'] for tag in CONTROLS}}
        result['controls']={tag:score_metrics(y,value) for tag,value in old.items()}
        result['delta']={tag:{key:result['ensemble'][key]-value for key,value in metrics.items()}
                         for tag,metrics in result['controls'].items()}
        result.update(stays=len(y),deaths=int(y.sum()))
        if name=='set_c':
            result['paired_ci']=contrast_c(y,old,{'kedgn':ensemble})['kedgn']
            print('Set C paired intervals complete.',flush=True)
        results[name]=result
        baselines[name]=old
        primary[name]=ensemble
    cohort=pd.read_parquet(OUT/'eicu_demo_external/cohort.parquet')
    for tag in NAMES:
        modes=('hourly','native')
        bounds=paired_intervals(cohort,{mode:baselines[f'eicu_{mode}'][tag] for mode in modes},
                                {mode:primary[f'eicu_{mode}'] for mode in modes})
        for mode,value in bounds.items(): results[f'eicu_{mode}'].setdefault('paired_ci',{})[tag]=value
        print(f'Patient-paired eICU intervals complete: {tag}',flush=True)
    summary=dict(training_stays=8000,training_deaths=1122,training_fits=8,evaluation_fits=0,
                 selected_epoch=selection['selected_epoch'],selection=selection,protocol=protocol,
                 runtime_verification=read(root/'runtime_verification.json'),
                 results=results,primary_prediction=inference['primary_prediction'],
                 bootstrap=dict(set_c=dict(replicates=2000,seed=20261013,unit='stay'),
                                eicu=dict(replicates=1000,seed=42,unit='patient')),
                 source_sha256={name:sha(root/name) for name in ('inputs/manifest.json','selection.json','protocol.json','frozen.json','inference.json')},
                 p53_summary_sha256=sha(P53/'summary.json'),
                 final_weights_sha256={str(seed):frozen['seeds'][str(seed)]['weights_sha256'] for seed in range(5)},
                 environment=(root/'environment.txt').read_text(encoding='utf-8-sig').splitlines())
    destination.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value['ensemble'] for key,value in results.items()},indent=2),flush=True)


def report(summary,folder):
    def interval(bounds): return f'[{bounds[0]:+.5f}, {bounds[1]:+.5f}]'
    lines=['# P54：A+B 重訓 KEDGN 的凍結轉移結果','',
           '依使用者指示，以 P12 A/B 開發並完成五種子重訓後，再評估 Set C 與 eICU Demo。訓練可使用 GPU，正式比較沒有新增擬合，R1 不復現。資料、折外機率、逐筆預測與權重留在 Git 排除目錄，本報告及 JSON 僅含整體統計。','',
           '## A/B 選擇與凍結','',
           f'使用原 A/B 8,000 次住院、1,122 次死亡及既有三折。固定作者已復現的參數，只在第 1–10 輪間依三折合併折外 AP 選輪次，選出 **{summary["selected_epoch"]} 輪**。每折縮放只由其訓練部分估計，全 A+B 縮放另行估計。選擇完成後訓練 seed 0–4，主結果為五種子機率等權平均；總計 3 次開發及 5 次最終擬合。所有選擇均在載入測試預測前封存。','',
           '| 輪次 | pooled OOF AP | pooled OOF AUROC | 三折 AP |','| --- | ---: | ---: | --- |']
    for row in summary['selection']['curve']:
        lines.append(f'| {row["epoch"]} | {row["pooled_oof_ap"]:.5f} | {row["pooled_oof_auroc"]:.5f} | '+ '／'.join(f'{v:.5f}' for v in row['fold_ap'])+' |')
    lines+=['','## 主要結果與配對比較','',
            'AP、AUROC 越高越好，Brier 越低越好。敏感度沿用固定 0.5 門檻，未在測試集校準或選門檻。差值方向為 A+B KEDGN 減對照。Set C 用 2,000 次住院配對重抽；eICU 用 1,000 次患者分組配對重抽，沿用 P53 種子。區間未作多重比較修正。','']
    for name,label in COHORTS.items():
        row=summary['results'][name]
        lines += [f'### {label}（{row["stays"]} 次住院／{row["deaths"]} 次死亡）','',
                  '| 模型 | AP | AUROC | Brier | 敏感度 @0.5 |','| --- | ---: | ---: | ---: | ---: |']
        for tag,metrics in [('ab',row['ensemble']),*row['controls'].items()]:
            title='A+B KEDGN（五種子平均）' if tag=='ab' else NAMES[tag]
            lines.append('| '+title+' | '+' | '.join(f'{metrics[k]:.5f}' for k in ('ap','auroc','brier','sensitivity'))+' |')
        lines+=['','| 對照 | ΔAP | 95% 區間 | ΔAUROC | 95% 區間 |','| --- | ---: | --- | ---: | --- |']
        for tag,delta in row['delta'].items():
            bounds=row['paired_ci'][tag]
            lines.append(f'| {NAMES[tag]} | {delta["ap"]:+.5f} | {interval(bounds["ap"])} | {delta["auroc"]:+.5f} | {interval(bounds["auroc"])} |')
        lines+=['']
    lines+=['## 種子結果','', '| 種子 | Set C AP | Set C AUROC | eICU 整點 AP | eICU 整點 AUROC | eICU 原始 AP | eICU 原始 AUROC |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in range(5):
        scores=[summary['results'][name]['individual_seeds'][seed][metric] for name in COHORTS for metric in ('ap','auroc')]
        lines.append(f'| {seed} | '+' | '.join(f'{v:.5f}' for v in scores)+' |')
    lines+=['','五種子指標平均／SD 及逐種子 Brier、敏感度完整保存於彙總 JSON。平均機率的指標與種子指標平均分別保存。','',
            '## 判讀與限制','',
            '本輪與本組原模型的訓練病例量皆為 A+B 8,000 筆，但 KEDGN 使用作者 36 通道、9 維靜態和正值 mask，表格模型使用不同特徵表示。保留作者平衡採樣與有效 batch 256。GPU 使用等價遮罩圖運算及 batch 256，正式訓練前核對與原作者的 logits、全部有效參數梯度，以及整批／分批梯度；FP32、關閉 TF32，啟用確定性運算。','',
            'P53 Set A 模型只有 3,198 筆訓練，且以作者 400 筆驗證選 checkpoint；本輪以三折選共同輪次。差值同時包含訓練資料、選擇方式與五種子訓練變異，不宜單獨歸因於訓練量。作者驗證程式用 sigmoid 單類 logit，本輪沿最終測試用雙類 softmax，輪次選擇也一致使用 softmax AP。','',
            'Set C 使用 P53 相同完整 4,000 筆，與 A/B ID 互斥；eICU 使用相同尿量 mL 對齊及兩種取樣，長序列不截斷。Set C 已反覆查看，eICU Demo 僅 91 次死亡；固定模型的配對區間未涵蓋重新訓練及跨醫院變異。P45 使用 Demo 做增強、P52 使用 24 小時窗，未在本輪 48 小時純轉移比較中混合排名。本輪沒有新增遮蔽測試。','',
            '## 重跑與來源','',
            '[A+B 重訓規格及命令](../related_work/KEDGN/AB_RETRAIN_PROTOCOL.md) · [Set A 凍結版 P53](P53_kedgn_frozen_transfer_results.md) · [整體彙總 JSON](P54_kedgn_ab_transfer_summary.json)','',
            '本機產物：`ML/artifacts/related_work/KEDGN/ab_dense_gpu_20261121/`。inputs 保存 A/B 與原三折，protocol／selection／frozen 依序封存規則、輪次及最終權重；各模型有每輪恢復 checkpoint，inference 保存測試預測雜湊，summary 保存配對結果。完整環境清單包含於 JSON。程式與結果分開提交；資料及模型不提交 Git。','']
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'P54_kedgn_ab_transfer_results.md').write_text('\n'.join(lines),encoding='utf-8')
    (folder/'P54_kedgn_ab_transfer_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT/'related_work/KEDGN/ab_dense_gpu_20261121')
    parser.add_argument('--report-directory',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if not (args.output/'summary.json').exists(): main(args.output)
        if args.report_directory: report(read(args.output/'summary.json'),args.report_directory)
