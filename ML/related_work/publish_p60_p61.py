"""Publish all seed/mask results, variability decomposition and fixed-model paired CIs."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import numpy as np
from sklearn.metrics import average_precision_score
from stress_missingness_c import score_metrics
from p60_seed_stability import ROOT,SEEDS,MODELS,read
from p61_missingness_inputs import ROOT as STRESS
from prepare import OUT,sha

REPORT=ML/'reports'
METRICS=('ap','auroc','brier','sensitivity')
KG='kedgn_ab'
NAMES={'frozen_full':'Full','frozen_elastic':'Elastic','gcs_only_full':'GCS-only Full',
       'gcs_only_elastic':'GCS-only Elastic','p26':'P26','p28':'P28','p33':'P33',KG:'KEDGN A+B'}

def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def md(path,lines): path.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')
def stat(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if a.size>1 else 0.,minimum=float(a.min()),maximum=float(a.max()))
def interval(values): return np.quantile(values,[.025,.975]).tolist()

def load(seed_only=False):
    evaluation=read(ROOT/'evaluation.json');masks=read(STRESS/'inputs.json')
    if seed_only:
        kedgn=read(OUT/'p55_kedgn_missingness/kedgn_manifest.json')
    else: kedgn=read(STRESS/'kedgn_evaluation.json')
    frozen=read(ROOT/'frozen.json')
    assert sha(ROOT/'frozen.json')==evaluation['frozen_sha256']
    assert sha(STRESS/'inputs.json')==evaluation['stress_inputs_sha256']
    if not seed_only: assert sha(STRESS/'inputs.json')==kedgn['input_manifest_sha256']
    assert len(frozen['seeds'])==5 and frozen['fit_count']==35 and len(masks['cases'])==39
    for s in SEEDS:
        for file,digest in frozen['seeds'][str(s)]['model_sha256'].items(): assert sha(ROOT/f'seed_{s}'/file)==digest
    data={};outcomes={};specs={c['name']:c for c in masks['cases']}
    for case,receipt in evaluation['cases'].items():
        if seed_only and case not in ('clean','gcs_full','six_full','eicu_hourly','eicu_native'): continue
        path=ML/receipt['file'];assert sha(path)==receipt['sha256']
        with np.load(path) as source:
            ids,y=source['ids'],source['y'];values=source['predictions']
            np.testing.assert_array_equal(source['seeds'],SEEDS);np.testing.assert_array_equal(source['models'],MODELS)
            assert values.shape==(5,7,len(y))
        if case.startswith('eicu_'):
            kgpath=OUT/f'related_work/KEDGN/ab_dense_gpu_20261121/{case}_predictions.npz'
        elif seed_only:
            tag={'clean':'clean','gcs_full':'full_GCS','six_full':'full_all_six'}[case]
            kgpath=OUT/f'p55_kedgn_missingness/ab_{tag}.npz'
            assert sha(kgpath)==kedgn['predictions'][kgpath.name]
        else:
            kgpath=STRESS/case/'kedgn_predictions.npz'
            assert sha(kgpath)==kedgn['cases'][case]['sha256']
        with np.load(kgpath) as source:
            np.testing.assert_array_equal(source['ids'],ids);np.testing.assert_array_equal(source['y'],y)
            probabilities=source['probabilities'].copy()
        data[case]={**{m:values[:,i] for i,m in enumerate(MODELS)},KG:probabilities}
        assert probabilities.shape==(5,len(y))
        outcomes[case]=y
    paths=[ROOT/'frozen.json',ROOT/'evaluation.json',STRESS/'inputs.json',
           OUT/'related_work/KEDGN/ab_dense_gpu_20261121/frozen.json',OUT/'p55_kedgn_missingness/kedgn_manifest.json']
    if not seed_only: paths.append(STRESS/'kedgn_evaluation.json')
    sources={str(p.relative_to(ML)):sha(p) for p in paths}
    return data,outcomes,specs,sources,frozen,kedgn

def seed_report(data,outcomes,sources,frozen):
    cases=('clean','gcs_full','six_full','eicu_hourly','eicu_native')
    rows=[];summary={};paired={}
    for case in cases:
        y=outcomes[case];summary[case]={};paired[case]={}
        for model,probs in data[case].items():
            results=[score_metrics(y,p) for p in probs]
            summary[case][model]=dict(metrics={k:stat([r[k] for r in results]) for k in METRICS},
                ensemble=score_metrics(y,probs.mean(axis=0,dtype=np.float64)))
            for index,m in enumerate(results):
                rows.append(dict(cohort=case,model=model,seed=int(index if model==KG else SEEDS[index]),
                    stays=len(y),deaths=int(y.sum()),**m))
        for model in MODELS:
            paired[case][model]={}
            for ref in ('frozen_full','p28'):
                differences=[score_metrics(y,a)['ap']-score_metrics(y,b)['ap'] for a,b in zip(data[case][model],data[case][ref])]
                paired[case][model][ref]=dict(**stat(differences),positive_seeds=int((np.asarray(differences)>0).sum()),values=differences)
    for case in ('gcs_full','six_full'):
        for model in MODELS:
            changes=[score_metrics(outcomes[case],p)['ap']-score_metrics(outcomes['clean'],c)['ap']
                     for p,c in zip(data[case][model],data['clean'][model])]
            summary[case][model]['ap_loss_from_clean']=dict(**stat(changes),values=changes)
    devices=[frozen['seeds'][str(seed)]['cnn']['device'] for seed in SEEDS]
    report=dict(seeds=list(SEEDS),kedgn_existing_seeds=list(range(5)),new_component_fits=35,
       new_selected_parameters=0,selected_seed=None,training_data='P12 A+B 8000/1122',cnn_training_devices=devices,
       per_seed=rows,summary=summary,paired_training_seed_ap=paired,
       source_sha256={k:v for k,v in sources.items() if not k.endswith('kedgn_evaluation.json')},
       limitations=['Five runs describe initialization/sampling variability, not independent test samples',
          'Augmentation locations and feature selection fixed','Repeatedly viewed Set C and Demo',
          'KEDGN seed labels are not paired to our seed labels','Mean of five metrics differs from score of mean probabilities'])
    write(REPORT/'P60_five_seed_stability_summary.json',report)
    csv_write(REPORT/'P60_all_seed_results.csv',rows)
    lines=['# P60：五種子固定配置穩定性','',
      '所有五套管線先凍結再測試。A/B 8,000／1,122，固定參數與原增強位置，新增35次元件擬合；沒有選最佳種子或重新調參。CNN五次訓練均使用CPU，樹模型也用原CPU演算法。KEDGN使用已凍結的原A+B五種子，不增加擬合。','',
      '## AP 平均 ± 樣本標準差','',
      '| 模型 | 未遮蔽 C | GCS 全遮 C | 六欄全遮 C | Demo 整點 | Demo 原始間隔 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in NAMES:
        cells=[summary[c][model]['metrics']['ap'] for c in cases]
        lines.append('| '+NAMES[model]+' | '+' | '.join(f'{v["mean"]:.5f} ± {v["sd"]:.5f}' for v in cells)+' |')
    lines+=['','## P33 對主要對照：每個種子都列入','',
      '| 情境 | 對照 | 平均 ΔAP | 種子差 SD | 最小／最大 ΔAP | 正向種子數 |',
      '| --- | --- | ---: | ---: | --- | ---: |']
    for case in cases:
        for ref in ('frozen_full','p28'):
            s=paired[case]['p33'][ref]
            lines.append(f'| {case} | {NAMES[ref]} | {s["mean"]:+.5f} | {s["sd"]:.5f} | {s["minimum"]:+.5f}／{s["maximum"]:+.5f} | {s["positive_seeds"]}/5 |')
    lines+=['','原Full沒有隨機行／欄抽樣，若跨種子分數相同屬其固定演算法的結果；不藉此複製成五份獨立測試。P26/P28/P33共享同種子的元件，差值亦有相依性。五次指標平均是穩定性描述，五種子機率平均的集成另存在JSON，不用較高者替換主結果。','',
      '## 穩定性與取捨','',
      'P33對Full的正向種子數（未遮蔽／GCS全遮／六欄全遮）：'+
          '／'.join(str(paired[c]['p33']['frozen_full']['positive_seeds'])+'/5' for c in cases[:3])+ '。對P28依相同順序為：'+
          '／'.join(str(paired[c]['p33']['p28']['positive_seeds'])+'/5' for c in cases[:3])+ '。Demo整點／原始間隔對P28為：'+
          '／'.join(str(paired[c]['p33']['p28']['positive_seeds'])+'/5' for c in cases[3:])+ '。需同時呈現增益與取捨，不把CNN的未遮蔽收益延伸為所有情境的收益。','',
      '五種子不是五個新外部資料集，不能替代患者抽樣區間或獨立資料確認。原P33的0.60447是先前單一凍結模型分數；本輪'+
          f'{summary["clean"]["p33"]["metrics"]["ap"]["mean"]:.5f}是事先固定五種子的指標平均，兩者分開保存。','',
      '每個種子的AP、AUROC、Brier、@0.5敏感度與所有範圍見CSV／JSON；本表SD不是患者bootstrap的95%區間，也未涵蓋重新選欄、重新調參及全部研究搜尋。','',
      '[固定規格](P60_five_seed_stability_protocol.md) · [全部200列種子結果](P60_all_seed_results.csv) · [完整JSON](P60_five_seed_stability_summary.json) · [連續與漸進缺失](P61_progressive_block_results.md)']
    md(REPORT/'P60_five_seed_stability_results.md',lines)
    print('Published all 200 individual seeded cohort metrics.',flush=True)
    return report

def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def stress_report(data,outcomes,specs,sources,kedgn):
    rows=[];cells={};measurements={}
    for case,description in specs.items():
        y=outcomes[case];measurements[case]={}
        for model,values in data[case].items():
            metrics=[score_metrics(y,p) for p in values];measurements[case][model]=metrics
            for s,m in enumerate(metrics):
                clean=score_metrics(outcomes['clean'],data['clean'][model][s])
                rows.append(dict(case=case,target=description['target'],design=description['design'],hours=description['hours'],
                   repeat=description['repeat'],model=model,seed=int(s if model==KG else SEEDS[s]),
                   actual_removed_fraction=description['actual_removed_fraction'],**m,ap_change_from_clean=m['ap']-clean['ap']))
    for target in ('gcs','six'):
        for design in ('random','block'):
            for length in (12,24,36):
                key=f'{target}_{design}_{length}';cells[key]={}
                for model in NAMES:
                    values=np.asarray([[measurements[f'{key}_r{r}'][model][s]['ap'] for r in range(3)] for s in range(5)])
                    cells[key][model]=dict(mean=float(values.mean()),training_seed_sd=float(values.mean(1).std(ddof=1)),
                      mask_position_sd=float(values.mean(0).std(ddof=1)),overall_descriptive_sd=float(values.std(ddof=1)),
                      minimum=float(values.min()),maximum=float(values.max()),seed_by_repeat=values.tolist())
    csv_write(REPORT/'P61_all_seed_mask_results.csv',rows)
    # Shared patient draws; average three mask-specific APs, never pool repeated patients.
    y=outcomes['clean'];rng=np.random.default_rng(2026112262)
    draws=rng.integers(0,len(y),size=(1000,len(y)),dtype=np.int32)
    focus=('frozen_full','p28','p33',KG)
    ensemble={case:{model:values.mean(0,dtype=np.float64) for model,values in models.items()} for case,models in data.items()}
    clean_samples={m:np.asarray([average_precision_score(y[idx],ensemble['clean'][m][idx]) for idx in draws]) for m in focus}
    samples={};contrasts={};design_contrasts={}
    for target in ('gcs','six'):
        for design in ('random','block'):
            key=f'{target}_{design}_24';names=[f'{key}_r{r}' for r in range(3)]
            samples[key]={m:np.asarray([np.mean([average_precision_score(y[idx],ensemble[c][m][idx]) for c in names]) for idx in draws]) for m in focus}
            points={m:float(np.mean([average_precision_score(y,ensemble[c][m]) for c in names])) for m in focus}
            contrasts[key]={}
            for ref in ('frozen_full','p28',KG):
                delta=samples[key]['p33']-samples[key][ref]
                retention=delta-(clean_samples['p33']-clean_samples[ref])
                point=points['p33']-points[ref]
                cleanpoint=average_precision_score(y,ensemble['clean']['p33'])-average_precision_score(y,ensemble['clean'][ref])
                contrasts[key][ref]=dict(ap_delta=point,ap_ci=interval(delta),retention_gain=point-cleanpoint,
                                         retention_ci=interval(retention))
            print(f'Completed 1000 shared-patient bootstrap draws: {key}',flush=True)
        design_contrasts[target]={m:dict(block_minus_random_ap=float(np.mean([
            average_precision_score(y,ensemble[f'{target}_block_24_r{r}'][m])-average_precision_score(y,ensemble[f'{target}_random_24_r{r}'][m]) for r in range(3)])),
            ci=interval(samples[f'{target}_block_24'][m]-samples[f'{target}_random_24'][m])) for m in focus}
    report=dict(stays=len(y),deaths=int(y.sum()),new_fits=0,case_count=39,mask_repeats=3,
       model_seeds=list(SEEDS),kedgn_seeds=list(range(5)),individual_metric_rows=len(rows),
       cases=[{k:v for k,v in c.items() if k!='files'} for c in specs.values()],curves=cells,
       primary_50_percent_ensemble_contrasts=contrasts,block_vs_random_24h=design_contrasts,
       bootstrap=dict(replicates=1000,seed=2026112262,unit='stay',mask_handling='Mean of three APs within each paired stay draw',model_handling='Equal probability mean of all five frozen seeds'),
       source_sha256=sources,kedgn_devices={k:v['devices'] for k,v in kedgn['cases'].items()},
       limitations=['Clock-hour proportion differs from actual removed measurement fraction',
          'Three mask positions and five seeds are crossed, not fifteen independent test cohorts',
          'Fixed-model bootstrap excludes retraining/model-search uncertainty','Artificial missingness, not clinical causal effects',
          'Repeatedly viewed Set C; no new external dataset'])
    write(REPORT/'P61_progressive_block_summary.json',report)
    lines=['# P61：漸進缺失與連續區段缺失','',
      '固定Set C 4,000／585，39個案例、每種部分缺失三組位置；本組七模型各五種子與既有A+B KEDGN五種子全部列入，新增擬合0次。隨機小時與連續區段具有相同移除時鐘長度；真實刪除觀測比例仍不同，完整保存在總表。','',
      '## AP 曲線摘要：五種子 × 三位置平均','',
      '| 目標 | 設計 | 移除時鐘小時 | Full | P28 | P33 | KEDGN |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,models in cells.items():
        target,design,length=key.split('_')
        lines.append(f'| {target} | {design} | {length} | '+' | '.join(f'{models[m]["mean"]:.5f}' for m in focus)+' |')
    lines+=['','圖與JSON另分解訓練種子SD（各種子先平均三個位置）及遮蔽位置SD（各位置先平均五個種子）。平均曲線沒有挑最好種子，也不把15組結果當15個獨立測試。0%／100%端點及所有8模型的逐種子結果完整列於CSV。','',
      '## 預定50%比較：五種子機率平均的配對區間','',
      '| 情境 | P33 對照 | ΔAP | 95%區間 | 少退化的AP | 95%區間 |','| --- | --- | ---: | --- | ---: | --- |']
    for key,comparators in contrasts.items():
        for ref,r in comparators.items():
            lines.append(f'| {key} | {NAMES[ref]} | {r["ap_delta"]:+.5f} | [{r["ap_ci"][0]:+.5f}, {r["ap_ci"][1]:+.5f}] | {r["retention_gain"]:+.5f} | [{r["retention_ci"][0]:+.5f}, {r["retention_ci"][1]:+.5f}] |')
    lines+=['','區間來自1,000次同住院配對重抽，每次先對各位置算AP再平均，不把重複病例疊成更大隊列。這裡評估的是五模型等權機率集成；前表則是五次指標平均，兩者不能混為一談。區間只反映固定模型的病例抽樣，未校正歷史多重比較。','',
      '連續缺失可能保留較長的完整片段，隨機缺失可能打散序列；實際哪個較難由結果判讀，不預設連續一定更差。人工缺失無法完整反映臨床決策、介面與跨院分布差異。','',
      '![缺失曲線](../figures/P61_missingness_curves.png)','',
      '[事先規格](P61_progressive_block_protocol.md) · [全部1560列](P61_all_seed_mask_results.csv) · [完整JSON](P61_progressive_block_summary.json) · [五種子穩定性](P60_five_seed_stability_results.md)']
    md(REPORT/'P61_progressive_block_results.md',lines)
    return report

def plots(p60,p61):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.5,8),sharey=True)
    focus=('frozen_full','p28','p33',KG)
    colors=('#727e89','#159589','#e67f32','#8665ad')
    for ax,(target,design) in zip(axes.flat,[(t,d) for t in ('gcs','six') for d in ('random','block')]):
        full=f'{target}_full'
        for model,color in zip(focus,colors):
            points=[p60['summary']['clean'][model]['metrics']['ap']['mean']]
            spread=[p60['summary']['clean'][model]['metrics']['ap']['sd']]
            for length in (12,24,36):
                r=p61['curves'][f'{target}_{design}_{length}'][model]
                points.append(r['mean']);spread.append(r['training_seed_sd'])
            points.append(p60['summary'][full][model]['metrics']['ap']['mean']);spread.append(p60['summary'][full][model]['metrics']['ap']['sd'])
            x=np.arange(5)*25;values=np.asarray(points);sd=np.asarray(spread)
            ax.plot(x,values,'o-',label=NAMES[model],color=color,lw=1.7,ms=4)
            ax.fill_between(x,values-sd,values+sd,color=color,alpha=.12)
        ax.set_title(f'{target.upper()} / {design} hours');ax.set_xlabel('Removed clock hours (%)')
        ax.set_xticks([0,25,50,75,100]);ax.set_ylim(.32,.64);ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Average precision')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False)
    fig.text(.07,.015,'5 frozen training seeds; 3 fixed mask positions per interior point. Band: SD of seed means, not 95% CI.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(ML/'figures/P61_missingness_curves.png',dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['60','all'],default='all');args=parser.parse_args()
    data,y,specs,sources,frozen,kedgn=load(seed_only=args.phase=='60')
    p60=seed_report(data,y,sources,frozen)
    if args.phase=='60': return
    p61=stress_report(data,y,specs,sources,kedgn)
    plots(p60,p61)
    print('Published all seeds, all masks, descriptive variation and prespecified comparisons.',flush=True)

if __name__=='__main__': main()
