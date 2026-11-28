"""Publish aggregate KEDGN audit evidence, without changing frozen baseline scores."""
import ast,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,recall_score
from audit_kedgn_implementation import ML,ROOT,AB,P53,AUTHOR,UP,COMMIT,read,sha

REPORT=ML/'reports'
METRICS=('ap','auroc','brier','sensitivity')
def score(y,p):return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
    brier=float(brier_score_loss(y,p)),sensitivity=float(recall_score(y,p>=.5)))
def csvfile(name,rows):
    with (REPORT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def table(columns,rows):return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def privacy(x):
    if isinstance(x,dict):
        assert not set(x)&{'ids','y','probabilities','predictions','RecordID','patientunitstayid','uniquepid','stay'}
        for v in x.values():privacy(v)
    elif isinstance(x,list):
        for v in x:privacy(v)

def main():
    sources=read(ROOT/'sources.json');data=read(ROOT/'data.json');runtime=read(ROOT/'runtime.json')
    author=read(ROOT/'author_test_replay.json');demo=read(ROOT/'demo_replay.json');parser=read(ROOT/'parser_test_sensitivity.json')
    checks=read(ROOT/'checkpoint_checks.json');coverage=read(ROOT/'demo_input_coverage.json')
    assert runtime['complete'] and demo['complete'] and checks['complete']
    assert all(x==3997 for x in data['author_work_vs_independent_raw_reference'].values())
    assert len(runtime['Set_C_original_forward_all_five_models'])==5 and len(demo['rows'])==10 and len(author['rows'])==5
    assert len(parser['rows'])==10 and parser['new_fits']==0 and parser['selected_on_test'] is False
    before=read(REPORT/'P63_report_five_seed_summary.json')
    assert before['cross_seed_probability_averaging'] is False
    primary={(r['cohort'],r['model'],r['seed']):r for r in before['individual_metrics']}
    # Bridge all three generations of stored common-cohort scores.
    with np.load(AB/'set_c_predictions.npz') as a,np.load(ML/'artifacts/p55_kedgn_missingness/ab_clean.npz') as b,np.load(ML/'artifacts/p63_report_five_seed/evaluation/clean.npz') as c:
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['y'],b['y'])
        assert np.array_equal(a['probabilities'],b['probabilities'])
        assert np.array_equal(b['probabilities'],c['predictions'][list(c['models']).index('kedgn_ab')])
    all_rows=[]
    for row in author['rows']:
        all_rows.append(dict(group='author_split_reproduction',seed=row['seed'],training_stays=3198,stays=399,deaths=51,**{m:row[m] for m in METRICS}))
    with np.load(P53/'set_c_predictions.npz') as a:
        for seed,p in enumerate(a['probabilities']):all_rows.append(dict(group='author_frozen_to_Set_C',seed=seed,
            training_stays=3198,stays=len(a['y']),deaths=int(a['y'].sum()),**score(a['y'],p)))
    for case,group in [('clean','AB_adapted_Set_C'),('eicu_hourly','AB_adapted_Demo_hourly'),('eicu_native','AB_adapted_Demo_native')]:
        for seed in range(5):
            row=primary[case,'kedgn_ab',seed]
            all_rows.append(dict(group=group,seed=seed,training_stays=8000,stays=row['stays'],deaths=row['deaths'],**{m:row[m] for m in METRICS}))
    means=[]
    for group in dict.fromkeys(row['group'] for row in all_rows):
        subset=[row for row in all_rows if row['group']==group]
        means.append(dict(group=group,training_stays=subset[0]['training_stays'],stays=subset[0]['stays'],deaths=subset[0]['deaths'],
            **{m+s:float(statistics.mean([r[m] for r in subset]) if s=='_mean' else statistics.stdev([r[m] for r in subset])) for m in METRICS for s in ('_mean','_sd')}))
    lookup={r['group']:r for r in means}
    parser_summary=[]
    for version in ('set_a','ab'):
        part=[r for r in parser['rows'] if r['version']==version]
        parser_summary.append(dict(version=version,changed_test_stays=parser['changed_stays'],
            corrected_parser_ap_mean=statistics.mean([r['corrected_parser']['ap'] for r in part]),
            positional_test_parser_ap_mean=statistics.mean([r['author_positional_test_parser']['ap'] for r in part]),
            mean_delta_ap=statistics.mean([r['delta_ap'] for r in part]),
            maximum_absolute_seed_delta_ap=max(abs(r['delta_ap']) for r in part)))
    # Verify mask semantics of the existing Demo inputs, including valid recorded zero urine.
    for case in coverage:
        events=pd.read_parquet(ML/f'artifacts/related_work/common_sequence_inputs/{case}_events.parquet',columns=['RecordID','variable','clean_value'])
        urine=events.loc[events.variable.eq('Urine')]
        coverage[case]['Urine_recorded_stays']=int(urine.RecordID.nunique())
        assert coverage[case]['Urine_observed_stays']==int(urine.loc[urine.clean_value>0,'RecordID'].nunique())
        coverage[case]['Urine_recorded_but_nonpositive_only']=coverage[case]['Urine_recorded_stays']-coverage[case]['Urine_observed_stays']
    blacklist_source=ast.parse((UP/'data/physionet/process_scripts/remove_outliers.py').read_text())
    blacklist=next(ast.literal_eval(x.value) for x in blacklist_source.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='blacklist' for t in x.targets))
    with np.load(AB/'inputs/ab.npz') as a,np.load(P53/'set_c_labels.npz') as c:
        retained=dict(AB=len(set(map(int,blacklist))&set(a['ids'])),Set_C=len(set(map(int,blacklist))&set(c['ids'])))
    numeric=dict(AB_probe_cases=runtime['trained_model_checks'][0]['probe_cases'],
        trained_five_seed_logits_max_abs=max(r['logits_max_abs'] for r in runtime['trained_model_checks']),
        trained_five_seed_gradients_max_abs=max(r['gradient_max_abs'] for r in runtime['trained_model_checks']),
        three_update_logits_max_abs=max(r['post_update_logits_max_abs'] for r in runtime['original_vs_dense_optimizer_updates']),
        original_Set_C_probabilities_max_abs=max(r['probability_max_abs'] for r in runtime['Set_C_original_forward_all_five_models']),
        original_Set_C_ap_max_abs=max(abs(r['ap_original']-r['ap_cached']) for r in runtime['Set_C_original_forward_all_five_models']),
        original_Demo_probabilities_max_abs=max(r['maximum_probability_difference_from_original_cache'] for r in demo['rows']),
        original_Demo_ap_max_abs=max(abs(r['ap_difference_from_original_cache']) for r in demo['rows']),
        Demo_changed_order_pairs_within_numeric_interval=all(r['changed_pairs_confined_to_numeric_error_interval'] for r in demo['rows']))
    evidence={name:sha(ROOT/name) for name in ('sources.json','data.json','runtime.json','author_test_replay.json','demo_replay.json','parser_test_sensitivity.json','checkpoint_checks.json','demo_input_coverage.json')}
    code={p.name:sha(p) for p in [Path(__file__),ML/'related_work/audit_kedgn_implementation.py',ML/'related_work/replay_kedgn_original_cohorts.py']}
    result=dict(complete=True,date='2026-11-28',author_commit=COMMIT,primary_results_changed=False,
        new_formal_fits=0,diagnostic_optimizers=2,diagnostic_updates_per_optimizer=3,
        cross_seed_probability_averaging=False,sample_sd_ddof=1,individual_metrics=all_rows,mean_sd=means,
        paper_reference=dict(task='Physionet in-hospital mortality',ap_mean=.571,ap_sd_reported=.018,auroc_mean=.879),
        all_original_author_raw_preprocessing_records_exact=data['author_work_vs_independent_raw_reference'],
        bundled_original_PTdict_available=data['bundled_original_PTdict_available'],
        data_audit=data,author_blacklist_retained=retained,checkpoint_checks=checks['rows'],numeric_audit=numeric,
        original_cell_replay_Demo=demo['rows'],test_only_parser_sensitivity=parser_summary,Demo_coverage=coverage,
        original_model_source_exact=True,declared_CPU_compatibility_patches_only=True,
        original_text_tensor_sha256=sources['embedding_sha256'],code_sha256=code,private_evidence_sha256=evidence,
        primary_individual_csv_sha256=sha(REPORT/'P63_all_seed_results.csv'),primary_mean_sd_csv_sha256=sha(REPORT/'P63_mean_sd_results.csv'),
        unresolved=['Full CPU/GPU training-trajectory identity was not tested',
                    'No A/B architecture hyperparameter grid was re-searched',
                    'Contribution of per-seed checkpoint selection and validation-score definition was not isolated by retraining',
                    'Test-only positional parser sensitivity does not estimate the effect of changing the training parser',
                    'No fresh all-patient original-cell re-evaluation of GCS/six-channel mask cohorts in this audit',
                    'Author Torch1.12 requirement differs from local Torch2.1.2; exact historical authors environment not recreated'])
    privacy(result)
    (REPORT/'P64_kedgn_implementation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    csvfile('P64_kedgn_individual_metrics.csv',all_rows);csvfile('P64_kedgn_mean_sd.csv',means)
    fmt=lambda key:f'{lookup[key]["ap_mean"]:.5f} ± {lookup[key]["ap_sd"]:.5f}'
    parser_table=table(['凍結模型','現有名稱解析 AP 平均','改用作者位置解析 AP 平均','平均 ΔAP'],[
        ['作者 Set A 五權重' if r['version']=='set_a' else 'A+B 適配五權重',f'{r["corrected_parser_ap_mean"]:.5f}',f'{r["positional_test_parser_ap_mean"]:.5f}',f'{r["mean_delta_ap"]:+.6f}'] for r in parser_summary])
    lines=['# P64：KEDGN 論文／作者實作與本組流程稽核','',
        '**結論：有流程出入。原作者切分的復現接近論文；目前六模型報告中的 KEDGN A+B 是共同隊列適配版，不能稱為原封不動的論文訓練流程。** 全量重播未發現錯載權重或 GPU 改寫造成大幅評分錯誤，但不能把全部差距直接歸因資料集不同。','',
        '2026-11-28。保留原結果與權重，不用 Set C／Demo 選參或挑種子。新增正式訓練0次；兩個數值檢查用 optimizer 各做三次 A/B 小批次更新，其模型不參與任何正式測試。本報告全部分數為個別種子指標平均，不放跨種子機率集成。','',
        '## 1. 先把不同結果放回正確的測試隊列','',
        table(['設定','訓練病例','測試病例／死亡','AP 平均 ± 樣本SD'],[
            ['作者論文 Physionet 欄','作者 Set A 訓練切分','作者切分','0.571（原文另報SD 0.018）'],
            ['本機作者流程復現','3,198','399／51',fmt('author_split_reproduction')],
            ['同一批 Set A 凍結權重移至 Set C','3,198','4,000／585',fmt('author_frozen_to_Set_C')],
            ['本組 A+B 適配版','8,000','4,000／585',fmt('AB_adapted_Set_C')],
            ['A+B 適配版 → Demo 整點','8,000','782／91',fmt('AB_adapted_Demo_hourly')],
            ['A+B 適配版 → Demo 原始間隔','8,000','782／91',fmt('AB_adapted_Demo_native')]]),'',
        f'原作者切分的 AP 平均為 {lookup["author_split_reproduction"]["ap_mean"]:.8f}，與論文0.571相差 {(lookup["author_split_reproduction"]["ap_mean"]-.571)*100:+.3f} 個百分點。這是接近報值，並非完全重現相同隨機軌跡。原作者程式用 np.std 的預設 ddof=0，該次復現為SD {author["population_sd"]:.5f}；本報告統一改列 ddof=1，因此為 {author["sample_sd"]:.5f}，均值不變。','',
        '同一批 Set A 模型換成 Set C 後，五次 AP 平均已由約0.568變成約0.525，A+B適配版則約0.546。這證明不能用作者小型測試切分的報值直接當 Set C 的預期分數，但兩套訓練流程也有以下變更，無法只將差距歸因訓練量或資料移轉。Set A→Set C列沿既有原作者運算快取重新計算；作者399筆、A+B Set C及Demo則另做本輪原始cell重播。','',
        '原文表1的 Physionet 是住院死亡任務；另一欄 P12 的標籤是住院時間超過三天，不是同一目標。原文五種子及驗證選模見 [論文§5.1、表1與附錄B/E](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c04aea54c2a60a632a47bd451cd2849-Paper-Conference.pdf)。','',
        '## 2. 已確認一致的部分','',
        f'- 固定作者 commit `{COMMIT}`，本機 upstream 的模型、訓練與工具檔無未提交差異。原作者CPU復現副本的 model.py 完全一致，train.py／utils.py 僅宣告的CUDA→CPU及權重載入位置修改。',
        '- 使用 KEDGN-ChatGPT：作者附的 ChatGPT描述／BERT向量，36個動態通道、9維靜態；向量檔與欄位順序逐一核對相同。hidden8、query5、node9、rarity alpha1、BERT768、Adam lr0.001、有效batch256沿作者Physionet配置。',
        '- 原始作者切分復現的全部3,997筆，對照獨立按原解析腳本重建的動態值、時間、靜態、擴展靜態與長度，五項全部逐值一致；標籤及作者陣列順序亦一致。作者repo未附PTdict_list.npy，因此此項是依原碼全量重建核對，不宣稱與不存在的作者原始陣列逐值比較。',
        '- 全A+B訓練統計與作者getStats／靜態處理核對相同；分塊張量化與直接原作者工具逐值一致。作者的正值mask、缺失零、靜態標準化後非正值歸零等行為均保留，沒有在此次稽核自行修正。',
        '- 最終五套權重均與最後訓練狀態完全一致，各完成9輪、每輪26批、共234次Adam更新；不是未訓練權重或誤載其他輪次。P54→P55→P63的Set C五次機率、病例及標籤相同。','',
        '## 3. 真正的出入與其意義','',
        table(['項目','作者原碼／原設定','本組A+B版','判讀'],[
            ['訓練／測試切分','Set A內固定80/10/10','A/B三折開發，全A+B重訓，Set C測試','共同隊列適配；不是同切分復現'],
            ['選checkpoint','每個種子各用驗證AP選最佳','seed0三折合併OOF選共同9輪，五種子固定9輪','不等同逐種子驗證選模'],
            ['驗證分數','sigmoid(正類logit)；最終測試才雙類softmax','選輪次及測試都用雙類softmax','排序可能不同；本輪未重訓隔離其影響'],
            ['原始欄位解析','前五列依位置當靜態；之後才讀動態','依名稱找入院靜態；掃描全部動態列','A/B189筆、Set C90筆的值有變化'],
            ['作者黑名單','刪除指定零觀測病例','保留共同隊列全部病例','A/B保留10筆、Set C保留2筆'],
            ['架構超參數搜尋','論文列出h/k、q、n、alpha網格搜尋','固定作者Physionet參數，只用A/B選輪次','不能稱已在A/B充分重新調優'],
            ['依賴／裝置','repo requirements列Torch1.12.0+cu116','作者流程復現Torch2.1.2 CPU；A+B Torch2.1.2+cu121','非相同歷史環境；不能排除訓練軌跡差異'],
            ['種子SD算法','np.std預設ddof0','報告統一ddof1','影響SD，不影響AP均值']]),'',
        'A/B解析差異：Set A90筆、Set B99筆的動態值與擴展靜態皆異於作者原解析；其中23筆的時間／序列長度也不同。Set C90筆有動態／靜態差異，其中19筆長度或時間不同。修復並非只改欄位順序，也可能保留原作者因「跳過前五列」而遺漏的量測。','',
        '**先前只抽30筆核對不足以支持「整體前處理完全相同」。** 本輪已擴展為全量12,000筆原始紀錄與現有快取核對；原3,997筆作者流程復現另逐值核對，界線應如此記錄。解析修復可以合理，但屬明確適配，不能與嚴格原流程混稱。','',
        '作者驗證／測試程式差異與五次均值算法見 [固定版本 train.py](https://github.com/qianlima-lab/KEDGN/blob/'+COMMIT+'/train.py#L222-L280)。論文附錄E說明架構參數網格，本組未重新搜尋該網格。','',
        '## 4. 原始cell與GPU改寫的實際核對','',
        table(['檢查','本輪結果'],[
            ['五套已訓練權重的原始／改寫輸出',f'{numeric["AB_probe_cases"]}筆A/B探針，最大logit差 {numeric["trained_five_seed_logits_max_abs"]:.3g}'],
            ['五套已訓練權重的梯度',f'有效與無梯度參數逐一核對，最大絕對差 {numeric["trained_five_seed_gradients_max_abs"]:.3g}'],
            ['相同初始化、同批次的三次Adam更新',f'更新後最大logit差 {numeric["three_update_logits_max_abs"]:.3g}'],
            ['原始cell重算全部Set C，五套權重',f'20,000次病例預測；最大機率差 {numeric["original_Set_C_probabilities_max_abs"]:.3g}，最大單種子ΔAP {numeric["original_Set_C_ap_max_abs"]:.3g}'],
            ['原始cell重算兩種Demo，五套權重',f'7,820次病例預測；最大機率差 {numeric["original_Demo_probabilities_max_abs"]:.3g}，最大單種子ΔAP {numeric["original_Demo_ap_max_abs"]:.3g}'],
            ['原作者399筆測試重新重播','五個種子與既有原始CPU紀錄一致']]),'',
        'Demo另檢查所有分數順序／同分變化：受影響病例對的原分數差均落在逐筆數值誤差可改變排序的範圍內。AP在同分處不連續，故本報告保留實際ΔAP，沒有因門檻失敗就聲稱完全相同，也不改寫正式P63分數。這些微小差值不能解釋約0.57到0.54的落差。','',
        '以上排除了本輪主隊列中錯載checkpoint、通道順序錯誤或前向GPU改寫造成大幅錯算的解釋。三次更新不等於整段9輪訓練軌跡的CPU/GPU等同性證明；亦未在本輪重新跑完所有GCS／六欄遮蔽的原始cell推論。','',
        '## 5. 解析差異究竟影響多少？先做凍結權重的測試側診斷','',
        f'只將Set C的{parser["changed_stays"]}筆差異紀錄改成作者「依前五列位置」解析，其餘病例、所有權重、訓練統計、標籤與評估方式固定；對五種子逐一算AP，不做集成。','',parser_table,'',
        '**這只量到測試側解析敏感度。** A+B權重是在修復解析後訓練，作者Set A權重則按原位置解析訓練；這張表不能估計「連訓練解析也一起改掉」的總影響，不能用較高測試分數選取新正式版本。本輪所有主結果維持原值。','',
        '## 6. Demo輸入也有核對','',
        table(['取樣','GCS有正值觀測','尿量有紀錄／有正值','超過215個時間點'],[
            [name,str(row['GCS_observed_stays'])+'/782',str(row['Urine_recorded_stays'])+'/'+str(row['Urine_observed_stays']),str(row['stays_over_215'])+'/782'] for name,row in coverage.items()]),'',
        '尿量已映射：545筆有紀錄，其中541筆有正值；另外4筆只有零值，按作者正值mask視為未觀測。這不是尿量全缺的舊映射。Demo有較長序列，程式保留全部時間點；這是新資料庫的輸入分布，原論文未提供同一Demo外部報值可直接對照。','',
        '## 7. 現在能說與仍不能說的事','',
        '- 可以說：原作者死亡任務切分的五種子已接近論文報值；目前A+B權重的共同隊列分數可由原作者運算重新得到。',
        '- 應改稱：共同主表的KEDGN是「固定作者架構／參數的A+B適配版」，並揭露解析、病例保留及選模流程差異。',
        '- 不能說：已證明0.5457的全部差距只來自資料集、解析或GPU；也不能藉此宣稱本組全面勝過充分調優的KEDGN。',
        '- 若要隔離原因，需要事先固定的A/B訓練對照，分別變更解析／checkpoint規則，重新估計訓練統計並只在A/B做選擇。此次未執行這類新正式擬合，也未重新搜尋架構參數。','',
        '補充：作者原碼的候選分支與門控更新細節亦不逐項等同論文公式（候選分支沒有再次做圖矩陣乘法）。本組保留作者公開程式，沒有自行改成另一個公式版本後混入復現；原碼見 [model.py](https://github.com/qianlima-lab/KEDGN/blob/'+COMMIT+'/model.py)。','',
        '## 8. 證據與重跑','',
        '[完整稽核JSON](P64_kedgn_implementation_audit.json) · [25列逐種子指標](P64_kedgn_individual_metrics.csv) · [五組平均／SD](P64_kedgn_mean_sd.csv) · [正式六模型結果](P63_report_five_seed_results.md)','',
        '從DSP根目錄執行，原模型／輸入需已在本機。診斷只新增本機artifacts/p64_kedgn_audit，不覆寫既有權重或評分。','',
        '```powershell',
        '& ML/.venv/Scripts/python.exe ML/related_work/audit_kedgn_implementation.py sources',
        '& ML/.venv/Scripts/python.exe ML/related_work/audit_kedgn_implementation.py data',
        '& ML/artifacts/related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe ML/related_work/audit_kedgn_implementation.py runtime --device cuda:0',
        '& ML/artifacts/related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe ML/related_work/replay_kedgn_original_cohorts.py',
        '& ML/.venv/Scripts/python.exe ML/related_work/publish_kedgn_implementation_audit.py',
        '```','',
        '資料、逐筆預測、差異病例索引、模型與正式報告PDF不進Git。公開檔僅含整體數值、來源雜湊及方法差異；歷史結果不改寫。']
    (REPORT/'P64_kedgn_implementation_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    assert sha(REPORT/'P63_all_seed_results.csv')==result['primary_individual_csv_sha256']
    assert sha(REPORT/'P63_mean_sd_results.csv')==result['primary_mean_sd_csv_sha256']
    print(json.dumps(dict(complete=True,new_formal_fits=0,author_split_AP=lookup['author_split_reproduction']['ap_mean'],
        AB_adapted_Set_C_AP=lookup['AB_adapted_Set_C']['ap_mean'],numeric=numeric,parser_sensitivity=parser_summary),indent=2),flush=True)

if __name__=='__main__':main()
