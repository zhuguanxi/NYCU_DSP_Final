"""Publish aggregate-only P62 evidence; never fit or modify the local final report."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
ML=Path(__file__).resolve().parents[1]
ROOT=ML/'artifacts/p62_ma_gbt_matched'
REPORT=ML/'reports'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def privacy(x):
    if isinstance(x,dict):
        assert not set(x)&{'ids','y','probabilities','patientunitstayid','uniquepid','RecordID'}
        for value in x.values():privacy(value)
    elif isinstance(x,list):
        for value in x:privacy(value)
def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
def alpha(a):return '關閉懲罰' if a is None else format(a,'g')
def label(tag):
    if tag.startswith('XGB_'):return tag.replace('XGB_GCS_','XGBoost GCS-only ').replace('XGB_','XGBoost ')
    name,aug,policy=tag.split('_')
    return name+'／'+('GCS60' if aug=='gcs60' else '無增強')+'／'+{'off':'無懲罰 GBT','ap':'MA-AP','rho95':'MA-ρ95'}[policy]
def main():
    evidence=read(ROOT/'evaluation/summary.json');frozen=read(ROOT/'frozen.json');selection=read(ROOT/'selection.json');protocol=read(ROOT/'protocol.json')
    assert evidence['complete'] and evidence['cv_fits']==frozen['cv_fits']==96 and not evidence['test_selected']
    assert evidence['frozen_sha256']==sha(ROOT/'frozen.json')
    assert sha(ROOT/'numeric_check.json')==protocol['numeric_check_sha256']
    for p,digest in protocol['source_sha256'].items():assert sha(ML/p)==digest
    for receipt in frozen['fits']:assert sha(ROOT/'models'/f'{receipt["key"]}.joblib')==receipt['model_sha256']
    # Everything in these files is a cohort/candidate/model aggregate.
    public={**evidence,'ab_candidates':selection['candidates'],'numeric_check':read(ROOT/'numeric_check.json'),
      'formal_final_fit_count':frozen['final_fits'],'diagnostic_fits':38,
      'publication_date':'2026-11-26','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ML,text=True).strip()}
    execution=dict(main_workers=6,helper_workers=0,gpu_models_trained=0,gpu_probe='synthetic compatibility check failed; no model trained on GPU')
    if (ROOT/'assist_execution.json').exists():
        legacy=read(ROOT/'assist_execution.json')
        execution['prior_assist']=legacy
        assert legacy['protocol_sha256']==sha(ROOT/'protocol.json')
        duplicates=0
        for helper in (ROOT/'assist_elastic_gcs/jobs').glob('*.json'):
            a=read(helper);b=read(ROOT/'jobs'/helper.name)
            for field in ('key','features','augmentation','alpha','fold','validation','protocol_sha256'):
                assert a[field]==b[field],(helper.name,field)
            duplicates+=a!=b
        execution['duplicate_cv_attempts']=duplicates
    if (ROOT/'resource_assignment.json').exists():execution['prior_cpu_assignment']=read(ROOT/'resource_assignment.json')
    if (ROOT/'execution_restart.json').exists():execution['restart']=read(ROOT/'execution_restart.json')
    if (ROOT/'parallel_resume_execution.json').exists():
        resume=read(ROOT/'parallel_resume_execution.json')
        for name,digest in resume['source_sha256'].items():assert sha(ML/'src'/name)==digest
        assert sha(ROOT/'parallel_numeric_check.json')==resume['numeric_check_sha256']
        execution['current_parallel_backend']=resume
        public['parallel_numeric_check']=read(ROOT/'parallel_numeric_check.json')
    execution['diagnostic_fit_note']='10 original equivalence/throughput fits; 9 unused prefix-batch fits; 9 initial parallel fits; 10 paired optimized-parallel fits. All A/B only.'
    public['execution']=execution
    public['cv_unique_jobs']=96
    privacy(public);save(REPORT/'P62_ma_gbt_matched_summary.json',public)
    primary=evidence['primary'];write_csv(REPORT/'P62_primary_metrics.csv',primary);write_csv(REPORT/'P62_all_stress_metrics.csv',evidence['stress'])
    cv=[]
    for r in selection['candidates']:
        flat={k:r[k] for k in ('features','augmentation','alpha')}
        for view,metrics in r['views'].items():flat.update({view+'_'+k:v for k,v in metrics.items()})
        cv.append(flat)
    write_csv(REPORT/'P62_ab_candidates.csv',cv)
    lookup={(r['cohort'],r['model']):r for r in primary}
    chosen={(r['features'],r['augmentation'],r['policy']):r for r in evidence['selected']}
    lines=['# P62：MA-GBT 相同容量與無懲罰控制結果','',
      '本輪依要求補足容量與控制差異。全部新模型固定 **300 棵、深度 3、學習率 0.05**，沿原 A/B 三折與 Full／Elastic 補值／編碼／欄位。全程 CPU、seed42；沒有用 Set C 或 Demo 選參。正式期末報告維持本機暫不公開。','',
      '## 設計與口徑','',
      '無懲罰 GBT 與 MA-GBT 使用同一作者實作；前者訓練不傳缺失懲罰矩陣，已核對與全零矩陣的樹相同。評估補值依賴一律傳真實缺失矩陣，因此無懲罰組的 rho 並不是零。這是同實作的懲罰消融，不把它冒稱 sklearn 或 XGBoost 的相同實作。','',
      '每組分無增強與 GCS60 加權副本；兩者使用完全相同的副本抽樣、權重、標籤與前處理。Full／Elastic、兩種增強條件各自用 A/B 在七個 alpha 中選擇。MA-AP 依平均驗證 AP；MA-ρ95 依候選內 AUROC 保留 95% 再最低 rho。兩種口徑全部報告，不依測試結果擇一。','',
      f'完成 **96 個唯一交叉驗證工作＋{frozen["final_fits"]} 次去重後最終訓練**；另有 38 次 A/B 數值等價／加速診斷擬合，與正式比較分開。切換等價執行版本時曾中斷尚未完成的程序，已完成的交叉驗證結果全部保留；未完成工作依原規格續跑，沒有依分數挑選。原低容量資料與模型未覆寫。','',
      '## A/B 選定參數','',table(['欄位','訓練條件','選模口徑','alpha','驗證 AP','驗證 AUROC','驗證 rho'],[
       [r['features'],'GCS60' if r['augmentation'] else '無增強',r['policy'],alpha(r['alpha']),
        *[f'{r["validation"]["clean"][m]:.5f}' for m in ('ap','auroc','rho')]] for r in evidence['selected']]),'',
      '完整網格、未遮蔽與兩種 A/B 全遮情境見 [全部 A/B 候選](P62_ab_candidates.csv)。這裡的三折平均不是 Set C 成績，也不是五種子平均。','',
      '## Set C：同病例與原全遮情境','']
    for name in ('Full','Elastic'):
        tags=[f'{name}_{a}_{p}' for a in ('plain','gcs60') for p in ('off','ap','rho95')]
        tags+=[f'XGB_{name}',f'XGB_GCS_{name}']
        lines += ['### '+name,'',table(['模型','未遮蔽 AP','GCS 全遮 AP','六欄全遮 AP','未遮蔽 rho'],[
          [label(tag),*[f'{lookup[c,tag]["ap"]:.5f}' for c in ('clean','gcs_full','six_full')],
           '未在本輪量測' if lookup['clean',tag]['rho'] is None else f'{lookup["clean",tag]["rho"]*100:.2f}%'] for tag in tags]),'']
    lines+=['## eICU Demo：同一 782 次住院／91 次死亡','',
      '沿已修正的明確尿量 mL 映射，整點／原始間隔兩套取樣均列出；它們是同一個 Demo 的敏感度設定。全部為 P12-only 訓練的凍結轉移。','',
      table(['模型','整點 AP','原始間隔 AP','整點 rho','原始間隔 rho'],[
       [label(tag),*[f'{lookup[c,tag]["ap"]:.5f}' for c in ('eicu_hourly','eicu_native')],
        *['未在本輪量測' if lookup[c,tag]['rho'] is None else f'{lookup[c,tag]["rho"]*100:.2f}%' for c in ('eicu_hourly','eicu_native')]]
       for tag in [r['model'] for r in primary if r['cohort']=='clean']]),'',
      '## 配對差：MA 與同實作無懲罰控制','',
      '以下為同一固定模型對的 AP 差，方向 MA 減無懲罰。Set C 為 2,000 次住院配對重抽；Demo 為 1,000 次患者配對重抽。區間未作歷史／多項比較修正，不涵蓋重訓與選參變異。','']
    for policy in ('ap','rho95'):
        lines+=['### '+('AP 選模' if policy=='ap' else '補值依賴取捨選模'),'',
         table(['隊列','欄位／增強','ΔAP','95% 區間'],[
          [case,name+'／'+aug,f'{r["delta"]:+.5f}',f'[{r["ci"][0]:+.5f}, {r["ci"][1]:+.5f}]']
          for case in ('clean','gcs_full','six_full','eicu_hourly','eicu_native')
          for name in ('Full','Elastic') for aug in ('plain','gcs60')
          for r in [evidence['paired'][case][f'{name}_{aug}_{policy}_minus_off']['ap']]]),'']
    lines+=['## 與原凍結 XGBoost 的 AP 配對差','',
      'MA-AP 依 A/B 的 AP 口徑選定；無增強對原 Full／Elastic，GCS60 對原 GCS-only 的相同欄位組。下表方向為 MA-AP 減 XGBoost，區間解讀同上；MA-rho95 與同作者無懲罰組的完整比較均保留於彙總 JSON。','',
      table(['隊列','MA-AP 欄位／增強','XGBoost 對照','ΔAP','95% 區間'],[
       [case,name+'／'+aug,label(ref),f'{r["delta"]:+.5f}',f'[{r["ci"][0]:+.5f}, {r["ci"][1]:+.5f}]']
       for case in ('clean','gcs_full','six_full','eicu_hourly','eicu_native')
       for name in ('Full','Elastic') for aug,ref in [('plain',f'XGB_{name}'),('gcs60',f'XGB_GCS_{name}')]
       for r in [evidence['paired'][case][f'{name}_{aug}_ap_minus_{ref}']['ap']]]),'']
    lines+=['## 全部缺失結果與限制','',
      '原 P9 的 13 種設定、相同 32 次匹配遮蔽均完成；逐模型／位置的整體 AP、AUROC、Brier、敏感度見 [缺失總表](P62_all_stress_metrics.csv)。主要三情境與 Demo 的 rho 另列於 [主指標表](P62_primary_metrics.csv)。本輪沒有重跑 P61 連續時間區段設定，不將整欄結果延伸為全部時間缺失的結論。P9 匹配比例保留最初的 Demo 映射，其中尿量當時未對齊；現在 Demo 評估已使用修正尿量映射。因此這是固定的歷史壓力情境，不代表目前 Demo 仍有六欄同等程度的缺失，也不是六個最重要欄位的排名。','',
      f'本輪有 96 個唯一交叉驗證工作；另有 {execution.get("duplicate_cv_attempts",0)} 次已完成的相同設定重算，科學輸出逐值一致，未據此挑較高者。初期主佇列／獨立區塊各使用 6／4 個 CPU 工作，之後改用 6 個工作、每工作 2 線程的等價特徵與排序平行化。逐候選與整個 A/B 折的樹／機率核對均完全一致；不使用近似 binning 或改動切分規則。GPU 僅做相容性測試，沒有訓練任何本輪模型。','',
      '相同樹數、深度與學習率僅對齊名義設定；作者 GBT 與 XGBoost 的梯度／切分／正則化仍不同。最接近機制控制的是同作者實作的 MA／無懲罰比較。各增強條件的 alpha 由 A/B 分別選定，兩個最終增強版本的差同時包含其選參結果；固定 alpha 的增強比較可在 A/B 全表查閱。另需注意：作者懲罰使用 mean(weight × missing)，分母是實際訓練行數；加權副本維持病例總權重，卻會改變行數。因此跨增強版本相同 alpha 不代表完全相同的有效懲罰強度；本輪沿作者公式，沒有另外修改正則化定義。','',
      'rho 僅統計直接讀到補值特徵的路徑；count、missing 旗標等已知輸入仍可被模型使用，低 rho 不等於完全不使用量測模式。','',
      '這輪為 seed42，沒有五種子重訓。補值依賴是任一預測路徑讀到補值的病例比例，樹數增加後此指標可能更易飽和；不能只看 AP 或只看 rho。原 XGBoost 參考列未在本輪量測 rho，保留空值，不補成零；同作者無懲罰 GBT 則有使用真實缺失矩陣量測。','',
      'Set C／Demo 已反覆查看，只有公開 Demo，尚無獨立完整版 eICU。未選最好種子、未依新測試結果改 alpha／主選模規則。原低容量與新配置同時改了樹數、深度及學習率，新舊差距不能全部稱作樹數效果。','',
      '[事前規格](P62_ma_gbt_matched_protocol.md) · [完整彙總與全部 AP／AUROC／Brier 配對差](P62_ma_gbt_matched_summary.json) · [原低容量對照](P59_ma_alignment_results.md)','',
      '## 重跑','',
      '使用原 MA CPU Python，PYTHONPATH 指向鎖定作者 upstream；另在獨立 target 安裝 numba==0.60.0、llvmlite==0.43.0。依序執行：','',
      '```powershell','python src/train_ma_matched.py check','# 下列兩個訓練入口擇一；不用同時啟動',
      'python src/train_ma_matched.py train --workers 6','# 等價較快入口：先逐值核對，再續跑相同 protocol',
      'python src/assist_ma_matched_parallel.py check','python src/resume_ma_matched_parallel.py',
      'python src/evaluate_ma_matched.py','# 公開彙總使用主環境，不載入模型','python src/publish_ma_matched.py','```','',
      '既有 protocol／selection／frozen 與每個已完成擬合會核對來源再續跑，拒絕覆寫不同來源。獨立重訓須先改用新的輸出根目錄保存原凍結成果；不要刪除既有醫療資料與模型。']
    (REPORT/'P62_ma_gbt_matched_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Published aggregate-only P62 tables and limitations; local final report untouched.')
if __name__=='__main__':main()
