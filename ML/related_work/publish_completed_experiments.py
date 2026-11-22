"""Generate aggregate-only P55/P57/P58/P59 experimental reports."""

import json
import argparse
from pathlib import Path

ML=Path(__file__).resolve().parents[1]
REPORT=ML/'reports'
NAMES={'frozen_full':'XGBoost Full','frozen_elastic':'XGBoost Elastic','gcs_only_full':'GCS-only Full',
       'gcs_only_elastic':'GCS-only Elastic','p26':'P26 分流','p28':'P28 混合','p33':'P33 序列混合',
       'kedgn_set_a':'KEDGN Set A','kedgn_ab':'KEDGN A+B','p49':'P49','p50':'P50 外部增強','p52':'P52','apache_iva':'APACHE IVa'}
VERSIONS={'original_1000':'原版 α=1000','original_10':'原版 α=10','augmented_10':'五欄增強 α=10'}


def read(name): return json.loads((REPORT/name).read_text(encoding='utf-8'))
def f(value): return f'{value:.5f}'
def ci(value): return f'[{value[0]:+.5f}, {value[1]:+.5f}]'
def write(name,lines): (REPORT/name).write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')


def p55():
    d=read('P55_kedgn_missingness_summary.json')
    lookup={(r['scenario'],r['model']):r for r in d['metrics']}
    cases=['clean','full_GCS','full_Lactate','full_Urine','full_all_six']
    lines=['# P55：KEDGN 與原模型的共同缺失比較','',
           '全部使用既有凍結模型，新增訓練 0 次。Set C 4,000 次住院／585 次死亡，八種情境均保留完整隊列。兩版 KEDGN 各為原五種子機率平均，另保存種子指標。表格、時間摘要與序列同步移除相同通道；KEDGN 保留時間網格與原長度，只清除指定通道值。','',
           '## AP 主表','', '| 模型 | 未遮蔽 | GCS 全遮 | 乳酸全遮 | 尿量全遮 | 六項全遮 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    models=list(dict.fromkeys(r['model'] for r in d['metrics']))
    for model in models:
        lines.append('| '+NAMES[model]+' | '+' | '.join(f(lookup[case,model]['ap']) for case in cases)+' |')
    lines+=['','侵入式 MAP、鎂、機械通氣的單欄結果及全部 AUROC、Brier、敏感度見整體 JSON／研究結果總表。以下差值為同一模型相對其未遮蔽 AP。','',
            '| 模型 | GCS 全遮 ΔAP | 六項全遮 ΔAP |','| --- | ---: | ---: |']
    for model in models:
        lines.append(f'| {NAMES[model]} | {lookup["full_GCS",model]["delta_from_clean"]["ap"]:+.5f} | {lookup["full_all_six",model]["delta_from_clean"]["ap"]:+.5f} |')
    lines+=['','## 主要配對比較','',
            '差值均為 A+B KEDGN 減對照，沿用 2,000 次住院配對重抽，seed 20261013。區間描述固定模型／隊列，不包含重新訓練變異，未調整歷史多重比較。','',
            '| 情境 | 對照 | ΔAP | 95% 區間 | ΔAUROC | 95% 區間 |','| --- | --- | ---: | --- | ---: | --- |']
    for case,bounds in d['primary_paired_ci'].items():
        for tag,b in bounds.items():
            a=lookup[case,'kedgn_ab']; old=lookup[case,tag]
            lines.append(f'| {case} | {NAMES[tag]} | {a["ap"]-old["ap"]:+.5f} | {ci(b["ap"])} | {a["auroc"]-old["auroc"]:+.5f} | {ci(b["auroc"])} |')
    lines+=['','## 解讀','',
            '單欄遮蔽是模型依賴的介入性敏感度，不是變數的臨床因果效果。KEDGN 的缺失 mask、文字知識與圖機制，也不保證在整欄資訊消失時維持未遮蔽分數。比較要同時看未遮蔽 AP、遮蔽後 AP 與相對損失，不能只以其中一欄挑選贏家。','',
            '同一 P33 的完整資料及遮蔽聯合證據另見 P58。Set A 版與 A+B 版的訓練量和選輪次方式不同；KEDGN 與表格模型的表示、清理、採樣及種子數也不同，因此本表不宣稱同容量架構因果優劣或全局 SOTA。','',
            '[事先規格與重跑](P55_kedgn_missingness_protocol.md) · [整體 JSON](P55_kedgn_missingness_summary.json)。逐筆預測與權重只留在本機 artifacts/p55_kedgn_missingness/。','']
    write('P55_kedgn_missingness_results.md',lines)


def p57():
    d=read('P57_demo_paired_summary.json')
    lines=['# P57：P52 與 P50 的 24 小時同病例比較','',
           '補齊 P52 對 P50 的缺口。P49/P52 為 P12-only 凍結遷移；P50 為 P12 加 Demo 訓練折病例的分組折外預測，兩種設定明確區分。依原分組與參數重現 P50 五次折內擬合，所有原指標重現，沒有新增選參或全 Demo 回測。','',
           '## 全隊列：1,398 次住院／121 次死亡','',
           '| 取樣 | 模型 | AP | AUROC | Brier | 0.5 敏感度 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for mode,r in d['results'].items():
        for tag,m in r['metrics'].items():
            lines.append(f'| {mode} | {NAMES[tag]} | '+' | '.join(f(m[k]) for k in ('ap','auroc','brier','sensitivity'))+' |')
    lines+=['','## 配對差值：P52 減對照','',
            '患者與院鍵區塊各重抽 1,000 次，seed 42。AP／AUROC 正差較好，Brier 負差較好。兩種取樣完整呈現，未因某指標不確定而改寫另一指標的意義。','',
            '| 取樣 | 對照 | ΔAP | 患者 95% 區間 | 院鍵區塊 95% 區間 | ΔBrier | 患者區間 | 區塊區間 |',
            '| --- | --- | ---: | --- | --- | ---: | --- | --- |']
    for mode,r in d['results'].items():
        for key,old in [('p52_vs_p50','p50'),('p52_vs_p49','p49')]:
            a,b=r['metrics']['p52'],r['metrics'][old]
            p,h=r['patient_ci'][key],r['hospital_block_ci'][key]
            lines.append(f'| {mode} | {NAMES[old]} | {a["ap"]-b["ap"]:+.5f} | {ci(p["ap"])} | {ci(h["ap"])} | {a["brier"]-b["brier"]:+.5f} | {ci(p["brier"])} | {ci(h["brier"])} |')
    lines+=['','P52 對 P50 的 AP 點估計較高，但兩種取樣、兩種重抽的區間均跨零，不能確認優劣。P52 對 P49 的 AP 改善也仍未確定；Brier 的差值區間在兩種取樣及兩種重抽均低於零，支持本 Demo 上相對 P49 的機率誤差下降。這與證明 AP 提升是不同結論。','',
            '## APACHE 共同隊列：1,166 次住院／100 次死亡','',
            '| 取樣 | 模型 | AP | AUROC | Brier |','| --- | --- | ---: | ---: | ---: |']
    for mode,r in d['results'].items():
        for tag,m in r['matched']['metrics'].items():
            lines.append(f'| {mode} | {NAMES[tag]} | {f(m["ap"])} | {f(m["auroc"])} | {f(m["brier"])} |')
    lines+=['','所有共同隊列的 P52 對 P50/P49/APACHE 的 AP／AUROC／Brier 患者及區塊區間，完整保存於 JSON。不得拿 48 小時模型分數與前 24 小時 APACHE 作無條件公平比較。','',
            '## GCS 分層與分組限制','',
            '| 取樣 | GCS | 住院／死亡 | P49 AP | P50 AP | P52 AP |','| --- | --- | ---: | ---: | ---: | ---: |']
    for mode,r in d['results'].items():
        for label,v in r['gcs_strata'].items():
            lines.append(f'| {mode} | {label} | {v["stays"]}／{v["deaths"]} | '+ ' | '.join(f(v['metrics'][k]['ap']) for k in ('p49','p50','p52'))+' |')
    meta=d['patient_grouping']
    lines+=['',f'共有 {meta["patients"]} 名患者、{meta["hospitals"]} 個發布院鍵、{meta["hospital_blocks"]} 個患者連結院鍵區塊。每折患者與院鍵區塊互斥；無 GCS 病例的三模型分數逐值相同。發布描述與原始院鍵數存在差異，實際醫院數不從院鍵推斷。分層只作描述，未依分層結果重新選模型。','',
            '[事先規格](P57_demo_paired_protocol.md) · [整體 JSON](P57_demo_paired_summary.json)。本輪證據仍來自同一 Demo，不能冒充全新獨立外部驗證。','']
    write('P57_demo_paired_results.md',lines)


def p58():
    d=read('P58_joint_objectives_summary.json')
    lines=['# P58：同一主模型的準確性、韌性與外部轉移','',
           '主模型固定為 P33，不依測試情境切換到各情境最高分的模型。P26 保留為重度缺失專家對照。全部結果使用已凍結分數，新增擬合 0 次。','',
           '## Set C 同病例結果','',
           '| 情境 | Full AP | P33 AP | ΔAP | 95% 配對區間 |','| --- | ---: | ---: | ---: | --- |']
    for case,r in d['set_c']['scenarios'].items():
        lines.append(f'| {case} | {f(r["metrics"]["frozen_full"]["ap"])} | {f(r["metrics"]["p33"]["ap"])} | {r["delta"]["ap"]:+.5f} | {ci(r["paired_ci"]["ap"])} |')
    lines+=['','三個固定情境的 AP 差值區間都高於零，支持同一模型在未遮蔽及指定遮蔽下優於 Full，並非需要事後各選一個測試贏家。','',
            '## 相對未遮蔽的 AP 損失','',
            '| 遮蔽 | Full ΔAP | P33 ΔAP | P33 少損失的 AP | 95% 區間 |','| --- | ---: | ---: | ---: | --- |']
    for case,r in d['set_c']['scenarios'].items():
        if case=='clean': continue
        changes=r['ap_change_from_clean']
        lines.append(f'| {case} | {changes["frozen_full"]:+.5f} | {changes["p33"]:+.5f} | {r["retention_gain"]:+.5f} | {ci(r["retention_gain_ci"])} |')
    lines+=['','兩個損失差的區間均高於零，支持 P33 的優勢同時包含較小的 AP 退化。人工全遮蔽仍不能代表所有臨床量測機制。','',
            '## eICU 純遷移','',
            '| 取樣 | 對照 | P33 AP | 對照 AP | ΔAP | 患者 95% 區間 |','| --- | --- | ---: | ---: | ---: | --- |']
    for mode,r in d['eicu_demo']['modes'].items():
        for tag in ('frozen_full','p28'):
            lines.append(f'| {mode} | {NAMES[tag]} | {f(r["metrics"]["p33"]["ap"])} | {f(r["metrics"][tag]["ap"])} | {r["delta"][tag]["ap"]:+.5f} | {ci(r["paired_ci"][tag]["ap"])} |')
    lines+=['','外部差值區間均跨零，因此上述 Set C 的同時改善不能直接外推成跨資料庫提升。Demo 的基礎死亡率、病例組成、量測介面與欄位可得性不同，跨庫原始 AP 的升降也不能單獨解讀成同分布效能變化。','',
            '## 證據界限','',
            'Set C 未參與擬合，但已多次查看；這是固定模型的探索性整合證據。2,000 次住院配對抽樣與 1,000 次患者配對抽樣沿原種子，未校正跨歷史模型搜尋，也未涵蓋重新訓練／跨醫院變異。P56 的新條件混合在 A/B 未勝出，故本輪保留原 P33。','',
            '[事先規格](P58_joint_objectives_protocol.md) · [整體 JSON](P58_joint_objectives_summary.json)。完整 AUROC／Brier／敏感度及差值保存於 JSON。','']
    write('P58_joint_objectives_results.md',lines)


def p59():
    d=read('P59_ma_alignment_summary.json')
    lines=['# P59：MA-GBT 與目前尿量映射的一致性比較','',
           '三組 MA-GBT、Full／Elastic 及兩種取樣皆固定，不重新訓練、不重新選 α 或增強強度。先重現原未對齊分數，再更新為 P23/P53 相同的明確尿量 mL 輸入，避免拿不同映射的數字直接比較模型。','',
           '| 配置 | 欄位 | 取樣 | 原 AP | 對齊 AP | ΔAP 95% 區間 | 原補值依賴 | 對齊補值依賴 |',
           '| --- | --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in d['eicu']:
        lines.append(f'| {VERSIONS[r["variant"]]} | {r["model"]} | {r["mode"]} | {f(r["before"]["ap"])} | {f(r["after"]["ap"])} | {ci(r["paired_ci"]["ap"])} | {100*r["rho_before"]:.2f}% | {100*r["rho_after"]:.2f}% |')
    lines+=['','## 發現','',
            '尿量對齊可以大幅降低部分原版 MA-GBT 的補值路徑依賴，但 AP 變化很小，配對區間仍跨零。五欄增強版對尿量映射的分數完全不變，與其既有低尿量依賴相符。這支持將「降低補值依賴」和「改善死亡排序／跨庫效能」分開量化，不能把依賴率下降直接當成 AP 改善。','',
            '## Set C 原分數核對','',
            '| 配置 | 欄位 | 未遮蔽 AP | GCS 全遮 AP | 六項全遮 AP |','| --- | --- | ---: | ---: | ---: |']
    values={(r['variant'],r['model'],r['scenario']):r['metrics']['ap'] for r in d['set_c']}
    for version in VERSIONS:
        for model in ('Full','Elastic'):
            lines.append(f'| {VERSIONS[version]} | {model} | '+' | '.join(f(values[version,model,c]) for c in ('clean','full_GCS','full_all_six'))+' |')
    lines+=['','MA-GBT 沿作者固定版本及原 10 棵、深度 2 的規格；原 α 選擇也包括 AUROC 保留門檻與補值依賴，與 AP-only 的大容量 boosting 選模目標不同。本比較是方法對照，不宣稱所有 MA-GBT 容量／參數均劣於其他架構。','',
            '[規格與原環境命令](P59_ma_alignment_protocol.md) · [整體 JSON](P59_ma_alignment_summary.json)。每組舊分數、凍結權重與編碼統計均核對，新增擬合 0 次。','']
    write('P59_ma_alignment_results.md',lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases',nargs='+',choices=('55','57','58','59'),default=['55','57','58','59'])
    for phase in parser.parse_args().phases: globals()[f'p{phase}']()
    # These summaries are hashed by the completion audit and pinned to LF in Git.
    for summary in REPORT.glob('P5[5-9]*_summary.json'):
        summary.write_text(summary.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    print('Published selected aggregate-only experimental reports.')
