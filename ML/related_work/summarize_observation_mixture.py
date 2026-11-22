"""Publish aggregate-only P56 selection evidence and a concise interpretation."""

import json
from pathlib import Path

ML=Path(__file__).resolve().parents[1]
ROOT=ML/'artifacts/p56_observation_mixture'


def main():
    value=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))
    if value['test_data_used']: raise ValueError('Unexpected test use in rule selection')
    if value['accepted']: raise RuntimeError('Accepted candidate needs frozen holdout evaluation before publication')
    public=dict(value)
    public['source_sha256']={str(Path(k).relative_to(ML)):v for k,v in value['source_sha256'].items()}
    target=ML/'reports/P56_observation_mixture_summary.json'
    target.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    baseline=value['baseline']
    lines=['# P56：觀測狀態條件混合的 A/B 消融','',
           '依事先規格，重擬原固定 CNN 的三個 A/B 訓練折，逐項重現 P33 的原候選分數。32 個條件規則中，最佳仍是有／無 GCS 均以 P28 與 CNN 作 75%／25% 固定機率混合，與原 P33 完全相同。**加權目標增益為 0，未達預定 0.001 接受條件，因此保留 P33。**','',
           '## 可支持的發現','',
           '本輪沒有證據顯示，單靠 GCS 可得性切換序列權重或將缺失病例轉交 P26，就能在原 A/B 評估上進一步改善。這個結論僅限事先指定的候選家族、固定成分及六個驗證視圖；不代表所有動態混合方法均無效。','',
           '本輪不進行新規則的 Set C／Demo 計分。原模型結果仍保留於 P33、P54、P55，避免看見測試結果後反覆選規則。這是負向消融的完整結果，不是未完成的實驗。','',
           '## A/B 基準重現','', '| 指標 | 原 P33／本輪重現 |','| --- | ---: |']
    for key in ('clean','full_gcs','shift','mapped','objective'):
        lines.append(f'| {key} | {baseline[key]:.8f} |')
    lines+=['','三折、8,000 次住院、1,122 次死亡。CNN 固定 plain、8 輪、原 seed 與 CPU 設定，只完成三次折內重現擬合；新最終擬合 0 次，外部選參 0 次。原 P26 分支折外分數與 P28 anchor 均經原 A/B 指標核對。','',
            '## 候選與門檻','',
            '有 GCS 時序列權重為 0、0.15、0.25 或 0.50；無 GCS 時表格成分為 P26 或 P28，序列權重取相同四值。預測僅使用當前可得的 GCS 觀測旗標。clean 和 GCS 全遮 AP 各不得低於 P33 超過 0.003；候選加權目標至少高 0.001 才能接受。目標權重沿 P33：clean 35%、mapped 35%、shift 15%、GCS 全遮 15%。','',
            '| 候選 | clean AP | GCS 全遮 AP | 跨庫缺失 AP | 尿量部分恢復 AP | 加權目標 | 通過保留門檻 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,row in value['candidates'].items():
        lines.append('| '+name+' | '+' | '.join(f'{row[k]:.5f}' for k in ('clean','full_gcs','shift','mapped','objective'))+' | '+('是' if name in value['eligible'] else '否')+' |')
    lines+=['','## 重跑與來源','',
            '[事先規格](P56_observation_mixture_protocol.md) · [整體 JSON](P56_observation_mixture_summary.json)。在 ML 的原 P33 CPU 環境執行 `related_work/select_observation_mixture.py`，再用主環境執行 `related_work/summarize_observation_mixture.py`。既有完整選擇檔會阻止覆寫，三個完整折外重現檔可供恢復。原資料、折外分數與來源檔只留在本機 `artifacts/p56_observation_mixture/`。','']
    (ML/'reports/P56_observation_mixture_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('P56 complete: retained original P33; no new holdout scoring.',flush=True)


if __name__=='__main__': main()
