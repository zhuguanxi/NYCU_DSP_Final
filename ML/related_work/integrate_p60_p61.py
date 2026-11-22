"""Integrate completed follow-up evidence into research navigation without replacing history."""
import hashlib
import json
from pathlib import Path
import re

ML=Path(__file__).resolve().parents[1]
ROOT=ML.parent
REPORT=ML/'reports'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def marked(path,body):
    text=path.read_text(encoding='utf-8')
    block='<!-- P60_P61_START -->\n'+body.rstrip()+'\n<!-- P60_P61_END -->'
    if '<!-- P60_P61_START -->' in text:
        text=re.sub(r'<!-- P60_P61_START -->.*?<!-- P60_P61_END -->',lambda _:block,text,flags=re.S)
    else:
        title,rest=text.split('\n',1);text=title+'\n\n'+block+'\n'+rest
    path.write_text(text,encoding='utf-8',newline='\n')

def main():
    audit=read(REPORT/'P60_P61_verification.json')
    assert audit['complete'] and audit['selected_seed'] is None
    for name,digest in audit['published_summary_sha256'].items():
        assert hashlib.sha256((REPORT/name).read_bytes()).hexdigest()==digest
    p60=read(REPORT/'P60_five_seed_stability_summary.json');p61=read(REPORT/'P61_progressive_block_summary.json')
    values=[p60['summary'][case]['p33']['metrics']['ap'] for case in ('clean','gcs_full','six_full')]
    ap='／'.join(f'{v["mean"]:.5f} ± {v["sd"]:.5f}' for v in values)
    story=['## P60／P61 新增補強：五種子與時間缺失','',
      '2026-11-22，依後續要求補完，固定參數、不選最佳種子。新增35次元件擬合；七個本組模型各五種子，KEDGN沿既有A+B五種子。39個缺失案例與兩種乾淨Demo取樣全部完成。','',
      f'P33在Set C未遮蔽／GCS全遮／六欄全遮的五種子AP平均±樣本SD為 **{ap}**；三個情境對Full均5/5正向。下方歷史P0–P59的0.60447等分數仍是原單一凍結P33，不能與本輪平均混稱。','',
      '### 收斂後的故事','',
      '**在整欄資訊失去時，缺失專家與表格混合提供韌性；CNN進一步改善域內完整資料排序，但有重度缺失與外部轉移取捨。** 五種子結果中，P33對P28在完整Set C為5/5正向；六欄全遮及Demo兩種取樣則均0/5。應保留這個分工與限制，不宣稱每一個模組在每一種情境都有收益。','',
      'P61比較隨機刪除12／24／36個時鐘小時，及同長度連續區段；GCS與六欄各三組固定位置，較強缺失包含較弱缺失的時段。原始目標事件移除後，表格、count/missing、時間摘要及CNN輸入同步重算。資料完整與部分缺失都不能只看單次最佳結果。','',
      '### 預定50%情境：P33對Full','',
      '以下為五種子機率平均模型，三組位置的AP差平均及1,000次住院配對區間；與上方五次AP平均分開判讀。','',
      '| 目標／設計 | ΔAP［95%區間］ | 少退化的AP［95%區間］ |','| --- | --- | --- |']
    for key,comparators in p61['primary_50_percent_ensemble_contrasts'].items():
        r=comparators['frozen_full']
        story.append(f'| {key} | {r["ap_delta"]:+.5f} [{r["ap_ci"][0]:+.5f}, {r["ap_ci"][1]:+.5f}] | {r["retention_gain"]:+.5f} [{r["retention_ci"][0]:+.5f}, {r["retention_ci"][1]:+.5f}] |')
    story+=['','四個50%情境的AP仍高於Full，但少退化的AP沒有一致支持，其中六欄連續24小時相對Full反而有較大退化的區間支持。因此研究故事必須區分「遮蔽後仍較高分」與「掉分較少」；整欄全遮的改善不能當成所有部分缺失的韌性結論。','',
      'P33對P28與KEDGN五種子集成的50% AP差均不確定；隨機24小時與連續24小時的差也未確認。這些是固定模型的病例抽樣區間；種子與位置波動另報，不把15組種子×位置當15個新測試集。Demo多種子仍不能消除患者數有限及反覆查看的限制。外部AP改善未由本輪證實，R1及其他已取消排程沒有恢復。','',
      '[五種子結果](P60_five_seed_stability_results.md) · [缺失曲線與全部比較](P61_progressive_block_results.md) · [全部種子CSV](P60_all_seed_results.csv) · [全部遮蔽CSV](P61_all_seed_mask_results.csv) · [驗收](P60_P61_verification.json) · [重跑](P60_P61_reproduction.md)','',
      '![漸進與連續區段缺失](../figures/P61_missingness_curves.png)']
    marked(REPORT/'研究結論與故事.md','\n'.join(story))
    short=(f'**最新補強已完成：P60／P61。** 固定五種子P33的Set C未遮蔽／GCS全遮／六欄全遮AP為 {ap}。'
       '35次元件擬合、39個缺失案例，全部種子與位置保留；P33相對P28在六欄全遮與Demo有穩定取捨。')
    for path,prefix in [(ROOT/'README.md','ML/'),(ML/'README.md','')]:
        marked(path,short+'\n\n'+f'[五種子結果]({prefix}reports/P60_five_seed_stability_results.md) · [漸進／區段缺失]({prefix}reports/P61_progressive_block_results.md) · [研究故事]({prefix}reports/研究結論與故事.md) · [重跑與驗收]({prefix}reports/P60_P61_reproduction.md)')
    marked(REPORT/'研究進度總覽.md',short+'\n\n'+
       '| 階段 | 已完成補強 | 證據 |\n| --- | --- | --- |\n'+
       '| P60 | 五種子重訓，全部200列整體指標，不選最佳種子 | [結果](P60_five_seed_stability_results.md) |\n'+
       '| P61 | 39案例、五種子×三位置、全部1560列與預定配對比較 | [結果](P61_progressive_block_results.md) |\n\n'+
       '[驗收](P60_P61_verification.json) · [重跑](P60_P61_reproduction.md)；下方P0–P59保留歷史單一凍結模型結果。')
    registry=read(REPORT/'研究實驗登錄.json')
    registry['followup_seed_missingness']=dict(complete=True,component_fits=35,case_count=39,
       selected_seed=None,verification='P60_P61_verification.json',reports=['P60_five_seed_stability_results.md','P61_progressive_block_results.md'])
    (REPORT/'研究實驗登錄.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Integrated verified follow-up findings and preserved original frozen-model history.',flush=True)

if __name__=='__main__': main()
