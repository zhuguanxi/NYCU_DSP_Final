"""Complete descriptive deliverables from saved results, without additional fits."""
import json
import joblib
import numpy as np
import pandas as pd
from study import X, y, FD, ART, REP, ROOT, frame_md, DATA_SHA
from finalize import NAMES


def main():
    fit_count_before = sum(1 for _ in (ART / 'fit_ledger.jsonl').open())
    # Training-only selector ranks and held-out permutation ranks are separate.
    ranks = []
    for fold in range(3):
        a = joblib.load(ART / 'selectors' / f'fold3_{fold}_adultFalse.joblib')
        assert a['feature_data_sha256'] == DATA_SHA
        for rank, variable in enumerate(a['rank'], 1):
            ranks.append(dict(variable=variable, fold=fold, selector_rank=rank,
                              selector_score=a['ranking_values'][variable]))
    ranks = pd.DataFrame(ranks)
    ranks.to_csv(ART / 'selector_variable_ranks.csv', index=False)
    rank_summary = ranks.groupby('variable').agg(
        selector_rank_min=('selector_rank', 'min'),
        selector_rank_max=('selector_rank', 'max'),
        selector_rank_mean=('selector_rank', 'mean'))
    perm = pd.read_csv(ART / 'group_permutation.csv')
    perm = perm[~perm.variable.str.startswith('CLUSTER:')].copy()
    perm['validation_rank'] = perm.groupby('fold').mean_AP_drop.rank(
        ascending=False, method='min')
    perm.to_csv(ART / 'validation_variable_ranks.csv', index=False)
    pr = perm.groupby('variable').agg(
        validation_rank_min=('validation_rank', 'min'),
        validation_rank_max=('validation_rank', 'max'))

    candidates = pd.read_csv(ART / 'candidate_variables_for_dl.csv').set_index('variable')
    candidates = candidates.drop(columns=list(rank_summary.columns) + list(pr.columns), errors='ignore')
    candidates = candidates.join(rank_summary).join(pr)
    for g in candidates.index:
        missing = (X[g].isna() if g.startswith('static_') else X[f'{g}__count'].eq(0))
        candidates.loc[g, 'development_no_observation_rate'] = float(missing.mean())
        candidates.loc[g, 'observed_record_n'] = int((~missing).sum())
    candidates['retention_reason'] = 'At least one summary retained by training-only Quality + nonzero Elastic Net in >=2 folds; channel reduction not established'
    candidates.to_csv(ART / 'candidate_variables_for_dl.csv', encoding='utf-8-sig')
    selected = pd.read_csv(ART / 'selected_features.csv')
    selected['clinical_name_zh'] = selected.variable.map(NAMES)
    selected['basis'] = 'Training-fold Quality/correlation filter then fixed Elastic Net C=1, l1_ratio=0.5, abs(coef)>1e-8'
    selected['retention_reason'] = np.where(selected.selected_folds >= 2,
        'Nonzero encoded coefficient in >=2 training folds after redundancy filter',
        'Nonzero encoded coefficient in 1 training fold; less stable candidate')
    selected.to_csv(ART / 'selected_features.csv', index=False, encoding='utf-8-sig')

    rows = []
    for code, ix in X.groupby('static_ICUType', dropna=False).groups.items():
        for variable in ['GCS', 'BUN', 'Lactate', 'Urine', 'HR']:
            count = X.loc[ix, f'{variable}__count']
            rows.append(dict(ICUType=code, variable=variable, n=len(ix),
                deaths=int(y.loc[ix].sum()), mortality_rate=float(y.loc[ix].mean()),
                no_observation_rate=float(count.eq(0).mean()),
                median_valid_count=float(count.median()), mean_valid_count=float(count.mean())))
    icu = pd.DataFrame(rows)
    icu.to_csv(ART / 'ICUType_measurement_patterns.csv', index=False)
    effects = pd.read_csv(ART / 'outcome_associations.csv')
    for idx, row in effects.iterrows():
        values = X[row.feature]
        for label, label_value in [('alive', 0), ('death', 1)]:
            valid = values[y.eq(label_value)].dropna()
            effects.loc[idx, label + '_q25'] = valid.quantile(.25)
            effects.loc[idx, label + '_q75'] = valid.quantile(.75)
    effects.to_csv(ART / 'outcome_associations.csv', index=False)

    coef = pd.read_csv(ART / 'LR_coefficient_stability.csv')
    wanted = ['GCS__last', 'static_Age', 'Urine__mean', 'BUN__last',
              'Lactate__last', 'GCS__count', 'Lactate__count', 'Lactate__missing']
    table = coef[coef.feature.isin(wanted)].copy()
    table['clinical_name_zh'] = table.variable.map(NAMES)
    table['direction'] = np.where(table['mean'] > 0, '正', '負')
    best = json.loads((ART / 'best_models.json').read_text())
    coding = []
    for fold in range(3):
        pipe, _, cols = joblib.load(ART / 'runs' / best['LR']['key'] / f'fold{fold}.joblib')
        encoder = pipe['pre'].named_transformers_['cat']['encode']
        cat_cols = [c for c in cols if FD.loc[c, 'kind'] == 'categorical']
        for name, categories, drop_idx in zip(cat_cols, encoder.categories_, encoder.drop_idx_):
            coding.append(dict(fold=fold, feature=name, training_categories=categories.tolist(),
                               reference_category=float(categories[drop_idx])))
    pd.DataFrame(coding).to_csv(ART / 'category_encoding.csv', index=False)
    ranked = candidates.sort_values('mean_AP_drop', ascending=False).head(10).reset_index()
    icutable = icu[icu.variable == 'Lactate'][['ICUType', 'n', 'deaths', 'mortality_rate', 'no_observation_rate', 'median_valid_count']]
    text = '\n\n## 補充：方向、排名與量測模式\n\n'
    text += 'GCS最後一次有效值在存活／死亡組的中位數為15／9；BUN最後值為18／31 mg/dL；每次記錄尿量的平均摘要為117.93／74.49 mL。這些分布差異與SHAP方向可互相對照，但尿量不是每天或48小時總尿量。年齡中位數為66／74歲。\n\n'
    text += '以下是最佳LR（C=0.1、Elastic Net比例0.5）的標準化連續欄位係數。same_sign_folds=3表示三折同方向且非零，低於3表示有零係數或方向不一致；係數已受正則化及其他相關輸入影響。\n\n'
    text += frame_md(table[['feature', 'clinical_name_zh', 'statistic', 'mean', 'std', 'direction', 'same_sign_folds']])
    text += '\n\n類別欄位採one-hot並drop first。本輪最佳LR三折均以Gender缺失碼−1作參考、ICUType=1作參考；例如Gender_0和Gender_1係數不能直接互相比較來宣稱性別因果效應。實際逐折編碼見[參考類別](../artifacts/category_encoding.csv)。'
    text += '\n\n數值摘要與count／missing可有不同方向；count反映量測次數，不能解讀為生理改善。沒有再據此調整selector。\n\n'
    text += frame_md(ranked[['variable', 'clinical_name_zh', 'selected_folds', 'selector_rank_min', 'selector_rank_max', 'validation_rank_min', 'validation_rank_max', 'development_no_observation_rate']])
    text += '\n\nselector排名只使用訓練折數值摘要係數；validation排名來自既有模型的分組置亂，兩者方法不同，不應要求相同。缺失率指整段無有效觀測的紀錄比例。完整排名見[訓練排名](../artifacts/selector_variable_ranks.csv)及[驗證排名](../artifacts/validation_variable_ranks.csv)。\n\n'
    text += '以乳酸為例，各ICUType的量測覆蓋與死亡率如下；這些組別不是不同醫院，不能直接當外部驗證或因果比較。其他核心變數的count與缺失率見[完整量測表](../artifacts/ICUType_measurement_patterns.csv)。\n\n'
    text += frame_md(icutable)
    text += '\n\n完整[分布表](../artifacts/outcome_associations.csv)含存活／死亡有效n、median、q25／q75及缺失率。\n\n'
    text += '本輪Elastic精簡保留全部37個動態變數，主要刪除重複或係數為零的摘要。不能把這次結果寫成「已找出少數最佳深度學習通道」。Top10／20在XGBoost三折都超過原定0.01 AP損失容忍值，後續如再精簡通道需重新驗證。\n'
    path = REP / 'P5_變數關係與解釋.md'
    path.write_text(path.read_text(encoding='utf-8').split('\n\n## 補充：方向、排名與量測模式')[0] + text, encoding='utf-8')

    memorandum = '''# 銜接深度學習的備忘錄

日期：2026-10-13。本輪已完成表格探索；以下是後續實作規格與研究假說，尚未訓練深度模型。

## 已有資料與時間定義

- [原始序列](../artifacts/observations_AB.parquet)欄位：RecordID、elapsed_minute、variable、raw_value、clean_value、quality_flag、source_order。包含靜態描述與動態觀測，建序列時須依變數角色分流。
- elapsed_minute以入ICU為0，使用0–2880分鐘（含端點）。不規則觀測直接保留；同一時間有多筆時用source_order保留順序。尚未規定重複時間點平均或重採樣，不應偷偷沿用表格last摘要。
- clean_value缺失或quality_flag非valid，不能當有效量測。某時刻完全沒有記錄與有紀錄但無效可另保留品質標記。
- 37個動態變數全部保留；5個靜態候選為Age、Gender、Height、ICUType及admission Weight。Weight的00:00值也在動態序列內，未來需固定是否允許這種重疊。
- GCS是序數、MechVent是狀態、Urine是每次記錄量；不能把三者全部當一般連續測量任意插值、加總或解讀為時間占比。

## mask、時間間隔與前處理

建議先固定一種資料表示：保留每筆觀測的時間與值，或明確定義時間網格。若用網格，mask=1只表示該格有合法觀測，補值位置仍為mask=0；padding另外有padding mask，不能與臨床缺失混在一起。時間間隔表示該變數距上次合法觀測多久，第一次尚未觀測需另標記，不能假設為0。

數值中心與尺度只從訓練折有效觀測估計。驗證／C沿用同一規則；類別編碼、補值與稀有值策略同樣限制在訓練折。沒有證据顯示整段缺失等於正常值。mask/count可能反映醫療量測流程，需透過有／無觀測訊號的消融確認可移植性。

## 如何使用這輪篩選結果

最終表格模型從435欄縮為322欄，三折模型平均約318欄，但仍保留所有37個動態臨床變數。這是摘要冗餘精簡，沒有證實深度模型可直接只用10或20通道。候選CSV是穩定性及模型依賴記錄，不是新的固定通道最佳解。

GCS、年齡、尿量、BUN與乳酸值得優先分析；低排名通道先保留。後續至少比較完整通道與訓練內產生的精簡通道；通道選擇要在每個訓練折內完成，不能用已看的驗證重要性選通道後，再把同一折當無偏成績。

## 共同比較方式

- 沿用[三折RecordID](../artifacts/folds3.csv)，只用A/B調方法；AP主指標，AUROC、Brier、訓練／推論成本輔助。AP與梯形AUPRC分開記錄。
- 完整／精簡表格XGBoost已保存於[模型目錄](../artifacts/final_models/metadata.json)，方便最後與深度方法共同評估。模型輸入為手工摘要，深度模型输入為原始序列，兩者不要求相同欄位數。
- 表格三折AP完整0.5450、精簡0.5385；五折完整0.5515、精簡0.5551。這些是開發成績，不能只挑較高的一組與不同protocol文獻比較。
- C未在本輪評估。等方法、前處理及閾值規則凍結，再一起評估C；先前EDA已探索C，需在報告揭露。
- 外部資料增補尚未實作。若加入不同病人資料，不可用人口比例將兩個人的紀錄合併；需另定義外部訓練、外部驗證或資料集層級資訊的用途。

## 優先驗證的三個假說

1. XGBoost從F0到F2的AP增加约0.0133：原始時間與缺失mask是否能提供更多資訊？需分別比較時間表示與mask，不能一次改全部。
2. BUN與Creatinine、不同血壓摘要有共同變化：跨變數模型是否超過一般序列模型？相關不等於已找到圖的因果邊。
3. Top10／20表格通道精簡損失較大：深度模型是否同樣需要較多通道，或能用序列資訊補償？以共同切分實測。

本輪原生缺失敏感性、class weight、閾值最佳化、時間分窗、PDP／ICE屬於核准計畫的可選延伸，未執行；不影響P0–P6主要工作完成。
'''
    (REP / '深度學習銜接備忘錄.md').write_text(memorandum.replace('证据', '證據').replace('输入', '輸入').replace('约', '約'), encoding='utf-8')
    readme = ROOT / 'ML/README.md'
    text = readme.read_text(encoding='utf-8').split('\n\n## 重現與審閱')[0].replace('评估', '評估')
    text += '''

## 重現與審閱

[深度學習銜接備忘錄](reports/深度學習銜接備忘錄.md)說明序列、mask、時間間隔與共同切分。精簡模型最終322欄，**仍包含全部37個動態變數**；候選表已補上選擇依據、跨折排名範圍、缺失率與保留理由。

專案專屬環境位於`.venv`，套件固定於[requirements-lock.txt](requirements-lock.txt)。以下命令從專題根目錄執行，已完成配置會使用現有快取，不必重新訓練。重建輸入可能使資料雜湊改變；若要完整重跑應先另存目前成果，保留訓練預算與來源紀錄。

```powershell
ML/.venv/Scripts/python.exe ML/src/prepare.py
ML/.venv/Scripts/python.exe ML/src/study.py baseline
ML/.venv/Scripts/python.exe ML/src/study.py tune
ML/.venv/Scripts/python.exe ML/src/study.py families
ML/.venv/Scripts/python.exe ML/src/study.py selection
ML/.venv/Scripts/python.exe ML/src/explain.py
ML/.venv/Scripts/python.exe ML/src/study.py confirm
ML/.venv/Scripts/python.exe ML/src/finalize.py
ML/.venv/Scripts/python.exe ML/src/complete_reports.py
```

訓練量已按本次追加核准設為最多170次；目前158次。調參、selector與最終模型保存皆計入。最高RSS數字僅涵蓋訓練採樣，不代表SHAP／序列讀取的整個流程峰值。
'''
    readme.write_text(text, encoding='utf-8')
    for name in ['P0_資料與特徵重建.md']:
        path = REP / name
        path.write_text(path.read_text(encoding='utf-8').replace('极端', '極端').replace('含未知名稱與缺失觀測', '保留原始名稱與缺失觀測'), encoding='utf-8')
    # This step is intentionally descriptive only.
    assert sum(1 for _ in (ART / 'fit_ledger.jsonl').open()) == fit_count_before
    print('Candidate metadata, clinical descriptions and DL memorandum completed; no additional fits')


if __name__ == '__main__':
    main()
