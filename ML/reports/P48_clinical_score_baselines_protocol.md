# P48 同一隊列的臨床評分比較規格

## P12

- 使用 PhysioNet v1.0.0 官方 Outcomes 表的 SAPS-I、SOFA 與住院死亡。兩個評分分別在 A/B 兩項分數有效的共同隊列上擬合單變量邏輯迴歸，固定設定並驗證斜率為正。AP、AUROC 須與直接使用原評分排序一致。 [PhysioNet P12 欄位定義](https://physionet.org/content/challenge-2012/1.0.0/#outcome-related-descriptors)
- 在原 Set C 兩項評分皆有效的住院次上比較，沿用未遮蔽輸入。回報納入及排除人次與死亡數，並報告各方法 AP、AUROC、Brier。P33 對兩項臨床評分與原凍結 Full 的 AP、AUROC 差值用原 Set C 的 2,000 次住院次配對重抽。
- 臨床分數單獨計分，原 XGBoost 與 P26、P28、P33、P38 沿用已凍結模型和原輸入。官方分數未依 Set C 遮蔽重算，故比較限於未遮蔽資料。

## eICU Demo

- 使用原凍結且完成尿量對齊的整點與原始間隔設定。只讀官方表中的 ICU 住院次鍵、APACHE 版本及預測住院死亡風險，限唯一 APACHE IVa 列且有效機率在 0–1 之間。住院死亡沿用原隊列的標籤。 [eICU APACHE 結果表](https://eicu.mit.edu/eicutables/apachepatientresult/) · [PhysioNet Demo 2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/)
- 所有方法在具有有效 APACHE IVa 機率的同一批住院次上計分。P28 為主要比較，P38 為次要比較，沿用 1,000 次患者配對重抽。記錄共同子集的覆蓋率與排除情況。APACHE IVa 與研究模型使用的觀察資訊及可用時點可能不同。

## 重跑

在 `ML` 執行，需先有先前各階段凍結產物及官方公開 Demo 目錄中的 `apachePatientResult.csv.gz`。所有計算使用 CPU。僅在 Git 排除的 `artifacts/p48_clinical_score_baselines/` 寫入彙總，已有結果時停止。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts\p21_cpu_packages\site'
& '.\.venv\Scripts\python.exe' src/evaluate_clinical_score_baselines.py
```
