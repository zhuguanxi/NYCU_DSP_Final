# P50 以 eICU 病例增強 24 小時模型

P49 的 24 小時分流在 eICU Demo 未確認有 AP 優勢。本輪將 P45 的真實外部增強移植到相同觀察窗，檢查能否改善排序並保留 GCS 全遮時的表現。 [P49 24 小時結果](P49_24h_landmark_results.md) · [P45 48 小時增強](P45_augmentation_branch_ablation_results.md)

- 沿用 P49 的 P12 A/B 風險集代理、24 小時特徵、時間摘要、標籤和固定 CPU XGBoost 參數。每個訓練折加入 eICU Demo 整點取樣的合格住院病例，每次 ICU 住院的訓練權重為 1。只用發布的住院死亡標籤，APACHE IVa 僅供測試比較。增強設定不再搜尋。
- GCS 有觀測時使用新增病例後重擬的 XGBoost。GCS 缺失時直接使用 P49 已凍結的 XGBoost 與無 GCS LightGBM 原混合。逐值核對無 GCS 的分數，確保 Set C GCS 全遮時完全重現 P49。
- eICU Demo 沿用 P49 對齊與合格住院條件，使用 P44 的固定五折方法與種子。共用患者把院鍵連為區塊，同一折的訓練與測試患者和院鍵區塊互斥。每次 ICU 住院僅有一組折外分數，兩種取樣共用同一套折外模型。主要比較全隊列整點 AP 對凍結 P49，次要比較原始間隔及有 APACHE IVa 風險的共同隊列。報告 1,000 次患者配對區間，另以院鍵區塊配對重抽作敏感度分析。
- 用全 P12 A/B 和全合格 Demo 訓練一個最終模型，在原 P49 的 Set C 風險集代理評估未遮蔽、GCS 全遮及跨庫模擬。AP／AUROC 使用 2,000 次住院次配對重抽。最終重擬模型只用於 Set C，Demo 成績來自分組折外預測。完整患者互斥 eICU 的確認需獨立資料。

在 `ML` 執行，使用 P21 CPU 環境與原凍結 P49 產物。資料、模型與彙總留在 Git 排除的 `artifacts/p50_24h_eicu_augmentation/`。重跑前保留既有結果。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts\p21_cpu_packages\site'
& '.\.venv\Scripts\python.exe' src/evaluate_eicu_24h_augmentation.py
```
