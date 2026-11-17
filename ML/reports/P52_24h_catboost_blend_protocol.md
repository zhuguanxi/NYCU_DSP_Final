# P52 24 小時 CatBoost 混合

P49 的 24 小時設計以 XGBoost 和 LightGBM 分流。這次在相同的前處理、三折與缺失視圖下測試 CatBoost 能否補足排序。用既有 P21 的 `cat5`、650 棵樹與 CPU 設定，在 24 小時表格和時間摘要上擬合。這次的候選只使用 P12 A/B。

- 按 P49 的凍結設定重跑折外參照，逐項與 P49 已保存的 A/B AP 核對。每折驗證資料只用未見過該折的模型。把 CatBoost 機率以 25% 或 50% 權重與 P49 混合，分別測試全病例混合及只在有 GCS 時混合。
- 沿用 P49 的六個視圖和加權 AP。未遮蔽與 GCS 全遮 AP 各須達 P49 減 0.005，加權 AP 改善須至少 0.001。若多組通過，依加權 AP、未遮蔽 AP 排序。若無候選通過，就保留 P49 並結束。
- 若候選通過，才用全 A/B 擬合、凍結，並一次計算 Set C 的同病例、同遮蔽比較及 eICU Demo 的整點與原始間隔比較。回報 AP、AUROC、Brier、配對區間、GCS 可用性分層樣本數及 AP。相對 APACHE IVa 的比較限共同隊列。隊列及比較方式沿用 P49。 [P49 規格](P49_24h_landmark_protocol.md)

在 `ML` 使用現有 P21 CPU 環境與 P49 凍結產物。輸出保留在 Git 排除的 `artifacts/p52_24h_catboost_blend/`。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
$py = '.\.venv\Scripts\python.exe'
& $py src/evaluate_24h_catboost_blend.py fit
# 只有選出新候選時才執行
& $py src/evaluate_24h_catboost_blend.py evaluate
```
