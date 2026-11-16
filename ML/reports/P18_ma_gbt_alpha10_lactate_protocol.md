# P18 MA-GBT 乳酸缺失增強

沿用 P16 的 alpha=10、10 棵深度 2 的樹、原 A/B 三折、凍結的 Full／Elastic 欄位與補值編碼。GCS 45%／ICU 類型 75%／尿量 75% 使用原遮蔽種子與位置，只有乳酸遮蔽比例由 A/B 選。

## 乳酸強度

- 乳酸訓練候選為 25%、75%、100%，以每折訓練內原本有乳酸的 stay 為分母。同折候選沿用同一排列的不同前段。整組設為缺失，包括乳酸統計量、量測次數及定義旗標。每筆 stay 仍使用一列，標籤及權重沿用原設計。
- Full 用三折驗證，主要分數為乳酸全遮 AP 和聯合遮蔽 AP 的等權平均。聯合遮蔽使用 GCS 25%／ICU 類型 90%／尿量 100%／乳酸 75%，同一折固定三次抽樣。計算乳酸補值依賴率並記錄未遮蔽、GCS 全遮兩種檢查。
- 優先從未遮蔽和 GCS 全遮 AP 各較 P16 下降最多 0.01 的候選選最高主要分數。若無符合兩項者，選通過 GCS 全遮門檻的候選。若仍無符合者，使用全候選並記錄。主要分數同分時，依乳酸補值依賴率較低、遮蔽比率較低排序。
- Elastic 共用選定強度並回報三折。全 A/B 各擬合一次後凍結，再開始評估。

## 對照

Set C 使用 P16 的原 cohort、P9 的 13 種遮蔽和同一組 32 次匹配抽樣。eICU Demo 沿用整點與原始間隔，AP／AUROC 配對差使用同一組 1,000 次患者分組 bootstrap。兩個對照為 P16 的尿量增強 MA-GBT 和凍結的 GCS-only XGBoost，並保留既有 AP、AUROC、Brier 和 0.5 門檻敏感度。

補值依賴率沿用 P17 定義，使用原路徑與原補值矩陣量測。比較整體及乳酸、尿量、GCS 三個欄位。完整來源雜湊及配對輸出保存在 Git 排除的實驗目錄。

## 重跑

在 `ML` 執行，沿用 P13 已建好的 CPU 環境、P16 凍結產物和 P17 路徑核對結果。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
$env:PYTHONPATH = Join-Path $maRuntime 'upstream'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$maPython = Join-Path $maRuntime '.venv\Scripts\python.exe'
& $maPython src/train_ma_gbt_alpha10_lactate.py --workers 4
if ($LASTEXITCODE -eq 0) {
    & $maPython src/evaluate_ma_gbt_alpha10_lactate.py
}
```

結果存於 `ML/artifacts/ma_gbt_alpha10_lactate/`。重跑前先保留既有輸出。
