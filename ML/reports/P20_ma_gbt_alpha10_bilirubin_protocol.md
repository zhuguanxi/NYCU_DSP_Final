# P20 MA-GBT 膽紅素缺失增強

沿用 alpha=10、10 棵深度 2 的樹、原 A/B 三折、凍結的 Full／Elastic 欄位及補值編碼。GCS 45%／ICU 類型 75%／尿量 75%／乳酸 25% 使用 P18 的遮蔽種子與位置。只有膽紅素比例由 A/B 選。

## A/B 強度

- 比較膽紅素 25%、75%、100%。以每折訓練資料內原本有膽紅素的 stay 為分母，各強度沿用同一排列的不同前段。整組更新統計量、量測次數及缺失／定義旗標。沿用一筆 stay 一列和原標籤及權重。
- 本輪目標是 Elastic，因此以 Elastic 原三折選強度。未遮蔽 AP 和 GCS 全遮 AP 的降幅各以 P18 Elastic 為參考，容許 0.01。先選通過兩門檻者，其次是只通過 GCS 全遮門檻者，最後使用全候選並記錄。
- 在符合門檻的候選中，依序選平均膽紅素補值依賴率最低、平均 AP 最高、任一欄位補值依賴率最低及遮蔽比例最低者。三個主要指標都對膽紅素全遮與聯合遮蔽取等權平均。聯合遮蔽使用 GCS 25%／ICU 類型 90%／尿量 100%／乳酸 75%／膽紅素 50%，固定三次抽樣。
- Full 共用強度並回報原三折。各用全 A/B 擬合並凍結後，再評估 Set C 和 eICU。全程 CPU。

## 對照

使用 P18 乳酸增強與凍結 GCS-only。Set C 保留原 P9 的 13 種遮蔽、32 次匹配抽樣，以及 P19 的膽紅素全遮和 32 次遮掉一半原觀測。記錄未遮蔽、GCS 全遮、膽紅素全遮和部分遮蔽等 AP，及原 AUROC、Brier、0.5 門檻敏感度。

eICU Demo 使用原整點與原始間隔、原標籤和同一組 1,000 次患者層級配對 bootstrap。沿用 P19 的路徑計數，回報膽紅素、膽紅素作為唯一補值欄位及任一欄位的補值依賴率。模型與配對彙總留在 Git 排除目錄。

## 重跑

在 `ML` 執行，使用既有 P13 CPU 環境、P18 凍結產物和 P19 路徑與遮蔽核對。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
$env:PYTHONPATH = Join-Path $maRuntime 'upstream'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$maPython = Join-Path $maRuntime '.venv\Scripts\python.exe'
& $maPython src/train_ma_gbt_alpha10_bilirubin.py --workers 4
if ($LASTEXITCODE -eq 0) {
    & $maPython src/evaluate_ma_gbt_alpha10_bilirubin.py
}
```

輸出存於 `ML/artifacts/ma_gbt_alpha10_bilirubin/`。重跑前先保留既有輸出。
