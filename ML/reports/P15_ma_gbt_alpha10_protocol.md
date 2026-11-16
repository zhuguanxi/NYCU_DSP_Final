# P15 MA-GBT 低懲罰對照

比較相同樹設定下的兩種缺失懲罰與兩種訓練條件。原版指直接使用 P12 A/B；增強組遮蔽原有 GCS 的 45% 與 ICU 類型的 75%。

| MA-GBT 缺失懲罰 | 原版 | GCS 45%／ICU 類型 75% |
| --- | --- | --- |
| alpha=10 | 本輪擬合 | 本輪擬合 |
| alpha=1000 | P13 凍結模型 | P14 凍結模型 |

另保留 P11 的 GCS-only XGBoost。alpha=10、兩欄強度、原前處理、10 棵深度 2 的樹及原折分配均沿用事前指定設定。

## A/B

- Full／Elastic 分別在同一 A/B 三折驗證，再用全 A/B 各擬合一次。共 12 次三折擬合及 4 次最終擬合，全部用 CPU。
- 原版保留 AB 觀測值，增強組沿用 P14 的遮蔽程式及種子。相同折的遮蔽位置與 P14 一致。每次擬合每筆 stay 只使用一列，沿用各折或最終模型的補值與 one-hot 編碼。
- 兩個新組別均記錄未遮蔽、固定部分聯合遮蔽、GCS 全遮、ICU 類型全遮和聯合全遮的 AP、AUROC 與缺失依賴率。原版 Full 的 alpha=10 未遮蔽折指標與 P13 既有紀錄逐項核對。強度與懲罰事前已確定，這輪三折用來檢驗差異。

## Set C 與 eICU Demo

新模型凍結後才讀入外部評估資料。Set C 沿用 P9 的 13 種遮蔽、原標籤與固定抽樣，核對遮蔽位置雜湊後與 P14 的逐次結果配對。eICU Demo 沿用原整點／原始間隔資料，以及 1,000 次同樣的患者層級 bootstrap。A/B、Set C 與 eICU 使用的欄位及相應補值均沿用原前處理。

報告 AP、AUROC、Brier、0.5 門檻敏感度，以及四個 MA-GBT 組別在未遮蔽 Set C 與 eICU 的缺失依賴率。主要配對差為 alpha=10 相對同訓練條件的 alpha=1000、alpha=10 增強相對 alpha=10 原版，以及兩組 alpha=10 各與 GCS-only 的差異。包含原 P9 測試的 GCS 全遮與六項匹配遮蔽。另檢查各組是否保留 GCS 數值與 ICU 類型的樹內分裂。

## 重跑

在 `ML` 執行，使用 P13 的 CPU 環境及 P13、P14 的模型和評估產物。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
$env:PYTHONPATH = Join-Path $maRuntime 'upstream'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$maPython = Join-Path $maRuntime '.venv\Scripts\python.exe'
& $maPython src/train_ma_gbt_alpha10.py --workers 4
if ($LASTEXITCODE -eq 0) {
    & $maPython src/evaluate_ma_gbt_alpha10.py
}
```

模型、完整指標及遮蔽配對產物保存在 Git 排除的 `ML/artifacts/ma_gbt_alpha10/`。重跑前先保留上一輪輸出。
