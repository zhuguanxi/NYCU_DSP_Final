# P14 MA-GBT 的 GCS 與 ICU 類型缺失增強

沿用 [P13 原版 MA-GBT](P13_ma_gbt_protocol.md) 的作者程式、缺失懲罰 alpha=1000、10 棵深度 2 的樹、A/B 三折及原凍結的 Full／Elastic 特徵與每折前處理。沿用作者的缺失矩陣及前一輪已核對的 CPU 數值比較。

## A/B 訓練與強度選擇

| 候選 | GCS 遮蔽已觀測值 | ICU 類型遮蔽已觀測值 |
| --- | ---: | ---: |
| 輕 | 25% | 50% |
| 中 | 45% | 75% |
| 強 | 60% | 90% |

- 在每個訓練折中先抽出有觀測值的 stay。兩欄獨立抽樣，同折各強度沿用同一排列的不同前段。GCS 沿用現有整組遮蔽函式，同步更新統計值、次數及定義指標。ICU 類型設為 NaN，套用原 one-hot 編碼並標記該原始類別欄位的缺失。
- 每筆 stay 在每次訓練只有一列，抽中的 stay 使用遮蔽後的特徵。保留原列數、標籤和權重，MA-GBT 懲罰中的平均值分母也沿用實際列數。訓練遮蔽與驗證遮蔽分開抽樣。
- Full 只用 A/B 三折選強度。驗證四個遮蔽情境的 AP 等權平均，涵蓋 GCS 25% 加 ICU 類型 90% 共三次抽樣、GCS 全遮、ICU 類型全遮、兩項全遮。所有候選看到同一組驗證遮蔽。
- 以原版 Full 三折未遮蔽 AP 為參考，先在未遮蔽 AP 下降最多 0.01 的候選中選遮蔽 AP 平均最高者。若沒有候選通過門檻，改用所有候選中遮蔽 AP 最佳者並記錄。遮蔽 AP 同分時以未遮蔽 AP 決定。
- Elastic 共用 Full 選出的強度並回報三折。最後 Full／Elastic 各在全 A/B 擬合一次。完成凍結與雜湊記錄後才開始 Set C 與 eICU 評估。

## 評估與重跑

使用 P9 同一 Set C、原 13 種遮蔽和同一組 32 次匹配抽樣，逐次配對 P13 原版 MA-GBT 與 P11 GCS-only。eICU Demo 沿用整點和原始間隔，AP／AUROC 的差值區間使用相同 1,000 次患者層級配對 bootstrap。兩個資料集沿用 AP、AUROC、Brier 和 0.5 門檻敏感度。原 Set C 遮蔽表沒有 ICU 類型情境，A/B 驗證包含該欄單獨與聯合遮蔽。eICU 保留原本 ICU 類型對齊後的缺失狀態。

在 `ML` 執行，使用 P13 的 CPU 環境、凍結前處理、模型與評估產物。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
$env:PYTHONPATH = Join-Path $maRuntime 'upstream'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$maPython = Join-Path $maRuntime '.venv\Scripts\python.exe'
& $maPython src/train_ma_gbt_augmentation.py --workers 4
if ($LASTEXITCODE -eq 0) {
    & $maPython src/evaluate_ma_gbt_augmentation.py
}
```

模型、完整彙總與逐遮蔽配對指標保存在 Git 排除的 `ML/artifacts/ma_gbt_gcs_icu_augmentation/`。重跑前先保留既有輸出，再使用原輸出路徑。
