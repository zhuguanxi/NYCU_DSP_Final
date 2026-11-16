# P16 MA-GBT 尿量缺失增強

沿用 [P15](P15_ma_gbt_alpha10_protocol.md) 的 alpha=10、原 A/B 三折、Full／Elastic 欄位、補值與 one-hot 編碼，及 10 棵深度 2 的樹。GCS 45% 和 ICU 類型 75% 繼續使用 P15 的訓練遮蔽種子與位置。另以 A/B 決定尿量遮蔽比例。

## A/B 強度

- 比較尿量 25%、75%、100%。各比例的分母為該次訓練內原本有尿量觀測的 stay。先沿用 GCS 和 ICU 類型的原抽樣，再抽尿量。各強度的尿量遮蔽使用同一排列的不同前段。
- 尿量沿用 P9 的整組遮蔽，統計值設為 NaN、量測次數設為 0、同步更新缺失及統計量定義旗標。保留一筆 stay 一列、原標籤、原權重及原補值。
- Full 先用三折比較尿量全遮和聯合遮蔽的平均 AP，兩種情境等權重。聯合遮蔽使用 GCS 25%、ICU 類型 90%、尿量 100%，同一驗證折固定三次位置抽樣。另記錄未遮蔽及 GCS 全遮。
- 優先在未遮蔽 AP 較 P15 的 Full 下降最多 0.01 的候選中選平均遮蔽 AP 最高者。若全組超過門檻，依相同指標在全候選中選擇並記錄。AP 同分依遮蔽情境平均尿量依賴率較低、遮蔽比例較低排序。
- Elastic 共用選定強度並回報三折。然後 Full／Elastic 各以全 A/B 擬合並凍結，凍結後才評估 Set C 和 eICU Demo。

## 評估與尿量依賴

沿用原 Set C、P9 的 13 種遮蔽與 32 次匹配位置抽樣。eICU Demo 使用原整點與原始間隔、住院死亡標籤和 1,000 次患者層級配對 bootstrap。四組均沿用 AP、AUROC、Brier 和 0.5 門檻敏感度。

尿量依賴率為每筆預測至少在一棵樹的路徑使用了一項補值尿量統計量的比例。MA-GBT 以作者的路徑依賴函式計算，只保留缺失矩陣的尿量欄。GCS-only 使用 XGBoost 原生結果所走的葉節點回溯路徑，並核對凍結前處理及既有預測。量測次數與缺失旗標屬於已知輸入，與補值統計量分開。Set C 回報未遮蔽及尿量全遮的依賴率，eICU Demo 回報兩種取樣設定。

## 重跑

在 `ML` 執行。使用 P13 已建好的 CPU 環境，以及 P15、P11 的凍結產物。以下各步需先成功完成。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
$env:PYTHONPATH = Join-Path $maRuntime 'upstream'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$maPython = Join-Path $maRuntime '.venv\Scripts\python.exe'
& $maPython src/train_ma_gbt_alpha10_urine.py --workers 4
& .\.venv\Scripts\python.exe src/evaluate_ma_gbt_alpha10_urine.py --gcs-reference
& $maPython src/evaluate_ma_gbt_alpha10_urine.py
```

完整輸出存於 Git 排除的 `ML/artifacts/ma_gbt_alpha10_urine/`。重跑前先保留舊輸出。
