# 作者切分的 CPU 對照

用本專題的 435 欄表格表示做固定 XGBoost 對照。每個模型只在相應作者訓練切分上重擬，包括中位數補值與類別編碼。樹模型採預設超參數與固定種子 42，沒有跑候選搜尋。每個切分檢查訓練、驗證和測試病例互斥。作者提供標籤陣列的切分另外逐筆核對死亡標籤。

| 作者切分 | 訓練／驗證／測試 | 測試死亡 | XGBoost AP | XGBoost AUROC |
| --- | ---: | ---: | ---: | ---: |
| Raindrop P12 第 1 折 | 9,590／1,199／1,199 | 154 | 0.51662 | 0.87034 |
| Raindrop P12 第 2 折 | 9,590／1,199／1,199 | 171 | 0.58295 | 0.88038 |
| Raindrop P12 第 3 折 | 9,590／1,199／1,199 | 167 | 0.55521 | 0.85654 |
| Raindrop P12 第 4 折 | 9,590／1,199／1,199 | 172 | 0.55471 | 0.86414 |
| Raindrop P12 第 5 折 | 9,590／1,199／1,199 | 202 | 0.58091 | 0.86819 |
| Raindrop P12 五折平均 | | | **0.55808** | **0.86792** |
| KEDGN Physionet 第 5 折 | 3,198／400／399 | 51 | **0.46588** | **0.85063** |
| Warpformer Set A | 2,560／640／800 | 123 | **0.55474** | **0.85652** |

這組 XGBoost 是作者切分下重新訓練的固定對照。Raindrop 訓練折含 DSP 的 Set C 病例，這些產物只供文獻比較。正式 DSP 管線沿用 A/B 開發與 Set C 評估。

## 與已出版分數的差距

表內差值為本機 XGBoost 減論文報值，單位為百分點。這些比較對齊了已核對的測試切分，仍受特徵表示、遺漏值處理、訓練方式及種子數影響。

| 論文 | 對應的 XGBoost 切分 | 論文 AUPRC／AUROC（%） | AP／AUROC 差（百分點） |
| --- | --- | ---: | ---: |
| ATENet | Raindrop 五折平均 | 53.31／85.54 | +2.50／+1.25 |
| ViTST Swin–RoBERTa（附錄表 9） | Raindrop 五折平均 | 49.80／85.60 | +6.01／+1.19 |
| WaveGNN | Raindrop 第一折 | 49.40／83.90 | +2.26／+3.13 |
| KEDGN | Physionet 第 5 折 | 57.10／87.90 | −10.51／−2.84 |
| Warpformer | Set A | 56.70／86.60 | −1.23／−0.95 |

已出版分數來自各論文的平均，XGBoost 在每個切分只訓練一個固定種子。AP 同用平均精確率公式時可比較數值，AUPRC 的積分方式需依作者實作判讀。本表不做優越性推論。 [ATENet](ATENet/README.md) · [ViTST](ViTST/README.md) · [WaveGNN](WaveGNN/README.md) · [KEDGN](KEDGN/README.md) · [Warpformer](Warpformer/README.md)

## 完整資料 CPU 排程

| 方法 | 本機量到的進度或成本 |
| --- | --- |
| KEDGN | 完整五種子、各 10 輪排程完成，測試 AP 56.8% ± 1.7%、AUROC 88.0% ± 0.8%。相同測試切分固定 XGBoost 為 AP 46.588%、AUROC 85.063%。 |
| Warpformer | 完整前處理通過，首種子完成 16 輪並進入第 17 輪。日誌停在訓練中，已保存 checkpoint，但未取得最終測試或其餘種子。 |
| ATENet | 論文參數的完整第一折已進入訓練，一輪 64 批。試跑因同時運算的記憶體壓力中止，當時尚未保存首輪 checkpoint。 |
| ViTST | 重用網格後的第一折 11,988 張影像已完成，實測 4 小時 4 分 26 秒。微調記錄到 6／340 更新步，耗時約 2 小時 51 分。缺少結束及測試結果。12 張 PNG 的逐像素加速核對已通過。 |
| WaveGNN | 先前原尺寸 CPU 初段約 17.84 秒／批、2,398 批／輪，估計近 12 小時／輪。已補齊論文的 dropout 0.1，完整三種子排程待跑。 |
| MA-GBT | 沿用既有 P13 的本專題 A/B、Set C 和 eICU Demo 重現。 |
| DBGL | 尚未找到作者公開實作。 |

本機日誌在 Git 排除目錄 `artifacts/related_work/`。KEDGN、Warpformer、ATENet 的工作目錄使用 `cpu_paper_01`。ViTST 原版和加速版分別使用 `cpu_paper_01` 和 `cpu_first_fold_01`。表內單輪、前段和驗證數字通常低於完整排程的證據量，故完整論文差距仍需以最後測試輸出核對。

## 本機重跑

在 `ML` 目錄使用原 `.venv`，備妥 P7 的 A/B 與 C 特徵、標籤、Full 前處理，以及各作者的資料和切分。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' related_work/evaluate_matched_xgboost.py
```

彙總與逐筆分數在 Git 排除的 `artifacts/related_work/matched_xgboost/`。重跑前另存既有輸出。

2026 年 11 月 20 日已核對落盤日誌及存活程序。上述未完成的排程目前未找到執行程序，中止原因在 Warpformer 與 ViTST 日誌中不明。 [完整進度](../reports/研究進度總覽.md)

2026 年 11 月 21 日已更正 ViTST 的變體對應，並建立可恢復作者排程；依後續使用者縮小範圍的指示，其他論文背景重跑已停止、R1 取消。只保留 KEDGN 的 Set C／eICU Demo 凍結評估，結果與訓練量差異見 [P53](../reports/P53_kedgn_frozen_transfer_results.md)。[原參數與共同測試（歷史規格）](COMPLETE_REPRODUCTION_PROTOCOL.md)
