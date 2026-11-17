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
| ViTST | Raindrop 五折平均 | 51.10／85.10 | +4.71／+1.69 |
| WaveGNN | Raindrop 第一折 | 49.40／83.90 | +2.26／+3.13 |
| KEDGN | Physionet 第 5 折 | 57.10／87.90 | −10.51／−2.84 |
| Warpformer | Set A | 56.70／86.60 | −1.23／−0.95 |

已出版分數來自各論文的平均，XGBoost 在每個切分只訓練一個固定種子。AP 同用平均精確率公式時可比較數值，AUPRC 的積分方式需依作者實作判讀。本表不做優越性推論。 [ATENet](ATENet/README.md) · [ViTST](ViTST/README.md) · [WaveGNN](WaveGNN/README.md) · [KEDGN](KEDGN/README.md) · [Warpformer](Warpformer/README.md)

## 本機重跑

在 `ML` 目錄使用原 `.venv`，備妥 P7 的 A/B 與 C 特徵、標籤、Full 前處理，以及各作者的資料和切分。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' related_work/evaluate_matched_xgboost.py
```

彙總與逐筆分數在 Git 排除的 `artifacts/related_work/matched_xgboost/`。重跑前另存既有輸出。
