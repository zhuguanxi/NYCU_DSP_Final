# KEDGN

作者版本 c82bc004cd1d1305fc58ca7150369d78682fccad，使用 physionet 的住院死亡設定及作者第 5 折。論文 ChatGPT 變體在該欄報告 AUROC 87.9%、AUPRC 57.1%。 [GitHub](https://github.com/qianlima-lab/KEDGN) · [論文表 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c04aea54c2a60a632a47bd451cd2849-Paper-Conference.pdf)

用官方 Set A 經作者腳本重建 3,997 筆序列，按作者已提交的結果陣列順序核對每筆 ID 和死亡標籤。採已附的 BERT 與 GPT 來源變數表示。CPU 工作副本修改寫死的 CUDA 呼叫、模型讀取位置與作者前處理成功後的非零結束碼。

首輪流程測試為 16／8／8 筆、batch 8、一輪，測試 AUROC 56.25%、AP 56.67%。作者原排程每輪 batch 256、10 輪和五個種子。 [本機小樣本測試日誌](../../artifacts/related_work/KEDGN/cpu_smoke_01/train.log)

另用作者第 5 折的完整 3,198／400／399 筆、原 batch 256、種子 0 跑完一輪。測試有 51 次住院死亡，AUROC 為 73.91%、AP 為 34.95%。初次單輪訓練與驗證約 3.09 分鐘。這是早期進度，後續完整五種子結果如下。可用 CPU 入口的 first_epoch 重現這一輪。 [完整切分一輪日誌](../../artifacts/related_work/KEDGN/cpu_native_epoch1/train.log)

## 完整五種子 CPU 結果

2026 年 11 月 20 日核對完整日誌，共有 50 輪驗證、5 次最終測試和五種子平均輸出。使用作者第 5 折的完整資料、batch 256、每種子 10 輪、驗證集選 checkpoint。

| 種子 | AP（%） | AUROC（%） | Accuracy（%） |
| --- | ---: | ---: | ---: |
| 0 | 54.77 | 87.51 | 81.20 |
| 1 | 57.85 | 89.27 | 79.20 |
| 2 | 56.84 | 88.13 | 77.44 |
| 3 | 59.26 | 88.03 | 81.20 |
| 4 | 55.14 | 86.85 | 78.95 |
| 五種子平均 ± SD | **56.8 ± 1.7** | **88.0 ± 0.8** | **79.6 ± 1.4** |

論文報值 AP 57.1%、AUROC 87.9%，與本機完整平均差約 −0.3／+0.1 個百分點。五段訓練日誌計時合計 322.977 分鐘，另計前處理與啟動。這次已完成作者 Physionet 死亡設定的五種子排程。

同切分固定 XGBoost 的 AP 46.588%、AUROC 85.063%，本機 KEDGN 五種子平均較高。特徵表示和種子數不同，尚未計算配對區間。原始日誌以兩位小數記錄單次分數，彙總採作者輸出的精度。

來源日誌 SHA-256 為 `947ff1d4cdc5a4d97767d39728e94f5a09a551fe6e0f084da0bf30bb733b54fc`。 [本機完整日誌](../../artifacts/related_work/KEDGN/cpu_paper_01/train.log) · [全專案進度](../../reports/研究進度總覽.md)

依 [CPU 設定](../SETUP.md) 使用文獻環境，在 ML 目錄執行。既有 tag 會阻止覆寫。

```powershell
$py = '.\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe'
& $py related_work/run_cpu.py KEDGN --profile paper_schedule --tag kedgn_cpu_rerun
```

## P53：凍結模型的 Set C 與 eICU Demo

2026 年 11 月 21 日已使用上述五個驗證 AP 最佳權重，先逐種子重現作者測試 AP／AUROC，再評估本組兩個資料集。模型擬合 0 次，全程 CPU；主要分數事先固定為五種子機率等權平均。

| 測試隊列 | 住院／死亡 | KEDGN AP | KEDGN AUROC |
| --- | ---: | ---: | ---: |
| P12 Set C | 4,000／585 | 0.54977 | 0.86636 |
| eICU Demo 整點 | 782／91 | 0.44998 | 0.81177 |
| eICU Demo 原始間隔 | 782／91 | 0.44596 | 0.81209 |

這些是凍結模型的測試結果。前文 56.8%／88.0% 是作者 Set A 內 399 筆測試的五種子指標平均，不能當成本組 Set C 成績。完整種子、七組原模型、配對差值區間及前處理限制見 [P53 結果](../../reports/P53_kedgn_frozen_transfer_results.md) 和 [凍結評估規格](FROZEN_TRANSFER_PROTOCOL.md)。

P53 KEDGN 訓練為作者 Set A 的 3,198 筆，本組原模型訓練為 A+B 8,000 筆。當時依凍結模型要求未重訓；後續 A+B 比較見 P54。R1 已取消。

## P54：A+B 三折開發與五種子重訓

依使用者後續指示，使用原 A/B 8,000 筆、1,122 次死亡和原三折，固定作者參數，只以三折合併折外 AP 從 1–10 輪中選共同輪次。選出 **9 輪**，A/B pooled OOF AP **0.52450**、AUROC **0.85330**。之後全 A+B 以 seed 0–4 各訓練 9 輪，凍結後才做同病例的 Set C 與 eICU Demo 推論。

使用 RTX 4060 Laptop GPU 與獨立 Torch 2.1.2+cu121 環境。保留作者架構、平衡採樣及有效 batch 256，將逐觀測節點的索引計算改成等價遮罩批次運算；與原作者 CPU 輸出及有效參數梯度核對通過，初始最大差約 8.94×10⁻⁸。資料、折外預測、checkpoint、權重及逐筆測試預測只留本機。

[P54 完整結果、各種子及配對區間](../../reports/P54_kedgn_ab_transfer_results.md) · [A+B 規格與重跑方式](AB_RETRAIN_PROTOCOL.md)。同訓練病例數下仍有特徵表示、前處理、五種子平均和選模規則的差異，限制保留於報告。

| 同病例測試 | A+B KEDGN AP | Set A KEDGN AP | A+B KEDGN AUROC |
| --- | ---: | ---: | ---: |
| P12 Set C | 0.57502 | 0.54977 | 0.87790 |
| eICU Demo 整點 | 0.46375 | 0.44998 | 0.81484 |
| eICU Demo 原始間隔 | 0.44640 | 0.44596 | 0.81308 |
