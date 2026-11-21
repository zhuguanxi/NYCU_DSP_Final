# 完整文獻復現與共同測試規格

依 2026 年 11 月 21 日的工作指示，先完成作者設定的死亡預測復現，再以同一架構在 P12 A/B 開發與重新訓練，最後評估 P12 Set C 及 eICU Demo。所有訓練和推論使用 CPU。資料與權重存於 Git 排除的 artifacts，程式與彙總分階段提交。

## 作者排程

| 作者版本 | 已核對的設定 | 完成條件 |
| --- | --- | --- |
| ATENet | 原 P12-M 切分 5 折，每折 20 輪，seed 0，batch 128，Adam，lr 0.0001，hidden 32，embed-time 128，heads 2，alpha 0.01，beta 0.1，label smoothing 0.1 | 五折各載入驗證 AUROC 最佳模型，輸出五組測試 AP／AUROC |
| KEDGN | 作者 Physionet 第 5 折，seed 0–4，各 10 輪，batch 256，lr 0.001，BERT 768 維、GPT 欄位描述、query 5、node 9、hidden 8、rarity alpha 1 | 已完成 50 輪驗證與 5 次最終測試，沿用已核對的完整日誌 |
| Warpformer | 作者 Set A train／test 名單，驗證為訓練名單後 20%，seed 0–4，各最多 50 輪，patience 5，batch 32，Adam lr 0.001，d_model／inner 64，d_k／d_v 8，heads 1，layers 3，dropout 0，warp 0_0.2_1 | 每種子依驗證 AUROC 早停與選模，輸出五組最終測試 |
| WaveGNN | Raindrop 第一折，3 次重複，各最多 20 輪、patience 10，batch 4、梯度累積 32、AdamW lr 0.0001、weight decay 0.00001，dropout 0.1，window 215，attention heads 8 | 三次完整測試與平均，各次分開保存 checkpoint |
| ViTST Swin–RoBERTa | 作者 5 折，seed 1799，4 輪，batch 48／eval 128，lr 0.00002，累積 4、warmup 0.1，少數類別上採樣，384×384 圖像，編碼器皆微調 | 五折全部完成及最終測試。以驗證集選模型，最後才計分測試 |
| MA-GBT | 沿用 P13–P20 已完成的固定作者版本、本組前處理與三折對照 | 在共同測試彙總中重用凍結預測與來源校驗 |

ATENet 參數沿用已核對論文附錄及固定程式版本。Warpformer 沿用作者 run.sh 的 physio 命令。本次保留作者程式的排程和數值處理，各種子為本機重現規格所選。WaveGNN 的作者程式以驗證 loss 做早停，以驗證 AUROC 保存最佳模型，而 scheduler 使用驗證 AP；此處遵循程式。其 config_p12.json 的 dropout 為 0.3，本輪以論文所寫的 0.1 覆寫。 [ATENet 附錄](https://openreview.net/pdf?id=l0kR6m9NDT) · [Warpformer](https://arxiv.org/html/2306.09368) · [WaveGNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC13068137/)

ViTST 的 Vision-Text 入口對應論文附錄表 9 的 Swin–RoBERTa，P12 報值為 AUROC 85.6%、AUPRC 49.8%。先前文件引用的主表 85.1%／51.1% 屬另一組論文報值，後續差距需用對應變體。公開程式把測試資料傳給訓練期間的 eval_dataset，本機已改接驗證資料並保留此差異。 [ViTST 原論文附錄表 9](https://arxiv.org/html/2303.12799#A2.T9)

## 可恢復 CPU 執行

每次執行保存命令、來源提交、工作副本程式雜湊、日誌、開始與結束時間、退出碼及完成收據。ATENet 的注意力沿 query 軸分塊重算以降低記憶體，與作者操作逐值及逐梯度核對，包含全缺失樣本。訓練 batch 和模型參數保留。

ATENet、Warpformer 和 WaveGNN 每輪保存模型、optimizer、scheduler、隨機狀態與選模狀態。ATENet 同時保存每輪打亂的病例索引。重新啟動時從完整輪次恢復。ViTST 使用 Hugging Face Trainer 的完整 checkpoint。作者舊版 Warpformer 的 checkpoint 只有最佳權重，缺少 optimizer 和隨機狀態，本輪另開工作副本重跑。

全套依序執行 ATENet、Warpformer、WaveGNN 和 ViTST，一次一個訓練程序。KEDGN 先核對現成的五種子完整結果。程序退出或最後指標不完整時保存失敗紀錄，最多自動重試三次，再留下需檢查狀態。

在 ML 目錄使用文獻 CPU 環境。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/related_work/cpu_packages'
$py = '.\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe'
& $py related_work/verify_author_cpu.py
& $py related_work/run_author_suite.py
```

執行狀態位於 artifacts/related_work/complete_author_schedules/status.json，各篇工作副本使用 recoverable_paper_20261121。準備階段可使用 --prepare-only 和 --models。舊檔案保留。

## 共同測試

兩個測試資料集沿用本組的 48 小時主要設定。P12 Set C 為 4,000 次住院、585 次死亡，eICU Demo 為 782 次住院、91 次死亡，並保留整點和原始間隔兩種表示。前處理、欄位與死亡定義以本組固定對齊為基礎，再接成各作者所需的序列、mask、時間及靜態輸入。

各作者的原訓練切分可能包含本組 Set C，故共同測試前須在 P12 A/B 重新訓練。沿用原三折決定 checkpoint 輪次，再在全 A/B 擬合最終模型。補值、縮放、圖像座標範圍及任何統計量只由 A/B 訓練部分估計。Set C 和 eICU 不參與參數、輪次或模型選擇。

共同彙總應包含未遮蔽、GCS 全遮與六項全遮的 Set C AP／AUROC／Brier，eICU 兩種取樣的 AP／AUROC／Brier，以及相對凍結 Full／Elastic、GCS-only、P26／P33 的同病例配對差。若模型採用較少動態通道，需記錄實際對齊欄位和缺失處理。

作者排程全部完成後才進行正式的共同測試計分。輸入適配、三折開發和最終模型仍待實作與驗證。DBGL 前次未找到作者程式，需先確認實作來源才能納入完整重現。

## 本輪已完成的輸入準備

共同序列保留未縮放的 36 個動態通道和五個靜態欄位。P12 A/B 保留 8,000 次住院及原三折，Set C 保留 4,000 次，兩種 eICU 取樣均保留 782 次。有效事件分別為 A/B 3,221,861 筆、Set C 1,614,314 筆、eICU 整點 354,898 筆及原始間隔 1,269,223 筆。尿量使用既定 mL 對齊。來源、病例互斥、標籤、36 通道、時間範圍與資料前後雜湊已核對。此階段模型擬合與測試預測均為零。

輸入在 artifacts/related_work/common_sequence_inputs/。主 Python 環境可執行 related_work/prepare_common_sequence_inputs.py，既有輸出會阻止覆寫。模型專用的序列長度、padding、缺失數值慣例及訓練折內統計仍須在各架構接入時核對。此資料包包含原始不規則時間，未替模型決定裁切或重採樣規則。
