# ViTST

作者版本 7774e593754e8ace77e7f1e14097f3d2705c6303，P12 死亡與 Raindrop 第一折。論文表 2 報告 AUROC 85.1%、AUPRC 51.1%。 [GitHub](https://github.com/Leezekun/ViTST) · [原論文](https://arxiv.org/html/2303.12799)

已用作者圖像腳本產生 16 張 384×384 的真實病例圖，分為 8／4／4 筆。圖像尺度使用作者完整訓練折。公開的 Swin 與 RoBERTa 權重皆已按版本下載並核對校驗值。CPU 試跑凍結兩個預訓練編碼器，只更新作者模型可訓練的分類及投影參數。

CPU 訓練、驗證、checkpoint 重載與測試均完成。作者 P12 入口固定最多四輪，試跑提前停止在 2.75 輪，訓練程式計時 293 秒。四筆測試 AUROC 25.00%、AP 50.00%。

處理了 Windows 不允許的星號路徑、寫死的 Linux 專案路徑、過時的整批套件匯入和舊版需連網的指標載入。作者公開訓練入口把測試集接到訓練期間的 eval_dataset。本地改用驗證集選 checkpoint，最後才在測試集評分。完整作者訓練還會調整全部編碼器，計算需求明顯高於本地試跑。 [本機圖像日誌](../../artifacts/related_work/ViTST/cpu_smoke_02/images.log) · [本機訓練日誌](../../artifacts/related_work/ViTST/cpu_smoke_02/train_retry.log)
