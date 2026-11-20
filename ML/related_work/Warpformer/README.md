# Warpformer

作者版本 377d64db2b146c7acf4197c970a236dca3ba0663，使用 Set A 死亡任務與庫內 train/test 記錄清單。論文報告 AUROC 86.6%、AUPRC 56.7%。 [GitHub](https://github.com/imJiawen/Warpformer) · [原論文表 1](https://arxiv.org/html/2306.09368)

完整 Set A 的原前處理已跑完。第一次完整前處理加一輪小樣本訓練與評估共 385 秒。另驗證較短的流程測試，先從原作者各切分取 16／8／8 筆再前處理，用一個種子、batch 4 跑一輪。該測試約 31 秒，AUROC 62.50%、AP 60.83%。

工作副本將裝置設為 CPU，並去除 Windows 檔名不支援的 URL 查詢尾碼。當地 Set A 可直接重用。作者原程式在分割前以全隊列估計正規化範圍。對齊 DSP 時，需要將正規化限於 A/B 訓練資料。完整論文排程使用原尺寸、五個種子及最多 50 輪。 [完整前處理日誌](../../artifacts/related_work/Warpformer/cpu_smoke_01/train_seed_0.log) · [較短測試日誌](../../artifacts/related_work/Warpformer/cpu_smoke_02/train_seed_0.log)

## 2026 年 11 月 20 日完整排程核對

首種子完整資料已完成 16 輪並進入第 17 輪，保存了 checkpoint，訓練日誌最後停在第 17 輪的 78／80 批。未取得最終測試結果或其餘種子日誌，目前未找到相關重跑程序。中止原因無法由日誌確認。

[全專案研究進度](../../reports/研究進度總覽.md)
