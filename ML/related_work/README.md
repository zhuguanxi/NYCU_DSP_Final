# ICU 住院死亡文獻復現

截至 2026 年 11 月 20 日，五個作者版本的小樣本 CPU 流程皆通過。KEDGN 已完成完整切分的五種子、每種子 10 輪排程。MA-GBT 沿用本專題已完成的對照。其餘完整排程缺口見下表。

| 分類 | 方法 | 本機進度 |
| --- | --- | --- |
| 有 GitHub，完整作者五種子排程完成 | [KEDGN](KEDGN/README.md) | AP 56.8% ± 1.7%、AUROC 88.0% ± 0.8%，接近論文 57.1%／87.9% |
| 有 GitHub，CPU 流程測試通過 | [WaveGNN](WaveGNN/README.md) | 原尺寸初段估計約 12 小時／輪，小樣本已完成全流程 |
| 有 GitHub，CPU 流程測試通過 | [ATENet](ATENet/README.md) | P12-M、論文參數，修正寫死 CUDA 與輸出路徑 |
| 有 GitHub，完整排程未完成 | [Warpformer](Warpformer/README.md) | 首種子完成 16 輪，第 17 輪中斷，缺少最終測試與其餘種子 |
| 有 GitHub，完整排程未完成 | [ViTST](ViTST/README.md) | 第一折 11,988 張影像完成，完整微調止於 6／340 更新步，缺少測試 |
| 有 GitHub，既有完整 DSP 同流程比較 | [MA-GBT](MA-GBT/README.md) | P13 已完成 A/B、Set C 與 eICU Demo |
| 尚未找到作者 GitHub | [DBGL](DBGL/README.md) | 待討論 |

每個小樣本保留作者的 train／validation／test 分組，組內刻意抽成兩類各半。各篇記下測試分數以便確認執行結果。論文精確率召回率指標對陽性比例敏感，這些試跑分數不能拿來排名。KEDGN 的完整成績使用真實作者測試隊列，詳見五種子報告。

[CPU 重跑入口](SETUP.md) 記錄套件、下載版本、資料來源與執行方式。程式副本、明細、圖像、資料與權重保留在本機 Git 排除目錄。

[作者切分的 XGBoost 對照與完整 CPU 排程](AUTHOR_SPLIT_COMPARISON.md)

[全專案研究進度與階段索引](../reports/研究進度總覽.md)

## 2026 年 11 月 21 日後續完整重跑

已準備 ATENet 五折、Warpformer 五種子、WaveGNN 三次重複和 ViTST 五折的作者參數與可恢復排程。KEDGN 沿用已完成結果。背景排程先執行 ATENet，完成後依序執行其他模型。本組 Set C／eICU 的共同序列輸入已完成病例、欄位與來源核對，模型專用張量轉換、A/B 訓練和共同測試尚待完成。 [完整參數、進度與共同測試規格](COMPLETE_REPRODUCTION_PROTOCOL.md)
