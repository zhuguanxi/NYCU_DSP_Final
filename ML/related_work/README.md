# ICU 住院死亡文獻復現

截至 2026 年 11 月 22 日，五個作者版本的小樣本 CPU 流程皆通過。KEDGN 已完成完整切分的五種子、每種子 10 輪排程。MA-GBT 沿用本專題已完成的對照。其餘完整排程缺口見下表。

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

使用者已縮小範圍：R1 取消，其餘論文背景重跑停止，保留既有紀錄。P53 已完成作者 Set A 3,198 筆訓練之 KEDGN 凍結轉移。P54 接著用原 A/B 三折選輪次、全 A+B 訓練五個種子，再評估同一 Set C／eICU Demo；訓練可用 GPU。 [P54 結果](../reports/P54_kedgn_ab_transfer_results.md) · [A+B 重訓規格](KEDGN/AB_RETRAIN_PROTOCOL.md) · [P53 規格](KEDGN/FROZEN_TRANSFER_PROTOCOL.md) · [原完整排程規格（歷史）](COMPLETE_REPRODUCTION_PROTOCOL.md)

## 最終共同隊列補充

P55 已完成兩版 KEDGN 與七組原模型的八種 Set C 遮蔽；P59 已補 MA-GBT 的尿量映射一致性。這些為固定模型敏感度比較，不把不同容量或不同作者切分視為架構因果結論。

[P55 結果](../reports/P55_kedgn_missingness_results.md) · [P59 結果](../reports/P59_ma_alignment_results.md) · [文獻定位與來源](../reports/文獻定位與可比性.md) · [研究主線](../reports/研究結論與故事.md)
