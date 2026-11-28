# ICU 住院死亡文獻復現

<!-- P64_START -->
## KEDGN 實作稽核（P64）

原作者切分的五種子 AP 平均 **0.56771**，接近論文 **0.571**；共同 Set C 的 **0.54569** 是固定作者架構／參數的 **A+B 適配版**，不是同一切分的復現。已確認解析修復影響 A/B189筆、Set C90筆，另有病例保留、共同輪次及驗證分數定義差異。原始cell全量重播與現有主分數在微小數值誤差內一致，未發現錯權重或GPU改寫造成大幅評分錯誤。選模與訓練解析對差距的獨立貢獻仍未隔離，不能據此宣稱全面勝過充分調優的KEDGN。

原400列逐種子與80列平均／SD不變；沒有新正式訓練或跨種子集成。

[完整差異／數值核對／重跑](../reports/P64_kedgn_implementation_audit.md) · [稽核證據](../reports/P64_kedgn_implementation_audit.json)。下方歷史敘述若稱整體前處理完全一致，應以本次全量核對的界線為準。
<!-- P64_END -->

<!-- P63_START -->
## 目前報告六模型比較（P63）

**2026-11-28 報告統一口徑：每個種子分別評估，再列五次指標算術平均 ± 樣本標準差（ddof=1）。** AP、AUROC、Brier 與固定0.5敏感度全部遵循同一規則；不平均不同種子模型的機率、不挑最好種子。

KEDGN採作者ChatGPT文字來源／BERT編碼的A+B重訓版，仍使用既有0–4五個種子權重；R1採匯入論文參數baseline，補足42–46五種子。兩者都先逐種子評分，再平均指標。

[共同四指標／限制](../reports/P63_report_five_seed_results.md) · [統一口徑](../reports/P63_reporting_standard.md)。下方作者切分復現及各階段狀態為歷史，不能替代目前共同Set C／Demo主表。
<!-- P63_END -->

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
