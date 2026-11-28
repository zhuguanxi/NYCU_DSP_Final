# MA GBT

<!-- P63_START -->
## P63 報告共同標準

**2026-11-28 報告統一口徑：每個種子分別評估，再列五次指標算術平均 ± 樣本標準差（ddof=1）。** AP、AUROC、Brier 與固定0.5敏感度全部遵循同一規則；不平均不同種子模型的機率、不挑最好種子。

Full 的無增強／GCS60，各自無懲罰、MA-AP、MA-rho95六程序均有五種子。正式六模型主表採 MA-AP Full＋GCS60，alpha=0.001沿原A/B選擇固定。Elastic與其他alpha仍保留歷史範圍。

Full／Elastic、GCS-only、R1 及 MA 六程序的五次逐筆預測相同，SD=0 如實保留；這不代表五份獨立測試。Set C／Demo 已反覆查看，外部改善仍未獨立確認。

[四指標與rho](../../reports/P63_report_five_seed_results.md) · [統一口徑](../../reports/P63_reporting_standard.md)。
<!-- P63_END -->

<!-- P62_START -->
## P62：300 棵／深度 3 的補強

全部新模型固定 300 棵、深度 3、學習率 0.05；同作者程式的無懲罰 GBT／MA-GBT 使用相同前處理、三折與 GCS60 加權副本。兩種 MA 選模口徑都只用 A/B，全部候選公開。 完成 96 個唯一 CV 工作、12 次最終擬合，並在凍結後完成 Set C 與兩種 Demo 取樣。保留 MA-AP 與 MA-rho95 兩個 A/B 選模口徑及其全部結果。

[事前規格](../../reports/P62_ma_gbt_matched_protocol.md) · [結果、配對區間與重跑](../../reports/P62_ma_gbt_matched_results.md) · [全部彙總](../../reports/P62_ma_gbt_matched_summary.json)。CPU 平行執行版本已核對候選分數、樹與機率逐值相同。GPU 相容性探測未通過，本輪沒有 GPU 訓練模型。

下方 P13–P20／P59 是舊 10 棵、深度 2 的歷史配置；沒有覆寫。相同名義容量仍不代表作者 GBT 與 XGBoost 為相同演算法；機制判斷優先看同作者無懲罰消融。這輪固定 seed42，沒有五種子或 P61 時間區段重跑。測試已反覆查看，Demo 亦非獨立完整版驗證。
<!-- P62_END -->

固定作者提交 fbbdd0ca2c9d0d8072f49d82ad35f4391098863f。既有 P13 已以 CPU 完成 P12 A/B 原三折、Set C 壓力測試及 eICU Demo，另完成 P14 至 P20 的增強和補值依賴檢查。加速入口事前逐值核對作者原版的切分、節點和預測。

既有 P13 為 DSP 同資料比較，Set C 未遮蔽 AP 0.4464、GCS 全遮 AP 0.2465。這次沿用已核對的 P13 產物。模型與明細在 Git 排除的 ML/artifacts/ma_gbt_p12/。 [固定作者版本](https://github.com/Healthy-AI/malearn/tree/fbbdd0ca2c9d0d8072f49d82ad35f4391098863f) · [本組規格](../../reports/P13_ma_gbt_protocol.md) · [既有結果](../../reports/P13_ma_gbt_results.md)
