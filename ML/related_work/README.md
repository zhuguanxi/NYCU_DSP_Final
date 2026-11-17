# ICU 住院死亡文獻復現

五個作者 GitHub 版本已固定並通過 Windows CPU 的小樣本訓練、驗證和測試。KEDGN 另跑完作者完整切分的一輪。MA-GBT 沿用已經完成的 DSP 同流程對照。這五篇的完整論文排程與重跑分數仍待驗證。

| 分類 | 方法 | 本機進度 |
| --- | --- | --- |
| 有 GitHub，完整作者切分跑了一輪 | [KEDGN](KEDGN/README.md) | 3,198／400／399 筆，第一輪測試 AP 34.95%，AUROC 73.91% |
| 有 GitHub，CPU 流程測試通過 | [WaveGNN](WaveGNN/README.md) | 原尺寸初段估計約 12 小時／輪，小樣本已完成全流程 |
| 有 GitHub，CPU 流程測試通過 | [ATENet](ATENet/README.md) | P12-M、論文參數，修正寫死 CUDA 與輸出路徑 |
| 有 GitHub，CPU 流程測試通過 | [Warpformer](Warpformer/README.md) | Set A 完整前處理已通過，也完成作者切分小樣本測試 |
| 有 GitHub，CPU 流程測試通過 | [ViTST](ViTST/README.md) | 真實圖像、公開預訓練權重、凍結編碼器與驗證集選模型 |
| 有 GitHub，既有完整 DSP 同流程比較 | [MA-GBT](MA-GBT/README.md) | P13 已完成 A/B、Set C 與 eICU Demo |
| 尚未找到作者 GitHub | [DBGL](DBGL/README.md) | 待討論 |

每個小樣本保留作者的 train／validation／test 分組，組內刻意抽成兩類各半。各篇記下測試分數以便確認執行結果。論文精確率召回率指標對陽性比例敏感，這些試跑分數不能拿來排名。KEDGN 一輪的分數使用真實作者測試隊列，但尚缺其餘輪次與種子。

[CPU 重跑入口](SETUP.md) 記錄套件、下載版本、資料來源與執行方式。程式副本、明細、圖像、資料與權重保留在本機 Git 排除目錄。

[作者切分的 XGBoost 對照與完整 CPU 排程](AUTHOR_SPLIT_COMPARISON.md)
