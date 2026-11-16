# P12 尿量-only 缺失增強規格

沿用 [P10](P10_augmentation_protocol.md) 與 [P11](P11_gcs_only_protocol.md) 的 P12 A/B 三折、原 Full／Elastic 欄位、已擬合前處理與 XGBoost 超參數。這輪只遮蔽尿量。所有訓練與選擇只用 A/B，全程 CPU 2 執行緒。

## A/B 選擇

- 比較 75% 與 100% 兩種強度，沿用 P10 的尿量增強候選。比例的分母為訓練折已有尿量觀測的 stay。選中的 stay 各有原列及一筆尿量遮蔽副本，各給 0.5 權重，其餘原列維持權重 1。只改動尿量的摘要、次數和缺失旗標。
- 沿用原種子。驗證包含未遮蔽、三次尿量 75% 遮蔽，以及一次尿量全遮。先等權平均兩種遮蔽情境 AP，再取三折平均作為缺失分數。
- 由 Full 選強度。在未遮蔽平均 AP 下降至多 0.0050 的方案中，選缺失分數較高者。如果兩案皆超過門檻，取未遮蔽 AP 較高者並記錄門檻狀態。完全同分取較低強度。
- Elastic 共用選定強度，另跑三折診斷。最後在全 A/B 上各擬合一次並凍結。預計 11 次新擬合。

## 凍結後評估

- Set C 沿用 [P9](P9_missingness_results.md) 的同一批 4,000 筆、未遮蔽與 13 種遮蔽。隨機情境沿用原種子及 32 次相同遮蔽位置。
- eICU Demo 沿用 [P8](P8_eicu_demo_results.md) 的同一批 782 次 stay、每小時與原生取樣、院內死亡標籤、病人叢集 bootstrap 及 0.5 閾值。
- 四組對照為原凍結、[GCS-only](P11_gcs_only_results.md)、[GCS＋尿量](P10_augmentation_results.md) 和尿量-only，分別呈現 Full／Elastic。主要比較未遮蔽 Set C 與 eICU 每小時取樣的 AP。另呈現 AUROC、Brier、固定閾值敏感度、尿量全遮及其他原遮蔽條件。
- Set C 與 eICU 皆在模型凍結後使用。兩隊列已曾評估，結果為固定條件的回溯比較。資料、模型及本機輸出留在 Git 排除目錄。
