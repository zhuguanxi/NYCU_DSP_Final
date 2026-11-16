# P32 eICU GCS 來源核對與機械通氣對齊

沿用 P8 的 eICU Demo 隊列及前 48 小時的觀測與登錄截止。只檢查具有明確語意與時間的原始觀測，在凍結模型上檢驗欄位對齊差異。全程 CPU，無重新訓練。

## GCS 來源稽核

核對護理 scores 類別的 eyes、verbal、motor 是否能在相同 stay 和相同分鐘形成完整且各分項一致的數字總分。眼、語言、運動分項的整數上限分別為 4、5、6。比對 P8 保留的明確總分，並分開計算無總分時可增加的分鐘數和 stay 數。同一分鐘若有原文總分或無法評估的總分紀錄，保留原來源的紀錄。在本輪預查中完整分項沒有增加 GCS 覆蓋，將用正式程式重現這項核對。[eICU 護理表](https://eicu.mit.edu/eicutables/nursecharting/) · [Glasgow Coma Scale 分項](https://www.glasgowcomascale.org/what-is-gcs/)

## MechVent 敏感性比較

P12 官方定義為二元欄位，A/B 的有效原始觀測全為 1。保持相同編碼，只增加在 eICU 觀察且在時窗內登錄的正向事件，按 stay 與分鐘去重。沒有事件時維持 NaN，不外推通氣期間。採兩種預先確定的訊號。

1. 明確呼吸照護標籤。使用 `O2 Device=Ventilator`、`Oxygen Delivery Method=Mechanical Ventilator`、`Mechanical Ventilator Mode=AC/CMV`、`Ventilator Support Mode=CMV/SIMV`。
2. 同時有正值設定。另加入 `respFlowSettings` 的同分鐘 `Tidal Volume (Set)` 和 `Vent Rate` 均為有限正數的事件。此屬設定紀錄的敏感性方案，是否可視為實際通氣或區分侵入式與非侵入式仍有限制。

兩組皆剔除同分鐘有明確其他供氧裝置或非侵入式通氣模式的衝突。使用 P12 原提取器更新 MechVent 相關欄位。APACHE 摘要、呼吸照護表的未目擊期間和通氣結束時間皆不轉成連續事件。[P12 官方變數](https://physionet.org/content/challenge-2012/1.0.0/) · [eICU 呼吸表](https://eicu.mit.edu/eicutables/respiratorycharting/)

## 固定評估

主比較沿用 P28 的尿量 mL 對齊與整點資料，原始間隔作敏感性比較。先逐值核對四個原凍結模型的 eICU 指標，再對每個模型比較同一設定的 AP、AUROC、Brier 和 0.5 門檻敏感度。凍結 Elastic、GCS-only Elastic、P26、P28 在同一版本使用相同輸入。AP 與 AUROC 差使用原 1,000 次患者配對 bootstrap。

此處模型與映射比較限於已反覆評估的 Demo。P29 的完整版規格維持既有主要對照及對齊。使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 及 P28 凍結產物，在 `ML` 運行 `& $py src/evaluate_eicu_ventilation.py`。個案資料與原始檔保留於 Git 排除目錄。
