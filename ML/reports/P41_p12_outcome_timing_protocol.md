# P41 P12 時間欄位稽核與 Set C 敏感度分析

PhysioNet 說明 P12 排除了 ICU 停留不足 48 小時的紀錄，並將 Length of stay 定義為 ICU 入住至出院的整天欄位。它涵蓋離開 ICU 後的住院時間。Survival 是從 ICU 入住至登記死亡的天數，−1 表示沒有登記死亡時間。公開結果表有 0、1 和其他特殊值。本輪先核對官方下載的三份結果表雜湊，統計欄位與原住院死亡標籤。另查 Set C 中 Survival 為 0 或 1 的住院死亡病例是否有晚期有效觀測。資料為整天值，精確的入住、死亡和出院時間未知。[PhysioNet 2012 資料定義](https://physionet.org/content/challenge-2012/1.0.0/#outcome-related-descriptors)

- 四組 Set C 敏感度隊列為全體、已知年齡至少 18 歲、成人且兩時間欄都沒有 0–1 天、成人且住院天數至少 3 並且已知存活天數至少 3。最後一組保留 Survival 為 −1 的住院存活者。−1 的住院天數在第三組視為未知。這些篩選是回溯性分析，考慮資料精度和母體差異。
- 沿用原標籤與凍結模型，先在全 Set C 核對未遮蔽、GCS 全遮和六項全遮。比較原 Full、Elastic、GCS-only Full、P26、P28、P33 和 P38。時間結果欄只切敏感度隊列，一律在給分後取分組指標。全程 CPU。
- 各隊列計算未遮蔽的 P33、P38 對凍結 Full 的配對 AP 差，以及 GCS 全遮的 P26 對 GCS-only Full 和 P38 對凍結 Full。以 P7 原 2,000 次 stay 配對 bootstrap 計算 95% 區間。六項全遮報同批病例的 AP 點估計。

在 `ML` 恢復 [P21 的 CPU 環境](P21_predictive_boosting_protocol.md)，執行 `& $py src/audit_p12_outcome_timing.py`。僅輸出彙總及來源雜湊至 Git 排除目錄 `artifacts/p41_p12_outcome_timing/`。
