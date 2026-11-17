# P45 真實外部增強的分支消融

P44 把 eICU Demo 真實病例加到兩個分支後，未遮蔽 Set C 與 Demo 折外 AP 點估計提高，GCS 全遮 AP 點估計下降。本輪拆開分支，比較只增強有 GCS 的 XGBoost、只增強無 GCS 的 LightGBM、P44 雙分支增強及凍結 P28。沿用既有資料、分流、表格成分和混合權重。分析設計參照已有 P44 結果，屬探索性比較。

- Set C 直接組合原有凍結模型，沿用 13 種遮蔽和原抽樣。逐個抽樣核對 P28、P44 既有指標。兩個混合規則只替換指定分支，原 GCS 是否有觀測決定路由。另以實際組裝的模型核對未遮蔽的分數與切換結果逐值一致。驗證 GCS 全遮時只增強有 GCS 的版本逐值重現 P28。
- eICU Demo 固定 P44 原五折，重現相同兩個分支的折外分數。所有測試患者與原始院鍵區塊都與當折訓練分離。沿用兩種取樣、原與尿量對齊。額外描述 GCS 有觀測／全無觀測兩組的排序和病例數。AP 以原 1,000 次患者配對重抽。
- Set C 未遮蔽、GCS 全遮及六項全遮的 AP 差記錄 2,000 次 stay 配對區間。合併比較四種已固定規則，沒有新的強度搜尋或最終模型擬合。獨立完整版仍依 [P40](P40_eicu_sequence_confirmation_protocol.md)。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md)，執行 `& $py src/audit_eicu_augmentation_branches.py`。只保存彙總、分支來源與雜湊到 Git 排除的 `artifacts/p45_augmentation_branch_ablation/`。
