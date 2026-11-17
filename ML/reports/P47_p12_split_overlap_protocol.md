# P47 P12 拆分與觀測重複稽核

使用原 P12 A、B、C 共 12,000 次 ICU stay 的封存來源。逐筆核對原資料準備及 Set C 評估時記錄的內容雜湊。核對原特徵檔、標籤及分組。CPU 稽核不涉及訓練或調整模型。

- 對 A–B、A–C、B–C 三組配對檢查不含 RecordID 列且忽略順序的完整紀錄、標準化且非空的動態事件集合、以及全部 435 個凍結特徵。分開彙總跨組相同配對數、涉及的 stay 數及死亡標籤是否衝突。
- 另檢查動態觀測的高重疊。事件以原 48 小時窗口每筆有效的相對分鐘、變數、數值為鍵。同一事件只計一次。兩邊各需至少 40 個事件、3 種變數及 12 個觀測分鐘。找 Jaccard 至少 0.90 的跨組配對。以全域頻率排序與必要的重疊下限篩候選，再計算確切重疊。
- 穩定的病例指紋可查找複製或近乎相同的輸入。P12 提供 ICU stay 的 RecordID，跨住院患者身分仍不可知。分割結果依可得資料解讀。 [PhysioNet P12 資料與識別鍵定義](https://physionet.org/content/challenge-2012/1.0.0/#general-descriptors)

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md)，執行 `& $py src/audit_p12_split_overlap.py`。只保存彙總、條件及來源雜湊到 Git 排除的 `artifacts/p47_p12_split_overlap/`。
